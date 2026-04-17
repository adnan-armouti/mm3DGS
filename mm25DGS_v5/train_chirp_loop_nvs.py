"""Chirp-loop NVS trainer (v5 CUDA kernels, no regularizers).

Trains per-point materials + normals on the 16 slow-time chirp loops that
live inside a single cascaded radar frame (.npy shape (16, 16, 12, 256)).
Each chirp loop occupies a slightly different physical pose because the
radar rig is moving at walking pace, so each loop gets its own
pose-interpolated Rasterizer state.

Two modes:
  * upper_bound: train on all 16 loops. Gives the performance ceiling at
    this frame assuming a parameter set exists that fits every loop.
  * held_out:    train on 15 loops, evaluate on the withheld middle loop.
    Directly measures interpolation-only generalization across the
    sub-mm pose gap between consecutive loops.

Pose synthesis
--------------
Per-frame radar pose is taken from the existing aligned cascade configs
in data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned.json.
Frame F=135 itself is alignment-broken on seq_0_frame_135 (z-component
of boresight is an outlier: -0.486 vs neighbours ~+0.03), so we anchor
on frames F-1 and F+1 and linearly interpolate per-loop poses through
F's time window.

Fractional time of chirp loop k within frame F, expressed in the
(F-1, F+1) range:
    alpha_k = 0.5 + k * loop_dt / (2 * frame_period)
With frame_period=100 ms (10 Hz cascade capture) and the user-measured
16-loop span of ~7.87 ms, alpha_k \in [0.5, 0.537] across the 16 loops
(3.7% of the inter-frame gap).

Usage
-----
    python -m mm25DGS_v5.train_chirp_loop_nvs \
        --scene seq_0_frame_135 --frame 135 \
        --mode upper_bound --iters 500

    python -m mm25DGS_v5.train_chirp_loop_nvs \
        --scene seq_0_frame_135 --frame 135 \
        --mode held_out --held_out_loop 8 --iters 500
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmir.data.ra_utils import adc_to_ra_complex
from mmir.data.io_utils import compute_range_res_from_cfg

from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch
from mm25DGS_v5 import cuda as v5cuda
from mm25DGS_v5.train_gaussian import (
    DEVICE,
    PointPrimitives,
    init_visible_weighted,
    cull_gaussians,
    render_gaussians,
    range_profile_to_ra_mag,
    build_polar_to_cart_grid,
    polar_to_cart_torch,
    cart_corr_torch,
    precompute_gt_loss_norm,
    compute_ra_loss_rp,
    get_lr_scale,
    rms_clip_grad,
    ITU_CONCRETE,
    USE_FACTORY_PATTERNS,
)
from mm25DGS_v5.load_pretrained import load_trained_config, load_pattern_data


# ---------------------------------------------------------------------------
# Per-loop pose synthesis
# ---------------------------------------------------------------------------

def _load_pose_from_config(cfg_path, device=DEVICE):
    """Read an aligned cascade JSON and return (tx_pos, rx_pos, tx_bore,
    rx_bore) tensors on `device`. Positions in metres."""
    with open(cfg_path) as f:
        d = json.load(f)
    tx_pos = torch.tensor([e['pos_mm']   for e in d['tx_array']],
                          dtype=torch.float32, device=device) / 1000.0
    rx_pos = torch.tensor([e['pos_mm']   for e in d['rx_array']],
                          dtype=torch.float32, device=device) / 1000.0
    tx_bore = torch.tensor([e['boresight'] for e in d['tx_array']],
                           dtype=torch.float32, device=device)
    rx_bore = torch.tensor([e['boresight'] for e in d['rx_array']],
                           dtype=torch.float32, device=device)
    return tx_pos, rx_pos, tx_bore, rx_bore


def build_per_loop_poses(cfg_A_path, cfg_B_path, n_loops=16,
                         loop_dt_s=7.87e-3 / 16.0,
                         frame_period_s=0.1,
                         device=DEVICE):
    """Build per-loop (tx_pos, rx_pos, tx_bore, rx_bore) by linearly
    interpolating between two aligned cascade configs.

    Time model: cfg_A corresponds to the (physical) middle of frame F-1,
    cfg_B to the middle of frame F+1. The 16 chirp loops live at times
    t_k = t_F + k * loop_dt, where t_F = midpoint(t_A, t_B).
    alpha_k = (t_k - t_A) / (t_B - t_A)
            = 0.5 + k * loop_dt / (2 * frame_period).
    """
    tx_A, rx_A, tb_A, rb_A = _load_pose_from_config(cfg_A_path, device)
    tx_B, rx_B, tb_B, rb_B = _load_pose_from_config(cfg_B_path, device)

    alphas = []
    poses = []
    for k in range(n_loops):
        alpha = 0.5 + k * loop_dt_s / (2.0 * frame_period_s)
        alphas.append(alpha)
        tx_p  = (1 - alpha) * tx_A + alpha * tx_B
        rx_p  = (1 - alpha) * rx_A + alpha * rx_B
        tb_k  = (1 - alpha) * tb_A + alpha * tb_B
        tb_k  = tb_k / tb_k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rb_k  = (1 - alpha) * rb_A + alpha * rb_B
        rb_k  = rb_k / rb_k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        poses.append({
            'alpha': alpha,
            'tx_positions':  tx_p.contiguous(),
            'rx_positions':  rx_p.contiguous(),
            'tx_boresights': tb_k.contiguous(),
            'rx_boresights': rb_k.contiguous(),
        })
    return poses, alphas


def apply_pose(rast, pose):
    with torch.no_grad():
        rast.tx_positions.copy_(pose['tx_positions'])
        rast.rx_positions.copy_(pose['rx_positions'])
        rast.tx_boresights.copy_(pose['tx_boresights'])
        rast.rx_boresights.copy_(pose['rx_boresights'])


# ---------------------------------------------------------------------------
# Per-loop GT tensors
# ---------------------------------------------------------------------------

def load_all_chirp_gt(adc_npy_path, loss_type='mse_raw',
                      device=DEVICE):
    """Build per-loop GT tensors for all 16 chirp loops in a cascaded frame.

    Returns a list of 16 dicts with keys:
        'gt_loss'      : tensor used as GT input to compute_ra_loss_rp
        'gt_cart_norm' : min-max normalized GT cart RA, for cart_corr
        'gt_cart_raw'  : raw GT cart (for sanity plots)
    """
    arr = np.load(adc_npy_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'expected cascaded ADC shape (16, RX, TX, K), got {arr.shape}')

    gt_list = []
    for k in range(16):
        ri = np.stack([arr[k].real, arr[k].imag], axis=-1).astype(np.float32)
        # (RX, TX, K, 2) -> (TX, RX, K, 2) matches the rasterizer's output
        # layout (rp_real.shape == (n_tx, n_rx, K))
        ri = ri.transpose(1, 0, 2, 3)
        gt_adc_ri = torch.from_numpy(ri).to(device)

        gt_loss = precompute_gt_loss_norm(gt_adc_ri, loss_type=loss_type)

        with torch.no_grad():
            ra_c = adc_to_ra_complex(gt_adc_ri)
            ra_mag = torch.abs(ra_c).float()

        gt_list.append({'gt_loss': gt_loss,
                        'gt_ra_polar_mag': ra_mag,
                        'gt_adc_ri': gt_adc_ri})

    return gt_list


def _finalize_gt_cart(gt_list, sample_grid):
    for g in gt_list:
        ra_cart = polar_to_cart_torch(g['gt_ra_polar_mag'], sample_grid)
        mn, mx = ra_cart.min(), ra_cart.max()
        g['gt_cart_norm'] = ((ra_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()
        g['gt_cart_raw'] = ra_cart.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Diagnostic: per-loop cart_corr of the untrained model
# ---------------------------------------------------------------------------

def per_loop_cart_corr(model, rast, vertex_areas, active_mask,
                       poses, gt_list, sample_grid):
    corrs = []
    with torch.no_grad():
        for k, pose in enumerate(poses):
            apply_pose(rast, pose)
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            cc = cart_corr_torch(ra_cart, gt_list[k]['gt_cart_norm']).item()
            corrs.append(cc)
            del rp_real, rp_imag, ra_polar, ra_cart
    return corrs


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_chirp_loop_nvs(scene='seq_0_frame_135',
                         frame_idx=135,
                         held_out_loop=None,
                         num_iters=500,
                         mat_lr=0.01,
                         rot_lr=5e-3,
                         loss_type='mse_raw',
                         target_n=90000,
                         frame_period_s=0.1,
                         loop_dt_s=7.87e-3 / 16.0,
                         use_pass2_alignment=False,
                         verbose=True,
                         output_dir=None):
    """Single-frame chirp-loop NVS training. Returns a result dict."""

    assert v5cuda.is_available(), (
        'v5 CUDA extension not built. '
        'cd mm25DGS_v5/cuda && python setup.py build_ext --inplace')

    mode = 'upper_bound' if held_out_loop is None else 'held_out'
    if verbose:
        print(f'{"="*72}\nchirp-loop NVS  mode={mode}  scene={scene}  frame={frame_idx}'
              f'  iters={num_iters}\n{"="*72}')

    # -------- paths --------------------------------------------------------
    config = load_trained_config(scene)
    data_dir = f'/home/adnan/Desktop/mm3DGS/data/{scene}'
    align_dir = f'{data_dir}/alignment_data/cascade'
    if not os.path.isdir(align_dir):
        # canonical layout: data/alignment_data/<scene>/cascade/
        align_dir = f'/home/adnan/Desktop/mm3DGS/data/alignment_data/{scene}/cascade'
    aligned_suffix = '_aligned_pass2' if use_pass2_alignment else '_aligned'
    cfg_A = f'{align_dir}/cascaded_frame_{frame_idx - 1}{aligned_suffix}.json'
    cfg_B = f'{align_dir}/cascaded_frame_{frame_idx + 1}{aligned_suffix}.json'
    adc_npy = f'{data_dir}/radar/cascaded_frame_{frame_idx}.npy'
    for p in (cfg_A, cfg_B, adc_npy):
        assert os.path.exists(p), f'missing: {p}'
    if verbose:
        print(f'  alignment source: {"pass-2" if use_pass2_alignment else "pass-1"}')

    # -------- per-loop poses ----------------------------------------------
    poses, alphas = build_per_loop_poses(
        cfg_A, cfg_B, n_loops=16,
        loop_dt_s=loop_dt_s, frame_period_s=frame_period_s)

    if verbose:
        tx_shift = (poses[-1]['tx_positions'] - poses[0]['tx_positions']).norm(dim=-1)
        print(f'  anchors: frame {frame_idx - 1} (alpha=0) and frame {frame_idx + 1} (alpha=1)')
        print(f'  alphas:  loop 0 = {alphas[0]:.4f}, loop 15 = {alphas[-1]:.4f}')
        print(f'  tx position shift across 16 loops: mean={tx_shift.mean().item()*1000:.3f} mm, '
              f'max={tx_shift.max().item()*1000:.3f} mm')

    # -------- rasterizer (pose is overridden per-loop) --------------------
    # Use cfg_A to seed radar parameters (slope, sample_rate, etc.); pose
    # is immediately overridden to loop 0's interpolated pose before init.
    rast = Rasterizer(
        config_file=cfg_A,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE)
    if not USE_FACTORY_PATTERNS:
        rast.inject_trained_params(pattern_data=load_pattern_data(scene))
    apply_pose(rast, poses[0])

    # -------- init model --------------------------------------------------
    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    # -------- polar -> cart grid ------------------------------------------
    range_res = compute_range_res_from_cfg(cfg_A)
    n_az, n_range = 127, 256
    sample_grid = build_polar_to_cart_grid(
        n_az, n_range, range_res, grid_res=400, device=DEVICE)

    # -------- load all 16 GT chirp loops ----------------------------------
    gt_list = load_all_chirp_gt(adc_npy, loss_type=loss_type)
    _finalize_gt_cart(gt_list, sample_grid)

    # -------- pre-training per-loop validation ----------------------------
    init_corrs = per_loop_cart_corr(
        model, rast, vertex_areas, active_mask, poses, gt_list, sample_grid)

    if verbose:
        print('\n  [validation] pre-training cart_corr by loop (untrained ITU concrete):')
        for k, c in enumerate(init_corrs):
            marker = '  <-- held-out' if k == held_out_loop else ''
            print(f'    loop {k:2d} alpha={alphas[k]:.4f}: cc={c:.4f}{marker}')
        print(f'    mean={np.mean(init_corrs):.4f}  std={np.std(init_corrs):.4f}  '
              f'min={np.min(init_corrs):.4f}  max={np.max(init_corrs):.4f}')

    # -------- choose train / eval sets ------------------------------------
    all_loops = list(range(16))
    if held_out_loop is None:
        train_loops = all_loops
        eval_loops = all_loops
    else:
        assert 0 <= held_out_loop < 16
        train_loops = [k for k in all_loops if k != held_out_loop]
        eval_loops = [held_out_loop]

    # -------- optimizer ---------------------------------------------------
    param_groups = [
        {'params': [model.raw_materials], 'lr': mat_lr, 'name': 'materials'},
        {'params': [model.rotations],     'lr': rot_lr, 'name': 'rotations'},
    ]
    clip_vals = {'materials': 1.0, 'rotations': 0.5}
    base_lrs = {g['name']: g['lr'] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # -------- training loop -----------------------------------------------
    best_mean_train_corr = -1.0
    best_iter = 0
    best_state = None
    history = []
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        per_loop_corr_train = []

        # Scale loss by 1/n_train_loops so total backward has the same
        # gradient magnitude as a single-loop run; keeps LRs comparable.
        loss_scale = 1.0 / len(train_loops)

        for k in train_loops:
            apply_pose(rast, poses[k])
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)

            loss_k, _ = compute_ra_loss_rp(
                rp_real, rp_imag, gt_list[k]['gt_loss'], loss_type=loss_type)
            (loss_k * loss_scale).backward()

            with torch.no_grad():
                ra_polar = range_profile_to_ra_mag(rp_real.detach(), rp_imag.detach())
                ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                cc = cart_corr_torch(ra_cart, gt_list[k]['gt_cart_norm']).item()
                per_loop_corr_train.append(cc)

            loss_sum += loss_k.item() * loss_scale
            del rp_real, rp_imag, ra_polar, ra_cart, loss_k

        # LR schedule (reuse v5 compressed: warmup=0, decay_start=200)
        lr_scale = get_lr_scale(
            it, total_iters=num_iters,
            warmup_iters=0, warmup_factor=1.0, decay_start=200)
        for g in optimizer.param_groups:
            g['lr'] = base_lrs[g['name']] * lr_scale

        for g in optimizer.param_groups:
            clip = clip_vals.get(g['name'], 1.0)
            for p in g['params']:
                rms_clip_grad(p, clip)

        optimizer.step()

        mean_tc = float(np.mean(per_loop_corr_train))
        history.append({
            'iter': it, 'loss': loss_sum,
            'mean_train_corr': mean_tc,
            'per_loop_train_corr': per_loop_corr_train,
        })

        if mean_tc > best_mean_train_corr:
            best_mean_train_corr = mean_tc
            best_iter = it
            best_state = {k: v.data.clone() for k, v in model.state_dict().items()}

        if verbose and (it % 25 == 0 or it == num_iters - 1):
            elapsed = time.time() - t0
            print(f'  iter {it:4d}: loss={loss_sum:.5e}  '
                  f'mean_train_cc={mean_tc:.4f} '
                  f'(best={best_mean_train_corr:.4f}@{best_iter})  '
                  f'[{elapsed:.1f}s]')

    train_elapsed = time.time() - t0

    # -------- restore best state & final per-loop eval --------------------
    if best_state is not None:
        with torch.no_grad():
            for k, v in best_state.items():
                model.state_dict()[k].copy_(v)

    final_corrs = per_loop_cart_corr(
        model, rast, vertex_areas, active_mask, poses, gt_list, sample_grid)

    train_cc_final = [final_corrs[k] for k in train_loops]
    eval_cc_final  = [final_corrs[k] for k in eval_loops]

    if verbose:
        print(f'\n  [final] per-loop cart_corr at best model state '
              f'(restored from iter {best_iter}):')
        for k, c in enumerate(final_corrs):
            tag = '(train)'
            if held_out_loop is not None and k == held_out_loop:
                tag = '(HELD-OUT)'
            print(f'    loop {k:2d}: init={init_corrs[k]:.4f} -> final={c:.4f}  '
                  f'delta={c - init_corrs[k]:+.4f}  {tag}')
        print(f'\n  train mean cc: {np.mean(train_cc_final):.4f} (init {np.mean([init_corrs[k] for k in train_loops]):.4f})')
        if held_out_loop is not None:
            print(f'  held-out cc : {eval_cc_final[0]:.4f} (init {init_corrs[held_out_loop]:.4f})')
        print(f'  elapsed: {train_elapsed:.1f}s  ({train_elapsed*1000/num_iters:.1f} ms/iter)')

    # -------- save --------------------------------------------------------
    if output_dir is None:
        tag = f'frame{frame_idx}_{mode}'
        if held_out_loop is not None:
            tag = f'frame{frame_idx}_heldout_loop{held_out_loop}'
        if use_pass2_alignment:
            tag = f'{tag}_pass2'
        output_dir = os.path.join(
            PROJECT_ROOT, 'mm25DGS_v5', 'output_chirp_loop', f'{scene}_{tag}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'scene': scene, 'frame_idx': frame_idx,
        'mode': mode, 'held_out_loop': held_out_loop,
        'num_iters': num_iters, 'best_iter': best_iter,
        'best_mean_train_corr': best_mean_train_corr,
        'train_loops': train_loops, 'eval_loops': eval_loops,
        'init_corrs_per_loop': init_corrs,
        'final_corrs_per_loop': final_corrs,
        'final_train_mean': float(np.mean(train_cc_final)),
        'final_eval_mean': float(np.mean(eval_cc_final)),
        'alphas': alphas,
        'frame_period_s': frame_period_s,
        'loop_dt_s': loop_dt_s,
        'mat_lr': mat_lr, 'rot_lr': rot_lr,
        'loss_type': loss_type,
        'elapsed_s': train_elapsed,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    # history can be large; dump only summary stats
    np.savez(os.path.join(output_dir, 'history.npz'),
             iters=np.array([h['iter'] for h in history]),
             loss=np.array([h['loss'] for h in history]),
             mean_train_corr=np.array([h['mean_train_corr'] for h in history]),
             per_loop_train_corr=np.array([h['per_loop_train_corr'] for h in history]))

    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))

    if verbose:
        print(f'\n  results saved to: {output_dir}')

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='seq_0_frame_135')
    ap.add_argument('--frame', type=int, default=135)
    ap.add_argument('--mode', choices=['upper_bound', 'held_out'],
                    default='upper_bound')
    ap.add_argument('--held_out_loop', type=int, default=8,
                    help='Only used when --mode held_out')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--mat_lr', type=float, default=0.01)
    ap.add_argument('--rot_lr', type=float, default=5e-3)
    ap.add_argument('--loss_type', default='mse_raw',
                    choices=['mse', 'pearson', 'mse_raw'])
    ap.add_argument('--target_n', type=int, default=90000)
    ap.add_argument('--frame_period_s', type=float, default=0.1,
                    help='Inter-frame period between cascaded frames (10 Hz -> 0.1 s)')
    ap.add_argument('--loop_dt_s', type=float, default=7.87e-3 / 16.0,
                    help='Per-chirp-loop time delta (7.87 ms / 16 by default)')
    ap.add_argument('--use_pass2_alignment', action='store_true',
                    help='Use cascaded_frame_<F>_aligned_pass2.json (trajectory-aware) '
                         'instead of the pass-1 _aligned.json configs')
    args = ap.parse_args()

    held = None if args.mode == 'upper_bound' else args.held_out_loop
    train_chirp_loop_nvs(
        scene=args.scene, frame_idx=args.frame,
        held_out_loop=held, num_iters=args.iters,
        mat_lr=args.mat_lr, rot_lr=args.rot_lr,
        loss_type=args.loss_type, target_n=args.target_n,
        frame_period_s=args.frame_period_s, loop_dt_s=args.loop_dt_s,
        use_pass2_alignment=args.use_pass2_alignment)
