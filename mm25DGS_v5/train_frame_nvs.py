"""Frame-level NVS trainer (v5 CUDA, pass-2 alignment).

Trains on all 16 chirp loops of N training frames (total 16·N RA maps),
holds out one test frame entirely, and evaluates on a single loop of the
test frame (default loop 0 — "first chirp"). The remaining 15 loops of
the test frame are discarded so the NVS test is a single held-out RA.

Pose synthesis:
  For each (frame, loop) pair we reuse the chirp-loop NVS pose
  interpolation (``build_per_loop_poses`` from train_chirp_loop_nvs):
  anchor on ``F-1`` and ``F+1`` aligned configs, interpolate per-loop
  alpha = 0.5 + k·loop_dt/(2·frame_period). Edge train frames missing a
  neighbour fall back to the frame's own aligned pose for all 16 loops.
  The test frame's loop-0 pose is interpolated between its own neighbours
  (test_frame-1 and test_frame+1), so the test pose is NVS-inferred —
  never taken from test_frame's own aligned config.

Usage (example):
    python -m mm25DGS_v5.train_frame_nvs \
        --scene seq_1_frame_438 \
        --test_frame 438 \
        --train_frames 434,435,436,437,439,440,441,442 \
        --iters 500 \
        --use_pass2_alignment
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

from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch  # noqa: F401
from mm25DGS_v5 import cuda as v5cuda
from mm25DGS_v5.train_gaussian import (
    DEVICE,
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
    USE_FACTORY_PATTERNS,
)
from mm25DGS_v5.load_pretrained import load_trained_config, load_pattern_data

# Reuse chirp-loop helpers (per-loop pose interpolation + pose applier + GT loader)
from mm25DGS_v5.train_chirp_loop_nvs import (
    build_per_loop_poses, apply_pose, _load_pose_from_config,
)


# ---------------------------------------------------------------------------
# Per-scene data loading (train and test frames)
# ---------------------------------------------------------------------------

def _aligned_config_path(scene, frame, use_pass2, data_root):
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    suffix = '_aligned_pass2' if use_pass2 else '_aligned'
    p = os.path.join(align_dir, f'cascaded_frame_{frame}{suffix}.json')
    return p if os.path.isfile(p) else None


def _pass3_config_path(scene, frame, chirp, data_root):
    """Per-chirp Stage 3 config path (produced by
    ``per_chirp_alignment.run_stage3_for_scene``)."""
    return os.path.join(
        data_root, 'alignment_data', scene, 'cascade', 'per_chirp',
        f'cascaded_frame_{frame}_chirp{chirp:02d}_aligned_pass3.json')


def _build_frame_poses(scene, frame, use_pass2, data_root,
                       loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.1,
                       anchor_source='pass2_lerp',
                       device=DEVICE):
    """Return a list of 16 per-loop pose dicts for ``frame``.

    ``anchor_source``:
      * ``'pass2_lerp'`` (default) — LERP between pass-2 F-1 and F+1
        aligned configs at per-chirp α. This is the historical
        behaviour of the trainer.
      * ``'pass3_per_chirp'`` — load each (frame, chirp)'s own
        Stage 3 refined config directly from
        ``data/alignment_data/<scene>/cascade/per_chirp/
          cascaded_frame_<F>_chirp<CC>_aligned_pass3.json``.
        Falls back to ``pass2_lerp`` for any chirp whose Stage 3 file
        is missing.

    For edge train frames (no F-1 or F+1 under pass2_lerp), falls back
    to ``frame``'s own aligned pose for all 16 loops (sub-mm
    chirp-loop motion is negligible vs the frame-level motion we're
    training on).
    """
    # Pass-3 path: load per-chirp configs directly, one per (F, c).
    if anchor_source == 'pass3_per_chirp':
        poses = []
        fallback_used = False
        for c in range(16):
            cfg_p = _pass3_config_path(scene, frame, c, data_root)
            if not os.path.isfile(cfg_p):
                fallback_used = True
                break
            tx, rx, tb, rb = _load_pose_from_config(cfg_p, device=device)
            poses.append({
                'alpha': 0.5 + c * loop_dt_s / (2 * frame_period_s),
                'tx_positions':  tx.contiguous(),
                'rx_positions':  rx.contiguous(),
                'tx_boresights': tb.contiguous(),
                'rx_boresights': rb.contiguous(),
            })
        if not fallback_used:
            return poses, 'pass3_per_chirp'
        # else fall through to pass2_lerp

    # Pass-2 LERP path (default / fallback).
    cfg_A = _aligned_config_path(scene, frame - 1, use_pass2, data_root)
    cfg_B = _aligned_config_path(scene, frame + 1, use_pass2, data_root)
    if cfg_A is not None and cfg_B is not None:
        poses, _ = build_per_loop_poses(
            cfg_A, cfg_B, n_loops=16,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            device=device)
        return poses, 'interp_neighbours'

    cfg_F = _aligned_config_path(scene, frame, use_pass2, data_root)
    assert cfg_F is not None, (
        f'{scene} f={frame}: no aligned config (neither neighbour nor '
        f'self); cannot build poses')
    tx, rx, tb, rb = _load_pose_from_config(cfg_F, device=device)
    pose = {
        'alpha': 0.5,
        'tx_positions':  tx.contiguous(),
        'rx_positions':  rx.contiguous(),
        'tx_boresights': tb.contiguous(),
        'rx_boresights': rb.contiguous(),
    }
    return [pose] * 16, 'frame_own_pose'


def _build_per_loop_gt(adc_npy_path, loop_idx, loss_type, device):
    """Load a single (frame, loop) GT as a dict with 'gt_loss' +
    'gt_ra_polar' (magnitude). Caller converts polar→cart later once the
    sample_grid is built.
    """
    arr = np.load(adc_npy_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'expected cascaded ADC shape (16,RX,TX,K); got {arr.shape}')
    ri = np.stack([arr[loop_idx].real, arr[loop_idx].imag],
                  axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)  # (TX, RX, K, 2)
    gt_adc_ri = torch.from_numpy(ri).to(device)
    gt_loss = precompute_gt_loss_norm(gt_adc_ri, loss_type=loss_type)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc_ri)
        ra_mag = torch.abs(ra_c).float()
    return {'gt_loss': gt_loss, 'gt_ra_polar': ra_mag}


def build_frame_level_dataset(scene, train_frames, test_frame,
                               held_out_loop, use_pass2,
                               train_loops=None,
                               loss_type='mse_raw',
                               loop_dt_s=7.87e-3 / 16.0,
                               frame_period_s=0.1,
                               data_root='/home/adnan/Desktop/mm3DGS/data',
                               device=DEVICE,
                               anchor_source='pass2_lerp',
                               verbose=True):
    """Return (train_samples, test_sample, diagnostics).

    train_samples is a list of |train_loops|·|train_frames| dicts with keys:
        'frame_idx', 'loop_idx', 'pose', 'gt_loss', 'gt_ra_polar'
    test_sample is a single dict with the same keys (+ nothing under
    'gt_loss'; test uses cart_corr only).

    ``train_loops`` defaults to all 16 chirp loops. Pass e.g. ``[0]`` to
    train only on the first chirp loop of each train frame (the
    "first-chirp" NVS variant).  Note: "single-chip" in this codebase
    refers specifically to the TI IWR1443 3×4 single-chip radar
    hardware, which is a different sensor from the TI MMWCAS 12×16
    cascaded radar this trainer uses; do NOT confuse the two.
    """
    if train_loops is None:
        train_loops = list(range(16))
    train_loops = list(train_loops)

    scene_dir = os.path.join(data_root, scene)
    radar_dir = os.path.join(scene_dir, 'radar')

    train_samples = []
    edges = []  # frames that fell back to frame-own pose
    for f in train_frames:
        poses, mode = _build_frame_poses(
            scene, f, use_pass2=use_pass2, data_root=data_root,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            anchor_source=anchor_source, device=device)
        if mode != 'interp_neighbours':
            edges.append((f, mode))
        adc_npy = os.path.join(radar_dir, f'cascaded_frame_{f}.npy')
        for k in train_loops:
            gt = _build_per_loop_gt(adc_npy, k, loss_type, device)
            train_samples.append({
                'frame_idx': int(f), 'loop_idx': int(k),
                'pose': poses[k],
                **gt,
            })

    # Test: one (frame, loop); pose interpolated from test_frame's neighbours
    # (or loaded directly from per-chirp Stage 3 config under anchor_source='pass3_per_chirp').
    test_poses, test_mode = _build_frame_poses(
        scene, test_frame, use_pass2=use_pass2, data_root=data_root,
        loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
        anchor_source=anchor_source, device=device)
    test_adc = os.path.join(radar_dir, f'cascaded_frame_{test_frame}.npy')
    test_gt = _build_per_loop_gt(test_adc, held_out_loop, loss_type, device)
    test_sample = {
        'frame_idx': int(test_frame), 'loop_idx': int(held_out_loop),
        'pose': test_poses[held_out_loop],
        **test_gt,
    }

    diag = {
        'n_train_samples': len(train_samples),
        'edge_frames': edges,
        'test_pose_mode': test_mode,
    }
    if verbose:
        print(f'  [data] train_samples: {len(train_samples)} '
              f'({len(train_frames)} frames × 16 loops)')
        if edges:
            print(f'  [data] edge frames (fell back to own pose): {edges}')
        print(f'  [data] test (frame={test_frame} loop={held_out_loop})  '
              f'pose_mode={test_mode}')
    return train_samples, test_sample, diag


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_frame_nvs(scene,
                    train_frames,
                    test_frame,
                    held_out_loop=0,
                    train_loops=None,
                    num_iters=500,
                    mat_lr=0.01,
                    rot_lr=5e-3,
                    loss_type='mse_raw',
                    target_n=90000,
                    frame_period_s=0.1,
                    loop_dt_s=7.87e-3 / 16.0,
                    use_pass2_alignment=True,
                    anchor_source='pass2_lerp',
                    data_root='/home/adnan/Desktop/mm3DGS/data',
                    verbose=True,
                    output_dir=None,
                    reg_l2_drift_lambda=0.0,
                    reg_active_top_frac=None,
                    reg_warm_iters=50,
                    reg_fisher_rot_lambda=0.0,
                    reg_fisher_rot_target='init',
                    reg_fisher_rot_ema_alpha=0.95,
                    reg_fisher_rot_threshold_deg=0.0,
                    seed_frame=None):
    assert v5cuda.is_available(), (
        'v5 CUDA extension not built. '
        'cd mm25DGS_v5/cuda && python setup.py build_ext --inplace')

    if train_loops is None:
        train_loops = list(range(16))
    train_loops = list(train_loops)
    test_in_train = int(test_frame) in [int(f) for f in train_frames]

    if verbose:
        print('=' * 72)
        print(f'frame-level NVS  scene={scene}')
        print(f'  train frames ({len(train_frames)}): {train_frames}')
        print(f'  train loops  ({len(train_loops)}): {train_loops}')
        print(f'  test  frame  = {test_frame}  (test loop={held_out_loop})'
              + ('   [test frame ALSO in train_frames → upper-bound run]'
                 if test_in_train else ''))
        print(f'  alignment    = {"pass-2" if use_pass2_alignment else "pass-1"}')
        print(f'  iters        = {num_iters}   loss_type={loss_type}')
        print('=' * 72)

    # ── paths ──
    config = load_trained_config(scene)

    # The seed frame's pose drives FOV culling + RX visibility before FPS
    # subsampling, so it determines *which* 90 000-point subset of pcl.npy
    # this run trains on. Default = `test_frame` so HO and UB variants share
    # an identical position grid (UB's middle-of-train-frames index already
    # lands on test_frame; HO's previously landed on `test_frame+1`,
    # producing two different ~90k subsets — see md/frame_nvs_analysis/
    # findings.md §0). Pass `seed_frame=<other>` to override (e.g. to
    # reproduce the legacy HO grid use `seed_frame=test_frame+1`).
    if seed_frame is None:
        seed_frame = int(test_frame)
    suffix = '_aligned_pass2' if use_pass2_alignment else '_aligned'
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    seed_cfg = os.path.join(align_dir,
                            f'cascaded_frame_{seed_frame}{suffix}.json')
    if verbose:
        print(f'  seed frame   = {seed_frame}  '
              f'(drives FOV+visibility before FPS → position grid)')
    assert os.path.exists(seed_cfg), f'missing: {seed_cfg}'

    # ── Rasterizer + model ──
    rast = Rasterizer(
        config_file=seed_cfg,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE)
    if not USE_FACTORY_PATTERNS:
        rast.inject_trained_params(pattern_data=load_pattern_data(scene))

    # Init model at seed pose — FOV will be recomputed per-sample via
    # cull_gaussians during render_gaussians. We need a Mitsuba scene for
    # the init-time visibility test.
    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    # FOV mask for the seed pose — reused as a conservative initial mask.
    # Per-sample rendering re-masks via cull_gaussians if needed.
    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    # ── polar → cart grid ──
    range_res = compute_range_res_from_cfg(seed_cfg)
    sample_grid = build_polar_to_cart_grid(127, 256, range_res, 400, DEVICE)

    # ── Load train + test data ──
    train_samples, test_sample, diag = build_frame_level_dataset(
        scene=scene, train_frames=train_frames, test_frame=test_frame,
        held_out_loop=held_out_loop, use_pass2=use_pass2_alignment,
        train_loops=train_loops,
        loss_type=loss_type, loop_dt_s=loop_dt_s,
        frame_period_s=frame_period_s, data_root=data_root,
        anchor_source=anchor_source,
        device=DEVICE, verbose=verbose,
    )

    # Finalize GT cart norms for cart_corr evaluation
    def _finalize_cart(sample):
        with torch.no_grad():
            ra_cart = polar_to_cart_torch(sample['gt_ra_polar'], sample_grid)
            mn, mx = ra_cart.min(), ra_cart.max()
            sample['gt_cart_norm'] = (
                (ra_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()
    for s in train_samples:
        _finalize_cart(s)
    _finalize_cart(test_sample)

    # ── Initial (pre-training) test cart_corr ──
    def _render_and_cart_corr(sample):
        apply_pose(rast, sample['pose'])
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            cc = cart_corr_torch(ra_cart, sample['gt_cart_norm']).item()
        return float(cc)

    init_test_cc = _render_and_cart_corr(test_sample)
    init_train_ccs = [_render_and_cart_corr(s) for s in train_samples]
    if verbose:
        print(f'\n  [init] test  cc = {init_test_cc:.4f}')
        print(f'  [init] train mean cc = {np.mean(init_train_ccs):.4f} '
              f'(std {np.std(init_train_ccs):.4f}) over {len(init_train_ccs)} samples')

    # ── Optimizer ──
    param_groups = [
        {'params': [model.raw_materials], 'lr': mat_lr, 'name': 'materials'},
        {'params': [model.rotations],     'lr': rot_lr, 'name': 'rotations'},
    ]
    clip_vals = {'materials': 1.0, 'rotations': 0.5}
    base_lrs = {g['name']: g['lr'] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    loss_scale = 1.0 / len(train_samples)

    # ── Regularisation setup ──
    # Snapshot init raw_materials (uniform ITU_CONCRETE across N at init time).
    init_raw = model.raw_materials.detach().clone()
    # Snapshot init surface normals (from the quaternion init computed from
    # pcl.npy LiDAR normals). Needed for S2's rotation-drift penalty.
    init_normals = model.get_normals().detach().clone()  # (N, 3)
    active_pts_mask = None
    # S2: per-point Fisher weights on rotations, snapshot at warmup using
    # Adam's exp_avg_sq. Stays None until reg_warm_iters.
    fisher_rot = None
    # S2 Option 2: EMA of normals during training (rolling target).
    normal_ema = None
    reg_enabled = (reg_l2_drift_lambda > 0.0) or (reg_active_top_frac is not None) \
        or (reg_fisher_rot_lambda > 0.0)
    reg_rot_thresh_rad = float(reg_fisher_rot_threshold_deg) * np.pi / 180.0
    if verbose and reg_enabled:
        print(f'  [reg] l2_drift_lambda={reg_l2_drift_lambda}  '
              f'active_top_frac={reg_active_top_frac}  '
              f'fisher_rot_lambda={reg_fisher_rot_lambda}  '
              f'warm_iters={reg_warm_iters}')
        if reg_fisher_rot_lambda > 0.0:
            print(f'  [reg] fisher_rot target={reg_fisher_rot_target}  '
                  f'ema_alpha={reg_fisher_rot_ema_alpha:.3f}  '
                  f'threshold_deg={reg_fisher_rot_threshold_deg}')

    # ── Training loop (full-batch: every sample every iter) ──
    best_mean_train_cc = -1.0
    best_iter = 0
    best_state = None
    history = []
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        per_sample_cc = []
        loss_sum = 0.0
        for s in train_samples:
            apply_pose(rast, s['pose'])
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            loss_k, _ = compute_ra_loss_rp(
                rp_real, rp_imag, s['gt_loss'], loss_type=loss_type)
            (loss_k * loss_scale).backward()
            with torch.no_grad():
                ra_polar = range_profile_to_ra_mag(rp_real.detach(), rp_imag.detach())
                ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                per_sample_cc.append(cc)
            loss_sum += loss_k.item() * loss_scale
            del rp_real, rp_imag, ra_polar, ra_cart, loss_k

        # H1 reg: L2 drift from init, mean over all elements (N·6).
        if reg_l2_drift_lambda > 0.0:
            drift = model.raw_materials - init_raw
            reg_loss = reg_l2_drift_lambda * drift.pow(2).mean()
            reg_loss.backward()
            loss_sum += float(reg_loss.item())

        # S2 reg: Fisher-weighted penalty on rotation drift from a target.
        # Target is either `init_normals` (Option 1) or an EMA of the
        # trajectory (Option 2 — reg_fisher_rot_target='ema'). Penalty is
        # either plain ||n − target||² or a Huber-style thresholded form
        # (Option 3 — reg_fisher_rot_threshold_deg > 0: penalise only
        # drift beyond threshold). Fisher weight F_i = Adam exp_avg_sq on
        # rotations (normalised to max = 1), snapshotted at warmup.
        if reg_fisher_rot_lambda > 0.0 and fisher_rot is not None:
            current_normals = model.get_normals()           # (N, 3) diff'ble
            target = normal_ema if reg_fisher_rot_target == 'ema' \
                                   and normal_ema is not None else init_normals
            if reg_rot_thresh_rad > 0.0:
                # Option 3: angle-thresholded. Penalise excess over θ₀.
                cos = (current_normals * target).sum(dim=-1).clamp(-1 + 1e-7,
                                                                    1 - 1e-7)
                angle = torch.acos(cos.abs())               # [0, π/2]
                excess = (angle - reg_rot_thresh_rad).clamp(min=0.0)
                drift_sq = excess.pow(2)                    # scalar per pt
            else:
                drift_sq = (current_normals - target).pow(2).sum(dim=-1)
            reg_loss_rot = reg_fisher_rot_lambda * (fisher_rot * drift_sq).sum()
            reg_loss_rot.backward()
            loss_sum += float(reg_loss_rot.item())

        # LR schedule (same as chirp-loop trainer)
        lr_scale = get_lr_scale(
            it, total_iters=num_iters,
            warmup_iters=0, warmup_factor=1.0, decay_start=200)
        for g in optimizer.param_groups:
            g['lr'] = base_lrs[g['name']] * lr_scale

        for g in optimizer.param_groups:
            clip = clip_vals.get(g['name'], 1.0)
            for p in g['params']:
                rms_clip_grad(p, clip)

        # H7 reg: snapshot a per-point training-Fisher proxy (Adam's
        # exp_avg_sq summed across columns) at warm-up, then zero gradients
        # on the bottom-(1-frac) points for the rest of training.
        if (reg_active_top_frac is not None
                and reg_active_top_frac > 0.0
                and it == reg_warm_iters):
            v = optimizer.state[model.raw_materials].get('exp_avg_sq')
            assert v is not None, 'Adam state missing at warmup boundary'
            per_pt_v = v.detach().sum(dim=-1)
            keep_n = int(reg_active_top_frac * per_pt_v.shape[0])
            threshold = per_pt_v.sort(descending=True).values[keep_n - 1]
            active_pts_mask = per_pt_v >= threshold
            if verbose:
                print(f'  [reg] iter {it}: activity mask — keeping '
                      f'{int(active_pts_mask.sum())}/{model.N} points '
                      f'(top {reg_active_top_frac*100:.1f}% by Adam v)')

        # S2 reg: snapshot per-point rotation Fisher (Adam exp_avg_sq
        # summed across 4 quaternion components) at warmup, normalise to
        # max=1. Used by the Fisher-weighted rotation drift penalty above.
        if (reg_fisher_rot_lambda > 0.0 and fisher_rot is None
                and it == reg_warm_iters):
            v = optimizer.state[model.rotations].get('exp_avg_sq')
            assert v is not None, 'Adam state missing for rotations at warmup'
            per_pt_v = v.detach().sum(dim=-1)                # (N,)
            denom = per_pt_v.max().clamp(min=1e-30)
            fisher_rot = (per_pt_v / denom).detach()         # (N,) in [0, 1]
            if verbose:
                srt = fisher_rot.sort(descending=True).values
                top_1 = int(0.01 * len(fisher_rot))
                top_5 = int(0.05 * len(fisher_rot))
                print(f'  [reg] iter {it}: rotation Fisher snapshot — '
                      f'top-1% mass={srt[:top_1].sum().item():.3f}  '
                      f'top-5% mass={srt[:top_5].sum().item():.3f}  '
                      f'(of total mass={fisher_rot.sum().item():.3f})')

        if active_pts_mask is not None:
            with torch.no_grad():
                if model.raw_materials.grad is not None:
                    model.raw_materials.grad[~active_pts_mask] = 0.0
                if model.rotations.grad is not None:
                    model.rotations.grad[~active_pts_mask] = 0.0

        optimizer.step()

        # S2 Option 2: update the normal EMA *after* the step. Initialised
        # to the snapshotted post-warmup normals so the target begins
        # where the optimiser settled during data-driven warmup.
        if (reg_fisher_rot_lambda > 0.0
                and reg_fisher_rot_target == 'ema'
                and fisher_rot is not None):
            with torch.no_grad():
                n_now = model.get_normals().detach()
                if normal_ema is None:
                    normal_ema = n_now.clone()
                else:
                    a = float(reg_fisher_rot_ema_alpha)
                    normal_ema = a * normal_ema + (1.0 - a) * n_now

        mean_tc = float(np.mean(per_sample_cc))
        history.append({
            'iter': it, 'loss': loss_sum,
            'mean_train_cc': mean_tc,
        })

        if mean_tc > best_mean_train_cc:
            best_mean_train_cc = mean_tc
            best_iter = it
            best_state = {k: v.data.clone() for k, v in model.state_dict().items()}

        if verbose and (it % 10 == 0 or it == num_iters - 1):
            elapsed = time.time() - t0
            print(f'  iter {it:4d}: loss={loss_sum:.5e}  '
                  f'mean_train_cc={mean_tc:.4f} '
                  f'(best={best_mean_train_cc:.4f}@{best_iter})  '
                  f'[{elapsed:.0f}s, n={len(train_samples)}]')

    train_elapsed = time.time() - t0

    # ── Restore best state + final test eval ──
    if best_state is not None:
        with torch.no_grad():
            for k, v in best_state.items():
                model.state_dict()[k].copy_(v)

    final_test_cc = _render_and_cart_corr(test_sample)
    final_train_ccs = [_render_and_cart_corr(s) for s in train_samples]
    final_train_mean = float(np.mean(final_train_ccs))

    if verbose:
        print(f'\n  [final] test  cc = {final_test_cc:.4f} '
              f'(init was {init_test_cc:.4f}, Δ {final_test_cc-init_test_cc:+.4f})')
        print(f'  [final] train mean cc = {final_train_mean:.4f}  '
              f'(best iter {best_iter})')
        print(f'  [final] elapsed {train_elapsed:.0f}s '
              f'({train_elapsed*1000/num_iters:.1f} ms/iter)')

    # ── Save ──
    if output_dir is None:
        tag = (f'train{len(train_frames)}frames_{len(train_loops)}loops'
               f'_test{test_frame}_loop{held_out_loop}')
        if test_in_train:
            tag = f'{tag}_ub'
        if use_pass2_alignment:
            tag = f'{tag}_pass2'
        if anchor_source == 'pass3_per_chirp':
            tag = f'{tag}_pass3anchor'
        if seed_frame != int(test_frame):
            tag = f'{tag}_seed{seed_frame}'
        if int(target_n) != 90000:
            tag = f'{tag}_N{int(target_n)}'
        if reg_l2_drift_lambda > 0.0:
            tag = f'{tag}_regL2{reg_l2_drift_lambda:g}'
        if reg_active_top_frac is not None:
            tag = f'{tag}_actTop{reg_active_top_frac:g}w{reg_warm_iters}'
        if reg_fisher_rot_lambda > 0.0:
            tag = f'{tag}_fshRot{reg_fisher_rot_lambda:g}w{reg_warm_iters}'
            if reg_fisher_rot_target != 'init':
                tag = f'{tag}_{reg_fisher_rot_target}'
            if reg_fisher_rot_threshold_deg > 0.0:
                tag = f'{tag}_thr{reg_fisher_rot_threshold_deg:g}'
        output_dir = os.path.join(
            PROJECT_ROOT, 'mm25DGS_v5', 'output_frame_nvs', f'{scene}_{tag}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'scene': scene,
        'train_frames': list(map(int, train_frames)),
        'train_loops': list(map(int, train_loops)),
        'test_frame': int(test_frame),
        'held_out_loop': int(held_out_loop),
        'test_in_train': bool(test_in_train),
        'num_iters': int(num_iters),
        'best_iter': int(best_iter),
        'best_mean_train_cc': float(best_mean_train_cc),
        'init_test_cc': float(init_test_cc),
        'final_test_cc': float(final_test_cc),
        'init_train_mean_cc': float(np.mean(init_train_ccs)),
        'final_train_mean_cc': float(final_train_mean),
        'init_train_std_cc': float(np.std(init_train_ccs)),
        'final_train_std_cc': float(np.std(final_train_ccs)),
        'n_train_samples': int(len(train_samples)),
        'edge_frames': diag['edge_frames'],
        'test_pose_mode': diag['test_pose_mode'],
        'alignment_source': 'pass-2' if use_pass2_alignment else 'pass-1',
        'anchor_source':    anchor_source,
        'seed_frame':       int(seed_frame),
        'mat_lr': float(mat_lr), 'rot_lr': float(rot_lr),
        'loss_type': loss_type,
        'target_n': int(target_n),
        'elapsed_s': float(train_elapsed),
        'reg_l2_drift_lambda': float(reg_l2_drift_lambda),
        'reg_active_top_frac': (None if reg_active_top_frac is None
                                 else float(reg_active_top_frac)),
        'reg_warm_iters': int(reg_warm_iters),
        'reg_active_n_kept': (None if active_pts_mask is None
                               else int(active_pts_mask.sum().item())),
        'reg_fisher_rot_lambda': float(reg_fisher_rot_lambda),
        'reg_fisher_rot_snapshotted': bool(fisher_rot is not None),
        'reg_fisher_rot_target': reg_fisher_rot_target,
        'reg_fisher_rot_ema_alpha': float(reg_fisher_rot_ema_alpha),
        'reg_fisher_rot_threshold_deg': float(reg_fisher_rot_threshold_deg),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(output_dir, 'history.npz'),
             iters=np.array([h['iter'] for h in history]),
             loss=np.array([h['loss'] for h in history]),
             mean_train_cc=np.array([h['mean_train_cc'] for h in history]))
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))

    if verbose:
        print(f'\n  results saved to: {output_dir}')
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Frame-level NVS trainer')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--test_frame', type=int, required=True)
    ap.add_argument('--train_frames', required=True,
                    help='comma-separated list, e.g. 434,435,436,437,439,440,441,442')
    ap.add_argument('--held_out_loop', type=int, default=0)
    ap.add_argument('--train_loops', default=None,
                    help='Comma-separated chirp-loop indices to train on; '
                         'default = all 16 loops. Pass "0" for first-chirp-'
                         'only training (NOT to be confused with single-chip '
                         'radar hardware — this is still the cascaded radar, '
                         'just restricted to its first chirp loop per frame).')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--mat_lr', type=float, default=0.01)
    ap.add_argument('--rot_lr', type=float, default=5e-3)
    ap.add_argument('--loss_type', default='mse_raw',
                    choices=['mse', 'pearson', 'mse_raw'])
    ap.add_argument('--target_n', type=int, default=90000)
    ap.add_argument('--frame_period_s', type=float, default=0.1)
    ap.add_argument('--loop_dt_s', type=float, default=7.87e-3 / 16.0)
    ap.add_argument('--anchor_source', default='pass2_lerp',
                    choices=['pass2_lerp', 'pass3_per_chirp'],
                    help='Per-chirp pose source. pass2_lerp (default): '
                         'LERP between pass-2 F-1 and F+1 (historical). '
                         'pass3_per_chirp: load each (frame, chirp)\'s Stage 3 '
                         'refined config directly; falls back to pass2_lerp '
                         'if any per-chirp file is missing.')
    ap.add_argument('--no_pass2', action='store_true',
                    help='Use pass-1 _aligned.json instead of pass-2')
    ap.add_argument('--reg_l2_drift_lambda', type=float, default=0.0,
                    help='H1: L2 penalty on raw_materials drift from init '
                         '(mean-over-N units). 0 = disabled.')
    ap.add_argument('--reg_active_top_frac', type=float, default=None,
                    help='H7: keep top-frac points by Adam exp_avg_sq '
                         '(training-set Fisher proxy) trainable after warm-up; '
                         'freeze the rest. None = disabled.')
    ap.add_argument('--reg_warm_iters', type=int, default=50,
                    help='H7 / S2: warmup iters before snapshotting '
                         'Fisher-weighted activity / rotation mask.')
    ap.add_argument('--reg_fisher_rot_lambda', type=float, default=0.0,
                    help='S2: Fisher-weighted L2 on rotation drift from '
                         'init. Penalty: λ · Σ_i F_i · ||n(q_i) − n(target_i)||². '
                         'F_i is the Adam exp_avg_sq on rotations, '
                         'summed across 4 quat components and normalised to '
                         'max=1, snapshotted at iter=reg_warm_iters. '
                         '0 = disabled.')
    ap.add_argument('--reg_fisher_rot_target', default='init',
                    choices=['init', 'ema'],
                    help='S2 target for the rotation drift penalty. '
                         '"init" (default, Option 1): LiDAR normal from '
                         'pcl.npy. "ema" (Option 2): running exponential '
                         'mean of the trajectory (see --reg_fisher_rot_ema_alpha).')
    ap.add_argument('--reg_fisher_rot_ema_alpha', type=float, default=0.95,
                    help='S2 Option 2: EMA decay for the rotation target '
                         '(only used when --reg_fisher_rot_target=ema).')
    ap.add_argument('--reg_fisher_rot_threshold_deg', type=float, default=0.0,
                    help='S2 Option 3: if > 0, penalise only rotation drift '
                         '*beyond* this threshold (in degrees). Soft Huber-'
                         'style: loss ∝ max(0, drift° − θ)². 0 = disabled '
                         '(full L2 penalty). Typical: 30–60°.')
    ap.add_argument('--seed_frame', type=int, default=None,
                    help='Frame whose pose seeds the rasterizer (drives FOV + '
                         'RX visibility before FPS, so it determines the 90k-'
                         'point subset of pcl.npy used for training). Default = '
                         '`test_frame` so HO and UB variants share an identical '
                         'position grid (see md/frame_nvs_analysis/findings.md '
                         '§0). Pass an explicit value to reproduce the legacy '
                         'HO grid (`--seed_frame {test_frame+1}`) or to probe '
                         'sensitivity to the seed.')
    args = ap.parse_args()

    train_frames = [int(x) for x in args.train_frames.split(',') if x.strip()]
    train_loops = (None if args.train_loops is None else
                   [int(x) for x in args.train_loops.split(',') if x.strip()])
    train_frame_nvs(
        scene=args.scene,
        train_frames=train_frames,
        test_frame=args.test_frame,
        held_out_loop=args.held_out_loop,
        train_loops=train_loops,
        num_iters=args.iters,
        mat_lr=args.mat_lr, rot_lr=args.rot_lr,
        loss_type=args.loss_type, target_n=args.target_n,
        frame_period_s=args.frame_period_s, loop_dt_s=args.loop_dt_s,
        use_pass2_alignment=not args.no_pass2,
        anchor_source=args.anchor_source,
        reg_l2_drift_lambda=args.reg_l2_drift_lambda,
        reg_active_top_frac=args.reg_active_top_frac,
        reg_warm_iters=args.reg_warm_iters,
        reg_fisher_rot_lambda=args.reg_fisher_rot_lambda,
        reg_fisher_rot_target=args.reg_fisher_rot_target,
        reg_fisher_rot_ema_alpha=args.reg_fisher_rot_ema_alpha,
        reg_fisher_rot_threshold_deg=args.reg_fisher_rot_threshold_deg,
        seed_frame=args.seed_frame)
