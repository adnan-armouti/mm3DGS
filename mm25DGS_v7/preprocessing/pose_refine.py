"""§6 — pass-3 sub-mm coherent-sum pose alignment.

Per-frame refinement of the rigid TX/RX array translation against GT
chirp-0 |RA|. Starts from pass-2 aligned configs; outputs
`_aligned_pass3.json` files compatible with the existing training
pipeline's pass-loading convention.

Rationale (see md/diagnostics/RESULTS.md §3/§6):
  - σ = 1 mm TX jitter drops |RA| CC from 1.0 → 0.62 (random-phase floor)
  - σ = 0.1 mm retains 0.99
  - Pass-2 is roughly mm-accurate → we are ON the random-phase floor
  - Pass-3 targets sub-mm residual alignment

Algorithm (mirrors v_ego_refine.py):
  Stage 0 (new): per-frame warm-start — train materials for
           ``warm_start_iters`` v5-style iters (pass-2 pose, frozen)
           to get a decent BSDF fit. The init-state CC of ~0.1 is too
           noisy an objective; training for 200 iters lifts
           per-frame CC to ~0.9 (on single-frame fit), giving a much
           cleaner pose-error signal.
  Stage 1: 5^3 = 125 grid search over rigid translation Δp ∈ ±1.5 mm,
           step 0.75 mm. Objective = flat Pearson CC between rendered
           chirp-0 |RA| and GT chirp-0 |RA| using the warm-started
           model.
  Stage 2: Nelder-Mead on Δp, bounded from pass-2, seeded from Stage 1.
  Stage 3: Cross-frame sanity — adjacent-frame translations should be
           smooth; flag outliers (> 1 mm adjacent delta).

Output:
  data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass3.json
  data/alignment_data/<scene>/cascade/pose_refine_pass3_manifest.json

Only translation (3 DOF) — rotation refinement is a follow-up. At 77 GHz
/ λ=3.9 mm, sub-mm translation is the dominant phase-coherence driver;
rotation at the array center primarily affects differential phase, which
is second-order.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.optimize import minimize


DATA_ROOT = '/home/adnan/Desktop/mm3DGS/data'


def _flat_pearson(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    am = a - a.mean(); bm = b - b.mean()
    num = (am * bm).sum()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum()).clamp_min(1e-30)
    return float((num / den).item())


def _load_cascade_cfg(scene: str, frame: int, use_pass2: bool = True) -> dict:
    suffix = '_aligned_pass2' if use_pass2 else '_aligned'
    p = os.path.join(DATA_ROOT, 'alignment_data', scene, 'cascade',
                      f'cascaded_frame_{frame}{suffix}.json')
    assert os.path.isfile(p), f'missing {p}'
    with open(p, 'r') as f:
        return json.load(f)


def _apply_translation(cfg: dict, delta_mm: np.ndarray) -> dict:
    """Return a copy of cfg with every TX/RX pos_mm shifted by delta_mm."""
    cfg2 = copy.deepcopy(cfg)
    for arr_key in ('tx_array', 'rx_array'):
        for ant in cfg2[arr_key]:
            p = np.asarray(ant['pos_mm'], dtype=np.float64)
            ant['pos_mm'] = (p + delta_mm).tolist()
    return cfg2


def _cfg_to_pose_dict(cfg: dict, device='cuda') -> dict:
    """Match the shape/dtype of poses returned by ``_build_frame_poses``
    and consumed by ``apply_pose``."""
    tx = torch.as_tensor(
        [ant['pos_mm'] for ant in cfg['tx_array']],
        dtype=torch.float32, device=device) * 1e-3  # mm → m
    rx = torch.as_tensor(
        [ant['pos_mm'] for ant in cfg['rx_array']],
        dtype=torch.float32, device=device) * 1e-3
    tb = torch.as_tensor(
        [ant['boresight'] for ant in cfg['tx_array']],
        dtype=torch.float32, device=device)
    rb = torch.as_tensor(
        [ant['boresight'] for ant in cfg['rx_array']],
        dtype=torch.float32, device=device)
    return {'alpha': 0.5,
             'tx_positions':  tx.contiguous(),
             'rx_positions':  rx.contiguous(),
             'tx_boresights': tb.contiguous(),
             'rx_boresights': rb.contiguous()}


def _load_gt_ra_chirp0(scene: str, frame: int, device='cuda') -> torch.Tensor:
    """GT chirp-0 |RA| magnitude (127, 256)."""
    from mmir.data.ra_utils import adc_to_ra_complex
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    adc = arr.transpose(0, 2, 1, 3)          # (CH, TX, RX, ADC)
    ri  = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    ri0 = torch.from_numpy(ri[0]).to(device) # chirp 0, (TX, RX, ADC, 2)
    return adc_to_ra_complex(ri0).abs().float()  # (127, 256)


def _warm_start_train(model, rast, active_mask, vertex_areas,
                       pose, gt_cart_norm_gt_loss: dict,
                       warm_iters: int = 200, device: str = 'cuda',
                       verbose: bool = False):
    """Quick v5-style training on a single (frame, chirp-0) to lift the
    BSDF fit out of init-state noise before pose refinement.

    Operates on the model in-place. Keeps lr schedule + clip in step
    with train_frame_nvs but omits regularizers (just a clean MSE fit).
    """
    import time
    from mm25DGS_v7.train_gaussian import (
        render_gaussians, compute_ra_loss_rp,
        get_lr_scale, rms_clip_grad,
    )
    from mm25DGS_v7.train_frame_nvs import apply_pose
    # Ensure learnable params are in grad mode
    model.raw_materials.requires_grad_(True)
    model.rotations.requires_grad_(True)

    mat_lr, rot_lr = 0.01, 5e-3
    optimizer = torch.optim.Adam([
        {'params': [model.raw_materials], 'lr': mat_lr, 'name': 'materials'},
        {'params': [model.rotations],     'lr': rot_lr, 'name': 'rotations'},
    ])
    base_lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
    clip_vals = {'materials': 1.0, 'rotations': 0.5}

    gt_loss = gt_cart_norm_gt_loss['gt_loss']

    apply_pose(rast, pose)
    t0 = time.time()
    for it in range(warm_iters):
        optimizer.zero_grad(set_to_none=True)
        rp_r, rp_i = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None)
        loss, _ = compute_ra_loss_rp(rp_r, rp_i, gt_loss, loss_type='mse_raw')
        loss.backward()
        # LR schedule + clip
        lr_scale = get_lr_scale(it, total_iters=warm_iters,
                                  warmup_iters=0, warmup_factor=1.0,
                                  decay_start=max(warm_iters // 3, 50))
        for g in optimizer.param_groups:
            g['lr'] = base_lrs[g['name']] * lr_scale
            for p in g['params']:
                rms_clip_grad(p, clip_vals.get(g['name'], 1.0))
        optimizer.step()
    if verbose:
        print(f'    [warm] {warm_iters} iter in {time.time()-t0:.1f}s')
    # Freeze
    model.raw_materials.requires_grad_(False)
    model.rotations.requires_grad_(False)


def _build_init_renderer_for_frame(
    scene: str, frame: int, use_pass2: bool = True,
    target_n: int = 20000, device: str = 'cuda'):
    """Init-state renderer at the frame's pass-2 pose. Returns the rast
    + init model, ready for ``apply_pose`` updates."""
    import mitsuba as mi
    try: mi.set_variant('cuda_ad_rgb')
    except Exception: pass

    from mm25DGS_v5.rasterizer import Rasterizer
    from mm25DGS_v7.train_gaussian import (
        init_visible_weighted, cull_gaussians, USE_FACTORY_PATTERNS)
    from mm25DGS_v7.load_pretrained import load_trained_config, load_pattern_data
    from mm25DGS_v7.train_frame_nvs import apply_pose

    config = load_trained_config(scene)
    # seed config for Rasterizer init = frame's own pass-2 aligned json
    seed_cfg = os.path.join(
        DATA_ROOT, 'alignment_data', scene, 'cascade',
        f'cascaded_frame_{frame}_aligned_pass2.json')
    if not os.path.isfile(seed_cfg):
        # fall back to the unsuffixed aligned config
        seed_cfg = os.path.join(
            DATA_ROOT, 'alignment_data', scene, 'cascade',
            f'cascaded_frame_{frame}_aligned.json')
    assert os.path.isfile(seed_cfg), f'no aligned config for {scene} F={frame}'

    rast = Rasterizer(
        config_file=seed_cfg, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=device)
    if not USE_FACTORY_PATTERNS:
        patterns = load_pattern_data(scene)
        if patterns is not None:
            rast.inject_trained_params(pattern_data=patterns)

    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    active_mask  = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=device)
    vertex_areas[active_mask] = 1.0
    return rast, model, active_mask, vertex_areas, apply_pose


def _render_ra_mag(rast, model, pose, active_mask, vertex_areas, apply_pose):
    from mm25DGS_v7.train_gaussian import render_gaussians
    from mm25DGS_v7.data.ra_utils import _batch_txrx_to_vx_el0, _azimuth_fft_on_vx86
    apply_pose(rast, pose)
    with torch.no_grad():
        rp_r, rp_i = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None)
        rp_c = torch.complex(rp_r, rp_i)
        stack86 = _batch_txrx_to_vx_el0(rp_c.unsqueeze(0))     # (1, 86, R)
        ra_c    = _azimuth_fft_on_vx86(stack86)[0]              # (127, R)
    return ra_c.abs().float()


# ---------------------------------------------------------------------------
# Stage 1 — coarse grid search (translation only)
# ---------------------------------------------------------------------------

def stage1_translation_grid(scene: str, frame: int,
                              axis_steps_mm=(-0.75, -0.375, 0.0, 0.375, 0.75),
                              target_n: int = 20000, device: str = 'cuda',
                              warm_iters: int = 200,
                              _ctx=None, verbose: bool = False) -> dict:
    if _ctx is None:
        rast, model, mask, va, apply_pose = _build_init_renderer_for_frame(
            scene, frame, target_n=target_n, device=device)
        # Stage 0 — warm-start training on this frame's chirp 0.
        if warm_iters > 0:
            from mm25DGS_v7.train_frame_nvs import _build_per_loop_gt
            adc_npy = os.path.join(
                DATA_ROOT, scene, 'radar', f'cascaded_frame_{frame}.npy')
            gt = _build_per_loop_gt(adc_npy, loop_idx=0,
                                     loss_type='mse_raw', device=device)
            cfg_pass2 = _load_cascade_cfg(scene, frame, use_pass2=True)
            pose_pass2 = _cfg_to_pose_dict(cfg_pass2, device=device)
            _warm_start_train(model, rast, mask, va, pose_pass2,
                                gt, warm_iters=warm_iters, device=device,
                                verbose=verbose)
    else:
        rast, model, mask, va, apply_pose = _ctx
    cfg0 = _load_cascade_cfg(scene, frame, use_pass2=True)
    gt_ra = _load_gt_ra_chirp0(scene, frame, device=device)

    best_cc = -1.0
    best_delta = np.zeros(3, dtype=np.float32)
    grid_rows = []
    for a in axis_steps_mm:
        for b in axis_steps_mm:
            for c in axis_steps_mm:
                d = np.array([a, b, c], dtype=np.float64)
                cfg_d = _apply_translation(cfg0, d)
                pose  = _cfg_to_pose_dict(cfg_d, device=device)
                ra_mag = _render_ra_mag(rast, model, pose, mask, va,
                                          apply_pose)
                cc = _flat_pearson(ra_mag, gt_ra)
                grid_rows.append({'delta_mm': [float(a), float(b), float(c)],
                                    'cc': cc})
                if cc > best_cc:
                    best_cc = cc
                    best_delta = d.astype(np.float32)

    seed_cc = next(r['cc'] for r in grid_rows
                    if r['delta_mm'] == [0.0, 0.0, 0.0])
    return {'scene': scene, 'frame': frame,
             'seed_cc':        seed_cc,
             'best_cc':        best_cc,
             'best_delta_mm':  best_delta.tolist(),
             'grid_rows':      grid_rows,
             '_ctx': (rast, model, mask, va, apply_pose),
             '_cfg0': cfg0, '_gt_ra': gt_ra}


# ---------------------------------------------------------------------------
# Stage 2 — Nelder-Mead fine refinement
# ---------------------------------------------------------------------------

def stage2_nelder_mead(stage1: dict, bound_mm: float = 1.0,
                         device: str = 'cuda') -> dict:
    rast, model, mask, va, apply_pose = stage1['_ctx']
    cfg0 = stage1['_cfg0']
    gt_ra = stage1['_gt_ra']

    x0 = np.asarray(stage1['best_delta_mm'], dtype=np.float32)

    def loss_fn(x_np):
        x = np.clip(x_np, -bound_mm, bound_mm)
        cfg_d = _apply_translation(cfg0, x.astype(np.float64))
        pose  = _cfg_to_pose_dict(cfg_d, device=device)
        ra    = _render_ra_mag(rast, model, pose, mask, va, apply_pose)
        return -_flat_pearson(ra, gt_ra)

    opt = minimize(
        loss_fn, x0, method='Nelder-Mead',
        options={
            'xatol':   5e-3,  # 0.005 mm precision per axis
            'fatol':   5e-4,  # 5e-4 CC
            'maxiter': 100,
            'adaptive': True,
        })
    x_best = np.clip(opt.x, -bound_mm, bound_mm)
    cfg_best = _apply_translation(cfg0, x_best.astype(np.float64))
    pose_best = _cfg_to_pose_dict(cfg_best, device=device)
    ra_best = _render_ra_mag(rast, model, pose_best, mask, va, apply_pose)
    cc_best = _flat_pearson(ra_best, gt_ra)

    return {'scene': stage1['scene'], 'frame': stage1['frame'],
             'final_delta_mm':  x_best.tolist(),
             'final_cc':        cc_best,
             'iter':            int(opt.nit),
             'fn_eval':         int(opt.nfev),
             'x0_cc':           stage1['best_cc'],
             'cfg_final':       cfg_best}


# ---------------------------------------------------------------------------
# Stage 3 — cross-frame consistency
# ---------------------------------------------------------------------------

def stage3_sanity_check(refined: Dict[int, dict],
                         max_adj_delta_mm: float = 1.0) -> dict:
    frames = sorted(refined.keys())
    flags = {}
    for i, F in enumerate(frames):
        p = np.asarray(refined[F]['final_delta_mm'])
        neighbours = []
        if i > 0:
            q = np.asarray(refined[frames[i-1]]['final_delta_mm'])
            neighbours.append(('prev', float(np.linalg.norm(p - q))))
        if i < len(frames) - 1:
            q = np.asarray(refined[frames[i+1]]['final_delta_mm'])
            neighbours.append(('next', float(np.linalg.norm(p - q))))
        max_d = max((d for _, d in neighbours), default=0.0)
        flags[F] = {
            'max_adj_delta_mm': float(max_d),
            'suspect': bool(max_d > max_adj_delta_mm),
            'neighbours_delta_mm': {k: v for k, v in neighbours},
        }
    return flags


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def refine_scene(scene: str, frames, target_n: int = 20000,
                  device: str = 'cuda', verbose: bool = True) -> dict:
    align_dir = os.path.join(DATA_ROOT, 'alignment_data', scene, 'cascade')
    os.makedirs(align_dir, exist_ok=True)

    refined = {}
    per_frame = {}
    for F in frames:
        if verbose: print(f'\n=== {scene} F={F} ===')

        s1 = stage1_translation_grid(scene, F, target_n=target_n, device=device)
        if verbose:
            print(f'  [S1] seed cc={s1["seed_cc"]:.4f}  '
                   f'best cc={s1["best_cc"]:.4f}  '
                   f'Δ={s1["best_delta_mm"]} mm')
        s2 = stage2_nelder_mead(s1, device=device)
        if verbose:
            print(f'  [S2] x0 cc={s2["x0_cc"]:.4f}  '
                   f'final cc={s2["final_cc"]:.4f}  '
                   f'Δ={s2["final_delta_mm"]} mm  '
                   f'(nit={s2["iter"]}, nfev={s2["fn_eval"]})')

        # Write the pass-3 config
        out_cfg_path = os.path.join(
            align_dir, f'cascaded_frame_{F}_aligned_pass3.json')
        cfg_final = s2['cfg_final']
        # Attach provenance
        cfg_final['_pass3_refinement'] = {
            'delta_mm':  s2['final_delta_mm'],
            'seed_cc':   s1['seed_cc'],
            'final_cc':  s2['final_cc'],
            'Δcc':       s2['final_cc'] - s1['seed_cc'],
            'method':    'pose_refine_pass3_translation_only',
        }
        with open(out_cfg_path, 'w') as f:
            json.dump(cfg_final, f, indent=2)
        if verbose: print(f'  [saved] {out_cfg_path}')

        refined[F]   = s2
        per_frame[F] = {
            'seed_cc':         s1['seed_cc'],
            'stage1_best_cc':  s1['best_cc'],
            'stage1_best_delta_mm': s1['best_delta_mm'],
            'final_cc':        s2['final_cc'],
            'final_delta_mm':  s2['final_delta_mm'],
            'Δcc':             s2['final_cc'] - s1['seed_cc'],
            'config_out':      out_cfg_path,
        }
        # Release ctx
        s1['_ctx'] = None
        s2['cfg_final'] = None

    # Stage 3
    sanity = stage3_sanity_check(refined)
    for F, sn in sanity.items():
        per_frame[F]['sanity'] = sn

    manifest = {
        'scene': scene,
        'frames': [int(f) for f in frames],
        'target_n': target_n,
        'per_frame': {int(F): p for F, p in per_frame.items()},
        'summary': {
            'mean_seed_cc':  float(np.mean([p['seed_cc'] for p in per_frame.values()])),
            'mean_final_cc': float(np.mean([p['final_cc'] for p in per_frame.values()])),
            'mean_Δcc':      float(np.mean([p['Δcc'] for p in per_frame.values()])),
            'mean_abs_delta_mm': float(np.mean(
                [np.linalg.norm(p['final_delta_mm']) for p in per_frame.values()])),
        },
    }
    mpath = os.path.join(align_dir, 'pose_refine_pass3_manifest.json')
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        s = manifest['summary']
        print(f'\n=== {scene} summary ===')
        print(f'  seed mean CC  = {s["mean_seed_cc"]:.4f}')
        print(f'  final mean CC = {s["mean_final_cc"]:.4f}')
        print(f'  Δ mean CC     = {s["mean_Δcc"]:+.4f}')
        print(f'  mean |Δp|     = {s["mean_abs_delta_mm"]:.3f} mm')
        print(f'[done] wrote {mpath}')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--frames', required=True,
                     help='comma-separated frame list')
    ap.add_argument('--target_n', type=int, default=20000)
    args = ap.parse_args()
    frames = [int(x) for x in args.frames.split(',') if x.strip()]
    refine_scene(args.scene, frames, target_n=args.target_n)


if __name__ == '__main__':
    main()
