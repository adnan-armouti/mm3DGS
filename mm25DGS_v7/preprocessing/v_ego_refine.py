"""§4.0 — dedicated v_ego refinement preprocessing.

Modelled on the 2-stage pose alignment pipeline
(mmir/preprocessing/alignment/cascaded_alignment.py). Per-frame,
gradient-free, offline, cache-able.

Stages:
  0. SEED    — GT-trajectory-interpolated v_ego (what v_ego.py already does).
  1. COARSE  — 3-D grid search in a box around Stage-0 seed.
               Objective = |RAD| CC (flattened Pearson) using the init-state
               renderer (no trained BSDF — just geometry + default materials).
  2. FINE    — Nelder-Mead refinement, bounded, seeded from Stage-1.
               Same objective. Terminates at ~0.01 m/s per axis precision.
  3. SANITY  — cross-frame consistency: adjacent train frames should have
               v_egos within ~0.5 m/s. Flag outliers; fall back to Stage-0.

Output cached per frame to
  data/v_ego_cache/<scene>/frame_<F>_v_ego_refined.npy

and a per-scene manifest
  data/v_ego_cache/<scene>/v_ego_refined_manifest.json

Usage:
  python -m mm25DGS_v7.preprocessing.v_ego_refine \\
      --scene seq_0_frame_135 --frames "131,132,133,134,135,136,137,138,139"

The preprocessing uses the init (untrained) model. This is intentional
— we want v_ego estimation to be driven by geometry + GT ADC alone,
independent of any learned materials, so training and v_ego refinement
don't chase each other.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

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


def _build_init_renderer_at_frame(scene: str, frame: int,
                                    target_n: int = 20000,
                                    device: str = 'cuda'):
    """Build init model + rasterizer + v_ego0 at this frame."""
    import mitsuba as mi
    try: mi.set_variant('cuda_ad_rgb')
    except Exception: pass

    from mm25DGS_v5.rasterizer import Rasterizer
    from mm25DGS_v7.train_gaussian import (
        init_visible_weighted, cull_gaussians, USE_FACTORY_PATTERNS)
    from mm25DGS_v7.load_pretrained import load_trained_config, load_pattern_data
    from mm25DGS_v7.train_frame_nvs import _build_frame_poses, apply_pose
    from mm25DGS_v7.preprocessing.v_ego import get_or_compute_v_ego

    config = load_trained_config(scene)
    seed_cfg = None
    for suffix in ('_pass2', ''):
        p = os.path.join(DATA_ROOT, scene, 'configs',
                          f'cascaded_frame_{frame}{suffix}.json')
        if os.path.exists(p):
            seed_cfg = p; break
    assert seed_cfg is not None, \
        f'no cascade config for {scene} F={frame}'

    rast = Rasterizer(
        config_file=seed_cfg, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=device)
    if not USE_FACTORY_PATTERNS:
        patterns = load_pattern_data(scene)
        if patterns is not None:
            rast.inject_trained_params(pattern_data=patterns)

    poses, _ = _build_frame_poses(
        scene, frame, use_pass2=True, data_root=DATA_ROOT,
        loop_dt_s=7.87e-3/16, frame_period_s=0.1,
        anchor_source='pass2_lerp', device=device)
    apply_pose(rast, poses[0])

    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    active_mask  = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=device)
    vertex_areas[active_mask] = 1.0

    v_ego0 = torch.as_tensor(
        get_or_compute_v_ego(scene, int(frame), data_root=DATA_ROOT),
        dtype=torch.float32, device=device)
    return model, rast, active_mask, vertex_areas, v_ego0


def _gt_rad_mag(scene, frame, device='cuda'):
    from mm25DGS_v7.data.ra_utils import adc_to_rad_complex
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    adc = arr.transpose(0, 2, 1, 3)
    ri  = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    return adc_to_rad_complex(torch.from_numpy(ri).to(device)).abs().float()


def _render_rad_mag(model, rast, v_ego_vec, active_mask, vertex_areas):
    from mm25DGS_v7.train_gaussian import render_gaussians_doppler
    from mm25DGS_v7.data.ra_utils import rp_stack_to_rad_complex
    with torch.no_grad():
        rp_r, rp_i = render_gaussians_doppler(
            model, rast, v_ego_vec,
            vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None, n_chirps=16)
        rp_c = torch.complex(rp_r, rp_i)
        return rp_stack_to_rad_complex(rp_c).abs().float()


# ---------------------------------------------------------------------------
# Stage 1 — coarse grid search
# ---------------------------------------------------------------------------

def stage1_coarse_grid(scene: str, frame: int, target_n: int = 20000,
                        axis_steps: Tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2),
                        device: str = 'cuda',
                        _ctx=None) -> dict:
    """3-D grid search for v_ego in the box
      v_ego0 + (a, b, c) with a, b, c ∈ axis_steps.

    Returns: dict with best v_ego, best CC, and full grid table.
    """
    if _ctx is None:
        model, rast, mask, va, v_ego0 = _build_init_renderer_at_frame(
            scene, frame, target_n=target_n, device=device)
    else:
        model, rast, mask, va, v_ego0 = _ctx
    gt_mag = _gt_rad_mag(scene, frame, device=device)

    rows = []
    best = {'cc': -1.0, 'delta': (0.0, 0.0, 0.0), 'v_ego': v_ego0.clone()}
    for a in axis_steps:
        for b in axis_steps:
            for c in axis_steps:
                d = torch.tensor([a, b, c], device=device, dtype=torch.float32)
                v  = v_ego0 + d
                pr = _render_rad_mag(model, rast, v, mask, va)
                cc = _flat_pearson(pr, gt_mag)
                rows.append({'delta': (a, b, c),
                              'v_ego': v.cpu().numpy().tolist(),
                              'cc': cc})
                if cc > best['cc']:
                    best = {'cc': cc, 'delta': (a, b, c),
                            'v_ego': v.clone()}
    return {'scene': scene, 'frame': frame,
             'seed_v_ego_mps':  v_ego0.cpu().numpy().tolist(),
             'best_v_ego_mps':  best['v_ego'].cpu().numpy().tolist(),
             'best_delta_mps':  list(best['delta']),
             'best_cc':         best['cc'],
             'seed_cc':         next(r['cc'] for r in rows
                                       if r['delta'] == (0.0, 0.0, 0.0)),
             'grid_rows':       rows,
             '_ctx': (model, rast, mask, va, v_ego0)}


# ---------------------------------------------------------------------------
# Stage 2 — fine Nelder-Mead refinement
# ---------------------------------------------------------------------------

def stage2_fine_nelder_mead(scene: str, frame: int,
                              stage1: dict,
                              device: str = 'cuda',
                              bound_mps: float = 0.3,
                              target_n: int = 20000) -> dict:
    """Nelder-Mead refinement starting from Stage-1 best. Objective:
    −CC(pred_RAD, gt_RAD). Bounded to a box of ±bound_mps around
    Stage-0 seed."""
    model, rast, mask, va, v_ego0 = stage1['_ctx']
    gt_mag = _gt_rad_mag(scene, frame, device=device)

    seed = np.asarray(stage1['seed_v_ego_mps'], dtype=np.float32)
    x0   = np.asarray(stage1['best_v_ego_mps'], dtype=np.float32)

    def loss_fn(x_np):
        # Bound projection
        x_b = np.clip(x_np, seed - bound_mps, seed + bound_mps)
        v   = torch.as_tensor(x_b, dtype=torch.float32, device=device)
        pred = _render_rad_mag(model, rast, v, mask, va)
        return -_flat_pearson(pred, gt_mag)

    opt = minimize(
        loss_fn, x0,
        method='Nelder-Mead',
        options={
            'xatol':   1e-2,          # 0.01 m/s per-axis tolerance
            'fatol':   5e-4,          # 5e-4 CC change
            'maxiter': 80,
            'adaptive': True,
        })
    v_best = np.clip(opt.x, seed - bound_mps, seed + bound_mps)
    v_best_t = torch.as_tensor(v_best, dtype=torch.float32, device=device)
    pred = _render_rad_mag(model, rast, v_best_t, mask, va)
    cc   = _flat_pearson(pred, gt_mag)
    return {'scene': scene, 'frame': frame,
             'final_v_ego_mps': v_best.tolist(),
             'final_cc':        cc,
             'iter':            int(opt.nit),
             'fn_eval':         int(opt.nfev),
             'success':         bool(opt.success),
             'x0_cc':           stage1['best_cc']}


# ---------------------------------------------------------------------------
# Stage 3 — cross-frame consistency
# ---------------------------------------------------------------------------

def stage3_sanity_check(refined_by_frame: dict,
                         max_adjacent_delta_mps: float = 0.5) -> dict:
    """Adjacent frames (by numeric index) should have v_ego within
    max_adjacent_delta_mps. Outliers are flagged — caller can fall
    back to Stage-0 for them.
    """
    frames = sorted(refined_by_frame.keys())
    flags = {}
    for i, f in enumerate(frames):
        v_f = np.asarray(refined_by_frame[f]['final_v_ego_mps'])
        neighbours = []
        if i > 0:
            v_prev = np.asarray(refined_by_frame[frames[i-1]]['final_v_ego_mps'])
            neighbours.append(('prev', np.linalg.norm(v_f - v_prev)))
        if i < len(frames) - 1:
            v_next = np.asarray(refined_by_frame[frames[i+1]]['final_v_ego_mps'])
            neighbours.append(('next', np.linalg.norm(v_f - v_next)))
        max_delta = max((d for _, d in neighbours), default=0.0)
        flags[f] = {
            'max_adjacent_delta_mps': float(max_delta),
            'suspect': bool(max_delta > max_adjacent_delta_mps),
            'neighbours_delta_mps': {k: float(v) for k, v in neighbours},
        }
    return flags


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def refine_scene(scene: str, frames, target_n: int = 20000,
                  device: str = 'cuda', verbose: bool = True) -> dict:
    cache_dir = os.path.join(DATA_ROOT, 'v_ego_cache', scene)
    os.makedirs(cache_dir, exist_ok=True)

    refined = {}
    stage1_all = {}
    stage2_all = {}
    for F in frames:
        if verbose:
            print(f'\n=== {scene} F={F} ===')

        # Stage 1
        s1 = stage1_coarse_grid(scene, F, target_n=target_n, device=device)
        if verbose:
            print(f'  [S1] seed cc={s1["seed_cc"]:.4f}  '
                   f'→ best cc={s1["best_cc"]:.4f}  '
                   f'Δ={s1["best_delta_mps"]}')

        # Stage 2
        s2 = stage2_fine_nelder_mead(scene, F, s1, device=device,
                                        target_n=target_n)
        if verbose:
            print(f'  [S2] x0 cc={s2["x0_cc"]:.4f}  '
                   f'→ final cc={s2["final_cc"]:.4f}  '
                   f'(nit={s2["iter"]}, nfev={s2["fn_eval"]})')

        # Save per-frame cache
        cache_path = os.path.join(cache_dir, f'frame_{F}_v_ego_refined.npy')
        np.save(cache_path, np.asarray(s2['final_v_ego_mps'], dtype=np.float32))

        stage1_all[F] = s1
        stage2_all[F] = s2
        refined[F]    = s2

        # Release ctx (large GPU state)
        del s1['_ctx']
        if verbose:
            print(f'  [saved] {cache_path}')

    # Stage 3
    flags = stage3_sanity_check(refined)

    manifest = {
        'scene': scene,
        'frames': [int(f) for f in frames],
        'target_n': target_n,
        'per_frame': {
            int(F): {
                'seed_v_ego_mps':  stage1_all[F]['seed_v_ego_mps'],
                'stage1_best_v_ego_mps': stage1_all[F]['best_v_ego_mps'],
                'stage1_best_cc':  stage1_all[F]['best_cc'],
                'final_v_ego_mps': stage2_all[F]['final_v_ego_mps'],
                'final_cc':        stage2_all[F]['final_cc'],
                'sanity':          flags[F],
            } for F in frames
        }
    }
    mpath = os.path.join(cache_dir, 'v_ego_refined_manifest.json')
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f'\n[done] wrote {mpath}')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--frames', required=True,
                     help='comma-separated frame list, e.g. "131,132,...,139"')
    ap.add_argument('--target_n', type=int, default=20000)
    args = ap.parse_args()
    frames = [int(x) for x in args.frames.split(',') if x.strip()]
    refine_scene(args.scene, frames, target_n=args.target_n)


if __name__ == '__main__':
    main()
