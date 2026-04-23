"""§2.3 + §2.4 — sensitivity sweeps on untrained (or init-only) renderer.

§2.3  v_ego sensitivity: scale v_ego by {0, 0.5, 0.8, 0.9, 0.95, 1.0,
      1.05, 1.1, 1.2, 1.5, 2.0, -1.0}, render |RAD|, compute CC vs GT.
      Peak of the curve = optimal scale. Width around the peak = how
      tolerant |RAD| CC is to v_ego magnitude error. Negative scale
      catches a sign bug.

§2.4  TX position jitter: add gaussian noise with σ ∈ {0.1, 0.3, 1, 3,
      10} mm to rast.tx_positions, render |RA|, compute CC vs
      unperturbed. Slope of curve quantifies how much alignment
      tolerance we have.

Both sweeps use the *init* model (before any training) so the
representation effect is isolated from training dynamics.

Runs on seq_0_frame_135 only (pilot); can be re-run per scene.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mm25DGS_v7.data.ra_utils import (
    adc_to_rad_complex,
    _batch_txrx_to_vx_el0, _azimuth_fft_on_vx86,
    rp_stack_to_rad_complex,
)
from mm25DGS_v7.preprocessing.v_ego import get_or_compute_v_ego
from mmir.data.ra_utils import adc_to_ra_complex as _mmir_adc_to_ra_complex

DATA_ROOT = '/home/adnan/Desktop/mm3DGS/data'
OUT_DIR   = '/home/adnan/Desktop/mm3DGS/md/diagnostics'


def _flat_pearson(a, b):
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    am = a - a.mean(); bm = b - b.mean()
    num = (am * bm).sum()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum()).clamp_min(1e-30)
    return float((num / den).item())


def _load_gt_ra(scene, frame, device='cuda'):
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    adc = arr.transpose(0, 2, 1, 3)
    ri  = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    ri  = torch.from_numpy(ri[0]).to(device)                     # chirp 0
    return _mmir_adc_to_ra_complex(ri)                           # (127, R) complex


def _load_gt_rad(scene, frame, device='cuda'):
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    adc = arr.transpose(0, 2, 1, 3)
    ri  = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    return adc_to_rad_complex(torch.from_numpy(ri).to(device))


def _build_init_model(scene, frame, target_n=20000, device='cuda'):
    """Re-use train_frame_nvs' Rasterizer construction path to get an
    init-state model at the target frame's pose."""
    import mitsuba as mi
    try: mi.set_variant('cuda_ad_rgb')
    except Exception: pass

    from mm25DGS_v5.rasterizer import Rasterizer
    from mm25DGS_v7.train_gaussian import (
        init_visible_weighted, cull_gaussians,
        USE_FACTORY_PATTERNS,
    )
    from mm25DGS_v7.load_pretrained import load_trained_config, load_pattern_data
    from mm25DGS_v7.train_frame_nvs import _build_frame_poses, apply_pose
    import os

    config = load_trained_config(scene)
    # Seed config for the rasterizer: use the frame's own aligned config
    data_root = DATA_ROOT
    seed_cfg = None
    for suffix in ('_pass2', ''):
        p = os.path.join(data_root, scene, 'configs',
                          f'cascaded_frame_{frame}{suffix}.json')
        if os.path.exists(p):
            seed_cfg = p; break
    assert seed_cfg is not None, f'no cascade config for {scene} F={frame}'

    rast = Rasterizer(
        config_file=seed_cfg,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=device)
    if not USE_FACTORY_PATTERNS:
        patterns = load_pattern_data(scene)
        if patterns is not None:
            rast.inject_trained_params(pattern_data=patterns)

    # Apply chirp-0 pose so FOV mask is consistent with the bench setup.
    poses, _ = _build_frame_poses(
        scene, frame, use_pass2=True, data_root=data_root,
        loop_dt_s=7.87e-3/16, frame_period_s=0.1,
        anchor_source='pass2_lerp', device=device)
    apply_pose(rast, poses[0])

    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    active_mask  = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=device)
    vertex_areas[active_mask] = 1.0
    return model, rast, active_mask, vertex_areas


def _render_ra(model, rast, active_mask, vertex_areas):
    from mm25DGS_v7.train_gaussian import render_gaussians
    with torch.no_grad():
        rp_r, rp_i = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None)
        rp_c = torch.complex(rp_r, rp_i)
        stack86 = _batch_txrx_to_vx_el0(rp_c.unsqueeze(0))
        return _azimuth_fft_on_vx86(stack86)[0]     # (127, R) complex


def _render_rad(model, rast, v_ego, active_mask, vertex_areas):
    from mm25DGS_v7.train_gaussian import render_gaussians_doppler
    with torch.no_grad():
        rp_r, rp_i = render_gaussians_doppler(
            model, rast, v_ego,
            vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None, n_chirps=16)
        rp_c = torch.complex(rp_r, rp_i)
        return rp_stack_to_rad_complex(rp_c)             # (D, 127, R) complex


# ---------------------------------------------------------------------------
# §2.3 v_ego sensitivity
# ---------------------------------------------------------------------------

def sweep_v_ego(scene, frame, device='cuda', target_n=20000):
    model, rast, mask, va = _build_init_model(scene, frame, target_n, device)
    gt_rad_mag = _load_gt_rad(scene, frame, device).abs().float()
    v_ego0 = torch.as_tensor(
        get_or_compute_v_ego(scene, int(frame), data_root=DATA_ROOT),
        dtype=torch.float32, device=device)

    scales = [-1.0, 0.0, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
    rows = []
    for s in scales:
        v_ego = v_ego0 * s
        pred_rad = _render_rad(model, rast, v_ego, mask, va).abs().float()
        cc = _flat_pearson(pred_rad, gt_rad_mag)
        rows.append({'scale': s, 'v_ego_norm_mps': float((v_ego0 * s).norm().item()),
                      'rad_cc': cc})
        print(f'  v_ego × {s:+.2f}:  |v|={float((v_ego0*s).norm().item()):.2f} m/s  '
              f'|RAD| cc={cc:.4f}')

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = [r['scale'] for r in rows]; ys = [r['rad_cc'] for r in rows]
    ax.plot(xs, ys, 'o-', lw=1.4)
    ax.axvline(1.0, color='g', ls='--', alpha=0.4, label='GT v_ego')
    ax.axvline(0.0, color='k', ls=':', alpha=0.4, label='v_ego = 0')
    ax.axhline(0.0, color='k', lw=0.3, alpha=0.3)
    ax.set_xlabel('v_ego scale (× GT)'); ax.set_ylabel('|RAD| CC vs GT')
    ax.set_title(f'{scene} F={frame}  v_ego sensitivity sweep  '
                  f'(|v_GT|={float(v_ego0.norm().item()):.2f} m/s)')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'sweep_vego_{scene}_F{frame}.png')
    fig.savefig(out_png, dpi=130); plt.close(fig)
    return {'rows': rows, 'plot': out_png,
             'v_ego_GT_mps': float(v_ego0.norm().item())}


# ---------------------------------------------------------------------------
# §2.4 TX position jitter
# ---------------------------------------------------------------------------

def sweep_tx_jitter(scene, frame, device='cuda', target_n=20000):
    model, rast, mask, va = _build_init_model(scene, frame, target_n, device)
    # Baseline render (no jitter)
    ra_base = _render_ra(model, rast, mask, va).abs().float()
    tx0 = rast.tx_positions.clone()

    sigmas_mm = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    rows = []
    torch.manual_seed(2026)
    for sigma_mm in sigmas_mm:
        sigma_m = sigma_mm * 1e-3
        noise = torch.randn_like(tx0) * sigma_m
        rast.tx_positions.copy_(tx0 + noise)
        try:
            ra_j = _render_ra(model, rast, mask, va).abs().float()
            cc_vs_base = _flat_pearson(ra_j, ra_base)
            rows.append({'sigma_mm': sigma_mm, 'cc_vs_unjittered': cc_vs_base})
            print(f'  σ={sigma_mm:>4.1f} mm  |  CC(jit, base) = {cc_vs_base:.4f}')
        finally:
            rast.tx_positions.copy_(tx0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = [r['sigma_mm'] for r in rows]; ys = [r['cc_vs_unjittered'] for r in rows]
    ax.semilogx(xs[1:], ys[1:], 'o-', lw=1.4)  # skip σ=0 for log axis
    ax.axhline(1.0, color='g', ls='--', alpha=0.4, label='no jitter')
    ax.set_xlabel('TX jitter σ (mm)'); ax.set_ylabel('CC(jittered, baseline)')
    ax.set_title(f'{scene} F={frame}  TX position jitter sensitivity')
    ax.grid(True, alpha=0.3, which='both'); ax.legend()
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'sweep_tx_jitter_{scene}_F{frame}.png')
    fig.savefig(out_png, dpi=130); plt.close(fig)
    return {'rows': rows, 'plot': out_png}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='seq_0_frame_135')
    ap.add_argument('--frame', type=int, default=135)
    ap.add_argument('--target_n', type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'\n===== {args.scene} F={args.frame} =====\n')
    print('§2.3 v_ego sensitivity sweep')
    res_v = sweep_v_ego(args.scene, args.frame, target_n=args.target_n)
    print('\n§2.4 TX position jitter sweep')
    res_j = sweep_tx_jitter(args.scene, args.frame, target_n=args.target_n)

    out = {'scene': args.scene, 'frame': args.frame,
            'target_n': args.target_n,
            'vego_sweep': res_v, 'tx_jitter_sweep': res_j}
    out_path = os.path.join(OUT_DIR,
                             f'sensitivity_{args.scene}_F{args.frame}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[done] {out_path}')


if __name__ == '__main__':
    main()
