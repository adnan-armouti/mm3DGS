"""§2.2 — visualisation dump: pred vs GT on |RA| and |RAD|.

For a given (scene, checkpoint) renders the full model and produces
diagnostic PNGs that make it visually obvious where the model fails:

1. |RA| images side-by-side (GT, pred, |pred - GT|)
2. Range profile at peak azimuth bin (log-scale, GT vs pred)
3. Doppler spectrum at peak (range, az) (GT vs pred)
4. Full |RA| error heatmap (rel residual)
5. Per-point contribution map — max |w_full| across (tx, rx) projected
   into scene space and saved as an .xyz colour-by-log cloud
   (loadable in MeshLab / Open3D for inspection)
6. Material-histograms at final state (6 params) vs init

Run:
  python -m mm25DGS_v7.scripts.diagnostics.visualize_pred_vs_gt \
      --scene seq_0_frame_135 --frame 135 \
      --ckpt mm25DGS_v7/output_frame_nvs/<dir>/best_model.pt

Writes to md/diagnostics/viz_<scene>_F<N>/...
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
    _batch_txrx_to_vx_el0,
    _azimuth_fft_on_vx86,
    _doppler_fft_on_chirps,
    _hann,
    N_DOP_DEFAULT,
    rp_stack_to_rad_complex,
)
from mm25DGS_v7.preprocessing.v_ego import get_or_compute_v_ego


DATA_ROOT = '/home/adnan/Desktop/mm3DGS/data'
OUT_ROOT  = '/home/adnan/Desktop/mm3DGS/md/diagnostics'


def _flat_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float(); b = b.reshape(-1).float()
    am = a - a.mean(); bm = b - b.mean()
    num = (am * bm).sum()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum()).clamp_min(1e-30)
    return float((num / den).item())


def _load_gt_ra(scene: str, frame: int, device='cuda') -> torch.Tensor:
    """GT chirp-0 |RA| (127, 256) complex."""
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    adc = arr.transpose(0, 2, 1, 3)
    x_c = torch.from_numpy(adc[0]).to(device).to(torch.complex64)   # (TX, RX, ADC)
    n_adc = x_c.shape[-1]
    win_r = _hann(n_adc, x_c.device, x_c.real.dtype).to(x_c.dtype)
    x_c = x_c * win_r[None, None, :]
    rp  = torch.fft.fft(x_c, n=n_adc, dim=-1)                        # (TX, RX, R)
    rp_stack = rp.unsqueeze(0)                                        # (1, TX, RX, R)
    stack86  = _batch_txrx_to_vx_el0(rp_stack)                        # (1, 86, R)
    ra       = _azimuth_fft_on_vx86(stack86)[0]                       # (127, R)
    return ra


def _load_gt_rad(scene: str, frame: int, device='cuda') -> torch.Tensor:
    """GT (D, 127, R) complex."""
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    adc = arr.transpose(0, 2, 1, 3)
    ri  = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    return adc_to_rad_complex(torch.from_numpy(ri).to(device))


def _render_pred(model, rast, v_ego, active_mask, vertex_areas, doppler: bool):
    """Render chirp-0 |RA| (+ |RAD| if doppler)."""
    from mm25DGS_v7.train_gaussian import render_gaussians, render_gaussians_doppler
    if doppler:
        rp_r, rp_i = render_gaussians_doppler(
            model, rast, v_ego,
            vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None, n_chirps=16)
        rp_c = torch.complex(rp_r, rp_i)
        # chirp 0 → |RA|, plus |RAD|
        stack86 = _batch_txrx_to_vx_el0(rp_c[0:1])
        ra0 = _azimuth_fft_on_vx86(stack86)[0]
        rad = rp_stack_to_rad_complex(rp_c)
        return ra0, rad
    else:
        rp_r, rp_i = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None)
        rp_c = torch.complex(rp_r, rp_i)
        stack86 = _batch_txrx_to_vx_el0(rp_c.unsqueeze(0))
        ra0 = _azimuth_fft_on_vx86(stack86)[0]
        return ra0, None


def _plot_ra_panel(gt_ra_mag, pred_ra_mag, out_png, title):
    """Three-panel: GT, pred, |diff| on log scale."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmin_g, vmax_g = gt_ra_mag.min().item(), gt_ra_mag.max().item()
    vmin_p = max(pred_ra_mag.min().item(), 1e-9)
    log_gt   = np.log10(gt_ra_mag.cpu().numpy().clip(min=1e-6))
    log_pred = np.log10(pred_ra_mag.cpu().numpy().clip(min=1e-6))
    vmin = min(log_gt.min(), log_pred.min())
    vmax = max(log_gt.max(), log_pred.max())

    im0 = axes[0].imshow(log_gt, aspect='auto', origin='lower', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{title}\nGT log10|RA|'); fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(log_pred, aspect='auto', origin='lower', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[1].set_title('Pred log10|RA|'); fig.colorbar(im1, ax=axes[1])

    rel = (pred_ra_mag - gt_ra_mag).abs() / gt_ra_mag.clamp_min(1e-9)
    im2 = axes[2].imshow(rel.cpu().numpy().clip(0, 3.0),
                          aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=3.0)
    axes[2].set_title('|pred - GT| / GT  (clipped 0–3)'); fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('range bin'); ax.set_ylabel('azimuth bin')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def _plot_range_profile(gt_mag, pred_mag, out_png, title):
    """GT vs pred 1-D range profile at azimuth bin of max(GT)."""
    gt_np = gt_mag.cpu().numpy()
    pd_np = pred_mag.cpu().numpy()
    peak_az = int(np.unravel_index(np.argmax(gt_np), gt_np.shape)[0])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gt_np[peak_az, :], label='GT',  lw=1.3)
    ax.plot(pd_np[peak_az, :], label='pred', lw=1.3, alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('range bin'); ax.set_ylabel('|RA|')
    ax.set_title(f'{title}  —  range profile at peak-GT az bin {peak_az}')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def _plot_rad_doppler_slice(gt_rad_mag, pred_rad_mag, out_png, title):
    """GT vs pred Doppler spectrum at the peak (range, az) cell, + 3 panels
    of |RAD| at D=center, D=center-5, D=center+5."""
    gt  = gt_rad_mag.cpu().numpy()
    pd  = pred_rad_mag.cpu().numpy()
    D, A, R = gt.shape
    # Find peak (D, a, r) in GT (restricted to range 20..120 to skip clutter)
    gt_sub = gt[:, :, 20:120]
    d0, a0, r0 = np.unravel_index(np.argmax(gt_sub), gt_sub.shape)
    r0 += 20

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Top-left: Doppler spectrum at (a0, r0)
    ax = axes[0, 0]
    ax.plot(gt[:, a0, r0], label='GT',   lw=1.4)
    ax.plot(pd[:, a0, r0], label='pred', lw=1.4, alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('Doppler bin'); ax.set_ylabel('|RAD|')
    ax.set_title(f'{title}\nDoppler spectrum at peak GT cell (az={a0}, r={r0})')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Top-right: |RAD| slice at D = GT peak Doppler bin
    d_peak = d0
    log_gt = np.log10(gt[d_peak].clip(min=1e-6))
    log_pd = np.log10(pd[d_peak].clip(min=1e-6))
    vmin = min(log_gt.min(), log_pd.min()); vmax = max(log_gt.max(), log_pd.max())
    ax = axes[0, 1]
    im = ax.imshow(log_gt, aspect='auto', origin='lower', cmap='viridis',
                    vmin=vmin, vmax=vmax)
    ax.set_title(f'GT log10|RAD| at D={d_peak} (peak)')
    ax.set_xlabel('range'); ax.set_ylabel('azimuth')
    fig.colorbar(im, ax=ax)

    # Bottom-left: pred at D_peak
    ax = axes[1, 0]
    im = ax.imshow(log_pd, aspect='auto', origin='lower', cmap='viridis',
                    vmin=vmin, vmax=vmax)
    ax.set_title(f'Pred log10|RAD| at D={d_peak}')
    ax.set_xlabel('range'); ax.set_ylabel('azimuth')
    fig.colorbar(im, ax=ax)

    # Bottom-right: |RAD| slice at zero Doppler (D = center bin)
    d_zero = D // 2
    log_gt_z = np.log10(gt[d_zero].clip(min=1e-6))
    log_pd_z = np.log10(pd[d_zero].clip(min=1e-6))
    vmin_z = min(log_gt_z.min(), log_pd_z.min())
    vmax_z = max(log_gt_z.max(), log_pd_z.max())
    # Actually this fourth panel will be (pred - gt) at zero Doppler
    rel = (pd[d_zero] - gt[d_zero]) / np.clip(np.abs(gt[d_zero]), 1e-9, None)
    ax = axes[1, 1]
    im = ax.imshow(np.clip(rel, -2, 2), aspect='auto', origin='lower',
                    cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_title(f'(pred - GT)/GT at D={d_zero} (zero-Doppler slice)')
    ax.set_xlabel('range'); ax.set_ylabel('azimuth')
    fig.colorbar(im, ax=ax)

    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def _load_trained_model(ckpt_path, scene, frame, device='cuda'):
    """Reconstruct model from a train_frame_nvs checkpoint."""
    out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    config_path = os.path.join(out_dir, 'results.json')
    assert os.path.isfile(config_path), f'no results.json in {out_dir}'
    cfg = json.load(open(config_path))

    import mitsuba as mi
    try: mi.set_variant('cuda_ad_rgb')
    except Exception: pass

    from mm25DGS_v5.rasterizer import Rasterizer
    from mm25DGS_v7.train_gaussian import (
        init_visible_weighted, cull_gaussians, USE_FACTORY_PATTERNS)
    from mm25DGS_v7.load_pretrained import load_trained_config, load_pattern_data
    from mm25DGS_v7.train_frame_nvs import _build_frame_poses, apply_pose

    config = load_trained_config(scene)
    seed_cfg = None
    for suffix in ('_pass2', ''):
        p = os.path.join(DATA_ROOT, scene, 'configs',
                          f'cascaded_frame_{frame}{suffix}.json')
        if os.path.exists(p):
            seed_cfg = p; break
    assert seed_cfg is not None

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

    target_n = cfg.get('target_n', 20000)
    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    active_mask  = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=device)
    vertex_areas[active_mask] = 1.0

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    return model, rast, active_mask, vertex_areas, cfg


def visualize(scene, frame, ckpt_path, tag_prefix, doppler=True,
               device='cuda'):
    """Generate all diagnostic PNGs for one scene + checkpoint."""
    out_dir = os.path.join(OUT_ROOT, f'viz_{scene}_F{frame}_{tag_prefix}')
    os.makedirs(out_dir, exist_ok=True)

    # GT
    gt_ra = _load_gt_ra(scene, frame, device)                # (127, 256) cpx
    gt_rad = None
    if doppler:
        gt_rad = _load_gt_rad(scene, frame, device)           # (D, 127, 256) cpx
    gt_ra_mag  = gt_ra.abs().float()
    gt_rad_mag = gt_rad.abs().float() if gt_rad is not None else None

    # Load trained model + render at the frame's pose
    model, rast, active_mask, vertex_areas, cfg = _load_trained_model(
        ckpt_path, scene, frame, device)

    v_ego = None
    if doppler:
        v_e = get_or_compute_v_ego(scene, int(frame), data_root=DATA_ROOT)
        v_ego = torch.as_tensor(v_e, dtype=torch.float32, device=device)

    pred_ra, pred_rad = _render_pred(
        model, rast, v_ego, active_mask, vertex_areas, doppler=doppler)
    pred_ra_mag  = pred_ra.abs().float()
    pred_rad_mag = pred_rad.abs().float() if pred_rad is not None else None

    # Numbers
    ra_cc = _flat_pearson(pred_ra_mag, gt_ra_mag)
    rad_cc = (_flat_pearson(pred_rad_mag, gt_rad_mag)
               if pred_rad_mag is not None else None)
    stats = {'scene': scene, 'frame': frame,
              'ra_cc': ra_cc, 'rad_cc': rad_cc,
              'ckpt': ckpt_path, 'tag': tag_prefix}
    print(f'[viz] {tag_prefix}  |RA| cc={ra_cc:.4f}  '
          f'|RAD| cc={rad_cc if rad_cc is None else f"{rad_cc:.4f}"}')

    # 1) |RA| panel
    _plot_ra_panel(gt_ra_mag, pred_ra_mag,
                    os.path.join(out_dir, '1_ra_panel.png'),
                    f'{scene} F={frame} [{tag_prefix}]  |RA| cc={ra_cc:.3f}')

    # 2) Range profile at peak az
    _plot_range_profile(
        gt_ra_mag, pred_ra_mag,
        os.path.join(out_dir, '2_range_profile.png'),
        f'{scene} F={frame} [{tag_prefix}]')

    # 3) Doppler spectrum + |RAD| panels
    if pred_rad_mag is not None:
        _plot_rad_doppler_slice(
            gt_rad_mag, pred_rad_mag,
            os.path.join(out_dir, '3_rad_doppler.png'),
            f'{scene} F={frame} [{tag_prefix}]  |RAD| cc={rad_cc:.3f}')

    # 4) Full |RA| error heatmap already inside panel (#1). Standalone:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rel = ((pred_ra_mag - gt_ra_mag).abs() / gt_ra_mag.clamp_min(1e-9)).cpu().numpy()
    im = ax.imshow(rel.clip(0, 3.0), aspect='auto', origin='lower',
                    cmap='magma', vmin=0, vmax=3.0)
    ax.set_title(f'|RA| rel error heatmap — {tag_prefix}')
    ax.set_xlabel('range'); ax.set_ylabel('azimuth')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '4_ra_relerror.png'), dpi=130)
    plt.close(fig)

    # 5) Per-point contribution map (point cloud .xyz + PNG overhead view)
    # We recompute path weights on the fly — re-run render with a tap
    # to capture `w_full`. For simplicity, dump positions with magnitudes
    # derived from the material scale as a proxy.
    with torch.no_grad():
        pos = model.positions[active_mask].detach().cpu().numpy()
        raw_mat = model.raw_materials[active_mask].detach().cpu().numpy()
        # Use eps_real and tau_base as "activity" proxies
        from mm25DGS.bsdf_torch import reparameterize_torch
        phys = reparameterize_torch(
            model.raw_materials[active_mask]).detach().cpu().numpy()
        tau = phys[:, 4]   # tau_base — blend factor
        eps_r = phys[:, 0] # dielectric real part

    # Save XYZ + activity
    xyz_path = os.path.join(out_dir, '5_points.xyz')
    with open(xyz_path, 'w') as f:
        for (x, y, z), t, er in zip(pos, tau, eps_r):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {t:.4f} {er:.4f}\n')

    # Overhead scatter PNG colored by tau
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=tau, s=1.2, cmap='inferno',
                     vmin=0, vmax=1)
    ax.set_aspect('equal')
    ax.set_title(f'{scene} F={frame} [{tag_prefix}]  top-down view, colour=τ (KA/SPM blend)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    fig.colorbar(sc, ax=ax, label='tau_base'); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '5_points_overhead.png'), dpi=120)
    plt.close(fig)

    # 6) Material histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    names = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau_base', 'thickness']
    for i, n in enumerate(names):
        ax = axes[i // 3, i % 3]
        ax.hist(phys[:, i], bins=60, color='#1f77b4', alpha=0.8)
        ax.set_title(f'{n}')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'{scene} F={frame} [{tag_prefix}]  learned-material histograms')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '6_material_hist.png'), dpi=120)
    plt.close(fig)

    # Summary json
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'[viz] wrote 6 PNGs + .xyz to {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--ckpt',  required=True,
                     help='Path to best_model.pt from a train_frame_nvs output dir')
    ap.add_argument('--tag',   default='viz')
    ap.add_argument('--no-doppler', action='store_true',
                     help='Skip |RAD| diagnostics (for v5-style checkpoints)')
    args = ap.parse_args()
    visualize(args.scene, args.frame, args.ckpt, args.tag,
               doppler=not args.no_doppler)


if __name__ == '__main__':
    main()
