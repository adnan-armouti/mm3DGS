"""Phase α: diagnose the rendered-vs-GT scale mismatch in the v4 forward model.

For each of the 7 benchmark scenes, at ITU-concrete init (no training):

  1. Build the scene exactly as `train_gaussians` does.
  2. One forward pass of `render_gaussians` → `(rp_real, rp_imag)` → |RA_rendered|.
  3. Load gt_adc_ri, run adc_to_ra_complex → |RA_gt|.
  4. Compute per-scene scale statistics:
       - scale_mean   = mean(|RA_rend|) / mean(|RA_gt|)
       - scale_median = median(|RA_rend|) / median(|RA_gt|)
       - log-magnitude histograms
       - per-range-bin ratio (vector length K)
       - per-channel ratio (n_tx, n_rx)

Dumps scale_report.json + per-scene .npz + summary PNG to
mm25DGS_v4/output/scale_investigation/.

The report is the input to Phase β (the fix).
"""

import os
import sys
import json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mm25DGS_v4.train_gaussian import (
    render_gaussians, init_visible_weighted, cull_gaussians,
    range_profile_to_ra,
    DEVICE,
)
from mm25DGS_v4.rasterizer import Rasterizer
from mm25DGS_v4.load_pretrained import load_trained_config, load_pattern_data, SCENES
from mmir.data.ra_utils import adc_to_ra_complex

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', 'scale_investigation')


def investigate_scene(scene, verbose=True, inject_patterns=True):
    config = load_trained_config(scene)
    rast = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )
    if inject_patterns:
        pattern_data = load_pattern_data(scene)
        rast.inject_trained_params(pattern_data=pattern_data)
    model = init_visible_weighted(scene, rast, target_n=50000)
    rast.free_mi_scene()

    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    # One forward pass at ITU-concrete init (no training, no grad)
    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None, bsdf_mode='full')
        ra_rend = range_profile_to_ra(rp_real, rp_imag)
        ra_rend_mag = ra_rend.abs().cpu().numpy()   # (n_tx, n_rx, K) — shape depends

    # GT magnitude via the same adc_to_ra_complex pipeline used in the loss
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_adc_ri = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        ra_gt = adc_to_ra_complex(gt_adc_ri)
        ra_gt_mag = ra_gt.abs().cpu().numpy()

    # Ensure matching shape — sometimes rp-side ra has a different axis order
    # The loss pipeline uses `range_profile_to_ra` for rendered and
    # `adc_to_ra_complex` for GT; both should produce (n_az, n_range) or similar.
    if verbose:
        print(f"\n--- {scene} ---")
        print(f"  ra_rend shape: {ra_rend_mag.shape}  dtype={ra_rend_mag.dtype}")
        print(f"  ra_gt   shape: {ra_gt_mag.shape}  dtype={ra_gt_mag.dtype}")
        print(f"  rend: min={ra_rend_mag.min():.3e} max={ra_rend_mag.max():.3e} mean={ra_rend_mag.mean():.3e} median={np.median(ra_rend_mag):.3e}")
        print(f"  gt  : min={ra_gt_mag.min():.3e} max={ra_gt_mag.max():.3e} mean={ra_gt_mag.mean():.3e} median={np.median(ra_gt_mag):.3e}")

    # Flatten and compute scale statistics
    scale_mean = float(ra_rend_mag.mean() / max(ra_gt_mag.mean(), 1e-30))
    scale_median = float(np.median(ra_rend_mag) / max(np.median(ra_gt_mag), 1e-30))

    # Per-axis ratio - need matching shape. Both are 2D (n_az, n_range) after
    # range_profile_to_ra / adc_to_ra_complex for the standard pipeline.
    if ra_rend_mag.shape == ra_gt_mag.shape and ra_rend_mag.ndim == 2:
        # (n_az, n_range) — per-range-bin ratio across azimuths
        rend_per_range = ra_rend_mag.mean(axis=0) + 1e-30
        gt_per_range = ra_gt_mag.mean(axis=0) + 1e-30
        scale_by_range = (rend_per_range / gt_per_range).astype(np.float32)
        scale_by_channel = None  # not available at the 2D stage
    else:
        scale_by_range = None
        scale_by_channel = None

    # Log-magnitude histograms for visualization
    log_rend = np.log10(ra_rend_mag + 1e-30).flatten()
    log_gt = np.log10(ra_gt_mag + 1e-30).flatten()
    rend_hist, rend_edges = np.histogram(log_rend, bins=80, range=(-10, 10))
    gt_hist, gt_edges = np.histogram(log_gt, bins=80, range=(-10, 10))

    report = {
        'scene': scene,
        'rend_mean': float(ra_rend_mag.mean()),
        'gt_mean': float(ra_gt_mag.mean()),
        'rend_median': float(np.median(ra_rend_mag)),
        'gt_median': float(np.median(ra_gt_mag)),
        'rend_max': float(ra_rend_mag.max()),
        'gt_max': float(ra_gt_mag.max()),
        'scale_mean': scale_mean,
        'scale_median': scale_median,
        'log10_scale_mean': float(np.log10(max(scale_mean, 1e-30))),
        'log10_scale_median': float(np.log10(max(scale_median, 1e-30))),
    }
    per_scene_dump = {
        'scale_by_range': scale_by_range,
        'log_rend_hist': rend_hist,
        'log_gt_hist': gt_hist,
        'log_edges': rend_edges,
    }
    return report, per_scene_dump


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--factory', action='store_true',
                        help='Use factory patterns (skip inject_trained_params)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_reports = []
    all_dumps = {}
    inject = not args.factory
    label = 'factory' if args.factory else 'mmIR'
    print(f"\n=== Using {label} antenna patterns ===")
    for scene in SCENES:
        try:
            report, dump = investigate_scene(scene, inject_patterns=inject)
            all_reports.append(report)
            all_dumps[scene] = dump
        except Exception as e:
            print(f"  {scene}: ERROR {e}")
            raise

    # Summary
    print("\n" + "=" * 70)
    print("SCALE MISMATCH SUMMARY")
    print("=" * 70)
    print(f"{'scene':<22}{'rend_mean':>14}{'gt_mean':>14}{'scale_mean':>14}{'log10':>10}")
    for r in all_reports:
        print(f"{r['scene']:<22}{r['rend_mean']:>14.3e}{r['gt_mean']:>14.3e}"
              f"{r['scale_mean']:>14.3e}{r['log10_scale_mean']:>10.2f}")
    log10_scales = [r['log10_scale_mean'] for r in all_reports]
    print(f"\nLog10 scale_mean across scenes: "
          f"min={min(log10_scales):.2f}  max={max(log10_scales):.2f}  "
          f"spread={max(log10_scales)-min(log10_scales):.2f}")
    print(f"Geometric mean scale_mean: {10 ** np.mean(log10_scales):.3e}")

    # Per-range-bin shape check on the first scene
    first_dump = all_dumps[SCENES[0]]
    if first_dump['scale_by_range'] is not None:
        sbr = first_dump['scale_by_range']
        print(f"\nPer-range-bin ratio for {SCENES[0]}: "
              f"shape={sbr.shape}  min={sbr.min():.2e}  max={sbr.max():.2e}  "
              f"mean={sbr.mean():.2e}  std={sbr.std():.2e}")
        print(f"  Range-bin variation (log10): "
              f"{np.log10(max(sbr.max(), 1e-30)) - np.log10(max(sbr.min(), 1e-30)):.2f} decades")

    # Dump report + histograms
    with open(os.path.join(OUTPUT_ROOT, 'scale_report.json'), 'w') as f:
        json.dump({
            'reports': all_reports,
            'log10_min': float(min(log10_scales)),
            'log10_max': float(max(log10_scales)),
            'log10_spread': float(max(log10_scales) - min(log10_scales)),
            'log10_mean': float(np.mean(log10_scales)),
        }, f, indent=2)

    # Plot log-magnitude histograms
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, scene in enumerate(SCENES):
        ax = axes.flat[i]
        dump = all_dumps[scene]
        edges = dump['log_edges']
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, dump['log_rend_hist'], width=0.25,
               color='C0', alpha=0.6, label='rendered')
        ax.bar(centers + 0.25, dump['log_gt_hist'], width=0.25,
               color='C3', alpha=0.6, label='GT')
        ax.set_xlabel('log10(|RA|)')
        ax.set_ylabel('count')
        ax.set_title(f'{scene}\nlog10 scale_mean = {all_reports[i]["log10_scale_mean"]:+.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes.flat[7].axis('off')
    fig.suptitle('Phase α: |RA_rendered| vs |RA_gt| at ITU-concrete init')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_ROOT, 'log_magnitude_histograms.png'), dpi=120)
    plt.close(fig)
    print(f"\nDumped report to {OUTPUT_ROOT}/")


if __name__ == '__main__':
    main()
