"""Crude-baseline sanity check: average the train GT RA maps at F-1
and F+1 (the bracketing train frames) and compute cart_corr against
the test GT RA map at F.

This is "zero-model interpolation": what cc do you get with no model,
just averaging the two adjacent training frames' GT? The number is a
lower-bound on what ANY useful renderer must beat.

Saves per-scene PNGs (linear + dB) of the prediction + GT, reports
per-scene cart_corr + 7-scene mean. Runs under v5 / M1 evaluation
conventions (same `adc_to_ra_complex` and polar→cart grid v5 uses,
bit-identical to the `final_test_cc` metric).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi  # noqa: E402
try:
    mi.set_variant('cuda_ad_rgb')
except Exception:
    pass  # already set in a previous import

from mmir.data.ra_utils import adc_to_ra_complex, save_ra_cartesian_png
from mmir.data.io_utils import compute_range_res_from_cfg
from mm25DGS_v5.train_gaussian import (
    build_polar_to_cart_grid, polar_to_cart_torch, cart_corr_torch, DEVICE,
)


SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_0_frame_390', 390),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
OUT_ROOT = os.path.join(PROJECT_ROOT,
                         'mm25DGS_v6/output_frame_nvs/_neighbour_avg_baseline')


def _load_adc_ri_chirp0(scene: str, frame: int) -> torch.Tensor:
    adc_path = os.path.join(DATA_ROOT, scene, 'radar',
                             f'cascaded_frame_{frame}.npy')
    arr = np.load(adc_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'unexpected ADC shape: {arr.shape}')
    ri = np.stack([arr[0].real, arr[0].imag], axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)                          # (TX, RX, K, 2)
    return torch.from_numpy(ri).to(DEVICE)


def _ra_polar_mag(adc_ri: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.abs(adc_to_ra_complex(adc_ri)).float()


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    rows = []
    for scene, F in SCENES:
        # v5 alignment pipeline uses pass-2 aligned config for range_res.
        cfg_path = os.path.join(
            DATA_ROOT, 'alignment_data', scene, 'cascade',
            f'cascaded_frame_{F}_aligned_pass2.json')
        range_res = compute_range_res_from_cfg(cfg_path)

        # Polar RA (magnitude) for F-1, F+1, F.
        ra_minus = _ra_polar_mag(_load_adc_ri_chirp0(scene, F - 1))
        ra_plus  = _ra_polar_mag(_load_adc_ri_chirp0(scene, F + 1))
        ra_test  = _ra_polar_mag(_load_adc_ri_chirp0(scene, F))

        # Cartesian via v5's exact grid.
        n_az, n_range = ra_test.shape
        sample_grid = build_polar_to_cart_grid(
            n_az, n_range, range_res, 400, DEVICE)

        cart_minus = polar_to_cart_torch(ra_minus, sample_grid)
        cart_plus  = polar_to_cart_torch(ra_plus,  sample_grid)
        cart_test  = polar_to_cart_torch(ra_test,  sample_grid)

        # Crude prediction: average of bracketing train cart maps.
        cart_pred = 0.5 * (cart_minus + cart_plus)

        # Normalise test exactly as v5 does for cart_corr, so this
        # matches the final_test_cc convention.
        def _norm(cart):
            mn, mx = cart.min(), cart.max()
            return (cart - mn) / (mx - mn).clamp(min=1e-30)

        cc = cart_corr_torch(_norm(cart_pred), _norm(cart_test)).item()

        # Save PNGs (linear + dB) for pred + GT.
        scene_dir = os.path.join(OUT_ROOT, f'{scene}_F{F}')
        os.makedirs(scene_dir, exist_ok=True)
        pred_np = cart_pred.cpu().numpy()
        gt_np   = cart_test.cpu().numpy()
        np.save(os.path.join(scene_dir, 'ra_pred_avg_cart.npy'), pred_np)
        np.save(os.path.join(scene_dir, 'ra_gt_cart.npy'),       gt_np)
        for scale in ('dB', 'linear'):
            save_ra_cartesian_png(
                gt_np,
                os.path.join(scene_dir, f'gt_ra_{scale}.png'),
                range_res=range_res, scale=scale,
                title=f'GT test ({scale}) — {scene} F={F}')
            save_ra_cartesian_png(
                pred_np,
                os.path.join(scene_dir, f'avg_pred_ra_{scale}.png'),
                range_res=range_res, scale=scale,
                title=f'½·(F-1 + F+1) avg ({scale}) — {scene}  '
                      f'cc={cc:.4f}')

        rows.append((scene, F, cc))
        print(f'  {scene:>20s}  F={F:<4d}  cart_cc = {cc:.4f}  '
              f'[saved {scene_dir}]')

    ccs = np.array([r[2] for r in rows])
    print()
    print('=' * 76)
    print(f'{"neighbour-average":<20s} |  '
          + '  '.join(f'{r[2]:.4f}' for r in rows)
          + f'  | mean = {ccs.mean():.4f}')
    print(f'{"v5 / M1 baseline":<20s} |  '
          + '0.6006  0.2947  0.5554  0.5667  0.5490  0.6492  0.4838'
          + f'  | mean = 0.5285')
    print('=' * 76)


if __name__ == '__main__':
    main()
