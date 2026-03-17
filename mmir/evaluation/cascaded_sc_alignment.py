#!/usr/bin/env python3
"""
Cascaded-Mediated Single-Chip Radar Alignment

Two-stage alignment that uses the cascaded radar's existing LiDAR alignment
to bootstrap the single-chip radar alignment:

  Stage 1: Transfer the rigid-body transform (GPS/IMU correction) found by
           cascaded alignment to the single-chip config via SVD Procrustes.
  Stage 2: (Optional) Small grid search over residual perturbations (range,
           azimuth, rotation) that maximizes rendered↔GT RA correlation.

Usage:
    # Stage 1 only (default):
    python mmir/fmcw/cascaded_mediated_sc_alignment.py \
        --scene-dir data/seq_1_frame_185 \
        --cascaded-frame 185 \
        --sc-frame 369

    # Stage 1 + Stage 2 grid search:
    python mmir/fmcw/cascaded_mediated_sc_alignment.py \
        --scene-dir data/seq_1_frame_185 \
        --cascaded-frame 185 \
        --sc-frame 369 \
        --run-stage2 --grid-steps 5
"""

# Must set Mitsuba variant before mmir.sensor/renderer imports mi.Vector3f
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate2d
from scipy.optimize import minimize
from scipy.stats import pearsonr

from mmir.evaluation.utils.single_view_viz import (
    load_config_positions_and_boresight,
)
from mmir.data.ra_utils import adc_to_ra_image_single_chip
from mmir.data.io_utils import compute_range_res_from_cfg
from mmir.losses.loss_utils import minmax_normalize_numpy


# ---------------------------------------------------------------------------
# Stage 1: Board-frame transform transfer
# ---------------------------------------------------------------------------

def _extract_all_positions(config_path: str) -> np.ndarray:
    """Extract all antenna positions (TX + RX) from a config, in meters."""
    tx_m, rx_m, _, _ = load_config_positions_and_boresight(config_path)
    return np.vstack([tx_m, rx_m])  # (N, 3)


def compute_board_transform(
    casc_orig_config: str,
    casc_aligned_config: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the rigid-body transform from original → aligned cascaded config.

    Uses SVD-based Procrustes alignment on all antenna positions (TX + RX)
    to find the optimal rotation R and translation t such that:
        P_aligned ≈ R @ P_orig + t

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    P_orig = _extract_all_positions(casc_orig_config)     # (N, 3)
    P_aligned = _extract_all_positions(casc_aligned_config)  # (N, 3)

    assert P_orig.shape == P_aligned.shape, (
        f"Position count mismatch: {P_orig.shape} vs {P_aligned.shape}"
    )

    # Centroids
    c1 = P_orig.mean(axis=0)
    c2 = P_aligned.mean(axis=0)

    # Cross-covariance matrix
    H = (P_orig - c1).T @ (P_aligned - c2)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # Translation
    t = c2 - R @ c1

    return R, t


def transfer_alignment_to_sc(
    casc_orig_config: str,
    casc_aligned_config: str,
    sc_orig_config: str,
    output_path: Optional[str] = None,
) -> str:
    """Stage 1: Transfer cascaded alignment to the single-chip config.

    Computes the rigid-body transform (R, t) from original → aligned cascaded,
    then applies it to all single-chip antenna positions and boresight.

    Args:
        casc_orig_config: Path to original cascaded config JSON
        casc_aligned_config: Path to aligned cascaded config JSON
        sc_orig_config: Path to original single-chip config JSON
        output_path: Output path for aligned SC config (auto-generated if None)

    Returns:
        Path to the aligned single-chip config
    """
    R, t = compute_board_transform(casc_orig_config, casc_aligned_config)

    # Load original SC config
    with open(sc_orig_config, 'r') as f:
        sc_cfg = json.load(f)

    # Transform each antenna position (stored in mm, work in meters)
    for arr_key in ('tx_array', 'rx_array'):
        for ant in sc_cfg[arr_key]:
            pos_m = np.array(ant['pos_mm'], dtype=np.float64) / 1000.0
            pos_aligned = R @ pos_m + t
            ant['pos_mm'] = (pos_aligned * 1000.0).tolist()

            # Transform boresight direction
            if 'boresight' in ant:
                bore = np.array(ant['boresight'], dtype=np.float64)
                bore_aligned = R @ bore
                bore_aligned /= np.linalg.norm(bore_aligned)
                ant['boresight'] = bore_aligned.tolist()

    # Save
    if output_path is None:
        stem = Path(sc_orig_config).stem
        output_path = str(Path(sc_orig_config).parent / f"{stem}_cascaded_aligned.json")

    with open(output_path, 'w') as f:
        json.dump(sc_cfg, f, indent=2)

    return output_path


# ---------------------------------------------------------------------------
# Stage 2: RA cross-correlation refinement
# ---------------------------------------------------------------------------

def _load_cascaded_ra(gt_adc_path: str) -> np.ndarray:
    """Load cascaded GT ADC and convert to RA magnitude map.

    GT format: (1, 256, 12, 16) complex64
    Returns: (127, 256) float32 magnitude
    """
    import torch
    from mmir.data.ra_utils import adc_to_ra_image

    arr = np.load(gt_adc_path)  # (1, 256, 12, 16) complex64
    cmpl = np.squeeze(arr[0]).transpose(1, 0, 2)  # (12, 256, 16) → (16, 12, 256) no...
    # load_target does: arr[0] → squeeze → transpose(1,0,2) → view_as_real
    # arr shape is (1, 256, 12, 16) complex64
    # arr[0] → (256, 12, 16)
    # squeeze is no-op if already 3D
    # transpose(1,0,2) → (12, 256, 16)
    cmpl = arr[0].transpose(1, 0, 2)  # (12, 256, 16)
    t = torch.view_as_real(torch.from_numpy(cmpl)).float()  # (12, 256, 16, 2)
    ra = adc_to_ra_image(t)  # (127, 256)
    return ra.numpy()


def _load_single_chip_ra(gt_adc_path: str) -> np.ndarray:
    """Load single-chip GT ADC and convert to RA magnitude map.

    GT format: (128, 4, 3, 128) complex128
    Returns: (63, 128) float32 magnitude
    """
    gt_raw = np.load(gt_adc_path)
    gt_complex = gt_raw[0].transpose(1, 0, 2)  # (3, 4, 128) complex
    gt_adc = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
    return adc_to_ra_image_single_chip(gt_adc)  # (63, 128)


def resample_ra_to_sc_grid(
    casc_ra: np.ndarray,
    casc_range_res: float,
    sc_n_az: int,
    sc_n_range: int,
    sc_range_res: float,
    az_smooth_sigma: float = 3.0,
) -> np.ndarray:
    """Resample cascaded RA (127, 256) to single-chip grid (63, 128).

    Both azimuth axes span the same angular range (FFT of virtual array);
    only the number of bins differs. Range axes have different resolutions.

    After resampling, applies Gaussian smoothing along azimuth to approximate
    the single-chip's broader beamwidth.
    """
    casc_n_az, casc_n_range = casc_ra.shape

    # Build coordinate grids
    # Azimuth: both span normalized [-1, 1] (sin-space after FFT)
    az_casc = np.linspace(0, 1, casc_n_az)
    az_sc = np.linspace(0, 1, sc_n_az)

    # Range: physical meters
    r_casc = np.arange(casc_n_range) * casc_range_res
    r_sc = np.arange(sc_n_range) * sc_range_res

    # Clip SC range to cascaded extent
    max_casc_range = r_casc[-1]
    r_sc_clipped = np.clip(r_sc, r_casc[0], max_casc_range)

    # Interpolate
    interp = RegularGridInterpolator(
        (az_casc, r_casc), casc_ra.astype(np.float64),
        method='linear', bounds_error=False, fill_value=0.0,
    )

    az_grid, r_grid = np.meshgrid(az_sc, r_sc_clipped, indexing='ij')
    resampled = interp((az_grid, r_grid)).astype(np.float32)

    # Smooth azimuth to approximate SC beamwidth
    if az_smooth_sigma > 0:
        resampled = gaussian_filter1d(resampled, sigma=az_smooth_sigma, axis=0)

    return resampled


def find_ra_shift(
    ref_ra: np.ndarray,
    target_ra: np.ndarray,
    max_shift_az: int = 10,
    max_shift_range: int = 5,
) -> Tuple[int, int, float]:
    """Find 2D pixel shift between reference and target RA using cross-correlation.

    Returns:
        (shift_az, shift_range, peak_correlation) — shifts to apply to ref to match target
    """
    # Normalize both
    ref = minmax_normalize_numpy(ref_ra)
    tgt = minmax_normalize_numpy(target_ra)

    # Full cross-correlation
    cc = correlate2d(tgt, ref, mode='full')

    # Center of cross-correlation = zero-shift position
    center_az = ref.shape[0] - 1
    center_r = ref.shape[1] - 1

    # Restrict search to max_shift window
    az_lo = max(0, center_az - max_shift_az)
    az_hi = min(cc.shape[0], center_az + max_shift_az + 1)
    r_lo = max(0, center_r - max_shift_range)
    r_hi = min(cc.shape[1], center_r + max_shift_range + 1)

    cc_window = cc[az_lo:az_hi, r_lo:r_hi]

    # Find peak
    peak_idx = np.unravel_index(np.argmax(cc_window), cc_window.shape)
    shift_az = peak_idx[0] + az_lo - center_az
    shift_r = peak_idx[1] + r_lo - center_r

    # Compute Pearson correlation at the optimal shift
    shifted_ref = np.roll(np.roll(ref, shift_az, axis=0), shift_r, axis=1)
    mask = np.isfinite(shifted_ref.ravel()) & np.isfinite(tgt.ravel())
    if mask.sum() > 1:
        corr, _ = pearsonr(shifted_ref.ravel()[mask], tgt.ravel()[mask])
    else:
        corr = 0.0

    return int(shift_az), int(shift_r), float(corr)


def refine_sc_alignment(
    casc_gt_adc_path: str,
    sc_gt_adc_path: str,
    sc_config_path: str,
    casc_config_path: str,
    output_path: Optional[str] = None,
    skip_range_bins: int = 5,
    az_smooth_sigma: float = 3.0,
    verbose: bool = True,
) -> Dict:
    """Stage 2: Refine single-chip alignment using cascaded↔SC GT RA correlation.

    Returns dict with: shift_az, shift_range, initial_corr, refined_corr, output_path
    """
    # Load range resolutions
    casc_range_res = compute_range_res_from_cfg(casc_config_path)
    sc_range_res = compute_range_res_from_cfg(sc_config_path)

    if verbose:
        print(f"  Cascaded range_res: {casc_range_res:.4f} m")
        print(f"  SC range_res: {sc_range_res:.4f} m")

    # Load RA images
    if verbose:
        print("  Loading cascaded GT RA...")
    casc_ra = _load_cascaded_ra(casc_gt_adc_path)
    if verbose:
        print(f"    Shape: {casc_ra.shape}")

    if verbose:
        print("  Loading single-chip GT RA...")
    sc_ra = _load_single_chip_ra(sc_gt_adc_path)
    if verbose:
        print(f"    Shape: {sc_ra.shape}")

    sc_n_az, sc_n_range = sc_ra.shape

    # Resample cascaded RA to SC grid
    if verbose:
        print("  Resampling cascaded RA to SC grid...")
    casc_resampled = resample_ra_to_sc_grid(
        casc_ra, casc_range_res, sc_n_az, sc_n_range, sc_range_res,
        az_smooth_sigma=az_smooth_sigma,
    )
    if verbose:
        print(f"    Resampled shape: {casc_resampled.shape}")

    # Skip TX-RX leakage bins
    casc_crop = casc_resampled[:, skip_range_bins:]
    sc_crop = sc_ra[:, skip_range_bins:]

    # Compute initial correlation (no shift)
    ref_norm = minmax_normalize_numpy(casc_crop)
    tgt_norm = minmax_normalize_numpy(sc_crop)
    initial_corr, _ = pearsonr(ref_norm.ravel(), tgt_norm.ravel())
    if verbose:
        print(f"  Initial casc↔SC correlation: {initial_corr:.4f}")

    # Find optimal shift
    shift_az, shift_r, refined_corr = find_ra_shift(
        casc_crop, sc_crop, max_shift_az=10, max_shift_range=5,
    )
    if verbose:
        print(f"  Optimal shift: az={shift_az} bins, range={shift_r} bins")
        print(f"  Refined correlation: {refined_corr:.4f}")

    # Convert pixel shift to physical offset
    # Azimuth: SC FFT has az_fft_size=64 with half-lambda spacing
    # Each bin covers Δsin(θ) = 1/32, which is ~1.79° near boresight
    az_fft_size = sc_n_az + 1  # 63 + 1 = 64
    sin_per_bin = 2.0 / az_fft_size  # 2/64 = 1/32
    az_bin_deg = np.degrees(np.arcsin(min(sin_per_bin, 1.0)))
    delta_azimuth_deg = shift_az * az_bin_deg
    delta_range_m = shift_r * sc_range_res

    if verbose:
        print(f"  Physical offset: range={delta_range_m:.3f} m, azimuth={delta_azimuth_deg:.2f}°")

    # Apply refinement if shift is non-trivial
    result = {
        'shift_az_bins': shift_az,
        'shift_range_bins': shift_r,
        'delta_range_m': delta_range_m,
        'delta_azimuth_deg': delta_azimuth_deg,
        'initial_corr': float(initial_corr),
        'refined_corr': float(refined_corr),
    }

    if abs(shift_az) > 0 or abs(shift_r) > 0:
        from mmir.preprocessing.alignment.cascaded_lidar import create_aligned_config

        if output_path is None:
            stem = Path(sc_config_path).stem
            output_path = str(Path(sc_config_path).parent / f"{stem}_final_aligned.json")

        # Apply small correction via create_aligned_config
        # Note: create_aligned_config applies rotation + translation
        # For a pure translation refinement, rotation params are 0
        final_path = create_aligned_config(
            config_path=sc_config_path,
            delta_range_m=delta_range_m,
            delta_azimuth_deg=delta_azimuth_deg,
            rotation_elev_deg=0.0,
            rotation_azim_deg=0.0,
            output_path=output_path,
        )
        result['output_path'] = final_path
        if verbose:
            print(f"  Refined config saved: {final_path}")
    else:
        result['output_path'] = sc_config_path
        if verbose:
            print("  No shift needed — Stage 1 config is final.")

    return result


# ---------------------------------------------------------------------------
# Stage 2: Render-based grid search refinement
# ---------------------------------------------------------------------------

def render_alignment_grid_search(
    base_config: str,
    gt_adc_path: str,
    mesh_file: str,
    training_dir: str,
    output_config_path: str,
    range_search_m: float = 0.5,
    azimuth_search_deg: float = 3.0,
    rotation_search_deg: float = 2.0,
    n_steps: int = 5,
    n_hits_per_rx: int = 2000,
    skip_range_bins: int = 5,
    metric: str = "cart_corr",
    verbose: bool = True,
) -> dict:
    """Grid search over small perturbations to maximize rendered↔GT RA correlation.

    Starting from the Stage 1 config, applies perturbations in range, azimuth,
    and rotation, re-renders single-chip ADC for each, and picks the parameters
    that maximize the chosen metric with the ground truth RA image.

    Uses RendererWrapper.update_antenna_config() to avoid rebuilding the mesh BVH
    for each grid point — only antenna positions/boresights change.

    Args:
        base_config: Stage 1 aligned SC config path
        gt_adc_path: GT single-chip ADC (.npy) path
        mesh_file: Scene mesh (.ply) path
        training_dir: Training output directory (for materials + render config)
        output_config_path: Where to save the best-aligned config
        range_search_m: ±range search in meters
        azimuth_search_deg: ±azimuth search in degrees
        rotation_search_deg: ±rotation search in degrees (around Z axis)
        n_steps: Grid steps per dimension (total evaluations = n_steps³)
        n_hits_per_rx: Hits per RX for rendering (lower = faster, noisier)
        skip_range_bins: Range bins to skip for TX-RX leakage
        metric: Optimization metric — "cart_corr" (cartesian RA) or "corr_dB" (polar dB)
        verbose: Print progress

    Returns:
        Dict with base_corr_dB, best_corr_dB, best_params, all_results, total_time_s
    """
    import io
    import contextlib
    import tempfile
    from mmir.preprocessing.alignment.cascaded_lidar import create_aligned_config
    from mmir.evaluation.renderer_wrapper import RendererWrapper

    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sc_pattern = os.path.join(_PROJECT_ROOT, 'assets', 'antenna_pattern', 'IWR1443', 'pattern_76.npy')

    # ---- Load and beamform GT ADC (once) ----
    gt_raw = np.load(gt_adc_path)
    gt_complex = gt_raw[0].transpose(1, 0, 2)  # (3, 4, 128) complex
    adc_gt = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
    ra_gt = adc_to_ra_image_single_chip(adc_gt)
    ra_gt_crop = ra_gt[:, skip_range_bins:]

    def _to_db(x, floor=-40.0):
        x_n = x / max(x.max(), 1e-30)
        return np.maximum(20.0 * np.log10(np.maximum(x_n, 1e-10)), floor)

    ra_gt_db = _to_db(ra_gt_crop)

    # Pre-compute GT cartesian RA for cart_corr metric
    use_cart_corr = (metric == "cart_corr")
    if use_cart_corr:
        from mmir.data.ra_utils import ra_polar_to_cartesian
        range_res = compute_range_res_from_cfg(base_config)
        ra_gt_cart = ra_polar_to_cartesian(ra_gt_crop, range_res)
        # Normalize GT cartesian for correlation
        gt_cart_flat = ra_gt_cart.flatten().astype(np.float64)
        gt_cart_flat_normed = gt_cart_flat / max(np.max(np.abs(gt_cart_flat)), 1e-30)

    # ---- Build renderer once ----
    if verbose:
        print("  Building renderer (one-time setup)...")
    wrapper = RendererWrapper.from_training_dir(
        mesh_file=mesh_file,
        config_file=base_config,
        training_dir=training_dir,
        tx_pattern_file=sc_pattern,
        rx_pattern_file=sc_pattern,
        verbose=verbose,
    )
    wrapper._render_config.n_hits_per_rx = n_hits_per_rx
    wrapper.renderer.config = wrapper._render_config
    wrapper.renderer.sampler.n_hits_per_rx = n_hits_per_rx
    wrapper.load_all_learned_params(training_dir)
    wrapper.verbose = False  # suppress per-render prints

    # ---- Generate search grid ----
    range_offsets = np.linspace(-range_search_m, range_search_m, n_steps)
    azimuth_offsets = np.linspace(-azimuth_search_deg, azimuth_search_deg, n_steps)
    rotation_offsets = np.linspace(-rotation_search_deg, rotation_search_deg, n_steps)

    total = len(range_offsets) * len(azimuth_offsets) * len(rotation_offsets)
    metric_label = "cart_corr" if use_cart_corr else "corr_dB"
    if verbose:
        print(f"\n  Grid search: {n_steps}³ = {total} evaluations (metric: {metric_label})")
        print(f"    Range:    ±{range_search_m}m")
        print(f"    Azimuth:  ±{azimuth_search_deg}°")
        print(f"    Rotation: ±{rotation_search_deg}°")
        print(f"    n_hits_per_rx: {n_hits_per_rx}")

    best_corr = -np.inf
    best_params = None
    all_results = []
    t_start = time.time()
    idx = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for dr_val in range_offsets:
            for da_val in azimuth_offsets:
                for drot_val in rotation_offsets:
                    idx += 1

                    # Create perturbed config from base
                    temp_path = os.path.join(tmpdir, f"cfg_{idx}.json")
                    with contextlib.redirect_stdout(io.StringIO()):
                        create_aligned_config(
                            base_config, dr_val, da_val, drot_val, 0.0,
                            output_path=temp_path,
                        )

                    # Update antenna positions (no mesh rebuild)
                    with open(temp_path) as f:
                        perturbed_cfg = json.load(f)
                    wrapper.update_antenna_config(perturbed_cfg)

                    # Render
                    adc = wrapper.render_forward(seed=42)

                    # Beamform → RA
                    ra = adc_to_ra_image_single_chip(adc)
                    ra_crop = ra[:, skip_range_bins:]

                    # Compute metrics
                    ra_db = _to_db(ra_crop)
                    corr_db, _ = pearsonr(ra_db.flatten(), ra_gt_db.flatten())

                    if use_cart_corr:
                        ra_cart = ra_polar_to_cartesian(ra_crop, range_res)
                        cart_flat = ra_cart.flatten().astype(np.float64)
                        cart_flat_normed = cart_flat / max(np.max(np.abs(cart_flat)), 1e-30)
                        cart_corr, _ = pearsonr(cart_flat_normed, gt_cart_flat_normed)
                        opt_metric = cart_corr
                    else:
                        cart_corr = None
                        opt_metric = corr_db

                    is_best = opt_metric > best_corr
                    entry = {
                        'delta_range_m': float(dr_val),
                        'delta_azimuth_deg': float(da_val),
                        'rotation_elev_deg': float(drot_val),
                        'corr_dB': float(corr_db),
                    }
                    if cart_corr is not None:
                        entry['cart_corr'] = float(cart_corr)
                    all_results.append(entry)

                    if is_best:
                        best_corr = opt_metric
                        best_params = entry.copy()

                    if verbose:
                        elapsed = time.time() - t_start
                        eta = elapsed / idx * (total - idx) if idx > 0 else 0
                        tag = " ***" if is_best else ""
                        m_val = f"cart_corr={cart_corr:.4f}" if use_cart_corr else f"corr_dB={corr_db:.4f}"
                        print(f"  [{idx:3d}/{total}] Δr={dr_val:+.3f}m Δaz={da_val:+.2f}° "
                              f"Δrot={drot_val:+.2f}° → {m_val}"
                              f"  ({elapsed:.0f}s, ETA {eta:.0f}s){tag}")

    # ---- Save best config ----
    is_identity = (
        best_params['delta_range_m'] == 0.0
        and best_params['delta_azimuth_deg'] == 0.0
        and best_params['rotation_elev_deg'] == 0.0
    )
    if is_identity:
        import shutil
        shutil.copy2(base_config, output_config_path)
    else:
        import io as _io
        import contextlib as _ctx
        with _ctx.redirect_stdout(_io.StringIO()):
            from mmir.preprocessing.alignment.cascaded_lidar import create_aligned_config as _cac
            _cac(
                base_config,
                best_params['delta_range_m'],
                best_params['delta_azimuth_deg'],
                best_params['rotation_elev_deg'],
                0.0,
                output_path=output_config_path,
            )

    wrapper.cleanup()

    total_time = time.time() - t_start

    # Find baseline (0,0,0) correlation
    base_corr_db = None
    base_metric = None
    for r in all_results:
        if (r['delta_range_m'] == 0.0
                and r['delta_azimuth_deg'] == 0.0
                and r['rotation_elev_deg'] == 0.0):
            base_corr_db = r['corr_dB']
            base_metric = r.get('cart_corr', r['corr_dB'])
            break

    if verbose:
        print(f"\n  Grid search complete in {total_time:.1f}s (metric: {metric_label})")
        if base_metric is not None:
            print(f"  Stage 1 baseline {metric_label}: {base_metric:.4f}")
        print(f"  Best {metric_label}: {best_corr:.4f}")
        if base_metric is not None:
            print(f"  Improvement: {best_corr - base_metric:+.4f}")
        print(f"  Best params: Δr={best_params['delta_range_m']:+.3f}m, "
              f"Δaz={best_params['delta_azimuth_deg']:+.2f}°, "
              f"Δrot={best_params['rotation_elev_deg']:+.2f}°")
        print(f"  Saved: {output_config_path}")

    return {
        'base_corr_dB': base_corr_db,
        'best_corr_dB': float(best_corr),
        'best_params': best_params,
        'all_results': all_results,
        'total_time_s': total_time,
        'output_config_path': output_config_path,
    }


# ---------------------------------------------------------------------------
# Multi-resolution grid search
# ---------------------------------------------------------------------------

def multi_resolution_grid_search(
    base_config: str,
    gt_adc_path: str,
    mesh_file: str,
    training_dir: str,
    output_config_path: str,
    # Coarse pass parameters
    coarse_range_m: float = 2.0,
    coarse_azimuth_deg: float = 10.0,
    coarse_rotation_deg: float = 5.0,
    coarse_steps: int = 11,
    # Fine pass parameters
    fine_range_m: float = 0.4,
    fine_azimuth_deg: float = 2.0,
    fine_rotation_deg: float = 1.0,
    fine_steps: int = 9,
    n_hits_per_rx: int = 2000,
    skip_range_bins: int = 5,
    metric: str = "cart_corr",
    verbose: bool = True,
) -> dict:
    """Two-pass grid search: coarse sweep over wide range, then fine refinement.

    Pass 1 (coarse): Wide range with moderate resolution to find approximate optimum.
    Pass 2 (fine): Narrow range centered on Pass 1 best for precise alignment.

    Total evaluations: coarse_steps³ + fine_steps³ (e.g. 11³ + 9³ = 1331 + 729 = 2060).

    Args:
        base_config: Stage 1 aligned SC config path (starting point)
        gt_adc_path: GT single-chip ADC (.npy) path
        mesh_file: Scene mesh (.ply) path
        training_dir: Training output directory (for materials + render config)
        output_config_path: Where to save the best-aligned config
        coarse_range_m: ±range for coarse pass (meters)
        coarse_azimuth_deg: ±azimuth for coarse pass (degrees)
        coarse_rotation_deg: ±rotation for coarse pass (degrees)
        coarse_steps: Steps per dimension for coarse pass
        fine_range_m: ±range for fine pass (meters)
        fine_azimuth_deg: ±azimuth for fine pass (degrees)
        fine_rotation_deg: ±rotation for fine pass (degrees)
        fine_steps: Steps per dimension for fine pass
        n_hits_per_rx: Hits per RX for rendering
        skip_range_bins: Range bins to skip for TX-RX leakage
        metric: Optimization metric — "cart_corr" (cartesian RA) or "corr_dB" (polar dB)
        verbose: Print progress

    Returns:
        Dict with coarse_result, fine_result, best_corr_dB, best_params, total_time_s
    """
    import tempfile
    t_start = time.time()

    if verbose:
        total_evals = coarse_steps ** 3 + fine_steps ** 3
        print(f"\n  Multi-resolution alignment: {total_evals} total evaluations")
        print(f"    Pass 1 (coarse): {coarse_steps}³ = {coarse_steps**3} evals, "
              f"±{coarse_range_m}m / ±{coarse_azimuth_deg}° / ±{coarse_rotation_deg}°")
        print(f"    Pass 2 (fine):   {fine_steps}³ = {fine_steps**3} evals, "
              f"±{fine_range_m}m / ±{fine_azimuth_deg}° / ±{fine_rotation_deg}°")

    # --- Pass 1: Coarse sweep ---
    with tempfile.TemporaryDirectory() as tmpdir:
        coarse_output = os.path.join(tmpdir, "coarse_best.json")

        if verbose:
            print(f"\n  === Pass 1: Coarse Sweep ===")

        coarse_result = render_alignment_grid_search(
            base_config=base_config,
            gt_adc_path=gt_adc_path,
            mesh_file=mesh_file,
            training_dir=training_dir,
            output_config_path=coarse_output,
            range_search_m=coarse_range_m,
            azimuth_search_deg=coarse_azimuth_deg,
            rotation_search_deg=coarse_rotation_deg,
            n_steps=coarse_steps,
            n_hits_per_rx=n_hits_per_rx,
            skip_range_bins=skip_range_bins,
            metric=metric,
            verbose=verbose,
        )

        if verbose:
            bp = coarse_result['best_params']
            print(f"\n  Coarse best: {metric}={coarse_result['best_corr_dB']:.4f}")
            print(f"    Δr={bp['delta_range_m']:+.3f}m, "
                  f"Δaz={bp['delta_azimuth_deg']:+.2f}°, "
                  f"Δrot={bp['rotation_elev_deg']:+.2f}°")

        # --- Pass 2: Fine refinement ---
        if verbose:
            print(f"\n  === Pass 2: Fine Refinement ===")

        fine_result = render_alignment_grid_search(
            base_config=coarse_output,
            gt_adc_path=gt_adc_path,
            mesh_file=mesh_file,
            training_dir=training_dir,
            output_config_path=output_config_path,
            range_search_m=fine_range_m,
            azimuth_search_deg=fine_azimuth_deg,
            rotation_search_deg=fine_rotation_deg,
            n_steps=fine_steps,
            n_hits_per_rx=n_hits_per_rx,
            skip_range_bins=skip_range_bins,
            metric=metric,
            verbose=verbose,
        )

    total_time = time.time() - t_start

    if verbose:
        print(f"\n  Multi-resolution complete in {total_time:.1f}s")
        print(f"  Coarse best: {coarse_result['best_corr_dB']:.4f}")
        print(f"  Fine best:   {fine_result['best_corr_dB']:.4f}")
        if coarse_result.get('base_corr_dB') is not None:
            print(f"  Total improvement: {fine_result['best_corr_dB'] - coarse_result['base_corr_dB']:+.4f}")
        print(f"  Saved: {output_config_path}")

    return {
        'coarse_result': coarse_result,
        'fine_result': fine_result,
        'best_corr_dB': fine_result['best_corr_dB'],
        'best_params': fine_result['best_params'],
        'total_time_s': total_time,
        'output_config_path': output_config_path,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_figures(
    casc_ra: np.ndarray,
    sc_ra: np.ndarray,
    casc_resampled: np.ndarray,
    shift_az: int,
    shift_r: int,
    output_dir: str,
    skip_range_bins: int = 5,
):
    """Generate diagnostic figures for the alignment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    def _to_db(x, floor=-40):
        x_n = x / max(x.max(), 1e-30)
        return np.maximum(20 * np.log10(np.maximum(x_n, 1e-10)), floor)

    # 1. Side-by-side: cascaded RA, SC RA, resampled cascaded
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(_to_db(casc_ra), aspect='auto', origin='lower')
    axes[0].set_title(f'Cascaded GT RA {casc_ra.shape}')
    axes[0].set_xlabel('Range bin'); axes[0].set_ylabel('Azimuth bin')

    axes[1].imshow(_to_db(sc_ra), aspect='auto', origin='lower')
    axes[1].set_title(f'Single-Chip GT RA {sc_ra.shape}')
    axes[1].set_xlabel('Range bin')

    axes[2].imshow(_to_db(casc_resampled), aspect='auto', origin='lower')
    axes[2].set_title(f'Cascaded Resampled to SC grid {casc_resampled.shape}')
    axes[2].set_xlabel('Range bin')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ra_comparison.png'), dpi=150)
    plt.close()

    # 2. Overlay after shift
    casc_crop = casc_resampled[:, skip_range_bins:]
    sc_crop = sc_ra[:, skip_range_bins:]
    shifted = np.roll(np.roll(casc_crop, shift_az, axis=0), shift_r, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(_to_db(sc_crop), aspect='auto', origin='lower')
    axes[0].set_title('SC GT RA (cropped)')

    axes[1].imshow(_to_db(shifted), aspect='auto', origin='lower')
    axes[1].set_title(f'Cascaded shifted (az={shift_az}, r={shift_r})')

    # Overlay
    sc_db = _to_db(sc_crop)
    casc_db = _to_db(shifted)
    sc_n = (sc_db - sc_db.min()) / max(sc_db.max() - sc_db.min(), 1e-10)
    casc_n = (casc_db - casc_db.min()) / max(casc_db.max() - casc_db.min(), 1e-10)
    overlay = np.stack([sc_n, casc_n, np.zeros_like(sc_n)], axis=-1)
    axes[2].imshow(overlay, aspect='auto', origin='lower')
    axes[2].set_title('Overlay (red=SC, green=Cascaded)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ra_overlay.png'), dpi=150)
    plt.close()

    print(f"  Figures saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cascaded-Mediated Single-Chip Radar Alignment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--scene-dir', required=True,
                        help='Scene directory (e.g. data/seq_1_frame_185)')
    parser.add_argument('--cascaded-frame', type=int, required=True,
                        help='Cascaded frame number (e.g. 185)')
    parser.add_argument('--sc-frame', type=int, required=True,
                        help='Single-chip frame number (e.g. 369)')
    parser.add_argument('--casc-aligned-suffix', default='_aligned_gpu',
                        help='Suffix of the aligned cascaded config')
    parser.add_argument('--run-stage2', action='store_true',
                        help='Run Stage 2 render grid search refinement')
    parser.add_argument('--skip-range-bins', type=int, default=5,
                        help='Range bins to skip for TX-RX leakage')
    parser.add_argument('--grid-range-m', type=float, default=0.5,
                        help='Grid search: ±range in meters')
    parser.add_argument('--grid-azimuth-deg', type=float, default=3.0,
                        help='Grid search: ±azimuth in degrees')
    parser.add_argument('--grid-rotation-deg', type=float, default=2.0,
                        help='Grid search: ±rotation in degrees')
    parser.add_argument('--grid-steps', type=int, default=5,
                        help='Grid search: steps per dimension (total = steps³)')
    parser.add_argument('--grid-n-hits', type=int, default=2000,
                        help='Grid search: n_hits_per_rx for rendering')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results')
    parser.add_argument('--viz', action='store_true',
                        help='Generate diagnostic figures')

    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    configs_dir = scene_dir / 'configs'
    radar_dir = scene_dir / 'radar'
    cf = args.cascaded_frame
    sf = args.sc_frame

    # Discover files
    casc_orig = str(configs_dir / f'cascaded_frame_{cf}.json')
    casc_aligned = str(configs_dir / f'cascaded_frame_{cf}{args.casc_aligned_suffix}.json')
    sc_orig = str(configs_dir / f'single_chip_frame_{sf}.json')
    casc_gt_adc = str(radar_dir / f'cascaded_frame_{cf}.npy')
    sc_gt_adc = str(radar_dir / f'single_chip_frame_{sf}.npy')

    # Validate
    for p, label in [
        (casc_orig, 'Cascaded original config'),
        (casc_aligned, 'Cascaded aligned config'),
        (sc_orig, 'Single-chip original config'),
        (casc_gt_adc, 'Cascaded GT ADC'),
        (sc_gt_adc, 'Single-chip GT ADC'),
    ]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    if args.output_dir is None:
        args.output_dir = f'output/sc_cascaded_alignment/{scene_dir.name}'
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('Cascaded-Mediated Single-Chip Radar Alignment')
    print('=' * 70)
    print(f'  Scene: {scene_dir.name}')
    print(f'  Cascaded frame: {cf}')
    print(f'  Single-chip frame: {sf}')

    # ---- Stage 1: Transfer cascaded alignment ----
    print(f'\n--- Stage 1: Transfer Cascaded Alignment ---')

    # Print board info before transform
    _, _, casc_center_orig, casc_bore_orig = load_config_positions_and_boresight(casc_orig)
    _, _, casc_center_aligned, casc_bore_aligned = load_config_positions_and_boresight(casc_aligned)
    _, _, sc_center_orig, sc_bore_orig = load_config_positions_and_boresight(sc_orig)

    print(f'  Cascaded original center:  [{casc_center_orig[0]:.4f}, {casc_center_orig[1]:.4f}, {casc_center_orig[2]:.4f}]')
    print(f'  Cascaded aligned center:   [{casc_center_aligned[0]:.4f}, {casc_center_aligned[1]:.4f}, {casc_center_aligned[2]:.4f}]')
    print(f'  SC original center:        [{sc_center_orig[0]:.4f}, {sc_center_orig[1]:.4f}, {sc_center_orig[2]:.4f}]')

    casc_to_sc_dist_orig = np.linalg.norm(sc_center_orig - casc_center_orig) * 1000
    print(f'  Cascaded-SC distance (original): {casc_to_sc_dist_orig:.1f} mm')

    R, t = compute_board_transform(casc_orig, casc_aligned)
    print(f'  Rotation angle: {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.3f}°')
    print(f'  Translation: [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm')

    sc_aligned_path = str(
        configs_dir / f'single_chip_frame_{sf}_cascaded_aligned.json'
    )
    transfer_alignment_to_sc(casc_orig, casc_aligned, sc_orig, sc_aligned_path)

    # Verify
    _, _, sc_center_aligned, sc_bore_aligned = load_config_positions_and_boresight(sc_aligned_path)
    casc_to_sc_dist_aligned = np.linalg.norm(sc_center_aligned - casc_center_aligned) * 1000
    print(f'  SC aligned center:         [{sc_center_aligned[0]:.4f}, {sc_center_aligned[1]:.4f}, {sc_center_aligned[2]:.4f}]')
    print(f'  Cascaded-SC distance (aligned): {casc_to_sc_dist_aligned:.1f} mm')
    print(f'  SC boresight change: orig=({sc_bore_orig[0]:.4f},{sc_bore_orig[1]:.4f},{sc_bore_orig[2]:.4f})'
          f' → aligned=({sc_bore_aligned[0]:.4f},{sc_bore_aligned[1]:.4f},{sc_bore_aligned[2]:.4f})')
    print(f'  Stage 1 config saved: {sc_aligned_path}')

    # ---- Stage 2: Render grid search refinement (opt-in) ----
    if not args.run_stage2:
        final_config = sc_aligned_path
    else:
        print(f'\n--- Stage 2: Render Grid Search Refinement ---')
        from mmir.evaluation.scene_registry import get_scene
        scene = get_scene(scene_dir.name)
        if scene is None or not scene.mesh_file or not scene.our_training_dir:
            print("ERROR: Could not find mesh or training dir via scene registry")
            sys.exit(1)

        stage2_output = str(
            configs_dir / f'single_chip_frame_{sf}_grid_aligned.json'
        )
        stage2_result = render_alignment_grid_search(
            base_config=sc_aligned_path,
            gt_adc_path=sc_gt_adc,
            mesh_file=scene.mesh_file,
            training_dir=scene.our_training_dir,
            output_config_path=stage2_output,
            range_search_m=args.grid_range_m,
            azimuth_search_deg=args.grid_azimuth_deg,
            rotation_search_deg=args.grid_rotation_deg,
            n_steps=args.grid_steps,
            n_hits_per_rx=args.grid_n_hits,
            skip_range_bins=args.skip_range_bins,
        )
        final_config = stage2_output

        # Save results JSON
        results_path = os.path.join(args.output_dir, 'grid_search_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'scene': scene_dir.name,
                'cascaded_frame': cf,
                'sc_frame': sf,
                'stage1_config': sc_aligned_path,
                'stage2_result': stage2_result,
                'final_config': final_config,
            }, f, indent=2, default=str)
        print(f'  Results saved: {results_path}')

    print(f'\n  Final aligned config: {final_config}')
    print('=' * 70)


if __name__ == '__main__':
    main()
