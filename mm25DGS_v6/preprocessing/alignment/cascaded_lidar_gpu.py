#!/usr/bin/env python3
"""
GPU-Accelerated LiDAR-Radar RA Map Alignment Tool

This is the GPU-accelerated version of lidar_radar_ra_alignment.py.
Uses CuPy for GPU array operations to achieve ~100-200x speedup on coarse grid search.

Key optimizations:
1. Point cloud kept on GPU for all evaluations
2. GPU-accelerated voxelization and histogram binning
3. Batched evaluation for coarse grid search
4. GPU metric computation

Usage:
    python lidar_radar_ra_alignment_GPU.py \
        --config-path /path/to/config.json \
        --lidar-path /path/to/lidar.ply \
        --gt-adc-path /path/to/radar_gt.npy \
        --viz

Requirements:
    - CuPy (pip install cupy-cuda11x or cupy-cuda12x depending on CUDA version)
    - NVIDIA GPU with CUDA support
"""

# Set Mitsuba variant before any mmir imports
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy.optimize import minimize

# Check for CuPy availability early
try:
    import cupy as cp
    HAS_CUPY = True
    _dev = cp.cuda.Device()
    _mem = _dev.mem_info
    print(f"[GPU] CuPy available. Using GPU ID: {_dev.id}")
    print(f"[GPU] GPU Memory: {_mem[1] / 1e9:.1f} GB total, {_mem[0] / 1e9:.1f} GB free")
except ImportError:
    HAS_CUPY = False
    cp = None
    print("[WARNING] CuPy not available. Falling back to CPU implementation.")
    print("         Install CuPy: pip install cupy-cuda11x (or cupy-cuda12x)")

import open3d as o3d

# Import from CPU version (reuse non-GPU-specific functions)
from .cascaded_lidar import (
    load_point_cloud,
    load_radar_config,
    radar_gt_to_ra_map,
    apply_rotation_around_elevation_axis,
    apply_rotation_around_azimuth_axis,
    apply_translation,
    compute_alignment_metrics,
    print_results,
    save_results_json,
    create_aligned_config,
    generate_visualizations,
    get_ra_map_for_params,
    load_antenna_pattern,
    interpolate_antenna_gain,
)

# Import from single_view_proc for grid generation
from mmir.evaluation.utils.single_view_proc import (
    make_angle_grids_np,
    _centers_to_edges,
)

from mmir.losses.loss_utils import minmax_normalize_numpy

# GPU utilities
if HAS_CUPY:
    from .gpu_utils.voxelize_gpu import (
        voxelize_lidar_gpu,
        rae_to_ra_map_gpu,
    )
    from .gpu_utils.metrics_gpu import (
        minmax_normalize_gpu,
        compute_alignment_metrics_gpu,
    )
    from .gpu_utils.batch_eval_gpu import (
        coarse_grid_search_gpu,
        apply_transformations_to_boresight_origin,
    )


def prepare_gpu_data(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    antenna_pattern_path: str = None
) -> Tuple:
    """
    Transfer data to GPU and prepare grid edges.

    Returns:
        pts_gpu: Point cloud positions on GPU
        intensities_gpu: Point intensities on GPU
        ra_radar_gpu: Normalized radar RA map on GPU
        az_edges_gpu: Azimuth bin edges on GPU
        el_edges_gpu: Elevation bin edges on GPU
        r_edges_gpu: Range bin edges on GPU
        antenna_weights_gpu: Antenna pattern weights on GPU
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy is required for GPU acceleration")

    # Get points and intensities
    pts = np.asarray(lidar_pcd.points, dtype=np.float32)

    # Get intensities from colors (grayscale)
    if lidar_pcd.has_colors():
        colors = np.asarray(lidar_pcd.colors, dtype=np.float32)
        intensities = colors[:, 0]  # R channel (grayscale)
    else:
        intensities = np.ones(pts.shape[0], dtype=np.float32)

    # Transfer to GPU
    pts_gpu = cp.asarray(pts)
    intensities_gpu = cp.asarray(intensities)

    # Normalize radar RA and transfer
    ra_radar_norm = minmax_normalize_numpy(ra_radar).astype(np.float32)
    ra_radar_gpu = cp.asarray(ra_radar_norm)

    # Generate grid edges
    fft_size_az = int(params['num_az_bins']) + 1
    fft_size_el = int(params['num_el_bins']) + 1
    az_cent, el_cent = make_angle_grids_np(fft_size_az, fft_size_el)
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))

    az_edges_gpu = cp.asarray(az_edges)
    el_edges_gpu = cp.asarray(el_edges)
    r_edges_gpu = cp.asarray(r_edges)

    # Compute antenna weights
    angles_deg, az_gain_db, el_gain_db = load_antenna_pattern(antenna_pattern_path)

    az_deg = np.degrees(az_cent)
    el_deg = np.degrees(el_cent)

    az_gain_interp = interpolate_antenna_gain(angles_deg, az_gain_db, az_deg)
    el_gain_interp = interpolate_antenna_gain(angles_deg, el_gain_db, el_deg)

    az_gain_linear = 10 ** (az_gain_interp / 10)
    el_gain_linear = 10 ** (el_gain_interp / 10)

    weights = np.outer(az_gain_linear, el_gain_linear)
    weights = (weights / weights.max()).astype(np.float32)

    antenna_weights_gpu = cp.asarray(weights)

    return (pts_gpu, intensities_gpu, ra_radar_gpu,
            az_edges_gpu, el_edges_gpu, r_edges_gpu, antenna_weights_gpu)


def evaluate_alignment_gpu(
    pts_gpu,
    intensities_gpu,
    ra_radar_gpu,
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    az_edges_gpu,
    el_edges_gpu,
    r_edges_gpu,
    antenna_weights_gpu,
    near_field_m: float = 1.5,
) -> Dict[str, float]:
    """
    Evaluate alignment for a single parameter set on GPU.

    Used for fine optimization stage.
    """
    # Compute transformed boresight and origin
    boresight, origin = apply_transformations_to_boresight_origin(
        base_boresight, base_origin,
        delta_range_m, delta_azimuth_deg,
        rotation_elev_deg, rotation_azim_deg
    )

    boresight_gpu = cp.asarray(boresight, dtype=cp.float32)
    origin_gpu = cp.asarray(origin, dtype=cp.float32)

    # Voxelize
    rae_tensor = voxelize_lidar_gpu(
        pts_gpu, intensities_gpu,
        origin_gpu, boresight_gpu,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m
    )

    # Collapse to RA
    ra_lidar = rae_to_ra_map_gpu(rae_tensor)

    # Handle shape mismatch
    if ra_lidar.shape[0] == 128 and ra_radar_gpu.shape[0] == 127:
        ra_lidar = ra_lidar[1:, :]

    # Normalize
    ra_lidar_norm = minmax_normalize_gpu(ra_lidar)

    # Compute metrics
    return compute_alignment_metrics_gpu(ra_lidar_norm, ra_radar_gpu)


def _prior_penalty_4dof(dr_m: float, da_deg: float, de_deg: float, dz_deg: float,
                        prior_weight: float) -> float:
    """Prior penalty matching the renderer path.
        penalty = w × (dr² + (da/10)² + de² + dz²)
    mm / degrees. Scales are chosen so a ~0.1 m range-offset and a ~1 deg
    angle offset contribute similarly. Call sites subtract this from cc:
        adjusted_score = cc - penalty.
    """
    if not prior_weight:
        return 0.0
    return float(prior_weight) * (
        dr_m * dr_m
        + (da_deg / 10.0) ** 2
        + de_deg * de_deg
        + dz_deg * dz_deg
    )


_DEFAULT_TX_PATTERN = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'assets', 'antenna_pattern', 'MMWCAS', 'tx1_76.npy'))


def optimize_alignment_gpu_with_prior(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    prior_boresight: np.ndarray,
    prior_origin: np.ndarray,
    search_radius: Optional[Dict[str, float]] = None,
    prior_weight: float = 0.01,
    antenna_pattern_path: Optional[str] = None,
    metric: str = 'correlation',
    coarse_steps: int = 7,
    near_field_m: float = 1.5,
    batch_size: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """GPU LiDAR-RA alignment with a trajectory-prior.

    Same pipeline as ``optimize_alignment_gpu`` (coarse grid + Nelder-Mead
    refinement) but centred on the Stage-A smoothed pose and penalised for
    drifting away from it.

    Arguments:
        prior_boresight, prior_origin: the Stage-A smoothed pose. The
            delta=(0,0,0,0) point *IS* this pose, so all grid candidates
            are evaluated as small deltas from it.
        search_radius: dict with keys ``range_m`` / ``azim_deg`` /
            ``elev_rot_deg`` / ``azim_rot_deg``; each DOF is searched in
            ``[-radius, +radius]``. Defaults to ±0.4 m / ±2° / ±3° / ±3°,
            matching the renderer path.
        prior_weight: λ in the penalty ``λ·(dr² + (da/10)² + de² + dz²)``.
            Subtracted from cc in both the coarse grid (post-hoc rescore
            of ``coarse_grid_search_gpu``'s output, no kernel changes) and
            the Nelder-Mead refinement objective.

    Returns a dict with the same schema as ``optimize_alignment_gpu``
    plus ``'adjusted_score'`` (cc − penalty) for the winner.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU optimization")

    if search_radius is None:
        search_radius = {
            'range_m': 0.4, 'azim_deg': 2.0,
            'elev_rot_deg': 3.0, 'azim_rot_deg': 3.0,
        }

    range_search_m = (-search_radius['range_m'], search_radius['range_m'])
    azimuth_search_deg = (-search_radius['azim_deg'], search_radius['azim_deg'])
    rotation_elev_search_deg = (-search_radius['elev_rot_deg'], search_radius['elev_rot_deg'])
    rotation_azim_search_deg = (-search_radius['azim_rot_deg'], search_radius['azim_rot_deg'])

    if verbose:
        print("\n[GPU prior] Preparing data for GPU...")
        t0 = time.time()

    _pattern_path = antenna_pattern_path or _DEFAULT_TX_PATTERN
    (pts_gpu, intensities_gpu, ra_radar_gpu,
     az_edges_gpu, el_edges_gpu, r_edges_gpu,
     antenna_weights_gpu) = prepare_gpu_data(lidar_pcd, ra_radar, params,
                                             antenna_pattern_path=_pattern_path)

    if verbose:
        print(f"[GPU prior] Data transfer complete in {time.time() - t0:.2f}s")
        print(f"[GPU prior] Points on GPU: {pts_gpu.shape[0]:,}")

    # -------- Phase 1: coarse grid search (unchanged kernels) -------------
    t1 = time.time()
    grid_best_params, grid_best_cc, grid_results = coarse_grid_search_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        prior_boresight, prior_origin,
        range_search_m, azimuth_search_deg,
        rotation_elev_search_deg, rotation_azim_search_deg,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m,
        metric, coarse_steps, batch_size, verbose=False,
    )
    if verbose:
        print(f"[GPU prior] Phase 1 (grid) in {time.time() - t1:.2f}s; "
              f"raw best {metric}={grid_best_cc:.4f}")

    # Re-rank grid_results with prior penalty. all_results columns:
    # [dr, da, re, ra, metric]
    maximize = metric in ('correlation', 'ssim', 'iou')
    dr_col = grid_results[:, 0]
    da_col = grid_results[:, 1]
    de_col = grid_results[:, 2]
    dz_col = grid_results[:, 3]
    cc_col = grid_results[:, 4]
    if prior_weight:
        penalty_col = prior_weight * (
            dr_col * dr_col
            + (da_col / 10.0) ** 2
            + de_col * de_col
            + dz_col * dz_col
        )
        adj_col = cc_col - penalty_col if maximize else cc_col + penalty_col
    else:
        adj_col = cc_col

    pick_idx = int(np.argmax(adj_col) if maximize else np.argmin(adj_col))
    best_params = grid_results[pick_idx, :4].copy()
    best_cc_pre_refine = float(cc_col[pick_idx])
    best_adj_pre_refine = float(adj_col[pick_idx])
    if verbose:
        print(f"[GPU prior] Prior-rescored grid pick: cc={best_cc_pre_refine:.4f} "
              f"adj={best_adj_pre_refine:.4f} "
              f"dr={best_params[0]:.3f}m da={best_params[1]:.2f}° "
              f"de={best_params[2]:.2f}° dz={best_params[3]:.2f}°")

    # -------- Phase 2: Nelder-Mead with penalty in objective --------------
    t2 = time.time()

    def objective(x):
        metrics = evaluate_alignment_gpu(
            pts_gpu, intensities_gpu, ra_radar_gpu,
            prior_boresight, prior_origin,
            float(x[0]), float(x[1]), float(x[2]), float(x[3]),
            az_edges_gpu, el_edges_gpu, r_edges_gpu,
            antenna_weights_gpu, near_field_m,
        )
        cc = float(metrics[metric])
        penalty = _prior_penalty_4dof(x[0], x[1], x[2], x[3], prior_weight)
        adj = cc - penalty if maximize else cc + penalty
        return -adj if maximize else adj

    result = minimize(
        objective, x0=best_params, method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001},
    )
    if verbose:
        print(f"[GPU prior] Phase 2 (refine) in {time.time() - t2:.2f}s")

    opt = {
        'delta_range_m':       float(result.x[0]),
        'delta_azimuth_deg':   float(result.x[1]),
        'rotation_elev_deg':   float(result.x[2]),
        'rotation_azim_deg':   float(result.x[3]),
    }

    # Evaluate the final pose raw (without penalty) + initial (prior-only)
    best_metrics = evaluate_alignment_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        prior_boresight, prior_origin,
        opt['delta_range_m'], opt['delta_azimuth_deg'],
        opt['rotation_elev_deg'], opt['rotation_azim_deg'],
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m,
    )
    initial_metrics = evaluate_alignment_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        prior_boresight, prior_origin,
        0.0, 0.0, 0.0, 0.0,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m,
    )

    adj_final = best_metrics[metric] - _prior_penalty_4dof(
        opt['delta_range_m'], opt['delta_azimuth_deg'],
        opt['rotation_elev_deg'], opt['rotation_azim_deg'],
        prior_weight)

    cp.cuda.Stream.null.synchronize()

    return {
        'optimal_params':  opt,
        'best_metrics':    best_metrics,
        'initial_metrics': initial_metrics,
        'adjusted_score':  float(adj_final),
        'prior_weight':    float(prior_weight),
        'search_radius':   search_radius,
        'grid_raw_best_cc':  float(grid_best_cc),
        'grid_adj_best_cc':  float(best_cc_pre_refine),
    }


def optimize_alignment_gpu(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    range_search_m: Tuple[float, float],
    azimuth_search_deg: Tuple[float, float],
    rotation_elev_search_deg: Tuple[float, float],
    rotation_azim_search_deg: Tuple[float, float],
    metric: str = 'correlation',
    coarse_steps: int = 7,
    near_field_m: float = 1.5,
    batch_size: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    GPU-accelerated coarse-to-fine alignment optimization.

    Phase 1: GPU-accelerated grid search (batched)
    Phase 2: Fine optimization with L-BFGS-B (single GPU evaluations)
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU optimization")

    # Prepare GPU data
    if verbose:
        print("\n[GPU] Preparing data for GPU...")
        t0 = time.time()

    (pts_gpu, intensities_gpu, ra_radar_gpu,
     az_edges_gpu, el_edges_gpu, r_edges_gpu,
     antenna_weights_gpu) = prepare_gpu_data(lidar_pcd, ra_radar, params)

    if verbose:
        print(f"[GPU] Data transfer complete in {time.time() - t0:.2f}s")
        print(f"[GPU] Points on GPU: {pts_gpu.shape[0]:,}")
        mem_used = cp.cuda.Device().mem_info
        print(f"[GPU] Memory used: {(mem_used[1] - mem_used[0]) / 1e9:.2f} GB")

    # Phase 1: GPU coarse grid search
    t1 = time.time()

    best_params, best_value, grid_results = coarse_grid_search_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        base_boresight, base_origin,
        range_search_m, azimuth_search_deg,
        rotation_elev_search_deg, rotation_azim_search_deg,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m,
        metric, coarse_steps, batch_size, verbose
    )

    if verbose:
        print(f"[GPU] Phase 1 complete in {time.time() - t1:.2f}s")

    # Phase 2: Fine optimization (unconstrained, matching original)
    maximize = metric in ['correlation', 'ssim', 'iou']

    if verbose:
        print("Phase 2: Fine optimization (Nelder-Mead, unconstrained)...")

    t2 = time.time()

    def objective(x):
        metrics = evaluate_alignment_gpu(
            pts_gpu, intensities_gpu, ra_radar_gpu,
            base_boresight, base_origin,
            x[0], x[1], x[2], x[3],
            az_edges_gpu, el_edges_gpu, r_edges_gpu,
            antenna_weights_gpu, near_field_m
        )
        value = metrics[metric]
        return -value if maximize else value

    # Use unconstrained Nelder-Mead (matching original lidar_radar_ra_alignment.py)
    result = minimize(
        objective,
        x0=best_params,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001}
    )

    if verbose:
        print(f"[GPU] Phase 2 complete in {time.time() - t2:.2f}s")

    optimal_params = {
        'delta_range_m': float(result.x[0]),
        'delta_azimuth_deg': float(result.x[1]),
        'rotation_elev_deg': float(result.x[2]),
        'rotation_azim_deg': float(result.x[3])
    }

    # Compute final metrics
    best_metrics = evaluate_alignment_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        base_boresight, base_origin,
        optimal_params['delta_range_m'],
        optimal_params['delta_azimuth_deg'],
        optimal_params['rotation_elev_deg'],
        optimal_params['rotation_azim_deg'],
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m
    )

    # Compute initial metrics
    if verbose:
        print("\n" + "=" * 60)
        print("DEBUG: Initial evaluation (no transformation)")
        print("=" * 60)

    initial_metrics = evaluate_alignment_gpu(
        pts_gpu, intensities_gpu, ra_radar_gpu,
        base_boresight, base_origin,
        0.0, 0.0, 0.0, 0.0,
        az_edges_gpu, el_edges_gpu, r_edges_gpu,
        antenna_weights_gpu, near_field_m
    )

    if verbose:
        print(f"  Optimization complete. Final {metric}: {best_metrics[metric]:.4f}")

    # Synchronize GPU before returning
    cp.cuda.Stream.null.synchronize()

    return {
        'optimal_params': optimal_params,
        'best_metrics': best_metrics,
        'initial_metrics': initial_metrics,
        'grid_search_results': grid_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated LiDAR-Radar RA Map Alignment Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config-path', required=True, help='Radar config JSON')
    parser.add_argument('--lidar-path', required=True, help='LiDAR point cloud (PLY or NPY)')
    parser.add_argument('--gt-adc-path', required=True, help='Radar ground truth ADC file (.npy)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--viz', action='store_true', help='Generate visualizations')
    parser.add_argument('--range-search-m', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='Range translation search limits (min max) in meters')
    parser.add_argument('--azimuth-search-deg', type=float, nargs=2, default=[-15.0, 15.0],
                        metavar=('MIN', 'MAX'),
                        help='Azimuth translation search limits in degrees')
    parser.add_argument('--rotation-elev-search-deg', type=float, nargs=2, default=[-10.0, 10.0],
                        metavar=('MIN', 'MAX'),
                        help='Rotation around elevation axis search limits in degrees')
    parser.add_argument('--rotation-azim-search-deg', type=float, nargs=2, default=[-10.0, 10.0],
                        metavar=('MIN', 'MAX'),
                        help='Rotation around azimuth axis search limits in degrees')
    parser.add_argument('--metric', default='correlation',
                        choices=['correlation', 'mse', 'ssim', 'iou'],
                        help='Metric to optimize')
    parser.add_argument('--coarse-steps', type=int, default=5,
                        help='Grid search steps per dimension')
    parser.add_argument('--near-field-m', type=float, default=1.5,
                        help='Near-field threshold in meters')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for GPU grid search')
    parser.add_argument('--save-aligned-config', action='store_true', default=True,
                        help='Save aligned config file (default: True)')
    parser.add_argument('--no-aligned-config', action='store_false', dest='save_aligned_config',
                        help='Do not save aligned config file')
    parser.add_argument('--aligned-config-suffix', type=str, default='_aligned',
                        help='Suffix for aligned config filename (default: _aligned)')
    parser.add_argument('--cpu-fallback', action='store_true',
                        help='Fall back to CPU if GPU unavailable')

    args = parser.parse_args()

    # Check GPU availability
    if not HAS_CUPY and not args.cpu_fallback:
        print("ERROR: CuPy not available and --cpu-fallback not specified.")
        print("       Install CuPy or use --cpu-fallback to use CPU version.")
        sys.exit(1)

    # Setup output directory
    if args.output_dir is None:
        config_stem = Path(args.config_path).stem
        args.output_dir = f"output/lidar_radar_alignment_gpu_{config_stem}"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("GPU-Accelerated LiDAR-Radar RA Map Alignment Tool")
    print("=" * 70)

    total_start = time.time()

    # 1. Load config
    print("\n[Step 1] Loading radar configuration...")
    params, base_origin, base_boresight = load_radar_config(args.config_path)
    print(f"  Grid: {params['num_az_bins']}x{params['num_el_bins']}x{params['num_adc']}")
    print(f"  Range resolution: {params['range_resolution']:.4f} m")
    print(f"  Origin: [{base_origin[0]:.3f}, {base_origin[1]:.3f}, {base_origin[2]:.3f}]")
    print(f"  Boresight: [{base_boresight[0]:.3f}, {base_boresight[1]:.3f}, {base_boresight[2]:.3f}]")

    # 2. Load LiDAR
    print("\n[Step 2] Loading LiDAR point cloud...")
    lidar_pcd = load_point_cloud(args.lidar_path)
    print(f"  Points: {len(np.asarray(lidar_pcd.points)):,}")

    # 3. Load and process radar GT
    print("\n[Step 3] Processing radar ground truth to RA map...")
    ra_radar = radar_gt_to_ra_map(args.gt_adc_path)
    print(f"  Radar RA shape: {ra_radar.shape}")

    # 4. Compute auto search ranges if not specified
    if args.range_search_m is None:
        max_range = params['num_adc'] * params['range_resolution']
        args.range_search_m = [-max_range * 0.1, max_range * 0.1]

    print(f"\n  Search ranges:")
    print(f"    Range: {args.range_search_m[0]:.2f} to {args.range_search_m[1]:.2f} m")
    print(f"    Azimuth: {args.azimuth_search_deg[0]:.1f} to {args.azimuth_search_deg[1]:.1f} deg")
    print(f"    Rotation (elev): {args.rotation_elev_search_deg[0]:.1f} to {args.rotation_elev_search_deg[1]:.1f} deg")
    print(f"    Rotation (azim): {args.rotation_azim_search_deg[0]:.1f} to {args.rotation_azim_search_deg[1]:.1f} deg")

    # 5. Run optimization
    print("\n[Step 4] Running GPU-accelerated alignment optimization...")

    if HAS_CUPY:
        results = optimize_alignment_gpu(
            lidar_pcd, ra_radar, params, base_boresight, base_origin,
            tuple(args.range_search_m),
            tuple(args.azimuth_search_deg),
            tuple(args.rotation_elev_search_deg),
            tuple(args.rotation_azim_search_deg),
            metric=args.metric,
            coarse_steps=args.coarse_steps,
            near_field_m=args.near_field_m,
            batch_size=args.batch_size,
            verbose=True
        )
    else:
        # Fallback to CPU
        print("[WARNING] Using CPU fallback...")
        from .cascaded_lidar import optimize_alignment
        results = optimize_alignment(
            lidar_pcd, ra_radar, params, base_boresight, base_origin,
            tuple(args.range_search_m),
            tuple(args.azimuth_search_deg),
            tuple(args.rotation_elev_search_deg),
            tuple(args.rotation_azim_search_deg),
            metric=args.metric,
            coarse_steps=args.coarse_steps,
            near_field_m=args.near_field_m,
            verbose=True
        )

    total_time = time.time() - total_start
    print(f"\n[GPU] Total optimization time: {total_time:.2f}s")

    # 6. Print results
    print_results(results, args.config_path, args.lidar_path, args.gt_adc_path)

    # 7. Save results
    search_params = {
        'range_search_m': list(args.range_search_m),
        'azimuth_search_deg': list(args.azimuth_search_deg),
        'rotation_elev_search_deg': list(args.rotation_elev_search_deg),
        'rotation_azim_search_deg': list(args.rotation_azim_search_deg),
        'metric': args.metric,
        'coarse_steps': args.coarse_steps,
        'gpu_accelerated': HAS_CUPY,
        'total_time_seconds': total_time
    }
    save_results_json(results, args.config_path, args.lidar_path, args.gt_adc_path,
                      search_params, args.output_dir)

    # 8. Create aligned config file
    if args.save_aligned_config:
        opt = results['optimal_params']
        config_dir = os.path.dirname(args.config_path)
        config_name = os.path.basename(args.config_path)
        name_stem = os.path.splitext(config_name)[0]
        aligned_output_path = os.path.join(config_dir, f"{name_stem}{args.aligned_config_suffix}.json")

        aligned_config_path = create_aligned_config(
            config_path=args.config_path,
            delta_range_m=opt['delta_range_m'],
            delta_azimuth_deg=opt['delta_azimuth_deg'],
            rotation_elev_deg=opt['rotation_elev_deg'],
            rotation_azim_deg=opt['rotation_azim_deg'],
            output_path=aligned_output_path
        )

    # 9. Generate visualizations (using CPU functions)
    if args.viz:
        generate_visualizations(
            lidar_pcd, ra_radar, params, base_boresight, base_origin,
            results, args.output_dir, args.near_field_m
        )

    print(f"\nOutput saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
