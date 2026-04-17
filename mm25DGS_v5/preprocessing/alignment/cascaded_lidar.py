#!/usr/bin/env python3
"""
LiDAR-Radar RA Map Alignment Tool

Voxelizes LiDAR point cloud to radar RAE grid and optimizes alignment
with radar ground truth RA map via translation and rotation search.

Usage:
    python lidar_radar_ra_alignment.py \
        --config-path /path/to/config.json \
        --lidar-path /path/to/lidar.ply \
        --gt-adc-path /path/to/radar_gt.npy \
        --viz
"""

# Set Mitsuba variant before any mmir imports
import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy.optimize import minimize

import open3d as o3d

# ============================================================================
# HELPER: Load point cloud from PLY or NPY
# ============================================================================

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Load point cloud from PLY or NPY file.

    For NPY files, assumes format:
    - (N, 3): x, y, z only
    - (N, 4): x, y, z, intensity
    - (N, 6): x, y, z, r, g, b OR x, y, z, n_x, n_y, n_z
    - (N, 7): x, y, z, n_x, n_y, n_z, intensity (our LiDAR format)

    Returns:
        Open3D PointCloud with points and optionally colors/intensities
    """
    path = str(path)

    if path.endswith('.ply'):
        return o3d.io.read_point_cloud(path)

    elif path.endswith('.npy'):
        data = np.load(path)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype(np.float64))

        ncols = data.shape[1]

        if ncols == 4:
            # x, y, z, intensity - store as grayscale colors
            intensity = data[:, 3]
            # Normalize intensity to [0, 1]
            if intensity.max() > intensity.min():
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            else:
                intensity_norm = np.ones_like(intensity)
            colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        elif ncols == 7:
            # x, y, z, n_x, n_y, n_z, intensity - use column 6 (intensity)
            # This is our LiDAR format: points + normals + intensity
            intensity = data[:, 6]  # Last column is intensity
            print(f"[load_point_cloud] 7-column format detected: using column 6 as intensity")
            print(f"  Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
            # Normalize intensity to [0, 1]
            if intensity.max() > intensity.min():
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            else:
                intensity_norm = np.ones_like(intensity)
            colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        elif ncols == 6:
            # x, y, z, r, g, b OR x, y, z, n_x, n_y, n_z
            # Check if values are in typical color range vs normal range
            vals = data[:, 3:6]
            if vals.min() >= 0 and vals.max() <= 255:
                # Likely RGB colors
                colors = vals
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            else:
                # Likely normals (can be negative) - use uniform intensity
                print(f"[load_point_cloud] 6-column format with negative values: assuming normals, using uniform intensity")
                intensity_norm = np.ones((data.shape[0],), dtype=np.float64)
                colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)
                pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    else:
        raise ValueError(f"Unsupported file format: {path}. Expected .ply or .npy")


# ============================================================================
# CATEGORY 1: DIRECT IMPORTS (100% reused, debugged code)
# ============================================================================

# Rotation utilities (Rodrigues' formula)
from mmir.evaluation.utils.gen_configs import (
    _axis_angle_to_matrix,  # gen_configs.py:9-29
    _norm,                   # gen_configs.py:5-7
)

# Config loading
from mmir.evaluation.utils.single_view_viz import (
    load_config_positions_and_boresight,  # single_view_viz.py:189-240
)

# Point cloud utilities
from mmir.evaluation.utils.single_view_proc import (
    get_pcl_intensities,   # single_view_proc.py:728-762
    make_angle_grids_np,   # single_view_proc.py:831-837
    _centers_to_edges,     # single_view_proc.py:839-851
)

# ADC/RA processing
from mmir.data.ra_utils import (
    adc_to_ra_image,          # ra_utils.py:37-64
    ra_polar_to_cartesian,    # ra_utils.py:225-254 - convert polar RA to cartesian
    save_ra_image,            # ra_utils.py:257-278 - save RA image
)
from mmir.data.data_utils import load_target
from mmir.losses.loss_utils import minmax_normalize_numpy  # loss_utils.py:51-67
from mmir.data.io_utils import compute_range_res_from_cfg  # io_utils.py:26-38


# ============================================================================
# ANTENNA BEAM PATTERN WEIGHTING
# ============================================================================

# Default antenna pattern path — MMWCAS TX (shared with v5 CUDA renderer).
# Resolved relative to the repo root so the vendored copy works the same
# regardless of cwd.
DEFAULT_ANTENNA_PATTERN_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'assets', 'antenna_pattern', 'MMWCAS', 'tx1_76.npy'))
if not os.path.isfile(DEFAULT_ANTENNA_PATTERN_PATH):
    DEFAULT_ANTENNA_PATTERN_PATH = None  # fall back; caller must pass explicit path


def load_antenna_pattern(pattern_path: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load antenna beam pattern from NPY file.

    The pattern file has shape (361, 2):
    - 361 rows: angles from -180° to +180° at 1° steps
    - Column 0: Elevation gain in dB (narrower beam, ~41° 0dB width)
    - Column 1: Azimuth gain in dB (wider beam, ~136° 0dB width)

    Args:
        pattern_path: Path to antenna pattern file. If None, uses default.

    Returns:
        angles_deg: Array of angles in degrees, shape (361,)
        az_gain_db: Azimuth gain in dB, shape (361,)
        el_gain_db: Elevation gain in dB, shape (361,)
    """
    if pattern_path is None:
        pattern_path = DEFAULT_ANTENNA_PATTERN_PATH

    pattern = np.load(pattern_path)

    # Pattern has 361 samples from -180 to +180 degrees
    angles_deg = np.linspace(-180, 180, pattern.shape[0])
    # Column 1 = azimuth (wider beam, ~136° 0dB width)
    # Column 0 = elevation (narrower beam, ~41° 0dB width)
    az_gain_db = pattern[:, 1]
    el_gain_db = pattern[:, 0]

    return angles_deg, az_gain_db, el_gain_db


def interpolate_antenna_gain(angles_deg: np.ndarray, gain_db: np.ndarray,
                              query_angles_deg: np.ndarray) -> np.ndarray:
    """
    Interpolate antenna gain for query angles.

    Args:
        angles_deg: Reference angles in degrees, shape (N,)
        gain_db: Gain values in dB at reference angles, shape (N,)
        query_angles_deg: Query angles in degrees, shape (...)

    Returns:
        Interpolated gain in dB, same shape as query_angles_deg
    """
    # Wrap query angles to [-180, 180]
    query_wrapped = np.mod(query_angles_deg + 180, 360) - 180

    # Interpolate (linear)
    return np.interp(query_wrapped, angles_deg, gain_db)


def compute_antenna_weight_grid(
    az_centers: np.ndarray,
    el_centers: np.ndarray,
    pattern_path: str = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute antenna beam pattern weights for a 2D (azimuth x elevation) grid.

    For each grid cell, the weight is computed from the antenna pattern gain
    at the cell's angular position. Weights are in linear scale (not dB).

    Args:
        az_centers: Azimuth bin centers in radians, shape (num_az,)
        el_centers: Elevation bin centers in radians, shape (num_el,)
        pattern_path: Path to antenna pattern file
        verbose: Print debug info

    Returns:
        weights: Shape (num_az, num_el), linear scale weights
    """
    # Load antenna pattern
    angles_deg, az_gain_db, el_gain_db = load_antenna_pattern(pattern_path)

    if verbose:
        print(f"\n[ANTENNA PATTERN] Loaded pattern:")
        print(f"  Angles: {angles_deg[0]:.0f}° to {angles_deg[-1]:.0f}° ({len(angles_deg)} samples)")
        print(f"  Az gain range: {az_gain_db.min():.1f} to {az_gain_db.max():.1f} dB")
        print(f"  El gain range: {el_gain_db.min():.1f} to {el_gain_db.max():.1f} dB")

    # Convert grid centers to degrees
    az_deg = np.degrees(az_centers)  # Shape: (num_az,)
    el_deg = np.degrees(el_centers)  # Shape: (num_el,)

    if verbose:
        print(f"  Grid azimuth range: {az_deg.min():.1f}° to {az_deg.max():.1f}°")
        print(f"  Grid elevation range: {el_deg.min():.1f}° to {el_deg.max():.1f}°")

    # Interpolate gains for each dimension
    az_gain_interp = interpolate_antenna_gain(angles_deg, az_gain_db, az_deg)  # (num_az,)
    el_gain_interp = interpolate_antenna_gain(angles_deg, el_gain_db, el_deg)  # (num_el,)

    # Convert from dB to linear scale
    # Total gain = az_gain + el_gain (in dB) -> multiply in linear
    # Normalize so peak = 1.0 (peak is at boresight, angle=0)
    az_gain_linear = 10 ** (az_gain_interp / 10)  # Shape: (num_az,)
    el_gain_linear = 10 ** (el_gain_interp / 10)  # Shape: (num_el,)

    # Create 2D weight grid (outer product)
    # weights[az_idx, el_idx] = az_gain[az_idx] * el_gain[el_idx]
    weights = np.outer(az_gain_linear, el_gain_linear)  # Shape: (num_az, num_el)

    # Normalize so max weight = 1
    weights = weights / weights.max()

    if verbose:
        print(f"  Weight grid shape: {weights.shape}")
        print(f"  Weight range: {weights.min():.4f} to {weights.max():.4f}")
        # Show weight at center (boresight)
        center_az = len(az_deg) // 2
        center_el = len(el_deg) // 2
        print(f"  Weight at center (az={az_deg[center_az]:.1f}°, el={el_deg[center_el]:.1f}°): {weights[center_az, center_el]:.4f}")

    return weights.astype(np.float32)


def apply_antenna_weights_to_rae(
    rae_tensor: np.ndarray,
    az_centers: np.ndarray,
    el_centers: np.ndarray,
    pattern_path: str = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Apply antenna beam pattern weights to RAE tensor.

    Each voxel at (az_idx, el_idx, range_idx) is multiplied by the
    antenna gain at that angular position.

    Args:
        rae_tensor: Shape (num_az, num_el, num_range)
        az_centers: Azimuth bin centers in radians, shape (num_az,)
        el_centers: Elevation bin centers in radians, shape (num_el,)
        pattern_path: Path to antenna pattern file
        verbose: Print debug info

    Returns:
        weighted_rae: Same shape as input, with antenna weights applied
    """
    # Compute 2D weight grid
    weights_2d = compute_antenna_weight_grid(az_centers, el_centers, pattern_path, verbose)

    # Expand to 3D to match RAE tensor: (num_az, num_el) -> (num_az, num_el, 1)
    weights_3d = weights_2d[:, :, np.newaxis]

    # Apply weights (broadcast along range dimension)
    weighted_rae = rae_tensor * weights_3d

    if verbose:
        print(f"\n[ANTENNA WEIGHTING] Applied weights:")
        print(f"  Input sum: {rae_tensor.sum():.2f}")
        print(f"  Output sum: {weighted_rae.sum():.2f}")
        print(f"  Ratio: {weighted_rae.sum() / (rae_tensor.sum() + 1e-10):.4f}")

    return weighted_rae


# ============================================================================
# STEP 1: Load Configuration and Data
# ============================================================================

def load_radar_config(config_path: str) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Load radar config - wraps existing load_config_positions_and_boresight.

    REUSE: Calls load_config_positions_and_boresight directly.

    Returns:
        params: Dict with num_az_bins, num_el_bins, num_adc, range_resolution
        cfg_center: Radar board center position (3,)
        cfg_bore: Boresight direction unit vector (3,)
    """
    tx_m, rx_m, cfg_center, cfg_bore = load_config_positions_and_boresight(config_path)

    # Build params dict (same pattern as single_view_viz.py:213-240)
    # Note: num_az_bins and num_el_bins are 127 (not 128) because radar FFT
    # drops the first bin (DC bin) before fft shift, resulting in 128-1=127 bins.
    # This ensures center azimuth bin (63) aligns with boresight direction.
    params = {
        'range_resolution': compute_range_res_from_cfg(config_path),
        'num_adc': 256,  # Default, override from config if available
        'num_az_bins': 127,  # 128-1: drop DC bin to match radar FFT output
        'num_el_bins': 127,  # 128-1: drop DC bin to match radar FFT output
    }

    # Load numAdcSamples from config if available
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if 'numAdcSamples' in cfg:
        params['num_adc'] = int(cfg['numAdcSamples'])

    return params, cfg_center, cfg_bore


# ============================================================================
# STEP 2: Transformation Functions for Boresight and Origin
# ============================================================================

def apply_rotation_around_elevation_axis(
    boresight: np.ndarray,
    origin: np.ndarray,
    angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate boresight around elevation axis (left-right sweep).
    This is equivalent to azimuth rotation around Z-axis.

    Reference: gen_configs.py uses rot_vec = [0, 0, angle_rad] for Z-axis rotation

    Args:
        boresight: Original boresight direction (3,)
        origin: Radar origin position (3,)
        angle_deg: Rotation angle in degrees (positive = right)

    Returns:
        new_boresight: Rotated boresight direction (normalized)
        new_origin: Origin (unchanged for pure rotation around board center)
    """
    angle_rad = np.radians(angle_deg)
    rot_vec = np.array([0.0, 0.0, angle_rad])  # Z-axis rotation
    R = _axis_angle_to_matrix(rot_vec)
    new_boresight = R @ boresight
    return _norm(new_boresight), origin.copy()


def apply_rotation_around_azimuth_axis(
    boresight: np.ndarray,
    origin: np.ndarray,
    angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate boresight around azimuth axis (up-down tilt).
    This rotates around the board's local X-axis (perpendicular to boresight, horizontal).

    Args:
        boresight: Original boresight direction (3,)
        origin: Radar origin position (3,)
        angle_deg: Rotation angle in degrees (positive = up)

    Returns:
        new_boresight: Rotated boresight direction (normalized)
        new_origin: Origin (unchanged for pure rotation around board center)
    """
    angle_rad = np.radians(angle_deg)

    # Compute board's local X-axis (azimuth direction)
    # X = cross(boresight, Z_world) normalized
    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight, z_world))

    # Handle case where boresight is parallel to Z
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])

    # Rotation around X-axis (azimuth axis)
    rot_vec = x_board * angle_rad
    R = _axis_angle_to_matrix(rot_vec)
    new_boresight = R @ boresight
    return _norm(new_boresight), origin.copy()


def apply_translation(
    boresight: np.ndarray,
    origin: np.ndarray,
    delta_range_m: float,
    delta_azimuth_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate radar origin in range and azimuth directions.

    Args:
        boresight: Boresight direction (unchanged)
        origin: Original radar origin position (3,)
        delta_range_m: Translation along boresight direction (meters)
        delta_azimuth_deg: Translation perpendicular to boresight in azimuth plane (degrees -> meters)

    Returns:
        new_boresight: Unchanged boresight
        new_origin: Translated origin
    """
    # Range translation: move along boresight direction
    range_offset = _norm(boresight) * delta_range_m

    # Azimuth translation: move perpendicular to boresight in horizontal plane
    # Convert degrees to approximate meters (arc length at reference range)
    # For small angles: arc_length ≈ reference_range * angle_rad
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)

    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight, z_world))

    # Handle case where boresight is parallel to Z
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])

    azimuth_offset = x_board * delta_azimuth_m

    new_origin = origin + range_offset + azimuth_offset
    return boresight.copy(), new_origin


def compute_rotation_to_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix R such that R @ src aligns with dst.

    Uses axis-angle representation via cross product.

    Args:
        src: Source direction vector (3,)
        dst: Destination direction vector (3,)

    Returns:
        R: 3x3 rotation matrix
    """
    src = _norm(src)
    dst = _norm(dst)

    dot = np.clip(np.dot(src, dst), -1.0, 1.0)

    # Nearly parallel - no rotation needed
    if dot > 0.9999:
        return np.eye(3)

    # Nearly anti-parallel - rotate 180 degrees around perpendicular axis
    if dot < -0.9999:
        # Find a perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _norm(np.cross(src, perp))
        rot_vec = axis * np.pi
        return _axis_angle_to_matrix(rot_vec)

    # General case: rotate around cross product axis
    axis = np.cross(src, dst)
    angle = np.arccos(dot)
    rot_vec = _norm(axis) * angle
    return _axis_angle_to_matrix(rot_vec)


# ============================================================================
# STEP 3: Voxelization with Transformed Grid
# ADAPTED from single_view_viz.py:327-419 (voxelize_lidar_to_radar_grid)
# Changes: Add origin/boresight params, return RAE tensor instead of points
# ============================================================================

def voxelize_lidar_with_transformed_grid(
    lidar_pcd: o3d.geometry.PointCloud,
    params: Dict[str, Any],
    boresight: np.ndarray,
    origin: np.ndarray,
    near_field_m: float = 1.5,
    apply_antenna_weights: bool = True,
    antenna_pattern_path: str = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Voxelize LiDAR point cloud to RAE tensor using a transformed grid.

    The grid is defined by:
    - Origin: radar center position (defines range=0 point)
    - Boresight: defines the azimuth=0, elevation=0 direction

    ADAPTED from single_view_viz.py:327-419.
    Changes:
    1. Takes origin and boresight as explicit parameters
    2. Returns full RAE tensor instead of collapsed (centers, vals, keys)
    3. Optionally applies antenna beam pattern weights before returning

    Args:
        lidar_pcd: Open3D point cloud with intensities
        params: Radar parameters (num_az_bins, num_el_bins, num_adc, range_resolution)
        boresight: Current boresight direction (3,)
        origin: Current radar origin (3,)
        near_field_m: Near-field threshold
        apply_antenna_weights: If True, apply antenna beam pattern weights to voxels
        antenna_pattern_path: Path to antenna pattern file (uses default if None)
        verbose: Print debug information

    Returns:
        rae_tensor: Shape (num_az_bins, num_el_bins, num_adc) - aggregated intensities
                    (weighted by antenna pattern if apply_antenna_weights=True)
    """
    # Get angle grids and edges (REUSED from single_view_proc)
    # IMPORTANT: make_angle_grids_np expects FFT size (128), not output bin count (127)
    # It internally removes DC bin, so FFT size 128 -> 127 bin centers
    fft_size_az = int(params['num_az_bins']) + 1  # 127 + 1 = 128
    fft_size_el = int(params['num_el_bins']) + 1  # 127 + 1 = 128
    az_cent, el_cent = make_angle_grids_np(fft_size_az, fft_size_el)
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Grid parameters:")
        print(f"  az_edges: min={az_edges[0]:.4f} max={az_edges[-1]:.4f} ({len(az_edges)} edges)")
        print(f"  el_edges: min={el_edges[0]:.4f} max={el_edges[-1]:.4f} ({len(el_edges)} edges)")
        print(f"  r_edges: min={r_edges[0]:.4f} max={r_edges[-1]:.4f} ({len(r_edges)} edges)")
        print(f"  Range resolution: {params['range_resolution']:.4f} m")
        print(f"  Max range: {r_edges[-1]:.2f} m")

    # 1. Get LiDAR points and intensities
    pts_world = np.asarray(lidar_pcd.points, dtype=np.float32)
    intensities = get_pcl_intensities(lidar_pcd).astype(np.float32)

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Input LiDAR (world coords):")
        print(f"  Total points: {pts_world.shape[0]:,}")
        print(f"  X range: [{pts_world[:, 0].min():.2f}, {pts_world[:, 0].max():.2f}]")
        print(f"  Y range: [{pts_world[:, 1].min():.2f}, {pts_world[:, 1].max():.2f}]")
        print(f"  Z range: [{pts_world[:, 2].min():.2f}, {pts_world[:, 2].max():.2f}]")

    if pts_world.shape[0] == 0:
        A, E, R = int(params['num_az_bins']), int(params['num_el_bins']), int(params['num_adc'])
        return np.zeros((A, E, R), dtype=np.float32)

    # 2. Translate to radar origin frame
    if verbose:
        print(f"\n[VOXELIZE DEBUG] Translation:")
        print(f"  Origin (radar center): [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

    pts_local = pts_world - origin.astype(np.float32)

    if verbose:
        print(f"  After translation (pts_local):")
        print(f"    X range: [{pts_local[:, 0].min():.2f}, {pts_local[:, 0].max():.2f}]")
        print(f"    Y range: [{pts_local[:, 1].min():.2f}, {pts_local[:, 1].max():.2f}]")
        print(f"    Z range: [{pts_local[:, 2].min():.2f}, {pts_local[:, 2].max():.2f}]")

    # 3. Compute rotation matrix to align boresight with +Y axis (radar default frame)
    default_bore = np.array([0.0, 1.0, 0.0])
    R_inv = compute_rotation_to_align(boresight, default_bore)

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Rotation:")
        print(f"  Boresight: [{boresight[0]:.4f}, {boresight[1]:.4f}, {boresight[2]:.4f}]")
        print(f"  Default bore (+Y): [{default_bore[0]:.4f}, {default_bore[1]:.4f}, {default_bore[2]:.4f}]")
        print(f"  R_inv @ boresight = [{(R_inv @ boresight)[0]:.4f}, {(R_inv @ boresight)[1]:.4f}, {(R_inv @ boresight)[2]:.4f}]")

    # 4. Rotate points to radar default frame
    pts_radar = (R_inv @ pts_local.T).T.astype(np.float32)

    if verbose:
        print(f"  After rotation (pts_radar):")
        print(f"    X range: [{pts_radar[:, 0].min():.2f}, {pts_radar[:, 0].max():.2f}]")
        print(f"    Y range: [{pts_radar[:, 1].min():.2f}, {pts_radar[:, 1].max():.2f}]")
        print(f"    Z range: [{pts_radar[:, 2].min():.2f}, {pts_radar[:, 2].max():.2f}]")

    # 5. Convert to spherical coordinates
    x = pts_radar[:, 0]
    y = pts_radar[:, 1]
    z = pts_radar[:, 2]
    r = np.linalg.norm(pts_radar, axis=1)

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Spherical conversion (before near-field filter):")
        print(f"  Range r: min={r.min():.2f} max={r.max():.2f} mean={r.mean():.2f}")

    # 6. Apply near-field filter
    keep = (r >= float(near_field_m)) & np.isfinite(r)
    n_before = x.size
    x, y, z, r, intensities = x[keep], y[keep], z[keep], r[keep], intensities[keep]

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Near-field filter (threshold={near_field_m}m):")
        print(f"  Points before: {n_before:,}")
        print(f"  Points after: {x.size:,}")
        print(f"  Points removed: {n_before - x.size:,}")

    if x.size == 0:
        A, E, R_bins = int(params['num_az_bins']), int(params['num_el_bins']), int(params['num_adc'])
        return np.zeros((A, E, R_bins), dtype=np.float32)

    # Spherical coordinates (matching radar FFT azimuth convention)
    # NOTE: Use -x to flip azimuth direction to match radar FFT convention
    # Radar FFT: positive frequency -> left side of image after fftshift + flip
    az = np.arctan2(-x, y)  # azimuth: -x/y (flipped to match radar convention)
    el = np.arcsin(np.clip(z / np.clip(r, 1e-12, None), -1.0, 1.0))  # elevation

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Spherical coordinates:")
        print(f"  Azimuth (rad): min={az.min():.4f} max={az.max():.4f}")
        print(f"  Azimuth (deg): min={np.degrees(az.min()):.2f} max={np.degrees(az.max()):.2f}")
        print(f"  Elevation (rad): min={el.min():.4f} max={el.max():.4f}")
        print(f"  Elevation (deg): min={np.degrees(el.min()):.2f} max={np.degrees(el.max()):.2f}")
        print(f"  Range (m): min={r.min():.2f} max={r.max():.2f}")

    # 7. Bin into RAE grid
    az_idx = np.searchsorted(az_edges, az, side='right') - 1
    el_idx = np.searchsorted(el_edges, el, side='right') - 1
    r_idx = np.searchsorted(r_edges, r, side='right') - 1

    # 8. Clamp to valid range
    A = int(params['num_az_bins'])
    E = int(params['num_el_bins'])
    R_bins = int(params['num_adc'])

    valid_bins = (az_idx >= 0) & (az_idx < A) & (el_idx >= 0) & (el_idx < E) & (r_idx >= 0) & (r_idx < R_bins)
    n_valid = valid_bins.sum()

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Binning:")
        print(f"  Grid shape: A={A} x E={E} x R={R_bins}")
        print(f"  Points in valid bins: {n_valid:,} / {az_idx.size:,} ({100*n_valid/max(1,az_idx.size):.1f}%)")
        print(f"  az_idx range: [{az_idx.min()}, {az_idx.max()}] (valid: 0 to {A-1})")
        print(f"  el_idx range: [{el_idx.min()}, {el_idx.max()}] (valid: 0 to {E-1})")
        print(f"  r_idx range: [{r_idx.min()}, {r_idx.max()}] (valid: 0 to {R_bins-1})")

    az_idx = az_idx[valid_bins]
    el_idx = el_idx[valid_bins]
    r_idx = r_idx[valid_bins]
    intensities = intensities[valid_bins]

    # 9. Aggregate into RAE tensor
    rae_tensor = np.zeros((A, E, R_bins), dtype=np.float32)

    if intensities.size > 0:
        # Use np.add.at for efficient aggregation
        linear_idx = az_idx * (E * R_bins) + el_idx * R_bins + r_idx
        np.add.at(rae_tensor.ravel(), linear_idx, intensities)

    if verbose:
        print(f"\n[VOXELIZE DEBUG] Output RAE tensor (before antenna weighting):")
        print(f"  Shape: {rae_tensor.shape}")
        print(f"  Non-zero voxels: {(rae_tensor > 0).sum():,}")
        print(f"  Total intensity: {rae_tensor.sum():.2f}")
        print(f"  Max intensity: {rae_tensor.max():.4f}")

    # 10. Apply antenna beam pattern weights (if enabled)
    if apply_antenna_weights:
        rae_tensor = apply_antenna_weights_to_rae(
            rae_tensor, az_cent, el_cent,
            pattern_path=antenna_pattern_path,
            verbose=verbose
        )

    if verbose:
        # Show distribution across azimuth
        ra_map = np.sum(rae_tensor, axis=1)  # Sum along elevation
        print(f"\n[VOXELIZE DEBUG] RA map (collapsed along elevation, after weighting):")
        print(f"  Shape: {ra_map.shape}")
        print(f"  Non-zero cells: {(ra_map > 0).sum():,}")
        print(f"  Azimuth sum (per bin): min={ra_map.sum(axis=1).min():.2f} max={ra_map.sum(axis=1).max():.2f}")
        print(f"  Range sum (per bin): min={ra_map.sum(axis=0).min():.2f} max={ra_map.sum(axis=0).max():.2f}")

    return rae_tensor


def rae_to_ra_map(rae_tensor: np.ndarray) -> np.ndarray:
    """
    Collapse RAE tensor along elevation axis to get RA map.

    Args:
        rae_tensor: Shape (num_az_bins, num_el_bins, num_adc)

    Returns:
        ra_map: Shape (num_az_bins, num_adc) - summed along elevation
    """
    return np.sum(rae_tensor, axis=1)  # Sum along elevation (axis=1)


# ============================================================================
# STEP 4: Process Radar Ground Truth to RA Map
# REUSE: 100% - just calls existing functions
# ============================================================================

def radar_gt_to_ra_map(gt_adc_path: str) -> np.ndarray:
    """
    Load radar ground truth ADC and convert to magnitude RA map.

    Calls load_target (mmir.data.data_utils) and adc_to_ra_image (mmir.data.ra_utils).

    Returns:
        ra_map: Shape (127, 256) - magnitude RA map (azimuth x range)
    """
    adc_data, _ = load_target(gt_adc_path, device='cpu', normalize=False)
    ra_map = adc_to_ra_image(adc_data)
    return ra_map.numpy()  # Shape (127, 256)


# ============================================================================
# STEP 5: Alignment Metrics
# NEW: Standard metrics using numpy/scipy/skimage
# ============================================================================

def compute_alignment_metrics(
    ra_lidar: np.ndarray,
    ra_radar: np.ndarray,
    threshold_percentile: float = 90.0
) -> Dict[str, float]:
    """
    Compute multiple alignment quality metrics.

    Both inputs should be normalized to [0, 1] before calling.

    Returns dict with:
        - correlation: Pearson correlation coefficient
        - mse: Mean squared error
        - ssim: Structural similarity index
        - iou: Intersection over union (binarized)
    """
    # Handle NaN/Inf
    ra_lidar = np.nan_to_num(ra_lidar, nan=0.0, posinf=0.0, neginf=0.0)
    ra_radar = np.nan_to_num(ra_radar, nan=0.0, posinf=0.0, neginf=0.0)

    # Flatten for correlation
    lidar_flat = ra_lidar.flatten()
    radar_flat = ra_radar.flatten()

    # Correlation
    if np.std(lidar_flat) < 1e-10 or np.std(radar_flat) < 1e-10:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(lidar_flat, radar_flat)[0, 1])

    # MSE
    mse = float(np.mean((ra_lidar - ra_radar) ** 2))

    # SSIM
    try:
        from skimage.metrics import structural_similarity as ssim_func
        ssim_val = float(ssim_func(ra_lidar, ra_radar, data_range=1.0))
    except ImportError:
        ssim_val = 0.0  # Fallback if skimage not available

    # IoU (binarize at percentile threshold)
    thresh_lidar = np.percentile(ra_lidar, threshold_percentile)
    thresh_radar = np.percentile(ra_radar, threshold_percentile)
    binary_lidar = ra_lidar > thresh_lidar
    binary_radar = ra_radar > thresh_radar
    intersection = np.logical_and(binary_lidar, binary_radar).sum()
    union = np.logical_or(binary_lidar, binary_radar).sum()
    iou = float(intersection / (union + 1e-8))

    return {
        'correlation': correlation if not np.isnan(correlation) else 0.0,
        'mse': mse,
        'ssim': ssim_val,
        'iou': iou
    }


# ============================================================================
# STEP 6: Single Evaluation Function
# NEW: Orchestration function
# ============================================================================

def evaluate_alignment(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    near_field_m: float = 1.5,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate alignment quality for a single set of transformation parameters.

    Steps:
    1. Apply rotation around elevation axis to boresight
    2. Apply rotation around azimuth axis to boresight
    3. Apply translation to origin
    4. Voxelize LiDAR with transformed grid -> RAE tensor
    5. Collapse RAE to RA map
    6. Normalize both RA maps
    7. Handle shape mismatch (LiDAR: 128x256, Radar: 127x256)
    8. Compute metrics
    """
    # 1-2. Apply rotations
    boresight = base_boresight.copy()
    boresight, _ = apply_rotation_around_elevation_axis(boresight, base_origin, rotation_elev_deg)
    boresight, _ = apply_rotation_around_azimuth_axis(boresight, base_origin, rotation_azim_deg)

    # 3. Apply translation
    _, origin = apply_translation(boresight, base_origin, delta_range_m, delta_azimuth_deg)

    if verbose:
        print(f"\n[EVAL DEBUG] Transformation parameters:")
        print(f"  delta_range_m: {delta_range_m:.4f}")
        print(f"  delta_azimuth_deg: {delta_azimuth_deg:.4f}")
        print(f"  rotation_elev_deg: {rotation_elev_deg:.4f}")
        print(f"  rotation_azim_deg: {rotation_azim_deg:.4f}")
        print(f"  Transformed boresight: [{boresight[0]:.4f}, {boresight[1]:.4f}, {boresight[2]:.4f}]")
        print(f"  Transformed origin: [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]")

    # 4. Voxelize with transformed grid
    rae_tensor = voxelize_lidar_with_transformed_grid(lidar_pcd, params, boresight, origin, near_field_m, verbose=verbose)

    # 5. Collapse to RA
    ra_lidar = rae_to_ra_map(rae_tensor)  # Shape: (128, 256)

    # 6. Normalize
    ra_lidar_norm = minmax_normalize_numpy(ra_lidar)
    ra_radar_norm = minmax_normalize_numpy(ra_radar)

    # 7. Handle shape mismatch: crop LiDAR RA to match radar RA (127, 256)
    # Drop first azimuth bin to match radar FFT output
    if ra_lidar_norm.shape[0] == 128 and ra_radar_norm.shape[0] == 127:
        ra_lidar_norm = ra_lidar_norm[1:, :]  # Drop first azimuth bin

    # 8. Compute metrics
    return compute_alignment_metrics(ra_lidar_norm, ra_radar_norm)


# ============================================================================
# STEP 7: Coarse-to-Fine Optimization
# NEW: Grid search + scipy.optimize.minimize
# ============================================================================

def optimize_alignment(
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
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Find optimal alignment using coarse-to-fine optimization.

    Phase 1 - Coarse Grid Search:
        - Evaluate metric at grid points across 4D parameter space
        - Grid: coarse_steps points per dimension

    Phase 2 - Fine Optimization:
        - Use scipy.optimize.minimize (Nelder-Mead) around best grid point

    Returns:
        {
            'optimal_params': dict with delta_range_m, delta_azimuth_deg, rotation_elev_deg, rotation_azim_deg
            'best_metrics': Dict[str, float]
            'initial_metrics': Dict[str, float]
            'grid_search_results': np.ndarray  # For visualization
        }
    """
    # Phase 1: Coarse grid search
    range_grid = np.linspace(range_search_m[0], range_search_m[1], coarse_steps)
    azimuth_grid = np.linspace(azimuth_search_deg[0], azimuth_search_deg[1], coarse_steps)
    rot_elev_grid = np.linspace(rotation_elev_search_deg[0], rotation_elev_search_deg[1], coarse_steps)
    rot_azim_grid = np.linspace(rotation_azim_search_deg[0], rotation_azim_search_deg[1], coarse_steps)

    # Determine if we maximize or minimize
    maximize = metric in ['correlation', 'ssim', 'iou']
    best_value = -np.inf if maximize else np.inf
    best_params = (0.0, 0.0, 0.0, 0.0)
    grid_results = []

    total_evals = coarse_steps ** 4
    if verbose:
        print(f"Phase 1: Coarse grid search ({coarse_steps}^4 = {total_evals} evaluations)...")

    eval_count = 0
    for dr in range_grid:
        for da in azimuth_grid:
            for re in rot_elev_grid:
                for ra in rot_azim_grid:
                    metrics = evaluate_alignment(
                        lidar_pcd, ra_radar, params, base_boresight, base_origin,
                        dr, da, re, ra, near_field_m
                    )
                    value = metrics[metric]
                    grid_results.append((dr, da, re, ra, value))

                    is_better = (value > best_value) if maximize else (value < best_value)
                    if is_better:
                        best_value = value
                        best_params = (dr, da, re, ra)

                    eval_count += 1
                    if verbose and eval_count % 500 == 0:
                        print(f"  Progress: {eval_count}/{total_evals} ({100*eval_count/total_evals:.1f}%)")

    if verbose:
        print(f"  Best grid point: range={best_params[0]:.2f}m, az={best_params[1]:.2f}deg, "
              f"rot_elev={best_params[2]:.2f}deg, rot_azim={best_params[3]:.2f}deg")
        print(f"  Best {metric}: {best_value:.4f}")

    # Phase 2: Fine optimization
    if verbose:
        print("Phase 2: Fine optimization (Nelder-Mead)...")

    def objective(x):
        metrics = evaluate_alignment(
            lidar_pcd, ra_radar, params, base_boresight, base_origin,
            x[0], x[1], x[2], x[3], near_field_m
        )
        value = metrics[metric]
        # Minimize negative for maximization metrics
        return -value if maximize else value

    result = minimize(
        objective,
        x0=best_params,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.0001}
    )

    optimal_params = {
        'delta_range_m': float(result.x[0]),
        'delta_azimuth_deg': float(result.x[1]),
        'rotation_elev_deg': float(result.x[2]),
        'rotation_azim_deg': float(result.x[3])
    }

    # Compute final metrics at optimal point
    best_metrics = evaluate_alignment(
        lidar_pcd, ra_radar, params, base_boresight, base_origin,
        optimal_params['delta_range_m'],
        optimal_params['delta_azimuth_deg'],
        optimal_params['rotation_elev_deg'],
        optimal_params['rotation_azim_deg'],
        near_field_m
    )

    # Compute initial metrics (no transformation) - with verbose output to debug voxelization
    if verbose:
        print("\n" + "=" * 60)
        print("DEBUG: Initial evaluation (no transformation)")
        print("=" * 60)
    initial_metrics = evaluate_alignment(
        lidar_pcd, ra_radar, params, base_boresight, base_origin,
        0.0, 0.0, 0.0, 0.0, near_field_m, verbose=verbose
    )

    if verbose:
        print(f"  Optimization complete. Final {metric}: {best_metrics[metric]:.4f}")

    return {
        'optimal_params': optimal_params,
        'best_metrics': best_metrics,
        'initial_metrics': initial_metrics,
        'grid_search_results': np.array(grid_results)
    }


# ============================================================================
# STEP 8: Helper function to get RA map for given params (for visualization)
# ============================================================================

def get_ra_map_for_params(
    lidar_pcd: o3d.geometry.PointCloud,
    params: Dict[str, Any],
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    near_field_m: float = 1.5
) -> np.ndarray:
    """Get LiDAR RA map for given transformation parameters."""
    # Apply rotations
    boresight = base_boresight.copy()
    boresight, _ = apply_rotation_around_elevation_axis(boresight, base_origin, rotation_elev_deg)
    boresight, _ = apply_rotation_around_azimuth_axis(boresight, base_origin, rotation_azim_deg)

    # Apply translation
    _, origin = apply_translation(boresight, base_origin, delta_range_m, delta_azimuth_deg)

    # Voxelize and collapse
    rae_tensor = voxelize_lidar_with_transformed_grid(lidar_pcd, params, boresight, origin, near_field_m)
    return rae_to_ra_map(rae_tensor)


# ============================================================================
# STEP 9: Visualization Functions
# NEW: Matplotlib visualizations
# ============================================================================

def add_radar_marker(ra_cart: np.ndarray, radius: int = 4, color: float = 255.0) -> np.ndarray:
    """
    Add a smooth circular marker at the radar position (bottom-center) on cartesian RA image.

    The radar is at physical coords (x=0, y=0), which maps to:
    - Column: center of image (width // 2)
    - Row: 0 (bottom, since we use origin='lower' when displaying)

    Args:
        ra_cart: Cartesian RA image, shape (H, W) for grayscale or (H, W, C) for RGB
        radius: Radius of the marker in pixels (default 4 for smaller marker)
        color: Color value for the marker (default white = 255)

    Returns:
        Modified RA image with marker
    """
    ra_marked = ra_cart.copy()

    # Handle both 2D (grayscale) and 3D (RGB) images
    is_rgb = ra_marked.ndim == 3
    if is_rgb:
        h, w, c = ra_marked.shape
    else:
        h, w = ra_marked.shape

    # Radar position: bottom-center
    # Row 0 is bottom when displayed with origin='lower'
    center_row = radius + 2  # Slightly offset so circle is fully visible
    center_col = w // 2

    # Draw smooth filled circle using anti-aliased distance formula
    # Use a slightly larger search area for anti-aliasing
    for r in range(max(0, center_row - radius - 1), min(h, center_row + radius + 2)):
        for c_idx in range(max(0, center_col - radius - 1), min(w, center_col + radius + 2)):
            dist = np.sqrt((r - center_row)**2 + (c_idx - center_col)**2)
            if dist <= radius - 0.5:
                # Fully inside circle
                if is_rgb:
                    ra_marked[r, c_idx, :] = color  # White on all channels
                else:
                    ra_marked[r, c_idx] = color
            elif dist <= radius + 0.5:
                # Anti-aliased edge (smooth transition)
                alpha = 1.0 - (dist - (radius - 0.5))
                if is_rgb:
                    for ch in range(c):
                        ra_marked[r, c_idx, ch] = ra_marked[r, c_idx, ch] * (1 - alpha) + color * alpha
                else:
                    ra_marked[r, c_idx] = ra_marked[r, c_idx] * (1 - alpha) + color * alpha

    return ra_marked


def generate_visualizations(
    lidar_pcd: o3d.geometry.PointCloud,
    ra_radar: np.ndarray,
    params: Dict[str, Any],
    base_boresight: np.ndarray,
    base_origin: np.ndarray,
    results: Dict[str, Any],
    output_dir: str,
    near_field_m: float = 1.5
):
    """Generate visualization images following convert_adc_to_ra.py pipeline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n[Visualization] Generating images...")

    # Get range resolution for cartesian conversion
    range_res = params['range_resolution']

    # Get initial and optimal RA maps (polar format: azimuth x range)
    ra_lidar_initial = get_ra_map_for_params(
        lidar_pcd, params, base_boresight, base_origin,
        0, 0, 0, 0, near_field_m
    )

    opt = results['optimal_params']
    ra_lidar_optimal = get_ra_map_for_params(
        lidar_pcd, params, base_boresight, base_origin,
        opt['delta_range_m'], opt['delta_azimuth_deg'],
        opt['rotation_elev_deg'], opt['rotation_azim_deg'],
        near_field_m
    )

    # Handle shape mismatch for polar RA maps (before cartesian conversion)
    ra_radar_polar = ra_radar.copy()
    ra_lidar_initial_polar = ra_lidar_initial.copy()
    ra_lidar_optimal_polar = ra_lidar_optimal.copy()

    if ra_lidar_initial_polar.shape[0] == 128 and ra_radar_polar.shape[0] == 127:
        ra_lidar_initial_polar = ra_lidar_initial_polar[1:, :]
        ra_lidar_optimal_polar = ra_lidar_optimal_polar[1:, :]

    # =========================================================================
    # Save individual RA images using convert_adc_to_ra.py pipeline
    # (polar -> cartesian -> normalize -> save)
    # =========================================================================
    print("  Converting RA maps to cartesian and saving...")

    # Convert polar to cartesian
    ra_radar_cart = ra_polar_to_cartesian(ra_radar_polar, range_res)
    ra_lidar_initial_cart = ra_polar_to_cartesian(ra_lidar_initial_polar, range_res)
    ra_lidar_optimal_cart = ra_polar_to_cartesian(ra_lidar_optimal_polar, range_res)

    print(f"    Radar GT cartesian shape: {ra_radar_cart.shape}")
    print(f"    LiDAR initial cartesian shape: {ra_lidar_initial_cart.shape}")
    print(f"    LiDAR optimal cartesian shape: {ra_lidar_optimal_cart.shape}")

    # Normalize and save individual RA images
    ra_radar_cart_norm = minmax_normalize_numpy(ra_radar_cart) * 255
    ra_lidar_initial_cart_norm = minmax_normalize_numpy(ra_lidar_initial_cart) * 255
    ra_lidar_optimal_cart_norm = minmax_normalize_numpy(ra_lidar_optimal_cart) * 255

    # Add small white circle marker at radar position (bottom-center) for ALL cartesian images
    ra_radar_cart_marked = add_radar_marker(ra_radar_cart_norm, radius=4, color=255.0)
    ra_lidar_initial_cart_marked = add_radar_marker(ra_lidar_initial_cart_norm, radius=4, color=255.0)
    ra_lidar_optimal_cart_marked = add_radar_marker(ra_lidar_optimal_cart_norm, radius=4, color=255.0)

    save_ra_image(ra_radar_cart_marked, os.path.join(output_dir, 'ra_radar_gt.png'))
    save_ra_image(ra_lidar_initial_cart_marked, os.path.join(output_dir, 'ra_lidar_initial.png'))
    save_ra_image(ra_lidar_optimal_cart_marked, os.path.join(output_dir, 'ra_lidar_optimized.png'))

    print(f"    Saved: ra_radar_gt.png, ra_lidar_initial.png, ra_lidar_optimized.png")

    # =========================================================================
    # Create overlay images (cartesian) - Red=LiDAR, Green=Radar, Yellow=overlap
    # =========================================================================
    print("  Creating cartesian overlay images...")

    # Normalize to [0, 1] for overlay
    ra_radar_cart_01 = minmax_normalize_numpy(ra_radar_cart)
    ra_lidar_initial_cart_01 = minmax_normalize_numpy(ra_lidar_initial_cart)
    ra_lidar_optimal_cart_01 = minmax_normalize_numpy(ra_lidar_optimal_cart)

    # Initial overlay: Red=LiDAR, Green=Radar
    overlay_initial = np.stack([
        ra_lidar_initial_cart_01,  # Red channel
        ra_radar_cart_01,           # Green channel
        np.zeros_like(ra_radar_cart_01)  # Blue channel
    ], axis=-1)
    overlay_initial_uint8 = (overlay_initial * 255).astype(np.uint8)
    # Add radar marker first, then flip along range axis
    overlay_initial_uint8 = add_radar_marker(overlay_initial_uint8, radius=4, color=255.0).astype(np.uint8)
    overlay_initial_uint8 = np.flipud(overlay_initial_uint8)
    from PIL import Image
    Image.fromarray(overlay_initial_uint8).save(os.path.join(output_dir, 'ra_lidar_initial_overlay.png'))

    # Optimized overlay: Red=LiDAR, Green=Radar
    overlay_optimal = np.stack([
        ra_lidar_optimal_cart_01,  # Red channel
        ra_radar_cart_01,           # Green channel
        np.zeros_like(ra_radar_cart_01)  # Blue channel
    ], axis=-1)
    overlay_optimal_uint8 = (overlay_optimal * 255).astype(np.uint8)
    # Add radar marker first, then flip along range axis
    overlay_optimal_uint8 = add_radar_marker(overlay_optimal_uint8, radius=4, color=255.0).astype(np.uint8)
    overlay_optimal_uint8 = np.flipud(overlay_optimal_uint8)
    Image.fromarray(overlay_optimal_uint8).save(os.path.join(output_dir, 'ra_lidar_optimized_overlay.png'))

    print(f"    Saved: ra_lidar_initial_overlay.png, ra_lidar_optimized_overlay.png")

    # =========================================================================
    # Create difference images (cartesian) - using hot colormap
    # =========================================================================
    print("  Creating cartesian difference images...")

    diff_initial_cart = np.abs(ra_lidar_initial_cart_01 - ra_radar_cart_01)
    diff_optimal_cart = np.abs(ra_lidar_optimal_cart_01 - ra_radar_cart_01)

    # Use matplotlib to apply colormap and save (already imported at top of function)

    # Normalize both differences to same scale for fair comparison
    vmax = max(diff_initial_cart.max(), diff_optimal_cart.max())

    # Apply 'hot' colormap
    cmap = plt.get_cmap('hot')

    # Initial difference
    diff_initial_colored = cmap(diff_initial_cart / (vmax + 1e-8))[:, :, :3]  # Drop alpha
    diff_initial_uint8 = (diff_initial_colored * 255).astype(np.uint8)
    # Add radar marker first, then flip along range axis
    diff_initial_uint8 = add_radar_marker(diff_initial_uint8, radius=4, color=255.0).astype(np.uint8)
    diff_initial_uint8 = np.flipud(diff_initial_uint8)
    Image.fromarray(diff_initial_uint8).save(os.path.join(output_dir, 'ra_lidar_initial_difference.png'))

    # Optimal difference
    diff_optimal_colored = cmap(diff_optimal_cart / (vmax + 1e-8))[:, :, :3]  # Drop alpha
    diff_optimal_uint8 = (diff_optimal_colored * 255).astype(np.uint8)
    # Add radar marker first, then flip along range axis
    diff_optimal_uint8 = add_radar_marker(diff_optimal_uint8, radius=4, color=255.0).astype(np.uint8)
    diff_optimal_uint8 = np.flipud(diff_optimal_uint8)
    Image.fromarray(diff_optimal_uint8).save(os.path.join(output_dir, 'ra_lidar_optimized_difference.png'))

    print(f"    Saved: ra_lidar_initial_difference.png, ra_lidar_optimized_difference.png")

    # =========================================================================
    # Also save comparison plots using polar format (for bin-level analysis)
    # =========================================================================

    # Normalize polar maps for comparison plots
    ra_lidar_initial_norm = minmax_normalize_numpy(ra_lidar_initial_polar)
    ra_lidar_optimal_norm = minmax_normalize_numpy(ra_lidar_optimal_polar)
    ra_radar_norm = minmax_normalize_numpy(ra_radar_polar)

    # 1. Individual RA maps comparison (polar format)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ra_lidar_initial_norm, cmap='plasma', origin='lower', aspect='auto')
    axes[0].set_title('LiDAR RA (Initial)')
    axes[0].set_xlabel('Range bins')
    axes[0].set_ylabel('Azimuth bins')

    axes[1].imshow(ra_radar_norm, cmap='plasma', origin='lower', aspect='auto')
    axes[1].set_title('Radar GT RA')
    axes[1].set_xlabel('Range bins')

    axes[2].imshow(ra_lidar_optimal_norm, cmap='plasma', origin='lower', aspect='auto')
    axes[2].set_title('LiDAR RA (Optimized)')
    axes[2].set_xlabel('Range bins')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ra_maps_comparison_polar.png'), dpi=150)
    plt.close()

    # 2. Individual RA maps comparison (cartesian format)
    ra_radar_cart_norm_01 = minmax_normalize_numpy(ra_radar_cart)
    ra_lidar_initial_cart_norm_01 = minmax_normalize_numpy(ra_lidar_initial_cart)
    ra_lidar_optimal_cart_norm_01 = minmax_normalize_numpy(ra_lidar_optimal_cart)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ra_lidar_initial_cart_norm_01, cmap='plasma', origin='lower', aspect='auto')
    axes[0].set_title('LiDAR RA (Initial) - Cartesian')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')

    axes[1].imshow(ra_radar_cart_norm_01, cmap='plasma', origin='lower', aspect='auto')
    axes[1].set_title('Radar GT RA - Cartesian')
    axes[1].set_xlabel('X (m)')

    axes[2].imshow(ra_lidar_optimal_cart_norm_01, cmap='plasma', origin='lower', aspect='auto')
    axes[2].set_title('LiDAR RA (Optimized) - Cartesian')
    axes[2].set_xlabel('X (m)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ra_maps_comparison_cartesian.png'), dpi=150)
    plt.close()

    # 3. Overlay comparison (before/after) - polar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before: Red=LiDAR, Green=Radar
    overlay_before = np.stack([
        ra_lidar_initial_norm,
        ra_radar_norm,
        np.zeros_like(ra_radar_norm)
    ], axis=-1)
    axes[0].imshow(overlay_before, origin='lower', aspect='auto')
    axes[0].set_title(f"Before (corr={results['initial_metrics']['correlation']:.3f})")
    axes[0].set_xlabel('Range bins')
    axes[0].set_ylabel('Azimuth bins')

    # After: Red=LiDAR, Green=Radar
    overlay_after = np.stack([
        ra_lidar_optimal_norm,
        ra_radar_norm,
        np.zeros_like(ra_radar_norm)
    ], axis=-1)
    axes[1].imshow(overlay_after, origin='lower', aspect='auto')
    axes[1].set_title(f"After (corr={results['best_metrics']['correlation']:.3f})")
    axes[1].set_xlabel('Range bins')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlay_comparison.png'), dpi=150)
    plt.close()

    # 4. Difference maps - polar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    diff_before = np.abs(ra_lidar_initial_norm - ra_radar_norm)
    diff_after = np.abs(ra_lidar_optimal_norm - ra_radar_norm)

    vmax = max(diff_before.max(), diff_after.max())

    im0 = axes[0].imshow(diff_before, cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=vmax)
    axes[0].set_title('Absolute Difference (Before)')
    axes[0].set_xlabel('Range bins')
    axes[0].set_ylabel('Azimuth bins')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(diff_after, cmap='hot', origin='lower', aspect='auto', vmin=0, vmax=vmax)
    axes[1].set_title('Absolute Difference (After)')
    axes[1].set_xlabel('Range bins')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'difference_maps.png'), dpi=150)
    plt.close()

    print(f"  Saved: ra_maps_comparison_polar.png, ra_maps_comparison_cartesian.png")
    print(f"  Saved: overlay_comparison.png, difference_maps.png")


# ============================================================================
# STEP 10: Output Functions
# ============================================================================

def print_results(results: Dict[str, Any], config_path: str, lidar_path: str, gt_adc_path: str):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print("ALIGNMENT RESULTS")
    print("=" * 70)

    print("\nInput Files:")
    print(f"  Config: {config_path}")
    print(f"  LiDAR:  {lidar_path}")
    print(f"  Radar:  {gt_adc_path}")

    print("\nInitial Alignment Metrics (no transformation):")
    for k, v in results['initial_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nOptimal Transformation Parameters:")
    opt = results['optimal_params']
    print(f"  Range translation:    {opt['delta_range_m']:+.3f} m")
    print(f"  Azimuth translation:  {opt['delta_azimuth_deg']:+.2f} deg")
    print(f"  Rotation (elev axis): {opt['rotation_elev_deg']:+.2f} deg")
    print(f"  Rotation (azim axis): {opt['rotation_azim_deg']:+.2f} deg")

    print("\nFinal Alignment Metrics (after optimization):")
    for k, v in results['best_metrics'].items():
        init_v = results['initial_metrics'][k]
        delta = v - init_v
        sign = '+' if delta >= 0 else ''
        print(f"  {k}: {v:.4f} ({sign}{delta:.4f})")


def save_results_json(
    results: Dict[str, Any],
    config_path: str,
    lidar_path: str,
    gt_adc_path: str,
    search_params: Dict[str, Any],
    output_dir: str
):
    """Save results to JSON file."""
    output = {
        'input_files': {
            'config': config_path,
            'lidar': lidar_path,
            'radar_gt': gt_adc_path
        },
        'search_parameters': search_params,
        'initial_metrics': results['initial_metrics'],
        'optimal_params': results['optimal_params'],
        'final_metrics': results['best_metrics']
    }

    output_path = os.path.join(output_dir, 'alignment_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# CREATE ALIGNED CONFIG
# ============================================================================

def create_aligned_config(
    config_path: str,
    delta_range_m: float,
    delta_azimuth_deg: float,
    rotation_elev_deg: float,
    rotation_azim_deg: float,
    output_path: Optional[str] = None
) -> str:
    """
    Create a new radar config file with updated antenna positions and boresights
    based on optimized alignment parameters.

    Transformation order (matching evaluation code):
    1. Rotation around elevation axis (Z-axis rotation for azimuth sweep)
    2. Rotation around azimuth axis (X-axis rotation for elevation tilt)
    3. Translation in rotated frame (range along boresight, azimuth perpendicular)

    Args:
        config_path: Path to original radar config JSON
        delta_range_m: Range translation in meters (along boresight)
        delta_azimuth_deg: Azimuth translation in degrees (converted to meters)
        rotation_elev_deg: Rotation around elevation axis in degrees
        rotation_azim_deg: Rotation around azimuth axis in degrees
        output_path: Output path (default: {original_name}_aligned.json in same folder)

    Returns:
        Path to saved aligned config file
    """
    # Load original config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract antenna positions (in meters)
    tx_pos = np.array([t['pos_mm'] for t in config['tx_array']], dtype=float) / 1000.0
    rx_pos = np.array([r['pos_mm'] for r in config['rx_array']], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = tx_pos.shape[0]
    n_rx = rx_pos.shape[0]

    # Compute board center (geometric mean of all antennas)
    board_center = np.mean(all_pos, axis=0)

    # Extract original boresight (normalized)
    base_boresight = np.array(config['tx_array'][0]['boresight'], dtype=float)
    base_boresight = _norm(base_boresight)

    # =========================================================================
    # STEP 1: Compute rotation matrices
    # =========================================================================

    # 1a. Rotation around elevation axis (Z-axis, azimuth sweep)
    rot_elev_rad = np.radians(rotation_elev_deg)
    rot_vec_elev = np.array([0.0, 0.0, rot_elev_rad])
    R_elev = _axis_angle_to_matrix(rot_vec_elev)

    # Apply first rotation to boresight
    boresight_after_elev = R_elev @ base_boresight
    boresight_after_elev = _norm(boresight_after_elev)

    # 1b. Rotation around azimuth axis (board X-axis, elevation tilt)
    # X-axis is perpendicular to boresight in horizontal plane
    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight_after_elev, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])

    rot_azim_rad = np.radians(rotation_azim_deg)
    rot_vec_azim = x_board * rot_azim_rad
    R_azim = _axis_angle_to_matrix(rot_vec_azim)

    # Combined rotation matrix
    R_combined = R_azim @ R_elev

    # Final rotated boresight
    new_boresight = R_combined @ base_boresight
    new_boresight = _norm(new_boresight)

    # =========================================================================
    # STEP 2: Apply rotation to antenna positions (rotate around board center)
    # =========================================================================

    # Antenna positions relative to board center
    rel_pos = all_pos - board_center

    # Rotate relative positions
    rel_pos_rotated = (R_combined @ rel_pos.T).T  # (N, 3)

    # =========================================================================
    # STEP 3: Compute translation in rotated frame
    # =========================================================================

    # Range translation: along new boresight direction
    range_offset = new_boresight * delta_range_m

    # Azimuth translation: perpendicular to boresight in horizontal plane
    # Convert degrees to meters using arc length at reference range
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)

    # New X-axis after rotation
    x_new = _norm(np.cross(new_boresight, z_world))
    if np.linalg.norm(x_new) < 1e-6:
        x_new = np.array([1.0, 0.0, 0.0])

    azimuth_offset = x_new * delta_azimuth_m

    # Total translation
    translation = range_offset + azimuth_offset

    # =========================================================================
    # STEP 4: Apply translation to board center, reconstruct antenna positions
    # =========================================================================

    new_board_center = board_center + translation
    new_all_pos = rel_pos_rotated + new_board_center

    # Split back into TX and RX
    new_tx_pos = new_all_pos[:n_tx]  # (n_tx, 3)
    new_rx_pos = new_all_pos[n_tx:]  # (n_rx, 3)

    # =========================================================================
    # STEP 5: Update config with new positions and boresights
    # =========================================================================

    aligned_config = json.loads(json.dumps(config))  # Deep copy

    # Update TX array
    for i, tx in enumerate(aligned_config['tx_array']):
        tx['pos_mm'] = (new_tx_pos[i] * 1000.0).tolist()
        tx['boresight'] = new_boresight.tolist()

    # Update RX array
    for i, rx in enumerate(aligned_config['rx_array']):
        rx['pos_mm'] = (new_rx_pos[i] * 1000.0).tolist()
        rx['boresight'] = new_boresight.tolist()

    # =========================================================================
    # STEP 6: Save aligned config
    # =========================================================================

    if output_path is None:
        # Default: save in same folder as original with _aligned suffix
        config_dir = os.path.dirname(config_path)
        config_name = os.path.basename(config_path)
        name_stem = os.path.splitext(config_name)[0]
        output_path = os.path.join(config_dir, f"{name_stem}_aligned.json")

    with open(output_path, 'w') as f:
        json.dump(aligned_config, f, indent=2)

    # Print summary
    print(f"\n[Aligned Config] Created: {output_path}")
    print(f"  Original boresight: [{base_boresight[0]:.4f}, {base_boresight[1]:.4f}, {base_boresight[2]:.4f}]")
    print(f"  New boresight:      [{new_boresight[0]:.4f}, {new_boresight[1]:.4f}, {new_boresight[2]:.4f}]")
    print(f"  Board center shift: [{translation[0]*1000:.2f}, {translation[1]*1000:.2f}, {translation[2]*1000:.2f}] mm")

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LiDAR-Radar RA Map Alignment Tool',
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
    parser.add_argument('--save-aligned-config', action='store_true', default=True,
                        help='Save aligned config file (default: True)')
    parser.add_argument('--no-aligned-config', action='store_false', dest='save_aligned_config',
                        help='Do not save aligned config file')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        config_stem = Path(args.config_path).stem
        args.output_dir = f"output/lidar_radar_alignment_{config_stem}"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("LiDAR-Radar RA Map Alignment Tool")
    print("=" * 70)

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
    print("\n[Step 4] Running alignment optimization...")
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

    # 6. Print results
    print_results(results, args.config_path, args.lidar_path, args.gt_adc_path)

    # 7. Save results
    search_params = {
        'range_search_m': list(args.range_search_m),
        'azimuth_search_deg': list(args.azimuth_search_deg),
        'rotation_elev_search_deg': list(args.rotation_elev_search_deg),
        'rotation_azim_search_deg': list(args.rotation_azim_search_deg),
        'metric': args.metric,
        'coarse_steps': args.coarse_steps
    }
    save_results_json(results, args.config_path, args.lidar_path, args.gt_adc_path,
                      search_params, args.output_dir)

    # 8. Create aligned config file
    if args.save_aligned_config:
        opt = results['optimal_params']
        aligned_config_path = create_aligned_config(
            config_path=args.config_path,
            delta_range_m=opt['delta_range_m'],
            delta_azimuth_deg=opt['delta_azimuth_deg'],
            rotation_elev_deg=opt['rotation_elev_deg'],
            rotation_azim_deg=opt['rotation_azim_deg']
        )

    # 9. Generate visualizations
    if args.viz:
        generate_visualizations(
            lidar_pcd, ra_radar, params, base_boresight, base_origin,
            results, args.output_dir, args.near_field_m
        )

    print(f"\nOutput saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
