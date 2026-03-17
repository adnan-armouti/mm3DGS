import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import sys
import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def points_to_voxel_keys(points, voxel_size_m):
    """Convert points to voxel keys using numpy.ravel_multi_index."""
    return np.ravel_multi_index(
        np.floor(points / voxel_size_m).astype(np.int32).T,
        (int(1e6), int(1e6), int(1e6)), mode='clip'
    )


def compute_0db_limits_numpy(pattern_file):
    """
    Compute azimuth and elevation half-widths at 0dB cutoff using pure numpy.

    This is a simplified numpy version of AntennaPatternLoader.compute_0db_limits()
    for evaluation purposes (no differentiability needed).

    Algorithm:
    1. Load E-plane and H-plane patterns (in dB)
    2. Convert to linear scale
    3. Use soft threshold (tanh) to find where gain ~ 0dB (1.0 linear)
    4. Find boundary regions where indicator is between 0.2 and 0.8
    5. Compute angular distance from boresight (180deg)
    6. Average the distances to get half-width

    Pattern boresight is at 180deg (pi radians), not at 0deg.
    Main lobe region is [90deg, 270deg] = [pi/2, 3pi/2].

    Args:
        pattern_file: Path to .npy antenna pattern file

    Returns:
        (azimuth_half_width_rad, elevation_half_width_rad): Tuple of floats
    """
    # Load pattern data
    pattern_data = np.load(pattern_file)

    # Extract E-plane (elevation) and H-plane (azimuth) in dB
    # Handle both formats: (2, N) for IWR1443 and (N, 2) for MMWCAS
    if pattern_data.shape[0] == 2:
        # IWR1443 format: (2, N) where rows are [E-plane, H-plane]
        E_plane_db = pattern_data[0, :]
        H_plane_db = pattern_data[1, :]
    else:
        # MMWCAS format: (N, 2) where columns are [E-plane, H-plane]
        E_plane_db = pattern_data[:, 0]
        H_plane_db = pattern_data[:, 1]

    # Convert to linear scale
    E_plane_linear = 10.0 ** (E_plane_db / 10.0)
    H_plane_linear = 10.0 ** (H_plane_db / 10.0)

    # Create angle grid (0 to 360 degrees)
    num_angles = E_plane_db.shape[0]
    angles_deg = np.linspace(0, 360, num_angles, dtype=np.float32)
    angles_rad = angles_deg * np.pi / 180.0

    # Threshold: 0 dB = 1.0 linear
    threshold_linear = 1.0

    # Boresight is at 180deg (pi radians)
    boresight_angle = np.pi

    # Main lobe region: [90deg, 270deg] = [pi/2, 3pi/2]
    main_lobe_mask = (angles_rad >= (np.pi / 2.0)) & (angles_rad <= (3.0 * np.pi / 2.0))

    # Tanh steepness for soft threshold
    tanh_steepness = 5.0

    # ========== H-PLANE (AZIMUTH) LIMIT ==========
    above_threshold_H = 0.5 * (1.0 + np.tanh(tanh_steepness * (H_plane_linear - threshold_linear)))
    is_at_boundary_H = (above_threshold_H >= 0.2) & (above_threshold_H <= 0.8)
    valid_mask_H = main_lobe_mask & is_at_boundary_H

    # Compute angular distance from boresight
    angle_distance_H = np.abs(angles_rad - boresight_angle)
    boundary_distances_H = np.where(valid_mask_H, angle_distance_H, 0.0)

    # Average of boundary distances
    sum_distances_H = np.sum(boundary_distances_H)
    count_boundary_H = np.sum(valid_mask_H.astype(np.float32))
    azimuth_half_width = sum_distances_H / (count_boundary_H + 1e-8)

    # Clamp to reasonable range: [10deg, 85deg]
    azimuth_half_width = np.clip(azimuth_half_width, np.deg2rad(10.0), np.deg2rad(85.0))

    # ========== E-PLANE (ELEVATION) LIMIT ==========
    above_threshold_E = 0.5 * (1.0 + np.tanh(tanh_steepness * (E_plane_linear - threshold_linear)))
    is_at_boundary_E = (above_threshold_E >= 0.2) & (above_threshold_E <= 0.8)
    valid_mask_E = main_lobe_mask & is_at_boundary_E

    angle_distance_E = np.abs(angles_rad - boresight_angle)
    boundary_distances_E = np.where(valid_mask_E, angle_distance_E, 0.0)

    sum_distances_E = np.sum(boundary_distances_E)
    count_boundary_E = np.sum(valid_mask_E.astype(np.float32))
    elevation_half_width = sum_distances_E / (count_boundary_E + 1e-8)

    # Clamp to reasonable range: [10deg, 85deg]
    elevation_half_width = np.clip(elevation_half_width, np.deg2rad(10.0), np.deg2rad(85.0))

    return float(azimuth_half_width), float(elevation_half_width)


def normalize_radar_intensities(radar_keys: np.ndarray,
                                 radar_vals_agg: np.ndarray) -> dict:
    """
    Normalize radar intensities using Option C: 90th percentile -> 0.1

    This ensures that 90% of occupied bins have normalized intensity <= 0.1,
    matching the interpretation that most radar points are weak returns.

    Args:
        radar_keys: [N] Polar voxel keys (int64)
        radar_vals_agg: [N] Aggregated radar intensities (power magnitude)

    Returns:
        dict mapping polar_key -> normalized_intensity [0.0, 1.0]
    """
    if len(radar_vals_agg) == 0:
        return {}

    # Compute 90th percentile
    p90 = np.percentile(radar_vals_agg, 90)

    # Scale so p90 -> 0.1
    # normalized = intensity / (p90 / 0.1) = intensity * (0.1 / p90)
    scale_factor = 0.1 / (p90 + 1e-20)  # Avoid division by zero
    normalized = radar_vals_agg * scale_factor

    # Clip to [0, 1] range
    normalized = np.clip(normalized, 0.0, 1.0)

    # Create dictionary mapping
    intensity_map = {}
    for key, norm_val in zip(radar_keys, normalized):
        intensity_map[int(key)] = float(norm_val)

    # Log statistics
    below_threshold = np.sum(normalized < 0.1) / len(normalized) * 100.0
    print(f"  Radar intensity normalization (Option C):")
    print(f"    90th percentile: {p90:.6e}")
    print(f"    Scale factor: {scale_factor:.6e}")
    print(f"    Bins with normalized intensity < 0.1: {below_threshold:.1f}%")
    print(f"    Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    return intensity_map


def filter_lidar_by_radar_intensity(lidar_pts: np.ndarray,
                                     radar_center: np.ndarray,
                                     radar_boresight: np.ndarray,
                                     radar_intensity_map: dict,
                                     range_resolution: float,
                                     num_azimuth_bins: int,
                                     num_elevation_bins: int,
                                     num_adc: int,
                                     intensity_threshold: float = 0.1) -> np.ndarray:
    """
    Filter LiDAR points based on radar intensity at their projected polar bins.

    Keep only LiDAR points where radar has normalized intensity >= threshold.
    For empty bins (no radar return), keep the point (radar might have missed it).

    Args:
        lidar_pts: [N, 3] LiDAR points in world coordinates
        radar_center: [3] Radar array center position
        radar_boresight: [3] Radar boresight direction (unit vector)
        radar_intensity_map: dict mapping polar_key -> normalized_intensity
        range_resolution: Range bin resolution in meters
        num_azimuth_bins: Number of azimuth bins
        num_elevation_bins: Number of elevation bins
        num_adc: Number of ADC samples (range bins)
        intensity_threshold: Normalized intensity threshold (default: 0.1)

    Returns:
        mask: [N] Boolean mask indicating which points to keep
    """
    if len(lidar_pts) == 0:
        return np.zeros(0, dtype=bool)

    # Transform LiDAR points to radar frame
    pts_centered = lidar_pts - radar_center.reshape(1, 3)

    # Build rotation matrix to align radar boresight with +Y axis
    y_local = radar_boresight / np.linalg.norm(radar_boresight)

    up_ref = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(y_local, up_ref)) > 0.99:
        x_ref = np.array([1.0, 0.0, 0.0])
    else:
        x_ref = np.cross(up_ref, y_local)
    x_local = x_ref / np.linalg.norm(x_ref)
    z_local = np.cross(x_local, y_local)

    R = np.column_stack([x_local, y_local, z_local])
    pts_local = pts_centered @ R  # [N, 3] in radar local frame

    # Convert to spherical coordinates
    x = pts_local[:, 0]
    y = pts_local[:, 1]
    z = pts_local[:, 2]

    range_m = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(x, y)
    elevation = np.arctan2(z, y)

    # Convert to polar bin indices
    range_bin = np.round(range_m / range_resolution).astype(np.int32)
    range_bin = np.clip(range_bin, 0, num_adc - 1)

    # Azimuth: map [-pi, pi] to [0, num_azimuth_bins)
    az_normalized = (azimuth + np.pi) / (2 * np.pi)
    az_bin = np.floor(az_normalized * num_azimuth_bins).astype(np.int32)
    az_bin = np.clip(az_bin, 0, num_azimuth_bins - 1)

    # Elevation: map [-pi/2, pi/2] to [0, num_elevation_bins)
    el_normalized = (elevation + np.pi/2) / np.pi
    el_bin = np.floor(el_normalized * num_elevation_bins).astype(np.int32)
    el_bin = np.clip(el_bin, 0, num_elevation_bins - 1)

    # Compute polar keys
    polar_keys = (
        range_bin.astype(np.int64) * (num_azimuth_bins * num_elevation_bins) +
        az_bin.astype(np.int64) * num_elevation_bins +
        el_bin.astype(np.int64)
    )

    # Filter based on radar intensity
    keep_mask = np.zeros(len(lidar_pts), dtype=bool)

    for i, key in enumerate(polar_keys):
        if key in radar_intensity_map:
            # Bin has radar return - check intensity threshold
            if radar_intensity_map[key] >= intensity_threshold:
                keep_mask[i] = True
        else:
            # Empty bin (no radar return) - keep the point
            # Radar might have missed it, but LiDAR detected it
            keep_mask[i] = True

    num_kept = np.sum(keep_mask)
    num_filtered = len(lidar_pts) - num_kept

    print(f"  Radar intensity filtering (threshold={intensity_threshold:.2f}):")
    print(f"    Kept: {num_kept} / {len(lidar_pts)} ({100*num_kept/len(lidar_pts):.1f}%)")
    print(f"    Filtered out: {num_filtered} ({100*num_filtered/len(lidar_pts):.1f}%)")

    return keep_mask


def lidar_points_to_polar_voxel_keys(lidar_pts: np.ndarray,
                                      radar_center: np.ndarray,
                                      radar_boresight: np.ndarray,
                                      range_resolution: float,
                                      num_azimuth_bins: int,
                                      num_elevation_bins: int,
                                      num_adc: int) -> np.ndarray:
    """
    Convert LiDAR points to polar voxel keys matching radar's coordinate system.

    This function exactly replicates the logic from voxelize_lidar_to_radar_grid():
    1. Translate points to radar center (origin)
    2. Rotate to align radar boresight with +Y axis using Rodrigues formula
    3. Convert to spherical coordinates using arctan2(x,y) and arcsin(z/r)
    4. Bin using searchsorted with angle grid edges
    5. Compute keys using formula: (az_idx * E + el_idx) * Rb + r_idx

    Args:
        lidar_pts: [N, 3] array of LiDAR point coordinates in world frame
        radar_center: [3] radar center position (cfg_center)
        radar_boresight: [3] radar boresight direction vector
        range_resolution: Range bin size in meters
        num_azimuth_bins: Number of azimuth bins
        num_elevation_bins: Number of elevation bins
        num_adc: Number of range bins

    Returns:
        [M] array of unique polar voxel keys (int64)
    """
    if len(lidar_pts) == 0:
        return np.zeros(0, dtype=np.int64)

    # 1. Translate to radar center (same as subtracting cfg_center)
    pts_centered = lidar_pts - radar_center.reshape(1, 3)

    # 2. Rotate to align radar boresight with +Y axis (inverse rotation)
    # Exact replication of R_inv computation from voxelize_lidar_to_radar_grid (lines 245-278)
    a = np.array([0.0, 1.0, 0.0], dtype=float)  # Target: +Y axis
    b = radar_boresight.astype(float)  # Source: cfg_bore
    b = b / (np.linalg.norm(b) + 1e-12)  # Normalize

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if abs(dot - 1.0) < 1e-8:
        # Already aligned
        R_inv = np.eye(3)
    elif abs(dot + 1.0) < 1e-8:
        # 180 degree rotation
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, a) * a
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ux, uy, uz = axis
        c = -1.0
        s = 0.0
        R_forward = np.array([
            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
        ], dtype=float)
        R_inv = R_forward.T
    else:
        # General rotation using Rodrigues formula
        axis = np.cross(a, b)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ang = np.arccos(dot)
        ux, uy, uz = axis
        c = np.cos(ang)
        s = np.sin(ang)
        R_forward = np.array([
            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
        ], dtype=float)
        R_inv = R_forward.T

    pts_local = (pts_centered @ R_inv.T).astype(np.float32)

    # 3. Convert to spherical coordinates (exact match to lines 283-290)
    x = pts_local[:, 0]
    y = pts_local[:, 1]
    z = pts_local[:, 2]

    r = np.linalg.norm(pts_local, axis=1)

    # CRITICAL: Use arctan2(x, y) for azimuth and arcsin(z/r) for elevation
    # (NOT arctan2(z, y) which was the bug!)
    az = np.arctan2(x, y)  # Line 289 in original
    el = np.arcsin(np.clip(z / np.clip(r, 1e-12, None), -1.0, 1.0))  # Line 290 in original

    # 4. Create bin edges using make_angle_grids_np and _centers_to_edges
    # (exact match to lines 239-242)
    eps = 1e-6
    t_az = np.arange(-num_azimuth_bins // 2 + 1, num_azimuth_bins // 2, dtype=np.float64) * (2.0 / float(num_azimuth_bins))
    t_el = np.arange(-num_elevation_bins // 2 + 1, num_elevation_bins // 2, dtype=np.float64) * (2.0 / float(num_elevation_bins))
    t_az = np.clip(t_az, -1.0 + eps, 1.0 - eps)
    t_el = np.clip(t_el, -1.0 + eps, 1.0 - eps)
    az_cent = np.arcsin(t_az)
    el_cent = np.arcsin(t_el)

    # Convert centers to edges
    def centers_to_edges(centers, low_clip, high_clip):
        centers = centers.astype(np.float64)
        if centers.size == 1:
            width = 1e-3
            return np.array([centers[0] - width, centers[0] + width], dtype=np.float64)
        diffs = np.diff(centers)
        edges = np.empty(centers.size + 1, dtype=np.float64)
        edges[1:-1] = centers[:-1] + 0.5 * diffs
        edges[0] = centers[0] - 0.5 * diffs[0]
        edges[-1] = centers[-1] + 0.5 * diffs[-1]
        edges[0] = max(edges[0], low_clip)
        edges[-1] = min(edges[-1], high_clip)
        return edges

    az_edges = centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(num_adc + 1, dtype=np.float32) * float(range_resolution))

    # 5. Bin using searchsorted (exact match to lines 291-293)
    az_idx = np.searchsorted(az_edges, az, side='right') - 1
    el_idx = np.searchsorted(el_edges, el, side='right') - 1
    r_idx = np.searchsorted(r_edges, r, side='right') - 1

    A = az_cent.size
    E = el_cent.size
    Rb = num_adc

    # 6. Validity check (exact match to line 296)
    valid = (az_idx >= 0) & (az_idx < A) & (el_idx >= 0) & (el_idx < E) & (r_idx >= 0) & (r_idx < Rb)
    az_idx = az_idx[valid]
    el_idx = el_idx[valid]
    r_idx = r_idx[valid]

    # 7. Compute polar voxel keys using CORRECT formula (line 300)
    # CRITICAL: (az_idx * E + el_idx) * Rb + r_idx (NOT range first!)
    polar_keys = (az_idx.astype(np.int64) * E + el_idx.astype(np.int64)) * Rb + r_idx.astype(np.int64)

    return polar_keys


def filter_lidar_by_antenna_beam(lidar_pts: np.ndarray,
                                  radar_center: np.ndarray,
                                  radar_boresight: np.ndarray,
                                  use_mmwcas_patterns: bool = True,
                                  intensity_percentile: float = 10.0,
                                  lidar_intensities: np.ndarray = None,
                                  radar_keys: np.ndarray = None,
                                  radar_vals_agg: np.ndarray = None,
                                  range_resolution: float = None,
                                  num_azimuth_bins: int = None,
                                  num_elevation_bins: int = None,
                                  num_adc: int = None,
                                  enable_radar_intensity_filter: bool = False) -> tuple:
    """
    Filter LiDAR points to only those visible to radar antenna beam.

    This function applies two types of filtering:
    1. Antenna beam geometry filtering (azimuth, elevation limits)
    2. LiDAR intensity percentile filtering (remove weakest 10%)
    3. [Optional] Radar intensity-based filtering (Option C: keep only points
       where radar has normalized intensity >= 0.1)

    Args:
        lidar_pts: [N, 3] LiDAR points in world coordinates
        radar_center: [3] Radar array center position
        radar_boresight: [3] Radar boresight direction (unit vector)
        use_mmwcas_patterns: If True, use MMWCAS patterns; else use IWR1443
        intensity_percentile: Percentile threshold to remove weakest points (default: 10%)
        lidar_intensities: [N] Optional intensity values for each point
        radar_keys: [M] Radar polar voxel keys (for intensity filtering)
        radar_vals_agg: [M] Radar aggregated intensities (for intensity filtering)
        range_resolution: Range bin resolution in meters (for intensity filtering)
        num_azimuth_bins: Number of azimuth bins (for intensity filtering)
        num_elevation_bins: Number of elevation bins (for intensity filtering)
        num_adc: Number of ADC samples (for intensity filtering)
        enable_radar_intensity_filter: If True, apply radar intensity filtering

    Returns:
        filtered_pts: [M, 3] Filtered LiDAR points
        filtered_intensities: [M] Filtered intensities (or None)
    """
    # Load antenna patterns and compute 0dB limits using numpy
    try:
        if use_mmwcas_patterns:
            tx_pattern_file = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy")
            rx_pattern_file = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy")
            print(f"  Using MMWCAS antenna patterns (TX: tx1_76.npy, RX: rx1_76.npy)")
        else:
            tx_pattern_file = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "IWR1443", "pattern_76.npy")
            rx_pattern_file = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "IWR1443", "pattern_76.npy")
            print(f"  Using IWR1443 antenna pattern (pattern_76.npy)")

        # Compute 0dB beam limits using numpy
        tx_az_hw, tx_el_hw = compute_0db_limits_numpy(tx_pattern_file)
        rx_az_hw, rx_el_hw = compute_0db_limits_numpy(rx_pattern_file)

    except Exception as e:
        print(f"[warn] Failed to load antenna patterns: {e}")
        print(f"  Proceeding with full LiDAR point cloud (no antenna beam filtering)")
        import traceback
        traceback.print_exc()
        return lidar_pts, lidar_intensities

    # Take more restrictive limits (smaller half-widths)
    az_hw = min(tx_az_hw, rx_az_hw)
    el_hw = min(tx_el_hw, rx_el_hw)

    print(f"  TX beam limits: az=+/-{np.rad2deg(tx_az_hw):.1f}deg, el=+/-{np.rad2deg(tx_el_hw):.1f}deg")
    print(f"  RX beam limits: az=+/-{np.rad2deg(rx_az_hw):.1f}deg, el=+/-{np.rad2deg(rx_el_hw):.1f}deg")
    print(f"  Using more restrictive: az=+/-{np.rad2deg(az_hw):.1f}deg, el=+/-{np.rad2deg(el_hw):.1f}deg")

    # Transform LiDAR points to radar frame
    # 1. Translate to radar center
    pts_centered = lidar_pts - radar_center.reshape(1, 3)

    # 2. Build rotation matrix to align radar boresight with +Y axis
    #    Radar local frame: +Y = boresight, +X = right, +Z = up
    y_local = radar_boresight / np.linalg.norm(radar_boresight)

    # Build orthonormal frame
    up_ref = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(y_local, up_ref)) > 0.99:
        # Boresight nearly vertical, use X as reference
        x_ref = np.array([1.0, 0.0, 0.0])
    else:
        x_ref = np.cross(up_ref, y_local)
    x_local = x_ref / np.linalg.norm(x_ref)
    z_local = np.cross(x_local, y_local)

    # Rotation matrix: world to local
    # Each column is a local axis expressed in world coordinates
    R = np.column_stack([x_local, y_local, z_local])

    # Transform points to local frame
    pts_local = pts_centered @ R  # [N, 3] in radar local frame

    # 3. Convert to spherical coordinates
    x = pts_local[:, 0]
    y = pts_local[:, 1]
    z = pts_local[:, 2]

    # Azimuth: angle from +Y in XY plane (right = positive)
    azimuth = np.arctan2(x, y)

    # Elevation: angle from +Y in YZ plane (up = positive)
    elevation = np.arctan2(z, y)

    # 4. Filter by beam limits
    within_beam = (np.abs(azimuth) <= az_hw) & (np.abs(elevation) <= el_hw)

    print(f"  LiDAR points within beam: {np.sum(within_beam)} / {len(lidar_pts)} ({100*np.mean(within_beam):.1f}%)")

    # 5. Filter by LiDAR intensity percentile (ONLY if radar intensity filtering is disabled)
    if lidar_intensities is not None and intensity_percentile > 0 and not enable_radar_intensity_filter:
        # Keep only points above the specified percentile
        threshold = np.percentile(lidar_intensities, intensity_percentile)
        above_threshold = lidar_intensities >= threshold

        # Combine filters
        final_mask = within_beam & above_threshold

        print(f"  Removing weakest {intensity_percentile:.0f}% by LiDAR intensity")
        print(f"  After antenna beam + LiDAR intensity filtering: {np.sum(final_mask)} / {len(lidar_pts)} ({100*np.mean(final_mask):.1f}%)")

        filtered_pts = lidar_pts[final_mask]
        filtered_intensities = lidar_intensities[final_mask]
    else:
        # No LiDAR intensity filtering (either disabled or using radar intensity filter instead)
        final_mask = within_beam
        filtered_pts = lidar_pts[final_mask]
        filtered_intensities = lidar_intensities[final_mask] if lidar_intensities is not None else None

        if enable_radar_intensity_filter:
            print(f"  After antenna beam filtering: {np.sum(final_mask)} / {len(lidar_pts)} ({100*np.mean(final_mask):.1f}%)")
            print(f"  (Skipping LiDAR intensity filtering - using radar intensity filter instead)")
        else:
            print(f"  After antenna beam filtering: {np.sum(final_mask)} / {len(lidar_pts)} ({100*np.mean(final_mask):.1f}%)")

    # 6. Apply radar intensity filtering (Option C)
    if enable_radar_intensity_filter:
        if (radar_keys is not None and radar_vals_agg is not None and
            range_resolution is not None and num_azimuth_bins is not None and
            num_elevation_bins is not None and num_adc is not None):

            print(f"\n  Applying radar intensity filtering (Option C):")

            # Normalize radar intensities
            radar_intensity_map = normalize_radar_intensities(radar_keys, radar_vals_agg)

            # Filter LiDAR points based on radar intensity
            radar_intensity_mask = filter_lidar_by_radar_intensity(
                filtered_pts,
                radar_center,
                radar_boresight,
                radar_intensity_map,
                range_resolution,
                num_azimuth_bins,
                num_elevation_bins,
                num_adc,
                intensity_threshold=0.1
            )

            # Apply the mask
            filtered_pts = filtered_pts[radar_intensity_mask]
            if filtered_intensities is not None:
                filtered_intensities = filtered_intensities[radar_intensity_mask]

            print(f"  Final filtered points (after all filters): {len(filtered_pts)} / {len(lidar_pts)} ({100*len(filtered_pts)/len(lidar_pts):.1f}%)")
        else:
            print(f"  [warn] Radar intensity filtering requested but required parameters missing. Skipping.")

    return filtered_pts, filtered_intensities


def compute_distance_based_occupancy_metrics(radar_pts: np.ndarray,
                                             lidar_pts: np.ndarray,
                                             distance_threshold: float = 0.5) -> dict:
    """
    Compute precision, recall, and accuracy based on nearest neighbor distances.

    Following the definitions:
    - Accuracy = (matched_P + matched_Q) / (|P| + |Q|)
    - Precision = matched_P / |P|
    - Recall = matched_Q / |Q|

    Where matched means within distance threshold tau.

    Args:
        radar_pts: [N_radar, 3] Radar point cloud (P)
        lidar_pts: [N_lidar, 3] LiDAR point cloud (Q - ground truth)
        distance_threshold: Distance threshold tau in meters (default: 0.5m)

    Returns:
        Dictionary with precision, recall, accuracy metrics
    """
    metrics = {}

    if len(radar_pts) == 0 or len(lidar_pts) == 0:
        print("  [warn] Empty point cloud, setting metrics to 0.0")
        return {
            'distance_precision': 0.0,
            'distance_recall': 0.0,
            'distance_specificity': 0.0,
            'distance_accuracy': 0.0,
            'distance_threshold_m': distance_threshold
        }

    # Build KD-trees for nearest neighbor search
    radar_tree = cKDTree(radar_pts)
    lidar_tree = cKDTree(lidar_pts)

    # Compute d(p, Q) = min distance from each radar point to LiDAR
    dist_radar_to_lidar, _ = lidar_tree.query(radar_pts, k=1)

    # Compute d(q, P) = min distance from each LiDAR point to radar
    dist_lidar_to_radar, _ = radar_tree.query(lidar_pts, k=1)

    # Count matches within threshold
    matched_radar = np.sum(dist_radar_to_lidar < distance_threshold)
    matched_lidar = np.sum(dist_lidar_to_radar < distance_threshold)

    # Compute metrics
    precision = matched_radar / len(radar_pts) if len(radar_pts) > 0 else 0.0
    recall = matched_lidar / len(lidar_pts) if len(lidar_pts) > 0 else 0.0
    accuracy = (matched_radar + matched_lidar) / (len(radar_pts) + len(lidar_pts))

    # Compute specificity as: (unmatched_lidar) / (unmatched_lidar + unmatched_radar)
    # This measures "what fraction of unmatched points are from LiDAR"
    # High specificity means radar has few false positives relative to LiDAR's coverage
    unmatched_radar = len(radar_pts) - matched_radar
    unmatched_lidar = len(lidar_pts) - matched_lidar
    specificity = unmatched_lidar / (unmatched_lidar + unmatched_radar) if (unmatched_lidar + unmatched_radar) > 0 else 0.0

    metrics['distance_precision'] = precision
    metrics['distance_recall'] = recall
    metrics['distance_specificity'] = specificity
    metrics['distance_accuracy'] = accuracy
    metrics['distance_threshold_m'] = distance_threshold
    metrics['matched_radar_points'] = int(matched_radar)
    metrics['matched_lidar_points'] = int(matched_lidar)
    metrics['total_radar_points'] = len(radar_pts)
    metrics['total_lidar_points'] = len(lidar_pts)

    return metrics
################################################################################
# NEW COMPOSABLE EVALUATION HELPERS
################################################################################

def compute_voxel_occupancy_metrics_polar(radar_voxel_keys: np.ndarray,
                                          lidar_voxel_keys: np.ndarray,
                                          num_azimuth_bins: int,
                                          num_elevation_bins: int,
                                          num_adc: int) -> dict:
    metrics: dict = {}
    print("\n1) VOXEL-BASED OCCUPANCY METRICS (polar keys)")
    print("-" * 40)
    if radar_voxel_keys is None or lidar_voxel_keys is None:
        print("[warn] Polar keys not provided; skipping occupancy metrics")
        metrics.update({'voxel_iou': 0.0, 'radar_hit_rate': 0.0, 'radar_false_alarm_rate': 0.0, 'lidar_coverage_by_radar': 0.0})
        return metrics

    R_set = set(map(int, np.unique(radar_voxel_keys.astype(np.int64))))
    L_set = set(map(int, np.unique(lidar_voxel_keys.astype(np.int64))))
    print(f"Radar occupied cells: {len(R_set)}")
    print(f"LiDAR occupied cells: {len(L_set)}")
    inter = R_set & L_set
    union = R_set | L_set
    iou = len(inter) / len(union) if len(union) > 0 else 0.0
    hit = len(inter) / len(R_set) if len(R_set) > 0 else 0.0
    fa  = (len(R_set) - len(inter)) / len(R_set) if len(R_set) > 0 else 0.0
    cov = len(inter) / len(L_set) if len(L_set) > 0 else 0.0
    metrics['voxel_iou'] = iou
    metrics['radar_hit_rate'] = hit
    metrics['radar_false_alarm_rate'] = fa
    metrics['lidar_coverage_by_radar'] = cov
    print(f"Voxel IoU: {iou:.4f}")
    print(f"Radar hit rate: {hit:.4f}")
    print(f"Radar false alarm rate: {fa:.4f}")
    print(f"LiDAR coverage by radar: {cov:.4f}")

    # Derive occupancy-style confusion matrix using cell universe consistent with DC drop
    try:
        total_cells = int(max(0, (num_azimuth_bins - 1)) * max(0, (num_elevation_bins - 1)) * max(0, num_adc))
        tp = int(len(inter))
        fp = int(len(R_set) - tp)
        fn = int(len(L_set) - tp)
        tn = int(max(0, total_cells - tp - fp - fn))
        precision_occ = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall_occ = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity_occ = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics.update({
            'occupancy_tp': tp,
            'occupancy_fp': fp,
            'occupancy_fn': fn,
            'occupancy_tn': tn,
            'occupancy_total_cells': total_cells,
            'occupancy_precision': precision_occ,
            'occupancy_recall': recall_occ,
            'occupancy_sensitivity': recall_occ,
            'occupancy_specificity': specificity_occ,
        })
        print("Occupancy metrics (LiDAR as reference):")
        print(f"  Precision:  {precision_occ:.4f}")
        print(f"  Recall:     {recall_occ:.4f} (Sensitivity)")
        print(f"  Specificity:{specificity_occ:.4f}")
    except Exception as e:
        print(f"[warn] Failed to compute occupancy precision/recall/specificity: {e}")
    return metrics


def compute_distance_to_mesh_metrics_one_sided(radar_pts: np.ndarray,
                                               lidar_vox_pts: np.ndarray,
                                               lidar_raw_pts: np.ndarray,
                                               mesh,
                                               mesh_sample_count: int = 200000,
                                               enable_mesh_filter: bool = False,
                                               mesh_filter_eps_m: float = 0.05) -> dict:
    metrics: dict = {}
    print("\n2) DISTANCE TO MESH METRICS (one-sided)")
    print("-" * 40)

    def _build_nn(mesh_geom, samples: int):
        try:
            samples = max(1000, int(samples))
        except Exception:
            samples = 200000
        try:
            mesh_samples = mesh_geom.sample_points_uniformly(number_of_points=samples)
            mesh_pts_l = np.asarray(mesh_samples.points, dtype=np.float32)
        except Exception as e:
            print(f"[warn] Mesh sampling for metrics failed: {e}")
            mesh_pts_l = np.asarray([], dtype=np.float32).reshape(0, 3)
        if mesh_pts_l.shape[0] == 0:
            return None, None
        try:
            import faiss
            index_l = faiss.IndexFlatL2(3)
            index_l.add(mesh_pts_l)
            return ('faiss', index_l, mesh_pts_l)
        except Exception:
            try:
                tree_l = cKDTree(mesh_pts_l)
                return ('kdtree', tree_l, mesh_pts_l)
            except Exception as e2:
                print(f"[warn] NN structure build failed: {e2}")
                return None, None

    def _one_sided_mesh_distances(points_np: np.ndarray, nn_tuple):
        if points_np is None or points_np.shape[0] == 0 or nn_tuple is None or nn_tuple[0] is None:
            return None
        mode = nn_tuple[0]
        nn = nn_tuple[1]
        if mode == 'faiss':
            D, _ = nn.search(points_np.astype(np.float32), 1)
            dist = np.sqrt(D[:, 0])
        else:
            if not isinstance(nn, cKDTree):
                return None
            dist, _ = nn.query(points_np.astype(np.float32), k=1)
        if dist.size == 0:
            return None
        return dist.astype(np.float32)

    if mesh is None:
        print("Mesh not provided; skipping distance-to-mesh metrics")
        return metrics

    nn_struct = _build_nn(mesh, mesh_sample_count)
    # Optional: filter raw LiDAR by mesh proximity for fair reporting
    if enable_mesh_filter and lidar_raw_pts is not None and lidar_raw_pts.shape[0] > 0:
        d_raw = _one_sided_mesh_distances(lidar_raw_pts, nn_struct)
        if d_raw is not None:
            eps_val = float(mesh_filter_eps_m)
            keep_raw = d_raw <= eps_val
            kept_n = int(np.sum(keep_raw))
            if kept_n == 0:
                try:
                    eps_adaptive = float(np.percentile(d_raw, 1.0))
                    keep_raw = d_raw <= eps_adaptive
                    kept_n = int(np.sum(keep_raw))
                    print(f"[LiDAR raw] No points within epsilon={eps_val:.3f}m; relaxed to epsilon_adapt={eps_adaptive:.3f}m, kept {kept_n}/{lidar_raw_pts.shape[0]}")
                except Exception:
                    print(f"[LiDAR raw] No points within epsilon={eps_val:.3f}m and adaptive relaxation failed")
            else:
                print(f"[LiDAR raw] Mesh proximity filter epsilon={eps_val:.3f}m: kept {kept_n}/{lidar_raw_pts.shape[0]} points")
            if kept_n > 0:
                lidar_raw_pts = lidar_raw_pts[keep_raw]

    def _emit(label: str, pts: np.ndarray, key_prefix: str):
        d = _one_sided_mesh_distances(pts, nn_struct)
        if d is None:
            print(f"{label}: N/A (no points or mesh)")
            metrics[f'{key_prefix}_count'] = 0
            metrics[f'{key_prefix}_l1_mean'] = float('inf')
            metrics[f'{key_prefix}_l2_rmse'] = float('inf')
            return
        l1_mean = float(np.mean(np.abs(d)))
        l2_rmse = float(np.sqrt(np.mean(d**2)))
        l1_median = float(np.median(np.abs(d)))
        l1_p95 = float(np.percentile(np.abs(d), 95))
        metrics[f'{key_prefix}_count'] = int(pts.shape[0]) if pts is not None else 0
        metrics[f'{key_prefix}_l1_mean'] = l1_mean
        metrics[f'{key_prefix}_l1_median'] = l1_median
        metrics[f'{key_prefix}_l1_p95'] = l1_p95
        metrics[f'{key_prefix}_l2_rmse'] = l2_rmse
        print(f"{label}: L1(mean)={l1_mean:.4f} m, L1(median)={l1_median:.4f} m, L1(p95)={l1_p95:.4f} m, L2(RMSE)={l2_rmse:.4f} m")

    if radar_pts is not None:
        _emit('Radar voxels -> Mesh', radar_pts, 'radar_mesh')
    if lidar_vox_pts is not None:
        _emit('LiDAR voxelized -> Mesh', lidar_vox_pts, 'lidar_vox_mesh')
    if lidar_raw_pts is not None:
        _emit('LiDAR raw -> Mesh', lidar_raw_pts, 'lidar_raw_mesh')

    return metrics


def compute_point_cloud_rmse_and_chamfer(radar_pts: np.ndarray,
                                          lidar_pts: np.ndarray) -> dict:
    """
    Compute RMSE and Relative Chamfer Distance between radar and lidar point clouds.

    RMSE: Root Mean Square Error between corresponding nearest neighbors
    R-CD: Relative Chamfer Distance (bidirectional nearest neighbor distance)
    """
    metrics: dict = {}
    print("\n3) POINT CLOUD RMSE AND CHAMFER DISTANCE")
    print("-" * 40)

    if radar_pts is None or lidar_pts is None or radar_pts.shape[0] == 0 or lidar_pts.shape[0] == 0:
        print("[warn] Insufficient points for RMSE/Chamfer; skipping")
        metrics.update({
            'pc_rmse': float('inf'),
            'chamfer_distance': float('inf'),
            'relative_chamfer_distance': float('inf')
        })
        return metrics

    try:
        # Build KD-trees for efficient nearest neighbor search
        from scipy.spatial import cKDTree
        radar_tree = cKDTree(radar_pts)
        lidar_tree = cKDTree(lidar_pts)

        # Compute nearest neighbor distances: radar -> lidar
        dist_r2l, _ = lidar_tree.query(radar_pts, k=1)

        # Compute nearest neighbor distances: lidar -> radar
        dist_l2r, _ = radar_tree.query(lidar_pts, k=1)

        # RMSE: average of bidirectional RMSE
        rmse_r2l = float(np.sqrt(np.mean(dist_r2l**2)))
        rmse_l2r = float(np.sqrt(np.mean(dist_l2r**2)))
        rmse = (rmse_r2l + rmse_l2r) / 2.0

        # Chamfer Distance (bidirectional)
        chamfer_r2l = float(np.mean(dist_r2l))
        chamfer_l2r = float(np.mean(dist_l2r))
        chamfer = chamfer_r2l + chamfer_l2r

        # Relative Chamfer Distance (normalized by scene scale)
        # Scene scale = max extent of lidar point cloud
        lidar_extent = float(np.max(np.ptp(lidar_pts, axis=0)))
        relative_chamfer = chamfer / lidar_extent if lidar_extent > 0 else float('inf')

        metrics['pc_rmse'] = rmse
        metrics['pc_rmse_r2l'] = rmse_r2l
        metrics['pc_rmse_l2r'] = rmse_l2r
        metrics['chamfer_distance'] = chamfer
        metrics['chamfer_r2l'] = chamfer_r2l
        metrics['chamfer_l2r'] = chamfer_l2r
        metrics['relative_chamfer_distance'] = relative_chamfer
        metrics['lidar_scene_extent'] = lidar_extent

        print(f"RMSE (bidirectional avg): {rmse:.4f} m")
        print(f"  RMSE radar->lidar: {rmse_r2l:.4f} m")
        print(f"  RMSE lidar->radar: {rmse_l2r:.4f} m")
        print(f"Chamfer Distance: {chamfer:.4f} m")
        print(f"  Chamfer radar->lidar: {chamfer_r2l:.4f} m")
        print(f"  Chamfer lidar->radar: {chamfer_l2r:.4f} m")
        print(f"Relative Chamfer Distance: {relative_chamfer:.4f}")
        print(f"LiDAR scene extent: {lidar_extent:.4f} m")

    except Exception as e:
        print(f"[error] Failed to compute RMSE/Chamfer: {e}")
        metrics.update({
            'pc_rmse': float('inf'),
            'chamfer_distance': float('inf'),
            'relative_chamfer_distance': float('inf')
        })

    return metrics


def compute_intensity_agreement_on_overlap(radar_keys: np.ndarray,
                                           radar_vals: np.ndarray,
                                           lidar_keys: np.ndarray,
                                           lidar_vals: np.ndarray,
                                           label_suffix: str = "") -> dict:
    metrics: dict = {}
    print("\n3) INTENSITY AGREEMENT ON OVERLAP" + (f" ({label_suffix})" if label_suffix else ""))
    print("-" * 40)
    try:
        rk = np.asarray(radar_keys if radar_keys is not None else [])
        rv = np.asarray(radar_vals if radar_vals is not None else [])
        lk = np.asarray(lidar_keys if lidar_keys is not None else [])
        lv = np.asarray(lidar_vals if lidar_vals is not None else [])
        if rk.size == 0 or rv.size == 0 or lk.size == 0 or lv.size == 0:
            print("Insufficient intensity data; skipping intensity overlap metrics")
            metrics['intensity_overlap_count'] = 0
            return metrics
        rk_idx = np.argsort(rk)
        lk_idx = np.argsort(lk)
        rk_s, rv_s = rk[rk_idx], rv[rk_idx]
        lk_s, lv_s = lk[lk_idx], lv[lk_idx]
        inter = np.intersect1d(rk_s, lk_s)
        if inter.size == 0:
            print("No overlapping occupied voxels between modalities for intensity metrics")
            metrics['intensity_overlap_count'] = 0
            return metrics

        def _gather(keys_sorted, vals_sorted, keys):
            pos = np.searchsorted(keys_sorted, keys)
            return vals_sorted[pos]

        radar_overlap_raw = _gather(rk_s, rv_s, inter).astype(np.float64)
        lidar_overlap_raw = _gather(lk_s, lv_s, inter).astype(np.float64)

        # Normalize using global min/max of the entire arrays (not just overlap)
        # LiDAR values are already in [0,1], but we still normalize against global min/max for consistency
        rv_all = rv.astype(np.float64)
        lv_all = lv.astype(np.float64)
        print(f"[debug:intensity] rv_all min/max={float(rv_all.min()):.6f}/{float(rv_all.max()):.6f} | lv_all min/max={float(lv_all.min()):.6f}/{float(lv_all.max()):.6f}")
        rv_min = float(np.min(rv_all)) if rv_all.size > 0 else 0.0
        rv_max = float(np.max(rv_all)) if rv_all.size > 0 else 1.0
        lv_min = float(np.min(lv_all)) if lv_all.size > 0 else 0.0
        lv_max = float(np.max(lv_all)) if lv_all.size > 0 else 1.0
        def _normalize_minmax(x, lo, hi):
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.zeros_like(x, dtype=np.float64)
            x_n = (x - lo) / (hi - lo)
            return np.clip(x_n.astype(np.float64), 0.0, 1.0)
        radar_overlap = _normalize_minmax(radar_overlap_raw, rv_min, rv_max)
        lidar_overlap = _normalize_minmax(lidar_overlap_raw, lv_min, lv_max)
        n = radar_overlap.size
        metrics['intensity_overlap_count'] = int(n)
        if n == 0:
            print("No overlapping intensity samples after mapping")
            return metrics
        diff = radar_overlap - lidar_overlap
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        medae = float(np.median(np.abs(diff)))
        delta = 0.1
        absd = np.abs(diff)
        huber = np.where(absd <= delta, 0.5 * diff ** 2, delta * (absd - 0.5 * delta))
        huber_mean = float(np.mean(huber))
        # Correlation metrics (guarded)
        try:
            from scipy.stats import pearsonr, spearmanr as _spearmanr
            unique_r = np.unique(radar_overlap); unique_l = np.unique(lidar_overlap)
            if unique_r.size < 2 or unique_l.size < 2:
                # One or both signals are constant over the overlap; correlations undefined
                pr, sr = 0.0, 0.0
                print("[info] Intensity overlap has a constant signal; Pearson/Spearman set to 0.0")
            else:
                pr, _ = pearsonr(radar_overlap, lidar_overlap)
                sr, _ = _spearmanr(radar_overlap, lidar_overlap)
        except Exception:
            pr, sr = float('nan'), float('nan')
        metrics.update({
            'intensity_mae': mae,
            'intensity_mse': mse,
            'intensity_median_abs_error': medae,
            'intensity_huber_mean': huber_mean,
            'intensity_pearson_r': float(pr),
            'intensity_spearman_rho': float(sr),
        })
        print(f"Overlap={n} | MAE={mae:.4f}, MSE={mse:.4f}, MedAE={medae:.4f}, Huber(delta=0.1)={huber_mean:.4f}, Pearson r={pr:.4f}, Spearman rho={sr:.4f}")
    except Exception as e:
        print(f"[warn] Intensity agreement metrics failed: {e}")
    return metrics


def eval_metrics(radar_pcd,
                 lidar_pcd_raw,
                 lidar_pcd_voxelized,
                 range_resolution: float,
                 num_azimuth_bins: int,
                 num_elevation_bins: int,
                 num_adc: int,
                 lidar_intensity_pcd_raw=None,
                 lidar_intensity_pcd_voxelized=None,
                 radar_voxel_keys=None,
                 lidar_voxel_keys=None,
                 mesh=None,
                 mesh_sample_count: int = 200000,
                 enable_mesh_filter: bool = False,
                 mesh_filter_eps_m: float = 0.05,
                 radar_keys_for_intensity=None,
                 radar_vals_for_intensity=None,
                 lidar_keys_for_intensity=None,
                 lidar_intensity_keys=None,
                 lidar_vals_for_intensity=None) -> dict:
    """High-level wrapper that calls the three composable metric groups."""
    metrics_all: dict = {}
    # Occupancy on polar keys
    occ = compute_voxel_occupancy_metrics_polar(radar_voxel_keys, lidar_voxel_keys,
                                                num_azimuth_bins, num_elevation_bins, num_adc)
    metrics_all.update(occ)

    # Distance-to-mesh metrics
    radar_pts = np.asarray(radar_pcd.points, dtype=np.float32) if radar_pcd is not None else np.empty((0, 3), dtype=np.float32)
    lidar_raw_pts = np.asarray(lidar_pcd_raw.points, dtype=np.float32) if lidar_pcd_raw is not None else np.empty((0, 3), dtype=np.float32)
    lidar_vox_pts = np.asarray(lidar_pcd_voxelized.points, dtype=np.float32) if lidar_pcd_voxelized is not None else np.empty((0, 3), dtype=np.float32)
    dmesh = compute_distance_to_mesh_metrics_one_sided(radar_pts, lidar_vox_pts, lidar_raw_pts,
                                                       mesh, mesh_sample_count, enable_mesh_filter, mesh_filter_eps_m)
    metrics_all.update(dmesh)

    # Intensity agreement
    # Prefer the single-frame intensity LiDAR keys if provided; fallback to regular LiDAR keys
    inten_lidar_keys = lidar_intensity_keys if lidar_intensity_keys is not None and np.size(lidar_intensity_keys) > 0 else lidar_keys_for_intensity
    # Compute linear-scale radar intensity agreement
    inten_lin = compute_intensity_agreement_on_overlap(radar_keys_for_intensity, radar_vals_for_intensity,
                                                       inten_lidar_keys, lidar_vals_for_intensity, label_suffix="radar-linear")
    metrics_all.update({f"{k}_linear": v for k, v in inten_lin.items()})
    # Compute log1p-scale radar intensity agreement
    try:
        rv_log = np.log1p(np.asarray(radar_vals_for_intensity, dtype=np.float64))
    except Exception:
        rv_log = np.asarray(radar_vals_for_intensity, dtype=np.float64)
    inten_log = compute_intensity_agreement_on_overlap(radar_keys_for_intensity, rv_log,
                                                       inten_lidar_keys, lidar_vals_for_intensity, label_suffix="radar-log1p")
    metrics_all.update({f"{k}_log": v for k, v in inten_log.items()})

    print("\n4) SUMMARY")
    print("-" * 40)
    print(f"Total metrics computed: {len(metrics_all)}")
    print(f"Key metrics:")
    print(f"  Voxel IoU: {metrics_all.get('voxel_iou', 0.0):.4f}")
    print(f"  Radar hit rate: {metrics_all.get('radar_hit_rate', 0.0):.4f}")
    print(f"  Chamfer median: {metrics_all.get('chamfer_median', float('inf')):.4f} m")
    print(f"  F-score @ tau: {metrics_all.get('fscore_at_tau', 0.0):.4f}")
    print(f"  Angular IoU: {metrics_all.get('angular_iou', 0.0):.4f}")

    return metrics_all


################################################################################
# EVALUATION METRICS
################################################################################
def evaluate_radar_vs_lidar_metrics(radar_pcd, lidar_pcd_raw, lidar_pcd_voxelized,
                                   range_resolution, num_azimuth_bins, num_elevation_bins,
                                   num_adc, voxel_size_m=0.05,
                                   radar_voxel_keys=None, lidar_voxel_keys=None,
                                   mesh=None, mesh_sample_count: int = 200000,
                                   enable_mesh_filter: bool = False,
                                   mesh_filter_eps_m: float = 0.05,
                                   radar_keys_for_intensity=None,
                                   radar_vals_for_intensity=None,
                                   lidar_keys_for_intensity=None,
                                   lidar_vals_for_intensity=None,
                                   radar_center=None,
                                   radar_boresight=None,
                                   use_mmwcas_patterns: bool = True):
    """
    Comprehensive evaluation metrics comparing radar and LiDAR point clouds.

    Computes two sets of comparisons:
    #1: Thresholded voxelized radar vs voxelized LiDAR
    #2: Thresholded voxelized radar vs filtered original LiDAR (antenna beam + intensity)
    """
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    import numpy as np

    print("\n" + "="*80)
    print("RADAR vs LIDAR EVALUATION METRICS")
    print("="*80)

    metrics = {}

    # Get point arrays
    radar_pts = np.asarray(radar_pcd.points, dtype=np.float32)
    lidar_raw_pts = np.asarray(lidar_pcd_raw.points, dtype=np.float32)
    lidar_vox_pts = np.asarray(lidar_pcd_voxelized.points, dtype=np.float32)

    print(f"Radar points: {len(radar_pts)}")
    print(f"LiDAR raw points: {len(lidar_raw_pts)}")
    print(f"LiDAR voxelized points: {len(lidar_vox_pts)}")

    # ============================================================================
    # COMPARISON #1: Radar vs Voxelized LiDAR
    # ============================================================================
    print("\n" + "="*80)
    print("="*80)
    print("COMPARISON #1: Thresholded Voxelized Radar vs Voxelized LiDAR")
    print("="*80)

    # 1) VOXEL-BASED OCCUPANCY METRICS (polar keys)
    print("\n1) VOXEL-BASED OCCUPANCY METRICS (polar keys)")
    print("-" * 40)
    if radar_voxel_keys is not None and lidar_voxel_keys is not None:
        R_set = set(map(int, np.unique(radar_voxel_keys.astype(np.int64))))
        L_set = set(map(int, np.unique(lidar_voxel_keys.astype(np.int64))))
        print(f"Radar occupied cells: {len(R_set)}")
        print(f"LiDAR occupied cells: {len(L_set)}")
        inter = R_set & L_set
        union = R_set | L_set
        iou = len(inter) / len(union) if len(union) > 0 else 0.0
        hit = len(inter) / len(R_set) if len(R_set) > 0 else 0.0
        fa  = (len(R_set) - len(inter)) / len(R_set) if len(R_set) > 0 else 0.0
        cov = len(inter) / len(L_set) if len(L_set) > 0 else 0.0
        metrics['voxel_iou'] = iou
        metrics['radar_hit_rate'] = hit
        metrics['radar_false_alarm_rate'] = fa
        metrics['lidar_coverage_by_radar'] = cov
        print(f"Voxel IoU: {iou:.4f}")
        print(f"Radar hit rate: {hit:.4f}")
        print(f"Radar false alarm rate: {fa:.4f}")
        print(f"LiDAR coverage by radar: {cov:.4f}")

        # Traditional occupancy-based metrics
        try:
            total_cells = int(max(0, (num_azimuth_bins - 1)) * max(0, (num_elevation_bins - 1)) * max(0, num_adc))
            tp = int(len(inter))
            fp = int(len(R_set) - tp)
            fn = int(len(L_set) - tp)
            tn = int(max(0, total_cells - tp - fp - fn))

            precision_occ = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_occ = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity_occ = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            accuracy_occ = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

            metrics['vox_occupancy_tp'] = tp
            metrics['vox_occupancy_fp'] = fp
            metrics['vox_occupancy_fn'] = fn
            metrics['vox_occupancy_tn'] = tn
            metrics['vox_occupancy_total_cells'] = total_cells
            metrics['vox_occupancy_precision'] = precision_occ
            metrics['vox_occupancy_recall'] = recall_occ
            metrics['vox_occupancy_sensitivity'] = recall_occ
            metrics['vox_occupancy_specificity'] = specificity_occ
            metrics['vox_occupancy_accuracy'] = accuracy_occ

            print("Occupancy metrics (LiDAR as reference):")
            print(f"  Precision:  {precision_occ:.4f}")
            print(f"  Recall:     {recall_occ:.4f} (Sensitivity)")
            print(f"  Specificity:{specificity_occ:.4f}")
            print(f"  Accuracy:   {accuracy_occ:.4f}")
        except Exception as e:
            print(f"[warn] Failed to compute occupancy metrics: {e}")
    else:
        print("[warn] Polar keys not provided; skipping occupancy metrics")
        metrics.update({'voxel_iou': 0.0, 'radar_hit_rate': 0.0, 'radar_false_alarm_rate': 0.0, 'lidar_coverage_by_radar': 0.0})

    # 2) POINT CLOUD RMSE AND CHAMFER DISTANCE
    pc_metrics_vox = compute_point_cloud_rmse_and_chamfer(radar_pts, lidar_vox_pts)
    # Prefix keys for voxelized comparison
    for key, value in pc_metrics_vox.items():
        metrics[f'vox_{key}'] = value

    print("="*80)

    # ============================================================================
    # COMPARISON #2: Radar vs Filtered Original LiDAR
    # ============================================================================
    print("\n" + "="*80)
    print("="*80)
    print("COMPARISON #2: Thresholded Voxelized Radar vs Filtered Original LiDAR")
    print("="*80)

    # Filter original LiDAR by antenna beam pattern and intensity
    if radar_center is not None and radar_boresight is not None:
        print("\nFiltering original LiDAR by antenna beam pattern and intensity:")

        # Try to get LiDAR intensities (if available)
        lidar_intensities = None
        try:
            if hasattr(lidar_pcd_raw, 'colors') and lidar_pcd_raw.has_colors():
                # Convert colors to grayscale intensity
                colors = np.asarray(lidar_pcd_raw.colors)
                lidar_intensities = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
        except:
            pass

        lidar_filtered_pts, _ = filter_lidar_by_antenna_beam(
            lidar_raw_pts,
            radar_center,
            radar_boresight,
            use_mmwcas_patterns=use_mmwcas_patterns,
            intensity_percentile=10.0,
            lidar_intensities=lidar_intensities,
            radar_keys=radar_keys_for_intensity,
            radar_vals_agg=radar_vals_for_intensity,
            range_resolution=range_resolution,
            num_azimuth_bins=num_azimuth_bins,
            num_elevation_bins=num_elevation_bins,
            num_adc=num_adc,
            enable_radar_intensity_filter=True  # Enable Option C filtering
        )

        # 3) DISTANCE-BASED OCCUPANCY METRICS
        print("\n3) DISTANCE-BASED OCCUPANCY METRICS (tau=0.5m)")
        print("-" * 40)
        dist_metrics = compute_distance_based_occupancy_metrics(radar_pts, lidar_filtered_pts, distance_threshold=0.5)
        for key, value in dist_metrics.items():
            metrics[f'raw_{key}'] = value

        print(f"Matched radar points: {dist_metrics['matched_radar_points']} / {dist_metrics['total_radar_points']}")
        print(f"Matched LiDAR points: {dist_metrics['matched_lidar_points']} / {dist_metrics['total_lidar_points']}")
        print(f"Precision:   {dist_metrics['distance_precision']:.4f}")
        print(f"Recall:      {dist_metrics['distance_recall']:.4f}")
        print(f"Specificity: {dist_metrics['distance_specificity']:.4f}")
        print(f"Accuracy:    {dist_metrics['distance_accuracy']:.4f}")

        # 4) POINT CLOUD RMSE AND CHAMFER DISTANCE
        pc_metrics_raw = compute_point_cloud_rmse_and_chamfer(radar_pts, lidar_filtered_pts)
        # Prefix keys for raw comparison
        for key, value in pc_metrics_raw.items():
            metrics[f'raw_{key}'] = value

        print("="*80)
    else:
        print("\n[warn] Radar center/boresight not provided; skipping original LiDAR comparison")
        print("="*80)

    # ============================================================================
    # COMPARISON #3: Radar vs Filtered + Voxelized LiDAR (POLAR)
    # ============================================================================
    print("\n" + "="*80)
    print("="*80)
    print("COMPARISON #3: Thresholded Voxelized Radar vs Filtered + Voxelized LiDAR (POLAR)")
    print("="*80)
    print("\nThis comparison applies antenna beam + radar intensity filtering to LiDAR")
    print("BEFORE voxelization, then voxelizes to POLAR grid (same as COMPARISON #1).")

    if radar_center is not None and radar_boresight is not None and 'lidar_filtered_pts' in locals():
        print(f"\nFiltered LiDAR points: {len(lidar_filtered_pts)}")

        # Convert filtered LiDAR points to polar voxel keys (same grid as radar)
        lidar_filtered_polar_keys = lidar_points_to_polar_voxel_keys(
            lidar_filtered_pts,
            radar_center,
            radar_boresight,
            range_resolution,
            num_azimuth_bins,
            num_elevation_bins,
            num_adc
        )
        lidar_filtered_polar_keys_unique = np.unique(lidar_filtered_polar_keys)

        # Use SAME radar polar voxel keys as COMPARISON #1
        print(f"Radar occupied cells (polar, same as COMPARISON #1): {len(np.unique(radar_voxel_keys))}")
        print(f"Filtered LiDAR occupied cells (polar): {len(lidar_filtered_polar_keys_unique)}")

        # 1) VOXEL-BASED OCCUPANCY METRICS (POLAR keys)
        print("\n1) VOXEL-BASED OCCUPANCY METRICS (polar voxel keys)")
        print("-" * 40)

        # Get unique voxel keys for occupancy comparison
        R_set = set(map(int, np.unique(radar_voxel_keys.astype(np.int64))))
        L_set = set(map(int, lidar_filtered_polar_keys_unique.astype(np.int64)))

        inter = R_set & L_set
        union = R_set | L_set

        iou = len(inter) / len(union) if len(union) > 0 else 0.0
        hit = len(inter) / len(R_set) if len(R_set) > 0 else 0.0
        fa  = (len(R_set) - len(inter)) / len(R_set) if len(R_set) > 0 else 0.0
        cov = len(inter) / len(L_set) if len(L_set) > 0 else 0.0

        metrics['filtered_vox_iou'] = iou
        metrics['filtered_radar_hit_rate'] = hit
        metrics['filtered_radar_false_alarm_rate'] = fa
        metrics['filtered_lidar_coverage_by_radar'] = cov

        print(f"Voxel IoU: {iou:.4f}")
        print(f"Radar hit rate: {hit:.4f}")
        print(f"Radar false alarm rate: {fa:.4f}")
        print(f"Filtered LiDAR coverage by radar: {cov:.4f}")

        # Traditional occupancy-based metrics
        try:
            # For polar voxels, total cells is well-defined
            total_cells = int(max(0, (num_azimuth_bins - 1)) * max(0, (num_elevation_bins - 1)) * max(0, num_adc))

            tp = int(len(inter))
            fp = int(len(R_set) - tp)
            fn = int(len(L_set) - tp)
            tn = int(max(0, total_cells - tp - fp - fn))

            precision_occ = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_occ = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity_occ = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            accuracy_occ = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

            metrics['filtered_vox_occupancy_tp'] = tp
            metrics['filtered_vox_occupancy_fp'] = fp
            metrics['filtered_vox_occupancy_fn'] = fn
            metrics['filtered_vox_occupancy_tn'] = tn
            metrics['filtered_vox_occupancy_total_cells'] = total_cells
            metrics['filtered_vox_occupancy_precision'] = precision_occ
            metrics['filtered_vox_occupancy_recall'] = recall_occ
            metrics['filtered_vox_occupancy_sensitivity'] = recall_occ
            metrics['filtered_vox_occupancy_specificity'] = specificity_occ
            metrics['filtered_vox_occupancy_accuracy'] = accuracy_occ

            print("Occupancy metrics (Filtered LiDAR as reference):")
            print(f"  Precision:  {precision_occ:.4f}")
            print(f"  Recall:     {recall_occ:.4f} (Sensitivity)")
            print(f"  Specificity:{specificity_occ:.4f}")
            print(f"  Accuracy:   {accuracy_occ:.4f}")
            print(f"  Total polar cells: {total_cells}")
        except Exception as e:
            print(f"[warn] Failed to compute filtered occupancy metrics: {e}")

        # 2) POINT CLOUD RMSE AND CHAMFER DISTANCE
        print("\n2) POINT CLOUD RMSE AND CHAMFER DISTANCE")
        print("-" * 40)
        # Use original radar points vs filtered LiDAR points (both point clouds, not voxelized)
        pc_metrics_filtered = compute_point_cloud_rmse_and_chamfer(
            radar_pts, lidar_filtered_pts
        )
        # Prefix keys for filtered comparison
        for key, value in pc_metrics_filtered.items():
            metrics[f'filtered_vox_{key}'] = value

        print("="*80)
    else:
        print("\n[warn] Radar center/boresight not provided or filtered LiDAR not available; skipping COMPARISON #3")
        print("="*80)

    # ============================================================================
    # 3) DISTANCE TO MESH METRICS (one-sided; points -> mesh)
    # ============================================================================
    print("\n3) DISTANCE TO MESH METRICS (one-sided)")
    print("-" * 40)
    def _build_nn(mesh_geom, samples: int):
        try:
            samples = max(1000, int(samples))
        except Exception:
            samples = 200000
        try:
            mesh_samples = mesh_geom.sample_points_uniformly(number_of_points=samples)
            mesh_pts_l = np.asarray(mesh_samples.points, dtype=np.float32)
        except Exception as e:
            print(f"[warn] Mesh sampling for metrics failed: {e}")
            mesh_pts_l = np.asarray([], dtype=np.float32).reshape(0, 3)
        if mesh_pts_l.shape[0] == 0:
            return None, None
        # Prefer FAISS; fallback to KDTree
        try:
            import faiss
            index_l = faiss.IndexFlatL2(3)
            index_l.add(mesh_pts_l)
            return ('faiss', index_l, mesh_pts_l)
        except Exception:
            try:
                from scipy.spatial import cKDTree
                tree_l = cKDTree(mesh_pts_l)
                return ('kdtree', tree_l, mesh_pts_l)
            except Exception as e2:
                print(f"[warn] NN structure build failed: {e2}")
                return None, None

    def _one_sided_mesh_distances(points_np: np.ndarray, nn_tuple):
        if points_np is None or points_np.shape[0] == 0 or nn_tuple is None or nn_tuple[0] is None:
            return None
        mode = nn_tuple[0]
        nn = nn_tuple[1]
        if mode == 'faiss':
            D, _ = nn.search(points_np.astype(np.float32), 1)
            dist = np.sqrt(D[:, 0])
        else:
            # KDTree
            from scipy.spatial import cKDTree
            if not isinstance(nn, cKDTree):
                return None
            dist, _ = nn.query(points_np.astype(np.float32), k=1)
        if dist.size == 0:
            return None
        return dist.astype(np.float32)

    if mesh is not None:
        nn_struct = _build_nn(mesh, mesh_sample_count)

        def _emit_pointset_metrics(label: str, pts: np.ndarray, key_prefix: str):
            d = _one_sided_mesh_distances(pts, nn_struct)
            if d is None:
                print(f"{label}: N/A (no points or mesh)")
                metrics[f'{key_prefix}_count'] = 0
                metrics[f'{key_prefix}_l1_mean'] = float('inf')
                metrics[f'{key_prefix}_l2_rmse'] = float('inf')
                return
            l1_mean = float(np.mean(np.abs(d)))
            l2_rmse = float(np.sqrt(np.mean(d**2)))
            l1_median = float(np.median(np.abs(d)))
            l1_p95 = float(np.percentile(np.abs(d), 95))
            metrics[f'{key_prefix}_count'] = int(pts.shape[0]) if pts is not None else 0
            metrics[f'{key_prefix}_l1_mean'] = l1_mean
            metrics[f'{key_prefix}_l1_median'] = l1_median
            metrics[f'{key_prefix}_l1_p95'] = l1_p95
            metrics[f'{key_prefix}_l2_rmse'] = l2_rmse
            print(f"{label}: L1(mean)={l1_mean:.4f} m, L1(median)={l1_median:.4f} m, L1(p95)={l1_p95:.4f} m, L2(RMSE)={l2_rmse:.4f} m")

        _emit_pointset_metrics('Radar voxels -> Mesh', radar_pts, 'radar_mesh')
        _emit_pointset_metrics('LiDAR voxelized -> Mesh', lidar_vox_pts, 'lidar_vox_mesh')
        _emit_pointset_metrics('LiDAR raw -> Mesh', lidar_raw_pts, 'lidar_raw_mesh')
    else:
        print("Mesh not provided; skipping distance-to-mesh metrics")

    # ============================================================================
    # 5) INTENSITY AGREEMENT ON OVERLAP (normalized per modality)
    # ============================================================================
    print("\n5) INTENSITY AGREEMENT ON OVERLAP")
    print("-" * 40)
    try:
        rk = np.asarray(radar_keys_for_intensity if radar_keys_for_intensity is not None else [])
        rv = np.asarray(radar_vals_for_intensity if radar_vals_for_intensity is not None else [])
        lk = np.asarray(lidar_keys_for_intensity if lidar_keys_for_intensity is not None else [])
        lv = np.asarray(lidar_vals_for_intensity if lidar_vals_for_intensity is not None else [])
        try:
            if rv.size > 0 and lv.size > 0:
                print(f"[debug:intensity] rv_all min/max={float(rv.min()):.6f}/{float(rv.max()):.6f} | lv_all min/max={float(lv.min()):.6f}/{float(lv.max()):.6f}")
        except Exception:
            pass
        if rk.size == 0 or rv.size == 0 or lk.size == 0 or lv.size == 0:
            print("Insufficient intensity data; skipping intensity overlap metrics")
            metrics['intensity_overlap_count'] = 0
        else:
            rk_idx = np.argsort(rk)
            lk_idx = np.argsort(lk)
            rk_s, rv_s = rk[rk_idx], rv[rk_idx]
            lk_s, lv_s = lk[lk_idx], lv[lk_idx]
            inter = np.intersect1d(rk_s, lk_s)
            if inter.size == 0:
                print("No overlapping occupied voxels between modalities for intensity metrics")
                metrics['intensity_overlap_count'] = 0
            else:
                def _gather(keys_sorted, vals_sorted, keys):
                    pos = np.searchsorted(keys_sorted, keys)
                    return vals_sorted[pos]
                radar_overlap_raw = _gather(rk_s, rv_s, inter).astype(np.float64)
                lidar_overlap_raw = _gather(lk_s, lv_s, inter).astype(np.float64)
                try:
                    print(f"[debug:intensity] overlap={inter.size} | radar_raw min/max={float(radar_overlap_raw.min()):.6f}/{float(radar_overlap_raw.max()):.6f} uniques={int(np.unique(radar_overlap_raw).size)} | lidar_raw min/max={float(lidar_overlap_raw.min()):.6f}/{float(lidar_overlap_raw.max()):.6f} uniques={int(np.unique(lidar_overlap_raw).size)}")
                except Exception:
                    pass
                # Normalize per modality on the overlap only
                def _normalize01_robust(x):
                    x = x.astype(np.float64)
                    if x.size == 0:
                        return x
                    lo = float(np.percentile(x, 1.0))
                    hi = float(np.percentile(x, 99.0))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo = float(np.min(x)); hi = float(np.max(x))
                    if hi <= lo:
                        return np.zeros_like(x, dtype=np.float64)
                    x_n = (x - lo) / (hi - lo)
                    x_n = np.clip(x_n, 0.0, 1.0)
                    return x_n
                radar_overlap = _normalize01_robust(radar_overlap_raw)
                lidar_overlap = _normalize01_robust(lidar_overlap_raw)
                try:
                    print(f"[debug:intensity] normalized | radar uniques={int(np.unique(radar_overlap).size)} | lidar uniques={int(np.unique(lidar_overlap).size)} | radar min/max={float(radar_overlap.min()):.6f}/{float(radar_overlap.max()):.6f} | lidar min/max={float(lidar_overlap.min()):.6f}/{float(lidar_overlap.max()):.6f}")
                except Exception:
                    pass
                n = radar_overlap.size
                metrics['intensity_overlap_count'] = int(n)
                if n == 0:
                    print("No overlapping intensity samples after mapping")
                else:
                    diff = radar_overlap - lidar_overlap
                    mae = float(np.mean(np.abs(diff)))
                    mse = float(np.mean(diff ** 2))
                    medae = float(np.median(np.abs(diff)))
                    delta = 0.1
                    absd = np.abs(diff)
                    huber = np.where(absd <= delta, 0.5 * diff ** 2, delta * (absd - 0.5 * delta))
                    huber_mean = float(np.mean(huber))
                    # Correlation metrics
                    try:
                        from scipy.stats import pearsonr, spearmanr
                        unique_r = np.unique(radar_overlap)
                        unique_l = np.unique(lidar_overlap)
                        if unique_r.size < 2 or unique_l.size < 2:
                            pr, sr = 0.0, 0.0
                            print("[info] Intensity overlap has a constant signal; Pearson/Spearman set to 0.0")
                        else:
                            pr, _ = pearsonr(radar_overlap, lidar_overlap)
                            sr, _ = spearmanr(radar_overlap, lidar_overlap)
                    except Exception:
                        pr, sr = float('nan'), float('nan')
                    metrics['intensity_mae'] = mae
                    metrics['intensity_mse'] = mse
                    metrics['intensity_median_abs_error'] = medae
                    metrics['intensity_huber_mean'] = huber_mean
                    metrics['intensity_pearson_r'] = float(pr)
                    metrics['intensity_spearman_rho'] = float(sr)
                    print(f"Overlap={n} | MAE={mae:.4f}, MSE={mse:.4f}, MedAE={medae:.4f}, Huber(delta=0.1)={huber_mean:.4f}, Pearson r={pr:.4f}, Spearman rho={sr:.4f}")
    except Exception as e:
        print(f"[warn] Intensity agreement metrics failed: {e}")

    # ============================================================================
    # 6) SUMMARY STATISTICS
    # ============================================================================
    print("\n6) SUMMARY")
    print("-" * 40)
    print(f"Total metrics computed: {len(metrics)}")
    print(f"Key metrics (Voxelized LiDAR):")
    print(f"  Voxel IoU: {metrics.get('voxel_iou', 0.0):.4f}")
    print(f"  Radar hit rate: {metrics.get('radar_hit_rate', 0.0):.4f}")
    print(f"  RMSE (PC): {metrics.get('vox_pc_rmse', float('inf')):.4f} m")
    print(f"  R-CD: {metrics.get('vox_relative_chamfer_distance', float('inf')):.4f}")
    if 'raw_distance_precision' in metrics:
        print(f"Key metrics (Filtered Original LiDAR):")
        print(f"  Precision: {metrics.get('raw_distance_precision', 0.0):.4f}")
        print(f"  Recall: {metrics.get('raw_distance_recall', 0.0):.4f}")
        print(f"  RMSE (PC): {metrics.get('raw_pc_rmse', float('inf')):.4f} m")

    return metrics

