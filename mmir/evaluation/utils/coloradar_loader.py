"""ColoRadar CASCADE radar data loader for benchmark visualization.

Loads CASCADE heatmap data from ColoRadar dataset, converts to 3D point cloud,
and returns world-frame coordinates + intensities for Open3D rendering.

"""

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np

# Default paths (None = must be provided by caller)
DEFAULT_DATASET_DIR = None
DEFAULT_CALIB_DIR = None


def _ensure_coloradar_tools():
    """Add ColoRadar tools to sys.path if not already present."""
    tools_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "utils", "preproc", "ColoRadar_tools"
    )
    tools_path = os.path.abspath(tools_path)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)


def _polar_to_cartesian(r_bin, az_bin, el_bin, params):
    """Convert polar heatmap bin to Cartesian coordinates."""
    range_m = r_bin * params['range_bin_width']
    point = np.zeros(3)
    point[0] = (range_m
                * math.cos(params['elevation_bins'][el_bin])
                * math.cos(params['azimuth_bins'][az_bin]))
    point[1] = (range_m
                * math.cos(params['elevation_bins'][el_bin])
                * math.sin(params['azimuth_bins'][az_bin]))
    point[2] = (range_m
                * math.sin(params['elevation_bins'][el_bin]))
    return point


def _get_heatmap_points(params, min_range=10):
    """Calculate 3D point locations for heatmap bins."""
    pcl = np.zeros([params['num_elevation_bins'],
                    params['num_azimuth_bins'],
                    params['num_range_bins'] - min_range,
                    5])

    for range_idx in range(params['num_range_bins'] - min_range):
        for az_idx in range(params['num_azimuth_bins']):
            for el_idx in range(params['num_elevation_bins']):
                pcl[el_idx, az_idx, range_idx, :3] = _polar_to_cartesian(
                    range_idx + min_range, az_idx, el_idx, params)

    return pcl.reshape(-1, 5)


def _transform_pcl(pcl, T):
    """Apply rigid transformation to point cloud."""
    in_points = pcl[:, :3]
    in_points = np.concatenate((in_points, np.ones((in_points.shape[0], 1))), axis=1)
    out_points = np.dot(T, np.transpose(in_points))
    out_points = np.transpose(out_points[:3, :])
    if pcl.shape[1] > 3:
        out_points = np.concatenate((out_points, pcl[:, 3:]), axis=1)
    return out_points


def _parse_scene_name(scene_name: str) -> Tuple[int, int]:
    """Parse scene name into (seq_idx, frame_idx).

    E.g. "seq_1_frame_185" → (1, 185)
    """
    parts = scene_name.split("_")
    seq_idx = int(parts[1])
    frame_idx = int(parts[3])
    return seq_idx, frame_idx


def load_coloradar_for_scene(
    scene_name: str,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    calib_dir: str = DEFAULT_CALIB_DIR,
    intensity_threshold: float = 0.2,
    min_range: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load ColoRadar CASCADE heatmap data for a given scene.

    Args:
        scene_name: Our scene name (e.g. "seq_1_frame_185").
        dataset_dir: Base path to ColoRadar kitti sequence directory.
            The seq index is appended directly (e.g. dataset_dir + "1").
            Matches mmir.preprocessing.preproc --dataset-dir convention.
        calib_dir: Path to ColoRadar calibration directory (contains cascade/).
        intensity_threshold: Normalized intensity threshold (0-1). Points below
            this threshold are discarded.
        min_range: Minimum range bin to include.

    Returns:
        (radar_xyz, radar_intensities, sensor_position) — (N,3), (N,), and (3,)
        numpy arrays, or (None, None, None) if data cannot be loaded.
        sensor_position is the ColoRadar T_bs translation (cascade sensor
        position in the body-sensor frame).
    """
    seq_idx, frame_idx = _parse_scene_name(scene_name)
    seq_dir = dataset_dir + str(seq_idx)

    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"ColoRadar sequence not found: {seq_dir}")
    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"ColoRadar calib not found: {calib_dir}")

    _ensure_coloradar_tools()
    from dataset_loaders import get_heatmap, get_cascade_params
    from scipy.spatial.transform import Rotation

    # Get CASCADE parameters
    all_cascade_params = get_cascade_params(calib_dir)
    heatmap_params = all_cascade_params['heatmap']

    # Build extrinsic transformation T_bs
    T_radar_bs = np.eye(4)
    T_radar_bs[:3, 3] = heatmap_params['translation']
    T_radar_bs[:3, :3] = Rotation.from_quat(heatmap_params['rotation']).as_matrix()

    # Pre-calculate heatmap point locations
    radar_pc_precalc = _get_heatmap_points(heatmap_params, min_range=min_range)

    # Load heatmap
    radar_hm = get_heatmap(frame_idx, seq_dir, heatmap_params)
    if radar_hm is None:
        raise ValueError(f"Failed to load CASCADE heatmap for frame {frame_idx}")

    # Assign intensity/doppler to pre-calculated points
    radar_pc_precalc[:, 3:] = radar_hm[:, :, min_range:, :].reshape(-1, 2)

    # Normalize and threshold (from ColoRadar animate_plot)
    radar_pc_local = radar_pc_precalc.copy()
    radar_pc_local[:, 3] -= radar_pc_local[:, 3].min()
    if radar_pc_local[:, 3].max() > 0:
        radar_pc_local[:, 3] /= radar_pc_local[:, 3].max()
    radar_pc_local = radar_pc_local[radar_pc_local[:, 3] > intensity_threshold]

    if len(radar_pc_local) == 0:
        return None, None, None

    # Re-normalize after thresholding
    radar_pc_local[:, 3] -= radar_pc_local[:, 3].min()
    if radar_pc_local[:, 3].max() > 0:
        radar_pc_local[:, 3] /= radar_pc_local[:, 3].max()

    # Transform to world frame
    radar_pc_world = _transform_pcl(radar_pc_local, T_radar_bs)

    radar_xyz = radar_pc_world[:, :3]
    radar_intensities = radar_pc_world[:, 3]
    sensor_position = np.array(heatmap_params['translation'])

    return radar_xyz, radar_intensities, sensor_position
