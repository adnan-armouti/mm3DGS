import os
import math
import random
import numpy as np
from numpy.linalg import eigh
import open3d as o3d
import ipywidgets as widgets
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ipywidgets import interact, interactive, fixed, interact_manual
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
import socket
import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes as skimage_marching_cubes
from typing import Literal, Optional
from mmir.preprocessing.ColoRadar_tools.dataset_loaders import (
    get_pointcloud,
    get_lidar_params,
    get_groundtruth_params,
    get_timestamps,
    get_groundtruth,
    get_heatmap,
    get_cascade_params,
    get_vicon_params,
    get_vicon,
)
from mmir.preprocessing.ColoRadar_tools.plot_pointclouds import (
    transform_pcl,
    interpolate_poses,
    downsample_pointcloud,
    polar_to_cartesian,
)
from mmir.preprocessing.ColoRadar_tools.calibration import (
    apply_phase_calibration,
    apply_frequency_calibration,
)




def get_heatmap_points(params, min_range):
    # transform range-azimuth-elevation heatmap to pointcloud
    pcl = np.zeros([params['num_elevation_bins'], params['num_azimuth_bins'], params['num_range_bins'] - min_range, 5])

    for range_idx in range(params['num_range_bins'] - min_range):
        for az_idx in range(params['num_azimuth_bins']):
            for el_idx in range(params['num_elevation_bins']):
                pcl[el_idx,az_idx,range_idx,:3] = polar_to_cartesian(range_idx + min_range, az_idx, el_idx, params)

    pcl = pcl.reshape(-1,5)
    return pcl


def get_radar_fov_bbox(
        seq_path,
        calib_path,
        radar_frame_idx,
        min_range_bin   = 10,
        intensity_thr   = 0.0
):
    """
    Return an axis-aligned bounding box (world frame) that encloses all
    radar voxels of one Cascaded-radar heat-map frame.

    Parameters
    ----------
    seq_path : str
        Sequence directory (.../raw/kitti/.../1).
    calib_path : str
        Calibration directory (.../utils/calib).
    radar_frame_idx : int
        Index of the radar heat-map frame to use.
    min_range_bin : int
        Ignore heat-map bins < this value (default 10).
    intensity_thr : float
        Drop voxels once their **normalised** intensity <= thr (0.0 keeps all).

    Returns
    -------
    bbox : dict
        {"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ...,
         "z_min": ..., "z_max": ...}
    """

    # -------- 1. sensor params & extrinsic ----------------------------
    radar_params = get_cascade_params(calib_path)['heatmap']
    T_bs = np.eye(4)
    T_bs[:3, 3]  = radar_params['translation']
    T_bs[:3, :3] = Rotation.from_quat(radar_params['rotation']).as_matrix()
    radar_params['T_bs'] = T_bs

    # -------- 2. pose (world<-base) at this radar time -----------------
    gt_params  = get_groundtruth_params()
    gt_ts      = np.asarray(get_timestamps(seq_path, gt_params), dtype=float)
    radar_ts   = np.asarray(get_timestamps(seq_path, radar_params), dtype=float)
    gt_poses   = get_groundtruth(seq_path)

    radar_gt, _ = interpolate_poses(gt_poses, gt_ts, radar_ts)
    if radar_frame_idx >= len(radar_gt):
        raise IndexError("radar_frame_idx out of range")

    T_wb = radar_gt[radar_frame_idx]          # world <- base

    # -------- 3. build full voxel cloud (sensor frame) ----------------
    vox = get_heatmap_points(radar_params, min_range_bin)
    hm  = get_heatmap(radar_frame_idx, seq_path, radar_params)
    vox[:, 3:] = hm[:, :, min_range_bin:, :].reshape(-1, 2)

    # normalise intensity, threshold
    vox[:, 3] -= vox[:, 3].min()
    vox[:, 3] /= (vox[:, 3].ptp() + 1e-9)
    vox = vox[vox[:, 3] > intensity_thr]

    # -------- 4. sensor -> world --------------------------------------
    T_ws = T_wb @ T_bs
    vox_world = transform_pcl(vox, T_ws)

    # -------- 5. bounding box ----------------------------------------
    x_min, x_max = vox_world[:, 0].min(), vox_world[:, 0].max()
    y_min, y_max = vox_world[:, 1].min(), vox_world[:, 1].max()
    z_min, z_max = vox_world[:, 2].min(), vox_world[:, 2].max()

    return dict(
        x_min=float(x_min), x_max=float(x_max),
        y_min=float(y_min), y_max=float(y_max),
        z_min=float(z_min), z_max=float(z_max)
    )


def load_lidar_world_cloud(
        seq_path      : str,
        calib_path    : str,
        frame_idx     : int
):
    """
    Load the LiDAR point cloud for a given `frame_idx` and return it in the
    *world* coordinate frame.  The function does **all** of the following
    internally:

    1.  Reads calibration / extrinsics for the LiDAR sensor.
    2.  Loads ground-truth poses and interpolates the world<-base transform
        at the LiDAR time-stamp.
    3.  Optionally voxel-grid downsamples the raw cloud.
    4.  Applies the rigid transform and outputs `[x, y, z, intensity]`.

    Parameters
    ----------
    seq_path : str
        Sequence directory, e.g. .../raw/kitti/2_28_2021_outdoors_run/1.
    calib_path : str
        .../utils/calib directory that contains the sensor calibration files.
    frame_idx : int
        Index of the LiDAR frame to load (0-based).
    vox_size : float or None, optional
        If given, voxel-grid down-sample the cloud in the sensor frame
        *before* transforming to world.  Units: metres.

    Returns
    -------
    cloud_world : (N,4) ndarray
        LiDAR points `[x, y, z, intensity]` expressed in the world frame.
    """

    # -- 1. sensor parameters & extrinsic (base<-sensor) -----------------
    lidar_params = get_lidar_params(calib_path)
    T_bs = np.eye(4)
    T_bs[:3, 3]  = lidar_params['translation']
    T_bs[:3, :3] = Rotation.from_quat(lidar_params['rotation']).as_matrix()
    lidar_params['T_bs'] = T_bs

    # -- 2.  world<-base transform at LiDAR timestamp -------------------
    gt_params  = get_groundtruth_params()
    gt_ts      = np.asarray(get_timestamps(seq_path, gt_params),  dtype=float)
    lidar_ts   = np.asarray(get_timestamps(seq_path, lidar_params), dtype=float)
    gt_poses   = get_groundtruth(seq_path)

    lidar_gt, lidar_indices = interpolate_poses(gt_poses, gt_ts, lidar_ts)

    if frame_idx >= len(lidar_gt):
        raise IndexError("frame_idx outside the valid LiDAR timestamp range")

    T_wb = lidar_gt[frame_idx]            # world <- base

    # -- 3.  load raw point cloud (sensor frame) ------------------------
    cloud_local = get_pointcloud(frame_idx, seq_path, lidar_params)  # (N,4)

    # -- 4.  sensor -> world --------------------------------------------
    T_ws = T_wb @ lidar_params['T_bs']     # world <- sensor
    cloud_world = transform_pcl(cloud_local, T_ws)

    return cloud_world

def find_matching_lidar_frame(
        seq_path         : str,
        radar_params     : dict,
        lidar_params     : dict,
        radar_frame_idx  : int,
        max_time_diff=None
):
    """
    Return the LiDAR frame index whose time-stamp is closest to the selected
    radar frame.

    Parameters
    ----------
    seq_path : str
        Path to the sequence directory that contains the sensor sub-folders
        (.../raw/kitti/2_28_*_run/1, etc.).
    radar_params : dict
        Dictionary describing the radar sensor & data type
        (e.g. one element of `get_cascade_params(calib)['heatmap']`).
    lidar_params : dict
        Dictionary returned by `get_lidar_params(calib_dir)`.
    radar_frame_idx : int
        Index of the radar frame of interest (0-based).
    max_time_diff : float or None, optional
        If given, require |Deltat| <= `max_time_diff` seconds; otherwise raise
        `ValueError`.  Default *None* never raises.

    Returns
    -------
    lidar_frame_idx : int
        Index (0-based) of the matching LiDAR frame.

    Raises
    ------
    IndexError
        If `radar_frame_idx` is out of range.
    ValueError
        If `max_time_diff` is set and no LiDAR frame falls within it.
    """
    # --- pull time-stamp vectors straight from disk -------------------
    radar_ts = np.asarray(get_timestamps(seq_path, radar_params), dtype=float)
    lidar_ts = np.asarray(get_timestamps(seq_path, lidar_params), dtype=float)

    if radar_frame_idx < 0 or radar_frame_idx >= len(radar_ts):
        raise IndexError("radar_frame_idx out of range")

    ref = radar_ts[radar_frame_idx]
    diffs = np.abs(lidar_ts - ref)
    lidar_idx = int(np.argmin(diffs))

    if (max_time_diff is not None) and (diffs[lidar_idx] > max_time_diff):
        raise ValueError("No LiDAR frame within {:.3f}s (closest Deltat {:.3f}s)"
                         .format(max_time_diff, diffs[lidar_idx]))
    return lidar_idx

def _pack_pose(p, q):
    T = np.eye(4)
    T[:3, 3]  = p
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    return T

def _pose_at_time(poses, stamps, t_query):
    """
    Return SE(3) at arbitrary time; handles 0-, 1-, or multi-pose lists.
    """
    n = len(poses)
    if n == 0:                          # no data -> identity
        return np.eye(4)
    if n == 1:                          # single pose -> constant
        return _pack_pose(poses[0]['position'], poses[0]['orientation'])

    # clamp to valid range
    if t_query <= stamps[0] + 1e-9:
        return _pack_pose(poses[0]['position'], poses[0]['orientation'])
    if t_query >= stamps[-1] - 1e-9:
        return _pack_pose(poses[-1]['position'], poses[-1]['orientation'])

    i = np.searchsorted(stamps, t_query) - 1
    a = (t_query - stamps[i]) / (stamps[i+1] - stamps[i])

    p = (1-a)*poses[i]['position'] + a*poses[i+1]['position']
    r = Slerp([0,1], Rotation.from_quat(
              [poses[i]['orientation'], poses[i+1]['orientation']]))([a])[0]
    return _pack_pose(p, r.as_quat())

def bbox_to_lidar_sensor_frame(seq_path,
                               calib_path,
                               bbox_world,
                               lidar_frame_idx):
    """
    Transform a world-frame bbox into the LiDAR sensor frame of any
    LiDAR frame index -- robust to missing / sparse VICON files.
    """
    # ---------- LiDAR extrinsic --------------------------------------
    lidar_par = get_lidar_params(calib_path)
    T_bs = np.eye(4)
    T_bs[:3,3]  = lidar_par['translation']
    T_bs[:3,:3] = Rotation.from_quat(lidar_par['rotation']).as_matrix()
    lidar_par['T_bs'] = T_bs

    lidar_ts = np.asarray(get_timestamps(seq_path, lidar_par))
    if lidar_frame_idx >= len(lidar_ts):
        raise IndexError("lidar_frame_idx out of range")

    # ---------- choose pose source -----------------------------------
    use_vicon = False
    if os.path.isdir(os.path.join(seq_path, 'vicon')):
        vic_par = get_vicon_params(calib_path); vic_par['data_type'] = 'pose'
        vic_poses = get_vicon(seq_path) or []          # None -> []
        if len(vic_poses) >= 2:
            pose_par, poses = vic_par, vic_poses
            use_vicon = True

    if not use_vicon:                                  # fall back to GT
        pose_par = get_groundtruth_params()
        poses    = get_groundtruth(seq_path) or []

    pose_ts = np.asarray(get_timestamps(seq_path, pose_par))
    T_wb = _pose_at_time(poses, pose_ts, lidar_ts[lidar_frame_idx])
    T_ws = T_wb @ T_bs
    T_sw = np.linalg.inv(T_ws)

    # ---------- bbox corners, transform ------------------------------
    xs = [bbox_world['x_min'], bbox_world['x_max']]
    ys = [bbox_world['y_min'], bbox_world['y_max']]
    zs = [bbox_world['z_min'], bbox_world['z_max']]
    corners_w = np.array([[x,y,z,1] for x in xs for y in ys for z in zs])
    corners_s = (T_sw @ corners_w.T).T[:, :3]

    bbox_s = dict(
        x_min = float(corners_s[:,0].min()), x_max = float(corners_s[:,0].max()),
        y_min = float(corners_s[:,1].min()), y_max = float(corners_s[:,1].max()),
        z_min = float(corners_s[:,2].min()), z_max = float(corners_s[:,2].max())
    )
    return corners_s, bbox_s

def _pack(p, q):
    T = np.eye(4); T[:3,3] = p; T[:3,:3] = Rotation.from_quat(q).as_matrix()
    return T

def _pose_at_time(poses, stamps, t):
    """Return pose at arbitrary time; handles 0/1/>=2 poses robustly."""
    n = len(poses)
    if n == 0:
        return np.eye(4)
    if n == 1:
        return _pack(poses[0]['position'], poses[0]['orientation'])

    if t <= stamps[0] + 1e-9:
        return _pack(poses[0]['position'], poses[0]['orientation'])
    if t >= stamps[-1] - 1e-9:
        return _pack(poses[-1]['position'], poses[-1]['orientation'])

    i = np.searchsorted(stamps, t) - 1
    a = (t - stamps[i]) / (stamps[i+1] - stamps[i])
    p = (1-a)*poses[i]['position'] + a*poses[i+1]['position']

    r = Slerp([0,1], Rotation.from_quat(
              [poses[i]['orientation'], poses[i+1]['orientation']]))([a])[0]
    return _pack(p, r.as_quat())
# ----------------------------------------------------------------------
def shift_bbox_by_lidar_translation(
        seq_path,
        calib_path,
        bbox_world_368,      # dict from frame 368  (world frame)
        lidar_idx_368,       # 368
        lidar_idx_369        # 369
):
    """
    Translate bbox of LiDAR-368 into the *world* bbox of LiDAR-369 by
    using only the translation of the LiDAR base frame (no rotation).
    """

    # ----- 1. pick best pose source, read poses + stamps --------------
    if os.path.isdir(os.path.join(seq_path, 'vicon')):
        pose_par = get_vicon_params(calib_path); pose_par['data_type'] = 'pose'
        poses    = get_vicon(seq_path)
    else:
        pose_par = get_groundtruth_params()
        poses    = get_groundtruth(seq_path)

    pose_ts  = np.asarray(get_timestamps(seq_path, pose_par), dtype=float)

    # LiDAR stamps (to know exact times for frame 368 & 369)
    lidar_par = {'sensor_type':'lidar','data_type':'pointcloud'}
    lidar_ts  = np.asarray(get_timestamps(seq_path, lidar_par), dtype=float)

    if max(lidar_idx_368, lidar_idx_369) >= len(lidar_ts):
        raise IndexError("LiDAR index outside time-stamp range")

    # ----- 2. get the two poses individually --------------------------
    T_wb_368 = _pose_at_time(poses, pose_ts, lidar_ts[lidar_idx_368])
    T_wb_369 = _pose_at_time(poses, pose_ts, lidar_ts[lidar_idx_369])

    dxyz = T_wb_369[:3,3] - T_wb_368[:3,3]     # translation of LiDAR base

    # ----- 3. shift bbox limits ---------------------------------------
    bbox_369 = dict(
        x_min = bbox_world_368['x_min'] + dxyz[0],
        x_max = bbox_world_368['x_max'] + dxyz[0],
        y_min = bbox_world_368['y_min'] + dxyz[1],
        y_max = bbox_world_368['y_max'] + dxyz[1],
        z_min = bbox_world_368['z_min'] + dxyz[2],
        z_max = bbox_world_368['z_max'] + dxyz[2]
    )
    return bbox_369, dxyz

def generate_fused_lidar_pointcloud(
    seq_idx=1,
    dataset_dir=None,
    calib_path=None,
    radar_mid_idx=182,
    num_lidar_frames=50,
    vox_size=0.0005,
    min_range_bin=1,
    intensity_thr=0.0,
    verbose: bool = True,          # <-- NEW
):
    """
    Generate a fused LiDAR point cloud from multiple frames with sensor positions removed.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    import open3d as o3d

    # tqdm (optional): only used when verbose=False
    if not verbose:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

    seq_path = dataset_dir + str(seq_idx)

    lidar_params = get_lidar_params(calib_path)
    radar_params = get_cascade_params(calib_path)['heatmap']

    radar_start_idx = radar_mid_idx - (num_lidar_frames // 2)
    lidar_start_idx = find_matching_lidar_frame(
        seq_path, radar_params, lidar_params, radar_start_idx, max_time_diff=0.5
    )

    lidar_mid_idx = find_matching_lidar_frame(
        seq_path, radar_params, lidar_params, radar_mid_idx, max_time_diff=0.5
    )

    radar_stop_idx = radar_mid_idx + (num_lidar_frames // 2)
    lidar_stop_idx = find_matching_lidar_frame(
        seq_path, radar_params, lidar_params, radar_stop_idx, max_time_diff=0.5
    )

    # --- one-line summary (always printed) ---
    print(
        f"Closest start LiDAR frame: {lidar_start_idx} | "
        f"Closest mid LiDAR frame: {lidar_mid_idx} | "
        f"Closest stop LiDAR frame: {lidar_stop_idx}"
    )

    # --- choose the radar frames whose FOVs you want to union ------
    radar_frames = np.arange(radar_start_idx, radar_stop_idx, 1).tolist()

    # --- collect per-frame BBOXes ----------------------------------
    fov_boxes = [
        get_radar_fov_bbox(
            seq_path=seq_path,
            calib_path=calib_path,
            radar_frame_idx=rf_idx,
            min_range_bin=min_range_bin,
            intensity_thr=intensity_thr
        )
        for rf_idx in radar_frames
    ]

    # --- take the component-wise extrema to get the cuboid ----------
    master_bbox = {
        k: (min if k.endswith('min') else max)(box[k] for box in fov_boxes)
        for k in ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
    }

    if verbose:
        print("Master (union) radar-FOV bbox:")
        for k, v in master_bbox.items():
            print(f"  {k}: {v:.3f}")
        print(f"master_bbox.keys(): {master_bbox.keys()} ")

    # --- helper: crop with an {x_min...} bbox ------------------------------
    def _crop_world(cloud_world, bbox):
        m = (
            (cloud_world[:, 0] > bbox['x_min']) & (cloud_world[:, 0] < bbox['x_max']) &
            (cloud_world[:, 1] > bbox['y_min']) & (cloud_world[:, 1] < bbox['y_max']) &
            (cloud_world[:, 2] > bbox['z_min']) & (cloud_world[:, 2] < bbox['z_max'])
        )
        return cloud_world[m]

    # Precompute LiDAR extrinsic and time-aligned world<-base poses once
    _lidar_params_cached = get_lidar_params(calib_path)
    _T_bs_cached = np.eye(4)
    _T_bs_cached[:3, 3] = _lidar_params_cached['translation']
    _T_bs_cached[:3, :3] = Rotation.from_quat(_lidar_params_cached['rotation']).as_matrix()
    _gt_params_cached = get_groundtruth_params()
    _gt_ts_cached = np.asarray(get_timestamps(seq_path, _gt_params_cached), dtype=float)
    _lidar_ts_cached = np.asarray(get_timestamps(seq_path, _lidar_params_cached), dtype=float)
    _gt_poses_cached = get_groundtruth(seq_path)
    _lidar_gt_cached, _ = interpolate_poses(_gt_poses_cached, _gt_ts_cached, _lidar_ts_cached)

    # 2. iterate through LiDAR frames, crop, accumulate (no bbox shifting)
    accum = []                     # list of (M_i,4) arrays
    lidar_positions = []           # list of LiDAR sensor positions in world frame

    # Prepare iterator (tqdm when verbose=False and tqdm is available)
    indices = range(lidar_start_idx, lidar_stop_idx + 1)
    iterator = indices
    pbar = None
    if not verbose and 'tqdm' in locals() and tqdm is not None:
        pbar = tqdm(indices, desc="Fusing LiDAR frames", unit="frame")
        iterator = pbar

    for idx in iterator:
        # Compute world<-sensor transform and extract position
        T_wb = _lidar_gt_cached[idx]
        T_ws = T_wb @ _T_bs_cached
        sensor_position = T_ws[:3, 3]
        lidar_positions.append(sensor_position)

        # Load raw cloud and transform to world using precomputed T_ws
        cloud_local = get_pointcloud(idx, seq_path, _lidar_params_cached)
        cloud_w = transform_pcl(cloud_local, T_ws)

        cropped = _crop_world(cloud_w, master_bbox)
        accum.append(cropped)

        if verbose:
            print(
                f"frame {idx:4d}  pts loaded {cloud_w.shape[0]:5d}  "
                f"cropped {cropped.shape[0]:5d}  "
                f"sensor pos=({sensor_position[0]:.2f}, {sensor_position[1]:.2f}, {sensor_position[2]:.2f})"
            )
        elif pbar is not None:
            # lightweight live stats in progress bar
            pbar.set_postfix({
                "loaded": int(cloud_w.shape[0]),
                "kept": int(cropped.shape[0]),
                "x": f"{sensor_position[0]:.2f}",
                "y": f"{sensor_position[1]:.2f}",
                "z": f"{sensor_position[2]:.2f}",
            })

    if pbar is not None:
        pbar.close()

    # ---------------------------------------------------------------------
    # fuse everything into one array (or keep the list if you prefer)
    fused_world = np.vstack(accum)
    print("Total fused points:", fused_world.shape[0])

    # ---------------------------------------------------------------
    # KEEP ONLY DISTINCT ROWS (x, y, z, i)  -- exact matches only
    # ---------------------------------------------------------------
    uniq_rows, uniq_idx = np.unique(fused_world, axis=0, return_index=True)
    fused_world_unique = fused_world[np.sort(uniq_idx)]
    print(
        f"Removed {fused_world.shape[0] - fused_world_unique.shape[0]} "
        f"exact duplicate points; new total: {fused_world_unique.shape[0]}"
    )

    # Create LiDAR sensor position visualization
    lidar_positions_array = np.array(lidar_positions)
    pcd_sensor_positions = o3d.geometry.PointCloud()
    pcd_sensor_positions.points = o3d.utility.Vector3dVector(lidar_positions_array)
    pcd_sensor_positions.colors = o3d.utility.Vector3dVector(
        np.full_like(lidar_positions_array, [1.0, 1.0, 0.0])  # Yellow for sensor positions
    )

    # Create coordinate frame markers for each sensor position (optional)
    coordinate_frames = []
    for i, _pos in enumerate(lidar_positions_array):
        lidar_params = get_lidar_params(calib_path)
        T_bs = np.eye(4)
        T_bs[:3, 3] = lidar_params['translation']
        T_bs[:3, :3] = Rotation.from_quat(lidar_params['rotation']).as_matrix()

        gt_params = get_groundtruth_params()
        gt_ts = np.asarray(get_timestamps(seq_path, gt_params), dtype=float)
        lidar_ts = np.asarray(get_timestamps(seq_path, lidar_params), dtype=float)
        gt_poses = get_groundtruth(seq_path)

        lidar_gt, _ = interpolate_poses(gt_poses, gt_ts, lidar_ts)
        T_wb = lidar_gt[lidar_start_idx + i]  # world <- base
        T_ws = T_wb @ T_bs  # world <- sensor

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord_frame.transform(T_ws)
        coordinate_frames.append(coord_frame)

    # Verify whether sensor positions are present in the fused array and remove them
    try:
        from scipy.spatial import cKDTree
        pts = fused_world_unique[:, :3]
        sensor_pts = np.asarray(pcd_sensor_positions.points)
        if pts.size == 0 or sensor_pts.size == 0:
            if verbose:
                print("[check] Skipped sensor-in-array check: empty input.")
        else:
            tree = cKDTree(pts)
            dists, _ = tree.query(sensor_pts, k=1)
            tol = 1e-9
            in_mask = dists <= tol
            num_present = int(np.count_nonzero(in_mask))
            num_total = int(sensor_pts.shape[0])
            print(f"[check] Sensor positions present in fused_world_unique within tol {tol:g}: {num_present}/{num_total}")
            if num_present < num_total:
                missing_d = dists[~in_mask]
                if missing_d.size:
                    print(f"[check] Missing min/mean/max distance: {missing_d.min():.3e} / {missing_d.mean():.3e} / {missing_d.max():.3e}")
            tree_sens = cKDTree(sensor_pts)
            d_ps, _ = tree_sens.query(pts, k=1)
            rm_mask = d_ps <= tol
            removed = int(np.count_nonzero(rm_mask))
            if removed > 0:
                fused_world_unique = fused_world_unique[~rm_mask]
            print(f"[remove] tol={tol:g} -> removed {removed} rows from fused_world_unique matching sensor positions. New count: {fused_world_unique.shape[0]}")
    except Exception as e:
        print(f"[warn] Sensor-in-array check failed: {e}")

    return fused_world_unique, lidar_positions_array



def get_radar_config_info(config_path):
    """
    Extract radar location (geometric center) and boresight direction from config file.
    
    Parameters
    ----------
    config_path : str
        Path to radar configuration JSON file
        
    Returns
    -------
    radar_center : np.ndarray (3,)
        Geometric center of all TX and RX antennas in meters
    boresight : np.ndarray (3,)
        Normalized boresight direction vector
    tx_positions : np.ndarray (N_tx, 3)
        TX antenna positions in meters
    rx_positions : np.ndarray (N_rx, 3)
        RX antenna positions in meters
    """
    import json
    
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Extract TX and RX positions (convert from mm to meters)
    tx_positions = np.array([tx['pos_mm'] for tx in cfg['tx_array']], dtype=float) / 1000.0
    rx_positions = np.array([rx['pos_mm'] for rx in cfg['rx_array']], dtype=float) / 1000.0
    
    # Calculate geometric center of all antennas
    all_positions = np.vstack([tx_positions, rx_positions])
    radar_center = np.mean(all_positions, axis=0)
    
    # Extract boresight direction (normalized)
    boresight = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
    boresight_norm = np.linalg.norm(boresight)
    if boresight_norm > 1e-9:
        boresight = boresight / boresight_norm
    else:
        # Fallback to default boresight if invalid
        boresight = np.array([0.0, 1.0, 0.0])
        print("[warn] Invalid boresight in config, using default [0,1,0]")
    
    print(f"Radar center: {radar_center}")
    print(f"Boresight direction: {boresight}")
    print(f"TX antennas: {len(tx_positions)}, RX antennas: {len(rx_positions)}")
    
    return radar_center, boresight, tx_positions, rx_positions

def remove_points_behind_radar(points, radar_position, radar_direction=None, buffer_distance=0.0):
    """
    Remove all points that are behind the radar location.
    
    Parameters
    ----------
    points : np.ndarray (N, 3) or (N, 4)
        Point cloud coordinates. If (N, 4), last column is intensity
    radar_position : np.ndarray (3,)
        Radar position in world coordinates
    radar_direction : np.ndarray (3,) or None
        Radar viewing direction (unit vector). If None, uses direction from origin to radar position
    buffer_distance : float
        Additional buffer distance behind radar to remove (meters)
        
    Returns
    -------
    filtered_points : np.ndarray (M, 3) or (M, 4)
        Points with those behind radar removed (same shape as input)
    mask : np.ndarray (N,)
        Boolean mask indicating which points were kept
    """
    points = np.asarray(points, dtype=np.float64)
    radar_pos = np.asarray(radar_position, dtype=np.float64)
    
    if points.shape[1] not in [3, 4]:
        raise ValueError("points must be (N, 3) or (N, 4)")
    if radar_pos.shape != (3,):
        raise ValueError("radar_position must be (3,)")
    
    # Extract coordinates (first 3 columns)
    coords = points[:, :3]
    
    # Determine radar direction
    if radar_direction is None:
        # Use direction from origin to radar position
        radar_dir = radar_pos - np.array([0.0, 0.0, 0.0])
        radar_dir = radar_dir / (np.linalg.norm(radar_dir) + 1e-12)
    else:
        radar_dir = np.asarray(radar_direction, dtype=np.float64)
        radar_dir = radar_dir / (np.linalg.norm(radar_dir) + 1e-12)
    
    # Vector from radar to each point
    point_vectors = coords - radar_pos  # (N, 3)
    
    # Distance from radar to each point
    distances_from_radar = np.linalg.norm(point_vectors, axis=1)  # (N,)
    
    # Projection of point vectors onto radar direction
    # Positive projection means point is in front of radar
    # Negative projection means point is behind radar
    projections = np.dot(point_vectors, radar_dir)  # (N,)
    
    # Keep points that are:
    # 1. In front of radar (positive projection) OR
    # 2. Very close to radar (within buffer_distance)
    # 3. Not too far behind radar (within reasonable distance)
    keep_mask = (projections >= -buffer_distance) | (distances_from_radar <= buffer_distance)
    
    filtered_points = points[keep_mask]
    
    print(f"Removed {np.sum(~keep_mask)} points behind radar")
    print(f"Kept {np.sum(keep_mask)} points")
    
    return filtered_points, keep_mask

def estimate_normals_gpu(
    points: np.ndarray,
    mode: str = "k",                # "k" or "radius"
    k: int = 30,
    radius: float | str = "auto",   # float or "auto" (bbox * auto_radius_frac)
    auto_radius_frac: float = 0.01, # ~CloudCompare default heuristic
    use_octree_like: bool = True,   # voxel accelerator for radius mode
    bucket_size: int = 32,          # ~points per voxel (radius mode)
    orient_toward_origin: bool = True,
    origin: np.ndarray | None = None,
    require_min_neighbors: int = 3,
    max_batch: int = 262144,        # query batch size
    faiss_nprobe: int = 64,         # IVF nprobe if we switch to IVF later
    verbose: bool = True,
):
    """
    Fast multi-GPU normal estimation using FAISS (+ optional voxel accelerator for radius).

    Returns
    -------
    normals : (N,3) float32
    """

    # ----------------------------
    # early checks
    # ----------------------------
    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N,3) array")
    N = P.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # ----------------------------
    # devices & FAISS
    # ----------------------------
    try:
        import faiss  # GPU build recommended
    except Exception as e:
        raise RuntimeError("FAISS not available; install faiss-gpu for best performance") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required") from e

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        # fall back to your CPU path if desired:
        from inspect import signature
        if "mode" in signature(estimate_normals_cpu).parameters:
            return estimate_normals_cpu(
                P, mode=mode, k=k, radius=radius, auto_radius_frac=auto_radius_frac,
                use_octree_like=use_octree_like, bucket_size=bucket_size,
                orient_toward_origin=orient_toward_origin, origin=origin,
                require_min_neighbors=require_min_neighbors
            )
        else:
            # your original GPU stubs fallback
            return estimate_normals_cpu(
                P, k=k, orient_toward_origin=orient_toward_origin,
                origin=origin, require_min_neighbors=require_min_neighbors
            )

    # ----------------------------
    # auto radius (CloudCompare-like)
    # ----------------------------
    if mode == "radius":
        if isinstance(radius, str) and radius.lower() == "auto":
            mn = P.min(axis=0); mx = P.max(axis=0)
            diag = float(np.linalg.norm(mx - mn))
            radius = float(diag * auto_radius_frac) if diag > 0 else 0.0
        radius = float(radius)
        if radius <= 0:
            raise ValueError("radius must be > 0 (or 'auto') in radius mode")

    # ----------------------------
    # build FAISS index on GPUs (sharded)
    # ----------------------------
    d = 3
    cpu_index = faiss.IndexFlatL2(d)  # exact L2
    # Copy points into the index (FAISS uses row-major float32)
    cpu_index.add(P)

    # Send to *all* GPUs with sharding
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True          # important: split index across GPUs
    co.useFloat16 = True     # speed/memory trade-off (fine for 3D)
    try:
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)  # multi-GPU
    except Exception:
        # single-GPU fallback
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # ----------------------------
    # helper: orientation target
    # ----------------------------
    if orient_toward_origin:
        O = np.zeros(3, dtype=np.float32) if origin is None else np.asarray(origin, np.float32)
    else:
        O = None

    # ----------------------------
    # helper: covariance -> normal (Torch GPU)
    # ----------------------------
    def normals_from_neighbors(idx_batch: np.ndarray, mask_batch: np.ndarray | None = None) -> np.ndarray:
        """
        idx_batch: (B, M) neighbor indices (includes -1 where invalid if mask provided)
        mask_batch: (B, M) bool mask of valid neighbors (optional)
        """
        device = torch.device("cuda")
        # gather neighbor points
        idx_t = torch.from_numpy(idx_batch.astype(np.int64)).to(device)
        pts_t = torch.from_numpy(P).to(device)
        neigh = pts_t[idx_t.clip(min=0)]                     # (B, M, 3)
        if mask_batch is not None:
            m = torch.from_numpy(mask_batch).to(device)      # (B, M)
            # zero-out invalid spots so they don't affect mean
            neigh = torch.where(m[..., None], neigh, torch.zeros_like(neigh))
            denom = torch.clamp(m.sum(dim=1, keepdim=True).to(torch.float32), min=1.0)
        else:
            denom = torch.tensor(idx_batch.shape[1], device=device, dtype=torch.float32).view(-1, 1)

        # center
        mean = neigh.sum(dim=1, keepdim=True) / denom[:, None, :]
        X = neigh - mean                                     # (B, M, 3)

        # covariance (B,3,3)
        cov = torch.bmm(X.transpose(1, 2), X)
        # eigh
        evals, evecs = torch.linalg.eigh(cov)                # ascending
        n = evecs[:, :, 0]                                   # smallest eigenvector

        # orient
        if O is not None:
            Q = pts_t[:idx_batch.shape[0]]                   # query points (assumes contiguous batches)
            v = (torch.from_numpy(O).to(device) - Q)         # toward origin
            flip = (n * v).sum(dim=1) < 0
            n[flip] = -n[flip]

        # normalize + handle tiny norms
        norm = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min_(1e-12)
        n = n / norm

        return n.detach().cpu().numpy().astype(np.float32)

    # ----------------------------
    # radius-mode neighbor production
    # ----------------------------
    def make_radius_neighbors():
        """
        Strategy:
        1) Optional voxel accelerator to lower candidate set.
        2) Over-ask FAISS kNN with k_cap ~ expected #points in radius (+margin).
        3) Distance-mask to keep <= radius.
        """
        # estimate average density for global k_cap
        # volume of sphere
        vol = (4.0/3.0) * math.pi * (radius ** 3)
        density = N / max(1e-9, float((P.max(0) - P.min(0)).prod()))
        k_cap = int(min(max(32, math.ceil(2.5 * density * vol)), 4096))  # sane cap
        # well do a single global FAISS search (exact L2) and then mask by radius
        if verbose:
            print(f"[radius] using k_cap={k_cap} (density={density:.3f}, vol={vol:.6f})")

        # batch query
        all_idx = np.empty((N, k_cap), dtype=np.int32)
        all_dist2 = np.empty((N, k_cap), dtype=np.float32)

        for s in tqdm(range(0, N, max_batch), desc="faiss search (radius)", unit="pts"):
            e = min(N, s + max_batch)
            D, I = gpu_index.search(P[s:e], k_cap)   # (B, k_cap)
            all_idx[s:e] = I.astype(np.int32)
            all_dist2[s:e] = D.astype(np.float32)

        # mask by radius
        rad2 = radius * radius
        mask = all_dist2 <= rad2
        # ensure we ignore self if present
        mask &= (all_idx != np.arange(N, dtype=np.int32)[:, None])

        return all_idx, mask

    # ----------------------------
    # k-mode neighbor production
    # ----------------------------
    def make_k_neighbors():
        # ask for k+1 (self may appear)
        kk = int(max(1, k + 1))
        all_idx = np.empty((N, kk), dtype=np.int32)

        for s in tqdm(range(0, N, max_batch), desc="faiss search (k)", unit="pts"):
            e = min(N, s + max_batch)
            D, I = gpu_index.search(P[s:e], kk)  # exact kNN
            all_idx[s:e] = I.astype(np.int32)

        # drop self if present in the first column, keep exactly k neighbors
        # (FAISS does not *guarantee* self is at 0-th slot in sharded mode,
        #  so we remove the row-wise occurrence of the query index if found)
        row = np.arange(N, dtype=np.int32)
        self_mask = (all_idx == row[:, None])
        # build a compacted (N, k) by skipping the self match
        compact = np.empty((N, k), dtype=np.int32)
        for i in tqdm(range(N), desc="postprocess (drop self)", disable=(N > 5_000_000)):
            m = ~self_mask[i]
            compact[i] = all_idx[i][m][:k]
        return compact

    # ----------------------------
    # produce neighbors
    # ----------------------------
    if mode.lower() == "k":
        nbr_idx = make_k_neighbors()
        # sanity: require_min_neighbors
        if require_min_neighbors > 1 and k < require_min_neighbors:
            raise ValueError(f"require_min_neighbors={require_min_neighbors} but k={k}")
        # normals (no mask path needed)
        normals = np.zeros((N, 3), dtype=np.float32)
        for s in tqdm(range(0, N, max_batch), desc="compute normals (k)", unit="pts"):
            e = min(N, s + max_batch)
            n = normals_from_neighbors(nbr_idx[s:e])
            normals[s:e] = n

        return normals

    elif mode.lower() == "radius":
        idx, mask = make_radius_neighbors()
        # enforce require_min_neighbors by zeroing those rows
        valid_counts = mask.sum(axis=1)
        normals = np.zeros((N, 3), dtype=np.float32)
        for s in tqdm(range(0, N, max_batch), desc="compute normals (radius)", unit="pts"):
            e = min(N, s + max_batch)
            n = normals_from_neighbors(idx[s:e], mask[s:e])
            normals[s:e] = n
        normals[valid_counts < max(require_min_neighbors, 1)] = 0.0
        return normals

    else:
        raise ValueError("mode must be 'k' or 'radius'")


def estimate_normals_cpu(
    points: np.ndarray,
    mode: str = "radius",                # "radius" or "k"
    k: int = 20,                         # used when mode="k"
    radius: float | str | None = "auto", # float, "auto", or None
    auto_radius_frac: float = 0.01,      # 1% of bbox diagonal ~ CC's bbox-based "gross first guess"
    use_octree_like: bool = True,        # bucket grid accelerator for radius queries
    bucket_size: int = 32,               # target points per bucket (coarser -> faster radius search)
    orient_toward_origin: bool = True,
    origin: np.ndarray | None = None,
    require_min_neighbors: int = 3,      # minimum neighbors to compute a plane
) -> np.ndarray:
    """
    Estimate per-point normals.

    Parameters
    ----------
    points : (N,3) float32/float64
    mode   : "radius" | "k"
    k      : k for k-NN (>=3 recommended)
    radius : when "radius": float (meters/units), or "auto" to guess from bbox
    auto_radius_frac : fraction of bbox diagonal if radius="auto"
    use_octree_like  : if True and mode="radius", uses a voxel/bucket grid to
                       limit neighbor checks (octree-like acceleration).
    bucket_size      : soft target points per bucket (controls voxel size)
    orient_toward_origin : if True, flip normals to point toward `origin`
    origin           : (3,) array; default is [0,0,0]
    require_min_neighbors : skip/zero normals if neighbors < this

    Returns
    -------
    normals : (N,3) float32 array; zero where not enough neighbors
    """
    P = np.asarray(points, dtype=np.float64)
    N = P.shape[0]
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if P.shape[1] != 3:
        raise ValueError("points must be (N,3)")

    # --- auto radius from bbox (CloudCompare-like guess from bounding box) ---
    if mode == "radius":
        if radius == "auto" or (radius is None):
            # bbox diagonal * fraction
            bb_min = P.min(axis=0)
            bb_max = P.max(axis=0)
            diag = float(np.linalg.norm(bb_max - bb_min))
            radius = auto_radius_frac * diag

    # --- neighbor search back-end(s) ---
    tree = cKDTree(P)  # kdtree; we'll still add a voxel/bucket pass as a coarse filter

    # Build a simple voxel grid index if asked (octree-like acceleration)
    # Choose voxel size so that ~bucket_size points fall per cell on average.
    # Heuristic: voxel_size ~ (cloud_volume / N)^(1/3) * (bucket_size)^(1/3)
    voxel_index = None
    if use_octree_like and mode == "radius":
        bb_min = P.min(axis=0)
        bb_max = P.max(axis=0)
        extent = bb_max - bb_min
        vol = float(np.prod(extent)) if np.all(extent > 0) else (1.0)
        mean_cell = max(1.0, vol / max(1, N))
        cell_len = (mean_cell * bucket_size) ** (1.0 / 3.0)
        # guard against degenerate/flat clouds
        if not np.isfinite(cell_len) or cell_len <= 0:
            cell_len = max(1e-6, float(np.linalg.norm(extent)) / 100.0)

        # map each point to integer voxel coords
        vcoords = np.floor((P - bb_min) / cell_len).astype(np.int64)
        # build hash map voxel -> indices
        voxel_index = {}
        for i, vc in enumerate(map(tuple, vcoords)):
            voxel_index.setdefault(vc, []).append(i)

        # Precompute neighbor voxels offsets (3x3x3 neighborhood)
        nb_vox_offsets = np.array([(dx, dy, dz)
                                   for dx in (-1, 0, 1)
                                   for dy in (-1, 0, 1)
                                   for dz in (-1, 0, 1)], dtype=np.int64)

    # --- output ---
    normals = np.zeros((N, 3), dtype=np.float64)

    # --- main loop ---
    if mode == "k":
        # Use k-NN directly
        k_eff = max(3, int(k))
        dists, idxs = tree.query(P, k=k_eff, workers=-1)
        # If k==1, SciPy returns 1D arrays; ensure 2D
        if k_eff == 1:
            idxs = idxs[:, None]
        for i in range(N):
            nn_idx = idxs[i]
            if nn_idx.ndim == 0:
                nn_idx = np.array([int(nn_idx)])
            if nn_idx.size < require_min_neighbors:
                continue
            normals[i] = _pca_plane_normal(P[nn_idx])

    elif mode == "radius":
        r = float(radius)
        r2 = r * r
        for i in range(N):
            if voxel_index is None:
                # straight radius query
                nn_idx = tree.query_ball_point(P[i], r)
            else:
                # coarse voxel prefilter, then exact distance filter
                # 1) collect candidates from 3x3x3 neighborhood
                bb_min = P.min(axis=0)  # (cost OK; constant)
                # we cached vcoords during build; recompute idx voxel:
                # (to avoid keeping full vcoords in memory we can recompute)
                # but keeping it is fine; do it faster by reusing:
                # vcoords array exists only in the local scope above--lets recompute
                vc = tuple(np.floor((P[i] - bb_min) / cell_len).astype(np.int64))
                cand = []
                for off in nb_vox_offsets:
                    key = (vc[0] + off[0], vc[1] + off[1], vc[2] + off[2])
                    if key in voxel_index:
                        cand.extend(voxel_index[key])
                if not cand:
                    nn_idx = []
                else:
                    cand = np.asarray(cand, dtype=np.int64)
                    # exact radius check
                    diff = P[cand] - P[i]
                    sel = np.einsum('ij,ij->i', diff, diff) <= r2
                    nn_idx = cand[sel].tolist()

            if len(nn_idx) < require_min_neighbors:
                continue
            normals[i] = _pca_plane_normal(P[nn_idx])

    else:
        raise ValueError("mode must be 'radius' or 'k'")

    # --- orient toward origin if requested ---
    if orient_toward_origin:
        O = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
        V = (O[None, :] - P)  # vector from point to origin
        # flip normals whose dot(n, O-P) < 0
        flip = np.einsum('ij,ij->i', normals, V) < 0
        normals[flip] *= -1.0

    # normalize & return
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    ok = n_norm[:, 0] > 0
    normals[ok] /= n_norm[ok]
    return normals.astype(np.float32)


def _pca_plane_normal(neigh_pts: np.ndarray) -> np.ndarray:
    """Smallest-eigenvector of covariance (PCA) -> normal."""
    C = np.cov(neigh_pts.T, bias=True)  # (3,3)
    vals, vecs = eigh(C)                # ascending eigenvalues
    n = vecs[:, 0]
    # guard against NaN
    if not np.all(np.isfinite(n)):
        return np.zeros(3)
    return n


@torch.no_grad()
def orient_normals_toward_sensors_gpu(
    points: np.ndarray,
    normals: np.ndarray,
    sensor_positions: np.ndarray,
    device: str = "cuda",
    chunk: int = 1_000_000,   # tune for memory; 1e6 works on 24GB GPUs
) -> np.ndarray:
    """
    Flip normals so they point toward the nearest sensor position.
    points           : (N,3) float32/float64 CPU numpy
    normals          : (N,3) float32/float64 CPU numpy (unit or not)
    sensor_positions : (M,3) float32/float64 CPU numpy
    returns (N,3) float32 numpy, oriented
    """
    if points.size == 0 or normals.size == 0 or sensor_positions.size == 0:
        return normals.astype(np.float32, copy=True)

    pts  = torch.as_tensor(points, dtype=torch.float32, device=device)
    nrm  = torch.as_tensor(normals, dtype=torch.float32, device=device)
    sens = torch.as_tensor(sensor_positions, dtype=torch.float32, device=device)

    N = pts.shape[0]
    out = torch.empty_like(nrm)

    # normalize normals defensively
    nrm = nrm / (torch.linalg.norm(nrm, dim=1, keepdim=True).clamp_min(1e-12))

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        P  = pts[s:e]          # (B,3)
        Nn = nrm[s:e]          # (B,3)

        # nearest sensor for each point: argmin ||P - S||
        # cdist is efficient for M~100 and B up to ~1e6 on a 4090
        d = torch.cdist(P, sens)            # (B,M)
        j = torch.argmin(d, dim=1)          # (B,)

        S = sens[j]                         # (B,3)
        v = S - P                           # vector toward nearest sensor
        flip = ( (Nn * v).sum(dim=1) < 0 )  # if pointing away, flip
        Nn[flip] = -Nn[flip]
        out[s:e] = Nn

    return out.detach().cpu().numpy().astype(np.float32)


# ==============================
# Visualization (function + CLI)
# ==============================

def create_normal_lineset(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    step: int = 100,
    scale: float = 0.1,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> o3d.geometry.LineSet:
    """
    Build a LineSet that visualizes normals as short line segments.

    Returns
    -------
    o3d.geometry.LineSet
    """
    if points.shape[0] == 0 or normals.shape[0] == 0:
        return o3d.geometry.LineSet()

    assert points.shape[0] == normals.shape[0], "points/normals must match"
    idx = np.arange(0, points.shape[0], max(1, int(step)))
    P = points[idx]
    N = normals[idx]

    starts = P
    ends   = P + N * float(scale)

    line_pts = np.vstack([starts, ends])
    K = starts.shape[0]
    lines = np.array([[i, i + K] for i in range(K)], dtype=np.int32)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(line_pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.repeat([color], K, axis=0))
    return ls

def visualize_points_and_normals(
    pcd: o3d.geometry.PointCloud,
    normals: np.ndarray | None = None,
    *,
    remote_viz: bool = False,
    point_size: int = 3,
    port: int = 8888,
    window_title: str = "Fused LiDAR -- Points + Normals",
    normals_every: int = 100,
    normals_scale: float = 0.1,
    normals_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
):
    """
    Show an Open3D point cloud with normals either:
      - locally in a native window (GLX/VNC/physical display), or
      - remotely in your browser via the WebRTC web visualizer.
    """

    # attach normals if provided
    if normals is not None and len(normals) == len(pcd.points):
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # keep your intensity colors if already set; else gray
    try:
        if np.asarray(pcd.colors).shape[0] != len(pcd.points):
            pcd.paint_uniform_color([0.6, 0.6, 0.6])
    except Exception:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

    # simple camera from bbox
    bbox = pcd.get_axis_aligned_bounding_box()
    c = np.asarray(bbox.get_center(), dtype=np.float32)
    extent = float(np.linalg.norm(np.asarray(bbox.get_extent(), dtype=np.float32)) + 1e-6)
    eye = c + np.array([0, 0, extent], dtype=np.float32)
    up  = np.array([0, 1, 0], dtype=np.float32)

    # Build normals LineSet (works in both local and web UIs)
    try:
        pts_np = np.asarray(pcd.points)
        nrm_np = np.asarray(pcd.normals) if pcd.has_normals() else None
        normal_lines = create_normal_lineset(
            pts_np, nrm_np,
            step=int(normals_every),
            scale=float(normals_scale),
            color=tuple(normals_color)
        ) if nrm_np is not None else None
    except Exception:
        normal_lines = None

    if remote_viz:
        # ------------------------------
        # WebRTC web visualizer (browser)
        # ------------------------------
        # Prefer EGL/OSMesa for remote servers, avoid GLX
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        if "DISPLAY" in os.environ:
            os.environ.pop("DISPLAY")

        # Pick requested port if free, else fall back to a random free port
        import socket
        def _pick_port(requested: int) -> int:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", requested))
                s.close()
                return requested
            except OSError:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.bind(("127.0.0.1", 0))
                    return s2.getsockname()[1]

        port = _pick_port(int(port))
        os.environ["WEBRTC_IP"] = "127.0.0.1"
        os.environ["WEBRTC_PORT"] = str(port)

        from open3d.visualization import webrtc_server
        webrtc_server.enable_webrtc()  # now honors WEBRTC_PORT

        print("=" * 80)
        print("Open3D Web Visualizer is running.")
        print("On your Mac, open a *new* terminal and create an SSH tunnel:")
        print(f"  ssh -L {port}:localhost:{port} user@128.84.85.120")
        print(f"Then open: http://localhost:{port}")
        print("=" * 80)

        # Set point size + normals programmatically in on_init
        def _on_init(vis: o3d.visualization.O3DVisualizer):
            try:
                vis.point_size = int(point_size)
            except Exception:
                pass
            # try to enable normals in the UI (varies by O3D versions)
            for flag in ("show_normals", "show_point_normals"):
                try:
                    setattr(vis, flag, True)
                    break
                except Exception:
                    continue
            try:
                vis.show_axes = True
            except Exception:
                pass
            # good default camera
            try:
                vis.setup_camera(60.0, bbox, c.tolist())
            except Exception:
                pass

        geoms = [pcd] if normal_lines is None else [pcd, normal_lines]
        o3d.visualization.draw(
            geoms,
            title=f"{window_title} (Web Visualizer)",
            show_ui=True,
            width=1280,
            height=800,
            bg_color=(1.0, 1.0, 1.0, 1.0),
            lookat=c, eye=eye, up=up,
            on_init=_on_init,
        )

    else:
        # ------------------------------
        # Local native window (server)
        # ------------------------------
        # NOTE: Requires a real display (physical, VNC, or X-forward with core profile)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1280, height=800)
        opt = vis.get_render_option()
        opt.point_size = int(point_size)
        opt.point_show_normal = False
        opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        vis.add_geometry(pcd)
        if normal_lines is not None:
            vis.add_geometry(normal_lines)
        try:
            ctr = vis.get_view_control()
            ctr.set_lookat(c.tolist())
            ctr.set_front((c - eye).tolist())
            ctr.set_up(up.tolist())
        except Exception:
            pass

        try:
            vis.run()
        finally:
            vis.destroy_window()

@torch.no_grad()
def poisson_reconstruct_gpu(
    points_xyz: np.ndarray,
    normals:    np.ndarray,
    depth: int = 9,
    output_density_sf: bool = True,
    *,
    sigma: float = 1.0,           # splat smoothing (in voxels)
    screening: float = 0.0,       # lambda for screened Poisson; 0 = classical
    iso_level: float | None = 0.0,# None -> median level; 0.0 typical
    device: str = "cuda",
    prefer_half: bool = False,    # memory saver (uses float16 where safe)
    chunk: int = 500_000,         # chunk size for splatting large clouds
    verbose: bool = True,
):
    """
    GPU Poisson surface reconstruction (uniform-grid variant).

    Parameters
    ----------
    points_xyz : (N,3) float32/float64
    normals    : (N,3) float32/float64 (unit-orientation preferred)
    depth      : octree-like depth -> grid size N=2**depth + 1
    output_density_sf : if True, returns per-vertex density scalar field
    sigma      : Gaussian/trilinear splat smoothness (in voxel units)
    screening  : lambda for (lambda - Delta)phi = lambda*chi - grad*V   (chi~sample occupancy); 0 for classical
    iso_level  : isosurface level (None -> robust median)
    device     : "cuda", "cuda:0", etc.
    prefer_half: use float16 where safe to reduce VRAM
    chunk      : points processed per chunk for splatting
    verbose    : print progress

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
    densities : np.ndarray (Nv,) or None
        Per-vertex density (scalar field), if output_density_sf=True
    """

    # ----------------------------
    # 0) Checks & setup
    # ----------------------------
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    assert normals.ndim    == 2 and normals.shape[1]    == 3
    assert points_xyz.shape[0] == normals.shape[0]
    assert depth >= 5, "depth>=5 recommended (grid >= 33^3)."
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if (prefer_half and torch_device.type == "cuda") else torch.float32

    # Grid resolution from "octree depth"
    N = int(2**depth + 1)
    if verbose:
        print(f"[PoissonGPU] depth={depth} -> grid={N}^3")

    # Memory estimate: 3*grid + div + phi ~ 5 * N^3 floats
    approx_gb = (5.0 * N**3 * (2 if dtype==torch.float16 else 4)) / (1024**3)
    if verbose:
        print(f"[PoissonGPU] approx working VRAM ~ {approx_gb:.2f} GB "
              f"({str(dtype).split('.')[-1]}); consider depth<=9 on a 24GB GPU")

    # ----------------------------
    # 1) Normalize to unit cube
    # ----------------------------
    P = np.asarray(points_xyz, np.float64)
    Nrm = np.asarray(normals,    np.float64)

    pmin = P.min(0)
    pmax = P.max(0)
    scale = (pmax - pmin).max()
    if scale <= 0:
        raise ValueError("Degenerate point cloud (zero extent).")
    Pn = (P - pmin) / scale  # [0,1]^3

    # ----------------------------
    # 2) Allocate grids on GPU
    # ----------------------------
    # Vector field V (3,H,W,D), density grid W (scalar), divergence div, potential phi
    Vgrid = torch.zeros((3, N, N, N), dtype=dtype, device=torch_device)
    Wgrid = torch.zeros((N, N, N),    dtype=dtype, device=torch_device)

    # ----------------------------
    # 3) Splat oriented points into V (trilinear)
    #    Each sample adds its normal to the 8 neighboring voxels (weighted)
    #    Also accumulate Wgrid as a "density" proxy
    # ----------------------------
    Pn_t  = torch.from_numpy(Pn).to(torch_device, non_blocking=True).to(torch.float32)
    Nrm_t = torch.from_numpy(Nrm).to(torch_device, non_blocking=True).to(torch.float32)
    # ensure normals are unit
    Nrm_t = Nrm_t / (torch.linalg.norm(Nrm_t, dim=1, keepdim=True) + 1e-12)

    # convert positions to voxel coordinates
    # grid coords in [0, N-1]
    X = Pn_t * (N - 1)

    def _splat_batch(xb, nb):
        # base integer voxel
        i0 = torch.floor(xb).to(torch.int64)  # (B,3)
        f  = xb - i0.to(torch.float32)        # frac
        wx = torch.stack([1.0 - f[:, 0], f[:, 0]], dim=1)  # (B,2)
        wy = torch.stack([1.0 - f[:, 1], f[:, 1]], dim=1)
        wz = torch.stack([1.0 - f[:, 2], f[:, 2]], dim=1)

        # eight corners offsets
        for dx in (0,1):
            ix = (i0[:,0] + dx).clamp_(0, N-1)
            wxd = wx[:, dx]
            for dy in (0,1):
                iy = (i0[:,1] + dy).clamp_(0, N-1)
                wyd = wy[:, dy]
                for dz in (0,1):
                    iz = (i0[:,2] + dz).clamp_(0, N-1)
                    w = (wxd * wyd * wz[:, dz]).to(dtype)

                    # flat index for scatter
                    # idx = ix * (N*N) + iy * N + iz
                    idx = (ix * (N*N) + iy * N + iz).to(torch.int64)

                    # scatter-add normals into vector field
                    for c in range(3):
                        Vgrid.view(3, -1)[c].index_add_(0, idx, (nb[:, c].to(dtype) * w))

                    # accumulate weights into Wgrid (density proxy)
                    Wgrid.view(-1).index_add_(0, idx, w)

    # chunked splatting
    total = Pn_t.shape[0]
    for s in range(0, total, chunk):
        e = min(total, s + chunk)
        _splat_batch(X[s:e], Nrm_t[s:e])
        if verbose and (e % (5*chunk) == 0 or e == total):
            print(f"[PoissonGPU] splatted {e}/{total} points")

    # optional Gaussian-like smoothing (very light) via separable conv
    if sigma and sigma > 0:
        # build 1D Gaussian kernel (on CPU then to GPU)
        import math
        rad = max(1, int(2.5 * sigma))
        xs = torch.arange(-rad, rad+1, dtype=torch.float32)
        k1 = torch.exp(-0.5 * (xs / sigma)**2)
        k1 = (k1 / k1.sum()).to(dtype).to(torch_device)

        def _blur3_depthwise(vol_5d: torch.Tensor, k1_1d: torch.Tensor) -> torch.Tensor:
            """
            Separable 3D blur for (1, C, D, H, W) using the same 1-D kernel per channel.
            Depthwise conv3d with groups=C to avoid channel mixing.
            Keeps output size == input size.
            """
            assert vol_5d.ndim == 5 and vol_5d.shape[0] == 1, f"expected (1,C,D,H,W), got {tuple(vol_5d.shape)}"
            C = vol_5d.shape[1]
            k1 = k1_1d.to(device=vol_5d.device, dtype=vol_5d.dtype)
            rad = (k1.numel() - 1) // 2

            kz = k1.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)  # (C,1,K,1,1)
            ky = k1.view(1, 1,  1, -1, 1).repeat(C, 1, 1, 1, 1)  # (C,1,1,K,1)
            kx = k1.view(1, 1,  1, 1, -1).repeat(C, 1, 1, 1, 1)  # (C,1,1,1,K)

            # Pad only the axis we convolve over (pad order: Wl, Wr, Hl, Hr, Dl, Dr)
            v = F.pad(vol_5d, (0, 0, 0, 0, rad, rad), mode='replicate')  # pad D
            v = F.conv3d(v, kz, groups=C)

            v = F.pad(v,      (0, 0, rad, rad, 0, 0), mode='replicate')  # pad H
            v = F.conv3d(v, ky, groups=C)

            v = F.pad(v,      (rad, rad, 0, 0, 0, 0), mode='replicate')  # pad W
            v = F.conv3d(v, kx, groups=C)
            return v

        # --- optional Gaussian-like smoothing ---
        if sigma and sigma > 0:
            rad = max(1, int(2.5 * sigma))
            xs  = torch.arange(-rad, rad+1, device=Vgrid.device, dtype=torch.float32)
            k1g = torch.exp(-0.5 * (xs / float(sigma))**2)
            k1g = (k1g / k1g.sum()).to(Vgrid.dtype)
            # Vgrid: (3, N, N, N) -> (1, 3, D, H, W)
            Vgrid = _blur3_depthwise(Vgrid.unsqueeze(0), k1g).squeeze(0)
            # Wgrid: (N, N, N) -> (1, 1, D, H, W)
            Wgrid = _blur3_depthwise(Wgrid.unsqueeze(0).unsqueeze(0), k1g).squeeze(0).squeeze(0)

    # ----------------------------
    # 4) Compute divergence: div V
    # ----------------------------
    # spacing in normalized cube
    h = 1.0 / (N - 1)
    # torch.gradient gives central differences inside domain
    dVx_dx = torch.gradient(Vgrid[0], spacing=(h,h,h), edge_order=2)[0]
    dVy_dy = torch.gradient(Vgrid[1], spacing=(h,h,h), edge_order=2)[1]
    dVz_dz = torch.gradient(Vgrid[2], spacing=(h,h,h), edge_order=2)[2]
    div = (dVx_dx + dVy_dy + dVz_dz).to(dtype)  # (N,N,N)

    # Optional screened term: (lambda - Delta)phi = -div + lambda*chi,
    # where chi is occupancy proxy (here normalized Wgrid)
    if screening and screening > 0:
        chi = (Wgrid / (Wgrid.max() + 1e-8)).to(dtype)
    else:
        chi = None

    # ----------------------------
    # 5) Solve Poisson via FFT: Deltaphi = -div  (or screened)
    # ----------------------------
    # FFT assumes periodic boundary; works well in practice for reconstruction.
    # Upcast to float32 for cuFFT (half requires power-of-two sizes)
    div32 = div.to(torch.float32)

    if screening and screening > 0 and chi is not None:
        chi32 = chi.to(torch.float32)
    else:
        chi32 = None

    # sanity: shapes must be N x N x N
    assert div32.shape == (N, N, N), f"div shape {tuple(div32.shape)} != {(N,N,N)}"

    div_k = torch.fft.fftn(div32)

    freqs = torch.fft.fftfreq(N, d=h).to(div32.device, dtype=torch.float32)
    kx, ky, kz = torch.meshgrid(freqs, freqs, freqs, indexing='ij')
    k2 = (kx*kx + ky*ky + kz*kz)                     # float32
    lap = (2*np.pi)**2 * k2
    lap[0,0,0] = 1.0

    if screening and screening > 0 and chi32 is not None:
        lam = torch.tensor(float(screening), dtype=torch.float32, device=div32.device)
        chi_k = torch.fft.fftn(chi32)
        phi_k = (-div_k + lam * chi_k) / (lam + lap)
    else:
        phi_k = (-div_k) / lap

    phi_k[0,0,0] = 0
    phi = torch.fft.ifftn(phi_k).real.to(torch.float32)  # keep phi in fp32

    # ----------------------------
    # 6) Extract mesh with marching cubes
    # ----------------------------
    vol = phi.detach().cpu().numpy()
    # pick iso level
    level = (np.median(vol) if iso_level is None else float(iso_level))
    if verbose:
        print(f"[PoissonGPU] marching cubes at iso_level={level:.5g}")
    verts, faces, norms, _ = skimage_marching_cubes(vol, level=level, spacing=(h, h, h))

    # map back to original coordinates: x' = pmin + scale * verts
    Vw = pmin[None, :] + scale * verts

    # build mesh in Open3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(Vw.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    # use MC normals (already in 'norms') or recompute on the watertight mesh
    mesh.vertex_normals = o3d.utility.Vector3dVector(norms.astype(np.float64))
    mesh.compute_triangle_normals()

    densities = None
    if output_density_sf:
        # Sample the smoothed density Wgrid at each vertex (trilinear)
        # Convert world verts -> normalized cube -> voxel coords
        vn = (Vw - pmin[None, :]) / scale
        vc = vn * (N - 1)
        vc = np.clip(vc, 0.0, N - 1 - 1e-6)

        ix = np.floor(vc).astype(np.int64)
        fx = vc - ix
        # gather 8 neighbors and trilinear interpolate
        W = Wgrid.detach().cpu().numpy()
        dens = np.zeros((Vw.shape[0],), dtype=np.float32)
        for dx in (0,1):
            wx = (1-fx[:,0]) if dx==0 else fx[:,0]
            ix0 = np.clip(ix[:,0]+dx, 0, N-1)
            for dy in (0,1):
                wy = (1-fx[:,1]) if dy==0 else fx[:,1]
                iy0 = np.clip(ix[:,1]+dy, 0, N-1)
                for dz in (0,1):
                    wz = (1-fx[:,2]) if dz==0 else fx[:,2]
                    iz0 = np.clip(ix[:,2]+dz, 0, N-1)
                    dens += (wx*wy*wz) * W[ix0, iy0, iz0]
        # normalize for nicer range (0..1)
        if dens.max() > 0:
            dens = dens / dens.max()
        densities = dens

        # Attach as (fake) colors or keep separate; Open3D has no native SF slot.
        # We'll store densities in vertex colors (grayscale) for quick inspection.
        col = np.repeat(densities[:,None], 3, axis=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.astype(np.float64))

    if verbose:
        nv, nf = np.asarray(mesh.vertices).shape[0], np.asarray(mesh.triangles).shape[0]
        print(f"[PoissonGPU] mesh: {nv} verts, {nf} faces")

    return mesh, densities

def _pick_free_port(requested: int) -> int:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", requested))
        s.close()
        return requested
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
            s2.bind(("127.0.0.1", 0))
            return s2.getsockname()[1]

def create_mesh_normal_lineset(
    mesh: o3d.geometry.TriangleMesh,
    step: int = 400,
    scale: float = 0.05,
    color=(1.0, 0.0, 0.0)
) -> o3d.geometry.LineSet | None:
    """Build a LineSet that draws a subset of vertex normals as arrows."""
    if not mesh.has_vertex_normals():
        try:
            m = mesh.clone()
            m.compute_vertex_normals()
            verts = np.asarray(m.vertices)
            vnorm = np.asarray(m.vertex_normals)
        except Exception:
            return None
    else:
        verts = np.asarray(mesh.vertices)
        vnorm = np.asarray(mesh.vertex_normals)

    if verts.size == 0:
        return None

    idx = np.arange(0, verts.shape[0], max(1, int(step)))
    P = verts[idx]
    N = vnorm[idx]

    # Build segments [p -> p + n*scale]
    pts = np.vstack([P, P + N * float(scale)])
    lines = []
    for i, _ in enumerate(idx):
        lines.append([i, i + len(idx)])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    col = np.tile(np.asarray(color, float)[None, :], (len(lines), 1))
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls



def _color_map_goyr(vals01: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear green->yellow->orange->red colormap.
    Input: vals01 in [0,1] (broadcastable 1D)
    Output: (N,3) in [0,1]
    Anchors:
      0.0 -> green  (0.00, 1.00, 0.00)
      0.50-> yellow (1.00, 1.00, 0.00)
      0.75-> orange (1.00, 0.50, 0.00)
      1.0 -> red    (1.00, 0.00, 0.00)
    """
    v = np.clip(vals01.astype(np.float64), 0.0, 1.0)
    c = np.zeros((v.shape[0], 3), dtype=np.float64)

    # [0, 0.5] : green -> yellow
    m = (v <= 0.5)
    t = np.zeros_like(v)
    if m.any():
        t[m] = v[m] / 0.5  # 0..1
        # green (0,1,0) -> yellow (1,1,0)
        c[m, 0] = t[m]            # R: 0 -> 1
        c[m, 1] = 1.0             # G: 1 -> 1
        c[m, 2] = 0.0             # B: 0

    # (0.5, 0.75] : yellow -> orange
    m = (v > 0.5) & (v <= 0.75)
    if m.any():
        t[m] = (v[m] - 0.5) / 0.25  # 0..1
        # yellow (1,1,0) -> orange (1,0.5,0)
        c[m, 0] = 1.0               # R: 1 -> 1
        c[m, 1] = 1.0 - 0.5 * t[m]  # G: 1 -> 0.5
        c[m, 2] = 0.0               # B: 0

    # (0.75, 1] : orange -> red
    m = (v > 0.75)
    if m.any():
        t[m] = (v[m] - 0.75) / 0.25  # 0..1
        # orange (1,0.5,0) -> red (1,0,0)
        c[m, 0] = 1.0                # R: 1 -> 1
        c[m, 1] = 0.5 * (1.0 - t[m]) # G: 0.5 -> 0
        c[m, 2] = 0.0                # B: 0

    return c

def visualize_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    # --- new coloring args ---
    color_scalar: np.ndarray | None = None,   # e.g., density_sf (Nv,)
    scalar_name: str = "density",
    vmin: float | None = None,
    vmax: float | None = None,
    robust_percentiles: tuple[float, float] = (1.0, 99.0),  # used if vmin/vmax are None
    # --- existing viz args ---
    remote_viz: bool = False,
    port: int = 8888,
    window_title: str = "Poisson Mesh",
    show_wireframe: bool = False,
    show_axes: bool = True,
    show_normals: bool = True,
    normals_every: int = 400,
    normals_scale: float = 0.05,
    bg_color=(1.0, 1.0, 1.0, 1.0),
):
    """
    Visualize a mesh (local native window or WebRTC browser viewer).

    If `color_scalar` is provided (length == #vertices), the mesh is colored
    using a green->yellow->orange->red ramp:
        low = green, mid = yellow/orange, high = red.

    You can control normalization with vmin/vmax; if omitted, robust
    [p1, p99] percentiles are used to suppress outliers.
    """
    # Ensure normals exist for shading & optional arrows
    if not mesh.has_vertex_normals():
        m = mesh.clone()
        m.compute_vertex_normals()
    else:
        m = mesh

    # ---- optional scalar-based coloring ----
    if color_scalar is not None:
        s = np.asarray(color_scalar, dtype=np.float64).reshape(-1)
        Nv = np.asarray(m.vertices).shape[0]
        if s.shape[0] != Nv:
            raise ValueError(f"{scalar_name} length {s.shape[0]} != #vertices {Nv}")

        # robust normalization if vmin/vmax not provided
        if vmin is None or vmax is None:
            s_valid = s[np.isfinite(s)]
            if s_valid.size == 0:
                vmin_eff, vmax_eff = 0.0, 1.0
            else:
                lo, hi = np.percentile(s_valid, robust_percentiles)
                vmin_eff = float(lo if vmin is None else vmin)
                vmax_eff = float(hi if vmax is None else vmax)
        else:
            vmin_eff, vmax_eff = float(vmin), float(vmax)

        if vmax_eff <= vmin_eff:
            vmax_eff = vmin_eff + 1e-12

        s01 = (s - vmin_eff) / (vmax_eff - vmin_eff)
        colors = _color_map_goyr(np.clip(s01, 0.0, 1.0))
        m.vertex_colors = o3d.utility.Vector3dVector(colors)

        # Helpful printout so it's crystal clear which end is which
        print(f"[visualize_mesh] Colored by '{scalar_name}':")
        print("  low values  -> green")
        print("  mid values  -> yellow / orange")
        print("  high values -> red")
        print(f"  normalization: vmin={vmin_eff:.5g}, vmax={vmax_eff:.5g} "
              f"(robust={robust_percentiles} used if you didn't pass vmin/vmax)")

    # Build normal arrows (as LineSet)
    normal_lines = None
    if show_normals:
        try:
            normal_lines = create_mesh_normal_lineset(
                m, step=int(normals_every), scale=float(normals_scale), color=(1.0, 0.0, 0.0)
            )
        except Exception:
            normal_lines = None

    # View parameters from bbox
    bbox = m.get_axis_aligned_bounding_box()
    c = np.asarray(bbox.get_center(), dtype=np.float32)
    extent = float(np.linalg.norm(np.asarray(bbox.get_extent(), dtype=np.float32)) + 1e-6)
    eye = c + np.array([0, 0, extent], dtype=np.float32)
    up  = np.array([0, 1, 0], dtype=np.float32)

    if remote_viz:
        # Headless-friendly: WebRTC viewer
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        if "DISPLAY" in os.environ:
            os.environ.pop("DISPLAY")

        port = _pick_free_port(int(port))
        os.environ["WEBRTC_IP"] = "127.0.0.1"
        os.environ["WEBRTC_PORT"] = str(port)

        from open3d.visualization import webrtc_server
        webrtc_server.enable_webrtc()

        print("=" * 80)
        print("Open3D Web Visualizer is running.")
        print("Create an SSH tunnel from your Mac:")
        print(f"  ssh -L {port}:localhost:{port} user@128.84.85.120")
        print(f"Then open: http://localhost:{port}")
        print("=" * 80)

        def _on_init(vis: o3d.visualization.O3DVisualizer):
            # camera & axes
            try:
                vis.setup_camera(60.0, bbox, c.tolist())
            except Exception:
                pass
            try:
                vis.show_axes = bool(show_axes)
            except Exception:
                pass
            # You can toggle wireframe from the UI; some versions expose:
            #   vis.show_wireframe = True/False
            try:
                if show_wireframe:
                    vis.show_wireframe = True
            except Exception:
                pass

        geoms = [m] if normal_lines is None else [m, normal_lines]
        o3d.visualization.draw(
            geoms,
            title=f"{window_title} (Web Visualizer)",
            show_ui=True,
            width=1280,
            height=800,
            bg_color=tuple(bg_color),
            lookat=c, eye=eye, up=up,
            on_init=_on_init,
        )

    else:
        # Local native window (needs real GL context; VNC/X11 core profile)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_title, width=1280, height=800)
        try:
            opt = vis.get_render_option()
            opt.background_color = np.array(bg_color[:3], dtype=np.float32)
            opt.mesh_show_back_face = True
            opt.mesh_show_wireframe = bool(show_wireframe)
            opt.light_on = True
        except Exception:
            pass

        vis.add_geometry(m)
        if normal_lines is not None:
            vis.add_geometry(normal_lines)

        try:
            ctr = vis.get_view_control()
            ctr.set_lookat(c.tolist())
            ctr.set_front((c - eye).tolist())
            ctr.set_up(up.tolist())
        except Exception:
            pass

        try:
            vis.run()
        finally:
            vis.destroy_window()
