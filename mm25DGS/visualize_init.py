"""Visualize point clouds before and after FOV filtering + FPS.

For each scene, generates two views (top-down and perspective) for:
  1. Original point cloud — points inside FOV colored green, outside gray.
  2. FPS-selected 25K points — colored by distance from radar.

Uses matplotlib with Agg backend (works headless, no display needed).
"""

import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm

import torch

from .config import RadarConfig, C
from .initialization import NEAR_FIELD_M, _radar_geometry, _farthest_point_sampling_gpu


def _compute_fov_mask(xyz, radar_cfg):
    """Reproduce the exact FOV filter from initialization.py."""
    radar_center, boresight = _radar_geometry(radar_cfg)
    max_range = radar_cfg.num_adc_samples * radar_cfg.range_resolution

    to_pts = xyz - radar_center
    dists = np.linalg.norm(to_pts, axis=1)
    to_pts_norm = to_pts / np.maximum(dists[:, None], 1e-12)

    cos_bore = np.dot(to_pts_norm, boresight)
    angle_from_bore = np.arccos(np.clip(cos_bore, -1.0, 1.0))

    mask = (
        (dists >= NEAR_FIELD_M)
        & (dists <= max_range)
        & (angle_from_bore <= np.pi / 2)
    )
    return mask, radar_center, boresight, max_range


def _subsample_for_plot(xyz, colors, max_points=80000):
    """Randomly subsample for plotting speed (matplotlib is slow with >100K points)."""
    if len(xyz) <= max_points:
        return xyz, colors
    idx = np.random.default_rng(42).choice(len(xyz), max_points, replace=False)
    return xyz[idx], colors[idx]


def _plot_pointcloud(ax, xyz, colors, point_size=0.3, alpha=0.6):
    """Scatter plot a point cloud on a 3D axis."""
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=point_size, alpha=alpha, edgecolors="none", rasterized=True)


def _plot_radar_marker(ax, radar_center, boresight, length):
    """Draw radar position (red dot) and boresight arrow."""
    ax.scatter(*radar_center, c="red", s=80, marker="^", zorder=10, depthshade=False)
    tip = radar_center + boresight * length
    ax.plot([radar_center[0], tip[0]],
            [radar_center[1], tip[1]],
            [radar_center[2], tip[2]], "r-", linewidth=2)


def _set_equal_aspect(ax, xyz):
    """Set equal aspect ratio for 3D axes."""
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def _create_figure(xyz_full, colors_full, xyz_highlight, colors_highlight,
                   radar_center, boresight, max_range, title_left, title_right,
                   elev, azim):
    """Create a side-by-side figure with two 3D views."""
    fig = plt.figure(figsize=(20, 9), dpi=150)

    # --- Left panel: original with FOV highlighted ---
    ax1 = fig.add_subplot(121, projection="3d")
    xyz_sub, col_sub = _subsample_for_plot(xyz_full, colors_full)
    _plot_pointcloud(ax1, xyz_sub, col_sub, point_size=0.2, alpha=0.4)
    _plot_radar_marker(ax1, radar_center, boresight, max_range * 0.15)
    _set_equal_aspect(ax1, xyz_full)
    ax1.set_title(title_left, fontsize=12, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.view_init(elev=elev, azim=azim)

    # --- Right panel: FPS result ---
    ax2 = fig.add_subplot(122, projection="3d")
    _plot_pointcloud(ax2, xyz_highlight, colors_highlight, point_size=1.5, alpha=0.8)
    _plot_radar_marker(ax2, radar_center, boresight, max_range * 0.15)
    _set_equal_aspect(ax2, xyz_full)
    ax2.set_title(title_right, fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig


def visualize_scene(pcl_path, radar_cfg, output_dir, target_n=25000, device="cuda:0"):
    """Generate before/after visualizations for one scene."""
    os.makedirs(output_dir, exist_ok=True)

    pcl = np.load(pcl_path)
    xyz_full = pcl[:, :3].astype(np.float64)
    n_orig = xyz_full.shape[0]

    # --- FOV mask ---
    fov_mask, radar_center, boresight, max_range = _compute_fov_mask(xyz_full, radar_cfg)
    n_in_fov = fov_mask.sum()

    # --- FPS on FOV-filtered points ---
    xyz_fov = xyz_full[fov_mask]
    if xyz_fov.shape[0] > target_n:
        fps_idx = _farthest_point_sampling_gpu(xyz_fov, target_n, device)
        xyz_fps = xyz_fov[fps_idx.cpu().numpy()]
    else:
        xyz_fps = xyz_fov.copy()

    # --- Colors for original: gray outside FOV, green inside ---
    colors_orig = np.full((n_orig, 4), [0.75, 0.75, 0.75, 0.15])  # gray, low alpha
    colors_orig[fov_mask] = [0.1, 0.7, 0.2, 0.6]                  # green, higher alpha

    # --- Colors for FPS: distance-based colormap ---
    dists = np.linalg.norm(xyz_fps - radar_center, axis=1)
    t = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12)
    colors_fps = cm.plasma(t)  # (N, 4) RGBA

    # --- Labels ---
    title_left = f"Original ({n_orig:,} pts)\nGreen = in FOV ({n_in_fov:,}), Gray = outside"
    title_right = f"After FPS ({xyz_fps.shape[0]:,} pts)\nColored by range from radar"

    # --- Top-down view ---
    # Determine azimuth angle that looks along -boresight projected to XY
    bore_az = np.degrees(np.arctan2(boresight[0], boresight[1]))

    fig_top = _create_figure(
        xyz_full, colors_orig, xyz_fps, colors_fps,
        radar_center, boresight, max_range,
        title_left, title_right,
        elev=75, azim=bore_az,
    )
    fig_top.savefig(os.path.join(output_dir, "topdown.png"), bbox_inches="tight")
    plt.close(fig_top)

    # --- Perspective view (behind radar, looking along boresight) ---
    fig_persp = _create_figure(
        xyz_full, colors_orig, xyz_fps, colors_fps,
        radar_center, boresight, max_range,
        title_left, title_right,
        elev=25, azim=bore_az + 30,
    )
    fig_persp.savefig(os.path.join(output_dir, "perspective.png"), bbox_inches="tight")
    plt.close(fig_persp)

    print(
        f"  Saved to {output_dir}/ "
        f"(original={n_orig:,}, in_FOV={n_in_fov:,}, FPS={xyz_fps.shape[0]:,})"
    )


def visualize_all_scenes(
    data_root="data",
    output_root="output/visualizations",
    target_n=25000,
    device="cuda:0",
):
    """Generate visualizations for all scenes."""
    scenes = sorted(glob.glob(os.path.join(data_root, "seq_*_frame_*/scene/pcl.npy")))
    print(f"Found {len(scenes)} scenes")

    for pcl_path in scenes:
        scene_dir = pcl_path.replace("/scene/pcl.npy", "")
        scene_name = os.path.basename(scene_dir)

        cfgs = sorted(glob.glob(
            os.path.join(scene_dir, "configs/cascaded_frame_*_aligned*.json")
        ))
        if not cfgs:
            print(f"  {scene_name}: no config found, skipping")
            continue

        radar_cfg = RadarConfig.from_json(cfgs[0])
        out_dir = os.path.join(output_root, scene_name)

        print(f"Processing {scene_name}...")
        visualize_scene(pcl_path, radar_cfg, out_dir, target_n, device)

    print("Done.")


if __name__ == "__main__":
    visualize_all_scenes()
