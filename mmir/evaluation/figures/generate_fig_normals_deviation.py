#!/usr/bin/env python3
"""Generate a single-row normal deviation figure for the supplement.

Single row showing per-vertex angular deviation between initial (LiDAR)
and optimized normals (inferno colormap: dark = small change, bright = large).

Usage:
    python -m mmir.evaluation.generate_fig_normals_deviation \
        --training_dir output/train_v11 \
        --output_dir output/postprocess_final_v11/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from PIL import Image

from .fig_common import (
    BACKGROUND_COLOR, FIG_WIDTH_INCHES, SCENE_SHORT_NAMES,
    add_rounded_bg, GridLayout,
)

SCENES = [
    "seq_0_frame_135", "seq_0_frame_390", "seq_1_frame_185",
    "seq_1_frame_438", "seq_2_frame_105", "seq_2_frame_160",
    "seq_2_frame_300",
]

DEFAULT_VIEW = "oblique_1"
VIEW_OVERRIDES = {
    "seq_1_frame_185": "oblique_4",
    "seq_2_frame_300": "oblique_4",
}

# Max angular deviation for normalization (degrees)
MAX_DEVIATION_DEG = 30.0


def rank_scenes_by_corr(training_dir):
    results = []
    for scene in SCENES:
        metrics_path = os.path.join(training_dir, scene, "best_metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        results.append((scene, m.get("cart_corr", 0)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _find_radar_config(scene_name, data_dir="data"):
    frame_id = scene_name.split("_")[-1]
    for suffix in ["_aligned_gpu", "_aligned_refined", "_aligned", ""]:
        path = os.path.join(data_dir, scene_name, "configs",
                            f"cascaded_frame_{frame_id}{suffix}.json")
        if os.path.exists(path):
            return path
    return None


def compute_angular_deviation(normals_a, normals_b):
    """Compute per-vertex angular deviation in degrees between two normal arrays."""
    # Normalize
    na = normals_a / np.maximum(np.linalg.norm(normals_a, axis=1, keepdims=True), 1e-8)
    nb = normals_b / np.maximum(np.linalg.norm(normals_b, axis=1, keepdims=True), 1e-8)
    # Clamp dot product for numerical stability
    dots = np.clip(np.sum(na * nb, axis=1), -1.0, 1.0)
    return np.rad2deg(np.arccos(np.abs(dots)))


def render_deviation(mesh_path, initial_normals, optimized_normals, view,
                     radar_config_path=None, width=2048, height=2048,
                     max_dev_deg=MAX_DEVIATION_DEG):
    """Render mesh colored by angular deviation between initial and optimized normals."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required")

    from .utils.visualization_open3d import (
        VIEWPOINTS, parse_radar_config, create_radar_visualization_geometries,
        _setup_renderer,
    )

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {mesh_path} has no triangles")
    mesh.compute_vertex_normals()

    n_vertices = len(mesh.vertices)

    # Handle vertex count mismatches
    init_n = initial_normals.copy()
    opt_n = optimized_normals.copy()
    for arr_name in ['init_n', 'opt_n']:
        arr = locals()[arr_name]
        if arr.shape[0] != n_vertices and abs(arr.shape[0] - n_vertices) <= 10:
            if arr.shape[0] < n_vertices:
                pad = np.tile(arr[-1:], (n_vertices - arr.shape[0], 1))
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:n_vertices]
            if arr_name == 'init_n':
                init_n = arr
            else:
                opt_n = arr

    # Compute angular deviation and map to colormap
    deviation_deg = compute_angular_deviation(init_n, opt_n)
    values = np.clip(deviation_deg / max_dev_deg, 0, 1)

    cmap = cm.get_cmap("inferno")
    colors_rgba = cmap(values)
    colors_rgb = colors_rgba[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_rgb)

    # Renderer setup
    renderer = _setup_renderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # Radar markers
    if radar_config_path:
        center, boresight = parse_radar_config(radar_config_path)
        if center is not None and boresight is not None:
            sphere, arrow = create_radar_visualization_geometries(
                center, boresight, sphere_radius=0.15, arrow_length=2.0
            )
            if sphere is not None:
                mat_s = o3d.visualization.rendering.MaterialRecord()
                mat_s.shader = "defaultLit"
                mat_s.base_color = [1.0, 0.0, 0.0, 1.0]
                renderer.scene.add_geometry("radar_sphere", sphere, mat_s)
            if arrow is not None:
                mat_a = o3d.visualization.rendering.MaterialRecord()
                mat_a.shader = "defaultLit"
                mat_a.base_color = [0.0, 1.0, 0.0, 1.0]
                renderer.scene.add_geometry("radar_arrow", arrow, mat_a)

    # Camera
    bbox = mesh.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()
    bbox_max = bbox.get_max_bound()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))

    vp = None
    for name, az, el, ds in VIEWPOINTS:
        if name == f"view_{view}" or name == view:
            vp = (az, el, ds)
            break
    if vp is None:
        vp = (45, 30, 1.5)

    az_rad = np.deg2rad(vp[0])
    el_rad = np.deg2rad(vp[1])
    distance = max_extent * vp[2]
    camera_pos = lookat + np.array([
        distance * np.cos(el_rad) * np.sin(az_rad),
        distance * np.cos(el_rad) * np.cos(az_rad),
        distance * np.sin(el_rad),
    ])

    min_camera_z = bbox_max[2] + extent[2] * 0.2
    if camera_pos[2] < min_camera_z:
        req_z = min_camera_z - lookat[2]
        req_xy = req_z / np.tan(el_rad) if el_rad > 0 else distance
        camera_pos = np.array([
            lookat[0] + req_xy * np.sin(az_rad),
            lookat[1] + req_xy * np.cos(az_rad),
            min_camera_z,
        ])

    renderer.setup_camera(60.0, lookat, camera_pos, np.array([0, 0, 1]))
    img_o3d = renderer.render_to_image()
    img_np = np.asarray(img_o3d)
    del renderer

    return Image.fromarray(img_np)


def _process_render(img):
    arr = np.array(img.convert("RGB"))
    bg_rgb = tuple(int(BACKGROUND_COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    light = np.all(arr > 210, axis=2)
    arr[light] = bg_rgb
    mask = ~light
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return Image.fromarray(arr)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 15
    r0, r1 = max(0, r0 - pad), min(arr.shape[0] - 1, r1 + pad)
    c0, c1 = max(0, c0 - pad), min(arr.shape[1] - 1, c1 + pad)
    return Image.fromarray(arr[r0:r1+1, c0:c1+1])


def generate_figure(training_dir, output_dir, data_dir="data", max_scenes=7):
    os.makedirs(output_dir, exist_ok=True)

    ranked = rank_scenes_by_corr(training_dir)[:max_scenes]
    n_scenes = len(ranked)
    if n_scenes == 0:
        print("No scenes found.")
        return None

    print(f"Generating normal deviation figure (single row) with {n_scenes} scenes:")

    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D required")

    all_images = {}
    for scene_name, corr in ranked:
        view = VIEW_OVERRIDES.get(scene_name, DEFAULT_VIEW)
        mesh_path = os.path.join(data_dir, scene_name, "scene", "mesh.ply")
        scene_training = os.path.join(training_dir, scene_name)
        radar_config = _find_radar_config(scene_name, data_dir)

        if not os.path.exists(mesh_path):
            print(f"  WARNING: mesh not found: {mesh_path}")
            all_images[scene_name] = None
            continue

        # Load initial normals from mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        initial_normals = np.asarray(mesh.vertex_normals).copy()

        # Load optimized normals
        normals_path = os.path.join(scene_training, "best_normals.npz")
        if os.path.exists(normals_path):
            data = np.load(normals_path)
            optimized_normals = data["normal_params"]
        else:
            print(f"  WARNING: no optimized normals for {scene_name}, skipping")
            all_images[scene_name] = None
            continue

        # Align lengths (Open3D may load ±1 vertex vs training)
        n_min = min(initial_normals.shape[0], optimized_normals.shape[0])
        dev = compute_angular_deviation(initial_normals[:n_min], optimized_normals[:n_min])
        print(f"  Rendering {scene_name} (view={view}, corr={corr:.3f}, "
              f"mean_dev={dev.mean():.2f}°, max_dev={dev.max():.1f}°)...")

        img = render_deviation(
            mesh_path, initial_normals, optimized_normals, view,
            radar_config_path=radar_config,
        )
        all_images[scene_name] = _process_render(img)

    # Get image aspect
    first_img = next((all_images[s] for s, _ in ranked if all_images[s] is not None), None)
    if first_img is None:
        raise RuntimeError("No images rendered")
    img_aspect = first_img.size[1] / first_img.size[0]

    n_rows = 1
    layout = GridLayout.from_image_aspect(
        n_rows, n_scenes, img_aspect=img_aspect,
        margin_in=0.08, col_gap_in=0.04, row_gap_in=0.04,
        header_in=0.18, label_w_in=0.50,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
    add_rounded_bg(fig)

    for col_idx, (scene_name, _) in enumerate(ranked):
        img = all_images[scene_name]
        if img is None:
            continue
        left, bottom, w, h = layout.cell_pos(0, col_idx)
        ax = fig.add_axes([left, bottom, w, h])
        ax.imshow(np.array(img), aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Column header
        fig.text(
            left + w / 2, layout.header_y(),
            SCENE_SHORT_NAMES.get(scene_name, scene_name),
            ha="center", va="top",
            fontsize=6, fontweight="bold",
            transform=fig.transFigure,
        )

    # Row label
    fig.text(
        layout.row_label_x(), layout.row_label_y(0),
        "Normal\nDeviation",
        ha="center", va="center",
        fontsize=6, fontweight="bold", rotation=90,
        transform=fig.transFigure,
    )

    out_pdf = os.path.join(output_dir, "normals_deviation_v1.pdf")
    out_png = os.path.join(output_dir, "normals_deviation_v1.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"\nSaved: {out_pdf}\nSaved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir", default="output/train_v11")
    parser.add_argument("--output_dir", default="output/postprocess_final_v11/figures")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_scenes", type=int, default=7)
    args = parser.parse_args()
    generate_figure(args.training_dir, args.output_dir, args.data_dir, args.max_scenes)


if __name__ == "__main__":
    main()
