#!/usr/bin/env python3
"""Generate a two-row material RGB figure for the supplement.

Two rows (Initial / Optimized) showing per-vertex material parameters
mapped to RGB channels:
  - R: Permittivity (eps_real, eps_imag) — electromagnetic properties
  - G: Roughness (sigma_h, l_c) — scattering properties
  - B: Structure (tau, thickness) — slab/layer properties

Each parameter is normalized to [0,1] within its physical range
(log-space for log-scale params), then the two parameters per channel
are averaged.

Usage:
    python -m mmir.evaluation.generate_fig_materials_rgb \
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

PARAM_NAMES = ["eps_real", "eps_imag", "sigma_h", "l_c", "tau", "thickness"]

PHYSICS_DEFAULTS = {
    "eps_real": 5.31,
    "eps_imag": 0.29,
    "sigma_h": 1e-4,
    "l_c": 0.01,
    "tau": 0.5,
    "thickness": 0.1,
}

PHYSICS_RANGES = {
    "eps_real": (1.5, 10.0),
    "eps_imag": (1e-3, 9e6),
    "sigma_h": (1e-7, 1e-3),
    "l_c": (5e-4, 0.1),
    "tau": (0.05, 0.95),
    "thickness": (1e-3, 0.3),
}

# Channel grouping: R=permittivity, G=roughness, B=structure
CHANNEL_GROUPS = {
    "R": ["eps_real", "eps_imag"],
    "G": ["sigma_h", "l_c"],
    "B": ["tau", "thickness"],
}


def _normalize_param(values, name):
    """Normalize parameter values to [0, 1]."""
    vmin, vmax = PHYSICS_RANGES[name]
    if name in ("eps_real", "tau"):
        return np.clip((values - vmin) / (vmax - vmin), 0, 1)
    else:
        log_vals = np.log10(np.clip(values, vmin, vmax))
        return np.clip((log_vals - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin)), 0, 1)


def physics_to_rgb(physics_params):
    """Map (n_vertices, 6) physics params to (n_vertices, 3) RGB in [0, 1].

    R = mean(norm(eps_real), norm(eps_imag))
    G = mean(norm(sigma_h), norm(l_c))
    B = mean(norm(tau), norm(thickness))
    """
    rgb = np.zeros((physics_params.shape[0], 3), dtype=np.float64)
    for ch_idx, (ch_name, param_names) in enumerate(CHANNEL_GROUPS.items()):
        channel_vals = []
        for pname in param_names:
            pidx = PARAM_NAMES.index(pname)
            channel_vals.append(_normalize_param(physics_params[:, pidx], pname))
        rgb[:, ch_idx] = np.mean(channel_vals, axis=0)
    return rgb


def render_materials_rgb(mesh_path, physics_params, view,
                         radar_config_path=None, width=2048, height=2048):
    """Render mesh colored by RGB-encoded material parameters.

    Args:
        mesh_path: Path to .ply mesh.
        physics_params: (n_vertices, 6) physics-space material array.
        view: Viewpoint name.
        radar_config_path: Optional radar config for markers.
        width, height: Render resolution.

    Returns:
        PIL Image (RGBA).
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required for material visualization")

    from .utils.visualization_open3d import (
        VIEWPOINTS, parse_radar_config, create_radar_visualization_geometries,
        _setup_renderer,
    )

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {mesh_path} has no triangles")
    mesh.compute_vertex_normals()

    n_vertices = len(mesh.vertices)

    # Handle small vertex count mismatches
    params = physics_params.copy()
    if params.shape[0] != n_vertices and abs(params.shape[0] - n_vertices) <= 10:
        if params.shape[0] < n_vertices:
            pad = np.tile(params[-1:], (n_vertices - params.shape[0], 1))
            params = np.vstack([params, pad])
        else:
            params = params[:n_vertices]

    # Compute RGB colors
    colors_rgb = physics_to_rgb(params)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_rgb)

    # Set up renderer
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

    print(f"Generating material RGB figure (2 rows) with {n_scenes} scenes:")
    print("  Channel encoding: R=permittivity, G=roughness, B=structure")

    from .material_loader import load_our_physics_params

    all_images = {}
    for scene_name, corr in ranked:
        view = VIEW_OVERRIDES.get(scene_name, DEFAULT_VIEW)
        mesh_path = os.path.join(data_dir, scene_name, "scene", "mesh.ply")
        scene_training = os.path.join(training_dir, scene_name)
        radar_config = _find_radar_config(scene_name, data_dir)

        if not os.path.exists(mesh_path):
            print(f"  WARNING: mesh not found: {mesh_path}")
            all_images[scene_name] = {"initial": None, "optimized": None}
            continue

        print(f"  Rendering {scene_name} (view={view}, corr={corr:.3f})...")

        # Load optimized physics params
        physics_opt, _ = load_our_physics_params(scene_training)
        n_vtx = physics_opt.shape[0]

        # Create initial (ITU default) params
        physics_init = np.zeros((n_vtx, 6), dtype=np.float64)
        for i, name in enumerate(PARAM_NAMES):
            physics_init[:, i] = PHYSICS_DEFAULTS[name]

        # Render both
        initial_img = render_materials_rgb(
            mesh_path, physics_init, view,
            radar_config_path=radar_config,
        )
        optimized_img = render_materials_rgb(
            mesh_path, physics_opt, view,
            radar_config_path=radar_config,
        )

        all_images[scene_name] = {
            "initial": _process_render(initial_img),
            "optimized": _process_render(optimized_img),
        }

    # Get image aspect
    first_img = None
    for scene_name, _ in ranked:
        if all_images[scene_name]["optimized"] is not None:
            first_img = all_images[scene_name]["optimized"]
            break
    if first_img is None:
        raise RuntimeError("No images rendered")
    img_aspect = first_img.size[1] / first_img.size[0]

    n_rows = 2
    layout = GridLayout.from_image_aspect(
        n_rows, n_scenes, img_aspect=img_aspect,
        margin_in=0.08, col_gap_in=0.04, row_gap_in=0.04,
        header_in=0.18, label_w_in=0.50,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
    add_rounded_bg(fig)

    row_map = {"initial": 0, "optimized": 1}
    for col_idx, (scene_name, _) in enumerate(ranked):
        imgs = all_images[scene_name]
        for key, row_idx in row_map.items():
            img = imgs.get(key)
            if img is None:
                continue
            left, bottom, w, h = layout.cell_pos(row_idx, col_idx)
            ax = fig.add_axes([left, bottom, w, h])
            ax.imshow(np.array(img), aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Column header
        left, _, w, _ = layout.cell_pos(0, col_idx)
        fig.text(
            left + w / 2, layout.header_y(),
            SCENE_SHORT_NAMES.get(scene_name, scene_name),
            ha="center", va="top",
            fontsize=6, fontweight="bold",
            transform=fig.transFigure,
        )

    # Row labels
    for label, row_idx in [("Initial\n(ITU Default)", 0), ("Optimized", 1)]:
        fig.text(
            layout.row_label_x(), layout.row_label_y(row_idx), label,
            ha="center", va="center",
            fontsize=6, fontweight="bold", rotation=90,
            transform=fig.transFigure,
        )

    out_pdf = os.path.join(output_dir, "materials_rgb_v1.pdf")
    out_png = os.path.join(output_dir, "materials_rgb_v1.png")
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
