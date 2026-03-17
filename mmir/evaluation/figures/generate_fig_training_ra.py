#!/usr/bin/env python3
"""Generate the training RA comparison figure for the paper.

Double-column figure with:
  - Columns: scenes (sorted by ours_corr, highest first)
  - Rows: Scene mesh (with radar marker), GT, Ours (mmIR), Sionna-RT (benchmark)
  - Linear-scale RA heatmaps from numpy data, no colorbar
  - Mesh row: mesh-only render (no lidar PCL) + red sphere + green arrow
  - Background tile with circular rounded corners (#ecebeb)

Usage:
    python -m mmir.evaluation.generate_fig_training_ra \
        --input_dir output/postprocess_final_v5/training_ra \
        --output_dir output/postprocess_final_v5/figures
"""

import argparse
import glob
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

CMAP = "hot"
MESH_VIEW = "oblique_1"
MESH_VIEW_OVERRIDES = {
    "seq_1_frame_185": "oblique_4",
    "seq_2_frame_300": "oblique_4",
    "seq_0_frame_390": "top",
}


def load_aggregate_metrics(input_dir: str) -> dict:
    with open(os.path.join(input_dir, "aggregate_metrics.json")) as f:
        return json.load(f)


def rank_scenes_by_corr(metrics: dict) -> list:
    per_scene = metrics["per_scene"]
    ranked = sorted(per_scene.items(), key=lambda x: x[1]["ours_corr"], reverse=True)
    return [(name, data) for name, data in ranked]


def ra_cart_to_linear(ra_cart: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_cart, dtype=np.float64)
    mn, mx = ra.min(), ra.max()
    return (ra - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(ra)


def load_scene_data(input_dir, scene_name, training_dir):
    scene_dir = os.path.join(input_dir, scene_name)
    data = {}
    for key, path in [
        ("gt", os.path.join(training_dir, scene_name, "ra_gt_cart.npy")),
        ("ours", os.path.join(scene_dir, "ours", "ra_rendered_cart.npy")),
        ("benchmark", os.path.join(scene_dir, "benchmark", "ra_rendered_cart.npy")),
    ]:
        data[key] = np.load(path) if os.path.exists(path) else None
    return data


def _find_radar_config(scene_name, data_dir="data"):
    """Find the radar config for a scene.

    Prefers dense_frame config (matches 3D occupancy renders), then falls
    back to cascaded configs.
    """
    frame_id = scene_name.split("_")[-1]
    # Prefer dense config (same as 3d_occupancy pipeline)
    dense_path = os.path.join(data_dir, scene_name, "configs",
                              f"dense_frame_{frame_id}.json")
    if os.path.exists(dense_path):
        return dense_path
    # Fallback: cascaded configs
    for suffix in ["", "_aligned_refined", "_aligned"]:
        path = os.path.join(data_dir, scene_name, "configs",
                            f"cascaded_frame_{frame_id}{suffix}.json")
        if os.path.exists(path):
            return path
    pattern = os.path.join(data_dir, scene_name, "configs", "cascaded_frame_*.json")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def render_mesh_views(scene_names, data_dir="data", cache_dir=None, view=MESH_VIEW):
    """Render mesh-only views with radar markers for all scenes.

    Returns dict mapping scene_name -> PIL Image (or None).
    """
    from .utils.visualization_open3d import render_mesh_only_views

    if cache_dir is None:
        cache_dir = os.path.join("output", "postprocess_final_v5", "mesh_only_renders")

    images = {}
    for scene_name in scene_names:
        scene_view = MESH_VIEW_OVERRIDES.get(scene_name, view)
        # Check cache first
        cached_path = os.path.join(cache_dir, scene_name,
                                   f"mesh_view_{scene_view}.png")
        if os.path.exists(cached_path):
            print(f"  {scene_name}: using cached mesh render ({scene_view})")
            images[scene_name] = _load_and_process_render(cached_path)
            continue

        mesh_path = os.path.join(data_dir, scene_name, "scene", "mesh.ply")
        if not os.path.exists(mesh_path):
            print(f"  WARNING: mesh not found: {mesh_path}")
            images[scene_name] = None
            continue

        config_path = _find_radar_config(scene_name, data_dir)
        print(f"  {scene_name}: rendering mesh-only view "
              f"(config={'found' if config_path else 'none'})")

        out_dir = os.path.join(cache_dir, scene_name)
        render_mesh_only_views(
            mesh_path, out_dir,
            radar_config_path=config_path,
            set_name="mesh",
        )

        rendered_path = os.path.join(out_dir, f"mesh_view_{scene_view}.png")
        if os.path.exists(rendered_path):
            images[scene_name] = _load_and_process_render(rendered_path)
        else:
            print(f"  WARNING: render failed for {scene_name}")
            images[scene_name] = None

    return images


def _load_and_process_render(path):
    """Load a rendered PNG image, replace white background with figure bg."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    bg_rgb = tuple(int(BACKGROUND_COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    light = np.all(arr > 210, axis=2)
    arr[light] = bg_rgb
    mask = ~light
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return Image.fromarray(arr)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 10
    r0, r1 = max(0, r0 - pad), min(arr.shape[0] - 1, r1 + pad)
    c0, c1 = max(0, c0 - pad), min(arr.shape[1] - 1, c1 + pad)
    return Image.fromarray(arr[r0:r1+1, c0:c1+1])


def generate_figure(
    input_dir, output_dir,
    training_dir="output/train_v4",
    data_dir="data",
    max_scenes=7, include_mesh=True,
):
    os.makedirs(output_dir, exist_ok=True)

    metrics = load_aggregate_metrics(input_dir)
    ranked = rank_scenes_by_corr(metrics)[:max_scenes]
    n_scenes = len(ranked)

    print(f"Generating RA comparison figure with {n_scenes} scenes:")
    for name, data in ranked:
        print(f"  {name}: ours_corr={data['ours_corr']:.3f}")

    all_data = {}
    for sn, _ in ranked:
        all_data[sn] = load_scene_data(input_dir, sn, training_dir)

    mesh_images = {}
    if include_mesh:
        scene_names = [sn for sn, _ in ranked]
        print("\nRendering mesh-only views:")
        mesh_images = render_mesh_views(scene_names, data_dir=data_dir)
        if not any(mesh_images.values()):
            print("  No mesh images available, disabling mesh row")
            include_mesh = False

    ra_rows = ["gt", "ours", "benchmark"]
    ra_labels = ["GT", "mmIR\n(Ours)", "Sionna-RT"]
    if include_mesh:
        all_rows = ["mesh"] + ra_rows
        all_labels = ["Scene"] + ra_labels
    else:
        all_rows = ra_rows
        all_labels = ra_labels

    n_rows = len(all_rows)

    # Layout: square cells (RA images are 399x399), inches-based
    layout = GridLayout.from_image_aspect(
        n_rows, n_scenes, img_aspect=1.0,
        margin_in=0.08, col_gap_in=0.03, row_gap_in=0.03,
        header_in=0.16, label_w_in=0.38,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
    add_rounded_bg(fig)

    for col_idx, (scene_name, _) in enumerate(ranked):
        scene_data = all_data[scene_name]

        for row_idx, row_key in enumerate(all_rows):
            left, bottom, w, h = layout.cell_pos(row_idx, col_idx)
            ax = fig.add_axes([left, bottom, w, h])

            if row_key == "mesh":
                mesh_img = mesh_images.get(scene_name)
                if mesh_img is not None:
                    ax.imshow(np.array(mesh_img), aspect="equal")
                    ax.set_facecolor(BACKGROUND_COLOR)
                else:
                    ax.set_facecolor(BACKGROUND_COLOR)
            else:
                ra_cart = scene_data.get(row_key)
                if ra_cart is not None:
                    ax.imshow(
                        ra_cart_to_linear(ra_cart), cmap=CMAP,
                        aspect="equal", origin="lower",
                        vmin=0.0, vmax=1.0, interpolation="bilinear",
                    )
                else:
                    ax.set_facecolor("black")

            ax.set_xticks([])
            ax.set_yticks([])
            if row_key == "mesh":
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                for sp in ax.spines.values():
                    sp.set_linewidth(0.3)
                    sp.set_color("#666666")

        # Column header
        left, _, w, _ = layout.cell_pos(0, col_idx)
        fig.text(
            left + w / 2, layout.header_y(),
            SCENE_SHORT_NAMES.get(scene_name, scene_name),
            ha="center", va="top",
            fontsize=5.5, fontweight="bold",
            transform=fig.transFigure,
        )

    # Row labels
    for row_idx, label in enumerate(all_labels):
        fig.text(
            layout.row_label_x(), layout.row_label_y(row_idx), label,
            ha="center", va="center",
            fontsize=5.5, fontweight="bold", rotation=90,
            transform=fig.transFigure,
        )

    out_pdf = os.path.join(output_dir, "training_single_v2.pdf")
    out_png = os.path.join(output_dir, "training_single_v2.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"\nSaved: {out_pdf}\nSaved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output/postprocess_final_v5/training_ra")
    parser.add_argument("--output_dir", default="output/postprocess_final_v5/figures")
    parser.add_argument("--training_dir", default="output/train_v4")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_scenes", type=int, default=7)
    parser.add_argument("--no_mesh", action="store_true")
    args = parser.parse_args()
    generate_figure(args.input_dir, args.output_dir, args.training_dir,
                    args.data_dir, args.max_scenes, not args.no_mesh)


if __name__ == "__main__":
    main()
