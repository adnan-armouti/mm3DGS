#!/usr/bin/env python3
"""Generate the material visualization comparison figure for the supplement.

Double-column figure with:
  - Columns: scenes (sorted by correlation, highest first)
  - Rows: Initial materials (ITU defaults), Optimized materials (from training)
  - Coloring: per-vertex deviation from ITU defaults (inferno colormap)
  - Background tile with circular rounded corners (#ecebeb)

Usage:
    python -m mmir.evaluation.generate_fig_materials \
        --training_dir output/train_v11 \
        --output_dir output/postprocess_final_v11/figures
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

# Apply NeurIPS paper typography (Times serif) to all figures.
apply_paper_font()

    apply_paper_font,
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


def rank_scenes_by_corr(training_dir):
    """Sort scenes by best cart_corr (highest first)."""
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
    """Find radar config for a scene."""
    frame_id = scene_name.split("_")[-1]
    for suffix in ["_aligned_gpu", "_aligned_refined", "_aligned", ""]:
        path = os.path.join(data_dir, scene_name, "configs",
                            f"cascaded_frame_{frame_id}{suffix}.json")
        if os.path.exists(path):
            return path
    return None


def render_material_pair(mesh_path, training_dir, scene_name, view,
                         radar_config_path=None, width=2048, height=2048):
    """Render initial (uniform default) and optimized material visualizations.

    Returns: (initial_img, optimized_img) as PIL Images.
    """
    from .visualize_materials import (
        render_material_visualization, compute_overall_change,
        PHYSICS_DEFAULTS, PHYSICS_RANGES, PARAM_NAMES,
    )

    # Optimized materials
    optimized_img = render_material_visualization(
        mesh_path, training_dir,
        view=view, param_name="overall_change", colormap="inferno",
        width=width, height=height, radar_config_path=radar_config_path,
    )

    # Initial materials: all at ITU defaults → overall_change = 0 everywhere
    # Re-use the same renderer but with a fake training_dir containing default params
    import tempfile
    from .material_loader import load_our_physics_params

    physics_opt, _ = load_our_physics_params(training_dir)
    n_vertices = physics_opt.shape[0]

    # Create default params array
    default_physics = np.zeros((n_vertices, 6), dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        default_physics[:, i] = PHYSICS_DEFAULTS[name]

    with tempfile.TemporaryDirectory() as tmpdir:
        np.savez(os.path.join(tmpdir, "best_materials.npz"),
                 raw_params=default_physics,
                 **{name: default_physics[:, i] for i, name in enumerate(PARAM_NAMES)})
        initial_img = render_material_visualization(
            mesh_path, tmpdir,
            view=view, param_name="overall_change", colormap="inferno",
            width=width, height=height, radar_config_path=radar_config_path,
        )

    return initial_img, optimized_img


def _process_render(img):
    """Replace white/light background with figure background color and crop."""
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

    print(f"Generating material comparison figure with {n_scenes} scenes:")
    for name, corr in ranked:
        print(f"  {name}: corr={corr:.3f}")

    # Render all scenes
    all_images = {}
    for scene_name, _ in ranked:
        view = VIEW_OVERRIDES.get(scene_name, DEFAULT_VIEW)
        mesh_path = os.path.join(data_dir, scene_name, "scene", "mesh.ply")
        scene_training = os.path.join(training_dir, scene_name)
        radar_config = _find_radar_config(scene_name, data_dir)

        if not os.path.exists(mesh_path):
            print(f"  WARNING: mesh not found: {mesh_path}")
            all_images[scene_name] = {"initial": None, "optimized": None}
            continue

        print(f"  Rendering {scene_name} (view={view})...")
        initial_img, optimized_img = render_material_pair(
            mesh_path, scene_training, scene_name, view,
            radar_config_path=radar_config,
        )
        all_images[scene_name] = {
            "initial": _process_render(initial_img),
            "optimized": _process_render(optimized_img),
        }

    # Determine image aspect from first available
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

    out_pdf = os.path.join(output_dir, "materials_comparison_v1.pdf")
    out_png = os.path.join(output_dir, "materials_comparison_v1.png")
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
