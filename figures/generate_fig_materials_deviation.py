#!/usr/bin/env python3
"""Generate a single-row material deviation figure for the supplement.

Single row showing per-vertex deviation from ITU defaults (inferno colormap)
for optimized materials only. Columns: scenes sorted by correlation.

Usage:
    python -m mmir.evaluation.generate_fig_materials_deviation \
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

    print(f"Generating material deviation figure (single row) with {n_scenes} scenes:")

    from .visualize_materials import render_material_visualization

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

        print(f"  Rendering {scene_name} (view={view}, corr={corr:.3f})...")
        img = render_material_visualization(
            mesh_path, scene_training,
            view=view, param_name="overall_change", colormap="inferno",
            width=2048, height=2048, radar_config_path=radar_config,
        )
        all_images[scene_name] = _process_render(img)

    # Get image aspect from first available
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
        "Material\nDeviation",
        ha="center", va="center",
        fontsize=6, fontweight="bold", rotation=90,
        transform=fig.transFigure,
    )

    out_pdf = os.path.join(output_dir, "materials_deviation_v1.pdf")
    out_png = os.path.join(output_dir, "materials_deviation_v1.png")
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
