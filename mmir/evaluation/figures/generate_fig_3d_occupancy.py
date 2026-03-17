#!/usr/bin/env python3
"""Generate the 3D occupancy comparison figure for the paper.

Double-column figure with:
  - Columns: scenes where ours outperforms cascade (sorted by improvement)
  - Rows: Ours (dense virtual array), Cascaded radar
  - No quantitative annotations
  - Background tile with circular rounded corners (#ecebeb)

Usage:
    python -m mmir.evaluation.generate_fig_3d_occupancy \
        --input_dir output/postprocess_final_v5/3d_occupancy \
        --output_dir output/postprocess_final_v5/figures
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

DEFAULT_VIEW = "oblique_1"
VIEW_OVERRIDES = {
    "seq_1_frame_185": "oblique_4",
    "seq_2_frame_300": "oblique_4",
}


def load_aggregate_metrics(input_dir):
    with open(os.path.join(input_dir, "aggregate_metrics.json")) as f:
        return json.load(f)


def rank_scenes_by_improvement(metrics):
    per_scene = metrics["per_scene"]
    results = []
    for name, data in per_scene.items():
        ours = data.get("relative_chamfer_distance")
        casc = data.get("coloradar_relative_chamfer_distance")
        if ours is not None and casc is not None:
            imp = casc - ours
            pct = imp / casc * 100 if casc > 0 else 0
            results.append((name, data, imp, pct))
    results.sort(key=lambda x: x[2], reverse=True)
    return [(n, d, i, p) for n, d, i, p in results if i > 0]


def process_3d_render(img, border_px=15):
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
    r0, r1 = max(0, r0 - border_px), min(arr.shape[0]-1, r1 + border_px)
    c0, c1 = max(0, c0 - border_px), min(arr.shape[1]-1, c1 + border_px)
    return Image.fromarray(arr[r0:r1+1, c0:c1+1])


def load_scene_images(input_dir, scene_name, view):
    scene_dir = os.path.join(input_dir, scene_name)
    images = {}
    for key, prefix in [("ours", "dense_radar"), ("cascade", "cascade_radar")]:
        path = os.path.join(scene_dir, prefix, f"{prefix}_view_{view}.png")
        if os.path.exists(path):
            images[key] = process_3d_render(Image.open(path).convert("RGB"))
        else:
            print(f"  Warning: missing {path}")
            images[key] = None
    return images


def generate_figure(input_dir, output_dir, max_scenes=4, view=DEFAULT_VIEW):
    os.makedirs(output_dir, exist_ok=True)

    metrics = load_aggregate_metrics(input_dir)
    ranked = rank_scenes_by_improvement(metrics)[:max_scenes]
    n_scenes = len(ranked)
    if n_scenes == 0:
        print("No scenes where ours outperforms cascade.")
        return None

    print(f"Generating 3D occupancy figure with {n_scenes} scenes:")
    for name, data, imp, pct in ranked:
        print(f"  {name}: ours={data['relative_chamfer_distance']:.3f}, "
              f"cascade={data['coloradar_relative_chamfer_distance']:.3f} ({pct:.0f}% better)")

    all_images = {}
    for sn, _, _, _ in ranked:
        scene_view = VIEW_OVERRIDES.get(sn, view)
        all_images[sn] = load_scene_images(input_dir, sn, scene_view)

    # Determine image aspect from first available
    first_img = None
    for sn, _, _, _ in ranked:
        if all_images[sn]["ours"] is not None:
            first_img = all_images[sn]["ours"]
            break
    if first_img is None:
        raise RuntimeError("No images found")
    img_aspect = first_img.size[1] / first_img.size[0]

    n_rows = 2
    layout = GridLayout.from_image_aspect(
        n_rows, n_scenes, img_aspect=img_aspect,
        margin_in=0.08, col_gap_in=0.04, row_gap_in=0.04,
        header_in=0.18, label_w_in=0.50,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
    add_rounded_bg(fig)

    row_map = {"ours": 0, "cascade": 1}
    for col_idx, (scene_name, _, _, _) in enumerate(ranked):
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

        # Column header — scene name only
        left, _, w, _ = layout.cell_pos(0, col_idx)
        fig.text(
            left + w / 2, layout.header_y(),
            SCENE_SHORT_NAMES.get(scene_name, scene_name),
            ha="center", va="top",
            fontsize=6, fontweight="bold",
            transform=fig.transFigure,
        )

    # Row labels
    for label, row_idx in [("Dense Array\n(Ours)", 0), ("Cascaded\nRadar", 1)]:
        fig.text(
            layout.row_label_x(), layout.row_label_y(row_idx), label,
            ha="center", va="center",
            fontsize=6, fontweight="bold", rotation=90,
            transform=fig.transFigure,
        )

    out_pdf = os.path.join(output_dir, "test_3d_v1.pdf")
    out_png = os.path.join(output_dir, "test_3d_v1.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"\nSaved: {out_pdf}\nSaved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output/postprocess_final_v11/3d_occupancy")
    parser.add_argument("--output_dir", default="output/postprocess_final_v11/figures")
    parser.add_argument("--max_scenes", type=int, default=4)
    parser.add_argument("--view", default=DEFAULT_VIEW,
                        choices=["front", "back", "left", "right", "top",
                                 "oblique_1", "oblique_2", "oblique_3", "oblique_4"])
    args = parser.parse_args()
    generate_figure(args.input_dir, args.output_dir, args.max_scenes, args.view)


if __name__ == "__main__":
    main()
