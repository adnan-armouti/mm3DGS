#!/usr/bin/env python3
"""Generate the log-loss vs no-log-loss comparison figure for the supplement.

Double-column figure with:
  - Columns: scenes (sorted by no-log correlation, highest first)
  - Rows: Scene mesh, GT, No-log (ours recommended), Log-loss (original)
  - Linear-scale RA heatmaps from numpy data, no colorbar
  - Correlation annotation on each rendered RA cell

Usage:
    python -m mmir.evaluation.supplement_figs.generate_fig_log_loss_comparison \
        --nolog_input_dir output/postprocess_final_v13/training_ra \
        --log_input_dir output/postprocess_final_v11/training_ra \
        --nolog_training_dir output/train_v13 \
        --log_training_dir output/train_v11 \
        --output_dir output/postprocess_final_v13/figures
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

# Use relative imports when run as module, absolute when run as script
try:
    from ..fig_common import (
        BACKGROUND_COLOR, FIG_WIDTH_INCHES, SCENE_SHORT_NAMES,
        add_rounded_bg, GridLayout,
    )
    from ..generate_fig_training_ra import (
        render_mesh_views, _load_and_process_render, _find_radar_config,
        MESH_VIEW, MESH_VIEW_OVERRIDES,
    )
except ImportError:
    from mmir.evaluation.fig_common import (
        BACKGROUND_COLOR, FIG_WIDTH_INCHES, SCENE_SHORT_NAMES,
        add_rounded_bg, GridLayout,
    )
    from mmir.evaluation.figures.generate_fig_training_ra import (
        render_mesh_views, _load_and_process_render, _find_radar_config,
        MESH_VIEW, MESH_VIEW_OVERRIDES,
    )

CMAP = "hot"


def ra_cart_to_linear(ra_cart: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_cart, dtype=np.float64)
    mn, mx = ra.min(), ra.max()
    return (ra - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(ra)


def load_metrics(input_dir: str) -> dict:
    path = os.path.join(input_dir, "aggregate_metrics.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"per_scene": {}}


def load_ra_cart(training_dir, scene_name, key="ra_rendered_cart"):
    """Load RA cartesian from training dir or postprocess dir."""
    path = os.path.join(training_dir, scene_name, f"{key}.npy")
    if os.path.isfile(path):
        return np.load(path)
    return None


def generate_figure(
    nolog_input_dir, log_input_dir,
    nolog_training_dir, log_training_dir,
    output_dir,
    data_dir="data",
    max_scenes=7,
    include_mesh=True,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics for both methods
    nolog_metrics = load_metrics(nolog_input_dir)
    log_metrics = load_metrics(log_input_dir)

    # Use no-log scenes, ranked by correlation
    per_scene_nolog = nolog_metrics.get("per_scene", {})
    ranked = sorted(per_scene_nolog.items(),
                    key=lambda x: x[1].get("ours_corr", 0), reverse=True)
    ranked = ranked[:max_scenes]
    n_scenes = len(ranked)

    print(f"Generating log-loss comparison figure with {n_scenes} scenes:")
    per_scene_log = log_metrics.get("per_scene", {})
    for name, data in ranked:
        nolog_corr = data.get("ours_corr", 0)
        log_corr = per_scene_log.get(name, {}).get("ours_corr", 0)
        print(f"  {name}: no-log={nolog_corr:.3f}, log={log_corr:.3f}")

    # Load RA data
    all_data = {}
    for sn, _ in ranked:
        gt = load_ra_cart(nolog_training_dir, sn, "ra_gt_cart")
        nolog = load_ra_cart(nolog_training_dir, sn, "ra_rendered_cart")
        if nolog is None:
            nolog_pp = os.path.join(nolog_input_dir, sn, "ours", "ra_rendered_cart.npy")
            if os.path.isfile(nolog_pp):
                nolog = np.load(nolog_pp)
        log_ra = load_ra_cart(log_training_dir, sn, "ra_rendered_cart")
        if log_ra is None:
            log_pp = os.path.join(log_input_dir, sn, "ours", "ra_rendered_cart.npy")
            if os.path.isfile(log_pp):
                log_ra = np.load(log_pp)
        all_data[sn] = {"gt": gt, "nolog": nolog, "log": log_ra}

    # Mesh images
    mesh_images = {}
    if include_mesh:
        scene_names = [sn for sn, _ in ranked]
        print("\nRendering mesh-only views:")
        mesh_images = render_mesh_views(scene_names, data_dir=data_dir)
        if not any(mesh_images.values()):
            print("  No mesh images available, disabling mesh row")
            include_mesh = False

    # Row layout
    ra_rows = ["gt", "nolog", "log"]
    ra_labels = ["GT", "Without\nLog Loss", "With\nLog Loss"]
    if include_mesh:
        all_rows = ["mesh"] + ra_rows
        all_labels = ["Scene"] + ra_labels
    else:
        all_rows = ra_rows
        all_labels = ra_labels

    n_rows = len(all_rows)

    # Layout
    layout = GridLayout.from_image_aspect(
        n_rows, n_scenes, img_aspect=1.0,
        margin_in=0.08, col_gap_in=0.03, row_gap_in=0.03,
        header_in=0.16, label_w_in=0.42,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
    add_rounded_bg(fig)

    for col_idx, (scene_name, scene_metrics) in enumerate(ranked):
        scene_data = all_data[scene_name]
        log_scene_metrics = per_scene_log.get(scene_name, {})

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
                    # Add correlation annotation for rendered rows
                    if row_key == "nolog":
                        corr = scene_metrics.get("ours_corr", 0)
                        ax.text(0.02, 0.98, f"{corr:.3f}",
                                transform=ax.transAxes, fontsize=4.5,
                                va="top", ha="left", color="white",
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          facecolor="black", alpha=0.6,
                                          edgecolor="none"))
                    elif row_key == "log":
                        corr = log_scene_metrics.get("ours_corr", 0)
                        ax.text(0.02, 0.98, f"{corr:.3f}",
                                transform=ax.transAxes, fontsize=4.5,
                                va="top", ha="left", color="white",
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          facecolor="black", alpha=0.6,
                                          edgecolor="none"))
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

    out_pdf = os.path.join(output_dir, "log_loss_comparison.pdf")
    out_png = os.path.join(output_dir, "log_loss_comparison.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"\nSaved: {out_pdf}\nSaved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Generate log-loss comparison figure for supplement")
    parser.add_argument("--nolog_input_dir",
                        default="output/postprocess_final_v13/training_ra")
    parser.add_argument("--log_input_dir",
                        default="output/postprocess_final_v11/training_ra")
    parser.add_argument("--nolog_training_dir",
                        default="output/train_v13")
    parser.add_argument("--log_training_dir",
                        default="output/train_v11")
    parser.add_argument("--output_dir",
                        default="output/postprocess_final_v13/figures")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_scenes", type=int, default=7)
    parser.add_argument("--no_mesh", action="store_true")
    args = parser.parse_args()

    generate_figure(
        args.nolog_input_dir, args.log_input_dir,
        args.nolog_training_dir, args.log_training_dir,
        args.output_dir, args.data_dir, args.max_scenes, not args.no_mesh,
    )


if __name__ == "__main__":
    main()
