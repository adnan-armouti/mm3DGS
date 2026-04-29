#!/usr/bin/env python3
"""Generate the antenna beam pattern comparison figure for the supplement.

Double-column figure with:
  - Columns: scenes (sorted by correlation, highest first)
  - Rows: Initial reference patterns, Optimized patterns (from training)
  - Each cell: polar plot with TX E/H-plane and RX E/H-plane
  - Background tile with circular rounded corners (#ecebeb)

The initial reference patterns are the same across all scenes since
train_simple_single_frame_final.py uses shared TX/RX patterns initialized
from assets/antenna_pattern/MMWCAS/. Each scene independently optimizes
a copy, producing scene-specific optimized patterns.

Usage:
    python -m mmir.evaluation.generate_fig_beam_patterns \
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

# Default initial pattern files (shared mode)
DEFAULT_TX_PATTERN = "assets/antenna_pattern/MMWCAS/tx1_76.npy"
DEFAULT_RX_PATTERN = "assets/antenna_pattern/MMWCAS/rx1_76.npy"


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


def load_initial_patterns(tx_path=DEFAULT_TX_PATTERN, rx_path=DEFAULT_RX_PATTERN):
    """Load initial reference antenna patterns.

    Returns dict with keys tx_E, tx_H, rx_E, rx_H (each shape (361,) in dBi).
    """
    tx_pat = np.load(tx_path)  # (361, 2): col0=E-plane, col1=H-plane
    rx_pat = np.load(rx_path)
    return {
        "tx_E": tx_pat[:, 0], "tx_H": tx_pat[:, 1],
        "rx_E": rx_pat[:, 0], "rx_H": rx_pat[:, 1],
    }


def load_optimized_patterns(training_dir, db_floor=None):
    """Load optimized patterns from best_patterns.npz.

    The training script saves E_plane_linear and H_plane_linear
    (linear amplitude). We convert to dBi for consistent comparison.

    Args:
        training_dir: Path to training output directory.
        db_floor: Floor dB value. If None, no flooring is applied.
    """
    path = os.path.join(training_dir, "best_patterns.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path)
    result = {}
    for key_prefix, pat_key in [("tx", "tx"), ("rx", "rx")]:
        e_key = f"{pat_key}_E_plane"
        h_key = f"{pat_key}_H_plane"
        if e_key in data and h_key in data:
            e_lin = np.array(data[e_key], dtype=np.float64)
            h_lin = np.array(data[h_key], dtype=np.float64)
            e_db = 20.0 * np.log10(np.maximum(e_lin, 1e-10))
            h_db = 20.0 * np.log10(np.maximum(h_lin, 1e-10))
            if db_floor is not None:
                e_db = np.maximum(e_db, db_floor)
                h_db = np.maximum(h_db, db_floor)
            result[f"{key_prefix}_E"] = e_db
            result[f"{key_prefix}_H"] = h_db
    return result if result else None


def draw_pattern_polar(ax, patterns, title=None):
    """Draw TX/RX E-plane and H-plane beam patterns on a polar axes.

    Args:
        ax: Matplotlib polar Axes.
        patterns: dict with tx_E, tx_H, rx_E, rx_H (dBi, shape (361,)).
        title: Optional title for the subplot.
    """
    angles_rad = np.deg2rad(np.linspace(-180, 180, 361))

    all_gains = np.concatenate([patterns[k] for k in ["tx_E", "tx_H", "rx_E", "rx_H"]])
    min_db = np.floor(all_gains.min() / 5) * 5

    ax.plot(angles_rad, patterns["tx_E"] - min_db, color="#c62828", lw=0.7,
            label="TX El", alpha=0.85)
    ax.plot(angles_rad, patterns["tx_H"] - min_db, color="#c62828", lw=0.7,
            label="TX Az", alpha=0.85, ls="--")
    ax.plot(angles_rad, patterns["rx_E"] - min_db, color="#1565c0", lw=0.7,
            label="RX El", alpha=0.85)
    ax.plot(angles_rad, patterns["rx_H"] - min_db, color="#1565c0", lw=0.7,
            label="RX Az", alpha=0.85, ls="--")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    max_shifted = all_gains.max() - min_db
    r_ticks = np.arange(0, max_shifted + 1, 10)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{t + min_db:.0f}" for t in r_ticks])
    ax.set_rlim(0, max_shifted)

    ax.set_rlabel_position(0)
    ax.tick_params(axis='y', labelsize=3, pad=-1)
    ax.tick_params(axis='x', labelsize=3, pad=-7)
    ax.grid(True, lw=0.15, alpha=0.4)
    ax.spines["polar"].set_linewidth(0.3)

    bg_rgb = tuple(int(BACKGROUND_COLOR.lstrip("#")[i:i+2], 16) / 255
                   for i in (0, 2, 4))
    ax.set_facecolor(bg_rgb)

    if title:
        ax.set_title(title, fontsize=5, fontweight="bold", pad=8)


def generate_figure(training_dir, output_dir, data_dir="data", max_scenes=7):
    os.makedirs(output_dir, exist_ok=True)

    ranked = rank_scenes_by_corr(training_dir)[:max_scenes]
    n_scenes = len(ranked)
    if n_scenes == 0:
        print("No scenes found.")
        return None

    print(f"Generating beam pattern comparison figure with {n_scenes} scenes:")

    # Load initial patterns (same for all scenes)
    initial = load_initial_patterns()

    # Compute dB floor from initial patterns so optimized plots use the same range
    db_floor = min(v.min() for v in initial.values())
    print(f"  Initial pattern floor: {db_floor:.1f} dB")

    # Load optimized patterns per scene, floored to match initial range
    optimized = {}
    for scene_name, corr in ranked:
        scene_training = os.path.join(training_dir, scene_name)
        opt = load_optimized_patterns(scene_training, db_floor=db_floor)
        if opt is None:
            print(f"  WARNING: no patterns for {scene_name}, using initial")
            opt = initial.copy()
        optimized[scene_name] = opt
        print(f"  {scene_name}: corr={corr:.3f}, patterns loaded")

    # Build figure: 2 rows (Initial / Optimized) × n_scenes columns
    # Shrink cells and increase gaps to prevent label overlap
    n_rows = 2
    plot_shrink = 0.92  # shrink polar plots to 92% of cell to leave label room
    margin_l_in = 0.50
    margin_r_in = 0.08
    margin_top_in = 0.18
    margin_bot_in = 0.08
    gap_col_in = 0.20  # increased from 0.04
    gap_row_in = 0.20  # increased from 0.04

    usable_w_in = (FIG_WIDTH_INCHES - margin_l_in - margin_r_in
                   - (n_scenes - 1) * gap_col_in)
    cell_w_in = usable_w_in / n_scenes
    cell_h_in = cell_w_in  # square cells
    fig_h = (margin_top_in + margin_bot_in
             + 2 * cell_h_in + gap_row_in)

    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, fig_h))
    add_rounded_bg(fig)

    # Convert to figure fractions
    margin_l = margin_l_in / FIG_WIDTH_INCHES
    margin_top = margin_top_in / fig_h
    gap_col = gap_col_in / FIG_WIDTH_INCHES
    gap_row = gap_row_in / fig_h
    cell_w = cell_w_in / FIG_WIDTH_INCHES
    cell_h = cell_h_in / fig_h

    row_labels = ["Initial", "Optimized"]
    row_data = [
        {sn: initial for sn, _ in ranked},  # row 0: initial (same for all)
        optimized,                            # row 1: optimized (per scene)
    ]

    for col_idx, (scene_name, _) in enumerate(ranked):
        for row_idx in range(2):
            # Center a shrunken polar plot within the cell
            cell_left = margin_l + col_idx * (cell_w + gap_col)
            cell_top = 1.0 - margin_top - row_idx * (cell_h + gap_row)
            cell_bottom = cell_top - cell_h
            inset = (1.0 - plot_shrink) / 2.0
            left = cell_left + inset * cell_w
            bottom = cell_bottom + inset * cell_h
            w = cell_w * plot_shrink
            h = cell_h * plot_shrink

            ax = fig.add_axes([left, bottom, w, h],
                              projection="polar")
            patterns = row_data[row_idx][scene_name]
            draw_pattern_polar(ax, patterns)

        # Column header
        header_x = margin_l + col_idx * (cell_w + gap_col) + cell_w / 2
        header_y = 1.0 - margin_top * 0.15
        fig.text(
            header_x, header_y,
            SCENE_SHORT_NAMES.get(scene_name, scene_name),
            ha="center", va="top",
            fontsize=6, fontweight="bold",
            transform=fig.transFigure,
        )

    # Row labels
    label_x = margin_l * 0.5
    for row_idx, label in enumerate(row_labels):
        row_center_y = (1.0 - margin_top - row_idx * (cell_h + gap_row)
                        - cell_h / 2)
        fig.text(
            label_x, row_center_y, label,
            ha="center", va="center",
            fontsize=6, fontweight="bold", rotation=90,
            transform=fig.transFigure,
        )

    out_pdf = os.path.join(output_dir, "beam_patterns_comparison_v1.pdf")
    out_png = os.path.join(output_dir, "beam_patterns_comparison_v1.png")
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
