"""Supplement figure: per-scene comparison across all 9 frames.

Produces ONE figure per scene (six total). Each figure has:
  - rows:   GT / Ours / RadarSplat / Radar Fields / DART
  - columns: F-4, F-3, F-2, F-1, F (TEST), F+1, F+2, F+3, F+4
  - the test frame is in the middle column with a light-red column tile
    sitting on top of the rounded grey background tile shared with Fig 2.
  - per-cell ra_corr overlaid in white in the bottom-right.

Layout, fonts, colors, and tile geometry all match
``figures/generate_fig_training_ra.py`` (Figure 2). Reads the same
on-disk artifacts as the main figure script + per-frame ``train_frames/
frame_<F>/`` subdirectories produced by the trainer / baseline runners.

Usage:
    python -m figures.generate_fig_training_ra_supplement \
        --baselines_dir baselines \
        --ours_dir mm25DGS_v5_v4/output_frame_nvs \
        --output_dir output/postprocess_final_v5/figures/supplement
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from figures.fig_common import (   # noqa: E402
    BACKGROUND_COLOR, FIG_WIDTH_INCHES, SCENE_SHORT_NAMES, TEST_COL_COLOR,
    FIGURE_BASE_PT, FIGURE_HEADER_PT, FIGURE_SMALL_PT,
    add_rounded_bg, add_column_highlight, apply_paper_font, GridLayout,
)


SCENES = [
    ("seq_0_frame_135", 135),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]

OURS_RUN_TEMPLATES = [
    "{scene}_train8frames_1loops_test{frame}_loop0_pass2_N20000",
    ("{scene}_train8frames_1loops_test{frame}_loop0_pass2_N20000_"
     "dnsfyjt0.05i100u400p0.02_dsigpos_grad_amp_lpos1e-05L2100"),
]


def _baseline_scene_dir(baselines_dir: str, baseline: str, scene: str) -> str:
    if baseline == "dart":
        return os.path.join(baselines_dir, "dart", "results", f"{scene}__cascaded")
    return os.path.join(baselines_dir, baseline, "results", scene)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    return np.load(path)


def _load_json(path: Optional[str]) -> Optional[dict]:
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_ours_run_dir(ours_dir: str, scene: str, frame: int) -> Optional[str]:
    candidates = []
    for tmpl in OURS_RUN_TEMPLATES:
        d = os.path.join(ours_dir, tmpl.format(scene=scene, frame=frame))
        results_p = os.path.join(d, "results.json")
        if os.path.exists(results_p):
            candidates.append((os.path.getmtime(results_p), d))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None


def load_ours_test(ours_run_dir: Optional[str]):
    if ours_run_dir is None:
        return None, None
    ra = _load_npy(os.path.join(ours_run_dir, "rendered_test_ra_cart.npy"))
    res = _load_json(os.path.join(ours_run_dir, "results.json"))
    cc = float(res["final_test_cc"]) if res and "final_test_cc" in res else None
    return ra, cc


def load_ours_train_frame(ours_run_dir: Optional[str], frame: int):
    if ours_run_dir is None:
        return None, None
    fdir = os.path.join(ours_run_dir, "train_frames", f"frame_{frame}")
    ra = _load_npy(os.path.join(fdir, "rendered_ra_cart.npy"))
    metrics = _load_json(os.path.join(fdir, "metrics.json"))
    cc = float(metrics["ra_corr"]) if metrics and "ra_corr" in metrics else None
    return ra, cc


def load_baseline_test(baselines_dir: str, baseline: str, scene: str):
    sd = _baseline_scene_dir(baselines_dir, baseline, scene)
    ra = _load_npy(os.path.join(sd, "rendered_ra_cart.npy"))
    metrics = _load_json(os.path.join(sd, "metrics.json"))
    cc = float(metrics["ra_corr"]) if metrics and "ra_corr" in metrics else None
    return ra, cc


def load_baseline_train_frame(baselines_dir: str, baseline: str, scene: str,
                                frame: int):
    sd = _baseline_scene_dir(baselines_dir, baseline, scene)
    fdir = os.path.join(sd, "train_frames", f"frame_{frame}")
    ra = _load_npy(os.path.join(fdir, "rendered_ra_cart.npy"))
    metrics = _load_json(os.path.join(fdir, "metrics.json"))
    cc = float(metrics["ra_corr"]) if metrics and "ra_corr" in metrics else None
    return ra, cc


def load_gt_test(baselines_dir: str, scene: str):
    for sub in ("radarsplat", "radarfields"):
        ra = _load_npy(os.path.join(_baseline_scene_dir(baselines_dir, sub, scene),
                                      "gt_ra_cart.npy"))
        if ra is not None:
            return ra
    return _load_npy(os.path.join(_baseline_scene_dir(baselines_dir, "dart", scene),
                                    "gt_ra_cart.npy"))


def load_gt_train_frame(baselines_dir: str, scene: str, frame: int,
                          ours_run_dir: Optional[str] = None):
    if ours_run_dir is not None:
        ra = _load_npy(os.path.join(ours_run_dir, "train_frames",
                                      f"frame_{frame}", "gt_ra_cart.npy"))
        if ra is not None:
            return ra
    for sub in ("radarsplat", "radarfields", "dart"):
        ra = _load_npy(os.path.join(
            _baseline_scene_dir(baselines_dir, sub, scene),
            "train_frames", f"frame_{frame}", "gt_ra_cart.npy",
        ))
        if ra is not None:
            return ra
    return None


# ---------------------------------------------------------------------------
# Display helpers (mirror generate_fig_training_ra.py exactly)
# ---------------------------------------------------------------------------

def ra_cart_to_linear(ra_cart: np.ndarray) -> np.ndarray:
    arr = np.asarray(ra_cart, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _draw_tile(ax, ra: Optional[np.ndarray], corr: Optional[float], cmap="hot"):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if ra is None:
        ax.text(0.5, 0.5, "missing",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=FIGURE_SMALL_PT, color="#999999")
        return
    ax.imshow(ra_cart_to_linear(ra), cmap=cmap, origin="lower",
              vmin=0.0, vmax=1.0, aspect="equal")
    if corr is not None:
        ax.text(0.97, 0.04, f"{corr:.2f}",
                ha="right", va="bottom",
                transform=ax.transAxes, fontsize=FIGURE_SMALL_PT,
                fontweight="bold", color="white")


# ---------------------------------------------------------------------------
# Per-scene figure builder
# ---------------------------------------------------------------------------

ROW_LABELS = ["GT", "Ours", "RadarSplat", "Radar Fields", "DART"]


def generate_one_scene(scene: str, test_frame: int,
                        ours_dir: str, baselines_dir: str,
                        output_path: str):
    """Build the 5-row × 9-col supplement figure for a single scene."""
    # Columns chronologically with F in the middle:
    # F-4, F-3, F-2, F-1, F (TEST), F+1, F+2, F+3, F+4
    col_offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    columns = [test_frame + d for d in col_offsets]
    column_is_test = [d == 0 for d in col_offsets]
    test_col_idx = col_offsets.index(0)

    ours_run_dir = find_ours_run_dir(ours_dir, scene, test_frame)

    # Load all cells.
    rows: List[List[tuple]] = []   # rows[i] = list of (ra, corr) cells

    # GT row
    gt_cells = []
    for f, is_test in zip(columns, column_is_test):
        if is_test:
            gt_cells.append((load_gt_test(baselines_dir, scene), None))
        else:
            gt_cells.append((load_gt_train_frame(baselines_dir, scene, f,
                                                    ours_run_dir), None))
    rows.append(gt_cells)

    # Ours row
    ours_cells = []
    for f, is_test in zip(columns, column_is_test):
        if is_test:
            ra, cc = load_ours_test(ours_run_dir)
        else:
            ra, cc = load_ours_train_frame(ours_run_dir, f)
        ours_cells.append((ra, cc))
    rows.append(ours_cells)

    for b in ("radarsplat", "radarfields", "dart"):
        cells = []
        for f, is_test in zip(columns, column_is_test):
            if is_test:
                ra, cc = load_baseline_test(baselines_dir, b, scene)
            else:
                ra, cc = load_baseline_train_frame(baselines_dir, b, scene, f)
            cells.append((ra, cc))
        rows.append(cells)

    n_rows = len(rows)
    n_cols = len(columns)

    # Use a wider layout for the supplement (9 cols vs 6 in main paper).
    # The col_gap is sized so that the red test-pose tile (which extends
    # ~55% into the gap on each side via add_column_highlight) leaves a
    # narrow but clearly visible grey strip between the red column and the
    # F-1 / F+1 image cells.
    layout = GridLayout.from_fig_width(
        n_rows=n_rows, n_cols=n_cols,
        fig_width_in=FIG_WIDTH_INCHES * 1.5,
        img_aspect=1.0,
        margin_in=0.10, col_gap_in=0.10, row_gap_in=0.05,
        header_in=0.30, label_w_in=0.30,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h),
                     facecolor=BACKGROUND_COLOR)
    add_rounded_bg(fig)
    add_column_highlight(fig, layout, test_col_idx,
                          color=TEST_COL_COLOR,
                          inset_in=layout.col_gap_in * 0.55)

    # Column headers
    for j, off in enumerate(col_offsets):
        x_left, _, w, _ = layout.cell_pos(0, j)
        cx = x_left + w / 2
        if off == 0:
            label = "$F$"
            color = "#9b1c1c"     # darker red for test pose label
        else:
            label = f"$F{off:+d}$"
            color = "black"
        fig.text(cx, layout.header_y(), label,
                  ha="center", va="top", fontsize=FIGURE_HEADER_PT,
                  color=color, fontweight="bold" if off == 0 else "normal")

    # Row labels (left column)
    for i, label in enumerate(ROW_LABELS):
        fig.text(layout.row_label_x(), layout.row_label_y(i), label,
                  ha="center", va="center", fontsize=FIGURE_HEADER_PT,
                  rotation=90, fontweight="bold")

    # Tiles
    for i, cells in enumerate(rows):
        for j, (ra, corr) in enumerate(cells):
            ax = fig.add_axes(layout.cell_pos(i, j))
            _draw_tile(ax, ra, corr)
    # NB: the scene name is supplied by the LaTeX caption (Figs. 3--8); we
    # intentionally do not draw an in-figure title here so the F column
    # header has no overlap.

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=BACKGROUND_COLOR,
                 bbox_inches=None)
    fig.savefig(output_path.replace(".pdf", ".png"), dpi=300,
                 facecolor=BACKGROUND_COLOR, bbox_inches=None)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines_dir", default="baselines")
    ap.add_argument("--ours_dir", default="mm25DGS_v5_v4/output_frame_nvs")
    ap.add_argument("--output_dir",
                    default="output/postprocess_final_v5/figures/supplement")
    args = ap.parse_args()

    apply_paper_font()

    for scene, test_frame in SCENES:
        out = os.path.join(args.output_dir, f"supplement_{scene}.pdf")
        print(f"  building {scene} F={test_frame}  →  {out}")
        generate_one_scene(scene, test_frame,
                            args.ours_dir, args.baselines_dir, out)

    print(f"\n[done] wrote 6 supplement figures under {args.output_dir}/")


if __name__ == "__main__":
    main()
