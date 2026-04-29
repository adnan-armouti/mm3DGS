"""Supplement figure: per-scene comparison across all 8 training frames.

Produces ONE figure per scene (six total). Each figure has:
  - rows: GT / Ours (mm3DGS v5_v4) / RadarSplat / Radar Fields / DART
  - columns: the 8 training frames F-4..F-1, F+1..F+4 plus the held-out
    test frame (column ordering: F-4, F-3, F-2, F-1, F+1, F+2, F+3, F+4,
    F_test). Column header for the held-out frame is rendered in red so it
    stands out from the train frames.
  - per-cell ra_corr overlaid in white in the bottom-right.

Reads:
  - Ours: ``mm25DGS_v5_v4/output_frame_nvs/<scene>.../train_frames/frame_<F>/
          rendered_ra_cart.npy + metrics.json`` for train frames; the
          top-level ``rendered_test_ra_cart.npy`` + ``results.json`` for
          the test frame.
  - Baselines: ``baselines/<method>/results/<scene>/train_frames/frame_<F>/
          rendered_ra_cart.npy + metrics.json`` for train frames; the
          top-level ``rendered_ra_cart.npy`` + ``metrics.json`` for the
          test frame. DART scene-dirs are ``<scene>__cascaded``.

Reuses the same display style as ``generate_fig_training_ra.py`` (per-row
log/linear normalisation, fig_common helpers, hot/plasma colormap matching
the baseline finalize PNGs).

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


# Add the repo root to sys.path so ``figures.fig_common`` resolves both as
# ``python -m figures.generate_fig_training_ra_supplement`` and direct script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from figures.fig_common import SCENE_SHORT_NAMES  # noqa: E402


# Six benchmark scenes + their held-out test frames.
SCENES = [
    ("seq_0_frame_135", 135),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]


# Run-name templates the trainer might have used (post- or pre-cleanup).
OURS_RUN_TEMPLATES = [
    "{scene}_train8frames_1loops_test{frame}_loop0_pass2_N20000",
    ("{scene}_train8frames_1loops_test{frame}_loop0_pass2_N20000_"
     "dnsfyjt0.05i100u400p0.02_dsigpos_grad_amp_lpos1e-05L2100"),
]


# Baselines: how to find the per-scene results dir.
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


def load_ours_test(ours_run_dir: str):
    """Return (cart_RA or None, ra_corr or None)."""
    if ours_run_dir is None:
        return None, None
    ra = _load_npy(os.path.join(ours_run_dir, "rendered_test_ra_cart.npy"))
    res = _load_json(os.path.join(ours_run_dir, "results.json"))
    cc = float(res["final_test_cc"]) if res and "final_test_cc" in res else None
    return ra, cc


def load_ours_train_frame(ours_run_dir: str, frame: int):
    """Return (cart_RA or None, ra_corr or None) for one training frame."""
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
    """GT is identical across all baselines for a given (scene, test_frame).
    Pull from RadarSplat (the most-canonical GT renderer)."""
    for sub in ("radarsplat", "radarfields"):
        ra = _load_npy(os.path.join(_baseline_scene_dir(baselines_dir, sub, scene),
                                      "gt_ra_cart.npy"))
        if ra is not None:
            return ra
    return _load_npy(os.path.join(_baseline_scene_dir(baselines_dir, "dart", scene),
                                    "gt_ra_cart.npy"))


def load_gt_train_frame(baselines_dir: str, scene: str, frame: int,
                          ours_run_dir: Optional[str] = None):
    """GT for a train frame, identical across methods. Try sources in order:
    1) Our trainer's export (always written alongside rendered_ra_cart.npy)
    2) Each baseline's train_frames/frame_<F>/gt_ra_cart.npy
    Returns None if none of those are populated yet.
    """
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
# Display helpers (mirror generate_fig_training_ra.py)
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
    if ra is None:
        ax.text(0.5, 0.5, "missing",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=6, color="gray")
        for spine in ax.spines.values():
            spine.set_visible(False)
        return
    ax.imshow(ra_cart_to_linear(ra), cmap=cmap, origin="lower",
              vmin=0.0, vmax=1.0, aspect="equal")
    if corr is not None:
        ax.text(0.97, 0.04, f"{corr:.2f}",
                ha="right", va="bottom",
                transform=ax.transAxes, fontsize=4.5,
                fontweight="bold", color="white")


# ---------------------------------------------------------------------------
# Per-scene figure
# ---------------------------------------------------------------------------

def generate_one_scene(scene: str, test_frame: int,
                        ours_dir: str, baselines_dir: str,
                        output_path: str):
    """Build the 5-row × 9-col supplement figure for a single scene."""
    train_frames: List[int] = [test_frame + d for d in (-4, -3, -2, -1, 1, 2, 3, 4)]
    columns = train_frames + [test_frame]
    column_is_test = [False] * 8 + [True]

    ours_run_dir = find_ours_run_dir(ours_dir, scene, test_frame)

    # Pre-load everything before plotting.
    rows = []  # list of (label, [(ra, corr), ...])
    # GT row
    gt_cells = []
    for f, is_test in zip(columns, column_is_test):
        if is_test:
            gt_cells.append((load_gt_test(baselines_dir, scene), None))
        else:
            gt_cells.append((load_gt_train_frame(baselines_dir, scene, f,
                                                    ours_run_dir), None))
    rows.append(("GT", gt_cells))

    # Ours row
    ours_cells = []
    for f, is_test in zip(columns, column_is_test):
        if is_test:
            ra, cc = load_ours_test(ours_run_dir)
        else:
            ra, cc = load_ours_train_frame(ours_run_dir, f)
        ours_cells.append((ra, cc))
    rows.append(("Ours\n(v5\\_v4)", ours_cells))

    for label, b in (("RadarSplat", "radarsplat"),
                      ("RadarFields", "radarfields"),
                      ("DART", "dart")):
        cells = []
        for f, is_test in zip(columns, column_is_test):
            if is_test:
                ra, cc = load_baseline_test(baselines_dir, b, scene)
            else:
                ra, cc = load_baseline_train_frame(baselines_dir, b, scene, f)
            cells.append((ra, cc))
        rows.append((label, cells))

    n_cols = len(columns)
    n_rows = len(rows)
    fig_w = 1.4 * n_cols + 0.7
    fig_h = 1.4 * n_rows + 0.4
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    if n_rows == 1:
        axes = np.array([axes])

    # Column headers
    for j, (f, is_test) in enumerate(zip(columns, column_is_test)):
        label = f"$F$" if is_test else f"$F{f - test_frame:+d}$"
        color = "#cf2e2e" if is_test else "black"
        weight = "bold" if is_test else "normal"
        axes[0, j].set_title(label, fontsize=7, color=color,
                              fontweight=weight, pad=3)

    # Row labels (left) + tile contents
    for i, (row_label, cells) in enumerate(rows):
        axes[i, 0].set_ylabel(row_label, fontsize=7, rotation=0,
                                ha="right", va="center", labelpad=18)
        for j, (ra, corr) in enumerate(cells):
            _draw_tile(axes[i, j], ra, corr)

    # Title
    short = SCENE_SHORT_NAMES.get(scene, scene)
    fig.suptitle(short, fontsize=9, y=0.99, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, left=0.07)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight")
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

    for scene, test_frame in SCENES:
        out = os.path.join(args.output_dir, f"supplement_{scene}.pdf")
        print(f"  building {scene} F={test_frame}  →  {out}")
        generate_one_scene(scene, test_frame,
                            args.ours_dir, args.baselines_dir, out)

    print(f"\n[done] wrote 6 supplement figures under {args.output_dir}/")


if __name__ == "__main__":
    main()
