#!/usr/bin/env python3
"""Generate the training RA comparison figure for the paper.

Double-column figure with:
  - Columns: scenes (sorted by ours_corr / ``final_test_cc``, highest first)
  - Rows (top to bottom):
      1. GT          -- ground-truth |RA| at the test frame
      2. Ours        -- our v5_v4 trained model's render at the test frame
      3. RadarSplat  -- baseline render
      4. RadarFields -- baseline render
      5. DART        -- baseline render
  - Linear-scale |RA| heatmaps, per-cell normalised to its own [min, max]
  - All baselines display ``rendered_ra_cart.npy`` (399x399); DART has only
    cartesian, RadarSplat / RadarFields also expose polar but for an
    apples-to-apples comparison every row uses the cartesian projection.
  - Per-cell ``ra_corr`` overlaid as small text in the lower-right corner.

Usage:
    python -m figures.generate_fig_training_ra \
        --baselines_dir baselines \
        --ours_dir mm25DGS_v5_v4/output_frame_nvs \
        --output_dir output/postprocess_final_v5/figures
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures.fig_common import (
    BACKGROUND_COLOR,
    SCENE_SHORT_NAMES,
    add_rounded_bg,
    GridLayout,
)


CMAP = "hot"

# The 6 benchmark scenes the comparison is reported on.
DEFAULT_SCENES = [
    "seq_0_frame_135",
    "seq_1_frame_185",
    "seq_1_frame_438",
    "seq_2_frame_105",
    "seq_2_frame_160",
    "seq_2_frame_300",
]

# Run-name template used by the v5_v4 sweep that produced our final numbers.
# The trained run for ``<scene>`` lives in
# ``<ours_dir>/<scene>_train8frames_1loops_test<F>_loop0_pass2_N20000_dnsfyjt0.05i100u400p0.02_dsigpos_grad_amp_lpos1e-05L2100/``.
OURS_RUN_TEMPLATE = (
    "{scene}_train8frames_1loops_test{frame}_loop0_pass2_N20000_"
    "dnsfyjt0.05i100u400p0.02_dsigpos_grad_amp_lpos1e-05L2100"
)

# Candidate filenames to look for under the Ours run dir for a pre-rendered
# test-frame RA cartesian image. None of these currently exist on disk; if a
# future export pipeline writes one of these names, this script will pick it
# up automatically. See TODO below.
OURS_RA_CANDIDATES = [
    "rendered_test_ra_cart.npy",
    "rendered_test_ra.npy",
    "rendered_ra_cart.npy",
    "test_render_ra_cart.npy",
]


def _scene_frame_id(scene_name: str) -> str:
    """``seq_2_frame_160`` -> ``'160'``."""
    return scene_name.split("_")[-1]


# ---------------------------------------------------------------------------
# Loaders for each row
# ---------------------------------------------------------------------------

def _load_npy(path):
    return np.load(path) if (path and os.path.exists(path)) else None


def _load_json(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_gt(baselines_dir: str, scene: str):
    """GTs are identical across baselines; pull from RadarSplat by default."""
    for sub in ("radarsplat", "radarfields"):
        path = os.path.join(baselines_dir, sub, "results", scene, "gt_ra_cart.npy")
        arr = _load_npy(path)
        if arr is not None:
            return arr
    # DART stores GT under ``<scene>__cascaded`` rather than ``<scene>``
    return _load_npy(
        os.path.join(baselines_dir, "dart", "results", f"{scene}__cascaded",
                     "gt_ra_cart.npy")
    )


def load_baseline(baselines_dir: str, baseline: str, scene: str):
    """Return (rendered_ra_cart, ra_corr) for a baseline at a scene.

    DART stores its results under ``<scene>__cascaded/``; RadarSplat /
    RadarFields use ``<scene>/``.
    """
    if baseline == "dart":
        scene_dir = os.path.join(baselines_dir, "dart", "results",
                                 f"{scene}__cascaded")
    else:
        scene_dir = os.path.join(baselines_dir, baseline, "results", scene)
    ra = _load_npy(os.path.join(scene_dir, "rendered_ra_cart.npy"))
    metrics = _load_json(os.path.join(scene_dir, "metrics.json"))
    ra_corr = metrics.get("ra_corr") if metrics else None
    return ra, ra_corr


def load_ours(ours_dir: str, scene: str):
    """Return (rendered_ra_cart_or_None, final_test_cc_or_None).

    NOTE: as of writing, the v5_v4 sweep saves ``best_model.pt`` /
    ``results.json`` / ``history.npz`` but does NOT export a rendered
    cartesian RA at the test frame. This loader falls back to ``None`` for
    the image and the figure will draw a placeholder tile.

    TODO(ours-ra-export): add a quick re-render pass that writes
    ``rendered_test_ra_cart.npy`` (399x399 |RA| in cart) into each Ours run
    dir. This script will pick it up automatically via OURS_RA_CANDIDATES.
    """
    frame_id = _scene_frame_id(scene)
    run_dir = os.path.join(
        ours_dir,
        OURS_RUN_TEMPLATE.format(scene=scene, frame=frame_id),
    )
    ra = None
    for name in OURS_RA_CANDIDATES:
        cand = os.path.join(run_dir, name)
        if os.path.exists(cand):
            ra = np.load(cand)
            break
    results = _load_json(os.path.join(run_dir, "results.json"))
    corr = results.get("final_test_cc") if results else None
    return ra, corr, run_dir


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def ra_cart_to_linear(ra_cart: np.ndarray) -> np.ndarray:
    """Per-cell normalise an |RA| image to [0, 1]."""
    ra = np.asarray(ra_cart, dtype=np.float64)
    ra = np.abs(ra)  # ensure |RA| (DART polar has tiny negatives)
    mn, mx = ra.min(), ra.max()
    return (ra - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(ra)


def _draw_ra(ax, ra_cart, ra_corr=None):
    """Draw a |RA| heatmap onto ``ax`` and overlay the corr score."""
    ax.imshow(
        ra_cart_to_linear(ra_cart), cmap=CMAP,
        aspect="equal", origin="lower",
        vmin=0.0, vmax=1.0, interpolation="bilinear",
    )
    if ra_corr is not None and np.isfinite(ra_corr):
        ax.text(
            0.97, 0.03, f"{ra_corr:.2f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=4.5, color="white", fontweight="bold",
        )


def _draw_placeholder(ax, msg):
    """Draw a clean stub tile with a short message."""
    ax.set_facecolor("#222222")
    ax.text(
        0.5, 0.5, msg,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=4.5, color="#ffd166",
        wrap=True,
    )


def _frame_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color("#666666")


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

ROW_KEYS = ["gt", "ours", "radarsplat", "radarfields", "dart"]
ROW_LABELS = ["GT", "Ours\n(v5_v4)", "RadarSplat", "RadarFields", "DART"]


def gather_scene_data(scenes, baselines_dir, ours_dir):
    """Collect per-scene tiles + corr scores for every row."""
    out = {}
    for scene in scenes:
        gt = load_gt(baselines_dir, scene)
        ours_ra, ours_corr, ours_run_dir = load_ours(ours_dir, scene)
        rsplat_ra, rsplat_corr = load_baseline(baselines_dir, "radarsplat", scene)
        rfields_ra, rfields_corr = load_baseline(baselines_dir, "radarfields", scene)
        dart_ra, dart_corr = load_baseline(baselines_dir, "dart", scene)
        out[scene] = {
            "gt":          {"ra": gt,         "corr": None},
            "ours":        {"ra": ours_ra,    "corr": ours_corr,
                            "run_dir": ours_run_dir},
            "radarsplat":  {"ra": rsplat_ra,  "corr": rsplat_corr},
            "radarfields": {"ra": rfields_ra, "corr": rfields_corr},
            "dart":        {"ra": dart_ra,    "corr": dart_corr},
        }
    return out


def sort_scenes(scene_data, scenes):
    """Sort scenes by ours corr (descending). Fall back to RadarSplat corr
    when Ours corr is missing."""
    def key(scene):
        ours = scene_data[scene]["ours"]["corr"]
        if ours is None:
            ours = scene_data[scene]["radarsplat"]["corr"] or -1.0
        return ours

    return sorted(scenes, key=key, reverse=True)


def generate_figure(scenes, baselines_dir, ours_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    scene_data = gather_scene_data(scenes, baselines_dir, ours_dir)
    ranked = sort_scenes(scene_data, scenes)
    n_scenes = len(ranked)
    n_rows = len(ROW_KEYS)

    # Stub-warning if Ours not exported.
    missing_ours = [s for s in ranked if scene_data[s]["ours"]["ra"] is None]
    if missing_ours:
        print("NOTE: Ours rendered_test_ra_cart.npy not found for these scenes:")
        for s in missing_ours:
            print(f"  {s} -> {scene_data[s]['ours']['run_dir']}")
        print("      Drawing placeholder tiles. See TODO(ours-ra-export) in script.")

    print("\nColumn order (sorted by ours final_test_cc, descending):")
    for s in ranked:
        oc = scene_data[s]["ours"]["corr"]
        rc = scene_data[s]["radarsplat"]["corr"]
        oc_s = f"{oc:.3f}" if oc is not None else "n/a"
        rc_s = f"{rc:.3f}" if rc is not None else "n/a"
        print(f"  {s}: ours_cc={oc_s}  radarsplat_corr={rc_s}")

    layout = GridLayout.from_image_aspect(
        n_rows, n_scenes, img_aspect=1.0,
        margin_in=0.08, col_gap_in=0.03, row_gap_in=0.03,
        header_in=0.16, label_w_in=0.50,
    )

    fig = plt.figure(figsize=(layout.fig_w, layout.fig_h))
    add_rounded_bg(fig)

    for col_idx, scene in enumerate(ranked):
        for row_idx, row_key in enumerate(ROW_KEYS):
            left, bottom, w, h = layout.cell_pos(row_idx, col_idx)
            ax = fig.add_axes([left, bottom, w, h])

            cell = scene_data[scene][row_key]
            ra = cell["ra"]
            corr = cell["corr"]

            if ra is not None:
                _draw_ra(ax, ra, ra_corr=corr)
            else:
                if row_key == "ours":
                    _draw_placeholder(
                        ax,
                        "Ours RA not yet exported\nsee TODO in script",
                    )
                else:
                    _draw_placeholder(ax, "missing")
            _frame_axes(ax)

        # Column header (scene short name)
        left, _, w, _ = layout.cell_pos(0, col_idx)
        fig.text(
            left + w / 2, layout.header_y(),
            SCENE_SHORT_NAMES.get(scene, scene),
            ha="center", va="top",
            fontsize=5.5, fontweight="bold",
            transform=fig.transFigure,
        )

    # Row labels
    for row_idx, label in enumerate(ROW_LABELS):
        fig.text(
            layout.row_label_x(), layout.row_label_y(row_idx), label,
            ha="center", va="center",
            fontsize=5.5, fontweight="bold", rotation=90,
            transform=fig.transFigure,
        )

    out_pdf = os.path.join(output_dir, "training_ra_comparison.pdf")
    out_png = os.path.join(output_dir, "training_ra_comparison.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)
    print(f"\nSaved: {out_pdf}\nSaved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baselines_dir", default="baselines",
        help="Root dir with radarsplat/, radarfields/, dart/ subtrees.",
    )
    parser.add_argument(
        "--ours_dir", default="mm25DGS_v5_v4/output_frame_nvs",
        help="Root dir holding the v5_v4 per-scene run directories.",
    )
    parser.add_argument(
        "--output_dir", default="output/postprocess_final_v5/figures",
    )
    parser.add_argument(
        "--scenes", nargs="+", default=DEFAULT_SCENES,
        help="Scenes to include as columns (sort order is computed).",
    )
    args = parser.parse_args()
    generate_figure(args.scenes, args.baselines_dir, args.ours_dir,
                    args.output_dir)


if __name__ == "__main__":
    main()
