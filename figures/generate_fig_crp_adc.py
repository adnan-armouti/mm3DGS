"""Qualitative figure for CRP / ADC product-agnostic evaluation.

Themed to match Figs 1 and 2:
  - Outer rounded card (BACKGROUND_COLOR) via add_rounded_bg.
  - Two inner section tiles (TILE_COLOR) for |CRP| and |ADC|.
  - Row labels = scene short names (S0-F135 ...) on the left.
  - Column subheaders ("GT" / "Ours") inside each section.
  - Per-cell metric overlay in the bottom-right of the *pred* cells only
    (heatmaps fig); per-cell VA index + key metric in the top-left of the
    plot (iq traces fig).
  - pcolormesh rasterized=True for heatmaps; line plots for iq traces.

Reads per-scene phase-corrected CRPs and ADCs saved by
``mmir.evaluation.eval_crp_adc``:
  output/crp_adc_eval/<scene>/{crp,adc}_{pred_corr,gt}.npy
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from figures.fig_common import (   # noqa: E402
    BACKGROUND_COLOR, FIG_WIDTH_INCHES, SCENE_SHORT_NAMES,
    FIGURE_BASE_PT, FIGURE_HEADER_PT, FIGURE_SMALL_PT,
    add_rounded_bg, apply_paper_font, _rounded_rect_path,
)


apply_paper_font()


# ── Visual tokens (match Figs 1 / 2) ──────────────────────────────────────
TILE_COLOR     = "#ebebeb"
SECTION_RADIUS = 0.04
SECTION_HEADER_PT = FIGURE_HEADER_PT  # 10pt — |CRP|, |ADC| section titles
SUBHEADER_PT      = FIGURE_HEADER_PT  # 10pt — GT, Ours column subheaders
ROW_LABEL_PT      = FIGURE_HEADER_PT  # 10pt — scene labels, range bin/ADC sample axis labels
METRIC_PT         = FIGURE_SMALL_PT   # 7pt  — per-cell metric overlays (must fit in CRP cell)

SAVE_DPI = 300

PAPER_SCENES = [
    "seq_0_frame_135", "seq_1_frame_185", "seq_1_frame_438",
    "seq_2_frame_105", "seq_2_frame_160", "seq_2_frame_300",
]


# ── Normalisation (peak-dB / min-max) ─────────────────────────────────────

def _norm_db(x: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """|x|/peak in dB, clipped at floor_db (matches save_ra_cartesian_png)."""
    m = np.abs(x).astype(np.float64)
    p = m.max()
    if p < 1e-30:
        return np.zeros_like(m) + floor_db
    db = 20.0 * np.log10(np.clip(m / p, 10 ** (floor_db / 20.0), 1.0))
    return db


def _norm_minmax(x: np.ndarray) -> np.ndarray:
    """|x| min-max normalised to [0, 1] (matches the metric domain)."""
    m = np.abs(x).astype(np.float64)
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-30:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


# ── Themed drawing helpers ────────────────────────────────────────────────

def _add_section_tile(fig, x_in, y_in, w_in, h_in,
                       color=TILE_COLOR, radius_in=SECTION_RADIUS,
                       zorder=-0.5):
    """Add an inner rounded section tile in inch coords, behind content."""
    fig_w, fig_h = fig.get_size_inches()
    path = _rounded_rect_path(
        x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h,
        fig_w, fig_h, radius_in)
    fig.patches.append(mpatches.PathPatch(
        path, transform=fig.transFigure,
        facecolor=color, edgecolor="none", linewidth=0, zorder=zorder,
    ))


def _ax_at(fig, x_in, y_in, w_in, h_in, zorder=2):
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes([x_in / fig_w, y_in / fig_h,
                        w_in / fig_w, h_in / fig_h])
    ax.patch.set_visible(False)
    ax.set_zorder(zorder)
    return ax


def _figtext(fig, x_in, y_in, text, **kw):
    fig_w, fig_h = fig.get_size_inches()
    return fig.text(x_in / fig_w, y_in / fig_h, text,
                     transform=fig.transFigure, **kw)


def _hide_spines_ticks(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.patch.set_visible(False)


def _render_heatmap(ax, complex_arr, scale, cmap="hot", floor_db=-40.0):
    """pcolormesh + rasterized=True so the heatmap is one PDF raster."""
    if scale == "dB":
        disp = _norm_db(complex_arr, floor_db)
        vmin, vmax = floor_db, 0.0
    elif scale == "linear":
        disp = _norm_minmax(complex_arr)
        vmin, vmax = 0.0, 1.0
    else:
        raise ValueError(f"unknown scale {scale}")
    ax.pcolormesh(
        disp, cmap=cmap, vmin=vmin, vmax=vmax,
        rasterized=True, shading="flat",
        linewidth=0, edgecolors="none", antialiased=False,
    )
    _hide_spines_ticks(ax)


def _metric_overlay(ax, lines, color="white"):
    ax.text(0.985, 0.04, "\n".join(lines),
             transform=ax.transAxes,
             ha="right", va="bottom",
             fontsize=METRIC_PT, color=color, fontweight="bold",
             linespacing=1.05)


# ──────────────────────────────────────────────────────────────────────────
# Figure: heatmaps (Figs 10 / 11)
# ──────────────────────────────────────────────────────────────────────────

def fig_heatmaps(eval_dir: str, results_json: str, output_pdf: str,
                  scenes=PAPER_SCENES, start_bin: int = 15,
                  scale: str = "dB"):
    with open(results_json) as f:
        results = json.load(f)
    per = results["per_scene"]
    n_scenes = len(scenes)

    fig_w = FIG_WIDTH_INCHES * 1.5
    margin_h    = 0.15
    label_w     = 0.20
    right_pad   = label_w
    section_gap = 0.20
    section_pad_top = 0.06
    section_pad_bot = 0.04
    section_lr_pad  = 0.06
    col_gap     = 0.05
    row_gap     = 0.04

    top_pad             = 0.18
    section_header_h    = 0.13
    section_header_gap  = 0.05
    subheader_h         = 0.14
    subheader_grid_gap  = 0.03
    grid_axis_label_gap = 0.05
    axis_label_h        = 0.13
    bot_pad             = 0.18

    content_w = fig_w - 2 * margin_h - label_w - right_pad
    section_w = (content_w - section_gap) / 2
    cell_w    = (section_w - 2 * section_lr_pad - col_gap) / 2
    cell_h    = cell_w * 0.50

    grid_h    = n_scenes * cell_h + (n_scenes - 1) * row_gap

    section_tile_h = (section_pad_top + subheader_h + subheader_grid_gap
                      + grid_h + section_pad_bot)
    fig_h = (top_pad + section_header_h + section_header_gap
             + section_tile_h
             + grid_axis_label_gap + axis_label_h + bot_pad)

    section_header_text_top = fig_h - top_pad
    section_header_y        = section_header_text_top - section_header_h
    section_tile_top        = section_header_y - section_header_gap
    section_tile_y          = section_tile_top - section_tile_h
    grid_top                = section_tile_top - section_pad_top - subheader_h - subheader_grid_gap
    subheader_y             = grid_top + subheader_grid_gap

    crp_section_x = margin_h + label_w
    adc_section_x = crp_section_x + section_w + section_gap
    crp_gt_x   = crp_section_x + section_lr_pad
    crp_pr_x   = crp_gt_x + cell_w + col_gap
    adc_gt_x   = adc_section_x + section_lr_pad
    adc_pr_x   = adc_gt_x + cell_w + col_gap

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=SAVE_DPI)
    add_rounded_bg(fig)
    _add_section_tile(fig, crp_section_x, section_tile_y,
                       section_w, section_tile_h)
    _add_section_tile(fig, adc_section_x, section_tile_y,
                       section_w, section_tile_h)

    _figtext(fig, crp_section_x + section_w / 2, section_header_y,
              r"$|\mathrm{CRP}|$",
              ha="center", va="bottom",
              fontsize=SECTION_HEADER_PT, fontweight="bold", color="#222")
    _figtext(fig, adc_section_x + section_w / 2, section_header_y,
              r"$|\mathrm{ADC}|$",
              ha="center", va="bottom",
              fontsize=SECTION_HEADER_PT, fontweight="bold", color="#222")

    for cx in (crp_gt_x, adc_gt_x):
        _figtext(fig, cx + cell_w / 2, subheader_y,
                  "GT", ha="center", va="bottom",
                  fontsize=SUBHEADER_PT, fontweight="bold", color="#444")
    for cx in (crp_pr_x, adc_pr_x):
        _figtext(fig, cx + cell_w / 2, subheader_y,
                  "Ours", ha="center", va="bottom",
                  fontsize=SUBHEADER_PT, fontweight="bold", color="#0e6b2c")

    for ri, sc in enumerate(scenes):
        row_top = grid_top - ri * (cell_h + row_gap)
        row_bot = row_top - cell_h
        cy      = (row_top + row_bot) / 2

        _figtext(fig, margin_h + label_w * 0.5, cy,
                  SCENE_SHORT_NAMES.get(sc, sc),
                  ha="center", va="center",
                  fontsize=ROW_LABEL_PT, fontweight="bold",
                  rotation=90, color="#222")

        crp_gt = np.load(os.path.join(eval_dir, sc, "crp_gt.npy"))[:, start_bin:]
        crp_pr = np.load(os.path.join(eval_dir, sc, "crp_pred_corr.npy"))[:, start_bin:]
        adc_gt = np.load(os.path.join(eval_dir, sc, "adc_gt.npy"))
        adc_pr = np.load(os.path.join(eval_dir, sc, "adc_pred_corr.npy"))
        c_metrics = per[sc]["test"]["crp"]
        a_metrics = per[sc]["test"]["adc"]

        ax = _ax_at(fig, crp_gt_x, row_bot, cell_w, cell_h)
        _render_heatmap(ax, crp_gt, scale)
        ax = _ax_at(fig, crp_pr_x, row_bot, cell_w, cell_h)
        _render_heatmap(ax, crp_pr, scale)
        _metric_overlay(ax, [
            fr"$\rho$={c_metrics['mag_corr']:.2f}  "
            fr"PSNR={c_metrics['mag_psnr']:.1f}  "
            fr"SSIM={c_metrics['mag_ssim']:.2f}  "
            fr"$|\rho|^{{\mathrm{{VR}}}}$={c_metrics['complex_corr_VR']:.2f}",
        ])

        ax = _ax_at(fig, adc_gt_x, row_bot, cell_w, cell_h)
        _render_heatmap(ax, adc_gt, scale)
        ax = _ax_at(fig, adc_pr_x, row_bot, cell_w, cell_h)
        _render_heatmap(ax, adc_pr, scale)
        _metric_overlay(ax, [
            fr"$\rho$={a_metrics['mag_corr']:.2f}  "
            fr"$|\rho|^{{\mathrm{{VR}}}}$={a_metrics['complex_corr_VR']:.2f}  "
            fr"$\sigma_\phi$={a_metrics['phase_rmse_deg_VR']:.0f}°",
        ])

    bot_label_y = section_tile_y - grid_axis_label_gap
    _figtext(fig, (crp_gt_x + crp_pr_x + cell_w) / 2,
              bot_label_y, fr"range bin (offset by {start_bin})",
              ha="center", va="top",
              fontsize=ROW_LABEL_PT, color="#444")
    _figtext(fig, (adc_gt_x + adc_pr_x + cell_w) / 2,
              bot_label_y, "ADC sample",
              ha="center", va="top",
              fontsize=ROW_LABEL_PT, color="#444")

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, dpi=SAVE_DPI, facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output_pdf}")


# ──────────────────────────────────────────────────────────────────────────
# Figure: I/Q traces (Fig 12)
# ──────────────────────────────────────────────────────────────────────────

GT_RE_COLOR  = "#1f77b4"
GT_IM_COLOR  = "#d62728"
PRED_RE_COLOR = GT_RE_COLOR
PRED_IM_COLOR = GT_IM_COLOR


def _pick_representative_va(crp_gt: np.ndarray, start_bin: int = 0) -> int:
    energy = np.max(np.abs(crp_gt[:, start_bin:]), axis=1)
    return int(np.argmax(energy))


def _render_iq(ax, gt, pr, x, *, vline=None, xticks=None, show_xticklabels=True):
    """4-line I/Q plot on a transparent axes; clean grid, no spines.

    Y-axis ticks at -1/0/+1, x-axis with explicit (irregular) ticks supplied
    by caller. Avoids the "50 100 150 200" 4-tick auto-pattern that triggers
    Preview's phone-number Data Detector.
    """
    norm = max(np.max(np.abs(gt)), 1e-30)
    if vline is not None:
        ax.axvline(vline, color="#888888", lw=0.5, ls="--",
                    alpha=0.6, zorder=2)
    ax.plot(x, gt.real / norm, color=GT_RE_COLOR, lw=0.9, zorder=3)
    ax.plot(x, gt.imag / norm, color=GT_IM_COLOR, lw=0.9, zorder=3)
    ax.plot(x, pr.real / norm, color=PRED_RE_COLOR, lw=0.9,
             ls=(0, (3, 2)), alpha=0.9, zorder=4)
    ax.plot(x, pr.imag / norm, color=PRED_IM_COLOR, lw=0.9,
             ls=(0, (3, 2)), alpha=0.9, zorder=4)
    # Y-lim slightly extended so per-cell label sits above the trace lines.
    ax.set_ylim(-1.18, 1.42)
    ax.set_xlim(x[0], x[-1])
    ax.set_yticks([-1, 0, 1])
    if xticks is not None:
        ax.set_xticks(xticks)
        if not show_xticklabels:
            ax.set_xticklabels([])
    else:
        ax.set_xticks([])
    ax.tick_params(axis="both", labelsize=METRIC_PT,
                    color="#888", length=2.5, width=0.4, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color="#999999", lw=0.35, alpha=0.30, zorder=1)
    ax.set_facecolor("none")
    ax.patch.set_visible(False)


def _draw_legend_strip(fig, cx_in, cy_in, fig_w, fig_h):
    items = [
        ("GT Re", GT_RE_COLOR, "-"),
        ("GT Im", GT_IM_COLOR, "-"),
        ("pred Re", PRED_RE_COLOR, (0, (3, 2))),
        ("pred Im", PRED_IM_COLOR, (0, (3, 2))),
    ]
    legend_ax = fig.add_axes([0, 0, 1, 1], zorder=10)
    legend_ax.set_xlim(0, fig_w); legend_ax.set_ylim(0, fig_h)
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    for sp in legend_ax.spines.values():
        sp.set_visible(False)
    legend_ax.patch.set_visible(False)

    item_gap = 0.55
    seg_len  = 0.22
    text_pad = 0.05
    item_w   = seg_len + text_pad + 0.45
    total_w  = len(items) * item_w + (len(items) - 1) * item_gap
    x = cx_in - total_w / 2
    for label, color, ls in items:
        legend_ax.plot([x, x + seg_len], [cy_in, cy_in],
                        color=color, lw=1.0, linestyle=ls, alpha=0.95)
        legend_ax.text(x + seg_len + text_pad, cy_in, label,
                        ha="left", va="center",
                        fontsize=FIGURE_HEADER_PT, color="#222")
        x += item_w + item_gap


def fig_iq_traces(eval_dir: str, results_json: str, output_pdf: str,
                   scenes=PAPER_SCENES, start_bin: int = 15):
    """6 rows × 2 sections (CRP / ADC), one trace plot each, themed card.

    Layout: section headers at top → row grid → x-axis labels under last row
    → legend at the very bottom.  Per-cell label sits above the data range
    (extended y-lim 1.18..1.42) inside a soft rounded bbox so it never
    collides with the trace lines.  X-axis has 3 explicit ticks at irregular
    positions to avoid the "50 100 150 200" 4-tick auto-pattern that
    triggers macOS Preview's phone-number Data Detector.
    """
    with open(results_json) as f:
        results = json.load(f)
    per = results["per_scene"]
    n_scenes = len(scenes)

    fig_w = FIG_WIDTH_INCHES * 1.5
    margin     = 0.10
    label_w    = 0.35
    section_gap = 0.20
    section_left_pad  = 0.32
    section_right_pad = 0.08
    row_gap    = 0.06
    section_header_h = 0.22
    header_h         = section_header_h + 0.18  # extra room so CRP / ADC titles
                                                  # have breathing space above the
                                                  # rounded section tile
    xaxis_label_h    = 0.32   # increased to clear the xtick numbers
    legend_h         = 0.22
    bot_margin       = xaxis_label_h + legend_h + 0.08  # tightened (was +0.30)
                                                          # to remove wasted space
                                                          # below the legend strip

    content_w = fig_w - 2 * margin - label_w
    section_w = (content_w - section_gap) / 2
    cell_w    = section_w - section_left_pad - section_right_pad
    cell_h    = 0.65

    grid_h    = n_scenes * cell_h + (n_scenes - 1) * row_gap
    fig_h     = margin + header_h + grid_h + bot_margin

    section_tile_h = grid_h + 2 * 0.06
    section_tile_y = margin + bot_margin - 0.06
    grid_top       = section_tile_y + 0.06 + grid_h
    section_header_y = grid_top + 0.20  # was 0.10 — bigger gap so the
                                         # CRP / ADC titles aren't flush
                                         # with the tile top

    crp_section_x = margin + label_w
    adc_section_x = crp_section_x + section_w + section_gap
    crp_cell_x = crp_section_x + section_left_pad
    adc_cell_x = adc_section_x + section_left_pad

    grid_bottom = grid_top - grid_h
    # Drop the "range bin" / "ADC sample" labels far enough below the
    # section tile that they don't collide with the bottom-row xtick
    # numbers (which sit just above section_tile_y).
    xaxis_label_y = section_tile_y - 0.20
    legend_y      = section_tile_y - 0.46

    # Regular ticks every 32: 8-9 ticks per axis with mixed digit counts.
    # Doesn't match phone-number formats (7/10/11 digits with specific group
    # boundaries), so Preview's Data Detector should leave it alone.
    crp_xticks = [16, 48, 80, 112, 144, 176, 208, 240]
    adc_xticks = [0, 32, 64, 96, 128, 160, 192, 224, 256]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=SAVE_DPI)
    add_rounded_bg(fig)
    _add_section_tile(fig, crp_section_x, section_tile_y,
                       section_w, section_tile_h)
    _add_section_tile(fig, adc_section_x, section_tile_y,
                       section_w, section_tile_h)

    _figtext(fig, crp_section_x + section_w / 2, section_header_y,
              r"$\mathrm{CRP}$",
              ha="center", va="bottom",
              fontsize=SECTION_HEADER_PT, fontweight="bold", color="#222")
    _figtext(fig, adc_section_x + section_w / 2, section_header_y,
              r"$\mathrm{ADC}$",
              ha="center", va="bottom",
              fontsize=SECTION_HEADER_PT, fontweight="bold", color="#222")

    grid_cy = (grid_top + grid_bottom) / 2
    for sx in (crp_section_x, adc_section_x):
        _figtext(fig, sx + 0.08, grid_cy,
                  "amplitude (norm.)",
                  ha="center", va="center",
                  rotation=90, fontsize=ROW_LABEL_PT, color="#444")

    label_bbox = dict(boxstyle="round,pad=0.18",
                       facecolor=TILE_COLOR, edgecolor="#cccccc",
                       linewidth=0.4, alpha=0.95)

    for ri, sc in enumerate(scenes):
        row_top = grid_top - ri * (cell_h + row_gap)
        row_bot = row_top - cell_h
        cy      = (row_top + row_bot) / 2
        is_last = (ri == n_scenes - 1)

        _figtext(fig, margin + label_w * 0.5, cy,
                  SCENE_SHORT_NAMES.get(sc, sc),
                  ha="center", va="center",
                  fontsize=ROW_LABEL_PT, fontweight="bold",
                  rotation=90, color="#222")

        crp_gt = np.load(os.path.join(eval_dir, sc, "crp_gt.npy"))
        crp_pr = np.load(os.path.join(eval_dir, sc, "crp_pred_corr.npy"))
        adc_gt = np.load(os.path.join(eval_dir, sc, "adc_gt.npy"))
        adc_pr = np.load(os.path.join(eval_dir, sc, "adc_pred_corr.npy"))
        va = _pick_representative_va(crp_gt, start_bin=start_bin)

        gt = crp_gt[va, start_bin:]
        pr = crp_pr[va, start_bin:]
        x  = np.arange(start_bin, crp_gt.shape[1])
        ax = _ax_at(fig, crp_cell_x, row_bot, cell_w, cell_h, zorder=3)
        _render_iq(ax, gt, pr, x, vline=start_bin,
                    xticks=crp_xticks, show_xticklabels=is_last)
        c_psnr_VR = per[sc]["test"]["crp"]["complex_psnr_VR"]
        ax.text(0.985, 0.97,
                  fr"VA {va}    "
                  fr"$\mathrm{{PSNR}}^{{\mathrm{{VR}}}}_{{\mathbb{{C}}}}"
                  fr"={c_psnr_VR:.1f}\,\mathrm{{dB}}$",
                  transform=ax.transAxes,
                  ha="right", va="top",
                  fontsize=METRIC_PT, color="#222",
                  fontweight="bold", bbox=label_bbox)

        gt_a = adc_gt[va]
        pr_a = adc_pr[va]
        x_a  = np.arange(adc_gt.shape[1])
        ax = _ax_at(fig, adc_cell_x, row_bot, cell_w, cell_h, zorder=3)
        _render_iq(ax, gt_a, pr_a, x_a,
                    xticks=adc_xticks, show_xticklabels=is_last)
        a_corr_VR  = per[sc]["test"]["adc"]["complex_corr_VR"]
        a_phase_VR = per[sc]["test"]["adc"]["phase_rmse_deg_VR"]
        ax.text(0.985, 0.97,
                  fr"VA {va}    "
                  fr"$|\rho|^{{\mathrm{{VR}}}}={a_corr_VR:.2f}$    "
                  fr"$\sigma_\phi={a_phase_VR:.0f}°$",
                  transform=ax.transAxes,
                  ha="right", va="top",
                  fontsize=METRIC_PT, color="#222",
                  fontweight="bold", bbox=label_bbox)

    _figtext(fig, crp_cell_x + cell_w / 2,
              xaxis_label_y, "range bin",
              ha="center", va="top",
              fontsize=ROW_LABEL_PT, color="#444")
    _figtext(fig, adc_cell_x + cell_w / 2,
              xaxis_label_y, "ADC sample",
              ha="center", va="top",
              fontsize=ROW_LABEL_PT, color="#444")

    _draw_legend_strip(fig, fig_w / 2, legend_y, fig_w, fig_h)

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, dpi=SAVE_DPI, facecolor="none", edgecolor="none")
    plt.close(fig)
    print(f"Saved {output_pdf}")


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="output/crp_adc_eval")
    ap.add_argument("--results_json",
                     default="output/crp_adc_eval/results.json")
    ap.add_argument("--output_dir",
                     default="output/crp_adc_eval/figures")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # The dB-scale heatmaps figure was dropped from the supplement (see
    # sec/A_supplement.tex: only the linear-scale heatmaps + I/Q traces
    # are kept). The fig_heatmaps function still supports scale="dB" if
    # ever needed; no caller emits it from the canonical pipeline.
    fig_heatmaps(args.eval_dir, args.results_json,
                  os.path.join(args.output_dir,
                                "fig_crp_adc_heatmaps_linear.pdf"),
                  scale="linear")
    fig_iq_traces(args.eval_dir, args.results_json,
                   os.path.join(args.output_dir,
                                 "fig_crp_adc_iq_traces.pdf"))


if __name__ == "__main__":
    main()
