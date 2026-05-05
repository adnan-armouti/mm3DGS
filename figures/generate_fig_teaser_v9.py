#!/usr/bin/env python3
"""3DPS oral teaser v9 — title-coloured, two variants, 3-row applications.

Variants:
  --variant a  : tile bg = light red/light green, NO border. Title coloured.
  --variant b  : thin coloured border around tile (image fills). Title matches.

Bottom row:
  Rendering: 3 rows (GT / 3DPS / RadarSplat) × 3 cols (ADC / CRP / |RA|)
             with arrows; RadarSplat shows ✗ on ADC/CRP cells.
  Compression: stacked-cards → arrow → compressed points.
  NVS: vertical trajectory bar on the LEFT, 3 RA images stacked on the
       right (top GT, middle Ours, bottom RadarSplat).
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
from matplotlib.path import Path
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from figures.fig_common import (
    CORNER_RADIUS_INCHES,
    FIG_WIDTH_INCHES,
    _rounded_rect_path,
    apply_paper_font,
)

apply_paper_font()


# ── Visual tokens ─────────────────────────────────────────────────────────

ROW_BG       = "#ECEBEB"
BOT_ROW_BG   = "#f5f5f5"          # bottom row outer (matches pipeline fig bg)
COL_TILE_BG  = "#ebebeb"          # bottom-row inner column tiles (= pipeline TILE_COLOR)
BLUE_TILE_BG = "#c5dbeb"          # NVS train sub-tiles (light blue)
RED_TILE_BG  = "#f3c4c4"          # NVS test sub-tile (light red — matches TRAJ_RED)
TOP_RED      = "#FBE7E6"          # variant-A tile fill (left two)
TOP_GREEN    = "#E5F4E5"          # variant-A tile fill (ours)
ROW_TEXT     = "#1a1a1a"
SUBTLE_TEXT  = "#5a5a5a"
APPL_HEADER  = "#222222"
TRAJ_BLUE    = "#4d8eb8"
TRAJ_RED     = "#eb4d4d"

CHECK     = r"$\checkmark$"
CROSS     = r"$\boldsymbol{\times}$"
GREEN_OK  = "#1d8a3a"
RED_BAD   = "#b3271e"

OURS_ACCENT  = "#0e6b2c"
B_RED_BORDER   = "#b3271e"
B_GREEN_BORDER = "#1d8a3a"


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _draw_rounded_bg(fig, x0, y0, w, h, fig_w, fig_h, color,
                      radius_in=CORNER_RADIUS_INCHES, zorder=0):
    path = _rounded_rect_path(x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h,
                               fig_w, fig_h, radius_in=radius_in)
    fig.patches.append(mpatches.PathPatch(
        path, transform=fig.transFigure,
        facecolor=color, edgecolor="none", linewidth=0, zorder=zorder,
    ))


def _rounded_rect_per_corner(x0, y0, w, h, fig_w, fig_h, radius_in,
                              corners=(True, True, True, True)):
    """Rounded-rect Path with selective corner rounding.

    corners = (TL, TR, BR, BL) — True rounds the corner, False makes it sharp.
    Coordinates are in figure-fraction.
    """
    rx = radius_in / fig_w
    ry = radius_in / fig_h
    rx = min(rx, w * 0.5)
    ry = min(ry, h * 0.5)
    k = 0.5523
    x1, y1 = x0 + w, y0 + h
    tl, tr, br, bl = corners

    verts, codes = [], []
    # Start near bottom-left
    if bl:
        verts.append((x0, y0 + ry)); codes.append(Path.MOVETO)
        verts += [(x0, y0 + ry * (1 - k)),
                  (x0 + rx * (1 - k), y0),
                  (x0 + rx, y0)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x0, y0)); codes.append(Path.MOVETO)
    # Bottom edge to BR
    if br:
        verts.append((x1 - rx, y0)); codes.append(Path.LINETO)
        verts += [(x1 - rx * (1 - k), y0),
                  (x1, y0 + ry * (1 - k)),
                  (x1, y0 + ry)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x1, y0)); codes.append(Path.LINETO)
    # Right edge to TR
    if tr:
        verts.append((x1, y1 - ry)); codes.append(Path.LINETO)
        verts += [(x1, y1 - ry * (1 - k)),
                  (x1 - rx * (1 - k), y1),
                  (x1 - rx, y1)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x1, y1)); codes.append(Path.LINETO)
    # Top edge to TL
    if tl:
        verts.append((x0 + rx, y1)); codes.append(Path.LINETO)
        verts += [(x0 + rx * (1 - k), y1),
                  (x0, y1 - ry * (1 - k)),
                  (x0, y1 - ry)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        verts.append((x0, y1)); codes.append(Path.LINETO)
    verts.append(verts[0]); codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


def _draw_rounded_bg_corners(fig, x0, y0, w, h, fig_w, fig_h, color,
                              radius_in, corners, zorder=0):
    path = _rounded_rect_per_corner(
        x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h,
        fig_w, fig_h, radius_in=radius_in, corners=corners)
    fig.patches.append(mpatches.PathPatch(
        path, transform=fig.transFigure,
        facecolor=color, edgecolor="none", linewidth=0, zorder=zorder,
    ))


def _draw_rounded_border(fig, x0, y0, w, h, fig_w, fig_h, color,
                         radius_in, lw=1.4, zorder=4):
    path = _rounded_rect_path(x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h,
                               fig_w, fig_h, radius_in=radius_in)
    fig.patches.append(mpatches.PathPatch(
        path, transform=fig.transFigure,
        facecolor="none", edgecolor=color, linewidth=lw, zorder=zorder,
    ))


def _imshow(fig, image_path, x_in, y_in, w_in, h_in, fig_w, fig_h,
             zorder=2, aspect="auto", clip_path=None):
    ax = fig.add_axes([x_in / fig_w, y_in / fig_h,
                        w_in / fig_w, h_in / fig_h])
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    img = np.asarray(Image.open(image_path))
    im = ax.imshow(img, aspect=aspect)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_zorder(zorder)
    if clip_path is not None:
        im.set_clip_path(clip_path, transform=fig.transFigure)
    return ax


def _x_cell(fig, x_in, y_in, w_in, h_in, fig_w, fig_h, zorder=2):
    """Render a 'not produced' cell — soft gray bg + centred ✗."""
    radius = CORNER_RADIUS_INCHES * 0.35
    _draw_rounded_bg(fig, x_in, y_in, w_in, h_in, fig_w, fig_h, "#dadada",
                      radius_in=radius, zorder=zorder)
    fig.text((x_in + w_in / 2) / fig_w, (y_in + h_in / 2) / fig_h,
             CROSS, ha="center", va="center",
             fontsize=10, fontweight="bold", color=RED_BAD,
             transform=fig.transFigure, zorder=zorder + 1)


def _figtext(fig, x_in, y_in, fig_w, fig_h, text, **kwargs):
    return fig.text(x_in / fig_w, y_in / fig_h, text,
                     transform=fig.transFigure, **kwargs)


def _draw_arrow(fig, p0_in, p1_in, fig_w, fig_h, color="#666",
                 lw=0.9, zorder=4, head_length=3, head_width=2.5):
    arrow = mpatches.FancyArrowPatch(
        (p0_in[0] / fig_w, p0_in[1] / fig_h),
        (p1_in[0] / fig_w, p1_in[1] / fig_h),
        transform=fig.transFigure,
        arrowstyle=f"->,head_length={head_length},head_width={head_width}",
        linewidth=lw, color=color, zorder=zorder,
        shrinkA=0, shrinkB=0,
    )
    fig.patches.append(arrow)


# ───────────────────────────────────────────────────────────────────────────
# Top-row tile
# ───────────────────────────────────────────────────────────────────────────

def _draw_top_tile(fig, x0_in, y0_in, w_in, tile_h_in, fig_w, fig_h,
                    title, image_path, checks,
                    title_color, accent_color, variant,
                    pill_w_in, pill_h_in, pill_offset_y_in):
    """Variant 'a' = filled tile bg, no border. Variant 'b' = thin border."""

    # Title above the tile, coloured to match border / bg
    title_y = y0_in + tile_h_in + 0.18
    _figtext(fig, x0_in + w_in / 2, title_y, fig_w, fig_h, title,
              ha="center", va="top",
              fontsize=8.6, fontweight="bold",
              color=title_color, zorder=3)

    radius = CORNER_RADIUS_INCHES * 0.7

    if variant == "a":
        # Image (with native colored bg) fills entire tile, no border, no padding
        img_clip = _rounded_rect_path(
            x0_in / fig_w, y0_in / fig_h, w_in / fig_w, tile_h_in / fig_h,
            fig_w, fig_h, radius_in=radius)
        _imshow(fig, image_path, x0_in, y0_in, w_in, tile_h_in, fig_w, fig_h,
                 zorder=2, aspect="auto", clip_path=img_clip)
    else:
        # Image fills entire tile + thin coloured border
        img_clip = _rounded_rect_path(
            x0_in / fig_w, y0_in / fig_h, w_in / fig_w, tile_h_in / fig_h,
            fig_w, fig_h, radius_in=radius)
        _imshow(fig, image_path, x0_in, y0_in, w_in, tile_h_in, fig_w, fig_h,
                 zorder=2, aspect="auto", clip_path=img_clip)
        _draw_rounded_border(fig, x0_in, y0_in, w_in, tile_h_in, fig_w, fig_h,
                              accent_color, radius_in=radius, lw=1.4, zorder=5)

    # Floating check pill — gray bg, black text. Larger radius so the
    # circular corner (r_x = r_y in inches) reads clearly as a circle.
    pill_x = x0_in + (w_in - pill_w_in) / 2
    pill_y = y0_in + pill_offset_y_in
    pill_radius = CORNER_RADIUS_INCHES * 0.85    # ≈0.068 in
    _draw_rounded_bg(fig, pill_x, pill_y, pill_w_in, pill_h_in,
                      fig_w, fig_h, ROW_BG, radius_in=pill_radius, zorder=6)
    n = len(checks)
    sym_x = pill_x + 0.05
    txt_x = pill_x + 0.18
    rows_top = pill_y + pill_h_in - 0.05
    row_step = (pill_h_in - 0.10) / max(n, 1)
    for i, (sym, txt) in enumerate(checks):
        y = rows_top - (i + 0.5) * row_step
        col = GREEN_OK if sym == CHECK else RED_BAD
        _figtext(fig, sym_x, y, fig_w, fig_h, sym,
                  ha="left", va="center",
                  fontsize=9.0, fontweight="bold", color=col, zorder=7)
        _figtext(fig, txt_x, y, fig_w, fig_h, txt,
                  ha="left", va="center",
                  fontsize=7.6, color="black", zorder=7)


# ───────────────────────────────────────────────────────────────────────────
# Bottom-row applications
# ───────────────────────────────────────────────────────────────────────────

def _draw_rendering_column(fig, x0_in, y0_in, w_in, h_in, fig_w, fig_h,
                            panel_dir):
    head_y = y0_in + h_in - 0.02
    _figtext(fig, x0_in + w_in / 2, head_y, fig_w, fig_h,
              "Product-Agnostic Rendering",
              ha="center", va="top", fontsize=7.4, fontweight="bold",
              color=APPL_HEADER, zorder=3)
    body_top = head_y - 0.16
    body_bot = y0_in + 0.02
    body_h = body_top - body_bot

    label_w = 0.16
    pad = 0.02
    cap_h = 0.10
    rows_avail = body_h - cap_h - 0.02
    row_gap = 0.025
    # Maximize cell size (height-limited) then use ~2/3-length arrows.
    cell_h = (rows_avail - 2 * row_gap) / 3
    arrow_w = 0.27       # ≈ 2/3 of prior 0.40
    avail_w_no_label = w_in - label_w - 2 * pad
    cell_w_from_arrow = (avail_w_no_label - 2 * arrow_w) / 3
    cell_w = min(cell_h, cell_w_from_arrow)
    cell_h = cell_w

    # Centre the entire content block within the column tile.
    content_w = label_w + pad + 3 * cell_w + 2 * arrow_w + pad
    block_x = x0_in + (w_in - content_w) / 2

    cap_y = body_top - 0.02
    col_xs = [block_x + label_w + pad + i * (cell_w + arrow_w) for i in range(3)]
    cap_color = "#6a6a6a"
    cap_labels = [r"$\mathrm{ADC}$", r"$\mathrm{CRP}$", r"$|\mathrm{RA}|$"]
    for cx, cap in zip(col_xs, cap_labels):
        _figtext(fig, cx + cell_w / 2, cap_y, fig_w, fig_h, cap,
                  ha="center", va="top",
                  fontsize=6.4, color=cap_color, zorder=3)

    rows = [
        ("GT",     "#444",
         ["panel_adc_gt_train.png", "panel_crp_gt.png", "panel_ra_gt_train.png"]),
        ("3DPS",   OURS_ACCENT,
         ["panel_adc_ours_train.png", "panel_crp_ours.png", "panel_ra_ours_train.png"]),
        ("RS",     RED_BAD,
         [None, None, "panel_ra_radarsplat_train.png"]),
    ]
    rows_top = cap_y - cap_h
    for ri, (rname, rcolor, paths) in enumerate(rows):
        cell_y = rows_top - (ri + 1) * cell_h - ri * row_gap
        _figtext(fig, block_x + label_w - 0.02, cell_y + cell_h / 2,
                  fig_w, fig_h, rname,
                  ha="right", va="center",
                  fontsize=7.2, fontweight="bold", color=rcolor, zorder=3)
        for ci, fname in enumerate(paths):
            cx = col_xs[ci]
            if fname is None:
                _x_cell(fig, cx, cell_y, cell_w, cell_h, fig_w, fig_h)
            else:
                full = os.path.join(panel_dir, fname)
                if os.path.exists(full):
                    _imshow(fig, full, cx, cell_y, cell_w, cell_h, fig_w,
                             fig_h, aspect="equal")
        ay = cell_y + cell_h / 2
        _draw_arrow(fig, (col_xs[1] - 0.015, ay),
                          (col_xs[0] + cell_w + 0.015, ay),
                          fig_w, fig_h, lw=0.5,
                          head_length=3, head_width=2.5)
        _draw_arrow(fig, (col_xs[1] + cell_w + 0.015, ay),
                          (col_xs[2] - 0.015, ay),
                          fig_w, fig_h, lw=0.5,
                          head_length=3, head_width=2.5)
        # FFT-direction labels above each arrow.
        ifft_x = (col_xs[0] + cell_w + col_xs[1]) / 2
        fft_x  = (col_xs[1] + cell_w + col_xs[2]) / 2
        _figtext(fig, ifft_x, ay + 0.04, fig_w, fig_h,
                  r"$\mathcal{F}^{-1}_{r}$",
                  ha="center", va="bottom",
                  fontsize=6.8, color=cap_color, zorder=3)
        _figtext(fig, fft_x, ay + 0.04, fig_w, fig_h,
                  r"$\mathcal{F}_{\theta}$",
                  ha="center", va="bottom",
                  fontsize=6.8, color=cap_color, zorder=3)


def _draw_compression_column(fig, x0_in, y0_in, w_in, h_in, fig_w, fig_h,
                              panel_dir):
    head_y = y0_in + h_in - 0.02
    _figtext(fig, x0_in + w_in / 2, head_y, fig_w, fig_h,
              "Compression, Material Reconstruction\nand Normals Estimation",
              ha="center", va="top", fontsize=6.6, fontweight="bold",
              color=APPL_HEADER, zorder=3, linespacing=1.05)
    body_top = head_y - 0.24       # extra room for 2-line title
    body_bot = y0_in + 0.02
    body_h = body_top - body_bot
    cap_h = 0.10
    body_used_h = body_h - cap_h

    arrow_w = 0.14
    side_pad = 0.02
    avail_w = w_in - 2 * side_pad - arrow_w
    half_w = avail_w / 2
    side = min(half_w, body_used_h)

    total = side * 2 + arrow_w
    inset_x = max((w_in - total) / 2, side_pad)
    cards_x = x0_in + inset_x
    pts_x = cards_x + side + arrow_w
    panel_y = body_bot + cap_h + (body_used_h - side) / 2

    _imshow(fig, os.path.join(panel_dir, "panel_stacked_cards.png"),
             cards_x, panel_y, side, side, fig_w, fig_h, aspect="equal")
    _imshow(fig, os.path.join(panel_dir, "panel_compressed_pts.png"),
             pts_x, panel_y, side, side, fig_w, fig_h, aspect="equal")
    arrow_x0 = cards_x + side + 0.015
    arrow_x1 = pts_x - 0.015
    arrow_y = panel_y + side / 2
    _draw_arrow(fig, (arrow_x0, arrow_y), (arrow_x1, arrow_y),
                 fig_w, fig_h, color="#444", lw=1.3,
                 head_length=4, head_width=3.5)
    cap_y = panel_y - 0.02
    _figtext(fig, cards_x + side / 2, cap_y, fig_w, fig_h,
              r"8 train $|\mathrm{RA}|$",
              ha="center", va="top", fontsize=7.2, color=SUBTLE_TEXT, zorder=3)
    _figtext(fig, pts_x + side / 2, cap_y, fig_w, fig_h,
              r"$N{=}20{,}000$ pts",
              ha="center", va="top", fontsize=7.2, color=SUBTLE_TEXT,
              zorder=3)


def _draw_nvs_column(fig, x0_in, y0_in, w_in, h_in, fig_w, fig_h, panel_dir,
                     test_cc):
    head_y = y0_in + h_in - 0.02
    _figtext(fig, x0_in + w_in / 2, head_y, fig_w, fig_h,
              "Novel View Synthesis",
              ha="center", va="top", fontsize=7.4, fontweight="bold",
              color=APPL_HEADER, zorder=3)
    body_top = head_y - 0.16
    body_bot = y0_in + 0.02
    body_h = body_top - body_bot

    # ── Geometry: 3 rows (GT/Ours/RS) × 5 cols (F-4, F-1, TEST, F+1, F+4) ──
    label_w   = 0.28           # widened so larger row labels fit + grid shifts right
    sub_pad   = 0.02
    row_gap   = 0.012
    ell_gap   = 0.07           # tighter — keeps ellipsis clear of RA images
    g_inner   = 0.0            # touch — no gap between blue/red sub-tiles
    outer_y   = 0.015
    traj_h    = 0.16
    intra_gap = 0.015

    avail_w = w_in - label_w - 0.04
    avail_h = body_h - 2 * outer_y - traj_h - intra_gap
    cell_w_max = (avail_w - 2 * ell_gap - 6 * sub_pad - 2 * g_inner) / 5
    cell_h_max = (avail_h - 2 * row_gap - 2 * sub_pad) / 3
    cell = min(cell_w_max, cell_h_max)

    side_w   = 2 * cell + ell_gap + 2 * sub_pad
    center_w = cell + 2 * sub_pad
    layout_w = 2 * side_w + center_w + 2 * g_inner
    grid_h   = 3 * cell + 2 * row_gap + 2 * sub_pad
    bg_h     = grid_h + intra_gap + traj_h     # bgs span grid + trajectory

    block_x0 = x0_in + label_w + (avail_w - layout_w) / 2
    sub_y    = body_bot + outer_y
    sub_x_left   = block_x0
    sub_x_center = sub_x_left + side_w + g_inner
    sub_x_right  = sub_x_center + center_w + g_inner

    # ── Tall sub-tile bgs spanning grid + trajectory band ──────────────────
    # Selective rounding so blue/red touch with no gap on the inner edges.
    sub_radius = CORNER_RADIUS_INCHES * 0.6
    # Left blue: round outer corners (TL, BL); inner edge sharp (TR, BR).
    _draw_rounded_bg_corners(fig, sub_x_left,   sub_y, side_w, bg_h,
                              fig_w, fig_h, BLUE_TILE_BG,
                              radius_in=sub_radius,
                              corners=(True, False, False, True), zorder=2)
    # Center red: all 4 corners sharp (touches both sides).
    _draw_rounded_bg_corners(fig, sub_x_center, sub_y, center_w, bg_h,
                              fig_w, fig_h, RED_TILE_BG,
                              radius_in=sub_radius,
                              corners=(False, False, False, False), zorder=2)
    # Right blue: round outer corners (TR, BR); inner edge sharp (TL, BL).
    _draw_rounded_bg_corners(fig, sub_x_right,  sub_y, side_w, bg_h,
                              fig_w, fig_h, BLUE_TILE_BG,
                              radius_in=sub_radius,
                              corners=(False, True, True, False), zorder=2)

    # ── Trajectory band at top of bg column ────────────────────────────────
    traj_y = sub_y + grid_h + intra_gap
    ax = fig.add_axes([block_x0 / fig_w, traj_y / fig_h,
                        layout_w / fig_w, traj_h / fig_h])
    ax.set_facecolor("none"); ax.patch.set_alpha(0)

    # Place 9 dots aligned to cell column centers (with F-3, F-2 interpolated
    # between F-4 and F-1; F+2, F+3 between F+1 and F+4).
    def _frac(x_abs):
        return (x_abs - block_x0) / layout_w
    f_m4 = _frac(sub_x_left   + sub_pad + cell / 2)
    f_m1 = _frac(sub_x_left   + sub_pad + cell + ell_gap + cell / 2)
    f_t  = _frac(sub_x_center + sub_pad + cell / 2)
    f_p1 = _frac(sub_x_right  + sub_pad + cell / 2)
    f_p4 = _frac(sub_x_right  + sub_pad + cell + ell_gap + cell / 2)
    train_xs = [
        f_m4,
        f_m4 + (f_m1 - f_m4) / 3,
        f_m4 + 2 * (f_m1 - f_m4) / 3,
        f_m1,
        f_p1,
        f_p1 + (f_p4 - f_p1) / 3,
        f_p1 + 2 * (f_p4 - f_p1) / 3,
        f_p4,
    ]
    ax.plot([0.0, 1.0], [0.22, 0.22], color="#bbb", linewidth=0.8, zorder=1)
    ax.scatter(train_xs, [0.22] * len(train_xs),
               color=TRAJ_BLUE, s=12, edgecolors="none", zorder=3)
    ax.scatter([f_t], [0.22], color=TRAJ_RED, s=26,
               edgecolors="none", zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_zorder(3)

    # ── Row labels (GT / Ours / RS) on left, kept inside column tile ──────
    row_meta = [("GT", "#444"), ("3DPS", OURS_ACCENT), ("RS", RED_BAD)]
    for ri, (rname, rcolor) in enumerate(row_meta):
        cell_y = sub_y + grid_h - sub_pad - (ri + 1) * cell - ri * row_gap
        _figtext(fig, sub_x_left - 0.04, cell_y + cell / 2,
                  fig_w, fig_h, rname,
                  ha="right", va="center",
                  fontsize=7.2, fontweight="bold", color=rcolor, zorder=4)

    # ── Place 3×5 RA cells ─────────────────────────────────────────────────
    frames   = [181, 184, 185, 186, 189]
    row_keys = ["gt", "ours", "rs"]
    col_xs = [
        sub_x_left   + sub_pad,
        sub_x_left   + sub_pad + cell + ell_gap,
        sub_x_center + sub_pad,
        sub_x_right  + sub_pad,
        sub_x_right  + sub_pad + cell + ell_gap,
    ]
    for ri, key in enumerate(row_keys):
        cell_y = sub_y + grid_h - sub_pad - (ri + 1) * cell - ri * row_gap
        for ci, F in enumerate(frames):
            path = os.path.join(panel_dir, f"panel_nvs_{key}_F{F}.png")
            if os.path.exists(path):
                _imshow(fig, path, col_xs[ci], cell_y, cell, cell,
                         fig_w, fig_h, aspect="equal", zorder=4)
        # Single ellipsis on middle row (left + right gaps).
        if ri == 1:
            for ell_cx in [sub_x_left  + sub_pad + cell + ell_gap / 2,
                            sub_x_right + sub_pad + cell + ell_gap / 2]:
                _figtext(fig, ell_cx, cell_y + cell / 2, fig_w, fig_h,
                          r"$\cdots$", ha="center", va="center",
                          fontsize=6, color="#3b6f8e", fontweight="bold",
                          zorder=5)


# ───────────────────────────────────────────────────────────────────────────


def generate(args):
    panel_dir = os.path.join(args.panel_dir, args.scene, "v4")
    if not os.path.isdir(panel_dir):
        sys.exit(f"Panel dir not found: {panel_dir}")
    run_dir = os.path.join(args.ours_dir,
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    test_cc = float(json.load(open(os.path.join(run_dir, "results.json")))
                       .get("final_test_cc", 0.0))

    # Layout
    fig_w = FIG_WIDTH_INCHES
    margin_in = 0.06
    row_pad_in = 0.06
    row_gap_in = 0.10

    n_top = 3
    inner_w = fig_w - 2 * margin_in - 2 * row_pad_in
    tile_gap = 0.12
    tile_w = (inner_w - (n_top - 1) * tile_gap) / n_top
    tile_h = tile_w / 1.5
    title_block_h = 0.22
    row_h = title_block_h + tile_h + 2 * row_pad_in

    fig_h = margin_in + row_h + row_gap_in + row_h + margin_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("none")

    top_row_y = margin_in + row_h + row_gap_in
    bot_row_y = margin_in
    row_x = margin_in
    row_w = fig_w - 2 * margin_in
    _draw_rounded_bg(fig, row_x, top_row_y, row_w, row_h, fig_w, fig_h,
                      ROW_BG, zorder=0)
    _draw_rounded_bg(fig, row_x, bot_row_y, row_w, row_h, fig_w, fig_h,
                      BOT_ROW_BG, zorder=0)

    tile_y = top_row_y + row_pad_in

    pill_w = min(tile_w * 0.65, 1.35)
    pill_h = 0.46
    pill_offset_y = 0.05

    mesh_img     = "panel_mesh_only_redbg.png"     if args.variant == "a" else "panel_mesh_only.png"
    implicit_img = "panel_implicit_redbg.png"      if args.variant == "a" else "panel_implicit.png"
    points_img   = "panel_3dps_points_greenbg.png" if args.variant == "a" else "panel_3dps_points.png"

    methods = [
        dict(title="Mesh + MC ray tracing",
             image=os.path.join(panel_dir, mesh_img),
             checks=[
                 (CHECK, "physics-based"),
                 (CHECK, "product-agnostic, complex"),
                 (CROSS, r"slow ($\sim$160 min)"),
             ],
             title_color=B_RED_BORDER, accent_color=(TOP_RED if args.variant == "a" else B_RED_BORDER)),
        dict(title="NeRF / 3DGS Primitives",
             image=os.path.join(panel_dir, implicit_img),
             checks=[
                 (CROSS, "approximated"),
                 (CROSS, r"power-only $|\mathrm{RA}|$"),
                 (CHECK, "fast"),
             ],
             title_color=B_RED_BORDER, accent_color=(TOP_RED if args.variant == "a" else B_RED_BORDER)),
        dict(title="3DPS: Point Primitives",
             image=os.path.join(panel_dir, points_img),
             checks=[
                 (CHECK, "physics-based"),
                 (CHECK, "product-agnostic, complex"),
                 (CHECK, r"fast ($\sim$3 min)"),
             ],
             title_color=B_GREEN_BORDER, accent_color=(TOP_GREEN if args.variant == "a" else B_GREEN_BORDER)),
    ]

    for i, m in enumerate(methods):
        x0 = row_x + row_pad_in + i * (tile_w + tile_gap)
        _draw_top_tile(fig, x0, tile_y, tile_w, tile_h, fig_w, fig_h,
                        m["title"], m["image"], m["checks"],
                        title_color=m["title_color"],
                        accent_color=m["accent_color"],
                        variant=args.variant,
                        pill_w_in=pill_w, pill_h_in=pill_h,
                        pill_offset_y_in=pill_offset_y)

    # Bottom row — column tile top (A) and outer row top (B); the
    # "Applications" label is vertically centered between them.
    bot_row_top = bot_row_y + row_h
    body_bot    = bot_row_y + 0.06
    header_band = 0.24                         # space reserved for "Applications"
    col_top     = bot_row_top - header_band
    body_top    = col_top
    body_h_app  = body_top - body_bot
    app_header_y = (col_top + bot_row_top) / 2     # midpoint
    _figtext(fig, fig_w / 2, app_header_y, fig_w, fig_h, "Applications",
              ha="center", va="center",
              fontsize=8.4, fontweight="bold",
              color=APPL_HEADER, zorder=3)

    # Three inner column tiles (#ebebeb), no borders. Widths match the top
    # row's method-tile widths so columns align between rows.
    col_gap = tile_gap                # = 0.12, same as top row
    col_widths = [tile_w, tile_w, tile_w]

    col_xs = [row_x + row_pad_in]
    for cw in col_widths[:-1]:
        col_xs.append(col_xs[-1] + cw + col_gap)

    tile_radius = CORNER_RADIUS_INCHES * 0.55
    for cx, cw in zip(col_xs, col_widths):
        _draw_rounded_bg(fig, cx, body_bot, cw, body_h_app, fig_w, fig_h,
                          COL_TILE_BG, radius_in=tile_radius, zorder=1)

    inset = 0.04
    _draw_rendering_column(fig, col_xs[0] + inset, body_bot + inset,
                            col_widths[0] - 2 * inset, body_h_app - 2 * inset,
                            fig_w, fig_h, panel_dir)
    _draw_compression_column(fig, col_xs[1] + inset, body_bot + inset,
                              col_widths[1] - 2 * inset, body_h_app - 2 * inset,
                              fig_w, fig_h, panel_dir)
    _draw_nvs_column(fig, col_xs[2] + inset, body_bot + inset,
                      col_widths[2] - 2 * inset, body_h_app - 2 * inset,
                      fig_w, fig_h, panel_dir, test_cc)

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_v9{args.variant}"
    out_pdf = os.path.join(args.output_dir, f"teaser_3dps{suffix}.pdf")
    out_png = os.path.join(args.output_dir, f"teaser_3dps{suffix}.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_pdf}\n       {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["a", "b"], default="b")
    p.add_argument("--scene", default="seq_1_frame_185")
    p.add_argument("--test_frame", type=int, default=185)
    p.add_argument("--panel_dir",
                   default=os.path.join(_REPO, "output/teaser_panels"))
    p.add_argument("--ours_dir",
                   default=os.path.join(_REPO, "mm25DGS_v5_v4/output_frame_nvs"))
    p.add_argument("--output_dir",
                   default=os.path.join(_REPO, "output/postprocess_final_v5/figures"))
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
