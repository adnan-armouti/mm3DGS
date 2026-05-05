#!/usr/bin/env python3
"""Generate the pipeline overview figure (Figure 2) for the paper.

Layout: 4 sections with outer panels, grouped sub-panels,
bidirectional dashed arrows (green=forward, red=gradient).

Usage:
    python -m mmir.evaluation.generate_fig_pipeline \\
        --output_dir output/postprocess_final_v5/figures \\
        --center_variant all
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys as _sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from figures.fig_common import (BACKGROUND_COLOR, FIG_WIDTH_INCHES,
                                  add_rounded_bg, apply_paper_font)

# Apply NeurIPS paper typography (Times serif) to all figures.
apply_paper_font()



# ══════════════════════════════════════════════════════════════════════════════
# Layout constants (all in inches)
# ══════════════════════════════════════════════════════════════════════════════

FIG_W = 5.03             # snug — row tile has ~0.03 in. empty left + right of
                          # the leftmost / rightmost column tile.
FIG_H = 1.30             # snug — row tile has ~0.03 in. empty below the column
                          # tiles.  Aspect ≈ 4.16 (was 4.29) — small drift to
                          # remove all wasted space.

# Section widths.  All inter-section gaps are pinned to GAP_REF = 0.14
# (the Sec1→Sec2 arrow width) so EVERY arrow in the figure has the same
# horizontal extent.  SEC2_W is sized so the Section-2 tile ends just
# past the Splat tile (= 2*col1_w + GAP_REF + outer insets ≈ 1.62 in).
SEC1_W = 1.77          # snug around the points-scene image: image axes width
                        # + 2 × top_pad inside the column tile.
SEC2_W = 1.62          # tight: col1 + 0.14 gap + Splat + outer insets
SEC3_W = 0.55
SEC4_W = 0.55

# Gaps between sections — all pinned to the same reference width.
GAP_REF = 0.14
GAP_12 = GAP_REF
GAP_23 = GAP_REF
GAP_34 = GAP_REF

# Outer margin around the figure-bg rounded rect — matches the teaser's
# row-tile margin (so the 3DPS row tile is visually flush with the teaser's).
MARGIN_OUT = 0.06
# Section content margin equal to MARGIN_OUT — sections start at the row-tile
# edge so the only padding between row tile and column tile is the per-section
# IPAD (0.03 in.).
MARGIN = 0.06

# Section x-positions
SEC1_L = MARGIN
SEC2_L = SEC1_L + SEC1_W + GAP_12
SEC3_L = SEC2_L + SEC2_W + GAP_23
SEC4_L = SEC3_L + SEC3_W + GAP_34

# Vertical layout — SEC_PAD set to MARGIN_OUT so the section title sits just
# below the row-tile top edge (no extra outer band).
SEC_PAD = MARGIN_OUT
SEC_TOP = FIG_H - SEC_PAD
SEC_BOT = SEC_PAD
SEC_H = SEC_TOP - SEC_BOT

HEADER_H = 0.10        # tightened (was 0.16) — fits 5.5pt section titles and
                        # lets the figure shrink without losing content.
LABEL_H = 0.10
CONTENT_TOP = SEC_TOP - HEADER_H
CONTENT_BOT = SEC_BOT + LABEL_H
CONTENT_H = CONTENT_TOP - CONTENT_BOT

# ── Colors: 2-layer system (light bg + darker tiles) ─────────────────────────
BG_COLOR = "#f5f5f5"           # figure background (overrides fig_common)
TILE_COLOR = "#ebebeb"         # all 6 inner tiles + forward model panel
BLOCK_COLOR = "#e0e0e0"        # forward model sub-blocks (darker than tile)
TILE_RADIUS = 0.025
SECTION_RADIUS = 0.04

# Arrow spacing: consistent physical gap between green/red pairs
ARROW_OFFSET_IN = 0.028   # offset from center line in inches
_ARROW_XOFF = ARROW_OFFSET_IN / FIG_W   # x-offset in figure fraction (~0.004)
_ARROW_YOFF = ARROW_OFFSET_IN / FIG_H   # y-offset in figure fraction (~0.011)

# ── Section 1 sub-panel layout ───────────────────────────────────────────────
SEC1_IPAD = 0.03
SUBPANEL_GAP = 0.06
SUBPANEL_W = SEC1_W - 2 * SEC1_IPAD
SUBPANEL_H = (CONTENT_H - SUBPANEL_GAP) / 2

SP_TITLE_H = 0.06
SP_CAPTION_H = 0.04
SP_PAD = 0.005
THUMB_GAP = 0.01
THUMB_W = (SUBPANEL_W - 2 * SP_PAD - THUMB_GAP) / 2
THUMB_H = SUBPANEL_H - SP_TITLE_H - SP_CAPTION_H - 2 * SP_PAD

# ── Column-tile geometry ──────────────────────────────────────────────────
# All 4 column tiles (Section 1 / 2 / 3 / 4) share the same height, set so
# Section 2 has 0.03 in. of empty space above Antenna Gain and 0.03 in. of
# empty space below Phase (i.e. SPLAT_H + 2 * 0.03 = TILE_H).  SPLAT_H is
# now an independent constant (not derived from CONTENT_H), so the figure
# height can shrink without shrinking the Splat.
SPLAT_H  = 0.99
TILE_H   = SPLAT_H + 0.06            # 1.05 — column tile height
TILE_TOP = CONTENT_TOP                # = top of column tile (= section content top)
TILE_BOT = TILE_TOP - TILE_H

SPLAT_TOP = TILE_TOP - 0.03           # 0.03 padding above Antenna Gain
SPLAT_BOT = TILE_BOT + 0.03           # 0.03 padding below Phase

# Backwards-compatibility shims for code that still references the old
# Section-2 inner-area names:
SEC2_INNER_TOP = SPLAT_TOP
SEC2_INNER_BOT = SPLAT_BOT
SEC2_INNER_H   = SPLAT_H

# ── Sections 3/4: CRP + Azimuth FFT + RA proportions ────────────────────────
SEC34_IPAD = 0.03
# Bumped vs mmIR (was 0.40/0.42) so the two tiles sit closer together inside
# the shorter 3DPS figure; remaining ~6% of CONTENT_H is the FFT arrow band.
ADC_FRAC = 0.47
RA_FRAC  = 0.47
FFT_FRAC = 1.0 - ADC_FRAC - RA_FRAC  # 0.06


# ══════════════════════════════════════════════════════════════════════════════
# Data paths (S1-F438)
# ══════════════════════════════════════════════════════════════════════════════

SCENE = "seq_1_frame_438"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", SCENE)
TRAIN_DIR = os.path.join(PROJECT_ROOT, "output", "train_v11", SCENE)
POSTPROC_DIR = os.path.join(PROJECT_ROOT, "output", "postprocess_final_v11")

MESH_PATH = os.path.join(DATA_DIR, "scene", "mesh.ply")
RADAR_CFG = os.path.join(DATA_DIR, "configs", "cascaded_frame_438_aligned_2dof.json")
GT_ADC_PATH = os.path.join(DATA_DIR, "radar", "cascaded_frame_438.npy")
RENDERED_ADC_PATH = os.path.join(
    POSTPROC_DIR, "training_ra", SCENE, "ours", "adc_rendered_cascaded.npy"
)
GT_RA_PATH = os.path.join(TRAIN_DIR, "ra_gt_cart.npy")
RENDERED_RA_PATH = os.path.join(TRAIN_DIR, "ra_rendered_cart.npy")
TX_PATTERN_PATH = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy")
RX_PATTERN_PATH = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy")
MESH_VIEW_PATH = os.path.join(
    POSTPROC_DIR, "mesh_only_renders", SCENE, "mesh_view_oblique_1.png"
)

# ── 3DPS-specific image sources (override mmIR ADC paths) ───────────────────
# Section 1 image: the points + normals scene render produced by
# figures/render_pipeline_panel_a.py (Open3D, teaser pattern).
_3DPS_SCENE = "seq_1_frame_185"
PANEL_A_PATH = os.path.join(PROJECT_ROOT, "output", "pipeline_panels",
                              _3DPS_SCENE, "panel_a_scene.png")
# Sections 3/4: pre-rendered CRP heatmaps from the teaser-prep pipeline.
_TEASER_PANELS = os.path.join(PROJECT_ROOT, "output", "teaser_panels",
                                _3DPS_SCENE, "v4")
RENDERED_CRP_PATH = os.path.join(_TEASER_PANELS, "panel_crp_ours.png")
GT_CRP_PATH       = os.path.join(_TEASER_PANELS, "panel_crp_gt.png")
# Override the RA paths with the same teaser-prep PNGs so all four section
# image sources are PNGs (no mixed PNG/npy code paths).
RENDERED_RA_PATH = os.path.join(_TEASER_PANELS, "panel_ra_ours_train.png")
GT_RA_PATH       = os.path.join(_TEASER_PANELS, "panel_ra_gt_train.png")


def _update_data_paths(train_dir=None, postproc_dir=None):
    """Update module-level data paths for a different training/postproc root."""
    global TRAIN_DIR, POSTPROC_DIR, RENDERED_ADC_PATH, GT_RA_PATH, RENDERED_RA_PATH, MESH_VIEW_PATH
    if train_dir is not None:
        TRAIN_DIR = os.path.join(train_dir, SCENE)
        GT_RA_PATH = os.path.join(TRAIN_DIR, "ra_gt_cart.npy")
        RENDERED_RA_PATH = os.path.join(TRAIN_DIR, "ra_rendered_cart.npy")
    if postproc_dir is not None:
        POSTPROC_DIR = postproc_dir
        RENDERED_ADC_PATH = os.path.join(
            POSTPROC_DIR, "training_ra", SCENE, "ours", "adc_rendered_cascaded.npy"
        )
        MESH_VIEW_PATH = os.path.join(
            POSTPROC_DIR, "mesh_only_renders", SCENE, "mesh_view_oblique_1.png"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _add_rounded_rect(fig, left_in, bot_in, w_in, h_in,
                      color=TILE_COLOR, radius_in=TILE_RADIUS,
                      edgecolor="none", linewidth=0,
                      zorder=-1):
    """Add rounded-rectangle patch at inch coordinates.

    Corner radius is specified in inches and converted to equal figure-fraction
    values for both axes so the Bezier curves trace circular arcs.
    """
    fw, fh = fig.get_size_inches()
    l, b, w, h = left_in / fw, bot_in / fh, w_in / fw, h_in / fh
    # Use the same fraction for rx and ry → circular corners on the rendered image.
    rx = radius_in / fw
    ry = radius_in / fh
    x0, y0, x1, y1 = l, b, l + w, b + h
    k = 0.5523
    verts = [
        (x0, y0 + ry),
        (x0, y0 + ry * (1 - k)), (x0 + rx * (1 - k), y0), (x0 + rx, y0),
        (x1 - rx, y0),
        (x1 - rx * (1 - k), y0), (x1, y0 + ry * (1 - k)), (x1, y0 + ry),
        (x1, y1 - ry),
        (x1, y1 - ry * (1 - k)), (x1 - rx * (1 - k), y1), (x1 - rx, y1),
        (x0 + rx, y1),
        (x0 + rx * (1 - k), y1), (x0, y1 - ry * (1 - k)), (x0, y1 - ry),
        (x0, y0 + ry),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = mpatches.PathPatch(
        Path(verts, codes), facecolor=color, edgecolor=edgecolor,
        linewidth=linewidth, transform=fig.transFigure, zorder=zorder,
    )
    fig.patches.append(patch)


def _ax_at(fig, left, bot, w, h, **kwargs):
    """Create axes at absolute inch coordinates with transparent bg."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([left / fw, bot / fh, w / fw, h / fh], **kwargs)
    ax.patch.set_alpha(0)
    return ax


def _get_overlay_ax(fig):
    """Get/create transparent full-figure axes for annotations."""
    for ax in fig.get_axes():
        if getattr(ax, "_is_overlay", False):
            return ax
    ax = fig.add_axes([0, 0, 1, 1], zorder=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.patch.set_alpha(0)
    ax._is_overlay = True
    return ax


def _fx(inches):
    return inches / FIG_W

def _fy(inches):
    return inches / FIG_H


def _load_and_crop_image(path, border_frac=0.05):
    """Load image and crop away white borders."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    mask = np.any(arr < 240, axis=2)
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pr = max(1, int((r1 - r0) * border_frac))
        pc = max(1, int((c1 - c0) * border_frac))
        arr = arr[max(0, r0 - pr):min(arr.shape[0], r1 + pr),
                  max(0, c0 - pc):min(arr.shape[1], c1 + pc)]
    return arr


def _lowpass_envelope(signal, cutoff_frac=0.05, order=3):
    """Extract low-frequency envelope from a complex signal via Butterworth LPF."""
    b, a = butter(order, cutoff_frac, btype='low')
    real_lp = filtfilt(b, a, signal.real)
    imag_lp = filtfilt(b, a, signal.imag)
    return real_lp + 1j * imag_lp


def _ra_cart_to_linear(ra):
    """Min-max normalize RA to [0,1] — matches training_single_v2 style."""
    vmin, vmax = ra.min(), ra.max()
    if vmax - vmin < 1e-30:
        return np.zeros_like(ra)
    return (ra - vmin) / (vmax - vmin)


# ── Bidirectional dashed arrows (thinner, more academic) ─────────────────────

def _draw_bidir_arrows(fig, x1, y1, x2, y2, bidir=True, swap_lr=False):
    """Draw paired green→ and red← dashed arrows.

    bidir   : if False, only the green forward arrow is drawn (no red).
    swap_lr : for *vertical* arrows only — flips the green/red sides so
              the green arrow ends up on the right (used by the Azimuth
              FFT inset where the symbol sits to the right of the arrows).
    """
    overlay = _get_overlay_ax(fig)
    is_vertical = abs(y2 - y1) > abs(x2 - x1)
    style = dict(lw=0.6, linestyle=(0, (4, 3)), mutation_scale=5)

    if is_vertical:
        green_off = +_ARROW_XOFF if swap_lr else -_ARROW_XOFF
        red_off   = -_ARROW_XOFF if swap_lr else +_ARROW_XOFF
        overlay.annotate(
            "", xy=(x2 + green_off, y2), xytext=(x1 + green_off, y1),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="#2ca25f", **style))
        if bidir:
            overlay.annotate(
                "", xy=(x1 + red_off, y1), xytext=(x2 + red_off, y2),
                xycoords="figure fraction", textcoords="figure fraction",
                arrowprops=dict(arrowstyle="-|>", color="#d73027", **style))
    else:
        overlay.annotate(
            "", xy=(x2, y2 + _ARROW_YOFF), xytext=(x1, y1 + _ARROW_YOFF),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="#2ca25f", **style))
        if not bidir:
            return
        overlay.annotate(
            "", xy=(x1, y1 - _ARROW_YOFF), xytext=(x2, y2 - _ARROW_YOFF),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="#d73027", **style))


def _draw_single_arrow(fig, x1, y1, x2, y2, color="#555", lw=0.5, dashed=False):
    """Draw a single thin arrow."""
    overlay = _get_overlay_ax(fig)
    ls = (0, (4, 3)) if dashed else "-"
    overlay.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        xycoords="figure fraction", textcoords="figure fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle=ls, mutation_scale=5),
    )


def _draw_bidir_s_arrows(fig, x1, y1, x2, y2, bidir=True):
    """Draw paired green/red S-shaped orthogonal arrows (H→V→H segments).

    If ``bidir`` is False, only the green forward arrow is drawn — used for
    paths that don't carry gradients (e.g. through antenna pattern, phase).
    """
    overlay = _get_overlay_ax(fig)
    xmid = (x1 + x2) / 2
    lw = 0.6
    dash = (0, (4, 3))
    ms = 5
    head_len = 0.008

    # Green forward: H-top → V-left → H-top
    gx = [x1, xmid - _ARROW_XOFF, xmid - _ARROW_XOFF, x2 - head_len]
    gy = [y1 + _ARROW_YOFF, y1 + _ARROW_YOFF, y2 + _ARROW_YOFF, y2 + _ARROW_YOFF]
    overlay.plot(gx, gy, color="#2ca25f", lw=lw, linestyle=dash,
                 clip_on=False, zorder=5)
    overlay.annotate(
        "", xy=(x2, y2 + _ARROW_YOFF), xytext=(x2 - head_len, y2 + _ARROW_YOFF),
        arrowprops=dict(arrowstyle="-|>", color="#2ca25f", lw=lw,
                        mutation_scale=ms),
    )

    if not bidir:
        return

    # Red backward: H-bottom → V-right → H-bottom
    rx = [x2, xmid + _ARROW_XOFF, xmid + _ARROW_XOFF, x1 + head_len]
    ry = [y2 - _ARROW_YOFF, y2 - _ARROW_YOFF, y1 - _ARROW_YOFF, y1 - _ARROW_YOFF]
    overlay.plot(rx, ry, color="#d73027", lw=lw, linestyle=dash,
                 clip_on=False, zorder=5)
    overlay.annotate(
        "", xy=(x1, y1 - _ARROW_YOFF), xytext=(x1 + head_len, y1 - _ARROW_YOFF),
        arrowprops=dict(arrowstyle="-|>", color="#d73027", lw=lw,
                        mutation_scale=ms),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Open3D rendering for thumbnails
# ══════════════════════════════════════════════════════════════════════════════

_TILE_RGB = [1.0, 1.0, 1.0]  # white background for thumbnails (matches SVG)
_BG_RGBA = _TILE_RGB + [1.0]


_THUMB_FOV = 30.0  # tighter FOV so the mesh fills the frame


def _render_normals_image():
    from .visualize_normals import render_normal_visualization
    return render_normal_visualization(
        MESH_PATH, view="oblique_1", width=1024, height=1024,
        radar_config_path=RADAR_CFG, bg_color=_BG_RGBA, fov=_THUMB_FOV,
    )


def _render_materials_image():
    from .visualize_materials import render_material_visualization
    return render_material_visualization(
        MESH_PATH, TRAIN_DIR, view="oblique_1",
        param_name="overall_change", colormap="inferno",
        width=1024, height=1024, radar_config_path=RADAR_CFG,
        bg_color=_BG_RGBA, fov=_THUMB_FOV,
    )


def _render_sensor_pose_image():
    from .utils.visualization_open3d import (
        _setup_renderer, parse_radar_config,
        create_radar_visualization_geometries, VIEWPOINTS,
    )
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(MESH_PATH)
    mesh.compute_vertex_normals()
    n = len(mesh.vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((n, 3), 0.78))

    renderer = _setup_renderer(1024, 1024)
    renderer.scene.set_background(_BG_RGBA)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    center, boresight = parse_radar_config(RADAR_CFG)
    if center is not None:
        sphere, arrow = create_radar_visualization_geometries(
            center, boresight, sphere_radius=0.2, arrow_length=2.5)
        if sphere is not None:
            ms = o3d.visualization.rendering.MaterialRecord()
            ms.shader = "defaultLit"
            ms.base_color = [1.0, 0.0, 0.0, 1.0]
            renderer.scene.add_geometry("s", sphere, ms)
        if arrow is not None:
            ma = o3d.visualization.rendering.MaterialRecord()
            ma.shader = "defaultLit"
            ma.base_color = [0.0, 0.8, 0.0, 1.0]
            renderer.scene.add_geometry("a", arrow, ma)

        # Azimuth & elevation orientation arrows (shorter, distinct colors)
        b = np.asarray(boresight, dtype=float)
        b = b / np.linalg.norm(b)
        up = np.array([0.0, 0.0, 1.0])
        # Azimuth axis: horizontal, perpendicular to boresight
        az_axis = np.cross(b, up)
        az_norm = np.linalg.norm(az_axis)
        if az_norm > 1e-6:
            az_axis /= az_norm
        else:
            az_axis = np.array([1.0, 0.0, 0.0])
        # Elevation axis: perpendicular to both boresight and azimuth
        el_axis = np.cross(az_axis, b)
        el_axis /= np.linalg.norm(el_axis)

        orient_len = 1.5
        for axis_vec, color, name in [
            (az_axis, [1.0, 0.55, 0.0, 1.0], "az"),   # orange
            (el_axis, [0.2, 0.5, 1.0, 1.0], "el"),     # blue
        ]:
            _, orient_arrow = create_radar_visualization_geometries(
                center, axis_vec, sphere_radius=0.15, arrow_length=orient_len)
            if orient_arrow is not None:
                mo = o3d.visualization.rendering.MaterialRecord()
                mo.shader = "defaultLit"
                mo.base_color = color
                renderer.scene.add_geometry(f"orient_{name}", orient_arrow, mo)

    bbox = mesh.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()
    ext = bbox.get_extent()
    max_ext = float(np.max(ext))
    bbox_max = bbox.get_max_bound()

    vp = (45, 30, 1.5)
    for name, az, el, ds in VIEWPOINTS:
        if "oblique_1" in name:
            vp = (az, el, ds)
            break
    az_r, el_r = np.deg2rad(vp[0]), np.deg2rad(vp[1])
    dist = max_ext * vp[2]
    cam = lookat + np.array([
        dist * np.cos(el_r) * np.sin(az_r),
        dist * np.cos(el_r) * np.cos(az_r),
        dist * np.sin(el_r),
    ])
    min_z = bbox_max[2] + ext[2] * 0.2
    if cam[2] < min_z:
        rz = min_z - lookat[2]
        rxy = rz / np.tan(el_r) if el_r > 0 else dist
        cam = np.array([lookat[0] + rxy * np.sin(az_r),
                        lookat[1] + rxy * np.cos(az_r), min_z])
    renderer.setup_camera(_THUMB_FOV, lookat, cam, np.array([0, 0, 1]))
    img = renderer.render_to_image()
    arr = np.asarray(img)
    del renderer
    return Image.fromarray(arr)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Differentiable Parameters (unified single block)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_thumbnail(fig, ax, render_fn, fallback_text="Image"):
    try:
        img = render_fn()
        arr = np.array(img)
        # Crop away white/near-white borders so the mesh fills the axes
        if arr.ndim == 3:
            mask = np.any(arr[:, :, :3] < 245, axis=2)
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            if rows.any() and cols.any():
                r0, r1 = np.where(rows)[0][[0, -1]]
                c0, c1 = np.where(cols)[0][[0, -1]]
                arr = arr[r0:r1 + 1, c0:c1 + 1]
        ax.imshow(arr)
    except Exception as e:
        print(f"  Thumbnail render failed ({e})")
        ax.text(0.5, 0.5, fallback_text, ha="center", va="center",
                transform=ax.transAxes, fontsize=5, color="#999")
    ax.axis("off")


def _draw_beam_pattern_thumbnail(fig, ax):
    from .visualize_antenna_patterns import draw_beam_pattern_thumbnail

    ax.set_visible(False)
    pos = ax.get_position()
    polar_ax = fig.add_axes(pos, projection="polar")
    polar_ax.set_facecolor("#ffffff")

    if os.path.exists(TX_PATTERN_PATH) and os.path.exists(RX_PATTERN_PATH):
        draw_beam_pattern_thumbnail(polar_ax, TX_PATTERN_PATH, RX_PATTERN_PATH)


def _draw_section1(fig):
    """Section 1: 3DPS Scene Representation — single tile holding the
    points + normals scene render (replaces mmIR's 4-thumbnail layout)."""
    # Section title at top
    fig.text(_fx(SEC1_L + SEC1_W / 2), _fy(SEC_TOP - HEADER_H / 2),
             "3DPS Scene Representation", ha="center", va="center",
             fontsize=5.5, fontweight="bold", color="#222")

    sp_left = SEC1_L + SEC1_IPAD
    sp_bot  = TILE_BOT                 # column tile shrinks to TILE_H
    sp_h    = TILE_H
    sp_w    = SUBPANEL_W

    _add_rounded_rect(fig, sp_left, sp_bot, sp_w, sp_h,
                      color=TILE_COLOR, zorder=-0.5)

    # Image goes at the TOP of the tile.  Its height is chosen so the
    # caption text sits *equidistantly* between the bottom of the image
    # and the bottom of the column tile.  Image height < SPLAT_H slightly
    # to make room for caption + symmetric gaps.
    caption_text = "Optimised 3DPS points (positions, normals, materials)"
    caption_h    = 0.05
    caption_gap  = 0.04                # gap above / below caption (equal)
    top_pad      = 0.03                # gap above image (matches splat above-AG pad)
    img_pad_x    = top_pad             # empty bands left/right of image = top_pad

    img_left = sp_left + img_pad_x
    img_top  = sp_bot + sp_h - top_pad
    img_h    = sp_h - top_pad - 2 * caption_gap - caption_h
    img_bot  = img_top - img_h

    # Centre the caption between img_bot and sp_bot (equidistant).
    caption_y = sp_bot + (img_bot - sp_bot) / 2
    fig.text(_fx(sp_left + sp_w / 2), _fy(caption_y),
             caption_text,
             ha="center", va="center", fontsize=4, color="#555")

    panel_path = PANEL_A_PATH
    # Square axes width chosen to maintain image aspect ratio (no stretch).
    img_axes_w = sp_w - 2 * img_pad_x  # default fill; corrected below if needed.
    if os.path.exists(panel_path):
        try:
            arr = np.asarray(Image.open(panel_path))
            # Crop only the bottom 1/3 (right side now uncropped so the image
            # has a more landscape aspect that fills the axes more evenly).
            h, _w = arr.shape[:2]
            arr = arr[: int(h * 2 / 3), :, :]
            img_aspect = arr.shape[1] / arr.shape[0]
            # Fit image inside (img_axes_w_max × img_h) while preserving
            # aspect — pick the side that limits.
            img_axes_w = min(sp_w - 2 * img_pad_x, img_h * img_aspect)
            img_left = sp_left + (sp_w - img_axes_w) / 2
            ax = _ax_at(fig, img_left, img_bot, img_axes_w, img_h)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.imshow(arr, aspect="auto", interpolation="bilinear")
        except Exception as e:
            print(f"  panel_a load failed ({e})")
            ax = _ax_at(fig, img_left, img_bot, img_axes_w, img_h)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.5, 0.5, "panel_a missing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=5, color="#999")
    else:
        ax = _ax_at(fig, img_left, img_bot, img_axes_w, img_h)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.5, 0.5, f"panel_a missing\n{panel_path}",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=4, color="#999")

    # Three y-coords used as connector source points for the 3 Section-2
    # tiles (BSDF / Antenna Gain / Phase) — top quarter, middle, bottom
    # quarter of the single Section-1 tile.
    cy_top = sp_bot + sp_h * 0.75
    cy_mid = sp_bot + sp_h * 0.50
    cy_bot = sp_bot + sp_h * 0.25
    return cy_top, cy_mid, cy_bot


# ══════════════════════════════════════════════════════════════════════════════
# Sections 3 & 4: Rendered Prediction / Ground Truth
# ══════════════════════════════════════════════════════════════════════════════

def _crp_ra_img_side(sec_inner_w, plot_pad):
    """Square CRP/RA image side that lets:
      * CRP top hit SPLAT_TOP exactly,
      * RA bottom hit SPLAT_BOT exactly,
      * a visible Azimuth-FFT arrow gap remain between the two images.
    Used by both Sections 3/4 and the Section 2 → Section 3 connector so
    the image positions and the arrow source match.
    """
    arrow_gap = 0.18
    return min(sec_inner_w - 2 * plot_pad, (SPLAT_H - arrow_gap) / 2)


def _plot_adc_signal(ax, adc_complex):
    """Plot ADC I/Q components (matches visualize_iq_data.py style)."""
    s = np.arange(len(adc_complex))
    i_comp = adc_complex.real
    q_comp = adc_complex.imag
    # I and Q in distinct colors (from visualize_iq_data.py)
    ax.plot(s, i_comp, color="#2E86AB", lw=0.5, alpha=0.9, label="I")
    ax.plot(s, q_comp, color="#A23B72", lw=0.5, alpha=0.9, label="Q")
    ax.set_facecolor(TILE_COLOR)
    ax.patch.set_alpha(1)
    ax.set_xlim(0, len(adc_complex) - 1)
    ax.margins(y=0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_prediction_section(fig, sec_left, sec_w, adc_path, ra_path,
                              section_title, adc_title, ra_title, is_gt=False,
                              adc_signal=None):
    """Draw Rendered Prediction or Ground Truth section."""

    # Section title at top
    fig.text(_fx(sec_left + sec_w / 2), _fy(SEC_TOP - HEADER_H / 2),
             section_title, ha="center", va="center",
             fontsize=5.5, fontweight="bold", color="#222")

    # Section column tile bg (same height TILE_H as Section 1 & 2 tiles).
    inner_w = sec_w - 2 * SEC34_IPAD
    inner_left = sec_left + SEC34_IPAD
    _add_rounded_rect(fig, inner_left, TILE_BOT, inner_w, TILE_H,
                      color=TILE_COLOR, zorder=-0.5)

    plot_pad = 0.025

    # CRP and RA image side (square) — same value for all 4 image
    # subplots; lets the images snap CRP-top→SPLAT_TOP and
    # RA-bottom→SPLAT_BOT identically across Sections 3 and 4.
    img_side = _crp_ra_img_side(inner_w, plot_pad)

    # ── CRP image (top edge = SPLAT_TOP) ────────────────────────────────
    adc_ax_top = SPLAT_TOP
    adc_ax_bot = adc_ax_top - img_side
    adc_ax_x   = inner_left + (inner_w - img_side) / 2
    ax_adc = _ax_at(fig, adc_ax_x, adc_ax_bot, img_side, img_side)
    try:
        arr = np.asarray(Image.open(adc_path))
        ax_adc.imshow(arr, aspect="equal", interpolation="bilinear")
    except Exception as e:
        print(f"  CRP load failed ({e})")
        ax_adc.text(0.5, 0.5, "CRP", ha="center", va="center",
                    transform=ax_adc.transAxes, fontsize=5, color="#999")
    ax_adc.set_xticks([]); ax_adc.set_yticks([])
    for sp in ax_adc.spines.values():
        sp.set_visible(False)

    # ── RA image (bottom edge = SPLAT_BOT) ───────────────────────────────
    ra_ax_bot = SPLAT_BOT
    ra_ax_top = ra_ax_bot + img_side
    ra_ax_x   = inner_left + (inner_w - img_side) / 2
    ax_ra = _ax_at(fig, ra_ax_x, ra_ax_bot, img_side, img_side)
    try:
        ra_img = np.asarray(Image.open(ra_path))
        ax_ra.imshow(ra_img, aspect="equal", interpolation="bilinear")
        ax_ra.patch.set_alpha(1)
    except Exception as e:
        print(f"  RA load failed ({e})")
        ax_ra.text(0.5, 0.5, "RA", ha="center", va="center",
                   transform=ax_ra.transAxes, fontsize=5, color="#999")
    ax_ra.set_xticks([]); ax_ra.set_yticks([])
    for sp in ax_ra.spines.values():
        sp.set_visible(False)

    # ── Azimuth FFT bidir vertical arrows + symbol ──────────────────────
    # Arrows span the gap between CRP-image-bottom and RA-image-top.
    # swap_lr=True puts the GREEN arrow on the right side (next to the
    # F_theta symbol) so the symbol annotates the forward path.
    cx = _fx(inner_left + inner_w / 2)
    _draw_bidir_arrows(fig, cx, _fy(adc_ax_bot - 0.003),
                       cx, _fy(ra_ax_top + 0.003), swap_lr=True)
    fft_cy = (adc_ax_bot + ra_ax_top) / 2
    fig.text(cx + _ARROW_XOFF + 0.003, _fy(fft_cy),
             r"$\mathcal{F}_{\theta}$",
             ha="left", va="center", fontsize=5.6, color="#666")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Rendering / Forward Model (center variants)
# ══════════════════════════════════════════════════════════════════════════════

SEC2_TILE_IPAD = 0.06  # inset for inner tile

def _draw_section2_panel(fig):
    """Draw Forward Model with inner tile; title above it."""
    tile_left = SEC2_L + SEC2_TILE_IPAD
    tile_w = SEC2_W - 2 * SEC2_TILE_IPAD
    _add_rounded_rect(fig, tile_left, TILE_BOT, tile_w, TILE_H,
                      color=TILE_COLOR, radius_in=TILE_RADIUS,
                      zorder=-0.5)
    # Title at top (aligned with other section titles)
    fig.text(_fx(SEC2_L + SEC2_W / 2), _fy(SEC_TOP - HEADER_H / 2),
             "Forward Model", ha="center", va="center",
             fontsize=5.5, fontweight="bold", color="#222")


def _draw_center_system_overview(fig):
    """3DPS forward model — 2-column layout:
        col 1: BSDF / Antenna Gain / Phase  (3 squarish tiles, stacked)
        col 2: Splat                         (1 almost-square tile)
    """
    _draw_section2_panel(fig)

    # Section 2 inner area
    inner_left  = SEC2_L + SEC2_TILE_IPAD + 0.03
    inner_right = SEC2_L + SEC2_W - SEC2_TILE_IPAD - 0.03
    inner_bot   = CONTENT_BOT + 0.03
    inner_top   = CONTENT_TOP - 0.03
    inner_w     = inner_right - inner_left
    inner_h     = inner_top - inner_bot

    # Splat is the SAME width as the col1 tiles (BSDF/AG/Phase). col_gap
    # = GAP_REF so the col1↔Splat arrows have the same horizontal extent
    # as the Sec1→Sec2 arrows. col1_w is fixed (not a fraction of inner_w)
    # so the layout is independent of SEC2_W changes.
    col1_w  = 0.65
    col2_w  = col1_w
    col_gap = GAP_REF                     # 0.14 in., matches all other arrow gaps
    col1_x  = inner_left
    col2_x  = inner_left + col1_w + col_gap

    left_blocks = [
        ("Antenna Gain",
         r"$G_{TX}(\theta_i)\,G_{RX}(\theta_o)$",
         "\u00a73.1"),
        ("BSDF",
         r"$f_r(\theta_i,\theta_o,\mathbf{n}_i,\varepsilon_r',\sigma,t)$",
         "\u00a73.2"),
        ("Phase",
         r"$\varphi_i = -2\pi\,(R_i^{TX}+R_i^{RX})/\lambda$",
         "\u00a73.1"),
    ]
    # ── Splat dimensions taken from module-level constants so Sections 1,
    # 3, 4 align to the same Splat top/bottom.
    splat_h   = SPLAT_H
    splat_top = SPLAT_TOP
    splat_bot = SPLAT_BOT

    n_left  = len(left_blocks)
    row_gap = 0.02                        # minimal vertical gap
    stack_top = splat_top                 # AG top   = Splat top
    stack_h   = splat_h                   # Phase bot = Splat bot
    block_h   = (stack_h - (n_left - 1) * row_gap) / n_left

    block_info = []

    # Left column: 3 stacked tiles (no § section refs)
    for i, (title, equation, _section) in enumerate(left_blocks):
        bot = stack_top - (i + 1) * block_h - i * row_gap
        block_info.append({
            "cy": bot + block_h / 2,
            "left": col1_x, "right": col1_x + col1_w,
            "bot": bot, "top": bot + block_h,
        })
        _add_rounded_rect(fig, col1_x, bot, col1_w, block_h,
                          color=BLOCK_COLOR, radius_in=TILE_RADIUS,
                          edgecolor="none", linewidth=0, zorder=-0.3)
        fig.text(_fx(col1_x + col1_w / 2), _fy(bot + block_h * 0.70),
                 title, ha="center", va="center",
                 fontsize=4.6, fontweight="bold", color="#333")
        eq_is_math = equation.startswith("$")
        fig.text(_fx(col1_x + col1_w / 2), _fy(bot + block_h * 0.30),
                 equation, ha="center", va="center",
                 fontsize=4.0 if eq_is_math else 3.3,
                 fontweight="normal" if eq_is_math else "bold",
                 color="#444" if eq_is_math else "#555")

    # Right column: Splat (dimensions set above; left UNCHANGED)
    _add_rounded_rect(fig, col2_x, splat_bot, col2_w, splat_h,
                      color=BLOCK_COLOR, radius_in=TILE_RADIUS,
                      edgecolor="none", linewidth=0, zorder=-0.3)
    fig.text(_fx(col2_x + col2_w / 2), _fy(splat_bot + splat_h * 0.84),
             "Splat", ha="center", va="center",
             fontsize=5.0, fontweight="bold", color="#333")
    fig.text(_fx(col2_x + col2_w / 2), _fy(splat_bot + splat_h * 0.55),
             r"$z_i = \sqrt{G_{TX}G_{RX}}\,f_r\,e^{\,j\varphi_i}/R_i^{2}$",
             ha="center", va="center",
             fontsize=4.0, color="#444")
    fig.text(_fx(col2_x + col2_w / 2), _fy(splat_bot + splat_h * 0.36),
             r"$\to$ bin $k_i = \mathrm{round}(R_i/\Delta r)$",
             ha="center", va="center",
             fontsize=4.0, color="#444")
    # Section ref "(§3.3)" removed.
    block_info.append({
        "cy": splat_bot + splat_h / 2,
        "left": col2_x, "right": col2_x + col2_w,
        "bot": splat_bot, "top": splat_bot + splat_h,
    })

    # Three perfectly horizontal arrows from col 1 tiles → Splat.  Each
    # exits its source tile at the tile's vertical centre and enters the
    # Splat tile at the SAME y so the arrow is straight (no S-shape).
    cx_right_frac = _fx(col2_x - 0.005)
    bidir_flags = [False, True, False]    # AG, BSDF, Phase
    for blk, bidir in zip(block_info[:3], bidir_flags):
        cx_left_frac = _fx(blk["right"] + 0.005)
        cy = _fy(blk["cy"])
        _draw_bidir_arrows(fig, cx_left_frac, cy,
                            cx_right_frac, cy, bidir=bidir)

    return block_info


def _draw_center_path_decomposition(fig):
    _draw_section2_panel(fig)
    schem_h = CONTENT_H * 0.27
    schem_w = SEC2_W * 0.88
    schem_x = SEC2_L + (SEC2_W - schem_w) / 2
    schem_gap = CONTENT_H * 0.04

    schematics = [
        ("Specular (SMS)", "#c62828", "specular"),
        ("Diffuse (MC)", "#1565c0", "diffuse"),
        ("Diffraction (FSD)", "#2e7d32", "diffraction"),
    ]
    for i, (label, color, ptype) in enumerate(schematics):
        bot = CONTENT_TOP - 0.08 - (i + 1) * schem_h - i * schem_gap
        ax = _ax_at(fig, schem_x, bot, schem_w, schem_h)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.set_facecolor(TILE_COLOR)
        ax.patch.set_alpha(1)

        if ptype == "specular":
            ax.plot([0.5, 9.5], [1, 1], color="#555", lw=1.2)
            ax.annotate("", xy=(5, 1), xytext=(1, 3.5),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
            ax.annotate("", xy=(9, 3.5), xytext=(5, 1),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
            ax.plot([5, 5], [1, 2.5], color="#aaa", lw=0.3, ls=":")
            ax.text(0.5, 3.5, "TX", fontsize=3.5, color=color, fontweight="bold")
            ax.text(8.5, 3.5, "RX", fontsize=3.5, color=color, fontweight="bold")
        elif ptype == "diffuse":
            xs = np.linspace(0.5, 9.5, 80)
            ys = 1 + 0.15 * np.sin(xs * 4) + 0.1 * np.sin(xs * 7)
            ax.plot(xs, ys, color="#555", lw=0.8)
            ax.annotate("", xy=(5, 1.2), xytext=(1, 3.5),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
            for a in [-25, -5, 15, 35]:
                dx = 2.5 * np.cos(np.deg2rad(90 - a))
                dy = 2.5 * np.sin(np.deg2rad(90 - a))
                ax.annotate("", xy=(5 + dx, 1.2 + dy), xytext=(5, 1.2),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.4, alpha=0.5))
            ax.text(0.5, 3.5, "TX", fontsize=3.5, color=color, fontweight="bold")
        else:
            ax.plot([1, 5], [1, 1], color="#555", lw=1.2)
            ax.plot([5, 5], [1, 2.5], color="#555", lw=1.2)
            ax.annotate("", xy=(5, 2), xytext=(1, 3.5),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
            ax.annotate("", xy=(9, 3), xytext=(5, 2),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8))
            ax.text(0.5, 3.5, "TX", fontsize=3.5, color=color, fontweight="bold")
            ax.text(8.5, 3.2, "RX", fontsize=3.5, color=color, fontweight="bold")

        ax.text(5, 0.15, label, ha="center", va="bottom",
                fontsize=4.5, fontweight="bold", color=color)
        ax.axis("off")
    return None


def _draw_center_mesh_rays(fig, color_by_type=False):
    _draw_section2_panel(fig)
    pad = 0.06
    inner_h = CONTENT_H - 0.06
    ax = _ax_at(fig, SEC2_L + pad, CONTENT_BOT + pad,
                SEC2_W - 2 * pad, inner_h - pad)
    if os.path.exists(MESH_VIEW_PATH):
        mesh_img = _load_and_crop_image(MESH_VIEW_PATH)
        ax.imshow(mesh_img)
        h, w = mesh_img.shape[:2]
        tx = (int(w * 0.15), int(h * 0.45))
        hits = [(int(w * 0.45), int(h * 0.55)),
                (int(w * 0.60), int(h * 0.40)),
                (int(w * 0.75), int(h * 0.60))]
        rx = (int(w * 0.18), int(h * 0.48))
        colors = (["#c62828", "#1565c0", "#2e7d32"] if color_by_type
                  else ["#e65100"] * 3)
        labels = (["Specular", "Diffuse", "Diffraction"] if color_by_type
                  else [None] * 3)
        for (hx, hy), c, lbl in zip(hits, colors, labels):
            ax.annotate("", xy=(hx, hy), xytext=tx,
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7, alpha=0.75))
            ax.annotate("", xy=rx, xytext=(hx, hy),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7,
                                        alpha=0.75, ls="--"))
            if lbl:
                ax.text(hx + 5, hy - 10, lbl, fontsize=3, color=c, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Mesh + Rays", ha="center", va="center",
                transform=ax.transAxes, fontsize=5, color="#999")
    ax.axis("off")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Inter-section connector arrows
# ══════════════════════════════════════════════════════════════════════════════

def _draw_connectors(fig, sp_top_cy, sp_mid_cy, sp_bot_cy, block_info=None):
    """Draw all inter-section connector arrows.  All arrows are PERFECTLY
    HORIZONTAL — source y = target y — except for the Azimuth-FFT arrows
    drawn inside each prediction section (which are vertical by design).

    sp_top_cy / sp_mid_cy / sp_bot_cy : kept for backward-compat; ignored
        in this layout (block-cy values are used as both source and target
        y so the arrows are guaranteed horizontal).
    block_info : list of dicts (cy, left, right, bot, top) — order
        [Antenna Gain, BSDF, Phase, Splat].
    """

    # ── Section 1 → Section 2 (3 horizontal bidir/forward-only pairs) ────
    sec1_tile_right = SEC1_L + SEC1_IPAD + SUBPANEL_W
    x_from = _fx(sec1_tile_right) + 0.005

    if block_info is not None and len(block_info) >= 3:
        bidir_flags = [False, True, False]   # AG, BSDF, Phase
        for blk, bidir in zip(block_info[:3], bidir_flags):
            x_to = _fx(blk["left"]) - 0.005
            cy   = _fy(blk["cy"])
            _draw_bidir_arrows(fig, x_from, cy, x_to, cy, bidir=bidir)

    # ── Section 2 (Splat) → Section 3 (CRP image): horizontal ───────────
    # Compute the same image side used in _draw_prediction_section so the
    # arrow target y matches the CRP image's vertical centre exactly.
    sec3_inner_w  = SEC3_W - 2 * SEC34_IPAD
    plot_pad      = 0.025
    img_side      = _crp_ra_img_side(sec3_inner_w, plot_pad)
    crp_cy_in     = SPLAT_TOP - img_side / 2

    # Splat right edge → CRP image left edge (CRP image is centred in
    # Section 3's inner area).
    sec3_inner_left = SEC3_L + SEC34_IPAD
    crp_x_left_in   = sec3_inner_left + (sec3_inner_w - img_side) / 2
    if block_info is not None and len(block_info) >= 1:
        x_from2 = _fx(block_info[-1]["right"]) + 0.005
    else:
        x_from2 = _fx(SEC2_L + SEC2_W) + 0.005
    x_to2 = _fx(crp_x_left_in) - 0.005
    cy_crp = _fy(crp_cy_in)
    _draw_bidir_arrows(fig, x_from2, cy_crp, x_to2, cy_crp, bidir=True)

    # ── RA Loss: Section 3 RA image right ↔ Section 4 RA image left ─────
    ra_cy_in = SPLAT_BOT + img_side / 2

    def _ra_image_x_edges(sec_left, sec_w):
        inner_w_local = sec_w - 2 * SEC34_IPAD
        inner_left_local = sec_left + SEC34_IPAD
        x_left_in  = inner_left_local + (inner_w_local - img_side) / 2
        x_right_in = x_left_in + img_side
        return x_left_in, x_right_in

    _, sec3_ra_right_in = _ra_image_x_edges(SEC3_L, SEC3_W)
    sec4_ra_left_in, _  = _ra_image_x_edges(SEC4_L, SEC4_W)

    ra_cy   = _fy(ra_cy_in)
    x_left  = _fx(sec3_ra_right_in)
    x_right = _fx(sec4_ra_left_in)
    x_mid   = (x_left + x_right) / 2

    PURPLE = "#7b2d8e"
    overlay = _get_overlay_ax(fig)
    overlay.annotate(
        "", xy=(x_right, ra_cy), xytext=(x_left, ra_cy),
        xycoords="figure fraction", textcoords="figure fraction",
        arrowprops=dict(arrowstyle="<|-|>", color=PURPLE, lw=0.6,
                        linestyle=(0, (4, 3)), mutation_scale=5),
    )
    fig.text(x_mid, ra_cy + 0.025, "RA Loss", ha="center", va="bottom",
             fontsize=4, fontweight="bold", color="#666")


# ══════════════════════════════════════════════════════════════════════════════
# Main figure assembly
# ══════════════════════════════════════════════════════════════════════════════

def generate_figure(output_dir, center_variant="system_overview",
                    train_dir=None, postproc_dir=None):
    if train_dir is not None or postproc_dir is not None:
        _update_data_paths(train_dir=train_dir, postproc_dir=postproc_dir)
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("none")
    # Background row tile inset by MARGIN_OUT so its width = FIG_W - 2*MARGIN_OUT
    # matches the teaser row-tile width (= 6.98").
    _add_rounded_rect(fig, MARGIN_OUT, MARGIN_OUT,
                       FIG_W - 2 * MARGIN_OUT, FIG_H - 2 * MARGIN_OUT,
                       color=BG_COLOR, radius_in=SECTION_RADIUS, zorder=-2)

    sp_top_cy, sp_mid_cy, sp_bot_cy = _draw_section1(fig)

    # Pre-process ADC: match rendered high-freq scale to GT + add GT low-freq
    rendered_adc_sig = None
    try:
        gt_raw = np.load(GT_ADC_PATH)
        gt_sig = gt_raw[0, 0, 0, :]
        gt_lowfreq = _lowpass_envelope(gt_sig, cutoff_frac=0.05, order=3)
        gt_highfreq = gt_sig - gt_lowfreq

        rend_raw = np.load(RENDERED_ADC_PATH)
        rend_sig = rend_raw[0, 0, :, 0] + 1j * rend_raw[0, 0, :, 1]
        rend_lowfreq = _lowpass_envelope(rend_sig, cutoff_frac=0.05, order=3)
        rend_highfreq = rend_sig - rend_lowfreq

        # Scale rendered high-freq to match GT high-freq RMS amplitude
        gt_hf_rms = np.sqrt(np.mean(np.abs(gt_highfreq) ** 2))
        rend_hf_rms = np.sqrt(np.mean(np.abs(rend_highfreq) ** 2))
        scale = gt_hf_rms / max(rend_hf_rms, 1e-30)
        rendered_adc_sig = gt_lowfreq + rend_highfreq * scale
    except Exception as e:
        print(f"  ADC LPF pre-processing failed ({e}), using raw signals")

    _draw_prediction_section(
        fig, SEC3_L, SEC3_W, RENDERED_CRP_PATH, RENDERED_RA_PATH,
        "Rendered", "Rendered CRP", "Rendered RA", is_gt=False,
        adc_signal=None)
    _draw_prediction_section(
        fig, SEC4_L, SEC4_W, GT_CRP_PATH, GT_RA_PATH,
        "Ground Truth", "GT CRP", "GT RA", is_gt=True)

    block_info = None
    if center_variant == "system_overview":
        block_info = _draw_center_system_overview(fig)
    elif center_variant == "path_decomposition":
        _draw_center_path_decomposition(fig)
    elif center_variant == "mesh_with_rays":
        _draw_center_mesh_rays(fig, color_by_type=False)
    elif center_variant == "mesh_path_types":
        _draw_center_mesh_rays(fig, color_by_type=True)
    else:
        raise ValueError(f"Unknown center_variant: {center_variant}")

    _draw_connectors(fig, sp_top_cy, sp_mid_cy, sp_bot_cy, block_info)

    base = f"pipeline_3dps_{center_variant}"
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"{base}.{ext}")
        fig.savefig(path, dpi=300,
                    facecolor="none" if ext == "pdf" else BG_COLOR,
                    edgecolor="none")
        print(f"  Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline figure (Figure 2)")
    parser.add_argument("--output_dir", default="output/postprocess_final_v11/figures")
    parser.add_argument(
        "--center_variant", default="all",
        choices=["system_overview", "path_decomposition",
                 "mesh_with_rays", "mesh_path_types", "all"],
    )
    parser.add_argument("--train_dir", default=None,
                        help="Training output root (e.g. output/train_v9)")
    parser.add_argument("--postproc_dir", default=None,
                        help="Postprocessing output root (e.g. output/postprocess_final_v9)")
    args = parser.parse_args()
    variants = (
        ["system_overview", "path_decomposition", "mesh_with_rays", "mesh_path_types"]
        if args.center_variant == "all"
        else [args.center_variant]
    )
    for v in variants:
        print(f"\n{'=' * 60}\nGenerating: {v}\n{'=' * 60}")
        generate_figure(args.output_dir, center_variant=v,
                        train_dir=args.train_dir, postproc_dir=args.postproc_dir)
    print(f"\nDone! Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
