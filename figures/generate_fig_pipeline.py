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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from .fig_common import BACKGROUND_COLOR, FIG_WIDTH_INCHES, add_rounded_bg


# ══════════════════════════════════════════════════════════════════════════════
# Layout constants (all in inches)
# ══════════════════════════════════════════════════════════════════════════════

FIG_W = FIG_WIDTH_INCHES  # 7.1
FIG_H = 2.45             # compact — matches SVG reference aspect ratio

# Section widths (Sec3/4 narrower for square-ish inner tiles)
SEC1_W = 2.15
SEC2_W = 2.10
SEC3_W = 1.05
SEC4_W = 1.05

# Gaps between sections
GAP_12 = 0.14
GAP_23 = 0.25
GAP_34 = 0.25    # wider for RA Loss label

MARGIN = 0.04

# Section x-positions
SEC1_L = MARGIN
SEC2_L = SEC1_L + SEC1_W + GAP_12
SEC3_L = SEC2_L + SEC2_W + GAP_23
SEC4_L = SEC3_L + SEC3_W + GAP_34

# Vertical layout
SEC_PAD = 0.02
SEC_TOP = FIG_H - SEC_PAD
SEC_BOT = SEC_PAD
SEC_H = SEC_TOP - SEC_BOT

HEADER_H = 0.16
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

# ── Sections 3/4: ADC + FFT + RA proportions (RA smaller) ───────────────────
SEC34_IPAD = 0.03
ADC_FRAC = 0.40
RA_FRAC = 0.42
FFT_FRAC = 1.0 - ADC_FRAC - RA_FRAC  # 0.18


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

def _draw_bidir_arrows(fig, x1, y1, x2, y2):
    """Draw paired green→ and red← dashed arrows.

    Convention (equal physical gap on both axes):
      Vertical:   green LEFT  (x − _ARROW_XOFF), red RIGHT (x + _ARROW_XOFF)
      Horizontal: green TOP   (y + _ARROW_YOFF), red BOTTOM (y − _ARROW_YOFF)
    """
    overlay = _get_overlay_ax(fig)
    is_vertical = abs(y2 - y1) > abs(x2 - x1)
    style = dict(lw=0.6, linestyle=(0, (4, 3)), mutation_scale=5)

    if is_vertical:
        overlay.annotate(
            "", xy=(x2 - _ARROW_XOFF, y2), xytext=(x1 - _ARROW_XOFF, y1),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="#2ca25f", **style))
        overlay.annotate(
            "", xy=(x1 + _ARROW_XOFF, y1), xytext=(x2 + _ARROW_XOFF, y2),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="#d73027", **style))
    else:
        overlay.annotate(
            "", xy=(x2, y2 + _ARROW_YOFF), xytext=(x1, y1 + _ARROW_YOFF),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=dict(arrowstyle="-|>", color="#2ca25f", **style))
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


def _draw_bidir_s_arrows(fig, x1, y1, x2, y2):
    """Draw paired green/red S-shaped orthogonal arrows (H→V→H segments).

    Convention (equal physical gap, matches _draw_bidir_arrows):
      Horizontal segments: green TOP (+_ARROW_YOFF), red BOTTOM (−_ARROW_YOFF)
      Vertical segment:    green LEFT (−_ARROW_XOFF), red RIGHT (+_ARROW_XOFF)
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
    """Section 1: Differentiable Parameters — two inner tiles, no column panel."""
    # Section title at top
    fig.text(_fx(SEC1_L + SEC1_W / 2), _fy(SEC_TOP - HEADER_H / 2),
             "Differentiable Parameters", ha="center", va="center",
             fontsize=5.5, fontweight="bold", color="#222")

    sp_left = SEC1_L + SEC1_IPAD

    # Consistent column widths for BOTH tiles (so left/right columns align)
    col1_left = sp_left + SP_PAD
    col2_left = col1_left + THUMB_W + THUMB_GAP

    # Title inset from tile top — matches Sec3/4 tiles (0.055")
    _TITLE_INSET = 0.055
    # Caption inset from tile bottom — matches Sec3/4 (plot_pad=0.025 + caption_h*0.45=0.036)
    _CAPTION_INSET = 0.061

    # Vertical content zone within each subpanel (between title and caption)
    _IMG_SCALE = 0.90  # 90% of THUMB size
    def _content_zone(sp_bot):
        zone_top = sp_bot + SUBPANEL_H - _TITLE_INSET - 0.03  # below title text
        zone_bot = sp_bot + _CAPTION_INSET + 0.09             # above caption text (+0.05 raise)
        return zone_bot, zone_top

    # ── Inner tile A: Radar Parameters (top half) ─────────────────────────
    sp_a_bot = CONTENT_TOP - SUBPANEL_H
    _add_rounded_rect(fig, sp_left, sp_a_bot, SUBPANEL_W, SUBPANEL_H,
                      color=TILE_COLOR, zorder=-0.5)
    fig.text(_fx(sp_left + SUBPANEL_W / 2),
             _fy(sp_a_bot + SUBPANEL_H - _TITLE_INSET),
             "Radar Parameters", ha="center", va="center",
             fontsize=5, fontweight="bold", color="#444")

    z_bot, z_top = _content_zone(sp_a_bot)
    zone_h = z_top - z_bot
    # Center scaled image in the zone
    img_w = THUMB_W * _IMG_SCALE
    img_h = THUMB_H * _IMG_SCALE
    img_bot = z_bot + (zone_h - img_h) / 2

    ax_pose = _ax_at(fig, col1_left + (THUMB_W - img_w) / 2, img_bot, img_w, img_h)
    _draw_thumbnail(fig, ax_pose, _render_sensor_pose_image, "Sensor Pose")
    fig.text(_fx(col1_left + THUMB_W / 2), _fy(sp_a_bot + _CAPTION_INSET),
             "Sensor Pose", ha="center", va="center", fontsize=4, color="#555")

    # Beam pattern: 75% of scaled THUMB size, centered in the same zone
    beam_scale = 0.75
    beam_w = img_w * beam_scale
    beam_h = img_h * beam_scale
    beam_left = col2_left + (THUMB_W - beam_w) / 2
    beam_bot = z_bot + (zone_h - beam_h) / 2
    ax_beam = _ax_at(fig, beam_left, beam_bot, beam_w, beam_h)
    _draw_beam_pattern_thumbnail(fig, ax_beam)
    fig.text(_fx(col2_left + THUMB_W / 2), _fy(sp_a_bot + _CAPTION_INSET),
             "Beam Patterns", ha="center", va="center", fontsize=4, color="#555")

    # ── Inner tile B: Scene Parameters (bottom half) ──────────────────────
    sp_b_bot = CONTENT_BOT
    _add_rounded_rect(fig, sp_left, sp_b_bot, SUBPANEL_W, SUBPANEL_H,
                      color=TILE_COLOR, zorder=-0.5)
    fig.text(_fx(sp_left + SUBPANEL_W / 2),
             _fy(sp_b_bot + SUBPANEL_H - _TITLE_INSET),
             "Scene Parameters", ha="center", va="center",
             fontsize=5, fontweight="bold", color="#444")

    z_bot_b, z_top_b = _content_zone(sp_b_bot)
    zone_h_b = z_top_b - z_bot_b
    img_bot_b = z_bot_b + (zone_h_b - img_h) / 2

    ax_norm = _ax_at(fig, col1_left + (THUMB_W - img_w) / 2, img_bot_b, img_w, img_h)
    _draw_thumbnail(fig, ax_norm, _render_normals_image, "Pos. & Normals")
    fig.text(_fx(col1_left + THUMB_W / 2), _fy(sp_b_bot + _CAPTION_INSET),
             "Positions & Normals", ha="center", va="center",
             fontsize=4, color="#555")

    ax_mat = _ax_at(fig, col2_left + (THUMB_W - img_w) / 2, img_bot_b, img_w, img_h)
    _draw_thumbnail(fig, ax_mat, _render_materials_image, "Materials")
    fig.text(_fx(col2_left + THUMB_W / 2), _fy(sp_b_bot + _CAPTION_INSET),
             "Materials", ha="center", va="center", fontsize=4, color="#555")

    return sp_a_bot + SUBPANEL_H / 2, sp_b_bot + SUBPANEL_H / 2


# ══════════════════════════════════════════════════════════════════════════════
# Sections 3 & 4: Rendered Prediction / Ground Truth
# ══════════════════════════════════════════════════════════════════════════════

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
    # No column panel — inner tiles sit directly on background

    # Section title at top
    fig.text(_fx(sec_left + sec_w / 2), _fy(SEC_TOP - HEADER_H / 2),
             section_title, ha="center", va="center",
             fontsize=5.5, fontweight="bold", color="#222")

    inner_w = sec_w - 2 * SEC34_IPAD
    inner_left = sec_left + SEC34_IPAD
    plot_pad = 0.025

    # ── ADC sub-panel ────────────────────────────────────────────────────
    adc_h = CONTENT_H * ADC_FRAC
    adc_bot = CONTENT_TOP - adc_h
    _add_rounded_rect(fig, inner_left, adc_bot, inner_w, adc_h,
                      color=TILE_COLOR, zorder=-0.5)

    fig.text(_fx(inner_left + inner_w / 2), _fy(adc_bot + adc_h - 0.055),
             adc_title, ha="center", va="center",
             fontsize=4, fontweight="bold", color="#555")

    caption_h = 0.08
    adc_ax_bot = adc_bot + plot_pad + caption_h
    adc_ax_h = adc_h - 2 * plot_pad - 0.08 - caption_h
    ax_adc = _ax_at(fig, inner_left + plot_pad, adc_ax_bot,
                    inner_w - 2 * plot_pad, adc_ax_h)
    try:
        if adc_signal is not None:
            sig = adc_signal
        else:
            adc_raw = np.load(adc_path)
            if is_gt:
                sig = adc_raw[0, 0, 0, :]
            else:
                sig = adc_raw[0, 0, :, 0] + 1j * adc_raw[0, 0, :, 1]
        _plot_adc_signal(ax_adc, sig)
    except Exception as e:
        print(f"  ADC load failed ({e})")
        ax_adc.text(0.5, 0.5, "ADC", ha="center", va="center",
                    transform=ax_adc.transAxes, fontsize=5, color="#999")
        ax_adc.set_xticks([])
        ax_adc.set_yticks([])

    fig.text(_fx(inner_left + inner_w / 2),
             _fy(adc_bot + plot_pad + caption_h * 0.45),
             "Complex-valued ADC", ha="center", va="center",
             fontsize=3.5, color="#666")

    # ── 2D FFT label + bidir arrows ─────────────────────────────────────
    ra_h = CONTENT_H * RA_FRAC
    ra_top_y = CONTENT_BOT + ra_h
    fft_cy = (adc_bot + ra_top_y) / 2
    cx = _fx(inner_left + inner_w / 2)

    # Bidir green/red arrows: ADC bottom → RA top (vertical)
    _draw_bidir_arrows(fig, cx, _fy(adc_bot - 0.005),
                       cx, _fy(ra_top_y + 0.005))
    # Text to the RIGHT of vertical arrows (tight to red arrow)
    fig.text(cx + _ARROW_XOFF + 0.003, _fy(fft_cy), "2D FFT", ha="left", va="center",
             fontsize=4, fontweight="bold", color="#666")

    # ── RA sub-panel ─────────────────────────────────────────────────────
    ra_bot = CONTENT_BOT
    _add_rounded_rect(fig, inner_left, ra_bot, inner_w, ra_h,
                      color=TILE_COLOR, zorder=-0.5)

    fig.text(_fx(inner_left + inner_w / 2), _fy(ra_bot + ra_h - 0.055),
             ra_title, ha="center", va="center",
             fontsize=4, fontweight="bold", color="#555")

    ra_ax_bot = ra_bot + plot_pad + caption_h
    ra_ax_h = ra_h - 2 * plot_pad - 0.10 - caption_h
    ax_ra = _ax_at(fig, inner_left + plot_pad, ra_ax_bot,
                   inner_w - 2 * plot_pad, ra_ax_h)
    try:
        ra = np.load(ra_path)
        # Linear min-max normalization (matches training_single_v2)
        ra_lin = _ra_cart_to_linear(ra)
        ax_ra.imshow(ra_lin, cmap="hot", aspect="equal", origin="lower",
                     vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax_ra.patch.set_alpha(1)
    except Exception as e:
        print(f"  RA load failed ({e})")
        ax_ra.text(0.5, 0.5, "RA", ha="center", va="center",
                   transform=ax_ra.transAxes, fontsize=5, color="#999")
    ax_ra.set_xticks([])
    ax_ra.set_yticks([])
    for sp in ax_ra.spines.values():
        sp.set_visible(False)

    fig.text(_fx(inner_left + inner_w / 2),
             _fy(ra_bot + plot_pad + caption_h * 0.45),
             "Magnitude-only RA", ha="center", va="center",
             fontsize=3.5, color="#666")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Rendering / Forward Model (center variants)
# ══════════════════════════════════════════════════════════════════════════════

SEC2_TILE_IPAD = 0.06  # inset for inner tile

def _draw_section2_panel(fig):
    """Draw Forward Model with inner tile; title above it."""
    # Inner tile (same TILE_COLOR as all other inner tiles)
    tile_left = SEC2_L + SEC2_TILE_IPAD
    tile_w = SEC2_W - 2 * SEC2_TILE_IPAD
    _add_rounded_rect(fig, tile_left, CONTENT_BOT, tile_w,
                      CONTENT_TOP - CONTENT_BOT,
                      color=TILE_COLOR, radius_in=TILE_RADIUS,
                      zorder=-0.5)
    # Title at top (aligned with other section titles)
    fig.text(_fx(SEC2_L + SEC2_W / 2), _fy(SEC_TOP - HEADER_H / 2),
             "Forward Model", ha="center", va="center",
             fontsize=5.5, fontweight="bold", color="#222")


def _draw_center_system_overview(fig):
    """Block diagram: MIMO Ray Tracing → mmWave BSDF → Ray Generation → MIMO ADC."""
    _draw_section2_panel(fig)

    # 4 compact blocks — equal spacing (top pad = gap = bottom pad)
    block_w = SEC2_W * 0.75
    block_x = SEC2_L + (SEC2_W - block_w) / 2

    # Each block: (title, equation/subtitle, section_ref, color)
    blocks = [
        ("MIMO Ray Tracing",
         r"$\hat{P}_r^{(D)} = \frac{P_t \lambda^2}{(4\pi)^2 N} \sum_{k=1}^{N} \frac{G_{r,k}}{\rho_{rx,k}} \prod_{\ell=1}^{D-1} \frac{f_\ell \mathcal{G}_\ell}{\rho_\ell} \cdot \frac{G_{t,k} \, c_{D \rightarrow tx} \, f_D}{d_{D,tx}^2} V_k$",
         "\u00a73.1", BLOCK_COLOR),
        ("mmWave BSDF",
         r"$f = A\,[\,\eta\, f_{\rm coh} + (1-\eta)\, f_{\rm inc}\,]$",
         "\u00a73.2", BLOCK_COLOR),
        ("Ray Generation",
         "Reservoir \u00b7 Specular Manifold \u00b7 Diffraction",
         "\u00a73.3", BLOCK_COLOR),
        ("MIMO Coherent ADC",
         r"$s[k] = \sum_p a_p \, e^{\,j2\pi(f_c + Sk)\tau_p}$",
         "\u00a73.1", BLOCK_COLOR),
    ]

    n_blocks = len(blocks)
    block_h = CONTENT_H * 0.20
    pad = 0.03  # minimal top/bottom padding from tile edge
    # Inter-block gaps fill remaining space after blocks + top/bottom pads
    spacing = (CONTENT_H - n_blocks * block_h - 2 * pad) / (n_blocks - 1)

    block_info = []
    for i, (title, equation, section, color) in enumerate(blocks):
        bot = CONTENT_TOP - pad - (i + 1) * block_h - i * spacing
        block_info.append({
            "cy": bot + block_h / 2,
            "left": block_x,
            "right": block_x + block_w,
            "bot": bot,
            "top": bot + block_h,
        })

        # Rounded rect background (no border, circular corners matching tiles)
        _add_rounded_rect(fig, block_x, bot, block_w, block_h,
                          color=color, radius_in=TILE_RADIUS,
                          edgecolor="none", linewidth=0, zorder=-0.3)

        # Title
        fig.text(_fx(block_x + block_w / 2), _fy(bot + block_h * 0.75),
                 title, ha="center", va="center",
                 fontsize=5, fontweight="bold", color="#333")
        # Equation or subtitle line — uniform font size
        eq_is_math = equation.startswith("$")
        fig.text(_fx(block_x + block_w / 2), _fy(bot + block_h * 0.40),
                 equation, ha="center", va="center",
                 fontsize=4.5 if eq_is_math else 3.5,
                 fontweight="normal" if eq_is_math else "bold",
                 color="#444" if eq_is_math else "#555")
        # Section ref
        fig.text(_fx(block_x + block_w / 2), _fy(bot + block_h * 0.12),
                 f"({section})", ha="center", va="center",
                 fontsize=3.5, color="#666", style="italic")

        # Vertical bidirectional arrows between blocks
        if i < len(blocks) - 1:
            cx = _fx(block_x + block_w / 2)
            y_top = _fy(bot - 0.005)
            y_bot = _fy(bot - spacing + 0.005)
            _draw_bidir_arrows(fig, cx, y_top, cx, y_bot)

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

def _draw_connectors(fig, sp_a_cy, sp_b_cy, block_info=None):
    """Draw all inter-section connector arrows using S-shaped orthogonal paths.

    block_info: list of dicts with keys cy, left, right, bot, top (inches)
    """

    # ── Section 1 → Section 2 (S-shaped orthogonal) ──────────────────────
    # Right edge of Sec1 inner tiles → left edge of forward model blocks
    sec1_tile_right = SEC1_L + SEC1_IPAD + SUBPANEL_W
    x_from = _fx(sec1_tile_right) + 0.005

    if block_info is not None and len(block_info) >= 2:
        x_to = _fx(block_info[0]["left"]) - 0.005
        # Radar Params → Ray Generation (S-shaped)
        _draw_bidir_s_arrows(fig, x_from, _fy(sp_a_cy),
                             x_to, _fy(block_info[0]["cy"]))
        # Scene Params → mmWave BSDF (S-shaped)
        x_to1 = _fx(block_info[1]["left"]) - 0.005
        _draw_bidir_s_arrows(fig, x_from, _fy(sp_b_cy),
                             x_to1, _fy(block_info[1]["cy"]))
    else:
        x_to = _fx(SEC2_L) - 0.005
        _draw_bidir_s_arrows(fig, x_from, _fy(sp_a_cy),
                             x_to, _fy(sp_a_cy))
        _draw_bidir_s_arrows(fig, x_from, _fy(sp_b_cy),
                             x_to, _fy(sp_b_cy))

    # ── Section 2 → Section 3 (S-shaped orthogonal) ──────────────────────
    sec3_inner_left = SEC3_L + SEC34_IPAD
    adc_h = CONTENT_H * ADC_FRAC
    adc_cy_in = CONTENT_TOP - adc_h / 2

    # Connect last block (MIMO Coherent ADC) → Rendered ADC
    last_block_idx = len(block_info) - 1 if block_info else 2
    if block_info is not None and len(block_info) >= 1:
        x_from2 = _fx(block_info[last_block_idx]["right"]) + 0.005
    else:
        x_from2 = _fx(SEC2_L + SEC2_W) + 0.005
    x_to2 = _fx(sec3_inner_left) - 0.005

    target_y = _fy(adc_cy_in)
    if block_info is not None and len(block_info) >= 1:
        _draw_bidir_s_arrows(fig, x_from2, _fy(block_info[last_block_idx]["cy"]),
                             x_to2, target_y)
    else:
        _draw_bidir_s_arrows(fig, x_from2, target_y, x_to2, target_y)

    # ── RA Loss: Section 3 ↔ Section 4 ───────────────────────────────────
    ra_h = CONTENT_H * RA_FRAC
    sec3_inner_right = SEC3_L + SEC3_W - SEC34_IPAD
    sec4_inner_left = SEC4_L + SEC34_IPAD
    ra_cy = _fy(CONTENT_BOT + ra_h / 2)
    x_left = _fx(sec3_inner_right) + 0.005
    x_right = _fx(sec4_inner_left) - 0.005
    x_mid = (x_left + x_right) / 2

    PURPLE = "#7b2d8e"
    overlay = _get_overlay_ax(fig)
    overlay.annotate(
        "", xy=(x_right, ra_cy), xytext=(x_left, ra_cy),
        xycoords="figure fraction", textcoords="figure fraction",
        arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=0.7,
                        linestyle=(0, (3, 2)), mutation_scale=6),
    )
    # Text ABOVE horizontal arrow (matches "2D FFT" style)
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
    add_rounded_bg(fig, color=BG_COLOR)

    sp_a_cy, sp_b_cy = _draw_section1(fig)

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
        fig, SEC3_L, SEC3_W, RENDERED_ADC_PATH, RENDERED_RA_PATH,
        "Rendered Prediction", "Rendered ADC", "Rendered RA", is_gt=False,
        adc_signal=rendered_adc_sig)
    _draw_prediction_section(
        fig, SEC4_L, SEC4_W, GT_ADC_PATH, GT_RA_PATH,
        "Ground Truth", "GT ADC", "GT RA", is_gt=True)

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

    _draw_connectors(fig, sp_a_cy, sp_b_cy, block_info)

    base = f"pipeline_{center_variant}"
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
