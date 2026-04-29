#!/usr/bin/env python3
"""Generate the teaser figure (Figure 1) for the paper.

Three-tile layout showing the core story: 2D input → mmIR → 3D output.
  - Tile 1 (left):   Input scene — mesh view + real 2D RA map from commodity radar
  - Tile 2 (center): mmIR inverse rendering — rendered RA + materials + normals
  - Tile 3 (right):  Output — 3D radar point cloud from dense virtual array

Colors match Figure 2 (pipeline overview):
  BG_COLOR   = "#f5f5f5"  (outer background)
  TILE_COLOR = "#ebebeb"  (inner tiles)

Usage:
    python -m mmir.evaluation.generate_fig_teaser \
        --train_dir output/train_v9 \
        --postproc_dir output/postprocess_final_v9 \
        --output_dir output/postprocess_final_v9/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np
from PIL import Image

from .fig_common import (
    FIG_WIDTH_INCHES, CORNER_RADIUS_INCHES,
    add_rounded_bg,
)

# ── Colors matching Figure 2 (pipeline) ──────────────────────────────────────
BG_COLOR = "#f5f5f5"
TILE_COLOR = "#ebebeb"

# ── Scene selection ───────────────────────────────────────────────────────────
SCENE = "seq_1_frame_185"
SCENE_LABEL = "S1-F185"
VIEW_3D = "oblique_4"       # from VIEW_OVERRIDES in generate_fig_3d_occupancy
VIEW_MESH = "oblique_4"     # match the 3D view for visual consistency

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers (from generate_fig_antenna_layouts.py)
# ══════════════════════════════════════════════════════════════════════════════

def _add_rounded_rect(fig, left_in, bot_in, w_in, h_in,
                      color=TILE_COLOR, radius_in=0.06):
    """Add rounded-rectangle patch at inch coordinates."""
    fw, fh = fig.get_size_inches()
    l, b, w, h = left_in / fw, bot_in / fh, w_in / fw, h_in / fh
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
    path = Path(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor=color, edgecolor="none",
        transform=fig.transFigure, zorder=-0.5,
    )
    fig.patches.append(patch)


def _load_and_crop_image(path, border_frac=0.02):
    """Load image and crop away white/light borders."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    mask = np.any(arr < 230, axis=2)
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pr = max(1, int((r1 - r0) * border_frac))
        pc = max(1, int((c1 - c0) * border_frac))
        arr = arr[max(0, r0 - pr):min(arr.shape[0], r1 + pr),
                  max(0, c0 - pc):min(arr.shape[1], c1 + pc)]
    return arr


def _process_3d_render(path):
    """Load 3D render and replace light background with tile color."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    # Replace near-white pixels with tile color
    bg_rgb = tuple(int(TILE_COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    light = np.all(arr > 210, axis=2)
    arr[light] = bg_rgb
    # Crop to content
    mask = ~light
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        border = 15
        r0, r1 = max(0, r0 - border), min(arr.shape[0]-1, r1 + border)
        c0, c1 = max(0, c0 - border), min(arr.shape[1]-1, c1 + border)
        arr = arr[r0:r1+1, c0:c1+1]
    return arr


def _ra_to_linear(ra):
    """Min-max normalize RA to [0,1]."""
    vmin, vmax = ra.min(), ra.max()
    if vmax - vmin < 1e-30:
        return np.zeros_like(ra)
    return (ra - vmin) / (vmax - vmin)


# ══════════════════════════════════════════════════════════════════════════════
# Open3D rendering for material / normal visualization
# ══════════════════════════════════════════════════════════════════════════════

def _render_material_view(mesh_path, training_dir, view, output_path):
    """Render mesh colored by overall material deviation (inferno colormap).

    Uses the same visualize_materials module as the pipeline figure.
    """
    from .visualize_materials import render_material_visualization

    img = render_material_visualization(
        mesh_path, training_dir,
        view=view, param_name="overall_change", colormap="inferno",
        width=2048, height=2048,
    )
    img.save(output_path)
    return output_path


def _render_normal_view(mesh_path, view, output_path):
    """Render mesh colored by surface normals using existing visualize_normals."""
    from .visualize_normals import render_normal_visualization

    bg_rgb = [int(x) / 255.0 for x in [255, 255, 255]]
    img = render_normal_visualization(
        mesh_path, view=view, width=2048, height=2048,
        bg_color=[bg_rgb[0], bg_rgb[1], bg_rgb[2], 1.0],
    )
    img.save(output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════════════════════════

def generate_figure(output_dir, train_dir, postproc_dir):
    os.makedirs(output_dir, exist_ok=True)

    # --- Resolve data paths ---
    scene_train = os.path.join(train_dir, SCENE)
    scene_postproc_3d = os.path.join(postproc_dir, "3d_occupancy", SCENE)
    scene_mesh_renders = os.path.join(postproc_dir, "mesh_only_renders", SCENE)

    # Get mesh path from scene config
    config_path = os.path.join(scene_train, "config.json")
    with open(config_path) as f:
        scene_config = json.load(f)
    mesh_path = scene_config["scene_file"]

    gt_ra_path = os.path.join(scene_train, "ra_gt_cart.npy")
    rendered_ra_path = os.path.join(scene_train, "ra_rendered_cart.npy")
    mesh_view_path = os.path.join(scene_mesh_renders, f"mesh_view_{VIEW_MESH}.png")
    dense_3d_path = os.path.join(
        scene_postproc_3d, "dense_radar", f"dense_radar_view_{VIEW_3D}.png"
    )
    # Pre-render material and normal views
    mat_render_path = os.path.join(output_dir, "teaser_material_view.png")
    norm_render_path = os.path.join(output_dir, "teaser_normal_view.png")
    print("Rendering material visualization...")
    _render_material_view(mesh_path, scene_train, VIEW_MESH, mat_render_path)
    print("Rendering normal visualization...")
    _render_normal_view(mesh_path, VIEW_MESH, norm_render_path)

    # --- Layout (inches) ---
    margin = 0.10
    n_tiles = 3
    title_h = 0.28          # space for title + subtitle
    inner_pad = 0.06        # padding inside tile around images
    caption_h = 0.14        # space for bottom caption
    arrow_zone = 0.48       # wider gap between tiles for arrows + labels

    # Compute tile width with arrow zones
    usable_w = FIG_WIDTH_INCHES - 2 * margin - (n_tiles - 1) * arrow_zone
    tile_w = usable_w / n_tiles
    # Square tiles: tile_h == tile_w
    tile_h = tile_w
    # Image height fills remaining space
    img_h = tile_h - title_h - caption_h - 2 * inner_pad
    fig_h = 2 * margin + tile_h

    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, fig_h))
    add_rounded_bg(fig, color=BG_COLOR)

    tile_configs = [
        {
            "title": "Input: Scene + Real Radar",
            "subtitle": "Cascaded radar (86 virt. elements)",
        },
        {
            "title": "mmIR Inverse Rendering",
            "subtitle": "Fit physics model to real ADC",
        },
        {
            "title": "Output: 3D Radar Imaging",
            "subtitle": "Dense array (10,000 virt. elements)",
        },
    ]

    tile_positions = []
    for i in range(n_tiles):
        tile_left = margin + i * (tile_w + arrow_zone)
        tile_bot = margin
        tile_positions.append((tile_left, tile_bot))

        # Draw tile background
        _add_rounded_rect(fig, tile_left, tile_bot, tile_w, tile_h,
                          color=TILE_COLOR, radius_in=0.06)

        # Title
        tx = (tile_left + tile_w / 2) / FIG_WIDTH_INCHES
        ty_title = (tile_bot + tile_h - inner_pad) / fig_h
        fig.text(tx, ty_title, tile_configs[i]["title"],
                 ha="center", va="top",
                 fontsize=6.5, fontweight="bold", color="#222",
                 transform=fig.transFigure)

        # Subtitle
        ty_sub = (tile_bot + tile_h - inner_pad - 0.13) / fig_h
        fig.text(tx, ty_sub, tile_configs[i]["subtitle"],
                 ha="center", va="top",
                 fontsize=5, color="#555",
                 transform=fig.transFigure)

    # Shared sub-image dimensions (two columns per tile)
    sub_gap = 0.04
    sub_w = (tile_w - 2 * inner_pad - sub_gap) / 2

    # --- Tile 1: Input (mesh view + GT RA side by side) ---
    t1_left, t1_bot = tile_positions[0]
    sub_h = img_h
    ax1_left = t1_left + inner_pad
    ax1_bot = t1_bot + caption_h + inner_pad

    # Mesh view
    ax_mesh = fig.add_axes([
        ax1_left / FIG_WIDTH_INCHES, ax1_bot / fig_h,
        sub_w / FIG_WIDTH_INCHES, sub_h / fig_h,
    ])
    if os.path.exists(mesh_view_path):
        mesh_img = _load_and_crop_image(mesh_view_path)
        ax_mesh.imshow(mesh_img)
    else:
        ax_mesh.text(0.5, 0.5, "Mesh", ha="center", va="center",
                     transform=ax_mesh.transAxes, fontsize=5, color="#999")
    ax_mesh.axis("off")

    # GT RA map
    ax_gt_ra = fig.add_axes([
        (ax1_left + sub_w + sub_gap) / FIG_WIDTH_INCHES, ax1_bot / fig_h,
        sub_w / FIG_WIDTH_INCHES, sub_h / fig_h,
    ])
    if os.path.exists(gt_ra_path):
        gt_ra = np.load(gt_ra_path)
        gt_lin = _ra_to_linear(gt_ra)
        ax_gt_ra.imshow(gt_lin, cmap="hot", aspect="equal", origin="lower",
                        vmin=0.0, vmax=1.0, interpolation="bilinear")
    else:
        ax_gt_ra.text(0.5, 0.5, "GT RA", ha="center", va="center",
                      transform=ax_gt_ra.transAxes, fontsize=5, color="#999")
    ax_gt_ra.axis("off")

    # Captions below images
    cap_y = (t1_bot + caption_h * 0.5) / fig_h
    fig.text((ax1_left + sub_w / 2) / FIG_WIDTH_INCHES, cap_y,
             "LiDAR mesh", ha="center", va="center",
             fontsize=4.5, color="#555", transform=fig.transFigure)
    fig.text((ax1_left + sub_w + sub_gap + sub_w / 2) / FIG_WIDTH_INCHES, cap_y,
             "Real RA (2D only)", ha="center", va="center",
             fontsize=4.5, color="#555", transform=fig.transFigure)

    # --- Tile 2: mmIR (materials + normals stacked left, rendered RA right) ---
    t2_left, t2_bot = tile_positions[1]
    ax2_left = t2_left + inner_pad
    ax2_bot = t2_bot + caption_h + inner_pad

    # Left: materials (top half) + normals (bottom half) stacked vertically
    stack_gap = 0.03  # gap between stacked images
    stack_h = (sub_h - stack_gap) / 2

    # Materials (top)
    ax_mat = fig.add_axes([
        ax2_left / FIG_WIDTH_INCHES, (ax2_bot + stack_h + stack_gap) / fig_h,
        sub_w / FIG_WIDTH_INCHES, stack_h / fig_h,
    ])
    if os.path.exists(mat_render_path):
        mat_img = _process_3d_render(mat_render_path)
        ax_mat.imshow(mat_img)
    else:
        ax_mat.text(0.5, 0.5, "Materials", ha="center", va="center",
                    transform=ax_mat.transAxes, fontsize=5, color="#999")
    ax_mat.axis("off")

    # Normals (bottom)
    ax_norm = fig.add_axes([
        ax2_left / FIG_WIDTH_INCHES, ax2_bot / fig_h,
        sub_w / FIG_WIDTH_INCHES, stack_h / fig_h,
    ])
    if os.path.exists(norm_render_path):
        norm_img = _process_3d_render(norm_render_path)
        ax_norm.imshow(norm_img)
    else:
        ax_norm.text(0.5, 0.5, "Normals", ha="center", va="center",
                     transform=ax_norm.transAxes, fontsize=5, color="#999")
    ax_norm.axis("off")

    # Right: rendered RA
    right_x = ax2_left + sub_w + sub_gap
    ax_rend_ra = fig.add_axes([
        right_x / FIG_WIDTH_INCHES, ax2_bot / fig_h,
        sub_w / FIG_WIDTH_INCHES, sub_h / fig_h,
    ])
    if os.path.exists(rendered_ra_path):
        rend_ra = np.load(rendered_ra_path)
        rend_lin = _ra_to_linear(rend_ra)
        ax_rend_ra.imshow(rend_lin, cmap="hot", aspect="equal", origin="lower",
                          vmin=0.0, vmax=1.0, interpolation="bilinear")
    else:
        ax_rend_ra.text(0.5, 0.5, "Rendered RA", ha="center", va="center",
                        transform=ax_rend_ra.transAxes, fontsize=5, color="#999")
    ax_rend_ra.axis("off")

    # Captions
    fig.text((ax2_left + sub_w / 2) / FIG_WIDTH_INCHES, cap_y,
             "Materials + Normals", ha="center", va="center",
             fontsize=4.5, color="#555", transform=fig.transFigure)
    fig.text((right_x + sub_w / 2) / FIG_WIDTH_INCHES, cap_y,
             "Rendered RA", ha="center", va="center",
             fontsize=4.5, color="#555", transform=fig.transFigure)

    # --- Tile 3: 3D Output (single large image) ---
    t3_left, t3_bot = tile_positions[2]
    ax3_left = t3_left + inner_pad
    ax3_bot = t3_bot + caption_h + inner_pad
    full_img_w = tile_w - 2 * inner_pad

    ax_3d = fig.add_axes([
        ax3_left / FIG_WIDTH_INCHES, ax3_bot / fig_h,
        full_img_w / FIG_WIDTH_INCHES, sub_h / fig_h,
    ])
    if os.path.exists(dense_3d_path):
        img_3d = _process_3d_render(dense_3d_path)
        ax_3d.imshow(img_3d, aspect="equal")
    else:
        ax_3d.text(0.5, 0.5, "3D Point Cloud", ha="center", va="center",
                   transform=ax_3d.transAxes, fontsize=5, color="#999")
    ax_3d.axis("off")

    # Caption
    fig.text((ax3_left + full_img_w / 2) / FIG_WIDTH_INCHES, cap_y,
             "3D occupancy from single frame", ha="center", va="center",
             fontsize=4.5, color="#555", transform=fig.transFigure)

    # --- Arrows between tiles ---
    overlay = fig.add_axes([0, 0, 1, 1], zorder=50)
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    overlay.patch.set_alpha(0)

    arrow_style = dict(
        arrowstyle="-|>", color="#444", lw=0.8, mutation_scale=10,
    )
    # Mid-height of the image area
    arrow_y = (ax1_bot + sub_h / 2) / fig_h

    for i in range(n_tiles - 1):
        t_left_i, _ = tile_positions[i]
        t_left_next, _ = tile_positions[i + 1]
        x_from = (t_left_i + tile_w + 0.04) / FIG_WIDTH_INCHES
        x_to = (t_left_next - 0.04) / FIG_WIDTH_INCHES
        overlay.annotate(
            "", xy=(x_to, arrow_y), xytext=(x_from, arrow_y),
            xycoords="figure fraction", textcoords="figure fraction",
            arrowprops=arrow_style,
        )

    # Arrow labels
    for i, label in enumerate(["Optimize", "Re-render"]):
        t_left_i, _ = tile_positions[i]
        t_left_next, _ = tile_positions[i + 1]
        x_mid = ((t_left_i + tile_w + t_left_next) / 2) / FIG_WIDTH_INCHES
        label_y = arrow_y + 0.05
        fig.text(x_mid, label_y, label,
                 ha="center", va="bottom",
                 fontsize=5.5, fontweight="bold", color="#444",
                 transform=fig.transFigure)

    # --- Save ---
    out_pdf = os.path.join(output_dir, "teaser_v3.pdf")
    out_png = os.path.join(output_dir, "teaser_v3.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none", edgecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="Generate teaser figure (Figure 1)")
    parser.add_argument("--train_dir", default="output/train_v11",
                        help="Training output root")
    parser.add_argument("--postproc_dir", default="output/postprocess_final_v11",
                        help="Postprocessing output root")
    parser.add_argument("--output_dir", default="output/postprocess_final_v9/figures",
                        help="Output directory for figure")
    args = parser.parse_args()
    generate_figure(args.output_dir, args.train_dir, args.postproc_dir)


if __name__ == "__main__":
    main()
