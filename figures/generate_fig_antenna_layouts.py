#!/usr/bin/env python3
"""Generate the antenna layout comparison figure for the paper.

Double-column figure with 1x3 tile layout:
  - 3 tiles (slightly darker background, circular rounded corners) on main bg
  - Each tile: Physical TX/RX (top) + Virtual aperture (bottom)
  - Tiles: Cascaded radar, Single-chip radar, Dense 100x100 array
  - Minimized vertical height

Usage:
    python -m mmir.evaluation.generate_fig_antenna_layouts \
        --output_dir output/postprocess_final_v11/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

from .fig_common import (
    BACKGROUND_COLOR, FIG_WIDTH_INCHES, CORNER_RADIUS_INCHES,
    add_rounded_bg,
)

TILE_COLOR = "#d4d3d3"  # visibly darker than #ecebeb for inner tiles


# ---------------------------------------------------------------------------
# Antenna config parsers
# ---------------------------------------------------------------------------

def parse_antenna_cfg(path: str) -> dict:
    """Parse antenna_cfg.txt files (half-wavelength unit positions)."""
    tx_positions, rx_positions = [], []
    f_design = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "F_design":
                f_design = float(parts[1])
            elif parts[0] == "rx":
                rx_positions.append((float(parts[2]), float(parts[3])))
            elif parts[0] == "tx":
                tx_positions.append((float(parts[2]), float(parts[3])))
    return {"tx": np.array(tx_positions), "rx": np.array(rx_positions),
            "f_design_ghz": f_design}


def compute_virtual_array(tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    virtual = []
    for rx_pos in rx:
        for tx_pos in tx:
            virtual.append(rx_pos + tx_pos)
    return np.array(virtual)


def generate_cascade_physical_layout() -> dict:
    """Generate TI MMWCAS-RF-EVM physical antenna positions from datasheet.

    Returns positions in wavelength (lambda) units, centered on the board
    bounding-box center.  After y-mirror the TX row sits above the RX row.
    """
    # Datasheet spacing constants (in lambda)
    A1 = 0.5    # RX element spacing
    A3 = 4.0    # gap between RX group B and C
    A4 = 16.0   # gap between RX group C and A
    A5 = 7.75   # TX row x-offset
    A6 = 2.0    # TX element spacing
    B1 = 19.0   # TX-RX vertical separation
    B2 = 0.5    # staggered TX y-offset 1
    B3 = 1.5    # staggered TX y-offset 2
    B4 = 1.0    # staggered TX y-offset 3

    # RX positions (3 groups along y=0)
    rx_B = [(i * A1, 0) for i in range(4)]
    start_C = 4 * A1 + A3
    rx_C = [(start_C + i * A1, 0) for i in range(4)]
    start_A = start_C + 4 * A1 + A4
    rx_A = [(start_A + i * A1, 0) for i in range(8)]
    rx = np.array(rx_B + rx_C + rx_A)

    # TX positions (9 in a row + 3 staggered)
    tx_list = [(A5 + i * A6, -B1) for i in range(9)]
    stagger_x = A5 + 2 * A6
    stagger_y = -B1
    tx_list.append((stagger_x + A1, stagger_y + B2))
    tx_list.append((stagger_x + 2 * A1, stagger_y + B2 + B3))
    tx_list.append((stagger_x + 3 * A1, stagger_y + B2 + B3 + B4))
    tx = np.array(tx_list)

    # Y-mirror (negate y) so TX row is above RX row
    rx[:, 1] *= -1
    tx[:, 1] *= -1

    # Center on bounding-box center
    all_pos = np.vstack([rx, tx])
    center = (all_pos.min(axis=0) + all_pos.max(axis=0)) / 2
    rx -= center
    tx -= center

    virtual = compute_virtual_array(tx, rx)
    return {"tx": tx, "rx": rx, "virtual": virtual}


def generate_single_chip_physical_layout() -> dict:
    """Generate TI AWR1843BOOSTEVM physical antenna positions.

    Positions from mmir/preprocessing/config_utils.py::antenna_layout_single_chip().
    Returns positions in wavelength (lambda) units, centered on bounding-box center.
    """
    c = 299792458.0
    lambda_mm = c / (77.0e9) * 1000.0  # ~3.896 mm

    A1 = 0.5   # RX element spacing (lambda)
    C1 = 1.0   # TX element spacing (lambda)
    B1 = 3.724 / lambda_mm  # RX-TX gap (mm -> lambda) ~0.956 lambda

    # 4 RX: uniform linear array at y=0 with 0.5 lambda spacing
    rx = np.array([(i * A1, 0.0) for i in range(4)])

    # 3 TX: after gap B1 from end of RX array, 1.0 lambda spacing
    start_x = 4 * A1 + B1
    tx = np.array([(start_x + i * C1, 0.0) for i in range(3)])
    tx[1, 1] += A1  # TX1 (middle) elevated by 0.5 lambda in y

    # Center on bounding-box center
    all_pos = np.vstack([rx, tx])
    center = (all_pos.min(axis=0) + all_pos.max(axis=0)) / 2
    rx -= center
    tx -= center

    virtual = compute_virtual_array(tx, rx)
    return {"tx": tx, "rx": rx, "virtual": virtual}


def generate_dense_array(num_antennas: int = 100) -> dict:
    tx_per_column = num_antennas // 2
    rx_per_row = num_antennas // 2
    tx, rx = [], []
    for y in np.arange(-(tx_per_column - 1) / 2, tx_per_column / 2, 1):
        tx.append((-(tx_per_column / 2), y))
        tx.append((tx_per_column / 2, y))
    for x in np.arange(-(rx_per_row - 1) / 2, rx_per_row / 2, 1):
        rx.append((x, -(tx_per_column / 2)))
        rx.append((x, tx_per_column / 2))
    tx, rx = np.array(tx), np.array(rx)
    virtual = np.array([(r[0]+t[0], r[1]+t[1]) for r in rx for t in tx])
    return {"tx": tx, "rx": rx, "virtual": virtual}


# ---------------------------------------------------------------------------
# Rounded rectangle tile helper
# ---------------------------------------------------------------------------

def add_tile_bg(fig, left, bottom, width, height, color=TILE_COLOR,
                radius_inches=0.06):
    """Add a rounded-rectangle tile with physically circular corners."""
    fig_w, fig_h = fig.get_size_inches()
    rx = radius_inches / fig_w
    ry = radius_inches / fig_h

    # Convert tile bounds to figure fractions
    x0, y0 = left, bottom
    x1, y1 = left + width, bottom + height

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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, bg_color=TILE_COLOR, equal_aspect=False):
    ax.set_facecolor(bg_color)
    if equal_aspect:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, linewidth=0.3, color="#999999")
    ax.tick_params(axis="both", labelsize=4, length=1.5, width=0.3, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("#888888")


def plot_tx_rx(ax, tx, rx):
    n = len(tx) + len(rx)
    s = max(4, min(18, 300 / n))
    ax.scatter(rx[:, 0], rx[:, 1], s=s, c="#1976D2", alpha=0.9,
               edgecolors="none", marker="o", zorder=3)
    ax.scatter(tx[:, 0], tx[:, 1], s=s, c="#D32F2F", alpha=0.9,
               edgecolors="none", marker="o", zorder=3)
    _style_ax(ax)
    all_pts = np.vstack([tx, rx])
    x_ptp = max(all_pts[:, 0].ptp(), 1)
    y_ptp = max(all_pts[:, 1].ptp(), 0.5)
    mx = 0.2 * x_ptp
    my = max(0.2 * y_ptp, 0.5)
    ax.set_xlim(all_pts[:, 0].min() - mx, all_pts[:, 0].max() + mx)
    ax.set_ylim(all_pts[:, 1].min() - my, all_pts[:, 1].max() + my)


def plot_virtual(ax, virtual):
    unique = np.unique(virtual, axis=0)
    n = len(unique)
    if n > 500:
        nbins = min(int(np.sqrt(n)) + 1, 150)
        h, xe, ye = np.histogram2d(unique[:, 0], unique[:, 1], bins=nbins)
        ax.imshow(h.T, origin="lower", cmap="Greens", aspect="auto",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="bilinear")
    else:
        s = max(4, min(18, 300 / n))
        ax.scatter(unique[:, 0], unique[:, 1], s=s, c="#388E3C", alpha=0.85,
                   edgecolors="none", marker="o", zorder=3)
    _style_ax(ax)
    x_ptp = max(unique[:, 0].ptp(), 1)
    y_ptp = max(unique[:, 1].ptp(), 0.5)
    mx = 0.15 * x_ptp
    my = max(0.15 * y_ptp, 0.5)
    ax.set_xlim(unique[:, 0].min() - mx, unique[:, 0].max() + mx)
    ax.set_ylim(unique[:, 1].min() - my, unique[:, 1].max() + my)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_figure(output_dir: str, calib_dir: str = None):
    os.makedirs(output_dir, exist_ok=True)
    if calib_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        calib_dir = os.path.join(root, "mmir", "utils", "preproc", "calib")

    cascade = generate_cascade_physical_layout()
    single = generate_single_chip_physical_layout()
    dense = generate_dense_array(100)

    cascade_v = cascade["virtual"]
    single_v = single["virtual"]

    n_cascade_v = len(np.unique(cascade_v, axis=0))
    n_single_v = len(np.unique(single_v, axis=0))
    n_dense_v = len(np.unique(dense["virtual"], axis=0))

    print("Antenna configurations:")
    print(f"  Cascaded: {len(cascade['tx'])} TX, {len(cascade['rx'])} RX "
          f"-> {n_cascade_v} virtual")
    print(f"  Single-chip: {len(single['tx'])} TX, {len(single['rx'])} RX "
          f"-> {n_single_v} virtual")
    print(f"  Dense: {len(dense['tx'])} TX, {len(dense['rx'])} RX "
          f"-> {n_dense_v} virtual")

    # --- Layout in inches ---
    # 1x3 horizontal tiles, each tile has 2 side-by-side subplots (TX/RX | Virtual)
    margin_in = 0.10          # outer margin
    tile_gap_in = 0.08        # gap between tiles
    inner_pad_left = 0.26    # left padding (room for y-axis ticks+labels)
    inner_pad_right = 0.16   # right padding (room for rightmost tick labels)
    inner_pad_bot = 0.22     # bottom padding (room for x-axis ticks+labels)
    inner_pad_top = 0.04     # top padding (above title)
    subplot_hgap_in = 0.24   # horizontal gap between subplots (room for right y-ticks + left y-ticks)
    title_h_in = 0.30        # height for tile title text (2 lines)

    n_tiles = 3
    usable_w = FIG_WIDTH_INCHES - 2 * margin_in - (n_tiles - 1) * tile_gap_in
    tile_w_in = usable_w / n_tiles

    # Two side-by-side subplots within each tile
    subplot_w_in = (tile_w_in - inner_pad_left - inner_pad_right - subplot_hgap_in) / 2
    subplot_h_in = subplot_w_in  # square subplots

    # Tile height: top_pad + title + subplot + bot_pad
    tile_h_in = inner_pad_top + title_h_in + subplot_h_in + inner_pad_bot
    fig_h = 2 * margin_in + tile_h_in

    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, fig_h))
    add_rounded_bg(fig)

    configs = [
        {
            "title": "Cascaded Radar",
            "subtitle": f"12 TX, 16 RX -> {n_cascade_v} virtual",
            "tx": cascade["tx"], "rx": cascade["rx"], "virtual": cascade_v,
        },
        {
            "title": "Single-Chip Radar",
            "subtitle": f"3 TX, 4 RX -> {n_single_v} virtual",
            "tx": single["tx"], "rx": single["rx"], "virtual": single_v,
        },
        {
            "title": "Dense Virtual Array",
            "subtitle": f"100 TX, 100 RX -> {n_dense_v} virtual",
            "tx": dense["tx"], "rx": dense["rx"], "virtual": dense["virtual"],
        },
    ]

    for i, cfg in enumerate(configs):
        # Tile position in inches
        tile_left_in = margin_in + i * (tile_w_in + tile_gap_in)
        tile_bottom_in = margin_in

        # Figure fractions
        tile_left = tile_left_in / FIG_WIDTH_INCHES
        tile_bottom = tile_bottom_in / fig_h
        tile_w = tile_w_in / FIG_WIDTH_INCHES
        tile_h = tile_h_in / fig_h

        # Draw tile background
        add_tile_bg(fig, tile_left, tile_bottom, tile_w, tile_h)

        # Tile title (top of tile, centered) — use inches for consistent spacing
        title_top_in = tile_bottom_in + tile_h_in - inner_pad_top
        title_y = title_top_in / fig_h
        subtitle_y = (title_top_in - 0.12) / fig_h  # 0.12 inches below title
        fig.text(
            tile_left + tile_w / 2, title_y,
            cfg["title"],
            ha="center", va="top",
            fontsize=6, fontweight="bold",
            transform=fig.transFigure,
        )
        fig.text(
            tile_left + tile_w / 2, subtitle_y,
            f"({cfg['subtitle']})",
            ha="center", va="top",
            fontsize=5, color="#555555",
            transform=fig.transFigure,
        )

        # Left subplot: Physical TX/RX
        ax_left_in = tile_left_in + inner_pad_left
        ax_bottom_in = tile_bottom_in + inner_pad_bot
        ax_left = ax_left_in / FIG_WIDTH_INCHES
        ax_bottom = ax_bottom_in / fig_h
        ax_w = subplot_w_in / FIG_WIDTH_INCHES
        ax_h = subplot_h_in / fig_h

        ax_phys = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
        plot_tx_rx(ax_phys, cfg["tx"], cfg["rx"])
        ax_phys.set_title("Physical Antennas", fontsize=4.5, fontstyle="italic",
                          pad=2, color="#555555")

        # Right subplot: Virtual aperture
        ax_right_in = ax_left_in + subplot_w_in + subplot_hgap_in
        ax_right = ax_right_in / FIG_WIDTH_INCHES

        ax_virt = fig.add_axes([ax_right, ax_bottom, ax_w, ax_h])
        plot_virtual(ax_virt, cfg["virtual"])
        ax_virt.set_title("Virtual Aperture", fontsize=4.5, fontstyle="italic",
                          pad=2, color="#555555")

    out_pdf = os.path.join(output_dir, "antenna_layouts.pdf")
    out_png = os.path.join(output_dir, "antenna_layouts.png")
    fig.savefig(out_pdf, dpi=300, facecolor="none")
    fig.savefig(out_png, dpi=300, facecolor=BACKGROUND_COLOR)
    plt.close(fig)

    print(f"\nSaved: {out_pdf}")
    print(f"Saved: {out_png}")
    return out_pdf


def main():
    parser = argparse.ArgumentParser(description="Generate antenna layout comparison figure")
    parser.add_argument("--output_dir", type=str,
                        default="output/postprocess_final_v11/figures")
    parser.add_argument("--calib_dir", type=str, default=None)
    args = parser.parse_args()
    generate_figure(args.output_dir, args.calib_dir)


if __name__ == "__main__":
    main()
