"""Shared utilities for paper figure generation scripts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

BACKGROUND_COLOR = "#ecebeb"
FIG_WIDTH_INCHES = 7.1  # Double-column
CORNER_RADIUS_INCHES = 0.08  # physical corner radius

SCENE_SHORT_NAMES = {
    "seq_0_frame_135": "S0-F135",
    "seq_0_frame_390": "S0-F390",
    "seq_1_frame_185": "S1-F185",
    "seq_1_frame_438": "S1-F438",
    "seq_2_frame_105": "S2-F105",
    "seq_2_frame_160": "S2-F160",
    "seq_2_frame_300": "S2-F300",
}


def add_rounded_bg(fig, radius_inches=CORNER_RADIUS_INCHES, color=BACKGROUND_COLOR):
    """Add a rounded-rectangle background with physically circular corners."""
    fig_w, fig_h = fig.get_size_inches()
    rx = radius_inches / fig_w
    ry = radius_inches / fig_h

    k = 0.5523  # Bezier approximation of a quarter-circle
    verts = [
        (0, ry),
        (0, ry * (1 - k)), (rx * (1 - k), 0), (rx, 0),
        (1 - rx, 0),
        (1 - rx * (1 - k), 0), (1, ry * (1 - k)), (1, ry),
        (1, 1 - ry),
        (1, 1 - ry * (1 - k)), (1 - rx * (1 - k), 1), (1 - rx, 1),
        (rx, 1),
        (rx * (1 - k), 1), (0, 1 - ry * (1 - k)), (0, 1 - ry),
        (0, ry),
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
        transform=fig.transFigure, zorder=-1,
    )
    fig.patches.append(patch)
    fig.patch.set_alpha(0.0)


class GridLayout:
    """Compute a grid layout in inches, converting to figure fractions.

    This ensures images maintain their aspect ratio regardless of the
    overall figure shape (avoiding stretching from fraction-based math).
    """

    def __init__(self, n_rows, n_cols, cell_w_in, cell_h_in, *,
                 margin_in=0.08, col_gap_in=0.04, row_gap_in=0.04,
                 header_in=0.20, label_w_in=0.42):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cell_w_in = cell_w_in
        self.cell_h_in = cell_h_in
        self.margin_in = margin_in
        self.col_gap_in = col_gap_in
        self.row_gap_in = row_gap_in
        self.header_in = header_in
        self.label_w_in = label_w_in

        self.fig_w = FIG_WIDTH_INCHES
        self.fig_h = (
            2 * margin_in + header_in
            + n_rows * cell_h_in
            + (n_rows - 1) * row_gap_in
        )

    def cell_pos(self, row, col):
        """Return (left, bottom, width, height) in figure-fraction coords."""
        fw, fh = self.fig_w, self.fig_h
        left = (self.margin_in + self.label_w_in
                + col * (self.cell_w_in + self.col_gap_in)) / fw
        top = 1.0 - (self.margin_in + self.header_in
                      + row * (self.cell_h_in + self.row_gap_in)) / fh
        bottom = top - self.cell_h_in / fh
        w = self.cell_w_in / fw
        h = self.cell_h_in / fh
        return left, bottom, w, h

    def header_y(self):
        """Y-position (figure frac) for column header text."""
        return 1.0 - (self.margin_in + self.header_in * 0.15) / self.fig_h

    def row_label_x(self):
        """X-position (figure frac) for row labels."""
        return (self.margin_in + self.label_w_in * 0.5) / self.fig_w

    def row_label_y(self, row):
        """Y-center (figure frac) for a given row label."""
        _, bottom, _, h = self.cell_pos(row, 0)
        return bottom + h / 2

    @classmethod
    def from_image_aspect(cls, n_rows, n_cols, img_aspect=1.0, **kwargs):
        """Create layout computing cell width from available space."""
        margin_in = kwargs.pop("margin_in", 0.08)
        col_gap_in = kwargs.pop("col_gap_in", 0.04)
        label_w_in = kwargs.pop("label_w_in", 0.42)

        usable_w = (FIG_WIDTH_INCHES - label_w_in - 2 * margin_in
                    - (n_cols - 1) * col_gap_in)
        cell_w = usable_w / n_cols
        cell_h = cell_w * img_aspect
        return cls(n_rows, n_cols, cell_w, cell_h,
                   margin_in=margin_in, col_gap_in=col_gap_in,
                   label_w_in=label_w_in, **kwargs)
