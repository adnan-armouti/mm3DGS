"""Shared utilities for paper figure generation scripts."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

BACKGROUND_COLOR = "#ecebeb"
TEST_COL_COLOR = "#f5c6c0"   # light red — a tad darker than the lightred macro
                              # in main.tex (#fadbd8) so it reads cleanly against
                              # the grey background tile.
FIG_WIDTH_INCHES = 7.1  # Double-column
CORNER_RADIUS_INCHES = 0.08  # physical corner radius

# ---------------------------------------------------------------------------
# Font / typography — match the NeurIPS 2026 paper body (Times, 10pt body;
# we use 7pt here so per-cell labels and column headers don't visually
# dominate the actual content). Call `apply_paper_font()` once at the top of
# any figure-generation script BEFORE creating the Figure.
# ---------------------------------------------------------------------------

PAPER_BODY_PT = 10.0          # NeurIPS 2026 body text size
FIGURE_BASE_PT = 9.0          # default figure inline / label / tick size
FIGURE_HEADER_PT = 10.0       # column headers / row labels / titles
FIGURE_SMALL_PT = 7.0         # per-cell overlays (CC values etc.)


def apply_paper_font():
    """Configure matplotlib rcParams to match the NeurIPS paper typography.

    Uses a Times-compatible serif family with Computer Modern math, so that
    figure text rendered by matplotlib visually matches the LaTeX body. Safe
    to call repeatedly. Does NOT enable usetex (figures stay portable
    without a LaTeX install in the matplotlib backend)."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Nimbus Roman", "Times New Roman",
                                "Liberation Serif", "DejaVu Serif"],
        "font.size":          FIGURE_BASE_PT,
        "axes.labelsize":     FIGURE_BASE_PT,
        "axes.titlesize":     FIGURE_HEADER_PT,
        "xtick.labelsize":    FIGURE_BASE_PT,
        "ytick.labelsize":    FIGURE_BASE_PT,
        "legend.fontsize":    FIGURE_BASE_PT,
        "figure.titlesize":   FIGURE_HEADER_PT,
        "mathtext.fontset":   "cm",
        "mathtext.rm":        "serif",
        "pdf.fonttype":       42,    # TrueType — embeddable, NeurIPS-safe
        "ps.fonttype":        42,
    })

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

    @classmethod
    def from_fig_width(cls, n_rows, n_cols, fig_width_in, img_aspect=1.0,
                        **kwargs):
        """Like ``from_image_aspect`` but with a custom total figure width.

        Useful for supplementary figures that need wider rows than the
        default 2-column body width (e.g. 9 columns vs 6)."""
        margin_in = kwargs.pop("margin_in", 0.08)
        col_gap_in = kwargs.pop("col_gap_in", 0.04)
        label_w_in = kwargs.pop("label_w_in", 0.42)
        usable_w = (fig_width_in - label_w_in - 2 * margin_in
                     - (n_cols - 1) * col_gap_in)
        cell_w = usable_w / n_cols
        cell_h = cell_w * img_aspect
        obj = cls(n_rows, n_cols, cell_w, cell_h,
                   margin_in=margin_in, col_gap_in=col_gap_in,
                   label_w_in=label_w_in, **kwargs)
        obj.fig_w = fig_width_in
        obj.fig_h = (
            2 * obj.margin_in + obj.header_in
            + n_rows * cell_h
            + (n_rows - 1) * obj.row_gap_in
        )
        return obj


def _rounded_rect_path(x0, y0, w, h, fig_w, fig_h, radius_in):
    """Build a matplotlib ``Path`` for a rounded rectangle with physically
    circular corners (i.e., radius in inches is the same in x and y, even
    though the figure width and height differ in fraction-of-figure units).

    Coordinates are in figure-fraction (matches ``transform=fig.transFigure``).
    """
    rx = radius_in / fig_w   # x-radius in figure fraction (corner is circular
    ry = radius_in / fig_h   # in inches, but rx/ry differ in fraction units)
    rx = min(rx, w * 0.5)
    ry = min(ry, h * 0.5)
    k = 0.5523               # cubic-Bezier quarter-circle coefficient
    x1, y1 = x0 + w, y0 + h
    verts = [
        (x0,           y0 + ry),
        (x0,           y0 + ry * (1 - k)),
        (x0 + rx * (1 - k), y0),
        (x0 + rx,      y0),
        (x1 - rx,      y0),
        (x1 - rx * (1 - k), y0),
        (x1,           y0 + ry * (1 - k)),
        (x1,           y0 + ry),
        (x1,           y1 - ry),
        (x1,           y1 - ry * (1 - k)),
        (x1 - rx * (1 - k), y1),
        (x1 - rx,      y1),
        (x0 + rx,      y1),
        (x0 + rx * (1 - k), y1),
        (x0,           y1 - ry * (1 - k)),
        (x0,           y1 - ry),
        (x0,           y0 + ry),
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
    return Path(verts, codes)


def add_column_highlight(fig, layout, col, color=TEST_COL_COLOR,
                          radius_inches=CORNER_RADIUS_INCHES * 0.65,
                          inset_in=0.025, include_header=True):
    """Highlight one full column (across all rows) with a rounded tile.

    Drawn between the grey background tile (``add_rounded_bg``, zorder=-1)
    and the row cells (default zorder 0), so it sits *on top of* the grey
    tile but *behind* the image cells. The column header text is rendered
    later via ``fig.text`` (default zorder 3) and therefore sits *on top
    of* this tile, which is why ``include_header=True`` (default) extends
    the tile vertically to cover the column-header strip too. Corners are
    physically circular (``radius_inches`` in both x and y) by construction.
    """
    left, top_bottom, w, _ = layout.cell_pos(0, col)
    _, bot_bottom, _, _ = layout.cell_pos(layout.n_rows - 1, col)
    top_of_first = top_bottom + layout.cell_h_in / layout.fig_h
    full_h = top_of_first - bot_bottom
    inset_x = inset_in / layout.fig_w
    inset_y = inset_in / layout.fig_h
    x0 = left - inset_x
    y0 = bot_bottom - inset_y
    rect_w = w + 2 * inset_x
    rect_h = full_h + 2 * inset_y
    if include_header:
        header_top_y = 1.0 - layout.margin_in / layout.fig_h
        rect_h = header_top_y - y0

    path = _rounded_rect_path(x0, y0, rect_w, rect_h,
                                layout.fig_w, layout.fig_h, radius_inches)
    patch = mpatches.PathPatch(
        path, transform=fig.transFigure, facecolor=color, edgecolor="none",
        zorder=-0.5, linewidth=0,
    )
    fig.patches.append(patch)
