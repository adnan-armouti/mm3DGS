"""Shared utilities for paper figure generation scripts (high-res variant).

Mirrors fig_common.py but with the zoom-invariance fixes from
md/FIGURE_CHANGES.md ported in:
  - Fix #1: image.composite_image=False (no PDF figimage compositing)
  - Fix #2: fig.patch.set_visible(False) inside add_rounded_bg
  - Fix #4: figimage_in_axes + needed_dpi_for_native_embed helpers
"""

import matplotlib
matplotlib.use("Agg")
# Fix #1 (FIGURE_CHANGES.md §1): disable matplotlib's automatic compositing of
# adjacent figimage rasters. The composite fills inter-image gaps with pure
# black before resampling, producing dark anti-aliasing rings around embedded
# panels. With this off, every figimage is emitted as its own raster.
matplotlib.rcParams["image.composite_image"] = False

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
FIGURE_BASE_PT = 7.0          # default figure inline / label / tick size
FIGURE_HEADER_PT = 8.0        # column headers / row labels / titles
FIGURE_SMALL_PT = 6.0         # per-cell overlays (CC values etc.)


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
        path, facecolor=color, edgecolor="none", linewidth=0,
        transform=fig.transFigure, zorder=-1,
    )
    fig.patches.append(patch)
    # Fix #2 (FIGURE_CHANGES.md §2 + §6): hide the figure's own rectangle
    # entirely. set_alpha(0.0) still emits a transparent rect path which some
    # PDF viewers anti-alias into a faint outline on top of the rounded card.
    fig.patch.set_visible(False)


# ---------------------------------------------------------------------------
# Fix #4 (FIGURE_CHANGES.md §4): native-resolution raster embedding helpers.
#
# `ax.imshow(arr)` resamples the source raster to panel pixel size at savefig
# DPI, destroying detail in 3D-mesh thumbnails or RA-heatmap overlays. The
# helpers below place the raster via `fig.figimage` at its native pixel grid,
# so the PDF stores it at source resolution and viewers can zoom in cleanly.
# ---------------------------------------------------------------------------

# Cap the longest dimension of any raster panel embedded via figimage.
# A ~2000-px source on a ~1" LaTeX panel gives ~20x zoom headroom — well past
# typical reader zoom — while keeping per-panel PDF bytes bounded.
DEFAULT_FIGIMAGE_MAX_DIM_PX = 2000


def _hex_to_rgb(hex_str):
    """'#ecebeb' -> (236, 235, 235)."""
    s = hex_str.lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))


def needed_dpi_for_native_embed(arrs, cell_w_in, cell_h_in, slack=0,
                                max_dim_px=DEFAULT_FIGIMAGE_MAX_DIM_PX):
    """Pick fig.dpi so the longer panel dimension is ~max_dim_px.

    Bounds the per-panel raster to max_dim_px regardless of source size while
    making panel pixels match the resize target so figimage_in_axes fills the
    panel (no white margins at panel edges). Pass the list of arrays you plan
    to embed for `arrs` (currently advisory; selection is by cell size).
    """
    longer_cell = max(cell_w_in, cell_h_in)
    if longer_cell <= 0:
        return 600
    return int(np.ceil(max_dim_px / longer_cell)) + slack


def figimage_in_axes(fig, arr, ax, max_dim_px=DEFAULT_FIGIMAGE_MAX_DIM_PX,
                     fit="equal", figure_bg=None, zorder=None):
    """Place arr via fig.figimage so it fills the axes panel.

    fit="equal" (default) preserves aspect like imshow's default; fit="auto"
    stretches to fill. The displayed array is Lanczos-resized to the panel
    pixel size (capped at max_dim_px on the longer side).

    Lanczos can introduce slight color shifts and edge ringing, which makes
    even a uniform-bg image show a faint outline against the figure bg. After
    resize we snap all near-bg pixels (any channel within 6 of the figure-bg
    RGB) to the exact figure bg color so the embedded raster blends in.

    zorder forwards to fig.figimage so the embedded raster can sit above
    rounded-tile patches drawn over the figure bg. fig.images have default
    zorder=0; rounded tile patches drawn at zorder>=1 will otherwise hide
    the figimage.
    """
    arr = np.asarray(arr)
    if arr.ndim < 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return False
    arr_h, arr_w = arr.shape[:2]

    fig.canvas.draw_idle()
    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    panel_w_px = max(1, int(round(bbox.width)))
    panel_h_px = max(1, int(round(bbox.height)))

    if fit == "auto":
        target_w = panel_w_px
        target_h = panel_h_px
    else:
        arr_aspect = arr_w / arr_h
        panel_aspect = panel_w_px / panel_h_px
        if arr_aspect >= panel_aspect:
            target_w = panel_w_px
            target_h = max(1, int(round(target_w / arr_aspect)))
        else:
            target_h = panel_h_px
            target_w = max(1, int(round(target_h * arr_aspect)))

    # Note: max_dim_px is intentionally NOT applied as a hard cap on target
    # below the panel pixel size. fig.figimage places the raster at native
    # pixel size, so capping target_w/h below panel_w/h would render the
    # image smaller than the panel with centered margins — exactly the bug
    # reported. Callers should size fig.dpi (via needed_dpi_for_native_embed)
    # using the LARGEST figimage panel so panel_w_px ≈ max_dim_px on that
    # panel and is below it on smaller panels. max_dim_px is kept here as a
    # safety bound only when source array is large but panel is tiny — not
    # the typical case.

    if (target_w, target_h) != (arr_w, arr_h):
        from PIL import Image as _PIL
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr = np.asarray(_PIL.fromarray(arr).resize(
            (target_w, target_h), _PIL.LANCZOS)).copy()

    # Bg-snap (Fix #4 + §5 threshold): match near-bg pixels to the figure bg
    # exactly so Lanczos shift / edge ringing don't leave a 1-px outline.
    fig_bg_hex = figure_bg
    if fig_bg_hex is None:
        fc = fig.get_facecolor()
        if fc and fc[3] > 0:
            fig_bg_hex = "#%02x%02x%02x" % (
                int(round(fc[0] * 255)), int(round(fc[1] * 255)),
                int(round(fc[2] * 255)),
            )
        else:
            fig_bg_hex = BACKGROUND_COLOR
    bg_rgb = np.array(_hex_to_rgb(fig_bg_hex), dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        diff = np.abs(arr[..., :3].astype(np.int16) - bg_rgb.astype(np.int16))
        near_bg = np.all(diff <= 6, axis=2)
        out = arr.copy()
        out[near_bg, 0] = bg_rgb[0]
        out[near_bg, 1] = bg_rgb[1]
        out[near_bg, 2] = bg_rgb[2]
        arr = out

    xo = bbox.x0 + (panel_w_px - target_w) / 2
    yo = bbox.y0 + (panel_h_px - target_h) / 2
    figimage_kwargs = {}
    if zorder is not None:
        figimage_kwargs["zorder"] = zorder
    fig.figimage(arr, xo=xo, yo=yo, **figimage_kwargs)
    # Hide the axes' own patch (Fix #6): set_facecolor("none") still emits a
    # transparent rect path that some PDF viewers anti-alias into a faint
    # outline on top of the rounded background patch.
    ax.patch.set_visible(False)
    ax.set_xlim(0, target_w)
    ax.set_ylim(target_h, 0)
    return True


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
