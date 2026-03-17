"""Figure generation utilities for paper publication."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def save_ra_comparison_figure(
    ra_images: Dict[str, np.ndarray],
    output_path: str,
    titles: Optional[Dict[str, str]] = None,
    cmap: str = "plasma",
    dpi: int = 150,
    figsize_per_panel: Tuple[float, float] = (4.0, 3.0),
):
    """Save side-by-side RA comparison figure.

    Args:
        ra_images: dict mapping label → 2D RA magnitude array
            e.g. {"GT": gt_ra, "Ours": our_ra, "Sionna": sionna_ra}
        output_path: path to save figure (PDF or PNG)
        titles: optional override for panel titles
        cmap: colormap
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ra_images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))
    if n == 1:
        axes = [axes]

    # Common vmin/vmax across all panels for fair comparison
    all_vals = np.concatenate([v.ravel() for v in ra_images.values()])
    vmin, vmax = np.percentile(all_vals[np.isfinite(all_vals)], [2, 98])

    for ax, (label, ra) in zip(axes, ra_images.items()):
        title = titles[label] if titles and label in titles else label
        im = ax.imshow(ra, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Range bin")
        ax.set_ylabel("Azimuth bin")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_convergence_figure(
    iterations: List[int],
    metrics_series: Dict[str, List[float]],
    output_path: str,
    xlabel: str = "Iteration",
    dpi: int = 150,
):
    """Save training convergence plot (loss/correlation vs iteration).

    Args:
        iterations: list of iteration numbers
        metrics_series: dict mapping metric name → list of values
            e.g. {"Loss": [...], "Correlation": [...]}
        output_path: path to save figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_metrics = len(metrics_series)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics_series.items()):
        # Filter out None values
        valid = [(it, v) for it, v in zip(iterations, values) if v is not None]
        if not valid:
            continue
        its, vals = zip(*valid)
        ax.plot(its, vals, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ra_slice_figure(
    rae_cube: np.ndarray,
    output_path: str,
    cmap: str = "plasma",
    dpi: int = 150,
):
    """Save RA slice visualization: azimuth=0 and elevation=0 cross-sections.

    Args:
        rae_cube: (Az, El, R) magnitude array from compute_rae_cube
        output_path: path to save figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_az, n_el, n_r = rae_cube.shape
    mid_el = n_el // 2  # elevation=0 slice
    mid_az = n_az // 2  # azimuth=0 slice

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # RA slice at elevation=0 (Az vs Range)
    ra_slice = rae_cube[:, mid_el, :]  # (Az, R)
    ra_db = 20 * np.log10(np.maximum(ra_slice, 1e-10))
    vmin, vmax = np.percentile(ra_db[np.isfinite(ra_db)], [5, 99])
    axes[0].imshow(ra_db, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    axes[0].set_title("RA slice (elevation=0)", fontsize=10)
    axes[0].set_xlabel("Range bin")
    axes[0].set_ylabel("Azimuth bin")

    # RE slice at azimuth=0 (El vs Range)
    re_slice = rae_cube[mid_az, :, :]  # (El, R)
    re_db = 20 * np.log10(np.maximum(re_slice, 1e-10))
    vmin, vmax = np.percentile(re_db[np.isfinite(re_db)], [5, 99])
    axes[1].imshow(re_db, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
    axes[1].set_title("RE slice (azimuth=0)", fontsize=10)
    axes[1].set_xlabel("Range bin")
    axes[1].set_ylabel("Elevation bin")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ra_cartesian_png(
    ra_cart: np.ndarray,
    output_path: str,
    range_res: float = 0.0557,
    scale: str = "dB",
    title: Optional[str] = None,
    cmap: str = "hot",
    db_floor: float = -40.0,
    dpi: int = 150,
):
    """Save a single RA Cartesian image with proper axis labels.

    X-axis = azimuth (cross-range, meters), Y-axis = range (meters).

    Args:
        ra_cart: 2D Cartesian RA magnitude array (rows=range, cols=azimuth).
        output_path: Path to save PNG.
        range_res: Range resolution in meters (for axis extents).
        scale: 'dB' or 'linear'.
        title: Optional figure title.
        cmap: Colormap name.
        db_floor: dB floor for dB scale.
        dpi: Output resolution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ra = np.asarray(ra_cart, dtype=np.float64)

    # Compute physical extent: rows=range, cols=azimuth
    n_range, n_az = ra.shape
    num_adc = n_range  # Cartesian grid matches range extent
    range_depth = num_adc * range_res
    range_width = range_depth / 2.0
    extent = [-range_width, range_width, 0.0, range_depth]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))

    if scale == "dB":
        ra_max = np.max(ra) if np.max(ra) > 0 else 1.0
        ra_ratio = np.clip(ra / ra_max, 1e-30, None)
        ra_disp = np.clip(20.0 * np.log10(ra_ratio), db_floor, 0.0)
        im = ax.imshow(ra_disp, cmap=cmap, aspect="auto", origin="lower",
                        vmin=db_floor, vmax=0.0, extent=extent)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("dB")
    else:
        mn, mx = ra.min(), ra.max()
        ra_disp = (ra - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(ra)
        im = ax.imshow(ra_disp, cmap=cmap, aspect="auto", origin="lower",
                        vmin=0, vmax=1, extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xlabel("Azimuth (m)")
    ax.set_ylabel("Range (m)")
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ra_error_map(
    ra_gt_cart: np.ndarray,
    ra_rend_cart: np.ndarray,
    output_path: str,
    range_res: float = 0.0557,
    title: Optional[str] = None,
    dpi: int = 150,
):
    """Save a signed error map (rendered - GT) on min-max normalized RA images.

    Red = rendered too bright, blue = rendered too dim.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(arr)

    error = _minmax(ra_rend_cart) - _minmax(ra_gt_cart)

    n_range, _ = error.shape
    range_depth = n_range * range_res
    range_width = range_depth / 2.0
    extent = [-range_width, range_width, 0.0, range_depth]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    im = ax.imshow(error, cmap="RdBu_r", aspect="auto", origin="lower",
                    vmin=-0.3, vmax=0.3, extent=extent)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Error")
    ax.set_xlabel("Azimuth (m)")
    ax.set_ylabel("Range (m)")
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_3d_pointcloud_figure(
    point_clouds: Dict[str, np.ndarray],
    output_path: str,
    colors: Optional[Dict[str, str]] = None,
    point_size: float = 0.5,
    elevation: float = 30.0,
    azimuth: float = -60.0,
    dpi: int = 150,
):
    """Save 3D point cloud comparison figure using matplotlib.

    Args:
        point_clouds: dict mapping label → (N, 3) point array
        output_path: path to save figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    default_colors = {"Radar": "tab:blue", "LiDAR": "tab:orange", "Mesh": "tab:green"}
    if colors is None:
        colors = default_colors

    n = len(point_clouds)
    fig = plt.figure(figsize=(6 * n, 5))

    for idx, (label, pts) in enumerate(point_clouds.items()):
        ax = fig.add_subplot(1, n, idx + 1, projection="3d")
        color = colors.get(label, f"C{idx}")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=color, alpha=0.3)
        ax.set_title(label, fontsize=10)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
