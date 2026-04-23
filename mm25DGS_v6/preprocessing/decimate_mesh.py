"""
Quadric Edge Decimation for LiDAR-derived Meshes.

Standalone CLI script using Open3D's simplify_quadric_decimation()
(Garland-Heckbert QEM algorithm) to reduce triangle count while
preserving geometry.

Usage:
    # Single mesh
    python decimate_mesh.py --mesh path/to/mesh.ply --target-ratio 0.10

    # Single scene directory
    python decimate_mesh.py --scene-dir path/to/scene/ --target-ratio 0.10

    # Batch all 6 benchmark scenes
    python decimate_mesh.py --batch --target-ratio 0.10 --visualize

    # Fixed target face count (overrides ratio)
    python decimate_mesh.py --batch --target-faces 10000 --visualize
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


# ============================================================================
# 6 Benchmark Scenes
# ============================================================================
BENCHMARK_SCENES = [
    "seq_0_frame_135",
    "seq_1_frame_185",
    "seq_1_frame_277",
    "seq_1_frame_438",
    "seq_2_frame_160",
    "seq_2_frame_300",
]

DEFAULT_DATA_ROOT = None


# ============================================================================
# Core Functions
# ============================================================================

def compute_mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    """Compute statistics for a triangle mesh."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_verts = len(vertices)
    n_faces = len(triangles)

    stats = {
        "n_vertices": n_verts,
        "n_faces": n_faces,
    }

    if n_faces == 0:
        stats.update({
            "area_min": 0.0, "area_max": 0.0,
            "area_mean": 0.0, "area_median": 0.0,
            "area_total": 0.0,
            "bbox_min": vertices.min(axis=0) if n_verts > 0 else np.zeros(3),
            "bbox_max": vertices.max(axis=0) if n_verts > 0 else np.zeros(3),
        })
        return stats

    # Compute face areas
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    stats.update({
        "area_min": float(areas.min()),
        "area_max": float(areas.max()),
        "area_mean": float(areas.mean()),
        "area_median": float(np.median(areas)),
        "area_total": float(areas.sum()),
        "bbox_min": vertices.min(axis=0),
        "bbox_max": vertices.max(axis=0),
    })
    return stats


def decimate_mesh(
    mesh_path: str,
    target_faces: Optional[int] = None,
    target_ratio: Optional[float] = None,
    boundary_weight: float = 1.0,
    verbose: bool = True,
) -> tuple:
    """
    Load mesh and run quadric edge collapse decimation.

    Uses Open3D's simplify_quadric_decimation (Garland-Heckbert QEM).

    Args:
        mesh_path: Path to input PLY mesh
        target_faces: Absolute target face count (overrides target_ratio)
        target_ratio: Fraction of faces to keep (e.g., 0.10 = 10%)
        boundary_weight: Boundary preservation weight
        verbose: Print statistics

    Returns:
        (original_mesh, decimated_mesh, stats_orig, stats_dec)
    """
    original = o3d.io.read_triangle_mesh(mesh_path)
    original.compute_vertex_normals()

    stats_orig = compute_mesh_stats(original)
    n_orig = stats_orig["n_faces"]

    if target_faces is not None:
        n_target = target_faces
    elif target_ratio is not None:
        n_target = max(1, int(n_orig * target_ratio))
    else:
        n_target = max(1, int(n_orig * 0.10))

    if verbose:
        print(f"  Original: {n_orig:,} faces, {stats_orig['n_vertices']:,} vertices")
        print(f"  Target:   {n_target:,} faces ({n_target/n_orig*100:.1f}%)")

    decimated = original.simplify_quadric_decimation(
        target_number_of_triangles=n_target,
    )
    decimated.compute_vertex_normals()

    stats_dec = compute_mesh_stats(decimated)

    if verbose:
        print(f"  Result:   {stats_dec['n_faces']:,} faces, {stats_dec['n_vertices']:,} vertices")
        area_increase = stats_dec["area_median"] / max(stats_orig["area_median"], 1e-12)
        print(f"  Median area: {stats_orig['area_median']:.6f} -> {stats_dec['area_median']:.6f} m² ({area_increase:.1f}x)")

    return original, decimated, stats_orig, stats_dec


def save_decimated_mesh(mesh: o3d.geometry.TriangleMesh, output_path: str, verbose: bool = True):
    """Save decimated mesh as PLY with vertex normals."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_normals=True)
    if verbose:
        print(f"  Saved: {output_path}")


# ============================================================================
# Visualization (matplotlib Poly3DCollection — no EGL/GPU needed)
# ============================================================================

def _plot_mesh_on_ax(
    ax,
    mesh: o3d.geometry.TriangleMesh,
    azimuth_deg: float,
    elevation_deg: float,
    max_faces_render: int = 25000,
):
    """Plot a mesh on a matplotlib 3D axis using Poly3DCollection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Subsample faces if too many for matplotlib rendering speed
    if len(triangles) > max_faces_render:
        idx = np.random.default_rng(42).choice(len(triangles), max_faces_render, replace=False)
        triangles = triangles[idx]

    # Build polygon list
    polys = vertices[triangles]

    # Simple face shading based on normal z-component
    v0 = polys[:, 0]
    v1 = polys[:, 1]
    v2 = polys[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals = normals / norms
    # Lambertian shading: light from above-right
    light_dir = np.array([0.3, 0.3, 1.0])
    light_dir /= np.linalg.norm(light_dir)
    shade = np.abs(normals @ light_dir)
    shade = 0.3 + 0.7 * shade  # ambient + diffuse

    face_colors = np.column_stack([shade * 0.75, shade * 0.75, shade * 0.78, np.ones_like(shade)])

    collection = Poly3DCollection(polys, linewidths=0.1, edgecolors=(0.2, 0.2, 0.2, 0.3))
    collection.set_facecolor(face_colors)
    ax.add_collection3d(collection)

    # Set axis limits
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.view_init(elev=elevation_deg, azim=azimuth_deg)
    ax.set_axis_off()


def visualize_before_after(
    original: o3d.geometry.TriangleMesh,
    decimated: o3d.geometry.TriangleMesh,
    stats_orig: dict,
    stats_dec: dict,
    output_path: str,
    scene_name: str = "",
    render_width: int = 800,
    render_height: int = 600,
):
    """
    Generate side-by-side before/after comparison image.

    Layout: 2 columns (original | decimated) x 3 rows (perspective, top-down, front).
    Uses matplotlib Poly3DCollection (no GPU/EGL needed).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viewpoints = [
        ("Perspective", 45, 30),
        ("Top-Down", 0, 89),
        ("Front", 0, 15),
    ]

    fig = plt.figure(figsize=(16, 18))

    for row_idx, (view_name, az, el) in enumerate(viewpoints):
        # Original (left column)
        ax_orig = fig.add_subplot(3, 2, row_idx * 2 + 1, projection="3d")
        _plot_mesh_on_ax(ax_orig, original, az, el)
        ax_orig.set_title(
            f"Original — {view_name}\n"
            f"{stats_orig['n_faces']:,} faces, {stats_orig['n_vertices']:,} verts\n"
            f"median area: {stats_orig['area_median']:.5f} m²",
            fontsize=11,
        )

        # Decimated (right column)
        ax_dec = fig.add_subplot(3, 2, row_idx * 2 + 2, projection="3d")
        _plot_mesh_on_ax(ax_dec, decimated, az, el)
        area_increase = stats_dec["area_median"] / max(stats_orig["area_median"], 1e-12)
        ax_dec.set_title(
            f"Decimated — {view_name}\n"
            f"{stats_dec['n_faces']:,} faces, {stats_dec['n_vertices']:,} verts\n"
            f"median area: {stats_dec['area_median']:.5f} m² ({area_increase:.1f}x)",
            fontsize=11,
        )

    ratio_pct = stats_dec["n_faces"] / max(stats_orig["n_faces"], 1) * 100
    title = f"Quadric Edge Decimation: {scene_name}" if scene_name else "Quadric Edge Decimation"
    title += f"\n{stats_orig['n_faces']:,} → {stats_dec['n_faces']:,} faces ({ratio_pct:.1f}%)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved: {output_path}")


# ============================================================================
# Scene Processing
# ============================================================================

def process_single_scene(
    scene_dir: str,
    target_faces: Optional[int] = None,
    target_ratio: Optional[float] = None,
    boundary_weight: float = 1.0,
    output_name: str = "mesh_decimated.ply",
    visualize: bool = False,
    verbose: bool = True,
) -> dict:
    """Process one scene: load mesh, decimate, save, optionally visualize."""
    scene_path = Path(scene_dir)
    mesh_path = scene_path / "scene" / "mesh.ply"

    if not mesh_path.exists():
        print(f"  WARNING: mesh not found at {mesh_path}, skipping")
        return {}

    scene_name = scene_path.name
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scene: {scene_name}")
        print(f"{'='*60}")

    original, decimated, stats_orig, stats_dec = decimate_mesh(
        str(mesh_path),
        target_faces=target_faces,
        target_ratio=target_ratio,
        boundary_weight=boundary_weight,
        verbose=verbose,
    )

    out_path = scene_path / "scene" / output_name
    save_decimated_mesh(decimated, str(out_path), verbose=verbose)

    if visualize:
        vis_path = scene_path / "scene" / f"decimation_comparison.png"
        visualize_before_after(
            original, decimated, stats_orig, stats_dec,
            str(vis_path), scene_name=scene_name,
        )

    return {
        "scene": scene_name,
        "original": stats_orig,
        "decimated": stats_dec,
        "output_path": str(out_path),
    }


def process_batch(
    data_root: str,
    scenes: list,
    target_faces: Optional[int] = None,
    target_ratio: Optional[float] = None,
    boundary_weight: float = 1.0,
    output_name: str = "mesh_decimated.ply",
    visualize: bool = False,
    verbose: bool = True,
) -> list:
    """Process all specified scenes and print summary table."""
    results = []
    for scene_name in scenes:
        scene_dir = os.path.join(data_root, scene_name)
        if not os.path.isdir(scene_dir):
            print(f"  WARNING: scene dir not found: {scene_dir}")
            continue
        result = process_single_scene(
            scene_dir,
            target_faces=target_faces,
            target_ratio=target_ratio,
            boundary_weight=boundary_weight,
            output_name=output_name,
            visualize=visualize,
            verbose=verbose,
        )
        if result:
            results.append(result)

    # Print summary table
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Scene':<25} | {'Original':>10} | {'Decimated':>10} | {'Ratio':>6} | {'Area Increase':>14}")
        print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*14}")
        for r in results:
            orig = r["original"]["n_faces"]
            dec = r["decimated"]["n_faces"]
            ratio = dec / max(orig, 1)
            area_inc = r["decimated"]["area_median"] / max(r["original"]["area_median"], 1e-12)
            print(f"{r['scene']:<25} | {orig:>10,} | {dec:>10,} | {ratio:>5.2f}x | {area_inc:>10.1f}x median")

    return results


# ============================================================================
# CLI
# ============================================================================

def cli():
    ap = argparse.ArgumentParser(
        description="Quadric edge decimation for LiDAR-derived meshes (Open3D Garland-Heckbert QEM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input modes (mutually exclusive)
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--mesh", type=str, help="Path to a single mesh PLY file")
    input_group.add_argument("--scene-dir", type=str, help="Path to a scene directory (expects scene/mesh.ply)")
    input_group.add_argument("--batch", action="store_true", help="Process all benchmark scenes")

    # Target specification
    target_group = ap.add_mutually_exclusive_group()
    target_group.add_argument("--target-faces", type=int, default=None,
                              help="Absolute target face count")
    target_group.add_argument("--target-ratio", type=float, default=None,
                              help="Fraction of faces to keep (e.g., 0.10 = 10%%). Default: 0.10")

    # Options
    ap.add_argument("--boundary-weight", type=float, default=1.0,
                    help="Boundary preservation weight (default: 1.0)")
    ap.add_argument("--output", type=str, default="mesh_decimated.ply",
                    help="Output filename (default: mesh_decimated.ply)")
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                    help=f"Root data directory for batch mode (default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="Specific scene names for batch mode (default: all 6)")
    ap.add_argument("--visualize", action="store_true",
                    help="Generate before/after comparison images")
    ap.add_argument("--verbose", action="store_true", default=True,
                    help="Print detailed statistics (default: True)")

    args = ap.parse_args()

    # Default ratio if neither target is specified
    target_faces = args.target_faces
    target_ratio = args.target_ratio
    if target_faces is None and target_ratio is None:
        target_ratio = 0.10

    if args.mesh:
        # Single mesh mode
        mesh_path = Path(args.mesh)
        original, decimated, stats_orig, stats_dec = decimate_mesh(
            str(mesh_path),
            target_faces=target_faces,
            target_ratio=target_ratio,
            boundary_weight=args.boundary_weight,
            verbose=args.verbose,
        )
        out_path = mesh_path.parent / args.output
        save_decimated_mesh(decimated, str(out_path), verbose=args.verbose)
        if args.visualize:
            vis_path = mesh_path.parent / "decimation_comparison.png"
            visualize_before_after(original, decimated, stats_orig, stats_dec, str(vis_path))

    elif args.scene_dir:
        process_single_scene(
            args.scene_dir,
            target_faces=target_faces,
            target_ratio=target_ratio,
            boundary_weight=args.boundary_weight,
            output_name=args.output,
            visualize=args.visualize,
            verbose=args.verbose,
        )

    elif args.batch:
        scenes = args.scenes if args.scenes else BENCHMARK_SCENES
        process_batch(
            args.data_root,
            scenes,
            target_faces=target_faces,
            target_ratio=target_ratio,
            boundary_weight=args.boundary_weight,
            output_name=args.output,
            visualize=args.visualize,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    cli()
