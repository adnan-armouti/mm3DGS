"""Render mesh with RGB normal-mapped vertex colors using Open3D offscreen rendering.

Normals in [-1, 1] are mapped to RGB in [0, 1] for the classic normal-map look.
"""

import os
from typing import Optional

import numpy as np
from PIL import Image

try:
    import open3d as o3d
except ImportError:
    o3d = None

from .utils.visualization_open3d import (
    VIEWPOINTS,
    parse_radar_config,
    create_radar_visualization_geometries,
    _setup_renderer,
)


def render_normal_visualization(
    mesh_path: str,
    view: str = "oblique_1",
    width: int = 2048,
    height: int = 2048,
    radar_config_path: Optional[str] = None,
    bg_color: Optional[list] = None,
    fov: float = 60.0,
) -> Image.Image:
    """Render mesh with per-vertex normal colors from a single viewpoint.

    Args:
        mesh_path: Path to .ply mesh file.
        view: Viewpoint name (e.g. "oblique_1", "front", "top").
        width, height: Render resolution.
        radar_config_path: Optional radar config JSON for red sphere + green arrow.
        bg_color: Background RGBA [0-1] (default white).

    Returns:
        PIL Image (RGBA).
    """
    if o3d is None:
        raise ImportError("Open3D is required for normal visualization")

    # Load mesh and compute normals
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {mesh_path} has no triangles")
    mesh.compute_vertex_normals()

    # Compute camera position first so we can orient normals toward it
    bbox = mesh.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))

    vp = None
    for name, az, el, ds in VIEWPOINTS:
        if name == f"view_{view}" or name == view:
            vp = (az, el, ds)
            break
    if vp is None:
        vp = (45, 30, 1.5)

    az_rad = np.deg2rad(vp[0])
    el_rad = np.deg2rad(vp[1])
    distance = max_extent * vp[2]
    camera_pos = lookat + np.array([
        distance * np.cos(el_rad) * np.sin(az_rad),
        distance * np.cos(el_rad) * np.cos(az_rad),
        distance * np.sin(el_rad),
    ])

    # World-space normal map: X→R, Y→G, Z→B with (n+1)/2 mapping.
    # Surfaces pointing up (Z+) → strong blue; tilted surfaces get
    # red/green mixed with blue → characteristic purple/magenta look.
    normals = np.asarray(mesh.vertex_normals).copy()
    vertices = np.asarray(mesh.vertices)

    # Flip normals that point away from the camera so back-faces
    # get consistent coloring
    to_cam = camera_pos[None, :] - vertices
    dots = np.sum(normals * to_cam, axis=1)
    normals[dots < 0] *= -1

    # Map [-1, 1] → [0, 1]  (world-space: X→R, Y→G, Z→B)
    colors = (normals + 1.0) / 2.0
    colors = np.clip(colors, 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Set up renderer
    renderer = _setup_renderer(width, height)
    if bg_color is not None:
        renderer.scene.set_background(bg_color)

    # Add mesh with unlit shader to preserve exact normal colors
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # Add radar markers if config provided
    if radar_config_path:
        center, boresight = parse_radar_config(radar_config_path)
        if center is not None and boresight is not None:
            sphere, arrow = create_radar_visualization_geometries(
                center, boresight, sphere_radius=0.15, arrow_length=2.0
            )
            if sphere is not None:
                mat_s = o3d.visualization.rendering.MaterialRecord()
                mat_s.shader = "defaultLit"
                mat_s.base_color = [1.0, 0.0, 0.0, 1.0]
                renderer.scene.add_geometry("radar_sphere", sphere, mat_s)
            if arrow is not None:
                mat_a = o3d.visualization.rendering.MaterialRecord()
                mat_a.shader = "defaultLit"
                mat_a.base_color = [0.0, 1.0, 0.0, 1.0]
                renderer.scene.add_geometry("radar_arrow", arrow, mat_a)

    # Camera setup — reuse bbox/vp already computed above for normal orientation
    bbox_max = bbox.get_max_bound()

    min_camera_z = bbox_max[2] + extent[2] * 0.2
    if camera_pos[2] < min_camera_z:
        req_z = min_camera_z - lookat[2]
        req_xy = req_z / np.tan(el_rad) if el_rad > 0 else distance
        cam_x = req_xy * np.sin(az_rad)
        cam_y = req_xy * np.cos(az_rad)
        camera_pos = np.array([lookat[0] + cam_x, lookat[1] + cam_y, min_camera_z])

    renderer.setup_camera(fov, lookat, camera_pos, np.array([0, 0, 1]))

    # Render
    img_o3d = renderer.render_to_image()
    img_np = np.asarray(img_o3d)
    del renderer

    return Image.fromarray(img_np)


def render_normal_views(
    mesh_path: str,
    output_dir: str,
    radar_config_path: Optional[str] = None,
    prefix: str = "normals",
    width: int = 2048,
    height: int = 2048,
):
    """Render normal-colored mesh from all 9 viewpoints and save PNGs."""
    os.makedirs(output_dir, exist_ok=True)
    for vp_name, _, _, _ in VIEWPOINTS:
        view = vp_name.replace("view_", "")
        img = render_normal_visualization(
            mesh_path, view=view, width=width, height=height,
            radar_config_path=radar_config_path,
        )
        out_path = os.path.join(output_dir, f"{prefix}_{vp_name}.png")
        img.save(out_path)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render mesh normal visualization")
    parser.add_argument("--mesh", required=True, help="Path to .ply mesh")
    parser.add_argument("--config", default=None, help="Radar config JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--view", default=None, help="Single viewpoint (or all)")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    args = parser.parse_args()

    if args.view:
        img = render_normal_visualization(
            args.mesh, view=args.view, width=args.width, height=args.height,
            radar_config_path=args.config,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, f"normals_view_{args.view}.png")
        img.save(out)
        print(f"Saved {out}")
    else:
        render_normal_views(
            args.mesh, args.output_dir, radar_config_path=args.config,
            width=args.width, height=args.height,
        )
