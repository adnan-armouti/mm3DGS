"""Open3D offscreen rendering for publication-quality 3D point cloud figures.

Renders point clouds (radar, LiDAR) overlaid on scene mesh from 9 viewpoints.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy Open3D import — graceful fallback if not available
# Note: EGL fix (system libstdc++ preload) is in mmir.evaluation.__init__
try:
    import open3d as o3d
except ImportError:
    o3d = None


# ── 9 viewpoints (exact copy from reference scripts) ────────────────────────
VIEWPOINTS = [
    ("view_front", 0, 20, 1.5),
    ("view_back", 180, 20, 1.5),
    ("view_left", -90, 20, 1.5),
    ("view_right", 90, 20, 1.5),
    ("view_top", 0, 89, 1.5),
    ("view_oblique_1", 45, 30, 1.5),
    ("view_oblique_2", -45, 30, 1.5),
    ("view_oblique_3", 135, 30, 1.5),
    ("view_oblique_4", -135, 30, 1.5),
]


# ── Radar config parsing (from reference script) ────────────────────────────

def parse_radar_config(config_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse radar config to extract sensor center and boresight direction.

    Returns:
        (radar_center_xyz_meters, boresight_unit_vector) or (None, None)
    """
    if config_path is None or not os.path.exists(config_path):
        return None, None

    with open(config_path, "r") as f:
        config = json.load(f)

    positions = []
    if "tx_array" in config:
        for tx in config["tx_array"]:
            positions.append(tx["pos_mm"])
    if "rx_array" in config:
        for rx in config["rx_array"]:
            positions.append(rx["pos_mm"])

    if not positions:
        return None, None

    positions = np.array(positions)
    radar_center = np.mean(positions, axis=0) / 1000.0  # mm → m

    boresight = None
    if "tx_array" in config and len(config["tx_array"]) > 0:
        boresight = np.array(config["tx_array"][0]["boresight"])
        boresight = boresight / np.linalg.norm(boresight)

    return radar_center, boresight


def create_radar_visualization_geometries(
    radar_center: np.ndarray,
    boresight_direction: np.ndarray,
    sphere_radius: float = 0.15,
    arrow_length: float = 2.0,
):
    """Create Open3D sphere (red) + arrow (green) for radar position/boresight.

    Extracted from reference script (lines 81-165).
    """
    if o3d is None or radar_center is None or boresight_direction is None:
        return None, None

    # Red sphere at sensor center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(radar_center)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sphere.compute_vertex_normals()

    # Green arrow: cylinder shaft + cone head
    arrow_cylinder_radius = sphere_radius * 0.2
    arrow_cone_radius = sphere_radius * 0.4
    arrow_cone_height = arrow_length * 0.15
    arrow_cylinder_height = arrow_length - arrow_cone_height

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=arrow_cylinder_radius, height=arrow_cylinder_height
    )
    cone = o3d.geometry.TriangleMesh.create_cone(
        radius=arrow_cone_radius, height=arrow_cone_height
    )

    # Rotation from Z-axis to boresight via Rodrigues' formula
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, boresight_direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, boresight_direction)

    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    elif c > 0:
        R = np.eye(3)
    else:
        R = -np.eye(3)

    cylinder_center = radar_center + boresight_direction * (arrow_cylinder_height / 2)
    cylinder.rotate(R, center=[0, 0, 0])
    cylinder.translate(cylinder_center)
    cylinder.paint_uniform_color([0.0, 1.0, 0.0])
    cylinder.compute_vertex_normals()

    cone_center = radar_center + boresight_direction * (arrow_cylinder_height + arrow_cone_height / 2)
    cone.rotate(R, center=[0, 0, 0])
    cone.translate(cone_center)
    cone.paint_uniform_color([0.0, 1.0, 0.0])
    cone.compute_vertex_normals()

    arrow = cylinder + cone
    return sphere, arrow


# ── Core rendering functions ─────────────────────────────────────────────────

def _setup_renderer(width: int, height: int):
    """Create and configure Open3D OffscreenRenderer with standard lighting."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light(
        [0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 75000
    )
    renderer.scene.scene.enable_indirect_light(True)
    renderer.scene.scene.set_indirect_light_intensity(45000)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    return renderer


def _add_mesh(renderer, mesh_path: str):
    """Load mesh, add as semi-transparent gray geometry. Returns bbox info."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh is None or not mesh.has_triangles():
        return None, None, None, None

    mesh.compute_vertex_normals()
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.compute_vertex_normals()
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.7, 0.7, 0.7, 0.5]
    mat.has_alpha = True
    renderer.scene.add_geometry("mesh", mesh_vis, mat)

    bbox = mesh.get_axis_aligned_bounding_box()
    return bbox.get_center(), bbox.get_max_bound(), bbox.get_extent(), np.max(bbox.get_extent())


def _render_viewpoints(renderer, lookat, bbox_max, extent, max_extent, output_dir, prefix):
    """Render all 9 viewpoints, saving PNGs. Returns list of saved paths."""
    saved = []
    for view_name, azimuth_deg, elevation_deg, distance_scale in VIEWPOINTS:
        try:
            azimuth_rad = np.deg2rad(azimuth_deg)
            elevation_rad = np.deg2rad(elevation_deg)
            distance = max_extent * distance_scale

            cam_x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
            cam_y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
            cam_z = distance * np.sin(elevation_rad)
            camera_pos = lookat + np.array([cam_x, cam_y, cam_z])

            # Ensure camera is above scene
            min_camera_z = bbox_max[2] + extent[2] * 0.2
            if camera_pos[2] < min_camera_z:
                required_z_offset = min_camera_z - lookat[2]
                required_xy_dist = (
                    required_z_offset / np.tan(elevation_rad) if elevation_rad > 0 else distance
                )
                cam_x = required_xy_dist * np.sin(azimuth_rad)
                cam_y = required_xy_dist * np.cos(azimuth_rad)
                camera_pos = np.array([lookat[0] + cam_x, lookat[1] + cam_y, min_camera_z])

            renderer.setup_camera(60.0, lookat, camera_pos, np.array([0, 0, 1]))

            img = renderer.render_to_image()
            fname = f"{prefix}_{view_name}.png" if prefix else f"{view_name}.png"
            img_path = os.path.join(output_dir, fname)
            o3d.io.write_image(img_path, img)
            saved.append(img_path)
        except Exception as e:
            print(f"  WARNING: Failed to render {view_name}: {e}")
    return saved


def _make_pcd(xyz: np.ndarray, colors: np.ndarray) -> "o3d.geometry.PointCloud":
    """Create Open3D point cloud from (N,3) positions and (N,3) RGB colors."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def _intensity_to_plasma(intensities: np.ndarray) -> np.ndarray:
    """Map intensity values to plasma colormap RGB (N,3)."""
    from matplotlib import cm

    vmin, vmax = float(intensities.min()), float(intensities.max())
    if vmax <= vmin:
        return np.full((len(intensities), 3), 0.5)
    vals_n = np.clip((intensities - vmin) / (vmax - vmin + 1e-12), 0, 1)
    return cm.get_cmap("plasma")(vals_n)[:, :3]


# ── Public API ───────────────────────────────────────────────────────────────

def render_pointcloud_views(
    point_cloud_xyz: np.ndarray,
    point_intensities: Optional[np.ndarray],
    mesh_path: str,
    output_dir: str,
    radar_config_path: Optional[str] = None,
    set_name: str = "",
    point_size: float = 5.0,
    width: int = 3000,
    height: int = 2400,
) -> List[str]:
    """Render intensity-colored point cloud + mesh from 9 viewpoints.

    Args:
        point_cloud_xyz: (N, 3) world-coordinate points.
        point_intensities: (N,) intensity values (or None for uniform color).
        mesh_path: Path to .ply mesh file.
        output_dir: Directory to save PNGs.
        radar_config_path: Optional radar config JSON for sensor marker.
        set_name: Prefix for output filenames (e.g. "dense_radar").
        point_size: Point size in pixels.
        width, height: Render resolution.

    Returns:
        List of saved PNG paths.
    """
    if o3d is None:
        print("WARNING: Open3D not available, skipping 3D rendering")
        return []

    os.makedirs(output_dir, exist_ok=True)
    renderer = _setup_renderer(width, height)

    # Add mesh
    lookat, bbox_max, extent, max_extent = _add_mesh(renderer, mesh_path)
    if lookat is None:
        print(f"WARNING: Could not load mesh from {mesh_path}")
        return []

    # Create colored point cloud
    if point_intensities is not None:
        colors = _intensity_to_plasma(point_intensities)
    else:
        colors = np.tile([1.0, 0.3, 0.0], (len(point_cloud_xyz), 1))  # orange
    pcd = _make_pcd(point_cloud_xyz, colors)

    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = point_size
    renderer.scene.add_geometry("radar", pcd, mat_pcd)

    # Add radar position/boresight markers if config provided
    if radar_config_path:
        center, boresight = parse_radar_config(radar_config_path)
        if center is not None and boresight is not None:
            sphere, arrow = create_radar_visualization_geometries(center, boresight)
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

    result = _render_viewpoints(renderer, lookat, bbox_max, extent, max_extent, output_dir, set_name)
    del renderer  # explicit cleanup to release GPU resources
    return result


def render_mesh_only_views(
    mesh_path: str,
    output_dir: str,
    radar_config_path: Optional[str] = None,
    set_name: str = "mesh",
    width: int = 3000,
    height: int = 2400,
) -> List[str]:
    """Render mesh with optional radar markers (no point cloud) from 9 viewpoints.

    Args:
        mesh_path: Path to .ply mesh file.
        output_dir: Directory to save PNGs.
        radar_config_path: Optional radar config JSON for sensor marker.
        set_name: Prefix for output filenames.
        width, height: Render resolution.

    Returns:
        List of saved PNG paths.
    """
    if o3d is None:
        print("WARNING: Open3D not available, skipping 3D rendering")
        return []

    os.makedirs(output_dir, exist_ok=True)
    renderer = _setup_renderer(width, height)

    lookat, bbox_max, extent, max_extent = _add_mesh(renderer, mesh_path)
    if lookat is None:
        print(f"WARNING: Could not load mesh from {mesh_path}")
        return []

    # Add radar position/boresight markers if config provided
    if radar_config_path:
        center, boresight = parse_radar_config(radar_config_path)
        if center is not None and boresight is not None:
            sphere, arrow = create_radar_visualization_geometries(center, boresight)
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

    result = _render_viewpoints(renderer, lookat, bbox_max, extent, max_extent, output_dir, set_name)
    del renderer
    return result


def render_lidar_views(
    lidar_xyz: np.ndarray,
    mesh_path: str,
    output_dir: str,
    set_name: str = "lidar",
    point_size: float = 4.0,
    width: int = 3000,
    height: int = 2400,
) -> List[str]:
    """Render LiDAR point cloud (blue) + mesh from 9 viewpoints.

    Args:
        lidar_xyz: (N, 3) world-coordinate LiDAR points.
        mesh_path: Path to .ply mesh file.
        output_dir: Directory to save PNGs.
        set_name: Prefix for output filenames.
        point_size: Point size in pixels.
        width, height: Render resolution.

    Returns:
        List of saved PNG paths.
    """
    if o3d is None:
        print("WARNING: Open3D not available, skipping 3D rendering")
        return []

    os.makedirs(output_dir, exist_ok=True)
    renderer = _setup_renderer(width, height)

    lookat, bbox_max, extent, max_extent = _add_mesh(renderer, mesh_path)
    if lookat is None:
        print(f"WARNING: Could not load mesh from {mesh_path}")
        return []

    # Blue uniform color for LiDAR
    colors = np.tile([0.0, 0.5, 1.0], (len(lidar_xyz), 1))
    pcd = _make_pcd(lidar_xyz, colors)

    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = point_size
    renderer.scene.add_geometry("lidar", pcd, mat_pcd)

    result = _render_viewpoints(renderer, lookat, bbox_max, extent, max_extent, output_dir, set_name)
    del renderer  # explicit cleanup to release GPU resources
    return result
