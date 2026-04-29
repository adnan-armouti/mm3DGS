"""Render per-vertex trained material parameters on mesh using Open3D.

Works with the 6-parameter physics model (eps_real, eps_imag, sigma_h, l_c,
tau, thickness) stored in best_materials.npz from training output.
"""

import os
from typing import Optional

import numpy as np
from matplotlib import cm
from PIL import Image

try:
    import open3d as o3d
except ImportError:
    o3d = None

from .material_loader import load_our_physics_params
from .utils.visualization_open3d import (
    VIEWPOINTS,
    parse_radar_config,
    create_radar_visualization_geometries,
    _setup_renderer,
)

# ITU concrete defaults at 77 GHz (used as reference for deviation metric)
PHYSICS_DEFAULTS = {
    "eps_real": 5.31,
    "eps_imag": 0.29,
    "sigma_h": 1e-4,
    "l_c": 0.01,
    "tau": 0.5,
    "thickness": 0.1,
}

# Parameter ranges from reparameterization.py (for normalization)
PHYSICS_RANGES = {
    "eps_real": (1.5, 10.0),
    "eps_imag": (1e-3, 9e6),
    "sigma_h": (1e-7, 1e-3),
    "l_c": (5e-4, 0.1),
    "tau": (0.05, 0.95),
    "thickness": (1e-3, 0.3),
}

PARAM_NAMES = ["eps_real", "eps_imag", "sigma_h", "l_c", "tau", "thickness"]


def compute_overall_change(physics_params: np.ndarray) -> np.ndarray:
    """Compute per-vertex deviation from ITU defaults, averaged across params.

    Log-space deviation for log-scale params (eps_imag, sigma_h, l_c, thickness),
    linear-space for sigmoid params (eps_real, tau).

    Args:
        physics_params: (n_vertices, 6) array of physics parameters.

    Returns:
        (n_vertices,) array in [0, 1] — 0 = default, 1 = maximum deviation.
    """
    deviations = []
    for i, name in enumerate(PARAM_NAMES):
        vals = physics_params[:, i]
        default = PHYSICS_DEFAULTS[name]
        vmin, vmax = PHYSICS_RANGES[name]

        if name in ("eps_real", "tau"):
            # Linear-space deviation
            dev = np.abs(vals - default) / (vmax - vmin)
        else:
            # Log-space deviation
            log_vals = np.log10(np.clip(vals, vmin, vmax))
            log_default = np.log10(default)
            log_range = np.log10(vmax) - np.log10(vmin)
            dev = np.abs(log_vals - log_default) / log_range

        deviations.append(np.clip(dev, 0, 1))

    return np.mean(deviations, axis=0)


def render_material_visualization(
    mesh_path: str,
    training_dir: str,
    view: str = "oblique_1",
    param_name: str = "overall_change",
    colormap: str = "inferno",
    width: int = 2048,
    height: int = 2048,
    radar_config_path: Optional[str] = None,
    bg_color: Optional[list] = None,
    fov: float = 60.0,
) -> Image.Image:
    """Render mesh colored by trained material parameter.

    Args:
        mesh_path: Path to .ply mesh.
        training_dir: Directory containing best_materials.npz.
        view: Viewpoint name.
        param_name: "overall_change" or one of the 6 physics param names.
        colormap: Matplotlib colormap name.
        width, height: Render resolution.
        radar_config_path: Optional radar config for markers.
        bg_color: Background RGBA [0-1].

    Returns:
        PIL Image (RGBA).
    """
    if o3d is None:
        raise ImportError("Open3D is required for material visualization")

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"Mesh at {mesh_path} has no triangles")
    mesh.compute_vertex_normals()

    n_vertices = len(mesh.vertices)

    # Load trained materials
    physics, metadata = load_our_physics_params(training_dir)
    if physics.shape[0] != n_vertices:
        # Handle small mismatches (e.g., mesh has 1 extra vertex from
        # Open3D vs trimesh loading). Pad or truncate to match.
        if abs(physics.shape[0] - n_vertices) <= 10:
            if physics.shape[0] < n_vertices:
                pad = np.tile(physics[-1:], (n_vertices - physics.shape[0], 1))
                physics = np.vstack([physics, pad])
            else:
                physics = physics[:n_vertices]
        else:
            raise ValueError(
                f"Material params ({physics.shape[0]}) don't match "
                f"mesh vertices ({n_vertices})"
            )

    # Compute values to colormap
    if param_name == "overall_change":
        values = compute_overall_change(physics)
    elif param_name in PARAM_NAMES:
        idx = PARAM_NAMES.index(param_name)
        vals = physics[:, idx]
        vmin, vmax = PHYSICS_RANGES[param_name]
        if param_name in ("eps_real", "tau"):
            values = (vals - vmin) / (vmax - vmin)
        else:
            log_vals = np.log10(np.clip(vals, vmin, vmax))
            values = (log_vals - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
        values = np.clip(values, 0, 1)
    else:
        raise ValueError(f"Unknown param_name: {param_name}")

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colors_rgba = cmap(values)
    colors_rgb = colors_rgba[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_rgb)

    # Set up renderer
    renderer = _setup_renderer(width, height)
    if bg_color is not None:
        renderer.scene.set_background(bg_color)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", mesh, mat)

    # Add radar markers
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

    # Camera
    bbox = mesh.get_axis_aligned_bounding_box()
    lookat = bbox.get_center()
    bbox_max = bbox.get_max_bound()
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))

    vp = None
    for name, az, el, ds in VIEWPOINTS:
        if name == f"view_{view}" or name == view:
            vp = (az, el, ds)
            break
    if vp is None:
        vp = (45, 30, 1.5)

    az_deg, el_deg, dist_scale = vp
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    distance = max_extent * dist_scale

    cam_x = distance * np.cos(el_rad) * np.sin(az_rad)
    cam_y = distance * np.cos(el_rad) * np.cos(az_rad)
    cam_z = distance * np.sin(el_rad)
    camera_pos = lookat + np.array([cam_x, cam_y, cam_z])

    min_camera_z = bbox_max[2] + extent[2] * 0.2
    if camera_pos[2] < min_camera_z:
        req_z = min_camera_z - lookat[2]
        req_xy = req_z / np.tan(el_rad) if el_rad > 0 else distance
        cam_x = req_xy * np.sin(az_rad)
        cam_y = req_xy * np.cos(az_rad)
        camera_pos = np.array([lookat[0] + cam_x, lookat[1] + cam_y, min_camera_z])

    renderer.setup_camera(fov, lookat, camera_pos, np.array([0, 0, 1]))

    img_o3d = renderer.render_to_image()
    img_np = np.asarray(img_o3d)
    del renderer

    return Image.fromarray(img_np)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render material visualization")
    parser.add_argument("--mesh", required=True, help="Path to .ply mesh")
    parser.add_argument("--training_dir", required=True, help="Training output dir")
    parser.add_argument("--config", default=None, help="Radar config JSON")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--param", default="overall_change")
    parser.add_argument("--colormap", default="inferno")
    parser.add_argument("--view", default="oblique_1")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=2048)
    args = parser.parse_args()

    img = render_material_visualization(
        args.mesh, args.training_dir,
        view=args.view, param_name=args.param, colormap=args.colormap,
        width=args.width, height=args.height, radar_config_path=args.config,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"materials_{args.param}_{args.view}.png")
    img.save(out)
    print(f"Saved {out}")
