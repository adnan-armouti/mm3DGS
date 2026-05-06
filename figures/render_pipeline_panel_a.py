#!/usr/bin/env python3
"""Open3D render for Panel (a) of the 3DPS pipeline figure.

Uses the same rendering pattern as figures/prep_teaser_panels_v3.py
(``defaultLit`` shader, paint_uniform_color, boresight-follow camera) so
the LiDAR mesh is shaded uniformly with no back-face artefacts. Each
optimised 3DPS point is rendered as a small *sphere* (looks like an
actual point cloud) with a thin cylindrical *normal indicator* sticking
out of it. No surfels — surfels mislead.
"""

import argparse
import ctypes
import json
import os
import sys

if "mitsuba" not in sys.modules:
    _libstdcxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if os.path.isfile(_libstdcxx):
        try:
            ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
_dri = "/usr/lib/x86_64-linux-gnu/dri"
if "LIBGL_DRIVERS_PATH" not in os.environ and os.path.isdir(_dri):
    os.environ["LIBGL_DRIVERS_PATH"] = _dri

import numpy as np
import torch
from PIL import Image

import open3d as o3d


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_to_normal(quats: np.ndarray) -> np.ndarray:
    q = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def _rot_align_z_to(target: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    t = target / (np.linalg.norm(target) + 1e-12)
    c = float(np.dot(z, t))
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, t)
    s = np.linalg.norm(v)
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _build_point_field(positions, normals,
                        sphere_color, line_color,
                        sphere_radius=0.025, sphere_res=10,
                        line_length=0.075, line_radius=0.0035,
                        line_res=8):
    """Return (spheres_mesh, normal_lines_mesh) — two merged TriangleMeshes
    suitable for direct add_geometry() into an Open3D OffscreenRenderer."""
    n = len(positions)

    # ── Spheres ────────────────────────────────────────────────────────────
    sph_t = o3d.geometry.TriangleMesh.create_sphere(
        radius=sphere_radius, resolution=sphere_res)
    sph_t.compute_vertex_normals()
    s_v = np.asarray(sph_t.vertices)
    s_t = np.asarray(sph_t.triangles)
    s_n = np.asarray(sph_t.vertex_normals)
    n_sv, n_st = s_v.shape[0], s_t.shape[0]

    all_sv = (np.tile(s_v, (n, 1))
              + np.repeat(positions, n_sv, axis=0))
    all_sn = np.tile(s_n, (n, 1))
    all_st = (np.tile(s_t, (n, 1))
              + np.repeat(np.arange(n) * n_sv, n_st)[:, None])
    all_sc = np.tile(np.asarray(sphere_color), (n * n_sv, 1))

    spheres = o3d.geometry.TriangleMesh()
    spheres.vertices = o3d.utility.Vector3dVector(all_sv)
    spheres.triangles = o3d.utility.Vector3iVector(all_st.astype(np.int32))
    spheres.vertex_normals = o3d.utility.Vector3dVector(all_sn)
    spheres.vertex_colors = o3d.utility.Vector3dVector(all_sc)

    # ── Normal lines (thin cylinders along the normal) ─────────────────────
    cyl_t = o3d.geometry.TriangleMesh.create_cylinder(
        radius=line_radius, height=line_length,
        resolution=line_res, split=1)
    cyl_t.compute_vertex_normals()
    c_v = np.asarray(cyl_t.vertices).copy()
    c_t = np.asarray(cyl_t.triangles).copy()
    c_n = np.asarray(cyl_t.vertex_normals).copy()
    n_cv, n_ct = c_v.shape[0], c_t.shape[0]

    all_cv = np.empty((n * n_cv, 3), dtype=np.float64)
    all_cn = np.empty_like(all_cv)
    all_ct = np.empty((n * n_ct, 3), dtype=np.int32)
    half = line_length / 2.0

    for i in range(n):
        R = _rot_align_z_to(normals[i])
        # Cylinder template is centred on origin along +z; offset so its base
        # sits on the point and tip extends along the normal.
        v = c_v @ R.T + positions[i] + half * normals[i]
        ni = c_n @ R.T
        s = i * n_cv
        all_cv[s:s + n_cv] = v
        all_cn[s:s + n_cv] = ni
        all_ct[i * n_ct:(i + 1) * n_ct] = c_t + s

    all_cc = np.tile(np.asarray(line_color), (n * n_cv, 1))

    lines = o3d.geometry.TriangleMesh()
    lines.vertices = o3d.utility.Vector3dVector(all_cv)
    lines.triangles = o3d.utility.Vector3iVector(all_ct)
    lines.vertex_normals = o3d.utility.Vector3dVector(all_cn)
    lines.vertex_colors = o3d.utility.Vector3dVector(all_cc)

    return spheres, lines


# ---------------------------------------------------------------------------
# Camera + render setup (mirrors prep_teaser_panels_v3.py)
# ---------------------------------------------------------------------------

def _setup_renderer(width, height, bg=(0.926, 0.922, 0.922, 1.0)):
    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1, 1, 1], 65000)
    r.scene.scene.enable_indirect_light(True)
    r.scene.scene.set_indirect_light_intensity(38000)
    r.scene.set_background(list(bg))
    return r


def _load_test_pose(scene, test_frame, alignment_dir):
    cands = [
        os.path.join(alignment_dir, scene, "cascade",
                     f"cascaded_frame_{test_frame}_aligned_pass2.json"),
        os.path.join(alignment_dir, scene, "cascade",
                     f"cascaded_frame_{test_frame}_aligned.json"),
    ]
    for c in cands:
        if os.path.exists(c):
            cfg = json.load(open(c))
            pos = []
            for k in ("tx_array", "rx_array"):
                for el in cfg.get(k, []):
                    pos.append(el["pos_mm"])
            center = np.mean(np.array(pos), axis=0) / 1000.0
            bore = np.array(cfg["tx_array"][0]["boresight"])
            bore /= np.linalg.norm(bore) + 1e-12
            return center, bore
    raise FileNotFoundError(f"No pose for {scene} F{test_frame}")


def _setup_boresight_camera(renderer, center, boresight,
                             back_dist=8.0, up_dist=4.5, forward_dist=4.0,
                             lookat_z_offset=-1.5, fov=44.0):
    bh = boresight.copy()
    bh[2] = 0.0
    nrm = np.linalg.norm(bh)
    bh = bh / nrm if nrm > 1e-6 else boresight
    cam = center - bh * back_dist + np.array([0, 0, up_dist])
    lookat = center + bh * forward_dist + np.array([0, 0, lookat_z_offset])
    renderer.setup_camera(fov, lookat, cam, np.array([0, 0, 1.0]))


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

# Consistent "3DPS pink" — pops against the gray mesh, matches downstream
# branding in the teaser.
PINK_RGB = (0.840, 0.235, 0.549)
NORMAL_LINE_RGB = (0.16, 0.16, 0.18)


def render_scene_panel(
    model_path: str,
    mesh_path: str,
    scene: str,
    test_frame: int,
    alignment_dir: str,
    out_path: str,
    width: int = 2400,
    height: int = 1940,         # 1.24:1 aspect to fit panel content area
    n_points: int = 20000,      # full optimised set
    sphere_radius: float = 0.010,
    line_length: float = 0.025,
    line_radius: float = 0.0012,
    seed: int = 0,
    mesh_alpha: float = 1.0,    # 1.0 = opaque (version A); <1.0 = transparent (version B)
):
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    positions = state["positions"].numpy()
    rotations = state["rotations"].numpy()

    rng = np.random.default_rng(seed)
    n_total = len(positions)
    n_keep = min(n_points, n_total)
    idx = rng.choice(n_total, n_keep, replace=False)
    P = positions[idx]
    Q = rotations[idx]
    N = _quat_to_normal(Q)

    spheres, lines = _build_point_field(
        P, N,
        sphere_color=PINK_RGB, line_color=NORMAL_LINE_RGB,
        sphere_radius=sphere_radius,
        line_length=line_length,
        line_radius=line_radius,
    )

    r = _setup_renderer(width, height,
                          bg=(0.926, 0.922, 0.922, 1.0))

    # Mesh — lighter / less saturated than the teaser recipe so the dense
    # point cloud stays visually dominant.  When mesh_alpha < 1, the mesh
    # is rendered with the transparent shader so the points really pop.
    if os.path.exists(mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        mesh_rgb = [0.88, 0.88, 0.90]
        mesh.paint_uniform_color(mesh_rgb)
        mat_mesh = o3d.visualization.rendering.MaterialRecord()
        if mesh_alpha < 1.0:
            mat_mesh.shader = "defaultLitTransparency"
            mat_mesh.has_alpha = True
        else:
            mat_mesh.shader = "defaultLit"
        mat_mesh.base_color = [mesh_rgb[0], mesh_rgb[1], mesh_rgb[2], mesh_alpha]
        r.scene.add_geometry("mesh", mesh, mat_mesh)

    # ── Spheres (point cloud) ─────────────────────────────────────────────
    mat_pts = o3d.visualization.rendering.MaterialRecord()
    mat_pts.shader = "defaultLit"
    mat_pts.base_color = [PINK_RGB[0], PINK_RGB[1], PINK_RGB[2], 1.0]
    r.scene.add_geometry("points", spheres, mat_pts)

    # ── Normal-direction indicators ───────────────────────────────────────
    mat_lines = o3d.visualization.rendering.MaterialRecord()
    mat_lines.shader = "defaultLit"
    mat_lines.base_color = [NORMAL_LINE_RGB[0], NORMAL_LINE_RGB[1],
                              NORMAL_LINE_RGB[2], 1.0]
    r.scene.add_geometry("normals", lines, mat_lines)

    center, bore = _load_test_pose(scene, test_frame, alignment_dir)
    _setup_boresight_camera(r, center, bore,
                             back_dist=7.5, up_dist=4.2,
                             forward_dist=4.5, lookat_z_offset=-1.5,
                             fov=42.0)

    img = np.asarray(r.render_to_image())
    del r
    Image.fromarray(img).save(out_path)
    print(f"Saved: {out_path}  ({width}×{height}, n_points={n_keep})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="seq_1_frame_185")
    ap.add_argument("--test_frame", type=int, default=185)
    ap.add_argument("--repo", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_points", type=int, default=20000)
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=1940)
    ap.add_argument("--mesh_alpha", type=float, default=1.0,
                    help="1.0 = opaque mesh (variant A); <1.0 = transparent (variant B).")
    args = ap.parse_args()

    repo = args.repo
    # Resolve run_dir for the new canonical (no_occlusion, bare scene-name)
    # layout, falling back to the legacy output_frame_nvs/<scene>_..._N20000
    # layout if no_occlusion is missing.
    no_occ = os.path.join(
        repo, "mm25DGS_v5_v4", "output_ablations",
        "tier1", "lidar_init", "no_occlusion", args.scene)
    legacy = os.path.join(
        repo, "mm25DGS_v5_v4", "output_frame_nvs",
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    if os.path.isfile(os.path.join(no_occ, "best_model.pt")):
        run_dir = no_occ
    elif os.path.isfile(os.path.join(legacy, "best_model.pt")):
        run_dir = legacy
    else:
        raise SystemExit(f"No 3DPS run found at {no_occ} or {legacy}")
    print(f"  using run_dir = {run_dir}")
    model_path = os.path.join(run_dir, "best_model.pt")
    mesh_path = os.path.join(repo, "data", args.scene, "scene", "mesh.ply")
    alignment_dir = os.path.join(repo, "data", "alignment_data")

    out = args.out or os.path.join(repo, "output", "pipeline_panels",
                                     args.scene, "panel_a_scene.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    render_scene_panel(
        model_path=model_path,
        mesh_path=mesh_path,
        scene=args.scene,
        test_frame=args.test_frame,
        alignment_dir=alignment_dir,
        out_path=out,
        width=args.width,
        height=args.height,
        n_points=args.n_points,
        mesh_alpha=args.mesh_alpha,
    )


if __name__ == "__main__":
    main()
