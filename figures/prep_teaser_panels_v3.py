#!/usr/bin/env python3
"""Pre-render all panels needed for the v3 oral teaser.

Panels (saved under output/teaser_panels/<scene>/v3/):

  Top row (primitive choice):
    panel_mesh_only.png       — clean mesh render (no points), rep. of mesh-MC
    panel_implicit.png        — stylised "implicit / 3DGS" render (fuzzy blobs)
    panel_3dps_points.png     — 3DPS oriented points coloured by eps_r'

  Bottom row (applications):
    panel_ra_gt.png           — ground-truth |RA|
    panel_ra_ours.png         — 3DPS rendered |RA|
    panel_adc_gt.png          — stack of 1D GT ADC magnitude traces
    panel_adc_ours.png        — stack of 1D 3DPS-derived ADC traces
    panel_train_strip.png     — small grid of 8 training RA maps (input)
    panel_compressed_pts.png  — point cloud (the "compressed" 3DPS scene)
    panel_nvs.png             — held-out test render with trajectory hint
"""

import argparse
import ctypes
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm as mpl_cm
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    import open3d as o3d
except ImportError:
    o3d = None


# Common camera params (boresight-follow on test pose)
DEFAULT_CAM = dict(back_dist=8.0, up_dist=4.5, forward_dist=4.0,
                   lookat_z_offset=-1.5, fov=44.0)


def _setup_renderer(width, height, bg=(1, 1, 1, 0)):
    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1, 1, 1], 65000)
    r.scene.scene.enable_indirect_light(True)
    r.scene.scene.set_indirect_light_intensity(38000)
    r.scene.set_background(list(bg))
    return r


def _load_test_pose(scene, test_frame, alignment_dir):
    import json
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
# Top row: primitive renders
# ---------------------------------------------------------------------------

def render_mesh_only(mesh_path, scene, test_frame, alignment_dir, out_path,
                     width=1500, height=1000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.62, 0.62, 0.68])
    r = _setup_renderer(width, height, bg=(0.926, 0.922, 0.922, 1.0))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.62, 0.62, 0.68, 1.0]
    r.scene.add_geometry("mesh", mesh, mat)
    center, bore = _load_test_pose(scene, test_frame, alignment_dir)
    _setup_boresight_camera(r, center, bore, **DEFAULT_CAM)
    img = np.asarray(r.render_to_image())
    del r
    Image.fromarray(img).save(out_path)


def render_implicit_style(mesh_path, model_path, scene, test_frame,
                           alignment_dir, out_path,
                           width=1500, height=1000,
                           n_blobs=900, blob_radius=0.28):
    """Stylised "implicit / NeRF / 3DGS" render — large translucent blobs,
    no clear surface structure, fuzzy. Communicates "approximated, not physical."
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Subsample points from the optimised 3DPS scene to use as anchor positions
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    pos = state["positions"].numpy()
    rng = np.random.default_rng(0)
    n_blobs = min(n_blobs, len(pos))
    idx = rng.choice(len(pos), n_blobs, replace=False)
    anchor = pos[idx]

    # Random colours from a tasteful palette (warm pastels)
    cmap = mpl_cm.get_cmap("Spectral")
    blob_colors = cmap(rng.random(n_blobs))[:, :3]

    r = _setup_renderer(width, height, bg=(0.926, 0.922, 0.922, 1.0))

    # Faint mesh in background for context
    mat_mesh = o3d.visualization.rendering.MaterialRecord()
    mat_mesh.shader = "defaultLitTransparency"
    mat_mesh.base_color = [0.86, 0.86, 0.88, 0.30]
    mat_mesh.has_alpha = True
    r.scene.add_geometry("mesh", mesh, mat_mesh)

    # Translucent blobs at each anchor
    blob_mesh = o3d.geometry.TriangleMesh()
    for i, (p, col) in enumerate(zip(anchor, blob_colors)):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=blob_radius,
                                                     resolution=10)
        # Anisotropic scale to mimic 3D Gaussians
        sx = rng.uniform(0.6, 1.6)
        sy = rng.uniform(0.6, 1.6)
        sz = rng.uniform(0.6, 1.6)
        verts = np.asarray(s.vertices) * np.array([sx, sy, sz])
        s.vertices = o3d.utility.Vector3dVector(verts)
        # Random rotation
        ang = rng.uniform(0, 2 * np.pi)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        s.rotate(R, center=[0, 0, 0])
        s.translate(p)
        s.paint_uniform_color(col.tolist())
        blob_mesh += s
    blob_mesh.compute_vertex_normals()
    mat_blob = o3d.visualization.rendering.MaterialRecord()
    mat_blob.shader = "defaultLitTransparency"
    mat_blob.base_color = [1.0, 1.0, 1.0, 0.90]
    mat_blob.has_alpha = True
    r.scene.add_geometry("blobs", blob_mesh, mat_blob)

    center, bore = _load_test_pose(scene, test_frame, alignment_dir)
    _setup_boresight_camera(r, center, bore, **DEFAULT_CAM)
    img = np.asarray(r.render_to_image())
    del r
    Image.fromarray(img).save(out_path)


def render_3dps_points(mesh_path, model_path, scene, test_frame,
                        alignment_dir, out_path,
                        width=1500, height=1000):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.86, 0.86, 0.88])

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    raw = state["raw_materials"].numpy()
    pos = state["positions"].numpy()
    eps_real = 1.0 + np.log1p(np.exp(np.clip(raw[:, 0], -50, 50)))
    lo, hi = np.percentile(eps_real, [5, 95])
    eps_norm = np.clip((eps_real - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = mpl_cm.get_cmap("plasma")
    cols = cmap(eps_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    r = _setup_renderer(width, height, bg=(0.926, 0.922, 0.922, 1.0))
    mat_mesh = o3d.visualization.rendering.MaterialRecord()
    mat_mesh.shader = "defaultLitTransparency"
    mat_mesh.base_color = [0.86, 0.86, 0.88, 0.65]
    mat_mesh.has_alpha = True
    r.scene.add_geometry("mesh", mesh, mat_mesh)

    mat_pts = o3d.visualization.rendering.MaterialRecord()
    mat_pts.shader = "defaultUnlit"
    mat_pts.point_size = 4.0
    r.scene.add_geometry("points", pcd, mat_pts)

    center, bore = _load_test_pose(scene, test_frame, alignment_dir)
    _setup_boresight_camera(r, center, bore, **DEFAULT_CAM)
    img = np.asarray(r.render_to_image())
    del r
    Image.fromarray(img).save(out_path)


# ---------------------------------------------------------------------------
# Bottom row: rendering panels
# ---------------------------------------------------------------------------

def _save_ra(ra_cart, out_path, dpi=300):
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi)
    ra = np.abs(np.asarray(ra_cart, dtype=np.float64))
    mn, mx = ra.min(), ra.max()
    norm = (ra - mn) / max(mx - mn, 1e-30)
    ax.imshow(norm, cmap="hot", aspect="equal", origin="lower",
              vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0,
                facecolor="none")
    plt.close(fig)


def _save_adc_traces(adc_complex, out_path, n_traces=8, dpi=300,
                      color="#6F4E7C"):
    """Stack of 1D |ADC| traces across representative (TX,RX) pairs."""
    if adc_complex.ndim == 4:        # (chirps, rx, tx, K)  — GT raw
        adc = adc_complex[0]         # take chirp 0
        adc = np.transpose(adc, (1, 0, 2))   # (tx, rx, K)
    else:                            # (tx, rx, K) — already inverse-FFT'd
        adc = adc_complex
    n_tx, n_rx, K = adc.shape
    pairs = [(t, r) for t in range(n_tx) for r in range(n_rx)]
    rng = np.random.default_rng(7)
    rng.shuffle(pairs)
    traces = [np.abs(adc[t, r]) for (t, r) in pairs[:n_traces]]
    traces = np.array(traces)
    # Normalise globally for consistent vertical scale across traces
    traces = traces / max(traces.max(), 1e-12)

    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi,
                           facecolor="none")
    offset_step = 1.05
    x = np.arange(K)
    for i, tr in enumerate(traces):
        y = tr + i * offset_step
        ax.fill_between(x, i * offset_step, y, color=color, alpha=0.18,
                         linewidth=0)
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.95)
    ax.set_xlim(0, K)
    ax.set_ylim(-0.05, n_traces * offset_step + 0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="none", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Compression + NVS panels
# ---------------------------------------------------------------------------

def render_compressed_points(mesh_path, model_path, out_path,
                              width=900, height=900):
    """Just the optimised point cloud (no mesh) — represents 'compressed scene'."""
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    raw = state["raw_materials"].numpy()
    pos = state["positions"].numpy()
    eps_real = 1.0 + np.log1p(np.exp(np.clip(raw[:, 0], -50, 50)))
    lo, hi = np.percentile(eps_real, [5, 95])
    eps_norm = np.clip((eps_real - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = mpl_cm.get_cmap("plasma")
    cols = cmap(eps_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    r = _setup_renderer(width, height, bg=(1, 1, 1, 0.0))
    mat_pts = o3d.visualization.rendering.MaterialRecord()
    mat_pts.shader = "defaultUnlit"
    mat_pts.point_size = 5.0
    r.scene.add_geometry("points", pcd, mat_pts)

    bbox = pcd.get_axis_aligned_bounding_box()
    lookat = np.asarray(bbox.get_center())
    extent = bbox.get_extent()
    max_extent = float(np.max(extent))
    az = np.deg2rad(40); el = np.deg2rad(28)
    distance = max_extent * 1.4
    cam = lookat + np.array([
        distance * np.cos(el) * np.sin(az),
        distance * np.cos(el) * np.cos(az),
        distance * np.sin(el),
    ])
    r.setup_camera(38.0, lookat, cam, np.array([0, 0, 1.0]))
    img = np.asarray(r.render_to_image())
    del r
    Image.fromarray(img).save(out_path)


def save_train_strip(run_dir, train_frames, out_path, dpi=300, n_show=8):
    """Stacked 8-up grid of training |RA| maps as a single tall strip."""
    n = min(n_show, len(train_frames))
    fig, axes = plt.subplots(n, 1, figsize=(1.0, 0.45 * n), dpi=dpi,
                              facecolor="none")
    if n == 1:
        axes = [axes]
    for ax, F in zip(axes, train_frames[:n]):
        path = os.path.join(run_dir, "train_frames", f"frame_{F}",
                             "gt_ra_cart.npy")
        if os.path.exists(path):
            ra = np.load(path)
            mn, mx = ra.min(), ra.max()
            norm = (ra - mn) / max(mx - mn, 1e-30)
            ax.imshow(norm, cmap="hot", aspect="equal", origin="lower",
                      vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0.04)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0,
                facecolor="none", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="seq_1_frame_185")
    p.add_argument("--test_frame", type=int, default=185)
    p.add_argument("--train_frames", type=int, nargs="+",
                   default=[181, 182, 183, 184, 186, 187, 188, 189])
    p.add_argument("--ours_dir",
                   default=os.path.join(_REPO, "mm25DGS_v5_v4/output_frame_nvs"))
    p.add_argument("--data_dir", default=os.path.join(_REPO, "data"))
    p.add_argument("--alignment_dir",
                   default=os.path.join(_REPO, "data/alignment_data"))
    p.add_argument("--output_dir",
                   default=os.path.join(_REPO, "output/teaser_panels"))
    args = p.parse_args()

    run_dir = os.path.join(args.ours_dir,
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    if not os.path.isdir(run_dir):
        sys.exit(f"Run dir not found: {run_dir}")

    out_dir = os.path.join(args.output_dir, args.scene, "v3")
    os.makedirs(out_dir, exist_ok=True)
    mesh_path = os.path.join(args.data_dir, args.scene, "scene", "mesh.ply")
    model_path = os.path.join(run_dir, "best_model.pt")

    # Top row
    print("Rendering mesh-only…")
    render_mesh_only(mesh_path, args.scene, args.test_frame, args.alignment_dir,
                      os.path.join(out_dir, "panel_mesh_only.png"))
    print("Rendering implicit/3DGS-style…")
    render_implicit_style(mesh_path, model_path, args.scene, args.test_frame,
                           args.alignment_dir,
                           os.path.join(out_dir, "panel_implicit.png"))
    print("Rendering 3DPS oriented points…")
    render_3dps_points(mesh_path, model_path, args.scene, args.test_frame,
                        args.alignment_dir,
                        os.path.join(out_dir, "panel_3dps_points.png"))

    # Bottom — Rendering: GT vs 3DPS for both RA and ADC
    print("Saving RA panels…")
    gt_ra = np.load(os.path.join(run_dir, "gt_test_ra_cart.npy"))
    pred_ra = np.load(os.path.join(run_dir, "rendered_test_ra_cart.npy"))
    _save_ra(gt_ra, os.path.join(out_dir, "panel_ra_gt.png"))
    _save_ra(pred_ra, os.path.join(out_dir, "panel_ra_ours.png"))

    print("Saving ADC trace panels…")
    rp_complex = np.load(os.path.join(run_dir, "rendered_test_rp_complex.npy"))
    pred_adc = np.fft.ifft(rp_complex, axis=-1)
    _save_adc_traces(pred_adc, os.path.join(out_dir, "panel_adc_ours.png"),
                      color="#0e6b2c")
    raw_adc_path = os.path.join(args.data_dir, args.scene, "radar",
                                 f"cascaded_frame_{args.test_frame}.npy")
    if os.path.exists(raw_adc_path):
        gt_adc = np.load(raw_adc_path)
        _save_adc_traces(gt_adc, os.path.join(out_dir, "panel_adc_gt.png"),
                          color="#444444")
    else:
        print(f"  WARN: no GT ADC at {raw_adc_path}")

    # Compression: train strip + compressed points
    print("Saving training-RA strip + compressed point cloud…")
    save_train_strip(run_dir, args.train_frames,
                      os.path.join(out_dir, "panel_train_strip.png"))
    render_compressed_points(mesh_path, model_path,
                              os.path.join(out_dir, "panel_compressed_pts.png"))

    # NVS — for the third application column we'll just reuse the held-out
    # |RA| render (already saved as panel_ra_ours.png) inside the figure
    # composition, plus a pose-trajectory diagram. We'll generate the latter
    # in the figure script directly using matplotlib.

    print(f"\nAll v3 panels under: {out_dir}/")


if __name__ == "__main__":
    main()
