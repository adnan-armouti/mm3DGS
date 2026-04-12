"""
Stage C: Train Gaussian surfels using the PyTorch rasterizer.

C2: Initialize Gaussians from mesh vertices with mmIR materials
C3: Train Gaussians from mmIR initialization
C4: Train Gaussians from scratch (LiDAR initialization)
C5: With density control

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v2.train_gaussian --scene seq_0_frame_135 --mode c3
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v2.train_gaussian --all --mode c4
"""

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmir.data.ra_utils import (
    adc_to_ra_image, adc_to_ra_complex, ra_polar_to_cartesian,
    compute_cartesian_ra_metrics, txrx_to_vx_chirps_torch,
)
from mmir.data.io_utils import compute_range_res_from_cfg
from mm25DGS_v2.render_mmIR import load_trained_config, load_best_params, TRAIN_OUTPUT_DIR
from mm25DGS_v2.rasterizer_torch import (
    RasterizerTorch, reparameterize_torch, inverse_reparameterize_torch,
)
from mm25DGS_v2.train_mesh import get_lr_scale, rms_clip_grad

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]

DEVICE = "cuda:0"


# =========================================================================
# Range profile → RA image (azimuth FFT only, no range FFT)
# =========================================================================

def range_profile_to_ra(rp_real, rp_imag):
    """Convert complex range profiles to RA image via azimuth FFT only.

    Input: (n_tx=12, n_rx=16, K=256) — complex range profiles per TX-RX channel.
    Output: (127, 256) — RA magnitude image, same as adc_to_ra_image output.

    This skips the range FFT (already done by splatting) and applies only:
    1. Virtual array rearrangement (TX-RX → 86 virtual elements)
    2. Azimuth Hann window + FFT to 128 bins
    3. Drop first azimuth bin → (127, 256)
    """
    # Stack to (n_tx, n_rx, K, 2) then permute to (RX, TX, K, 2) for vx mapping
    rp_ri = torch.stack([rp_real, rp_imag], dim=-1)              # (12, 16, K, 2)
    rp_ri = rp_ri.permute(1, 0, 2, 3)                            # (RX=16, TX=12, K=256, 2)
    rp_c = torch.complex(rp_ri[..., 0].contiguous(),
                          rp_ri[..., 1].contiguous())              # (16, 12, 256) complex
    rp_c = rp_c.unsqueeze(0)                                      # (1, 16, 12, 256)

    # Virtual array rearrangement: (1, 16, 12, 256) → (1, 7, 86, 256)
    vx = txrx_to_vx_chirps_torch(rp_c)
    ra = vx[0, 0, :, :]                                           # (86, 256) — elevation row 0

    # NOTE: no range FFT here — range profiles are already in range domain

    # Azimuth window and FFT (same as adc_to_ra_image lines 58-63)
    num_vx = ra.size(0)
    ra = ra * torch.hann_window(num_vx, device=ra.device, dtype=ra.real.dtype).to(ra.dtype)[:, None]
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]                                                 # drop first azimuth bin
    ra = torch.fft.fftshift(ra, dim=0)

    return ra  # (127, 256) complex


def range_profile_to_ra_mag(rp_real, rp_imag):
    """Range profiles → RA magnitude image (float)."""
    return range_profile_to_ra(rp_real, rp_imag).abs().float()


def compute_ra_loss_rp(rp_real, rp_imag, gt_adc_ri):
    """Compute RA loss from range profiles vs GT ADC.

    The rendered range profiles go through azimuth FFT only.
    The GT ADC goes through the full pipeline (range FFT + azimuth FFT).
    """
    # Rendered: range profile → azimuth FFT → RA
    ra_rendered = range_profile_to_ra(rp_real, rp_imag)
    ra_rend_mag = torch.abs(ra_rendered)

    # GT: ADC → full pipeline → RA
    ra_gt = adc_to_ra_complex(gt_adc_ri)
    ra_gt_mag = torch.abs(ra_gt)

    # Min-max normalize (separate, linear, no log)
    def _mm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn) if mx - mn > 1e-30 else torch.zeros_like(x)

    loss = torch.mean((_mm(ra_rend_mag) - _mm(ra_gt_mag)) ** 2)
    return loss, {"ra_mse": loss.item()}
ITU_CONCRETE = np.array([5.31, 0.0326, 1e-4, 5e-3, 0.5, 0.15], dtype=np.float32)


# =========================================================================
# Gaussian Surfel model (C1)
# =========================================================================

class GaussianSurfels(torch.nn.Module):
    """Gaussian surfel model for radar rendering.

    Each surfel has: position(3), quaternion(4), log_scales(2),
    logit_opacity(1), raw_materials(6). Total: 16 params.
    """

    def __init__(self, N, device=DEVICE):
        super().__init__()
        self.device = device
        self.positions = torch.nn.Parameter(torch.zeros(N, 3, device=device))
        self.rotations = torch.nn.Parameter(torch.zeros(N, 4, device=device))
        self.log_scales = torch.nn.Parameter(torch.zeros(N, 2, device=device))
        self.logit_opacities = torch.nn.Parameter(torch.zeros(N, 1, device=device))
        self.raw_materials = torch.nn.Parameter(torch.zeros(N, 6, device=device))

        with torch.no_grad():
            self.rotations[:, 0] = 1.0  # identity quaternion [w,x,y,z]

    @property
    def N(self):
        return self.positions.shape[0]

    def get_normals(self):
        """(N, 3) surface normals from quaternion."""
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # Third column of rotation matrix
        nx = 2 * (x * z + w * y)
        ny = 2 * (y * z - w * x)
        nz = 1 - 2 * (x * x + y * y)
        return torch.stack([nx, ny, nz], dim=-1)

    def get_opacities(self):
        """(N,) opacities in [0, 1]."""
        return torch.sigmoid(self.logit_opacities.squeeze(-1))

    def get_scales(self):
        """(N, 2) positive lateral scales."""
        return torch.exp(self.log_scales)


# =========================================================================
# Initialization (C2)
# =========================================================================

def init_from_mesh(scene, device=DEVICE):
    """Initialize Gaussians at mesh vertices with mmIR trained materials.

    One Gaussian per vertex: position=vertex, normal from mesh,
    materials from mmIR, opacity=0.5.
    """
    config = load_trained_config(scene)
    raw_params, normal_params, pattern_data = load_best_params(scene)

    import trimesh
    mesh = trimesh.load(config.scene_file)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    N = len(vertices)

    if normal_params is not None:
        normals = normal_params.astype(np.float32)
    else:
        normals = np.array(mesh.vertex_normals, dtype=np.float32)

    model = GaussianSurfels(N, device=device)

    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(vertices).to(device))
        model.raw_materials.copy_(torch.from_numpy(raw_params.astype(np.float32)).to(device))

        # Quaternion from normal
        quats = _normals_to_quaternions(normals)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

        # Scales and vertex areas from face areas
        scales, vertex_areas = _compute_scales_and_areas(vertices, normals, faces, mesh)
        model.log_scales.copy_(torch.from_numpy(np.log(scales)).to(device))

        # Opacity = 1.0 -> logit = large positive (so opacity*area = area)
        model.logit_opacities.fill_(5.0)  # sigmoid(5) ~ 0.993

    vertex_areas_t = torch.from_numpy(vertex_areas).to(device)
    return model, config, pattern_data, vertex_areas_t


def init_from_lidar(scene, target_n=None, device=DEVICE):
    """Initialize Gaussians from LiDAR point cloud with ITU concrete defaults.

    Uses FPS downsampling and mesh normals at nearest vertices.
    """
    config = load_trained_config(scene)
    _, _, pattern_data = load_best_params(scene)

    import trimesh
    mesh = trimesh.load(config.scene_file)
    mesh_verts = np.array(mesh.vertices, dtype=np.float32)
    mesh_normals = np.array(mesh.vertex_normals, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    if target_n is None:
        target_n = len(mesh_verts)  # same count as mesh

    # Load point cloud
    pcl_path = config.scene_file.replace('scene/mesh.ply', 'scene/pcl.npy')
    if os.path.exists(pcl_path):
        pcl = np.load(pcl_path)
        xyz = pcl[:, :3].astype(np.float32)
    else:
        # Fall back to mesh vertices
        xyz = mesh_verts.copy()

    # FPS downsample
    if len(xyz) > target_n:
        pts_t = torch.from_numpy(xyz).to(device)
        selected = _farthest_point_sampling(pts_t, target_n)
        xyz = xyz[selected.cpu().numpy()]

    N = len(xyz)

    # Get normals from nearest mesh vertex
    tree = KDTree(mesh_verts)
    _, idx = tree.query(xyz)
    normals = mesh_normals[idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)

    model = GaussianSurfels(N, device=device)

    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

        # ITU concrete defaults
        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        model.raw_materials.copy_(
            torch.from_numpy(np.tile(raw_default, (N, 1))).to(device))

        # Quaternion from normal
        quats = _normals_to_quaternions(normals)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

        # Scales from k-NN PCA (fully vectorized)
        tree2 = KDTree(xyz)
        k_nn = min(20, N - 1)
        _, knn_idx = tree2.query(xyz, k=k_nn + 1)
        knn_idx = knn_idx[:, 1:]  # (N, k_nn) exclude self

        # Neighbor offsets: (N, k_nn, 3)
        neighbors = xyz[knn_idx] - xyz[:, None, :]

        # Project onto tangent plane: remove normal component
        # proj = neighbors - (neighbors · n) × n
        n_exp = normals[:, None, :]                          # (N, 1, 3)
        dot_n = (neighbors * n_exp).sum(axis=-1, keepdims=True)  # (N, k_nn, 1)
        proj = neighbors - dot_n * n_exp                     # (N, k_nn, 3)

        # Covariance: (N, 3, 3) = projᵀ @ proj / k_nn
        # Use einsum for batch matmul: (N, 3, k_nn) @ (N, k_nn, 3) -> (N, 3, 3)
        cov = np.einsum('nki,nkj->nij', proj, proj) / k_nn  # (N, 3, 3)

        # Batch eigendecomposition
        eigvals = np.linalg.eigvalsh(cov)                    # (N, 3) sorted ascending

        # Scales = sqrt of two largest eigenvalues, clamped
        scales = np.zeros((N, 2), dtype=np.float32)
        scales[:, 0] = np.clip(np.sqrt(np.maximum(eigvals[:, 2], 1e-12)), 0.01, 0.5)
        scales[:, 1] = np.clip(np.sqrt(np.maximum(eigvals[:, 1], 1e-12)), 0.01, 0.5)
        model.log_scales.copy_(torch.from_numpy(np.log(scales)).to(device))

        model.logit_opacities.fill_(0.0)

    # Uniform areas for LiDAR init
    vertex_areas_t = torch.ones(N, device=device)
    return model, config, pattern_data, vertex_areas_t


def _ray_test_visibility_batched(points_np, rx_center_np, mi_scene, chunk=200000):
    """Batched ray test from points toward RX center. Returns visible mask."""
    import mitsuba as mi
    import drjit as dr
    N = len(points_np)
    visible = np.zeros(N, dtype=bool)
    eps = 1e-3
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        pts = points_np[i:j]
        delta = rx_center_np - pts                             # (n, 3)
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)    # (n,)
        direction = delta / dist[:, None]                       # (n, 3)
        origins = pts + eps * direction
        o = mi.Point3f(
            mi.Float(origins[:, 0]), mi.Float(origins[:, 1]), mi.Float(origins[:, 2]))
        d = mi.Vector3f(
            mi.Float(direction[:, 0]), mi.Float(direction[:, 1]), mi.Float(direction[:, 2]))
        rays = mi.Ray3f(o, d)
        rays.maxt = mi.Float(dist - 2 * eps)
        occluded = mi_scene.ray_test(rays)
        visible[i:j] = ~np.array(occluded)
    return visible


def init_from_lidar_visible_weighted(scene, rast, target_n=None,
                                     cos_bore_min=0.1,
                                     n_intermediate=200000,
                                     device=DEVICE):
    """Initialize LiDAR FPS Gaussians restricted to RX-visible, cosine-weighted.

    Pipeline:
      1. Load pcl (1.5-2.5M points)
      2. FOV restrict: cos_bore > cos_bore_min, within radar range
      3. RX-side ray-traced visibility (non-occluded only)
      4. Cosine-hemisphere importance resample (prob ∝ cos_bore)
      5. FPS on the resampled set → target_n points

    Requires rast with _mi_scene available (call before freeing Mitsuba).
    """
    config = load_trained_config(scene)
    _, _, pattern_data = load_best_params(scene)

    import trimesh
    mesh = trimesh.load(config.scene_file)
    mesh_verts = np.array(mesh.vertices, dtype=np.float32)
    mesh_normals = np.array(mesh.vertex_normals, dtype=np.float32)

    if target_n is None:
        target_n = len(mesh_verts)

    # Load full point cloud
    pcl_path = config.scene_file.replace('scene/mesh.ply', 'scene/pcl.npy')
    if os.path.exists(pcl_path):
        pcl = np.load(pcl_path)
        xyz_full = pcl[:, :3].astype(np.float32)
    else:
        xyz_full = mesh_verts.copy()

    print(f"  [visible-weighted init] Full pcl: {len(xyz_full)} points")

    # --- Step 1: FOV restrict ---
    rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()
    boresight = rast.tx_boresights.mean(dim=0).cpu().numpy()
    boresight = boresight / max(np.linalg.norm(boresight), 1e-8)

    delta = xyz_full - rx_center
    dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
    dir_to_point = delta / dist[:, None]
    cos_bore_full = (dir_to_point * boresight).sum(axis=-1)

    # Max range from FMCW config
    max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

    fov_mask = (cos_bore_full > cos_bore_min) & (dist > 1.5) & (dist < max_range)
    xyz_fov = xyz_full[fov_mask]
    cos_bore_fov = cos_bore_full[fov_mask]
    print(f"  [visible-weighted init] After FOV: {len(xyz_fov)} points")

    # --- Step 2: RX visibility ray tracing ---
    # Force-load the Mitsuba scene if not yet loaded
    if not hasattr(rast, '_mi_scene') or rast._mi_scene is None:
        from mmir.renderer.scene_context import SceneContext
        scene_ctx = SceneContext.from_files(
            config_file=rast._config_file, scene_file=rast._mesh_file,
            pattern_file=None,
            tx_pattern_file=rast._tx_pattern_file,
            rx_pattern_file=rast._rx_pattern_file,
            material_type="metal", enable_gradients=False, verbose=False,
        )
        rast._mi_scene = scene_ctx.scene

    visible = _ray_test_visibility_batched(xyz_fov, rx_center, rast._mi_scene)
    xyz_vis = xyz_fov[visible]
    cos_bore_vis = cos_bore_fov[visible]
    print(f"  [visible-weighted init] After visibility: {len(xyz_vis)} points")

    if len(xyz_vis) < 100:
        print(f"  [visible-weighted init] WARNING: very few visible points, falling back")
        return init_from_lidar(scene, target_n=target_n, device=device)

    # --- Step 3: Cosine-hemisphere importance resample ---
    # Clip cos_bore to avoid zero probabilities at horizon
    weights = np.maximum(cos_bore_vis, 0.01)
    probs = weights / weights.sum()
    n_resample = min(n_intermediate, 3 * target_n)
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(xyz_vis), size=n_resample, replace=True, p=probs)
    unique_idx = np.unique(sampled_idx)
    xyz_weighted = xyz_vis[unique_idx]
    print(f"  [visible-weighted init] After cosine importance resample: {len(xyz_weighted)} points")

    # --- Step 4: FPS to target_n ---
    if len(xyz_weighted) > target_n:
        pts_t = torch.from_numpy(xyz_weighted).to(device)
        selected = _farthest_point_sampling(pts_t, target_n)
        xyz = xyz_weighted[selected.cpu().numpy()]
    else:
        xyz = xyz_weighted
    print(f"  [visible-weighted init] After FPS: {len(xyz)} points")

    N = len(xyz)

    # --- Normals from nearest mesh vertex ---
    tree = KDTree(mesh_verts)
    _, idx = tree.query(xyz)
    normals = mesh_normals[idx]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)

    model = GaussianSurfels(N, device=device)

    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        model.raw_materials.copy_(
            torch.from_numpy(np.tile(raw_default, (N, 1))).to(device))

        quats = _normals_to_quaternions(normals)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

        # Scales from k-NN PCA
        tree2 = KDTree(xyz)
        k_nn = min(20, N - 1)
        _, knn_idx = tree2.query(xyz, k=k_nn + 1)
        knn_idx = knn_idx[:, 1:]
        neighbors = xyz[knn_idx] - xyz[:, None, :]
        n_exp = normals[:, None, :]
        dot_n = (neighbors * n_exp).sum(axis=-1, keepdims=True)
        proj = neighbors - dot_n * n_exp
        cov = np.einsum('nki,nkj->nij', proj, proj) / k_nn
        eigvals = np.linalg.eigvalsh(cov)
        scales = np.zeros((N, 2), dtype=np.float32)
        scales[:, 0] = np.clip(np.sqrt(np.maximum(eigvals[:, 2], 1e-12)), 0.01, 0.5)
        scales[:, 1] = np.clip(np.sqrt(np.maximum(eigvals[:, 1], 1e-12)), 0.01, 0.5)
        model.log_scales.copy_(torch.from_numpy(np.log(scales)).to(device))

        model.logit_opacities.fill_(0.0)

    vertex_areas_t = torch.ones(N, device=device)
    return model, config, pattern_data, vertex_areas_t


def _normals_to_quaternions(normals):
    """Convert normal vectors to quaternions [w,x,y,z]. Fully vectorized."""
    N = len(normals)
    # Normalize
    nrm = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    n = normals / nrm

    # Rotation from up=(0,0,1) to n
    # dot = n[:, 2]  (since up = [0,0,1])
    dot = n[:, 2].copy()

    # axis = cross(up, n) = [-n_y, n_x, 0]
    axis = np.zeros((N, 3), dtype=np.float32)
    axis[:, 0] = -n[:, 1]
    axis[:, 1] = n[:, 0]
    # axis[:, 2] = 0 already

    axis_len = np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    axis = axis / axis_len

    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    half_angle = angle / 2.0

    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = np.cos(half_angle)
    quats[:, 1] = axis[:, 0] * np.sin(half_angle)
    quats[:, 2] = axis[:, 1] * np.sin(half_angle)
    quats[:, 3] = axis[:, 2] * np.sin(half_angle)

    # Handle near-parallel (dot > 0.9999): identity quaternion
    parallel = dot > 0.9999
    quats[parallel] = [1, 0, 0, 0]

    # Handle near-antiparallel (dot < -0.9999): 180° around x
    anti = dot < -0.9999
    quats[anti] = [0, 1, 0, 0]

    return quats


def _compute_scales_and_areas(vertices, normals, faces, mesh):
    """Compute per-vertex scales and areas from face areas."""
    N = len(vertices)
    face_areas = mesh.area_faces
    vertex_areas = np.zeros(N, dtype=np.float32)
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], face_areas / 3.0)
    # Scale ~ sqrt(area)
    scales = np.sqrt(np.maximum(vertex_areas, 1e-12))
    scales = np.clip(scales, 0.01, 0.5)
    return np.column_stack([scales, scales]), vertex_areas


def _farthest_point_sampling(pts, n_samples):
    """GPU FPS."""
    N = pts.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=pts.device)

    selected = torch.empty(n_samples, dtype=torch.long, device=pts.device)
    min_dist = torch.full((N,), float("inf"), device=pts.device)

    centroid = pts.mean(dim=0, keepdim=True)
    first_idx = torch.argmin(torch.norm(pts - centroid, dim=1))
    selected[0] = first_idx
    min_dist = torch.norm(pts - pts[first_idx], dim=1)
    min_dist[first_idx] = 0.0

    for j in range(1, n_samples):
        idx = torch.argmax(min_dist)
        selected[j] = idx
        d = torch.norm(pts - pts[idx], dim=1)
        min_dist = torch.minimum(min_dist, d)
        min_dist[idx] = 0.0

    return selected


# =========================================================================
# Gaussian renderer adapter
# =========================================================================

def _get_visible_vertices(hits, faces):
    """Get unique vertex IDs from reservoir sampler hits."""
    import drjit as dr
    prim_ids = np.unique(np.array(hits.hit_prim_ids))
    vis_verts = np.unique(faces[prim_ids].ravel())
    return vis_verts


def _compute_hit_frequency_weights(hits, model, rast, device='cuda:0'):
    """Compute per-Gaussian importance weights from reservoir hit frequency.

    For each Gaussian, count how many reservoir hits are nearby (within the
    Gaussian's Voronoi cell). This count approximates the MC sampling density
    at that location. Gaussians in heavily-sampled regions get higher weight,
    matching the 1/(pdf × N_attempted) correction that mmIR uses.

    Returns: (N,) tensor of importance weights on device.
    """
    from scipy.spatial import KDTree

    # Hit positions from reservoir sampler
    hit_P = np.column_stack([
        np.array(hits.hit_P.x),
        np.array(hits.hit_P.y),
        np.array(hits.hit_P.z),
    ]).astype(np.float32)
    n_hits = hit_P.shape[0]

    # Gaussian positions
    gauss_pos = model.positions.detach().cpu().numpy().astype(np.float32)
    N = gauss_pos.shape[0]

    # Build KDTree on Gaussian positions, query each hit's nearest Gaussian
    tree = KDTree(gauss_pos)
    _, nearest_gauss = tree.query(hit_P)  # (n_hits,) index of nearest Gaussian

    # Count hits per Gaussian
    hit_counts = np.bincount(nearest_gauss, minlength=N).astype(np.float32)

    # Normalize: mean weight = 1 (so the overall amplitude scale is unchanged)
    mean_count = hit_counts[hit_counts > 0].mean() if (hit_counts > 0).any() else 1.0
    importance = hit_counts / max(mean_count, 1e-10)

    # Gaussians with zero hits get weight 0 (they're not radar-visible)
    # But keep a small floor so gradients can still flow to move them
    importance = np.maximum(importance, 0.01)

    return torch.from_numpy(importance).to(device)


def _compute_shadow_mask(positions, rast, device='cuda:0'):
    """Pre-compute per-(Gaussian, TX) shadow mask using Mitsuba ray test.

    For each Gaussian × TX pair, casts a shadow ray from the Gaussian toward
    the TX element. If the ray hits another surface before reaching the TX,
    that path is marked as occluded.

    Args:
        positions: (M, 3) Gaussian positions on device
        rast: RasterizerTorch with _mi_scene set

    Returns:
        (M, n_tx) bool tensor — True = visible, False = occluded
    """
    import mitsuba as mi
    import drjit as dr

    if not hasattr(rast, '_mi_scene') or rast._mi_scene is None:
        # No scene available — assume all visible
        return torch.ones(positions.shape[0], rast.n_tx, dtype=torch.bool, device=device)

    M = positions.shape[0]
    n_tx = rast.n_tx
    pos_np = positions.detach().cpu().numpy()
    tx_pos_np = rast.tx_positions.cpu().numpy()

    shadow_mask = np.ones((M, n_tx), dtype=bool)

    for t in range(n_tx):
        # Direction from Gaussian to TX
        delta = tx_pos_np[t] - pos_np                               # (M, 3)
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)         # (M,)
        direction = delta / dist[:, None]                            # (M, 3)

        # Shadow ray: offset origin slightly along direction to avoid self-hit
        eps = 1e-4
        origins = pos_np + eps * direction

        # Create Mitsuba rays
        o = mi.Point3f(mi.Float(origins[:, 0]), mi.Float(origins[:, 1]), mi.Float(origins[:, 2]))
        d = mi.Vector3f(mi.Float(direction[:, 0]), mi.Float(direction[:, 1]), mi.Float(direction[:, 2]))
        rays = mi.Ray3f(o, d)
        rays.maxt = mi.Float(dist - 2 * eps)

        occluded = rast._mi_scene.ray_test(rays)
        shadow_mask[:, t] = ~np.array(occluded)

    return torch.from_numpy(shadow_mask).to(device)


def _compute_rx_visibility(positions, rast, device='cuda:0'):
    """Test RX-side visibility for each Gaussian using Mitsuba ray tracing.

    Casts a ray from each Gaussian toward the RX array center. If the ray
    hits another surface before reaching the RX, the Gaussian is occluded.

    Returns:
        (M,) bool tensor — True = visible from RX, False = occluded
    """
    import mitsuba as mi
    import drjit as dr

    if not hasattr(rast, '_mi_scene') or rast._mi_scene is None:
        return torch.ones(positions.shape[0], dtype=torch.bool, device=device)

    pos_np = positions.detach().cpu().numpy()
    rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()

    delta = rx_center - pos_np                                     # (M, 3)
    dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)           # (M,)
    direction = delta / dist[:, None]                               # (M, 3)

    eps = 1e-3
    origins = pos_np + eps * direction

    o = mi.Point3f(mi.Float(origins[:, 0]), mi.Float(origins[:, 1]), mi.Float(origins[:, 2]))
    d = mi.Vector3f(mi.Float(direction[:, 0]), mi.Float(direction[:, 1]), mi.Float(direction[:, 2]))
    rays = mi.Ray3f(o, d)
    rays.maxt = mi.Float(dist - 2 * eps)

    occluded = rast._mi_scene.ray_test(rays)
    visible = ~np.array(occluded)
    return torch.from_numpy(visible).to(device)


def cull_gaussians(model, rast, cos_threshold=0.05):
    """Pre-filter Gaussians to only those visible to the radar.

    Applies range, FOV angle, and cosine filtering to match what the
    reservoir sampler would select.
    """
    with torch.no_grad():
        positions = model.positions
        normals = model.get_normals()

        # Radar center and boresight
        radar_center = (rast.tx_positions.mean(0) + rast.rx_positions.mean(0)) / 2
        boresight = rast.tx_boresights.mean(0)
        boresight = boresight / boresight.norm().clamp(min=1e-6)

        # Direction from vertex to radar
        to_radar = radar_center - positions
        dist = to_radar.norm(dim=-1)
        to_radar_dir = to_radar / dist.clamp(min=1e-6).unsqueeze(-1)

        # Cosine of normal with radar direction (double-sided)
        cos_normal = torch.abs((normals * to_radar_dir).sum(-1))

        # FOV angle: dot product of direction-to-vertex with boresight
        from_radar = -to_radar_dir
        cos_bore = (from_radar * boresight).sum(-1)

        # Max range from FMCW parameters
        max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

        active = (
            (dist > 1.5) &                   # not in near field
            (dist < max_range) &              # within max range
            (cos_bore > 0.0) &                # in front hemisphere
            (cos_normal > cos_threshold)      # somewhat facing radar
        )

        return active


def _compute_density_ratio_weights(positions, rast, k=8):
    """Compute hemisphere MC weight via density-ratio reweighting.

    For each Gaussian, computes: w = p(ω) / q(ω)
    where p = cos_bore/π (cosine hemisphere prior, mmIR's target distribution)
    and q = local angular density of FPS points on the RX sphere (from k-NN).

    This converts FPS samples into effective cosine-hemisphere samples,
    matching mmIR's MC weight chain without ray tracing.
    """
    from scipy.spatial import cKDTree

    rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()
    pos_np = positions.detach().cpu().numpy()
    bore_np = rast.rx_boresights.mean(dim=0).cpu().numpy()  # average boresight
    bore_np = bore_np / max(np.linalg.norm(bore_np), 1e-8)

    # Project onto unit sphere around RX center
    diff = pos_np - rx_center
    dist = np.linalg.norm(diff, axis=-1, keepdims=True).clip(min=1e-6)
    omega = diff / dist  # (M, 3) unit directions

    # cos_bore for each Gaussian
    cos_bore = (omega * bore_np).sum(axis=-1).clip(min=0.1)  # (M,)

    # Estimate actual angular density q(ω) via k-NN on the unit sphere
    M = len(omega)
    k_use = min(k, M - 1)
    tree = cKDTree(omega)
    dists_knn, _ = tree.query(omega, k=k_use + 1)
    theta_k = dists_knn[:, k_use]  # chord distance to k-th neighbor ≈ angle

    # q(ω_i) ≈ k / (M × π × θ_k²)
    # p(ω_i) = cos_bore_i / π
    # w_i = p/q = cos_bore_i × M × θ_k² / k
    w = cos_bore * M * theta_k ** 2 / k_use

    return torch.from_numpy(w.astype(np.float32)).to(positions.device)


def render_gaussians_factorized(model, rast, vertex_areas=None,
                                detach_phase=True, active_mask=None,
                                chunk_size=2000, importance_weights=None,
                                shadow_mask=None):
    """Range-profile splatting renderer. Returns (rp_real, rp_imag).

    Output is complex range profiles (n_tx, n_rx, K), NOT ADC.
    Use range_profile_to_ra() or compute_ra_loss_rp() for RA conversion.

    Args:
        importance_weights: (N,) per-Gaussian importance (Option 2B). Currently unused.
        shadow_mask: (N, n_tx) bool — pre-computed TX shadow visibility (F7).
        hemisphere_weights: If True, use density-ratio MC weights instead of areas.
    """
    from mm25DGS_v3.rasterizer_factorized import render_factorized
    from mm25DGS_v2.rasterizer_torch import reparameterize_torch

    positions = model.positions
    normals = model.get_normals()
    opacities = model.get_opacities()
    raw_materials = model.raw_materials

    # vertex_areas carries either surface areas (c4) or precomputed
    # hemisphere density-ratio weights (c6). Both get multiplied by opacity.
    if vertex_areas is not None:
        areas = vertex_areas * opacities
    else:
        areas = opacities

    # Option 2B: apply reservoir hit-frequency importance weights
    if importance_weights is not None:
        areas = areas * importance_weights

    # Slice shadow mask by active_mask
    sm = None
    if shadow_mask is not None and active_mask is not None:
        sm = shadow_mask[active_mask]
    elif shadow_mask is not None:
        sm = shadow_mask

    if active_mask is not None:
        positions = positions[active_mask]
        normals = normals[active_mask]
        raw_materials = raw_materials[active_mask]
        areas = areas[active_mask]

    return render_factorized(
        positions, normals, areas, raw_materials, rast,
        reparameterize_torch, detach_phase=detach_phase,
        chunk_size=chunk_size, shadow_mask=sm)


# =========================================================================
# Training loop
# =========================================================================

def train_gaussians(scene, mode='c3', num_iters=500, target_n=None, verbose=True):
    """Train Gaussian surfels.

    Args:
        mode: 'c2' = init only (no training), 'c3' = from mmIR init,
              'c4' = from scratch (LiDAR), 'c5' = from scratch with density control
        num_iters: Training iterations
        target_n: Number of Gaussians for LiDAR init (None = mesh count)
    """
    use_hemisphere = (mode == 'c6')

    # Initialize
    if mode in ('c2', 'c3'):
        model, config, pattern_data, vertex_areas = init_from_mesh(scene)
        rast = RasterizerTorch(
            config_file=config.config_file,
            mesh_file=config.scene_file,
            tx_pattern_file=config.tx_pattern_file,
            rx_pattern_file=config.rx_pattern_file,
            device=DEVICE,
        )
    elif mode in ('c4', 'c5'):
        model, config, pattern_data, vertex_areas = init_from_lidar(scene, target_n=target_n)
        rast = RasterizerTorch(
            config_file=config.config_file,
            mesh_file=config.scene_file,
            tx_pattern_file=config.tx_pattern_file,
            rx_pattern_file=config.rx_pattern_file,
            device=DEVICE,
        )
    elif mode == 'c6':
        # c6 needs rast available during init for visibility + cosine weighting
        config = load_trained_config(scene)
        rast = RasterizerTorch(
            config_file=config.config_file,
            mesh_file=config.scene_file,
            tx_pattern_file=config.tx_pattern_file,
            rx_pattern_file=config.rx_pattern_file,
            device=DEVICE,
        )
        model, config, pattern_data, vertex_areas = init_from_lidar_visible_weighted(
            scene, rast, target_n=target_n)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Inject patterns from mmIR (for antenna patterns only)
    raw_params_mmIR, _, _ = load_best_params(scene)
    rast.inject_trained_params(raw_params_mmIR, None, pattern_data)

    # Use reservoir sampler to identify visible vertices AND compute
    # per-Gaussian importance weights (Option 2B).
    hits = rast._run_reservoir_sampler(seed=42)
    visible_verts = _get_visible_vertices(hits, rast.faces_np)
    visible_mask = torch.zeros(model.N, dtype=torch.bool, device=DEVICE)
    visible_mask[torch.from_numpy(visible_verts).long().to(DEVICE)] = True

    # Option 2B: Compute per-Gaussian importance weights (currently disabled).
    hit_importance = None

    # C6 hemisphere: compute RX-side visibility + density-ratio weights
    # (must be done before freeing Mitsuba scene)
    rx_visible = None
    if use_hemisphere:
        rx_visible = _compute_rx_visibility(model.positions, rast, device=DEVICE)
        if verbose:
            print(f"  RX-visible Gaussians: {rx_visible.sum().item()}/{model.N}")

    # F7: Pre-compute shadow mask using Mitsuba scene (before freeing it).
    # For each (Gaussian, TX) pair, test if the line-of-sight is occluded.
    shadow = _compute_shadow_mask(model.positions, rast, device=DEVICE)

    # Free Mitsuba scene to reclaim GPU memory for training
    del hits
    rast._mi_scene = None
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # GT ADC
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_adc_ri = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32)).to(DEVICE)

    range_res = compute_range_res_from_cfg(config.config_file)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Gaussian Training: {scene} (mode={mode})")
        print(f"  Gaussians: {model.N}")
        print(f"  Iterations: {num_iters}")
        print(f"{'='*60}")

    # Pre-compute active mask using reservoir visibility + FOV culling.
    # For mesh-init (C2/C3), visible_mask identifies radar-visible vertices.
    # For LiDAR-init (C4/C5), we find the nearest visible mesh vertex for each
    # Gaussian and keep those within a distance threshold.
    if mode in ('c2', 'c3'):
        active_mask = visible_mask
    elif use_hemisphere:
        # C6: RX-side ray-traced visibility + FOV culling (no occluded Gaussians)
        active_mask = rx_visible & cull_gaussians(model, rast)
    else:
        # C4/C5: mesh-vertex visible_mask if applicable, else FOV culling only
        if model.N == len(rast.vertices_np) and model.N == len(visible_mask):
            active_mask = visible_mask & cull_gaussians(model, rast)
        else:
            active_mask = cull_gaussians(model, rast)
    n_active = active_mask.sum().item()

    # C6: uniform hemisphere weights on the clean visible set
    # The cosine bias is encoded in the visible-weighted FPS placement,
    # so each visible Gaussian gets w = 1.0 (no per-Gaussian density correction).
    if use_hemisphere:
        hemisphere_weights_full = torch.zeros(model.N, device=DEVICE)
        hemisphere_weights_full[active_mask] = 1.0
        vertex_areas = hemisphere_weights_full
        if verbose:
            print(f"  Hemisphere weights: uniform = 1.0")

    if verbose:
        print(f"  Active after culling: {n_active}/{model.N}")

    # C2: just evaluate, no training
    if mode == 'c2':
        with torch.no_grad():
            rp_r, rp_i = render_gaussians_factorized(
                model, rast, vertex_areas=vertex_areas, active_mask=active_mask,
                shadow_mask=None)
            ra_mag = range_profile_to_ra_mag(rp_r, rp_i)
            ra_polar = ra_mag.cpu().numpy()
            del rp_r, rp_i
            ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
            ra_gt_cart = ra_polar_to_cartesian(
                adc_to_ra_image(torch.from_numpy(gt_adc_ri.cpu().numpy()).float()).numpy(),
                range_res)
            metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)
            if verbose:
                print(f"  C2 cart_corr: {metrics['cart_corr']:.4f}")
            return metrics['cart_corr'], 0

    # Option E: make antenna patterns learnable
    tx_E = torch.nn.Parameter(rast.tx_antenna.E.clone())
    tx_H = torch.nn.Parameter(rast.tx_antenna.H.clone())
    rx_E = torch.nn.Parameter(rast.rx_antenna.E.clone())
    rx_H = torch.nn.Parameter(rast.rx_antenna.H.clone())

    # Optimizer (C3/C4/C5)
    param_groups = [
        {"params": [model.raw_materials], "lr": 0.5, "name": "materials"},
        {"params": [model.positions], "lr": 1.6e-4, "name": "positions"},
        {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
        {"params": [model.log_scales], "lr": 5e-3, "name": "scales"},
        {"params": [model.logit_opacities], "lr": 5e-2, "name": "opacities"},
        {"params": [tx_E, tx_H, rx_E, rx_H], "lr": 0.05, "name": "patterns"},
    ]
    clip_vals = {
        "materials": 1.0, "positions": 1.0, "rotations": 0.5,
        "scales": 1.0, "opacities": 1.0, "patterns": 1.0,
    }
    base_lrs = {g["name"]: g["lr"] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Density control state (C5)
    if mode == 'c5':
        grad_accum = torch.zeros(model.N, device=DEVICE)
        grad_count = torch.zeros(model.N, device=DEVICE)
        densify_interval = 100
        prune_opacity_thresh = 0.01

    best_corr = -1.0
    best_iter = 0
    best_state = None
    best_ra_rend_cart = None
    best_ra_gt_cart = None
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad()

        # Recompute culling mask periodically for C4/C5 (positions may move)
        if mode in ('c4', 'c5', 'c6') and it % 50 == 0 and it > 0:
            active_mask = cull_gaussians(model, rast)

        # Option E: inject learnable patterns into rasterizer
        rast.tx_antenna.E = tx_E
        rast.tx_antenna.H = tx_H
        rast.rx_antenna.E = rx_E
        rast.rx_antenna.H = rx_H

        # Factorized renderer: exact BSDF + factorized phase, no checkpoint needed
        adc_real, adc_imag = render_gaussians_factorized(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, chunk_size=2000,
            shadow_mask=shadow)
        loss, loss_dict = compute_ra_loss_rp(adc_real, adc_imag, gt_adc_ri)
        loss.backward()

        del adc_real, adc_imag, loss

        # Gradient clipping
        for group in optimizer.param_groups:
            clip = clip_vals.get(group["name"], 1.0)
            for p in group["params"]:
                rms_clip_grad(p, clip)

        # LR warmup
        lr_scale = get_lr_scale(it, total_iters=num_iters, decay_start=100)
        for group in optimizer.param_groups:
            group["lr"] = base_lrs[group["name"]] * lr_scale

        optimizer.step()

        # C5: density control — prune low-opacity Gaussians
        if mode == 'c5' and (it + 1) % densify_interval == 0 and it > 0:
            with torch.no_grad():
                opacities = model.get_opacities()
                keep = opacities > prune_opacity_thresh
                n_prune = (~keep).sum().item()

                if n_prune > 0 and keep.sum() > 100:
                    keep_idx = keep.nonzero(as_tuple=True)[0]
                    N_new = keep_idx.shape[0]
                    new_model = GaussianSurfels(N_new, device=DEVICE)
                    new_model.positions.data.copy_(model.positions[keep_idx])
                    new_model.rotations.data.copy_(model.rotations[keep_idx])
                    new_model.log_scales.data.copy_(model.log_scales[keep_idx])
                    new_model.logit_opacities.data.copy_(model.logit_opacities[keep_idx])
                    new_model.raw_materials.data.copy_(model.raw_materials[keep_idx])
                    model = new_model

                    # Update vertex_areas and active_mask
                    vertex_areas = vertex_areas[keep_idx] if vertex_areas is not None else None
                    active_mask = cull_gaussians(model, rast)

                    # Rebuild optimizer
                    param_groups = [
                        {"params": [model.raw_materials], "lr": base_lrs["materials"], "name": "materials"},
                        {"params": [model.positions], "lr": base_lrs["positions"], "name": "positions"},
                        {"params": [model.rotations], "lr": base_lrs["rotations"], "name": "rotations"},
                        {"params": [model.log_scales], "lr": base_lrs["scales"], "name": "scales"},
                        {"params": [model.logit_opacities], "lr": base_lrs["opacities"], "name": "opacities"},
                    ]
                    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

                    if verbose:
                        print(f"    Pruned {n_prune}, N: {N_new}")

        # Evaluate periodically
        if it % 50 == 0 or it == num_iters - 1:
            with torch.no_grad():
                eval_r, eval_i = render_gaussians_factorized(
                    model, rast, vertex_areas=vertex_areas,
                    active_mask=active_mask, chunk_size=2000,
                    shadow_mask=None)
                # Range profile → RA via azimuth FFT only
                ra_mag = range_profile_to_ra_mag(eval_r, eval_i)
                ra_polar = ra_mag.cpu().numpy()
                del eval_r, eval_i
                ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
                ra_gt_cart = ra_polar_to_cartesian(
                    adc_to_ra_image(torch.from_numpy(gt_adc_ri.cpu().numpy()).float()).numpy(),
                    range_res)
                metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)
                cart_corr = metrics['cart_corr']

                if cart_corr > best_corr:
                    best_corr = cart_corr
                    best_iter = it
                    # Save best state (model + patterns + RA maps)
                    best_state = {
                        'model': {k: v.data.clone() for k, v in model.state_dict().items()},
                        'tx_E': tx_E.data.clone(),
                        'tx_H': tx_H.data.clone(),
                        'rx_E': rx_E.data.clone(),
                        'rx_H': rx_H.data.clone(),
                    }
                    best_ra_rend_cart = ra_cart.copy()
                    best_ra_gt_cart = ra_gt_cart.copy()

                if verbose:
                    elapsed = time.time() - t0
                    print(f"  iter {it:4d}: loss={loss_dict['ra_mse']:.6f}, "
                          f"cart_corr={cart_corr:.4f} (best={best_corr:.4f}@{best_iter}) "
                          f"[{elapsed:.1f}s, N={model.N}]")

    if verbose:
        print(f"\n  Best cart_corr: {best_corr:.4f} at iter {best_iter}")

    # Save best checkpoint (model + patterns + RA maps)
    output_dir = os.path.join(PROJECT_ROOT, 'mm25DGS_v3', 'output',
                              f'train_gaussian_{mode}', scene)
    os.makedirs(output_dir, exist_ok=True)

    if best_state is not None:
        from mmir.data.ra_utils import save_ra_cartesian_png

        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))
        np.save(os.path.join(output_dir, 'ra_rendered_cart.npy'), best_ra_rend_cart)
        np.save(os.path.join(output_dir, 'ra_gt_cart.npy'), best_ra_gt_cart)

        # Individual RA PNGs (same format as train.py)
        for scale in ('dB', 'linear'):
            save_ra_cartesian_png(
                best_ra_gt_cart,
                os.path.join(output_dir, f'gt_ra_{scale}.png'),
                range_res=range_res, scale=scale, title=f'GT ({scale})')
            save_ra_cartesian_png(
                best_ra_rend_cart,
                os.path.join(output_dir, f'rendered_ra_{scale}.png'),
                range_res=range_res, scale=scale, title=f'Rendered ({scale})')

        # Side-by-side comparison PNG
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def _mm(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(a)
        r_db = 10 * np.log10(np.maximum(best_ra_rend_cart, 1e-10))
        g_db = 10 * np.log10(np.maximum(best_ra_gt_cart, 1e-10))
        db_max = max(r_db.max(), g_db.max())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(g_db.T, origin='lower', aspect='auto', cmap='viridis',
                       vmin=db_max - 40, vmax=db_max)
        axes[0].set_title('Ground Truth (dB)')
        axes[1].imshow(r_db.T, origin='lower', aspect='auto', cmap='viridis',
                       vmin=db_max - 40, vmax=db_max)
        axes[1].set_title('Rendered (dB)')
        error = _mm(best_ra_rend_cart) - _mm(best_ra_gt_cart)
        axes[2].imshow(error.T, origin='lower', aspect='auto', cmap='RdBu_r',
                       vmin=-0.3, vmax=0.3)
        axes[2].set_title('Error')
        plt.suptitle(f'{scene} | cart_corr={best_corr:.4f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ra_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'best_cart_corr': best_corr,
            'best_iter': best_iter,
            'num_iters': num_iters,
            'n_gaussians': model.N,
            'mode': mode,
        }, f, indent=2)

    return best_corr, best_iter


def run_all_scenes(mode='c3', num_iters=500, target_n=None):
    """Run training on all 7 scenes."""
    results = {}
    for scene in SCENES:
        corr, it = train_gaussians(scene, mode=mode, num_iters=num_iters,
                                   target_n=target_n)
        mmIR_metrics = json.load(open(
            os.path.join(TRAIN_OUTPUT_DIR, scene, 'best_metrics.json')))
        results[scene] = {
            'gauss_corr': corr,
            'mmIR_corr': mmIR_metrics['cart_corr'],
            'gap': abs(corr - mmIR_metrics['cart_corr']),
        }

    print(f"\n{'='*70}")
    print(f"Stage {mode.upper()} Results")
    print(f"{'='*70}")
    print(f"{'Scene':<25} {'mmIR':>8} {'Gauss':>8} {'Gap':>6} {'Status':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    if mode == 'c2':
        target = 0.02
    elif mode == 'c3':
        target = 0.03
    else:
        target = 0.05

    all_pass = True
    for scene in SCENES:
        r = results[scene]
        status = "PASS" if r['gap'] < target else "FAIL"
        if r['gap'] >= target:
            all_pass = False
        print(f"{scene:<25} {r['mmIR_corr']:>8.4f} {r['gauss_corr']:>8.4f} "
              f"{r['gap']:>6.4f} {status:>8}")

    print(f"\nTarget: gap < {target:.2f}")
    print(f"{'OVERALL: PASS' if all_pass else 'OVERALL: FAIL'}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--mode', type=str, default='c3',
                        choices=['c2', 'c3', 'c4', 'c5', 'c6'])
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--n-gaussians', type=int, default=None,
                        help='Target Gaussian count for C4')
    args = parser.parse_args()

    if args.all:
        run_all_scenes(mode=args.mode, num_iters=args.iters,
                       target_n=args.n_gaussians)
    elif args.scene:
        train_gaussians(args.scene, mode=args.mode, num_iters=args.iters,
                        target_n=args.n_gaussians)
    else:
        print("Usage: --scene <name> or --all")
