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

from mmir.data.ra_utils import adc_to_ra_image, adc_to_ra_complex, ra_polar_to_cartesian, compute_cartesian_ra_metrics
from mmir.data.io_utils import compute_range_res_from_cfg
from mm25DGS_v2.render_mmIR import load_trained_config, load_best_params, TRAIN_OUTPUT_DIR
from mm25DGS_v2.rasterizer_torch import (
    RasterizerTorch, reparameterize_torch, inverse_reparameterize_torch,
)
from mm25DGS_v2.train_mesh import compute_ra_loss, get_lr_scale, rms_clip_grad

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]

DEVICE = "cuda:0"
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

        # Scales from k-NN PCA
        tree2 = KDTree(xyz)
        _, knn_idx = tree2.query(xyz, k=min(21, N))
        knn_idx = knn_idx[:, 1:]  # exclude self
        scales = np.zeros((N, 2), dtype=np.float32)
        for i in range(N):
            neighbors = xyz[knn_idx[i]] - xyz[i]
            n_i = normals[i]
            proj = neighbors - np.outer(neighbors @ n_i, n_i)
            cov = (proj.T @ proj) / len(knn_idx[i])
            eigvals = np.linalg.eigvalsh(cov)
            scales[i, 0] = np.clip(np.sqrt(max(eigvals[2], 1e-12)), 0.01, 0.5)
            scales[i, 1] = np.clip(np.sqrt(max(eigvals[1], 1e-12)), 0.01, 0.5)
        model.log_scales.copy_(torch.from_numpy(np.log(scales)).to(device))

        model.logit_opacities.fill_(0.0)

    # Uniform areas for LiDAR init
    vertex_areas_t = torch.ones(N, device=device)
    return model, config, pattern_data, vertex_areas_t


def _normals_to_quaternions(normals):
    """Convert normal vectors to quaternions [w,x,y,z]."""
    N = len(normals)
    quats = np.zeros((N, 4), dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)

    for i in range(N):
        n = normals[i]
        n = n / max(np.linalg.norm(n), 1e-12)

        # Rotation from (0,0,1) to n
        dot = np.dot(up, n)
        if dot > 0.9999:
            quats[i] = [1, 0, 0, 0]
        elif dot < -0.9999:
            quats[i] = [0, 1, 0, 0]  # 180 deg around x
        else:
            axis = np.cross(up, n)
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            angle = np.arccos(np.clip(dot, -1, 1))
            quats[i, 0] = np.cos(angle / 2)
            quats[i, 1:] = axis * np.sin(angle / 2)

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


def render_gaussians(model, rast, vertex_areas=None, detach_phase=True,
                     chunk_size=500, active_mask=None):
    """Render ADC from Gaussian surfels using the rasterizer.

    Shadow test is skipped for performance (cosine filtering handles occlusion).
    Pre-culling via active_mask reduces the number of paths.

    Args:
        vertex_areas: (N,) pre-computed vertex areas. If None, uses uniform.
        active_mask: (N,) bool mask of active Gaussians. If None, uses all.
    """
    positions = model.positions
    normals = model.get_normals()
    opacities = model.get_opacities()
    raw_materials = model.raw_materials

    if vertex_areas is not None:
        areas = vertex_areas * opacities
    else:
        areas = opacities

    # Apply culling mask
    if active_mask is not None:
        positions = positions[active_mask]
        normals = normals[active_mask]
        raw_materials = raw_materials[active_mask]
        areas = areas[active_mask]

    return rast.render_differentiable(
        raw_materials, normals, positions, areas,
        detach_phase=detach_phase, chunk_size=chunk_size,
        skip_shadow=True)


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
    # Initialize
    if mode in ('c2', 'c3'):
        model, config, pattern_data, vertex_areas = init_from_mesh(scene)
    elif mode in ('c4', 'c5'):
        model, config, pattern_data, vertex_areas = init_from_lidar(scene, target_n=target_n)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create rasterizer
    rast = RasterizerTorch(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )

    # Inject patterns from mmIR (for antenna patterns only)
    raw_params_mmIR, _, _ = load_best_params(scene)
    rast.inject_trained_params(raw_params_mmIR, None, pattern_data)

    # Use reservoir sampler to identify visible vertices (better culling)
    hits = rast._run_reservoir_sampler(seed=42)
    visible_verts = _get_visible_vertices(hits, rast.faces_np)
    visible_mask = torch.zeros(model.N, dtype=torch.bool, device=DEVICE)
    visible_mask[torch.from_numpy(visible_verts).long().to(DEVICE)] = True

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
    else:
        # For LiDAR init: keep Gaussians near visible mesh vertices
        from scipy.spatial import KDTree
        vis_verts_np = rast.vertices_np[visible_verts]
        tree = KDTree(vis_verts_np)
        gauss_pos_np = model.positions.detach().cpu().numpy()
        dists, _ = tree.query(gauss_pos_np)
        # Keep Gaussians within 0.5m of a visible vertex
        near_visible = torch.from_numpy(dists < 0.5).to(DEVICE)
        active_mask = near_visible & cull_gaussians(model, rast)
    n_active = active_mask.sum().item()

    # Cap active count to avoid OOM (12K verts * 192 MIMO fits in ~13 GB)
    MAX_ACTIVE = 12000
    if n_active > MAX_ACTIVE:
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        # Subsample uniformly
        perm = torch.randperm(n_active, device=DEVICE)[:MAX_ACTIVE]
        new_mask = torch.zeros_like(active_mask)
        new_mask[active_indices[perm]] = True
        active_mask = new_mask
        n_active = MAX_ACTIVE

    if verbose:
        print(f"  Active after culling: {n_active}/{model.N}")

    # C2: just evaluate, no training
    if mode == 'c2':
        with torch.no_grad():
            adc_real, adc_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas, active_mask=active_mask)
            adc_ri = torch.stack([adc_real, adc_imag], dim=-1).cpu().numpy()
            ra_polar = adc_to_ra_image(torch.from_numpy(adc_ri).float()).numpy()
            ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
            ra_gt_cart = ra_polar_to_cartesian(
                adc_to_ra_image(torch.from_numpy(gt_adc_ri.cpu().numpy()).float()).numpy(),
                range_res)
            metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)
            if verbose:
                print(f"  C2 cart_corr: {metrics['cart_corr']:.4f}")
            return metrics['cart_corr'], 0

    # Optimizer (C3/C4/C5)
    param_groups = [
        {"params": [model.raw_materials], "lr": 0.5, "name": "materials"},
        {"params": [model.positions], "lr": 1.6e-4, "name": "positions"},
        {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
        {"params": [model.log_scales], "lr": 5e-3, "name": "scales"},
        {"params": [model.logit_opacities], "lr": 5e-2, "name": "opacities"},
    ]
    clip_vals = {
        "materials": 1.0, "positions": 1.0, "rotations": 0.5,
        "scales": 1.0, "opacities": 1.0,
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
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad()

        # Recompute culling mask periodically for C4/C5 (positions may move)
        if mode in ('c4', 'c5') and it % 50 == 0 and it > 0:
            active_mask = cull_gaussians(model, rast)

        # Mini-batch: sample up to BATCH_SIZE active vertices per iteration
        # to keep the computation graph small enough for backward pass.
        BATCH_SIZE = 4000
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        if len(active_indices) > BATCH_SIZE:
            perm = torch.randperm(len(active_indices), device=DEVICE)[:BATCH_SIZE]
            batch_mask = torch.zeros_like(active_mask)
            batch_mask[active_indices[perm]] = True
        else:
            batch_mask = active_mask

        adc_real, adc_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=batch_mask, chunk_size=200)
        loss, loss_dict = compute_ra_loss(adc_real, adc_imag, gt_adc_ri)
        loss.backward()

        del adc_real, adc_imag, loss

        # Gradient clipping
        for group in optimizer.param_groups:
            clip = clip_vals.get(group["name"], 1.0)
            for p in group["params"]:
                rms_clip_grad(p, clip)

        # LR warmup
        lr_scale = get_lr_scale(it)
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
                    n_active = active_mask.sum().item()
                    MAX_ACTIVE = 12000
                    if n_active > MAX_ACTIVE:
                        ai = active_mask.nonzero(as_tuple=True)[0]
                        p = torch.randperm(n_active, device=DEVICE)[:MAX_ACTIVE]
                        active_mask = torch.zeros(model.N, dtype=torch.bool, device=DEVICE)
                        active_mask[ai[p]] = True

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
                eval_r, eval_i = render_gaussians(
                    model, rast, vertex_areas=vertex_areas,
                    active_mask=active_mask, chunk_size=200)
                adc_ri = torch.stack([eval_r, eval_i], dim=-1).cpu().numpy()
                del eval_r, eval_i
                ra_polar = adc_to_ra_image(torch.from_numpy(adc_ri).float()).numpy()
                ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
                ra_gt_cart = ra_polar_to_cartesian(
                    adc_to_ra_image(torch.from_numpy(gt_adc_ri.cpu().numpy()).float()).numpy(),
                    range_res)
                metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)
                cart_corr = metrics['cart_corr']

                if cart_corr > best_corr:
                    best_corr = cart_corr
                    best_iter = it

                if verbose:
                    elapsed = time.time() - t0
                    print(f"  iter {it:4d}: loss={loss_dict['ra_mse']:.6f}, "
                          f"cart_corr={cart_corr:.4f} (best={best_corr:.4f}@{best_iter}) "
                          f"[{elapsed:.1f}s, N={model.N}]")

    if verbose:
        print(f"\n  Best cart_corr: {best_corr:.4f} at iter {best_iter}")

    # Save
    output_dir = os.path.join(PROJECT_ROOT, 'mm25DGS_v2', 'output',
                              f'train_gaussian_{mode}', scene)
    os.makedirs(output_dir, exist_ok=True)
    model.save = lambda path: torch.save({
        k: v.data for k, v in model.state_dict().items()
    }, path)
    model.save(os.path.join(output_dir, 'best_model.pt'))
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
                        choices=['c2', 'c3', 'c4', 'c5'])
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
