"""Initialise GaussianModel from an aggregated LiDAR point cloud.

Pipeline:
  1. Load full point cloud.
  2. Filter to radar FOV: range within [near_field, num_adc * range_res],
     angle from boresight within ±π/2 (the hard bin-edge limit from the
     RAE grid, matching mmir/evaluation/utils/single_view_viz.py).
  3. Farthest-point-sample (FPS) to target count (~25K for single 4090).
  4. Local PCA on k-NN to get tangent frames and lateral scales.
"""

import numpy as np
import torch
from scipy.spatial import KDTree

from .config import RadarConfig, C
from .gaussian_model import GaussianModel
from .reparameterization import inverse_reparameterize, ITU_DEFAULTS

# Fixed constant from mmir/evaluation/eval_3d_occupancy.py line 28.
NEAR_FIELD_M = 1.5


def initialize_from_lidar(
    pcl_path: str,
    radar_cfg: RadarConfig,
    device: str = "cuda:0",
    target_n_gaussians: int = 25_000,
    k_neighbors: int = 20,
    scale_clamp_min: float = 0.01,
    scale_clamp_max: float = 0.50,
    initial_material: str = "concrete",
    mesh_path: str = None,
) -> GaussianModel:
    """Create a GaussianModel from an aggregated LiDAR point cloud.

    Args:
        pcl_path: .npy with shape (M, 7) [x, y, z, nx, ny, nz, intensity].
        radar_cfg: Parsed radar configuration (provides positions, boresight,
                   bandwidth -> range limits).
        device: PyTorch device.
        target_n_gaussians: Number of Gaussians after FOV filtering + FPS.
        k_neighbors: k for local PCA neighbour search.
        scale_clamp_min/max: Bounds on initial lateral scale (metres).
        mesh_path: Optional path to .ply mesh. If provided, Gaussian normals
                   are initialised from nearest mesh vertex normals (Poisson
                   reconstruction normals), which are significantly more
                   accurate than LiDAR PCA normals.
        initial_material: ITU material name for default material params.

    Returns:
        GaussianModel with N ≤ target_n_gaussians.
    """
    pcl = np.load(pcl_path)
    xyz = pcl[:, :3].astype(np.float64)
    normals_raw = pcl[:, 3:6].astype(np.float64)
    n_orig = xyz.shape[0]

    # Normalise normals
    norms = np.linalg.norm(normals_raw, axis=1, keepdims=True)
    normals = normals_raw / np.maximum(norms, 1e-12)

    # ------------------------------------------------------------------
    # Step 1: Filter to radar FOV
    # ------------------------------------------------------------------
    radar_center, boresight = _radar_geometry(radar_cfg)
    max_range = radar_cfg.num_adc_samples * radar_cfg.range_resolution

    to_pts = xyz - radar_center
    dists = np.linalg.norm(to_pts, axis=1)
    to_pts_norm = to_pts / np.maximum(dists[:, None], 1e-12)

    # Angle from boresight (FOV hard limit is ±π/2 from bin edges,
    # matching _centers_to_edges(..., low_clip=-π/2, high_clip=π/2))
    cos_bore = np.dot(to_pts_norm, boresight)
    angle_from_bore = np.arccos(np.clip(cos_bore, -1.0, 1.0))

    fov_mask = (
        (dists >= NEAR_FIELD_M)
        & (dists <= max_range)
        & (angle_from_bore <= np.pi / 2)
    )

    xyz = xyz[fov_mask]
    normals = normals[fov_mask]
    n_after_fov = xyz.shape[0]

    print(
        f"FOV filter: {n_orig:,} -> {n_after_fov:,} "
        f"(range [{NEAR_FIELD_M}m, {max_range:.1f}m], FOV ±90°)"
    )

    # ------------------------------------------------------------------
    # Step 2: Farthest-point sampling to target count (GPU-accelerated)
    # ------------------------------------------------------------------
    if n_after_fov > target_n_gaussians:
        fps_idx = _farthest_point_sampling_gpu(xyz, target_n_gaussians, device)
        fps_idx_np = fps_idx.cpu().numpy()
        xyz = xyz[fps_idx_np]
        normals = normals[fps_idx_np]
        print(f"FPS (GPU): {n_after_fov:,} -> {xyz.shape[0]:,}")
    else:
        print(f"No subsampling needed ({n_after_fov:,} ≤ {target_n_gaussians:,})")

    xyz = xyz.astype(np.float32)
    normals = normals.astype(np.float32)
    N = xyz.shape[0]

    # ------------------------------------------------------------------
    # Step 2b: Override normals from mesh if provided
    # ------------------------------------------------------------------
    if mesh_path is not None:
        import trimesh
        mesh = trimesh.load(mesh_path)
        mesh_verts = np.array(mesh.vertices, dtype=np.float32)
        mesh_norms = np.array(mesh.vertex_normals, dtype=np.float32)
        tree = KDTree(mesh_verts)
        _, idx = tree.query(xyz)
        normals = mesh_norms[idx]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
        print(f"Normals from mesh: {mesh_path}")

    # ------------------------------------------------------------------
    # Step 3: Build GaussianModel
    # ------------------------------------------------------------------
    model = GaussianModel(N, device=device)

    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

    # Local PCA for tangent frames and scales
    quats, scales = _compute_local_frames_and_scales(
        xyz, normals, k_neighbors, scale_clamp_min, scale_clamp_max
    )
    with torch.no_grad():
        model.rotations.copy_(torch.from_numpy(quats).to(device))
        model.log_scales.copy_(torch.from_numpy(np.log(scales)).to(device))

    # Materials: ITU default in raw space
    physics_default = ITU_DEFAULTS[initial_material]
    raw_default = inverse_reparameterize(physics_default)
    with torch.no_grad():
        model.raw_materials.copy_(
            torch.from_numpy(raw_default).unsqueeze(0).expand(N, -1).to(device)
        )

    return model


# -----------------------------------------------------------------------
#  Filtering helpers
# -----------------------------------------------------------------------

def _radar_geometry(radar_cfg: RadarConfig):
    """Compute radar centre position and mean boresight (world frame)."""
    tx_pos = radar_cfg.tx_positions_m
    rx_pos = radar_cfg.rx_positions_m
    radar_center = (tx_pos.mean(axis=0) + rx_pos.mean(axis=0)) / 2
    boresight = radar_cfg.tx_boresights.mean(axis=0)
    boresight = boresight / (np.linalg.norm(boresight) + 1e-12)
    return radar_center, boresight


def _farthest_point_sampling_gpu(
    points: np.ndarray,
    n_samples: int,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Farthest-point sampling on GPU.

    Each iteration is a single vectorised distance computation + argmax
    over all N points — fully parallel on the GPU.  The Python loop
    iterates n_samples times but each step is O(N) GPU work.

    Args:
        points: (N, 3) numpy array.
        n_samples: number of points to select.
        device: CUDA device.

    Returns:
        (n_samples,) long tensor of selected indices.
    """
    pts = torch.from_numpy(points.astype(np.float32)).to(device)  # (N, 3)
    N = pts.shape[0]

    if n_samples >= N:
        return torch.arange(N, device=device)

    selected = torch.empty(n_samples, dtype=torch.long, device=device)
    min_dist = torch.full((N,), float("inf"), device=device)

    # Start from centroid-nearest point
    centroid = pts.mean(dim=0, keepdim=True)
    first_idx = torch.argmin(torch.norm(pts - centroid, dim=1))
    selected[0] = first_idx

    # Initial distances
    min_dist = torch.norm(pts - pts[first_idx], dim=1)
    min_dist[first_idx] = 0.0

    for j in range(1, n_samples):
        idx = torch.argmax(min_dist)
        selected[j] = idx

        d = torch.norm(pts - pts[idx], dim=1)
        min_dist = torch.minimum(min_dist, d)
        min_dist[idx] = 0.0

    return selected
    return selected


# -----------------------------------------------------------------------
#  Local PCA helpers
# -----------------------------------------------------------------------

def _compute_local_frames_and_scales(
    xyz: np.ndarray,
    normals: np.ndarray,
    k: int,
    s_min: float,
    s_max: float,
) -> tuple:
    """Compute per-point rotation quaternion [w,x,y,z] and lateral scales."""
    tree = KDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    idx = idx[:, 1:]

    N = xyz.shape[0]
    quats = np.zeros((N, 4), dtype=np.float32)
    scales = np.zeros((N, 2), dtype=np.float32)

    for i in range(N):
        n_i = normals[i]

        neighbours = xyz[idx[i]] - xyz[i]
        proj = neighbours - np.outer(neighbours @ n_i, n_i)

        cov = (proj.T @ proj) / k
        eigvals, eigvecs = np.linalg.eigh(cov)

        t1 = eigvecs[:, 2]
        t2 = eigvecs[:, 1]

        if np.dot(np.cross(t1, t2), n_i) < 0:
            t2 = -t2

        R = np.column_stack([t1, t2, n_i])
        quats[i] = _rotation_matrix_to_quaternion(R)

        s1 = np.clip(np.sqrt(max(eigvals[2], 1e-12)), s_min, s_max)
        s2 = np.clip(np.sqrt(max(eigvals[1], 1e-12)), s_min, s_max)
        scales[i] = [s1, s2]

    return quats, scales


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [w, x, y, z] quaternion (Shepperd's method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)
