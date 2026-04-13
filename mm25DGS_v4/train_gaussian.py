"""mm25DGS_v4 — c6 hemisphere integral training (point-based primitive).

Single training mode: visible-weighted FPS init + uniform hemisphere weights.

Pipeline:
  1. Load pcl.npy (1.5–2.5M points). Columns 0–2 are xyz, 3–5 are unit normals.
  2. FOV restrict (cos_bore > cos_bore_min, within radar range).
  3. RX-side ray-traced visibility against the Mitsuba scene (the ONLY mesh use).
  4. Cosine-hemisphere importance resample (prob ∝ cos_bore).
  5. FPS to target_n.
  6. Train per-point quaternions (→ surface normals) + 6 material params
     + global TX/RX antenna E/H planes. Each parameter group is gated by
     a LEARN_* flag (matches mmIR's convention). Positions are FROZEN by
     default — they come from the LiDAR pcl and stay fixed.

Primitive: a POINT, not a Gaussian. There is no scale and no opacity in
the model. The renderer reads only position, normal, and material per
point; the hemisphere weight on the active set is uniform = 1.0.

Ray tracing footprint (one-shot at init, freed before training loop):
  - One per-point RX-visibility ray test on the full FOV-filtered point
    cloud (~1.5M rays, batched). Filters out occluded points before FPS,
    so every point in the model is RX-visible by construction.
  - No TX shadow mask. The A/B test in v4_ray_tracing_minimization_plan.md
    showed shadow OFF is at least as good as shadow ON (mean +0.0068,
    6/7 scenes improved or flat).
  - No reservoir sampler. No per-Gaussian post-FPS sanity-check ray test.
  - The renderer itself does ZERO ray tracing.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v4.train_gaussian --scene seq_0_frame_135 --iters 500
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v4.train_gaussian --all --iters 500
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

from mm25DGS_v4.rasterizer import (
    Rasterizer, reparameterize_torch, inverse_reparameterize_torch,
)
from mm25DGS_v4.load_pretrained import (
    load_trained_config, load_pattern_data, TRAIN_OUTPUT_DIR, SCENES,
)
from mm25DGS_v4.material_diagnostics import MaterialDiagnostics

DEVICE = "cuda:0"

# =========================================================================
# Learnable parameter toggles (matches mmIR's LEARN_* pattern in train.py)
# =========================================================================
# Each flag controls (a) whether the underlying parameter has requires_grad
# and (b) whether it appears in the optimizer's param_groups. Flip a flag,
# and the parameter is frozen / unfrozen uniformly.
LEARN_POSITIONS = False    # frozen — pcl positions are treated as fixed reference
LEARN_NORMALS   = True     # quaternion → surface normal; refines noisy pcl normals
LEARN_MATERIALS = True     # 6 ITU/Cook-Torrance params per point
LEARN_PATTERNS  = True     # TX/RX antenna E/H planes (361 samples each)

# Default starting per-point material (ITU concrete).
# sigma_h lowered from 1e-4 (100 μm) to 5e-5 (50 μm) to stay inside the SPM
# validity regime (h_max_1 = 0.1/k ≈ 62 μm at 77 GHz). At 100 μm the
# enforce_spm_validity clamp routes grad through a constant, making sigma_h
# dead in backward — confirmed via gradient/drift/Fisher diagnostics.
ITU_CONCRETE = np.array([5.31, 0.0326, 5e-5, 5e-3, 0.5, 0.15], dtype=np.float32)


# =========================================================================
# Range profile -> RA image (azimuth FFT only, no range FFT)
# =========================================================================

# Module-level cache for the azimuth Hann window. Keyed by (length, device,
# complex dtype). The window depends only on the static array geometry, so
# computing it once per process is plenty.
_HANN_CACHE = {}


def _get_hann_window(num_vx, device, complex_dtype):
    key = (num_vx, str(device), complex_dtype)
    w = _HANN_CACHE.get(key)
    if w is None:
        # complex_dtype.real returns float32 if complex64, float64 if complex128
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
        w = torch.hann_window(num_vx, device=device, dtype=real_dtype).to(complex_dtype)[:, None]
        _HANN_CACHE[key] = w
    return w


def range_profile_to_ra(rp_real, rp_imag):
    """Convert complex range profiles to RA image via azimuth FFT only."""
    rp_ri = torch.stack([rp_real, rp_imag], dim=-1)              # (12, 16, K, 2)
    rp_ri = rp_ri.permute(1, 0, 2, 3)                            # (RX=16, TX=12, K, 2)
    rp_c = torch.complex(rp_ri[..., 0].contiguous(),
                          rp_ri[..., 1].contiguous())              # (16, 12, 256)
    rp_c = rp_c.unsqueeze(0)                                      # (1, 16, 12, 256)

    vx = txrx_to_vx_chirps_torch(rp_c)                            # (1, 7, 86, 256)
    ra = vx[0, 0, :, :]                                           # (86, 256)

    num_vx = ra.size(0)
    ra = ra * _get_hann_window(num_vx, ra.device, ra.dtype)
    ra = torch.fft.ifftshift(ra, dim=0)
    ra = torch.fft.fft(ra, n=128, dim=0)
    ra = ra[1:, :]
    ra = torch.fft.fftshift(ra, dim=0)
    return ra  # (127, 256) complex


def range_profile_to_ra_mag(rp_real, rp_imag):
    return range_profile_to_ra(rp_real, rp_imag).abs().float()


# =========================================================================
# GPU eval helpers (replace scipy griddata + numpy corrcoef)
# =========================================================================

def build_polar_to_cart_grid(n_az, n_range, range_res, grid_res=400, device=DEVICE):
    """Pre-compute the (grid_res-1, grid_res-1, 2) sample grid for grid_sample.

    Replicates the indexing of mmIR's ra_polar_to_cartesian (which uses
    scipy.griddata with linear interpolation) using F.grid_sample bilinear
    interpolation on a regular polar→cart grid.

    Polar grid layout (matches range_profile_to_ra output):
        shape   (n_az, n_range)        e.g. (127, 256)
        az bin i → sin(angle) = (i - n_az/2) * 2/(n_az+1)  ← linear in sin(angle)
        range bin j → r = j * range_res

    Cartesian grid:
        xi ∈ [-range_width, range_width]
        yi ∈ [0, range_depth]
        range_depth = n_range * range_res
        range_width = range_depth / 2

    For each (xi, yi):
        r       = sqrt(xi² + yi²)            → range bin
        sin_θ   = xi / r                     → az bin (since the polar grid
                                              is uniform in sin(angle))

    Returned grid is normalized to [-1, 1] for F.grid_sample with
    align_corners=True. The grid samples the polar image (interpreted as
    a 4D tensor of shape (1, 1, n_az, n_range)).

    The mmIR reference also (a) drops the last row/col, (b) flips the
    azimuth axis (`zi[:, ::-1]`). We replicate both: drop and flip applied
    to the grid construction so the resulting cart image lines up with
    the CPU reference.
    """
    range_depth = n_range * range_res
    range_width = range_depth / 2.0

    # mmIR uses linspace(grid_res) then drops the last col/row → (grid_res-1, grid_res-1)
    xi = torch.linspace(-range_width, range_width, grid_res, device=device)
    yi = torch.linspace(0.0, range_depth, grid_res, device=device)
    xi = xi[:-1]
    yi = yi[:-1]
    xx, yy = torch.meshgrid(xi, yi, indexing='xy')   # (grid_res-1, grid_res-1)

    # mmIR reverses the cart image along the azimuth (last) axis at the end:
    #   `return zi[:, ::-1]`
    # We bake the flip into the grid by flipping xx along its last axis.
    xx = torch.flip(xx, dims=[-1])

    r = torch.sqrt(xx * xx + yy * yy).clamp(min=1e-9)        # (h, w)
    sin_theta = (xx / r).clamp(-1.0, 1.0)                    # (h, w)

    # Polar grid az index from sin(theta).
    # mmIR's polar grid: bin k → sin(angle) = (-num_angle_bins/2 + 1 + k) * 2/num_angle_bins
    # where num_angle_bins = n_az + 1. For n_az=127 → num_angle_bins=128:
    #   bin 0   → sin = -126/128 ≈ -0.984
    #   bin 63  → sin = 0
    #   bin 126 → sin = +126/128 ≈ +0.984
    # Inverting: k = (n_az - 1)/2 + sin_theta * (n_az + 1)/2
    az_idx = (n_az - 1) / 2.0 + sin_theta * (n_az + 1) / 2.0

    # Polar grid range index: r_idx = r / range_res
    range_idx = r / range_res

    # Normalize to [-1, 1] for F.grid_sample with align_corners=True.
    # grid coords: (x in [-1,1] = column = range_idx, y in [-1,1] = row = az_idx)
    # F.grid_sample uses (x, y) order in the last dim where x is the WIDTH dim.
    # We treat the polar tensor as (1, 1, H=n_az, W=n_range), so:
    #   sample x ↔ range, sample y ↔ azimuth
    range_norm = 2.0 * range_idx / (n_range - 1) - 1.0
    az_norm    = 2.0 * az_idx    / (n_az - 1)    - 1.0

    grid = torch.stack([range_norm, az_norm], dim=-1)        # (h, w, 2)
    return grid.unsqueeze(0)                                 # (1, h, w, 2)


def polar_to_cart_torch(ra_polar, sample_grid):
    """GPU polar → cartesian via F.grid_sample.

    Args:
        ra_polar: (n_az, n_range) float tensor on GPU
        sample_grid: (1, h, w, 2) precomputed by build_polar_to_cart_grid

    Returns: (h, w) float tensor on GPU.
    """
    img = ra_polar.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)
    out = F.grid_sample(
        img, sample_grid, mode='bilinear',
        padding_mode='zeros', align_corners=True)
    return out[0, 0]                                          # (h, w)


def cart_corr_torch(rend_cart, gt_cart_normalized):
    """GPU Pearson correlation between rendered cart RA and pre-normalized GT.

    Both inputs are 2D float tensors. The GT is expected to be already
    min-max normalized (cached once at init). The rendered tensor is
    normalized inside this function.
    """
    rend_min = rend_cart.min()
    rend_max = rend_cart.max()
    rend_range = (rend_max - rend_min).clamp(min=1e-30)
    rend_norm = (rend_cart - rend_min) / rend_range

    rend_flat = rend_norm.flatten()
    gt_flat = gt_cart_normalized.flatten()
    rend_mean = rend_flat.mean()
    gt_mean = gt_flat.mean()
    rend_centered = rend_flat - rend_mean
    gt_centered = gt_flat - gt_mean

    num = (rend_centered * gt_centered).sum()
    den = torch.sqrt(
        (rend_centered ** 2).sum() * (gt_centered ** 2).sum()
    ).clamp(min=1e-30)
    return num / den


def precompute_gt_loss_norm(gt_adc_ri):
    """One-shot computation of the GT min-max-normalized RA magnitude for the loss.

    The GT does not change during training; the loss recomputed it every
    iter for no reason (~5 ms wasted per iter). Call this once before the
    training loop and pass the result into compute_ra_loss_rp.
    """
    with torch.no_grad():
        ra_gt = adc_to_ra_complex(gt_adc_ri)
        ra_gt_mag = torch.abs(ra_gt)
        mn, mx = ra_gt_mag.min(), ra_gt_mag.max()
        return ((ra_gt_mag - mn) / (mx - mn).clamp(min=1e-30)).detach()


def compute_ra_loss_rp(rp_real, rp_imag, gt_norm_cached, loss_type='mse'):
    """RA loss between rendered range profiles and the cached GT.

    loss_type:
      'mse'     — min-max-normalized MSE on RA magnitude (default, current)
      'pearson' — 1 - Pearson(rendered_flat, gt_norm_cached_flat); uses
                  standard deviation normalization + dot product, i.e. the
                  same structural metric we evaluate with (cart_corr),
                  applied in polar space.
    """
    ra_rendered = range_profile_to_ra(rp_real, rp_imag)
    ra_rend_mag = torch.abs(ra_rendered)

    if loss_type == 'mse':
        mn = ra_rend_mag.min()
        mx = ra_rend_mag.max()
        rend_norm = (ra_rend_mag - mn) / (mx - mn).clamp(min=1e-30)
        loss = torch.mean((rend_norm - gt_norm_cached) ** 2)
        return loss, {"ra_mse": loss.item()}
    elif loss_type == 'pearson':
        r_flat = ra_rend_mag.reshape(-1)
        g_flat = gt_norm_cached.reshape(-1)
        r_mean = r_flat.mean()
        g_mean = g_flat.mean()
        r_c = r_flat - r_mean
        g_c = g_flat - g_mean
        num = (r_c * g_c).sum()
        den = torch.sqrt((r_c * r_c).sum() * (g_c * g_c).sum()).clamp(min=1e-30)
        corr = num / den
        loss = 1.0 - corr
        return loss, {"pearson_loss": loss.item()}
    else:
        raise ValueError(f"Unknown loss_type {loss_type}")


# =========================================================================
# Point primitive model
# =========================================================================

class PointPrimitives(torch.nn.Module):
    """Point-based primitives for radar rendering.

    Per-point state: position(3) + quaternion(4) + raw_materials(6) = 13 floats.
    Each parameter's `requires_grad` is set from the LEARN_* flags above so that
    flipping a flag uniformly freezes/unfreezes the parameter for both autograd
    and the optimizer (matches mmIR's LEARN_* convention).

    Plus 1444 global antenna pattern values (TX/RX × E/H × 361 samples), gated
    by LEARN_PATTERNS in the training loop.
    """

    def __init__(self, N, device=DEVICE):
        super().__init__()
        self.device = device
        self.positions = torch.nn.Parameter(
            torch.zeros(N, 3, device=device), requires_grad=LEARN_POSITIONS)
        self.rotations = torch.nn.Parameter(
            torch.zeros(N, 4, device=device), requires_grad=LEARN_NORMALS)
        self.raw_materials = torch.nn.Parameter(
            torch.zeros(N, 6, device=device), requires_grad=LEARN_MATERIALS)

        with torch.no_grad():
            self.rotations[:, 0] = 1.0  # identity quaternion [w,x,y,z]

    @property
    def N(self):
        return self.positions.shape[0]

    def get_normals(self):
        """(N, 3) surface normals = third column of rotation matrix."""
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        nx = 2 * (x * z + w * y)
        ny = 2 * (y * z - w * x)
        nz = 1 - 2 * (x * x + y * y)
        return torch.stack([nx, ny, nz], dim=-1)


# =========================================================================
# Geometry helpers (no mesh required — pcl carries normals directly)
# =========================================================================

def _normals_to_quaternions(normals):
    """Convert (N, 3) unit normals to (N, 4) quaternions [w,x,y,z] that
    rotate the local +z axis onto the surface normal."""
    N = len(normals)
    nrm = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    n = normals / nrm

    # axis = cross(up=(0,0,1), n) = (-n_y, n_x, 0)
    axis = np.zeros((N, 3), dtype=np.float32)
    axis[:, 0] = -n[:, 1]
    axis[:, 1] = n[:, 0]

    axis_len = np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    axis = axis / axis_len

    dot = n[:, 2].copy()
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    half_angle = angle / 2.0

    quats = np.zeros((N, 4), dtype=np.float32)
    quats[:, 0] = np.cos(half_angle)
    quats[:, 1] = axis[:, 0] * np.sin(half_angle)
    quats[:, 2] = axis[:, 1] * np.sin(half_angle)
    quats[:, 3] = axis[:, 2] * np.sin(half_angle)

    parallel = dot > 0.9999
    quats[parallel] = [1, 0, 0, 0]
    anti = dot < -0.9999
    quats[anti] = [0, 1, 0, 0]
    return quats


def _farthest_point_sampling(pts, n_samples):
    """GPU farthest-point sampling.

    Sequentially picks n_samples points such that each new pick is the
    one farthest (Euclidean) from all already-picked points. Inherently
    sequential — no parallelism across iterations possible.

    Three optimizations vs the textbook implementation:
      1. Use squared distances throughout (avoids per-iter sqrt). Argmax of
         d² is the same as argmax of d.
      2. `torch.index_select` instead of fancy indexing `pts[idx]` for the
         single-row anchor lookup. Significantly faster on CUDA — fancy
         indexing dispatches a slower kernel for variable-length gathers.
      3. Skip the explicit `min_dist[idx] = 0` after each pick. The next
         (pts - pts[idx])² computation already produces a 0 at the picked
         row, and the torch.minimum() carries it through.

    Empirical (130K → 50K, RTX 4090): textbook 2.94 s → this version 1.77 s
    (~40% faster). Sample diversity unchanged (same average pairwise
    distance, same bounding-box coverage as the textbook variant).
    """
    N = pts.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=pts.device)

    selected = torch.empty(n_samples, dtype=torch.long, device=pts.device)
    selected[0] = 0
    diff = pts - pts[0:1]
    min_dist_sq = (diff * diff).sum(-1)

    for j in range(1, n_samples):
        idx = torch.argmax(min_dist_sq)
        selected[j] = idx
        anchor = torch.index_select(pts, 0, idx.unsqueeze(0))
        diff = pts - anchor
        d_sq = (diff * diff).sum(-1)
        min_dist_sq = torch.minimum(min_dist_sq, d_sq)
    return selected


# =========================================================================
# Init-time ray tracing (the only ray tracing in v4)
# =========================================================================

def _ray_test_visibility_batched(points_np, rx_center_np, mi_scene, chunk=200000):
    """Batched RX-side visibility ray test for the full point cloud.

    For each point, casts a ray from the point toward the RX array center.
    If the ray hits any mesh surface before reaching the RX, the point is
    occluded. This is the ONE ray-tracing call we keep in v4 — it filters
    the raw point cloud down to only RX-visible points before FPS.
    """
    import drjit as dr  # noqa: F401
    N = len(points_np)
    visible = np.zeros(N, dtype=bool)
    eps = 1e-3
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        pts = points_np[i:j]
        delta = rx_center_np - pts
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
        direction = delta / dist[:, None]
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


# =========================================================================
# FOV culling (no mesh, pure radar geometry)
# =========================================================================

def cull_gaussians(model, rast, cos_threshold=0.05):
    """Pre-filter Gaussians to only those visible to the radar (FOV only)."""
    with torch.no_grad():
        positions = model.positions
        normals = model.get_normals()

        radar_center = (rast.tx_positions.mean(0) + rast.rx_positions.mean(0)) / 2
        boresight = rast.tx_boresights.mean(0)
        boresight = boresight / boresight.norm().clamp(min=1e-6)

        to_radar = radar_center - positions
        dist = to_radar.norm(dim=-1)
        to_radar_dir = to_radar / dist.clamp(min=1e-6).unsqueeze(-1)

        cos_normal = torch.abs((normals * to_radar_dir).sum(-1))

        from_radar = -to_radar_dir
        cos_bore = (from_radar * boresight).sum(-1)

        max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

        active = (
            (dist > 1.5) &
            (dist < max_range) &
            (cos_bore > 0.0) &
            (cos_normal > cos_threshold)
        )
        return active


# =========================================================================
# Init: visible-weighted FPS from raw point cloud
# =========================================================================

def init_visible_weighted(scene, rast, target_n=50000,
                          cos_bore_min=0.1,
                          n_intermediate=200000,
                          device=DEVICE):
    """Initialize Gaussians via visible-weighted FPS over the raw point cloud.

    Pipeline:
      1. Load pcl.npy. Columns 0–2 are xyz, 3–5 are unit normals.
      2. FOV restrict by cos_bore_min and radar range.
      3. RX-side visibility ray test (the ONLY mesh use in v4 init).
      4. Cosine-hemisphere importance resample (prob ∝ cos_bore).
      5. FPS to target_n.
      6. Build the PointPrimitives model with normals from the pcl.

    Returns the PointPrimitives model. The caller is responsible for
    config/pattern_data loading.
    """
    pcl_path = rast._mesh_file.replace('scene/mesh.ply', 'scene/pcl.npy')
    if not os.path.exists(pcl_path):
        raise FileNotFoundError(
            f"pcl.npy not found at {pcl_path}. v4 requires the point cloud "
            f"(with normals in columns 3–5).")

    pcl = np.load(pcl_path)
    if pcl.shape[1] < 6:
        raise ValueError(
            f"{pcl_path} has shape {pcl.shape}; expected at least 6 columns "
            f"(xyz + unit normals).")

    xyz_full = pcl[:, :3].astype(np.float32)
    nrm_full = pcl[:, 3:6].astype(np.float32)
    nrm_full = nrm_full / np.maximum(
        np.linalg.norm(nrm_full, axis=1, keepdims=True), 1e-12)

    print(f"  [v4 init] Full pcl: {len(xyz_full)} points")

    # --- Step 1: FOV restrict ---
    rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()
    boresight = rast.tx_boresights.mean(dim=0).cpu().numpy()
    boresight = boresight / max(np.linalg.norm(boresight), 1e-8)

    delta = xyz_full - rx_center
    dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
    dir_to_point = delta / dist[:, None]
    cos_bore_full = (dir_to_point * boresight).sum(axis=-1)

    max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))

    fov_mask = (cos_bore_full > cos_bore_min) & (dist > 1.5) & (dist < max_range)
    xyz_fov = xyz_full[fov_mask]
    nrm_fov = nrm_full[fov_mask]
    cos_bore_fov = cos_bore_full[fov_mask]
    print(f"  [v4 init] After FOV: {len(xyz_fov)} points")

    # --- Step 2: RX visibility ray tracing (the only mesh use) ---
    rast.load_mi_scene()
    visible = _ray_test_visibility_batched(xyz_fov, rx_center, rast._mi_scene)
    xyz_vis = xyz_fov[visible]
    nrm_vis = nrm_fov[visible]
    cos_bore_vis = cos_bore_fov[visible]
    print(f"  [v4 init] After RX visibility: {len(xyz_vis)} points")

    if len(xyz_vis) < 100:
        raise RuntimeError(
            f"v4 init: only {len(xyz_vis)} points survived visibility — "
            f"check the Mitsuba scene and FOV settings.")

    # --- Step 3: Cosine-hemisphere importance resample ---
    weights = np.maximum(cos_bore_vis, 0.01)
    probs = weights / weights.sum()
    n_resample = min(n_intermediate, 3 * target_n)
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(xyz_vis), size=n_resample, replace=True, p=probs)
    unique_idx = np.unique(sampled_idx)
    xyz_weighted = xyz_vis[unique_idx]
    nrm_weighted = nrm_vis[unique_idx]
    print(f"  [v4 init] After cosine importance resample: {len(xyz_weighted)} points")

    # --- Step 4: FPS to target_n ---
    if len(xyz_weighted) > target_n:
        pts_t = torch.from_numpy(xyz_weighted).to(device)
        sel = _farthest_point_sampling(pts_t, target_n).cpu().numpy()
        xyz = xyz_weighted[sel]
        normals = nrm_weighted[sel]
    else:
        xyz = xyz_weighted
        normals = nrm_weighted
    print(f"  [v4 init] After FPS: {len(xyz)} points")

    N = len(xyz)

    # --- Step 5: Build the model ---
    model = PointPrimitives(N, device=device)
    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        model.raw_materials.copy_(
            torch.from_numpy(np.tile(raw_default, (N, 1))).to(device))

        quats = _normals_to_quaternions(normals)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

    return model


# =========================================================================
# Renderer wrapper (uniform hemisphere weights)
# =========================================================================

def render_gaussians(model, rast, vertex_areas, active_mask=None,
                     shadow_mask=None, detach_phase=True, bsdf_mode='full',
                     disabled_components=None):
    """Range-profile splatting renderer wrapper.

    `vertex_areas` carries the precomputed per-point hemisphere weight
    (uniform = 1.0 for active points, 0.0 for inactive).

    Renders the full active set in one shot — no chunking. The dominant
    intermediate is the (M, n_tx, n_rx) BSDF tensor which at our scales
    (M ≈ 45K active, n_tx=12, n_rx=16, fp32) is ~35 MB. Total renderer
    working set is well under 1 GB on a 4090.
    """
    from mm25DGS_v4.rasterizer_factorized import render_factorized

    positions = model.positions
    normals = model.get_normals()
    raw_materials = model.raw_materials

    # Point-based primitive: no opacity multiply. vertex_areas is already
    # the binary 1.0/0.0 active mask for the hemisphere weight.
    areas = vertex_areas

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
        shadow_mask=sm, bsdf_mode=bsdf_mode,
        disabled_components=disabled_components)


# =========================================================================
# Training loop (LR schedule + grad clip helpers inlined)
# =========================================================================

def get_lr_scale(iteration, warmup_iters=5, warmup_factor=0.3,
                 total_iters=None, decay_start=100, min_lr_factor=0.01):
    """Linear warmup -> cosine decay to min_lr_factor."""
    if iteration < warmup_iters:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup_iters, 1))
    if total_iters is None or iteration < decay_start:
        return 1.0
    progress = (iteration - decay_start) / max(total_iters - decay_start, 1)
    progress = min(progress, 1.0)
    return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + math.cos(math.pi * progress))


def rms_clip_grad(param, max_rms):
    """Per-parameter RMS gradient clipping."""
    if param.grad is None:
        return
    g = param.grad.data
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    rms = torch.sqrt(torch.mean(g ** 2))
    if rms > max_rms:
        g.mul_(max_rms / rms)
    param.grad.data = g


def train_gaussians(scene, num_iters=500, target_n=50000, verbose=True,
                    diagnostics_dir=None, run_name=None,
                    freeze_mat_cols=None, mat_mode='per_point',
                    disabled_components=None,
                    learn_normals=None, learn_materials=None, learn_patterns=None,
                    capture_grad_stats=False,
                    symmetry_break_std=0.0,
                    material_clusters=0,
                    loss_type='mse'):
    """Train v4 c6 hemisphere Gaussians for one scene.

    If `diagnostics_dir` is provided, captures material parameter trajectories,
    drift, and Fisher diagonal at end and dumps a `<scene>.npz` under that dir.

    If `freeze_mat_cols` is provided (an iterable of column indices in 0..5),
    those columns of `raw_materials` are frozen at their init values via a
    gradient mask. Adam state for frozen columns stays zero. This is the
    Phase 2 LOO/TOO mechanism.

    `mat_mode` controls Phase 1 baselines:
      'per_point' (default): full (M, 6) per-point materials, full BSDF
      'global':              all rows of (M, 6) constrained to share one
                             material vector via grad averaging + post-step
                             broadcast
      'scalar':              BSDF replaced with sigmoid(rho) * cos_i,
                             rho = raw_materials[:, 0]; columns 1..5 frozen
      'fixed':               equivalent to freeze_mat_cols=[0,1,2,3,4,5]
                             (B0 baseline)
    """
    if mat_mode not in ('per_point', 'global', 'scalar', 'fixed'):
        raise ValueError(f"mat_mode must be one of per_point/global/scalar/fixed, got {mat_mode}")

    # Per-call LEARN_* overrides (None → use module-level constant)
    _learn_normals = LEARN_NORMALS if learn_normals is None else learn_normals
    _learn_materials = LEARN_MATERIALS if learn_materials is None else learn_materials
    _learn_patterns = LEARN_PATTERNS if learn_patterns is None else learn_patterns
    if mat_mode == 'fixed':
        freeze_mat_cols = list(range(6))
    elif mat_mode == 'scalar':
        # Only column 0 is the learnable reflectivity; freeze 1..5
        freeze_mat_cols = [1, 2, 3, 4, 5]
    bsdf_mode = 'scalar' if mat_mode == 'scalar' else 'full'
    config = load_trained_config(scene)
    pattern_data = load_pattern_data(scene)

    rast = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )
    rast.inject_trained_params(pattern_data=pattern_data)

    model = init_visible_weighted(scene, rast, target_n=target_n)

    # Per-call LEARN_* override: set requires_grad on the model parameters.
    # The PointPrimitives constructor uses module-level constants by default;
    # we may need to flip them off for the LEARN-flag matrix runs.
    model.rotations.requires_grad_(_learn_normals)
    model.raw_materials.requires_grad_(_learn_materials)

    # T1: symmetry-breaking noise on raw_materials at init. Gives per-point
    # material diversity from iter 0 so the optimizer has gauge-variant
    # gradient signal before normals absorb the improvement. std is in the
    # same (raw) space as raw_materials, i.e. the learnable space post-reparam.
    if symmetry_break_std > 0.0:
        with torch.no_grad():
            noise = torch.randn_like(model.raw_materials) * symmetry_break_std
            model.raw_materials.add_(noise)
        if verbose:
            print(f"  T1 symmetry break: +N(0, {symmetry_break_std}) on raw_materials")

    # T3: k-means clustering on (positions, normals) at init.
    # Material params of all points in the same cluster are tied together.
    # Implemented as a backward hook that averages gradient within each
    # cluster + a post-step copy that broadcasts cluster centroids back.
    cluster_assignment = None
    cluster_idx_long = None
    if material_clusters > 0:
        with torch.no_grad():
            feat_pos = model.positions.detach().cpu().numpy()         # (N, 3)
            feat_nrm = model.get_normals().detach().cpu().numpy()     # (N, 3)
            # Scale position by 0.1 so normals and positions contribute
            # roughly equally to the k-means distance
            feats = np.concatenate([feat_pos * 0.1, feat_nrm], axis=1)  # (N, 6)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=material_clusters, n_init=5,
                        random_state=42, max_iter=100)
            labels = km.fit_predict(feats)
            cluster_assignment = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)
            cluster_idx_long = cluster_assignment  # (N,)
        if verbose:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  T3 clusters: {material_clusters} groups, sizes min={counts.min()} "
                  f"max={counts.max()} mean={counts.mean():.0f}")

        # Backward hook: replace each row's grad with the mean grad of its cluster
        counts_t = torch.zeros(material_clusters, device=DEVICE)
        counts_t.scatter_add_(0, cluster_idx_long,
                              torch.ones_like(cluster_idx_long, dtype=torch.float32))
        counts_t = counts_t.clamp(min=1.0)
        def _cluster_grad_hook(grad):
            # grad: (M, 6); we need cluster-averaged grad[i] = mean over cluster of grad[j]
            sum_per_cluster = torch.zeros(material_clusters, 6, device=DEVICE, dtype=grad.dtype)
            sum_per_cluster.index_add_(0, cluster_idx_long, grad)
            mean_per_cluster = sum_per_cluster / counts_t.unsqueeze(-1)
            return mean_per_cluster[cluster_idx_long]
        model.raw_materials.register_hook(_cluster_grad_hook)

    # All points emitted by init_visible_weighted are RX-visible by construction
    # (already ray-traced against the Mitsuba scene before FPS), so no per-point
    # post-FPS visibility re-test is needed.

    # No TX shadow mask. The A/B test in v4_ray_tracing_minimization_plan.md
    # showed shadow OFF is at least as good as shadow ON across 7 scenes
    # (mean +0.0068, 6/7 scenes improved or flat). The shadow mask was
    # dropped to remove ~600K rays/scene of init cost AND a function that
    # was net-hurting training quality.

    # Free Mitsuba scene — no more ray tracing for this scene
    rast.free_mi_scene()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # GT ADC
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_adc_ri = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32)).to(DEVICE)

    range_res = compute_range_res_from_cfg(config.config_file)

    # Pre-compute and cache the GT cartesian RA image. The GT does not change
    # between iterations.
    #
    # GPU pipeline (used by per-iter cart_corr_torch):
    #   * gt_polar (torch, GPU) — for the loss path
    #   * gt_cart_gpu_norm (torch, GPU, min-max normalized) — for cart_corr_torch
    #   * sample_grid (built once) — polar→cart sampling grid for grid_sample
    #
    # CPU image (used only at the end of training to save the best GT PNG):
    #   * ra_gt_cart_cached (numpy, via scipy griddata) — for save_ra_cartesian_png
    _gt_adc_for_eval = torch.from_numpy(gt_adc_ri.cpu().numpy()).float()
    _gt_ra_polar_cpu = adc_to_ra_image(_gt_adc_for_eval).numpy()
    ra_gt_cart_cached = ra_polar_to_cartesian(_gt_ra_polar_cpu, range_res)
    del _gt_adc_for_eval

    # GPU GT cart for the per-iter metric
    _gt_polar_gpu = torch.from_numpy(_gt_ra_polar_cpu.astype(np.float32)).to(DEVICE)
    n_az_polar, n_range_polar = _gt_polar_gpu.shape
    sample_grid = build_polar_to_cart_grid(
        n_az_polar, n_range_polar, range_res, grid_res=400, device=DEVICE)
    _gt_cart_gpu = polar_to_cart_torch(_gt_polar_gpu, sample_grid)
    _gt_min, _gt_max = _gt_cart_gpu.min(), _gt_cart_gpu.max()
    gt_cart_gpu_norm = (_gt_cart_gpu - _gt_min) / (_gt_max - _gt_min).clamp(min=1e-30)
    del _gt_polar_gpu, _gt_cart_gpu, _gt_min, _gt_max, _gt_ra_polar_cpu

    # Tier A1: pre-compute the GT min-max-normalized RA magnitude for the loss
    # path. The GT does not change between iters; the previous code was
    # recomputing adc_to_ra_complex(gt_adc_ri) inside compute_ra_loss_rp every
    # iter (~5 ms wasted/iter). Now computed once and reused on every backward.
    gt_loss_norm_cached = precompute_gt_loss_norm(gt_adc_ri)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Gaussian Training: {scene} (v4 hemisphere)")
        print(f"  Gaussians: {model.N}")
        print(f"  Iterations: {num_iters}")
        print(f"{'='*60}")

    # Active mask: FOV culling only (RX visibility is guaranteed at init)
    active_mask = cull_gaussians(model, rast)
    n_active = active_mask.sum().item()

    # Uniform hemisphere weights on the active set
    hemisphere_weights = torch.zeros(model.N, device=DEVICE)
    hemisphere_weights[active_mask] = 1.0
    vertex_areas = hemisphere_weights

    if verbose:
        print(f"  Active after culling: {n_active}/{model.N}")
        print(f"  Hemisphere weights: uniform = 1.0")

    # Antenna patterns are wrapped as Parameters only if learnable
    if _learn_patterns:
        tx_E = torch.nn.Parameter(rast.tx_antenna.E.clone())
        tx_H = torch.nn.Parameter(rast.tx_antenna.H.clone())
        rx_E = torch.nn.Parameter(rast.rx_antenna.E.clone())
        rx_H = torch.nn.Parameter(rast.rx_antenna.H.clone())
    else:
        tx_E = tx_H = rx_E = rx_H = None

    # Optimizer groups gated by LEARN_* flags (per-call overrides)
    param_groups = []
    clip_vals = {}
    if _learn_materials:
        param_groups.append(
            {"params": [model.raw_materials], "lr": 0.7, "name": "materials"})
        clip_vals["materials"] = 1.0
    if LEARN_POSITIONS:
        param_groups.append(
            {"params": [model.positions], "lr": 1.6e-4, "name": "positions"})
        clip_vals["positions"] = 1.0
    if _learn_normals:
        param_groups.append(
            {"params": [model.rotations], "lr": 2e-3, "name": "rotations"})
        clip_vals["rotations"] = 0.5
    if _learn_patterns:
        param_groups.append(
            {"params": [tx_E, tx_H, rx_E, rx_H], "lr": 0.05, "name": "patterns"})
        clip_vals["patterns"] = 1.0

    if not param_groups:
        # All LEARN_* flags off — pure inference, no optimization. Just do
        # one forward pass to measure cart_corr at init. This is the B0_zero
        # baseline / the (0,0,0) corner of the LEARN-flag matrix.
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode=bsdf_mode,
                disabled_components=disabled_components)
            ra_polar_t = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart_gpu = polar_to_cart_torch(ra_polar_t, sample_grid)
            cart_corr_zero = cart_corr_torch(ra_cart_gpu, gt_cart_gpu_norm).item()
        if verbose:
            print(f"  All learnable groups disabled — pure inference. cart_corr={cart_corr_zero:.4f}")
        if diagnostics_dir is not None:
            diag = MaterialDiagnostics(model.raw_materials, enabled=True)
            diag.finalize(model.raw_materials)
            diag.save(
                output_dir=diagnostics_dir,
                run_name=f'{run_name or "run"}__{scene}',
                mean_cart_corr=cart_corr_zero,
                per_scene_corr=[cart_corr_zero],
                ms_per_iter=0.0,
                drift=np.zeros(6, dtype=np.float32),
                fisher=np.zeros(6, dtype=np.float32))
        return cart_corr_zero, 0

    base_lrs = {g["name"]: g["lr"] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # mat_mode='global': all rows of raw_materials share one (6,) vector.
    # Backward hook averages gradients across rows so the update is identical
    # for every row. Post-step copy of row 0 → all rows keeps them in sync
    # against float drift. The (M, 6) tensor itself is unchanged.
    if mat_mode == 'global':
        def _global_grad_hook(grad):
            return grad.mean(dim=0, keepdim=True).expand_as(grad)
        model.raw_materials.register_hook(_global_grad_hook)
        if verbose:
            print(f"  mat_mode=global: rows are tied (grad averaged + post-step broadcast)")

    # Material column freeze: zero gradient columns listed in freeze_mat_cols.
    # Hook fires inside backward; must return the modified grad. Stays
    # zero in Adam state too because Adam moments init to zero and the
    # masked column never receives a non-zero update.
    if freeze_mat_cols is not None and len(freeze_mat_cols) > 0:
        cols = sorted(set(int(c) for c in freeze_mat_cols))
        if any(c < 0 or c > 5 for c in cols):
            raise ValueError(f"freeze_mat_cols must be in 0..5, got {cols}")
        mask = torch.ones(6, device=DEVICE)
        for c in cols:
            mask[c] = 0.0
        def _freeze_hook(grad):
            return grad * mask
        model.raw_materials.register_hook(_freeze_hook)
        if verbose:
            from mm25DGS_v4.material_diagnostics import PARAM_NAMES
            frozen_names = [PARAM_NAMES[c] for c in cols]
            print(f"  Frozen material columns: {cols} ({frozen_names})")

    if verbose:
        active_groups = [g["name"] for g in param_groups]
        print(f"  Learnable groups: {active_groups}")

    best_corr = -1.0
    best_iter = 0
    best_state = None
    best_ra_rend_cart = None
    best_ra_gt_cart = None
    t0 = time.time()

    diagnostics = MaterialDiagnostics(
        model.raw_materials,
        checkpoint_every=50,
        enabled=(diagnostics_dir is not None),
        capture_grad_stats=capture_grad_stats)

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        # Inject learnable antenna patterns (only when learn_patterns=True;
        # otherwise the rasterizer keeps its original tensors set at init).
        if _learn_patterns:
            rast.tx_antenna.E = tx_E
            rast.tx_antenna.H = tx_H
            rast.rx_antenna.E = rx_E
            rast.rx_antenna.H = rx_H

        # ONE forward pass per iter, used for BOTH backward and the metric.
        # We compute rp_real, rp_imag with grad enabled for the loss path,
        # and then compute the metric on the same tensors detached from the
        # autograd graph (no second render call).
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask,
            shadow_mask=None,
            bsdf_mode=bsdf_mode,
            disabled_components=disabled_components)

        loss, loss_dict = compute_ra_loss_rp(rp_real, rp_imag, gt_loss_norm_cached, loss_type=loss_type)

        # Per-iter cart_corr metric, computed on the same forward pass.
        # We detach from the autograd graph since the metric is logging-only.
        with torch.no_grad():
            ra_polar_t = range_profile_to_ra_mag(
                rp_real.detach(), rp_imag.detach())
            ra_cart_gpu = polar_to_cart_torch(ra_polar_t, sample_grid)
            cart_corr_t = cart_corr_torch(ra_cart_gpu, gt_cart_gpu_norm)
            cart_corr = cart_corr_t.item()

        loss.backward()

        # D2: record per-iter material gradient mean/std (across points)
        # BEFORE gradient clipping and optimizer step.
        diagnostics.record_grad(model.raw_materials, it)

        del loss, ra_polar_t

        for group in optimizer.param_groups:
            clip = clip_vals.get(group["name"], 1.0)
            for p in group["params"]:
                rms_clip_grad(p, clip)

        # Compressed schedule for 500-iter training (Option C):
        # warmup_iters=0 (skip), decay_start=200 (more iters at full LR
        # before cosine decay kicks in).
        lr_scale = get_lr_scale(
            it, total_iters=num_iters,
            warmup_iters=0, warmup_factor=1.0,
            decay_start=200)
        for group in optimizer.param_groups:
            group["lr"] = base_lrs[group["name"]] * lr_scale

        optimizer.step()

        # Global mode: enforce identical rows after the optimizer step
        if mat_mode == 'global':
            with torch.no_grad():
                model.raw_materials.copy_(
                    model.raw_materials[0:1].expand_as(model.raw_materials))

        # T3: post-step cluster tying — overwrite each row with its
        # cluster's centroid (mean over cluster of raw_materials)
        if cluster_idx_long is not None:
            with torch.no_grad():
                sum_per_cluster = torch.zeros(
                    material_clusters, 6, device=DEVICE, dtype=model.raw_materials.dtype)
                sum_per_cluster.index_add_(0, cluster_idx_long, model.raw_materials)
                mean_per_cluster = sum_per_cluster / counts_t.unsqueeze(-1)
                model.raw_materials.copy_(mean_per_cluster[cluster_idx_long])

        diagnostics.maybe_checkpoint(model.raw_materials, it)

        # Best-state tracking. Now happens every iter (no longer rounded
        # to multiples of 50) since the metric is computed every iter.
        if cart_corr > best_corr:
            best_corr = cart_corr
            best_iter = it
            best_state = {
                'model': {k: v.data.clone() for k, v in model.state_dict().items()},
            }
            if _learn_patterns:
                best_state['tx_E'] = tx_E.data.clone()
                best_state['tx_H'] = tx_H.data.clone()
                best_state['rx_E'] = rx_E.data.clone()
                best_state['rx_H'] = rx_H.data.clone()
            # Save the cart image only when the best improves (5-15 times
            # per scene typically). This is the only per-iter CPU touchpoint
            # in the training loop and it's bounded by the number of
            # improvements, not the iter count.
            best_ra_rend_cart = ra_cart_gpu.cpu().numpy()
            best_ra_gt_cart = ra_gt_cart_cached

        del rp_real, rp_imag, ra_cart_gpu

        # Logging cadence stays at every 50 iters to avoid flooding stdout.
        if it % 50 == 0 or it == num_iters - 1:
            if verbose:
                elapsed = time.time() - t0
                print(f"  iter {it:4d}: loss={loss_dict['ra_mse']:.6f}, "
                      f"cart_corr={cart_corr:.4f} (best={best_corr:.4f}@{best_iter}) "
                          f"[{elapsed:.1f}s, N={model.N}]")

    train_elapsed = time.time() - t0
    ms_per_iter = train_elapsed * 1000.0 / max(num_iters, 1)

    if verbose:
        print(f"\n  Best cart_corr: {best_corr:.4f} at iter {best_iter}")

    # Diagnostics: drift + Fisher + dump
    diagnostics.finalize(model.raw_materials)
    drift = diagnostics.compute_drift()
    fisher = np.zeros(6, dtype=np.float32)
    if diagnostics_dir is not None and model.raw_materials.requires_grad:
        def _fisher_render(model, rast, vertex_areas, active_mask, shadow_mask):
            return render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=shadow_mask,
                bsdf_mode=bsdf_mode,
                disabled_components=disabled_components)
        fisher = diagnostics.compute_fisher(
            model, rast, vertex_areas, active_mask,
            gt_loss_norm_cached,
            render_gaussians_fn=_fisher_render,
            compute_ra_loss_rp_fn=lambda r, i, g: compute_ra_loss_rp(r, i, g, loss_type=loss_type))
    if diagnostics_dir is not None:
        diagnostics.save(
            output_dir=diagnostics_dir,
            run_name=f'{run_name or "run"}__{scene}',
            mean_cart_corr=best_corr,
            per_scene_corr=[best_corr],
            ms_per_iter=ms_per_iter,
            drift=drift,
            fisher=fisher)

    # Save outputs
    output_dir = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', scene)
    os.makedirs(output_dir, exist_ok=True)

    if best_state is not None:
        from mmir.data.ra_utils import save_ra_cartesian_png

        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))
        np.save(os.path.join(output_dir, 'ra_rendered_cart.npy'), best_ra_rend_cart)
        np.save(os.path.join(output_dir, 'ra_gt_cart.npy'), best_ra_gt_cart)

        for scale in ('dB', 'linear'):
            save_ra_cartesian_png(
                best_ra_gt_cart,
                os.path.join(output_dir, f'gt_ra_{scale}.png'),
                range_res=range_res, scale=scale, title=f'GT ({scale})')
            save_ra_cartesian_png(
                best_ra_rend_cart,
                os.path.join(output_dir, f'rendered_ra_{scale}.png'),
                range_res=range_res, scale=scale, title=f'Rendered ({scale})')

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
            'mode': 'v4',
        }, f, indent=2)

    return best_corr, best_iter


def run_all_scenes(num_iters=500, target_n=50000):
    """Run v4 c6 training on all 7 benchmark scenes."""
    results = {}
    for scene in SCENES:
        corr, it = train_gaussians(scene, num_iters=num_iters, target_n=target_n)
        mmIR_metrics = json.load(open(
            os.path.join(TRAIN_OUTPUT_DIR, scene, 'best_metrics.json')))
        results[scene] = {
            'gauss_corr': corr,
            'mmIR_corr': mmIR_metrics['cart_corr'],
            'gap': abs(corr - mmIR_metrics['cart_corr']),
        }

    print(f"\n{'='*70}")
    print(f"v4 Hemisphere Results")
    print(f"{'='*70}")
    print(f"{'Scene':<25} {'mmIR':>8} {'v4':>8} {'Gap':>6}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*6}")
    for scene in SCENES:
        r = results[scene]
        print(f"{scene:<25} {r['mmIR_corr']:>8.4f} {r['gauss_corr']:>8.4f} "
              f"{r['gap']:>6.4f}")
    mean_corr = np.mean([results[s]['gauss_corr'] for s in SCENES])
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*6}")
    print(f"{'Mean':<25} {'':>8} {mean_corr:>8.4f}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--target_n', type=int, default=50000,
                        help='Target Gaussian count after FPS')
    args = parser.parse_args()

    if args.all:
        run_all_scenes(num_iters=args.iters, target_n=args.target_n)
    elif args.scene:
        train_gaussians(args.scene, num_iters=args.iters, target_n=args.target_n)
    else:
        print("Usage: --scene <name> or --all")
