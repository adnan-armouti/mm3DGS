"""mm25DGS_v4 — Novel View Synthesis (NVS) training across 9 radar frames.

This is a COPY of train_gaussian.py modified for multi-frame training. The
single-frame file is NOT edited to avoid risking bugs in the single-frame
pipeline. See md/v4_nvs_plan.md for the design rationale.

NVS setup:
  Each scene has 9 sequentially-aligned radar frames (e.g. scene
  seq_0_frame_135 has frames 131..139). The aligned configs live in
  /data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned.json.

  Train on frames {0, 2, 4, 6, 8}, test on {1, 3, 5, 7} (configurable
  via --train_indices / --test_indices).

NVS pipeline:
  1. Load all 9 aligned configs + GT ADC files for the scene.
  2. Create 9 Rasterizer instances (one per frame — same antenna pattern,
     different radar poses).
  3. Multi-frame init (init_visible_weighted_nvs):
     a. Load pcl.npy
     b. Union FOV: keep points within FOV of ANY of the 9 frames
     c. Per-frame visibility ray-test over the union
     d. Union of visibilities: keep points visible from at least one frame
     e. Cosine importance resample using max cos_bore across frames
     f. FPS to target_n (default 90K, same limit as single-frame)
     g. Compute per-frame active matrix (M, 9) — FOV & visibility per frame
  4. Training loop:
     - Each iter, accumulate gradients across all training frames
     - Forward pass per training frame uses active_mask = active_matrix[:, f]
     - Sum of per-frame MSE losses, one optimizer step per iter
  5. Evaluation:
     - Render each train AND test frame at the best model state
     - Save per-frame GT / rendered PNGs
     - Report mean cart_corr on train frames and on test frames separately

Outputs go to mm25DGS_v4/output_nvs/<scene>/ to avoid clobbering the
single-frame results in mm25DGS_v4/output/<scene>/.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v4.train_gaussian_nvs --scene seq_0_frame_135
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v4.train_gaussian_nvs --all
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
# NVS config loader — reads all 9 aligned radar configs per scene
# =========================================================================

ALIGNMENT_DATA_DIR = '/home/adnan/Desktop/mm3DGS/data/alignment_data'


def _list_aligned_frame_indices(scene):
    """Return the 9 radar frame indices for a scene by scanning
    alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned.json.
    """
    cascade_dir = os.path.join(ALIGNMENT_DATA_DIR, scene, 'cascade')
    if not os.path.isdir(cascade_dir):
        raise FileNotFoundError(
            f"NVS needs aligned configs at {cascade_dir}. Not found.")
    frames = []
    for fname in os.listdir(cascade_dir):
        if fname.startswith('cascaded_frame_') and fname.endswith('_aligned.json'):
            # Skip the *_alignment_log.json and *_aligned_2dof/gpu etc. variants
            mid = fname[len('cascaded_frame_'):-len('_aligned.json')]
            if mid.isdigit():
                frames.append(int(mid))
    frames.sort()
    if len(frames) != 9:
        raise RuntimeError(
            f"Expected 9 aligned frames in {cascade_dir}, found {len(frames)}: {frames}")
    return frames


def load_nvs_scene(scene):
    """Load all 9 frames of a scene for NVS training.

    Returns a dict with:
      'scene': scene name
      'frames': list of 9 ints — radar frame numbers (sorted)
      'configs': list of 9 str — path to cascaded_frame_<F>_aligned.json
      'adc_files': list of 9 str — path to cascaded_frame_<F>.npy
      'scene_file': str — path to mesh.ply (shared across frames)
      'tx_pattern_file': str — factory MMWCAS TX pattern .npy
      'rx_pattern_file': str — factory MMWCAS RX pattern .npy

    Uses the single-frame config loader to get the scene_file and factory
    antenna pattern paths, then overrides per-frame config and ADC files
    from alignment_data/<scene>/cascade/ and data/<scene>/radar/.
    """
    # Get scene_file and pattern paths from the single-frame loader
    base_cfg = load_trained_config(scene)
    scene_file = base_cfg.scene_file
    tx_pattern_file = base_cfg.tx_pattern_file
    rx_pattern_file = base_cfg.rx_pattern_file

    frames = _list_aligned_frame_indices(scene)
    cascade_dir = os.path.join(ALIGNMENT_DATA_DIR, scene, 'cascade')
    radar_dir = os.path.join(os.path.dirname(base_cfg.gt_adc_file))

    configs = []
    adc_files = []
    for f in frames:
        cfg_path = os.path.join(cascade_dir, f'cascaded_frame_{f}_aligned.json')
        adc_path = os.path.join(radar_dir, f'cascaded_frame_{f}.npy')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"{cfg_path} not found")
        if not os.path.exists(adc_path):
            raise FileNotFoundError(f"{adc_path} not found")
        configs.append(cfg_path)
        adc_files.append(adc_path)

    return {
        'scene': scene,
        'frames': frames,
        'configs': configs,
        'adc_files': adc_files,
        'scene_file': scene_file,
        'tx_pattern_file': tx_pattern_file,
        'rx_pattern_file': rx_pattern_file,
    }

# =========================================================================
# Learnable parameter toggles (matches mmIR's LEARN_* pattern in train.py)
# =========================================================================
# Each flag controls (a) whether the underlying parameter has requires_grad
# and (b) whether it appears in the optimizer's param_groups. Flip a flag,
# and the parameter is frozen / unfrozen uniformly.
LEARN_POSITIONS = False    # frozen — pcl positions are treated as fixed reference
LEARN_NORMALS   = True     # quaternion → surface normal; refines noisy pcl normals
LEARN_MATERIALS = True     # 6 ITU/Cook-Torrance params per point
LEARN_PATTERNS  = False    # TX/RX antenna E/H planes — PERMANENT False per user direction (2026-04-13)

# When True (default after the raw-MSE plan), skip mmIR-trained antenna
# pattern injection at init and use factory patterns from the MMWCAS .npy
# files instead. mmIR-trained patterns absorb per-scene amplitude bias and
# break the cross-scene scale uniformity required for raw MSE. See
# md/v4_unnormalized_mse_loss_plan.md Phase α findings.
USE_FACTORY_PATTERNS = True

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


def precompute_gt_loss_norm(gt_adc_ri, loss_type='mse_raw'):
    """One-shot computation of the GT tensor for the loss path.

    The GT does not change during training; the loss used to recompute it
    every iter (~5 ms wasted/iter). Call this once before the training loop
    and pass the result into compute_ra_loss_rp.

    For loss_type='mse'    : returns GT min-max normalized RA magnitude.
    For loss_type='pearson': same (Pearson is scale/shift invariant, so
                             normalization doesn't affect the result).
    For loss_type='mse_raw': returns raw GT RA magnitude (no normalization).
    """
    with torch.no_grad():
        ra_gt = adc_to_ra_complex(gt_adc_ri)
        ra_gt_mag = torch.abs(ra_gt)
        if loss_type == 'mse_raw':
            return ra_gt_mag.detach()
        mn, mx = ra_gt_mag.min(), ra_gt_mag.max()
        return ((ra_gt_mag - mn) / (mx - mn).clamp(min=1e-30)).detach()


def compute_ra_loss_rp(rp_real, rp_imag, gt_cached, loss_type='mse_raw'):
    """RA loss between rendered range profiles and the cached GT.

    loss_type:
      'mse'     — min-max-normalized MSE on RA magnitude (legacy)
      'pearson' — 1 - Pearson(rendered_flat, gt_flat); scale/shift invariant
      'mse_raw' — MSE on raw |RA| magnitude, no normalization. Requires the
                  forward-model scale to match GT scale (see the 100x C_radar
                  boost); otherwise the loss is dominated by the mismatch.
    """
    ra_rendered = range_profile_to_ra(rp_real, rp_imag)
    ra_rend_mag = torch.abs(ra_rendered)

    if loss_type == 'mse':
        mn = ra_rend_mag.min()
        mx = ra_rend_mag.max()
        rend_norm = (ra_rend_mag - mn) / (mx - mn).clamp(min=1e-30)
        loss = torch.mean((rend_norm - gt_cached) ** 2)
        return loss, {"ra_mse": loss.item()}
    elif loss_type == 'pearson':
        r_flat = ra_rend_mag.reshape(-1)
        g_flat = gt_cached.reshape(-1)
        r_mean = r_flat.mean()
        g_mean = g_flat.mean()
        r_c = r_flat - r_mean
        g_c = g_flat - g_mean
        num = (r_c * g_c).sum()
        den = torch.sqrt((r_c * r_c).sum() * (g_c * g_c).sum()).clamp(min=1e-30)
        corr = num / den
        loss = 1.0 - corr
        return loss, {"pearson_loss": loss.item()}
    elif loss_type == 'mse_raw':
        # Raw MSE on |RA| magnitude — depends on rendered scale matching GT
        # scale. Divide by the square of GT mean to bring the loss into an
        # O(1) range regardless of absolute scale (helps LR tuning) without
        # introducing any scale invariance into the gradient path.
        gt_scale = gt_cached.mean().detach().clamp(min=1e-30) ** 2
        loss = ((ra_rend_mag - gt_cached) ** 2).mean() / gt_scale
        return loss, {"ra_mse_raw": loss.item()}
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

def init_visible_weighted_nvs(scene, rasts, target_n=90000,
                              cos_bore_min=0.1,
                              n_intermediate=None,
                              device=DEVICE):
    """Multi-frame NVS init: union FOV + per-frame visibility + per-frame
    active matrix.

    Args:
      rasts: list of Rasterizer instances, one per frame (typically 9).
             All must share the same mesh_file; different poses per frame.
      target_n: final number of points after FPS.

    Returns: tuple (model, active_matrix) where
      model — PointPrimitives with positions, normals, raw_materials init'd
      active_matrix — (target_n, n_frames) bool tensor on device. True if
                      point is both in the FOV of frame f AND visible from
                      frame f's RX center. Use as per-frame render mask.

    Pipeline:
      1. Load pcl.npy (shared across all frames).
      2. Per-frame FOV mask + cos_bore.
      3. Union FOV: keep points in FOV of at least one frame.
      4. Per-frame visibility ray-tests over the union.
      5. Union visibility: keep points visible from at least one frame.
      6. Cosine importance resample weighted by max_f cos_bore_f.
      7. FPS to target_n.
      8. Compute final per-frame active matrix for the selected points.

    Memory note: the (target_n, 9) active matrix is ~810 KB bool, trivial.
    Per-frame visibility ray tests are ~1.5 M rays × 9 frames = 14 M rays
    but each is fast; total init time ~3 s per scene.
    """
    if n_intermediate is None:
        n_intermediate = max(200000, 3 * target_n)

    n_frames = len(rasts)
    if n_frames < 1:
        raise ValueError(f"init_visible_weighted_nvs needs ≥1 rasterizer, got {n_frames}")

    # The pcl.npy lives alongside mesh.ply in the scene directory.
    pcl_path = rasts[0]._mesh_file.replace('scene/mesh.ply', 'scene/pcl.npy')
    if not os.path.exists(pcl_path):
        raise FileNotFoundError(
            f"pcl.npy not found at {pcl_path}. NVS init requires the point cloud.")

    pcl = np.load(pcl_path)
    if pcl.shape[1] < 6:
        raise ValueError(
            f"{pcl_path} has shape {pcl.shape}; expected ≥6 cols (xyz + normals).")

    xyz_full = pcl[:, :3].astype(np.float32)
    nrm_full = pcl[:, 3:6].astype(np.float32)
    nrm_full = nrm_full / np.maximum(
        np.linalg.norm(nrm_full, axis=1, keepdims=True), 1e-12)
    # LiDAR return intensity (col 6) — the radar/LiDAR analog of RGB used
    # by edge-preserving smoothness in optical inverse-rendering papers.
    # If the pcl doesn't have intensity, fall back to a constant.
    if pcl.shape[1] >= 7:
        int_full = pcl[:, 6].astype(np.float32)
    else:
        int_full = np.zeros(len(xyz_full), dtype=np.float32)
    N_full = len(xyz_full)
    print(f"  [nvs init] Full pcl: {N_full} points, {n_frames} frames "
          f"(intensity range {int_full.min():.0f}-{int_full.max():.0f})")

    # --- Step 1: Per-frame FOV + cos_bore ---
    # For each frame, compute cos_bore, dist, fov_mask against that frame's
    # rx_center and boresight. Then union the fov masks, and track the max
    # cos_bore across frames (for importance resample weighting).
    fov_union = np.zeros(N_full, dtype=bool)
    max_cos_bore = np.full(N_full, -1.0, dtype=np.float32)
    per_frame_fov_mask = []   # list of bool arrays (N_full,), for later indexing
    per_frame_cos_bore = []   # list of float arrays (N_full,)
    per_frame_rx_center = []  # list of np.ndarray (3,)

    for f_idx, rast in enumerate(rasts):
        rx_center = rast.rx_positions.mean(dim=0).cpu().numpy()
        boresight = rast.tx_boresights.mean(dim=0).cpu().numpy()
        boresight = boresight / max(np.linalg.norm(boresight), 1e-8)

        delta = xyz_full - rx_center
        dist = np.linalg.norm(delta, axis=1).clip(min=1e-6)
        dir_to_point = delta / dist[:, None]
        cos_bore_f = (dir_to_point * boresight).sum(axis=-1)

        max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))
        fov_f = (cos_bore_f > cos_bore_min) & (dist > 1.5) & (dist < max_range)

        fov_union |= fov_f
        max_cos_bore = np.maximum(max_cos_bore, np.where(fov_f, cos_bore_f, -1.0))
        per_frame_fov_mask.append(fov_f)
        per_frame_cos_bore.append(cos_bore_f)
        per_frame_rx_center.append(rx_center)

    print(f"  [nvs init] After union FOV ({n_frames} frames): {fov_union.sum()} points")

    # --- Step 2: Per-frame visibility ray tests over the union ---
    # Run one ray test per frame. Each test only operates on the union-FOV
    # subset, so total work is ~9 × ~1.5 M = ~14 M rays. Still fast.
    xyz_union = xyz_full[fov_union]

    # We need the Mitsuba scene. Load on the first rasterizer and reuse its
    # scene for all frame visibility tests (scene geometry is frame-
    # independent; only the RX center changes).
    rasts[0].load_mi_scene()
    mi_scene = rasts[0]._mi_scene

    visibility_union_per_frame = []   # list of bool arrays over xyz_union indices
    for f_idx, rast in enumerate(rasts):
        rx_center_f = per_frame_rx_center[f_idx]
        vis_f = _ray_test_visibility_batched(xyz_union, rx_center_f, mi_scene)
        visibility_union_per_frame.append(vis_f)

    rasts[0].free_mi_scene()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # --- Step 3: Union of visibilities over union-FOV subset ---
    vis_any = np.zeros(len(xyz_union), dtype=bool)
    for v in visibility_union_per_frame:
        vis_any |= v
    n_vis = vis_any.sum()
    print(f"  [nvs init] After union visibility: {n_vis} points")

    if n_vis < 100:
        raise RuntimeError(f"NVS init: only {n_vis} points survived union visibility.")

    xyz_pool = xyz_union[vis_any]
    nrm_pool = nrm_full[fov_union][vis_any]
    int_pool = int_full[fov_union][vis_any]
    # max_cos_bore is over the FULL pcl; index it down to the pool.
    max_cos_bore_pool = max_cos_bore[fov_union][vis_any]

    # --- Step 4: Cosine-hemisphere importance resample ---
    weights = np.maximum(max_cos_bore_pool, 0.01)
    probs = weights / weights.sum()
    n_resample = min(n_intermediate, 3 * target_n)
    rng = np.random.default_rng(42)
    sampled_idx = rng.choice(len(xyz_pool), size=n_resample, replace=True, p=probs)
    unique_idx = np.unique(sampled_idx)
    xyz_resample = xyz_pool[unique_idx]
    nrm_resample = nrm_pool[unique_idx]
    int_resample = int_pool[unique_idx]
    print(f"  [nvs init] After cosine importance resample: {len(xyz_resample)} points")

    # --- Step 5: FPS to target_n ---
    if len(xyz_resample) > target_n:
        pts_t = torch.from_numpy(xyz_resample).to(device)
        sel = _farthest_point_sampling(pts_t, target_n).cpu().numpy()
        xyz_final = xyz_resample[sel]
        nrm_final = nrm_resample[sel]
        int_final = int_resample[sel]
    else:
        xyz_final = xyz_resample
        nrm_final = nrm_resample
        int_final = int_resample
    N_final = len(xyz_final)
    print(f"  [nvs init] After FPS: {N_final} points")

    # --- Step 6: Build the model ---
    model = PointPrimitives(N_final, device=device)
    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz_final).to(device))
        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        model.raw_materials.copy_(
            torch.from_numpy(np.tile(raw_default, (N_final, 1))).to(device))
        quats = _normals_to_quaternions(nrm_final)
        model.rotations.copy_(torch.from_numpy(quats).to(device))

    # --- Step 7: Per-frame active matrix for the FINAL selected points ---
    # For each selected point, check FOV and visibility against each frame.
    # FOV: recompute the cos_bore > min threshold per frame using the final
    # positions.
    # Visibility: ray-test the final positions against each frame's rx_center.
    print(f"  [nvs init] Computing per-frame active matrix ({N_final} pts × {n_frames} frames)...")
    active_matrix_np = np.zeros((N_final, n_frames), dtype=bool)

    # Re-load the Mitsuba scene for the final visibility passes
    rasts[0].load_mi_scene()
    mi_scene = rasts[0]._mi_scene

    for f_idx, rast in enumerate(rasts):
        rx_center_f = per_frame_rx_center[f_idx]
        boresight_f = rast.tx_boresights.mean(dim=0).cpu().numpy()
        boresight_f = boresight_f / max(np.linalg.norm(boresight_f), 1e-8)

        delta_f = xyz_final - rx_center_f
        dist_f = np.linalg.norm(delta_f, axis=1).clip(min=1e-6)
        dir_f = delta_f / dist_f[:, None]
        cos_bore_final = (dir_f * boresight_f).sum(axis=-1)
        max_range = rast.K * 299792458.0 / (2.0 * rast.slope * (rast.K / rast.sample_rate))
        fov_final_f = (cos_bore_final > cos_bore_min) & (dist_f > 1.5) & (dist_f < max_range)

        # Ray-test the FOV-surviving subset
        if fov_final_f.any():
            vis_final_f = _ray_test_visibility_batched(
                xyz_final[fov_final_f], rx_center_f, mi_scene)
            active_f = fov_final_f.copy()
            fov_idx = np.where(fov_final_f)[0]
            active_f[fov_idx] = vis_final_f
        else:
            active_f = np.zeros(N_final, dtype=bool)

        active_matrix_np[:, f_idx] = active_f

    rasts[0].free_mi_scene()
    gc.collect()
    torch.cuda.empty_cache()

    active_matrix = torch.from_numpy(active_matrix_np).to(device)
    intensity_t = torch.from_numpy(int_final).to(device)
    # Print per-frame active counts
    for f_idx in range(n_frames):
        n_active = int(active_matrix[:, f_idx].sum().item())
        print(f"  [nvs init] Frame {f_idx}: {n_active}/{N_final} active "
              f"({100.0*n_active/N_final:.1f}%)")

    return model, active_matrix, intensity_t


# =========================================================================
# Renderer wrapper (uniform hemisphere weights)
# =========================================================================

def render_gaussians(model, rast, vertex_areas, active_mask=None,
                     shadow_mask=None, detach_phase=True, bsdf_mode='full',
                     disabled_components=None, raw_materials_override=None):
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
    raw_materials = (raw_materials_override if raw_materials_override is not None
                     else model.raw_materials)

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


# =========================================================================
# NVS training loop — multi-frame gradient accumulation
# =========================================================================

def _build_rasts_for_scene(scene_data, device=DEVICE):
    """Build a Rasterizer per frame, sharing the mesh + factory antenna."""
    rasts = []
    for cfg_path in scene_data['configs']:
        r = Rasterizer(
            config_file=cfg_path,
            mesh_file=scene_data['scene_file'],
            tx_pattern_file=scene_data['tx_pattern_file'],
            rx_pattern_file=scene_data['rx_pattern_file'],
            device=device,
        )
        rasts.append(r)
    return rasts


def _load_gt_for_frame(adc_path, device=DEVICE):
    """Load a GT ADC .npy and return the (TX,RX,K,2) torch tensor on device."""
    gt_adc_np = np.load(adc_path)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    return torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32)).to(device)


class _LerpRast:
    """Temporarily mutate a Rasterizer's TX/RX positions and boresights to
    the linear interpolation between rast_a and rast_b at parameter alpha
    in [0, 1] (0 = a, 1 = b). Restores on __exit__.

    Usage:
        with _LerpRast(rast_a, rast_b, alpha=0.5):
            # rast_a now holds the midpoint pose
            render_gaussians(model, rast_a, ...)
    """
    def __init__(self, rast_a, rast_b, alpha):
        self.rast_a = rast_a
        self.rast_b = rast_b
        self.alpha = float(alpha)

    def __enter__(self):
        a, b = self.rast_a, self.rast_b
        self._tx_pos = a.tx_positions.clone()
        self._rx_pos = a.rx_positions.clone()
        self._tx_bore = a.tx_boresights.clone()
        self._rx_bore = a.rx_boresights.clone()
        with torch.no_grad():
            t = self.alpha
            a.tx_positions.copy_((1 - t) * a.tx_positions + t * b.tx_positions)
            a.rx_positions.copy_((1 - t) * a.rx_positions + t * b.rx_positions)
            # Linear interpolation of boresights then re-normalize each
            new_tx_bore = (1 - t) * self._tx_bore + t * b.tx_boresights
            new_tx_bore = new_tx_bore / new_tx_bore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            new_rx_bore = (1 - t) * self._rx_bore + t * b.rx_boresights
            new_rx_bore = new_rx_bore / new_rx_bore.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            a.tx_boresights.copy_(new_tx_bore)
            a.rx_boresights.copy_(new_rx_bore)
        return self.rast_a

    def __exit__(self, *args):
        a = self.rast_a
        a.tx_positions.copy_(self._tx_pos)
        a.rx_positions.copy_(self._rx_pos)
        a.tx_boresights.copy_(self._tx_bore)
        a.rx_boresights.copy_(self._rx_bore)
        return False


class _PerturbedRast:
    """Temporarily mutate a Rasterizer's TX/RX positions and boresights to a
    randomly perturbed pose for one render. Restores on __exit__.

    Usage:
        with _PerturbedRast(rast, eps_pos=0.01, eps_rot=0.02):
            render_gaussians(model, rast, ...)
    """
    def __init__(self, rast, eps_pos, eps_rot, gen=None):
        self.rast = rast
        self.eps_pos = eps_pos
        self.eps_rot = eps_rot
        self.gen = gen

    def __enter__(self):
        r = self.rast
        self._tx_pos = r.tx_positions.clone()
        self._rx_pos = r.rx_positions.clone()
        self._tx_bore = r.tx_boresights.clone()
        self._rx_bore = r.rx_boresights.clone()

        with torch.no_grad():
            # Random translation applied uniformly to all TX/RX (rigid body)
            delta = torch.randn(3, device=r.tx_positions.device,
                                generator=self.gen) * self.eps_pos
            r.tx_positions.add_(delta)
            r.rx_positions.add_(delta)

            # Random small-angle rotation around a random axis, applied to
            # all boresights (rigid body). Use Rodrigues' formula at small
            # angle: v' ≈ v + θ (k × v), where k is a unit axis.
            axis = torch.randn(3, device=r.tx_positions.device,
                               generator=self.gen)
            axis = axis / axis.norm().clamp(min=1e-8)
            theta = (torch.rand(1, device=r.tx_positions.device,
                                generator=self.gen).item() * 2 - 1) * self.eps_rot
            # rotation matrix via Rodrigues for small θ
            K = torch.tensor([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]],
                             device=r.tx_positions.device, dtype=r.tx_boresights.dtype)
            R = torch.eye(3, device=K.device, dtype=K.dtype) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
            r.tx_boresights.copy_(r.tx_boresights @ R.T)
            r.rx_boresights.copy_(r.rx_boresights @ R.T)
        return self

    def __exit__(self, *args):
        r = self.rast
        r.tx_positions.copy_(self._tx_pos)
        r.rx_positions.copy_(self._rx_pos)
        r.tx_boresights.copy_(self._tx_bore)
        r.rx_boresights.copy_(self._rx_bore)
        return False


def _render_and_metric(model, rast, vertex_areas, active_mask,
                       sample_grid, gt_cart_norm, bsdf_mode='full',
                       raw_materials_override=None,
                       disabled_components=None):
    """Render one frame, return (rp_real, rp_imag, ra_cart_gpu, cart_corr)."""
    rp_real, rp_imag = render_gaussians(
        model, rast, vertex_areas=vertex_areas,
        active_mask=active_mask, shadow_mask=None,
        bsdf_mode=bsdf_mode,
        disabled_components=disabled_components,
        raw_materials_override=raw_materials_override)
    with torch.no_grad():
        ra_polar_t = range_profile_to_ra_mag(rp_real.detach(), rp_imag.detach())
        ra_cart_gpu = polar_to_cart_torch(ra_polar_t, sample_grid)
        cc = cart_corr_torch(ra_cart_gpu, gt_cart_norm).item()
    return rp_real, rp_imag, ra_cart_gpu, cc


def train_nvs(scene,
              train_indices=(0, 2, 4, 6, 8),
              test_indices=(1, 3, 5, 7),
              num_iters=300,
              target_n=90000,
              mat_lr=0.01,
              rot_lr=5e-3,
              loss_type='pearson',
              material_clusters=0,
              mat_basis='none',           # 'none' or 'poly2'
              mat_basis_lr=0.05,
              lambda_normal_anchor=0.0,   # >0 enables anchor reg
              facing_grad_weight=False,   # >>> NEW: gradient weighting by per-point facing score
              facing_power=1.0,           # exponent on the facing weight
              render_consist=False,       # >>> NEW: render-consistency reg
              render_consist_lambda=0.1,  # weight on the render-consistency loss
              render_consist_eps_pos=0.01,  # radar position perturbation in metres
              render_consist_eps_rot=0.02,  # boresight rotation in radians (~1.15°)
              consist_anchor=False,         # >>> NEW: anchored-smoothness reg (correct version)
              consist_anchor_lambda=0.1,    # weight on the anchored-smoothness loss
              pose_lerp=False,              # >>> NEW: pose-midpoint linearity reg
              pose_lerp_lambda=0.1,
              bracket_anchor=False,         # >>> NEW: render @ test pose, supervise on avg bracket
              bracket_anchor_lambda=0.1,
              bracket_target='model',       # 'model' (avg of model renders) or 'gt' (avg of GT |RA|)
              bracket_detach=False,         # if True, brackets are rendered w/o grad (target is fixed)
              bracket_pose_weighting=False, # if True, per-frame λ ∝ exp(-α · max bracket pose dist)
              bracket_pose_alpha=1.5,       # exponent for the pose-distance decay
              edge_smooth=False,            # >>> NEW: edge-preserving spatial smoothness on materials
              edge_smooth_lambda=1.0,
              edge_smooth_k=8,              # kNN neighborhood size
              edge_smooth_sigma_p=0.3,      # position scale (m)
              edge_smooth_sigma_n=0.3,      # normal angular scale
              edge_smooth_sigma_i=200.0,    # intensity scale (raw LiDAR units)
              intensity_clusters=0,         # >>> NEW: K clusters in (intensity, normal) space
              intensity_residual_lambda=0.01,  # L2 penalty on per-point residual
              intensity_lr=0.05,            # LR for cluster centers
              disabled_components=None,     # diagnostic: disable BSDF lobes {'ka','spm',...}
              init_state_dict=None,
              verbose=True):
    """Multi-frame NVS trainer for one scene.

    Trains on `train_indices` frames (indices into the 9 sorted aligned
    frames) and evaluates on all 9 (train + test) at the best model state.

    Returns dict with per-frame cart_corr at best-train state.
    """
    train_indices = list(train_indices)
    test_indices = list(test_indices)
    bsdf_mode = 'full'

    scene_data = load_nvs_scene(scene)
    n_frames = len(scene_data['frames'])
    if verbose:
        print(f"\n{'='*60}")
        print(f"NVS Training: {scene}")
        print(f"  frames: {scene_data['frames']}")
        print(f"  train idx: {train_indices}  test idx: {test_indices}")
        print(f"{'='*60}")

    # Build rasterizers (one per frame). Mitsuba scene loaded only during init.
    rasts = _build_rasts_for_scene(scene_data)

    # Multi-frame init — produces (model, active_matrix [N, n_frames])
    model, active_matrix, intensity_t = init_visible_weighted_nvs(
        scene, rasts, target_n=target_n)

    # Optional warm-start: copy a pre-trained state into the model. Only
    # works when (scene, target_n) match the source run since FPS point
    # selection is deterministic — same scene + same target_n → identical
    # point set → identical state dict shapes.
    if init_state_dict is not None:
        model_keys = set(model.state_dict().keys())
        provided = set(init_state_dict.keys())
        if model_keys != provided:
            raise ValueError(
                f"init_state_dict keys mismatch. expected {model_keys}, got {provided}")
        with torch.no_grad():
            for k, v in init_state_dict.items():
                model.state_dict()[k].copy_(v)
        if verbose:
            print(f"  WARM START: loaded init_state_dict ({sum(v.numel() for v in init_state_dict.values())} params)")

    # Free Mitsuba scene refs on all rasts after init (no more ray-tracing)
    for r in rasts:
        r.free_mi_scene()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Per-frame caches: GT loss tensor, sample_grid, gt_cart_gpu_norm,
    # range_res, ra_gt_cart_cached (CPU, for PNG saving at the end).
    per_frame = []
    for f_idx in range(n_frames):
        gt_adc_ri = _load_gt_for_frame(scene_data['adc_files'][f_idx])
        range_res = compute_range_res_from_cfg(scene_data['configs'][f_idx])

        # CPU GT cart for end-of-training PNG output
        _gt_polar_cpu = adc_to_ra_image(gt_adc_ri.cpu()).numpy()
        ra_gt_cart_cached = ra_polar_to_cartesian(_gt_polar_cpu, range_res)

        _gt_polar_gpu = torch.from_numpy(_gt_polar_cpu.astype(np.float32)).to(DEVICE)
        n_az_polar, n_range_polar = _gt_polar_gpu.shape
        sample_grid = build_polar_to_cart_grid(
            n_az_polar, n_range_polar, range_res, grid_res=400, device=DEVICE)
        _gt_cart_gpu = polar_to_cart_torch(_gt_polar_gpu, sample_grid)
        _gt_min, _gt_max = _gt_cart_gpu.min(), _gt_cart_gpu.max()
        gt_cart_gpu_norm = (_gt_cart_gpu - _gt_min) / (_gt_max - _gt_min).clamp(min=1e-30)
        del _gt_polar_gpu, _gt_cart_gpu, _gt_min, _gt_max, _gt_polar_cpu

        gt_loss_norm_cached = precompute_gt_loss_norm(gt_adc_ri, loss_type=loss_type)

        # Raw |RA| (not normalized) — used by anchored-smoothness reg to take
        # pairwise differences in compatible units.
        with torch.no_grad():
            gt_ra_raw = adc_to_ra_complex(gt_adc_ri).abs().detach()

        per_frame.append({
            'f_idx': f_idx,
            'frame': scene_data['frames'][f_idx],
            'gt_adc_ri': gt_adc_ri,
            'range_res': range_res,
            'sample_grid': sample_grid,
            'gt_cart_gpu_norm': gt_cart_gpu_norm,
            'gt_loss_cached': gt_loss_norm_cached,
            'gt_ra_raw': gt_ra_raw,
            'ra_gt_cart_cached': ra_gt_cart_cached,
        })

    # T3-style spatial material clustering. Tie raw_materials within each
    # cluster via a backward hook (averages grad across cluster members)
    # plus a post-step copy that broadcasts each cluster's centroid back
    # to its members. Reduces effective material dof from (N,6) → (K,6).
    cluster_idx_long = None
    cluster_counts_t = None
    if material_clusters > 0:
        with torch.no_grad():
            feat_pos = model.positions.detach().cpu().numpy()
            feat_nrm = model.get_normals().detach().cpu().numpy()
            feats = np.concatenate([feat_pos * 0.1, feat_nrm], axis=1)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=material_clusters, n_init=5,
                        random_state=42, max_iter=100)
            labels = km.fit_predict(feats)
            cluster_idx_long = torch.from_numpy(labels.astype(np.int64)).to(DEVICE)
        if verbose:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  T3 clusters: {material_clusters} groups, sizes "
                  f"min={counts.min()} max={counts.max()} mean={counts.mean():.0f}")

        cluster_counts_t = torch.zeros(material_clusters, device=DEVICE)
        cluster_counts_t.scatter_add_(
            0, cluster_idx_long,
            torch.ones_like(cluster_idx_long, dtype=torch.float32))
        cluster_counts_t = cluster_counts_t.clamp(min=1.0)

        def _cluster_grad_hook(grad):
            sum_per = torch.zeros(material_clusters, 6,
                                  device=DEVICE, dtype=grad.dtype)
            sum_per.index_add_(0, cluster_idx_long, grad)
            mean_per = sum_per / cluster_counts_t.unsqueeze(-1)
            return mean_per[cluster_idx_long]
        model.raw_materials.register_hook(_cluster_grad_hook)

    # ------------------------------------------------------------
    # Facing-score gradient weighting (motivated by scenario_diff
    # analysis: ~19 % of points are near-vertical surfels seen at
    # grazing angles by the horizontal radar; their BSDF gradient is
    # weak and noise-dominated, and S1 vs S2 normals drift in opposite
    # directions for them. Weighting the gradient on raw_materials and
    # rotations by a per-point "facing score" suppresses these noisy
    # updates while preserving learning at well-facing points.)
    #
    # facing_score[i] = max over TRAIN frames f of:
    #     |cos(normal_init[i], radar_dir[i,f])| * max(0, cos(boresight_f, -radar_dir[i,f]))
    #
    # That is, max over train frames of (how facing the point is) ×
    # (how in-FOV the point is for that frame). Score is in [0, 1].
    # ------------------------------------------------------------
    facing_weight_t = None
    if facing_grad_weight:
        with torch.no_grad():
            n0_t = model.get_normals().detach()                 # (N, 3) init normals
            pos_t = model.positions.detach()                     # (N, 3)
            facing_score = torch.zeros(model.N, device=DEVICE, dtype=torch.float32)
            for f_idx in train_indices:
                if not (0 <= f_idx < n_frames):
                    continue
                rast = rasts[f_idx]
                rx_c = rast.rx_positions.mean(dim=0)             # (3,)
                bs = rast.tx_boresights.mean(dim=0)
                bs = bs / bs.norm().clamp(min=1e-6)
                delta = rx_c.unsqueeze(0) - pos_t                # (N, 3) point→radar
                dist = delta.norm(dim=-1).clamp(min=1e-6)
                dir_to_rad = delta / dist.unsqueeze(-1)
                cos_normal = (n0_t * dir_to_rad).sum(-1).abs()    # (N,) facing
                cos_bore = ((-dir_to_rad) * bs).sum(-1).clamp(min=0.0)  # (N,) in-FOV
                score_f = cos_normal * cos_bore
                facing_score = torch.maximum(facing_score, score_f)
            facing_weight_t = facing_score.clamp(min=0.0, max=1.0)
            if facing_power != 1.0:
                facing_weight_t = facing_weight_t.pow(facing_power)

        if verbose:
            fw_np = facing_weight_t.cpu().numpy()
            print(f"  FACING WEIGHT: power={facing_power} | "
                  f"mean={fw_np.mean():.3f}  median={np.median(fw_np):.3f}  "
                  f"<0.1: {100*(fw_np<0.1).mean():.1f}%  "
                  f"<0.3: {100*(fw_np<0.3).mean():.1f}%  "
                  f"<0.5: {100*(fw_np<0.5).mean():.1f}%")

        # Backward hooks: gradient on materials/normals scaled by the
        # per-point facing weight. The hook fires inside .backward() and
        # transforms grad before the optimizer sees it.
        fw_mat = facing_weight_t.unsqueeze(-1)   # (N, 1) → broadcasts to (N, 6)
        fw_rot = facing_weight_t.unsqueeze(-1)   # (N, 1) → broadcasts to (N, 4)
        model.raw_materials.register_hook(lambda g, w=fw_mat: g * w)
        model.rotations.register_hook(lambda g, w=fw_rot: g * w)

    # Vertex areas: uniform 1.0 across the model. Per-frame active masks
    # come from active_matrix; render_gaussians takes active_mask which
    # both indexes positions/normals/raw_materials AND the areas tensor.
    vertex_areas = torch.ones(model.N, device=DEVICE)

    # ------------------------------------------------------------
    # Material gauge collapse — replace per-point materials with a
    # learnable function of (position, init_normal). This kills the
    # per-point degree of freedom that the parameter analysis showed
    # is gauge-ambiguous in the 5-frame loss.
    #
    # Parameterization:
    #   features[i]   = poly2(pos_norm[i], n0[i])    # (N, F), frozen
    #   raw_mat[i]    = base + W @ features[i]       # (N, 6) differentiable
    # where:
    #   pos_norm = (positions - mean) / std         # standardized once
    #   n0       = init normals (from pcl)          # frozen
    #   base     = (6,)                              # learnable
    #   W        = (F, 6)                            # learnable
    #
    # Total learnable: 6 + F*6 ≈ 168 params (vs 60K for per-point at N=10K)
    # ------------------------------------------------------------
    mat_basis_params = None
    mat_basis_features = None
    if mat_basis == 'poly2':
        with torch.no_grad():
            pos = model.positions
            n0_init = model.get_normals()
            pos_mean = pos.mean(0, keepdim=True)
            pos_std = pos.std(0, keepdim=True).clamp(min=1e-6)
            pos_norm = (pos - pos_mean) / pos_std        # (N, 3) standardized
            xyz = pos_norm
            nrm = n0_init                                  # already unit
            # Polynomial features of degree 2 over (xyz, nrm) → 6 + 21 = 27
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            nx, ny, nz = nrm[:, 0], nrm[:, 1], nrm[:, 2]
            ones = torch.ones_like(x)
            linear = torch.stack([x, y, z, nx, ny, nz], dim=1)         # (N, 6)
            quad = torch.stack([
                x*x, y*y, z*z, x*y, x*z, y*z,
                nx*nx, ny*ny, nz*nz, nx*ny, nx*nz, ny*nz,
                x*nx, x*ny, x*nz, y*nx, y*ny, y*nz, z*nx, z*ny, z*nz,
            ], dim=1)                                                  # (N, 21)
            mat_basis_features = torch.cat(
                [ones.unsqueeze(1), linear, quad], dim=1).contiguous()  # (N, 28)
            F_dim = mat_basis_features.shape[1]
        mat_base = torch.nn.Parameter(
            torch.zeros(6, device=DEVICE, dtype=mat_basis_features.dtype))
        mat_W = torch.nn.Parameter(
            torch.zeros(F_dim, 6, device=DEVICE, dtype=mat_basis_features.dtype))
        # Initialize base from the inverse-reparam of ITU concrete so that
        # at iter 0 the materials match the per-point mode default.
        with torch.no_grad():
            init_raw = inverse_reparameterize_torch(ITU_CONCRETE)
            mat_base.copy_(torch.from_numpy(init_raw).to(DEVICE))
        mat_basis_params = (mat_base, mat_W)
        # Freeze the per-point raw_materials parameter — it's no longer
        # used for gradient flow in basis mode.
        model.raw_materials.requires_grad_(False)
        if verbose:
            print(f"  MAT BASIS: poly2, F={F_dim}, total mat params="
                  f"{6 + F_dim * 6} (vs {model.N * 6} per-point)")

    # ------------------------------------------------------------
    # Intensity-keyed material codebook (Proposal 5).
    # Cluster points in (intensity, normal) feature space — NOT position
    # — so points with similar LiDAR return AND similar normal direction
    # share materials. material[i] = cluster_centers[idx[i]] + residual[i],
    # with residuals L2-regularized to be small.
    #
    # The clustering is in (intensity, normal) only: position is excluded
    # because spatially-adjacent points can be different materials
    # (wall meets floor), while points with similar intensity tend to
    # share physical reflectivity properties.
    # ------------------------------------------------------------
    cb_centers = None
    cb_residual = None
    cb_idx = None
    if intensity_clusters > 0:
        with torch.no_grad():
            n0_init_np = model.get_normals().detach().cpu().numpy()
            int_np = intensity_t.detach().cpu().numpy()
            # Standardize features so intensity and normal contribute
            # comparably to the k-means distance.
            int_std = (int_np - int_np.mean()) / (int_np.std() + 1e-6)
            feats_cb = np.concatenate(
                [int_std[:, None], n0_init_np], axis=1)  # (N, 4)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=intensity_clusters, n_init=5,
                        random_state=42, max_iter=100)
            labels = km.fit_predict(feats_cb).astype(np.int64)
            cb_idx = torch.from_numpy(labels).to(DEVICE)
            unique, counts = np.unique(labels, return_counts=True)
            if verbose:
                print(f"  INTENSITY CODEBOOK: K={intensity_clusters}, "
                      f"sizes min={counts.min()} max={counts.max()} "
                      f"mean={counts.mean():.0f}, residual λ={intensity_residual_lambda}")

        # Cluster centers initialized at ITU concrete (everyone starts the same)
        init_raw = inverse_reparameterize_torch(ITU_CONCRETE)
        init_raw_t = torch.from_numpy(init_raw).to(DEVICE)
        cb_centers = torch.nn.Parameter(
            init_raw_t.unsqueeze(0).expand(intensity_clusters, 6).clone())
        cb_residual = torch.nn.Parameter(
            torch.zeros(model.N, 6, device=DEVICE))
        # Freeze the per-point materials parameter (we override it).
        model.raw_materials.requires_grad_(False)

    # ------------------------------------------------------------
    # Normal anchor target — captured at init time (pcl normals).
    # Used by the anchor loss term to keep learned normals from
    # drifting independently per-point.
    # ------------------------------------------------------------
    if lambda_normal_anchor > 0:
        with torch.no_grad():
            normal_anchor_target = model.get_normals().detach().clone()
        if verbose:
            print(f"  NORMAL ANCHOR: λ={lambda_normal_anchor}")
    else:
        normal_anchor_target = None

    # ------------------------------------------------------------
    # Edge-preserving smoothness graph (Proposal 1).
    # Build a kNN graph on point positions; weight each edge by the
    # geometric AND material similarity of its two endpoints. The loss
    # term Σ w_ij ||material_i - material_j||² fires only on edges
    # where two points are likely on the same surface and have
    # similar LiDAR intensity (and therefore probably the same
    # physical material).
    # ------------------------------------------------------------
    edge_idx_a = None  # (E,) long
    edge_idx_b = None  # (E,) long
    edge_w = None      # (E,) float
    if edge_smooth:
        with torch.no_grad():
            pos_np = model.positions.detach().cpu().numpy()
            n0_np = model.get_normals().detach().cpu().numpy()
            int_np = intensity_t.detach().cpu().numpy()

            # kNN graph (excluding self)
            from scipy.spatial import cKDTree
            tree = cKDTree(pos_np)
            _, knn_idx = tree.query(pos_np, k=edge_smooth_k + 1)
            knn_idx = knn_idx[:, 1:]   # drop self → (N, k)

            N_pts = len(pos_np)
            i_arr = np.repeat(np.arange(N_pts), edge_smooth_k)
            j_arr = knn_idx.reshape(-1)
            # Edge features
            d_pos2 = ((pos_np[i_arr] - pos_np[j_arr]) ** 2).sum(axis=-1)
            d_n2 = ((n0_np[i_arr] - n0_np[j_arr]) ** 2).sum(axis=-1)
            d_i2 = (int_np[i_arr] - int_np[j_arr]) ** 2
            w = np.exp(
                -d_pos2 / (edge_smooth_sigma_p ** 2)
                -d_n2 / (edge_smooth_sigma_n ** 2)
                -d_i2 / (edge_smooth_sigma_i ** 2))
            # Drop edges where w is essentially zero (cleaner gradient
            # and lighter compute). Keep top half by weight.
            keep = w > 1e-3
            i_arr = i_arr[keep]
            j_arr = j_arr[keep]
            w = w[keep]

            edge_idx_a = torch.from_numpy(i_arr.astype(np.int64)).to(DEVICE)
            edge_idx_b = torch.from_numpy(j_arr.astype(np.int64)).to(DEVICE)
            edge_w = torch.from_numpy(w.astype(np.float32)).to(DEVICE)

        if verbose:
            print(f"  EDGE SMOOTH: k={edge_smooth_k}, kept {len(edge_w)}/{N_pts*edge_smooth_k} edges, "
                  f"mean w={edge_w.mean().item():.3f}, "
                  f"σ_p={edge_smooth_sigma_p} σ_n={edge_smooth_sigma_n} σ_i={edge_smooth_sigma_i}, "
                  f"λ={edge_smooth_lambda}")

    # Helper: compute the (N, 6) raw_materials override from the basis
    # parameterization (or None if per-point mode).
    def _compute_mat_override():
        if cb_centers is not None:
            return cb_centers[cb_idx] + cb_residual
        if mat_basis_params is not None:
            mb, mw = mat_basis_params
            return mb.unsqueeze(0) + mat_basis_features @ mw
        return None

    # Optimizer
    param_groups = []
    clip_vals = {}
    if cb_centers is not None:
        param_groups.append(
            {"params": [cb_centers], "lr": intensity_lr, "name": "cb_centers"})
        clip_vals["cb_centers"] = 1.0
        param_groups.append(
            {"params": [cb_residual], "lr": mat_lr, "name": "cb_residual"})
        clip_vals["cb_residual"] = 1.0
    elif mat_basis_params is not None:
        param_groups.append(
            {"params": list(mat_basis_params), "lr": mat_basis_lr, "name": "mat_basis"})
        clip_vals["mat_basis"] = 1.0
    elif LEARN_MATERIALS:
        param_groups.append(
            {"params": [model.raw_materials], "lr": mat_lr, "name": "materials"})
        clip_vals["materials"] = 1.0
    if LEARN_NORMALS:
        param_groups.append(
            {"params": [model.rotations], "lr": rot_lr, "name": "rotations"})
        clip_vals["rotations"] = 0.5
    base_lrs = {g["name"]: g["lr"] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    if verbose:
        print(f"  model N={model.N}, learnable groups={[g['name'] for g in param_groups]}")
        print(f"  per-frame active counts: ", end='')
        for f_idx in range(n_frames):
            print(f"{int(active_matrix[:, f_idx].sum().item())} ", end='')
        print()

    best_mean_train_corr = -1.0
    best_iter = 0
    best_state = None
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        per_frame_corr_train = []
        per_frame_loss_vals = []
        n_train_active = sum(
            1 for f_idx in train_indices
            if int(active_matrix[:, f_idx].sum().item()) > 0)
        if n_train_active == 0:
            raise RuntimeError(f"NVS: no training frames had any active points")
        loss_scale = 1.0 / n_train_active

        # Per-frame backward to keep peak memory at one frame's autograd
        # graph, not 5 frames stacked. Adam state accumulates the gradient
        # contributions from each frame; we step once after the loop.
        for f_idx in train_indices:
            f = per_frame[f_idx]
            active_mask_f = active_matrix[:, f_idx]
            if int(active_mask_f.sum().item()) == 0:
                continue

            # Compute material override from basis params (recomputed per
            # frame so each backward gets fresh autograd nodes — basis
            # params accumulate gradient across frames the same way the
            # per-point materials do).
            mat_override = _compute_mat_override()

            rp_real, rp_imag, ra_cart_gpu, cc = _render_and_metric(
                model, rasts[f_idx], vertex_areas, active_mask_f,
                f['sample_grid'], f['gt_cart_gpu_norm'], bsdf_mode=bsdf_mode,
                raw_materials_override=mat_override,
                disabled_components=disabled_components)
            per_frame_corr_train.append(cc)

            loss_f, _ = compute_ra_loss_rp(
                rp_real, rp_imag, f['gt_loss_cached'], loss_type=loss_type)

            # Normal anchor — penalize drift of learned normals from the
            # pcl init in MSE space (resolved unit-vector form so the
            # quaternion sign ambiguity is automatically handled).
            if normal_anchor_target is not None:
                cur_normals = model.get_normals()
                anchor_loss = ((cur_normals - normal_anchor_target) ** 2).mean()
                loss_f = loss_f + lambda_normal_anchor * anchor_loss

            # Render-consistency reg: render at a slightly perturbed pose
            # and penalize the squared deviation from the un-perturbed
            # render. This forces the model's BSDF to be locally smooth
            # in pose space — exactly what NVS interpolation needs.
            if render_consist:
                with _PerturbedRast(rasts[f_idx],
                                    eps_pos=render_consist_eps_pos,
                                    eps_rot=render_consist_eps_rot):
                    rp_r2, rp_i2 = render_gaussians(
                        model, rasts[f_idx], vertex_areas=vertex_areas,
                        active_mask=active_mask_f, shadow_mask=None,
                        bsdf_mode=bsdf_mode,
                        raw_materials_override=mat_override)
                # Scale-normalized squared difference (matches mse_raw scaling)
                with torch.no_grad():
                    scale = (rp_real.detach().abs().mean()
                             + rp_imag.detach().abs().mean()).clamp(min=1e-8)
                consist_loss = (
                    ((rp_real - rp_r2) ** 2).mean()
                    + ((rp_imag - rp_i2) ** 2).mean()
                ) / (scale ** 2)
                loss_f = loss_f + render_consist_lambda * consist_loss
                del rp_r2, rp_i2

            # Intensity-codebook residual L2 penalty (Proposal 5).
            # Encourages each point's material to stay close to its
            # cluster centroid. Applied per-frame so the gradient
            # accumulates over the loop the same way as the data loss.
            if cb_residual is not None:
                resid_loss = (cb_residual ** 2).mean()
                loss_f = loss_f + intensity_residual_lambda * resid_loss

            (loss_f * loss_scale).backward()
            per_frame_loss_vals.append(loss_f.item())

            del rp_real, rp_imag, ra_cart_gpu, loss_f
            if mat_override is not None:
                del mat_override
            torch.cuda.empty_cache()

        # ----------------------------------------------------------
        # Edge-preserving spatial smoothness on materials (Proposal 1).
        # Σ w_ij ||material_i - material_j||² over the precomputed
        # kNN graph. Edges are weighted by geometric + intensity
        # similarity so the loss only fires inside surfaces with
        # consistent LiDAR return — not across material boundaries.
        # One scalar loss per iter, single backward, cheap.
        # ----------------------------------------------------------
        if edge_smooth and edge_idx_a is not None:
            mat_for_smooth = _compute_mat_override()
            if mat_for_smooth is None:
                mat_for_smooth = model.raw_materials
            mat_a = mat_for_smooth[edge_idx_a]   # (E, 6)
            mat_b = mat_for_smooth[edge_idx_b]
            sq_diff = ((mat_a - mat_b) ** 2).sum(dim=-1)   # (E,) per-edge
            smooth_loss = (edge_w * sq_diff).sum() / edge_w.sum().clamp(min=1e-8)
            (smooth_loss * edge_smooth_lambda).backward()
            del mat_for_smooth, mat_a, mat_b, sq_diff, smooth_loss

        # ----------------------------------------------------------
        # Bracket-anchor regularizer: for each TEST frame t, find the
        # bracketing train frames (a, b) with a < t < b in time order,
        # render at the EXACT test pose (using rasts[t]), and penalize
        # deviation from 0.5 * (render_a + render_b). The test pose is
        # a known geometric fact (no GT leakage); only the test radar
        # measurement is forbidden.
        # ----------------------------------------------------------
        if bracket_anchor and len(test_indices) > 0 and len(train_indices) >= 2:
            sorted_train = sorted(i for i in train_indices if 0 <= i < n_frames)
            for t in test_indices:
                if not (0 <= t < n_frames):
                    continue
                # Find immediate brackets in the sorted train list
                a_idx = max((i for i in sorted_train if i < t), default=None)
                b_idx = min((i for i in sorted_train if i > t), default=None)
                if a_idx is None or b_idx is None:
                    continue   # not bracketed (extrapolation), skip
                am_t = active_matrix[:, t]
                if int(am_t.sum().item()) == 0:
                    continue

                # Per-frame bracket quality weighting by pose distance.
                # Wide brackets (large max pose dist) → low weight; tight
                # brackets (small pose dist) → full weight. Downweights
                # unreliable targets automatically.
                if bracket_pose_weighting:
                    with torch.no_grad():
                        pose_t = rasts[t].tx_positions.mean(dim=0)
                        pose_a = rasts[a_idx].tx_positions.mean(dim=0)
                        pose_b = rasts[b_idx].tx_positions.mean(dim=0)
                        d_a = (pose_t - pose_a).norm().item()
                        d_b = (pose_t - pose_b).norm().item()
                        pose_mult = float(np.exp(-bracket_pose_alpha * max(d_a, d_b)))
                else:
                    pose_mult = 1.0

                mat_override_anc = _compute_mat_override()

                # Render at the test pose with grad through model
                rp_rt, rp_it = render_gaussians(
                    model, rasts[t], vertex_areas=vertex_areas,
                    active_mask=am_t, shadow_mask=None, bsdf_mode=bsdf_mode,
                    raw_materials_override=mat_override_anc)

                w = (t - a_idx) / max(b_idx - a_idx, 1)

                if bracket_target == 'gt':
                    # Use GT |RA| at the bracketing TRAIN frames as the
                    # supervision target. This is allowed: GT at train
                    # frames is already used in the per-frame loss; we
                    # are NOT using any test-frame GT. The render at the
                    # test pose is supervised against an interpolation
                    # of the train GTs.
                    ra_t = range_profile_to_ra_mag(rp_rt, rp_it)  # (n_az, n_range)
                    target = (1 - w) * per_frame[a_idx]['gt_ra_raw'] + w * per_frame[b_idx]['gt_ra_raw']
                    with torch.no_grad():
                        var = target.detach().var().clamp(min=1e-8)
                    bracket_loss = ((ra_t - target.detach()) ** 2).mean() / var
                else:
                    am_a = active_matrix[:, a_idx]
                    am_b = active_matrix[:, b_idx]
                    if int(am_a.sum().item()) == 0 or int(am_b.sum().item()) == 0:
                        del rp_rt, rp_it
                        continue
                    if bracket_detach:
                        # Target is fixed — render brackets without grad so
                        # the optimizer can only adjust the test-pose render
                        # to match, not modify the brackets themselves.
                        with torch.no_grad():
                            rp_ra, rp_ia = render_gaussians(
                                model, rasts[a_idx], vertex_areas=vertex_areas,
                                active_mask=am_a, shadow_mask=None, bsdf_mode=bsdf_mode,
                                raw_materials_override=mat_override_anc)
                            rp_rb, rp_ib = render_gaussians(
                                model, rasts[b_idx], vertex_areas=vertex_areas,
                                active_mask=am_b, shadow_mask=None, bsdf_mode=bsdf_mode,
                                raw_materials_override=mat_override_anc)
                            target_re = (1 - w) * rp_ra + w * rp_rb
                            target_im = (1 - w) * rp_ia + w * rp_ib
                    else:
                        rp_ra, rp_ia = render_gaussians(
                            model, rasts[a_idx], vertex_areas=vertex_areas,
                            active_mask=am_a, shadow_mask=None, bsdf_mode=bsdf_mode,
                            raw_materials_override=mat_override_anc)
                        rp_rb, rp_ib = render_gaussians(
                            model, rasts[b_idx], vertex_areas=vertex_areas,
                            active_mask=am_b, shadow_mask=None, bsdf_mode=bsdf_mode,
                            raw_materials_override=mat_override_anc)
                        target_re = (1 - w) * rp_ra + w * rp_rb
                        target_im = (1 - w) * rp_ia + w * rp_ib
                    with torch.no_grad():
                        var = (target_re.detach().var() + target_im.detach().var()).clamp(min=1e-8)
                    bracket_loss = (
                        ((rp_rt - target_re) ** 2).mean()
                        + ((rp_it - target_im) ** 2).mean()
                    ) / var
                    del rp_ra, rp_ia, rp_rb, rp_ib, target_re, target_im

                (bracket_loss * loss_scale * bracket_anchor_lambda * pose_mult).backward()

                del rp_rt, rp_it, bracket_loss
                if mat_override_anc is not None:
                    del mat_override_anc
                torch.cuda.empty_cache()

        # ----------------------------------------------------------
        # Pose-midpoint linearity regularizer: for each pair of adjacent
        # train frames (a, b), render at the linear-interpolated midpoint
        # pose and penalize the deviation from 0.5*(render_a + render_b).
        # This forces the model's pose response to be locally linear
        # between adjacent train poses, killing the sharp dips the
        # optimizer would otherwise carve out at the in-between (test)
        # poses. NOTE: this does NOT use any test-frame GT — only the
        # interpolation of train-pose configs and renders.
        # ----------------------------------------------------------
        if pose_lerp and len(train_indices) >= 2:
            valid_train = [i for i in train_indices if 0 <= i < n_frames]
            pairs = list(zip(valid_train[:-1], valid_train[1:]))
            for a_idx, b_idx in pairs:
                am_a = active_matrix[:, a_idx]
                am_b = active_matrix[:, b_idx]
                if int(am_a.sum().item()) == 0 or int(am_b.sum().item()) == 0:
                    continue

                mat_override_anc = _compute_mat_override()

                # Endpoint renders (with grad through model parameters)
                rp_ra, rp_ia = render_gaussians(
                    model, rasts[a_idx], vertex_areas=vertex_areas,
                    active_mask=am_a, shadow_mask=None, bsdf_mode=bsdf_mode,
                    raw_materials_override=mat_override_anc)
                rp_rb, rp_ib = render_gaussians(
                    model, rasts[b_idx], vertex_areas=vertex_areas,
                    active_mask=am_b, shadow_mask=None, bsdf_mode=bsdf_mode,
                    raw_materials_override=mat_override_anc)
                # Midpoint render — same active-set as a (to keep things
                # comparable). Mutate rast_a in-place to the lerp pose.
                with _LerpRast(rasts[a_idx], rasts[b_idx], alpha=0.5):
                    rp_rm, rp_im = render_gaussians(
                        model, rasts[a_idx], vertex_areas=vertex_areas,
                        active_mask=am_a, shadow_mask=None, bsdf_mode=bsdf_mode,
                        raw_materials_override=mat_override_anc)
                # Pose-linearity penalty: midpoint render should be the
                # average of the two endpoint renders. This kills sharp
                # dips. Normalize by the variance of (r_a + r_b)/2 so
                # the loss is dimensionless.
                avg_re = 0.5 * (rp_ra + rp_rb)
                avg_im = 0.5 * (rp_ia + rp_ib)
                with torch.no_grad():
                    var = (avg_re.detach().var() + avg_im.detach().var()).clamp(min=1e-8)
                lerp_loss = (
                    ((rp_rm - avg_re) ** 2).mean()
                    + ((rp_im - avg_im) ** 2).mean()
                ) / var
                (lerp_loss * loss_scale * pose_lerp_lambda).backward()

                del rp_ra, rp_ia, rp_rb, rp_ib, rp_rm, rp_im
                del avg_re, avg_im, lerp_loss
                if mat_override_anc is not None:
                    del mat_override_anc
                torch.cuda.empty_cache()

        # ----------------------------------------------------------
        # Anchored-smoothness regularizer: for each pair of adjacent
        # train frames (a, b), penalize
        #   ||(render_b - render_a) - (GT_b - GT_a)||² / scale²
        # The model's pose-derivative is forced to match the data's
        # pose-derivative. Unlike naive render-consistency this does
        # NOT pull toward "uniform output" — the data difference is
        # the target, so the model must produce smoothly-different
        # renders matching the smoothly-different GTs.
        # ----------------------------------------------------------
        if consist_anchor and len(train_indices) >= 2:
            valid_train = [i for i in train_indices if 0 <= i < n_frames]
            pairs = list(zip(valid_train[:-1], valid_train[1:]))
            for a_idx, b_idx in pairs:
                am_a = active_matrix[:, a_idx]
                am_b = active_matrix[:, b_idx]
                if int(am_a.sum().item()) == 0 or int(am_b.sum().item()) == 0:
                    continue

                mat_override_anc = _compute_mat_override()

                rp_ra, rp_ia = render_gaussians(
                    model, rasts[a_idx], vertex_areas=vertex_areas,
                    active_mask=am_a, shadow_mask=None, bsdf_mode=bsdf_mode,
                    raw_materials_override=mat_override_anc)
                rp_rb, rp_ib = render_gaussians(
                    model, rasts[b_idx], vertex_areas=vertex_areas,
                    active_mask=am_b, shadow_mask=None, bsdf_mode=bsdf_mode,
                    raw_materials_override=mat_override_anc)
                ra_a = range_profile_to_ra_mag(rp_ra, rp_ia)
                ra_b = range_profile_to_ra_mag(rp_rb, rp_ib)
                gt_a = per_frame[a_idx]['gt_ra_raw']
                gt_b = per_frame[b_idx]['gt_ra_raw']
                delta_render = ra_b - ra_a
                delta_gt = (gt_b - gt_a).detach()
                with torch.no_grad():
                    scale = gt_a.mean().clamp(min=1e-8) ** 2
                consist_loss = ((delta_render - delta_gt) ** 2).mean() / scale
                (consist_loss * loss_scale * consist_anchor_lambda).backward()

                del rp_ra, rp_ia, rp_rb, rp_ib, ra_a, ra_b
                del delta_render, delta_gt, consist_loss
                if mat_override_anc is not None:
                    del mat_override_anc
                torch.cuda.empty_cache()

        total_loss_val = float(np.mean(per_frame_loss_vals))

        for group in optimizer.param_groups:
            clip = clip_vals.get(group["name"], 1.0)
            for p in group["params"]:
                rms_clip_grad(p, clip)

        lr_scale = get_lr_scale(
            it, total_iters=num_iters,
            warmup_iters=0, warmup_factor=1.0,
            decay_start=int(0.4 * num_iters))
        for group in optimizer.param_groups:
            group["lr"] = base_lrs[group["name"]] * lr_scale

        optimizer.step()

        # T3 post-step: overwrite each row with its cluster's centroid
        if cluster_idx_long is not None:
            with torch.no_grad():
                sum_per = torch.zeros(
                    material_clusters, 6, device=DEVICE,
                    dtype=model.raw_materials.dtype)
                sum_per.index_add_(0, cluster_idx_long, model.raw_materials)
                mean_per = sum_per / cluster_counts_t.unsqueeze(-1)
                model.raw_materials.copy_(mean_per[cluster_idx_long])

        mean_tc = float(np.mean(per_frame_corr_train))
        if mean_tc > best_mean_train_corr:
            best_mean_train_corr = mean_tc
            best_iter = it
            best_state = {k: v.data.clone() for k, v in model.state_dict().items()}
            if mat_basis_params is not None:
                best_state['__mat_base__'] = mat_basis_params[0].data.clone()
                best_state['__mat_W__'] = mat_basis_params[1].data.clone()
            if cb_centers is not None:
                best_state['__cb_centers__'] = cb_centers.data.clone()
                best_state['__cb_residual__'] = cb_residual.data.clone()

        # Per-iter test cc trajectory (diagnostic only — does NOT influence
        # best-state selection, which still uses train cc). Computed every
        # 25 iters on the test frames if any. Adds one render per test
        # frame per logged iter (~no_grad, cheap).
        test_cc_str = ""
        valid_test_idx = [i for i in test_indices if 0 <= i < n_frames]
        if (it % 25 == 0 or it == num_iters - 1) and len(valid_test_idx) > 0:
            with torch.no_grad():
                test_ccs = []
                eval_mat_override = _compute_mat_override()
                for f_idx in valid_test_idx:
                    f = per_frame[f_idx]
                    am = active_matrix[:, f_idx]
                    if int(am.sum().item()) == 0:
                        continue
                    rp_r, rp_i = render_gaussians(
                        model, rasts[f_idx], vertex_areas=vertex_areas,
                        active_mask=am, shadow_mask=None, bsdf_mode=bsdf_mode,
                        disabled_components=disabled_components,
                        raw_materials_override=eval_mat_override)
                    rap = range_profile_to_ra_mag(rp_r, rp_i)
                    rcg = polar_to_cart_torch(rap, f['sample_grid'])
                    test_ccs.append(cart_corr_torch(rcg, f['gt_cart_gpu_norm']).item())
                    del rp_r, rp_i, rap, rcg
                if test_ccs:
                    test_cc_str = f"  test={np.mean(test_ccs):.4f}"

        if verbose and (it % 25 == 0 or it == num_iters - 1):
            elapsed = time.time() - t0
            print(f"  iter {it:4d}: loss={total_loss_val:.6f}  "
                  f"mean_train_corr={mean_tc:.4f}{test_cc_str} "
                  f"(best_train={best_mean_train_corr:.4f}@{best_iter}) "
                  f"[{elapsed:.1f}s]")

    if verbose:
        print(f"\n  Best mean-train cart_corr: {best_mean_train_corr:.4f} at iter {best_iter}")

    # Restore best state
    if best_state is not None:
        with torch.no_grad():
            for k, v in best_state.items():
                if k.startswith('__'):
                    continue   # basis params handled below
                model.state_dict()[k].copy_(v)
            if mat_basis_params is not None and '__mat_base__' in best_state:
                mat_basis_params[0].data.copy_(best_state['__mat_base__'])
                mat_basis_params[1].data.copy_(best_state['__mat_W__'])
            if cb_centers is not None and '__cb_centers__' in best_state:
                cb_centers.data.copy_(best_state['__cb_centers__'])
                cb_residual.data.copy_(best_state['__cb_residual__'])

    # In basis mode, materialize the basis-derived materials into the
    # model's raw_materials slot so the eval path uses them.
    if mat_basis_params is not None or cb_centers is not None:
        with torch.no_grad():
            mat_final = _compute_mat_override()
            model.raw_materials.data.copy_(mat_final)

    # Final evaluation on all 9 frames at the best model state
    output_dir = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output_nvs', scene)
    os.makedirs(output_dir, exist_ok=True)

    results = evaluate_all_frames(
        model, rasts, active_matrix, vertex_areas, per_frame,
        train_indices, test_indices, output_dir, scene_data,
        bsdf_mode=bsdf_mode, disabled_components=disabled_components)

    results['best_iter'] = best_iter
    results['num_iters'] = num_iters
    results['n_gaussians'] = model.N
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n  Train mean cart_corr: {results['mean_train_cart_corr']:.4f}")
        print(f"  Test  mean cart_corr: {results['mean_test_cart_corr']:.4f}")
        print(f"  Saved → {output_dir}")

    results['model_state_dict'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return results


def evaluate_all_frames(model, rasts, active_matrix, vertex_areas, per_frame,
                        train_indices, test_indices, output_dir, scene_data,
                        bsdf_mode='full', disabled_components=None):
    """Render every frame at the current model state, save per-frame PNGs.

    Saves under output_dir/frame_<F>/:
      gt_ra_dB.png, gt_ra_linear.png
      rendered_ra_dB.png, rendered_ra_linear.png
      ra_comparison.png
      metrics.json (cart_corr, frame index, set name)

    Returns a dict with per-frame cart_corr + train/test means.
    """
    from mmir.data.ra_utils import save_ra_cartesian_png
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_frames = len(rasts)
    train_set = set(train_indices)
    test_set = set(test_indices)

    per_frame_corr = []

    with torch.no_grad():
        for f_idx in range(n_frames):
            f = per_frame[f_idx]
            active_mask_f = active_matrix[:, f_idx]
            tag = ('train' if f_idx in train_set
                   else ('test' if f_idx in test_set else 'unused'))

            if int(active_mask_f.sum().item()) == 0:
                per_frame_corr.append({
                    'f_idx': f_idx, 'frame': f['frame'], 'tag': tag,
                    'cart_corr': float('nan'),
                })
                continue

            rp_real, rp_imag = render_gaussians(
                model, rasts[f_idx], vertex_areas=vertex_areas,
                active_mask=active_mask_f, shadow_mask=None,
                bsdf_mode=bsdf_mode,
                disabled_components=disabled_components)
            ra_polar_t = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart_gpu = polar_to_cart_torch(ra_polar_t, f['sample_grid'])
            cc = cart_corr_torch(ra_cart_gpu, f['gt_cart_gpu_norm']).item()
            ra_rend_cart = ra_cart_gpu.cpu().numpy()
            ra_gt_cart = f['ra_gt_cart_cached']

            per_frame_corr.append({
                'f_idx': f_idx, 'frame': f['frame'], 'tag': tag,
                'cart_corr': cc,
            })

            # PNGs
            frame_dir = os.path.join(output_dir, f"frame_{f['frame']:03d}_{tag}")
            os.makedirs(frame_dir, exist_ok=True)
            np.save(os.path.join(frame_dir, 'ra_rendered_cart.npy'), ra_rend_cart)
            np.save(os.path.join(frame_dir, 'ra_gt_cart.npy'), ra_gt_cart)
            for scale in ('dB', 'linear'):
                save_ra_cartesian_png(
                    ra_gt_cart,
                    os.path.join(frame_dir, f'gt_ra_{scale}.png'),
                    range_res=f['range_res'], scale=scale,
                    title=f"GT f={f['frame']} ({tag}, {scale})")
                save_ra_cartesian_png(
                    ra_rend_cart,
                    os.path.join(frame_dir, f'rendered_ra_{scale}.png'),
                    range_res=f['range_res'], scale=scale,
                    title=f"Rendered f={f['frame']} ({tag}, cc={cc:.3f}, {scale})")

            r_db = 10 * np.log10(np.maximum(ra_rend_cart, 1e-10))
            g_db = 10 * np.log10(np.maximum(ra_gt_cart, 1e-10))
            db_max = max(r_db.max(), g_db.max())
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(g_db.T, origin='lower', aspect='auto', cmap='viridis',
                           vmin=db_max - 40, vmax=db_max)
            axes[0].set_title('GT (dB)')
            axes[1].imshow(r_db.T, origin='lower', aspect='auto', cmap='viridis',
                           vmin=db_max - 40, vmax=db_max)
            axes[1].set_title('Rendered (dB)')
            mn = ra_rend_cart.min(); mx = ra_rend_cart.max()
            r_norm = (ra_rend_cart - mn) / max(mx - mn, 1e-30)
            mn = ra_gt_cart.min(); mx = ra_gt_cart.max()
            g_norm = (ra_gt_cart - mn) / max(mx - mn, 1e-30)
            axes[2].imshow((r_norm - g_norm).T, origin='lower', aspect='auto',
                           cmap='RdBu_r', vmin=-0.3, vmax=0.3)
            axes[2].set_title('Error')
            plt.suptitle(f"{scene_data['scene']} frame {f['frame']} ({tag}) "
                         f"cart_corr={cc:.4f}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(frame_dir, 'ra_comparison.png'),
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

            with open(os.path.join(frame_dir, 'metrics.json'), 'w') as fp:
                json.dump({'frame': f['frame'], 'tag': tag,
                           'cart_corr': cc, 'f_idx': f_idx}, fp, indent=2)

            del rp_real, rp_imag, ra_polar_t, ra_cart_gpu

    train_corrs = [r['cart_corr'] for r in per_frame_corr
                   if r['tag'] == 'train' and not np.isnan(r['cart_corr'])]
    test_corrs = [r['cart_corr'] for r in per_frame_corr
                  if r['tag'] == 'test' and not np.isnan(r['cart_corr'])]

    return {
        'per_frame': per_frame_corr,
        'mean_train_cart_corr': float(np.mean(train_corrs)) if train_corrs else float('nan'),
        'mean_test_cart_corr': float(np.mean(test_corrs)) if test_corrs else float('nan'),
        'train_indices': list(train_indices),
        'test_indices': list(test_indices),
    }


def run_all_scenes_nvs(num_iters=300, target_n=90000,
                       train_indices=(0, 2, 4, 6, 8),
                       test_indices=(1, 3, 5, 7)):
    """Run NVS training over all benchmark scenes."""
    results = {}
    for scene in SCENES:
        try:
            r = train_nvs(scene,
                          train_indices=train_indices,
                          test_indices=test_indices,
                          num_iters=num_iters, target_n=target_n)
            results[scene] = r
        except Exception as e:
            print(f"[{scene}] FAILED: {e}")
            results[scene] = {'error': str(e)}

    print(f"\n{'='*70}")
    print(f"v4 NVS Results")
    print(f"{'='*70}")
    print(f"{'Scene':<25} {'train_cc':>10} {'test_cc':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")
    train_all, test_all = [], []
    for scene in SCENES:
        r = results.get(scene, {})
        if 'error' in r:
            print(f"{scene:<25} {'ERROR':>10} {'ERROR':>10}")
            continue
        tr = r.get('mean_train_cart_corr', float('nan'))
        te = r.get('mean_test_cart_corr', float('nan'))
        train_all.append(tr)
        test_all.append(te)
        print(f"{scene:<25} {tr:>10.4f} {te:>10.4f}")
    print(f"{'-'*25} {'-'*10} {'-'*10}")
    if train_all:
        print(f"{'Mean':<25} {np.mean(train_all):>10.4f} {np.mean(test_all):>10.4f}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--target_n', type=int, default=90000,
                        help='Total points after multi-frame FPS')
    parser.add_argument('--train_indices', type=str, default='0,2,4,6,8',
                        help='Comma-separated frame indices to train on')
    parser.add_argument('--test_indices', type=str, default='1,3,5,7',
                        help='Comma-separated frame indices to test on')
    parser.add_argument('--loss', type=str, default='pearson',
                        choices=['mse', 'mse_raw', 'pearson'])
    parser.add_argument('--clusters', type=int, default=0,
                        help='Material k-means clusters (0=per-point)')
    args = parser.parse_args()

    train_idx = tuple(int(x) for x in args.train_indices.split(',') if x.strip())
    test_idx = tuple(int(x) for x in args.test_indices.split(',') if x.strip())

    if args.all:
        run_all_scenes_nvs(num_iters=args.iters, target_n=args.target_n,
                           train_indices=train_idx, test_indices=test_idx)
    elif args.scene:
        train_nvs(args.scene, train_indices=train_idx, test_indices=test_idx,
                  num_iters=args.iters, target_n=args.target_n,
                  loss_type=args.loss, material_clusters=args.clusters)
    else:
        print("Usage: --scene <name> or --all")
