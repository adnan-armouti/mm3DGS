# Gaussian Rasterizer: Implementation Plan

Transforms mm25DGS into a pure PyTorch rasterizer. Directory:
`/home/adnan/Desktop/mm3DGS/mm25DGS/`.

---

## What Changes

| File | Status | Change |
|------|--------|--------|
| `bsdf_torch.py` | **REWRITE** | Native PyTorch Fresnel (field amplitude convention) |
| `antenna_torch.py` | **REWRITE** | PyTorch tensor pattern with differentiable interp |
| `adc_synthesis.py` | **MODIFY** | Native shading, √(power) amplitude, add γ (separate step) |
| `gaussian_model.py` | **MODIFY** | Add σ_em parameter |
| `config.py` | **MODIFY** | Shading tier config, σ_em LR/clip |
| `multibounce.py` | **REWRITE** | Pair-based (spatial hash via `torch_cluster` or voxel hash) |
| `ray_surfel.py` | **DELETE** | Replaced by pair enumeration |
| `initialization.py` | **MINOR** | σ_em initialised in GaussianModel |
| `optimizer.py` | **MODIFY** | Add σ_em parameter group |
| `training.py` | **MODIFY** | Remove DrJit init calls |
| `reparameterization.py` | **KEEP** | Same ITU raw ↔ physics transforms |
| `culling.py` | **KEEP** | Same contribution-based culling |
| `losses.py` | **KEEP** | Same FFT + RA loss |
| `density_control.py` | **KEEP** | Same clone/split/prune |
| `rae_grid.py` | **KEEP** | Diagnostics / visualization |
| `eval_adapter.py` | **MODIFY** | Remove DrJit dependency |

---

## File Specifications

### 1. `bsdf_torch.py` — REWRITE

```python
"""Native PyTorch BSDF for mmWave. No DrJit, full autograd.

Returns POWER-domain values. The amplitude chain takes sqrt() to convert
to field amplitude, matching mmIR's convention:
  weight = radar_scale × √(f_cos × G × path_loss)
See mmir/renderer/integrator/synthesis_e2e.py lines 498-503.

Tier 1: Fresnel + Rayleigh + cosine (4 of 6 ITU params)
  Captures: specular reflection, material absorption, surface coherence
  Misses: KA/SPM lobe shaping (l_c, tau) — for rough surfaces, upgrade to Tier 2
"""

import torch
from torch import Tensor
import math

C = 299_792_458.0
KAPPA_77GHZ = 2 * math.pi * 77e9 / C


def fresnel_power_reflectance(
    cos_theta_i: Tensor,    # (*,) clamped to [eps, 1]
    eps_real: Tensor,        # (*,)
    eps_imag: Tensor,        # (*,)
) -> Tensor:
    """Complex Fresnel power reflectance R = 0.5(|r_s|² + |r_p|²).

    Returns POWER reflectance in [0, 1]. Caller must sqrt() for field amplitude.
    """
    eps = torch.complex(eps_real, -eps_imag)
    n = torch.sqrt(eps)

    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_i_sq = 1.0 - cos_i ** 2
    # Snell's law for complex n: cos_t = sqrt(1 - sin²θ_i / ε)
    cos_t = torch.sqrt((1.0 - sin_i_sq / eps).to(torch.complex64))
    cos_i_c = cos_i.to(torch.complex64)

    r_s = (cos_i_c - n * cos_t) / (cos_i_c + n * cos_t + 1e-10)
    r_p = (n * cos_i_c - cos_t) / (n * cos_i_c + cos_t + 1e-10)

    R = 0.5 * (torch.abs(r_s) ** 2 + torch.abs(r_p) ** 2)
    return R.real.clamp(0.0, 1.0)


def rayleigh_factor(
    cos_theta_i: Tensor,
    sigma_h: Tensor,
    kappa: float = KAPPA_77GHZ,
) -> Tensor:
    """η = exp(-(2κσ_h cosθ)²). Coherent fraction."""
    return torch.exp(-(2.0 * kappa * sigma_h * cos_theta_i) ** 2)


def evaluate_bsdf_tier1(
    cos_theta_i: Tensor,
    eps_real: Tensor,
    eps_imag: Tensor,
    sigma_h: Tensor,
    thickness: Tensor,
) -> Tensor:
    """Tier 1: Fresnel × Rayleigh × cosθ. Returns POWER (not field amp).

    Uses 4 of 6 ITU params. l_c and tau are unused (they control the
    KA/SPM lobe shape, which requires the full Tier 2 model).
    """
    R = fresnel_power_reflectance(cos_theta_i, eps_real, eps_imag)
    eta = rayleigh_factor(cos_theta_i, sigma_h)
    return R * eta * cos_theta_i
```

### 2. `antenna_torch.py` — REWRITE

```python
"""Native PyTorch antenna pattern. No DrJit.

Loads .npy pattern files (same format as mmIR), stores as PyTorch tensors.
Evaluates via differentiable linear interpolation with periodic wrapping.
"""

import torch
from torch import Tensor
import numpy as np
import math


class AntennaPattern:
    """Antenna pattern stored as a PyTorch tensor."""

    def __init__(self, pattern_path: str, device: str = "cuda:0"):
        data = np.load(pattern_path)    # (361, 2) columns [E_dB, H_dB]
        E_lin = np.power(10.0, data[:, 0] / 10.0)
        H_lin = np.power(10.0, data[:, 1] / 10.0)

        # Scaling factor C (same formula as mmIR's AntennaPatternLoader)
        G_max = max(E_lin.max(), H_lin.max())
        P_sep = E_lin.max() * H_lin.max()
        C_scale = G_max / P_sep if P_sep > 0 else 1.0

        self.E = torch.from_numpy((E_lin * C_scale).astype(np.float32)).to(device)
        self.H = torch.from_numpy((H_lin * C_scale).astype(np.float32)).to(device)
        self.device = device

    def evaluate(self, directions: Tensor, orientations: Tensor) -> Tensor:
        """(N,) linear power gain. directions and orientations are (N,3) unit vecs."""
        # Build local frame via Gram-Schmidt from boresight
        fwd = orientations
        world_up = torch.tensor([0., 0., 1.], device=self.device).expand_as(fwd)
        right = torch.cross(fwd, world_up, dim=-1)
        right_norm = right.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        right = right / right_norm
        up_local = torch.cross(right, fwd, dim=-1)

        # Project direction to local frame
        d_fwd = (directions * fwd).sum(-1)
        d_right = (directions * right).sum(-1)
        d_up = (directions * up_local).sum(-1)

        # E-plane (elevation): angle in fwd-up plane
        angle_E_deg = torch.rad2deg(torch.atan2(d_up, d_fwd)) % 360.0
        # H-plane (azimuth): angle in fwd-right plane
        angle_H_deg = torch.rad2deg(torch.atan2(d_right, d_fwd)) % 360.0

        gain_E = self._interp(angle_E_deg, self.E)
        gain_H = self._interp(angle_H_deg, self.H)
        return (gain_E * gain_H).clamp(min=0.0)

    def _interp(self, deg: Tensor, pattern: Tensor) -> Tensor:
        """Differentiable linear interp with periodic wrap on [0, 360]."""
        idx_lo = deg.long() % 360
        idx_hi = (idx_lo + 1) % 360
        frac = deg - deg.floor()
        return pattern[idx_lo] * (1.0 - frac) + pattern[idx_hi] * frac


# Singletons
_tx_pattern: AntennaPattern = None
_rx_pattern: AntennaPattern = None

def load_patterns(tx_path: str, rx_path: str, device: str = "cuda:0"):
    global _tx_pattern, _rx_pattern
    _tx_pattern = AntennaPattern(tx_path, device)
    _rx_pattern = AntennaPattern(rx_path, device)

def evaluate_tx_gain(directions: Tensor, orientations: Tensor) -> Tensor:
    return _tx_pattern.evaluate(directions, orientations)

def evaluate_rx_gain(directions: Tensor, orientations: Tensor) -> Tensor:
    return _rx_pattern.evaluate(-directions, orientations)
```

### 3. `gaussian_model.py` — ADD σ_em

```python
# In __init__, add after raw_materials:
self.log_sigma_em = nn.Parameter(
    torch.full((N, 1), math.log(0.62e-3), device=device)
)   # λ/(2π) at 77 GHz

# New method:
def get_sigma_em(self) -> Tensor:
    """(N, 1) EM coherence scale in metres. Clamped to [0.01mm, 5mm]."""
    return torch.exp(self.log_sigma_em.clamp(-11.5, -5.3))

# Update save() and load() to include log_sigma_em.
```

### 4. `adc_synthesis.py` — MODIFY

Key changes (in `synthesize_adc_single_bounce`):

```python
# === REPLACE DrJit BSDF wrapper ===
# OLD:
#   from .bsdf_torch import eval_bsdf_physics_batch  # DrJit
#   bsdf_val = eval_bsdf_physics_batch(wo, wi, n, ...)
# NEW:
from .bsdf_torch import evaluate_bsdf_tier1
from .reparameterization import reparameterize

physics_mat = reparameterize(raw_mat)   # (M, 6), autograd-connected!

# Incidence cosine for each (Gaussian, TX, RX)
# cos_theta[g, i, j] = |dot(n_g, normalize(d_ij))|  where d_ij = (r̂_tx_i + r̂_rx_j)/2
cos_theta = _compute_incidence_cosine(normals, d_tx_dirs, d_rx_dirs)  # (M, N_tx, N_rx)

# BSDF returns POWER  (broadcastable: material (M,1,1), angle (M,Ntx,Nrx))
f_cos_power = evaluate_bsdf_tier1(
    cos_theta,
    physics_mat[:, 0:1].unsqueeze(-1),   # eps_real (M,1,1)
    physics_mat[:, 1:2].unsqueeze(-1),   # eps_imag
    physics_mat[:, 2:3].unsqueeze(-1),   # sigma_h
    physics_mat[:, 5:6].unsqueeze(-1),   # thickness
)  # (M, N_tx, N_rx) — full autograd to material params

# === REPLACE DrJit antenna wrapper ===
# Same function names, but antenna_torch.py is now pure PyTorch
G_tx = evaluate_tx_gain(dir_tx_flat, bore_tx_flat).reshape(M, N_tx)
G_rx = evaluate_rx_gain(dir_rx_flat, bore_rx_flat).reshape(M, N_rx)

# === ADD coherence factor γ ===
sigma_em = model.get_sigma_em()[active_mask]    # (M, 1)
d_perp_sq = _compute_bistatic_tangent_proj(normals, t1, t2, d_tx_dirs, d_rx_dirs)
gamma = torch.exp(-0.5 * KAPPA_77GHZ**2 * sigma_em**2 * d_perp_sq)  # (M, N_tx, N_rx)

# === AMPLITUDE: √(power product) — field amplitude convention ===
power_product = (
    f_cos_power * gamma
    * G_tx.unsqueeze(-1) * G_rx.unsqueeze(-2)
    / (d_tx.unsqueeze(-1)**2 * d_rx.unsqueeze(-2)**2 + 1e-20)
)
A = opacities.unsqueeze(-1).unsqueeze(-1) * torch.sqrt(power_product.clamp(min=1e-20))

# ... phase and phasor scatter unchanged ...
```

Helper functions needed (small, ~5 lines each):

```python
def _compute_incidence_cosine(normals, d_tx_dirs, d_rx_dirs):
    """Cosine of incidence angle for each (Gaussian, TX, RX).
    Uses the bisector of TX and RX directions as the effective incidence."""
    # d_tx_dirs: (M, N_tx, 3), d_rx_dirs: (M, N_rx, 3), normals: (M, 3)
    # Bisector: (r̂_tx + r̂_rx) / |r̂_tx + r̂_rx|
    bisector = d_tx_dirs.unsqueeze(2) + d_rx_dirs.unsqueeze(1)  # (M, Ntx, Nrx, 3)
    bisector = F.normalize(bisector, dim=-1)
    cos_theta = torch.abs((normals.unsqueeze(1).unsqueeze(2) * bisector).sum(-1))
    return cos_theta  # (M, N_tx, N_rx)

def _compute_bistatic_tangent_proj(normals, t1, t2, d_tx_dirs, d_rx_dirs):
    """||d_ij,⊥||² = (t1·d_ij)² + (t2·d_ij)² for coherence factor γ."""
    d_ij = d_tx_dirs.unsqueeze(2) + d_rx_dirs.unsqueeze(1)  # (M, Ntx, Nrx, 3)
    proj_t1 = (t1.unsqueeze(1).unsqueeze(2) * d_ij).sum(-1)  # (M, Ntx, Nrx)
    proj_t2 = (t2.unsqueeze(1).unsqueeze(2) * d_ij).sum(-1)
    return proj_t1**2 + proj_t2**2
```

### 5. `multibounce.py` — REWRITE

```python
"""Multi-bounce via Gaussian pair interaction.

Replaces ray-surfel intersection with spatial proximity queries.
Uses torch_cluster.radius for the spatial index (pip install torch_cluster),
or falls back to chunked pairwise distance.
"""

import torch
from torch import Tensor
from .gaussian_model import GaussianModel
from .config import RadarConfig, C
from .bsdf_torch import evaluate_bsdf_tier1

try:
    from torch_cluster import radius
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False


def find_neighbor_pairs(positions: Tensor, r_max: float) -> Tensor:
    """Find all pairs within interaction radius.

    Returns (P, 2) long tensor of pair indices (i < j to avoid duplicates).

    Uses torch_cluster.radius if available (O(N log N) with spatial hashing).
    Falls back to chunked pairwise distance (O(N² / chunk_size) memory).
    """
    if HAS_TORCH_CLUSTER:
        # radius() returns (target_idx, source_idx) for all pairs within r
        row, col = radius(positions, positions, r=r_max,
                          max_num_neighbors=256)
        # Remove self-pairs and duplicates (keep i < j)
        mask = row < col
        return torch.stack([row[mask], col[mask]], dim=1)
    else:
        return _find_pairs_bruteforce(positions, r_max)


def _find_pairs_bruteforce(positions: Tensor, r_max: float,
                            chunk_size: int = 8192) -> Tensor:
    """Chunked pairwise distance fallback. O(N²) compute, O(chunk×N) memory."""
    N = positions.shape[0]
    pairs_list = []
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        # Distance from chunk to ALL points
        dist = torch.cdist(positions[i_start:i_end], positions)  # (chunk, N)
        # Find pairs within radius (i < j)
        row_local, col = torch.where((dist < r_max) & (dist > 1e-6))
        row = row_local + i_start
        mask = row < col  # avoid duplicates
        if mask.any():
            pairs_list.append(torch.stack([row[mask], col[mask]], dim=1))
    if pairs_list:
        return torch.cat(pairs_list, dim=0)
    return torch.zeros(0, 2, dtype=torch.long, device=positions.device)


def prune_pairs(
    pairs: Tensor,
    positions: Tensor,
    normals: Tensor,
    opacities: Tensor,
    min_opacity: float = 0.01,
) -> Tensor:
    """Remove pairs that cannot produce meaningful two-bounce paths.

    Prunes:
    - Pairs where either Gaussian has very low opacity
    - Pairs where g1's normal faces away from g2 AND g2's faces away from g1
    """
    g1_idx, g2_idx = pairs[:, 0], pairs[:, 1]

    # Opacity check
    op_ok = (opacities[g1_idx] > min_opacity) & (opacities[g2_idx] > min_opacity)

    # Normal compatibility: at least one must face the other
    d12 = positions[g2_idx] - positions[g1_idx]
    d12_norm = torch.nn.functional.normalize(d12, dim=-1)
    n1_dot = (normals[g1_idx] * d12_norm).sum(-1)     # g1 faces g2?
    n2_dot = (normals[g2_idx] * (-d12_norm)).sum(-1)   # g2 faces g1?
    normal_ok = (n1_dot > 0) | (n2_dot > 0)  # at least one faces the other

    mask = op_ok.squeeze(-1) & normal_ok
    return pairs[mask]


def synthesize_multibounce(
    model: GaussianModel,
    radar_cfg: RadarConfig,
    r_max: float = 2.0,
    detach_phase: bool = True,
    chunk_size: int = 1024,
) -> tuple:
    """Two-bounce ADC via Gaussian pair interaction.

    Returns (adc_real, adc_imag) each (N_tx, N_rx, K).
    """
    # ... find pairs, prune, then chunked phasor accumulation
    # over pairs (same pattern as single-bounce but over P pairs
    # instead of N Gaussians). Each pair contributes one phasor
    # with total path length d(TX, g1) + d(g1, g2) + d(g2, RX).
    ...
```

### 6. `config.py` — MODIFY

Add to `TrainingConfig`:

```python
    # --- Shading ---
    shading_tier: int = 1
    # Tier 1: Fresnel + Rayleigh (4/6 ITU params, fast)
    # Tier 2: Full KA+SPM (all 6 params, future)
    # Tier 3: Fresnel + Legendre residual (6 + L+1 params)
    legendre_order: int = 0     # L for Tier 3 (0 = no residual)

    # --- Coherence ---
    enable_coherence_gamma: bool = True
    lr_sigma_em: float = 1e-3
    clip_sigma_em: float = 1.0

    # --- Multi-bounce ---
    multibounce_interaction_radius: float = 2.0
```

### 7. `optimizer.py` — MODIFY

```python
# Add to _build_groups:
{"name": "sigma_em", "params": [model.log_sigma_em],
 "lr": cfg.lr_sigma_em, "clip": cfg.clip_sigma_em},
```

### 8. `training.py` — MODIFY

```python
# REMOVE: any import of mitsuba, drjit, or mi.set_variant
# The only mmIR imports that remain:
from mmir.data.data_utils import load_target         # data loading
from mmir.data.ra_utils import adc_to_ra_complex     # FFT pipeline (pure PyTorch)

# load_patterns() now creates PyTorch tensors, not DrJit arrays.
# No mi.set_variant() call needed anywhere.
```

### 9. DELETE `ray_surfel.py`

Replaced by pair enumeration in `multibounce.py`.

---

## Implementation Order

| Step | Files | Verify | Notes |
|------|-------|--------|-------|
| **1** | `bsdf_torch.py` rewrite | Fresnel matches DrJit within 5% | Run both on concrete/glass/metal at 0°,30°,60° |
| **2** | `antenna_torch.py` rewrite | Gain matches DrJit within 1% | Compare at boresight and ±30° |
| **3** | `gaussian_model.py` + σ_em | save/load round-trip | Check σ_em init = 0.62mm |
| **4** | `adc_synthesis.py` modify (**without γ**) | Rendered ADC correlation >0.95 vs DrJit version | Same Gaussians, new shading, no γ yet |
| **5** | **Gradient check** | Finite-difference test on eps_real | Perturb by δ=1e-4, compare ∂L/∂ε_r × δ to ΔL |
| **6** | `config.py` + `optimizer.py` | Config parses, optimizer includes σ_em | |
| **7** | `training.py` modify | Full training runs without DrJit | 50 iters on seq_0_frame_135 |
| **8** | **Quality check** (no γ) | RA correlation within 0.02 of DrJit pipeline | This validates the Fresnel simplification |
| **9** | `adc_synthesis.py` + γ | RA correlation improves or stays same | γ is a refinement, shouldn't hurt |
| **10** | `multibounce.py` rewrite | Multi-bounce produces non-zero ADC | Pair count, pruning ratio |
| **11** | Delete `ray_surfel.py` | Nothing imports it | |
| **12** | Full benchmark | All 9 scenes, all metrics, compare to mmIR | |

**Key principle**: Steps 4 and 8 validate WITHOUT γ. This isolates the
Fresnel-vs-KA+SPM difference from the γ effect. γ is added separately
in step 9 and validated as an incremental improvement.

---

## Validation Criteria

### Step 1: BSDF correctness

Compare `fresnel_power_reflectance()` output to DrJit `BSDFmmWaveScalar`:

```python
# For each material in {concrete, glass, metal}:
#   For each angle in {0°, 30°, 60°}:
#     torch_R = fresnel_power_reflectance(cos_theta, eps_r, eps_i)
#     drjit_f = bsdf.eval_f_cos_physics(wo, wi, n, eps_r, eps_i, sigma_h, l_c, tau, thick)
#     # Tier 1 omits KA/SPM lobes, so match is approximate:
#     assert abs(torch_R * eta * cos - drjit_f) / drjit_f < 0.3  # 30% for Tier 1
#     # NOTE: Tier 1 will diverge from DrJit at oblique angles where
#     # the SPM incoherent lobe dominates. This is expected and acceptable.
```

### Step 5: Gradient correctness (finite difference)

```python
# Perturb eps_real for one Gaussian by δ = 1e-3
# Compute loss at (eps_real) and (eps_real + δ)
# Compare: |∂L/∂eps_real × δ - ΔL| / |ΔL| < 0.05  (5% relative error)
```

This catches:
- Sign errors in the gradient
- The power-vs-amplitude bug (would show as 2× error)
- Missing factors from reparameterization

### Step 8: Quality (no γ)

Train on seq_0_frame_135 for 500 iterations. Compare:
- RA Pearson correlation: within 0.02 of DrJit-based pipeline
- If significantly worse: the Fresnel simplification (missing KA/SPM lobes)
  is the likely cause → proceed to Tier 2 implementation

### Step 9: Quality (with γ)

Add γ and retrain. Compare:
- RA correlation: same or better than without γ
- If worse: γ initialization or σ_em learning rate needs tuning

---

## Phase Plan (PyTorch → CUDA)

Per the user's analysis, PyTorch is sufficient for Phase 1.

| Phase | Approach | Trigger |
|-------|----------|---------|
| **1** (now) | Pure PyTorch | Validate framework, get gradients working |
| **2** | Fused phasor kernel (.cu) | If phasor scatter memory > 50% of GPU |
| **3** | Spatial hash kernel (.cu) | If multi-bounce pair finding > 1s |
| **4** | Full custom rasterizer (.cu) | If targeting real-time or > 500K Gaussians |

`torch.utils.cpp_extension` makes wrapping .cu files straightforward.
The phasor scatter kernel (Phase 2) is the highest-value target:
it eliminates the (chunk, N_tx, N_rx, K) intermediate tensor.
