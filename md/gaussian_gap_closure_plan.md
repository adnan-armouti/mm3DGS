# mm25DGS Gaussian Gap Closure: Implementation Plan

## Current Status

### Correlation Summary (cart_corr vs GT RA)

| Scene | mmIR | Stage B (mesh) | C3 (Gaussian) | C3-mmIR gap | B-C3 gap |
|-------|------|----------------|---------------|-------------|----------|
| seq_0_frame_135 | 0.920 | 0.891 | 0.870 | 0.050 | 0.021 |
| seq_0_frame_390 | 0.937 | 0.962 | 0.826 | 0.110 | 0.136 |
| seq_1_frame_185 | 0.954 | 0.954 | 0.870 | 0.084 | 0.084 |
| seq_1_frame_438 | 0.941 | 0.953 | 0.849 | 0.092 | 0.104 |
| seq_2_frame_105 | 0.886 | 0.913 | 0.829 | 0.057 | 0.085 |
| seq_2_frame_160 | 0.863 | 0.862 | 0.775 | 0.088 | 0.087 |
| seq_2_frame_300 | 0.906 | 0.938 | 0.851 | 0.055 | 0.088 |
| **Mean** | **0.915** | **0.925** | **0.838** | **0.077** | **0.086** |

### Key Observation

Stage B mesh training matches or exceeds mmIR (mean 0.925 vs 0.915). The physics engine is correct. The entire 0.077 gap comes from the Gaussian representation itself.

---

## Root Cause Analysis

### 1. Scales are dead parameters — surfels are not truly Gaussians

Every Gaussian paper (3DGS, 3DGRT, EVER, VC-3DGS) has the same core property: scale parameters determine spatial extent and directly enter the rendering equation. In mm25DGS_v2, `log_scales` is optimized (lr=5e-3) but never used — the area passed to the rasterizer is a static `vertex_areas` from mesh init:

```python
# render_gaussians (line 396-397):
areas = vertex_areas * opacities   # vertex_areas is FIXED from init
```

Scales receive zero gradient and waste 2N parameters.

### 2. Intra-surfel phase cancellation is unmodeled

A surfel of lateral extent `s` spans a range of path lengths from TX to different points on the surfel to RX. The phase variation is `delta_phi ~ 2*pi*2s/lambda ~ 2*pi*2s/3.9mm`. For a typical surfel with s=5cm, this is ~160 radians — many wavelengths. Different parts of the surfel contribute phasors that partially cancel (destructive interference). Our point-scatterer model evaluates phase at the surfel center only, systematically over-estimating the contribution of large surfels.

This is the core physical limitation of our rendering equation: it treats each surfel as a coherent point scatterer, when in reality only surfels smaller than ~lambda/2 (~2mm) scatter coherently across their full extent. VC-3DGS's ray-space Gaussian evaluation (gamma, beta parameters) translates to a physics-based solution for this (see Stage F4).

### 3. No adaptive densification (from 3DGS)

3DGS's clone/split/prune is fundamental to quality. mm25DGS C5 has basic pruning only.

### 4. No progressive BSDF complexity (from 3DGS progressive SH)

All 6 BSDF parameters train simultaneously, allowing roughness to compensate for incorrect permittivity.

### 5. MAX_ACTIVE cap drops ~80% of surfels

Scenes have 50K-113K Gaussians, capped at 12K random. Stage B uses all ~24K hits.

### 6. Flat LR schedule, 500 iterations

All 4 papers use LR decay. Several scenes peak at iter 499.

---

## Opacity vs Density: What Makes Sense for Radar?

In optical Gaussian rendering:
- **3DGS/VC-3DGS opacity**: controls how much light a primitive *blocks* (transmittance). `alpha_i` determines front-to-back blending order.
- **EVER density**: volumetric attenuation `sigma` per unit length through the ellipsoid interior. Coupled to scale via `sigma = -log(1-0.99*alpha) / min(s)`.

In our radar renderer, **neither model applies directly**:
- There is no transmittance — phasor summation is order-independent. Every surfel contributes additively.
- There is no ray-interior volume — surfels are flat discs, not 3D blobs.

Our "opacity" is physically a **scattering efficiency** in [0, 1]: what fraction of the surfel's geometric area is a coherent scattering surface. A value of 0.5 means half the surfel area scatters effectively (the rest is sub-wavelength roughness, gaps, or mixed material below the surfel resolution).

**Decision: keep opacity as a [0,1] scattering efficiency (not volumetric density).** The natural coupling between scale and contribution comes from physics, not from an artificial regularizer: VC-3DGS's range-domain apodization (see Stage F4) shows that larger surfels naturally contribute less per unit area due to intra-surfel phase cancellation. This provides inherent gradient pressure against unbounded scale growth — no hyperparameter needed.

---

## Implementation Stages

Each stage is a single, isolated change. After each stage, all 7 scenes are run and results are recorded in `md/gap_closure_results.md`.

### Stage F1: Remove MAX_ACTIVE Cap

**Change:** Remove the 12K random subsampling cap. Render all visible surfels.

**Rationale:** This is the simplest change (delete code) with the largest expected impact. It addresses the coverage bottleneck without modifying the model or loss.

**What changes:**
```python
# train_gaussian.py: DELETE lines 496-504
# REMOVE:
MAX_ACTIVE = 12000
if n_active > MAX_ACTIVE:
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    perm = torch.randperm(n_active, device=DEVICE)[:MAX_ACTIVE]
    new_mask = torch.zeros_like(active_mask)
    new_mask[active_indices[perm]] = True
    active_mask = new_mask
    n_active = MAX_ACTIVE
```

Also remove the corresponding MAX_ACTIVE block inside the C5 density control section (lines 619-624).

**Files:** `train_gaussian.py`

**Fallback if OOM:** Replace random subsampling with importance-weighted top-k using `compute_analytical_weights`.

**Expected impact:** +0.02-0.04

**Run:** All 7 scenes, C3 mode, 500 iters. Record to results table.

---

### Stage F2: Cosine LR Decay + 1500 Iterations

**Change:** Replace the flat post-warmup LR with cosine decay. Increase training to 1500 iterations.

**Rationale:** All 4 comparison papers (3DGS, 3DGRT, EVER, VC-3DGS) use decaying LR schedules. Multiple scenes peak at iter 499, indicating incomplete convergence.

**What changes:**
```python
# train_mesh.py: modify get_lr_scale
def get_lr_scale(iteration, warmup_iters=5, warmup_factor=0.3,
                 total_iters=1500, decay_start=100, min_lr_factor=0.01):
    if iteration < warmup_iters:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup_iters, 1))
    if iteration < decay_start:
        return 1.0
    progress = (iteration - decay_start) / max(total_iters - decay_start, 1)
    return min_lr_factor + 0.5 * (1.0 - min_lr_factor) * (1 + math.cos(math.pi * progress))
```

```python
# train_gaussian.py: change default num_iters
def train_gaussians(scene, mode='c3', num_iters=1500, ...):
```

**Files:** `train_mesh.py`, `train_gaussian.py`

**Expected impact:** +0.01-0.02

**Run:** All 7 scenes, C3 mode, 1500 iters (includes F1). Record to results table.

---

### Stage F3: Live Surfel Areas from Scales

**Change:** Replace static `vertex_areas` with dynamically computed areas from learnable scales: `A_i = pi * s1_i * s2_i`.

**Rationale (from papers):** In every Gaussian paper, scale determines the primitive's spatial contribution and receives gradients through that path. Currently mm25DGS scales are dead parameters — optimized but never used.

**Source:** 3DGS (scale -> covariance -> 2D splat area), EVER (scale -> ellipsoid boundary), VC-3DGS (scale -> ray-space beta)

**What changes:**
```python
# render_gaussians — BEFORE:
areas = vertex_areas * opacities

# render_gaussians — AFTER:
scales = model.get_scales()                            # (N, 2) = exp(log_scales)
surfel_areas = math.pi * scales[:, 0] * scales[:, 1]  # elliptical disc area
areas = surfel_areas * opacities
```

Remove `vertex_areas` from the render path entirely. The `vertex_areas_t` return from `init_from_mesh` / `init_from_lidar` is kept only to initialize `log_scales` to match the initial vertex areas (so that F3 starts from the same effective areas as F1/F2).

Initialization adjustment — ensure `log_scales` are set so that `pi * s1 * s2 = vertex_area`:
```python
# init_from_mesh: set scales so pi*s1*s2 = vertex_area
# Currently: scales = sqrt(vertex_area), so pi*s1*s2 = pi*vertex_area (too large by pi)
# Fix: scales = sqrt(vertex_area / pi)
scale_val = np.sqrt(np.maximum(vertex_areas, 1e-12) / math.pi)
scales = np.column_stack([scale_val, scale_val])
```

**Files:** `train_gaussian.py` (`render_gaussians`, `init_from_mesh`, `init_from_lidar`)

**Expected impact:** +0.02-0.04 (scales now receive gradients and can adapt)

**Run:** All 7 scenes, C3 mode, 1500 iters (includes F1+F2). Record to results table. Additionally record: mean/std of scale values at init vs end of training to verify scales are evolving.

---

### Stage F4: Range-Domain Gaussian Apodization (from VC-3DGS)

**Change:** Add a physics-based attenuation factor that models intra-surfel phase cancellation. Each surfel's contribution is multiplied by a Gaussian window in range, whose width is determined by the surfel's spatial extent along the RX direction.

**Rationale (from papers):** VC-3DGS evaluates the 3D Gaussian kernel analytically along each ray, computing `gamma` (peak location) and `beta = 1/sqrt(d^T Sigma^-1 d)` (1D spread along the ray). Translated to radar (see `optics_to_radar_translation.md`), the integral of `Gaussian(t) * exp(j*phi(t))` over the surfel extent evaluates to a Gaussian-apodized phasor:

```
E_scat ~ W * sqrt(2*pi) * beta * exp(j*phi_center) * exp(-2*pi^2 * beta^2 * f_beat^2 / c^2)
```

The last exponential is the apodization: larger surfels (larger `beta`) produce weaker, broader range responses. This is physically correct — a scatterer spanning many wavelengths cannot scatter coherently across its full extent.

**Why this replaces the artificial contribution regularizer (old F4):**

The old F4 used `L_contrib = mean((area*opacity - init)^2)` — an artificial constraint with a tunable lambda. The apodization achieves the same goal (preventing scale divergence) through physics: the optimizer naturally avoids large surfels because they contribute less per unit area. No hyperparameter needed.

**Source:** VC-3DGS (ray-space beta evaluation), adapted for FMCW coherent radar

**What changes in `_render_chunk_torch`:**
```python
# After computing weight (line ~462) and before phase scatter:

# VC-3DGS range-domain apodization: surfel extent attenuates contribution
# beta = 1D extent of surfel along the RX direction
# For a 2D surfel with scales (s1, s2) and normal N, the extent along
# direction d_rx is: beta = sqrt(s1^2*(e1.d)^2 + s2^2*(e2.d)^2)
# where e1, e2 are the surfel's tangent directions.
#
# Simplified: for a circular surfel (s1~s2~s), beta ~ s*sin(theta_rx)
# where theta_rx is the angle between normal and RX direction.
# At normal incidence (theta_rx=0), beta=0 (no spread, full contribution).
# At grazing (theta_rx=90), beta=s (max spread, most cancellation).
#
# Apodization factor: exp(-2*pi^2 * beta^2 * S^2 * tau^2 / c^2)
# where S is FMCW chirp slope and tau is delay.
# But more directly: the apodization in ADC sample domain is
# exp(-(pi * beta * slope * tau_k / c)^2) for each ADC sample k.
#
# For simplicity, compute the scalar apodization at the center delay:
sin_theta_rx = torch.sqrt((1.0 - cos_theta_out**2).clamp(min=0.0))
beta = areas_t[v_idx].sqrt() * sin_theta_rx  # approx: sqrt(A)*sin(theta)
# Range-domain apodization factor
apod = torch.exp(-2.0 * (math.pi * self.slope * beta * tau_delay / C) ** 2)
weight = weight * apod
```

This computes `beta` approximately as `sqrt(A) * sin(theta_rx)` — the surfel's projected extent along the RX line-of-sight. The `sin(theta_rx)` factor means surfels viewed face-on (normal incidence) have `beta=0` and no attenuation, while surfels viewed at grazing have maximum attenuation. This is physically correct: a flat surface viewed face-on scatters coherently regardless of size.

Also add EVER-style anisotropy regularization (separate concern from scale-contribution coupling — prevents degenerate elongated shapes):
```python
scales = model.get_scales()
s_max = scales.max(dim=-1).values
s_min = scales.min(dim=-1).values
L_aniso = ((1 - opacities.detach()) * (s_max - s_min)).mean()
loss = loss + 0.01 * L_aniso
```

**Files:** `rasterizer_torch.py` (`_render_chunk_torch`), `train_gaussian.py` (anisotropy reg in loss)

**Expected impact:** +0.01-0.03 (corrects systematic over-estimation of large surfels, provides physics-based scale coupling)

**Run:** All 7 scenes, C3 mode, 1500 iters (includes F1-F3). Record to results table. Additionally record:
- Distribution of `beta` values (mean, max) to verify apodization is active
- Distribution of `apod` values (mean, min) to see how much large surfels are attenuated
- Whether scales converge to smaller values (expected: yes, since smaller surfels are now more efficient)

---

### Stage F5: Progressive BSDF Activation

**Change:** Activate BSDF parameters progressively: Fresnel -> roughness -> slab.

**Rationale (from papers):** 3DGS activates SH bands progressively (degree 0 -> 1 -> 2 -> 3, every 1000 iters). This prevents higher-order coefficients from compensating for incorrect lower-order terms. The radar analog: prevent roughness parameters from compensating for incorrect permittivity.

**Source:** 3DGS (Section 5.2, progressive SH activation)

**What changes:**
```python
# After loss.backward(), before optimizer.step():
if model.raw_materials.grad is not None:
    if it < 300:
        # Fresnel only: freeze roughness (sigma_h, l_c) and slab (tau, thickness)
        model.raw_materials.grad[:, 2:] = 0.0
    elif it < 800:
        # Fresnel + roughness: freeze slab only
        model.raw_materials.grad[:, 4:] = 0.0
    # else: all 6 params active
```

**Files:** `train_gaussian.py` training loop

**Expected impact:** +0.01-0.02 (better-conditioned optimization landscape)

**Run:** All 7 scenes, C3 mode, 1500 iters (includes F1-F4). Record to results table. Additionally record: per-parameter mean values at each activation boundary (iter 0, 300, 800, 1500) to verify progressive convergence.

---

### Stage F6: 3DGS-Style Adaptive Densification

**Change:** Implement clone/split/prune cycle based on accumulated position gradients.

**Rationale (from papers):** 3DGS Section 5.2 — clone under-reconstructed (small + high grad), split over-reconstructed (large + high grad), prune near-transparent. This is the only way to change the surfel count and distribution after initialization.

**Source:** 3DGS (clone/split/prune) + AbsGS (ACM MM 2024) absolute-value gradient fix, with scale thresholds adapted for radar surfel sizes.

**What changes:**
```python
# Before training loop:
grad_accum = torch.zeros(model.N, device=DEVICE)
grad_count = torch.zeros(model.N, device=DEVICE)
densify_interval = 200
densify_start = 500   # after progressive BSDF Phase A

# After backward, for active surfels:
# AbsGS fix: use sum of absolute values, not norm of gradient vector.
# Prevents "gradient collision" where opposing per-path gradients cancel
# in the norm, causing large surfels in complex regions to never split.
# Source: AbsGS (Ye et al., ACM MM 2024), GOF (Yu et al., SIGGRAPH Asia 2024)
if model.positions.grad is not None:
    active_grads = model.positions.grad.abs().sum(dim=-1)  # NOT .norm()
    grad_accum += active_grads
    grad_count += (active_grads > 0).float()

# Every densify_interval iters, starting at densify_start:
if it >= densify_start and it % densify_interval == 0:
    avg_grad = grad_accum / grad_count.clamp(min=1)
    scales = model.get_scales()
    scale_mean = scales.mean(dim=-1)
    opacities = model.get_opacities()

    # Clone: small surfels with high grad
    tau_grad = avg_grad[avg_grad > 0].quantile(0.9)  # top 10% gradient
    scale_thresh = scale_mean.median()
    clone_mask = (avg_grad > tau_grad) & (scale_mean < scale_thresh)

    # Split: large surfels with high grad
    split_mask = (avg_grad > tau_grad) & (scale_mean >= scale_thresh)

    # Prune: near-transparent
    prune_mask = opacities < 0.01

    # Execute: build new model with cloned/split/pruned surfels
    # Clone: duplicate at slight position offset along gradient
    # Split: two children at half scale, offset along major axis
    # Prune: remove from model
    ...
    # Rebuild optimizer with new param groups
    ...
    # Reset accumulators
    grad_accum = torch.zeros(model.N, device=DEVICE)
    grad_count = torch.zeros(model.N, device=DEVICE)
```

Note: the F4 apodization gives densification a new dimension. The optimizer has gradient pressure to split large surfels (since they're apodized) and clone small ones (since they're more efficient). Without F4, the optimizer has no reason to prefer small surfels.

**Files:** `train_gaussian.py`

**Expected impact:** +0.01-0.02 (especially for C4 from-scratch mode)

**Run:** All 7 scenes, C3 and C4 modes, 1500 iters (includes F1-F5). Record to results table. Additionally record: surfel count evolution over training (N at init, after each densify step, at end).

---

### Stage F7: Pre-Computed Shadow Visibility

**Change:** Compute per-surfel TX visibility once before training using Mitsuba BVH, then apply as a static mask during rendering.

**Rationale (from papers):** 3DGRT demonstrates efficient shadow testing via hardware BVH. Currently mm25DGS skips shadows entirely during training, letting occluded surfels contribute spurious energy.

**Source:** 3DGRT (OptiX BVH shadow rays)

**What changes:**
```python
# After reservoir sampler run, before freeing Mitsuba scene:
with torch.no_grad():
    active_pos = model.positions[active_mask].detach()
    N_act = active_pos.shape[0]
    tx_vis = torch.ones(N_act, rast.n_tx, dtype=torch.bool, device=DEVICE)
    for t in range(rast.n_tx):
        tx_pos = rast.tx_positions[t]
        delta = tx_pos - active_pos
        d = delta.norm(dim=-1)
        dirs = delta / d.clamp(min=1e-6).unsqueeze(-1)
        occluded = rast._shadow_test_torch(active_pos, dirs, d)
        tx_vis[:, t] = occluded  # _shadow_test_torch returns visible (True=visible)
# NOW free Mitsuba scene
rast._mi_scene = None

# Pass tx_vis to render_gaussians -> _render_chunk_torch as additional mask
```

Modify `_render_chunk_torch` to accept an optional `tx_visibility` tensor and mask paths accordingly, replacing the current `_shadow_test_torch` call.

**Files:** `train_gaussian.py`, `rasterizer_torch.py`

**Expected impact:** +0.005-0.01 (scene-dependent — matters most for self-occluding geometries)

**Run:** All 7 scenes, C3 mode, 1500 iters (includes F1-F6). Record to results table.

---

## Stage Summary

| Stage | Change | Source Paper | Est. impact |
|-------|--------|-------------|-------------|
| F1 | Remove MAX_ACTIVE cap | — (engineering) | +0.02-0.04 |
| F2 | Cosine LR decay + 1500 iters | 3DGS, EVER, VC-3DGS | +0.01-0.02 |
| F3 | Live surfel areas from scales | 3DGS, EVER, VC-3DGS, 3DGRT | +0.02-0.04 |
| F4 | Range-domain Gaussian apodization + anisotropy reg | **VC-3DGS** (ray-space beta), **EVER** (aniso reg) | +0.01-0.03 |
| F5 | Progressive BSDF activation | **3DGS** (progressive SH) | +0.01-0.02 |
| F6 | Adaptive densification | **3DGS** (clone/split/prune) | +0.01-0.02 |
| F7 | Pre-computed shadow visibility | **3DGRT** (BVH shadow) | +0.005-0.01 |

**Total estimated: 0.08-0.14** (should close the 0.077 gap)

---

## Results Tracking

After each stage, run all 7 scenes and record results in `md/gap_closure_results.md` using this format:

```markdown
## Stage FN: [Description]

### Changes Made
- [file:line] description of change

### Results

| Scene | mmIR | Baseline C3 | FN result | FN-mmIR gap | Improvement over baseline |
|-------|------|-------------|-----------|-------------|--------------------------|
| seq_0_frame_135 | 0.920 | 0.870 | ??? | ??? | ??? |
| ... | | | | | |
| **Mean** | **0.915** | **0.838** | **???** | **???** | **???** |

### Diagnostics
- [F3+: scale distribution at init vs end]
- [F4: beta distribution, apodization factor distribution]
- [F5: per-parameter values at phase boundaries]
- [F7: surfel count evolution]

### Observations
- [Did scales evolve? Did opacities change? Did surfel count change?]
- [Any scenes that regressed? Why?]
- [Is the improvement consistent across scenes or scene-dependent?]

### Decision: KEEP / REVERT
- [Reason]
```

### Revert Policy

If a stage degrades mean cart_corr by more than 0.005, it is reverted before proceeding to the next stage. Scene-specific regressions of up to 0.01 are acceptable if the mean improves, given MC noise of +/-0.03.

---

## What NOT to Change

- **BSDF evaluation** (`bsdf_torch.py`): Verified correct in Stage A/B.
- **Antenna gain evaluation**: Verified correct, already learnable.
- **Radar constants / phase computation**: Calibrated, detached by design.
- **RX-sphere splatting formula**: Algebraically correct, already implemented. The apodization in F4 augments it (multiplicative factor on weight), it does not replace it.
- **Stage B mesh training**: No changes — it's the reference that validates our physics.
