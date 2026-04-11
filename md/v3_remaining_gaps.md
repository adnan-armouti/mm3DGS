# v3 Remaining Gaps: Analysis and Options

Current v3 C4 mean gap: **0.065** from mmIR. v2 Stage B (mesh training with MC weights) mean gap: **-0.006** (matches mmIR). The 0.07 gap comes from the transition from MC-sampled hit positions to deterministic Gaussian positions.

## Exhaustive list of differences between v2 Stage B (matches mmIR) and v3 C4

| # | Difference | v2 Stage B | v3 C4 | Impact est. |
|---|-----------|------------|-------|-------------|
| 1 | **Scatterer positions** | Reservoir-sampled hit points on triangle surfaces | Gaussian centers (mesh vertices or LiDAR points) | High |
| 2 | **Area/weight factor** | MC correction `1/(pdf × N_attempted)` | Geometric `vertex_area × opacity × cos_o / d_rx²` | High |
| 3 | **Shadow rays** | Yes (Mitsuba ray_test per TX) | None | Medium |
| 4 | **Barycentric material interpolation** | Yes (smooth blending across triangles) | No (per-Gaussian direct values) | Low-medium |
| 5 | **Number of scatterers** | ~24K reservoir hits (importance-sampled) | ~12K-50K vertices (visible subset) | Medium |
| 6 | **Normal flipping convention** | Per-path (per TX-RX pair) | Per-TX only (cos_o uses abs of unflipped normal) | Low |
| 7 | **Carrier phase frequency** | `min_freq` (start of chirp) | `center_freq` | Low (but may affect azimuth) |
| 8 | **Path loss** | `1/d_tx²` (d_rx in MC Jacobian) | `1/(d_tx² × d_rx²)` via dΩ (explicit bistatic) | Already correct |
| 9 | **Output domain** | ADC → range FFT → azimuth FFT → RA | Range profile (PSF splat) → azimuth FFT → RA | Verified equivalent (0.15% error) |

---

## Issue 1: Shadow rays (Stage F7)

### Problem
v3 has no occlusion handling. Gaussians behind other geometry still contribute energy. In scenes with self-occlusion (building corners, overhangs), this adds spurious signal.

### Options

**Option A: Pre-computed shadow mask (3DGRT-inspired)**

Run the reservoir sampler ONCE at init to identify which (Gaussian, TX) pairs are occluded. Store as a binary mask `shadow_mask[m, t]`. Apply during rendering:
```python
active = active & shadow_mask[active_gaussians][:, :, None].expand(-1, -1, n_rx)
```

- Cost: one reservoir sampler call at init (~1s) + O(M × n_tx) shadow ray tests
- The mask is fixed (doesn't update as positions move). Recompute every N iterations if positions move significantly.
- Rasterization-native: no ray tracing during training, only at init.

**Option B: Depth-buffer occlusion**

For each TX element, rasterize Gaussians sorted by distance. Mark closer Gaussians as "occluding" later ones along the same direction. Approximate but fully differentiable and rasterization-native.

- More complex to implement
- Differentiable (opacity can learn to "unblock" occluded paths)
- Standard approach in 3DGS (alpha compositing with sorted primitives)

**Recommendation: Option A for simplicity.** The shadow mask is computed once using the existing Mitsuba scene (which we already load for reservoir-based culling). It covers the dominant occlusion effects. Option B is the medium-term upgrade.

### Implementation sketch for Option A:
```python
# At init, after reservoir sampler:
# For each Gaussian × TX pair, test shadow ray
shadow_mask = torch.ones(M, n_tx, dtype=torch.bool, device=device)
for t in range(n_tx):
    origins = positions + 1e-4 * dir_to_tx[:, t]
    rays = mi.Ray3f(origins, dir_to_tx[:, t])
    rays.maxt = d_tx[:, t] - 2e-4
    occluded = scene.ray_test(rays)
    shadow_mask[:, t] = ~occluded
# In rasterizer_factorized.py: w_full *= shadow_mask[..., None].float()
```

---

## Issue 2: Sampling distribution / MC correction

### The fundamental mismatch

In mmIR, the received signal is estimated via MC integration over the RX hemisphere:
```
E = (1/N) × Σ_j f(ω_j) / p(ω_j)
```
where `p(ω)` is the cosine-hemisphere sampling PDF and `1/(p × N)` corrects for the non-uniform sampling. This correction ensures that the estimate is **unbiased** regardless of where the samples land.

In v3, we have Gaussians at fixed positions (mesh vertices). There is no sampling PDF — the positions are deterministic. The weight `vertex_area × opacity × dΩ_rx` replaces the MC correction. But this weight assumes the Gaussians uniformly tile the scene surface, which they don't — mesh vertices are denser in highly-tessellated regions and sparser in large-triangle regions.

### Options

**Option 2A: Redistribute Gaussians to match cosine-hemisphere density**

Your suggestion: add/move Gaussians so their density on the RX hemisphere follows a cosine distribution. This would make the deterministic sum equivalent to an MC estimate with cosine sampling.

- Implementation: for each RX element, project all Gaussian centers onto the unit sphere, compute the local density, and compare with `cos(θ)/π`. Under-represented regions get new Gaussians; over-represented regions get pruned.
- Problem: the "correct" distribution is per-RX, but we have 16 RX elements at different positions. The optimal distribution differs for each.
- This is essentially adaptive density control guided by the MC sampling distribution.

**Option 2B: Compute per-Gaussian importance weights from the reservoir sampler**

Run the reservoir sampler once at init. For each Gaussian, count how many reservoir hits land near it (within some radius). This hit frequency IS the MC sampling density at that location. Use it as the area weight:

```python
# At init:
hit_positions = reservoir_sampler_hits  # (24K, 3)
for each Gaussian m:
    n_nearby_hits = count(hits within radius r of Gaussian m)
    importance_weight[m] = n_nearby_hits / (total_hits × density_normalization)
```

This makes the Gaussian weights approximate the `1/(pdf × N)` correction that mmIR uses. Gaussians in regions with many hits (high visibility) get higher weight.

- Simple to implement
- Bridges the gap between MC and deterministic summation
- One-time computation at init
- Doesn't require changing Gaussian positions

**Option 2C: Analytically compute the equivalent MC weight per Gaussian**

For a Gaussian at position μ visible from RX element r, the equivalent MC weight is:
```
w_mc = 1 / (p(ω_r(μ)) × N_eff)
```
where `p(ω) = cos(θ_rx) / π` (cosine-hemisphere PDF) and `N_eff` is the effective number of samples (related to total Gaussian count within the RX hemisphere).

```python
cos_theta_rx = |dot(normal_to_rx, rx_boresight)|
p_cosine = cos_theta_rx / pi
N_eff = n_active_in_hemisphere  # count of Gaussians visible from this RX
mc_weight = 1.0 / (p_cosine * N_eff)
```

Currently we use `vertex_area × cos_o / d_rx²`. The ratio between this and `1/(p × N)` is the "correction factor" needed:
```
correction = mc_weight / (vertex_area × cos_o / d_rx²)
           = d_rx² / (vertex_area × cos_o × p × N)
           = d_rx² × π / (vertex_area × cos_o² × N)
```

This correction depends on d_rx (per RX element) and cos_o (per Gaussian × RX), making it a per-path scalar. It's cheap to compute and could be applied as an additional multiplicative factor on `alpha_rx`.

- Analytically motivated
- Per-(Gaussian, RX) correction
- No reservoir sampler needed
- May over-correct in regions where Gaussian density doesn't match the cosine distribution

**Option 2D: Learn the importance weight (Option J from closing_c3_gap.md)**

Add a learnable `log_importance` parameter per Gaussian. The optimizer discovers the correct weighting by minimizing the RA loss. Initialize from the analytical weight (Option 2C) or from reservoir hit frequency (Option 2B).

- Most flexible
- Absorbs all weighting errors (MC, area, occlusion)
- Risk of overfitting
- Adds 1 parameter per Gaussian

**Recommendation: Start with Option 2B** (reservoir hit frequency). It's simple, principled, and directly bridges the MC-to-deterministic gap. If insufficient, upgrade to Option 2D (learnable weights).

---

## Issue 3: Other differences worth addressing

### 3A: Carrier phase frequency mismatch

v2 Stage B uses `self.min_freq` (= `carrierFrequency` from config = 77 GHz) for the carrier phase. v3 uses `rast.center_freq` which is also `carrierFrequency`. So these should be the same — but worth verifying explicitly.

If they differ, the azimuth encoding (which relies on carrier phase differences across TX-RX elements) would be wrong, corrupting the RA image's angular structure.

**Action: Verify `rast.center_freq == rast.min_freq` in the code. If not, fix to match v2.**

### 3B: Normal flipping inconsistency

v3 flips normals per-TX (line 113-118):
```python
cos_i = (wi * normals[:, None, :]).sum(-1)  # may be negative
normal_sign = where(cos_i < 0, -1, 1)
n_eff = normals[:, None, :] * normal_sign.unsqueeze(-1)
```

But `cos_o` is computed with the ORIGINAL normals (line 196):
```python
cos_o_raw = (dir_hit_to_rx * normals[:, None, :]).sum(-1)
cos_o = cos_o_raw.abs()
```

In v2's per-path renderer (`_render_chunk_torch`), the normal is flipped ONCE per path based on the RX direction, then BOTH `cos_i` and `cos_o` use the flipped normal consistently.

This means in v3:
- `cos_i` uses the TX-flipped normal (correct for BSDF evaluation)
- `cos_o` uses `abs()` of the unflipped normal (correct magnitude but loses sign information for the BSDF's `wo·n` terms)
- The BSDF's KA lobe uses `wo_dot_n = einsum(wo, n_eff)` where `n_eff` is TX-flipped — so the KA lobe's `cos_o` IS based on the flipped normal. Inconsistency is between the BSDF's internal `cos_o` and the external `cos_o` used for `dOmega`.

**Impact: Low.** Both `cos_o` values have the same magnitude (due to `abs()`). The sign only matters for the BSDF, which uses `n_eff` internally. The `dOmega` computation correctly uses `abs(cos_o)`.

### 3C: Barycentric material interpolation (missing in v3)

v2 Stage B interpolates materials and normals across triangle faces using barycentric coordinates. This provides smooth spatial variation. v3 evaluates materials at discrete Gaussian centers — no interpolation.

**Impact: Low-medium.** The material field is piecewise-constant in v3 (one value per Gaussian) vs piecewise-linear in v2 (barycentric interpolation). This is inherent to the Gaussian representation and not easily fixable without changing the representation.

### 3D: min_freq vs center_freq for carrier phase

v2's `_render_chunk_torch` line 457: `phi_const = TWO_PI * self.min_freq * tau_delay`
v3's rasterizer line 183: `phi_tx_const = TWO_PI * rast.center_freq * tau_tx`

Both `self.min_freq` and `rast.center_freq` are set from `radar_cfg['carrierFrequency']` (rasterizer_torch.py line 166,170). So they should be identical. But the NAMING is confusing — `min_freq` suggests start-of-chirp while `center_freq` suggests center. Worth a one-line verification.

---

## Priority ranking

| Issue | Expected impact | Effort | Recommendation |
|-------|----------------|--------|----------------|
| **2B: Reservoir hit-frequency weights** | High (bridges MC gap) | Low (init-time computation) | **Do first** |
| **1A: Pre-computed shadow mask** | Medium (removes spurious energy) | Low-medium | **Do second** |
| 3A: Verify carrier frequency | Low (likely already correct) | Trivial | Quick check |
| 3B: Normal flip consistency | Very low | Low | Fix if easy |
| 3C: Barycentric interpolation | Inherent to representation | N/A | Accept |
