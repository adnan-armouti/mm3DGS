# mmIR vs v3: Step-by-Step Forward Model Comparison

## mmIR's steps (from hit positions to phasors)

Given N hit positions on the scene surface, expanded to N × n_tx × n_rx paths:

| Step | mmIR operation | Formula |
|------|---------------|---------|
| 1 | **Geometry**: compute directions and distances per path | `wi = (TX - hit)/d_tx`, `wo = (hit - RX)/d_rx` (negated for convention) |
| 2 | **Normal flip**: double-sided, per path | `if dot(wo, N) < 0: N = -N` |
| 3 | **Cosine filter**: skip back-facing | `active = (cos_i > ε) & (cos_o > ε)` |
| 4 | **Shadow test**: ray-trace to TX | `active &= !occluded(hit → TX)` |
| 5 | **BSDF**: Jones f_cos (power × cosine) | `brdf_weight = f_cos(wo, wi, N, materials)` |
| 6 | **Antenna gain**: TX and RX patterns | `brdf_weight *= G_tx × G_rx` |
| 7 | **MC correction**: importance weight | `brdf_weight *= 1 / (pdf × N_attempted)` |
| 8 | **Path loss**: TX-only (RX absorbed by MC) | `path_loss = 1 / d_tx²` |
| 9 | **Amplitude**: power → field | `weight = C_radar × sqrt(brdf_weight × path_loss)` |
| 10 | **Phase + scatter**: phasor into ADC | `ADC[t,r,k] += weight × cos(φ_const + φ_slope × t_k)` |

The key: `brdf_weight` at step 9 is the product `f_cos × G_tx × G_rx × mc_correction`, and the final weight is `C × sqrt(that × 1/d_tx²)`.

---

## v3's steps (from Gaussian centers to range-profile phasors)

| Step | v3 operation | Same as mmIR? | Notes |
|------|-------------|---------------|-------|
| 1 | **Geometry**: directions and distances | **YES** | Identical: `wi = (TX - μ)/d_tx`, RX direction per element |
| 2 | **Normal flip**: per TX (not per path) | **~YES** | Minor: v3 flips by TX direction, mmIR by RX. `cos_o = abs()` compensates |
| 3 | **Cosine filter** | **YES** | `active_i & active_o` |
| 4 | **Shadow test** | **NO** | Disabled (`shadow_mask=None`). Could add but currently hurts (self-occlusion at vertices) |
| 5 | **BSDF**: factorized Jones f_cos | **YES** | Exact same BSDF, factorized layout but identical math |
| 6 | **Antenna gain** | **YES** | Same patterns, same evaluation |
| 7 | **MC correction** | **NO** | v3 uses `mc_correction = π / cos(θ_bore)`. mmIR uses `1/(hit_pdf × N_attempted)` where `hit_pdf = cos(θ)/π` from actual reservoir sampling |
| 8 | **Path loss** | **NO** | v3: `1/d_tx²` in alpha_tx, PLUS the mc_correction replaces what was `dΩ = A×cos_o/d_rx²`. mmIR: `1/d_tx²` only, with `d_rx²` absorbed by MC Jacobian |
| 9 | **Amplitude** | **YES** (given correct inputs) | Same formula: `C × sqrt(product × path_loss)` |
| 10 | **Phase + scatter** | **~YES** | v3 uses range-profile splatting (PSF), mmIR uses ADC scatter. Verified equivalent to 0.15% |

---

## Steps that differ: 7 and 8

### Step 7: MC correction

**mmIR**: `mc_correction = 1 / (pdf_j × N_attempted)`

where `pdf_j` is the cosine-hemisphere sampling PDF of the ray that found hit j. This is `cos(θ_rx_surface)/π` where `θ_rx_surface` is the angle between the RX ray direction and the surface normal at the hit point.

**v3 (current)**: `mc_correction = π / cos(θ_bore)` 

where `θ_bore` is the angle between the RX-to-Gaussian direction and the RX boresight.

**These are different quantities:**
- mmIR's `pdf` uses `cos(θ)` relative to the SURFACE NORMAL at the hit (from cosine-hemisphere sampling oriented along the surface normal)
- v3's correction uses `cos(θ)` relative to the RX BORESIGHT

Actually, re-reading mmIR's sampler (sampler.py L85):
```
hit_pdf: Per-hit sampling PDF ρ_rx(ω_r) = cos(θ)/π for cosine-weighted hemisphere
```

This is `cos(θ)/π` where θ is the angle from the RX hemisphere's zenith (which is the RX boresight direction, since rays are cast from the RX). So mmIR's PDF IS relative to the RX boresight, not the surface normal.

So v3's `mc_correction = π / cos(θ_bore)` should give the same `1/pdf` as mmIR's `1/(cos(θ_bore)/π) = π/cos(θ_bore)`. ✓

The remaining difference: mmIR also divides by `N_attempted` (number of rays fired per RX). v3 doesn't. But `N_attempted` is a constant (same for all hits from the same RX), so it's a per-RX scale factor. Since the loss uses **min-max normalization**, a per-RX scale factor divides out.

Wait — it's per-RX, not global. If different RX elements fired different numbers of rays, the relative weighting between RX channels would differ. Let me check:

mmIR uses `n_attempted_per_rx` which may vary per RX. But in practice, the reservoir sampler fires the same `n_rays_per_res × n_res` rays per RX, so `N_attempted` is constant across RX elements.

**Conclusion: Step 7 is actually correct in v3** (modulo the constant N_attempted which cancels in min-max normalization).

### Step 8: Path loss

**mmIR**: `path_loss = 1 / d_tx²` (only TX distance, RX absorbed by MC Jacobian)

**v3**: `alpha_tx = sqrt(G_tx) / d_tx` → gives `1/d_tx²` in the squared weight. ✓

But `alpha_rx = sqrt(G_rx × mc_correction)` = `sqrt(G_rx × π / cos_bore)`. There's NO `1/d_rx²` here, which is correct for the MC formulation (d_rx² absorbed).

Wait — but `areas` is no longer used anywhere in the weight! The function signature has `areas` but it's never multiplied in. The `mc_correction` replaced it entirely.

**This means opacity is also not being used.** In `render_gaussians_factorized`, `areas = vertex_areas × opacities`, and this is passed to `render_factorized`, but `render_factorized` never uses `areas` in the weight computation.

## The actual bugs

### Bug A: `areas` (and thus opacity) is disconnected from the weight

When I changed from `dOmega = areas × cos_o / d_rx²` to `mc_correction = π / cos_bore`, I removed the only place where `areas` enters the weight. Now `areas` is passed in but never used. Opacity has no effect on the rendering.

**Fix**: The MC formulation doesn't use areas. But opacity should still modulate the weight:
```python
alpha_rx = sqrt(G_rx * mc_correction) * opacity_per_gaussian
```

Or alternatively, multiply `w_full` by opacity.

### Bug B: vertex_areas should NOT be multiplied with MC correction

In the MC formulation, the correction `1/(pdf × N)` replaces the area entirely. The area IS the MC weight — not `area × mc_weight`. The v3 code currently computes `areas = vertex_areas × opacities` but doesn't use it. If we re-connect areas, we must NOT multiply vertex_areas — only opacity.

**Fix**: In render_factorized, the weight should include opacity but NOT vertex_areas:
```python
w_full = C_radar * sqrt(f_cos) * alpha_tx * alpha_rx * opacity
```

where `alpha_rx = sqrt(G_rx × π / cos_bore)` and opacity comes from the model.

---

## Summary of what needs to change

| # | What | Current | Should be | Impact |
|---|------|---------|-----------|--------|
| A | `areas` / opacity | Disconnected (not used in weight) | Multiply `w_full` by opacity | **Critical** — opacity has no effect |
| B | `vertex_areas` in MC mode | Passed in but unused | Should NOT enter the weight chain | Already correct (accidentally) |
| C | `mc_correction` formula | `π / cos(θ_bore)` | Same — this IS correct | No change |
| D | Path loss | `1/d_tx²` only | Same — correct for MC formulation | No change |

**The only required fix is Bug A: reconnect opacity to the weight.**
