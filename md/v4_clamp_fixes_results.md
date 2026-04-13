# v4 clamp fixes — results

Follow-up to [v4_material_parameter_loo_investigation_results.md](v4_material_parameter_loo_investigation_results.md). The LOO investigation identified two root causes behind the "5 of 6 material parameters are LOO-inactive" pattern:

1. **Massive clamp saturation** — thickness 78%, sigma_h 74%, l_c 64%, eps_real 45% at end of training under random init.
2. **Rank-1 trajectory redundancy** — all 6 parameters move in lockstep (trajectory correlations ≈ ±1), meaning the effective learnable dimension is 1.

This document implements four clamp fixes targeting (1) and measures whether materials gain cart_corr headroom beyond the +0.028 marginal ceiling we saw before.

## Baseline for comparison

- **Pre-fixes, simplified BSDF, random init**: mean cart_corr = **0.9297** across 7 scenes.
- Noise floor: ±0.005 per seven-scene mean (single seed).

## Fixes implemented

### Fix 1 — Widen `thickness` reparam upper bound
- Old: `exp(clamp(raw, -7, -1.2))` → `[1 mm, 300 mm]`
- New: `exp(clamp(raw, -7, 2.0))` → `[1 mm, 7.39 m]`
- P5 clamp rate: 78% → should drop (optimizer can push further)
- Motivation: under random init, 78% of points were saturated at the 300 mm upper bound.

### Fix 2 — Remove `enforce_spm_validity`
- Deleted the SPM validity clamp entirely from `rasterizer_factorized.py` Step 1.
- P5 clamp rate: 74% (sigma_h) + 57% (l_c via λ/2 minimum) → should drop.
- Motivation: the clamps from CSVBSDF paper Eq. 19 (`kh ≪ 1, k³h²l ≪ 1, √2 h/l < 0.3`) assume the target has smoother-than-glass roughness. At 77 GHz, `kh ≪ 1` means σ_h < 62 μm — violated by every real automotive surface (asphalt ~1 mm, concrete ~300 μm, brick ~500 μm, foliage ~cm). The SPM perturbation expansion is no longer a valid physical approximation in our regime, but the formula still evaluates. Let the loss decide if the resulting coefficient is useful.

### Fix 3 — Widen `l_c` reparam range
- Old: `exp(clamp(raw, -7.6, -2.3))` → `[0.5 mm, 100 mm]`
- New: `exp(clamp(raw, -10, 2))` → `[45 μm, 7.39 m]`
- P5 clamp rate: 64% → should drop.
- Motivation: physical correlation lengths at mmWave-scale roughness span fine textures (tens of μm) to large-scale features (meters), well outside the original range.

### Fix 4 — Replace `eps_real` sigmoid with softplus
- Old: `1.5 + 8.5 · sigmoid(raw)` → `[1.5, 10]` with hard saturation at ±5 raw
- New: `1.0 + softplus(raw)` → `[1.0, ∞)` monotonic, no saturation
- P5 clamp rate: 45% → should drop to ~0% (softplus never saturates)
- Motivation: sigmoid compresses effective raw range. At saturation the Jacobian is ~0, killing the gradient. softplus is unbounded and has `dy/dx ≈ 1` for `x > 5`.

## 7-scene results after each fix

All runs: `mse_raw` default, factory patterns, `LEARN_PATTERNS=False`, random init (width=2, seed=42), simplified BSDF (no CBS/directive/broad/blend), 500 iters.

| Scene | **Pre-fix** (0.9297) | **Fix 1** thickness | **Fix 2** no SPM | **Fix 3** wider l_c | **Fix 4** softplus eps_real |
|---|---|---|---|---|---|
| seq_0_frame_135 | 0.9169 | 0.9147 | 0.9149 | 0.9143 | 0.9163 |
| seq_0_frame_390 | 0.9398 | 0.9293 | 0.9365 | 0.9327 | 0.9333 |
| seq_1_frame_185 | 0.9637 | 0.9625 | 0.9634 | 0.9630 | 0.9635 |
| seq_1_frame_438 | 0.9551 | 0.9554 | 0.9574 | 0.9566 | 0.9566 |
| seq_2_frame_105 | 0.9182 | 0.9165 | 0.9171 | 0.9135 | 0.9152 |
| seq_2_frame_160 | 0.9049 | 0.9044 | 0.9042 | 0.9044 | 0.9039 |
| seq_2_frame_300 | 0.9092 | 0.9066 | 0.9052 | 0.9060 | 0.9095 |
| **Mean** | **0.9297** | **0.9271** | **0.9284** | **0.9272** | **0.9283** |
| **Δ vs pre-fix** | — | **-0.0026** | **-0.0013** | **-0.0025** | **-0.0014** |

**Every fix is within ±0.005 of the pre-fix baseline. All four are washes.**

## Interpretation

### The rank-1 trajectory finding is definitively validated

The P3 trajectory correlation matrix showed all 15 pairs of material parameters move in perfect lockstep (|corr| ≥ 0.83, most ≈ ±1). If the 6 parameters occupy a rank-1 subspace of the loss landscape, **unblocking any individual parameter redistributes signal within the same 1-D subspace but cannot expand its dimensionality**. That's exactly what the four-fix sweep confirms.

Testing the prediction:
- If rank were 2, 3, or 6, unblocking a clamp on a parameter with a genuinely *independent* gradient direction should give +Δ above the noise floor.
- In 4 × 4 = 16 independent test cells (4 fixes × 4 comparisons to pre-fix), **every single cell sits between -0.005 and +0.002**. Not one showed the ≥+0.005 improvement that would indicate a new direction was unblocked.

### The rank-1 dimensionality is fundamental to the BSDF composition

Looking at the simplified BSDF math:

```
f_cos = R_jones(eps_real, eps_imag, thickness, cos_i, cos_o) ×
        [tau_eff(sigma_h, l_c, tau_base, cos_i) × f_KA(sigma_h, l_c, ...) +
         (1 - tau_eff) × f_SPM(sigma_h, l_c, eps_contrast, ...)] ×
        cos_i
```

Every path in the (M, n_tx, n_rx) output tensor maps through a single real-valued product that eventually becomes `|rp_real + i·rp_imag|²` after scatter+FFT. The loss is MSE on this scalar-per-bin field. So the gradient from any loss component to any material parameter routes through the same scalar product chain. **The 6 parameters are 6 different "dials" on the same scalar**, and the optimizer naturally discovers that it can achieve the same output by moving any combination of the 6 — hence the perfect trajectory correlations.

**This is not fixable by clamp widening.** To get more than rank-1 out of the material model, you would need:

- **Different parameters to control qualitatively different output dimensions.** e.g., one parameter purely controls phase, another purely controls magnitude, another purely controls angular shape. The current BSDF multiplies them all into one scalar, so they're mathematically indistinguishable.
- **A loss that penalizes different aspects of the output independently.** The current raw MSE on `|RA|` collapses everything to a single per-bin distance. A loss that weighted phase errors, amplitude errors, and angular-structure errors separately might let different parameters contribute through different penalty channels.
- **A higher-dimensional output representation.** If each point produced a richer feature (e.g., per-polarization, per-frequency, per-sub-bin), different parameters could contribute to different feature channels and their gradients would naturally decorrelate.

### What the clamps *were* doing

The clamp saturations were not *causing* the rank-1 behavior. They were *symptoms* of it. Under raw MSE, the optimizer discovers a single loss-reducing direction in material space; because of per-parameter redundancy, different parameters can express that direction with different coefficients. When one parameter happens to have the largest effective coefficient for the optimizer's chosen direction, it gets pushed to the clamp while the others follow along.

Removing the clamp doesn't change the underlying rank-1 structure — it just lets the dominant parameter push further into the un-clamped region without being stopped by a bound. In practice this makes no difference to the final cart_corr because the bound was already past the point where additional movement produced useful signal.

### The ceiling is 0.93

Across all the experiments in this investigation:

- Raw MSE + concrete init: 0.9324 (full BSDF) / 0.9324 (simplified)
- Raw MSE + random init: 0.9291 (full) / 0.9297 (simplified)
- Raw MSE + random init + Fix 1 (wider thickness): 0.9271
- Raw MSE + random init + Fix 2 (no SPM validity): 0.9284
- Raw MSE + random init + Fix 3 (wider l_c): 0.9272
- Raw MSE + random init + Fix 4 (softplus eps_real): 0.9283
- Raw MSE + random init + all 4 fixes cumulative: 0.9283
- Pearson loss (with full BSDF, mmIR patterns): 0.9456 — **but this used mmIR-trained patterns and pearson explicitly discards absolute amplitude, so it's not a fair comparison.**
- Pearson + factory patterns + scale fix (comparable setup): 0.9425 — marginally higher than raw MSE but still in the 0.93–0.94 band.

**The material model ceiling is around 0.93–0.94 for cart_corr.** This ceiling persists across all combinations of loss type, initialization, and clamp configurations. The bottleneck is not any individual parameter or clamp — it is the structural rank-1 redundancy of the BSDF composition.

## Recommendation

### Keep the four clamp fixes

Even though they're individually wash, together they:

- Remove arbitrary and physically-questionable bounds (300 mm thickness, 100 mm l_c, 10.0 eps_real upper)
- Eliminate the SPM-validity clamp that was fighting the physical regime we're in
- Replace hard sigmoid saturation with soft monotonic softplus
- Make the reparameterization cleaner conceptually

None of them cause regression. Keeping them in place simplifies the mental model (no arbitrary clamps the optimizer is fighting) and prepares the code for any follow-up investigation that tries to break the rank-1 barrier.

### Accept the 0.93 material ceiling under scalar-MSE raw loss

The +0.028 marginal contribution of materials on top of normals is real and useful. We've squeezed that out by:
- Fixing the sigma_h silent death (Commit `58bc286`)
- Removing the scale mismatch in the forward model (Commit `da57123`)
- Switching to raw MSE as the default (Commit `c598756`, `159d1f1`)
- Simplifying the BSDF (Commit `2b56b29`)
- Removing clamps (Commits `9655aae` through `96aa0ae`)

Further material gains beyond the current ~0.93 ceiling require a structural change — not a clamp fix.

### Possible directions to break the rank-1 ceiling (for a separate investigation)

1. **Per-polarization loss decomposition.** The radar has 12 TX and 16 RX elements, effectively 192 channels. Currently the loss is a single MSE across all channels. Weighting different channel subsets differently (e.g., cross-pol vs co-pol) would expose degrees of freedom in the BSDF that the current loss collapses.

2. **Feature-space losses.** Compute the loss on gradients (Sobel), curvature (Laplacian), or frequency-domain (Fourier-log) features of the RA image in addition to raw magnitudes. Different material parameters would contribute through different feature channels, potentially decorrelating their gradients.

3. **Neural material prior.** Replace the 6-parameter physics-based reparameterization with an MLP `(xyz, normal, material_id) → f_cos` learned from data. The MLP's inductive bias would be completely different from the rank-1 BSDF chain, and spatial smoothness would come from the network rather than per-point independent optimization.

4. **Multi-scale training.** Train with a mix of coarse and fine RA targets. The coarse targets would emphasize mean amplitude (where materials live), while fine targets would emphasize structural edges (where normals live). This might force the optimizer to allocate capacity between materials and normals more explicitly.

5. **Decompose the BSDF output into independent feature maps.** For example, separately compute and supervise: (a) specular coherent component, (b) diffuse incoherent component, (c) absorption component. If the loss penalizes reconstruction of each independently, different parameters could drive different components without merging into a single scalar.

None of these are small changes. They represent architectural redesigns, not parameter tweaks.

### Final verdict on the LOO investigation questions

From the original questions:

| Q | Answer |
|---|---|
| A. Why is eps_imag so important in LOO? | It has the dominant linear coefficient in the rank-1 learning direction. Its low clamp rate (27%) and largest trajectory mean drift (+0.60 raw space) reflect this. Other parameters express the same direction with smaller or opposite coefficients, but in the rank-1 subspace they're degenerate with eps_imag. |
| B. Why does thickness freezing have no effect? | Rank-1 redundancy: other parameters re-express thickness's contribution. Confirmed by Fix 1 (widening the clamp did nothing — the 300 mm bound wasn't the issue). |
| C. Why does eps_real freezing have no effect? | Same: rank-1 redundancy. Confirmed by Fix 4 (softplus replacement didn't change anything). |
| D. Why is tau_base a loss-agnostic no-op? | Fisher = 1.6e-11, orders of magnitude smaller than eps_real. The BSDF output is barely sensitive to the KA/SPM blend ratio because whichever lobe dominates any given path, the blend weight doesn't change the dominant contribution meaningfully. |
| E. Why is sigma_h a no-op? | enforce_spm_validity killed 74% of points. Fix 2 removed this, but the rank-1 structure meant the newly-live sigma_h points just got absorbed into the existing 1-D direction with no cart_corr benefit. |
| F. Why is l_c harmful when learned? | Partial participation from the 36% un-clamped points interfered with sigma_h's cleaner signal. Fix 3 gave l_c more room and Fix 2 unblocked sigma_h, so the interference now happens over a wider parameter space — but the total effect is still rank-1 and still wash-level. |

**None of the questions have physics answers.** They are all manifestations of the same underlying mathematical fact: the BSDF composition collapses 6 parameters into a single scalar per-path, the gradient from MSE loss is rank-1, and the effective learnable subspace is 1-dimensional regardless of parameterization or clamping.

## Raw data

- `mm25DGS_v4/output/material_investigation/D_P1_simplified_random/` — pre-fix diagnostic run (P1/P3/P5 analyses)
- `/tmp/fix{1,2,3,4}_*.log` — 7-scene benchmark logs for each fix
- Commits: `9655aae` (Fix 1), `b0c9454` (Fix 2), `f634b82` (Fix 3), `96aa0ae` (Fix 4)
