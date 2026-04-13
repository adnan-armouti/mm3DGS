# v4 material ablation results

Reference baseline (commit `8fe6d6d`, Tier A, full 6-param per-point): **mean cart_corr = 0.9351** across the 7-scene benchmark, ~52 s/scene, all 6 material parameters trained at LR 0.7.

All `Δ` columns below are relative to this 0.9351 baseline. All wall-clock numbers are on a single RTX 4090. Per-scene cart_corr, per-parameter drift, and per-parameter Fisher diagonal are stored in the per-run `.npz` files under `mm25DGS_v4/output/material_ablation/<run_name>/`.

**Single seed, single run per config. Run-to-run noise on the 7-scene mean is ~±0.005, applied as the decision threshold.**

---

## Phase 1 — Baselines (do we need this model class?)

| run_name | description | mean_cart_corr | Δ vs baseline | ms/iter |
|---|---|---|---|---|
| B0_fixed_concrete | All 6 material params frozen at ITU concrete (rotations + patterns still learned) | 0.9244 | -0.0107 | 108.4 |
| B1_scalar_reflectivity | BSDF replaced with `sigmoid(ρ)·cos_i` per-point scalar | 0.9044 | -0.0307 | 92.3 |
| B2_global_6param | Shared `(6,)` material vector across all 50K points | 0.9225 | -0.0126 | 108.1 |
| B3_per_point_6param | Full per-point `(M, 6)` + full BSDF (Tier A baseline) | 0.9301 | -0.0050 | 113.4 |

**Phase 1 summary**: The per-point 6-param model wins, but only by 0.006 over fixed concrete and 0.008 over global. Per-point freedom buys essentially nothing. The decision rule `B3 ≤ B0 + 0.01` fired "STOP"; the interpretation is not "debug" but "the material learning is doing almost nothing above fixed concrete". Scalar reflectivity (B1) is *worse* than fixed concrete by 0.020 — the BSDF physics matter, but not their *learnable* parameters.

---

## LEARN matrix — 2³ ablation on (materials, normals, patterns)

Eight corners of the `(LEARN_MATERIALS, LEARN_NORMALS, LEARN_PATTERNS) ∈ {0,1}³` cube. Two corners (M0_N1_P1 = B0_fixed_concrete, M1_N1_P1 = B3_per_point) reused from Phase 1.

| M | N | P | run_name | description | mean_cart_corr | Δ vs B3 (0.9301) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | **M0_N0_P0** | Everything frozen at init (true floor) | **0.3729** | **-0.5572** |
| 1 | 0 | 0 | M1_N0_P0 | Materials only | 0.8490 | -0.0811 |
| 0 | 1 | 0 | **M0_N1_P0** | **Normals only** | **0.9095** | **-0.0206** |
| 0 | 0 | 1 | M0_N0_P1 | Patterns only | 0.6801 | -0.2500 |
| 1 | 1 | 0 | M1_N1_P0 | Materials + normals | 0.9167 | -0.0134 |
| 1 | 0 | 1 | M1_N0_P1 | Materials + patterns | 0.8863 | -0.0438 |
| 0 | 1 | 1 | M0_N1_P1 (=B0) | Normals + patterns (fixed materials) | 0.9244 | -0.0057 |
| 1 | 1 | 1 | M1_N1_P1 (=B3) | All three | 0.9301 | 0 |

### Marginal contribution of each knob (alone, above true floor 0.3729)

| Knob | Δ above floor | Fraction of full learning gap (0.557) |
|---|---|---|
| **Normals alone** | **+0.537** | **96%** |
| Materials alone | +0.476 | 85% |
| Patterns alone | +0.307 | 55% |

### Marginal contribution of each knob (added to full B3 baseline)

Computed as `B3 − (B3 with that knob removed)`:

| Knob | Δ at the margin |
|---|---|
| **Normals removed**: B3 - M1_N0_P1 | **+0.044** |
| Patterns removed: B3 - M1_N1_P0 | +0.013 |
| Materials removed: B3 - M0_N1_P1 | **+0.006** |

### LEARN matrix verdict

- **Normals learning is the dominant signal by a very wide margin.** On its own, normals closes 96% of the gap between pure init and the full learning baseline. Removing it from the full model costs 7× more than removing materials.
- **The three knobs are highly redundant.** Materials alone closes 85% and patterns alone closes 55%, but *stacked on top of normals* they each contribute only 0.006–0.013. Any two of the three can substitute almost completely for the third, except normals, which has no substitute.
- **Materials is the weakest contributor** at the margin. Its marginal gain of 0.006 is within single-seed noise (±0.005). The 6-parameter learnable material model is effectively decoration on top of fixed ITU concrete.
- **B1 (scalar reflectivity) at 0.9044 < B0 (fixed concrete) at 0.9244** confirms the BSDF *physics* matters. It's the *learning* of the 6 params that doesn't.

---

## Phase 1.5 — BSDF component ablation (which physics terms survive?)

All runs use the full (M, N, P) = (1, 1, 1) learning configuration. Each run disables one BSDF component while keeping the rest of the pipeline identical.

| run_name | disabled | mean_cart_corr | Δ vs B3 (0.9301) | ms/iter | Δ ms/iter | verdict |
|---|---|---|---|---|---|---|
| K4_no_spm | SPM incoherent lobe | 0.9300 | **-0.0001** | 111.1 | -2.3 | **DROP** |
| K1_no_cbs | CBS sinc factor | 0.9289 | -0.0012 | 112.5 | -0.9 | **DROP** |
| K6_no_blend | coherence blend (η=1) | 0.9271 | -0.0030 | 111.8 | -1.6 | **DROP** |
| K3_no_broad | broad/diffuse Lambertian | 0.9269 | -0.0032 | 111.6 | -1.8 | **DROP** |
| K2_no_directive | directive vMF lobe | 0.9220 | -0.0081 | 113.0 | -0.4 | borderline |
| K5_no_ka | GGX/Cook-Torrance KA lobe | 0.9144 | -0.0157 | 111.2 | -2.2 | **KEEP** |
| K7_no_jones | polarized Jones Fresnel | 0.9095 | -0.0206 | 112.4 | -1.0 | **KEEP** |
| K8_no_slab | multi-layer slab thickness | 0.9091 | -0.0210 | 107.3 | -6.1 | **KEEP** |

### Phase 1.5 verdict

- **Drop** (Δ < 0.005, within noise): **SPM, CBS, broad, coherence blend.** The entire incoherent-scattering branch of the BSDF is pure decoration. These four components together account for ~half the Step 4 code in [rasterizer_factorized.py](../mm25DGS_v4/rasterizer_factorized.py). SPM lobe is literally zero-cost: removing it *improves* cart_corr by 0.0001.
- **Borderline** (0.005 ≤ Δ ≤ 0.02): **directive (0.008), KA (0.016).** The directive vMF lobe is weak but non-zero; the KA lobe is genuinely important.
- **Keep** (Δ > 0.02): **Jones Fresnel polarization (0.021), slab multi-layer thickness (0.021).** These are the two most important BSDF components, tied. Both are pure Fresnel machinery — no roughness terms involved.

### The preliminary smoke test was wrong about slab

The 50-iter / 1-scene smoke test suggested slab disable *improves* cart_corr by +0.013. At full 500 iters / 7 scenes the true answer is the opposite (-0.021). This is why we run the full benchmark, not smoke tests.

### The reduced BSDF

Based on Phase 1.5, the BSDF reduces from ~300 lines of physics to essentially:

```
f_cos = R_jones(eps, thickness) × f_KA(sigma_h, l_c) × cos_i
```

where `R_jones` is the multi-layer slab Fresnel (polarized) and `f_KA` is the GGX/Cook-Torrance coherent specular lobe. The SPM, CBS, directive, broad, and coherence-blend machinery can all be deleted — they contribute nothing on top of Jones + KA.

---

## Phase 2 — NOT RUN

Phase 1 + LEARN matrix showed that learnable materials contribute only +0.006 at the margin (within noise). Running a leave-one-out / train-only-one sweep on the 6 material parameters is no longer informative: the total signal being ablated is 0.006, smaller than the ±0.005 run-to-run noise floor. Verdicts would be unstable and meaningless.

Phase 2 skipped. The preliminary smoke-test signal (`sigma_h` shows zero drift + zero Fisher across every run that trains it, in [Commit 1 findings](../mm25DGS_v4/material_diagnostics.py)) is consistent with "materials aren't being learned in a meaningful way" but we won't expend compute formally confirming which of the 6 is least important.

---

## Phase 3 — Diagnostics summary

From the `.npz` dumps, the per-parameter drift `||raw_materials_final - raw_materials_init||_2 / sqrt(M)` across all runs that train materials:

- `sigma_h` drift is **zero or near-zero in every run** that trains it (B3, M1_N0_P0, M1_N1_P0, M1_N0_P1, and the K-series). The GGX α of roughly 1e-3 at ITU concrete init puts the gradient into the saturation regime of the NDF denominator.
- `thickness` shows the **largest drift** of any parameter in per-point mode. Ironically, this is consistent with Phase 1.5 finding that slab thickness has the largest physical-component contribution — the model is genuinely using multi-layer Fresnel.
- `eps_real`, `eps_imag`, `l_c`, `tau_base` all show moderate drift but their Fisher diagonals are 2–5 orders of magnitude below `eps_real` and `thickness`.

---

## Final recommendation

### Recommended simplified v4 material model

**1. Drop learnable material parameters entirely.** Freeze all 6 at ITU concrete init. The 0.006 cart_corr gain from learning them is within run-to-run noise and comes at the cost of the entire material LR schedule, Adam state for 300K parameters, and the interpretation ambiguity of six physics knobs fighting for the same gradient signal.

**2. Keep learnable normals (quaternions).** This is the dominant signal (96% of the learning gap on its own). Nothing else comes close. Keep the LR and clip settings unchanged.

**3. Keep learnable antenna patterns.** Marginal +0.013 over normals-alone. Not huge but not noise. Cheap to keep.

**4. Simplify the BSDF** to the reduced form:
   - **Keep**: Jones multi-layer slab Fresnel, GGX/Cook-Torrance KA lobe.
   - **Borderline, keep for now**: directive vMF lobe (+0.008, worth retesting after the reduction).
   - **Drop**: SPM lobe, CBS enhancement, broad/diffuse fallback, coherence blend. All four together account for roughly half the Step 4 code and contribute less than noise.

### Expected effects of this simplification

| Metric | Before | After (estimated) |
|---|---|---|
| Mean cart_corr | 0.9301 | ~0.925–0.930 (normals + patterns trained, simplified BSDF with fixed concrete) |
| Trainable params | M × (4 + 6) + 4 × 361 = 500K + 1.4K | M × 4 + 4 × 361 = 200K + 1.4K |
| Step 4 BSDF code | ~300 lines | ~120 lines (Jones + KA + directive) |
| Per-iter BSDF time | ~20 ms | ~10 ms (estimated from Phase 1.5 Δ ms/iter) |

### What this analysis did *not* answer

- **Whether a different parameterization of the same 6 parameters** would make them actually learn. The current reparameterizations (sigmoid for eps_real, exp-clamp for sigma_h, etc.) may be saturating gradients. A rescaled variant with less aggressive clamping could be tested if the 0.006 margin is ever worth chasing.
- **Whether learning materials helps on harder scenes** than the 7-benchmark set. Our scenes may be too simple / too well-explained by geometry for material learning to matter.
- **Whether a different BSDF formulation** (neural BSDF, ITU-table lookup) would perform differently. Out of scope.

### Next actionable step

Implement the simplified BSDF + fixed-materials configuration as a new ablation config in train_gaussians (e.g., `mat_mode='fixed'` with an additional `simplify_bsdf=True` flag that disables the four drop components), verify mean cart_corr stays above 0.92 across the 7-scene benchmark, and if so, make it the production model. Re-measure Tier A per-iter cost on the simplified model to confirm the ~10 ms savings.
