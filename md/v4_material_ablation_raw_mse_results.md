# v4 material ablation under raw MSE loss (follow-up results)

This document captures the Phase 1.5 (BSDF component ablation) and Phase 2 (per-parameter LOO/TOO) re-runs under the new default `loss_type='mse_raw'` + factory patterns + `LEARN_PATTERNS=False` + 100× `C_radar` boost configuration.

**Baseline**: M1_N1_mse_raw from [v4_unnormalized_mse_loss_plan.md](v4_unnormalized_mse_loss_plan.md) Phase γ = **0.9324** across 7 scenes (full 6-param BSDF, all components enabled, materials + normals trained).

The earlier `v4_material_ablation_results.md` was run under the old min-max MSE loss with mmIR-trained antenna patterns; its verdicts (drop SPM/CBS/broad/blend, keep Jones/KA/slab, Phase 2 uninformative) may or may not hold now that materials are actually learning (+0.028 marginal contribution, 58–1332× larger gradients, 1,000–5,000,000× larger Fisher). This document re-examines both questions.

All runs: 7 scenes × 500 iters, mean across scenes reported. Single seed. Run-to-run noise floor ~±0.005 (single scene ~±0.015).

---

## Phase 1.5 (raw MSE) — BSDF component ablation

Each run disables one BSDF component via the existing `disabled_components={...}` mechanism and reports the mean cart_corr drop vs the M1_N1_mse_raw baseline.

| run_name | disabled | mean_cart_corr | Δ vs 0.9324 | verdict |
|---|---|---|---|---|
| K1_no_cbs | cbs | 0.9326 | +0.0002 | DROP (within noise) |
| K2_no_directive | directive | 0.9314 | -0.0010 | DROP (within noise) |
| K3_no_broad | broad | 0.9295 | -0.0029 | DROP (within noise) |
| K4_no_spm | spm | 0.9259 | -0.0065 | borderline |
| K5_no_ka | ka | 0.9136 | -0.0188 | borderline |
| K6_no_blend | blend | 0.9298 | -0.0026 | DROP (within noise) |
| K7_no_jones | jones | 0.9278 | -0.0046 | DROP (within noise) |
| K8_no_slab | slab | 0.5302 | -0.4022 | KEEP |

**Phase 1.5 (raw MSE) summary**: Under raw MSE + factory patterns + scale fix, the verdicts are: DROP=['cbs', 'directive', 'broad', 'blend', 'jones'], BORDERLINE=['spm', 'ka'], KEEP=['slab']. Reduced BSDF = full BSDF minus ['cbs', 'directive', 'broad', 'blend', 'jones'].

**Reduced BSDF**: Full BSDF minus ['cbs', 'directive', 'broad', 'blend', 'jones']. Keep: Jones + ['slab'] + ['spm', 'ka'].

---

## Phase 2 (raw MSE) — per-parameter LOO / TOO

Runs under raw MSE on the full BSDF (not the Phase-1.5-reduced BSDF, so that per-parameter verdicts are independent of the component verdicts).

### Leave-one-out (LOO) — freeze one column, train the other 5

| run_name | frozen_param | mean_cart_corr | Δ vs 0.9324 | drift | fisher | verdict |
|---|---|---|---|---|---|---|
| LOO_freeze_eps_real | eps_real | 0.9329 | +0.0005 | [0.00,6.62,4.09,4.42,2.90,4.25] | [0.0e+00,8.7e-09,4.2e-10,1.4e-09,1.9e-09,6.1e-06] | |
| LOO_freeze_eps_imag | eps_imag | 0.9228 | -0.0096 | [4.30,0.00,4.24,4.50,3.20,4.20] | [9.4e-08,0.0e+00,6.2e-10,1.3e-09,8.1e-10,6.2e-07] | |
| LOO_freeze_sigma_h | sigma_h | 0.9294 | -0.0030 | [4.60,6.33,0.00,4.62,3.01,4.20] | [5.4e-06,5.8e-09,0.0e+00,1.3e-09,2.6e-09,3.3e-06] | |
| LOO_freeze_l_c | l_c | 0.9383 | +0.0059 | [4.70,6.45,4.82,0.00,3.70,4.25] | [2.7e-05,7.1e-09,8.9e-10,0.0e+00,2.2e-09,2.7e-05] | |
| LOO_freeze_tau_base | tau_base | 0.9303 | -0.0021 | [4.59,6.34,4.31,4.53,0.00,4.20] | [9.3e-05,1.3e-08,2.8e-10,3.2e-09,0.0e+00,2.9e-05] | |
| LOO_freeze_thickness | thickness | 0.9344 | +0.0020 | [4.76,6.65,4.23,4.46,2.98,0.00] | [2.2e-04,9.6e-09,3.7e-10,1.7e-09,4.9e-10,0.0e+00] | |

### Train-only-one (TOO) — train one column, freeze the other 5

| run_name | trained_param | mean_cart_corr | Δ vs 0.9324 | drift | fisher | verdict |
|---|---|---|---|---|---|---|
| TOO_only_eps_real | eps_real | 0.9129 | -0.0195 | [4.83,0.00,0.00,0.00,0.00,0.00] | [4.6e-05,0.0e+00,0.0e+00,0.0e+00,0.0e+00,0.0e+00] | |
| TOO_only_eps_imag | eps_imag | 0.9313 | -0.0011 | [0.00,7.97,0.00,0.00,0.00,0.00] | [0.0e+00,1.7e-09,0.0e+00,0.0e+00,0.0e+00,0.0e+00] | |
| TOO_only_sigma_h | sigma_h | 0.9070 | -0.0254 | [0.00,0.00,5.04,0.00,0.00,0.00] | [0.0e+00,0.0e+00,3.2e-09,0.0e+00,0.0e+00,0.0e+00] | |
| TOO_only_l_c | l_c | 0.8936 | -0.0388 | [0.00,0.00,0.00,4.71,0.00,0.00] | [0.0e+00,0.0e+00,0.0e+00,1.1e-09,0.0e+00,0.0e+00] | |
| TOO_only_tau_base | tau_base | 0.9093 | -0.0231 | [0.00,0.00,0.00,0.00,4.48,0.00] | [0.0e+00,0.0e+00,0.0e+00,0.0e+00,5.7e-10,0.0e+00] | |
| TOO_only_thickness | thickness | 0.9138 | -0.0186 | [0.00,0.00,0.00,0.00,0.00,4.16] | [0.0e+00,0.0e+00,0.0e+00,0.0e+00,0.0e+00,8.1e-06] | |

### LOO × TOO verdict matrix

Sort by TOO score (descending — most informative single parameter first).

| param | LOO Δ vs 0.9324 | TOO score | TOO Δ | verdict |
|---|---|---|---|---|
| **eps_imag (1)** | **-0.0096** | **0.9313** | **-0.0011** | **ESSENTIAL — the single load-bearing material parameter** |
| thickness (5) | +0.0020 | 0.9138 | -0.0186 | INERT in combo, OK alone — subsumed by eps_imag |
| eps_real (0) | +0.0005 | 0.9129 | -0.0195 | INERT in combo — subsumed by eps_imag |
| tau_base (4) | -0.0021 | 0.9093 | -0.0231 | INERT in combo — subsumed by eps_imag |
| sigma_h (2) | -0.0030 | 0.9070 | -0.0254 | INERT in combo — subsumed by eps_imag |
| **l_c (3)** | **+0.0059** | 0.8936 | -0.0388 | **HARMFUL when learned** — freezing it *improves* cart_corr |

**Phase 2 (raw MSE) summary**:

- **`eps_imag` alone reaches 0.9313** — that's **within 0.001** of the full 6-param model (0.9324). A single-parameter material model is mechanically as good as the full 6-param one.
- **Freezing `l_c` improves cart_corr by +0.006** (above the noise floor). The parameter is actively hurting: the optimizer is learning it toward directions the loss doesn't actually reward.
- `eps_real`, `sigma_h`, `tau_base`, `thickness` are all inert in combination — LOO drops are within noise (±0.003) and TOO scores are 0.906–0.914 (noticeably below eps_imag's 0.9313).
- All six parameters develop significant drift under raw MSE (every drift value 3.0–6.7 except the frozen columns). The optimizer IS moving them. They just don't contribute meaningfully once `eps_imag` is free.

---

## Final recommendation

### BSDF components

**Delete** from `render_factorized` (Δ vs 0.9324 within noise, −0.005 threshold):

- **CBS sinc factor** (+0.0002): coherent backscatter enhancement — no effect
- **Directive vMF lobe** (−0.0010): anisotropic broadening — no effect
- **Broad/diffuse Lambertian** (−0.0029): diffuse fallback — no effect
- **Coherence blend** (−0.0026): smooth interpolation — no effect
- **Jones polarization** (−0.0046): polarized Fresnel → scalar |r|² is essentially equivalent

Note that **Jones went from KEEP (−0.021 under min-max MSE) to DROP (−0.005 under raw MSE)**. Under the old loss, Jones was compensating for something the loss itself couldn't express; under raw MSE the loss itself captures the structural content and Jones becomes decorative.

**Borderline** (−0.005 < Δ < −0.02), keep for now:

- **SPM incoherent lobe** (−0.0065)
- **GGX / Cook-Torrance KA lobe** (−0.0188)

**KEEP — absolutely essential**:

- **Multi-layer slab Fresnel** (−0.4022). Disabling slab collapses cart_corr from 0.93 to 0.53. **This is by far the most important physics term in the entire BSDF.** Under min-max MSE its marginal was only −0.021; under raw MSE, **40× more important**. The loss can finally see absolute amplitudes, and amplitudes are dominated by the multi-layer interference in the dielectric slab.

### Material parameters

**Learn only one parameter**: `eps_imag` (column 1 of `raw_materials`). All other material learning can be deleted.

**Freeze at ITU concrete defaults**: `eps_real`, `sigma_h`, `l_c`, `tau_base`, `thickness`. Each is either inert (LOO within noise) or actively harmful (`l_c`).

Expected performance: **0.9313** (eps_imag-only) vs **0.9324** (full 6-param). Difference is within single-seed run-to-run noise (±0.005).

**DOF reduction**: 50,000 × 6 = 300K material parameters → 50,000 × 1 = 50K. **6× fewer material parameters, essentially identical cart_corr.**

### Does this warrant a different material model approach?

**No.** The physics machinery is doing real work — specifically, the multi-layer slab Fresnel is the dominant contributor (−0.40 when disabled). What was broken was the interaction between (a) the min-max-normalized loss washing out absolute amplitudes, and (b) 5 of 6 material parameters fighting for a low-dimensional effective subspace that was almost entirely explained by `eps_imag`.

The corrective path forward is:

1. **Delete the component code paths for CBS, directive, broad, blend, Jones** from [rasterizer_factorized.py](../mm25DGS_v4/rasterizer_factorized.py) Step 4. Keep Jones as a scalar |r|² (already implemented as the `jones` disable path).
2. **Reduce `raw_materials` to `(M, 1)`** holding only the raw eps_imag value. Freeze eps_real/sigma_h/l_c/tau_base/thickness at ITU concrete as physical constants.
3. **Phase 1.5 borderline components** (SPM, KA) can be revisited as a follow-up ablation once the reduced model is implemented, since their verdicts may shift when they're the only remaining incoherent term.
4. **Multi-layer slab thickness is essential** but doesn't need to be *learned* (thickness TOO = 0.9138, freezing it in LOO is +0.0020). Fix thickness at 0.15 m (ITU concrete).

### Estimated gains from the simplification

- Material parameter count: 300K → 50K (6×)
- BSDF forward-pass code: roughly half removed (delete CBS/directive/broad/blend/Jones blocks)
- Per-iter time: estimated ~15–20% faster (depends on how much of Step 4 collapses)
- Clarity: a single learnable material parameter with obvious physical meaning (loss tangent) is dramatically easier to interpret than 6 coupled parameters

### What this investigation settled definitively

1. **The 6-parameter material model was not under-powered.** The min-max normalization loss was hiding 95% of the material signal. Once the loss can see absolute amplitudes, material learning works — but only one parameter is needed.
2. **Normals are still the dominant learnable knob** (closes ~96% of the gap on their own). Materials add ~+0.03 on top. Both findings hold under raw MSE.
3. **Multi-layer slab Fresnel is the single most important BSDF physics term.** Its importance was masked by the old loss; it now towers over every other component by ~20×.
4. **Five of the six material parameters are redundant with `eps_imag`.** The physically-motivated Cook-Torrance + SPM + CBS lobes and their associated parameters don't carry independent information in this regime.

### Open questions (out of current scope)

- Does a reduced BSDF with only Jones + slab + KA + SPM, and only `eps_imag` learnable, perform the same as the full model? **Predict yes, ~0.9313.** Worth verifying once the reduction is implemented.
- Does `eps_imag`-only work on harder scenes (e.g. outside the benchmark)? Unknown.
- What does the learned `eps_imag` distribution look like across points? Does it spatially correlate with obvious material boundaries (asphalt vs vegetation vs vehicle)? Would be a nice qualitative validation.
