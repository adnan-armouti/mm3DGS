# v4 material ablation under raw MSE — random initialization

Follow-up to [v4_material_ablation_raw_mse_results.md](v4_material_ablation_raw_mse_results.md): re-runs the full Phase 1.5 + Phase 2 battery under **random per-point material initialization** instead of the uniform ITU concrete init used before. The question this answers: **are the concrete-init findings robust to starting point, or are they artifacts of starting all 50K points at the same material vector?**

## Setup

- **Initialization**: `raw_materials[m, k] = concrete_raw[k] + U(-2, +2)` per-point per-column. Seed 42.
- **In physics space**, each point starts with a unique combination of:
  - `eps_real` ∈ ~[2.3, 8.8] (concrete = 5.31)
  - `eps_imag` ∈ ~[0.004, 0.24] (concrete = 0.033)
  - `sigma_h` ∈ ~[7 μm, 370 μm] (concrete = 50 μm, subject to SPM validity clamp)
  - `l_c` ∈ ~[0.7 mm, 37 mm] (concrete = 5 mm)
  - `tau_base` ∈ ~[0.11, 0.89] (concrete = 0.5)
  - `thickness` ∈ ~[20 mm, 300 mm] (concrete = 150 mm, upper clamp at 300 mm)
- Every other setting matches the concrete-init ablation: `loss_type='mse_raw'`, factory patterns, `LEARN_PATTERNS=False`, `C_radar_gt_match=100`, 500 iters × 7 scenes, single seed, materials + normals trained.
- **Random-init baseline**: `M1_N1` (full 6-param, both knobs learned) = **0.9291** across 7 scenes. Compare to concrete-init baseline 0.9324 (Δ = −0.003, within noise).

One process died mid-ablation without exit code around `LOO_freeze_sigma_h`; a resume script completed the remaining 10 runs from where it stopped. All 21 runs now have aggregate.json files.

---

## Phase 1.5 — BSDF component ablation (random init)

| run_name | disabled | mean_cart_corr | Δ vs 0.9291 | verdict |
|---|---|---|---|---|
| K1_no_cbs | cbs | 0.9283 | −0.0008 | DROP (within noise) |
| K2_no_directive | directive | 0.9284 | −0.0007 | DROP (within noise) |
| K3_no_broad | broad | 0.9265 | −0.0026 | DROP (within noise) |
| K6_no_blend | blend | 0.9269 | −0.0022 | DROP (within noise) |
| K7_no_jones | jones | 0.9231 | −0.0060 | borderline (slightly weaker than concrete) |
| K4_no_spm | spm | 0.9185 | −0.0106 | borderline |
| K5_no_ka | ka | 0.9135 | −0.0156 | borderline |
| **K8_no_slab** | slab | **0.7271** | **−0.2020** | **KEEP** |

### Head-to-head with concrete init

| component | concrete-init Δ | random-init Δ | verdict change |
|---|---|---|---|
| cbs | −0.0002 | −0.0008 | **none** — still DROP |
| directive | −0.0010 | −0.0007 | **none** — still DROP |
| broad | −0.0029 | −0.0026 | **none** — still DROP |
| blend | −0.0026 | −0.0022 | **none** — still DROP |
| jones | −0.0046 | −0.0060 | DROP → marginal, but still within the "borderline" band |
| spm | −0.0065 | −0.0106 | borderline both, slightly stronger under random |
| ka | −0.0188 | −0.0156 | borderline both, slightly weaker under random |
| **slab** | **−0.4022** | **−0.2020** | **KEEP both, dominance halved** |

**Phase 1.5 verdicts are stable under random init.** All 5 "DROP" components stay droppable, both "borderline" components (SPM, KA) stay borderline, and the slab stays essential.

**Slab's dominance is halved** under random init (−0.20 vs −0.40) but still 2× any other component. Likely reason: with per-point random thicknesses ranging from 20 mm to 300 mm, some points happen to have a thickness that crudely approximates the correct multi-layer interference even when the slab component is disabled; under concrete init, all points share the same 150 mm thickness so there's no such backup.

---

## Phase 2 — per-parameter LOO / TOO (random init)

### Leave-one-out (LOO) — random init vs concrete init

| param | concrete Δ | random Δ | both verdict |
|---|---|---|---|
| **eps_imag** | **−0.0096** | **−0.0175** | **ESSENTIAL (more load-bearing under random init)** |
| eps_real | +0.0005 | +0.0008 | INERT |
| sigma_h | −0.0030 | +0.0002 | INERT (slightly more inert under random) |
| **l_c** | **+0.0059** | **+0.0037** | **HARMFUL when learned — both inits** |
| tau_base | −0.0021 | −0.0005 | INERT |
| thickness | +0.0020 | +0.0006 | INERT |

`eps_imag` is the only parameter with a meaningful negative LOO drop under either init. Freezing it costs **nearly 2× more under random init** (−0.018 vs −0.010) — the loss is more sensitive to eps_imag when the per-point material distribution is diverse, because that's exactly the regime where the optimizer needs an amplitude-controlling degree of freedom to fit varied points.

**`l_c` is still harmful when learned** under both inits. The sign of the LOO drop is positive for l_c in both columns — freezing l_c is a net improvement, by +0.006 (concrete) or +0.004 (random). The counterintuitive finding replicates.

### Train-only-one (TOO) — random init vs concrete init

| param | concrete TOO | random TOO | rank under random |
|---|---|---|---|
| **eps_imag** | **0.9313** | **0.9271** | **1 (same)** |
| thickness | 0.9138 | 0.9058 | 2 (same) |
| eps_real | 0.9129 | 0.9087 | 3 (same) |
| tau_base | 0.9093 | 0.9019 | 4 (same) |
| sigma_h | 0.9070 | 0.8978 | 5 (same) |
| l_c | 0.8936 | 0.8911 | 6 (same) |

**The TOO ranking is IDENTICAL under both initializations.** Not just the winner (eps_imag) but the entire ordering of all 6 parameters is preserved: eps_imag > thickness > eps_real > tau_base > sigma_h > l_c.

### Headline: TOO_only_eps_imag is still within noise of the full 6-param model

- Under **concrete** init: TOO_only_eps_imag = **0.9313**, full 6-param = 0.9324, Δ = **−0.0011**
- Under **random** init: TOO_only_eps_imag = **0.9271**, full 6-param = 0.9291, Δ = **−0.0020**

Training **only `eps_imag` and freezing the other 5 parameters at per-point random values** (which are *not* physically correct for any actual material) gets within 0.002 of the full 6-parameter model under random init. The same pattern holds under concrete init (within 0.001). **The single-parameter material model is sufficient regardless of init.**

---

## LOO × TOO verdict matrix (random init)

Sort by TOO score (descending — most informative single parameter first).

| param | LOO Δ vs 0.9291 | TOO score | TOO Δ | verdict |
|---|---|---|---|---|
| **eps_imag (1)** | **−0.0175** | **0.9271** | **−0.0020** | **ESSENTIAL — the single load-bearing parameter** |
| thickness (5) | +0.0006 | 0.9058 | −0.0233 | INERT in combo, OK alone |
| eps_real (0) | +0.0008 | 0.9087 | −0.0204 | INERT in combo |
| tau_base (4) | −0.0005 | 0.9019 | −0.0272 | INERT in combo |
| sigma_h (2) | +0.0002 | 0.8978 | −0.0313 | INERT in combo |
| **l_c (3)** | **+0.0037** | 0.8911 | −0.0380 | **HARMFUL when learned** |

Every verdict matches the concrete-init result.

---

## Robustness conclusions

### What's robust to random init

1. **`eps_imag` is the single essential material parameter.** Ranked #1 under both inits by TOO, has the largest LOO drop under both, and gets within noise of the full 6-param baseline when alone. Under random init it's *even more* load-bearing (LOO −0.018 vs −0.010).

2. **Five of six material parameters are redundant or harmful.** Same five under both inits: `eps_real`, `sigma_h`, `l_c`, `tau_base`, `thickness`. `l_c` is specifically harmful (freezing it *improves* cart_corr under both inits).

3. **TOO ranking is preserved exactly.** eps_imag > thickness > eps_real > tau_base > sigma_h > l_c under both initializations, without a single rank inversion.

4. **Slab multi-layer Fresnel is the essential BSDF component.** Its dominance is halved under random init (−0.20 vs −0.40) but still 2× any other component and the only "KEEP" verdict.

5. **The 5 droppable BSDF components stay droppable.** CBS, directive, broad, blend, and Jones are all within the ±0.006 noise band under both inits.

6. **Borderline components (SPM, KA) stay borderline.** Small verdict shuffling (SPM goes from −0.007 → −0.011, KA goes from −0.019 → −0.016) but neither crosses a decision boundary.

### What shifts under random init

- **Slab's dominance** halves (−0.40 → −0.20) because per-point random thicknesses give the optimizer a crude backup for multi-layer interference even when the slab code path is disabled.
- **`eps_imag`'s importance** nearly doubles (−0.010 → −0.018) because the loss needs more amplitude-fitting capacity when per-point materials are diverse at init.
- **Absolute cart_corr values** drop ~0.004–0.01 across the board under random init. Not surprising — starting closer to the ITU concrete sweet spot is a mild advantage for convergence.

### What does NOT shift under random init

- The set of droppable components
- The set of inert parameters
- The identity of the single essential parameter
- The TOO ranking of all 6 parameters
- The `l_c` is harmful finding
- The overall conclusion that a 1-parameter material model suffices

---

## Final recommendation (unchanged)

The simplification recommended in [v4_material_ablation_raw_mse_results.md](v4_material_ablation_raw_mse_results.md) is **robust to material initialization**:

1. **Delete** from `render_factorized` Step 4: CBS sinc, directive vMF lobe, broad diffuse, coherence blend, Jones polarization (replaced with scalar `|r|²`).
2. **Keep**: Multi-layer slab Fresnel (essential under both inits), KA lobe + SPM lobe (borderline under both inits — keep for now, revisit after reduction).
3. **Reduce `raw_materials` from `(M, 6)` to `(M, 1)`**: learn only `eps_imag`. Freeze `eps_real`, `sigma_h`, `l_c`, `tau_base`, `thickness` at ITU concrete.

Expected performance under either init: within 0.002 of the full 6-param model. Material parameter count drops from 300K → 50K (6× reduction).

### Additional insight from the random-init test

The fact that `TOO_only_eps_imag` reaches 0.9271 when the other 5 parameters are **frozen at randomly chosen per-point values** (not at ITU concrete) is stronger evidence than the concrete-init test: the optimizer doesn't need correct initial values for 5 of the 6 parameters. It can recover essentially full cart_corr using only `eps_imag` regardless of what the other parameters are frozen at, as long as they're in a reasonable physics range. This strengthens the "over-parameterized, not under-powered" conclusion from the previous investigation.

---

## Raw data

Per-run aggregates in `mm25DGS_v4/output/material_ablation_raw_mse_random_init/<run_name>/aggregate.json`.

Drivers:
- `mm25DGS_v4/run_raw_mse_ablation_random_init.py` (full 21-run battery)
- `mm25DGS_v4/resume_random_init_ablation.py` (resume-aware, skips runs with existing aggregate.json)

The random init is triggered via `train_gaussians(random_init_width=2.0, random_init_seed=42)`. Setting `random_init_width=0` (default) gives the ITU concrete init used in the previous ablation.
