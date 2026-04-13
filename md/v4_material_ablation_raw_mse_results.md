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
| _pending_ | | | | |

**Phase 1.5 (raw MSE) summary**: _pending_

**Reduced BSDF**: _pending_

---

## Phase 2 (raw MSE) — per-parameter LOO / TOO

Runs under raw MSE on the full BSDF (not the Phase-1.5-reduced BSDF, so that per-parameter verdicts are independent of the component verdicts).

### Leave-one-out (LOO) — freeze one column, train the other 5

| run_name | frozen_param | mean_cart_corr | Δ vs 0.9324 | drift | fisher | verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

### Train-only-one (TOO) — train one column, freeze the other 5

| run_name | trained_param | mean_cart_corr | Δ vs 0.9324 | drift | fisher | verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

### LOO × TOO verdict matrix

_Filled in after all Phase 2 runs complete._

| param | LOO drop | TOO score | drift | fisher | verdict |
|---|---|---|---|---|---|
| eps_real (0) | | | | | |
| eps_imag (1) | | | | | |
| sigma_h (2) | | | | | |
| l_c (3) | | | | | |
| tau_base (4) | | | | | |
| thickness (5) | | | | | |

**Phase 2 (raw MSE) summary**: _pending_

---

## Final recommendation

_Written after all runs complete. Names: (1) which BSDF components to delete from the code path, (2) which material parameters to freeze or remove, (3) whether a different material model approach is warranted._
