# v4 material ablation results

Reference baseline (commit `8fe6d6d`, Tier A, full 6-param per-point): **mean cart_corr = 0.9351** across the 7-scene benchmark, ~52 s/scene, all 6 material parameters trained at LR 0.7.

All `Δ` columns below are relative to this 0.9351 baseline. All wall-clock numbers are on a single RTX 4090.

This file is updated incrementally by `mm25DGS_v4/run_material_ablation.py` as runs complete. Per-scene cart_corr, per-parameter drift, and per-parameter Fisher diagonal are stored in the per-run `.npz` files under `mm25DGS_v4/output/material_ablation/<run_name>/` and summarized into the tables below.

---

## Phase 1 — Baselines (do we need this model class?)

| run_name | description | mean_cart_corr | Δ vs baseline | ms/iter | verdict |
|---|---|---|---|---|---|
| B3_per_point_6param | Full per-point (M, 6) material model with full BSDF (Tier A baseline) | 0.9301 | -0.0050 | 113.4 | _pending_ |

**Phase 1 summary**: B0=0.9244, B1=0.9044, B2=0.9225, B3=0.9301. Per-point margin over global: +0.0076. Per-point margin over scalar: +0.0256. Per-point margin over fixed: +0.0057.

**Phase 1 decision**: STOP. Nothing being learned — debug first.

---

## LEARN matrix — 2^3 ablation on (materials, normals, patterns)

Three learnable parameter groups, all 8 corners. Two corners (M0_N1_P1, M1_N1_P1) are reused from Phase 1.

| run_name | description | mean_cart_corr | Δ vs baseline | ms/iter |
|---|---|---|---|---|
| _pending_ | | | | |

**Phase LEARN matrix summary**: _pending._

**Phase LEARN matrix decision**: _pending._

---

## Phase 1.5 — BSDF component ablation (which physics terms survive?)

_Only run if Phase 1 says per-point 6-param wins._

| run_name | disabled_component | mean_cart_corr | Δ vs baseline | ms/iter | Δ ms/iter | verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

**Phase 1.5 summary**: _pending — written after all 8 component runs complete._

**Phase 1.5 reduced model**: _pending — list of components retained._

---

## Phase 2 — Parameter ablation of the (reduced) model

_Only run if Phase 1.5 leaves a non-trivial parameter set._

### Leave-one-out (LOO)

| run_name | frozen_param | mean_cart_corr | Δ vs baseline | drift | fisher | verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

### Train-only-one (TOO)

| run_name | trained_param | mean_cart_corr | Δ vs baseline | drift | fisher | verdict |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

### Group freezes

| run_name | frozen_group | mean_cart_corr | Δ vs baseline | verdict |
|---|---|---|---|---|
| _pending_ | | | | |

### Cardinality elbow

| run_name | active_params | mean_cart_corr | Δ vs baseline | verdict |
|---|---|---|---|---|
| _pending_ | | | | |

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

**Phase 2 summary**: _pending._

---

## Phase 4 — Init sensitivity (optional)

_Only run if a Phase 2 cell shows "essential but doesn't move"._

| run_name | init | mean_cart_corr | Δ vs baseline | verdict |
|---|---|---|---|---|
| _pending_ | | | | |

---

## Identifiability cross-checks (Phase 3, automatic)

_Aggregated from per-run `.npz` dumps after all runs complete._

### Pairwise param trajectory correlation (averaged across 50K points, across all Phase 2 LOO runs)

| | eps_real | eps_imag | sigma_h | l_c | tau_base | thickness |
|---|---|---|---|---|---|---|
| eps_real | 1.00 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| eps_imag | | 1.00 | _pending_ | _pending_ | _pending_ | _pending_ |
| sigma_h | | | 1.00 | _pending_ | _pending_ | _pending_ |
| l_c | | | | 1.00 | _pending_ | _pending_ |
| tau_base | | | | | 1.00 | _pending_ |
| thickness | | | | | | 1.00 |

Pairs with `|corr| > 0.9` are flagged as degenerate.

### Converged-value histograms

_Per-parameter PNG files dumped to `mm25DGS_v4/output/material_ablation/B3_per_point/histograms/`. Filenames listed here once produced._

---

## Final recommendation

_Written after all warranted phases complete. Names the recommended material model: component set, parameter set, per-point vs global, and the supporting evidence._
