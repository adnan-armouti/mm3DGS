# v4 material parameter LOO investigation plan

Follow-up to [v4_material_ablation_raw_mse_random_init_results.md](v4_material_ablation_raw_mse_random_init_results.md) investigating **why** each of the 6 material parameters has the LOO behavior it does under raw MSE loss. User wants to understand the root causes before freezing anything.

## The six questions

| param | LOO Δ (random init) | LOO Δ (concrete init) | question |
|---|---|---|---|
| **A. eps_imag** | **−0.0175** | **−0.0096** | Why is it so much more important than any other parameter? |
| **B. thickness** | +0.0006 | +0.0020 | Why does freezing have no effect, despite the slab Fresnel (which uses thickness) being by far the most important BSDF component? |
| **C. eps_real** | +0.0008 | +0.0005 | Why no effect, despite being the primary dielectric constant of the Fresnel term? |
| **D. tau_base** | −0.0005 | −0.0021 | Why no effect, despite controlling the KA/SPM blend that is the whole lobe structure? |
| **E. sigma_h** | +0.0002 | −0.0030 | Why essentially no effect, despite feeding GGX α, SPM κ, and gamma blend? |
| **F. l_c** | **+0.0037** | **+0.0059** | Why does freezing *improve* cart_corr — what is the optimizer doing wrong with it? |

The parameter-to-BSDF-equation map (from the simplified BSDF after 2026-04-13 removal of CBS/directive/broad/blend):

```
Step 1 reparam: raw_materials[:, k] → (eps_real, eps_imag, sigma_h, l_c, tau_base, thickness)

Used by:
  eps_real   → itu_slab_fresnel (Jones reflection coefficient, complex permittivity real part)
               → eps_contrast → eps_factor → SPM amplitude modulation
  eps_imag   → itu_slab_fresnel (complex permittivity imaginary part, absorption)
               → eps_contrast → eps_factor → SPM amplitude modulation
  sigma_h    → enforce_spm_validity (clamped to SPM regime h_max ≈ 62 μm)
               → alpha_ggx (GGX NDF α for KA lobe)
               → roughness_slope → kappa_SPM (SPM vMF concentration)
  l_c        → enforce_spm_validity (clamped to l_c_min = λ/2 ≈ 1.95 mm)
               → l_c_lam → kappa_SPM (SPM concentration)
               → roughness_slope → kappa_SPM
  tau_base   → tau_eff (via w_KA = tau_angle × tau_base × v_KA, etc.) → KA/SPM blend
  thickness  → itu_slab_fresnel (multi-layer slab phase q = (2π/λ)·d·√(η - sin²θ))

Step 4 output: f_cos = R_jones(eps_real, eps_imag, thickness, cos_i) ×
                      [tau_eff(sigma_h, l_c, tau_base) × f_KA(sigma_h, l_c) +
                       (1 - tau_eff) × f_SPM(sigma_h, l_c, eps_contrast)] × cos_i
```

---

## Investigation phases

### Phase P1 — Per-parameter trajectory and gradient audit

For each of the 6 parameters, from the existing ablation `.npz` dumps (random + concrete init), compute:

1. **Per-iter trajectory mean** `traj_mean[t, k] = mean_m(raw_materials[t, m, k])` — does the population mean drift significantly during training?
2. **Per-iter trajectory std** `traj_std[t, k]` — does per-point diversity grow or collapse?
3. **Per-iter gradient mean** `|mean_m(grad[t, m, k])|` — is there a net pull in one direction?
4. **Per-iter gradient std** `std_m(grad[t, m, k])` — is there per-point signal?
5. **Gradient sign-cancellation** `frac(grad[t, m, k] > 0)` — how many points want to increase vs decrease?
6. **Reparameterization clamp hit rate** — for each column, what fraction of points are at the clamp boundary at end of training?

Output: 6 parameters × 6 statistics = a single diagnostic table + trajectory PNG. No new training runs needed — all data is in the existing `.npz` files.

**Expected findings**:
- (A) eps_imag: large |grad.std|, low sign-cancellation (consistent direction per-point), no clamp hits.
- (B) thickness: large |grad.std| but also large sign-cancellation (no net direction), OR trajectory mean drifts but final distribution is effectively uniform (no structural per-point signal).
- (E) sigma_h: high clamp hit rate at the `enforce_spm_validity` upper bound.
- (F) l_c: gradient direction inconsistent with what improves the loss — hard to diagnose purely from L1 alone; needs P3.

### Phase P2 — Pathway isolation (each parameter × each code path)

To answer "which code path does each parameter actually influence the loss through", run a series of **pairwise ablations**: freeze the parameter AND disable one BSDF code path at a time. If freezing + disable gives the same cart_corr as disable alone, that parameter had no effect through that pathway.

Specifically (6 params × 2 relevant paths = 12 runs × 7 scenes):

| param | pathway | experiment | interpretation |
|---|---|---|---|
| eps_imag | Jones (slab Fresnel) | freeze eps_imag + `disabled={'jones'}` | compared against `disabled={'jones'}` alone — tells us how much eps_imag does via the Jones path |
| eps_imag | SPM eps_factor | freeze eps_imag + `disabled={'spm'}` | how much via SPM |
| eps_real | Jones | freeze eps_real + `disabled={'jones'}` | |
| eps_real | SPM eps_factor | freeze eps_real + `disabled={'spm'}` | |
| thickness | Jones (slab) | freeze thickness + `disabled={'slab'}` | tells us whether thickness matters beyond the slab path (should be 0) |
| sigma_h | KA (GGX α) | freeze sigma_h + `disabled={'ka'}` | |
| sigma_h | SPM (κ_SPM) | freeze sigma_h + `disabled={'spm'}` | |
| l_c | KA | freeze l_c + `disabled={'ka'}` | |
| l_c | SPM | freeze l_c + `disabled={'spm'}` | |
| tau_base | (has only one path: KA/SPM blend) | freeze tau_base + `disabled={'ka'}` | |
| tau_base | | freeze tau_base + `disabled={'spm'}` | |
| (note: we can re-enable a 'jones' disable because we kept the Jones path) | | | |

This gives a **parameter × pathway contribution matrix**. The value in each cell = (freeze alone cart_corr) − (freeze + disable cart_corr). If that value is near zero, the parameter doesn't contribute via that pathway.

Cost: ~11 × 7 × 500 ≈ 80 min.

**Expected findings**:
- eps_imag's LOO drop will localize to either Jones or SPM, not both. If mostly Jones, the SPM eps_factor is a dead path.
- thickness's no-op behavior should show up as "zero contribution via slab" — meaning the optimizer isn't actually using thickness to improve R_jones.
- sigma_h's no-op should show up as "zero contribution via KA OR SPM" (probably the clamp).

### Phase P3 — Loss landscape curvature (parameter × parameter Hessian proxy)

LOO tells us whether *a* parameter matters given the others. To understand **redundancy between parameters**, we need the Hessian of the loss with respect to `raw_materials`. Computing the full Hessian is expensive, but we can compute:

1. **Fisher off-diagonals** `E[∂L/∂p_i · ∂L/∂p_j]` via per-iter gradient statistics. If two parameters have correlated gradients across iterations, they're redundant.
2. **Gradient covariance** at convergence: compute on the trained model `cov(grad_col_i, grad_col_j)` across iterations of a short warm-up.
3. **Hessian-vector product approximation**: `(L(θ + ε·v) − L(θ)) / ε` for specific perturbation directions `v` like `e_i` and `e_i + e_j`. 2 × 6 + 2 × (6 choose 2) = 42 forward passes, ~1 min total.

Output: a 6×6 Fisher / Hessian proxy matrix. If `eps_imag`–`thickness` is off-diagonal dominant, that explains why learning thickness is redundant with eps_imag.

**Expected findings**:
- Strong positive off-diagonal between `eps_imag` and `eps_real` (they both feed the Fresnel permittivity).
- Strong positive off-diagonal between `eps_imag` and `thickness` (they jointly determine slab Fresnel output).
- `sigma_h`–`l_c` off-diagonal, because both feed `alpha_ggx` via `alpha_raw = 4π·σ_h/λ` AND `kappa_SPM = sqrt(l_c_lam)/(1 + 3·σ_h/l_c)`.

### Phase P4 — The l_c harmfulness deep dive

This is the most mysterious finding: **freezing `l_c` at its random init value IMPROVES cart_corr** (+0.004 under random init, +0.006 under concrete init). Why would the optimizer's updates be net-negative?

Five hypotheses to test:

**H1: l_c gradient is dominated by sign-noise**. If the per-point gradient sign flips randomly across iterations, the parameter walks randomly and occasionally lands in a bad configuration. Test: run P1 on l_c specifically and look at per-iter gradient sign consistency.

**H2: l_c saturates a clamp**. The `kappa_SPM.clamp(0.5, 10.0)` clamp could be active at the learned values. When clamped, l_c's gradient disappears but its upstream updates (from other sources) still push it — in directions that don't actually affect the loss. Test: count clamp hits at convergence under "learn l_c" vs "freeze l_c".

**H3: l_c has anti-correlated contributions from KA and SPM**. `l_c` affects `alpha_ggx` (KA) one way and `kappa_SPM` (SPM) another way. If the two lobes' gradients cancel (same magnitude, opposite signs in l_c's update direction), the net update is near-zero and noise-dominated. Test: decompose the l_c gradient into its KA contribution and its SPM contribution (requires inspecting `grad` through partial forward passes).

**H4: l_c learning breaks the SPM validity constraint post-hoc**. `enforce_spm_validity` clamps `sigma_h` to a max that depends on `l_c`: `h_max_3 = 0.21 × l_c_clamped`. If l_c decreases, sigma_h gets clamped tighter, and sigma_h's gradient flows through the clamp. Test: look at what happens to sigma_h's learned distribution when l_c is frozen vs trained.

**H5: l_c is genuinely fitting noise**. Its per-point gradient has entropy but no signal — the optimizer moves it in all directions with equal magnitude, producing a random walk. Test: correlation between l_c's trajectory and the cart_corr trajectory.

Best single test to triage: **(H2+H3 combined)**. Run a short training (50 iters) logging `grad_l_c` per-point per-iter, then inspect:
- Fraction of points where `kappa_SPM` is at the [0.5, 10] clamp
- Sign of `grad_l_c` coming from KA vs from SPM (via two forward passes with one lobe disabled each)
- Time-correlation of `grad_l_c` with cart_corr changes

### Phase P5 — SPM validity clamp audit (answers E partially)

`enforce_spm_validity` clamps `sigma_h` to `min(h_max_1, h_max_2, h_max_3)` where all three depend on `l_c`, not on `sigma_h` itself. So `sigma_h`'s gradient is:
- Zero wherever `sigma_h > h_max` (clamp saturation)
- Reduced wherever `h_max` is close to `sigma_h` (partial clamp)

Under random init, σ_h ∈ [7, 370] μm — **most of this range is above the 62 μm constraint**. So a large fraction of points start with sigma_h clamped, and the gradient is zero for those points.

Check: what fraction of 50K points × 7 scenes have `sigma_h > h_max` at (a) init and (b) after 500 iters?

This likely explains E entirely: sigma_h is mostly clamped dead by SPM validity, and the random init made this worse (more points start above the clamp). The concrete init result should show fewer clamp hits.

### Phase P6 — eps_imag pathway dominance (answers A)

From Phase P2, we'll know whether eps_imag contributes via Jones or SPM or both. The likely answer: Jones dominates because:
- In R_jones, `eps_imag` determines the loss tangent (absorption) through the slab. This is the ONLY way absolute absorption depth is encoded.
- In SPM, `eps_imag` only appears as `eps_factor = (|eps_real - 1| + eps_imag)/5`, a 1-D amplitude scalar. Clamped to [0.2, 1.0]. Not much leverage.

So eps_imag wins LOO because **it's the only parameter that controls absolute absorption**, and under raw MSE absolute absorption matters (it's where most of the cart_corr signal comes from for materials). eps_real also affects Fresnel but its effect is more about phase (real part of n_complex) than magnitude. thickness affects slab phase similarly. eps_imag is the one that directly changes |r|².

To test: in Phase P2, measure `freeze_eps_imag + disable_jones` vs `disable_jones` alone. If the gap disappears, eps_imag's contribution is 100% via Jones.

---

## Expected outputs

1. **A diagnostic .md file** (`md/v4_material_parameter_loo_investigation_results.md`) containing:
   - The 6-parameter × 6-statistic table from P1
   - The parameter × pathway matrix from P2
   - The 6×6 Fisher/Hessian matrix from P3
   - The l_c deep-dive from P4
   - The SPM clamp audit from P5
   - A plain-English answer to each of the 6 questions with supporting evidence

2. **Plots** per parameter:
   - trajectory mean+std
   - gradient mean+std
   - sign cancellation
   - clamp hit rate (per iter)

3. **A recommended action per parameter**:
   - KEEP LEARNING if gradient signal is real and loss-coupled
   - FREEZE if gradient signal is zero, noise-dominated, or harmful
   - FIX (e.g. change reparam clamp, remove pathway) if the mechanism is diagnosable and fixable

4. **The experimental data**: per-run aggregate.json + selected .npz files for reproducibility.

## Compute budget

| phase | new training runs | forward passes | approx time |
|---|---|---|---|
| P1 | 0 (uses existing .npz) | 0 | ~5 min analysis |
| P2 | ~11 × 7 scenes × 500 iters | — | ~80 min |
| P3 | 0 (uses existing trained state) | ~42 | ~2 min |
| P4 | 1 × 7 scenes × 50 iters | — | ~10 min |
| P5 | 0 (uses existing .npz) | 0 | ~2 min |
| P6 | 0 (uses P2 results) | 0 | ~1 min |
| **total** | **~12 runs** | **~42** | **~100 min** |

## Execution order

1. **P1** (analysis-only, 5 min) — fastest, grounds all later analyses in the gradient + trajectory data we already have.
2. **P5** (analysis-only, 2 min) — likely explains E immediately; if so, sigma_h's question is settled.
3. **P3** (Hessian proxy, 2 min) — tells us the redundancy structure of the 6-parameter space.
4. **P2** (pathway isolation, 80 min) — the biggest compute block, needs to run in the background.
5. **P4** (l_c deep-dive, 10 min) — can run in parallel with P2 on the other GPU if convenient.
6. **P6** — uses P2 output, 1 min of interpretation.

## Out of scope

- Reparameterization changes (e.g. softer clamps, different priors). If Phase P5 says sigma_h is clamp-killed, the fix is a separate investigation.
- Alternative BSDF formulations (neural, tabulated). Not the question we're answering.
- Cross-scene transfer evaluation. Explicitly out of scope per user direction.
- Multi-seed runs for error bars. Single-seed with ±0.005 noise band is the accepted regime.
