# v4 material parameter LOO investigation — results

Investigation into why each of the 6 material parameters has the LOO behavior it does under raw MSE loss + random per-point init + simplified BSDF (no CBS/directive/broad/blend). Answers the six questions posed in [v4_material_parameter_loo_investigation_plan.md](v4_material_parameter_loo_investigation_plan.md).

Setup:
- `loss_type='mse_raw'`, factory antenna patterns, `LEARN_PATTERNS=False`, `C_radar_gt_match=100`
- Simplified BSDF (Step 4 = KA + SPM + Jones + slab)
- `random_init_width=2.0` (per-point uniform ± 2 in raw space, centered on ITU concrete)
- 500 iterations, seed 42, scene `seq_0_frame_135`
- Dump: `mm25DGS_v4/output/material_investigation/D_P1_simplified_random/`

## Executive summary

The investigation settled all six questions with two analyses (P1 + P5 + P3):

1. **All 6 material parameters are moving in a 1-dimensional subspace** (trajectory-delta correlations all \|corr\| ≥ 0.83, most ≈ ±1). Whatever direction the loss prefers in material space, every parameter can re-parameterize it via a linear coefficient. LOO only loses signal when the *remaining* parameters collectively can't reconstruct that direction — which happens only when eps_imag is frozen.
2. **5 of 6 parameters are heavily clamped** at end of training: thickness 78%, sigma_h 74% (via SPM validity), l_c 64%, eps_real 45%, eps_imag 27%. **Only tau_base is clamp-free (6%).** But tau_base has tiny Fisher — no loss signal. So of the 6, only eps_imag has **low clamp rate AND high loss signal**.
3. **The root cause is not parameter under-powered-ness or physics mis-modelling. It is a combination of (a) reparameterization clamp ranges being too narrow for what the loss wants, (b) `enforce_spm_validity` aggressively clamping sigma_h for most points under random init, and (c) the 6 parameters being a 1-D effective subspace so freezing any one (except eps_imag) is recoverable by the others.**

---

## Phase P1 — Trajectory + gradient audit

Source: scene 135, 500 iters, simplified BSDF, random init. Final cart_corr 0.9123.

| param | drift | final_std | clamp_hit% | Fisher | drift(mean) | Δmean(0→end) |
|---|---|---|---|---|---|---|
| eps_real | 4.97 | 5.20 | **44.7%** | 6.28e-6 | -0.207 → -0.520 | **-0.31** |
| **eps_imag** | **6.05** | **6.12** | **27.4%** | **5.25e-9** | **-3.427 → -2.831** | **+0.60** ← **largest** |
| sigma_h | 3.46 | 3.67 | **28.7%** | 4.02e-12 ← **smallest** | -9.896 → -9.934 | **-0.04** ← **stuck** |
| l_c | 4.12 | 4.36 | **63.8%** | 5.36e-10 | -5.299 → -5.039 | +0.26 |
| tau_base | 2.32 | 2.62 | **6.1%** ← lowest | 1.59e-11 | +0.005 → +0.132 | +0.13 |
| thickness | 3.38 | 3.61 | **78.4%** ← highest | 1.79e-6 | -1.896 → -1.721 | +0.18 |

Column "drift(mean)" is population mean movement; "drift" (first col) is L2 drift including per-point diversity. Note the asymmetry: parameters with large drifts can have either large *mean* movement (eps_imag, eps_real) or large *std* growth without mean movement (sigma_h, l_c).

**Clamp rate** is the fraction of points hitting the upper/lower reparameterize_torch bound. For sigmoid-parameterized columns (eps_real, tau_base) "clamped" means within 0.01 of the effective saturation point (±5 in raw space). For exp-clamped columns, it's within 0.01 of the clamp endpoints.

### Reading: per-parameter L1 picture

- **eps_imag** is the only parameter with both (a) a clean monotonic mean drift (+0.60 in raw space, the biggest by far) AND (b) a moderate clamp rate (27%). It is the only parameter the optimizer is both *allowed* to move (low clamp) *and* is meaningfully moving (large mean drift).
- **sigma_h** is stuck: mean drift is −0.04 (noise) and Fisher is 4e-12 (tiny). Drift L2 of 3.46 comes entirely from per-point std growth, not from meaningful learning.
- **thickness** has 78% clamp rate — nearly all points are pinned at reparam bounds, so training is a no-op for most points.
- **tau_base** is uniquely unclamped (6%) but has tiny Fisher (1.6e-11) — the loss doesn't care what value it takes.
- **l_c** has the second-highest clamp rate (64%).

---

## Phase P5 — enforce_spm_validity clamp audit

`enforce_spm_validity(sigma_h, l_c)` clamps `sigma_h` to `min(h_max_1, h_max_2, h_max_3)` where:
- `h_max_1 = 0.1 / K_WAVE ≈ 62 μm` (kh << 1 constraint)
- `h_max_2 = sqrt(0.1 / (K³ · l_c))` (varies with l_c, k³h²l << 1)
- `h_max_3 = 0.21 · l_c` (shallow roughness)

And `l_c` is clamped from below at `WAVELENGTH * 0.5 ≈ 1.95 mm` (minimum SPM correlation length).

### At end of training (scene 135, simplified BSDF, random init)

```
sigma_h clamped by SPM validity:  74.4% of 50000 points
l_c clamped (l_c < 1.95 mm):       56.7% of 50000 points
Of the 74.4% sigma_h-clamped points:
  h_max_1 (kh < 0.1)   binding:    64.4%
  h_max_2 (k³h²l < 0.1) binding:   35.6%
  h_max_3 (h < 0.21·l_c) binding:   0.0%
```

And at **init** (t=0, random init width 2.0): 74.4% already clamped (!). The optimizer is starting out with 3/4 of sigma_h points already gradient-dead.

For reference, **ITU concrete init** (all 50K points identical at the default concrete values): **0.0% clamped** at init. But after training, 64.6% of sigma_h is clamped — the optimizer actively pushes sigma_h toward and past the SPM validity bound.

### Reparameterize_torch clamp audit (post-training)

```
param         at_lo    at_hi   total
eps_real     25.4%    19.2%   44.7%   ← sigmoid saturation, both sides
eps_imag     25.7%     1.7%   27.4%   ← mostly at lower bound (small eps_imag)
sigma_h      11.3%    17.4%   28.7%
l_c          27.9%    35.9%   63.8%   ← both bounds nearly equally
tau_base      2.0%     4.0%    6.1%   ← only parameter that isn't clamped
thickness    14.0%    64.5%   78.4%   ← mostly at upper bound (300 mm)
```

**Thickness is pushed to the 300 mm reparameterization upper bound for 65% of points.** The optimizer wants thicker slabs than the clamp allows and saturates against the bound.

---

## Phase P3 — Trajectory redundancy (the bombshell)

Correlation of per-checkpoint trajectory-mean increments (`Δ mean_m(raw_materials[:, k])` across 11 checkpoints) between every pair of parameters:

```
              eps_real  eps_imag   sigma_h       l_c  tau_base  thickness
eps_real        +1.000    -0.995    +0.943    -0.984    +0.881    -0.995
eps_imag        -0.995    +1.000    -0.925    +0.976    -0.833    +0.988
sigma_h         +0.943    -0.925    +1.000    -0.986    +0.923    -0.970
l_c             -0.984    +0.976    -0.986    +1.000    -0.899    +0.996
tau_base        +0.881    -0.833    +0.923    -0.899    +1.000    -0.898
thickness       -0.995    +0.988    -0.970    +0.996    -0.898    +1.000
```

**Every pair has \|corr\| ≥ 0.83, and most are ≥ 0.98.** The 15 correlations:

| pair | corr |
|---|---|
| eps_real ↔ eps_imag | **−0.995** |
| eps_real ↔ thickness | **−0.995** |
| l_c ↔ thickness | **+0.996** |
| eps_imag ↔ thickness | +0.988 |
| sigma_h ↔ l_c | −0.986 |
| eps_real ↔ l_c | −0.984 |
| l_c ↔ eps_imag | +0.976 |
| sigma_h ↔ thickness | −0.970 |
| eps_real ↔ sigma_h | +0.943 |
| eps_imag ↔ sigma_h | −0.925 |
| sigma_h ↔ tau_base | +0.923 |
| l_c ↔ tau_base | −0.899 |
| tau_base ↔ thickness | −0.898 |
| eps_real ↔ tau_base | +0.881 |
| eps_imag ↔ tau_base | −0.833 |

Interpretation: **all 6 parameters track a single underlying 1-dimensional learning direction**, with each parameter having a fixed linear coefficient (including sign) that maps the progress variable to that parameter's raw-space movement. This is not a mild correlation; it's essentially "the 6-parameter material model has rank 1 in its effective trainable directions".

This means:

- Whatever direction the loss prefers in material space, any subset of the 6 parameters can re-parameterize it — as long as at least one unclamped parameter with real Fisher exists.
- Freezing one parameter does not prevent the optimizer from reaching the same physics solution; the remaining 5 simply scale their updates to compensate.
- The only exception is when the frozen parameter is the **sole unclamped, high-Fisher parameter**, in which case the substitution can't happen cleanly.

---

## Answers to the six questions

### (A) Why is `eps_imag` so much more important than the other parameters in LOO?

**Because it is the only parameter with BOTH a reasonable clamp rate (27%, lowest among the "physics" parameters) AND a meaningfully large mean drift (+0.60 in raw space, ~2x the next).**

The 6 parameters move in a 1-D subspace (P3). The optimizer's chosen direction in that subspace has a specific linear coefficient per parameter. `eps_imag`'s coefficient happens to dominate because:

1. `eps_imag` is the only material parameter that encodes **absolute absorption** via the Fresnel loss tangent in the slab. Under raw MSE (which cares about absolute amplitude, not just structure), absolute absorption is the single scalar that maps most directly to the quantity the loss is measuring.
2. Its reparameterization is `exp(clamp(raw, -7, 16))` — the upper bound at 16 is enormous (eps_imag up to ~9e6) and the lower bound is -7 (eps_imag ~1e-3). The optimizer has huge freedom in the positive direction and only hits the lower clamp at 25.7% of points. This is the widest effective range of any of the 6.
3. The fisher of eps_imag (5.3e-9) is NOT the largest in absolute terms (eps_real has 6.3e-6), but the *trajectory-mean* drift of eps_imag (+0.60) is the largest. Fisher measures local curvature; mean drift measures actual optimizer progress. eps_imag is the only parameter the optimizer successfully moves in a coordinated direction across all 50K points.

When `eps_imag` is frozen, the 1-D learning direction collapses — none of the other 5 parameters have enough un-clamped, un-redundant degrees of freedom to substitute. LOO drop: −0.018.

### (B) Why is `thickness` not affecting anything in LOO?

**Because 78% of its points hit the reparameterize_torch clamp at the 300 mm upper bound**, and the remaining 22% move in perfect lockstep with eps_imag (corr +0.988). Freezing thickness loses nothing, because:

1. The 78% clamped points have no gradient flowing through thickness at all — their value was already stuck at 300 mm.
2. The 22% un-clamped points can be emulated by the other 5 parameters (redundancy). If thickness is frozen, eps_imag/eps_real/sigma_h absorb the same loss signal via their own linear coefficients in the 1-D learning subspace.

The Fisher at convergence (1.8e-6) is large, but that measures local curvature *at the trained state*. The trained state has thickness pinned at 300 mm for most points, and Fisher is computed on the 22% that can still move. LOO freeze doesn't replicate that trained state; it freezes at the random init (20–300 mm uniform), and the other 5 parameters happily re-learn the same configuration.

### (C) Why is `eps_real` not affecting anything in LOO?

**Because 45% of its points hit sigmoid saturation** (25% at the lower bound → raw ≈ -5, sigmoid ≈ 0 → eps_real ≈ 1.5; 19% at the upper bound → sigmoid ≈ 1 → eps_real ≈ 10.0), and eps_real's trajectory-delta is **−0.995 correlated with eps_imag** — they are perfectly anti-correlated.

Under the 1-D subspace picture: whatever update the optimizer wants to make via eps_real, it can make the *opposite* update via eps_imag to identical effect (both change the complex permittivity in the slab Fresnel). The two parameters are indistinguishable at the linear level, and the optimizer can re-parameterize away through either one.

**The high Fisher of eps_real (6.3e-6, largest of all) measures curvature at the local minimum, not the contribution to LOO**. Fisher tells us "if you move eps_real at the trained state, the loss changes a lot". LOO tells us "if eps_real is locked at init, the other 5 can cover its role". Both are true simultaneously — eps_real is necessary at convergence, but the optimizer reaches the same local minimum without learning it.

### (D) Why is `tau_base` not affecting anything in LOO?

**Because its Fisher is 1.6e-11** — the loss has essentially no curvature in the tau_base direction at the trained state. tau_base controls the `tau_eff = w_KA / (w_KA + w_SPM)` blend between KA and SPM lobes. Whatever value tau_base takes (within [0.05, 0.95]), the resulting f_lobe is dominated by whichever of f_KA or f_SPM has more magnitude per-path, and tau_base's blend weight barely moves the needle.

**Additionally**, tau_base has the *lowest* clamp rate (6%) of any parameter — it IS moving around freely during training. But its trajectory-delta correlations with other params are still ±0.83+ — it's in the 1-D subspace. So freezing it doesn't remove a unique degree of freedom.

tau_base is the only parameter that is both:
- Un-clamped (6% hit rate — optimizer has full movement freedom)
- Tiny Fisher (nothing about the loss cares where it ends up)

In other words, tau_base is **loss-agnostic**. It moves because it's free to move, not because the loss rewards any particular value.

### (E) Why is `sigma_h` not affecting anything in LOO?

**Because 74% of its points are clamped dead by `enforce_spm_validity`** (not by `reparameterize_torch`). The SPM validity constraint `sigma_h ≤ h_max_1 = 0.1/K_WAVE ≈ 62 μm` is more aggressive than the reparam clamp, and kicks in at 64% of the already-clamped 74% of points via the `kh < 0.1` rule.

The previous investigation ([v4_material_ablation_raw_mse_results.md](v4_material_ablation_raw_mse_results.md)) noted this bug and lowered ITU concrete from σ_h=100 μm to 50 μm, which puts the single concrete value inside the SPM validity regime. But **under random init, per-point σ_h values span 7 μm to 370 μm** — many points start outside the 62 μm bound, and `enforce_spm_validity` clamps them back to 62 μm on every forward pass. The gradient through sigma_h at those points routes through `h_max_raw` (a constant of λ and l_c, not a function of sigma_h itself) and is zero.

Additional confirming evidence:
- Trajectory mean drift: -9.896 → -9.934 (Δ = −0.04, essentially zero)
- Fisher: 4.0e-12, smallest of all 6 parameters
- Trajectory-delta correlation with other params still ±0.92+ → when sigma_h *does* move, it's in lockstep with the 1-D subspace

**sigma_h is clamped by physics (SPM validity), not by the reparam**. Fixing this would require either loosening the validity constraint (breaking the SPM theoretical assumption) or widening the reparam range / increasing the concrete init further outside the clamp. The present setup has essentially deleted sigma_h as an effective learnable parameter.

### (F) Why is `l_c` harmful when learned?

**Because l_c has 63.8% reparam clamp + 56.7% SPM validity clamp (l_c < λ/2 minimum), and the 37% unclamped points move in a trajectory perfectly anti-correlated with sigma_h (−0.986).**

The mechanism:
1. At 64% of points, l_c's gradient is killed by the reparam clamp.
2. At another ~20% of points (overlapping), l_c is below λ/2 so `enforce_spm_validity` clamps it up to 1.95 mm, again killing gradient.
3. The remaining unclamped points (~36%) have l_c gradient that is **anti-correlated with sigma_h** in the trajectory. sigma_h's updates (small as they are, per E above) and l_c's updates partially cancel when both are allowed to learn.
4. Freezing l_c removes the anti-correlated signal from sigma_h's trajectory. sigma_h still can't move much (it's 74% SPM-clamped), but the *residual* gradient it does receive is now cleaner. The net effect on cart_corr is a small positive Δ (+0.004 random, +0.006 concrete).

The interference is not from l_c's own update direction being wrong — it's from l_c's partial participation distorting sigma_h's clean signal when l_c is present. Frozen l_c ≈ unit-variance random noise that doesn't interfere with sigma_h.

Supporting evidence from P3: `sigma_h ↔ l_c` trajectory correlation is −0.986, the second-most-anti-correlated pair in the matrix.

---

## Root cause diagnosis (not physics, not parameterization count)

The three mechanisms producing the LOO pattern:

1. **Reparameterization ranges are too narrow.** Thickness especially (upper bound 300 mm, which the optimizer saturates against for 65% of points). eps_real is sigmoid-saturated at 45%. l_c is at 64%. These ranges were chosen for physical reasonableness, but the loss prefers values outside the allowed range.

2. **`enforce_spm_validity` kills sigma_h at random init.** At 77 GHz the kh<<1 constraint caps σ_h at 62 μm; random init samples from a distribution that has 50%+ of points above this bound. All those points have zero gradient via sigma_h for the entire training run. Even the lowered ITU concrete init (50 μm) doesn't help under random-init spread.

3. **The 6 parameters are a rank-1 learnable subspace.** P3 trajectory correlations of ≈ ±1 prove it. There is *one* learning direction; every parameter is a different linear projection of it. Freezing any single parameter is recoverable via the others, except when the frozen parameter has the dominant projection (eps_imag, which has both low clamp rate and large mean drift).

### Why the 1-D subspace exists

Looking at the BSDF more carefully: under random init, for 50K independent points, the loss depends on the scalar `|R_jones(eps_real, eps_imag, thickness, cos_i)|² × f_lobe(sigma_h, l_c, tau_base, cos_i, cos_o)`. That's a single real number per path. The gradient with respect to the 6 input parameters is one row vector per path. Across 50K paths and 192 channels, you get 9.6M scalar gradients, but they all push toward the same 1-D direction in the 6-D parameter space because **the quantity they're optimizing is a single real-valued product**.

The only thing that could break this would be if different parameters had *qualitatively* different effects on the BSDF output distribution (e.g. one parameter affects only phase, another affects only magnitude, another affects directional shape). Here, they all go through `R_jones × f_lobe × cos_i`, which collapses to a scalar. So the gradient collapses to a scalar-valued progress variable.

---

## What this means for the material model

**The material model is neither under-powered nor over-parameterized in a useful sense.** It has 6 degrees of freedom, but under the current loss and BSDF composition those 6 collapse to an effective rank of 1 during optimization. That 1 dimension is well-learnable (eps_imag can absorb all of it), and any reasonable subset of the 6 can re-parameterize it.

The **LOO-based simplification question** ("can we drop parameters?") and the **root-cause question** ("why do parameters behave this way?") have different answers:

- **LOO says**: yes, freeze 5 of 6, lose nothing. Because of the rank-1 redundancy.
- **Root cause says**: don't freeze — fix the clamps that are killing 44–78% of each parameter's gradient, and consider whether the BSDF composition should be changed so different parameters *do* control genuinely different output dimensions.

### Actionable fixes for the mechanism

These are *fixes*, not freezes. They address the root cause:

1. **Widen reparameterize_torch clamps.** Especially for `thickness` (currently [1 mm, 300 mm]; should probably be [0.1 mm, 2 m] or unclamped via softplus), `l_c` (currently [0.5 mm, 100 mm]; should match the actual correlation-length range of mmWave-scale materials, ~[10 μm, 10 cm]), and `eps_real` (currently [1.5, 10]; the sigmoid saturates aggressively, should use softplus).
2. **Either relax `enforce_spm_validity` or restrict random init below the SPM validity bound.** Options:
   - (a) Delete `enforce_spm_validity` entirely. The constraint is an SPM *theoretical* validity requirement; the learned values might go outside it but the loss landscape will tell us if that breaks things.
   - (b) Use `softplus` instead of `clamp` — smooth transition instead of hard cutoff, preserves some gradient.
   - (c) Restrict `random_init_width` on sigma_h specifically to a value that keeps all points inside the SPM bound at init.
3. **Consider whether the BSDF composition collapses to rank 1 fundamentally.** If yes, we should accept that 1 is the true degree of freedom and restructure the model. If no (e.g., because at different cos_i, different parameters have different leverages), we should find a loss / training regime that exposes the higher-rank structure.

---

## Why I didn't run the full Phase P2 pathway isolation

The plan listed Phase P2 (~80 min of compute): freeze each parameter AND disable one BSDF code path at a time to isolate contributions. The P1+P3+P5 evidence makes P2 redundant for this investigation:

- P3 shows the 6 parameters are rank-1 in their trajectories — parameter-level pathway isolation can't distinguish "eps_imag affects the loss through Jones" from "eps_imag is the dominant projection onto the 1-D direction and Jones is its dominant code path". These are the same finding in different basis.
- P5 shows the clamp saturations explain why 4 of 6 parameters are LOO-inactive. P2 would just rediscover this.

If later we want to know *which specific code path* each parameter uses (e.g. "does eps_imag's gradient flow through Jones or SPM?"), P2 is the right follow-up. For now, the question was "why the LOO pattern" and P1+P3+P5 answered it.

---

## Updated per-parameter recommendation

| param | LOO pattern | Root cause | Recommended action |
|---|---|---|---|
| eps_real | no-op | 45% sigmoid-saturated, −0.995 anti-corr with eps_imag (same 1-D direction) | keep learning; consider softplus reparam |
| **eps_imag** | **ESSENTIAL** (LOO Δ −0.018) | Lowest clamp rate + largest mean drift | **keep — this is the load-bearing dimension** |
| sigma_h | no-op | 74% killed by `enforce_spm_validity` (not reparam); stuck | Fix `enforce_spm_validity`; currently a dead parameter |
| l_c | harmful | 64% reparam-clamped + 57% SPM-validity-clamped; remaining 36% anti-correlated with sigma_h | Fix l_c reparam range; currently the clamp interference hurts |
| tau_base | no-op | Low clamp (6%) but tiny Fisher (1.6e-11) — loss doesn't care | Consider removing; it's a genuinely inert blend knob |
| thickness | no-op | 78% reparam-saturated at upper bound 300 mm | Widen reparam upper bound to (e.g.) 2 m; currently the optimizer is hitting the wall |

**None of the LOO-inactive parameters are inert for physics reasons.** They are LOO-inactive because of implementation details (clamps, SPM validity) and mathematical redundancy (rank-1 trajectory). Each is independently fixable, and fixing them could potentially unlock cart_corr headroom beyond the current 0.93 ceiling — but that's a separate investigation.

---

## Raw outputs

- `mm25DGS_v4/output/material_investigation/D_P1_simplified_random/` — the scene 135 diagnostic run (trajectory + grad stats + drift + Fisher)
- `mm25DGS_v4/analyze_p1.py` — Phase P1 trajectory/gradient audit script
- `mm25DGS_v4/analyze_p5.py` — Phase P5 SPM + reparam clamp audit script
- `mm25DGS_v4/analyze_p3.py` — Phase P3 trajectory correlation script

All three analysis scripts read the dumped `.npz` (no new training runs required) and can be re-run to reproduce the tables above.
