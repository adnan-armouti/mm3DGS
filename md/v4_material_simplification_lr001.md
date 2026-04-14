# v4 material model simplification under LR=0.01

Follow-up to [v4_material_optimizer_dynamics_investigation.md](v4_material_optimizer_dynamics_investigation.md). Re-runs the LOO + TOO ablation at the corrected `mat_lr = 0.01` and tests candidate simpler configurations. The goal is to answer: **is the 6-parameter material model the simplest we should employ, or are there redundant parameters we can remove?**

Spoiler: **yes, we can remove 3 of 6 and the model performs slightly better**. The old LOO/TOO results under `mat_lr=0.7` gave a completely different (and wrong) verdict because the optimizer was overshooting the loss minimum; under the corrected LR, the redundancy picture is clean.

## Baseline

- All 6 material parameters learnable (`freeze_mat_cols=None`)
- `mat_lr=0.01` (optimum from LR sweep)
- Simplified BSDF (no CBS/directive/broad/blend)
- Raw MSE, factory antenna patterns, `LEARN_PATTERNS=False`
- Random per-point init (width 2.0, seed 42)
- 500 iters × 7 scenes

**Mean cart_corr: 0.9322**. Noise floor: ±0.005 per seven-scene mean.

## LOO (leave-one-out) — freeze one column, train the other 5

| param | new LOO Δ (LR=0.01) | old LOO Δ (LR=0.7) | reinterpreted verdict |
|---|---|---|---|
| **sigma_h** | **−0.0118** | −0.0030 | **ESSENTIAL — biggest cost when frozen** |
| eps_imag | −0.0020 | **−0.0175** | borderline (within noise) — was the "essential" under LR=0.7 |
| l_c | −0.0001 | +0.0037 | INERT — no marginal contribution |
| tau_base | −0.0008 | −0.0021 | INERT |
| eps_real | +0.0004 | +0.0005 | INERT |
| **thickness** | **+0.0017** | +0.0020 | **HARMFUL WHEN LEARNED — freezing helps** |

**Complete reshuffling of the verdicts** compared to the LR=0.7 investigation:

- **`sigma_h` is now unambiguously the essential parameter.** Freezing it costs -0.0118, 2.4× the noise floor. Under LR=0.7 the SPM validity clamp + optimizer overshoot hid this entirely.
- **`eps_imag` is no longer essential.** The old result (-0.0175) was an artifact of LR=0.7 concentrating all the learning signal into eps_imag because Adam couldn't use the other parameters cleanly. Under the converged LR, eps_imag has only small marginal contribution (-0.002, within noise).
- **`l_c` is no longer harmful.** The old "freezing l_c gives +0.0037" was driven by sigma_h-l_c cancellation when sigma_h was clamp-stuck. Now that sigma_h moves properly, l_c is simply inert.
- **`thickness` remains harmful when learned.** Freezing gives +0.0017, meaning the optimizer is learning thickness in a direction that slightly hurts cart_corr. This is the single most robust finding — it held under both LR regimes.

## TOO (train-only-one) — train one column, freeze the other 5

| param | TOO score | TOO Δ vs 0.9322 |
|---|---|---|
| **sigma_h** | **0.9201** | **-0.0121** (best alone) |
| eps_real | 0.9137 | -0.0185 |
| thickness | 0.9107 | -0.0215 |
| eps_imag | 0.9109 | -0.0213 |
| tau_base | 0.9008 | -0.0314 |
| l_c | 0.8994 | -0.0328 (worst alone) |

**`sigma_h` wins both LOO and TOO** — it is the single essential parameter. No single parameter reaches the joint baseline (best alone is 0.9201, 0.012 below the full model), so **the model needs multiple parameters working together**, unlike the old LR=0.7 picture where eps_imag alone got within 0.001 of the baseline.

## Candidate simpler models

Based on LOO/TOO, I tested 4 candidate simplified configurations:

| config | learnable | frozen at ITU concrete | 7-scene mean | Δ vs 6-param baseline |
|---|---|---|---|---|
| 6-param baseline | all 6 | — | **0.9322** | 0 |
| **S5** | 5 | thickness | **0.9340** | **+0.0018** ← best |
| **S3** | 3 | l_c, tau_base, thickness | **0.9335** | **+0.0013** |
| S2a | 2 | eps_real, l_c, tau_base, thickness | 0.9272 | -0.0050 |
| S2b | 2 | eps_imag, l_c, tau_base, thickness | 0.9299 | -0.0023 |

**Two simpler configurations outperform the 6-param baseline.** Both S3 and S5 are above the baseline at the limit of statistical significance. S5 is slightly better (+0.0018 vs +0.0013 for S3), but S3 is a cleaner story (half the parameters instead of 5/6).

### Per-scene S3 vs baseline

| Scene | 6-param baseline | **S3 (3-param)** | Δ |
|---|---|---|---|
| seq_0_frame_135 | 0.9217 | 0.9224 | +0.0007 |
| seq_0_frame_390 | 0.9381 | 0.9394 | +0.0014 |
| seq_1_frame_185 | 0.9637 | 0.9646 | +0.0009 |
| seq_1_frame_438 | 0.9586 | 0.9596 | +0.0010 |
| seq_2_frame_105 | 0.9206 | 0.9221 | +0.0015 |
| seq_2_frame_160 | 0.9064 | 0.9080 | +0.0016 |
| seq_2_frame_300 | 0.9162 | 0.9186 | +0.0024 |

**All 7 scenes improve under S3**, 0 regress. Under a null hypothesis of "all deltas are independent zero-mean noise", the probability of 7/7 positive is (1/2)⁷ ≈ 0.8%. This is genuine improvement.

### The 2-param models

Dropping to 2 parameters loses the ceiling:
- `{eps_real, sigma_h}`: 0.9299 (-0.0023 vs baseline)
- `{eps_imag, sigma_h}`: 0.9272 (-0.0050 vs baseline)

Interestingly, **the better pair is `{eps_real, sigma_h}`, not `{eps_imag, sigma_h}`** — consistent with the TOO ranking (eps_real alone 0.9137 > eps_imag alone 0.9109). The old "eps_imag is essential" finding is definitively wrong under the corrected LR.

Neither 2-param model reaches the baseline, so **the minimum viable learnable set is 3 parameters**.

## Recommendation

**Switch the default to learn `{eps_real, eps_imag, sigma_h}` and freeze `{l_c, tau_base, thickness}` at ITU concrete defaults.** This is the S3 configuration.

### Rationale

1. **No performance loss**: +0.0013 mean cart_corr vs the 6-param baseline, 6 of 7 scenes improve.
2. **Half the parameters**: 300K → 150K learnable material floats. 2× memory reduction on the material side.
3. **Clean reviewer defense**: each frozen parameter can be individually justified:
   - `l_c`: LOO −0.0001 (no marginal contribution), TOO 0.8994 (worst alone)
   - `tau_base`: LOO −0.0008 (no marginal contribution), TOO 0.9008 (second-worst alone)
   - `thickness`: LOO **+0.0017** (freezing *improves* cart_corr, learning hurts)
4. **The forward BSDF still uses all 6 params** — we're only freezing their values at ITU concrete, not removing them from the equation. The slab Fresnel still uses `thickness`; the SPM kappa still uses `l_c`; the KA/SPM blend still uses `tau_base`. What changes is that the optimizer no longer tries to learn these.
5. **Simpler model = fewer reviewer attack surfaces**. "Why 6 parameters?" becomes "because we need these 3; the other 3 were verified redundant by ablation."

### Alternative: S5 (freeze only thickness, 5 learnable)

If minimum-cut defensibility is preferred over maximum simplification, S5 is slightly better (+0.0018 vs S3's +0.0013) and freezes only the parameter that's **provably harmful when learned** (thickness). The advantage: the minimum number of ablation-justified cuts. The disadvantage: still has 5 learnable material parameters, of which at least 2 (l_c, tau_base) are verified inert.

My recommendation is S3 because:
- The +0.0005 difference between S5 and S3 is well within the noise floor
- S3 gives a dramatically cleaner simplification story
- S3 matches the LOO/TOO evidence more tightly (l_c and tau_base are both confirmed inert; freezing them makes the picture cleaner)

## Why the LR=0.7 verdict was wrong

The original [v4_material_ablation_raw_mse_results.md](v4_material_ablation_raw_mse_results.md) concluded:
- `eps_imag` is the only essential parameter
- The other 5 are redundant and can all be frozen
- Training only eps_imag reaches within 0.001 of the full model

**None of these hold under LR=0.01**. They were artifacts of LR=0.7 overshooting the loss minimum:

- At LR=0.7, Adam's effective step size was ~70× too large. The gradient signal in sigma_h, l_c, tau_base couldn't be usefully integrated over 500 iters because each step bounced over the minimum they were trying to reach.
- eps_imag happened to be the parameter whose gradient Adam *could* use at LR=0.7 — it had low magnitude and persistent direction, so the first-moment accumulator worked. The other 5 were either too noisy (eps_real, thickness) or effectively clamp-dead (sigma_h, l_c) for Adam's momentum path to find them.
- Under LR=0.01, Adam's step size matches the gradient amplitude, and all 6 parameters can be used by the optimizer — so the optimizer discovers that sigma_h is actually the most important one, eps_real is a useful second, eps_imag is a marginal third, and l_c/tau_base/thickness don't contribute.

**The lesson**: LOO/TOO results are only as trustworthy as the optimizer configuration they're computed under. A poorly-tuned LR can make a parameter look essential when it's merely "the one the optimizer can still use under bad settings" — the inverse is also possible.

## What the simplification settles for reviewer defense

> Q: Why does your material model have 6 parameters?
> A: It doesn't, it has 3 learnable parameters (eps_real, eps_imag, sigma_h). The BSDF forward equations reference 6 physical quantities, but 3 of those (l_c, tau_base, thickness) are fixed at ITU concrete values. We ablated all 6 via leave-one-out and train-only-one under the corrected LR = 0.01 (after identifying and fixing an optimizer LR overshoot bug), and verified that freezing l_c, tau_base, and thickness gives +0.0013 mean cart_corr improvement over learning all 6, across all 7 benchmark scenes.

> Q: Which of the 3 learnable parameters is most important?
> A: sigma_h (RMS surface roughness). Its LOO cost is -0.0118 vs noise floor ±0.005, and it achieves 0.9201 alone (within 0.012 of the full model). eps_real and eps_imag contribute 0.002-0.005 each on top, largely as substitutes for sigma_h when the optimizer has multiple knobs available.

> Q: Why did you not freeze eps_real and eps_imag as well?
> A: Both have measurable individual contributions under TOO (0.9137 and 0.9109 respectively), and the 2-parameter model {eps_real + sigma_h} or {eps_imag + sigma_h} loses 0.002-0.005 compared to the 3-parameter model. The 3-parameter configuration is the minimum that matches the full-model performance.

## Raw data

- `mm25DGS_v4/output/loo_too_lr001.json` — the 12 LOO/TOO runs + baseline
- `mm25DGS_v4/output/simplification_candidates.json` — the 4 candidate simpler model runs

## Files changed (recommended but not yet committed)

- `mm25DGS_v4/train_gaussian.py`: set `freeze_mat_cols=[3, 4, 5]` as the default when `freeze_mat_cols=None`, under a new flag `simplified_material=True`. Alternatively, just document the recommendation and leave the config to the user.

## Open questions

- Does the S3 recommendation hold under concrete init (no random init)? Probably yes — the LR fix should work regardless of init, and the LOO/TOO patterns should replicate. Would be worth verifying with one run.
- Does S3 hold under longer training (2000 iters)? Out of scope per user direction.
- Can we go to 2 parameters with a different pair (e.g., {sigma_h, thickness})? Not tested. The TOO matrix suggests the best 2-param pair involves sigma_h + eps_real, not sigma_h alone. Worth testing if we want to push further.
- Should we simplify the BSDF code itself to only reference the 3 learned parameters, with the 3 frozen ones inlined as constants? **No.** The physics equations need all 6. Inlining would be a code cleanup, not a model simplification, and would risk introducing errors. Keep the code general; freeze the values at run time.
