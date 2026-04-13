# v4 unnormalized MSE loss: forward-model scale fix and material learning plan

## Motivation

The T2 Pearson loss investigation yielded **+0.018 cart_corr** over the min-max-normalized MSE baseline (0.9274 → 0.9456) across all 7 scenes. But Pearson is scale- and shift-invariant — meaning **it throws away exactly the absolute-amplitude signal that materials control**. Materials govern reflection magnitude (ITU brick ~0.01, concrete ~0.05, metal ~0.9), so a loss that ignores absolute amplitude is probably giving the Pearson +0.018 mostly to normals (structural) and not to materials.

**The real solution**: train with raw MSE (no min-max normalization) so that absolute-amplitude errors carry gradient signal, and materials can learn the *physical quantity they actually control*. The only thing blocking this today is the **scale mismatch between rendered RA and GT RA** — we originally added min-max normalization to paper over this mismatch rather than fixing it.

This plan (a) diagnoses the scale mismatch in the forward model, (b) fixes it (or absorbs residuals into a learnable gain), (c) re-runs a 2² `(LEARN_MATERIALS, LEARN_NORMALS)` ablation under raw MSE to cleanly measure each knob's contribution, and (d) re-runs the D1/D2/D3 diagnostics on scene 135 to confirm the mechanistic picture.

`LEARN_PATTERNS` is permanently set to `False` going forward per user direction (the TI manufacturer spec sheets give accurate antenna patterns; learning them adds only +0.013 over not learning them, within noise).

---

## Phase α — Diagnose the scale mismatch

**~30 min of implementation + 1 min of compute.**

### α1. Quantify the mismatch

Write a small standalone script that, for each of the 7 scenes, at ITU-concrete init (no training):

- Load the trained scene context exactly as `train_gaussians` would (init model + rast, freeze Mitsuba).
- One forward pass of `render_gaussians` at init → `rp_real, rp_imag` → `range_profile_to_ra` → `|RA_rendered|` (numpy).
- Load `gt_adc_ri`, compute `|RA_gt|` via the same `adc_to_ra_complex` pipeline.
- Compute these statistics and report them per scene:
  - `scale_mean = mean(|RA_rendered|) / mean(|RA_gt|)` — overall mean ratio
  - `scale_median = median(|RA_rendered|) / median(|RA_gt|)` — robust estimate
  - `scale_by_range[k]` — ratio averaged over azimuth, as a function of range bin (length K = 256)
  - `scale_by_channel[t, r]` — ratio averaged over range bins, as a function of (TX, RX)
  - `log_rend_hist`, `log_gt_hist` — log10-histograms of rendered and GT magnitudes

### α2. Classify the mismatch pattern

From the above, determine which of the following dominates:

| Pattern | Likely cause | Fix path |
|---|---|---|
| **Constant scalar** (`scale_by_range` flat, `scale_by_channel` flat) | Missing `Pt` transmit-power factor, wrong `C_radar` prefactor, ADC unit scale | Multiply rendered by a constant in `render_factorized` |
| **`~R^n` range dependence** | `R⁴` propagation loss mis-applied or double-applied | Check `d_tx`, `d_rx` use in `render_factorized` |
| **`~cos(θ)^n` angle dependence** | Antenna pattern normalization | Check `alpha_tx`, `alpha_rx` in rasterizer |
| **Per-channel non-uniform** (`scale_by_channel` varies) | Per-antenna gain miscalibration | Per-channel learnable gain |
| **Per-scene varying** | Scene-dependent calibration | Need per-scene learnable gain |

### α3. Report and persist

Dump α1 + α2 results to `mm25DGS_v4/output/scale_investigation/scale_report.json` and print a summary. This report decides which path Phase β takes.

---

## Phase β — Fix the forward model scale

**~30 min to 3 hours depending on what α2 says.**

### β1. Simple scalar fix

If the mismatch is a constant scalar (say, factor of 100×), identify the missing term in `render_factorized` and add it. The obvious candidates:

- **Transmit power `Pt`**: radar equation includes `Pt` that's probably set to 1.0 implicitly; physical radars transmit ~10–100 mW.
- **`4π` factors**: the radar equation has `(4π)³` in the denominator; easy to miscount.
- **ADC bit depth / voltage**: TI MMWCAS ADCs are 12-bit signed with a specific voltage range.
- **Antenna gain normalization**: `C_radar` in the code is a manually-tuned scale factor.

Fix in one of: `render_factorized` step 5 (before splat), or in `C_radar` at rasterizer init.

### β2. Range-dependent fix

If the mismatch varies as `R^n`, the radar equation's `R⁴` path-loss is probably wrong. Inspect `d_tx`, `d_rx` usage in `render_factorized`.

### β3. Learnable global gain (fallback)

If none of the above cleanly explains the mismatch, or if it varies per-scene, add a **single learnable scalar** `log_gain = nn.Parameter(torch.zeros(1))`. The rendered output gets multiplied by `exp(log_gain)` before the loss. This is a 1-parameter calibration head that absorbs residual scale without contaminating the material/normal gradient paths.

- LR: `1e-2` on `log_gain` (moves in log space, so 1e-2 is roughly a 1% per-step update)
- Initialization: `log_gain = 0` (gain = 1.0)
- Gated off by a new flag `LEARN_GAIN = True` (separate from `LEARN_MATERIALS`/`LEARN_NORMALS`).

### β4. Verify

Re-run α1 after the fix. Target: **at init, `scale_mean` within factor of 2 of 1.0 across all 7 scenes**. If yes, we can drop min-max normalization safely.

---

## Phase γ — Raw MSE ablation (2² on materials × normals)

**~25 min compute for the 2² matrix, plus ~75 min to comparison-table the other losses.**

### γ1. Permanent changes

- `LEARN_PATTERNS = False` in `train_gaussian.py` (permanent, per user direction).
- Add `loss_type='mse_raw'` to `compute_ra_loss_rp`:
  - Drop the min-max normalization on both rendered and GT.
  - Compute MSE directly on `|RA_rendered| - |RA_gt|` in absolute units.
  - `precompute_gt_loss_norm` becomes `precompute_gt_mag` — just `torch.abs(adc_to_ra_complex(gt_adc_ri))`, no min/max.

### γ2. 2² LEARN matrix

Four 7-scene runs under `loss_type='mse_raw'` and the post-β fixed forward model:

| M | N | Run name | Tests |
|---|---|---|---|
| 0 | 0 | `R0_mse_raw_M0_N0` | True floor under raw MSE (rendered vs GT at init, rendered is fixed concrete + fixed normals, no optimization loop) |
| 1 | 0 | `R1_mse_raw_M1_N0` | Materials only — the headline experiment |
| 0 | 1 | `R2_mse_raw_M0_N1` | Normals only — sanity check that normals still learn without Pearson/min-max helping them |
| 1 | 1 | `R3_mse_raw_M1_N1` | Both — marginal contribution of each knob |

### γ3. Comparison table

Build a table in `md/v4_material_ablation_results.md` comparing the 2² matrix under three loss types:

| config | mse (min-max) | pearson | mse_raw |
|---|---|---|---|
| M0_N0 | 0.3729 (existing B0_zero minus patterns) | _new_ | _new_ |
| M1_N0 | 0.8490 (existing) | _new_ | _new_ |
| M0_N1 | 0.9095 (existing) | _new_ | _new_ |
| M1_N1 | 0.9301 (existing B3 roughly, minus patterns) | 0.9456 (T2) | _new_ |

The **M1_N0 cell under mse_raw is the headline**. Under min-max MSE it was 0.849; under Pearson we don't have a number yet. If raw MSE gives materials their true physical signal, **M1_N0 should move substantially** — potentially matching or exceeding M0_N1.

### γ4. Decision rules

Interpret the raw-MSE 2² matrix with:

- **If `M1_N0_raw > 0.849 + 0.02`**: materials benefit from the raw MSE loss. The hypothesis is confirmed — our current findings about "material model is structurally limited" are an artifact of the min-max normalization loss.
- **If `M1_N0_raw ≈ 0.849 ± 0.005`**: materials are neither helped nor hurt. The amplitude signal doesn't carry material-specific information, suggesting materials really are structurally bounded.
- **If `M1_N0_raw < 0.849 - 0.02`**: materials are *hurt* by raw MSE. The likely cause is a residual scale mismatch dominating the loss; β1 needs to be revisited.
- **If `M1_N1_raw > max(M1_N1_mse, M1_N1_pearson)`**: raw MSE + good scale alignment is actually the best-overall loss.
- **Combine with the diagnostics (Phase δ)**: per-iter gradient stats and Fisher under raw MSE should show significantly larger material Fisher than min-max MSE.

---

## Phase δ — Diagnostic re-run on scene 135

**~5 min compute.**

Re-run `investigate_materials.py` on scene 135 with `loss_type='mse_raw'` and compare against the existing `loss_type='mse'` (min-max) diagnostics:

1. **D1 (loss invariance)**: at the trained state, sweep ε and measure cart_corr. Under raw MSE, the uniform-offset Δ should be **larger** (the gauge direction now carries gradient, so the trained state should be steeper in it).
2. **D2 (per-iter gradient stats)**: the `|grad.mean|` component (gauge-invariant direction) should be **larger** under raw MSE. The current ratio of grad_std / |grad.mean| is ~300× under min-max; under raw MSE it should be closer to ~10× or lower.
3. **D3 (trajectory decomposition)**: per-column trajectory mean should drift more under raw MSE (the optimizer can now move materials globally).
4. **Fisher diagonal**: larger on material columns under raw MSE — ideally 10–100× larger.

If these four predictions hold, we have mechanistic confirmation that the loss change is doing what the theory says, and any cart_corr improvement in Phase γ can be attributed to materials learning their physical quantity rather than the loss accidentally benefitting normals.

---

## Implementation checklist (in execution order)

1. **α1–α3**: write `mm25DGS_v4/investigate_scale.py`, run, dump report to JSON. Report pattern type. *[30 min + 1 min run]*
2. **β1–β4**: implement the fix suggested by α2. Re-run α1 to verify scale is within 2× of 1.0 at init. *[30 min–3 hr]*
3. **γ1**: permanent `LEARN_PATTERNS = False`, add `loss_type='mse_raw'`. *[10 min]*
4. **γ2**: 4 runs × 7 scenes × 500 iters under `loss_type='mse_raw'`. *[~25 min]*
5. **γ3**: fill the comparison table in `md/v4_material_ablation_results.md`. If data is missing for comparison cells, backfill with ~20 min of runs. *[20–80 min compute]*
6. **γ4**: interpret decision rules, update results .md with the verdict.
7. **δ**: run `investigate_materials.py --loss-type mse_raw` on scene 135. Compare to existing mse results. *[5 min]*
8. Commit all changes as one or two commits with descriptive messages. Update `md/v4_material_ablation_results.md` with a new section documenting the raw MSE path.

---

## What this experiment can **NOT** answer

- **Per-scene absolute amplitude calibration transfer** to the single-chip IWR1443 radar. That's the Evaluation #2 problem (explicitly out of scope for the current project per user direction).
- **Whether the "true" physical reflection coefficient per point was actually recovered**. Cart_corr is a structural metric; even with raw MSE, the learned materials might be only approximately correct in absolute units.
- **Whether neural / tabulated BRDF alternatives** would work better. Out of scope, tabled for later.

---

## Fallbacks

- **If Phase α reveals a pathological mismatch** (e.g., scale varies 10³× across scenes or is wildly non-uniform within a scene), the clean path is β3 (learnable global gain) — it trades theoretical rigor for a clean empirical fix that unblocks Phase γ.
- **If Phase γ's M1_N0_raw doesn't improve**, we should not spend more time on "fix the material model" and instead return to the Phase 1.5 reduced-BSDF recommendation from the earlier ablation (Jones + slab + KA, drop SPM/CBS/broad/blend/directive).

---

## Results (Phases α, β, γ complete)

### Phase α (factory patterns, no C_radar boost)
- mmIR-trained patterns carried per-scene amplitude bias (scene 390 was 2–3× louder).
- Factory patterns collapsed cross-scene log10 spread from **1.42 → 0.22 decades**.
- Residual: rendered was uniformly ~**100× dimmer** than GT.

### Phase β (100× C_radar boost)
- Applied `C_radar_gt_match = 100` in the rasterizer.
- All 7 scenes now within **1.3×** of GT at ITU-concrete init.
- `USE_FACTORY_PATTERNS = True` default, `LEARN_PATTERNS = False` permanent.

### Phase γ (2² × 3 losses)

| config | mse | pearson | mse_raw |
|---|---|---|---|
| M0_N0 (floor) | 0.3348 | 0.3348 | 0.3348 |
| M1_N0 (materials only) | 0.8396 | 0.8689 | 0.8686 |
| M0_N1 (normals only) | 0.9158 | **0.9389** | 0.9040 |
| **M1_N1 (both)** | 0.9205 | **0.9425** | 0.9324 |

Marginal contribution of each knob on top of the other:

| loss | materials-on-top-of-normals | normals-on-top-of-materials |
|---|---|---|
| mse | +0.0047 | +0.0809 |
| pearson | +0.0036 | +0.0736 |
| **mse_raw** | **+0.0284** | +0.0638 |

**Hypothesis confirmed**: under raw MSE, materials contribute **+0.028 on top of normals** — 6× more than under min-max MSE, 8× more than under Pearson. The previous "materials are structurally limited" finding was an artifact of the min-max normalization loss.

**But**: Pearson is still the best *total* cart_corr (0.9425) because normals are weaker under mse_raw (M0_N1_raw = 0.9040 vs 0.9389). Normals want a structural loss; materials want a raw-amplitude loss.

### Natural follow-up

A **hybrid loss** `α·mse_raw + (1−α)·pearson` should let both knobs play to their strengths simultaneously. If materials can retain their +0.028 under mse_raw and normals retain their +0.074 under Pearson, the hybrid could reach cart_corr > 0.95. This is the next experiment.
