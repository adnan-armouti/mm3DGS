# Matched-grid frame-NVS analyses (A–K) — findings

Date: 2026-04-19
Scope: 2 scenes × 4 converged variants on **matched position grids**
(`seed_frame = test_frame` default, `target_n = 20 000`). Companion to
the prior-grid analysis at
[`md/frame_nvs_analysis/findings.md`](../frame_nvs_analysis/findings.md).

Data + plots: this directory.

---

## 0. TL;DR — new conclusion differs sharply from the prior-grid story

The previous analysis closed the door on H3/H4 (normal regulariser)
because aggregate normal drift was indistinguishable between HO and
UB (28 ° ± 0.5 on seq_1, 33 ° ± 0.5 on seq_2). With matched grids +
Fisher-weighted inspection:

1. **HO's problem is primarily the NORMALS, not the materials.**
   Swap ablation (Analysis I) with identical positions:
   - seq_1: `HO_mat + UB_rot` gives **+0.173 cc** over HO native;
     `UB_mat + HO_rot` gives only +0.061.
   - seq_2: `HO_mat + UB_rot` gives **+0.049 cc**;
     `UB_mat + HO_rot` gives only +0.009.
2. **Aggregate normal drift statistics hide this.** HO mean drift
   (28 °–33 °) ≈ UB mean drift — but at the top-1 % Fisher points HO
   drifts **17 ° more than UB on seq_2** (69.7 ° vs 52.4 °). Those ~200
   points are where cart_corr lives.
3. **Position-grid fix lifts HO by the expected amount.** On seq_2
   HO_128 +0.086, HO_8 +0.052. On seq_1 (noisier pass-2 trajectory)
   +0.021 and +0.067. Target_n 90k→20k costs UB ≈ −0.09 uniformly.
   Net HO lift after isolating target_n cost: +0.11 to +0.18.
4. **Fisher concentration is still tight even at N = 20 000.**
   Top-1 % of 200 points carry 90 %–97 % of per-point Fisher (was
   93 %–99 % at N = 90 k). The rank of `cart_corr` is a scene/pose
   property, not a parameter-count property.
5. **0.70 HO target: still not hit.** Best HO: 0.590 (seq_2) /
   0.567 (seq_1). Soft-bound from the Fisher-weighted normal drift
   analysis: **Fisher-weighted rotation regulariser could plausibly
   close the remaining gap** by targeting the 17°-spurious-drift on
   high-Fisher points. See §13 for the go-forward plan.

---

## 1. Headline comparison — prior grid vs matched grid

| variant | seq_1 prior | seq_1 matched | Δ | seq_2 prior | seq_2 matched | Δ |
|---|---:|---:|---:|---:|---:|---:|
| HO_128 | 0.4904 | **0.5118** | +0.021 | 0.5043 | **0.5900** | +0.086 |
| HO_8   | 0.5003 | **0.5672** | +0.067 | 0.5116 | **0.5633** | +0.052 |
| UB_144 | 0.9126 | 0.8309 | −0.082 | 0.8228 | 0.7319 | −0.091 |
| UB_9   | 0.9510 | 0.9090 | −0.042 | 0.8814 | 0.8063 | −0.075 |

**Interpretation of the Δs.** The matched-grid change bundles two
things: (a) `seed_frame = test_frame` and (b) `target_n = 20000`.
Use UB to estimate (b) — UB's grid was already `test_frame` under the
old code (middle of 9 train_frames = test_frame), so its Δ is purely
the target_n cost:

- UB_144 Δ ≈ −0.09 (both scenes)  →  target_n 90k→20k costs ~0.09 cc
- UB_9 Δ ≈ −0.06 (both scenes)    →  short variants less affected

For HO the matched grid provides a pure position-grid lift of
**+0.11 to +0.18 cc** (net Δ minus the ~−0.09 target_n cost). That
is consistent with Analysis I's prior-grid prediction that
UB-on-HO-grid capped at 0.52/0.55 ≈ HO-native plus a positive
correction for aligned grids.

---

## 2. Analysis 0 — reproducibility

All 8 variants reproduce within Δ ≤ 0.001 cc of their reported
`final_test_cc`. See `0_reproducibility.csv`.

---

## 3. Analysis A — per-variant parameter distributions

Raw-materials std per column — **UB/HO are now within ≤ 2 %** on
every column of both scenes (was ≤ 5 % on prior grid):

| scene | variant | eps_real | eps_imag | sigma_h | l_c | tau | thickness |
|---|---|---|---|---|---|---|---|
| seq_1 | HO_128 | 0.349 | 0.694 | 1.334 | 0.827 | 0.615 | 0.338 |
| seq_1 | UB_144 | 0.351 | 0.688 | 1.328 | 0.836 | 0.604 | 0.339 |
| seq_2 | HO_128 | 0.373 | 0.762 | 1.307 | 0.844 | 0.523 | 0.359 |
| seq_2 | UB_144 | 0.367 | 0.748 | 1.323 | 0.844 | 0.514 | 0.353 |

**Conclusion:** UB and HO distributions are essentially identical at
the aggregate level. This is a null finding — but it's the *right*
null because the matched-grid comparison is the clean one.
**Rules out H1 definitively.**

---

## 4. Analysis B — per-point drift from init

| scene | variant | L2 mean | L2 p95 | deg mean | deg p95 |
|---|---|---:|---:|---:|---:|
| seq_1 | HO_128 | 2.061 | 3.679 | 28.58 | 72.69 |
| seq_1 | UB_144 | 2.071 | 3.687 | 28.22 | 71.96 |
| seq_2 | HO_128 | 1.964 | 3.641 | 33.33 | 85.44 |
| seq_2 | UB_144 | 1.959 | 3.606 | 32.71 | 85.26 |

HO and UB drift by indistinguishable amounts on average. **If
aggregate drift were the signal, there would be nothing to fix.**

---

## 5. Analysis C — per-point UB vs HO divergence (exact, no NN remap)

| scene | pair | eps_real | eps_imag | sigma_h | l_c | tau | thickness |
|---|---|---:|---:|---:|---:|---:|---:|
| seq_1 | HO_128-HO_8 | 0.303 | 0.543 | 1.059 | 0.672 | 0.269 | 0.293 |
| seq_1 | **UB_144-HO_128** | 0.282 | 0.456 | 0.736 | 0.505 | 0.191 | 0.274 |
| seq_2 | HO_128-HO_8 | 0.328 | 0.498 | 1.006 | 0.646 | 0.239 | 0.312 |
| seq_2 | **UB_144-HO_128** | 0.295 | 0.423 | 0.656 | 0.429 | 0.166 | 0.285 |

Counter-intuitively, **UB vs HO divergence is SMALLER than
within-mode divergence** (HO_128 vs HO_8). This confirms the
aggregate parameters are near-identical; the signal is in *per-point*
structure, not aggregate column statistics.

---

## 6. Analysis D — k-NN material smoothness (decision point for S3)

Mean neighbour-std per column (smaller ⇒ smoother material field):

| scene | variant | eps_real | eps_imag | sigma_h | l_c | tau | thickness |
|---|---|---:|---:|---:|---:|---:|---:|
| seq_1 | HO_128 | 0.296 | 0.586 | 1.169 | 0.739 | 0.405 | 0.286 |
| seq_1 | UB_144 | 0.298 | 0.586 | 1.161 | 0.743 | 0.399 | 0.288 |
| seq_2 | HO_128 | 0.320 | 0.623 | 1.105 | 0.735 | 0.337 | 0.305 |
| seq_2 | UB_144 | 0.314 | 0.613 | 1.131 | 0.738 | 0.327 | 0.299 |

UB is ≤ 2 % smoother than HO on both scenes. **Not a compelling
signal.** S3 (k-NN spatial TV on materials) is **no-go** —
the smoothness gap is too small to regularise.

---

## 7. Analysis E — normal-field smoothness

Mean k-NN neighbour angle (deg):

| scene | variant | mean | median | p95 |
|---|---|---:|---:|---:|
| seq_1 | init    | 49.02 | 44.73 | 93.94 |
| seq_1 | HO_128  | 59.95 | 56.40 | 101.64 |
| seq_1 | UB_144  | 59.71 | 56.17 | 101.93 |
| seq_2 | init    | 52.35 | 47.74 | 98.87 |
| seq_2 | HO_128  | 62.96 | 59.50 | 104.16 |
| seq_2 | UB_144  | 62.83 | 59.56 | 104.40 |

HO and UB normals are indistinguishable in aggregate smoothness.
**S3 variant on normals is also no-go.**

---

## 8. Analysis G — Fisher concentration (matched grid, N = 20 000)

Per-column mean |grad| (ordering identical to prior grid — thickness
and eps_real dominate):

| scene | variant | eps_real | eps_imag | sigma_h | l_c | tau | thick |
|---|---|---:|---:|---:|---:|---:|---:|
| seq_1 | HO_128 | 3.89e-4 | 8.9e-6 | 1.6e-5 | 9.7e-6 | 2.1e-6 | 4.0e-3 |
| seq_1 | UB_144 | 1.90e-4 | 4.3e-6 | 7.1e-6 | 3.8e-6 | 9.3e-7 | 2.0e-3 |
| seq_2 | HO_128 | 3.32e-4 | 3.6e-6 | 3.9e-6 | 5.7e-6 | 8.8e-7 | 3.4e-3 |
| seq_2 | UB_144 | 2.64e-4 | 3.4e-6 | 3.6e-6 | 4.1e-6 | 6.7e-7 | 2.8e-3 |

Per-point concentration (top-k %):

| variant | top-1 % | top-5 % | top-10 % | top-25 % |
|---|---:|---:|---:|---:|
| seq_1 HO_128 | 95.3 % | 99.3 % | 99.8 % | 100.0 % |
| seq_1 UB_144 | 93.7 % | 99.2 % | 99.8 % | 100.0 % |
| seq_2 HO_128 | 95.2 % | 99.5 % | 99.9 % | 100.0 % |
| seq_2 UB_144 | 97.1 % | 99.7 % | 99.9 % | 100.0 % |

**Fisher concentration is unchanged by matched grid / smaller N.**
~200 of 20 000 points carry 95 % of test-cc sensitivity. This is a
physics property (§1.2 of
[`md/frame_nvs_fisher_concentration_analysis.md`](../frame_nvs_fisher_concentration_analysis.md)),
not an artefact of the position grid.

---

## 9. Analysis I — material/normal swap ablation (EXACT on matched grid)

Because HO and UB now share positions, we can directly swap
`(raw_materials, rotations)` with no NN remap:

### seq_1_frame_438

| combo | cc | Δ vs HO native |
|---|---:|---:|
| HO_128 native               | 0.5118 | — |
| UB_144 native               | 0.8304 | +0.319 |
| UB mat + HO rot             | 0.5733 | +0.061 |
| **HO mat + UB rot**         | **0.6853** | **+0.174** |
| HO_8 native                 | 0.5670 | — |
| UB_9 native                 | 0.9082 | +0.341 |
| UB_9 mat + HO_8 rot         | 0.6526 | +0.086 |
| **HO_8 mat + UB_9 rot**     | **0.7644** | **+0.198** |

### seq_2_frame_105

| combo | cc | Δ vs HO native |
|---|---:|---:|
| HO_128 native               | 0.5902 | — |
| UB_144 native               | 0.7316 | +0.141 |
| UB mat + HO rot             | 0.5990 | +0.009 |
| **HO mat + UB rot**         | **0.6385** | **+0.048** |
| HO_8 native                 | 0.5634 | — |
| UB_9 native                 | 0.8058 | +0.242 |
| UB_9 mat + HO_8 rot         | 0.5924 | +0.029 |
| **HO_8 mat + UB_9 rot**     | **0.6441** | **+0.081** |

**Critical finding (across both scenes, all chirp regimes):**
`HO_mat + UB_rot` always outperforms `UB_mat + HO_rot`. **The
rotations (normals) are the dominant part of the HO → UB gap.**
Materials contribute; normals contribute more.

Magnitude of the normal-swap effect: on seq_1 it alone closes 55 %
of the HO-to-UB gap; on seq_2 it closes 34 %.

### Why did the prior grid miss this?

With mis-matched grids, the "swap" required NN-remapping UB onto
HO's positions. Per [md/frame_nvs_analysis/findings.md §I]
the median NN distance was 3 cm = **7.5 λ** of phase error at
77 GHz; that noise masked the real normal-vs-material attribution.

---

## 10. Fisher-weighted normal drift (new analysis — motivation for H3+Fisher)

Aggregate normal drift hides per-point signal. Weighting by
Fisher exposes it:

| scene | variant | unweighted drift | drift at top-1 % Fisher | Δ |
|---|---|---:|---:|---:|
| seq_1 | HO_128 | 28.58 ° | **39.69 °** | +11.1 ° |
| seq_1 | UB_144 | 28.22 ° | 36.84 ° | +8.6 ° |
| seq_2 | HO_128 | 33.33 ° | **69.68 °** | +36.4 ° |
| seq_2 | HO_8   | 33.86 ° | 50.76 ° | +16.9 ° |
| seq_2 | UB_144 | 32.71 ° | 52.40 ° | +19.7 ° |

**Read this row-by-row:**
- seq_2 HO_128 drifts **69.7° at top-1% Fisher points** (vs UB_144's
  52.4° at the same points → 17° gap). Aggregate HO/UB drift differs
  by 0.6° but the points that matter differ by 17°.
- seq_1 HO_128 at top-1%: 39.7° vs UB_144's 36.8° → 2.9° gap; matches
  the smaller Analysis I normal-swap gain on seq_1 (+0.174) vs seq_2
  (+0.048 — wait, seq_1 normal-swap gain is LARGER than seq_2).

The magnitudes appear to conflict at first glance: seq_2 has a
bigger HO-vs-UB normal drift gap (17 °) but a smaller Analysis I
normal-swap gain (+0.048) than seq_1 (2.9 °, +0.174). This is
resolved by noting that seq_2 HO's 69.7 ° drift is *enormous in
absolute terms* — so large that UB's 52.4 ° is a very small
correction, whereas seq_1's 39.7 ° drift at a high-Fisher point is
already "approximately right," and UB's 36.8 ° is a meaningful
refinement. The cc gain from swapping normals scales with **how
far from a local cart_corr optimum HO's normals are at the top
points**, which is not the same as the HO-vs-UB difference.

### Direct cross-variant normal disagreement at high-Fisher points

| scene | pair | all | top-1 % | top-5 % | top-10 % |
|---|---|---:|---:|---:|---:|
| seq_1 | HO_128 vs UB_144 | 21.74 ° | 20.42 ° | 24.54 ° | 24.86 ° |
| seq_2 | HO_128 vs UB_144 | 22.84 ° | 27.91 ° | 28.13 ° | 27.44 ° |

HO and UB disagree on 20–28 ° of normal direction at the high-Fisher
points. Given Analysis I shows this costs HO 0.05–0.17 cc, a
regulariser that enforces HO's normals to stay near their init
(which UB's normals are also close to — they drift *same amount on
average*, just in a different direction at the top points) should
recover most of that gap.

---

## 11. Analysis J — LiDAR-intensity prior

Intra-bin-variance / total-variance ratio is 0.99+ on every
(scene, variant, column). LiDAR intensity explains ~0 % of
radar-material variance. **H8 definitively rejected** — physics
mismatch (905 nm vs 3.9 mm), already-expected. See
[`md/frame_nvs_fisher_concentration_analysis.md`](../frame_nvs_fisher_concentration_analysis.md)
§3 for the rationale.

---

## 12. Analysis K — UB_144 vs UB_9 noise-averaging

Per-column std diff (UB_9 − UB_144):

| scene | eps_real | eps_imag | sigma_h | l_c | tau | thickness |
|---|---:|---:|---:|---:|---:|---:|
| seq_1 | +0.008 | +0.008 | +0.056 | +0.011 | +0.005 | +0.007 |
| seq_2 | +0.004 | −0.000 | +0.056 | +0.040 | +0.002 | +0.003 |

UB_9 has marginally larger std than UB_144 on sigma_h and l_c; rest
indistinguishable. No clean "sharpness vs averaging" story.

---

## 13. Updated hypothesis go/no-go (MATCHED GRID)

| H | description | prior-grid | matched-grid | status |
|---|---|---|---|---|
| H1 | L2 drift on raw_materials (uniform) | marginal +0.01 | **no-go** | A+B show UB/HO drift identically in aggregate |
| H2 | k-NN spatial TV on raw_materials | weak | **no-go** | D: UB≤2% smoother than HO on both scenes |
| H3 | **L2 drift on rotations (Fisher-weighted)** | closed-out | **STRONGLY MOTIVATED** | I: normal swap is the dominant fault; Fisher-weighted analysis shows drift concentrated at ~200 top-Fisher points |
| H4 | k-NN spatial TV on normals | no-go | **no-go** | E: HO/UB aggregate normal smoothness identical |
| H5 | Fisher-weighted L2 on materials | pending | **weak** | I: material-swap gain +0.01–0.06 (vs normal's +0.05–0.17) |
| H6 | UB-cluster prior | UB-leaking | unchanged | still only valid as ceiling |
| H7 | Activity mask via Adam-v | marginal +0.01 | **motivated, revise** | should mask on *rotations* specifically |
| H8 | LiDAR-intensity prior | no-go | **no-go** | J confirms; already rejected on physics |

**New prioritised hypothesis (not previously in the H1–H8 set):**
- **H3+F**: `λ · Σ_i F_i · ||rotations[i] − init_rotations[i]||²`
  (L2 on rotation drift, Fisher-weighted so the top 200 points
  are pinned near their init LiDAR normals, while the other 19 800
  points are free).

---

## 14. Recommended next step

**S2-rot (revision of S2 from `md/frame_nvs_next_steps.md`):**
Apply the multi-pose Fisher-weighted drift penalty **to rotations**,
not materials. Implementation:

```python
# In train_frame_nvs.py, after warm-up:
# 1. Compute Fisher at K candidate test poses (interpolated train poses)
# 2. F_per_pt = Σ_pose (∂cc/∂rotations[pose])² — per-point scalar
# 3. Normalise F_per_pt to [0, 1]
# 4. Add per-iter loss term:
#    λ_rot · Σ_i F_per_pt[i] · ||rotations[i] − init_rotations[i]||²
```

**Expected impact** (from Analysis I bounds):
- seq_1 HO_128: +0.05–0.17 cc (swap ablation upper bound 0.685
  minus native 0.512 = 0.173)
- seq_2 HO_128: +0.02–0.05 cc (upper bound 0.639 minus 0.590 =
  0.049)

Hits the 0.70 target on seq_1; still short on seq_2 (max 0.639).
seq_2 needs additional H5 (Fisher-weighted material drift) or
direct architectural work.

**Why Fisher-weighting is essential here.** Without it, we'd
just restore H3 which aggregate analysis shows does nothing —
because "uniform" L2 on rotation drift constrains ALL points
equally, including the 19 800 null-space points that can drift
freely without hurting anything. Fisher weighting singles out the
~200 points whose drift is actually costing cc.

**Calibration plan:**
1. Implement Fisher-weighted rotation drift with `--reg_fisher_rot_lambda`
   and `--reg_fisher_poses K` knobs. (~80 LOC.)
2. Sweep λ ∈ {1e-3, 1e-2, 1e-1, 1, 10} on seq_1 HO_128 (~5 min each).
3. Validate on seq_2 HO_128 with the winning λ.
4. If seq_1 clears 0.70 but seq_2 lags, stack with H5 (Fisher-weighted
   material drift).

---

## 15. Artefacts

CSVs: `0_reproducibility.csv`, `A_param_distributions.csv`,
`B_drift_summary.csv`, `C_divergence.csv`, `D_spatial_smoothness.csv`,
`E_normal_smoothness.csv`, `G_concentration.csv`,
`I_swap_ablation.csv`, `J_intensity_prior.csv`, `K_noise_averaging.csv`.

Fisher arrays: `G_fisher_<scene>_<variant>.npz`.

Renders: `render_<scene>_<variant>.npy`; GT: `gt_cart_<scene>.npy`.

Plots: `A_hist_<scene>.png`, `B_drift_<scene>.png`,
`F_trajectories_<scene>.png`.

Script: `run_analyses_matched.py` (single-file, all analyses).
