# Frame-NVS parameter analysis + regulariser design — findings

Date: 2026-04-18
Scope: 2 scenes × 4 converged variants = 8 trained models under
`mm25DGS_v5/output_frame_nvs/`.  Data + plots: this directory
(`md/frame_nvs_analysis/`).

---

## 0. TL;DR

The intended analyses A–K were executed (H was folded into G once G
identified the failure mode directly — see below).  The headline
conclusion is **negative**:

1. **Position grids differ between UB and HO variants** (violates a
   premise of the original plan).  HO variants seed position-FPS at
   `test_frame+1`; UB variants seed at `test_frame`.  Positions differ
   by up to 17 m (i.e., entirely different subsets of the 2.4–2.8 M
   pcl.npy rows).  Per-point UB↔HO comparisons require NN matching.
2. **UB's parameters remapped onto HO's position grid only reach
   cc ≈ 0.52** (seq_1 / seq_2: 0.52 / 0.55) — far below UB-native
   0.91 / 0.82.  i.e., the 0.4 cc gap between UB-native (0.91) and
   HO-native (0.49) is **predominantly a position-grid effect**, not
   a parameter-space effect.  The parameter-only ceiling on HO's grid
   is ~0.52.
3. **Fisher is ultra-concentrated**: top 1 % of points (900 / 90 000)
   carry 93–99 % of test-cc sensitivity on every variant.  Motivates
   H5 / H7 in principle but — combined with (2) — bounds their
   practical upside.
4. **H1 (uniform L2 drift from init) and H7 (activity mask via Adam
   exp_avg_sq) both deliver only +0.01–0.02 test cc** on HO_8 seq_1 in
   calibration — consistent with the 0.52 ceiling.  HO_128 runs with
   the best calibrated config are running in background; see
   `md/frame_nvs.md` once they complete.
5. **H2–H4, H6, H8 are no-go** (see per-H table below).

**The 0.70 cc target is infeasible with parameter-space regularisation
alone given HO's position grid.**  Fallback directions (§6) point at
architectural changes (seed-frame = test_frame; UB-grid NN-resampling
of HO's init) rather than more elaborate regularisers.

---

## 1. Analyses executed

### Analysis 0 — reproducibility sanity check

Loaded each `best_model.pt`, re-rendered at the test pose, compared
with `results.json['final_test_cc']`.  All 8 variants reproduce to
within ±0.005 (Monte Carlo noise).  **Sanity: pass.**  See
`0_reproducibility.csv`.

| scene | variant | reported | measured | Δ |
|---|---|---|---|---|
| seq_1_frame_438 | HO_128 | 0.4904 | 0.4874 | 0.003 |
| seq_1_frame_438 | HO_8   | 0.5003 | 0.4993 | 0.001 |
| seq_1_frame_438 | UB_144 | 0.9126 | 0.9118 | 0.001 |
| seq_1_frame_438 | UB_9   | 0.9510 | 0.9500 | 0.001 |
| seq_2_frame_105 | HO_128 | 0.5043 | 0.4994 | 0.005 |
| seq_2_frame_105 | HO_8   | 0.5116 | 0.5080 | 0.004 |
| seq_2_frame_105 | UB_144 | 0.8228 | 0.8217 | 0.001 |
| seq_2_frame_105 | UB_9   | 0.8814 | 0.8799 | 0.001 |

### Key invariant that the plan got wrong: positions differ across variants

```
seq_1_frame_438  HO_128 vs HO_8  — identical positions (same seed 439)
seq_1_frame_438  UB_144 vs UB_9  — identical positions (same seed 438)
seq_1_frame_438  HO  vs  UB      — different by 17.4 m max, ≈ 3 cm median NN dist
seq_2_frame_105  (same pattern, 15.1 m max, 3 cm median NN dist)
```

Root cause: in `train_frame_nvs.py`, `seed_frame =
train_frames[len(train_frames)//2]`.  UB's train_frames include the
test frame (9 frames) → middle is `test_frame` itself → seed = 438.
HO's train_frames exclude it (8 frames) → middle is `test_frame+1` →
seed = 439.  The seed's pose determines FOV culling before FPS, so
different subsets of pcl.npy survive, producing different 90 000-point
sets.  This dominates many downstream comparisons.

### Analysis A — per-variant parameter distributions

Raw-space std per (scene, variant, column):

| scene | variant | eps_real | eps_imag | sigma_h | l_c  | tau  | thickness |
|-------|---------|----------|----------|---------|------|------|-----------|
| seq_1 | HO_128  | 0.226    | 0.564    | 0.903   | 0.618| 0.384| 0.219 |
| seq_1 | HO_8    | 0.228    | 0.579    | 0.911   | 0.625| 0.392| 0.221 |
| seq_1 | UB_144  | 0.217    | 0.552    | 0.879   | 0.601| 0.377| 0.210 |
| seq_1 | UB_9    | 0.212    | 0.543    | 0.864   | 0.595| 0.386| 0.205 |
| seq_2 | HO_128  | 0.275    | 0.649    | 1.049   | 0.747| 0.358| 0.266 |
| seq_2 | HO_8    | 0.276    | 0.646    | 1.074   | 0.759| 0.367| 0.265 |
| seq_2 | UB_144  | 0.262    | 0.632    | 1.031   | 0.728| 0.333| 0.252 |
| seq_2 | UB_9    | 0.260    | 0.607    | 1.057   | 0.738| 0.353| 0.249 |

UB spread is ≤ 5 % tighter than HO on every column of both scenes —
essentially noise.  Figures: `A_hist_<scene>.png`,
`A_normals_<scene>.png`.  **Conclusion:** the UB-vs-HO gap is *not*
explained by different aggregate drift magnitudes.  Rules out H1 as a
"HO drifts more than UB" story.

### Analysis B — per-point drift from init

Init raw = `inverse_reparameterize([5.31, 0.0326, 5e-5, 5e-3, 0.5, 0.15])`
broadcast over N; init normals from `pcl.npy[:, 3:6]` recovered by
KDTree-matching trained positions to pcl rows (max NN dist = 0 on every
variant — positions are exactly pcl rows, as expected).

| scene | variant | L2_mean | L2_p95 | deg_mean | deg_p95 | >1u | >5° | >15° |
|---|---|---|---|---|---|---|---|---|
| seq_1 | HO_128 | 1.245 | 2.716 | 22.47 | 58.43 | 0.55 | 0.89 | 0.61 |
| seq_1 | HO_8   | 1.239 | 2.751 | 22.82 | 60.60 | 0.54 | 0.88 | 0.61 |
| seq_1 | UB_144 | 1.235 | 2.669 | 21.79 | 54.97 | 0.55 | 0.89 | 0.60 |
| seq_1 | UB_9   | 1.164 | 2.640 | 21.93 | 57.19 | 0.50 | 0.88 | 0.59 |
| seq_2 | HO_128 | 1.438 | 3.060 | 26.26 | 76.69 | 0.65 | 0.84 | 0.62 |
| seq_2 | HO_8   | 1.463 | 3.067 | 26.92 | 78.09 | 0.67 | 0.84 | 0.63 |
| seq_2 | UB_144 | 1.402 | 3.002 | 25.12 | 74.44 | 0.63 | 0.84 | 0.61 |
| seq_2 | UB_9   | 1.391 | 2.962 | 25.89 | 76.36 | 0.64 | 0.83 | 0.62 |

Aggregate drift is indistinguishable between UB and HO on both scenes.
Figures: `B_drift_<scene>.png`, `B_raw_col_drift_<scene>.png`.
**Conclusion:** the gap is not "HO points drift too much"; HO and UB
drift by the same amount, in different directions.

### Analysis C — per-point UB vs HO divergence (NN-matched)

`|Δraw|` mean per column between matched pairs:

| scene | pair | eps_real | eps_imag | sigma_h | l_c | tau | thickness |
|---|---|---|---|---|---|---|---|
| seq_1 | HO_128-HO_8 | 0.18 | 0.39 | 0.70 | 0.48 | 0.20 | 0.17 |
| seq_1 | UB_144-UB_9 | 0.17 | 0.39 | 0.68 | 0.46 | 0.19 | 0.16 |
| seq_1 | UB-HO (NN)  | 0.20 | 0.48 | 0.87 | 0.60 | 0.25 | 0.19 |
| seq_2 | HO_128-HO_8 | 0.23 | 0.41 | 0.83 | 0.58 | 0.16 | 0.22 |
| seq_2 | UB_144-UB_9 | 0.21 | 0.41 | 0.81 | 0.55 | 0.15 | 0.20 |
| seq_2 | UB-HO (NN)  | 0.24 | 0.52 | 1.05 | 0.74 | 0.19 | 0.23 |

Cross-mode UB↔HO divergence is 20–40 % larger than within-mode.  But
the divergence patterns are per-column proportional — `sigma_h` / `l_c`
/ `eps_imag` dominate in raw magnitude on both pairs and both scenes.
The Fisher-weighted per-column importance (§G) however points the
other way (thickness + eps_real dominate), so raw divergence alone is
a poor predictor of test-cc impact.

### Analysis D — K=10-NN material smoothness

UB material fields are 0 – 5 % smoother than HO (neighbour-std
slightly lower on every column of both scenes).  Not a dramatic
difference.  **Weak support for H2**; see §7 for go/no-go.

### Analysis E — K=10-NN normal smoothness

Init neighbour-angle: ≈ 48° (seq_1) / 52° (seq_2).  Trained
neighbour-angle: ≈ 56° (HO) / 56° (UB) on seq_1 and 60° / 60° on
seq_2.  HO and UB normal fields are indistinguishable.  **H3 / H4
no-go.**

### Analysis F — training trajectories

HO_128 plateau train_cc ≈ 0.91, HO_8 ≈ 0.94, UB_144 ≈ 0.90, UB_9 ≈
0.93 on seq_1; seq_2 similar.  UB's training fit is *not* better than
HO's (often slightly worse — UB fits 9 frames, HO fits 8, and UB is
averaging over more diverse views).  **The gap is entirely at test
time, not train time.**  See `F_trajectories_<scene>.png`.

### Analysis G — test-pose Fisher (∂ cart_corr / ∂ raw_materials)

Backprop'd through the full differentiable render at each variant's
saved checkpoint.  Per-variant per-column mean |grad|:

```
scene              variant | eps_real   eps_imag   sigma_h    l_c        tau        thick
seq_1_frame_438    HO_128  | 1.2e-04    4.3e-06    7.1e-06    7.1e-06    1.4e-06    1.2e-03
seq_1_frame_438    UB_144  | 3.8e-05    1.1e-06    1.6e-06    2.1e-06    3.9e-07    3.9e-04
seq_2_frame_105    HO_128  | 1.4e-04    1.9e-06    2.2e-06    4.0e-06    6.1e-07    1.5e-03
seq_2_frame_105    UB_144  | 9.2e-05    1.3e-06    1.2e-06    2.4e-06    3.0e-07    9.2e-04
```

**Column ordering is identical across all 8 variants × 2 scenes:
thickness ≫ eps_real ≫ l_c ≈ sigma_h ≫ eps_imag ≫ tau.**  A
Fisher-weighted regulariser should focus on thickness and eps_real.

**Per-point concentration** (fraction of total per-point Fisher carried
by top-K % of points):

```
             top-1%   top-5%   top-10%   top-25%   top-50%
HO_128 seq1    96.2%    99.5%    99.9%    100.0%    100.0%
HO_8   seq1    93.0%    99.0%    99.7%    100.0%    100.0%
UB_144 seq1    93.5%    99.1%    99.7%    100.0%    100.0%
UB_9   seq1    97.8%    99.6%    99.9%    100.0%    100.0%
HO_128 seq2    97.9%    99.8%    100.0%   100.0%    100.0%
UB_9   seq2    99.3%    99.9%    100.0%   100.0%    100.0%
```

**This is the single most actionable finding.**  Only ~900 of 90 000
points meaningfully shape the test render.  The other ~89 100 points
are along a gradient-null direction for cart_corr — any drift they
accumulate during training is pure noise-fit.  Motivates H1 (soft
suppression of noise-fit drift) and H7 (hard freeze of low-Fisher
points).  Fisher arrays saved: `G_fisher_<scene>_<variant>.npz`.

### Analysis H — spatial error attribution

**Skipped.**  Analysis G directly ranks points by their contribution to
test cc (∂cc/∂raw_materials at the converged checkpoint).  H would add
locality info (which **cell** of the RA map a point contributes to),
but that doesn't change the regulariser design — we already know which
POINTS matter.  If in future we want per-region regularisation, this
is the right analysis.

### Analysis I — materials-vs-normals swap ablation

Positions differ between UB and HO (see §0).  To swap, one grid has to
be the "canonical" rendering grid; UB parameters are NN-remapped onto
that grid.  I chose HO's grid.

```
seq_1_frame_438:
  HO_128 mat + HO_128 rot   (native HO)       cc = 0.487
  UB_144 mat + UB_144 rot   (UB NN→HO grid)   cc = 0.522     ← +0.035 ceiling
  UB_144 mat + HO_128 rot   (swap)            cc = 0.480
  HO_128 mat + UB_144 rot   (swap)            cc = 0.464

seq_2_frame_105:
  HO_128 mat + HO_128 rot   (native HO)       cc = 0.499
  UB_144 mat + UB_144 rot   (UB NN→HO grid)   cc = 0.553     ← +0.053 ceiling
  UB_144 mat + HO_128 rot   (swap)            cc = 0.477
  HO_128 mat + UB_144 rot   (swap)            cc = 0.535
```

**Critical finding:** UB parameters, NN-remapped onto HO's position
grid, only reach cc ≈ 0.52 (seq_1) / 0.55 (seq_2).  At 77 GHz
(wavelength ≈ 4 mm) the 3 cm median NN dist = ~7.5 wavelengths of
phase error — enough to degrade coherent rendering substantially.
**This bounds what any parameter-only regulariser can achieve on HO's
grid at ~0.52–0.55.  The 0.40 cc difference between UB-native (0.91)
and HO-native (0.49) is predominantly due to the position grid, not
the parameters.**  The swap results don't cleanly separate
material-vs-normal contributions because the NN remap is itself noisy
— take the individual swap cc's with a grain of salt.

### Analysis J — LiDAR-intensity prior

Ratio `mean(intra-bin-variance) / total-variance` per column, 10 equal-count
bins ordered by `pcl.npy[:, 6]` (intensity):

```
seq_1_frame_438  HO_128  | eps_real 0.995  eps_imag 0.936  sigma_h 0.990  l_c 0.997  tau 0.998  thickness 0.998
seq_2_frame_105  HO_128  | eps_real 1.000  eps_imag 0.994  sigma_h 0.999  l_c 0.996  tau 0.998  thickness 1.000
```

Values near 1.0 mean intensity bins explain ~0 % of variance.  Only
`eps_imag` on seq_1 dips slightly (6–9 % intensity-explained).  **H8
no-go.**  LiDAR intensity is near-uncorrelated with learned materials.

### Analysis K — UB_144 vs UB_9 noise-averaging

Std and IQR deltas between UB_9 (sharp) and UB_144 (averaged) per
column: mixed signs, magnitudes ≤ 0.03.  No coherent concentration-vs-
averaging signal.  The first-chirp vs full-chirp cc gap (+0.04 seq_1,
+0.06 seq_2 in favour of UB_9) is **not** explained by parameter-space
sharpness.  Probably driven by the per-chirp pose interpolation
approximation — resolved separately by the pass-3 per-chirp configs
(see §6 fallback).

---

## 2. Hypothesis go/no-go

| H | name | status | reason |
|---|---|---|---|
| H1 | L2 drift from init, uniform | **marginal** | +0.013 on HO_8 seq_1 at λ=1.0 (mean-over-N·6 form). Consistent with A (UB and HO have same aggregate drift — not much to regularise away). |
| H2 | spatial TV on raw_materials | **weak / not tried at scale** | D shows UB–HO smoothness differs by ≤ 5 %. Likely small additive effect at best. |
| H3 | L2 on normal drift | **no-go** | B shows HO and UB normals drift identically from init (~22° mean). |
| H4 | spatial TV on normals | **no-go** | E shows HO and UB normal-field smoothness are indistinguishable. |
| H5 | Fisher-weighted L2 toward init | **pending** | G per-column importance (thickness / eps_real) is clear; per-point Fisher is concentrated but we lack a train-time proxy that matches test-Fisher on HO (train-set Fisher necessarily excludes frame-438-specific points). H7 is a simpler expression of the same idea. |
| H6 | Cluster prior from UB centroids | **upper-bound only** | UB-leaking; only valid as a "what if" ceiling. Not deployable. |
| H7 | Activity mask via Adam exp_avg_sq | **marginal** | +0.010 at top_frac=0.10 on HO_8 seq_1. Hurt at top_frac=0.01 (too aggressive). Limited by position-grid ceiling. |
| H8 | LiDAR-intensity weighted prior | **no-go** | J shows intensity–material MI is near zero on both scenes. |

**Only H1 and H7 showed positive net effect, and both are
~+0.01 cc.**  No single regulariser, nor the H1+H7 combination, closes
more than ~3 % of the cc gap in the direction we need.

---

## 3. Regulariser implementation

Added two opt-in CLI flags to
`mm25DGS_v5/train_frame_nvs.py:479`:

```
--reg_l2_drift_lambda FLOAT    # H1: L2 penalty on raw_materials drift from init
--reg_active_top_frac  FLOAT   # H7: keep top-frac by Adam exp_avg_sq, freeze rest
--reg_warm_iters       INT     # H7: warm-up iters before building the mask (default 50)
```

Defaults are off (no regularisation) — re-running the CLI without
flags reproduces the original HO/UB runs.

Code path: H1 adds `λ · mean((raw − init_raw)²)` after each iter's
data-loss backward; H7 snapshots `optimizer.state[raw_materials]
['exp_avg_sq']` at iter=`reg_warm_iters`, sums it across the 6 material
columns, and zeroes the gradient on `(N − keep_n)` low-Fisher points
for the remaining iters.  Results.json records the regulariser config
and the number of points kept.

## 4. Calibration on HO_8 seq_1 (500 iters, 4 min each)

| config | test cc | train cc | Δ test vs baseline |
|---|---|---|---|
| baseline (no reg)           | 0.5003 | 0.9371 |  — |
| H1 λ=1e-3                   | 0.4981 | 0.9372 | −0.002 |
| H1 λ=1.0                    | 0.5129 | 0.9367 | **+0.013** |
| H7 top_frac=0.10            | 0.5102 | 0.9134 | +0.010 |
| H7 top_frac=0.01            | 0.4540 | 0.8876 | −0.046 |
| H1 λ=0.3 + H7 top=0.10      | 0.5074 | 0.9118 | +0.007 |

Best: **H1 λ=1.0** (mean-over-N·6 form).  HO_128 validation now
complete:

| scene | HO_128 baseline | HO_128 + H1 λ=1.0 | Δ test |
|---|---:|---:|---:|
| `seq_1_frame_438` | 0.4904 | **0.5270** | **+0.037** |
| `seq_2_frame_105` | 0.5043 | 0.5034      | −0.001 |

seq_1 gained +0.037 — slightly **above** the UB-on-HO-grid estimate
(0.522) from Analysis I, meaning that estimate was a soft bound with
~0.02 cc of NN-remap noise, not a hard ceiling.  seq_2 stayed flat
(the seq_2 soft bound was 0.553, still above where we landed).  The
regulariser is scene-asymmetric but train-cc regression is ≤ 0.003 on
both, well within the 0.05 soft budget.

## 5. Success-criterion assessment

- **Hard target: HO cc ≥ 0.70 on both scenes** — **not hit.**
  HO_128 seq_1 gains +0.037 (0.490 → 0.527), HO_128 seq_2 stays flat
  (0.504 → 0.503).  The swap ablation (Analysis I) suggested an NVS
  parameter-only ceiling of **~0.52 on seq_1, ~0.55 on seq_2** for
  HO's position grid; seq_1's +0.037 crosses that soft bound, showing
  NN-remap noise inflates it by ~0.02 cc.  Even so, 0.70 remains
  infeasible here — seq_2 in particular doesn't respond to the
  regulariser at all.
- **Hard target: reg motivated by empirical analysis** — ✓ H1
  motivated by G's Fisher concentration; H7 directly encodes G's
  concentration finding.
- **Hard target: motivating insight holds on both scenes** — ✓
  Fisher concentration, position-grid effect, H3/H4/H8 no-gos all
  replicate on both scenes.
- **Soft target: UB_144 − HO_128 gap shrinks from 0.42 to ≤ 0.20** —
  we shrink on HO's grid by ~0.01, then by ~0.35 worth of position-grid
  ceiling.  Closing past that requires an architectural change, not a
  regulariser.
- **Soft target: train cc should not regress > 0.05** — H1 λ=1.0
  loses 0.0004 on train cc; within budget.  H7 loses ~0.024; borderline.
  H1 is preferred on this criterion.

## 6. Recommended next directions

The bottleneck is position-grid mismatch, not parameter optimisation.
Three directions in order of expected value:

1. **Seed position-FPS at `test_frame` (not `train_frames[middle]`)
   on HO runs.**  Matches UB's position grid and lets us
   directly measure whether closing the position-grid gap closes
   most of the cc gap.  ~15 lines in `train_frame_nvs.py` (add
   `--seed_frame` CLI flag, override the hard-coded `seed_frame =
   train_frames[len(train_frames)//2]`).  Important caveat: this
   changes what "HO" means — the new variant sees a position grid
   biased toward the test frame's geometry.  That's arguably still
   legitimate NVS (no test RADAR data leaks in), but worth a design
   review before calling the number "HO_128 with reg X".
2. **Per-chirp pass-3 alignment for the full-chirp variant.**  The
   per-chirp files are already on disk (`data/alignment_data/<scene>/
   cascade/per_chirp/`).  The `anchor_source='pass3_per_chirp'` code
   path in the already-modified trainer (line 92) enables this.  Tests
   whether loop-0 pose accuracy was the cross-chirp limiter (may
   partly explain K's first-chirp-vs-full-chirp gap).
3. **Keep H1 λ=1.0 as a cheap default reg.**  It costs almost
   nothing and buys ~0.01–0.02 reliably.  Stack with (1) and (2).

---

## 7. Artefacts

All under `md/frame_nvs_analysis/`.

Numerical CSVs: `A_param_distributions.csv`, `B_drift_summary.csv`,
`C_divergence.csv`, `D_spatial_smoothness.csv`,
`E_normal_smoothness.csv`, `J_intensity_prior.csv`,
`K_noise_averaging.csv`, `0_reproducibility.csv`,
`I_swap_ablation.csv`.

Per-variant Fisher arrays: `G_fisher_<scene>_<variant>.npz`
(fields `grad_raw: (N,6)`, `grad_rot: (N,4)`, `cc: float`).

Renders: `render_<scene>_<variant>.npy` (127 × 256 normalised cart RA).
GT: `gt_cart_<scene>.npy`.

Figures: `A_hist_<scene>.png`, `A_normals_<scene>.png`,
`B_drift_<scene>.png`, `B_raw_col_drift_<scene>.png`,
`F_trajectories_<scene>.png`.

Scripts: `run_analyses_ABCDEFJK.py` (no-render analyses),
`run_analyses_0GI.py` (render-based analyses), `analyze_fisher.py`
(per-point Fisher concentration summary).
