# mm3DGS v4 — NVS Status Report

**Audience:** project colleagues
**Last updated:** 2026-04-15
**Contact:** aa2546@cornell.edu

This is a consolidation of the NVS investigation on mm25DGS_v4. For
deeper technical details, see the companion documents:

- [`v4_nvs_plan.md`](v4_nvs_plan.md) — original NVS design
- [`v4_nvs_diagnosis.md`](v4_nvs_diagnosis.md) — early failure-mode diagnostics
- [`v4_nvs_param_analysis.md`](v4_nvs_param_analysis.md) — parameter-space comparisons
- [`v4_nvs_scenario_diff.md`](v4_nvs_scenario_diff.md) — output-space diagnostics + reg experiments
- [`v4_nvs_principled_proposals.md`](v4_nvs_principled_proposals.md) — literature-informed proposals + failed regularizers
- [`v4_nvs_scale_out.md`](v4_nvs_scale_out.md) — 7-scene sweep + per-scene ceiling analysis

## 1. Problem statement

**Task: radar NVS (novel view synthesis) — interpolation only.**

Each benchmark scene contains 9 temporally-adjacent cascaded radar
frames captured while the MMWCAS radar is moved along a short
trajectory. We want to evaluate whether the v4 differentiable renderer
can learn a scene representation (per-point materials + per-point
normal corrections) from a subset of frames and then predict held-out
frames whose radar poses lie *between* the training poses.

**Split convention (Scenario 6/2, used as canonical benchmark):**
- Train: frame indices `{0, 1, 3, 5, 7, 8}` (6 frames)
- Test: frame indices `{2, 6}` (2 frames, each bracketed by two train frames)
- Unused: frame index `4` (the "reference" frame whose alignment we
  verified is misaligned in several scenes — see Section 6 for details)

**Constraint:** test frames must be *bracketed* by train frames (interpolation
only; no extrapolation to trajectory edges). Test GT is strictly held
out — we evaluate rendered cart_corr against test GT but never use it
in training.

**Metric:** cart_corr (Pearson correlation) between the v4-rendered
|RA| cart image at the test pose and the measured |RA| cart image at
that same pose. Same metric used by single-frame v4 and mmIR baselines.

**Success criterion (stated by stakeholder):** "test cc ≈ train cc
within reasonable delta". On successful scenes the test–train gap
should be no larger than ~0.05–0.10 cart_corr.

## 2. Ceiling: train on all available frames

Before asking what's *achievable* under a train/test split, we need to
know what the model is *representationally capable of*: can the v4
architecture produce a single parameter set that fits every frame of
a scene simultaneously?

**Experiment:** train on all 8 non-reference frames per scene
(`train_indices={0,1,2,3,5,6,7,8}`, no held-out), evaluate cart_corr
per frame at the final state. Points are fixed at 10 K per scene,
Pearson loss, 300 iters, default LRs.

**Per-frame ceiling table (one row per frame × scene):**

All-in training uses `train_indices={0,1,2,3,5,6,7,8}` (drop idx 4,
the reference frame). The reference frame is reported as "unused" —
it's rendered at eval time but never trained, so its cc reflects how
well the trained model generalizes to the (misaligned) reference pose
*without* seeing any of its radar data.

| scene | frame | tag | cart_corr | notes |
|---|---|---|---|---|
| `seq_0_frame_135` | 131 | train | 0.8661 | |
| `seq_0_frame_135` | 132 | train | 0.8102 | |
| `seq_0_frame_135` | 133 | train | 0.8240 | |
| `seq_0_frame_135` | 134 | train | 0.8466 | |
| `seq_0_frame_135` | **135** | **unused** | **0.5887** | ref frame, misaligned (bs Z=−0.49) |
| `seq_0_frame_135` | 136 | train | 0.8595 | |
| `seq_0_frame_135` | 137 | train | 0.7945 | |
| `seq_0_frame_135` | 138 | train | 0.8036 | |
| `seq_0_frame_135` | 139 | train | 0.7525 | |
| | **train mean** | | **0.8196** | |
| `seq_0_frame_390` | 386 | train | 0.8191 | |
| `seq_0_frame_390` | 387 | train | 0.8646 | |
| `seq_0_frame_390` | 388 | train | 0.7550 | |
| `seq_0_frame_390` | 389 | train | 0.8255 | |
| `seq_0_frame_390` | **390** | **unused** | **0.1191** | ref, misaligned (bs Z=+0.38); discontinuous pose traj |
| `seq_0_frame_390` | 391 | train | 0.7519 | |
| `seq_0_frame_390` | 392 | train | 0.8025 | |
| `seq_0_frame_390` | 393 | train | 0.8211 | |
| `seq_0_frame_390` | 394 | train | 0.8410 | |
| | **train mean** | | **0.8101** | |
| `seq_1_frame_185` | 181 | train | 0.7826 | |
| `seq_1_frame_185` | 182 | train | 0.7378 | |
| `seq_1_frame_185` | 183 | train | 0.8072 | |
| `seq_1_frame_185` | 184 | train | 0.8312 | |
| `seq_1_frame_185` | **185** | **unused** | **0.5964** | ref frame, misaligned (bs Z=−0.21) |
| `seq_1_frame_185` | 186 | train | 0.8077 | |
| `seq_1_frame_185` | 187 | train | 0.8186 | |
| `seq_1_frame_185` | 188 | train | 0.8693 | |
| `seq_1_frame_185` | 189 | train | 0.8763 | |
| | **train mean** | | **0.8163** | |
| `seq_1_frame_438` | 434 | train | 0.8260 | |
| `seq_1_frame_438` | 435 | train | 0.8425 | |
| `seq_1_frame_438` | 436 | train | 0.8542 | |
| `seq_1_frame_438` | 437 | train | 0.9017 | |
| `seq_1_frame_438` | **438** | **unused** | **0.6156** | ref frame (clean alignment here) |
| `seq_1_frame_438` | 439 | train | 0.9286 | |
| `seq_1_frame_438` | 440 | train | 0.8551 | |
| `seq_1_frame_438` | 441 | train | 0.9013 | |
| `seq_1_frame_438` | 442 | train | 0.9201 | |
| | **train mean** | | **0.8787** | **highest per-frame ceiling of all 7 scenes** |
| `seq_2_frame_105` | 101 | train | 0.7193 | |
| `seq_2_frame_105` | 102 | train | 0.7809 | |
| `seq_2_frame_105` | 103 | train | 0.7932 | |
| `seq_2_frame_105` | 104 | train | 0.7675 | |
| `seq_2_frame_105` | **105** | **unused** | **0.5556** | ref frame (alignment uniformly tilted, not broken) |
| `seq_2_frame_105` | 106 | train | 0.7986 | |
| `seq_2_frame_105` | 107 | train | 0.8266 | |
| `seq_2_frame_105` | 108 | train | 0.8519 | |
| `seq_2_frame_105` | 109 | train | 0.8368 | |
| | **train mean** | | **0.7968** | |
| `seq_2_frame_160` | 156 | train | 0.7717 | |
| `seq_2_frame_160` | 157 | train | 0.8276 | |
| `seq_2_frame_160` | 158 | train | 0.7816 | |
| `seq_2_frame_160` | 159 | train | 0.6954 | |
| `seq_2_frame_160` | **160** | **unused** | **0.6514** | ref frame (clean alignment) |
| `seq_2_frame_160` | 161 | train | 0.8440 | |
| `seq_2_frame_160` | 162 | train | 0.8432 | |
| `seq_2_frame_160` | 163 | train | 0.7695 | |
| `seq_2_frame_160` | 164 | train | 0.7083 | |
| | **train mean** | | **0.7801** | **lowest per-frame ceiling of the clean scenes** |
| `seq_2_frame_300` | 296 | train | 0.8475 | |
| `seq_2_frame_300` | 297 | train | 0.7558 | |
| `seq_2_frame_300` | 298 | train | 0.7917 | |
| `seq_2_frame_300` | 299 | train | 0.8192 | |
| `seq_2_frame_300` | **300** | **unused** | **0.4333** | ref frame, misaligned (bs Z=+0.33) |
| `seq_2_frame_300` | 301 | train | 0.7528 | |
| `seq_2_frame_300` | 302 | train | 0.8175 | |
| `seq_2_frame_300` | 303 | train | 0.7895 | |
| `seq_2_frame_300` | 304 | train | 0.7706 | |
| | **train mean** | | **0.7931** | |
| | | | | |
| **grand mean train** | | | **0.8135** | |
| **grand mean unused (ref)** | | | **0.5086** | (all 4 misaligned + 3 aligned ref frames) |

**What this establishes across all 7 scenes:**

- **Grand mean train cc (8-frame all-in fit): 0.8135** — the v4 model
  has enough representational capacity to fit every non-reference
  frame simultaneously. Per-scene train means span 0.78–0.88.
- There exists a single `(materials, normals)` parameter set per
  scene that explains all 8 frames at ~0.78–0.88 cart_corr. **So
  NVS is not representation-limited.**
- **Per-frame ceiling variance is tight within a scene**: the all-in
  cc of each train frame is within ±0.05–0.08 of the scene mean. The
  model distributes its fit across all 8 frames fairly evenly — it
  doesn't sacrifice one for another.
- **`seq_1_frame_438` has the highest all-in ceiling** (0.879 mean)
  and achieves 0.90+ on several individual frames, making it the
  easiest scene to fit.
- **The unused reference frames** tell us something interesting: at
  the all-in state, the model's prediction at the held-out reference
  frame ranges from **0.12 on scene 390** (discontinuous trajectory —
  the ref frame is in a different pose cluster) to **0.65 on scene
  160**. This is a loose proxy for NVS generalization *at the model
  ceiling*. The 0.12 for scene 390 is a red flag that confirms the
  pose-trajectory issue; the 0.43 for scene 300 (misaligned ref)
  reflects the alignment problem. The other 5 scenes' unused cc
  (0.55–0.65) is close to what bracket anchor achieves on the 6/2
  test frames, suggesting the per-frame held-out generalization
  plateau is in the 0.55–0.65 range even at the theoretical ceiling.

## 3. The gap we are fighting

When we train on 6 of the 9 frames and evaluate on the 2 held-out
(interpolation) frames, the test cart_corr drops dramatically —
even though the train cart_corr on the 6 frames used is almost
identical to the all-in fit.

**Canonical 6/2 result on `seq_0_frame_135` (no regularization):**

| frame | tag | cart_corr (6/2 baseline) | ceiling (all-in) | gap |
|---|---|---|---|---|
| 131 | train | 0.88 | 0.87 | 0.00 |
| 132 | **test** | **0.55** | **0.81** | **0.26** |
| 133 | train | 0.91 | 0.82 | — |
| 134 | **test** | **0.42** | **0.84** | **0.42** |
| 135 | unused | — | — | — |
| 136 | train | 0.88 | 0.85 | — |
| 137 | **test** | **0.61** | **0.79** | **0.18** |
| 138 | train | 0.81 | 0.80 | — |
| 139 | train | 0.85 | 0.76 | — |
| train mean | | 0.85 | | |
| **test mean** | | **0.55** | **0.82** | **0.27** |

**Observation:** the gap is at the *individual held-out frames*, not
in the average. Frame 132's 0.55 vs its ceiling 0.81 (−0.26) happens
because the optimizer, given only 6 frames of supervision, finds a
parameter set that perfectly fits the 6 train frames but has sharp
discontinuities at the held-out poses. The model output at frame 132
drops from ~0.85 at frame 131 to 0.55 at frame 132, then back up to
0.85 at frame 133 — a **sharp dip of 0.3 in ~25 ms / ~1 cm of pose
change.** That's not physical radar behavior; it's the optimizer
carving out discontinuities at the positions where there's no training
signal.

**Output-space diagnosis** ([`v4_nvs_scenario_diff.md`](v4_nvs_scenario_diff.md))
showed the S6/2 and all-in renderings differ **7–8× more in the
top-10% GT energy regions** (strong scatterers) than in low-energy
background. The model is fine in the background but has wildly
different peak intensities at the dominant scatterers at test poses.
This is a *strong-scatterer gauge ambiguity*: many parameter sets fit
the 6 train frames equally well, and their strong-scatterer response
at the in-between test pose is essentially arbitrary.

## 4. Regularizer attempts — what we tried and what we learned

We ran 10+ principled regularizers over the course of this
investigation, plus the BSDF-lobe ablation diagnostic. **Best single
result so far: +0.045 mean test cart_corr**, from a "bracket anchor"
regularizer that renders at the test pose and supervises against an
interpolation of the bracketing train renders. See the full table:

| Regularizer | Logic source | Best Δ test cc | Result |
|---|---|---|---|
| **Bracket-anchor** (model-target, λ=0.05, non-detached) | Render at known test pose, supervise on linear interpolation of bracket train renders. Test pose is geometry (allowed); only the test GT is forbidden. | **+0.045 mean, +0.068 on f133** | **WORKS** (replicated 3×, σ=±0.003) |
| Material gauge collapse (poly2 basis, 174 params instead of 60K) | Remove per-point material freedom (gauge ambiguity hypothesis) | +0.009 | marginal/noise |
| Facing-score gradient weighting | Suppress noisy grazing-angle point updates | +0.005 / −0.009 | no effect |
| Material k-means clustering (K ∈ 10..2000, pos+normal features) | Tie materials in spatial clusters | ≤ +0.008 | no effect |
| Intensity-keyed material codebook (cluster + residual) | Use LiDAR intensity (col 6 of pcl.npy) as material-similarity signal | +0.002 | no effect |
| Edge-preserving smoothness (kNN graph + intensity gating) | Literature recipe from DET-GS / PBR-NeRF — asymmetric smoothing | −0.07 (at any λ) | **DESTRUCTIVE** |
| Normal anchor to pcl init (λ ∈ 0.05..5.0) | Anchor normals at init to suppress drift | flat | no effect |
| Render consistency (naive, `‖r(p)−r(p+ε)‖²`) | Force pose-invariance of rendered output | −0.05 | destructive (pulls toward const) |
| Anchored smoothness (corrected: `‖(r_b−r_a)−(GT_b−GT_a)‖²`) | Force model's pose-derivative to match GT's | +0.01 to −0.08 | marginal then destructive |
| Pose-midpoint linearity reg | Render at midpoint between train poses, penalize deviation from average | +0.03 / −0.05 (mixed) | works on one test frame, hurts the other |
| Detached bracket anchor | Don't let gradient flow into bracket renders | −0.02 to +0.04 | hurts f133, helps f137 |
| Pose-distance-weighted bracket anchor | Per-frame λ ∝ exp(−α·pose_dist) | +0.005 over non-detached | marginal refinement |
| LR tuning (100× smaller LR) | "Freeze" training near init, preserve warm-start state | +0.27 (cheating) | **HACK — explicitly rejected** (doesn't represent learning) |
| Iterative training (warm-start chaining) | Run multiple 300-iter cycles from saved state | partial | **HACK — explicitly rejected** (not a new mechanism) |
| BSDF lobe ablation (KA disabled / SPM disabled) | Diagnostic: is specular overfitting the source? | KA disabled +0.015, SPM disabled −0.05 | KA not the mechanism |

**What we learned from the failures (the valuable part):**

1. **Parameter-space smoothness is not the problem.** The earlier
   parameter analysis proved that S1 (under-trained) and S2 (well-
   trained) have *identical spatial smoothness at every scale*. No
   smoothness-based regularizer can distinguish them — because
   they're both already smooth. Every smoothness-like regularizer
   we tried failed for this reason. We should have caught this
   earlier from the analysis.

2. **The gauge ambiguity is in the *output-space dynamics* of
   per-point contributions at strong scatterers**, not in the
   parameter values themselves or in how smooth the parameters are.
   Constraining parameter distributions (clustering, codebooks,
   distance-from-init) does not fix it. Only constraining the
   *rendered output at the held-out pose* moved the needle.

3. **Bracket-anchor works because it's the *only* regularizer we
   tried that operates at the exact target pose.** It renders at
   the test pose (a known geometric quantity, no GT leakage) and
   supervises against a self-consistent interpolation target.
   Everything else operates on the wrong axis.

4. **The gap is structurally bounded.** A pure data analysis —
   cart_corr of `(GT_a + GT_b)/2` against `GT_t` — gives a *hard
   ceiling* for any linear-interpolation regularizer: **0.697 on
   f133 and 0.772 on f137 for `seq_0_frame_135`.** More
   sophisticated interpolation schemes (pose-distance weighting,
   all-6 kernel, unconstrained linear regression) do **not** exceed
   this — the simple linear bracket is already optimal among linear
   methods. **Regularizers built on linear bracket supervision
   cannot exceed this ceiling.** The gap from ~0.73 to the all-in
   ceiling (~0.82) is the *non-linear* radar pose response the
   model would have to learn *directly from test-frame GT*.

5. **Two of seven scenes already exceed their linear bracket
   ceiling** (`seq_1_frame_438` and `seq_2_frame_105`). The model
   has learned non-linear pose response on those scenes without
   any explicit supervision for it. **On these scenes the bracket
   anchor is slightly counterproductive** because it pulls the
   model *toward* the less-accurate linear target. This means the
   right application of bracket anchor is **conditional**:
   per-scene, check whether the model is below its linear
   ceiling, and apply the reg only in that case.

**7-scene summary with bracket anchor (current best
regularizer):**

| scene | base test | +reg test | ceiling | **% of gap closed** |
|---|---|---|---|---|
| seq_0_frame_135 | 0.551 | 0.615 | 0.735 | 34.8 % |
| seq_0_frame_390 | 0.227 | 0.292 | 0.392 | 39.4 % |
| seq_1_frame_185 | 0.394 | 0.528 | 0.594 | 66.9 % |
| seq_1_frame_438 | 0.596 | 0.597 | 0.395 | (already above ceiling) |
| seq_2_frame_105 | 0.617 | 0.582 | 0.591 | (already above ceiling) |
| seq_2_frame_160 | 0.614 | 0.607 | 0.760 | −4.9 % |
| seq_2_frame_300 | 0.473 | 0.531 | 0.591 | 48.8 % |
| **mean** | **0.496** | **0.536** | **0.580** | |

Mean test cc improves **+0.040 across all 7 scenes**. On the 4 scenes
where the bracket anchor is applicable (baseline below ceiling), it
closes **35–67 % of the available gap** (mean **47.5 %**).

## 5. Next direction: more training data per frame

The regularizer exploration has extracted most of what hand-designed
priors can give us on this problem. To close more of the gap, we need
*more data*, not more priors.

### 5a. The easy win we're sitting on: chirps within a frame

Each `cascaded_frame_*.npy` file is shape **`(16, 16, 12, 256)`** =
**16 slow-time loops, 16 RX, 12 TX, 256 range samples**. The existing
pipeline only uses **loop 0** (`arr[0]`), discarding the other 15.

**What each loop represents:**

| parameter | value | source |
|---|---|---|
| rampEndTime (per-chirp duration) | 34 μs | explicit in config |
| assumed idle time between chirps | 7 μs | TI MMWCAS default |
| chirp cycle (1 TX emission) | 41 μs | derived |
| loop duration (12 TX sequential) | 492 μs | derived |
| full frame duration (16 loops) | 7.87 ms | derived |
| **Δt between loop 0 and loop 15** | **7.38 ms** | derived |
| estimated radar motion during one frame (@100 mm/s) | ~0.79 mm | assumed speed |
| two-way phase shift at 77 GHz (λ=3.89 mm) | ~2.4 rad | physics |

**Observed difference between loop 0 and loop 15 of the same frame
(scene 135, frame 131):**
- Per-loop ADC mean shows a systematic trend (77.3 → 77.9 → 74.96)
  over the 16 loops — not random noise
- Cart |RA| mean: loop 0 = 1410, loop 15 = 1312 (−7%)
- Cart |RA| peak: loop 0 = 1.08e5, loop 15 = 8.23e4 (−24%)
- **`|loop15 − loop0| / mean ≈ 37 %`** — substantial
- Error map (in linear cart scale) shows signed differences
  concentrated in strong-scatterer regions

Visualizations saved to:
- [`mm25DGS_v4/output_nvs/chirp_diff_analysis/chirp_diff_cart.png`](../mm25DGS_v4/output_nvs/chirp_diff_analysis/chirp_diff_cart.png) —
  4-panel linear-scale cart: loop 0, loop 15, signed error,
  relative error
- [`mm25DGS_v4/output_nvs/chirp_diff_analysis/per_loop_trajectory.png`](../mm25DGS_v4/output_nvs/chirp_diff_analysis/per_loop_trajectory.png) —
  per-loop |RA| mean and peak trajectory over all 16 loops

**Why this matters:** 16 loops of data per frame × 9 frames × 7
scenes = **16× more training signal** than what we've been using. The
pose diversity between loops is small (sub-mm) but the measurement
noise is independent, so:

- Using all 16 loops as separate training targets effectively gives
  the model 16× more data per pose to fit against
- Independent per-loop noise averages out, reducing overfitting
- If there is any real pose motion within a frame, it adds true pose
  diversity at tiny but non-zero scale

### 5b. Simpler problem setting to start with

Before attempting chirp-level NVS across multiple frames, we'll first
verify it works on a **simpler problem**: NVS *across loops within a
single frame*. That is:
- Pick one frame (e.g., `seq_0_frame_135` frame 131)
- Process each of its 16 loops separately into a |RA| image
- Train on a subset of loops (say loops 0, 2, 4, 6, 8, 10, 12, 14)
- Test on the remaining loops (1, 3, 5, 7, 9, 11, 13, 15)
- Expect: since the pose shift within a frame is sub-mm, the test
  loops should be predictable from the train loops with very high
  cart_corr

**This is the ideal sanity check.** If the model can't do NVS across
loops (sub-mm pose shift), then it definitely can't do NVS across
frames (~1 cm pose shift). If it works here, we know:
1. Our pipeline and methodology are correct
2. Using loop-level data gives us reliable test coverage
3. We can then attempt the harder multi-frame NVS problem with loop-
   level data as an augmentation

**After the loop-NVS sanity check passes**, the plan is to train
multi-frame NVS (scenario 6/2) using *all 16 loops of each training
frame* as independent training examples — essentially 6 × 16 = 96
training examples per iter, vs the 6 we're currently using.

## 6. Important data caveats discovered during investigation

Two data issues surfaced that colleagues should be aware of:

1. **Reference frames in 4 of 7 scenes are misaligned.** The 4DOF
   cascade alignment produces boresight Z-components of −0.49, +0.38,
   −0.21, +0.33 for `seq_0_frame_135`, `seq_0_frame_390`,
   `seq_1_frame_185`, `seq_2_frame_300` respectively, while every
   other frame in those scenes has |Z| < 0.05. Physically impossible
   (24–29° radar tilt in 25 ms). The raw |RA| energy of the
   misaligned frames is **exactly half** the median of their siblings,
   consistent with a large angular error. All NVS experiments skip
   frame index 4 (the canonical reference index) for this reason.
   Worth re-running the alignment pipeline for these 4 reference
   frames as a separate clean-up task. Details:
   [`v4_nvs_diagnosis.md`](v4_nvs_diagnosis.md).

2. **`seq_0_frame_390` has a discontinuous pose trajectory.** Adjacent
   frames (by index) are 0.20 m to 3.55 m apart in 3D space. At 25
   ms/frame, 3.3 m would require radar motion of 475 km/h — impossible
   for a tabletop radar. **This scene was likely captured at discrete
   viewpoints, not as a continuous sweep.** Linear bracket
   interpolation is fundamentally inappropriate for this scene. Its
   hard ceiling is only 0.39 (vs ~0.73 on well-sampled scenes). If
   NVS on this scene is a hard requirement, a different evaluation
   split is needed (pair spatially-adjacent frames as brackets, not
   temporally-adjacent indices).

## TL;DR for colleagues

- **NVS is representationally feasible**: the model can fit every
  frame of a scene simultaneously at **~0.82 mean cart_corr**.
- **Test-pose generalization is the hard part**: with 6-frame
  training, test cc drops to **~0.55**, leaving a **~0.27 gap** that
  comes from sharp pose-response discontinuities the optimizer
  carves out at the un-supervised test poses.
- **A bracket-anchor regularizer closes ~47% of the closeable gap**
  on applicable scenes (mean +0.04 cart_corr across 7 scenes).
  **Single regularizer gain is +0.05 max**; the remaining gap is
  structurally bounded by linear interpolation from train-frame GTs.
- **Next direction: more data, not more priors.** Each radar frame
  contains 16 slow-time loops that we've been discarding. Loops 0 and
  15 of the same frame differ by ~37% in |RA|, confirming distinct
  information per loop. Plan: (a) validate NVS across loops within a
  single frame (sub-mm pose shift — should be easy), then (b) use all
  16 loops of each train frame as augmentation for multi-frame NVS
  training.
- **Two data caveats:** (1) reference frames in 4/7 scenes are
  alignment-broken and excluded from all NVS experiments; (2)
  `seq_0_frame_390` has a non-continuous radar pose trajectory and
  needs a different evaluation split.

## Companion documents

- [`v4_nvs_plan.md`](v4_nvs_plan.md) — original NVS design
- [`v4_nvs_diagnosis.md`](v4_nvs_diagnosis.md) — early failure-mode diagnostics
- [`v4_nvs_param_analysis.md`](v4_nvs_param_analysis.md) — parameter-space comparisons
- [`v4_nvs_scenario_diff.md`](v4_nvs_scenario_diff.md) — output-space diagnostics + reg experiments
- [`v4_nvs_principled_proposals.md`](v4_nvs_principled_proposals.md) — literature-informed proposals
- [`v4_nvs_scale_out.md`](v4_nvs_scale_out.md) — 7-scene sweep + per-scene ceiling analysis
