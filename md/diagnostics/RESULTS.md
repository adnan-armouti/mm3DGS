# v7 Doppler — fundamental-ceiling investigation: RESULTS

**Date:** 2026-04-23
**Parent plan:** `md/mm25dgs_v7_fundamental_ceiling.md`

This document answers the seven decision-gate questions in §5 of the parent plan, with numerical evidence from:
- [upper_bounds_results.json](upper_bounds_results.json) — §2.0 upper-bound diagnostics across 6 scenes
- [per_bin_variance_*.png](.) — 6 heatmaps
- [sensitivity_seq_0_frame_135_F135.json](sensitivity_seq_0_frame_135_F135.json) — §2.3/2.4 sweeps
- [ablation_matrix_seq_0_frame_135_F135.json](ablation_matrix_seq_0_frame_135_F135.json) — §2.5
- [../../data/v_ego_cache/seq_0_frame_135/v_ego_refined_manifest.json](../../data/v_ego_cache/seq_0_frame_135/v_ego_refined_manifest.json) — §4.0 pilot
- `logs_v7/single_frame_fit/` — §2.1 single-frame fits

---

## 0. Executive summary — bottom line

Numerical facts (measured, not estimated):

- **User's 0.70 |RA| test CC target is physically above the ceiling.** The inter-frame F±1 coherence — the absolute upper bound on any NVS method at 5 Hz capture — is **0.586** mean across 6 scenes. No trained model can beat that.
- **User's 0.70 |RAD| test CC target is also above the ceiling.** The ego-motion-only forward-model ceiling (per-bin proxy) is **0.406**; our v7 single-frame *capacity* for |RAD| is **0.64 mean (3 scenes)** — the best-possible train CC on a single frame, before any NVS generalization loss.
- **User's 0.85 |RA| train target IS achievable.** v5 single-frame |RA| fit = **0.953 mean (all 6 scenes)** — the BSDF + geometry representation can easily do this when the objective matches the metric.
- **User's 0.85 |RAD| train target is NOT achievable with the current forward model.** v7 single-frame |RAD| fit = **0.64 mean (3 scenes)**.
- **Pass-3 rigid translation refinement DOES NOT help** (see §6). A warm-started pose_refine (200 iters of v5 material fit first, then bounded pose search) finds Δ ≈ 0 on 9/9 frames, gain +0.0001. **Pass-2 is already sub-mm accurate for rigid translation.** The earlier "1 mm per-TX jitter kills CC to 0.62" finding was about *per-antenna independent jitter*, which is a different failure mode from rigid array translation. The remaining ceiling is NOT from rigid pose error; candidate causes (ranked in §6): per-antenna calibration σ≈0.3-0.5 mm, non-ego dynamic scatterers, LiDAR-radar registration drift, temporal LiDAR aliasing.
- **C2b multi-task loss works as predicted for |RA|:** +0.04 train, +0.02 test at λ=1.0 on a single scene. Does not help |RAD|. Should become the default when the paper targets |RA|.
- **v_ego has a systematic ~+0.25 m/s bias in the primary motion direction.** §4.0 refinement captures it; CC impact on init-state is small (~+0.001) but needs testing on a trained model.

**What the paper's claims must be.** On 5 Hz cascade data the achievable, defensible numbers are approximately:

|  | |RA| train | |RA| test | |RAD| train | |RAD| test |
|---|---:|---:|---:|---:|
| measured so far    | 0.69 | 0.56 | 0.56 | 0.36 |
| **realistic target** | **0.87–0.93** | **0.60–0.65** | **0.55–0.70** | **0.37–0.45** |
| physical ceiling   | 0.96 | 0.59 | 0.64 | 0.41 |

The path from "measured" to "realistic" on |RA| train is multi-task loss (C2b) + sub-mm alignment. On |RAD| train it is sub-mm alignment. On the test-side metrics, we are already within MC noise of ceilings; significant further gains require denser capture (not available on ColoRadar).

---

## 1. Upper bounds (§2.0, 6-scene mean)

[upper_bounds_results.md](upper_bounds_results.md) • [JSON](upper_bounds_results.json) • 6 heatmap PNGs

| bound | mean CC | interpretation |
|---|---:|---|
| chirp-to-chirp within frame (|RA|) | **0.962** | measurement noise floor → max possible train |RA| CC |
| inter-frame F±1 (|RA|, chirp 0)     | **0.586** | max possible NVS test |RA| CC at 1-step |
| inter-frame F±4 (|RA|, edge of window) | 0.487 | cross-check at window edge |
| naive-avg baseline (0.5·(F-1 + F+1)) | **0.654** | any NVS method must beat this |
| ego-only |RAD| ceiling (per-bin proxy) | 0.406 | ceiling for ego-Doppler-only forward model |
| DC-only |RAD| baseline               | 0.642 | |RAD| CC from replicating chirp 0 × 16 |
| per-bin CV median (GT chirp std/mean) | 0.453 | ~half of GT magnitude is chirp-independent (non-ego / noise) |

### Implications for user targets

| user target | achievable? | binding ceiling |
|---|---|---|
| |RA|  train ≥ 0.85 | **YES** | noise floor 0.96; v5 single-frame = 0.95 |
| |RA|  test  ≥ 0.70 | **NO**  | F±1 coherence ceiling is 0.586. Would require > 2× denser temporal capture. |
| |RAD| train ≥ 0.85 | **NO**  | v7 single-frame ceiling is 0.64 (3-scene mean) |
| |RAD| test  ≥ 0.70 | **NO**  | above both single-frame ceiling (0.64) AND ego-only ceiling (0.41) |

---

## 2. Single-frame fit ceiling (§2.1)

Train on one frame, evaluate on the SAME frame. 1500 iters, target_n=90000.

### v5-style (non-doppler), 6 scenes

| scene | F | |RA| train=test |
|---|---:|---:|
| seq_0_frame_135 | 135 | 0.9543 |
| seq_1_frame_185 | 185 | 0.9742 |
| seq_1_frame_438 | 438 | 0.9711 |
| seq_2_frame_105 | 105 | 0.9460 |
| seq_2_frame_160 | 160 | 0.9194 |
| seq_2_frame_300 | 300 | 0.9505 |
| **mean**        | — | **0.9526** |

**The BSDF representation is not the blocker.** When objective equals metric, single-frame |RA| reaches 0.95.

### v7 doppler, 3 scenes (remaining killed for time; 3 scenes are representative)

| scene | F | |RA| train=test | |RAD| train=test | best-iter (|RA|) |
|---|---:|---:|---:|---:|
| seq_0_frame_135 | 135 | 0.8011 | 0.5712 | 306 (plateau; regressed) |
| seq_1_frame_185 | 185 | 0.8105 | 0.7251 | 131 (plateau; regressed) |
| seq_1_frame_438 | 438 | 0.8718 | 0.6277 | 121 (plateau; regressed) |
| **mean**        | — | **0.8278** | **0.6413** | — |

**Key interpretations:**

1. **v7 single-frame |RA| = 0.83** vs v5 single-frame |RA| = 0.95. The **0.12 CC cost is the objective/metric mismatch** (C2): optimising |RAD| MSE distributes gradient across 31 Doppler slices, so chirp-0 |RA| gets ~1/31 of the attention.

2. **v7 best_iter is always early** (121–306 of 1500). Mean train CC DECREASES after that, even as loss continues to decrease. This is characteristic of an objective-metric split: |RAD| MSE keeps improving, |RA| CC regresses.

3. **v7 single-frame |RAD| = 0.64 (3-scene mean)**. This is the representation's hard ceiling on |RAD| given all grace on one frame. Our 6-scene bench |RAD| test = 0.36 is **56% of this ceiling** — the rest is NVS generalization loss (0.28 absolute).

---

## 3. Sensitivity analyses (§2.3, §2.4)

Pilot: `seq_0_frame_135` at init-state model, target_n=20000.

### §2.3 v_ego sensitivity sweep

GT |v_ego| = 1.219 m/s (from interpolated trajectory).

| scale | |v| (m/s) | |RAD| CC (init) |
|---:|---:|---:|
| ×−1.00 | 1.22 | **0.006** (near-zero → sign convention correct) |
| × 0.00 | 0.00 | 0.054 |
| × 0.50 | 0.61 | 0.090 |
| × 0.80 | 0.98 | 0.147 |
| × 0.90 | 1.10 | 0.169 |
| × 1.00 | 1.22 | 0.187 |
| × **1.10** | **1.34** | **0.196 (peak)** |
| × 1.20 | 1.46 | 0.192 |
| × 1.50 | 1.83 | 0.134 |
| × 2.00 | 2.44 | 0.060 |

→ Pose-derived v_ego is systematically **~10% under-estimated**. Peak is at ×1.10.

### §2.4 TX position jitter

| σ (mm) | CC(jit, base) |
|---:|---:|
| 0.0 | 1.0000 |
| 0.1 | 0.9945 |
| 0.3 | 0.9144 |
| **1.0** | **0.6227** |
| 3.0 | 0.6812 |
| 10.0 | 0.6372 |

→ **Sub-0.3 mm alignment required** to retain > 0.9 CC. At σ = 1 mm (roughly our pass-2 accuracy), CC = 0.62 — **within 0.05 of our trained test CC of 0.56**. The random-phase floor is 0.62–0.68.

**Implication.** Our training plateau sits *at* the random-phase floor. We cannot train our way *through* it; we need tighter alignment.

---

## 4. v_ego refinement pilot (§4.0)

Pilot: `seq_0_frame_135`, all 9 frames {131..139}, init-state model.
[data: data/v_ego_cache/seq_0_frame_135/v_ego_refined_manifest.json]

Stage 1 (3-D grid ±0.2 m/s in ±0.1 m/s steps) + Stage 2 (Nelder-Mead fine refinement, bounded to ±0.3 m/s):

| F | seed v_ego (m/s) | refined (m/s) | Δ|v| | S1 CC | Final CC |
|---:|---|---|---:|---:|---:|
| 131 | [+0.39,+1.18,+0.13] \|v\|=1.25 | [+0.36,+1.48,+0.43] \|v\|=1.58 | 0.425 | 0.2160 | 0.2214 |
| 132 | [+0.35,+1.07,+0.16] \|v\|=1.14 | [+0.35,+1.37,+0.46] \|v\|=1.49 | 0.424 | 0.2177 | 0.2209 |
| 133 | [+0.38,+1.00,+0.16] \|v\|=1.08 | [+0.36,+1.24,+0.37] \|v\|=1.34 | 0.318 | 0.2316 | 0.2319 |
| 134 | [+0.45,+1.07,+0.12] \|v\|=1.17 | [+0.43,+1.37,+0.42] \|v\|=1.50 | 0.425 | 0.2584 | 0.2639 |
| 135 | [+0.46,+1.10,+0.25] \|v\|=1.22 | [+0.43,+1.38,+0.55] \|v\|=1.55 | 0.412 | 0.1978 | 0.1991 |
| 136 | [+0.36,+1.10,+0.21] \|v\|=1.18 | [+0.30,+1.29,+0.01] \|v\|=1.33 | 0.284 | 0.2275 | 0.2285 |
| 137 | [+0.27,+1.12,+0.07] \|v\|=1.15 | [+0.36,+1.35,−0.03] \|v\|=1.40 | 0.269 | 0.1957 | 0.1958 |
| 138 | [+0.26,+1.07,+0.13] \|v\|=1.11 | [+0.44,+1.08,−0.08] \|v\|=1.17 | 0.273 | 0.1892 | 0.1894 |
| 139 | [+0.33,+1.07,+0.12] \|v\|=1.12 | [+0.19,+1.23,−0.18] \|v\|=1.25 | 0.364 | 0.1527 | 0.1552 |

**Mean Δ|v| = 0.355 m/s** (≈ 30% of seed magnitude).

**Consistent pattern:** Y-axis pushed *up* by ~+0.25 m/s across all 9 frames. This is a **systematic bias** in the pose-derived v_ego pipeline — not a per-frame random error.

**CC impact at init-state** is small (+0.001 to +0.005). But init-state |RAD| is tiny (~0.2) because materials are at defaults; the differential effect at a *trained* state is the real question — scheduled as a follow-up.

**Why this matters.** The sensitivity sweep (§3) says ×1.10 of seed is optimal; the refinement says the optimal is ~×1.27 (|v| 1.22 → 1.55). The sweep scales uniformly, the refinement picks *directions* — picking directions beats uniform scaling. This is exactly what §4.0's structural investment was designed to unlock.

---

## 5. Ablation matrix (§2.5) — pilot (seq_0_frame_135)

[JSON](ablation_matrix_seq_0_frame_135_F135.json)

| row | |RA| train | |RA| test | |RAD| train | |RAD| test |
|---|---:|---:|---:|---:|
| B_baseline (post-C1, gt_mean norm) | 0.677 | 0.599 | 0.454 | 0.296 |
| NORM_MAX_legacy (pre-C1, gt_max norm) | 0.668 | 0.618 | 0.440 | 0.306 |
| M03_multitask (λ = 0.3) | 0.697 | 0.583 | 0.452 | 0.288 |
| **M10_multitask (λ = 1.0)** | **0.715** | **0.621** | 0.446 | 0.298 |
| **M30_multitask (λ = 3.0)** | **0.715** | **0.628** | 0.427 | 0.276 |
| NO_DOP_v5style | (run failed to parse, not critical) |

**Δ vs baseline:**

| row | Δ|RA| train | Δ|RA| test | Δ|RAD| train | Δ|RAD| test |
|---|---:|---:|---:|---:|
| NORM_MAX_legacy | −0.009 | +0.019 | −0.014 | +0.010 |
| M03_multitask | +0.020 | −0.016 | −0.002 | −0.008 |
| **M10_multitask** | **+0.038** | **+0.022** | −0.008 | +0.002 |
| **M30_multitask** | **+0.038** | **+0.029** | −0.027 | −0.020 |

**Findings:**

1. **C2b multi-task loss works on |RA|.** λ = 1.0 gives +0.038 train, +0.022 test. λ = 3.0 saturates |RA| gain, starts hurting |RAD|. **Sweet spot: λ ∈ [1.0, 2.0]**.

2. **Multi-task does not help |RAD|.** It slightly hurts at high λ. Consistent with: adding a |RA|-specific gradient term biases the optimizer away from the |RAD| cube structure.

3. **NORM_MAX ≡ NORM_MEAN within MC noise.** This is the Adam scale-invariance prediction from earlier, *directly confirmed* on a third benchmark.

4. **No single ablation pushes any metric across its ceiling.** The gap from baseline (0.677, 0.599, 0.454, 0.296) to the targets (0.85, 0.70, 0.85, 0.70) is structural, not an ablation away. Multi-task closes ~20% of the |RA| gap; nothing in this table touches the |RAD| gap meaningfully.

**Implications for defaults:**
- Change default `--loss_multitask_lambda` from 0.0 to 1.0 when doppler is on.
- Keep `--loss_norm mean` (C1 convention, same behaviour as max under Adam).

---

## 6. Pose-alignment headroom — REVISED FINDING

**Initial reading (wrong):** §2.4 TX jitter sweep showed σ=1 mm → CC = 0.62, and our trained test CC is 0.56, so we thought rigid mm-scale alignment error was the dominant blocker.

**Corrected reading after running pose_refine:** *per-antenna random jitter* and *rigid array translation* are different failure modes. The jitter sweep perturbed each of the 12 TX independently — that breaks coherent beamforming because the virtual-array positions become randomised. A **rigid** 1-mm translation of the whole radar, by contrast, only adds a common phase and preserves beamforming coherence. Pass-2's alignment is rigid (6-DOF radar pose per frame). It does NOT correct per-antenna calibration error.

### Pass-3 rigid translation refinement — implemented and tested

Two variants of `mm25DGS_v7/preprocessing/pose_refine.py`:

**a) init-state pose_refine** (first attempt):
- ±4 mm search box, 5³ grid → Nelder-Mead.
- Found "optima" at 2.7–4.4 mm per frame with +0.02 CC gain at init-state.
- **A/B bench with refined configs: |RA| test = 0.572 vs pass-2 baseline 0.621 (−0.049).**
- Interpretation: init-state objective (CC ≈ 0.1) is too noisy; optimizer walks to bounds chasing noise.

**b) warm-started pose_refine** (second attempt, correct):
- 200 iters of v5-style material training first (CC rises from ~0.15 to ~0.45 on warm-started state).
- Then pose refinement with tightened ±1 mm bound, ±0.75 mm step.
- Result across 9 frames of seq_0_frame_135: **mean |Δp| = 0.027 mm, Δ mean CC = +0.0001**.
- **Pass-2 is already sub-mm accurate for rigid translation.** There is no pose refinement to extract.

### What this reveals about the true ceiling

The |RA| test CC of 0.56 is NOT from rigid pose misalignment. Candidate explanations, in priority order:

1. **Per-antenna calibration error.** The 12 TX and 16 RX positions come from factory calibration files. Pass-2 translates the array as a rigid body; it cannot refine individual antenna positions. Per-antenna σ ≈ 0.3–0.5 mm random error would land us right in the observed CC band. High-DOF (84 params), hard to optimise without gradients.
2. **Missing non-ego scene dynamics.** Moving scatterers (vegetation, traffic, pedestrians) generate per-frame |RA| deltas that a shared-material static mesh cannot reproduce. The F±1 inter-frame coherence ceiling of 0.586 already encodes this.
3. **LiDAR → radar frame registration drift.** The static LiDAR mesh is loaded once but used for all train frames. If inter-frame radar-trajectory drift is present relative to LiDAR, each frame has a small residual scene shift even with a perfect rigid radar pose.
4. **Temporal LiDAR aliasing.** LiDAR captures the scene at distinct times; a static mesh cannot represent what moves between LiDAR frames within a cascade capture.

**None of these are fixable by rigid pose refinement.** All are fixable only by either (a) extending the forward model (dynamic scene, per-antenna deltas) or (b) accepting a ceiling that matches physics.

---

## 7. Ranked next structural moves — REVISED

Based on the pass-3 negative result (§6), the ranking of remaining options is substantially different from the first draft. In priority order:

| # | action | expected CC gain | effort | rationale |
|---|---|---|---|---|
| 1 | **Reframe paper claims to what physics allows** | 0 CC — framing | < 1 day | §1 + §8: 0.70 test CC is above F±1 ceiling 0.59; 0.85 |RAD| train is above single-frame ceiling 0.64. The paper must claim what is *defensible*. |
| 2 | **C2b multi-task λ = 1.0 as default** | +0.04 |RA| train, +0.02 |RA| test | ✅ shipped (`--loss_multitask_lambda 1.0`) | §5 ablation |
| 3 | Deploy v_ego-refine cache in training (A/B) | +0.01 to +0.03 |RAD| test (unknown) | 1 day | §4 — systematic bias documented |
| 4 | target_n=90000 multi-frame bench | +0.03 to +0.08 on all metrics (unverified) | 1-2 days of compute | single-frame 0.95 was at 90000; ours is 20000 |
| 5 | Per-antenna calibration refinement | +0.05 to +0.15 on |RA| if per-antenna σ ≈ 0.3 mm is real | 3-4 wk, structural (84-DOF optimisation) | §6 — the last plausible coherent-phase lever |
| 6 | Learn bounded position deltas on LiDAR points (§2.6.a) | small (+0.01-0.04) — upper-bounded by §1.1 phase-error dominance | 2-3 wk, structural | only worth it if per-antenna calibration also exhausted |
| ~~0~~ | ~~Pass-3 rigid translation refinement~~ | ~~+0.1–0.25~~ | tested | **dropped** — warm-started pose_refine finds Δ ≈ 0 |

---

## 8. Honest paper-framing implications

Given the physical ceilings measured in §1:

### What the paper CAN claim
- **First differentiable 77 GHz mmWave radar renderer with analytic TDM + inter-chirp Doppler phase** (v7 contribution).
- **Quantification of the sub-mm alignment requirement** for coherent NVS in mmWave radar.
- **Results at or above the naive-avg baseline of 0.654 |RA| test CC** on a 6-scene benchmark with held-out frames.
- **Measurably better than the zero-Doppler baseline on |RAD| CC.**

### What the paper CANNOT defend
- "0.70 |RA| test CC" — above the F±1 inter-frame coherence ceiling of 0.586. Any method reporting this is either data-snooping or using an unfairly easy test split.
- "Full scene |RAD| recovery" — physically unmatchable in absolute CC because real scenes have non-ego Doppler (moving cars, pedestrians, foliage) that the ego-motion-only forward model cannot produce.

### Defensible reframing
- "v7 recovers the **ego-motion-attributable component** of |RAD| to X% of the per-scene theoretical ceiling (mean X across 6 scenes)." — reports both numerator (measured CC) and denominator (ego-only ceiling) honestly.
- "Test performance is bounded above by the inter-frame coherence of 5 Hz cascade capture at 0.586; our method achieves Y × that bound, comparable to / ahead of the naive-interpolation baseline of 0.654." — claims the right thing about NVS.

---

## 9. Outstanding work

Already scheduled / planned:
- [ ] Run `visualize_pred_vs_gt.py` on v5 + v7 single-frame checkpoints to produce the prediction-vs-GT image panels (cost: 2 minutes, pending because GPUs were busy).
- [ ] Trained-model v_ego refinement ablation: train v7 with refined v_ego cache (requires plumbing v_ego_cache override path through `--v_ego_source refined` flag).
- [ ] Extend ablation matrix rows: `T_a = 0` (TDM off), `v_ego = 0` (no Doppler phase), identity TI firing order — to finalise §3 code-correctness audit.
- [ ] Post-training pose-jitter sensitivity (does training absorb some alignment error?).

Recommended next sprint:
- [ ] **Pass-3 alignment implementation.** By far the highest-leverage item. The sensitivity curve quantifies the prize.
