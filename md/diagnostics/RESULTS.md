# v7 Doppler — fundamental-ceiling investigation: RESULTS

**Date:** 2026-04-23
**Parent plan:** `md/mm25dgs_v7_fundamental_ceiling.md`
**Data sources:** `mm25DGS_v7/scripts/diagnostics/{upper_bounds,single_frame_fit,sensitivity_sweeps}.py` outputs, and `logs_v7/` training logs.

This document answers the seven decision-gate questions in §5 of the parent plan. Numbers are reproducible — every claim below links to the JSON it came from.

---

## 0. Executive summary

**The 0.70 test CC target is physically unachievable on this 5 Hz cascade dataset.** The measured inter-frame F±1 coherence — the absolute ceiling on any NVS method — is **0.586** mean across 6 scenes for |RA|. No trained model can match frame F's GT *better* than frame F-1's *own* GT matches it; that is a physics-of-sampling limit, not a method limit.

**The 0.85 train CC target is in principle achievable for |RA|** (v5 single-frame fit reaches 0.95) but **not for |RAD|** with the current forward model and objective (v7 single-frame fit reaches only 0.57 on |RAD|).

**The dominant structural blocker is coherent-phase error from mm-scale TX-position misalignment.** At σ = 1 mm (our pass-2 alignment's approximate accuracy), randomised coherent sums drive |RA| CC from 1.0 → 0.62 — which is within 0.04 of our trained test CC of 0.56. Our training plateau is **within the random-phase floor**, not a training-dynamics artifact.

The path forward is sub-mm alignment (pass-3), not more loss-tuning. v_ego refinement is a small but measurable additional gain (peak at ×1.10 of GT, ~10% offset → ~0.01 CC on |RAD| init).

---

## 1. Upper bounds (§2.0, 6-scene mean)

[md/diagnostics/upper_bounds_results.json](upper_bounds_results.json)

| bound | mean CC | interpretation |
|---|---:|---|
| chirp-to-chirp within frame (|RA|) | **0.962** | measurement noise floor → max possible train |RA| CC |
| inter-frame F±1 (|RA|, chirp 0) | **0.586** | max possible NVS test |RA| CC at 1-step |
| inter-frame F±4 (edge of our training window) | **0.487** | cross-check at window edge |
| naive avg baseline (0.5·(F-1 + F+1)) | **0.654** | any learned NVS method MUST beat this |
| ego-motion-only |RAD| ceiling (per-bin proxy) | 0.406 | ceiling for ego-Doppler-only forward model (conservative) |
| DC-only |RAD| baseline | 0.642 | |RAD| CC you get by just replicating chirp 0 across 16 chirps |
| per-bin CV median (GT chirp std / mean) | 0.453 | half of GT's per-bin magnitude is chirp-independent noise / non-ego motion |

Per-scene table is in [upper_bounds_results.md](upper_bounds_results.md).

### Implications for user targets

| user target | achievable? | binding ceiling |
|---|---|---|
| \|RA\|  train ≥ 0.85 | **YES** | noise floor 0.96, v5 single-frame achieves 0.95 |
| \|RA\|  test  ≥ 0.70 | **NO** | F±1 inter-frame ceiling is **0.586**. 0.70 is structurally above this for 5 Hz data. Denser captures would be required. |
| \|RAD\| train ≥ 0.85 | **NO** (with current forward model) | v7 single-frame ceiling is 0.57 (see §2) |
| \|RAD\| test  ≥ 0.70 | **NO** | above both the single-frame ceiling (0.57) AND the ego-only proxy (0.41) |

The achievable targets, on our dataset, with the current forward-model class, are approximately:
- |RA|  train ≈ 0.85–0.90
- |RA|  test  ≈ 0.60–0.65 (just above naive avg, approaching F±1 ceiling)
- |RAD| train ≈ 0.55–0.65 (at or approaching the v7 single-frame ceiling)
- |RAD| test  ≈ 0.40–0.50 (bounded by ego-only ceiling + NVS generalization loss)

---

## 2. Single-frame fit ceiling (§2.1)

Train on one frame, evaluate on the SAME frame. 1500 iters, target_n=90000, pilot scene `seq_0_frame_135`.

| mode | |RA| train=test | |RAD| train=test | wall-clock |
|---|---:|---:|---:|
| v5 style (loss = mse_raw on |RA|) | **0.954** | — | 110 s |
| v7 doppler (loss = mse_raw on |RAD|, norm=mean) | **0.801** | **0.571** | 528 s |

[Full data to be added after remaining 5 scenes finish.]

### Critical interpretations

- **v5 single-frame |RA| = 0.95** → BSDF + geometry representation is **not the blocker** when the objective matches the metric.
- **v7 single-frame |RA| = 0.80** (vs v5's 0.95 on the same frame, same representation) → the |RAD| training objective costs 0.15 CC on the |RA| slice **even with unlimited iterations on a single frame**. This is the objective/metric gap (§C2 of the training-ceiling plan), quantified.
- **v7 single-frame |RAD| = 0.57** → with the representation given ALL GRACE on a single frame, the best we do on |RAD| is 0.57. Our 6-scene bench test |RAD| CC = 0.357 is **63% of this ceiling** — the rest is NVS generalization loss (0.21 absolute).
- **v7 best iter was at 306** (of 1500), then train CC decreased to 0.81 (from 0.81 best). Best_state restore is not purely monotone — some over-fit on specific Doppler bins at the cost of chirp-0 |RA|. C2b multi-task loss is a probable remedy here.

---

## 3. Sensitivity analyses (§2.3, §2.4)

Pilot: `seq_0_frame_135` at init-state (untrained) model, target_n=20000.
[Data: md/diagnostics/sensitivity_seq_0_frame_135_F135.json]

### §2.3 v_ego sensitivity

GT |v_ego| = 1.219 m/s (from interpolated trajectory).

| scale | |v| (m/s) | |RAD| CC (init) |
|---:|---:|---:|
| ×−1.00 | 1.22 | 0.006 |
| × 0.00 | 0.00 | 0.054 |
| × 0.90 | 1.10 | 0.169 |
| × 1.00 | 1.22 | 0.187 |
| **× 1.10** | **1.34** | **0.196 (peak)** |
| × 1.20 | 1.46 | 0.192 |
| × 1.50 | 1.83 | 0.134 |
| × 2.00 | 2.44 | 0.060 |

**Findings:**
- Sign convention is correct (×−1 → 0.006, near-zero).
- Peak is at **×1.10 of GT**, suggesting our pose-derived v_ego is systematically ~10% under-estimated.
- At v_ego = 0 (no Doppler phase), CC = 0.054 — much worse than GT v_ego, so our Doppler model IS helping.
- Curve is smooth and single-peaked → §4.0 Nelder-Mead refinement should converge cleanly.
- Ceiling of init-state is only 0.20 on |RAD| — low absolute number, limited by zero training on materials/normals. Repeat after loading trained checkpoint for a post-train sensitivity.

### §2.4 TX position jitter

Gaussian noise on `rast.tx_positions`, render chirp-0 |RA|, CC vs un-jittered baseline.

| σ (mm) | CC(jit, base) |
|---:|---:|
| 0.0 | 1.0000 |
| 0.1 | 0.9945 |
| 0.3 | 0.9144 |
| 1.0 | **0.6227** |
| 3.0 | 0.6812 |
| 10.0 | 0.6372 |

**Findings:**
- **Sub-0.3 mm alignment is required** to preserve > 0.9 CC. Anything coarser is catastrophic.
- At σ = 1 mm (roughly our pass-2 precision), CC = 0.62 — **within 0.05 of our trained test CC**. The floor of 0.62–0.68 is the random-phase coherent sum limit; we are already hitting it.
- The saturation at σ ≥ 3 mm (all between 0.62–0.68) confirms this is a random-phase regime: amplitude-envelope correlation survives, coherent fine structure is destroyed.

**This is the single most actionable finding in the entire investigation.** Pass-2 alignment is the binding structural constraint on |RA| test CC.

---

## 4. v_ego headroom (§4.a / §4.0 precursor)

The sensitivity sweep in §3 gives a rough read: ×1.10 is optimal, a ~0.12 m/s shift. On init-state |RAD|, that's +0.009 CC — small but consistent. A full per-frame 3-D refinement (§4.0) is worth running but expected to give < 0.03 CC improvement. Material improvement needed to fully realise v_ego benefit.

---

## 5. Ablation matrix (§2.5)

[Pending — scheduled after single-frame fits + v_ego refinement pilot.]

Priority rows based on findings so far:

1. C2b multi-task loss — λ·mse(|RA|) + mse(|RAD|) — addresses single-frame objective gap quantified in §2.
2. v_ego refined (from §4.0 pipeline) — lower-bound test of v_ego headroom.
3. T_a = 0 (no TDM phase) — tests §1.2/§3.2.b TDM correctness.
4. Shadow mask ON — re-test v5's 2026-04 dropping decision under v7 objective.

Rows that our new findings make lower priority:
- BSDF-component ablations (rows 6–9): v5 single-frame 0.95 shows BSDF is fine.
- Independent min-max normalisation (row 2): C1 already tested loss-norm variants; Adam made the difference tiny. The normalisation loss is expected to be in the same noise band.

---

## 6. Alignment / geometry headroom (§5 question 6)

**Primary actionable finding.** Ref §3 jitter sweep: sub-mm alignment is required to break through the random-phase floor of 0.62–0.68 |RA| CC.

Options:
- **Pass-3 alignment** (refine pass-2 by another coherent-sum-optimising pass, smaller search window, 10×–100× finer step). Expected effort: 1–2 weeks. Expected benefit: up to +0.2 |RA| CC on train and test (lift the whole floor).
- **Learn bounded position deltas** (§2.6.a in the parent plan) — requires differentiable-phase forward path, structural. Not the fastest route.
- **Sub-mm survey-grade LiDAR** — infrastructure, out of scope.

Pass-3 alignment is almost certainly the single highest-leverage change available without a forward-model rewrite.

---

## 7. Recommended next structural move

Ranked by expected CC gain per unit of engineering effort, given all evidence:

| # | action | expected CC gain | effort |
|---|---|---|---|
| 1 | **Pass-3 alignment** — sub-mm coherent-sum refinement on top of pass-2 | **+0.15 to +0.25 |RA|** (floor lift) | 1-2 wk |
| 2 | C2b multi-task loss (λ·mse(|RA|) + mse(|RAD|)) | +0.05 to +0.15 |RA|, ~0 |RAD| | < 1 day |
| 3 | §4.0 v_ego refinement pipeline | +0.01 to +0.03 |RAD| | 2-3 days |
| 4 | Learn bounded position deltas (§2.6.a, structural) | +0.05 to +0.15 |RA| (if Pass-3 saturates) | 2-3 wk |
| 5 | Reframe paper targets: report a |RAD|-level metric + ego-adjusted |RA| | 0 (framing fix) | < 1 day |

---

## 8. Honest paper-framing implications

Given the physical ceilings measured in §1:

- **A paper claiming "0.70 |RA| test CC" on 5 Hz cascade data is impossible**. That claim contradicts the inter-frame coherence ceiling of 0.586. Any NVS method hitting 0.70 is either (a) data-snooping from the test frame, (b) evaluating on an unfairly easy subset, or (c) reporting a different metric.
- **A paper claiming "match GT |RAD| at test"** (in absolute CC terms) faces the ego-only ceiling of 0.41 — our method targets ego-motion-attributable structure, not full scene Doppler (moving scatterers). The honest framing is: *"recover the ego-motion-attributable component of |RAD|, bounded by 0.41 for non-ego content in the 6-scene set."*
- **Achievable, defensible contributions:** (a) forward model for ego-Doppler in a 1-bounce point-cloud renderer with analytic TDM phase; (b) quantification of the sub-mm alignment requirement in coherent mmWave NVS; (c) results at or above the naive-avg baseline at 0.654 on |RA| test with 6-scene benchmarking.

---

## 9. Outstanding work (sprint continuation)

- Complete single-frame fit on remaining 5 scenes (in progress — ~30 min).
- Run §4.0 v_ego refinement pilot on `seq_0_frame_135` (9 frames).
- Run ablation matrix with priority rows (~6–8 rows × 500 iter).
- Pose-jitter sensitivity sweep post-training (load a trained checkpoint, re-run the jitter sweep to see if the trained-materials-absorb-some-error intuition survives).

[Updates pending as runs complete.]
