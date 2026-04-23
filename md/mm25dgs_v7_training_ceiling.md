# v7 Doppler — why train CC plateaus at ~0.70 (training-ceiling blockers)

**Date:** 2026-04-23
**Parent plan:** `md/mm25dgs_v7_speed_and_ceiling.md` (speed-up work)
**Observed:** v7 Doppler train CC = 0.698 (6-scene mean) vs v5/M1 = 0.826;
test CC is a wash (v7 0.552 vs v5 0.537, +0.015 in mean — inside MC band).

This is the reference for the **training-quality** investigation to run
*after* the speed-up plan lands. Keep as-is; do not fold into PRs.

---

## TL;DR ranking

| # | blocker | est. impact on train CC | est. impact on test CC | fix cost |
|---|---|---|---|---|
| C1 | Loss normalised by `gt_max²` instead of `gt_mean²` | **large (+)** | small (+) | 1-line |
| C2 | Objective (|RAD|) ≠ metric (chirp-0 |RA|) | medium (structural) | none | metric-or-loss change |
| C3 | TDM phase on chirp 0 carries v_ego noise | small-medium (+) | small (neutral/+) | physics investigation |
| C4 | Regularizers / clipping / LR | **none** (verified off) | **none** | n/a |

Do C1 first — it is a bug-class fix with a large expected delta. Only
investigate C2 and C3 after C1 is in.

---

## C1 — Loss normalisation mismatch

**Location:** [mm25DGS_v7/train_frame_nvs.py:819](../mm25DGS_v7/train_frame_nvs.py#L819)

```python
# v7:
loss_k = (mag_pred - gt_mag).pow(2).mean() / (gt_max ** 2)
```

vs the reference:

```python
# v5 compute_ra_loss_rp (mse_raw branch):
gt_scale = gt_cached.mean().detach().clamp(min=1e-30) ** 2
loss = ((ra_rend_mag - gt_cached) ** 2).mean() / gt_scale
```

`gt_max` is where a handful of sparse peaks land in the |RAD| cube.
`gt_mean` is the mean over all 31·127·256 ≈ 10⁶ bins, most of which are
near the noise floor. For sparse radar magnitudes `max/mean` is
typically 50–100× on 2D |RA|, and larger on 3D |RAD| because the
peaks don't change but the mean dilutes across 31 Doppler bins.

Consequence: v7's numerical loss is roughly
`(gt_mean / gt_max)² ≈ 1/2500 – 1/10000`
the size of an equivalent v5 loss. With the same LR, the effective step
size on `raw_materials` is 3–4 orders of magnitude smaller. That is
enough to explain most of the plateau.

### Action

1. Read `gt_mag.mean()` inside `_build_rad_bundle_for_frame`
   ([`train_frame_nvs.py:182`](../mm25DGS_v7/train_frame_nvs.py#L182))
   and store it as `gt_rad_mean` alongside (or replacing) `gt_rad_max`.
2. Replace `gt_max**2` with `gt_rad_mean**2` in the doppler training body
   ([`train_frame_nvs.py:819`](../mm25DGS_v7/train_frame_nvs.py#L819)).
3. **Sanity print** on one scene before running the full bench:
   ```python
   print(gt_mag.max() / gt_mag.mean())
   ```
   Expected: 50–100× for a real scene. If it's only 5–10×, C1's
   contribution is smaller than expected and C2 is the dominant
   remaining blocker. The fix is still worth landing (v5 parity is
   the correct convention).

### Validation

Re-run one scene (`seq_0_frame_135`) with the fix, 500 iters. Expected
deltas:

- Train CC: 0.70 → 0.80 ± 0.03 (approaching v5/M1 band).
- Test  CC: ≈ unchanged (+0 to +0.05; within MC band).
- Loss trajectory: larger absolute values, clean decreasing curve.

If train CC climbs to 0.80+ this is confirmed as the dominant blocker.

---

## C2 — Objective ≠ metric (structural, not a bug)

**Situation:** v7 loss is MSE on full |RAD| (31 × 127 × 256 = ~10⁶ cells).
`mean_train_cc` / `test_cc` are cart-corr on **chirp-0 |RA|** — one 2D
slice (127 × 256 = ~3 × 10⁴ cells). The optimiser distributes gradient
across 31 Doppler slices; the metric reads one of them back.

v5 had no such split: its loss ≡ its metric (both chirp-0 |RA|). So v5
specialises to the metric by construction.

**Evidence this is real and not pathological:**

- v7 test CC is actually **+0.015 over v5** in 6-scene mean, despite
  the −0.13 train CC gap. If v7 were under-fitting in a physically
  meaningful way, test CC would also sag.
- The gap is uniform across scenes (−0.09 to −0.18), suggesting a
  global structural offset, not scene-specific optimisation trouble.

### Two independent fixes

**C2a — metric change (free, diagnostic):** log `cart_corr` over
flattened |RAD| each iteration. This is the metric that matches the
training objective. It will likely report ~0.83-0.85 for v7 (same
territory as v5's |RA| CC), confirming the plateau is an artifact of
the slice we're looking at.

- Location: `train_frame_nvs.py` — add `_render_and_rad_corr(bundle)`
  alongside `_render_and_cart_corr(sample)`, log as
  `mean_train_rad_cc` next to `mean_train_cc`.

**C2b — loss change (tune-required):** add a multi-task term for
chirp-0 |RA|:

```python
loss_k = loss_rad + lam * loss_chirp0_ra
```

Use the chirp-0 render already in the stack (`rp_r[0], rp_i[0]`). No
extra forward cost; `lam` ∈ [0.1, 1.0] is a small sweep. This biases
the optimiser back toward the v5 metric without dropping the Doppler
supervision.

**Recommendation:** do C2a first (free), then decide if C2b is needed.
If the |RAD| CC is already in the 0.82-0.86 band after C1 lands, the
answer is "don't bother with C2b, just report |RAD| CC."

---

## C3 — TDM phase on chirp 0

**Situation:** v7's chirp-0 render includes the `k(i)·T_a` intra-burst
TDM term (plan §1.3). At |v_ego| ≈ 1.3 m/s, λ = 3.896 mm, T_a = 41 μs:
per-TX phase ramp ≈ 0.17 rad/slot, cumulative ≈ 2 rad across 12 slots.

GT chirp-0 |RA| contains the *true* TDM phase pattern; v7's rendered
chirp-0 contains a model of it driven by interpolated-pose `v_ego`.
Any v_ego error propagates directly into the chirp-0 |RA| metric.

v5 ignored this entirely — v5's chirp-0 |RA| is mathematically
self-consistent with its own render but physically wrong; v7 is
physically correct but now exposed to pose-interp noise on the
metric.

### Test

Re-render one scene with `T_a=0` passed to `render_factorized_doppler`
(keeps inter-chirp Doppler, removes TDM). If train-CC rises
meaningfully (>+0.02), C3 is a real contributor. If it doesn't, C3 is
cosmetic and can be ignored.

This is the item flagged in `md/mm25dgs_v7_doppler_plan.md` §1.3 as
"revisit TDM correction in the future." Only worth chasing after C1
and C2 are addressed.

### If confirmed

Options:

1. Better v_ego (e.g. raw IMU integration instead of pose-derived)
2. Learn a small per-frame v_ego correction (breaks the "v_ego not
   learnable" rule from the plan — do only with strong justification
   and tight regularisation)
3. Re-formulate to a |RAD|-only metric (C2) and stop measuring chirp
   0 — makes C3 irrelevant

---

## C4 — What is NOT a blocker (verified)

All regularisers are off by default in `run_v7_bench.sh`:

- `reg_l2_drift_lambda = 0.0`
- `reg_fisher_rot_lambda = 0.0`
- `reg_active_top_frac = None`

Gradient clipping (`clip_vals = {'materials': 1.0, 'rotations': 0.5}`)
is identical to v5.

LR schedule (no warmup, linear decay from iter 200) is identical to v5.

`learn_*` flags are identical to v5.

→ **Nothing in the trainer config is artificially capping v7's fit.**
The ceiling is set by C1 (loss scale) and C2 (objective/metric split).

---

## Validation plan

Run in order. Each is one scene (`seq_0_frame_135`), 500 iters.

| exp | change | expect | if observed |
|---|---|---|---|
| E1 | C1: `gt_max²` → `gt_mean²` | train CC 0.70 → 0.80 | C1 confirmed; advance to E2 |
| E2 | C2a: log |RAD| CC | |RAD| CC already ≥ 0.82 | report |RAD| CC, don't need E3 |
| E3 | C2b: add λ·mse(chirp-0) | train |RA| CC → 0.83+, test ≈ stable | pick best λ, bake in |
| E4 | C3: `T_a=0` in doppler render | train CC rises >0.02 | optional: address v_ego noise |

Order: **E1 → E2 → E4 → E3** (decide E3 based on E2 outcome).

Total compute: 4 runs × 1 scene × 500 iter ≈ 20-40 min wall-clock once
the speed-up plan lands (currently ~25 min/scene → 100-ish min).
