# Per-chirp alignment — Stage 3 plan (LERP-anchored refinement)

## Summary

Today the cascade-radar alignment pipeline runs in two stages — both at
the *frame* level and both keyed off chirp 0 of each frame:

| stage | scope | input | output |
|---|---|---|---|
| 1 (pass 1) | per-frame (chirp 0) | unaligned config | `cascaded_frame_<F>_aligned.json` |
| 2 (pass 2) | per-frame, trajectory prior across 9 frames | pass-1 + scene trajectory | `cascaded_frame_<F>_aligned_pass2.json` |

Downstream NVS training needs a pose at every (frame, chirp). The
current trainer builds those poses by **linearly interpolating** the
pass-2 poses of frame F-1 and frame F+1 at each chirp's fractional time.
That assumes linear motion between neighbouring frames — approximately
correct at walking pace over 100 ms, but *not* validated against
chirp-k's own radar signal.

**Stage 3** adds a per-chirp refinement that takes the LERP pose as
input, searches a tight physical neighbourhood for a better-scoring
pose against chirp k's own RA, and either commits the refinement or
falls back to LERP when the refinement is off-trajectory or inferior.
The output is a per-chirp aligned config that is a refinement of LERP,
not an independent alignment.

This explicitly addresses the failure mode we observed on 2026-04-18:
running a *fully independent* per-chirp alignment (the existing
`mm25DGS_v5.preprocessing.alignment.per_chirp_alignment`) produced
per-chirp poses that deviated from LERP by 300–2000 mm — driven by
noise in the single-chirp RA signal, not by real motion (real
chirp-to-chirp motion at walking pace is ~0.5 mm). Stage 3 bounds the
refinement so it can't wander.

## Motivation

- **Data we have (2026-04-18):** per-chirp pass-1 alignment cart_corr
  is 0.27–0.35 (vs 0.45–0.55 for per-frame chirp-0 pass-1). Low
  single-chirp SNR → individual chirp alignment objectives have shallow
  maxima, many local optima at the mm-cm scale.
- **Data we need to explain:** in the frame-NVS experiments (2026-04-18),
  UB_144 (all 16 chirps per frame × 9 frames, test included)
  *underperforms* UB_9 (first chirp only). Probable drivers:
  (a) ADC-noise averaging over 16 chirp noise realisations; or
  (b) chirp-pose interpolation error (LERP wrong at a level that
      matters). Stage 3 output lets us isolate (b) via an A/B swap of
  the chirp pose source without changing anything else.

## Non-goals

- Stage 3 does **not** replace Stage 1 or Stage 2.
- Stage 3 does **not** re-align chirp 0 — its output for chirp 0 is
  identical to the Stage 2 per-frame config by construction (LERP at
  α=0.5 reproduces the per-frame pose for the centre chirp, modulo the
  small time offset of chirp 0 from frame-centre).
- Stage 3 does **not** aim for "independent" per-chirp alignments. The
  null hypothesis is "LERP is already right for this chirp"; Stage 3
  commits a refinement only when the data clearly outvotes that.

## Design

### Inputs

For a given `(scene, frame F, chirp c)`:

1. `cascaded_frame_F_aligned_pass2.json` (absolute anchor at frame F).
2. `cascaded_frame_{F-1}_aligned_pass2.json` and
   `cascaded_frame_{F+1}_aligned_pass2.json` (used by the LERP anchor
   option only).
3. `data/<scene>/radar/cascaded_frame_<F>.npy`, from which chirp `c`
   is extracted for alignment loss.
4. Scene mesh + pcl + antenna patterns (via `load_trained_config`).
5. **Dataset groundtruth poses** —
   `<raw_dataset>/<sequence>/groundtruth/groundtruth_poses.txt`
   plus `groundtruth/timestamps.txt`. The ColoRadar dataset already
   ships these at ~9 Hz per sequence; the preprocessing pipeline
   loads them via `ColoRadar_tools.dataset_loaders.get_groundtruth`
   and aligns them to radar timestamps via
   `preprocessing.utils.interpolate_poses`.
6. Cascade timestamps —
   `<raw_dataset>/<sequence>/cascade/adc_samples/timestamps.txt` gives
   the per-frame radar start time.

Edge frames (first and last of a scene's window) lack one LERP
anchor and are handled by falling back to LERP of the nearest two
available frames; Stage 3 still attempts refinement against the
chirp's RA but uses the fallback as the anchor.

### Per-chirp anchoring — three options, one recommended

The anchor is the pose at which Stage 3's refinement starts. The
search is bounded tightly around the anchor (±10 mm translation,
±0.5° rotations), so getting the anchor right is what determines
whether Stage 3 can find a data-supported refinement.

#### Option A (original): **LERP between pass-2 F-1 and F+1**

```
anchor_F_c = LERP(pass2_{F-1}, pass2_{F+1}, α_c)
α_c = 0.5 + c · T_loop / (2 · T_frame)
```

- **Pros**: no new data dependency; pass-2 anchors already trusted.
- **Cons**: inherits pass-2's frame-to-frame noise. Measured on
  seq_1_frame_438 (2026-04-18): **pass-2 frame-to-frame translation
  std = 153 mm over a 9-frame window** (vs the ~7 mm walking-pace
  std predicted by dataset GT). LERP carries that 153 mm jitter into
  every chirp's anchor.

#### Option B: **GT-absolute pose interpolated at chirp timestamps**

Use the dataset's `groundtruth_poses.txt` directly (9 Hz, reliable —
comes from LiDAR SLAM or VICON, depending on the sequence). Slerp the
GT pose to chirp k's timestamp, then transform world-base → world-
cascade via `base_to_cascade.txt` extrinsic.

- **Pros**: absolute position stability (GT frame-to-frame σ = 7-12
  mm on both target scenes — 20× better than pass-2 on seq_1).
  Handles per-chirp timing via slerp with sub-mm accuracy.
- **Cons**: GT has an absolute offset vs our pass-2 aligned configs
  — 100-200 mm translation and (apparent) ~90° boresight
  (convention mismatch, fixable). Using GT as the sole anchor
  **discards the radar-data-driven corrections** that pass-1/pass-2
  introduced.

#### Option C (**RECOMMENDED**): **Hybrid = pass-2 absolute at frame F + GT relative motion to chirp k**

```
# Absolute anchor at the radar frame: data-driven, from pass-2
anchor_center_F = pass2_F.center           # world-frame mm
anchor_bore_F   = pass2_F.boresight        # world-frame unit vector

# Relative motion from chirp 0 to chirp c, derived from GT (sub-mm stable)
t_F      = cascade_ts[F]                   # radar frame timestamp (s)
t_chirp_c = t_F + c · T_loop               # chirp c timestamp
Δpos_world    = GT_interp(t_chirp_c).pos - GT_interp(t_F).pos      # metres
ΔR_world      = GT_interp(t_chirp_c).rot @ GT_interp(t_F).rot.T    # small rotation

# Apply the relative delta to the per-frame anchor
anchor_F_c.center  = anchor_center_F + 1000 · Δpos_world            # mm
anchor_F_c.bore    = ΔR_world @ anchor_bore_F
```

- **Pros**: combines the radar-data-driven absolute pose (pass-2 has
  centimeter-level accuracy in the data-aligned world frame) with the
  GT's sub-mm inter-sample stability for per-chirp motion. Avoids both
  pass-2's frame-to-frame jitter AND GT's absolute-offset issue.
- **Cons**: an extra data dependency (the raw ColoRadar GT poses
  must be reachable during Stage 3). Minor: the hybrid assumes the
  GT-world and pass-2-world frames share *scale and orientation* — if
  the GT world frame is rotated vs the pass-2 world frame, `ΔR_world`
  rotates around the wrong axis. Verify on one frame before rolling
  out (see §P2 smoke test).

#### What IMU adds (or doesn't)

- **IMU rate is 100 Hz** = 10 ms/sample → only **1 sample lands
  inside the 7.87 ms chirp burst**. So IMU carries *no* trajectory
  information within a frame; integrating IMU over the burst is a
  random walk at sub-mm scale.
- **GT rate is 9 Hz** (~110 ms between samples). Consecutive
  radar frames (200 ms) are bracketed by 1-2 GT samples; slerp fills
  the gap cleanly for chirp-timescale queries.
- **GT quality dominates IMU quality** for anything above 10 ms
  resolution. Since we only need pose at chirp granularity (~0.5 ms
  stride), and GT at 9 Hz + slerp gives us sub-mm interpolation, IMU
  doesn't add value.
- **Where IMU might help**: high-curvature motion events (e.g.,
  sudden stops, rapid turns) where GT's 9 Hz undersampling could
  miss a feature visible at 100 Hz. The ColoRadar runs we're
  training on are steady walking segments — none of that regime.
  If future sequences include aggressive motion, reconsider. For
  now, **don't integrate IMU**.

### Pipeline

### Pipeline

```
         ┌──────────────────────────────┐
         │ Stage 2 per-frame pass-2     │   (unchanged)
         │ → cascaded_frame_<F>_pass2   │
         └──────────────┬───────────────┘
                        │
          ┌─────────────┴─────────────┐
          │  Stage 3a: LERP ANCHOR    │
          │  pose_anchor(F, c)        │
          │  = LERP(pass2_{F-1},      │
          │         pass2_{F+1},      │
          │         α_c)              │
          │   α_c = 0.5 + c · T_loop /│
          │         (2 · T_frame)     │
          └─────────────┬─────────────┘
                        │
     ┌──────────────────┴──────────────────┐
     │ Stage 3b: chirp-k-aware refinement  │
     │                                     │
     │ ctx = build_alignment_context(      │
     │       config = synthetic anchor pose,│
     │       chirp_idx = c)                │
     │                                     │
     │ candidates:                         │
     │   A) renderer 4DOF + prior, small   │
     │      radius around anchor           │
     │   B) lidar 4DOF + prior, same bound │
     │   C) anchor itself (no-op, LERP)    │
     │                                     │
     │ score all three with ctx cart_corr  │
     │ (comparable metric).                │
     └──────────────────┬──────────────────┘
                        │
     ┌──────────────────┴──────────────────┐
     │ Stage 3c: trajectory gate + winner  │
     │                                     │
     │ accept A or B only if:              │
     │  (i) cc > cc_anchor + margin        │
     │  (ii) ||pose - pose_anchor|| < bound│
     │       (componentwise)               │
     │ else winner = anchor (LERP)         │
     └──────────────────┬──────────────────┘
                        │
                        v
         ┌──────────────────────────────┐
         │ cascaded_frame_<F>_chirp<CC> │
         │         _aligned_pass3.json  │
         └──────────────────────────────┘
```

### Key parameters

These are the knobs. Defaults are picked to match physical
expectations at walking pace. All should be stored in the pass-3
summary JSON.

| parameter | default | reasoning |
|---|---|---|
| **anchor source** | **hybrid (Option C)** | pass-2 absolute + GT relative motion; see §"Per-chirp anchoring" |
| search radius: range | ±10 mm | 20× max chirp-to-chirp motion at 1 m/s over 7.87 ms |
| search radius: azimuth | ±0.5° | matches the sub-degree resolution of the radar |
| search radius: elev-rot | ±0.5° | same |
| search radius: azim-rot | ±0.5° | same |
| grid resolution | 5 ticks/DOF | 5⁴ = 625 cells, ~2 s/chirp at v5 CUDA rates |
| refine max-evals | 30 (Nelder-Mead) | converges in ~10 s/chirp |
| prior penalty λ | 0.05 | same as per-frame pass 2 |
| gate margin | 0.005 cc | refinement must beat anchor by ≥0.005 cc to be accepted |
| gate pose bound | componentwise radii above | anything beyond the search radius is already rejected; this is a redundancy |

CLI flag `--anchor-source {lerp, gt, hybrid}` exposes the anchor
option for ablation; default `hybrid`. The LERP option is kept as a
fallback for sequences where the raw ColoRadar dataset is
unavailable (e.g. a sanitised checkout of this repo).

### Candidate scoring: CUDA-consistent metric

This was a mistake in the 2026-04-18 implementation that we must fix:
**all candidates must be scored through the same CUDA renderer
cart_corr** against chirp k's GT, not through the lidar-voxel
objective. The lidar method can still *propose* a pose (it's useful
because its objective landscape is different), but the final winner
is picked by the CUDA cart_corr — which is what NVS training actually
optimises. This is the same pattern used in per-frame pass 2 B1-L.

### Edge-frame handling

For `F = first` or `F = last` of a scene's 9-frame window, one of
`pass2_{F-1}` or `pass2_{F+1}` is missing. Two options, each
principled:

**Option A (simpler):** fall back to the nearest two available
consecutive frames and LERP-extrapolate. For seq_1_frame_438 frame
434, use `pass2_{435}` and `pass2_{436}` and extrapolate backward to
frame 434's expected position. Then Stage 3 proceeds normally around
that (possibly less accurate) anchor. Accept the slightly wider error
bar on the anchor by loosening the gate.

**Option B (more work):** extend the pass-2 trajectory fit to cover
one extra frame on each side of the window, using the scene's next
actual frame from the raw ColoRadar data (if available). Out of
scope for Stage 3; defer.

Stage 3 uses Option A.

## Expected outcomes

Stage 3 is an experiment with three possible outcomes, all
acceptable:

1. **Most chirps accept the refinement** (say >50%). Then pose
   accuracy WAS contributing to the HO→UB gap in frame NVS, and
   re-running NVS with Stage-3 poses should improve the all-chirp
   variants. Measured improvement quantifies the effect.

2. **Most chirps fall back to LERP** (say <20% accepted). Then LERP
   was already at the signal-limited noise floor; pose accuracy is
   not the NVS gap driver. No further work needed; the artefact
   produced by Stage 3 is essentially identical to LERP and can be
   deleted after confirming.

3. **Mixed per-frame** — some frames benefit, others don't (a pattern
   emerges, e.g., "faster-moving frames benefit more"). In that case
   the per-chirp configs are worth keeping for ALL NVS training as a
   safety net at minimal overhead.

Whichever outcome, the experiment costs ~20 min/scene and settles the
question. The 2026-04-18 per-chirp run was *not* this experiment —
that run had no anchor, so the poses drifted freely. Stage 3 is the
proper test.

## Implementation plan

### File layout

- `mm25DGS_v5/preprocessing/alignment/per_chirp_alignment.py` — existing
  file. Will be **extended**, not replaced. Add:
  - `stage3_refine_chirp(ctx, anchor_pose, base_config, search_radius,
     prior_weight, gate_margin)` — the inner refinement
  - `run_stage3_for_scene(scene, ...)` — the driver that iterates
    (frame, chirp) and writes `cascaded_frame_<F>_chirp<CC>_aligned_pass3.json`
  - The existing stage-1 (per-chirp independent) and stage-2 (LOWESS
    smoothing) code stays but is marked deprecated in the docstring —
    Stage 3 is the recommended path.
- No changes needed to the v5 CUDA backend (`build_alignment_context`
  + `update_gt_for_chirp` already support `chirp_idx`).
- No changes needed to the trainer — Stage 3 JSON has the same
  schema as Stage 2, just a different suffix (`_aligned_pass3`). The
  trainer gets a new flag `--alignment-source {pass1,pass2,pass3}`
  that routes to the appropriate files. **Backward compatibility
  preserved**: omitting the flag reproduces today's per-frame LERP
  behaviour.

### Output layout

```
data/alignment_data/<scene>/cascade/per_chirp/
    cascaded_frame_<F>_chirp<CC>_aligned_pass3.json        ← final per-chirp pose
    cascaded_frame_<F>_chirp<CC>_alignment_log_pass3.json  ← per-chirp decision log
    pass3_summary.json                                     ← scene-level stats
```

`alignment_log_pass3.json` per chirp records:
- anchor pose (LERP deltas relative to F-1, F+1)
- three candidate scores (renderer, lidar, anchor itself)
- winner label + cc + pose-bound residual
- elapsed time

### Phases

| phase | what | time |
|---|---|---|
| **P1** Vendor the Stage 3 inner loop | ~150 LOC added to `per_chirp_alignment.py` | 1 hour |
| **P2** Smoke test on 1 frame (9 chirps of seq_1_frame_438 f=438) | verify: renderer + lidar candidates behave, gate accepts/rejects sensibly, poses are within expected bound of LERP | 15 min |
| **P3** Calibrate search radius + gate margin | sweep `search_radius ∈ {5, 10, 20, 50} mm` on one frame, measure acceptance rate and per-chirp cc improvement | 30 min |
| **P4** Run full Stage 3 on 2 target scenes | 2 scenes × 144 chirps × ~10 s/chirp = 48 min, parallelised across GPUs → ~25 min wall | 25 min |
| **P5** A/B test downstream impact on frame-NVS | re-run `HO (128)` and `UB (144)` variants using `--alignment-source pass3` vs the default. Compare test cc. | 2 × 2 scenes × ~60 min = 4 GPU-hours; 2 GPUs parallel → 2 hours | 
| **P6** Update `md/frame_nvs.md` | add Stage-3 columns, document whether the experiment validated or refuted the pose-accuracy hypothesis | 15 min |

Total: ~5 hours wall.

### Test predicates

- **P2 must pass**: for the 9 chirps tested, each runs to completion
  without error, and the per-chirp `winner_log` shows
  ≥1 renderer candidate and ≥1 lidar candidate scored.
- **P3 must show**: monotone-ish relationship between search radius
  and acceptance rate (bigger radius → more candidates beat the anchor).
  Pick the radius that gives ~30–50% acceptance rate (not 100% because
  that means the gate is meaningless; not 0% because that means no
  refinement is possible at that radius).
- **P4 must finish** in <60 minutes wall time on 2 GPUs.
- **P5 must either**:
  - Show ≥ +0.02 cc improvement on the all-chirp variants (Stage 3
    is useful), OR
  - Show ≤ ±0.005 cc difference (LERP was already signal-limited;
    Stage 3 can be retired).
- **P6 must document** the outcome numerically and add a column to
  `md/frame_nvs.md` with the Stage-3 HO and UB cc for both scenes.

### Optional extension (deferred)

If P5 shows Stage 3 helps, it's worth asking whether **joint
Stage-2 + Stage-3 optimisation** does even better. Currently pass-2
smooths over 9 frame-level poses independently of chirp-level signal.
A unified per-chirp-level pass-2 would fit the smoothed trajectory
over all 144 (frame, chirp) points using the per-chirp RA as a
weight. This is a 1-day engineering project and only worth doing if
Stage 3's standalone impact is meaningful.

## GT / IMU and the existing per-frame alignment

Stage 3 is the main consumer of the ColoRadar GT data, but the same
GT signal could tighten the **per-frame (Stage 2) alignment** as
well. Keep this in mind — it's a separate improvement, orthogonal
to Stage 3.

### What GT can add to Stage 2

Stage 2's current trajectory-prior is a LOWESS fit over the 9 pass-1
per-frame poses. That fit has no outside anchor — if pass-1 is
globally biased (as it sometimes is via the 2-DOF-only renderer
method that can't correct boresight z), Stage 2 can only smooth
around that biased estimate.

Using GT poses as an **additional** prior (scaled so it doesn't
dominate the radar-driven cost but contributes a regularisation
term) would give Stage 2 an outside reference. Specifically:

1. For each frame F, compute `gt_pose_F = interpolate(GT, cascade_ts[F])`
   then apply `base_to_cascade` to get the cascade-frame pose
   predicted by GT.
2. Add to Stage 2 B1-R / B1-L objective a term
   `λ_gt · ||pose - gt_pose_F||²` (per-DOF scaled). The weight
   `λ_gt` controls how much GT is believed: small (0.001-0.01) →
   radar-data dominates, GT is a soft guide; large (0.1+) → GT
   dominates, radar only adjusts fine details.
3. Quantify the effect: does adding a weak GT prior to Stage 2 shrink
   the frame-to-frame translation std on seq_1_frame_438 from the
   observed 153 mm down toward GT's 7 mm?

This was not done in the current pass-2 implementation because the
GT data is an external dependency (in `/home/adnan/Documents/Data/`,
not in the vendored repo `data/`). Before adding a GT-dependent
term to Stage 2, validate that every downstream user of pass-2
configs has access to the raw ColoRadar GT or can fall back to
GT-less alignment gracefully.

### What IMU can (and can't) add

IMU at 100 Hz is too slow for per-chirp (7.87 ms burst window, 1
sample). It's plenty fast for per-frame (200 ms frame period, ~20
samples per frame). Potential uses:

- **Motion classification**: integrate IMU acceleration to flag
  frames during which the rig accelerated hard (not walking-pace).
  Those frames may warrant different search radii / prior weights in
  Stage 2 / Stage 3. Cheap to do — a few lines of numpy.
- **Rotation rate prior**: gyroscope readings give angular velocity
  between frames. Integrate gyro over the ~200 ms inter-frame gap to
  predict the orientation delta, use as a prior on the frame-to-
  frame pose rotation in Stage 2. Useful only if pass-2 has
  orientation jitter (we observed translation jitter on seq_1;
  haven't quantified rotation jitter — worth an additional
  diagnostic).
- **Drift correction for missing GT**: on sequences where GT
  `groundtruth_poses.txt` is absent or unreliable, integrated IMU
  (starting from a known rigid frame, corrected by Kalman-filter
  state) could substitute. Not needed on the 7 target scenes.

Prior experience (per user, 2026-04-18) says the ColoRadar IMU is
"extremely noisy", so aggressive IMU reliance is discouraged. The
safest uses are IMU-as-classifier (did the rig move suddenly?) or
IMU-as-tie-breaker between GT and pass-1 (pick GT if they disagree,
since GT is the more reliable of the two for steady walking).

### Coordinate conventions (for implementers)

Everything below follows the conventions already present in
`mmir/preprocessing/config_utils.py::compute_world_pose_and_boresight`:

| quantity | convention |
|---|---|
| Cascade pose in world | `T_ws = T_wb @ T_bc` with `T_wb` from GT slerp and `T_bc` from `calib/transforms/base_to_cascade.txt` |
| Cascade boresight in world | `T_ws[:3, 0]` — the sensor-frame +X axis rotated to world frame |
| Cascade center in world (m) | `T_ws[:3, 3]` |
| Aligned config position units | millimetres (field `pos_mm`) — divide by 1000 for metres before arithmetic with GT |
| Groundtruth quaternion format | `[qx, qy, qz, qw]` — scipy's `Rotation.from_quat` order |

Extrinsic decoding (2026-04-18, this repo):

```
base_to_cascade.txt:  +89.95° about +Z  (cascade +X = base +Y ≈ "forward")
                      translation (base←cascade) = [0.03, 0.12, -0.09] m
base_to_imu.txt:      +179.98° about (+X+Y)/√2  (IMU-Z = base-(-Z), IMU-X = base +Y, IMU-Y = base +X)
                      translation (base←IMU) = [0, 0, 0]
base_to_lidar.txt:    ~+92° about -Z   (lidar +X = base -Y)
                      translation (base←lidar) = [-0.075, -0.02, 0.036] m
```

The IMU extrinsic is verified against the data: accelerometer at rest
reads `[0.78, 0.11, -9.77]` m/s² ≈ `[0, 0, -9.81]` on IMU-Z, which
`T_bi` maps to `[0, 0, +9.81]` in base frame — i.e. gravity points
in base `-Z` and IMU-Z is up. So anyone integrating IMU should:

1. Load raw IMU vectors in IMU frame.
2. Rotate via `T_bi[:3, :3]` to get vectors in base frame.
3. Rotate via `T_wb[:3, :3]` (from GT slerp) to get world-frame.
4. Subtract `[0, 0, -9.81]` from world-frame accel to remove gravity.

### Diagnostic results (2026-04-18, correct units + boresight convention)

**GT vs pass-2 absolute pose at the centre frame:**

| scene | |Δcenter| | ∠Δboresight |
|---|---:|---:|
| seq_1_frame_438 f=438 | 100 mm | **5.06°** |
| seq_2_frame_105 f=105 | 198 mm | **5.91°** |

Both scenes show a consistent ~5-6° boresight correction applied by
pass-2 on top of the GT-derived starting pose. The ~5° is
suspiciously stable across scenes — likely a systematic error in
`calib/transforms/base_to_cascade.txt` that pass-2 routinely fixes.
This tells us GT alone (without pass-2's correction) would be
5-6° wrong on every frame, so the recommended hybrid anchor must
always use pass-2 for orientation, not GT.

**Frame-to-frame motion σ (9-frame window):**

| scene | GT Δ-frame mean ± σ | pass-2 Δ-frame mean ± σ |
|---|---:|---:|
| seq_1_frame_438 | **249 ± 8.8 mm** | 468 ± **153** mm |
| seq_2_frame_105 | 255 ± 14.4 mm | 249 ± 28.2 mm |

On seq_1_frame_438, pass-2 frame-to-frame translation jitter is 17×
larger than GT's (std 153 mm vs 8.8 mm). This is the root cause of
the per-chirp LERP noise documented in the "Option A cons" section
above.

**Per-chirp GT-predicted displacement (centre frame):**

```
chirp 0..15 cumulative (mm):
seq_1_frame_438: 0.00 0.57 1.13 1.70 2.27 2.84 3.40 3.97 4.54 5.11 5.67 6.24 6.81 7.37 7.94 8.51
seq_2_frame_105: 0.00 0.66 1.33 1.99 2.65 3.32 3.98 4.64 5.31 5.97 6.63 7.30 7.96 8.62 9.29 9.95
```

Both match walking pace (~1.2 m/s × 7.4 ms ≈ 8.9 mm total) to within
1 mm. Per-chirp boresight rotation across the burst: 0.08-0.13° total
— effectively zero. Confirms Stage 3's refinement bound (±10 mm
translation, ±0.5° rotation) is physically calibrated.

### A second (pre-existing) issue surfaced by this analysis

The current trainer (`train_chirp_loop_nvs.py` and `train_frame_nvs.py`,
via `build_per_loop_poses`) places chirp 0 of frame F at the **LERP
midpoint of pass-2_{F-1} and pass-2_{F+1}** — not at pass-2_F directly.

Measured LERP-midpoint vs pass-2_F disagreement:

| scene | LERP-midpoint − pass-2_F |
|---|---:|
| seq_1_frame_438 f=438 | **458 mm** |
| seq_2_frame_105 f=105 | 24 mm |

This is historical: the LERP-between-neighbours approach was
introduced when REF frame 135's pass-1 alignment was z-flipped, and
we were asked to skip frame 135 as its own anchor and use 134+136
instead. Pass 2 now fixes those broken REFs, so using pass-2_F
directly as the per-frame anchor is both **cleaner** and (on
noise-prone scenes like seq_1) **458 mm more accurate**.

Fixing this is independent of per-chirp refinement — it applies at
the frame level first. The recommended hybrid anchor (Option C above)
already does this by construction, because it uses pass-2_F as the
absolute anchor for chirp 0. But it is worth noting that this fix
should be backported to the chirp-loop NVS trainer too — see
implementation notes below.

### Implementation impact (beyond Stage 3)

The Option C hybrid anchor implicitly fixes both the LERP-midpoint
issue and the sub-frame motion issue in one change. The same formula
— "pass-2_F as absolute anchor, GT for relative per-chirp motion" —
can be applied at the **frame-level only** (single pose per frame,
shared across all 16 chirps) for the existing chirp-loop NVS trainer
without doing any per-chirp alignment at all. This gives a lightweight
win estimated at:

| scene | LERP-midpoint error | 1st-chirp HO cc gain if hybrid is used (order-of-magnitude estimate)   |
|---|---:|---|
| seq_1_frame_438 | 458 mm | potentially substantial — anchor moves by half a metre |
| seq_2_frame_105 | 24 mm | negligible — within existing noise |

This is ALSO worth a separate mini-experiment before Stage 3's
per-chirp refinement: retrain frame-NVS on seq_1 with the hybrid
frame anchor (but not per-chirp), see if HO and UB move. If seq_1's
HO cc shifts meaningfully from the current 0.49, that quantifies
the frame-anchor portion of the Stage 3 benefit. Stage 3's remaining
contribution is then isolated to the per-chirp-refinement part.

### What IMU adds to Stage 3 specifically

With the GT's 9 Hz rate giving sub-mm interpolation for per-chirp
timing, IMU at 100 Hz adds no signal at the chirp timescale. IMU is
useful at the **scene level**:

- **Motion classification during training data generation**: flag
  frames where the rig accelerated non-uniformly (e.g. a stop, a
  turn). Stage 3 / hybrid could relax or tighten its per-chirp
  refinement bound for those frames.
- **GT sanity**: IMU integration over 0.2 s (a radar frame period)
  should match the GT-derived relative motion to within IMU
  noise. Useful for detecting GT dropouts or timestamp
  desynchronisation; not an online correction.

Prior experience (user, 2026-04-18) says the ColoRadar IMU is
"extremely noisy." The analyses above suggest GT is a much more
trustworthy signal for pose-related work on these sequences; use IMU
only for the diagnostic roles above.

## Constraints and gotchas

- **Do not re-align chirp 0.** The LERP anchor for chirp 0 at α=0.5 is
  essentially identical to the Stage 2 per-frame pose for frame F
  (they were produced by different routes but agree in the
  well-aligned cases). Stage 3 should skip refinement of chirp 0 and
  copy the per-frame Stage 2 pose directly, to avoid introducing
  inconsistency.
- **Pose bound must be componentwise.** A single scalar "Euclidean
  distance" bound masks the failure mode where translation is small
  but rotation drifts a degree. Use the radii above as per-DOF bounds.
- **Prior-weight re-use.** The per-frame pass 2 uses λ=0.05 for its
  prior penalty on `||pose - prior||²`. The chirp-level search has
  a tighter radius, so effectively the prior dominates at the edges
  of the search box already. λ=0.05 should remain.
- **Test-RA cart_corr scoring requires sample_grid.** Build it once
  per frame inside the ctx, not per chirp. Already handled by
  `build_alignment_context`.
- **Gate margin must exceed MC noise.** The per-iter `cart_corr`
  variance in a CUDA render is O(0.003). A gate margin of 0.005 is
  2σ above that. Lower margins will accept noise; higher margins
  will miss real improvements.
- **LOWESS pass-2 smoothing is deprecated.** Stage 3 replaces it.
  Keep the code path in `per_chirp_alignment.py` for reproducibility
  of the 2026-04-18 experiment, but the new CLI entry point
  (`--stage 3`) bypasses it.

## Why this is the right design

The 2026-04-18 experiment tried to answer "what's the best per-chirp
pose?" by aligning each chirp independently. That question is
ill-posed at single-chirp SNR: the objective has many near-equivalent
optima at the mm-cm scale. The right question is:

> **Given the validated per-frame pass-2 pose and its neighbours, is
  there evidence in chirp k's RA that the chirp's pose differs from
  the LERP estimate by more than noise?**

Stage 3 frames this hypothesis test correctly by anchoring the search
at LERP and requiring a statistically meaningful improvement to
accept a deviation. Its output is either "yes, here's the refinement"
or "no, LERP was right" — both actionable.
