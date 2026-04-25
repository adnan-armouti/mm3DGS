# DART Baseline Plan (minimal-changes benchmark for NeurIPS 2026)

Upstream: WiseLabCMU/dart (CVPR 2024). Paper: arXiv:2403.03896.
JAX + dm-haiku + optax. Implicit neural reflectance field, Doppler-aware
radar NVS via doppler-column integration.

Goal: produce a single defensible number per scene against mm3DGS. NOT the
strongest possible DART. Every deviation is justified and called out as a
limitation.

------------------------------------------------------------------------

## 1. Repo summary

DART fits a 3D reflectance/occupancy field from a radar + IMU/SLAM
trajectory. Unlike NeRF, it does NOT integrate along camera rays at render
time; it integrates along Doppler-iso-circles: for each (frame, doppler-bin)
pair, it samples the locus of unit-sphere directions whose Doppler
projection onto the ego velocity equals that bin, then ray-marches range to
produce one "Doppler column" of the range-Doppler-azimuth (RDA) cube.

- Forward model: `VirtualRadar._render_column` in
  `WiseLabCMU/dart:dart/sensor.py:112-148`. For each range bin it samples
  `k` directions on the Doppler iso-circle (via `project_angle`
  `dart/pose.py:83-110`), evaluates the implicit field `sigma(x, dx)` to
  get per-point (sigma, alpha), accumulates transmittance cumulatively
  across range, multiplies by antenna gain, and sums over the azimuth/k
  samples. Output: one (Nr, Na) slab per doppler column, scaled by an
  integration weight `psi_min/pi/s`.
- Field: hash-grid NGP with plenoctree-style SH view-dependence ("ngpsh"),
  `WiseLabCMU/dart:dart/fields/ngp.py`. Trained field is `sigma : R^3 x S^2
  -> (reflectance, alpha)`.
- Loss: L1 (default) on per-column (Nr, Na) output vs. ground truth
  range-doppler column (`WiseLabCMU/dart:dart/components/loss.py:13-54`).
  Optional sqrt/log magnitude weighting. Adjustable alpha clipping
  schedule (`train.py:36-39`).
- Training loop: mini-batch over Doppler columns, Adam, default lr=0.01,
  epochs=3, batch=1024 (`WiseLabCMU/dart:train.py:26-45`,
  `dart/script.py:15-55`).
- Native output: 3D field. For evaluation, `tools/evaluate.py` synthesises
  the full RDA cube at each trajectory pose by calling `sensor.render`
  (`WiseLabCMU/dart:dart/sensor.py:170-197`).

Input representation expected: range-Doppler-azimuth per frame (`rad` of
shape `[Ni, Nr, Nd, Na]`), plus per-frame pose (x, A), ego-velocity (v),
ego-speed (s), and (p,q) basis (`WiseLabCMU/dart:docs/datasets.md`). The
dataloader then EXPLODES each frame into Nd "doppler columns" each of
shape (Nr, Na) with its own per-column doppler value and weight
(`WiseLabCMU/dart:tools/dataset.py:40-66`,
`WiseLabCMU/dart:dart/dataset.py:100-170`).

------------------------------------------------------------------------

## 2. Upstream data format

Source of truth: `WiseLabCMU/dart:docs/datasets.md` (Training Dataset
Format section) and `WiseLabCMU/dart:tools/dataset.py:40-99`.

DART reads a single `data.h5` per dataset plus a `sensor.json`. Upstream
also expects a `trajectory.h5` and `radar.h5` that are collapsed into
`data.h5` by `manage.py dataset -p <dataset>`.

### `sensor.json`
Consumed by `VirtualRadar.from_config`
(`WiseLabCMU/dart:dart/sensor.py:38-52`):
- `r : [r_min, r_max, Nr]`  — `linspace` args for range bins (metres).
- `d : [d_min, d_max, Nd]`  — `linspace` args for Doppler bins (m/s).
- `k : int`                 — stochastic ray samples per Doppler column
                              (CLI default 128, `train.py:20`).
- `gain : str`              — one of `rect`, `awr1843boost`,
                              `awr1843boost_az8`
                              (`dart/components/antenna.py`). "az8" means
                              `Na=8` azimuth bins produced by analytic DFT
                              over an 8-element virtual array.

### `data.h5` — per-column records (exploded by upstream preprocessing)

Columns are created by `tools/dataset.py:_load_data`
(`WiseLabCMU/dart:tools/dataset.py:40-66`); one entry per (frame,
doppler-bin):

| key          | dtype / shape      | semantic |
| ------------ | ------------------ | -------- |
| `x`          | `float[N,3]`       | radar position, FLU world (m) |
| `A`          | `float[N,3,3]`     | rotation sensor->world (FLU) |
| `v`          | `float[N,3]`       | velocity direction (unit) |
| `s`          | `float[N]`         | speed (m/s) — required for Doppler iso-circle |
| `p`, `q`     | `float[N,3]`       | orthonormal basis derived from v |
| `weight`     | `float[N]`         | integration weight = psi_min/pi/s |
| `doppler`    | `float[N]`         | doppler velocity value (m/s) |
| `doppler_idx`| `uint16[N]`        | metadata only |
| `frame_idx`  | `uint16[N]`        | metadata only |
| `rad`        | `float16[N,Nr,Na]` | ground-truth column of RDA cube |

Before slicing into columns, the full 4D cube is `[Ni, Nr, Nd, Na]`
(`tools/dataset.py:56-66`). That is the format the DART forward model
actually targets. Columns with `weight == 0` (Doppler fully outside the
iso-circle) are filtered out (`tools/dataset.py:75-83`).

### Pose convention (critical)
Front-Left-Up (FLU): `+x` forward (FOV centre), `+y` left, `+z` up.
`A` maps sensor-space -> world-space. `v` is in WORLD frame and gets
rotated to sensor space inside `make_pose`
(`WiseLabCMU/dart:dart/pose.py:17-45`).

Ego-velocity (`v`, `s`) is MANDATORY — the entire Doppler iso-circle
construction divides by `s` and projects onto `v`
(`dart/pose.py:83-110`, `dart/sensor.py:68-93`). DART will not train on
static frames.

------------------------------------------------------------------------

## 3. Adapter (minimum transformation, bullet list, no code)

Task: produce one `data.h5` + `sensor.json` per scene, in DART's format,
from `data/seq_X_frame_Y/`. Each bullet names the source and destination.

- Source: `data/seq_X_frame_Y/radar/cascaded_frame_{131..139}.npy`
  (complex128, `[16, 16, 12, 256]` = chirps, RX, TX, ADC)  OR
  `data/seq_X_frame_Y/radar/single_chip_frame_{261..277 step 2}.npy`
  (complex128, `[128, 4, 3, 128]`).
  Destination: `data.h5["rad"]` shape `[Ni, Nr, Nd, Na]`, float16.
  Steps (adapter only; no model change):
    1. MIMO collapse to a virtual array along azimuth (see Section 4).
    2. Range FFT along ADC axis -> `Nr`.
    3. Doppler FFT along chirp axis -> `Nd`.
    4. Azimuth beamforming (FFT over virtual RX) -> `Na`.
    5. Magnitude (|.|), then crop range to bins 15..110 BEFORE `Nr`
       passed to DART (Section 6 uses same crop). Keep the full Nd.
    6. Normalise to the range DART expects (per `tools/dataset.py:45`
       the upstream norm is `1e4`; we will pick a per-scene norm so the
       99th percentile of `rad` sits in a sane range — note this as a
       deviation in Section 9).
- Source: mm3DGS per-frame sensor pose from
  `data/seq_X_frame_Y/configs/cascaded_frame_{N}_aligned.json` (the
  aligned variant used by mm3DGS training). Destination: `data.h5`
  fields `x [N,3]`, `A [N,3,3]`.
  - Convert pose from mm3DGS frame convention to DART's FLU. Mitsuba in
    mm3DGS uses +x forward in scene frame (consistent with FLU +x) — but
    verify handedness and y/z axis flips (needs verification — check
    `mmir/preprocessing/alignment/cascaded_alignment.py`).
  - `A` in DART is sensor->world. In mm3DGS the stored rotation is
    scene->world; confirm it's the matching direction (needs verification
    — check Section 9).
- Source: `data/v_ego_cache/seq_X_frame_Y/frame_{N}_v_ego.npy`.
  Destination: `data.h5["v"] [N,3]` (unit vector) and `data.h5["s"] [N]`
  (speed, m/s).
  - We have a cache of ego velocities per frame. Load, check magnitude
    is non-zero (DART errors on `s=0`), set `v = v_ego / ||v_ego||`,
    `s = ||v_ego||`. If mag < ~0.05 m/s for any frame, drop that frame
    (Section 9 risk: if the held-out test frame is too slow, the
    Doppler iso-circle integration weight `psi_min/pi/s` blows up or
    drops to 0).
  - Derive `p, q` with `dart.pose.make_pose` directly; do NOT re-derive
    manually.
- Source: radar-config fields from
  `data/seq_X_frame_Y/configs/cascaded_frame_{N}.json` (range
  resolution = c / (2 * BW); Doppler resolution derived from chirp
  period and Nd).
  Destination: `sensor.json` fields `r`, `d`, `k`, `gain`.
  - `r = [r_min, r_max, Nr]`. With 8 MHz sample rate and 79 MHz/us slope
    over 256 ADC, range resolution ~ 0.19 m and unambiguous range
    ~ 48 m. After cropping bins 15..110 we get Nr=96 over roughly
    [2.85 m, 20.9 m]. For single-chip (128 ADC), re-derive from its
    config (needs verification — check
    `data/seq_X_frame_Y/configs/cascaded_frame_*.json` + the single-chip
    equivalent; the single-chip params may be in a different config
    file).
  - `d = [-d_max, d_max, Nd]`. With 16 chirps (cascaded) Nd=16; with
    128 chirps (single-chip) Nd=128.
  - `k = 128` (upstream CLI default; leave alone — Section 7).
  - `gain`: upstream only ships `rect`, `awr1843boost`, and
    `awr1843boost_az8`. Our cascaded antenna (MMWCAS, 12x16 = 192 virt)
    and our single-chip (IWR1443, 3x4 = 12 virt) do NOT match these
    exactly. Use `awr1843boost_az8` (closest, designed for IWR1843 and
    TI cascade-board single-chip daughter boards) and record the gain
    mismatch as a limitation (Section 9). DO NOT rewrite antenna
    patterns — minimal-changes rule.
- `doppler`, `doppler_idx`, `frame_idx`, `weight`: DO NOT write these
  directly. Run DART's upstream `manage.py dataset -p <scene>` (
  `WiseLabCMU/dart:tools/dataset.py`) over the intermediate
  `radar.h5 + trajectory.h5` we've produced. This guarantees `weight =
  psi_min/pi/s` is computed with DART's own `get_psi_min`
  (`dart/sensor.py:54-81`) and that invalid columns are filtered the
  same way as on upstream datasets.

Out-of-scope for adapter: any phase/complex retention. DART operates on
magnitude RDA cubes only (`tools/dataset.py:45`); our complex
range-Doppler cubes get magnituded and lose phase. This is consistent with
upstream.

------------------------------------------------------------------------

## 4. Geometry / chirp reconciliation — REVISED 2026-04-25

### Decision: CASCADED (16 chirps)

This section is REVISED from the original plan. The original plan picked
single-chip for Doppler viability and antenna-pattern match. **Per user
direction (2026-04-25): cross-baseline comparability requires that ALL
three baselines (RadarSplat, RadarFields, DART) use the same sensor
modality.** RadarSplat and RadarFields are cascade-only by construction.
DART is therefore also moved to cascade, and the GT used for evaluation
is the cascade GT (matching mm3DGS-v6/v7 headline numbers).

What this costs:

1. **Coarser Doppler axis.** 16 chirps → 16 Doppler bins (vs upstream's
   256). PLAN Section 9(a) flagged this as the original gating concern;
   we proceed and accept the resolution loss as a documented limitation.
2. **Azimuth-FFT downsampled to 8 bins** to match upstream's
   `awr1843boost_az8` gain function (the only stock gain that ships 8
   azimuth bins). The cascade's 86-element row-0 virtual array is
   coherently FFT'd to 8 azimuth bins (vs the natural 127). This is a
   resolution loss but no architecture change.
3. **Verified `psi_min` viability.** Our ego speeds are 1.3–1.6 m/s
   (per ``data/v_ego_cache``). With 16 Doppler bins covering ±d_max ≈
   5 m/s, the iso-circle integration weight ``psi_min/pi/s`` is well
   above zero across the working window. The PLAN Section 9(a) failure
   mode (collapse to weight=0) does not trigger.
4. **Cascade GT for evaluation.** The same cascade GT used by
   RadarSplat / RadarFields / mm3DGS-v6/v7 is used here. DART renders a
   cascade-shaped (Na=8 azim × Nr=256 range × Nd=16 doppler) cube; we
   sum over Doppler to get a (Na=8, Nr=256) RA polar, then run through
   ``mmir.data.ra_utils.ra_polar_to_cartesian`` and
   ``compute_cart_ra_metrics`` against the cascade GT cart.

### What we change (adapter + config only)

- `sensor.json`: `r=[0, 256*range_res, 256]`, `d=[-5.0, 5.0, 16]`,
  `k=128`, `gain=awr1843boost_az8`.
- Adapter pipeline (per cascade frame):
  1. Load cascade ADC `(16, 16, 12, 256)` complex.
  2. Chirp-by-chirp build the 86-element row-0 virtual array.
  3. Range-FFT along ADC axis → `Nr=256`.
  4. Doppler-FFT along 16-chirp axis → `Nd=16`.
  5. Azimuth-FFT to 8 bins → `Na=8`.
  6. Magnitude → `(Nr, Nd, Na) = (256, 16, 8)`.

### What we do NOT change

- No chirp synthesis, no Doppler up-sampling, no fabricated bins.
- No new antenna-pattern function (use stock `awr1843boost_az8`).
- No model architecture / loss / sensor.py / pose.py edits.

### Single-chip is now out of scope

The original `data_dart/<scene>/data.h5` (built from single-chip ADC)
and the SC-specific helpers in the adapter are replaced.

------------------------------------------------------------------------

## 5. NVS split

Protocol (shared across all baselines and mm3DGS): train on outer 8
frames per scene, test on middle (index 4) frame.

### Cascaded frame indices (for reference; we are NOT training DART on these)

| scene             | cascaded train (8)             | cascaded test (1) |
|-------------------|--------------------------------|-------------------|
| seq_0_frame_135   | 131,132,133,134,136,137,138,139| 135               |
| seq_0_frame_390   | 386,387,388,389,391,392,393,394| 390               |
| seq_1_frame_185   | 181,182,183,184,186,187,188,189| 185               |
| seq_1_frame_438   | 434,435,436,437,439,440,441,442| 438               |
| seq_2_frame_105   | 101,102,103,104,106,107,108,109| 105               |
| seq_2_frame_160   | 156,157,158,159,161,162,163,164| 160               |
| seq_2_frame_300   | 296,297,298,299,301,302,303,304| 300               |

(Needs verification — check
`data/seq_*/radar/cascaded_frame_*.npy` listing to confirm the ±4 window
convention; the repo uses frame names like `seq_0_frame_135` to mean the
MIDDLE frame, with ±4 neighbours on disk.)

### Single-chip frame indices (actual DART training)

Single-chip frames are paired with cascaded at ~2:1 cadence (seen for
seq_0_frame_135: cascaded 131..139 vs single-chip 261..277 step 2 = 9
frames). Train on index {0,1,2,3,5,6,7,8} of the sorted 9-file list,
test on index 4.

| scene             | single-chip train (8 indices by filename)                  | single-chip test (1) |
|-------------------|-------------------------------------------------------------|----------------------|
| seq_0_frame_135   | 261,263,265,267,271,273,275,277                             | 269                  |
| seq_0_frame_390   | (needs verification — ls `data/seq_0_frame_390/radar/`)     | index 4              |
| seq_1_frame_185   | (needs verification — ls `data/seq_1_frame_185/radar/`)     | index 4              |
| seq_1_frame_438   | (needs verification — ls `data/seq_1_frame_438/radar/`)     | index 4              |
| seq_2_frame_105   | (needs verification — ls `data/seq_2_frame_105/radar/`)     | index 4              |
| seq_2_frame_160   | (needs verification — ls `data/seq_2_frame_160/radar/`)     | index 4              |
| seq_2_frame_300   | (needs verification — ls `data/seq_2_frame_300/radar/`)     | index 4              |

Needs verification for each remaining scene: `ls
data/seq_X_frame_Y/radar/single_chip_frame_*.npy | sort`, take the 9
files, pick index 4 as test.

Enforcement in DART: `doppler_columns` splits val by a proportion via
`pval` (`WiseLabCMU/dart:dart/dataset.py:104-139`). We will NOT use
`pval`; instead we will:
- Write a pre-filtered `data.h5` that contains ONLY the 8 training
  frames, and a second `data_test.h5` containing only frame index 4.
- Train with `pval=0`, `iid=False`.
- Evaluate (Section 6) over `data_test.h5` by running `sensor.render`
  at the held-out pose.

NVS interpolation guarantee: middle-frame-bracketed by definition —
consistent with the "NVS = interpolation only" rule.

------------------------------------------------------------------------

## 6. Metrics harness

mm3DGS metrics live at:
- `mmir/evaluation/eval_training_ra_v2.py` (orchestration)
- `mmir/evaluation/utils/metrics.py::compute_cart_ra_metrics` (Pearson
  on Cartesian RA image + range-profile corr, bins 15..110).

DART native output is RDA (Nr, Nd, Na). mm3DGS metric expects RA
(Nr, Na) in Cartesian. Procedure:

1. Load trained DART checkpoint
   (`WiseLabCMU/dart:tools/evaluate.py:48-87`).
2. For the held-out pose of each scene, call `dart.render` (wraps
   `sensor.render`,
   `WiseLabCMU/dart:dart/sensor.py:170-197`). Output:
   `[Nr, Nd, Na]` float16.
3. Collapse the Doppler axis: RA = sum over Nd of |RDA|. This matches
   how the ground-truth RA is computed in mm3DGS (incoherent Doppler
   integration). Needs verification — check
   `mmir/evaluation/utils/metrics.py` for the exact GT RA definition
   (per-bin magnitude vs. coherent sum).
4. Crop range axis to bins 15..110 (as adapter already did — but
   confirm Nr of DART output matches the cropped Nr).
5. Interpolate to Cartesian RA grid using the SAME function mm3DGS
   uses (reference: `mmir/evaluation/utils/metrics.py`). Do NOT reinvent
   the Cartesian warp.
6. Pipe (pred_cart_RA, gt_cart_RA) into
   `compute_cart_ra_metrics`. Record Pearson RA, range-profile Pearson.
7. Average over 7 benchmark scenes.

Important: we do NOT call DART's own `metrics.py` (which computes SSIM
over RD images). The table we report to reviewers uses mm3DGS's metrics
only, for consistency.

------------------------------------------------------------------------

## 7. Env setup

Conda env name: `dart` (isolated from `mmir`).

Upstream pins (`WiseLabCMU/dart:requirements-pinned.txt`):
- python == 3.11
- CUDA == 11.8
- cuDNN matching CUDA 11.8
- jax == 0.4.10 (with `jax[cuda11_local]` wheel; manual per README)
- numpy == 1.26.0, scipy == 1.10.1, tensorflow == 2.12.0
- dm-haiku == 0.0.9, optax == 0.1.5
- h5py == 3.8.0, matplotlib == 3.7.1, pandas == 2.0.2
- jaxtyping == 0.2.15, beartype == 0.14.0

NVIDIA driver floor for CUDA 11.8 runtime: >= 450.80.02 (any driver on
a functioning 4090 exceeds this).

Isolation rationale:
- jax 0.4.10 + CUDA 11.8 frequently conflicts with parallel pytorch
  installs (pytorch bundles CUDA 12.x runtime). `mmir` env is pytorch.
  A clean `dart` env prevents libcudart / libcudnn shadowing.
- Use `conda create -n dart python=3.11`, then `pip install --upgrade
  "jax[cuda11_local]==0.4.10"`, then `pip install -r
  requirements-pinned.txt`. Confirm `jax.devices()` returns
  `[cuda(id=0), cuda(id=1)]` before training.

------------------------------------------------------------------------

## 8. Compute estimate

Upstream defaults (`WiseLabCMU/dart:train.py:26-34`):
- epochs: 3
- batch: 1024 Doppler columns
- k (samples per column): 128

Per scene dataset size (single-chip, 8 train frames):
- Ni = 8, Nd = 128, so total raw columns = 1024. After `weight > 0`
  filtering, expect ~500-1000 usable columns (Section 9 risk — some
  frames may have too-low ego speed).
- Batch size 1024 means ~1 batch/epoch. Upstream datasets have ~100k
  columns — our 8-frame scene is ~100x smaller. We SHOULD scale
  `epochs` up by ~100x to see comparable training, but upstream's 24 h
  / per-scene cap limits us.
- Proposal: keep `epochs=3` initially, measure seconds/epoch on the
  first scene, then scale to 24 h cap. Document the epoch count we
  actually used.

Per-iter cost: dominated by `k * Nr * Nd = 128 * 96 * 128 ~= 1.6M`
field evaluations per batch. Empirically the hash-grid NGP evaluates
~10M points/s on a 4090 => ~0.16 s/batch. With very few batches/epoch,
even 10k epochs is ~20 min. Most of the 24h budget will go to JIT
compile, checkpoint IO, and evaluation.

XLA first-step compile overhead: 30-120 s cold for a hash-grid haiku
module + `jit(step)` (`WiseLabCMU/dart:dart/dart.py:140-160`). Once.

Total over 7 scenes serial: budget 24 h per scene -> 7 days wall-time
if we saturate; realistic ~12-24 h total if training is cheap and we
cap epochs. Two 4090s in parallel via `CUDA_VISIBLE_DEVICES=0` and
`=1` would halve wall-time.

------------------------------------------------------------------------

## 9. Open questions / risks

Ordered by severity; all items that require disk/file inspection are
tagged "needs verification — check <path>".

(a) **Doppler spectrum viability at 128 chirps on slow-moving ego.**
    DART's `weight = psi_min/pi/s` is proportional to `1/s`. If our
    rover ego speed is below ~0.5 m/s, most Doppler bins fall outside
    the observable iso-circle and `weight -> 0`.
    Needs verification — check `data/v_ego_cache/seq_*/frame_*_v_ego.npy`
    magnitudes for typical ego speed distribution. If median is
    < 0.3 m/s, DART will have almost no training signal.

(b) **Per-scene vs. multi-scene training.** Upstream runs one model per
    dataset (`WiseLabCMU/dart:Makefile:21`). We do the same (per-scene).
    Confirmed — no deviation.

(c) **Ego-velocity source compatibility.** Our `v_ego_cache/` files are
    `frame_{N}_v_ego.npy`. Needs verification — check exact shape,
    dtype, frame (world vs. sensor), handedness, and unit (m/s vs.
    mm/s vs. pixels/frame). Upstream wants world-frame m/s
    (`dart/pose.py:17`).

(d) **Single-chip integration time across NVS split.** 128 chirps per
    frame at upstream ~500 us/chirp is ~64 ms; frame-to-frame spacing
    at our `step 2` is longer. Needs verification — check
    `data/seq_X_frame_Y/configs/*single_chip*.json` or the companion
    cascaded config for inter-frame dt. If our test frame's pose is
    NOT bracketed by train frames in actual rover trajectory (e.g. if
    the rover reverses direction between frame 4 and neighbours), NVS
    is not interpolating.

(e) **MIMO virtual-array handling.** DART was built for
    single-TX-single-RX OR IWR1843 (3x4 = 12 virt) with
    `awr1843boost_az8` gain. IWR1443 is also 3x4. This should work.
    For cascaded (192 virt), there is no stock support. This is the
    main reason for choosing single-chip (Section 4). Needs
    verification — check the single-chip pattern at
    `assets/antenna_pattern/IWR1443/pattern_76.npy` shape vs.
    `awr1843boost_az8` assumption of 8 azimuth bins; we may need to
    project to 8 azimuth bins via the same DFT beamforming.

(f) **Antenna gain mismatch.** `awr1843boost` is an AWR1843 pattern; we
    are using IWR1443. Both are 77 GHz TI single-chip radars with
    similar layouts, but the gain envelope differs. Limitation — not a
    blocker. Document in paper.

(g) **Pose-frame conversion.** mm3DGS stores pose in Mitsuba/scene
    frame (not guaranteed FLU). Needs verification — check
    `mmir/preprocessing/alignment/cascaded_alignment.py` and
    `data/seq_X_frame_Y/configs/*aligned*.json` for the rotation and
    translation convention. A wrong handedness silently flips Doppler
    sign and trains DART on a mirrored scene.

(h) **`weight` filtering could drop the test frame.** If the
    held-out middle frame has ego speed outside DART's observable
    Doppler, its `weight == 0` columns are dropped and we cannot
    synthesise an RDA cube for it. Needs verification — run the
    adapter on ONE scene end-to-end before committing to all 7.

(i) **Normalisation constant.** Upstream uses `norm=1e4`
    (`tools/dataset.py:45`). Our radar cube magnitudes are in
    different units. Picking a bad `norm` bricks the L1 loss
    (either saturates or vanishes). Limitation: pick per-scene
    `norm = percentile(rad, 99)`; document as a deviation.

(j) **Loss asymmetry vs. mm3DGS.** DART loss is L1 on magnitude
    RDA columns. mm3DGS loss is different. This is fine — we are
    comparing trained-model outputs under the same eval metric, not
    the losses themselves. But reviewers may ask.

------------------------------------------------------------------------

## 10. Out-of-scope

The following are NOT part of this baseline; DO NOT implement:

- Hyperparameter tuning (lr, batch, k, epochs beyond scaling to 24h).
- Model-architecture variants (`grid`, `ngp`, `ngpsh2`) — use `ngpsh`
  (paper default) only.
- Adjustment module (`--adj`) — upstream default disables it
  (`WiseLabCMU/dart:train.py:40`); leave at `Identity`.
- Alpha-clipping schedule tweaks; use upstream default 0.05
  (`train.py:41`).
- Custom antenna patterns for MMWCAS or IWR1443.
- Doppler chirp synthesis / upsampling / interpolation.
- Rewriting `sensor.py` to accept complex RDA cubes.
- Cascaded-radar adapter (see Section 4 rationale).
- Multi-scene joint training — per-scene only.
- Using DART's native SSIM metric
  (`WiseLabCMU/dart:tools/metrics.py`) — we use mm3DGS metrics.
- Camera rendering / slice video / map video — pure training + RA
  synthesis + metric.
- Pose-adjustment regularisation search.
- Any evaluation on the transfer sensor (i.e. cascaded) for a
  single-chip-trained DART — document but do not run.

------------------------------------------------------------------------

## 11. GPU-safety preamble (MANDATORY first step of any execution plan)

Before running ANY DART training or evaluation:

1. Run `nvidia-smi`.
2. If ANY process is active on ANY GPU (column "GPU-Util" > 0 OR any
   PID listed under "Processes"), ABORT. Partial occupancy does NOT
   count as free.
3. Only when both GPUs report 0% util AND no listed processes, proceed.
4. Pin the job to a specific device with `CUDA_VISIBLE_DEVICES=<0|1>`
   AND `--device 0` (DART CLI flag, `train.py:14`). Never rely on
   default device selection.
5. When running across 7 scenes, pin scenes 1-4 to GPU 0 and scenes
   5-7 to GPU 1 in separate shell sessions, re-checking `nvidia-smi`
   between launches.

This rule overrides convenience. If `nvidia-smi` is ambiguous (e.g.
ghost process pinned memory but 0% util), still abort.

------------------------------------------------------------------------

## Summary footer

- Radar choice: single-chip (128 chirps) — see Section 4.
- NVS split: standard 8-train / 1-test middle-frame-bracketed —
  viable CONDITIONAL on risk (a) and (h) checking out (Section 9).
- Biggest open question: risk (a) — whether our ego speed distribution
  yields enough non-zero-weight Doppler columns for DART to train at
  all. Must be checked before committing compute.
