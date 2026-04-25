# Radar-NVS Baselines for mm3DGS (NeurIPS 2026)

This directory plans three radar-NVS baselines benchmarked against mm3DGS under
an IDENTICAL NVS protocol and metrics. Per-baseline plans live in:

- [radarsplat/PLAN.md](radarsplat/PLAN.md) — Gaussian splatting, 360° spinning radar (umautobots/radarsplat)
- [radarfields/PLAN.md](radarfields/PLAN.md) — neural field, 360° spinning radar (princeton-computational-imaging/RadarFields)
- [dart/PLAN.md](dart/PLAN.md) — Doppler-aware NeRF, single-chip radar (WiseLabCMU/dart)

Each `PLAN.md` is self-contained (11 numbered sections). This README captures the
shared infrastructure that should be factored ONCE rather than re-implemented three
times. It does not repeat per-baseline detail.

---

## Shared contract

All three baselines must produce, for every held-out frame, a rendered image that
can be reduced to the same two scalars mm3DGS reports:

- RA correlation — Pearson correlation of the Cartesian-resampled Range-Azimuth
  magnitude image, rendered vs. ground truth.
- Range-profile correlation — Pearson correlation of per-range-bin magnitude
  summed over azimuth.

Both metrics are computed over range bins 15..110 (drop TX-RX coupling leakage
and DFT wrap-around) and averaged over the seven benchmark scenes.

Reference metric code:
- [mmir/evaluation/eval_training_ra_v2.py](../mmir/evaluation/eval_training_ra_v2.py)
- [mmir/evaluation/utils/metrics.py](../mmir/evaluation/utils/metrics.py) — `compute_cart_ra_metrics`
- [mmir/data/ra_utils.py](../mmir/data/ra_utils.py) — `adc_to_ra_image_numpy`, `ra_polar_to_cartesian`

**No baseline introduces its own metric code.** The harness feeds each baseline's
rendered RA (or ADC that the harness reduces to RA) through the same functions
used by mm3DGS. This is non-negotiable for reviewer defensibility.

---

## Benchmark scenes

Seven scenes under `data/`. Excluded: `seq_1_frame_277`, `seq_0_frame_451`.

| scene | cascaded center | cascaded train (8) | cascaded test (1) | single-chip center | single-chip test (1) |
| --- | ---: | --- | ---: | ---: | ---: |
| seq_0_frame_135 | 135 | 131,132,133,134,136,137,138,139 | 135 | 269 | 269 |
| seq_0_frame_390 | 390 | 386,387,388,389,391,392,393,394 | 390 | 779 | 779 |
| seq_1_frame_185 | 185 | 181,182,183,184,186,187,188,189 | 185 | 369 | 369 |
| seq_1_frame_438 | 438 | 434,435,436,437,439,440,441,442 | 438 | 875 | 875 |
| seq_2_frame_105 | 105 | 101,102,103,104,106,107,108,109 | 105 | 209 | 209 |
| seq_2_frame_160 | 160 | 156,157,158,159,161,162,163,164 | 160 | 319 | 319 |
| seq_2_frame_300 | 300 | 296,297,298,299,301,302,303,304 | 300 | 599 | 599 |

Single-chip frames are acquired at 2× the cascaded rate, so the pairing is
`sc_center = cascaded_center * 2` (−1 in some scenes due to acquisition offset —
each scene ships with its actual 9-frame sequence under `radar/single_chip_frame_*.npy`;
the table above lists center only, with the 8 neighbors read off the on-disk
sequence at the same step, analogous to cascaded). Each per-baseline plan lists
its chosen train/test set in Section 5.

NVS protocol (identical for every baseline and mm3DGS):
- Train on frame indices `{0,1,2,3, 5,6,7,8}` of the 9-frame sequence.
- Test on frame index `4` (the middle frame — bracketed by train frames).
- No extrapolation to trajectory edges. No split bending.

---

## Shared infrastructure (factor once)

These pieces are needed identically for all three baselines. Build them once
under `baselines/common/` when execution begins.

### 1. Split generator — `baselines/common/nvs_split.py`

Given a scene name, returns:

```
{
  "cascaded": {
    "train_files": [<8 paths to data/<scene>/radar/cascaded_frame_*.npy>],
    "train_configs": [<8 paths to data/<scene>/configs/cascaded_frame_*_aligned.json>],
    "test_file":    "<path to middle cascaded_frame_*.npy>",
    "test_config":  "<path to middle cascaded_frame_*_aligned.json>",
  },
  "single_chip": {
    "train_files": [<8 paths to data/<scene>/radar/single_chip_frame_*.npy>],
    "train_configs": [<8 paths to data/<scene>/configs/single_chip_frame_*_aligned.json>],
    "test_file":    "<path to middle single_chip_frame_*.npy>",
    "test_config":  "<path to middle single_chip_frame_*_aligned.json>",
  }
}
```

Consumers: all three baselines. The split is enumerated by sorting
on-disk frame numbers and taking indices `{0,1,2,3,5,6,7,8}` for train
and `4` for test. One module, not three copies.

### 2. Data adapter primitives — `baselines/common/adapters.py`

Stateless functions, all three baselines use a subset:

- `load_cascaded_adc(scene, frame_idx) -> complex128 ndarray (16, 16, 12, 256)`
- `load_single_chip_adc(scene, frame_idx) -> complex128 ndarray (128, 4, 3, 128)`
- `load_config(path) -> dict` — returns aligned sensor config (TX/RX poses, FMCW params)
- `pose_from_config(config) -> (T_world_from_sensor: 4x4, R: 3x3, t: 3)` — unit: meters; right-handed
- `adc_to_polar_ra(adc, config) -> float32 (H_az, W_range)` — wraps `mmir.data.ra_utils.adc_to_ra_image_numpy`
- `adc_to_cart_ra(adc, config) -> float32 (H, W)` — wraps polar→Cartesian resample
- `fov_wedge_mask(cart_ra_shape, az_range_deg=(-21, 69)) -> bool` — the 90° forward wedge
- `range_crop(x, bins=(15, 110))` — applied uniformly before every metric call

Motivating the shared module:
- Pose convention (meters, world-from-sensor, right-handed, Z-up) is identical
  across baselines; divergent conversions are a silent-bug risk.
- ADC→RA conversion must be bit-identical to what mm3DGS uses or the metrics
  are not comparable — mandated by the "no new metric code" rule above.
- The 90° FoV wedge applies to RadarSplat and RadarFields; writing it twice
  invites drift.

### 3. Metric harness — `baselines/common/eval.py`

Wraps `mmir.evaluation.utils.metrics.compute_cart_ra_metrics` with the exact
bin crop and wedge crop the per-PLAN Section 6 specifies. Emits per-scene and
7-scene-averaged JSON in the same schema the mm3DGS evaluation produces, so
we can drop rows straight into the existing LaTeX table generator at
[mmir/evaluation/](../mmir/evaluation/).

Signature:

```
run_eval(baseline_name: str, scene: str,
         rendered_ra_cart: np.ndarray,
         gt_ra_cart: np.ndarray) -> dict
```

Returns `{"ra_corr": float, "range_profile_corr": float, "n_bins": int, ...}`.

### 4. Scene iterator — `baselines/common/scenes.py`

Single source of truth for the 7-scene list and excluded scenes. One import,
no per-baseline hard-coded lists.

### 5. Result schema

Each baseline writes per-scene results to
`baselines/<name>/results/<scene>/metrics.json` with keys:

```
{
  "baseline": "radarsplat" | "radarfields" | "dart",
  "scene": "seq_X_frame_Y",
  "ra_corr": float,
  "range_profile_corr": float,
  "test_frame": int,
  "train_frames": [int x 8],
  "wall_time_seconds": float,
  "peak_gpu_mem_mib": float,
  "deviations_from_reference": [str, ...],
}
```

A cross-baseline aggregator reads all three baselines' `metrics.json` files
and emits a single markdown table and LaTeX row.

---

## GPU safety (applies before any execution)

**Non-negotiable.** Every per-PLAN Section 11 repeats this; it is the first step
of any future execution plan.

1. Run `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv`.
2. If ANY process is active on ANY GPU, ABORT. Do not share a GPU. Partial
   occupancy does NOT count as free.
3. The user re-runs when the machine is clear.

This applies to short sanity/smoke runs too, not just full training.

---

## Environment isolation

One conda env per baseline. PyTorch and JAX versions are incompatible across
baselines and must not share an env.

| baseline | env name | framework | key pin source |
| --- | --- | --- | --- |
| RadarSplat | `radarsplat` | PyTorch + custom CUDA (gsplat fork) | upstream `environment.yml` |
| RadarFields | `radarfields` | PyTorch (Princeton pins) | upstream `environment.yml` |
| DART | `dart` | JAX + CUDA-toolkit | upstream `pyproject.toml` |

Each per-PLAN Section 7 has the concrete pinned versions.

---

## Compute budget

Cap: 24h wall-time per scene on ONE RTX 4090. Over 7 scenes serial per baseline:
~7 days each. Over all three baselines: ~21 days serial worst-case, but the
sub-agent estimates show RadarSplat (~2h total, 7 scenes) and RadarFields
(~6min total, 7 scenes) are far under budget; only DART meaningfully occupies
the cap. Running baselines on separate days with the GPU-safety rule in place
is the default schedule — do not interleave on the same GPU.

---

## What is NOT shared

Explicitly do NOT factor across baselines:

- Baseline-specific data formats (TUM pose files for RadarSplat, `data.h5`
  for DART, `configs/radarfields.ini` for RadarFields) — these are upstream
  formats and belong in each baseline's adapter, not a shared module.
- Training loops — each baseline runs its own upstream training script with
  minimal config edits. No wrapping in a shared trainer.
- Upstream repos — clone each into `baselines/<name>/upstream/` as a submodule
  or pinned clone. Do not vendor into a shared location.

---

## Summary of per-baseline risks (from sub-agent reports)

| baseline | biggest open question | 8/1 split viability | compute risk |
| --- | --- | --- | --- |
| RadarSplat | TUM pose frame convention (`w2r_mats` naming vs. values) + circular-padding leak on 90° wedge | Marginal — upstream trains on 40-frame windows; 8 frames is a local overfit and a documented weakness | None (~10 min/scene) |
| RadarFields | Azimuth-span hard-coded at `sampler.py:31-33` (360°/num_az); whether other paths silently assume 360° too | Functionally runs at `bs=8`; recommend disabling `--refine_poses` to remove a confound | None (~1 min/scene) |
| DART | Ego-speed distribution — if median rover speed is below ~0.3 m/s, Doppler integration weight `psi_min/pi/s` collapses and there is no training signal | Conditional — single-chip (128 chirps, IWR1443) chosen as closer to DART's 256-chirp design point; viability hinges on ego-speed check | Fits 24h/scene under scaled iterations |

Each risk is documented in its PLAN Section 9 with a "needs verification —
check <path>" pointer rather than a guess.

---

## Next steps (not to be done now)

When GPU becomes clear and the user authorizes execution:

1. `nvidia-smi` check per Section 11 of the relevant PLAN.
2. Build `baselines/common/` (5 modules listed above).
3. Clone upstream into `baselines/<name>/upstream/` at the pinned commit.
4. Run the smoke test that each PLAN proposes (one scene, short iterations).
5. Full 7-scene run per baseline.
6. Aggregate into the shared results schema and regenerate the LaTeX table.
