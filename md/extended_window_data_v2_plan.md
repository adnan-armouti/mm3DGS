# Implementation plan — `data_v2` with per-scene extended training windows

Date: 2026-04-22
Status: **PROPOSAL — awaiting approval**. No code changes until approved.
Owner: Adnan Armouti

## 1. Goal

Generate a new data tree at [/home/adnan/Desktop/mm3DGS/data_v2](/home/adnan/Desktop/mm3DGS/data_v2) that contains **per-scene asymmetric radar frame windows** (instead of the current 9-frame fixed-symmetric windows in `data/`). Then run the v5/M1 training baseline on these larger windows (first chirp only) to see if simply giving the model more training frames — where those frames are *visually relevant* to the test-frame structure (as manually-annotated from the Q1 videos) — materially improves test cc.

**No model-side or loss-side changes. No Doppler. Same v5 `mse_raw` on FFT-RA.** We change **only the training-frame selection**, and we measure whether that alone closes the gap to (or past) the naive-neighbour-average baseline and whether it separates the v5 renderer from the trivial predictor on the benchmark.

## 2. Per-scene frame windows (target)

| Scene | test F | offset lo | offset hi | train frames (count) | symmetric window size (for preproc) |
|---|---:|---:|---:|---:|---:|
| seq_0_frame_135 | 135 | −21 | +18 | 39 | 43 (half=21) |
| seq_1_frame_185 | 185 | −25 | +25 | 50 | 51 (half=25) |
| seq_1_frame_277 | 277 | −25 | +30 | 55 | 61 (half=30) |
| seq_1_frame_438 | 438 | −35 | +30 | 65 | 71 (half=35) |
| seq_2_frame_160 | 160 | −15 | +15 | 30 | 31 (half=15) |
| seq_2_frame_300 | 300 | −19 | +24 | 43 | 49 (half=24) |

**Window asymmetry handling**: the existing `mmir/preprocessing/radar_utils.py::generate_adc_window` requires `num_radar_frames` to be odd and symmetric. Rather than extend the preproc API, we will **preprocess a symmetric max-window** per scene (the larger of |lo| and |hi|, doubled+1), then select the asymmetric subset at **training time** via the `--train_frames` CLI list. Costs: a small amount of unused .npy/.json on disk (≤5 extra files per scene). Gain: zero preproc-code changes for asymmetry.

## 3. Directory strategy

- Untouched: [/home/adnan/Desktop/mm3DGS/data/](/home/adnan/Desktop/mm3DGS/data/), [/home/adnan/Desktop/mm3DGS/mmir/preprocessing/](/home/adnan/Desktop/mm3DGS/mmir/preprocessing/).
- New data tree: [/home/adnan/Desktop/mm3DGS/data_v2/](/home/adnan/Desktop/mm3DGS/data_v2/) — mirror layout of `data/`:
  - `data_v2/<scene>/{scene,configs,radar,lidar}/…`
  - `data_v2/alignment_data/<scene>/cascade/…`
- New preprocessing tree (copied + locally edited): [/home/adnan/Desktop/mm3DGS/mm25DGS_v6/preprocessing/](/home/adnan/Desktop/mm3DGS/mm25DGS_v6/preprocessing/).
  - Already present (inherited from prior work): `alignment/` subdir (pass-2 edits). Leave intact.
  - Needs to be copied from `mmir/preprocessing/` → `mm25DGS_v6/preprocessing/`:
    - `preproc.py` (orchestrator)
    - `radar_utils.py` (ADC window)
    - `config_utils.py` (config window + adjustments)
    - `lidar_utils.py` (fused pcl + per-frame lidar)
    - `mesh_utils.py` (Poisson reconstruction)
    - `decimate_mesh.py` (optional mesh decimation)
    - `io_paths.py` (base_dir helpers — will be rerouted to `data_v2`)
    - `utils.py` (pose / transform helpers)
    - `viz_utils.py` (safe to copy but unused in our path)
    - `ColoRadar_tools/` (devkit wrappers)
    - `calib/` (sensor calibration files)
- **Do not modify** anything under `mmir/preprocessing/`. All edits happen in the copies under `mm25DGS_v6/preprocessing/`.

## 4. Pipeline (step-by-step)

Concretely, for each scene:

**Stage 0 — scaffolding (one-time, done once for all 6 scenes before any per-scene work)**:
1. Create `data_v2/` and `mm25DGS_v6/preprocessing/` skeleton.
2. Copy the non-alignment files from `mmir/preprocessing/` → `mm25DGS_v6/preprocessing/`. Verify `mm25DGS_v6/preprocessing/alignment/` (already present) is preserved.
3. Apply the minimal edits listed in §5 to the COPIED files.

**Stage 1 — preprocessing (per scene; generates `scene/`, `configs/`, `radar/`, `lidar/`)**:
4. Run `python -m mm25DGS_v6.preprocessing.preproc all \
    --seq <S> --frame <F> --num-radar-frames <N_sym> \
    --dataset-dir /home/adnan/Documents/Data/coloRadar/raw/kitti/2_28_2021_outdoors_run \
    --calib-path mm25DGS_v6/preprocessing/calib \
    --out-root data_v2 \
    --cascade --num-lidar-frames <N_lidar> \
    --buffer-distance 1.0 --normals-radius 0.1 --remove-behind-radar --verbose`

   Per-scene params:
   | Scene | --num-radar-frames (odd symmetric) | --num-lidar-frames |
   |---|---:|---:|
   | seq_0_frame_135 | 43 | 50 (covers ±21) |
   | seq_1_frame_185 | 51 | 52 (covers ±25) |
   | seq_1_frame_277 | 61 | 62 (covers ±30) |
   | seq_1_frame_438 | 71 | 72 (covers ±35) |
   | seq_2_frame_160 | 31 | 50 (default OK, covers ±15) |
   | seq_2_frame_300 | 49 | 50 (covers ±24) |

   `--num-lidar-frames` is the LiDAR-fusion radius that drives scene-pcl and mesh construction. It must cover the full cascade window, or the rasterizer will visibility-mask out geometry the radar at extreme-offset poses could see. Set to the smallest even/default that covers max(|lo|, |hi|).

   Outputs (per scene):
   - `data_v2/<scene>/scene/pcl.npy`, `scene/mesh.ply`
   - `data_v2/<scene>/configs/cascaded_frame_<F±k>.json` for k ∈ [−half, +half]
   - `data_v2/<scene>/radar/cascaded_frame_<F±k>.npy`
   - `data_v2/<scene>/lidar/lidar_frame_<L>.npy` (per-cascade matched LiDAR)

5. **Do not generate single-chip ADC/configs** (`--single-chip` flag omitted). We only use cascaded.

**Stage 2 — alignment pass 1 + pass 2 (per scene; generates `alignment_data/<scene>/cascade/`)**:
6. Run `python -m mm25DGS_v6.preprocessing.alignment.run_alignment \
    --scene <scene> --data-root data_v2` (need to make `--data-root` overridable — see §5).

   Outputs per scene (same file names as current):
   - `cascaded_frame_<F±k>_aligned.json` (pass 1)
   - `cascaded_frame_<F±k>_aligned_pass2.json` (pass 2)
   - `pass2_summary.json`, `pass2_triage.json`, etc.

7. **Pass 3 is EXPLICITLY SKIPPED.** `run_alignment.py` has `--skip-pass-3` behaviour by default (pass 3 is opt-in only); do NOT pass `--run-pass-3`.

**Stage 3 — training (per scene; v5 `mse_raw`, first chirp only, 500 iters, target_n=20000)**:
8. For each scene, call `train_frame_nvs.py` with:
   - `--scene <scene>`
   - `--test_frame <F>`
   - `--train_frames "<comma-sep list>"` — *asymmetric subset of the symmetric preproc window, per §2*
   - `--train_loops 0` (first chirp only)
   - `--data_root data_v2`
   - `--iters 500 --loss_type mse_raw --target_n 20000`
   - `--v6_milestone M1` (plain v5 loss)
   - `--save_ra_pngs` (capture best-iter RA maps + cc_history)

Concrete train-frames lists:
| Scene | train_frames |
|---|---|
| seq_0_frame_135 | 114..134, 136..153 (F=135 excluded) |
| seq_1_frame_185 | 160..184, 186..210 |
| seq_1_frame_277 | 252..276, 278..307 |
| seq_1_frame_438 | 403..437, 439..468 |
| seq_2_frame_160 | 145..159, 161..175 |
| seq_2_frame_300 | 281..299, 301..324 |

## 5. Code ports + edits (minimal and deterministic)

### 5.1 Pure copy (no edits required)

No-op copy into `mm25DGS_v6/preprocessing/`:
- `calib/` (entire directory)
- `ColoRadar_tools/` (entire directory)
- `mesh_utils.py`
- `decimate_mesh.py`
- `utils.py`
- `viz_utils.py`

### 5.2 Copy + minimal edit

The following copies need small edits. Each edit is scoped to the copied file in `mm25DGS_v6/preprocessing/` only.

**`mm25DGS_v6/preprocessing/preproc.py`** (copy of `mmir/preprocessing/preproc.py`):
- **No functional edits needed** — the existing `--out-root` CLI arg already controls where outputs land. We invoke with `--out-root data_v2`. The only change is the module path used to invoke it (`python -m mm25DGS_v6.preprocessing.preproc` instead of `mmir.preprocessing`).
- **One optional clean-up** (not strictly required): the top imports are `from radar_utils import ...` etc. (no package prefix). Either (a) leave as-is and ensure we run from the project root (current behaviour), or (b) convert to `from .radar_utils import ...` so the module is `-m`-runnable. I'll pick (b) — converting the top imports to relative — so we don't inherit the execution-directory fragility. This is a 6-line diff.

**`mm25DGS_v6/preprocessing/io_paths.py`** (copy of `mmir/preprocessing/io_paths.py`):
- Contains `get_base_dir()` and `find_lowest_cascaded_config()`. `get_base_dir(seq_idx, center_frame_idx, out_root)` accepts `out_root` as a positional arg, so already supports `data_v2` routing via CLI. **No edits needed**.

**`mm25DGS_v6/preprocessing/radar_utils.py`** (copy of `mmir/preprocessing/radar_utils.py`):
- The existing `generate_adc_window` accepts `num_radar_frames` as CLI-plumbed arg. Enforces odd and symmetric, which fits our symmetric-max strategy (§2). **No edits needed**.

**`mm25DGS_v6/preprocessing/config_utils.py`** (copy of `mmir/preprocessing/config_utils.py`):
- Similar structure to `radar_utils.py`. Should parameterise correctly via `num_radar_frames`. **Expected no edits needed**; confirm empirically when running.

**`mm25DGS_v6/preprocessing/lidar_utils.py`** (copy of `mmir/preprocessing/lidar_utils.py`):

This is the file with the memory concerns the user flagged. Expected edits:

- **Line 254 `vox_size=0.0005`** parameter: unused in the current code path but documents intent. Leave as-is.
- **Lines 303–321 (`accum = []` + loop that appends cropped clouds then `np.vstack`)**: for larger cascade windows, the LiDAR fusion spans more frames. With `num_lidar_frames=72` for seq_1_frame_438, we accumulate ~72 clouds (each ~100k–1M points after cropping), totalling 10s–100s of millions of points before `np.vstack` + `np.unique`. Memory-wise this is the single biggest risk.

  **Proposed edits** to `generate_fused_lidar_pointcloud`:
  1. **Chunked vstack + early dedup**. Instead of one final `np.vstack`, accumulate in chunks of (e.g.) 10 frames, run `np.unique` after each chunk, stream to a pre-allocated numpy array on disk (`np.lib.format.open_memmap`) or keep in-RAM but periodically dedup. Net effect: peak RAM drops from "sum of all chunks" to "~2× largest chunk".
  2. **Skip the final `np.unique(..., axis=0)` on the full fused cloud** (line 337). It's O(N log N) and materialises a sorted copy of the full array. Replace with FAISS-based hashing or just skip (we later do a more aggressive spatial dedup via the mesh Poisson step).
  3. **cKDTree fallback with FAISS** (lines 350–367): the sensor-position-in-fused check uses `scipy.spatial.cKDTree`. For large N this is a CPU tree-build and k-NN, slow but tolerable. Leave as-is unless it becomes the wall-clock bottleneck.

  If (1) + (2) aren't enough to keep the 72-frame case in RAM budget, fall back to **reducing `num_lidar_frames` for seq_1_frame_438 specifically** (e.g., to 50 — same as other scenes — and accept the loss of scene coverage for the extreme-offset frames). Preferred order: try unchanged → try (1)+(2) → fall back.

- **`estimate_normals_gpu`** (line 423): uses FAISS GPU. For very large fused clouds (>10M points), the index build can OOM one GPU. Mitigation: **drop the z-range below ground level via the existing `remove_points_behind_radar`**, and keep `faiss_devices=[0]` single-GPU only (no multi-GPU sharding beyond current code) — because the multi-GPU sharding in `faiss.index_cpu_to_gpu_multiple` has historically failed on this workstation (already wrapped in try/except; the fallback `index_cpu_to_gpu` will fire). No code change needed.

- **`generate_lidar_window`** (line 89): saves `num_radar_frames` separate LiDAR files. For N=71 (seq_1_frame_438) that's 71 files, each ≤~5 MB → ~350 MB. Fine.

Summary: **minimum edit = zero code changes for correctness**; **recommended edit = two lines in `generate_fused_lidar_pointcloud` (chunked vstack/dedup)** for seq_1_frame_438's 72-LiDAR-frame fusion. Implementation cost for the recommended edit: ~20 LOC of straightforward numpy.

**`mm25DGS_v6/preprocessing/alignment/run_alignment.py`** (ALREADY COPIED — current v6 version has pass-2 + pass-3 logic):
- Must support `--data-root data_v2` to point at the new data tree. Existing code likely hardcodes `data/` somewhere (I'll grep the 3 alignment files at implementation time); if it does, minimal edit is to add a CLI flag `--data-root` that is threaded through `cascaded_alignment.py`, `pass2/trajectory_fit.py`, `pass2/run_pass2.py`. Estimated: ~30 LOC across 3 files.

**`mm25DGS_v6/preprocessing/alignment/cascaded_lidar.py` / `cascaded_lidar_gpu.py`**:
- Same: likely reads from `data/<scene>/...`. If a default data-root is hardcoded, swap to CLI-driven. Estimated ~10 LOC.

### 5.3 Trainer edit

**`mm25DGS_v6/train_frame_nvs.py`**:
- Add `--data_root` CLI flag (default `data/`). Thread through `_aligned_config_path()`, `build_frame_level_dataset()`, etc. Existing `data_root` parameter is already plumbed through most of the function — just needs the CLI wire-up. Estimated: ~5 LOC.

### 5.4 Total edit budget

- Port + pure copies: ~0 LOC (copy-only).
- `preproc.py` relative imports: ~6 LOC.
- `lidar_utils.py` chunked fusion: ~20 LOC.
- `alignment/*` data-root plumbing: ~40 LOC across 3-4 files.
- `train_frame_nvs.py` data-root CLI: ~5 LOC.

**Total: ~75 LOC of edits.** No functional-logic changes — only data-root + memory-chunking plumbing.

## 6. Memory / wall-clock budget

Per scene, for each of the 6 target scenes:

| Step | Expected peak RAM | Expected GPU VRAM | Expected wallclock |
|---|---:|---:|---:|
| Stage 1 — scene/pcl + mesh (Poisson d=8) | 8–24 GB (seq_1_frame_438 worst case) | 4–8 GB (FAISS normals) | 3–8 min |
| Stage 1 — cascaded configs + ADCs (N=71 max) | 1 GB | 0 | < 1 min |
| Stage 1 — per-frame LiDAR files | 1 GB | 0 | < 1 min |
| Stage 2 — alignment pass 1 (per-frame, serial) | 4 GB | 6 GB (CUDA v5 renderer) | ~15 s/frame × N = 2–18 min |
| Stage 2 — alignment pass 2 (trajectory + retry) | 4 GB | 6 GB | 1–3 min |
| Stage 3 — training (500 iters, 30–65 train samples/iter) | 16 GB | 8–20 GB (higher with more samples/iter) | 2–6 min (proportional to train-frames count) |

**Total per scene**: 10–35 min of wallclock.
**6-scene total**: 60–200 min = **~1–3.5 hours** depending on scale (seq_1_frame_438 is the tentpole).

With 2-GPU parallelism: Stage-1 + Stage-2 can run two scenes at once. Stage 3 benchmark takes ~2–6 min each, trivially parallelised across 2 GPUs. So total wallclock ~ **45–90 min** realistic with sensible GPU assignment.

Memory headroom check: the workstation has ~128 GB RAM (per CLAUDE.md). Even the seq_1_frame_438 fused-LiDAR peak of ~30 GB fits comfortably; the chunked fusion edit is a safety margin, not a correctness requirement.

## 7. Validation gates — do not proceed if any fails

After Stage 1 (scene/pcl + ADCs/configs generated):
- **G1.1**: `data_v2/<scene>/scene/pcl.npy` exists, N_points ≥ 1M.
- **G1.2**: `data_v2/<scene>/scene/mesh.ply` exists, n_vertices ≥ 10k.
- **G1.3**: `data_v2/<scene>/radar/cascaded_frame_*.npy` count = expected symmetric window size (43/51/61/71/31/49).
- **G1.4**: bit-identity smoke test — pick one frame (say F = scene center) and assert that `data_v2/<scene>/radar/cascaded_frame_<F>.npy` is **byte-identical** to `data/<scene>/radar/cascaded_frame_<F>.npy`. Same for `configs/cascaded_frame_<F>.json`. This guarantees the preproc port didn't silently change a calibration or axis convention.

After Stage 2 (alignment):
- **G2.1**: `data_v2/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json` exists for all N frames.
- **G2.2**: `pass2_summary.json` reports 0 or ≤1 frames flagged as gate-failures (matches current behaviour on `data/`).

After Stage 3 (training) — for the 6-scene batch:
- **G3.1**: The F (middle-test) v5 result on `data_v2` matches the existing `data/` v5 result within MC noise (±0.03 per scene). This is the master bit-identity check: if the middle-frame training-frame window on `data_v2` (selecting only F±4) produces the same final_test_cc as the original `data/` v5 run, we know the preproc port is faithful.

If any gate fails, stop and investigate before running the full extended-window training.

## 8. Expected outcomes and how to interpret them

We will report the following 3 values per scene:

1. **`cc_data_v1`** — existing v5 result on `data/` (F±4 training, baseline from prior runs).
2. **`cc_data_v2_F±4`** — v5 on `data_v2` restricted to F±4 training (bit-identity smoke, gate G3.1 above).
3. **`cc_data_v2_full`** — v5 on `data_v2` with the full asymmetric extended window.
4. **`cc_naive_avg`** — naive `½·(GT[F-1]+GT[F+1])` predictor (unchanged from prior).

Interpretations:

- **(3) ≫ (1), (2)**: extending the training window helps. The benchmark was not saturated on seq-consistency grounds alone; it was train-data-limited. We would then expand the analysis: does the lift track with the decorrelation slope (Q1)? Does it beat naive-avg on all 7 scenes, not just seq_1_frame_438?
- **(3) ≈ (1), (2)**: extending doesn't help. The benchmark is saturated on scene-consistency / model-capacity grounds; more training data doesn't extract more learnable signal at this grid resolution / rasterizer primitive. This validates pivoting to the Doppler axis (Q4 plan).
- **(3) < (1), (2)**: extending HURTS. Most likely cause: extra frames at large offsets introduce pose noise that outpaces the signal, matching the decorrelation-plot predictions on seq_1_frame_438. We would then prune down to scene-adapted windows (the decorrelation-plot k-threshold).

The result shape alone (of pattern 1–3) is the answer to the user's motivating question. We don't need anything downstream before interpreting.

## 9. Rollout order

1. **Approval.** Halt here until the user OKs this plan.
2. **Stage 0**: create `data_v2/` + port preprocessing dir. Diff-check copied files to their `mmir/preprocessing/` originals to confirm only the documented edits landed.
3. **Stage 1 smoke**: run `seq_2_frame_160` (smallest window, N=31, <5 min) end-to-end through stages 1–3 at F±4. If gate G1.4 and G3.1 both pass, we know the port works.
4. **Stage 1 full**: run stages 1 + 2 on all 6 scenes. Allow ~1–2 hours wallclock.
5. **Stage 3 full**: run stage 3 (extended-window training) on all 6 scenes. Allow ~30–60 min wallclock.
6. **Report**: 6-scene table with cc_data_v1 / cc_data_v2_F±4 / cc_data_v2_full / cc_naive_avg per scene + mean.
7. **Decision**: go/no-go on pushing further vs reverting to the Doppler plan.

## 10. What this plan does NOT do

- **No** model-side changes (no loss variant, no rasterizer change, no doppler).
- **No** modifications to `/home/adnan/Desktop/mm3DGS/mmir/preprocessing/`.
- **No** pass-3 (per-chirp) alignment. We stay on first-chirp only.
- **No** `--single-chip` preproc — only cascaded.
- **No** retraining of UB/HO_128/HO_9 variants — just the F±|asymmetric| extension.
- **No** sweeping over multiple `--num-lidar-frames` values — we pick one per scene and stick.
- **No** re-running naive-avg baselines — those numbers are already captured and do not depend on training data.

## 11. Known risks

- **Pose-alignment quality at large offsets.** The pass-1/pass-2 alignment was empirically robust on the current F±4 window (~0.5 s of motion). Extending to F±35 means aligning poses over ~7 s of motion — the trajectory-fit stage (Stage A of pass 2) should still be fine (LOWESS + Huber over the full cascade window is designed exactly for this), but individual frame poses at extreme offsets may have worse pass-1 cc. Gate G2.2 will catch egregious cases.
- **Memory during fused-LiDAR Poisson reconstruction.** If the Poisson step OOMs on a very large fused cloud, the mesh_utils script already supports depth-truncation (`--depth 8`) and point-weight flooring; fall back to depth=7 if needed. Not a correctness issue.
- **Scene-pcl coverage gaps.** If the trajectory turns sharply within a large window, the LiDAR-fusion FOV might not cover the full radar-FOV at extreme offsets, and the visibility-mask will drop those points. This would manifest as rendering "holes" in extreme-offset training frames — worth spot-checking in the `cc_history.png` of Stage 3 runs. Not a show-stopper.

---

**Awaiting approval. No files will be created or edited until approval lands.**
