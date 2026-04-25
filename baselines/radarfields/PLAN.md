# RadarFields Baseline — Minimal-Changes Plan

Target: defensible benchmark number for the RadarFields neural-field radar method on our 7 scenes, under the same NVS protocol and metric harness used by mm3DGS. This is NOT "strongest possible RadarFields" — every deviation from the reference is documented as a limitation.

Upstream: `princeton-computational-imaging/RadarFields` (SIGGRAPH 2024). All citations use `owner/repo:path:LINE`.

---

## 1. Repo summary

RadarFields represents a scene as a learned 3D neural field `f(x, d) -> (alpha, rd)` where `alpha` is per-point occupancy in [0,1] and `rd` is a view-dependent reflectance-like scalar. A forward FMCW model integrates per-ray bundles through an antenna gain LUT, multiplies `rcs = alpha * rd`, and maps to FFT intensity via `log10(rcs + offset) * scaler` (`princeton-computational-imaging/RadarFields:radarfields/radar.py:6`). Training minimizes FFT-intensity reconstruction plus occupancy regularizers against the vendor-provided *range-azimuth FFT image* (a 2D polar heatmap; NOT raw ADC). Pose refinement is optional.

- Entry point: `princeton-computational-imaging/RadarFields:main.py:37-71` (train → test).
- Model: `princeton-computational-imaging/RadarFields:radarfields/nn/models.py:46-165` (`RadarField`), HashGrid XYZ encoding + spherical-harmonics direction encoding + tcnn MLPs.
- Forward: `princeton-computational-imaging/RadarFields:radarfields/train.py:515-558` (`predict_waveform`).
- Loss: `princeton-computational-imaging/RadarFields:radarfields/train.py:427-513` (`compute_loss`): FFT L1/MSE, occupancy KL vs. thresholded FFT, bimodal penalty, grounding penalty, above-mounting-height penalty, optional pose regs.
- Training budget: `iters=800` by default (`princeton-computational-imaging/RadarFields:parse.py:59`, `configs/radarfields.ini:11`), batch size 10 frames, `num_rays_radar=100` azimuths per frame, `num_fov_samples=10` super-samples per ray, `num_range_samples=1125`.

Input tensor: per-sample 2D polar FFT image `[num_azimuths=400, num_range_bins=7536]` of float intensities already in 2·dB + noise-floor space (`radarfields/dataset.py:77-91`, `utils/data.py:9-21`). Pose is a 4x4 `radar2world` matrix in SE(3), meters (`radarfields/dataset.py:72-76`). Radar intrinsics: `opening_h=1.8°`, `opening_v=40°`, `bin_size_radar=0.044 m`, `num_azimuths_radar=400` (`parse.py:24-32`).

Output tensor at test time: per frame, a reconstructed polar FFT `[400, R]` where `R = max_range_bin - min_range_bin + 1` (`dataset.py:101-105, 143, 156-160`). Visualized as BEV via `render_outputs_supersampled` (`train.py:284-304`).

## 2. Upstream data format

Dataloader: `princeton-computational-imaging/RadarFields:radarfields/dataset.py:16-208`.

Expected on-disk layout, rooted at `<repo>/data/<seq>/`:

```
<repo>/data/<seq>/
    radar/                                  # FFT images (PNG, one per frame)
        <timestamp>.png                     # shape after read: [400, 7536]; first 11 rows stripped as metadata (utils/data.py:14)
    azimuth.csv                             # antenna azimuth gain profile LUT (offset_deg, dBic)
    elevation.csv                           # antenna elevation gain profile LUT

<repo>/preprocess_results/
    <preprocess_file>.json                  # consumed by RadarDataset (dataset.py:58-69)
    thresholded_fft/<seq>/<timestamp>.npy   # optional, if --train_thresholded (dataset.py:82-86)
    occupancy_component/<preprocess_stem>/<timestamp>.npy   # if --reg_occ (dataset.py:93-99)
```

`preprocess_file` JSON schema (inferred from `dataset.py:60-99`):
- `timestamps_radar`: list of frame filenames (strings, used as keys on disk).
- `radar2worlds`: list of 4x4 float matrices (SE(3), right-handed, meters, X-forward by convention of `get_radar_rays`; see `radarfields/sampler.py:66-71` where `[1,0,0]` is the forward unit vector in radar frame).
- `offsets`, `scalers`: XYZ normalization offsets / scales so normalized XYZ fits a unit box (`dataset.py:67-69`, applied `train.py:527-532`).
- `train_indices`, `test_indices`: integer lists into the `timestamps_radar` array (`dataset.py:64`).

Radar metadata (defaults, see `parse.py:24-32`):
- Center freq: not consumed directly by the model; FFT is already preprocessed.
- Azimuth resolution: `360 / 400 = 0.9°/bin` (spinning radar).
- Max range: `7536 * 0.044 m ≈ 331.6 m`.
- Opening angle (effective beam): 1.8° azimuth, 40° elevation.

The method is written for a 360° spinning radar; `azim_samples` are drawn uniformly from `[0, 400)` (`radarfields/sampler.py:5-12`), and `get_radar_rays` maps bin index → yaw degrees as `azim_degrees = (360/num_azimuths) * azim_samples` (`radarfields/sampler.py:31-33`). This is the primary mismatch with our forward-facing cascade.

## 3. Adapter — minimum transformation: mm3DGS → RadarFields input

We transform our data into the format RadarFields expects. We do NOT modify its input pipeline.

- **ADC → RA polar image** (cascade only, single-chip is not used for RF training).
    - Source: `data/seq_X_frame_Y/radar/cascaded_frame_{131..139}.npy` — complex128 `(chirps=16, RX=16, TX=12, ADC=256)`.
    - Pipeline (reuse existing): average-over-chirps → `mmir.data.ra_utils.adc_to_ra_image_numpy` (see `/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py:174-200`) which does `txrx_to_virtual_array_numpy` (`ra_utils.py:107-140`) then `virtual_array_to_ra_polar_numpy` (`ra_utils.py:143-171`). Result: `(127 azimuth bins, 256 range bins)` magnitude float32.
    - Dest: `<repo>/data/<seq>/radar/<timestamp>.png` — 16-bit PNG or grayscale float-quantized PNG readable by `utils/data.py:9-21`.
    - NOTE: upstream pre-strips the top 11 rows as metadata; write our PNGs with 11 blank rows prepended so the strip is a no-op.

- **Intensity normalization**. The upstream dataloader assumes FFT in ~[0,1] float (stored via `transforms.ToTensor()`, `utils/data.py:19-21`). Min-max normalize each RA magnitude to [0,1] per-frame. Document this as a deviation from upstream's "2·dB + noise floor" convention since we do not have their exact preprocessing.

- **Radar intrinsics override** (`parse.py:24-32`). Pass through `--intrinsics_radar`:
    - `num_azimuths_radar`: 127 (our RA image azimuth count) — override the 400 default.
    - `num_range_bins`: 256 (our ADC count) — override 7536.
    - `bin_size_radar`: range resolution from `mmir.data.io_utils.compute_range_res_from_cfg` applied to `data/seq_X_frame_Y/configs/cascaded_frame_{Y}_aligned.json` → ~0.117 m/bin for our sensor (see `mmir.evaluation.utils.ra_processing._range_resolution_from_config`, `/home/adnan/Desktop/mm3DGS/mmir/evaluation/utils/ra_processing.py:86-99`).
    - `opening_h`, `opening_v`: 1.8°, 40° — keep upstream defaults (cascade's physical 3-dB beamwidth is comparable enough for a baseline; document as a limitation).

- **Pose-frame conversion**. Source: per-frame `configs/cascaded_frame_{Y}_aligned.json` (pass-2 aligned, matches production training). The `tx_array[0].pos_mm` + `boresight` fields define sensor pose; build a 4x4 `radar2world` matrix by:
    1. Use `tx_array[0].pos_mm` (convert mm → m) as translation.
    2. Use `boresight` as the local +X (forward) axis; construct an orthonormal basis with +Z = world up and +Y = Z × X.
    3. Stash as float32 4x4 under `preprocess["radar2worlds"][i]`.
    - Handedness: RadarFields uses right-handed, forward = +X (`sampler.py:66-71`). Our cascade configs store boresights in the same convention (verified by `mmir/preprocessing/alignment/cascaded_alignment.py` — needs verification — check `/home/adnan/Desktop/mm3DGS/mmir/preprocessing/alignment/`).
    - Units: meters (RadarFields sampler multiplies range-in-meters by direction unit vector, `sampler.py:114`).

- **`preprocess.json` assembly** (new file under `<repo>/preprocess_results/mm3dgs_<scene>.json`):
    - `timestamps_radar`: `["131.png", "132.png", "133.png", "134.png", "135.png", "136.png", "137.png", "138.png", "139.png"]` (names derived from the cascade frame index; center = scene's `frame_Y`).
    - `radar2worlds`: 9 matrices from the step above.
    - `train_indices`: `[0,1,2,3,5,6,7,8]`, `test_indices`: `[4]`.
    - `offsets`, `scalers`: compute to center and rescale the 8 training poses into a unit cube. Use the train-pose translation centroid as `-offsets` and the max abs coordinate (with a 20 m range margin) as `scalers`. Document formula.

- **Antenna LUTs**. RadarFields expects `azimuth.csv`, `elevation.csv` two-column (offset_deg, dBic) (`utils/data.py:40-48`).
    - Source: `assets/antenna_pattern/MMWCAS/tx1_76.npy`, `rx1_76.npy`. These are azimuth-only forward-looking patterns (no elevation cut).
    - Azimuth LUT: project tx*rx combined pattern to azimuth, convert amplitude → dBic `10*log10(|g|)`, export 2-col CSV covering ±45° for safety.
    - Elevation LUT: our cascade has no measured elevation pattern. Insert a flat 0 dBic LUT across ±40° (i.e., all gains equal) and document the limitation.

- **FoV truncation on input**. Our 127-azimuth RA is already forward-facing (≈ ±90° beamformed, energy concentrated in the ~90° forward wedge). We keep the full 127 columns and let the azimuth LUT + training loss handle off-beam bins (they'll be near-zero from the antenna pattern). **Do NOT zero-pad to 400 bins.** (See Section 4 for the upstream consequences.)

- **Range crop on input**. Keep all 256 range bins on input. Restrict sampling via `--min_range_bin 15 --max_range_bin 110` (inclusive, 1-indexed per `dataset.py:156-160`). This aligns with our established evaluation crop "bins 15..110" and also matches upstream's coarse-to-fine behavior. Note: 1-indexed in upstream vs. 0-indexed in our metric code — add a comment in the adapter.

- **Occupancy mask (`--reg_occ`)**. Pre-compute the upstream "occupancy component" per frame via `radarfields.radar.compute_occupancy_component` (`princeton-computational-imaging/RadarFields:radarfields/radar.py:87-107`) on the min-max-normalized RA image. Store under `preprocess_results/occupancy_component/mm3dgs_<scene>/<timestamp>.npy`. Thresholds: use upstream defaults; document as a limitation (not tuned for our data).

- **Thresholded FFT (`--train_thresholded`)**. Pre-compute via `compute_spherical_grid_noise_threshold` (`radar.py:152-165`) on each RA image. Store under `preprocess_results/thresholded_fft/<scene>/<timestamp>.npy`.

## 4. Geometry / chirp reconciliation

The hard mismatch is 360° azimuth → ~90° forward wedge. RadarFields' network/forward has NO fixed 360° geometry in its weights — the spinning assumption lives in the sampler, not the MLP. Crucial upstream lines:

- `princeton-computational-imaging/RadarFields:radarfields/sampler.py:31-33`: azimuth-degrees conversion `azim_degrees = (360.0/num_azimuths) * azim_samples`. **This is the single line that hard-codes 360°.** We override `num_azimuths_radar` to 127 *and* multiply by `90/127` instead of `360/127`.
    - Minimum fix: add one line in the adapter that patches the sampler, or pass a new CLI arg. Concretely: replace the `360.0` constant with `azim_span_deg`, default 360.0, new default for us = 90.0. Single-line edit.
    - Keep `num_azimuths_radar = 127`.

- `princeton-computational-imaging/RadarFields:radarfields/dataset.py:101-105`: at test time, `num_rays_radar` is set to `num_azimuths_radar` so the whole frame is rendered. With our override this renders exactly 127 rays across the 90° wedge, matching GT shape `[127, R]`. No network output width depends on 360° — the forward is pointwise per (x, d). **Good: no fixed-width output layer to resize.**

- `radarfields/sampler.py:66-71`: forward is `[1,0,0]` in radar frame. OK for our cascade (boresight convention matches our aligned configs, needs verification — check `mmir/preprocessing/alignment/cascaded_alignment.py`).

- GT crop in metric harness (Section 6): crop rendered `[127, 256]` to `[:, 15:110]` and the GT identically, so no change needed on the output-shape side.

- The coarse-to-fine hashgrid mask (`--mask`, `train.py:316`, `models.py:9-20`) scales by epoch, not azimuth, so not affected.

Justification: this is the minimum change. Modifying the sampler constant is strictly necessary (otherwise 8 training poses are stretched across 360° of azimuth and the model learns an empty scene 3/4 of the time). Every other geometry is untouched.

## 5. NVS split

Protocol (IDENTICAL to mm3DGS/RadarSplat/DART):
- 9 cascade frames per scene, indices 0..8 → frame numbers `{center-4, ..., center+4}`.
- Train on `{0,1,2,3,5,6,7,8}` (outer 8), test on `{4}` (middle; bracketed by train poses).

| Scene | Center frame | Train frames (8, cascaded) | Test frame (1, cascaded) |
|---|---|---|---|
| seq_0_frame_135 | 135 | 131, 132, 133, 134, 136, 137, 138, 139 | 135 |
| seq_0_frame_390 | 390 | 386, 387, 388, 389, 391, 392, 393, 394 | 390 |
| seq_1_frame_185 | 185 | 181, 182, 183, 184, 186, 187, 188, 189 | 185 |
| seq_1_frame_438 | 438 | 434, 435, 436, 437, 439, 440, 441, 442 | 438 |
| seq_2_frame_105 | 105 | 101, 102, 103, 104, 106, 107, 108, 109 | 105 |
| seq_2_frame_160 | 160 | 156, 157, 158, 159, 161, 162, 163, 164 | 160 |
| seq_2_frame_300 | 300 | 296, 297, 298, 299, 301, 302, 303, 304 | 300 |

Excluded from benchmark (per mm3DGS convention): `seq_1_frame_277`, `seq_0_frame_451`.

One `preprocess_results/mm3dgs_<scene>.json` file per scene. `train_indices=[0,1,2,3,5,6,7,8]`, `test_indices=[4]` in all 7 files.

## 6. Metrics harness

Pipeline at test time:
1. RadarFields test run produces `pred_fft` of shape `[1, 127, 96]` (after our FoV edit and range crop 15..110). Extracted from the trainer's test path, then saved to `<workspace>/imgs/<name>/rf_pred_ra_<scene>.npy`.
2. Adapter post-processing:
    - GT: load `data/seq_X_frame_Y/radar/cascaded_frame_<center>.npy`, convert via `mmir.data.ra_utils.adc_to_ra_image_numpy` (`/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py:174-200`) → `(127, 256)` polar magnitude. Crop `[:, 15:110]`.
    - Rendered: RadarFields already returns `[127, 96]` polar intensity (normalized units). Leave in this space.
    - Convert both to Cartesian via `mmir.data.ra_utils.ra_polar_to_cartesian(ra_polar, range_res)` (`/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py:429-458`) with `range_res` from `mmir.data.io_utils.compute_range_res_from_cfg`.
3. Compute metrics via `mmir.evaluation.utils.metrics.compute_cart_ra_metrics(ra_gt_cart, ra_rend_cart)` (`/home/adnan/Desktop/mm3DGS/mmir/evaluation/utils/metrics.py:87-117`) → dict with `cart_corr`, `mse`, `rmse`, `psnr`, `ssim`. Primary number is `cart_corr`.
4. Range-profile correlation: `sum_azimuth(ra_polar[:, 15:110])` for rendered and GT, then `np.corrcoef(rend, gt)[0,1]`. Reference pattern in `/home/adnan/Desktop/mm3DGS/mmir/evaluation/eval_training_ra_v2.py` (see `compute_cart_ra_metrics` import at line 36 and the structure around lines 89-100).
5. Average `cart_corr` and range-profile correlation over 7 benchmark scenes.

Output directory convention: `/home/adnan/Desktop/mm3DGS/baselines/radarfields/output/<scene>/`
- `metrics.json` — per-scene dict.
- `ra_gt_polar.npy`, `ra_gt_cart.npy`, `ra_rend_polar.npy`, `ra_rend_cart.npy`.
- `rf_workspace/` — upstream workspace directory (checkpoints + logs).

Aggregate runner at `/home/adnan/Desktop/mm3DGS/baselines/radarfields/aggregate_metrics.py` (writes `aggregate.json` with mean/std of cart_corr + range-profile corr across 7 scenes).

## 7. Env setup

Conda env name: `radarfields`. Upstream pins (see `princeton-computational-imaging/RadarFields:environment.yml`):
- Python 3.9.18
- PyTorch 2.0.1 + CUDA 11.7 (`pytorch=2.0.1=py3.9_cuda11.7_cudnn8.5.0_0`, `pytorch-cuda=11.7`)
- torchvision 0.15.2, torchaudio 2.0.2, torchtriton 2.0.0
- tcnn (`tiny-cuda-nn`): `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- configargparse 1.7, numpy 1.24.4, scipy, scikit-image 0.20.0, pyyaml 6.0

Install steps (to run later, NOT now):
```
conda env create -f <clone>/environment.yml   # creates env "radarfields"
conda activate radarfields
# tcnn requires matching CUDA; RTX 4090 is sm_89, tcnn works on 11.7 but verify
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install -e <clone>                        # installs "radarfields" package
```

Risk (Section 9): CUDA 11.7 + RTX 4090. sm_89 was supported starting CUDA 11.8; PyTorch 2.0.1 CUDA 11.7 build does not officially include sm_89 kernels. Two possible fixes — (a) bump PyTorch to 2.0.1+cu118 wheel (minor deviation, documented), (b) rebuild tcnn against cu118. Prefer (a).

Additional pins our adapter introduces:
- `Pillow>=9.5` (already in env) for PNG write.
- No other dependencies beyond stock NumPy/PyTorch.

## 8. Compute estimate

Upstream default: `iters=800`, `bs=10` frames, 8 train frames → 1 batch per epoch → 800 epochs total (`parse.py:59`, `configs/radarfields.ini:11`). Per forward pass with `num_rays_radar=100`, `num_fov_samples=10`, `num_range_samples=1125` → `100 * 10 * 1125 = 1.125e6` XYZ queries per frame, `1.125e7` per batch. With tcnn HashGrid + small MLPs this is ~200-400 ms/iter on a 4090 (rough; needs verification — check `/tmp/rf_train.py` perf). Upper-bound estimate: 400 ms * 800 iters ≈ 320 s per scene ≈ **5-6 min per scene**.

Our configuration override:
- `num_range_samples` reduces to 96 (15..110). `96 / 1125 ≈ 8.5%` of queries → **~30-50 s per scene** at most.
- 7 scenes serial → **~6 minutes total wall time** on one 4090.

Well under the 24 h/scene cap. No knobs to turn down. If iters need to scale up for quality (e.g., 3000 iters matching `iters` cited in related work), we still fit easily (~30 min/scene).

Under-cap risk: we may be undertraining relative to the paper's intended budget (paper's full default spec had `iters=800` but with `num_range_bins=7536` — our 96 range bins is far sparser, which *helps* convergence per-epoch but also dramatically reduces supervision diversity per frame). See Section 9.

## 9. Open questions / risks

Each item is "needs verification — check <path>" (not a guess).

- **(a) Is the single-frame FoV legal?** The model is pointwise so the MLP does not care about wedge vs. full scan, but the coarse-to-fine hashgrid mask schedule, the occupancy grounding penalty (`train.py:473-485`), and the azimuth LUT assumption may all break if the azimuth span passed to the sampler is not 360°. Needs verification — check `princeton-computational-imaging/RadarFields:radarfields/sampler.py:31-33` and `radarfields/train.py:467-485` for any hidden 360° dependency beyond the constant.

- **(b) Is 8 frames viable?** Upstream default is `bs=10` frames per batch (`configs/radarfields.ini:9`). With only 8 train frames, one batch of 10 is not filled; the `SubsetRandomSampler` (`dataset.py:64-65`) will just iterate 8 indices. Drop `bs` to 8 (adapter change). Upstream models appear to be trained on longer driving sequences (tens of frames, see `preprocess_results.json` default, `parse.py:99`); 8 frames is well below that. This is a fundamental regime difference and the most likely source of poor numbers. Flag as limitation.

- **(c) Pose coordinate / handedness.** Upstream assumes `+X = forward` in radar frame (`sampler.py:66-71`) and RH world. Needs verification — check `/home/adnan/Desktop/mm3DGS/mmir/preprocessing/alignment/cascaded_alignment.py` for the convention our aligned configs produce. If ours uses `+Y = forward`, the adapter must permute axes in `radar2worlds` before writing.

- **(d) Polarization / beamformed vs. raw virtual-array input.** Upstream consumes a *beamformed 2D RA FFT image already in intensity space* (PNG, `utils/data.py:9-21`). Our cascade ADC → RA path goes ADC → virtual array → 2D FFT → magnitude. That matches intensity space, BUT:
    - Upstream images appear pre-normalized to a noise floor (`--noise_floor=0.1525`, `parse.py:100`). We use a per-frame min-max. Needs verification — check `princeton-computational-imaging/RadarFields:utils/vis.py::render_FFT_batch` (`train.py:292`) for the exact normalization used at viz time so we match it.
    - Polarization: cascade is V-polarized (see `cascaded_frame_135_aligned.json:polarization: "V"`). Upstream was trained on Navtech spinning radar (also single polarization, different frequency). Not a functional blocker but a domain gap. Note as limitation.

- **(e) Elevation gain LUT.** Our cascade has no measured elevation pattern; we insert a flat LUT (Section 3). The grounding penalty (`train.py:473-485`) uses angular offsets to punish overhead occupancy. With a flat elevation LUT the penalty still works but loses its intended geometry prior. Flag.

- **(f) `preprocess_file` occupancy-component path.** The dataloader expects `preprocess_results/occupancy_component/<preprocess_stem>/<timestamp>.npy` (`dataset.py:94-99`). Ensure our adapter writes filenames without the `.png` suffix (`str(fft_frame).split('.')[0] + '.npy'`).

- **(g) Pose refinement at test.** With `--refine_poses` on and `test_indices=[4]` marked non-trainable, the test pose is *interpolated* from refined neighbors (`train.py:332`, `PoseOptimizer.interp_test_poses`). Small train set (n=8) may produce a poor interpolant for the held-out middle frame. Option: disable `--refine_poses` for this baseline to avoid a confound, and note as deviation.

- **(h) Metric-space mismatch.** Upstream metrics are on FFT-intensity. Our `compute_cart_ra_metrics` uses independent min-max on each image, so absolute scale does not matter — but distributional shape does. RadarFields' `rcs_to_intensity` is in log space (`radar.py:6-12`). Cart correlation is robust to monotonic transform, so this should be fine, but it is worth confirming.

- **(i) Range-resolution conversion.** RadarFields `bin_size_radar` (0.044 m default) is used to convert bin indices to meters (`utils/train.py:60-65`). We set it from our config (~0.117 m/bin); if the conversion is wrong by a constant factor, the rays and the hashgrid live on different scales, causing training to silently diverge. Needs verification — check `mmir.data.io_utils.compute_range_res_from_cfg` against the actual cascade BW (256 ADC samples × 79 MHz/µs × (1/8 MHz) ≈ 2.53 GHz → range_res = c / (2*BW) ≈ 0.059 m — this disagrees with the 0.117 m default above and must be double-checked before running).

## 10. Out of scope

- Hyperparameter tuning of RadarFields on our data (iters, LR schedule, loss weights, hashgrid resolution, occupancy thresholds).
- Ablations of any loss term (`--bimodal`, `--ground_occ`, `--penalize_above`, `--reg_occ`).
- Tuning `num_rays_radar`, `num_fov_samples` beyond what is needed to fit our data.
- Pose refinement beyond "off" or "on-default".
- Voxel-grid visualizations (`--voxels`, `--voxel_only`) and BEV figure rendering beyond what the metric harness needs.
- Upstream features we ignore:
    - Pre-trained demo checkpoint loading (`demo.py`) — we train from scratch.
    - `--learned_norm` offset/scaler training (we keep `initial_offset=1.0`, `initial_scaler=1.0`).
    - Cylindrical / 360° full-scan visualization paths in `utils/vis.py`.
- 3D reconstruction / occupancy-point-cloud evaluation against LiDAR — compared in mm3DGS separately but not in this baseline plan; the RadarFields field already emits `alpha` volumetrically so this could be added later but is out of scope here.
- Transfer-to-single-chip evaluation — RadarFields does not produce a physics forward model transferable to a different radar array, so this evaluation is skipped for RadarFields (document as "method not applicable"). Only the RA NVS number goes on the benchmark row.

## 10a. Upstream files touched — full inventory

This is the complete list of files the baseline will create, edit, or read. No file outside this list is touched.

**Edited upstream files (one-line patches only):**

| File | Lines | Change |
|---|---|---|
| `radarfields/sampler.py` | 31-33 | Replace `360.0` constant with `azim_span_deg` parameter plumbed from `intrinsics_radar`. Single-line diff plus one arg-passing change in `get_radar_rays`. |

**Read-only upstream files (unchanged, just invoked):**

| File | Purpose |
|---|---|
| `main.py` | Entry point (as-is). |
| `parse.py` | CLI parser; we pass overrides via `--intrinsics_radar` YAML dict. |
| `radarfields/dataset.py` | Dataloader, reads our adapter output. |
| `radarfields/nn/models.py` | RadarField network, unchanged. |
| `radarfields/train.py` | Trainer, unchanged. Test-time output ends up in `workspace/imgs/<name>/`. |
| `radarfields/radar.py` | Forward model + occupancy-component helpers we re-use from the adapter. |
| `radarfields/figures.py`, `utils/vis.py` | Visualization only; unchanged. |
| `utils/data.py`, `utils/train.py` | Unchanged. |
| `configs/radarfields.ini` | Base config; overridden per-scene via CLI. |
| `environment.yml` | Conda env spec. |

**New files under `/home/adnan/Desktop/mm3DGS/baselines/radarfields/`:**

| File | Purpose |
|---|---|
| `PLAN.md` | This plan. |
| `patches/sampler_azim_span.patch` | One-line diff against upstream sampler. |
| `adapter/build_dataset.py` | mm3DGS → RadarFields data converter. |
| `adapter/__init__.py` | Marker. |
| `run_all_scenes.sh` | Serial driver over the 7 scenes. |
| `aggregate_metrics.py` | Harness: loads RF outputs + GT, runs `compute_cart_ra_metrics`, emits aggregate JSON. |
| `output/<scene>/metrics.json` | Per-scene metrics (cart_corr, range_profile_corr). |
| `output/aggregate.json` | Mean/std across 7 scenes. |

**Read-only mm3DGS files (imported by adapter + harness):**

| File | Lines | Purpose |
|---|---|---|
| `/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py` | 107-200 | ADC → RA polar conversion. |
| `/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py` | 429-458 | RA polar → cartesian resampling. |
| `/home/adnan/Desktop/mm3DGS/mmir/data/io_utils.py` | `compute_range_res_from_cfg` | Range resolution from per-frame config. |
| `/home/adnan/Desktop/mm3DGS/mmir/evaluation/utils/metrics.py` | 87-117 | `compute_cart_ra_metrics`. |
| `/home/adnan/Desktop/mm3DGS/mmir/evaluation/utils/ra_processing.py` | 86-99 | Alternate range-res helper (sanity cross-check). |
| `/home/adnan/Desktop/mm3DGS/assets/antenna_pattern/MMWCAS/tx1_76.npy` | — | Azimuth LUT source. |
| `/home/adnan/Desktop/mm3DGS/assets/antenna_pattern/MMWCAS/rx1_76.npy` | — | Azimuth LUT source. |
| `/home/adnan/Desktop/mm3DGS/data/seq_X_frame_Y/radar/cascaded_frame_*.npy` | — | ADC source for all 9 frames. |
| `/home/adnan/Desktop/mm3DGS/data/seq_X_frame_Y/configs/cascaded_frame_<center>_aligned.json` | — | Pose + sensor config (center frame's aligned config; per-frame non-center configs stay in their raw form since only relative pose is needed and alignment was solved at the center frame). |

**Not touched (explicitly excluded):**
- Any other baselines under `baselines/` (`dart/`, `radarsplat/`).
- `mm25DGS_v6/`, `mm25DGS_v7*/` production training code.
- Any existing mm3DGS training outputs.

## 10b. Execution-plan skeleton (informational, not code)

Purpose: show the end-to-end pipeline a future execution step will run, so the reviewer can see that the adapter surface is small.

1. GPU check — `nvidia-smi`, abort if any GPU occupied (Section 11).
2. Clone upstream — `git clone https://github.com/princeton-computational-imaging/RadarFields <repo>` (pinned to commit of default branch as of 2024-09-11).
3. Create `radarfields` conda env per Section 7; install tcnn; `pip install -e <repo>`.
4. Apply minimal patch to `<repo>/radarfields/sampler.py` (one-line `azim_span_deg` parameter threaded from `intrinsics_radar`); keep the diff in `baselines/radarfields/patches/sampler_azim_span.patch` for auditability.
5. Run adapter (one new file `baselines/radarfields/adapter/build_dataset.py`) on each of the 7 scenes:
    - Reads `data/seq_X_frame_Y/radar/cascaded_frame_<t>.npy` + `configs/cascaded_frame_<t>_aligned.json` for t in the 9 frames.
    - Writes `<repo>/data/mm3dgs_<scene>/radar/<t>.png` (11-row metadata stripe + normalized RA image).
    - Writes `<repo>/data/mm3dgs_<scene>/azimuth.csv`, `elevation.csv` from `assets/antenna_pattern/MMWCAS/*.npy`.
    - Writes `<repo>/preprocess_results/mm3dgs_<scene>.json` (poses, train/test indices, offsets/scalers).
    - Writes `<repo>/preprocess_results/thresholded_fft/mm3dgs_<scene>/<t>.npy`.
    - Writes `<repo>/preprocess_results/occupancy_component/mm3dgs_<scene>/<t>.npy`.
6. Train per scene (serial):
    ```
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --config configs/radarfields.ini \
        --name mm3dgs_<scene> \
        --seq mm3dgs_<scene> \
        --preprocess_file mm3dgs_<scene>.json \
        --min_range_bin 15 --max_range_bin 110 --num_range_samples 96 \
        --sample_all_ranges \
        --bs 8 \
        --intrinsics_radar '{"opening_h":1.8,"opening_v":40.0,"num_azimuths_radar":127,"num_range_bins":256,"bin_size_radar":<from_cfg>,"azim_span_deg":90.0}' \
        --reg_occ --train_thresholded --save_loss_plot
    ```
    - Upstream auto-runs test after train (`main.py:70-71`) and writes predictions under `<workspace>/imgs/<name>/`.
7. Run harness `baselines/radarfields/aggregate_metrics.py`:
    - For each scene, load rendered polar from RadarFields output, load GT from our `adc_to_ra_image_numpy`, crop `[:, 15:110]`, cartesian-resample, `compute_cart_ra_metrics`.
    - Range-profile correlation = `np.corrcoef(sum_az(rend), sum_az(gt))[0,1]`.
    - Write `aggregate.json` + per-scene `metrics.json`.
8. Log final table: scene | cart_corr | range_profile_corr, plus mean ± std.

Total new code footprint: one patch (~5 lines) + `adapter/build_dataset.py` (~200 lines) + `aggregate_metrics.py` (~100 lines). No refactors.

## 10c. Sanity checks before reporting a number

- **Trivial-pose check.** Run on one scene with all 9 frames used as train (overfit). `cart_corr` on a train frame should exceed 0.80; if not, the pipeline is broken.
- **LUT check.** Visualize the azimuth LUT — peak should be at offset_deg=0, fall off symmetrically by >10 dB at ±45°. If not, the sign convention is wrong.
- **Pose sanity.** Plot refined vs. original poses (`train.py:322` already does this) — refined train poses should stay within a few cm / few degrees of originals.
- **FFT render check.** Overlay `render_FFT_batch` (`train.py:291-297`) output on our GT cartesian image; scene structure should roughly coincide.

None of these override the reported number — if sanity fails, fix; if sanity passes but the number is low, we report the low number with limitations from Section 9.

## 10d. Reported numbers (what a reviewer gets)

Per-scene row in the final benchmark table:
- Method: "RadarFields"
- RA Cart Correlation (mean over 1 held-out frame per scene, single seed).
- Range-profile Correlation (same held-out frame).
- 3D reconstruction metrics: N/A (method does not produce a transferable material; skipped with note).
- Transfer-to-single-chip metrics: N/A (same reason).

Aggregate row:
- Mean ± std of RA Cart Corr across 7 scenes.
- Mean ± std of Range-profile Corr across 7 scenes.

All values rounded to 3 decimals in LaTeX; raw floats retained in `output/aggregate.json`.

Seed policy: single seed (upstream default `--seed 0`, `parse.py:13`). MC noise of ~±0.03 per scene (per mm3DGS experience) applies, so mean over 7 scenes is the comparable number. No multi-seed re-runs unless the reviewer explicitly asks.

Failure reporting: if any scene fails to train (NaN, OOM, pose divergence), report `nan` for that scene's `cart_corr` and document the failure in `output/<scene>/failure.txt`. Aggregate is computed on the completed scenes with a footnote stating which failed — do NOT silently drop scenes.

## 10e. Explicit non-deviations (for reviewer defensibility)

To pre-empt "but you changed X":
- Network architecture: verbatim from upstream (`models.py:46-165`).
- Loss weights: verbatim from upstream `configs/radarfields.ini:30-34`.
- Optimizer: Adam, betas (0.9, 0.99), eps 1e-15, LR 1e-3 with decay to 0.1x (`main.py:39-44`) — verbatim.
- LR schedule: LambdaLR `0.1 ** min(iter/iters, 1)` — verbatim.
- Coarse-to-fine HashGrid mask: verbatim (`--mask` on, `models.py:9-20`).
- Super-sampling within beam (`num_fov_samples=10`) and integration via antenna LUT (`--integrate_rays`): verbatim.
- `rcs_to_intensity` with `--approximate_fft` (i.e., no 1/r² term): verbatim upstream default (`configs/radarfields.ini:22`).
- Checkpoint policy: keep-2, save per epoch — verbatim.

Deviations (all in Section 3/4/9):
- Azimuth span constant 360° → 90° (forced by our sensor's forward FoV).
- `num_azimuths_radar` 400 → 127 (forced by our beamforming output width).
- `num_range_bins` 7536 → 256, `bin_size_radar` default → computed from config.
- `bs` 10 → 8 (forced by 8 train frames).
- Elevation LUT from measured → flat (our cascade has no measured elevation pattern).
- Intensity normalization: min-max per-frame (no upstream noise-floor value for our data).
- Pose refinement: disabled by default to remove a confound on a small-n training set (revisit if needed).

Every deviation is a direct consequence of sensor-geometry or data-scale mismatch, not a tuning choice.

## 11. GPU-safety preamble (MANDATORY)

Before any future execution step that touches a GPU (training, testing, figure rendering, even `import torch` to a real device), the first command MUST be:

```bash
nvidia-smi
```

Rule: if ANY process is listed on ANY GPU — regardless of memory load or utilization % — **abort the execution plan and surface it to the user**. Do NOT share a GPU. Partial occupancy does NOT count as free. This applies to both GPU 0 and GPU 1 on this 2x RTX 4090 workstation. Selecting a specific device via `CUDA_VISIBLE_DEVICES=0` is fine only if that device is idle per the same `nvidia-smi` check; occupying it with a second user is still disallowed.

This is section 11 for a reason: it is the gate, not the afterthought.
