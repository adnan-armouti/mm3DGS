# RadarSplat Baseline Plan (NeurIPS 2026)

Target: a defensible, minimal-changes RadarSplat baseline on the mm3DGS 7-scene benchmark.
Not: the strongest possible RadarSplat. Every deviation is a documented limitation.

Upstream: `umautobots/radarsplat` (ICCV 2025). Paper: arXiv 2506.01379.
Default branch: `release`. Last push: 2026-02-10. Commit/tag pinning is an Open Question (Section 9).

---

## 1. Repo summary

RadarSplat fits per-scene 3D Gaussians that rasterize to a Cartesian or polar range-azimuth
(RA) image and match a measured 360° scanning-radar intensity image. It extends gsplat with
a custom `_radar_rasterization` kernel, explicit "noise probability" per Gaussian,
post-rasterization antenna-gain convolution along azimuth, and sinc-shaped spectral-leakage
convolution along range. Inputs are uint8 PNG polar RA images (H azimuth rows x W range cols),
per-frame `radar_trajectory.tum` world-from-sensor poses, a per-frame synced LiDAR point
cloud, a per-frame LiDAR "map" (local window fusion), a precomputed per-frame noise mask /
"radar average map" used as an occupancy target, and a precomputed per-frame multipath
reconstruction. Loss = L1(render, raw) + lambda_ssim * (1 - SSIM) + lambda_occ * L1(occ, mask)
+ lambda_maxsize * max-scale-relu + lambda_reg * opa-noise-cap (upstream
`examples/radar_simple_trainer.py:600-740`). Default training loop is 2000 steps (upstream
`examples/demo_scripts/run_radarsplat.sh:131-160`).

Citations:
- `umautobots/radarsplat:examples/radar_simple_trainer.py:519` — main forward with
  `_radar_rasterization(..., sph=use_polar)`.
- `umautobots/radarsplat:gsplat/rendering.py:810` — `_radar_rasterization` signature
  (means, quats, scales, opacities, noise_probs, colors, viewmats, Ks, width, height,
  camera_model="ortho", sph).
- `umautobots/radarsplat:gsplat/rendering.py:1362-1384` — `spectral_leakage` Gaussian along
  range, width from `sinc_width / range_resolution`.
- `umautobots/radarsplat:gsplat/rendering.py:1385-1413` — `azimuth_antenna_gain_projection`;
  convolution along azimuth with **circular padding** (`torch.cat([raw[-p:], raw, raw[:p]])`).
  This is the hardwired 360° assumption (see Section 4 and 9).
- `umautobots/radarsplat:examples/radar_simple_trainer.py:668-730` — loss composition.
- `umautobots/radarsplat:examples/demo_scripts/run_radarsplat.sh:140-163` —
  `--max_steps 2000`, `--max_range 50`, `--init_num_pts 20000`, `--init_scale 0.5`,
  `--test-every 5`.

Input / output tensor shapes (upstream defaults, `boreas/data_processing/sensor.yaml`):

| Tensor | Shape | Units |
| --- | --- | --- |
| GT polar image (pre-crop) | (400, 3371) uint8 | intensity /255 in [0,1] |
| GT polar image after `max_range=50` | (400, 839) uint8 | range bins at 0.0596 m |
| GT polar image at `intermediate_azimuth_resolution=0.1deg` | (3600, 839) | after stride-9 azim gain conv -> (400, 839) |
| TUM pose | (4,4) float | world-from-sensor, SE(3), meters |
| Gaussians | N=20_000 init, (3 means, 4 quats, 3 scales, 1 opacity, 1 noise_prob, (sh_deg+1)^2 x 3 SH) | world coords, meters |
| K (ortho polar) | diag(1/range_res, 1/(azim_res_rad), 1) | px/m, px/rad |

---

## 2. Upstream data format

`WaveSensorDataParser.__init__` (upstream `radar/dataset/dataloader.py:99-190`) expects a
per-sequence directory:

```
<data_dir>/<seq_name>/
  sensor.yaml                                  # sensor intrinsics, see below
  radar_trajectory.tum                         # world-from-sensor SE(3) per frame
  images/<frame_name>.png                      # uint8 polar RA, shape (H, W_metadata + W)
                                               # leading W_metadata=11 cols = timestamp blob
                                               # (stripped in __getitem__)
  synced_lidar/<frame_name>.pcd                # per-frame lidar point cloud
  synced_lidar_map_win5/<frame_name>.pcd       # 5-frame-window fused lidar map
  radar_average_map_polar/<subdir>/<frame>.png # uint8 occupancy target (thresholded
                                               # per-scene long-window average radar map)
  multipath_model/dist:50/<frame_name>.npy     # dict: azi_id_list, range_id_list,
                                               # reconstructed_signal for multipath background
```

Citations:
- `umautobots/radarsplat:radar/dataset/dataloader.py:108-126` — poses loader, `sensor.yaml`
  parse, `image_dir = os.path.join(data_dir, "images")`.
- `umautobots/radarsplat:radar/dataset/dataloader.py:128-135` — `multipath_dir =
  os.path.join(data_dir, "multipath_model/dist:50")`.
- `umautobots/radarsplat:radar/dataset/dataloader.py:152-173` — polar K built from
  `sensor['range_resolution']`, `intermediate_azimuth_resolution`, expects `H = 360 /
  intermediate_azimuth_resolution = 3600` rows for polar rendering (not the file H=400).
- `umautobots/radarsplat:radar/dataset/dataloader.py:267-318` — `__getitem__`: reads PNG,
  strips first `W_metadata` cols, crops to `max_range`, loads lidar pcds + multipath npy.
- `umautobots/radarsplat:boreas/data_processing/sensor.yaml` — `sensor_type: scanning_radar`,
  `use_polar: True`, `H: 400`, `W: 3371`, `W_metadata: 11`, `range_resolution: 0.0596`,
  `azimuth_resolution: 0.9`, `azimuth_beamwidth: 1.8`, `azimuth_coverage: 360`.

Pose convention (upstream `radar/dataset/dataloader.py:29-56`): TUM `timestamp tx ty tz qx
qy qz qw`. The loader stores these directly as the 4x4 `radarposes[k]` without inversion.
Variable name is `w2r_mats` but the values are world-from-sensor (same as every TUM file);
the gsplat rasterizer's `viewmats` argument is called with `radarposes` (upstream
`examples/radar_simple_trainer.py:510`). This is inconsistent with gsplat's usual convention
(viewmats = sensor-from-world) — **needs verification** (Section 9 item g).

Image layout: polar, H = azimuth rows (0..360 deg going top to bottom), W = range columns
(0 = sensor origin, increasing range to the right). Range bin size 0.0596 m. Azimuth row
size 360/H deg = 0.9 deg at H=400, 0.1 deg at the intermediate (3600) rendering resolution
which is stride-downsampled through the Gaussian azimuth-gain kernel to H=400.

---

## 3. Adapter: mm3DGS data -> RadarSplat format

We transform our data to match what the upstream dataloader reads. We do NOT modify the
dataloader (principle: minimal changes to the method). The adapter runs once per scene as a
preprocessing step and writes to `baselines/radarsplat/data_radarsplat/<scene>/...`.

Per scene (7 benchmark scenes), per frame (9 cascaded frames), produce:

- **`images/<frame_id>.png`** — uint8 polar RA, H=400 rows x (11 + W_range) cols.
  - Source: `data/<scene>/radar/cascaded_frame_<N>.npy` (complex128, (16,16,12,256))
  - Step 1: `mmir.data.ra_utils.adc_to_ra_image_numpy(adc)` -> polar RA (float, ~86
    azimuth bins x 256 range bins), magnitude already (|FFT|). Internally does the MIMO
    `txrx_to_virtual_array_numpy` + range-FFT + azimuth-FFT path we already use for GT.
  - Step 2: **zero-pad azimuth into 400 rows** placing the ~86-bin mm3DGS azimuth wedge
    (roughly -21..+69 deg relative to sensor forward) into the matching rows of a 400-row
    full-360 buffer, rest = 0. This mirrors what the sonar branch at
    `examples/radar_simple_trainer.py:654-658` does (pad to 360, loss computed only on
    populated rows). We do the padding OURSELVES in the adapter and additionally provide
    a "valid-row" mask (Section 4) so the loss only sees real data.
  - Step 3: range axis reshape. mm3DGS native range bin size is
    `c / (2 * fs * N_fft) * (fs / slope / Tc)` = ~0.0376 m at 8 MHz, 79 MHz/us, 34 us. To
    feed RadarSplat's fixed `range_resolution=0.0596` without modifying the upstream
    dataloader, we resample (linear interp) the RA image along range to a 0.0596 m grid of
    length W=3371. Range bins beyond our physical 20 m max are zero.
  - Step 4: rescale magnitude to uint8. Per-scene 99.5-percentile normalization to [0,255],
    clip. Persist the per-scene scale factor to `baselines/radarsplat/data_radarsplat/<scene>/
    normalization.json` (used only for diagnostics, not training).
  - Step 5: prepend an 11-column zero metadata strip (dataloader strips it, but shape must
    match `W_metadata=11`).

- **`radar_trajectory.tum`** — 9 lines `timestamp tx ty tz qx qy qz qw`.
  - Source: per-frame pose already in `data/<scene>/configs/cascaded_frame_<N>_aligned.json`
    (the hybrid-pass-2 aligned variant used by mm3DGS production code; needs verification
    which exact field, typically `T_world_cascade`). Convert to quaternion + translation
    using scipy `Rotation.from_matrix(...).as_quat()` (note scipy's (qx,qy,qz,qw) order
    matches TUM).
  - Units: mm3DGS poses are already in meters, right-handed, z-up — same convention as
    Boreas. No axis flips expected. Timestamp: use the frame number (131..139) as a float
    "timestamp" (order-only, not used for interpolation).

- **`sensor.yaml`** — identical to upstream except:
  - `H: 400`, `W: 3371`, `W_metadata: 11`, `range_resolution: 0.0596`,
    `azimuth_resolution: 0.9`, `azimuth_beamwidth: 1.8`, `use_polar: True`.
  - **`azimuth_coverage`** is set to 90 (not 360) in our yaml. See Section 4 — this is read
    by the dataloader but its consumer is essentially absent in the release code (only
    `self.parser.azimuth_coverage = ...`, used downstream only if we add an explicit guard).

- **`synced_lidar/<frame_id>.pcd`** — from `data/<scene>/lidar/lidar_frame_*.npy` if
  available; otherwise reuse `scene/pcl.npy` XYZ columns (col 0..2). Open3D `.pcd` format.
  Used only for viz + eval_geometry, NOT for training loss.

- **`synced_lidar_map_win5/<frame_id>.pcd`** — same as above (we do not have a 5-frame
  lidar fusion; duplicate the per-frame pcd). Documented as limitation.

- **`radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0/<frame_id>.png`**
  — uint8 polar "occupancy" target. Upstream computes this by averaging radar returns over a
  window and thresholding. We approximate: for each frame, take the RA image from step 1-4,
  apply `compute_spherical_grid_noise_threshold` (available in upstream
  `radar/dataset/dataloader.py:58-71`) and threshold at 0.10 (matches
  `--radar_map_thres 0.10` in the run script). Written as a per-frame PNG.
  - Limitation: upstream `radar_average_map` is a scene-level aggregate, not per-frame. With
    9 frames we cannot build a meaningful aggregate. Document.

- **`multipath_model/dist:50/<frame_id>.npy`** — dict with keys `azi_id_list`,
  `range_id_list`, `reconstructed_signal`. Upstream fills this from a separate preprocessing
  script that is not in the release README table (Section 9 item h — needs verification,
  check `boreas/data_processing/radar_grid_map.py`). Minimal-changes choice: **write a
  degenerate file with empty lists and a zero `reconstructed_signal`** so the `multipath_bg`
  addition at `examples/radar_simple_trainer.py:695-700` is a no-op. Then set
  `--multipath_weight 0.0` in our run (this is the `no_mp_modeling` ablation variant from
  upstream's own `run_all_radarsplat_abla.sh`). Document: we run RadarSplat without its
  multipath module because we have no multipath-source estimation pipeline and building one
  violates minimal changes.

Adapter source file: `baselines/radarsplat/adapter/mm3dgs_to_radarsplat.py` (to be written at
execution time, not now).

---

## 4. Geometry / chirp reconciliation

RadarSplat is hardwired for 360° scanning radar. Our cascaded radar is a forward-facing
~90° wedge. Strategy: feed the wedge as a 360°-shaped image with zero rows outside the wedge,
render the full 360° polar image at train time, but mask the loss so only populated rows
contribute. Upstream already has exactly this code path for sonar (130°):

- `umautobots/radarsplat:examples/radar_simple_trainer.py:654-667` — `if sensor_type ==
  "sonar": out_img_130 = out_img[:pixels.shape[0], :]` and `l1loss = F.l1_loss(out_img_130,
  pixels)`.

**Minimum edit A (one-line on the method):** change the `if sensor_type == "sonar"` guard
at `examples/radar_simple_trainer.py:654` and `:661` to also fire for
`sensor_type == "scanning_radar" and azimuth_coverage < 360`. We reuse the sonar branch
mechanically. No architectural or loss change.

**Minimum edit B (sensor.yaml):** set `azimuth_coverage: 90` (up from 360) in our adapter's
yaml output. The sonar branch reads `pixels.shape[0]` not `azimuth_coverage`, so we must
also shape `pixels` such that it covers only the populated wedge rows, not the zero-padded
360. Concretely: at `examples/radar_simple_trainer.py:625-628` keep `pixels` at 400 rows but
crop to the wedge rows (e.g. rows [i_min : i_max]) BEFORE the loss. The sonar branch already
expects `pixels` to be less than 360° tall. We implement this crop in a new override in the
trainer — see Edit C.

**Minimum edit C (trainer, ~3 lines):** inside `train()` in `radar_simple_trainer.py` right
after `pixels = data["image"] ...`, if `self.parser.azimuth_coverage < 360`, slice
`pixels = pixels[:, i_min:i_max, :]` and `pixels_occ = pixels_occ[:, i_min:i_max, :]`, with
`i_min, i_max` computed once from the parser (corresponds to azimuth rows covering
[-21°, +69°] at 0.9°/row).

**NOT changed:**
- Gaussian model (means/quats/scales/opacities/noise_probs/SH) — identical.
- `_radar_rasterization` CUDA + Python wrapper — identical.
- `spectral_leakage` sinc-shaped range kernel — identical (runs on the full 400-row tensor
  before the crop, which is fine — it's range-axis only).
- `azimuth_antenna_gain_projection` — identical; circularly convolves along azimuth before
  the crop. This is suboptimal (wraps zero rows into the wedge and vice-versa) but crucially
  the stride=9 downsample from 3600 -> 400 rows still happens on the full 360°, so the wedge
  retains the right azimuthal alignment. Any leakage from the wrap is suppressed by the crop.
  **Document as a limitation.**
- Densification strategy — identical.
- Optimizer hyperparameters — identical.
- Loss coefficients — identical to `default` variant (multipath_weight overridden to 0 per
  Section 3).

**GT cropping:** same slice rows [i_min:i_max] applied to the GT polar image (already in the
adapter, but kept symbolic in Edit C so the code can be re-run without regenerating data).

**Range crop:** we set `--max_range 20` (not 50) because our mm3DGS renderer is validated
only out to ~20 m and bins 15..110 of the native 256-bin grid correspond to
~0.56 .. 4.13 m at 0.0376 m/bin — but in the resampled 0.0596 m RadarSplat grid bin 110 is
~6.56 m. We keep `--max_range 20` to be safe and then compute metrics only on the overlap of
RadarSplat bin indices 15..(335) with our physical bins. Exact bin mapping needs
verification (Section 9 item i). The 15..110 native-bin crop from mm3DGS protocol is applied
at metric time (Section 6), not at RadarSplat input time.

**Chirp / Doppler:** RadarSplat consumes the magnitude polar image. It does not see chirps,
Doppler, TX/RX geometry, or polarization. Our 12x16 MIMO collapses to a single post-FFT
azimuth spectrum via `adc_to_ra_image_numpy`. No per-TX / per-polarization change.

---

## 5. NVS split

Identical to mm3DGS, RadarFields, DART. 8 outer cascaded frames train, middle frame test.
RadarSplat's dataloader uses `indices % test_every == 0` for val. We set `test_every=9` and
place the test frame at index 4 of 9 by ordering the filenames appropriately — **but this
does not match `% test_every == 0`**.

Workaround (minimal): name files so the test frame has index 0 alphabetically? No — that
would extrapolate (test at edge). Instead, after loading, we monkey-patch
`WaveSensorDataset.indices` with an explicit `[4]` for val and `[0,1,2,3,5,6,7,8]` for
train in one small override in our trainer wrapper. This is an ~8-line change to the
trainer entry point, not an algorithmic change. Document.

| Scene | Train cascaded frames (8) | Test cascaded frame (1) |
| --- | --- | --- |
| seq_0_frame_135 | 131, 132, 133, 134, 136, 137, 138, 139 | 135 |
| seq_0_frame_390 | 386, 387, 388, 389, 391, 392, 393, 394 | 390 |
| seq_1_frame_185 | 181, 182, 183, 184, 186, 187, 188, 189 | 185 |
| seq_1_frame_438 | 434, 435, 436, 437, 439, 440, 441, 442 | 438 |
| seq_2_frame_105 | 101, 102, 103, 104, 106, 107, 108, 109 | 105 |
| seq_2_frame_160 | 156, 157, 158, 159, 161, 162, 163, 164 | 160 |
| seq_2_frame_300 | 296, 297, 298, 299, 301, 302, 303, 304 | 300 |

Frame-number convention here assumes the 9 cascaded frames per scene are
`center_frame_number +- {0,1,2,3,4}` spaced by 1 index. Verify against each
`data/<scene>/radar/cascaded_frame_*.npy` listing before running (Section 9 item j).

Viability: 8 training views with ~20 cm inter-frame translation is marginal for Gaussian
splatting. Upstream's default sequences use 40-frame training windows
(`examples/demo_scripts/seq_all.txt`, e.g. "boreas-2021-09-02-11-42 27 67" = 40 frames).
This is explicitly called out in Section 9 item d and is expected to degrade RadarSplat's
numbers. We accept that — it is the fair-split weakness of the baseline.

---

## 6. Metrics harness

No new metric code. We reuse mm3DGS's existing pipeline:

1. Train RadarSplat on 8 frames -> checkpoint.
2. Run RadarSplat's val step on the held-out frame -> rendered **polar** RA image
   (H=400, W=839 at `max_range=50`, or W=336 at `max_range=20`).
3. Extract the wedge rows [i_min:i_max] from the rendered polar image; these correspond to
   the mm3DGS 90° azimuth FoV.
4. Convert rendered polar -> Cartesian via
   `mmir.data.ra_utils.ra_polar_to_cartesian(ra_wedge, range_res=0.0596)`
   (`/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py:429`).
5. Load GT ADC `data/<scene>/radar/cascaded_frame_<test>.npy`, run
   `adc_to_ra_image_numpy` (`/home/adnan/Desktop/mm3DGS/mmir/data/ra_utils.py:174`)
   -> polar -> `ra_polar_to_cartesian`. Use mm3DGS's native range resolution here; the
   Cartesian step resamples both onto a common grid.
6. Compute metrics with
   `mmir.evaluation.utils.metrics.compute_cart_ra_metrics(ra_gt_cart, ra_rend_cart)`
   (`/home/adnan/Desktop/mm3DGS/mmir/evaluation/utils/metrics.py:87`). Returns Pearson RA
   correlation and range-profile correlation. Range-bin crop 15..110 is applied **before
   Cartesian conversion** inside `compute_cart_ra_metrics` (or inside the caller, per
   `eval_training_ra_v2.py:289-292`). Double-check the crop is applied equivalently to
   RadarSplat's output (Section 9 item k).
7. Mean / std over the 7 benchmark scenes.

Harness script (to be written at execution time):
`baselines/radarsplat/eval/eval_radarsplat.py` — loads RadarSplat checkpoint via a thin
wrapper around upstream's `Runner.eval_one_frame` (or equivalent; the upstream eval path is
`examples/radar_simple_trainer.py:745+` **needs verification**), renders the test frame,
plumbs the polar image into `compute_cart_ra_metrics` exactly like
`eval_training_ra_v2.py:289-292`.

---

## 7. Env setup

From upstream README (citations: `umautobots/radarsplat:README.md:23-68`):

```
conda create -n radarsplat -y python=3.9
conda activate radarsplat

# PyTorch (CUDA 11.8 path, aligns with 4090):
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Repo + gsplat submodule (custom with _radar_rasterization)
git clone --recursive https://github.com/umautobots/radarsplat.git
cd radarsplat
cd examples && pip install -r requirements.txt && cd ..
pip install open3d wandb
pip install -e . --no-build-isolation --config-settings editable_mode=compat

# Known breakage: jaxtyping/nerfview. Fix:
pip install nerfview --no-deps
pip install jaxtyping==0.2.19

pip install asrl-pyboreas  # only for upstream data preprocessing; NOT used by our adapter
```

Pins we add (to `baselines/radarsplat/requirements.txt`):

- `python==3.9.x` (conda)
- `torch==2.1.2+cu118`, `torchvision==0.16.2+cu118`
- CUDA toolkit 11.8 via conda (for gsplat/tiny-cuda-nn compile)
- GPU driver floor: R525+ for CUDA 11.8 runtime (4090 supported).
- `jaxtyping==0.2.19`, `nerfview` (no-deps)
- `open3d` (any recent), `wandb`
- Pin upstream repo to a specific commit in `baselines/radarsplat/UPSTREAM_COMMIT.txt`
  (commit hash needs to be chosen at clone time; see Section 9 item a).

Verify build: `python -c "import gsplat; from gsplat import _radar_rasterization; print('ok')"`.

**Isolation guarantee:** `radarsplat` env is separate from `mmir`. mm3DGS code is never
imported inside the `radarsplat` env at train time. At **metric time**, we either (a)
reimplement `compute_cart_ra_metrics` as a 30-line pure-numpy function inside the baselines
tree and call it from `radarsplat` env, or (b) run the metric step in `mmir` env on a
numpy-dumped rendered image. Option (b) is preferred — zero duplication.

---

## 8. Compute estimate

Upstream default: `--max_steps 2000`, `--init_num_pts 20000`, 40 training frames per scene,
polar image ~400x839. Reported wall-time is **not explicit** in README; rough order from
the iteration count (2000) and per-step cost of a ~20k-Gaussian polar rasterization at
400x839 on an RTX-class GPU: ~2-5 minutes/scene on a 4090. This is an estimate
(needs verification, Section 9 item c).

Our setting: 8 training frames, 2000 steps, same 20k Gaussians, wedge-cropped loss. Per-step
cost should be lower or identical. Expected wall-time per scene: **under 10 minutes**.

Total for 7 scenes serial: **< 2 hours**, well under the 24 h/scene cap.

If it unexpectedly overruns 24 h/scene, knobs (in priority order):
1. Reduce `--max_steps` from 2000 to 1000. Upstream calls 2000 "default"; halving is safe
   for a baseline number.
2. Reduce `--init_num_pts` from 20000 to 10000. Smaller scene, fewer views.
3. Reduce `--intermediate_azimuth_resolution` from 0.1 to 0.2 (halves the 3600-row
   intermediate buffer).
4. Turn off SSIM loss (`--ssim_lambda 0`) — kills the fused_ssim cost on 400x839.

None of these are expected to be needed.

---

## 9. Open questions / risks

Each item is tagged (a)..(l). Resolve before execution, or document as baseline limitation.

(a) **Upstream commit pin.** Default branch is `release`, last push 2026-02-10. No tag.
Needs verification — clone, record commit hash in `baselines/radarsplat/UPSTREAM_COMMIT.txt`
before any run.

(b) **360° circular padding in `azimuth_antenna_gain_projection`.** File
`umautobots/radarsplat:gsplat/rendering.py:1385-1413` hardcodes
`torch.cat([raw[-p:], raw, raw[:p]], dim=0)`. With our 90° wedge padded into a 360° buffer
of zeros, the wrap leaks zeros into wedge rows near the edges. The wedge crop at loss time
masks this, but the rendered wedge near its edges has ~1 beamwidth (1.8°) of underestimated
energy. Risk: Pearson correlation of the edge azimuths is biased downward. Mitigation: the
mm3DGS metric is a single scalar over the full cart image; bias is small (~2 rows out of
100). Documented as limitation. **Needs verification** by comparing two runs: zero-padded
vs. full-360 (out of scope for minimal-changes).

(c) **Wall-time per scene.** Upstream README reports no timing. 2-5 min/scene is a
semi-educated guess from Gaussian count + step count. Needs verification on first scene.

(d) **Is 8 frames enough to initialize Gaussians?** Upstream uses 40 frames at init (see
`seq_all.txt`). With `init_type=random` over the full scene bbox (which is
`scene_scale = max(radar_location_dist_from_mean) + 2 * W * range_res ~= ~100 m` for a
40-frame trajectory but only ~5 m for our 9-frame trajectory), random init of 20k Gaussians
uniformly in a 5x5x0 m box with `init_scale=0.5 m` is a DENSE overfit to a small local
volume. May overfit catastrophically, may work fine. Upstream also supports
`init_type=predefined` using `parser.points` (from a point cloud). We could optionally
predefine from our `scene/pcl.npy` — but this is a deviation. Stick with `random`.
**Expected outcome: numbers are worse than "fair" RadarSplat. Accept as split weakness.**

(e) **`radar_average_map_polar` approximation.** Upstream builds this as a multi-frame
aggregate with `CR_thres:0.21_smooth:3.0`. Our per-frame synthetic version uses
`compute_spherical_grid_noise_threshold`. Semantically different. Impact: the occupancy
loss (`l1occloss_lambda=10`) pulls Gaussians toward a noisier target. Risk: mild
degradation of reconstruction + more spurious Gaussians. Minimal-changes choice: keep the
loss on, document. Alternative: disable occupancy loss via `--variant no_occ`
(`l1occloss_lambda=0`, upstream's own ablation). Open question: run `default` or `no_occ`?
**Recommended: report both** — adds 7 more scenes of compute but still under 4 h total.

(f) **Multipath model absence.** Covered in Section 3. We ship a zero multipath and set
`--multipath_weight 0`. This is upstream's `no_mp_modeling` ablation variant. Documented.

(g) **TUM pose frame convention.** Upstream variable `w2r_mats` suggests "world-from-radar"
(world coords expressed in radar frame, i.e. sensor-from-world). TUM files canonically store
the opposite (radar pose in world, world-from-sensor). Upstream passes these directly as
`viewmats` to `_radar_rasterization`, which in standard gsplat = sensor-from-world. If the
comment is wrong (the values really are world-from-sensor), the rendered image is inverted.
**Needs verification**: compare one forward render against GT for frame index 0 before
running the full benchmark. Check by comparing with `load_tum_poses` vs. the Boreas TUM
convention documented at pyboreas.

(h) **Multipath-model preprocessing script.** The README preprocessing pipeline
(`process_seq_paper.sh`) does not mention multipath generation explicitly. The
`multipath_model/dist:50/<frame>.npy` is consumed at
`radar/dataset/dataloader.py:128-135` and `examples/radar_simple_trainer.py:686-696`.
Needs verification — grep upstream for the producer, likely an undocumented script in
`boreas/data_processing/`. If producer is absent, confirms our zero-multipath choice.

(i) **Range-bin crop alignment.** mm3DGS protocol: bins 15..110 of the 256-bin native grid
at 0.0376 m/bin = 0.56..4.13 m. RadarSplat grid: 0.0596 m/bin. Our GT-in-RadarSplat-grid
goes 0..50 m. The mm3DGS metric code already crops at the Cartesian stage
(`eval_training_ra_v2.py:289-292`). Risk: if the crop is applied on bin indices rather than
physical range, the two grids see different physical ranges. **Needs verification** — read
`compute_cart_ra_metrics` body (`mmir/evaluation/utils/metrics.py:87`) and confirm the crop
is in meters, not bin indices.

(j) **Exact cascaded frame numbering per scene.** Table in Section 5 assumes the 9 frames
per scene are `center +- {0..4}`. Verify once per scene with `ls data/<scene>/radar/`.

(k) **RadarSplat polar H vs. the 400-row assumption.** At `--max_range 20` the W changes to
335 range bins but H is still 400 rows and the FoV-crop rows are determined by
`azimuth_coverage=90`. Verify `i_min, i_max` math: forward direction = +x, cascade FoV is
-21° to +69° at the sensor. Upstream polar layout: row 0 = 0° azimuth (which direction is
"0°"?). **Needs verification** against upstream azimuth convention (probably 0° = +y or +x,
increasing CCW or CW) — read a reference frame at
`boreas/data_processing/radar_map_cart2polar.py` azimuth ordering.

(l) **Metric env isolation.** Running the mm3DGS metric on output from the `radarsplat`
env requires either dumping intermediate numpy arrays and loading in `mmir`, or
re-implementing `compute_cart_ra_metrics` in the `radarsplat` env. Section 7 decides (b):
dump numpy, evaluate in `mmir` env. No code issue, just process. **Documented, not verified.**

---

## 10. Out of scope

Explicitly not doing in the baseline run:

- Hyperparameter tuning (`ssim_lambda`, `maxsize_lambda`, `l1occloss_lambda`,
  `init_num_pts`, `init_scale`, `refine_*` — all left at upstream's `default` variant
  except `multipath_weight=0`).
- Replacing the `azimuth_antenna_gain_projection` circular padding with a windowed wedge
  version (would improve edge rows; out of scope because it's a method-internal change).
- Rebuilding `multipath_model/dist:50` from our cascaded ADC. Separate preprocessing
  pipeline; would double the engineering cost.
- Rebuilding `radar_average_map_polar` from a multi-scene aggregate (we only have 9 frames
  per scene; the aggregate would be degenerate).
- Implementing a cascade-specific Gaussian init from `scene/pcl.npy`. Upstream supports
  `init_type=predefined` but using it is a non-minimal change.
- 360° cylindrical projection — upstream feature that is active by default; we keep it on
  but mask the loss, so this is a cost issue, not correctness.
- All of RadarSplat's ablations (`no_occ`, `no_sl`, `no_mp_modeling`, `no_noise_prob`)
  beyond the multipath override. If time permits, re-run with `no_occ` as a sensitivity
  check (Section 9 item e).
- Transfer evaluation to IWR1443 single-chip. RadarSplat has no material/physical model —
  Gaussians are tied to the sensor's antenna pattern. A fair transfer eval requires
  retraining on single-chip data, which is a separate experiment, not a baseline number.
- eval_3d_reconstruction / eval_3d_occupancy in mm3DGS. Our benchmark number is RA +
  range-profile correlation only.

---

## 11. GPU-safety preamble (MANDATORY first step of any execution plan)

Before any `python`, `pip install -e`, `nvcc`, CUDA-touching invocation:

1. Run `nvidia-smi`.
2. If **any** row in the "Processes" table shows **any** PID on **any** GPU — including
   display servers, other users' jobs, jupyter kernels, vscode-server CUDA probes, or our
   own leftover python — **ABORT**. Do not proceed. Do not set `CUDA_VISIBLE_DEVICES` to
   the other GPU and try to run there. Do not kill the other process. Do not use
   `CUDA_MPS`. Do not share.
3. Inform the user. The user will clear the GPU and re-issue the run.
4. Only when `nvidia-smi` shows **0** processes under "Processes" and **0 MiB** used on
   **both** 4090s is it safe to start.
5. Partial occupancy (one free GPU, one busy) does NOT count as free. The RadarSplat
   compile step and the first few training iterations transiently touch both GPUs
   (driver init, NCCL probe via DDP import). Abort if either GPU is busy.

This rule is verbatim the hard-rule 1 in the plan spec and applies to the execution phase
only. This planning file touched no GPU.
