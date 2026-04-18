# Frame-NVS parameter analysis + regularizer design

**Use this document as the starting prompt in a new Claude Code session.**
It is self-contained: it specifies the problem, the data on disk, the
analyses to run, the hypotheses to validate, and the success criteria. No
prior conversation history is needed.

---

## 0. TL;DR

Two training scenes (`seq_1_frame_438`, `seq_2_frame_105`) each have 4
converged parameter sets on disk (**UB** = test frame was in training,
**HO** = test frame was held out; full-chirp = 16 loops/frame,
first-chirp = 1 loop/frame). On both scenes the UB converges cleanly
(cart_corr **0.88–0.95** on the held-out RA), but the HO variants
stall at **0.49–0.51**. That ~0.4 gap is the failure mode.

**Goal:** analyse the 4 converged parameter sets for each scene, identify
the parameter-space differences between UB (ceiling) and HO (current),
hypothesise regularisation terms motivated by those differences, retrain
HO with the chosen regulariser, and hit **HO cart_corr ≥ 0.7** on both
scenes. Any insight you derive **must hold across both scenes** — no
single-scene overfits.

This problem is well-posed: UB proves a parameter set exists that maps
the training RAs plus the test RA to high cc. We need to find a
regulariser that steers the HO optimiser toward that same basin without
access to the test RA.

---

## 1. Codebase state & data layout

Everything referenced below is on disk. Do **not** re-run any of the
training or alignment — just load the artefacts and analyse.

### 1.1 Training runs (`mm25DGS_v5/output_frame_nvs/<scene>_<tag>/`)

For each of the two target scenes (`seq_1_frame_438`, `seq_2_frame_105`)
there are **4 variants** under this root:

| variant | tag suffix | n_train_samples | test frame in train? | test HO cc (seq_1 / seq_2) |
|---|---|---|---|---|
| HO (128 RA)             | `train8frames_test<F>_loop0_pass2`           | 8 × 16 = 128  | no  | 0.4904 / 0.5043 |
| HO (8 RA, first-chirp)  | `train8frames_1loops_test<F>_loop0_pass2`    | 8 × 1  = 8    | no  | 0.5003 / 0.5116 |
| UB (144 RA)             | `train9frames_16loops_test<F>_loop0_ub_pass2`| 9 × 16 = 144  | yes | 0.9126 / 0.8228 |
| UB (9 RA, first-chirp)  | `train9frames_1loops_test<F>_loop0_ub_pass2` | 9 × 1  = 9    | yes | 0.9510 / 0.8814 |

`<F>` is the centre-frame index (438 for seq_1_frame_438, 105 for
seq_2_frame_105). The test RA is always **loop 0 of the centre frame**.

Each run directory contains:

- `results.json` — all metadata: train_frames, train_loops, test_in_train,
  init_test_cc, final_test_cc, init_train_mean_cc, final_train_mean_cc,
  best_iter, elapsed_s, edge_frames, test_pose_mode
- `best_model.pt` — torch `state_dict` with these exact shapes:
    - `positions`     `(N, 3)`  float32 — **frozen**, the LiDAR-FPS point set
    - `rotations`     `(N, 4)`  float32 — **trainable** quaternion `[w, x, y, z]`
    - `raw_materials` `(N, 6)`  float32 — **trainable** 6 raw material params
- `history.npz` — per-iter `iters`, `loss`, `mean_train_cc` (500 iters each)

`N = 90000` on every run (FPS target point count). The FPS seed is
fixed (42), so **point `i` in any run is the same physical xyz
point** — you can do per-point cross-variant comparisons without
alignment.

Loading:
```python
import torch
state = torch.load(
    'mm25DGS_v5/output_frame_nvs/seq_1_frame_438_train8frames_test438_loop0_pass2/best_model.pt',
    weights_only=False, map_location='cpu')
positions = state['positions']     # (N, 3)
rotations = state['rotations']     # (N, 4) quaternion
raw       = state['raw_materials'] # (N, 6)
```

### 1.2 Init state (starting point for every training run)

- Positions: deterministic FPS from `data/<scene>/scene/pcl.npy`.
- Rotations: quaternions that rotate `+z` onto each point's LiDAR
  normal (loaded from `pcl.npy[:, 3:6]`). See
  `mm25DGS_v5.train_gaussian._normals_to_quaternions`.
- Raw materials: `inverse_reparameterize_torch(ITU_CONCRETE)` broadcast
  to all N points. All points start identical.
  ```python
  from mm25DGS_v5.train_gaussian import ITU_CONCRETE
  # ITU_CONCRETE = np.array([5.31, 0.0326, 5e-5, 5e-3, 0.5, 0.15])
  # meaning: [eps_real, eps_imag, sigma_h, l_c, tau, thickness]
  ```

### 1.3 Converting raw → physical materials

The trainer optimises in **raw** space (unconstrained). Physical values
come from `reparameterize_torch`:
```python
from mm25DGS_v5.rasterizer import reparameterize_torch
physics = reparameterize_torch(raw)   # (N, 6)
# physics[:, 0] = eps_real    in [1.0, ∞)   via softplus (1 + softplus(raw[0]))
# physics[:, 1] = eps_imag    in [~1e-3, ~9e6] via exp(clamp(raw[1], -7, 16))
# physics[:, 2] = sigma_h     in [~1e-7, ~1e-3] via exp(clamp(raw[2], -16, -7))
# physics[:, 3] = l_c         in [~45e-6, ~7.39] via exp(clamp(raw[3], -10, 2))
# physics[:, 4] = tau         in [0.05, 0.95]   via 0.05 + 0.9·sigmoid(raw[4])
# physics[:, 5] = thickness   in [~1e-3, ~7.39] via exp(clamp(raw[5], -7, 2))
```

### 1.4 Converting rotations → surface normals

Quaternion `[w, x, y, z]` → rotation matrix → third column (local +z
rotated into world frame). The exact code used by the trainer:
```python
import torch.nn.functional as F
def quat_to_normal(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1)
```

To recover the **init normal** for each point (for drift analysis):
load `pcl.npy[:, 3:6]`, FOV-cull + visibility-cull + FPS with the same
seed (42). The easiest way is to re-run `init_visible_weighted` on the
scene and capture `model.rotations` before any optimiser step; then
`quat_to_normal(model.rotations)` gives init normals. Alternatively
the quaternion formula above applied to the raw init rotation gives
the same result since quaternions are init-constructed from pcl
normals directly — you can verify with an assertion.

### 1.5 Pose + GT loading (needed for Fisher / test-sensitivity analysis)

The test pose and test GT are reproducible from the aligned configs:

```python
# Pose: interpolate between test_frame-1 and test_frame+1 pass-2 configs
from mm25DGS_v5.train_chirp_loop_nvs import build_per_loop_poses
cfg_A = f'data/alignment_data/{scene}/cascade/cascaded_frame_{F-1}_aligned_pass2.json'
cfg_B = f'data/alignment_data/{scene}/cascade/cascaded_frame_{F+1}_aligned_pass2.json'
poses, _ = build_per_loop_poses(cfg_A, cfg_B, n_loops=16)
test_pose = poses[0]   # test loop is always 0

# GT: loop 0 of the test frame's ADC .npy
import numpy as np, torch
from mmir.data.ra_utils import adc_to_ra_complex
arr = np.load(f'data/{scene}/radar/cascaded_frame_{F}.npy')
ri = np.stack([arr[0].real, arr[0].imag], axis=-1).transpose(1, 0, 2, 3).astype(np.float32)
gt_adc = torch.from_numpy(ri).cuda()
gt_ra_complex = adc_to_ra_complex(gt_adc)   # (127, 256) complex
gt_ra_mag = gt_ra_complex.abs()             # (127, 256) float
```

### 1.6 Rendering at an arbitrary pose / parameter set

```python
from mm25DGS_v5.rasterizer_factorized import render_factorized
from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch
from mm25DGS_v5.train_gaussian import (
    render_gaussians, range_profile_to_ra_mag, polar_to_cart_torch,
    build_polar_to_cart_grid, cart_corr_torch, cull_gaussians)
# See train_frame_nvs._render_and_cart_corr for the full recipe.
```

### 1.7 Pass-2 alignment configs + per-chirp configs

Per-frame pass-2 configs (one per frame, anchors 437, 439 for
seq_1_frame_438 etc.):
```
data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json
```

Per-chirp pass-2 configs (one per frame × chirp, 144 per scene):
```
data/alignment_data/<scene>/cascade/per_chirp/cascaded_frame_<F>_chirp<CC>_aligned.json
```

Per-chirp files exist for both target scenes (produced 2026-04-18). If
a follow-up experiment switches from the interpolated pose (current
trainer default) to these per-chirp configs for the all-chirp variant,
the expected improvement is independent of the regularisation work.

---

## 2. Analyses to run (in order)

Each analysis produces figures + tabular summary + written findings.
Save under `md/frame_nvs_analysis/` (create that dir). Use the
`mmir` conda env (`/home/adnan/.conda/envs/mmir/bin/python`).

### Analysis 0 (sanity): reproduce every variant's `final_test_cc`

Before any interpretation, load each of the 8 `best_model.pt` files,
re-render at the saved test pose, compute `cart_corr` vs the held-out
GT, and **verify it matches `results.json['final_test_cc']`** to 3
decimals. Any mismatch means we're analysing corrupted state — abort
and investigate. The starter code in §8 includes this step as
`verify_reproducibility()`.

Also useful: capture the rendered cart RA at the test pose for every
variant and save it (`.npy`) next to the state dict. Later analyses
(spatial error attribution, ablations) reuse these renders.

### Analysis A: Per-variant parameter distributions

For each `(scene, variant)` pair:

1. Load `raw_materials` and `rotations` from `best_model.pt`.
2. Compute `physics = reparameterize_torch(raw_materials)`.
3. Compute `normals = quat_to_normal(rotations)`.
4. For each of the 6 material columns and the 3 normal components,
   report:
   - `min, max, mean, median, std, IQR`
   - 1D histogram (log x-axis where the param spans many decades:
     `eps_imag`, `sigma_h`, `l_c`, `thickness`)

Expected findings:
- If **UB has tighter distributions** (lower std, smaller IQR) than HO
  → optimiser is finding a focused solution; HO is diffuse. Supports
  H1 (drift penalty).
- If **UB and HO have similar distributions but different modes**
  → HO converges to a wrong basin; supports H5 (Fisher-weighted),
  not H1.
- Compare **first-chirp vs full-chirp** of the same mode: is
  first-chirp's distribution narrower? (Our finding was first-chirp
  slightly beats full-chirp on both HO and UB; why? The distributions
  might tell us.)

### Analysis B: Per-point parameter drift from init

For each `(scene, variant)`:

1. Compute `drift_raw[i, c] = raw_materials[i, c] - init_raw[c]`
   (raw units — signed).
2. Compute `drift_material_L2[i] = ||drift_raw[i, :]||₂` (magnitude
   across all 6 raw params).
3. Compute `drift_normal_deg[i] = arccos(trained_normal[i] · init_normal[i])`
   in degrees (unsigned).

Report per variant:
- Histogram of `drift_material_L2` and `drift_normal_deg`.
- What fraction of points drifted > 1.0 raw unit in materials? >1°
  in normals? >5°? >15°?
- 3D scatter (matplotlib 3D or a 2D top-down projection) of the
  point cloud coloured by drift. Do drifting points cluster (spatially
  coherent movement) or spread uniformly?

Cross-variant findings to extract:
- **UB − HO drift difference**: does UB drift more (because it can
  exploit test info) or less (because it's in a sharp basin)?
- **Same-mode consistency**: is drift pattern in HO_128 vs HO_8
  spatially aligned? Same question for UB_144 vs UB_9.

### Analysis C: Material divergence between matched variants

Pairwise, per-point, for each scene:
1. `delta_raw_UB_HO[i, c] = raw_UB_144[i, c] - raw_HO_128[i, c]`
2. `delta_physics_UB_HO[i, c] = physics_UB[i, c] - physics_HO[i, c]`
3. Same for UB_9 − HO_8 (first-chirp mode), and UB_144 − UB_9 (to
   isolate the "adding 15 noisier chirps" effect).

Report:
- For each material column c: histogram of `delta_raw_UB_HO[:, c]`.
- Top-1% divergence points: where are they (spatially)? Are they the
  same in the two scenes (e.g., "points near the ground", "points
  near the radar", "points far from radar")?
- Which column has the largest divergence? (Likely `eps_imag` and
  `sigma_h` based on prior v4 observations, but verify.)

Findings drive H1 (which params to penalise) and H6 (clustering).

### Analysis D: Spatial smoothness of the material field

For each `(scene, variant)`:
1. Build a k-d tree over `positions`; find K=10 nearest neighbours of
   every point.
2. For each point i, compute `neighbour_material_std[i, c] =
   std(raw_materials[neighbours, c])`.
3. Report: distribution of per-column neighbour-std across all points.

If UB has **lower neighbour-std** than HO, the optimiser converged to
a spatially coherent material field in UB that HO didn't recover.
This supports H2 (spatial TV penalty on materials).

Stretch: compute neighbour-std vs neighbour-distance to see whether
smoothness scales with distance (i.e., is it a local-radius property
or something else). If UB is smoother only at small radii, the
regulariser bandwidth matters.

### Analysis E: Normal field geometry

For each `(scene, variant)`:
1. Compute `drift_normal_deg` per Analysis B.
2. Compute neighbour consistency: for each point, the mean angle
   between its trained normal and its K=10 nearest neighbours'
   trained normals. Compare to the same quantity for init normals.

If UB has **tighter neighbour-consistency** than HO → H4 (spatial TV
on normals) is supported. If both UB and HO preserve the init
consistency well, normals aren't where the action is; skip H3/H4.

### Analysis F: Training trajectories

For each `(scene, variant)`:
- Plot `history['mean_train_cc']` vs iter.
- Overlay init / final / best-iter markers.
- Plot `loss` vs iter.

Cross-variant: do the HO variants plateau earlier (suggesting they
saturate on the train-set fit but can't extrapolate) while UB
continues climbing? If so, the HO optimiser needs a different
objective, not just more iters — regularisation is the right lever.

### Analysis G: Parameter sensitivity via gradient (cheap Fisher proxy)

For each `(scene, variant)`, at the converged model:
1. Render the test pose with the converged parameters → `cart_corr_base`.
2. Use PyTorch autograd on a **differentiable** cart_corr surrogate
   (either the exact `cart_corr_torch` used in training or a
   Pearson-correlation proxy) to get
   `∂cart_corr / ∂raw_materials` → `(N, 6)` tensor and
   `∂cart_corr / ∂rotations` → `(N, 4)` tensor in one backward pass.
3. Per-column mean `|∂cart_corr / ∂raw_c|` across all N points tells
   which material column dominates. Across all 4 variants × 2 scenes,
   the top-2 columns should be stable — regularise those first (H5).
4. **Per-point** Fisher: `fisher[i] = Σ_c (∂cart_corr / ∂raw_materials[i, c])²`.
   Rank points by fisher[i]. The top 1–5% are the points that actually
   matter for the test render. This is the key signal for H7 (activity
   mask / selective freezing) and for prioritising which points need
   correct materials.
5. Cross-reference per-point Fisher with **spatial error** (Analysis H):
   high-Fisher points in high-error cells are the primary regularisation
   targets.

Important caveat: this is test-pose Fisher. At HO training time we
don't have the test RA; we can only compute train-set Fisher on the 128
(or 8) training RAs. Analyse both: if train-set Fisher rankings match
test-set Fisher rankings, a train-only Fisher-weighted regulariser will
transfer. If they diverge, we need a proxy.

### Analysis H: Spatial error attribution (test RA → contributing points)

This is the most direct route from "bad HO test cc" to "which points
to regularise." For each `(scene, HO-variant)`:

1. Render the test pose with the converged HO parameters → `rend_cart`.
2. Compute per-cell error: `err_cart = |rend_cart − gt_cart_norm|`
   (both normalised to [0, 1]). Note: sign matters — over-estimation
   vs under-estimation points to different material fixes.
3. Identify the top-K error cells (say top 5% by `err_cart` magnitude).
4. For each top-error cell, **backpropagate** via autograd to determine
   which 3D points contribute most: pick one cell, compute
   `d(rend_cart[az_idx, range_idx]) / d(raw_materials)` via `.backward()`.
   The points with large `|grad|` at that cell are the "contributors".
5. Overlay contributor-point heatmap on the 3D scene. Repeat for the
   UB render: are UB's contributors different?

Expected outcome: in HO, a small region of the scene is systematically
mis-materialed, producing an error pattern in a few (az, range) cells.
Fixing those contributor points (via regularisation pulling them toward
UB's materials at those points, or toward init) should recover the cc.

### Analysis I: Swap-ablation — materials vs normals

For each scene, compute these 4 hybrid models at the test pose:
1. UB_144 materials + UB_144 normals → baseline UB cc
2. HO_128 materials + HO_128 normals → baseline HO cc
3. **UB materials + HO normals**
4. **HO materials + UB normals**

If (3) ≈ UB, the fault is in HO's normals; regularise normals (H3, H4).
If (4) ≈ UB, the fault is in HO's materials; regularise materials (H1, H2, H5).
If both (3) and (4) underperform UB, the fault is coupled — need both.
Very likely the latter, but the magnitudes tell us the mixing ratio.

Repeat for UB_9 vs HO_8 (first-chirp pair) to check the finding is
robust across the noise-regime.

### Analysis J: Intensity-prior validation (for H8)

For each `(scene, variant)`:
1. Load `pcl.npy[:, 6]` (LiDAR intensity, normalised per-scene).
2. Compute per-point variance of raw_materials within bins of intensity
   (e.g., 10 equal-count bins). If UB has lower intra-bin variance
   than HO, intensity-similar points have more similar materials in UB
   — the intensity prior (H8) is validated.
3. Also compute mutual information between intensity and each raw
   material column, per variant. Cross-check: intensity-material MI
   higher in UB than HO?

If yes, intensity-weighted spatial smoothness (H8) is supported.

### Analysis K: UB_144 vs UB_9 — the "noise-averaging penalty"

Both UB variants include the test RA in training. UB_9 finishes at
higher cc than UB_144 on both scenes. This is not an NVS phenomenon
(no held-out frame) — it's an optimisation phenomenon. Analyse:

1. Per-point parameter diff UB_144 − UB_9 (raw and physics).
2. Is there a pattern? (e.g., UB_144 is more "averaged", lower
   peak-to-trough variation; UB_9 is sharper but potentially more
   noise-fit.)
3. The difference might explain why first-chirp beats full-chirp on HO
   too — same mechanism.

If confirmed, the implication for the HO regulariser is: **avoid
over-averaging**. Regularisers should preserve sharpness (peak values)
while constraining off-task drift. Plain L2 may actually hurt by
smoothing both. A concentration-preserving penalty (e.g., sparsity
or L1 on drift) might be better than L2 — flag this as a variant to try.

---

## 3. Regularisation hypotheses

Each hypothesis is *conditional*: implement only if the analyses above
support it with data from both scenes.

| H | regulariser | validated by | raw form |
|---|---|---|---|
| **H1** | L2 on raw-material drift from init | Analysis A (tighter UB dist) + B (lower UB drift) | `λ₁ · ||raw_materials − init_raw||²` |
| **H2** | Spatial TV on raw_materials over k-NN | Analysis D (UB has lower neighbour-std) | `λ₂ · Σ_i Σ_{j∈NN(i)} ||raw[i] − raw[j]||²` |
| **H3** | L2 on normal drift from init | Analysis B + E (UB preserves init normals better) | `λ₃ · Σ_i (1 − trained_n[i] · init_n[i])` |
| **H4** | Spatial TV on normals over k-NN | Analysis E (UB tighter neighbour-normal consistency) | `λ₄ · Σ_i Σ_{j∈NN(i)} (1 − trained_n[i] · trained_n[j])` |
| **H5** | Fisher-weighted L2 on raw_materials | Analysis G (a subset of params dominate test cc) | `λ₅ · Σ_c fisher[c] · (raw[:, c] − init[c])²` |
| **H6** | Cluster prior on materials (K-means on UB → centroids) | Analysis A + C (UB materials cluster in K modes) | `λ₆ · Σ_i ||raw[i] − nearest_centroid(raw[i])||²` — centroids computed from the **scene's own UB_144 run** |
| **H7** | Activity mask / freeze low-Fisher points | Analysis G per-point Fisher | set `requires_grad=False` on bottom-X% Fisher-score points |
| **H8** | Intensity prior | LiDAR `pcl.npy[:, 6]` (intensity) | `λ₈ · Σ_{(i,j)∈NN} w_ij · ||raw[i] − raw[j]||²`, `w_ij = exp(−|I_i − I_j|²)` — neighbours with similar LiDAR intensity should share materials |

H6 is the most "obvious" (clusters from UB give HO direction toward
the right basin) but also leaks from UB; it's acceptable as an upper
bound on "how much regularisation could help", not as a practical
regulariser for deployment.

H1, H2, H5, H8 are deployable (no UB leakage). Prioritise these.

### Priority ranking

Starting points, in order of expected value:

1. **H5 (Fisher-weighted drift from init)** — most principled. If
   Analysis G shows a small number of columns / points dominate the
   test cc, constraining only those is the minimal-interference
   regulariser. Requires per-point or per-column Fisher.
2. **H7 (activity mask from per-point Fisher)** — cheap and
   complementary to H5. Freezing low-Fisher points hurts nothing if
   they genuinely don't affect the test cc.
3. **H1 (L2 drift from init)** — simplest baseline. Use as a sanity
   check: if a plain L2 on all parameters doesn't help at all, the
   problem is not "too much drift" but "wrong direction of drift" and
   H5/H6/H8 are needed.
4. **H2 (spatial TV on materials)** — if Analysis D shows UB has
   smoother material fields, H2 is the route. Needs a k-d tree per
   scene; ~1 ms/iter overhead.
5. **H8 (intensity-weighted spatial TV)** — conditional on Analysis J.
   Similar cost to H2 but more targeted.
6. **H6 (cluster prior from UB centroids)** — *only as an
   upper-bound experiment*, never for deployment. It leaks test info
   via UB. Tells us the ceiling of what a "material prior" can
   achieve; if this hits 0.70 easily, the problem is solvable and we
   just need a non-leaky version of it.
7. **H3, H4 (normal drift + spatial TV)** — run only if Analysis I
   shows normals are part of the failure. Lower prior because
   per-v4-work, materials dominate the test cc.

### Regulariser scheduling

Plain constant λ may underperform even for the right regulariser.
Always try two schedules:
1. **Constant** — λ fixed across all 500 iters.
2. **Warm-start** — λ starts at λ/10 for the first 100 iters (let the
   optimiser find the basin), then ramps linearly to full λ over iters
   100-200, holds through iter 500. Typically helps by +0.02-0.05 cc.

If a regulariser works on constant schedule, warm-start either helps
more or doesn't matter. If it doesn't work on constant, warm-start can
rescue it in some cases (regulariser was over-constraining from iter 0).

---

## 4. Implementation plan (post-analysis)

After Analyses A–G identify the 2 or 3 most-supported hypotheses:

1. Extend `mm25DGS_v5/train_frame_nvs.py` with a `reg_terms` dict
   argument + CLI flags (`--reg_drift_lambda`, `--reg_tv_lambda`,
   `--reg_fisher_lambda` etc.). Add the chosen term(s) to the
   per-iter backward pass.
2. **Calibrate λ on one scene** (pick seq_1_frame_438 — cleanest
   trajectory): sweep λ ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1} for each
   chosen regulariser. Run the HO variant (128 RA). Record test cc.
3. Pick the λ that maximises HO test cc on seq_1_frame_438.
4. **Validate on seq_2_frame_105** with the same λ — no re-tuning.
   If seq_2_frame_105 test cc also improves, the regulariser is
   robust. If it degrades, back off.
5. If single-regulariser λ-tuning doesn't hit 0.70, try a pair (e.g.
   H1 + H2) with a coarse 3×3 grid.
6. If the full-chirp variant's HO cc passes 0.70 with the new
   regulariser, also test the first-chirp HO variant (8 RA). Expect
   a similar boost since the regulariser targets parameter-space
   (not training-set) structure.

---

## 5. Success criteria

- **Hard**: HO cart_corr ≥ 0.70 on both `seq_1_frame_438` and
  `seq_2_frame_105`, centre-frame loop 0, with the SAME regulariser
  hyperparameters.
- **Hard**: regulariser must be motivated by empirical analysis 0–K
  (cite at least one numerical finding per included term).
- **Hard**: the motivating insight must appear in **both scenes** (no
  scene-specific artefacts).
- **Soft**: gap (UB_144 − HO_128) shrinks from current 0.4 to ≤ 0.2 cc.
- **Soft**: train mean cc at convergence should not regress by more
  than 0.05 (regularisation shouldn't destroy training-set fit).

### Fallback if no single-term regulariser hits 0.70

Escalate in this order:

1. **Try pairs** (e.g. H5 + H7, H1 + H2) — grid 3×3 over the two λ's.
2. **Use per-chirp alignment configs** (see
   `md/per_chirp_alignment_stage3_plan.md`) — re-run the full-chirp
   variant with pose-refined chirps to test whether pose accuracy was
   part of the gap. Combine with the best single-term reg.
3. **Try a different material parameterisation** — the raw-space
   parameters have skewed scales (eps_real ~ O(1), eps_imag ~ O(0.03)
   physically but raw[1] ~ O(−3) after log). Per-column LR scaling or
   whitening might change the optimisation landscape enough to close
   the gap.
4. **Reduce problem scope** — if 0.70 is unreachable with our 90k
   point set, test whether a subset (e.g. only the high-Fisher 10k
   points, with low-Fisher frozen) learns a better HO. If yes, the
   full model is over-parameterised for the data — propose a
   structured simplification before regularisation.
5. **Report honestly** — if 0.70 is infeasible within the constraints,
   quantify what IS reachable (e.g., 0.62) and the bound on further
   improvement given the measured signal in the test RA.

---

## 6. Constraints (from prior-session memory)

These constraints were set as project guardrails; honour them.

- **No iterative warm-start retraining** — don't chain runs. Each
  experiment starts from ITU concrete + init normals.
- **Max 500 iters per run**.
- **Don't use tiny LRs as a workaround** — if generalisation is bad,
  analyse parameter differences and add a targeted regulariser.
- **NVS = interpolation only** — don't extrapolate past the
  trajectory edges.
- Default LRs: `mat_lr=0.01`, `rot_lr=5e-3`. Don't change them
  without a reason grounded in analysis.
- `pcl.npy` has 7 columns including intensity at col 6 — available
  for H8 regulariser.
- Output-file conventions (keep them):
  - Training outputs to
    `mm25DGS_v5/output_frame_nvs/<scene>_<tag>/{results.json, best_model.pt, history.npz}`
  - Plots / markdown to `md/frame_nvs_analysis/`.
- Use `/home/adnan/.conda/envs/mmir/bin/python` — the `mmir` env.
- Two RTX 4090s; parallelise via `CUDA_VISIBLE_DEVICES` across GPUs
  when running two scenes at once.

---

## 7. What to deliver back

1. `md/frame_nvs_analysis/findings.md` with:
   - Figures for each analysis (saved as PNGs, linked in-line).
   - Per-scene numerical tables.
   - A short written conclusion per analysis.
   - Explicit go/no-go on each hypothesis H1–H8.
2. The chosen regulariser(s) implemented in
   `mm25DGS_v5/train_frame_nvs.py` behind CLI flags.
3. Two training runs per scene with the new regulariser:
   - Re-run HO (128 RA) with the new loss.
   - Re-run HO (8 RA, first-chirp) with the same loss (same λ).
4. Updated `md/frame_nvs.md` with a new column showing the new HO
   test cc (both scenes).
5. One paragraph of commentary at the end of `md/frame_nvs.md`
   explaining which regularisers were applied and why — grounded in
   the analysis findings.

---

## 8. Starter code (copy-pasteable)

```python
# load_all_variants.py — loader for Analysis A–G
import json, os, torch, numpy as np
from mm25DGS_v5.rasterizer import reparameterize_torch

SCENES = ['seq_1_frame_438', 'seq_2_frame_105']
VARIANTS = {
    'HO_128':  lambda F: f'train8frames_test{F}_loop0_pass2',
    'HO_8':    lambda F: f'train8frames_1loops_test{F}_loop0_pass2',
    'UB_144':  lambda F: f'train9frames_16loops_test{F}_loop0_ub_pass2',
    'UB_9':    lambda F: f'train9frames_1loops_test{F}_loop0_ub_pass2',
}
SCENE_FRAME = {'seq_1_frame_438': 438, 'seq_2_frame_105': 105}
ROOT = 'mm25DGS_v5/output_frame_nvs'

def quat_to_normal(q):
    import torch.nn.functional as F
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1)

all_data = {}
for scene in SCENES:
    F = SCENE_FRAME[scene]
    all_data[scene] = {}
    for vname, tag_fn in VARIANTS.items():
        d = os.path.join(ROOT, f'{scene}_{tag_fn(F)}')
        state = torch.load(os.path.join(d, 'best_model.pt'),
                           weights_only=False, map_location='cpu')
        meta = json.load(open(os.path.join(d, 'results.json')))
        all_data[scene][vname] = {
            'positions': state['positions'].numpy(),
            'rotations': state['rotations'].numpy(),
            'raw':       state['raw_materials'].numpy(),
            'physics':   reparameterize_torch(state['raw_materials']).numpy(),
            'normals':   quat_to_normal(state['rotations']).numpy(),
            'meta':      meta,
        }
# all_data[scene][variant] now has positions, rotations, raw, physics,
# normals, meta — ready for per-point + per-variant analyses.
```

```python
# init_state.py — recover the init state (what every run started from)
import numpy as np
from mm25DGS_v5.train_gaussian import (
    ITU_CONCRETE, PointPrimitives, init_visible_weighted, DEVICE,
    _normals_to_quaternions,
)
from mm25DGS_v5.rasterizer import Rasterizer, inverse_reparameterize_torch
from mm25DGS_v5.load_pretrained import load_trained_config
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

def make_init_model(scene):
    config = load_trained_config(scene)
    rast = Rasterizer(
        config_file=config.config_file, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file, device=DEVICE)
    model = init_visible_weighted(scene, rast, target_n=90000)
    init_raw    = model.raw_materials.detach().cpu().numpy()  # (N, 6)
    init_rot    = model.rotations.detach().cpu().numpy()      # (N, 4)
    init_normal = quat_to_normal(torch.from_numpy(init_rot)).numpy()
    return {
        'positions':   model.positions.detach().cpu().numpy(),
        'init_raw':    init_raw,
        'init_rot':    init_rot,
        'init_normal': init_normal,
    }
```

```python
# render_at_test_pose.py — render a saved variant at its test pose, return cart_corr
import torch, numpy as np, json, os
import mitsuba as mi; mi.set_variant('cuda_ad_rgb')
from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch
from mm25DGS_v5.train_gaussian import (
    DEVICE, render_gaussians, range_profile_to_ra_mag,
    build_polar_to_cart_grid, polar_to_cart_torch, cart_corr_torch,
    cull_gaussians, PointPrimitives,
)
from mm25DGS_v5.train_chirp_loop_nvs import build_per_loop_poses, apply_pose
from mm25DGS_v5.load_pretrained import load_trained_config
from mmir.data.ra_utils import adc_to_ra_complex
from mmir.data.io_utils import compute_range_res_from_cfg

def render_test_cc(scene, test_frame, state_dict, target_n=90000):
    """Render the test pose (loop 0 of test_frame, interpolated from
    pass-2 aligned configs of test_frame-1 and test_frame+1) with the
    given state dict. Returns (cart_corr, rend_cart_norm, gt_cart_norm)."""
    align_dir = f'/home/adnan/Desktop/mm3DGS/data/alignment_data/{scene}/cascade'
    cfg_A = f'{align_dir}/cascaded_frame_{test_frame-1}_aligned_pass2.json'
    cfg_B = f'{align_dir}/cascaded_frame_{test_frame+1}_aligned_pass2.json'
    poses, _ = build_per_loop_poses(cfg_A, cfg_B, n_loops=16, device=DEVICE)
    test_pose = poses[0]

    config = load_trained_config(scene)
    rast = Rasterizer(
        config_file=cfg_A, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file, device=DEVICE)

    # Build a model and load the trained state
    from mm25DGS_v5.train_gaussian import init_visible_weighted
    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    with torch.no_grad():
        model.positions.copy_(state_dict['positions'].to(DEVICE))
        model.rotations.copy_(state_dict['rotations'].to(DEVICE))
        model.raw_materials.copy_(state_dict['raw_materials'].to(DEVICE))

    apply_pose(rast, test_pose)
    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    range_res = compute_range_res_from_cfg(cfg_A)
    sample_grid = build_polar_to_cart_grid(127, 256, range_res, 400, DEVICE)

    # Load test GT
    adc_npy = f'/home/adnan/Desktop/mm3DGS/data/{scene}/radar/cascaded_frame_{test_frame}.npy'
    arr = np.load(adc_npy)
    ri = np.stack([arr[0].real, arr[0].imag], axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)
    gt_adc = torch.from_numpy(ri).to(DEVICE)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc)
        gt_cart = polar_to_cart_torch(torch.abs(ra_c).float(), sample_grid)
        mn, mx = gt_cart.min(), gt_cart.max()
        gt_cart_norm = (gt_cart - mn) / (mx - mn).clamp(min=1e-30)

    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas, active_mask=active_mask)
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        rend_cart = polar_to_cart_torch(ra_polar, sample_grid)
        mn, mx = rend_cart.min(), rend_cart.max()
        rend_cart_norm = (rend_cart - mn) / (mx - mn).clamp(min=1e-30)
        cc = cart_corr_torch(rend_cart, gt_cart_norm).item()
    return cc, rend_cart_norm.cpu().numpy(), gt_cart_norm.cpu().numpy()


def verify_reproducibility():
    """Analysis 0: load every variant, re-render, verify final_test_cc matches."""
    rows = []
    for scene in SCENES:
        F = SCENE_FRAME[scene]
        for vname, tag_fn in VARIANTS.items():
            d = os.path.join(ROOT, f'{scene}_{tag_fn(F)}')
            state = torch.load(os.path.join(d, 'best_model.pt'),
                               weights_only=False, map_location='cpu')
            meta = json.load(open(os.path.join(d, 'results.json')))
            cc, _, _ = render_test_cc(scene, F, state)
            reported = meta['final_test_cc']
            rows.append((scene, vname, reported, cc, abs(cc - reported)))
            print(f'{scene:<22} {vname:<10} reported={reported:.4f} '
                  f'measured={cc:.4f} delta={abs(cc - reported):.4f}')
    return rows
```

```python
# fisher_like.py — differentiable test-cc gradient w.r.t. raw_materials + rotations
import torch

def compute_fisher_at_test(scene, test_frame, state_dict, target_n=90000):
    """Return per-point per-column gradient of cart_corr at the test pose.

    grad_raw:      (N, 6)  — d cart_corr / d raw_materials[i, c]
    grad_rot:      (N, 4)  — d cart_corr / d rotations[i, k]
    grad_normal_angle: (N,) — d cart_corr / d normal_rotation_angle[i]
                              (collapsed to one scalar per point via the
                              magnitude of grad_rot projected onto normal-change)

    Use these to rank per-column and per-point sensitivity.
    """
    # Build model + rast as in render_test_cc, but with requires_grad=True on
    # raw_materials and rotations. Enable autograd through the render. Then
    # loss = −cart_corr (we're maximising); loss.backward() populates .grad.
    # Pull .grad.detach() for the Fisher-like tensors.
    # (~400 lines combined, reuses render_test_cc's skeleton)
    raise NotImplementedError(
        'Fill in following render_test_cc; enable grads on raw_materials + rotations.')
```

The `fisher_like.py` skeleton is worth keeping terse — the new session
has everything it needs from `render_test_cc` to fill in the body in
~30 lines. The critical steps: (i) keep `raw_materials` and `rotations`
with `requires_grad=True`, (ii) do the render in autograd mode (no
`torch.no_grad()`), (iii) compute `cc = cart_corr_torch(...)` as the
scalar to `backward()` on, (iv) read `state_dict['raw_materials'].grad`.

---

## 9. Notes on why this is tractable

- The FPS seed is fixed (42), so every run's 90k-point set is identical
  — per-point comparisons are well-defined.
- UB_144 already produces cc 0.91 (seq_1) / 0.82 (seq_2) — there IS a
  converged parameter set in the hypothesis class that explains the
  held-out RA. The question is how to guide the HO optimiser to it.
- Most of the "damage" in HO is that raw_materials drift too much in
  too-diffuse a direction (our suspicion from v4 work; verify). If
  true, H1 alone might close half the gap.
- The ~0.01 cc gap between full-chirp and first-chirp HO tells us the
  additional 15 chirp-loops/frame aren't helpful — probably the
  cross-chirp noise/pose variance cancels out the information gain.
  The regulariser work targets this by controlling what the optimiser
  does with parameter updates, not how many RAs supervise them.

---

Good luck. Produce a findings report that a reviewer could read
end-to-end and say "yes, those regulariser choices are earned."
