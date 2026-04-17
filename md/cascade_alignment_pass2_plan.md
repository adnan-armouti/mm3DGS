# Cascade alignment — pass 2 plan

## Context

The first-pass cascade alignment runs two methods per frame, independently:

- **Renderer 2-DOF** (`mmir/preprocessing/alignment/cascaded_renderer.py`):
  grid search over `(range_m, azimuth_deg)` with a Mitsuba MC renderer
  inside `render_and_evaluate`, then `scipy.optimize.minimize`
  refinement. Only range + azimuth deltas are parametrised.
- **Lidar 4-DOF** (`mmir/preprocessing/alignment/cascaded_lidar.py`,
  GPU via `cascaded_lidar_gpu.py`): voxelize LiDAR into an RAE map,
  correlate against a radar GT RAE, optimize 4 DOF of the radar pose
  (range + azimuth + elevation + roll-ish, details in the file).

The alignment orchestrator (`cascaded_alignment.py:select_best_alignment`)
runs **both** methods per frame and writes whichever has higher cart_corr
to the winner config, logging the loser's score in
`cascaded_frame_<F>_alignment_log.json`.

The per-frame alignment logs live in
`data/alignment_data/<scene>/cascade/cascaded_frame_<F>_alignment_log.json`;
the winning configs are written to both that dir and to
`data/<scene>/configs/cascaded_frame_<F>_aligned.json`.

Because no smoothness / trajectory prior is applied to either method,
and the two methods are evaluated per-frame independently, the objective
surfaces' local optima cause systematic failures — most visibly
**boresight-z flipping** when the radar is moving near-horizontally
and the alignment objective has a secondary maximum above/below the
scene. Five of seven "reference" frames (the training-target frame for
each scene, which also receives special alignment treatment) have a
broken z.

### 2026-04-17 outlier catalog (from user inspection)

⚠Z! denotes a boresight whose z-component is an outlier relative to its
local (±4-frame) neighbourhood. REF is the primary trained frame for
each scene.

| scene | broken frames (⚠Z!) | REF broken? |
| --- | --- | --- |
| seq_0_frame_135 | 135 (REF, z=-0.486) | yes |
| seq_0_frame_390 | 390 (REF, z=+0.382), 393, 394 | yes |
| seq_1_frame_185 | 181, 182, 185 (REF, z=-0.209), 187, 188 | yes |
| seq_1_frame_438 | — | no |
| seq_2_frame_105 | 101–109 (ALL, z≈-0.22) | — (see note) |
| seq_2_frame_160 | — | no |
| seq_2_frame_300 | 300 (REF, z=+0.333), 302 | yes |
| seq_0_frame_451 | unknown — not yet inspected | unknown |
| seq_1_frame_277 | unknown — not yet inspected | unknown |

Note on `seq_2_frame_105`: every frame in the window has z ≈ -0.22.
Either the scene genuinely tilts the rig ~13° downward (possible) or
the alignment objective has a shared wrong optimum across all 9 frames
(also possible; a scene-constant rig tilt is a valid alignment if it is
stable and gives high cart_corr). Decide after trajectory fitting — if
the trajectory is smooth *in itself* (no per-frame z-jumps), leave it
alone even if z is unusual.

Two unlisted scenes (`seq_0_frame_451`, `seq_1_frame_277`) must also
be audited before pass 2 runs.

## Why pass 1 fails

1. **No trajectory prior**: both objective surfaces are multi-modal;
   without a prior each frame can settle into a different mode. A pose
   that gives the best cart_corr / LiDAR correlation for a single
   frame can be incompatible with its neighbours' poses.
2. **Renderer method is 2-DOF only**: it cannot reach the boresight
   elevation / z-component directly. Whatever z the `base_config`
   carries is what the 2-DOF method outputs — so if the base config
   itself is noisy, z stays broken.
3. **LiDAR method is 4-DOF** but its objective is the LiDAR-voxel ↔
   GT-RAE correlation, which has its own multi-modality; it also
   sometimes picks a z-flipped optimum (see `cascaded_frame_*_aligned_gpu.json`
   variants in `data/seq_0_frame_135/configs/`, some of which have
   bs.z ≈ +0.88).
4. **REF-frame special path**: reference frames are aligned first, and
   neighbour frames use the REF as a prior via `sc_trajectory_transfer`.
   A broken REF therefore poisons its neighbours in the current pass.
5. **Winner selection by cart_corr alone**: the per-frame winner is
   whichever method scored higher, not whichever is physically
   consistent with the trajectory. Frame 135 wins with cc=0.475
   despite a clearly wrong z.
6. **Renderer alignment is slow**: each MC render in `render_and_evaluate`
   takes seconds on Mitsuba, so the renderer grid search is too
   expensive to sweep fine range/azimuth grids or to add an explicit
   elevation DOF without blowing the compute budget. This is a
   silent driver of pass-1 bias toward the LiDAR method on noisy
   frames.

## Goal

Re-align every frame in all 9 training scenes (9 frames each, 81 total)
such that the resulting per-frame poses:

- lie on a trajectory whose per-component second derivatives are small
  (no discontinuous jumps between consecutive frames);
- retain or improve per-frame cart_corr vs the first-pass winner
  (re-alignment should not regress alignment quality);
- eliminate every ⚠Z! outlier listed above, and any auto-detected
  outlier found in the two unlisted scenes.

A clean trajectory is a prerequisite for honest per-chirp-loop NVS:
the 73 mm "apparent motion" we currently synthesise across 7.87 ms
comes entirely from pass-1 noise between neighbouring frames, not from
real motion. Fixing that artefact is Deliverable 1.

## Proposed pipeline

Two stages. Stage A is prior construction (no re-alignment); Stage B is
constrained re-alignment of the subset of frames that Stage A flagged.
Both pass-1 methods (renderer + LiDAR) participate in Stage B; winner
selection runs the same as pass 1, with an added trajectory-consistency
gate.

### Stage A — trajectory fit + outlier detection (offline, fast)

**A1. Per-component robust fit.**

For each scene's 9-frame trajectory, extract 6 time-series over the
frame indices `f_i`:

- position centre: `p_x(f), p_y(f), p_z(f)` — mean of tx_array + rx_array positions
- boresight: `b_x(f), b_y(f), b_z(f)` — boresight of `tx_array[0]`

Fit each component independently using a robust estimator:

- **Default**: LOWESS (locally weighted scatterplot smoothing) with
  bandwidth ~4 frames and Huber residual weighting — handles sparse
  series better than polynomial fits.
- **Fallback**: Theil–Sen for short series (9 points is borderline
  for LOWESS).
- Re-normalise the fitted boresight to unit length after per-component
  smoothing (components are interpolated independently, so the norm
  drifts).

**A2. Flag outliers.**

For each frame, compute per-component residuals `r_c(f) = v_c(f) - v̂_c(f)`,
expressed in units of the robust residual std (MAD × 1.4826).

- Hard outlier: any component with `|r_c| > 3` MAD — definite ⚠ flag.
- Soft outlier: `max(|r_c|)` in `[2, 3]` MAD — audit manually.
- Boresight norm sanity: any raw boresight with `||b|| ∉ [0.98, 1.02]`
  after the pass-1 write is a flag regardless of component residuals.

Output: `alignment_data/<scene>/cascade/pass2_triage.json` listing for
each frame `{status: 'ok'|'soft'|'hard', residuals_mad: [...], smoothed_pose: {...}}`.

**A3. Scope-limit decision.** For each scene, decide whether to
(a) re-align only the hard-flagged frames (cheapest, preserves the
good ones), (b) re-align every frame in the scene (most robust), or
(c) skip re-alignment entirely and just rewrite configs using the
Stage-A smoothed pose as the final pose (zero-cost, no second
optimisation).

Default recommendation: **(a)** for scenes with isolated outliers
(seq_0_frame_135, seq_0_frame_390, seq_1_frame_185, seq_2_frame_300),
**(c)** for scenes whose whole trajectory is smooth (seq_1_frame_438,
seq_2_frame_160), and **(a) + inspection** for seq_2_frame_105 (decide
case-by-case whether the consistent −0.22 z is real).

### Stage B — constrained re-alignment of flagged frames

Both methods from pass 1 participate. Each is extended to accept a
trajectory-prior pose and a prior weight; the orchestrator then selects
the winner the same way pass 1 did, with an added trajectory-consistency
gate.

**B1-R. Renderer method (v5-CUDA, extended to 4-DOF).**

Extend the renderer alignment driver (see "Vendoring & CUDA
acceleration" below) to accept an optional prior pose and to
parametrise the full 4 DOF that the lidar method already outputs
(range + azimuth + elevation + roll-ish, matching
`apply_4dof_to_config`):

```python
run_single_frame_renderer(
    frame_idx, base_config,
    pose_prior: Optional[dict] = None,       # {tx_pos, rx_pos, tx_bore, rx_bore}
    prior_weight: float = 0.0,               # λ in loss
    search_radius: dict = None,              # per-DOF grid bounds
    dof: int = 4,                            # 2 (pass-1 compat) or 4
    ...
)
```

The grid search is modified to:

1. Use the prior pose as the centre of the grid (not the raw base
   config).
2. Constrain the grid radius (e.g. ±0.1 m range, ±3° azimuth, ±2°
   elevation, ±2° roll) around the prior. This prevents the z-flip.
3. Add `prior_weight × ||pose − prior||²` (componentwise, with each
   DOF scaled by its unit — mm for positions, degrees for angles) to
   the negated cart_corr objective in both the grid search and the
   `scipy.optimize.minimize` refinement.

The 4-DOF grid is enabled by the CUDA swap below; on the MC renderer
this would be infeasible (hundreds of cells × seconds per render).

**B1-L. LiDAR method.**

Extend the LiDAR alignment driver with the same `pose_prior`,
`prior_weight`, `search_radius` kwargs:

```python
run_single_frame_lidar(
    frame_idx, base_config,
    pose_prior: Optional[dict] = None,
    prior_weight: float = 0.0,
    search_radius: dict = None,
    ...
)
```

Same semantics: prior-centred search, bounded radius, prior penalty
added to the LiDAR correlation objective. The GPU variant
(`cascaded_lidar_gpu.py`) is the one we extend; the CPU variant stays
as a reference / fallback.

**B2. Winner selection.**

Run both B1-R and B1-L for each flagged frame, then:

1. Keep the existing `select_best_alignment` comparison
   (renderer cart_corr vs LiDAR cart_corr). Both candidates now
   carry the prior term so "cc" is really "cc − λ·||pose−prior||²".
2. Add a trajectory-consistency gate: a candidate is rejected if
   its per-component MAD residual against the Stage-A smoothed
   trajectory exceeds 2.5 MAD (soft) even if its adjusted score
   is higher.
3. If both candidates fail the gate, fall back to the smoothed
   prior pose itself with
   `{winner: "prior_only", reason: "all_candidates_off_trajectory"}`.
4. Record both methods' scores *and* their MAD residuals in a pass-2
   alignment log parallel to pass 1's, so the gating decisions are
   auditable after the fact.

**B3. Re-run Stage A on the output.**

After Stage B writes new aligned configs, re-run the trajectory fit
and verify:

- no hard outliers remain;
- mean per-component residual decreased vs pass 1;
- mean winner cart_corr did not regress by more than 0.02 from pass 1.

Any scene failing this gate reverts to its pass-1 configs pending
investigation.

## Vendoring & CUDA acceleration

The pass-2 code lives entirely inside `mm25DGS_v5/`. The existing
`mmir/preprocessing/alignment/` tree is copied in with minimal edits
to retarget imports, and the renderer method's MC path is replaced
with v5 CUDA rendering.

### Folder layout

```
mm25DGS_v5/
  preprocessing/
    __init__.py
    alignment/
      __init__.py                     # re-exports pass-2 CLI + helpers
      cascaded_alignment.py           # vendored orchestrator (+ pass-2 subcommand)
      cascaded_renderer.py            # vendored, MC renderer path RIPPED OUT
      cascaded_renderer_cuda.py       # NEW — v5 CUDA rendering backend
      cascaded_lidar.py               # vendored, unchanged (CPU reference)
      cascaded_lidar_gpu.py           # vendored, extended with prior kwargs
      sc_trajectory_transfer.py       # vendored, unchanged
      gpu_utils/                      # vendored, unchanged (CuPy helpers)
      pass2/
        __init__.py
        trajectory_fit.py             # NEW — Stage A
        run_pass2.py                  # NEW — Stage B driver + CLI
```

Dependencies that also need to live inside (or be importable from)
`mm25DGS_v5` so the preprocessing module stands alone:

- `mmir.data.ra_utils` (`adc_to_ra_image`, `ra_polar_to_cartesian`,
  `adc_to_ra_complex`) — used by both the renderer and lidar paths.
  Leave in place; `mm25DGS_v5.preprocessing.alignment` imports from
  `mmir.data.*` directly.
- `mmir.data.io_utils.compute_range_res_from_cfg` — same treatment.
- `mmir.renderer.*` (Mitsuba-based) — **ripped out of
  `cascaded_renderer.py`** once the CUDA backend replaces it. Keep
  `mmir.renderer` intact elsewhere in the tree (other callers use it);
  it's only the alignment module that drops it.
- `mmir.preprocessing.config_utils`, `mmir.preprocessing.lidar_utils`
  — vendor only if lidar alignment imports them. Otherwise skip and
  import from `mmir.preprocessing.*`.

Policy: vendoring is code-copy, not symlink. Drift between the vendored
copy and the original `mmir/preprocessing/alignment/` is acceptable and
expected (the v5 copy receives the pass-2 extensions + CUDA swap; the
mmir copy stays pass-1 for reproducibility).

### Swapping the MC renderer for v5 CUDA

The renderer method's render is concentrated in
`cascaded_renderer.py:render_and_evaluate` (~10 LOC) and its three
call sites (`grid_search_2dof`, `refine_2dof`, and
`run_single_frame`). Pass-1 uses Mitsuba MC through a `Renderer`
object; each call does a full Monte-Carlo ray trace and is slow.

The v5 CUDA replacement works as follows:

1. **Shared CUDA renderer context per frame.**
   Build one `Rasterizer` per frame (v5's class — holds TX/RX pose,
   antenna patterns, radar FMCW params). Point cloud and per-point
   material are set once at frame-start (`init_visible_weighted` +
   ITU concrete defaults) and reused across the hundreds of pose
   candidates in the grid search. Pose is swapped in-place via the
   same `apply_pose` pattern already used in
   `mm25DGS_v5/train_chirp_loop_nvs.py`:

   ```python
   with torch.no_grad():
       rast.tx_positions.copy_(candidate_tx_pos)
       rast.rx_positions.copy_(candidate_rx_pos)
       rast.tx_boresights.copy_(candidate_tx_bore)
       rast.rx_boresights.copy_(candidate_rx_bore)
   rp_real, rp_imag = render_factorized(
       positions, normals, areas, raw_materials, rast,
       reparameterize_torch, detach_phase=True, use_cuda_kernels=True)
   ra_mag = range_profile_to_ra_mag(rp_real, rp_imag)
   ra_cart = polar_to_cart_torch(ra_mag, sample_grid)
   cc = cart_corr_torch(ra_cart, gt_cart_norm).item()
   ```

2. **`cascaded_renderer_cuda.py`** hosts the replacement:
   - `build_alignment_context(scene, frame_idx, gt_adc_path) -> Ctx`
     (one-time per frame: FPS init, GT cart precompute, sample grid)
   - `render_and_evaluate_cuda(ctx, pose_candidate) -> float`
     (called in the inner loop of grid_search / refine)
   - `apply_{2,4}dof_to_pose(prior_pose, deltas) -> pose_candidate`
     (operates on pose dicts, not on config files — avoids repeated
     JSON writes in the grid search)

3. **`cascaded_renderer.py`** (vendored) keeps the grid/refine
   control flow but:
   - `render_and_evaluate` delegates to
     `render_and_evaluate_cuda` via dependency injection,
   - `apply_2dof_to_config` / `apply_4dof_to_config` still exist,
     but only for writing the final winner config to disk (not
     inside the inner loop).
   - All `mitsuba`, `drjit`, `mmir.renderer` imports removed.

**Speed expectation.** v5 CUDA render ≈ 3 ms/call vs MC render ≈ a
few seconds. A 10×10 grid = 100 evals drops from ~5–10 min to
~0.3 s. That headroom is what pays for: (a) enabling the full 4-DOF
grid in the renderer path (~900 evals, still ≈ 2.7 s), (b) more refine
iterations, and (c) running renderer-4-DOF on every flagged frame in
a scene in under a minute.

**Correctness.** v5 rendering is *not* MC sampled; it's an analytic
BSDF evaluation over the FPS'd point cloud. The cart_corr numbers
won't match Mitsuba's to full precision, but they are the same
numbers we use for training (`train_gaussian.py`, `train_chirp_loop_nvs.py`),
which is more consistent with the downstream task than the MC
renderer was. Run a sanity pass on 3–5 frames with both backends
and confirm the *ranking* of pose candidates agrees; small absolute
differences are expected and fine.

## Alternative approaches (considered, not chosen)

- **Pure smoothing (no re-alignment) (Stage-A-only, option (c)).**
  Simplest. Use LOWESS-smoothed poses as the final poses. Risk:
  throws away signal in scenes where the rig genuinely moved non-
  smoothly. Keeping it as a per-scene fallback
  (seq_1_frame_438, seq_2_frame_160) is cheap; using it globally is
  premature.
- **Joint trajectory optimisation.** Solve all 9 frames as one
  optimisation over a cubic-spline-parametrised trajectory. More
  principled, but requires rewriting the alignment loss to operate
  on a batch of frames simultaneously. Defer to a future pass if
  per-frame B1 proves insufficient.
- **IMU / groundtruth use.** ColoRadar may ship IMU and groundtruth
  trajectories for each sequence; if they are reliable in the
  training windows, they eclipse any trajectory fit we can build
  from 9 frames. Check `data/<scene>/` and the ColoRadar tools for
  an IMU/groundtruth time-series before committing to B1. **Flagged
  open question** — see below.
- **Skip the LiDAR method in pass 2, renderer-only with CUDA.**
  Tempting because the CUDA renderer is so much faster, but loses
  the cross-method diversity pass 1 relied on when the objective
  surfaces disagree. Keep both.
- **Keep the MC renderer, just add the prior.** Works but the grid
  search cost balloons when you add the elevation DOF needed to
  fix z. The CUDA swap is what makes 4-DOF renderer alignment
  tractable.

## Implementation plan

### File map (new + modified)

- **New**
  - `mm25DGS_v5/preprocessing/__init__.py`
  - `mm25DGS_v5/preprocessing/alignment/__init__.py`
  - `mm25DGS_v5/preprocessing/alignment/cascaded_renderer_cuda.py`
    (~250 LOC) — v5 CUDA rendering backend for alignment
  - `mm25DGS_v5/preprocessing/alignment/pass2/__init__.py`
  - `mm25DGS_v5/preprocessing/alignment/pass2/trajectory_fit.py`
    (~200 LOC) — Stage A
  - `mm25DGS_v5/preprocessing/alignment/pass2/run_pass2.py`
    (~300 LOC) — Stage B driver + CLI
- **Vendored (copy-in, edit locally)**
  - `mm25DGS_v5/preprocessing/alignment/cascaded_renderer.py`
    — strip mitsuba / drjit / `mmir.renderer` imports; swap
    `render_and_evaluate` → `render_and_evaluate_cuda`; add prior
    kwargs to `run_single_frame`, `grid_search_2dof`, `refine_2dof`;
    extend to 4-DOF.
  - `mm25DGS_v5/preprocessing/alignment/cascaded_lidar.py` — mostly
    unchanged; kept as CPU reference.
  - `mm25DGS_v5/preprocessing/alignment/cascaded_lidar_gpu.py` —
    add `pose_prior`, `prior_weight`, `search_radius` kwargs.
  - `mm25DGS_v5/preprocessing/alignment/cascaded_alignment.py`
    — add `pass2` CLI subcommand; route select_best_alignment
    through the trajectory-consistency gate; write outputs to
    `*_pass2.json`.
  - `mm25DGS_v5/preprocessing/alignment/sc_trajectory_transfer.py`,
    `gpu_utils/*` — vendored unchanged.
- **Outputs**
  - `data/alignment_data/<scene>/cascade/pass2_triage.json` (Stage A)
  - `data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json` (Stage B winner)
  - `data/alignment_data/<scene>/cascade/cascaded_frame_<F>_alignment_log_pass2.json` (Stage B log, includes both methods' scores + MAD residuals + gate decision)
  - `data/alignment_data/<scene>/cascade/pass2_summary.json` (Stage B3 verification)

### Output routing

Write pass-2 configs to *parallel* files (`_pass2.json`), not in-place.
Add an env-variable / CLI flag in the downstream consumers
(`train_gaussian.py`, `train_chirp_loop_nvs.py`, etc.) to opt into
pass-2 configs via
`/home/adnan/Desktop/mm3DGS/data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json`.
Once pass 2 is validated on at least one downstream task, swap the
default. Never overwrite pass-1 artefacts.

### Phasing

1. **Phase 0 (0.5 day).** Check for IMU / groundtruth trajectories
   in `data/` and in the ColoRadar tools. If reliable, redesign —
   maybe skip pass 2 entirely.
2. **Phase 1 (0.5 day).** Vendor `mmir/preprocessing/alignment/*`
   into `mm25DGS_v5/preprocessing/alignment/`; strip `mitsuba`,
   `drjit`, and `mmir.renderer` imports from `cascaded_renderer.py`;
   stub `render_and_evaluate` to raise `NotImplementedError` until
   Phase 2 lands.
3. **Phase 2 (1–1.5 days).** Implement `cascaded_renderer_cuda.py`
   (the CUDA rendering backend). Wire
   `render_and_evaluate_cuda` into the vendored `cascaded_renderer.py`.
   Sanity-check on 3–5 frames: pass-1 winner pose should re-score
   with the CUDA backend to a similar cart_corr and pose-candidate
   ranking (exact numbers won't match MC, but ordering should).
4. **Phase 3 (1 day).** Implement Stage A (`trajectory_fit.py`);
   run on all 9 scenes. Commit `pass2_triage.json` per scene.
   Inspect the soft flags manually and decide scope-limit per scene.
5. **Phase 4 (1 day).** Extend both B1-R (renderer, already CUDA)
   and B1-L (lidar_gpu) with `pose_prior`, `prior_weight`,
   `search_radius`, and enable 4-DOF in the renderer path.
   Smoke-test on `seq_0_frame_135` frame 135. Confirm
   z-component is pulled back near the smoothed value and the
   winner's cart_corr does not regress by more than 0.02.
6. **Phase 5 (1–2 days).** Full B2 (winner + trajectory gate) and
   B3 (re-verify Stage A on outputs). Roll out across all scenes.
   Commit `cascaded_frame_<F>_aligned_pass2.json` per re-aligned
   frame and `pass2_summary.json` per scene.
7. **Phase 6 (0.5 day).** Downstream wiring:
   - add a `use_pass2_alignment: bool` flag to
     `train_chirp_loop_nvs.py` and `train_gaussian_nvs.py`
     (default False for now);
   - re-run Experiment A (upper bound) and Experiment B
     (held-out loop 8) with pass-2 configs on seq_0_frame_135
     and compare to the pass-1 baselines (0.89 / 0.81);
   - expected outcome: per-loop position shift across 16 loops
     shrinks from 73 mm → ~7 mm; pre-training cart_corr rises;
     held-out loop 8 cart_corr rises modestly (interpolation was
     already near the noise ceiling, so the ceiling is noise-limited
     either way).

## Success criteria

- **Hard**: every frame in the outlier catalog has its ⚠Z! flag
  removed (z-component within 3 MAD of the smoothed trajectory).
- **Hard**: per-scene mean cart_corr of pass-2 winner configs ≥
  pass-1 mean cart_corr − 0.02.
- **Hard**: renderer-method single-frame wall time ≤ 10 s (down
  from minutes in pass 1) with the CUDA backend on a 4-DOF, ~10×10×5×5 grid.
- **Soft**: second derivative of the per-scene trajectory
  (finite-differenced over the 9 frames) reduced by ≥2× on average.
- **Soft**: downstream chirp-loop NVS pre-training cart_corr on
  seq_0_frame_135 rises from 0.48 (pass 1) to ≥ 0.55 (pass 2).

## Open questions

- **Is IMU / groundtruth trajectory data available?** If yes, it
  replaces Stage A entirely. Phase 0 must resolve this before any
  implementation.
- **Is seq_2_frame_105's consistent -0.22 z real or artefact?**
  Needs a manual look at the scene geometry and the raw LiDAR rig
  pose. If the rig is genuinely angled down, leave the pose alone
  and flag the outlier detector to ignore scene-constant deviations.
- **Does the 4-DOF `apply_4dof_to_config` reach elevation
  directly?** Verify that its output can represent any boresight
  on the unit sphere within the prior's radius. If it parametrises
  only in-plane rotations chained together, extend it with an
  explicit elevation delta rather than chaining.
- **CUDA renderer vs MC renderer ranking agreement.** Phase 2 must
  explicitly compare the ordering of top-10 pose candidates between
  the two backends on a sample frame; if they disagree badly, fall
  back to MC for the renderer method and accept the slower walltime.
- **Priority ordering of the two unlisted scenes**
  (`seq_0_frame_451`, `seq_1_frame_277`). Run Stage A on them first
  to find out.

## Impact on chirp-loop NVS

This plan exists to remove the 73 mm-shift artefact produced by
pass-1 alignment noise in the Experiment A/B results above. Once
pass 2 lands:

- `train_chirp_loop_nvs.py` anchors on
  `cascaded_frame_<F-1>_aligned_pass2.json` and
  `cascaded_frame_<F+1>_aligned_pass2.json`.
- Expected per-loop position shift: ≤10 mm (consistent with walking
  pace), vs 73 mm today.
- Pre-training per-loop cart_corr expected to rise and tighten (all
  16 loops at ≥0.55 and std ≤0.01 at init).
- Upper-bound and held-out experiments can then be re-run, giving a
  clean baseline for multi-frame NVS without the pass-1 pose noise
  confound.
