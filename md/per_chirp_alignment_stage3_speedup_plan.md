# Stage 3 per-chirp alignment — speedup plan

Stage 3 currently runs at ~20–30 s/chirp on 1× RTX 4090 (≈45–70 min/scene for
9 frames × 16 chirps, extrapolating to ~7 h for the full 9-scene benchmark).
This document explains where the time goes, identifies the cheap wins, and
proposes a phased speedup roadmap aligned with the empirical utility of
Stage 3 (see `md/per_chirp_alignment_stage3_plan.md` for the accuracy
story — Stage 3 is net-positive only for all-chirp training on clean scenes).

## 1. Observed cost breakdown

Wall-clock from `data/alignment_data/<scene>/cascade/per_chirp/pass3_summary.json`:

| scene | wall | chirps | avg / chirp | winner breakdown |
|---|---:|---:|---:|---|
| seq_1_frame_438 | 4173 s (~70 min) | 144 | **~29 s** | anchor 10 · renderer 123 · **lidar 11** |
| seq_2_frame_105 | 2692 s (~45 min) | 144 | **~19 s** | anchor  9 · renderer 132 · **lidar  3** |

Per-chirp the compute falls into four buckets:

| stage | call path | default cost (ms) | % of chirp |
|---|---|---:|---:|
| ctx anchor eval | `render_and_evaluate_cuda` (1 render, FOV cull on) | ~5 | <1 % |
| renderer-4DOF **grid** | `grid_search_4dof` over `5×5×5×5 = 625` cells | ~2 500 | ~10–15 % |
| renderer-4DOF **refine** | `refine_4dof` Nelder-Mead up to 30 evals | ~120 | <1 % |
| **lidar-4DOF** (`optimize_alignment_gpu_with_prior`) | cupy coarse grid (`coarse_steps=5`, batch=100) + Nelder-Mead | **~15 000–25 000** | **~70–90 %** |

The lidar candidate dominates Stage 3 wall time and contributes
**<8 % of winning poses** (14/288 across both scenes). This is the single
largest speedup target.

## 2. Why Stage 3 is "this slow"

1. **LiDAR path always runs, regardless of renderer confidence.** The
   lidar branch in `stage3_refine_chirp` is only gated on
   `ctx._lidar_pcd is not None and ctx._ra_radar_chirp is not None`, both
   of which the scene driver sets unconditionally. There is no early-exit
   when the renderer candidate is already comfortably above the anchor
   gate margin.
2. **Lidar coarse grid is big.** `coarse_steps=5` on 4 DOF is 625 cells,
   each kernel-launched in batches of 100 — but also preceded by
   `prepare_gpu_data()` which transfers the lidar point cloud to GPU
   every call (no across-chirp caching).
3. **Renderer-4DOF grid is 625 cells for a ±10 mm / ±0.5° box.** With
   Nelder-Mead refine right after, a `3×3×3×3 = 81` grid is sufficient to
   bracket the basin. The current 5^4 grid spends 7× more evals than
   necessary.
4. **Every grid cell recomputes the active FOV mask.** The search radius
   is ±10 mm / ±0.5°, well inside a single FOV — `recompute_active_mask`
   can be disabled in the hot loop after the anchor eval.
5. **No parallelism across scenes.** 2× RTX 4090 available; driver runs
   single-GPU. Scene-level parallel execution is trivial (fully
   independent work).
6. **No parallelism across frames within a scene.** Each frame builds its
   own ctx; frames are fully independent but processed sequentially.
7. **Chirp-0 special case already exists** (writes `pass-2_F` directly,
   skips refinement) — nothing to gain here.

## 3. Speedup roadmap — phased

The phases are ordered by expected wall-time reduction per hour of
engineering effort. All phases preserve Stage 3 accuracy; the most
aggressive (P0 and P1) trade off a small, quantifiable accuracy risk that
can be measured directly by the existing frame-NVS A/B harness.

### Phase P0 — disable lidar by default (**~3–5× speedup**)

**Change**: add `--skip-lidar` flag (default **ON** for Stage 3), and set
`ctx._lidar_pcd = None` on the driver when the flag is on. No code
changes to `stage3_refine_chirp` itself; the existing guard will skip
the lidar branch.

**Expected impact**: seq_1 70 min → ~15 min; seq_2 45 min → ~10 min.

**Accuracy risk**: lidar wins 11/144 (seq_1, 7.6 %) and 3/144 (seq_2,
2.1 %) with gate margin 0.005 cc. Reverting these winners to the
renderer candidate *or* the anchor lowers mean winner cc by ~0.001
(upper bound, computed from the gate margin × winner rate). Far below
the MC noise floor (~0.03 cc).

**Validation**: rerun `HO_128 pass3` on both scenes with `--skip-lidar`
and compare against the existing pass-3 columns in
`md/frame_nvs.md`. If ΔHO cc > −0.005 on both scenes, ship.

### Phase P1 — tighten renderer grid to 3^4 (**~10–15 % additional**)

**Change**: in `stage3_refine_chirp`, call `grid_search_4dof` with step
sizes that produce 3 points per DOF instead of 5:
```python
grid_search_4dof(..., range_step=rng, az_step=az,
                 elev_rot_step=e, azim_rot_step=z, ...)
```
This yields 3^4 = 81 cells. Nelder-Mead `refine_4dof` (already called
right after, `max_evals=30`) will descend from the best-of-81 starting
point; the ±10 mm / ±0.5° basin is small enough that 81 samples bracket
the optimum as tightly as 625.

**Expected impact**: renderer 4DOF grid 2.5 s → 0.3 s (factor 7×).
Combined with P0 that's ~1.5–2 s/chirp saved.

**Accuracy risk**: negligible. Nelder-Mead on a 4 D strictly-concave cc
landscape recovers the max regardless of the initial lattice density
once the basin is bracketed.

**Validation**: same as P0 (frame-NVS A/B).

### Phase P2 — freeze active mask inside the grid (**~5–10 % additional**)

**Change**: inside `stage3_refine_chirp`, wrap the grid + refine calls
with a mask-freeze flag:
```python
# After anchor eval (recompute=True), freeze for grid/refine:
ctx.recompute_active_mask_override = False
... grid_search_4dof / refine_4dof calls ...
ctx.recompute_active_mask_override = True
```
and plumb `recompute_active_mask_override` into
`render_and_evaluate_cuda`. Justified by the ±10 mm translation bound:
culling changes <<1 % of vertices across this box.

**Expected impact**: 15–20 % reduction on the renderer path, ~0.3 s/chirp.

**Accuracy risk**: negligible; the culling affects at most edge
gaussians whose contribution is ~0.

### Phase P3 — early-exit on high anchor cc (**~15–25 % on clean scenes**)

**Change**: at the start of `stage3_refine_chirp`, if the anchor cc is
within `gate_margin` of 1.0 (e.g. cc_anchor > 0.95), skip both grid
calls. No refinement can beat this by the gate margin, so the anchor
wins deterministically.

**Expected impact**: Only fires on chirps with near-saturated anchor cc
(not common on our benchmark scenes; anchor cc ~0.23–0.31 mean on the
two tested scenes). **Low-priority** for current data.

**Accuracy risk**: zero by construction.

### Phase P4 — scene-level parallelism across GPUs (**~2× on 2-GPU box**)

**Change**: add a `--scenes s1,s2,...` + `--gpus 0,1` mode to the
driver that forks one subprocess per scene pinned to one GPU each via
`CUDA_VISIBLE_DEVICES`. Scenes are fully independent at the file level
(each writes to its own `alignment_data/<scene>/cascade/per_chirp/`).

**Expected impact**: 2× wall-time reduction when 2+ scenes are queued.

**Accuracy risk**: zero.

### Phase P5 — frame-level parallelism within a scene (**~4–9× per scene**)

**Change**: within `run_stage3_for_scene`, dispatch the per-frame loop
to a `ThreadPoolExecutor` with one worker per GPU (or
`ProcessPoolExecutor` if CUDA contexts cannot be shared). Each frame
already builds its own ctx, so the only shared state is disk writes.
Compose with P4 for full 2-GPU × 9-frame fanout.

**Expected impact**: 4–9× within-scene speedup. Much more effort than
P0–P4 (CUDA context management, proc-level fan-in) and harder to debug.
**Defer until P0–P4 are shipped and measured.**

**Accuracy risk**: zero (pure parallelism of independent work).

### Phase P6 — batched multi-pose rendering (optional, **~3–5× on grid**)

**Change**: extend `Rasterizer` to accept a `(B, …)` batch of TX/RX
poses and evaluate them in a single fused launch. Then the renderer
4DOF grid becomes one kernel launch instead of 81–625 sequential calls.
This is a renderer-level refactor; only pursue if P0–P5 do not hit
budget.

**Expected impact**: renderer grid 0.3 s → ~0.05 s. Marginal after P1.

**Accuracy risk**: zero if the batched path replicates the sequential
numerics exactly.

## 4. Recommended rollout order

1. **Ship P0 immediately** (skip-lidar default). Measure impact in
   wall time + HO_128 cc A/B on both scenes. Expected: ~4× speedup, cc
   within ±0.003 of current Stage 3.
2. **Layer P1 on top** (3^4 grid) in the same PR. Expected: additional
   ~10–15 % on top of P0.
3. **Add P4** (scene × GPU parallelism) to the run-alignment driver.
   This is the easiest multi-GPU win and needs no Stage-3 changes.
4. Leave P2, P3, P5, P6 behind a performance flag; re-measure after
   P0+P1+P4 to see if they are still needed.

## 5. Target Stage 3 timing after P0 + P1 + P4

- Per-chirp: ~29 s (seq_1) → **~5 s**; ~19 s (seq_2) → **~3.5 s**.
- Per-scene: 70 min → **~12 min** (seq_1); 45 min → **~8 min** (seq_2).
- Full 9-scene benchmark on 2× RTX 4090: **~30–40 min** (vs. current
  ~7 h).

This brings Stage 3 into a regime where it is a viable *default* for
all-chirp training variants, rather than a one-shot debugging run.

## 6. Out of scope — do not pursue

- **2 DOF-only Stage 3** (skip elev/azim rotation). Rotation ∈ ±0.5°
  only matters for the rotation refinement that Stage 3 is explicitly
  designed to capture (see plan doc §3.2); cutting it defeats the
  purpose.
- **Changing the gate margin** (0.005 cc). Tied to MC noise; not a
  speedup lever.
- **Anchor source switching for speed**. `hybrid` is the accuracy
  default and costs nothing beyond `lerp` (the GT load is sub-ms per
  chirp after caching).

## 7. Checklist — ship in this order

- [ ] P0: add `--skip-lidar` CLI flag (default `True`) to
      `per_chirp_alignment.py`; thread into `run_stage3_for_scene`;
      re-run HO_128 A/B on seq_1 & seq_2; update
      `md/frame_nvs.md`.
- [ ] P1: tighten `grid_search_4dof` call in `stage3_refine_chirp` to
      3 points per DOF; re-run same A/B.
- [ ] P4: extend `run_alignment.py` with `--scenes S1,S2,...` +
      `--gpus G1,G2` multi-process fan-out; include Stage 3 in the
      same driver (see companion change).
- [ ] Rerun end-to-end wall-time measurement; confirm ≤12 min/scene.
- [ ] (Optional) P2, P3, P5, P6 — measure marginal gains; only ship
      if needed.
