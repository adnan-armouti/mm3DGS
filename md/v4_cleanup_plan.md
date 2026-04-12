# mm25DGS_v4 cleanup plan

Goal: keep only the c6 hemisphere code path. Remove c2/c3/c4/c5 from Stage C, remove Stage B dependencies, remove all mesh use except shadow ray tracing, and use point-cloud normals directly (no mesh-vertex copy).

Verified facts before planning:
- `pcl.npy` is shape `(N_pcl, 7)` float32. Columns 0–2 are xyz, columns 3–5 are unit normals (norm = 1.0 confirmed for 3 sample scenes), column 6 is a per-point scalar (intensity/index, unused).
- v3 c6 currently re-computes normals from the mesh via KDTree → nearest mesh vertex. This is unnecessary work.
- v3 imports four things from `mm25DGS_v2` (`render_mmIR`, `rasterizer_torch`, `train_mesh`) and two from `mm25DGS` (`bsdf_torch`).
- Best result on commit `c73d761`: visible-weighted FPS init + uniform hemisphere weights @1500 iters → mean cart_corr 0.8767 across 7 scenes.

## Files to delete

| File | Reason |
|---|---|
| `mm25DGS_v4/test_spread.py` | Imports from `mm25DGS_v2` and references c2/c3 helpers |
| `mm25DGS_v4/run_c3_c4_parallel.py` | c3/c4 only |
| `mm25DGS_v4/output/` | Stale outputs from earlier modes |

## Files to modify

### `mm25DGS_v4/train_gaussian.py`

**Remove**:
- Module docstring lines mentioning c2/c3/c4/c5; rewrite for c6 only.
- `init_from_mesh` function (lines ~168–214). c2/c3 only.
- `init_from_lidar` function (lines ~216–296). c4/c5 only.
- `_compute_density_ratio_weights` function. Replaced by uniform weights in commit `c73d761`.
- `_compute_spherical_voronoi_weights` function (if present in v4 copy). Tested and abandoned.
- Mode dispatch branches for c2, c3, c4, c5 in `train_gaussians`.
- All `if mode == 'c2'`, `'c3'`, `'c4'`, `'c5'` conditional branches throughout (active mask logic, density-control branches, culling-recompute checks).
- `_get_visible_vertices` import and `visible_mask` computation block — only used by c2/c3 active mask logic; for c6 the active mask comes from `rx_visible & cull_gaussians`.
- `cull_gaussians` `cos_threshold` parameter is c4/c5-tuned; review whether the same defaults work for c6 (it uses `cos_normal > 0.05` which should be fine).
- C5 density control block (`if mode == 'c5'` densify/prune section).
- Argparse `choices=['c2','c3','c4','c5','c6']` → `choices=['c6']` (or remove `--mode` entirely).
- Argparse `--n-gaussians` → keep but rename to `--target_n` for clarity; remove "C4" from help string.
- `target_n` default behavior. Currently defaults to `len(mesh_verts)` (which depends on the mesh). Change to a fixed default (e.g., 50_000) and document it.

**Replace** mesh-derived normal lookup in `init_from_lidar_visible_weighted`:
- Currently loads `mesh.ply` via trimesh, KDTrees on `mesh_verts`, copies nearest mesh normal to each FPS point.
- Replace with: read columns 3–5 of `pcl.npy` directly. Carry the normal column through the FOV/visibility/cosine-resample/FPS pipeline alongside xyz.
- This removes the `import trimesh`, the KDTree-on-mesh, and the `mesh_verts/mesh_normals` references inside this function.

**Rename** `init_from_lidar_visible_weighted` → `init` (only one init in v4).

### `mm25DGS_v4/rasterizer_factorized.py`

- No changes required for the cleanup itself. The renderer is c6-compatible as-is (uniform weights → `dOmega = areas[:, None]` is unchanged).
- Optional: remove the `shadow_mask` parameter and back-face culling related to it if we drop TX shadow ray tracing (see `ray_tracing_minimization_plan.md`).

### `mm25DGS_v4/__init__.py`

- Remove any exports of removed functions.

## Stage B (mm25DGS_v2) dependency removal

v3/v4 currently imports from `mm25DGS_v2`:

| Symbol | From | What it does | Replacement strategy |
|---|---|---|---|
| `load_trained_config` | `mm25DGS_v2.render_mmIR` | Loads `SceneConfig` (paths to mesh, pcl, antenna patterns, GT ADC) | Inline a minimal `SceneConfig` loader inside v4 |
| `load_best_params` | `mm25DGS_v2.render_mmIR` | Loads mmIR-trained antenna patterns + raw materials | Keep as a single helper file `v4/load_pretrained.py`. The patterns are external assets; we're not removing the dependency on the trained outputs themselves, only the v2 code path |
| `TRAIN_OUTPUT_DIR` | `mm25DGS_v2.render_mmIR` | Path constant | Move to v4 constants |
| `RasterizerTorch` | `mm25DGS_v2.rasterizer_torch` | Wraps Mitsuba scene + antenna patterns + reservoir sampler | **Hardest dependency.** Two options below |
| `evaluate_bsdf_jones_f_cos` | `mm25DGS_v2.rasterizer_torch` | BSDF | Already replaced inside `rasterizer_factorized.py` for the inner loop, but the import is still pulled. Drop the import |
| `reparameterize_torch` | `mm25DGS_v2.rasterizer_torch` | Material parameter reparameterization | Move to `mm25DGS_v4/materials.py` (small standalone module) |
| `get_lr_scale`, `rms_clip_grad` | `mm25DGS_v2.train_mesh` | LR schedule + grad clipping | Inline these in `train_gaussian.py` (each is <20 lines) |

### Two options for `RasterizerTorch`

**Option A — Keep the import, vendor only what we use.** Create `mm25DGS_v4/rasterizer.py` that contains a slimmed `RasterizerTorch` with only:
- antenna pattern loading (`tx_antenna`, `rx_antenna`)
- TX/RX positions and boresights
- Mitsuba scene loader (`_load_mi_scene` — just the SceneContext + scene assignment, *without* the reservoir sampler)
- Configuration constants (`K`, `slope`, `sample_rate`, `center_freq`, `n_tx`, `n_rx`)
- `inject_trained_params(raw_materials, normals, pattern_data)`

Drop the `_run_reservoir_sampler` and `_prepare_hit_data` methods entirely — they're only used for c4/c5 visibility heuristics, which we've replaced with `_ray_test_visibility_batched`.

**Option B — Keep the v2 import for now, just remove the unused methods at the call sites.** Less work but doesn't actually break the v2 dependency.

**Recommendation: Option A.** The slimmed rasterizer is ~150 lines and removes the v2 directory dependency entirely.

## Argparse / CLI

Current: `python -m mm25DGS_v3.train_gaussian --mode c6 --scene seq_0_frame_135 --iters 1500`

Proposed v4: `python -m mm25DGS_v4.train_gaussian --scene seq_0_frame_135 --iters 1500 --target_n 50000`

- Drop `--mode`
- `--target_n` defaults to 50000 (no longer derived from mesh vertex count)
- Add `--all` flag to run all 7 scenes (already exists)

## Verification after cleanup

Run a single scene with the cleaned v4 and confirm the result matches the baseline within run-to-run noise:

```
CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v4.train_gaussian --scene seq_0_frame_390 --iters 1500
# Expected: cart_corr ≈ 0.87 ± 0.04 (matches commit c73d761 single-scene result)
```

Then run all 7 and confirm mean ≈ 0.876.

## Out of scope

- Performance optimization (chunking, fp16, etc.)
- New training schedules
- New loss functions

These are tracked separately if/when we want to push past 0.9.
