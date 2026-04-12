# mm25DGS_v4 ray-tracing minimization plan

Goal: in v4, the only ray tracing should be the per-point RX-side visibility test used to filter the raw point cloud at init. Everything else (reservoir sampler, redundant per-Gaussian visibility test, TX shadow rays) should be removed.

## Current ray tracing in v3 c6 (4 distinct uses)

| # | Function | Where | Input | Cost | Necessity for c6 |
|---|---|---|---|---|---|
| 1 | `rast._run_reservoir_sampler(seed=42)` | `train_gaussian.py:867` | n/a (samples from Mitsuba scene) | ~24K rays from RX (16 RX × 1500 hits) | **None** — used to compute `visible_verts` for the c2/c3/c4/c5 active mask path. c6 active mask uses `rx_visible & cull_gaussians`, not `visible_mask` |
| 2 | `_ray_test_visibility_batched(xyz_fov, rx_center, rast._mi_scene)` | `train_gaussian.py:391` | ~1.5M point-cloud points (after FOV filter) | ~1.5M rays | **Required** — this is the only reason c6 has 100% visible Gaussians by construction |
| 3 | `_compute_rx_visibility(model.positions, rast)` | `train_gaussian.py:880` | 50K FPS Gaussians | ~50K rays | **Redundant** — see explanation below |
| 4 | `_compute_shadow_mask(model.positions, rast)` | `train_gaussian.py:885` | 50K Gaussians × n_tx (12) | ~600K rays | **Optional** — F7 was reverted on c4/c5 due to self-occlusion at mesh vertices, but for c6 the FPS points lie on the visible mesh surface (not at vertices), so self-occlusion is less likely. Currently passed to renderer but evaluation needed |

Total c6 init ray cost today: ~24K + 1.5M + 50K + 600K ≈ **2.17M rays**.
After this plan: ~1.5M rays. **27% reduction** with no functional change.

## Why `_compute_rx_visibility` and `_ray_test_visibility_batched` are redundant

Both functions cast a ray from a candidate point toward the RX array center and check if any other surface intercepts it. They are mechanically the same test:

```python
delta = rx_center - point_position
direction = delta / |delta|
origin = point_position + eps * direction
ray_test(origin, direction, maxt=|delta| - 2eps)
```

The only differences are:
- `_ray_test_visibility_batched` takes a numpy array, batches into chunks of 200K, returns a numpy mask. Used inside `init_from_lidar_visible_weighted` on the **full point cloud (1.5M points)** before FPS.
- `_compute_rx_visibility` takes a torch tensor, single batch (no chunking), returns a torch mask. Used in the training loop init on the **50K FPS Gaussians** after init.

**The pipeline order makes the second call redundant for c6:**

1. `_ray_test_visibility_batched` runs on 1.5M points → keeps only the ~500K-700K visible ones.
2. Cosine importance resample → ~150K points (all from the visible subset).
3. FPS → 50K points (all from the visible subset).
4. Model is built from these 50K points.
5. `_compute_rx_visibility` runs on the 50K model points → returns 100% visible (we observed `RX-visible Gaussians: 50499/50499` and `107353/107353` in actual runs).

The second call always returns all-true because every Gaussian was already visibility-filtered before being placed in the model. It's a leftover from when c6 used the plain `init_from_lidar` (no pre-FPS visibility filter) and needed a post-FPS cleanup pass.

**Conclusion:** in v4, drop `_compute_rx_visibility` entirely. Drop the `rx_visible` variable. Replace the active mask `rx_visible & cull_gaussians(model, rast)` with just `cull_gaussians(model, rast)` (FOV culling, since visibility is already guaranteed).

## Why we can also drop `_run_reservoir_sampler` for c6

The reservoir sampler exists to:
1. Identify radar-visible mesh vertices (`visible_verts` → `visible_mask`) for c2/c3/c4/c5 active mask construction.
2. Provide MC hit positions and weights, used historically for the failed MC weight transfer experiment.

For c6:
- The active mask uses `cull_gaussians` (FOV) only — `visible_mask` is computed but unused.
- We're using uniform hemisphere weights — no MC weight transfer.
- Visibility comes from the dedicated `_ray_test_visibility_batched` on the raw point cloud.

In the v3 c6 code, `visible_mask` is computed but never enters the c6 branch of the active mask logic. This is dead code for c6.

**Conclusion:** drop the entire `hits = rast._run_reservoir_sampler(seed=42)` block, the `_get_visible_vertices` call, and the `visible_mask` tensor for c6. Saves ~24K Mitsuba rays and removes a major dependency on the reservoir sampler module.

## Decision needed: TX shadow mask (`_compute_shadow_mask`)

The F7 commit message says it was reverted on c4/c5 due to self-occlusion at mesh vertices. But:
- c6 places Gaussians at LiDAR FPS positions (not exactly at mesh vertices)
- The shadow ray uses a 1e-4 m offset along the ray direction to avoid self-hit
- The shadow mask is currently computed and passed to the renderer in c6

**Two options:**

1. **Drop shadow mask for v4.** Saves ~600K rays (the largest single ray-tracing cost). Risk: paths blocked from the TX side will be incorrectly counted, possibly hurting rendering quality. The ~0.06 gap to mmIR could partly be from missing TX occlusion.

2. **Keep shadow mask for v4 but verify it actually helps c6.** Run a quick A/B: shadow on vs shadow off, single scene, 1500 iters. If shadow helps, keep it. If not (or if it hurts), drop it.

**Recommendation:** keep the shadow mask code but run the A/B test as the first verification step in v4. If A/B shows no improvement (or regression like F7 saw on c4/c5), drop it and note in the plan.

## Final ray tracing inventory for v4 (after this plan)

| # | Function | When | Cost | Purpose |
|---|---|---|---|---|
| 1 | `_ray_test_visibility_batched` | Init, once, on raw point cloud | ~1.5M rays | Filter point cloud to RX-visible subset before FPS |
| 2 (optional) | `_compute_shadow_mask` | Init, once, on FPS Gaussians | ~600K rays (50K × 12 TX) | TX-side shadow culling per (Gaussian, TX) path |

**Zero ray tracing during the training loop.** All ray tracing is one-shot at init. The Mitsuba scene is freed after init.

If shadow mask is dropped: total init cost = 1.5M rays. If kept: 2.1M rays.

## Implementation steps for v4

1. Delete `_compute_rx_visibility` function entirely.
2. Delete the `if use_hemisphere: rx_visible = ...` block.
3. Change c6 active mask to just `active_mask = cull_gaussians(model, rast)`.
4. Delete the `hits = rast._run_reservoir_sampler(seed=42)` block, `visible_verts`, `visible_mask`.
5. Delete the `_get_visible_vertices` import.
6. Move `_ray_test_visibility_batched` call (currently inside `init_from_lidar_visible_weighted`) earlier so it runs as the first step after FOV filtering on the raw point cloud — this is already the case, no change needed.
7. **A/B test the shadow mask** on one scene. Decision: keep or drop.
8. The Mitsuba scene loader (`SceneContext.from_files`) can be slimmed to load only what's needed for `_ray_test_visibility_batched` (no antenna patterns required for the ray test itself, only the scene mesh).

## Why we still need the mesh for ray tracing

The point cloud is just a sparse sample of the surface — it has no connectivity. To test "is this ray blocked by anything?", we need a watertight or near-watertight surface representation that supports `intersect(ray, surface)`. The mesh.ply provides exactly this. There's no way to do shadow/occlusion ray testing on raw point cloud points without first building a surface representation (mesh, BVH over disks, signed distance field, etc.).

So: the mesh's ONLY remaining role in v4 is as the Mitsuba scene used for ray testing. Everything else (vertex normals, vertex positions, vertex count) is removed in `v4_cleanup_plan.md` because:
- Normals are in `pcl.npy` columns 3–5
- Positions are in `pcl.npy` columns 0–2
- Target Gaussian count is set explicitly via `--target_n`, not derived from the mesh
