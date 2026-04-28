# v5_v4 — voxel-aware initialization plan

Replaces the FPS-based init in `mm25DGS_v5_v4.train_gaussian.init_visible_weighted` /
`init_visible_weighted_radar_aware` with a sampler that respects the radar's
actual resolution structure: **one Gaussian per resolvable RA cell, allocated by
multi-view cosine-hemisphere weighting, with normal-aligned within-cell
selection.**

## Why this exists

`Phase A` diagnostics ([md/diagnostics_v5_v4_fisher/concentration.md](diagnostics_v5_v4_fisher/concentration.md))
showed that with the current FPS-based init:

- **top 10% of points hold ~76 % of total `‖∇p L‖`** even at iter 100 (before
  any densification);
- **bottom 50 % (10 k pts) hold < 2 %** — half the budget is dead weight;
- **B1/B2 strict-AND visibility init** (the first proposed lever) made the
  concentration *worse* (+5 pp) and gave neutral test cc.

Conclusion from the negative B1/B2 result: filtering and re-sampling within an
already-radar-biased candidate set is not enough. We need an init that
**explicitly respects the radar's resolution lattice and aggregates importance
across all train + test poses**, so each Gaussian slot is provably useful.

## Anchor numbers (cascade radar at chirp 0)

Source of truth: `mmir.data.io_utils.compute_range_res_from_cfg` and
`mm25DGS_v5_v4.train_gaussian.build_polar_to_cart_grid(127, 256, range_res, …)`,
which mirrors `mmir.data.ra_utils.ra_polar_to_cartesian` /
`save_ra_image`'s arcsin convention.

| quantity | value |
|---|---|
| Azimuth bins | **127** |
| Range bins | **256** |
| Range resolution | 5.93 cm |
| Max range | **15.18 m** (256 × 5.93 cm) |
| Azimuth FOV half-cone | **79.86°** (arcsin(126/128)) |
| `cos_bore_min` (canonical) | **0.1761** |
| Azimuth bin width — boresight | 0.895° |
| Azimuth bin width — edge | 4.22° |
| Total RA cells | 127 × 256 = **32,512** |

**Voxel sizes** (range × azimuth, by scene distance):

| range | boresight | edge |
|---|---|---|
| 3 m | 5.9 × 4.7 cm | 5.9 × 22 cm |
| 5 m | 5.9 × 7.8 cm | 5.9 × 37 cm |
| 7 m | 5.9 × 11 cm | 5.9 × 52 cm |
| 10 m | 5.9 × 16 cm | 5.9 × 74 cm |
| 15 m | 5.9 × 23 cm | 5.9 × 110 cm |

Scene admission (seq_0_frame_135): ~1.35 M pcl candidates pass
cone + range, ~5,337 pcl points beyond max_range (drop).

---

## V1 pipeline (no iterative refinement)

### Stage 1 — Tighten FOV cone to the arcsin azimuth FOV
Replace the current `cos_bore_min = 0.10` (baseline) / `0.05` (B1/B2) with
**`cos_bore_min = 0.1761`** (`= cos(arcsin(126/128))`). Excludes points the
radar cannot resolve to a unique azimuth bin.

**Files**: `_per_pose_visibility_masks` and `init_visible_weighted` FOV cull
in [`mm25DGS_v5_v4/train_gaussian.py`](../mm25DGS_v5_v4/train_gaussian.py).

### Stage 2 — Strict-AND visibility (B2 form, all 9 poses)
For each pose F ∈ {train_F-4, …, train_F-1, train_F+1, …, train_F+4, test_F},
do per-pose Mitsuba LOS raycast. AND-reduce → "all-9-poses-visible" set
(~600 k for seq_0_frame_135). Test pose contributes geometry only, never
signal — same convention as v5's `seed_frame=test_frame` default.

### Stage 3 — World-space voxel grid (defined by seed pose)
Build a fixed grid of 32,512 wedges in world coordinates by projecting the
seed pose's RA grid:
```
For each (range_bin j ∈ [0, 256), az_bin i ∈ [0, 127)):
    r_center  = (j + 0.5) · 5.93 cm
    az_center = arcsin( (i − 63) · 2/128 )      # radians
    voxel_origin_world = seed_rx_center + R_world←radar(seed) @
                          [r_center · sin(az_center),
                           r_center · cos(az_center),
                           0]
```
**Reuses `mm25DGS_v5_v4/preprocessing/alignment/gpu_utils/voxelize_gpu.py`** —
the cascade-alignment voxelizer already does this style of binning into
RAE grids.

### Stage 4 — Bin AND-survivors into voxels
GPU vectorized: each AND-survivor → (r_bin, a_bin) under the seed pose's
polar transform. Output: per-voxel candidate list. Voxels with zero
candidates get zero budget.

### Stage 5 — Multi-view importance score (per candidate)
This is the key change vs B1/B2. For each candidate P in the AND set:
```
score(P) = Σ_{F ∈ all 9 poses}  cos(boresight_F, dir_F→P)
                                · max(0, normal_P · −dir_F→P)^α
                                · (1 / d_F(P)²)
                                · LiDAR_intensity_P^β
```
with `α=1` (cosine hemisphere), `β=0.5` (matches A4's intensity prior).
Each candidate is now scored by its **aggregate utility across all 9 views**,
not just the seed pose.

### Stage 6 — Per-voxel budget allocation
For each voxel V (with N_V > 0 candidates):
```
W_V       = Σ_{P ∈ V} score(P)
budget_V  = round(N_total · W_V / Σ_V W_V)
budget_V  = clip(budget_V, 0, N_V)
```
Voxels with non-empty candidates but tiny aggregate importance get budget = 0
(or 1, configurable). Empty voxels get 0. Total budget tuned to land on
N_total = 20,000 (rounding adjustment over voxels in importance order).

### Stage 7 — Within-voxel selection (replaces FPS)
Within voxel V, pick the top-`budget_V` candidates by `score(P)` (the
multi-view aggregate from Stage 5). Ties broken by RNG with seed = 42.

### Stage 8 — Build PointPrimitives
Same as baseline: `positions = xyz`, `rotations = quat(normal)`,
`raw_materials = ITU_CONCRETE`. Implemented as a new init variant
**`C1_voxel_v1`** in `_INIT_VARIANTS`.

---

## V2 pipeline — ACORN-style RA-signal-aware refinement

Augments V1 with measured-signal feedback. Strictly uses train RA only.

### Stage 9 (V2) — Motion-compensated train-RA stack
1. Load all 8 train RA maps (chirp 0).
2. For each train frame F: project its polar RA → seed pose's world voxel grid
   using the known relative pose `T_seed→F`.
3. Average the 8 projected maps → one denoised "training RA average" in the
   seed grid.

This works because at 5 Hz with slow ego-motion, frame-to-frame translation
is < 20 cm — sub-bin to ~1-2 bin shift in azimuth. After motion compensation
the 8 maps over-sample the same scene structure → averaging reduces MC noise
by √8 ≈ 2.8×.

### Stage 10 (V2) — Signal-modulated voxel weights
For each voxel V, look up its measured RA-signal magnitude `|RA|_V` (mean
across motion-compensated train frames). Modify the V1 weight:
```
W_V_v2 = W_V · |RA|_V^γ
```
with γ = 1 to start. Voxels where the radar measures essentially nothing get
budget zero; voxels with strong measured returns get amplified budget.

### Stage 11 (V2) — Iterate (optional, 1–2 rounds)
Re-allocate budget under the modulated weights. The "ACORN connection" is
the per-region adaptive resolution: high-signal voxels get more capacity,
low-signal ones less.

---

## Multi-view handling — how the plan stays consistent across the 9 views

This is the most important conceptual question, addressed explicitly:

### Each pose has its OWN polar grid — that's a fact, not a problem to fix
At pose F, the radar produces an RA map indexed by *F's* polar grid. A scene
point P gets binned into:
- voxel `V_seed(P)` under the seed pose's polar transform
- voxel `V_F(P)` under pose F's polar transform — generally `V_F(P) ≠ V_seed(P)`

At 5 Hz with cm-scale ego-motion, the disagreement is **at most ~1–2 azimuth
bins** between adjacent frames at typical scene distances. It is real but
small.

### How V1/V2 handle it

**The voxel grid is bookkeeping for the init sampler — it never participates
in the loss or rendering.** The rendering pipeline evaluates each pose's RA
pixel independently from world-space Gaussians via
`render_gaussians + apply_pose(rast, train_pose)`. The voxel grid is discarded
after init. So per-pose grid disagreements are irrelevant downstream.

**The seed pose's grid is just a spatial uniformity control.** We use it to
ensure no patch of world space gets way more Gaussians than another. We do
NOT use it to pick which Gaussians are "good" — that's what the multi-view
score does (Stage 5).

**Multi-view consistency lives in the score, not the grid.** Stage 5 sums
contributions across all 9 poses. A candidate that is well-positioned for the
seed pose only (high boresight cosine, near range, good normal) but badly
positioned for the 4 other train poses gets a *low* aggregate score. A
candidate equally useful from all 9 viewpoints gets a *high* aggregate score.
Per-voxel budget is allocated proportional to the aggregate, then within
each voxel we keep the top-K by aggregate. So the *selection* is multi-view
optimal even though the *binning* references one pose.

### Why not use multiple voxel grids and union/intersect them?
- An intersection of 9 polar grids creates ~10× more cells, each tiny
  (intersection of 9 wedges shrinks fast); budget allocation becomes
  numerically fragile and computationally expensive.
- A union creates fuzzy cell boundaries that don't map back to any single
  pose's pixel — spatial uniformity loses its physical interpretation.
- The seed-pose grid is sufficient because (a) it's a Voronoi-like
  partition of world space at radar resolution, and (b) the multi-view
  score handles the "is this point useful from other views?" question.

### Why not use an axis-aligned cubic voxel grid in world space?
- Cubic voxels lose the "match radar resolution" property — a 10 cm cube
  at 15 m range covers ~10 azimuth bins (over-coarse near range, under-coarse
  far away).
- Polar wedges scale naturally with range — exactly the property we want.

### Test-time semantics
At inference, the same Gaussian model is rendered from the test pose's RA
grid. The voxel grid was only used at init. Test pose's geometry was used
in Stage 2 (visibility) and Stage 3 (we set `seed_frame = test_frame`); test
pose's signal is never touched.

If reviewer hygiene around test-pose-in-init becomes a concern: set
`seed_frame = first_train_frame` instead. The B1 vs B2 result already
showed the AND survivor count is identical with or without the test pose
(at 5 Hz, the test pose's visibility cone is fully covered by the union of
train cones), so this swap costs nothing.

---

## Implementation milestones

| # | scope | ETA | depends |
|---|---|---|---|
| **M0** | Tighten FOV cone to `cos_bore_min = 0.1761`. 2-line change. Bench on 6 scenes vs current defaults (= combo_jitter mean 0.5863). | 30 min | — |
| **M1** | World-space voxel grid (Stage 3). Reuse `voxelize_lidar_gpu`. Unit-test bin assignment is consistent w/ `build_polar_to_cart_grid`. | 2-3 hrs | M0 |
| **M2** | Multi-view score + per-voxel budget + within-voxel selection (Stages 4-7). New init variant `C1_voxel_v1`. | 4-6 hrs | M1 |
| **M3** | Bench V1 on 6 scenes. Compare Fisher concentration (top-k % at iter 100/200/300/400) and final test cc against baseline. | 30 min run + analysis | M2 |
| **M4** | **Decision point.** Proceed to V2 only if V1 shows ≥ +0.02 mean test cc OR a clear flatter Fisher distribution at iter 100. | — | M3 |
| **M5** | Motion-compensated train-RA stack (Stage 9). Reuse pose alignment from `_build_frame_poses`. | 4-6 hrs | M4 |
| **M6** | V2 sampler with RA-weighted budget (Stages 10-11). New variant `C2_voxel_v2`. | 3-4 hrs | M5 |
| **M7** | Bench V2 on 6 scenes. Final comparison. | 30 min | M6 |

Total: **M0–M3 ≈ 1 day** (V1); **M5–M7 ≈ +1 day** (V2 if warranted).

---

## Open risks

1. **Voxel cell count vs N=20 k**: occupied voxels are typically 5–15 k;
   small-budget voxels (budget < 1) need rounding logic. Track total points
   and adjust until it lands on 20 k. Trivial.
2. **Per-voxel budget might still concentrate**: if multi-view importance
   itself is concentrated (a few corners dominate from all 9 views), then
   budget concentrates too. The Phase A diagnostic showed concentration is
   largely scene-intrinsic — V1 may not flatten as much as hoped. V2's
   measured-signal modulation is the second line of defense.
3. **Elevation handling**: we use 2D RA wedges (full elevation extent per
   wedge) since v5's loss is also 2D RA. If we later move to RAE, voxels
   become 3D and the grid grows ~10×. Out of scope for V1/V2.
4. **Edge azimuth voxels are large** (1.1 m at 15 m, edge bin): one Gaussian
   per such cell may miss intra-cell structure. The cosine-hemisphere weight
   naturally allocates them low budget, mitigating the issue.
5. **Test-pose-in-init**: B1 vs B2 showed empirically irrelevant for this
   data. If reviewers object, swap `seed_frame = first_train_frame`. No code
   change beyond CLI default.

---

## Recommended start

**M0 first.** It's a 2-line change with a 6-scene bench; isolates whether the
admit-cone width has been the silent culprit all along, before any voxel
machinery is built. If M0 alone moves the mean test cc by +0.01 or more,
that's a free win and we recalibrate expectations for M1–M3.
