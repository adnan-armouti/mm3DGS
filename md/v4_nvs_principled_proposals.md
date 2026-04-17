# NVS principled regularizer proposals (working document)

## Where we are

We're investigating why the v4 model produces a sharp ~0.28 cart_corr drop
at held-out interpolation poses (e.g. frame 133 between train frames 132
and 134) when training on 6 of 9 frames. The all-in (8 frame) ceiling on
those same held-out frames is 0.79-0.83. Current best:

| Setup (Scenario 6/2: train {0,1,3,5,7,8}, test {2,6}) | f133 | f137 | mean train |
|---|---|---|---|
| Baseline (no reg) | 0.536 | 0.606 | 0.837 |
| Bracket-anchor model-target λ=0.05 (replicated 3×) | **0.604** | **0.629** | 0.838 |
| **All-in S2 ceiling** | **0.825** | **0.790** | (different fit) |

Bracket-anchor closed ~20% of the gap on f133 with no train-cc cost. To
close the rest, we need a *qualitatively different* mechanism — not a
hyperparameter sweep or warm-start of the same loss.

## What we have ruled out

- **Capacity reduction** (90K → 10K points): zero effect on test cc
- **Loss type** (mse_raw → pearson): +0.04 on test, included in baseline
- **Material gauge collapse to poly2 basis (174 params)**: +0.009 (within noise)
- **Normal anchor to init** (λ ∈ {0.05, 0.5, 5}): flat
- **Facing-score gradient weighting** (suppressing grazing-angle points): flat
- **Material k-means clustering** (K ∈ {10..2000}): ≤ +0.008
- **Render-consistency to perturbed pose** (pulls toward const): -0.05
- **Anchored render smoothness** (||(r_b−r_a) − (gt_b−gt_a)||²): +0.011 to -0.076
- **Pose-midpoint linearity reg**: +0.03 on f133 / -0.05 on f137 (mixed)
- **GT-bracket target** (instead of model-bracket target): underperforms model-target

The pattern of failures: every parameter-side or naive output-space
regularizer either does nothing or fights the train fit. The one that
works (bracket anchor) addresses the failure mode at the test pose
itself, not a downstream symptom.

## What the literature does for this exact problem

Three patterns recur in 2024-2025 papers on few-view inverse rendering /
SAR / RF / Gaussian splatting:

### Pattern A — Spatial / geometric smoothness priors on materials

Multiple papers report this as the standard fix:
- **Depth-Regularized Optimization for 3D GS in Few-Shot Images** (CVPRW
  2024, [arxiv 2311.13398](https://arxiv.org/html/2311.13398v3)): kNN
  smoothness on neighbor 3D points, "neighbor 3D points have similar depths"
- **PBR-NeRF** ([arxiv 2412.09680](https://arxiv.org/html/2412.09680v1)):
  basic BRDF smoothness priors that "discourage abrupt spatial changes in
  material properties"
- **DET-GS — Depth- and Edge-Aware Regularization** ([arxiv 2508.04099](https://arxiv.org/html/2508.04099v1)):
  RGB-guided edge-preserving total variation that "selectively smooths
  homogeneous regions while rigorously retaining high-frequency details"
- **Neural Microfacet Fields for Inverse Rendering**: data-driven BRDF
  priors with simple smoothness terms

The key sophistication: anisotropic / edge-aware smoothness, not uniform.
The smoothness is keyed to a per-point signal (RGB color, depth gradient,
semantic label) so the loss only smooths *across the same surface*, not
*across material boundaries*.

### Pattern B — Tied / region-level material parameterization

- **MaterialFusion** ([arxiv 2409.15273](https://arxiv.org/html/2409.15273v1)):
  uses a learned material diffusion prior — effectively a learned codebook
- **Spherical Voronoi appearance representation**: partitions the directional
  domain into learnable regions with smooth boundaries
- **Semantic-adaptive material segmentation and clustering**: multiple papers
  cluster surfels and tie materials within each cluster

### Pattern C — Radar/RF specific work

- **Inverse Rendering of Near-Field mmWave MIMO Radar for Material
  Reconstruction** (IEEE Jan 2025,
  [doi 10.1109/...](https://ieeexplore.ieee.org/document/10892639/)):
  exactly our problem class. Material properties recovered from MIMO radar
  data + multi-view stereo geometry.
- **Radio-Frequency Inverse Rendering for Wireless Environment Modeling**
  ([arxiv 2604.07086](https://arxiv.org/abs/2604.07086)): "RF-aware BSDF
  embedded in Gaussian splatting; physically grounded decoupling of
  RF emission, geometry, and material EM properties"
- **URF-GS — Unified Radio-Optical Radiation Field**
  ([arxiv 2601.19216](https://arxiv.org/abs/2601.19216)): claims +24.7%
  spatial spectrum prediction over NeRF baselines via decoupled
  geometry/material modelling.
- **Radiometrically Consistent Gaussian Surfels** ([arxiv 2603.01491](https://arxiv.org/html/2603.01491)):
  per-point radiometric consistency across views.

## Available signals we haven't used

The pcl.npy file is **7-dimensional**, not 6 as the v4 trainer assumes:

```
shape: (2777247, 7), dtype: float32
  col 0: x      (10.35, 26.44)
  col 1: y      (-12.44, 12.16)
  col 2: z      (-1.05, 8.37)
  col 3: n_x    (-0.42, 0.76)
  col 4: n_y    (-1.0, 1.0)
  col 5: n_z    (-1.0, 1.0)
  col 6: intensity   (8, 3436)   ← LiDAR return intensity, NEVER USED
```

The init pipeline (`init_visible_weighted_nvs`) only reads cols 0-5
(position + normal). **Intensity (col 6) is the radar/LiDAR analog of RGB
in optical inverse rendering** — points with similar intensity probably
share material/radiometric properties. This is a strong, free, per-point
prior signal that we've been ignoring.

## Proposals (kept + new)

### Proposal 1 — Edge-preserving spatial smoothness (geometry + intensity)

The proven literature recipe, generalized to use LiDAR intensity as an
edge signal in addition to geometry.

For each point, build a kNN graph (k ≈ 8) on positions. For each edge
`(i, j)` compute a weight that's high when the two points are likely on
the same surface AND have similar physical properties:

```
w_ij = exp( -||pos_i − pos_j||² / σ_p²
            -||n_i  − n_j ||² / σ_n²
            -(I_i  − I_j )²  / σ_I² )
```

where `I` is LiDAR intensity (col 6 of pcl.npy). Add to the loss:

```
L_smooth = λ_s · Σ_ij w_ij · ||material_i − material_j||²
```

The loss only fires when w_ij is high — i.e., when neighbors are on the
same flat surface AND have similar intensity. Material discontinuities
across edges (different surfaces) or across intensity boundaries
(different physical materials) are NOT penalized. This is exactly the
"RGB-guided total variation" of DET-GS adapted to LiDAR.

**Implementation:** ~1 hour. New CLI flags: `--smooth_lambda`, `--smooth_k`,
`--smooth_sigma_p`, `--smooth_sigma_n`, `--smooth_sigma_I`. The kNN graph
is precomputed at init time (seconds). Runtime cost per iter: one k×N
gather + a sum, negligible compared to the renderer.

**Why we expect this to work where uniform smoothness might not:** the
parameter analysis showed that S1 and S2 have *equal* spatial smoothness
at every k=5/30/100 — so a uniform smoothness penalty cannot
distinguish them. But edge-preserving smoothness applies *unequal*
pressure: it heavily smooths inside surfaces (where S2 is consistent)
and barely touches material boundaries (where the intensity difference
gates the loss). This breaks the symmetry.

### Proposal 3 — Per-point contribution smoothness across train poses (physics-based)

**Physical motivation:** for any single point, its rendered contribution
to (range, azimuth) bins should vary *smoothly* as the radar pose changes
by 1 cm. The Cook-Torrance BSDF is a smooth function of incident /
exitant angles, and the geometry change is small, so the contribution at
adjacent poses should differ only by a small smooth factor. The
optimizer is currently free to pick parameters where individual points
have wildly different per-pose contributions (because nothing constrains
this), and the SUMS happen to fit the per-frame GTs.

**Implementation:** open the renderer to expose the per-point
contribution tensor (currently summed inside `render_factorized`).
For each pair of adjacent train frames, compute per-point contribution
to (TX, RX, range_bin) and penalize the L2 norm of the difference of
contributions across the pair, normalized by an expected-from-physics
smoothness scale.

A simpler version that doesn't require renderer surgery: penalize the
difference between the **active mask × material** product across adjacent
train frames. This proxy captures the physical intuition without
exposing internals.

**Implementation cost:** ~3-4 hours for the full version, ~1 hour for the
proxy. Physically the most principled, but most code.

### Proposal 4 — Half-vector consistency

**Physical motivation:** in a Cook-Torrance microfacet BSDF, the response
depends on the half-vector `h = normalize(wi + wo)` (incident + exitant
direction at each point). Two adjacent train poses produce *similar*
half-vectors at each point (small angle change). The BSDF response should
be a smooth function of `h`, with derivatives bounded by the BSDF
parameters (roughness etc.).

**Loss:** for each train frame pair, compute the half-vector at each
active point under each pose, take the angular difference Δh, and
penalize the BSDF response change beyond what a first-order Taylor
expansion of the BSDF would predict. This is a constraint on the
**model's BSDF Jacobian**, derived from the physics rather than from data.

**Implementation cost:** ~2 hours. Requires computing the half-vector and
the analytic BSDF Jacobian (or a finite-difference approximation).

### Proposal 5 — LiDAR-intensity-keyed material codebook (combines patterns A + B)

Cluster points by **(intensity, normal)** rather than by position+normal.
Use a Gaussian mixture or k-means in (intensity, normal) space. Within
each cluster, tie all points to a learnable per-cluster material vector
+ small per-point residual (penalized).

**Why intensity-keyed clustering is different from earlier k-means:** the
earlier experiment used position+normal as features. That clusters
*spatially adjacent points with similar geometry*. But spatially adjacent
points can be different materials (e.g. wall meets floor). Intensity
captures the *physical* similarity directly. Cluster centroids in
intensity space correspond to discrete material classes (high-intensity =
metal/glass, low-intensity = concrete/carpet).

**Implementation cost:** ~1-2 hours. The cluster framework already exists
(was used for k-means in pos+normal); only the feature vector changes.

### Proposal 6 — Augment edge-preserving smoothness with normal anchor in same edge graph

Combination of Proposal 1 + the failed normal anchor. Anchor each
**normal** to a smoothness target derived from the kNN edge weights:
each point's normal should be similar to the weighted average of its
kNN neighbors' normals, with weights from the same edge graph as
Proposal 1.

The previous normal anchor failed because it pulled all normals back to
init (which both S1 and S2 drift away from in similar directions). This
version anchors normals to the local average across the edge graph,
which is preserved as the optimizer learns the structured drift.

**Implementation cost:** ~30 minutes once Proposal 1 is in place.

## Proposal 1 — RESULT: NEGATIVE (and should have been predicted)

| config (Scenario 6/2, 10K, pearson, 300 iters) | train | test | f133 | f137 |
|---|---|---|---|---|
| baseline | 0.8354 | 0.5797 | 0.5626 | 0.5967 |
| edge λ=0.1 | 0.8309 | 0.5718 | 0.5433 | 0.6003 |
| edge λ=1.0 | 0.8258 | 0.5534 | 0.5105 | 0.5963 |
| edge λ=10 | 0.8179 | 0.5544 | 0.4935 | 0.6153 |
| edge λ=100 | 0.7969 | 0.5368 | 0.4686 | 0.6051 |

Train cc drops monotonically with λ AND test cc drops too. Edge smoothness
*hurts* in this regime.

**Why it failed (predictable in hindsight):** the earlier parameter-difference
analysis in `md/v4_nvs_param_analysis.md` already verified that S1 (under-
trained) and S2 (well-trained) have **identical spatial smoothness at every
scale**:

```
B/A smoothness ratio (k=10 NN std on each material column)
  eps_real:  0.94    eps_imag:   1.00    sigma_h:  1.02
  l_c:       0.97    tau_base:   1.00    thickness: 0.94
```

S1 and S2 are equally smooth. Pushing the model toward "smoother" doesn't
move it toward S2 — it moves it sideways into another equally-smooth
solution that happens to be different from both. Adding the intensity edge
weight doesn't change this: it still pushes along the smoothness axis,
which is not the axis S1 and S2 differ on.

**This also rules out Proposal 6 (normal smoothness via the same edge
graph)** — same pathology, parameters are already smooth enough on both
sides of the gap.

## Recommended priority (revised after Proposal 1 failed)

The earlier parameter-side analysis ruled out:
- Capacity (10K = 90K)
- Distributional priors (S1 and S2 have identical mean/std per column)
- Spatial smoothness at any scale (just confirmed by Proposal 1)
- L2 distance from init (S1 and S2 drift similar amounts)

What it did NOT rule out: differences in *output-space* behavior or in
*per-point dynamics across poses*. The bracket-anchor result (+0.045
test cc) was the first regularizer that touched output-space, and it
worked. The remaining proposals all live in that space.

1. ~~**Proposal 1** (edge-preserving smoothness)~~ — NEGATIVE, see above.
2. ~~**Proposal 6** (normal anchor via edge graph)~~ — ruled out by same
   logic as Proposal 1.
3. ~~**Proposal 5** (intensity-keyed material codebook)~~ — NEGATIVE. See below.
4. **Proposal 4** (half-vector consistency). Physics-derived from the
   Cook-Torrance BSDF Jacobian. Operates on output-space dynamics, not
   parameter values. Different axis — NOT RULED OUT.
5. **Proposal 3** (per-point contribution smoothness across train poses).
   Most physically principled. Operates on per-point output-space
   dynamics. Biggest code change. NOT RULED OUT.

## Proposal 5 — RESULT: NEGATIVE

Clustered points into K groups by (intensity, normal) features,
parameterized material as `cluster_center[idx[i]] + residual[i]` with
L2 penalty on residuals.

| config (Scenario 6/2, 10K, pearson, 300 iters) | train | test | f133 | f137 |
|---|---|---|---|---|
| baseline | 0.8355 | 0.5705 | 0.5400 | 0.6009 |
| K=8  λ_r=0.01 | 0.8487 | 0.5659 | 0.5461 | 0.5858 |
| K=16 λ_r=0.01 | 0.8485 | 0.5683 | 0.5450 | 0.5915 |
| K=32 λ_r=0.01 | 0.8513 | 0.5622 | 0.5139 | 0.6105 |
| K=64 λ_r=0.01 | 0.8550 | 0.5704 | 0.5290 | 0.6118 |
| K=16 λ_r=0.1 | 0.8545 | 0.5719 | 0.5426 | 0.6011 |
| K=16 λ_r=1.0 | 0.8454 | 0.5403 | 0.4788 | 0.6017 |

All configs within MC noise on test cc. Train cc actually rises (+0.01 to
+0.02) because the codebook + residual is effectively as expressive as
per-point.

**Why it fails:** the per-point residual absorbs whatever structure the
codebook would have imposed. At weak `λ_r` the model is back to per-point
freedom; at strong `λ_r` the hard clustering hurts without helping (f133
drops 0.06 at `λ_r=1.0`). The cluster assignment is fixed at init — the
model can't correct a bad grouping — and the gauge ambiguity seen on
per-point materials transfers directly to the (codebook + residual)
parameterization because the residual dimension is the gauge.

**This also effectively rules out "tied cluster materials" as a family of
approaches** — the earlier k-means in (pos, normal) also failed, and now
k-means in (intensity, normal) fails with the same pathology. The cluster-
based parameterization is not the right axis.

## Open questions

- What's the right scale for `σ_I` in the intensity edge weight?
  Intensity ranges 8-3436 with std 213. Try `σ_I ≈ 100` (half a std) as
  a starting point, then sweep.
- Should the normal anchor in Proposal 6 use `||quat - quat_local_avg||²`
  or the resolved normal-vector dot product?
- Does intensity from a static LiDAR scan match the radar's view-dependent
  return well enough to be a useful prior at 77 GHz? (Both are EM
  scattering at non-trivial wavelength but the operating frequencies
  differ by 2-3 orders of magnitude.)

## Sources

- [Inverse Rendering of Near-Field mmWave MIMO Radar (IEEE 2025)](https://ieeexplore.ieee.org/document/10892639/)
- [Radio-Frequency Inverse Rendering for Wireless Environment Modeling (arxiv 2604.07086)](https://arxiv.org/abs/2604.07086)
- [URF-GS Unified Radio-Optical Radiation Field (arxiv 2601.19216)](https://arxiv.org/abs/2601.19216)
- [Multi-view 3D surface reconstruction from SAR (arxiv 2502.10492)](https://arxiv.org/abs/2502.10492)
- [Differentiable Rendering for SAR Imagery (arxiv 2204.01248)](https://arxiv.org/abs/2204.01248)
- [Depth-Regularized Optimization for 3D GS in Few-Shot Images (arxiv 2311.13398)](https://arxiv.org/html/2311.13398v3)
- [DET-GS Depth- and Edge-Aware Regularization (arxiv 2508.04099)](https://arxiv.org/html/2508.04099v1)
- [PBR-NeRF Inverse Rendering with Physics-Based Neural Fields (arxiv 2412.09680)](https://arxiv.org/html/2412.09680v1)
- [MaterialFusion Material Diffusion Priors (arxiv 2409.15273)](https://arxiv.org/html/2409.15273v1)
- [Radiometrically Consistent Gaussian Surfels (arxiv 2603.01491)](https://arxiv.org/html/2603.01491)
- [Awesome-Inverse-Rendering paper list](https://github.com/ingra14m/Awesome-Inverse-Rendering)
- [Building Differentiable Simulators for mmWave Radar Signals (Penn 2024)](https://www.seas.upenn.edu/~richeek/blog/2024/mmwave-material-extraction/)
