# v5_v4 — pruning / cloning / splitting ideas

We're locked in at **mean test cc = 0.59 (M0)**. Across 11 init variants and
3 MLP-A variants, no architectural lever has broken the F±1 single-NN
ceiling (0.586). The current densification recipe is `pos_grad_amp` signal
+ jitter children + 5%/round split-and-prune for 4 rounds (iter 100→400).

This doc lists pruning / cloning / splitting candidates, ranked by
expected lift × effort.

## Anchor: what the current default does

| stage | mechanism |
|---|---|
| init | Cosine-resample → FPS to 20k from the post-visibility LiDAR pool |
| warmup | 100 iters of pure rendering (no densify), positions learnable lr=1e-5 |
| densify | iters 100, 200, 300, 400: split top-5% by `pos_grad_amp` (1000 pts) → replace bottom-5% (1000 pts) by jittered children of parents (pos σ=2 cm) |
| stop | iter 400: no more densify; iters 400→500 just refine |

The Phase A diagnostic showed top 10% of points carry **76% of total ‖∇p L‖**
even at iter 100. So 90% of the budget is doing little work. But C1's
attempt to "flatten by allocating more cells" hurt test cc (over-spread);
C1b's "1 Gaussian per top-N cells" matched M0 within noise.

## Key constraint

We're at the **F±1 single-NN copy ceiling**. The data simply does not
contain enough multi-frame coherence to enable a static-Gaussian model to
exceed 0.586 reliably. **No pruning/cloning/splitting fix can push us to
0.7** — that's an architectural / data limit. But we *might* be able to
gain a few +0.01 from smarter densification.

## Tier 1 — most likely to give a small (+0.01 to +0.02) lift

### Idea 1: Visibility-weighted opacity for "inactive" points
Instead of a hard binary `vertex_areas` mask (1.0 active / 0.0 inactive),
use a soft opacity = `cos θ_bore × cos θ_i × visibility_AND_count` — points
visible from few train poses get smaller opacity weight, not zero. Lets
the renderer naturally down-weight uncertain points without a hard cull.

* **Pros**: principled; eliminates the boundary effects of binary culling
* **Cons**: need to tune the soft-cull formula; might destabilize training
* **Effort**: 2-3 hrs

### Idea 2: Multi-criterion prune (top-Fisher AND tail-opacity)
Currently we prune bottom-5% by Fisher signal alone. Add an OR with
"opacity (or vertex_areas) below threshold across all train poses." Catches
points that are spatially redundant even when per-point Fisher is small.

* **Pros**: catches dead points that have small but non-zero Fisher
* **Cons**: small expected gain; might prune useful "background" points
* **Effort**: 2 hrs

### Idea 3: Progressive growth (start small, grow to N)
Initialize with N=5k FPS-uniform points; densify by cloning top-Fisher
parents at iters 50, 100, 150, 200 to grow to 20k. Avoids the dead-point
problem from FPS sampling at 20k upfront.

* **Pros**: every point has been "earned" via the gradient signal
* **Cons**: slower convergence; clone events disrupt training
* **Effort**: 4-6 hrs

### Idea 4: Per-region densify quota
Currently the top-5% candidates can all come from one region (a single
high-Fisher corner). Add a spatial diversity constraint: e.g., for each
densify event, take the top-K candidates *per voxel* (using the radar's
RA cell grid). This cap prevents over-concentration. Different from C1
because the constraint is on the **densify**, not the init.

* **Pros**: addresses the over-concentration failure mode of densify
* **Cons**: still operates within current static-Gaussian limit
* **Effort**: 4-6 hrs

## Tier 2 — likely neutral or marginal

### Idea 5: Mini-Splatting depth-reinit
After a render, identify pixels with high residual where no Gaussian
contributes; place new Gaussians at the depth of the highest-magnitude GT
RA bin. Mirror the Mini-Splatting idea but for radar.

* **Pros**: targeted at "where the GT says signal exists but model doesn't"
* **Cons**: implementing depth-from-RA is non-trivial (range bin → 3D
  position requires antenna pattern + multipath disambiguation)
* **Effort**: 2-3 days

### Idea 6: Gradient-weighted FPS replacement
When choosing children from the LiDAR pool (currently jitter), use FPS in
4D (xyz + per-pose-aggregate gradient). Ensures children are spatially
spread *and* high-gradient.

* **Pros**: one knob fewer than pool_knn
* **Cons**: pool_knn already lost vs jitter (-0.02 vs combo_jitter), so
  similar idea
* **Effort**: 1 day

### Idea 7: Adaptive split fraction
Schedule split frac to ramp: 5% at iter 100, 10% at 200, 25% at 300, 25%
at 400. Increases churn in later rounds when Fisher signal is more
focused.

* **Pros**: trivial to implement
* **Cons**: tested implicitly via combo_jitter, didn't help
* **Effort**: 30 min

## Tier 3 — probably not worth trying

### Idea 8: SampleNet differentiable sampler
Replace FPS with a learned MLP that picks 20k from 600k. We've shown 11
init strategies plateau — adding a 12th learned one is unlikely to break
the ceiling.

### Idea 9: Per-Gaussian latent + view-conditioned decoder (MLP-B)
We tried MLP-A (deformation MLP) and it was neutral. MLP-B has *more*
capacity and would compound the overfitting issue MLP-A v3 (PE=4) showed.

### Idea 10: Hash-encoded position MLP
Init replacement; 11 init variants already plateau.

## Recommended sequence

1. **Idea 1 (visibility-weighted opacity)** — 3 hrs — most principled, addresses a known sharp edge
2. **Idea 4 (per-region densify quota)** — 6 hrs — directly attacks the over-concentration failure mode
3. **Idea 3 (progressive growth)** — 6 hrs — only if 1 and 4 net <+0.01

If this trio doesn't move the needle by +0.02, we're at the data ceiling.
Pivot to either (a) a higher-frame-rate dataset (see
`md/datasets_paired_radar_lidar.md`) or (b) accept 0.586 as the achievable
NVS performance and write up.

## What this doc is NOT

This is not a detailed implementation plan. Each idea would need a
small design doc + smoke + 6-scene bench. Given the established plateau,
my honest expectation is **+0.005 to +0.015 mean test cc** from the
combination of all three top-tier ideas — not enough to break F±1.

The right move is probably to use this doc to inform the **paper's
ablation section** (showing what doesn't work is informative for
reviewers) rather than chase another round of variants.
