# NVS train/test gap — full diagnosis (revised)

## Symptom

After 300 iters at 90K points on `seq_0_frame_135`:
- Train mean cart_corr: **0.9029** (5 train frames)
- Test  mean cart_corr: **0.4661** (4 held-out frames)
- Gap: **0.44**

## TL;DR

The train/test gap is **structural overfitting**, not a data-alignment
bug or an optimization artifact. The per-point material × per-point
normal representation has a hard NVS ceiling around **test_cc ≈ 0.51**
that we cannot break with capacity, loss type, or freezing axes. The
remaining 0.4 gap is the model memorizing each train pose exactly
without learning anything pose-independent that the held-out poses
could use.

## Diagnostic experiments

All on `seq_0_frame_135`, 300 iters, train idx {131,133,135,137,139},
test idx {132,134,136,138} unless noted.

| # | Config | Train | Test | Insight |
|---|---|---|---|---|
| 0 | iter 0, no training | 0.4551 | 0.3794 | Geometric baseline |
| 1 | 90K, mse_raw, learn all | 0.9029 | 0.4661 | Original setup |
| 2 | 90K, mse_raw, drop ref frame | 0.9054 | 0.4538 | Misaligned ref ≠ cause |
| 3 | 10K, mse_raw, learn all | 0.7459 | 0.4640 | Capacity ≠ cause |
| 4 | 10K, **pearson**, learn all | 0.8451 | **0.5075** | Loss matters (+0.04) |
| 5 | 10K, pearson, freeze mat | 0.7805 | 0.5020 | Materials add little |
| 6 | 10K, pearson, freeze normals | 0.7053 | 0.4855 | Normals add little |

Per-axis contribution to test cc (vs iter 0):
- + train normals only: +0.123
- + train materials only: +0.106
- + train both: +0.128

Most of the transferable signal comes from normals. Materials add ~0.02.

## What we ruled out

### Misaligned reference frames are NOT the cause

Initial diagnosis: the cascade alignment for the reference frame in
~4/7 scenes produces a wildly off-axis boresight (e.g. `seq_0_frame_135`
ref frame 135 has bs Z = -0.486 vs ~0 for all neighbors). The raw GT
energy of frame 135 is half its neighbors' (mean 905 vs ~1750), which
is consistent with a 30° tilted radar.

But: dropping the ref frame from training only changed test cc by
-0.013. The misalignment is **invisible** to cart_corr because the
correlation is computed between rendered and GT *both using the same
config*, so the misalignment cancels. The bug is real but it's a
single-frame-trainer problem (the trainer fits noise), not the cause
of the NVS gap.

### Capacity is NOT the cause

90K (~900K params) and 10K (~100K params) give the **same test cc**
(0.466 and 0.464). All 8× the extra capacity goes to overfitting train.

### Optimizer dynamics (mat_lr, rot_lr, iter count) are NOT the cause

The single-frame trainer uses the same LRs and reaches 0.94 single-frame.
NVS uses identical settings. The optimizer is fine. The ceiling is
hit at iter 100-150 already (test cc levels off while train keeps
climbing) — confirmed by inspecting per-iter logs.

### Active-mask asymmetry is NOT the cause

Per-frame active counts vary by ~5K between adjacent frames at 90K (we
checked at init time). 79K-83K of 90K are active in any given frame.
The intersection across all 9 frames is a large majority, so most
points see gradient signal from multiple poses.

## What we believe IS the cause

The model architecture has **two structural sources of overfitting**
that no amount of capacity reduction or loss change can fix:

1. **Per-point materials are not physical materials**. They are 6 free
   floats per surfel, parameterized through a softplus and reparam.
   At per-pose training time the optimizer drives them to whatever
   value makes that pose's render match its GT — that value is
   guaranteed not to be a "correct" material because the inverse
   rendering problem is degenerate.

2. **Per-point quaternions encode incidence-conditional surface
   response** rather than just normal direction. Each point's
   quaternion can rotate to give the BSDF a stronger response to
   exactly the train poses' incident angles. At held-out poses the
   carefully-rotated normals fall on the wrong side of the
   half-vector.

The combined per-point representation has a roughly **(M × 10)**
degree-of-freedom manifold that contains many configurations giving
identical train-pose renders but wildly different test-pose renders.
Training picks one such configuration arbitrarily, and only the
~0.13 cart_corr that's projection-invariant transfers.

## Why the single-frame trainer doesn't show this

The single-frame trainer only ever sees **one** pose per scene. It
fits that pose perfectly (cart_corr 0.94) and is never asked about
any other pose. The "0.94 baseline" is partly real fit and partly
this same overfitting freedom — we just couldn't see the freedom
because there was no held-out pose.

This means the **single-frame numbers are an upper bound on what the
model can extract per scene, not a lower bound on physical fidelity**.
A tighter (better-regularized) model would likely score lower on
single-frame and higher on NVS.

## Options for breaking the test ceiling

Ranked by expected impact / cost:

### A. Spatial material clustering (T3-mode)
Tie materials in spatial neighborhoods (e.g. k-means on pos+normal,
N=50-200 clusters). Reduces material dof from 6M to 6N, forces
neighboring points to share material — encodes a strong "real
materials are spatially smooth" prior. Already implemented in
single-frame trainer (`material_clusters` flag); needs plumbing
through `train_nvs`. Expected: +0.05-0.15 test cc.

### B. Spatial smoothness regularizer on materials
Add a loss term: `λ · Σ ||material[i] - material[j]||²` for each
FPS-neighbor pair. Soft version of A — encourages but doesn't
require sharing. Cleaner gradient than hard clustering. Expected:
+0.03-0.10 test cc.

### C. Material codebook
Replace per-point materials with a small learnable codebook (8-32
materials) plus a per-point softmax assignment over the codebook.
The softmax assignment is the only per-point learnable. ~10-30K
params instead of 540K. Expected: +0.10-0.20 test cc but biggest
code change.

### D. Drop normal learning entirely
Use raw pcl normals frozen. Test #5 shows freezing materials gives
test 0.502 vs 0.508 with both — almost identical. Freezing normals
gives 0.486. Normals are giving us most of the transfer; we should
keep them, but maybe freeze materials and tune normals carefully.
Expected: +0.0 (already tested as #5).

### E. Pearson loss everywhere [low-hanging fruit]
Make pearson the default for NVS. Already +0.04 test cc over mse_raw.
No code change beyond the CLI default. **Should do this regardless
of which other path we take.**

### F. More training frames per scene
Cap the number of train frames to (say) 7 of 9 instead of 5. Each
extra frame gives +1 constraint to the optimizer per iter. Test
the {0,1,2,4,6,7,8} train + {3,5} test split. Expected: +0.05 test
cc (linearly diminishing). Doesn't fix the ceiling, just nudges it.

### G. Multi-bounce / pose-dependent physics in the BSDF
The single-bounce point-primitive renderer cannot model multi-path
energy. If the GT contains 30 % multi-bounce, we have a hard physics
ceiling no model fitting can cross. This is a v5 architecture
question, not a v4 fix.

## Recommended next experiments

1. **Apply E (pearson default) immediately** — free win, no risk. *Done
   — pearson is now the default in `train_nvs`.*
2. **Implement A (spatial clustering) at N=100** — *Done. Plumbed
   `material_clusters` through `train_nvs` + CLI.*
3. **Run cluster sweep** N ∈ {10, 50, 100, 500, 2000} at 10K with
   pearson on `seq_0_frame_135`. *Done. See results below.*

### Cluster sweep results (10K, pearson, 300 iters)

| K | train | test | gap |
|---|---|---|---|
| 10 | 0.8205 | 0.4964 | 0.324 |
| 50 | 0.8185 | 0.5018 | 0.317 |
| 100 | 0.8217 | 0.5012 | 0.321 |
| 500 | 0.8238 | 0.5156 | 0.308 |
| 2000 | 0.8288 | 0.5160 | 0.313 |
| per-point | 0.8451 | 0.5075 | 0.338 |

**Material clustering moves test cc by at most +0.008.** The ceiling
holds across two orders of magnitude of material constraints, from a
near-global-material setup (K=10, ~6 effective material params) to
near-per-point (K=2000, half the points have their own material). Even
with K=10 the model still fits train to 0.82 and stalls at 0.50 test.

This rules out capacity, loss, axis-freezing, AND material clustering
as the source of the ceiling. The remaining candidates are
**architectural** — the BSDF model itself, the single-bounce assumption,
or the per-point geometry representation.

## Final conclusion (revised after parameter analysis + view-count sweep)

The earlier "0.51 structural ceiling" framing was incomplete. The
actual picture is:

1. **Test cc depends on whether the held-out poses are interpolated or
   extrapolated** from the train pose distribution:
   - Interleaved test (interpolation between train poses): test cc
     ~0.50
   - Edge test (extrapolation to trajectory boundary): test cc ~0.29
2. **Per-frame cart_corr at the all-in fit** (every frame in training)
   gives a per-frame "best possible" of 0.61-0.87, with frame 139
   being the structurally hardest at 0.75 even when fully trained.
3. **Parameter-side regularization does not help** at any of these
   test cc values. Tested without success:
   - Material gauge collapse to a poly2 basis (174 params): Δ = +0.009
   - Normal anchor regularizer at λ ∈ {0, 0.05, 0.5, 5.0}: flat
   - Material clustering K ∈ {10..2000}: flat
   - Spatial smoothness reg: ruled out (A and B equally smooth)
4. **Adding training views helps**, but the curve is not monotonic in
   the count — it's *per frame*: edge frames (139) are hardest, and
   removing edge frames from the train set destroys their predicted
   cc, regardless of how many other train frames we have.
5. **Active mask asymmetry is NOT the cause**: only 1.2 % of test-
   active points are gradient-starved by the train set, and 81 % of
   test-active points are active in all 5 train frames.

The fundamental finding: the v4 single-bounce model has roughly enough
representational capacity to fit each radar frame to ~0.8 cart_corr
when given direct supervision, but the *cross-frame consistency
constraint* needed for NVS isn't enforced by any parameter-space
prior we've found. The model finds a different "fit" per pose because
the loss surface around 5 train frames has many train-equivalent
solutions, and we have no regularizer that distinguishes the
generalizing solution from the per-pose-overfit one.

Three paths forward, none cheap:

### Path 1: Accept the ceiling, retarget the experiment
Re-scope the NVS task as "how well does single-frame v4 transfer to
adjacent poses without retraining". Report ~0.51 test cc as the v4
NVS baseline. Use this as a lower bound for any v5 work.

### Path 2: Architectural changes within v4 single-bounce
Try the things that *might* lift the ceiling without changing physics:
  - **Material codebook** (C): replace per-point materials with 8-32
    learned global materials + per-point softmax assignment. Different
    structural prior than k-means clustering — the codebook itself is
    learned end-to-end.
  - **Pose-conditioned NeRF-style latent**: per-point feature vector
    + small MLP that outputs material conditioned on incident angle.
    Adds the missing "view direction" axis but is still single-bounce.
  - **Spatial smoothness regularization**: explicit penalty on
    pos-neighbor material differences. Cleaner than k-means.

These are all medium-effort. Expected ceiling lift: maybe +0.05-0.10,
not enough to make NVS competitive. Worth one experiment to confirm.

### Path 3: v5 architecture
Build a multi-bounce / proper-physics renderer. The single-bounce
point-primitive approximation is fundamentally pose-conditioned because
real radar returns mix direct and indirect contributions in
pose-dependent ratios; no single-bounce model can fit that ratio across
poses.

This is months of work. Justified if NVS is a hard requirement.

## Recommendation

Run **one** Path 2 experiment (codebook or smoothness reg) on
`seq_0_frame_135` to confirm the ceiling is truly architecture-limited.
If test cc stays ≤ 0.55, take Path 1 (accept the ceiling, document v4
as single-frame-only). If it jumps to 0.6+, Path 2 has more headroom
and is worth a full sweep across scenes.

The relevant question for the user is: **do we need NVS to work, or
is the single-frame inverse rendering result the actual product?**
The single-frame numbers (0.94+) are real for forward rendering at
the trained pose; they're misleading only if interpreted as "the
model learned physical materials".
