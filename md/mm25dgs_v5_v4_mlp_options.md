# v5_v4 — MLP options ranked by likelihood to break the F±1 ceiling

## What we need to break

Current best (M0, locked in): **mean test cc = 0.5897**, equal to the **F±1
single-NN copy ceiling** (0.586). Across 11 init variants we converge to
~0.58–0.59 — init has plateaued.

To progress we need something that **uses information from multiple train
frames simultaneously** in a way that isn't naive averaging. Reference
ceilings:

  * F±1 single-NN copy ceiling                        : **0.586**
  * naive_avg (0.5·(F-1 + F+1) vs F)                 : **0.654**
  * 2-NN frame-averaging ceiling                      : **~0.65**
  * User target                                       : **0.700**

We're at 0.586. Naive 2-NN averaging would already give 0.654. So **any
mechanism that combines two adjacent train frames non-trivially** would
move us toward 0.65. Pushing past 0.65 is harder — that's where the
non-static scene content lives (moving traffic, vegetation flutter, …).

## Why MLPs help here

The current model is **static** (Gaussian parameters are pose-independent;
each pose just gets rendered from the same parameter set). At 5 Hz with
slow ego-motion, the **per-frame variation in measured RA is dominated by
sub-Gaussian-scale geometric shifts and per-pose multipath shifts** — not
by changes in scene material. A static model can't capture this; an MLP
that *modulates* the Gaussian properties per pose can.

The cleanest formulation:

```
parameters_at_pose_F = base_parameters + Δ(pose_F, scene)
```

where Δ is a small MLP. At test pose F*, `Δ(pose_F*, scene)` is the
*interpolation* between the train-pose Δ values — exactly the thing
naive averaging implements coarsely.

---

## The ranked list

### Tier 1 — most likely to break F±1, justifiable effort

#### MLP-A: Pose-conditioned per-Gaussian deformation MLP
**Architecture**:
```
input  = [pose_6d, gaussian_canonical_pos]    # 6 + 3 = 9 dims
hidden = MLP (e.g. 64-64-64, ReLU + LayerNorm)
output = (Δposition (3), Δopacity (1), Δmaterial_scalar (1))     # 5 dims
```
Each Gaussian's *effective* state at pose F is `base + MLP(pose_F, base_pos)`.
The MLP learns smooth pose-conditioned deformation across the 8 train poses;
test pose interpolates naturally.

* **Mechanism**: Generalises Phase 4's discrete D matrix (per-Gaussian-per-train-view binary opacity) to continuous deformation. Phase 4 gave +0.001 in p4_combo on top of combo_jitter; the continuous version should do much better because the test pose can interpolate.
* **Pros**: Direct attack on the multi-frame combination problem; well-trodden in NeRF literature (NeRF-W, dynamic NeRF, D-NeRF); minimal architectural shift from current pipeline.
* **Risks**: MLP may overfit the 8 train poses (test pose extrapolation poor) — mitigate with strong L2 anchor on Δposition, freeze MLP for first 100 iters.
* **Expected lift**: **+0.04 to +0.07** (target: ~0.65, near naive_avg ceiling).
* **ETA**: 2 days.
* **Priority**: **Build this first.**

#### MLP-B: Per-Gaussian latent + view-conditioned decoder
**Architecture**:
```
per Gaussian i: learnable latent z_i ∈ R^D            # D = 8 typical
input  = [pose_6d, z_i]                                # 6 + D dims
output = (Δposition (3), Δopacity (1), Δmaterial (k))
```
Same shape as MLP-A but each Gaussian has its own learnable feature
conditioning the MLP. More expressive at the cost of N×D extra parameters.

* **Mechanism**: Same as A but per-Gaussian features let different Gaussians have different pose-response patterns. Useful if some scene parts (e.g., moving vegetation) need different temporal behaviour than rigid surfaces.
* **Pros**: Strictly more expressive than A. Latents can be initialized via PCA of Phase 4's D matrix to warm-start.
* **Cons**: 20k × 8 = 160k extra parameters (~+1 % of Gaussians' material params). Higher overfit risk. Slower training.
* **Expected lift**: **+0.05 to +0.08** if A works; **0** if A doesn't (latents help only if global pose conditioning is already useful).
* **ETA**: 2-3 days.
* **Priority**: After A, if A shows +0.02+.

### Tier 2 — narrower interventions, smaller expected lift

#### MLP-C: Pose-conditioned global feature + per-Gaussian projection
**Architecture**:
```
input    = pose_6d
encoder  = MLP → feature ∈ R^F      # F = 32 typical
output   = per-Gaussian Δ via learned projection: W_i @ feature + b_i
```
Cheap but limited — all Gaussians get a deformation that's a linear
combination of one shared feature.

* **Pros**: Trivially few parameters; very fast.
* **Cons**: Underfits unless scene has uniform pose response across all
  points. Probably doesn't help if A doesn't.
* **Expected lift**: **+0.02 to +0.04** if mechanism is right; less if not.
* **ETA**: 1-2 days.
* **Priority**: Skip if A is in the works — A subsumes C.

#### MLP-D: Hash-encoded position MLP (Instant-NGP-style position adjustment)
**Architecture**:
```
input  = canonical_position (3)
encoder = multi-resolution hash grid (Müller 2022)
output  = position offset (3)
```
Replaces the explicit `init_positions + L2_anchor` setup with a learnable
hash-grid-encoded position adjustment.

* **Pros**: Spatial generalization is free (hash grid generalizes across
  scene structure).
* **Cons**: Doesn't condition on pose, so doesn't help with multi-frame
  combination. More an init-replacement than a NVS lever.
* **Expected lift**: **+0.00 to +0.02**. We've shown init isn't the lever.
* **Priority**: Skip.

### Tier 3 — addresses specific confounds, not the main problem

#### MLP-E: Pose-aware antenna pattern correction
**Architecture**: small MLP from (azimuth, elevation, pose) → multiplicative
gain correction on top of the nominal antenna pattern.

* **Pros**: Targets a known confound (antenna pattern at off-boresight angles
  has multipath sidelobes the nominal pattern doesn't capture).
* **Cons**: Probably 5–10 % of total residual; not the load-bearing factor.
* **Expected lift**: **+0.01 to +0.02**.
* **ETA**: 1-2 days.
* **Priority**: Skip unless A/B already converged and we're trying to squeeze
  the last few cc points.

#### MLP-F: Image-space residual MLP
**Architecture**: rendered RA → MLP → corrected RA (per-pixel small
correction).

* **Pros**: Trivial to add, decouples from Gaussian model.
* **Cons**: Image-space corrections don't generalize to test pose well —
  they'd just memorize per-train-pose residuals. Likely high train cc
  improvement, no test cc improvement.
* **Expected lift**: **+0.00 to +0.01 test cc**, +large train cc.
* **Priority**: Skip — would inflate train metrics without test gain.

### Tier 4 — too big a change for the marginal gain

#### MLP-G: Full implicit neural radar field (NeRF for radar)
**Architecture**: f(x, y, z, pose) → reflectivity, opacity. Replace
Gaussians entirely with a continuous MLP-based radar field.

* **Pros**: No point budget issue; continuous representation; well-studied
  in NeRF literature; could in principle reach 0.7+.
* **Cons**: ~1-2 weeks of work; slow rendering; abandons all the v5_v4
  Gaussian infrastructure; rendering integration over rays is non-trivial
  for radar (must include phase + amplitude path correctly).
* **Expected lift**: Unknown. Could be **+0.05 to +0.15** but high variance.
* **Priority**: Only if A/B fail.

#### MLP-H: SampleNet-style differentiable sampler at init
**Architecture**: learn a sampler MLP that selects 20k points from the
850k AND set, end-to-end with the renderer.

* **Pros**: The principled FPS-replacement.
* **Cons**: We've now empirically shown that 11 different init strategies
  all converge to 0.58-0.59. SampleNet would just be a learned version of
  the same lever. Init isn't the bottleneck.
* **Expected lift**: **+0.00 to +0.02**.
* **Priority**: Skip.

---

## Recommended sequence

1. **MLP-A** (pose-conditioned per-Gaussian deformation MLP) — 2 days.
   This is the principled NeRF-style attack on the F±1 ceiling and the
   most likely to reach 0.65.

2. **Decision point**: if MLP-A reaches ≥ 0.62 mean test cc, proceed to
   MLP-B for an additional lift. If MLP-A is < 0.60, rethink — probably
   the static-Gaussian assumption itself is the ceiling and MLP-G (full
   INR) becomes the only path.

3. **If A succeeds → B** (per-Gaussian latents on top of A).

4. **If A succeeds + B saturates → E** (antenna-pattern correction) for
   the last 0.01-0.02.

## Hyperparameter sketch for MLP-A

* Hidden: `64 → 64 → 64` ReLU+LayerNorm, output `5` (Δp + Δα + Δmat).
* L2 anchor on Δposition: λ_pos = 1000 (cap drift to 1 mm).
* L2 anchor on Δopacity: λ_op = 100.
* MLP LR: 5e-4 (Adam).
* Warm-up: freeze MLP for first 50 iters (let Gaussian materials settle).
* Test deployment: feed test pose's 6D directly; **never** access test RA.
* Phase guard: detach phase path on Δposition (same as Phase 1 — amplitude
  gradients only).

## What's the test-time semantics?

The MLP takes the **test pose** (geometry only, never test signal) and
outputs per-Gaussian deformations. This is the same kind of test-pose-in-
inference that NeRF uses every day at the test view. Standard, defensible.
