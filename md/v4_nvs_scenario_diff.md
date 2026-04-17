# NVS scenario-difference analysis (working document)

## Setup

Two trainings on `seq_0_frame_135`, identical except for which frames
are in the train set. Both at:

- 10K points
- pearson loss
- 300 iters
- mat_lr=0.01, rot_lr=5e-3 (defaults)
- frame index 4 (frame 135) **dropped entirely** because its alignment
  config is misaligned (boresight Z = -0.486 vs ~0 for neighbors)

| | Scenario 1 | Scenario 2 |
|---|---|---|
| train indices | {0, 2, 6, 8} | {0, 1, 2, 3, 5, 6, 7, 8} |
| test indices | {1, 3, 5, 7} | (none, ceiling) |
| frames in train | 131, 133, 137, 139 | 131, 132, 133, 134, 136, 137, 138, 139 |
| train cart_corr (mean) | 0.8571 | 0.8170 |
| test cart_corr (mean) | **0.5146** | n/a |
| frame 132 (test in S1) | 0.5146 | **0.8083** |
| frame 134 (test in S1) | 0.5146 | **0.8440** |
| frame 136 (test in S1) | 0.5146 | **0.8471** |
| frame 138 (test in S1) | 0.5146 | **0.7973** |
| → mean of 132/134/136/138 | 0.5146 | **0.8242** |

Per-frame mean cart_corr on the 4 S1-test frames: 0.5146 vs 0.8242.
**Gap = 0.31 cart_corr** that we need the analysis to explain.

State files saved at:

```
mm25DGS_v4/output_nvs/seq_0_frame_135__param_analysis/scenario1_state.pt
mm25DGS_v4/output_nvs/seq_0_frame_135__param_analysis/scenario2_state.pt
```

Both runs share the same FPS-deterministic point ordering (positions
identical), so per-point comparisons are valid: `paramA[i]` and
`paramB[i]` describe the same physical surfel.

## Lens 1 — Per-column material distributions are identical

Statistics of `raw_materials` (the learnable, pre-reparameterize space)
are within 5-10 % between S1 and S2 for every column:

| col | S1 μ | S1 σ | S2 μ | S2 σ | \|μ₁-μ₂\| | σ ratio |
|---|---|---|---|---|---|---|
| eps_real | 4.226 | 0.264 | 4.214 | 0.295 | 0.012 | 0.90 |
| eps_imag | -4.310 | 0.447 | -4.447 | 0.434 | 0.138 | 1.03 |
| sigma_h | -10.86 | 0.958 | -10.98 | 0.966 | 0.119 | 0.99 |
| l_c | -4.937 | 0.752 | -4.893 | 0.799 | 0.044 | 0.94 |
| tau_base | -0.028 | 0.168 | -0.033 | 0.184 | 0.006 | 0.91 |
| thickness | -1.933 | 0.258 | -1.942 | 0.289 | 0.009 | 0.89 |

**Implication:** S1 and S2 reach essentially the same global material
distribution. Whatever distinguishes them is *per-point*, not
distributional.

## Lens 2 — Per-point material assignment is weakly correlated

| col | pearson(S1, S2) | spearman | \|S1-S2\| / σ |
|---|---|---|---|
| eps_real | 0.171 | 0.235 | 0.77 |
| eps_imag | 0.424 | 0.402 | 0.88 |
| sigma_h | 0.562 | 0.562 | 0.62 |
| l_c | 0.593 | 0.585 | 0.59 |
| tau_base | 0.348 | 0.534 | 0.32 |
| thickness | 0.155 | 0.218 | 0.77 |

Per-point pearson 0.16-0.59. The mean per-point disagreement is roughly
1σ — very large. **S1 and S2 converge to two different per-point
assignments that satisfy the global statistics in different ways.**

`eps_real` and `thickness` are essentially uncorrelated (0.16). These
are the two least-determined material columns under either loss.

## Lens 3 — Spatial structure of disagreement is uniform

Mean per-point disagreement (L2 across cols, σ-normalized) by x-axis bin
along the radar boresight:

| x bin | x range | n points | mean disagreement |
|---|---|---|---|
| 1 | [10.4, 13.6] | 1155 | 2.06 |
| 2 | [13.6, 16.8] | 2204 | 1.92 |
| 3 | [16.8, 20.0] | 2352 | 2.26 |
| 4 | [20.0, 23.2] | 3456 | 2.36 |
| 5 | [23.2, 26.4] | 832 | 2.15 |

Disagreement is roughly flat across space (1.9-2.4). **There is no
spatial cluster of "bad points"**. The disagreement is distributed
uniformly across the scene, which rules out simple spatial-region
masking as a fix.

## Lens 4 — Normals: shared drift direction, bimodal divergence

Per-point angular differences (degrees):

| | mean | P90 |
|---|---|---|
| S1 vs init pcl | 19.91° | 39.52° |
| S2 vs init pcl | 20.00° | 38.92° |
| **S1 vs S2** | **14.91°** | 36.87° |

Both runs drift ~20° from init on average. If they were independent
random fits, S1 vs S2 would be ~28°; it's only 15°. **There is shared
structure in the direction of normal drift.**

Drift-direction cosine (between S1 - init and S2 - init) on the 8906
points that moved more than 0.05 in unit-vector space:

| | value |
|---|---|
| mean | 0.585 |
| **median** | **0.943** |
| % with cos > 0.5 (drift same direction) | 72.8 % |
| % with cos < 0 (drift opposite direction) | **18.8 %** |

The distribution is **bimodal**: median 0.94 but mean 0.59. 73 % of
points drift in nearly the same direction in S1 and S2; the remaining
19 % drift in *opposite* directions (cos < 0).

### What distinguishes the 19 % "opposite" subset?

| | aligned (73%) | in-between (8%) | **opposite (19%)** |
|---|---|---|---|
| Mean material disagreement | 2.01σ | 2.72σ | **2.71σ** |
| Mean # train frames active | 3.65 | 3.51 | 3.62 |
| Mean init normal nz | +0.475 | +0.538 | **+0.552** |
| % with \|nz\| > 0.7 (near vertical) | 66 % | 77 % | **83 %** |
| Drift magnitude (S1) | 0.385 | 0.371 | 0.337 |
| Drift magnitude (S2) | 0.384 | 0.376 | 0.316 |

**The opposite-drift points are mostly near-vertical surfels (83 % have
\|nz\| > 0.7).** They are not in any specific spatial region (positions
span the full scene), they are NOT gradient-starved (active in 3.62 / 4
train frames on average), and they drift *less* in magnitude than
aligned points (0.34 vs 0.38). Their high material disagreement (2.71σ)
correlates with the bad normal direction.

**Geometric interpretation:** with the radar pointing horizontally,
near-vertical surfels (floor / ceiling / table tops) are seen at
grazing angles. The BSDF gradient at grazing incidence is weak and
ill-conditioned, so per-iter normal updates are dominated by
stochastic gradient noise — different across S1 and S2 because the
two losses sum different per-frame signals. The optimizer "wanders" in
random directions because the cost landscape near these points is
shallow.

## Lens 5 — Active-mask asymmetry: 1.4 % gradient-starved

For S1 (train {0,2,6,8}, test {1,3,5,7}):

| partition | n | % |
|---|---|---|
| active in train ∩ test | 9754 | 97.5 % |
| active **only in test** (gradient-starved) | 136 | 1.4 % |
| active only in train | 71 | 0.7 % |
| inactive in both | 39 | 0.4 % |

The gradient-starved points (136, 1.4 %) are **exactly at init** in S1
(per-column drift = 0.0000 — they never received gradient). At those
points, S2 has trained values — disagreement on `eps_imag` is 1.98σ
(vs 0.87σ for both-active points) and on `sigma_h` is 1.17σ (vs 0.62σ).

But 1.4 % of points cannot account for a 0.31 cart_corr gap on its own.
Combined with the bimodal-drift finding above, the picture is:

- **136 points (1.4 %)** are completely untrained in S1 → frozen at ITU
  concrete materials and pcl init normals
- **1676 points (16.8 %, the "opposite" subset)** are weakly constrained
  → gradient noise dominates, both materials and normals diverge
  significantly between S1 and S2
- The remaining ~80 % are well-constrained and reach similar (though
  per-point uncorrelated) material values in both runs

## Synthesis

The 5-frame loss has two distinct failure modes that the analysis has
isolated:

**A. Grazing-angle gauge ambiguity (the dominant effect).** ~19 % of
points are near-vertical surfels seen at grazing angles. Their BSDF
gradient is weak, so the optimizer takes noisy steps, and the noise
direction depends on which exact frames are in the train set. The same
points see very different (materials, normals) in S1 and S2 — but
neither version is "right", they're both noise-dominated.

**B. Active-mask gradient starvation (a small effect).** 1.4 % of
points active in test frames have no gradient signal in S1 train and
remain at ITU concrete + init normals. Their parameters are wrong by
1-2σ and contribute to the test cc gap, but the magnitude is bounded
by the small population.

What the analysis does **NOT** support:

- Per-column distributional reg → distributions are already matched
- Spatial smoothness reg → A and B equally smooth at all scales
  (verified in earlier param analysis)
- Spatial-region masking → disagreement is uniform across space
- Distance-from-init L2 on materials → both runs drift similar amounts
- Anchor normals to init globally → both run drift ~20° from init in
  *similar directions*, an init anchor would fight that real signal
  for the 73 % aligned points

What the analysis DOES support:

1. **Geometry-weighted normal anchor**: anchor strength inversely
   proportional to the point's geometric "facing" score across train
   frames. Strong anchor (≈ "freeze") for grazing-angle points,
   ~no anchor for facing points. Targets failure mode A directly.

2. **Init-clamping for gradient-starved points**: explicitly identify
   points active in test but not in train (cheap to compute from
   active matrix), and either (a) freeze them at init, or (b) borrow
   materials from spatially-nearest train-active points. This pulls
   them off ITU concrete and onto something locally consistent.
   Targets failure mode B directly.

3. **Down-weighting grazing-angle gradient contribution**: rather
   than anchoring, multiply each point's loss contribution by its
   facing score. Lower-information points contribute less to the
   gradient → optimizer doesn't get distracted by their noise.

## Next experiments

### Experiment 1 — Facing-score gradient weighting (option 3): NEGATIVE

| config | train | test | Δ test |
|---|---|---|---|
| baseline (Scenario 1, no reg) | 0.8575 | 0.5274 | — |
| facing p=1 (median weight 0.24) | 0.8596 | 0.5324 | +0.005 |
| facing p=2 (median weight 0.06) | 0.8496 | 0.5187 | −0.009 |
| S2 ceiling on these 4 test frames | — | **0.8242** | — |

Both p=1 and p=2 are within MC noise of baseline. **The grazing-angle
gradient noise was a symptom, not a cause.** Even at p=2 (which zeros
out 80 % of points' gradients), train cc only drops by 0.008, meaning
the high-facing 10 % of points alone carry enough signal to fit the
train frames — and test cc remains pinned at ~0.52.

This is the **second** failed regularizer informed by parameter
analysis (after the earlier material gauge-collapse + normal anchor
test in `v4_nvs_param_analysis.md`).

### Pattern — parameter-side analysis is hitting symptoms, not causes

Two cycles of "diagnose parameter differences → design regularizer" have
both failed. The differences between S1 and S2 parameters are real but
appear to be downstream of a deeper cause. The bimodal normal drift,
gradient-starved points, gauge-ambiguous per-point materials — all are
real correlates of the gap, but constraining any of them does not
break the test ceiling.

What this likely means: **the parameter-space lens is the wrong scope**.
The gap is in the *loss-landscape geometry*, not in the parameter
values themselves. Specifically, the 4-frame loss has a manifold of
parameter sets that achieve similar train loss; the 8-frame loss has a
much smaller intersection. Both the "good" (S2) and "bad" (S1)
parameter sets are valid local minima under their respective losses,
and per-parameter constraints can't distinguish them because the
loss surface has the same shape near both.

### What to look at instead — output-space lens

Rather than asking "what's different about the parameters", ask "what's
different about the **renders**". Specifically:

- For each test frame f, render with S1 params and S2 params. Where
  in the cartesian RA image do they disagree most?
- Is the disagreement in dominant scatterers (high-magnitude regions)
  or in low-energy filler?
- Do the disagreements overlap with specific point subsets (e.g., the
  19 % near-vertical, the 1.4 % gradient-starved)?

This is a different kind of analysis — it would tell us where the
actual rendered image diverges, which is closer to the cart_corr
metric we care about.

### Output-space candidates for regularization

If the output-space analysis shows specific failure modes (e.g.,
specific test poses produce specific image regions that miss), the
right reg shape might be:

1. **Pose-interpolation augmentation** (a true output-space loss):
   For each pair of adjacent train frames, synthesize an intermediate
   radar pose by interpolating positions/boresights. Render at the
   intermediate pose and require that render to equal the linear
   interpolation of the train-frame renders. This forces the model's
   pose-dependence to be smooth — directly enforcing what NVS needs.
2. **Output-space matching**: don't fit GT directly, fit to the
   *difference* between adjacent train frames. The difference is a
   purely pose-dependent signal that the model has to capture
   correctly to generalize.
3. **Cycle consistency**: render at pose P_train, perturb the
   model state slightly, render at the same pose, require similar
   output. This makes the parameter manifold more constrained without
   needing test poses.

Of these, (1) is the most defensible and the most direct test of
"can this BSDF model represent smooth pose interpolation".

### Output-space diff analysis: RESULT

For each S1 test frame, render with S1 params and S2 params separately,
re-evaluating cart_corr at the same FPS-deterministic point set:

| frame | S1 cc | S2 cc | \|S1-S2\| top-10% | \|S1-S2\| bot-90% | ratio |
|---|---|---|---|---|---|
| 132 | 0.5493 | 0.8083 | 367.4 | 45.8 | **8.0×** |
| 134 | 0.4168 | 0.8440 | 505.6 | 68.1 | **7.4×** |
| 136 | 0.6173 | 0.8471 | 350.1 | 47.9 | **7.3×** |
| 138 | 0.4748 | 0.7973 | 330.2 | 39.5 | **8.4×** |

**The disagreement is 7-8× concentrated in the top 10 % of GT energy.**
S1 and S2 produce very similar background returns, but radically
different peak intensities at the dominant scatterers. The cart_corr
metric is dominated by the strong scatterers, so this disagreement
fully explains the 0.31 cart_corr gap.

L1 distance from GT (after min-max norm) is small in both cases
(0.009-0.017), confirming both renders are "close" overall — but the
shapes of the peak distributions differ.

### Mechanism

S1 fits its 4 train frames' strong scatterers exactly because per-point
material freedom allows it. The same per-point assignment, however,
gives wrong peak intensities at TEST poses because the BSDF response of
each scatterer point depends on incidence/exitance geometry, which
changes between poses.

S2 fits 8 frames simultaneously, which constrains each point's material
to give plausible BSDF response across a wider range of incidence
angles. The result is a "compromise" set of materials that scores
slightly worse on any individual frame but generalizes across all of
them.

**The constraint we need is local pose-smoothness of the rendered
output**, not any kind of constraint on parameter values. That is what
the previous parameter-side regularizers were missing.

### Proposed regularizer — render-consistency

For each train frame f at each iter:
1. Render at the actual radar pose → r_f
2. Render at a perturbed pose r_f + ε (e.g. translate radar by 1 cm,
   rotate boresight by 1°) → r_f_perturbed
3. Add loss term: `λ_consist * ||r_f - r_f_perturbed||² / scale²`

This penalizes models whose rendered output changes sharply with small
pose changes. Forces local smoothness in pose space, which is the exact
property the test poses (~25 ms / a few cm away from train poses) need.

No new GT data is required. Implementation: add a method to Rasterizer
that returns a copy with perturbed TX/RX positions + boresight, then
render with both. ε can be calibrated to match the actual inter-frame
pose differences (~1-3 cm translation, ~1-2° boresight rotation).

### Render-consistency regularizer: NEGATIVE

| config | train | test | Δ test |
|---|---|---|---|
| baseline (Scenario 1) | 0.8595 | 0.5286 | — |
| render_consist λ=0.1 | 0.7405 | 0.4927 | −0.036 |
| render_consist λ=1.0 | 0.4937 | 0.4780 | −0.051 |
| render_consist λ=10.0 | 0.4630 | 0.4845 | −0.044 |

At λ=1 train and test cc both collapse to ~0.49 — they **converge**.
At λ=0.1 train drops 12 pp to 0.74 while test still falls slightly.

**Why it failed:** the loss `||render(pose) - render(pose+ε)||²` has its
minimum at `render = const` (pose-invariant). With *only* a smoothness
penalty and no positive signal pushing toward "the right pose-
derivative", the regularizer drags the model toward producing the same
output at every pose. The train GTs DO differ between adjacent poses —
the model SHOULD reflect that. We need a regularizer that says "be
smooth, **but as smooth as the data is**", not "be uniform".

The right form would be something like
`|| (render(pose+ε) − render(pose)) − ε·∂_pose GT ||²` — penalize the
deviation between the model's pose-derivative and the data's pose-
derivative. But ∂_pose GT requires intermediate ground truth or a
finite difference between adjacent train frames. The cleanest version
is: at each iter pick two adjacent train frames a, b, compute
`render(b) - render(a)` and require it to equal `GT_b - GT_a`. This
exists in the existing loss already (because both renders are fit to
their GTs separately) — the question is whether explicitly anchoring
the *difference* helps.

## Status (after 3 failed regularizers + output-space + parameter analyses)

We have tried:

| Regularizer | Logic | Result |
|---|---|---|
| Material gauge collapse (poly2 basis, 174 params) | Kill per-point material freedom (gauge ambiguity) | **+0.009** |
| Normal anchor to init (λ ∈ {0.05, 0.5, 5.0}) | Suppress per-point normal divergence | **flat** |
| Facing-score gradient weighting (p=1, p=2) | Suppress noisy grazing-angle updates | **+0.005 / −0.009** |
| Render-consistency (λ ∈ {0.1, 1.0, 10.0}) | Force pose-smooth output | **−0.036 to −0.051** |

The best we can do in Scenario 1 is **0.51-0.53 cart_corr on the test
frames**, vs the **0.82 ceiling** that S2 demonstrates is achievable
with the same model class.

The pattern of failures suggests that **the difference between S1 and
S2 is NOT in the parameters or the rendered outputs in any way that
parameter-side or naive output-side constraints can detect**. Both
analyses identified strong correlates of the gap (gauge ambiguity,
strong-scatterer disagreement) but constraining those correlates
does not move test cc.

The remaining hypothesis is **basin geometry**. S1 and S2 occupy
different local minima of their respective loss surfaces. S2's
minimum is narrower because it has more constraints; S1's is wider
and the optimizer lands somewhere in the corner that minimizes its
loss but doesn't generalize. No "local" constraint (per-point,
per-pose, smoothness) breaks out of that basin because the basin
*is* a valid optimum of the 5-frame loss.

## Anchored smoothness (corrected render-consistency): NEGATIVE

Penalty: `||(render_b - render_a) - (GT_b - GT_a)||² / scale²` over
adjacent train frame pairs (0,2), (2,6), (6,8). Scenario 1, 10K, pearson,
300 iters, default LRs.

| config | train | test | Δ test |
|---|---|---|---|
| baseline | 0.8598 | 0.5164 | — |
| anchored λ=0.1 | 0.8360 | 0.5277 | **+0.011** |
| anchored λ=1.0 | 0.7621 | 0.4990 | −0.017 |
| anchored λ=10 | 0.7357 | 0.4402 | −0.076 |

λ=0.1 is within noise. λ ≥ 1 actively degrades. **The mathematically
correct version of pose-derivative anchoring also fails.**

## Final regularizer score card

| Regularizer | Logic source | Result |
|---|---|---|
| Material gauge collapse (poly2 basis) | parameter analysis (gauge ambiguity) | +0.009 |
| Normal anchor to init (λ ∈ {0.05..5.0}) | parameter analysis (normal divergence) | flat |
| Facing-score gradient weighting | parameter analysis (grazing-angle noise) | +0.005 / -0.009 |
| Render-consistency (`||r-r'||²` to perturbed pose) | output-space analysis | -0.036 to -0.051 |
| Anchored smoothness (corrected) | output-space + cross-frame mathematics | +0.011 / -0.076 |

Nothing moves test cc by more than MC noise. Every regularizer was
informed by an analysis that identified a real correlate of the gap;
none of those correlates were causal.

## Final synthesis

The gap between Scenario 1 (test cc 0.51) and Scenario 2 (per-frame
cc 0.82 on the same 4 frames) is **a sample complexity gap, not a
prior-shape gap**. With only 4 train frames the loss landscape
admits many parameter sets that achieve similar train loss; the
information required to disambiguate them is *in the other 4 frames*.

No parameter-side prior, output-space prior, or cross-frame
consistency penalty we tried can substitute for that information,
because none of them is shaped exactly like "the response at the
held-out poses". They are all proxies, and proxies are too weak to
narrow the loss manifold the way more frames do.

This is consistent with the broader observation in inverse rendering:
when the unknowns (per-point materials × normals = 100K free
parameters in the 10K case, 900K in the 90K case) outnumber the
constraints (5 frames × ~1500 active pixels each ≈ 7,500 pixel
constraints), the problem is *structurally* underdetermined, and no
hand-designed prior we have found provides the right
1000-dimensional inductive bias to choose the generalizing solution.

## Honest options

1. **Accept the v4 NVS limit at ~0.51 cart_corr for Scenario 1.**
   Document it as the achievable performance under v4 with this train
   density, and lean on the single-frame inverse-rendering results
   (which are 0.94+ and physically meaningful for forward rendering at
   the trained pose).

2. **Increase train density** by using more than 4 of 9 frames per
   scene. Earlier 6-frame split gave test 0.57; 7 frames gave (per-
   frame variable) 0.58-0.65; 8 frames gave the 0.82 ceiling. Test cc
   scales with train view count more or less linearly until 6, then
   accelerates. This is genuinely more data, not a prior.

3. **Move to v5 architecture** with view-dependent latent codes,
   multi-bounce physics, or a learned per-point feature → MLP →
   material map. This adds the structural inductive bias that the
   hand-designed v4 priors couldn't provide. Multi-day project.

4. **Use cross-scene transfer**: train on all 7 scenes' all-in fits,
   build a `material = MLP(local_geometry, scene_id)` prior, then
   freeze it and fine-tune per scene. The prior is implicit in the
   trained MLP weights and may capture the right shape that we
   couldn't write by hand.
