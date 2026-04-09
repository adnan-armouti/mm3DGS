# Closing the C3 Gaussian Training Gap

**Status**: C3 (Gaussian surfels, mmIR material init) achieves mean cart_corr gap of 0.107 vs mmIR across 7 scenes. Stage B (mesh-vertex training using the same PyTorch physics) matches mmIR within 0.006. The gap is not in the physics — it's in how the Gaussian renderer samples the scene.

## Root cause diagnosis

Eight concrete differences between Stage B (gap 0.006) and Stage C3 (gap 0.107) were identified. Ranked by estimated impact:

| # | Difference | Stage B (works) | Stage C3 (gap) | Est. impact |
|---|-----------|-----------------|----------------|-------------|
| 1 | **Train/eval mismatch** | All 24K hits every iter | Random 4K mini-batch for train, all for eval | **Dominant** |
| 2 | **Area weights** | MC importance weights: 1/(pdf × n_attempted) | Geometric vertex areas × opacity | High |
| 3 | **Hit positions** | Reservoir-sampled surface points (bary interp) | Raw mesh vertex positions | High |
| 4 | **Shadow test** | Per-path TX shadow rays via Mitsuba | Disabled (skip_shadow=True) | Medium |
| 5 | **Antenna patterns** | Learnable (lr=0.05) | Frozen from mmIR | Medium |
| 6 | **Culling** | Reservoir returns only truly visible hits | Superset (all 3 vertices of any hit triangle) | Low-medium |
| 7 | **Normal parameterization** | Direct (N,3) param, post-step normalize | Quaternion → rotation matrix → 3rd column | Low |
| 8 | **Extra parameters** | 3 groups (materials, normals, patterns) | 5 groups (+ positions, rotations, scales, opacity) | Low |

---

## Options

### Option A: Fix the mini-batch train/eval mismatch (address #1)

**What**: Render ALL active Gaussians during training (not a random 4K subset). Use gradient accumulation across micro-batches to stay within GPU memory: render chunks without graph, accumulate the full ADC in a detached buffer, then do ONE differentiable forward pass using the cached weights.

**Concretely**: 
1. Forward pass 1 (no_grad): render all active Gaussians, get full ADC
2. Compute loss on full ADC vs GT
3. Forward pass 2 (with grad): render the SAME Gaussians again (or a large deterministic subset), backward through that
4. Or: use "surrogate loss" — compute per-Gaussian weight in no_grad pass, then do one differentiable pass with those weights frozen

**Advantages**:
- Eliminates the dominant gap source directly
- Training and evaluation see the same rendered RA image
- Gradient signal is consistent — optimizing a Gaussian to reduce loss actually reduces loss
- Conceptually simple fix

**Disadvantages**:
- Requires careful memory management (gradient accumulation or two-pass approach)
- Two forward passes per iteration doubles wall-clock time
- With 12K active Gaussians, the full graph may still be tight on 24 GB

**Fit for rasterization**: Neutral. This is an optimization engineering issue, not an architectural choice.

---

### Option B: Use reservoir-sampled hit positions as Gaussian evaluation points (address #1, #2, #3 simultaneously)

**What**: Keep the reservoir sampler in the training loop. Each iteration, run the sampler to get ~24K hit positions with PDFs. For each hit, find the nearest Gaussian(s) and interpolate their parameters (materials, normals) at the hit position using a kernel function (e.g., Gaussian RBF weighted by proximity). Use the MC importance weights (1/pdf) as area weights. This makes the Gaussian forward pass identical to Stage B's forward pass, except materials come from nearby Gaussians instead of mesh vertices.

**Concretely**:
1. Run reservoir sampler → 24K hits with (position, barycentric, pdf, n_attempted)
2. For each hit, find K nearest Gaussians (K=1 or K=3)
3. Interpolate materials from those Gaussians (distance-weighted or barycentric)
4. Render using hit positions + interpolated materials + MC weights (exactly like Stage B)
5. Backprop gradients through interpolation weights to Gaussian parameters

**Advantages**:
- Directly reuses the proven Stage B rendering pipeline (which achieves gap 0.006)
- Importance sampling provides physically correct RA weighting
- Shadow test is implicitly handled (sampler only returns visible hits)
- Expected to nearly match Stage B quality immediately
- Gradients flow to Gaussian parameters through differentiable interpolation

**Disadvantages**:
- Requires the reservoir sampler (DrJit/Mitsuba) in every training iteration — expensive (~1s per call) and non-differentiable for position gradients
- Gaussian positions can't be optimized through the sampler (positions change → different hits, but this isn't differentiable)
- Tightly couples the Gaussian system to mmIR's sampling infrastructure
- Not a pure rasterization approach — still relies on ray tracing for hit generation

**Fit for rasterization**: **Poor**. This re-introduces ray tracing into the training loop, which defeats the architectural goal of moving to a rasterization-based renderer. The Gaussians become a material interpolation layer on top of ray-traced hit positions, not an independent scene representation.

---

### Option C: Compute proper per-Gaussian importance weights analytically (address #2)

**What**: Replace geometric vertex areas with analytically computed radar importance weights. For each Gaussian, compute the solid angle it subtends from the radar array, weighted by the cosine factor and antenna gain. This approximates the MC importance weight without needing the reservoir sampler.

**Concretely**:
For Gaussian i at position μ_i with area A_i and normal n_i:
```
w_i = A_i × |cos(θ_i)| × G_antenna(dir_i) / d_i²
```
where θ_i is the angle between n_i and the radar direction, G_antenna is the antenna gain toward μ_i, and d_i is the distance. This is the "view factor" from radiative transfer — the fraction of solid angle that Gaussian i occupies as seen from the radar.

**Advantages**:
- Pure rasterization — no ray tracing needed
- Physically motivated: approximates the radar cross-section contribution of each Gaussian
- Differentiable w.r.t. Gaussian parameters (position, normal, area)
- Fast to compute (one pass over all Gaussians)
- Addresses the area weight mismatch directly

**Disadvantages**:
- Doesn't handle occlusion (Gaussian behind another Gaussian still gets full weight)
- Antenna gain computation adds overhead per-Gaussian
- Approximation quality depends on scene geometry (works well for convex scenes, worse for self-occluding scenes)

**Fit for rasterization**: **Good**. This is the natural rasterization-native replacement for MC importance weights. View-factor weighting is the standard approach in real-time rendering for determining surface contribution.

---

### Option D: Enable shadow testing for Gaussians (address #4)

**What**: Re-enable the Mitsuba shadow test for the Gaussian forward pass. After running the reservoir sampler (to get `_mi_scene`), keep the scene in memory and run shadow rays for each (Gaussian, TX) pair.

**Concretely**:
- Don't free `_mi_scene` after reservoir sampling
- Set `skip_shadow=False` in `render_gaussians`
- Reduce `chunk_size` to keep per-chunk shadow test memory manageable

**Advantages**:
- Removes spurious energy from occluded Gaussians
- Exact same shadow test as Stage B
- Simple code change

**Disadvantages**:
- Shadow test is non-differentiable (Gaussian positions can't be optimized through it)
- Mitsuba scene occupies ~8-10 GB GPU memory, leaving only ~14 GB for training
- Shadow test involves CPU↔GPU data transfer per chunk (slow)
- With 12K Gaussians × 192 MIMO paths, shadow test is ~4.6M rays per iteration

**Fit for rasterization**: **Poor**. Shadow testing via ray tracing contradicts the rasterization paradigm. A rasterization-native alternative would be depth-buffer occlusion or surfel-to-surfel visibility precomputation.

---

### Option E: Make antenna patterns learnable in C3 (address #5)

**What**: Add antenna pattern parameters (tx_E, tx_H, rx_E, rx_H) to the C3 optimizer, matching Stage B.

**Advantages**:
- Simple change (add parameters to optimizer, inject each iteration)
- Gives the optimizer an extra degree of freedom to compensate for sampling differences
- Stage B uses this and it helps

**Disadvantages**:
- Won't close the gap alone (patterns compensate for ~0.01-0.02 at most)
- Risk of overfitting patterns to compensate for structural rendering differences

**Fit for rasterization**: Neutral. Antenna patterns are sensor parameters, not a rendering architecture choice.

---

### Option F: Tighten culling to match reservoir visibility exactly (address #6)

**What**: Instead of marking all 3 vertices of every hit triangle as visible, use a more precise criterion: only include vertices that are directly visible from at least one RX element (via the hit barycentrics). Weight each vertex by the number of times it appears in the reservoir sample.

**Concretely**:
- For each reservoir hit, compute barycentric contribution to each of the 3 vertices
- Sum contributions per vertex → this gives a "visibility importance" per vertex
- Only include vertices with importance above a threshold (e.g., top-K by importance)
- Use the importance sum as the area weight instead of geometric face area

**Advantages**:
- Closer to the reservoir sampler's actual hit distribution
- Vertex importance weights approximate the MC weights
- No extra computation beyond what's already done in `_prepare_hit_data`

**Disadvantages**:
- Still vertex-based (not surface-point-based)
- Importance depends on the stochastic reservoir sample (varies with seed)
- Only an approximation of Option C

**Fit for rasterization**: **Good**. This is essentially pre-computing rasterization-style visibility weights from a one-time ray-traced sample.

---

### Option G: Remove mini-batch, use gradient accumulation (address #1 directly)

**What**: Instead of rendering a random 4K subset, render ALL active Gaussians but accumulate gradients across micro-batches. Each micro-batch of 200 Gaussians produces a partial ADC. Detach the partial ADC from the graph, add to a running sum. After all micro-batches, compute loss on the full ADC. Then do a SECOND pass (with grad) on a single micro-batch for the actual backward.

This is a "straight-through estimator" variant: the loss is computed on the full render (accurate), but gradients flow through only one micro-batch (approximate but unbiased in expectation).

**Concretely**:
```python
# Forward: full render (no grad)
with torch.no_grad():
    adc_full = sum of all chunks

# Backward: one chunk with grad, scaled loss
chunk_adc = render_one_chunk(random_chunk, with_grad=True)
# Scale so gradient magnitude matches full render
scale = n_active / chunk_size
loss = compute_loss(chunk_adc * scale, gt) 
loss.backward()
```

**Advantages**:
- Training loss matches evaluation (no train/eval mismatch)
- Memory-bounded (only one chunk in the graph)
- Each Gaussian gets gradient updates every `n_active/chunk_size` iterations on average
- Simple to implement

**Disadvantages**:
- Gradient is noisy (only from one chunk)
- The scaling heuristic (`n_active / chunk_size`) is approximate
- Convergence may be slower than full-batch gradient

**Fit for rasterization**: **Good**. This is a standard SGD approach used in all rasterization-based differentiable renderers (3DGS, NeRF, etc.).

---

### Option H: Hybrid reservoir-Gaussian approach (address #1, #2, #3)

**What**: Use the reservoir sampler ONCE at init to establish hit positions and weights. Cache these. During training, evaluate Gaussian materials at the cached hit positions using nearest-Gaussian interpolation. This combines Stage B's sampling with Stage C's learnable Gaussians.

**Concretely**:
1. At init: run reservoir sampler, cache hit positions, barycentrics, PDFs
2. Build a mapping: for each hit, find the K nearest Gaussians and store interpolation weights (precomputed, fixed)
3. Each training iteration:
   - Interpolate materials from Gaussians at cached hit positions (differentiable)
   - Render using cached positions + interpolated materials + MC weights (same as Stage B)
   - Backprop through interpolation to Gaussian material parameters

**Advantages**:
- Exactly matches Stage B's rendering for a given set of hits
- Expected to achieve gap ~0.006 (same as Stage B) immediately for materials
- Gaussian material gradients are exact (interpolation is differentiable)
- Reservoir sampler only runs ONCE (not every iteration)
- Cached hit positions are fixed → deterministic training

**Disadvantages**:
- Gaussian positions can't be optimized (would change the interpolation mapping)
- Need to rebuild the mapping if Gaussians are added/removed (density control)
- Memory cost of storing hit-to-Gaussian mapping (~24K hits × K neighbors)
- Still uses the reservoir sampler at init (but this is already done for culling)

**Fit for rasterization**: **Moderate**. The init uses ray tracing, but training is pure rasterization (interpolation + forward pass). This is analogous to how 3DGS initializes from SfM points (structure from motion = ray tracing) then trains via rasterization.

---

### Option I: Depth-sorted accumulation with early termination (address #4, #6)

**What**: Sort Gaussians by distance to the radar. Accumulate contributions front-to-back with a transmittance factor (like classical 3DGS alpha compositing). Gaussians behind high-opacity Gaussians contribute less — a rasterization-native alternative to shadow rays.

**Concretely**:
- Sort Gaussians by distance to radar center
- For each TX-RX pair, accumulate phasor contributions with transmittance:
  ```
  T_i = prod(1 - alpha_j) for j < i  (all Gaussians closer than i)
  contribution_i = T_i × alpha_i × weight_i × phasor_i
  ```
- This naturally attenuates contributions from behind other Gaussians

**Advantages**:
- Rasterization-native occlusion handling (no ray tracing)
- Differentiable w.r.t. opacity and position
- Provides meaningful gradients for opacity (learn to be transparent if occluded)
- Standard 3DGS technique, well-understood

**Disadvantages**:
- Per-TX-RX sorting is expensive (192 sorts per iteration, each over 12K Gaussians)
- 1D distance sorting doesn't perfectly model 3D occlusion (Gaussian A can be closer but not actually in front of Gaussian B from a specific TX-RX pair)
- Adds complexity to the phase accumulation loop
- The transmittance model may not be physically appropriate for radar (mmWave penetration differs from optical opacity)

**Fit for rasterization**: **Excellent**. This is the canonical rasterization approach to visibility. It's exactly what 3DGS does for optical rendering. Adapting it for radar phasor accumulation is the natural extension.

---

### Option J: Learned per-Gaussian importance weights (address #2, partially #4, #6)

**What**: Add a learnable scalar `log_importance` parameter per Gaussian, trained jointly with materials. This weight modulates each Gaussian's contribution to the ADC — the renderer learns *how much* each Gaussian should contribute, rather than relying on analytically prescribed weights (Option C) or heuristic vertex areas.

**Concretely**:
- Add `log_importance` as an `nn.Parameter` of shape `(N,)` to `GaussianSurfels`
- The effective area weight becomes: `w_i = exp(log_importance_i) × opacity_i`
- Initialize from the analytical weight (Option C formula) or from the reservoir sampler's per-vertex hit frequency, so the starting point is physically grounded
- Train with lr comparable to opacity (e.g., 0.05), with RMS clipping

**Why this is distinct from opacity**: Opacity in 3DGS controls alpha-compositing transmittance (how much a Gaussian occludes what's behind it). Importance controls the *amplitude scale* of a Gaussian's radar return. A Gaussian can be fully opaque (blocks everything behind) but have low importance (small radar cross-section), or vice versa. Conflating them into one parameter forces a single scalar to serve two physical roles.

**What it learns**: The importance weight absorbs several effects that are hard to compute analytically:
- **Effective radar cross-section**: how much energy a surface patch at this location actually returns (depends on orientation, material, multi-path effects)
- **Sampling density correction**: if the Gaussian spacing is non-uniform, importance weights compensate (analogous to the MC 1/pdf correction in Stage B)
- **Implicit occlusion**: Gaussians that are partially shadowed learn lower importance, providing a soft approximation to shadow rays without ray tracing
- **Antenna pattern interaction**: the combined effect of TX and RX gain patterns at each location

**Advantages**:
- Pure rasterization — no ray tracing, no analytical formulas to get right
- Subsumes Option C (can learn the analytical weight) and partially Option I (soft occlusion)
- Adds only 1 parameter per Gaussian (negligible memory)
- Fully differentiable — gradients from the RA loss directly teach each Gaussian how important it is
- The loss naturally supervises importance: if a Gaussian's weight is too high, the rendered RA overshoots the GT at the corresponding range bin, producing a gradient that reduces the importance
- Works for any Gaussian distribution (mesh-init, LiDAR-init, density-controlled)

**Disadvantages**:
- Risk of overfitting: with N free importance weights, the optimizer might find a degenerate solution where a few Gaussians dominate and the rest collapse to zero (similar to mode collapse). Regularization may be needed (e.g., entropy penalty on the importance distribution, or a prior that importance should be smooth across neighbors)
- Doesn't provide physical interpretability — the learned weight is a black-box correction factor
- Initialization matters: starting from uniform weights (all 1.0) may converge slowly; starting from analytical weights (Option C) gives a warm start
- If importance absorbs too much of the signal, material gradients may be suppressed (the optimizer might prefer adjusting importance over materials to reduce loss)

**Mitigation strategies**:
- Initialize from Option C analytical weights (warm start)
- Use a lower learning rate than materials (e.g., 0.01 vs 0.5) so materials are adjusted first
- Add an L2 regularizer toward the initial analytical weight: `lambda × ||log_w - log_w_init||²`
- Periodically reset importance to analytical values (similar to opacity reset in 3DGS)

**Fit for rasterization**: **Excellent**. Learned per-primitive weights are the natural rasterization paradigm — this is exactly how differentiable rasterizers handle the "which primitives matter" question when there's no ray tracer to importance-sample for you. It's the learned analog of the analytical importance weights (Option C), with strictly more expressive power.

---

## Recommendation

**Immediate fix (highest impact, lowest risk)**: Combine **Option G** (gradient accumulation to fix train/eval mismatch) + **Option C** (analytical importance weights) + **Option E** (learnable patterns).

This combination:
1. **Option G** eliminates the dominant gap source (#1) — training and eval see the same RA
2. **Option C** replaces heuristic vertex areas with physically motivated weights (#2) — pure rasterization
3. **Option E** is a trivial change that adds ~0.01-0.02 improvement (#5)

Expected gap after these three: **0.03-0.05** (from 0.107), based on:
- Stage B shows the physics pipeline can match mmIR to 0.006
- The remaining gap would come from vertex positions vs surface points (#3) and no shadow test (#4)

**Medium-term (best rasterization-native architecture)**: Add **Option I** (depth-sorted alpha compositing) + **Option F** (tighter visibility-based culling). This gives a proper rasterization-based occlusion model and better initial weights.

**If gap remains > 0.05**: Use **Option H** (hybrid cached reservoir + Gaussian interpolation) for materials, which should achieve ~0.01 gap by matching Stage B's rendering exactly. This is the most pragmatic path but less "purely rasterization".

**Avoid**: Option B (reservoir sampler in training loop) — it re-introduces ray tracing into every iteration, defeating the purpose of the rasterization architecture. Option D (Mitsuba shadow test) has the same problem.

---

## Summary table

| Option | Addresses | Gap reduction | Memory safe | Rasterization fit | Complexity |
|--------|-----------|--------------|-------------|-------------------|------------|
| A: Full render training | #1 | ~0.04 | Risky | Neutral | Medium |
| B: Reservoir in loop | #1,2,3 | ~0.09 | OK | **Poor** | Medium |
| C: Analytical weights | #2 | ~0.02 | Yes | **Good** | Low |
| D: Shadow test | #4 | ~0.01 | Risky | **Poor** | Low |
| E: Learnable patterns | #5 | ~0.01 | Yes | Neutral | Trivial |
| F: Tighter culling | #6 | ~0.01 | Yes | **Good** | Low |
| G: Grad accumulation | #1 | ~0.04 | Yes | **Good** | Medium |
| H: Hybrid cached | #1,2,3 | ~0.08 | Yes | Moderate | Medium |
| I: Depth-sorted alpha | #4,6 | ~0.02 | Yes | **Excellent** | High |

**Recommended combination: G + C + E** (immediate), then **I + F** (medium-term).
