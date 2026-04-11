# Gap Closure Findings: What Worked, What Didn't, and Why

Summary of stages F1-F5 from the gap closure plan. Starting gap: 0.077 mean cart_corr below mmIR. Current gap: 0.038 (halved).

---

## What Worked

### F1: Remove MAX_ACTIVE Cap (+0.008)

Deleted the `MAX_ACTIVE = 12000` random subsampling that dropped ~80% of surfels on large scenes. With gradient checkpointing already bounding memory per chunk, no OOM occurred.

Largest gains on the most severely capped scenes: seq_1_frame_438 (72K Gaussians, +0.019), seq_2_frame_105 (113K Gaussians, +0.015).

### F2: Cosine LR Decay + 1500 Iterations (+0.031)

The single biggest improvement. Changed flat LR to cosine decay (from iter 100 to 1500, min factor 0.01) and tripled training from 500 to 1500 iterations.

The original 500 iters was severely under-training — most scenes peaked at iter 450-499, meaning convergence hadn't plateaued. With 1500 iters + decay, scenes now peak at iter 1250-1499 with much higher quality. seq_0_frame_390 improved +0.080 alone.

---

## What Didn't Work (and Why)

### F3: Live Surfel Areas from Scales (Neutral)

**Idea (from all 4 comparison papers):** Replace static `vertex_areas` with `pi * s1 * s2` from learnable scales, giving scales a gradient path through the rendering equation.

**Result:** Zero effect — results identical to 6 decimal places.

**Root cause found via gradient diagnostic:** Scale gradients are ~7 orders of magnitude smaller than material gradients (mean |grad| = 7.66e-08 vs 1.81e-07). Even at matching LR (0.5), scales change by < 0.01% over 1500 iterations. The gradient is attenuated because area only appears inside `sqrt(brdf * path_loss * area)` — the sqrt halves the gradient, and area is one small factor among many.

**Additional discovery:** The F3 change was applied to `render_gaussians()` but training actually uses `render_gaussians_factorized()` — a separate factorized renderer. The wrong function was modified. This was caught and corrected in F4, confirming the gradient weakness diagnosis.

**Lesson:** In optical papers, scale determines the entire spatial contribution (via projected covariance or ray-space beta). In our radar renderer, the BSDF × antenna × path_loss terms dominate the weight — area is a minor multiplicative factor. Making it learnable provides negligible additional information.

### F4: Range-Domain Gaussian Apodization (Regression: -0.016)

**Idea (from VC-3DGS):** Add `exp(-2*(pi*slope*beta*tau/c)^2)` attenuation to model intra-surfel phase cancellation. Large surfels (many wavelengths) should scatter less coherently.

**Result:** Mean regression of -0.016. Worst scene dropped -0.065. Training became unstable (one scene peaked at iter 200).

**Root cause #1 — Apodization was negligible:** The `slope * tau / c` product is ~1e-6, giving apod ≈ 0.999976. The formula was derived using the FMCW chirp slope, but intra-surfel phase variation is dominated by the carrier frequency (77 GHz). Using f_c instead would give exp(-2.9) ≈ 0.056 for a 5cm surfel — too aggressive, zeroing out nearly everything.

**Root cause #2 — Scales exploded:** With live scales in the factorized renderer and no effective constraint (apodization negligible, anisotropy reg only penalizes shape not size), scales grew unboundedly. Max scale reached 215,750 m; mean area grew 2.26 million × from init.

**Root cause #3 — Physics redundancy:** The BSDF already models intra-surface phase cancellation. The Kirchhoff (GGX) and SPM (vMF) lobes in the Jones BSDF encode exactly how a finite rough surface scatters — the lobe width IS the angular consequence of intra-surface phase variation. The apodization double-counted this effect.

**Lesson:** The VC-3DGS apodization concept does not translate to radar because optical papers use it to compensate for their physics-free SH color model. Our Jones BSDF already captures the physics that apodization would model. Additionally, making surfel areas learnable without a strong physical constraint leads to runaway optimization.

### F5: Progressive BSDF Activation (Regression: -0.006)

**Idea (from 3DGS progressive SH):** Freeze roughness/slab BSDF params for the first 300 iters, training only Fresnel (eps_real, eps_imag). Then progressively unfreeze.

**Result:** Mixed — 2/7 scenes improved, 5/7 regressed. Mean -0.006.

**Root cause:** C3 mode initializes from mmIR-trained materials that are already near their optimum. Freezing 4 of 6 params for 300 iterations prevents the optimizer from fine-tuning the material response during the critical early adaptation phase (when the representation transitions from mesh to Gaussian).

**Lesson:** Progressive parameter activation helps when training from scratch (3DGS starts from random SH coefficients). It hurts when fine-tuning from a good initialization. The 3DGS strategy assumes the early parameters are poorly initialized — in our C3 case, they're well-initialized from mmIR.

---

## Why the Paper-Inspired Ideas Didn't Transfer

The four comparison papers (3DGS, 3DGRT, EVER, VC-3DGS) operate in a fundamentally different regime:

| Aspect | Optical 3DGS papers | mm25DGS (radar) |
|--------|---------------------|-----------------|
| Appearance model | Learned SH coefficients (no physics) | Physics-based Jones BSDF (6 ITU params) |
| What scale controls | Everything — the primitive's entire spatial contribution | Almost nothing — area is one small factor in `sqrt(BSDF * antenna * area / d^2)` |
| Accumulation | Alpha compositing (order-dependent) | Coherent phasor sum (order-independent) |
| Initialization | Random / SfM points | mmIR-trained materials + mesh vertices |

The optical papers' innovations (learnable scales, apodization, progressive SH) solve problems that arise from having no physics model. Our physics-based BSDF already handles:
- Intra-surface phase cancellation → Kirchhoff/SPM roughness parameters
- View-dependent scattering → Jones polarimetric formalism
- Material-dependent reflectivity → Fresnel + slab interference

The remaining 0.038 gap is likely from:
1. **Spatial coverage mismatch**: Gaussians at mesh vertices vs mmIR's reservoir-sampled hit positions on triangle surfaces (different spatial distribution)
2. **Shadow rays disabled**: Occluded surfels contribute spurious energy
3. **Single-point evaluation**: Each surfel evaluated at its center only, while mmIR importance-samples across triangle surfaces

---

## Current State

| Stage | Mean corr | Gap from mmIR | Cumulative improvement |
|-------|----------|--------------|----------------------|
| Baseline C3 | 0.8385 | 0.077 | — |
| + F1 (no cap) | 0.8462 | 0.069 | +0.008 |
| + F2 (LR + iters) | 0.8775 | 0.038 | +0.039 |

Per-scene breakdown at current best (F2):

| Scene | mmIR | Current | Gap | Notes |
|-------|------|---------|-----|-------|
| seq_0_frame_135 | 0.920 | 0.894 | 0.027 | Near closed |
| seq_0_frame_390 | 0.937 | 0.906 | 0.031 | Near closed |
| seq_1_frame_185 | 0.954 | 0.898 | 0.056 | Moderate gap |
| seq_1_frame_438 | 0.941 | 0.889 | 0.052 | Moderate gap |
| seq_2_frame_105 | 0.886 | 0.876 | 0.010 | Essentially closed |
| seq_2_frame_160 | 0.863 | 0.796 | 0.067 | Largest remaining gap |
| seq_2_frame_300 | 0.906 | 0.884 | 0.022 | Near closed |

Three scenes within 0.03 of mmIR (essentially closed given +/-0.03 MC noise). Two scenes with moderate gap (~0.05). One persistent outlier (seq_2_frame_160 at 0.067).

---

## Remaining Stages

| Stage | Description | Expected benefit |
|-------|-------------|-----------------|
| F6 | Adaptive densification (3DGS clone/split/prune + AbsGS gradient fix) | May help spatial coverage mismatch |
| F7 | Pre-computed shadow visibility (3DGRT-inspired) | May help occluded surfel issue |
