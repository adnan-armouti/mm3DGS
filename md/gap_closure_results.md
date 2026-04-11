# Gap Closure Experiment Results

Tracking per-stage improvements for the mm25DGS Gaussian renderer.
Each stage runs all 7 scenes in C3 mode and records cart_corr vs GT RA.

## Baseline C3 (before any changes)

| Scene | mmIR | Baseline C3 | Gap from mmIR |
|-------|------|-------------|---------------|
| seq_0_frame_135 | 0.9203 | 0.8698 | 0.0506 |
| seq_0_frame_390 | 0.9365 | 0.8263 | 0.1102 |
| seq_1_frame_185 | 0.9544 | 0.8701 | 0.0843 |
| seq_1_frame_438 | 0.9407 | 0.8488 | 0.0919 |
| seq_2_frame_105 | 0.8862 | 0.8288 | 0.0574 |
| seq_2_frame_160 | 0.8627 | 0.7750 | 0.0877 |
| seq_2_frame_300 | 0.9058 | 0.8506 | 0.0552 |
| **Mean** | **0.9152** | **0.8385** | **0.0768** |

---

## Stage F1: Remove MAX_ACTIVE Cap

### Changes Made
- `train_gaussian.py:526-535`: Removed MAX_ACTIVE=12000 random subsampling block
- `train_gaussian.py:647-652`: Removed MAX_ACTIVE block inside C5 density control
- `train_gaussian.py:452`: Added `tag` parameter to `train_gaussians()` for separate output dirs

### Results

| Scene | mmIR | Baseline C3 | F1 result | F1-mmIR gap | Improvement over baseline |
|-------|------|-------------|-----------|-------------|--------------------------|
| seq_0_frame_135 | 0.9203 | 0.8698 | 0.8708 | 0.0495 | +0.0010 |
| seq_0_frame_390 | 0.9365 | 0.8263 | 0.8334 | 0.1031 | +0.0071 |
| seq_1_frame_185 | 0.9544 | 0.8701 | 0.8737 | 0.0807 | +0.0036 |
| seq_1_frame_438 | 0.9407 | 0.8488 | 0.8681 | 0.0726 | +0.0193 |
| seq_2_frame_105 | 0.8862 | 0.8288 | 0.8442 | 0.0420 | +0.0154 |
| seq_2_frame_160 | 0.8627 | 0.7750 | 0.7747 | 0.0880 | -0.0003 |
| seq_2_frame_300 | 0.9058 | 0.8506 | 0.8588 | 0.0470 | +0.0082 |
| **Mean** | **0.9152** | **0.8385** | **0.8462** | **0.0690** | **+0.0078** |

### Observations
- No OOM on any scene — gradient checkpointing with chunk_size=200 handles the full active set.
- Mean improvement: +0.0078 (0.8385 → 0.8462). Mean gap reduced from 0.077 to 0.069.
- Largest improvements on scenes that were most heavily capped:
  - seq_1_frame_438 (72K Gaussians): +0.0193
  - seq_2_frame_105 (113K Gaussians): +0.0154
  - seq_2_frame_300 (87K Gaussians): +0.0082
  - seq_0_frame_390 (107K Gaussians): +0.0071
- seq_2_frame_160 essentially unchanged (-0.0003, within MC noise).
- All scenes peak at iter 350-499, confirming training converges within 500 iters but several are still at the boundary (499), reinforcing the need for F2 (more iters).

### Decision: KEEP
- Consistent improvement across 6/7 scenes, no regressions. +0.008 mean from deleting code.

---

## Stage F2: Cosine LR Decay + 1500 Iterations (cumulative with F1)

### Changes Made
- `train_mesh.py:99-113`: Modified `get_lr_scale()` to accept `total_iters` parameter; added cosine decay from `decay_start=100` to `total_iters` with `min_lr_factor=0.01`
- `train_gaussian.py:452`: Changed default `num_iters` from 500 to 1500
- `train_gaussian.py:608`: Pass `total_iters=num_iters` to `get_lr_scale()`

### Results

| Scene | mmIR | Baseline C3 | F1 result | F2 result | F2-mmIR gap | Improvement over baseline |
|-------|------|-------------|-----------|-----------|-------------|--------------------------|
| seq_0_frame_135 | 0.9203 | 0.8698 | 0.8708 | 0.8938 | 0.0266 | +0.0240 |
| seq_0_frame_390 | 0.9365 | 0.8263 | 0.8334 | 0.9058 | 0.0307 | +0.0795 |
| seq_1_frame_185 | 0.9544 | 0.8701 | 0.8737 | 0.8982 | 0.0563 | +0.0281 |
| seq_1_frame_438 | 0.9407 | 0.8488 | 0.8681 | 0.8890 | 0.0517 | +0.0402 |
| seq_2_frame_105 | 0.8862 | 0.8288 | 0.8442 | 0.8758 | 0.0104 | +0.0470 |
| seq_2_frame_160 | 0.8627 | 0.7750 | 0.7747 | 0.7962 | 0.0665 | +0.0212 |
| seq_2_frame_300 | 0.9058 | 0.8506 | 0.8588 | 0.8836 | 0.0222 | +0.0330 |
| **Mean** | **0.9152** | **0.8385** | **0.8462** | **0.8775** | **0.0378** | **+0.0390** |

### Observations
- Massive improvement: +0.039 mean over baseline (0.8385 → 0.8775). Mean gap halved from 0.077 to 0.038.
- F2 alone (incremental over F1): +0.031 mean (0.8462 → 0.8775). The extra 1000 iterations + cosine decay are the biggest single factor so far.
- seq_0_frame_390 improved the most: +0.080 over baseline (0.826 → 0.906). Was the worst scene, now near mmIR.
- seq_2_frame_105 improved +0.047 (0.829 → 0.876), closing to within 0.01 of mmIR.
- Most scenes peak at iter 1250-1499, confirming the original 500 iters was severely under-training.
- seq_2_frame_160 remains the weakest scene (0.796 vs 0.863 mmIR, gap 0.067), though it improved +0.021.
- Two scenes (seq_0_frame_135, seq_2_frame_300) now within 0.03 of mmIR.
- seq_2_frame_105 now within 0.01 of mmIR — effectively closed.

### Decision: KEEP
- +0.039 mean improvement over baseline. Gap halved from 0.077 to 0.038. No regressions.

---

## Stage F3: Live Surfel Areas from Scales (cumulative with F1+F2)

### Changes Made
- `train_gaussian.py:242-252`: Fixed `_compute_scales_and_areas` to set `s = sqrt(vertex_area / pi)` so that `pi*s1*s2 = vertex_area` at init
- `train_gaussian.py:393-404`: `render_gaussians` now computes `areas = pi * s1 * s2 * opacity` from learnable scales instead of static `vertex_areas * opacity`
- `train_gaussian.py:562`: Increased scales LR from 5e-3 to 0.5 (matching materials LR) after diagnosing that scale gradients are ~7 orders of magnitude smaller than material gradients

### Results

| Scene | mmIR | Baseline C3 | F2 result | F3 result | F3-mmIR gap | Improvement over F2 |
|-------|------|-------------|-----------|-----------|-------------|---------------------|
| seq_0_frame_135 | 0.9203 | 0.8698 | 0.8938 | 0.8938 | 0.0266 | +0.0000 |
| seq_0_frame_390 | 0.9365 | 0.8263 | 0.9058 | 0.9058 | 0.0307 | +0.0000 |
| seq_1_frame_185 | 0.9544 | 0.8701 | 0.8982 | 0.8982 | 0.0563 | +0.0000 |
| seq_1_frame_438 | 0.9407 | 0.8488 | 0.8890 | 0.8890 | 0.0517 | +0.0000 |
| seq_2_frame_105 | 0.8862 | 0.8288 | 0.8758 | 0.8758 | 0.0104 | +0.0000 |
| seq_2_frame_160 | 0.8627 | 0.7750 | 0.7962 | 0.7962 | 0.0665 | +0.0000 |
| seq_2_frame_300 | 0.9058 | 0.8506 | 0.8836 | 0.8836 | 0.0222 | +0.0000 |
| **Mean** | **0.9152** | **0.8385** | **0.8775** | **0.8775** | **0.0378** | **+0.0000** |

### Diagnostics
- Scale gradient magnitude: mean |grad| = 7.66e-08 (vs materials: 1.81e-07, i.e. ~2x smaller per-element, but materials have 6 elements so total gradient norm is ~5x larger)
- After 1500 iters at LR 0.5: log_scale values changed by < 1e-4 from init
- Root cause: area only appears inside sqrt() in the weight computation, attenuating the gradient by 0.5/sqrt(...). The BSDF terms have much more nonlinear dependence on their parameters.

### Observations
- F3 is **neutral** — no improvement, no regression. Making scales live is necessary infrastructure, but the gradient through `sqrt(area)` is too weak to drive meaningful changes alone.
- This confirms that F4 (range-domain apodization) is needed to give scales a stronger gradient signal. The apodization applies a direct multiplicative factor `exp(-beta^2 * ...)` on weight, where beta depends on scale — providing a steeper gradient.
- F3 is correctly implemented (verified: init areas match, gradient flows to log_scales, LR was increased). The issue is physics, not code.

### Decision: KEEP (infrastructure, no regression)

**Post-mortem:** F3 was applied to `render_gaussians` but training uses `render_gaussians_factorized`. The factorized renderer was not updated. This was fixed as part of F4.

---

## Stage F4: Range-Domain Apodization + Live Scales in Factorized Renderer (cumulative)

### Changes Made
- `train_gaussian.py:436-439`: Updated `render_gaussians_factorized` to use `pi*s1*s2*opacity` (F3 fix for the right renderer)
- `rasterizer_factorized.py:225-234`: Added range-domain Gaussian apodization `exp(-2*(pi*slope*beta*tau/c)^2)` per (surfel, RX) pair
- `train_gaussian.py:601-608`: Added EVER-style anisotropy regularization `0.01 * mean((1-opacity)*(s_max-s_min))`
- `train_gaussian.py:562`: Scales LR was 0.5 (from F3)

### Results

| Scene | mmIR | F2 result | F4 result | Diff vs F2 |
|-------|------|-----------|-----------|------------|
| seq_0_frame_135 | 0.9203 | 0.8938 | 0.8783 | -0.0155 |
| seq_0_frame_390 | 0.9365 | 0.9058 | 0.8411 | **-0.0647** |
| seq_1_frame_185 | 0.9544 | 0.8982 | 0.8898 | -0.0084 |
| seq_1_frame_438 | 0.9407 | 0.8890 | 0.8831 | -0.0059 |
| seq_2_frame_105 | 0.8862 | 0.8758 | 0.8637 | -0.0121 |
| seq_2_frame_160 | 0.8627 | 0.7962 | 0.7993 | +0.0032 |
| seq_2_frame_300 | 0.9058 | 0.8836 | 0.8748 | -0.0088 |
| **Mean** | **0.9152** | **0.8775** | **0.8615** | **-0.0160** |

### Diagnostics
- Apodization factor was negligible: 0.999976 for typical surfels. `slope * tau / c` is ~1e-6, producing apod_arg ~1e-5. Zero effect.
- **Scales exploded**: max scale = 215,750 m, mean area grew from 0.024 to 53,577 m² (×2.26 million).
- Root cause: live scales with no effective constraint. The apodization was designed to prevent scale divergence, but its magnitude was negligible because the relevant frequency for intra-surfel phase cancellation is the carrier frequency f_c (77 GHz), not the chirp slope. Using f_c would over-attenuate everything because **the BSDF already models intra-surface phase cancellation** through its roughness parameters (sigma_h, l_c → Kirchhoff/SPM lobe widths).
- The apodization is **redundant with the BSDF** — it double-counts physics already captured by the Kirchhoff approximation.
- seq_1_frame_438 peaked at iter 200 (training instability from exploding scales).

### Decision: REVERT
- All F4 changes reverted (apodization, live scales in factorized renderer, anisotropy reg).
- F3 infrastructure kept in `render_gaussians` (non-factorized, C2 path only) and init fix.
- Scales remain non-functional for training. The BSDF already captures the scattering physics that scales would model.
- **Lesson: the VC-3DGS apodization concept does not translate to radar because the BSDF already models intra-surface phase cancellation. The optical papers need apodization because their SH color model has no physics; our Jones BSDF already does.**

### State after F4 revert: equivalent to F2 (mean 0.8775, gap 0.038)

---

## Stage F5: Progressive BSDF Activation (cumulative with F1+F2)

### Changes Made
- `train_gaussian.py:604-613`: After backward, zero gradients for BSDF params 2-5 (iters 0-300) and 4-5 (iters 300-800). Only Fresnel (eps_real, eps_imag) trains first.

### Results

| Scene | mmIR | F2 result | F5 result | Diff vs F2 |
|-------|------|-----------|-----------|------------|
| seq_0_frame_135 | 0.9203 | 0.8938 | 0.8802 | -0.0136 |
| seq_0_frame_390 | 0.9365 | 0.9058 | 0.8957 | -0.0102 |
| seq_1_frame_185 | 0.9544 | 0.8982 | 0.8789 | -0.0193 |
| seq_1_frame_438 | 0.9407 | 0.8890 | 0.9054 | +0.0164 |
| seq_2_frame_105 | 0.8862 | 0.8758 | 0.8550 | -0.0209 |
| seq_2_frame_160 | 0.8627 | 0.7962 | 0.7932 | -0.0029 |
| seq_2_frame_300 | 0.9058 | 0.8836 | 0.8945 | +0.0110 |
| **Mean** | **0.9152** | **0.8775** | **0.8719** | **-0.0056** |

### Observations
- Regression of -0.006 mean. Mixed: 2 scenes improved (seq_1_f438 +0.016, seq_2_f300 +0.011), 5 regressed.
- Root cause: C3 mode starts from mmIR-trained materials that are already near optimal. Freezing roughness/slab for the first 300 iters prevents fine-tuning during early adaptation to the Gaussian representation, slowing convergence.
- The 3DGS progressive SH strategy assumes training from scratch. For mmIR-initialized C3, all 6 BSDF params need to adapt simultaneously from the start.
- May still benefit C4 (from-scratch) mode where materials start at ITU concrete defaults.

### Decision: REVERT
- Progressive BSDF reverted. All 6 material params train from iter 0.
- **Lesson: progressive parameter activation helps for from-scratch training (3DGS starts from random SH) but hurts for fine-tuning from a good initialization (our C3 starts from mmIR).**

### State after F5 revert: equivalent to F2 (mean 0.8775, gap 0.038)
