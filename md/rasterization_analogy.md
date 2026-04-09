# Optical 3DGS vs mmWave 3DGS: The Rasterization Analogy

## Optical 3DGS Pipeline (no ray tracing)

```
For each Gaussian g:
    1. PROJECT:    μ_g  →  (u, v) pixel coords        [camera model, pure math]
    2. FOOTPRINT:  Σ_3D →  Σ_2D on image plane        [Jacobian of camera projection]
    3. SHADE:      SH coefficients → RGB color          [stored on Gaussian, no BRDF eval]
    4. SPLAT:      for each pixel in footprint:
                     weight = Gaussian2D(pixel; u,v, Σ_2D)
                     framebuffer[pixel] += weight × opacity × color    [alpha-composite]
```

Key properties:
- NO ray tracing, NO BVH, NO intersection testing
- NO material model evaluation (color is stored, not computed)
- The Gaussian's shape (Σ) determines its footprint on the image
- Pure forward scatter: primitive → pixels

## What is the radar analog?

```
For each Gaussian g:
    1. PROJECT:    μ_g  →  beat frequency f_beat = 2S·R/c     [FMCW model, pure math]
                   μ_g  →  phase ramp across VA               [array geometry, pure math]
    2. FOOTPRINT:  ???  →  ???                                  [what is the Gaussian's "footprint" on ADC?]
    3. SHADE:      ???  →  complex amplitude                    [what replaces SH coefficients?]
    4. SPLAT:      for each ADC sample k:
                     ADC[tx,rx,k] += amplitude × exp(j·φ(t_k))  [phasor scatter-add]
```

## The missing pieces

### Step 2 — FOOTPRINT

In optics: the Gaussian's 3D shape projects to a 2D blob on the image.
Each pixel in the blob gets a Gaussian-weighted contribution.

In radar: a point scatterer contributes a sinusoid to ALL K ADC samples.
There is no localized "blob" in ADC space. But there IS structure:

- The sinusoid's FREQUENCY (= beat frequency) is set by range
- The sinusoid's PHASE RAMP across virtual array elements is set by azimuth
- The sinusoid's AMPLITUDE could vary across VA elements and ADC samples

The "footprint" is not spatial (which ADC samples?) but spectral
(at what frequency and phase pattern?). After FFT, this footprint
BECOMES spatial — it maps to a localized region in the RA image.

The Gaussian's shape modulates this footprint via γ (the coherence
factor). But as we showed, γ is effectively a scalar at 77 GHz.

So: **the Gaussian's "footprint" on ADC is a complex sinusoid** —
characterized by (frequency, phase pattern, amplitude). Not a blob.

### Step 3 — SHADE

In optics: SH coefficients give view-dependent color. No BRDF evaluation.
The SH coefficients are learned per-Gaussian parameters.

In radar: what is the analog of "learned color per Gaussian"?

The amplitude A_g for a given TX-RX pair depends on:
  - Material properties (BSDF evaluation)
  - Antenna gains (pattern evaluation)  
  - Path loss (1/d)
  - Opacity

Currently we evaluate BSDF and antenna via DrJit — this is like
calling a ray tracing engine for shading, which defeats the purpose.

**The analog of SH coefficients: learned per-Gaussian complex
amplitude that is a function of viewing direction.**

Options:
  (a) Per-Gaussian scalar amplitude (view-independent) — 1 param
  (b) Per-Gaussian amplitude × cos(θ) model — 1 param + geometry
  (c) Per-Gaussian low-order SH (view-dependent) — 4-9 params
  (d) Per-Gaussian Fresnel + cosine (physics-lite) — 2 params (ε_r, opacity)

## The complete radar rasterization pipeline

```
For each Gaussian g:
    1. PROJECT:    d_tx = ||μ_g - p_tx||,  d_rx = ||μ_g - p_rx||
                   R_tot = d_tx + d_rx
                   f_beat = S · R_tot / c
                   θ_g = viewing angle from surfel normal

    2. FOOTPRINT:  The contribution spans all K ADC samples as a sinusoid
                   at frequency f_beat. (No spatial footprint to compute.)

    3. SHADE:      A_g = opacity_g × reflectivity_g(θ_g) / (d_tx · d_rx)
                   where reflectivity_g(θ) is a LEARNED function stored
                   on the Gaussian (not computed from an external BSDF).

    4. SPLAT:      φ(t_k) = 2π · (f₀ + S·t_k) · R_tot / c
                   ADC[tx, rx, k] += A_g · exp(j · φ(t_k))
```

This is pure forward rasterization:
- NO ray tracing, NO BVH, NO intersection testing
- NO external physics engine (no DrJit/Mitsuba calls)
- The Gaussian's "radar color" (reflectivity) is stored, not computed
- Pure forward scatter: primitive → ADC samples

## What this framework enables

1. The entire forward pass is pure PyTorch tensor operations
2. End-to-end differentiable without framework bridges (no DrJit↔PyTorch)
3. The learned reflectivity can capture effects the ITU BSDF can't
4. Training is faster (no BSDF/antenna evaluation overhead)
5. The Gaussian's orientation still matters (through θ_g in reflectivity)

## What is lost

1. Physical interpretability of material parameters (no ITU model)
2. Cross-sensor transfer (learned reflectivity is sensor-specific)
   → Mitigation: regularize toward a simple physics model, or learn
     a sensor-independent reflectivity + sensor-specific correction

## The reflectivity function

The simplest physically-motivated choice:

  reflectivity_g(θ) = R_g · |cos(θ)|

where R_g is a learned per-Gaussian scalar (effective radar cross-section).
This captures the dominant angular dependence (cosine falloff) with one
learnable parameter.

More expressive:

  reflectivity_g(θ) = Σ_{l=0}^{L} c_{g,l} · P_l(cos θ)

Legendre polynomial expansion with L+1 coefficients per Gaussian.
L=0: isotropic (1 param). L=1: dipole (2 params). L=2: quadrupole (3 params).
This is the radar analog of spherical harmonics in optical 3DGS.
