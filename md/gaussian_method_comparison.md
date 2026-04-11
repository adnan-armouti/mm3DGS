# Gaussian Rendering Methods: Structured Comparison

Comparison of mm25DGS_v2 (our RX-sphere splatting radar renderer) against four recent Gaussian rendering papers in the optical domain.

---

## 1. Rendering Primitive

| Method | Primitive | Description |
|--------|-----------|-------------|
| **mm25DGS_v2 (Ours)** | **2D Gaussian surfel** | Flat disc parameterized by position (3), quaternion (4), 2 lateral log-scales, opacity, and 6 ITU material params. Normal derived from quaternion's third column. Area = product of lateral scales. No thickness along normal. |
| **3DGS** (Kerbl 2023) | 3D anisotropic Gaussian | Full 3D covariance via scale (3) + quaternion (4). Projected to 2D screen-space splat via affine Jacobian approximation. Opacity scalar + SH color. |
| **3DGRT** (Moenne-Loccoz 2024) | 3D Gaussian particle | Same 3D Gaussian as 3DGS, but evaluated directly in 3D along each ray (no 2D projection). Also supports generalized kernels (degree-2, cosine-modulated, flat surfels). |
| **EVER** (Mai 2024) | Constant-density 3D ellipsoid | Hard-boundary solid with uniform interior density sigma. Scale (3) + quaternion (4). Density parameterized via opacity proxy to avoid vanishing gradients. |
| **VC-3DGS** (Talegaonkar 2025) | 3D anisotropic Gaussian | Same primitive as 3DGS but evaluated volumetrically along the ray in 3D (no affine 2D projection). Computes gamma_j (peak density along ray) and beta_j (1D std dev along ray) analytically from the 3D covariance. |

**Our distinction:** The surfel is a planar disc with physically meaningful parameters (ITU materials), not a volumetric blob. The "Gaussian" aspect enters only through scale-derived area weighting — the surfel itself is a point scatterer with area, not a spatial density kernel.

---

## 2. Rendering Equation

| Method | Equation Type | Formula |
|--------|--------------|---------|
| **mm25DGS_v2 (Ours)** | **Coherent phasor summation** | `E[t,r,k] = C * sum_i sqrt(f_cos_i * G_t_i * G_r_i * dOmega_i^r / d_t_i^2) * alpha_i * exp(j*phi_i(k))` where `dOmega_i^r = A_i * |cos(theta_rx)| / d_rx^2` is the solid angle at RX, and `phi_i(k) = 2*pi*(f_c*tau_i + S*tau_i*t_k)` encodes FMCW beat phase. Complex-valued accumulation via scatter_add into ADC buffer. |
| **3DGS** | Alpha compositing | `C = sum_i c_i * alpha_i * prod_{j<i}(1 - alpha_j)` where `alpha_i = opacity_i * G_2D(pixel)`. Front-to-back blending, scalar (RGB). |
| **3DGRT** | Volume rendering integral (discrete) | Same alpha-compositing formula as 3DGS but evaluated per-ray: `C = sum_i c_i(d) * alpha_i * prod_{j<i}(1 - alpha_j)` where `alpha_i = sigma_i * rho_i(x_i)` at the point of maximum Gaussian response along the ray. |
| **EVER** | Exact emission-absorption integral | `C = sum_i c_hat_i * (1 - exp(-sigma_i * Delta_t_i)) * prod_{j<i} exp(-sigma_j * Delta_t_j)`. Piecewise-constant density between 2N intersection events (entry/exit per ellipsoid). Analytic closed-form per interval. |
| **VC-3DGS** | Analytic volumetric alpha compositing | Same structure as 3DGS but `alpha_i = 1 - exp(-kappa_i * G_i(gamma_i*d) * sqrt(2*pi) * beta_i)` where gamma_i and beta_i are the ray-space peak location and 1D standard deviation computed from the full 3D covariance. |

**Our distinction:** The rendering equation is fundamentally different: complex-valued (amplitude + phase), not real-valued (color). The sqrt(power) field amplitude is multiplied by a phasor `exp(j*phi)`, and the sum is coherent — constructive/destructive interference determines the output. There is no transmittance/occlusion compositing; all surfels contribute additively (no depth-ordered blending).

---

## 3. Visibility / Occlusion Handling

| Method | Mechanism | Per-ray? | Exact? |
|--------|-----------|----------|--------|
| **mm25DGS_v2 (Ours)** | **Cosine culling + optional shadow rays** | N/A | Approximate |
| **3DGS** | Tile-based depth sort by Gaussian center | No (per-tile) | Approximate |
| **3DGRT** | Hardware BVH (OptiX) + per-ray k-buffer (k=16) + transmittance early termination | Yes | Exact (consistent) |
| **EVER** | Hardware BVH (OptiX) + per-ray entry/exit event sorting + exact transmittance accumulation | Yes | Exact |
| **VC-3DGS** | Tile-based depth sort (same as 3DGS) | No (per-tile) | Approximate |

**Our distinction:** Radar rendering does not use transmittance-based occlusion. All surfels that face the radar (cos > threshold) contribute to the received signal — the radar "sees through" semi-transparent structures because the phasor sum is coherent. Shadow rays (Mitsuba scene.ray_test) are available but skipped during Gaussian training for performance. Culling is geometric: range, FOV, and cosine filters.

---

## 4. View-Dependent Effects

| Method | Mechanism | Physical basis? |
|--------|-----------|----------------|
| **mm25DGS_v2 (Ours)** | **Per-surfel Jones BSDF** (6 ITU material params) + **per-element antenna gain patterns** | Yes — full polarimetric Jones formalism with Kirchhoff + SPM + CBS lobes, ITU P.2040-4 slab Fresnel, coherent/incoherent blend |
| **3DGS** | Spherical harmonics (degree 3, 48 coeffs) | No — empirical appearance fit |
| **3DGRT** | Spherical harmonics (degree 3, 48 coeffs) + secondary rays (reflections, refractions, shadows) | Partially — SH is empirical but secondary rays follow geometric optics |
| **EVER** | Spherical harmonics (degree 2, progressive) | No — empirical appearance fit |
| **VC-3DGS** | Spherical harmonics (same as 3DGS) | No — empirical appearance fit |

**Our distinction:** Every surfel evaluates a full physics-based BSDF with 6 learnable material parameters (eps_real, eps_imag, sigma_h, l_c, tau, thickness). The BSDF includes Fresnel reflection (complex amplitude, s/p polarization), GGX microfacet scattering (Kirchhoff approx), von Mises-Fisher diffuse (SPM), coherent backscatter enhancement, and ITU slab interference. Antenna gain patterns (TX and RX, E-plane and H-plane) are also learnable. This is qualitatively different from SH color fitting.

---

## 5. Differentiability

| Method | Differentiable parameters | AD framework | Special considerations |
|--------|--------------------------|--------------|----------------------|
| **mm25DGS_v2 (Ours)** | positions (3), quaternions (4), log_scales (2), logit_opacities (1), raw_materials (6), antenna patterns (4x361) | PyTorch autograd + gradient checkpointing per vertex chunk | Phase is detached (non-differentiable) to match mmIR. Material params use bounded reparameterization (sigmoid/exp/clamp). RMS gradient clipping per param group. |
| **3DGS** | positions (3), quaternions (4), scales (3), opacities (1), SH coeffs (48) | Hand-coded CUDA backward pass | Analytic gradients, back-to-front re-traversal to recover intermediate transmittance. No limit on gradient-receiving Gaussians per pixel. |
| **3DGRT** | positions (3), quaternions (4), scales (3), opacities (1), SH coeffs (48) | Hand-coded OptiX backward pass | Forward rays re-cast during backward. Gradients scatter-added with atomics. World-space position gradients (no screen-space). |
| **EVER** | positions (3), quaternions (4), scales (3), opacity proxy (1), SH coeffs | Slang.D adjoint rendering | Backward reconstructs ray state via inverse accumulation function, avoiding storing intermediates. |
| **VC-3DGS** | positions (3), quaternions (4), scales (3), density theta (1), SH coeffs | Slang.D | Density reparameterized as `kappa = -log(1 - 0.99*theta) * mean(1/s_i)` to couple density with scale. |

**Our distinction:** The gradient flows through physics (BSDF, antenna gains, radar equation) rather than through appearance (SH). Gradient checkpointing is critical because each surfel generates n_tx * n_rx * K paths (192 MIMO channels x 256 ADC samples). Phase detachment is a deliberate design choice matching the reference ray tracer.

---

## 6. Sorting / Ordering

| Method | Strategy | Global sort? | Handles overlap? |
|--------|----------|-------------|-----------------|
| **mm25DGS_v2 (Ours)** | **None** — direct scatter_add into ADC buffer | No | N/A (coherent sum, order-independent) |
| **3DGS** | Tile-based GPU radix sort by (tile_id, depth) | Yes (one global sort) | Approximate (per-tile, not per-pixel) |
| **3DGRT** | Per-ray BVH traversal with sorted k=16 hit buffer | No (per-ray) | Exact (consistent) |
| **EVER** | Per-ray BVH traversal with entry/exit event sorting | No (per-ray) | Exact (unlimited overlap) |
| **VC-3DGS** | Same tile-based sort as 3DGS | Yes (one global sort) | Approximate (per-tile) |

**Our distinction:** Coherent phasor summation is commutative — the order of accumulation does not matter. `scatter_add` adds each surfel's complex contribution to the appropriate (tx, rx, k) ADC bin regardless of order. This is fundamentally different from alpha compositing where front-to-back order determines the result. No sorting infrastructure is needed.

---

## 7. Physical Correctness

| Method | Level | Details |
|--------|-------|---------|
| **mm25DGS_v2 (Ours)** | **Physically based (radar)** | Bistatic radar equation with calibrated constants (Pt, antenna loss, lambda^2/16pi^2). Jones polarimetric BSDF with ITU P.2040-4 material model. FMCW beat phase encoding. Solid-angle weighting replaces MC importance sampling. Remaining approximation: single-bounce only, no multipath; cosine culling instead of exact shadow rays. |
| **3DGS** | Not physically based | Empirical alpha compositing of SH-colored splats. No energy conservation, no material model, no light transport. |
| **3DGRT** | Partially | Learned radiance field (not PB) but secondary rays follow geometric optics for reflections/refractions/shadows. |
| **EVER** | Volumetrically correct | Exact emission-absorption integral for constant-density ellipsoids. 3D-consistent (no popping). But still learned appearance, not PB material model. |
| **VC-3DGS** | More correct than 3DGS | Fixes three 3DGS approximations: (1) exponential transmittance instead of linear, (2) self-occlusion within Gaussian, (3) no affine covariance projection. Still learned appearance. |

**Our distinction:** mm25DGS_v2 is the only method with a physically grounded rendering equation tied to measurable material properties. The radar equation parameters (transmit power, antenna loss, wavelength) are calibrated from hardware specs, and the BSDF parameters correspond to real ITU material categories. The optical methods optimize for visual appearance reproduction; we optimize for electromagnetic scattering fidelity.

---

## 8. Domain

| Method | Domain | Sensor model |
|--------|--------|-------------|
| **mm25DGS_v2 (Ours)** | **77 GHz mmWave radar** | TI MMWCAS cascaded radar (12TX x 16RX), FMCW chirps, ADC -> Range-Azimuth images |
| **3DGS** | Optical RGB | Pinhole cameras, multi-view photographs |
| **3DGRT** | Optical RGB | Pinhole + fisheye + rolling shutter cameras |
| **EVER** | Optical RGB | Pinhole + fisheye cameras |
| **VC-3DGS** | Optical RGB (+ sparse CT) | Pinhole cameras; also demonstrated on X-ray tomography |

---

## Summary Comparison Table

| Axis | mm25DGS_v2 | 3DGS | 3DGRT | EVER | VC-3DGS |
|------|-----------|------|-------|------|---------|
| Primitive | 2D surfel | 3D Gaussian | 3D Gaussian | 3D ellipsoid | 3D Gaussian |
| Rendering eq. | Coherent phasor sum | Alpha compositing | Alpha compositing | Exact vol. rendering | Analytic vol. alpha |
| Visibility | Cosine cull | Tile depth sort | BVH + k-buffer | BVH + event sort | Tile depth sort |
| View-dep. | Jones BSDF + antenna | SH (deg 3) | SH + secondary rays | SH (deg 2) | SH (deg 3) |
| Differentiable | Materials, pos, rot, scale, opacity, antenna | All Gaussian params + SH | All params (world-space) | All params (adjoint) | All params (Slang.D) |
| Sorting | None (order-independent) | Tile-based global | Per-ray BVH | Per-ray BVH | Tile-based global |
| Physical | PB radar equation | Not PB | Partial (secondary rays) | Volumetrically exact | More correct than 3DGS |
| Domain | 77 GHz radar | RGB | RGB | RGB | RGB (+CT) |

---

## Architectural Proximity Analysis

### Which paper is mm25DGS_v2 closest to?

**3DGRT** is architecturally closest, for several reasons:

1. **Per-primitive evaluation along physical paths.** 3DGRT evaluates each Gaussian at the exact ray direction, not via a screen-space projection. mm25DGS_v2 similarly evaluates each surfel along the exact TX-surfel-RX bistatic path, computing direction-dependent BSDF and antenna gains per path.

2. **No tile-based rasterization.** Both methods bypass the tile-sort-blend pipeline. 3DGRT uses BVH ray tracing; mm25DGS_v2 uses direct scatter-add into the ADC buffer.

3. **Secondary physical effects.** 3DGRT demonstrates reflections, refractions, and shadows via secondary rays — the closest analog to mm25DGS_v2's shadow ray testing and physically-grounded BSDF evaluation.

4. **World-space gradients.** 3DGRT computes position gradients in 3D world space (no screen-space Jacobian); mm25DGS_v2 similarly optimizes in 3D world space.

However, mm25DGS_v2 diverges fundamentally in the accumulation model: coherent phasor summation (order-independent, complex-valued) vs. transmittance-weighted alpha compositing (order-dependent, real-valued).

### Innovations from each paper that could benefit mm25DGS_v2

**From 3DGRT:**
- **Hardware BVH for shadow/visibility testing.** Currently mm25DGS_v2 skips shadow rays during training for performance. 3DGRT's OptiX BVH with early transmittance termination could enable efficient per-path occlusion testing, improving physical accuracy for scenes with significant self-shadowing.
- **Generalized kernel shapes.** The cosine-modulated Gaussian (CSGM2) could model spatially varying scattering within a single surfel, potentially reducing the number of surfels needed for complex surfaces.

**From EVER:**
- **Volumetric density parameterization.** EVER's density reparameterization `sigma = -log(1 - 0.99*alpha) / min(s)` avoids vanishing gradients for near-opaque primitives. A similar reparameterization for surfel opacity could improve training stability in mm25DGS_v2.
- **Anisotropy regularization.** EVER's loss term penalizing highly elongated ellipsoids (`L_aniso = (1-alpha) * (s_max - s_min)`) could prevent degenerate surfel geometries during mm25DGS_v2 training.

**From VC-3DGS:**
- **Scale-coupled density.** VC-3DGS's `kappa = f(theta) * mean(1/s_i)` couples density with scale, discouraging large-but-dense or small-but-transparent Gaussians. For mm25DGS_v2, coupling opacity with surfel area (`alpha ~ area^{-1}`) could prevent large surfels from dominating the signal while remaining nearly transparent.
- **Volumetric self-attenuation.** The analytic computation of how much a Gaussian attenuates its own interior (via the ray-space beta parameter) has a radar analog: large surfels should not scatter with full strength across their entire extent because the radar wavefront attenuates through the structure. This is currently not modeled.

**From 3DGS:**
- **Adaptive density control (clone/split/prune).** mm25DGS_v2's C5 mode implements basic pruning, but 3DGS's full adaptive densification — cloning small Gaussians in under-reconstructed regions and splitting large Gaussians with high positional gradients — could significantly improve reconstruction quality, especially for the LiDAR-initialized (C4) mode where the initial surfel distribution may not match the scene's scattering geometry.
- **Progressive SH / parameter scheduling.** The progressive activation of SH bands during training (degree 0 first, then higher) could inspire progressive activation of BSDF complexity — e.g., train Fresnel-only first, then add Kirchhoff and SPM lobes, then CBS.
