# 2D Gaussian Surfels and Bounded Ellipsoids: Analysis for Radar

Survey of papers using flat/bounded primitives — the primitive class closest to our mm25DGS surfels. For each paper: what is their primitive, how does it compare to ours, and can we benefit?

---

## Our Primitive for Reference

mm25DGS uses a **flat elliptical disc** (2D surfel):
- Position (3), quaternion (4) → normal from quaternion's 3rd column
- Two lateral scales `(s1, s2)` → area `A = pi * s1 * s2`
- Opacity (fill-factor), 6 ITU material parameters
- Evaluated as a **point scatterer at center** — the surfel's spatial extent enters only through the solid angle `dOmega = A * cos(theta) / d^2`
- No ray-surfel intersection, no Gaussian falloff profile, no depth extent
- Accumulation: coherent phasor sum (complex, order-independent)

---

## Paper Analysis

### 1. 2DGS — 2D Gaussian Splatting (SIGGRAPH 2024)
Huang, Yu, Chen, Geiger, Gao — arXiv: 2403.17888

**Primitive:** 2D oriented elliptical Gaussian disc. Same shape as ours. Gaussian falloff in the UV plane (infinite tail). Zero thickness.

**How it's evaluated:** Analytic ray-splat intersection. A ray `r(t) = o + t*d` is intersected with the surfel plane: `t* = n^T(mu - o) / (n^T * d)`. The hit point is transformed into the surfel's local UV frame. The 2D Gaussian kernel `G(u, v) = exp(-0.5 * (u^2/s1^2 + v^2/s2^2))` is evaluated at the hit UV coordinates.

**How it differs from us:** 2DGS evaluates the Gaussian profile at the actual ray-hit point. We evaluate everything at the surfel center. In optics, this matters because different pixels see different parts of the surfel. In radar, different (TX, RX) pairs illuminate the surfel from different bistatic angles, and the path length through different parts of the surfel varies.

**What we could adopt:**

The ray-surfel intersection gives the **exact bistatic path length** for each (TX, surfel, RX) triplet:

```python
# Current: path length computed at surfel center only
R_total = d_rx_center + d_tx_center

# 2DGS-inspired: compute actual intersection point on surfel plane
# For the TX->surfel leg:
t_tx = n^T(mu - tx_pos) / (n^T * dir_tx_to_surfel)
hit_tx = tx_pos + t_tx * dir_tx_to_surfel
# For the surfel->RX leg:
t_rx = n^T(mu - rx_pos) / (n^T * dir_rx_to_surfel)
hit_rx = rx_pos + t_rx * dir_rx_to_surfel
# These are different points on the surfel — the bistatic "bounce point"
# is where the incidence + reflection law is satisfied
```

However, for a flat specular surfel, the correct bounce point is determined by the law of reflection, not by projecting rays independently. The specular point satisfies `angle_in = angle_out` on the surfel plane. Computing this requires solving a reflection equation per (TX, surfel, RX) tuple — expensive.

**Verdict:** The exact ray-intersection idea is physically correct but computationally expensive for our O(N * n_tx * n_rx) path count. More relevant as a diagnostic: run it once to measure how much the center-only approximation affects path length accuracy. If the error is < lambda/10 (~0.4mm), the center approximation is fine. For our scene scales (surfels at 2-30m range, ~5cm size), the max path length error is ~s^2/(2*d) ~ 0.0025^2/(2*2) ~ 0.001mm — completely negligible. **Not needed.**

---

### 2. Gaussian Surfels (SIGGRAPH 2024)
Dai, Xu, Xie, Liu, Wang, Xu — arXiv: 2404.17774

**Primitive:** 3D Gaussian with z-scale forced to zero → flat 2D ellipse. Mathematically equivalent to ours. Normal = z-axis of the rotation.

**How it's evaluated:** Standard splatting (3D→2D projection), NOT ray-intersection. The zero z-scale means the projected 2D covariance is rank-deficient in one direction, creating a thin splat.

**Key contribution relevant to us:** The **normal-depth consistency loss**:

```
L_nd = mean(1 - (n_rendered . n_from_depth)^2)
```

where `n_rendered` is the surfel normal (from quaternion) and `n_from_depth` is the normal estimated from the rendered depth gradient. This forces the surfel's declared normal to agree with the surface geometry implied by the depth map.

**What we could adopt:**

A "normal-range consistency" loss for radar. The rendered range profile (from ADC→FFT) implicitly encodes surface geometry. If surfel normals are inconsistent with the range structure, the phase relationships will be wrong. However, the radar RA image does not have pixel-wise spatial correspondence like a depth map — range-azimuth cells are formed by FFT, not spatial projection. There's no straightforward "depth gradient → normal" computation in RA space.

**Verdict:** The normal consistency idea is elegant but doesn't translate well to radar because our output domain (RA image) doesn't have the spatial structure of a depth map. **Not applicable.**

---

### 3. EVER — Exact Volumetric Ellipsoid Rendering (arXiv 2024)
Mai, Hedman, Kopanas, Verbin, Futschik, Xu, Kuester, Barron, Zhang — arXiv: 2410.01804

**Primitive:** 3D constant-density ellipsoid. Hard boundary. Uniform interior density sigma. Not flat.

**How it's evaluated:** Analytic ray-ellipsoid intersection gives entry/exit t-values. Between consecutive events (entry/exit across all ellipsoids), density is piecewise-constant. Closed-form transmittance per interval.

**How it differs from us:** Our surfels are flat (no interior), so there's no volumetric integration. EVER's contribution is the **hard boundary** and **exact integral** — concepts that translate differently for flat primitives.

**What we could adopt:**

EVER's anisotropy regularization is already in our plan (F4). The density reparameterization (`sigma = -log(1-0.99*alpha)/min(s)`) doesn't apply because our opacity is a fill-factor, not volumetric density (as discussed with user).

One EVER insight that hasn't been fully exploited: **the hard boundary eliminates the "infinite tail" problem**. Standard 3DGS Gaussians contribute everywhere (just weakly at large distances). EVER's ellipsoids contribute exactly zero outside the boundary. For radar, a surfel that contributes zero beyond its geometric extent is physically correct — there is no scattering from empty space.

Currently, our surfels already have implicit hard boundaries (they contribute at their center point only, with area-weighted solid angle). The infinite-tail issue doesn't arise because we're already point-evaluating, not integrating a density profile along a ray.

**Verdict:** EVER's innovations that transfer to radar are already in the plan (anisotropy reg). The hard-boundary concept is automatically satisfied by our point-scatterer model. **No further additions needed.**

---

### 4. Don't Splat Your Gaussians (SIGGRAPH 2025)
Condor, Speierer, Bode, Bozic, Green, Didyk, Jarabo — arXiv: 2405.15425

**Primitive:** Two kernel types: (1) 3D anisotropic Gaussian (truncated at 3-sigma), (2) **Epanechnikov kernel** — the only exactly compact-support kernel in the Gaussian splatting literature.

The Epanechnikov kernel: `K(x) = (3/4)(1 - d(x)^2) for d(x) <= 1, else 0`, where `d(x)` is the Mahalanobis distance from the kernel center.

**How it's evaluated:** Analytic ray-primitive entry/exit. Closed-form transmittance integrals for both Gaussian and Epanechnikov kernels. Used inside a full path tracer with multiple scattering.

**Most physics-complete paper in the survey.** Models volumetric scattering, emission, anisotropic phase functions, and reciprocal light transport. This is the only Gaussian-family paper that does real volumetric scattering physics.

**How it differs from us:** This is designed for participating media (smoke, clouds, fog), not surface scattering. The primitives are volumetric — rays pass through them and accumulate scattering.

**What we could adopt:**

At 77 GHz, rain and fog DO cause volumetric scattering. If mm25DGS were extended to weather conditions, the Epanechnikov/Gaussian volumetric primitives from this paper would model atmospheric attenuation and scattering. Each rain droplet volume would be an Epanechnikov kernel with known scattering cross-section (Mie theory at 77 GHz).

For the current surface-scattering problem, the **anisotropic phase function** concept is relevant: it generalizes the direction-dependent scattering weight beyond a surface BSDF to a volumetric phase function `p(theta)`. For a flat surfel, the "phase function" is equivalent to the BSDF projected onto the hemisphere — so this doesn't add anything we don't already have.

**Verdict:** Relevant for future weather-condition modeling (volumetric scattering at 77 GHz). Not applicable to current gap closure. **File for future work.**

---

### 5. IRGS — Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing (CVPR 2025)
Gu et al. — arXiv: 2412.15867

**Primitive:** 2D Gaussian surfel. Same as 2DGS/ours. Bounded for BVH by an adaptive icosahedron mesh (20 triangles enclosing the surfel's effective support at an alpha_min threshold).

**How it's evaluated:** Primary view: standard splatting. Secondary illumination (inter-reflections): **2D Gaussian ray tracing**. Rays are cast, intersected analytically with the surfel plane, and the Gaussian weight evaluated at the UV hit point. The icosahedron mesh enables OptiX hardware BVH traversal.

**Key contribution:** The **2D Gaussian ray tracing** pipeline — the first paper to ray-trace flat Gaussian surfels efficiently via BVH. This enables shadow rays, inter-reflections, and full BSDF evaluation on flat surfels.

**What we could adopt:**

The icosahedron-bounded 2D Gaussian ray tracing is directly relevant to our F7 (pre-computed shadow visibility). Instead of testing shadow rays against the mesh (which doesn't account for surfel coverage), we could test against the surfel BVH. A ray from surfel A to TX that passes through surfel B's icosahedron would be flagged as (partially) occluded.

More interesting: IRGS's inter-reflection pipeline could model **multipath** in radar. First-bounce energy from surfel A reflects to surfel B, which re-scatters to the RX. This is the dominant multipath mechanism in indoor/urban radar scenes. IRGS shows how to do this efficiently on 2D Gaussian surfels using BVH ray tracing.

**Verdict:** The BVH-on-surfels shadow testing is a better version of our F7. The multipath capability is a significant future extension. **Consider upgrading F7 to use surfel BVH instead of mesh BVH.** Multipath is post-gap-closure research.

---

### 6. DRK — Deformable Radial Kernel Splatting (CVPR 2025)
Huang et al. — arXiv: 2412.11752

**Primitive:** Learnable planar radial kernel. Generalizes the 2D Gaussian disc — the falloff profile, boundary sharpness, and angular shape are all learnable. Can range from hard-edged opaque disc to smooth Gaussian blob.

**How it differs from us:** Our surfels have a fixed profile (effectively a delta function at center, weighted by area). DRK's kernel has a continuous, learnable radial profile that determines how the primitive's contribution varies across its extent.

**What we could adopt:**

For radar, the surfel's radial profile affects the **intra-surfel phase integral** (our VC-3DGS apodization in F4). Currently F4 assumes a Gaussian profile for the surfel extent, giving a Gaussian apodization in range. If the actual surfel profile is non-Gaussian (e.g., hard-edged like a building wall), the apodization should be a sinc instead:

```
Gaussian profile → Gaussian range apodization (exp(-beta^2 * ...))
Hard-edged disc → sinc range apodization (sin(x)/x with sidelobes)
```

DRK's learnable kernel could be translated to a learnable apodization profile: instead of fixing `exp(-2*pi^2*beta^2*...)`, learn the range-domain window shape per surfel. This captures whether each surfel is smooth-edged (Gaussian, no sidelobes in range) or hard-edged (sinc, range sidelobes).

**Verdict:** Interesting conceptual extension of F4, but over-engineered for gap closure. The Gaussian apodization is a reasonable approximation for most surfaces. **File for future work.**

---

### 7. GES — When Gaussian Meets Surfel (ACM TOG 2025)
Ye, Shao, Zhou — arXiv: 2504.17545

**Primitive:** **Opaque 2D surfel** — binary alpha (fully opaque or invisible). No Gaussian falloff. Hard disc boundary. 675 FPS at 1080p. Supplemented by sparse 3D Gaussians for fine details.

**How it differs from us:** GES surfels are fully opaque with binary transparency — the closest to our model where each surfel either scatters (active) or doesn't (culled). No continuous opacity.

**What we could adopt:**

GES's binary-alpha approach maps directly to our cosine culling: surfels are either active (cos > threshold) or inactive (cos <= threshold). GES shows that this binary approach, combined with a depth buffer for visibility, achieves extremely high rendering speed.

For radar, the question is whether continuous opacity adds value over binary. With the apodization (F4) handling scale-dependent attenuation, and the BSDF handling material-dependent scattering, what role does continuous opacity play? If it's primarily a fill-factor, binary might suffice: opacity > 0.5 → active, else pruned.

**Verdict:** The binary-alpha concept simplifies the model but reduces expressiveness. Not useful for gap closure (we need the continuous opacity gradient signal). **Not applicable now**, but interesting for inference-time speedup after training.

---

### 8. RadiosityGS — Differentiable Light Transport on Gaussian Surfels (SIGGRAPH Asia 2025)
Jiang, Sun, Li, Wang, Li, Ramamoorthi — arXiv: 2509.18497

**Primitive:** 2D Gaussian surfel (same as 2DGS). Used as radiosity patches.

**Key contribution relevant to us:** RadiosityGS solves **global illumination** on Gaussian surfels using a radiosity framework. Light transport between surfels is computed via form factors in SH space, with analytic backward passes.

**What we could adopt:**

Radiosity computes energy transfer between surface patches — the same physics as radar multipath. The form factor `F_{ij}` between surfels i and j is:

```
F_ij = (1/A_i) * integral_Ai integral_Aj (cos_i * cos_j) / (pi * r_ij^2) * V(i,j) dA_j dA_i
```

where `V(i,j)` is the visibility between patches. For radar multipath, the analogous quantity is the secondary-bounce transfer: how much of surfel i's scattered energy reaches surfel j, and how much of that re-scatters toward the RX.

RadiosityGS shows that this can be computed efficiently on Gaussian surfels using SH-space transport. The radar version would compute multipath contributions as a "radar radiosity" pass after the primary single-bounce rendering.

**Verdict:** This is the most principled approach to multipath modeling using Gaussian surfels. Not for gap closure (single-bounce is sufficient to match mmIR, which is also single-bounce), but the most promising path for future work on multipath.

---

### 9. RadioGS — Radiometrically Consistent Gaussian Surfels (ICLR 2026)
Han et al. — arXiv: 2603.01491

**Primitive:** 2D Gaussian surfel. Uses IRGS's 2D Gaussian ray tracing pipeline.

**Key contribution:** A **radiometric consistency** constraint: each surfel's learned radiance must match its physically-based rendered counterpart. This provides supervision for unobserved views, improving material decomposition.

**What we could adopt:**

The radar analog of radiometric consistency: for each surfel, the scattered field computed by the Jones BSDF should be consistent across all training frames. If surfel i produces field E_i in frame 1 and E_i' in frame 2 (different seeds/noise), the material parameters should explain both observations. This is automatically handled by our deterministic rendering (no MC noise in the Gaussian renderer), so the consistency constraint is trivially satisfied.

However, a related idea: **cross-sensor consistency**. If we train on cascaded radar (12TX×16RX) and evaluate on single-chip (3TX×4RX), the materials learned should be consistent across sensors. This is already how eval works (transfer test), but it could also be used as a training constraint: render with the single-chip virtual array configuration and penalize inconsistency.

**Verdict:** Radiometric consistency is automatically satisfied by our deterministic renderer. The cross-sensor consistency idea is interesting for the transfer evaluation but not for gap closure. **Not applicable now.**

---

## Summary: Which Surfel Papers Matter Most for mm25DGS?

| Paper | Relevance | Key Takeaway |
|-------|-----------|-------------|
| **2DGS** | Low | Ray-surfel intersection gives exact path length, but the error from center-only evaluation is ~0.001mm at our scene scales — negligible. |
| **EVER** | Already incorporated | Anisotropy regularization in F4. Hard boundary doesn't apply (we're already point-evaluating). |
| **IRGS** | Medium-High | BVH-on-surfels for shadow testing (upgrade F7). Multipath via inter-reflection (future work). |
| **Don't Splat** | Future | Volumetric scattering for rain/fog at 77 GHz. Not surface scattering. |
| **DRK** | Future | Learnable apodization profile per surfel. Over-engineered for now; Gaussian apodization in F4 is sufficient. |
| **GES** | Low | Binary opacity for inference speedup. Not useful during training. |
| **RadiosityGS** | Future (high) | Most principled path to multipath modeling on Gaussian surfels. |
| **RadioGS** | Low | Consistency constraint auto-satisfied by deterministic rendering. |

**Bottom line:** Our flat-surfel, point-scatterer, phasor-sum architecture is well-suited to the surface scattering problem. The surfel papers confirm that the 2D Gaussian disc is the right primitive for surface representations. The main gap between our approach and the state-of-the-art surfel papers is not the primitive shape but the **training dynamics** (scales, densification, LR — addressed in F1-F6) and the **intra-surfel physics** (addressed in F4 via VC-3DGS apodization). The surfel papers that would add most value post-gap-closure are IRGS (shadow/multipath via BVH) and RadiosityGS (principled multipath via radiosity).
