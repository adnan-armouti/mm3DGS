# Translating Gaussian Rendering Ideas from Optics to Radar

Builds on [gaussian_method_comparison.md](gaussian_method_comparison.md). For each paper, we analyze: what does their core idea mean physically when translated to our RX-sphere splatting radar renderer? And critically: is our projection approach valid, or should we move away from it?

---

## What "Projection" Means in Each Method

First, a disambiguation. "Projection" means very different things across these methods:

| Method | What is projected | From → To | Approximation? |
|--------|-------------------|-----------|----------------|
| **3DGS** | 3D covariance matrix | World space → 2D screen space | Yes — affine Jacobian `Σ' = JWΣWᵀJᵀ` drops the third dimension, breaks near the camera |
| **3DGRT** | Nothing projected | Evaluated in 3D along each ray | No projection — ray queries BVH, evaluates `ρ(x)` at peak response |
| **EVER** | Nothing projected | Ray-ellipsoid intersection computed analytically | No projection — entry/exit t-values computed in 3D |
| **VC-3DGS** | 3D Gaussian → 1D along ray | World space → ray-parameterized 1D | Exact — `γ = μᵀΣ⁻¹d / dᵀΣ⁻¹d` and `β = 1/√(dᵀΣ⁻¹d)` are analytic |
| **mm25DGS_v2 (Ours)** | Surfel area → solid angle | 3D surface element → unit sphere at RX | Exact for a flat disc: `dΩ = A|cosθ|/d²` is the definition of solid angle |

The trend in optical rendering is clear: **3DGS's screen-space projection is the approximation that everyone is moving away from.** 3DGRT, EVER, and VC-3DGS all evaluate primitives directly in 3D (or in ray-parameterized space) without collapsing to 2D screen space.

But our "projection" is not the same kind of operation as 3DGS's. We are not projecting a 3D shape onto a 2D image plane. We are computing the solid angle that a surface element subtends at a receiver — a physically exact geometric quantity. This distinction is critical.

---

## Is Our RX-Sphere Projection Valid?

### What it computes

For surfel `i` and RX element `r`, the solid angle is:

```
dΩᵢʳ = Aᵢ × |cos(θᵢʳ)| / dᵢʳ²
```

where `Aᵢ` is surfel area, `θᵢʳ` is the angle between the surfel normal and the RX direction, and `dᵢʳ` is the surfel-to-RX distance. This enters the rendering equation as:

```
E[t,r,k] = C × Σᵢ sqrt(f_cos × G_t × G_r × dΩᵢʳ / d_t²) × αᵢ × exp(jφᵢ(k))
```

### Why it is physically correct

The received electric field at an RX antenna from a scattering surface is obtained by integrating over the solid angle subtended at the receiver:

```
E_r = ∫_Ω sqrt(radar equation terms) × exp(jφ) dΩ
```

For Monte Carlo evaluation (mmIR), each sample approximates `dΩ ≈ 1/(p × N)`. For deterministic surface summation (our approach), each surfel contributes `dΩ = A cos θ / d²`. Both are valid quadrature rules for the same integral. The solid angle formula is not an approximation — it is the definition of the integration measure.

### Where it breaks down

The solid angle formula `A cos θ / d²` is exact for an **infinitesimal** flat surface element. For a finite surfel with spatial extent, three issues arise:

**Issue 1: Phase variation across the surfel.** A surfel of lateral extent `s` spans a range of path lengths from TX to different points on the surfel to RX. The phase variation across the surfel is:

```
Δφ ≈ 2π × 2s / λ ≈ 2π × 2s / 3.9mm
```

For a surfel with s = 5cm (typical scale), `Δφ ≈ 160 radians` — many wavelengths. This means the phasor contributions from different parts of the surfel would partially cancel (destructive interference). Our point-scatterer model evaluates the phase at the surfel center only, ignoring this intra-surfel cancellation.

This is physically significant: the cancellation effect is exactly what the Kirchhoff/SPM BSDF models via the roughness parameters (`sigma_h`, `l_c`). But those parameters model sub-surfel roughness, not the surfel's own geometric extent. A perfectly smooth, flat surfel of 5cm would still have massive intra-surfel phase cancellation that we don't capture.

**Issue 2: Distance variation across the surfel.** The `1/d²` factor varies across the surfel extent. For a surfel at range 10m with scale 5cm, the variation is `(10±0.05)² / 10² ≈ ±1%` — negligible. This is not a concern.

**Issue 3: BSDF/antenna variation across the surfel.** The BSDF and antenna gains are direction-dependent. For a 5cm surfel at 10m, the angular subtense is ~0.3 degrees — well below the angular resolution of both the BSDF lobes and the antenna patterns. This is not a concern.

**Conclusion: Issue 1 (intra-surfel phase variation) is the only real problem.** The other two are negligible for the scene scales and surfel sizes in our benchmark.

---

## Per-Paper Translation to Radar Domain

### 3DGS → Radar

**What 3DGS does in optics:** Projects 3D Gaussian to 2D screen-space covariance `Σ' = JWΣWᵀJᵀ`, evaluates 2D Gaussian at each pixel, alpha-composites front-to-back.

**Radar translation — tile-based range-azimuth splatting:** Instead of projecting onto a camera image, project onto the Range-Azimuth (RA) plane:
- Range = `(d_tx + d_rx) / 2` (bistatic range)
- Azimuth = angle from radar boresight to the surfel

Each surfel would become a 2D "splat" in RA space, where the splat shape depends on the surfel's 3D covariance projected through the FMCW range mapping and angular beamforming. The Gaussian's extent in range would determine how many range bins it affects; its extent in azimuth would determine its angular spread.

**Why this does NOT work for radar:**
1. The RA projection is highly nonlinear — range is `(d_tx + d_rx)/2` not a linear camera model, and azimuth is not an affine function of position. The affine Jacobian approximation that 3DGS uses would introduce even larger errors than it does in optics.
2. Alpha compositing in RA space is physically wrong — radar signals from multiple range bins superpose coherently (phasor sum), not as transmittance-weighted blending.
3. The RA image is formed by FFT of the ADC signal, not by direct spatial projection. The surfel does not contribute to a localized pixel — it contributes a complex sinusoid across all ADC samples.

**Verdict:** 3DGS-style RA splatting is fundamentally incompatible with radar physics. Our scatter-add into ADC buffer is correct; RA-space splatting would require bypassing the FFT formation model.

---

### 3DGRT → Radar

**What 3DGRT does in optics:** Casts rays through a BVH of 3D Gaussians. For each ray, evaluates the Gaussian kernel at the point of maximum response, accumulates radiance via alpha compositing in depth order. Full ray tracing enables secondary effects (reflections, shadows).

**Radar translation — RX-centric ray tracing through Gaussian field:** For each RX element, cast rays over a hemisphere. Each ray traverses the BVH of Gaussian surfels. At each intersection, evaluate the full radar equation:
1. Compute BSDF at the ray-surfel intersection
2. Compute TX antenna gain from the surfel to each TX element
3. Compute FMCW beat phase from the bistatic path length
4. Accumulate complex phasor contribution into the ADC buffer

This is essentially **what mmIR already does** — the reference renderer is an RX-centric Monte Carlo ray tracer. The "Gaussian" part would only change the BVH traversal from triangle mesh intersection to Gaussian kernel evaluation.

**What it would gain:**
- Correct per-ray evaluation of the Gaussian kernel (not a point evaluation at center)
- Natural support for the Gaussian's spatial extent affecting the phase integral
- Hardware BVH acceleration via OptiX

**What it would cost:**
- Loses the key advantage of Gaussian splatting: no ray casting needed
- Requires OptiX/hardware RT in the forward pass, complicating PyTorch autograd
- Stochastic sampling (MC noise) returns — we'd be back to mmIR's variance problem
- Training would require differentiating through BVH traversal

**Verdict:** 3DGRT's approach translated to radar is essentially "mmIR but with Gaussian primitives instead of triangles." This eliminates the advantage of the Gaussian representation. We already have this capability via the reference renderer; the Gaussian renderer's purpose is to avoid ray casting entirely.

However, 3DGRT's **shadow testing via BVH** is directly transferable without adopting the full ray-tracing paradigm. Pre-computing visibility per (surfel, TX) pair using the mesh BVH gives us shadow correctness without runtime ray casting. This is proposed as Stage F8 in the gap closure plan.

---

### EVER → Radar

**What EVER does in optics:** Models each primitive as a constant-density 3D ellipsoid. Rays analytically compute entry/exit t-values. Between successive events, density is piecewise-constant, giving an exact closed-form volume rendering integral. The key insight: constant interior density makes the integral trivially solvable per interval.

**Radar translation — volumetric scattering ellipsoids:** Each primitive would be a 3D ellipsoid of uniform scattering density. A ray passing through it would accumulate scattering contributions over the path length inside the ellipsoid:

```
E_scat = ∫_{t_entry}^{t_exit} σ_scat × BSDF(t) × exp(jφ(t)) dt
```

where `σ_scat` is the volumetric scattering cross-section density and `φ(t)` varies linearly with path length (FMCW phase).

**Why this partially works but changes the physics:**
- This models **volumetric** scattering (e.g., rain, fog, vegetation at 77 GHz), not surface scattering. The ITU material model (Fresnel + Kirchhoff + SPM) is a surface scattering model — it assumes a sharp air-material interface. Volumetric scattering would require a different BSDF (Mie/Rayleigh scattering cross-sections).
- For surface scattering, what we want is a sharp material boundary, not a fuzzy volumetric density. A flat surfel IS the correct primitive for surface scattering — an EVER-style ellipsoid would blur the interface.
- The phasor integral `∫ σ exp(jφ(t)) dt` over the ellipsoid interior could be computed in closed form (it's a chirped Gaussian integral), which is elegant. But it answers a different physical question than what our renderer asks.

**What transfers regardless of the volumetric model:**
- **Density reparameterization**: EVER's `σ = -log(1-0.99α) / min(s)` is a gradient-friendly mapping from a learned proxy to a physical quantity. The specific formula assumes volumetric density, but the principle — parameterize in a space where gradients don't vanish at extremes — applies to our opacity as well.
- **Anisotropy regularization**: `L_aniso = (1-α)(s_max - s_min)` prevents degenerate shapes. Directly applicable to our surfels.
- **3D-consistency guarantee**: EVER proves that constant-density primitives with exact per-ray sorting produce a continuous function of camera pose (zero "popping"). Our coherent phasor sum is already inherently consistent — the sum is order-independent, so there is no popping by construction.

**Verdict:** EVER's volumetric formulation translates to a physically different scattering regime (volumetric, not surface). For our surface scattering problem, flat surfels remain the correct primitive. EVER's training stability innovations (reparameterization, regularization) transfer directly.

---

### VC-3DGS → Radar

**What VC-3DGS does in optics:** Evaluates the 3D Gaussian kernel analytically along each ray. At the point of peak density (`γ = μᵀΣ⁻¹d / dᵀΣ⁻¹d`), the kernel reaches its maximum. The 1D standard deviation along the ray (`β = 1/√(dᵀΣ⁻¹d)`) determines how much the Gaussian "spreads" in depth. The alpha is then `1 - exp(-κ G(γd) √(2π) β)`, which accounts for self-occlusion within the primitive.

**Radar translation — ray-parameterized Gaussian scattering kernel:** For each (surfel, TX, RX) path:
1. Parameterize the ray from RX through the Gaussian's center
2. Compute `γ` (peak) and `β` (spread) from the 3D Gaussian covariance
3. Integrate the scattering contribution over the Gaussian extent along the ray:

```
E_scat = ∫ sqrt(BSDF(t) × G_t(t) × G_r(t)) × κ × g(t) × exp(jφ(t)) dt
```

where `g(t) = G(γd) × exp(-(t-γ)²/(2β²))` is the 1D Gaussian profile along the ray and `φ(t) = 2π(f_c + S×t_ADC) × (d_tx(t) + d_rx(t))/c` is the FMCW phase.

**The key insight — Gaussian extent as a phase kernel:** If we assume BSDF, antenna gains, and `1/d²` are approximately constant over the surfel (true for small surfels at long range), the integral becomes:

```
E_scat ≈ W × κ × G(γd) × ∫ exp(-(t-γ)²/(2β²)) × exp(j 2π S t_ADC × 2t/c) dt
```

This is a **Fourier transform of a Gaussian** — it evaluates to another Gaussian in the frequency (range) domain:

```
E_scat ≈ W × κ × G(γd) × √(2π) β × exp(jφ_center) × exp(-2π²β² f_beat²/c²)
```

The last exponential term is a **range-domain apodization**: the surfel's spatial extent causes its range response to be smoothed by a Gaussian window of width proportional to `β`. Larger surfels (larger β) produce broader range responses. This is physically correct — a scatterer with spatial extent `β` cannot be resolved to better than `β` in range.

**What this means for our renderer:**
- Currently, each surfel contributes a single phasor `exp(jφ)` at its center position. This is a delta function in range — it contributes equally to all range bins via the FMCW phase.
- With VC-3DGS-style evaluation, the surfel would contribute a Gaussian-windowed phasor, concentrated around its center range bin and falling off with β.
- This would automatically handle the **intra-surfel phase cancellation** problem identified above: the Gaussian window is exactly the result of integrating the phase variation across the surfel extent.

**Verdict:** VC-3DGS's ray-parameterized evaluation translates elegantly to radar. The Gaussian extent along the ray becomes a range-domain apodization that captures intra-surfel phase variation. This is the most physically meaningful translation of any of the four papers. It does not require ray tracing — the `γ` and `β` parameters can be computed analytically from the surfel geometry and the RX direction, then applied as a multiplicative factor on the phasor contribution. However, this is a non-trivial change to the rendering equation and should be considered a future extension rather than an immediate gap-closure item.

---

## Should We Do Away With Projection?

### The short answer: No. Our projection is valid and should be kept.

### The longer answer:

The optical Gaussian papers are moving away from **3DGS's specific affine screen-space projection** because it introduces three known errors:
1. Non-linear distortion near the camera (Jacobian breaks down)
2. Loss of depth information (3D → 2D collapse)
3. View-inconsistency (same Gaussian produces different 2D covariances from different views, causing popping)

Our solid-angle projection `dΩ = A cos θ / d²` suffers from **none of these problems**:
1. It is exact for any distance (no linearization / Jacobian)
2. It preserves 3D information (the surfel's position, normal, and area all remain in the computation)
3. It is view-consistent (the solid angle is a deterministic function of surfel geometry and RX position)

The reason the optical papers move to ray-based evaluation (3DGRT) or analytic ray-space integration (VC-3DGS, EVER) is that their rendering equation requires evaluating the primitive's contribution at each pixel along the ray, with transmittance-based occlusion ordering. Our rendering equation is fundamentally different:
- No transmittance, no ordering — coherent phasor sum is commutative
- No pixel grid — accumulation target is the ADC buffer `(tx, rx, k)`
- No camera model — each (surfel, TX, RX) tuple defines a unique bistatic path

The "projection" in our case is not a geometric simplification but a **change of integration variable** from surface area to solid angle:

```
∫_surface f(x) dA(x) = ∫_Ω f(x(ω)) × (d²/cosθ) dΩ(ω)
```

This is standard radiometry and is used universally in both optical and radar remote sensing. It is not an approximation to move away from.

### What we SHOULD consider (from VC-3DGS)

The one idea worth adopting is VC-3DGS's **ray-space Gaussian extent**. Currently we treat each surfel as a point at its center. The surfel's spatial extent only enters through the area `A` in the solid angle formula. But the surfel also has extent along the RX direction, which causes phase variation. Modeling this as a range-domain Gaussian window (the VC-3DGS translation above) would:

1. Correctly attenuate the contribution of large surfels (intra-surfel phase cancellation)
2. Give the optimizer a reason to keep surfels small (smaller β → sharper range response → better range resolution)
3. Couple scale to the rendering equation through a second path beyond area (scale → β → range window)

This is not "doing away with projection" — it is augmenting our exact solid-angle projection with an exact range-domain evaluation of the Gaussian extent.

---

## Summary: What Transfers and What Doesn't

| Paper | Core Idea | Translates to Radar? | How |
|-------|-----------|---------------------|-----|
| **3DGS** | 2D screen-space splatting | No | RA space is nonlinear, alpha compositing is wrong for coherent signals, ADC buffer cannot be formed by spatial splatting |
| **3DGRT** | Per-ray 3D evaluation via BVH | Partially | Equivalent to mmIR ray tracing — valid but eliminates the advantage of splatting. Shadow testing via BVH transfers directly. |
| **EVER** | Constant-density volumetric ellipsoids | No (surface scattering) | Models volumetric scattering (rain/fog), not surface reflection. Training tricks (reparameterization, anisotropy reg) transfer. |
| **VC-3DGS** | Analytic ray-space Gaussian evaluation | Yes | Range-domain Gaussian apodization from surfel extent. Physically captures intra-surfel phase cancellation. Most natural translation. |

| Paper | Training Innovation | Transfers to Radar? |
|-------|-------------------|---------------------|
| **3DGS** | Progressive SH activation | Yes — progressive BSDF complexity (Fresnel → roughness → slab) |
| **3DGS** | Adaptive densification (clone/split/prune) | Yes — directly applicable to surfel management |
| **3DGS** | Exponential LR decay on positions | Yes — all papers agree: decay LR |
| **3DGRT** | World-space position gradients | Already used (we have no screen-space) |
| **3DGRT** | BVH shadow testing | Yes — pre-compute per (surfel, TX) visibility |
| **EVER** | Density reparameterization | Concept transfers — reparameterize opacity for gradient stability |
| **EVER** | Anisotropy regularization | Yes — directly applicable |
| **VC-3DGS** | Scale-coupled density | Concept transfers — couple opacity with surfel area |
| **VC-3DGS** | No affine projection | Not needed — our projection is already exact |

### The Projection Verdict

Our RX-sphere solid-angle projection is **physically exact, mathematically sound, and not the same operation that the optical papers are moving away from.** The optical papers abandon 3DGS's affine Jacobian screen-space approximation — a lossy dimensionality reduction that introduces artifacts. Our projection is a change of integration variable from surface area to solid angle, which is exact and used throughout radiometry.

The one enhancement worth considering is VC-3DGS's range-space Gaussian extent, which would capture intra-surfel phase cancellation without changing the fundamental projection framework. This should be evaluated after the gap-closure stages (F1-F8), as it changes the rendering equation itself.
