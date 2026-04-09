# Gaussian-to-ADC Rendering Equation (v2)

Addresses all critique points from v1 review: operating regime analysis, second-order phase bound, trainability, post-FFT characterization, and κ(t_k) variation honesty.

---

## 1. Rendering Equation (unchanged from v1)

$$\text{ADC}_{ij}[k] = \sum_{g} A_g^{(ij)} \;\cdot\; \gamma_g^{(ij)}[k] \;\cdot\; \exp\!\big(j\,\phi_g^{(ij)}(t_k)\big)$$

- **A**: amplitude (opacity × BSDF × antenna × path loss) — evaluated at surfel center
- **γ**: coherence modulation — Fourier transform of Gaussian density at local phase gradient
- **φ**: FMCW phase — from exact surfel center distance

$$\gamma_g^{(ij)}[k] = \exp\!\left(-\tfrac{1}{2}\,\kappa(t_k)^2 \cdot \sigma_g^{2,(ij)}\right)$$

$$\sigma_g^{2,(ij)} = \mathbf{d}_{ij}^\top \boldsymbol{\Sigma}_g \,\mathbf{d}_{ij} = s_1^2(\mathbf{t}_1 \cdot \mathbf{d}_{ij})^2 + s_2^2(\mathbf{t}_2 \cdot \mathbf{d}_{ij})^2$$

---

## 2. Operating Regime of γ at 77 GHz

At f₀ = 77 GHz, κ₀ = 2πf₀/c ≈ 1614 rad/m and λ ≈ 3.9 mm.

γ transitions from ~1 to ~0 when κ₀σ ≈ 1, i.e. **σ ≈ λ/(2π) ≈ 0.62 mm**.

The projected variance σ² = s²·sin²(θ) where s is the Gaussian's lateral scale and θ is the tilt from face-on. The table below shows γ at κ = κ₀:

| Scale s | θ = 0° | 1° | 5° | 10° | 20° | 45° |
|---------|--------|----|----|-----|-----|-----|
| 0.5 mm | 1.000 | 1.000 | 0.998 | 0.990 | 0.963 | 0.850 |
| 1.0 mm | 1.000 | 1.000 | 0.990 | 0.962 | 0.859 | 0.522 |
| 2.0 mm | 1.000 | 0.998 | 0.961 | 0.855 | 0.544 | 0.074 |
| 5.0 mm | 1.000 | 0.990 | 0.781 | 0.375 | 0.022 | 0.000 |
| 10 mm  | 1.000 | 0.961 | 0.372 | 0.020 | 0.000 | 0.000 |
| 50 mm  | 1.000 | 0.371 | 0.000 | 0.000 | 0.000 | 0.000 |

**Conclusion**: For cm-scale Gaussians (our initialization), γ acts as a near-binary face-on/invisible gate. Smooth modulation occurs only when σ = s·sin(θ) ∈ [0.03 mm, 1.5 mm].

---

## 3. Decoupled Scales: Geometric vs Electromagnetic

The analysis above reveals a fundamental tension: geometric coverage requires cm-scale Gaussians (to tile the scene with ~100K surfels), but coherence modulation requires sub-mm effective extent.

**Resolution: decouple the geometric scale from the electromagnetic scale.**

Each Gaussian carries:
- **Geometric scales** (s₁, s₂): control spatial coverage and FPS-based tiling. Set from LiDAR PCA. Typical: 1–50 mm.
- **EM coherence scale** (σ_em): controls γ. Learnable. Initialized to λ/(2π) ≈ 0.62 mm.

The coherence factor becomes:

$$\gamma_g^{(ij)}[k] = \exp\!\left(-\tfrac{1}{2}\,\kappa(t_k)^2 \cdot \sigma_{\text{em},g}^2 \cdot \|\mathbf{d}_{ij,\perp}\|^2\right)$$

where $\mathbf{d}_{ij,\perp}$ is the bistatic direction projected onto the tangent plane (unit-free angular factor):

$$\|\mathbf{d}_{ij,\perp}\|^2 = (\mathbf{t}_1 \cdot \mathbf{d}_{ij})^2 + (\mathbf{t}_2 \cdot \mathbf{d}_{ij})^2 = \|\mathbf{d}_{ij}\|^2 - (\mathbf{n} \cdot \mathbf{d}_{ij})^2$$

**Physical motivation**: A large wall patch (geometric scale = 50 cm) does not scatter coherently from its entire surface at mmWave. Surface roughness, curvature, and material inhomogeneity limit the effective coherent area. The EM scale σ_em captures this effective coherent extent — analogous to the Fresnel zone size but learnable from data.

**Relationship to BSDF roughness**: The BSDF's Rayleigh factor η = exp(−(2kσ_h cosθ)²) captures micro-scale roughness (σ_h ~ 0.01–1 mm). The Gaussian coherence factor γ captures macro-scale spatial coherence (σ_em ~ 0.1–2 mm). These are complementary — different physical mechanisms at different spatial scales. Both should be present.

### Per-Gaussian parameter count (updated)

| Parameter | Shape | Description |
|-----------|-------|-------------|
| μ | (3,) | Position |
| q | (4,) | Rotation (tangent frame) |
| s₁, s₂ | (2,) | Geometric scales (coverage) |
| σ_em | (1,) | EM coherence scale (learned) |
| α | (1,) | Opacity |
| mat | (6,) | Material (ITU 6-param) |
| **Total** | **17** | 68 bytes per Gaussian |

---

## 4. Second-Order Phase Correction (Fresnel Term)

The first-order expansion assumes phase varies linearly across the surfel. The quadratic correction is:

$$\Delta\phi_{\text{quad}} \approx \frac{\kappa \cdot s^2}{2R}$$

| R | s = 5 mm | s = 20 mm | s = 50 mm |
|---|----------|-----------|-----------|
| 2 m | 0.01 rad (0.6°) ✓ | 0.16 rad (9°) ✓ | 1.01 rad (58°) ✗ |
| 5 m | 0.004 rad ✓ | 0.06 rad ✓ | 0.40 rad (23°) ✗ |
| 10 m | 0.002 rad ✓ | 0.03 rad ✓ | 0.20 rad (12°) ✓ |

**Validity criterion**: The first-order approximation holds when Δφ_quad ≪ π, i.e.:

$$s \ll \sqrt{2\pi R / \kappa} = \sqrt{\lambda R}$$

At R = 5m: s ≪ √(0.0039 × 5) = 0.14 m = 140 mm. For σ_em ≈ 0.6 mm, the first-order approximation is excellent (Δφ < 10⁻⁴ rad).

**The decoupled σ_em solves this too**: Since σ_em ≪ s_geometric, the Fresnel correction is negligible for the coherence integral even when it would be significant for the geometric scale.

If needed, the second-order correction modifies the Gaussian FT into a Fresnel integral:

$$\hat{\rho}(\boldsymbol{\xi}) = \alpha_g \cdot \frac{1}{\sqrt{1 + j\,\kappa\,\sigma_{\text{em}}^2/R}} \cdot \exp\!\left(-\frac{\sigma_{\text{em}}^2\,\xi^2}{2(1 + j\,\kappa\,\sigma_{\text{em}}^2/R)}\right)$$

This introduces a slight asymmetric broadening (defocus) of the range-FFT peak, but for σ_em ~ 0.6 mm and R > 2 m, the correction is < 0.01 dB.

---

## 5. Trainability of σ_em

### The vanishing gradient problem

$$\frac{\partial \gamma}{\partial \sigma_{\text{em}}} = -\kappa^2\,\sigma_{\text{em}}\,\|\mathbf{d}_\perp\|^2 \cdot \gamma$$

When γ → 0 (σ_em too large), this gradient vanishes. The optimizer cannot shrink σ_em back.

### Solution: log-amplitude parameterization

Store σ_em in log-space: σ_em = exp(ρ), where ρ is the learnable parameter.

The gradient in log-γ space:

$$\frac{\partial \log \gamma}{\partial \rho} = -\kappa^2\,\sigma_{\text{em}}^2\,\|\mathbf{d}_\perp\|^2$$

This is proportional to σ_em² — it **never vanishes** regardless of how large σ_em gets. The log-amplitude gradient is well-behaved everywhere.

**Implementation**: The loss already operates on RA magnitude, which is log-compressed. Combined with log-parameterization of σ_em, the gradient chain is:

$$\frac{\partial \mathcal{L}}{\partial \rho} = \frac{\partial \mathcal{L}}{\partial \log|RA|} \cdot \frac{\partial \log|RA|}{\partial \log\gamma} \cdot \frac{\partial \log\gamma}{\partial \rho}$$

All three terms are well-conditioned.

### Numerical verification

| σ_em | γ | ∂γ/∂σ | ∂log(γ)/∂σ | Status |
|------|---|-------|------------|--------|
| 0.1 mm | 0.987 | −257 | −260 | Smooth ✓ |
| 0.3 mm | 0.889 | −695 | −781 | Smooth ✓ |
| 0.6 mm | 0.626 | −978 | −1563 | Smooth ✓ |
| 1.0 mm | 0.272 | −708 | −2604 | Smooth ✓ |
| 2.0 mm | 0.005 | −29 | −5209 | γ-grad dying, log-grad fine ✓ |
| 5.0 mm | ~0 | ~0 | −13022 | γ-grad dead, log-grad fine ✓ |

The log-domain gradient grows monotonically with σ_em — the optimizer always has signal to push σ_em toward the right value.

### Initialization

Initialize ρ = log(λ/(2π)) ≈ log(0.62 mm) ≈ −7.4 (in log-metres). This places γ in the smooth transition region where gradients are maximally informative.

### Bounds

Clamp ρ ∈ [log(0.01 mm), log(5 mm)] = [−11.5, −5.3]. Below 0.01 mm, the surfel is a perfect point scatterer (γ = 1). Above 5 mm, it's fully incoherent (γ = 0).

---

## 6. Post-Range-FFT Characterization

γ(k) creates a Gaussian envelope on the ADC beat signal. After range FFT, this envelope broadens and attenuates the range peak.

| σ_em | γ at chirp midpoint | Peak attenuation | Range peak broadening |
|------|--------------------|-----------------|-----------------------|
| 0 (point) | 1.000 | 0.0 dB | None (Hann window only) |
| 0.3 mm | 0.889 | −1.2 dB | Negligible |
| 0.6 mm | 0.626 | −4.2 dB | Slight (~5%) |
| 1.0 mm | 0.272 | −11.5 dB | Moderate (~15%) |

The broadening is the convolution of the Hann window with the Gaussian envelope's FT (another Gaussian in range). For σ_em in the useful range (0.3–1.0 mm), broadening is small relative to the range resolution (59 mm).

---

## 7. Honesty Section: What γ Actually Does

### What γ provides (real effects):
1. **Angular selectivity per TX-RX pair**: γ varies across the virtual array because d_ij differs per element. This modulates the virtual array phase pattern, affecting azimuth FFT output. A large σ_em decorrelates across the array → broadened azimuth response.
2. **Physically motivated amplitude decay**: Tilted surfels contribute less coherent energy. This is a real effect complementary to the BSDF cosine factor.
3. **Learnable per-Gaussian coherent area**: σ_em captures effective scattering area — smaller for rough/curved surfaces, larger for smooth/flat ones.

### What γ does NOT provide (be honest):
1. **Smooth ADC-sample-dependent modulation**: κ(t_k) varies only 3.3% across the chirp. The per-sample variation in γ is < 5%. The dominant effect is per-TX-RX, not per-sample.
2. **Range-domain splatting in the optical sense**: The Gaussian's footprint in the range-FFT domain is not a smooth Gaussian blob — it's a slightly attenuated and broadened version of the point-scatterer peak. The "splatting" is a second-order correction, not a primary rendering mechanism.
3. **Occlusion handling**: Gaussians contribute additively regardless of depth ordering.

### The honest pitch:
γ is best understood as a **per-TX-RX angular coherence filter** derived from the Gaussian's spatial extent. Its primary effect is modulating the virtual array pattern (azimuth), not the range response. It makes the Gaussian's shape matter for rendering in a physically correct way, but it is a refinement to the amplitude model, not a fundamentally different rendering paradigm.

---

## 8. Updated Rendering Equation (with decoupled σ_em)

$$\boxed{\text{ADC}_{ij}[k] = \sum_{g} A_g^{(ij)} \;\cdot\; \gamma_g^{(ij)} \;\cdot\; \exp\!\big(j\,\phi_g^{(ij)}(t_k)\big)}$$

where γ is now **constant across ADC samples** (dropping the k-dependence, since it's < 5%):

$$\gamma_g^{(ij)} = \exp\!\left(-\tfrac{1}{2}\,\kappa_0^2 \;\cdot\; \sigma_{\text{em},g}^2 \;\cdot\; \|\mathbf{d}_{ij,\perp}\|^2\right)$$

This is computationally cheaper — γ has shape (M, N_tx, N_rx), same as the amplitude tensor A, and can be folded directly into the amplitude:

$$\tilde{A}_g^{(ij)} = A_g^{(ij)} \cdot \gamma_g^{(ij)}$$

No modification to the phasor loop is needed. The coherence factor multiplies the amplitude before the existing chunked phasor accumulation.
