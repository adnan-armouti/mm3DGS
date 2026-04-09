# Gaussian-to-ADC Rendering Equation: Formal Derivation

## 1. Setup: FMCW Signal Model

A transmitted FMCW chirp has instantaneous frequency $f_0 + S \cdot t$, where $f_0$ is the carrier frequency and $S$ is the chirp slope. After dechirp (mixing received with transmitted), a point scatterer at round-trip delay $\tau$ produces the intermediate frequency (IF) signal:

$$s_{\text{IF}}(t) = a \cdot \exp\!\Big(j \cdot 2\pi\,\tau\,(f_0 + S\,t)\Big)$$

Sampled at $t_k = k / f_s$ for $k = 0, \ldots, K{-}1$, this becomes the ADC output:

$$s_{\text{IF}}[k] = a \cdot \exp\!\Big(j \cdot 2\pi\,\tau\,(f_0 + S\,t_k)\Big)$$

where $\tau = R_{\text{tot}} / c$ and $R_{\text{tot}} = d_{\text{tx}} + d_{\text{rx}}$ is the total path length from transmitter to scatterer to receiver.

---

## 2. Gaussian Surfel as a Spatially-Extended Reflector

A 2.5D Gaussian surfel $g$ is a planar disc centered at $\boldsymbol{\mu}_g$ with tangent frame $(\mathbf{t}_1, \mathbf{t}_2, \mathbf{n})$ where $\mathbf{n} = \mathbf{t}_1 \times \mathbf{t}_2$, and lateral scales $(s_1, s_2)$.

A point on the surfel at local coordinates $(u, v) \in \mathbb{R}^2$:

$$\mathbf{p}(u, v) = \boldsymbol{\mu}_g + u\,\mathbf{t}_1 + v\,\mathbf{t}_2$$

The scattering density follows a 2D Gaussian:

$$\rho_g(u, v) = \frac{\alpha_g}{2\pi\,s_1\,s_2}\;\exp\!\left(-\frac{1}{2}\left(\frac{u^2}{s_1^2} + \frac{v^2}{s_2^2}\right)\right)$$

where $\alpha_g \in [0, 1]$ is the surfel opacity.

Unlike a mesh triangle (zero-dimensional hit point) or a point scatterer, the surfel has continuous spatial extent. Every infinitesimal element $du\,dv$ on the surfel scatters independently.

---

## 3. ADC Contribution: The Integral

The total IF signal from surfel $g$, for transmitter $i$ and receiver $j$, at ADC sample $k$:

$$\text{ADC}_{g}^{(ij)}[k] = \int_{-\infty}^{\infty}\!\int_{-\infty}^{\infty} w_g(\mathbf{p}(u,v),\,i,\,j) \;\cdot\; \rho_g(u,v) \;\cdot\; \exp\!\Big(j\,\phi\big(\mathbf{p}(u,v),\,i,\,j,\,t_k\big)\Big)\;du\;dv$$

where:

- **$w_g$** encapsulates BSDF evaluation, antenna gains, and path loss at position $\mathbf{p}$
- **$\phi$** is the FMCW dechirp phase:

$$\phi(\mathbf{p},\,i,\,j,\,t_k) = 2\pi \cdot \frac{\|\mathbf{p} - \mathbf{p}_{\text{tx}_i}\| + \|\mathbf{p} - \mathbf{p}_{\text{rx}_j}\|}{c} \cdot (f_0 + S\,t_k)$$

---

## 4. First-Order Expansion (Far-Field / Small-Surfel Approximation)

For a surfel whose extent is small relative to the distance to the radar (always true: cm-scale surfels at metre-scale ranges), we Taylor-expand both $w$ and $\phi$ around the surfel center $\boldsymbol{\mu}_g$.

**Amplitude** varies slowly across the surfel, so:

$$w_g(\mathbf{p}(u,v),\,i,\,j) \;\approx\; w_g(\boldsymbol{\mu}_g,\,i,\,j) \;\equiv\; A_g^{(ij)}$$

**Phase** varies rapidly (at 77 GHz, $\lambda \approx 4\,\text{mm}$), so we keep the first-order term:

$$\phi(\mathbf{p}(u,v),\,i,\,j,\,t_k) \;\approx\; \phi_0^{(ij)}(t_k) \;+\; \boldsymbol{\xi}_{ij}(t_k) \cdot \begin{pmatrix} u \\ v \end{pmatrix}$$

where:

$$\phi_0^{(ij)}(t_k) = 2\pi\,\tau_g^{(ij)}\,(f_0 + S\,t_k) \qquad\qquad \tau_g^{(ij)} = \frac{d_{\text{tx},g}^{(i)} + d_{\text{rx},g}^{(j)}}{c}$$

The **spatial frequency vector** $\boldsymbol{\xi}$ is the phase gradient projected onto the tangent plane:

$$\boldsymbol{\xi}_{ij}(t_k) = \mathbf{J}_{\text{tang}}^\top \;\nabla_{\mathbf{p}}\,\phi \Big|_{\boldsymbol{\mu}_g}$$

with $\mathbf{J}_{\text{tang}} = [\,\mathbf{t}_1 \;|\; \mathbf{t}_2\,]$ (the $3 \times 2$ tangent frame matrix) and:

$$\nabla_{\mathbf{p}}\,\phi = \kappa(t_k) \cdot \mathbf{d}_{ij}$$

where we define:

- **Instantaneous wavenumber**: $\;\kappa(t_k) = \dfrac{2\pi(f_0 + S\,t_k)}{c}$

- **Bistatic direction sum**: $\;\mathbf{d}_{ij} = \hat{\mathbf{r}}_{\text{tx}_i} + \hat{\mathbf{r}}_{\text{rx}_j}$, with $\hat{\mathbf{r}}_{\text{tx}_i} = \dfrac{\boldsymbol{\mu}_g - \mathbf{p}_{\text{tx}_i}}{d_{\text{tx},g}^{(i)}}$ and $\hat{\mathbf{r}}_{\text{rx}_j} = \dfrac{\boldsymbol{\mu}_g - \mathbf{p}_{\text{rx}_j}}{d_{\text{rx},g}^{(j)}}$

So the spatial frequency components are:

$$\xi_1^{(ij)}(t_k) = \kappa(t_k) \cdot (\mathbf{t}_1 \cdot \mathbf{d}_{ij}), \qquad \xi_2^{(ij)}(t_k) = \kappa(t_k) \cdot (\mathbf{t}_2 \cdot \mathbf{d}_{ij})$$

---

## 5. Evaluating the Integral: Fourier Transform of a Gaussian

Substituting the first-order expansion:

$$\text{ADC}_{g}^{(ij)}[k] \;\approx\; A_g^{(ij)} \cdot e^{\,j\,\phi_0^{(ij)}(t_k)} \cdot \underbrace{\int\!\!\int \rho_g(u,v) \cdot e^{\,j\,(\xi_1\,u + \xi_2\,v)}\;du\;dv}_{\text{Fourier transform of } \rho_g \text{ at } \boldsymbol{\xi}}$$

The Fourier transform of a 2D Gaussian with covariance $\text{diag}(s_1^2, s_2^2)$ is itself a Gaussian:

$$\hat{\rho}_g(\boldsymbol{\xi}) = \alpha_g \cdot \exp\!\left(-\frac{1}{2}\big(s_1^2\,\xi_1^2 + s_2^2\,\xi_2^2\big)\right)$$

Substituting the spatial frequencies:

$$s_1^2\,\xi_1^2 + s_2^2\,\xi_2^2 = \kappa(t_k)^2 \cdot \Big[\,s_1^2\,(\mathbf{t}_1 \cdot \mathbf{d}_{ij})^2 + s_2^2\,(\mathbf{t}_2 \cdot \mathbf{d}_{ij})^2\,\Big]$$

Recognising the 3D covariance $\boldsymbol{\Sigma}_{g} = s_1^2\,\mathbf{t}_1\mathbf{t}_1^\top + s_2^2\,\mathbf{t}_2\mathbf{t}_2^\top$, this becomes:

$$s_1^2\,\xi_1^2 + s_2^2\,\xi_2^2 = \kappa(t_k)^2 \cdot \mathbf{d}_{ij}^\top\,\boldsymbol{\Sigma}_{g}\,\mathbf{d}_{ij}$$

---

## 6. The Rendering Equation

Combining everything, the complete ADC output for the full scene:

$$\boxed{\text{ADC}_{ij}[k] = \sum_{g=1}^{N} \; A_g^{(ij)} \;\cdot\; \gamma_g^{(ij)}[k] \;\cdot\; \exp\!\Big(j\,\phi_g^{(ij)}(t_k)\Big)}$$

with three factors per Gaussian:

### Factor 1: Amplitude $A_g^{(ij)}$

$$A_g^{(ij)} = \alpha_g \;\cdot\; f_{\cos}\!\big(\hat{\mathbf{w}}_o,\;\hat{\mathbf{w}}_i,\;\mathbf{n}_g,\;\text{mat}_g\big) \;\cdot\; \frac{G_{\text{tx}}^{(i)}\;\cdot\;G_{\text{rx}}^{(j)}}{d_{\text{tx},g}^{(i)} \;\cdot\; d_{\text{rx},g}^{(j)}}$$

This is the standard radar amplitude: opacity × BSDF × antenna gains / path loss. **Same as current implementation.**

### Factor 2: Coherence Modulation $\gamma_g^{(ij)}[k]$ (THE NEW ELEMENT)

$$\gamma_g^{(ij)}[k] = \exp\!\left(-\frac{1}{2}\;\kappa(t_k)^2 \;\cdot\; \mathbf{d}_{ij}^\top\,\boldsymbol{\Sigma}_{g}\,\mathbf{d}_{ij}\right)$$

Defining the **projected spatial variance** (scalar, computed once per Gaussian × TX × RX):

$$\sigma_{g}^{2,(ij)} = \mathbf{d}_{ij}^\top\,\boldsymbol{\Sigma}_{g}\,\mathbf{d}_{ij} = s_1^2\,(\mathbf{t}_1 \cdot \mathbf{d}_{ij})^2 + s_2^2\,(\mathbf{t}_2 \cdot \mathbf{d}_{ij})^2$$

this simplifies to:

$$\gamma_g^{(ij)}[k] = \exp\!\left(-\frac{1}{2}\;\kappa(t_k)^2 \;\cdot\; \sigma_{g}^{2,(ij)}\right)$$

**Physical interpretation:**

| Regime | Condition | $\gamma$ | Meaning |
|--------|-----------|----------|---------|
| Face-on surfel | $\mathbf{d}_{ij} \perp$ tangent plane | $\sigma_g^2 = 0 \Rightarrow \gamma = 1$ | All points on surfel at same range → fully coherent |
| Tilted surfel | $\mathbf{d}_{ij}$ has tangent component | $\sigma_g^2 > 0 \Rightarrow \gamma < 1$ | Range varies across surfel → partial decorrelation |
| Edge-on surfel | $\mathbf{d}_{ij} \parallel$ tangent plane | $\sigma_g^2 = s_{\max}^2 \Rightarrow \gamma \approx 0$ | Maximum range spread → complete decorrelation |

**Key properties:**
- Depends on **Gaussian shape** ($\boldsymbol{\Sigma}_g$) → the Gaussian representation matters for rendering
- Depends on **ADC sample index** $k$ (through $\kappa(t_k)$) → varies across the chirp (time-domain envelope)
- Depends on **TX-RX pair** (through $\mathbf{d}_{ij}$) → varies across virtual array elements
- Is differentiable w.r.t. all Gaussian parameters (position, rotation, scales)

### Factor 3: Phase $\phi_g^{(ij)}(t_k)$

$$\phi_g^{(ij)}(t_k) = 2\pi\,\tau_g^{(ij)}\,(f_0 + S\,t_k)$$

Exact phase from Gaussian center position. **Same as current implementation.**

---

## 7. Decomposition for Efficient Computation

The coherence factor decomposes into a TX-RX-dependent term and a sample-dependent term:

$$\gamma_g^{(ij)}[k] = \exp\!\left(-\frac{1}{2}\;\kappa(t_k)^2 \;\cdot\; \sigma_{g}^{2,(ij)}\right)$$

**Step 1** — Compute $\sigma_{g}^{2,(ij)}$ once per (Gaussian, TX, RX):

$$\sigma_{g}^{2,(ij)} = s_1^2\,(\mathbf{t}_1 \cdot \mathbf{d}_{ij})^2 + s_2^2\,(\mathbf{t}_2 \cdot \mathbf{d}_{ij})^2$$

Shape: $(M, N_{\text{tx}}, N_{\text{rx}})$ — same as existing amplitude tensor $A$.

**Step 2** — Compute $\kappa(t_k)^2$ once for all ADC samples:

$$\kappa(t_k)^2 = \left(\frac{2\pi(f_0 + S\,t_k)}{c}\right)^2$$

Shape: $(K,)$ — a 1D precomputed vector.

**Step 3** — The full coherence factor is an outer product + exp:

$$\gamma_g^{(ij)}[k] = \exp\!\Big(-\tfrac{1}{2}\;\kappa_k^2\;\cdot\;\sigma_{g}^{2,(ij)}\Big)$$

Shape: $(M, N_{\text{tx}}, N_{\text{rx}}, K)$ — same as the phasor tensor $\phi$.

This slots directly into the existing chunked phasor loop, modifying each chunk's contribution from:

$$\text{(current):} \quad A_c \cdot \cos(\phi), \quad A_c \cdot \sin(\phi)$$

to:

$$\text{(new):} \quad A_c \cdot \gamma_c \cdot \cos(\phi), \quad A_c \cdot \gamma_c \cdot \sin(\phi)$$

where $\gamma_c$ is computed inside the chunk from $\sigma_g^2$ and $\kappa^2$.

---

## 8. Summary: What Changed from Point-Scatterer Model

| | Point scatterer (mmIR, current mm25DGS) | Gaussian surfel (this derivation) |
|---|---|---|
| Surface element | Zero-dimensional hit point | 2D Gaussian with covariance $\boldsymbol{\Sigma}$ |
| ADC contribution | $A \cdot e^{j\phi}$ | $A \cdot \gamma \cdot e^{j\phi}$ |
| Gaussian shape affects rendering? | No | Yes, through $\gamma$ |
| Varies across ADC samples? | Phase only | Phase AND amplitude (via $\gamma(k)$) |
| Varies across virtual array? | Phase only | Phase AND amplitude (via $\gamma(i,j)$) |
| Physical basis | Geometric optics | Fourier optics (spatial coherence) |

The coherence modulation $\gamma$ is the **Fourier transform of the Gaussian spatial distribution evaluated at the local phase gradient** — it is the mathematically exact consequence of treating the surfel as a spatially-extended reflector rather than a point.
