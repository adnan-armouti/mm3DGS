# Extended Gaussian Splatting Survey: Optics-to-Radar Translation Analysis

18 papers beyond the 4 already analyzed in [gaussian_method_comparison.md](gaussian_method_comparison.md). For each: what does it do, what would the radar translation look like, and would we benefit?

---

## Category A: Radar / RF / SAR — Directly Relevant

---

### A1. RadarSplat (ICCV 2025)
**Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes**
Kung, Harisha, Vasudevan, Eid, Skinner — U. Michigan
arXiv: 2506.01379

**What it does:** First 3DGS formulation for imaging radar. Each Gaussian stores a power return ratio `sigma_i = rho_i * min(alpha_i + eta_i, 1)` where `rho_i` is view-dependent reflectivity (SH), `alpha_i` is occupancy, and `eta_i` is a noise probability. Rendering uses additive power accumulation (not alpha-compositing):

```
P_r(theta, n) = sum_i [P_t * G(phi_i)^2 * sigma_i] / [(4*pi)^3 * R_n^4]
```

Explicitly models multipath (FFT detection + removal), receiver saturation, and speckle.

**Radar translation:** This IS a radar Gaussian paper. Direct comparison with our approach:

| Aspect | RadarSplat | mm25DGS_v2 (Ours) |
|--------|-----------|-------------------|
| Radar type | Automotive imaging radar (1TX, few RX) | MMWCAS cascaded (12TX, 16RX) |
| Rendering eq. | Additive power sum | Coherent phasor sum (complex) |
| Phase | Not modeled (power-only) | Full FMCW beat phase |
| BSDF | Learned SH reflectivity | Physics-based Jones BSDF (6 ITU params) |
| Noise model | Explicit (multipath, saturation, speckle) | None |
| ADC synthesis | No (renders RA directly) | Yes (renders ADC, FFT to RA) |
| Antenna model | Antenna gain pattern G(phi) | Per-element TX/RX patterns (learnable) |

**Would we benefit?** Yes, in three specific ways:
1. **Noise probability parameter `eta_i`**: Each Gaussian has a learned noise/speckle probability. For mm25DGS, this could model sub-surfel scattering incoherence — surfels in cluttered regions (vegetation, gravel) produce noise-like returns that our deterministic BSDF cannot capture.
2. **Multipath detection/removal**: RadarSplat detects multipath in the RA image via FFT peak analysis and removes it before supervision. We could similarly pre-process GT RA images to suppress multipath artifacts that our single-bounce renderer cannot reproduce.
3. **Direct RA rendering (bypassing ADC)**: RadarSplat renders directly into range-azimuth space. This is less physically accurate than our ADC synthesis + FFT pipeline, but dramatically faster. Could be useful as a coarse initialization or for real-time inference after training.

---

### A2. GSRF (NeurIPS 2025)
**Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis**
Yang, Dong, Ji, Du, Srivastava
arXiv: 2502.01826

**What it does:** Extends 3DGS to RF with **complex-valued Gaussians**. Each Gaussian carries amplitude AND phase. Uses a hybrid Fourier-Legendre basis for directional and phase-dependent radiance. Implements complex-valued ray tracing grounded in wavefront propagation (Huygens' principle). +21.2% PSNR, +56.4% MSE improvement over NeRF2.

**Radar translation:** Extremely close to our formulation. Both use complex-valued (amplitude + phase) accumulation. Key differences:

| Aspect | GSRF | mm25DGS_v2 (Ours) |
|--------|------|-------------------|
| Complex representation | Per-Gaussian complex amplitude | Per-path phasor exp(j*phi) |
| Directional basis | Fourier-Legendre (learned) | Physics-based Jones BSDF |
| Phase model | Learned per-Gaussian | Deterministic FMCW: phi = 2*pi*(f_c*tau + S*tau*t_k) |
| Ray tracing | Complex-valued custom CUDA RT | No ray tracing (scatter-add) |

**Would we benefit?** Possibly, in one important way:
- **Learned complex directional basis**: GSRF uses Fourier-Legendre functions instead of a physics BSDF. For mm25DGS, a hybrid approach could work: use our physics BSDF for the amplitude, but add a small learned complex residual per Gaussian to capture effects the 6-parameter ITU model cannot (e.g., surface waves, creeping waves, multiple scattering within a surfel's neighborhood). This is analogous to GaussianShader's "SH residual" on top of analytic BRDF.

---

### A3. SAR-GS (arXiv 2025)
**Gaussian Splatting Based SAR Images Rendering and Target Reconstruction**
Li, Lei, Wei, Xu
arXiv: 2506.21633

**What it does:** Introduces a SAR Differentiable Gaussian Splatting Rasterizer (SDGR). Uses a Mapping and Projection Algorithm (MPA) to compute electromagnetic scattering intensities and render SAR images. Each Gaussian has a scattering coefficient (RCS proxy). Contributions are added **coherently** (amplitude + phase).

**Radar translation:** SAR is coherent imaging — the closest modality to FMCW radar. Both use phasor summation. Key difference: SAR synthesizes aperture via platform motion, while FMCW uses MIMO virtual aperture. The MPA (Mapping and Projection Algorithm) maps 3D scatterer positions to SAR image coordinates accounting for range compression and azimuth synthesis — analogous to our ADC-to-RA pipeline.

**Would we benefit?**
- **Scattering + Shadow dual splatting**: The follow-up SAR-3DGS (IEEE 2025) uses two separate splatting passes: one for EM returns (scattering splatting) and one for radar beam occlusion (shadow splatting). Radar shadows are physically distinct from optical shadows — they model foreshortening and layover. For mm25DGS, a similar dual-pass could replace our binary shadow test with a soft shadow model that accounts for partial occlusion (diffracting edges, partial beam blockage).

---

### A4. RF-3DGS (IEEE Trans. Wireless Comm. 2025)
**Wireless Channel Modeling with Radio Radiance Field and 3D Gaussian Splatting**
Zhang, Sun, Berweger, Gentile, Hu
arXiv: 2411.19420

**What it does:** Represents the wireless channel as a "radio radiance field" using 3DGS. Each Gaussian represents a scattering cluster with gain, delay, AoA, and AoD parameters. Renders spatial spectra (channel gain, delay, angle-of-arrival, angle-of-departure) at arbitrary positions.

**Radar translation:** Conceptually related but at a higher abstraction level. RF-3DGS models the aggregate multipath channel, not individual scatterers. For mm25DGS, the useful insight is:
- **Gaussians as scattering clusters**: Instead of one Gaussian per mesh vertex, consider grouping nearby surfels into "scattering clusters" with aggregate parameters. This could reduce the surfel count while preserving the aggregate scattering response — useful for the MAX_ACTIVE bottleneck.

---

## Category B: Non-Optical Wave-Based — Transferable Physics

---

### B1. SonarSplat (IEEE RA-L 2025)
**Novel View Synthesis of Imaging Sonar via Gaussian Splatting**
Sethuraman, Rucker, Bagoren, Kung, Amutha, Skinner — U. Michigan
arXiv: 2504.00159

**What it does:** Extends 3DGS to forward-looking imaging sonar (FLS). Renders range-azimuth images — the **same output format as mm25DGS**. Each Gaussian has acoustic reflectivity (scalar) and an azimuth-streaking probability. Novel rasterizer projects into sonar range-azimuth space.

**Radar translation:** Very close to our problem:
- Both render **range-azimuth images**
- Both are wave-based sensors with range gating
- Sonar has azimuth streaking (PSF leakage) ≈ radar has sidelobe leakage

Key difference: sonar is incoherent (intensity sum), radar is coherent (phasor sum).

**Would we benefit?** Yes:
- **Azimuth streaking model**: SonarSplat models per-Gaussian azimuth PSF leakage. The radar equivalent is antenna sidelobe spreading — each surfel's energy leaks into adjacent azimuth bins proportional to the antenna pattern sidelobes. Currently we model antenna gain per path but don't model the sidelobe PSF in the RA domain. Adding a per-Gaussian sidelobe model could improve azimuth accuracy.
- **Range-azimuth direct rendering**: SonarSplat projects directly into RA space. While less physically accurate than our ADC synthesis, it's an interesting comparison point.

---

### B2. NAS-GS (arXiv 2026)
**Noise-Aware Sonar Gaussian Splatting**
Xu, Jiang, Willners, Wang
arXiv: 2601.06285

**What it does:** Addresses sonar's **bidirectional** intensity accumulation and complex noise. Introduces "Two-Ways Splatting" for bidirectional intensity + transmittance, and a GMM-based noise model to prevent overfitting to sensor artifacts.

**Radar translation:**
- **GMM noise model in the loss**: Instead of treating GT RA as noise-free, NAS-GS models the GT noise distribution as a Gaussian mixture and incorporates it into the loss function. For mm25DGS, this could help with GT measurement noise — our GT ADC has thermal noise, quantization noise, and phase noise that we currently ignore in the MSE loss. A noise-aware loss would weight high-SNR regions more than noisy background.
- **Bidirectional accumulation**: Not directly applicable (radar is one-way TX→scatter→RX).

---

### B3. UltraGS (arXiv 2025)
**Real-Time Physically-Decoupled Gaussian Splatting for Ultrasound Novel View Synthesis**
Yang, Ruan, Cai, Dong, Yang, Dong, Jin, Dai
arXiv: 2511.07743

**What it does:** Physics-decoupled rendering for ultrasound: separates rendering into depth attenuation (Beer-Lambert), specular reflection (acoustic impedance mismatch), and volume scattering. Uses SH-DARS (SH + Depth Attenuation, Reflection, Scattering) — a physics-inspired rendering function.

**Radar translation:** The physics decomposition maps well to radar:

| Ultrasound component | Radar equivalent | Status in mm25DGS |
|---------------------|-----------------|-------------------|
| Depth attenuation (Beer-Lambert) | Free-space path loss (1/d^2) | Implemented |
| Specular reflection (impedance) | Fresnel reflection (permittivity) | Implemented (Jones BSDF) |
| Volume scattering | Diffuse/incoherent scattering | Implemented (SPM + broad lobes) |
| Depth-dependent gain (TGC) | Range-dependent ADC gain (not modeled) | Missing |

**Would we benefit?**
- **Decoupled physics components in the loss**: UltraGS trains each physics component separately with dedicated losses. For mm25DGS, we could add a range-profile loss (compare rendered vs GT range profiles marginalized over azimuth) alongside the full RA loss. This would specifically supervise the path-loss and range-dependent weighting.

---

### B4. Neural Acoustic Multipole Splatting — NAMS (ICASSP 2026)
**Neural Acoustic Multipole Splatting for Room Impulse Response Synthesis**
Baek, Choi
arXiv: 2509.17410

**What it does:** Places neural acoustic multipoles (monopoles, dipoles) in 3D space. Each multipole's emitted signal and directivity is predicted by a neural network. Synthesizes room impulse responses. The "rendering equation" is a multipole expansion of the **Helmholtz wave equation**.

**Radar translation:** This is the most physically rigorous wave-based Gaussian paper. The Helmholtz equation governs both acoustics and EM at the same level. For radar at 77 GHz:
- Each surfel could be modeled as a scattering multipole (monopole = isotropic, dipole = cos-theta pattern)
- The far-field BSDF lobes (Kirchhoff, SPM) are effectively multipole decompositions of the scattered field
- The multipole directivity is analogous to our BSDF × antenna pattern product

**Would we benefit?**
- **Principled connection to wave physics**: NAMS shows that Gaussians can directly encode solutions to the Helmholtz equation. For radar, this suggests that our BSDF (which is an approximation to the EM scattered field) could be replaced or augmented with learned multipole coefficients that directly satisfy Maxwell's equations in the far field. This is a longer-term research direction, not an immediate gap-closure item.

---

## Category C: Physics-Based Optical — Transferable Concepts

---

### C1. OMG: Opacity Matters in Material Modeling (ICLR 2025)
**Opacity Matters in Material Modeling with Gaussian Splatting**
Yong, Muniyandi et al. — CMU
arXiv: 2502.10988

**What it does:** Derives opacity from Beer-Lambert law: `alpha_i = 1 - exp(-n_i * sigma_v_i)` where `n_i` is number density and `sigma_v_i = f(material_i)` is a material cross-section predicted by an MLP from (albedo, roughness, specular). This couples opacity to material properties with proper gradient flow through both color AND opacity paths.

**Radar translation:** Directly applicable concept. Currently in mm25DGS, opacity is an independent learned scalar with no connection to the BSDF material parameters. But physically, a surfel's scattering efficiency SHOULD depend on its material:
- A metal surface (high eps_real, high eps_imag) should scatter nearly all incident energy → high effective opacity
- A low-permittivity dielectric (eps_real ≈ 1) should scatter almost nothing → low effective opacity

OMG's approach translated to radar:
```python
# Current: opacity is independent of materials
alpha = sigmoid(logit_opacity)

# OMG-inspired: opacity depends on material reflectivity
R_total = fresnel_power_avg(eps_real, eps_imag, cos_theta=0.5)  # avg reflectivity
alpha = sigmoid(logit_opacity) * (0.5 + 0.5 * R_total)
# Now opacity receives gradients through BOTH logit_opacity AND material params
```

**Would we benefit?** Yes — this creates a second gradient path from loss to material parameters (through opacity, not just through BSDF weight). This could improve material estimation, especially for low-reflectivity materials where BSDF gradients are small.

---

### C2. SVG-IR: Spatially-Varying Gaussian Splatting for Inverse Rendering (CVPR 2025)
Sun, Gao, Xie, Yang, Wang
openaccess.thecvf.com

**What it does:** Addresses the fact that each Gaussian spans a non-trivial spatial extent — a single Gaussian covers multiple material regions, but prior methods assign one constant BRDF. SVG-IR introduces spatially-varying BRDF parameters within each Gaussian, analogous to texture maps on triangles.

**Radar translation:** Directly relevant to our intra-surfel phase cancellation problem (Root Cause #2). Currently each surfel has a single set of 6 ITU material parameters. But a 5cm surfel at 77 GHz may span multiple material types (e.g., a wall-window boundary, or concrete with embedded rebar).

**Would we benefit?** Possibly for future work:
- **Intra-surfel material variation**: Instead of one BSDF per surfel, evaluate the BSDF at multiple sub-positions within the surfel and average the phasor contributions. This would partially capture the intra-surfel phase cancellation from material variation (complementing the VC-3DGS geometric apodization in F4).
- However, this dramatically increases computation (multiple BSDF evaluations per surfel per path). Not practical for gap closure; file as future research.

---

### C3. GaussianShader (CVPR 2024)
**3D Gaussian Splatting with Shading Functions for Reflective Surfaces**
Jiang, Tu, Liu, Gao, Long, Wang, Ma
arXiv: 2311.17977

**What it does:** Replaces SH color with a per-Gaussian shading function: diffuse + GGX specular + SH residual. The SH residual absorbs effects the analytic BRDF cannot capture (indirect lighting, interreflections).

**Radar translation:** Our Jones BSDF is the radar equivalent of their analytic BRDF. The concept of an **SH residual** on top of the physics model is interesting:

**Would we benefit?**
- **Learned residual on top of physics BSDF**: Add a small per-Gaussian learned correction `delta_f` to the BSDF output: `f_total = f_jones + delta_f`. This captures effects the 6-parameter ITU model cannot (surface waves, edge effects, near-field coupling). The residual would be a low-degree SH or simply a learned scalar per Gaussian.
- Risk: the residual could absorb errors that should be corrected in the physics model, making material parameters less physically meaningful. Best used as a diagnostic: if the residual is large, the physics model is inadequate for that surfel.

---

### C4. Relightable 3D Gaussians (ECCV 2024)
Gao, Gu, Lin, Li, Zhu, Cao, Zhang, Yao
arXiv: 2311.16043

**What it does:** Augments each Gaussian with normals + Disney BRDF + per-Gaussian directional incident lighting (64 Fibonacci-sampled directions). Uses **BVH-accelerated point-based ray tracing** for shadow computation on Gaussians.

**Radar translation:** The BVH shadow testing on Gaussians (not meshes) is relevant. Currently our F8 (pre-computed shadow visibility) uses the mesh BVH. But if surfels move during training, the mesh BVH becomes stale. Building a BVH directly on the Gaussian point cloud (as Relightable 3DGS does) would support dynamic shadow updates.

**Would we benefit?**
- **Point-based BVH for Gaussian shadow testing**: Build BVH on surfel positions (not mesh). Recompute every N iterations as positions evolve. More accurate than mesh-based shadows for trained Gaussians whose positions have moved away from mesh vertices.

---

### C5. GeoSplatting (ICCV 2025)
**Towards Geometry Guided Gaussian Splatting for Physically-Based Inverse Rendering**
Ye, Gao, Li, Chen, Chen
arXiv: 2410.24204

**What it does:** Anchors Gaussians to an extracted mesh surface. The mesh provides accurate normals and geometric consistency. Two-stage: scalar field → mesh extraction → Gaussian seeding → joint PBR optimization.

**Radar translation:** Our C3 mode already initializes one Gaussian per mesh vertex — conceptually similar. But GeoSplatting goes further: it constrains Gaussian positions and normals to stay on the mesh during optimization, preventing drift.

**Would we benefit?**
- **Mesh-constrained optimization**: For C3 mode (mesh-initialized), constrain positions to stay on or near the mesh surface. This prevents surfels from drifting to physically impossible locations. Implement as a soft constraint: `L_mesh = mean(point_to_mesh_distance^2)`.
- This would be especially useful WITH the live-scales change (F3): as scales grow, surfels could drift off the surface. The mesh constraint prevents this.

---

## Category D: Surfel / Optimization — Architectural Insights

---

### D1. 2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (SIGGRAPH 2024)
Huang, Yu, Chen, Geiger, Gao
arXiv: 2403.17888

**What it does:** Replaces 3D Gaussians with 2D oriented planar disks. Uses ray-splat intersection (not screen-space projection) for perspective-correct evaluation. Depth distortion + normal consistency losses enforce surface alignment.

**Radar translation:** Our surfels ARE 2D Gaussians — this is the closest optical primitive to ours. The key difference: 2DGS evaluates each surfel via ray-splat intersection (computing where each pixel's ray intersects the surfel plane), while we evaluate each surfel as a point scatterer at its center.

**Would we benefit?**
- **Ray-surfel intersection for exact path length**: Instead of evaluating each surfel at its center, compute where the TX-surfel-RX bistatic path actually intersects the surfel plane. This gives the exact geometric path length for that specific (TX, RX) pair. For large surfels at close range, this could improve phase accuracy compared to center-only evaluation.
- **Depth distortion loss (radar analog)**: 2DGS's depth distortion loss concentrates Gaussian alpha-weight along each ray. The radar analog: a "range concentration" loss that penalizes surfels whose contributions are spread across many range bins. This would encourage compact, well-localized surfels.

---

### D2. AbsGS: Recovering Fine Details (ACM MM 2024)
Ye et al.
arXiv: 2404.10484

**What it does:** Identifies "gradient collision" — large Gaussians in high-detail regions receive opposing gradients from different pixels that cancel in the sum, so the standard densification criterion `||grad_mu||` stays below threshold and splitting never triggers. Fix: use sum-of-absolute-values instead of norm:

```
g_i = sum_j |dL_j / d_mu_i,x|    (not ||sum_j dL_j / d_mu_i||)
```

**Radar translation:** This is directly applicable to our F7 (adaptive densification). In radar, "gradient collision" would occur when a large surfel covers range bins with both positive and negative RA errors — the gradients from overestimated bins cancel those from underestimated bins, making the position gradient norm small even though the surfel badly needs splitting.

**Would we benefit?** Yes:
- **Absolute-value gradient accumulation for densification**: Replace `grad_accum += model.positions.grad.norm(dim=-1)` with `grad_accum += model.positions.grad.abs().sum(dim=-1)`. This ensures large surfels in complex scattering regions get flagged for splitting even when gradient directions cancel.

---

### D3. GOF: Gaussian Opacity Fields (SIGGRAPH Asia 2024)
Yu, Sattler, Geiger
arXiv: 2404.10772

**What it does:** Defines a view-consistent 3D opacity field from 3D Gaussians by evaluating opacity along explicit ray-Gaussian intersections. The ray projects into the Gaussian's local frame, giving a 1D Gaussian profile with closed-form peak and width. Uses the minimum accumulated opacity across all training views as the opacity field value.

**Radar translation:** The ray-Gaussian 1D projection (`gamma = mu^T Sigma^-1 d / d^T Sigma^-1 d`, `beta = 1/sqrt(d^T Sigma^-1 d)`) is exactly the VC-3DGS formulation that we're already incorporating as F4 (range-domain apodization). GOF's contribution is the **modified densification criterion**: it accumulates `sum |dL/dmu_pixel|` (per-pixel absolute values) rather than `||sum dL/dmu||` (gradient norm). This is the same fix as AbsGS but discovered independently.

**Would we benefit?**
- **Same as AbsGS** (absolute-value gradient for densification). GOF's additional insight: **clone positions sampled from the Gaussian distribution** rather than duplicated at the center. For radar, this means cloned surfels start at a random position within the parent surfel's extent, giving better spatial coverage.

---

### D4. Scaffold-GS (CVPR 2024 Highlight)
Lu, Yu et al.
arXiv: 2312.00109

**What it does:** Two-level hierarchy: sparse anchor points + per-anchor MLPs predict child Gaussian attributes (color, opacity, scale, rotation) conditioned on view direction and distance. Importance-based densification: anchors grow where child Gaussians have high accumulated opacity contribution.

**Radar translation:** The anchor hierarchy concept is interesting for radar but requires careful adaptation:
- **Anchors as scattering centers**: Sparse anchors at key scattering locations (corners, edges, specular reflection points) predict the attributes of nearby surfels. This could reduce the parameter count while maintaining quality.
- **View-dependent prediction**: In Scaffold-GS, the MLP outputs are view-dependent. For radar, the "view" is the (TX, RX) pair — so the MLP would predict surfel BSDF/opacity conditioned on bistatic angle. This is physically motivated: the BSDF IS a function of (TX, RX) directions.

**Would we benefit?** Longer-term research direction. The anchor concept could replace the fixed one-Gaussian-per-vertex initialization, but requires significant architectural changes. Not a gap-closure item.

---

### D5. SuGaR: Surface-Aligned Gaussian Splatting (CVPR 2024)
Guédon, Lepetit
arXiv: 2311.12775

**What it does:** Regularizes 3DGS to produce flat, surface-aligned Gaussians. After regularization, Poisson reconstruction extracts a mesh, and a second refinement stage binds new thin Gaussians to mesh triangles. Each mesh triangle owns N flat Gaussians with fixed barycentric coordinates.

**Radar translation:** The "mesh-bound Gaussians" concept from SuGaR's Stage 2 is directly relevant. Instead of free-floating Gaussians that can drift off the surface, bind each Gaussian to a mesh triangle with fixed barycentric coordinates. The Gaussian position is computed from triangle vertices (which are shared parameters), not stored independently.

**Would we benefit?**
- **Mesh-bound surfel parameterization**: Instead of `model.positions` as free parameters, parameterize positions as `p_i = b0*v0 + b1*v1 + b2*v2` where `(b0,b1,b2)` are fixed barycentric coordinates and `(v0,v1,v2)` are shared vertex positions. This dramatically reduces the parameter count (N_vertices << N_surfels) and ensures surfels stay on the surface. Normal consistency is automatic. This is essentially a mesh parameterization with per-surfel material/opacity — a middle ground between our Stage B (per-vertex materials on mesh) and Stage C (free Gaussians).

---

### D6. OMG Opacity Model (ICLR 2025, revisited for architecture)

Already covered in C1. Additional architectural detail worth noting: OMG shows that coupling opacity to material parameters via a simple MLP creates a **second gradient pathway** from loss to materials. The gradient through opacity is:

```
dL/d_material = dL/d_alpha * d_alpha/d_sigma_v * d_sigma_v/d_material
```

This is independent of and additive with the standard gradient through the BSDF:

```
dL/d_material = dL/d_brdf * d_brdf/d_material
```

For mm25DGS, materials receive gradients only through the BSDF path. Adding the opacity path (even a simple one like `alpha = sigmoid(logit) * fresnel_reflectivity`) doubles the gradient signal to materials.

---

## Summary: What Would We Benefit From?

### High relevance (folded into gap closure plan)

| Paper | Idea | Radar translation | Benefit |
|-------|------|-------------------|---------|
| **AbsGS / GOF** | Absolute-value gradient for densification | Replace grad norm with abs-value sum in F6 | Prevents gradient collision, triggers splitting in complex regions |
| **GOF** | Clone from Gaussian distribution (not center) | Cloned surfels start at random offset within parent | Better spatial coverage after densification |

### Medium relevance (future extensions worth exploring)

| Paper | Idea | Radar translation | Benefit |
|-------|------|-------------------|---------|
| **GSRF** | Learned complex residual | delta_f residual on top of physics BSDF | Captures effects beyond 6-param ITU (see detailed analysis below) |
| **NAMS** | Helmholtz multipole decomposition | Surfels as EM multipole sources | Principled wave-physics connection (see detailed analysis below) |
| **SVG-IR** | Intra-Gaussian material variation | Sub-surfel BSDF evaluation | Captures material boundaries within surfels |
| **GeoSplatting** | Mesh-constrained optimization | L_mesh = point-to-mesh dist penalty | Prevents surfel drift |
| **SuGaR** | Mesh-bound barycentric parameterization | Positions from shared vertex params | Reduced param count, guaranteed surface |
| **Relightable 3DGS** | Point-based BVH for Gaussian shadows | BVH on surfel positions, not mesh | Dynamic shadow updates as surfels move |
| **2DGS** | Ray-surfel intersection | Exact bistatic path through surfel plane | Better phase accuracy for large/close surfels |

### Lower relevance (with reasoning)

| Paper | Idea | Why lower relevance for us |
|-------|------|---------------------------|
| **OMG** | Material-coupled opacity | Our BSDF already returns scattering power per direction — opacity coupling would double-count Fresnel reflectivity that's already inside f_cos. Opacity in our system is a fill-factor, not a scattering efficiency. |
| **SonarSplat** | RA sidelobe PSF model | We render to ADC and apply FFT identically to GT — sidelobe structure is captured automatically by the shared FFT. No need for explicit PSF. |
| **RadarSplat** | Noise probability eta_i | mmIR achieved strong results without noise modeling. Our physics BSDF handles the scattering physics; remaining gap is representational, not noise-related. Interesting conceptually but not a priority. |
| **NAS-GS** | GMM noise-aware loss | Our simple MSE loss worked well for mmIR. Over-engineering the loss function risks masking real model deficiencies. |
| **RF-3DGS** | Gaussians as scattering clusters | Operates at wireless channel level, not individual scatterer level. |
| **SAR-GS** | SAR forward model | Different radar geometry (side-looking vs forward-looking). |
| **Scaffold-GS** | Anchor hierarchy + MLP | Major architectural redesign, not incremental. |

---

## Deep Dive: GSRF — Complex-Valued Gaussians for RF

### What GSRF does that we don't

GSRF (Yang et al., NeurIPS 2025) is the closest existing work to mm25DGS in terms of physics:
- Both use **complex-valued accumulation** (amplitude + phase)
- Both model **coherent wave propagation** with interference
- Both target **RF frequencies** (though GSRF is sub-6 GHz WiFi, we are 77 GHz mmWave)

The key architectural difference: GSRF replaces the physics BSDF with a **learned Fourier-Legendre directional basis**. Each Gaussian stores complex coefficients for a truncated Fourier-Legendre expansion:

```
E_i(d, f) = sum_l sum_m  c_lm^i * Y_l(d) * F_m(f)
```

where `Y_l` are Legendre polynomials over direction, `F_m` are Fourier basis functions over frequency, and `c_lm^i` are complex-valued learned coefficients per Gaussian. The total received field at a point is the coherent sum over all Gaussians:

```
E_total = sum_i  E_i(d_i, f) * G(x, mu_i, Sigma_i) * exp(j * k * ||x - mu_i||)
```

### What a radar translation would look like

For mm25DGS, the GSRF approach suggests a **hybrid physics + learned residual**:

```python
# Current rendering equation per path:
weight = C * sqrt(f_cos_jones * G_tx * G_rx * dOmega / d_tx^2) * alpha

# GSRF-inspired hybrid:
f_learned = sum_l  c_l^i * P_l(cos_theta_bistatic)   # learned Legendre residual
weight = C * sqrt((f_cos_jones + f_learned) * G_tx * G_rx * dOmega / d_tx^2) * alpha
```

Where `c_l^i` are per-Gaussian complex Legendre coefficients (e.g., 4-8 coefficients for degrees 0-3/7). The physics BSDF handles the dominant scattering behavior, and the Legendre residual captures:
- Surface waves / creeping waves along curved surfaces
- Near-field coupling effects between closely-spaced surfels
- Multiple scattering within a surfel's local neighborhood
- Any systematic bias in the 6-parameter ITU model

**Implementation cost:** Low — add 8-16 learnable parameters per Gaussian (4-8 complex coefficients). Evaluate as a dot product with pre-computed Legendre basis at the bistatic angle. ~5% compute overhead.

**Risk:** The learned residual could absorb errors that should be corrected by the physics model, making material parameters less physically meaningful. Mitigation: add a sparsity penalty on the residual `L_sparse = lambda * mean(|c_l|)`, and monitor the ratio `|f_learned| / |f_jones|` — if it exceeds ~0.3, the physics model is inadequate.

**When to try this:** After the gap closure stages (F1-F7). If a persistent gap remains that cannot be explained by coverage, scale, or training dynamics, the learned residual would diagnose whether the BSDF model itself is the bottleneck.

---

## Deep Dive: NAMS — Helmholtz Multipole Splatting for Acoustics

### What NAMS does

NAMS (Baek & Choi, ICASSP 2026) places neural acoustic multipoles in 3D space. Each multipole is a solution to the Helmholtz equation:

```
(nabla^2 + k^2) p(x) = 0
```

A monopole source radiates isotropically: `p(x) = (A / r) * exp(jkr)`. A dipole radiates with cos(theta) directivity. Higher-order multipoles produce more complex radiation patterns. NAMS learns the multipole coefficients (amplitude and directivity) via a neural network, then synthesizes room impulse responses by summing contributions from all multipoles at the listener position.

### The deep connection to radar scattering

At 77 GHz (lambda = 3.9mm), the scattered far-field from a finite surface element IS a multipole expansion. The Physical Optics (PO) approximation — which is the basis of our Kirchhoff BSDF — is equivalent to saying that the scattered field from a flat conducting plate is dominated by the specular (zeroth-order) term. Higher-order multipole terms correspond to edge diffraction, surface currents, and creeping waves.

Our Jones BSDF (Kirchhoff + SPM + CBS) is effectively a parametric approximation to the first few multipole orders:
- **Kirchhoff (GGX lobe)** ≈ specular term (monopole + focused directivity)
- **SPM (vMF lobe)** ≈ first-order diffuse scattering (dipole-like)
- **CBS** ≈ retro-reflection enhancement (specific directional structure)

### What a radar translation would look like

Replace or augment the parametric BSDF with a learned multipole expansion per surfel:

```python
# Per surfel i, scattered field in direction (theta, phi) relative to normal:
E_scat_i(theta, phi) = sum_{l=0}^{L} sum_{m=-l}^{l}  a_lm^i * Y_lm(theta, phi)
```

where `a_lm^i` are complex-valued spherical harmonic coefficients of the scattered field, and `Y_lm` are the spherical harmonics.

For a flat surfel with known material, the coefficients `a_lm` could be:
- **Initialized from the Jones BSDF**: compute `f_jones(theta, phi)` at a grid of directions, then project onto the SH basis to get initial `a_lm` values. This ensures the learned representation starts from the physics.
- **Fine-tuned during training**: allow the `a_lm` to deviate from the BSDF prediction. The deviation captures effects the parametric model misses.

**Key advantage over the GSRF Legendre approach:** SH coefficients have a direct physical interpretation as multipole orders. Degree 0 = monopole (isotropic scattering), degree 1 = dipole (cos-theta), degree 2 = quadrupole, etc. The number of coefficients needed depends on `k * a` where `a` is the surfel radius — electrically large surfels need more orders.

**For our surfel sizes:** A surfel with radius ~5cm at 77 GHz has `k*a ~ 2*pi*0.05/0.0039 ~ 80`. This is very large — meaning the scattered field has high-frequency angular structure requiring many SH coefficients (L ~ k*a ~ 80, giving ~6400 coefficients). This is impractical per surfel.

**Resolution:** Use the physics BSDF as the broadband predictor and only learn a **low-order correction** (L = 2-3, ~9-16 complex coefficients). The high-frequency structure is handled by the parametric Kirchhoff/SPM model; the SH residual captures low-frequency systematic errors.

**When to try this:** This is a research direction, not a gap-closure item. It's worth exploring if:
1. The gap closure stages (F1-F7) leave a persistent residual
2. The GSRF Legendre residual (simpler) shows that the BSDF is a bottleneck
3. The goal shifts from matching mmIR to exceeding it

---

## Recommended Addition to the Gap Closure Plan

### AbsGS-style absolute gradient accumulation (already folded into F6)

In Stage F6 (adaptive densification), the gradient accumulation has been updated to use absolute-value sum instead of norm:

```python
# AbsGS fix (already in gap_closure_plan.md):
active_grads = model.positions.grad.abs().sum(dim=-1)  # NOT .norm()
```

Source: AbsGS (ACM MM 2024), GOF (SIGGRAPH Asia 2024). Zero implementation cost.
