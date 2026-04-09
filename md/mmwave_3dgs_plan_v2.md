# mmWave 2.5D Gaussian Splatting: Implementation Plan (v2)

## Changelog from v1

- **Representation**: 3DGS -> 2.5DGS. Each Gaussian is a 2D surfel (disc) with two lateral scale axes; thickness is modelled as an ITU material parameter rather than a geometric axis. The surface normal is intrinsic to the surfel orientation (no separate normal parameter needed).
- **Multi-bounce**: Proxy mesh eliminated. Multi-bounce uses direct ray-surfel intersection against the 2D Gaussian discs, which is analytically simple and differentiable.
- **Gaussian count**: No voxel downsampling. Use the full LiDAR point cloud. During ADC synthesis, cull by cumulative contribution threshold (top 95-98%) rather than reducing the representation.
- **Decided**: Per-Gaussian exact phase (Strategy A). Hybrid PyTorch + DrJit. Full 3D RAE grid with RA-slice training loss. Per-Gaussian exact phase for large splatting footprints.

---

## 1. Motivation & Context

### Current system (mmIR): mesh-based differentiable inverse renderer
- LiDAR point cloud frames are aggregated and Poisson-reconstructed into a triangle mesh (~200K triangles).
- Rays are traced from RX elements through the mesh; each hit position is evaluated by all 12TX x 16RX pairs for MIMO phase coherence.
- Budget limit of ~1500 reservoir hits/RX means the mesh cannot be fully covered without unrealistic ray counts.
- Training takes 10-15 min/scene at 500 iterations; a single dense-array render takes ~6s.

### Proposed system (mm3DGS): 2.5D Gaussian splatting for mmWave
Replace the mesh with 2.5D Gaussian surfels and replace RX-centric reservoir ray tracing with splatting to a Range-Azimuth-Elevation (RAE) grid. ADC is synthesized directly, preserving compatibility with FFT processing.

### Advantages over mesh-based mmIR
1. **Full scene coverage.** Every Gaussian contributes every iteration (no sampling budget).
2. **Simpler single-bounce path.** Splatting gives per-Gaussian amplitude weights; ADC synthesis uses exact per-Gaussian phase.
3. **ADC-level synthesis.** Rendering to ADC means the same FFT pipeline applies to both rendered and ground-truth data, enabling modelling of windowing, sidelobes, and cross-bin leakage.
4. **Native multi-bounce without mesh construction.** Ray-surfel intersection against 2D Gaussian discs is analytical and differentiable -- no proxy mesh needed.
5. **No Poisson reconstruction.** Eliminates the LiDAR-to-mesh pipeline and its artefacts (phantom triangles, topology errors, over-smoothing).

### Advantages over existing radar 3DGS methods (e.g. RadarGS, mmGaussian)
Existing methods splat Gaussians directly to a 2D RA image, supervise with magnitude loss only, and cannot model FFT artefacts or multi-bounce. By splatting to a 3D RAE volume and synthesising raw ADC, mm3DGS:
- Preserves phase coherence across the virtual array.
- Allows identical FFT processing on rendered and measured data.
- Enables multi-bounce modelling (Section 6).

---

## 2. 2.5D Gaussian Representation

### 2.1 Why 2.5DGS (not 3DGS)

Standard 3DGS uses full 3D ellipsoids -- three scale axes define a volume. For mmWave surfaces this is over-parameterised: radar scattering is a surface phenomenon governed by the ITU slab model. 2DGS (flat surfels/tassels) is the natural choice, and the ray tracing community has already adopted 2D Gaussians for exactly this reason: they have well-defined surface normals intrinsic to their orientation, enabling physically-based surface interactions.

Our "2.5D" extension adds one detail: the ITU material model already carries a `thickness` parameter that governs multi-layer Fresnel reflections inside the slab. This thickness is not a geometric axis (it doesn't inflate the surfel into a volume) but it gives the surfel a physical depth for electromagnetic purposes. Hence "2.5D": geometrically a 2D disc, electromagnetically a slab with thickness.

### 2.2 Per-Gaussian parameters

Each 2.5D Gaussian g is defined by:

| Symbol | Description | Shape | Init | Learnable |
|--------|-------------|-------|------|-----------|
| mu_g | Centre position (world frame) | (3,) | LiDAR point | Yes |
| q_g | Rotation quaternion | (4,) | From LiDAR normal | Yes |
| s_g | Log-scale (2 lateral axes) | (2,) | From local PCA | Yes |
| alpha_g | Logit-opacity | (1,) | 0 (sigmoid -> 0.5) | Yes |
| mat_g | Material parameters (raw-space) | (6,) | ITU concrete default | Yes |

Material parameters (6 DOF, same as mmIR):
```
eps_real, eps_imag, sigma_h, l_c, tau, thickness
```
with the same reparameterization (exp/sigmoid) and bounds as `train.py:ParameterManagerSionna`.

**Notable**: No separate `n_g` normal parameter. The normal is derived from the rotation: `n_g = R(q_g)[:, 2]` (the third column of the rotation matrix). This is discussed further in Section 2.4.

### 2.3 Covariance construction

The surfel's tangent frame from rotation R(q_g):
```
t1 = R(q_g)[:, 0]    # first tangent direction
t2 = R(q_g)[:, 1]    # second tangent direction
n  = R(q_g)[:, 2]    # surface normal (outward)
```

The 2D covariance in the tangent plane (rank-2 in 3D):
```
Sigma_g = exp(s_g[0])^2 * t1 @ t1^T  +  exp(s_g[1])^2 * t2 @ t2^T
```

This is a degenerate (rank-2) 3D covariance -- the Gaussian has zero extent in the normal direction. It defines a disc in the plane spanned by (t1, t2), centred at mu_g.

For rasterisation we use the 2x2 tangent-plane covariance:
```
Sigma_2D = diag(exp(s_g[0])^2, exp(s_g[1])^2)
```
projected to RAE space via the composed Jacobian (Section 3.2).

### 2.4 Normal consistency (resolved)

In v1, normal consistency was an open question with three options. The 2.5DGS representation resolves this: **the normal is the surfel's orientation, derived from the rotation quaternion q_g.** There is no ambiguity or redundancy.

This is cleaner than any of the v1 options because:
1. The normal is intrinsic to the representation (no separate parameter to keep consistent).
2. Rotating q_g simultaneously rotates the surfel shape AND the normal.
3. BSDF evaluation uses `n_g = R(q_g)[:, 2]`, which is differentiable w.r.t. q_g.
4. The optimizer can rotate a surfel to align with the surface, which jointly optimises the scattering geometry and the splatting footprint.

### 2.5 Initialization from LiDAR

1. Load the aggregated LiDAR point cloud (`data/seq_X_frame_Y/scene/pcl.npy`, columns x,y,z,nx,ny,nz,intensity).
2. **Use all points** -- no voxel downsampling. Typical count: 100K-300K points per scene (see Section 2.6 for why this is tractable).
3. Set `mu_g` = point position.
4. Set `q_g` from the LiDAR normal: align R(q_g)[:, 2] with (nx, ny, nz). The tangent directions (columns 0, 1) are initialised from local PCA of the k-nearest neighbours (k=20).
5. Set `s_g` from the two largest PCA eigenvalues of the local neighbourhood, clamped to [1cm, 50cm] (log-space).
6. Initialize material `mat_g` from the same ITU concrete default as mmIR.
7. Optional: initialise additional Gaussians at positions where mmIR's error map shows under-fitting.

### 2.6 Contribution-based culling (instead of downsampling)

Rather than discarding points upfront, we keep the full set and dynamically cull during ADC synthesis:

1. **Precompute per-Gaussian amplitude estimate** (cheap, once per iteration):
   ```
   A_g_est = sigmoid(alpha_g) * |cos(angle(n_g, radar_dir))| * G_antenna(az_g, el_g) / r_g^2
   ```
   This approximates each Gaussian's contribution magnitude without full BSDF/TX-RX expansion.

2. **Sort** Gaussians by A_g_est in descending order.
3. **Cumulative threshold**: Select the top-K Gaussians accounting for X% (default 97%) of the total estimated amplitude: `cumsum(A_g_est) / sum(A_g_est) <= 0.97`.
4. **Only these K active Gaussians** enter the MIMO-expanded ADC synthesis loop (the expensive part).

**Why this works:**
- Parameter storage for 300K Gaussians: 300K x 16 floats x 4 bytes = 19 MB. Adam state: 38 MB. Total: ~57 MB -- negligible on a 24 GB GPU.
- The cost bottleneck is ADC synthesis: O(K x N_tx x N_rx x K_adc). Culling from 300K to ~100K active Gaussians cuts this by ~3x.
- **All Gaussians still receive gradient updates** for their opacity and position (through the culling threshold). A Gaussian just below the cutoff gets gradient signal that would increase its amplitude and promote it to the active set next iteration. No information is permanently lost.
- Stochastic inclusion: Every M iterations (e.g. M=10), include ALL Gaussians regardless of threshold to prevent dead zones.

---

## 3. Splatting to Range-Azimuth-Elevation Grid

### 3.1 RAE grid definition

The RAE grid is fully determined by the radar configuration (reuse logic from `eval_3d_occupancy.py` / `single_view_viz.py:build_fov_wireframe()`):

| Dimension | # Bins | Spacing |
|-----------|--------|---------|
| Range | N_r = 256 | Delta_R = c / (2B) ~ 59mm |
| Azimuth | N_az = 127 (128 FFT bins, DC removed) | arcsin-spaced, ~0.016 rad near boresight |
| Elevation | N_el = 127 (128 FFT bins, DC removed) | arcsin-spaced, ~0.016 rad near boresight |

Bin centres (r_k, theta_az_m, theta_el_n) are computed exactly as in `make_angle_grids_np()` and stored as a precomputed lookup table.

Cartesian bin-centre positions:
```
x_b = r_k * cos(theta_el_n) * sin(theta_az_m)
y_b = r_k * cos(theta_el_n) * cos(theta_az_m)
z_b = r_k * sin(theta_el_n)
```

### 3.2 Cartesian-to-RAE Jacobian

Given Gaussian centre mu_g = (x, y, z) in the radar's local frame:
```
r   = sqrt(x^2 + y^2 + z^2)
az  = atan2(x, y)
el  = asin(z / r)
rho = sqrt(x^2 + y^2)
```

Jacobian J_RAE of the (r, az, el) <- (x, y, z) mapping:
```
J_RAE = [ x/r           y/r          z/r          ]   # d(r)/d(x,y,z)
        [ y/rho^2      -x/rho^2      0            ]   # d(az)/d(x,y,z)
        [-xz/(r^2*rho) -yz/(r^2*rho) rho/r^2      ]   # d(el)/d(x,y,z)
```

For a 2.5D Gaussian, the 3D covariance is rank-2: Sigma_g = [t1 t2] @ Sigma_2D @ [t1 t2]^T. The projected covariance in RAE space:
```
Sigma_RAE = J_RAE @ Sigma_g @ J_RAE^T
```

This is a 3x3 matrix but rank-2 (the surfel is flat). For rasterisation purposes, we use the 3x3 form and evaluate the 3D Gaussian weight normally -- points off the surfel plane get exponentially suppressed because the covariance has near-zero extent in the normal direction.

### 3.3 Rasterisation

For each active Gaussian g:
1. Compute its RAE centre (r_g, az_g, el_g) and projected covariance Sigma_RAE.
2. Find the bounding box in RAE bin indices (+/-3 sigma in each dimension, using the eigenvalues of Sigma_RAE).
3. For each bin (k, m, n) within the bounding box, evaluate the Gaussian weight:
   ```
   delta = [r_k - r_g, az_m - az_g, el_n - el_g]
   w_g(k,m,n) = exp(-0.5 * delta^T @ Sigma_RAE^{+} @ delta)
   ```
   where Sigma_RAE^{+} is the pseudo-inverse (handles the rank-2 case).
4. Multiply by opacity: `alpha_g_effective = sigmoid(alpha_g) * w_g(k,m,n)`.

### 3.4 Per-bin aggregation: amplitude compositing

Unlike optical 3DGS which uses front-to-back alpha compositing (because closer objects fully occlude farther ones), radar signals at different ranges do NOT occlude each other -- they occupy different range bins. However, within the same RAE bin, multiple Gaussians interfere.

**Aggregation strategy (per RAE bin):**

Sort Gaussians contributing to each bin by range (front-to-back within that bin). For each bin (k, m, n), compute:

```
A_bin = sum_g [ alpha_g_eff * BSDF_g * T_g ]
T_g = prod_{j < g, in same bin} (1 - alpha_j_eff)    # transmittance
```

where `BSDF_g` is the evaluated BSDF amplitude (see Section 4.2). The transmittance models progressive attenuation of the wavefront through overlapping Gaussians within the same resolution cell.

**Important**: Gaussians in DIFFERENT range bins contribute independently to different range samples in the ADC. The compositing is only among Gaussians mapped to the SAME bin.

### 3.5 Efficient GPU implementation

The rasterisation can be implemented as a CUDA kernel following the tile-based approach of standard 2DGS, extended to 3D RAE coordinates:

1. **Bin assignment**: Parallel over Gaussians. Each Gaussian computes its bounding box in RAE indices, writes (Gaussian_id, bin_key) pairs.
2. **Sort**: Sort pairs by bin_key (radix sort on GPU).
3. **Accumulate**: Parallel over bins. Each bin iterates over its sorted Gaussians, accumulates amplitude via alpha compositing.

For the initial implementation, a simpler scatter-add approach (no sorting, just atomic adds) is sufficient and easier to implement in PyTorch.

---

## 4. ADC Synthesis (Single Bounce)

### 4.1 Phase computation: per-Gaussian exact phase

Each Gaussian contributes to ADC with its own phase computed from its exact centre position mu_g. The splatting weight determines the amplitude; the phase comes from the true Gaussian position.

For Gaussian g, TX element i, RX element j:
```
d_tx_g = ||mu_g - p_tx_i||
d_rx_g = ||mu_g - p_rx_j||
R_tot_g = d_tx_g + d_rx_g
tau_g = R_tot_g / c

phi_g(t_k) = 2*pi * (f_0 * tau_g + S * tau_g * t_k)

ADC_ij[k] += A_g * cos(phi_g(t_k))    # real
ADC_ij[k] += A_g * sin(phi_g(t_k))    # imag
```

where `A_g` is the aggregate amplitude from splatting weight, BSDF, antenna gain, and path loss.

**Complexity**: O(K_active * N_tx * N_rx * K_adc). With contribution culling (Section 2.6) reducing K_active to ~100K and N_tx=12, N_rx=16, K_adc=256: ~50 billion multiply-adds. On an RTX 4090 (~80 TFLOPS FP32), this is <1 second per forward pass.

**Key advantage**: Exact phase preserves sub-bin interference patterns, enabling the FFT to correctly resolve scatterers.

### 4.2 BSDF evaluation

Reuse the existing `BSDFmmWaveScalar` or `BSDFmmWaveJones` from `mmir/renderer/bsdf/`. For each Gaussian g, evaluate:

```
wo = normalize(p_rx_j - mu_g)       # outgoing direction (toward RX)
wi = normalize(p_tx_i - mu_g)       # incoming direction (from TX)
n  = R(q_g)[:, 2]                   # surfel normal (from rotation)

f_cos = BSDF.eval_f_cos(wo, wi, n, mat_g)
```

The BSDF evaluation is per-Gaussian-per-TX-RX, reusing the KA+SPM hybrid model with ITU slab Fresnel. This is identical to mmIR's BSDF evaluation but at Gaussian centres instead of ray hit positions.

Note: the surfel orientation naturally handles the double-sided normal convention -- if `dot(n, wi) < 0`, flip n. This is the same convention as mmIR's mesh renderer.

### 4.3 Antenna gain

Evaluate TX and RX antenna patterns at each Gaussian centre:
```
G_tx = pattern_tx(normalize(mu_g - p_tx_i), boresight_tx_i)
G_rx = pattern_rx(normalize(p_rx_j - mu_g), boresight_rx_j)
```

Reuse `evaluate_combined_gain()` from `mmir/sensor/element_patterns.py`.

### 4.4 Path loss

Apply inverse-square law from the radar equation:
```
path_loss_g = 1 / (d_tx_g * d_rx_g)           # E-field amplitude model
```

Match the convention from mmIR's `synthesize_end_to_end()` (RX-centric MC normalisation with sqrt(radar_constant) / d^2 terms). Since we are not doing Monte Carlo sampling (all Gaussians contribute deterministically), the 1/pdf MC correction term is replaced by the splatting weight.

### 4.5 Complete single-bounce ADC synthesis

Putting it all together:

```python
# Vectorised over active Gaussians (K_active) and all TX-RX pairs
# All loops become parallel GPU operations

for g in active_gaussians:            # parallel over ~100K active
    n_g = R(q_g)[:, 2]               # surfel normal
    for i in range(N_tx):             # parallel over 12 TX
        for j in range(N_rx):         # parallel over 16 RX
            # Geometry
            d_tx = ||mu_g - p_tx[i]||
            d_rx = ||mu_g - p_rx[j]||
            R_tot = d_tx + d_rx
            tau = R_tot / c

            # Amplitude
            wo = normalize(p_rx[j] - mu_g)
            wi = normalize(p_tx[i] - mu_g)
            f_cos = BSDF.eval_f_cos(wo, wi, n_g, mat_g)
            G_tx = antenna_tx(normalize(mu_g - p_tx[i]))
            G_rx = antenna_rx(normalize(p_rx[j] - mu_g))
            A = alpha_g_eff * f_cos * G_tx * G_rx / (d_tx * d_rx) * radar_const

            # Phase (per ADC sample)
            phi_const = 2*pi * f0 * tau
            phi_slope = 2*pi * S * tau
            for k in range(K):        # vectorised over 256 ADC samples
                phi_k = phi_const + phi_slope * t[k]
                adc_real[i, j, k] += A * cos(phi_k)
                adc_imag[i, j, k] += A * sin(phi_k)
```

In practice, all loops are eliminated by broadcasting:
- Gaussians: (K_active, 1, 1, 1)
- TX: (1, N_tx, 1, 1)
- RX: (1, 1, N_rx, 1)
- ADC: (1, 1, 1, K_adc)

The full tensor has shape (K_active, N_tx, N_rx, K_adc) which is reduced by summing over the Gaussian dimension via `scatter_add` or direct summation.

### 4.6 Gradient flow and differentiability

The full forward pass is differentiable with respect to:
- **Gaussian positions** mu_g: Through distances d_tx, d_rx (affects phase AND amplitude).
- **Gaussian rotations** q_g: Through the surfel normal n_g = R(q_g)[:, 2] (affects BSDF) and through the projected covariance (affects splatting weights).
- **Gaussian scales** s_g: Through the projected covariance (affects splatting footprint).
- **Material parameters** mat_g: Through BSDF evaluation.
- **Opacity** alpha_g: Through the effective amplitude.

**Phase gradient handling** (same as mmIR): At lambda ~ 4mm, a 1mm shift in mu_g causes a ~1.6 radian phase change, creating extremely noisy gradients. **Solution: detach phase from the AD graph** (same approach as mmIR's `enable_grad_phase=False`), flowing gradients only through amplitude. The phase is correct in the forward pass but treated as a constant during backprop.

---

## 5. FFT Processing & Loss

### 5.1 ADC-to-RA pipeline

The synthesised ADC `adc[N_tx, N_rx, K, 2]` is processed identically to ground truth:

1. **TX-RX to Virtual Array mapping** (reuse `ra_utils.py:adc_to_ra_complex()`):
   ```
   VA[az_idx, el_idx, k] = mean over (tx, rx) pairs mapping to same VA position
   ```
2. **Range FFT** (fast-time): Hann window + FFT over K=256 samples.
3. **Azimuth FFT**: Hann window + FFT over N_az virtual elements (zero-pad to 128).
4. **RA image**: Take magnitude of (az, range) slice at elevation 0.

This is fully differentiable via PyTorch's `torch.fft.fft`.

### 5.2 Loss functions

Primary loss (match mmIR training):
```
L_RA = MSE(normalize(|RA_render|), normalize(|RA_gt|))
```

Additional losses:
- **ADC magnitude loss**: L_ADC_mag = MSE(|ADC_render|, |ADC_gt|) -- supervises before FFT.
- **Phase loss** (optional, weighted by SNR): L_phase = weighted_MSE(angle(RA_render), angle(RA_gt)).
- **Material regularization**: Laplacian smoothness on material parameters across neighbouring Gaussians (k-NN graph).
- **Gaussian regularization**: Penalise Gaussians with extreme scales or very low opacity.

Reuse the multi-chirp loss from `mmir/losses/multi_chirp_loss.py` for per-chirp alignment if using multiple chirps.

### 5.3 Advantage: FFT effects are modelled correctly

Because we synthesise ADC before applying FFT:
- **Windowing effects** (Hann sidelobes) apply identically to rendered and measured data.
- **Spectral leakage** from scatterers not exactly at bin centres is captured.
- **Zero-padding artefacts** in the azimuth FFT are matched.

Methods that render directly to RA (existing radar 3DGS) cannot capture these effects.

---

## 6. Multi-Bounce Extension

### 6.1 The multi-bounce problem

Multi-bounce radar returns arise when a transmitted wave reflects off surface A, then surface B, then returns to the receiver. The path length is d(TX, A) + d(A, B) + d(B, RX), producing a return at a range larger than any single object's range.

### 6.2 Ray-surfel intersection (no mesh required)

The 2.5DGS representation provides a natural primitive for multi-bounce ray tracing: the oriented disc. Each Gaussian IS a disc with centre mu_g, normal n_g = R(q_g)[:, 2], and lateral extents (exp(s_g[0]), exp(s_g[1])).

**Ray-surfel intersection (analytical, differentiable):**

Given a ray `r(t) = o + t * d` and surfel g:

```
# 1. Ray-plane intersection
t_hit = dot(mu_g - o, n_g) / dot(d, n_g)
if t_hit < 0: miss   # surfel behind ray origin

# 2. Intersection point
p_hit = o + t_hit * d

# 3. Project to surfel-local coordinates
delta = p_hit - mu_g
u = dot(delta, t1_g)    # t1_g = R(q_g)[:, 0]
v = dot(delta, t2_g)    # t2_g = R(q_g)[:, 1]

# 4. Evaluate 2D Gaussian weight at intersection
w = exp(-0.5 * (u^2 / exp(s_g[0])^2 + v^2 / exp(s_g[1])^2))

# 5. Effective opacity at intersection
alpha_hit = sigmoid(alpha_g) * w
```

If `alpha_hit > epsilon`, the ray has hit surfel g with opacity alpha_hit. This is differentiable w.r.t. all Gaussian parameters (mu_g, q_g, s_g, alpha_g).

### 6.3 Multi-bounce forward model

**Second bounce:**
1. Select first-bounce Gaussians for multi-bounce tracing (importance-sampled by single-bounce amplitude, Section 6.5).
2. For each selected first-bounce Gaussian g1:
   a. Sample N_secondary outgoing directions from BSDF(mat_g1, wi, n_g1).
   b. For each sampled direction omega_o, trace a ray from mu_g1 in direction omega_o.
   c. Test ray-surfel intersection against all other Gaussians (or an accelerated subset, Section 6.4).
   d. Accumulate hits front-to-back with alpha compositing until cumulative opacity > 0.95:
      ```
      T = 1.0                  # running transmittance
      for g2 in hit_surfels sorted by t_hit:
          alpha_2 = alpha_hit(g2)
          weight_2 = T * alpha_2
          T *= (1 - alpha_2)

          # Two-bounce contribution
          d_path = d(TX_i, mu_g1) + d(mu_g1, mu_g2_hit) + d(mu_g2_hit, RX_j)
          A_2b = A_g1 * f_BSDF(g1, omega_o) * weight_2 * f_BSDF(g2, ...) * G_rx / d(...)
          phi_2b(t_k) = 2*pi * (f0 + S*t_k) * d_path / c
          ADC[i,j,k] += A_2b * cos(phi_2b(t_k))
          # ... (and imag part)

          if T < 0.05: break  # early termination
      ```
3. **Higher bounces**: Apply recursively with Russian roulette termination at bounce >= 3 (same as mmIR, p_terminate = 0.5).

### 6.4 Spatial acceleration for ray-surfel intersection

Brute-force ray-surfel testing (N_rays x N_gaussians) is feasible for moderate counts:
- 5000 first-bounce x 8 rays x 100K surfels = 4 billion intersection tests
- Each test: ~20 FLOPs (dot products, exp)
- Total: ~80 GFLOPS, <1s on RTX 4090

For larger scenes or more rays, use spatial acceleration:

**Option A: Uniform grid.** Divide the scene into a 3D grid. Each cell stores indices of Gaussians whose centres fall within it. Ray traversal visits cells in order (DDA algorithm). Implementation: ~100 lines of PyTorch.

**Option B: BVH via DrJit/Mitsuba.** Register surfel centres as a point cloud with Mitsuba's `ShapeKDTree`. Use `mi.Ray3f` for hardware-accelerated traversal. Requires a custom intersection kernel (or approximate by testing a small sphere at each Gaussian centre).

**Option C: KD-tree (PyTorch3D / scipy).** Build a KD-tree over Gaussian centres. For each ray, query the K nearest Gaussians within a cylinder along the ray direction. Fast to build, well-suited to nearest-neighbour style queries.

**Recommendation**: Start with brute-force (simplest, fast enough for initial experiments). Move to Option A if scaling beyond 200K Gaussians.

### 6.5 Stochastic Gaussian sampling for multi-bounce

Not all Gaussians need to participate in multi-bounce:
1. Compute single-bounce amplitudes for all active Gaussians.
2. Build a CDF over Gaussians weighted by their single-bounce amplitude.
3. Sample N_multi Gaussians (e.g. 2000-5000) as first-bounce origins.
4. For each sampled first-bounce Gaussian, trace N_secondary rays (e.g. 4-8).
5. Accumulate multi-bounce ADC contributions with proper MC normalisation (1/pdf_sample).

This makes multi-bounce cost controllable and independent of the total Gaussian count.

### 6.6 Gradient flow through multi-bounce

The ray-surfel intersection is differentiable:
- t_hit depends on mu_g and q_g (through n_g).
- p_hit depends on t_hit.
- (u, v) depend on p_hit and the tangent frame.
- w depends on (u, v) and s_g.
- The multi-bounce path length depends on mu_g1 and p_hit on g2, giving gradients to both Gaussians' positions.

As with single bounce, phase gradients through multi-bounce paths are detached to avoid oscillatory noise. Amplitude gradients flow normally.

---

## 7. Training Pipeline

### 7.1 Optimiser

Use Adam (matching mmIR), with per-parameter-group learning rates:

| Parameter Group | LR | Clip | Notes |
|----------------|-----|------|-------|
| Positions mu_g | 1.6e-4 (decay to 1.6e-6) | 1.0 | Exponential decay schedule |
| Rotations q_g | 1e-3 | 0.5 | Controls surfel orientation + normal |
| Scales s_g | 5e-3 | 1.0 | 2 values per Gaussian (lateral only) |
| Opacity alpha_g | 5e-2 | 1.0 | |
| Materials mat_g | 0.5 | 1.0 | Per-column sub-LR from mmIR |

Material sub-LR scales (inherited from mmIR):
```
eps_real: 0.3, eps_imag: 0.5, sigma_h: 0.3, l_c: 0.3, tau: 0.5, thickness: 0.1
```

### 7.2 Adaptive density control

Periodically (every 100 iterations):

1. **Densification (clone)**: Gaussians with high positional gradient magnitude (> tau_grad) and small scale. Clone = duplicate with slight position offset.
2. **Densification (split)**: Gaussians with high gradient and large scale. Split = replace with two half-size Gaussians.
3. **Pruning**: Remove Gaussians with opacity below threshold (sigmoid(alpha_g) < 0.01) or outside the radar's useful range (< 1.5m or > 30m).
4. **Opacity reset**: Every 500 iterations, reset all opacities to sigmoid^{-1}(0.5) = 0 to allow redistribution.

Radar-specific density control:
- **RA residual-guided densification**: Identify RAE bins with high RA error. For each such bin, if no Gaussian has strong contribution, create new Gaussians at the bin centre with default material. This mirrors mmIR's error-map initialisation.

### 7.3 Training loop

```
for iter in range(max_iterations):
    # 1. Contribution-based culling
    active_mask = cull_by_contribution(gaussians, radar_config, threshold=0.97)
    if iter % 10 == 0:
        active_mask[:] = True   # periodic full inclusion

    # 2. Forward: synthesise ADC (single bounce)
    adc_rendered = synthesize_adc(gaussians[active_mask], tx_pos, rx_pos, radar_config)

    # 3. Optional: add multi-bounce contributions
    if iter > warmup_iters and enable_multibounce:
        adc_mb = synthesize_multibounce(gaussians, active_mask, ...)
        adc_rendered = adc_rendered + adc_mb

    # 4. Process through FFT (identical to ground truth)
    ra_rendered = adc_to_ra(adc_rendered, virtual_array_map)
    ra_gt = adc_to_ra(adc_gt, virtual_array_map)

    # 5. Compute loss
    loss = compute_loss(ra_rendered, ra_gt, adc_rendered, adc_gt)

    # 6. Backward and step
    loss.backward()
    optimizer.step()

    # 7. Adaptive density control
    if iter % densify_interval == 0:
        densify_and_prune(gaussians, grad_accum)

    # 8. Update contribution culling threshold
    if iter % 50 == 0:
        update_culling_stats(gaussians)
```

### 7.4 Multi-frame training

For each scene, 9 cascaded radar frames are available. Training can use:
- **Single-frame**: Optimise Gaussians on one frame (fast, matches mmIR evaluation protocol).
- **Multi-frame**: Cycle through frames, optimising shared Gaussians from different viewpoints (better coverage, requires accurate inter-frame alignment).

Start with single-frame to match mmIR.

---

## 8. Evaluation

### 8.1 Training RA quality (Eval #1)

Directly comparable to mmIR Table 1:
- Render ADC with optimised Gaussians.
- Apply identical FFT pipeline (Hann window, zero-pad, etc.).
- Compute Pearson correlation, PSNR, SSIM, RMSE on RA images.
- Compare against ground truth RA from measured ADC.

### 8.2 Cross-sensor transfer (Eval #2)

Transfer learned Gaussian materials to single-chip radar (IWR1443, 3TX x 4RX):
- Keep Gaussian positions, rotations, scales, and materials fixed.
- Swap antenna configuration (MMWCAS -> IWR1443).
- Re-render ADC with single-chip parameters.
- Apply single-chip FFT pipeline (8 virtual elements, 8-bin azimuth FFT).
- Compute metrics on transferred RA images.

### 8.3 Dense virtual-aperture 3D reconstruction (Eval #3)

Synthesise ADC for a dense 100TX x 100RX array:
- Construct virtual array at lambda/2 spacing.
- Render ADC for all 10K TX-RX pairs (batched).
- Apply 3D FFT (range + azimuth + elevation) to get RAE cube.
- Extract 3D point cloud from RAE magnitude.
- Compare against LiDAR ground truth using Chamfer distance, precision, recall.

---

## 9. Module Architecture

### 9.1 Directory structure

```
mmir/
  gaussian_splatting/                # <-- NEW sub-directory
    __init__.py
    gaussian_model.py                # GaussianModel class (2.5D surfel params)
    initialization.py                # LiDAR -> Gaussian init (PCA, no downsampling)
    rae_grid.py                      # RAE grid construction, bin centres, Jacobian
    splatting.py                     # Cartesian-to-RAE projection, rasterisation
    adc_synthesis.py                 # Per-Gaussian ADC contribution (phase + amplitude)
    culling.py                       # Contribution-based active set selection
    ray_surfel.py                    # Ray-surfel intersection (analytical, differentiable)
    multibounce.py                   # Multi-bounce via ray-surfel tracing
    training.py                      # Training loop, optimiser, density control
    losses.py                        # Loss functions (wraps mmir/losses/ with FFT pipeline)
    config.py                        # GaussianSplattingConfig dataclass
    eval_adapter.py                  # Adapter for mmir/evaluation/ eval scripts
```

### 9.2 Module responsibilities

#### `gaussian_model.py`
- `GaussianModel`: Holds all N Gaussian parameters as contiguous tensors.
  - `positions`: (N, 3) float32
  - `rotations`: (N, 4) float32 (quaternions)
  - `log_scales`: (N, 2) float32 -- **2 axes, not 3** (surfel is flat)
  - `logit_opacities`: (N, 1) float32
  - `raw_materials`: (N, 6) float32
- Derived quantities (computed on-the-fly):
  - `get_normals()`: returns R(q)[:, 2] for each Gaussian
  - `get_tangent_frame()`: returns (t1, t2, n) from R(q)
  - `get_covariance_2d()`: returns diag(exp(s[0])^2, exp(s[1])^2)
  - `get_covariance_3d()`: returns rank-2 3D covariance from tangent frame + scales
- Methods: `reparameterize_materials()`, `clone()`, `split()`, `prune()`.

#### `rae_grid.py`
- `RAEGrid`: Precomputes bin centres, edges, grid dimensions from radar config.
  - `bin_centers_cartesian`: (N_r, N_az, N_el, 3)
  - `range_axis`, `az_angles`, `el_angles`
- Methods: `cartesian_to_rae(points)`, `compute_jacobian(points)`, `get_bin_indices(points)`.

#### `splatting.py`
- `splat_gaussians_to_rae(gaussian_model, rae_grid, radar_pose)`:
  1. Transform Gaussian centres to radar-local frame.
  2. Compute RAE coordinates and Jacobians.
  3. Project covariances to RAE space.
  4. Rasterise to RAE bins (scatter-add or tile-based).
  5. Return per-bin aggregated amplitudes.
- Uses PyTorch.

#### `culling.py`
- `compute_contribution_estimates(gaussian_model, radar_config)`: Fast per-Gaussian amplitude estimate.
- `select_active_set(contributions, threshold=0.97)`: Cumulative-sum threshold selection.
- Returns active mask (boolean tensor).

#### `ray_surfel.py`
- `intersect_ray_surfel(ray_origins, ray_dirs, gaussian_model)`:
  Batch ray-surfel intersection. Returns t_hit, alpha_hit, surfel indices.
- `intersect_ray_surfel_bvh(...)`: Accelerated version with spatial index.
- All operations are differentiable (PyTorch autograd).

#### `adc_synthesis.py`
- `synthesize_adc_single_bounce(gaussian_model, active_mask, tx_pos, rx_pos, radar_config, bsdf)`:
  Vectorised per-Gaussian ADC contribution with exact phase.
  Returns `adc_real, adc_imag` of shape (N_tx, N_rx, K).
- Uses PyTorch.

#### `multibounce.py`
- `synthesize_multibounce(gaussian_model, active_mask, tx_pos, rx_pos, radar_config, bsdf, n_first_bounce, n_secondary_rays)`:
  1. Importance-sample first-bounce Gaussians.
  2. Sample BSDF directions at each first-bounce Gaussian.
  3. Call `ray_surfel.intersect_ray_surfel()` for second-bounce hits.
  4. Accumulate two-bounce ADC contributions.
  Returns additional `adc_real, adc_imag`.

#### `training.py`
- `GaussianSplattingTrainer`: Orchestrates training.
  - Loads data (ADC frames, configs).
  - Manages optimiser with per-group LR.
  - Runs forward pass (cull + synthesise + FFT).
  - Computes loss and backward pass.
  - Applies adaptive density control.
  - Saves checkpoints.

### 9.3 Integration with existing infrastructure

| Existing module | How mm3DGS uses it |
|----------------|-------------------|
| `mmir/renderer/bsdf/` | Import and call `eval_f_cos_physics()` for BSDF at surfel centres |
| `mmir/sensor/element_patterns.py` | Import `evaluate_combined_gain()` for antenna patterns |
| `mmir/data/ra_utils.py` | Import `adc_to_ra_complex()` for the FFT pipeline |
| `mmir/data/config_loader.py` | Import config loading for radar parameters |
| `mmir/losses/multi_chirp_loss.py` | Import loss computation (per-chirp alignment) |
| `mmir/evaluation/` | Use eval adapters for training RA, transfer, and 3D metrics |
| `mmir/preprocessing/alignment/` | Reuse aligned configs (no new alignment needed) |

---

## 10. Implementation Phases

### Phase 1: Core representation + single-bounce ADC (Weeks 1-3)

**Goal**: Render a single RA image from initialised 2.5D Gaussians and compare to ground truth.

1. Implement `GaussianModel` with 2.5D surfel parameters.
2. Implement `initialization.py`: LiDAR point cloud to Gaussians (full cloud, no downsampling).
3. Implement `RAEGrid` from radar config.
4. Implement `splatting.py`: Cartesian-to-RAE projection and rasterisation.
5. Implement `culling.py`: Contribution estimation and active set selection.
6. Implement `adc_synthesis.py`: Per-Gaussian exact phase ADC synthesis.
7. Integrate with existing FFT pipeline (`adc_to_ra_complex()`).
8. Render an RA image from initialised (untrained) Gaussians; visually verify it shows structure.

**Deliverable**: A forward-only rendering pipeline from LiDAR init to RA image.

### Phase 2: Training loop + optimisation (Weeks 3-5)

**Goal**: Train Gaussians on a single scene and match or approach mmIR's training RA metrics.

1. Implement `losses.py` wrapping existing loss functions.
2. Implement `training.py` with Adam optimiser and per-group LR.
3. Add gradient detachment for phase (amplitude-only gradients).
4. Add adaptive density control (densification + pruning).
5. Train on `seq_0_frame_135` as the primary test scene.
6. Evaluate: Pearson correlation, PSNR, SSIM on training RA.

**Deliverable**: Trained Gaussian model with quantitative RA metrics.

### Phase 3: Multi-bounce via ray-surfel tracing (Weeks 5-7)

**Goal**: Add multi-bounce support using direct ray-surfel intersection.

1. Implement `ray_surfel.py`: analytical ray-surfel intersection (differentiable).
2. Implement `multibounce.py`: importance-sampled first bounce, traced second bounce.
3. Add multi-bounce ADC contributions to training loop.
4. Implement RA residual-guided Gaussian densification.
5. Tune hyperparameters across all 9 benchmark scenes.

**Deliverable**: Full multi-bounce pipeline, side-by-side comparison with mmIR.

### Phase 4: Transfer evaluation + dense array (Weeks 7-9)

**Goal**: Validate generalisation and 3D reconstruction quality.

1. Implement `eval_adapter.py` for cross-sensor transfer (cascaded -> single-chip).
2. Implement dense virtual-aperture rendering (100x100 array, batched).
3. Run all 9 scenes through training RA, transfer, and 3D occupancy evaluations.
4. Tabulate and compare against mmIR results.

**Deliverable**: Complete evaluation results, ready for paper.

---

## 11. Design Decisions (Resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Phase computation | Per-Gaussian exact phase | Bin-centre approx introduces ~47 rad error at 77 GHz |
| Representation | 2.5DGS (2D surfel + ITU thickness) | Natural surface primitive; intrinsic normal; direct ray intersection |
| Multi-bounce | Ray-surfel intersection (no mesh) | Analytical, differentiable, no mesh construction overhead |
| Compute framework | Hybrid PyTorch + DrJit | PyTorch for splatting/ADC, DrJit only if needed for BVH |
| Elevation | Full 3D RAE grid, RA-slice loss | Enables 3D recon; loss on elevation-0 matches mmIR evaluation |
| Normal consistency | Derived from rotation quaternion | Intrinsic to 2.5DGS; no separate parameter needed |
| Gaussian count | Full LiDAR cloud, contribution-culled | No information loss; 97% cumulative threshold; ~57 MB memory |

---

## 12. Summary

mm3DGS replaces mmIR's mesh + ray tracing with **2.5D Gaussian surfels** + **RAE splatting** while preserving the physics-based ITU material model, MIMO-coherent ADC synthesis, and FFT processing pipeline. The key technical contributions are:

1. **2.5DGS representation**: Flat surfels with ITU slab thickness -- geometrically 2D for clean normals and ray intersection, electromagnetically 2.5D for multi-layer Fresnel.
2. **RAE splatting**: Projecting 2D Gaussian surfels to a radar-native coordinate grid via Jacobian-based covariance projection.
3. **Per-Gaussian ADC synthesis with exact phase**: Each Gaussian contributes to the ADC with its exact distance-based phase, preserving sub-bin interference.
4. **Mesh-free multi-bounce**: Analytical ray-surfel intersection against 2.5D Gaussians enables multi-bounce path tracing without mesh construction.
5. **Contribution-based culling**: Dynamic active-set selection preserves the full representation while controlling compute cost.
6. **ADC-level supervision**: Synthesising raw ADC before FFT enables modelling of windowing, sidelobes, and spectral leakage -- something existing radar 3DGS methods cannot do.

The implementation reuses mmIR's BSDF model, antenna patterns, loss functions, and evaluation suite, making direct comparison straightforward.
