# mmWave 3D Gaussian Splatting: Implementation Plan

## 1. Motivation & Context

### Current system (mmIR): mesh-based differentiable inverse renderer
- LiDAR point cloud frames are aggregated and Poisson-reconstructed into a triangle mesh (~200K triangles).
- Rays are traced from RX elements through the mesh; each hit position is evaluated by all 12TX x 16RX pairs for MIMO phase coherence.
- Budget limit of ~1500 reservoir hits/RX means the mesh cannot be fully covered without unrealistic ray counts.
- Training takes 10-15 min/scene at 500 iterations; a single dense-array render takes ~6s.

### Proposed system (mm3DGS): Gaussian splatting for mmWave
Replace the mesh representation with 3D Gaussians and replace the RX-centric reservoir ray tracing with splatting to a Range-Azimuth-Elevation (RAE) grid. ADC is synthesized directly, preserving compatibility with FFT processing.

### Advantages over mesh-based mmIR
1. **Full scene coverage.** Every Gaussian contributes every iteration (no sampling budget).
2. **Simpler single-bounce path.** Splatting gives per-bin aggregate reflectivity; ADC synthesis only needs TX/RX-to-bin-center distances.
3. **ADC-level synthesis.** Rendering to ADC rather than directly to RA means the same FFT pipeline applies to both rendered and ground-truth data, enabling modelling of windowing, sidelobes, and cross-bin leakage.
4. **Multi-bounce via ray tracing from bin centers.** The RAE grid provides a natural set of sparse virtual reflectors for secondary-bounce ray tracing.

### Advantages over existing radar 3DGS methods (e.g. RadarGS, mmGaussian)
Existing methods splat Gaussians directly to a 2D RA image, supervise with magnitude loss only, and cannot model FFT artefacts or multi-bounce. By splatting to a 3D RAE volume and synthesising raw ADC, mm3DGS:
- Preserves phase coherence across the virtual array.
- Allows identical FFT processing on rendered and measured data.
- Enables multi-bounce modelling (Section 6).

---

## 2. 3D Gaussian Representation

### 2.1 Per-Gaussian parameters

Each Gaussian g is defined by:

| Symbol | Description | Shape | Init |
|--------|-------------|-------|------|
| mu_g | Center position (world frame) | (3,) | LiDAR point |
| q_g | Rotation quaternion | (4,) | Estimated from local PCA normal |
| s_g | Log-scale (3 axes) | (3,) | From local point density |
| alpha_g | Logit-opacity | (1,) | 0 (sigmoid -> 0.5) |
| n_g | Surface normal | (3,) | LiDAR normal estimate |
| mat_g | Material parameters (raw-space) | (6,) | ITU concrete init (matching mmIR) |

Material parameters reuse the existing mmIR ITU model (6 DOF):
```
eps_real, eps_imag, sigma_h, l_c, tau, thickness
```
with the same reparameterization (exp/sigmoid) and bounds as `train.py:ParameterManagerSionna`.

### 2.2 Covariance construction

The 3D covariance matrix from rotation R(q_g) and scale S(s_g):
```
Sigma_g = R @ diag(exp(s_g))^2 @ R^T
```
This is standard 3DGS parameterization, ensuring Sigma is always symmetric positive-definite.

### 2.3 Initialization from LiDAR

1. Load the aggregated LiDAR point cloud (`data/seq_X_frame_Y/scene/pcl.npy`, columns x,y,z,nx,ny,nz,intensity).
2. Voxel-downsample to ~50K-100K points (target density ~1 point per 3-5cm).
3. Set `mu_g` = point position; `n_g` = point normal.
4. Estimate local PCA from k-nearest neighbors (k=20) to initialize rotation `q_g` (align the third eigenvector with the normal) and scale `s_g` (set the three eigenvalues as initial scales, clamped to [1cm, 50cm]).
5. Initialize material `mat_g` from the same ITU concrete default as mmIR.
6. Optional: initialize additional Gaussians at positions where mmIR's error map shows under-fitting (high residual regions in the RA image).

---

## 3. Splatting to Range-Azimuth-Elevation Grid

### 3.1 RAE grid definition

The RAE grid is fully determined by the radar configuration (reuse logic from `eval_3d_occupancy.py` / `single_view_viz.py:build_fov_wireframe()`):

| Dimension | # Bins | Spacing |
|-----------|--------|---------|
| Range | N_r = 256 | Delta_R = c / (2B) ~ 59mm |
| Azimuth | N_az = 127 (128 FFT bins, DC removed) | arcsin-spaced, ~0.016 rad near boresight |
| Elevation | N_el = 127 (128 FFT bins, DC removed) | arcsin-spaced, ~0.016 rad near boresight |

Bin centers (r_k, theta_az_m, theta_el_n) are computed exactly as in `make_angle_grids_np()` and stored as a precomputed lookup table.

Cartesian bin-center positions:
```
x_b = r_k * cos(theta_el_n) * sin(theta_az_m)
y_b = r_k * cos(theta_el_n) * cos(theta_az_m)
z_b = r_k * sin(theta_el_n)
```

### 3.2 Cartesian-to-RAE Jacobian

Given Gaussian center mu_g = (x, y, z) in the radar's local frame:
```
r   = sqrt(x^2 + y^2 + z^2)
az  = atan2(x, y)
el  = asin(z / r)
rho = sqrt(x^2 + y^2)
```

Jacobian J of the (r, az, el) <- (x, y, z) mapping:
```
J = [ x/r        y/r       z/r       ]   # d(r)/d(x,y,z)
    [ y/rho^2   -x/rho^2   0         ]   # d(az)/d(x,y,z)
    [-xz/(r^2*rho) -yz/(r^2*rho) rho/r^2]   # d(el)/d(x,y,z)
```

Projected covariance in RAE space:
```
Sigma_RAE = J @ Sigma_g @ J^T
```

This 3x3 covariance tells us the Gaussian's footprint in (range, azimuth, elevation) coordinates. We rasterize this 3D Gaussian onto the RAE grid.

### 3.3 Rasterization

For each Gaussian g:
1. Compute its RAE center (r_g, az_g, el_g) and projected covariance Sigma_RAE.
2. Find the bounding box in RAE bin indices (e.g. +/-3 sigma in each dimension).
3. For each bin (k, m, n) within the bounding box, evaluate the unnormalized Gaussian weight:
   ```
   delta = [r_k - r_g, az_m - az_g, el_n - el_g]
   w_g(k,m,n) = exp(-0.5 * delta^T @ Sigma_RAE^{-1} @ delta)
   ```
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

The rasterization can be implemented as a CUDA kernel following the tile-based approach of standard 3DGS, but extended to 3D:

1. **Bin assignment**: Parallel over Gaussians. Each Gaussian computes its bounding box in RAE indices, writes (Gaussian_id, bin_key) pairs.
2. **Sort**: Sort pairs by bin_key (radix sort on GPU).
3. **Accumulate**: Parallel over bins. Each bin iterates over its sorted Gaussians, accumulates amplitude via alpha compositing.

For the initial implementation, a simpler scatter-add approach (no sorting, just atomic adds) is sufficient and much easier to implement in PyTorch/DrJit.

---

## 4. ADC Synthesis (Single Bounce)

### 4.1 Phase computation

This is the central technical challenge. Two strategies are proposed; both are worth implementing.

#### Strategy A: Per-Gaussian exact phase (recommended)

Each Gaussian contributes to ADC with its own phase computed from its exact center position mu_g. The splatting weight determines the amplitude, but the phase comes from the true Gaussian position.

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

**Complexity**: O(N_gaussians * N_tx * N_rx * K). For N_g=50K, N_tx=12, N_rx=16, K=256: ~2.5 x 10^9 multiply-adds. Feasible on RTX 4090 (~80 TFLOPS) in <1 second via vectorized scatter-add. This is actually similar to the current renderer's phasor scatter-add but with Gaussians instead of ray hits.

**Key advantage**: Exact phase preserves sub-bin interference patterns, enabling the FFT to correctly resolve scatterers.

**Key disadvantage**: N_gaussians >> N_hits (50K vs 1.5K), so the MIMO expansion is ~33x more expensive per iteration. But eliminating the expensive reservoir sampling + ray tracing (which takes the bulk of mmIR's render time) may compensate.

#### Strategy B: Bin-center approximate phase (faster, less accurate)

Aggregate all Gaussian contributions at each bin center first, then compute ADC from occupied bin centers only.

For bin (k, m, n) with aggregate amplitude A_bin:
```
p_bin = bin_center_cartesian(k, m, n)
d_tx_bin = ||p_bin - p_tx_i||
d_rx_bin = ||p_bin - p_rx_j||
R_tot_bin = d_tx_bin + d_rx_bin
tau_bin = R_tot_bin / c

phi_bin(t_k) = 2*pi * (f_0 * tau_bin + S * tau_bin * t_k)

ADC_ij[k] += A_bin * cos(phi_bin(t_k))
ADC_ij[k] += A_bin * sin(phi_bin(t_k))
```

**Complexity**: O(N_occupied_bins * N_tx * N_rx * K). With N_bins ~ 5K-20K occupied, this is 1-4x cheaper than Strategy A.

**Key advantage**: Bin-center distances can be precomputed (they don't depend on Gaussian parameters), so only A_bin needs recomputation during training.

**Key disadvantage**: Phase error up to 2*pi * Delta_R / (2 * lambda) ~ 47 radians per range bin. This is significant at 77 GHz (lambda ~ 4mm). The azimuth/elevation binning introduces smaller but non-negligible phase errors.

**Mitigation**: Add a learnable per-bin phase offset delta_phi_bin that absorbs the residual. This doubles the per-bin parameter count but is cheap.

#### Recommendation

**Start with Strategy A** (per-Gaussian exact phase). It is more physically correct and the computational cost is manageable on an RTX 4090. Strategy B can serve as a faster approximation for prototyping or for very large scenes.

If Strategy A proves too slow, a hybrid is possible: use Strategy A for the N_top Gaussians with highest amplitude contribution and Strategy B for the rest.

### 4.2 BSDF evaluation

Reuse the existing `BSDFmmWaveScalar` or `BSDFmmWaveJones` from `mmir/renderer/bsdf/`. For each Gaussian g, evaluate:

```
wo = normalize(p_rx_j - mu_g)       # outgoing direction (toward RX)
wi = normalize(p_tx_i - mu_g)       # incoming direction (from TX)
n  = n_g                             # Gaussian's surface normal

f_cos = BSDF.eval_f_cos(wo, wi, n, mat_g)
```

The BSDF evaluation is per-Gaussian-per-TX-RX, reusing the KA+SPM hybrid model with ITU slab Fresnel. This is identical to mmIR's BSDF evaluation but at Gaussian centers instead of ray hit positions.

### 4.3 Antenna gain

Evaluate TX and RX antenna patterns at each Gaussian center:
```
G_tx = pattern_tx(normalize(mu_g - p_tx_i), boresight_tx_i)
G_rx = pattern_rx(normalize(p_rx_j - mu_g), boresight_rx_j)
```

Reuse `evaluate_combined_gain()` from `mmir/sensor/element_patterns.py`.

### 4.4 Path loss

Apply inverse-square law from the radar equation:
```
path_loss_g = 1 / (d_tx_g^2 * d_rx_g^2)      # if using RCS model
# OR
path_loss_g = 1 / (d_tx_g * d_rx_g)           # if using E-field amplitude model
```

Match the convention from mmIR's `synthesize_end_to_end()` (currently uses RX-centric MC normalization with sqrt(radar_constant) / d^2 terms).

### 4.5 Complete single-bounce ADC synthesis

Putting it all together (Strategy A):

```python
for g in active_gaussians:
    for i in range(N_tx):
        for j in range(N_rx):
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
            for k in range(K):
                phi_k = phi_const + phi_slope * t[k]
                adc_real[i, j, k] += A * cos(phi_k)
                adc_imag[i, j, k] += A * sin(phi_k)
```

In practice, all loops are vectorized as parallel GPU operations (DrJit scatter-add or PyTorch `scatter_add_`).

### 4.6 Gradient flow and differentiability

The full forward pass is differentiable with respect to:
- **Gaussian positions** mu_g: Through distances d_tx, d_rx (affects phase AND amplitude).
- **Gaussian rotations/scales**: Through the splatting weights (affects which bins receive energy).
- **Gaussian normals** n_g: Through BSDF evaluation.
- **Material parameters** mat_g: Through BSDF evaluation.
- **Opacity** alpha_g: Through the effective amplitude.

Phase gradient challenge (same as mmIR): At lambda ~ 4mm, a 1mm shift in mu_g causes a ~1.6 radian phase change. This creates extremely noisy gradients. **Solution: detach phase from the AD graph** (same approach as mmIR's `enable_grad_phase=False`), flowing gradients only through amplitude. The phase is correct in the forward pass but treated as a constant during backprop.

---

## 5. FFT Processing & Loss

### 5.1 ADC-to-RA pipeline

The synthesized ADC `adc[N_tx, N_rx, K, 2]` is processed identically to ground truth:

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

Additional losses to consider:
- **ADC magnitude loss**: L_ADC_mag = MSE(|ADC_render|, |ADC_gt|) -- supervises before FFT.
- **Phase loss** (optional, weighted by SNR): L_phase = weighted_MSE(angle(RA_render), angle(RA_gt)).
- **Material regularization**: Laplacian smoothness on material parameters across neighboring Gaussians.
- **Gaussian regularization**: Penalize Gaussians with extreme scales or very low opacity.

Reuse the multi-chirp loss from `mmir/losses/multi_chirp_loss.py` for per-chirp alignment if using multiple chirps.

### 5.3 Advantage: FFT effects are modelled correctly

Because we synthesize ADC before applying FFT:
- **Windowing effects** (Hann sidelobes) apply identically to rendered and measured data.
- **Spectral leakage** from scatterers not exactly at bin centers is captured.
- **Zero-padding artifacts** in the azimuth FFT are matched.

Methods that render directly to RA (existing radar 3DGS) cannot capture these effects.

---

## 6. Multi-Bounce Extension

### 6.1 The multi-bounce problem in Gaussian splatting

Multi-bounce radar returns arise when a transmitted wave reflects off surface A, then surface B, then returns to the receiver. The path length is d(TX, A) + d(A, B) + d(B, RX), producing a return at a range larger than any single object's range.

In the current mesh-based renderer, multi-bounce is handled by recursive ray tracing (up to 3 bounces) with next-event estimation at each bounce. With Gaussians, we need a different approach.

### 6.2 Proposed approach: ray tracing from Gaussian centers

After single-bounce splatting produces the first-bounce contributions, we handle multi-bounce as follows:

**Second bounce:**
1. For each first-bounce Gaussian g1 (or a subset, importance-sampled by first-bounce amplitude):
   - Sample N_secondary outgoing directions from BSDF(mat_g1, wo, n_g1).
   - For each direction, find the nearest Gaussian g2 along that direction.
   - The "nearest Gaussian" query can be implemented as:
     a. **KD-tree lookup**: Find Gaussians whose centers are close to the ray. Fast but approximate.
     b. **Ray-Gaussian intersection**: Compute the ray's closest approach to each Gaussian's center; if it falls within the Gaussian's effective radius (e.g. 2*sigma_max), count it as a hit.
     c. **Proxy mesh ray tracing**: Build a simplified mesh from Gaussian centers and normals (one oriented disc per Gaussian); use Mitsuba's ray-triangle intersection. Most robust and reuses existing infrastructure.
   - For each g1-g2 pair:
     ```
     R_tot = d(TX_i, mu_g1) + d(mu_g1, mu_g2) + d(mu_g2, RX_j)
     A_2bounce = A_g1 * f_BSDF(g1, bounce_dir) * A_g2 / d(g1, g2)^2
     ```
   - Accumulate into ADC with the two-bounce phase.

**Higher bounces**: Apply recursively with Russian roulette termination (same as mmIR).

### 6.3 Proxy disc mesh for secondary tracing

Build a lightweight triangle mesh from Gaussians for secondary ray tracing:
1. For each Gaussian g, create a small disc (2 triangles) centered at mu_g with normal n_g and radius proportional to the largest scale eigenvalue.
2. Register this mesh with Mitsuba 3 (`mi.load_dict(...)`) for hardware-accelerated ray intersection.
3. Update the disc mesh when Gaussian positions/normals change (every N iterations or when densifying).

This allows reusing mmIR's battle-tested multi-bounce synthesis code (`synthesize_end_to_end_multibounce()`) with minimal modifications.

### 6.4 Stochastic Gaussian sampling for multi-bounce

For computational efficiency, not all Gaussians need to participate in multi-bounce:
1. Compute single-bounce amplitudes for all Gaussians.
2. Build a CDF over Gaussians weighted by their single-bounce amplitude.
3. Sample N_multi Gaussians (e.g. 2000-5000) as first-bounce origins.
4. For each sampled first-bounce Gaussian, trace N_secondary rays (e.g. 4-8) via the proxy disc mesh.
5. Accumulate multi-bounce ADC contributions with proper MC normalization (1/pdf).

This makes multi-bounce cost independent of the total Gaussian count.

### 6.5 Alternative: Gaussian-to-Gaussian volumetric scattering

A more "native" approach treats the Gaussians as a scattering volume:
1. For each first-bounce Gaussian g1, march rays through the Gaussian field.
2. At each step, compute the "local scattering density" from nearby Gaussians (weighted by their opacity and distance).
3. When cumulative density exceeds a threshold, record a second-bounce hit.

This avoids the proxy mesh but is more complex to implement and differentiate. Recommend as a future extension after the proxy mesh approach is validated.

---

## 7. Training Pipeline

### 7.1 Optimizer

Use Adam (matching mmIR), with per-parameter-group learning rates:

| Parameter Group | LR | Clip |
|----------------|-----|------|
| Positions mu_g | 1.6e-4 (decay to 1.6e-6) | 1.0 |
| Rotations q_g | 1e-3 | 0.5 |
| Scales s_g | 5e-3 | 1.0 |
| Opacity alpha_g | 5e-2 | 1.0 |
| Normals n_g | 1e-2 | 0.5 |
| Materials mat_g | 0.5 | 1.0 |

These are adapted from standard 3DGS (positions, rotations, scales, opacity) and mmIR (normals, materials). Tuning will be needed.

### 7.2 Adaptive density control

Following standard 3DGS practice, periodically (every 100 iterations):

1. **Densification**: Clone Gaussians with high positional gradient magnitude (> tau_grad) and small scale. Split Gaussians with high gradient and large scale into two smaller Gaussians.
2. **Pruning**: Remove Gaussians with opacity below threshold (sigmoid(alpha_g) < 0.01) or very far outside the radar's FOV.
3. **Opacity reset**: Every 500 iterations, reset all opacities to a moderate value to allow redistribution.

Additionally, radar-specific density control:
- **Range-gated pruning**: Remove Gaussians at ranges < 1.5m (TX-RX coupling zone, bins 0-14) or > 30m (beyond useful range).
- **RA residual-guided densification**: Densify in regions where the RA error map is high (similar to mmIR's error-map initialization).

### 7.3 Training loop

```
for iter in range(max_iterations):
    # 1. Forward: splat Gaussians to RAE grid
    rae_amplitudes = splat_to_rae(gaussians, radar_config)

    # 2. Forward: synthesize ADC (single bounce)
    adc_rendered = synthesize_adc(gaussians, tx_positions, rx_positions, radar_config)

    # 3. Optional: add multi-bounce contributions
    if iter > warmup_iterations and enable_multibounce:
        adc_multibounce = synthesize_multibounce(gaussians, proxy_mesh, ...)
        adc_rendered = adc_rendered + adc_multibounce

    # 4. Process through FFT (identical to ground truth)
    ra_rendered = adc_to_ra(adc_rendered, virtual_array_map)
    ra_gt = adc_to_ra(adc_gt, virtual_array_map)

    # 5. Compute loss
    loss = compute_loss(ra_rendered, ra_gt, adc_rendered, adc_gt)

    # 6. Backward and step
    loss.backward()
    optimizer.step()

    # 7. Adaptive density control
    if iter % 100 == 0:
        densify_and_prune(gaussians, grad_accum)
```

### 7.4 Multi-frame training

For each scene, 9 cascaded radar frames are available. Training can use:
- **Single-frame**: Optimize Gaussians on one frame (fast, matches mmIR).
- **Multi-frame**: Cycle through frames, optimizing shared Gaussians from different viewpoints (better coverage, requires accurate inter-frame alignment).

Start with single-frame to match mmIR's evaluation protocol.

---

## 8. Evaluation

### 8.1 Training RA quality (Eval #1)

Directly comparable to mmIR Table 1:
- Render ADC with optimized Gaussians.
- Apply identical FFT pipeline (Hann window, zero-pad, etc.).
- Compute Pearson correlation, PSNR, SSIM, RMSE on RA images.
- Compare against ground truth RA from measured ADC.

### 8.2 Cross-sensor transfer (Eval #2)

Transfer learned Gaussian materials to single-chip radar (IWR1443, 3TX x 4RX):
- Keep Gaussian positions, normals, and materials fixed.
- Swap antenna configuration (MMWCAS -> IWR1443).
- Re-render ADC with single-chip parameters.
- Apply single-chip FFT pipeline (8 virtual elements, 8-bin azimuth FFT).
- Compute metrics on transferred RA images.

### 8.3 Dense virtual-aperture 3D reconstruction (Eval #3)

Synthesize ADC for a dense 100TX x 100RX array:
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
  gaussian_splatting/           # <-- NEW sub-directory
    __init__.py
    gaussian_model.py           # GaussianModel class: stores all per-Gaussian params
    initialization.py           # LiDAR -> Gaussian init (voxel downsample, PCA)
    rae_grid.py                 # RAE grid construction, bin centers, Jacobian
    splatting.py                # Cartesian-to-RAE projection, rasterization
    adc_synthesis.py            # Per-Gaussian ADC contribution (phase + amplitude)
    multibounce.py              # Proxy mesh construction, secondary bounce tracing
    training.py                 # Training loop, optimizer, density control
    losses.py                   # Loss functions (wraps mmir/losses/ with FFT pipeline)
    config.py                   # GaussianSplattingConfig dataclass
    eval_adapter.py             # Adapter for mmir/evaluation/ eval scripts
```

### 9.2 Module responsibilities

#### `gaussian_model.py`
- `GaussianModel`: Holds all N Gaussian parameters as contiguous tensors.
  - `positions`: (N, 3) float32
  - `rotations`: (N, 4) float32 (quaternions)
  - `log_scales`: (N, 3) float32
  - `logit_opacities`: (N, 1) float32
  - `normals`: (N, 3) float32
  - `raw_materials`: (N, 6) float32
- Methods: `get_covariance_3d()`, `get_opacity()`, `reparameterize_materials()`, `clone()`, `split()`, `prune()`.

#### `rae_grid.py`
- `RAEGrid`: Precomputes bin centers, bin edges, grid dimensions from radar config.
  - `bin_centers_cartesian`: (N_r, N_az, N_el, 3) -- precomputed world-frame positions.
  - `range_axis`, `az_angles`, `el_angles` -- 1D arrays.
- Methods: `cartesian_to_rae(points)`, `compute_jacobian(points)`, `get_bin_indices(points)`.

#### `splatting.py`
- `splat_gaussians_to_rae(gaussian_model, rae_grid, radar_pose)`:
  1. Transform Gaussian centers to radar-local frame.
  2. Compute RAE coordinates and Jacobians.
  3. Project covariances to RAE space.
  4. Rasterize to RAE bins (scatter-add).
  5. Return per-bin aggregated amplitudes and material properties.
- Uses PyTorch for GPU computation (avoid Mitsuba/DrJit dependency for splatting).

#### `adc_synthesis.py`
- `synthesize_adc_single_bounce(gaussian_model, tx_pos, rx_pos, radar_config, bsdf)`:
  1. For each Gaussian: compute distances to all TX/RX.
  2. Evaluate BSDF at each Gaussian for each TX-RX pair.
  3. Compute antenna gains.
  4. Compute phase and scatter-add to ADC array.
  5. Return `adc_real, adc_imag` of shape (N_tx, N_rx, K).
- Can be implemented in PyTorch (for prototyping) or DrJit (for performance/AD compatibility).

#### `multibounce.py`
- `build_proxy_mesh(gaussian_model)`: Create oriented disc mesh from Gaussians.
- `trace_secondary_bounces(proxy_mesh, first_bounce_gaussians, bsdf, radar_config)`: Ray trace from first-bounce Gaussian centers through proxy mesh.
- Returns additional ADC contributions from multi-bounce paths.

#### `training.py`
- `GaussianSplattingTrainer`: Orchestrates training.
  - Loads data (ADC frames, configs).
  - Manages optimizer with per-group LR.
  - Runs forward pass (splat + synthesize + FFT).
  - Computes loss and backward pass.
  - Applies adaptive density control.
  - Saves checkpoints.

### 9.3 Integration with existing infrastructure

| Existing module | How mm3DGS uses it |
|----------------|-------------------|
| `mmir/renderer/bsdf/` | Import and call `eval_f_cos_physics()` for BSDF evaluation at Gaussian centers |
| `mmir/sensor/element_patterns.py` | Import `evaluate_combined_gain()` for antenna patterns |
| `mmir/data/ra_utils.py` | Import `adc_to_ra_complex()` for the FFT pipeline |
| `mmir/data/config_loader.py` | Import config loading for radar parameters |
| `mmir/losses/multi_chirp_loss.py` | Import loss computation (per-chirp alignment) |
| `mmir/evaluation/` | Use eval adapters for training RA, transfer, and 3D metrics |
| `mmir/preprocessing/alignment/` | Reuse aligned configs (no new alignment needed) |

---

## 10. Implementation Phases

### Phase 1: Core splatting + single-bounce ADC (Weeks 1-3)

**Goal**: Render a single RA image from initialized Gaussians and compare to ground truth.

1. Implement `GaussianModel` with all parameter storage.
2. Implement `initialization.py`: LiDAR point cloud to Gaussians.
3. Implement `RAEGrid` from radar config.
4. Implement `splatting.py`: Cartesian-to-RAE projection and rasterization.
5. Implement `adc_synthesis.py`: Strategy A (per-Gaussian exact phase).
6. Integrate with existing FFT pipeline (`adc_to_ra_complex()`).
7. Render an RA image from initialized (untrained) Gaussians; visually verify it shows structure.

**Deliverable**: A forward-only rendering pipeline from LiDAR init to RA image.

### Phase 2: Training loop + optimization (Weeks 3-5)

**Goal**: Train Gaussians on a single scene and match or approach mmIR's training RA metrics.

1. Implement `losses.py` wrapping existing loss functions.
2. Implement `training.py` with Adam optimizer and per-group LR.
3. Add gradient detachment for phase (amplitude-only gradients).
4. Add adaptive density control (densification + pruning).
5. Train on `seq_0_frame_135` as the primary test scene.
6. Evaluate: Pearson correlation, PSNR, SSIM on training RA.

**Deliverable**: Trained Gaussian model with quantitative RA metrics.

### Phase 3: Multi-bounce + refinement (Weeks 5-7)

**Goal**: Add multi-bounce support and refine the pipeline.

1. Implement `multibounce.py`: proxy disc mesh + secondary bounce tracing.
2. Add multi-bounce ADC contributions to training loop.
3. Implement Gaussian-specific density control (RA residual-guided).
4. Tune hyperparameters across all 9 benchmark scenes.
5. Compare against mmIR on all training RA metrics.

**Deliverable**: Full multi-bounce pipeline, side-by-side comparison with mmIR.

### Phase 4: Transfer evaluation + dense array (Weeks 7-9)

**Goal**: Validate generalization and 3D reconstruction quality.

1. Implement `eval_adapter.py` for cross-sensor transfer (cascaded -> single-chip).
2. Implement dense virtual-aperture rendering (100x100 array, batched).
3. Run all 9 scenes through training RA, transfer, and 3D occupancy evaluations.
4. Tabulate and compare against mmIR results.

**Deliverable**: Complete evaluation results, ready for paper.

---

## 11. Open Questions & Design Decisions

### 11.1 PyTorch vs DrJit for splatting/ADC synthesis

**Option A: Full PyTorch.** Easier autograd, wider ecosystem, simpler debugging. But loses access to Mitsuba's ray-triangle intersection for multi-bounce.

**Option B: Full DrJit.** Consistent with mmIR, native Mitsuba integration. But DrJit's scatter-add and custom kernels are less ergonomic than PyTorch.

**Option C: Hybrid.** Use PyTorch for splatting/ADC synthesis (forward/backward are straightforward tensor ops) and DrJit/Mitsuba only for multi-bounce ray tracing (where hardware-accelerated BVH is needed).

**Recommendation: Option C.** The splatting and ADC synthesis are essentially large matrix operations that PyTorch handles well. Multi-bounce requires ray tracing that Mitsuba handles well. The DrJit-to-PyTorch bridge already exists in mmIR (`mmir/losses/` uses PyTorch loss on DrJit outputs).

### 11.2 Handling elevation

The cascaded radar (MMWCAS) has a 7-row elevation aperture but most virtual elements are in the azimuth plane. The current mmIR evaluation uses only the azimuth row (elevation index 0) for RA images.

**For mm3DGS**: Still splat to the full 3D RAE grid (enables 3D reconstruction), but the training loss operates on the RA image extracted from the elevation-0 slice. Elevation information is implicitly constrained by the Gaussian positions (which come from LiDAR, providing vertical structure).

### 11.3 Splatting footprint size

If the projected Gaussian in RAE space is much larger than one bin (e.g. a large flat wall), it contributes to many bins with nearly uniform weight. This is correct physically (a large wall reflects to a spread of range/angle bins). But it means the splatting weight doesn't strongly localize the contribution.

The effective localization comes from the PHASE: even though the wall contributes to many bins, each bin has a unique phase that the FFT correctly resolves. This is another reason to prefer Strategy A (per-Gaussian exact phase).

### 11.4 Gaussian normal consistency

In standard 3DGS, Gaussians don't have explicit normals (they're implicit from the flattest axis of the covariance ellipsoid). For mmWave, explicit normals are needed for BSDF evaluation.

**Options:**
1. Always use the LiDAR-initialized normal, with a learnable rotation applied.
2. Derive the normal from the covariance (eigenvector of smallest eigenvalue).
3. Store an explicit normal and regularize it to be consistent with the covariance orientation.

**Recommendation: Option 1** (explicit normal with learnable correction). This matches mmIR's learnable normals and gives the optimizer direct control over the BSDF-critical quantity.

### 11.5 Number of Gaussians

**Target**: 30K-100K Gaussians per scene (comparable to the ~100K vertices / ~200K triangles in mmIR's meshes). Starting from 50K initialized from LiDAR, with adaptive densification expected to grow to ~80K-120K.

The radar's angular resolution at 77 GHz is coarse (~2deg azimuth, ~15deg elevation for cascaded), so fewer Gaussians than optical 3DGS (which typically uses 100K-1M+) should suffice.

---

## 12. Summary

mm3DGS replaces mmIR's mesh + ray tracing with 3D Gaussians + RAE splatting while preserving the physics-based material model, MIMO-coherent ADC synthesis, and FFT processing pipeline. The key technical contributions are:

1. **RAE splatting**: Projecting 3D Gaussians to a radar-native coordinate grid via Jacobian-based covariance projection.
2. **Per-Gaussian ADC synthesis with exact phase**: Each Gaussian contributes to the ADC with its exact distance-based phase, preserving sub-bin interference.
3. **Multi-bounce via proxy mesh**: Oriented discs from Gaussian positions enable secondary-bounce ray tracing using existing infrastructure.
4. **ADC-level supervision**: Synthesizing raw ADC before FFT enables modelling of windowing, sidelobes, and spectral leakage.

The implementation reuses mmIR's BSDF model, antenna patterns, loss functions, and evaluation suite, making direct comparison straightforward.
