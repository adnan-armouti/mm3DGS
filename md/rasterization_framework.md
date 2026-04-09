# mmWave Gaussian Rasterization: Full Framework

## The Core Pipeline

Every component is a pure tensor operation in one framework (PyTorch).
No DrJit, no Mitsuba, no external physics engine.

```
┌─────────────┐     ┌───────────┐     ┌──────────┐     ┌─────────────┐
│  Gaussians  │────▶│  PROJECT  │────▶│  SHADE   │────▶│  SCATTER    │
│  (scene)    │     │ (geometry)│     │ (PyTorch)│     │  (to ADC)   │
└─────────────┘     └───────────┘     └──────────┘     └─────────────┘
                         │                 │                  │
                    distances d       amplitude A       phasor sum
                    angles θ         (per g,tx,rx)      into ADC[tx,rx,k]
```

**Single-bounce**: O(N × N_tx × N_rx) — one evaluation per Gaussian
per TX-RX pair. No spatial acceleration structure needed.

**Multi-bounce**: O(P × N_tx × N_rx) — where P is the number of
interacting Gaussian pairs. Requires a spatial index (voxel hash or
`torch_cluster.radius`) to find pairs — this replaces the BVH used
in ray-tracing renderers with a simpler, non-directional spatial query.

**Initialization**: From LiDAR point cloud via FOV filtering + FPS
(existing `initialization.py`). Radar-only initialization (from RA
backprojection) is a future extension.

---

## 1. Shading Model: Pluggable, PyTorch-Native

The rasterization framework separates the PIPELINE (project → shade →
scatter) from the SHADING MODEL (how amplitude is computed). The
shading model is a pluggable component.

### Amplitude convention: field amplitude (not power)

mmIR's `synthesize_end_to_end()` computes the phasor weight as:

    weight = radar_scale × √(f_cos × G_antenna × path_loss)

The `sqrt` converts from the power domain (BSDF returns power in 1/sr)
to the field amplitude domain (what ADC samples represent). This is the
radar equation in field form: E ∝ √(P_t × G × σ / R⁴).

All shading tiers below follow this convention: the BSDF returns POWER,
and the amplitude chain takes the square root.

### Tier 1: ITU Fresnel in PyTorch (default)

Re-implements the core of mmIR's ITU material model natively in PyTorch.
Same 6 parameters, same physical interpretation, same cross-sensor transfer.

```python
def fresnel_power_reflectance(cos_theta_i, eps_real, eps_imag):
    """Fresnel power reflectance R = 0.5(|r_s|² + |r_p|²).
    Returns POWER (to be sqrt'd in the amplitude chain)."""
    n = torch.sqrt(torch.complex(eps_real, -eps_imag))
    cos_t = torch.sqrt(1 - (1 - cos_theta_i**2) / (eps_real - 1j*eps_imag))
    cos_i = cos_theta_i.to(torch.complex64)
    r_s = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    r_p = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    return 0.5 * (torch.abs(r_s)**2 + torch.abs(r_p)**2)   # POWER
```

The full single-bounce amplitude chain:

```python
# BSDF (power domain, uses 4 of 6 ITU params — l_c, tau reserved for Tier 2)
R_power = fresnel_power_reflectance(cos_theta, eps_real, eps_imag)
eta = torch.exp(-(2 * kappa * sigma_h * cos_theta)**2)       # Rayleigh factor
f_cos_power = R_power * eta * cos_theta                       # power × cosθ

# Antenna gains (linear power)
G_tx = antenna_pattern(dir_tx, boresight_tx)                  # power gain
G_rx = antenna_pattern(dir_rx, boresight_rx)

# Coherence factor γ (scalar per Gaussian, see derivation_v2.md)
gamma = torch.exp(-0.5 * kappa**2 * sigma_em**2 * d_perp_sq)

# Path loss
path_loss = 1.0 / (d_tx**2 * d_rx**2)                        # 1/R⁴ two-way

# Field amplitude = sqrt(power product)
A_g = opacity * torch.sqrt(f_cos_power * gamma * G_tx * G_rx * path_loss + 1e-20)
```

**Note on Tier 1 simplification**: This uses 4 of 6 ITU parameters.
`l_c` (correlation length) and `tau` (KA/SPM blend) require the full
Kirchhoff+SPM hybrid model. Tier 1 captures the dominant physics
(Fresnel reflection + Rayleigh coherence factor) and is appropriate
for smooth-to-moderate roughness surfaces. For highly rough surfaces
where the incoherent SPM lobe dominates, Tier 2 is needed.

### Tier 2: Full KA+SPM in PyTorch (physics-complete)

Full re-implementation of mmIR's KA+SPM hybrid BSDF. Uses all 6
parameters. ~200 lines of PyTorch. Provides direct numerical comparison
with mmIR. Implement when Tier 1 quality plateaus.

### Tier 3: Physics + learned residual (most expressive)

Each Gaussian stores the 6 ITU parameters PLUS L+1 Legendre coefficients:

```python
R_physics = fresnel_power_reflectance(cos_theta, eps_real, eps_imag) * eta
R_residual = evaluate_legendre(coeffs, cos_theta)
R_total = R_physics + R_residual
```

The physics component enables cross-sensor transfer. The residual
captures effects the ITU model can't represent. For transfer: set
residual to zero and re-render with physics only.

### Per-Gaussian parameters

| Tier | Physics params | Learned params | Total | Transfer |
|------|---------------|----------------|-------|----------|
| 1 | 6 (ITU) | 0 | 6 | Yes |
| 2 | 6 (ITU, all used) | 0 | 6 | Yes |
| 3 | 6 (ITU) + L+1 (residual) | L+1 | 6+L+1 | Physics part only |

---

## 2. Multi-Bounce Without Ray Tracing

### Single-bounce vs multi-bounce spatial requirements

Single-bounce requires NO spatial acceleration structure — it's a direct
evaluation at each Gaussian's known position.

Multi-bounce requires finding nearby Gaussian pairs. This uses a
**spatial index** (voxel hash or `torch_cluster.radius`), which is a
different structure from a BVH:
- BVH: directional query ("what does this ray hit?") — O(log N) per ray
- Spatial hash: proximity query ("what's near this point?") — O(1) expected

The spatial hash is simpler (no tree construction, no traversal) but
serves a different purpose. We are NOT claiming "no spatial structure" —
we are replacing directional ray queries with proximity queries.

### Pair enumeration

```
For each pair (g1, g2) with ||μ_g1 - μ_g2|| < r_interaction:

    # Geometry
    d1  = ||μ_g1 - p_tx||,  d12 = ||μ_g1 - μ_g2||,  d2 = ||μ_g2 - p_rx||

    # Directions and angles
    θ_g1_in  = angle(TX → g1, n_g1)
    θ_g1_out = angle(g1 → g2, n_g1)
    θ_g2_in  = angle(g1 → g2, n_g2)
    θ_g2_out = angle(g2 → RX, n_g2)

    # Amplitude (shading model at both bounces, field amplitude)
    f1 = shade(θ_g1_in, θ_g1_out, mat_g1)
    f2 = shade(θ_g2_in, θ_g2_out, mat_g2)
    A_2b = opacity_g1 * opacity_g2 * sqrt(f1 * f2 / (d1² × d12² × d2²))

    # Phase and scatter (same as single-bounce)
    R_tot = d1 + d12 + d2
    φ(t_k) = 2π(f₀ + S·t_k) · R_tot / c
    ADC[tx, rx, k] += A_2b · exp(j · φ(t_k))
```

### Pruning (critical for efficiency)

Evaluating ALL pairs within r_max is wasteful — most contribute negligible
energy. Pruning strategies:

1. **Normal compatibility**: Skip pairs where n_g1 points away from g2
   AND n_g2 points away from g1 (no physical path exists).
2. **Amplitude threshold**: Pre-compute an upper bound on A_2b from
   opacity and distance; skip pairs below threshold.
3. **Importance sampling**: Instead of all pairs, sample P pairs
   weighted by estimated contribution. Include MC normalisation (1/pdf).

With pruning, typically 10-30% of geometric neighbors contribute
meaningfully.

### Cost analysis (corrected)

With N = 100K Gaussians, r_max = 2m, ~100 neighbors per Gaussian,
30% pass pruning:

- Candidate pairs: 100K × 100 / 2 = 5M (undirected)
- After pruning: ~1.5M active pairs
- Per pair per TX-RX: ~50 FLOPs (geometry + shade + phase)
- Total: 1.5M × 192 × 50 = 14.4B FLOPs

At 82 TFLOPS peak (4090 FP32), this is ~0.2ms compute. However, the
operation is **memory-bandwidth bound** — each pair reads positions,
normals, and materials for two Gaussians plus TX/RX positions. With
~200 bytes read per pair per TX-RX: 1.5M × 192 × 200 = 55 GB of reads.
At 1 TB/s bandwidth (4090): ~55ms.

With chunking and memory-efficient implementation: **~50-100ms per
multi-bounce iteration** is a realistic estimate.

---

## 3. Sensor-Agnostic Representation

The Gaussian's material parameters (ITU or ITU + residual) are properties
of the SURFACE — they don't depend on the radar hardware.

The sensor-specific components are:
- TX/RX positions → determines distances and angles (geometry)
- Antenna pattern → determines per-element gain (applied AFTER material)
- Chirp parameters → determines phase (applied AFTER material)

A trained scene can be re-rendered for any sensor by swapping the
sensor config. The material parameters transfer because they describe
the surface, not the observation.

The antenna pattern is stored as a PyTorch tensor (loaded from the same
.npy files as mmIR) and evaluated via differentiable bilinear
interpolation. It can optionally be learned jointly with the scene
(shared across all Gaussians).

---

## 4. Doppler / Temporal Modeling

Each Gaussian can carry a velocity vector v_g:

```
v_radial = v_g · (r̂_tx + r̂_rx) / 2
f_doppler = 2 · v_radial / λ
φ(t_k) = 2π(f₀ + S·t_k) · R_tot/c  +  2π · f_doppler · t_k
```

For multi-chirp (slow-time) simulation, each chirp is rendered with
time-advanced positions:

    μ_g(m) = μ_g(0) + v_g · m · T_chirp_rep

**Cost**: Rendering M chirps costs M× the single-chirp render time.
For a typical 128-chirp frame at ~500ms/chirp, this is ~64s — too slow
for real-time but feasible for offline simulation.

**Approximation for speed**: If Gaussians move slowly (v × T_frame ≪
range resolution), the position change across chirps is negligible.
Only the PHASE changes: Δφ_slow(m) = 2π · f_doppler · m · T_chirp_rep.
This is a linear phase ramp that can be applied WITHOUT re-rendering:

```python
ADC_frame[m, tx, rx, k] = ADC_single[tx, rx, k] * exp(j * 2π * f_doppler * m * T_rep)
```

Cost: O(M × N_tx × N_rx × K) multiply — negligible.
Validity: when max(v) × M × T_rep ≪ Δrange (~59mm). For v = 30 m/s,
M = 128, T_rep = 50μs: displacement = 0.2mm ≪ 59mm. Valid for
automotive speeds.

---

## 5. Differentiable Sensor Design

The pipeline is differentiable w.r.t. sensor parameters (TX/RX positions,
chirp parameters, antenna pattern). Gradients exist in the continuous
sense.

**Practical caveat**: Real antenna design involves discrete constraints
(integer element count, minimum λ/2 spacing, fabrication tolerances).
Continuous gradient descent on positions is useful for:
- Fine-tuning a given array layout (small perturbations)
- Correcting antenna position errors (calibration)
- Learning effective antenna patterns (shared across Gaussians)

It does NOT solve the full combinatorial array design problem (how many
elements, which topology). That requires discrete optimization methods
(genetic algorithms, simulated annealing) which could use this
differentiable renderer as a fast inner-loop evaluator.

---

## 6. Composable Scenes

Gaussians are self-contained primitives. Scenes can be composed by
concatenating Gaussian sets:

```python
scene = merge(building_gaussians, vehicle_gaussians, ground_gaussians)
adc = render(scene, radar_config)
```

Enables scene editing, counterfactual simulation, and training data
augmentation for radar perception models.

---

## 7. What the Framework Looks Like

### Per-Gaussian parameters

| Parameter | Size | Description |
|-----------|------|-------------|
| μ | 3 | Position |
| q | 4 | Rotation (tangent frame → normal) |
| s₁, s₂ | 2 | Geometric scales |
| α | 1 | Opacity (logit) |
| ITU material | 6 | eps_real, eps_imag, sigma_h, l_c, tau, thickness |
| σ_em | 1 | EM coherence scale |
| **Total** | **17** | 68 bytes per Gaussian (Tier 1) |

Optional extensions: +L+1 Legendre coefficients (Tier 3), +3 velocity (Doppler).

### Forward pass (pure PyTorch)

```python
def render(gaussians, sensor_config):
    # 1. PROJECT: distances and angles
    d_tx = pairwise_distance(gaussians.positions, sensor_config.tx_pos)   # (N, N_tx)
    d_rx = pairwise_distance(gaussians.positions, sensor_config.rx_pos)   # (N, N_rx)
    cos_theta = compute_incidence_cosine(gaussians, sensor_config)        # (N, N_tx, N_rx)

    # 2. SHADE: physics BSDF in PyTorch (returns POWER)
    f_cos_power = bsdf(cos_theta, gaussians.material)                    # (N, N_tx, N_rx)
    G = antenna_pattern(cos_theta, sensor_config.pattern)                 # (N, N_tx, N_rx)
    gamma = coherence_factor(gaussians, sensor_config)                    # (N, N_tx, N_rx)
    path_loss = 1.0 / (d_tx**2 * d_rx**2)                               # (N, N_tx, N_rx)

    # FIELD AMPLITUDE = sqrt(power product)  [matches mmIR convention]
    A = gaussians.opacity * torch.sqrt(
        f_cos_power * gamma * G * path_loss + 1e-20)                     # (N, N_tx, N_rx)

    # 3. PHASE: exact from distances (detached from AD graph)
    tau = (d_tx + d_rx) / C
    phi = 2 * pi * tau * (f0 + S * t_grid)                               # (N, N_tx, N_rx, K)

    # 4. SCATTER: phasor accumulation (checkpointed)
    ADC = phasor_scatter(A, phi)                                          # (N_tx, N_rx, K)
    return ADC
```

---

## Summary

| Property | mmIR | Gaussian Rasterizer |
|----------|------|-------------------|
| Scene traversal | Ray tracing (BVH) | Direct evaluation (no BVH) |
| Material model | DrJit BSDF (no grads to materials) | PyTorch Fresnel (full autograd) |
| Antenna model | DrJit patterns | PyTorch tensor (differentiable) |
| Multi-bounce | Ray tracing (stochastic MC) | Pair enumeration (spatial hash, deterministic) |
| Doppler | Not supported | Velocity per Gaussian |
| Amplitude convention | √(BSDF × G × 1/R⁴) | √(BSDF × G × 1/R⁴) (same) |
| Sensor transfer | Re-render with new config | Re-render (same physics params) |
| Framework | DrJit + PyTorch bridge | Pure PyTorch |
| Training speed | 10-15 min/scene | ~4 min/scene (expected faster without DrJit) |
