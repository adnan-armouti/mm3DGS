# Option 3: Analytic Intra-Surfel Phase Integration

## The problem (quantified)

Forward-only rendering (no training) with mmIR-trained materials:

| Renderer | cart_corr | What it does |
|----------|-----------|-------------|
| v2 Stage B | 0.870 | 24K importance-sampled surface hits, MC weights `1/(pdf×N)` |
| v3 C3 (mesh verts) | 0.581 | 50K mesh vertices, geometric face-area weights |
| v3 C4 (LiDAR 50K) | 0.472 | 50K FPS LiDAR points, uniform weights (=1) |
| v3 C4 (LiDAR 200K) | 0.519 | 200K LiDAR points, uniform weights (=1) |

The 0.29 gap between Stage B and v3 persists even at 200K Gaussians. More points doesn't help. The issue is in the **integration scheme**, not the point count.

## Root cause analysis

### What Stage B computes (MC integral)

For each (TX, RX) channel, the received field is an integral over the visible surface:

```
E[t,r] = ∫_S w(x) × exp(j×φ(x)) dA(x)
```

The MC estimator evaluates this as:

```
E[t,r] ≈ (1/N) × Σ_j w(x_j) × exp(j×φ(x_j)) / p(x_j)
```

where `x_j` are reservoir-sampled points on triangle surfaces, and `p(x_j)` is the sampling PDF. The key: **each sample has a different phase `φ(x_j)`** because each is at a different position on the surface.

### What v3 computes (point evaluation)

```
E[t,r] ≈ Σ_i w(μ_i) × exp(j×φ(μ_i)) × A_i
```

where `μ_i` is the Gaussian center and `A_i` is its area weight. Each Gaussian contributes a **single phasor** at the phase determined by its center position.

### Why this fails

A flat surface patch of area A centered at position μ with normal n contributes:

```
E_patch = ∫_patch w(x) × exp(j×φ(x)) dA(x)
```

The v3 point-evaluation approximation replaces this with:

```
E_point = w(μ) × exp(j×φ(μ)) × A
```

This assumes `φ(x)` is constant across the patch. But `φ(x) = 2π(f₀+S×t_k)(d_tx(x)+d_rx(x))/c`, and the distances `d_tx(x)`, `d_rx(x)` vary linearly across the patch. For a patch of extent β perpendicular to the radar direction:

```
Δφ = 2π × (f₀ + S×t_k) × Δd / c
```

where `Δd ≈ β × sin(θ)` (θ = angle between surface normal and radar direction).

At 77 GHz, `f₀/c ≈ 257 rad/mm`. For a 1cm patch at 45° incidence: `Δφ ≈ 257 × 10 × 0.707 = 1817 rad = 289 full cycles`. The phasor rotates 289 times across the patch. Summing to a single point sample is wildly wrong.

**This is why more points barely help** — even at 200K points, each still represents a mm-scale patch with hundreds of wavelengths of phase variation.

## Option 3: Analytic integration over the surfel

Instead of evaluating the phasor at a single point, analytically integrate it over the surfel's spatial extent.

### The integral

For a Gaussian surfel at center μ with tangent vectors t₁, t₂ and scales s₁, s₂:

```
E_surfel = w(μ) × ∫∫ exp(j×φ(μ + u×t₁ + v×t₂)) × G(u,v; s₁,s₂) du dv
```

where `G(u,v) = exp(-(u²/2s₁² + v²/2s₂²))` is the Gaussian spatial profile.

### Linear phase approximation

The phase `φ(x) = 2π×f_k×(d_tx(x)+d_rx(x))/c` varies approximately linearly across a small patch:

```
φ(μ + δ) ≈ φ(μ) + ∇φ · δ
```

where the phase gradient is:

```
∇φ = 2π×f_k/c × (∇d_tx + ∇d_rx) = 2π×f_k/c × (dir_to_tx + dir_to_rx)
```

Wait — `∇d_tx = -dir_hit_to_tx` (the unit vector from hit toward TX, negated because distance increases away from TX). So:

```
∇φ = 2π×f_k/c × (-dir_to_tx - dir_to_rx) = -2π×f_k/c × (dir_to_tx + dir_to_rx)
```

Actually, more carefully: `d_tx(x) = |TX - x|`, so `∇_x d_tx = -(TX - x)/|TX - x| = -dir_hit_to_tx`. And `d_rx(x) = |x - RX|`, so `∇_x d_rx = (x - RX)/|x - RX| = dir_hit_away_from_rx = -dir_hit_to_rx`.

So: `∇φ = 2π×f_k/c × (-dir_to_tx - dir_to_rx)` — this is the **bistatic wave vector**.

### Gaussian Fourier transform (closed form)

The integral of a Gaussian times a complex exponential has a closed-form solution:

```
∫∫ exp(-u²/2σ₁² - v²/2σ₂²) × exp(j×(k₁u + k₂v)) du dv
= 2π σ₁ σ₂ × exp(-σ₁²k₁²/2 - σ₂²k₂²/2)
```

This is the **Fourier transform of a 2D Gaussian** evaluated at spatial frequency (k₁, k₂).

### Applying to the surfel

The phase gradient projected onto the surfel's tangent plane gives the spatial frequency:

```
k₁ = ∇φ · t₁ = -2π×f_k/c × (dir_to_tx + dir_to_rx) · t₁
k₂ = ∇φ · t₂ = -2π×f_k/c × (dir_to_tx + dir_to_rx) · t₂
```

The intra-surfel integration factor (the "apodization") is:

```
apod = exp(-s₁²k₁²/2 - s₂²k₂²/2)
```

This is a real scalar ∈ (0, 1] that attenuates the contribution of large surfels with rapid intra-surfel phase variation.

### Physical interpretation

- **Small surfel** (s₁, s₂ << λ): k₁s₁ << 1, apod ≈ 1. The surfel is small enough that phase is ~constant across it. No attenuation.
- **Large surfel facing radar** (θ ≈ 0): `(dir_to_tx + dir_to_rx) · t₁ ≈ 0` because the bistatic direction is along the normal, perpendicular to the tangent plane. apod ≈ 1. Head-on surfels scatter coherently regardless of size.
- **Large surfel at grazing angle**: `(dir_to_tx + dir_to_rx) · t₁ ≈ sin(θ)` is large. apod → 0. Tilted large surfels have rapid phase variation across them and their coherent contribution cancels.

This is exactly the physics that the MC integral captures naturally (different sample points get different phases, leading to partial cancellation), but the point-evaluation misses entirely.

### What needs to change in the code

The change is minimal. In `rasterizer_factorized.py`, after computing `w_full` (the amplitude weight per path), multiply by the apodization factor:

```python
# Phase gradient projected onto surfel tangent plane
# bistatic_dir = dir_to_tx + dir_to_rx  (M, n_tx, n_rx, 3)
# But we don't form this 4D tensor. Instead:
# k1 = (2π f_k / c) × (dir_to_tx · t1 + dir_to_rx · t1)
# k2 = (2π f_k / c) × (dir_to_tx · t2 + dir_to_rx · t2)

# Need tangent vectors t1, t2 from the quaternion rotation
t1, t2, _ = model.get_tangent_frame()  # (M, 3) each

# dir_to_tx · t1: (M, n_tx) = einsum('mtj,mj->mt', wi, t1)
# dir_to_rx · t1: (M, n_rx) = einsum('mrj,mj->mr', dir_hit_to_rx, t1)
# Sum: (M, n_tx, n_rx) = wi_dot_t1[:,:,None] + rx_dot_t1[:,None,:]

# This IS separable in TX and RX!

# k1_total(m,t,r) = (2π f_k / c) × (wi·t1[m,t] + wo·t1[m,r])
# k2_total(m,t,r) = (2π f_k / c) × (wi·t2[m,t] + wo·t2[m,r])

# But f_k varies per ADC sample... For range-profile splatting, we use the
# CENTER frequency f0 (the dominant phase contributor):
k_scale = 2 * pi * f0 / c  # scalar

wi_dot_t1 = einsum('mtj,mj->mt', wi, t1)      # (M, n_tx)
rx_dot_t1 = einsum('mrj,mj->mr', wo, t1)       # (M, n_rx)
k1 = k_scale * (wi_dot_t1[:,:,None] + rx_dot_t1[:,None,:])  # (M, n_tx, n_rx)

wi_dot_t2 = einsum('mtj,mj->mt', wi, t2)
rx_dot_t2 = einsum('mrj,mj->mr', wo, t2)
k2 = k_scale * (wi_dot_t2[:,:,None] + rx_dot_t2[:,None,:])

# Surfel scales
s1 = model.get_scales()[:, 0]  # (M,)
s2 = model.get_scales()[:, 1]  # (M,)

# Apodization factor
apod = exp(-(s1[:,None,None]**2 * k1**2 + s2[:,None,None]**2 * k2**2) / 2)

# Apply
w_full = w_full * apod
```

### What this changes in practice

For typical scenes:
- Surfels at normal incidence (facing radar): apod ≈ 1 → no change
- Large surfels at grazing angles: apod → 0 → correctly attenuated
- Small surfels (<< λ): apod ≈ 1 → no change

The apodization acts as a **physics-informed spatial filter** that correctly models the intra-surfel phase cancellation. It reduces the effective contribution of surfels that the current renderer incorrectly treats as coherent point sources.

### Key differences from F4 (the failed attempt)

F4 used `exp(-2*(π×slope×β×τ/c)²)` which:
1. Used the chirp slope (79 THz/s) instead of carrier frequency (77 GHz)
2. Used a 1D projected extent instead of 2D tangent-plane projection
3. Applied the same attenuation to all TX-RX pairs (no per-path variation)
4. Had no directional dependence (didn't account for incidence angle)

Option 3 uses `exp(-(s₁²k₁² + s₂²k₂²)/2)` which:
1. Uses the carrier frequency (correct physical scale)
2. Projects the bistatic wave vector onto the 2D tangent plane
3. Varies per (M, n_tx, n_rx) path
4. Naturally captures the angle dependence (head-on = no attenuation, grazing = full attenuation)

### Inputs needed from the model

- `t1, t2`: tangent vectors from `model.get_tangent_frame()` — already available
- `s1, s2`: surfel scales from `model.get_scales()` — already available
- `wi`, `wo`: TX and RX directions — already computed in steps 2 and 3
- `f0`: carrier frequency — already available as `rast.center_freq`

### Expected impact

The apodization will reduce the amplitude of large surfels at non-normal incidence. This should:
1. Reduce the "noise floor" in the RA image from incorrectly coherent large surfels
2. Improve the range profile accuracy (the dominant error source)
3. Bring the forward model closer to Stage B's MC integral

Whether it closes the full 0.29 gap depends on how well the linear phase approximation holds across the surfel extents in these scenes. For surfels smaller than ~λ/4 ≈ 1mm, the approximation is excellent. For larger surfels, higher-order phase terms matter.

### Risk

The surfel scales from PCA initialization may be poorly calibrated. If scales are too large (as seen in F4 where scales exploded), the apodization may be too aggressive. But with the scales FIXED at their init values (not optimized in this experiment), this should be stable.
