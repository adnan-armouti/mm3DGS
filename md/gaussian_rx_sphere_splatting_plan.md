# Gaussian-to-RX-Sphere Splatting: Implementation Plan

## Core Idea

Replace the Gaussian renderer's area weighting with a solid-angle projection onto each RX element's unit sphere. This makes the Gaussian rendering equation algebraically identical to mmIR's RX-centric MC formulation.

## Mathematical Equivalence

### mmIR (MC over RX hemisphere)

mmIR integrates the received field over the solid angle at each RX:

```
E[t,r,k] = C * integral_Omega_r sqrt(f_cos * G_t * G_r / d_t^2) * exp(j*phi) dOmega_r
```

MC estimate with N samples at PDF p(omega):

```
E[t,r,k] ≈ C * (1/N) * sum_j sqrt(f_cos_j * G_t_j * G_r_j / d_t_j^2) * exp(j*phi_j) / p(omega_j)
```

The weight per sample:

```
w_j = C * sqrt(f_cos_j * G_t_j * G_r_j / (d_t_j^2 * p_j * N))
```

### Gaussian splatting (direct sum, current broken approach)

The current code sums over Gaussians with area weights:

```
E[t,r,k] = C * sum_i sqrt(f_cos_i * G_t_i * G_r_i * A_i * alpha_i / d_t_i^2) * exp(j*phi_i)
```

This is wrong because it's missing the solid-angle Jacobian. It treats the integration as if it's over surface area, but the radar equation's `1/(d_t^2 * d_r^2)` path loss comes from the fact that received power falls off with both distances. The current code only has `1/d_t^2`.

### Gaussian splatting (RX-sphere projection, correct approach)

Each Gaussian subtends a solid angle at each RX element:

```
dOmega_i^r = A_i * |cos(theta_rx_i^r)| / d_r_i^{r,2}
```

Substituting this as the integration measure:

```
E[t,r,k] = C * sum_i sqrt(f_cos_i * G_t_i * G_r_i * dOmega_i^r / d_t_i^2) * exp(j*phi_i)

         = C * sum_i sqrt(f_cos_i * G_t_i * G_r_i * A_i * |cos(theta_rx_i^r)| / (d_t_i^2 * d_r_i^{r,2})) * exp(j*phi_i)
```

This is **algebraically equivalent** to mmIR's formulation:
- mmIR: `dOmega` comes from the MC sample's effective solid angle `1/(p * N)`
- Splatting: `dOmega` comes from the Gaussian's geometric projection `A * |cos(theta_rx)| / d_r^2`

Both compute `C * sqrt(f_cos * G_t * G_r * dOmega / d_t^2)`. The only difference is how `dOmega` is obtained — stochastic sampling vs deterministic geometry.

## How our current approach differs from RX-sphere projection

The current code (`_render_chunk_torch` line 444) computes:

```
path_loss = 1 / d_tx^2
weight = C * sqrt(brdf_weight * path_loss)
       = C * sqrt(f_cos * G_t * G_r * A_i * alpha_i / d_tx^2)
```

The RX-sphere splatting computes:

```
path_loss = 1 / d_tx^2
dOmega_rx = A_i * |cos(theta_rx)| / d_rx^2
weight = C * sqrt(f_cos * G_t * G_r * dOmega_rx * alpha_i / d_tx^2)
       = C * sqrt(f_cos * G_t * G_r * A_i * alpha_i * |cos(theta_rx)| / (d_tx^2 * d_rx^2))
```

The difference is exactly two factors:

| Factor | Current | RX-sphere | Ratio |
|--------|---------|-----------|-------|
| `1/d_rx^2` | Missing | Present | `d_rx^2` (range 2.25 to 900 for 1.5-30m scenes) |
| `\|cos(theta_rx)\|` | Missing | Present | 0 to 1 (angular dependence) |

The `1/d_rx^2` factor is the dominant error — it can produce up to 400x error in relative weighting between near and far Gaussians.

The `|cos(theta_rx)|` factor provides grazing-angle suppression: Gaussians viewed edge-on from the RX contribute less solid angle. This is physically correct (a tilted surface subtends less solid angle) and was implicitly handled by mmIR's cosine hemisphere PDF cancellation.

Note: the bistatic fix currently running (from the previous session) adds `1/d_rx^2` but NOT `|cos(theta_rx)|`. The RX-sphere splatting includes both.

## Implementation

### What changes

**One code change** in `_render_chunk_torch` (rasterizer_torch.py):

Before the radar equation (currently around line 441), compute the per-path solid angle:

```python
# RX-sphere splatting: project Gaussian area onto RX unit sphere
# dOmega = A * |cos(theta_rx)| / d_rx^2
# This replaces the scalar area weight with a per-(Gaussian, RX) solid angle.
if bistatic_path_loss:
    cos_theta_rx = (dir_to_rx * hit_N).sum(-1).abs()  # already computed as cos_theta_out
    dOmega_rx = areas_t[v_idx] * cos_theta_rx / (d_rx.clamp(min=1e-4) ** 2)
    brdf_weight = brdf_weight / areas_t[v_idx] * dOmega_rx  # replace area with solid angle
    # path_loss stays as 1/d_tx^2 (d_rx^2 is now in dOmega_rx)
    path_loss = 1.0 / (d_safe_tx * d_safe_tx)
```

Or equivalently (simpler, same result):

```python
if bistatic_path_loss:
    cos_theta_rx = cos_theta_out  # already computed, = dot(dir_to_rx, hit_N)
    path_loss = cos_theta_rx / (d_safe_tx ** 2 * d_rx.clamp(min=1e-4) ** 2)
else:
    path_loss = 1.0 / (d_safe_tx ** 2)
```

The `cos_theta_rx` is already available as `cos_theta_out` (computed on line 408 for cosine filtering). We just need to use it as a multiplicative weight instead of a binary threshold.

### What does NOT change

- `_render_chunk_torch` for MC-weighted paths (`bistatic_path_loss=False`): unchanged, preserves Stage A/B
- The BSDF evaluation: unchanged
- The antenna gain evaluation: unchanged
- The phase computation: unchanged
- The radar constants: unchanged
- The `render()` method (non-differentiable, MC mode): unchanged

### Where the flag is set

- `render_gaussians()` in `train_gaussian.py`: sets `bistatic_path_loss=True`
- `render()` and `render_differentiable()` for MC paths: `bistatic_path_loss=False` (default)

## Experiments to re-run

**Stage A**: NO re-run. MC path unchanged.

**Stage B**: NO re-run. Uses MC weights, `bistatic_path_loss=False`.

**Stage C3**: Re-run all 7 scenes (Gaussian from mmIR init, 500 iters).

**Stage C4**: Re-run all 7 scenes (Gaussian from scratch, 500 iters).

Total: 14 scene runs. ~8 min each = ~2 hours.

## Expected impact

The RX-sphere splatting makes the Gaussian rendering equation algebraically equivalent to mmIR's. The remaining differences are:

1. **Sampling density**: mmIR importance-samples ~24K points on triangle surfaces; Gaussians are fixed at ~12K vertex positions. The solid angle coverage per Gaussian is `A_i * |cos| / d_r^2`, while the MC coverage per sample is `1/(p * N)`. These produce different spatial distributions, but both are valid discretizations of the same integral.

2. **Shadow test**: Still disabled for Gaussians. Some occluded Gaussians contribute energy that mmIR would block.

3. **Position**: Gaussians are at mesh vertices, not on triangle surfaces. Materials are per-vertex, not barycentric-interpolated.

The `1/d_rx^2` fix alone should significantly improve the range profile accuracy. The `|cos(theta_rx)|` factor adds physically correct grazing-angle suppression. Together, these make the Gaussian formulation as close to mmIR as possible without re-introducing ray tracing.
