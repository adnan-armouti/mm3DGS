# Radar Rendering Equation: mmIR vs mm25DGS Derivation Comparison

## 1. Physical Radar Equation (Ground Truth)

The bistatic radar equation for the received power from a differential surface element `dA` at point `x`, illuminated by TX element `t` and received by RX element `r`:

```
dP_r(x) = (P_t * L_ant * lambda^2) / (4*pi)^2
           * f_bsdf(wo, wi, n, materials) * cos(theta_in)
           * G_t(dir_t) * G_r(dir_r)
           * dA / (d_t^2 * d_r^2)
```

where:
- `P_t` = TX power (watts)
- `L_ant` = antenna loss factor
- `lambda` = wavelength
- `f_bsdf * cos(theta_in)` = BSDF power (1/sr) times cosine = `f_cos`
- `G_t, G_r` = TX and RX antenna gains (linear)
- `d_t = |x - TX_pos|`, `d_r = |x - RX_pos|`
- `dA` = surface area element

The total received **field amplitude** (E-field) from the entire surface:

```
E_rx[t,r,k] = integral_S  C * sqrt(f_cos * G_t * G_r / (d_t^2 * d_r^2))
                           * exp(j * phi(x, k))
                           * sqrt(dA)
```

where `C = sqrt(P_t * L_ant * lambda^2 / (4*pi)^2) * rx_scale * adc_scale` absorbs all constants, and the `sqrt()` converts power to field amplitude.

**Key**: the physical equation has **`1/(d_t^2 * d_r^2)`** — both TX and RX distances appear.

---

## 2. mmIR Implementation (MC Integration over RX Hemisphere)

mmIR evaluates the integral using Monte Carlo integration over the hemisphere at each RX element. Rays are cast from RX into the scene; hits are found via the reservoir sampler.

### Change of variables: surface area -> RX solid angle

The hemisphere solid angle element and surface area element are related by:

```
d_omega_r = cos(theta_rx) * dA / d_r^2
```

where `theta_rx` is the angle between the surface normal and the RX direction.

Substituting `dA = d_r^2 * d_omega_r / cos(theta_rx)` into the power integral:

```
P_r = C^2 * integral_Omega_r [f_cos * G_t * G_r / (d_t^2 * d_r^2)]
                               * [d_r^2 / cos(theta_rx)]
                               * d_omega_r

    = C^2 * integral_Omega_r [f_cos * G_t * G_r] / [d_t^2 * cos(theta_rx)]
                               * d_omega_r
```

**Key observation**: the `d_r^2` in the denominator cancels with the `d_r^2` from the area-to-solid-angle Jacobian. Only `d_t^2` remains.

### MC estimator with cosine hemisphere sampling

With PDF `p(omega) = cos(theta_rx) / pi`:

```
P_r ≈ C^2 * (1/N) * sum_j [f_cos_j * G_t_j * G_r_j] / [d_t_j^2 * cos(theta_rx_j) * p(omega_j)]

    = C^2 * (1/N) * sum_j [f_cos_j * G_t_j * G_r_j] / [d_t_j^2 * cos(theta_rx_j)]
                    * [pi / cos(theta_rx_j)]
```

But mmIR's code simplifies this differently. Looking at `synthesis_forward.py` line 1137:

```
P_r = (P_t * lambda^2) / (4*pi)^2 * (1/N_r) * sum_j [G_t * G_r * c_t * f * V] / [d_t^2 * rho_rx(omega_j)]
```

where `rho_rx(omega_j)` is the sampling PDF and `1/N_r` is the `1/n_attempted` factor.

### What mmIR actually computes (code trace)

**File**: `synthesis_forward.py`, lines 1000-1275

```
# Step 5: BSDF (power/sr * cosine)
brdf_weight = bsdf.eval_f_cos_physics(wo, wi, n, materials)    # [n_total]

# Step 6: Antenna gain (linear power)
brdf_weight *= G_tx_linear * G_rx_linear                        # [n_total]

# Step 7: MC correction
brdf_weight *= 1 / (pdf * n_attempted)                          # [n_total]

# Step 9: Path loss + field amplitude
path_loss = 1 / d_t^2                                           # ONLY TX distance
weight = C_radar * sqrt(brdf_weight * path_loss)                 # amplitude
```

Fully expanded:
```
weight_j = C_radar * sqrt( f_cos_j * G_t_j * G_r_j / (pdf_j * N * d_t_j^2) )
```

This is **correct** for the MC hemisphere estimator because the `d_r^2` cancellation is implicit in the change of variables.

---

## 3. DrJit Rasterizer (`rasterizer.py`, hits mode)

**File**: `rasterizer.py`, `_render_vertex_chunk`, lines 336-484

The DrJit rasterizer uses the SAME reservoir sampler hits as mmIR, with the SAME MC weights:

```
# From render() line 506-507:
areas = 1.0 / (pdf * n_attempted)                # MC importance weight

# In _render_vertex_chunk:
brdf_weight = bsdf.eval_f_cos_physics(wo, wi, n, materials)
brdf_weight *= G_tx * G_rx
brdf_weight *= areas[v_idx]                       # = 1/(pdf*N)
path_loss = 1 / d_tx^2                            # ONLY TX distance
weight = C_radar * sqrt(brdf_weight * path_loss)
```

**Result**: `weight_j = C_radar * sqrt(f_cos_j * G_t_j * G_r_j / (pdf_j * N * d_t_j^2))`

**Identical to mmIR**. This is why Stage A verification passed with RA corr > 0.999.

---

## 4. PyTorch Rasterizer (`rasterizer_torch.py`)

**File**: `rasterizer_torch.py`, `_render_chunk_torch`, lines 362-485

Identical physics chain to DrJit rasterizer:

```
physics = reparameterize_torch(raw_params_t)
brdf_weight = evaluate_bsdf_jones_f_cos(cos_theta_in, wo, wi, n, eps_r, eps_i, sigma_h, l_c, tau, thickness)
brdf_weight *= G_tx * G_rx
brdf_weight *= areas_t[v_idx]                     # passed in from caller
path_loss = 1 / d_tx^2
weight = C_radar * sqrt(brdf_weight * path_loss)
```

When called from `render()` (line 328): `areas = 1/(pdf * n_attempted)` -- MC weights, same as mmIR.
When called from `render_differentiable()`: whatever `areas_t` is passed in from the caller.

**Identical to mmIR when MC weights are used** (Stage A, Stage B).

---

## 5. Gaussian Renderer (`train_gaussian.py` via `render_differentiable`)

**File**: `train_gaussian.py`, `render_gaussians`, lines 372-411

```python
positions = model.positions           # Gaussian centers (NOT hit positions)
normals = model.get_normals()         # from quaternion (NOT mesh vertex normals)
opacities = model.get_opacities()     # sigmoid(logit)
raw_materials = model.raw_materials

areas = vertex_areas * opacities      # <-- CRITICAL DIFFERENCE
```

Then calls `rast.render_differentiable(raw_materials, normals, positions, areas, ...)` which runs the same `_render_chunk_torch`:

```
brdf_weight = evaluate_bsdf_jones_f_cos(...)
brdf_weight *= G_tx * G_rx
brdf_weight *= areas[v_idx]           # = vertex_area_i * opacity_i
path_loss = 1 / d_tx^2               # ONLY TX distance
weight = C_radar * sqrt(brdf_weight * path_loss)
```

Fully expanded:
```
weight_i = C_radar * sqrt( f_cos_i * G_t_i * G_r_i * A_i * alpha_i / d_t_i^2 )
```

---

## 6. What the Gaussian Renderer SHOULD Compute

The Gaussian renderer performs direct summation over discrete surface elements (Gaussians) rather than MC integration. The correct equation for direct summation over surface area is:

```
E_rx[t,r,k] = sum_i C_radar * sqrt(f_cos_i * G_t_i * G_r_i * A_i / (d_t_i^2 * d_r_i^2))
                    * exp(j * phi_i(k))
```

This follows directly from the physical radar equation (Section 1) with `dA -> A_i`.

### What the Gaussian renderer actually computes:

```
E_rx[t,r,k] = sum_i C_radar * sqrt(f_cos_i * G_t_i * G_r_i * A_i * alpha_i / d_t_i^2)
                    * exp(j * phi_i(k))
```

---

## 7. Errors and Deviations

### Error 1: MISSING `1/d_r^2` factor (CRITICAL)

| | mmIR (MC) | Gaussian (direct sum) |
|---|-----------|----------------------|
| Path loss | `1/d_t^2` | `1/d_t^2` |
| Expected | `1/d_t^2` (correct, d_r^2 absorbed by MC Jacobian) | **`1/(d_t^2 * d_r^2)`** (direct sum needs both) |
| Actual | `1/d_t^2` (correct) | `1/d_t^2` (WRONG — missing d_r^2) |

**Explanation**: In mmIR's MC formulation, rays are cast from the RX into the hemisphere. The change of variables `dA = d_r^2 * d_omega / cos(theta_rx)` absorbs the `d_r^2` from the radar equation denominator. When switching to direct surface-area summation, there is no such change of variables — the full `1/(d_t^2 * d_r^2)` must be present.

**Impact**: Gaussians at different RX distances are weighted incorrectly. A Gaussian at `d_r = 20m` should contribute `(20/2)^2 = 100x` less than one at `d_r = 2m`, but the current code treats them equally (modulo `d_t` differences). Since scenes span 1.5m to 30m in range, this is a factor of up to 400x error in relative weighting.

### Error 2: MISSING `cos(theta_rx) / d_r^2` Jacobian factor

More precisely, the conversion from MC hemisphere to direct surface-area sum requires:

```
1/(pdf * N) --> A_i * cos(theta_rx_i) / d_r_i^2
```

The `cos(theta_rx)` (cosine of the angle between surface normal and RX direction) appears because the solid-angle-to-area Jacobian includes it. In the MC estimator, this cosine cancels with the cosine in the sampling PDF (`p(omega) = cos(theta)/pi`). In direct summation, it must be included explicitly.

The current code uses `A_i * alpha_i` as the weight, which is missing both `cos(theta_rx)` and `1/d_r^2`.

**Note**: The `cos(theta_rx)` factor is partially captured by the BSDF's cosine filtering (`active = cos_theta_out > 1e-6`), which zeros out back-facing paths. But it does NOT apply the actual `cos(theta_rx)` as a continuous weight — it's a binary threshold, not a multiplicative factor.

### Error 3: Area weight represents wrong physical quantity

| Weight | mmIR | Gaussian renderer |
|--------|------|-------------------|
| Formula | `1/(pdf_j * N)` | `vertex_area_i * opacity_i` |
| Physical meaning | MC correction = effective solid angle per sample | Geometric area of Voronoi cell times learned opacity |
| Units | sr (solid angle, at RX) | m^2 (surface area) |
| Depends on d_r? | Implicitly, via `dA = d_r^2 * d_omega / cos(theta)` | No |
| Depends on viewing angle? | Implicitly, via hemisphere PDF | No |

These are fundamentally different quantities. The MC weight accounts for the sampling geometry (distance, viewing angle); the vertex area does not. Even with the `1/d_r^2` fix, the vertex area does not perfectly replace the MC weight because the Gaussians may not uniformly cover the surface.

### Error 4: Positions are at vertices, not on triangle surfaces

mmIR (and Stage B mesh training) uses reservoir-sampled hit positions that lie on triangle **surfaces** with barycentric interpolation of materials and normals. The Gaussian renderer uses mesh **vertex** positions. For the same mesh, these are different points:

- Hit positions sample the interior of triangles (smooth coverage of the surface)
- Vertex positions are only at triangle corners (irregular, sparse at large triangles)

This means the Gaussian renderer under-samples large triangles and over-samples vertices shared by many small triangles. The barycentric material interpolation (which smoothly blends materials across triangle faces) is lost.

### Error 5: Shadow test disabled

mmIR and the DrJit/PyTorch rasterizers (Stage A/B) use Mitsuba shadow rays to test TX-side occlusion. The Gaussian renderer sets `skip_shadow=True`, so occluded Gaussians contribute energy that should be blocked. This adds spurious signal at certain range-angle bins.

---

## 8. Summary: Correct Gaussian Rendering Equation

The physically correct weight for direct-summation Gaussian rasterization is:

```
weight_i = C_radar * sqrt(
    f_cos_i * G_t_i * G_r_i * A_i * alpha_i
    / (d_t_i^2 * d_r_i^2)
)
```

where:
- `f_cos_i = BSDF(wo_i, wi_i, n_i, materials_i) * cos(theta_in_i)` — power/sr * cosine
- `G_t_i, G_r_i` — TX and RX antenna gains at the direction from element to Gaussian
- `A_i` — geometric area of Gaussian surfel
- `alpha_i` — opacity (learned, [0,1])
- `d_t_i = |Gaussian_i - TX_pos|` — TX distance
- `d_r_i = |Gaussian_i - RX_pos|` — RX distance

The current code computes:

```
weight_i = C_radar * sqrt(
    f_cos_i * G_t_i * G_r_i * A_i * alpha_i
    / d_t_i^2
)
```

**The missing `1/d_r_i^2` is the dominant error.** For a MIMO radar with 16 RX elements, each at slightly different positions, the d_r values are per (Gaussian, RX) pair. This factor must be computed inside the per-path loop, not as a pre-computed area weight.

---

## 9. Where Each Error Lives in Code

| Error | File | Line(s) | Current | Correct |
|-------|------|---------|---------|---------|
| Missing 1/d_r^2 | `rasterizer_torch.py` | 444 | `path_loss = 1/d_tx^2` | `path_loss = 1/(d_tx^2 * d_rx^2)` when not using MC weights |
| Missing cos(theta_rx) | `rasterizer_torch.py` | 441-448 | Not included | `brdf_weight *= cos_theta_out` |
| Area weight semantic | `train_gaussian.py` | 349 | `areas = vertex_areas * opacities` | Correct if 1/d_r^2 is added to path loss |
| Vertex vs surface positions | `train_gaussian.py` | 343-346 | `model.positions` (vertices) | Inherent to Gaussian representation (not fixable) |
| Shadow test | `train_gaussian.py` | 411 | `skip_shadow=True` | Implement rasterization-native occlusion |
