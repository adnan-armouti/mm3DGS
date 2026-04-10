# Factorized Gaussian-to-ADC Splatting: Implementation Plan

## Motivation

The current Gaussian renderer enumerates all N × 12 × 16 = 192N paths explicitly, running the full Jones BSDF, antenna gain, and phase computation per path. This is architecturally identical to ray tracing (per-path evaluation) — just with deterministic positions instead of stochastic ray hits.

True splatting eliminates the 192× path expansion by exploiting the factorizable structure of the FMCW radar equation.

## The factorization

The ADC contribution of Gaussian i to channel (t, r, k) is:

```
ADC[t,r,k] += w_i(t,r) × exp(j × φ_i(t,r,k))
```

### Phase factorization (EXACT)

```
φ_i(t,r,k) = 2π × f_k × (d_tx_i(t) + d_rx_i(r)) / c
            = 2π × f_k × d_tx_i(t)/c  +  2π × f_k × d_rx_i(r)/c
            = φ_tx_i(t,k)  +  φ_rx_i(r,k)
```

Therefore:
```
exp(jφ) = exp(jφ_tx) × exp(jφ_rx)
```

This is **exact** — no approximation. The FMCW beat phase is additive in TX and RX distances.

### Weight factorization

The amplitude weight from the radar equation is:

```
w_i(t,r) = C × sqrt( f_cos_i(t,r) × G_tx_i(t) × G_rx_i(r) × dΩ_rx_i(r) / d_tx_i(t)² )
```

This factors as:

```
w_i(t,r) = C × sqrt(f_cos_i(t,r)) × [sqrt(G_tx_i(t)) / d_tx_i(t)] × [sqrt(G_rx_i(r) × dΩ_rx_i(r))]
         = C × sqrt(f_cos_i(t,r)) × α_tx_i(t) × α_rx_i(r)
```

where:
- `α_tx_i(t) = sqrt(G_tx_i(t)) / d_tx_i(t)` — depends on (Gaussian i, TX t) only
- `α_rx_i(r) = sqrt(G_rx_i(r) × A_i × |cos θ_rx_i(r)| / d_rx_i(r)²)` — depends on (Gaussian i, RX r) only
- `f_cos_i(t,r)` — the BSDF, depends on BOTH TX and RX directions (not separable)

The BSDF `f_cos` is the only non-separable term. See options below.

---

## BSDF Factorization Options

The BSDF is `f_total = R_jones × (η × f_coh + (1-η) × f_inc)` where each component has different wo/wi dependency. Below we decompose each component to find what's truly coupled vs what separates.

### Internal structure analysis

**The full BSDF (line 494):**
```
f_total = R_jones(wi, wo, n, materials) 
        × [ η(wi,n,materials) × (τ_eff(wi,materials) × f_KA(wo,wi,n,materials)
                                 + (1-τ_eff) × f_SPM(wo,wi,n,materials))
          + (1-η) × (γ(materials) × f_dir(wo,wi,n,materials) 
                     + (1-γ) × f_broad(wi,n))
                   × cbs(wo,wi,n,materials) / cbs_mean(materials) ]
```

**Per-component dependency on wo vs wi:**

| Component | Depends on | Factorizable? |
|-----------|-----------|---------------|
| `R_jones` | `r_s(wi,n)`, `r_p(wi,n)`, `tx_s(wi,n)`, `tx_p(wi,n)`, `rx_s(wi,n)`, `rx_p(wo,n)` | Almost: only `p_out = cross(s_in, wo)` introduces wo |
| `η` (coherent frac) | `cos_theta_i = wi·n`, materials | **wi-only** |
| `τ_eff` (KA/SPM blend) | `cos_theta_i`, materials | **wi-only** |
| `γ` (dir/broad ratio) | materials only | **material-only** |
| `cbs_mean` | materials only | **material-only** |
| `f_KA` (GGX) | `h = norm(wo+wi)`, `cos_i = wi·n`, `cos_o = wo·n` | **Coupled** through h and Smith G |
| `f_SPM` (vMF) | `cos_dev = wo · reflect(wi,n)` | **Coupled** through specular reflection |
| `f_dir` (vMF) | `cos_dev = wo · reflect(wi,n)` | **Coupled** through specular reflection |
| `f_broad` (Lambertian) | `cos_i`, `cos_o` independently | **Separable** |
| `cbs` | `wo · (-reflect(wi,n))` | **Coupled** |

### The key insight: most coupling is through KNOWN geometric quantities

The coupling between wo and wi is NOT arbitrary — it's through specific geometric constructs:
- **Half vector**: `h = normalize(wo + wi)` → `h·n` determines KA lobe
- **Specular reflection**: `wi_r = 2(wi·n)n - wi` → `wo·wi_r` determines SPM/directive lobes
- **Retroreflection**: `retro = -wi_r` → `wo·retro` determines CBS

For a given surface point with normal n, these are all functions of the **angles** of wo and wi relative to n. We can precompute the wi-dependent parts and express the wo-dependent parts as functions of `cos_o = wo·n` and the deviation angle from specular.

---

### Option 1: Monostatic approximation (1 eval per Gaussian)

**REJECTED** — produces 0.4-0.5 cart_corr, catastrophic quality loss. The angular variation across the array is small in BSDF value but the BSDF is multiplied into a phasor that is summed coherently. Even tiny BSDF amplitude errors, when multiplied by the wrong phase, destroy coherent cancellation patterns in the RA image.

---

### Option 2: Per-TX evaluation with shared wo (N × 12 evals)

Evaluate BSDF for each (Gaussian, TX) pair with actual wi but mean-RX wo.

**Cost**: 12 evals per Gaussian (16× cheaper than per-path).

**Problem**: Same issue as monostatic for the wo dimension — the 16 RX elements produce different wo directions, and the BSDF amplitude variation across RX, while small, corrupts the coherent azimuth FFT.

**LIKELY INSUFFICIENT** for the same reason monostatic fails.

---

### Option 3: Decompose BSDF into wi-only and (wo,wi)-coupled terms

Rewrite the BSDF as:

```
f_total(wo, wi) = A(wi, n, materials) × B(wo, wi, n, materials)
```

where A captures everything that depends only on wi, and B captures the wo-dependent terms.

From the code:
```
A(wi) = η(wi) × τ_eff(wi)           -- wi-only blend factors
B_KA(wo, wi) = f_KA(wo, wi)          -- coupled
B_SPM(wo, wi) = f_SPM(wo, wi)        -- coupled
B_dir(wo, wi) = f_dir(wo, wi)        -- coupled  
B_broad(wo) = 1/π                     -- wo-only (trivially separable)
cbs(wo, wi) = 1 + sinc²(...)         -- coupled
```

The scattering lobes (KA, SPM, directive) are all functions of `cos_dev = wo · wi_r` where `wi_r = reflect(wi, n)`. For a fixed Gaussian (fixed n, fixed materials), `wi_r` depends only on wi. So `cos_dev = wo · wi_r(wi)` is a dot product between wo and a wi-dependent vector.

**Key**: for 16 RX elements, we can precompute `wi_r` once per (Gaussian, TX), then evaluate `cos_dev_r = wo_r · wi_r` for each RX as a simple dot product. The vMF/GGX evaluation using this `cos_dev` is cheap (~5 ops).

**Cost**: Per Gaussian: 12 × (full BSDF for wi-dependent parts + 16 × lightweight wo-evaluation)
= 12 full + 192 lightweight, where "lightweight" = a few dot products + exp.

**This is roughly 12 + 192×(1/10) ≈ 31 full-equivalent BSDF evals** per Gaussian, vs 192 currently. ~6× speedup on BSDF.

---

### Option 4: Fully factorized BSDF via precomputed specular frame

The most aggressive factorization. For each Gaussian with normal n:

**Step 1** (once per Gaussian, cost: materials only):
```
physics = reparameterize(raw_materials)
sh, lc = enforce_spm_validity(sigma_h, l_c)
η = coherent_blend(sh, lc, cos_theta_i_mean)     -- approximate with mean cos_theta_i
γ = sigmoid(..., sigma_h, wavelength)
cbs_mean_val = cbs_mean(lc)
kappa_SPM, kappa_dir = precompute_vmf_params(sh, lc)
alpha_KA = precompute_ggx_alpha(sh)
```

**Step 2** (per TX, 12 evaluations):
```
for each TX t:
    wi_t = normalize(TX_t - μ)
    cos_i_t = |wi_t · n|
    wi_r_t = 2(wi_t·n)n - wi_t              -- specular reflection
    retro_t = -wi_r_t                         -- CBS retroreflection
    
    # Jones Fresnel (wi-dependent core)
    s_in, p_in = sp_basis(wi_t, n)
    tx_s, tx_p = project_pol(tx_pol, s_in, p_in)
    r_s, r_p = slab_fresnel(eps_real, eps_imag, cos_i_t, thickness)
    E_s_out = r_s × tx_s                     -- complex
    E_p_out = r_p × tx_p                     -- complex
    
    # Blend factors
    η_t = coherent_blend(cos_i_t, ...)
    τ_eff_t = validity_blend(cos_i_t, ...)
    
    # Smith G1 for incidence (KA)
    G1_i_t = smith_g1(cos_i_t, alpha_KA)
    
    # Store per-TX precomputed state:
    TX_state[t] = {wi_r_t, retro_t, E_s_out, E_p_out, 
                   η_t, τ_eff_t, cos_i_t, G1_i_t, s_in}
```

**Step 3** (per RX, 16 evaluations using precomputed TX state):
```
for each RX r:
    wo_r = normalize(RX_r - μ)
    cos_o_r = |wo_r · n|
    G1_o_r = smith_g1(cos_o_r, alpha_KA)      -- Smith G1 for observation
    
    # Jones Fresnel (wo-dependent part: p_out projection)
    p_out_r = normalize(cross(s_in, wo_r))     -- s_in from any TX (same)
    rx_s_r = dot(rx_pol, s_in)                 -- same for all TX  
    rx_p_r = dot(rx_pol, p_out_r)              -- varies per RX
    
    Store RX_state[r] = {wo_r, cos_o_r, G1_o_r, rx_s_r, rx_p_r}
```

**Step 4** (per TX×RX pair, 192 LIGHTWEIGHT evaluations):
```
for each (t, r):
    # Half vector for KA
    h = normalize(wo_r + wi_t)
    h_dot_n = h · n
    D_KA = GGX_NDF(h_dot_n, alpha_KA)              -- 3 ops
    G_KA = 1 / (1 + (1/G1_i_t - 1) + (1/G1_o_r - 1))  -- ~5 ops (approx separable Smith G)
    f_KA = D_KA × G_KA / (4 × cos_i_t × cos_o_r)  -- 3 ops
    
    # SPM: cos_dev = wo · wi_r (precomputed wi_r from TX)
    cos_dev_SPM = wo_r · wi_r_t                     -- 1 dot = 3 ops
    f_SPM = vmf_eval(cos_dev_SPM, kappa_SPM) × eps_factor  -- ~5 ops
    
    # Directive: same structure as SPM
    cos_dev_dir = wo_r · wi_r_t                     -- same dot product
    f_dir = vmf_eval(cos_dev_dir, kappa_dir)        -- ~5 ops
    
    # Broad
    f_broad = 1/π  (if cos_i > 0 and cos_o > 0)    -- 0 ops
    
    # CBS
    cos_bs = wo_r · retro_t                          -- 1 dot = 3 ops
    cbs = 1 + sinc²(k × lc × sqrt(1-cos_bs²))      -- ~8 ops
    
    # Coherent + incoherent blend
    f_coh = τ_eff_t × f_KA + (1-τ_eff_t) × f_SPM   -- 3 ops
    f_inc = (γ × f_dir + (1-γ) × f_broad) × cbs / cbs_mean  -- 5 ops
    f_lobe = η_t × f_coh + (1-η_t) × f_inc          -- 3 ops
    
    # Jones Fresnel power (using precomputed E_s_out, E_p_out from TX, rx projections from RX)
    E_rx = E_s_out × rx_s_r + E_p_out × rx_p_r      -- 2 complex multiply + add
    R_jones = |E_rx|²                                 -- 2 ops
    
    f_total(t,r) = R_jones × f_lobe                   -- 1 op
    
    # TOTAL PER (t,r): ~45 ops (mostly scalar)
```

**Cost comparison for 192 paths:**

| | Full BSDF | Option 4 (factorized) |
|---|-----------|----------------------|
| Per Gaussian (step 1) | 0 | ~20 scalar ops |
| Per TX (step 2) | 0 | 12 × ~50 ops (Fresnel + basis) = 600 |
| Per RX (step 3) | 0 | 16 × ~15 ops = 240 |
| Per (TX,RX) pair (step 4) | 192 × ~300 ops = 57,600 | 192 × ~45 ops = 8,640 |
| **Total** | **~57,600 ops** | **~9,500 ops** |
| **Speedup** | 1× | **~6×** |

And critically: the per-(TX,RX) inner loop in Step 4 is all **scalar arithmetic** (dot products, exp, multiply) — no tensor allocation, no complex number construction, no normalize(). It can be vectorized as a single (192,) tensor operation.

**Accuracy**: This is **EXACT** — no approximation. Every term is computed precisely, just reorganized to avoid redundant computation.

---

### Option 5: Approximate separable Smith G (further speedup for Option 4)

The Smith masking-shadowing in KA uses `G = 1/(1 + Λ_i + Λ_o)` which is NOT a product. But the Heitz (2014) height-correlated form can be approximated:

```
G(cos_i, cos_o, α) ≈ G1(cos_i, α) × G1(cos_o, α)    (uncorrelated heights)
```

This is the standard "separable Smith G" used in real-time rendering. Error is <5% for typical roughness values. With this:

```
G_KA ≈ G1_i_t × G1_o_r    (precomputed in steps 2 and 3)
```

The per-(TX,RX) KA evaluation drops from ~11 ops to ~6 ops (no division by G1 terms).

---

### Option 6: Tabulated BSDF (precompute on angular grid)

For each Gaussian, precompute the BSDF on a (θ_i, θ_o, φ_o-φ_i) grid and interpolate. Since the BSDF depends on materials (6 params), this table would need to be per-Gaussian.

**Grid size**: 32 × 32 × 64 = 65,536 entries × 4 bytes = 256 KB per Gaussian. For 12K Gaussians: 3 GB. Too large.

**Alternative**: low-dimensional basis. The BSDF for a given material is smooth and well-approximated by a small number of basis functions. Could use spherical harmonics or Zernike polynomials on the (θ_i, θ_o, Δφ) domain.

**Cost**: ~10-20 coefficients per Gaussian, evaluated via polynomial in cos(θ_i), cos(θ_o), cos(Δφ).

**Pros**: Very fast evaluation (~20 ops per path). Smooth gradients for material optimization.
**Cons**: Requires fitting the basis (offline or differentiable). Approximation error depends on basis order. Complex to implement.

---

### Recommendation

**Option 4 (fully factorized, exact)** is the best balance:
- **6× speedup** over full per-path BSDF
- **Zero approximation error** — algebraic reorganization only
- Combined with phase factorization (7×) and no checkpoint (3×): total **~15× speedup**
- Can be further improved with Option 5 (separable Smith G) if needed

Option 1 (monostatic) is rejected — it destroys quality.
Option 6 (tabulated) is interesting for future work but too complex for now.

---

## Full factorized forward pass

### Per-Gaussian computation (M evaluations)

```python
# Inputs
mu = positions[active]                          # (M, 3)
normals = get_normals()[active]                  # (M, 3)
raw_materials = materials[active]                # (M, 6)
A = vertex_areas[active]                         # (M,)
alpha = opacities[active]                        # (M,)

# BSDF (monostatic, 1 eval per Gaussian)
radar_center = mean(tx_pos + rx_pos) / 2         # (3,)
wi = normalize(radar_center - mu)                # (M, 3)
cos_theta_i = |dot(wi, normals)|                 # (M,)
physics = reparameterize(raw_materials)           # (M, 6)
f_cos = bsdf_jones_f_cos(cos_theta_i, wi, wi, normals, *physics)  # (M,)

# Common amplitude factor
C_radar = radar_constant * rx_dBFS_scale * adc_scale
amp_common = C_radar * alpha * sqrt(f_cos.clamp(min=1e-20))  # (M,)
```

### Per-(Gaussian, TX) computation (M × 12)

```python
d_tx = ||mu[:, None, :] - tx_pos[None, :, :]||  # (M, 12)
dir_tx = (tx_pos - mu[:, None, :]) / d_tx[..., None]  # (M, 12, 3)

# TX antenna gain
G_tx = tx_antenna.evaluate(dir_tx.reshape(-1,3), ...).reshape(M, 12)  # (M, 12)

# TX amplitude and phase factors
alpha_tx = sqrt(G_tx) / d_tx.clamp(min=1e-4)     # (M, 12)
tau_tx = d_tx / C                                  # (M, 12)
phi_tx_const = 2π × f0 × tau_tx                    # (M, 12)
phi_tx_slope = 2π × S × tau_tx                     # (M, 12)
```

### Per-(Gaussian, RX) computation (M × 16)

```python
d_rx = ||mu[:, None, :] - rx_pos[None, :, :]||  # (M, 16)
dir_rx = (mu[:, None, :] - rx_pos) / d_rx[..., None]  # (M, 16, 3)

# RX antenna gain
G_rx = rx_antenna.evaluate(dir_rx.reshape(-1,3), ...).reshape(M, 16)  # (M, 16)

# RX-sphere solid angle
cos_theta_rx = |dot(normals[:, None, :], normalize(rx_pos - mu[:, None, :]))|  # (M, 16)
dOmega = A[:, None] * cos_theta_rx / d_rx.clamp(min=1e-4)²  # (M, 16)

# RX amplitude and phase factors
alpha_rx = sqrt(G_rx * dOmega)                     # (M, 16)
tau_rx = d_rx / C                                   # (M, 16)
phi_rx_const = 2π × f0 × tau_rx                     # (M, 16)
phi_rx_slope = 2π × S × tau_rx                      # (M, 16)
```

### ADC assembly via outer product

For each chunk of Gaussians (to control memory):

```python
t_k = arange(K) / fs                              # (K,)
ADC_real = zeros(N_tx, N_rx, K)
ADC_imag = zeros(N_tx, N_rx, K)

for m_start in range(0, M, chunk_size):
    m_end = min(m_start + chunk_size, M)
    c = slice(m_start, m_end)

    # TX phasor: (chunk, 12, K)
    phi_tx = phi_tx_const[c, :, None] + phi_tx_slope[c, :, None] * t_k
    tx_real = alpha_tx[c, :, None] * cos(phi_tx)
    tx_imag = alpha_tx[c, :, None] * sin(phi_tx)

    # RX phasor: (chunk, 16, K)
    phi_rx = phi_rx_const[c, :, None] + phi_rx_slope[c, :, None] * t_k
    rx_real = alpha_rx[c, :, None] * cos(phi_rx)
    rx_imag = alpha_rx[c, :, None] * sin(phi_rx)

    # Complex outer product: (a+jb)(c+jd) = (ac-bd) + j(ad+bc)
    w = amp_common[c]  # (chunk,)

    # ADC_real += Σ_m w[m] × (tx_real[m,t,k]×rx_real[m,r,k] - tx_imag[m,t,k]×rx_imag[m,r,k])
    # ADC_imag += Σ_m w[m] × (tx_real[m,t,k]×rx_imag[m,r,k] + tx_imag[m,t,k]×rx_real[m,r,k])
    ADC_real += einsum('mtk,mrk,m->trk', tx_real, rx_real, w) \
              - einsum('mtk,mrk,m->trk', tx_imag, rx_imag, w)
    ADC_imag += einsum('mtk,mrk,m->trk', tx_real, rx_imag, w) \
              + einsum('mtk,mrk,m->trk', tx_imag, rx_real, w)
```

### Memory per chunk (chunk_size = 2000)

| Tensor | Shape | Size |
|--------|-------|------|
| phi_tx, tx_real, tx_imag | (2K, 12, 256) | 3 × 24 MB = 72 MB |
| phi_rx, rx_real, rx_imag | (2K, 16, 256) | 3 × 32 MB = 96 MB |
| einsum intermediates | (12, 16, 256) | negligible |
| **Total per chunk** | | **~170 MB** |

Very safe for 24 GB GPU.

---

## Cost comparison

| Operation | Current (per-path) | Factorized | Speedup |
|-----------|-------------------|------------|---------|
| BSDF evals | M × 192 | M × 1 | **192×** |
| Antenna evals | M × 384 | M × 28 | **14×** |
| cos/sin (phase) | M × 192 × K | M × 28 × K | **7×** |
| Accumulation | scatter_add × 60 chunks | einsum × 6 chunks | **10×** |
| Checkpoint recompute | 3× everything | Not needed | **3×** |
| **Total estimated** | **6.7 s/iter** | **0.3-0.5 s/iter** | **15-20×** |

---

## Implementation steps

### Step 1: Write `render_gaussians_factorized()`

New function in `train_gaussian.py`. Implements the factorized forward pass. Uses real-valued phasors (cos/sin) to stay in float32.

### Step 2: Verify numerical equivalence

Run both renderers on 1 scene. Compare ADC output.
Target: `corr(factorized, per_path) > 0.99` (allowing ~1-3% monostatic error).

### Step 3: Plug into training loop

Drop-in replacement for `render_gaussians()`.

### Step 4: Benchmark

Compare per-iteration wall clock on seq_2_frame_105 (the slowest scene).

### Step 5: Re-run C3 and C4

All 7 scenes, 500 iterations. Report updated cart_corr table.

---

## What this does NOT change

- The physics (same BSDF, antenna, radar eq, RX-sphere projection)
- The loss function
- The optimizer
- Stage A/B results

## What this changes

- Per-iteration speed: 6.7s → ~0.4s (estimated)
- Total C3+C4 runtime: ~2 hours → ~15 min
- BSDF evaluation: per-path → monostatic (1 eval per Gaussian, <3% error)

---

## Experiments to re-run

C3 and C4 only, all 7 scenes each.

---

## Risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Monostatic BSDF error > expected | Low | Upgrade to Option 2 (per-TX) |
| einsum memory issue | Very low (170 MB/chunk) | Reduce chunk size |
| Quality regression | Low (same physics) | Compare 1 scene first |
| Gradient dynamics change | Medium | Same optimizer, but different graph structure |

---

## Phase 2 extensions (future)

- **Option 2 BSDF** (per-TX, 12× cost) if monostatic insufficient
- **Custom CUDA kernel** to fuse the factorized computation
- **Direct RA-space splatting** (skip ADC+FFT entirely)
- **Learned per-Gaussian importance** (Option J from gap closure plan)
