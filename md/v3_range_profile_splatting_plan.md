# mm25DGS_v3: Range-Profile Splatting

## Overview

Replace ADC-domain rendering (v2) with direct splatting into range profiles. Each Gaussian deposits its complex amplitude at 1-5 range bins per (TX, RX) channel, eliminating the K=256 phase loop that dominates v2's runtime. The azimuth FFT operates on the range profiles exactly as in the standard radar pipeline — no approximation.

## Architecture

```
v2 (current):  Gaussians → BSDF → [cos/sin × K=256 per path] → ADC → range FFT → azimuth FFT → RA
v3 (proposed): Gaussians → BSDF → [scatter to ~3 range bins per path] → Range Profile → azimuth FFT → RA
                                    ↑ 50× less work                       ↑ skip range FFT
```

## The rendering equation

### Physical formulation

The range profile at channel (t, r) is the DFT of the ADC signal:

```
RP[t, r, n] = Σ_k ADC[t,r,k] × W[k] × exp(-j 2π n k / K)
```

where W[k] is the range FFT window (Hann, Blackman, etc.).

Substituting the ADC signal from a Gaussian at position μ:

```
ADC[t,r,k] = w × cos(φ_const + φ_slope × k/fs)
```

where `φ_const = 2π f₀ τ`, `φ_slope = 2π S τ`, `τ = (d_tx + d_rx)/c`.

The DFT of this sinusoid is a peak at frequency index:

```
n_peak = φ_slope × K / (2π × fs) = S × τ × K / fs = (d_tx + d_rx) / (2 × range_res)
```

where `range_res = c × fs / (2 × S × K) = c / (2 × BW)`.

### What each Gaussian contributes to the range profile

For Gaussian m at channel (t, r):

```
RP[t, r, n] += w_m(t,r) × exp(j × φ_carrier_m(t,r)) × PSF(n - n_peak_m(t,r))
```

where:
- `w_m(t,r) = C_radar × sqrt(f_cos_m(t,r) × G_tx × G_rx × dΩ / d_tx²)` — same amplitude as v2
- `φ_carrier_m(t,r) = 2π × f₀ × τ_m(t,r)` — carrier phase (encodes azimuth information)
- `n_peak_m(t,r) = (d_tx_m(t) + d_rx_m(r)) / (2 × range_res)` — range bin (real-valued, not integer)
- `PSF(δ) = Σ_k W[k] × exp(-j 2π δ k / K)` — the range point spread function from the window

### PSF handling (sub-bin interpolation)

For a rectangular window (no windowing): `PSF(δ) = sin(πδ) / sin(πδ/K) × exp(-jπδ(K-1)/K)` (Dirichlet kernel). At integer δ=0, this is K. At δ=±1, it's 0. The main lobe width is ~2 bins.

For a Hann window: main lobe width is ~4 bins, sidelobes are -31 dB.

**Practical approach**: deposit to the nearest `W_spread` bins (e.g., W_spread=5) using precomputed PSF weights. For each Gaussian:

```python
n_center = floor(n_peak)
for dn in range(-W_spread//2, W_spread//2 + 1):
    n = n_center + dn
    if 0 <= n < K:
        psf_weight = PSF(n - n_peak)   # complex, from precomputed kernel or analytic formula
        RP[t, r, n] += w × exp(j × φ_carrier) × psf_weight
```

**Differentiability**: `n_peak` depends on `d_tx + d_rx` which depends on Gaussian position μ. The PSF weights are smooth functions of `n_peak` (they're values of a sinc/window function at fractional offsets). Gradients of PSF weights w.r.t. n_peak (and hence μ) are well-defined.

### What the azimuth FFT sees

The range profile `RP[t, r, n]` is a complex-valued tensor of shape `(n_tx, n_rx, K)`. The azimuth FFT operates on the virtual array dimension (across TX-RX pairs) at each range bin:

```
RA[n_range, n_az] = Σ_v RP[virtual_element_v, n_range] × exp(-j 2π n_az v / N_az)
```

This is **exactly** how the standard radar processing pipeline works. The phase `φ_carrier = 2π f₀ τ` varies across TX-RX elements due to the different path lengths, encoding the angular information. The azimuth FFT extracts this angle — no approximation, no PSF modeling for azimuth.

## Implementation plan

### Directory structure

```
mm25DGS_v3/
    __init__.py
    rasterizer.py          # Range-profile splatting renderer
    train.py               # Training loop (reuses v2 components where possible)
    psf.py                 # Range PSF computation (analytic + precomputed)
    config.py              # Configuration (or reuse v2/mm25DGS config)
    test_equivalence.py    # Verify v3 matches v2 ADC→RA pipeline
```

### Step 1: Range PSF module (`psf.py`)

Precompute the range PSF for the radar's FFT windowing.

```python
class RangePSF:
    def __init__(self, K, window='hann', spread=5):
        """Precompute the range PSF kernel.
        
        The PSF is the DFT of the window function evaluated at fractional offsets.
        For a Hann window of length K:
            PSF(δ) = Σ_k W[k] × exp(-j 2π δ k / K)
        
        We precompute PSF(δ) on a fine grid δ ∈ [-spread/2, spread/2] and
        interpolate during rendering.
        """
        self.K = K
        self.spread = spread
        self.window = get_window(window, K)   # (K,) tensor
        # Precompute PSF on fine grid for fast lookup
        # Or: use analytic formula for Hann window PSF
    
    def evaluate(self, delta):
        """Evaluate PSF at fractional offset delta.
        
        Args:
            delta: (*,) fractional range bin offsets
        Returns:
            (*,) complex PSF values
        """
        # Analytic for rectangular: dirichlet_kernel(delta, K)
        # Analytic for Hann: sum of 3 shifted dirichlet kernels
        # Or: interpolate from precomputed table
```

The PSF evaluation must be differentiable w.r.t. delta (for position gradients).

**Key detail**: the PSF must match exactly what `adc_to_ra_image` computes. The function `adc_to_ra_image` (from `mmir/data/ra_utils.py`) applies a specific window and FFT. The PSF must be the DFT of that same window.

### Step 2: Range-profile renderer (`rasterizer.py`)

Core rendering function that splats Gaussians into range profiles.

```python
def render_range_profiles(
    positions,       # (M, 3)
    normals,         # (M, 3)
    areas,           # (M,)
    raw_materials,   # (M, 6)
    rast,            # RasterizerTorch (for sensor config + antenna patterns)
    reparameterize_fn,
    psf,             # RangePSF instance
    detach_phase=True,
):
    """Splat Gaussians into complex range profiles.
    
    Returns:
        rp_real: (n_tx, n_rx, K) real part of range profile
        rp_imag: (n_tx, n_rx, K) imag part of range profile
    """
    # Steps 1-4: BSDF computation (identical to rasterizer_factorized.py)
    # Produces: w_full (M, n_tx, n_rx), phi_carrier (M, n_tx, n_rx), n_peak (M, n_tx, n_rx)
    
    # Step 5: Range-bin splatting
    # For each Gaussian m and channel (t, r):
    #   n_peak = (d_tx + d_rx) / (2 × range_res)
    #   For dn in [-spread//2, spread//2]:
    #     RP[t, r, n_peak + dn] += w × exp(j × φ_carrier) × PSF(dn - frac(n_peak))
```

The splatting can be implemented as:

```python
# n_peak: (M, n_tx, n_rx) — real-valued range bins
# w_full: (M, n_tx, n_rx) — amplitude weights
# phi_carrier: (M, n_tx, n_rx) — carrier phases

n_floor = n_peak.floor().long()        # integer part
n_frac = n_peak - n_floor.float()      # fractional part [0, 1)

rp_real = zeros(n_tx, n_rx, K)
rp_imag = zeros(n_tx, n_rx, K)

for dn in range(-spread//2, spread//2 + 1):
    n_bin = n_floor + dn                                    # (M, n_tx, n_rx)
    valid = (n_bin >= 0) & (n_bin < K)
    
    psf_val = psf.evaluate(dn - n_frac)                     # (M, n_tx, n_rx) complex
    contrib = w_full * psf_val                               # complex amplitude
    carrier = exp(j × phi_carrier)                           # complex carrier
    total = contrib * carrier                                # (M, n_tx, n_rx) complex
    
    # Scatter-add to range profile bins
    # Flatten (n_tx, n_rx, n_bin) → flat index, scatter_add
    flat_idx = t_idx * (n_rx * K) + r_idx * K + n_bin
    rp_real.view(-1).scatter_add_(0, flat_idx[valid].view(-1), total.real[valid].view(-1))
    rp_imag.view(-1).scatter_add_(0, flat_idx[valid].view(-1), total.imag[valid].view(-1))
```

**Cost**: M × 192 × spread = M × 192 × 5 = 960M scatter ops (vs 49,152M cos/sin for v2).

### Step 3: RA image computation

After splatting, the range profiles are combined into the RA image via azimuth FFT:

```python
# rp: (n_tx, n_rx, K) complex range profiles
# Apply azimuth FFT exactly as in adc_to_ra_image, but skip the range FFT
# (range FFT is already done by splatting)

# The standard pipeline: adc_to_ra_image does:
#   1. Range FFT (along k dimension) with windowing
#   2. Virtual array rearrangement
#   3. Azimuth FFT (along virtual array dimension)
#   4. Take magnitude

# v3: skip step 1 (range FFT), feed range profiles directly into steps 2-4.
# Need to extract the azimuth FFT portion from adc_to_ra_image.
```

**Important**: we need to either:
- (a) Refactor `adc_to_ra_image` to accept pre-computed range profiles, or
- (b) Write a new `range_profile_to_ra_image` function that does steps 2-4

Option (b) is safer — it avoids modifying the shared mmIR utility.

### Step 4: Loss function

Same as v2: RA magnitude MSE with min-max normalization, linear. The loss is computed on the RA image, which is now produced by:

```
Gaussians → range_profile_splat → azimuth_FFT → RA → loss
```

instead of:

```
Gaussians → ADC_synthesis → range_FFT → azimuth_FFT → RA → loss
```

Gradients flow back through: loss → RA → azimuth FFT (differentiable) → range profile → scatter (differentiable) → Gaussian parameters.

### Step 5: Training loop (`train.py`)

Largely reuse v2's `train_gaussian.py`:
- Same `GaussianSurfels` model
- Same initialization (`init_from_mesh`, `init_from_lidar`)
- Same optimizer (per-group Adam)
- Same culling
- Same learnable antenna patterns (Option E)
- Replace `render_gaussians_factorized` with `render_range_profiles`
- Replace `compute_ra_loss(adc_real, adc_imag, gt_adc_ri)` with `compute_ra_loss_from_rp(rp, gt_rp)` or convert both to RA and compare

### Step 6: Verification (`test_equivalence.py`)

Verify that v3's range profile → azimuth FFT → RA produces the same RA image as v2's ADC → range FFT → azimuth FFT → RA.

Test on 1 scene with mmIR materials:
- Render ADC with v2 factorized renderer → range FFT → RA (reference)
- Render range profile with v3 → azimuth FFT → RA
- Compare: `corr(RA_v3, RA_v2) > 0.999`

The only difference should be the PSF approximation (spreading to W_spread bins instead of exact DFT). With W_spread=5 and Hann window, the truncated sidelobes are at -60 dB — negligible.

## Key implementation details

### Matching `adc_to_ra_image` exactly

The critical constraint: v3's range profile output must produce the same RA image as v2's ADC when passed through the azimuth FFT. This means:

1. **Read `adc_to_ra_image`** to understand exactly what window function, zero-padding, and normalization it applies during the range FFT
2. **The PSF must be the DFT of that exact window** — not an approximation
3. **Zero-padding**: if `adc_to_ra_image` zero-pads the ADC before the range FFT (e.g., K=256 → 512), the range profile has 512 bins, and the PSF is the DFT of the zero-padded windowed signal
4. **Normalization**: the FFT normalization (1/K or 1/√K) must be baked into the PSF

### Handling the (M, n_tx, n_rx) BSDF tensor

The BSDF computation from `rasterizer_factorized.py` / `rasterizer_nufft.py` produces `w_full` of shape `(M, n_tx, n_rx)`. This is 12K × 12 × 16 = 2.3M floats = 9 MB. This fits easily in GPU memory.

The range bin `n_peak` is also `(M, n_tx, n_rx)` because `d_tx(m,t) + d_rx(m,r)` varies per channel.

The splatting loop (over W_spread=5 offsets) touches `M × 192 × 5` elements — each is a scatter_add to one range bin. This is 11.5M scatter ops. On GPU, scatter_add runs at ~10 Gops/s, so ~1ms.

### Memory budget

| Tensor | Shape | Size |
|--------|-------|------|
| w_full | (M, n_tx, n_rx) | 9 MB |
| phi_carrier | (M, n_tx, n_rx) | 9 MB |
| n_peak | (M, n_tx, n_rx) | 9 MB |
| rp_real, rp_imag | (n_tx, n_rx, K) | 0.2 MB |
| PSF weights (per spread offset) | (M, n_tx, n_rx) | 9 MB × 5 = 45 MB |
| **Total** | | **~80 MB** |

Extremely comfortable on a 24 GB GPU.

### Gradient flow

All operations are differentiable:
- `n_peak = (d_tx + d_rx) / (2 × range_res)` — differentiable w.r.t. Gaussian position
- `PSF(dn - frac(n_peak))` — differentiable w.r.t. n_peak (PSF is a smooth function)
- `scatter_add` — differentiable (PyTorch supports autograd through scatter_add)
- `azimuth FFT` — differentiable (torch.fft)
- `carrier phase exp(j × φ)` — differentiable (when not detached)

Position gradients: moving a Gaussian changes its range bin `n_peak`, which shifts the PSF weights. The gradient flows through the PSF evaluation to the position. This gives the optimizer range-direction position information — something v2 couldn't provide (v2 detaches phase).

## Expected performance

### Forward pass

| Component | v2 (factorized) | v3 (range splat) |
|-----------|-----------------|------------------|
| BSDF (steps 1-4) | 3 ms | 3 ms (same) |
| Phase accumulation | 56 ms (einsum) | **~1 ms** (scatter) |
| Range FFT | 0 (done in loss) | 0 (skipped) |
| Azimuth FFT | ~1 ms (in loss) | ~1 ms (same) |
| **Total forward** | **59 ms** | **~5 ms** |
| **Speedup** | 1× | **~12×** |

### Training iteration (forward + backward)

| | v2 (factorized) | v3 (range splat) |
|---|---|---|
| Forward | 59 ms | ~5 ms |
| Backward | ~100 ms | ~10 ms (much smaller graph) |
| Optimizer step | ~2 ms | ~2 ms |
| **Total per iter** | **~160 ms** | **~17 ms** |
| **500 iters** | **80 s** | **~9 s** |

### Full C3 + C4 experiment (7 scenes each)

| | v2 | v3 |
|---|---|---|
| C3 (7 scenes) | 7 min | **~1 min** |
| C4 (7 scenes) | 30 min | **~3 min** |
| Total | 37 min | **~4 min** |

## What v3 reuses from v2

- `GaussianSurfels` model (train_gaussian.py)
- `init_from_mesh`, `init_from_lidar` initialization
- `reparameterize_torch` material mapping (rasterizer_torch.py)
- `AntennaPatternTorch` antenna evaluation (rasterizer_torch.py)
- `RasterizerTorch` for sensor config loading + reservoir sampler (culling)
- `_compute_bsdf_and_phase` factorized BSDF (rasterizer_nufft.py)
- Loss function structure (train_gaussian.py / train_mesh.py)
- `cull_gaussians`, `_get_visible_vertices` culling utilities
- Per-group Adam optimizer setup
- `run_c3_c4_parallel.py` parallel runner

## What v3 adds

- `psf.py`: Range PSF computation matching `adc_to_ra_image`
- `rasterizer.py`: Range-profile splatting (replaces ADC synthesis)
- `train.py`: Training loop using range-profile rendering
- `test_equivalence.py`: Verification against v2

## Experiments to run

1. **Verification**: v3 RA matches v2 RA on 1 scene (corr > 0.999)
2. **C3**: all 7 scenes, 500 iters (compare cart_corr with v2 C3)
3. **C4**: all 7 scenes, 500 iters (compare cart_corr with v2 C4)
4. **Timing**: per-iteration wall clock comparison v2 vs v3

## Risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| PSF truncation (W_spread=5) introduces artifacts | Very low (-60 dB sidelobes) | Increase spread to 7 or 9 |
| `adc_to_ra_image` window doesn't match PSF | Medium (need to read the code) | Read and match exactly |
| Scatter_add contention (many Gaussians same bin) | Low (atomic add is fast) | Profile, use float64 if needed |
| Position gradients through PSF are noisy | Low (PSF is smooth) | Verify on 1 scene |
| Quality regression vs v2 | Very low (same physics) | Verify before full run |
