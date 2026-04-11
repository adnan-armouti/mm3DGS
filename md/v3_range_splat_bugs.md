# v3 Range Splat: Bug Analysis and Fixes

## Summary

The Hann PSF is correct and compact (SPREAD=5 captures 99.98% of a single Gaussian's energy). The 10% error at 12K Gaussians comes from two bugs in the splatting code.

## Bug 1 (DOMINANT): Missing modular wrapping of range bins

**The problem**: `n_peak` can exceed K=256 (max observed: 287.6). The DFT is periodic with period K, so bin 260 should wrap to bin 4. But the current code clips:

```python
valid = (n_bin >= 0) & (n_bin < K) & (w_flat > 1e-20)
```

This DISCARDS 28,394 out of 2,304,000 paths (1.2%) whose n_peak > 253. Their energy vanishes. More importantly, their sidelobes at bins 0-5 (which contribute to other bins via SPREAD) are also lost.

**The fix**: Replace clipping with modular wrapping:

```python
n_bin = (n_floor_flat + dn) % K    # wrap around [0, K)
valid = w_flat > 1e-20              # no range clipping needed
```

This is correct because the DFT output `F[n]` for `n = 0..K-1` is periodic: `F[n+K] = F[n]`. A Gaussian at n_peak=260.3 wraps to an effective n_peak of 4.3, with its PSF sidelobes correctly distributed around bins 2-7.

**Impact**: Restores the 1.2% of discarded paths plus their sidelobe contributions to neighboring bins. Expected to recover most of the 15% missing energy.

## Bug 2: Sidelobe accumulation from many Gaussians

**The problem**: With 12K Gaussians spread across ~100 range bins, each bin n receives sidelobe contributions from ~12K Gaussians beyond the SPREAD window. At SPREAD=5, each truncated sidelobe at offset δ>2 has magnitude ~1/δ² × peak. Summed over 12K Gaussians:

```
Sidelobe energy at bin n ≈ Σ_{m: |peak_m - n| > 2} |PSF(peak_m - n)|²
```

For the Hann window, sidelobes decay as ~1/δ². With 12K Gaussians uniformly distributed over ~100 bins, the average sidelobe contribution at any bin is:

```
~12K × Σ_{δ=3}^{128} 1/δ⁴ ≈ 12K × 0.01 ≈ 120 energy units
```

This is significant relative to the direct contribution at that bin.

**The fix**: Increase SPREAD. The energy capture vs SPREAD (measured for a single Gaussian):

| SPREAD | Energy captured | Error |
|--------|----------------|-------|
| 3 | 99.50% | 0.50% |
| 5 | 99.98% | 0.02% |
| 7 | 99.996% | 0.004% |
| 9 | 99.999% | 0.001% |
| 11 | 99.9996% | 0.0004% |

For 12K Gaussians, multiply the per-Gaussian error by ~12K to estimate accumulated sidelobe error. SPREAD=9 gives per-Gaussian error 0.001%, × 12K = 12% accumulated. SPREAD=15 gives 0.0001% × 12K = 1.2%.

**Recommended**: SPREAD=21 or higher. At SPREAD=21, the per-Gaussian truncation is < 0.00001%, and even 12K× accumulation gives < 0.1% total error.

**Speed impact**: SPREAD=21 means 21 scatter ops per path instead of 5. Total: 12K × 192 × 21 = 48M scatter ops. This is still 12× less than the 589M cos/sin in the ADC approach.

## Changes required to rasterizer_factorized.py

### Change 1: Modular bin wrapping (line 398-399)

Before:
```python
n_bin = n_floor_flat + dn
valid = (n_bin >= 0) & (n_bin < K) & (w_flat > 1e-20)
```

After:
```python
n_bin = (n_floor_flat + dn) % K    # periodic wrapping
valid = w_flat > 1e-20              # no range clip
```

### Change 2: Increase SPREAD (line 376)

Before:
```python
SPREAD = 5
```

After:
```python
SPREAD = 21
```

### Change 3: Remove the safety clamp on flat_idx (line 411)

Before:
```python
flat_idx = t_idx_flat * (n_rx * K) + r_idx_flat * K + n_bin
flat_idx = flat_idx.clamp(0, n_tx * n_rx * K - 1)  # safety clamp
```

After:
```python
flat_idx = t_idx_flat * (n_rx * K) + r_idx_flat * K + n_bin
# No clamp needed — n_bin is in [0, K) after modular wrapping
```

## Expected results after fixes

With Bug 1 fixed (wrapping) + SPREAD=21:
- Energy ratio: >0.999 (vs current 0.849)
- Per-channel correlation: >0.999 (vs current 0.892)
- Phase error: <0.5° mean (vs current 5.3°)
- Speed: ~4× less work than ADC approach (21/256 ratio × scatter vs trig overhead)
