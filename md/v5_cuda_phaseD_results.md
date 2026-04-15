# mm25DGS_v5 CUDA Phase D — fused Step-5 range-profile splat

Builds on: Phase C (commit 5eb6c2e).
Plan reference: `md/v4_cuda_kernel_plan.md` Phase D.

## What Phase D shipped

A fused CUDA Step-5 kernel (`step5_fused_forward_kernel` + matching
backward in `mm25DGS_v5/cuda/scatter_splat.cu`) that replaces the
entire Step 5 pipeline — not just the scatter_add. The kernel takes
`(w_full, phi_carrier, n_peak, psf_real, psf_imag)` and produces
`(rp_real, rp_imag)` in one launch, computing everything in registers:

  1. sin/cos of phi (carrier phasor)
  2. PSF table linear interpolation at `n_frac`
  3. complex multiply `contrib = carrier · psf`
  4. `(n_floor + offset) mod K` bin index
  5. atomic scatter into `rp_real`/`rp_imag`

All `SPREAD=15` offsets are handled in an unrolled inner loop per
thread, so no intermediate `(SPREAD, P)` tensors are materialized.

### Why a drop-in scatter_splat didn't help

My first attempt (Phase D.1) was a drop-in replacement of just the
`scatter_add_` calls, wrapped as an `autograd.Function` with a gather
backward. **It saved 0 ms** — PyTorch's native `scatter_add_` is
already well-optimized and the scatter itself is not the bottleneck.

The real cost in Step 5 is the **540 MB of intermediate contrib
tensors** (`contrib_real_all`, `contrib_imag_all`, `flat_idx_all`
each shape `(SPREAD=15, P_active≈3M)`) that get allocated and freed
every iteration. These cause:
  - Huge memory traffic (write 540 MB, read 540 MB)
  - CUDA allocator overhead
  - L2 cache thrashing

Fusing everything into one kernel that keeps contribs in registers is
what unlocks the win.

### Backward design

The fused backward is simpler than the forward: only `grad_w_full`
needs to be computed because `phi_carrier` and `n_peak` are always
detached in the production path (`detach_phase=True`; `n_peak` is
explicitly detached). This means:
  - **No atomics in the backward** — each thread owns its output slot
    `grad_w_full[m, t, r]`
  - **Simpler chain rule** — `∂contrib_re/∂w = c·psf_r − s·psf_i`
    summed over all SPREAD offsets, where c = cos(phi), s = sin(phi)
  - The PSF tables are treated as non-differentiable constants

## Timing

### Forward-only (scene 135, target_n=90K, no_grad, 50 iters)

| Path | ms/iter | Speedup |
|---|---|---|
| PyTorch | 126.3 | 1.00× |
| CUDA (Phase A–C: fused BSDF, PyTorch Step 5) | 84.0 | 1.50× |
| **CUDA (Phase D: fused BSDF + fused Step 5)** | **11.9** | **10.57×** |

### Forward+backward (scene 135, 20 iters with grad)

| Path | ms/iter | Speedup |
|---|---|---|
| PyTorch | 209.1 | 1.00× |
| CUDA (Phase C) | 162.4 | 1.29× |
| **CUDA (Phase D)** | **63.0** | **3.32×** |

### Full training (scene 135, 500 iters, seed=42, random_init_width=2.0)

| Path | cart_corr | wall time | ms/iter | Speedup |
|---|---|---|---|---|
| PyTorch baseline | 0.9292 | 130 s | ~260 | 1.00× |
| CUDA Phase C | 0.9295 | 97 s | ~194 | 1.34× |
| **CUDA Phase D** | **0.9293** | **56 s** | **~112** | **2.34×** |
| **Δcart_corr** | **+0.0001** | | | |

## Correctness

### Per-kernel

- All 9 pytest tests pass (6 existing Phase A-C + 3 for fused Step 5)
- The 14.5% max relative error on individual `rp` bins is due to
  atomic-order non-determinism at bins where many contributions
  near-cancel (different fp accumulation order between my kernel's
  atomicAdd and PyTorch's scatter_add). Mean relative error on `|rp|`
  is **0.0017%**, and the end-to-end cart_corr matches to **+0.0001**.

### End-to-end

- cart_corr diff matches the Phase C level (+0.0001 vs +0.0002),
  confirming the fused kernel doesn't introduce new training drift

## Remaining gap vs plan targets

| Phase | Plan target ms/iter | Actual ms/iter | Delta |
|---|---|---|---|
| PyTorch baseline | 260 | 260 | — |
| Phase B (fwd CUDA) | ~200 | 260→298 | Phase B alone was net slowdown due to PyTorch bwd fallback |
| Phase C (fwd+bwd CUDA) | ~100 | **194** | 94 ms over target — Phase C Step-4 kernel needs optimization |
| Phase D (+ fused scatter) | ~75 | **112** | 37 ms over target — still have Phase C's overhead in the chain |
| Phase E (CUDA graph) | ~65 | TBD | — |

**Phase D was actually a bigger win than the plan anticipated** (~82 ms
saved vs plan's ~25 ms). The plan assumed Step 5 ran at ~45+25=70 ms
and Phase D would drop it to ~20 ms; in reality the fused kernel
dropped the whole Step 5 forward+backward segment from ~100 ms to
~18 ms.

The remaining 37 ms gap vs the 75 ms target is mostly in Phase C's
backward kernel (currently ~51 ms with the fused Step 5, should be
~30-40 ms with the warp-shuffle per-m reduction and slab Fresnel
analytical derivatives we discussed). Phase E + Phase C revisit are
the next big wins.

## Files changed

Modified:
```
mm25DGS_v5/cuda/scatter_splat.cu   — added step5_fused_{forward,backward}_kernel
mm25DGS_v5/cuda/bindings.cpp       — step5_fused_{forward,backward} Python entries
mm25DGS_v5/cuda/__init__.py        — Step5FusedFn autograd.Function +
                                     ScatterSplatFn (unused drop-in)
mm25DGS_v5/rasterizer_factorized.py — Step 5 routes through CUDA when enabled
```

## Next up

1. **Phase E (CUDA Graph capture)**: absorbs Python overhead (kernel
   launches, tensor allocs, Adam state updates). Expected savings:
   ~15-25 ms (targeting ~90 ms/iter).
2. **Phase C revisit**: warp-shuffle per-m reduction + kernel split +
   analytical slab Fresnel + Steps 1-3 CUDA port. Expected savings:
   ~15-30 ms (targeting ~75 ms/iter).
