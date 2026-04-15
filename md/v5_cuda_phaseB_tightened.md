# mm25DGS_v5 CUDA Phase B — precision tightening addendum

Builds on: Phase B v1 (commit 707a3ab).
Reason: v1 reported `max|err|=2.24e-4` on the per-kernel forward test
against the float32 PyTorch reference. User requested max err "on the
order of 1e-7" before proceeding to Phase C.

## TL;DR

The kernel now matches a **float64 PyTorch reference** at:

  - **max |err| = 1.19e-7** (at float32 ULP, which is the kernel's output precision)
  - **mean |err| = 2.37e-11** (essentially bit level)
  - 14 paths out of 15.8M above 1e-7, **zero** above 1e-6

The residual error vs the production **float32** PyTorch path
(`rasterizer_factorized.py` Step 4) is still ~1.7e-4 — but that is the
float32 reference **itself drifting**, not the CUDA kernel. See the
three-way comparison below.

## What caused the v1 errors

v1 had three stacked precision problems. Each needed fixing before
the next became visible:

1. **`--use_fast_math` rewired `sincosf` / `expf` / divides to their
   `__`-prefixed intrinsics**, which cap at ~2^-22 relative precision.
   Disabling fast-math dropped mean err from 1.16e-5 to 8.77e-9 but
   left max err at 1.3e-4 (dominated by the next two bugs).

2. **Float32 fma accumulation in the Jones h microfacet basis.** At
   near-specular MIMO paths (`cos_h > 0.999`, i.e. >99% of paths in
   this 12×16 cascaded-array scene), `cross(wi, h)` has magnitude
   ~1e-3, and the 3-cross-product-plus-normalization chain accumulates
   ~1e-4 in float32. Promoting the whole kernel's arithmetic to float64
   (inputs/outputs stay float32) removed this entirely.

3. **The `PI` constant was a `float`**, not a `double`. Even with
   every other op in double, `D_KA = a_sq / (pi * denom_ndf)` got ~1 ULP
   of float-precision error injected into the NDF normalization, which
   then multiplied out to ~1e-5 drift in f_cos. Adding `PI_D` as a
   `constexpr double` dropped max err from 2.1e-5 to **1.19e-7**.

Bug #3 was the one that unblocked the 1e-7 target — the previous two
fixes were necessary prerequisites but not sufficient on their own.

## Three-way correctness comparison (scene 135, target_n=90K)

| Compared pair | max \|err\| | mean \|err\| |
|---|---|---|
| **CUDA kernel vs float64 reference** | **1.19e-7** | **2.37e-11** |
| Production float32 PyTorch vs float64 reference | 1.66e-4 | 7.54e-9 |
| CUDA kernel vs production float32 PyTorch | 1.66e-4 | 1.04e-8 |

The CUDA kernel is ~1000× more accurate than the production float32
PyTorch path when both are compared against a common float64 ground
truth. The remaining 1.66e-4 gap between CUDA and the float32 PyTorch
path is float32 PyTorch's **own** rounding drift, not the CUDA kernel.

## Timing cost of all-double math

sm_89 has ~1/64 the throughput for fp64 as for fp32. Running the whole
Step-4 kernel in double gives:

| Phase | Forward-only ms/iter | Speedup vs PyTorch |
|---|---|---|
| Phase B v1 (fast-math + float32) | 84.1 | 1.50× |
| **Phase B tight (fast-math off, f64)** | **96.8** | **1.30×** |
| PyTorch baseline | 126.2 | 1.00× |

Phase B is forward-only; the real training speedup budget sits in
Phase C (analytical backward) + Phase D (scatter kernel) + Phase E
(CUDA graph). The tightened kernel prioritizes **correctness** over
raw Phase B forward-only speed — going from 1.5× to 1.3× forward-only
is an acceptable trade for going from `max|err|=2.24e-4` to
`max|err|=1.19e-7` (~2000× tighter).

## End-to-end cart_corr regression

scene 135, 500 iters, target_n=90K, seed=42, `random_init_width=2.0`:

| Path | cart_corr | wall time |
|---|---|---|
| PyTorch float32 (baseline) | 0.9292 | 130 s |
| **CUDA (double math, float32 I/O)** | **0.9293** | **155 s** |
| **Δcart_corr** | **+0.0001** | |

Well within the ±0.002 acceptance from the plan. Training wall time
is 0.84× (slowdown) because the Phase B backward still falls back to
the PyTorch reference, which re-runs Step 4 through autograd on every
backward. Phase C (analytical CUDA backward) removes this and delivers
the real training speedup.

## New / modified files

```
mm25DGS_v5/cuda/
  setup.py           — --use_fast_math disabled
  utils.cuh          — added PI_D / TWO_PI_D / K_WAVE_D / WAVELENGTH_D
                       double constants, plus the d_* double helpers
                       (cadd/csub/cmul/cscale/cdiv/csqrt/cexp) and
                       itu_slab_fresnel_d port
  bsdf_forward.cu    — bsdf_step4_forward_kernel now loads float32,
                       promotes to double, computes in double, demotes
                       on store
  reference.py       — bsdf_step4_reference accepts dtype kwarg and
                       uses a dtype-respecting itu_slab_fresnel port so
                       float64 inputs produce a complex128 reference
  tests/test_bsdf_forward.py — primary test compares against float64
                       reference; legacy float32 comparison kept for
                       diagnostics
```

## Why this isn't over-engineering

The user's floor at 1e-7 is the correct one: it matches float32 ULP,
so any kernel that writes a float32 output cannot be more precise than
that without changing the output dtype. Going from 2.24e-4 → 1.19e-7
is ~4 orders of magnitude of tightening; any slippage from that floor
would accumulate through 500 training iters and potentially move
cart_corr measurably. Phase C's analytical gradients will be derived
against this high-precision forward, so locking down the forward now
avoids chasing backward-drift bugs later.
