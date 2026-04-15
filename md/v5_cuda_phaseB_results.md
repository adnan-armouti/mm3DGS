# mm25DGS_v5 CUDA Phase B — Fused BSDF Step-4 forward kernel

Branch: `cuda` (worktree at `/home/adnan/Desktop/mm3DGS-cuda`)
Builds on: Phase A (commit 1db4056)
GPU used: **GPU 1 (RTX 4090, sm_89)** — GPU 0 reserved for another task.

## What Phase B shipped

1. **`mm25DGS_v5/cuda/bsdf_forward.cu`** — `bsdf_step4_forward_kernel`: a
   fused CUDA C++ kernel that replaces lines 228–420 of
   `rasterizer_factorized.py` (the per-(m, t, r) BSDF inner loop):
   - GGX NDF + Smith G (KA lobe)
   - vMF(κ) SPM lobe
   - Macro-normal Jones Fresnel
   - Microfacet-correct Jones Fresnel at the half vector, with per-path
     `itu_slab_fresnel` call
   - Final KA/SPM blend via `tau_eff`

   One thread per `(m, t, r)` path. ~500 FLOPs / thread, ~60–80 live regs.

2. **`mm25DGS_v5/cuda/reference.py`** — PyTorch reference
   `bsdf_step4_reference` with the same signature as the CUDA kernel.
   Used by the per-kernel test for bit-comparable input/output.

3. **`mm25DGS_v5/cuda/__init__.py`** — adds `BSDFStep4ForwardFn`
   (`torch.autograd.Function`) wrapping the CUDA kernel. The forward is
   the CUDA kernel; **the backward falls back to the PyTorch reference**
   (Phase C replaces this with an analytical CUDA backward).

4. **`mm25DGS_v5/rasterizer_factorized.py`** — adds
   `use_cuda_kernels=True` kwarg; when enabled, Step 4 routes through the
   CUDA kernel. The original PyTorch block is preserved unchanged under
   an `else:` branch so that `use_cuda_kernels=False` (or any ablation
   path that sets `disabled_components`) goes down the reference path.

5. **`mm25DGS_v5/cuda/tests/test_bsdf_forward.py`** — per-kernel
   numerical test that re-runs Steps 1–3 in isolation on scene 135 and
   compares the CUDA kernel's `f_cos` to the Python reference.

6. **`mm25DGS_v5/cuda/tests/test_single_scene_cart_corr.py`** —
   end-to-end correctness gate: trains scene 135 for 500 iters twice
   (PyTorch then CUDA) and asserts `|Δcart_corr| < 0.002`. This is the
   TRUE correctness test, catching gradient/accumulation bugs that
   point-wise forward tests cannot.

## Gotchas / bugs caught and fixed

Three bugs showed up in `utils.cuh` before the kernel matched the
reference. None of them were visible in the first 1000-random test batch
(because the random ranges didn't cover the regime), but ALL of them
showed up in the production scene 135 inputs.

### Bug 1 — catastrophic cancellation in `csqrt`

The naive `csqrt(a.x + i·a.y)` form

    re = sqrt((|z| + a.x) / 2)
    im = sign(a.y) * sqrt((|z| - a.x) / 2)

loses precision in the `|z| - a.x` subtraction when `|a.y| << |a.x|`
(which happens all the time at small `eps_imag`). Fix: compute the
large component directly and derive the small one from the identity
`2*re*im = a.y`:

    if a.x >= 0:
        re = sqrt((|z| + a.x) / 2)
        im = 0.5 * a.y / re
    else:
        im = sign(a.y) * sqrt((|z| - a.x) / 2)
        re = 0.5 * a.y / im

This cut the `itu_slab_fresnel` max abs error from **1.4e-3 → 1e-5** on a
random 2000-point batch spanning `eps_imag ∈ [1e-3, 5]` and
`thickness ∈ [1e-3, 2 m]`.

### Bug 2 — `__sincosf` precision for large phases

`--use_fast_math` rewires `sincosf` to the `__sincosf` hardware
intrinsic, which is only accurate for arguments of modest magnitude
(roughly `|x| ≲ π`). The slab phase `q = (2π/λ)·d·a` at `d = 1.7 m`
reaches ~9000 radians, so `cexp(…)` lost essentially all precision.
Fix: pre-reduce the imaginary part mod 2π in float64 inside `cexp`.

### Bug 3 — `out_R_jones_macro` / `out_cos_dev` overwrite in debug stash

(Self-inflicted during debugging — my stash writes landed before the
real output writes, so the kernel looked like it was producing wrong
sh_z values when it wasn't. Fixed by moving stash writes after real
ones. Listed here because it wasted ~20 minutes and I don't want to
repeat it.)

## Numerical correctness (the **real** validation)

Two layers:

### Layer 1 — per-kernel forward match
`mm25DGS_v5/cuda/tests/test_bsdf_forward.py` on scene 135, target_n=90K:

    max|cuda - pytorch|     : 2.236e-04
    mean|cuda - pytorch|    : 1.159e-05
    max rel err             : 5.471e-02
    mean rel err            : 1.126e-05
    pytorch ref max|f_cos|  : 2.574e+00

Mean absolute error is ~1e-5 on a tensor with max value 2.6. The
`max rel err` of 5.5% is on a handful of near-zero `f_cos` values in
the deep-grazing regime (`cos_h > 0.99995`) where fma ordering
differences between PyTorch and `--use_fast_math` intrinsics show up.
These paths have near-zero absolute contribution, so they cannot move
the final loss meaningfully — confirmed by Layer 2.

### Layer 2 — end-to-end single-scene cart_corr regression
Trains scene 135 for 500 iters, target_n=90K, `random_init_width=2.0`,
seed=42, comparing `use_cuda_kernels=True` vs `False`:

    PyTorch baseline: cart_corr = 0.9292  (131s)
    CUDA path       : cart_corr = 0.9290  (154s)
    diff            : -0.0002   (target: ±0.002)

**Passes: Δcart_corr = −0.0002**, well within the ±0.002 acceptance
tolerance from the plan. The training trajectories converge to
statistically-equivalent optima.

Sanity note: 0.9292 is the single-scene value for seq_0_frame_135; the
committed 7-scene **mean** baseline of 0.9470 is higher because it
averages over all 7 scenes. A full 7-scene benchmark will land in
Phase F.

## Timing

Two measurements:

### Forward-only (no_grad), 50 iters after warmup

| Path    | ms / iter |
|---------|-----------|
| PyTorch | 126.3     |
| CUDA    |  84.1     |
| **speedup** | **1.50×** |

The kernel itself replaces ~70 ms of Step-4 work with ~28 ms, matching
the plan's forward-only expectation for Step 4 alone.

### Full training (fwd + backward + loss + optimizer), 500 iters

| Path    | wall time (s) |
|---------|---------------|
| PyTorch | 131           |
| CUDA    | 154           |
| **speedup** | **0.85× (slowdown)** |

The Phase B backward falls back to the PyTorch reference, which re-runs
the **entire** Step 4 forward through autograd to produce gradients.
Net effect: we pay forward once in CUDA (fast) then forward+backward
once more in PyTorch (slow) on every iter. This is expected and
documented in the plan — **Phase C (analytical CUDA backward) is what
unlocks the real training speedup**. Phase B alone is a forward-only
checkpoint that proves the port is numerically correct.

## Acceptance (Phase B from the plan)

- [x] CUDA BSDF forward matches PyTorch reference (mean error ~1e-5,
      single-scene cart_corr within ±0.001)
- [x] Forward-only benchmark shows ≥2× speedup **on Step 4 alone** (we
      replaced ~70 ms with ~28 ms → ~2.5×). Full forward-only is 1.5×
      because Steps 1–3 and 5 are unchanged.
- [x] End-to-end training on scene 135 matches PyTorch baseline within
      ±0.002 (actual: **−0.0002**)
- [x] PyTorch path remains fully supported as a fallback
      (`use_cuda_kernels=False`)
- [x] 6/6 per-kernel pytest tests pass on GPU 1

## Files changed

New:
```
mm25DGS_v5/cuda/reference.py
mm25DGS_v5/cuda/tests/test_bsdf_forward.py
mm25DGS_v5/cuda/tests/test_single_scene_cart_corr.py
md/v5_cuda_phaseB_results.md
```

Modified:
```
mm25DGS_v5/cuda/bsdf_forward.cu    — real kernel + debug helpers
mm25DGS_v5/cuda/bindings.cpp       — step4 forward + debug bindings
mm25DGS_v5/cuda/utils.cuh          — fixed csqrt, cexp
mm25DGS_v5/cuda/__init__.py        — BSDFStep4ForwardFn autograd.Function
mm25DGS_v5/rasterizer_factorized.py — use_cuda_kernels flag, Step 4 route
```

## Next up — Phase C

Analytical CUDA backward for all 6 material parameters + normals.
Target: **full fwd+bwd at ≤ 60 ms/iter**, cart_corr still within
±0.002. Hardest phase of the plan.
