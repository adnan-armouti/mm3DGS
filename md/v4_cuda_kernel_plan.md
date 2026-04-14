# mm25DGS_v4 CUDA kernel speedup plan

Goal: reduce per-iter training time from ~260 ms at `target_n=90K` on an RTX 4090 to ≤100 ms via custom hand-written CUDA C++ kernels wrapped in a PyTorch extension, without changing training semantics or cart_corr beyond ±0.002 of the committed baseline (0.9470).

## Current baseline (per-iter time, target_n=90K, 4090)

Measured 500 iters / 130 s / scene = **260 ms/iter**. Approximate breakdown of where time goes:

| Stage | Time @ 90K | % of iter | Workload |
|---|---|---|---|
| **Step 4 BSDF inner loop** (forward) | 70 ms | 27% | `(M, 12, 16)` tensors — GGX, SPM, microfacet Jones, per-path Fresnel |
| **Step 4 backward** (autograd chain through BSDF) | 80 ms | 31% | Dense backward traversal of ~40 intermediate tensors |
| **Step 5 splat + scatter_add** | 45 ms | 17% | 8.6M paths × 5 spread = 48M atomicAdds into `(12, 16, 256)` buffer |
| **Step 5 backward** (gather) | 25 ms | 10% | Already a custom `SplatScatterFn` (gather for scatter backward) |
| Step 1–3 geometry + antenna + macro Jones | 15 ms | 6% | Per-TX, per-RX reductions |
| Range-profile FFT + magnitude + polar→cart | 5 ms | 2% | O(1) in M — fixed cost |
| Optimizer step + grad clip | 10 ms | 4% | Adam update on all parameters |
| Loss computation + backward init | 5 ms | 2% | MSE path |
| Python / CUDA launch overhead | 5 ms | 2% | ~50 kernel launches per iter |
| **Total** | **260 ms** | **100%** | |

**The hot targets are Step 4 (forward + backward = 150 ms, 58%) and Step 5 (forward + backward = 70 ms, 27%).** Together they are 85% of per-iter time. A fused set of custom CUDA kernels that eliminates intermediate tensor allocation, keeps state in registers, and uses contention-aware scatter can drop these ~70% — taking per-iter from 260 ms → ~100 ms, a 2.6× speedup. With CUDA Graph capture on top, ~4×.

## Why PyTorch is slow here

1. **Intermediate tensor allocation**. The BSDF inner loop creates ~40 intermediate tensors during forward. PyTorch allocates each from the CUDA allocator, and autograd holds all of them until backward. Each allocation + later deallocation is ~10–50 μs. At 40 allocs × 2 (forward + backward), that's ~4 ms just in allocator overhead.

2. **Memory bandwidth, not FLOPs**. The BSDF is memory-bound: typical ops touch 50–200 bytes per element but do ~20 FLOPs per element. On the 4090 (1008 GB/s memory bandwidth, 83 TFLOPs compute), this is ~50× off the compute peak. A fused kernel that keeps intermediates in registers and streams inputs once through the cache gets closer to the memory peak.

3. **Kernel launch overhead**. ~50 kernel launches per iter × ~10 μs each = 500 μs. Small but real, and fused kernels eliminate most of them.

4. **Atomic scatter contention**. `scatter_add_` into a shared output buffer has atomic contention on popular bins. For a typical RA image, ~5% of bins capture 50% of the contributions, so atomic serialization bites. A sorted-scatter kernel that groups paths by target bin can reduce contention dramatically.

## Approach: CUDA C++ extension via `torch.utils.cpp_extension`

Write the hot kernels in raw CUDA C++, compile as a loadable PyTorch extension, wrap as `torch.autograd.Function` so they drop into the existing training loop. This is the standard path for high-performance custom kernels in PyTorch (used by xformers, flash-attention, gsplat, nerfacc, and similar libraries).

**Build system**: `torch.utils.cpp_extension.load()` or a `setup.py` build. Both compile CUDA sources to a shared library at runtime (`.load()`) or build time (`setup.py`). `.load()` is simpler for iteration; `setup.py` is cleaner for shipping.

**Data path**: CUDA kernels receive PyTorch tensor data pointers via `tensor.data_ptr<float>()`. Zero-copy, same CUDA stream, same memory allocator.

**Autograd integration**: subclass `torch.autograd.Function` in Python. `forward()` calls the forward CUDA kernel via the loaded extension; `backward()` calls the backward CUDA kernel. The existing Python autograd engine calls these through the normal graph traversal.

**Reference kernels we're learning from**: gsplat's diff-gaussian-rasterization (splat+backward in pure CUDA), NVIDIA's CUB for sort/reduce primitives, PyTorch's own ATen CUDA implementations.

### Kernels to write

1. **`fused_bsdf_forward_kernel.cu`** — the biggest single win
   - Inputs: `positions (M, 3)`, `normals (M, 3)`, `raw_materials (M, 6)`, TX/RX geometry, antenna patterns (already fp32, uploaded once), material reparam constants
   - Outputs: `f_cos (M, n_tx, n_rx)`, `n_peak (M, n_tx, n_rx)`, `phi_carrier (M, n_tx, n_rx)`, `alpha_tx (M, n_tx)`, `alpha_rx (M, n_rx)`
   - Fuses: material reparam, per-TX geometry, Fresnel at half-vector, f_KA, f_SPM, microfacet Jones, cos_i weighting
   - All intermediates kept in registers (estimated 60–80 registers per thread, well within 255 limit)
   - Grid: 1D over `(M * n_tx * n_rx)`, block size 256
   - Eliminates ~30 intermediate tensor allocations

2. **`fused_bsdf_backward_kernel.cu`** — paired with forward
   - Inputs: `grad_f_cos`, forward inputs, saved forward outputs
   - Outputs: `grad_raw_materials (M, 6)`, `grad_normals (M, 3)` (via atomic accumulation if needed)
   - Recomputes forward intermediates on the fly from inputs (trades compute for memory; saves ~30 float32 tensors of `(M, 12, 16)` each = ~500 MB at 90K)
   - Gradient chain rule applied manually per material parameter
   - ~500 FLOPs per thread, ~10 reads + ~3 atomic writes per thread

3. **`scatter_splat_kernel.cu`** — replaces the `SplatScatterFn` forward
   - Inputs: `contrib_real (S, M*n_tx*n_rx)`, `contrib_imag (S, M*n_tx*n_rx)`, `flat_idx (S, M*n_tx*n_rx)` where S=5 spread
   - Outputs: `rp_real (n_tx, n_rx, K)`, `rp_imag (n_tx, n_rx, K)` via `atomicAdd`
   - Grid: 1D over `(S * M * n_tx * n_rx)`, each thread does one atomic add per output
   - Optional: sorted-path variant (Phase D+) for reduced contention

4. **`scatter_splat_backward_kernel.cu`** — custom gather for scatter backward
   - Trivial to write (gather with no contention). Currently implemented in the existing `SplatScatterFn` via `index_select`, which is already efficient. **Skip unless profiling shows it's a meaningful bottleneck.**

5. **Optional: `slab_fresnel_kernel.cu`** — replaces the per-path `itu_slab_fresnel` call
   - Inputs: `eps_real, eps_imag, cos_theta, thickness` per path
   - Outputs: `r_s, r_p` per path (complex)
   - Only worth writing if the Python `itu_slab_fresnel` shows up as a bottleneck after the fused BSDF kernel absorbs most of Step 4.

## Optional add-ons

### Sorted scatter (Phase D+)

**What**: Before scatter_add, sort paths by target bin using CUB `DeviceRadixSort::SortPairs`. Within a sorted chunk, all threads writing to the same bin accumulate into shared memory first, then one thread does a single `atomicAdd` per unique bin.

**Why**: Reduces atomic contention from ~50 GB/s effective to ~500 GB/s (close to sequential memory throughput) for the hot bins.

**Cost**: Sorting adds ~5 ms overhead but saves ~20 ms on the scatter itself = ~15 ms net win.

**Complexity**: Moderate. CUB's radix sort is well-documented. Add as an optional pass after Phase D.

### CUDA Graph capture (Phase E)

**What**: After the first iter, capture the entire sequence of kernel launches into a CUDA Graph and replay it on subsequent iters. Eliminates Python overhead, kernel launch overhead, and allows the CUDA driver to optimize across calls.

**Caveats**:
- Tensor shapes must be static (they are — `M` is fixed at `target_n` after init)
- Any Python-level conditionals that change iter-to-iter break capture (we have none in the steady-state loop)
- Best-state tracking (the `if cart_corr > best_corr` branch) is compatible because it's a CPU-side comparison after the GPU computation completes

**Effort**: ~1 day to wrap the training step in `torch.cuda.graph()` context.

**Expected speedup**: ~10–20% on top of the fused kernels.

### Full end-to-end CUDA (not recommended)

Moving the entire training step (forward render + loss + backward + optimizer update) into one monolithic CUDA extension. The plan's build infrastructure supports this, but the extra effort (~1 month) vs the marginal additional speedup over Phase B+C+D+E (~5× total vs ~4×) is not worth it. The targeted kernels get most of the benefit with much less code.

## Recommended execution plan (phased)

### Phase A — Infrastructure + baseline (2–3 days)

1. **Set up CUDA extension directory structure**:
   ```
   mm25DGS_v4/cuda/
     setup.py                  # torch.utils.cpp_extension.CUDAExtension
     bindings.cpp              # pybind11 wrapper, minimal — one function per kernel
     bsdf_forward.cu           # fused BSDF forward kernel
     bsdf_backward.cu          # fused BSDF backward kernel
     scatter_splat.cu          # scatter_add replacement
     sorted_scatter.cu         # (Phase D+) sorted variant
     utils.cuh                 # shared device-side helpers (complex math, Fresnel, etc.)
     reference.py              # PyTorch reference implementations (for tests only)
     tests/
       test_bsdf.py            # Forward + backward numerical equivalence tests
       test_scatter.py
       test_gradcheck.py       # PyTorch gradcheck on the full path
     __init__.py               # Exports the autograd.Function wrappers
   ```

2. **Write the build config**. `setup.py` using `torch.utils.cpp_extension.CUDAExtension` with:
   - `-O3 -use_fast_math` for CUDA
   - `-std=c++17` for host code
   - Compute capability `sm_89` (Ada Lovelace, 4090)
   - Include paths for CUB (bundled with CUDA toolkit)

3. **Write the pybind11 bindings**. `bindings.cpp` exports one function per kernel, each taking PyTorch tensors as arguments. Use `TORCH_CHECK(tensor.is_cuda())` and `tensor.is_contiguous()` guards.

4. **Write the `__init__.py` with `torch.autograd.Function` wrappers**. Each wrapper calls the forward CUDA kernel in `.forward()`, saves required inputs for backward, and calls the backward CUDA kernel in `.backward()`.

5. **Verify build**. Run `python setup.py build_ext --inplace` (or the `.load()` equivalent) and import the extension. Should produce a `.so` file and import without error.

6. **Instrument per-stage timing** in `render_factorized` with `torch.cuda.synchronize()` + timers. Record baseline stage-by-stage times at `target_n=90K` to `mm25DGS_v4/output/baseline_stage_times.json`.

7. **Capture one forward pass** of all intermediate tensors (via hooks) on scene 135. Save them as a reference dump for later numerical validation: `mm25DGS_v4/output/kernel_validation/reference_forward.npz`.

### Phase B — Fused BSDF forward kernel (1.5 weeks)

1. **Write `fused_bsdf_forward_kernel` in CUDA C++**. Start from the existing PyTorch code in `rasterizer_factorized.py` Step 4. Port to device code:
   - Material reparam (softplus, exp clamps, sigmoids) as `__device__ float reparam_eps_real(float raw)` etc.
   - GGX NDF, Smith G, f_KA using standard float math
   - SPM vMF lobe using `expf`, `sinhf`
   - Microfacet Jones: cross products, normalizations, dot products — all scalar float ops
   - Per-path slab Fresnel as a `__device__ __inline__` function (port the existing `itu_slab_fresnel`)
   - All intermediates in `float` or `float2` (for complex) registers
   - One thread per `(m, t, r)` path
   - Block size 256, grid sized to cover `M * n_tx * n_rx`

2. **Set up numerical validation harness**:
   - Single-path test: input one `(m=0, t=0, r=0)` path with hand-computed reference
   - Full-scene test: run on scene 135, compare `f_cos`, `n_peak`, `phi_carrier` against the PyTorch reference saved in Phase A, with `rtol=1e-5, atol=1e-6`
   - If mismatches exceed tolerance, debug math before proceeding

3. **Wrap as `torch.autograd.Function`**. `forward()` calls the new CUDA kernel; `backward()` calls the existing PyTorch reference implementation as a temporary measure (full CUDA backward is Phase C). This lets us validate the forward kernel without needing the backward to work yet.

4. **Swap into `render_factorized`**. Add a flag `use_cuda_kernels=True` that routes through the CUDA extension if available (try to import; fall back to PyTorch on ImportError). Run the 7-scene benchmark with `use_cuda_kernels=True` and `use_cuda_kernels=False` to verify cart_corr is unchanged within ±0.001.

5. **Benchmark**: time the forward pass alone (exclude backward) at target_n=90K on scene 135. Expected: **70 ms → ~20 ms (3.5× forward-only speedup)**.

### Phase C — Fused BSDF backward kernel (1.5 weeks)

1. **Write `fused_bsdf_backward_kernel` in CUDA C++**. This is the hard part. For each material parameter, derive the chain rule analytically and implement on the device:
   - `∂f_cos/∂eps_real` through the slab Fresnel → `∂L/∂eps_real += grad_f_cos × <chain>`
   - Similarly for `eps_imag`, `sigma_h`, `l_c`, `tau_base`, `thickness`
   - Use `atomicAdd` to accumulate gradients across the `(n_tx, n_rx)` dimensions since multiple threads contribute to the same `raw_materials[m, k]`
   - For `normals`, similar — accumulate across TX and RX directions
   - Recompute forward intermediates on the fly (don't save them; recomputation is cheaper than memory traffic)

2. **Validate against PyTorch autograd**:
   - For each material parameter column, compute the gradient via both the CUDA backward and PyTorch autograd
   - Compare with `rtol=1e-4, atol=1e-5` (looser than forward because accumulated rounding)
   - Repeat for normals
   - Fix any mismatches before proceeding

3. **Swap into the autograd Function** as the real backward.

4. **Benchmark**: time the full forward + backward at target_n=90K on scene 135. Expected: **150 ms → ~50 ms (3× forward+backward speedup)**.

5. **End-to-end validation**: run the 7-scene benchmark and confirm mean cart_corr is within ±0.001 of the PyTorch reference (0.9470 ± 0.001).

### Phase D — Scatter splat kernel + sorted scatter (4 days)

1. **Write `scatter_splat_kernel.cu`** — basic version. One thread per `(s, m, t, r)` quadruple, each does one `atomicAdd` to `rp_real` and one to `rp_imag` at the target bin.

2. **Validate**: numerical equivalence to the existing `SplatScatterFn.apply`.

3. **Benchmark without sort**: expected ~5–10 ms savings over the current scatter_add (PyTorch's scatter_add is already pretty well-tuned).

4. **Sorted scatter (optional but recommended)**:
   - Use CUB's `cub::DeviceRadixSort::SortPairs` to sort paths by `flat_idx`
   - After sorting, process sorted chunks — threads in the same warp write to adjacent bins, so atomic contention is reduced
   - Alternative: use a segmented reduce within each bin after sorting
   - Expected additional ~10–15 ms savings

5. **Combined expected speedup from scatter**: 45 ms → ~20 ms (2.25×) or better with sort.

### Phase E — CUDA Graph capture (1–2 days)

1. **Wrap the training step in `torch.cuda.graph(g)`**. After the first 5 warm-up iters (to let Adam initialize state and the CUDA allocator stabilize), capture iters 5–10 as a CUDA graph.

2. **Replay the graph** for iters 10–499. Use `g.replay()` each iter.

3. **Handle the metric path**. The per-iter cart_corr computation reads from GPU tensors via `.item()` — this is a synchronization point that's compatible with graph replay.

4. **Handle the best-state tracking**. The `if cart_corr > best_corr` branch is a CPU-side operation that runs between GPU replays and doesn't affect the captured graph.

5. **Benchmark**: expected 10–20% reduction on top of the CUDA kernels, i.e. ~80 ms/iter → ~65 ms/iter.

### Phase F — Final validation + benchmarks (3 days)

1. **Numerical equivalence test**: for each kernel, compare against the PyTorch reference at tight tolerances. Write as unit tests in `cuda/tests/`.

2. **Gradient correctness test**: for a small synthetic scene (100 points), compute `torch.autograd.gradcheck()` on the whole training path using both CUDA and PyTorch. All gradients match to 1e-4 relative.

3. **End-to-end cart_corr test**: run the 7-scene benchmark with the CUDA kernels enabled. Confirm mean ∈ [0.9450, 0.9490] (within ±0.002 of the committed baseline 0.9470 — looser than ±0.001 to allow for float rounding differences between `fmadd` / `__fmaf_rn` / `fma` variants).

4. **Regression test**: run the LOO ablation on scene 135 and verify the parameter rankings match the PyTorch baseline (within ±0.005 per run). This catches off-by-one errors and subtle gradient bugs.

5. **Final per-iter timing** at target_n=90K, expected cumulative:

| Phase | Per iter | 7 scenes |
|---|---|---|
| Current PyTorch | 260 ms | 15 min |
| Phase B (CUDA forward) | ~200 ms | ~12 min |
| Phase C (CUDA forward+backward) | ~100 ms | ~6 min |
| Phase D (CUDA scatter + sort) | ~75 ms | ~5 min |
| Phase E (CUDA graph) | **~65 ms** | **~4 min** |

**Total expected speedup: 4× (260 → 65 ms/iter). 7-scene benchmark: 15 min → ~4 min.**

## Files that will change

### New files

```
mm25DGS_v4/cuda/
  setup.py                    # torch.utils.cpp_extension CUDAExtension
  bindings.cpp                # pybind11 wrapper
  bsdf_forward.cu             # fused forward kernel
  bsdf_backward.cu            # fused backward kernel
  scatter_splat.cu            # scatter+splat
  sorted_scatter.cu           # optional sorted variant
  utils.cuh                   # device-side helpers
  reference.py                # PyTorch references for tests
  __init__.py                 # autograd.Function wrappers + ext loader
  tests/
    __init__.py
    test_bsdf.py              # numerical equivalence + gradcheck
    test_scatter.py
    test_gradcheck.py
```

### Modified files

```
mm25DGS_v4/rasterizer_factorized.py
  - Add `use_cuda_kernels: bool = True` arg to render_factorized
  - Route Step 4 through the CUDA BSDF kernels if available
  - Route Step 5 through the CUDA scatter kernel if available
  - Fall back to existing PyTorch code if the CUDA extension isn't built

mm25DGS_v4/train_gaussian.py
  - Optionally wrap the training step in torch.cuda.graph() (Phase E)
  - Default use_cuda_kernels=True in the call to render_factorized
```

### Unchanged files

- Everything else stays exactly as is. The CUDA path is a drop-in replacement; the PyTorch path remains fully supported as a fallback.

## Risk register

| Risk | Mitigation |
|---|---|
| **Kernel register spilling** (too many live values) | Check `nvcc --ptxas-options=-v` output after compile; limit block size; split kernel into two passes if needed |
| **Numerical divergence** from PyTorch reference | Unit tests in Phase A validate each kernel at `rtol=1e-5, atol=1e-6` forward and `1e-4, 1e-5` backward; end-to-end cart_corr test in Phase F bounds drift to ±0.002 |
| **Gradient bug** in custom backward | `torch.autograd.gradcheck()` on a 100-point synthetic scene in Phase F; LOO ablation smoke test to catch off-by-one errors |
| **Non-deterministic training results** from sorted scatter atomic ordering | Sort with a stable ordering key; verify two consecutive runs give identical cart_corr to 1e-6 |
| **CUDA graph capture fails** due to dynamic shapes | Fix `target_n` before capture; fall back to non-graph execution if capture fails |
| **Compiler version mismatch** (nvcc vs torch) | Document the required CUDA + PyTorch versions; fail fast in setup.py if they're incompatible |
| **Backward kernel is much harder than expected** | Can defer: Phase B can ship with PyTorch backward as a temporary measure. Phase C becomes the "stretch" goal. |
| **Sorted scatter doesn't pay off** | It's optional. Skip if profiling shows contention isn't actually the bottleneck — move straight to Phase E (CUDA graph). |
| **Performance below target** (not reaching 4×) | Profile with NVIDIA Nsight Compute; identify per-kernel bottlenecks; optimize in priority order (memory access patterns → register pressure → warp divergence → instruction-level parallelism) |

## Acceptance criteria

**Phase B ships when**:
- CUDA BSDF forward produces identical output to PyTorch reference (`rtol=1e-5, atol=1e-6`) on a test scene
- Forward-only benchmark shows ≥2× speedup (70 ms → ≤35 ms)
- End-to-end training on scene 135 produces cart_corr within ±0.001 of PyTorch baseline (forward is used, backward still PyTorch)

**Phase C ships when**:
- CUDA BSDF backward produces matching gradients (`rtol=1e-4, atol=1e-5`) via `torch.autograd.gradcheck()` on a 100-point test case
- Full forward+backward benchmark shows ≥2× speedup (150 ms → ≤75 ms)
- 7-scene benchmark mean cart_corr within ±0.002 of 0.9470

**Phase D ships when**:
- Scatter kernel shows ≥1.5× speedup and identical output to `SplatScatterFn`

**Phase E ships when**:
- CUDA graph capture succeeds on iter 5, replay works for iters 10–499
- Final per-iter timing at target_n=90K is ≤100 ms
- 7-scene benchmark mean cart_corr within ±0.002 of 0.9470

**Entire plan ships when**:
- Per-iter time at target_n=90K ≤100 ms (target: ≤65 ms)
- 7-scene benchmark time ≤8 min (target: ≤5 min)
- cart_corr mean within ±0.002 of the PyTorch baseline (0.9470)
- All unit tests pass
- `gradcheck()` passes for the full training path
- LOO ablation on scene 135 matches the PyTorch-baseline parameter rankings within ±0.005

## Out of scope

- **Mixed precision (fp16/bf16)**: defers to a follow-up. Adds another 1.5–2× on memory-bound kernels but has numerical correctness risk (exp/sqrt underflow in BSDF).
- **Multi-GPU**: out of scope.
- **FP16 atomic scatter**: may give 2× on the scatter stage but requires careful accumulator management to avoid precision loss.
- **Tensor cores / wmma**: the BSDF is memory-bound, not compute-bound, so tensor cores don't help unless we change the op structure.
- **Rewriting the init pipeline** (FPS, visibility, FOV): init is one-shot and fast (~2 s). Not worth CUDA porting.
- **Rewriting antenna pattern evaluation**: the current PyTorch `AntennaPatternTorch.evaluate` is <1 ms per iter, negligible.
- **Full end-to-end CUDA training**: covered as "not recommended" above. Targeted kernels get most of the speedup with much less code.

## Estimated timeline

| Phase | Work | Elapsed |
|---|---|---|
| A | Infrastructure, build system, bindings, baseline dumps | 2–3 days |
| B | Fused BSDF forward kernel | 1.5 weeks |
| C | Fused BSDF backward kernel | 1.5 weeks |
| D | Scatter splat kernel + optional sorted scatter | 4 days |
| E | CUDA Graph capture | 1–2 days |
| F | Final validation + benchmarks | 3 days |
| **Total** | | **~5 weeks** of focused effort |

Expected final speedup: **4× total (260 → 65 ms/iter)**. 7-scene benchmark from ~15 min to ~4 min. Without CUDA graph capture, ~3.5× and ~4.5 min.

## References

- **`torch.utils.cpp_extension` documentation**: https://pytorch.org/tutorials/advanced/cpp_extension.html
- **PyTorch CUDA extension example**: https://github.com/pytorch/extension-cpp
- **CUB DeviceRadixSort**: for sorted scatter
- **gsplat's diff-gaussian-rasterization** (reference for a similar rasterizer's CUDA+PyTorch integration): https://github.com/nerfstudio-project/gsplat
- **NVIDIA Nsight Compute**: for per-kernel profiling
- **Walter et al. 2007** "Microfacet Models for Refraction through Rough Surfaces" — for verifying the BSDF math in the kernel matches our Python implementation
- **Current PyTorch implementation**: `mm25DGS_v4/rasterizer_factorized.py` (commit `ef492a3` for the full microfacet Jones, `3c653e6` for target_n=90K)
