# mm25DGS_v4 CUDA kernel speedup plan

Goal: reduce the per-iter training time from ~260 ms at `target_n=90K` on an RTX 4090 to <100 ms via custom CUDA/Triton kernels, without changing training semantics or cart_corr.

## Current baseline (per-iter time, target_n=90K, 4090)

Measured 500 iters / 130 s / scene = **260 ms/iter**. Approximate breakdown of where time goes (from the performance profiling discussion):

| Stage | Time @ 90K | % of iter | Workload |
|---|---|---|---|
| **Step 4 BSDF inner loop** (forward) | 70 ms | 27% | `(M, 12, 16)` tensors — GGX, SPM, microfacet Jones, per-path Fresnel |
| **Step 4 backward** (autograd chain through BSDF) | 80 ms | 31% | Dense backward traversal of ~40 intermediate tensors |
| **Step 5 splat + scatter_add** | 45 ms | 17% | 8.6M paths × 5 spread = 48M atomicAdds into `(12, 16, 256)` buffer |
| **Step 5 backward** (gather) | 25 ms | 10% | Already custom `SplatScatterFn` (gather for scatter backward) |
| Step 1–3 geometry + antenna + macro Jones | 15 ms | 6% | Per-TX, per-RX reductions |
| Range-profile FFT + magnitude + polar→cart | 5 ms | 2% | O(1) in M — fixed cost |
| Optimizer step + grad clip | 10 ms | 4% | Adam update on all parameters |
| Loss computation + backward init | 5 ms | 2% | MSE path |
| Python / CUDA launch overhead | 5 ms | 2% | ~50 kernel launches per iter |
| **Total** | **260 ms** | **100%** | |

**The hot targets are Step 4 (forward + backward = 150 ms, 58%) and Step 5 (forward + backward = 70 ms, 27%).** Together they are 85% of per-iter time. If we eliminate 70% of their cost with fused CUDA kernels, we drop from 260 ms → ~100 ms per iter, a 2.5× speedup.

## Why PyTorch is slow here

1. **Intermediate tensor allocation**. The BSDF inner loop creates ~40 intermediate tensors during forward. PyTorch allocates each from the CUDA allocator, and autograd holds all of them until backward. Each allocation + later deallocation is ~10–50 μs. At 40 allocs × 2 (forward+backward), that's ~4 ms just in allocator overhead.

2. **Memory bandwidth, not FLOPs**. The BSDF is memory-bound: typical ops touch 50–200 bytes per element but do ~20 FLOPs per element. On the 4090 (1008 GB/s bandwidth, 83 TFLOPs), this is ~50× off the compute peak. A fused kernel that keeps intermediates in registers and streams inputs once through the cache gets closer to the memory peak.

3. **Small kernel launch overhead**. ~50 kernel launches per iter × ~10 μs overhead = 500 μs. Small but real, and fused kernels eliminate most of them.

4. **Atomic scatter contention**. `scatter_add_` into a shared output buffer has atomic contention on popular bins. For a typical RA image, ~5% of bins capture 50% of the contributions, so atomic serialization bites. A sorted-scatter kernel that groups paths by target bin can reduce contention.

## Kernel implementation options (from easiest to most impactful)

### Option 1 — Triton kernels (recommended first)

**What**: Write the hot kernels in OpenAI's Triton DSL. Triton is a Python-embedded language that compiles to efficient CUDA PTX. It integrates natively with PyTorch tensors (zero-copy) and supports autograd via `torch.autograd.Function`. Typical Triton kernels reach 85–95% of hand-written CUDA performance with ~10× less code.

**Why first**: Minimum friction. No separate build system, no C++ glue code, no MSVC toolchain issues. One `.py` file per kernel. Incremental — can port one kernel at a time and keep the rest in PyTorch.

**Key kernels to write**:

1. **`fused_bsdf_forward_kernel`** — the biggest single win
   - Inputs: `positions (M, 3)`, `normals (M, 3)`, `raw_materials (M, 6)`, TX/RX geometry, antenna patterns, constants
   - Outputs: `f_cos (M, n_tx, n_rx)`, `n_peak (M, n_tx, n_rx)`, `phi_carrier (M, n_tx, n_rx)`
   - Fuses: material reparam, per-TX geometry, Fresnel at half-vector, f_KA, f_SPM, microfacet Jones, cos_i weighting
   - All intermediates kept in registers
   - Grid: 1D over `(M * n_tx * n_rx)`, block size 256
   - Eliminates ~30 intermediate tensor allocations

2. **`fused_bsdf_backward_kernel`** — paired with forward
   - Inputs: `grad_f_cos`, forward inputs, saved forward outputs
   - Outputs: `grad_raw_materials`, `grad_normals`, (other inputs if needed)
   - Recomputes forward intermediates on the fly from inputs (trades compute for memory; saves ~30 float32 tensors of `(M, 12, 16)` each = ~500 MB at 90K)
   - Gradient chain rule applied manually

3. **`scatter_add_splat_kernel`** — replaces the `SplatScatterFn` wrapped scatter
   - Inputs: `contrib_real`, `contrib_imag`, `flat_idx`, all of shape `(S, M*n_tx*n_rx)` where S=5 spread
   - Outputs: `rp_real`, `rp_imag` of shape `(n_tx, n_rx, K)`
   - Uses `tl.atomic_add` in Triton (warp-level atomics)
   - Option: sort paths by target bin first for reduced contention (see Option 3 below)

4. **`scatter_add_splat_backward_kernel`** — custom gather for scatter backward
   - Already done at the PyTorch level (`SplatScatterFn`) — porting to Triton gives only marginal speedup. **Skip unless profiling shows it's a bottleneck.**

5. **`fused_loss_kernel`** — RA magnitude + polar→cart + MSE (optional, very small gain)
   - This is already O(1) in M, so the expected gain is small (~2 ms). Lowest priority.

**Effort**: ~1–2 weeks for all 5 kernels, including testing and autograd integration. The biggest time sink is kernel correctness validation against PyTorch reference.

**Expected speedup**: Per iter from 260 ms → ~120 ms (2.2×). 7-scene benchmark from ~15 min → ~7 min.

### Option 2 — Raw CUDA C++ kernels via `torch.utils.cpp_extension`

**What**: Write the same kernels in hand-tuned CUDA C++ and expose them via PyTorch's `cpp_extension.load()` or a `setup.py` build. Same functional scope as Option 1, but ~2× more code and harder to iterate on.

**When to use**: If Triton underperforms expectations (which shouldn't happen for memory-bound kernels) or if you need very specific optimizations that Triton doesn't expose (warp shuffles, tensor cores for mixed-precision, async memory copy).

**Effort**: ~3–4 weeks for a full port. Mostly equivalent to Option 1 in final performance.

**Expected speedup**: Same as Option 1 (~2.5× total).

### Option 3 — Sorted scatter (add-on to Options 1 or 2)

**What**: Before scatter_add, sort paths by target bin. Within a sorted chunk, all threads writing to bin `k` can accumulate their contributions in shared memory first, then a single thread does one atomic add per unique bin.

**Why**: Reduces atomic contention from ~50 GB/s effective down to ~500 GB/s for the hot bins (close to sequential memory throughput).

**Cost**: Sorting adds ~5 ms overhead but saves ~20 ms on the scatter itself = ~15 ms net win.

**Complexity**: Moderate. Use CUB `DeviceRadixSort::SortPairs` or thrust::sort_by_key. Triton has a sort primitive but it's less mature than CUB.

**Effort**: ~3 days on top of Option 1.

**Expected additional speedup**: ~10-15 ms/iter on top of the fused kernels.

### Option 4 — CUDA Graph capture

**What**: After the first iter, capture the entire sequence of kernel launches into a CUDA Graph and replay it on subsequent iters. Eliminates Python overhead, kernel launch overhead, and allows the CUDA driver to optimize across calls.

**When**: Complementary to Options 1–3. Apply after the kernels are stable.

**Caveats**:
- Doesn't work with `torch.compile` (in older PyTorch; check current support)
- Tensor shapes must be static (they are in our case — M is fixed at target_n after init)
- Breaks if any Python-level conditionals change behavior between iters (we have none in the training loop after warm-up)

**Effort**: ~1 day to wrap the training step in `torch.cuda.graph()` context.

**Expected speedup**: ~10–20% on top of whatever the fused kernels give.

### Option 5 — Full end-to-end CUDA rendering (biggest lift)

**What**: Move the entire training step (forward render + loss + backward + optimizer update) into a single monolithic CUDA extension. The PyTorch side becomes a thin wrapper that handles data loading and checkpointing.

**Why the user said this is OK**: They explicitly said "moving the entire process to CUDA files and wrapping in pytorch for usability" is fine.

**Pros**:
- Maximum fusion, minimum PyTorch overhead
- Can implement sophisticated memory management (bump allocators, per-iter scratchpads)
- Custom kernel launches for scatter can use sorted layouts end-to-end

**Cons**:
- ~1 month of engineering effort
- Large debugging surface
- Loses PyTorch's automatic differentiation — need to write all backward paths by hand
- Harder to experiment with (e.g., if you want to try a new BSDF term, you have to write forward + backward in CUDA)
- Not clear the extra speedup over Option 1+3+4 (fused Triton + sorted scatter + graph capture) is worth it

**Expected speedup**: ~5× total, vs ~3–4× from the Triton+graph approach. Marginal additional gain for large extra effort.

**Recommendation**: **Don't start with Option 5.** Start with Option 1 (Triton fused kernels), add Option 3 (sorted scatter) and Option 4 (CUDA graph) as incremental refinements. Only fall back to Options 2 or 5 if Option 1 doesn't reach the target speedup.

## Recommended execution plan (phased)

### Phase A — Infrastructure + baseline (1–2 days)

1. **Set up Triton**. Verify `triton >= 2.1` is available in the `mmir` conda env. If not, `pip install triton`.
2. **Create `mm25DGS_v4/cuda_kernels/` directory** with the following files:
   - `__init__.py` — exports the kernel wrapper functions
   - `bsdf_forward.py` — Triton kernel for the fused BSDF forward
   - `bsdf_backward.py` — Triton kernel for BSDF backward
   - `scatter_splat.py` — Triton kernel for scatter+splat
   - `reference.py` — reference PyTorch implementations for numerical validation
   - `tests/test_kernels.py` — unit tests comparing Triton vs reference
3. **Instrument per-stage timing** in `render_factorized` with `torch.cuda.synchronize()` + timers. Record baseline stage-by-stage times at `target_n=90K` to `mm25DGS_v4/output/baseline_stage_times.json`.
4. **Capture one forward pass** of all intermediate tensors (via hooks) on scene 135. Save them as a reference dump for later numerical validation of the kernels. Path: `mm25DGS_v4/output/kernel_validation/reference_forward.npz`.

### Phase B — Fused BSDF forward kernel (1 week)

1. **Write `bsdf_forward_kernel` in Triton**. Start from the existing PyTorch code in `rasterizer_factorized.py` Step 4. Hard-code the simplified BSDF (KA + SPM + Jones + slab, no CBS/directive/broad/blend — those have been removed).

2. **Validate against reference**:
   - Single-path test: input one `(m=0, t=0, r=0)` path, compare Triton output against a hand-computed reference
   - Full-scene test: run on scene 135, compare `f_cos`, `n_peak`, `phi_carrier` against the PyTorch reference saved in Phase A, with `rtol=1e-5, atol=1e-6`
   - If mismatches > tolerance, debug math before proceeding

3. **Wrap as `torch.autograd.Function`**. Use `@triton.autograd`? — if not available, define forward as the Triton kernel and backward as the reference PyTorch implementation (temporary). Full Triton backward is Phase C.

4. **Swap into `render_factorized`**. Add a flag `use_triton_bsdf=True` that routes through the new kernel. Default to `True` when Triton is available, fall back to PyTorch otherwise. Run the 7-scene benchmark with `use_triton_bsdf=True` and `use_triton_bsdf=False` to verify cart_corr is unchanged (within ±0.001).

5. **Benchmark**: time the forward pass alone (exclude backward) at target_n=90K on scene 135. Expected result: 70 ms → ~20 ms (3.5×).

### Phase C — Fused BSDF backward kernel (1 week)

1. **Write `bsdf_backward_kernel` in Triton**. This recomputes forward intermediates on the fly (from inputs + saved outputs) and applies chain rule to produce `grad_raw_materials` and `grad_normals`.

2. **Validate against PyTorch autograd**:
   - For each material parameter column, compute the gradient via both Triton and PyTorch autograd
   - Compare with `rtol=1e-4, atol=1e-5` (looser than forward because accumulated rounding)
   - Repeat for normals
   - Fix any mismatches before proceeding

3. **Swap into the autograd Function** as the true backward.

4. **Benchmark**: time the full forward+backward at target_n=90K on scene 135. Expected result: 150 ms → ~50 ms (3×).

5. **End-to-end validation**: run the 7-scene benchmark and confirm mean cart_corr is within ±0.001 of the PyTorch reference (0.9470 ± 0.001).

### Phase D — Scatter kernel + sorted scatter (3 days)

1. **Write `scatter_splat_kernel` in Triton** using `tl.atomic_add`. Port the current SplatScatterFn.apply's forward to Triton.

2. **Validate**: numerical equivalence to the existing SplatScatterFn.

3. **Benchmark without sort**: expected ~5 ms savings over the current PyTorch scatter_add.

4. **(Optional) Sorted scatter**: precompute a sort permutation of paths by flat_idx and apply it before scatter. Will need a separate small Triton kernel or thrust::sort_by_key. Expected additional ~10 ms savings.

5. **Combined expected speedup from scatter**: 45 ms → ~20 ms (2.25×) or better with sort.

### Phase E — CUDA Graph capture (1 day)

1. **Wrap the training step in `torch.cuda.graph(g)`**. After the first few warm-up iters (to let Adam initialize state, allocators stabilize), capture iters 5–10 as a CUDA graph.

2. **Replay the graph** for iters 10–499.

3. **Handle the metric path**. The per-iter cart_corr metric reads from GPU tensors — make sure those reads are compatible with graph capture (they are, but the `item()` call is a synchronization point).

4. **Benchmark**: expected 10–20% reduction on top of the Triton kernels, i.e. from ~80 ms/iter → ~65 ms/iter.

### Phase F — Per-stage validation + end-to-end benchmark (2 days)

1. **Numerical equivalence test**: for each kernel, compare against the PyTorch reference at tight tolerances. Write this as a unit test in `tests/test_cuda_kernels.py`.

2. **Gradient correctness test**: for a small synthetic scene, compute `torch.autograd.gradcheck()` on the whole training path using both Triton and PyTorch. Ensure all gradients match to 1e-4 relative.

3. **End-to-end cart_corr test**: run the 7-scene benchmark with the Triton kernels enabled. Confirm mean ∈ [0.9460, 0.9480] (within ±0.001 of the committed baseline 0.9470).

4. **Final per-iter timing**: at target_n=90K, expected:
   - Current: 260 ms/iter
   - After Phase B (Triton forward): 200 ms/iter (~1.3×)
   - After Phase C (Triton backward): 100 ms/iter (~2.6×)
   - After Phase D (Triton scatter + sort): 75 ms/iter (~3.5×)
   - After Phase E (CUDA graph): 65 ms/iter (~4×)
   - **Total expected speedup: 4× (260 → 65 ms/iter)**
   - **7-scene benchmark: 15 min → ~4 min**

## Files that will change

### New files

```
mm25DGS_v4/cuda_kernels/
  __init__.py
  bsdf_forward.py          # Triton fused BSDF forward kernel
  bsdf_backward.py          # Triton BSDF backward kernel
  scatter_splat.py          # Triton scatter+splat (forward + gather backward)
  sorted_scatter.py         # Optional: sorted-path scatter for atomic contention reduction
  reference.py              # PyTorch reference implementations (used only for tests)
  tests/
    __init__.py
    test_bsdf.py            # Forward + backward numerical equivalence tests
    test_scatter.py
    test_gradcheck.py       # PyTorch gradcheck on the full path
```

### Modified files

```
mm25DGS_v4/rasterizer_factorized.py
  - Add `use_triton_kernels: bool = True` arg to `render_factorized`
  - Route Step 4 through the Triton BSDF kernel if available
  - Route Step 5 through the Triton scatter kernel if available
  - Fall back to existing PyTorch code if Triton not installed or disabled

mm25DGS_v4/train_gaussian.py
  - Optionally wrap the training step in `torch.cuda.graph` (Phase E)
  - Default `use_triton_kernels=True` in the call to render_factorized
```

### Unchanged files

- Everything else stays exactly as is. The Triton path is a drop-in replacement; the PyTorch path remains fully supported.

## Risk register

| Risk | Mitigation |
|---|---|
| **Triton version incompatibility** with current torch | Pin `triton>=2.1`, test at install time; fall back to PyTorch if Triton fails to import |
| **Numerical divergence** from PyTorch reference | Unit tests in Phase A validate each kernel at `rtol=1e-5, atol=1e-6` for forward and `1e-4, 1e-5` for backward; end-to-end cart_corr test in Phase F bounds drift to ±0.001 |
| **Gradient bug** in custom backward kernel | `torch.autograd.gradcheck()` on a small synthetic scene in Phase F; LOO ablation smoke test to catch off-by-one errors |
| **Non-deterministic training results** from sorted scatter | Sort with a stable ordering key (flat_idx, then path_index); verify two consecutive runs give identical cart_corr |
| **CUDA graph capture fails** due to dynamic shapes | Fix `target_n` before capture; use `torch.cuda.graph()` not `torch.compile` (different API); fall back to non-graph execution if capture fails |
| **Memory explosion** in kernel due to register spilling | Monitor with `nvcc -Xptxas=-v` or `tl.trans()` to inspect register usage; reduce block size if spills detected |
| **Performance below target** (not reaching 4×) | Profile with NVIDIA Nsight Compute; identify bottleneck ops; consider Phase B with raw CUDA C++ as a fallback for the worst offenders |

## Acceptance criteria

Phase B ships when:
- Triton BSDF forward produces identical output to PyTorch reference (rtol=1e-5, atol=1e-6) on a test scene
- Forward-only benchmark shows ≥2× speedup (70 ms → ≤35 ms)
- End-to-end training on scene 135 produces cart_corr within ±0.001 of PyTorch baseline

Phase C ships when:
- Triton BSDF backward produces matching gradients (rtol=1e-4, atol=1e-5) via `torch.autograd.gradcheck()` on a 100-point test case
- Full forward+backward benchmark shows ≥2× speedup (150 ms → ≤75 ms)
- 7-scene benchmark mean cart_corr within ±0.002 of 0.9470 (slightly looser because two Triton kernels compound)

Phase D ships when:
- Scatter kernel shows ≥1.5× speedup and identical output to SplatScatterFn

Phase E ships when:
- CUDA graph capture succeeds on iter 5, replay works for iters 10–499
- Final per-iter timing at target_n=90K is ≤100 ms
- 7-scene benchmark mean cart_corr within ±0.002 of 0.9470

Entire plan ships when:
- Per-iter time at target_n=90K ≤100 ms (target: ≤65 ms)
- 7-scene benchmark time ≤8 min (target: ≤5 min)
- cart_corr mean within ±0.002 of the PyTorch baseline (0.9470)
- All unit tests pass
- `gradcheck()` passes for the full training path

## Out of scope

- **Mixed precision (fp16/bf16)**: defers to a follow-up investigation. Adds another 1.5–2× but has numerical correctness risk (exp/sqrt underflow in BSDF).
- **Multi-GPU**: out of scope. Single-scene training on one GPU.
- **FP16 atomic scatter**: may give 2× on the scatter stage but requires careful accumulator management to avoid precision loss.
- **tensor cores / wmma**: the BSDF is memory-bound, not compute-bound, so tensor cores don't help unless we change the op structure.
- **Rewriting the init pipeline (FPS, visibility, FOV)**: init is one-shot and fast (~2 s). Not worth CUDA porting.
- **Rewriting antenna pattern evaluation**: the current PyTorch `AntennaPatternTorch.evaluate` is <1 ms per iter, negligible.

## Estimated timeline

| Phase | Work | Elapsed |
|---|---|---|
| A | Infrastructure | 1–2 days |
| B | Fused forward | 1 week |
| C | Fused backward | 1 week |
| D | Scatter + sort | 3 days |
| E | CUDA graph | 1 day |
| F | Validation + benchmarks | 2 days |
| **Total** | | **~3.5 weeks** of focused effort |

Expected final speedup: **4× total (260 → 65 ms/iter)**. 7-scene benchmark from ~15 min to ~4 min. Without CUDA graph capture, ~3.5× and ~4.5 min.

## References

- **Triton documentation**: https://triton-lang.org/
- **CUB DeviceRadixSort**: for sorted scatter
- **PyTorch Custom Autograd Functions**: https://pytorch.org/docs/stable/notes/extending.html
- **NVIDIA Nsight Compute**: for kernel profiling
- **Walter et al. 2007** "Microfacet Models for Refraction through Rough Surfaces" — for verifying the BSDF math in the kernel matches our Python implementation
- **Current PyTorch implementation**: `mm25DGS_v4/rasterizer_factorized.py` (commit `ef492a3` for the full microfacet Jones, `3c653e6` for target_n=90K)
