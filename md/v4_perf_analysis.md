# mm25DGS_v4 performance analysis and optimization roadmap

State as of commit `36d698b` (the optimized FPS commit, after the 7-commit perf-cleanup batch).

All measurements on a single RTX 4090, CUDA-synchronized timing, scene `seq_0_frame_135`, M_active = 45860, after 5 warm-up iterations.

## Per-iter cost (105 ms total)

| Stage | Time | % of iter | What it does |
|---|---:|---:|---|
| **`backward`** | **54.9 ms** | **52%** | Autograd through render → loss. Dominant cost. |
| **`render_forward`** | **31.5 ms** | **30%** | `render_factorized` → `(rp_real, rp_imag)`. |
| **`compute_ra_loss_rp`** | **12.6 ms** | **12%** | `range_profile_to_ra` (rendered + GT) + min-max + MSE. |
| `metric` (cart_corr) | 5.0 ms | 5% | `range_profile_to_ra_mag` + `polar_to_cart_torch` + `cart_corr_torch`. |
| `grad_clip` | 0.4 ms | 0.4% | Per-param `nan_to_num` + RMS clip. |
| `opt_step` | 0.3 ms | 0.3% | Adam update. |
| `zero_grad` | 0.03 ms | 0% | `set_to_none=True`. |
| `metric_sync_item` | 0.009 ms | 0% | `cart_corr.item()`. |
| `lr_update` | 0.002 ms | 0% | Trivial. |
| `antenna_inject` | 0.001 ms | 0% | Trivial. |

**95% of per-iter time is in `forward` + `backward` + `loss`.** Everything else is rounding.

The `.item()` sync at `cart_corr.item()` is essentially free (9 µs), invalidating the earlier suspicion that it was a meaningful bottleneck.

## Loss breakdown (12.6 ms total)

| Component | Time | Notes |
|---|---:|---|
| `range_profile_to_ra` (rendered) | 4.74 ms | Stack + permute + complex + virtual array remap + Hann window + 2× FFT (range + azimuth). Necessary work. |
| **`adc_to_ra_complex` (GT)** | **4.80 ms** | **WASTE** — recomputed every iter even though GT is constant. |
| `abs + min/max + normalize` (rendered) | 0.03 ms | Cheap. |
| `abs + min/max + normalize` (GT) | 0.03 ms | Also wasted (GT side). |
| MSE diff + mean | 0.01 ms | Cheap. |

**Loss with cached GT (estimated): 4.85 ms** — saves ~5 ms per iter (~2.5 s over 500 iters per scene).

## Init-time breakdown (~2.2 s total)

| Stage | Time | Notes |
|---|---:|---|
| **`FPS GPU + CPU index`** | **1458 ms** | **66% of init.** Already optimized once (40% faster than textbook), but the 50K-step Python loop is still the bottleneck. |
| FOV restrict | 185 ms | Pure numpy on 2.7M points. CPU. |
| Load Mitsuba scene | 140 ms | Single mesh.ply parse + BVH build. |
| Unpack pcl + normalize | 126 ms | Numpy on 2.7M points. |
| Ray test (1.5M rays) | 80 ms | The actual visibility test — fast given the count. |
| Cosine importance resample | 40 ms | Numpy multinomial + unique. |
| Load pcl.npy | 27 ms | 70 MB float32. |
| Other (active mask, GT load, GPU eval setup, optimizer setup) | ~140 ms | All small. |

## Renderer forward stages

The renderer's 5 internal stages, measured standalone in no-grad mode:

| Stage | Approx time | Notes |
|---|---:|---|
| Step 1: material reparam | 0.12 ms | Cheap (6 sigmoids/exps over 45K). |
| Step 2: per-TX geometry | 0.03 ms | Cheap. |
| Step 2: TX antenna eval | 0.38 ms | Custom interpolation kernel. |
| Step 3: RX antenna eval | 0.33 ms | Same as TX. |
| Step 2 cont: Fresnel + Cook-Torrance prep (per-TX) | ~3 ms | Many ops per (M, n_tx) tensor. |
| **Step 4: BSDF inner loop (per-(M, n_tx, n_rx))** | **~20 ms** | **Dominant.** ~12 einsums + 6 exps + 2 sin + sqrt + clamp + 5 broadcasts over (45K, 12, 16) ≈ 8.6M-element tensors. |
| **Step 5: PSF splatting** | **~10 ms** | Trig (cos/sin) + indexing + scatter_add over 8.6M paths × 15 spread = 130M atomic adds. |

**Total ≈ 31 ms forward** (in-loop measurement). The standalone no-grad benchmark gave 53 ms (~70% slower), probably because the in-loop benchmark benefits from steady-state CUDA kernel caching that the cold no-grad reps don't.

## Hot spots, ranked by total per-scene cost over 500 iters

| # | Hot spot | Per-iter | Per-scene (×500) | Why it's #1 |
|---|---|---:|---:|---|
| 1 | `backward` | 54.9 ms | **27.5 s** | Dominates everything else combined. |
| 2 | `render_forward` (Steps 4 + 5) | 31.5 ms | 15.8 s | The forward of the autograd graph. |
| 3 | `loss` (with wasted GT) | 12.6 ms | 6.3 s | Half is pure waste. |
| 4 | `metric` | 5.0 ms | 2.5 s | Already GPU. |
| 5 | **FPS init** | (init-only) | **1.5 s** | Once per scene. |
| 6 | Mitsuba load + ray test | (init-only) | 0.22 s | Once per scene. |

---

# Optimization roadmap

## Tier A: Free wins (no risk, low effort, no CUDA kernels)

### A1. Cache GT loss tensor
The biggest free win. Same idea as commit `fe7bbff` (GT eval cache) but applied to the loss path. The GT tensor that goes into the loss MSE doesn't change between iters; we can compute `range_profile_to_ra` of the GT once before the loop and reuse it. Also caches the GT min/max normalization.

- **Effort**: ~30 minutes
- **Per-iter saving**: ~5 ms
- **Per-scene saving**: ~2.5 s over 500 iters
- **Risk**: zero

### A2. Cache Hann windows
`torch.hann_window(num_vx, device=...)` is currently called per iter inside `range_profile_to_ra`. Move to module-level cache keyed on (length, device, dtype).

- **Effort**: ~5 minutes
- **Per-iter saving**: ~0.5 ms
- **Risk**: zero

### A3. Precompute static splatting indices
The `t_idx_full`, `r_idx_full`, `dn_offsets` tensors in Step 5 are recomputed every render but only depend on (n_tx, n_rx, K, SPREAD, M_active), all of which are fixed at scene init. Cache them on the rasterizer or as `render_factorized` static state.

- **Effort**: ~30 minutes
- **Per-iter saving**: ~0.5-1 ms
- **Risk**: low (must be invalidated if M changes — it doesn't in v4 since positions are frozen)

**Tier A combined: ~6-7 ms/iter saved (~3 s/scene). Roughly 1 hour of work.**

## Tier B: Medium effort, real wins (still pure PyTorch)

### B1. `torch.compile` the renderer forward
Steps 1-5 are a fixed-shape pipeline of pure-tensor ops. Inductor can fuse the elementwise chains into bigger kernels and reduce launch overhead. Forward fusion typically gives 1.3-2× speedup on this kind of pipeline. Backward also benefits because autograd traces the compiled graph.

- **Effort**: ~1 day (handle shape dynamism, debug fusion issues)
- **Risks**: 
  - Shape dynamism on the active mask (workaround: keep `M_active` fixed since positions are frozen — easy)
  - `cudagraph_mark_step_begin` / cuda-graph tensor lifetime issues with the optimizer state (workaround: use `mode="default"` not `mode="reduce-overhead"`)
  - Recompilation churn if input shapes change between scenes (workaround: clear the compile cache between scenes)
- **Per-iter saving**: ~10-15 ms (forward) + 5-10 ms (backward via autograd traversal)
- **Per-scene saving**: ~7-12 s over 500 iters

### B2. Custom `torch.autograd.Function` for PSF splatting
The `scatter_add_` + autograd-tracked indexing in Step 5 has expensive backward because autograd must materialize the index gradient flow through the gather. A hand-written backward (just an `index_select` from `grad_output`) is much faster.

- **Effort**: ~half day (write forward + backward + numerical correctness check)
- **Risks**: low — this is pure refactor
- **Per-iter saving**: ~5-10 ms on backward
- **Per-scene saving**: ~3-5 s over 500 iters

**Tier B combined: 15-25 ms/iter saved (~10 s/scene). 1-2 days of work.**

## Tier C: CUDA kernels (.cu files, biggest wins)

This is where we move into native CUDA. Three prime targets, in order of return-on-effort.

### C1. CUDA `psf_splat.cu` — Step 5
Single fused kernel that takes `(w_full, n_peak, phi_carrier)` and writes directly to `(rp_real, rp_imag)`.

**Inputs**:
- `w_full (M, n_tx, n_rx)` float32
- `n_peak (M, n_tx, n_rx)` float32
- `phi_carrier (M, n_tx, n_rx)` float32
- `psf_table (n_grid, SPREAD)` precomputed (already exists in v4)
- Constants: `K`, `SPREAD`, `n_tx`, `n_rx`

**Output**: `rp_real (n_tx, n_rx, K)`, `rp_imag (n_tx, n_rx, K)`.

**Why it'll be fast**:
- Currently this stage involves ~8 separate kernels: `cos(phi)`, `sin(phi)`, `* w`, PSF lookup, complex multiply, two `scatter_add_` kernels, plus the active-mask filtering. Each has ~5-10 µs launch overhead and round-trips intermediate tensors through global memory.
- One fused kernel does all the math in registers and uses `atomicAdd` to write directly to the output. With ~8.6M paths × 15 spread positions = ~130M atomic adds spread across 49152 output bins, average ~2600 atomics per bin — well within the contention regime atomics handle efficiently.

**Forward estimate**: 10 ms → 2-3 ms (~3-4× faster).
**Backward**: write a custom backward kernel using gather (the transpose of scatter). Estimate: ~4-6 ms vs current ~20 ms.

- **Effort**: ~1 week (forward kernel + custom autograd Function + backward kernel + numerical check)
- **Risk**: medium (atomic correctness, gradient correctness)

### C2. CUDA `bsdf_inner_loop.cu` — Step 4 (THE biggest single win)
Single fused kernel that takes the precomputed per-TX and per-RX tensors and produces `f_cos` of shape `(M, n_tx, n_rx)` in one pass.

**Inputs (precomputed by Step 1-3 in PyTorch)**:
- `wi (M, n_tx, 3)`, `wo (M, n_rx, 3)`, `n_eff (M, n_tx, 3)`
- `cos_i (M, n_tx)`, `cos_o (M, n_rx)`, `lambda_i (M, n_tx)`, `lambda_o (M, n_rx)`
- `alpha_sq (M,)`, `kappa_SPM (M,)`, `kappa_dir (M,)`, `norm_SPM (M,)`, `norm_dir (M,)`
- `eps_factor (M,)`, `gamma (M,)`, `cbs_mean (M,)`, `tau_eff (M, n_tx)`, `eta (M, n_tx)`
- `E_s_out (M, n_tx) complex`, `E_p_out (M, n_tx) complex`, `s_in (M, n_tx, 3)`
- `wi_r (M, n_tx, 3)`, `retro (M, n_tx, 3)`, `lc (M,)`
- `rx_pol (3,)` constant

**Output**: `f_cos (M, n_tx, n_rx)` float32.

**Why it'll be fast**:
- One thread per (m, t, r) path → 8.6M threads, perfectly mapped to GPU.
- All intermediate values (`f_KA`, `f_SPM`, `f_dir`, `f_broad`, `f_lobe`, `R_jones`, half-vectors, etc.) live in registers — no global memory traffic. PyTorch currently materializes ALL of these to global memory between each einsum/exp/clamp.
- ~12 separate PyTorch kernel launches → 1 launch.
- Memory bandwidth: 8.6M output writes vs PyTorch's ~10× that for intermediates.

**Forward estimate**: 20 ms → 4-6 ms (~3-4× faster).
**Backward**: hand-coded backward kernel using the chain rule for each material parameter. Wrapped in a `torch.autograd.Function`. Backward estimate: ~8-12 ms vs current ~30 ms.

- **Effort**: 1-2 weeks (most code; lots of math; gradients for 6 material params; fp32 numerical correctness)
- **Risk**: high (math is complex; gradient correctness is the hard part)

### C3. CUDA `fps.cu` — init-only, easiest first kernel
Replaces the Python loop in `_farthest_point_sampling`.

**Inputs**: `pts (N, 3)`, `n_samples`.
**Output**: `selected (n_samples,)` long.

**Why it'll be fast**:
- The Python loop currently does 50K iterations of `argmax` + `index_select` + `sub` + `square` + `sum` + `min`. Each iter has ~15 µs Python+launch overhead.
- One CUDA kernel can hold the entire loop in device code, only return when done. Per-iter cost drops from ~30 µs to ~5 µs (just memory bandwidth to scan ~130K floats once).

**Estimate**: 1.5 s → ~50-100 ms (~15-30× faster). **Init-only** — won't speed up training iters.

- **Effort**: ~3-5 days (sequential reduction + parallel scan; no autograd needed)
- **Risk**: low (no gradient path; pure spatial geometry)
- **Best entry-point CUDA kernel** because it's standalone, no autograd, simple math.

**Tier C combined: ~25-35 ms/iter saved (~12-17 s per 500 iters), plus ~1.4 s init savings. 3-4 weeks of work for someone fluent in CUDA.**

## Tier D: Mixed precision

### D1. bf16 forward, fp32 backward
The forward involves enough exp/sigmoid that fp16 may saturate, but bf16 has the same exponent range as fp32 — should be safe.

- **Effort**: ~1 day (per-op precision policy + careful clamping)
- **Per-iter saving**: 1.5-2× on the forward, partial speedup on backward
- **Risks**:
  - The Cook-Torrance `1/(4 cos_i cos_o)` can produce huge values at grazing angles that overflow even bf16. Need careful clamping.
  - Scatter atomics in bf16 have lower precision — accumulation rounding error in `scatter_add_` could affect gradients.

---

## Recommended execution order

| Order | What | Effort | Per-iter saving | Cumulative saving |
|---|---|---|---:|---:|
| 1 | **Tier A** (cache GT loss + Hann windows + splat indices) | 1 hour | ~6 ms | ~6 ms |
| 2 | **Tier B** (`torch.compile` + custom PSF autograd Fn) | 1-2 days | ~20 ms | ~26 ms |
| 3 | **C1** (`psf_splat.cu`) | 1 week | ~15 ms | ~41 ms |
| 4 | **C3** (`fps.cu`) | 3-5 days | (init-only ~1.4 s) | (no per-iter change) |
| 5 | **C2** (`bsdf_inner_loop.cu`) | 1-2 weeks | ~25 ms | **~66 ms total** |
| 6 | **D1** (bf16 forward) | 1 day | ~5-10 ms | **~75 ms total** |

After Tier A + B (no CUDA kernels): per-iter ~75 ms (down from 105 ms, ~30% faster).
After Tier C + D: per-iter ~30-40 ms (down from 105 ms, ~3× faster).
Combined: a 500-iter scene drops from ~56 s to ~17-20 s of training, plus ~1.4 s of init savings.

## Where to start CUDA work

If/when we move into custom CUDA kernels:

1. **`fps.cu` first** — pure spatial geometry, no autograd, simple math. Best learning project.
2. **`psf_splat.cu` second** — math is simple (cos/sin/scatter), backward is well-defined (gather), scatter atomics are well-understood. Bigger payoff than FPS.
3. **`bsdf_inner_loop.cu` last** — biggest single win but lots of math, custom backward for 6 material params, hardest correctness.

## Out of scope for this analysis

- Distributed / multi-GPU training (we already use both GPUs in parallel for the 7-scene benchmark, but not for a single scene).
- Network compression (the model is already tiny — 50K × 13 floats ≈ 2.6 MB).
- Different optimizers (Adam is fine for this regime).
- Different rasterization paradigms (e.g., using Mitsuba's differentiable backend) — we deliberately moved away from this for cleanliness.
