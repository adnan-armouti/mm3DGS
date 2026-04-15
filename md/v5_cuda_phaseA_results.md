# mm25DGS_v5 CUDA Phase A — Infrastructure + baseline

Branch: `cuda` (worktree at `/home/adnan/Desktop/mm3DGS-cuda`)
Base commit: `f42e933` (the `pt` branch's plan commit).
GPU used: **GPU 1 (RTX 4090, sm_89)** — GPU 0 reserved for another task.

## What Phase A shipped

1. **`mm25DGS_v5/` subdirectory** — copy of the relevant `mm25DGS_v4` files
   (`rasterizer.py`, `rasterizer_factorized.py`, `train_gaussian.py`,
   `psf.py`, `load_pretrained.py`, `material_diagnostics.py`, `__init__.py`),
   with intra-package imports rewritten `mm25DGS_v4` → `mm25DGS_v5`. The v4
   tree is untouched.

2. **`mm25DGS_v5/cuda/` extension directory**:

   ```
   mm25DGS_v5/cuda/
     setup.py          # CUDAExtension, -O3 --use_fast_math -std=c++17, sm_89
     bindings.cpp      # pybind11 wrappers with CUDA/contig/dtype guards
     utils.cuh         # float2 complex ops, device-side reparam6(),
                       #   itu_slab_fresnel() port (matches Python sign conv)
     bsdf_forward.cu   # Phase A stub (zero-fill); Phase B will replace
     bsdf_backward.cu  # Phase A stub (zero-fill); Phase C will replace
     scatter_splat.cu  # Working atomicAdd scatter splat
     __init__.py       # Loader: imports torch, then loads .so from _HERE
     .gitignore        # build/, *.so, *.o
     tests/
       __init__.py
       test_scatter.py # 5 tests: ext loads, stub zero-fills, scatter_splat
                       #   matches torch.scatter_add_ at 3 sizes
     benchmark_baseline.py  # Phase A baseline dump + reference capture
   ```

3. **Build verified**. Extension compiles with nvcc 12.8 against
   torch 2.7.1+cu118 after bypassing
   `torch.utils.cpp_extension._check_cuda_version` (cu118 runtime is forward
   compatible with the 12.x driver; no cu12-only features are used). Build
   command:

   ```bash
   cd mm25DGS_v5/cuda
   CUDA_VISIBLE_DEVICES=1 TORCH_CUDA_ARCH_LIST="8.9" \
     /home/adnan/.conda/envs/mmir/bin/python setup.py build_ext --inplace
   ```

4. **Tests pass (5/5)**:

   ```
   mm25DGS_v5/cuda/tests/test_scatter.py::test_scatter_splat_matches_scatter_add[100-64]     PASSED
   mm25DGS_v5/cuda/tests/test_scatter.py::test_scatter_splat_matches_scatter_add[10000-4096] PASSED
   mm25DGS_v5/cuda/tests/test_scatter.py::test_scatter_splat_matches_scatter_add[1000000-49152] PASSED
   mm25DGS_v5/cuda/tests/test_scatter.py::test_ext_loads                                     PASSED
   mm25DGS_v5/cuda/tests/test_scatter.py::test_bsdf_forward_stub_zero_fills                  PASSED
   ```

5. **Baseline capture** (`mm25DGS_v5/output/baseline_stage_times.json`):

   ```json
   {
     "render_total_ms": 126.6,
     "n_iter": 30,
     "note": "forward-only (no_grad); full fwd+bwd timing adds ~2x.",
     "scene": "seq_0_frame_135",
     "N": 90000,
     "active": 82528
   }
   ```

   This is forward-only render_factorized time on scene 135 at target_n=90K.
   The plan's 260 ms/iter figure is for full training step (forward + loss +
   backward + optimizer). Forward-only at 126.6 ms is consistent with the
   plan's Step 1-5 forward breakdown (~145 ms).

6. **Reference forward snapshot** at
   `mm25DGS_v5/output/kernel_validation/reference_forward.npz`. Holds
   positions / normals / areas / raw_materials / TX+RX geometry / radar
   constants / rp_real / rp_imag for scene 135. Used by Phase B to numerically
   validate the fused BSDF forward kernel against the PyTorch reference
   (`rtol=1e-5, atol=1e-6`).

## Gotchas encountered

- **CUDA/nvcc mismatch**: torch 2.7.1 was built against cu118, system nvcc
  is 12.8. Fixed by monkey-patching `_check_cuda_version` in `setup.py`.
- **`libc10.so` import error** when pytest loaded the .so directly. Fixed
  by adding `import torch` at the top of `mm25DGS_v5/cuda/__init__.py` so
  libtorch/libc10 are preloaded before dlopen of the extension.
- **OOM in the timing loop** when calling `render_factorized` without
  `torch.no_grad()` — each iter's autograd graph holds ~30 intermediate
  `(M, n_tx, n_rx)` tensors and 30 accumulated iters ran past 22 GB. Wrap
  the forward timing loop in `torch.no_grad()`.
- **Port sanity checks for `itu_slab_fresnel`**. The Python impl uses
  `eta = complex(eps_real, -eps_imag)` (negative imaginary convention) and
  the ITU slab formula
  `R = r (1 - exp(-2jq)) / (1 - r² exp(-2jq))` with `q = (2π/λ)·d·a`.
  The CUDA port in `utils.cuh` mirrors this exactly (see the inline
  derivation in the docstring). Verified by code comparison; numerical
  validation is part of Phase B.

## What is intentionally deferred

- **Per-stage timing split**. Instead of instrumenting `render_factorized`
  with cuda events for each of Steps 1-5, Phase A captures only total
  forward time. The split is added in Phase B at the same time we
  route Step 4 through the new CUDA forward kernel, which gives us a
  natural seam to place the timer bracket.
- **Real BSDF kernel implementation**. `bsdf_forward.cu` and
  `bsdf_backward.cu` are zero-fill stubs so the build + bindings path is
  exercised end-to-end; the Python autograd wrapper is not wired into
  `render_factorized` yet.

## Acceptance (Phase A)

- [x] Extension directory structure matches the plan layout.
- [x] `python setup.py build_ext --inplace` succeeds under the mmir conda
      env with nvcc 12.8.
- [x] Extension loads from Python (`mm25DGS_v5.cuda.is_available()` → True).
- [x] `scatter_splat` numerically matches `torch.scatter_add_` at three
      test sizes (largest: 1M items → 49K output bins, the realistic
      Step 5 working set shape).
- [x] Baseline per-iter forward time dumped to JSON.
- [x] Reference forward .npz captured for Phase B validation.
- [x] 5/5 pytest tests pass.

Phase A is complete. Phase B starts with porting the fused BSDF forward
kernel and validating against `reference_forward.npz`.
