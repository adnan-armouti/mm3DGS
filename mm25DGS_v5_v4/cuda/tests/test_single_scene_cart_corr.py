"""End-to-end correctness gate: train scene 135 with CUDA kernels enabled
and verify cart_corr stays within ±0.002 of the PyTorch baseline.

This is the TRUE correctness test for each CUDA phase. Per-kernel tests
catch math bugs in isolation; this one catches accumulation / gradient /
autograd subtleties that only show up in the 500-iter optimization loop.

Slow: ~4-5 minutes. Skipped unless explicitly requested with --runslow.
"""
import os
import time

import numpy as np
import pytest
import torch

from mm25DGS_v5 import cuda as v5cuda


pytestmark = [
    pytest.mark.skipif(not v5cuda.is_available(),
                       reason="CUDA extension not built"),
    pytest.mark.slow,
]


def _run_training(use_cuda_kernels: bool):
    import mm25DGS_v5_v4.rasterizer_factorized as rf
    from mm25DGS_v5_v4.train_gaussian import train_gaussians

    # Monkey-patch render_factorized to force the flag
    _orig = rf.render_factorized
    def _patched(*a, **kw):
        kw['use_cuda_kernels'] = use_cuda_kernels
        return _orig(*a, **kw)
    rf.render_factorized = _patched
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        t0 = time.time()
        corr, it = train_gaussians(
            'seq_0_frame_135', num_iters=500, verbose=False,
            random_init_width=2.0,
        )
        dt = time.time() - t0
    finally:
        rf.render_factorized = _orig
    return corr, dt


def test_single_scene_cart_corr_within_tolerance():
    """cart_corr with CUDA kernels must match PyTorch baseline to ±0.002."""
    corr_py, t_py = _run_training(use_cuda_kernels=False)
    corr_cu, t_cu = _run_training(use_cuda_kernels=True)
    print(
        f"\n  PyTorch baseline: cart_corr = {corr_py:.4f}  ({t_py:.0f}s)"
        f"\n  CUDA path       : cart_corr = {corr_cu:.4f}  ({t_cu:.0f}s)"
        f"\n  diff = {corr_cu - corr_py:+.4f}"
        f"\n  speedup = {t_py/t_cu:.2f}x"
    )
    assert abs(corr_cu - corr_py) < 0.002, (
        f"cart_corr drift too large: py={corr_py:.4f}, cu={corr_cu:.4f}, "
        f"diff={corr_cu - corr_py:+.4f} (limit: ±0.002)"
    )
