"""Phase F: 7-scene validation.

Trains each benchmark scene twice (PyTorch vs CUDA kernels) and reports
per-scene and mean cart_corr + wall-clock. The CUDA path must stay within
+-0.002 of the PyTorch baseline on the *mean* (per-scene MC noise is
~+-0.03 so individual scenes may wobble more).

Skipped by default because it takes ~20 minutes. Run explicitly:
    CUDA_VISIBLE_DEVICES=1 /home/adnan/.conda/envs/mmir/bin/python \
        -m pytest mm25DGS_v5_v4/cuda/tests/test_phase_f_all_scenes.py -m slow -s
"""
import time

import numpy as np
import pytest
import torch

from mm25DGS_v5 import cuda as v5cuda
from mm25DGS_v5_v4.load_pretrained import SCENES


pytestmark = [
    pytest.mark.skipif(not v5cuda.is_available(),
                       reason="CUDA extension not built"),
    pytest.mark.slow,
]


def _train(scene, use_cuda_kernels):
    import mm25DGS_v5_v4.rasterizer_factorized as rf
    from mm25DGS_v5_v4.train_gaussian import train_gaussians

    _orig = rf.render_factorized
    def _patched(*a, **kw):
        kw['use_cuda_kernels'] = use_cuda_kernels
        return _orig(*a, **kw)
    rf.render_factorized = _patched
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        t0 = time.time()
        corr, _ = train_gaussians(
            scene, num_iters=500, verbose=False,
            random_init_width=2.0,
        )
        dt = time.time() - t0
    finally:
        rf.render_factorized = _orig
    return corr, dt


def test_phase_f_seven_scenes_within_tolerance():
    results = []
    for scene in SCENES:
        corr_py, t_py = _train(scene, use_cuda_kernels=False)
        corr_cu, t_cu = _train(scene, use_cuda_kernels=True)
        results.append((scene, corr_py, t_py, corr_cu, t_cu))
        print(
            f"  {scene:<22} py={corr_py:.4f} ({t_py:5.1f}s)  "
            f"cu={corr_cu:.4f} ({t_cu:5.1f}s)  "
            f"diff={corr_cu - corr_py:+.4f}  "
            f"speedup={t_py / t_cu:.2f}x"
        )

    py_mean = float(np.mean([r[1] for r in results]))
    cu_mean = float(np.mean([r[3] for r in results]))
    py_total = float(np.sum([r[2] for r in results]))
    cu_total = float(np.sum([r[4] for r in results]))

    print(f"\n  {'MEAN':<22} py={py_mean:.4f} ({py_total:5.1f}s)  "
          f"cu={cu_mean:.4f} ({cu_total:5.1f}s)  "
          f"diff={cu_mean - py_mean:+.4f}  "
          f"speedup={py_total / cu_total:.2f}x")

    assert abs(cu_mean - py_mean) <= 0.002, (
        f"Mean cart_corr diff {cu_mean - py_mean:+.4f} exceeds +-0.002"
    )
