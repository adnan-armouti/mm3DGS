"""Build the mm25DGS_v7 CUDA extension (step5_doppler fused kernel only).

v7 reuses mm25DGS_v5's BSDF + step5_fused extension for everything else;
this extension adds a single fused step5+Doppler kernel. Keeping it in
its own module keeps v5 stable.

Usage:
  cd mm25DGS_v7/cuda
  /home/adnan/.conda/envs/mmir/bin/python setup.py build_ext --inplace

Target: sm_89 (RTX 4090 / Ada Lovelace).
"""
import os
import torch.utils.cpp_extension as _cpp_ext
# PyTorch 2.7.1+cu118 was compiled against CUDA 11.8 but the system has CUDA
# 12.8. Bypass the mismatch check (matches mm25DGS_v5/cuda/setup.py).
_cpp_ext._check_cuda_version = lambda *a, **kw: None

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

ext = CUDAExtension(
    name='mm25dgs_v7_cuda',
    sources=[
        os.path.join(HERE, 'bindings.cpp'),
        os.path.join(HERE, 'step5_doppler.cu'),
    ],
    include_dirs=[HERE],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '--extended-lambda',
            '-std=c++17',
            '--ptxas-options=-v',
            '-gencode=arch=compute_89,code=sm_89',
            '--expt-relaxed-constexpr',
            '-lineinfo',
        ],
    },
)

setup(
    name='mm25dgs_v7_cuda',
    version='0.1.0',
    description='mm25DGS_v7 CUDA kernels (step5_doppler fused)',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
