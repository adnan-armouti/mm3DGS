"""Build the mm25DGS_v5 CUDA extension.

Usage:
  cd mm25DGS_v5/cuda && /home/adnan/.conda/envs/mmir/bin/python setup.py build_ext --inplace

Target: sm_89 (RTX 4090 / Ada Lovelace). PyTorch >= 2.1 with torch.utils.cpp_extension.
"""
import os
import torch.utils.cpp_extension as _cpp_ext
# PyTorch 2.7.1+cu118 was compiled against CUDA 11.8 but the system has CUDA
# 12.8. The generated objects run fine at runtime (cu118 libs are forward
# compatible with the 12.x driver + no cu12-only features used here), so we
# bypass the pre-build check that torch.utils.cpp_extension runs.
_cpp_ext._check_cuda_version = lambda *a, **kw: None

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))

ext = CUDAExtension(
    name='mm25dgs_v5_cuda',
    sources=[
        os.path.join(HERE, 'bindings.cpp'),
        os.path.join(HERE, 'bsdf_forward.cu'),
        os.path.join(HERE, 'bsdf_backward.cu'),
        os.path.join(HERE, 'scatter_splat.cu'),
    ],
    include_dirs=[HERE],
    extra_compile_args={
        'cxx': ['-O3', '-std=c++17'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-std=c++17',
            '-gencode=arch=compute_89,code=sm_89',
            '--expt-relaxed-constexpr',
            '-lineinfo',
        ],
    },
)

setup(
    name='mm25dgs_v5_cuda',
    version='0.1.0',
    description='mm25DGS_v5 CUDA kernels (fused BSDF + scatter splat)',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
