"""mm25DGS_v5 CUDA extension loader + autograd.Function wrappers.

Tries to import the compiled extension. If it isn't built yet, sets the
module-level ``ext`` to None so callers can fall back to the PyTorch path.

Build:
    cd mm25DGS_v5/cuda
    /home/adnan/.conda/envs/mmir/bin/python setup.py build_ext --inplace
"""

import importlib
import importlib.util
import os
import sys

# torch must be imported before the extension .so so that libc10/libtorch are
# preloaded into the process (otherwise ImportError on libc10.so).
import torch  # noqa: F401

__all__ = [
    "ext",
    "is_available",
    "scatter_splat",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
ext = None
_LOAD_ERROR = None


def _load_extension():
    """Load the compiled extension from this directory (build_ext --inplace
    drops the .so next to setup.py). Falls back to sys.path if not found.
    """
    # First, try to find an .so next to this __init__.py
    for fname in os.listdir(_HERE):
        if fname.startswith("mm25dgs_v5_cuda") and fname.endswith(".so"):
            so_path = os.path.join(_HERE, fname)
            spec = importlib.util.spec_from_file_location(
                "mm25dgs_v5_cuda", so_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["mm25dgs_v5_cuda"] = mod
            return mod
    # Fallback: sys.path import
    return importlib.import_module("mm25dgs_v5_cuda")


try:
    ext = _load_extension()
except Exception as e:  # ImportError, OSError, FileNotFoundError, etc.
    ext = None
    _LOAD_ERROR = repr(e)


def is_available() -> bool:
    """True if the compiled CUDA extension is importable."""
    return ext is not None


def load_error() -> str:
    """Return the last import error (useful for debugging) or '' if loaded."""
    return _LOAD_ERROR or ""


def scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag):
    """In-place atomic scatter splat. Equivalent to:

        rp_real.view(-1).scatter_add_(0, flat_idx, contrib_real)
        rp_imag.view(-1).scatter_add_(0, flat_idx, contrib_imag)

    but issues a single fused CUDA kernel launch.
    """
    if ext is None:
        raise RuntimeError(
            "mm25dgs_v5_cuda extension is not built. "
            "Run: cd mm25DGS_v5/cuda && python setup.py build_ext --inplace"
        )
    ext.scatter_splat(contrib_real, contrib_imag, flat_idx, rp_real, rp_imag)
