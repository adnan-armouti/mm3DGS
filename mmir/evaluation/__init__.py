"""Unified post-processing and evaluation pipeline for mmIR paper results."""

# Fix EGL initialization for Open3D offscreen rendering.
# Mesa's swrast_dri.so needs GLIBCXX_3.4.30 from the system libstdc++,
# but conda's version may lack it. Preloading via ctypes fixes dlopen.
# IMPORTANT: Skip when Mitsuba/DrJit is loaded — RTLD_GLOBAL overrides
# their bundled libstdc++ symbols and causes a segfault during CUDA ops.
import os as _os
import sys as _sys
if "mitsuba" not in _sys.modules:
    import ctypes as _ctypes
    _sys_libstdcxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if _os.path.isfile(_sys_libstdcxx):
        try:
            _ctypes.CDLL(_sys_libstdcxx, mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            pass
if "LIBGL_DRIVERS_PATH" not in _os.environ:
    _dri_path = "/usr/lib/x86_64-linux-gnu/dri"
    if _os.path.isdir(_dri_path):
        _os.environ["LIBGL_DRIVERS_PATH"] = _dri_path

from .scene_registry import SceneInfo, get_benchmark_scenes, get_scene, EXCLUDED_SCENES
from .base_evaluator import BaseEvaluator
from .material_loader import load_our_materials, load_benchmark_materials

# Lazy import: RendererWrapper pulls in mitsuba/drjit which is heavy.
# Import explicitly when needed: from mmir.evaluation.renderer_wrapper import RendererWrapper

__all__ = [
    "SceneInfo",
    "get_benchmark_scenes",
    "get_scene",
    "EXCLUDED_SCENES",
    "BaseEvaluator",
    "load_our_materials",
    "load_benchmark_materials",
]
