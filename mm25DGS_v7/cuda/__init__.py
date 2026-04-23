"""mm25DGS_v7 CUDA extension loader + autograd.Function wrappers.

Exposes the step5_doppler_fused kernel (per plan
md/mm25dgs_v7_speed_and_ceiling.md §B2a). Everything else (BSDF, v5
step5_fused) is served from the mm25DGS_v5 extension; v7 callers
import those directly from ``mm25DGS_v5.cuda``.

Build:
    cd mm25DGS_v7/cuda
    /home/adnan/.conda/envs/mmir/bin/python setup.py build_ext --inplace
"""

import importlib
import importlib.util
import os
import sys

# torch must be imported before the extension .so so that libc10/libtorch are
# preloaded into the process.
import torch  # noqa: F401

__all__ = [
    "ext",
    "is_available",
    "load_error",
    "step5_doppler_fused",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
ext = None
_LOAD_ERROR = None


def _load_extension():
    for fname in os.listdir(_HERE):
        if fname.startswith("mm25dgs_v7_cuda") and fname.endswith(".so"):
            so_path = os.path.join(_HERE, fname)
            spec = importlib.util.spec_from_file_location(
                "mm25dgs_v7_cuda", so_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["mm25dgs_v7_cuda"] = mod
            return mod
    return importlib.import_module("mm25dgs_v7_cuda")


try:
    ext = _load_extension()
except Exception as e:
    ext = None
    _LOAD_ERROR = repr(e)


def is_available() -> bool:
    """True if the compiled v7 CUDA extension is importable."""
    return ext is not None


def load_error() -> str:
    return _LOAD_ERROR or ""


class Step5DopplerFusedFn(torch.autograd.Function):
    """Autograd wrapper for the fused v7 step5+Doppler kernel.

    Forward inputs (all fp32, CUDA, contiguous):
      w_full   : (M, n_tx, n_rx)
      phi_base : (M, n_tx, n_rx)       — carrier phase, detached
      n_peak   : (M, n_tx, n_rx)       — fractional bin, detached
      A_vec    : (M,)                  — Doppler factor per path, detached
      t_off    : (n_chirps, n_tx)      — per-chirp TX time offset, detached
      psf_real : (spread, n_grid)
      psf_imag : (spread, n_grid)
      K        : int (range-profile length)
      w_threshold : float

    Returns ``(rp_real, rp_imag)`` of shape ``(n_chirps, n_tx, n_rx, K)``.

    Backward: only emits grad_w_full. phi_base/n_peak/A_vec/t_off are
    taken as non-differentiable (matches step5_fused convention — the
    training path always detaches phase inputs).
    """

    @staticmethod
    def forward(ctx, w_full, phi_base, n_peak, A_vec, t_off,
                psf_real, psf_imag, K, w_threshold):
        w_full   = w_full.contiguous()
        phi_base = phi_base.contiguous()
        n_peak   = n_peak.contiguous()
        A_vec    = A_vec.contiguous()
        t_off    = t_off.contiguous()
        psf_real = psf_real.contiguous()
        psf_imag = psf_imag.contiguous()
        rp_real, rp_imag = ext.step5_doppler_fused_forward(
            w_full, phi_base, n_peak, A_vec, t_off,
            psf_real, psf_imag, int(K), float(w_threshold),
        )
        ctx.save_for_backward(
            w_full, phi_base, n_peak, A_vec, t_off, psf_real, psf_imag)
        ctx.w_threshold = float(w_threshold)
        return rp_real, rp_imag

    @staticmethod
    def backward(ctx, grad_rp_real, grad_rp_imag):
        (w_full, phi_base, n_peak, A_vec, t_off,
         psf_real, psf_imag) = ctx.saved_tensors
        grad_w = ext.step5_doppler_fused_backward(
            grad_rp_real.contiguous(), grad_rp_imag.contiguous(),
            w_full, phi_base, n_peak, A_vec, t_off, psf_real, psf_imag,
            ctx.w_threshold,
        )
        # grads for: w_full, phi_base, n_peak, A_vec, t_off,
        # psf_real, psf_imag, K, w_threshold
        return grad_w, None, None, None, None, None, None, None, None


def step5_doppler_fused(w_full, phi_base, n_peak, A_vec, t_off,
                         psf_real, psf_imag, K, w_threshold=1e-20):
    """Autograd-friendly entry point for the v7 fused kernel."""
    if ext is None:
        raise RuntimeError(
            "mm25dgs_v7_cuda extension is not built. Run:\n"
            "  cd mm25DGS_v7/cuda && python setup.py build_ext --inplace\n"
            f"Last load error: {_LOAD_ERROR}"
        )
    return Step5DopplerFusedFn.apply(
        w_full, phi_base, n_peak, A_vec, t_off,
        psf_real, psf_imag, K, w_threshold,
    )
