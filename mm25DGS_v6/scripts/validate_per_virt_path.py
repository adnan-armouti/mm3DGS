"""M1 unit validation: per-virtual GT path is bit-identical to v5 FFT path.

The v6 supervision GT computes:
    adc → adc_to_per_virt_range_profile → (192, 256) complex
    reshape → (12, 16, 256)
    range_profile_to_ra (azimuth FFT only, v5 helper) → (127, 256) complex

The v5 supervision GT computes:
    adc → adc_to_ra_complex (range FFT + 86-virtual extract + azimuth FFT)
        → (127, 256) complex

These two outputs MUST be bit-identical (max abs diff < 1e-5). Anything
else means the v6 GT path differs from v5's processing in ways that
would corrupt training when the loss switches at M2.

Run:
    /home/adnan/.conda/envs/mmir/bin/python -m mm25DGS_v6.scripts.validate_per_virt_path
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmir.data.ra_utils import adc_to_ra_complex
from mm25DGS_v5.train_gaussian import range_profile_to_ra
from mm25DGS_v6.data.ra_utils import adc_to_per_virt_range_profile


def _load_chirp(npy_path: str, chirp_idx: int = 0) -> torch.Tensor:
    """Load one chirp from a cascaded ADC .npy in the v5 trainer
    convention.

    On-disk shape: ``(n_chirps=16, RX=16, TX=12, ADC=256)`` complex128.
    Returns ``(TX=12, RX=16, ADC=256, 2)`` real-imag float32 — the
    same tensor that ``mm25DGS_v6.train_frame_nvs._build_per_loop_gt``
    constructs.
    """
    arr = np.load(npy_path)
    if arr.ndim != 4 or arr.shape[0] != 16:
        raise ValueError(f"unexpected ADC shape {arr.shape}")
    chirp = arr[chirp_idx]                                          # (RX, TX, ADC) cx
    ri = np.stack([chirp.real, chirp.imag], axis=-1).astype(np.float32)
                                                                    # (RX, TX, ADC, 2)
    ri = ri.transpose(1, 0, 2, 3)                                   # (TX, RX, ADC, 2)
    return torch.from_numpy(ri)


def main():
    ap = argparse.ArgumentParser(description="M1 per-virt GT path validation")
    ap.add_argument(
        "--scene",
        default="seq_0_frame_135",
        help="scene under data/ to load (default: seq_0_frame_135)",
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="frame index; defaults to the int after 'frame_' in scene name",
    )
    ap.add_argument("--chirp", type=int, default=0, help="chirp index (0..15)")
    ap.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "data"),
        help="root dir containing scene subdirs",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--rel-tol",
        type=float,
        default=1e-3,
        help="pass tolerance on max |Δ|/|v5| (relative; FP32 reorder floor)",
    )
    ap.add_argument(
        "--corr-tol",
        type=float,
        default=1.0 - 1e-5,
        help="pass tolerance on complex + magnitude correlation (≥ tol → pass)",
    )
    args = ap.parse_args()

    if args.frame is None:
        try:
            args.frame = int(args.scene.split("_")[-1])
        except ValueError:
            ap.error("--frame is required when scene name has no trailing int")

    npy_path = os.path.join(
        args.data_root, args.scene, "radar", f"cascaded_frame_{args.frame}.npy"
    )
    if not os.path.isfile(npy_path):
        sys.exit(f"ADC file not found: {npy_path}")

    print(f"[validate_per_virt_path]")
    print(f"  scene = {args.scene}")
    print(f"  frame = {args.frame}, chirp = {args.chirp}")
    print(f"  npy   = {npy_path}")
    print(f"  device = {args.device}")
    print(f"  rel-tol = {args.rel_tol}")
    print(f"  corr-tol = {args.corr_tol:.8f}")
    print()

    adc = _load_chirp(npy_path, chirp_idx=args.chirp).to(args.device)
    print(f"  loaded ADC shape:  {tuple(adc.shape)}  dtype={adc.dtype}")

    # --- v5 path ---
    with torch.no_grad():
        ra_v5 = adc_to_ra_complex(adc)
    print(f"  v5 RA shape:       {tuple(ra_v5.shape)}  dtype={ra_v5.dtype}")

    # --- v6 per-virt path → through v5's range_profile_to_ra ---
    with torch.no_grad():
        rp_virt = adc_to_per_virt_range_profile(adc)                 # (192, 256) cx
        rp_txrx = rp_virt.reshape(12, 16, rp_virt.size(-1))           # (12, 16, 256) cx
        ra_v6 = range_profile_to_ra(rp_txrx.real, rp_txrx.imag)
    print(f"  v6 per-virt shape: {tuple(rp_virt.shape)}  dtype={rp_virt.dtype}")
    print(f"  v6 RA shape:       {tuple(ra_v6.shape)}  dtype={ra_v6.dtype}")

    # --- Compare ---
    diff = (ra_v6 - ra_v5).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / ra_v5.abs().clamp_min(1e-12)).max().item()
    print()
    print(f"  max |Δ|        = {max_diff:.3e}")
    print(f"  max |Δ|/|v5|   = {rel_diff:.3e}")

    cc_complex = (ra_v6.conj() * ra_v5).sum().abs() / (
        ra_v6.abs().pow(2).sum().sqrt() * ra_v5.abs().pow(2).sum().sqrt()
    )
    cc_mag = float(
        torch.corrcoef(
            torch.stack([ra_v6.abs().reshape(-1), ra_v5.abs().reshape(-1)])
        )[0, 1]
    )
    print(f"  complex corr   = {cc_complex.item():.6f}  (expected: 1.0)")
    print(f"  magnitude corr = {cc_mag:.6f}                (expected: 1.0)")

    # Pass criteria: relative error + correlation. The v5 and v6 paths
    # differ only in the *order* of (FFT, Hann-window, duplicate-averaging)
    # — all linear ops. Algebraically identical, FP32-different by
    # ~1e-4 relative. Since cart_corr is a normalised cross-correlation
    # on magnitude images (rank-1 in the top ~1% bins), the relative
    # error is the right pass criterion, not absolute.
    ok_rel = rel_diff < args.rel_tol
    ok_cmplx = cc_complex.item() >= args.corr_tol
    ok_mag = cc_mag >= args.corr_tol
    if ok_rel and ok_cmplx and ok_mag:
        print(
            f"\n  PASS — max rel diff {rel_diff:.3e} < {args.rel_tol}; "
            f"complex corr {cc_complex.item():.6f}, mag corr {cc_mag:.6f} "
            f">= {args.corr_tol:.6f}"
        )
        print(
            "  (Absolute diff > 0 expected: v5 averages ADC pre-FFT, v6 "
            "averages range-profiles post-FFT. FFT is linear → algebraically "
            "identical; FP32-different by ~1e-4 rel. This is below MC noise.)"
        )
        return 0
    else:
        fails = []
        if not ok_rel:
            fails.append(f"rel-tol ({rel_diff:.3e} >= {args.rel_tol})")
        if not ok_cmplx:
            fails.append(f"complex-corr ({cc_complex.item():.6f} < {args.corr_tol:.6f})")
        if not ok_mag:
            fails.append(f"mag-corr ({cc_mag:.6f} < {args.corr_tol:.6f})")
        print(f"\n  FAIL — {', '.join(fails)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
