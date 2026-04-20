"""v6 radar data utilities — supervision-side ADC/range-profile processing.

M1: `adc_to_per_virt_range_profile` produces the complex per-virtual
range-profile tensor from cascaded ADC data. Used both as GT for
training supervision (feeds the Gram-matrix loss at M2) and as a
pre-azimuth intermediate that v5's `range_profile_to_ra` can consume
(this is how the M1 bit-identity validation works).

M1 loss and M2 Doppler FFT are NOT implemented here — those live in
``mm25DGS_v6/losses/rd_losses.py`` and
``mm25DGS_v6/renderer/doppler_synthesis.py`` respectively.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = [
    "adc_to_per_virt_range_profile",
    "config_tx_to_adc_tx_perm",
    "reorder_rendered_to_adc",
    "gram_correlation_per_range",
    "gram_correlation_mean",
]


# ---------------------------------------------------------------------------
# TX channel ordering: config-order (rasterizer's tx_positions) vs
# ADC channel order (on-disk .npy TX axis, matching TX_LOCATIONS).
# ---------------------------------------------------------------------------
#
# mmir/preprocessing/config_utils.antenna_layout_cascade builds the
# config's tx_array as [TX1, TX2, ..., TX12] where the NAMES are in
# ascending-config-index order but the PHYSICAL ASCENDING-AZIMUTH
# order is [TX1, TX2, TX3, TX10, TX11, TX12, TX4, TX5, TX6, TX7, TX8, TX9].
#
# The hardware (and hence on-disk .npy files via
# ColoRadar_tools.get_adc_frame) uses **ascending-physical-azimuth**
# ordering: ADC TX channel i carries data from physical TX_LOCATIONS[i].
#
# So ADC channel → config slot mapping:
#   ADC ch 0 → physical TX1 = config slot 0
#   ADC ch 1 → physical TX2 = config slot 1
#   ADC ch 2 → physical TX3 = config slot 2
#   ADC ch 3 → physical TX10 = config slot 9   ← mismatch begins here
#   ADC ch 4 → physical TX11 = config slot 10
#   ADC ch 5 → physical TX12 = config slot 11
#   ADC ch 6 → physical TX4 = config slot 3
#   ADC ch 7 → physical TX5 = config slot 4
#   ADC ch 8 → physical TX6 = config slot 5
#   ADC ch 9 → physical TX7 = config slot 6
#   ADC ch 10 → physical TX8 = config slot 7
#   ADC ch 11 → physical TX9 = config slot 8
#
# IMPORTANT — this discrepancy has been latent in v5: ``range_profile_to_ra``
# passes the renderer output (config-ordered) to ``txrx_to_vx_chirps_torch``
# which interprets the TX axis as ADC-channel-index via ``TX_LOCATIONS``.
# That is a bug: it places physical TX4's rendered signal at virtual
# position intended for physical TX10, and vice versa. v5 survives
# because it supervises on FFT *magnitude*, which is partly robust to
# such shuffles. The Gram correlation (strictly per-channel) fully
# exposes the mismatch → ~0 cc without this permutation.
#
# For v6 we leave the v5 FFT path untouched (final_test_cc stays
# bit-identical) and apply the correction on the Gram / per-virt loss
# path by reordering the renderer output from config order to ADC
# channel order before computing the Gram.
CONFIG_TX_TO_ADC_TX_PERM: tuple = (0, 1, 2, 9, 10, 11, 3, 4, 5, 6, 7, 8)
# Usage: ``rend_adc_order[i] = rend_config_order[CONFIG_TX_TO_ADC_TX_PERM[i]]``


def config_tx_to_adc_tx_perm() -> torch.Tensor:
    """Return the TX permutation (int64 (12,) tensor) that reorders
    a config-ordered TX axis to ADC channel order."""
    return torch.as_tensor(CONFIG_TX_TO_ADC_TX_PERM, dtype=torch.long)


def reorder_rendered_to_adc(
    rp_tx_rx_k: torch.Tensor,
) -> torch.Tensor:
    """Permute the TX axis of a rendered per-(tx, rx, K) tensor from
    **config order** (rasterizer's native output) to **ADC channel
    order** (matches the .npy file's TX axis, which is what
    adc_to_per_virt_range_profile produces).

    Input shape: ``(n_tx=12, n_rx=16, K=256)`` — may be real, complex,
    or any dtype. Output: same shape in ADC order.
    """
    if rp_tx_rx_k.size(0) != 12:
        raise ValueError(
            f"expected TX as first axis with size 12; got shape "
            f"{tuple(rp_tx_rx_k.shape)}"
        )
    perm = config_tx_to_adc_tx_perm().to(rp_tx_rx_k.device)
    return rp_tx_rx_k.index_select(0, perm)


def adc_to_per_virt_range_profile(
    adc_ri: torch.Tensor,
    range_window: str = "hann",
) -> torch.Tensor:
    """Range FFT + MIMO channel flatten on a single-chirp ADC tensor.

    **Input**: ``adc_ri`` is a 4-D real-imag tensor with 12 TX, 16 RX,
    256 ADC samples, last dim = 2 (I/Q). Matches the shape the v5
    trainer already constructs at
    ``mm25DGS_v6/train_frame_nvs.py::_build_per_loop_gt`` — axis order
    auto-detected, last dim = 2. Typical concrete shape:
    ``(TX=12, RX=16, ADC=256, 2)`` float32.

    **Output**: ``(n_virt=192, n_range=256)`` complex. Channel index
    ``v = tx * 16 + rx`` matches the renderer's
    ``render_factorized(...)`` return shape ``(n_tx, n_rx, K)`` flattened
    in the natural ``(tx, rx)`` order (rx varies fastest).

    **Window / FFT choice**: Hann window then range FFT, matching
    ``mmir/data/ra_utils.adc_to_ra_complex`` bit-for-bit. Windowing on
    the range (ADC) axis is per-channel so it commutes with any
    downstream virtual-axis operation. This means feeding our output
    (reshaped back to ``(tx, rx, K)``) through v5's
    ``range_profile_to_ra`` produces an RA image that is bit-identical
    to the v5 GT path's output (validated by
    ``mm25DGS_v6/scripts/validate_per_virt_path.py``).

    No azimuth processing is applied (no DBF, no FFT). This is the
    pre-azimuth supervision tensor for M2's Gram loss.
    """
    if adc_ri.ndim != 4 or adc_ri.size(-1) != 2:
        raise ValueError(
            f"expected 4-D (..., 2) real-imag tensor; got {tuple(adc_ri.shape)}"
        )

    # Auto-detect TX/RX/ADC axes exactly as v5's adc_to_ra_complex does.
    # The real-imag channel is the last axis.
    dims = list(adc_ri.shape[:-1])
    try:
        idx_adc = int(dims.index(256))
        idx_tx = int(dims.index(12))
        idx_rx = int(dims.index(16))
    except ValueError as e:
        raise ValueError(
            f"could not find (12, 16, 256) axes in shape "
            f"{tuple(adc_ri.shape)}; v5 convention is (TX, RX, ADC, 2)"
        ) from e

    # Permute to (TX=12, RX=16, ADC=256, 2). Unlike v5, we keep TX as
    # the first axis so the flatten matches the renderer's (tx, rx, K)
    # layout.
    order = [idx_tx, idx_rx, idx_adc, len(dims)]
    x = adc_ri.permute(order).contiguous()                         # (12, 16, 256, 2)
    x_c = torch.complex(x[..., 0].contiguous(), x[..., 1].contiguous())
                                                                    # (12, 16, 256) cx

    # Range window (Hann), per-channel on the ADC axis.
    if range_window == "hann" or range_window is None:
        # Use torch.hann_window identically to v5's adc_to_ra_complex.
        n_adc = x_c.size(-1)
        win = torch.hann_window(n_adc, device=x_c.device,
                                 dtype=x_c.real.dtype).to(x_c.dtype)
        x_c = x_c * win[None, None, :]
    elif range_window == "none":
        pass
    else:
        raise ValueError(f"unknown range_window: {range_window!r}")

    # Range FFT on the last axis.
    rp = torch.fft.fft(x_c, n=x_c.size(-1), dim=-1)                # (12, 16, 256) cx

    # Flatten (tx, rx) → v : v = tx * 16 + rx (matches the renderer's
    # natural (tx, rx, K) → (n_virt=192, n_range=256) layout).
    n_tx, n_rx = rp.size(0), rp.size(1)
    rp_virt = rp.reshape(n_tx * n_rx, rp.size(-1))                 # (192, 256) cx

    return rp_virt


# ---------------------------------------------------------------------------
# Gram-matrix correlation helpers (M1 diagnostic; M2 loss uses these)
# ---------------------------------------------------------------------------

def gram_correlation_per_range(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Per-range-bin normalised Gram correlation.

    ``v_pred`` and ``v_gt`` are ``(..., N_virt, N_range)`` complex. The
    last two axes are (virtual-antenna, range). Any leading batch dims
    (frame, Doppler, etc.) are broadcast over.

    Returns ``(...,  N_range)`` real in ``[0, 1]`` — 1 means the two
    per-virtual vectors at that range bin agree up to a complex scalar
    (i.e. up to absolute phase + amplitude), 0 means orthogonal.

    The scalar per-bin formula is
        g = |v_p^H v_g|² / (‖v_p‖² · ‖v_g‖²)
    evaluated independently at every range bin. See
    ``md/gram_vs_fft_derivation.md`` §10 for the reduction from the
    Frobenius² Gram-matrix distance.
    """
    if v_pred.shape != v_gt.shape:
        raise ValueError(
            f"shape mismatch: v_pred {tuple(v_pred.shape)} vs "
            f"v_gt {tuple(v_gt.shape)}"
        )

    # Hermitian inner product over the virtual axis (dim=-2).
    # ⟨v_p, v_g⟩ = Σ_n conj(v_p[n]) · v_g[n]
    inner = (v_pred.conj() * v_gt).sum(dim=-2)                     # (..., N_range) cx
    inner_mag_sq = inner.abs().pow(2)                              # (..., N_range) real

    norm_p_sq = v_pred.abs().pow(2).sum(dim=-2)                    # (..., N_range)
    norm_g_sq = v_gt.abs().pow(2).sum(dim=-2)                      # (..., N_range)

    denom = (norm_p_sq * norm_g_sq).clamp_min(eps)
    return (inner_mag_sq / denom).clamp(0.0, 1.0)


def gram_correlation_mean(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Average normalised Gram correlation across range (and any other
    leading axes). Returns a 0-D real tensor in ``[0, 1]``."""
    return gram_correlation_per_range(v_pred, v_gt, eps=eps).mean()
