"""v7 radar utility — GT Range-Azimuth-Doppler (RAD) cube pipeline.

Extends the v5 GT |RA| pipeline (``mmir.data.ra_utils.adc_to_ra_complex``)
with a Doppler FFT on the chirp axis. Matches v5's azimuth FFT
convention exactly:

  Per-chirp:  range Hann × range-FFT → txrx_to_vx_chirps → pick el=0
               → azimuth Hann × ifftshift × FFT(n=128) × drop bin[0]
               × fftshift  →  (127, 256) complex.
  Cross-chirp: chirp-axis Hann × ifftshift × FFT(n=N_DOP) × drop bin[0]
               × fftshift  →  (D = N_DOP - 1, 127, 256) complex.

N_DOP = 32 by default (2× zero-pad over the 16 physical chirps),
matching v5's azimuth zero-pad ratio of 128/86 ≈ 1.5×.

Per user directive (2026-04-23): identical step-for-step FFT chain
on the chirp axis as on the azimuth axis. Zero-padding does NOT add
resolution (PDF §V.A); 16/32/64 are all valid sizes (pick 32 as
v5-closest).
"""
from __future__ import annotations

import torch


N_DOP_DEFAULT: int = 32


def _hann(n: int, device, dtype):
    return torch.hann_window(n, device=device, dtype=dtype)


# =========================================================================
# Precomputed batched-chirp el=0 projection (plan speedup §B3)
# =========================================================================
# Replaces the per-chirp Python loop over ``txrx_to_vx_chirps_torch``
# with a single matmul. The (RX, TX) → (ele=0, az_bin) mapping with the
# legacy cumulative-averaging weights is fully determined by
# ``rx_locations`` / ``tx_locations`` from ``mmir.data.ra_utils``. Build
# the 86×192 weight matrix once per device and reuse.

_RX_LOCS = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0),
            (14, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0),
            (51, 0), (52, 0), (53, 0)]
_TX_LOCS = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6),
            (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]

_vx_el0_weights_cache: dict = {}


def _vx_el0_weights(device, dtype) -> torch.Tensor:
    """Return the (86, n_rx·n_tx) complex weight matrix for the el=0
    row of ``txrx_to_vx_chirps_torch``, with the legacy cumulative-
    averaging convention (first write wins at weight 1; each subsequent
    write halves existing coefficients and adds 0.5).

    Flat index into (n_rx·n_tx) is ``rx_id·n_tx + tx_id``, matching the
    permutation used in the caller.
    """
    key = (str(device), str(dtype))
    cached = _vx_el0_weights_cache.get(key)
    if cached is not None:
        return cached

    n_rx, n_tx, n_az = 16, 12, 86
    # Coefficient table: list of (rx, tx, coef) per az bin.
    contribs: list[list[tuple[int, int, float]]] = [[] for _ in range(n_az)]
    for rx_id, rx_loc in enumerate(_RX_LOCS):
        for tx_id, tx_loc in enumerate(_TX_LOCS):
            if tx_loc[1] != 0:
                continue                                    # el != 0
            az = rx_loc[0] + tx_loc[0]
            if not contribs[az]:
                contribs[az].append((rx_id, tx_id, 1.0))
            else:
                contribs[az] = [(r, t, c * 0.5)
                                 for (r, t, c) in contribs[az]]
                contribs[az].append((rx_id, tx_id, 0.5))

    W = torch.zeros((n_az, n_rx * n_tx), dtype=dtype, device=device)
    for az in range(n_az):
        for rx_id, tx_id, coef in contribs[az]:
            W[az, rx_id * n_tx + tx_id] = coef
    _vx_el0_weights_cache[key] = W
    return W


def _batch_txrx_to_vx_el0(rp_stack_ch_tx_rx_r: torch.Tensor) -> torch.Tensor:
    """Batched el=0 projection matching the behaviour of
    ``txrx_to_vx_chirps_torch(...)[0, 0]`` for every chirp at once.

    Input:  ``(CH, TX=12, RX=16, R=256)`` complex.
    Output: ``(CH, 86, R)`` complex.
    """
    assert rp_stack_ch_tx_rx_r.ndim == 4 and rp_stack_ch_tx_rx_r.is_complex(), (
        f'expected (CH, TX, RX, R) complex; got '
        f'{tuple(rp_stack_ch_tx_rx_r.shape)}')
    n_ch, n_tx, n_rx, n_r = rp_stack_ch_tx_rx_r.shape
    # Legacy iteration order: rx outer, tx inner → flat = rx·n_tx + tx.
    rp_flat = rp_stack_ch_tx_rx_r.permute(0, 2, 1, 3).reshape(
        n_ch, n_rx * n_tx, n_r)                                        # (CH, 192, R)
    W = _vx_el0_weights(rp_flat.device, rp_flat.dtype)                 # (86, 192)
    # matmul: (86, 192) · (CH, 192, R) → (CH, 86, R)
    return torch.matmul(W, rp_flat)


def _azimuth_fft_on_vx86(vx86: torch.Tensor) -> torch.Tensor:
    """Apply v5's azimuth FFT chain on a ``(..., 86, R)`` complex input.
    Returns ``(..., 127, R)`` complex.
    """
    assert vx86.shape[-2] == 86, (
        f'expected 86 azimuth bins; got {vx86.shape}')
    n_az = 86
    win = _hann(n_az, vx86.device, vx86.real.dtype).to(vx86.dtype)
    shape = [1] * vx86.ndim
    shape[-2] = n_az
    vx = vx86 * win.view(*shape)
    vx = torch.fft.ifftshift(vx, dim=-2)
    vx = torch.fft.fft(vx, n=128, dim=-2)
    vx = vx.narrow(-2, 1, 127)
    vx = torch.fft.fftshift(vx, dim=-2)
    return vx


def _doppler_fft_on_chirps(stack: torch.Tensor,
                             n_dop: int = N_DOP_DEFAULT,
                             dim: int = 0) -> torch.Tensor:
    """Apply the azimuth-identical FFT chain on the chirp axis."""
    assert stack.is_complex()
    n_ch = stack.shape[dim]
    assert n_dop >= n_ch, f'n_dop ({n_dop}) < n_chirps ({n_ch})'
    win = _hann(n_ch, stack.device, stack.real.dtype).to(stack.dtype)
    shape = [1] * stack.ndim
    shape[dim] = n_ch
    x = stack * win.view(*shape)
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fft(x, n=n_dop, dim=dim)
    x = x.narrow(dim, 1, n_dop - 1)
    x = torch.fft.fftshift(x, dim=dim)
    return x


def adc_to_rad_complex(adc_ri_all_chirps: torch.Tensor,
                        n_dop: int = N_DOP_DEFAULT) -> torch.Tensor:
    """GT path — ADC over all chirps → complex RAD cube.

    Input ``adc_ri_all_chirps``: ``(n_chirps=16, n_tx=12, n_rx=16,
    n_adc=256, 2)`` real-imag float32.

    Output: ``(D = n_dop - 1, 127, n_range)`` complex.
    """
    assert adc_ri_all_chirps.ndim == 5 and adc_ri_all_chirps.size(-1) == 2, (
        f'expected (CH, TX, RX, ADC, 2); got {tuple(adc_ri_all_chirps.shape)}')
    n_ch, n_tx, n_rx, n_adc, _ = adc_ri_all_chirps.shape
    assert (n_ch, n_tx, n_rx, n_adc) == (16, 12, 16, 256), (
        f'unexpected shape {tuple(adc_ri_all_chirps.shape)}')

    x_c = torch.complex(
        adc_ri_all_chirps[..., 0].contiguous(),
        adc_ri_all_chirps[..., 1].contiguous(),
    )                                                               # (CH, TX, RX, ADC)

    win_r = _hann(n_adc, x_c.device, x_c.real.dtype).to(x_c.dtype)
    x_c = x_c * win_r[None, None, None, :]
    rp = torch.fft.fft(x_c, n=n_adc, dim=-1)                        # (CH, TX, RX, R)

    stack86 = _batch_txrx_to_vx_el0(rp)                             # (CH, 86, R)

    ra_stack = _azimuth_fft_on_vx86(stack86)                        # (CH, 127, R)
    rad = _doppler_fft_on_chirps(ra_stack, n_dop=n_dop, dim=0)      # (D, 127, R)
    return rad


def rp_stack_to_rad_complex(rp_stack: torch.Tensor,
                             n_dop: int = N_DOP_DEFAULT) -> torch.Tensor:
    """Pred path — stacked per-chirp renderer output → complex RAD cube.

    Input ``rp_stack``: ``(n_chirps, n_tx=12, n_rx=16, R=256)`` complex
    — e.g. the output of ``render_factorized_doppler``.

    Output: ``(D = n_dop - 1, 127, R)`` complex, matching
    ``adc_to_rad_complex`` step-for-step.
    """
    assert rp_stack.ndim == 4 and rp_stack.is_complex(), (
        f'expected (CH, TX, RX, R) complex; got {tuple(rp_stack.shape)}')

    stack86 = _batch_txrx_to_vx_el0(rp_stack)                       # (CH, 86, R)
    ra_stack = _azimuth_fft_on_vx86(stack86)                        # (CH, 127, R)
    rad = _doppler_fft_on_chirps(ra_stack, n_dop=n_dop, dim=0)      # (D, 127, R)
    return rad
