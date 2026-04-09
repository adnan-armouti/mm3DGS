"""Loss functions for mm25DGS training.

Uses the same FFT pipeline (adc_to_ra_complex) as mmIR so that
windowing, zero-padding, and sidelobes are modelled identically.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple

from mmir.data.ra_utils import adc_to_ra_complex


def compute_loss(
    adc_real: Tensor,       # (N_tx, N_rx, K)
    adc_imag: Tensor,       # (N_tx, N_rx, K)
    gt_adc_ri: Tensor,      # (N_tx, N_rx, K, 2) real/imag
    w_ra_mag: float = 1.0,
    w_adc_mag: float = 0.0,
    w_phase: float = 0.0,
    ra_use_log: bool = True,
    log_epsilon: float = 1e-6,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute training loss.

    Pipeline:
      1. Stack rendered ADC into (N_tx, N_rx, K, 2).
      2. Run identical FFT to RA domain on rendered and GT.
      3. RA magnitude loss (primary).
      4. Optional ADC magnitude and phase losses.

    Returns:
        (total_loss, loss_dict).
    """
    device = adc_real.device
    rendered_ri = torch.stack([adc_real, adc_imag], dim=-1)  # (N_tx, N_rx, K, 2)

    # FFT to RA domain (differentiable via torch.fft)
    ra_rendered = adc_to_ra_complex(rendered_ri)   # (127, 256) complex
    ra_gt = adc_to_ra_complex(gt_adc_ri)           # (127, 256) complex

    loss_dict: Dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # --- RA magnitude loss ---
    if w_ra_mag > 0:
        ra_rend_mag = torch.abs(ra_rendered)
        ra_gt_mag = torch.abs(ra_gt)

        if ra_use_log:
            ra_rend_mag = torch.log(log_epsilon + ra_rend_mag)
            ra_gt_mag = torch.log(log_epsilon + ra_gt_mag)

        ra_rend_norm = _minmax_normalize(ra_rend_mag)
        ra_gt_norm = _minmax_normalize(ra_gt_mag)

        ra_loss = torch.mean((ra_rend_norm - ra_gt_norm) ** 2)
        total_loss = total_loss + w_ra_mag * ra_loss
        loss_dict["ra_mag"] = ra_loss.item()

    # --- ADC magnitude loss ---
    if w_adc_mag > 0:
        adc_rend_mag = torch.sqrt(adc_real ** 2 + adc_imag ** 2)
        adc_gt_mag = torch.sqrt(gt_adc_ri[..., 0] ** 2 + gt_adc_ri[..., 1] ** 2)
        adc_rend_norm = _minmax_normalize(adc_rend_mag)
        adc_gt_norm = _minmax_normalize(adc_gt_mag)
        adc_loss = torch.mean((adc_rend_norm - adc_gt_norm) ** 2)
        total_loss = total_loss + w_adc_mag * adc_loss
        loss_dict["adc_mag"] = adc_loss.item()

    # --- Phase loss (unit phasor, magnitude-weighted) ---
    if w_phase > 0:
        ra_rend_unit = ra_rendered / (torch.abs(ra_rendered) + 1e-10)
        ra_gt_unit = ra_gt / (torch.abs(ra_gt) + 1e-10)
        weights = torch.abs(ra_gt)
        weights = weights / (weights.max() + 1e-10)
        phase_err = torch.abs(ra_rend_unit - ra_gt_unit) ** 2
        phase_loss = (weights * phase_err).mean()
        total_loss = total_loss + w_phase * phase_loss
        loss_dict["phase"] = phase_loss.item()

    loss_dict["total"] = total_loss.item()
    return total_loss, loss_dict


def _minmax_normalize(x: Tensor) -> Tensor:
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-30:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)
