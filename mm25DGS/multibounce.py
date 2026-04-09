"""Multi-bounce via Gaussian pair interaction (pure PyTorch, no ray tracing).

Stub: pair-based multi-bounce will be implemented in Step 10.
For now, returns zero ADC to allow single-bounce training to proceed.
"""

import torch
from torch import Tensor

from .gaussian_model import GaussianModel
from .config import RadarConfig


def synthesize_multibounce(
    model: GaussianModel,
    radar_cfg: RadarConfig,
    r_max: float = 2.0,
    detach_phase: bool = True,
    chunk_size: int = 1024,
) -> tuple:
    """Two-bounce ADC via Gaussian pair interaction.

    Returns (adc_real, adc_imag) each (N_tx, N_rx, K).

    TODO (Step 10): Implement pair enumeration + two-bounce shading.
    """
    device = model.device
    K = radar_cfg.num_adc_samples
    return (torch.zeros(radar_cfg.n_tx, radar_cfg.n_rx, K, device=device),
            torch.zeros(radar_cfg.n_tx, radar_cfg.n_rx, K, device=device))
