"""Material reparameterization: raw (optimiser) space <-> physics space.

Replicates the exact transforms from train.py:ParameterManagerSionna so that
materials trained in mm25DGS are directly comparable to mmIR.
"""

import numpy as np
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# ITU default material parameters (physics space)
# [eps_real, eps_imag, sigma_h, l_c, tau, thickness]
# ---------------------------------------------------------------------------
ITU_DEFAULTS = {
    "concrete": np.array([5.31, 0.0326, 1e-4, 5e-3, 0.5, 0.15], dtype=np.float32),
    "brick": np.array([3.75, 0.038, 3e-4, 8e-3, 0.4, 0.10], dtype=np.float32),
    "glass": np.array([6.27, 0.0043, 1e-5, 1e-3, 0.8, 0.006], dtype=np.float32),
    "metal": np.array([1.0, 1e6, 1e-5, 1e-3, 0.9, 0.01], dtype=np.float32),
    "wood": np.array([1.99, 0.012, 5e-4, 1e-2, 0.3, 0.05], dtype=np.float32),
}

# ---------------------------------------------------------------------------
# Raw-space bounds (matching TrainingConfigSionna defaults)
# ---------------------------------------------------------------------------
EPS_IMAG_RAW_UPPER = 4.6      # exp(4.6) ~ 100
SIGMA_H_RAW_UPPER = -2.3      # exp(-2.3) ~ 0.1 m
THICKNESS_RAW_UPPER = -0.7    # exp(-0.7) ~ 0.5 m


def reparameterize(raw: Tensor) -> Tensor:
    """Map unconstrained raw parameters -> bounded physics parameters.

    Args:
        raw: (*, 6) tensor in raw (optimiser) space.

    Returns:
        (*, 6) tensor in physics space:
          col 0  eps_real    [1.5, 10.0]   sigmoid * 8.5 + 1.5
          col 1  eps_imag    [~1e-3, ~100] exp(clamp)
          col 2  sigma_h     [~1e-7, ~0.1] exp(clamp)
          col 3  l_c         [~5e-4, 0.1]  exp(clamp)
          col 4  tau         [0.05, 0.95]  sigmoid * 0.9 + 0.05
          col 5  thickness   [~1e-3, ~0.5] exp(clamp)
    """
    out = torch.empty_like(raw)
    out[..., 0] = 1.5 + 8.5 * torch.sigmoid(raw[..., 0])
    out[..., 1] = torch.exp(torch.clamp(raw[..., 1], -7.0, EPS_IMAG_RAW_UPPER))
    out[..., 2] = torch.exp(torch.clamp(raw[..., 2], -16.0, SIGMA_H_RAW_UPPER))
    out[..., 3] = torch.exp(torch.clamp(raw[..., 3], -7.6, -2.3))
    out[..., 4] = 0.05 + 0.9 * torch.sigmoid(raw[..., 4])
    out[..., 5] = torch.exp(torch.clamp(raw[..., 5], -7.0, THICKNESS_RAW_UPPER))
    return out


def inverse_reparameterize(physics: np.ndarray) -> np.ndarray:
    """Map physics parameters -> raw (optimiser) space.  NumPy, used at init.

    Args:
        physics: (6,) or (N, 6) array in physics space.

    Returns:
        Same shape in raw space.
    """
    raw = np.empty_like(physics)

    def _logit(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(x / (1.0 - x))

    raw[..., 0] = _logit((physics[..., 0] - 1.5) / 8.5)
    raw[..., 1] = np.log(np.clip(physics[..., 1], 1e-3, np.exp(EPS_IMAG_RAW_UPPER)))
    raw[..., 2] = np.log(np.clip(physics[..., 2], 1e-7, np.exp(SIGMA_H_RAW_UPPER)))
    raw[..., 3] = np.log(np.clip(physics[..., 3], 5e-4, 0.1))
    raw[..., 4] = _logit((physics[..., 4] - 0.05) / 0.9)
    raw[..., 5] = np.log(np.clip(physics[..., 5], 1e-3, np.exp(THICKNESS_RAW_UPPER)))
    return raw.astype(np.float32)
