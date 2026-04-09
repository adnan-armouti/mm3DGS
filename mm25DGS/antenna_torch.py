"""Native PyTorch antenna pattern evaluation.  No DrJit.

Loads the same .npy pattern files as mmIR (361 samples, columns [E_dB, H_dB]),
stores as PyTorch tensors, evaluates via differentiable linear interpolation
with periodic wrapping.
"""

import math
import numpy as np
import torch
from torch import Tensor


class AntennaPattern:
    """Antenna pattern stored as a PyTorch tensor."""

    def __init__(self, pattern_path: str, device: str = "cuda:0"):
        data = np.load(pattern_path)  # (361, 2) columns [E_dB, H_dB]

        E_lin = np.power(10.0, data[:, 0] / 10.0)
        H_lin = np.power(10.0, data[:, 1] / 10.0)

        # Scaling factor C (matching mmIR element_patterns.py L420-428 exactly):
        #   G_max = 10^(max(dB) / 10)      ← from dB peak, not linear peak
        #   P_max = max(E_linear * H_linear) ← max of element-wise PRODUCT
        #   C = G_max / P_max
        G_max_dB = max(data[:, 0].max(), data[:, 1].max())
        G_max_lin = 10.0 ** (G_max_dB / 10.0)
        P_max = (E_lin * H_lin).max()
        C_scale = G_max_lin / P_max if P_max > 1e-12 else 1.0

        # Store RAW pattern (not scaled). C_scale is applied once in evaluate().
        self.E = torch.from_numpy(E_lin.astype(np.float32)).to(device)  # (361,)
        self.H = torch.from_numpy(H_lin.astype(np.float32)).to(device)  # (361,)
        self.C_scale = C_scale
        self.device = device

    def evaluate(self, directions: Tensor, orientations: Tensor) -> Tensor:
        """Evaluate antenna gain.

        Args:
            directions:   (N, 3) unit vectors (world frame).
            orientations: (N, 3) boresight unit vectors.

        Returns:
            (N,) linear power gain (product of E-plane and H-plane gains).
        """
        # Build antenna-local frame (matching mmIR: element_patterns.py L638-651)
        # Y = boresight (fwd), Z = normalize(Y × aux), X = Z × Y
        y_local = orientations  # (N, 3)

        # Auxiliary vector for Gram-Schmidt (X-axis, unless parallel to boresight)
        aux = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(y_local)
        parallel = torch.abs((y_local * aux).sum(-1)) > 0.99
        if parallel.any():
            alt = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(y_local)
            aux = torch.where(parallel.unsqueeze(-1), alt, aux)

        z_local = torch.cross(y_local, aux, dim=-1)
        z_local = z_local / z_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_local = torch.cross(z_local, y_local, dim=-1)

        # Project direction into local frame [x, y, z]
        d_x = (directions * x_local).sum(-1)   # right
        d_y = (directions * y_local).sum(-1)    # forward (boresight)
        d_z = (directions * z_local).sum(-1)    # up

        # Rename for clarity in angle computation
        d_fwd = d_y
        d_right = d_x
        d_up = d_z

        # E-plane angle: elevation (fwd-up plane)
        # +180° offset: pattern convention has boresight at index 180°
        # (mmIR: element_patterns.py line 689)
        angle_E_deg = torch.rad2deg(torch.atan2(d_up, d_fwd)) + 180.0
        # H-plane angle: azimuth (fwd-right plane)
        angle_H_deg = torch.rad2deg(torch.atan2(d_right, d_fwd)) + 180.0

        gain_E = self._interp(angle_E_deg, self.E)
        gain_H = self._interp(angle_H_deg, self.H)
        return (self.C_scale * gain_E * gain_H).clamp(min=0.0)

    def _interp(self, angle_deg: Tensor, pattern: Tensor) -> Tensor:
        """Differentiable linear interpolation with periodic wrap on [0, 360)."""
        idx_f = angle_deg % 360.0
        idx_lo = idx_f.long() % 360
        idx_hi = (idx_lo + 1) % 360
        frac = idx_f - idx_f.floor()
        return pattern[idx_lo] * (1.0 - frac) + pattern[idx_hi] * frac


# Module-level singletons
_tx_pattern: AntennaPattern = None
_rx_pattern: AntennaPattern = None


def load_patterns(tx_path: str, rx_path: str, device: str = "cuda:0"):
    """Load TX and RX antenna patterns from .npy files (called once)."""
    global _tx_pattern, _rx_pattern
    _tx_pattern = AntennaPattern(tx_path, device)
    _rx_pattern = AntennaPattern(rx_path, device)


def evaluate_tx_gain(directions: Tensor, orientations: Tensor) -> Tensor:
    """(N,) linear TX gain.  direction = TX → Gaussian (outgoing)."""
    return _tx_pattern.evaluate(directions, orientations)


def evaluate_rx_gain(directions: Tensor, orientations: Tensor) -> Tensor:
    """(N,) linear RX gain.  direction = RX → Gaussian (from element toward target).

    Note: mmIR Path 1 passes dir_hit_to_tx (surface→element) for TX, giving
    near-zero TX gain. The product TX×RX is ~0 regardless. This effectively
    disables antenna pattern weighting in both codebases. The negation here
    matches mmIR's effective behavior (one direction reversed → product ≈ 0).
    """
    return _rx_pattern.evaluate(-directions, orientations)
