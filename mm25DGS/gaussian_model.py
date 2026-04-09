"""2.5D Gaussian Surfel model.

Each Gaussian is a flat disc (2D surfel) with:
  - Centre position mu (3,)
  - Rotation quaternion q (4,) -> tangent frame [t1, t2, n]
  - Two lateral log-scales s (2,)  (no scale in normal direction)
  - Logit-opacity alpha (1,)
  - Raw material parameters (6,) using mmIR reparameterisation
  - EM coherence scale sigma_em (1,) — controls coherence factor gamma

The surface normal is intrinsic: n = R(q)[:, 2].
Thickness is modelled as an ITU material parameter, not a geometric axis.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class GaussianModel(nn.Module):
    """Container for all per-Gaussian parameters (contiguous GPU tensors)."""

    def __init__(self, N: int, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.positions = nn.Parameter(torch.zeros(N, 3, device=device))
        self.rotations = nn.Parameter(torch.zeros(N, 4, device=device))
        self.log_scales = nn.Parameter(torch.zeros(N, 2, device=device))
        self.logit_opacities = nn.Parameter(torch.zeros(N, 1, device=device))
        self.raw_materials = nn.Parameter(torch.zeros(N, 6, device=device))
        # EM coherence scale: init to λ/(2π) ≈ 0.62mm at 77 GHz
        self.log_sigma_em = nn.Parameter(
            torch.full((N, 1), math.log(0.62e-3), device=device)
        )

        # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
        with torch.no_grad():
            self.rotations[:, 0] = 1.0

    @property
    def N(self) -> int:
        return self.positions.shape[0]

    # ------------------------------------------------------------------
    #  Derived geometric quantities
    # ------------------------------------------------------------------

    def get_rotation_matrices(self) -> Tensor:
        """Quaternion (N, 4) [w, x, y, z] -> rotation matrix (N, 3, 3).

        Columns of R are [t1, t2, normal].
        """
        q = torch.nn.functional.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros(self.N, 3, 3, device=self.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def get_normals(self) -> Tensor:
        """(N, 3) surfel normals = third column of rotation matrix."""
        R = self.get_rotation_matrices()
        return R[:, :, 2]

    def get_tangent_frame(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (t1, t2, n) each (N, 3)."""
        R = self.get_rotation_matrices()
        return R[:, :, 0], R[:, :, 1], R[:, :, 2]

    def get_scales(self) -> Tensor:
        """(N, 2) positive lateral scales."""
        return torch.exp(self.log_scales)

    def get_opacities(self) -> Tensor:
        """(N, 1) opacities in [0, 1]."""
        return torch.sigmoid(self.logit_opacities)

    def get_sigma_em(self) -> Tensor:
        """(N, 1) EM coherence scale in metres, clamped to [0.01mm, 5mm]."""
        return torch.exp(self.log_sigma_em.clamp(-11.5, -5.3))

    def get_covariance_3d(self) -> Tensor:
        """(N, 3, 3) rank-2 world-space covariance.

        Sigma = s1^2 * t1 @ t1^T  +  s2^2 * t2 @ t2^T
        """
        t1, t2, _ = self.get_tangent_frame()
        s = self.get_scales()
        s1_sq = (s[:, 0] ** 2).unsqueeze(-1).unsqueeze(-1)
        s2_sq = (s[:, 1] ** 2).unsqueeze(-1).unsqueeze(-1)

        t1_outer = t1.unsqueeze(-1) @ t1.unsqueeze(-2)
        t2_outer = t2.unsqueeze(-1) @ t2.unsqueeze(-2)
        return s1_sq * t1_outer + s2_sq * t2_outer

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all parameters to a .pt file."""
        torch.save(
            {
                "positions": self.positions.data,
                "rotations": self.rotations.data,
                "log_scales": self.log_scales.data,
                "logit_opacities": self.logit_opacities.data,
                "raw_materials": self.raw_materials.data,
                "log_sigma_em": self.log_sigma_em.data,
            },
            path,
        )

    def load(self, path: str):
        """Load parameters from a .pt file (must match N)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        with torch.no_grad():
            self.positions.copy_(ckpt["positions"])
            self.rotations.copy_(ckpt["rotations"])
            self.log_scales.copy_(ckpt["log_scales"])
            self.logit_opacities.copy_(ckpt["logit_opacities"])
            self.raw_materials.copy_(ckpt["raw_materials"])
            if "log_sigma_em" in ckpt:
                self.log_sigma_em.copy_(ckpt["log_sigma_em"])
