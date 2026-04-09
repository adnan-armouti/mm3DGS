"""Range-Azimuth-Elevation grid construction and coordinate transforms.

Bin centres are computed identically to
mmir/evaluation/utils/single_view_proc.py:make_angle_grids_np().

Coordinate convention (matching mmIR evaluation):
  x -> azimuth  (sin(az))
  y -> range    (forward / boresight)
  z -> elevation (sin(el))
"""

import torch
from torch import Tensor

from .config import RadarConfig


class RAEGrid:
    """Precomputed Range-Azimuth-Elevation grid for a given radar config."""

    def __init__(
        self,
        radar_cfg: RadarConfig,
        az_fft_size: int = 128,
        el_fft_size: int = 128,
        device: str = "cuda:0",
    ):
        self.device = device

        # --- Range axis ---
        self.N_r = radar_cfg.num_adc_samples
        self.range_res = radar_cfg.range_resolution
        self.range_axis = (
            torch.arange(self.N_r, device=device, dtype=torch.float32)
            * self.range_res
        )

        # --- Azimuth axis (arcsin-spaced, DC removed) ---
        self.N_az_full = az_fft_size
        self.N_az = az_fft_size - 1  # 127 after DC removal
        t_az = (
            torch.arange(
                -az_fft_size // 2 + 1, az_fft_size // 2,
                device=device, dtype=torch.float32,
            )
            * (2.0 / az_fft_size)
        )
        t_az = torch.clamp(t_az, -1.0 + 1e-6, 1.0 - 1e-6)
        self.az_angles = torch.arcsin(t_az)  # (N_az,) radians

        # --- Elevation axis ---
        self.N_el_full = el_fft_size
        self.N_el = el_fft_size - 1
        t_el = (
            torch.arange(
                -el_fft_size // 2 + 1, el_fft_size // 2,
                device=device, dtype=torch.float32,
            )
            * (2.0 / el_fft_size)
        )
        t_el = torch.clamp(t_el, -1.0 + 1e-6, 1.0 - 1e-6)
        self.el_angles = torch.arcsin(t_el)  # (N_el,) radians

    # ------------------------------------------------------------------

    def cartesian_to_rae(self, points: Tensor) -> Tensor:
        """(*, 3) Cartesian -> (*, 3) [range, azimuth, elevation]."""
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-12)
        az = torch.atan2(x, y)
        el = torch.asin(torch.clamp(z / r, -1.0 + 1e-6, 1.0 - 1e-6))
        return torch.stack([r, az, el], dim=-1)

    def compute_jacobian(self, points: Tensor) -> Tensor:
        """Jacobian d(r, az, el)/d(x, y, z) at each point.

        Args:
            points: (N, 3) Cartesian positions.
        Returns:
            (N, 3, 3) Jacobian matrices.
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-12)
        rho = torch.sqrt(x ** 2 + y ** 2 + 1e-12)

        J = torch.zeros(points.shape[0], 3, 3, device=points.device)
        J[:, 0, 0] = x / r
        J[:, 0, 1] = y / r
        J[:, 0, 2] = z / r
        J[:, 1, 0] = y / (rho ** 2)
        J[:, 1, 1] = -x / (rho ** 2)
        # J[:, 1, 2] = 0  (already zero)
        J[:, 2, 0] = -x * z / (r ** 2 * rho)
        J[:, 2, 1] = -y * z / (r ** 2 * rho)
        J[:, 2, 2] = rho / (r ** 2)
        return J
