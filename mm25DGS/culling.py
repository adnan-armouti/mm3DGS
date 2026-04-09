"""Contribution-based active-set selection.

Instead of downsampling the LiDAR cloud, we keep all Gaussians and
dynamically select the top contributors (by estimated amplitude) for
ADC synthesis each iteration.
"""

import torch
from torch import Tensor

from .gaussian_model import GaussianModel


def compute_contribution_estimates(
    model: GaussianModel,
    radar_center: Tensor,      # (3,)
    radar_boresight: Tensor,   # (3,) unit vector
) -> Tensor:
    """Fast per-Gaussian amplitude estimate.  Returns (N,) tensor.

    Uses opacity, normal alignment with radar direction, rough antenna
    gain (cosine from boresight), and 1/r^2 path loss.
    """
    mu = model.positions                              # (N, 3)
    normals = model.get_normals()                      # (N, 3)
    opacities = model.get_opacities().squeeze(-1)      # (N,)

    to_g = mu - radar_center.unsqueeze(0)              # (N, 3)
    dist = torch.norm(to_g, dim=-1).clamp(min=0.1)
    to_g_norm = to_g / dist.unsqueeze(-1)

    # |cos(angle between normal and radar direction)|
    cos_n = torch.abs((normals * to_g_norm).sum(dim=-1))

    # Rough antenna gain: cosine of angle from boresight
    cos_bore = (to_g_norm * radar_boresight.unsqueeze(0)).sum(dim=-1).clamp(min=0)

    return opacities * cos_n * cos_bore / (dist ** 2 + 1e-8)


def select_active_set(
    contributions: Tensor,   # (N,)
    threshold: float = 0.97,
) -> Tensor:
    """Select top-contributing Gaussians covering *threshold* of total amplitude.

    Returns:
        Boolean mask (N,) — True for active Gaussians.
    """
    sorted_vals, sorted_idx = torch.sort(contributions, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    total = cumsum[-1].clamp(min=1e-12)
    cutoff_idx = torch.searchsorted(cumsum, total * threshold)
    cutoff_idx = min(cutoff_idx.item() + 1, contributions.shape[0])

    mask = torch.zeros_like(contributions, dtype=torch.bool)
    mask[sorted_idx[:cutoff_idx]] = True
    return mask
