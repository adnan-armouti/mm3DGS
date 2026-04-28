"""MLP-A: pose-conditioned per-Gaussian deformation MLP.

Continuous version of the v5_v4 Phase 4 D matrix. At each train (or test)
pose F, the effective state of Gaussian i is:

    effective_pos_i  = base_pos_i + Δp(pose_F, base_pos_i)
    effective_α_i    = base_α_i  + Δα(pose_F, base_pos_i)

where Δp ∈ R^3 and Δα ∈ R are produced by a small MLP that takes the
pose's 6D representation concatenated with the Gaussian's canonical
(scene-relative) position.

Test-time semantics: the test pose's geometry (rx_center, boresight)
is a known input at inference, same as the test camera pose in NeRF.
The test signal (RA/ADC) is never accessed by the MLP — it's only used
for evaluation. This is the standard NVS convention.

The MLP starts as ~identity (output ≈ 0) by initialising the final
layer's weights with a small scale; combined with the L2 anchor on
Δp during training, this means MLP-A reduces to "M0 init" at iter 0
and only deviates as the loss demands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def encode_pose_6d(rx_center: torch.Tensor,
                    boresight: torch.Tensor,
                    rx_center_seed: torch.Tensor,
                    R_world2radar_seed: torch.Tensor,
                    ) -> torch.Tensor:
    """Encode a pose into a 6D vector relative to the seed pose.

    The first 3 dims are (rx_F − rx_seed) expressed in seed-pose radar
    coords (so the seed pose itself maps to [0,0,0,0,1,0]). The last 3
    dims are the boresight unit-vector also expressed in seed-pose
    radar coords.

    Args:
      rx_center: (3,) world coords of pose F's RX centroid.
      boresight: (3,) world unit vector of pose F's average TX boresight.
      rx_center_seed: (3,) world coords of the seed pose's RX centroid.
      R_world2radar_seed: (3, 3) rotation that maps seed-pose boresight → +Y.
    Returns: (6,) tensor on the same device as rx_center.
    """
    delta = rx_center - rx_center_seed                              # (3,)
    rel_pos = R_world2radar_seed @ delta                            # (3,)
    rel_bs = R_world2radar_seed @ boresight                         # (3,)
    return torch.cat([rel_pos, rel_bs], dim=0)                      # (6,)


class PoseConditionedDeformationMLP(nn.Module):
    """Per-Gaussian (Δposition, Δopacity) field, conditioned on pose_6d.

    Architecture:
      input:    pose_6d (6) + gaussian_pos_local (3) = 9
      hidden:   3 layers × hidden_dim (default 64) + ReLU + LayerNorm
      output:   4 = (Δp_xyz, Δα)

    Δp is bounded via tanh + scale (default 5 cm). Δα is bounded similarly
    (default tanh × 1.0 so it can shut a Gaussian off entirely).

    The final layer is initialized with a small scale so the MLP outputs
    near-zero at iter 0 (acts as identity).
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 max_dpos_m: float = 0.05,
                 max_dalpha: float = 1.0,
                 init_scale: float = 1e-4,
                 ):
        super().__init__()
        self.max_dpos_m = float(max_dpos_m)
        self.max_dalpha = float(max_dalpha)
        in_dim = 6 + 3
        out_dim = 4
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)
        # Near-zero init on the head — MLP starts as identity.
        nn.init.normal_(self.head.weight, std=init_scale)
        nn.init.zeros_(self.head.bias)

    def forward(self, pose_6d: torch.Tensor, pos_local: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-Gaussian (Δposition, Δopacity) for a given pose.

        Args:
          pose_6d:   (6,) tensor — pose encoding shared across all Gaussians.
          pos_local: (N, 3) tensor — Gaussian positions in seed-pose's
                     radar local frame (so they're scene-scale, ~ metres).
        Returns:
          dp:    (N, 3) position deltas in WORLD frame, bounded by max_dpos_m.
          dalpha: (N,) opacity deltas, bounded by max_dalpha.
        """
        N = pos_local.shape[0]
        pose_b = pose_6d.unsqueeze(0).expand(N, -1)                # (N, 6)
        x = torch.cat([pose_b, pos_local], dim=-1)                  # (N, 9)
        h = self.trunk(x)                                            # (N, hidden)
        out = self.head(h)                                           # (N, 4)
        # Tanh-bounded outputs.
        dp_local = torch.tanh(out[:, :3]) * self.max_dpos_m         # (N, 3) in seed-radar frame
        dalpha   = torch.tanh(out[:, 3]) * self.max_dalpha          # (N,)
        return dp_local, dalpha


def encode_pose_from_dict(pose_dict, rx_center_seed_np, R_world2radar_seed_np,
                            device='cuda') -> torch.Tensor:
    """Convenience: build pose_6d for a `train_poses_chirp0`-style pose dict."""
    rx = pose_dict['rx_positions'].mean(dim=0)                     # (3,) torch
    bs = pose_dict['tx_boresights'].mean(dim=0)
    bs = bs / bs.norm().clamp(min=1e-8)
    rx_seed = torch.from_numpy(rx_center_seed_np).to(rx)
    R = torch.from_numpy(R_world2radar_seed_np).to(rx)
    return encode_pose_6d(rx, bs, rx_seed, R)
