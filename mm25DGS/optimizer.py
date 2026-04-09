"""Per-group Adam optimizer with RMS gradient clipping.

Mirrors mmIR's SimpleAdamSionna but operates on PyTorch nn.Parameters.
"""

import torch
from .gaussian_model import GaussianModel
from .config import TrainingConfig


class PerGroupAdam:
    """Adam with per-parameter-group LR, gradient clipping, and material sub-LR scales."""

    def __init__(self, model: GaussianModel, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = model.device
        self.step_count = 0
        self.groups = self._build_groups(model, cfg)

    def _build_groups(self, model: GaussianModel, cfg: TrainingConfig) -> list:
        mat_lr_scales = torch.tensor(cfg.material_lr_scales, device=self.device)
        return [
            {"name": "positions", "params": [model.positions],
             "lr": cfg.lr_positions, "clip": cfg.clip_positions},
            {"name": "rotations", "params": [model.rotations],
             "lr": cfg.lr_rotations, "clip": cfg.clip_rotations},
            {"name": "scales", "params": [model.log_scales],
             "lr": cfg.lr_scales, "clip": cfg.clip_scales},
            {"name": "opacities", "params": [model.logit_opacities],
             "lr": cfg.lr_opacities, "clip": cfg.clip_opacities},
            {"name": "materials", "params": [model.raw_materials],
             "lr": cfg.lr_materials, "clip": cfg.clip_materials,
             "lr_scales": mat_lr_scales},
            {"name": "sigma_em", "params": [model.log_sigma_em],
             "lr": cfg.lr_sigma_em, "clip": cfg.clip_sigma_em},
        ]

    def setup(self):
        """Initialise Adam state (m, v) for all parameters."""
        for group in self.groups:
            group["m"] = [torch.zeros_like(p) for p in group["params"]]
            group["v"] = [torch.zeros_like(p) for p in group["params"]]

    def step(
        self,
        lr_scale: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """One Adam step with per-group RMS clipping."""
        self.step_count += 1
        t = self.step_count

        for group in self.groups:
            lr = group["lr"] * lr_scale
            clip_val = group["clip"]
            col_scales = group.get("lr_scales", None)

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                g = p.grad.data.clone()

                # NaN safety
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

                # RMS gradient clipping
                rms = torch.sqrt(torch.mean(g ** 2))
                if rms > clip_val:
                    g = g * (clip_val / rms)

                # Per-column LR scaling (for materials: [0.3, 0.5, ...])
                if col_scales is not None and g.dim() == 2:
                    g = g * col_scales.unsqueeze(0)

                m = group["m"][i]
                v = group["v"][i]
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                p.data.add_(m_hat / (torch.sqrt(v_hat) + eps), alpha=-lr)

    def zero_grad(self):
        for group in self.groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()
