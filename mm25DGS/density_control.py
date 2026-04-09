"""Adaptive density control: clone, split, and prune Gaussians."""

import torch
from torch import Tensor

from .gaussian_model import GaussianModel
from .config import TrainingConfig


def densify_and_prune(
    model: GaussianModel,
    grad_accum: Tensor,    # (N,) accumulated position gradient magnitudes
    grad_count: Tensor,    # (N,) number of gradient accumulations
    cfg: TrainingConfig,
    radar_center: Tensor,  # (3,)
) -> GaussianModel:
    """Clone small high-grad, split large high-grad, prune low-opacity / out-of-range.

    Returns a *new* GaussianModel with an updated Gaussian count.
    """
    device = model.device

    avg_grad = grad_accum / grad_count.clamp(min=1)
    scales = model.get_scales()                     # (N, 2)
    max_scale = scales.max(dim=-1).values           # (N,)
    opacities = model.get_opacities().squeeze(-1)   # (N,)
    dist = torch.norm(model.positions - radar_center, dim=-1)

    # --- Masks ---
    clone_mask = (avg_grad > cfg.densify_grad_threshold) & (max_scale < 0.02)
    split_mask = (avg_grad > cfg.densify_grad_threshold) & (max_scale >= 0.02)
    prune_mask = (
        (opacities < cfg.prune_opacity_threshold)
        | (dist < cfg.prune_range_min)
        | (dist > cfg.prune_range_max)
    )

    clone_idx = clone_mask.nonzero(as_tuple=True)[0]
    split_idx = split_mask.nonzero(as_tuple=True)[0]
    keep_mask = ~prune_mask & ~split_mask

    N_clone = clone_idx.shape[0]
    N_split = split_idx.shape[0]
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    N_new = keep_idx.shape[0] + N_clone + 2 * N_split

    new_model = GaussianModel(N_new, device=device)

    with torch.no_grad():
        ptr = 0
        n_keep = keep_idx.shape[0]

        # --- Keep ---
        new_model.positions.data[ptr:ptr + n_keep] = model.positions[keep_idx]
        new_model.rotations.data[ptr:ptr + n_keep] = model.rotations[keep_idx]
        new_model.log_scales.data[ptr:ptr + n_keep] = model.log_scales[keep_idx]
        new_model.logit_opacities.data[ptr:ptr + n_keep] = model.logit_opacities[keep_idx]
        new_model.raw_materials.data[ptr:ptr + n_keep] = model.raw_materials[keep_idx]
        ptr += n_keep

        # --- Clone (duplicate + small offset) ---
        if N_clone > 0:
            clone_pos = model.positions[clone_idx].clone()
            clone_pos += torch.randn_like(clone_pos) * 0.001
            new_model.positions.data[ptr:ptr + N_clone] = clone_pos
            new_model.rotations.data[ptr:ptr + N_clone] = model.rotations[clone_idx]
            new_model.log_scales.data[ptr:ptr + N_clone] = model.log_scales[clone_idx]
            new_model.logit_opacities.data[ptr:ptr + N_clone] = model.logit_opacities[clone_idx]
            new_model.raw_materials.data[ptr:ptr + N_clone] = model.raw_materials[clone_idx]
            ptr += N_clone

        # --- Split (two half-scale copies offset along t1) ---
        if N_split > 0:
            t1, _, _ = model.get_tangent_frame()
            t1_split = t1[split_idx]
            offset_mag = max_scale[split_idx].unsqueeze(-1) * 0.5
            log_scale_half = model.log_scales[split_idx] - 0.693  # log(0.5)

            for sign in [1.0, -1.0]:
                pos = model.positions[split_idx].clone() + sign * t1_split * offset_mag
                new_model.positions.data[ptr:ptr + N_split] = pos
                new_model.rotations.data[ptr:ptr + N_split] = model.rotations[split_idx]
                new_model.log_scales.data[ptr:ptr + N_split] = log_scale_half
                new_model.logit_opacities.data[ptr:ptr + N_split] = model.logit_opacities[split_idx]
                new_model.raw_materials.data[ptr:ptr + N_split] = model.raw_materials[split_idx]
                ptr += N_split

    return new_model


def reset_opacities(model: GaussianModel):
    """Reset all opacities to sigmoid^{-1}(0.5) = 0."""
    with torch.no_grad():
        model.logit_opacities.fill_(0.0)
