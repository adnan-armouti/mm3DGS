"""DrJit-compatible RA loss with gradient flow through PyTorch FFT operations.

This module provides a differentiable RA loss that:
1. Accepts DrJit tensor inputs
2. Internally uses PyTorch FFT for RA conversion
3. Properly propagates gradients back to DrJit tensors using custom backward

This enables gradient-based optimization of RA image structure.
"""

import torch
import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Tuple, Optional, Dict

from mmir.data.ra_utils import adc_to_ra_image, adc_to_ra_complex
from mmir.losses.loss_utils import compute_ssim_loss, circular_phase_loss, relative_phase_loss, relative_phase_angle_loss


class RALossWithGradients:
    """Stateful wrapper for RA loss computation with gradient caching.

    This class:
    - Computes RA loss using PyTorch FFT (forward)
    - Caches PyTorch gradients during forward pass
    - Provides gradients to DrJit during backward pass
    """

    def __init__(self):
        self.grad_real_cache = None
        self.grad_imag_cache = None

    def compute_forward_and_cache_gradients(
        self,
        rendered_adc_real: mi.Float,
        rendered_adc_imag: mi.Float,
        gt_adc_full: torch.Tensor,
        NT: int,
        NR: int,
        K: int,
        normalize: bool = True
    ) -> mi.Float:
        """Compute RA loss and cache gradients for backward pass.

        Args:
            rendered_adc_real: DrJit tensor of real ADC values (NT*NR*K,)
            rendered_adc_imag: DrJit tensor of imag ADC values (NT*NR*K,)
            gt_adc_full: Ground truth ADC tensor (NT, NR, K, 2)
            NT: Number of TX elements
            NR: Number of RX elements
            K: Number of ADC samples
            normalize: Whether to normalize RA images separately

        Returns:
            DrJit scalar loss value (with custom gradient support)
        """
        # Step 1: Extract current ADC values from DrJit (this evaluates them)
        # We need the values to compute loss, gradients will be handled separately
        with dr.suspend_grad():
            adc_real_np = np.array(rendered_adc_real).reshape(NT, NR, K)
            adc_imag_np = np.array(rendered_adc_imag).reshape(NT, NR, K)

        # Step 2: Create PyTorch tensors with gradient tracking
        adc_real_torch = torch.from_numpy(adc_real_np.copy()).float()
        adc_imag_torch = torch.from_numpy(adc_imag_np.copy()).float()

        adc_real_torch.requires_grad = True
        adc_imag_torch.requires_grad = True

        # Step 3: Assemble full ADC tensor for RA conversion
        rendered_adc_full = torch.stack([adc_real_torch, adc_imag_torch], dim=-1)

        # Step 4: Compute complex RA using PyTorch FFT
        ra_rendered_complex = adc_to_ra_complex(rendered_adc_full)
        ra_gt_complex = adc_to_ra_complex(gt_adc_full)

        # Extract real and imaginary parts
        ra_rendered_real = ra_rendered_complex.real
        ra_rendered_imag = ra_rendered_complex.imag
        ra_gt_real = ra_gt_complex.real
        ra_gt_imag = ra_gt_complex.imag

        # Step 5: Compute magnitudes for loss
        ra_rendered_mag = torch.abs(ra_rendered_complex)
        ra_gt_mag = torch.abs(ra_gt_complex)

        # Step 6: MIN-MAX normalization if requested (matches inspect_ra_coir.py)
        if normalize:
            eps = 1e-12

            # Rendered RA: min-max normalize magnitude
            ra_rendered_min = torch.min(ra_rendered_mag)
            ra_rendered_max = torch.max(ra_rendered_mag)
            ra_rendered_range = torch.maximum(ra_rendered_max - ra_rendered_min, torch.tensor(eps))
            ra_rendered_mag_norm = (ra_rendered_mag - ra_rendered_min) / ra_rendered_range

            # GT RA: min-max normalize magnitude
            ra_gt_min = torch.min(ra_gt_mag)
            ra_gt_max = torch.max(ra_gt_mag)
            ra_gt_range = torch.maximum(ra_gt_max - ra_gt_min, torch.tensor(eps))
            ra_gt_mag_norm = (ra_gt_mag - ra_gt_min) / ra_gt_range
        else:
            ra_rendered_mag_norm = ra_rendered_mag
            ra_gt_mag_norm = ra_gt_mag

        # Step 7: Compute L2 loss on magnitude only (decoupled from phase)
        # NOTE: Previous version used complex L2 (diff_real^2 + diff_imag^2)
        # which created coupling where magnitude changes affected phase gradients.
        ra_loss_torch = torch.mean((ra_rendered_mag_norm - ra_gt_mag_norm) ** 2)

        # Step 8: Compute PyTorch gradients and cache them
        ra_loss_torch.backward()

        # Cache gradients for use in DrJit backward pass
        self.grad_real_cache = adc_real_torch.grad.numpy().flatten().astype(np.float32)
        self.grad_imag_cache = adc_imag_torch.grad.numpy().flatten().astype(np.float32)

        # Step 9: Create DrJit scalar with custom gradient
        loss_value = float(ra_loss_torch.item())
        loss_drjit = mi.Float(loss_value)

        # Attach custom backward using DrJit's enqueue mechanism
        # We'll handle gradients through the combined loss backward

        return loss_drjit

    def get_gradients_for_drjit(self) -> Tuple[mi.Float, mi.Float]:
        """Get cached gradients as DrJit tensors.

        Returns:
            (grad_real, grad_imag): DrJit gradient tensors
        """
        if self.grad_real_cache is None or self.grad_imag_cache is None:
            raise RuntimeError("Gradients not computed yet. Call compute_forward_and_cache_gradients first.")

        grad_real_drjit = mi.Float(self.grad_real_cache)
        grad_imag_drjit = mi.Float(self.grad_imag_cache)

        return grad_real_drjit, grad_imag_drjit


def compute_ra_loss_with_gradients(
    rendered_adc_real: mi.Float,
    rendered_adc_imag: mi.Float,
    gt_adc_full: torch.Tensor,
    NT: int,
    NR: int,
    K: int,
    normalize: bool = True,
    weight: float = 1.0,
    ssim_weight: float = 0.0,
    lpips_weight: float = 0.0,
    lpips_model = None,
    ra_use_log: bool = False,
    ra_loss_type: str = "l2",
    log_epsilon: float = 1e-6,
    log_scale: float = 1.0,
    # ADC phase loss parameters
    adc_phase_weight: float = 0.0,
    adc_phase_loss_type: str = "none",
    phase_magnitude_weighting: bool = True,
    phase_magnitude_power: float = 0.5,
    phase_reference_channel: int = 0,
    phase_use_gt_magnitude: bool = True,
    phase_magnitude_clip_max: float = 0.0,
    # Normalization mode
    use_joint_normalization: bool = False  # If True, both normalized by GT max (preserves scale for gain learning)
) -> Tuple[mi.Float, mi.Float, mi.Float, Optional[Dict[str, float]]]:
    """Compute RA and ADC phase losses with gradient flow back to DrJit tensors.

    This function uses PyTorch's FFT to compute the RA loss, then manually
    injects the gradients into the DrJit computation graph using backward_from().

    All loss terms share the same ADC/RA computation to ensure consistency.

    Args:
        rendered_adc_real: DrJit tensor of real ADC values (NT*NR*K,)
        rendered_adc_imag: DrJit tensor of imag ADC values (NT*NR*K,)
        gt_adc_full: Ground truth ADC tensor (NT, NR, K, 2)
        NT: Number of TX elements
        NR: Number of RX elements
        K: Number of ADC samples
        normalize: Whether to normalize RA images separately
        weight: Weight for RA magnitude loss (gradient scaling)
        ssim_weight: Weight for SSIM loss (0.0 = disabled)
        lpips_weight: Weight for LPIPS perceptual loss (0.0 = disabled)
        lpips_model: LPIPS model instance (required if lpips_weight > 0)
        ra_use_log: If True, apply log-compression to RA magnitude before loss
        ra_loss_type: "l1" | "l2" | "ssim" - loss type on (optionally log-compressed) magnitude
        log_epsilon: Epsilon for log compression: log(eps + scale * |x|)
        log_scale: Scale factor for log compression
        adc_phase_weight: Weight for ADC phase loss (0.0 = disabled)
        adc_phase_loss_type: "none" | "circular" | "relative_circular"
        phase_magnitude_weighting: Weight phase loss by target magnitude
        phase_magnitude_power: Exponent for magnitude weighting (0.5 = sqrt, 1.0 = linear)
        phase_reference_channel: Reference channel for relative phase (-1 for mean)
        phase_use_gt_magnitude: If True, use GT magnitude for weighting (recommended).
                               If False, use predicted magnitude (original behavior).
        phase_magnitude_clip_max: Clip magnitude weights to this value (0 = no clipping).
                                 Helps ignore phase from extreme outlier samples.

    Returns:
        (total_loss, grad_real, grad_imag, loss_dict): Loss value, gradients, and component breakdown
        loss_dict contains: {'ra_l2': float, 'ra_ssim_loss': float, 'ra_lpips_loss': float,
                            'adc_phase_loss': float, 'ra_use_log': bool}
    """
    # Step 1: Extract current values (without breaking gradient graph)
    with dr.suspend_grad():
        adc_real_np = np.array(rendered_adc_real).reshape(NT, NR, K)
        adc_imag_np = np.array(rendered_adc_imag).reshape(NT, NR, K)

    # Step 2: Create PyTorch tensors with gradient tracking
    adc_real_torch = torch.from_numpy(adc_real_np.copy()).float()
    adc_imag_torch = torch.from_numpy(adc_imag_np.copy()).float()

    adc_real_torch.requires_grad = True
    adc_imag_torch.requires_grad = True

    # Step 3: Compute complex RA using PyTorch FFT (SHARED across all loss terms)
    rendered_adc_full = torch.stack([adc_real_torch, adc_imag_torch], dim=-1)

    ra_rendered_complex = adc_to_ra_complex(rendered_adc_full)  # Complex tensor (127, 256)
    ra_gt_complex = adc_to_ra_complex(gt_adc_full)  # Complex tensor (127, 256)

    # Compute magnitude for SSIM/LPIPS (before normalization for consistency)
    ra_rendered_mag = torch.abs(ra_rendered_complex)
    ra_gt_mag = torch.abs(ra_gt_complex)

    # Extract real and imaginary parts for L2 loss
    ra_rendered_real = ra_rendered_complex.real
    ra_rendered_imag = ra_rendered_complex.imag
    ra_gt_real = ra_gt_complex.real
    ra_gt_imag = ra_gt_complex.imag

    # MIN-MAX normalization (matches inspect_ra_coir.py)
    # Two modes: separate (each normalized independently) or joint (using GT range)
    eps = 1e-12
    if normalize:
        if use_joint_normalization:
            # JOINT normalization: both normalized by GT range
            # This preserves scale information for gain learning
            ra_gt_min = torch.min(ra_gt_mag)
            ra_gt_max = torch.max(ra_gt_mag)
            ra_gt_range = torch.maximum(ra_gt_max - ra_gt_min, torch.tensor(eps))

            # Min-max normalize both using GT range
            ra_rendered_mag_norm = (ra_rendered_mag - ra_gt_min) / ra_gt_range
            ra_gt_mag_norm = (ra_gt_mag - ra_gt_min) / ra_gt_range

            # Scale real/imag to preserve phase
            ra_rendered_scale = ra_rendered_mag_norm / torch.maximum(ra_rendered_mag, torch.tensor(eps))
            ra_rendered_real = ra_rendered_real * ra_rendered_scale
            ra_rendered_imag = ra_rendered_imag * ra_rendered_scale

            ra_gt_scale = ra_gt_mag_norm / torch.maximum(ra_gt_mag, torch.tensor(eps))
            ra_gt_real = ra_gt_real * ra_gt_scale
            ra_gt_imag = ra_gt_imag * ra_gt_scale
        else:
            # SEPARATE normalization: each normalized by its own min-max range
            # Scale information is lost (gain cancels out)
            ra_rendered_min = torch.min(ra_rendered_mag)
            ra_rendered_max = torch.max(ra_rendered_mag)
            ra_rendered_range = torch.maximum(ra_rendered_max - ra_rendered_min, torch.tensor(eps))
            ra_rendered_mag_norm = (ra_rendered_mag - ra_rendered_min) / ra_rendered_range

            ra_gt_min = torch.min(ra_gt_mag)
            ra_gt_max = torch.max(ra_gt_mag)
            ra_gt_range = torch.maximum(ra_gt_max - ra_gt_min, torch.tensor(eps))
            ra_gt_mag_norm = (ra_gt_mag - ra_gt_min) / ra_gt_range

            # Scale real/imag to preserve phase
            ra_rendered_scale = ra_rendered_mag_norm / torch.maximum(ra_rendered_mag, torch.tensor(eps))
            ra_rendered_real = ra_rendered_real * ra_rendered_scale
            ra_rendered_imag = ra_rendered_imag * ra_rendered_scale

            ra_gt_scale = ra_gt_mag_norm / torch.maximum(ra_gt_mag, torch.tensor(eps))
            ra_gt_real = ra_gt_real * ra_gt_scale
            ra_gt_imag = ra_gt_imag * ra_gt_scale
    else:
        ra_rendered_mag_norm = ra_rendered_mag
        ra_gt_mag_norm = ra_gt_mag

    # Initialize total loss and loss dict
    total_loss = torch.tensor(0.0, requires_grad=True)
    loss_dict = {'ra_l2': 0.0, 'ra_ssim_loss': 0.0, 'ra_lpips_loss': 0.0, 'adc_phase_loss': 0.0,
                 'ra_use_log': ra_use_log, 'ra_loss_type': ra_loss_type, 'adc_phase_loss_type': adc_phase_loss_type}

    # Compute RA magnitude loss based on configuration
    if weight > 0:
        if ra_use_log:
            # Log-compressed magnitude loss: log(eps + scale * |RA|)
            ra_rendered_log = torch.log(log_epsilon + log_scale * ra_rendered_mag_norm)
            ra_gt_log = torch.log(log_epsilon + log_scale * ra_gt_mag_norm)

            if ra_loss_type == "l1":
                ra_mag_loss = torch.mean(torch.abs(ra_rendered_log - ra_gt_log))
            elif ra_loss_type == "ssim":
                # SSIM on log-compressed magnitude
                # Normalize to [0, 1] for SSIM
                max_val = torch.max(ra_gt_log.max(), ra_rendered_log.max())
                min_val = torch.min(ra_gt_log.min(), ra_rendered_log.min())
                range_val = max_val - min_val + 1e-12
                ra_rendered_ssim = (ra_rendered_log - min_val) / range_val
                ra_gt_ssim = (ra_gt_log - min_val) / range_val
                ra_mag_loss = compute_ssim_loss(ra_rendered_ssim, ra_gt_ssim)
            else:  # l2
                ra_mag_loss = torch.mean((ra_rendered_log - ra_gt_log) ** 2)
        else:
            # Standard (non-log) loss on complex RA or magnitude
            if ra_loss_type == "l1":
                # L1 on magnitude only
                ra_mag_loss = torch.mean(torch.abs(ra_rendered_mag_norm - ra_gt_mag_norm))
            elif ra_loss_type == "ssim":
                # SSIM on magnitude
                ra_mag_loss = compute_ssim_loss(ra_rendered_mag_norm, ra_gt_mag_norm)
            else:  # l2 - magnitude-only L2 loss (decoupled from phase)
                # NOTE: Previous version used complex L2 (diff_real^2 + diff_imag^2)
                # which created coupling where magnitude changes affected phase gradients.
                # Now using magnitude-only L2 for proper decoupling.
                ra_mag_loss = torch.mean((ra_rendered_mag_norm - ra_gt_mag_norm) ** 2)

        total_loss = total_loss + weight * ra_mag_loss
        loss_dict['ra_l2'] = float(ra_mag_loss.item())  # Keep key name for compatibility

    # Additional SSIM loss on magnitude images (separate from ra_loss_type)
    if ssim_weight > 0:
        ssim_loss = compute_ssim_loss(ra_rendered_mag_norm, ra_gt_mag_norm)
        total_loss = total_loss + ssim_weight * ssim_loss
        loss_dict['ra_ssim_loss'] = float(ssim_loss.item())

    # LPIPS loss on magnitude images
    if lpips_weight > 0 and lpips_model is not None:
        # Prepare for LPIPS: (1, 3, H, W) in [0, 1] range
        ra_pred_lpips = ra_rendered_mag_norm.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        ra_gt_lpips = ra_gt_mag_norm.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

        # Move to LPIPS model device
        device = next(lpips_model.parameters()).device
        ra_pred_lpips = ra_pred_lpips.to(device)
        ra_gt_lpips = ra_gt_lpips.to(device)

        lpips_loss = lpips_model(ra_pred_lpips, ra_gt_lpips).mean()
        total_loss = total_loss + lpips_weight * lpips_loss
        loss_dict['ra_lpips_loss'] = float(lpips_loss.item())

    # ADC phase loss (computed in ADC space, not RA space)
    if adc_phase_weight > 0 and adc_phase_loss_type != "none":
        # ADC tensors are already (NT, NR, K) shape
        gt_adc_real = gt_adc_full[..., 0]  # (NT, NR, K)
        gt_adc_imag = gt_adc_full[..., 1]  # (NT, NR, K)

        if adc_phase_loss_type == "circular":
            # Simple circular phase loss on each sample
            phase_loss = circular_phase_loss(
                pred=torch.stack([adc_real_torch, adc_imag_torch], dim=-1),
                target=torch.stack([gt_adc_real, gt_adc_imag], dim=-1),
                magnitude_weighting=phase_magnitude_weighting,
                magnitude_power=phase_magnitude_power,
                use_gt_magnitude=phase_use_gt_magnitude,
                magnitude_clip_max=phase_magnitude_clip_max
            )
        elif adc_phase_loss_type == "relative_circular":
            # Relative phase loss across RX channels (key for MIMO beamforming)
            phase_loss = relative_phase_loss(
                pred=torch.stack([adc_real_torch, adc_imag_torch], dim=-1),
                target=torch.stack([gt_adc_real, gt_adc_imag], dim=-1),
                reference_channel=phase_reference_channel,
                magnitude_weighting=phase_magnitude_weighting,
                magnitude_power=phase_magnitude_power,
                use_gt_magnitude=phase_use_gt_magnitude,
                magnitude_clip_max=phase_magnitude_clip_max
            )
        elif adc_phase_loss_type == "relative_angle":
            # NEW: Angle-based relative phase loss - matches metric computation exactly
            # Uses MSE on actual phase angles (not unit phasors) with magnitude threshold
            phase_loss = relative_phase_angle_loss(
                pred=torch.stack([adc_real_torch, adc_imag_torch], dim=-1),
                target=torch.stack([gt_adc_real, gt_adc_imag], dim=-1),
                reference_channel=phase_reference_channel,
                magnitude_threshold=0.1,  # Match metric: only samples > 10% of max mag
                loss_type="mse"
            )
        else:
            phase_loss = torch.tensor(0.0)

        total_loss = total_loss + adc_phase_weight * phase_loss
        loss_dict['adc_phase_loss'] = float(phase_loss.item())

    # Step 4: Compute gradients (combined from all loss terms)
    total_loss.backward()

    grad_real_np = adc_real_torch.grad.numpy().flatten().astype(np.float32)
    grad_imag_np = adc_imag_torch.grad.numpy().flatten().astype(np.float32)

    # Step 5: Convert to DrJit tensors
    loss_drjit = mi.Float(float(total_loss.item()))
    grad_real_drjit = mi.Float(grad_real_np)
    grad_imag_drjit = mi.Float(grad_imag_np)

    return loss_drjit, grad_real_drjit, grad_imag_drjit, loss_dict


def compute_ra_magnitude_and_phase_loss_with_gradients(
    rendered_adc_real: mi.Float,
    rendered_adc_imag: mi.Float,
    gt_adc_full: torch.Tensor,
    NT: int,
    NR: int,
    K: int,
    weight_ra_mag: float = 1.0,
    weight_phase: float = 1.0
) -> Tuple[mi.Float, mi.Float, mi.Float, dict]:
    """Compute RA magnitude + ADC relative phase loss with proper gradient flow.

    This loss function focuses on:
    1. RA magnitude matching (structure in RA space, magnitude only)
    2. ADC relative phase matching (phase relationships in ADC space)

    Both losses use SEPARATE normalization (GT normalized by GT, rendered by rendered).

    Args:
        rendered_adc_real: DrJit tensor of real ADC values (NT*NR*K,)
        rendered_adc_imag: DrJit tensor of imag ADC values (NT*NR*K,)
        gt_adc_full: Ground truth ADC tensor (NT, NR, K, 2)
        NT: Number of TX elements
        NR: Number of RX elements
        K: Number of ADC samples
        weight_ra_mag: Weight for RA magnitude loss
        weight_phase: Weight for ADC relative phase loss

    Returns:
        (total_loss, grad_real, grad_imag, loss_dict): Total loss, gradients, and loss components
    """
    # Step 1: Extract current values (without breaking gradient graph)
    with dr.suspend_grad():
        adc_real_np = np.array(rendered_adc_real).reshape(NT, NR, K)
        adc_imag_np = np.array(rendered_adc_imag).reshape(NT, NR, K)

    # Step 2: Create PyTorch tensors with gradient tracking
    adc_real_torch = torch.from_numpy(adc_real_np.copy()).float()
    adc_imag_torch = torch.from_numpy(adc_imag_np.copy()).float()

    adc_real_torch.requires_grad = True
    adc_imag_torch.requires_grad = True

    # Ground truth
    gt_adc_real = gt_adc_full[..., 0]  # (NT, NR, K)
    gt_adc_imag = gt_adc_full[..., 1]  # (NT, NR, K)

    # ========================================
    # Loss 1: RA Magnitude Matching
    # ========================================

    # Convert ADC to complex RA using FFT
    rendered_adc_ri = torch.stack([adc_real_torch, adc_imag_torch], dim=-1)  # (NT, NR, K, 2)
    gt_adc_ri = torch.stack([gt_adc_real, gt_adc_imag], dim=-1)  # (NT, NR, K, 2)

    ra_rendered_complex = adc_to_ra_complex(rendered_adc_ri)  # (127, 256) complex
    ra_gt_complex = adc_to_ra_complex(gt_adc_ri)  # (127, 256) complex

    # Take MAGNITUDE only (discard phase of RA)
    ra_rendered_mag = torch.abs(ra_rendered_complex)  # (127, 256) real
    ra_gt_mag = torch.abs(ra_gt_complex)  # (127, 256) real

    # MIN-MAX normalization using SHARED range (preserves spatial alignment and relative magnitudes)
    # This forces the optimizer to match both structure AND spatial placement
    eps = 1e-12

    # Compute shared min/max across both
    shared_min = torch.min(torch.min(ra_rendered_mag), torch.min(ra_gt_mag))
    shared_max = torch.max(torch.max(ra_rendered_mag), torch.max(ra_gt_mag))
    shared_range = torch.maximum(shared_max - shared_min, torch.tensor(eps))

    # Min-max normalize both using shared range
    ra_rendered_mag_norm = (ra_rendered_mag - shared_min) / shared_range
    ra_gt_mag_norm = (ra_gt_mag - shared_min) / shared_range

    # L2 loss on min-max normalized RA magnitude (using shared range)
    # Now penalizes both wrong intensity AND wrong location
    loss_ra_mag = torch.mean((ra_rendered_mag_norm - ra_gt_mag_norm) ** 2)

    # ========================================
    # Loss 2: ADC Relative Phase Matching
    # ========================================

    # Sum over TX (coherent accumulation)
    gt_adc_real_sum = gt_adc_real.sum(dim=0)  # (NR, K)
    gt_adc_imag_sum = gt_adc_imag.sum(dim=0)  # (NR, K)
    rendered_adc_real_sum = adc_real_torch.sum(dim=0)  # (NR, K)
    rendered_adc_imag_sum = adc_imag_torch.sum(dim=0)  # (NR, K)

    # Complex ADC (NR, K)
    adc_gt_complex = torch.complex(gt_adc_real_sum, gt_adc_imag_sum)
    adc_rendered_complex = torch.complex(rendered_adc_real_sum, rendered_adc_imag_sum)

    # Normalize magnitude (keep only phase information)
    # This removes amplitude scale, keeping only phase relationships
    adc_gt_phase_only = adc_gt_complex / (torch.abs(adc_gt_complex) + 1e-12)
    adc_rendered_phase_only = adc_rendered_complex / (torch.abs(adc_rendered_complex) + 1e-12)

    # Remove global phase offset per RX channel (relative phase only)
    # Multiply by conjugate of first sample to set reference phase = 0
    ref_gt = adc_gt_phase_only[:, 0:1]  # (NR, 1)
    ref_rendered = adc_rendered_phase_only[:, 0:1]  # (NR, 1)

    adc_gt_relative_phase = adc_gt_phase_only * torch.conj(ref_gt)
    adc_rendered_relative_phase = adc_rendered_phase_only * torch.conj(ref_rendered)

    # L2 loss on relative phase structure
    # |a - b|^2 for complex numbers measures phase difference
    loss_phase = torch.mean(torch.abs(adc_rendered_relative_phase - adc_gt_relative_phase) ** 2)

    # ========================================
    # Combine losses with equal contribution
    # ========================================

    # Store raw loss values for logging
    loss_ra_mag_val = float(loss_ra_mag.item())
    loss_phase_val = float(loss_phase.item())

    # DISABLED AUTO-BALANCING - it was making gradients too small!
    # Just use the user-specified weights directly
    effective_weight_ra = weight_ra_mag
    effective_weight_phase = weight_phase

    # Apply weights and combine
    total_loss = effective_weight_ra * loss_ra_mag + effective_weight_phase * loss_phase

    # Step 4: Compute gradients
    total_loss.backward()

    grad_real_np = adc_real_torch.grad.numpy().flatten().astype(np.float32)
    grad_imag_np = adc_imag_torch.grad.numpy().flatten().astype(np.float32)

    # Step 5: Convert to DrJit tensors
    total_loss_val = float(total_loss.item())
    loss_drjit = mi.Float(total_loss_val)
    grad_real_drjit = mi.Float(grad_real_np)
    grad_imag_drjit = mi.Float(grad_imag_np)

    # Return loss components for logging
    loss_dict = {
        'total': total_loss_val,
        'ra_magnitude': loss_ra_mag_val,
        'adc_phase': loss_phase_val,
        'weight_ra_mag': float(effective_weight_ra),
        'weight_phase': float(effective_weight_phase),
        'ra_mag_max_gt': float(ra_gt_max.item()),
        'ra_mag_max_rendered': float(ra_rendered_max.item())
    }

    return loss_drjit, grad_real_drjit, grad_imag_drjit, loss_dict


def compute_ra_loss_simple(
    rendered_adc_full: torch.Tensor,
    gt_adc_full: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compute RA L2 loss using PyTorch (no DrJit gradient connection).

    This is the simple version for monitoring/logging only.

    Args:
        rendered_adc_full: Rendered ADC tensor (NT, NR, K, 2)
        gt_adc_full: Ground truth ADC tensor (NT, NR, K, 2)
        normalize: Whether to normalize RA images separately

    Returns:
        PyTorch scalar loss value
    """
    # Compute RA images using PyTorch FFT
    ra_rendered = adc_to_ra_image(rendered_adc_full)
    ra_gt = adc_to_ra_image(gt_adc_full)

    # Min-max normalize separately if requested (matches inspect_ra_coir.py)
    if normalize:
        eps = 1e-12

        # Rendered: min-max normalization
        ra_rendered_min = torch.min(ra_rendered)
        ra_rendered_max = torch.max(ra_rendered)
        ra_rendered_range = torch.maximum(ra_rendered_max - ra_rendered_min, torch.tensor(eps))
        ra_rendered = (ra_rendered - ra_rendered_min) / ra_rendered_range

        # GT: min-max normalization
        ra_gt_min = torch.min(ra_gt)
        ra_gt_max = torch.max(ra_gt)
        ra_gt_range = torch.maximum(ra_gt_max - ra_gt_min, torch.tensor(eps))
        ra_gt = (ra_gt - ra_gt_min) / ra_gt_range

    # Compute L2 loss
    ra_loss = torch.mean((ra_rendered - ra_gt) ** 2)

    return ra_loss
