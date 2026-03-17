"""Combined loss for coherent ADC + incoherent power channels (Migration Plan v3).

This module implements the dual-channel loss function:
    L_total = lambda_coh * L_ADC + lambda_pow * L_power

Where:
- L_ADC: Range-azimuth loss on complex ADC (coherent paths)
- L_power: Direct power loss on power accumulator (incoherent paths)
- lambda_coh, lambda_pow: Weights for balancing the two channels

Key Design:
-----------
1. Coherent paths (alpha <= alpha_lo or smooth in transition) -> L_ADC via complex FFT
2. Incoherent paths (alpha >= alpha_hi or rough in transition) -> L_power via direct comparison
3. Separate gradient paths ensure proper attribution:
   - Smooth materials get gradients from coherent ADC
   - Rough materials get gradients from power channel
4. Adaptive weighting based on material distribution (optional)

Physical Rationale:
------------------
Per Migration Plan v3, rough microfacets (alpha > alpha_mirror) produce incoherent power
with decorrelated random phases. Supervising them via complex ADC would introduce
phase noise. Instead, supervise via power-domain loss which matches the physics.
"""

import torch
import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Tuple, Optional

from mmir.data.ra_utils import adc_to_ra_complex


class CombinedCoherentIncoherentLoss:
    """Combined loss for coherent ADC and incoherent power channels.

    This class computes:
    1. L_ADC: RA loss on complex ADC samples (coherent paths)
    2. L_power: Direct power loss on power accumulator (incoherent paths)
    3. L_total: Weighted combination

    Gradients are cached separately for each channel and combined during backward.
    """

    def __init__(self,
                 lambda_coh: float = 1.0,
                 lambda_pow: float = 1.0,
                 normalize_adc: bool = True,
                 normalize_power: bool = True):
        """Initialize combined loss.

        Args:
            lambda_coh: Weight for coherent ADC loss (default: 1.0)
            lambda_pow: Weight for incoherent power loss (default: 1.0)
            normalize_adc: Whether to normalize RA images separately
            normalize_power: Whether to normalize power images
        """
        self.lambda_coh = lambda_coh
        self.lambda_pow = lambda_pow
        self.normalize_adc = normalize_adc
        self.normalize_power = normalize_power

        # Gradient caches
        self.grad_adc_real_cache = None
        self.grad_adc_imag_cache = None
        self.grad_power_cache = None

    def compute_forward_and_cache_gradients(
        self,
        # Coherent ADC inputs
        rendered_adc_real: mi.Float,
        rendered_adc_imag: mi.Float,
        gt_adc_full: torch.Tensor,
        # Incoherent power inputs
        rendered_power: mi.Float,
        gt_power: torch.Tensor,
        # Dimensions
        NT: int,
        NR: int,
        K: int
    ) -> Tuple[mi.Float, mi.Float, mi.Float]:
        """Compute combined loss and cache gradients.

        Args:
            rendered_adc_real: DrJit real ADC values from coherent paths (NT*NR*K,)
            rendered_adc_imag: DrJit imag ADC values from coherent paths (NT*NR*K,)
            gt_adc_full: Ground truth complex ADC tensor (NT, NR, K, 2)
            rendered_power: DrJit power values from incoherent paths (NT*NR*K,)
            gt_power: Ground truth power tensor (NT, NR, K)
            NT: Number of TX elements
            NR: Number of RX elements
            K: Number of ADC samples

        Returns:
            (loss_total, loss_adc, loss_power): Total loss and component losses
        """
        # ====================================================================
        # COHERENT CHANNEL: Complex ADC -> RA Loss
        # ====================================================================

        # Extract ADC values
        with dr.suspend_grad():
            adc_real_np = np.array(rendered_adc_real).reshape(NT, NR, K)
            adc_imag_np = np.array(rendered_adc_imag).reshape(NT, NR, K)

        # Create PyTorch tensors with gradient tracking
        adc_real_torch = torch.from_numpy(adc_real_np.copy()).float()
        adc_imag_torch = torch.from_numpy(adc_imag_np.copy()).float()

        adc_real_torch.requires_grad = True
        adc_imag_torch.requires_grad = True

        # Assemble full ADC tensor for RA conversion
        rendered_adc_full = torch.stack([adc_real_torch, adc_imag_torch], dim=-1)

        # Compute complex RA using PyTorch FFT
        ra_rendered_complex = adc_to_ra_complex(rendered_adc_full)
        ra_gt_complex = adc_to_ra_complex(gt_adc_full)

        # Extract real and imaginary parts
        ra_rendered_real = ra_rendered_complex.real
        ra_rendered_imag = ra_rendered_complex.imag
        ra_gt_real = ra_gt_complex.real
        ra_gt_imag = ra_gt_complex.imag

        # Normalize separately if requested
        if self.normalize_adc:
            # Rendered RA: normalize by its max magnitude
            ra_rendered_mag = torch.abs(ra_rendered_complex)
            ra_rendered_max = torch.maximum(torch.max(ra_rendered_mag), torch.tensor(1e-12))
            ra_rendered_real = ra_rendered_real / ra_rendered_max
            ra_rendered_imag = ra_rendered_imag / ra_rendered_max

            # GT RA: normalize by its max magnitude
            ra_gt_mag = torch.abs(ra_gt_complex)
            ra_gt_max = torch.maximum(torch.max(ra_gt_mag), torch.tensor(1e-12))
            ra_gt_real = ra_gt_real / ra_gt_max
            ra_gt_imag = ra_gt_imag / ra_gt_max

        # Compute L2 loss on complex values
        diff_real = ra_rendered_real - ra_gt_real
        diff_imag = ra_rendered_imag - ra_gt_imag
        loss_adc_torch = torch.mean(diff_real ** 2 + diff_imag ** 2)

        # ====================================================================
        # INCOHERENT CHANNEL: Power -> Direct Loss
        # ====================================================================

        # Extract power values
        with dr.suspend_grad():
            power_np = np.array(rendered_power).reshape(NT, NR, K)

        # Create PyTorch tensor with gradient tracking
        power_torch = torch.from_numpy(power_np.copy()).float()
        power_torch.requires_grad = True

        # Normalize if requested
        if self.normalize_power:
            # Rendered power: normalize by its max
            power_max = torch.maximum(torch.max(power_torch), torch.tensor(1e-12))
            power_torch_norm = power_torch / power_max

            # GT power: normalize by its max
            gt_power_max = torch.maximum(torch.max(gt_power), torch.tensor(1e-12))
            gt_power_norm = gt_power / gt_power_max
        else:
            power_torch_norm = power_torch
            gt_power_norm = gt_power

        # Compute L2 loss on power
        diff_power = power_torch_norm - gt_power_norm
        loss_power_torch = torch.mean(diff_power ** 2)

        # ====================================================================
        # COMBINED LOSS
        # ====================================================================

        loss_total_torch = self.lambda_coh * loss_adc_torch + self.lambda_pow * loss_power_torch

        # Compute PyTorch gradients
        loss_total_torch.backward()

        # Cache gradients for DrJit backward pass
        self.grad_adc_real_cache = adc_real_torch.grad.numpy().flatten().astype(np.float32)
        self.grad_adc_imag_cache = adc_imag_torch.grad.numpy().flatten().astype(np.float32)
        self.grad_power_cache = power_torch.grad.numpy().flatten().astype(np.float32)

        # Convert to DrJit scalars
        loss_total_value = float(loss_total_torch.item())
        loss_adc_value = float(loss_adc_torch.item())
        loss_power_value = float(loss_power_torch.item())

        loss_total_drjit = mi.Float(loss_total_value)
        loss_adc_drjit = mi.Float(loss_adc_value)
        loss_power_drjit = mi.Float(loss_power_value)

        return loss_total_drjit, loss_adc_drjit, loss_power_drjit

    def get_gradients_for_drjit(self) -> Tuple[mi.Float, mi.Float, mi.Float]:
        """Get cached gradients as DrJit tensors.

        Returns:
            (grad_adc_real, grad_adc_imag, grad_power): DrJit gradient tensors
        """
        if self.grad_adc_real_cache is None:
            raise RuntimeError("Gradients not computed yet. Call compute_forward_and_cache_gradients first.")

        grad_adc_real = mi.Float(self.grad_adc_real_cache)
        grad_adc_imag = mi.Float(self.grad_adc_imag_cache)
        grad_power = mi.Float(self.grad_power_cache)

        return grad_adc_real, grad_adc_imag, grad_power


def compute_combined_loss_with_gradients(
    # Coherent ADC inputs
    rendered_adc_real: mi.Float,
    rendered_adc_imag: mi.Float,
    gt_adc_full: torch.Tensor,
    # Incoherent power inputs
    rendered_power: mi.Float,
    gt_power: torch.Tensor,
    # Dimensions
    NT: int,
    NR: int,
    K: int,
    # Loss weights
    lambda_coh: float = 1.0,
    lambda_pow: float = 1.0,
    normalize_adc: bool = True,
    normalize_power: bool = True
) -> Tuple[mi.Float, mi.Float, mi.Float, mi.Float, mi.Float, mi.Float]:
    """Compute combined coherent+incoherent loss with gradient flow.

    This is a convenience function that creates a CombinedCoherentIncoherentLoss
    instance and computes the loss + gradients in one call.

    Args:
        rendered_adc_real: DrJit real ADC values from coherent paths (NT*NR*K,)
        rendered_adc_imag: DrJit imag ADC values from coherent paths (NT*NR*K,)
        gt_adc_full: Ground truth complex ADC tensor (NT, NR, K, 2)
        rendered_power: DrJit power values from incoherent paths (NT*NR*K,)
        gt_power: Ground truth power tensor (NT, NR, K)
        NT: Number of TX elements
        NR: Number of RX elements
        K: Number of ADC samples
        lambda_coh: Weight for coherent ADC loss
        lambda_pow: Weight for incoherent power loss
        normalize_adc: Whether to normalize RA images separately
        normalize_power: Whether to normalize power images

    Returns:
        (loss_total, loss_adc, loss_power, grad_real, grad_imag, grad_power)
    """
    loss_fn = CombinedCoherentIncoherentLoss(
        lambda_coh=lambda_coh,
        lambda_pow=lambda_pow,
        normalize_adc=normalize_adc,
        normalize_power=normalize_power
    )

    loss_total, loss_adc, loss_power = loss_fn.compute_forward_and_cache_gradients(
        rendered_adc_real=rendered_adc_real,
        rendered_adc_imag=rendered_adc_imag,
        gt_adc_full=gt_adc_full,
        rendered_power=rendered_power,
        gt_power=gt_power,
        NT=NT,
        NR=NR,
        K=K
    )

    grad_real, grad_imag, grad_power = loss_fn.get_gradients_for_drjit()

    return loss_total, loss_adc, loss_power, grad_real, grad_imag, grad_power
