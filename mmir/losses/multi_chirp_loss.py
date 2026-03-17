"""Multi-chirp loss computation with per-chirp alignment.

This module provides differentiable loss functions for FMCW radar training
with multiple chirps, including:
1. Per-chirp complex scalar alignment (removes nuisance global phase)
2. Magnitude loss (averaged over chirps for noise reduction)
3. Phase loss using unit phasor after alignment (Option A)

Key design decisions:
- Per-chirp alignment: Each chirp gets its own complex scalar α_c that
  absorbs LO phase drift, timing jitter, and arbitrary reference phase
- Unit phasor loss: Avoids angle wrapping issues (2(1-cos(Δφ)) ≡ |u-u_hat|²)
- Magnitude weighting: Only penalize phase at high-SNR bins
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .loss_utils import minmax_normalize_torch


def per_chirp_alignment(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Align prediction to GT with per-chirp complex scalar.

    For each chirp c, solves for optimal α_c:
        α_c = Σ(w · gt · pred*) / Σ(w · |pred|²)

    where w is the magnitude weight (higher weight for strong signals).

    Args:
        pred: Complex tensor [n_chirp, n_rx, n_tx, n_adc]
        gt: Complex tensor [n_chirp, n_rx, n_tx, n_adc]
        eps: Numerical stability constant

    Returns:
        Aligned prediction [n_chirp, n_rx, n_tx, n_adc]
    """
    n_chirp = pred.shape[0]
    pred_aligned = torch.zeros_like(pred)

    for c in range(n_chirp):
        pred_c = pred[c]  # [rx, tx, adc]
        gt_c = gt[c]

        # Magnitude weights (only fit alignment on high-SNR bins)
        gt_mag = gt_c.abs()
        gt_mag_max = gt_mag.amax().clamp_min(eps)
        w = (gt_mag / gt_mag_max).detach().clamp(0, 1)

        # Optimal complex scalar: α = (w * gt * pred*) / (w * |pred|²)
        num = (w * gt_c * pred_c.conj()).sum()
        den = (w * pred_c.abs().pow(2)).sum().clamp_min(eps)
        alpha_c = num / den  # Complex scalar (amplitude + phase)

        pred_aligned[c] = alpha_c * pred_c

    return pred_aligned


def per_chirp_alignment_batched(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Vectorized per-chirp alignment (faster than loop version).

    For each chirp c, computes:
        α_c = Σ(w · gt · pred*) / Σ(w · |pred|²)

    where sums are over (rx, tx, adc) dimensions.

    Args:
        pred: Complex tensor [n_chirp, n_rx, n_tx, n_adc]
        gt: Complex tensor [n_chirp, n_rx, n_tx, n_adc]
        eps: Numerical stability constant

    Returns:
        Aligned prediction [n_chirp, n_rx, n_tx, n_adc]
    """
    # Magnitude weights per chirp
    gt_mag = gt.abs()
    gt_mag_max = gt_mag.amax(dim=(-3, -2, -1), keepdim=True)  # [n_chirp, 1, 1, 1]
    w = (gt_mag / gt_mag_max.clamp_min(eps)).detach()
    w = w.clamp(0, 1)

    # Per-chirp complex scalar: sum over (rx, tx, adc), keep chirp dim
    num = (w * gt * pred.conj()).sum(dim=(-3, -2, -1))  # [n_chirp]
    den = (w * pred.abs().pow(2)).sum(dim=(-3, -2, -1)).clamp_min(eps)  # [n_chirp]
    alpha = num / den  # [n_chirp] complex

    # Reshape for broadcasting: [n_chirp] -> [n_chirp, 1, 1, 1]
    alpha = alpha.view(-1, 1, 1, 1)

    return alpha * pred


def unit_phasor_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute phase loss using unit phasor representation.

    Loss = |u_hat - u|² where u = z / |z| (unit phasor)
    Equivalent to 2(1 - cos(Δφ)) for unit vectors.

    This avoids issues with angle wrapping that plague torch.angle().

    Args:
        rendered: Complex tensor (any shape)
        gt: Complex tensor (same shape as rendered)
        eps: Numerical stability constant

    Returns:
        Scalar loss tensor
    """
    u_hat = rendered / rendered.abs().clamp_min(eps)
    u = gt / gt.abs().clamp_min(eps)

    # Circular distance on unit circle: |u_hat - u|²
    loss = (u_hat - u).abs().pow(2).mean()
    return loss


def magnitude_weighted_phase_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    use_gt_magnitude: bool = True,
    top_percentile: Optional[float] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute phase loss weighted by magnitude.

    Only penalizes phase error where signal is strong.
    Phase at low-SNR bins is noise - don't waste gradients on it.

    Args:
        rendered: Complex tensor [*, rx, tx, adc] or [chirp, rx, tx, adc]
        gt: Complex tensor (same shape)
        use_gt_magnitude: If True, use GT magnitude for weighting (more stable)
        top_percentile: If set (e.g., 0.7), only include bins above this percentile
        eps: Small value for numerical stability

    Returns:
        Scalar loss tensor
    """
    # Compute magnitude weights
    mag = gt.abs() if use_gt_magnitude else rendered.abs()

    # Normalize per-chirp (if chirp dim exists) or globally
    if mag.dim() >= 4:  # Has chirp dimension
        mag_max = mag.amax(dim=(-3, -2, -1), keepdim=True)
    else:
        mag_max = mag.amax()

    w = (mag / mag_max.clamp_min(eps)).detach()
    w = w.clamp(0, 1)

    # Optional: only top-p percentile
    if top_percentile is not None:
        threshold = torch.quantile(mag.flatten(), top_percentile)
        mask = (mag >= threshold).float()
        w = w * mask

    # Unit phasor difference
    u_hat = rendered / rendered.abs().clamp_min(eps)
    u = gt / gt.abs().clamp_min(eps)

    # |u_hat - u|² for unit vectors = 2(1 - cos(Δφ))
    phase_diff = (u_hat - u).abs().pow(2)

    # Weighted mean
    loss = (w * phase_diff).sum() / w.sum().clamp_min(eps)
    return loss


def compute_multi_chirp_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    w_adc_mag: float = 1.0,
    w_ra_mag: float = 1.0,
    w_phase: float = 0.1,
    adc_use_log: bool = False,
    ra_use_log: bool = False,
    adc_loss_type: str = "l2",
    ra_loss_type: str = "l2",
    log_epsilon: float = 1e-6,
    log_scale: float = 1.0,
    top_magnitude_percentile: Optional[float] = 0.9,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss over multiple chirps with proper phase handling.

    Pipeline:
    1. Convert real/imag to complex
    2. Per-chirp complex scalar alignment (removes nuisance phase)
    3. Magnitude loss (averaged over chirps)
    4. Phase loss using unit phasor after alignment (Option A)

    Args:
        pred: Prediction [n_chirp, n_rx, n_tx, n_adc, 2] (real/imag)
              OR [n_rx, n_tx, n_adc, 2] (single chirp, will be broadcast)
              OR complex tensor of same shapes
        gt: Ground truth [n_chirp, n_rx, n_tx, n_adc, 2] or complex
        w_adc_mag: Weight for ADC magnitude loss
        w_ra_mag: Weight for RA magnitude loss (FFT domain)
        w_phase: Weight for phase loss (warm up from 0 → target)
        adc_use_log: Use log for ADC magnitude loss
        ra_use_log: Use log for RA magnitude loss
        adc_loss_type: "l1" or "l2" for ADC magnitude loss
        ra_loss_type: "l1" or "l2" for RA magnitude loss
        log_epsilon: Epsilon for log(eps + scale * |x|)
        log_scale: Scale factor for log(eps + scale * |x|)
        top_magnitude_percentile: Only use phase from top bins (e.g., 0.7 = top 30%)
        eps: Numerical stability

    Returns:
        total_loss: Scalar loss tensor
        loss_dict: Dictionary of individual loss components
    """
    # Step 1: Convert to complex if needed
    if not pred.is_complex():
        if pred.shape[-1] == 2:
            pred_complex = torch.complex(pred[..., 0], pred[..., 1])
        else:
            raise ValueError(f"Expected pred shape [..., 2] for real/imag, got {pred.shape}")
    else:
        pred_complex = pred

    if not gt.is_complex():
        if gt.shape[-1] == 2:
            gt_complex = torch.complex(gt[..., 0], gt[..., 1])
        else:
            raise ValueError(f"Expected gt shape [..., 2] for real/imag, got {gt.shape}")
    else:
        gt_complex = gt

    # Handle single-chirp prediction: broadcast to all chirps
    # pred: [rx, tx, adc] -> [n_chirp, rx, tx, adc]
    if pred_complex.dim() == 3 and gt_complex.dim() == 4:
        n_chirp = gt_complex.shape[0]
        pred_complex = pred_complex.unsqueeze(0).expand(n_chirp, -1, -1, -1)

    n_chirp = gt_complex.shape[0]
    device = pred_complex.device

    # Step 2: Per-chirp alignment (removes nuisance global phase)
    pred_aligned = per_chirp_alignment_batched(pred_complex, gt_complex, eps)

    # Step 3: ADC Magnitude loss (average over chirps for noise reduction)
    if adc_use_log:
        mag_pred = torch.log(log_epsilon + log_scale * pred_aligned.abs())
        mag_gt = torch.log(log_epsilon + log_scale * gt_complex.abs())
    else:
        # Min-max normalize to [0, 1] for scale-invariant comparison
        mag_pred = minmax_normalize_torch(pred_aligned.abs())
        mag_gt = minmax_normalize_torch(gt_complex.abs())

    if adc_loss_type == "l1":
        loss_adc_mag = F.l1_loss(mag_pred, mag_gt)
    else:  # l2 (default)
        loss_adc_mag = F.mse_loss(mag_pred, mag_gt)

    # Step 3b: RA domain magnitude loss (FFT along ADC dimension)
    loss_ra_mag = torch.tensor(0.0, device=device)
    ra_per_chirp_mse = None  # For RMSE computation
    ra_per_chirp_rmse_avg = 0.0  # Average of per-chirp RMSEs
    if w_ra_mag > 0:
        # FFT along ADC dimension (last dim)
        ra_pred = torch.fft.fft(pred_aligned, dim=-1)
        ra_gt = torch.fft.fft(gt_complex, dim=-1)

        if ra_use_log:
            ra_mag_pred = torch.log(log_epsilon + log_scale * ra_pred.abs())
            ra_mag_gt = torch.log(log_epsilon + log_scale * ra_gt.abs())
        else:
            # Min-max normalize to [0, 1] for scale-invariant comparison
            ra_mag_pred = minmax_normalize_torch(ra_pred.abs())
            ra_mag_gt = minmax_normalize_torch(ra_gt.abs())

        # Compute per-chirp MSE (vectorized) - used for both L1 and L2 metrics
        ra_per_chirp_mse = ((ra_mag_pred - ra_mag_gt) ** 2).mean(dim=(-3, -2, -1))  # [n_chirp]
        # Per-chirp RMSE, then average (NOT sqrt of averaged MSE)
        ra_per_chirp_rmse_avg = ra_per_chirp_mse.sqrt().mean().item()

        if ra_loss_type == "l1":
            # Compute per-chirp L1, then average (vectorized)
            # Shape: [n_chirp, rx, tx, adc] -> per-chirp mean -> [n_chirp] -> average
            per_chirp_l1 = (ra_mag_pred - ra_mag_gt).abs().mean(dim=(-3, -2, -1))  # [n_chirp]
            loss_ra_mag = per_chirp_l1.mean()
        else:  # l2 (default)
            # Per-chirp MSE averaged (gives equal weight to each chirp)
            loss_ra_mag = ra_per_chirp_mse.mean()

    # Step 4: Phase loss using unit phasor after alignment (Option A)
    loss_phase = torch.tensor(0.0, device=device)
    if w_phase > 0:
        loss_phase = magnitude_weighted_phase_loss(
            pred_aligned, gt_complex,
            use_gt_magnitude=True,
            top_percentile=top_magnitude_percentile,
            eps=eps
        )

    # Total loss
    total_loss = (
        w_adc_mag * loss_adc_mag +
        w_ra_mag * loss_ra_mag +
        w_phase * loss_phase
    )

    loss_dict = {
        'adc_mag': loss_adc_mag.item(),
        'ra_mag': loss_ra_mag.item() if isinstance(loss_ra_mag, torch.Tensor) else loss_ra_mag,
        'ra_rmse': ra_per_chirp_rmse_avg,  # Per-chirp RMSE averaged (for metrics tracking)
        'phase': loss_phase.item() if isinstance(loss_phase, torch.Tensor) else loss_phase,
        'total': total_loss.item()
    }

    return total_loss, loss_dict


def chirp_averaged_magnitude_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    domain: str = "adc",
    use_log: bool = True
) -> torch.Tensor:
    """
    Average magnitude (or power) over chirps before computing loss.

    Averaging over chirps reduces noise variance - the incoherent
    (power) average is more stable than individual chirp comparisons.

    Args:
        pred: [n_chirp, n_rx, n_tx, n_adc] complex
        gt: [n_chirp, n_rx, n_tx, n_adc] complex
        domain: Compute in "adc" or "ra" (FFT) domain
        use_log: Apply log1p to magnitude

    Returns:
        Scalar loss tensor
    """
    if domain == "ra":
        pred = torch.fft.fft(pred, dim=-1)
        gt = torch.fft.fft(gt, dim=-1)

    # Average magnitude over chirps (noise reduction)
    pred_mag_avg = pred.abs().mean(dim=0)  # [rx, tx, adc]
    gt_mag_avg = gt.abs().mean(dim=0)

    if use_log:
        pred_mag_avg = torch.log1p(pred_mag_avg)
        gt_mag_avg = torch.log1p(gt_mag_avg)

    return F.mse_loss(pred_mag_avg, gt_mag_avg)


def get_alignment_scalars(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Get the per-chirp alignment scalars (for debugging/monitoring).

    These scalars tell you:
    - |α_c|: amplitude correction factor per chirp
    - angle(α_c): phase correction per chirp

    Large variation in |α_c| suggests inconsistent rendering vs GT gain.
    Large variation in angle(α_c) suggests phase drift across chirps.

    Args:
        pred: Complex tensor [n_chirp, n_rx, n_tx, n_adc]
        gt: Complex tensor [n_chirp, n_rx, n_tx, n_adc]
        eps: Numerical stability

    Returns:
        Complex tensor [n_chirp] of alignment scalars
    """
    # Same computation as per_chirp_alignment_batched, but return scalars
    gt_mag = gt.abs()
    gt_mag_max = gt_mag.amax(dim=(-3, -2, -1), keepdim=True)
    w = (gt_mag / gt_mag_max.clamp_min(eps)).detach().clamp(0, 1)

    num = (w * gt * pred.conj()).sum(dim=(-3, -2, -1))
    den = (w * pred.abs().pow(2)).sum(dim=(-3, -2, -1)).clamp_min(eps)
    alpha = num / den

    return alpha
