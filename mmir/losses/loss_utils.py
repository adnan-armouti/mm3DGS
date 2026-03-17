"""Loss computation and image processing utilities for training."""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier (pseudo-Huber) robust loss between x and y."""
    r = x - y
    return torch.mean(torch.sqrt(r * r + (eps * eps)))


def huber(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss between x and y."""
    r = torch.abs(x - y)
    quad = 0.5 * (r * r)
    lin  = delta * (r - 0.5 * delta)
    return torch.mean(torch.where(r <= delta, quad, lin))


def robust_l1(x: torch.Tensor, y: torch.Tensor, kind: str = "charbonnier") -> torch.Tensor:
    """Select robust loss kind for regression in RA domain.
    kind: "charbonnier" | "huber" | "l1"
    """
    k = (kind or "").lower()
    if k == "charbonnier":
        return charbonnier(x, y)
    if k == "huber":
        return huber(x, y)
    return torch.mean(torch.abs(x - y))


def scale_to_255(img: torch.Tensor, shared_max: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scale non-negative image to [0,255] using shared_max if provided."""
    if shared_max is None:
        m = float(img.max().item())
    else:
        m = float(shared_max.item())
    if not math.isfinite(m) or m <= 0:
        m = 1.0
    return (img / m) * 255.0


# =============================================================================
# Min-Max Normalization Utilities
# =============================================================================

def minmax_normalize_numpy(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize numpy array to [0, 1] range.

    Formula: (x - min) / (max - min + eps)
    Matches inspect_ra_coir.py normalization.

    Args:
        x: Input numpy array
        eps: Small value to prevent division by zero

    Returns:
        Normalized array in [0, 1] range
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def minmax_normalize_torch(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Min-max normalize PyTorch tensor to [0, 1] range.

    Formula: (x - min) / (max - min + eps)
    Matches inspect_ra_coir.py normalization.

    Args:
        x: Input PyTorch tensor
        eps: Small value to prevent division by zero

    Returns:
        Normalized tensor in [0, 1] range
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def minmax_normalize_complex_torch(real: torch.Tensor, imag: torch.Tensor,
                                    eps: float = 1e-9) -> tuple:
    """Min-max normalize complex signal while preserving phase.

    Applies min-max normalization to the magnitude and scales real/imag
    components proportionally to preserve phase relationships.

    Args:
        real: Real part of complex signal
        imag: Imaginary part of complex signal
        eps: Small value to prevent division by zero

    Returns:
        (real_norm, imag_norm): Normalized real and imaginary parts
    """
    # Compute magnitude
    mag = torch.sqrt(real * real + imag * imag)

    # Min-max normalize magnitude
    mag_min = mag.min()
    mag_max = mag.max()
    mag_range = mag_max - mag_min + eps
    mag_minmax = (mag - mag_min) / mag_range

    # Scale factor to convert original magnitude to min-max magnitude
    # scale = minmax_mag / original_mag
    scale = mag_minmax / (mag + eps)

    # Apply scale to real and imag (preserves phase)
    real_norm = real * scale
    imag_norm = imag * scale

    return real_norm, imag_norm


def as_image2d_mag(t: torch.Tensor) -> torch.Tensor:
    """Convert real-imag ADC (...,2) to magnitude and reshape to (1,1,H,W).
    H is product of all dims except the last two; W is the second-to-last dim.
    """
    t2 = t.detach().float().contiguous()
    if t2.dim() < 2:
        mag = t2.abs()
        return mag.reshape(1, 1, 1, mag.numel())
    if t2.shape[-1] == 2:
        mag = torch.linalg.norm(t2, dim=-1)
    else:
        mag = t2
    if mag.dim() == 1:
        return mag.reshape(1, 1, 1, mag.numel())
    h = int(np.prod(mag.shape[:-1])) if mag.dim() >= 2 else 1
    w = int(mag.shape[-1])
    return mag.reshape(1, 1, h, w)


def compute_psnr(x: torch.Tensor, y: torch.Tensor, data_range: Optional[float] = None) -> float:
    """Compute Peak Signal-to-Noise Ratio between two tensors."""
    x_img = as_image2d_mag(x)
    y_img = as_image2d_mag(y)
    if data_range is None:
        dr = float((y_img.max() - y_img.min()).item())
        data_range = dr if dr > 0 else 1.0
    mse = F.mse_loss(x_img, y_img, reduction="mean").item()
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(float(mse))


def gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    coords = torch.arange(window_size).float() - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    window = window_2d.repeat(channels, 1, 1, 1)
    return window


def compute_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Optional[float] = None,
                 window_size: int = 11, sigma: float = 1.5) -> float:
    """Compute Structural Similarity Index between two tensors."""
    x_img = as_image2d_mag(x)
    y_img = as_image2d_mag(y)
    if data_range is None:
        dr = float((y_img.max() - y_img.min()).item())
        data_range = dr if dr > 0 else 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    channel_count = x_img.shape[1]
    window = gaussian_window(window_size, sigma, channel_count).to(x_img.device, x_img.dtype)

    padding = window_size // 2
    mu_x = F.conv2d(x_img, window, padding=padding, groups=channel_count)
    mu_y = F.conv2d(y_img, window, padding=padding, groups=channel_count)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x_img * x_img, window, padding=padding, groups=channel_count) - mu_x_sq
    sigma_y_sq = F.conv2d(y_img * y_img, window, padding=padding, groups=channel_count) - mu_y_sq
    sigma_xy = F.conv2d(x_img * y_img, window, padding=padding, groups=channel_count) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def compute_ssim_loss(x: torch.Tensor, y: torch.Tensor,
                      window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Compute differentiable SSIM loss (1 - SSIM) between two tensors.

    Unlike compute_ssim(), this returns a torch.Tensor to preserve gradients.
    Assumes inputs are already normalized to [0, 1] range.

    Args:
        x: Predicted tensor, shape (H, W) or (1, 1, H, W)
        y: Target tensor, same shape as x
        window_size: Size of Gaussian window (default 11)
        sigma: Standard deviation of Gaussian window (default 1.5)

    Returns:
        torch.Tensor: SSIM loss = 1 - SSIM (scalar, lower is better)
    """
    # Ensure (1, 1, H, W) shape
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    channel_count = x.shape[1]

    # Create Gaussian window (same as gaussian_window but inline for gradient flow)
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel_count, 1, window_size, window_size).contiguous()

    # SSIM constants (for data_range=1.0)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channel_count)
    mu_y = F.conv2d(y, window, padding=padding, groups=channel_count)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channel_count) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channel_count) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channel_count) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + 1e-12)

    # Return 1 - SSIM as loss (lower is better, higher SSIM means more similar)
    return 1.0 - ssim_map.mean()


# ============================================================================
# ADVANCED LOSS FUNCTIONS FOR MIMO RADAR TRAINING
# ============================================================================

def get_magnitude(t: torch.Tensor) -> torch.Tensor:
    """Extract magnitude from complex or real/imag tensor.

    Args:
        t: Tensor that is either:
           - Complex dtype
           - Real tensor with shape (..., 2) for real/imag

    Returns:
        Magnitude tensor
    """
    if t.is_complex():
        return torch.abs(t)
    elif t.shape[-1] == 2:
        return torch.sqrt(t[..., 0] ** 2 + t[..., 1] ** 2)
    else:
        # Assume already magnitude
        return t


def to_complex(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to complex dtype if not already.

    Args:
        t: Tensor that is either complex or real/imag (..., 2)

    Returns:
        Complex tensor
    """
    if t.is_complex():
        return t
    else:
        return torch.complex(t[..., 0], t[..., 1])


def magnitude_loss(pred: torch.Tensor, target: torch.Tensor,
                   use_log: bool = False, loss_type: str = "l2",
                   eps: float = 1e-6, log_scale: float = 1.0,
                   huber_delta: float = 1.0) -> torch.Tensor:
    """Unified ADC/signal magnitude loss with optional log-compression.

    Args:
        pred: Predicted complex tensor (..., 2) or complex dtype
        target: Target complex tensor, same shape
        use_log: If True, apply log-compression before loss
        loss_type: "l1" | "l2" | "huber"
        eps: Small constant to avoid log(0)
        log_scale: Scale factor α for log(ε + α·|x|)
        huber_delta: Delta for Huber loss

    Returns:
        Scalar loss tensor
    """
    pred_mag = get_magnitude(pred)
    target_mag = get_magnitude(target)

    if use_log:
        pred_mag = torch.log(eps + log_scale * pred_mag)
        target_mag = torch.log(eps + log_scale * target_mag)

    diff = pred_mag - target_mag

    if loss_type == "l1":
        return torch.mean(torch.abs(diff))
    elif loss_type == "huber":
        abs_diff = torch.abs(diff)
        quad = 0.5 * diff ** 2
        linear = huber_delta * (abs_diff - 0.5 * huber_delta)
        return torch.mean(torch.where(abs_diff <= huber_delta, quad, linear))
    else:  # l2
        return torch.mean(diff ** 2)


def circular_phase_loss(pred: torch.Tensor, target: torch.Tensor,
                        magnitude_weighting: bool = True,
                        magnitude_power: float = 0.5,
                        use_gt_magnitude: bool = True,
                        magnitude_clip_max: float = 0.0,
                        eps: float = 1e-8) -> torch.Tensor:
    """Unit-phasor circular phase loss: 1 - Re(û · u*).

    This is equivalent to 1 - cos(Δφ), properly wrap-aware.
    Important for MIMO: phase determines beamforming direction.

    With amplitude weighting, the formula becomes:
        L = sum_i w_i * (1 - cos(Δφ_i))
    where w_i = stopgrad(clip(|z^{gt}_i|^p, 0, w_max))

    This reduces the optimizer's incentive to change geometry/materials
    just to fit noisy phase in low-SNR bins that don't matter.

    Args:
        pred: Predicted complex tensor (..., 2) for real/imag or complex dtype
        target: Target complex tensor (ground truth)
        magnitude_weighting: Weight by magnitude (phase at low mag is meaningless)
        magnitude_power: Exponent for magnitude weighting (0.5 = sqrt, 1.0 = linear)
        use_gt_magnitude: If True, use GT magnitude for weighting (recommended).
                          If False, use predicted magnitude (original behavior).
        magnitude_clip_max: Clip weights to this max value (0 = no clipping).
                           Recommended: set to ~95th percentile of GT magnitude.
        eps: Small constant for numerical stability

    Returns:
        Scalar loss tensor
    """
    pred_c = to_complex(pred)
    target_c = to_complex(target)

    # Magnitudes for normalization
    pred_mag = torch.abs(pred_c)
    target_mag = torch.abs(target_c)

    # Unit phasors
    u_pred = pred_c / (pred_mag + eps)
    u_target = target_c / (target_mag + eps)

    # Circular distance: 1 - Re(û · u*)
    cosine_sim = (u_pred * u_target.conj()).real
    phase_loss = 1.0 - cosine_sim

    if magnitude_weighting:
        # Choose which magnitude to use for weighting
        if use_gt_magnitude:
            # Use GT magnitude - phase at low GT mag is meaningless regardless of prediction
            weights = target_mag.detach() ** magnitude_power
        else:
            # Original behavior: use predicted magnitude
            weights = pred_mag.detach() ** magnitude_power

        # Apply clipping if specified (reduces influence of outliers)
        if magnitude_clip_max > 0:
            weights = torch.clamp(weights, 0, magnitude_clip_max)

        # Normalize to mean=1 (preserves loss scale)
        weights = weights / (weights.mean() + eps)
        return torch.mean(weights * phase_loss)
    else:
        return torch.mean(phase_loss)


def relative_phase_loss(pred: torch.Tensor, target: torch.Tensor,
                        reference_channel: int = 0,
                        magnitude_weighting: bool = True,
                        magnitude_power: float = 0.5,
                        use_gt_magnitude: bool = True,
                        magnitude_clip_max: float = 0.0,
                        eps: float = 1e-8) -> torch.Tensor:
    """Relative phase loss across channels - key for MIMO beamforming.

    Computes phase differences relative to a reference channel, then applies
    circular loss. This trains the inter-channel phase that determines angle.

    The key insight: absolute phase is unidentifiable (LO phase, hardware delays),
    but relative phase across channels determines the angle-of-arrival.

    With amplitude weighting, low-SNR samples (where phase is meaningless) are
    down-weighted to prevent noisy gradients from corrupting geometry/materials.

    Args:
        pred: Predicted ADC tensor (NT, NR, K, 2) or complex (NT, NR, K)
        target: Target ADC tensor, same shape (ground truth)
        reference_channel: Index of reference RX channel (or -1 for mean)
        magnitude_weighting: Weight by magnitude (phase at low mag is meaningless)
        magnitude_power: Exponent for magnitude weighting (0.5 = sqrt, 1.0 = linear)
        use_gt_magnitude: If True, use GT magnitude for weighting (recommended).
                          If False, use predicted magnitude.
        magnitude_clip_max: Clip weights to this max value (0 = no clipping).
                           Recommended: set to ~95th percentile of GT magnitude.
        eps: Numerical stability

    Returns:
        Scalar loss tensor
    """
    pred_c = to_complex(pred)
    target_c = to_complex(target)

    # Handle different input shapes
    if pred_c.dim() == 3:
        # (NT, NR, K) - standard ADC shape
        channel_dim = 1
    elif pred_c.dim() == 2:
        # (NR, K) - single TX
        pred_c = pred_c.unsqueeze(0)
        target_c = target_c.unsqueeze(0)
        channel_dim = 1
    else:
        # Flatten to 3D for channel-wise processing
        original_shape = pred_c.shape
        pred_c = pred_c.reshape(-1, original_shape[-2], original_shape[-1])
        target_c = target_c.reshape(-1, original_shape[-2], original_shape[-1])
        channel_dim = 1

    NR = pred_c.shape[channel_dim]

    # Get reference channel
    if reference_channel >= 0 and reference_channel < NR:
        pred_ref = pred_c.select(channel_dim, reference_channel).unsqueeze(channel_dim)
        target_ref = target_c.select(channel_dim, reference_channel).unsqueeze(channel_dim)
    else:
        # Use mean across channels as reference
        pred_ref = pred_c.mean(dim=channel_dim, keepdim=True)
        target_ref = target_c.mean(dim=channel_dim, keepdim=True)

    # Compute relative phase: angle(x_c / x_ref) = angle(x_c * conj(x_ref))
    pred_rel = pred_c * pred_ref.conj()
    target_rel = target_c * target_ref.conj()

    # Apply circular phase loss on the relative phases
    pred_rel_mag = torch.abs(pred_rel)
    target_rel_mag = torch.abs(target_rel)

    u_pred = pred_rel / (pred_rel_mag + eps)
    u_target = target_rel / (target_rel_mag + eps)

    # Circular distance
    cosine_sim = (u_pred * u_target.conj()).real
    phase_loss = 1.0 - cosine_sim

    if magnitude_weighting:
        # Choose which magnitude to use for weighting
        if use_gt_magnitude:
            # Use GT magnitude - phase at low GT mag is meaningless regardless of prediction
            weights = torch.abs(target_c).detach() ** magnitude_power
        else:
            # Use predicted magnitude
            weights = torch.abs(pred_c).detach() ** magnitude_power

        # Apply clipping if specified (reduces influence of outliers)
        if magnitude_clip_max > 0:
            weights = torch.clamp(weights, 0, magnitude_clip_max)

        # Normalize to mean=1 (preserves loss scale)
        weights = weights / (weights.mean() + eps)
        return torch.mean(weights * phase_loss)
    else:
        return torch.mean(phase_loss)


def relative_phase_angle_loss(pred: torch.Tensor, target: torch.Tensor,
                               reference_channel: int = 0,
                               magnitude_threshold: float = 0.1,
                               loss_type: str = "mse",
                               eps: float = 1e-8) -> torch.Tensor:
    """Relative phase loss using actual angle values - matches metric computation.

    This computes MSE/L1 on the wrapped phase difference, which directly optimizes
    for the phase error metrics (L1, MSE, Pearson correlation).

    Unlike circular_phase_loss (1 - cos), this uses actual angle values and
    matches how the metrics are computed.

    Args:
        pred: Predicted ADC tensor (NT, NR, K, 2) or complex (NT, NR, K)
        target: Target ADC tensor, same shape
        reference_channel: Index of reference RX channel
        magnitude_threshold: Fraction of max magnitude to consider valid (0.1 = 10%)
        loss_type: "mse" | "l1" | "huber"
        eps: Numerical stability

    Returns:
        Scalar loss tensor
    """
    pred_c = to_complex(pred)
    target_c = to_complex(target)

    # Handle different input shapes
    if pred_c.dim() == 3:
        # (NT, NR, K) - standard ADC shape
        channel_dim = 1
    elif pred_c.dim() == 2:
        # (NR, K) - single TX
        pred_c = pred_c.unsqueeze(0)
        target_c = target_c.unsqueeze(0)
        channel_dim = 1
    else:
        # Flatten to 3D
        original_shape = pred_c.shape
        pred_c = pred_c.reshape(-1, original_shape[-2], original_shape[-1])
        target_c = target_c.reshape(-1, original_shape[-2], original_shape[-1])
        channel_dim = 1

    NR = pred_c.shape[channel_dim]

    # Get reference channel (exactly as in metric computation)
    if reference_channel >= 0 and reference_channel < NR:
        pred_ref = pred_c.select(channel_dim, reference_channel).unsqueeze(channel_dim)
        target_ref = target_c.select(channel_dim, reference_channel).unsqueeze(channel_dim)
    else:
        pred_ref = pred_c.mean(dim=channel_dim, keepdim=True)
        target_ref = target_c.mean(dim=channel_dim, keepdim=True)

    # Compute relative phase: angle(x_c * conj(x_ref))
    # This is exactly how the metric computes it
    pred_rel = pred_c * pred_ref.conj()
    target_rel = target_c * target_ref.conj()

    # Get phase angles (in [-pi, pi])
    pred_rel_phase = torch.angle(pred_rel)
    target_rel_phase = torch.angle(target_rel)

    # Compute phase difference and wrap to [-pi, pi]
    phase_diff = pred_rel_phase - target_rel_phase
    # Wrap: atan2(sin(x), cos(x)) returns x wrapped to [-pi, pi]
    phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

    # Apply magnitude threshold (like the metric does)
    target_mag = torch.abs(target_c)
    mag_threshold_val = magnitude_threshold * target_mag.max()
    valid_mask = target_mag > mag_threshold_val

    # Only compute loss on valid samples
    if valid_mask.sum() < 10:
        # Not enough valid samples - return loss that still connects to inputs
        # Use a tiny fraction of all phase differences to maintain gradient flow
        phase_diff_normalized = phase_diff_wrapped / torch.pi
        return torch.mean(phase_diff_normalized ** 2) * 0.0 + 1e-8

    phase_diff_valid = phase_diff_wrapped[valid_mask]

    # Normalize phase to [-1, 1] for consistent scale (divide by pi)
    phase_diff_normalized = phase_diff_valid / torch.pi

    # Compute loss
    if loss_type == "l1":
        return torch.mean(torch.abs(phase_diff_normalized))
    elif loss_type == "huber":
        delta = 0.5  # ~90 degrees in normalized scale
        abs_diff = torch.abs(phase_diff_normalized)
        quad = 0.5 * phase_diff_normalized ** 2
        linear = delta * (abs_diff - 0.5 * delta)
        return torch.mean(torch.where(abs_diff <= delta, quad, linear))
    else:  # mse
        return torch.mean(phase_diff_normalized ** 2)


# =============================================================================
# Unit Phasor and Wrapped Phase Losses
# =============================================================================

def unit_phasor_loss(rendered: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute phase loss using unit phasor representation.

    Loss = |u_hat - u|² where u = z / |z| (unit phasor)
    Equivalent to 2(1 - cos(Δφ)) for unit vectors.

    This avoids issues with angle wrapping that plague torch.angle().

    Args:
        rendered: Complex tensor (any shape) or real/imag (..., 2)
        gt: Complex tensor (same shape as rendered)
        eps: Numerical stability constant

    Returns:
        Scalar loss tensor
    """
    rendered_c = to_complex(rendered)
    gt_c = to_complex(gt)

    u_hat = rendered_c / rendered_c.abs().clamp_min(eps)
    u = gt_c / gt_c.abs().clamp_min(eps)

    # Circular distance on unit circle: |u_hat - u|²
    loss = (u_hat - u).abs().pow(2).mean()
    return loss


def wrapped_phase_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute phase loss with proper wrapping to [-π, π].

    Uses atan2(sin(Δφ), cos(Δφ)) for proper wrapping.

    Note: This uses torch.angle() which can have issues at low magnitudes.
    For robustness, prefer unit_phasor_loss or magnitude_weighted_phase_loss.

    Args:
        rendered: Complex tensor or real/imag (..., 2)
        gt: Complex tensor (same shape)

    Returns:
        Scalar MSE loss on wrapped phase difference
    """
    rendered_c = to_complex(rendered)
    gt_c = to_complex(gt)

    dphi = torch.angle(rendered_c) - torch.angle(gt_c)
    # Wrap to [-π, π] using atan2(sin, cos)
    dphi = torch.atan2(torch.sin(dphi), torch.cos(dphi))
    return (dphi ** 2).mean()


def magnitude_weighted_unit_phasor_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    use_gt_magnitude: bool = True,
    top_percentile: Optional[float] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute phase loss weighted by magnitude using unit phasors.

    Only penalizes phase error where signal is strong.
    Phase at low-SNR bins is noise - don't waste gradients on it.

    Args:
        rendered: Complex tensor [*, rx, tx, adc] or real/imag (..., 2)
        gt: Complex tensor (same shape)
        use_gt_magnitude: If True, use GT magnitude for weighting (more stable)
        top_percentile: If set (e.g., 0.7), only include bins above this percentile
        eps: Small value for numerical stability

    Returns:
        Scalar loss tensor
    """
    rendered_c = to_complex(rendered)
    gt_c = to_complex(gt)

    # Compute magnitude weights
    mag = gt_c.abs() if use_gt_magnitude else rendered_c.abs()

    # Normalize globally or per-batch depending on dimensions
    if mag.dim() >= 4:  # Has batch/chirp dimension
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
    u_hat = rendered_c / rendered_c.abs().clamp_min(eps)
    u = gt_c / gt_c.abs().clamp_min(eps)

    # |u_hat - u|² for unit vectors = 2(1 - cos(Δφ))
    phase_diff = (u_hat - u).abs().pow(2)

    # Weighted mean
    loss = (w * phase_diff).sum() / w.sum().clamp_min(eps)
    return loss


def ra_magnitude_loss(pred_ra: torch.Tensor, target_ra: torch.Tensor,
                      use_log: bool = False, loss_type: str = "l2",
                      eps: float = 1e-6, log_scale: float = 1.0,
                      window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Unified RA magnitude loss with optional log-compression.

    Supports L1, L2, and SSIM loss types on optionally log-compressed RA images.

    Args:
        pred_ra: Predicted RA tensor (H, W) or complex
        target_ra: Target RA tensor, same shape
        use_log: If True, apply log-compression before loss
        loss_type: "l1" | "l2" | "ssim"
        eps: Numerical stability for log
        log_scale: Scale factor α for log(1 + α·|x|)
        window_size: SSIM window size (for ssim loss_type)
        sigma: SSIM Gaussian sigma (for ssim loss_type)

    Returns:
        Scalar loss tensor
    """
    # Get magnitudes
    pred_mag = get_magnitude(pred_ra)
    target_mag = get_magnitude(target_ra)

    if use_log:
        # Log compression: log(1 + α·|x|)
        pred_mag = torch.log(1.0 + log_scale * pred_mag)
        target_mag = torch.log(1.0 + log_scale * target_mag)

    if loss_type == "ssim":
        # Min-max normalize to [0, 1] for SSIM (matches inspect_ra_coir.py)
        eps_norm = 1e-12
        pred_norm = minmax_normalize_torch(pred_mag, eps=eps_norm)
        target_norm = minmax_normalize_torch(target_mag, eps=eps_norm)
        return compute_ssim_loss(pred_norm, target_norm, window_size, sigma)
    elif loss_type == "l1":
        return torch.mean(torch.abs(pred_mag - target_mag))
    else:  # l2
        return torch.mean((pred_mag - target_mag) ** 2)








