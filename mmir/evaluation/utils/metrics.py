"""Metric computation utilities — wraps eval_adc and eval_single_view."""

from typing import Optional

import numpy as np


def compute_adc_metrics(pred: np.ndarray, gt: np.ndarray, lpips_model=None) -> dict:
    """Compute ADC-level metrics between predicted and ground truth.

    Wraps mmir.evaluation.utils.eval_adc.compute_metrics().

    Args:
        pred: Predicted ADC or RA magnitude array
        gt: Ground truth ADC or RA magnitude array
        lpips_model: Optional LPIPS model for perceptual loss

    Returns dict with: pearson_corr, psnr, ssim, mse, mae, lpips (if model provided)
    """
    from mmir.evaluation.utils.eval_adc import compute_metrics
    return compute_metrics(pred, gt, lpips_model=lpips_model)


def compute_ra_metrics(ra_pred: np.ndarray, ra_gt: np.ndarray) -> dict:
    """Compute RA-level metrics: correlation, PSNR, SSIM, MSE.

    Both inputs should be 2D magnitude arrays of the same shape.
    """
    from scipy.stats import pearsonr
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    pred = ra_pred.astype(np.float64)
    gt = ra_gt.astype(np.float64)

    # Ensure same shape
    if pred.shape != gt.shape:
        min_h = min(pred.shape[0], gt.shape[0])
        min_w = min(pred.shape[1], gt.shape[1])
        pred = pred[:min_h, :min_w]
        gt = gt[:min_h, :min_w]

    # Pearson correlation
    mask = np.isfinite(pred.ravel()) & np.isfinite(gt.ravel())
    if mask.sum() > 1:
        corr, _ = pearsonr(pred.ravel()[mask], gt.ravel()[mask])
    else:
        corr = 0.0

    # MSE
    mse = float(np.mean((pred - gt) ** 2))

    # PSNR (requires positive data range)
    data_range = max(gt.max() - gt.min(), 1e-10)
    try:
        psnr_val = float(peak_signal_noise_ratio(gt, pred, data_range=data_range))
    except Exception:
        psnr_val = 0.0

    # SSIM
    try:
        win_size = min(7, min(pred.shape[0], pred.shape[1]))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size >= 3:
            ssim_val = float(structural_similarity(gt, pred, data_range=data_range, win_size=win_size))
        else:
            ssim_val = 0.0
    except Exception:
        ssim_val = 0.0

    return {
        "correlation": float(corr),
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mse": mse,
    }


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-30:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_cart_ra_metrics(ra_gt_cart: np.ndarray, ra_rend_cart: np.ndarray) -> dict:
    """Compute all RA image metrics from Cartesian RA magnitude images.

    Both inputs are 2D float arrays.
    Metrics are computed on independently min-max normalized images.
    """
    gt_norm = _minmax_normalize(ra_gt_cart)
    rend_norm = _minmax_normalize(ra_rend_cart)

    gt_flat = gt_norm.ravel()
    rend_flat = rend_norm.ravel()

    cart_corr = float(np.corrcoef(gt_flat, rend_flat)[0, 1])

    mse = float(np.mean((rend_norm - gt_norm) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")

    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim_val = float(ssim_fn(gt_norm, rend_norm, data_range=1.0))
    except ImportError:
        ssim_val = None

    return {
        "cart_corr": cart_corr,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim_val,
    }


def compute_3d_occupancy_metrics(
    radar_pts: np.ndarray,
    lidar_pts: np.ndarray,
    distance_threshold: float = 0.5,
) -> dict:
    """Compute 3D occupancy metrics between radar and LiDAR point clouds.

    Wraps eval_single_view.compute_distance_based_occupancy_metrics() and
    eval_single_view.compute_point_cloud_rmse_and_chamfer().
    """
    from mmir.evaluation.utils.eval_single_view import (
        compute_distance_based_occupancy_metrics,
        compute_point_cloud_rmse_and_chamfer,
    )

    occ_metrics = compute_distance_based_occupancy_metrics(
        radar_pts, lidar_pts, distance_threshold=distance_threshold
    )
    chamfer_metrics = compute_point_cloud_rmse_and_chamfer(radar_pts, lidar_pts)

    return {**occ_metrics, **chamfer_metrics}
