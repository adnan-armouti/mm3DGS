"""ADC → RA conversion utilities. Wraps mmir.evaluation.utils.single_view_proc."""

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch


def adc_to_ra_polar(
    adc_path: str,
    config_path: Optional[str] = None,
    num_ant: int = 100,
    num_adc: int = 256,
    num_azimuth_bins: int = 128,
    num_elevation_bins: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ADC .npy file to Range-Azimuth polar map.

    Wraps single_view_proc.process_adc_to_ra_map_enhanced().

    Returns:
        ra_map: (n_azimuth, n_range) magnitude array
        range_axis: (n_range,) in meters
        azimuth_axis: (n_azimuth,) in degrees
    """
    from mmir.evaluation.utils.single_view_proc import process_adc_to_ra_map_enhanced

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute range resolution from config if available
    range_res = 0.117  # default
    if config_path and os.path.isfile(config_path):
        range_res = _range_resolution_from_config(config_path)

    ra_map, range_axis, azimuth_axis = process_adc_to_ra_map_enhanced(
        adc_data_path=adc_path,
        angle_deg=0.0,
        DEVICE=device,
        NUM_ANT=num_ant,
        NUM_ADC=num_adc,
        NUM_AZIMUTH_BINS=num_azimuth_bins,
        NUM_ELEVATION_BINS=num_elevation_bins,
        RANGE_RESOLUTION=range_res,
    )
    return ra_map, range_axis, azimuth_axis


def adc_array_to_ra_polar(
    adc_ri: np.ndarray,
    config_path: Optional[str] = None,
    num_ant: int = 100,
    num_azimuth_bins: int = 128,
    num_elevation_bins: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert in-memory ADC array (n_tx, n_rx, n_adc, 2) to RA polar map.

    Saves to a temp file and calls adc_to_ra_polar. This avoids duplicating
    the FFT/beamforming logic in single_view_proc.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, adc_ri)
        tmp_path = f.name

    try:
        num_adc = adc_ri.shape[2]
        return adc_to_ra_polar(
            tmp_path,
            config_path=config_path,
            num_ant=num_ant,
            num_adc=num_adc,
            num_azimuth_bins=num_azimuth_bins,
            num_elevation_bins=num_elevation_bins,
            device=device,
        )
    finally:
        os.unlink(tmp_path)


def _range_resolution_from_config(config_path: str) -> float:
    """Compute range resolution from radar config JSON."""
    with open(config_path) as f:
        cfg = json.load(f)

    c = 3e8
    sample_rate = cfg.get("sampleRate", 8e6)
    freq_slope = cfg.get("freqSlope", 34.014e12)
    num_adc = cfg.get("numAdcSamples", 256)
    bandwidth = sample_rate * num_adc / sample_rate  # just num_adc samples
    # Actual bandwidth = freq_slope * (num_adc / sample_rate)
    bw = freq_slope * (num_adc / sample_rate)
    range_res = c / (2 * bw)
    return range_res


def compute_ra_correlation(
    ra_pred: np.ndarray,
    ra_gt: np.ndarray,
) -> dict:
    """Compute Pearson correlation between two RA maps (polar or Cartesian).

    Both arrays should be 2D magnitude maps of the same shape.

    Returns dict with polar_corr and additional stats.
    """
    from scipy.stats import pearsonr

    # Flatten and compute correlation
    pred_flat = ra_pred.ravel().astype(np.float64)
    gt_flat = ra_gt.ravel().astype(np.float64)

    # Remove any NaN/inf
    mask = np.isfinite(pred_flat) & np.isfinite(gt_flat)
    pred_flat = pred_flat[mask]
    gt_flat = gt_flat[mask]

    if len(pred_flat) < 2:
        return {"correlation": 0.0, "p_value": 1.0}

    corr, p_val = pearsonr(pred_flat, gt_flat)
    return {
        "correlation": float(corr),
        "p_value": float(p_val),
    }
