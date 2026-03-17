"""ADC-level reconstruction metrics.

Computes metrics on complex-valued ADC data: magnitude (log + min-max normalized),
phase (unit-phasor or relative), and virtual-array phase coherence.
"""

from typing import Dict

import numpy as np

from mmir.losses.loss_utils import minmax_normalize_numpy


def _compute_va_phase_coherence(rendered_complex, gt_complex, top_percentile=0.9):
    """Virtual Array Phase Coherence metric.

    For each range bin, measures how well the relative phase pattern across
    the virtual array (TX x RX) matches between rendered and GT ADC.

    Complex correlation: rho[k] = |v_hat_r . conj(v_hat_g)| / N_vx
    where v_hat = v / |v| (unit-magnitude normalization per element).
    Global phase offsets cancel via the magnitude of the dot product.

    Averaged over top-percentile range bins by GT energy.

    Returns value in [0, 1] where 1 = perfect relative phase match,
    ~1/N_vx = random phase.
    """
    NT, NR, K = rendered_complex.shape
    N_vx = NT * NR
    eps = 1e-10

    r = rendered_complex.reshape(N_vx, K)
    g = gt_complex.reshape(N_vx, K)

    gt_energy = np.sum(np.abs(g) ** 2, axis=0)
    thresh = np.percentile(gt_energy, top_percentile * 100)
    valid = gt_energy >= thresh

    if np.sum(valid) < 2:
        return float('nan')

    r_hat = r / (np.abs(r) + eps)
    g_hat = g / (np.abs(g) + eps)

    dot = np.sum(r_hat[:, valid] * np.conj(g_hat[:, valid]), axis=0)
    rho = np.abs(dot) / N_vx

    return float(np.mean(rho))


def compute_adc_metrics(
    rendered_adc_complex: np.ndarray,
    gt_adc_complex: np.ndarray,
    adc_use_log: bool = False,
    log_epsilon: float = 1e-6,
    log_scale: float = 1.0,
    top_percentile: float = 0.9,
    phase_metrics_type: str = "unit_phasor"
) -> Dict[str, float]:
    """Compute reconstruction metrics on complex-valued ADC data.

    Computes metrics with BOTH log and min-max normalization:
    1. Log-normalized ADC magnitude: L1, L2, MSE, RMSE, MAE, Pearson
    2. Min-max normalized ADC magnitude: L1, L2, MSE, RMSE, MAE, Pearson
    3. ADC Phase: unit phasor 2(1-cos(delta_phi)) or relative phase metrics
    """
    metrics = {}
    eps = 1e-10

    rendered_mag = np.abs(rendered_adc_complex)
    gt_mag = np.abs(gt_adc_complex)

    # Log normalization
    rendered_mag_log = np.log(log_epsilon + log_scale * rendered_mag)
    gt_mag_log = np.log(log_epsilon + log_scale * gt_mag)
    # Min-max normalization
    rendered_mag_minmax = minmax_normalize_numpy(rendered_mag)
    gt_mag_minmax = minmax_normalize_numpy(gt_mag)

    # === Log-normalized ADC magnitude metrics ===
    mag_diff_log = rendered_mag_log - gt_mag_log
    metrics['adc_log_mag_l1'] = float(np.mean(np.abs(mag_diff_log)))
    metrics['adc_log_mag_l2'] = float(np.sum(mag_diff_log ** 2))
    metrics['adc_log_mag_mse'] = float(np.mean(mag_diff_log ** 2))
    metrics['adc_log_mag_rmse'] = float(np.sqrt(metrics['adc_log_mag_mse']))
    metrics['adc_log_mag_mae'] = metrics['adc_log_mag_l1']
    r_log_flat = rendered_mag_log.flatten()
    g_log_flat = gt_mag_log.flatten()
    if np.std(r_log_flat) < eps or np.std(g_log_flat) < eps:
        metrics['adc_log_mag_pearson'] = 0.0
    else:
        metrics['adc_log_mag_pearson'] = float(np.corrcoef(r_log_flat, g_log_flat)[0, 1])

    # === Min-max normalized ADC magnitude metrics ===
    mag_diff_mm = rendered_mag_minmax - gt_mag_minmax
    metrics['adc_minmax_mag_l1'] = float(np.mean(np.abs(mag_diff_mm)))
    metrics['adc_minmax_mag_l2'] = float(np.sum(mag_diff_mm ** 2))
    metrics['adc_minmax_mag_mse'] = float(np.mean(mag_diff_mm ** 2))
    metrics['adc_minmax_mag_rmse'] = float(np.sqrt(metrics['adc_minmax_mag_mse']))
    metrics['adc_minmax_mag_mae'] = metrics['adc_minmax_mag_l1']
    r_mm_flat = rendered_mag_minmax.flatten()
    g_mm_flat = gt_mag_minmax.flatten()
    if np.std(r_mm_flat) < eps or np.std(g_mm_flat) < eps:
        metrics['adc_minmax_mag_pearson'] = 0.0
    else:
        metrics['adc_minmax_mag_pearson'] = float(np.corrcoef(r_mm_flat, g_mm_flat)[0, 1])

    # === Legacy metrics (based on adc_use_log flag) ===
    mag_diff = mag_diff_log if adc_use_log else mag_diff_mm
    r_norm_flat = r_log_flat if adc_use_log else r_mm_flat
    g_norm_flat = g_log_flat if adc_use_log else g_mm_flat

    metrics['adc_l1'] = float(np.mean(np.abs(mag_diff)))
    metrics['adc_l2'] = float(np.sum(mag_diff ** 2))
    metrics['adc_mse'] = float(np.mean(mag_diff ** 2))
    metrics['adc_rmse'] = float(np.sqrt(metrics['adc_mse']))
    metrics['adc_mae'] = metrics['adc_l1']
    if np.std(r_norm_flat) < eps or np.std(g_norm_flat) < eps:
        metrics['adc_pearson'] = 0.0
    else:
        metrics['adc_pearson'] = float(np.corrcoef(r_norm_flat, g_norm_flat)[0, 1])
    for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
        metrics[f'adc_mag_{s}'] = metrics[f'adc_{s}']

    # === ADC Phase Metrics ===
    NT, NR, K = rendered_adc_complex.shape
    mag_threshold = np.percentile(gt_mag.flatten(), top_percentile * 100)
    valid_mask = gt_mag >= mag_threshold

    if phase_metrics_type == "unit_phasor":
        if np.sum(valid_mask) > 10:
            rendered_phase = np.angle(rendered_adc_complex)
            gt_phase = np.angle(gt_adc_complex)
            phase_diff = rendered_phase - gt_phase
            unit_phasor_loss = 2.0 * (1.0 - np.cos(phase_diff))
            metrics['adc_phase_l1'] = float(np.mean(np.abs(unit_phasor_loss[valid_mask])))
            metrics['adc_phase_mse'] = float(np.mean(unit_phasor_loss[valid_mask]))
            metrics['adc_phase_rmse'] = float(np.sqrt(metrics['adc_phase_mse']))
            metrics['adc_phase_mae'] = float(np.mean(unit_phasor_loss[valid_mask]))
            metrics['adc_phase_l2'] = float(np.sum(unit_phasor_loss[valid_mask]))
            cos_r = np.cos(rendered_phase[valid_mask]).flatten()
            cos_g = np.cos(gt_phase[valid_mask]).flatten()
            if np.std(cos_r) < eps or np.std(cos_g) < eps:
                metrics['adc_phase_pearson'] = 0.0
            else:
                metrics['adc_phase_pearson'] = float(np.corrcoef(cos_r, cos_g)[0, 1])
        else:
            for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
                metrics[f'adc_phase_{s}'] = float('nan')
    else:  # relative_phase
        if NR > 1 and np.sum(valid_mask) > 10:
            r_ref = rendered_adc_complex[:, 0:1, :]
            g_ref = gt_adc_complex[:, 0:1, :]
            r_rel = np.angle(rendered_adc_complex * np.conj(r_ref))
            g_rel = np.angle(gt_adc_complex * np.conj(g_ref))
            pd = r_rel - g_rel
            pd = np.arctan2(np.sin(pd), np.cos(pd))
            pd_valid = pd[valid_mask] / np.pi
            metrics['adc_phase_l1'] = float(np.mean(np.abs(pd_valid)))
            metrics['adc_phase_l2'] = float(np.sum(pd_valid ** 2))
            metrics['adc_phase_mse'] = float(np.mean(pd_valid ** 2))
            metrics['adc_phase_rmse'] = float(np.sqrt(metrics['adc_phase_mse']))
            metrics['adc_phase_mae'] = metrics['adc_phase_l1']
            r_rel_v = r_rel[valid_mask].flatten()
            g_rel_v = g_rel[valid_mask].flatten()
            if np.std(r_rel_v) < eps or np.std(g_rel_v) < eps:
                metrics['adc_phase_pearson'] = 0.0
            else:
                metrics['adc_phase_pearson'] = float(np.corrcoef(r_rel_v, g_rel_v)[0, 1])
        else:
            for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
                metrics[f'adc_phase_{s}'] = float('nan') if NR <= 1 else 0.0

    # Phase aliases for dashboard compatibility
    for pfx in ['adc_relphase', 'adc_log_phase', 'adc_minmax_phase']:
        for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
            metrics[f'{pfx}_{s}'] = metrics[f'adc_phase_{s}']

    # Virtual array phase coherence
    metrics['va_phase_coherence'] = _compute_va_phase_coherence(
        rendered_adc_complex, gt_adc_complex, top_percentile)

    return metrics
