#!/usr/bin/env python3
"""
ADC Data Evaluation Script
==========================

Computes various metrics (L1, L2, SSIM, PSNR, PCC, MSE, MAE, LPIPS) between
simulated ADC data and ground truth data.

Usage:
    python eval_adc.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio as psnr
# import lpips
import os
from pathlib import Path
import sys

from mmir.data.adc_normalization import normalize_adc_signals


def load_adc_data(filepath):
    """Load ADC data from .npy file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = np.load(filepath)
    print(f"Loaded {filepath}: shape {data.shape}, dtype {data.dtype}")
    return data


def complex_to_magnitude_phase(data):
    """Convert complex data to magnitude and phase components."""
    if np.iscomplexobj(data):
        magnitude = np.abs(data)
        phase = np.angle(data)
        return magnitude, phase
    else:
        return data, None


def normalize_data(data, method='minmax'):
    """Normalize data to [0, 1] range."""
    if method == 'minmax':
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            return (data - data_min) / (data_max - data_min)
        else:
            return np.zeros_like(data)
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            return (data - mean) / std
        else:
            return np.zeros_like(data)
    else:
        return data


def compute_metrics(pred, gt, lpips_model=None):
    """Compute all evaluation metrics."""
    metrics = {}
    
    # Convert to float32 if needed
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    # Flatten for some metrics
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # L1 Loss (MAE)
    l1_loss = np.mean(np.abs(pred - gt))
    metrics['L1'] = l1_loss
    
    # L2 Loss (MSE)
    l2_loss = np.mean((pred - gt) ** 2)
    metrics['L2'] = l2_loss
    
    # MSE
    mse = l2_loss
    metrics['MSE'] = mse
    
    # MAE
    mae = l1_loss
    metrics['MAE'] = mae
    
    # PSNR
    if mse > 0:
        psnr_val = 20 * np.log10(np.max(gt) / np.sqrt(mse))
    else:
        psnr_val = float('inf')
    metrics['PSNR'] = psnr_val
    
    # Pearson Correlation Coefficient (PCC)
    try:
        pcc, _ = pearsonr(pred_flat, gt_flat)
        metrics['PCC'] = pcc
    except:
        metrics['PCC'] = 0.0
    
    
    # # LPIPS (requires torch tensors)
    # if lpips_model is not None:
    #     try:
    #         # Convert to torch tensors and add batch/channel dimensions
    #         pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    #         gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
            
    #         # Ensure tensors are in [0, 1] range for LPIPS
    #         pred_tensor = torch.clamp(pred_tensor, 0, 1)
    #         gt_tensor = torch.clamp(gt_tensor, 0, 1)
            
    #         # For complex data, use magnitude
    #         if torch.is_complex(pred_tensor):
    #             pred_tensor = torch.abs(pred_tensor)
    #         if torch.is_complex(gt_tensor):
    #             gt_tensor = torch.abs(gt_tensor)
            
    #         with torch.no_grad():
    #             lpips_val = lpips_model(pred_tensor, gt_tensor).item()
    #         metrics['LPIPS'] = lpips_val
    #     except Exception as e:
    #         print(f"LPIPS computation failed: {e}")
    #         metrics['LPIPS'] = float('inf')
    
    return metrics


def evaluate_adc_data():
    """Main evaluation function."""
    print("ADC Data Evaluation")
    print("=" * 50)
    
    # File paths
    baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    optimized_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    gt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    
    # Load data
    print("Loading data...")
    try:
        baseline_data = load_adc_data(baseline_path)
        optimized_data = load_adc_data(optimized_path)
        gt_data = load_adc_data(gt_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Process ground truth data to match baseline/optimized format
    print("Processing ground truth data...")
    print(f"Original GT shape: {gt_data.shape}")
    # GT is (NUM_CHIRPS, NUM_RX, NUM_TX, NUM_ADC) - complex valued
    # Extract first chirp: (1, 4, 3, 128)
    gt_first_chirp = gt_data[0:1, :, :, :]
    print(f"After first chirp extraction: {gt_first_chirp.shape}")
    # Squeeze to remove chirp dimension: (4, 3, 128)
    gt_squeezed = np.squeeze(gt_first_chirp, axis=0)
    print(f"After squeeze: {gt_squeezed.shape}")
    # Reshape from (NUM_RX, NUM_TX, NUM_ADC) to (NUM_TX, NUM_RX, NUM_ADC): (3, 4, 128)
    gt_reshaped = np.transpose(gt_squeezed, (1, 0, 2))
    print(f"After reshape: {gt_reshaped.shape}")
    # Convert from complex to real-valued with separate real/imag channels: (3, 4, 128, 2)
    gt_real = np.real(gt_reshaped)
    gt_imag = np.imag(gt_reshaped)
    gt_processed = np.stack([gt_real, gt_imag], axis=-1)
    print(f"Final GT shape: {gt_processed.shape}")
    # Update gt_data to use processed version
    gt_data = gt_processed

    # Normalize all data using energy normalization (same as training)
    print("Normalizing data using energy normalization...")
    # Normalize baseline data
    baseline_normalized, baseline_stats = normalize_adc_signals(
        baseline_data, method="energy", return_stats=True
    )
    print(f"Baseline normalization stats: {baseline_stats}")
    # Normalize optimized data
    optimized_normalized, optimized_stats = normalize_adc_signals(
        optimized_data, method="energy", return_stats=True
    )
    print(f"Optimized normalization stats: {optimized_stats}")
    # Normalize ground truth data
    gt_normalized, gt_stats = normalize_adc_signals(
        gt_data, method="energy", return_stats=True
    )
    print(f"GT normalization stats: {gt_stats}")
    
    # Update data variables
    baseline_data = baseline_normalized
    optimized_data = optimized_normalized
    gt_data = gt_normalized
    print(f"Normalized shapes - Baseline: {baseline_data.shape}, Optimized: {optimized_data.shape}, GT: {gt_data.shape}")
    
    # # Initialize LPIPS model
    # print("Initializing LPIPS model...")
    # try:
    #     lpips_model = lpips.LPIPS(net='alex')
    # except Exception as e:
    #     print(f"LPIPS initialization failed: {e}")
    #     lpips_model = None
    
    # Ensure all data have the same shape
    print(f"Data shapes - Baseline: {baseline_data.shape}, Optimized: {optimized_data.shape}, GT: {gt_data.shape}")
    
    # Resize if necessary (simple nearest neighbor)
    target_shape = gt_data.shape
    if baseline_data.shape != target_shape:
        print(f"Resizing baseline data from {baseline_data.shape} to {target_shape}")
        baseline_data = torch.from_numpy(baseline_data).unsqueeze(0).unsqueeze(0)
        baseline_data = F.interpolate(baseline_data, size=target_shape, mode='nearest')
        baseline_data = baseline_data.squeeze().numpy()
    
    if optimized_data.shape != target_shape:
        print(f"Resizing optimized data from {optimized_data.shape} to {target_shape}")
        optimized_data = torch.from_numpy(optimized_data).unsqueeze(0).unsqueeze(0)
        optimized_data = F.interpolate(optimized_data, size=target_shape, mode='nearest')
        optimized_data = optimized_data.squeeze().numpy()
    
    # Compute metrics for baseline vs GT
    print("\nComputing metrics for Baseline vs Ground Truth...")
    # baseline_metrics = compute_metrics(baseline_data, gt_data, lpips_model)
    baseline_metrics = compute_metrics(baseline_data, gt_data)
    
    # Compute metrics for optimized vs GT
    print("Computing metrics for Optimized vs Ground Truth...")
    # optimized_metrics = compute_metrics(optimized_data, gt_data, lpips_model)
    optimized_metrics = compute_metrics(optimized_data, gt_data)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n{'Metric':<10} {'Baseline vs GT':<15} {'Optimized vs GT':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for metric in ['L1', 'L2', 'MSE', 'MAE', 'PSNR', 'PCC']:
        if metric in baseline_metrics and metric in optimized_metrics:
            baseline_val = baseline_metrics[metric]
            optimized_val = optimized_metrics[metric]
            
            # Calculate improvement
            if metric in ['PSNR', 'PCC']:
                # Higher is better
                improvement = optimized_val - baseline_val
                improvement_str = f"{improvement:+.4f}"
            else:
                # Lower is better
                improvement = baseline_val - optimized_val
                improvement_str = f"{improvement:+.4f}"
            
            print(f"{metric:<10} {baseline_val:<15.4f} {optimized_val:<15.4f} {improvement_str:<15}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    better_metrics = 0
    total_metrics = 0
    
    for metric in ['L1', 'L2', 'MSE', 'MAE', 'PSNR', 'PCC']:
        if metric in baseline_metrics and metric in optimized_metrics:
            total_metrics += 1
            baseline_val = baseline_metrics[metric]
            optimized_val = optimized_metrics[metric]
            
            if metric in ['PSNR', 'PCC']:
                if optimized_val > baseline_val:
                    better_metrics += 1
            else:
                if optimized_val < baseline_val:
                    better_metrics += 1
    
    print(f"Optimized model performs better on {better_metrics}/{total_metrics} metrics")
    
    if better_metrics > total_metrics / 2:
        print("[PASS] Training/inverse rendering process shows significant improvement!")
    else:
        print("[WARNING] Training/inverse rendering process shows limited improvement")


if __name__ == "__main__":
    evaluate_adc_data()
