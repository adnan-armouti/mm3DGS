"""
ADC Signal Normalization for Radar Training
===========================================

This module provides methods to normalize raw ADC signals in a way that preserves
the visual appearance of range-azimuth maps while making the signals suitable for
training (similar to how RGB images are normalized to [0,1] in computer graphics).

The key insight is that we want to normalize the raw ADC signals before FFT processing,
but in a way that the final range-azimuth maps look identical when plotted.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def normalize_adc_signals(
    adc_data: Union[np.ndarray, torch.Tensor],
    method: str = "energy",
    eps: float = 1e-12,
    return_stats: bool = False
) -> Union[Union[np.ndarray, torch.Tensor], Tuple[Union[np.ndarray, torch.Tensor], dict]]:
    """
    Normalize raw ADC signals while preserving visual appearance of range-azimuth maps.
    
    Parameters
    ----------
    adc_data : np.ndarray or torch.Tensor
        Raw ADC data of shape (N_TX, N_RX, N_ADC, 2) where last dim is [real, imag]
        or complex-valued data of shape (N_TX, N_RX, N_ADC)
    method : str, optional
        Normalization method:
        - "energy": Normalize by total signal energy (preserves relative amplitudes)
        - "max": Normalize by maximum absolute value across all signals
        - "rms": Normalize by root mean square across all signals
        - "per_channel": Normalize each TX-RX channel independently
    eps : float, optional
        Small epsilon to prevent division by zero
    return_stats : bool, optional
        Whether to return normalization statistics
    
    Returns
    -------
    normalized_data : np.ndarray or torch.Tensor
        Normalized ADC data with same shape as input
    stats : dict, optional
        Normalization statistics (only if return_stats=True)
    """
    is_torch = isinstance(adc_data, torch.Tensor)
    is_complex = adc_data.dtype in [np.complex64, np.complex128] or (
        is_torch and adc_data.dtype in [torch.complex64, torch.complex128]
    )
    
    # Convert to complex if needed
    if not is_complex and adc_data.shape[-1] == 2:
        if is_torch:
            adc_complex = adc_data[..., 0] + 1j * adc_data[..., 1]
        else:
            adc_complex = adc_data[..., 0] + 1j * adc_data[..., 1]
    else:
        adc_complex = adc_data
    
    # Apply normalization based on method
    if method == "energy":
        # Normalize by total signal energy (preserves relative amplitudes)
        if is_torch:
            energy = torch.sum(torch.abs(adc_complex) ** 2)
            norm_factor = torch.sqrt(energy + eps)
        else:
            energy = np.sum(np.abs(adc_complex) ** 2)
            norm_factor = np.sqrt(energy + eps)
        
        normalized = adc_complex / norm_factor
        stats = {"method": method, "norm_factor": norm_factor, "energy": energy}
        
    elif method == "max":
        # Normalize by maximum absolute value
        if is_torch:
            max_val = torch.max(torch.abs(adc_complex))
        else:
            max_val = np.max(np.abs(adc_complex))
        
        normalized = adc_complex / (max_val + eps)
        stats = {"method": method, "norm_factor": max_val}
        
    elif method == "rms":
        # Normalize by root mean square
        if is_torch:
            rms = torch.sqrt(torch.mean(torch.abs(adc_complex) ** 2))
        else:
            rms = np.sqrt(np.mean(np.abs(adc_complex) ** 2))
        
        normalized = adc_complex / (rms + eps)
        stats = {"method": method, "norm_factor": rms}
        
    elif method == "per_channel":
        # Normalize each TX-RX channel independently
        if is_torch:
            # Compute max for each TX-RX pair
            max_vals = torch.max(torch.abs(adc_complex), dim=-1, keepdim=True)[0]
            normalized = adc_complex / (max_vals + eps)
        else:
            # Compute max for each TX-RX pair
            max_vals = np.max(np.abs(adc_complex), axis=-1, keepdims=True)
            normalized = adc_complex / (max_vals + eps)
        
        stats = {"method": method, "per_channel": True}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Convert back to real-imag format if input was in that format
    if not is_complex and adc_data.shape[-1] == 2:
        if is_torch:
            normalized = torch.stack([normalized.real, normalized.imag], dim=-1)
        else:
            normalized = np.stack([normalized.real, normalized.imag], axis=-1)
    
    if return_stats:
        return normalized, stats
    else:
        return normalized


def denormalize_adc_signals(
    normalized_data: Union[np.ndarray, torch.Tensor],
    stats: dict,
    eps: float = 1e-12
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize ADC signals using the statistics from normalization.
    
    Parameters
    ----------
    normalized_data : np.ndarray or torch.Tensor
        Normalized ADC data
    stats : dict
        Statistics from normalize_adc_signals
    eps : float, optional
        Small epsilon to prevent division by zero
    
    Returns
    -------
    denormalized_data : np.ndarray or torch.Tensor
        Denormalized ADC data
    """
    method = stats["method"]
    
    is_torch = isinstance(normalized_data, torch.Tensor)
    is_complex = normalized_data.dtype in [np.complex64, np.complex128] or (
        is_torch and normalized_data.dtype in [torch.complex64, torch.complex128]
    )
    
    # Convert to complex if needed
    if not is_complex and normalized_data.shape[-1] == 2:
        if is_torch:
            data_complex = normalized_data[..., 0] + 1j * normalized_data[..., 1]
        else:
            data_complex = normalized_data[..., 0] + 1j * normalized_data[..., 1]
    else:
        data_complex = normalized_data
    
    # Apply denormalization
    if method == "energy":
        norm_factor = stats["norm_factor"]
        denormalized = data_complex * norm_factor
        
    elif method == "max":
        norm_factor = stats["norm_factor"]
        denormalized = data_complex * norm_factor
        
    elif method == "rms":
        norm_factor = stats["norm_factor"]
        denormalized = data_complex * norm_factor
        
    elif method == "per_channel":
        # For per-channel normalization, we need the original max values
        if "original_max_vals" not in stats:
            raise ValueError("Per-channel denormalization requires original_max_vals in stats")
        
        if is_torch:
            denormalized = data_complex * (stats["original_max_vals"] + eps)
        else:
            denormalized = data_complex * (stats["original_max_vals"] + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Convert back to real-imag format if input was in that format
    if not is_complex and normalized_data.shape[-1] == 2:
        if is_torch:
            denormalized = torch.stack([denormalized.real, denormalized.imag], dim=-1)
        else:
            denormalized = np.stack([denormalized.real, denormalized.imag], axis=-1)
    
    return denormalized


def process_adc_to_range_azimuth(
    adc_data: Union[np.ndarray, torch.Tensor],
    num_adc: int = 256,
    num_vx: int = 86,
    num_azimuth_bins: int = 128,
    range_resolution: float = 0.117,
    apply_hann: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Process ADC data to range-azimuth map using the same pipeline as in the notebook.
    
    Parameters
    ----------
    adc_data : np.ndarray or torch.Tensor
        ADC data of shape (N_TX, N_RX, N_ADC, 2) or (N_TX, N_RX, N_ADC) complex
    num_adc : int
        Number of ADC samples
    num_vx : int
        Number of virtual antennas
    num_azimuth_bins : int
        Number of azimuth bins for FFT
    range_resolution : float
        Range resolution in meters
    apply_hann : bool
        Whether to apply Hann windowing
    
    Returns
    -------
    range_azimuth_map : np.ndarray or torch.Tensor
        Range-azimuth map of shape (num_azimuth_bins-1, num_adc)
    """
    is_torch = isinstance(adc_data, torch.Tensor)
    
    # Convert to complex if needed
    if adc_data.shape[-1] == 2:
        if is_torch:
            adc_complex = adc_data[..., 0] + 1j * adc_data[..., 1]
        else:
            adc_complex = adc_data[..., 0] + 1j * adc_data[..., 1]
    else:
        adc_complex = adc_data
    
    # Reshape to virtual array (simplified - you may need to implement txrx_to_vx_chirps)
    # For now, we'll work with the first TX-RX pair
    if adc_complex.shape[0] > 1 or adc_complex.shape[1] > 1:
        # Take the first TX-RX pair for simplicity
        adc_curr = adc_complex[0, 0, :, :] if adc_complex.ndim == 4 else adc_complex[0, 0, :]
    else:
        adc_curr = adc_complex.squeeze()
    
    # Apply Hann windowing if requested
    if apply_hann:
        if is_torch:
            hann_adc = torch.hann_window(num_adc, device=adc_curr.device)
            hann_vx = torch.hann_window(num_vx, device=adc_curr.device)
            adc_curr = adc_curr * hann_adc[None, :]
        else:
            hann_adc = np.hanning(num_adc)
            hann_vx = np.hanning(num_vx)
            adc_curr = adc_curr * hann_adc[None, :]
    
    # Apply range FFT
    if is_torch:
        adc_curr = torch.fft.fft(adc_curr, n=num_adc, dim=-1)
    else:
        adc_curr = np.fft.fft(adc_curr, n=num_adc, axis=-1)
    
    # Apply azimuth FFT
    if apply_hann:
        if is_torch:
            adc_curr = adc_curr * hann_vx[:, None]
        else:
            adc_curr = adc_curr * hann_vx[:, None]
    
    if is_torch:
        adc_curr = torch.fft.ifftshift(adc_curr, dims=(0,))
        adc_curr = torch.fft.fft(adc_curr, n=num_azimuth_bins, dim=0)
        adc_curr = adc_curr[1:, :]  # Cut off first azimuth angle bin
        adc_curr = torch.fft.fftshift(adc_curr, dims=(0,))
        range_azimuth_map = torch.abs(adc_curr)
    else:
        adc_curr = np.fft.ifftshift(adc_curr, axes=(0,))
        adc_curr = np.fft.fft(adc_curr, n=num_azimuth_bins, axis=0)
        adc_curr = adc_curr[1:, :]  # Cut off first azimuth angle bin
        adc_curr = np.fft.fftshift(adc_curr, axes=(0,))
        range_azimuth_map = np.abs(adc_curr)
    
    return range_azimuth_map


def verify_normalization_preserves_visualization(
    original_adc: Union[np.ndarray, torch.Tensor],
    normalized_adc: Union[np.ndarray, torch.Tensor],
    tolerance: float = 1e-3
) -> bool:
    """
    Verify that normalization preserves the visual appearance of range-azimuth maps.
    
    Parameters
    ----------
    original_adc : np.ndarray or torch.Tensor
        Original ADC data
    normalized_adc : np.ndarray or torch.Tensor
        Normalized ADC data
    tolerance : float
        Tolerance for numerical comparison
    
    Returns
    -------
    preserves_visualization : bool
        True if the range-azimuth maps are visually identical
    """
    # Process both to range-azimuth maps
    original_ra = process_adc_to_range_azimuth(original_adc)
    normalized_ra = process_adc_to_range_azimuth(normalized_adc)
    
    # Check if they are proportional (same visual appearance)
    if isinstance(original_ra, torch.Tensor):
        # Compute ratio between corresponding elements
        ratio = original_ra / (normalized_ra + 1e-12)
        # Check if ratio is constant (within tolerance)
        ratio_std = torch.std(ratio)
        return ratio_std < tolerance
    else:
        # Compute ratio between corresponding elements
        ratio = original_ra / (normalized_ra + 1e-12)
        # Check if ratio is constant (within tolerance)
        ratio_std = np.std(ratio)
        return ratio_std < tolerance


# Example usage and testing functions
def test_normalization():
    """Test the normalization methods with synthetic data."""
    # Create synthetic ADC data
    np.random.seed(42)
    original_adc = np.random.randn(12, 16, 256, 2) * 100  # Large scale
    
    print("Testing normalization methods...")
    
    for method in ["energy", "max", "rms", "per_channel"]:
        print(f"\nMethod: {method}")
        
        # Normalize
        normalized_adc, stats = normalize_adc_signals(original_adc, method=method, return_stats=True)
        
        # Check if visualization is preserved
        preserves_viz = verify_normalization_preserves_visualization(original_adc, normalized_adc)
        print(f"  Preserves visualization: {preserves_viz}")
        
        # Check scale of normalized data
        if method == "energy":
            energy = np.sum(np.abs(normalized_adc[..., 0] + 1j * normalized_adc[..., 1]) ** 2)
            print(f"  Normalized energy: {energy:.6f}")
        elif method == "max":
            max_val = np.max(np.abs(normalized_adc[..., 0] + 1j * normalized_adc[..., 1]))
            print(f"  Normalized max: {max_val:.6f}")
        
        # Test denormalization
        denormalized_adc = denormalize_adc_signals(normalized_adc, stats)
        denorm_error = np.max(np.abs(original_adc - denormalized_adc))
        print(f"  Denormalization error: {denorm_error:.2e}")


if __name__ == "__main__":
    test_normalization()
