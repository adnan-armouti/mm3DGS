"""Data loading and processing utilities for training."""

import os
import glob
import json
import torch
import numpy as np
from typing import Iterator, Tuple, Dict

from mmir.data.adc_normalization import normalize_adc_signals


def load_target(path: str, device: str, normalize: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Target ADC frame (12,16,256,2) as float32 real-imag tensor.
    Optionally applies energy normalization.
    
    Returns:
        target: Normalized or unnormalized target tensor
        norm_stats: Normalization statistics (empty dict if not normalized)
    """
    arr  = np.load(path)                       # (1,256,12,16) complex64
    cmpl = np.squeeze(arr[0]).transpose(1, 0, 2)
    target = torch.view_as_real(torch.from_numpy(cmpl)).float().to(device)
    
    norm_stats = {}
    if normalize:
        # Convert to numpy for normalization
        target_np = target.detach().cpu().numpy()
        normalized_np, norm_stats = normalize_adc_signals(
            target_np, method="energy", return_stats=True
        )
        # Convert back to tensor
        target = torch.from_numpy(normalized_np).to(device)
    
    return target, norm_stats


def iter_views(scene_root: str) -> Iterator[Tuple[str, str]]:
    """Yield (adc_path, cfg_path) pairs for cascaded frames under scene_root."""
    radar_glob = os.path.join(scene_root, "radar", "cascaded_frame_*.npy")
    for adc_path in sorted(glob.glob(radar_glob)):
        stem = os.path.splitext(os.path.basename(adc_path))[0]  # cascaded_frame_XXX
        idx  = stem.split("_")[-1]
        cfg_path = os.path.join(scene_root, "configs", f"cascaded_frame_{idx}.json")
        yield adc_path, cfg_path


def scan_expected_counts(scene_root: str) -> Tuple[int, int]:
    """Scan all cascaded configs and ensure consistent TX/RX counts.

    Returns (n_tx, n_rx) if consistent; raises ValueError otherwise.
    """
    cfg_glob = os.path.join(scene_root, "configs", "cascaded_frame_*.json")
    n_tx_set: set[int] = set()
    n_rx_set: set[int] = set()
    any_cfgs = 0
    for cfg_path in sorted(glob.glob(cfg_glob)):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            n_tx_set.add(len(cfg.get("tx_array", [])))
            n_rx_set.add(len(cfg.get("rx_array", [])))
            any_cfgs += 1
        except Exception:
            continue
    if any_cfgs == 0:
        raise FileNotFoundError(f"No cascaded_frame_*.json under {os.path.join(scene_root, 'configs')}")
    if len(n_tx_set) != 1 or len(n_rx_set) != 1:
        raise ValueError(f"Inconsistent TX/RX counts across configs: TX={sorted(n_tx_set)}, RX={sorted(n_rx_set)}")
    return int(next(iter(n_tx_set))), int(next(iter(n_rx_set)))










