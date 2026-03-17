"""I/O and configuration utilities for training."""

import os
import sys
import json
import contextlib
from typing import Dict, Any


@contextlib.contextmanager
def suppress_io(enabled: bool):
    """Suppress stdout/stderr when enabled=True."""
    if not enabled:
        yield
        return
    devnull = open(os.devnull, 'w')
    oldout, olderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        devnull.close()


def compute_range_res_from_cfg(cfg_path: str) -> float:
    """Compute range resolution R = c / (2  S  (rampEndTime - adcStartTime))."""
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    c = 299_792_458.0
    slope_hz_per_s = float(cfg["freqSlope"])  # Hz/s
    t_adc = float(cfg["adcStartTime"])        # s
    t_end = float(cfg["rampEndTime"])         # s
    T_chirp = t_end - t_adc
    if slope_hz_per_s <= 0 or T_chirp <= 0:
        return 0.117  # sensible default from prior code
    bandwidth = slope_hz_per_s * T_chirp
    return float(c / (2.0 * bandwidth))


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_training_config(config: Dict[str, Any], output_path: str) -> None:
    """Save training configuration to JSON file."""
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
