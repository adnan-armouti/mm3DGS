"""Configuration loading and validation utilities."""

import os
import json
from typing import Dict, Any


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file with validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = [
        "scene_root", "pattern_csv", "registration_mode", 
        "epochs", "lr", "device"
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Set defaults for optional fields
    defaults = {
        "learn_mat": True,
        "learn_vtx": False,
        "learn_nrm": True,
        "learn_pat": True,
        "num_bounces": 1,
        "hits_per_rx": 500,
        "rays_per_res": 16,
        "seed": 42,
        "base_seed": 42,
        "backend": "nccl",
        "accum_steps": 1,
        "ra_loss_weight": 0.0,
        "ra_loss_use_db": False,
        "ra_save_frequency": 10,
        "verbose": False
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    return config


def create_default_config(output_path: str, scene_root: str = None) -> None:
    """Create a default training configuration file."""
    default_config = {
        "scene_root": scene_root or None,
        "pattern_csv": None,
        "registration_mode": "radar",
        "learn_mat": True,
        "learn_vtx": False,
        "learn_nrm": True,
        "learn_pat": True,
        "num_bounces": 1,
        "hits_per_rx": 500,
        "rays_per_res": 16,
        "seed": 42,
        "epochs": 500,
        "lr": 0.1,
        "accum_steps": 1,
        "device": "cuda",
        "ra_loss_weight": 1.0,
        "ra_loss_use_db": False,
        "ra_save_frequency": 10,
        "verbose": False
    }
    
    with open(output_path, "w") as f:
        json.dump(default_config, f, indent=2)
