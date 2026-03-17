"""Load trained materials from our method and the Sionna benchmark."""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


def load_our_materials(training_dir: str) -> Tuple[np.ndarray, dict]:
    """Load best_materials.npz from our training output.

    Returns:
        raw_params: np.ndarray shape (n_vertices, 6) — physics params in unconstrained space
        metadata: dict with available metadata (iteration, correlation, mse)
    """
    path = os.path.join(training_dir, "best_materials.npz")
    data = np.load(path)
    raw_params = data["raw_params"]  # (n_vertices, 6)

    metadata = {}
    for key in ["iteration", "correlation", "mse"]:
        if key in data.files:
            val = data[key]
            metadata[key] = float(val) if np.ndim(val) == 0 else val.tolist()

    return raw_params, metadata


def load_our_physics_params(training_dir: str) -> Tuple[np.ndarray, dict]:
    """Load physics-space material arrays from best_materials.npz.

    Returns:
        physics_params: np.ndarray shape (n_vertices, 6) columns:
            [eps_real, eps_imag, sigma_h, l_c, tau, thickness]
        metadata: dict
    """
    path = os.path.join(training_dir, "best_materials.npz")
    data = np.load(path)

    physics = np.column_stack([
        data["eps_real"],
        data["eps_imag"],
        data["sigma_h"],
        data["l_c"],
        data["tau"],
        data["thickness"],
    ])

    metadata = {}
    for key in ["iteration", "correlation", "mse"]:
        if key in data.files:
            val = data[key]
            metadata[key] = float(val) if np.ndim(val) == 0 else val.tolist()

    return physics, metadata


def load_benchmark_materials(training_dir: str) -> Tuple[dict, dict]:
    """Load best_params.json from Sionna benchmark output.

    Returns:
        params: dict with material params (d, eta_r, s, sigma, xpd_coefficient)
        metadata: dict with iteration, cart_corr, polar_corr, pose
    """
    path = os.path.join(training_dir, "best_params.json")
    with open(path) as f:
        data = json.load(f)

    param_keys = ["d", "eta_r", "s", "sigma", "xpd_coefficient"]
    params = {k: data[k] for k in param_keys if k in data}

    metadata = {}
    for key in ["iteration", "cart_corr", "polar_corr"]:
        if key in data:
            metadata[key] = data[key]
    if "pose" in data:
        metadata["pose"] = data["pose"]

    return params, metadata


def load_our_training_history(training_dir: str) -> dict:
    """Load training_history.json from our training output.

    Returns the raw JSON dict. Key fields are lists indexed by iteration:
        iteration, total_loss, polar_corr, cart_corr, mse, lr_multiplier, ...
    """
    path = os.path.join(training_dir, "training_history.json")
    with open(path) as f:
        return json.load(f)


def load_benchmark_training_history(training_dir: str) -> dict:
    """Load training_history.json from Sionna benchmark output.

    Returns dict with:
        scene, init_cart_corr, init_polar_corr, best_train_cart_corr,
        best_train_iter, final_cart_corr, final_polar_corr,
        history: {iteration: [...], loss: [...], polar_corr: [...], cart_corr: [...], ...}
    """
    path = os.path.join(training_dir, "training_history.json")
    with open(path) as f:
        return json.load(f)


def load_train_config(training_dir: str) -> dict:
    """Load train_config.json (or config.json fallback) — the hyperparameters used during training."""
    path = os.path.join(training_dir, "train_config.json")
    if not os.path.isfile(path):
        path = os.path.join(training_dir, "config.json")
    with open(path) as f:
        return json.load(f)


def load_learned_normals(training_dir: str) -> Optional[np.ndarray]:
    """Load best_normals.npz from training output.

    Returns:
        normal_params: np.ndarray shape (n_vertices, 3) or None if not available.
    """
    path = os.path.join(training_dir, "best_normals.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return data["normal_params"]


def load_learned_patterns(training_dir: str) -> Optional[Dict]:
    """Load best_patterns.npz from training output.

    Returns:
        dict with keys like tx_E_plane, tx_H_plane, rx_E_plane, rx_H_plane,
        or None if not available.
    """
    path = os.path.join(training_dir, "best_patterns.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}
