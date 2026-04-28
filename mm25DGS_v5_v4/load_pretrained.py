"""Load mmIR-trained antenna patterns and scene config for v4.

Vendored from mm25DGS_v2/render_mmIR.py — only the bits c6 needs.

These functions load EXTERNAL artifacts from the mmIR training run
(/home/adnan/Desktop/mmIR/output/train_v13/<scene>/) and the scene
config files. The trained materials are used only as starting values
for the antenna patterns; the per-vertex material parameters are NOT
loaded (c6 trains its own per-Gaussian materials from scratch).
"""

import os
import sys
import numpy as np

# Project root for `train.py` import (TrainingConfigSionna lives there)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import load_config_from_json

# External path constants — match the mmIR training output layout
TRAIN_OUTPUT_DIR = '/home/adnan/Desktop/mmIR/output/train_v13'
DATA_DIR = '/home/adnan/Desktop/mm3DGS/data'

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]


def _remap_path(path: str) -> str:
    """Remap paths from /home/adnan/Desktop/mmIR/ to /home/adnan/Desktop/mm3DGS/."""
    return path.replace('/home/adnan/Desktop/mmIR/', '/home/adnan/Desktop/mm3DGS/')


def load_trained_config(scene: str):
    """Load and path-remap the training config for a scene.

    Returns a TrainingConfigSionna object with .config_file, .scene_file
    (mesh.ply), .gt_adc_file, .tx_pattern_file, .rx_pattern_file all pointing
    into mm3DGS/data and mm3DGS/assets.
    """
    config_path = os.path.join(TRAIN_OUTPUT_DIR, scene, 'config.json')
    config = load_config_from_json(config_path)
    config.config_file = _remap_path(config.config_file)
    config.scene_file = _remap_path(config.scene_file)
    config.gt_adc_file = _remap_path(config.gt_adc_file)
    config.tx_pattern_file = _remap_path(config.tx_pattern_file)
    config.rx_pattern_file = _remap_path(config.rx_pattern_file)
    return config


def load_pattern_data(scene: str):
    """Load mmIR-trained antenna patterns (TX/RX E and H planes).

    Returns a dict with keys 'tx_E_plane', 'tx_H_plane', 'rx_E_plane',
    'rx_H_plane' or None if best_patterns.npz does not exist for this scene.

    Note: this loads ONLY the antenna patterns. The trained per-vertex
    materials and normals are NOT loaded — c6 starts with default ITU
    concrete materials and trains its own per-Gaussian values.
    """
    pat_path = os.path.join(TRAIN_OUTPUT_DIR, scene, 'best_patterns.npz')
    if not os.path.exists(pat_path):
        return None
    pat = np.load(pat_path)
    return {k: pat[k] for k in pat.files}
