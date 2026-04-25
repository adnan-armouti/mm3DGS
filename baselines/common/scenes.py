"""Benchmark scene list — the single source of truth for the 7 scenes.

Excluded scenes (documented in ``baselines/README.md``):
    seq_1_frame_277, seq_0_frame_451.
"""

from __future__ import annotations

import os

# 7 scenes, alphabetical. Every baseline imports from here; do not hard-code
# lists in per-baseline modules.
BENCHMARK_SCENES: tuple[str, ...] = (
    "seq_0_frame_135",
    "seq_0_frame_390",
    "seq_1_frame_185",
    "seq_1_frame_438",
    "seq_2_frame_105",
    "seq_2_frame_160",
    "seq_2_frame_300",
)

EXCLUDED_SCENES: tuple[str, ...] = (
    "seq_0_frame_451",
    "seq_1_frame_277",
)

# Project root (two levels up from this file: baselines/common/scenes.py → repo root).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
ALIGNMENT_ROOT = os.path.join(DATA_ROOT, "alignment_data")


def scene_dir(scene: str) -> str:
    return os.path.join(DATA_ROOT, scene)


def cascade_alignment_dir(scene: str) -> str:
    """Where per-frame aligned cascaded configs live (one per 9 frames)."""
    return os.path.join(ALIGNMENT_ROOT, scene, "cascade")


def single_chip_alignment_dir(scene: str) -> str:
    """Where per-frame aligned single-chip configs live (one per 9 frames)."""
    return os.path.join(ALIGNMENT_ROOT, scene, "single_chip")
