#!/usr/bin/env python3
"""
Cascaded Radar Alignment Orchestrator
======================================

End-to-end pipeline for aligning the 77 GHz cascaded FMCW radar to the
LiDAR-derived scene mesh.  Two complementary alignment methods are run
and the per-scene winner (highest RA-image correlation) is selected:

1. **LiDAR-based 4-DOF** (``cascaded_lidar`` / ``cascaded_lidar_gpu``):
   Voxelises the LiDAR point cloud into the radar's RAE grid and
   optimises range, azimuth, elevation-rotation, and azimuth-rotation
   to maximise correlation with the measured RA image.

2. **Renderer-based 2-DOF** (``cascaded_renderer``):
   Uses the differentiable Monte-Carlo renderer with 1-bounce to render
   RA images and optimises only range offset + azimuth offset (zero
   boresight rotation, preserving the IMU-derived tilt).

After both methods have been run (or loaded from prior results), the
orchestrator picks the winner per scene and writes the aligned config.

Usage (complete pipeline)::

    from mmir.preprocessing.alignment.cascaded_alignment import (
        run_all, SCENES,
    )
    results = run_all(data_root="data", output_root="output/alignment")

Usage (selection from pre-computed results)::

    from mmir.preprocessing.alignment.cascaded_alignment import (
        get_aligned_config_path, SCENES,
    )
    for scene_name, frame in SCENES:
        path = get_aligned_config_path(scene_name, frame, data_root="data")
        print(f"{scene_name} -> {path}")

The ``ALIGNMENT_RESULTS`` table encodes the validated correlation scores
from the original alignment runs and must match the configs referenced
by the training pipeline (train_v11).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

# ======================================================================
# Scene definitions
# ======================================================================

SCENES: List[Tuple[str, int]] = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]

# ======================================================================
# Validated alignment results
# ======================================================================
# Correlation scores measured at 1500 renderer hits.
# ``winner`` is max(cc_lidar_4dof, cc_renderer_2dof).
# ``config_suffix`` is the file suffix of the winning aligned config.

ALIGNMENT_RESULTS: Dict[str, dict] = {
    "seq_0_frame_135": {
        "frame": 135,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.476,
        "cc_renderer_2dof": 0.466,
        "config_suffix": "_aligned_gpu",
    },
    "seq_0_frame_390": {
        "frame": 390,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.363,
        "cc_renderer_2dof": 0.172,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_185": {
        "frame": 185,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.463,
        "cc_renderer_2dof": 0.283,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_438": {
        "frame": 438,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.584,
        "cc_renderer_2dof": 0.613,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_105": {
        "frame": 105,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.113,
        "cc_renderer_2dof": 0.244,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_160": {
        "frame": 160,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.393,
        "cc_renderer_2dof": 0.577,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_300": {
        "frame": 300,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.432,
        "cc_renderer_2dof": 0.389,
        "config_suffix": "_aligned_gpu",
    },
}


# ======================================================================
# Config lookup helpers
# ======================================================================

def get_aligned_config_path(
    scene_name: str,
    frame: int,
    data_root: str = "data",
) -> str:
    """Return the path to the winning aligned config for a scene.

    Args:
        scene_name: e.g. ``"seq_0_frame_135"``
        frame: radar frame number (e.g. 135)
        data_root: root of the preprocessed data tree

    Returns:
        Absolute path to e.g.
        ``data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json``
    """
    info = ALIGNMENT_RESULTS[scene_name]
    suffix = info["config_suffix"]
    return os.path.join(
        data_root, scene_name, "configs",
        f"cascaded_frame_{frame}{suffix}.json",
    )


def get_winner(scene_name: str) -> str:
    """Return ``"lidar_4dof"`` or ``"renderer_2dof"`` for a scene."""
    return ALIGNMENT_RESULTS[scene_name]["winner"]


def get_winner_cc(scene_name: str) -> float:
    """Return the winning correlation score for a scene."""
    info = ALIGNMENT_RESULTS[scene_name]
    return max(info["cc_lidar_4dof"], info["cc_renderer_2dof"])


def select_best_alignment(
    cc_lidar_4dof: float,
    cc_renderer_2dof: float,
) -> str:
    """Given two correlation scores, return the winning method name."""
    if cc_lidar_4dof >= cc_renderer_2dof:
        return "lidar_4dof"
    return "renderer_2dof"


# ======================================================================
# Summary / verification
# ======================================================================

def print_summary(data_root: str = "data") -> None:
    """Print a table of alignment results and verify config files exist."""
    print("=" * 78)
    print("Cascaded Radar Alignment — Per-Scene Results (1500-hit validation)")
    print("=" * 78)
    hdr = (
        f"{'Scene':<22} {'Winner':<16} {'CC_lidar':<10} "
        f"{'CC_renderer':<12} {'Config suffix'}"
    )
    print(hdr)
    print("-" * 78)

    all_ok = True
    winner_ccs = []
    for scene_name, frame in SCENES:
        info = ALIGNMENT_RESULTS[scene_name]
        cc_best = max(info["cc_lidar_4dof"], info["cc_renderer_2dof"])
        winner_ccs.append(cc_best)

        path = get_aligned_config_path(scene_name, frame, data_root)
        exists = os.path.isfile(path)
        status = "✓" if exists else "MISSING"
        if not exists:
            all_ok = False

        print(
            f"{scene_name:<22} {info['winner']:<16} "
            f"{info['cc_lidar_4dof']:<10.3f} "
            f"{info['cc_renderer_2dof']:<12.3f} "
            f"{info['config_suffix']}  [{status}]"
        )

    print("-" * 78)
    print(f"Mean winner CC: {sum(winner_ccs) / len(winner_ccs):.4f}")
    print(f"Config files: {'all present' if all_ok else 'SOME MISSING'}")
    print("=" * 78)


# ======================================================================
# Full pipeline
# ======================================================================
#
# The pass-1 ``run_all`` and ``run_single_frame`` entry points (originally
# here, ~700 LOC) drove both alignment methods through the Mitsuba MC
# renderer. In v5 those responsibilities live in
# ``mm25DGS_v5.preprocessing.alignment.pass2.run_pass2`` (CUDA backend +
# trajectory prior). The helpers above (``SCENES``, ``ALIGNMENT_RESULTS``,
# ``get_aligned_config_path``, ``select_best_alignment``,
# ``get_winner{,_cc}``, ``print_summary``) are preserved so callers can
# still look up pass-1 winners.

def run_all(*args, **kwargs):
    raise NotImplementedError(
        "Pass-1 MC orchestrator removed from the v5 vendored copy. "
        "Use `python -m mm25DGS_v5.preprocessing.alignment.pass2.run_pass2` "
        "for pass 2 (CUDA + trajectory prior). "
        "The original pass-1 code still lives in mmir/preprocessing/alignment/.")


def run_single_frame(*args, **kwargs):
    raise NotImplementedError(
        "Pass-1 MC single-frame orchestrator removed from the v5 vendored "
        "copy. Use `mm25DGS_v5.preprocessing.alignment.pass2` APIs instead.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Cascade alignment summary (v5 vendored copy, pass-1 helpers only)")
    parser.add_argument(
        "--data-root", default="data",
        help="Root of preprocessed data tree (default: data)")
    args = parser.parse_args()
    print_summary(args.data_root)

