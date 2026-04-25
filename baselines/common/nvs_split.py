"""NVS split generator — the 8-train / 1-test middle-frame-bracketed split.

Enumerates the 9 on-disk frames per sensor, sorts by frame number, and picks
index 4 as test and indices {0..3, 5..8} as train. This matches every PLAN's
Section 5. One module, not three copies.

For each scene and each of {cascaded, single_chip} the returned dict holds:

    {
      "train_files":    [<8 paths to radar/<sensor>_frame_*.npy>],
      "train_configs":  [<8 paths to aligned per-frame config JSON>],
      "train_frames":   [<8 ints — frame numbers parsed from filenames>],
      "test_file":      "<path to middle cascaded_/single_chip_frame_*.npy>",
      "test_config":    "<path to middle aligned config JSON>",
      "test_frame":     <int>,
    }

Aligned per-frame configs live under ``data/alignment_data/<scene>/{cascade,single_chip}/``,
not under ``data/<scene>/configs/`` — the latter only holds the center-frame
aligned config.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List

from .scenes import (
    cascade_alignment_dir,
    scene_dir,
    single_chip_alignment_dir,
)


_CASC_RE = re.compile(r"cascaded_frame_(\d+)\.npy$")
_SC_RE = re.compile(r"single_chip_frame_(\d+)\.npy$")


def _frame_number(path: str, pattern: re.Pattern) -> int:
    m = pattern.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"could not parse frame number from {path}")
    return int(m.group(1))


def _pick_split(sorted_files: List[str], frame_pattern: re.Pattern,
                configs_for_frame) -> Dict[str, object]:
    if len(sorted_files) != 9:
        raise ValueError(
            f"expected exactly 9 frames per sensor, got {len(sorted_files)}: "
            f"{sorted_files}"
        )
    frames = [_frame_number(p, frame_pattern) for p in sorted_files]
    train_idx = [0, 1, 2, 3, 5, 6, 7, 8]
    test_idx = 4

    train_files = [sorted_files[i] for i in train_idx]
    train_frames = [frames[i] for i in train_idx]
    train_configs = [configs_for_frame(frames[i]) for i in train_idx]

    test_file = sorted_files[test_idx]
    test_frame = frames[test_idx]
    test_config = configs_for_frame(test_frame)

    missing = [p for p in train_configs + [test_config] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"aligned config missing for frames — cannot enumerate split: {missing}"
        )

    return {
        "train_files": train_files,
        "train_configs": train_configs,
        "train_frames": train_frames,
        "test_file": test_file,
        "test_config": test_config,
        "test_frame": test_frame,
    }


def cascaded_split(scene: str) -> Dict[str, object]:
    radar_glob = os.path.join(scene_dir(scene), "radar", "cascaded_frame_*.npy")
    files = sorted(glob.glob(radar_glob), key=lambda p: _frame_number(p, _CASC_RE))
    align_dir = cascade_alignment_dir(scene)

    def _cfg(frame: int) -> str:
        return os.path.join(align_dir, f"cascaded_frame_{frame}_aligned.json")

    return _pick_split(files, _CASC_RE, _cfg)


def single_chip_split(scene: str) -> Dict[str, object]:
    radar_glob = os.path.join(scene_dir(scene), "radar", "single_chip_frame_*.npy")
    files = sorted(glob.glob(radar_glob), key=lambda p: _frame_number(p, _SC_RE))
    align_dir = single_chip_alignment_dir(scene)

    def _cfg(frame: int) -> str:
        return os.path.join(align_dir, f"single_chip_frame_{frame}_aligned.json")

    return _pick_split(files, _SC_RE, _cfg)


def scene_split(scene: str) -> Dict[str, Dict[str, object]]:
    """Return {"cascaded": {...}, "single_chip": {...}} for one scene."""
    return {
        "cascaded": cascaded_split(scene),
        "single_chip": single_chip_split(scene),
    }
