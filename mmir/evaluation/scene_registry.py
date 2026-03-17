"""Scene registry: defines benchmark scenes, paths, and exclusions."""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Project root — all paths relative to this
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

EXCLUDED_SCENES = {"seq_1_frame_277", "seq_0_frame_451"}

# Training output directories
OUR_TRAINING_ROOT = os.path.join(OUTPUT_DIR, "train_v9")
BENCHMARK_TRAINING_ROOT = os.path.join(OUTPUT_DIR, "benchmark_v4")
BENCHMARK_SIONNA_ROOT = os.path.join(OUTPUT_DIR, "benchmark_sionna_ad")


@dataclass
class SceneInfo:
    """All paths and metadata for a single benchmark scene."""

    seq: int
    frame: int
    name: str  # e.g. "seq_1_frame_185"
    data_dir: str  # e.g. "<project>/data/seq_1_frame_185"

    # Configs
    cascaded_config: str  # aligned cascaded config JSON
    single_chip_configs: List[str] = field(default_factory=list)
    dense_config: Optional[str] = None

    # Scene geometry
    mesh_file: str = ""

    # Ground truth
    gt_cascaded_adc: str = ""
    gt_single_chip_adcs: Dict[int, str] = field(default_factory=dict)  # {frame_num: path}
    lidar_pcl: str = ""

    # Training outputs
    our_training_dir: str = ""
    benchmark_training_dir: str = ""
    benchmark_sionna_dir: str = ""

    def get_closest_single_chip_frame(self, target_frame: Optional[int] = None) -> Optional[int]:
        """Return the single-chip frame number closest to target (default: cascaded_frame * 2)."""
        if not self.gt_single_chip_adcs:
            return None
        if target_frame is None:
            target_frame = self.frame * 2
        return min(self.gt_single_chip_adcs.keys(), key=lambda f: abs(f - target_frame))

    def get_single_chip_config(self, sc_frame: int) -> Optional[str]:
        """Return single-chip config path for a given frame number.

        Preference order:
        1. _aligned.json — fully aligned (Procrustes + grid search)
        2. Original (unaligned) config — fallback
        """
        candidates = [cfg for cfg in self.single_chip_configs if f"frame_{sc_frame}" in cfg]
        if not candidates:
            return None
        aligned = [c for c in candidates if c.endswith("_aligned.json")]
        if aligned:
            return aligned[0]
        # Fallback: original (unaligned)
        original = [c for c in candidates if "_aligned" not in c]
        return original[0] if original else candidates[0]

    def get_single_chip_gt_adc(self, sc_frame: int) -> Optional[str]:
        """Return GT single-chip ADC path for a given frame number."""
        return self.gt_single_chip_adcs.get(sc_frame)


def _discover_scene(scene_dir: str) -> Optional[SceneInfo]:
    """Discover a scene from its data directory structure."""
    name = os.path.basename(scene_dir)
    match = re.match(r"seq_(\d+)_frame_(\d+)", name)
    if not match:
        return None

    seq, frame = int(match.group(1)), int(match.group(2))
    configs_dir = os.path.join(scene_dir, "configs")
    radar_dir = os.path.join(scene_dir, "radar")
    scene_subdir = os.path.join(scene_dir, "scene")

    # Find cascaded config — prefer the one used in training
    our_training_dir = os.path.join(OUR_TRAINING_ROOT, name)
    cascaded_config = ""
    if os.path.isfile(os.path.join(our_training_dir, "train_config.json")):
        with open(os.path.join(our_training_dir, "train_config.json")) as f:
            train_cfg = json.load(f)
        cascaded_config = train_cfg.get("config_file", "")
    if not cascaded_config or not os.path.isfile(cascaded_config):
        # Fallback: look for aligned_gpu variant
        candidates = sorted(glob.glob(os.path.join(configs_dir, f"cascaded_frame_{frame}_aligned_gpu.json")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(configs_dir, f"cascaded_frame_{frame}_aligned*.json")))
        if not candidates:
            candidates = sorted(glob.glob(os.path.join(configs_dir, f"cascaded_frame_{frame}.json")))
        cascaded_config = candidates[0] if candidates else ""

    # Find mesh — prefer the one used in training
    mesh_file = ""
    if os.path.isfile(os.path.join(our_training_dir, "train_config.json")):
        with open(os.path.join(our_training_dir, "train_config.json")) as f:
            train_cfg = json.load(f)
        mesh_file = train_cfg.get("scene_file", "")
    if not mesh_file or not os.path.isfile(mesh_file):
        mesh_file = os.path.join(scene_subdir, "mesh.ply")

    # Single-chip configs
    sc_configs = sorted(glob.glob(os.path.join(configs_dir, "single_chip_frame_*.json")))

    # Dense config
    dense_candidates = sorted(glob.glob(os.path.join(configs_dir, "dense_frame_*.json")))
    dense_config = dense_candidates[0] if dense_candidates else None

    # GT cascaded ADC
    gt_cascaded = os.path.join(radar_dir, f"cascaded_frame_{frame}.npy")
    if not os.path.isfile(gt_cascaded):
        gt_cascaded = ""

    # GT single-chip ADCs: extract frame numbers from filenames
    gt_sc_adcs = {}
    for f_path in sorted(glob.glob(os.path.join(radar_dir, "single_chip_frame_*.npy"))):
        m = re.search(r"single_chip_frame_(\d+)\.npy", f_path)
        if m:
            gt_sc_adcs[int(m.group(1))] = f_path

    # LiDAR point cloud
    lidar_pcl = os.path.join(scene_subdir, "pcl.npy")
    if not os.path.isfile(lidar_pcl):
        lidar_pcl = ""

    # Benchmark training dir (finite-difference)
    benchmark_dir = os.path.join(BENCHMARK_TRAINING_ROOT, name)
    if not os.path.isdir(benchmark_dir):
        benchmark_dir = ""

    # Benchmark Sionna AD training dir
    benchmark_sionna = os.path.join(BENCHMARK_SIONNA_ROOT, name)
    if not os.path.isdir(benchmark_sionna):
        benchmark_sionna = ""

    return SceneInfo(
        seq=seq,
        frame=frame,
        name=name,
        data_dir=scene_dir,
        cascaded_config=cascaded_config,
        single_chip_configs=sc_configs,
        dense_config=dense_config,
        mesh_file=mesh_file,
        gt_cascaded_adc=gt_cascaded,
        gt_single_chip_adcs=gt_sc_adcs,
        lidar_pcl=lidar_pcl,
        our_training_dir=our_training_dir if os.path.isdir(our_training_dir) else "",
        benchmark_training_dir=benchmark_dir,
        benchmark_sionna_dir=benchmark_sionna,
    )


def get_all_scenes() -> List[SceneInfo]:
    """Discover all scenes from the data directory (including excluded ones)."""
    scenes = []
    if not os.path.isdir(DATA_DIR):
        return scenes
    for entry in sorted(os.listdir(DATA_DIR)):
        scene_dir = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(scene_dir):
            continue
        scene = _discover_scene(scene_dir)
        if scene is not None:
            scenes.append(scene)
    return scenes


def get_benchmark_scenes() -> List[SceneInfo]:
    """Return all benchmark scenes excluding EXCLUDED_SCENES."""
    return [s for s in get_all_scenes() if s.name not in EXCLUDED_SCENES]


def get_scene(name: str) -> Optional[SceneInfo]:
    """Return a single scene by name, or None if not found."""
    scene_dir = os.path.join(DATA_DIR, name)
    if os.path.isdir(scene_dir):
        return _discover_scene(scene_dir)
    return None
