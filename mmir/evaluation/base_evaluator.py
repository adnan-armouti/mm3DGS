"""Abstract base class for all evaluation types."""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from .scene_registry import SceneInfo


class BaseEvaluator(ABC):
    """Base class for all evaluation types."""

    def __init__(self, scene: SceneInfo, output_root: str, verbose: bool = True):
        self.scene = scene
        self.output_dir = os.path.join(output_root, self.eval_name, scene.name)
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    @abstractmethod
    def eval_name(self) -> str:
        """Evaluation identifier, e.g. 'training_ra', 'radar_transfer'."""

    @abstractmethod
    def run(self) -> dict:
        """Run evaluation, return metrics dict."""

    @abstractmethod
    def generate_figures(self, metrics: dict) -> List[str]:
        """Generate figures, return list of saved file paths."""

    def generate_latex_row(self, metrics: dict) -> str:
        """Generate a LaTeX table row for this scene's results."""
        return ""

    def save_metrics(self, metrics: dict, filename: str = "metrics.json"):
        """Save metrics dict to JSON in output_dir."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        if self.verbose:
            print(f"  Saved metrics → {path}")
        return path

    def log(self, msg: str):
        if self.verbose:
            print(f"[{self.eval_name}/{self.scene.name}] {msg}")

    @staticmethod
    def _clip_to_mesh_bbox(pts: np.ndarray, mesh_path: str, margin: float = 0.0) -> np.ndarray:
        """Clip points to the axis-aligned bounding box of the mesh (+ margin)."""
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        bbox = mesh.get_axis_aligned_bounding_box()
        bb_min = np.asarray(bbox.get_min_bound()) - margin
        bb_max = np.asarray(bbox.get_max_bound()) + margin
        mask = np.all((pts >= bb_min) & (pts <= bb_max), axis=1)
        return pts[mask]


def aggregate_metrics(
    per_scene_metrics: Dict[str, dict],
    keys: List[str],
) -> dict:
    """Compute mean ± std across scenes for specified metric keys."""
    import numpy as np

    agg = {}
    for key in keys:
        vals = [m[key] for m in per_scene_metrics.values() if key in m and m[key] is not None]
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n": len(vals),
            }
    return agg


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
