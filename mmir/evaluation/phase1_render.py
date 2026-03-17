"""Phase 1: Generate all rendered data needed by evaluations #2, #3, #4.

Execution order:
1. Single-chip renders (fast: 3×4 = 12 virtual elements, ~seconds/scene)
2. Dense single-frame renders (medium: 100×100 = 10,000 elements, ~minutes/scene)
3. Dense spinning renders (slow: 90 frames × 10,000 elements, ~hours/scene)
"""

import time
from typing import List, Optional, Tuple

from .scene_registry import SceneInfo


def phase1_render(
    scenes: List[SceneInfo],
    output_root: str,
    evaluations: Optional[List[str]] = None,
    batch_size: int = 5000,
    azimuth_range: Tuple[float, float] = (-21.0, 69.0),
    azimuth_step: float = 1.0,
    verbose: bool = True,
):
    """Phase 1: Generate all rendered data.

    Args:
        scenes: List of scenes to process.
        output_root: Root output directory.
        evaluations: Which evaluations to render for.
            Options: "transfer", "occupancy", "reconstruction".
            Default: all three.
        batch_size: Ray batch size for dense rendering.
        azimuth_range: Azimuth sweep range for spinning.
        azimuth_step: Azimuth step for spinning.
        verbose: Print progress.
    """
    if evaluations is None:
        evaluations = ["transfer", "occupancy", "reconstruction"]

    t_start = time.time()

    # --- 1. Single-chip renders (radar transfer) ---
    if "transfer" in evaluations:
        print("\n" + "=" * 60)
        print("Phase 1a: Single-chip renders (radar transfer)")
        print("=" * 60)
        from .eval_radar_transfer import RadarTransferEvaluator

        for scene in scenes:
            t0 = time.time()
            evaluator = RadarTransferEvaluator(scene, output_root, verbose=verbose)
            evaluator.render()
            if verbose:
                print(f"  {scene.name}: {time.time() - t0:.1f}s")

    # --- 2. Dense single-frame renders (3D occupancy) ---
    if "occupancy" in evaluations:
        print("\n" + "=" * 60)
        print("Phase 1b: Dense single-frame renders (3D occupancy)")
        print("=" * 60)
        from .eval_3d_occupancy import OccupancyEvaluator

        for scene in scenes:
            if not scene.dense_config:
                if verbose:
                    print(f"  {scene.name}: no dense config, skipping")
                continue
            t0 = time.time()
            evaluator = OccupancyEvaluator(
                scene, output_root, batch_size=batch_size, verbose=verbose
            )
            evaluator.render()
            if verbose:
                print(f"  {scene.name}: {time.time() - t0:.1f}s")

    # --- 3. Dense spinning renders (3D reconstruction) ---
    if "reconstruction" in evaluations:
        print("\n" + "=" * 60)
        print("Phase 1c: Dense spinning renders (3D reconstruction)")
        print("=" * 60)
        from .eval_3d_reconstruction import ReconstructionEvaluator

        for scene in scenes:
            if not scene.dense_config:
                if verbose:
                    print(f"  {scene.name}: no dense config, skipping")
                continue
            t0 = time.time()
            evaluator = ReconstructionEvaluator(
                scene, output_root,
                azimuth_range=azimuth_range,
                azimuth_step=azimuth_step,
                batch_size=batch_size,
                verbose=verbose,
            )
            evaluator.render()
            if verbose:
                print(f"  {scene.name}: {time.time() - t0:.1f}s")

    total_time = time.time() - t_start
    print(f"\nPhase 1 complete: {total_time:.1f}s total")
