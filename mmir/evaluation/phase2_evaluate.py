"""Phase 2: Run all evaluations on pre-generated (or live-generated) data."""

import time
from typing import Dict, List, Optional, Tuple

from .scene_registry import SceneInfo


def phase2_evaluate(
    scenes: List[SceneInfo],
    output_root: str,
    evaluations: Optional[List[str]] = None,
    skip_figures: bool = False,
    skip_latex: bool = False,
    batch_size: int = 5000,
    azimuth_range: Tuple[float, float] = (-21.0, 69.0),
    azimuth_step: float = 1.0,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Phase 2: Run evaluations and generate outputs.

    Args:
        scenes: List of scenes to evaluate.
        output_root: Root output directory.
        evaluations: Which evaluations to run.
            Options: "training_ra", "transfer", "occupancy", "reconstruction".
            Default: all four.
        skip_figures: Skip figure generation.
        skip_latex: Skip LaTeX table generation.
        verbose: Print progress.

    Returns:
        Dict mapping evaluation name → per-scene results.
    """
    if evaluations is None:
        evaluations = ["training_ra", "transfer", "occupancy", "reconstruction"]

    all_results = {}
    t_start = time.time()

    # --- Evaluation #1: Training RA ---
    if "training_ra" in evaluations:
        print("\n" + "=" * 60)
        print("Evaluation #1: Training RA Results")
        print("=" * 60)
        from .eval_training_ra_v2 import run_training_ra_all

        results = run_training_ra_all(
            scenes, output_root,
            skip_figures=skip_figures, skip_latex=skip_latex,
            verbose=verbose,
        )
        all_results["training_ra"] = results

    # --- Evaluation #2: Radar Transfer ---
    if "transfer" in evaluations:
        print("\n" + "=" * 60)
        print("Evaluation #2: Radar Transfer")
        print("=" * 60)
        from .eval_radar_transfer import run_radar_transfer_all

        results = run_radar_transfer_all(
            scenes, output_root, render=False,
            skip_figures=skip_figures, skip_latex=skip_latex,
            verbose=verbose,
        )
        all_results["radar_transfer"] = results

    # --- Evaluation #3: 3D Occupancy ---
    if "occupancy" in evaluations:
        print("\n" + "=" * 60)
        print("Evaluation #3: 3D RA Occupancy")
        print("=" * 60)
        from .eval_3d_occupancy import run_3d_occupancy_all

        results = run_3d_occupancy_all(
            scenes, output_root, render=False, batch_size=batch_size,
            skip_figures=skip_figures, skip_latex=skip_latex,
            verbose=verbose,
        )
        all_results["3d_occupancy"] = results

    # --- Evaluation #4: 3D Reconstruction ---
    if "reconstruction" in evaluations:
        print("\n" + "=" * 60)
        print("Evaluation #4: 3D Dense Reconstruction")
        print("=" * 60)
        from .eval_3d_reconstruction import run_3d_reconstruction_all

        results = run_3d_reconstruction_all(
            scenes, output_root, render=False,
            batch_size=batch_size,
            azimuth_range=azimuth_range,
            azimuth_step=azimuth_step,
            skip_figures=skip_figures, skip_latex=skip_latex,
            verbose=verbose,
        )
        all_results["3d_reconstruction"] = results

    total_time = time.time() - t_start
    print(f"\nPhase 2 complete: {total_time:.1f}s total")

    return all_results
