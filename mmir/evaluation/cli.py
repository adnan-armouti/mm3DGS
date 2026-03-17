"""Main CLI entry point for the unified post-processing pipeline.

Usage:
    python -m mmir.evaluation.cli <command> [options]

Commands:
    render          Phase 1: generate all rendered data
    evaluate        Phase 2: run evaluations on rendered data
    all             Run both phases sequentially
    training-ra     Evaluation #1 only (no rendering needed)
    radar-transfer  Evaluation #2 (render + evaluate)
    occupancy       Evaluation #3 (render + evaluate)
    reconstruction  Evaluation #4 (render + evaluate)
"""

import argparse
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="mmIR Unified Post-Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments (added to all subparsers)
    def add_common_args(sub):
        sub.add_argument(
            "--scenes", type=str, default=None,
            help="Comma-separated scene names (default: all benchmark scenes)",
        )
        sub.add_argument(
            "--output-root", type=str, default=None,
            help="Output directory (default: output/postprocess_final)",
        )
        sub.add_argument(
            "--batch-size", type=int, default=5000,
            help="Ray batch size for dense rendering (default: 5000)",
        )
        sub.add_argument("--verbose", action="store_true", default=True)
        sub.add_argument("--quiet", action="store_true", default=False)

    # Rendering-specific arguments
    def add_render_args(sub):
        sub.add_argument("--skip-transfer", action="store_true", help="Skip single-chip render")
        sub.add_argument("--skip-occupancy", action="store_true", help="Skip dense single-frame render")
        sub.add_argument("--skip-reconstruction", action="store_true", help="Skip spinning render")

    # Spinning-specific arguments
    def add_spinning_args(sub):
        sub.add_argument("--az-start", type=float, default=-21.0, help="Azimuth start (degrees)")
        sub.add_argument("--az-end", type=float, default=69.0, help="Azimuth end (degrees)")
        sub.add_argument("--az-step", type=float, default=1.0, help="Azimuth step (degrees)")

    # Visualization / 3D eval arguments
    def add_3d_args(sub):
        sub.add_argument(
            "--coloradar-dataset-dir", type=str,
            default=None,
            help="Base path to ColoRadar kitti sequence directory. "
                 "Seq index is appended (e.g. dir + '1'). "
                 "Matches preproc.py --dataset-dir convention.",
        )
        sub.add_argument(
            "--coloradar-calib-dir", type=str,
            default=None,
            help="Path to ColoRadar calibration directory (contains cascade/).",
        )
        sub.add_argument(
            "--viz-radar-percentile", type=float, default=99.9,
            help="Radar intensity percentile threshold (0-100) for dense radar visualization. "
                 "Points with RAE magnitude below this percentile are removed. "
                 "Default: 99.9 (keep top 0.1%%)",
        )

    # Evaluation-specific arguments
    def add_eval_args(sub):
        sub.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
        sub.add_argument("--skip-latex", action="store_true", help="Skip LaTeX table generation")

    # --- render ---
    sub_render = subparsers.add_parser("render", help="Phase 1: generate rendered data")
    add_common_args(sub_render)
    add_render_args(sub_render)
    add_spinning_args(sub_render)

    # --- evaluate ---
    sub_eval = subparsers.add_parser("evaluate", help="Phase 2: run evaluations")
    add_common_args(sub_eval)
    add_eval_args(sub_eval)
    add_spinning_args(sub_eval)

    # --- all ---
    sub_all = subparsers.add_parser("all", help="Run both phases")
    add_common_args(sub_all)
    add_render_args(sub_all)
    add_eval_args(sub_all)
    add_spinning_args(sub_all)
    add_3d_args(sub_all)

    # --- training-ra ---
    sub_tra = subparsers.add_parser("training-ra", help="Evaluation #1: training RA results")
    add_common_args(sub_tra)
    add_eval_args(sub_tra)

    # --- radar-transfer ---
    sub_rt = subparsers.add_parser("radar-transfer", help="Evaluation #2: radar transfer")
    add_common_args(sub_rt)
    add_eval_args(sub_rt)

    # --- occupancy ---
    sub_occ = subparsers.add_parser("occupancy", help="Evaluation #3: 3D occupancy")
    add_common_args(sub_occ)
    add_eval_args(sub_occ)
    add_3d_args(sub_occ)

    # --- reconstruction ---
    sub_rec = subparsers.add_parser("reconstruction", help="Evaluation #4: 3D reconstruction")
    add_common_args(sub_rec)
    add_eval_args(sub_rec)
    add_spinning_args(sub_rec)
    sub_rec.add_argument(
        "--viz-radar-percentile", type=float, default=99.9,
        help="Radar intensity percentile threshold for visualization (default: 99.9)",
    )
    sub_rec.add_argument(
        "--radar-percentile", type=float, default=98.0,
        help="Post-aggregation composite-score percentile (0-100). "
             "Keeps top (100-p)%% of voxels by view_div*max_intensity. (default: 99.5)",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    return args


def resolve_scenes(args):
    """Resolve scene list from args."""
    from .scene_registry import get_benchmark_scenes, get_scene

    if args.scenes:
        scene_names = [s.strip() for s in args.scenes.split(",")]
        scenes = []
        for name in scene_names:
            scene = get_scene(name)
            if scene is None:
                print(f"Warning: scene '{name}' not found, skipping")
            else:
                scenes.append(scene)
        return scenes
    else:
        return get_benchmark_scenes()


def resolve_output_root(args):
    """Resolve output root directory."""
    if args.output_root:
        return args.output_root
    from .scene_registry import PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "output", "postprocess_final")


def main():
    args = parse_args()
    verbose = not args.quiet

    scenes = resolve_scenes(args)
    output_root = resolve_output_root(args)
    os.makedirs(output_root, exist_ok=True)

    if verbose:
        print(f"mmIR Post-Processing Pipeline")
        print(f"  Command: {args.command}")
        print(f"  Scenes: {[s.name for s in scenes]}")
        print(f"  Output: {output_root}")
        print()

    t_start = time.time()

    # Ensure Mitsuba variant is set before any imports that use mi.Vector3f
    # at class definition time.
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")

    if args.command == "render":
        from .phase1_render import phase1_render

        evals = []
        if not getattr(args, "skip_transfer", False):
            evals.append("transfer")
        if not getattr(args, "skip_occupancy", False):
            evals.append("occupancy")
        if not getattr(args, "skip_reconstruction", False):
            evals.append("reconstruction")

        phase1_render(
            scenes, output_root,
            evaluations=evals,
            batch_size=args.batch_size,
            azimuth_range=(args.az_start, args.az_end),
            azimuth_step=args.az_step,
            verbose=verbose,
        )

    elif args.command == "evaluate":
        from .phase2_evaluate import phase2_evaluate

        phase2_evaluate(
            scenes, output_root,
            skip_figures=getattr(args, "skip_figures", False),
            skip_latex=getattr(args, "skip_latex", False),
            batch_size=args.batch_size,
            azimuth_range=(args.az_start, args.az_end),
            azimuth_step=args.az_step,
            verbose=verbose,
        )

    elif args.command == "all":
        from .phase1_render import phase1_render
        from .phase2_evaluate import phase2_evaluate

        evals = []
        if not getattr(args, "skip_transfer", False):
            evals.append("transfer")
        if not getattr(args, "skip_occupancy", False):
            evals.append("occupancy")
        if not getattr(args, "skip_reconstruction", False):
            evals.append("reconstruction")

        phase1_render(
            scenes, output_root,
            evaluations=evals,
            batch_size=args.batch_size,
            azimuth_range=(args.az_start, args.az_end),
            azimuth_step=args.az_step,
            verbose=verbose,
        )

        eval_list = ["training_ra"] + evals
        phase2_evaluate(
            scenes, output_root,
            evaluations=eval_list,
            skip_figures=getattr(args, "skip_figures", False),
            skip_latex=getattr(args, "skip_latex", False),
            batch_size=args.batch_size,
            azimuth_range=(args.az_start, args.az_end),
            azimuth_step=args.az_step,
            verbose=verbose,
        )

    elif args.command == "training-ra":
        from .eval_training_ra_v2 import run_training_ra_all
        run_training_ra_all(
            scenes, output_root,
            skip_figures=getattr(args, "skip_figures", False),
            skip_latex=getattr(args, "skip_latex", False),
            verbose=verbose,
        )

    elif args.command == "radar-transfer":
        from .eval_radar_transfer import run_radar_transfer_all
        run_radar_transfer_all(
            scenes, output_root, render=True,
            skip_figures=getattr(args, "skip_figures", False),
            skip_latex=getattr(args, "skip_latex", False),
            verbose=verbose,
        )

    elif args.command == "occupancy":
        from .eval_3d_occupancy import run_3d_occupancy_all
        run_3d_occupancy_all(
            scenes, output_root, render=True,
            batch_size=args.batch_size,
            coloradar_dataset_dir=getattr(args, "coloradar_dataset_dir",
                None),
            coloradar_calib_dir=getattr(args, "coloradar_calib_dir",
                None),
            viz_radar_percentile=getattr(args, "viz_radar_percentile", 99.9),
            skip_figures=getattr(args, "skip_figures", False),
            skip_latex=getattr(args, "skip_latex", False),
            verbose=verbose,
        )

    elif args.command == "reconstruction":
        from .eval_3d_reconstruction import run_3d_reconstruction_all
        run_3d_reconstruction_all(
            scenes, output_root, render=True,
            batch_size=args.batch_size,
            azimuth_range=(args.az_start, args.az_end),
            azimuth_step=args.az_step,
            radar_percentile=getattr(args, "radar_percentile", 98.0),
            aggregate_percentile=getattr(args, "aggregate_percentile", 90.0),
            skip_figures=getattr(args, "skip_figures", False),
            skip_latex=getattr(args, "skip_latex", False),
            verbose=verbose,
        )

    total = time.time() - t_start
    print(f"\nTotal time: {total:.1f}s")


if __name__ == "__main__":
    main()
