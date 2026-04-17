#!/usr/bin/env python3
"""End-to-end cascade alignment pipeline (pass-1 + pass-2).

Single convenience entry point that runs both alignment passes per scene,
producing the ``cascaded_frame_<F>_aligned.json`` (pass 1) and
``cascaded_frame_<F>_aligned_pass2.json`` (pass 2) artefacts the rest of
the repo (training, evaluation, chirp-loop NVS) consumes.

Per scene the driver runs:

  1. **Pass 1** — per-frame independent alignment (LiDAR-voxel 4-DOF +
     Mitsuba MC renderer 2-DOF, winner selected by cart_corr):
     ``mmir.preprocessing.alignment.sc_trajectory_transfer.align_all_cascade_frames``
     Outputs: ``data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned*.json``
     + per-frame alignment logs.

  2. **Pass 2** — trajectory-aware re-alignment of pass-1 outliers
     (LOWESS smooth + MAD outlier flag + v5-CUDA renderer-4-DOF *and*
     LiDAR-4-DOF both re-optimised with a soft pose-prior penalty, plus
     a trajectory-consistency gate):
     ``mm25DGS_v5.preprocessing.alignment.pass2.run_pass2.run_pass2_for_scene``
     Outputs: ``cascaded_frame_<F>_aligned_pass2.json`` +
     ``..._alignment_log_pass2.json`` + ``pass2_triage.json`` +
     ``pass2_summary.json``.

Prerequisites
-------------
* Preprocessing done — each ``data/<scene>/`` has ``scene/mesh.ply``,
  ``scene/pcl.npy``, ``radar/cascaded_frame_*.npy``, and
  ``configs/cascaded_frame_*.json``.
* v5 CUDA extension built (for pass 2)::

      cd mm25DGS_v5/cuda && python setup.py build_ext --inplace

* Conda env ``mmir`` active (mitsuba + drjit + open3d + cupy + torch).

Usage
-----
    # Both passes on every scene under data/
    python -m mm25DGS_v5.preprocessing.alignment.run_alignment --all

    # Single scene
    python -m mm25DGS_v5.preprocessing.alignment.run_alignment \
        --scene seq_0_frame_135

    # Only pass 2 (pass-1 configs already on disk)
    python -m mm25DGS_v5.preprocessing.alignment.run_alignment \
        --all --skip-pass-1

    # Only pass 1 (skip the trajectory-aware refinement)
    python -m mm25DGS_v5.preprocessing.alignment.run_alignment \
        --all --skip-pass-2

Timing (1× RTX 4090):
    pass 1: ~5–15 min / scene (Mitsuba MC is the bottleneck; dominated by
            per-frame renderer 2-DOF grid search and LiDAR 4-DOF optim.)
    pass 2: ~2–3 min / scene (v5 CUDA grid + Nelder-Mead is fast)
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _discover_scenes(data_root):
    """Return sorted scene names (``seq_*``) found under ``data_root``."""
    if not os.path.isdir(data_root):
        return []
    return sorted(
        d for d in os.listdir(data_root)
        if d.startswith('seq_')
        and os.path.isdir(os.path.join(data_root, d, 'radar'))
    )


def run_pass1(scene, data_root, output_root, verbose=True):
    """Pass-1 alignment for every cascade frame in a scene. Produces
    ``cascaded_frame_<F>_aligned*.json`` + per-frame alignment logs under
    ``{output_root}/{scene}/cascade/``.
    """
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant('cuda_ad_rgb')
    from mmir.preprocessing.alignment.sc_trajectory_transfer import (
        align_all_cascade_frames,
    )
    return align_all_cascade_frames(
        scene_name=scene,
        data_root=data_root,
        output_root=output_root,
        verbose=verbose,
    )


def run_pass2(scene, data_root, prior_weight, gate_mad, target_n, verbose=True):
    """Pass-2 trajectory-aware re-alignment. Produces
    ``cascaded_frame_<F>_aligned_pass2.json`` + logs + ``pass2_summary.json``
    under ``{data_root}/alignment_data/{scene}/cascade/``.
    """
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant('cuda_ad_rgb')
    from mm25DGS_v5.preprocessing.alignment.pass2.run_pass2 import (
        run_pass2_for_scene,
    )
    return run_pass2_for_scene(
        scene,
        data_root=data_root,
        prior_weight=prior_weight,
        gate_mad_threshold=gate_mad,
        target_n=target_n,
        refit_stage_a=True,
        also_realign_soft=True,
        verbose=verbose,
    )


def main():
    ap = argparse.ArgumentParser(
        description='mm3DGS cascade alignment (pass-1 + pass-2) end-to-end',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--scene', default=None,
                    help='Run a single scene (e.g. seq_0_frame_135)')
    ap.add_argument('--all', action='store_true',
                    help='Run every scene found under --data-root')
    ap.add_argument('--data-root', default='data',
                    help='Root of preprocessed data (contains seq_* dirs)')
    ap.add_argument('--pass1-output-root', default='data/alignment_data',
                    help='Where pass-1 writes aligned configs + logs. The '
                         'tree layout is {root}/{scene}/cascade/...')
    ap.add_argument('--skip-pass-1', action='store_true',
                    help='Only run pass 2. Pass-1 configs must already be on disk.')
    ap.add_argument('--skip-pass-2', action='store_true',
                    help='Only run pass 1.')
    # Pass-2 knobs (match run_pass2 defaults)
    ap.add_argument('--prior-weight', type=float, default=0.01,
                    help='Pass-2 prior penalty weight λ in cc - λ·||δ||²')
    ap.add_argument('--gate-mad', type=float, default=2.5,
                    help='Pass-2 trajectory-consistency gate (MAD units)')
    ap.add_argument('--target-n', type=int, default=30000,
                    help='Pass-2 FPS target point count for CUDA alignment ctx')
    ap.add_argument('--skip-on-error', action='store_true',
                    help='Continue to the next scene on per-scene failure '
                         'instead of aborting')
    args = ap.parse_args()

    if args.skip_pass_1 and args.skip_pass_2:
        ap.error('--skip-pass-1 and --skip-pass-2 together do nothing')

    if args.all:
        scenes = _discover_scenes(args.data_root)
        if not scenes:
            sys.exit(f'no seq_* scenes under {args.data_root!r}')
    elif args.scene:
        scenes = [args.scene]
    else:
        ap.error('specify --scene <name> or --all')

    print('=' * 72)
    print(f'mm3DGS alignment: {len(scenes)} scene(s)')
    print(f'  data_root     = {args.data_root}')
    print(f'  pass1_output  = {args.pass1_output_root}')
    print(f'  pass1 enabled = {not args.skip_pass_1}')
    print(f'  pass2 enabled = {not args.skip_pass_2}')
    if not args.skip_pass_2:
        print(f'  pass2 knobs   = prior_weight={args.prior_weight}, '
              f'gate={args.gate_mad} MAD, target_n={args.target_n}')
    print('=' * 72)

    t_total = time.time()
    n_ok, n_fail = 0, 0

    for sc in scenes:
        t_scene = time.time()
        print(f'\n{"#" * 72}\n# {sc}\n{"#" * 72}')
        try:
            if not args.skip_pass_1:
                print(f'\n── {sc} · pass 1 ──')
                run_pass1(sc, data_root=args.data_root,
                          output_root=args.pass1_output_root,
                          verbose=True)
            else:
                print(f'── {sc} · pass 1 skipped ──')

            if not args.skip_pass_2:
                print(f'\n── {sc} · pass 2 ──')
                run_pass2(sc, data_root=args.data_root,
                          prior_weight=args.prior_weight,
                          gate_mad=args.gate_mad,
                          target_n=args.target_n,
                          verbose=True)
            else:
                print(f'── {sc} · pass 2 skipped ──')

            n_ok += 1
            print(f'\n[{sc}] done in {time.time() - t_scene:.1f}s')
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f'\n[{sc}] FAILED after {time.time() - t_scene:.1f}s: {e}')
            if not args.skip_on_error:
                raise

    elapsed = time.time() - t_total
    print()
    print('=' * 72)
    print(f'run_alignment summary')
    print('=' * 72)
    print(f'  scenes ok      : {n_ok}')
    print(f'  scenes failed  : {n_fail}')
    print(f'  total elapsed  : {elapsed:.1f}s ({elapsed / 60:.1f} min)')


if __name__ == '__main__':
    main()
