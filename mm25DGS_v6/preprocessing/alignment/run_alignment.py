#!/usr/bin/env python3
"""End-to-end cascade alignment pipeline (pass-1 + pass-2 [+ optional pass-3]).

Single convenience entry point that runs both per-frame alignment passes
per scene — plus, when explicitly enabled via ``--run-pass-3``, the
per-chirp Stage 3 refinement on top — producing the
``cascaded_frame_<F>_aligned.json`` (pass 1),
``cascaded_frame_<F>_aligned_pass2.json`` (pass 2), and (optionally)
``per_chirp/cascaded_frame_<F>_chirp<CC>_aligned_pass3.json`` (pass 3)
artefacts the rest of the repo (training, evaluation, chirp-loop NVS,
frame NVS) consumes.

Per scene the driver runs:

  1. **Pass 1** — per-frame independent alignment (v5-CUDA renderer-2-DOF
     + cupy LiDAR-4-DOF, winner selected by cart_corr against the same
     CUDA renderer objective):
     ``mm25DGS_v5.preprocessing.alignment.cascaded_alignment.run_all``
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

  3. **Pass 3** (*optional*, off by default) — per-chirp anchored
     refinement on top of pass 2. Produces one aligned config per
     (frame, chirp) pair under ``per_chirp/``. Enable with
     ``--run-pass-3``. The chirp-loop NVS and frame-NVS trainers only
     need pass-3 configs for specific all-chirp training variants; see
     ``md/per_chirp_alignment_stage3_plan.md`` and the ``pass2 vs pass3``
     A/B in ``md/frame_nvs.md`` for when this is worth the extra
     compute. Default is **pass 1 + pass 2 only**.

All passes use the v5 CUDA rendering backend (no Mitsuba MC).

Prerequisites
-------------
* Preprocessing done — each ``data/<scene>/`` has ``scene/mesh.ply``,
  ``scene/pcl.npy``, ``radar/cascaded_frame_*.npy``, and
  ``configs/cascaded_frame_*.json``.
* v5 CUDA extension built (required for both passes)::

      cd mm25DGS_v5/cuda && python setup.py build_ext --inplace

* Conda env ``mmir`` active (mitsuba + drjit + open3d + cupy + torch).

Usage
-----
    # Default: pass 1 + pass 2 on every scene under data/
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

    # Include the optional Stage 3 per-chirp refinement (OFF by default)
    python -m mm25DGS_v5.preprocessing.alignment.run_alignment \
        --all --run-pass-3

    # Only Stage 3 (pass-1 + pass-2 configs already on disk)
    python -m mm25DGS_v5.preprocessing.alignment.run_alignment \
        --all --skip-pass-1 --skip-pass-2 --run-pass-3

Timing (1× RTX 4090):
    pass 1: ~1–2 min / scene (CUDA renderer-2-DOF + cupy LiDAR-4-DOF)
    pass 2: ~2–3 min / scene (CUDA renderer-4-DOF + cupy LiDAR-4-DOF
            with trajectory prior)
    pass 3: ~45–70 min / scene (per-chirp anchored refinement — see
            ``md/per_chirp_alignment_stage3_speedup_plan.md`` for the
            planned reductions)
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


def run_pass1(scene, data_root, output_root, target_n=30000,
              skip_if_exists=True, verbose=True):
    """Pass-1 alignment for every cascade frame in a scene (CUDA backend).
    Produces ``cascaded_frame_<F>_aligned*.json`` + per-frame alignment
    logs under ``{output_root}/{scene}/cascade/``.
    """
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant('cuda_ad_rgb')
    from mm25DGS_v5.preprocessing.alignment.cascaded_alignment import run_all
    return run_all(
        data_root=data_root,
        output_root=output_root,
        scenes=[scene],
        skip_if_exists=skip_if_exists,
        target_n=target_n,
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


def run_pass3(scene, data_root, anchor_source, prior_weight, gate_margin,
              target_n, verbose=True):
    """Pass-3 per-chirp anchored refinement on top of pass 2. Produces
    ``per_chirp/cascaded_frame_<F>_chirp<CC>_aligned_pass3.json`` +
    per-chirp alignment logs + ``pass3_summary.json`` under
    ``{data_root}/alignment_data/{scene}/cascade/``.

    Requires pass-2 configs to exist on disk (Stage 3 anchors on
    ``cascaded_frame_<F>_aligned_pass2.json``).
    """
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant('cuda_ad_rgb')
    from mm25DGS_v5.preprocessing.alignment.per_chirp_alignment import (
        run_stage3_for_scene,
    )
    return run_stage3_for_scene(
        scene,
        data_root=data_root,
        anchor_source=anchor_source,
        target_n=target_n,
        prior_weight=prior_weight,
        gate_margin=gate_margin,
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
                    help='Do not run pass 1. Pass-1 configs must already be on disk.')
    ap.add_argument('--skip-pass-2', action='store_true',
                    help='Do not run pass 2.')
    ap.add_argument('--run-pass-3', action='store_true',
                    help='Additionally run pass 3 (per-chirp Stage 3 '
                         'refinement). OFF by default — pass-2/Stage-2 '
                         'configs are the recommended default for '
                         'downstream training (see '
                         'md/per_chirp_alignment_stage3_plan.md).')
    # Pass-2 knobs (match run_pass2 defaults)
    ap.add_argument('--prior-weight', type=float, default=0.01,
                    help='Pass-2 prior penalty weight λ in cc - λ·||δ||²')
    ap.add_argument('--gate-mad', type=float, default=2.5,
                    help='Pass-2 trajectory-consistency gate (MAD units)')
    # Pass-3 knobs (only used when --run-pass-3)
    ap.add_argument('--pass3-anchor-source', default='hybrid',
                    choices=['lerp', 'gt', 'hybrid'],
                    help='Pass-3 anchor source. hybrid (recommended): '
                         'pass-2_F absolute + GT relative per-chirp motion.')
    ap.add_argument('--pass3-prior-weight', type=float, default=0.05,
                    help='Pass-3 prior penalty weight λ around the anchor')
    ap.add_argument('--pass3-gate-margin', type=float, default=0.005,
                    help='Pass-3 refinement must beat anchor cc by ≥ this margin')
    ap.add_argument('--target-n', type=int, default=30000,
                    help='FPS target point count for CUDA alignment ctx '
                         '(used by all passes)')
    ap.add_argument('--force', action='store_true',
                    help='Pass-1: re-run alignment even if existing outputs '
                         'are on disk (otherwise skip_if_exists=True)')
    ap.add_argument('--skip-on-error', action='store_true',
                    help='Continue to the next scene on per-scene failure '
                         'instead of aborting')
    args = ap.parse_args()

    if args.skip_pass_1 and args.skip_pass_2 and not args.run_pass_3:
        ap.error('--skip-pass-1 and --skip-pass-2 together do nothing '
                 '(pass --run-pass-3 to run only pass 3)')

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
    print(f'  pass3 enabled = {args.run_pass_3}  (default OFF — '
          f'Stage-2 is the recommended training anchor)')
    if not args.skip_pass_2:
        print(f'  pass2 knobs   = prior_weight={args.prior_weight}, '
              f'gate={args.gate_mad} MAD, target_n={args.target_n}')
    if args.run_pass_3:
        print(f'  pass3 knobs   = anchor_source={args.pass3_anchor_source}, '
              f'prior_weight={args.pass3_prior_weight}, '
              f'gate_margin={args.pass3_gate_margin} cc')
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
                          target_n=args.target_n,
                          skip_if_exists=not args.force,
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

            if args.run_pass_3:
                print(f'\n── {sc} · pass 3 ──')
                run_pass3(sc, data_root=args.data_root,
                          anchor_source=args.pass3_anchor_source,
                          prior_weight=args.pass3_prior_weight,
                          gate_margin=args.pass3_gate_margin,
                          target_n=args.target_n,
                          verbose=True)
            else:
                print(f'── {sc} · pass 3 skipped (default; '
                      f'pass --run-pass-3 to enable) ──')

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
