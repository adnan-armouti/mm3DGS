"""Chirp-loop NVS sweep orchestrator.

Launches ``train_chirp_loop_nvs.py`` once per (scene, frame, mode) triple
on a single GPU. The companion ``chirp_loop_report.py`` aggregates the
per-experiment ``results.json`` files into a markdown report.

Design:
  * One process per experiment (subprocess) — simplest correct resume:
    if ``results.json`` already exists for a (scene, frame, mode), the
    experiment is skipped.
  * Per-frame discovery: a frame ``F`` is runnable iff aligned configs
    for ``F-1`` *and* ``F+1`` exist (chirp-loop NVS anchors on both
    neighbours; edge frames of each scene's 9-frame window are dropped).
  * Stdout/stderr of each experiment → ``{log_dir}/{scene}_frame{F}_{mode}.log``
    to keep the driver's own log clean.

Usage:
    # All target scenes, upper_bound mode, GPU 0
    python -m mm25DGS_v5_v4.chirp_loop_sweep --mode upper_bound --gpu 0

    # All target scenes, held_out loop 8, GPU 1
    python -m mm25DGS_v5_v4.chirp_loop_sweep --mode held_out --gpu 1

    # Custom scene list
    python -m mm25DGS_v5_v4.chirp_loop_sweep --mode upper_bound --gpu 0 \
        --scenes seq_0_frame_135,seq_1_frame_438
"""

import os
import sys
import glob
import time
import json
import argparse
import subprocess


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

# Scenes prioritised by the user (2026-04-17). seq_0_frame_451 and
# seq_1_frame_277 are intentionally excluded.
DEFAULT_SCENES = [
    'seq_0_frame_135',
    'seq_0_frame_390',
    'seq_1_frame_185',
    'seq_1_frame_438',
    'seq_2_frame_105',
    'seq_2_frame_160',
    'seq_2_frame_300',
]

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v5_v4', 'output_chirp_loop')


def discover_runnable_frames(scene, data_root, use_pass2=True):
    """Return sorted list of cascade frame indices ``F`` such that
    aligned configs for ``F-1`` *and* ``F+1`` exist under
    ``{data_root}/alignment_data/{scene}/cascade/`` *and* the radar
    ADC for ``F`` is present. Frames without both neighbours are
    silently skipped (edges of the 9-frame scene window).
    """
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    radar_dir = os.path.join(data_root, scene, 'radar')
    suffix = '_aligned_pass2' if use_pass2 else '_aligned'

    existing = set()
    for p in glob.glob(os.path.join(align_dir, f'cascaded_frame_*{suffix}.json')):
        base = os.path.basename(p).replace(f'{suffix}.json', '')
        idx = base.replace('cascaded_frame_', '')
        try:
            existing.add(int(idx))
        except ValueError:
            pass

    runnable = []
    for F in sorted(existing):
        if (F - 1) in existing and (F + 1) in existing:
            if os.path.isfile(os.path.join(radar_dir, f'cascaded_frame_{F}.npy')):
                runnable.append(F)
    return runnable


def experiment_output_dir(scene, frame, mode, held_out_loop, use_pass2):
    """Match the output path convention used by train_chirp_loop_nvs.py."""
    tag = f'frame{frame}_{mode}'
    if mode == 'held_out':
        tag = f'frame{frame}_heldout_loop{held_out_loop}'
    if use_pass2:
        tag = f'{tag}_pass2'
    return os.path.join(OUTPUT_ROOT, f'{scene}_{tag}')


def run_experiment(scene, frame, mode, gpu, iters, held_out_loop,
                   use_pass2, log_dir, verbose=True):
    """Run a single train_chirp_loop_nvs.py invocation as a subprocess.

    Returns the path to the resulting ``results.json``, or ``None`` if
    the run failed or was unable to produce output.
    """
    out_dir = experiment_output_dir(scene, frame, mode, held_out_loop, use_pass2)
    results_path = os.path.join(out_dir, 'results.json')
    if os.path.isfile(results_path):
        if verbose:
            print(f'  [skip] {scene} f={frame} {mode}: {results_path} exists')
        return results_path

    cmd = [
        sys.executable, '-m', 'mm25DGS_v5_v4.train_chirp_loop_nvs',
        '--scene', scene, '--frame', str(frame),
        '--mode', mode, '--iters', str(iters),
    ]
    if mode == 'held_out':
        cmd += ['--held_out_loop', str(held_out_loop)]
    if use_pass2:
        cmd += ['--use_pass2_alignment']

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{scene}_frame{frame}_{mode}.log')

    t0 = time.time()
    if verbose:
        print(f'  [run ] {scene} f={frame} {mode}  '
              f'(gpu={gpu}, log={log_path})', flush=True)
    with open(log_path, 'w') as log_fh:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=log_fh, stderr=subprocess.STDOUT, check=False)
    dt = time.time() - t0

    if proc.returncode != 0:
        if verbose:
            print(f'  [FAIL] {scene} f={frame} {mode} '
                  f'(rc={proc.returncode}, {dt:.0f}s); see {log_path}')
        return None

    if not os.path.isfile(results_path):
        if verbose:
            print(f'  [WARN] {scene} f={frame} {mode}: exit 0 but no '
                  f'results.json at {results_path}')
        return None

    if verbose:
        try:
            r = json.load(open(results_path))
            train = r.get('final_train_mean', float('nan'))
            test = r.get('final_eval_mean', float('nan'))
            print(f'  [done] {scene} f={frame} {mode} '
                  f'train={train:.4f} test={test:.4f}  ({dt:.0f}s)')
        except Exception:
            print(f'  [done] {scene} f={frame} {mode}  ({dt:.0f}s)')
    return results_path


def main():
    ap = argparse.ArgumentParser(
        description='Chirp-loop NVS sweep over scenes × frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--mode', choices=['upper_bound', 'held_out'],
                    required=True,
                    help='Which training mode to sweep')
    ap.add_argument('--gpu', type=int, required=True,
                    help='CUDA_VISIBLE_DEVICES index to use for every run')
    ap.add_argument('--scenes', default=None,
                    help='Comma-separated scene list; default is the '
                         '7-scene priority set baked into this file')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--held-out-loop', type=int, default=8)
    ap.add_argument('--no-pass2', action='store_true',
                    help='Use pass-1 _aligned.json anchors instead of '
                         'pass-2 _aligned_pass2.json')
    ap.add_argument('--log-dir', default='/tmp/chirp_loop_sweep')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the experiment list and exit')
    args = ap.parse_args()

    scenes = (args.scenes.split(',') if args.scenes
              else list(DEFAULT_SCENES))
    use_pass2 = not args.no_pass2

    # Discover and print the full queue first
    queue = []
    for sc in scenes:
        frames = discover_runnable_frames(
            sc, args.data_root, use_pass2=use_pass2)
        for f in frames:
            queue.append((sc, f))
    if not queue:
        sys.exit('no runnable (scene, frame) pairs; check --data-root '
                 'and alignment artefacts')

    print('=' * 72)
    print(f'chirp-loop NVS sweep  mode={args.mode}  gpu={args.gpu}')
    print(f'  use_pass2={use_pass2}  iters={args.iters}  '
          f'held_out_loop={args.held_out_loop if args.mode == "held_out" else "-"}')
    print(f'  scenes={len(scenes)}  experiments={len(queue)}')
    print('=' * 72)
    for sc, f in queue:
        print(f'  {sc}  f={f}')
    if args.dry_run:
        return

    t_total = time.time()
    n_done = 0
    n_skip = 0
    n_fail = 0
    for sc, f in queue:
        r = run_experiment(
            scene=sc, frame=f, mode=args.mode, gpu=args.gpu,
            iters=args.iters, held_out_loop=args.held_out_loop,
            use_pass2=use_pass2, log_dir=args.log_dir, verbose=True,
        )
        if r is None:
            n_fail += 1
        else:
            # Skip = results existed before; done = freshly produced.
            # We only know 'skip' by checking existence before run_experiment,
            # but since run_experiment handles both and returns results_path,
            # we can't distinguish post-hoc. Treat all successes as 'done'.
            n_done += 1

    elapsed = time.time() - t_total
    print()
    print('=' * 72)
    print(f'sweep done  mode={args.mode}  gpu={args.gpu}')
    print(f'  done        : {n_done}')
    print(f'  failed      : {n_fail}')
    print(f'  total time  : {elapsed:.0f}s ({elapsed / 60:.1f} min)')


if __name__ == '__main__':
    main()
