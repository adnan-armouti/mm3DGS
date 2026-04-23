"""§2.5 — ablation matrix driver.

Runs a set of single-axis variants on one scene (pilot) and tabulates
four metrics per row: |RA| train, |RA| test, |RAD| train, |RAD| test.

Priority rows (the ones we believe will move the needle based on §2.0
and §2.1 findings):

  B   baseline                                — post-C1, gt_mean norm, doppler on
  M03 multi-task λ = 0.3                       — C2b toward |RA|
  M10 multi-task λ = 1.0                       — stronger C2b bias
  V0  v_ego = 0 (zero out doppler phase)       — measures Doppler's net contribution
  NO_DOP disable doppler (v5-style)            — the v5 fallback at 16 loops
  NORM_MAX   (legacy gt_max norm)              — sanity re-run of pre-C1

Wall-clock: 6 rows × 500 iter × ~600 ms ≈ 30 min per scene serial,
~15 min with 2-GPU parallelism.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

REPO_ROOT = '/home/adnan/Desktop/mm3DGS'
OUT_DIR   = '/home/adnan/Desktop/mm3DGS/md/diagnostics'
PY        = '/home/adnan/.conda/envs/mmir/bin/python'
TRAIN_MOD = 'mm25DGS_v7.train_frame_nvs'

BASE_FLAGS = dict(
    iters=500,
    target_n=20000,
    loss_type='mse_raw',
)


def _build_rows_for(scene, F):
    train_frames = ','.join(str(F+k) for k in [-4,-3,-2,-1,1,2,3,4])
    train_loops  = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
    common = [
        '--scene', scene, '--test_frame', str(F),
        '--train_frames', train_frames,
        '--train_loops', train_loops,
        '--iters', str(BASE_FLAGS['iters']),
        '--loss_type', BASE_FLAGS['loss_type'],
        '--target_n', str(BASE_FLAGS['target_n']),
    ]

    rows = [
        ('B_baseline',       common + ['--doppler', '--loss_norm', 'mean']),
        ('NORM_MAX_legacy',  common + ['--doppler', '--loss_norm', 'max']),
        ('M03_multitask',    common + ['--doppler', '--loss_norm', 'mean',
                                         '--loss_multitask_lambda', '0.3']),
        ('M10_multitask',    common + ['--doppler', '--loss_norm', 'mean',
                                         '--loss_multitask_lambda', '1.0']),
        ('M30_multitask',    common + ['--doppler', '--loss_norm', 'mean',
                                         '--loss_multitask_lambda', '3.0']),
        ('NO_DOP_v5style',   common),     # no --doppler → v5 path
        # Note: train_loops=16 without --doppler runs 8 frames × 16 loops
        # on the v5 path (if supported) else is coerced. Output dir tag
        # will differ for this row.
    ]
    return rows


def _run_one(scene, F, tag, cli_args, gpu, mode_tag='norm_mean'):
    """Invoke train_frame_nvs in a subprocess. Returns results.json content
    (or None on failure)."""
    log_dir = os.path.join(REPO_ROOT, 'logs_v7', 'ablation_matrix',
                            f'{scene}_F{F}')
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f'{tag}.log')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print(f'[gpu {gpu}] {scene} F={F} {tag}  — launching  log={log}')
    t0 = time.time()
    with open(log, 'w') as f:
        proc = subprocess.run([PY, '-m', TRAIN_MOD] + cli_args,
                                stdout=f, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT, env=env)
    dt = time.time() - t0
    print(f'[gpu {gpu}] {scene} F={F} {tag}  — done in {dt:.0f}s  '
          f'(exit={proc.returncode})')

    # Find the output dir — reconstruct the tag
    # Reach into the log to find the actual printed output dir path.
    with open(log, 'r') as f:
        log_text = f.read()
    out_dir = None
    for line in log_text.splitlines():
        if 'results saved to' in line:
            out_dir = line.split('results saved to:')[-1].strip()
            # Strip ANSI color codes
            out_dir = out_dir.split('\x1b')[0].strip().rstrip('[0m').strip()
            break
    if out_dir is None or not os.path.isdir(out_dir):
        print(f'  !! could not locate output dir from log')
        return None
    res_path = os.path.join(out_dir, 'results.json')
    if not os.path.isfile(res_path):
        print(f'  !! no results.json at {res_path}')
        return None
    with open(res_path) as f:
        res = json.load(f)
    res['_ablation_tag']    = tag
    res['_ablation_scene']  = scene
    res['_ablation_F']      = F
    res['_ablation_log']    = log
    res['_ablation_dt_s']   = float(dt)
    res['_ablation_outdir'] = out_dir
    return res


def run_scene(scene, F, use_2gpu=True):
    rows = _build_rows_for(scene, F)
    results = []
    if not use_2gpu or len(rows) <= 1:
        for tag, args in rows:
            r = _run_one(scene, F, tag, args, gpu=0)
            if r: results.append(r)
        return results

    # 2-GPU round-robin
    half = (len(rows) + 1) // 2
    from threading import Thread
    out = {}
    def worker(gpu, batch):
        for tag, args in batch:
            r = _run_one(scene, F, tag, args, gpu=gpu)
            if r: out[tag] = r
    t0 = Thread(target=worker, args=(0, rows[:half]))
    t1 = Thread(target=worker, args=(1, rows[half:]))
    t0.start(); t1.start(); t0.join(); t1.join()
    # Preserve order
    results = [out[tag] for tag, _ in rows if tag in out]
    return results


def _fmt_table(results, scene, F):
    lines = []
    lines.append(f'\n## {scene} F={F}\n')
    lines.append('| row | |RA| train | |RA| test | |RAD| train | |RAD| test | elapsed (s) |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for r in results:
        def _f(k): v = r.get(k); return '---' if v is None else f'{v:.4f}'
        lines.append(
            f'| {r["_ablation_tag"]} | '
            f'{_f("final_train_mean_cc")} | '
            f'{_f("final_test_cc")} | '
            f'{_f("final_train_rad_cc_mean")} | '
            f'{_f("final_test_rad_cc")} | '
            f'{r["_ablation_dt_s"]:.0f} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='seq_0_frame_135')
    ap.add_argument('--frame', type=int, default=135)
    ap.add_argument('--use_2gpu', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'\n===== Ablation matrix: {args.scene} F={args.frame} =====')
    rs = run_scene(args.scene, args.frame, use_2gpu=args.use_2gpu)
    # Write JSON
    jpath = os.path.join(OUT_DIR,
                          f'ablation_matrix_{args.scene}_F{args.frame}.json')
    with open(jpath, 'w') as f:
        json.dump({'scene': args.scene, 'F': args.frame,
                    'iters': BASE_FLAGS['iters'],
                    'target_n': BASE_FLAGS['target_n'],
                    'rows': rs}, f, indent=2)
    # Append markdown row to ablation_matrix.md
    mpath = os.path.join(OUT_DIR, 'ablation_matrix.md')
    if not os.path.exists(mpath):
        with open(mpath, 'w') as f:
            f.write('# §2.5 Ablation matrix\n\n'
                     'Single-axis variants; 500 iter, target_n=20000.\n')
    with open(mpath, 'a') as f:
        f.write(_fmt_table(rs, args.scene, args.frame) + '\n')
    print(f'\n[done] wrote {jpath}')
    print(f'[done] updated {mpath}')


if __name__ == '__main__':
    main()
