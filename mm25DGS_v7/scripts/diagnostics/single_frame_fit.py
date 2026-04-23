"""§2.1 — single-frame fit ceiling.

Train on ONE frame, eval on the SAME frame. Zero NVS, zero
generalization. Pure representational-capacity test: can the renderer
match GT |RA| / |RAD| of a single frame given unlimited grace?

If train CC plateaus below 0.9 on |RA| or |RAD|, the forward model /
point-cloud geometry / BSDF IS the blocker — no optimization trick
saves us. If train CC reaches > 0.95, the representation is fine and
the multi-frame plateau we see is a generalization / NVS issue.

Runs:
  (1) v5/M1-style: --doppler OFF, loss = mse_raw on |RA|
  (2) v7 doppler: --doppler ON,  loss = mse_raw on |RAD|, loss_norm=mean

Each uses 2000 iters, target_n=90000, seed_frame=F, train=[F], test=F.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]

REPO_ROOT = '/home/adnan/Desktop/mm3DGS'
OUT_DIR   = '/home/adnan/Desktop/mm3DGS/md/diagnostics'
PY        = '/home/adnan/.conda/envs/mmir/bin/python'

# Reserve these many iters — we want the plateau, not early progress.
ITERS = 2000
TARGET_N = 90000


def _run_one(scene: str, F: int, mode: str, gpu: int) -> dict:
    """mode in {'v5_style', 'v7_doppler'}. Returns result-dict path."""
    tag = f'{scene}_SINGLE_{mode}'
    log = os.path.join(REPO_ROOT, 'logs_v7', 'single_frame_fit', f'{tag}.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)

    cmd = [
        PY, '-m', 'mm25DGS_v7.train_frame_nvs',
        '--scene', scene,
        '--test_frame', str(F),
        # ***train on test frame itself*** — single-frame capacity test
        '--train_frames', str(F),
        '--train_loops', '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
            if mode == 'v7_doppler' else '0',
        '--iters', str(ITERS),
        '--loss_type', 'mse_raw',
        '--target_n', str(TARGET_N),
    ]
    if mode == 'v7_doppler':
        cmd += ['--doppler', '--loss_norm', 'mean']

    # Use distinct output dir suffix so this doesn't collide with the bench
    # (train_frame_nvs builds its tag from train_frames + train_loops count
    # so train=[F] + single loop already disambiguates).

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print(f'[gpu {gpu}] {tag} launching ({ITERS} iters, target_n={TARGET_N})')
    t0 = time.time()
    with open(log, 'w') as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                               cwd=REPO_ROOT)
    print(f'[gpu {gpu}] {tag} done in {time.time()-t0:.0f}s '
          f'(exit={proc.returncode})  log={log}')

    # Infer output dir from the tagging convention
    # train_frame_nvs: f'train{len(train_frames)}frames_{len(train_loops)}loops_test{F}_loop{held_out_loop}...'
    # In-train flag adds '_ub' suffix when test_frame in train_frames (it is here).
    if mode == 'v7_doppler':
        out_tag = (f'train1frames_16loops_test{F}_loop0_ub_pass2'
                    f'_N{TARGET_N}_v7doppler_normmean')
    else:
        out_tag = f'train1frames_1loops_test{F}_loop0_ub_pass2_N{TARGET_N}'
    out_dir = os.path.join(REPO_ROOT, 'mm25DGS_v7', 'output_frame_nvs',
                            f'{scene}_{out_tag}')
    return {'mode': mode, 'scene': scene, 'F': F,
             'log': log, 'out_dir': out_dir,
             'exit_code': proc.returncode}


def main():
    """Serial runs on GPU 0 for simplicity (single-frame fits are fast)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    # We kick them serial since each uses target_n=90000 which is mem-heavy
    # and we want clean timing.
    for mode in ['v5_style', 'v7_doppler']:
        for scene, F in SCENES:
            r = _run_one(scene, F, mode, gpu=0)
            # Load results.json if run succeeded
            res_path = os.path.join(r['out_dir'], 'results.json')
            if os.path.isfile(res_path):
                res = json.load(open(res_path))
                r['final_train_mean_cc']    = res.get('final_train_mean_cc')
                r['final_test_cc']          = res.get('final_test_cc')
                r['final_train_rad_cc']     = res.get('final_train_rad_cc_mean')
                r['final_test_rad_cc']      = res.get('final_test_rad_cc')
                r['best_mean_train_cc']     = res.get('best_mean_train_cc')
            results.append(r)

    json_path = os.path.join(OUT_DIR, 'single_frame_fit_results.json')
    with open(json_path, 'w') as f:
        json.dump({'iters': ITERS, 'target_n': TARGET_N, 'runs': results},
                   f, indent=2)

    # Markdown report
    md_path = os.path.join(OUT_DIR, 'single_frame_fit_results.md')
    with open(md_path, 'w') as f:
        f.write(f'# §2.1 Single-frame fit ceiling\n\n')
        f.write(f'Train on ONE frame, eval on the SAME frame. '
                 f'{ITERS} iters, target_n={TARGET_N}.\n\n')
        f.write('**Purpose.** If train CC < 0.9 here, representation is '
                 'the blocker. If ≥ 0.95, it is NVS / generalization.\n\n')

        for mode in ['v5_style', 'v7_doppler']:
            f.write(f'## {mode}\n\n')
            f.write('| scene | F | |RA| train | |RA| test (=train) | '
                     '|RAD| train | |RAD| test |\n')
            f.write('|---|---:|---:|---:|---:|---:|\n')
            for r in results:
                if r['mode'] != mode:
                    continue
                ra_t  = r.get('final_train_mean_cc')
                ra_T  = r.get('final_test_cc')
                rad_t = r.get('final_train_rad_cc')
                rad_T = r.get('final_test_rad_cc')
                def _fmt(x): return '---' if x is None else f'{x:.4f}'
                f.write(f'| {r["scene"]} | {r["F"]} | {_fmt(ra_t)} | '
                         f'{_fmt(ra_T)} | {_fmt(rad_t)} | {_fmt(rad_T)} |\n')
            # Mean row
            def _mean(key, mode_):
                v = [r.get(key) for r in results
                      if r['mode'] == mode_ and r.get(key) is not None]
                return f'{sum(v)/len(v):.4f}' if v else '---'
            f.write(f'| **mean** | — | '
                     f'{_mean("final_train_mean_cc", mode)} | '
                     f'{_mean("final_test_cc", mode)} | '
                     f'{_mean("final_train_rad_cc", mode)} | '
                     f'{_mean("final_test_rad_cc", mode)} |\n\n')

    print(f'\n[done] wrote {json_path}')
    print(f'[done] wrote {md_path}')


if __name__ == '__main__':
    main()
