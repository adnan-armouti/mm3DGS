"""Plot v5/M1 vs v7 Doppler train/test CC curves per scene.

Curves:
  v5 train_cc (per iter, from history.npz)
  v5 test_cc  (per iter, from history.npz)
  v7 train_cc (per iter, from history.npz)
  v7 test_cc  markers: init at iter=0, final at iter=500
             (per-iter tracking was added AFTER the current bench;
              re-run pending user direction).

Writes ``cc_compare_v5_v7.png`` into each v7 output dir.
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


V5_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'
V7_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v7/output_frame_nvs'

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]


def _v5_dir(scene, F):
    return (f'{V5_ROOT}/{scene}_train8frames_1loops_test{F}'
            f'_loop0_pass2_N20000_v6M1')


def _v7_dir(scene, F):
    return (f'{V7_ROOT}/{scene}_train8frames_16loops_test{F}'
            f'_loop0_pass2_N20000_v7doppler')


def _plot_scene(scene, F):
    v5d = _v5_dir(scene, F)
    v7d = _v7_dir(scene, F)
    if not (os.path.isdir(v5d) and os.path.isdir(v7d)):
        print(f'{scene}: missing dir v5={os.path.isdir(v5d)} '
              f'v7={os.path.isdir(v7d)} — skip')
        return

    v5_h = np.load(os.path.join(v5d, 'history.npz'))
    v7_h = np.load(os.path.join(v7d, 'history.npz'))
    v7_r = json.load(open(os.path.join(v7d, 'results.json')))
    v5_r = json.load(open(os.path.join(v5d, 'results.json')))

    iters_v5 = v5_h['iters']
    iters_v7 = v7_h['iters']
    v5_train = v5_h['mean_train_cc']
    v5_test  = v5_h['test_cc']
    v7_train = v7_h['mean_train_cc']

    v7_init_test = float(v7_r['init_test_cc'])
    v7_final_test = float(v7_r['final_test_cc'])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(iters_v5, v5_train, color='#1f77b4', lw=1.4,
            label=f'v5/M1 train (final {v5_train[-1]:.3f})')
    ax.plot(iters_v5, v5_test,  color='#1f77b4', lw=1.4, ls='--',
            label=f'v5/M1 test  (final {v5_test[-1]:.3f})')
    ax.plot(iters_v7, v7_train, color='#d62728', lw=1.4,
            label=f'v7 Dop train (final {v7_train[-1]:.3f})')

    ax.scatter([iters_v7[0], iters_v7[-1]],
               [v7_init_test, v7_final_test],
               color='#d62728', marker='o', s=60, zorder=5,
               label=f'v7 Dop test endpoints '
                     f'({v7_init_test:.3f} → {v7_final_test:.3f})')
    ax.plot([iters_v7[0], iters_v7[-1]],
            [v7_init_test, v7_final_test],
            color='#d62728', ls=':', lw=1.0, alpha=0.6)

    ax.axhline(0.0, color='k', lw=0.4, alpha=0.3)
    ax.set_xlabel('iteration')
    ax.set_ylabel('cart-corr (chirp-0 |RA|)')
    ax.set_title(f'{scene}  F={F}   v5/M1 vs v7 Doppler')
    ax.set_ylim(min(-0.02,
                    float(min(v5_test.min(), v7_train.min())) - 0.02),
                max(1.0,
                    float(max(v5_train.max(), v7_train.max())) + 0.02))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

    out = os.path.join(v7d, 'cc_compare_v5_v7.png')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'  wrote {out}')


def main():
    print('generating v5 vs v7 CC comparison plots...')
    for scene, F in SCENES:
        _plot_scene(scene, F)
    print('done.')


if __name__ == '__main__':
    main()
