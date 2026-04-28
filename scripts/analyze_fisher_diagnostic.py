"""Phase A — Fisher-imbalance diagnostic analysis.

Loads `fisher_diagnostic.pt` from each scene's v5_v4 output_frame_nvs
directory and prints + plots:

  * Cumulative concentration (top-k% of points → what % of total ‖∇p L‖)
  * Per-densify-window snapshot table (iter 100, 200, 300, 400)
  * 3D scatter coloured by Fisher quantile (one PNG per scene)

Usage (run after `run_v5_v4_defaults_6scene.sh` with instrumented code):

    python -m scripts.analyze_fisher_diagnostic
"""
from __future__ import annotations

import json
import os
import sys
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
OUT_GLOB = os.path.join(
    PROJECT_ROOT,
    'mm25DGS_v5_v4', 'output_frame_nvs',
    '*_dnsfy*_dsigpos_grad_amp_lpos1e-05L2100',
)
OUT_DIR = os.path.join(PROJECT_ROOT, 'md', 'diagnostics_v5_v4_fisher')
os.makedirs(OUT_DIR, exist_ok=True)


def find_scene_diagnostic(scene: str) -> str | None:
    """Find the most recent fisher_diagnostic.pt for a given scene."""
    paths = []
    for d in glob.glob(OUT_GLOB):
        bn = os.path.basename(d)
        if not bn.startswith(scene + '_'):
            continue
        diag = os.path.join(d, 'fisher_diagnostic.pt')
        if os.path.exists(diag):
            paths.append((os.path.getmtime(diag), diag))
    if not paths:
        return None
    paths.sort(reverse=True)
    return paths[0][1]


def cumulative_topk_table(grad_per_pt: np.ndarray) -> dict:
    sg = np.sort(grad_per_pt)[::-1]
    total = max(sg.sum(), 1e-30)
    cumsum = np.cumsum(sg) / total
    N = len(sg)
    out = {'N': N}
    for frac in (0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00):
        k = max(1, int(frac * N))
        out[f'top_{frac:.4f}'] = float(cumsum[k - 1]) * 100.0
    return out


def plot_3d_quantile(positions: np.ndarray, grad_per_pt: np.ndarray,
                     scene: str, out_path: str):
    """3D scatter of points coloured by Fisher quantile bucket."""
    rank = np.argsort(np.argsort(-grad_per_pt))
    quantile = rank / max(1, len(grad_per_pt) - 1)
    fig = plt.figure(figsize=(14, 5))
    for i, (lo, hi, label, color) in enumerate([
        (0.0,    0.01,  'top 1%',     'red'),
        (0.01,   0.10,  'top 1-10%',  'orange'),
        (0.10,   1.00,  'bottom 90%', 'lightgrey'),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        m = (quantile >= lo) & (quantile < hi)
        # Draw bottom 90% behind everything for context
        if i == 2:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        s=0.5, c='lightgrey', alpha=0.2)
            ax.scatter(positions[m, 0], positions[m, 1], positions[m, 2],
                        s=0.5, c=color, alpha=0.4)
        else:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        s=0.3, c='lightgrey', alpha=0.1)
            ax.scatter(positions[m, 0], positions[m, 1], positions[m, 2],
                        s=4 if i == 0 else 2, c=color, alpha=0.9)
        ax.set_title(f'{scene}\n{label}  (n={int(m.sum())})')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    scenes = ['seq_0_frame_135', 'seq_1_frame_185', 'seq_1_frame_438',
              'seq_2_frame_105', 'seq_2_frame_160', 'seq_2_frame_300']
    rows_career = []
    rows_window = []
    rows_log = []  # one row per (scene, snapshot iter)
    for s in scenes:
        diag_path = find_scene_diagnostic(s)
        if diag_path is None:
            print(f'[!] no diagnostic found for {s}')
            continue
        d = torch.load(diag_path, map_location='cpu', weights_only=False)
        career = d['career_pos_grad'].numpy()
        last   = d['last_window_pos_grad'].numpy() if d['last_window_pos_grad'] is not None else None
        positions = d['final_positions'].numpy()

        ct = cumulative_topk_table(career); ct['scene'] = s
        ct['final_test_cc'] = d['final_test_cc']
        rows_career.append(ct)

        if last is not None:
            wt = cumulative_topk_table(last); wt['scene'] = s
            rows_window.append(wt)

        for snap in d['concentration_log']:
            row = {'scene': s, 'iter': snap['iter']}
            for k, v in snap.items():
                if k.startswith('top_'):
                    row[k] = v
            rows_log.append(row)

        plot_path = os.path.join(OUT_DIR, f'{s}_3d_quantile.png')
        plot_3d_quantile(positions, career, s, plot_path)
        print(f'  [{s}] N={ct["N"]}  test_cc={ct["final_test_cc"]:.4f}  '
              f'top-1%={ct["top_0.0100"]:.1f}%  '
              f'top-5%={ct["top_0.0500"]:.1f}%  '
              f'top-10%={ct["top_0.1000"]:.1f}%  '
              f'top-25%={ct["top_0.2500"]:.1f}%  '
              f'plot={plot_path}')

    def fmt_table(rows: list[dict], title: str) -> str:
        if not rows:
            return f'## {title}\n\n_no data_\n'
        keys = ['scene', 'top_0.0010', 'top_0.0050', 'top_0.0100',
                'top_0.0500', 'top_0.1000', 'top_0.2500', 'top_0.5000']
        if 'final_test_cc' in rows[0]:
            keys.append('final_test_cc')
        out = f'## {title}\n\n| ' + ' | '.join(
            ['scene', 'top 0.1%', 'top 0.5%', 'top 1%', 'top 5%',
             'top 10%', 'top 25%', 'top 50%']
            + (['test cc'] if 'final_test_cc' in rows[0] else [])
        ) + ' |\n|' + '|'.join(['---'] * len(keys)) + '|\n'
        for r in rows:
            cells = [r['scene']]
            for k in keys[1:]:
                v = r.get(k)
                if v is None:
                    cells.append('—')
                elif k == 'final_test_cc':
                    cells.append(f'{v:.4f}')
                else:
                    cells.append(f'{v:.1f}%')
            out += '| ' + ' | '.join(cells) + ' |\n'
        # mean row
        if len(rows) > 1:
            cells = ['**mean**']
            for k in keys[1:]:
                vs = [r[k] for r in rows if r.get(k) is not None]
                if not vs:
                    cells.append('—')
                elif k == 'final_test_cc':
                    cells.append(f'**{np.mean(vs):.4f}**')
                else:
                    cells.append(f'**{np.mean(vs):.1f}%**')
            out += '| ' + ' | '.join(cells) + ' |\n'
        return out + '\n'

    md_path = os.path.join(OUT_DIR, 'concentration.md')
    with open(md_path, 'w') as f:
        f.write('# Phase A — Fisher imbalance diagnostic\n\n')
        f.write('Reads `fisher_diagnostic.pt` from each scene\'s v5_v4 '
                'default (combo_jitter recipe) output dir. Each row = '
                'one scene. Each column = the cumulative fraction of '
                'total ‖∇p L‖ held by the top-k% of points.\n\n')
        f.write(fmt_table(rows_career,
                          'Career accumulator (entire training run)'))
        f.write(fmt_table(rows_window,
                          'Last densify window only (iter 400–500)'))
        if rows_log:
            f.write('## Per-densify snapshot (each scene × each densify '
                    'event = one row)\n\n')
            f.write('| scene | iter | top 0.5% | top 1% | top 5% | '
                    'top 10% | top 25% | top 50% |\n'
                    '|---|---:|---:|---:|---:|---:|---:|---:|\n')
            for r in rows_log:
                f.write(f'| {r["scene"]} | {r["iter"]} | '
                        f'{r.get("top_0.0050", float("nan")):.1f}% | '
                        f'{r.get("top_0.0100", float("nan")):.1f}% | '
                        f'{r.get("top_0.0500", float("nan")):.1f}% | '
                        f'{r.get("top_0.1000", float("nan")):.1f}% | '
                        f'{r.get("top_0.2500", float("nan")):.1f}% | '
                        f'{r.get("top_0.5000", float("nan")):.1f}% |\n')
        f.write('\n## 3D scatter plots\n\n'
                '(One PNG per scene, three panels: top 1% / top 1-10% / '
                'bottom 90% in 3D space.)\n\n')
        for s in scenes:
            png = f'{s}_3d_quantile.png'
            if os.path.exists(os.path.join(OUT_DIR, png)):
                f.write(f'![{s}]({png})\n\n')
    print(f'\n[done] wrote {md_path}')


if __name__ == '__main__':
    main()
