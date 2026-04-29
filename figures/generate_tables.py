"""Generate the two LaTeX results tables for the NeurIPS paper.

Tables:
  1. test_ra_results.tex   — per-scene + mean test |RA| Pearson CC for
                              Ours / DART / RadarSplat / RadarFields,
                              held-out NVS test frame.
  2. train_ra_results.tex  — per-scene + mean train |RA| Pearson CC,
                              ours only (baselines do not expose
                              per-train-frame metrics).

Reads:
  - mm25DGS_v5_v4/output_frame_nvs/<scene>.../results.json
        (final_test_cc, final_train_mean_cc, final_train_std_cc)
  - baselines/{dart,radarsplat,radarfields}/results/<scene>/metrics.json
        (ra_corr — held-out test frame)
  - For DART, also baselines/dart/results/<scene>__cascaded/metrics.json
        is the cascaded-radar variant (default in our paper).

Writes the two `.tex` files into the paper's `figs/` (or wherever
specified via --output_dir).

Usage:
    python -m figures.generate_tables \
        --ours_dir   mm25DGS_v5_v4/output_frame_nvs \
        --baselines_dir baselines \
        --output_dir latex/NeurIPS_2026_unpacked/Physically_Grounded_Novel_View_Synthesis_for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering/tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional


SCENES = [
    'seq_0_frame_135',
    'seq_1_frame_185',
    'seq_1_frame_438',
    'seq_2_frame_105',
    'seq_2_frame_160',
    'seq_2_frame_300',
]
SCENE_SHORT = {
    'seq_0_frame_135': 'S0\\,F135',
    'seq_1_frame_185': 'S1\\,F185',
    'seq_1_frame_438': 'S1\\,F438',
    'seq_2_frame_105': 'S2\\,F105',
    'seq_2_frame_160': 'S2\\,F160',
    'seq_2_frame_300': 'S2\\,F300',
}


def find_ours_results(scene: str, ours_dir: str) -> Optional[str]:
    pat1 = os.path.join(ours_dir, f'{scene}_*_pass2_N20000', 'results.json')
    pat2 = os.path.join(ours_dir,
        f'{scene}_*_pass2_N20000_dnsfyjt0.05i100u400p0.02_dsigpos_grad_amp_lpos1e-05L2100',
        'results.json')
    matches = sorted(glob.glob(pat1) + glob.glob(pat2),
                      key=os.path.getmtime, reverse=True)
    matches = [m for m in matches if '_initC' not in m and '_initB' not in m
                and '_initA' not in m and '_p4d' not in m]
    return matches[0] if matches else None


def load_ours(scene: str, ours_dir: str) -> dict | None:
    p = find_ours_results(scene, ours_dir)
    if p is None:
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        'test':  float(d['final_test_cc']),
        'train': float(d['final_train_mean_cc']),
        'train_std': float(d.get('final_train_std_cc', 0.0)),
        'init_test': float(d.get('init_test_cc', 0.0)),
    }


def load_baseline(baseline: str, scene: str, baselines_dir: str) -> dict | None:
    if baseline == 'dart':
        p = os.path.join(baselines_dir, 'dart', 'results',
                          f'{scene}__cascaded', 'metrics.json')
    else:
        p = os.path.join(baselines_dir, baseline, 'results', scene, 'metrics.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        'test': float(d.get('ra_corr', float('nan'))),
        'rp':   float(d.get('range_profile_corr', float('nan'))),
    }


def fmt(x: float | None, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return '--'
    return f'{x:.{decimals}f}'


def write_test_table(rows: dict[str, dict[str, float]], path: str):
    """rows: {method: {scene: cc, 'mean': mean_cc}}."""
    methods = ['ours', 'radarsplat', 'radarfields', 'dart']
    pretty = {'ours': '\\textbf{mm3DGS (Ours)}',
              'radarsplat': 'RadarSplat~\\cite{kung2025radarsplat}',
              'radarfields': 'Radar Fields~\\cite{10.1145/3641519.3657510}',
              'dart': 'DART~\\cite{huang2024dart}'}
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Held-out novel-view test |RA| Pearson correlation '
                  'across six ColoRadar scenes. For each scene we train on the '
                  '8 cascaded radar frames adjacent to the held-out test frame '
                  '($F\\!\\pm\\!1\\!\\ldots\\!\\pm\\!4$, chirp 0 only) and '
                  'evaluate on the held-out test frame ($F$, chirp 0). Higher '
                  'is better. Bold marks the best per scene.}')
    lines.append('\\label{tab:test_ra}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    lines.append('\\begin{tabular}{l|cccccc|c}')
    lines.append('\\toprule')
    header = '\\textbf{Method} & ' + ' & '.join(
        SCENE_SHORT[s] for s in SCENES) + ' & \\textbf{Mean} \\\\'
    lines.append(header)
    lines.append('\\midrule')
    # Find best per scene to bold.
    bests = {}
    for s in SCENES:
        vals = {m: rows.get(m, {}).get(s) for m in methods}
        best_m = max((v for v in vals.values() if v is not None), default=None)
        for m, v in vals.items():
            if v is not None and v == best_m:
                bests[s] = m
    for m in methods:
        cells = [pretty[m]]
        for s in SCENES:
            v = rows.get(m, {}).get(s)
            cell = fmt(v)
            if bests.get(s) == m:
                cell = f'\\textbf{{{cell}}}'
            cells.append(cell)
        cells.append(f'\\textbf{{{fmt(rows.get(m, {}).get("mean"))}}}')
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_train_table(ours_rows: dict[str, dict], path: str):
    """ours_rows[scene] = {'train': cc, 'train_std': std, 'init_test': init}.
    Plus a 'mean' key with means.
    """
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Training-view |RA| Pearson correlation. mm3DGS '
                  'mean across the 8 train frames per scene (chirp 0 only); '
                  'std is the across-frame deviation. Pre-training (init) '
                  'CC is reported alongside to quantify the optimization '
                  'lift. Baseline methods do not expose per-train-frame '
                  'metrics in their public release; we mark them as ``--\'\' '
                  '(re-rendering each baseline at all 8 train poses is left '
                  'to future work).}')
    lines.append('\\label{tab:train_ra}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{6pt}')
    lines.append('\\begin{tabular}{l|cccc}')
    lines.append('\\toprule')
    lines.append('\\textbf{Scene} & '
                  '\\textbf{init test CC} & '
                  '\\textbf{train mean CC (std)} & '
                  '\\textbf{Baselines} \\\\')
    lines.append('\\midrule')
    for s in SCENES:
        r = ours_rows.get(s, {})
        init = fmt(r.get('init_test'))
        tr = fmt(r.get('train'))
        std = fmt(r.get('train_std'))
        lines.append(
            f'{SCENE_SHORT[s]} & {init} & {tr}\\,({std}) & -- \\\\')
    mr = ours_rows.get('mean', {})
    lines.append('\\midrule')
    lines.append(
        f'\\textbf{{Mean}} & '
        f'\\textbf{{{fmt(mr.get("init_test"))}}} & '
        f'\\textbf{{{fmt(mr.get("train"))}}} & -- \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours_dir', default='mm25DGS_v5_v4/output_frame_nvs')
    ap.add_argument('--baselines_dir', default='baselines')
    ap.add_argument('--output_dir', default=(
        'latex/NeurIPS_2026_unpacked/Physically_Grounded_Novel_View_Synthesis_'
        'for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering/tables'))
    args = ap.parse_args()

    rows = {'ours': {}, 'dart': {}, 'radarsplat': {}, 'radarfields': {}}
    ours_rows = {}

    for s in SCENES:
        o = load_ours(s, args.ours_dir)
        if o is not None:
            rows['ours'][s] = o['test']
            ours_rows[s] = o
        else:
            print(f'[!] no ours results for {s}')
        for b in ('dart', 'radarsplat', 'radarfields'):
            r = load_baseline(b, s, args.baselines_dir)
            if r is not None:
                rows[b][s] = r['test']

    # Per-method test mean
    for m in rows:
        vals = list(rows[m].values())
        rows[m]['mean'] = sum(vals) / len(vals) if vals else float('nan')

    # Ours train mean across scenes
    if ours_rows:
        ours_rows['mean'] = {
            'init_test': sum(o['init_test'] for o in ours_rows.values()) / len(ours_rows),
            'train': sum(o['train'] for o in ours_rows.values()) / len(ours_rows),
            'train_std': sum(o['train_std'] for o in ours_rows.values()) / len(ours_rows),
        }

    # Write
    write_test_table(rows,
                     os.path.join(args.output_dir, 'test_ra_results.tex'))
    write_train_table(ours_rows,
                      os.path.join(args.output_dir, 'train_ra_results.tex'))
    print(f'[done] wrote tables under {args.output_dir}/')

    # Print summary
    print('\n=== per-method test |RA| CC mean ===')
    for m in ('ours', 'radarsplat', 'radarfields', 'dart'):
        print(f'  {m:>14}: {rows[m].get("mean", float("nan")):.4f}')
    print('\n=== ours per-scene ===')
    for s in SCENES:
        if s in ours_rows:
            o = ours_rows[s]
            print(f'  {s:>20}: init {o["init_test"]:.4f}  '
                  f'test {o["test"]:.4f}  train {o["train"]:.4f}±{o["train_std"]:.4f}')


if __name__ == '__main__':
    main()
