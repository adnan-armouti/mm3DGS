"""Aggregate v5-rerun + hybrid_mag sweep.

Compares:
  - v5 rerun (M1 milestone, plain v5 mse_raw, with PNGs saved).
  - hybrid_mag (M1.5 milestone, L_v5 + λ · per-ant mag MSE).

Reports per-scene final_test_cc + 7-scene mean, and points at the saved
PNG directories for qualitative inspection.
"""
from __future__ import annotations
import glob
import json
import os

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390',
    'seq_1_frame_185', 'seq_1_frame_438',
    'seq_2_frame_105', 'seq_2_frame_160', 'seq_2_frame_300',
]
OUTPUT_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'


def find_run(scene: str, tag: str) -> str | None:
    if tag == 'v5_rerun':
        # M1 milestone (loss unchanged), fresh rerun with PNGs.
        cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_v6M1')
        cand = [c for c in cand if '_nopngs_archived' not in c
                                   and '_v6M1_5' not in c]
    elif tag == 'hybrid_mag':
        cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_v6M1_5_hybrid_mag')
    elif tag == 'm1_original':
        cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_v6M1_nopngs_archived')
    else:
        return None
    return cand[0] if cand else None


def load_cc(d: str | None) -> float | None:
    if d is None:
        return None
    try:
        return float(json.load(open(os.path.join(d, 'results.json')))['final_test_cc'])
    except Exception:
        return None


def main():
    import numpy as np

    rows = {}
    for tag in ('m1_original', 'v5_rerun', 'hybrid_mag'):
        rows[tag] = {s: load_cc(find_run(s, tag)) for s in SCENES}

    def mean(d):
        vals = [v for v in d.values() if v is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    print()
    print('Hybrid loss benchmark — 7-scene HO_8')
    print('=' * 106)
    header = f'{"variant":<24s} | ' + '  '.join([f'{s[-9:]:>9s}' for s in SCENES]) + f' | {"mean":>7s}'
    print(header)
    print('-' * 106)

    def fmt(x):
        return '  ???  ' if x is None else f'{x:7.4f}'

    labels = {
        'm1_original': 'v5/M1 baseline (old)',
        'v5_rerun':    'v5 rerun (L = L_v5)',
        'hybrid_mag':  'hybrid  (L=L_v5+λ·L_mag)',
    }
    order = ['m1_original', 'v5_rerun', 'hybrid_mag']
    for tag in order:
        d = rows[tag]
        m = mean(d)
        label = labels[tag]
        print(f'{label:<24s} | ' + '  '.join([fmt(d[s]) for s in SCENES])
              + f' | {m:7.4f}')

    v5_mean = mean(rows['v5_rerun'])
    hybrid_mean = mean(rows['hybrid_mag'])
    m1_mean = mean(rows['m1_original'])

    print('=' * 106)
    if not any(v is None for v in list(rows['v5_rerun'].values())
                              + list(rows['hybrid_mag'].values())):
        print()
        print('Δ hybrid  − v5 rerun  = '
              f'{hybrid_mean - v5_mean:+.4f}  '
              f'(regressions: {sum(1 for s in SCENES if rows["hybrid_mag"][s] < rows["v5_rerun"][s])}/7)')
        print('Δ v5 rerun − M1 orig  = '
              f'{v5_mean - m1_mean:+.4f}  (MC-noise check: should be within ±0.03)')

    # Point at PNG dirs
    print()
    print('Saved RA PNGs + cc_history.png (qualitative inspection):')
    for scene in SCENES:
        v5d = find_run(scene, 'v5_rerun')
        hyd = find_run(scene, 'hybrid_mag')
        print(f'  {scene}:')
        print(f'    v5_rerun    -> {v5d}')
        print(f'    hybrid_mag  -> {hyd}')


if __name__ == '__main__':
    main()
