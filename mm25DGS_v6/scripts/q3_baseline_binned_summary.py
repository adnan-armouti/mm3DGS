"""Q3 summary: baseline-binned Gram loss variants × 7 scenes."""
from __future__ import annotations
import glob
import json
import os

import numpy as np

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390',
    'seq_1_frame_185', 'seq_1_frame_438',
    'seq_2_frame_105', 'seq_2_frame_160', 'seq_2_frame_300',
]
VARIANTS = [
    'baseline_binned',
    'baseline_binned_wiener',
    'baseline_binned_wiener_skipb0',
]
OUTPUT_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'

# v5/M1 baseline (from the new PNG-enabled re-run, recorded earlier).
V5_PER_SCENE = {
    'seq_0_frame_135': 0.5860,
    'seq_0_frame_390': 0.2872,
    'seq_1_frame_185': 0.5595,
    'seq_1_frame_438': 0.5757,
    'seq_2_frame_105': 0.5596,
    'seq_2_frame_160': 0.6544,
    'seq_2_frame_300': 0.4762,
}
NAIVE_AVG_PER_SCENE = {
    'seq_0_frame_135': 0.6599,
    'seq_0_frame_390': 0.2783,
    'seq_1_frame_185': 0.5863,
    'seq_1_frame_438': 0.3282,
    'seq_2_frame_105': 0.6068,
    'seq_2_frame_160': 0.7722,
    'seq_2_frame_300': 0.5949,
}


def _load(scene: str, variant: str) -> float | None:
    cand = glob.glob(
        f'{OUTPUT_ROOT}/{scene}_*_v6M1_5_{variant}')
    if not cand:
        return None
    try:
        r = json.load(open(os.path.join(cand[0], 'results.json')))
        return float(r['final_test_cc'])
    except Exception:
        return None


def main():
    rows = {}
    for v in VARIANTS:
        rows[v] = {s: _load(s, v) for s in SCENES}

    def safe_mean(d):
        vals = [x for x in d.values() if x is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    print()
    print('Q3 — baseline-binned loss (7-scene HO_8)')
    print('=' * 110)
    hdr = f'{"variant":<34s} | ' + '  '.join([f'{s[-9:]:>9s}' for s in SCENES]) + f' | {"mean":>7s}  {"vs v5":>8s}  {"vs naive":>9s}'
    print(hdr)
    print('-' * 110)

    def fmt(x): return '  ???  ' if x is None else f'{x:7.4f}'
    def fmts(d):
        m = d if isinstance(d, float) else None
        if m is None:
            return ' ???'
        sign = '+' if m >= 0 else ''
        return f'{sign}{m:6.4f}'

    v5_mean = sum(V5_PER_SCENE.values()) / 7
    naive_mean = sum(NAIVE_AVG_PER_SCENE.values()) / 7
    print(f'{"v5 / M1 baseline":<34s} | ' + '  '.join(
        [fmt(V5_PER_SCENE[s]) for s in SCENES])
          + f' | {v5_mean:7.4f}  {fmts(0.0)}  {fmts(v5_mean - naive_mean)}')
    print(f'{"naive ½·(F-1+F+1) avg":<34s} | ' + '  '.join(
        [fmt(NAIVE_AVG_PER_SCENE[s]) for s in SCENES])
          + f' | {naive_mean:7.4f}  {fmts(naive_mean - v5_mean)}  {fmts(0.0)}')
    print('-' * 110)
    for v, d in rows.items():
        m = safe_mean(d)
        print(f'{v:<34s} | ' + '  '.join([fmt(d[s]) for s in SCENES])
              + f' | {m:7.4f}  {fmts(m - v5_mean)}  {fmts(m - naive_mean)}')
    print('=' * 110)


if __name__ == '__main__':
    main()
