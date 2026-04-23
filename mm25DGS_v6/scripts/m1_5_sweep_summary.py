"""Aggregate M1.5 loss-variant sweep results across 7 scenes.

For each variant present in mm25DGS_v6/output_frame_nvs/ with tag
`_v6M1_5` (frobenius baseline) or `_v6M1_5_<variant>` (sweep members),
reads per-scene results.json and reports per-scene final_test_cc plus
the 7-scene mean, ranked. Also compares against the v5/M1 baseline.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390',
    'seq_1_frame_185', 'seq_1_frame_438',
    'seq_2_frame_105', 'seq_2_frame_160', 'seq_2_frame_300',
]
VARIANTS = [
    'frobenius',  # baseline tagged as just _v6M1_5 (no suffix)
    'coarray', 'smooth_alpha',
    'mag_weighted', 'baseline_weighted', 'inv_variance',
    'diag_offdiag', 'range_integrated', 'modulus',
]
OUTPUT_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'


def find_run_dir(scene: str, variant: str) -> str | None:
    """Locate the output dir matching scene+variant."""
    if variant == 'frobenius':
        # Tagged as plain `_v6M1_5` with nothing after.
        cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_v6M1_5')
    else:
        cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_v6M1_5_{variant}')
    # Filter out archived _normalized etc.
    cand = [c for c in cand if not c.endswith('_normalized')]
    return cand[0] if cand else None


def load_cc(scene: str, variant: str) -> float | None:
    d = find_run_dir(scene, variant)
    if d is None:
        return None
    try:
        r = json.load(open(os.path.join(d, 'results.json')))
    except Exception:
        return None
    return float(r.get('final_test_cc', float('nan')))


def load_diag(scene: str, variant: str, key: str) -> float | None:
    d = find_run_dir(scene, variant)
    if d is None:
        return None
    try:
        r = json.load(open(os.path.join(d, 'results.json')))
    except Exception:
        return None
    v = r.get(key)
    return None if v is None else float(v)


def load_v5_baseline(scene: str) -> float | None:
    """v5/M1 baseline is stored under _v6M1 tag (M1 plumbed but did not
    alter v5's mse_raw loss, so final_test_cc is v5-equivalent within
    MC noise)."""
    cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_v6M1')
    cand = [c for c in cand if '_v6M1_5' not in c]  # exclude M1.5 dirs
    if not cand:
        return None
    try:
        r = json.load(open(os.path.join(cand[0], 'results.json')))
    except Exception:
        return None
    return float(r.get('final_test_cc', float('nan')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='store_true')
    args = ap.parse_args()

    # Collect all variants with >=1 scene result so we display partial
    # progress cleanly too.
    data = {}  # variant -> dict of scene -> cc
    for v in VARIANTS:
        d = {s: load_cc(s, v) for s in SCENES}
        if any(cc is not None for cc in d.values()):
            data[v] = d

    v5 = {s: load_v5_baseline(s) for s in SCENES}
    v5_mean = (
        sum(x for x in v5.values() if x is not None) /
        max(1, sum(1 for x in v5.values() if x is not None))
    )

    # Ranked table
    def safe_mean(d):
        vals = [x for x in d.values() if x is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    ranking = sorted(data.items(), key=lambda kv: -safe_mean(kv[1]))

    # Print header
    print()
    print('M1.5 loss-variant sweep — 7-scene HO_8 benchmark')
    print('=' * 102)
    hdr = f'{"variant":<20s} | ' + '  '.join([f'{s[-9:]:>9s}' for s in SCENES]) + f' | {"mean":>7s}  {"Δ vs v5":>8s}'
    print(hdr)
    print('-' * 102)

    def fmt_cc(x):
        return '  ???  ' if x is None else f'{x:7.4f}'

    def fmt_delta(d):
        sign = '+' if d >= 0 else '-'
        return f'{sign}{abs(d):6.4f}'

    v5_line = f'{"v5 / M1 baseline":<20s} | ' + '  '.join([fmt_cc(v5[s]) for s in SCENES]) + f' | {v5_mean:7.4f}  {fmt_delta(0.0)}'
    print(v5_line)
    print('-' * 102)

    for v, d in ranking:
        m = safe_mean(d)
        delta = m - v5_mean
        row = f'{v:<20s} | ' + '  '.join([fmt_cc(d[s]) for s in SCENES]) + f' | {m:7.4f}  {fmt_delta(delta)}'
        print(row)

    print('=' * 102)

    # Diag gram-cc columns (supplementary)
    print()
    print('Supplementary: diag_normalised_gram_cc_test_mag (per-virt magnitude'
          '-only Gram at test; v5/M1 ≈ 0.57)')
    print('-' * 102)
    for v, d in ranking:
        diag_mag = [load_diag(s, v, 'diag_normalised_gram_cc_test_mag') for s in SCENES]
        vals = [x for x in diag_mag if x is not None]
        m = sum(vals)/len(vals) if vals else float('nan')
        row = f'{v:<20s} | ' + '  '.join([fmt_cc(x) for x in diag_mag]) + f' | mean = {m:.4f}'
        print(row)

    if args.csv:
        print()
        print('# CSV')
        print('variant,' + ','.join(SCENES) + ',mean,delta_v5')
        for v, d in ranking:
            m = safe_mean(d)
            vals = [d[s] if d[s] is not None else float('nan') for s in SCENES]
            print(f'{v},' + ','.join(f'{x:.4f}' for x in vals) + f',{m:.4f},{m-v5_mean:+.4f}')


if __name__ == '__main__':
    main()
