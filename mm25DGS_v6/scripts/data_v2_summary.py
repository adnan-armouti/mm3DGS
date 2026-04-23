"""Aggregate data_v2 training results.

Columns per scene (all v5/M1, 500 iters, first chirp only):
  - cc_data_v1       : existing v5/M1 from the data/ tree (F±4 HO_8)
                        — baselines from prior sweeps.
  - cc_data_v2_fpm4  : v5/M1 on data_v2, F±4 window (G3.1 bit-identity).
  - cc_data_v2_ext   : v5/M1 on data_v2, extended asymmetric window.
  - cc_naive_avg     : ½·(GT[F-1]+GT[F+1]) predictor on the test frame.

Mean over 6 scenes and per-scene deltas reported.
"""
from __future__ import annotations
import glob
import json
import os

# Hard-coded for clarity: v5/M1 baselines from the earlier hybrid sweep
# re-runs (see mm25DGS_v6/scripts/hybrid_summary.py output).
V5_M1_DATA_V1 = {
    'seq_0_frame_135': 0.5860,
    'seq_1_frame_185': 0.5595,
    'seq_1_frame_438': 0.5757,
    'seq_2_frame_105': 0.5490,
    'seq_2_frame_160': 0.6544,
    'seq_2_frame_300': 0.4762,
}
NAIVE_AVG = {
    'seq_0_frame_135': 0.6599,
    'seq_1_frame_185': 0.5863,
    'seq_1_frame_438': 0.3282,
    'seq_2_frame_105': 0.6068,
    'seq_2_frame_160': 0.7722,
    'seq_2_frame_300': 0.5949,
}

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]

OUTPUT_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'


def _find_run(scene: str, F_test: int, n_train: int) -> str | None:
    """Find a v6M1 run for this scene + test frame + train-frame count."""
    # Tag pattern: {scene}_train{n}frames_1loops_test{F}_loop0_pass2_N20000_v6M1
    cand = glob.glob(
        f'{OUTPUT_ROOT}/{scene}_train{n_train}frames_1loops_test{F_test}_loop0_pass2_N20000_v6M1'
    )
    cand = [c for c in cand if '_v6M1_5' not in c
                               and '_nopngs_archived' not in c]
    return cand[0] if cand else None


def _load_cc(d: str | None) -> float | None:
    if d is None:
        return None
    try:
        r = json.load(open(os.path.join(d, 'results.json')))
        return float(r['final_test_cc'])
    except Exception:
        return None


def main():
    # Per-scene frame counts (extended, for lookup by n_train).
    # Extended bracket size = (hi - lo) frames (center excluded).
    EXT = {
        'seq_0_frame_135': 39,    # hi - lo = 18 - (-21)
        'seq_1_frame_185': 50,    # 25 - (-25)
        'seq_1_frame_438': 65,    # 30 - (-35)
        'seq_2_frame_105': 33,    # 14 - (-19)
        'seq_2_frame_160': 30,    # 15 - (-15)
        'seq_2_frame_300': 43,    # 24 - (-19)
    }

    rows = []
    for scene, F in SCENES:
        fpm4_dir = _find_run(scene, F, 8)                # F±4 = 8 train frames
        ext_dir  = _find_run(scene, F, EXT[scene])
        cc_fpm4 = _load_cc(fpm4_dir)
        cc_ext  = _load_cc(ext_dir)
        rows.append({
            'scene': scene, 'F': F,
            'cc_v1': V5_M1_DATA_V1.get(scene),
            'cc_v2_fpm4': cc_fpm4,
            'cc_v2_ext': cc_ext,
            'cc_naive': NAIVE_AVG.get(scene),
        })

    def fmt(x):
        if x is None:
            return '  ---  '
        return f'{x:7.4f}'
    def fmts(a, b):
        if a is None or b is None:
            return '  ---  '
        d = a - b
        s = '+' if d >= 0 else ''
        return f'{s}{d:6.4f}'

    print()
    print('Data_v2 Stage 3 — extended vs F±4 vs naive-avg, 6-scene v5/M1 HO_8')
    print('=' * 116)
    hdr = (f'{"scene":<20s}  {"F":>4s} | {"v5 data/":>9s} | {"v5 data_v2 F±4":>16s} | '
           f'{"v5 data_v2 ext":>15s}  {"(Δ v2-ext - v1)":>17s} | '
           f'{"naive avg":>10s}  {"(Δ ext - naive)":>17s}')
    print(hdr)
    print('-' * 116)
    sums = {k: [0.0, 0] for k in ['v1', 'v2_fpm4', 'v2_ext', 'naive']}
    for r in rows:
        for k, key in [('v1', 'cc_v1'), ('v2_fpm4', 'cc_v2_fpm4'),
                        ('v2_ext', 'cc_v2_ext'), ('naive', 'cc_naive')]:
            v = r[key]
            if v is not None:
                sums[k][0] += v
                sums[k][1] += 1
        print(f'{r["scene"]:<20s}  {r["F"]:>4d} | '
              f'{fmt(r["cc_v1"]):>9s} | {fmt(r["cc_v2_fpm4"]):>16s} | '
              f'{fmt(r["cc_v2_ext"]):>15s}  {fmts(r["cc_v2_ext"], r["cc_v1"]):>17s} | '
              f'{fmt(r["cc_naive"]):>10s}  '
              f'{fmts(r["cc_v2_ext"], r["cc_naive"]):>17s}')

    print('-' * 116)
    def m(key):
        s, n = sums[key]
        return (s / n) if n > 0 else None

    print(f'{"6-scene mean":<20s}  ---- | '
          f'{fmt(m("v1")):>9s} | {fmt(m("v2_fpm4")):>16s} | '
          f'{fmt(m("v2_ext")):>15s}  {fmts(m("v2_ext"), m("v1")):>17s} | '
          f'{fmt(m("naive")):>10s}  {fmts(m("v2_ext"), m("naive")):>17s}')
    print('=' * 116)

    # G3.1 check: data_v2_fpm4 should be within MC noise (±0.03) of data_v1
    print()
    print('Gate G3.1 (bit-identity: v5 data_v2 F±4 within MC noise of data/ baseline):')
    for r in rows:
        if r['cc_v1'] is None or r['cc_v2_fpm4'] is None:
            continue
        d = r['cc_v2_fpm4'] - r['cc_v1']
        ok = abs(d) <= 0.03
        print(f'  {r["scene"]:<20s}  data/ = {r["cc_v1"]:.4f}  '
              f'data_v2 = {r["cc_v2_fpm4"]:.4f}  Δ = {d:+.4f}  '
              f'{"✓ PASS" if ok else "✗ FAIL"}')


if __name__ == '__main__':
    main()
