"""Aggregate v6 M2 (Doppler-extended v5 loss on |RAD|) results.

Compares M2 against prior baselines per scene + 6-scene mean:
  - v5/M1 on data/                 (prior reference)
  - v5/M1 on data_v2, F±4 window   (same-port G3.1-comparable)
  - v5 data_v2 extended window     (Stage-3 extended-data experiment)
  - naive ½(F-1+F+1) avg predictor (zero-model baseline)
  - v6 M2 on data_v2, F±4 window   (this)
"""
from __future__ import annotations
import glob
import json
import os

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]
OUTPUT_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'

V5_DATA_V1 = {
    'seq_0_frame_135': 0.5860,
    'seq_1_frame_185': 0.5595,
    'seq_1_frame_438': 0.5757,
    'seq_2_frame_105': 0.5490,
    'seq_2_frame_160': 0.6544,
    'seq_2_frame_300': 0.4762,
}
V5_DATA_V2_FPM4 = {
    'seq_0_frame_135': 0.4952,
    'seq_1_frame_185': 0.6342,
    'seq_1_frame_438': 0.5417,
    'seq_2_frame_105': 0.4555,
    'seq_2_frame_160': 0.6374,
    'seq_2_frame_300': 0.4602,
}
V5_DATA_V2_EXT = {
    'seq_0_frame_135': 0.4563,
    'seq_1_frame_185': 0.6282,
    'seq_1_frame_438': 0.4884,
    'seq_2_frame_105': 0.4345,
    'seq_2_frame_160': 0.6370,
    'seq_2_frame_300': 0.4420,
}
NAIVE_AVG = {
    'seq_0_frame_135': 0.6599,
    'seq_1_frame_185': 0.5863,
    'seq_1_frame_438': 0.3282,
    'seq_2_frame_105': 0.6068,
    'seq_2_frame_160': 0.7722,
    'seq_2_frame_300': 0.5949,
}


def _find_m2(scene: str, F: int) -> str | None:
    cand = glob.glob(
        f'{OUTPUT_ROOT}/{scene}_train8frames_16loops_test{F}_'
        f'loop0_pass2_N20000_v6M2'
    )
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
    rows = []
    for scene, F in SCENES:
        m2_dir = _find_m2(scene, F)
        cc_m2  = _load_cc(m2_dir)
        rows.append({
            'scene': scene, 'F': F,
            'cc_v1':       V5_DATA_V1.get(scene),
            'cc_v2_fpm4':  V5_DATA_V2_FPM4.get(scene),
            'cc_v2_ext':   V5_DATA_V2_EXT.get(scene),
            'cc_naive':    NAIVE_AVG.get(scene),
            'cc_m2':       cc_m2,
        })

    def fmt(x):
        if x is None: return '  ---  '
        return f'{x:7.4f}'

    def fmts(a, b):
        if a is None or b is None: return '   ---  '
        d = a - b
        s = '+' if d >= 0 else ''
        return f'{s}{d:6.4f}'

    def safe_mean(rows, key):
        v = [r[key] for r in rows if r[key] is not None]
        return (sum(v) / len(v)) if v else None

    print()
    print('v6 M2 — Doppler-extended v5 mse_raw on |RAD|, 6-scene HO_8')
    print('=' * 128)
    hdr = (f'{"scene":<20s} {"F":>4s} | '
           f'{"v5 data/":>9s} {"v5 v2 F±4":>10s} {"v5 v2 ext":>10s} '
           f'{"naive avg":>10s} | {"v6 M2":>9s}  '
           f'{"Δ M2-v1":>9s} {"Δ M2-v2fpm4":>12s} {"Δ M2-naive":>11s}')
    print(hdr)
    print('-' * 128)
    for r in rows:
        print(f'{r["scene"]:<20s} {r["F"]:>4d} | '
              f'{fmt(r["cc_v1"]):>9s} {fmt(r["cc_v2_fpm4"]):>10s} '
              f'{fmt(r["cc_v2_ext"]):>10s} {fmt(r["cc_naive"]):>10s} | '
              f'{fmt(r["cc_m2"]):>9s}  '
              f'{fmts(r["cc_m2"], r["cc_v1"]):>9s} '
              f'{fmts(r["cc_m2"], r["cc_v2_fpm4"]):>12s} '
              f'{fmts(r["cc_m2"], r["cc_naive"]):>11s}')
    print('-' * 128)
    m_v1    = safe_mean(rows, 'cc_v1')
    m_v2fp4 = safe_mean(rows, 'cc_v2_fpm4')
    m_v2ext = safe_mean(rows, 'cc_v2_ext')
    m_naive = safe_mean(rows, 'cc_naive')
    m_m2    = safe_mean(rows, 'cc_m2')
    print(f'{"6-scene mean":<20s} {"--":>4s} | '
          f'{fmt(m_v1):>9s} {fmt(m_v2fp4):>10s} '
          f'{fmt(m_v2ext):>10s} {fmt(m_naive):>10s} | '
          f'{fmt(m_m2):>9s}  '
          f'{fmts(m_m2, m_v1):>9s} '
          f'{fmts(m_m2, m_v2fp4):>12s} '
          f'{fmts(m_m2, m_naive):>11s}')
    print('=' * 128)


if __name__ == '__main__':
    main()
