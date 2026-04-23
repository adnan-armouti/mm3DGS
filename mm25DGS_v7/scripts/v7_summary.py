"""Aggregate v7 Doppler bench results.

Columns per scene + 6-scene mean:
  - v5/M1 on data/                 (canonical baseline)
  - v6 M2 on data/                 (Option A — 16 physical renders)
  - v7 Doppler on data/            (Option B1 — analytic Doppler phase)
  - naive ½(F-1 + F+1) avg         (zero-model)
  - final_test_cc_RAD              (v7's training objective summed
                                      over Doppler axis, if logged)
"""
from __future__ import annotations
import glob
import json
import os

OUTPUT_V5 = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'          # v5/M1 runs live here
OUTPUT_V6 = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'          # v6 M2 also here
OUTPUT_V7 = '/home/adnan/Desktop/mm3DGS/mm25DGS_v7/output_frame_nvs'

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]

# Previously recorded baselines (see prior summary scripts)
V5_DATA = {
    'seq_0_frame_135': 0.5860,
    'seq_1_frame_185': 0.5595,
    'seq_1_frame_438': 0.5757,
    'seq_2_frame_105': 0.5490,
    'seq_2_frame_160': 0.6544,
    'seq_2_frame_300': 0.4762,
}
V6_M2 = {
    'seq_0_frame_135': 0.5152,
    'seq_1_frame_185': 0.5308,
    'seq_1_frame_438': 0.4258,
    'seq_2_frame_105': 0.4974,
    'seq_2_frame_160': 0.6063,
    'seq_2_frame_300': 0.4051,
}
NAIVE_AVG = {
    'seq_0_frame_135': 0.6599,
    'seq_1_frame_185': 0.5863,
    'seq_1_frame_438': 0.3282,
    'seq_2_frame_105': 0.6068,
    'seq_2_frame_160': 0.7722,
    'seq_2_frame_300': 0.5949,
}


def _find_v7(scene: str, F: int) -> str | None:
    cand = glob.glob(
        f'{OUTPUT_V7}/{scene}_train8frames_16loops_test{F}_'
        f'loop0_pass2_N20000_v7doppler'
    )
    return cand[0] if cand else None


def _load_cc(d: str | None, key='final_test_cc') -> float | None:
    if d is None:
        return None
    try:
        return float(json.load(open(os.path.join(d, 'results.json')))[key])
    except Exception:
        return None


def main():
    rows = []
    for scene, F in SCENES:
        d_v7 = _find_v7(scene, F)
        rows.append({
            'scene': scene, 'F': F,
            'cc_v5':      V5_DATA.get(scene),
            'cc_v6m2':    V6_M2.get(scene),
            'cc_v7':      _load_cc(d_v7),
            'cc_naive':   NAIVE_AVG.get(scene),
        })

    def fmt(x):
        return '  ---  ' if x is None else f'{x:7.4f}'

    def fmts(a, b):
        if a is None or b is None: return '   ---  '
        d = a - b
        s = '+' if d >= 0 else ''
        return f'{s}{d:6.4f}'

    def mean(rows, key):
        v = [r[key] for r in rows if r[key] is not None]
        return (sum(v) / len(v)) if v else None

    print()
    print('v7 Doppler bench — 6-scene HO_8')
    print('=' * 126)
    hdr = (f'{"scene":<20s} {"F":>4s} | '
           f'{"v5/M1":>9s}   {"v6 M2":>8s}   {"v7 Dop":>9s}   {"naive avg":>10s}'
           f' | {"Δ v7-v5":>8s}  {"Δ v7-v6m2":>10s}  {"Δ v7-naive":>11s}')
    print(hdr)
    print('-' * 126)
    for r in rows:
        print(f'{r["scene"]:<20s} {r["F"]:>4d} | '
              f'{fmt(r["cc_v5"]):>9s}   {fmt(r["cc_v6m2"]):>8s}   '
              f'{fmt(r["cc_v7"]):>9s}   {fmt(r["cc_naive"]):>10s} |'
              f'  {fmts(r["cc_v7"], r["cc_v5"]):>8s}  '
              f'{fmts(r["cc_v7"], r["cc_v6m2"]):>10s}  '
              f'{fmts(r["cc_v7"], r["cc_naive"]):>11s}')
    print('-' * 126)
    m_v5    = mean(rows, 'cc_v5')
    m_v6m2  = mean(rows, 'cc_v6m2')
    m_v7    = mean(rows, 'cc_v7')
    m_naive = mean(rows, 'cc_naive')
    print(f'{"6-scene mean":<20s} {"--":>4s} | '
          f'{fmt(m_v5):>9s}   {fmt(m_v6m2):>8s}   '
          f'{fmt(m_v7):>9s}   {fmt(m_naive):>10s} |'
          f'  {fmts(m_v7, m_v5):>8s}  '
          f'{fmts(m_v7, m_v6m2):>10s}  '
          f'{fmts(m_v7, m_naive):>11s}')
    print('=' * 126)


if __name__ == '__main__':
    main()
