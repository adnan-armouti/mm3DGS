"""Q2 summary: test-frame-position sweep.

For each of 6 scenes, 5 test-frame positions (indices 2..6 in the
9-frame window) × v5/M1 loss → mean test cc per position, averaged
over the 6 scenes.

Indices map to offsets from F_center: 2→F-2, 3→F-1, 4→F, 5→F+1, 6→F+2.
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_277', 277),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]
OFFSETS = [-2, -1, 0, 1, 2]  # idx 2, 3, 4, 5, 6
OUTPUT_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v6/output_frame_nvs'


def _load(scene: str, F_test: int) -> float | None:
    cand = glob.glob(f'{OUTPUT_ROOT}/{scene}_*_test{F_test}_*_v6M1')
    cand = [c for c in cand if '_v6M1_5' not in c
                               and '_nopngs_archived' not in c]
    if not cand:
        return None
    try:
        r = json.load(open(os.path.join(cand[0], 'results.json')))
        return float(r['final_test_cc'])
    except Exception:
        return None


def main():
    # rows[scene] = {offset: cc}
    rows = {}
    for scene, Fc in SCENES:
        rows[scene] = {}
        for off in OFFSETS:
            F_test = Fc + off
            rows[scene][off] = _load(scene, F_test)

    print()
    print('Q2 — test-frame-position sweep (v5/M1 loss, 500 iters, 6 scenes)')
    print('=' * 110)
    hdr = f'{"scene":<22s}  F_center | ' + ' '.join(
        f'idx{2+off+2}={" F"+("+"+str(off)) if off>=0 else ("-"+str(-off))}'.replace(" F+-","F-").replace(" F++","F+").ljust(10)
        for off in OFFSETS)
    # simpler header
    hdr = f'{"scene":<22s}  F  |  ' + '  '.join(
        [f'F{("+"+str(off) if off>0 else str(off) if off<0 else "")}' for off in OFFSETS]
    ).ljust(55) + ' | mean'
    # Even simpler and accurate:
    hdr = f'{"scene":<22s}  F   | ' + '  '.join(
        f'F{"+"+str(off):>3s}' if off > 0 else f'F{str(off):>3s}' if off < 0 else f' F   '
        for off in OFFSETS) + '  | scene-mean'
    print(hdr)
    print('-' * 110)

    per_offset_means = {off: [] for off in OFFSETS}
    per_scene_means = []
    for scene, Fc in SCENES:
        cells = []
        vals = []
        for off in OFFSETS:
            v = rows[scene][off]
            if v is None:
                cells.append('   ???  ')
            else:
                cells.append(f'{v:7.4f}')
                vals.append(v)
                per_offset_means[off].append(v)
        scene_mean = sum(vals) / len(vals) if vals else float('nan')
        per_scene_means.append(scene_mean)
        print(f'{scene:<22s}  {Fc:<4d} | ' + '  '.join(cells)
              + f'  | {scene_mean:7.4f}')

    print('-' * 110)
    om_line = f'{"6-scene mean":<22s}  ---  | '
    for off in OFFSETS:
        vals = per_offset_means[off]
        m = sum(vals) / len(vals) if vals else float('nan')
        om_line += f'{m:7.4f}  '
    total_mean = sum(per_scene_means) / len(per_scene_means)
    om_line += f' | {total_mean:7.4f}'
    print(om_line)
    print('=' * 110)


if __name__ == '__main__':
    main()
