"""Tabulate v7 Doppler bench results with loss_norm ∈ {max, mean}.

Reports both |RA| (chirp-0 cart_corr) and |RAD| (flattened Pearson)
train + test CC per scene, plus 6-scene means.
"""
from __future__ import annotations
import json
import os

V7_ROOT = '/home/adnan/Desktop/mm3DGS/mm25DGS_v7/output_frame_nvs'

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]


def _dir(scene, F, norm):
    return (f'{V7_ROOT}/{scene}_train8frames_16loops_test{F}'
            f'_loop0_pass2_N20000_v7doppler_norm{norm}')


def _load(scene, F, norm):
    d = _dir(scene, F, norm)
    path = os.path.join(d, 'results.json')
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def _fmt(x):
    return '  ---   ' if x is None else f'{x:7.4f} '


def main():
    rows = []
    for scene, F in SCENES:
        r_max  = _load(scene, F, 'max')
        r_mean = _load(scene, F, 'mean')
        rows.append({'scene': scene, 'F': F,
                      'max':  r_max, 'mean': r_mean})

    def pick(r, key):
        return (None if r is None else r.get(key))

    def avg(rs, key):
        vs = [pick(r, key) for r in rs if r is not None]
        vs = [v for v in vs if v is not None]
        return (sum(vs) / len(vs)) if vs else None

    header1 = (f'{"":22s} {"":>4s} | '
               f'{"|RA| train":^19s}   {"|RA| test":^19s}   '
               f'{"|RAD| train":^19s}   {"|RAD| test":^19s}')
    header2 = (f'{"scene":<22s} {"F":>4s} | '
               f'{"max":>8s} {"mean":>8s}   '
               f'{"max":>8s} {"mean":>8s}   '
               f'{"max":>8s} {"mean":>8s}   '
               f'{"max":>8s} {"mean":>8s}')
    width = len(header2)
    print()
    print('v7 Doppler — loss_norm pre-C1 (max) vs post-C1 (mean)')
    print('=' * width)
    print(header1)
    print(header2)
    print('-' * width)
    for r in rows:
        max_d, mean_d = r['max'], r['mean']
        ra_train_max  = pick(max_d,  'final_train_mean_cc')
        ra_train_mean = pick(mean_d, 'final_train_mean_cc')
        ra_test_max   = pick(max_d,  'final_test_cc')
        ra_test_mean  = pick(mean_d, 'final_test_cc')
        rad_train_max  = pick(max_d,  'final_train_rad_cc_mean')
        rad_train_mean = pick(mean_d, 'final_train_rad_cc_mean')
        rad_test_max   = pick(max_d,  'final_test_rad_cc')
        rad_test_mean  = pick(mean_d, 'final_test_rad_cc')
        print(f'{r["scene"]:<22s} {r["F"]:>4d} | '
              f'{_fmt(ra_train_max):>8s}{_fmt(ra_train_mean):>8s}  '
              f'{_fmt(ra_test_max):>8s}{_fmt(ra_test_mean):>8s}  '
              f'{_fmt(rad_train_max):>8s}{_fmt(rad_train_mean):>8s}  '
              f'{_fmt(rad_test_max):>8s}{_fmt(rad_test_mean):>8s}')
    print('-' * width)
    avgs_max  = [r['max']  for r in rows]
    avgs_mean = [r['mean'] for r in rows]
    print(f'{"6-scene mean":<22s} {"":>4s} | '
          f'{_fmt(avg(avgs_max, "final_train_mean_cc")):>8s}'
          f'{_fmt(avg(avgs_mean, "final_train_mean_cc")):>8s}  '
          f'{_fmt(avg(avgs_max, "final_test_cc")):>8s}'
          f'{_fmt(avg(avgs_mean, "final_test_cc")):>8s}  '
          f'{_fmt(avg(avgs_max, "final_train_rad_cc_mean")):>8s}'
          f'{_fmt(avg(avgs_mean, "final_train_rad_cc_mean")):>8s}  '
          f'{_fmt(avg(avgs_max, "final_test_rad_cc")):>8s}'
          f'{_fmt(avg(avgs_mean, "final_test_rad_cc")):>8s}')
    print('=' * width)

    # Δ table — mean − max for quick impact read.
    print()
    print(f'Δ (mean − max)  —  positive = C1 helps')
    print('-' * 80)
    dhdr = (f'{"scene":<22s} {"F":>4s} | '
            f'{"Δ|RA| train":>12s} {"Δ|RA| test":>12s} '
            f'{"Δ|RAD| train":>13s} {"Δ|RAD| test":>13s}')
    print(dhdr)
    print('-' * len(dhdr))

    def dfmt(x):
        if x is None: return '   ---  '
        sign = '+' if x >= 0 else ''
        return f'{sign}{x:7.4f}'

    for r in rows:
        ma, me = r['max'], r['mean']
        if ma is None or me is None:
            continue
        d_ra_t  = pick(me,'final_train_mean_cc')  - pick(ma,'final_train_mean_cc')
        d_ra_T  = pick(me,'final_test_cc')        - pick(ma,'final_test_cc')
        d_rad_t = pick(me,'final_train_rad_cc_mean') - pick(ma,'final_train_rad_cc_mean')
        d_rad_T = pick(me,'final_test_rad_cc')    - pick(ma,'final_test_rad_cc')
        print(f'{r["scene"]:<22s} {r["F"]:>4d} | '
              f'{dfmt(d_ra_t):>12s} {dfmt(d_ra_T):>12s} '
              f'{dfmt(d_rad_t):>13s} {dfmt(d_rad_T):>13s}')


if __name__ == '__main__':
    main()
