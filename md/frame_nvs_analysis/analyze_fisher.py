"""Per-point Fisher concentration analysis."""
import os, numpy as np

OUT_DIR = '/home/adnan/Desktop/mm3DGS/md/frame_nvs_analysis'
SCENES = ['seq_1_frame_438', 'seq_2_frame_105']
VARIANTS = ['HO_128', 'HO_8', 'UB_144', 'UB_9']
PNAMES = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thick']

for scene in SCENES:
    print(f'\n=== {scene} ===')
    for v in VARIANTS:
        f = np.load(os.path.join(OUT_DIR, f'G_fisher_{scene}_{v}.npz'))
        grad_raw = f['grad_raw']  # (N, 6)
        per_pt = (grad_raw ** 2).sum(axis=1)  # sum over cols: per-point squared grad
        # Sort descending
        srt = np.sort(per_pt)[::-1]
        total = per_pt.sum()
        print(f'  {v:<8}  N={len(per_pt)}  sum_per_pt={total:.3e}  '
              f'max={per_pt.max():.3e}  mean={per_pt.mean():.3e}  '
              f'median={np.median(per_pt):.3e}')
        # Concentration: what fraction of total fisher comes from top-K percent of points?
        for frac in [0.01, 0.05, 0.10, 0.25, 0.50]:
            k = int(frac * len(per_pt))
            cum = srt[:k].sum()
            print(f'    top-{int(frac*100):3d}% ({k} pts) carry {100*cum/total:5.1f}% of total per-pt Fisher')
        # Per-column contribution (matches earlier mean |grad|, but summed)
        col_sum = (grad_raw ** 2).sum(axis=0)
        print(f'    col^2 sum: ' + '  '.join(f'{p}={v:.3e}' for p, v in zip(PNAMES, col_sum)))
