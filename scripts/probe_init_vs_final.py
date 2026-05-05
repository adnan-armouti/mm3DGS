"""Phase A.6 — Were live points already clustered at INIT, or did they
migrate to clusters during training?

Critical for the 'is init the problem?' question. With the current
recipe (learn_positions_lr=1e-5, L2 anchor λ=100), positions can
drift at most ~5 mm/iter steady state, so 500 iters of training
caps drift well below 1 m. But the OBSERVED median d(live→live)
= 8 mm tells us the live points are clumped. Question: were they
clumped at iter 0 (FPS already over-concentrated) or did the 2k
top-Fisher slots happen to land on a few tight features?

Outputs per scene:
  d_init(live→live):    median NN distance among the live points
                         using their INITIAL positions
  d_final(live→live):   median NN distance using FINAL positions
  drift:                median ‖final − init‖ for live points
  drift_dead:           median ‖final − init‖ for dead points
"""
from __future__ import annotations

import os
import glob

import numpy as np
import torch


PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
OUT_DIR = os.path.join(PROJECT_ROOT, 'md', 'diagnostics_v5_v4_fisher')


def find_scene_diagnostic(scene: str) -> str | None:
    paths = []
    pattern = os.path.join(
        PROJECT_ROOT, 'mm25DGS_v5_v4', 'output_frame_nvs',
        scene + '_*_dnsfy*_dsigpos_grad_amp_lpos1e-05L2100',
    )
    for d in glob.glob(pattern):
        diag = os.path.join(d, 'fisher_diagnostic.pt')
        if os.path.exists(diag):
            paths.append((os.path.getmtime(diag), diag))
    if not paths:
        return None
    paths.sort(reverse=True)
    return paths[0][1]


def nearest_distance(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(query).float().cuda()
    t = torch.from_numpy(target).float().cuda()
    out = torch.empty(len(q), device='cuda')
    bs = 4096
    for i in range(0, len(q), bs):
        d = torch.cdist(q[i:i + bs], t)
        # exclude self at zero (set to inf)
        out[i:i + bs] = torch.where(d > 1e-6, d,
                                     torch.full_like(d, float('inf'))
                                    ).min(dim=1).values
    return out.cpu().numpy()


def main():
    scenes = ['seq_0_frame_135', 'seq_1_frame_185', 'seq_1_frame_438',
              'seq_2_frame_105', 'seq_2_frame_160', 'seq_2_frame_300']
    md = ['# Phase A.6 — Were live points already clustered at INIT?\n\n']
    md.append('| scene | d_init(live→live) m | d_final(live→live) m | '
              'drift_live m | drift_dead m | test cc |\n')
    md.append('|---|---:|---:|---:|---:|---:|\n')

    for s in scenes:
        diag_path = find_scene_diagnostic(s)
        if diag_path is None:
            print(f'[!] no diagnostic for {s}')
            continue
        d = torch.load(diag_path, map_location='cpu', weights_only=False)
        pos_init = d['init_positions'].numpy()
        pos_final = d['final_positions'].numpy()
        career = d['career_pos_grad'].numpy()
        N = len(pos_init)
        order = np.argsort(-career)
        live = order[:int(0.10 * N)]
        dead = order[int(0.10 * N):]

        live_init = pos_init[live]
        live_fin = pos_final[live]
        dead_init = pos_init[dead]
        dead_fin = pos_final[dead]

        d_init_ll = nearest_distance(live_init, live_init)
        d_fin_ll = nearest_distance(live_fin, live_fin)
        drift_live = np.linalg.norm(live_fin - live_init, axis=1)
        drift_dead = np.linalg.norm(dead_fin - dead_init, axis=1)

        row_d_init = float(np.median(d_init_ll))
        row_d_fin = float(np.median(d_fin_ll))
        row_drift_l = float(np.median(drift_live))
        row_drift_d = float(np.median(drift_dead))

        md.append(f'| {s} | {row_d_init:.3f} | {row_d_fin:.3f} | '
                  f'{row_drift_l:.4f} | {row_drift_d:.4f} | '
                  f'{d["final_test_cc"]:.4f} |\n')
        print(f'  [{s}] d_init(live→live)={row_d_init:.3f} m  '
              f'd_final(live→live)={row_d_fin:.3f} m  '
              f'drift_live(med)={row_drift_l*1000:.2f} mm  '
              f'drift_dead(med)={row_drift_d*1000:.2f} mm')

    md.append('\n## Interpretation\n')
    md.append('- **d_init ≈ d_final**: live points were already clustered '
              'at init. FPS over-concentrated them. → Init IS the problem.\n')
    md.append('- **d_init ≫ d_final**: live points migrated to clusters '
              'during training. → Init was OK; the issue is the optimizer '
              'collapsing onto hot spots.\n')
    md.append('- **drift_live ≈ drift_dead ≈ a few mm**: matches the '
              'L2 anchor analytical bound (5 mm). Positions barely move. '
              'So the configuration we observe is essentially the init '
              'configuration, with materials/normals adapted on top.\n')

    with open(os.path.join(OUT_DIR, 'init_vs_final.md'), 'w') as f:
        f.writelines(md)
    print(f'\n[done] wrote {OUT_DIR}/init_vs_final.md')


if __name__ == '__main__':
    main()
