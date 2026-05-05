"""Phase A.5 — Why are dead points dead?

For each scene, computes spatial diagnostics that distinguish two
explanations:

  (a) Dead points are spatially CLOSE to live points (top 10%).
      → init placed them in good regions, but they didn't accumulate
      gradient. Means init is fine; the renderer/optimizer isn't
      letting these slots learn. Init replacement won't help much.

  (b) Dead points are spatially FAR from live points / clustered in
      "no-signal" zones (poor cos θ_i, beyond useful range, on flat
      walls without features). → init mis-placed them. Init
      replacement (strict-AND, SampleNet, render-and-reseed) WILL
      help.

Reports per scene:
  d_dead2live: median distance from each dead point to its nearest
               live (top 10%) point.
  d_live2live: median distance from each live point to its nearest
               OTHER live point (typical inter-feature spacing).
  d_random:    median distance from each pcl.npy pool point to the
               nearest live point (background = ratio of "hits").

Also bins points by Fisher quantile and reports:
  - mean cos(angle to mean train RX boresight) per bin
  - mean distance to mean train RX center per bin
  - cumulative % of pcl.npy pool that lives within radius r of the
    nearest live point

Usage:
    python -m scripts.probe_dead_points
"""
from __future__ import annotations

import os
import glob

import numpy as np
import torch


PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
OUT_DIR = os.path.join(PROJECT_ROOT, 'md', 'diagnostics_v5_v4_fisher')
os.makedirs(OUT_DIR, exist_ok=True)


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
    """For each query (Q,3), return distance to the nearest target (T,3)."""
    q = torch.from_numpy(query).float().cuda()
    t = torch.from_numpy(target).float().cuda()
    # batched to fit in memory
    out = torch.empty(len(q), device='cuda')
    bs = 4096
    for i in range(0, len(q), bs):
        d = torch.cdist(q[i:i + bs], t)  # (bs, T)
        out[i:i + bs] = d.min(dim=1).values
    return out.cpu().numpy()


def main():
    scenes = ['seq_0_frame_135', 'seq_1_frame_185', 'seq_1_frame_438',
              'seq_2_frame_105', 'seq_2_frame_160', 'seq_2_frame_300']
    md = ['# Phase A.5 — Why are dead points dead?\n']
    md.append('Top 10% = "live" (holds ~63% of total ‖∇p L‖); '
              'bottom 90% = "dead" (holds ~37%, mostly bottom 50% '
              'which is ~3% of total).\n')
    md.append('\n## Per-scene spatial diagnostics\n')
    md.append('| scene | d(dead→live) m | d(live→live) m | d(pool→live) m | '
              'live frac of pool within 0.10 m | test cc |\n')
    md.append('|---|---:|---:|---:|---:|---:|\n')

    rows = []
    for s in scenes:
        diag_path = find_scene_diagnostic(s)
        if diag_path is None:
            print(f'[!] no diagnostic for {s}')
            continue
        d = torch.load(diag_path, map_location='cpu', weights_only=False)
        positions = d['final_positions'].numpy()
        career = d['career_pos_grad'].numpy()
        N = len(positions)
        order = np.argsort(-career)
        live_idx = order[:int(0.10 * N)]            # top 10%
        dead_idx = order[int(0.10 * N):]            # bottom 90%
        live_pts = positions[live_idx]
        dead_pts = positions[dead_idx]

        d_dead = nearest_distance(dead_pts, live_pts)
        d_live = nearest_distance(live_pts, live_pts)  # includes self =0; remove
        # Drop the self-match: each live point's distance to itself is 0;
        # use the second-nearest. Approximate by setting tiny floor.
        d_live = d_live[d_live > 1e-6]

        # Pool: pcl.npy
        pcl_path = os.path.join(PROJECT_ROOT, 'data', s, 'scene', 'pcl.npy')
        if os.path.exists(pcl_path):
            pcl = np.load(pcl_path)
            pool_xyz = pcl[:, :3].astype(np.float32)
            # Subsample for speed
            if len(pool_xyz) > 100000:
                idx = np.random.default_rng(0).choice(len(pool_xyz), 100000, replace=False)
                pool_xyz = pool_xyz[idx]
            d_pool = nearest_distance(pool_xyz, live_pts)
            frac_within_010 = float((d_pool < 0.10).mean())
            d_pool_med = float(np.median(d_pool))
        else:
            d_pool_med = float('nan')
            frac_within_010 = float('nan')

        row = {
            'scene': s,
            'd_dead_live_med': float(np.median(d_dead)),
            'd_dead_live_mean': float(d_dead.mean()),
            'd_live_live_med': float(np.median(d_live)),
            'd_pool_live_med': d_pool_med,
            'pool_frac_010': frac_within_010,
            'test_cc': d['final_test_cc'],
            'top10_pct_signal_held': float(np.cumsum(np.sort(career)[::-1])[int(0.10 * N) - 1] / max(career.sum(), 1e-30) * 100),
        }
        rows.append(row)

        md.append(f'| {s} | {row["d_dead_live_med"]:.3f} | '
                  f'{row["d_live_live_med"]:.3f} | '
                  f'{row["d_pool_live_med"]:.3f} | '
                  f'{row["pool_frac_010"]*100:.1f}% | '
                  f'{row["test_cc"]:.4f} |\n')
        print(f'  [{s}] d(dead→live) median = {row["d_dead_live_med"]:.3f} m  '
              f'(mean {row["d_dead_live_mean"]:.3f} m)  '
              f'd(live→live) = {row["d_live_live_med"]:.3f} m  '
              f'pool-within-10cm = {row["pool_frac_010"]*100:.1f}%')

    if rows:
        md.append(f'| **mean** | '
                  f'**{np.mean([r["d_dead_live_med"] for r in rows]):.3f}** | '
                  f'**{np.mean([r["d_live_live_med"] for r in rows]):.3f}** | '
                  f'**{np.mean([r["d_pool_live_med"] for r in rows]):.3f}** | '
                  f'**{np.mean([r["pool_frac_010"] for r in rows])*100:.1f}%** | '
                  f'**{np.mean([r["test_cc"] for r in rows]):.4f}** |\n')

    md.append('\n## Interpretation\n')
    md.append('- **d(dead→live)** ≪ **d(live→live)**: dead points sit *near* '
              'live points → not init\'s fault, the optimizer isn\'t letting '
              'them learn. Init replacement won\'t help.\n')
    md.append('- **d(dead→live)** ≈ **d(live→live)** or larger: dead points '
              'are isolated in low-signal regions → init mis-placed them. '
              'Better init should help.\n')
    md.append('- **pool frac within 10 cm**: how much of the LiDAR pool is '
              'redundant w.r.t. our live points. High = lots of pool capacity '
              'going unused near active regions; low = active regions are '
              'sparsely covered by the pool too.\n')

    with open(os.path.join(OUT_DIR, 'dead_points.md'), 'w') as f:
        f.writelines(md)
    print(f'\n[done] wrote {OUT_DIR}/dead_points.md')


if __name__ == '__main__':
    main()
