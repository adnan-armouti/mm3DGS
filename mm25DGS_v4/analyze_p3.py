"""Phase P3: parameter-parameter redundancy via gradient correlation.

Uses the per-iter gradient statistics (grad_mean, grad_std) from the
D_P1_simplified_random .npz dump. Computes:

  1. Pearson correlation between grad_mean[t, i] and grad_mean[t, j]
     across iterations for each pair (i, j). This is the sign-aligned
     gauge direction correlation.

  2. Same for grad_std[t, i] vs grad_std[t, j] (per-point signal
     magnitude correlation).

High |corr| between two parameters indicates they tend to move together
during training — evidence of redundancy. A sign-negative correlation
indicates they cancel each other (the l_c harmfulness hypothesis).
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm25DGS_v4.material_diagnostics import PARAM_NAMES

NPZ = '/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_P1_simplified_random/D_P1_simplified_random__seq_0_frame_135.npz'

d = np.load(NPZ, allow_pickle=True)
gm = d['grad_mean']   # (T, 6)  — gauge direction per iter
gs = d['grad_std']    # (T, 6)  — per-point signal per iter
print(f'T iterations: {gm.shape[0]}')
print(f'Trained cart_corr: {float(d["mean_cart_corr"]):.4f}')


def print_corr_matrix(mat, label):
    print(f"\n--- {label} ---")
    hdr = '        ' + ' '.join(f'{n:>10}' for n in PARAM_NAMES)
    print(hdr)
    for i, ni in enumerate(PARAM_NAMES):
        row = f'{ni:<8}'
        for j in range(6):
            v = mat[i, j]
            row += f'{v:>+10.3f}'
        print(row)


# Normalize each column to zero mean, unit variance
gm_std = (gm - gm.mean(axis=0, keepdims=True)) / (gm.std(axis=0, keepdims=True) + 1e-30)
gs_std = (gs - gs.mean(axis=0, keepdims=True)) / (gs.std(axis=0, keepdims=True) + 1e-30)

corr_mean = gm_std.T @ gm_std / gm.shape[0]   # (6, 6)
corr_std  = gs_std.T @ gs_std / gs.shape[0]    # (6, 6)

print_corr_matrix(corr_mean, "Correlation of grad_mean[:, k] across iters (gauge direction)")
print_corr_matrix(corr_std,  "Correlation of grad_std[:, k] across iters (per-point signal magnitude)")

# Also compute the DOT product between trajectory mean updates
traj = d['trajectory']  # (T, M, 6)
traj_mean = traj.mean(axis=1)  # (T, 6)
deltas = traj_mean[1:] - traj_mean[:-1]  # (T-1, 6)
deltas_std = (deltas - deltas.mean(axis=0, keepdims=True)) / (deltas.std(axis=0, keepdims=True) + 1e-30)
corr_traj = deltas_std.T @ deltas_std / deltas.shape[0]
print_corr_matrix(corr_traj, "Correlation of per-checkpoint Δ(trajectory_mean)")

print("\n=== Highlights (|corr| > 0.5) ===")
for mat_name, mat in [('grad_mean corr', corr_mean),
                      ('grad_std corr', corr_std),
                      ('traj_delta corr', corr_traj)]:
    print(f"\n{mat_name}:")
    for i in range(6):
        for j in range(i + 1, 6):
            if abs(mat[i, j]) > 0.5:
                print(f"  {PARAM_NAMES[i]:<10} ↔ {PARAM_NAMES[j]:<10}  corr = {mat[i, j]:+.3f}")
