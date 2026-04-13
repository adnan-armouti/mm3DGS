"""Phase P1: per-parameter trajectory + gradient audit.

Reads the D_P1_simplified_random .npz and computes the six diagnostics
per parameter column:

  1. Trajectory mean per iter  (gauge-invariant drift)
  2. Trajectory std per iter   (per-point diversity)
  3. Grad |mean| per iter      (global update direction)
  4. Grad std per iter          (per-point signal)
  5. Grad sign split per iter  (+: frac grad>0, -: frac grad<0, 0: frac ≈0)
  6. Clamp hit rate at end      (fraction of points at reparam clamp bound)

Also compares against the existing D_mse_raw (concrete init) where possible.
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm25DGS_v4.material_diagnostics import PARAM_NAMES

D_SIMPLIFIED = '/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_P1_simplified_random/D_P1_simplified_random__seq_0_frame_135.npz'
D_CONCRETE = '/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_mse_raw/D_mse_raw__seq_0_frame_135.npz'

# Reparam clamp ranges (in raw space)
CLAMP_RANGES = {
    'eps_real':  (-5.0, 5.0),    # effectively saturates (sigmoid)
    'eps_imag':  (-7.0, 16.0),
    'sigma_h':   (-16.0, -7.0),
    'l_c':       (-7.6, -2.3),
    'tau_base':  (-5.0, 5.0),    # effectively saturates (sigmoid)
    'thickness': (-7.0, -1.2),
}
CLAMP_MARGIN = 0.01  # distance-from-bound counted as "hit"


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def analyze(dump, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    init = dump['init']          # (M, 6)
    final = dump['final']        # (M, 6)
    traj = dump['trajectory']    # (T, M, 6)
    drift = dump['drift']
    fisher = dump['fisher']

    M = init.shape[0]

    print(f"\nM = {M:,}, T checkpoints = {traj.shape[0]}")
    print(f"Trained cart_corr = {float(dump['mean_cart_corr']):.4f}\n")

    print(f"{'param':<12}{'drift':>10}{'init_std':>12}{'final_std':>12}"
          f"{'init_mean':>12}{'final_mean':>12}{'clamp_hit%':>12}{'fisher':>14}")
    for k, name in enumerate(PARAM_NAMES):
        lo, hi = CLAMP_RANGES[name]
        at_lo = (final[:, k] < lo + CLAMP_MARGIN).mean()
        at_hi = (final[:, k] > hi - CLAMP_MARGIN).mean()
        clamp_hit = (at_lo + at_hi) * 100.0
        print(f"  {name:<10}"
              f"{drift[k]:>10.3f}"
              f"{init[:, k].std():>12.4f}"
              f"{final[:, k].std():>12.4f}"
              f"{init[:, k].mean():>12.4f}"
              f"{final[:, k].mean():>12.4f}"
              f"{clamp_hit:>11.1f}%"
              f"{fisher[k]:>14.3e}")

    # Trajectory evolution: mean and std at first, mid, last checkpoint
    T = traj.shape[0]
    print(f"\n--- Trajectory mean/std at t=0, t={T//2}, t={T-1} ---")
    print(f"{'param':<12}{'mean(0)':>12}{'mean(mid)':>12}{'mean(end)':>12}"
          f"{'std(0)':>12}{'std(mid)':>12}{'std(end)':>12}")
    for k, name in enumerate(PARAM_NAMES):
        m0 = traj[0, :, k].mean()
        mm = traj[T // 2, :, k].mean()
        me = traj[-1, :, k].mean()
        s0 = traj[0, :, k].std()
        sm = traj[T // 2, :, k].std()
        se = traj[-1, :, k].std()
        print(f"  {name:<10}{m0:>+12.3f}{mm:>+12.3f}{me:>+12.3f}"
              f"{s0:>12.4f}{sm:>12.4f}{se:>12.4f}")

    if 'grad_mean' in dump:
        gm = dump['grad_mean']
        gs = dump['grad_std']
        print(f"\n--- Per-iter gradient (averaged across {len(gm)} iters) ---")
        print(f"{'param':<12}{'<|grad.mean|>':>16}{'<grad.std>':>14}"
              f"{'std/|mean|':>12}")
        for k, name in enumerate(PARAM_NAMES):
            m = np.mean(np.abs(gm[:, k]))
            s = np.mean(gs[:, k])
            ratio = s / max(m, 1e-30)
            print(f"  {name:<10}{m:>16.3e}{s:>14.3e}{ratio:>12.1f}")

    return dump


d_simp = load(D_SIMPLIFIED)
analyze(d_simp, 'SIMPLIFIED BSDF + RANDOM INIT (mse_raw, scene 135)')

if os.path.exists(D_CONCRETE):
    d_conc = load(D_CONCRETE)
    analyze(d_conc, 'FULL BSDF + CONCRETE INIT (mse_raw, scene 135, for reference)')
