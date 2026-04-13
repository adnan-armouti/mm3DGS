"""Compare gradient diagnostics between LR=0.7 (baseline) and LR=0.01.

Answers the question: does lowering the material LR actually stop the
per-point gradient cancellation that we hypothesized, or is the +0.0041
cart_corr gain coming from something else?

Reads both .npz dumps, computes per-iter column norms and mean drift,
and prints a side-by-side comparison per parameter.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm25DGS_v4.material_diagnostics import PARAM_NAMES

NPZ_LR07 = '/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_rank_check/D_rank_check__seq_0_frame_135.npz'
NPZ_LR01 = '/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_rank_check_lr001/D_rank_check_lr001__seq_0_frame_135.npz'


def load(path):
    return np.load(path, allow_pickle=True)


def analyze(d, label):
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")
    init = d['init']            # (M, 6)
    final = d['final']          # (M, 6)
    traj = d['trajectory']      # (T, M, 6)
    drift = d['drift']          # (6,)
    fisher = d['fisher']        # (6,)

    print(f"\nTrained cart_corr: {float(d['mean_cart_corr']):.4f}")

    # Trajectory mean drift (population mean from iter 0 to iter 499)
    mean0 = traj[0].mean(axis=0)
    mean_end = traj[-1].mean(axis=0)
    std0 = traj[0].std(axis=0)
    std_end = traj[-1].std(axis=0)

    print(f"\n--- Trajectory mean drift per parameter (iter 0 → iter 499, raw space) ---")
    print(f"{'param':<12}{'mean(0)':>12}{'mean(end)':>12}{'Δmean':>12}{'std(0)':>12}{'std(end)':>12}{'drift L2':>12}")
    for k, name in enumerate(PARAM_NAMES):
        print(f"  {name:<10}"
              f"{mean0[k]:>+12.4f}{mean_end[k]:>+12.4f}{(mean_end[k]-mean0[k]):>+12.4f}"
              f"{std0[k]:>12.4e}{std_end[k]:>12.4e}{drift[k]:>12.4f}")

    # Per-iter gradient stats (from capture_grad_stats)
    if 'grad_mean' in d.files:
        gm = d['grad_mean']  # (T_iter, 6)
        gs = d['grad_std']   # (T_iter, 6)
        T = len(gm)
        print(f"\n--- Per-iter gradient stats (averaged over {T} iters) ---")
        print(f"{'param':<12}{'<|mean|>':>16}{'<std>':>16}{'signal/noise':>16}")
        for k, name in enumerate(PARAM_NAMES):
            mmean = np.mean(np.abs(gm[:, k]))
            mstd = np.mean(gs[:, k])
            # Proxy SNR: |mean across points| / std across points, averaged per iter
            snr_per_iter = np.abs(gm[:, k]) / (gs[:, k] + 1e-30)
            mean_snr = np.mean(snr_per_iter)
            print(f"  {name:<10}{mmean:>16.4e}{mstd:>16.4e}{mean_snr:>16.4e}")

    # Full gradient snapshots: column norms per snapshot
    if 'grad_full_iters' in d.files:
        iters = d['grad_full_iters']
        print(f"\n--- Per-snapshot column norm ‖grad[:, k]‖_2 ---")
        print(f"{'param':<12}", end='')
        for it in iters:
            print(f"{'iter ' + str(int(it)):>12}", end='')
        print()
        col_norms = {}
        for k, name in enumerate(PARAM_NAMES):
            norms = []
            for it in iters:
                g = d[f'grad_full_iter_{it}']
                norms.append(float(np.linalg.norm(g[:, k])))
            col_norms[name] = norms
            print(f"  {name:<10}", end='')
            for n in norms:
                print(f"{n:>12.3e}", end='')
            print()
        return col_norms, iters
    return None, None


# ---------------------------------------------------------------------
# Load both
# ---------------------------------------------------------------------

d_lr07 = load(NPZ_LR07)
d_lr01 = load(NPZ_LR01)

col_07, iters = analyze(d_lr07, 'LR=0.7  (baseline)')
col_01, _     = analyze(d_lr01, 'LR=0.01 (Option A)')


# ---------------------------------------------------------------------
# Side-by-side: column-norm ratio per iter
# ---------------------------------------------------------------------

print(f"\n{'='*78}")
print(f"  SIDE-BY-SIDE: column norm ratio (LR=0.01) / (LR=0.7) per iter")
print(f"{'='*78}")
print(f"\n{'param':<12}", end='')
for it in iters:
    print(f"{'iter ' + str(int(it)):>12}", end='')
print()
for name in PARAM_NAMES:
    print(f"  {name:<10}", end='')
    for n07, n01 in zip(col_07[name], col_01[name]):
        ratio = n01 / max(n07, 1e-30)
        print(f"{ratio:>12.3f}", end='')
    print()

print(f"\nRatio > 1 means LR=0.01 has LARGER column norm at that iter.")
print(f"Ratio < 1 means LR=0.01 has SMALLER column norm at that iter.")

# ---------------------------------------------------------------------
# Sign consistency: fraction of points with positive gradient
# ---------------------------------------------------------------------

print(f"\n{'='*78}")
print(f"  SIGN CONSISTENCY: |2·frac(g>0) - 1|  (closer to 1 = more consistent)")
print(f"{'='*78}")

def sign_consistency(g_col):
    """Returns 0 (random walk) to 1 (all same sign)."""
    pos_frac = (g_col > 0).mean()
    return abs(2.0 * pos_frac - 1.0)

print(f"\n--- LR=0.7 ---")
print(f"{'param':<12}", end='')
for it in iters:
    print(f"{'iter ' + str(int(it)):>12}", end='')
print()
for k, name in enumerate(PARAM_NAMES):
    print(f"  {name:<10}", end='')
    for it in iters:
        g = d_lr07[f'grad_full_iter_{it}'][:, k]
        sc = sign_consistency(g)
        print(f"{sc:>12.3f}", end='')
    print()

print(f"\n--- LR=0.01 ---")
print(f"{'param':<12}", end='')
for it in iters:
    print(f"{'iter ' + str(int(it)):>12}", end='')
print()
for k, name in enumerate(PARAM_NAMES):
    print(f"  {name:<10}", end='')
    for it in iters:
        g = d_lr01[f'grad_full_iter_{it}'][:, k]
        sc = sign_consistency(g)
        print(f"{sc:>12.3f}", end='')
    print()
