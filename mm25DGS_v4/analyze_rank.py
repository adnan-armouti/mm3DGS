"""Real rank check: per-iter 6x6 gradient covariance across 50K points.

At each captured iter, we have a (50000, 6) per-point gradient tensor.
We compute:

  1. The 6x6 gradient covariance matrix (across points)
  2. Its eigenvalue spectrum (normalized to fraction of total variance)
  3. The effective rank (participation ratio + number of eigenvalues above
     threshold)
  4. The same on the normalized gradient matrix (per-column z-score),
     which gives correlation instead of covariance

This is the real rank check the LOO analysis should have run.

Participation ratio: PR = (sum(λ))² / sum(λ²). For a rank-1 spectrum,
PR=1. For a uniform spectrum with all eigenvalues equal, PR=K (full rank).
PR lies in [1, 6] and gives a continuous "effective rank" estimate.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm25DGS_v4.material_diagnostics import PARAM_NAMES

NPZ = '/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_rank_check/D_rank_check__seq_0_frame_135.npz'

d = np.load(NPZ, allow_pickle=True)
iters = d['grad_full_iters']
print(f"Loaded snapshots at iters: {list(iters)}")
print(f"Trained cart_corr: {float(d['mean_cart_corr']):.4f}")

# ----------------------------------------------------------------------
# For each snapshot, compute the real rank analysis
# ----------------------------------------------------------------------

def analyze_one(grad, label):
    """grad: (M, K) per-point gradient at one iter."""
    M, K = grad.shape
    print(f"\n{'='*70}")
    print(f"  {label}  (M={M}, K={K})")
    print(f"{'='*70}")

    # Per-column magnitudes
    col_mean = grad.mean(axis=0)
    col_std = grad.std(axis=0)
    col_l2 = np.linalg.norm(grad, axis=0)
    print(f"\n{'param':<12}{'mean':>14}{'std (per-point)':>18}{'||col||_2':>16}")
    for k, name in enumerate(PARAM_NAMES):
        print(f"  {name:<10}{col_mean[k]:>14.3e}{col_std[k]:>18.3e}{col_l2[k]:>16.3e}")

    # --- Covariance across points ---
    # Center columns (remove the gauge-invariant component)
    gc = grad - col_mean[None, :]
    cov = (gc.T @ gc) / M   # (K, K)

    # --- Correlation across points (unit-variance normalization) ---
    # Avoid divide-by-zero
    std_nz = np.where(col_std > 1e-30, col_std, 1.0)
    corr = cov / (std_nz[:, None] * std_nz[None, :])

    # Eigendecomposition of covariance
    eigs_cov = np.linalg.eigvalsh(cov)[::-1]  # descending
    eigs_corr = np.linalg.eigvalsh(corr)[::-1]

    # Participation ratios
    def pr(eig):
        eig = np.clip(eig, 0, None)
        s = eig.sum()
        if s < 1e-30:
            return 0.0
        return float(s * s / (eig * eig).sum())

    print(f"\n--- Covariance eigenspectrum (variance explained) ---")
    total = max(eigs_cov.sum(), 1e-30)
    cum = 0.0
    for i, e in enumerate(eigs_cov):
        frac = e / total
        cum += frac
        print(f"  λ_{i+1} = {e:.3e}   ({frac*100:5.1f}%  cum {cum*100:5.1f}%)")
    print(f"  effective rank (participation ratio) = {pr(eigs_cov):.3f}  / 6")

    print(f"\n--- Correlation eigenspectrum ---")
    # Correlation matrix has trace = K, eigenvalues sum to K
    cum = 0.0
    for i, e in enumerate(eigs_corr):
        frac = e / K
        cum += frac
        print(f"  λ_{i+1} = {e:.3f}   ({frac*100:5.1f}%  cum {cum*100:5.1f}%)")
    print(f"  effective rank (participation ratio) = {pr(eigs_corr):.3f}  / 6")

    # Correlation matrix (prettier to print directly)
    print(f"\n--- 6x6 correlation matrix (across {M} points) ---")
    hdr = '          ' + ' '.join(f'{n:>10}' for n in PARAM_NAMES)
    print(hdr)
    for i, ni in enumerate(PARAM_NAMES):
        row = f'{ni:<10}'
        for j in range(K):
            row += f'{corr[i, j]:>+10.3f}'
        print(row)

    return eigs_cov, eigs_corr


results = {}
for it in iters:
    key = f'grad_full_iter_{it}'
    if key in d.files:
        grad = d[key]   # (M, 6)
        results[int(it)] = analyze_one(grad, f'ITER {it}')

# Summary
print("\n" + "=" * 70)
print("  SUMMARY: effective rank per iter")
print("=" * 70)
print(f"{'iter':>8}  {'cov PR':>10}  {'corr PR':>10}  {'λ1_cov/Σ':>12}  {'λ1_corr/6':>12}")
for it, (cov_eigs, corr_eigs) in sorted(results.items()):
    def pr(eig):
        eig = np.clip(eig, 0, None)
        s = eig.sum()
        return float(s * s / (eig * eig).sum()) if s > 1e-30 else 0.0
    cov_pr = pr(cov_eigs)
    corr_pr = pr(corr_eigs)
    lam1_cov = cov_eigs[0] / max(cov_eigs.sum(), 1e-30)
    lam1_corr = corr_eigs[0] / 6
    print(f"{it:>8}  {cov_pr:>10.3f}  {corr_pr:>10.3f}  {lam1_cov*100:>11.1f}%  {lam1_corr*100:>11.1f}%")
