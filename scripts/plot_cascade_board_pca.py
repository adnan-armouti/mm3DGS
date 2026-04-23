"""Visualise the cascaded radar board's TX + RX antenna layout in its
own PCA-best-fit plane, with labels in config order.

Use: verify TDM firing-order assumption against TI documentation by
visually identifying TX positions on the board.

Output: PNG at
    /home/adnan/Desktop/mm3DGS/output/cascade_board_pca_<scene>.png
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np


def load_positions(cfg_path: str):
    cfg = json.load(open(cfg_path))
    tx = [np.asarray(e['pos_mm'], dtype=np.float64) for e in cfg['tx_array']]
    rx = [np.asarray(e['pos_mm'], dtype=np.float64) for e in cfg['rx_array']]
    bore = np.asarray(cfg['tx_array'][0]['boresight'], dtype=np.float64)
    tx_names = [e['name'] for e in cfg['tx_array']]
    rx_names = [e['name'] for e in cfg['rx_array']]
    return np.vstack(tx), np.vstack(rx), tx_names, rx_names, bore


def pca_plane(positions: np.ndarray):
    """Return (centre, V_top2, V_normal) where V_top2 is (3, 2) with
    the columns being the top-two principal axes of the positions,
    and V_normal is the orthogonal (out-of-plane) axis."""
    centre = positions.mean(axis=0)
    X = positions - centre
    # SVD: X = U · Σ · V^T ; right-singular vectors V rows are principal axes
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T                                                      # (3, 3)
    V_top2 = V[:, :2]                                              # first two PCs
    V_normal = V[:, 2]
    return centre, V_top2, V_normal, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='seq_0_frame_135')
    ap.add_argument('--frame', type=int, default=135)
    ap.add_argument('--data-root', default='/home/adnan/Desktop/mm3DGS/data')
    ap.add_argument('--out-dir', default='/home/adnan/Desktop/mm3DGS/output')
    ap.add_argument('--use-pass2', action='store_true',
                     help='Use pass-2 aligned config instead of raw config')
    args = ap.parse_args()

    if args.use_pass2:
        cfg_path = os.path.join(
            args.data_root, 'alignment_data', args.scene, 'cascade',
            f'cascaded_frame_{args.frame}_aligned_pass2.json')
        cfg_label = 'pass-2 aligned'
    else:
        cfg_path = os.path.join(
            args.data_root, args.scene, 'configs',
            f'cascaded_frame_{args.frame}.json')
        cfg_label = 'raw config'

    tx_pos, rx_pos, tx_names, rx_names, bore = load_positions(cfg_path)
    all_pos = np.vstack([tx_pos, rx_pos])                         # (28, 3)

    centre, V, V_n, S = pca_plane(all_pos)
    print(f'centre (mm): [{centre[0]:.1f}, {centre[1]:.1f}, {centre[2]:.1f}]')
    print(f'principal singular values (mm): {S}')
    print(f'board plane normal vs boresight dot-product: '
          f'{np.dot(V_n, bore):.4f}  (close to 1 or -1 = board normal '
          f'parallel to boresight, as expected)')

    # Project into plane
    tx_centered = tx_pos - centre
    rx_centered = rx_pos - centre
    tx_2d = tx_centered @ V                                        # (12, 2) mm
    rx_2d = rx_centered @ V                                        # (16, 2) mm

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # Scatter + labels
    ax.scatter(tx_2d[:, 0], tx_2d[:, 1], s=120, c='tab:red',
                 marker='s', edgecolor='black', linewidth=1, zorder=3,
                 label='TX (config order)')
    ax.scatter(rx_2d[:, 0], rx_2d[:, 1], s=100, c='tab:blue',
                 marker='o', edgecolor='black', linewidth=1, zorder=3,
                 label='RX (config order)')

    for i, (x, y) in enumerate(tx_2d):
        ax.annotate(tx_names[i], (x, y), xytext=(7, 7),
                     textcoords='offset points', fontsize=9,
                     fontweight='bold', color='darkred', zorder=4)
    for i, (x, y) in enumerate(rx_2d):
        ax.annotate(rx_names[i], (x, y), xytext=(7, -12),
                     textcoords='offset points', fontsize=8,
                     color='darkblue', zorder=4)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.set_xlabel(f'PC1 (mm)   —   σ = {S[0]:.1f} mm')
    ax.set_ylabel(f'PC2 (mm)   —   σ = {S[1]:.1f} mm')
    ax.set_title(
        f'Cascaded radar board — TX/RX layout in PCA best-fit plane\n'
        f'{args.scene}  F={args.frame}  ({cfg_label})'
    )
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9)

    # Add orientation indicator: show the sign of PC1 vs boresight projection
    bore_in_plane = np.array([np.dot(bore, V[:, 0]),
                               np.dot(bore, V[:, 1])])
    if np.linalg.norm(bore_in_plane) > 0:
        scale = max(abs(tx_2d).max(), abs(rx_2d).max()) * 0.4
        b_x, b_y = bore_in_plane / (np.linalg.norm(bore_in_plane) + 1e-9) * scale
        ax.annotate('', xy=(b_x, b_y), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='green',
                                      lw=2, alpha=0.7))
        ax.text(b_x, b_y, '  boresight\n  (in-plane\n  component)',
                 color='green', fontsize=9, va='center')

    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'cascade_board_pca_{args.scene}_F{args.frame}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')

    # Also dump a small text table for cross-reference
    print()
    print('=== TX positions (config order, in-plane PCA coords) ===')
    print(f'{"name":<6s} {"pos_mm (world)":>32s} {"PC1 (mm)":>10s} {"PC2 (mm)":>10s}')
    for i, name in enumerate(tx_names):
        p = tx_pos[i]
        print(f'{name:<6s} [{p[0]:>9.2f}, {p[1]:>9.2f}, {p[2]:>9.2f}]'
              f' {tx_2d[i, 0]:>10.2f} {tx_2d[i, 1]:>10.2f}')
    print()
    print('=== RX positions (config order, in-plane PCA coords) ===')
    print(f'{"name":<6s} {"pos_mm (world)":>32s} {"PC1 (mm)":>10s} {"PC2 (mm)":>10s}')
    for i, name in enumerate(rx_names):
        p = rx_pos[i]
        print(f'{name:<6s} [{p[0]:>9.2f}, {p[1]:>9.2f}, {p[2]:>9.2f}]'
              f' {rx_2d[i, 0]:>10.2f} {rx_2d[i, 1]:>10.2f}')


if __name__ == '__main__':
    main()
