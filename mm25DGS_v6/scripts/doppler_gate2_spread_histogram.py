"""Doppler Gate 2 — predicted per-point Doppler spread histogram.

Gating check from md/doppler_forward_model_plan.md §"Sanity checks
before M2 sign-off" item (2):

    "Compute f_d(p) for every FPS-selected point in the seed-frame
    model, histogram. Expected: spread of ~0.5 – 2 Hz (at |v_ego| ~ 5
    m/s, λ = 4 mm, this is bins 1..4 of a 16-bin cube over 7.87 ms). If
    < 1 bin of spread, the axis won't discriminate points — M2 becomes
    cosmetic."

Note: the doc's "~0.5 – 2 Hz" spread figure is **per-Hz**; the relevant
quantity for M2 discriminability is bin-count spread, which at
127 Hz/bin needs to be ≥ 1 bin to give training signal and ~3–5 bins
to give reasonable angular discrimination.

Outputs (under mm25DGS_v6/output_frame_nvs/_doppler_gate/):

  <scene>/doppler_spread_hist.png   — per-point f_d distribution with
                                        bin boundaries, ego peak, and
                                        n_occupied-bin count.
  <scene>/doppler_spread.json       — numeric summary per scene.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Constants consistent with Gate 1 + the plan doc
LAMBDA_M = 3.0e8 / 77e9
T_BURST = 7.87e-3
N_CHIRPS = 16
DT_CHIRP = T_BURST / N_CHIRPS
F_S_SLOW = 1.0 / DT_CHIRP
DOPPLER_BIN_HZ = F_S_SLOW / N_CHIRPS

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
OUT_ROOT = os.path.join(
    PROJECT_ROOT, 'mm25DGS_v6/output_frame_nvs/_doppler_gate')

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_160', 160),
]


def _radar_center_and_vego(scene: str, frame: int):
    """Return (radar_center_m, v_ego, boresight_unit)."""
    align_dir = os.path.join(DATA_ROOT, 'alignment_data', scene, 'cascade')
    def _pose(k):
        p = os.path.join(align_dir, f'cascaded_frame_{k}_aligned_pass2.json')
        if not os.path.isfile(p):
            return None
        cfg = json.load(open(p))
        tx_array = cfg['tx_array']
        positions_mm = np.array([t['pos_mm'] for t in tx_array],
                                 dtype=np.float64)
        bores = np.array([t['boresight'] for t in tx_array],
                          dtype=np.float64)
        pos_m = positions_mm.mean(axis=0) / 1000.0
        bore  = bores.mean(axis=0)
        bore  = bore / (np.linalg.norm(bore) + 1e-12)
        return pos_m, bore
    pos_m, _    = _pose(frame - 1)
    pos_p, _    = _pose(frame + 1)
    pos_c, bore = _pose(frame)
    v_ego = (pos_p - pos_m) / (2.0 * 0.1)
    return pos_c, v_ego, bore


def _wrap_hz(f_hz: np.ndarray, f_s: float) -> np.ndarray:
    return ((f_hz + f_s / 2.0) % f_s) - f_s / 2.0


def _scene_pcl_subsample(scene: str, max_points: int = 20000) -> np.ndarray:
    """Load scene/pcl.npy and FPS-subsample to the typical training
    target-n (20 000). Returns (N, 3) XYZ in world frame."""
    path = os.path.join(DATA_ROOT, scene, 'scene', 'pcl.npy')
    arr = np.load(path)                                          # (N, 7): xyz, normal, intensity
    xyz = arr[:, :3].astype(np.float32)
    if xyz.shape[0] <= max_points:
        return xyz
    # Uniform random downsample — stand-in for the trainer's FPS.
    # For a histogram we don't need geometric coverage optimality, just
    # a representative sample of the scene extent.
    rng = np.random.default_rng(42)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f'[gate2] per-point Doppler spread — λ={LAMBDA_M*1e3:.3f} mm, '
          f'{DOPPLER_BIN_HZ:.1f} Hz/bin, N_bins={N_CHIRPS}')
    print('=' * 100)

    for scene, F in SCENES:
        out_dir = os.path.join(OUT_ROOT, scene)
        os.makedirs(out_dir, exist_ok=True)

        pos_c, v_ego, bore = _radar_center_and_vego(scene, F)
        speed = float(np.linalg.norm(v_ego))
        xyz = _scene_pcl_subsample(scene)
        # Direction radar→point (unit)
        diff = xyz - pos_c[None, :]
        r = np.linalg.norm(diff, axis=1).clip(min=1e-9)
        u = diff / r[:, None]
        # Per-point Doppler frequency (radial velocity projection)
        f_d = (2.0 / LAMBDA_M) * (u @ v_ego)                     # (N,)
        f_d_max = (2.0 / LAMBDA_M) * speed
        # Wrap into unambiguous band
        f_d_wrapped = _wrap_hz(f_d, F_S_SLOW)

        # Histogram at bin resolution (Doppler-bin centres = k·DOPPLER_BIN_HZ
        # with k ∈ [-N/2, N/2))
        bin_edges = (np.arange(-N_CHIRPS // 2, N_CHIRPS // 2 + 1)
                      - 0.5) * DOPPLER_BIN_HZ
        hist_wrapped, _ = np.histogram(f_d_wrapped, bins=bin_edges)
        n_nonempty = int((hist_wrapped > 0).sum())
        # Top-1 bin share
        top_share = float(hist_wrapped.max()) / max(1, hist_wrapped.sum())

        # Physical range of f_d (pre-wrap), measures the ego-motion
        # modulation depth across scene.
        f_d_range_hz = float(f_d.max() - f_d.min())
        f_d_range_bins = f_d_range_hz / DOPPLER_BIN_HZ

        info = {
            'scene': scene, 'F': int(F),
            'speed_mps': speed,
            'lambda_mm': LAMBDA_M * 1e3,
            'f_d_max_hz_unaliased': float(f_d_max),
            'f_d_max_unambig_hz': float(F_S_SLOW / 2.0),
            'signal_aliases': bool(f_d_max > F_S_SLOW / 2.0),
            'f_d_range_hz_raw': f_d_range_hz,
            'f_d_range_bins_raw': f_d_range_bins,
            'n_bins_occupied_after_wrap': n_nonempty,
            'top_bin_fraction': top_share,
            'n_points_sampled': int(xyz.shape[0]),
        }
        with open(os.path.join(out_dir, 'doppler_spread.json'), 'w') as f:
            json.dump(info, f, indent=2)

        # Plot: (a) raw f_d distribution (shows unaliased range), (b)
        # wrapped f_d histogram into the N_CHIRPS bins.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.hist(f_d, bins=60, color='#ff7733', edgecolor='white', linewidth=0.5)
        ax1.axvline(0.0, color='cyan', linestyle=':', linewidth=1, label='f_d=0')
        ax1.axvspan(-F_S_SLOW/2, F_S_SLOW/2, color='lime', alpha=0.12,
                     label=f'unambig band  ±{F_S_SLOW/2:.0f} Hz')
        ax1.axvline(+f_d_max, color='red', linestyle='--', linewidth=1,
                     label=f'±f_d_max = ±{f_d_max:.0f} Hz')
        ax1.axvline(-f_d_max, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('Per-point Doppler f_d (Hz, unaliased)')
        ax1.set_ylabel('# points')
        ax1.set_title(f'{scene}  raw — |v_ego|={speed:.2f} m/s  '
                       f'range = {f_d_range_bins:.1f} bins')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        bin_ctrs = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax2.bar(bin_ctrs, hist_wrapped, width=DOPPLER_BIN_HZ * 0.9,
                 color='#ffb300', edgecolor='black', linewidth=0.3)
        ax2.axvline(0.0, color='cyan', linestyle=':', linewidth=1, label='static d=0')
        ax2.set_xlabel('Doppler bin centre (Hz, wrapped into ±f_s/2)')
        ax2.set_ylabel('# points')
        ax2.set_title(f'{scene}  wrapped — '
                       f'{n_nonempty}/{N_CHIRPS} bins occupied, '
                       f'top = {top_share*100:.0f}%')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'doppler_spread_hist.png'),
                     dpi=110, bbox_inches='tight')
        plt.close(fig)

        aliased_str = ' (ALIASED)' if info['signal_aliases'] else ''
        print(f'{scene}  F={F}  |v|={speed:.2f} m/s{aliased_str}')
        print(f'  raw f_d span = {f_d.min():+.0f} .. {f_d.max():+.0f} Hz '
              f'= {f_d_range_bins:.1f} bins')
        print(f'  after wrap: {n_nonempty}/{N_CHIRPS} bins occupied, '
              f'top-bin share = {top_share*100:.1f}%')
        print(f'  → plots in {out_dir}/')

    print('=' * 100)
    print('Gate 2 PASS criterion: wrapped-Doppler distribution occupies '
          '≥ 4 of 16 bins on at least 2/3 scenes, AND top-bin fraction '
          '< 60% (i.e. not all energy collapses to one bin).')


if __name__ == '__main__':
    main()
