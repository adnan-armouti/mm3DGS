"""Doppler Gate 1 — GT Range-Doppler cube visualisation.

Gating check from md/doppler_forward_model_plan.md §"Sanity checks
before M2 sign-off" item (1):

    "Compute adc_to_rd_cube on 1 scene's GT, view |RD[test_frame]| for
    the 16 Doppler bins. Expected: static-scene scatterers concentrate
    at d = d_ego, dynamic scatterers (cars, pedestrians) occupy nearby
    bins. If everything collapses to a single bin, the Doppler axis
    carries no discriminating information for our scenes and M2 is
    premature."

Outputs (saved under mm25DGS_v6/output_frame_nvs/_doppler_gate/):

  <scene>/rd_cube_log.png       — |RD| summed-over-virtuals vs (d, r),
                                    dB scale, with expected d_ego
                                    bin marked.
  <scene>/rd_cube_linear.png    — same, linear scale.
  <scene>/rd_slices.png         — |RD| for 5 strongest range bins,
                                    across Doppler bins.
  <scene>/ego_info.json         — v_ego estimate, expected d_ego bin,
                                    frame-time params.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi  # noqa: E402
try:
    mi.set_variant('cuda_ad_rgb')
except Exception:
    pass

from mmir.data.io_utils import compute_range_res_from_cfg

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
OUT_ROOT = os.path.join(
    PROJECT_ROOT, 'mm25DGS_v6/output_frame_nvs/_doppler_gate')

# Scenes to sanity check
SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_438', 438),      # dynamic scene (naive-avg fails here)
    ('seq_2_frame_160', 160),      # static scene (naive-avg wins here)
]

LAMBDA_M = 3.0e8 / 77e9            # ≈ 3.896 mm, nominal MMWCAS carrier
T_FRAME = 0.1                      # 10 Hz cascade frame rate
# Slow-time Doppler sampling: chirp period Δt (7.87 ms / 16 ≈ 492 μs)
# between burst-start of consecutive chirps. f_s_slow = 1/Δt ≈ 2033 Hz
# across D=16 → per-bin resolution ≈ 127 Hz.
T_BURST = 7.87e-3
N_CHIRPS = 16
DT_CHIRP = T_BURST / N_CHIRPS
F_S_SLOW = 1.0 / DT_CHIRP
DOPPLER_BIN_HZ = F_S_SLOW / N_CHIRPS


def _load_adc_all_chirps(scene: str, frame: int) -> torch.Tensor:
    """Load all 16 chirps as (n_chirps=16, n_tx=12, n_rx=16, n_adc=256,
    2) real-imag on DEVICE. Input .npy is (16, 16_rx, 12_tx, 256) complex."""
    path = os.path.join(DATA_ROOT, scene, 'radar',
                         f'cascaded_frame_{frame}.npy')
    arr = np.load(path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'unexpected ADC shape: {arr.shape}')
    # (n_chirps=16, n_rx=16, n_tx=12, n_adc=256) cplx
    # Reorder to (n_chirps, n_tx, n_rx, n_adc) and split to (..., 2) real-imag.
    arr = arr.transpose(0, 2, 1, 3)                               # (CH, TX, RX, ADC)
    ri = np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)
    return torch.from_numpy(ri).to(DEVICE)                        # (CH, TX, RX, ADC, 2)


def _compute_rd_cube(adc_all_chirps: torch.Tensor) -> torch.Tensor:
    """ADC → per-virtual RD cube, matching the plan's GT pipeline.

    Input: ``(n_chirps=16, TX=12, RX=16, ADC=256, 2)`` real-imag.

    Steps:
      (1) Hann × range-FFT on ADC axis, per-chirp, per-(TX, RX).
      (2) Flatten (TX, RX) → n_virt=192 via v = tx * 16 + rx.
      (3) Hann × slow-time FFT on chirp axis → Doppler dim D=16.
      (4) fftshift along D so d=0 is centred (static scatterers land
          at centre given static scene; ego-motion shifts them).

    Returns ``(D=16, N_virt=192, R=256)`` complex on DEVICE.
    """
    assert adc_all_chirps.ndim == 5 and adc_all_chirps.shape[-1] == 2, (
        f'expected (CH, TX, RX, ADC, 2); got {tuple(adc_all_chirps.shape)}')
    x_c = torch.complex(
        adc_all_chirps[..., 0].contiguous(),
        adc_all_chirps[..., 1].contiguous(),
    )                                                              # (CH, TX, RX, ADC)
    n_ch, n_tx, n_rx, n_adc = x_c.shape

    # (1) Range window + FFT on ADC axis
    win_r = torch.hann_window(n_adc, device=x_c.device,
                               dtype=x_c.real.dtype).to(x_c.dtype)
    x_c = x_c * win_r[None, None, None, :]
    rp = torch.fft.fft(x_c, n=n_adc, dim=-1)                       # (CH, TX, RX, R)

    # (2) Flatten (tx, rx) → v
    rp = rp.reshape(n_ch, n_tx * n_rx, rp.shape[-1])               # (CH, N_virt=192, R)

    # (3) Slow-time Doppler window + FFT
    win_d = torch.hann_window(n_ch, device=rp.device,
                               dtype=rp.real.dtype).to(rp.dtype)
    rp = rp * win_d[:, None, None]
    rd = torch.fft.fft(rp, n=n_ch, dim=0)                          # (D, N_virt, R)

    # (4) fftshift along D so zero-Doppler is centred
    rd = torch.fft.fftshift(rd, dim=0)
    return rd


def _ego_velocity_from_configs(scene: str, frame: int) -> tuple:
    """Estimate v_ego(F) ≈ (T(F+1).pos − T(F−1).pos) / (2·T_frame).

    Uses the pass-2 aligned configs from the existing data/ tree.
    Returns (v_ego_vec, speed_mps, F_minus, F_plus, boresight_unit).
    """
    align_dir = os.path.join(
        DATA_ROOT, 'alignment_data', scene, 'cascade')
    def _pose(k):
        p = os.path.join(align_dir, f'cascaded_frame_{k}_aligned_pass2.json')
        if not os.path.isfile(p):
            return None
        try:
            cfg = json.load(open(p))
        except Exception:
            return None
        # Pass-2 aligned configs store per-TX dicts with pos_mm (mm) and
        # boresight (unit vec). Radar centre = mean of TX positions;
        # boresight = mean (then renormalised) of TX boresights.
        tx_array = cfg.get('tx_array')
        if tx_array is None or not isinstance(tx_array, list):
            return None
        positions_mm = np.array([t['pos_mm'] for t in tx_array],
                                 dtype=np.float64)
        bores = np.array([t['boresight'] for t in tx_array],
                          dtype=np.float64)
        pos_m = positions_mm.mean(axis=0) / 1000.0                  # mm → m
        bore  = bores.mean(axis=0)
        bore  = bore / (np.linalg.norm(bore) + 1e-12)
        return pos_m, bore

    p_minus = _pose(frame - 1)
    p_plus  = _pose(frame + 1)
    p_mid   = _pose(frame)
    if p_minus is None or p_plus is None:
        raise RuntimeError(
            f'{scene}: cannot find pass-2 configs for F±1 around {frame}')
    pos_m, _     = p_minus
    pos_p, _     = p_plus
    pos_c, bore  = p_mid
    v_ego = (pos_p - pos_m) / (2.0 * T_FRAME)
    speed = float(np.linalg.norm(v_ego))
    return v_ego, speed, frame - 1, frame + 1, bore


def _wrap_hz(f_hz: float, f_s: float) -> float:
    """Wrap ``f_hz`` into the unambiguous Doppler band [-f_s/2, f_s/2)
    (how an aliased signal shows up in the per-bin FFT)."""
    return ((f_hz + f_s / 2.0) % f_s) - f_s / 2.0


def _plot_rd_cube(rd: torch.Tensor, out_path_prefix: str,
                   range_res: float, f_d_ego: float,
                   extra_title: str = '') -> None:
    """Render + save the RD cube visualisations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # |RD|_summed_over_virt: (D, R) real, energy-like
    mag_sq = rd.real.pow(2) + rd.imag.pow(2)                    # (D, N, R)
    rd_energy = mag_sq.sum(dim=1).cpu().numpy()                 # (D, R)
    d_count = rd_energy.shape[0]
    r_count = rd_energy.shape[1]

    # Convert Doppler-bin index to Hz (shifted so 0 at centre).
    d_bin_ctr = d_count // 2
    d_hz = (np.arange(d_count) - d_bin_ctr) * DOPPLER_BIN_HZ
    # Expected d_ego, wrapped into the unambiguous Doppler band. Our
    # chirp repetition period (~492 µs) gives f_s ≈ 2033 Hz so
    # |f_d_max_unambig| = 1017 Hz — below typical ego-motion signal
    # magnitudes (2000-2500 Hz). Static-scene scatterers alias into
    # this band; the observed ego peak is at the WRAPPED frequency.
    f_d_wrapped = _wrap_hz(f_d_ego, F_S_SLOW)
    d_ego_bin = int(np.round(f_d_wrapped / DOPPLER_BIN_HZ)) + d_bin_ctr
    d_ego_bin = max(0, min(d_count - 1, d_ego_bin))

    # Range axis in metres
    r_m = np.arange(r_count) * range_res

    # --- Plot 1: linear
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    vmin, vmax = rd_energy.min(), rd_energy.max()
    rd_norm = (rd_energy - vmin) / max(vmax - vmin, 1e-30)
    im = ax.imshow(rd_norm, cmap='hot', aspect='auto', origin='lower',
                    extent=[r_m[0], r_m[-1], d_hz[0], d_hz[-1]])
    ax.axhline(0.0, color='cyan', linestyle=':', linewidth=1, alpha=0.8,
                label='static d=0')
    ax.axhline(f_d_wrapped, color='lime', linestyle='--', linewidth=1.2,
                label=f'ego peak (aliased)  raw={f_d_ego:+.0f} Hz  '
                      f'wrapped={f_d_wrapped:+.1f} Hz '
                      f'(bin {d_ego_bin - d_bin_ctr:+d})')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Doppler (Hz)')
    ax.set_title(f'|RD| sum-over-virt (linear) — {extra_title}')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path_prefix + '_rd_cube_linear.png',
                 dpi=110, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: dB
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    rd_db = 10.0 * np.log10(np.clip(rd_energy / max(vmax, 1e-30), 1e-6, 1.0))
    im = ax.imshow(rd_db, cmap='hot', aspect='auto', origin='lower',
                    vmin=-40, vmax=0,
                    extent=[r_m[0], r_m[-1], d_hz[0], d_hz[-1]])
    ax.axhline(0.0, color='cyan', linestyle=':', linewidth=1, alpha=0.8,
                label='static d=0')
    ax.axhline(f_d_wrapped, color='lime', linestyle='--', linewidth=1.2,
                label=f'ego peak (aliased)  raw={f_d_ego:+.0f} Hz  '
                      f'wrapped={f_d_wrapped:+.1f} Hz '
                      f'(bin {d_ego_bin - d_bin_ctr:+d})')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Doppler (Hz)')
    ax.set_title(f'|RD| sum-over-virt (dB) — {extra_title}')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('dB')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path_prefix + '_rd_cube_log.png',
                 dpi=110, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 3: strongest-range Doppler profiles
    r_energy = rd_energy.sum(axis=0)                             # (R,) total energy/range
    top_r = np.argsort(r_energy)[::-1][:5]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for r_idx in top_r:
        ax.plot(d_hz, rd_energy[:, r_idx] / vmax,
                 label=f'r={r_idx} ({r_m[r_idx]:.1f} m)',
                 linewidth=1.3)
    ax.axvline(0.0, color='cyan', linestyle=':', linewidth=1,
                alpha=0.8, label='static d=0')
    ax.axvline(f_d_wrapped, color='lime', linestyle='--', linewidth=1.2,
                label=f'ego peak wrapped  {f_d_wrapped:+.1f} Hz')
    ax.set_xlabel('Doppler (Hz)')
    ax.set_ylabel('Relative |RD|² (norm to max)')
    ax.set_title(f'Top-5-range Doppler slices — {extra_title}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path_prefix + '_rd_slices.png',
                 dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f'[gate1] Doppler cube sanity check — λ = {LAMBDA_M*1e3:.3f} mm,'
          f' Δt = {DT_CHIRP*1e6:.1f} µs, per-bin Doppler = {DOPPLER_BIN_HZ:.1f} Hz')
    print('=' * 100)

    for scene, F in SCENES:
        out_dir = os.path.join(OUT_ROOT, scene)
        os.makedirs(out_dir, exist_ok=True)

        # v_ego + boresight direction → expected Doppler for a scatterer
        # at boresight.
        v_ego, speed, F_m, F_p, bore = _ego_velocity_from_configs(scene, F)
        f_d_boresight = 2.0 / LAMBDA_M * float(np.dot(v_ego, bore))
        f_d_max       = 2.0 / LAMBDA_M * speed

        # ADC → RD cube
        adc = _load_adc_all_chirps(scene, F)
        rd = _compute_rd_cube(adc)                              # (D=16, 192, R=256)

        # Range res from aligned config (consistent with v5 pipeline)
        cfg = os.path.join(DATA_ROOT, 'alignment_data', scene, 'cascade',
                            f'cascaded_frame_{F}_aligned_pass2.json')
        range_res = compute_range_res_from_cfg(cfg)

        title = f'{scene}  F={F}  |v_ego|={speed:.2f} m/s'
        _plot_rd_cube(rd, out_path_prefix=os.path.join(out_dir, scene),
                       range_res=range_res, f_d_ego=f_d_boresight,
                       extra_title=title)

        # Doppler-bin statistics
        mag_sq = (rd.real.pow(2) + rd.imag.pow(2))              # (D, N, R)
        rd_energy = mag_sq.sum(dim=1).cpu().numpy()             # (D, R)
        # Wrap the expected ego-motion Doppler into the unambiguous
        # band so the bin lookup lands inside the 16-bin tensor.
        d_bin_ctr = rd_energy.shape[0] // 2
        f_d_wrapped_local = _wrap_hz(f_d_boresight, F_S_SLOW)
        d_ego_bin_offset = int(np.round(f_d_wrapped_local / DOPPLER_BIN_HZ))
        d_ego_bin = max(0, min(rd_energy.shape[0] - 1,
                                d_bin_ctr + d_ego_bin_offset))
        total_e = float(rd_energy.sum())
        # Widen to ±2 for Hann leakage.
        d_lo = max(0, d_ego_bin - 2)
        d_hi = min(rd_energy.shape[0], d_ego_bin + 3)
        concentrated_e = float(rd_energy[d_lo:d_hi, :].sum())
        frac_concentrated = concentrated_e / max(total_e, 1e-30)

        # Proxy for "dynamic content": energy ≥3 bins away from static+ego
        d_static = d_bin_ctr
        # Remove bins within ±2 of either static or ego
        static_mask = np.zeros(rd_energy.shape[0], dtype=bool)
        for c in (d_static, d_ego_bin):
            lo = max(0, c - 2); hi = min(rd_energy.shape[0], c + 3)
            static_mask[lo:hi] = True
        dyn_e = float(rd_energy[~static_mask, :].sum())
        frac_dynamic = dyn_e / max(total_e, 1e-30)

        info = {
            'scene': scene, 'F': int(F), 'F_minus': int(F_m), 'F_plus': int(F_p),
            'v_ego_xyz': list(map(float, v_ego)),
            'speed_mps': float(speed),
            'boresight_xyz': list(map(float, bore)),
            'f_d_boresight_hz_raw': float(f_d_boresight),
            'f_d_boresight_hz_wrapped': float(f_d_wrapped_local),
            'f_d_max_hz': float(f_d_max),
            'f_s_slow_hz': float(F_S_SLOW),
            'f_d_max_unambig_hz': float(F_S_SLOW / 2.0),
            'signal_aliases': bool(abs(f_d_boresight) > F_S_SLOW / 2.0),
            'doppler_bin_hz': float(DOPPLER_BIN_HZ),
            'ego_peak_bin_offset_from_center': int(d_ego_bin_offset),
            'frac_energy_within_2_bins_of_ego_peak': float(frac_concentrated),
            'frac_energy_far_from_both_static_and_ego': float(frac_dynamic),
            'range_res_m': float(range_res),
        }
        with open(os.path.join(out_dir, 'ego_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        aliased_str = ' (ALIASED)' if info['signal_aliases'] else ''
        print(f'{scene}  F={F}')
        print(f'  |v_ego| = {speed:.2f} m/s,'
              f'  f_d_bore_raw = {f_d_boresight:+.1f} Hz'
              f'{aliased_str}')
        print(f'  f_d_bore_wrapped = {f_d_wrapped_local:+.1f} Hz'
              f'  (observed peak at bin {d_ego_bin_offset:+d}'
              f' after fftshift, centre bin = {d_bin_ctr})')
        print(f'  frac energy within ±2 bins of ego peak      = {frac_concentrated*100:.1f}%')
        print(f'  frac energy far from both static + ego peaks = {frac_dynamic*100:.1f}%')
        print(f'  → plots saved under {out_dir}/')

    print('=' * 100)
    print('Gate 1 PASS criterion: ≥ 3 of the tested scenes show non-trivial '
          'RD structure — i.e. the ego-peak-concentrated fraction is NOT '
          '~100% (which would mean "all energy in one bin, no '
          'discrimination"). Typical pass value: 50–90% concentrated, '
          '10–50% dynamic/spread.')


if __name__ == '__main__':
    main()
