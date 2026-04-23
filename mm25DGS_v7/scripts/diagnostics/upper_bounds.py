"""§2.0 — empirical upper bounds on |RA| / |RAD| CC.

Answers questions 1 + 4 of the fundamental-ceiling decision gate
(md/mm25dgs_v7_fundamental_ceiling.md §5):

  §2.0.a  intra-frame chirp coherence    — CC(chirp_0 |RA|, chirp_k |RA|)
  §2.0.b  inter-frame NVS ceiling        — CC(|RA|_F, |RA|_{F±k})
  §2.0.c  naive-avg baseline             — CC(0.5·(F-1 + F+1), F)
  §2.0.d  per-bin GT variance heatmap    — std/mean across 16 chirps
  §2.0.e  ego-motion-only |RAD| ceiling  — synth from chirp-0 + v_ego

Outputs:
  md/diagnostics/upper_bounds_results.json  — all numbers
  md/diagnostics/upper_bounds_results.md    — pretty table
  md/diagnostics/per_bin_variance_<scene>.png — heatmaps (one per scene)

All six bench scenes; cheap (a few minutes total).
"""
from __future__ import annotations

import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmir.data.ra_utils import adc_to_ra_complex
from mm25DGS_v7.data.ra_utils import (
    adc_to_rad_complex,
    _batch_txrx_to_vx_el0,
    _azimuth_fft_on_vx86,
    _doppler_fft_on_chirps,
    _hann,
    N_DOP_DEFAULT,
)
from mm25DGS_v7.preprocessing.v_ego import get_or_compute_v_ego


DATA_ROOT = '/home/adnan/Desktop/mm3DGS/data'
OUT_DIR   = '/home/adnan/Desktop/mm3DGS/md/diagnostics'

SCENES = [
    ('seq_0_frame_135', 135),
    ('seq_1_frame_185', 185),
    ('seq_1_frame_438', 438),
    ('seq_2_frame_105', 105),
    ('seq_2_frame_160', 160),
    ('seq_2_frame_300', 300),
]

# Frames around the test frame we consider for NVS coherence (train window).
INTER_FRAME_DELTAS = [1, 2, 3, 4]


def _flat_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    am = a - a.mean(); bm = b - b.mean()
    num = (am * bm).sum()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum()).clamp_min(1e-30)
    return float((num / den).item())


def _adc_ri_tensor(scene: str, frame: int, device='cuda') -> torch.Tensor:
    """Load cascaded ADC as (16, 12, 16, 256, 2) float32."""
    arr = np.load(f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{frame}.npy')
    assert arr.ndim == 4 and arr.shape[0] == 16, arr.shape
    adc = arr.transpose(0, 2, 1, 3)   # (CH, TX, RX, ADC)
    ri  = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    return torch.from_numpy(ri).to(device)


def _chirp_range_fft(adc_ri_5d: torch.Tensor) -> torch.Tensor:
    """(CH, TX, RX, ADC, 2) → (CH, TX, RX, R) complex, after Hann + range FFT."""
    x_c = torch.complex(
        adc_ri_5d[..., 0].contiguous(),
        adc_ri_5d[..., 1].contiguous(),
    )                                          # (CH, TX, RX, ADC)
    n_adc = x_c.shape[-1]
    win_r = _hann(n_adc, x_c.device, x_c.real.dtype).to(x_c.dtype)
    x_c = x_c * win_r[None, None, None, :]
    return torch.fft.fft(x_c, n=n_adc, dim=-1)


def _chirp_range_fft_to_ra_mag(rp_chirps: torch.Tensor) -> torch.Tensor:
    """(CH, TX, RX, R) complex → (CH, 127, R) |RA| magnitudes."""
    stack86 = _batch_txrx_to_vx_el0(rp_chirps)        # (CH, 86, R) complex
    ra_complex = _azimuth_fft_on_vx86(stack86)        # (CH, 127, R) complex
    return ra_complex.abs().float()                   # (CH, 127, R)


# ---------------------------------------------------------------------------
# §2.0.a — intra-frame chirp coherence
# ---------------------------------------------------------------------------

def intra_frame_chirp_coherence(scene: str, F: int, device='cuda') -> dict:
    """|RA| CC between chirp 0 and every other chirp, same frame."""
    adc = _adc_ri_tensor(scene, F, device)              # (16, 12, 16, 256, 2)
    rp  = _chirp_range_fft(adc)                         # (16, 12, 16, 256) cpx
    ra_mag = _chirp_range_fft_to_ra_mag(rp)             # (16, 127, 256)

    ccs = []
    for k in range(1, 16):
        ccs.append(_flat_pearson(ra_mag[0], ra_mag[k]))
    return {
        'scene': scene, 'F': F,
        'cc_chirp0_vs_chirpK_mean': float(np.mean(ccs)),
        'cc_chirp0_vs_chirpK_std':  float(np.std(ccs)),
        'cc_chirp0_vs_chirpK_min':  float(np.min(ccs)),
        'cc_chirp0_vs_chirpK_max':  float(np.max(ccs)),
    }


# ---------------------------------------------------------------------------
# §2.0.b — inter-frame NVS ceiling
# ---------------------------------------------------------------------------

def inter_frame_nvs_ceiling(scene: str, F: int, device='cuda') -> dict:
    """|RA| CC of chirp-0 at F vs chirp-0 at F±k, for k in INTER_FRAME_DELTAS."""
    # Load once; one chirp per frame (chirp 0)
    rp_by_frame = {}
    for k in [0] + INTER_FRAME_DELTAS + [-d for d in INTER_FRAME_DELTAS]:
        g = F + k
        radar_path = f'{DATA_ROOT}/{scene}/radar/cascaded_frame_{g}.npy'
        if not os.path.isfile(radar_path):
            continue
        adc = _adc_ri_tensor(scene, g, device)          # (16, 12, 16, 256, 2)
        rp  = _chirp_range_fft(adc[0:1])                # chirp 0 only
        ra_mag = _chirp_range_fft_to_ra_mag(rp)[0]       # (127, 256)
        rp_by_frame[k] = ra_mag

    out = {'scene': scene, 'F': F}
    for k in INTER_FRAME_DELTAS:
        if k in rp_by_frame and -k in rp_by_frame:
            cc_plus  = _flat_pearson(rp_by_frame[0], rp_by_frame[k])
            cc_minus = _flat_pearson(rp_by_frame[0], rp_by_frame[-k])
            out[f'cc_F_vs_F+{k}'] = cc_plus
            out[f'cc_F_vs_F-{k}'] = cc_minus
            out[f'cc_F_vs_F±{k}_mean'] = 0.5 * (cc_plus + cc_minus)
    return out


# ---------------------------------------------------------------------------
# §2.0.c — naive-avg baseline
# ---------------------------------------------------------------------------

def naive_avg_baseline(scene: str, F: int, device='cuda') -> dict:
    """|RA| CC(0.5*(|RA|_{F-1} + |RA|_{F+1}), |RA|_F), chirp 0 only."""
    rp_by_frame = {}
    for k in [-1, 0, 1]:
        g = F + k
        adc = _adc_ri_tensor(scene, g, device)
        rp  = _chirp_range_fft(adc[0:1])
        ra_mag = _chirp_range_fft_to_ra_mag(rp)[0]
        rp_by_frame[k] = ra_mag
    naive = 0.5 * (rp_by_frame[-1] + rp_by_frame[+1])
    return {
        'scene': scene, 'F': F,
        'cc_naive_avg_vs_F': _flat_pearson(naive, rp_by_frame[0]),
    }


# ---------------------------------------------------------------------------
# §2.0.d — per-bin GT variance heatmap
# ---------------------------------------------------------------------------

def per_bin_variance_heatmap(scene: str, F: int, device='cuda') -> dict:
    adc = _adc_ri_tensor(scene, F, device)
    rp  = _chirp_range_fft(adc)
    ra_mag = _chirp_range_fft_to_ra_mag(rp)             # (16, 127, 256)
    mu  = ra_mag.mean(0)
    std = ra_mag.std(0)
    cv  = std / mu.clamp_min(1e-30)

    cv_np = cv.cpu().numpy()
    mu_np = mu.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im0 = axes[0].imshow(np.log10(mu_np.clip(min=1e-6)),
                          aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_title(f'{scene} F={F} — log10 mean |RA| across 16 chirps')
    axes[0].set_xlabel('range bin'); axes[0].set_ylabel('azimuth bin')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cv_np, aspect='auto', cmap='magma',
                          origin='lower', vmin=0, vmax=1.0)
    axes[1].set_title(
        f'{scene} F={F} — per-bin std/mean (coeff of variation) across 16 chirps\n'
        f'(high = moving scatterer / low SNR; low = stable target)')
    axes[1].set_xlabel('range bin'); axes[1].set_ylabel('azimuth bin')
    fig.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, f'per_bin_variance_{scene}_F{F}.png')
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    cv_flat = cv_np.flatten()
    return {
        'scene': scene, 'F': F,
        'per_bin_cv_mean':    float(cv_flat.mean()),
        'per_bin_cv_median':  float(np.median(cv_flat)),
        'per_bin_cv_p90':     float(np.percentile(cv_flat, 90)),
        'per_bin_cv_p99':     float(np.percentile(cv_flat, 99)),
        'heatmap_png':        out_png,
    }


# ---------------------------------------------------------------------------
# §2.0.e — ego-motion-only |RAD| ceiling
# ---------------------------------------------------------------------------

def ego_motion_only_rad_ceiling(scene: str, F: int, device='cuda') -> dict:
    """Synthesise a "perfect ego-Doppler" |RAD| from chirp-0 GT |RA| alone,
    compare to true GT |RAD|.

    Construction:
      - take GT chirp-0 complex |RA| (127, 256)
      - for bin (a, r): assign direction û_bin based on azimuth angle +
        range r (boresight = 0° azimuth, radar at origin)
      - A_bin = -(4π/λ)·⟨û_bin, v_ego⟩                                (per-bin)
      - synthetic_chirp_m_RA = chirp_0_RA · exp(i · A_bin · m · T_c)
      - stack 16 synthetic chirps, Doppler FFT (same chain as GT pipeline)
      - CC against true GT |RAD|.

    If the true scene is static and ego-compliant, this reproduces GT RAD
    exactly. The gap is "how much non-ego / moving content is in GT RAD".
    """
    # GT |RAD|
    adc_ri = _adc_ri_tensor(scene, F, device)           # (16, 12, 16, 256, 2)
    gt_rad = adc_to_rad_complex(adc_ri).abs().float()   # (D, 127, 256)

    # GT chirp-0 complex |RA| (same FFT chain as adc_to_rad_complex but
    # stopping after azimuth FFT for chirp 0 only)
    rp_all = _chirp_range_fft(adc_ri)                   # (16, 12, 16, 256) cpx
    stack86 = _batch_txrx_to_vx_el0(rp_all[0:1])        # (1, 86, 256) cpx
    ra_chirp0 = _azimuth_fft_on_vx86(stack86)[0]        # (127, 256) complex

    # Azimuth bin index → angle (rad). Our FFT chain: 86-sized, padded to
    # 128, drop bin 0, fftshift. After fftshift on 127 (odd), center bin
    # is at index 63; azimuth axis spans approximately ±π/2 across bins.
    # We use sin(θ) = (bin - 63) · (2 / 127) · (1) as a first-order
    # approximation — matches the typical half-space mapping from virtual
    # array FFT for a lateral ULA.
    az_bins = 127
    sin_theta = (torch.arange(az_bins, device=device, dtype=torch.float32)
                  - (az_bins - 1) / 2.0) * (2.0 / az_bins)
    sin_theta = sin_theta.clamp(-1.0, 1.0)
    cos_theta = torch.sqrt((1 - sin_theta ** 2).clamp_min(0))
    # û direction: (x = range·cos, y = range·sin, z = 0), but we only need
    # the UNIT direction — (cos, sin, 0).
    u_hat = torch.stack([cos_theta, sin_theta,
                          torch.zeros_like(sin_theta)], dim=-1)  # (127, 3)

    # v_ego for this frame (GT-trajectory-interp)
    v_ego = get_or_compute_v_ego(scene, int(F), data_root=DATA_ROOT)
    v_ego = torch.as_tensor(v_ego, dtype=torch.float32, device=device)

    # Per-az Doppler factor A = -(4π/λ)·⟨û, v_ego⟩. Shape: (127,)
    wavelength = 3.0e8 / 77e9
    A_per_az = (-(4.0 * math.pi / wavelength) *
                 (u_hat * v_ego.view(1, 3)).sum(-1))

    # T_c = 7.87e-3 / 16 (per v7 doppler constants)
    T_c = 7.87e-3 / 16.0

    n_chirps = 16
    m_idx = torch.arange(n_chirps, device=device, dtype=torch.float32)
    phi_m_az = A_per_az.unsqueeze(0) * m_idx.unsqueeze(1) * T_c  # (CH, 127)

    # Broadcast to (CH, 127, R=256)
    R = ra_chirp0.shape[-1]
    phase = phi_m_az.unsqueeze(-1).expand(-1, -1, R)             # (CH, 127, R)
    synth_chirps_ra = ra_chirp0.unsqueeze(0) * torch.exp(
        torch.complex(torch.zeros_like(phase), phase))           # (CH, 127, R)

    # Doppler FFT on chirp axis (same chain as GT RAD pipeline)
    synth_rad = _doppler_fft_on_chirps(
        synth_chirps_ra, n_dop=N_DOP_DEFAULT, dim=0).abs().float()

    assert synth_rad.shape == gt_rad.shape

    cc_all = _flat_pearson(synth_rad, gt_rad)

    # Also compute "DC-only" baseline: no ego doppler at all (scene static
    # AND radar stationary). synth_rad_DC = chirp_0 replicated 16 times
    # then Doppler FFT — energy concentrated at DC.
    synth_chirps_dc = ra_chirp0.unsqueeze(0).expand(n_chirps, -1, -1)
    synth_rad_dc = _doppler_fft_on_chirps(
        synth_chirps_dc, n_dop=N_DOP_DEFAULT, dim=0).abs().float()
    cc_dc = _flat_pearson(synth_rad_dc, gt_rad)

    return {
        'scene': scene, 'F': F,
        'cc_ego_motion_rad_ceiling': cc_all,
        'cc_dc_only_rad_baseline':   cc_dc,
        'v_ego_norm_m_per_s':        float(v_ego.norm().item()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {'scenes': []}

    for scene, F in SCENES:
        print(f'\n=== {scene} F={F} ===')
        r = {'scene': scene, 'F': F}

        print('  §2.0.a intra-frame chirp coherence ...')
        r.update(intra_frame_chirp_coherence(scene, F))

        print('  §2.0.b inter-frame NVS ceiling ...')
        r.update(inter_frame_nvs_ceiling(scene, F))

        print('  §2.0.c naive-avg baseline ...')
        r.update(naive_avg_baseline(scene, F))

        print('  §2.0.d per-bin variance ...')
        r.update(per_bin_variance_heatmap(scene, F))

        print('  §2.0.e ego-motion-only |RAD| ceiling ...')
        r.update(ego_motion_only_rad_ceiling(scene, F))

        all_results['scenes'].append(r)
        print(f'  done: chirp CC={r["cc_chirp0_vs_chirpK_mean"]:.4f}  '
              f'naive_avg={r["cc_naive_avg_vs_F"]:.4f}  '
              f'ego_RAD_ceil={r["cc_ego_motion_rad_ceiling"]:.4f}')

    # Aggregate
    def mean(key):
        vs = [s[key] for s in all_results['scenes'] if key in s]
        return float(np.mean(vs)) if vs else None

    summary = {
        'chirp_coherence_mean':   mean('cc_chirp0_vs_chirpK_mean'),
        'inter_F±1_mean':         mean('cc_F_vs_F±1_mean'),
        'inter_F±4_mean':         mean('cc_F_vs_F±4_mean'),
        'naive_avg_mean':         mean('cc_naive_avg_vs_F'),
        'ego_RAD_ceiling_mean':   mean('cc_ego_motion_rad_ceiling'),
        'dc_only_RAD_mean':       mean('cc_dc_only_rad_baseline'),
        'per_bin_cv_median_mean': mean('per_bin_cv_median'),
    }
    all_results['summary'] = summary

    # Write JSON
    json_path = os.path.join(OUT_DIR, 'upper_bounds_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Write markdown table
    md_path = os.path.join(OUT_DIR, 'upper_bounds_results.md')
    with open(md_path, 'w') as f:
        f.write('# §2.0 Upper-bound diagnostics\n\n')
        f.write('Generated by `mm25DGS_v7/scripts/diagnostics/upper_bounds.py`.\n\n')
        f.write('## Per-scene\n\n')
        f.write('| scene | F | chirp CC | F±1 | F±4 | naive avg | ego-RAD ceil | DC-only | CV median |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for s in all_results['scenes']:
            f.write(f'| {s["scene"]} | {s["F"]} | '
                     f'{s.get("cc_chirp0_vs_chirpK_mean", 0):.4f} | '
                     f'{s.get("cc_F_vs_F±1_mean", 0):.4f} | '
                     f'{s.get("cc_F_vs_F±4_mean", 0):.4f} | '
                     f'{s.get("cc_naive_avg_vs_F", 0):.4f} | '
                     f'{s.get("cc_ego_motion_rad_ceiling", 0):.4f} | '
                     f'{s.get("cc_dc_only_rad_baseline", 0):.4f} | '
                     f'{s.get("per_bin_cv_median", 0):.3f} |\n')
        f.write('\n## 6-scene means\n\n')
        for k, v in summary.items():
            f.write(f'- **{k}**: {v:.4f}\n' if v is not None else f'- {k}: —\n')

    print(f'\n[done] wrote {json_path}')
    print(f'[done] wrote {md_path}')
    print(f'[done] {len(all_results["scenes"])} heatmap PNGs under {OUT_DIR}/')


if __name__ == '__main__':
    main()
