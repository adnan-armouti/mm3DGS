"""Unit test — fused step5_doppler kernel vs 16× step5_fused reference.

Run:
    cd /home/adnan/Desktop/mm3DGS
    /home/adnan/.conda/envs/mmir/bin/python -m mm25DGS_v7.cuda.tests.test_step5_doppler
"""
from __future__ import annotations

import math
import torch

from mm25DGS_v5.cuda import step5_fused  # reference (per-chirp autograd)
from mm25DGS_v7.cuda import step5_doppler_fused


def _make_psf_table(K=256, spread=15, n_grid=1024, device='cuda'):
    """Build the same Hann-PSF lookup table that HannPSFTable uses."""
    from mm25DGS_v5.psf import HannPSFTable
    return HannPSFTable(K, spread, n_grid=n_grid, device=device)


def _reference(w_full, phi_base, n_peak, A, t_off, psf_tab, K, w_threshold):
    """Replicate the pre-fusion Python loop using step5_fused per chirp."""
    n_chirps, n_tx = t_off.shape
    n_rx = w_full.shape[-1]
    rad_r, rad_i = [], []
    for m in range(n_chirps):
        phi_doppler = A.unsqueeze(-1) * t_off[m].unsqueeze(0)          # (M, n_tx)
        phi_m = phi_base + phi_doppler.unsqueeze(-1)                   # (M, n_tx, n_rx)
        rp_r, rp_i = step5_fused(
            w_full.contiguous(), phi_m.contiguous(), n_peak.contiguous(),
            psf_tab.psf_real, psf_tab.psf_imag, K, w_threshold,
        )
        rad_r.append(rp_r); rad_i.append(rp_i)
    return torch.stack(rad_r, dim=0), torch.stack(rad_i, dim=0)


def main():
    torch.manual_seed(123)
    device = 'cuda'
    M, n_tx, n_rx, K = 20_000, 12, 16, 256
    n_chirps = 16
    w_threshold = 1e-20

    # Realistic magnitudes — phi in ~[-10π, 10π], n_peak in [0, K),
    # w in [0, ~5] with many near-zero.
    w_full   = torch.rand(M, n_tx, n_rx, device=device) * 5.0
    w_full  *= (torch.rand(M, n_tx, n_rx, device=device) > 0.3).float()
    phi_base = (torch.rand(M, n_tx, n_rx, device=device) - 0.5) * 20.0 * math.pi
    n_peak   = torch.rand(M, n_tx, n_rx, device=device) * float(K)
    A        = (torch.rand(M, device=device) - 0.5) * 5000.0
    t_off    = torch.arange(n_chirps, device=device, dtype=torch.float32).unsqueeze(-1) * 4.9e-4 \
               + torch.arange(n_tx, device=device, dtype=torch.float32).unsqueeze(0) * 4.1e-5
    psf_tab  = _make_psf_table(K=K, device=device)

    # Make w_full a leaf with grad for the backward check.
    w_full_ref = w_full.clone().requires_grad_(True)
    w_full_f   = w_full.clone().requires_grad_(True)

    # ----- Forward correctness -----
    rp_r_ref, rp_i_ref = _reference(
        w_full_ref, phi_base, n_peak, A, t_off, psf_tab, K, w_threshold)
    rp_r_f, rp_i_f = step5_doppler_fused(
        w_full_f, phi_base, n_peak, A, t_off,
        psf_tab.psf_real, psf_tab.psf_imag, K, w_threshold,
    )

    assert rp_r_ref.shape == rp_r_f.shape == (n_chirps, n_tx, n_rx, K)

    mag_ref = (rp_r_ref.pow(2) + rp_i_ref.pow(2)).sqrt()
    mag_f   = (rp_r_f.pow(2)   + rp_i_f.pow(2)).sqrt()
    abs_diff = (mag_ref - mag_f).abs()
    rel      = abs_diff.max() / mag_ref.abs().mean().clamp_min(1e-30)
    print('FORWARD:')
    print(f'  ref mag: mean={mag_ref.mean():.3f}  max={mag_ref.max():.3f}')
    print(f'  fused   mag: mean={mag_f.mean():.3f}  max={mag_f.max():.3f}')
    print(f'  |diff| mean={abs_diff.mean():.3e}  max={abs_diff.max():.3e}')
    print(f'  rel (max_abs / ref_mean) = {rel:.3e}')
    assert rel < 5e-3, f'forward mismatch: rel={rel:.3e}'
    print('  PASS')

    # ----- Backward correctness -----
    # Use a non-trivial upstream grad; compare grad_w tensors.
    g_rp_r = torch.randn_like(rp_r_ref)
    g_rp_i = torch.randn_like(rp_i_ref)

    (rp_r_ref * g_rp_r).sum().backward(retain_graph=True)
    (rp_i_ref * g_rp_i).sum().backward()
    grad_w_ref = w_full_ref.grad.clone()

    (rp_r_f * g_rp_r).sum().backward(retain_graph=True)
    (rp_i_f * g_rp_i).sum().backward()
    grad_w_f = w_full_f.grad.clone()

    b_abs = (grad_w_ref - grad_w_f).abs()
    b_rel = b_abs.max() / grad_w_ref.abs().mean().clamp_min(1e-30)
    print('BACKWARD:')
    print(f'  |grad_w|: ref_mean={grad_w_ref.abs().mean():.3e}  '
          f'fused_mean={grad_w_f.abs().mean():.3e}')
    print(f'  |diff| mean={b_abs.mean():.3e}  max={b_abs.max():.3e}')
    print(f'  rel (max_abs / ref_mean) = {b_rel:.3e}')
    assert b_rel < 5e-3, f'backward mismatch: rel={b_rel:.3e}'
    print('  PASS')

    print('\nstep5_doppler_fused: forward + backward match 16× step5_fused reference.')


if __name__ == '__main__':
    main()
