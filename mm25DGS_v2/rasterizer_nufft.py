"""
NUFFT-accelerated Gaussian-to-ADC renderer (Option C).

Uses the factorized BSDF (exact, Option 4) from rasterizer_factorized.py
for the amplitude computation, then replaces the einsum phase accumulation
with a batched GPU NUFFT.

The NUFFT computes: F[k] = Σ_m c[m] × exp(j × ω[m] × k) in O(M + K log K)
instead of O(M × K) for the direct sum.

For M=12K Gaussians, K=256 ADC samples, 192 channels:
  Direct:  M × K × 192 = 589M multiply-adds
  NUFFT:   192 × (M + K log K) ≈ 2.7M multiply-adds (~200× less)
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
import pytorch_finufft.functional as finufft

from mm25DGS.bsdf_torch import (
    enforce_spm_validity, permittivity_to_ior,
    itu_slab_fresnel, compute_sp_basis,
    _get_default_pol, _cpx_abs_sq,
    K_WAVE, WAVELENGTH, INV_PI, TWO_PI,
    _sigmoid,
)

C_LIGHT = 299_792_458.0
PI = math.pi


def render_nufft(
    positions,          # (M, 3) Gaussian centers
    normals,            # (M, 3) surface normals
    areas,              # (M,) area weights (vertex_area × opacity)
    raw_materials,      # (M, 6) raw material params
    rast,               # RasterizerTorch instance
    reparameterize_fn,  # reparameterize_torch function
    detach_phase=True,
    chunk_size=4000,    # chunk over Gaussians for BSDF (not for NUFFT)
):
    """NUFFT-accelerated Gaussian renderer.

    Steps 1-4 (BSDF, antenna, geometry) are identical to rasterizer_factorized.py.
    Step 5 (phase accumulation) uses batched NUFFT instead of einsum.

    Returns (adc_real, adc_imag) each of shape (N_tx, N_rx, K).
    """
    device = positions.device
    M = positions.shape[0]
    n_tx = rast.n_tx
    n_rx = rast.n_rx
    K = rast.K
    n_chan = n_tx * n_rx  # 192

    if M == 0:
        return (torch.zeros(n_tx, n_rx, K, device=device),
                torch.zeros(n_tx, n_rx, K, device=device))

    # ================================================================
    # Steps 1-4: Identical to rasterizer_factorized.py
    # Compute w_full (M, n_tx, n_rx), phi_const (M, n_tx, n_rx), phi_slope (M, n_tx, n_rx)
    # ================================================================
    w_full, phi_const_total, phi_slope_total = _compute_bsdf_and_phase(
        positions, normals, areas, raw_materials, rast, reparameterize_fn,
        detach_phase=detach_phase)

    # ================================================================
    # Step 5: NUFFT phase accumulation
    # ================================================================
    # Per-channel NUFFT: each (t,r) channel has different frequencies
    # because τ(m,t,r) = (d_tx(m,t) + d_rx(m,r)) / c differs per channel.
    # We loop over channels and call 1D NUFFT for each.

    adc_real = torch.zeros(n_tx, n_rx, K, device=device)
    adc_imag = torch.zeros(n_tx, n_rx, K, device=device)

    for t in range(n_tx):
        for r in range(n_rx):
            w_tr = w_full[:, t, r]                                 # (M,)
            pc = phi_const_total[:, t, r]                          # (M,)
            ps = phi_slope_total[:, t, r]                          # (M,)

            # Skip channels with negligible weight
            if w_tr.max() < 1e-15:
                continue

            # NUFFT frequency: ω = phi_slope / sample_rate, wrapped to [-π, π)
            omega = ps / rast.sample_rate
            omega_wrapped = ((omega + PI) % TWO_PI) - PI           # (M,)

            if detach_phase:
                omega_wrapped = omega_wrapped.detach()

            # Phase shift for index convention: NUFFT outputs at -K/2..K/2-1
            # We want k=0..K-1. Shift: c' = c × exp(j × K/2 × ω)
            phase_shift = (K // 2) * omega_wrapped
            total_phase = pc + phase_shift
            c_complex = torch.complex(
                w_tr * torch.cos(total_phase),
                w_tr * torch.sin(total_phase))                     # (M,) complex

            # 1D NUFFT Type 1: (M,) -> (K,)
            # points: (1, M), values: (1, M) -> output: (1, K)
            result = finufft.finufft_type1(
                omega_wrapped.unsqueeze(0),    # (1, M)
                c_complex.unsqueeze(0),        # (1, M)
                (K,),
                isign=1,                       # exp(+j n x) convention
            ).squeeze(0)                       # (K,) complex

            # Shift from [-K/2, K/2-1] to [0, K-1]
            result = torch.fft.fftshift(result)

            adc_real[t, r] = result.real
            adc_imag[t, r] = result.imag

    return adc_real, adc_imag


def _compute_bsdf_and_phase(
    positions, normals, areas, raw_materials, rast, reparameterize_fn,
    detach_phase=True,
):
    """Compute BSDF weights and phase parameters for all (M, n_tx, n_rx).

    Returns:
        w_full: (M, n_tx, n_rx) amplitude weights
        phi_const_total: (M, n_tx, n_rx) constant phase (2π f0 τ)
        phi_slope_total: (M, n_tx, n_rx) slope phase (2π S τ)
    """
    device = positions.device
    M = positions.shape[0]
    n_tx = rast.n_tx
    n_rx = rast.n_rx

    # --- Step 1: Material prep (M) ---
    physics = reparameterize_fn(raw_materials)
    eps_real, eps_imag = physics[:, 0], physics[:, 1]
    sigma_h, l_c = physics[:, 2], physics[:, 3]
    tau_base, thickness = physics[:, 4], physics[:, 5]

    sh, lc = enforce_spm_validity(sigma_h, l_c)

    alpha_raw = 4.0 * PI * sh / WAVELENGTH
    alpha_ggx = torch.sqrt(alpha_raw.clamp(min=0.0025)).clamp(0.05, 0.95)
    alpha_sq = alpha_ggx ** 2

    l_c_lam = lc / WAVELENGTH
    roughness_slope = sh / lc.clamp(min=1e-8)
    kappa_SPM = (torch.sqrt(l_c_lam.clamp(min=0.0)) / (1.0 + 3.0 * roughness_slope)).clamp(0.5, 10.0)
    kappa_SPM_c = kappa_SPM.clamp(max=50.0)
    sinh_SPM = (torch.exp(kappa_SPM_c) - torch.exp(-kappa_SPM_c)) / 2.0
    norm_SPM = kappa_SPM / (4.0 * PI * sinh_SPM.clamp(min=1e-10))
    eps_contrast = torch.abs(eps_real - 1.0) + eps_imag
    eps_factor = (eps_contrast / 5.0).clamp(0.2, 1.0)

    kappa_base_dir = torch.sqrt(l_c_lam.clamp(min=0.0))
    broadening = 1.0 + 5.0 * roughness_slope
    kappa_dir = (kappa_base_dir / broadening).clamp(0.3, 5.0)
    kappa_dir_c = kappa_dir.clamp(max=50.0)
    sinh_dir = (torch.exp(kappa_dir_c) - torch.exp(-kappa_dir_c)) / 2.0
    norm_dir = kappa_dir / (4.0 * PI * sinh_dir.clamp(min=1e-10))

    roughness_ratio = sh / WAVELENGTH
    gamma = _sigmoid((0.05 - roughness_ratio) * 50.0).clamp(0.1, 0.9)
    cbs_alpha = K_WAVE * lc
    cbs_mean = (1.0 + 1.0 / (1.0 + cbs_alpha)).clamp(min=1.0)
    C_radar = rast.radar_constant * rast.rx_dBFS_scale * rast.adc_scale

    # --- Step 2: Per-TX (M, n_tx) ---
    diff_tx = rast.tx_positions[None, :, :] - positions[:, None, :]
    d_tx = diff_tx.norm(dim=-1).clamp(min=1e-6)
    wi = diff_tx / d_tx.unsqueeze(-1)

    cos_i = (wi * normals[:, None, :]).sum(-1)
    normal_sign = torch.where(cos_i < 0, -torch.ones_like(cos_i), torch.ones_like(cos_i))
    cos_i = cos_i.abs().clamp(min=1e-6)
    n_eff = normals[:, None, :] * normal_sign.unsqueeze(-1)

    wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)
    wi_r = 2.0 * wi_dot_n * n_eff - wi
    retro = -wi_r

    tan_sq_i = (1.0 / cos_i.clamp(min=1e-6) ** 2) - 1.0
    lambda_i = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_i).clamp(min=0.0))) / 2.0

    wi_flat = wi.reshape(-1, 3)
    n_eff_flat = n_eff.reshape(-1, 3)
    s_in_flat, p_in_flat = compute_sp_basis(wi_flat, n_eff_flat)
    s_in = s_in_flat.reshape(M, n_tx, 3)
    p_in = p_in_flat.reshape(M, n_tx, 3)

    tx_pol = _get_default_pol(device)
    tx_s = (tx_pol * s_in).sum(-1)
    tx_p = (tx_pol * p_in).sum(-1)

    cos_i_flat = cos_i.reshape(-1)
    eps_r_flat = eps_real[:, None].expand(-1, n_tx).reshape(-1)
    eps_i_flat = eps_imag[:, None].expand(-1, n_tx).reshape(-1)
    thick_flat = thickness[:, None].expand(-1, n_tx).reshape(-1)
    R_TE, R_TM, _, _ = itu_slab_fresnel(eps_r_flat, eps_i_flat, cos_i_flat, thick_flat)
    r_s = R_TE.reshape(M, n_tx)
    r_p = R_TM.reshape(M, n_tx)
    E_s_out = r_s * tx_s.to(torch.complex64)
    E_p_out = r_p * tx_p.to(torch.complex64)

    g_coh = (2.0 * K_WAVE * sh[:, None] * cos_i) ** 2
    eta = torch.exp(-g_coh).clamp(0.01, 0.99)
    tau_angle = _sigmoid((cos_i - 0.94) * 20.0)
    kl = K_WAVE * lc
    v_KA = _sigmoid((kl - 6.0) * 2.0)
    kh = K_WAVE * sh
    v_SPM = _sigmoid((0.3 - kh) * 10.0)
    w_KA = tau_angle * tau_base[:, None] * v_KA[:, None]
    w_SPM = (1.0 - tau_angle) * (1.0 - tau_base[:, None]) * v_SPM[:, None]
    tau_eff = (w_KA / (w_KA + w_SPM).clamp(min=1e-6)).clamp(0.01, 0.99)

    tx_bore_exp = rast.tx_boresights[None, :, :].expand(M, -1, -1)
    G_tx = rast.tx_antenna.evaluate(wi.reshape(-1, 3), tx_bore_exp.reshape(-1, 3)).reshape(M, n_tx)
    alpha_tx = torch.sqrt(G_tx.clamp(min=0.0)) / d_tx.clamp(min=1e-4)

    tau_tx = d_tx / C_LIGHT
    phi_tx_const = TWO_PI * rast.center_freq * tau_tx
    phi_tx_slope = TWO_PI * rast.slope * tau_tx

    # --- Step 3: Per-RX (M, n_rx) ---
    diff_rx = positions[:, None, :] - rast.rx_positions[None, :, :]
    d_rx = diff_rx.norm(dim=-1).clamp(min=1e-6)
    dir_hit_to_rx = -diff_rx / d_rx.unsqueeze(-1)
    dir_rx_to_hit = diff_rx / d_rx.unsqueeze(-1)

    cos_o = (dir_hit_to_rx * normals[:, None, :]).sum(-1).abs().clamp(min=1e-6)
    dOmega = areas[:, None] * cos_o / (d_rx.clamp(min=1e-4) ** 2)

    tan_sq_o = (1.0 / cos_o.clamp(min=1e-6) ** 2) - 1.0
    lambda_o = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_o).clamp(min=0.0))) / 2.0

    rx_bore_exp = rast.rx_boresights[None, :, :].expand(M, -1, -1)
    G_rx = rast.rx_antenna.evaluate(dir_rx_to_hit.reshape(-1, 3), rx_bore_exp.reshape(-1, 3)).reshape(M, n_rx)
    alpha_rx = torch.sqrt((G_rx * dOmega).clamp(min=0.0))

    tau_rx = d_rx / C_LIGHT
    phi_rx_const = TWO_PI * rast.center_freq * tau_rx
    phi_rx_slope = TWO_PI * rast.slope * tau_rx

    # --- Step 4: Per-(TX, RX) lightweight BSDF (M, n_tx, n_rx) ---
    wo = dir_hit_to_rx
    wo_dot_wi = torch.einsum('mrj,mtj->mtr', wo, wi)
    wo_dot_n = torch.einsum('mrj,mtj->mtr', wo, n_eff)
    h_dot_n_num = wo_dot_n + cos_i.unsqueeze(-1)
    h_len = torch.sqrt((2.0 + 2.0 * wo_dot_wi).clamp(min=1e-10))
    h_dot_n = (h_dot_n_num / h_len).clamp(min=0.0)

    denom_ndf = (h_dot_n ** 2 * (alpha_sq[:, None, None] - 1.0) + 1.0) ** 2
    D_KA = alpha_sq[:, None, None] / (PI * denom_ndf.clamp(min=1e-20))
    G_KA = 1.0 / (1.0 + lambda_i.unsqueeze(-1) + lambda_o.unsqueeze(-2)).clamp(min=1e-10)
    f_KA = D_KA * G_KA / (4.0 * cos_i.unsqueeze(-1) * cos_o.unsqueeze(-2)).clamp(min=1e-10)

    cos_dev = torch.einsum('mrj,mtj->mtr', wo, wi_r).clamp(-1.0, 1.0)
    f_SPM = norm_SPM[:, None, None] * torch.exp(kappa_SPM[:, None, None] * (cos_dev - 1.0)) * eps_factor[:, None, None]
    f_dir = norm_dir[:, None, None] * torch.exp(kappa_dir[:, None, None] * (cos_dev - 1.0))

    cos_bs = torch.einsum('mrj,mtj->mtr', wo, retro).clamp(-1.0, 1.0)
    sin_bs = torch.sqrt((1.0 - cos_bs ** 2).clamp(min=1e-20))
    x_cbs = K_WAVE * lc[:, None, None] * sin_bs
    sinc_x = torch.where(x_cbs.abs() > 1e-6, torch.sin(x_cbs) / x_cbs, torch.ones_like(x_cbs))
    cbs = 1.0 + sinc_x ** 2

    f_coh = tau_eff.unsqueeze(-1) * f_KA + (1.0 - tau_eff.unsqueeze(-1)) * f_SPM
    f_inc_raw = gamma[:, None, None] * f_dir + (1.0 - gamma[:, None, None]) * INV_PI
    f_inc = f_inc_raw * cbs / cbs_mean[:, None, None]
    f_lobe = eta.unsqueeze(-1) * f_coh + (1.0 - eta.unsqueeze(-1)) * f_inc

    rx_pol = _get_default_pol(device)
    rx_s = (rx_pol * s_in).sum(-1)
    rx_pol_cross_s = torch.cross(rx_pol.expand_as(s_in), s_in, dim=-1)
    rx_p = torch.einsum('mrj,mtj->mtr', wo, rx_pol_cross_s)
    s_cross_wo_sq = 1.0 - torch.einsum('mrj,mtj->mtr', wo, s_in) ** 2
    p_out_norm = torch.sqrt(s_cross_wo_sq.clamp(min=1e-12))
    rx_p = rx_p / p_out_norm

    E_rx = E_s_out.unsqueeze(-1) * rx_s.unsqueeze(-1).to(torch.complex64) \
         + E_p_out.unsqueeze(-1) * rx_p.to(torch.complex64)
    R_jones = _cpx_abs_sq(E_rx).clamp(0.0, 1.0)

    f_cos = R_jones * f_lobe * cos_i.unsqueeze(-1)

    sqrt_f_cos = torch.sqrt(f_cos.clamp(min=1e-20))
    w_full = C_radar * sqrt_f_cos * alpha_tx.unsqueeze(-1) * alpha_rx.unsqueeze(-2)

    active_i = cos_i > 1e-6
    active_o = cos_o > 1e-6
    active = active_i.unsqueeze(-1) & active_o.unsqueeze(-2)
    w_full = w_full * active.float()

    # --- Phase totals (M, n_tx, n_rx) ---
    phi_const_total = phi_tx_const.unsqueeze(-1) + phi_rx_const.unsqueeze(-2)
    phi_slope_total = phi_tx_slope.unsqueeze(-1) + phi_rx_slope.unsqueeze(-2)

    if detach_phase:
        phi_const_total = phi_const_total.detach()
        phi_slope_total = phi_slope_total.detach()

    return w_full, phi_const_total, phi_slope_total
