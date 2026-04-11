"""
Factorized Gaussian renderer with range-profile splatting (v3).

Steps 1-4 (BSDF, antenna, geometry): identical to v2 factorized renderer.
Step 5: splat into range profiles instead of synthesizing ADC.

Each Gaussian deposits its complex amplitude at ~5 range bins per (TX, RX)
channel, instead of computing cos/sin for all K=256 ADC samples.
The azimuth FFT is then applied on the range profiles (same as standard pipeline).
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from mm25DGS.bsdf_torch import (
    enforce_spm_validity, permittivity_to_ior,
    itu_slab_fresnel, compute_sp_basis,
    _get_default_pol, _cpx_abs_sq,
    K_WAVE, WAVELENGTH, INV_PI, TWO_PI,
    _sigmoid,
)

C_LIGHT = 299_792_458.0


def render_factorized(
    positions,          # (M, 3) Gaussian centers
    normals,            # (M, 3) surface normals
    areas,              # (M,) area weights (vertex_area × opacity)
    raw_materials,      # (M, 6) raw material params
    rast,               # RasterizerTorch instance (for constants + antenna)
    reparameterize_fn,  # reparameterize_torch function
    detach_phase=True,
    chunk_size=2000,    # unused (kept for API compat)
    shadow_mask=None,   # (M, n_tx) bool — False = occluded, zero weight
):
    """Range-profile splatting renderer. Returns (rp_real, rp_imag).

    Output shape: (n_tx, n_rx, K) complex range profiles.
    These must be passed through azimuth FFT (not range FFT) to get RA image.

    Cost: M BSDF evals + M×28 antenna + M×192×5 scatter ops.
    """
    device = positions.device
    M = positions.shape[0]
    n_tx = rast.n_tx
    n_rx = rast.n_rx
    K = rast.K

    if M == 0:
        return (torch.zeros(n_tx, n_rx, K, device=device),
                torch.zeros(n_tx, n_rx, K, device=device))

    # ================================================================
    # Step 1: Per-Gaussian material prep (M evaluations)
    # ================================================================
    physics = reparameterize_fn(raw_materials)
    eps_real = physics[:, 0]
    eps_imag = physics[:, 1]
    sigma_h = physics[:, 2]
    l_c = physics[:, 3]
    tau_base = physics[:, 4]
    thickness = physics[:, 5]

    sh, lc = enforce_spm_validity(sigma_h, l_c)

    # GGX alpha for KA lobe
    alpha_raw = 4.0 * math.pi * sh / WAVELENGTH
    alpha_ggx = torch.sqrt(alpha_raw.clamp(min=0.0025)).clamp(0.05, 0.95)
    alpha_sq = alpha_ggx ** 2

    # vMF parameters for SPM lobe
    l_c_lam = lc / WAVELENGTH
    roughness_slope = sh / lc.clamp(min=1e-8)
    kappa_SPM = (torch.sqrt(l_c_lam.clamp(min=0.0)) / (1.0 + 3.0 * roughness_slope)).clamp(0.5, 10.0)
    kappa_SPM_c = kappa_SPM.clamp(max=50.0)
    sinh_SPM = (torch.exp(kappa_SPM_c) - torch.exp(-kappa_SPM_c)) / 2.0
    norm_SPM = kappa_SPM / (4.0 * math.pi * sinh_SPM.clamp(min=1e-10))

    # SPM material modulation
    eps_contrast = torch.abs(eps_real - 1.0) + eps_imag
    eps_factor = (eps_contrast / 5.0).clamp(0.2, 1.0)

    # vMF parameters for directive lobe
    kappa_base_dir = torch.sqrt(l_c_lam.clamp(min=0.0))
    broadening = 1.0 + 5.0 * roughness_slope
    kappa_dir = (kappa_base_dir / broadening).clamp(0.3, 5.0)
    kappa_dir_c = kappa_dir.clamp(max=50.0)
    sinh_dir = (torch.exp(kappa_dir_c) - torch.exp(-kappa_dir_c)) / 2.0
    norm_dir = kappa_dir / (4.0 * math.pi * sinh_dir.clamp(min=1e-10))

    # Blend factors that depend on materials only
    roughness_ratio = sh / WAVELENGTH
    gamma = _sigmoid((0.05 - roughness_ratio) * 50.0).clamp(0.1, 0.9)  # (M,)

    # CBS mean (materials only)
    cbs_alpha = K_WAVE * lc
    cbs_mean = (1.0 + 1.0 / (1.0 + cbs_alpha)).clamp(min=1.0)

    # Radar constant
    C_radar = rast.radar_constant * rast.rx_dBFS_scale * rast.adc_scale

    # ================================================================
    # Step 2: Per-TX computation (M × n_tx)
    # ================================================================
    # Distances and directions: (M, n_tx)
    diff_tx = rast.tx_positions[None, :, :] - positions[:, None, :]  # (M, n_tx, 3)
    d_tx = diff_tx.norm(dim=-1).clamp(min=1e-6)                      # (M, n_tx)
    wi = diff_tx / d_tx.unsqueeze(-1)                                 # (M, n_tx, 3) hit->TX

    # Cosine of incidence angle (M, n_tx)
    cos_i = (wi * normals[:, None, :]).sum(-1)                        # may be negative
    # Double-sided: flip sign where cos_i < 0
    normal_sign = torch.where(cos_i < 0, -torch.ones_like(cos_i), torch.ones_like(cos_i))
    cos_i = cos_i.abs().clamp(min=1e-6)
    # Effective normals per (M, n_tx) for reflection computation
    n_eff = normals[:, None, :] * normal_sign.unsqueeze(-1)           # (M, n_tx, 3)

    # Specular reflection of wi about n_eff: wi_r = 2(wi·n)n - wi
    wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)                     # (M, n_tx, 1)
    wi_r = 2.0 * wi_dot_n * n_eff - wi                               # (M, n_tx, 3)

    # Retroreflection direction for CBS
    retro = -wi_r                                                      # (M, n_tx, 3)

    # Smith G1 for incidence (KA)
    tan_sq_i = (1.0 / cos_i.clamp(min=1e-6) ** 2) - 1.0
    lambda_i = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_i).clamp(min=0.0))) / 2.0

    # Jones Fresnel: wi-dependent terms
    # s/p basis from incidence direction
    # compute_sp_basis expects (*, 3) — reshape to (M*n_tx, 3)
    wi_flat = wi.reshape(-1, 3)
    n_eff_flat = n_eff.reshape(-1, 3)
    s_in_flat, p_in_flat = compute_sp_basis(wi_flat, n_eff_flat)
    s_in = s_in_flat.reshape(M, n_tx, 3)                              # (M, n_tx, 3)
    p_in = p_in_flat.reshape(M, n_tx, 3)

    # TX polarization projection
    tx_pol = _get_default_pol(device)  # (3,)
    tx_s = (tx_pol * s_in).sum(-1)                                     # (M, n_tx)
    tx_p = (tx_pol * p_in).sum(-1)                                     # (M, n_tx)

    # Fresnel coefficients (depend on cos_theta_i and materials)
    cos_i_flat = cos_i.reshape(-1)
    eps_r_flat = eps_real[:, None].expand(-1, n_tx).reshape(-1)
    eps_i_flat = eps_imag[:, None].expand(-1, n_tx).reshape(-1)
    thick_flat = thickness[:, None].expand(-1, n_tx).reshape(-1)

    R_TE, R_TM, _, _ = itu_slab_fresnel(eps_r_flat, eps_i_flat, cos_i_flat, thick_flat)
    r_s = R_TE.reshape(M, n_tx)
    r_p = R_TM.reshape(M, n_tx)

    # Reflected field (complex): E_s_out = r_s × tx_s, E_p_out = r_p × tx_p
    E_s_out = r_s * tx_s.to(torch.complex64)                          # (M, n_tx) complex
    E_p_out = r_p * tx_p.to(torch.complex64)                          # (M, n_tx) complex

    # Blend factors that depend on cos_theta_i (M, n_tx)
    g_coh = (2.0 * K_WAVE * sh[:, None] * cos_i) ** 2
    eta = torch.exp(-g_coh).clamp(0.01, 0.99)                         # (M, n_tx)

    tau_angle = _sigmoid((cos_i - 0.94) * 20.0)
    kl = K_WAVE * lc
    v_KA = _sigmoid((kl - 6.0) * 2.0)
    kh = K_WAVE * sh
    v_SPM = _sigmoid((0.3 - kh) * 10.0)
    w_KA = tau_angle * tau_base[:, None] * v_KA[:, None]
    w_SPM = (1.0 - tau_angle) * (1.0 - tau_base[:, None]) * v_SPM[:, None]
    tau_eff = (w_KA / (w_KA + w_SPM).clamp(min=1e-6)).clamp(0.01, 0.99)  # (M, n_tx)

    # TX antenna gain (M, n_tx)
    tx_bore_exp = rast.tx_boresights[None, :, :].expand(M, -1, -1)    # (M, n_tx, 3)
    G_tx = rast.tx_antenna.evaluate(
        wi.reshape(-1, 3), tx_bore_exp.reshape(-1, 3)
    ).reshape(M, n_tx)

    # TX amplitude factor: sqrt(G_tx) / d_tx
    alpha_tx = torch.sqrt(G_tx.clamp(min=0.0)) / d_tx.clamp(min=1e-4)  # (M, n_tx)

    # TX phase factors
    tau_tx = d_tx / C_LIGHT                                            # (M, n_tx)
    phi_tx_const = TWO_PI * rast.center_freq * tau_tx                  # (M, n_tx)
    phi_tx_slope = TWO_PI * rast.slope * tau_tx                        # (M, n_tx)

    # ================================================================
    # Step 3: Per-RX computation (M × n_rx)
    # ================================================================
    diff_rx = positions[:, None, :] - rast.rx_positions[None, :, :]    # (M, n_rx, 3)
    d_rx = diff_rx.norm(dim=-1).clamp(min=1e-6)                        # (M, n_rx)
    dir_hit_to_rx = -diff_rx / d_rx.unsqueeze(-1)                      # (M, n_rx, 3) hit->RX
    dir_rx_to_hit = diff_rx / d_rx.unsqueeze(-1)                       # (M, n_rx, 3) RX->hit

    # cos(theta_rx) for RX-sphere projection (use mean normal for double-sided)
    # We need this per-RX. Use the raw normals (not flipped per-TX).
    cos_o_raw = (dir_hit_to_rx * normals[:, None, :]).sum(-1)          # (M, n_rx)
    cos_o = cos_o_raw.abs().clamp(min=1e-6)                            # (M, n_rx) double-sided

    # RX-sphere solid angle: dΩ = A × |cos(θ_rx)| / d_rx²
    dOmega = areas[:, None] * cos_o / (d_rx.clamp(min=1e-4) ** 2)     # (M, n_rx)

    # Smith G1 for observation (KA)
    tan_sq_o = (1.0 / cos_o.clamp(min=1e-6) ** 2) - 1.0
    lambda_o = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_o).clamp(min=0.0))) / 2.0

    # RX Jones: p_out depends on wo
    # s_out = s_in (same for all RX, but varies per TX)
    # p_out = cross(s_out, wo) — varies per (TX, RX)
    # We'll handle this in the inner loop (Step 4)

    # RX polarization projections: rx_s = rx_pol · s_out (same as s_in, per TX)
    rx_pol = _get_default_pol(device)
    # rx_s depends on s_in which varies per TX — computed in inner loop
    # But s_in is the same across RX for a given TX, so rx_s is per-TX only

    # RX antenna gain (M, n_rx)
    rx_bore_exp = rast.rx_boresights[None, :, :].expand(M, -1, -1)
    G_rx = rast.rx_antenna.evaluate(
        dir_rx_to_hit.reshape(-1, 3), rx_bore_exp.reshape(-1, 3)
    ).reshape(M, n_rx)

    # RX amplitude factor: sqrt(G_rx × dΩ)
    alpha_rx = torch.sqrt((G_rx * dOmega).clamp(min=0.0))             # (M, n_rx)

    # RX phase factors
    tau_rx = d_rx / C_LIGHT                                            # (M, n_rx)
    phi_rx_const = TWO_PI * rast.center_freq * tau_rx                  # (M, n_rx)
    phi_rx_slope = TWO_PI * rast.slope * tau_rx                        # (M, n_rx)

    # ================================================================
    # Step 4: Per-(TX, RX) lightweight BSDF inner loop
    # Compute f_total(t,r) for all M × n_tx × n_rx using precomputed terms
    # ================================================================

    # wo directions: (M, n_rx, 3) — dir_hit_to_rx
    wo = dir_hit_to_rx  # (M, n_rx, 3)

    # We need f_bsdf(t,r) for each (M, n_tx, n_rx).
    # Precomputed per-TX: wi_r (M, n_tx, 3), retro (M, n_tx, 3),
    #   E_s_out (M, n_tx), E_p_out (M, n_tx), eta (M, n_tx), tau_eff (M, n_tx),
    #   lambda_i (M, n_tx), s_in (M, n_tx, 3), cos_i (M, n_tx)
    # Precomputed per-RX: cos_o (M, n_rx), lambda_o (M, n_rx), wo (M, n_rx, 3)
    # Precomputed per-Gaussian: alpha_sq (M,), kappa_SPM/dir (M,), norm_SPM/dir (M,),
    #   eps_factor (M,), gamma (M,), cbs_mean (M,)

    # --- KA lobe: need half vector h = norm(wo + wi), then h·n ---
    # wi: (M, n_tx, 3), wo: (M, n_rx, 3)
    # h: (M, n_tx, n_rx, 3) — this is the 192× expansion, but only for h·n (scalar)
    # h·n = (wo + wi)·n / |wo + wi|
    # Numerator: wo·n + wi·n = cos_o_signed + cos_i_signed
    # But we need to use the TX-specific flipped normal for wi·n consistency.
    #
    # More carefully: h = normalize(wo + wi). h·n depends on the actual vectors.
    # We can compute h·n without forming the full (M, n_tx, n_rx, 3) tensor:
    #
    # (wo + wi)·n = wo·n + wi·n
    # |wo + wi|² = |wo|² + |wi|² + 2(wo·wi) = 2 + 2(wo·wi)  (unit vectors)
    # |wo + wi| = sqrt(2 + 2 wo·wi)
    #
    # wo·wi: (M, n_tx, n_rx) via einsum
    wo_dot_wi = torch.einsum('mrj,mtj->mtr', wo, wi)                  # (M, n_tx, n_rx)

    # wo·n_eff and wi·n_eff (using TX-flipped normals)
    # wi·n_eff = cos_i (already computed, (M, n_tx))
    # wo·n_eff: need per (M, n_tx, n_rx) since n_eff varies per TX
    wo_dot_n = torch.einsum('mrj,mtj->mtr', wo, n_eff)                # (M, n_tx, n_rx)

    h_dot_n_num = wo_dot_n + cos_i.unsqueeze(-1)                      # (M, n_tx, n_rx)
    h_len = torch.sqrt((2.0 + 2.0 * wo_dot_wi).clamp(min=1e-10))     # (M, n_tx, n_rx)
    h_dot_n = (h_dot_n_num / h_len).clamp(min=0.0)                    # (M, n_tx, n_rx)

    # GGX NDF: D = α² / (π ((h·n)²(α²-1)+1)²)
    denom_ndf = (h_dot_n ** 2 * (alpha_sq[:, None, None] - 1.0) + 1.0) ** 2
    D_KA = alpha_sq[:, None, None] / (math.pi * denom_ndf.clamp(min=1e-20))

    # Smith G (joint form, exact): G = 1 / (1 + Λ_i + Λ_o)
    G_KA = 1.0 / (1.0 + lambda_i.unsqueeze(-1) + lambda_o.unsqueeze(-2)).clamp(min=1e-10)

    # Cook-Torrance: f_KA = D × G / (4 cos_i cos_o)
    f_KA = D_KA * G_KA / (4.0 * cos_i.unsqueeze(-1) * cos_o.unsqueeze(-2)).clamp(min=1e-10)

    # --- SPM lobe: cos_dev = wo · wi_r ---
    cos_dev = torch.einsum('mrj,mtj->mtr', wo, wi_r)                  # (M, n_tx, n_rx)
    cos_dev = cos_dev.clamp(-1.0, 1.0)
    f_SPM = norm_SPM[:, None, None] * torch.exp(kappa_SPM[:, None, None] * (cos_dev - 1.0))
    f_SPM = f_SPM * eps_factor[:, None, None]

    # --- Directive lobe: same cos_dev ---
    f_dir = norm_dir[:, None, None] * torch.exp(kappa_dir[:, None, None] * (cos_dev - 1.0))

    # --- Broad lobe ---
    f_broad = INV_PI  # scalar constant

    # --- CBS factor ---
    cos_bs = torch.einsum('mrj,mtj->mtr', wo, retro)                  # (M, n_tx, n_rx)
    cos_bs = cos_bs.clamp(-1.0, 1.0)
    sin_bs = torch.sqrt((1.0 - cos_bs ** 2).clamp(min=1e-20))
    x_cbs = K_WAVE * lc[:, None, None] * sin_bs
    sinc_x = torch.where(x_cbs.abs() > 1e-6, torch.sin(x_cbs) / x_cbs, torch.ones_like(x_cbs))
    cbs = 1.0 + sinc_x ** 2

    # --- Coherent + incoherent blend ---
    f_coh = tau_eff.unsqueeze(-1) * f_KA + (1.0 - tau_eff.unsqueeze(-1)) * f_SPM
    f_inc_raw = gamma[:, None, None] * f_dir + (1.0 - gamma[:, None, None]) * f_broad
    f_inc = f_inc_raw * cbs / cbs_mean[:, None, None]
    f_lobe = eta.unsqueeze(-1) * f_coh + (1.0 - eta.unsqueeze(-1)) * f_inc  # (M, n_tx, n_rx)

    # --- Jones Fresnel power per (TX, RX) ---
    # E_s_out: (M, n_tx) complex — from TX
    # rx_s: rx_pol · s_in — s_in varies per TX: (M, n_tx, 3)
    rx_s = (rx_pol * s_in).sum(-1)                                     # (M, n_tx) -- same for all RX

    # p_out = cross(s_in, wo) -- varies per (TX, RX)
    # s_in: (M, n_tx, 3), wo: (M, n_rx, 3)
    # p_out: (M, n_tx, n_rx, 3) — this is the 192× expansion but just for a cross product
    # rx_p = rx_pol · p_out = rx_pol · cross(s_in, wo)
    # Using triple product: rx_pol · (s_in × wo) = wo · (rx_pol × s_in)
    rx_pol_cross_s = torch.cross(
        rx_pol.expand_as(s_in), s_in, dim=-1)                          # (M, n_tx, 3)
    # rx_p = wo · (rx_pol × s_in) for each (M, n_tx, n_rx)
    rx_p = torch.einsum('mrj,mtj->mtr', wo, rx_pol_cross_s)           # (M, n_tx, n_rx)
    # Normalize p_out: |cross(s_in, wo)| — but rx_p should use normalized p_out
    # |s_in × wo| = sin(angle between s_in and wo)
    p_out_len = torch.einsum('mrj,mtj->mtr', wo, wo).unsqueeze(-1)    # dummy
    # Actually: cross(s, wo) has magnitude |s||wo|sin(θ) = sin(θ) (unit vecs)
    # We need rx_p / |cross(s_in, wo)|
    s_cross_wo_sq = 1.0 - torch.einsum('mrj,mtj->mtr', wo, s_in) ** 2  # sin²(θ)
    p_out_norm = torch.sqrt(s_cross_wo_sq.clamp(min=1e-12))
    rx_p = rx_p / p_out_norm                                           # (M, n_tx, n_rx)

    # R_jones = |E_s_out × rx_s + E_p_out × rx_p|²
    # E_s_out: (M, n_tx) complex, rx_s: (M, n_tx) real -> broadcast to (M, n_tx, 1)
    # E_p_out: (M, n_tx) complex, rx_p: (M, n_tx, n_rx) real
    E_rx = E_s_out.unsqueeze(-1) * rx_s.unsqueeze(-1).to(torch.complex64) \
         + E_p_out.unsqueeze(-1) * rx_p.to(torch.complex64)           # (M, n_tx, n_rx) complex
    R_jones = _cpx_abs_sq(E_rx).clamp(0.0, 1.0)                       # (M, n_tx, n_rx)

    # --- Final BSDF: f_cos = R_jones × f_lobe × cos_theta_i ---
    f_cos = R_jones * f_lobe * cos_i.unsqueeze(-1)                    # (M, n_tx, n_rx)

    # ================================================================
    # Step 5: Range-profile splatting
    # ================================================================
    # Instead of computing cos/sin for K=256 ADC samples per path,
    # deposit each Gaussian's amplitude at ~5 range bins per channel.

    from mm25DGS_v3.psf import hann_psf

    # Weight per (M, n_tx, n_rx)
    sqrt_f_cos = torch.sqrt(f_cos.clamp(min=1e-20))
    w_full = C_radar * sqrt_f_cos * alpha_tx.unsqueeze(-1) * alpha_rx.unsqueeze(-2)

    # Zero out back-facing and occluded paths
    active_i = cos_i > 1e-6
    if shadow_mask is not None:
        active_i = active_i & shadow_mask
    active_o = cos_o > 1e-6
    active = active_i.unsqueeze(-1) & active_o.unsqueeze(-2)
    w_full = w_full * active.float()

    # Range bin for each (M, n_tx, n_rx) path:
    # n_peak = (d_tx + d_rx) × S × K / (c × fs)
    #        = phi_slope_total / (2π × fs / K)  [since phi_slope = 2πSτ and τ = d/c]
    # Or equivalently: n_peak = (d_tx + d_rx) / (2 × range_res)
    # where range_res = c / (2 × BW) = c × fs / (2 × S × K)
    d_total = d_tx.unsqueeze(-1) + d_rx.unsqueeze(-2)          # (M, n_tx, n_rx)
    range_res = C_LIGHT * rast.sample_rate / (2.0 * rast.slope * K)
    n_peak = d_total / (2.0 * range_res)                        # (M, n_tx, n_rx) real-valued

    # Carrier phase: 2π × f0 × τ  (encodes azimuth information)
    phi_carrier = phi_tx_const.unsqueeze(-1) + phi_rx_const.unsqueeze(-2)  # (M, n_tx, n_rx)

    if detach_phase:
        n_peak = n_peak.detach()
        phi_carrier = phi_carrier.detach()

    # Splat to range bins with Hann PSF (precomputed lookup table)
    SPREAD = 15

    from mm25DGS_v3.psf import HannPSFTable

    # Lazy-init PSF table (created once, reused across calls)
    if not hasattr(render_factorized, '_psf_table') or \
       render_factorized._psf_table.K != K or \
       render_factorized._psf_table.spread != SPREAD or \
       render_factorized._psf_table.device != str(device):
        render_factorized._psf_table = HannPSFTable(K, SPREAD, n_grid=1024, device=str(device))
    psf_table = render_factorized._psf_table

    n_floor = n_peak.floor().long()
    n_frac = n_peak - n_floor.float()

    # Flatten all (M, n_tx, n_rx) paths
    w_flat = w_full.reshape(-1)
    phi_flat = phi_carrier.reshape(-1)
    n_floor_flat = n_floor.reshape(-1)
    n_frac_flat = n_frac.reshape(-1)

    # Active paths only
    active_paths = w_flat > 1e-20
    w_act = w_flat[active_paths]
    phi_act = phi_flat[active_paths]
    n_floor_act = n_floor_flat[active_paths]
    n_frac_act = n_frac_flat[active_paths]

    # Channel indices for active paths
    t_idx_full = torch.arange(n_tx, device=device).unsqueeze(0).unsqueeze(-1).expand(M, -1, n_rx).reshape(-1)
    r_idx_full = torch.arange(n_rx, device=device).unsqueeze(0).unsqueeze(0).expand(M, n_tx, -1).reshape(-1)
    base_idx = t_idx_full[active_paths] * (n_rx * K) + r_idx_full[active_paths] * K

    # Carrier phasor
    carrier_real = w_act * torch.cos(phi_act)
    carrier_imag = w_act * torch.sin(phi_act)

    # PSF lookup (table interpolation — no trig at runtime)
    psf_r, psf_i = psf_table.evaluate(n_frac_act)                 # (SPREAD, P_active)

    # Bin indices: (SPREAD, P_active)
    dn_offsets = torch.arange(-(SPREAD // 2), SPREAD // 2 + 1, device=device)
    bin_all = (n_floor_act[None, :] + dn_offsets[:, None]) % K
    flat_idx_all = base_idx[None, :] + bin_all

    # Contributions: carrier × PSF
    contrib_real_all = carrier_real[None, :] * psf_r - carrier_imag[None, :] * psf_i
    contrib_imag_all = carrier_real[None, :] * psf_i + carrier_imag[None, :] * psf_r

    # Single scatter_add
    rp_real = torch.zeros(n_tx, n_rx, K, device=device)
    rp_imag = torch.zeros(n_tx, n_rx, K, device=device)
    rp_real.view(-1).scatter_add_(0, flat_idx_all.reshape(-1), contrib_real_all.reshape(-1))
    rp_imag.view(-1).scatter_add_(0, flat_idx_all.reshape(-1), contrib_imag_all.reshape(-1))

    return rp_real, rp_imag
