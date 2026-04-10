"""
Factorized Gaussian-to-ADC renderer.

Eliminates the 192× per-path expansion by factorizing:
  - Phase: exp(jφ) = exp(jφ_tx) × exp(jφ_rx)  [EXACT]
  - BSDF: precompute wi-dependent terms per TX, evaluate lightweight
           wo-dependent terms per (TX,RX) pair  [EXACT, Option 4]
  - Antenna: evaluate per-TX and per-RX separately
  - Accumulation: einsum outer product instead of scatter_add

The rendering equation is algebraically equivalent to the per-path
renderer (_render_chunk_torch) — no approximations.
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
    chunk_size=2000,
):
    """Factorized Gaussian renderer. Returns (adc_real, adc_imag).

    Cost: M×1 BSDF material prep + M×12 TX prep + M×16 RX prep
          + M×192 lightweight inner loop + chunked einsum accumulation.
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
    # Step 5: Assemble ADC via factorized phase + einsum
    # ================================================================
    # Weight per (M, n_tx, n_rx):
    # w = C_radar × sqrt(f_cos) × alpha_tx × alpha_rx
    sqrt_f_cos = torch.sqrt(f_cos.clamp(min=1e-20))                   # (M, n_tx, n_rx)
    w_full = C_radar * sqrt_f_cos * alpha_tx.unsqueeze(-1) * alpha_rx.unsqueeze(-2)  # (M, n_tx, n_rx)

    # Zero out back-facing paths
    active_i = cos_i > 1e-6                                            # (M, n_tx)
    active_o = cos_o > 1e-6                                            # (M, n_rx)
    active = active_i.unsqueeze(-1) & active_o.unsqueeze(-2)          # (M, n_tx, n_rx)
    w_full = w_full * active.float()

    # Phase: φ(t,r,k) = φ_tx(t,k) + φ_rx(r,k)
    # φ_tx(t,k) = phi_tx_const + phi_tx_slope × t_k
    # φ_rx(r,k) = phi_rx_const + phi_rx_slope × t_k
    t_k = torch.arange(K, device=device, dtype=torch.float32) / rast.sample_rate  # (K,)

    if detach_phase:
        phi_tx_const = phi_tx_const.detach()
        phi_tx_slope = phi_tx_slope.detach()
        phi_rx_const = phi_rx_const.detach()
        phi_rx_slope = phi_rx_slope.detach()

    # Accumulate ADC in chunks over Gaussians
    adc_real = torch.zeros(n_tx, n_rx, K, device=device)
    adc_imag = torch.zeros(n_tx, n_rx, K, device=device)

    for m0 in range(0, M, chunk_size):
        m1 = min(m0 + chunk_size, M)
        c = slice(m0, m1)

        # TX phasors: (chunk, n_tx, K)
        phi_tx = phi_tx_const[c, :, None] + phi_tx_slope[c, :, None] * t_k  # (chunk, n_tx, K)
        cos_tx = torch.cos(phi_tx)
        sin_tx = torch.sin(phi_tx)

        # RX phasors: (chunk, n_rx, K)
        phi_rx = phi_rx_const[c, :, None] + phi_rx_slope[c, :, None] * t_k  # (chunk, n_rx, K)
        cos_rx = torch.cos(phi_rx)
        sin_rx = torch.sin(phi_rx)

        # Weight for this chunk: (chunk, n_tx, n_rx)
        w_c = w_full[c]

        # ADC[t,r,k] = Σ_m w[m,t,r] × cos(φ_tx[m,t,k] + φ_rx[m,r,k])
        #            = Σ_m w[m,t,r] × (cos_tx cos_rx - sin_tx sin_rx)
        # and similarly for imag (cos_tx sin_rx + sin_tx cos_rx)

        # Efficient: contract w with phasors via einsum
        # real part: Σ_m w[m,t,r] (cos_tx[m,t,k] cos_rx[m,r,k] - sin_tx[m,t,k] sin_rx[m,r,k])
        # = Σ_m (w[m,t,r] cos_tx[m,t,k]) cos_rx[m,r,k] - Σ_m (w[m,t,r] sin_tx[m,t,k]) sin_rx[m,r,k]

        # w_cos_tx[m,t,r,k] = w[m,t,r] × cos_tx[m,t,k]  -- but this is (chunk,n_tx,n_rx,K) = too big
        # Instead: use two einsums per term
        # Term 1: Σ_m w[m,t,r] cos_tx[m,t,k] cos_rx[m,r,k]
        #       = Σ_m (w_cos_tx)[m,t,k,r] × cos_rx[m,r,k]  -- still (M,tx,rx,K) intermediate

        # Better approach: factor w[m,t,r] = w_full is NOT separable in t,r.
        # But we can loop over TX (only 12) to avoid the 4D tensor:
        for t in range(n_tx):
            # w_t: (chunk, n_rx)
            w_t = w_c[:, t, :]
            # cos/sin_tx_t: (chunk, K)
            ct = cos_tx[:, t, :]
            st = sin_tx[:, t, :]

            # (chunk, n_rx, K) = w_t[:,:,None] * cos_rx * ct[:,None,:] etc
            # But (chunk, n_rx, K) may be large. With chunk=2000, n_rx=16, K=256: 32MB. Fine.

            # weighted_cos_tx: (chunk, K) -- w_t already applied per-rx below
            # Actually: ADC[t, r, k] += Σ_m w_t[m,r] × (ct[m,k] cos_rx[m,r,k] - st[m,k] sin_rx[m,r,k])

            # Expand: (chunk, 1, K) * (chunk, n_rx, K) * (chunk, n_rx, 1)
            # = w_t[:,:,None] * ct[:,None,:] * cos_rx  →  sum over chunk dim
            real_contrib = torch.einsum('mk,mrk,mr->rk',
                                        ct, cos_rx, w_t) \
                         - torch.einsum('mk,mrk,mr->rk',
                                        st, sin_rx, w_t)
            imag_contrib = torch.einsum('mk,mrk,mr->rk',
                                        ct, sin_rx, w_t) \
                         + torch.einsum('mk,mrk,mr->rk',
                                        st, cos_rx, w_t)
            adc_real[t] += real_contrib
            adc_imag[t] += imag_contrib

    return adc_real, adc_imag
