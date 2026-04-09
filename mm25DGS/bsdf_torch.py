"""Full Jones BSDF for mmWave radar — pure PyTorch, no DrJit.

Port of mmir/renderer/bsdf/mmwave_jones.py + mmwave_scalar.py.
Uses all 6 ITU parameters.  Full autograd through every path.

Components:
  - Complex Fresnel reflection (amplitude + phase)
  - ITU-R P.2040-4 slab model (multi-layer interference)
  - Kirchhoff Approximation lobe (GGX NDF)
  - Small Perturbation Method lobe (vMF)
  - Directive + Lambertian incoherent lobes
  - Coherent/incoherent split (Rayleigh factor)
  - Jones polarization (s/p decomposition)
  - Coherent backscatter enhancement (CBS)

Returns POWER.  Caller must sqrt() for field amplitude.
"""

import math
import torch
from torch import Tensor
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_LIGHT = 299_792_458.0
WAVELENGTH = 3.9e-3                   # metres (matches mmIR's WAVELENGTH_77GHZ exactly)
K_WAVE = 2.0 * math.pi / WAVELENGTH  # ~1611 rad/m
INV_PI = 1.0 / math.pi
TWO_PI = 2.0 * math.pi

# Default TX/RX polarization: vertical = (0, 0, 1)
DEFAULT_POL = None  # set lazily on first use


# ===================================================================
#  Complex number helpers (using PyTorch native complex where useful)
# ===================================================================

def _cpx(r: Tensor, i: Tensor) -> Tensor:
    """Build complex tensor from real and imaginary parts."""
    return torch.complex(r.float(), i.float())


def _cpx_abs_sq(z: Tensor) -> Tensor:
    """Power = |z|² = real² + imag²."""
    return z.real ** 2 + z.imag ** 2


# ===================================================================
#  Permittivity → IOR
# ===================================================================

def permittivity_to_ior(eps_real: Tensor, eps_imag: Tensor) -> Tuple[Tensor, Tensor]:
    """Complex permittivity → (n, kappa) refractive index components.

    ε* = ε_r - jε_i  →  n* = n - jκ = sqrt(ε*)
    """
    eps_mag = torch.sqrt((eps_real ** 2 + eps_imag ** 2).clamp(min=1e-20))
    n = torch.sqrt(((eps_mag + eps_real) / 2.0).clamp(min=1e-20))
    kappa = torch.sqrt(((eps_mag - eps_real) / 2.0).clamp(min=0.0))
    return n, kappa


# ===================================================================
#  Fresnel reflection
# ===================================================================

def fresnel_complex_amplitude(
    n: Tensor, kappa: Tensor, cos_theta_i: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Complex Fresnel field reflection coefficients r_s, r_p.

    Returns (r_s, r_p) as PyTorch complex tensors.
    """
    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2

    # ñ² = (n - jκ)²
    n_sq = _cpx(n ** 2 - kappa ** 2, -2.0 * n * kappa)

    # ξ = ñ² - sin²θ
    xi = n_sq - sin_sq.to(torch.complex64)

    # √ξ = a + jb  (principal square root)
    a_plus_jb = torch.sqrt(xi)

    # r_s = (cosθ - √ξ) / (cosθ + √ξ)
    cos_c = cos_i.to(torch.complex64)
    r_s = (cos_c - a_plus_jb) / (cos_c + a_plus_jb + 1e-10)

    # r_p = (ñ²cosθ - √ξ) / (ñ²cosθ + √ξ)
    n_sq_cos = n_sq * cos_c
    r_p = (n_sq_cos - a_plus_jb) / (n_sq_cos + a_plus_jb + 1e-10)

    return r_s, r_p


def fresnel_power(n: Tensor, kappa: Tensor, cos_theta_i: Tensor) -> Tuple[Tensor, Tensor]:
    """Power reflectance R_s, R_p = |r_s|², |r_p|²."""
    r_s, r_p = fresnel_complex_amplitude(n, kappa, cos_theta_i)
    return _cpx_abs_sq(r_s), _cpx_abs_sq(r_p)


# ===================================================================
#  ITU Slab Fresnel (P.2040-4 Eqs 43-44)
# ===================================================================

def itu_slab_fresnel(
    eps_real: Tensor, eps_imag: Tensor,
    cos_theta_i: Tensor, thickness: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """ITU-R P.2040-4 single-layer slab reflection + transmission.

    Returns (R_TE, R_TM, T_TE, T_TM) as complex tensors.
    """
    cos_i = cos_theta_i.clamp(1e-6, 1.0)
    sin_sq = 1.0 - cos_i ** 2

    # Complex permittivity
    eta = _cpx(eps_real, -eps_imag)

    # √(η - sin²θ)
    a = torch.sqrt(eta - sin_sq.to(torch.complex64))

    cos_c = cos_i.to(torch.complex64)

    # Single-interface Fresnel coefficients
    r_te = (cos_c - a) / (cos_c + a + 1e-10)
    r_tm = (eta * cos_c - a) / (eta * cos_c + a + 1e-10)

    # Phase delay through slab
    q = (TWO_PI / WAVELENGTH) * thickness.to(torch.complex64) * a

    # exp(-jq) and exp(-j2q)
    ejq = torch.exp(-1j * q)
    ej2q = torch.exp(-2j * q)

    # Slab reflection: R = r(1 - exp(-j2q)) / (1 - r²exp(-j2q))
    R_TE = r_te * (1.0 - ej2q) / (1.0 - r_te ** 2 * ej2q + 1e-10)
    R_TM = r_tm * (1.0 - ej2q) / (1.0 - r_tm ** 2 * ej2q + 1e-10)

    # Slab transmission: T = (1 - r²)exp(-jq) / (1 - r²exp(-j2q))
    T_TE = (1.0 - r_te ** 2) * ejq / (1.0 - r_te ** 2 * ej2q + 1e-10)
    T_TM = (1.0 - r_tm ** 2) * ejq / (1.0 - r_tm ** 2 * ej2q + 1e-10)

    return R_TE, R_TM, T_TE, T_TM


def compute_slab_energy_gate(
    eps_real: Tensor, eps_imag: Tensor,
    cos_theta_i: Tensor, thickness: Tensor,
    frac_s: float = 0.5, frac_p: float = 0.5,
) -> Tensor:
    """Energy gate A = weighted power reflectance from slab model."""
    # Thin objects: fall back to single-interface Fresnel
    thin = thickness < 1e-6
    n, kap = permittivity_to_ior(eps_real, eps_imag)
    R_s_single, R_p_single = fresnel_power(n, kap, cos_theta_i)
    A_single = frac_s * R_s_single + frac_p * R_p_single

    # Slab model
    R_TE, R_TM, _, _ = itu_slab_fresnel(eps_real, eps_imag, cos_theta_i, thickness)
    A_slab = frac_s * _cpx_abs_sq(R_TE) + frac_p * _cpx_abs_sq(R_TM)

    return torch.where(thin, A_single, A_slab).clamp(0.0, 1.0)


# ===================================================================
#  SPM Validity + Parameter Enforcement
# ===================================================================

def enforce_spm_validity(sigma_h: Tensor, l_c: Tensor) -> Tuple[Tensor, Tensor]:
    """Clamp sigma_h and l_c to SPM validity regime."""
    h_max_1 = 0.1 / K_WAVE                            # kh ≪ 1
    l_c_min = WAVELENGTH * 0.5

    l_c_out = l_c.clamp(min=l_c_min)

    h_max_2 = torch.sqrt((0.1 / (K_WAVE ** 3 * l_c_out)).clamp(min=1e-20))
    h_max_3 = 0.21 * l_c_out

    h_max = torch.minimum(torch.minimum(
        torch.full_like(sigma_h, h_max_1), h_max_2), h_max_3)
    sigma_h_out = sigma_h.clamp(min=1e-7, max=None)
    sigma_h_out = torch.minimum(sigma_h_out, h_max)

    return sigma_h_out, l_c_out


# ===================================================================
#  Scattering Lobes
# ===================================================================

def eval_lobe_KA(
    wo: Tensor, wi: Tensor, n: Tensor,
    sigma_h: Tensor, l_c: Tensor,
) -> Tensor:
    """Kirchhoff Approximation lobe (GGX microfacet NDF, no Fresnel).

    Args: wo, wi, n all (*, 3).  sigma_h, l_c (*,).
    Returns: f_KA (*,) power BSDF (1/sr).
    """
    cos_i = (wi * n).sum(-1).clamp(min=1e-6)
    cos_o = (wo * n).sum(-1).clamp(min=1e-6)

    # GGX alpha from roughness
    alpha_raw = 4.0 * math.pi * sigma_h / WAVELENGTH
    alpha = torch.sqrt(alpha_raw.clamp(min=0.0025)).clamp(0.05, 0.95)
    alpha_sq = alpha ** 2

    # Half vector
    h = torch.nn.functional.normalize(wo + wi, dim=-1)
    h_dot_n = (h * n).sum(-1).clamp(min=0.0)

    # GGX NDF: D = α² / (π (h·n)²(α²-1)+1)²)
    denom_ndf = (h_dot_n ** 2 * (alpha_sq - 1.0) + 1.0) ** 2
    D = alpha_sq / (math.pi * denom_ndf.clamp(min=1e-20))

    # Smith G: G = 1 / (1 + Λ_i + Λ_o)  where Λ(cos) = (-1+√(1+α²tan²))/2
    def _lambda_ggx(cos_v):
        tan_sq = (1.0 / cos_v.clamp(min=1e-6) ** 2) - 1.0
        return (-1.0 + torch.sqrt((1.0 + alpha_sq * tan_sq).clamp(min=0.0))) / 2.0

    G = 1.0 / (1.0 + _lambda_ggx(cos_i) + _lambda_ggx(cos_o)).clamp(min=1e-10)

    # Cook-Torrance (no Fresnel — that's applied separately)
    f_KA = D * G / (4.0 * cos_i * cos_o).clamp(min=1e-10)
    return f_KA


def eval_lobe_SPM(
    wo: Tensor, wi: Tensor, n: Tensor,
    sigma_h: Tensor, l_c: Tensor,
    eps_real: Tensor, eps_imag: Tensor,
) -> Tensor:
    """SPM lobe (von Mises-Fisher around specular direction)."""
    # Specular reflection of wi
    wi_r = 2.0 * (wi * n).sum(-1, keepdim=True) * n - wi
    cos_dev = (wo * wi_r).sum(-1).clamp(-1.0, 1.0)

    # vMF concentration
    l_c_lam = l_c / WAVELENGTH
    roughness_slope = sigma_h / l_c.clamp(min=1e-8)
    kappa = (torch.sqrt(l_c_lam.clamp(min=0.0)) / (1.0 + 3.0 * roughness_slope)).clamp(0.5, 10.0)

    # vMF: f = κ/(4π sinh(κ)) × exp(κ(cosθ-1))
    kappa_c = kappa.clamp(max=50.0)
    sinh_k = (torch.exp(kappa_c) - torch.exp(-kappa_c)) / 2.0
    norm = kappa / (4.0 * math.pi * sinh_k.clamp(min=1e-10))
    f_vmf = norm * torch.exp(kappa * (cos_dev - 1.0))

    # Material modulation
    eps_contrast = torch.abs(eps_real - 1.0) + eps_imag
    eps_factor = (eps_contrast / 5.0).clamp(0.2, 1.0)

    return f_vmf * eps_factor


def eval_lobe_directive(
    wo: Tensor, wi: Tensor, n: Tensor,
    sigma_h: Tensor, l_c: Tensor,
) -> Tensor:
    """Broader vMF lobe for incoherent directive scattering."""
    wi_r = 2.0 * (wi * n).sum(-1, keepdim=True) * n - wi
    cos_dev = (wo * wi_r).sum(-1).clamp(-1.0, 1.0)

    l_c_lam = l_c / WAVELENGTH
    kappa_base = torch.sqrt(l_c_lam.clamp(min=0.0))
    roughness_slope = sigma_h / l_c.clamp(min=1e-8)
    broadening = 1.0 + 5.0 * roughness_slope
    kappa = (kappa_base / broadening).clamp(0.3, 5.0)

    kappa_c = kappa.clamp(max=50.0)
    sinh_k = (torch.exp(kappa_c) - torch.exp(-kappa_c)) / 2.0
    norm = kappa / (4.0 * math.pi * sinh_k.clamp(min=1e-10))
    return norm * torch.exp(kappa * (cos_dev - 1.0))


def eval_lobe_broad(wo: Tensor, wi: Tensor, n: Tensor) -> Tensor:
    """Lambertian broad lobe: f = 1/π."""
    cos_i = (wi * n).sum(-1).clamp(min=0.0)
    cos_o = (wo * n).sum(-1).clamp(min=0.0)
    valid = (cos_i > 1e-6) & (cos_o > 1e-6)
    return torch.where(valid, torch.full_like(cos_i, INV_PI), torch.zeros_like(cos_i))


# ===================================================================
#  Coherent / Incoherent Blend Factors
# ===================================================================

def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x.clamp(-50.0, 50.0))


def compute_coherent_incoherent_blend(
    sigma_h: Tensor, l_c: Tensor, cos_theta_i: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Returns (eta, gamma): coherent fraction and directive/broad ratio."""
    g = (2.0 * K_WAVE * sigma_h * cos_theta_i) ** 2
    eta = torch.exp(-g).clamp(0.01, 0.99)

    roughness_ratio = sigma_h / WAVELENGTH
    gamma = _sigmoid((0.05 - roughness_ratio) * 50.0).clamp(0.1, 0.9)

    return eta, gamma


def compute_validity_aware_blend(
    tau_base: Tensor, cos_theta_i: Tensor,
    sigma_h: Tensor, l_c: Tensor,
) -> Tensor:
    """Angle + validity-aware KA/SPM blend (tau_eff)."""
    tau_angle = _sigmoid((cos_theta_i - 0.94) * 20.0)

    kl = K_WAVE * l_c
    v_KA = _sigmoid((kl - 6.0) * 2.0)

    kh = K_WAVE * sigma_h
    v_SPM = _sigmoid((0.3 - kh) * 10.0)

    w_KA = tau_angle * tau_base * v_KA
    w_SPM = (1.0 - tau_angle) * (1.0 - tau_base) * v_SPM
    total = (w_KA + w_SPM).clamp(min=1e-6)

    return (w_KA / total).clamp(0.01, 0.99)


# ===================================================================
#  Coherent Backscatter Enhancement
# ===================================================================

def compute_cbs_factor(
    wo: Tensor, wi: Tensor, n: Tensor, l_c: Tensor,
) -> Tensor:
    """CBS enhancement factor ∈ [1, 2]."""
    wi_r = 2.0 * (wi * n).sum(-1, keepdim=True) * n - wi
    retro = -wi_r

    cos_bs = (wo * retro).sum(-1).clamp(-1.0, 1.0)
    sin_bs = torch.sqrt((1.0 - cos_bs ** 2).clamp(min=1e-20))

    x = K_WAVE * l_c * sin_bs
    sinc_x = torch.where(x.abs() > 1e-6, torch.sin(x) / x, torch.ones_like(x))

    return 1.0 + sinc_x ** 2


def compute_cbs_mean(l_c: Tensor) -> Tensor:
    """Hemispherical mean CBS for normalization."""
    alpha = K_WAVE * l_c
    return 1.0 + 1.0 / (1.0 + alpha)


# ===================================================================
#  Jones Polarization
# ===================================================================

def _get_default_pol(device):
    """Lazy-init default vertical polarization vector."""
    global DEFAULT_POL
    if DEFAULT_POL is None or DEFAULT_POL.device != device:
        DEFAULT_POL = torch.tensor([0.0, 0.0, 1.0], device=device)
    return DEFAULT_POL


def compute_sp_basis(wi: Tensor, n: Tensor) -> Tuple[Tensor, Tensor]:
    """s/p polarization basis for plane of incidence.

    s = (wi × n) / |wi × n|  (perpendicular to incidence plane)
    p = (s × wi) / |s × wi|  (in incidence plane)
    """
    s = torch.cross(wi, n, dim=-1)
    s_norm = s.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    s = s / s_norm

    # Fallback for grazing incidence
    grazing = s_norm.squeeze(-1) < 1e-4
    if grazing.any():
        fallback_ref = torch.tensor([1.0, 0.0, 0.0], device=wi.device).expand_as(wi)
        s_fb = torch.cross(wi, fallback_ref, dim=-1)
        s_fb = s_fb / s_fb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        s = torch.where(grazing.unsqueeze(-1), s_fb, s)

    p = torch.cross(s, wi, dim=-1)
    p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return s, p


def jones_fresnel_power(
    wi: Tensor, wo: Tensor, n: Tensor,
    eps_real: Tensor, eps_imag: Tensor,
    cos_theta_i: Tensor, thickness: Tensor,
    tx_pol: Tensor = None, rx_pol: Tensor = None,
    use_slab: bool = True,
) -> Tensor:
    """Polarization-weighted Fresnel power via Jones formalism.

    Handles TX→surface→RX polarization projection.
    Returns R_jones (*,) in [0, 1].
    """
    device = wi.device
    if tx_pol is None:
        tx_pol = _get_default_pol(device)
    if rx_pol is None:
        rx_pol = _get_default_pol(device)

    # s/p basis from incidence direction
    s_in, p_in = compute_sp_basis(wi, n)

    # Project TX polarization
    tx_s = (tx_pol * s_in).sum(-1)
    tx_p = (tx_pol * p_in).sum(-1)

    # Fresnel coefficients
    if use_slab and thickness is not None:
        R_TE, R_TM, _, _ = itu_slab_fresnel(eps_real, eps_imag, cos_theta_i, thickness)
        r_s = R_TE
        r_p = R_TM
    else:
        n_ior, kap = permittivity_to_ior(eps_real, eps_imag)
        r_s, r_p = fresnel_complex_amplitude(n_ior, kap, cos_theta_i)

    # Apply Jones reflection (diagonal: E_s_out = r_s × E_s_in)
    E_s_out = r_s * tx_s.to(torch.complex64)
    E_p_out = r_p * tx_p.to(torch.complex64)

    # Outgoing s/p basis
    s_out = s_in  # s is preserved under reflection
    p_out = torch.cross(s_out, wo, dim=-1)
    p_out = p_out / p_out.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Project onto RX polarization
    rx_s = (rx_pol * s_out).sum(-1)
    rx_p = (rx_pol * p_out).sum(-1)

    # Received field
    E_rx = E_s_out * rx_s.to(torch.complex64) + E_p_out * rx_p.to(torch.complex64)

    return _cpx_abs_sq(E_rx).clamp(0.0, 1.0)


# ===================================================================
#  Main BSDF Entry Points
# ===================================================================

def evaluate_bsdf_jones(
    cos_theta_i: Tensor,
    wo: Tensor, wi: Tensor, n: Tensor,
    eps_real: Tensor, eps_imag: Tensor,
    sigma_h: Tensor, l_c: Tensor,
    tau_base: Tensor, thickness: Tensor,
) -> Tensor:
    """Full Jones BSDF: KA+SPM coherent + directive+broad incoherent.

    Uses all 6 ITU parameters.  Returns POWER (1/sr).
    Caller must sqrt() for field amplitude and multiply by cosθ.

    Args: All tensors broadcastable to same shape.
    Returns: f_total (*,) BSDF power.
    """
    # Enforce SPM validity
    sh, lc = enforce_spm_validity(sigma_h, l_c)

    # Jones Fresnel (polarization-weighted power reflectance)
    R_jones = jones_fresnel_power(
        wi, wo, n, eps_real, eps_imag, cos_theta_i, thickness,
    )

    # Validity-aware KA/SPM blend
    tau_eff = compute_validity_aware_blend(tau_base, cos_theta_i, sh, lc)

    # Coherent lobes
    f_KA = eval_lobe_KA(wo, wi, n, sh, lc)
    f_SPM = eval_lobe_SPM(wo, wi, n, sh, lc, eps_real, eps_imag)
    f_coh = tau_eff * f_KA + (1.0 - tau_eff) * f_SPM

    # Incoherent lobes
    f_dir = eval_lobe_directive(wo, wi, n, sh, lc)
    f_broad = eval_lobe_broad(wo, wi, n)

    # Blend factors
    eta, gamma = compute_coherent_incoherent_blend(sh, lc, cos_theta_i)
    f_inc = gamma * f_dir + (1.0 - gamma) * f_broad

    # CBS correction
    cbs = compute_cbs_factor(wo, wi, n, lc)
    cbs_mean = compute_cbs_mean(lc)
    f_inc = f_inc * cbs / cbs_mean.clamp(min=1.0)

    # Energy-gated coherent + incoherent
    f_total = R_jones * (eta * f_coh + (1.0 - eta) * f_inc)

    return f_total


def evaluate_bsdf_jones_f_cos(
    cos_theta_i: Tensor,
    wo: Tensor, wi: Tensor, n: Tensor,
    eps_real: Tensor, eps_imag: Tensor,
    sigma_h: Tensor, l_c: Tensor,
    tau_base: Tensor, thickness: Tensor,
) -> Tensor:
    """BSDF × |cosθ_i|.  Returns POWER × cosine.  Main entry point."""
    f = evaluate_bsdf_jones(
        cos_theta_i, wo, wi, n,
        eps_real, eps_imag, sigma_h, l_c, tau_base, thickness,
    )
    return f * cos_theta_i


# ===================================================================
#  Backward-compatible aliases (used by adc_synthesis.py)
# ===================================================================

# Keep Tier 1 for testing/fallback
def evaluate_bsdf_tier1(cos_theta_i, eps_real, eps_imag, sigma_h, thickness):
    """Tier 1 (Fresnel only) — kept for comparison."""
    n, kap = permittivity_to_ior(eps_real, eps_imag)
    R_s, R_p = fresnel_power(n, kap, cos_theta_i)
    R = 0.5 * (R_s + R_p)
    eta_rayleigh = torch.exp(-((2.0 * K_WAVE * sigma_h * cos_theta_i) ** 2))
    return R * eta_rayleigh * cos_theta_i


KAPPA_77GHZ = K_WAVE  # alias for adc_synthesis.py
