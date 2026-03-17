"""
Physics-first hybrid BSDF for 77 GHz mmWave radar (CSV-BSDF equivalent).

Implements KA + SPM mixture based on arXiv:2401.01175 with:
- Complex permittivity Fresnel (energy gate)
- Angle-dependent blending (KA for small angles, SPM for large)
- Validity regime constraints from the paper
- Polarization through power fractions in s/p

CRITICAL: eval_f() returns POWER (1/sr).
The integrator must convert to amplitude via sqrt() for phasor synthesis.
"""

import numpy as np
import drjit as dr
import mitsuba as mi

from .base import BSDFBase
from ..utils.math import sample_cosine_hemisphere_concentric, to_global


# =============================================================================
# CONSTANTS
# =============================================================================

WAVELENGTH_77GHZ = 3.9e-3  # λ at 77 GHz (m)
K_77GHZ = 2.0 * np.pi / WAVELENGTH_77GHZ  # Wavenumber ≈ 1611 m⁻¹


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _sigmoid(x: 'mi.Float') -> 'mi.Float':
    """
    Sigmoid function: σ(x) = 1 / (1 + exp(-x))

    DrJit doesn't have a built-in sigmoid, so we implement it manually.
    Uses numerically stable form to avoid overflow for large |x|.
    """
    # For numerical stability:
    # If x >= 0: sigmoid(x) = 1 / (1 + exp(-x))
    # If x < 0:  sigmoid(x) = exp(x) / (1 + exp(x))
    pos_mask = x >= mi.Float(0.0)

    # Clamp to avoid overflow
    x_clamped = dr.clamp(x, mi.Float(-50.0), mi.Float(50.0))

    # Standard sigmoid
    exp_neg_x = dr.exp(-x_clamped)
    sigmoid_pos = mi.Float(1.0) / (mi.Float(1.0) + exp_neg_x)

    # Alternative form for negative x
    exp_x = dr.exp(x_clamped)
    sigmoid_neg = exp_x / (mi.Float(1.0) + exp_x)

    return dr.select(pos_mask, sigmoid_pos, sigmoid_neg)


# =============================================================================
# PHASE 1: CORE UTILITIES
# =============================================================================

def permittivity_to_ior(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float'
) -> tuple:
    """
    Convert complex permittivity to complex index of refraction.

    ε* = ε' - jε''
    n* = n - jκ = sqrt(ε*)

    Args:
        eps_real: ε' - Real relative permittivity
        eps_imag: ε'' - Imaginary permittivity (loss)

    Returns:
        (n, kappa): Real and imaginary parts of complex IOR
    """
    eps_mag = dr.sqrt(dr.maximum(eps_real * eps_real + eps_imag * eps_imag, mi.Float(1e-20)))
    n = dr.sqrt(dr.maximum((eps_mag + eps_real) / 2.0, mi.Float(1e-20)))
    kappa = dr.sqrt(dr.maximum((eps_mag - eps_real) / 2.0, mi.Float(1e-20)))
    return n, kappa


def fresnel_complex_power(
    n: 'mi.Float',
    kappa: 'mi.Float',
    cos_theta_i: 'mi.Float',
) -> tuple:
    """
    Compute Fresnel power reflectance |r_s|² and |r_p|².

    Uses full complex Fresnel equations for absorbing media.

    Args:
        n: Real part of complex IOR
        kappa: Imaginary part of complex IOR (extinction coefficient)
        cos_theta_i: Cosine of incident angle

    Returns:
        (R_s, R_p): Power reflectance for s and p polarizations
    """
    sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i

    # ñ² = (n - jκ)² = n² - κ² - 2jnκ
    n2_sq_real = n * n - kappa * kappa
    n2_sq_imag = -2.0 * n * kappa

    # ξ = ñ² - sin²(θ_i)
    xi_real = n2_sq_real - sin2_theta_i
    xi_imag = n2_sq_imag

    # sqrt(ξ) = a + jb (epsilon in sqrt/atan2 to prevent NaN backward at degenerate inputs)
    xi_mag_sq = xi_real * xi_real + xi_imag * xi_imag
    xi_mag = dr.sqrt(dr.maximum(xi_mag_sq, mi.Float(1e-20)))
    xi_arg = dr.atan2(xi_imag, dr.maximum(xi_real, mi.Float(1e-10)))  # safe atan2
    a = dr.sqrt(dr.maximum(xi_mag, mi.Float(1e-20))) * dr.cos(xi_arg / 2.0)
    b = dr.sqrt(dr.maximum(xi_mag, mi.Float(1e-20))) * dr.sin(xi_arg / 2.0)

    # R_s = |(cos_θ - (a+jb)) / (cos_θ + (a+jb))|²
    rs_num_sq = (cos_theta_i - a) * (cos_theta_i - a) + b * b
    rs_den_sq = dr.maximum((cos_theta_i + a) * (cos_theta_i + a) + b * b, mi.Float(1e-10))
    R_s = rs_num_sq / rs_den_sq

    # R_p = |(ñ² cos_θ - (a+jb)) / (ñ² cos_θ + (a+jb))|²
    n2cos_real = n2_sq_real * cos_theta_i
    n2cos_imag = n2_sq_imag * cos_theta_i

    rp_num_sq = (n2cos_real - a) * (n2cos_real - a) + (n2cos_imag - b) * (n2cos_imag - b)
    rp_den_sq = dr.maximum((n2cos_real + a) * (n2cos_real + a) + (n2cos_imag + b) * (n2cos_imag + b), mi.Float(1e-10))
    R_p = rp_num_sq / rp_den_sq

    return R_s, R_p


# =============================================================================
# PHASE 2: ITU SINGLE-LAYER SLAB FRESNEL (ITU-R P.2040-4, Eqs 43-44)
# =============================================================================

def _cpx_sqrt(x_real: 'mi.Float', x_imag: 'mi.Float') -> tuple:
    """
    Complex square root using the algebraic formula.
    Matches Sionna's cpx_sqrt implementation.

    sqrt(a + jb) = (r + a)/(2r)^0.5 + j*sign(b)*(r - a)/(2r)^0.5
    where r = |a + jb|.
    """
    r = dr.sqrt(dr.maximum(x_real * x_real + x_imag * x_imag, mi.Float(1e-20)))
    y_real = dr.sqrt(dr.maximum(mi.Float(0.5) * (r + x_real), mi.Float(1e-20)))
    y_imag = dr.sign(x_imag) * dr.sqrt(dr.maximum(mi.Float(0.5) * (r - x_real), mi.Float(1e-20)))
    return y_real, y_imag


def _cpx_mul(a_r, a_i, b_r, b_i):
    """Complex multiply: (a_r + j*a_i) * (b_r + j*b_i)."""
    return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r


def _cpx_div(a_r, a_i, b_r, b_i):
    """Complex divide: (a_r + j*a_i) / (b_r + j*b_i)."""
    denom = dr.maximum(b_r * b_r + b_i * b_i, mi.Float(1e-30))
    return (a_r * b_r + a_i * b_i) / denom, (a_i * b_r - a_r * b_i) / denom


def _cpx_sq(a_r, a_i):
    """Complex square: (a_r + j*a_i)^2."""
    return a_r * a_r - a_i * a_i, mi.Float(2.0) * a_r * a_i


def _cpx_exp(a_r, a_i):
    """Complex exp: exp(a_r + j*a_i) = exp(a_r) * (cos(a_i) + j*sin(a_i))."""
    e = dr.exp(a_r)
    return e * dr.cos(a_i), e * dr.sin(a_i)


def _cpx_abs_sq(a_r, a_i):
    """Complex magnitude squared: |a_r + j*a_i|^2."""
    return a_r * a_r + a_i * a_i


def itu_slab_fresnel(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
    thickness: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """
    ITU-R P.2040-4 single-layer slab Fresnel model (Eqs 43a, 43b, 44).

    Accounts for multiple internal reflections within a lossy dielectric slab
    of thickness d. Returns complex reflection and transmission coefficients
    for TE and TM polarizations.

    This matches Sionna's itu_coefficients_single_layer_slab() exactly.

    Args:
        eps_real: Real relative permittivity (eps_r from ITU model)
        eps_imag: Imaginary permittivity magnitude (sigma / (omega * eps_0))
        cos_theta_i: Cosine of incidence angle
        thickness: Slab thickness (m)
        wavelength: Wavelength (m)

    Returns:
        (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
         T_TE_r, T_TE_i, T_TM_r, T_TM_i): Real/imag parts of complex coefficients
    """
    sin2_theta = mi.Float(1.0) - cos_theta_i * cos_theta_i

    # Complex permittivity: eta = eps_real - j*eps_imag
    eta_r = eps_real
    eta_i = -eps_imag  # Convention: eta = eps_r - j*eps_i

    # sqrt(eta - sin^2(theta))
    arg_r = eta_r - sin2_theta
    arg_i = eta_i
    a_r, a_i = _cpx_sqrt(arg_r, arg_i)

    # Single-interface Fresnel coefficients (Eqs 37a, 37b)
    # r_TE = (cos_theta - a) / (cos_theta + a)
    rte_num_r = cos_theta_i - a_r
    rte_num_i = -a_i
    rte_den_r = cos_theta_i + a_r
    rte_den_i = a_i
    rte_r, rte_i = _cpx_div(rte_num_r, rte_num_i, rte_den_r, rte_den_i)

    # r_TM = (eta * cos_theta - a) / (eta * cos_theta + a)
    # eta * cos_theta
    etac_r = eta_r * cos_theta_i
    etac_i = eta_i * cos_theta_i
    rtm_num_r = etac_r - a_r
    rtm_num_i = etac_i - a_i
    rtm_den_r = etac_r + a_r
    rtm_den_i = etac_i + a_i
    rtm_r, rtm_i = _cpx_div(rtm_num_r, rtm_num_i, rtm_den_r, rtm_den_i)

    # Phase delay through slab: q = 2*pi*d/lambda * a  (Eq 44)
    phase_scale = mi.Float(2.0 * np.pi) * thickness / mi.Float(wavelength)
    q_r = phase_scale * a_r
    q_i = phase_scale * a_i

    # exp(-j*q) = exp(-j*(q_r + j*q_i)) = exp(q_i) * exp(-j*q_r)
    # = exp(q_i) * (cos(q_r) - j*sin(q_r))
    ejq_r, ejq_i = _cpx_exp(q_i, -q_r)         # exp(-j*q)
    ej2q_r, ej2q_i = _cpx_exp(mi.Float(2.0) * q_i, mi.Float(-2.0) * q_r)  # exp(-j*2q)

    # r_p^2
    rte_sq_r, rte_sq_i = _cpx_sq(rte_r, rte_i)
    rtm_sq_r, rtm_sq_i = _cpx_sq(rtm_r, rtm_i)

    # Denominator: 1 - r_p^2 * exp(-j*2q)
    prod_te_r, prod_te_i = _cpx_mul(rte_sq_r, rte_sq_i, ej2q_r, ej2q_i)
    denom_te_r = mi.Float(1.0) - prod_te_r
    denom_te_i = -prod_te_i

    prod_tm_r, prod_tm_i = _cpx_mul(rtm_sq_r, rtm_sq_i, ej2q_r, ej2q_i)
    denom_tm_r = mi.Float(1.0) - prod_tm_r
    denom_tm_i = -prod_tm_i

    # Slab reflection (Eq 43a): R = r_p * (1 - exp(-j*2q)) / denom
    one_minus_ej2q_r = mi.Float(1.0) - ej2q_r
    one_minus_ej2q_i = -ej2q_i

    rte_num2_r, rte_num2_i = _cpx_mul(rte_r, rte_i, one_minus_ej2q_r, one_minus_ej2q_i)
    R_TE_r, R_TE_i = _cpx_div(rte_num2_r, rte_num2_i, denom_te_r, denom_te_i)

    rtm_num2_r, rtm_num2_i = _cpx_mul(rtm_r, rtm_i, one_minus_ej2q_r, one_minus_ej2q_i)
    R_TM_r, R_TM_i = _cpx_div(rtm_num2_r, rtm_num2_i, denom_tm_r, denom_tm_i)

    # Slab transmission (Eq 43b): T = (1 - r_p^2) * exp(-j*q) / denom
    one_minus_rte_sq_r = mi.Float(1.0) - rte_sq_r
    one_minus_rte_sq_i = -rte_sq_i
    tte_num_r, tte_num_i = _cpx_mul(one_minus_rte_sq_r, one_minus_rte_sq_i, ejq_r, ejq_i)
    T_TE_r, T_TE_i = _cpx_div(tte_num_r, tte_num_i, denom_te_r, denom_te_i)

    one_minus_rtm_sq_r = mi.Float(1.0) - rtm_sq_r
    one_minus_rtm_sq_i = -rtm_sq_i
    ttm_num_r, ttm_num_i = _cpx_mul(one_minus_rtm_sq_r, one_minus_rtm_sq_i, ejq_r, ejq_i)
    T_TM_r, T_TM_i = _cpx_div(ttm_num_r, ttm_num_i, denom_tm_r, denom_tm_i)

    return (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
            T_TE_r, T_TE_i, T_TM_r, T_TM_i)


def itu_slab_fresnel_rough(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
    thickness: 'mi.Float',
    sigma_h: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """
    Slab Fresnel with roughness attenuation on the transmitted field.

    The coherent transmitted beam through a rough surface is attenuated by the
    transmission Rayleigh factor. This accounts for scattering of transmitted
    energy into non-specular directions.

    The reflection side is NOT attenuated here because that is handled by the
    coherent/incoherent blend (eta from Rayleigh factor in Phase 3).

    NOTE: cos_theta_t uses the real-part approximation sqrt(eps_real) rather
    than Re(sqrt(eta_complex)). For the ITU materials in our database the
    imaginary part of permittivity is small enough at 77 GHz that the error
    is negligible, but for very lossy materials (wet ground, metal) this
    should be replaced with Re(sqrt(eta_complex)).

    Returns same 8-tuple as itu_slab_fresnel, with T attenuated.
    """
    (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
     T_TE_r, T_TE_i, T_TM_r, T_TM_i) = itu_slab_fresnel(
        eps_real, eps_imag, cos_theta_i, thickness, wavelength
    )

    k = mi.Float(2.0 * np.pi / wavelength)

    # Transmission Rayleigh factor
    sin2_theta_i = mi.Float(1.0) - cos_theta_i * cos_theta_i
    sin2_theta_t = sin2_theta_i / dr.maximum(eps_real, mi.Float(1.001))
    cos_theta_t = dr.sqrt(dr.maximum(mi.Float(1.0) - sin2_theta_t, mi.Float(1e-20)))
    sqrt_eps = dr.sqrt(dr.maximum(eps_real, mi.Float(1.001)))

    delta_kz = k * sigma_h * (cos_theta_i - sqrt_eps * cos_theta_t)
    g_t = delta_kz * delta_kz
    rayleigh_t = dr.exp(-g_t)

    # Attenuate coherent transmission
    T_TE_r = T_TE_r * rayleigh_t
    T_TE_i = T_TE_i * rayleigh_t
    T_TM_r = T_TM_r * rayleigh_t
    T_TM_i = T_TM_i * rayleigh_t

    return (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
            T_TE_r, T_TE_i, T_TM_r, T_TM_i)


def compute_slab_energy_gate(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
    thickness: 'mi.Float',
    frac_s: 'mi.Float',
    frac_p: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
    sigma_h: 'mi.Float' = None,
) -> 'mi.Float':
    """
    Compute energy gate A(omega_i) using the ITU slab Fresnel model.

    Replaces the single-interface compute_energy_gate() with thickness-dependent
    slab reflection that accounts for multiple internal reflections.

    When thickness = 0, falls back to single-interface Fresnel.

    When sigma_h is provided, uses itu_slab_fresnel_rough() which attenuates the
    *transmission* coefficients by the transmission Rayleigh factor. The energy
    gate A = |R|² is NOT directly affected by sigma_h through this path because
    itu_slab_fresnel_rough() returns *reflection* coefficients unchanged.
    Reflection-side roughness is handled separately by the Rayleigh factor
    η = exp(-(2kσ_h cos θ_i)²) in compute_coherent_incoherent_blend(), which
    splits reflected energy into coherent and incoherent components.
    This is correct physics: reflection roughness → coherent/incoherent split;
    transmission roughness → beam attenuation through the slab.

    Args:
        eps_real: Real relative permittivity
        eps_imag: Imaginary permittivity magnitude
        cos_theta_i: Cosine of incidence angle
        thickness: Slab thickness (m). 0 = single-interface fallback.
        frac_s: Power fraction in TE polarization
        frac_p: Power fraction in TM polarization
        wavelength: Wavelength (m)
        sigma_h: RMS surface height (m). If provided, uses rough slab model.

    Returns:
        A: Energy gate in [0, 1]
    """
    use_slab = thickness > mi.Float(1e-6)

    # Single-interface Fresnel (fallback)
    n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
    R_s_single, R_p_single = fresnel_complex_power(n_ior, kappa, cos_theta_i)
    A_single = frac_s * R_s_single + frac_p * R_p_single

    # Slab Fresnel (with optional rough transmission)
    if sigma_h is not None:
        # Rough slab: sigma_h attenuates transmission only, reflection unchanged
        (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
         T_TE_r, T_TE_i, T_TM_r, T_TM_i) = itu_slab_fresnel_rough(
            eps_real, eps_imag, cos_theta_i, thickness,
            sigma_h, wavelength  # Note: sigma_h before wavelength per function signature
        )
    else:
        (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
         T_TE_r, T_TE_i, T_TM_r, T_TM_i) = itu_slab_fresnel(
            eps_real, eps_imag, cos_theta_i, thickness, wavelength
        )
    R_s_slab = _cpx_abs_sq(R_TE_r, R_TE_i)
    R_p_slab = _cpx_abs_sq(R_TM_r, R_TM_i)
    A_slab = frac_s * R_s_slab + frac_p * R_p_slab

    A = dr.select(use_slab, A_slab, A_single)
    return dr.clamp(A, mi.Float(0.0), mi.Float(1.0))


def compute_sp_basis(wi: 'mi.Vector3f', n: 'mi.Vector3f') -> tuple:
    """
    Compute s and p polarization basis vectors.

    s: perpendicular to plane of incidence
    p: in plane of incidence, perpendicular to wi

    Args:
        wi: Incident direction (pointing towards surface)
        n: Surface normal

    Returns:
        (s, p): Polarization basis vectors
    """
    s_unnorm = dr.cross(wi, n)
    s_len = dr.norm(s_unnorm)

    # Handle grazing incidence
    valid = s_len > 1e-6
    fallback_s = dr.cross(wi, mi.Vector3f(1, 0, 0))
    fallback_len = dr.norm(fallback_s)
    fallback_s = dr.select(
        fallback_len > 1e-6,
        fallback_s / dr.maximum(fallback_len, mi.Float(1e-10)),
        dr.normalize(dr.cross(wi, mi.Vector3f(0, 1, 0)))
    )

    s = dr.select(valid, s_unnorm / dr.maximum(s_len, mi.Float(1e-10)), fallback_s)
    p = dr.normalize(dr.cross(s, wi))

    return s, p


def compute_polarization_fractions(
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    antenna_pol: 'mi.Vector3f' = None,
) -> tuple:
    """
    Compute power fractions in s and p polarization.

    For vertical antenna polarization, project onto local s/p basis.

    Args:
        wi: Incident direction
        n: Surface normal
        antenna_pol: Antenna polarization vector (default: vertical)

    Returns:
        (frac_s, frac_p): Power fractions summing to ~1
    """
    if antenna_pol is None:
        antenna_pol = mi.Vector3f(0, 0, 1)  # Vertical polarization

    s, p = compute_sp_basis(wi, n)

    amp_s = dr.abs(dr.dot(antenna_pol, s))
    amp_p = dr.abs(dr.dot(antenna_pol, p))

    total_power = amp_s * amp_s + amp_p * amp_p

    # At normal incidence (or when polarization is along the surface normal),
    # the s/p basis becomes degenerate. Fall back to unpolarized (50/50 split).
    is_degenerate = total_power < mi.Float(1e-6)
    frac_s = dr.select(is_degenerate, mi.Float(0.5), (amp_s * amp_s) / dr.maximum(total_power, mi.Float(1e-10)))
    frac_p = dr.select(is_degenerate, mi.Float(0.5), (amp_p * amp_p) / dr.maximum(total_power, mi.Float(1e-10)))

    return frac_s, frac_p


def compute_energy_gate(
    n_ior: 'mi.Float',
    kappa: 'mi.Float',
    cos_theta_i: 'mi.Float',
    frac_s: 'mi.Float',
    frac_p: 'mi.Float',
) -> 'mi.Float':
    """
    Compute energy gate A(ωi) from complex Fresnel.

    A(ωi) = frac_s × R_s + frac_p × R_p

    Args:
        n_ior: Real part of complex IOR
        kappa: Imaginary part of complex IOR
        cos_theta_i: Cosine of incident angle
        frac_s: Power fraction in s polarization
        frac_p: Power fraction in p polarization

    Returns:
        A: Energy gate ∈ [0, 1]
    """
    R_s, R_p = fresnel_complex_power(n_ior, kappa, cos_theta_i)
    A = frac_s * R_s + frac_p * R_p
    return dr.clamp(A, mi.Float(0.0), mi.Float(1.0))


def enforce_spm_validity(
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """
    Clamp parameters to SPM validity regime.

    SPM validity (Eq. 19 from paper):
    - kh ≪ 1  →  h ≪ λ/(2π)  →  h < 0.1 × λ/(2π) ≈ 62 μm at 77 GHz
    - k³h²l ≪ 1  →  h²l < λ³/(8π³) ≈ 1.9e-9 m³ at 77 GHz
    - √2 h/l < 0.3  →  h < 0.21 × l

    Args:
        sigma_h: RMS surface height (meters)
        l_c: Correlation length (meters)
        wavelength: Wavelength in meters

    Returns:
        (sigma_h_clamped, l_c_clamped): Clamped parameters
    """
    k = 2.0 * np.pi / wavelength

    # Constraint 1: kh < 0.1 (≪ 1)
    h_max_1 = mi.Float(0.1 / k)  # ≈ 62 μm at 77 GHz

    # Set minimum l_c to avoid division issues
    l_c_min = mi.Float(wavelength * 0.5)  # Minimum correlation length = λ/2
    l_c_clamped = dr.maximum(l_c, l_c_min)

    # Constraint 3: √2 h/l < 0.3 → h < 0.21 × l
    h_max_3 = 0.21 * l_c_clamped
    h_max = dr.minimum(h_max_1, h_max_3)

    # Minimum h for numerical stability
    h_min = mi.Float(1e-7)  # 0.1 μm

    sigma_h_clamped = dr.clamp(sigma_h, h_min, h_max)

    # Verify constraint 2: k³h²l < 0.1
    k3 = mi.Float(k * k * k)
    h_max_2 = dr.sqrt(mi.Float(0.1) / (k3 * l_c_clamped))
    sigma_h_clamped = dr.minimum(sigma_h_clamped, h_max_2)

    return sigma_h_clamped, l_c_clamped


def compute_ka_validity(
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """
    Compute soft validity weight for KA regime.

    Returns v_KA ∈ [0, 1] where:
    - v_KA → 1 when KA is valid
    - v_KA → 0 when KA is invalid

    KA validity conditions:
    - kl > 6 (correlation length much larger than wavelength)
    - l² > 2.76 × h × λ (Gaussian surface constraint)

    Args:
        sigma_h: RMS surface height
        l_c: Correlation length
        wavelength: Wavelength in meters

    Returns:
        v_KA: Validity weight ∈ [0.01, 1]
    """
    k = mi.Float(2.0 * np.pi / wavelength)

    # Condition 1: kl > 6
    kl = k * l_c
    v1 = _sigmoid((kl - mi.Float(6.0)) * mi.Float(2.0))  # Soft threshold at kl = 6

    # Condition 4: l² > 2.76 × h × λ
    l2_threshold = mi.Float(2.76) * sigma_h * mi.Float(wavelength)
    v4 = _sigmoid((l_c * l_c - l2_threshold) / mi.Float(wavelength * wavelength) * mi.Float(10.0))

    # Combined validity
    v_KA = v1 * v4

    return dr.clamp(v_KA, mi.Float(0.01), mi.Float(1.0))  # Never fully zero for gradient stability


def compute_validity_aware_blend(
    tau_base: 'mi.Float',
    cos_theta_i: 'mi.Float',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """
    Compute blend coefficient τ with validity weighting.

    Combines:
    1. Angle-based blending (KA for small angles, SPM for large)
    2. KA validity gating (reduce KA weight when invalid)
    3. SPM validity gating (reduce SPM weight when invalid)

    Args:
        tau_base: Base blending coefficient from material
        cos_theta_i: Cosine of incident angle
        sigma_h: RMS surface height
        l_c: Correlation length
        wavelength: Wavelength in meters

    Returns:
        τ_eff: Effective blending coefficient for KA vs SPM
    """
    k = mi.Float(2.0 * np.pi / wavelength)

    # Angle-based blend
    # θ = 20° → cos(20°) ≈ 0.94
    THETA_THRESHOLD_COS = mi.Float(0.94)
    tau_angle = _sigmoid((cos_theta_i - THETA_THRESHOLD_COS) * mi.Float(20.0))

    # KA validity
    kl = k * l_c
    v_KA = _sigmoid((kl - mi.Float(6.0)) * mi.Float(2.0))

    # SPM validity (kh < 0.3 is "good")
    kh = k * sigma_h
    v_SPM = _sigmoid((mi.Float(0.3) - kh) * mi.Float(10.0))

    # Weighted blend
    # At small angles: prefer KA if valid
    # At large angles: prefer SPM if valid
    w_KA = tau_angle * tau_base * v_KA
    w_SPM = (mi.Float(1.0) - tau_angle) * (mi.Float(1.0) - tau_base) * v_SPM

    # Normalize
    total = dr.maximum(w_KA + w_SPM, mi.Float(1e-6))
    tau_eff = w_KA / total

    return dr.clamp(tau_eff, mi.Float(0.01), mi.Float(0.99))


def map_renderer_params_to_physical(
    albedo: 'mi.Float',
    roughness: 'mi.Float',
    metallic: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> tuple:
    """
    Map CG-style parameters to physical mmWave parameters.

    Args:
        albedo: Diffuse reflectance [0, 1]
        roughness: GGX alpha [0, 1]
        metallic: Metal factor [0, 1]
        wavelength: Wavelength in meters

    Returns:
        (eps_real, eps_imag, sigma_h, l_c, tau): Physical parameters
    """
    k = 2.0 * np.pi / wavelength

    # === Electromagnetic properties ===
    eps_real = dr.select(
        metallic > mi.Float(0.5),
        mi.Float(1.0),
        mi.Float(2.0) + albedo * mi.Float(4.0)
    )

    eps_imag = dr.select(
        metallic > mi.Float(0.5),
        mi.Float(1e6),
        mi.Float(0.05) + albedo * mi.Float(0.3)
    )

    # === Roughness statistics ===
    # Map GGX alpha to physical roughness
    # Use SPM validity bounds: h < 0.1/k ≈ 62 μm at 77 GHz
    h_max = mi.Float(0.1 / k)
    sigma_h = roughness * h_max
    sigma_h = dr.clamp(sigma_h, mi.Float(1e-7), h_max)

    # Correlation length: smooth surfaces have larger l_c
    # Ensure √2 h/l < 0.3 → l > √2 h / 0.3 ≈ 4.7 h
    l_c_min = mi.Float(5.0) * sigma_h  # Slightly above minimum for stability
    l_c = mi.Float(wavelength) * (mi.Float(1.0) + mi.Float(20.0) * (mi.Float(1.0) - roughness))
    l_c = dr.maximum(l_c, l_c_min)

    # === Blending coefficient ===
    # Metals: more specular (higher τ)
    # Dielectrics: more diffuse (lower τ)
    # Rougher surfaces: more diffuse
    tau = dr.select(
        metallic > mi.Float(0.5),
        mi.Float(0.9) - roughness * mi.Float(0.4),
        mi.Float(0.6) - roughness * mi.Float(0.5)
    )
    tau = dr.clamp(tau, mi.Float(0.1), mi.Float(0.95))

    return eps_real, eps_imag, sigma_h, l_c, tau


# =============================================================================
# PHASE 2: LOBE IMPLEMENTATIONS
# =============================================================================

def _eval_lambda_ggx(alpha_sq: 'mi.Float', cos_theta: 'mi.Float') -> 'mi.Float':
    """
    GGX Lambda function for Smith masking.

    Args:
        alpha_sq: Squared roughness parameter
        cos_theta: Cosine of angle

    Returns:
        Lambda value for masking computation
    """
    valid = cos_theta > mi.Float(0)
    cos_theta_sq = cos_theta * cos_theta
    tan_theta_sq = dr.maximum(mi.Float(1.0) - cos_theta_sq, mi.Float(0.0)) / dr.maximum(cos_theta_sq, mi.Float(1e-10))
    result = mi.Float(0.5) * (mi.Float(-1.0) + dr.sqrt(mi.Float(1.0) + alpha_sq * tan_theta_sq))
    return dr.select(valid, result, mi.Float(0.0))


def eval_lobe_KA(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """
    Evaluate KA (Kirchhoff Approximation) specular lobe.

    From PLAN Section 5.1:
    - Uses GGX microfacet as angular shape
    - α = 4π σ_h / λ (Rayleigh-based physical roughness mapping)
    - Standard Cook-Torrance: D × G / (4 × cos_i × cos_o)
    - Fresnel is NOT included here (applied via energy gate)

    Args:
        wo: Outgoing direction (towards receiver)
        wi: Incident direction (towards transmitter)
        n: Surface normal
        sigma_h: RMS surface height (meters)
        l_c: Correlation length (meters) - unused but kept for API
        wavelength: Wavelength in meters

    Returns:
        f_KA: Normalized power-domain BSDF value (1/sr)
    """
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Map physical roughness to GGX alpha
    #
    # The plan's formula α = 4π × σ_h / λ produces very small values (0.02-0.1)
    # which create extremely peaked specular (D ∝ 1/α²). This doesn't match
    # typical rendering behavior where α ∈ [0.1, 0.9].
    #
    # We use a scaled mapping: α = sqrt(4π × σ_h / λ)
    # This preserves the physical relationship while producing GGX-compatible values:
    #   σ_h = 6 μm  → α = sqrt(0.02) ≈ 0.14 (smooth)
    #   σ_h = 24 μm → α = sqrt(0.08) ≈ 0.28 (moderate)
    #   σ_h = 35 μm → α = sqrt(0.11) ≈ 0.33 (rough)
    #
    alpha_raw = mi.Float(4.0 * np.pi) * sigma_h / mi.Float(wavelength)
    alpha = dr.sqrt(alpha_raw)
    alpha = dr.clamp(alpha, mi.Float(0.05), mi.Float(0.95))

    # Half-vector (safe normalize to avoid NaN backward at grazing angles
    # where wo + wi ≈ 0, since dr.normalize backward → 1/norm → inf)
    h_unnorm = wo + wi
    h_len_sq = dr.dot(h_unnorm, h_unnorm)
    h_len = dr.sqrt(dr.maximum(h_len_sq, mi.Float(1e-20)))
    h = h_unnorm / h_len
    h_dot_n = dr.dot(h, n)

    # GGX NDF: D(h) = α² / (π × ((n·h)² × (α² - 1) + 1)²)
    a2 = alpha * alpha
    d_denom = (h_dot_n * h_dot_n) * (a2 - mi.Float(1.0)) + mi.Float(1.0)
    D = a2 / (d_denom * d_denom * mi.Float(np.pi))

    # Smith GGX masking-shadowing: G = 1 / (1 + Λ(wi) + Λ(wo))
    lambda_i = _eval_lambda_ggx(a2, cos_theta_i)
    lambda_o = _eval_lambda_ggx(a2, cos_theta_o)
    G = mi.Float(1.0) / (mi.Float(1.0) + lambda_i + lambda_o)

    # Standard Cook-Torrance (PLAN Section 5.1):
    # f_KA = D × G / (4 × cos_i × cos_o)
    # NO Fresnel here - that's in the energy gate
    denom = mi.Float(4.0) * dr.maximum(cos_theta_i * cos_theta_o, mi.Float(1e-6))
    f_KA = D * G / denom

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_KA, mi.Float(0.0))


def eval_lobe_SPM(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
    eps_real: 'mi.Float' = None,
    eps_imag: 'mi.Float' = None,
) -> 'mi.Float':
    """
    Evaluate SPM (Small Perturbation Method) diffuse lobe.

    From PLAN Section 5.2:
    - Uses von Mises-Fisher distribution centered on specular reflection
    - κ = (k × l_c)² where k = 2π/λ
    - Normalization: κ / (4π × sinh(κ))
    - Includes permittivity-based modulation

    Args:
        wo: Outgoing direction (towards receiver)
        wi: Incident direction (towards transmitter)
        n: Surface normal
        sigma_h: RMS surface height (unused but kept for API)
        l_c: Correlation length (meters)
        wavelength: Wavelength in meters
        eps_real: Real permittivity (for material contrast modulation)
        eps_imag: Imaginary permittivity

    Returns:
        f_SPM: Normalized power-domain BSDF value (1/sr)
    """
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Specular reflection direction: r = 2(n·d)n - d
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    cos_dev = dr.dot(wo, wi_reflected)

    # SPM lobe width: kappa determines concentration
    #
    # The SPM lobe should be broader than KA (specular) but still directive.
    # Using sqrt(l_c / λ) gives reasonable concentration values:
    # - l_c = 5λ  → kappa ≈ 2.2 (broader lobe)
    # - l_c = 20λ → kappa ≈ 4.5 (moderate concentration)
    #
    # This keeps kappa in a range where vMF gives meaningful non-zero values
    # while still being narrower than Lambertian.
    l_c_wavelengths = l_c / mi.Float(wavelength)
    # Couple SPM lobe width to surface slope (sigma_h / l_c).
    # Factor of 3 (vs 5 for directive lobe) because SPM is already more peaked.
    roughness_slope = sigma_h / dr.maximum(l_c, mi.Float(1e-8))
    kappa_vmf = dr.sqrt(l_c_wavelengths) / (mi.Float(1.0) + mi.Float(3.0) * roughness_slope)
    kappa_vmf = dr.clamp(kappa_vmf, mi.Float(0.5), mi.Float(10.0))

    # von Mises-Fisher distribution: f ∝ exp(κ × (cos_dev - 1))
    # The -1 normalizes the peak to exp(0) = 1 at specular
    f_vmf = dr.exp(kappa_vmf * (cos_dev - mi.Float(1.0)))

    # PLAN Section 5.2: Normalization = κ / (4π × sinh(κ))
    # For numerical stability, use sinh(x) = (exp(x) - exp(-x)) / 2
    # When κ is large, sinh(κ) ≈ exp(κ)/2, so norm ≈ 2κ / (4π × exp(κ)) = κ / (2π × exp(κ))
    kappa_clamped = dr.minimum(kappa_vmf, mi.Float(50.0))  # Prevent overflow
    sinh_kappa = (dr.exp(kappa_clamped) - dr.exp(-kappa_clamped)) / mi.Float(2.0)
    norm_factor = kappa_vmf / (mi.Float(4.0 * np.pi) * dr.maximum(sinh_kappa, mi.Float(1e-6)))
    f_SPM = f_vmf * norm_factor

    # Material modulation for SPM
    # The SPM directive lobe should be weighted by material properties.
    # We use permittivity contrast AND add a base scattering factor to ensure
    # reasonable energy at non-specular angles.
    #
    # For dielectric (eps_real=3.2): contrast = 2.2 + 0.14 = 2.34
    # eps_factor = clamp(2.34/10, 0.1, 1.0) = 0.234
    if eps_real is not None and eps_imag is not None:
        eps_contrast = dr.abs(eps_real - mi.Float(1.0)) + eps_imag
        eps_factor = dr.clamp(eps_contrast / mi.Float(5.0), mi.Float(0.2), mi.Float(1.0))
        f_SPM = f_SPM * eps_factor

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_SPM, mi.Float(0.0))


# =============================================================================
# PHASE 7: INCOHERENT LOBES (Enabled early for validation)
# =============================================================================

def eval_lobe_directive(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """
    Evaluate directive incoherent lobe.

    This is a broader version of SPM for surfaces where SPM validity breaks down
    or for multiple-scattering effects. Uses a wider vMF-like distribution
    centered on specular.

    Args:
        wo: Outgoing direction
        wi: Incident direction
        n: Surface normal
        sigma_h: RMS surface height
        l_c: Correlation length
        wavelength: Wavelength in meters

    Returns:
        f_dir: Directive incoherent lobe value (1/sr)
    """
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Specular reflection direction
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    cos_dev = dr.dot(wo, wi_reflected)

    # Directive lobe width coupled to surface slope RMS (sigma_h / l_c).
    # The RMS slope s = sqrt(2) * sigma_h / l_c determines the angular spread
    # of scattered energy. Rougher relative to correlation → broader lobe.
    l_c_wavelengths = l_c / mi.Float(wavelength)
    kappa_spm = dr.sqrt(l_c_wavelengths)  # SPM-equivalent concentration
    roughness_slope = sigma_h / dr.maximum(l_c, mi.Float(1e-8))
    broadening = mi.Float(1.0) + mi.Float(5.0) * roughness_slope  # 1x smooth → ~2-3x rough
    kappa_dir = kappa_spm / broadening
    kappa_dir = dr.clamp(kappa_dir, mi.Float(0.3), mi.Float(5.0))

    # vMF distribution
    f_vmf = dr.exp(kappa_dir * (cos_dev - mi.Float(1.0)))

    # Normalization: κ / (4π × sinh(κ))
    sinh_kappa = (dr.exp(kappa_dir) - dr.exp(-kappa_dir)) / mi.Float(2.0)
    norm_factor = kappa_dir / (mi.Float(4.0 * np.pi) * dr.maximum(sinh_kappa, mi.Float(1e-6)))
    f_dir = f_vmf * norm_factor

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_dir, mi.Float(0.0))


def eval_lobe_broad(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    albedo: 'mi.Float',
) -> 'mi.Float':
    """
    Evaluate broad incoherent lobe (Lambertian-like).

    This represents volumetric/subsurface scattering and extreme roughness cases
    where surface features are much larger than the wavelength.

    Uses cosine-weighted hemisphere distribution (Lambertian).

    Args:
        wo: Outgoing direction
        wi: Incident direction
        n: Surface normal
        albedo: Material albedo (controls overall scattering strength)

    Returns:
        f_broad: Broad incoherent lobe value (1/sr)
    """
    cos_theta_i = dr.dot(wi, n)
    cos_theta_o = dr.dot(wo, n)

    # Lambertian: f = ρ / π
    # Scale by albedo squared for physical energy balance at mmWave
    # (albedo affects both absorption and scattering)
    f_broad = albedo / mi.Float(np.pi)

    valid = (cos_theta_i > mi.Float(1e-6)) & (cos_theta_o > mi.Float(1e-6))
    return dr.select(valid, f_broad, mi.Float(0.0))


def compute_cbs_factor(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """
    Coherent Backscatter Enhancement (CBS) factor.

    In monostatic/near-monostatic radar geometry, time-reversed scattering paths
    interfere constructively, producing a factor-of-2 enhancement in the exact
    backscatter direction. The enhancement decays over angular width
    ~ lambda / (2*pi*l_c).

    CBS = 1 + sinc^2(k * l_c * sin(theta_bs))

    where theta_bs is the angular deviation from exact retroreflection.

    Args:
        wo: Outgoing direction (towards receiver)
        wi: Incident direction (towards transmitter)
        n: Surface normal
        l_c: Correlation length (m)
        wavelength: Wavelength (m)

    Returns:
        cbs: Enhancement factor in [1, 2]
    """
    k = mi.Float(2.0 * np.pi / wavelength)

    # Retroreflection direction: reflect wi through surface, then negate
    # If wo == -wi_reflected, we're at exact backscatter
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    retro_dir = -wi_reflected  # wo should equal this for exact backscatter

    # Angular deviation from exact backscatter
    cos_bs = dr.clamp(dr.dot(wo, retro_dir), mi.Float(-1.0), mi.Float(1.0))
    sin_bs = dr.sqrt(dr.maximum(mi.Float(1.0) - cos_bs * cos_bs, mi.Float(1e-20)))

    # CBS kernel: sinc^2(k * l_c * sin_theta_bs)
    x = k * l_c * sin_bs
    sinc_x = dr.select(
        x < mi.Float(1e-6),
        mi.Float(1.0),
        dr.sin(x) / dr.maximum(x, mi.Float(1e-10))
    )
    cbs = mi.Float(1.0) + sinc_x * sinc_x  # Range: [1, 2]
    return cbs


def compute_cbs_hemispherical_mean(
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
) -> 'mi.Float':
    """
    Cosine-weighted hemispherical average of the CBS factor.

    Used to normalize CBS into an energy-preserving redistribution:
        f_inc *= cbs / cbs_mean

    The CBS factor is: CBS(theta) = 1 + sinc^2(k * l_c * sin theta)

    The cosine-weighted mean over the hemisphere is:
        <CBS> = 1 + I(k*l_c)
    where I(alpha) = 2 * integral_0^1 sinc^2(alpha*u) * u du

    For alpha >> 1: I(alpha) ~ 1/alpha  (narrow peak, small excess)
    For alpha -> 0: I(alpha) -> 1       (broad peak, doubles energy)

    We use a smooth rational approximation: I(alpha) ~ 1 / (1 + alpha)
    which satisfies both limits and is conservative (never creates energy).

    Args:
        l_c: Correlation length (m)
        wavelength: Wavelength (m)

    Returns:
        cbs_mean: Average CBS factor >= 1 (used as normalization denominator)
    """
    k = mi.Float(2.0 * np.pi / wavelength)
    alpha = k * l_c

    I_alpha = mi.Float(1.0) / (mi.Float(1.0) + alpha)
    cbs_mean = mi.Float(1.0) + I_alpha

    return cbs_mean


def compute_coherent_incoherent_blend(
    sigma_h: 'mi.Float',
    l_c: 'mi.Float',
    wavelength: float = WAVELENGTH_77GHZ,
    cos_theta_i: 'mi.Float' = None,
) -> tuple:
    """
    Compute η (coherent/incoherent) and γ (directive/broad) blend coefficients
    using the Rayleigh roughness factor.

    η is the coherent power fraction, computed as:
        g = (2 * k * sigma_h * cos_theta_i)^2
        η = exp(-g)

    This is physically motivated: the Rayleigh factor describes how much of
    the scattered field remains coherent (specular) vs incoherent (diffuse).
    Crucially, it is angle-dependent: even rough surfaces appear smooth at
    grazing incidence (cos_theta_i → 0 → g → 0 → η → 1).

    For incoherent blend:
    - Moderate roughness: Directive dominates (γ → 1)
    - Very rough: Broad dominates (γ → 0)

    Args:
        sigma_h: RMS surface height
        l_c: Correlation length
        wavelength: Wavelength
        cos_theta_i: Cosine of incidence angle. If None, uses 1.0 (normal incidence).

    Returns:
        (eta, gamma): Blend coefficients ∈ [0, 1]
    """
    k = mi.Float(2.0 * np.pi / wavelength)

    # Rayleigh factor for coherent power fraction
    if cos_theta_i is None:
        cos_theta_i = mi.Float(1.0)

    g = (mi.Float(2.0) * k * sigma_h * cos_theta_i)
    g = g * g  # g = (2*k*sigma_h*cos_theta)^2
    eta = dr.exp(-g)
    eta = dr.clamp(eta, mi.Float(0.01), mi.Float(0.99))

    # Directive vs broad: based on roughness relative to wavelength
    # sigma_h/λ < 0.01: mostly directive
    # sigma_h/λ > 0.1: mostly broad
    roughness_ratio = sigma_h / mi.Float(wavelength)
    gamma = _sigmoid((mi.Float(0.05) - roughness_ratio) * mi.Float(50.0))
    gamma = dr.clamp(gamma, mi.Float(0.1), mi.Float(0.9))

    return eta, gamma


# Reparameterization functions moved to bsdf/reparameterization.py
# Re-export for backward compatibility
from .reparameterization import (
    reparameterize_physics_params,
    inverse_reparameterize,
    reparameterize_physics_params_drjit,
    create_drjit_raw_params,
    LR_SCALES,
)


# =============================================================================
# COMPLETE BSDF CLASS
# =============================================================================

class BSDFmmWaveScalar(BSDFBase):
    """
    Physics-first hybrid BSDF for 77 GHz mmWave radar (CSV-BSDF equivalent).

    Implements KA + SPM mixture with:
    - Complex permittivity Fresnel (energy gate)
    - Angle-dependent blending (KA for small angles, SPM for large)
    - Validity regime constraints from the paper
    - Polarization through power fractions in s/p

    CRITICAL: eval_f() returns POWER (1/sr).
    The integrator must convert to amplitude via sqrt() for phasor synthesis.

    For coherent-only mode (default), this is equivalent to the CSV-BSDF paper.
    """

    WAVELENGTH = WAVELENGTH_77GHZ  # λ at 77 GHz (m)
    K = K_77GHZ  # Wavenumber

    def __init__(
        self,
        polarization: str = 'vertical',
        enable_incoherent: bool = True,
        enable_slab_fresnel: bool = True,
        default_thickness: float = 0.1,
        enable_cbs: bool = True,
    ):
        """
        Initialize scalar mmWave BSDF.

        Args:
            polarization: 'vertical', 'horizontal', or 'unpolarized'
            enable_incoherent: If False, use coherent-only mode (CSV-BSDF equivalent)
            enable_slab_fresnel: If True, use ITU slab Fresnel model with thickness.
                If False, use single-interface Fresnel (legacy behavior).
            default_thickness: Default slab thickness (m) when not provided per-vertex.
                Set to 0 to disable slab model. Default 0.1m (10cm wall).
            enable_cbs: If True, apply coherent backscatter enhancement to
                incoherent component (factor-of-2 at exact retroreflection).
        """
        self.polarization = polarization
        self.enable_incoherent = enable_incoherent
        self.enable_slab_fresnel = enable_slab_fresnel
        self.default_thickness = default_thickness
        self.enable_cbs = enable_cbs

        if polarization == 'vertical':
            self.antenna_pol = mi.Vector3f(0, 0, 1)
        elif polarization == 'horizontal':
            self.antenna_pol = mi.Vector3f(1, 0, 0)
        else:
            self.antenna_pol = None

    # =========================================================================
    # CORE EVALUATION (shared by legacy and physics paths)
    # =========================================================================

    def _eval_f_core(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau_base: 'mi.Float',
        thickness: 'mi.Float',
        albedo_broad: 'mi.Float' = None,
    ) -> 'mi.Float':
        """
        Core BSDF evaluation from physical parameters.

        Shared implementation for both legacy (CG params → map → physics) and
        direct physics parameter paths. Returns POWER (1/sr).

        Energy gating: f_total = A × [η × f_coh + (1 − η) × f_inc]
        All scattered energy (coherent and incoherent) comes from the reflected
        fraction A. The (1 − A) fraction is transmission + absorption.

        Args:
            albedo_broad: Scale for the broad Lambertian lobe. In legacy mode,
                this is the CG albedo (preserving exact backward compatibility).
                In physics mode, this is None → defaults to 1.0 (pure 1/π
                Lambertian; all energy scaling handled by gating chain).
        """
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        # Energy gate
        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)

        if self.enable_slab_fresnel:
            A = compute_slab_energy_gate(
                eps_real, eps_imag, cos_theta_i, thickness,
                frac_s, frac_p, self.WAVELENGTH,
                sigma_h=sigma_h
            )
        else:
            n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
            A = compute_energy_gate(n_ior, kappa, cos_theta_i, frac_s, frac_p)

        # Blend coefficients
        tau_eff = compute_validity_aware_blend(tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH)

        # Coherent lobes
        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh = tau_eff * f_KA + (mi.Float(1.0) - tau_eff) * f_SPM
        f_coh_gated = A * f_coh

        # Incoherent lobes
        if self.enable_incoherent:
            f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
            broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
            f_broad = eval_lobe_broad(wo, wi, n, broad_scale)

            eta, gamma = compute_coherent_incoherent_blend(
                sigma_h, l_c, self.WAVELENGTH, cos_theta_i
            )
            f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad

            if self.enable_cbs:
                cbs = compute_cbs_factor(wo, wi, n, l_c, self.WAVELENGTH)
                cbs_mean = compute_cbs_hemispherical_mean(l_c, self.WAVELENGTH)
                f_inc = f_inc * (cbs / cbs_mean)

            f_inc_gated = A * f_inc
            f_total = eta * f_coh_gated + (mi.Float(1.0) - eta) * f_inc_gated
        else:
            f_total = f_coh_gated

        return f_total

    def _eval_non_ka_core(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau_base: 'mi.Float',
        thickness: 'mi.Float',
        albedo_broad: 'mi.Float' = None,
    ) -> 'mi.Float':
        """
        Core non-KA BSDF evaluation. KA lobe excluded (handled by image method).

        Returns POWER (1/sr). Same physics as _eval_f_core minus the KA term.
        """
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        # Energy gate
        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)
        if self.enable_slab_fresnel:
            A = compute_slab_energy_gate(
                eps_real, eps_imag, cos_theta_i, thickness,
                frac_s, frac_p, self.WAVELENGTH,
                sigma_h=sigma_h
            )
        else:
            n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
            A = compute_energy_gate(n_ior, kappa, cos_theta_i, frac_s, frac_p)

        # Blend coefficients
        tau_eff = compute_validity_aware_blend(tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH)
        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        # Coherent SPM only (KA excluded — handled by image method)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh_no_ka = eta * A * (mi.Float(1.0) - tau_eff) * f_SPM

        # Incoherent (full)
        if self.enable_incoherent:
            f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
            broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
            f_broad = eval_lobe_broad(wo, wi, n, broad_scale)
            f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad

            if self.enable_cbs:
                cbs = compute_cbs_factor(wo, wi, n, l_c, self.WAVELENGTH)
                cbs_mean = compute_cbs_hemispherical_mean(l_c, self.WAVELENGTH)
                f_inc = f_inc * (cbs / cbs_mean)

            f_inc_gated = (mi.Float(1.0) - eta) * A * f_inc
        else:
            f_inc_gated = mi.Float(0.0)

        f_total = f_coh_no_ka + f_inc_gated

        return f_total

    def _eval_f_cos_components_core(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau_base: 'mi.Float',
        thickness: 'mi.Float',
        albedo_broad: 'mi.Float' = None,
    ) -> dict:
        """
        Core components evaluation. Same physics as _eval_f_core but returns
        per-lobe contributions and physics diagnostics for visualization.
        """
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        # Energy gate
        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)
        if self.enable_slab_fresnel:
            A = compute_slab_energy_gate(
                eps_real, eps_imag, cos_theta_i, thickness,
                frac_s, frac_p, self.WAVELENGTH,
                sigma_h=sigma_h
            )
        else:
            n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
            A = compute_energy_gate(n_ior, kappa, cos_theta_i, frac_s, frac_p)

        tau_eff = compute_validity_aware_blend(tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH)

        # Evaluate all lobes
        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
        f_broad = eval_lobe_broad(wo, wi, n, broad_scale)

        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        # Combined lobes
        f_coh = tau_eff * f_KA + (mi.Float(1.0) - tau_eff) * f_SPM
        f_coh_gated = A * f_coh

        if self.enable_incoherent:
            f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad
            f_inc_gated = A * f_inc
            f_total = eta * f_coh_gated + (mi.Float(1.0) - eta) * f_inc_gated

            f_ka_gated = A * tau_eff * f_KA * eta
            f_spm_gated = A * (mi.Float(1.0) - tau_eff) * f_SPM * eta
            f_dir_gated = A * gamma * f_dir * (mi.Float(1.0) - eta)
            f_broad_gated = A * (mi.Float(1.0) - gamma) * f_broad * (mi.Float(1.0) - eta)
        else:
            f_total = f_coh_gated
            f_ka_gated = A * tau_eff * f_KA
            f_spm_gated = A * (mi.Float(1.0) - tau_eff) * f_SPM
            f_dir_gated = mi.Float(0.0)
            f_broad_gated = mi.Float(0.0)
            f_inc_gated = mi.Float(0.0)

        return {
            'f_ka_cos': f_ka_gated * cos_theta_i,
            'f_spm_cos': f_spm_gated * cos_theta_i,
            'f_directive_cos': f_dir_gated * cos_theta_i,
            'f_broad_cos': f_broad_gated * cos_theta_i,
            'f_coherent_cos': (f_coh_gated * eta if self.enable_incoherent else f_coh_gated) * cos_theta_i,
            'f_incoherent_cos': (f_inc_gated if self.enable_incoherent else mi.Float(0.0)) * cos_theta_i,
            'f_total_cos': f_total * cos_theta_i,
            # Physics diagnostics (Phase 1D)
            'A': A,
            'eta': eta if self.enable_incoherent else mi.Float(1.0),
            'gamma': gamma if self.enable_incoherent else mi.Float(1.0),
            'tau_eff': tau_eff,
            'thickness': thickness,
        }

    # =========================================================================
    # LEGACY ENTRY POINTS (3-param: albedo, roughness, metallic)
    # =========================================================================

    def eval_f(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> 'mi.Float':
        """
        Evaluate BSDF f(wo, wi) - returns POWER (1/sr). Legacy CG param path.

        Maps CG params → physics via map_renderer_params_to_physical(), then
        delegates to _eval_f_core(). Uses global default_thickness (known
        limitation: per-triangle thickness requires --physics-mode).
        """
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                  thickness, albedo_broad=albedo)

    def eval_f_cos(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> 'mi.Float':
        """Evaluate BSDF × |cos(θ_i)|. Legacy CG param path."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_f(wo, wi, n, albedo, roughness, metallic)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_components(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> dict:
        """Evaluate BSDF × |cos(θ_i)| with per-lobe components. Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_cos_components_core(
            wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
            thickness, albedo_broad=albedo
        )

    def eval_non_ka(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> 'mi.Float':
        """
        Evaluate BSDF minus KA lobe (image method handles KA). Legacy CG param path.

        Returns POWER (1/sr). Integrator must sqrt() for amplitude.
        """
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_non_ka_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness, albedo_broad=albedo)

    def eval_non_ka_cos(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> 'mi.Float':
        """eval_non_ka × |cos(θ_i)| for MC integration. Legacy CG param path."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_non_ka(wo, wi, n, albedo, roughness, metallic)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    # =========================================================================
    # PHYSICS ENTRY POINTS (6-param: eps_real, eps_imag, sigma_h, l_c, tau,
    #                        thickness)
    # =========================================================================

    def eval_ka_only_physics(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """
        Evaluate KA-only BSDF: A × η × τ × f_KA. Returns POWER (1/sr).

        Used by Phase B specular to avoid double-counting with diffuse MC.
        Diffuse MC evaluates non-KA BSDF (everything except KA), so specular
        Phase B should evaluate only the KA component. Together they sum to
        the full BSDF without overlap.
        """
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        # Energy gate (same as _eval_f_core)
        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)
        if self.enable_slab_fresnel:
            A = compute_slab_energy_gate(
                eps_real, eps_imag, cos_theta_i, thickness,
                frac_s, frac_p, self.WAVELENGTH,
                sigma_h=sigma_h_v
            )
        else:
            n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
            A = compute_energy_gate(n_ior, kappa, cos_theta_i, frac_s, frac_p)

        # Blend coefficients
        tau_eff = compute_validity_aware_blend(tau, cos_theta_i, sigma_h_v, l_c_v, self.WAVELENGTH)
        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.WAVELENGTH, cos_theta_i
        )

        # KA lobe only
        f_KA = eval_lobe_KA(wo, wi, n, sigma_h_v, l_c_v, self.WAVELENGTH)

        return A * eta * tau_eff * f_KA

    def eval_specular_reflectance(self, cos_theta_i, eps_real, eps_imag, sigma_h, l_c,
                                    tau, thickness):
        """Compute integrated specular reflectance R_specular = η × τ_eff × A.

        This matches the SMS solver's _compute_specular_weight_physics exactly.
        Used for deterministic specular path weights (SMS), where the total
        reflected power matters rather than the directional BSDF density.
        """
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)

        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)
        if self.enable_slab_fresnel:
            A = compute_slab_energy_gate(
                eps_real, eps_imag, cos_theta_i, thickness,
                frac_s, frac_p, self.WAVELENGTH,
                sigma_h=sigma_h_v
            )
        else:
            n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
            A = compute_energy_gate(n_ior, kappa, cos_theta_i, frac_s, frac_p)

        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.WAVELENGTH, cos_theta_i
        )

        tau_eff = compute_validity_aware_blend(
            tau, cos_theta_i, sigma_h_v, l_c_v, self.WAVELENGTH
        )

        return eta * tau_eff * A

    def eval_f_physics(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """
        Evaluate BSDF f(wo, wi) from direct physics params. Returns POWER (1/sr).

        Bypasses map_renderer_params_to_physical() — all 6 physics DoF are
        independent. Broad lobe uses pure 1/π Lambertian (albedo_broad=None).
        """
        return self._eval_f_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                  thickness)

    def eval_f_cos_physics(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """Evaluate BSDF × |cos(θ_i)| from direct physics params."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_f_physics(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                 thickness)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_components_physics(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> dict:
        """
        Evaluate BSDF × |cos(θ_i)| with per-lobe components and physics
        diagnostics from direct physics params.

        Returns dict with f_ka_cos, f_spm_cos, ..., plus A, eta, gamma,
        tau_eff, thickness for diagnostic visualization.
        """
        return self._eval_f_cos_components_core(
            wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau, thickness
        )

    def eval_non_ka_physics(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """Evaluate BSDF minus KA lobe from direct physics params."""
        return self._eval_non_ka_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness)

    def eval_non_ka_cos_physics(
        self,
        wo: 'mi.Vector3f',
        wi: 'mi.Vector3f',
        n: 'mi.Vector3f',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """eval_non_ka × |cos(θ_i)| from direct physics params."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_non_ka_physics(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                      thickness)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def pdf_wi(self, wo, wi, n, roughness):
        """
        PDF for sampling (cosine-weighted hemisphere).

        Args:
            wo: Outgoing direction
            wi: Incident direction
            n: Surface normal
            roughness: Surface roughness (unused for cosine sampling)

        Returns:
            pdf: Probability density
        """
        cos_theta_i = dr.dot(wi, n)
        valid = cos_theta_i > mi.Float(1e-6)
        return dr.select(valid, cos_theta_i / mi.Float(np.pi), mi.Float(0.0))

    def sample_f(self, wo, n, albedo, roughness, metallic, samples):
        """
        Sample incident direction using cosine-weighted hemisphere.

        Args:
            wo: Outgoing direction
            n: Surface normal
            albedo: Diffuse reflectance
            roughness: Surface roughness
            metallic: Metallic factor
            samples: Random samples [n, 2]

        Returns:
            (wi_world, f_cos, pdf, is_delta): Sampled direction and associated values
        """
        wi_local, pdf = sample_cosine_hemisphere_concentric(samples)
        wi_world = to_global(wi_local, n)
        f_cos = self.eval_f_cos(wo, wi_world, n, albedo, roughness, metallic)
        is_delta = mi.Bool(False)
        return wi_world, f_cos, pdf, is_delta


# Tier-2 sampling functions moved to bsdf/sampling.py
# Re-export for backward compatibility
from .sampling import (
    sample_ggx_vndf_drjit,
    eval_ggx_pdf_drjit,
    sample_vmf_drjit,
    eval_vmf_pdf_drjit,
    compute_lobe_weights_drjit,
    compute_mixture_pdf_drjit,
    _build_tangent_frame_drjit,
    _world_to_local,
    _local_to_world,
)




# =============================================================================
# MATERIAL DATABASE
# =============================================================================
#
# Full ITU-R P.2040-4 database is in itu_materials.py.
# Import here for backward compatibility.

from ..materials.itu_materials import (
    ITU_MATERIAL_PROPERTIES,
    DEFAULT_ROUGHNESS,
    DEFAULT_THICKNESS,
    DEFAULT_THICKNESS_FALLBACK,
    get_itu_properties,
    get_material_properties,
    complex_relative_permittivity as complex_relative_permittivity_scalar,
)

# Legacy alias for code that references MATERIAL_DB_77GHZ directly
def _build_legacy_db():
    """Build 77 GHz snapshot for backward compatibility."""
    db = {}
    freq_hz = 77e9
    omega = 2.0 * np.pi * freq_hz
    for name in ITU_MATERIAL_PROPERTIES:
        try:
            props = get_material_properties(name, freq_hz)
            db[name] = {
                'eps_real': props['eps_real'],
                'eps_imag': props['eps_imag'],
                'sigma_h': props['sigma_h'],
                'l_c': props['l_c'],
                'tau': 0.5,  # Default; actual tau computed by map_renderer_params_to_physical
            }
        except Exception:
            pass
    return db

MATERIAL_DB_77GHZ = _build_legacy_db()
