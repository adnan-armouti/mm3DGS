"""
Polarization-aware Jones BSDF for 77 GHz mmWave radar.

This extends the CSV-BSDF equivalent (bsdf_mmwave_scalar.py) with:
- Full complex Fresnel coefficients (amplitude + phase)
- Jones matrix formalism for polarization transformation
- TX/RX polarization state tracking
- Polarization-dependent power computation
- Optional Fresnel phase in coherent scattering

The Jones formalism properly accounts for:
1. Phase relationships between s and p polarizations
2. Cross-polarization coupling at reflection
3. Coherent interference from polarization mixing
4. Elliptical polarization states (if TX/RX are not purely linear)

MODES:
- use_fresnel_phase=False: eval_f() returns POWER (compatible with standard integrator)
- use_fresnel_phase=True: eval_f_cos_complex() returns complex amplitude for coherent term

Reference: arXiv:2401.01175 (CSV-BSDF paper)
"""

import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Tuple

from .base import BSDFBase
from ..utils.math import sample_cosine_hemisphere_concentric, to_global

# Import shared utilities from scalar version
from .mmwave_scalar import (
    WAVELENGTH_77GHZ,
    K_77GHZ,
    _sigmoid,
    permittivity_to_ior,
    enforce_spm_validity,
    compute_validity_aware_blend,
    map_renderer_params_to_physical,
    eval_lobe_KA,
    eval_lobe_SPM,
    eval_lobe_directive,
    eval_lobe_broad,
    compute_coherent_incoherent_blend,
    compute_cbs_factor,
    compute_cbs_hemispherical_mean,
    itu_slab_fresnel,
    compute_slab_energy_gate,
    # Tier 2 sampling primitives
    sample_ggx_vndf_drjit,
    eval_ggx_pdf_drjit,
    sample_vmf_drjit,
    eval_vmf_pdf_drjit,
    compute_lobe_weights_drjit,
    compute_mixture_pdf_drjit,
    _build_tangent_frame_drjit,
    _local_to_world,
    _world_to_local,
)


# =============================================================================
# JONES POLARIZATION UTILITIES
# =============================================================================

def fresnel_complex_amplitude(
    n: 'mi.Float',
    kappa: 'mi.Float',
    cos_theta_i: 'mi.Float',
) -> tuple:
    """
    Compute COMPLEX Fresnel amplitude coefficients r_s and r_p.

    Unlike fresnel_complex_power() which returns |r|², this returns the
    complex coefficients with both magnitude and phase information.

    For complex IOR ñ = n - jκ:
    - r_s = (cos(θ_i) - ξ) / (cos(θ_i) + ξ)
    - r_p = (ñ² cos(θ_i) - ξ) / (ñ² cos(θ_i) + ξ)

    where ξ = sqrt(ñ² - sin²(θ_i)) = a + jb

    Args:
        n: Real part of complex IOR
        kappa: Imaginary part of complex IOR (extinction coefficient)
        cos_theta_i: Cosine of incident angle

    Returns:
        (r_s_real, r_s_imag, r_p_real, r_p_imag): Complex Fresnel coefficients
    """
    sin2_theta_i = mi.Float(1.0) - cos_theta_i * cos_theta_i

    # ñ² = (n - jκ)² = n² - κ² - 2jnκ
    n2_sq_real = n * n - kappa * kappa
    n2_sq_imag = mi.Float(-2.0) * n * kappa

    # ξ = ñ² - sin²(θ_i) = (n² - κ² - sin²θ) - 2jnκ
    xi_real = n2_sq_real - sin2_theta_i
    xi_imag = n2_sq_imag

    # Complex square root: sqrt(ξ) = a + jb
    # Using polar form: sqrt(|ξ|) × exp(j × arg(ξ)/2)
    xi_mag_sq = xi_real * xi_real + xi_imag * xi_imag
    xi_mag = dr.sqrt(dr.maximum(xi_mag_sq, mi.Float(1e-20)))
    xi_arg = dr.atan2(xi_imag, dr.maximum(xi_real, mi.Float(1e-10)))

    a = dr.sqrt(dr.maximum(xi_mag, mi.Float(1e-20))) * dr.cos(xi_arg / mi.Float(2.0))
    b = dr.sqrt(dr.maximum(xi_mag, mi.Float(1e-20))) * dr.sin(xi_arg / mi.Float(2.0))

    # r_s = (cos_θ - (a+jb)) / (cos_θ + (a+jb))
    # = ((cos_θ - a) - jb) / ((cos_θ + a) + jb)
    rs_num_real = cos_theta_i - a
    rs_num_imag = -b
    rs_den_real = cos_theta_i + a
    rs_den_imag = b

    # Complex division: (a+jb)/(c+jd) = ((ac+bd) + j(bc-ad)) / (c²+d²)
    rs_den_sq = dr.maximum(rs_den_real * rs_den_real + rs_den_imag * rs_den_imag, mi.Float(1e-10))
    r_s_real = (rs_num_real * rs_den_real + rs_num_imag * rs_den_imag) / rs_den_sq
    r_s_imag = (rs_num_imag * rs_den_real - rs_num_real * rs_den_imag) / rs_den_sq

    # r_p = (ñ² cos_θ - (a+jb)) / (ñ² cos_θ + (a+jb))
    # ñ² cos_θ = (n2_sq_real + j*n2_sq_imag) × cos_θ
    n2cos_real = n2_sq_real * cos_theta_i
    n2cos_imag = n2_sq_imag * cos_theta_i

    rp_num_real = n2cos_real - a
    rp_num_imag = n2cos_imag - b
    rp_den_real = n2cos_real + a
    rp_den_imag = n2cos_imag + b

    rp_den_sq = dr.maximum(rp_den_real * rp_den_real + rp_den_imag * rp_den_imag, mi.Float(1e-10))
    r_p_real = (rp_num_real * rp_den_real + rp_num_imag * rp_den_imag) / rp_den_sq
    r_p_imag = (rp_num_imag * rp_den_real - rp_num_real * rp_den_imag) / rp_den_sq

    return r_s_real, r_s_imag, r_p_real, r_p_imag


def compute_sp_basis_jones(wi: 'mi.Vector3f', n: 'mi.Vector3f') -> tuple:
    """
    Compute s and p polarization basis vectors (incident side).

    Conventions:
    - s: perpendicular to plane of incidence, "transverse electric" (TE)
    - p: in plane of incidence, "transverse magnetic" (TM)

    For incident wave traveling in direction -wi:
    - s = wi × n (normalized)
    - p = s × wi (in plane containing wi and n)

    Args:
        wi: Incident direction (pointing towards surface)
        n: Surface normal

    Returns:
        (s, p): Polarization basis vectors (normalized)
    """
    # s is perpendicular to plane of incidence
    s_unnorm = dr.cross(wi, n)
    s_len = dr.norm(s_unnorm)

    # Handle grazing incidence (s becomes undefined)
    valid = s_len > mi.Float(1e-6)

    # Fallback: use global x or y axis
    fallback_s = dr.cross(wi, mi.Vector3f(1, 0, 0))
    fallback_len = dr.norm(fallback_s)
    fallback_s2 = dr.cross(wi, mi.Vector3f(0, 1, 0))
    fallback_s2_len = dr.sqrt(dr.maximum(dr.dot(fallback_s2, fallback_s2), mi.Float(1e-20)))
    fallback_s = dr.select(
        fallback_len > mi.Float(1e-6),
        fallback_s / dr.maximum(fallback_len, mi.Float(1e-10)),
        fallback_s2 / fallback_s2_len
    )

    s = dr.select(valid, s_unnorm / dr.maximum(s_len, mi.Float(1e-10)), fallback_s)

    # p = s × wi gives vector in plane of incidence, perpendicular to wi
    p_unnorm = dr.cross(s, wi)
    p_len = dr.sqrt(dr.maximum(dr.dot(p_unnorm, p_unnorm), mi.Float(1e-20)))
    p = p_unnorm / p_len

    return s, p


def project_polarization_to_sp(
    pol_vec: 'mi.Vector3f',
    s: 'mi.Vector3f',
    p: 'mi.Vector3f',
) -> tuple:
    """
    Project a polarization vector onto the s/p basis.

    For a linearly polarized wave with E-field direction pol_vec,
    compute the amplitude components in s and p directions.

    Args:
        pol_vec: Polarization direction (E-field direction, normalized)
        s: s-polarization basis vector
        p: p-polarization basis vector

    Returns:
        (amp_s, amp_p): Complex amplitudes in s and p (real for linear polarization)
    """
    amp_s = dr.dot(pol_vec, s)
    amp_p = dr.dot(pol_vec, p)
    return amp_s, amp_p


def apply_jones_reflection(
    E_s_in: 'mi.Float',
    E_p_in: 'mi.Float',
    r_s_real: 'mi.Float',
    r_s_imag: 'mi.Float',
    r_p_real: 'mi.Float',
    r_p_imag: 'mi.Float',
) -> tuple:
    """
    Apply Jones matrix for reflection.

    The reflection Jones matrix is diagonal (no polarization mixing for
    isotropic surfaces):

    [E_s_out]   [r_s   0 ] [E_s_in]
    [E_p_out] = [0    r_p] [E_p_in]

    Args:
        E_s_in, E_p_in: Input field amplitudes (real for linear input)
        r_s_real, r_s_imag: Complex s-polarization reflection coefficient
        r_p_real, r_p_imag: Complex p-polarization reflection coefficient

    Returns:
        (E_s_out_real, E_s_out_imag, E_p_out_real, E_p_out_imag): Output field
    """
    # E_s_out = r_s × E_s_in = (r_s_real + j*r_s_imag) × E_s_in
    # For real input: E_out = E_in × r = E_in × (r_r + j*r_i)
    E_s_out_real = r_s_real * E_s_in
    E_s_out_imag = r_s_imag * E_s_in

    E_p_out_real = r_p_real * E_p_in
    E_p_out_imag = r_p_imag * E_p_in

    return E_s_out_real, E_s_out_imag, E_p_out_real, E_p_out_imag


def compute_received_field_complex(
    E_s_out_real: 'mi.Float',
    E_s_out_imag: 'mi.Float',
    E_p_out_real: 'mi.Float',
    E_p_out_imag: 'mi.Float',
    rx_amp_s: 'mi.Float',
    rx_amp_p: 'mi.Float',
) -> Tuple['mi.Float', 'mi.Float']:
    """
    Compute received complex field by projecting scattered field onto RX polarization.

    E_received = E_s_out × rx_amp_s + E_p_out × rx_amp_p

    Args:
        E_s_out_real, E_s_out_imag: Scattered s-polarization field (complex)
        E_p_out_real, E_p_out_imag: Scattered p-polarization field (complex)
        rx_amp_s, rx_amp_p: RX antenna polarization in s/p basis

    Returns:
        (E_rx_real, E_rx_imag): Complex received field amplitude
    """
    # E_received = E_s × rx_s + E_p × rx_p (complex)
    E_rx_real = E_s_out_real * rx_amp_s + E_p_out_real * rx_amp_p
    E_rx_imag = E_s_out_imag * rx_amp_s + E_p_out_imag * rx_amp_p

    return E_rx_real, E_rx_imag


def compute_received_power(
    E_s_out_real: 'mi.Float',
    E_s_out_imag: 'mi.Float',
    E_p_out_real: 'mi.Float',
    E_p_out_imag: 'mi.Float',
    rx_amp_s: 'mi.Float',
    rx_amp_p: 'mi.Float',
) -> 'mi.Float':
    """
    Compute received power by projecting scattered field onto RX polarization.

    Power = |E_received|² = |E_s_out × rx_amp_s + E_p_out × rx_amp_p|²

    Args:
        E_s_out_real, E_s_out_imag: Scattered s-polarization field (complex)
        E_p_out_real, E_p_out_imag: Scattered p-polarization field (complex)
        rx_amp_s, rx_amp_p: RX antenna polarization in s/p basis

    Returns:
        P_rx: Received power (proportional to |E·pol|²)
    """
    E_rx_real, E_rx_imag = compute_received_field_complex(
        E_s_out_real, E_s_out_imag,
        E_p_out_real, E_p_out_imag,
        rx_amp_s, rx_amp_p
    )

    # Power = |E|²
    P_rx = E_rx_real * E_rx_real + E_rx_imag * E_rx_imag

    return P_rx


# =============================================================================
# MAIN BSDF CLASS
# =============================================================================

# =============================================================================
# PHASE 7: POLARIMETRIC SPM KERNELS
# =============================================================================

def spm_polarimetric_kernel(
    wo: 'mi.Vector3f',
    wi: 'mi.Vector3f',
    n: 'mi.Vector3f',
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
) -> tuple:
    """
    First-order SPM scattering coefficients for HH and VV polarizations.

    Implements the alpha_hh and alpha_vv polarimetric coupling coefficients
    from Ulaby, Moore & Fung, "Microwave Remote Sensing", Vol II, Eq 12.12.

    These modulate the SPM lobe to account for polarization-dependent scattering
    cross-sections. HH and VV have different angular responses, especially near
    the Brewster angle where VV scattering vanishes.

    IMPORTANT: phi_s is the azimuthal scattering angle measured relative to the
    plane of incidence. Since our BSDF receives wo/wi in world coordinates, we
    extract phi_s from the azimuthal difference:
        phi_s = atan2(wo_proj.y, wo_proj.x) - atan2(wi_r_proj.y, wi_r_proj.x)
    where wi_r is the specular reflection of wi and projections are into the
    surface tangent plane.

    Args:
        wo: Outgoing direction (towards receiver)
        wi: Incident direction (towards transmitter)
        n: Surface normal
        eps_real: Real relative permittivity
        eps_imag: Imaginary permittivity (magnitude)

    Returns:
        (alpha_hh_sq, alpha_vv_sq): |alpha_HH|^2 and |alpha_VV|^2 (power)
    """
    cos_i = dr.maximum(dr.dot(wi, n), mi.Float(1e-6))
    cos_o = dr.maximum(dr.dot(wo, n), mi.Float(1e-6))
    sin_i = dr.sqrt(dr.maximum(mi.Float(1.0) - cos_i * cos_i, mi.Float(1e-20)))
    sin_o = dr.sqrt(dr.maximum(mi.Float(1.0) - cos_o * cos_o, mi.Float(1e-20)))

    # Compute azimuthal angle phi_s between wo and specular reflection of wi
    # Project wo and wi_reflected into tangent plane, then compute azimuthal diff
    wi_reflected = mi.Float(2.0) * dr.dot(wi, n) * n - wi
    wo_proj = wo - dr.dot(wo, n) * n
    wir_proj = wi_reflected - dr.dot(wi_reflected, n) * n
    wo_proj_len = dr.maximum(dr.norm(wo_proj), mi.Float(1e-10))
    wir_proj_len = dr.maximum(dr.norm(wir_proj), mi.Float(1e-10))

    # cos(phi_s) from dot product of tangent-plane projections
    cos_phi = dr.dot(wo_proj, wir_proj) / (wo_proj_len * wir_proj_len)
    cos_phi = dr.clamp(cos_phi, mi.Float(-1.0), mi.Float(1.0))

    # Complex permittivity: eta = eps_real - j*eps_imag
    # For the SPM kernels we need sqrt(eta - sin^2(theta))
    # Use real-part approximation for the denominator terms
    eta_r = eps_real
    eta_i = -eps_imag

    # sqrt(eta - sin^2(theta_i)) approx
    arg_i_r = eta_r - sin_i * sin_i
    arg_i_mag = dr.sqrt(dr.maximum(arg_i_r * arg_i_r + eta_i * eta_i, mi.Float(1e-20)))
    sqrt_eta_sin_i = dr.sqrt(dr.maximum(arg_i_mag, mi.Float(1e-20)))

    arg_o_r = eta_r - sin_o * sin_o
    arg_o_mag = dr.sqrt(dr.maximum(arg_o_r * arg_o_r + eta_i * eta_i, mi.Float(1e-20)))
    sqrt_eta_sin_o = dr.sqrt(dr.maximum(arg_o_mag, mi.Float(1e-20)))

    # Denominators
    den_i = cos_i + sqrt_eta_sin_i
    den_o = cos_o + sqrt_eta_sin_o
    denom = dr.maximum(den_i * den_o, mi.Float(1e-10))

    # alpha_HH = -(eta - 1) * cos_phi / denom
    eta_minus_1_mag_sq = (eta_r - mi.Float(1.0)) * (eta_r - mi.Float(1.0)) + eta_i * eta_i
    alpha_hh_sq = eta_minus_1_mag_sq * cos_phi * cos_phi / (denom * denom)

    # alpha_VV = eta * (eta - 1) * (sin_i*sin_o - (1+cos_i*cos_o)*cos_phi) / denom
    # |eta|^2 for the eta prefactor
    eta_mag_sq = eta_r * eta_r + eta_i * eta_i
    bracket = sin_i * sin_o - (mi.Float(1.0) + cos_i * cos_o) * cos_phi
    alpha_vv_sq = eta_mag_sq * eta_minus_1_mag_sq * bracket * bracket / (denom * denom)

    # Normalize: scale so that at normal incidence (cos_i=cos_o=1, phi_s=pi),
    # alpha_hh and alpha_vv are comparable to 1
    # This ensures the SPM kernel modulates but doesn't dominate
    norm = dr.maximum(alpha_hh_sq + alpha_vv_sq, mi.Float(1e-10))
    alpha_hh_sq = alpha_hh_sq / norm
    alpha_vv_sq = alpha_vv_sq / norm

    return alpha_hh_sq, alpha_vv_sq


class BSDFmmWaveJones(BSDFBase):
    """
    Polarization-aware Jones BSDF for 77 GHz mmWave radar.

    Extends the KA + SPM mixture model with proper polarization handling:
    - Complex Fresnel coefficients with phase
    - Jones matrix reflection transformation
    - TX/RX polarization state matching

    Key differences from BSDFmmWaveScalar:
    1. Computes complex r_s, r_p (not just R_s, R_p power reflectances)
    2. Applies Jones matrix to transform TX polarization through reflection
    3. Projects onto RX polarization to compute received power
    4. Optionally returns complex amplitude with Fresnel phase (use_fresnel_phase=True)

    MODES:
    - use_fresnel_phase=False (default): Returns power, compatible with standard integrator
    - use_fresnel_phase=True: Returns complex amplitude for coherent term via eval_f_cos_complex()

    CRITICAL:
    - eval_f() and eval_f_cos() always return POWER (1/sr)
    - eval_f_cos_complex() returns (E_coh_real, E_coh_imag, f_inc_power) when use_fresnel_phase=True
    """

    WAVELENGTH = WAVELENGTH_77GHZ
    K = K_77GHZ

    def __init__(
        self,
        tx_polarization: 'mi.Vector3f' = None,
        rx_polarization: 'mi.Vector3f' = None,
        enable_incoherent: bool = True,
        use_fresnel_phase: bool = False,
        enable_cbs: bool = True,
        enable_polarimetric_spm: bool = True,
        enable_slab_fresnel: bool = True,
        default_thickness: float = 0.1,
    ):
        """
        Initialize Jones BSDF.

        Args:
            tx_polarization: TX antenna polarization direction (E-field)
                             Default: vertical (0, 0, 1)
            rx_polarization: RX antenna polarization direction (E-field)
                             Default: same as TX (matched polarization)
            enable_incoherent: Enable incoherent scattering lobes
            use_fresnel_phase: If True, eval_f_cos_complex() returns complex amplitude
                              with Fresnel phase for coherent term. Incoherent term
                              always returns power (no Fresnel phase).
            enable_cbs: If True, apply coherent backscatter enhancement to
                incoherent component (factor-of-2 at exact retroreflection).
            enable_polarimetric_spm: If True, modulate SPM lobe by HH/VV
                polarimetric coupling coefficients from first-order SPM theory.
            enable_slab_fresnel: If True, use ITU slab Fresnel for complex
                coefficients (thickness-dependent interference with phase).
            default_thickness: Default slab thickness (m) for legacy path.
        """
        if tx_polarization is None:
            tx_polarization = mi.Vector3f(0, 0, 1)  # Vertical
        if rx_polarization is None:
            rx_polarization = tx_polarization  # Matched by default

        self.tx_polarization = tx_polarization
        self.rx_polarization = rx_polarization
        self.enable_incoherent = enable_incoherent
        self.use_fresnel_phase = use_fresnel_phase
        self.enable_cbs = enable_cbs
        self.enable_polarimetric_spm = enable_polarimetric_spm
        self.enable_slab_fresnel = enable_slab_fresnel
        self.default_thickness = default_thickness

    def _compute_jones_fresnel_complex(
        self,
        wi: 'mi.Vector3f',
        wo: 'mi.Vector3f',
        n: 'mi.Vector3f',
        n_ior: 'mi.Float',
        kappa: 'mi.Float',
        cos_theta_i: 'mi.Float',
        eps_real: 'mi.Float' = None,
        eps_imag: 'mi.Float' = None,
        thickness: 'mi.Float' = None,
    ) -> Tuple['mi.Float', 'mi.Float']:
        """
        Compute polarization-weighted COMPLEX Fresnel field using Jones formalism.

        Full pipeline:
        1. Compute s/p basis from incident direction
        2. Project TX polarization onto s/p
        3. Apply complex Fresnel reflection (preserves phase!)
           - If enable_slab_fresnel and thickness provided: use itu_slab_fresnel()
             for thickness-dependent interference with Fresnel phase
           - Otherwise: single-interface fresnel_complex_amplitude()
        4. Project onto RX polarization
        5. Return complex field (E_real, E_imag)

        Args:
            wi: Incident direction (towards TX)
            wo: Outgoing direction (towards RX)
            n: Surface normal
            n_ior: Real part of complex IOR
            kappa: Imaginary part of complex IOR
            cos_theta_i: Cosine of incident angle
            eps_real: Real permittivity (needed for slab Fresnel)
            eps_imag: Imaginary permittivity (needed for slab Fresnel)
            thickness: Slab thickness (needed for slab Fresnel)

        Returns:
            (E_real, E_imag): Complex Fresnel field amplitude (includes phase!)
        """
        # Step 1: Compute s/p basis for incident wave
        s_in, p_in = compute_sp_basis_jones(wi, n)

        # Step 2: Project TX polarization onto incident s/p basis
        tx_amp_s, tx_amp_p = project_polarization_to_sp(self.tx_polarization, s_in, p_in)

        # Step 3: Compute complex Fresnel coefficients
        if self.enable_slab_fresnel and thickness is not None and eps_real is not None:
            # Slab Fresnel: complex R with thickness-dependent interference phase.
            # itu_slab_fresnel returns (R_TE_r, R_TE_i, R_TM_r, R_TM_i, T_TE_r, T_TE_i, T_TM_r, T_TM_i)
            R_TE_r, R_TE_i, R_TM_r, R_TM_i, _, _, _, _ = itu_slab_fresnel(
                eps_real, eps_imag, cos_theta_i, thickness, self.WAVELENGTH
            )
            r_s_real, r_s_imag = R_TE_r, R_TE_i
            r_p_real, r_p_imag = R_TM_r, R_TM_i
        else:
            # Single-interface Fresnel (legacy)
            r_s_real, r_s_imag, r_p_real, r_p_imag = fresnel_complex_amplitude(
                n_ior, kappa, cos_theta_i
            )

        # Step 4: Apply Jones reflection matrix
        E_s_out_real, E_s_out_imag, E_p_out_real, E_p_out_imag = apply_jones_reflection(
            tx_amp_s, tx_amp_p,
            r_s_real, r_s_imag,
            r_p_real, r_p_imag
        )

        # Step 5: Compute s/p basis for outgoing wave
        s_out = s_in  # Same as incident (perpendicular to plane of incidence)
        p_out_unnorm = dr.cross(s_out, wo)
        p_out_len = dr.sqrt(dr.maximum(dr.dot(p_out_unnorm, p_out_unnorm), mi.Float(1e-20)))
        p_out = p_out_unnorm / p_out_len

        # Step 6: Project RX polarization onto outgoing s/p basis
        rx_amp_s, rx_amp_p = project_polarization_to_sp(self.rx_polarization, s_out, p_out)

        # Step 7: Compute received COMPLEX field (preserves Fresnel phase!)
        E_rx_real, E_rx_imag = compute_received_field_complex(
            E_s_out_real, E_s_out_imag,
            E_p_out_real, E_p_out_imag,
            rx_amp_s, rx_amp_p
        )

        return E_rx_real, E_rx_imag

    def _compute_jones_fresnel_power(
        self,
        wi: 'mi.Vector3f',
        wo: 'mi.Vector3f',
        n: 'mi.Vector3f',
        n_ior: 'mi.Float',
        kappa: 'mi.Float',
        cos_theta_i: 'mi.Float',
        eps_real: 'mi.Float' = None,
        eps_imag: 'mi.Float' = None,
        thickness: 'mi.Float' = None,
    ) -> 'mi.Float':
        """
        Compute polarization-weighted Fresnel POWER reflectance using Jones formalism.

        This collapses the complex field to power: R = |E|²

        When eps_real, eps_imag, and thickness are provided and enable_slab_fresnel
        is True, uses the slab Fresnel model with thickness-dependent interference.

        Returns:
            R_jones: Polarization-weighted power reflectance
        """
        E_real, E_imag = self._compute_jones_fresnel_complex(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )
        R_jones = E_real * E_real + E_imag * E_imag
        return dr.clamp(R_jones, mi.Float(0.0), mi.Float(1.0))

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
        """Core Jones BSDF evaluation. Returns POWER (1/sr)."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )

        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh = tau_eff * f_KA + (mi.Float(1.0) - tau_eff) * f_SPM
        f_coh_gated = R_jones * f_coh

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

            f_inc_gated = R_jones * f_inc
            f_total = eta * f_coh_gated + (mi.Float(1.0) - eta) * f_inc_gated
        else:
            f_total = f_coh_gated

        return f_total

    def _eval_f_cos_complex_core(
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
    ) -> Tuple['mi.Float', 'mi.Float', 'mi.Float']:
        """Core complex coherent + power incoherent evaluation."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        # Complex Fresnel with slab if thickness provided
        E_fresnel_real, E_fresnel_imag = self._compute_jones_fresnel_complex(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        R_jones = E_fresnel_real * E_fresnel_real + E_fresnel_imag * E_fresnel_imag
        R_jones = dr.clamp(R_jones, mi.Float(0.0), mi.Float(1.0))

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )

        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh = tau_eff * f_KA + (mi.Float(1.0) - tau_eff) * f_SPM
        A_coh_lobe = dr.sqrt(dr.maximum(f_coh, mi.Float(1e-20)))

        E_coh_real = E_fresnel_real * A_coh_lobe * cos_theta_i
        E_coh_imag = E_fresnel_imag * A_coh_lobe * cos_theta_i

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

            f_inc_gated = R_jones * f_inc

            sqrt_eta = dr.sqrt(dr.maximum(eta, mi.Float(1e-20)))
            E_coh_real = E_coh_real * sqrt_eta
            E_coh_imag = E_coh_imag * sqrt_eta
            f_inc_cos = (mi.Float(1.0) - eta) * f_inc_gated * cos_theta_i
        else:
            f_inc_cos = mi.Float(0.0)

        return E_coh_real, E_coh_imag, f_inc_cos

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
        """Core non-KA Jones BSDF evaluation. Returns POWER (1/sr)."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )
        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_coh_no_ka = eta * R_jones * (mi.Float(1.0) - tau_eff) * f_SPM

        if self.enable_incoherent:
            f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
            broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
            f_broad = eval_lobe_broad(wo, wi, n, broad_scale)
            f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad

            if self.enable_cbs:
                cbs = compute_cbs_factor(wo, wi, n, l_c, self.WAVELENGTH)
                cbs_mean = compute_cbs_hemispherical_mean(l_c, self.WAVELENGTH)
                f_inc = f_inc * (cbs / cbs_mean)

            f_inc_gated = (mi.Float(1.0) - eta) * R_jones * f_inc
        else:
            f_inc_gated = mi.Float(0.0)

        f_total = f_coh_no_ka + f_inc_gated

        return f_total

    def _eval_ka_only_core(
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
    ) -> 'mi.Float':
        """KA-only Jones BSDF: R_jones × η × τ_eff × f_KA. Returns POWER (1/sr).

        Used by specular synthesis to avoid double-counting with diffuse MC
        which evaluates non-KA BSDF. Together KA-only + non-KA = full BSDF.
        """
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )
        eta, _ = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)

        return eta * R_jones * tau_eff * f_KA

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
        """Core components evaluation with per-lobe contributions."""
        sigma_h, l_c = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)
        cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))

        R_jones = self._compute_jones_fresnel_power(
            wi, wo, n, n_ior, kappa, cos_theta_i,
            eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
        )

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.WAVELENGTH
        )

        f_KA = eval_lobe_KA(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        f_SPM = eval_lobe_SPM(wo, wi, n, sigma_h, l_c, self.WAVELENGTH, eps_real, eps_imag)
        f_dir = eval_lobe_directive(wo, wi, n, sigma_h, l_c, self.WAVELENGTH)
        broad_scale = albedo_broad if albedo_broad is not None else mi.Float(1.0)
        f_broad = eval_lobe_broad(wo, wi, n, broad_scale)

        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h, l_c, self.WAVELENGTH, cos_theta_i
        )

        f_coh = tau_eff * f_KA + (mi.Float(1.0) - tau_eff) * f_SPM
        f_coh_gated = R_jones * f_coh

        if self.enable_incoherent:
            f_inc = gamma * f_dir + (mi.Float(1.0) - gamma) * f_broad
            f_inc_gated = R_jones * f_inc
            f_total = eta * f_coh_gated + (mi.Float(1.0) - eta) * f_inc_gated

            f_ka_gated = R_jones * tau_eff * f_KA * eta
            f_spm_gated = R_jones * (mi.Float(1.0) - tau_eff) * f_SPM * eta
            f_dir_gated = R_jones * gamma * f_dir * (mi.Float(1.0) - eta)
            f_broad_gated = R_jones * (mi.Float(1.0) - gamma) * f_broad * (mi.Float(1.0) - eta)
        else:
            f_total = f_coh_gated
            f_ka_gated = R_jones * tau_eff * f_KA
            f_spm_gated = R_jones * (mi.Float(1.0) - tau_eff) * f_SPM
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
            # Physics diagnostics
            'R_jones': R_jones,
            'eta': eta if self.enable_incoherent else mi.Float(1.0),
            'gamma': gamma if self.enable_incoherent else mi.Float(1.0),
            'tau_eff': tau_eff,
            'thickness': thickness,
        }

    # =========================================================================
    # LEGACY ENTRY POINTS (3-param: albedo, roughness, metallic)
    # =========================================================================

    def eval_f(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF f(wo, wi) - returns POWER (1/sr). Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                  thickness, albedo_broad=albedo)

    def eval_f_cos(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF × |cos(θ_i)|. Legacy CG param path."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_f(wo, wi, n, albedo, roughness, metallic)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_complex(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate complex coherent + power incoherent. Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_cos_complex_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                              thickness, albedo_broad=albedo)

    def eval_non_ka(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF minus KA lobe. Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_non_ka_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness, albedo_broad=albedo)

    def eval_non_ka_cos(self, wo, wi, n, albedo, roughness, metallic):
        """eval_non_ka × |cos(θ_i)|. Legacy CG param path."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_non_ka(wo, wi, n, albedo, roughness, metallic)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_components(self, wo, wi, n, albedo, roughness, metallic):
        """Evaluate BSDF × |cos(θ_i)| with per-lobe components. Legacy CG param path."""
        eps_real, eps_imag, sigma_h, l_c, tau = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.WAVELENGTH)
        thickness = mi.Float(self.default_thickness)
        return self._eval_f_cos_components_core(
            wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
            thickness, albedo_broad=albedo
        )

    # =========================================================================
    # PHYSICS ENTRY POINTS (6-param: eps_real, eps_imag, sigma_h, l_c, tau,
    #                        thickness)
    # =========================================================================

    def eval_f_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                        thickness):
        """Evaluate BSDF f(wo, wi) from direct physics params. Returns POWER."""
        return self._eval_f_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                  thickness)

    def eval_f_cos_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                            thickness):
        """Evaluate BSDF × |cos(θ_i)| from direct physics params."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_f_physics(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                 thickness)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_f_cos_complex_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                    thickness):
        """Evaluate complex coherent + power incoherent from direct physics params."""
        return self._eval_f_cos_complex_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                              thickness)

    def eval_non_ka_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                             thickness):
        """Evaluate BSDF minus KA lobe from direct physics params."""
        return self._eval_non_ka_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness)

    def eval_non_ka_cos_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                 thickness):
        """eval_non_ka × |cos(θ_i)| from direct physics params."""
        cos_theta_i = dr.dot(wi, n)
        f = self.eval_non_ka_physics(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                      thickness)
        return f * dr.maximum(cos_theta_i, mi.Float(0.0))

    def eval_ka_only_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                              thickness):
        """Evaluate KA-only BSDF: R_jones × η × τ_eff × f_KA from direct physics params."""
        return self._eval_ka_only_core(wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                        thickness)

    def eval_specular_reflectance(self, cos_theta_i, eps_real, eps_imag, sigma_h, l_c,
                                    tau, thickness):
        """Compute integrated specular reflectance R_specular = η × τ_eff × A.

        This matches the SMS solver's _compute_specular_weight_physics exactly.
        Used for deterministic specular path weights (SMS), where the total
        reflected power matters rather than the directional BSDF density.

        Unlike eval_ka_only_physics (which includes the directional f_KA lobe),
        this returns a scalar reflectance coefficient independent of direction.
        """
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.WAVELENGTH)

        A = compute_slab_energy_gate(
            eps_real, eps_imag, cos_theta_i, thickness,
            mi.Float(0.5), mi.Float(0.5),
            wavelength=self.WAVELENGTH,
            sigma_h=sigma_h_v,
        )

        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.WAVELENGTH, cos_theta_i
        )

        tau_eff = compute_validity_aware_blend(
            tau, cos_theta_i, sigma_h_v, l_c_v, self.WAVELENGTH
        )

        return eta * tau_eff * A

    def eval_f_cos_components_physics(self, wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau,
                                       thickness):
        """Evaluate BSDF × |cos(θ_i)| with per-lobe components from direct physics params."""
        return self._eval_f_cos_components_core(
            wo, wi, n, eps_real, eps_imag, sigma_h, l_c, tau, thickness
        )

    # =========================================================================
    # SAMPLING
    # =========================================================================

    def pdf_wi(self, wo, wi, n, roughness):
        """PDF for sampling (cosine-weighted hemisphere)."""
        cos_theta_i = dr.dot(wi, n)
        valid = cos_theta_i > mi.Float(1e-6)
        return dr.select(valid, cos_theta_i / mi.Float(np.pi), mi.Float(0.0))

    def sample_f(self, wo, n, albedo, roughness, metallic, samples):
        """Sample incident direction using cosine-weighted hemisphere."""
        wi_local, pdf = sample_cosine_hemisphere_concentric(samples)
        wi_world = to_global(wi_local, n)
        f_cos = self.eval_f_cos(wo, wi_world, n, albedo, roughness, metallic)
        is_delta = mi.Bool(False)
        return wi_world, f_cos, pdf, is_delta


# =============================================================================
# TIER 2: JONES-AWARE MIXTURE SAMPLING WEIGHTS
# =============================================================================


def compute_lobe_weights_jones_drjit(
    bsdf_jones,
    wo, wi_dummy, n,
    eps_real, eps_imag, sigma_h, l_c, tau_base, thickness,
    wavelength=WAVELENGTH_77GHZ,
):
    """
    Compute per-lobe blend weights using Jones polarization-aware Fresnel.

    Like compute_lobe_weights_drjit but uses R_jones (polarization-weighted
    Fresnel power) as the energy gate instead of scalar A.

    Args:
        bsdf_jones: BSDFmmWaveJones instance (provides TX/RX polarization)
        wo: mi.Vector3f [N] outgoing direction
        wi_dummy: mi.Vector3f [N] dummy incoming direction (used for Jones basis)
            Typically -prev_dir for computing reflection plane
        n: mi.Vector3f [N] surface normal
        eps_real..thickness: mi.Float [N] material parameters
        wavelength: float

    Returns:
        (p_KA, p_SPM, p_dir, p_broad, alpha, kappa_spm, kappa_dir):
        First 4 are selection probabilities summing to 1.0.
        Last 3 are the distribution parameters needed for sampling.
    """
    cos_theta_o = dr.maximum(dr.dot(wo, n), mi.Float(1e-6))
    n_ior, kappa_ior = permittivity_to_ior(eps_real, eps_imag)

    # Use Jones Fresnel power as energy gate
    R_jones = bsdf_jones._compute_jones_fresnel_power(
        wi_dummy, wo, n, n_ior, kappa_ior, cos_theta_o,
        eps_real=eps_real, eps_imag=eps_imag, thickness=thickness
    )

    return compute_lobe_weights_drjit(
        wo, n,
        eps_real, eps_imag, sigma_h, l_c, tau_base, thickness,
        wavelength=wavelength,
        enable_incoherent=bsdf_jones.enable_incoherent,
        enable_slab_fresnel=bsdf_jones.enable_slab_fresnel,
        energy_gate=R_jones,
    )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_jones_bsdf_vertical(use_fresnel_phase: bool = False):
    """Create Jones BSDF with vertical (z-axis) TX/RX polarization."""
    return BSDFmmWaveJones(
        tx_polarization=mi.Vector3f(0, 0, 1),
        rx_polarization=mi.Vector3f(0, 0, 1),
        use_fresnel_phase=use_fresnel_phase,
    )


def create_jones_bsdf_horizontal(use_fresnel_phase: bool = False):
    """Create Jones BSDF with horizontal (x-axis) TX/RX polarization."""
    return BSDFmmWaveJones(
        tx_polarization=mi.Vector3f(1, 0, 0),
        rx_polarization=mi.Vector3f(1, 0, 0),
        use_fresnel_phase=use_fresnel_phase,
    )


def create_jones_bsdf_cross_polar(use_fresnel_phase: bool = False):
    """Create Jones BSDF with cross-polarization (TX=V, RX=H)."""
    return BSDFmmWaveJones(
        tx_polarization=mi.Vector3f(0, 0, 1),  # Vertical TX
        rx_polarization=mi.Vector3f(1, 0, 0),  # Horizontal RX
        use_fresnel_phase=use_fresnel_phase,
    )


__all__ = [
    'BSDFmmWaveJones',
    'create_jones_bsdf_vertical',
    'create_jones_bsdf_horizontal',
    'create_jones_bsdf_cross_polar',
    'fresnel_complex_amplitude',
    'compute_sp_basis_jones',
    'project_polarization_to_sp',
    'apply_jones_reflection',
    'compute_received_field_complex',
    'compute_received_power',
    # Tier 2 sampling (re-exported from scalar + Jones-specific)
    'compute_lobe_weights_jones_drjit',
    'sample_ggx_vndf_drjit',
    'eval_ggx_pdf_drjit',
    'sample_vmf_drjit',
    'eval_vmf_pdf_drjit',
    'compute_lobe_weights_drjit',
    'compute_mixture_pdf_drjit',
]
