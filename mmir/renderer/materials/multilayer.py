"""
Multi-layer ABCD transfer matrix method for electromagnetic wave propagation.

Implements ITU-R P.2040-4 Attachment 1 (Eqs 60-62) for computing reflection
and transmission coefficients through N-layer wall structures.

The ABCD (transfer) matrix formulation chains per-layer matrices to model
arbitrary layered dielectrics with loss. Each layer is characterized by its
complex permittivity, conductivity, and thickness.

Usage:
    from .multilayer import abcd_multilayer_fresnel
    from .itu_materials import get_wall_preset

    layers = get_wall_preset('double_glazing', 77e9)
    R_TE_r, R_TE_i, R_TM_r, R_TM_i, T_TE_r, T_TE_i, T_TM_r, T_TM_i = \
        abcd_multilayer_fresnel(layers, cos_theta_i, 77e9)
"""

import numpy as np
import drjit as dr
import mitsuba as mi

from ..bsdf.mmwave_scalar import (
    _cpx_sqrt, _cpx_mul, _cpx_div, _cpx_exp, _cpx_abs_sq, _cpx_sq,
    WAVELENGTH_77GHZ,
)


# =============================================================================
# ABCD MATRIX UTILITIES (2x2 complex matrices as 8-tuples)
# =============================================================================
#
# A 2x2 complex matrix [[A, B], [C, D]] is stored as:
#   (A_r, A_i, B_r, B_i, C_r, C_i, D_r, D_i)

def _mat2x2_identity():
    """Return 2x2 identity matrix."""
    return (mi.Float(1.0), mi.Float(0.0),   # A = 1
            mi.Float(0.0), mi.Float(0.0),   # B = 0
            mi.Float(0.0), mi.Float(0.0),   # C = 0
            mi.Float(1.0), mi.Float(0.0))   # D = 1


def _mat2x2_mul(m1, m2):
    """
    Multiply two 2x2 complex matrices.

    M1 = [[a, b], [c, d]]
    M2 = [[e, f], [g, h]]
    Result = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    """
    a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i = m1
    e_r, e_i, f_r, f_i, g_r, g_i, h_r, h_i = m2

    # Row 0, Col 0: A*E + B*G
    ae_r, ae_i = _cpx_mul(a_r, a_i, e_r, e_i)
    bg_r, bg_i = _cpx_mul(b_r, b_i, g_r, g_i)
    r00_r = ae_r + bg_r
    r00_i = ae_i + bg_i

    # Row 0, Col 1: A*F + B*H
    af_r, af_i = _cpx_mul(a_r, a_i, f_r, f_i)
    bh_r, bh_i = _cpx_mul(b_r, b_i, h_r, h_i)
    r01_r = af_r + bh_r
    r01_i = af_i + bh_i

    # Row 1, Col 0: C*E + D*G
    ce_r, ce_i = _cpx_mul(c_r, c_i, e_r, e_i)
    dg_r, dg_i = _cpx_mul(d_r, d_i, g_r, g_i)
    r10_r = ce_r + dg_r
    r10_i = ce_i + dg_i

    # Row 1, Col 1: C*F + D*H
    cf_r, cf_i = _cpx_mul(c_r, c_i, f_r, f_i)
    dh_r, dh_i = _cpx_mul(d_r, d_i, h_r, h_i)
    r11_r = cf_r + dh_r
    r11_i = cf_i + dh_i

    return (r00_r, r00_i, r01_r, r01_i,
            r10_r, r10_i, r11_r, r11_i)


def _build_layer_matrix_te(kz_r, kz_i, Z_r, Z_i, d):
    """
    Build ABCD transfer matrix for a single layer (TE polarization).

    M = [[cos(kz*d),    j*Z*sin(kz*d)],
         [j*sin(kz*d)/Z, cos(kz*d)    ]]

    where kz is the complex propagation constant in the layer
    and Z is the characteristic impedance.
    """
    # kz * d (complex)
    kd_r = kz_r * d
    kd_i = kz_i * d

    # cos(kz*d) and sin(kz*d) for complex argument
    # cos(a+jb) = cos(a)cosh(b) - j*sin(a)sinh(b)
    # sin(a+jb) = sin(a)cosh(b) + j*cos(a)sinh(b)
    cosh_b = dr.cosh(kd_i)
    sinh_b = dr.sinh(kd_i)
    cos_a = dr.cos(kd_r)
    sin_a = dr.sin(kd_r)

    cos_kd_r = cos_a * cosh_b
    cos_kd_i = -sin_a * sinh_b
    sin_kd_r = sin_a * cosh_b
    sin_kd_i = cos_a * sinh_b

    # j * Z * sin(kz*d): multiply (0+j) * Z * sin
    # j * (Z_r + jZ_i) = -Z_i + jZ_r
    jZ_r = -Z_i
    jZ_i = Z_r
    B_r, B_i = _cpx_mul(jZ_r, jZ_i, sin_kd_r, sin_kd_i)

    # j * sin(kz*d) / Z: multiply (0+j) * sin / Z
    j_sin_r = -sin_kd_i
    j_sin_i = sin_kd_r
    C_r, C_i = _cpx_div(j_sin_r, j_sin_i, Z_r, Z_i)

    return (cos_kd_r, cos_kd_i,  # A
            B_r, B_i,             # B
            C_r, C_i,             # C
            cos_kd_r, cos_kd_i)   # D


# =============================================================================
# MAIN FUNCTION
# =============================================================================

EPSILON_0 = 8.854187817e-12   # Vacuum permittivity (F/m)
MU_0 = 4.0e-7 * np.pi        # Vacuum permeability (H/m)
C0 = 299792458.0              # Speed of light (m/s)


def abcd_multilayer_fresnel(
    layers: list,
    cos_theta_i: 'mi.Float',
    freq_hz: float,
) -> tuple:
    """
    N-layer ABCD transfer matrix method for reflection and transmission.

    Implements ITU-R P.2040-4 Attachment 1 (Eqs 60-62).

    For each polarization (TE, TM), chains per-layer transfer matrices and
    extracts the total reflection and transmission coefficients.

    Args:
        layers: List of (eps_r, sigma, thickness_d) tuples, one per layer.
            Each tuple: (real permittivity, conductivity S/m, thickness m).
            Ordered from incident side to transmitted side.
        cos_theta_i: Cosine of incidence angle (DrJit array)
        freq_hz: Frequency in Hz

    Returns:
        (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
         T_TE_r, T_TE_i, T_TM_r, T_TM_i): Real/imag parts of complex coefficients
    """
    omega = 2.0 * np.pi * freq_hz
    k0 = omega / C0
    sin2_theta = mi.Float(1.0) - cos_theta_i * cos_theta_i

    # Process both polarizations
    results = {}
    for pol in ['TE', 'TM']:
        # Initialize with identity matrix
        M = _mat2x2_identity()

        for eps_r, sigma, d in layers:
            # Complex permittivity: eta = eps_r - j * sigma/(omega*eps_0)
            eta_r = mi.Float(eps_r)
            eta_i = mi.Float(-sigma / (omega * EPSILON_0))

            # kz in this layer: k0 * sqrt(eta - sin^2(theta))
            arg_r = eta_r - sin2_theta
            arg_i = eta_i
            kz_r, kz_i = _cpx_sqrt(arg_r, arg_i)
            kz_r = mi.Float(k0) * kz_r
            kz_i = mi.Float(k0) * kz_i

            # Characteristic impedance depends on polarization
            if pol == 'TE':
                # Z_TE = omega * mu_0 / kz
                num_r = mi.Float(omega * MU_0)
                num_i = mi.Float(0.0)
                Z_r, Z_i = _cpx_div(num_r, num_i, kz_r, kz_i)
            else:  # TM
                # Z_TM = kz / (omega * eps_0 * eta)
                # Numerator: kz
                # Denominator: omega * eps_0 * eta
                den_r = mi.Float(omega * EPSILON_0) * eta_r
                den_i = mi.Float(omega * EPSILON_0) * eta_i
                Z_r, Z_i = _cpx_div(kz_r, kz_i, den_r, den_i)

            # Build layer transfer matrix
            M_layer = _build_layer_matrix_te(kz_r, kz_i, Z_r, Z_i, mi.Float(d))

            # Chain multiply: M = M @ M_layer
            M = _mat2x2_mul(M, M_layer)

        # Extract R, T from total ABCD matrix
        A_r, A_i, B_r, B_i, C_r, C_i, D_r, D_i = M

        # Free-space impedance
        kz0 = mi.Float(k0) * cos_theta_i
        if pol == 'TE':
            Z0_r = mi.Float(omega * MU_0) / dr.maximum(kz0, mi.Float(1e-10))
            Z0_i = mi.Float(0.0)
        else:  # TM
            Z0_r = kz0 / mi.Float(omega * EPSILON_0)
            Z0_i = mi.Float(0.0)

        # R = (A + B/Z0 - C*Z0 - D) / (A + B/Z0 + C*Z0 + D)
        BdZ_r, BdZ_i = _cpx_div(B_r, B_i, Z0_r, Z0_i)
        CZ_r, CZ_i = _cpx_mul(C_r, C_i, Z0_r, Z0_i)

        num_r = (A_r + BdZ_r - CZ_r - D_r)
        num_i = (A_i + BdZ_i - CZ_i - D_i)
        den_r = (A_r + BdZ_r + CZ_r + D_r)
        den_i = (A_i + BdZ_i + CZ_i + D_i)

        R_r, R_i = _cpx_div(num_r, num_i, den_r, den_i)

        # T = 2 / (A + B/Z0 + C*Z0 + D)
        T_r, T_i = _cpx_div(mi.Float(2.0), mi.Float(0.0), den_r, den_i)

        results[pol] = (R_r, R_i, T_r, T_i)

    R_TE_r, R_TE_i, T_TE_r, T_TE_i = results['TE']
    R_TM_r, R_TM_i, T_TM_r, T_TM_i = results['TM']

    return (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
            T_TE_r, T_TE_i, T_TM_r, T_TM_i)


def compute_multilayer_energy_gate(
    layers: list,
    cos_theta_i: 'mi.Float',
    freq_hz: float,
    frac_s: 'mi.Float',
    frac_p: 'mi.Float',
) -> 'mi.Float':
    """
    Compute energy gate A using multi-layer ABCD transfer matrix.

    A = frac_s * |R_TE|^2 + frac_p * |R_TM|^2

    Args:
        layers: List of (eps_r, sigma, thickness) tuples
        cos_theta_i: Cosine of incidence angle
        freq_hz: Frequency (Hz)
        frac_s: TE polarization fraction
        frac_p: TM polarization fraction

    Returns:
        A: Energy gate in [0, 1]
    """
    (R_TE_r, R_TE_i, R_TM_r, R_TM_i,
     T_TE_r, T_TE_i, T_TM_r, T_TM_i) = abcd_multilayer_fresnel(
        layers, cos_theta_i, freq_hz
    )

    R_TE_sq = _cpx_abs_sq(R_TE_r, R_TE_i)
    R_TM_sq = _cpx_abs_sq(R_TM_r, R_TM_i)
    A = frac_s * R_TE_sq + frac_p * R_TM_sq
    return dr.clamp(A, mi.Float(0.0), mi.Float(1.0))
