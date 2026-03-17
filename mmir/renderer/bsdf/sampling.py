"""
Tier-2 BSDF importance sampling primitives for mixture importance sampling.

Per-lobe sampling functions (GGX VNDF for KA, vMF for SPM/directive, cosine for broad)
that enable drawing samples proportional to BSDF lobe shape rather than uniform/cosine.

Each sampling function returns (sampled_dir, pdf) in DrJit vectorized form.
"""

import numpy as np
import drjit as dr
import mitsuba as mi

from .mmwave_scalar import (
    WAVELENGTH_77GHZ,
    enforce_spm_validity,
    compute_slab_energy_gate,
    compute_coherent_incoherent_blend,
    permittivity_to_ior,
    compute_energy_gate,
    compute_validity_aware_blend,
)


# =============================================================================
# TANGENT FRAME CONSTRUCTION
# =============================================================================

def _build_tangent_frame_drjit(n):
    """
    Build orthonormal tangent frame from normal vector (Frisvad method).

    Returns:
        t1, t2: mi.Vector3f tangent vectors such that (t1, t2, n) is ONB
    """
    abs_nz = dr.abs(n.z)

    # Standard Frisvad method
    denom = mi.Float(1.0) + n.z
    denom_safe = dr.select(denom > 1e-6, denom, mi.Float(1e-6))
    a = mi.Float(-1.0) / denom_safe
    b = n.x * n.y * a
    t1_x = mi.Float(1.0) + n.x * n.x * a
    t1_y = b
    t1_z = -n.x
    t2_x = b
    t2_y = mi.Float(1.0) + n.y * n.y * a
    t2_z = -n.y

    # Fallback for n.z close to -1
    use_fallback = abs_nz > 0.9999
    t1_x = dr.select(use_fallback, mi.Float(0.0), t1_x)
    t1_y = dr.select(use_fallback, mi.Float(-1.0), t1_y)
    t1_z = dr.select(use_fallback, mi.Float(0.0), t1_z)
    t2_x = dr.select(use_fallback, mi.Float(-1.0), t2_x)
    t2_y = dr.select(use_fallback, mi.Float(0.0), t2_y)
    t2_z = dr.select(use_fallback, mi.Float(0.0), t2_z)

    t1 = mi.Vector3f(t1_x, t1_y, t1_z)
    t2 = mi.Vector3f(t2_x, t2_y, t2_z)
    return t1, t2


def _world_to_local(v, t1, t2, n):
    """Transform vector from world to local (t1, t2, n) frame."""
    return mi.Vector3f(dr.dot(v, t1), dr.dot(v, t2), dr.dot(v, n))


def _local_to_world(v, t1, t2, n):
    """Transform vector from local (t1, t2, n) frame to world."""
    return mi.Vector3f(
        v.x * t1.x + v.y * t2.x + v.z * n.x,
        v.x * t1.y + v.y * t2.y + v.z * n.y,
        v.x * t1.z + v.y * t2.z + v.z * n.z,
    )


# =============================================================================
# GGX VISIBLE NORMAL DISTRIBUTION SAMPLING (Heitz 2018)
# =============================================================================

def sample_ggx_vndf_drjit(wo, n, alpha, n_paths, seed):
    """
    GGX Visible Normal Distribution Function sampling (Heitz 2018).

    Samples microfacet normal h visible from wo, then reflects to get wi.

    Args:
        wo: mi.Vector3f [n_paths] outgoing direction (world frame)
        n: mi.Vector3f [n_paths] surface normal (world frame)
        alpha: mi.Float [n_paths] GGX roughness parameter
        n_paths: int number of paths
        seed: int RNG seed

    Returns:
        wi: mi.Vector3f [n_paths] sampled incident direction (world frame)
        pdf: mi.Float [n_paths] PDF in solid angle of wi
    """
    # Build tangent frame
    t1, t2 = _build_tangent_frame_drjit(n)

    # Transform wo to local frame
    wo_local = _world_to_local(wo, t1, t2, n)

    # Ensure wo is in upper hemisphere (for double-sided)
    wo_local_z_safe = dr.maximum(wo_local.z, mi.Float(1e-6))

    # Step 1: Stretch wo by (1/alpha, 1/alpha, 1) and renormalize
    wo_s = mi.Vector3f(alpha * wo_local.x, alpha * wo_local.y, wo_local_z_safe)
    wo_s_len = dr.maximum(dr.norm(wo_s), mi.Float(1e-10))
    wo_s = mi.Vector3f(wo_s.x / wo_s_len, wo_s.y / wo_s_len, wo_s.z / wo_s_len)

    # Step 2: Build orthonormal basis around stretched wo
    # T1_s perpendicular to wo_s in the tangent plane
    len_sq_xy = wo_s.x * wo_s.x + wo_s.y * wo_s.y
    has_xy = len_sq_xy > mi.Float(1e-8)
    inv_len_xy = mi.Float(1.0) / dr.maximum(dr.sqrt(dr.maximum(len_sq_xy, mi.Float(1e-20))), mi.Float(1e-10))

    T1_x = dr.select(has_xy, -wo_s.y * inv_len_xy, mi.Float(1.0))
    T1_y = dr.select(has_xy, wo_s.x * inv_len_xy, mi.Float(0.0))
    T1_z = mi.Float(0.0)
    T1 = mi.Vector3f(T1_x, T1_y, T1_z)

    # T2 = wo_s x T1
    T2 = dr.cross(wo_s, T1)

    # Step 3: Sample visible hemisphere via uniform disk → hemisphere projection
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(seed, n_paths)
    u1 = sampler.next_1d()
    u2 = sampler.next_1d()

    # Concentric disk sampling
    a_disk = mi.Float(2.0) * u1 - mi.Float(1.0)
    b_disk = mi.Float(2.0) * u2 - mi.Float(1.0)

    # Handle center
    at_center = (dr.abs(a_disk) < 1e-8) & (dr.abs(b_disk) < 1e-8)
    r_disk = mi.Float(0.0)
    phi_disk = mi.Float(0.0)

    use_a = dr.abs(a_disk) > dr.abs(b_disk)
    r_disk = dr.select(use_a, a_disk, b_disk)
    ratio = dr.select(use_a,
        b_disk / dr.select(dr.abs(a_disk) > 1e-8, a_disk, mi.Float(1.0)),
        a_disk / dr.select(dr.abs(b_disk) > 1e-8, b_disk, mi.Float(1.0)))
    phi_disk = dr.select(use_a,
        mi.Float(np.pi / 4.0) * ratio,
        mi.Float(np.pi / 2.0) - mi.Float(np.pi / 4.0) * ratio)
    r_disk = dr.select(at_center, mi.Float(0.0), r_disk)
    phi_disk = dr.select(at_center, mi.Float(0.0), phi_disk)

    disk_x = r_disk * dr.cos(phi_disk)
    disk_y = r_disk * dr.sin(phi_disk)

    # Project to hemisphere: lerp between hemisphere and disk based on wo_s.z
    s = mi.Float(0.5) * (mi.Float(1.0) + wo_s.z)
    disk_y_proj = dr.select(s > 1e-6,
        (mi.Float(1.0) - s) * dr.sqrt(dr.maximum(mi.Float(1.0) - disk_x * disk_x, mi.Float(1e-20))) + s * disk_y,
        dr.sqrt(dr.maximum(mi.Float(1.0) - disk_x * disk_x, mi.Float(1e-20))))

    # Hemisphere sample in stretched space
    disk_z = dr.sqrt(dr.maximum(
        mi.Float(1.0) - disk_x * disk_x - disk_y_proj * disk_y_proj,
        mi.Float(1e-20)))

    # Normal in stretched local frame
    nh_s = mi.Vector3f(
        disk_x * T1.x + disk_y_proj * T2.x + disk_z * wo_s.x,
        disk_x * T1.y + disk_y_proj * T2.y + disk_z * wo_s.y,
        disk_x * T1.z + disk_y_proj * T2.z + disk_z * wo_s.z,
    )

    # Step 4: Unstretch to get microfacet normal in local frame
    h_local = mi.Vector3f(alpha * nh_s.x, alpha * nh_s.y, dr.maximum(nh_s.z, mi.Float(1e-10)))
    h_len = dr.maximum(dr.norm(h_local), mi.Float(1e-10))
    h_local = mi.Vector3f(h_local.x / h_len, h_local.y / h_len, h_local.z / h_len)

    # Step 5: Transform h back to world frame
    h_world = _local_to_world(h_local, t1, t2, n)

    # Step 6: Reflect wo about h to get wi
    wo_dot_h = dr.dot(wo, h_world)
    wi = mi.Vector3f(
        mi.Float(2.0) * wo_dot_h * h_world.x - wo.x,
        mi.Float(2.0) * wo_dot_h * h_world.y - wo.y,
        mi.Float(2.0) * wo_dot_h * h_world.z - wo.z,
    )

    # Step 7: Compute PDF
    # VNDF PDF: pdf_wi = D_wi(h) / (4 × |wo·h|)
    # where D_wi(h) = G1(wo) × max(wo·h, 0) × D(h) / (n·wo)

    h_dot_n = dr.dot(h_world, n)
    a2 = alpha * alpha

    # GGX NDF D(h)
    d_denom = (h_dot_n * h_dot_n) * (a2 - mi.Float(1.0)) + mi.Float(1.0)
    D = a2 / (d_denom * d_denom * mi.Float(np.pi))

    # Smith G1 for wo
    cos_o = dr.maximum(dr.dot(wo, n), mi.Float(1e-6))
    cos_o_sq = cos_o * cos_o
    tan_o_sq = dr.maximum(mi.Float(1.0) - cos_o_sq, mi.Float(0.0)) / dr.maximum(cos_o_sq, mi.Float(1e-10))
    lambda_o = mi.Float(0.5) * (mi.Float(-1.0) + dr.sqrt(mi.Float(1.0) + a2 * tan_o_sq))
    G1_o = mi.Float(1.0) / (mi.Float(1.0) + lambda_o)

    # VNDF PDF in solid angle of wi
    wo_dot_h_safe = dr.maximum(dr.abs(wo_dot_h), mi.Float(1e-6))
    pdf = D * G1_o * dr.maximum(wo_dot_h, mi.Float(0.0)) / (cos_o * mi.Float(4.0) * wo_dot_h_safe)

    # Clamp degenerate cases
    wi_dot_n = dr.dot(wi, n)
    valid = (wi_dot_n > mi.Float(0.0)) & (h_dot_n > mi.Float(0.0)) & (wo_dot_h > mi.Float(0.0))
    pdf = dr.select(valid, dr.maximum(pdf, mi.Float(1e-10)), mi.Float(1e-10))

    return wi, pdf


def eval_ggx_pdf_drjit(wi, wo, n, alpha):
    """
    Evaluate GGX VNDF sampling PDF at direction wi.

    Args:
        wi: mi.Vector3f [N] incident direction
        wo: mi.Vector3f [N] outgoing direction
        n: mi.Vector3f [N] surface normal
        alpha: mi.Float [N] GGX roughness

    Returns:
        pdf: mi.Float [N]
    """
    # Half vector
    h_unnorm = wo + wi
    h_len = dr.maximum(dr.norm(h_unnorm), mi.Float(1e-10))
    h = mi.Vector3f(h_unnorm.x / h_len, h_unnorm.y / h_len, h_unnorm.z / h_len)

    h_dot_n = dr.dot(h, n)
    wo_dot_h = dr.dot(wo, h)
    a2 = alpha * alpha

    # GGX NDF D(h)
    d_denom = (h_dot_n * h_dot_n) * (a2 - mi.Float(1.0)) + mi.Float(1.0)
    D = a2 / (d_denom * d_denom * mi.Float(np.pi))

    # Smith G1(wo)
    cos_o = dr.maximum(dr.dot(wo, n), mi.Float(1e-6))
    cos_o_sq = cos_o * cos_o
    tan_o_sq = dr.maximum(mi.Float(1.0) - cos_o_sq, mi.Float(0.0)) / dr.maximum(cos_o_sq, mi.Float(1e-10))
    lambda_o = mi.Float(0.5) * (mi.Float(-1.0) + dr.sqrt(mi.Float(1.0) + a2 * tan_o_sq))
    G1_o = mi.Float(1.0) / (mi.Float(1.0) + lambda_o)

    wo_dot_h_safe = dr.maximum(dr.abs(wo_dot_h), mi.Float(1e-6))
    pdf = D * G1_o * dr.maximum(wo_dot_h, mi.Float(0.0)) / (cos_o * mi.Float(4.0) * wo_dot_h_safe)

    wi_dot_n = dr.dot(wi, n)
    valid = (wi_dot_n > mi.Float(0.0)) & (h_dot_n > mi.Float(0.0)) & (wo_dot_h > mi.Float(0.0))
    return dr.select(valid, dr.maximum(pdf, mi.Float(1e-10)), mi.Float(1e-10))


# =============================================================================
# VON MISES-FISHER SAMPLING
# =============================================================================

def sample_vmf_drjit(wo, n, kappa, n_paths, seed):
    """
    von Mises-Fisher importance sampling centered on specular reflection.

    Samples directions from vMF(kappa, mu) where mu = reflect(wo, n).

    Args:
        wo: mi.Vector3f [n_paths] outgoing direction (world frame)
        n: mi.Vector3f [n_paths] surface normal (world frame)
        kappa: mi.Float [n_paths] concentration parameter
        n_paths: int number of paths
        seed: int RNG seed

    Returns:
        wi: mi.Vector3f [n_paths] sampled incident direction (world frame)
        pdf: mi.Float [n_paths] vMF PDF value
    """
    # Specular reflection direction (center of vMF)
    wo_dot_n = dr.dot(wo, n)
    mu = mi.Vector3f(
        mi.Float(2.0) * wo_dot_n * n.x - wo.x,
        mi.Float(2.0) * wo_dot_n * n.y - wo.y,
        mi.Float(2.0) * wo_dot_n * n.z - wo.z,
    )

    # Build tangent frame around mu
    t1, t2 = _build_tangent_frame_drjit(mu)

    # Sample vMF: cos_theta = 1 + (1/kappa) * ln(u1 + (1-u1)*exp(-2*kappa))
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(seed, n_paths)
    u1 = sampler.next_1d()
    u2 = sampler.next_1d()

    # Numerically stable sampling
    kappa_safe = dr.maximum(kappa, mi.Float(0.01))
    exp_m2k = dr.exp(mi.Float(-2.0) * kappa_safe)
    inner = u1 + (mi.Float(1.0) - u1) * exp_m2k
    inner = dr.maximum(inner, mi.Float(1e-30))  # Prevent log(0)
    cos_theta = mi.Float(1.0) + dr.log(inner) / kappa_safe
    cos_theta = dr.clamp(cos_theta, mi.Float(-1.0), mi.Float(1.0))

    sin_theta = dr.sqrt(dr.maximum(mi.Float(1.0) - cos_theta * cos_theta, mi.Float(1e-20)))
    phi = mi.Float(2.0 * np.pi) * u2

    # Direction in local frame (around mu)
    local_x = sin_theta * dr.cos(phi)
    local_y = sin_theta * dr.sin(phi)
    local_z = cos_theta

    # Transform to world
    wi = _local_to_world(mi.Vector3f(local_x, local_y, local_z), t1, t2, mu)

    # Normalize for safety
    wi_len = dr.maximum(dr.norm(wi), mi.Float(1e-10))
    wi = mi.Vector3f(wi.x / wi_len, wi.y / wi_len, wi.z / wi_len)

    # Clamp to upper hemisphere: if wi·n < 0, reflect through n
    wi_dot_n = dr.dot(wi, n)
    below = wi_dot_n < mi.Float(0.0)
    # Reflect: wi' = wi - 2*(wi·n)*n
    wi_reflected_x = wi.x - mi.Float(2.0) * wi_dot_n * n.x
    wi_reflected_y = wi.y - mi.Float(2.0) * wi_dot_n * n.y
    wi_reflected_z = wi.z - mi.Float(2.0) * wi_dot_n * n.z
    wi = dr.select(below,
        mi.Vector3f(wi_reflected_x, wi_reflected_y, wi_reflected_z),
        wi)

    # Compute PDF
    cos_dev = dr.dot(wi, mu)
    pdf = eval_vmf_pdf_drjit(cos_dev, kappa_safe)

    return wi, pdf


def eval_vmf_pdf_drjit(cos_dev_or_wi, kappa_or_wo=None, n=None, kappa=None):
    """
    Evaluate von Mises-Fisher PDF.

    Two calling conventions:
    1. eval_vmf_pdf_drjit(cos_dev, kappa) — direct cos_dev and kappa
    2. eval_vmf_pdf_drjit(wi, wo, n, kappa) — compute cos_dev from directions

    Returns:
        pdf: mi.Float [N]
    """
    if n is not None and kappa is not None:
        # Convention 2: wi, wo, n, kappa
        wi = cos_dev_or_wi
        wo = kappa_or_wo
        wo_dot_n = dr.dot(wo, n)
        mu = mi.Vector3f(
            mi.Float(2.0) * wo_dot_n * n.x - wo.x,
            mi.Float(2.0) * wo_dot_n * n.y - wo.y,
            mi.Float(2.0) * wo_dot_n * n.z - wo.z,
        )
        cos_dev = dr.dot(wi, mu)
        kappa_val = kappa
    else:
        # Convention 1: cos_dev, kappa
        cos_dev = cos_dev_or_wi
        kappa_val = kappa_or_wo

    kappa_safe = dr.maximum(kappa_val, mi.Float(0.01))

    # Correct vMF PDF: kappa / (4*pi*sinh(kappa)) * exp(kappa * cos_dev)
    # Numerically stable form: kappa / (2*pi * (1 - exp(-2*kappa))) * exp(kappa*(cos_dev - 1))
    kappa_clamped = dr.minimum(kappa_safe, mi.Float(50.0))
    one_minus_exp2k = mi.Float(1.0) - dr.exp(mi.Float(-2.0) * kappa_clamped)
    one_minus_exp2k = dr.maximum(one_minus_exp2k, mi.Float(1e-10))
    norm = kappa_safe / (mi.Float(2.0 * np.pi) * one_minus_exp2k)

    pdf = norm * dr.exp(kappa_safe * (cos_dev - mi.Float(1.0)))
    return dr.maximum(pdf, mi.Float(1e-10))


# =============================================================================
# MIXTURE IMPORTANCE SAMPLING
# =============================================================================

def compute_lobe_weights_drjit(
    wo, n,
    eps_real, eps_imag, sigma_h, l_c, tau_base, thickness,
    wavelength=WAVELENGTH_77GHZ,
    enable_incoherent=True,
    enable_slab_fresnel=True,
    energy_gate=None,
):
    """
    Compute per-lobe blend weights matching _eval_f_core decomposition.

    Returns normalized probabilities for mixture sampling selection.

    Args:
        wo: mi.Vector3f [N] outgoing direction
        n: mi.Vector3f [N] surface normal
        eps_real..thickness: mi.Float [N] material parameters
        wavelength: float
        enable_incoherent: bool
        enable_slab_fresnel: bool
        energy_gate: Optional mi.Float [N] pre-computed energy gate (e.g. R_jones).
            If None, computes scalar Fresnel energy gate internally.

    Returns:
        (p_KA, p_SPM, p_dir, p_broad, alpha, kappa_spm, kappa_dir):
        First 4 are selection probabilities summing to 1.0.
        Last 3 are the distribution parameters needed for sampling.
    """
    sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, wavelength)
    cos_theta_o = dr.maximum(dr.dot(wo, n), mi.Float(1e-6))

    # Energy gate: use pre-computed (e.g. Jones R_jones) or compute scalar Fresnel
    if energy_gate is not None:
        A = energy_gate
    elif enable_slab_fresnel:
        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)
        A = compute_slab_energy_gate(
            eps_real, eps_imag, cos_theta_o, thickness,
            frac_s, frac_p, wavelength, sigma_h=sigma_h_v)
    else:
        frac_s, frac_p = mi.Float(0.5), mi.Float(0.5)
        n_ior, kappa_ior = permittivity_to_ior(eps_real, eps_imag)
        A = compute_energy_gate(n_ior, kappa_ior, cos_theta_o, frac_s, frac_p)

    # Coherent/incoherent blend
    k = mi.Float(2.0 * np.pi / wavelength)
    if enable_incoherent:
        eta, gamma = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, wavelength, cos_theta_o)
    else:
        eta = mi.Float(1.0)
        gamma = mi.Float(1.0)

    # KA/SPM blend
    tau_eff = compute_validity_aware_blend(
        tau_base, cos_theta_o, sigma_h_v, l_c_v, wavelength)

    # Raw lobe weights
    w_KA = A * eta * tau_eff
    w_SPM = A * eta * (mi.Float(1.0) - tau_eff)
    if enable_incoherent:
        w_dir = A * (mi.Float(1.0) - eta) * gamma
        w_broad = A * (mi.Float(1.0) - eta) * (mi.Float(1.0) - gamma)
    else:
        w_dir = mi.Float(0.0)
        w_broad = mi.Float(0.0)

    # Normalize to probabilities
    w_total = dr.maximum(w_KA + w_SPM + w_dir + w_broad, mi.Float(1e-8))
    p_KA = w_KA / w_total
    p_SPM = w_SPM / w_total
    p_dir = w_dir / w_total
    p_broad = w_broad / w_total

    # Compute distribution parameters
    # GGX alpha for KA
    alpha_raw = mi.Float(4.0 * np.pi) * sigma_h_v / mi.Float(wavelength)
    alpha = dr.clamp(dr.sqrt(alpha_raw), mi.Float(0.05), mi.Float(0.95))

    # vMF kappa for SPM
    l_c_wavelengths = l_c_v / mi.Float(wavelength)
    roughness_slope = sigma_h_v / dr.maximum(l_c_v, mi.Float(1e-8))
    kappa_spm = dr.clamp(
        dr.sqrt(l_c_wavelengths) / (mi.Float(1.0) + mi.Float(3.0) * roughness_slope),
        mi.Float(0.5), mi.Float(10.0))

    # vMF kappa for directive (broader)
    kappa_spm_base = dr.sqrt(l_c_wavelengths)
    broadening = mi.Float(1.0) + mi.Float(5.0) * roughness_slope
    kappa_dir = dr.clamp(kappa_spm_base / broadening, mi.Float(0.3), mi.Float(5.0))

    return p_KA, p_SPM, p_dir, p_broad, alpha, kappa_spm, kappa_dir


def compute_mixture_pdf_drjit(
    wi, wo, n,
    p_KA, p_SPM, p_dir, p_broad,
    alpha, kappa_spm, kappa_dir,
):
    """
    Evaluate the mixture PDF at sampled direction wi.

    Uses the same lobe decomposition as compute_lobe_weights_drjit:
      pdf = p_KA * pdf_ggx + p_SPM * pdf_vmf_spm + p_dir * pdf_vmf_dir + p_broad * pdf_cos

    Args:
        wi: mi.Vector3f [N] incident direction
        wo: mi.Vector3f [N] outgoing direction
        n: mi.Vector3f [N] surface normal
        p_KA..p_broad: mi.Float [N] lobe selection probabilities
        alpha: mi.Float [N] GGX roughness
        kappa_spm: mi.Float [N] vMF concentration for SPM
        kappa_dir: mi.Float [N] vMF concentration for directive

    Returns:
        pdf_mix: mi.Float [N] mixture PDF value
    """
    # GGX VNDF PDF for KA
    pdf_ka = eval_ggx_pdf_drjit(wi, wo, n, alpha)

    # vMF PDF for SPM
    pdf_spm = eval_vmf_pdf_drjit(wi, wo, n, kappa=kappa_spm)

    # vMF PDF for directive
    pdf_dir = eval_vmf_pdf_drjit(wi, wo, n, kappa_dir)

    # Cosine PDF for broad
    cos_theta_i = dr.maximum(dr.dot(wi, n), mi.Float(0.0))
    pdf_cos = cos_theta_i / mi.Float(np.pi)
    pdf_cos = dr.maximum(pdf_cos, mi.Float(1e-10))

    # Mixture
    pdf_mix = p_KA * pdf_ka + p_SPM * pdf_spm + p_dir * pdf_dir + p_broad * pdf_cos
    return dr.maximum(pdf_mix, mi.Float(1e-10))
