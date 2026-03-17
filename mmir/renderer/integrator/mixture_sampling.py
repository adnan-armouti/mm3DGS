"""
BSDF mixture importance sampling for multi-lobe evaluation.

Combines cosine hemisphere, GGX VNDF, von Mises-Fisher, and FSD sampling
into a single mixture sampler weighted by BSDF lobe probabilities.
"""

import numpy as np
import drjit as dr
import mitsuba as mi


def _sample_cosine_hemisphere_drjit(normal, n_paths, seed):
    """
    Sample cosine-weighted directions in the hemisphere defined by `normal`.

    Returns:
        sampled_dir: mi.Vector3f [n_paths] sampled direction (AD through normal)
        pdf: mi.Float [n_paths] cosine-weighted PDF = cos(theta)/pi
    """
    # Generate quasi-random samples using PCG
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(seed, n_paths)
    sample1 = sampler.next_1d()
    sample2 = sampler.next_1d()

    # Cosine-weighted hemisphere sampling in local frame
    cos_theta = dr.sqrt(dr.maximum(mi.Float(1.0) - sample1, mi.Float(1e-20)))
    sin_theta = dr.sqrt(dr.maximum(sample1, mi.Float(1e-20)))
    phi = mi.Float(2.0 * np.pi) * sample2
    local_x = sin_theta * dr.cos(phi)
    local_y = sin_theta * dr.sin(phi)
    local_z = cos_theta

    # Build tangent frame from normal (Frisvad method)
    # Handle degenerate normals
    n = normal
    abs_nz = dr.abs(n.z)

    # When n.z is not close to -1
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

    # Transform to world space
    world_x = local_x * t1_x + local_y * t2_x + local_z * n.x
    world_y = local_x * t1_y + local_y * t2_y + local_z * n.y
    world_z = local_x * t1_z + local_y * t2_z + local_z * n.z

    sampled_dir = mi.Vector3f(world_x, world_y, world_z)
    # Normalize for safety
    d_len = dr.maximum(dr.norm(sampled_dir), mi.Float(1e-10))
    sampled_dir = mi.Vector3f(sampled_dir.x / d_len, sampled_dir.y / d_len, sampled_dir.z / d_len)

    # PDF = cos(theta) / pi
    pdf = cos_theta / mi.Float(np.pi)

    return sampled_dir, pdf, sampler


def _sample_mixture_bsdf_drjit(
    wo, n,
    eps_real, eps_imag, sigma_h, l_c, tau_base, thickness,
    n_paths, seed,
    wavelength=3.9e-3,
    enable_incoherent=True,
    enable_slab_fresnel=True,
    energy_gate=None,
    # FSD (5th lobe) parameters — optional
    fsd_flat=None,           # FlatEdgeData or None
    fsd_aperture_idx=None,   # mi.UInt32 [n_paths] or None
    fsd_beta=None,           # mi.Float [n_paths] or None (energy borrowing fraction)
    fsd_k=None,              # float wavenumber
    fsd_tables=None,         # FsdSamplingTables or None
    fsd_sir_candidates=8,    # int SIR candidates
):
    """
    Combined BSDF mixture importance sampling (Tier 2).

    Samples continuation direction from a mixture of 4 (+1 optional) BSDF lobes:
    - KA: GGX visible normal sampling
    - SPM: von Mises-Fisher centered on specular reflection
    - Directive: broader vMF
    - Broad: cosine-weighted hemisphere
    - FSD: free-space diffraction importance sampling (optional, if aperture data provided)

    When FSD data is provided, the BSDF lobe weights are reduced by (1-β) and
    FSD gets weight β, forming a 5-component mixture.

    Args:
        wo: mi.Vector3f [n_paths] outgoing direction
        n: mi.Vector3f [n_paths] surface normal (possibly flipped)
        eps_real..thickness: mi.Float [n_paths] material parameters
        n_paths: int number of active paths
        seed: int RNG seed
        wavelength: float
        enable_incoherent: bool
        enable_slab_fresnel: bool
        energy_gate: Optional mi.Float [n_paths] pre-computed energy gate (R_jones)
        fsd_flat: Optional FlatEdgeData — packed edge data for FSD sampling
        fsd_aperture_idx: Optional mi.UInt32 [n_paths] — aperture index per path
        fsd_beta: Optional mi.Float [n_paths] — FSD energy borrowing fraction
        fsd_k: Optional float — wavenumber for FSD
        fsd_tables: Optional FsdSamplingTables — precomputed CDF tables
        fsd_sir_candidates: int — number of SIR candidates (default 8)

    Returns:
        sampled_dir: mi.Vector3f [n_paths] sampled direction
        mixture_pdf: mi.Float [n_paths] mixture PDF at sampled direction
    """
    from ..bsdf.mmwave_scalar import (
        sample_ggx_vndf_drjit,
        eval_ggx_pdf_drjit,
        sample_vmf_drjit,
        eval_vmf_pdf_drjit,
        compute_lobe_weights_drjit,
        compute_mixture_pdf_drjit,
    )

    # Step 1: Compute lobe probabilities and distribution parameters
    p_KA, p_SPM, p_dir, p_broad, alpha, kappa_spm, kappa_dir = \
        compute_lobe_weights_drjit(
            wo, n, eps_real, eps_imag, sigma_h, l_c, tau_base, thickness,
            wavelength=wavelength,
            enable_incoherent=enable_incoherent,
            enable_slab_fresnel=enable_slab_fresnel,
            energy_gate=energy_gate)

    # Step 1b: If FSD is active, reduce reflection lobes and add FSD probability
    has_fsd = (fsd_flat is not None and fsd_aperture_idx is not None
               and fsd_beta is not None and fsd_tables is not None)
    if has_fsd:
        one_minus_beta = mi.Float(1.0) - fsd_beta
        p_KA = p_KA * one_minus_beta
        p_SPM = p_SPM * one_minus_beta
        p_dir = p_dir * one_minus_beta
        p_broad = p_broad * one_minus_beta
        p_FSD = fsd_beta
    else:
        p_FSD = mi.Float(0.0)

    # Step 2: Draw lobe selector
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(seed, n_paths)
    u_lobe = sampler.next_1d()

    # Cumulative thresholds (5 lobes)
    c1 = p_KA
    c2 = c1 + p_SPM
    c3 = c2 + p_dir
    c4 = c3 + p_broad
    # c5 = 1.0 (FSD fills remainder if present, else broad absorbs all)

    select_KA = u_lobe < c1
    select_SPM = (u_lobe >= c1) & (u_lobe < c2)
    select_dir = (u_lobe >= c2) & (u_lobe < c3)
    if has_fsd:
        select_broad = (u_lobe >= c3) & (u_lobe < c4)
        select_FSD = u_lobe >= c4
    else:
        select_broad = u_lobe >= c3
        select_FSD = mi.Bool(False)

    # Step 3: Sample from each selected lobe (lazy — only compute per-lobe)
    # Use different seed offsets so each lobe gets independent random numbers
    sampled_dir = mi.Vector3f(0, 0, 1)  # placeholder

    # KA lobe: GGX VNDF sampling
    dir_ka, _ = sample_ggx_vndf_drjit(wo, n, alpha, n_paths, seed + 100)
    sampled_dir = dr.select(select_KA, dir_ka, sampled_dir)

    # SPM lobe: vMF sampling
    dir_spm, _ = sample_vmf_drjit(wo, n, kappa_spm, n_paths, seed + 200)
    sampled_dir = dr.select(select_SPM, dir_spm, sampled_dir)

    # Directive lobe: broader vMF
    dir_directive, _ = sample_vmf_drjit(wo, n, kappa_dir, n_paths, seed + 300)
    sampled_dir = dr.select(select_dir, dir_directive, sampled_dir)

    # Broad lobe: cosine hemisphere
    dir_cos, _, _ = _sample_cosine_hemisphere_drjit(n, n_paths, seed + 400)
    sampled_dir = dr.select(select_broad, dir_cos, sampled_dir)

    # FSD lobe: diffraction importance sampling via SIR
    if has_fsd:
        from ..diffraction.fsd_sampling import sample_fsd_direction_drjit
        # Use safe aperture indices (clamp invalid 0xFFFFFFFF to 0) for gather safety
        fsd_valid = fsd_aperture_idx != mi.UInt32(0xFFFFFFFF)
        safe_ap_idx_fsd = dr.select(fsd_valid, fsd_aperture_idx, mi.UInt32(0))
        fsd_wx, fsd_wy, fsd_wz, _ = sample_fsd_direction_drjit(
            fsd_flat, safe_ap_idx_fsd, fsd_k, fsd_sir_candidates,
            seed + 500, fsd_tables)
        dir_fsd = mi.Vector3f(fsd_wx, fsd_wy, fsd_wz)
        # Only apply FSD direction for paths that were FSD-selected AND have valid apertures
        use_fsd = select_FSD & fsd_valid
        sampled_dir = dr.select(use_fsd, dir_fsd, sampled_dir)

    # Normalize for safety
    d_len = dr.maximum(dr.norm(sampled_dir), mi.Float(1e-10))
    sampled_dir = mi.Vector3f(sampled_dir.x / d_len, sampled_dir.y / d_len, sampled_dir.z / d_len)

    # Ensure above hemisphere
    s_dot_n = dr.dot(sampled_dir, n)
    below = s_dot_n < mi.Float(0.0)
    # Reflect through n
    sampled_dir = dr.select(below,
        mi.Vector3f(
            sampled_dir.x - mi.Float(2.0) * s_dot_n * n.x,
            sampled_dir.y - mi.Float(2.0) * s_dot_n * n.y,
            sampled_dir.z - mi.Float(2.0) * s_dot_n * n.z,
        ),
        sampled_dir)

    # Step 4: Evaluate mixture PDF at the sampled direction (all lobe PDFs)
    # The base 4-lobe mixture PDF at the final (above-hemisphere) direction
    mixture_pdf = compute_mixture_pdf_drjit(
        sampled_dir, wo, n,
        p_KA, p_SPM, p_dir, p_broad,
        alpha, kappa_spm, kappa_dir)

    # Account for below-hemisphere fold: two pre-image directions map to each
    # above-hemisphere direction (ω itself and reflect(ω,n)), so the effective
    # sampling density is p_mix(ω) + p_mix(reflect(ω,n)).
    s_dot_n_fold = dr.dot(sampled_dir, n)
    dir_reflected = mi.Vector3f(
        sampled_dir.x - mi.Float(2.0) * s_dot_n_fold * n.x,
        sampled_dir.y - mi.Float(2.0) * s_dot_n_fold * n.y,
        sampled_dir.z - mi.Float(2.0) * s_dot_n_fold * n.z)
    pdf_reflected = compute_mixture_pdf_drjit(
        dir_reflected, wo, n,
        p_KA, p_SPM, p_dir, p_broad,
        alpha, kappa_spm, kappa_dir)
    mixture_pdf = mixture_pdf + pdf_reflected

    # Add FSD PDF component if active
    if has_fsd:
        from ..diffraction.fsd_sampling import eval_fsd_pdf_drjit, world_to_screen_drjit

        # Only paths with valid apertures (idx != 0xFFFFFFFF) need FSD PDF
        fsd_valid = fsd_aperture_idx != mi.UInt32(0xFFFFFFFF)
        # Clamp invalid indices to 0 for safe gather (result masked out below)
        safe_ap_idx = dr.select(fsd_valid, fsd_aperture_idx, mi.UInt32(0))

        # Upload per-aperture screen frame arrays to GPU
        g_tang_x = mi.Float(fsd_flat.ap_tangent[:, 0])
        g_tang_y = mi.Float(fsd_flat.ap_tangent[:, 1])
        g_tang_z = mi.Float(fsd_flat.ap_tangent[:, 2])
        g_bt_x = mi.Float(fsd_flat.ap_bitangent[:, 0])
        g_bt_y = mi.Float(fsd_flat.ap_bitangent[:, 1])
        g_bt_z = mi.Float(fsd_flat.ap_bitangent[:, 2])
        g_wo_x = mi.Float(fsd_flat.ap_wo_dir[:, 0])
        g_wo_y = mi.Float(fsd_flat.ap_wo_dir[:, 1])
        g_wo_z = mi.Float(fsd_flat.ap_wo_dir[:, 2])

        t_x = dr.gather(mi.Float, g_tang_x, safe_ap_idx)
        t_y = dr.gather(mi.Float, g_tang_y, safe_ap_idx)
        t_z = dr.gather(mi.Float, g_tang_z, safe_ap_idx)
        b_x = dr.gather(mi.Float, g_bt_x, safe_ap_idx)
        b_y = dr.gather(mi.Float, g_bt_y, safe_ap_idx)
        b_z = dr.gather(mi.Float, g_bt_z, safe_ap_idx)
        w_x = dr.gather(mi.Float, g_wo_x, safe_ap_idx)
        w_y = dr.gather(mi.Float, g_wo_y, safe_ap_idx)
        w_z = dr.gather(mi.Float, g_wo_z, safe_ap_idx)

        fsd_xi_x, fsd_xi_y = world_to_screen_drjit(
            sampled_dir.x, sampled_dir.y, sampled_dir.z,
            t_x, t_y, t_z, b_x, b_y, b_z, w_x, w_y, w_z)

        fsd_pdf = eval_fsd_pdf_drjit(
            fsd_flat, fsd_xi_x, fsd_xi_y, safe_ap_idx, fsd_k)

        # Zero out FSD PDF for paths without valid apertures
        fsd_pdf = dr.select(fsd_valid, fsd_pdf, mi.Float(0.0))

        mixture_pdf = mixture_pdf + p_FSD * fsd_pdf

    return sampled_dir, mixture_pdf
