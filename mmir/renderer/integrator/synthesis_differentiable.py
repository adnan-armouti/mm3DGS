"""
Phase B differentiable re-synthesis from cached geometry.

Contains the two-phase differentiable rendering pipeline where Phase A
(non-differentiable) caches geometry, and Phase B (this module) re-evaluates
BSDF with AD-attached material parameters:
- synthesize_differentiable (main Phase B entry point)
- _recompute_* helpers (positions, normals, antenna gains)
- _synthesize_specular_* helpers (specular path processing)
"""

from typing import Optional, TYPE_CHECKING, Tuple
import drjit as dr
import mitsuba as mi
import numpy as np
import time

from ..bsdf.mmwave_scalar import BSDFmmWaveScalar
from ..bsdf.mmwave_jones import BSDFmmWaveJones
from ..utils.math import (
    to_local,
    check_visibility_bidirectional,
    safe_normalize,
    compute_efield_energy,
    gather_point3f,
    gather_vector3f,
)
from ..sampler import ReservoirHits
from ..specular.image_method import SpecularPaths
from ..materials.parameterization import MaterialParameterization, PerTriangleParameterization

from .cached_geometry import ADCResult, ADCComponentResult, CachedGeometry
from .math_primitives import diff_ray_triangle_intersect, get_ad_triangle_vertices
from .mixture_sampling import _sample_cosine_hemisphere_drjit, _sample_mixture_bsdf_drjit

# Import diffraction modules (optional)
try:
    from ..diffraction import (DiffractionConfig, TriangleSpatialHash, FsdAperture,
                               construct_aperture, FsdBSDF, FlatEdgeData,
                               pack_apertures, eval_batch, eval_batch_drjit)
    from ..diffraction.triangle_search import build_triangle_hash_from_scene
    _HAS_DIFFRACTION = True
except ImportError:
    _HAS_DIFFRACTION = False

# Import pattern evaluation function
try:
    from mmir.sensor.element_patterns import (
        evaluate_combined_gain,
        AntennaPatternLoader,
        load_pattern_rrts_format,
        evaluate_gain_rrts_style_numpy,
        evaluate_gain_rrts_style_vectorized,
        evaluate_gain_rrts_style_drjit,
        evaluate_gain_rrts_style_drjit_product,
        RRTSPatternLoader,
    )
except ImportError:
    evaluate_combined_gain = None
    AntennaPatternLoader = None
    load_pattern_rrts_format = None
    evaluate_gain_rrts_style_numpy = None
    evaluate_gain_rrts_style_vectorized = None
    evaluate_gain_rrts_style_drjit = None
    evaluate_gain_rrts_style_drjit_product = None
    RRTSPatternLoader = None

# Physical Constants
C = 299792458.0


def synthesize_differentiable(
    self,
    cached_geom: 'CachedGeometry',
    physics_params: list,
    pose_params: Optional[dict] = None,
    normal_params: Optional[list] = None,
    pattern_loaders: Optional[dict] = None,
    vertex_offset_params: Optional[list] = None,
) -> Tuple['mi.Float', 'mi.Float']:
    """
    Phase B: Differentiable ADC re-synthesis using cached geometry.

    Re-evaluates material-dependent and optionally geometry-dependent
    quantities while keeping ray-tracing topology frozen.

    Supports multiple differentiable parameter groups:
    - physics_params: Always used. 6 mi.Float arrays for material properties.
    - pose_params: Optional. Dict with 'pitch','roll','yaw','tx','ty','tz'.
        Recomputes TX/RX→hit distances and directions from live pose.
    - normal_params: Optional. List of 3 mi.Float arrays [nx, ny, nz] per vertex.
        Recomputes normals via barycentric interpolation.
    - pattern_loaders: Optional. Dict with 'tx' and 'rx' AntennaPatternLoader
        instances (grad-enabled). Recomputes antenna gains.
    - vertex_offset_params: Optional. List of 3 mi.Float arrays [dx, dy, dz]
        per vertex. Recomputes hit positions and ALL distances.

    Args:
        cached_geom: CachedGeometry from Phase A (frozen geometry/visibility).
        physics_params: list of 6 mi.Float arrays (grad-enabled), each of
            length n_params: [eps_real, eps_imag, sigma_h, l_c, tau, thickness].
        pose_params: Optional dict of mi.Float scalars for 6DOF pose.
        normal_params: Optional list [nx, ny, nz] of mi.Float arrays per vertex.
        pattern_loaders: Optional dict {'tx': loader, 'rx': loader}.
        vertex_offset_params: Optional list [dx, dy, dz] of mi.Float per vertex.

    Returns:
        (adc_real_flat, adc_imag_flat): Live DrJit mi.Float arrays [n_tx * n_rx * K].
    """
    from ..utils.math import gather_point3f, gather_vector3f

    n_tx = cached_geom.n_tx
    n_rx = cached_geom.n_rx
    n_total = cached_geom.n_total
    K = self.num_samples
    active = cached_geom.active

    # ====================================================================
    # Step 0: Compute live geometry overrides (pose, normals, positions)
    # ====================================================================

    # --- Hit positions: default frozen, override if vertex offsets provided ---
    hit_P = cached_geom.hit_P_expanded
    if vertex_offset_params is not None and cached_geom.hit_bary_u is not None:
        hit_P = self._recompute_hit_positions(cached_geom, vertex_offset_params)

    # --- TX/RX positions: default from cache, override if pose provided ---
    if pose_params is not None and cached_geom.tx_positions is not None:
        from ..utils.transforms import transform_positions
        tx_pos_live = transform_positions(cached_geom.tx_positions, pose_params)
        rx_pos_live = transform_positions(cached_geom.rx_positions, pose_params)
    else:
        tx_pos_live = cached_geom.tx_positions
        rx_pos_live = cached_geom.rx_positions

    # --- Recompute distances and directions if pose or geometry changed ---
    if pose_params is not None or vertex_offset_params is not None:
        # Gather per-path TX/RX positions from per-element arrays
        tx_pos_expanded = gather_point3f(tx_pos_live, cached_geom.tx_idx)
        rx_pos_expanded = gather_point3f(rx_pos_live, cached_geom.rx_idx)

        # Recompute distances
        delta_tx = tx_pos_expanded - hit_P
        d_hit_to_tx = dr.norm(delta_tx)
        delta_rx = hit_P - rx_pos_expanded
        d_rx_to_hit = dr.norm(delta_rx)

        # Recompute directions
        dir_hit_to_tx = mi.Vector3f(
            delta_tx.x / dr.maximum(d_hit_to_tx, 1e-10),
            delta_tx.y / dr.maximum(d_hit_to_tx, 1e-10),
            delta_tx.z / dr.maximum(d_hit_to_tx, 1e-10),
        )
        dir_hit_to_rx = mi.Vector3f(
            -delta_rx.x / dr.maximum(d_rx_to_hit, 1e-10),
            -delta_rx.y / dr.maximum(d_rx_to_hit, 1e-10),
            -delta_rx.z / dr.maximum(d_rx_to_hit, 1e-10),
        )
    else:
        d_hit_to_tx = cached_geom.d_hit_to_tx
        d_rx_to_hit = cached_geom.d_rx_to_hit
        dir_hit_to_tx = cached_geom.dir_hit_to_tx
        dir_hit_to_rx = cached_geom.dir_hit_to_rx

    # --- Normals: default frozen, override if normal_params provided ---
    if normal_params is not None and cached_geom.hit_bary_u is not None:
        hit_N = self._recompute_normals(cached_geom, normal_params)
        # Re-apply double-sided flip (consistent with Phase A)
        double_sided = getattr(self.render_config, 'double_sided', True)
        if double_sided:
            cos_out = dr.dot(dir_hit_to_rx, hit_N)
            need_flip = cos_out < 0
            hit_N = dr.select(
                need_flip,
                mi.Vector3f(-hit_N.x, -hit_N.y, -hit_N.z),
                hit_N,
            )
    else:
        hit_N = cached_geom.hit_N_expanded

    # ====================================================================
    # Step 1: Gather/interpolate per-hit materials from grad-enabled arrays
    # ====================================================================
    if self.material_param is not None:
        per_hit = self.material_param.gather(physics_params, cached_geom)
    else:
        per_hit = [
            dr.gather(mi.Float, physics_params[i], cached_geom.hit_prim_ids, active)
            for i in range(len(physics_params))
        ]
    eps_real, eps_imag, sigma_h, l_c, tau, thickness = per_hit

    # Pre-compute per-triangle physics params for diffraction & SMS paths
    # (these index by face/prim IDs, not vertex IDs)
    from ..materials.parameterization import PerVertexParameterization
    if isinstance(self.material_param, PerVertexParameterization):
        physics_params_tri = self.material_param.vertex_to_triangle_drjit(physics_params)
    else:
        physics_params_tri = physics_params

    # ====================================================================
    # Step 2: Evaluate BSDF (differentiable w.r.t. material + normal params)
    # ====================================================================
    _eval_f_cos = getattr(self.bsdf, 'eval_f_cos_physics', self.bsdf.eval_f_cos)
    brdf_weight = _eval_f_cos(
        wo=dir_hit_to_rx,
        wi=dir_hit_to_tx,
        n=hit_N,
        eps_real=eps_real, eps_imag=eps_imag,
        sigma_h=sigma_h, l_c=l_c, tau=tau,
        thickness=thickness,
    )

    # ====================================================================
    # Step 3: Apply antenna gain (frozen or live) and MC correction
    # ====================================================================
    if pattern_loaders is not None and evaluate_combined_gain is not None:
        # Recompute antenna gains from live pattern parameters
        antenna_gain = self._recompute_antenna_gains(
            cached_geom, pattern_loaders, dir_hit_to_tx, dir_hit_to_rx)
        brdf_weight = brdf_weight * antenna_gain
    elif cached_geom.antenna_gain_combined is not None:
        brdf_weight = brdf_weight * cached_geom.antenna_gain_combined

    if cached_geom.mc_correction is not None:
        brdf_weight = brdf_weight * cached_geom.mc_correction

    # ====================================================================
    # Step 3b: Diffraction energy borrowing (live β if materials are grad-enabled)
    # ====================================================================
    diff_config = self.diffraction_config
    if (diff_config is not None and diff_config.enabled
            and diff_config.energy_borrowing
            and cached_geom.diff_beta_np is not None):
        if (cached_geom.diff_flat is not None
                and cached_geom.diff_flat.n_edges > 0
                and dr.grad_enabled(physics_params[0])):
            # Fix 6: Recompute β with live materials for gradient flow
            one_minus_beta = self._recompute_beta_differentiable(
                cached_geom, physics_params_tri)
        else:
            one_minus_beta = mi.Float(1.0 - cached_geom.diff_beta_np)
        brdf_weight = brdf_weight * one_minus_beta

    # ====================================================================
    # Step 4: Compute weight (radar equation or legacy)
    # ====================================================================
    if self.render_config.use_radar_equation:
        # Clamp distance to avoid exploding 1/d² gradients when d→0
        d_safe = dr.maximum(d_hit_to_tx, mi.Float(1e-4))  # 0.1mm minimum
        path_loss = mi.Float(1.0) / (d_safe * d_safe)
        radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
        # Use epsilon in sqrt to avoid NaN backward: d(sqrt(x))/dx = 1/(2*sqrt(x)) → inf at x=0
        weight = radar_scale * dr.sqrt(dr.maximum(brdf_weight * path_loss, mi.Float(1e-20)))
    else:
        weight = brdf_weight

    # ====================================================================
    # Steps 5-7: Phase computation + scatter-add accumulation
    # Fast path: use pre-computed phasors when pose/geometry are frozen
    # ====================================================================
    use_cached_phasors = (
        cached_geom.cached_cos_phi is not None
        and pose_params is None
        and vertex_offset_params is None
    )

    if not isinstance(weight, mi.Float):
        weight = mi.Float(weight)

    if use_cached_phasors:
        # === FAST PATH: use pre-computed cos/sin phasors ===
        # Weight expansion via gather (replaces dr.tile)
        weight_expanded = dr.gather(mi.Float, weight, cached_geom.cached_path_idx)

        contrib_real = weight_expanded * cached_geom.cached_cos_phi
        contrib_imag = weight_expanded * cached_geom.cached_sin_phi

        adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
        adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

        dr.scatter_add(adc_real_flat, contrib_real,
                       cached_geom.cached_flat_idx, cached_geom.cached_active_3d)
        dr.scatter_add(adc_imag_flat, contrib_imag,
                       cached_geom.cached_flat_idx, cached_geom.cached_active_3d)
    else:
        # === STANDARD PATH: compute phase from scratch ===
        R_total = d_rx_to_hit + d_hit_to_tx
        tau_delay = R_total / C

        TWO_PI = 2.0 * np.pi
        phi_const = TWO_PI * self.min_freq * tau_delay
        phi_slope = TWO_PI * self.slope * tau_delay

        t_grid_dr = self._t_grid_dr  # [K] (cached on GPU)

        # Expand to [n_total × K]
        phi_const_3d = dr.tile(phi_const, K)
        phi_slope_3d = dr.tile(phi_slope, K)
        t_k = dr.repeat(t_grid_dr, n_total)
        active_3d = dr.tile(active, K)

        phi = phi_const_3d + phi_slope_3d * t_k
        cos_phi = dr.cos(phi)
        sin_phi = dr.sin(phi)
        if not self.render_config.enable_grad_phase:
            cos_phi = dr.detach(cos_phi)
            sin_phi = dr.detach(sin_phi)

        # Weight expansion
        weight_3d = dr.tile(weight, K)

        contrib_real = weight_3d * cos_phi
        contrib_imag = weight_3d * sin_phi

        # Flat index computation for scatter_add
        tx_idx_3d = dr.tile(cached_geom.tx_idx, K)
        rx_idx_3d = dr.tile(cached_geom.rx_idx, K)

        base_idx = dr.arange(mi.UInt32, n_total * K)
        k_idx = base_idx // mi.UInt32(n_total)

        flat_idx = tx_idx_3d * mi.UInt32(n_rx * K) + rx_idx_3d * mi.UInt32(K) + k_idx

        adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
        adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

        dr.scatter_add(adc_real_flat, contrib_real, flat_idx, active_3d)
        dr.scatter_add(adc_imag_flat, contrib_imag, flat_idx, active_3d)

    # ====================================================================
    # Step 8: Differentiable specular path contribution (Fix 5)
    # ====================================================================
    if cached_geom.specular_n_paths > 0 and cached_geom.specular_valid is not None:
        self._synthesize_specular_differentiable(
            cached_geom, physics_params_tri,
            adc_real_flat, adc_imag_flat,
            n_tx, n_rx, K,
            pose_params=pose_params,
            pattern_loaders=pattern_loaders,
            normal_params=normal_params,
            vertex_offset_params=vertex_offset_params,
        )

    return adc_real_flat, adc_imag_flat

def _recompute_hit_positions(self, cached_geom, vertex_offset_params):
    """Recompute hit positions from live vertex offsets + cached barycentrics."""
    u = cached_geom.hit_bary_u
    v = cached_geom.hit_bary_v
    w = mi.Float(1.0) - u - v

    vi0 = cached_geom.hit_vertex_ids_0
    vi1 = cached_geom.hit_vertex_ids_1
    vi2 = cached_geom.hit_vertex_ids_2
    active = cached_geom.active
    dx, dy, dz = vertex_offset_params

    # Gather frozen base positions from cached hit positions
    # (barycentric centroid of base mesh vertices)
    # For offset parameterization, base = frozen mesh, offset = live
    # hit_P = bary_interp(base_v + offset_v)
    # We gather the offsets for each vertex of each hit triangle
    ox0 = dr.gather(mi.Float, dx, vi0, active)
    ox1 = dr.gather(mi.Float, dx, vi1, active)
    ox2 = dr.gather(mi.Float, dx, vi2, active)
    oy0 = dr.gather(mi.Float, dy, vi0, active)
    oy1 = dr.gather(mi.Float, dy, vi1, active)
    oy2 = dr.gather(mi.Float, dy, vi2, active)
    oz0 = dr.gather(mi.Float, dz, vi0, active)
    oz1 = dr.gather(mi.Float, dz, vi1, active)
    oz2 = dr.gather(mi.Float, dz, vi2, active)

    # Barycentric interpolation of the offset
    offset_x = w * ox0 + u * ox1 + v * ox2
    offset_y = w * oy0 + u * oy1 + v * oy2
    offset_z = w * oz0 + u * oz1 + v * oz2

    # Add to frozen base position
    return mi.Point3f(
        cached_geom.hit_P_expanded.x + offset_x,
        cached_geom.hit_P_expanded.y + offset_y,
        cached_geom.hit_P_expanded.z + offset_z,
    )

def _recompute_normals(self, cached_geom, normal_params):
    """Recompute normals from live per-vertex normal params + cached barycentrics."""
    u = cached_geom.hit_bary_u
    v = cached_geom.hit_bary_v
    w = mi.Float(1.0) - u - v

    vi0 = cached_geom.hit_vertex_ids_0
    vi1 = cached_geom.hit_vertex_ids_1
    vi2 = cached_geom.hit_vertex_ids_2
    active = cached_geom.active
    nx_arr, ny_arr, nz_arr = normal_params

    # Gather per-vertex normals
    nx0 = dr.gather(mi.Float, nx_arr, vi0, active)
    nx1 = dr.gather(mi.Float, nx_arr, vi1, active)
    nx2 = dr.gather(mi.Float, nx_arr, vi2, active)
    ny0 = dr.gather(mi.Float, ny_arr, vi0, active)
    ny1 = dr.gather(mi.Float, ny_arr, vi1, active)
    ny2 = dr.gather(mi.Float, ny_arr, vi2, active)
    nz0 = dr.gather(mi.Float, nz_arr, vi0, active)
    nz1 = dr.gather(mi.Float, nz_arr, vi1, active)
    nz2 = dr.gather(mi.Float, nz_arr, vi2, active)

    # Barycentric interpolation
    nx_interp = w * nx0 + u * nx1 + v * nx2
    ny_interp = w * ny0 + u * ny1 + v * ny2
    nz_interp = w * nz0 + u * nz1 + v * nz2

    # Normalize (avoid division by zero)
    n_interp = mi.Vector3f(nx_interp, ny_interp, nz_interp)
    n_len = dr.maximum(dr.norm(n_interp), 0.01)
    return mi.Vector3f(nx_interp / n_len, ny_interp / n_len, nz_interp / n_len)

def _recompute_antenna_gains(self, cached_geom, pattern_loaders,
                              dir_hit_to_tx, dir_hit_to_rx):
    """Recompute antenna gains from live pattern parameters."""
    from ..utils.math import gather_vector3f

    tx_loader = pattern_loaders.get('tx')
    rx_loader = pattern_loaders.get('rx')

    gain = mi.Float(1.0)

    if tx_loader is not None and cached_geom.tx_boresights is not None:
        tx_boresights_exp = gather_vector3f(cached_geom.tx_boresights, cached_geom.tx_idx)
        gain_tx = evaluate_combined_gain(tx_loader, dir_hit_to_tx, tx_boresights_exp)
        gain = gain * gain_tx

    if rx_loader is not None and cached_geom.rx_boresights is not None:
        rx_boresights_exp = gather_vector3f(cached_geom.rx_boresights, cached_geom.rx_idx)
        # Direction from RX to hit = -dir_hit_to_rx
        dir_rx_to_hit = mi.Vector3f(-dir_hit_to_rx.x, -dir_hit_to_rx.y, -dir_hit_to_rx.z)
        gain_rx = evaluate_combined_gain(rx_loader, dir_rx_to_hit, rx_boresights_exp)
        gain = gain * gain_rx

    return gain

def _recompute_specular_antenna_gains(self, cached_geom, pattern_loaders,
                                       sp_dir_to_tx, sp_dir_to_rx):
    """Recompute antenna gains for specular paths from live pattern parameters.

    Mirrors _recompute_antenna_gains() but uses specular-specific indices.
    Direction convention matches diffuse path: TX gets dir_hit_to_tx (from
    hit toward TX), RX gets -dir_hit_to_rx (from RX toward hit).
    """
    from ..utils.math import gather_vector3f

    tx_loader = pattern_loaders.get('tx')
    rx_loader = pattern_loaders.get('rx')

    gain = mi.Float(1.0)

    if tx_loader is not None and cached_geom.tx_boresights is not None:
        tx_boresights_exp = gather_vector3f(
            cached_geom.tx_boresights, cached_geom.specular_tx_idx)
        # sp_dir_to_tx = from hit toward TX (same as dir_hit_to_tx in diffuse)
        gain_tx = evaluate_combined_gain(tx_loader, sp_dir_to_tx, tx_boresights_exp)
        gain = gain * gain_tx

    if rx_loader is not None and cached_geom.rx_boresights is not None:
        rx_boresights_exp = gather_vector3f(
            cached_geom.rx_boresights, cached_geom.specular_rx_idx)
        # sp_dir_to_rx = from hit toward RX; negate for RX antenna frame
        # (matches diffuse: dir_rx_to_hit = -dir_hit_to_rx)
        dir_rx_to_hit = mi.Vector3f(
            -sp_dir_to_rx.x, -sp_dir_to_rx.y, -sp_dir_to_rx.z)
        gain_rx = evaluate_combined_gain(rx_loader, dir_rx_to_hit, rx_boresights_exp)
        gain = gain * gain_rx

    return gain

def _recompute_specular_positions_ift(self, cached_geom, vertex_offset_params):
    """
    Recompute specular hit positions with IFT-derived vertex gradients.

    Forward: Applies barycentric interpolation of vertex offsets to get the
    naive position shift, then applies a first-order IFT correction by
    re-evaluating the half-vector constraint with the shifted surface.

    Backward: DrJit's AD automatically propagates gradients through the
    constraint evaluation and IFT correction, giving correct dx*/dvertex.

    Returns:
        Tuple of (live_P, live_dp_du, live_dp_dv, live_N) with gradient tracking.
    """
    from ..utils.math import safe_normalize

    gi = cached_geom.specular_grad_info
    dx, dy, dz = vertex_offset_params
    valid = cached_geom.specular_valid

    u = cached_geom.specular_bary_u
    v = cached_geom.specular_bary_v
    w = mi.Float(1.0) - u - v

    vi0 = cached_geom.specular_vertex_ids_0
    vi1 = cached_geom.specular_vertex_ids_1
    vi2 = cached_geom.specular_vertex_ids_2

    # Gather vertex offsets for the three vertices of each converged triangle
    ox0 = dr.gather(mi.Float, dx, vi0, valid)
    ox1 = dr.gather(mi.Float, dx, vi1, valid)
    ox2 = dr.gather(mi.Float, dx, vi2, valid)
    oy0 = dr.gather(mi.Float, dy, vi0, valid)
    oy1 = dr.gather(mi.Float, dy, vi1, valid)
    oy2 = dr.gather(mi.Float, dy, vi2, valid)
    oz0 = dr.gather(mi.Float, dz, vi0, valid)
    oz1 = dr.gather(mi.Float, dz, vi1, valid)
    oz2 = dr.gather(mi.Float, dz, vi2, valid)

    # Barycentric interpolation: naive shift of specular point
    naive_dx = w * ox0 + u * ox1 + v * ox2
    naive_dy = w * oy0 + u * oy1 + v * oy2
    naive_dz = w * oz0 + u * oz1 + v * oz2

    # Base (frozen) specular position + naive shift
    p_base = mi.Point3f(
        cached_geom.specular_hit_P.x + naive_dx,
        cached_geom.specular_hit_P.y + naive_dy,
        cached_geom.specular_hit_P.z + naive_dz)

    # Live tangent vectors: dp/du = (v1+δv1) - (v0+δv0), dp/dv = (v2+δv2) - (v0+δv0)
    # The base dp/du from the Newton solver is already v1-v0, so the live offset is:
    #   live_dp_du = dp_du + (δv1 - δv0), live_dp_dv = dp_dv + (δv2 - δv0)
    live_dp_du = mi.Vector3f(
        gi.dp_du.x + (ox1 - ox0),
        gi.dp_du.y + (oy1 - oy0),
        gi.dp_du.z + (oz1 - oz0))
    live_dp_dv = mi.Vector3f(
        gi.dp_dv.x + (ox2 - ox0),
        gi.dp_dv.y + (oy2 - oy0),
        gi.dp_dv.z + (oz2 - oz0))

    # Live normal: n = normalize(dp_du × dp_dv)
    cross = dr.cross(live_dp_du, live_dp_dv)
    n_len = dr.maximum(dr.norm(cross), mi.Float(1e-10))
    live_N = mi.Vector3f(cross.x / n_len, cross.y / n_len, cross.z / n_len)

    # Evaluate half-vector constraint at the shifted point
    # wi = normalize(rx - p), wo = normalize(tx - p)
    wi_raw = mi.Vector3f(
        gi.rx_pos.x - p_base.x,
        gi.rx_pos.y - p_base.y,
        gi.rx_pos.z - p_base.z)
    wo_raw = mi.Vector3f(
        gi.tx_pos.x - p_base.x,
        gi.tx_pos.y - p_base.y,
        gi.tx_pos.z - p_base.z)

    wi = safe_normalize(wi_raw)
    wo = safe_normalize(wo_raw)
    h = safe_normalize(mi.Vector3f(wi.x + wo.x, wi.y + wo.y, wi.z + wo.z))

    # Tangent frame via Gram-Schmidt
    n_dot_dpdu = dr.dot(live_N, live_dp_du)
    s_raw = mi.Vector3f(
        live_dp_du.x - live_N.x * n_dot_dpdu,
        live_dp_du.y - live_N.y * n_dot_dpdu,
        live_dp_du.z - live_N.z * n_dot_dpdu)
    s = safe_normalize(s_raw)
    t = dr.cross(live_N, s)

    # Constraint residual: C = [s·h, t·h] (should be ≈ 0 if offsets are small)
    C_s = dr.dot(s, h)
    C_t = dr.dot(t, h)

    # IFT correction: δ(u,v) = -J⁻¹ · C
    delta_u = -(gi.inv_J00 * C_s + gi.inv_J01 * C_t)
    delta_v = -(gi.inv_J10 * C_s + gi.inv_J11 * C_t)

    # Corrected position: x* = p_base + dp/du · δu + dp/dv · δv
    live_P = mi.Point3f(
        p_base.x + live_dp_du.x * delta_u + live_dp_dv.x * delta_v,
        p_base.y + live_dp_du.y * delta_u + live_dp_dv.y * delta_v,
        p_base.z + live_dp_du.z * delta_u + live_dp_dv.z * delta_v)

    return live_P, live_dp_du, live_dp_dv, live_N

def _recompute_beta_differentiable(self, cached_geom, physics_params):
    """Recompute diffraction energy-borrowing β from live material params.

    Uses cached FlatEdgeData with per-edge face indices and cos_theta to
    recompute Fresnel opacity with live materials. The correction ratio
    (live_opacity / phase_a_opacity) scales the per-aperture β.

    Returns:
        one_minus_beta: mi.Float [n_total] with gradient tracking through
                        material params → Fresnel opacity → β → (1-β).
    """
    flat = cached_geom.diff_flat
    beta_np = cached_geom.diff_beta_np
    n_total = len(beta_np)

    if flat is None or flat.n_apertures == 0 or flat.n_edges == 0:
        return mi.Float(1.0 - beta_np)

    from ..diffraction.fsd_bsdf import (
        _fresnel_opacity_drjit,
    )

    # Step 1: Gather live material at each edge's face/vertex
    edge_face = flat.edge_face_idx  # [E] int32
    edge_cos = flat.edge_cos_theta  # [E] float64
    edge_opacity_phase_a = flat.edge_opacity  # [E] float64

    # Gather live ε at edges (per-face)
    face_idx_dr = mi.UInt32(edge_face.astype(np.int64) % (2**32))
    eps_r_live = dr.gather(mi.Float, physics_params[0], face_idx_dr)
    eps_i_live = dr.gather(mi.Float, physics_params[1], face_idx_dr)

    # Compute live Fresnel opacity: (|R_s|^2 + |R_p|^2) / 2
    cos_theta_dr = mi.Float(edge_cos.astype(np.float32))
    live_opacity = _fresnel_opacity_drjit(eps_r_live, eps_i_live, cos_theta_dr)

    # Step 2: Per-aperture correction ratio
    # For each aperture: sum(live_opacity_e * |P_t_e|) / sum(phase_a_opacity_e * |P_t_e|)
    # We approximate |P_t_e| as equal for all edges in an aperture (geometry is frozen),
    # so the ratio simplifies to: mean(live_opacity) / mean(phase_a_opacity)
    phase_a_opacity_dr = mi.Float(edge_opacity_phase_a.astype(np.float32))

    # CSR reduction: compute mean opacity per aperture
    n_ap = flat.n_apertures
    live_op_sum = dr.zeros(mi.Float, n_ap)
    pa_op_sum = dr.zeros(mi.Float, n_ap)
    edge_count_f = dr.zeros(mi.Float, n_ap)

    # Build per-edge aperture index
    edge_ap_idx = np.zeros(flat.n_edges, dtype=np.int32)
    for a in range(n_ap):
        s = flat.ap_edge_start[a]
        c = flat.ap_edge_count[a]
        edge_ap_idx[s:s+c] = a
    edge_ap_dr = mi.UInt32(edge_ap_idx)

    dr.scatter_add(live_op_sum, live_opacity, edge_ap_dr)
    dr.scatter_add(pa_op_sum, phase_a_opacity_dr, edge_ap_dr)
    dr.scatter_add(edge_count_f, mi.Float(1.0), edge_ap_dr)

    # Correction ratio per aperture
    pa_mean = pa_op_sum / dr.maximum(edge_count_f, mi.Float(1.0))
    live_mean = live_op_sum / dr.maximum(edge_count_f, mi.Float(1.0))
    correction = live_mean / dr.maximum(pa_mean, mi.Float(1e-8))

    # Step 3: Scale per-path β by correction ratio
    # Map aperture correction to paths
    eval_path_idx = cached_geom.diff_eval_path_idx
    eval_ap_idx = cached_geom.diff_eval_ap_idx
    if eval_path_idx is None or eval_ap_idx is None:
        return mi.Float(1.0 - beta_np)

    beta_dr = dr.zeros(mi.Float, n_total)
    beta_base_per_eval = mi.Float(flat.ap_beta[eval_ap_idx].astype(np.float32))
    corr_per_eval = dr.gather(mi.Float, correction, mi.UInt32(eval_ap_idx))
    beta_live_per_eval = dr.minimum(beta_base_per_eval * corr_per_eval,
                                     mi.Float(self.diffraction_config.beta_max))
    dr.scatter(beta_dr, beta_live_per_eval, mi.UInt32(eval_path_idx))

    return mi.Float(1.0) - beta_dr

def _synthesize_specular_chains_e2e(
    self,
    chains: 'MultibounceSpecularChain',
    adc_real_flat: 'mi.Float',
    adc_imag_flat: 'mi.Float',
    n_tx: int, n_rx: int, K: int,
    physics_params: list,
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    pattern_loaders: Optional[dict] = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader=None,
    rx_pattern_loader=None,
    antenna_gain_linear: bool = True,
    vertex_offset_params: Optional[list] = None,
    normal_params: Optional[list] = None,
    scene: Optional['mi.Scene'] = None,
    verbose: bool = False,
    _dist_accum: Optional[list] = None,
):
    """
    Synthesize ADC contributions from multibounce specular chains.

    For each k-bounce chain:
    1. Optionally apply IFT to get AD-attached positions
    2. Recompute directions and distances (AD through positions)
    3. Evaluate KA-only BSDF at each vertex (AD through materials)
    4. Compute product weight across all bounces
    5. Compute total path distance and phase
    6. Accumulate into ADC arrays via K-chunked scatter_add
    """
    from ..utils.math import gather_point3f
    from ..specular.sms import MultibounceSpecularChain

    k = chains.k
    n_chains = chains.n_chains
    sp_valid = chains.valid

    n_valid = int(dr.sum(mi.UInt32(sp_valid))[0])
    if n_valid == 0:
        if verbose:
            print(f"  [E2E MB-Spec] No valid specular chains, skipping")
        return

    if verbose:
        print(f"\n  [E2E MB-Spec] Synthesizing {n_valid:,} valid {k}-bounce specular chains")

    # ==================================================================
    # Step 1: IFT gradient attachment (optional)
    # ==================================================================
    ift_active = (vertex_offset_params is not None
                  and chains.grad_info is not None
                  and hasattr(self, '_scene_params') and hasattr(self, '_vp_key'))

    if ift_active:
        mesh = scene.shapes()[0] if scene is not None and len(scene.shapes()) > 0 else None
        if mesh is None:
            ift_active = False

    if ift_active:
        vp_buf = self._scene_params[self._vp_key]
        from ..specular.sms import SpecularManifoldSampler
        sms = SpecularManifoldSampler(wavelength=self.bsdf.wavelength if hasattr(self.bsdf, 'wavelength') else 3.896e-3)
        positions, normals = sms._apply_ift_multibounce_diff(
            chains, vp_buf, scene, normal_params=normal_params)
        if verbose:
            print(f"    IFT applied: {k} vertex positions AD-attached")
    else:
        # Use frozen positions from SMS solver.
        # Detach from stale AD graph — SceneContext loads vertex
        # positions/normals with grad enabled, so SMS chain positions
        # inherit stale AD connections that cause dr.backward() to hang.
        positions = [
            mi.Point3f(dr.detach(p.x), dr.detach(p.y), dr.detach(p.z))
            for p in chains.hit_P
        ]
        normals = [
            mi.Vector3f(dr.detach(n.x), dr.detach(n.y), dr.detach(n.z))
            for n in chains.hit_N
        ]
        # Also detach triangle areas (may inherit stale AD from vertex positions)
        chains.A_tri = [mi.Float(dr.detach(a)) for a in chains.A_tri]

    # ==================================================================
    # Step 2: Recompute directions and distances (AD through positions)
    # ==================================================================
    # Expand TX/RX positions per chain
    tx_pos_exp = gather_point3f(tx_positions, chains.tx_idx)
    rx_pos_exp = gather_point3f(rx_positions, chains.rx_idx)

    # Compute per-segment distances and directions
    d_segments = []
    dir_segments = []

    for i in range(k + 1):
        if i == 0:
            seg_from = rx_pos_exp
            seg_to = positions[0]
        elif i == k:
            seg_from = positions[k-1]
            seg_to = tx_pos_exp
        else:
            seg_from = positions[i-1]
            seg_to = positions[i]

        delta = mi.Vector3f(
            seg_to.x - seg_from.x,
            seg_to.y - seg_from.y,
            seg_to.z - seg_from.z)
        d = dr.norm(delta)
        d_segments.append(d)
        dir_segments.append(mi.Vector3f(
            delta.x / dr.maximum(d, mi.Float(1e-10)),
            delta.y / dr.maximum(d, mi.Float(1e-10)),
            delta.z / dr.maximum(d, mi.Float(1e-10))))

    # Total path distance
    R_total = dr.zeros(mi.Float, n_chains)
    for d_seg in d_segments:
        R_total = R_total + d_seg

    if _dist_accum is not None:
        _dist_accum[0] = _dist_accum[0] + dr.sum(R_total)

    # ==================================================================
    # Step 3: Evaluate BSDF and compute product weight at each vertex
    # ==================================================================
    specular_weight = mi.Float(1.0)

    for i in range(k):
        # Incoming and outgoing directions at vertex i
        # wi: from previous segment (toward RX or previous vertex)
        dir_in = dir_segments[i]     # direction from prev → this vertex
        dir_out = dir_segments[i+1]  # direction from this vertex → next

        # The BSDF convention: wo = toward receiver, wi = toward source
        # For vertex i: wo = -dir_in (back toward RX/prev), wi = dir_out (toward TX/next)
        wo_i = mi.Vector3f(-dir_in.x, -dir_in.y, -dir_in.z)
        wi_i = mi.Vector3f(dir_out.x, dir_out.y, dir_out.z)

        # Normal at vertex i: use learnable normal_params if available,
        # else use IFT/frozen normals
        if normal_params is not None:
            nx_arr, ny_arr, nz_arr = normal_params
            _u_sp = chains.bary_u[i]
            _v_sp = chains.bary_v[i]
            _w_sp = mi.Float(1.0) - _u_sp - _v_sp
            mesh_sp = scene.shapes()[0]
            face_idx_sp = mesh_sp.face_indices(chains.prim_ids[i], sp_valid)
            vi0_sp, vi1_sp, vi2_sp = face_idx_sp[0], face_idx_sp[1], face_idx_sp[2]
            nx_sp = (_w_sp * dr.gather(mi.Float, nx_arr, vi0_sp, sp_valid) +
                     _u_sp * dr.gather(mi.Float, nx_arr, vi1_sp, sp_valid) +
                     _v_sp * dr.gather(mi.Float, nx_arr, vi2_sp, sp_valid))
            ny_sp = (_w_sp * dr.gather(mi.Float, ny_arr, vi0_sp, sp_valid) +
                     _u_sp * dr.gather(mi.Float, ny_arr, vi1_sp, sp_valid) +
                     _v_sp * dr.gather(mi.Float, ny_arr, vi2_sp, sp_valid))
            nz_sp = (_w_sp * dr.gather(mi.Float, nz_arr, vi0_sp, sp_valid) +
                     _u_sp * dr.gather(mi.Float, nz_arr, vi1_sp, sp_valid) +
                     _v_sp * dr.gather(mi.Float, nz_arr, vi2_sp, sp_valid))
            nl_sp = dr.maximum(dr.sqrt(dr.maximum(
                nx_sp*nx_sp + ny_sp*ny_sp + nz_sp*nz_sp,
                mi.Float(1e-20))), mi.Float(0.01))
            ni = mi.Vector3f(nx_sp / nl_sp, ny_sp / nl_sp, nz_sp / nl_sp)
        else:
            ni = normals[i]
        cos_out = dr.dot(wo_i, ni)
        need_flip = cos_out < mi.Float(0.0)
        ni = mi.Vector3f(
            dr.select(need_flip, -ni.x, ni.x),
            dr.select(need_flip, -ni.y, ni.y),
            dr.select(need_flip, -ni.z, ni.z))

        cos_theta_i = dr.maximum(dr.dot(wi_i, ni), mi.Float(0.0))

        # Gather live materials at this vertex's triangle
        pid = chains.prim_ids[i]
        if self.material_param is not None:
            class _ChainGeom:
                pass
            _geom = _ChainGeom()
            _geom.hit_prim_ids = pid
            _geom.active = sp_valid
            if chains.bary_u[i] is not None:
                mesh = scene.shapes()[0]
                face_idx = mesh.face_indices(pid)
                _geom.hit_vertex_ids_0 = face_idx[0]
                _geom.hit_vertex_ids_1 = face_idx[1]
                _geom.hit_vertex_ids_2 = face_idx[2]
                _geom.hit_bary_u = chains.bary_u[i]
                _geom.hit_bary_v = chains.bary_v[i]
            else:
                _geom.hit_vertex_ids_0 = None
                _geom.hit_vertex_ids_1 = None
                _geom.hit_vertex_ids_2 = None
                _geom.hit_bary_u = None
                _geom.hit_bary_v = None
            v_materials = self.material_param.gather(physics_params, _geom)
        else:
            v_materials = [
                dr.gather(mi.Float, physics_params[j], pid, sp_valid)
                for j in range(len(physics_params))
            ]

        v_eps_real, v_eps_imag, v_sigma_h, v_l_c, v_tau, v_thickness = v_materials

        # Evaluate KA-only BSDF at each specular bounce, matching the
        # single-bounce specular path and the Sionna Phase B reference.
        _eval_ka = getattr(self.bsdf, 'eval_ka_only_physics', None)
        if _eval_ka is not None:
            v_brdf = _eval_ka(
                wo=wo_i, wi=wi_i, n=ni,
                eps_real=v_eps_real, eps_imag=v_eps_imag,
                sigma_h=v_sigma_h, l_c=v_l_c, tau=v_tau,
                thickness=v_thickness)
        else:
            # Fallback: full BSDF if KA-only not available
            _eval_f = getattr(self.bsdf, 'eval_f_physics', self.bsdf.eval_f)
            v_brdf = _eval_f(
                wo=wo_i, wi=wi_i, n=ni,
                eps_real=v_eps_real, eps_imag=v_eps_imag,
                sigma_h=v_sigma_h, l_c=v_l_c, tau=v_tau,
                thickness=v_thickness)

        # Accumulate: weight *= BRDF * cos_theta * A_tri
        v_A_tri = chains.A_tri[i]
        specular_weight = specular_weight * v_brdf * cos_theta_i * v_A_tri

    specular_weight = dr.select(sp_valid, specular_weight, mi.Float(0.0))

    # ==================================================================
    # Step 4: Path loss (1/R_total^2 for k-bounce)
    # ==================================================================
    path_loss = mi.Float(1.0) / dr.maximum(R_total * R_total, mi.Float(1e-20))

    # ==================================================================
    # Step 5: Antenna gains (first and last segments)
    # ==================================================================
    sp_antenna_gain = mi.Float(1.0)

    if pattern_loaders is not None and evaluate_combined_gain is not None \
            and tx_boresights is not None and rx_boresights is not None:
        # Differentiable antenna patterns (AD-attached loaders)
        tx_loader = pattern_loaders.get('tx')
        rx_loader = pattern_loaders.get('rx')

        if tx_loader is not None:
            tx_bs_exp = gather_vector3f(tx_boresights, chains.tx_idx)
            # dir_segments[k] = direction from last vertex toward TX
            gain_tx = evaluate_combined_gain(tx_loader, dir_segments[k], tx_bs_exp)
            sp_antenna_gain = sp_antenna_gain * gain_tx

        if rx_loader is not None:
            rx_bs_exp = gather_vector3f(rx_boresights, chains.rx_idx)
            dir_rx_rev = mi.Vector3f(-dir_segments[0].x, -dir_segments[0].y, -dir_segments[0].z)
            gain_rx = evaluate_combined_gain(rx_loader, dir_rx_rev, rx_bs_exp)
            sp_antenna_gain = sp_antenna_gain * gain_rx

    elif (tx_pattern_loader is not None or rx_pattern_loader is not None) \
            and evaluate_combined_gain is not None \
            and tx_boresights is not None and rx_boresights is not None:
        # GPU-native non-differentiable antenna patterns (no GPU→CPU sync)
        if tx_pattern_loader is not None:
            tx_bs_exp = gather_vector3f(tx_boresights, chains.tx_idx)
            gain_tx = evaluate_combined_gain(tx_pattern_loader, dir_segments[k], tx_bs_exp)
            sp_antenna_gain = sp_antenna_gain * gain_tx

        if rx_pattern_loader is not None:
            rx_bs_exp = gather_vector3f(rx_boresights, chains.rx_idx)
            dir_rx_rev = mi.Vector3f(-dir_segments[0].x, -dir_segments[0].y, -dir_segments[0].z)
            gain_rx = evaluate_combined_gain(rx_pattern_loader, dir_rx_rev, rx_bs_exp)
            sp_antenna_gain = sp_antenna_gain * gain_rx

    elif tx_boresights is not None and rx_boresights is not None:
        # CPU fallback: RRTS-style angle-based patterns
        tx_bs = gather_point3f(mi.Point3f(tx_boresights), chains.tx_idx)
        tx_bs_v = mi.Vector3f(tx_bs.x, tx_bs.y, tx_bs.z)
        cos_tx = dr.dot(dir_segments[k], tx_bs_v)
        cos_tx = dr.clamp(cos_tx, mi.Float(-1.0), mi.Float(1.0))

        rx_bs = gather_point3f(mi.Point3f(rx_boresights), chains.rx_idx)
        rx_bs_v = mi.Vector3f(rx_bs.x, rx_bs.y, rx_bs.z)
        dir_rx_rev = mi.Vector3f(-dir_segments[0].x, -dir_segments[0].y, -dir_segments[0].z)
        cos_rx = dr.dot(dir_rx_rev, rx_bs_v)
        cos_rx = dr.clamp(cos_rx, mi.Float(-1.0), mi.Float(1.0))

        if tx_pattern_rrts is not None and rx_pattern_rrts is not None:
            theta_tx = dr.acos(cos_tx)
            theta_rx = dr.acos(cos_rx)
            sp_antenna_gain = self._eval_antenna_gain_rrts(
                theta_tx, theta_rx,
                tx_pattern_rrts, rx_pattern_rrts,
                antenna_gain_linear)

    # ==================================================================
    # Step 6: K-chunked phase accumulation
    # ==================================================================
    amplitude = specular_weight * path_loss * sp_antenna_gain
    amplitude = dr.select(sp_valid, amplitude, mi.Float(0.0))

    sample_rate = self.render_config.sample_rate if hasattr(self.render_config, 'sample_rate') else self.sample_rate
    slope = self.render_config.chirp_slope if hasattr(self.render_config, 'chirp_slope') else self.slope
    fc = self.render_config.center_freq if hasattr(self.render_config, 'center_freq') else self.center_freq

    SPEED_OF_LIGHT = 299_792_458.0
    tau_delay = R_total / mi.Float(SPEED_OF_LIGHT)
    beat_freq = slope * tau_delay
    phase_offset = mi.Float(2.0 * np.pi) * fc * tau_delay

    # MIMO indexing
    flat_mimo_idx = chains.rx_idx * mi.UInt32(n_tx) + chains.tx_idx

    K_CHUNK = 64
    for k_start in range(0, K, K_CHUNK):
        k_end = min(k_start + K_CHUNK, K)
        for k_idx in range(k_start, k_end):
            t_k = mi.Float(k_idx) / mi.Float(sample_rate)
            phase = mi.Float(2.0 * np.pi) * beat_freq * t_k - phase_offset
            _cos = dr.cos(phase)
            _sin = dr.sin(phase)
            if not self.render_config.enable_grad_phase:
                _cos = dr.detach(_cos)
                _sin = dr.detach(_sin)
            re = amplitude * _cos
            im = amplitude * _sin

            adc_idx = flat_mimo_idx * mi.UInt32(K) + mi.UInt32(k_idx)
            dr.scatter_add(adc_real_flat, dr.select(sp_valid, re, mi.Float(0.0)), adc_idx)
            dr.scatter_add(adc_imag_flat, dr.select(sp_valid, im, mi.Float(0.0)), adc_idx)

        dr.eval(adc_real_flat, adc_imag_flat)

    if verbose:
        amp_np = np.array(dr.detach(amplitude))
        valid_amp = amp_np[np.array(sp_valid)]
        if len(valid_amp) > 0:
            print(f"    Amplitude range: [{valid_amp.min():.2e}, {valid_amp.max():.2e}]")
            print(f"    Mean amplitude: {valid_amp.mean():.2e}")
        print(f"    [E2E MB-Spec] Done: {n_valid:,} chains synthesized")

def _synthesize_specular_differentiable(
    self,
    cached_geom: 'CachedGeometry',
    physics_params: list,
    adc_real_flat: 'mi.Float',
    adc_imag_flat: 'mi.Float',
    n_tx: int, n_rx: int, K: int,
    pose_params: Optional[dict] = None,
    pattern_loaders: Optional[dict] = None,
    normal_params: Optional[list] = None,
    vertex_offset_params: Optional[list] = None,
):
    """
    Phase B specular path contribution with differentiable parameters.

    Supports gradient flow through:
    - Material params: via live BSDF re-evaluation (Fix 0: single cos)
    - Antenna patterns: via live pattern recomputation (Fix 2)
    - Radar pose: via live distance/direction recomputation (Fix 3)
    - Surface normals: via per-triangle normal gather (Fix 4B)
    - Vertex positions: via IFT constraint re-evaluation (Fix 5)
    """
    from ..utils.math import gather_point3f, gather_vector3f

    sp_valid = cached_geom.specular_valid
    n_spec = cached_geom.specular_n_paths

    if n_spec == 0:
        return

    # ====================================================================
    # Step S0: Gather live materials at specular hit triangles
    # ====================================================================
    sp_prim_ids = cached_geom.specular_prim_ids
    if sp_prim_ids is None:
        return

    sp_materials = [
        dr.gather(mi.Float, physics_params[i], sp_prim_ids, sp_valid)
        for i in range(len(physics_params))
    ]
    sp_eps_real, sp_eps_imag, sp_sigma_h, sp_l_c, sp_tau, sp_thickness = sp_materials

    # ====================================================================
    # Step S1: Recompute specular positions via IFT (Fix 5: vertex positions)
    # ====================================================================
    # Detach frozen specular geometry from stale AD graph.
    # SceneContext loads vertex positions/normals with grad enabled.
    # SMS solver (Phase A) computes specular paths using these AD-attached
    # arrays, so cached_geom.specular_* fields inherit stale AD connections.
    # If not detached, dr.backward() traverses the entire vertex AD graph → hangs.
    # Only physics_params (and optionally vertex_offset_params / normal_params)
    # should carry AD connections into this function.
    def _detach_p3f(p):
        return mi.Point3f(dr.detach(p.x), dr.detach(p.y), dr.detach(p.z))
    def _detach_v3f(v):
        return mi.Vector3f(dr.detach(v.x), dr.detach(v.y), dr.detach(v.z))

    sp_hit_P = _detach_p3f(cached_geom.specular_hit_P)  # Frozen positions (detached)

    ift_active = (vertex_offset_params is not None
                  and cached_geom.specular_grad_info is not None
                  and cached_geom.specular_bary_u is not None
                  and cached_geom.specular_vertex_ids_0 is not None)

    if ift_active:
        sp_hit_P, ift_dp_du, ift_dp_dv, ift_N = \
            self._recompute_specular_positions_ift(cached_geom, vertex_offset_params)
        # IFT gives us live normals from the moved triangle surface
        sp_hit_N = ift_N
    elif normal_params is not None:
        # Fix 4B: Per-vertex normal gather with barycentric interpolation
        nx_arr, ny_arr, nz_arr = normal_params
        # Use specular vertex IDs + barycentrics (same data cached for IFT)
        svi0 = cached_geom.specular_vertex_ids_0
        svi1 = cached_geom.specular_vertex_ids_1
        svi2 = cached_geom.specular_vertex_ids_2
        if svi0 is not None and cached_geom.specular_bary_u is not None:
            su = cached_geom.specular_bary_u
            sv = cached_geom.specular_bary_v
            sw = mi.Float(1.0) - su - sv
            # Gather per-vertex normals at each triangle corner
            sp_nx = sw * dr.gather(mi.Float, nx_arr, svi0, sp_valid) \
                  + su * dr.gather(mi.Float, nx_arr, svi1, sp_valid) \
                  + sv * dr.gather(mi.Float, nx_arr, svi2, sp_valid)
            sp_ny = sw * dr.gather(mi.Float, ny_arr, svi0, sp_valid) \
                  + su * dr.gather(mi.Float, ny_arr, svi1, sp_valid) \
                  + sv * dr.gather(mi.Float, ny_arr, svi2, sp_valid)
            sp_nz = sw * dr.gather(mi.Float, nz_arr, svi0, sp_valid) \
                  + su * dr.gather(mi.Float, nz_arr, svi1, sp_valid) \
                  + sv * dr.gather(mi.Float, nz_arr, svi2, sp_valid)
        else:
            # Fallback: per-triangle gather (legacy)
            sp_nx = dr.gather(mi.Float, nx_arr, sp_prim_ids, sp_valid)
            sp_ny = dr.gather(mi.Float, ny_arr, sp_prim_ids, sp_valid)
            sp_nz = dr.gather(mi.Float, nz_arr, sp_prim_ids, sp_valid)
        n_len = dr.maximum(
            dr.sqrt(dr.maximum(sp_nx * sp_nx + sp_ny * sp_ny + sp_nz * sp_nz,
                                mi.Float(1e-20))),
            mi.Float(1e-10))
        sp_hit_N = mi.Vector3f(sp_nx / n_len, sp_ny / n_len, sp_nz / n_len)
    else:
        sp_hit_N = _detach_v3f(cached_geom.specular_hit_N)

    # ====================================================================
    # Step S2: Recompute distances/directions from live positions/pose
    # ====================================================================
    # Determine live TX/RX positions (from pose if available)
    need_distance_recompute = (pose_params is not None or ift_active)
    if pose_params is not None and cached_geom.tx_positions is not None:
        from ..utils.transforms import transform_positions
        tx_pos_live = transform_positions(cached_geom.tx_positions, pose_params)
        rx_pos_live = transform_positions(cached_geom.rx_positions, pose_params)
    else:
        tx_pos_live = cached_geom.tx_positions
        rx_pos_live = cached_geom.rx_positions

    if need_distance_recompute and sp_hit_P is not None and tx_pos_live is not None:
        tx_pos_exp = gather_point3f(tx_pos_live, cached_geom.specular_tx_idx)
        rx_pos_exp = gather_point3f(rx_pos_live, cached_geom.specular_rx_idx)

        delta_tx = mi.Vector3f(
            tx_pos_exp.x - sp_hit_P.x,
            tx_pos_exp.y - sp_hit_P.y,
            tx_pos_exp.z - sp_hit_P.z)
        delta_rx = mi.Vector3f(
            rx_pos_exp.x - sp_hit_P.x,
            rx_pos_exp.y - sp_hit_P.y,
            rx_pos_exp.z - sp_hit_P.z)

        sp_d_tx = dr.norm(delta_tx)
        sp_d_rx = dr.norm(delta_rx)

        # Directions: from hit toward TX/RX
        sp_dir_to_tx = mi.Vector3f(
            delta_tx.x / dr.maximum(sp_d_tx, mi.Float(1e-10)),
            delta_tx.y / dr.maximum(sp_d_tx, mi.Float(1e-10)),
            delta_tx.z / dr.maximum(sp_d_tx, mi.Float(1e-10)))
        sp_dir_to_rx = mi.Vector3f(
            delta_rx.x / dr.maximum(sp_d_rx, mi.Float(1e-10)),
            delta_rx.y / dr.maximum(sp_d_rx, mi.Float(1e-10)),
            delta_rx.z / dr.maximum(sp_d_rx, mi.Float(1e-10)))
    else:
        sp_d_tx = mi.Float(dr.detach(cached_geom.specular_d_tx))
        sp_d_rx = mi.Float(dr.detach(cached_geom.specular_d_rx))
        sp_dir_to_tx = _detach_v3f(cached_geom.specular_dir_to_tx)
        sp_dir_to_rx = _detach_v3f(cached_geom.specular_dir_to_rx)

    # ====================================================================
    # Step S3: Evaluate KA-only BSDF with live materials (AD-attached)
    # ====================================================================
    # Use eval_ka_only_physics (KA lobe only = R_jones × η × τ_eff × f_KA)
    # to match the Sionna reference Phase B differentiable path.
    # This gives richer angular gradient information than the flat
    # reflectance coefficient eval_specular_reflectance.
    sp_cos_theta_r = dr.maximum(dr.dot(sp_dir_to_rx, sp_hit_N), mi.Float(0.0))
    sp_cos_theta_i_live = dr.maximum(dr.dot(sp_dir_to_tx, sp_hit_N), mi.Float(0.0))

    _eval_ka = getattr(self.bsdf, 'eval_ka_only_physics', None)
    if _eval_ka is not None:
        sp_brdf = _eval_ka(
            wo=sp_dir_to_rx,
            wi=sp_dir_to_tx,
            n=sp_hit_N,
            eps_real=sp_eps_real, eps_imag=sp_eps_imag,
            sigma_h=sp_sigma_h, l_c=sp_l_c, tau=sp_tau,
            thickness=sp_thickness,
        )
    else:
        # Fallback: full BSDF if KA-only not available
        _eval_f = getattr(self.bsdf, 'eval_f_physics', None)
        if _eval_f is not None:
            sp_brdf = _eval_f(
                wo=sp_dir_to_rx,
                wi=sp_dir_to_tx,
                n=sp_hit_N,
                eps_real=sp_eps_real, eps_imag=sp_eps_imag,
                sigma_h=sp_sigma_h, l_c=sp_l_c, tau=sp_tau,
                thickness=sp_thickness,
            )
        else:
            sp_brdf = getattr(cached_geom, 'specular_R_specular', mi.Float(0.5))

    # Detach frozen cos_theta_i and A_tri from stale AD
    sp_cos_theta_i_cached = mi.Float(dr.detach(cached_geom.specular_cos_theta_i))
    sp_A_tri_cached = mi.Float(dr.detach(cached_geom.specular_A_tri))
    specular_power = sp_brdf * sp_cos_theta_i_cached * sp_cos_theta_r * sp_A_tri_cached

    # ====================================================================
    # Step S4: Antenna gain (Fix 2: recompute from live patterns)
    # ====================================================================
    if pattern_loaders is not None and evaluate_combined_gain is not None:
        sp_antenna_gain = self._recompute_specular_antenna_gains(
            cached_geom, pattern_loaders, sp_dir_to_tx, sp_dir_to_rx)
        specular_power = specular_power * sp_antenna_gain
    elif cached_geom.specular_antenna_gain is not None:
        specular_power = specular_power * mi.Float(dr.detach(cached_geom.specular_antenna_gain))

    # ====================================================================
    # Step S5: Path loss and amplitude (uses live distances from Fix 3)
    # ====================================================================
    path_loss = mi.Float(1.0) / (sp_d_tx * sp_d_tx * sp_d_rx * sp_d_rx)

    if self.render_config.use_radar_equation:
        radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
        E_amp = radar_scale * dr.sqrt(dr.maximum(specular_power * path_loss, mi.Float(1e-20)))
    else:
        E_amp = dr.sqrt(dr.maximum(specular_power * path_loss, mi.Float(1e-20)))

    # ====================================================================
    # Steps S6-S7: Phase computation + scatter-add
    # Fast path: use pre-computed specular phasors when pose/geometry frozen
    # ====================================================================
    use_sp_cached = (
        cached_geom.sp_cached_cos_phi is not None
        and pose_params is None
        and vertex_offset_params is None
    )

    if use_sp_cached:
        # === FAST PATH: use pre-computed specular phasors ===
        E_amp_expanded = dr.gather(mi.Float, E_amp, cached_geom.sp_cached_path_idx)

        contrib_real = E_amp_expanded * cached_geom.sp_cached_cos_phi
        contrib_imag = E_amp_expanded * cached_geom.sp_cached_sin_phi

        dr.scatter_add(adc_real_flat, contrib_real,
                       cached_geom.sp_cached_flat_idx, cached_geom.sp_cached_valid_3d)
        dr.scatter_add(adc_imag_flat, contrib_imag,
                       cached_geom.sp_cached_flat_idx, cached_geom.sp_cached_valid_3d)
    else:
        # === STANDARD PATH: compute specular phase from scratch ===
        # K-chunked gather loop (same pattern as E2E path).
        # dr.tile()/dr.repeat() create AD graph structures that cause
        # dr.backward() to hang; dr.gather() in chunks works reliably.
        R_total_spec = sp_d_tx + sp_d_rx
        tau_spec = R_total_spec / mi.Float(C)

        TWO_PI = 2.0 * np.pi
        phi_const = mi.Float(TWO_PI * self.min_freq) * tau_spec
        phi_slope = mi.Float(TWO_PI * self.slope) * tau_spec

        CHUNK = getattr(self.render_config, 'e2e_phase_chunk_size', 64)

        for k_start in range(0, K, CHUNK):
            k_end = min(k_start + CHUNK, K)
            k_chunk = k_end - k_start
            n_elems = n_spec * k_chunk

            path_idx = dr.arange(mi.UInt32, n_elems) // mi.UInt32(k_chunk)
            k_local = dr.arange(mi.UInt32, n_elems) % mi.UInt32(k_chunk)
            k_global = k_local + mi.UInt32(k_start)

            pc = dr.gather(mi.Float, phi_const, path_idx)
            ps = dr.gather(mi.Float, phi_slope, path_idx)
            ea = dr.gather(mi.Float, E_amp, path_idx)
            act = dr.gather(mi.Bool, sp_valid, path_idx)

            t_k = mi.Float(k_global) / mi.Float(self.sample_rate)
            phi = pc + ps * t_k
            _cos = dr.cos(phi)
            _sin = dr.sin(phi)
            if not self.render_config.enable_grad_phase:
                _cos = dr.detach(_cos)
                _sin = dr.detach(_sin)
            contrib_real = ea * _cos
            contrib_imag = ea * _sin

            tx_for_elem = dr.gather(mi.UInt32, cached_geom.specular_tx_idx, path_idx)
            rx_for_elem = dr.gather(mi.UInt32, cached_geom.specular_rx_idx, path_idx)
            flat_adc_idx = (tx_for_elem * mi.UInt32(n_rx * K)
                            + rx_for_elem * mi.UInt32(K)
                            + k_global)

            dr.scatter_add(adc_real_flat, contrib_real, flat_adc_idx, act)
            dr.scatter_add(adc_imag_flat, contrib_imag, flat_adc_idx, act)

def _synthesize_specular_paths(
    self,
    specular_paths: SpecularPaths,
    adc_real_flat: 'mi.Float',
    adc_imag_flat: 'mi.Float',
    n_tx: int,
    n_rx: int,
    K: int,
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    rx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    antenna_gain_linear: bool = False,
    verbose: bool = True,
):
    """
    Synthesize ADC contributions from deterministic specular paths.

    These are added directly into the existing adc_real_flat/adc_imag_flat arrays
    via scatter_add.

    Specular path weight (from v2 plan):
      specular_power = R_specular × cos_theta_i × A_tri × G_tx × G_rx
      path_loss = 1/(d_tx² × d_rx²)
      E_amplitude_spec = radar_scale × sqrt(specular_power × path_loss)

    Args:
        specular_paths: SpecularPaths from image method
        adc_real_flat/imag_flat: Existing ADC arrays to accumulate into [n_tx*n_rx*K]
        n_tx, n_rx, K: ADC dimensions
        tx/rx_boresights: For antenna pattern evaluation
        tx/rx_pattern_rrts: Antenna patterns
        antenna_gain_linear: Whether to convert dB to linear
        verbose: Print progress
    """
    from ..utils.math import gather_vector3f

    sp = specular_paths
    valid = sp.valid
    n_paths = sp.n_paths

    n_valid_spec = int(dr.sum(mi.UInt32(valid))[0]) if hasattr(dr.sum(mi.UInt32(valid)), '__getitem__') else int(dr.sum(mi.UInt32(valid)))

    if verbose:
        print(f"\n  [Specular Branch] Synthesizing {n_valid_spec:,} valid specular paths")

    if n_valid_spec == 0:
        return None  # Return None antenna gain

    # ====================================================================
    # Step S1: Compute specular power
    # ====================================================================
    # For deterministic specular paths, use integrated reflectance
    # R_specular = η × τ_eff × A (dimensionless), NOT the BSDF density.
    # This matches the reference renderer's Phase A which uses sp.R_specular.
    _eval_sr = getattr(self.bsdf, 'eval_specular_reflectance', None)
    if (_eval_sr is not None
            and self.triangle_materials is not None
            and self.triangle_materials.shape[1] >= 6
            and sp.prim_ids is not None):
        # Gather per-triangle materials at specular path locations
        prim_np = np.array(sp.prim_ids).astype(np.int64)
        prim_np = np.clip(prim_np, 0, self.triangle_materials.shape[0] - 1)
        mat = self.triangle_materials[prim_np]
        _sp_eps_r = mi.Float(mat[:, 0].astype(np.float32))
        _sp_eps_i = mi.Float(mat[:, 1].astype(np.float32))
        _sp_sh = mi.Float(mat[:, 2].astype(np.float32))
        _sp_lc = mi.Float(mat[:, 3].astype(np.float32))
        _sp_tau = mi.Float(mat[:, 4].astype(np.float32))
        _sp_thick = mi.Float(mat[:, 5].astype(np.float32))
        sp_reflectance = _eval_sr(
            cos_theta_i=sp.cos_theta_i,
            eps_real=_sp_eps_r, eps_imag=_sp_eps_i,
            sigma_h=_sp_sh, l_c=_sp_lc, tau=_sp_tau,
            thickness=_sp_thick,
        )
        specular_power = sp_reflectance * sp.cos_theta_i * sp.cos_theta_r * sp.A_tri
    else:
        # Fallback: use pre-computed R_specular (eta * tau * A, no angular shape)
        specular_power = sp.R_specular * sp.cos_theta_i * sp.cos_theta_r * sp.A_tri

    # ====================================================================
    # Step S2: Antenna pattern evaluation (if available)
    # ====================================================================
    antenna_gain = mi.Float(1.0)
    pattern_mode = getattr(self.render_config, 'pattern_mode', 'fixed')

    # GPU-native path
    _use_gpu_spec = (
        pattern_mode == 'fixed'
        and evaluate_combined_gain is not None
        and (tx_pattern_loader is not None or rx_pattern_loader is not None)
    )

    if _use_gpu_spec:
        # Antenna direction: from antenna to specular point = -(dir_to_tx/rx)
        dir_tx_to_spec = -sp.dir_to_tx
        dir_rx_to_spec = -sp.dir_to_rx

        # Gather per-path boresights
        tx_boresights_exp = gather_vector3f(tx_boresights, sp.tx_idx) if tx_boresights is not None else None
        rx_boresights_exp = gather_vector3f(rx_boresights, sp.rx_idx) if rx_boresights is not None else None

        if tx_pattern_loader is not None:
            gain_tx_lin = evaluate_combined_gain(tx_pattern_loader, dir_tx_to_spec, tx_boresights_exp)
        else:
            gain_tx_lin = mi.Float(1.0)

        if rx_pattern_loader is not None:
            gain_rx_lin = evaluate_combined_gain(rx_pattern_loader, dir_rx_to_spec, rx_boresights_exp)
        else:
            gain_rx_lin = mi.Float(1.0)

        antenna_gain = gain_tx_lin * gain_rx_lin
        specular_power = specular_power * antenna_gain

    elif tx_pattern_rrts is not None or rx_pattern_rrts is not None:
        # DrJit GPU fallback: all-GPU evaluation (no CPU↔GPU transfers)
        dir_tx_to_spec = mi.Vector3f(-sp.dir_to_tx.x, -sp.dir_to_tx.y, -sp.dir_to_tx.z)
        dir_rx_to_spec = mi.Vector3f(-sp.dir_to_rx.x, -sp.dir_to_rx.y, -sp.dir_to_rx.z)

        if pattern_mode == 'legacy':
            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_tx_dB = evaluate_gain_rrts_style_drjit(dir_tx_to_spec, tx_pattern_rrts)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_paths)
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_rx_dB = evaluate_gain_rrts_style_drjit(dir_rx_to_spec, rx_pattern_rrts)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_paths)
        else:
            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                tx_boresights_exp = gather_vector3f(tx_boresights, sp.tx_idx) \
                    if tx_boresights is not None else mi.Vector3f(0.0, 1.0, 0.0)
                gain_tx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_tx_to_spec, tx_pattern_rrts, tx_boresights_exp)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_paths)
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                rx_boresights_exp = gather_vector3f(rx_boresights, sp.rx_idx) \
                    if rx_boresights is not None else mi.Vector3f(0.0, 1.0, 0.0)
                gain_rx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_rx_to_spec, rx_pattern_rrts, rx_boresights_exp)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_paths)

        if antenna_gain_linear or self.render_config.use_radar_equation:
            gain_tx_lin = dr.power(mi.Float(10.0), gain_tx_dB * mi.Float(0.1))
            gain_rx_lin = dr.power(mi.Float(10.0), gain_rx_dB * mi.Float(0.1))
        else:
            gain_tx_lin = gain_tx_dB
            gain_rx_lin = gain_rx_dB

        antenna_gain = gain_tx_lin * gain_rx_lin
        specular_power = specular_power * antenna_gain

    # ====================================================================
    # Step S3: Compute amplitude
    # ====================================================================
    # Path loss: 1/(d_tx² × d_rx²) — both distances appear explicitly
    # because specular branch is deterministic (no cosine hemisphere sampling)
    path_loss = mi.Float(1.0) / (sp.d_tx * sp.d_tx * sp.d_rx * sp.d_rx)

    if self.render_config.use_radar_equation:
        radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
        E_amplitude_spec = radar_scale * dr.sqrt(
            dr.maximum(specular_power * path_loss, mi.Float(0.0))
        )
    else:
        E_amplitude_spec = dr.sqrt(
            dr.maximum(specular_power * path_loss, mi.Float(0.0))
        )

    # ====================================================================
    # Step S4: Phase computation and scatter-add
    # ====================================================================
    R_total_spec = sp.d_tx + sp.d_rx
    tau_spec = R_total_spec / mi.Float(C)

    TWO_PI = 2.0 * np.pi
    phi_const_spec = mi.Float(TWO_PI * self.min_freq) * tau_spec
    phi_slope_spec = mi.Float(TWO_PI * self.slope) * tau_spec

    # Time grid
    t_grid_dr = self._t_grid_dr  # [K] (cached on GPU)

    # Expand to [n_paths × K]
    phi_const_3d = dr.tile(phi_const_spec, K)
    phi_slope_3d = dr.tile(phi_slope_spec, K)
    t_k = dr.repeat(t_grid_dr, n_paths)
    weight_3d = dr.tile(E_amplitude_spec, K)
    active_3d = dr.tile(valid, K)

    # Phase
    phi = phi_const_3d + phi_slope_3d * t_k
    cos_phi = dr.cos(phi)
    sin_phi = dr.sin(phi)
    if not self.render_config.enable_grad_phase:
        cos_phi = dr.detach(cos_phi)
        sin_phi = dr.detach(sin_phi)

    contrib_real = weight_3d * cos_phi
    contrib_imag = weight_3d * sin_phi

    # Flat indices
    tx_idx_3d = dr.tile(sp.tx_idx, K)
    rx_idx_3d = dr.tile(sp.rx_idx, K)
    base_idx = dr.arange(mi.UInt32, n_paths * K)
    k_idx = base_idx // mi.UInt32(n_paths)

    flat_idx = tx_idx_3d * mi.UInt32(n_rx * K) + rx_idx_3d * mi.UInt32(K) + k_idx

    # Accumulate into existing ADC arrays
    dr.scatter_add(adc_real_flat, contrib_real, flat_idx, active_3d)
    dr.scatter_add(adc_imag_flat, contrib_imag, flat_idx, active_3d)

    if verbose:
        if n_valid_spec > 0:
            dr.eval(E_amplitude_spec, R_total_spec, specular_power, path_loss)
            E_np = np.array(E_amplitude_spec)
            valid_E = E_np[np.array(valid)]
            print(f"  Specular E_amplitude range: [{valid_E.min():.6e}, {valid_E.max():.6e}]")
            print(f"  Specular E_amplitude mean: {valid_E.mean():.6e}")
            _Rt_fwd = np.array(R_total_spec)[np.array(valid)]
            print(f"  Specular R_total range: [{_Rt_fwd.min():.4f}, {_Rt_fwd.max():.4f}], mean={_Rt_fwd.mean():.4f}, std={_Rt_fwd.std():.4f}")
            _sp_fwd = np.array(specular_power)[np.array(valid)]
            print(f"  Specular specular_power range: [{_sp_fwd.min():.6e}, {_sp_fwd.max():.6e}], mean={_sp_fwd.mean():.6e}")
            _pl_fwd = np.array(path_loss)[np.array(valid)]
            print(f"  Specular path_loss range: [{_pl_fwd.min():.6e}, {_pl_fwd.max():.6e}], mean={_pl_fwd.mean():.6e}")
            _Atri_fwd = np.array(sp.A_tri)[np.array(valid)]
            print(f"  Specular A_tri range: [{_Atri_fwd.min():.6e}, {_Atri_fwd.max():.6e}], mean={_Atri_fwd.mean():.6e}")
            _txidx_fwd = np.array(sp.tx_idx)[np.array(valid)]
            _rxidx_fwd = np.array(sp.rx_idx)[np.array(valid)]
            print(f"  Specular TX idx range: [{_txidx_fwd.min()}, {_txidx_fwd.max()}], unique={len(np.unique(_txidx_fwd))}")
            print(f"  Specular RX idx range: [{_rxidx_fwd.min()}, {_rxidx_fwd.max()}], unique={len(np.unique(_rxidx_fwd))}")
            print(f"  Specular sum(E_amp^2)={(_Rt_fwd*0 + valid_E**2).sum():.6e}")
        print(f"  Specular ADC synthesis complete")

    return antenna_gain

