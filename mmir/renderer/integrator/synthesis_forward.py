"""
Forward render synthesis pipeline (Phase A).

Contains the primary vectorized Monte Carlo + deterministic specular pipeline:
- synthesize_vectorized (deprecated legacy)
- synthesize_with_patterns (wrapper)
- synthesize_shared_hits_vectorized (main pipeline)
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


def synthesize_vectorized(
    self,
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    reservoir_hits: ReservoirHits,
    scene: 'mi.Scene',
    verbose: bool = True
) -> ADCResult:
    """
    Vectorized version of synthesize().

    Each hit is only processed with its OWNING RX element, but connects to ALL TX elements.
    This differs from synthesize_shared_hits_vectorized() where each hit is processed
    with ALL (TX, RX) pairs.

    Vectorizes over:
    - TX elements (each hit connects to all TXs)
    - K samples (phase computation)
    Chunks over hits to avoid GPU OOM.

    Key difference from synthesize_shared_hits_vectorized:
    - synthesize_shared_hits_vectorized: n_valid → n_valid × n_tx × n_rx
    - synthesize_vectorized:             n_valid → n_valid × n_tx (owning RX only)

    Args:
        tx_positions: TX element positions [n_tx]
        rx_positions: RX element positions [n_rx]
        reservoir_hits: Reservoir sampling results
        scene: Mitsuba scene for visibility checks
        verbose: Print progress

    Returns:
        ADCResult with synthesized ADC signals
    """
    # Import helper functions from this module (defined for synthesize_shared_hits_vectorized)
    from ..utils.math import (
        expand_point3f_for_mimo,
        expand_vector3f_for_mimo,
        generate_mimo_indices,
        gather_point3f,
        gather_vector3f,
    )

    n_tx = dr.width(tx_positions)
    n_rx = reservoir_hits.n_rx
    K = self.num_samples

    if verbose:
        print(f"\n[SBRIntegratorRef] Synthesizing ADC signals (VECTORIZED - RX-owned hits)")
        print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
        print(f"  ADC samples: {K}")
        print(f"  f0 = {self.min_freq/1e9:.3f} GHz")
        print(f"  S = {self.slope/1e12:.3f} THz/s")

    # ====================================================================
    # Step 1: Extract all valid hits with their owning RX indices
    # ====================================================================
    n_valid = int(dr.sum(mi.UInt32(reservoir_hits.valid))[0])

    if verbose:
        print(f"  Total valid hits: {n_valid}")

    if n_valid == 0:
        return ADCResult(
            adc_real=np.zeros((n_tx, n_rx, K), dtype=np.float32),
            adc_imag=np.zeros((n_tx, n_rx, K), dtype=np.float32),
            n_tx=n_tx,
            n_rx=n_rx,
            n_samples=K,
            total_paths=0
        )

    # Get valid hit indices (compressed on GPU, then to numpy for remaining indexing)
    valid_indices_dr = dr.compress(reservoir_hits.valid)
    valid_indices_np = np.array(valid_indices_dr)

    # Compute owning RX for each valid hit
    # hit_slot_idx // n_hits_per_rx = rx_idx
    owning_rx = valid_indices_np // reservoir_hits.n_hits_per_rx  # [n_valid]

    # Gather valid hit data
    valid_P = gather_point3f(reservoir_hits.hit_P, valid_indices_dr)  # [n_valid]
    valid_N = gather_vector3f(reservoir_hits.hit_N, valid_indices_dr)  # [n_valid]
    valid_rho = reservoir_hits.hit_rho[valid_indices_np]  # [n_valid, 3 or 4] numpy

    # ====================================================================
    # Step 2: TX Expansion - [n_valid] -> [n_valid × n_tx]
    # Each hit connects to all TXs, but only its owning RX
    # ====================================================================
    n_total = n_valid * n_tx

    if verbose:
        print(f"  TX expansion: {n_valid} hits × {n_tx} TX = {n_total:,} paths")

    # Expand hit positions and normals for TX dimension
    # For each hit i, we create n_tx copies (one for each TX)
    hit_P_expanded = mi.Point3f(
        dr.repeat(valid_P.x, n_tx),
        dr.repeat(valid_P.y, n_tx),
        dr.repeat(valid_P.z, n_tx)
    )  # [n_total]

    hit_N_expanded = mi.Vector3f(
        dr.repeat(valid_N.x, n_tx),
        dr.repeat(valid_N.y, n_tx),
        dr.repeat(valid_N.z, n_tx)
    )  # [n_total]

    # Expand material properties
    physics_mode = valid_rho.shape[1] == 6

    if physics_mode:
        rho_cols_expanded = [
            mi.Float(np.repeat(valid_rho[:, c], n_tx)) for c in range(6)
        ]
        rho_eps_real = rho_cols_expanded[0]
        rho_eps_imag = rho_cols_expanded[1]
        rho_sigma_h = rho_cols_expanded[2]
        rho_l_c = rho_cols_expanded[3]
        rho_tau = rho_cols_expanded[4]
        rho_thickness = rho_cols_expanded[5]
        rho_x_expanded = None
        rho_y_expanded = None
        rho_z_expanded = None
    else:
        rho_x_expanded = mi.Float(np.repeat(valid_rho[:, 0], n_tx))
        rho_y_expanded = mi.Float(np.repeat(valid_rho[:, 1], n_tx))
        rho_z_expanded = mi.Float(np.repeat(valid_rho[:, 2], n_tx))

    # Expand owning RX indices: each hit repeats its owning RX n_tx times
    rx_idx_expanded = mi.UInt32(np.repeat(owning_rx, n_tx))  # [n_total]

    # TX indices: for each hit, cycle through 0, 1, ..., n_tx-1
    tx_idx_expanded = mi.UInt32(np.tile(np.arange(n_tx), n_valid))  # [n_total]

    # Gather TX and RX positions for each path
    tx_pos_expanded = gather_point3f(tx_positions, tx_idx_expanded)  # [n_total]
    rx_pos_expanded = gather_point3f(rx_positions, rx_idx_expanded)  # [n_total]

    # ====================================================================
    # Step 3: Vectorized Geometry Computation
    # ====================================================================
    # Distance from RX to hit
    delta_rx = hit_P_expanded - rx_pos_expanded
    d_rx_to_hit = dr.norm(delta_rx)  # [n_total]

    # Distance from hit to TX
    delta_tx = tx_pos_expanded - hit_P_expanded
    d_hit_to_tx = dr.norm(delta_tx)  # [n_total]

    # Direction from hit to RX (wo - outgoing)
    dir_hit_to_rx = mi.Vector3f(
        -delta_rx.x / dr.maximum(d_rx_to_hit, 1e-10),
        -delta_rx.y / dr.maximum(d_rx_to_hit, 1e-10),
        -delta_rx.z / dr.maximum(d_rx_to_hit, 1e-10)
    )

    # Direction from hit to TX (wi - incident/light direction)
    dir_hit_to_tx = mi.Vector3f(
        delta_tx.x / dr.maximum(d_hit_to_tx, 1e-10),
        delta_tx.y / dr.maximum(d_hit_to_tx, 1e-10),
        delta_tx.z / dr.maximum(d_hit_to_tx, 1e-10)
    )

    # ====================================================================
    # Step 4: Vectorized Visibility Check (SINGLE CALL)
    # ====================================================================
    epsilon = 1e-4
    shadow_origins = mi.Point3f(
        hit_P_expanded.x + epsilon * dir_hit_to_tx.x,
        hit_P_expanded.y + epsilon * dir_hit_to_tx.y,
        hit_P_expanded.z + epsilon * dir_hit_to_tx.z
    )

    shadow_rays = mi.Ray3f(shadow_origins, dir_hit_to_tx)
    shadow_rays.maxt = d_hit_to_tx - 2 * epsilon

    # Single visibility check for ALL paths
    occluded = scene.ray_test(shadow_rays)
    visible = ~occluded  # [n_total]

    n_visible = int(dr.sum(mi.UInt32(visible))[0])
    if verbose:
        print(f"  Visible paths: {n_visible:,} / {n_total:,}")

    # ====================================================================
    # Step 5: Vectorized BRDF Computation
    # ====================================================================
    if physics_mode and hasattr(self.bsdf, 'eval_f_cos_physics'):
        brdf_weight = self.bsdf.eval_f_cos_physics(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N_expanded,
            eps_real=rho_eps_real, eps_imag=rho_eps_imag,
            sigma_h=rho_sigma_h, l_c=rho_l_c, tau=rho_tau,
            thickness=rho_thickness,
        )
    else:
        brdf_weight = self.bsdf.eval_f_cos(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N_expanded,
            albedo=rho_x_expanded, roughness=rho_y_expanded,
            metallic=rho_z_expanded,
        )

    # Zero out occluded paths
    brdf_weight = dr.select(visible, brdf_weight, mi.Float(0.0))

    # ====================================================================
    # Step 6: Compute path weights
    # ====================================================================
    # Total path length
    R_total = d_rx_to_hit + d_hit_to_tx  # [n_total]

    # Round-trip delay
    tau = R_total / C  # [n_total]

    # Weight = BRDF weight (no radar equation for simple synthesize)
    weight = brdf_weight  # [n_total]

    # ====================================================================
    # Step 7: Phase computation and scatter-add (vectorized over K)
    # ====================================================================
    # Create time grid as DrJit array
    t_grid_dr = self._t_grid_dr  # [K] (cached on GPU)

    # Phase constants for each path
    phi_const = 2.0 * np.pi * self.min_freq * tau  # [n_total]
    phi_slope = 2.0 * np.pi * self.slope * tau      # [n_total]

    # Expand for K: [n_total] -> [n_total × K]
    # Use dr.repeat to replicate each path K times, then add phase per sample
    phi_const_3d = dr.repeat(phi_const, K)   # [n_total × K]
    phi_slope_3d = dr.repeat(phi_slope, K)   # [n_total × K]
    t_k = dr.tile(t_grid_dr, n_total)        # [n_total × K]
    weight_3d = dr.repeat(weight, K)         # [n_total × K]
    active_3d = dr.repeat(visible, K)        # [n_total × K]

    # Compute phase for ALL paths × ALL samples at once
    phi = phi_const_3d + phi_slope_3d * t_k  # [n_total × K]

    # Complex phasor
    cos_phi = dr.cos(phi)
    sin_phi = dr.sin(phi)
    contrib_real = weight_3d * cos_phi
    contrib_imag = weight_3d * sin_phi

    # ====================================================================
    # Step 8: Compute flat indices for scatter_add
    # ====================================================================
    tx_idx_3d = dr.repeat(tx_idx_expanded, K)  # [n_total × K]
    rx_idx_3d = dr.repeat(rx_idx_expanded, K)  # [n_total × K]

    # k_idx: for dr.repeat layout, each path gets K consecutive indices
    # path i gets indices [i*K, i*K+1, ..., i*K+(K-1)]
    base_idx = dr.arange(mi.UInt32, n_total * K)
    k_idx = base_idx % mi.UInt32(K)

    # Flat index into [n_tx, n_rx, K] array
    flat_idx = tx_idx_3d * mi.UInt32(n_rx * K) + rx_idx_3d * mi.UInt32(K) + k_idx

    # ====================================================================
    # Step 9: Atomic scatter-add accumulation
    # ====================================================================
    adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
    adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

    dr.scatter_add(adc_real_flat, contrib_real, flat_idx, active_3d)
    dr.scatter_add(adc_imag_flat, contrib_imag, flat_idx, active_3d)

    # Reshape to [n_tx, n_rx, K]
    adc_real = np.array(adc_real_flat).reshape(n_tx, n_rx, K)
    adc_imag = np.array(adc_imag_flat).reshape(n_tx, n_rx, K)

    if verbose:
        print(f"  ADC synthesis complete")

    return ADCResult(
        adc_real=adc_real.astype(np.float32),
        adc_imag=adc_imag.astype(np.float32),
        n_tx=n_tx,
        n_rx=n_rx,
        n_samples=K,
        total_paths=n_visible
    )

# =========================================================================
# REMOVED: synthesize_shared_hits()
# This was a loop-based implementation that is 135-146x slower than
# synthesize_shared_hits_vectorized(). Use synthesize_shared_hits_vectorized() instead.
# =========================================================================

def synthesize_with_patterns(
    self,
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    tx_boresights: 'mi.Vector3f',
    rx_boresights: 'mi.Vector3f',
    reservoir_hits: ReservoirHits,
    scene: 'mi.Scene',
    tx_pattern: Optional[callable] = None,
    rx_pattern: Optional[callable] = None,
    verbose: bool = True
) -> ADCResult:
    """
    Synthesize ADC signals with antenna patterns.

    Extended version that applies TX and RX antenna patterns.

    Args:
        tx_positions: TX element positions
        rx_positions: RX element positions
        tx_boresights: TX boresight directions
        rx_boresights: RX boresight directions
        reservoir_hits: Reservoir sampling results
        scene: Mitsuba scene
        tx_pattern: TX antenna pattern function (direction -> gain)
        rx_pattern: RX antenna pattern function (direction -> gain)
        verbose: Print progress

    Returns:
        ADCResult with synthesized ADC signals
    """
    # Delegate to vectorized synthesize
    # Pattern application can be added later
    return self.synthesize_vectorized(
        tx_positions, rx_positions, reservoir_hits, scene, verbose
    )

def synthesize_shared_hits_vectorized(
    self,
    tx_positions: 'mi.Point3f',
    rx_positions: 'mi.Point3f',
    tx_boresights: Optional['mi.Vector3f'] = None,
    rx_boresights: Optional['mi.Vector3f'] = None,
    reservoir_hits: ReservoirHits = None,
    scene: 'mi.Scene' = None,
    tx_pattern_rrts: Optional[np.ndarray] = None,
    rx_pattern_rrts: Optional[np.ndarray] = None,
    tx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    rx_pattern_loader: Optional['AntennaPatternLoader'] = None,
    antenna_gain_linear: bool = False,
    verbose: bool = True,
    return_components: bool = False,
    use_non_ka_bsdf: bool = False,
    specular_paths: Optional[SpecularPaths] = None,
    patch_has_specular: Optional['mi.Bool'] = None,
    tri_to_patch: Optional['mi.Int32'] = None,
    return_cached_geometry: bool = False,
) -> ADCResult:
    """
    FULLY VECTORIZED ADC synthesis with radar equation support - NO PYTHON LOOPS.

    This is a vectorized version of synthesize_shared_hits() that produces
    identical output but uses DrJit MIMO expansion and scatter_add for
    massive speedup (~50-75x faster).

    Key features preserved from looped version:
    - Full radar equation when use_radar_equation=True
    - GGX specular + Lambertian diffuse BRDF
    - RRTS-style antenna pattern evaluation with orientation support

    Args:
        tx_positions: TX element positions [n_tx]
        rx_positions: RX element positions [n_rx]
        tx_boresights: TX boresight directions [n_tx] for antenna pattern evaluation
        rx_boresights: RX boresight directions [n_rx] for antenna pattern evaluation + sampling PDF
        reservoir_hits: Reservoir sampling results
        scene: Mitsuba scene for visibility checks
        tx_pattern_rrts: TX antenna pattern in RRTS format [361, 2, 5] (dB)
        rx_pattern_rrts: RX antenna pattern in RRTS format [361, 2, 5] (dB)
        tx_pattern_loader: GPU-native TX antenna pattern (AntennaPatternLoader)
        rx_pattern_loader: GPU-native RX antenna pattern (AntennaPatternLoader)
        antenna_gain_linear: If True, convert dB to linear scale
        verbose: Print progress

    Returns:
        ADCResult with synthesized ADC signals (identical to looped version)
    """
    from ..utils.math import (
        expand_for_mimo,
        expand_point3f_for_mimo,
        expand_vector3f_for_mimo,
        generate_mimo_indices,
        gather_point3f,
        gather_vector3f,
    )

    n_tx = dr.width(tx_positions)
    n_rx = reservoir_hits.n_rx
    K = self.num_samples

    if verbose:
        print(f"\n[SBRIntegratorRef] Synthesizing ADC signals (FULLY VECTORIZED - NO LOOPS)")
        print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
        print(f"  ADC samples: {K}")
        print(f"  f0 = {self.min_freq/1e9:.3f} GHz")
        print(f"  S = {self.slope/1e12:.3f} THz/s")
        print(f"  λ = {self.wavelength*1000:.2f} mm")
        gain_scale_str = "linear" if antenna_gain_linear else "logarithmic dB"
        print(f"  Antenna gain scale: {gain_scale_str}")
        if self.render_config.use_radar_equation:
            print(f"  Radar equation: ENABLED")
            print(f"    Radar constant = {self.radar_constant:.6e}")
            print(f"    Total scale factor = {self.radar_constant * self.rx_dBFS_scale * self.adc_scale:.6e}")
        else:
            print(f"  Radar equation: DISABLED (legacy BRDF-only mode)")

    # ====================================================================
    # Step 1: Extract and flatten all valid hits
    # ====================================================================
    # Get all hit data
    hit_P_all = reservoir_hits.hit_P
    hit_N_all = reservoir_hits.hit_N
    hit_rho_all = reservoir_hits.hit_rho  # [n_total_slots, 3] numpy array

    # Count valid hits on GPU (avoids full boolean transfer)
    n_valid = int(dr.sum(mi.UInt32(reservoir_hits.valid))[0])

    if verbose:
        print(f"  Total valid hits: {n_valid}")

    if n_valid == 0:
        empty_result = ADCResult(
            adc_real=np.zeros((n_tx, n_rx, K), dtype=np.float32),
            adc_imag=np.zeros((n_tx, n_rx, K), dtype=np.float32),
            n_tx=n_tx,
            n_rx=n_rx,
            n_samples=K,
            total_paths=0
        )
        if return_cached_geometry:
            return empty_result, None
        return empty_result

    # Get valid hit indices (compressed on GPU, then to numpy for remaining indexing)
    valid_indices_dr = dr.compress(reservoir_hits.valid)
    valid_indices_np = np.array(valid_indices_dr)

    # Gather valid hits into DrJit arrays
    valid_P = gather_point3f(hit_P_all, valid_indices_dr)  # [n_valid]
    valid_N = gather_vector3f(hit_N_all, valid_indices_dr)  # [n_valid]
    valid_rho = hit_rho_all[valid_indices_np]  # [n_valid, 3] numpy

    # Extract primitive IDs for differentiable re-synthesis (dr.gather on materials)
    valid_prim_ids = None
    if return_cached_geometry and reservoir_hits.hit_ID is not None:
        valid_prim_ids = dr.gather(mi.Int32, reservoir_hits.hit_ID, valid_indices_dr)
        valid_prim_ids = mi.UInt32(dr.maximum(valid_prim_ids, mi.Int32(0)))  # Clamp negatives

    # Extract barycentric coordinates + vertex indices for differentiable
    # normal/vertex recomputation in Phase B (needed regardless of material mode)
    valid_bary_u = None
    valid_bary_v = None
    valid_vertex_ids_0 = valid_vertex_ids_1 = valid_vertex_ids_2 = None
    if (return_cached_geometry
            and reservoir_hits.hit_bary_u is not None):
        valid_bary_u = dr.gather(mi.Float, reservoir_hits.hit_bary_u, valid_indices_dr)
        valid_bary_v = dr.gather(mi.Float, reservoir_hits.hit_bary_v, valid_indices_dr)
        # Get vertex indices from mesh face connectivity
        mesh = scene.shapes()[0] if len(scene.shapes()) > 0 else None
        if mesh is not None and valid_prim_ids is not None:
            face_idx = mesh.face_indices(valid_prim_ids)
            valid_vertex_ids_0 = face_idx[0]  # UInt32 [n_valid]
            valid_vertex_ids_1 = face_idx[1]
            valid_vertex_ids_2 = face_idx[2]

    # Extract PDF and n_attempted for MC probability chain (Option B normalization)
    # hit_pdf: [n_total_slots] -> [n_valid] (PDF for each valid hit)
    # n_attempted_per_rx: [n_rx] -> need to map to [n_valid] based on which RX owns each hit
    valid_pdf = None
    valid_n_attempted = None
    if reservoir_hits.hit_pdf is not None and reservoir_hits.n_attempted_per_rx is not None:
        valid_pdf = reservoir_hits.hit_pdf[valid_indices_np]  # [n_valid] numpy

        # Map each valid hit to its owning RX to get n_attempted
        # The flat index i maps to rx_idx = i // n_hits_per_rx
        valid_hit_slot_indices = valid_indices_np  # Original slot indices for valid hits
        valid_rx_indices = valid_hit_slot_indices // reservoir_hits.n_hits_per_rx
        valid_n_attempted = reservoir_hits.n_attempted_per_rx[valid_rx_indices]  # [n_valid]

        if verbose:
            print(f"  MC probability chain: PDF range [{valid_pdf.min():.4f}, {valid_pdf.max():.4f}]")
            print(f"  MC probability chain: n_attempted range [{valid_n_attempted.min()}, {valid_n_attempted.max()}]")

    # ====================================================================
    # Step 2: MIMO Expansion - [n_valid] -> [n_valid × n_tx × n_rx]
    # ====================================================================
    n_total = n_valid * n_tx * n_rx

    if verbose:
        print(f"  MIMO expansion: {n_valid} hits × {n_tx} TX × {n_rx} RX = {n_total:,} paths")

    # Expand hit positions and normals
    hit_P_expanded = expand_point3f_for_mimo(valid_P, n_tx, n_rx)  # [n_total]
    hit_N_expanded = expand_vector3f_for_mimo(valid_N, n_tx, n_rx)  # [n_total]

    # Expand material properties
    # Detect physics mode from column count
    physics_mode = valid_rho.shape[1] == 6

    # Lazy-upload materials to GPU for dr.gather-based expansion
    if self.triangle_materials is not None:
        if (self._gpu_material_cols is None
                or id(self.triangle_materials) != self._gpu_material_src_id):
            self._gpu_material_cols = [
                mi.Float(self.triangle_materials[:, c].astype(np.float32))
                for c in range(self.triangle_materials.shape[1])
            ]
            self._gpu_material_src_id = id(self.triangle_materials)

    # Try GPU-resident material expansion via dr.gather (avoids numpy->DrJit transfer)
    _n_mimo = n_tx * n_rx
    _use_gpu_mats = (
        self._gpu_material_cols is not None
        and reservoir_hits.hit_ID is not None
        and len(self._gpu_material_cols) >= (6 if physics_mode else 3)
    )

    if _use_gpu_mats:
        # Get per-hit prim_ids (reuse if already computed for cached geometry)
        if valid_prim_ids is None:
            _vpi = dr.gather(mi.Int32, reservoir_hits.hit_ID, valid_indices_dr)
            _vpi = mi.UInt32(dr.maximum(_vpi, mi.Int32(0)))
        else:
            _vpi = valid_prim_ids
        _n_tris_gpu = dr.width(self._gpu_material_cols[0])
        _vpi = dr.minimum(_vpi, mi.UInt32(_n_tris_gpu - 1))

        if physics_mode:
            rho_cols_expanded = [
                dr.repeat(dr.gather(mi.Float, self._gpu_material_cols[c], _vpi), _n_mimo)
                for c in range(6)
            ]
            rho_eps_real = rho_cols_expanded[0]
            rho_eps_imag = rho_cols_expanded[1]
            rho_sigma_h = rho_cols_expanded[2]
            rho_l_c = rho_cols_expanded[3]
            rho_tau = rho_cols_expanded[4]
            rho_thickness = rho_cols_expanded[5]
            rho_x_expanded = None
            rho_y_expanded = None
            rho_z_expanded = None
        else:
            rho_x_expanded = dr.repeat(dr.gather(mi.Float, self._gpu_material_cols[0], _vpi), _n_mimo)
            rho_y_expanded = dr.repeat(dr.gather(mi.Float, self._gpu_material_cols[1], _vpi), _n_mimo)
            rho_z_expanded = dr.repeat(dr.gather(mi.Float, self._gpu_material_cols[2], _vpi), _n_mimo)
    else:
        # Fallback: numpy -> DrJit path
        if physics_mode:
            rho_cols_expanded = [
                mi.Float(np.repeat(valid_rho[:, c], n_tx * n_rx)) for c in range(6)
            ]
            rho_eps_real = rho_cols_expanded[0]
            rho_eps_imag = rho_cols_expanded[1]
            rho_sigma_h = rho_cols_expanded[2]
            rho_l_c = rho_cols_expanded[3]
            rho_tau = rho_cols_expanded[4]
            rho_thickness = rho_cols_expanded[5]
            rho_x_expanded = None
            rho_y_expanded = None
            rho_z_expanded = None
        else:
            rho_x_expanded = mi.Float(np.repeat(valid_rho[:, 0], n_tx * n_rx))
            rho_y_expanded = mi.Float(np.repeat(valid_rho[:, 1], n_tx * n_rx))
            rho_z_expanded = mi.Float(np.repeat(valid_rho[:, 2], n_tx * n_rx))

    # Expand PDF and n_attempted for MC probability chain (Option B normalization)
    pdf_expanded = None
    n_attempted_expanded = None
    if valid_pdf is not None and valid_n_attempted is not None:
        # Each valid hit is replicated n_tx × n_rx times in MIMO expansion
        pdf_expanded = mi.Float(np.repeat(valid_pdf, n_tx * n_rx))  # [n_total]
        n_attempted_expanded = mi.Float(np.repeat(valid_n_attempted, n_tx * n_rx))  # [n_total]

    # Generate TX/RX indices for each expanded path
    tx_idx, rx_idx = generate_mimo_indices(n_valid, n_tx, n_rx)

    # Gather TX and RX positions for each path
    tx_pos_expanded = gather_point3f(tx_positions, tx_idx)  # [n_total]
    rx_pos_expanded = gather_point3f(rx_positions, rx_idx)  # [n_total]

    # Gather TX and RX boresights for pattern evaluation (if provided)
    tx_boresights_expanded = None
    if tx_boresights is not None:
        tx_boresights_expanded = gather_vector3f(tx_boresights, tx_idx)  # [n_total]

    rx_boresights_expanded = None
    if rx_boresights is not None:
        rx_boresights_expanded = gather_vector3f(rx_boresights, rx_idx)  # [n_total]

    # ====================================================================
    # Step 3: Vectorized Geometry Computation
    # ====================================================================
    # Distance from RX to hit
    delta_rx = hit_P_expanded - rx_pos_expanded
    d_rx_to_hit = dr.norm(delta_rx)  # [n_total]

    # Distance from hit to TX
    delta_tx = tx_pos_expanded - hit_P_expanded
    d_hit_to_tx = dr.norm(delta_tx)  # [n_total]

    # Direction from hit to RX (wo - outgoing)
    dir_hit_to_rx = mi.Vector3f(
        -delta_rx.x / dr.maximum(d_rx_to_hit, 1e-10),
        -delta_rx.y / dr.maximum(d_rx_to_hit, 1e-10),
        -delta_rx.z / dr.maximum(d_rx_to_hit, 1e-10)
    )

    # Direction from hit to TX (wi - incident/light direction)
    dir_hit_to_tx = mi.Vector3f(
        delta_tx.x / dr.maximum(d_hit_to_tx, 1e-10),
        delta_tx.y / dr.maximum(d_hit_to_tx, 1e-10),
        delta_tx.z / dr.maximum(d_hit_to_tx, 1e-10)
    )

    # cos theta values (initial, before potential normal flip)
    cos_theta_out_raw = dr.dot(dir_hit_to_rx, hit_N_expanded)  # [n_total]
    cos_theta_in_raw = dr.dot(dir_hit_to_tx, hit_N_expanded)   # [n_total]

    # ====================================================================
    # Double-sided rendering: flip normals for backfacing surfaces
    # ====================================================================
    # When enabled, surfaces with normals pointing away from the RX are
    # treated as if they face the RX. This is essential for:
    # - LiDAR meshes with inconsistent normal orientation
    # - Thin walls/guardrails that should scatter from both sides
    # - Improving sidewall visibility in staircase scenes
    double_sided = getattr(self.render_config, 'double_sided', True)

    if double_sided:
        # Flip normal if it points away from RX (cos_theta_out < 0)
        # This is equivalent to: N = faceforward(N, wo)
        need_flip = cos_theta_out_raw < 0
        hit_N_expanded = dr.select(
            need_flip,
            mi.Vector3f(-hit_N_expanded.x, -hit_N_expanded.y, -hit_N_expanded.z),
            hit_N_expanded
        )
        # Recompute cos_theta with potentially flipped normals
        cos_theta_out = dr.abs(cos_theta_out_raw)
        cos_theta_in = dr.select(need_flip, -cos_theta_in_raw, cos_theta_in_raw)

        if verbose:
            n_flipped = int(dr.sum(mi.Float(need_flip))[0]) if hasattr(dr.sum(mi.Float(need_flip)), '__getitem__') else int(dr.sum(mi.Float(need_flip)))
            print(f"  Double-sided: flipped {n_flipped:,} normals ({100*n_flipped/n_total:.1f}%)")
            # Diagnostic: check cos_theta distribution after flip
            cos_out_np = np.array(cos_theta_out)
            cos_in_np = np.array(cos_theta_in)
            print(f"  After flip: cos_theta_out range [{cos_out_np.min():.4f}, {cos_out_np.max():.4f}]")
            print(f"  After flip: cos_theta_in range [{cos_in_np.min():.4f}, {cos_in_np.max():.4f}]")
            n_still_backfacing_in = np.sum(cos_in_np < 0)
            n_grazing_out = np.sum((cos_out_np >= 0) & (cos_out_np < 0.1))
            n_grazing_in = np.sum((cos_in_np >= 0) & (cos_in_np < 0.1))
            print(f"  After flip: {n_still_backfacing_in:,} paths still have cos_theta_in < 0 ({100*n_still_backfacing_in/n_total:.1f}%)")
            print(f"  After flip: {n_grazing_out:,} paths have grazing cos_theta_out (0-0.1)")
            print(f"  After flip: {n_grazing_in:,} paths have grazing cos_theta_in (0-0.1)")
    else:
        cos_theta_out = cos_theta_out_raw
        cos_theta_in = cos_theta_in_raw

    # ====================================================================
    # Step 4: Vectorized Visibility Check (SINGLE CALL)
    # ====================================================================
    epsilon = 1e-4
    shadow_origins = mi.Point3f(
        hit_P_expanded.x + epsilon * dir_hit_to_tx.x,
        hit_P_expanded.y + epsilon * dir_hit_to_tx.y,
        hit_P_expanded.z + epsilon * dir_hit_to_tx.z
    )

    shadow_rays = mi.Ray3f(shadow_origins, dir_hit_to_tx)
    shadow_rays.maxt = d_hit_to_tx - 2 * epsilon

    # Single visibility check for ALL paths
    occluded = scene.ray_test(shadow_rays)
    visible = ~occluded  # [n_total]

    # ====================================================================
    # Step 5: Vectorized BRDF Computation
    # Handles both scalar BSDF (power) and complex Jones BSDF (Fresnel phase)
    # Per-patch adaptive BSDF: when patch_has_specular is provided, select
    # eval_non_ka_cos vs eval_f_cos per-hit based on patch specular coverage
    # Physics mode: dispatches to *_physics() methods with 6-param signature
    # ====================================================================
    # Check if BSDF uses Fresnel phase (Jones with use_fresnel_phase=True)
    use_fresnel_phase = getattr(self.bsdf, 'use_fresnel_phase', False)

    # Build BSDF call kwargs based on mode
    if physics_mode:
        bsdf_kwargs = dict(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N_expanded,
            eps_real=rho_eps_real, eps_imag=rho_eps_imag,
            sigma_h=rho_sigma_h, l_c=rho_l_c, tau=rho_tau,
            thickness=rho_thickness,
        )
        _eval_f_cos = getattr(self.bsdf, 'eval_f_cos_physics', self.bsdf.eval_f_cos)
        _eval_non_ka_cos = getattr(self.bsdf, 'eval_non_ka_cos_physics', self.bsdf.eval_non_ka_cos)
        _eval_f_cos_complex = getattr(self.bsdf, 'eval_f_cos_complex_physics',
                                       getattr(self.bsdf, 'eval_f_cos_complex', None))
        _eval_f_cos_components = getattr(self.bsdf, 'eval_f_cos_components_physics',
                                          getattr(self.bsdf, 'eval_f_cos_components', None))
    else:
        bsdf_kwargs = dict(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N_expanded,
            albedo=rho_x_expanded, roughness=rho_y_expanded,
            metallic=rho_z_expanded,
        )
        _eval_f_cos = self.bsdf.eval_f_cos
        _eval_non_ka_cos = getattr(self.bsdf, 'eval_non_ka_cos', None)
        _eval_f_cos_complex = getattr(self.bsdf, 'eval_f_cos_complex', None)
        _eval_f_cos_components = getattr(self.bsdf, 'eval_f_cos_components', None)

    if use_non_ka_bsdf and patch_has_specular is not None and tri_to_patch is not None \
            and _eval_non_ka_cos is not None and reservoir_hits.hit_ID is not None:
        # Per-patch adaptive BSDF: select BSDF per-hit based on patch specular coverage
        valid_hit_ID = dr.gather(mi.Int32, reservoir_hits.hit_ID, valid_indices_dr)
        hit_ID_expanded = dr.repeat(valid_hit_ID, n_tx * n_rx)
        safe_hit_ID = mi.UInt32(dr.maximum(hit_ID_expanded, mi.Int32(0)))
        hit_patch_id = dr.gather(mi.Int32, tri_to_patch, safe_hit_ID)
        use_non_ka_per_hit = dr.gather(mi.Bool, patch_has_specular, mi.UInt32(hit_patch_id))

        brdf_non_ka = _eval_non_ka_cos(**bsdf_kwargs)
        brdf_full = _eval_f_cos(**bsdf_kwargs)
        brdf_weight = dr.select(use_non_ka_per_hit, brdf_non_ka, brdf_full)
        E_coh_real = None
        E_coh_imag = None
        f_inc_cos = None
        use_fresnel_phase = False

        if verbose:
            n_non_ka = int(dr.sum(mi.UInt32(use_non_ka_per_hit))[0])
            n_full = n_total - n_non_ka
            print(f"  Per-patch BSDF: {n_non_ka:,} hits use non-KA, {n_full:,} hits use full BSDF")

    elif use_non_ka_bsdf and _eval_non_ka_cos is not None:
        brdf_weight = _eval_non_ka_cos(**bsdf_kwargs)
        E_coh_real = None
        E_coh_imag = None
        f_inc_cos = None
        use_fresnel_phase = False

        if verbose:
            print(f"  Using eval_non_ka_cos (KA excluded, handled by image method)")
    elif use_fresnel_phase and _eval_f_cos_complex is not None:
        E_coh_real, E_coh_imag, f_inc_cos = _eval_f_cos_complex(**bsdf_kwargs)
        brdf_weight = E_coh_real * E_coh_real + E_coh_imag * E_coh_imag + f_inc_cos

        if verbose:
            print(f"  Using Jones BSDF with Fresnel phase (complex coherent + power incoherent)")
    else:
        brdf_weight = _eval_f_cos(**bsdf_kwargs)
        E_coh_real = None
        E_coh_imag = None
        f_inc_cos = None

    if verbose and physics_mode:
        print(f"  Physics mode: 6-param direct BSDF dispatch")

    # ====================================================================
    # Step 5b: Evaluate BSDF components (if requested)
    # ====================================================================
    component_weights = None
    if return_components and _eval_f_cos_components is not None:
        component_weights = _eval_f_cos_components(**bsdf_kwargs)
        if verbose:
            print(f"  Component-level rendering enabled (6 components)")

    # ====================================================================
    # Step 5c: Free-Space Diffraction BSDF (fsdBSDF)
    # Additive combination: f_total = (1-β)×f_refl + f_diff
    # Only evaluated at hit points near diffracting edges.
    #
    # Optimization: aperture depends on (hit_pos, wo) where wo points
    # toward RX. Same (hit, RX) → same aperture, reused across all TX.
    # MIMO layout: flat_idx = h*(n_tx*n_rx) + t*n_rx + r
    # ====================================================================
    # Initialize diffraction cache variables (used later in CachedGeometry)
    _diff_cache_flat = None
    _diff_cache_eval_path_idx = None
    _diff_cache_eval_ap_idx = None
    _diff_cache_eval_wi = None
    _diff_cache_n_hits = 0
    _diff_cache_beta_np = None

    diff_config = self.diffraction_config
    if (_HAS_DIFFRACTION and diff_config is not None and diff_config.enabled
            and self.tri_hash is not None and self.fsd_bsdf is not None):

        # Convert DrJit arrays to numpy for per-hit aperture construction
        hit_P_np = np.stack([np.array(hit_P_expanded.x),
                             np.array(hit_P_expanded.y),
                             np.array(hit_P_expanded.z)], axis=1)  # [n_total, 3]
        wo_np = np.stack([np.array(dir_hit_to_rx.x),
                          np.array(dir_hit_to_rx.y),
                          np.array(dir_hit_to_rx.z)], axis=1)      # [n_total, 3]
        wi_np = np.stack([np.array(dir_hit_to_tx.x),
                          np.array(dir_hit_to_tx.y),
                          np.array(dir_hit_to_tx.z)], axis=1)      # [n_total, 3]

        k = self.fsd_bsdf.k
        beam_sigma = self.fsd_bsdf.beam_sigma

        n_diffraction_hits = 0
        visible_np = np.array(visible)

        # Per-triangle material arrays for material-dependent diffraction
        tri_eps_r_np, tri_eps_i_np = self.diffraction_tri_eps
        use_mat_opacity = (diff_config.material_opacity
                           and tri_eps_r_np is not None
                           and tri_eps_i_np is not None)

        # Jones polarization: extract TX/RX E-field directions from BSDF
        jones_mode = diff_config.jones_polarization and use_mat_opacity
        tx_pol_np = None
        rx_pol_np = None
        if jones_mode:
            if hasattr(self.bsdf, 'tx_polarization'):
                tx_pol_np = np.array([
                    float(self.bsdf.tx_polarization.x[0]),
                    float(self.bsdf.tx_polarization.y[0]),
                    float(self.bsdf.tx_polarization.z[0]),
                ])
                rx_pol_np = np.array([
                    float(self.bsdf.rx_polarization.x[0]),
                    float(self.bsdf.rx_polarization.y[0]),
                    float(self.bsdf.rx_polarization.z[0]),
                ])
            else:
                # No Jones BSDF → fall back to scalar
                jones_mode = False

        # === Batch aperture construction (replaces serial loop) ===
        # Reference positions for all valid hits (TX=0, RX=0)
        ref_flat_indices = np.arange(n_valid) * n_tx * n_rx
        hit_pos_ref = hit_P_np[ref_flat_indices]   # [n_valid, 3]
        wo_ref = wo_np[ref_flat_indices]            # [n_valid, 3]

        # Pre-filter: only process hits with at least one visible path
        vis_2d = visible_np.reshape(n_valid, n_tx * n_rx)
        visible_per_hit = np.any(vis_2d, axis=1)    # [n_valid]
        vis_hit_indices = np.where(visible_per_hit)[0]

        flat = None
        n_skipped_flat = 0

        if len(vis_hit_indices) > 0:
            # GPU aperture construction (CPU path is deprecated)
            from ..diffraction.fsd_aperture_gpu import construct_apertures_batch_gpu
            result = construct_apertures_batch_gpu(
                hit_pos_ref[vis_hit_indices],
                wo_ref[vis_hit_indices],
                self._gpu_spatial_hash, k, beam_sigma,
                diff_config.max_tessellation_depth,
                diff_config.max_edges_per_hit,
                diff_config.fill_min, diff_config.fill_max,
                tri_eps_r_np if use_mat_opacity else None,
                tri_eps_i_np if use_mat_opacity else None,
                tx_pol_np if jones_mode else None,
                rx_pol_np if jones_mode else None,
                jones_mode, self.fsd_bsdf.beta_max, verbose,
                edge_angle_threshold_deg=diff_config.edge_angle_threshold_deg,
                tri_hash=self.tri_hash,
            )

            if result is not None:
                flat, active_local_idx = result

                # Map batch-local indices back to original hit indices
                active_hit_orig = vis_hit_indices[active_local_idx]  # [A]
                n_diffraction_hits = flat.n_apertures
                n_skipped_flat = n_valid - n_diffraction_hits

                # Build ap_to_path_indices: for each aperture, find visible paths
                bases = active_hit_orig * n_tx * n_rx  # [A]
                pair_offsets = np.arange(n_tx * n_rx, dtype=np.int32)
                all_pairs = bases[:, None] + pair_offsets[None, :]  # [A, n_tx*n_rx]
                vis_expanded = visible_np[all_pairs]  # [A, n_tx*n_rx]
                flat.ap_to_path_indices = [
                    all_pairs[a, vis_expanded[a]].astype(np.int32)
                    for a in range(flat.n_apertures)
                ]

        # --- Phase B: Batch-evaluate all (aperture, TX) pairs at once ---
        # Use GPU (DrJit) path: results stay on GPU, no CPU→GPU round-trip
        f_diff_dr = dr.zeros(mi.Float, n_total)
        beta_np = np.zeros(n_total, dtype=np.float64)
        psi_diff_real_dr = dr.zeros(mi.Float, n_total)
        psi_diff_imag_dr = dr.zeros(mi.Float, n_total)

        if flat is not None and flat.n_apertures > 0:

            # Build evaluation arrays using vectorized concat + repeat
            path_counts = np.array([len(p) for p in flat.ap_to_path_indices], dtype=np.int32)
            total_evals = int(np.sum(path_counts))

            if total_evals > 0:
                eval_path_idx = np.concatenate(flat.ap_to_path_indices)  # [N_eval]
                eval_ap_idx = np.repeat(np.arange(flat.n_apertures, dtype=np.int32), path_counts)
                eval_wi = wi_np[eval_path_idx]  # [N_eval, 3]

                # GPU eval: returns DrJit mi.Float arrays
                f_diff_batch, psi_r_batch, psi_i_batch = eval_batch_drjit(
                    flat, eval_wi, k, aperture_indices=eval_ap_idx
                )

                # Scatter results directly on GPU (no CPU→GPU transfer)
                eval_path_idx_dr = mi.UInt32(eval_path_idx)
                dr.scatter(f_diff_dr, f_diff_batch, eval_path_idx_dr)

                if diff_config.complex_output:
                    dr.scatter(psi_diff_real_dr, psi_r_batch, eval_path_idx_dr)
                    dr.scatter(psi_diff_imag_dr, psi_i_batch, eval_path_idx_dr)

                if diff_config.energy_borrowing:
                    beta_per_eval = flat.ap_beta[eval_ap_idx]
                    beta_np[eval_path_idx] = beta_per_eval

                # Cache for Phase B differentiable re-evaluation
                if diff_config.material_opacity and return_cached_geometry:
                    _diff_cache_flat = flat
                    _diff_cache_eval_path_idx = eval_path_idx
                    _diff_cache_eval_ap_idx = eval_ap_idx
                    _diff_cache_eval_wi = eval_wi
                    _diff_cache_n_hits = n_diffraction_hits
                    _diff_cache_beta_np = beta_np

        # Energy-conserving edge diffraction for backscatter radar:
        #
        # The FSD BSDF (Steinberg et al. 2024) is a TRANSMISSIVE lobe —
        # the reference implementation (free_space_diffraction.cpp) explicitly
        # restricts evaluation to the opposite hemisphere from wi:
        #   eval():  cosθ(wo) × cosθ(wi) < 0   (line 112)
        #   FSDpossible(): dot(wi, -wo) > 0.35  (line 270)
        # Both guards prevent FSD from producing backscatter.
        #
        # For our single-bounce backscatter radar, edges affect reflection
        # through shadow/blocking: β fraction of the beam is intercepted by
        # edges, reducing reflected energy by (1-β).  The intercepted energy
        # goes to forward diffraction (transmission) — a loss in our renderer,
        # analogous to how we model transmission through materials as a loss.
        #
        # Energy budget per path:
        #   Non-intercepted: (1-β) × R  →  reflected back (captured)
        #   Edge-intercepted: β          →  forward-diffracted (lost)
        #   Absorbed:         (1-β)(1-R) →  absorbed (lost)
        if diff_config.energy_borrowing and np.any(beta_np > 0):
            one_minus_beta = mi.Float(1.0 - beta_np)
            brdf_weight = brdf_weight * one_minus_beta

            if use_fresnel_phase and E_coh_real is not None:
                sqrt_one_minus_beta = mi.Float(np.sqrt(1.0 - beta_np))
                E_coh_real = E_coh_real * sqrt_one_minus_beta
                E_coh_imag = E_coh_imag * sqrt_one_minus_beta
                f_inc_cos = f_inc_cos * one_minus_beta

        if verbose:
            f_diff_np = np.array(f_diff_dr)
            print(f"  Diffraction: {n_diffraction_hits:,} apertures near edges "
                  f"(of {n_valid * n_rx:,} hit×RX pairs, {n_total:,} total paths), "
                  f"{n_skipped_flat:,}/{n_valid:,} hits pre-filtered (flat surface)")
            if n_diffraction_hits > 0:
                n_evals = int(np.sum(f_diff_np > 0))
                active_diff = f_diff_np[f_diff_np > 0]
                if len(active_diff) > 0:
                    print(f"    f_diff range: [{active_diff.min():.4e}, {active_diff.max():.4e}] ({n_evals:,} active)")
                active_beta = beta_np[beta_np > 0]
                if len(active_beta) > 0:
                    print(f"    beta range: [{active_beta.min():.4f}, {active_beta.max():.4f}]")

    # ====================================================================
    # Step 6: Vectorized Antenna Pattern Evaluation (with orientation support)
    # ====================================================================
    # Track combined scale factor for component weights (Steps 6+7)
    antenna_gain_combined = None
    mc_correction = None

    # GPU-native path: use AntennaPatternLoader (DrJit, no GPU→CPU sync)
    # Note: pattern_mode check removed to match E2E path (synthesis_e2e.py)
    # which always uses evaluate_combined_gain when loaders are available
    _use_gpu_pattern = (
        evaluate_combined_gain is not None
        and (tx_pattern_loader is not None or rx_pattern_loader is not None)
    )

    if _use_gpu_pattern:
        # Direction from antenna to hit (negate hit-to-antenna vectors)
        dir_rx_to_hit = -dir_hit_to_rx
        dir_tx_to_hit = -dir_hit_to_tx

        # RX gain (GPU-native: C × gain_E × gain_H, returns linear)
        if rx_pattern_loader is not None:
            gain_rx_linear = evaluate_combined_gain(
                rx_pattern_loader, dir_rx_to_hit, rx_boresights_expanded)
        else:
            gain_rx_linear = mi.Float(1.0)

        # TX gain (GPU-native)
        if tx_pattern_loader is not None:
            gain_tx_linear = evaluate_combined_gain(
                tx_pattern_loader, dir_tx_to_hit, tx_boresights_expanded)
        else:
            gain_tx_linear = mi.Float(1.0)

        # evaluate_combined_gain returns linear scale directly
        if not antenna_gain_linear and not self.render_config.use_radar_equation:
            # Convert linear to dB for legacy dB-as-multiplier mode
            gain_rx_db = mi.Float(10.0) * dr.log(dr.maximum(gain_rx_linear, mi.Float(1e-30))) / dr.log(mi.Float(10.0))
            gain_tx_db = mi.Float(10.0) * dr.log(dr.maximum(gain_tx_linear, mi.Float(1e-30))) / dr.log(mi.Float(10.0))
            antenna_gain_combined = gain_rx_db * gain_tx_db
        else:
            antenna_gain_combined = gain_rx_linear * gain_tx_linear

        brdf_weight = brdf_weight * antenna_gain_combined

        if verbose:
            dr.eval(antenna_gain_combined)
            print(f"  Antenna pattern: GPU-native (AntennaPatternLoader)")

    elif tx_pattern_rrts is not None or rx_pattern_rrts is not None:
        # DrJit GPU fallback: all-GPU evaluation (no CPU↔GPU transfers)
        dir_rx_to_hit = mi.Vector3f(-dir_hit_to_rx.x, -dir_hit_to_rx.y, -dir_hit_to_rx.z)
        dir_tx_to_hit = mi.Vector3f(-dir_hit_to_tx.x, -dir_hit_to_tx.y, -dir_hit_to_tx.z)

        if pattern_mode == 'legacy':
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_rx_dB = evaluate_gain_rrts_style_drjit(dir_rx_to_hit, rx_pattern_rrts)
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_total)

            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit is not None:
                gain_tx_dB = evaluate_gain_rrts_style_drjit(dir_tx_to_hit, tx_pattern_rrts)
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_total)
        else:
            if rx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                gain_rx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_rx_to_hit, rx_pattern_rrts,
                    rx_boresights_expanded if rx_boresights_expanded is not None
                    else mi.Vector3f(0.0, 1.0, 0.0),
                )
            else:
                gain_rx_dB = dr.zeros(mi.Float, n_total)

            if tx_pattern_rrts is not None and evaluate_gain_rrts_style_drjit_product is not None:
                gain_tx_dB = evaluate_gain_rrts_style_drjit_product(
                    dir_tx_to_hit, tx_pattern_rrts,
                    tx_boresights_expanded if tx_boresights_expanded is not None
                    else mi.Vector3f(0.0, 1.0, 0.0),
                )
            else:
                gain_tx_dB = dr.zeros(mi.Float, n_total)

        if antenna_gain_linear or self.render_config.use_radar_equation:
            gain_rx_linear = dr.power(mi.Float(10.0), gain_rx_dB * mi.Float(0.1))
            gain_tx_linear = dr.power(mi.Float(10.0), gain_tx_dB * mi.Float(0.1))
        else:
            gain_rx_linear = gain_rx_dB
            gain_tx_linear = gain_tx_dB

        antenna_gain_combined = gain_rx_linear * gain_tx_linear
        brdf_weight = brdf_weight * antenna_gain_combined

        if verbose:
            print(f"  Antenna pattern: DrJit GPU fallback")

    # ====================================================================
    # Step 7: Apply Monte Carlo correction (Option B: 1/(pdf × n_attempted))
    # ====================================================================
    # According to rx_centric_radar_mc_estimators.tex Eq. 4:
    # P_r = (P_t λ²)/(4π)² × (1/N_r) × Σⱼ [G_t G_r c_t f V] / [d_t² × ρ_rx(ωⱼ)]
    #
    # Option B normalization: normalize by n_attempted (not n_hits)
    # This accounts for the conditional distribution p(ω|hit) = p(ω) × 1[hit(ω)] / P_hit
    # where P_hit = n_hits / n_attempted
    #
    # The correction factor is: 1 / (ρ_rx(ω) × n_attempted_per_rx)
    if pdf_expanded is not None and n_attempted_expanded is not None:
        # Compute MC correction: 1 / (pdf × n_attempted)
        # Use small epsilon to avoid division by zero for invalid hits
        EPS_PDF = 1e-8
        mc_correction = mi.Float(1.0) / (dr.maximum(pdf_expanded, EPS_PDF) * n_attempted_expanded)
        brdf_weight = brdf_weight * mc_correction

        if verbose:
            mc_corr_np = np.array(mc_correction)
            valid_mc = mc_corr_np[np.array(visible) & (np.array(cos_theta_out) > 1e-6) & (np.array(cos_theta_in) > 1e-6)]
            if len(valid_mc) > 0:
                print(f"  MC correction range: [{valid_mc.min():.4e}, {valid_mc.max():.4e}]")
    else:
        if verbose:
            print(f"  WARNING: MC probability chain not available (hit_pdf or n_attempted missing)")
            print(f"  This will result in biased estimates!")

    # DEBUG: Match e2e debug prints for comparison
    if verbose:
        dr.eval(brdf_weight, antenna_gain_combined if antenna_gain_combined is not None else mi.Float(1.0))
        _bw = np.array(brdf_weight)
        _act_mask = np.array(visible) & (np.array(cos_theta_out) > 1e-6) & (np.array(cos_theta_in) > 1e-6)
        if _act_mask.sum() > 0:
            print(f"  [FWD-DEBUG] After BSDF+antenna+MC:")
            print(f"    brdf_weight[active]: mean={_bw[_act_mask].mean():.6e}, max={_bw[_act_mask].max():.6e}, min={_bw[_act_mask].min():.6e}")
            if antenna_gain_combined is not None:
                _ag = np.array(antenna_gain_combined)
                print(f"    antenna_gain[active]: mean={_ag[_act_mask].mean():.6e}, max={_ag[_act_mask].max():.6e}")

    # ====================================================================
    # Step 8: Active mask (visibility + geometry validity)
    # ====================================================================
    active = visible & (cos_theta_out > 1e-6) & (cos_theta_in > 1e-6)

    # Count visible paths
    active_count = dr.sum(mi.Float(active))
    total_paths = int(active_count[0]) if hasattr(active_count, '__getitem__') else int(active_count)

    if verbose:
        print(f"  Visible paths: {total_paths:,}")

    # ====================================================================
    # Step 8b: Build CachedGeometry for differentiable re-synthesis
    # ====================================================================
    if return_cached_geometry and valid_prim_ids is not None:
        # Expand prim IDs to MIMO layout (same as positions/normals)
        hit_prim_ids_expanded = expand_for_mimo(valid_prim_ids, n_tx, n_rx)

        # Expand per-vertex fields to MIMO layout if available
        bary_u_expanded = None
        bary_v_expanded = None
        vi0_expanded = vi1_expanded = vi2_expanded = None
        if valid_bary_u is not None:
            bary_u_expanded = expand_for_mimo(valid_bary_u, n_tx, n_rx)
            bary_v_expanded = expand_for_mimo(valid_bary_v, n_tx, n_rx)
        if valid_vertex_ids_0 is not None:
            vi0_expanded = expand_for_mimo(valid_vertex_ids_0, n_tx, n_rx)
            vi1_expanded = expand_for_mimo(valid_vertex_ids_1, n_tx, n_rx)
            vi2_expanded = expand_for_mimo(valid_vertex_ids_2, n_tx, n_rx)

        cached_geometry = CachedGeometry(
            hit_P_expanded=hit_P_expanded,
            hit_N_expanded=hit_N_expanded,
            dir_hit_to_rx=dir_hit_to_rx,
            dir_hit_to_tx=dir_hit_to_tx,
            d_rx_to_hit=d_rx_to_hit,
            d_hit_to_tx=d_hit_to_tx,
            active=active,
            tx_idx=tx_idx,
            rx_idx=rx_idx,
            hit_prim_ids=hit_prim_ids_expanded,
            antenna_gain_combined=antenna_gain_combined,
            mc_correction=mc_correction,
            n_tx=n_tx,
            n_rx=n_rx,
            n_valid=n_valid,
            n_total=n_total,
            # Per-vertex parameterization fields
            hit_bary_u=bary_u_expanded,
            hit_bary_v=bary_v_expanded,
            hit_vertex_ids_0=vi0_expanded,
            hit_vertex_ids_1=vi1_expanded,
            hit_vertex_ids_2=vi2_expanded,
            # Per-element TX/RX data for pose/pattern recomputation (Fix 1, 2)
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            tx_boresights=tx_boresights,
            rx_boresights=rx_boresights,
        )

        # Cache diffraction data for Phase B differentiable re-evaluation
        if _diff_cache_flat is not None:
            cached_geometry.diff_flat = _diff_cache_flat
            cached_geometry.diff_eval_path_idx = _diff_cache_eval_path_idx
            cached_geometry.diff_eval_ap_idx = _diff_cache_eval_ap_idx
            cached_geometry.diff_eval_wi = _diff_cache_eval_wi
            cached_geometry.diff_n_diffraction_hits = _diff_cache_n_hits
            cached_geometry.diff_beta_np = _diff_cache_beta_np
            # Store clamped cos_theta_in for Phase B cos scaling (Fix 2b)
            cached_geometry.diff_cos_theta_in = np.maximum(np.array(cos_theta_in, dtype=np.float32), 0.0)

    # ====================================================================
    # Step 9: Compute weight based on radar equation or legacy mode
    # ====================================================================
    R_total = d_rx_to_hit + d_hit_to_tx  # [n_total]

    if self.render_config.use_radar_equation:
        # RX-centric Monte Carlo estimator (see rx_centric_radar_mc_estimators.tex Eq. 4):
        # P_r = (P_t λ²)/(4π)² × [G_t G_r c_t f V] / [d_t² × ρ_rx(ω_r)]
        #
        # Key insight: In RX-centric sampling, the change of variables
        # dω_r = (c_r / d_r²) × dA_x absorbs both c_r and 1/d_r²
        # Therefore only d_t (hit→TX distance) appears, with exponent 2 (not 4)
        d_t = d_hit_to_tx  # Distance from hit to TX only [n_total]

        # Path loss: 1 / d_t² (NOT 1/R⁴)
        path_loss = mi.Float(1.0) / (d_t * d_t)
        sqrt_path_loss = dr.sqrt(path_loss)

        # Radar scaling factor
        radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)

        if use_fresnel_phase and E_coh_real is not None:
            # Jones BSDF with Fresnel phase: separate coherent and incoherent
            # Coherent: E_coh already has sqrt(f_lobe) baked in, apply path loss
            weight_coh_real = E_coh_real * sqrt_path_loss * radar_scale
            weight_coh_imag = E_coh_imag * sqrt_path_loss * radar_scale
            # Incoherent: f_inc is power, convert to amplitude
            weight_inc = dr.sqrt(dr.maximum(f_inc_cos, mi.Float(0.0))) * sqrt_path_loss * radar_scale
        else:
            # Standard scalar: sqrt(power × path_loss) × scale
            E_amplitude = radar_scale * dr.sqrt(brdf_weight * path_loss)
            weight = E_amplitude
    else:
        # Legacy mode: BRDF weight only (no physical power scaling)
        if use_fresnel_phase and E_coh_real is not None:
            # For legacy mode with Fresnel phase, just use the complex amplitude directly
            weight_coh_real = E_coh_real
            weight_coh_imag = E_coh_imag
            weight_inc = dr.sqrt(dr.maximum(f_inc_cos, mi.Float(0.0)))
        else:
            weight = brdf_weight

    # ====================================================================
    # Step 10: Vectorized Phase Computation + Scatter-Add Accumulation
    # ====================================================================
    tau = R_total / C  # [n_total]

    # Phase constants (propagation phase)
    TWO_PI = 2.0 * np.pi
    phi_const = TWO_PI * self.min_freq * tau   # [n_total]
    phi_slope = TWO_PI * self.slope * tau      # [n_total]

    # Time grid as DrJit array
    t_grid_dr = self._t_grid_dr  # [K] (cached on GPU)

    # Expand to [n_total × K] using dr.tile and dr.repeat
    phi_const_3d = dr.tile(phi_const, K)   # [n_total × K]
    phi_slope_3d = dr.tile(phi_slope, K)   # [n_total × K]
    t_k = dr.repeat(t_grid_dr, n_total)    # [n_total × K]
    active_3d = dr.tile(active, K)         # [n_total × K]

    # Compute propagation phase for ALL paths × ALL samples at once
    phi = phi_const_3d + phi_slope_3d * t_k  # [n_total × K]

    # Complex phasor computation
    cos_phi = dr.cos(phi)
    sin_phi = dr.sin(phi)

    if use_fresnel_phase and E_coh_real is not None:
        # Jones BSDF with Fresnel phase:
        # Coherent phasor: E_coh × exp(j × phi_propagation)
        # = (E_coh_real + j*E_coh_imag) × (cos(phi) + j*sin(phi))
        # = (E_coh_real*cos(phi) - E_coh_imag*sin(phi)) + j*(E_coh_real*sin(phi) + E_coh_imag*cos(phi))
        weight_coh_real_3d = dr.tile(weight_coh_real, K)
        weight_coh_imag_3d = dr.tile(weight_coh_imag, K)
        weight_inc_3d = dr.tile(weight_inc, K)

        # Coherent contribution (with Fresnel phase combined with propagation phase)
        contrib_coh_real = weight_coh_real_3d * cos_phi - weight_coh_imag_3d * sin_phi
        contrib_coh_imag = weight_coh_real_3d * sin_phi + weight_coh_imag_3d * cos_phi

        # Incoherent contribution (propagation phase only, no Fresnel phase)
        contrib_inc_real = weight_inc_3d * cos_phi
        contrib_inc_imag = weight_inc_3d * sin_phi

        # Total contribution = coherent + incoherent
        contrib_real = contrib_coh_real + contrib_inc_real
        contrib_imag = contrib_coh_imag + contrib_inc_imag
    else:
        # Standard scalar BSDF: weight × exp(j × phi)
        # Ensure weight is mi.Float for proper dr.tile behavior
        if not isinstance(weight, mi.Float):
            weight = mi.Float(weight)
        weight_3d = dr.tile(weight, K)
        contrib_real = weight_3d * cos_phi
        contrib_imag = weight_3d * sin_phi

    # ====================================================================
    # Step 11: Compute flat indices for scatter_add
    # ====================================================================
    tx_idx_3d = dr.tile(tx_idx, K)  # [n_total × K]
    rx_idx_3d = dr.tile(rx_idx, K)  # [n_total × K]

    # k_idx: for dr.tile layout, k = i // n_total
    base_idx = dr.arange(mi.UInt32, n_total * K)
    k_idx = base_idx // mi.UInt32(n_total)

    flat_idx = tx_idx_3d * mi.UInt32(n_rx * K) + rx_idx_3d * mi.UInt32(K) + k_idx

    # ====================================================================
    # Step 12: Atomic scatter-add accumulation
    # ====================================================================
    adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
    adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

    dr.scatter_add(adc_real_flat, contrib_real, flat_idx, active_3d)
    dr.scatter_add(adc_imag_flat, contrib_imag, flat_idx, active_3d)

    # ====================================================================
    # Step 12b: Specular path synthesis (image method branch)
    # ====================================================================
    if specular_paths is not None and specular_paths.n_paths > 0:
        sp_antenna_gain = self._synthesize_specular_paths(
            specular_paths=specular_paths,
            adc_real_flat=adc_real_flat,
            adc_imag_flat=adc_imag_flat,
            n_tx=n_tx, n_rx=n_rx, K=K,
            tx_boresights=tx_boresights,
            rx_boresights=rx_boresights,
            tx_pattern_rrts=tx_pattern_rrts,
            rx_pattern_rrts=rx_pattern_rrts,
            tx_pattern_loader=tx_pattern_loader,
            rx_pattern_loader=rx_pattern_loader,
            antenna_gain_linear=antenna_gain_linear,
            verbose=verbose,
        )
        # Cache antenna gain for Phase B differentiable re-evaluation
        if return_cached_geometry and cached_geometry is not None and sp_antenna_gain is not None:
            cached_geometry.specular_antenna_gain = sp_antenna_gain

    # Reshape to [n_tx, n_rx, K]
    adc_real = np.array(adc_real_flat).reshape(n_tx, n_rx, K)
    adc_imag = np.array(adc_imag_flat).reshape(n_tx, n_rx, K)

    # ====================================================================
    # Step 13: Component-level scatter-add (if requested)
    # ====================================================================
    if return_components and component_weights is not None:
        # Compute component ADCs using the same phase/index setup
        # CRITICAL: Apply the same antenna gain and MC correction to components
        # as was applied to brdf_weight in Steps 6-7. Without these multipliers,
        # component values would be orders of magnitude too small.
        component_names = ['f_ka_cos', 'f_spm_cos', 'f_directive_cos', 'f_broad_cos', 'f_coherent_cos', 'f_incoherent_cos']
        component_adcs = {}

        if verbose:
            # Debug: log whether corrections are being applied
            has_antenna = antenna_gain_combined is not None
            has_mc = mc_correction is not None
            print(f"  Component corrections: antenna_gain={has_antenna}, mc_correction={has_mc}")

        for comp_name in component_names:
            comp_weight = component_weights[comp_name]

            # Apply antenna pattern gains (same as Step 6)
            if antenna_gain_combined is not None:
                comp_weight = comp_weight * antenna_gain_combined

            # Apply MC correction (same as Step 7)
            if mc_correction is not None:
                comp_weight = comp_weight * mc_correction

            # Apply path loss if radar equation is enabled
            if self.render_config.use_radar_equation:
                d_t = d_hit_to_tx
                path_loss = mi.Float(1.0) / (d_t * d_t)
                sqrt_path_loss = dr.sqrt(path_loss)
                radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
                comp_amplitude = radar_scale * dr.sqrt(dr.maximum(comp_weight, mi.Float(0.0))) * sqrt_path_loss
            else:
                # Legacy mode: direct weight
                comp_amplitude = dr.sqrt(dr.maximum(comp_weight, mi.Float(0.0)))

            # Expand to 3D and compute phasor
            if not isinstance(comp_amplitude, mi.Float):
                comp_amplitude = mi.Float(comp_amplitude)
            comp_weight_3d = dr.tile(comp_amplitude, K)
            comp_real = comp_weight_3d * cos_phi
            comp_imag = comp_weight_3d * sin_phi

            # Scatter-add
            comp_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
            comp_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
            dr.scatter_add(comp_real_flat, comp_real, flat_idx, active_3d)
            dr.scatter_add(comp_imag_flat, comp_imag, flat_idx, active_3d)

            # Reshape and store as complex
            comp_real_arr = np.array(comp_real_flat).reshape(n_tx, n_rx, K)
            comp_imag_arr = np.array(comp_imag_flat).reshape(n_tx, n_rx, K)
            component_adcs[comp_name] = (comp_real_arr + 1j * comp_imag_arr).astype(np.complex64)

        # Diffraction component ADC (from f_diff_dr computed in Step 5c)
        diff_comp_weight = f_diff_dr  # [n_total] DrJit mi.Float

        # Apply same antenna/MC/path-loss corrections as BSDF components
        if antenna_gain_combined is not None:
            diff_comp_weight = diff_comp_weight * antenna_gain_combined
        if mc_correction is not None:
            diff_comp_weight = diff_comp_weight * mc_correction

        if self.render_config.use_radar_equation:
            d_t = d_hit_to_tx
            path_loss = mi.Float(1.0) / (d_t * d_t)
            sqrt_path_loss = dr.sqrt(path_loss)
            radar_scale = mi.Float(self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
            diff_amplitude = radar_scale * dr.sqrt(dr.maximum(diff_comp_weight, mi.Float(0.0))) * sqrt_path_loss
        else:
            diff_amplitude = dr.sqrt(dr.maximum(diff_comp_weight, mi.Float(0.0)))

        if not isinstance(diff_amplitude, mi.Float):
            diff_amplitude = mi.Float(diff_amplitude)
        diff_weight_3d = dr.tile(diff_amplitude, K)
        diff_real_3d = diff_weight_3d * cos_phi
        diff_imag_3d = diff_weight_3d * sin_phi

        diff_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
        diff_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
        dr.scatter_add(diff_real_flat, diff_real_3d, flat_idx, active_3d)
        dr.scatter_add(diff_imag_flat, diff_imag_3d, flat_idx, active_3d)

        diff_real_arr = np.array(diff_real_flat).reshape(n_tx, n_rx, K)
        diff_imag_arr = np.array(diff_imag_flat).reshape(n_tx, n_rx, K)
        diff_adc = (diff_real_arr + 1j * diff_imag_arr).astype(np.complex64)

        if verbose:
            print(f"  Component ADC synthesis complete (6 BSDF + diffraction)")

        # Return ADCComponentResult
        return ADCComponentResult(
            ka=component_adcs['f_ka_cos'],
            spm=component_adcs['f_spm_cos'],
            directive=component_adcs['f_directive_cos'],
            broad=component_adcs['f_broad_cos'],
            coherent=component_adcs['f_coherent_cos'],
            incoherent=component_adcs['f_incoherent_cos'],
            diffraction=diff_adc,
            total=(adc_real + 1j * adc_imag).astype(np.complex64),
            n_tx=n_tx,
            n_rx=n_rx,
            n_samples=K,
            total_paths=total_paths
        )

    if verbose:
        print(f"  ADC synthesis complete")

    adc_result = ADCResult(
        adc_real=adc_real.astype(np.float32),
        adc_imag=adc_imag.astype(np.float32),
        n_tx=n_tx,
        n_rx=n_rx,
        n_samples=K,
        total_paths=total_paths
    )

    if return_cached_geometry and cached_geometry is not None:
        # Cache specular path data for differentiable re-evaluation
        if specular_paths is not None and specular_paths.n_paths > 0:
            cached_geometry.specular_hit_P = specular_paths.hit_P
            cached_geometry.specular_hit_N = specular_paths.hit_N
            cached_geometry.specular_dir_to_tx = specular_paths.dir_to_tx
            cached_geometry.specular_dir_to_rx = specular_paths.dir_to_rx
            cached_geometry.specular_d_tx = specular_paths.d_tx
            cached_geometry.specular_d_rx = specular_paths.d_rx
            cached_geometry.specular_cos_theta_i = specular_paths.cos_theta_i
            cached_geometry.specular_cos_theta_r = specular_paths.cos_theta_r
            cached_geometry.specular_A_tri = specular_paths.A_tri
            cached_geometry.specular_tx_idx = specular_paths.tx_idx
            cached_geometry.specular_rx_idx = specular_paths.rx_idx
            cached_geometry.specular_valid = specular_paths.valid
            cached_geometry.specular_n_paths = specular_paths.n_paths
            # Extract prim_ids from specular paths if available
            if hasattr(specular_paths, 'prim_ids') and specular_paths.prim_ids is not None:
                cached_geometry.specular_prim_ids = specular_paths.prim_ids

            # Cache IFT data (barycentrics, vertex IDs, grad_info) from SMS
            if hasattr(specular_paths, 'grad_info') and specular_paths.grad_info is not None:
                cached_geometry.specular_grad_info = specular_paths.grad_info
            if hasattr(specular_paths, 'bary_u') and specular_paths.bary_u is not None:
                cached_geometry.specular_bary_u = specular_paths.bary_u
                cached_geometry.specular_bary_v = specular_paths.bary_v

                # Extract vertex IDs from mesh face indices
                sp_prim = cached_geometry.specular_prim_ids
                if sp_prim is not None and scene is not None:
                    try:
                        mesh = scene.shapes()[0] if len(scene.shapes()) > 0 else None
                        if mesh is not None and hasattr(mesh, 'face_count'):
                            params = mi.traverse(mesh)
                            fi = mi.UInt32(params.get('faces', None))
                            if fi is not None:
                                n_faces = mesh.face_count()
                                safe_pid = dr.minimum(sp_prim, mi.UInt32(max(n_faces - 1, 0)))
                                cached_geometry.specular_vertex_ids_0 = dr.gather(mi.UInt32, fi, safe_pid * 3)
                                cached_geometry.specular_vertex_ids_1 = dr.gather(mi.UInt32, fi, safe_pid * 3 + 1)
                                cached_geometry.specular_vertex_ids_2 = dr.gather(mi.UInt32, fi, safe_pid * 3 + 2)
                    except Exception:
                        pass  # Vertex ID extraction not critical

        return adc_result, cached_geometry
    if return_cached_geometry:
        return adc_result, None
    return adc_result

# DELETED: _compute_full_brdf_vectorized()
# Moved to bsdf_rrts.py as BSDFRRTSDrJit.eval_f_cos()
# All BRDF evaluation now goes through self.bsdf.eval_f_cos()

# =========================================================================
# End-to-End Differentiable Synthesis
# =========================================================================

