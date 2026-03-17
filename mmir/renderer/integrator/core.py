"""
SBR Integrator core: class definition, initialization, and radar constants.

The SBRIntegratorRef class orchestrates FMCW ADC signal synthesis from
reservoir hits. Methods for forward, differentiable, and end-to-end
rendering are attached from separate modules.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple
import drjit as dr
import mitsuba as mi
import numpy as np

from ..bsdf.mmwave_scalar import BSDFmmWaveScalar
from ..bsdf.mmwave_jones import BSDFmmWaveJones
from ..materials.parameterization import MaterialParameterization, PerTriangleParameterization

# Import diffraction modules (optional)
try:
    from ..diffraction import (DiffractionConfig, TriangleSpatialHash, FsdAperture,
                               construct_aperture, FsdBSDF, FlatEdgeData,
                               pack_apertures, eval_batch, eval_batch_drjit)
    from ..diffraction.triangle_search import build_triangle_hash_from_scene
    _HAS_DIFFRACTION = True
except ImportError:
    _HAS_DIFFRACTION = False

if TYPE_CHECKING:
    from mmir.sensor.config import FMCWConfig
    from ..config import RenderConfigRef

# Physical Constants
C = 299792458.0  # Speed of light (m/s)
K_BOLTZMANN = 1.38e-23  # Boltzmann constant (J/K)
T_SYSTEM = 290.0  # System temperature (K)


class SBRIntegratorRef:
    """
    RRTS-exact FMCW ADC synthesis integrator.

    Synthesizes ADC signals from reservoir hits using RX-centric geometry.
    Matches RRTS synth_adc_fmcw.slang exactly.

    Key implementation details:
    - Time grid: t[k] = k / sample_rate
    - Phase: phi = 2π(S*tau*t + f0*tau) where tau = (d_rx_to_hit + d_hit_to_tx) / c
    - BRDF weight: Diffuse_light = INVPI * cos(theta_in)

    RX-Centric Monte Carlo Radar Equation (when use_radar_equation=True):
    - P_r = (P_t * λ²) / (4π)² × [G_t * G_r * c_t * f * V] / [d_t² * ρ_rx]
    - d_t = distance from hit to TX (NOT round-trip or one-way average)
    - c_r and d_r² are absorbed by the solid-angle change of variables
    - See rx_centric_radar_mc_estimators.tex for derivation
    """

    def __init__(
        self,
        config: 'FMCWConfig',
        render_config: 'RenderConfigRef' = None,
    ):
        """
        Initialize integrator.

        Args:
            config: FMCW radar configuration
            render_config: Render configuration with radar hardware parameters
        """
        self.config = config

        # Import RenderConfigRef here to avoid circular import
        from ..config import RenderConfigRef
        self.render_config = render_config if render_config is not None else RenderConfigRef()

        # Extract FMCW parameters
        self.center_freq = config.center_freq
        self.bandwidth = config.bandwidth
        self.chirp_duration = config.chirp_duration
        self.num_samples = config.num_adc_samples
        self.sample_rate = config.adc_sample_rate

        # Derived parameters
        # CRITICAL: carrierFrequency IS the start frequency (f0) in FMCW
        # RRTS uses MIN_FREQ = carrierFrequency directly (77 GHz)
        # The chirp sweeps from f0 to f0 + bandwidth
        self.min_freq = self.center_freq  # f0 = carrierFrequency (NOT center - BW/2!)
        self.slope = config.chirp_slope   # S = freqSlope from config directly

        # Wavelength computation
        self.wavelength = C / self.center_freq  # λ in meters
        self.lambda_squared = self.wavelength ** 2

        # CRITICAL: Time grid uses k/sample_rate, NOT linspace!
        # This is a key difference from mmIR
        self._build_time_grid()

        # Precompute radar equation constants if enabled
        if self.render_config.use_radar_equation:
            self._precompute_radar_constants()

        # Initialize BSDF evaluator based on config
        # All BRDF evaluation goes through this class
        bsdf_model = getattr(self.render_config, 'bsdf_model', 'rrts')

        # Read new material model config options
        enable_slab = getattr(self.render_config, 'enable_slab_fresnel', True)
        default_thick = getattr(self.render_config, 'default_thickness', 0.1)
        enable_cbs = getattr(self.render_config, 'enable_cbs', True)

        if bsdf_model == 'mmwave_scalar':
            # Physics-first KA+SPM+incoherent hybrid BSDF for 77 GHz (CSV-BSDF paper)
            mmwave_pol = getattr(self.render_config, 'mmwave_polarization', 'vertical')
            self.bsdf = BSDFmmWaveScalar(
                polarization=mmwave_pol,
                enable_incoherent=True,
                enable_slab_fresnel=enable_slab,
                default_thickness=default_thick,
                enable_cbs=enable_cbs,
            )
        elif bsdf_model == 'mmwave_jones':
            # Polarization-enabled Jones BSDF with complex Fresnel (magnitude only)
            mmwave_pol = getattr(self.render_config, 'mmwave_polarization', 'vertical')
            if mmwave_pol == 'vertical':
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            elif mmwave_pol == 'horizontal':
                tx_pol = mi.Vector3f(1, 0, 0)
                rx_pol = mi.Vector3f(1, 0, 0)
            else:
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            self.bsdf = BSDFmmWaveJones(
                tx_polarization=tx_pol,
                rx_polarization=rx_pol,
                enable_incoherent=True,
                use_fresnel_phase=False,
                enable_cbs=enable_cbs,
            )
        elif bsdf_model == 'mmwave_jones_phase':
            # Polarization-enabled Jones BSDF WITH Fresnel phase in coherent term
            mmwave_pol = getattr(self.render_config, 'mmwave_polarization', 'vertical')
            if mmwave_pol == 'vertical':
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            elif mmwave_pol == 'horizontal':
                tx_pol = mi.Vector3f(1, 0, 0)
                rx_pol = mi.Vector3f(1, 0, 0)
            else:
                tx_pol = mi.Vector3f(0, 0, 1)
                rx_pol = mi.Vector3f(0, 0, 1)
            self.bsdf = BSDFmmWaveJones(
                tx_polarization=tx_pol,
                rx_polarization=rx_pol,
                enable_incoherent=True,
                use_fresnel_phase=True,
                enable_cbs=enable_cbs,
            )
        else:
            raise ValueError(f"Unknown bsdf_model: '{bsdf_model}'. "
                           f"Supported: 'mmwave_scalar', 'mmwave_jones', 'mmwave_jones_phase'")

        # Material parameterization strategy (default: per-triangle)
        self.material_param: Optional[MaterialParameterization] = None

        # Free-space diffraction BSDF (initialized lazily when triangle hash is set)
        self.fsd_bsdf = None
        self.tri_hash = None
        # FSD CDF tables for importance sampling (set by renderer when direction_sampling='bsdf')
        self._fsd_cdf_tables = None
        self._fsd_sir_candidates = 8
        self.diffraction_config = getattr(self.render_config, 'diffraction_config', None)

        # Per-triangle material arrays for material-dependent diffraction (Phase A).
        # Set by renderer from triangle_materials when diffraction_config.material_opacity is True.
        # Shape: (eps_real_np [n_faces], eps_imag_np [n_faces]) or (None, None).
        self.diffraction_tri_eps: tuple = (None, None)

        # GPU-resident spatial hash data (set by renderer when use_gpu_apertures=True)
        self._gpu_spatial_hash = None

        # Per-triangle material array for Phase A specular BSDF evaluation.
        # Set by renderer. Shape: [n_triangles, 6] for physics mode, None otherwise.
        self.triangle_materials: Optional[np.ndarray] = None

        # GPU-resident material columns (lazy-cached from triangle_materials)
        self._gpu_material_cols: Optional[list] = None
        self._gpu_material_src_id: Optional[int] = None

    def _precompute_radar_constants(self):
        """
        Precompute constants for radar equation.

        Radar equation: Pr = (Pt * Gt * Gr * λ² * σ * L_ant) / ((4π)³ * R⁴)

        We precompute the constant parts so per-path computation is efficient:
        - radar_constant = sqrt(Pt * L_ant * λ²) / sqrt((4π)³)
        - adc_scale = sqrt(Z_0) * ADC_full_scale (converts sqrt(W) to ADC counts)
        - Per path: E = radar_constant * sqrt(Gt * Gr * σ_eff / R⁴) * rx_scale * adc_scale
        """
        rc = self.render_config

        # Convert TX power from dBm to Watts: P_t = 10^(Pt_dBm/10) / 1000
        self.Pt_watts = 10 ** (rc.Pt_dBm / 10) / 1000.0

        # Antenna loss (linear): L = 10^(-loss_dB/10) (loss is subtracted, so negative)
        self.antenna_loss_linear = 10 ** (-rc.antenna_loss_dB / 10)

        # (4π)² normalization factor for RX-centric Monte Carlo estimator
        # See rx_centric_radar_mc_estimators.tex Eq. 4:
        # P_r = (P_t λ²)/(4π)² × integral term
        # NOT (4π)³ - the RX solid-angle sampling absorbs one factor of 4π
        self.four_pi_squared = (4 * np.pi) ** 2

        # RX gain and dBFS scaling factor (amplitude domain uses /20, not /10)
        # E_scale = 10^((rx_gain_dB - dBm_to_dBFS_offset) / 20)
        self.rx_dBFS_scale = 10 ** ((rc.rx_gain_dB - rc.dBm_to_dBFS_offset) / 20)

        # ADC quantization scale: converts E-field from sqrt(Watts) to ADC counts
        # E-field in sqrt(W) needs to be converted to voltage: V = sqrt(P * Z_0) = E * sqrt(Z_0)
        # Then voltage to ADC counts: ADC = V * (ADC_full_scale / V_full_scale)
        # For TI radars, V_full_scale ≈ 1V corresponds to 0 dBFS = 13 dBm
        # Combined: adc_scale = sqrt(Z_0) * ADC_full_scale
        self.adc_scale = np.sqrt(rc.system_impedance) * rc.adc_full_scale

        # Monte Carlo normalization factor for solid angle sampling
        # This accounts for the hemisphere integral and sampling PDF compensation
        # NOTE: This factor is calibrated for n_hits_per_rx = 500 (the default).
        # For coherent phasor summation with deterministic phases, signal scales
        # approximately with sqrt(N), so using different n_hits_per_rx will affect
        # the absolute magnitude.
        self.mc_normalization = rc.mc_normalization

        # Combined constant: sqrt(Pt * L_ant * λ² * mc_norm) / sqrt((4π)²)
        # For RX-centric MC estimator: P_r ∝ (Pt λ²)/(4π)² × [G_t G_r c_t f V / d_t²]
        # The mc_normalization factor provides calibration for the sampling geometry
        self.radar_constant = np.sqrt(
            self.Pt_watts * self.antenna_loss_linear * self.lambda_squared *
            self.mc_normalization / self.four_pi_squared
        )

    def _build_time_grid(self):
        """
        Build time grid for phase computation.

        RRTS uses: t[k] = k / sample_rate
        mmIR uses: t = linspace(0, T_chirp, K)

        These differ by ~6.25% which causes range offset.
        """
        K = self.num_samples

        # RRTS time grid: t[k] = k / sample_rate
        self.t_grid = np.arange(K) / self.sample_rate
        self._t_grid_dr = mi.Float(self.t_grid)  # Cache on GPU once

        # For comparison - mmIR linspace would be:
        # self.t_grid_linspace = np.linspace(0, self.chirp_duration, K)
        # dt_rrts = 1/sample_rate = 125 ns (for 8 MHz)
        # dt_linspace = T_chirp/K = 132.8 ns (for 8.5us chirp, 64 samples)

    def precompute_phasors(self, cached_geom: 'CachedGeometry'):
        """Pre-compute phase-related arrays that don't depend on materials.

        Must be called after CachedGeometry is built (Phase A) and before
        fast Phase B synthesis. Only valid when pose/geometry are frozen.
        Stores cos(phi)/sin(phi) at [n_total × K] in cached_geom (~2.4 GB).
        """
        n_total = cached_geom.n_total
        n_rx = cached_geom.n_rx
        K = self.num_samples

        # --- Diffuse path phasors ---
        R_total = cached_geom.d_rx_to_hit + cached_geom.d_hit_to_tx
        tau = R_total / mi.Float(C)
        TWO_PI = 2.0 * np.pi
        phi_const = mi.Float(TWO_PI * self.min_freq) * tau
        phi_slope = mi.Float(TWO_PI * self.slope) * tau

        phi_const_3d = dr.tile(phi_const, K)
        phi_slope_3d = dr.tile(phi_slope, K)
        t_k = dr.repeat(self._t_grid_dr, n_total)
        phi = phi_const_3d + phi_slope_3d * t_k

        cached_geom.cached_cos_phi = dr.cos(phi)
        cached_geom.cached_sin_phi = dr.sin(phi)
        cached_geom.cached_active_3d = dr.tile(cached_geom.active, K)

        # Flat indices for scatter_add
        tx_idx_3d = dr.tile(cached_geom.tx_idx, K)
        rx_idx_3d = dr.tile(cached_geom.rx_idx, K)
        base_idx = dr.arange(mi.UInt32, n_total * K)
        k_idx = base_idx // mi.UInt32(n_total)
        cached_geom.cached_flat_idx = (
            tx_idx_3d * mi.UInt32(n_rx * K)
            + rx_idx_3d * mi.UInt32(K)
            + k_idx
        )

        # Path index for weight gather (replaces dr.tile on weight)
        cached_geom.cached_path_idx = base_idx % mi.UInt32(n_total)

        # Force evaluation to materialize GPU arrays
        dr.eval(cached_geom.cached_cos_phi, cached_geom.cached_sin_phi,
                cached_geom.cached_flat_idx, cached_geom.cached_active_3d,
                cached_geom.cached_path_idx)

        # --- Specular path phasors (if present, ~8 MB — negligible) ---
        n_spec = cached_geom.specular_n_paths
        if n_spec > 0 and cached_geom.specular_d_tx is not None:
            sp_R = cached_geom.specular_d_tx + cached_geom.specular_d_rx
            sp_tau = sp_R / mi.Float(C)
            sp_phi_const = mi.Float(TWO_PI * self.min_freq) * sp_tau
            sp_phi_slope = mi.Float(TWO_PI * self.slope) * sp_tau

            sp_phi_c3 = dr.tile(sp_phi_const, K)
            sp_phi_s3 = dr.tile(sp_phi_slope, K)
            sp_tk = dr.repeat(self._t_grid_dr, n_spec)
            sp_phi = sp_phi_c3 + sp_phi_s3 * sp_tk

            cached_geom.sp_cached_cos_phi = dr.cos(sp_phi)
            cached_geom.sp_cached_sin_phi = dr.sin(sp_phi)
            cached_geom.sp_cached_valid_3d = dr.tile(cached_geom.specular_valid, K)

            sp_tx3 = dr.tile(cached_geom.specular_tx_idx, K)
            sp_rx3 = dr.tile(cached_geom.specular_rx_idx, K)
            sp_base = dr.arange(mi.UInt32, n_spec * K)
            sp_k = sp_base // mi.UInt32(n_spec)
            cached_geom.sp_cached_flat_idx = (
                sp_tx3 * mi.UInt32(n_rx * K)
                + sp_rx3 * mi.UInt32(K)
                + sp_k
            )
            cached_geom.sp_cached_path_idx = sp_base % mi.UInt32(n_spec)

            dr.eval(cached_geom.sp_cached_cos_phi, cached_geom.sp_cached_sin_phi,
                    cached_geom.sp_cached_flat_idx, cached_geom.sp_cached_valid_3d,
                    cached_geom.sp_cached_path_idx)

    # =========================================================================
    # REMOVED: synthesize() and _check_visibility_batch()
    # These were loop-based implementations that are 160-210x slower than
    # synthesize_vectorized(). All BRDF evaluation uses self.bsdf.eval_f_cos()
    # =========================================================================
