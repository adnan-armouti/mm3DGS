"""
Cached geometry dataclasses for two-phase differentiable rendering.

ADCResult: Forward render output container.
ADCComponentResult: Per-lobe BSDF component breakdown for visualization.
CachedGeometry: Phase A cache consumed by Phase B differentiable re-synthesis.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ADCResult:
    """
    Result of ADC synthesis.

    Attributes:
        adc_real: Real component [n_tx, n_rx, n_samples]
        adc_imag: Imaginary component [n_tx, n_rx, n_samples]
        n_tx: Number of TX elements
        n_rx: Number of RX elements
        n_samples: Number of ADC samples
        total_paths: Total number of visible paths accumulated
    """
    adc_real: np.ndarray
    adc_imag: np.ndarray
    n_tx: int
    n_rx: int
    n_samples: int
    total_paths: int

    def get_complex(self) -> np.ndarray:
        """Get complex ADC as [n_tx, n_rx, n_samples]."""
        return self.adc_real + 1j * self.adc_imag

    def get_ri_array(self) -> np.ndarray:
        """Get real/imag stacked as [n_tx, n_rx, n_samples, 2]."""
        return np.stack([self.adc_real, self.adc_imag], axis=-1)


@dataclass
@dataclass
class ADCComponentResult:
    """
    Result of component-level ADC synthesis for visualization.

    Contains ADC arrays for each BSDF component:
    - ka: KA (Kirchhoff Approximation) specular lobe
    - spm: SPM (Small Perturbation Method) diffuse lobe
    - directive: Directive incoherent lobe
    - broad: Broad diffuse lobe
    - coherent: Combined coherent (KA + SPM with blend)
    - incoherent: Combined incoherent (directive + broad with blend)
    - diffraction: Free-space diffraction BSDF (fsdBSDF) contribution
    - total: Total BSDF (same as ADCResult)

    Each component is stored as complex [n_tx, n_rx, n_samples].
    """
    ka: np.ndarray
    spm: np.ndarray
    directive: np.ndarray
    broad: np.ndarray
    coherent: np.ndarray
    incoherent: np.ndarray
    diffraction: np.ndarray
    total: np.ndarray
    n_tx: int
    n_rx: int
    n_samples: int
    total_paths: int

    def get_component(self, name: str) -> np.ndarray:
        """Get a specific component by name."""
        return getattr(self, name)

    def get_all_components(self) -> dict:
        """Get all components as a dictionary."""
        return {
            'ka': self.ka,
            'spm': self.spm,
            'directive': self.directive,
            'broad': self.broad,
            'coherent': self.coherent,
            'incoherent': self.incoherent,
            'diffraction': self.diffraction,
            'total': self.total,
        }


@dataclass
@dataclass
class CachedGeometry:
    """
    Cached geometry from Phase A (non-differentiable) render.

    Stores all material-independent intermediate values needed to re-synthesize
    ADC signals differentiably in Phase B. All DrJit arrays are detached
    (no grad tracking) — they serve as constants in the differentiable path.

    Shapes: n_total = n_valid × n_tx × n_rx (MIMO-expanded)
    """
    hit_P_expanded: 'mi.Point3f'       # [n_total] hit positions
    hit_N_expanded: 'mi.Vector3f'      # [n_total] hit normals (after double-sided flip)
    dir_hit_to_rx: 'mi.Vector3f'       # [n_total] outgoing direction (wo)
    dir_hit_to_tx: 'mi.Vector3f'       # [n_total] incident direction (wi)
    d_rx_to_hit: 'mi.Float'            # [n_total] RX→hit distance
    d_hit_to_tx: 'mi.Float'            # [n_total] hit→TX distance
    active: 'mi.Bool'                  # [n_total] visibility + geometry validity mask
    tx_idx: 'mi.UInt32'                # [n_total] TX element index per path
    rx_idx: 'mi.UInt32'                # [n_total] RX element index per path
    hit_prim_ids: 'mi.UInt32'          # [n_total] triangle primitive IDs for dr.gather
    antenna_gain_combined: Optional['mi.Float']  # [n_total] combined TX×RX gain (linear)
    mc_correction: Optional['mi.Float']          # [n_total] 1/(pdf × n_attempted)
    n_tx: int
    n_rx: int
    n_valid: int
    n_total: int                       # = n_valid × n_tx × n_rx

    # Per-vertex parameterization support (populated when mode='per_vertex'):
    hit_bary_u: Optional['mi.Float'] = None        # [n_total] barycentric u coord
    hit_bary_v: Optional['mi.Float'] = None        # [n_total] barycentric v coord
    hit_vertex_ids_0: Optional['mi.UInt32'] = None  # [n_total] vertex index 0
    hit_vertex_ids_1: Optional['mi.UInt32'] = None  # [n_total] vertex index 1
    hit_vertex_ids_2: Optional['mi.UInt32'] = None  # [n_total] vertex index 2

    # For pose recomputation (Fix 1): per-element TX/RX arrays
    tx_positions: Optional['mi.Point3f'] = None      # [n_tx] per-element TX positions
    rx_positions: Optional['mi.Point3f'] = None      # [n_rx] per-element RX positions

    # For pattern recomputation (Fix 2): per-element boresights
    tx_boresights: Optional['mi.Vector3f'] = None    # [n_tx] per-element TX boresights
    rx_boresights: Optional['mi.Vector3f'] = None    # [n_rx] per-element RX boresights

    # For image method / SMS differentiable re-evaluation
    specular_hit_P: Optional['mi.Point3f'] = None    # [n_spec] converged specular positions
    specular_hit_N: Optional['mi.Vector3f'] = None   # [n_spec] surface normals
    specular_dir_to_tx: Optional['mi.Vector3f'] = None  # [n_spec]
    specular_dir_to_rx: Optional['mi.Vector3f'] = None  # [n_spec]
    specular_d_tx: Optional['mi.Float'] = None       # [n_spec]
    specular_d_rx: Optional['mi.Float'] = None       # [n_spec]
    specular_cos_theta_i: Optional['mi.Float'] = None  # [n_spec]
    specular_cos_theta_r: Optional['mi.Float'] = None  # [n_spec]
    specular_A_tri: Optional['mi.Float'] = None      # [n_spec]
    specular_antenna_gain: Optional['mi.Float'] = None  # [n_spec] combined TX×RX gain
    specular_tx_idx: Optional['mi.UInt32'] = None    # [n_spec]
    specular_rx_idx: Optional['mi.UInt32'] = None    # [n_spec]
    specular_valid: Optional['mi.Bool'] = None       # [n_spec]
    specular_prim_ids: Optional['mi.UInt32'] = None  # [n_spec] triangle IDs for material gather
    specular_n_paths: int = 0

    # IFT vertex gradient support (populated by SMS, not image method)
    specular_grad_info: Optional[object] = None       # SpecularGradInfo from SMS
    specular_bary_u: Optional['mi.Float'] = None      # [n_spec] barycentric u
    specular_bary_v: Optional['mi.Float'] = None      # [n_spec] barycentric v
    specular_vertex_ids_0: Optional['mi.UInt32'] = None  # [n_spec] vertex index 0
    specular_vertex_ids_1: Optional['mi.UInt32'] = None  # [n_spec] vertex index 1
    specular_vertex_ids_2: Optional['mi.UInt32'] = None  # [n_spec] vertex index 2

    # V2: Pre-computed phasor cache for fast Phase B synthesis
    # Populated by precompute_phasors(), valid when pose/geometry are frozen
    cached_cos_phi: Optional['mi.Float'] = None       # [n_total × K]
    cached_sin_phi: Optional['mi.Float'] = None       # [n_total × K]
    cached_flat_idx: Optional['mi.UInt32'] = None      # [n_total × K]
    cached_active_3d: Optional['mi.Bool'] = None       # [n_total × K]
    cached_path_idx: Optional['mi.UInt32'] = None      # [n_total × K]
    # Specular phasor cache (~8 MB, negligible)
    sp_cached_cos_phi: Optional['mi.Float'] = None     # [n_spec × K]
    sp_cached_sin_phi: Optional['mi.Float'] = None     # [n_spec × K]
    sp_cached_flat_idx: Optional['mi.UInt32'] = None   # [n_spec × K]
    sp_cached_valid_3d: Optional['mi.Bool'] = None     # [n_spec × K]
    sp_cached_path_idx: Optional['mi.UInt32'] = None   # [n_spec × K]

    # Diffraction cache (populated in Phase A, reused in Phase B)
    diff_flat: Optional[object] = None             # FlatEdgeData from Phase A
    diff_eval_path_idx: Optional[np.ndarray] = None  # [N_eval] int32 — flat path indices
    diff_eval_ap_idx: Optional[np.ndarray] = None    # [N_eval] int32 — aperture indices
    diff_eval_wi: Optional[np.ndarray] = None        # [N_eval, 3] float64 — TX directions
    diff_n_diffraction_hits: int = 0
    diff_beta_np: Optional[np.ndarray] = None        # [n_total] float64 — energy borrowing
    diff_cos_theta_in: Optional[np.ndarray] = None   # [n_total] float32 — clamped cos(θ_in) for diffraction

