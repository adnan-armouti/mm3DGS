"""
Virtual screen aperture construction for the fsdBSDF.

Implements Algorithm 1 of Steinberg et al. "A Free-Space Diffraction BSDF" (2024):
  1. Find triangles near the hit point (via TriangleSpatialHash)
  2. Cull back-facing triangles
  3. Project front-facing triangles onto virtual screen (perpendicular to traced ray)
  4. Tessellate long projected edges (ensure PLA accuracy)
  5. Extract boundary edges (edges with exactly one front-facing neighbor)
  6. Compute beam amplitudes a_j, b_j at edge endpoints
  7. Compute obstacle power P_A, central lobe ψ(0), and clamped diffracted power P̂_A

Convention: the virtual screen is perpendicular to the RX→hit traced ray.
  - z-axis of virtual screen frame: wo (hit → RX direction)
  - "front-facing" triangles: dot(wo, face_normal) > 0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .triangle_search import TriangleSpatialHash


@dataclass
class FsdEdge:
    """A single boundary edge on the virtual screen."""
    e: np.ndarray       # [2] 2D edge vector (u2 - u1) on screen
    v: np.ndarray       # [2] 2D edge midpoint on screen
    a: complex          # Complex beam amplitude at first vertex (includes √opacity)
    b: complex          # Complex beam amplitude at second vertex (includes √opacity)
    Phat: float         # Edge-diffracted power (for sampling weight)
    Phat_accum: float   # Cumulative power (for binary search in SIR)
    opacity: float = 1.0      # Fresnel opacity [0, 1] (for Phase B re-computation)
    face_idx: int = -1         # Global face index of owning triangle
    cos_theta: float = 1.0     # cos(θ_i) at this edge's triangle surface
    # Per-edge barycentric coordinates of the edge midpoint within the owning triangle.
    # Used for per-vertex material interpolation in Phase B (more accurate than centroid).
    # Convention: param = (1-bary_u-bary_v)*p[vi0] + bary_u*p[vi1] + bary_v*p[vi2]
    bary_u: float = -1.0       # -1 = not set (fall back to per-face gather)
    bary_v: float = -1.0
    vert_idx_0: int = -1       # Global vertex indices of owning triangle
    vert_idx_1: int = -1
    vert_idx_2: int = -1
    # Jones polarization data (for Phase B complex correction)
    jones_real: float = 1.0    # Re(E_jones) from Phase A
    jones_imag: float = 0.0    # Im(E_jones) from Phase A
    tx_amp_s: float = 0.0      # TX polarization s-component (frozen geometry)
    tx_amp_p: float = 0.0      # TX polarization p-component (frozen geometry)
    rx_amp_s: float = 0.0      # RX polarization s-component (frozen geometry)
    rx_amp_p: float = 0.0      # RX polarization p-component (frozen geometry)


@dataclass
class FsdAperture:
    """
    Complete aperture data for one hit point.

    This is the output of Algorithm 1 and the input to the fsdBSDF eval.
    """
    edges: List[FsdEdge] = field(default_factory=list)

    # Virtual screen frame (columns: tangent, bitangent, wo)
    tangent: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bitangent: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wo_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))  # z-axis = toward RX

    # Powers (paper notation)
    P_A: float = 0.0           # Total obstacle power P_Ā^(PL) [Eq. 31] (opacity-weighted)
    psi_0: float = 0.0         # Central lobe peak amplitude ψ(0) [Eq. 34]
    P_central: float = 0.0     # Power in central (0th-order) lobe P̃_central [Eq. 35]
    P_A_hat: float = 0.0       # Clamped diffracted power P̂_Ā [Eq. 36] (opacity-weighted)
    P_A_hat_bare: float = 0.0  # Bare geometric P̂_Ā (no material modulation) — for beta
    P_A_bare: float = 0.0     # Bare geometric P_Ā (no material modulation) — for normalization (matches reference)
    sum_Phat_j: float = 0.0    # Sum of edge powers (for PDF normalization)

    # Covariance of the 0th-order lobe (stored as [Σ_xx, Σ_xy, Σ_yy])
    Sigma0: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Wavenumber and beam parameters
    k: float = 0.0
    beam_sigma: float = 0.0

    # Cached numpy arrays for batch packing (populated by construct_aperture)
    _cached_e: Optional[np.ndarray] = field(default=None, repr=False)   # [n, 2] float64
    _cached_v: Optional[np.ndarray] = field(default=None, repr=False)   # [n, 2] float64
    _cached_a: Optional[np.ndarray] = field(default=None, repr=False)   # [n] complex128
    _cached_b: Optional[np.ndarray] = field(default=None, repr=False)   # [n] complex128
    _cached_opacity: Optional[np.ndarray] = field(default=None, repr=False)    # [n] float64
    _cached_face_idx: Optional[np.ndarray] = field(default=None, repr=False)   # [n] int32
    _cached_cos_theta: Optional[np.ndarray] = field(default=None, repr=False)  # [n] float64
    # Per-edge barycentric + vertex data for per-vertex material interpolation
    _cached_bary_u: Optional[np.ndarray] = field(default=None, repr=False)     # [n] float64
    _cached_bary_v: Optional[np.ndarray] = field(default=None, repr=False)     # [n] float64
    _cached_vert_idx_0: Optional[np.ndarray] = field(default=None, repr=False) # [n] int32
    _cached_vert_idx_1: Optional[np.ndarray] = field(default=None, repr=False) # [n] int32
    _cached_vert_idx_2: Optional[np.ndarray] = field(default=None, repr=False) # [n] int32
    # Jones polarization cached arrays
    _cached_jones_real: Optional[np.ndarray] = field(default=None, repr=False)  # [n] float64
    _cached_jones_imag: Optional[np.ndarray] = field(default=None, repr=False)  # [n] float64
    _cached_tx_amp_s: Optional[np.ndarray] = field(default=None, repr=False)    # [n] float64
    _cached_tx_amp_p: Optional[np.ndarray] = field(default=None, repr=False)    # [n] float64
    _cached_rx_amp_s: Optional[np.ndarray] = field(default=None, repr=False)    # [n] float64
    _cached_rx_amp_p: Optional[np.ndarray] = field(default=None, repr=False)    # [n] float64

    @property
    def has_diffraction(self) -> bool:
        """True if this aperture has boundary edges and non-zero diffracted power."""
        return len(self.edges) > 0 and self.P_A_hat > 0

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def _cache_edge_arrays(self):
        """Cache edge data as contiguous numpy arrays (called once after construction)."""
        if not self.edges:
            return
        n = len(self.edges)
        self._cached_e = np.empty((n, 2), dtype=np.float64)
        self._cached_v = np.empty((n, 2), dtype=np.float64)
        self._cached_a = np.empty(n, dtype=np.complex128)
        self._cached_b = np.empty(n, dtype=np.complex128)
        self._cached_opacity = np.empty(n, dtype=np.float64)
        self._cached_face_idx = np.empty(n, dtype=np.int32)
        self._cached_cos_theta = np.empty(n, dtype=np.float64)
        self._cached_bary_u = np.empty(n, dtype=np.float64)
        self._cached_bary_v = np.empty(n, dtype=np.float64)
        self._cached_vert_idx_0 = np.empty(n, dtype=np.int32)
        self._cached_vert_idx_1 = np.empty(n, dtype=np.int32)
        self._cached_vert_idx_2 = np.empty(n, dtype=np.int32)
        self._cached_jones_real = np.empty(n, dtype=np.float64)
        self._cached_jones_imag = np.empty(n, dtype=np.float64)
        self._cached_tx_amp_s = np.empty(n, dtype=np.float64)
        self._cached_tx_amp_p = np.empty(n, dtype=np.float64)
        self._cached_rx_amp_s = np.empty(n, dtype=np.float64)
        self._cached_rx_amp_p = np.empty(n, dtype=np.float64)
        for j, edge in enumerate(self.edges):
            self._cached_e[j] = edge.e
            self._cached_v[j] = edge.v
            self._cached_a[j] = edge.a
            self._cached_b[j] = edge.b
            self._cached_opacity[j] = edge.opacity
            self._cached_face_idx[j] = edge.face_idx
            self._cached_cos_theta[j] = edge.cos_theta
            self._cached_bary_u[j] = edge.bary_u
            self._cached_bary_v[j] = edge.bary_v
            self._cached_vert_idx_0[j] = edge.vert_idx_0
            self._cached_vert_idx_1[j] = edge.vert_idx_1
            self._cached_vert_idx_2[j] = edge.vert_idx_2
            self._cached_jones_real[j] = edge.jones_real
            self._cached_jones_imag[j] = edge.jones_imag
            self._cached_tx_amp_s[j] = edge.tx_amp_s
            self._cached_tx_amp_p[j] = edge.tx_amp_p
            self._cached_rx_amp_s[j] = edge.rx_amp_s
            self._cached_rx_amp_p[j] = edge.rx_amp_p

    def get_edge_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Return edge data as contiguous numpy arrays for batch processing.

        Returns:
            (e_arr [n,2], v_arr [n,2], a_arr [n] complex, b_arr [n] complex)
            or None if no edges.
        """
        if not self.edges:
            return None
        n = len(self.edges)
        e_arr = np.empty((n, 2), dtype=np.float64)
        v_arr = np.empty((n, 2), dtype=np.float64)
        a_arr = np.empty(n, dtype=np.complex128)
        b_arr = np.empty(n, dtype=np.complex128)
        for j, edge in enumerate(self.edges):
            e_arr[j] = edge.e
            v_arr[j] = edge.v
            a_arr[j] = edge.a
            b_arr[j] = edge.b
        return e_arr, v_arr, a_arr, b_arr


# ============================================================================
# Precomputed constants
# ============================================================================

# Numerically integrated total powers (Eq. 32-33, Supplemental S2.36-S2.37)
_INTEGRAL_1 = 0.0045255085   # ∫(1-χ)|α₁|² for Phat
_INTEGRAL_2 = 0.114875434    # ∫(1-χ)|α₂|² for Phat
_INTEGRAL_1_FULL = 0.004973515  # ∫|α₁|² for full Pj (with central lobe)
_INTEGRAL_2_FULL = 14.2569085397  # ∫|α₂|² for full Pj (with central lobe)


def _phi_gaussian(u: np.ndarray, z: float, beam_sigma: float) -> float:
    """
    Gaussian beam profile (peak amplitude function).

    φ(u, z) = exp(-(|u|² + z²) / (4σ²)) / (√(2π) σ)

    Args:
        u: 2D position on virtual screen [2].
        z: Depth along propagation direction.
        beam_sigma: Beam width σ in meters.

    Returns:
        Scalar beam amplitude.
    """
    r2 = float(u[0] ** 2 + u[1] ** 2 + z ** 2)
    return np.exp(-0.25 * r2 / (beam_sigma ** 2)) / (np.sqrt(2 * np.pi) * beam_sigma)


def _fresnel_opacity_numpy(eps_real: float, eps_imag: float, cos_theta_i: float) -> float:
    """
    Compute Fresnel power reflectance averaged over polarization (numpy, scalar).

    opacity = (|R_s|² + |R_p|²) / 2

    Uses full complex Fresnel equations for absorbing media.
    Matches the DrJit version in fsd_bsdf.py for cross-validation.

    Args:
        eps_real: Real relative permittivity ε'.
        eps_imag: Imaginary permittivity ε'' (loss).
        cos_theta_i: Cosine of incidence angle (positive).

    Returns:
        Scalar opacity in [0, 1].
    """
    eps_mag = np.sqrt(eps_real**2 + eps_imag**2)
    n = np.sqrt((eps_mag + eps_real) / 2.0)
    kappa = np.sqrt(max(0.0, (eps_mag - eps_real) / 2.0))

    sin2 = 1.0 - cos_theta_i**2
    n2_real = n * n - kappa * kappa
    n2_imag = -2.0 * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = np.sqrt(xi_real**2 + xi_imag**2)
    xi_arg = np.arctan2(xi_imag, xi_real)
    a = np.sqrt(xi_mag) * np.cos(xi_arg / 2.0)
    b = np.sqrt(xi_mag) * np.sin(xi_arg / 2.0)

    # R_s = |(cos_θ - (a+jb)) / (cos_θ + (a+jb))|²
    rs_num = (cos_theta_i - a)**2 + b**2
    rs_den = max((cos_theta_i + a)**2 + b**2, 1e-10)
    R_s = rs_num / rs_den

    # R_p = |(ñ²cos_θ - (a+jb)) / (ñ²cos_θ + (a+jb))|²
    n2cos_r = n2_real * cos_theta_i
    n2cos_i = n2_imag * cos_theta_i
    rp_num = (n2cos_r - a)**2 + (n2cos_i - b)**2
    rp_den = max((n2cos_r + a)**2 + (n2cos_i + b)**2, 1e-10)
    R_p = rp_num / rp_den

    return float(np.clip((R_s + R_p) / 2.0, 0.0, 1.0))


def _jones_reflectance_numpy(
    eps_real: float,
    eps_imag: float,
    cos_theta_i: float,
    wo: np.ndarray,           # [3] direction toward RX
    face_normal: np.ndarray,  # [3] surface normal (unit)
    tx_pol: np.ndarray,       # [3] TX polarization direction (unit)
    rx_pol: np.ndarray,       # [3] RX polarization direction (unit)
) -> tuple:
    """
    Compute complex Jones reflectance coefficient for a diffracting edge.

    Full pipeline: s/p basis → project TX → complex Fresnel → apply Jones →
    project RX → scalar complex coefficient.

    Args:
        eps_real: Real relative permittivity ε'.
        eps_imag: Imaginary permittivity ε'' (loss).
        cos_theta_i: Cosine of incidence angle (positive).
        wo: Direction from hit point toward RX (unit).
        face_normal: Surface normal of owning triangle (unit).
        tx_pol: TX antenna E-field polarization direction (unit).
        rx_pol: RX antenna E-field polarization direction (unit).

    Returns:
        (jones_real, jones_imag, jones_power, tx_s, tx_p, rx_s, rx_p):
            Complex Jones coefficient, its squared magnitude, and the
            frozen TX/RX s/p projections (for Phase B re-computation).
    """
    # --- Step 1: Compute s/p polarization basis ---
    # s = wo × face_normal (perpendicular to plane of incidence)
    s_unnorm = np.cross(wo, face_normal)
    s_len = np.linalg.norm(s_unnorm)
    if s_len < 1e-6:
        # Grazing incidence: use fallback
        fallback = np.cross(wo, np.array([1.0, 0.0, 0.0]))
        fl = np.linalg.norm(fallback)
        if fl < 1e-6:
            fallback = np.cross(wo, np.array([0.0, 1.0, 0.0]))
            fl = np.linalg.norm(fallback)
        s = fallback / max(fl, 1e-10)
    else:
        s = s_unnorm / s_len

    # p = s × wo (in plane of incidence, perpendicular to wo)
    p = np.cross(s, wo)
    p_len = np.linalg.norm(p)
    p = p / max(p_len, 1e-10)

    # --- Step 2: Project TX polarization onto s/p ---
    tx_s = float(np.dot(tx_pol, s))
    tx_p = float(np.dot(tx_pol, p))

    # --- Step 3: Compute complex Fresnel r_s, r_p ---
    eps_mag = np.sqrt(eps_real**2 + eps_imag**2)
    n = np.sqrt((eps_mag + eps_real) / 2.0)
    kappa = np.sqrt(max(0.0, (eps_mag - eps_real) / 2.0))

    sin2 = 1.0 - cos_theta_i**2
    n2_real = n * n - kappa * kappa
    n2_imag = -2.0 * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = np.sqrt(xi_real**2 + xi_imag**2)
    xi_arg = np.arctan2(xi_imag, xi_real)
    a = np.sqrt(xi_mag) * np.cos(xi_arg / 2.0)
    b = np.sqrt(xi_mag) * np.sin(xi_arg / 2.0)

    # r_s = (cos_θ - (a+jb)) / (cos_θ + (a+jb))
    rs_num = complex(cos_theta_i - a, -b)
    rs_den = complex(cos_theta_i + a, b)
    r_s = rs_num / rs_den if abs(rs_den) > 1e-10 else 0.0 + 0.0j

    # r_p = (ñ²cos_θ - (a+jb)) / (ñ²cos_θ + (a+jb))
    n2cos = complex(n2_real * cos_theta_i, n2_imag * cos_theta_i)
    rp_num = n2cos - complex(a, b)
    rp_den = n2cos + complex(a, b)
    r_p = rp_num / rp_den if abs(rp_den) > 1e-10 else 0.0 + 0.0j

    # --- Step 4: Apply Jones reflection ---
    # E_s_out = r_s * tx_s, E_p_out = r_p * tx_p (complex)
    E_s_out = r_s * tx_s
    E_p_out = r_p * tx_p

    # --- Step 5: Outgoing s/p basis ---
    s_out = s  # s preserved across reflection
    p_out = np.cross(s_out, wo)
    p_out_len = np.linalg.norm(p_out)
    p_out = p_out / max(p_out_len, 1e-10)

    # --- Step 6: Project RX polarization onto outgoing s/p ---
    rx_s = float(np.dot(rx_pol, s_out))
    rx_p = float(np.dot(rx_pol, p_out))

    # --- Step 7: Received field ---
    E_jones = E_s_out * rx_s + E_p_out * rx_p

    jones_real = float(E_jones.real)
    jones_imag = float(E_jones.imag)
    jones_power = jones_real**2 + jones_imag**2

    return jones_real, jones_imag, jones_power, tx_s, tx_p, rx_s, rx_p


def _fresnel_opacity_numpy_batch(
    eps_real: np.ndarray,
    eps_imag: np.ndarray,
    cos_theta_i: np.ndarray,
) -> np.ndarray:
    """
    Batch-vectorized Fresnel opacity. All inputs are [K] float64 arrays.
    Returns [K] opacity in [0, 1].
    """
    eps_mag = np.sqrt(eps_real**2 + eps_imag**2)
    n = np.sqrt((eps_mag + eps_real) / 2.0)
    kappa = np.sqrt(np.maximum(0.0, (eps_mag - eps_real) / 2.0))

    sin2 = 1.0 - cos_theta_i**2
    n2_real = n * n - kappa * kappa
    n2_imag = -2.0 * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = np.sqrt(xi_real**2 + xi_imag**2)
    xi_arg = np.arctan2(xi_imag, xi_real)
    a = np.sqrt(xi_mag) * np.cos(xi_arg / 2.0)
    b = np.sqrt(xi_mag) * np.sin(xi_arg / 2.0)

    rs_num = (cos_theta_i - a)**2 + b**2
    rs_den = np.maximum((cos_theta_i + a)**2 + b**2, 1e-10)
    R_s = rs_num / rs_den

    n2cos_r = n2_real * cos_theta_i
    n2cos_i = n2_imag * cos_theta_i
    rp_num = (n2cos_r - a)**2 + (n2cos_i - b)**2
    rp_den = np.maximum((n2cos_r + a)**2 + (n2cos_i + b)**2, 1e-10)
    R_p = rp_num / rp_den

    return np.clip((R_s + R_p) / 2.0, 0.0, 1.0)


def _jones_reflectance_numpy_batch(
    eps_real: np.ndarray,       # [K]
    eps_imag: np.ndarray,       # [K]
    cos_theta_i: np.ndarray,    # [K]
    wo: np.ndarray,             # [3] direction toward RX
    fn: np.ndarray,             # [K, 3] unit face normals
    tx_pol: np.ndarray,         # [3] TX polarization direction
    rx_pol: np.ndarray,         # [3] RX polarization direction
) -> tuple:
    """
    Batch-vectorized Jones reflectance. Returns tuple of [K] arrays:
    (jones_real, jones_imag, jones_power, tx_s, tx_p, rx_s, rx_p).
    """
    K = len(eps_real)

    # s = wo × fn: [K, 3]
    s_unnorm = np.cross(wo[np.newaxis, :], fn)  # [K, 3]
    s_len = np.linalg.norm(s_unnorm, axis=1, keepdims=True)  # [K, 1]

    # Fallback for grazing incidence
    fallback = np.cross(wo, np.array([1.0, 0.0, 0.0]))
    fl = np.linalg.norm(fallback)
    if fl < 1e-6:
        fallback = np.cross(wo, np.array([0.0, 1.0, 0.0]))
        fl = np.linalg.norm(fallback)
    fallback = fallback / max(fl, 1e-10)

    need_fallback = (s_len.ravel() < 1e-6)  # [K]
    s = np.where(need_fallback[:, np.newaxis],
                 fallback[np.newaxis, :],
                 s_unnorm / np.maximum(s_len, 1e-10))

    # p = s × wo: [K, 3]
    p = np.cross(s, wo[np.newaxis, :])
    p_len = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / np.maximum(p_len, 1e-10)

    # TX projections: [K]
    tx_s = np.einsum('j,ij->i', tx_pol, s)
    tx_p = np.einsum('j,ij->i', tx_pol, p)

    # Complex Fresnel r_s, r_p
    eps_mag = np.sqrt(eps_real**2 + eps_imag**2)
    n = np.sqrt((eps_mag + eps_real) / 2.0)
    kappa = np.sqrt(np.maximum(0.0, (eps_mag - eps_real) / 2.0))

    sin2 = 1.0 - cos_theta_i**2
    n2_real = n * n - kappa * kappa
    n2_imag = -2.0 * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = np.sqrt(xi_real**2 + xi_imag**2)
    xi_arg = np.arctan2(xi_imag, xi_real)
    a = np.sqrt(xi_mag) * np.cos(xi_arg / 2.0)
    b = np.sqrt(xi_mag) * np.sin(xi_arg / 2.0)

    # r_s = (cos_θ - (a+jb)) / (cos_θ + (a+jb))
    rs_num = (cos_theta_i - a) - 1j * b
    rs_den = (cos_theta_i + a) + 1j * b
    r_s = np.where(np.abs(rs_den) > 1e-10, rs_num / rs_den, 0.0 + 0.0j)

    # r_p = (ñ²cos_θ - (a+jb)) / (ñ²cos_θ + (a+jb))
    n2cos = (n2_real + 1j * n2_imag) * cos_theta_i
    rp_num = n2cos - (a + 1j * b)
    rp_den = n2cos + (a + 1j * b)
    r_p = np.where(np.abs(rp_den) > 1e-10, rp_num / rp_den, 0.0 + 0.0j)

    # Jones: E_s_out = r_s * tx_s, E_p_out = r_p * tx_p
    E_s_out = r_s * tx_s
    E_p_out = r_p * tx_p

    # RX projections (outgoing s/p basis)
    s_out = s
    p_out = np.cross(s_out, wo[np.newaxis, :])
    p_out_len = np.linalg.norm(p_out, axis=1, keepdims=True)
    p_out = p_out / np.maximum(p_out_len, 1e-10)

    rx_s = np.einsum('j,ij->i', rx_pol, s_out)
    rx_p = np.einsum('j,ij->i', rx_pol, p_out)

    # E_jones = E_s_out * rx_s + E_p_out * rx_p
    E_jones = E_s_out * rx_s + E_p_out * rx_p

    jones_real = E_jones.real.astype(np.float64)
    jones_imag = E_jones.imag.astype(np.float64)
    jones_power = jones_real**2 + jones_imag**2

    return jones_real, jones_imag, jones_power, tx_s, tx_p, rx_s, rx_p


def _triangle_projected_area(u1, u2, u3) -> float:
    """Signed area of a 2D triangle via shoelace formula."""
    return abs(
        -u1[1] * u2[0] + u1[0] * u2[1]
        + u1[1] * u3[0] - u2[1] * u3[0]
        - u1[0] * u3[1] + u2[0] * u3[1]
    )


def _Pt(u1, u2, u3, ph1, ph2, ph3) -> float:
    """
    Power incident on a projected triangle [Eq. 31 / reference Pt()].

    P_t = |signed_area| × (φ₁² + φ₂² + φ₃² + φ₁φ₂ + φ₁φ₃ + φ₂φ₃) / 12
    """
    area = _triangle_projected_area(u1, u2, u3)
    return area * (
        ph3 ** 2 + ph2 ** 2 + ph1 ** 2 + ph2 * ph3 + ph1 * ph2 + ph1 * ph3
    ) / 12.0


def _Psi0t(u1, u2, u3, ph1, ph2, ph3) -> float:
    """
    Contribution to ψ(0) from one projected triangle [reference Psi0t()].

    ψ_0,t = (φ₁ + φ₂ + φ₃) × |signed_area| / 6
    """
    area = _triangle_projected_area(u1, u2, u3)
    return (ph1 + ph2 + ph3) * area / 6.0


def _Sigmat(u1, u2, u3, ph1, ph2, ph3) -> np.ndarray:
    """
    Contribution to Σ₀⁻¹ covariance from one projected triangle [reference Sigmat()].

    Returns [Σ_xx, Σ_xy, Σ_yy] (the 3 unique elements of the 2x2 symmetric matrix).
    """
    u1x, u1y = u1[0], u1[1]
    u2x, u2y = u2[0], u2[1]
    u3x, u3y = u3[0], u3[1]
    area = abs(
        -u1y * u2x + u1x * u2y + u1y * u3x - u2y * u3x - u1x * u3y + u2x * u3y
    )

    a = area * (
        (3 * ph1 + ph2 + ph3) * u1x ** 2
        + (ph1 + 3 * ph2 + ph3) * u2x ** 2
        + (ph1 + 2 * (ph2 + ph3)) * u2x * u3x
        + (ph1 + ph2 + 3 * ph3) * u3x ** 2
        + u1x * ((2 * (ph1 + ph2) + ph3) * u2x + (2 * ph1 + ph2 + 2 * ph3) * u3x)
    ) / 60.0

    b = area * (
        u1x * (2 * (3 * ph1 + ph2 + ph3) * u1y + (2 * (ph1 + ph2) + ph3) * u2y + (2 * ph1 + ph2 + 2 * ph3) * u3y)
        + u3x * ((2 * ph1 + ph2 + 2 * ph3) * u1y + (ph1 + 2 * (ph2 + ph3)) * u2y + 2 * (ph1 + ph2 + 3 * ph3) * u3y)
        + u2x * ((2 * (ph1 + ph2) + ph3) * u1y + 2 * (ph1 + 3 * ph2 + ph3) * u2y + (ph1 + 2 * (ph2 + ph3)) * u3y)
    ) / 120.0

    c = area * (
        (3 * ph1 + ph2 + ph3) * u1y ** 2
        + (ph1 + 3 * ph2 + ph3) * u2y ** 2
        + (ph1 + 2 * (ph2 + ph3)) * u2y * u3y
        + (ph1 + ph2 + 3 * ph3) * u3y ** 2
        + u1y * ((2 * (ph1 + ph2) + ph3) * u2y + (2 * ph1 + ph2 + 2 * ph3) * u3y)
    ) / 60.0

    return np.array([a, b, c], dtype=np.float64)


def _Pjhat(a: complex, b: complex, e: np.ndarray) -> float:
    """
    Edge-diffracted power with central lobe removed [Eq. 32].

    P̂_j = |e|² × [|a-b|² × I₁ + |a+b|²/4 × I₂]
    """
    e_len_sq = float(e[0] ** 2 + e[1] ** 2)
    return e_len_sq * (
        abs(a - b) ** 2 * _INTEGRAL_1 + abs(a + b) ** 2 / 4.0 * _INTEGRAL_2
    )


def _intersect_circle_line_2d(r, ax, ay, bx, by):
    """
    Intersect line segment a->b with circle of radius r centered at origin.

    Returns: (n_int, p1x, p1y, p2x, p2y)
    n_int = 0, 1, or 2 intersections on the segment interior (0 < t < 1).
    When n_int=1, the valid intersection is in (p1x, p1y).

    Reference: fsdUtils.h lines 86-105.
    """
    dx = bx - ax
    dy = by - ay
    A = dx * dx + dy * dy
    if A < 1e-24:
        return 0, 0.0, 0.0, 0.0, 0.0
    adotd = ax * dx + ay * dy
    C = ax * ax + ay * ay - r * r
    disc = 4.0 * adotd * adotd - 4.0 * A * C
    if disc <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    sqrt_disc = np.sqrt(disc)
    t1 = (-adotd + 0.5 * sqrt_disc) / A
    t2 = (-adotd - 0.5 * sqrt_disc) / A
    if t2 < t1:
        t1, t2 = t2, t1
    p1x, p1y = ax + t1 * dx, ay + t1 * dy
    p2x, p2y = ax + t2 * dx, ay + t2 * dy
    p1v = (t1 > 0) and (t1 < 1)
    p2v = (t2 > 0) and (t2 < 1)
    if not p1v and not p2v:
        return 0, 0.0, 0.0, 0.0, 0.0
    if p1v and p2v:
        return 2, p1x, p1y, p2x, p2y
    if not p1v:
        return 1, p2x, p2y, 0.0, 0.0
    return 1, p1x, p1y, 0.0, 0.0


def _area_circ_sector(r, theta):
    """Area of circular sector with angle theta and radius r."""
    theta = abs(theta)
    if theta > np.pi:
        theta = 2.0 * np.pi - theta
    return 0.5 * r * r * theta


def _area_tri_2d_scalar(ax, ay, bx, by):
    """Signed area of triangle (origin, a, b): 0.5 * (ax*by - bx*ay)."""
    return 0.5 * (ax * by - bx * ay)


def _area_circ_sector_line(r, ax, ay, bx, by):
    """
    Signed area between line segment a->b and the arc of circle(0, r).

    Handles 3 cases: 0, 1, or 2 intersections of segment with circle.
    Reference: fsdUtils.h lines 119-142.
    """
    n_int, p1x, p1y, p2x, p2y = _intersect_circle_line_2d(r, ax, ay, bx, by)

    theta_a = np.arctan2(ay, ax)
    theta_b = np.arctan2(by, bx)
    theta_p1 = np.arctan2(p1y, p1x) if n_int > 0 else 0.0
    theta_p2 = np.arctan2(p2y, p2x) if n_int > 1 else 0.0

    wnd = theta_a - theta_b
    if wnd < -np.pi:
        wnd += 2.0 * np.pi
    if wnd > np.pi:
        wnd -= 2.0 * np.pi
    sgn = -1.0 if wnd < 0 else 1.0

    if n_int == 2:
        return sgn * (abs(_area_tri_2d_scalar(p1x, p1y, p2x, p2y))
                      + _area_circ_sector(r, theta_a - theta_p1)
                      + _area_circ_sector(r, theta_p2 - theta_b))
    elif n_int == 1:
        a_inside = (ax * ax + ay * ay) < r * r
        if a_inside:
            return sgn * (abs(_area_tri_2d_scalar(ax, ay, p1x, p1y))
                          + _area_circ_sector(r, theta_b - theta_p1))
        else:
            return sgn * (abs(_area_tri_2d_scalar(p1x, p1y, bx, by))
                          + _area_circ_sector(r, theta_a - theta_p1))
    else:
        a_inside = (ax * ax + ay * ay) < r * r
        if a_inside:
            return sgn * abs(_area_tri_2d_scalar(ax, ay, bx, by))
        else:
            return sgn * _area_circ_sector(r, theta_a - theta_b)


def _area_circle_tri(radius: float, u1, u2, u3) -> float:
    """
    Exact area of intersection between circle(0, radius) and triangle(u1, u2, u3).

    Decomposes into three signed sector-line areas and takes absolute value.
    Reference: fsdUtils.h lines 143-147 (areaCircleTri).
    """
    return abs(
        _area_circ_sector_line(radius, u1[0], u1[1], u2[0], u2[1])
        + _area_circ_sector_line(radius, u2[0], u2[1], u3[0], u3[1])
        + _area_circ_sector_line(radius, u3[0], u3[1], u1[0], u1[1])
    )


def _area_circle_tri_batch_numpy(radius: float, u1_all, u2_all, u3_all) -> np.ndarray:
    """
    Batch area of intersection between circle(0, radius) and K triangles.

    Vectorized numpy implementation for the fill fraction computation.

    Args:
        radius: Circle radius (scalar).
        u1_all, u2_all, u3_all: [K, 2] triangle vertices (2D projected).

    Returns:
        [K] float64 areas of circle-triangle intersection.
    """
    K = len(u1_all)
    if K == 0:
        return np.empty(0, dtype=np.float64)

    r2 = radius * radius
    r2_half = 0.5 * r2

    def _batch_circ_sector_line(ax, ay, bx, by):
        """Vectorized areaCircSectorLine for [K] segments."""
        dx = bx - ax
        dy = by - ay
        A = dx * dx + dy * dy
        adotd = ax * dx + ay * dy
        C = ax * ax + ay * ay - r2
        disc = 4.0 * adotd * adotd - 4.0 * A * C

        safe_A = np.maximum(A, 1e-24)
        has_disc = disc > 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

        t1_raw = (-adotd + 0.5 * sqrt_disc) / safe_A
        t2_raw = (-adotd - 0.5 * sqrt_disc) / safe_A
        t1 = np.minimum(t1_raw, t2_raw)
        t2 = np.maximum(t1_raw, t2_raw)

        p1v = (t1 > 0) & (t1 < 1)
        p2v = (t2 > 0) & (t2 < 1)
        n_int = np.where(has_disc & p1v & p2v, 2,
                         np.where(has_disc & (p1v | p2v), 1, 0))

        p1x = ax + t1 * dx
        p1y = ay + t1 * dy
        p2x = ax + t2 * dx
        p2y = ay + t2 * dy

        # When n_int==1 and only p2 is valid, swap to p1 position
        swap = (n_int == 1) & (~p1v)
        p1x = np.where(swap, p2x, p1x)
        p1y = np.where(swap, p2y, p1y)

        theta_a = np.arctan2(ay, ax)
        theta_b = np.arctan2(by, bx)
        theta_p1 = np.where(n_int > 0, np.arctan2(p1y, p1x), 0.0)
        theta_p2 = np.where(n_int > 1, np.arctan2(p2y, p2x), 0.0)

        wnd = theta_a - theta_b
        wnd = np.where(wnd < -np.pi, wnd + 2 * np.pi, wnd)
        wnd = np.where(wnd > np.pi, wnd - 2 * np.pi, wnd)
        sgn = np.where(wnd < 0, -1.0, 1.0)

        a_inside = (ax * ax + ay * ay) < r2

        def _tri_a(px, py, qx, qy):
            return 0.5 * np.abs(px * qy - qx * py)

        def _sec_a(theta):
            t = np.abs(theta)
            t = np.where(t > np.pi, 2 * np.pi - t, t)
            return r2_half * t

        # Case 2: two intersections
        area_2 = sgn * (_tri_a(p1x, p1y, p2x, p2y)
                        + _sec_a(theta_a - theta_p1)
                        + _sec_a(theta_p2 - theta_b))
        # Case 1: one intersection
        area_1a = sgn * (_tri_a(ax, ay, p1x, p1y) + _sec_a(theta_b - theta_p1))
        area_1b = sgn * (_tri_a(p1x, p1y, bx, by) + _sec_a(theta_a - theta_p1))
        area_1 = np.where(a_inside, area_1a, area_1b)
        # Case 0: no intersections
        area_0_in = sgn * _tri_a(ax, ay, bx, by)
        area_0_out = sgn * _sec_a(theta_a - theta_b)
        area_0 = np.where(a_inside, area_0_in, area_0_out)

        return np.where(n_int == 2, area_2, np.where(n_int == 1, area_1, area_0))

    return np.abs(
        _batch_circ_sector_line(u1_all[:, 0], u1_all[:, 1], u2_all[:, 0], u2_all[:, 1])
        + _batch_circ_sector_line(u2_all[:, 0], u2_all[:, 1], u3_all[:, 0], u3_all[:, 1])
        + _batch_circ_sector_line(u3_all[:, 0], u3_all[:, 1], u1_all[:, 0], u1_all[:, 1])
    )


def construct_aperture(
    hit_pos: np.ndarray,         # [3] hit point in world space
    wo: np.ndarray,              # [3] direction from hit toward RX (unit vector)
    tri_hash: TriangleSpatialHash,
    k: float,                    # wavenumber 2π/λ
    beam_sigma: float,           # beam width in meters
    max_tessellation_depth: int = 5,
    max_edges: int = 256,
    fill_min: float = 1e-6,
    fill_max: float = 1.0 - 1e-6,
    tri_eps_real: 'Optional[np.ndarray]' = None,  # [n_faces] float32 — per-triangle ε'
    tri_eps_imag: 'Optional[np.ndarray]' = None,  # [n_faces] float32 — per-triangle ε''
    mesh_faces: 'Optional[np.ndarray]' = None,    # [n_faces, 3] int32 — mesh vertex indices
    tx_polarization: 'Optional[np.ndarray]' = None,  # [3] TX E-field direction
    rx_polarization: 'Optional[np.ndarray]' = None,  # [3] RX E-field direction
    jones_mode: bool = False,
    edge_angle_threshold_deg: float = 15.0,  # min dihedral angle (degrees) for diffracting edges
) -> FsdAperture:
    """
    Construct the diffracting aperture for a hit point (Algorithm 1 of paper).

    Args:
        hit_pos: World-space hit point [3].
        wo: Unit direction from hit toward RX (virtual screen z-axis).
        tri_hash: Prebuilt triangle spatial hash.
        k: Wavenumber 2π/λ.
        beam_sigma: Gaussian beam spatial standard deviation (meters).
        max_tessellation_depth: Max recursive subdivision depth.
        max_edges: Maximum boundary edges to keep.
        fill_min: Min projected fill fraction for early exit.
        fill_max: Max projected fill fraction for early exit.
        tri_eps_real: Per-triangle real permittivity (indexed by global face index).
            When provided, edge amplitudes are modulated by Fresnel opacity.
        tri_eps_imag: Per-triangle imaginary permittivity (loss).
        tx_polarization: TX E-field polarization direction [3] (unit vector).
            Required when jones_mode=True.
        rx_polarization: RX E-field polarization direction [3] (unit vector).
            Required when jones_mode=True.
        jones_mode: If True, compute full Jones reflectance per edge instead
            of scalar opacity. Modulates amplitudes with complex E_jones.

    Returns:
        FsdAperture with boundary edges and power integrals.
    """
    search_radius = 3.0 * beam_sigma
    p = hit_pos.astype(np.float64)
    wo_unit = wo.astype(np.float64)
    wo_unit = wo_unit / max(np.linalg.norm(wo_unit), 1e-12)

    # Build virtual screen frame: tangent, bitangent, wo
    # Choose a tangent vector not parallel to wo
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(wo_unit, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])

    tangent = up - np.dot(wo_unit, up) * wo_unit
    t_len = np.linalg.norm(tangent)
    if t_len < 1e-7:
        # Fallback: use cross product with a different vector
        alt = np.array([wo_unit[1], -wo_unit[2], wo_unit[0]])
        tangent = np.cross(alt, wo_unit)
        t_len = np.linalg.norm(tangent)
    tangent = tangent / t_len
    bitangent = np.cross(wo_unit, tangent)

    aperture = FsdAperture(
        tangent=tangent.astype(np.float32),
        bitangent=bitangent.astype(np.float32),
        wo_dir=wo_unit.astype(np.float32),
        k=k,
        beam_sigma=beam_sigma,
    )

    # Query triangles near hit point
    face_indices, v0_arr, v1_arr, v2_arr, normals, neighbor_norms = \
        tri_hash.query_sphere_with_data(hit_pos.astype(np.float32), search_radius)

    if len(face_indices) == 0:
        return aperture

    # Cull back-facing triangles: keep dot(wo, face_normal) > 0
    dots = np.einsum('j,ij->i', wo_unit, normals.astype(np.float64))
    front_mask = dots > 0

    if not np.any(front_mask):
        return aperture

    # Early exit: check fill fraction (approximate)
    # Batch-project all front-facing triangle vertices at once
    front_indices_local = np.where(front_mask)[0]
    n_front = len(front_indices_local)
    circle_area = np.pi * search_radius ** 2

    # Batch projection: [K, 3] deltas → [K, 2] screen coords via matrix multiply
    delta_v0 = v0_arr[front_mask].astype(np.float64) - p  # [K, 3]
    delta_v1 = v1_arr[front_mask].astype(np.float64) - p
    delta_v2 = v2_arr[front_mask].astype(np.float64) - p

    proj_mat = np.stack([tangent, bitangent], axis=1)  # [3, 2]
    u1_all = delta_v0 @ proj_mat  # [K, 2]
    u2_all = delta_v1 @ proj_mat
    u3_all = delta_v2 @ proj_mat

    # Proper circle-triangle intersection area for fill fraction
    # (replaces shoelace approximation which over-counted triangles beyond circle)
    areas = _area_circle_tri_batch_numpy(search_radius, u1_all, u2_all, u3_all)
    total_proj_area = float(np.sum(areas))

    fill_fraction = total_proj_area / circle_area
    if fill_fraction > fill_max or fill_fraction < fill_min:
        return aperture

    # Batch compute z-depths (along -wo direction)
    z1_all = -delta_v0 @ wo_unit  # [K]
    z2_all = -delta_v1 @ wo_unit
    z3_all = -delta_v2 @ wo_unit

    # Batch boundary edge detection with dihedral angle filtering
    nn_front = neighbor_norms[front_mask].astype(np.float64)  # [K, 3, 3]
    fn_front = normals[front_mask].astype(np.float64)  # [K, 3] — face normals
    wo_dots = np.einsum('j,kij->ki', wo_unit, nn_front)  # [K, 3]
    nn_norms = np.linalg.norm(nn_front, axis=2)  # [K, 3]
    is_boundary_edge = nn_norms < 0.5  # [K, 3] — zero normal = boundary
    has_neighbor = ~is_boundary_edge

    # Dihedral angle: cos(angle) = dot(face_normal, neighbor_normal) / |nn|
    # (face normals are already unit-length from tri_hash)
    fn_expanded = fn_front[:, np.newaxis, :]  # [K, 1, 3]
    cos_dihedral = np.einsum('kij,kij->ki', fn_expanded.repeat(3, axis=1), nn_front)  # [K, 3]
    cos_dihedral = cos_dihedral / np.maximum(nn_norms, 1e-8)  # normalize by |nn|

    # Pre-computed edge suppression flags (Filters E/D/C)
    edge_suppressed_front = tri_hash.edge_is_suppressed[face_indices[front_mask]]  # [K, 3]

    # An edge is diffracting if:
    # 1. Has a neighbor AND neighbor is back-facing (wo_dot <= 0)
    # 2. AND the dihedral angle exceeds the threshold (not coplanar)
    # 3. AND the edge is not suppressed by mesh-quality filters
    # Boundary edges (no neighbor) are NOT automatically diffracting — on LiDAR
    # meshes, missing neighbors are usually mesh artifacts, not real geometric edges.
    cos_threshold = np.cos(np.radians(edge_angle_threshold_deg))
    is_significant_edge = cos_dihedral < cos_threshold  # dihedral > threshold degrees
    edge_is_diffracting = has_neighbor & (wo_dots <= 0) & is_significant_edge & ~edge_suppressed_front  # [K, 3]

    # --- Main loop: tessellate and extract boundary edges ---
    # (tessellation must remain sequential due to recursion)
    # Accumulators for power integrals
    P_A = 0.0
    P_A_geom = 0.0   # Scalar-opacity P_A for threshold check (not attenuated by Jones)
    P_A_bare = 0.0    # Bare geometric P_A (no material modulation) — for normalization
    psi_0 = 0.0
    psi_0_bare = 0.0  # Bare psi_0 (no material modulation) — for P_central_bare
    Sigmat_0 = np.zeros(3, dtype=np.float64)
    Sigmat_0_bare = np.zeros(3, dtype=np.float64)  # Bare Sigmat (no material modulation)
    sum_Phat_j = 0.0
    e_avg = 0.0  # power-weighted average edge length

    edges: List[FsdEdge] = []
    max_edge_length_sq = beam_sigma ** 2  # tessellate edges longer than sigma

    def _winding(e_vec, v_mid, centroid):
        """Determine if outward normal m̂ = (e.y, -e.x) points away from centroid."""
        m = np.array([e_vec[1], -e_vec[0]])
        return 1.0 if np.dot(m, v_mid - centroid) > 0 else -1.0

    # Material opacity flag: only compute opacity if material arrays are provided
    use_material_opacity = (tri_eps_real is not None and tri_eps_imag is not None)

    def _add_edge(u1_2d, u2_2d, z1, z2, ph1, ph2, tri_centroid_2d,
                  edge_opacity=1.0, edge_face_idx=-1, edge_cos_theta=1.0,
                  b1=None, b2=None, tri_vi=None,
                  edge_jones_real=1.0, edge_jones_imag=0.0,
                  edge_tx_s=0.0, edge_tx_p=0.0, edge_rx_s=0.0, edge_rx_p=0.0):
        """Add a boundary edge with correct winding and optional opacity modulation.

        Args:
            b1, b2: Barycentric coordinates [2] of the two edge endpoints
                within the original mesh triangle (for per-vertex interpolation).
            tri_vi: Vertex indices [3] of the owning mesh triangle.
            edge_jones_real/imag: Complex Jones reflectance from Phase A.
            edge_tx_s/p, edge_rx_s/p: Frozen TX/RX s/p projections.
        """
        nonlocal sum_Phat_j, e_avg

        v_mid = (u1_2d + u2_2d) / 2.0
        e_vec = u2_2d - u1_2d

        # Ensure outward normal points away from triangle centroid (into aperture)
        w = _winding(e_vec, v_mid, tri_centroid_2d)
        if w < 0:
            e_vec = -e_vec
            ph1, ph2 = ph2, ph1
            z1, z2 = z2, z1

        # Complex amplitudes with phase from z-depth [Eq. 24]
        # Reference uses std::polar(a, k*z) = a * exp(i*k*z)
        # Convention: positive k*z phase (matching reference code line 445)
        ca = ph1 * np.exp(1j * k * z1)
        cb = ph2 * np.exp(1j * k * z2)

        # Material-dependent modulation:
        # Jones mode: scale by complex E_jones (magnitude + phase)
        # Scalar mode: scale by √opacity (real, no phase)
        if jones_mode and use_material_opacity:
            E_jones = complex(edge_jones_real, edge_jones_imag)
            ca = ca * E_jones
            cb = cb * E_jones
        elif edge_opacity < 1.0:
            sqrt_opacity = np.sqrt(edge_opacity)
            ca = ca * sqrt_opacity
            cb = cb * sqrt_opacity

        P = _Pjhat(ca, cb, e_vec)
        if P <= 0:
            return

        sum_Phat_j += P
        e_len = np.sqrt(e_vec[0] ** 2 + e_vec[1] ** 2)
        e_avg += P * e_len

        # Edge midpoint barycentric coordinates (average of endpoint barycentrics)
        if b1 is not None and b2 is not None:
            b_mid = (b1 + b2) * 0.5
            edge_bary_u = float(b_mid[0])
            edge_bary_v = float(b_mid[1])
        else:
            edge_bary_u = -1.0
            edge_bary_v = -1.0

        edges.append(FsdEdge(
            e=e_vec.astype(np.float32),
            v=v_mid.astype(np.float32),
            a=ca,
            b=cb,
            Phat=P,
            Phat_accum=sum_Phat_j,
            opacity=edge_opacity,
            face_idx=edge_face_idx,
            cos_theta=edge_cos_theta,
            bary_u=edge_bary_u,
            bary_v=edge_bary_v,
            vert_idx_0=int(tri_vi[0]) if tri_vi is not None else -1,
            vert_idx_1=int(tri_vi[1]) if tri_vi is not None else -1,
            vert_idx_2=int(tri_vi[2]) if tri_vi is not None else -1,
            jones_real=edge_jones_real,
            jones_imag=edge_jones_imag,
            tx_amp_s=edge_tx_s,
            tx_amp_p=edge_tx_p,
            rx_amp_s=edge_rx_s,
            rx_amp_p=edge_rx_p,
        ))

    def _add_tri(
        edge12: bool, edge13: bool, edge23: bool,
        u1, u2, u3, z1, z2, z3,
        depth: int,
        tri_opacity: float = 1.0,
        tri_face_idx: int = -1,
        tri_cos_theta: float = 1.0,
        b1=None, b2=None, b3=None,  # Barycentrics [2] of sub-vertices in original triangle
        tri_vi=None,                 # Vertex indices [3] of original mesh triangle
        tri_jones_real: float = 1.0, tri_jones_imag: float = 0.0,
        tri_jones_power: float = 1.0,
        tri_tx_s: float = 0.0, tri_tx_p: float = 0.0,
        tri_rx_s: float = 0.0, tri_rx_p: float = 0.0,
    ):
        """
        Recursively tessellate and extract boundary edges from a projected triangle.

        edge12/13/23: whether edge (1-2), (1-3), (2-3) is a boundary (diffracting) edge.
        tri_opacity: Fresnel opacity of the owning triangle (inherited during tessellation).
        tri_face_idx: Global face index of the owning triangle.
        tri_cos_theta: cos(θ_i) at the owning triangle surface.
        b1, b2, b3: Barycentric coordinates of (u1, u2, u3) within the original triangle.
            Threaded through tessellation so each sub-vertex knows its position.
        tri_vi: Vertex indices of the original mesh triangle (inherited unchanged).
        tri_jones_real/imag: Complex Jones reflectance (inherited during tessellation).
        tri_jones_power: |E_jones|² (replaces opacity in Jones mode).
        tri_tx_s/p, tri_rx_s/p: Frozen TX/RX s/p projections.
        """
        nonlocal P_A, P_A_geom, P_A_bare, psi_0, psi_0_bare, Sigmat_0, Sigmat_0_bare

        area = _area_circle_tri(search_radius, u1, u2, u3)
        if area < 1e-10:
            return

        centroid_2d = (u1 + u2 + u3) / 3.0
        z0 = (z1 + z2 + z3) / 3.0

        # Barycentric centroid (average of 3 sub-vertex barycentrics)
        bc = (b1 + b2 + b3) / 3.0 if b1 is not None else None

        # Check if any edge needs tessellation
        len12_sq = float(np.sum((u2 - u1) ** 2))
        len13_sq = float(np.sum((u3 - u1) ** 2))
        len23_sq = float(np.sum((u3 - u2) ** 2))
        sub12 = len12_sq > max_edge_length_sq
        sub13 = len13_sq > max_edge_length_sq
        sub23 = len23_sq > max_edge_length_sq
        n_sub = int(sub12) + int(sub13) + int(sub23)
        sss = (n_sub == 1)  # single-side subdivision

        # Jones params shorthand for recursive calls
        _jkw = dict(tri_jones_real=tri_jones_real, tri_jones_imag=tri_jones_imag,
                     tri_jones_power=tri_jones_power,
                     tri_tx_s=tri_tx_s, tri_tx_p=tri_tx_p,
                     tri_rx_s=tri_rx_s, tri_rx_p=tri_rx_p)

        if depth < max_tessellation_depth and (sub12 or sub13 or sub23):
            # Recursive tessellation (matching reference code lines 470-489)
            # All sub-triangles inherit the parent's material properties
            c = centroid_2d
            zc = z0

            # Barycentric midpoints (linear interpolation of barycentrics)
            bm12 = (b1 + b2) / 2.0 if b1 is not None else None
            bm13 = (b1 + b3) / 2.0 if b1 is not None else None
            bm23 = (b2 + b3) / 2.0 if b1 is not None else None

            if sub12:
                m12 = (u1 + u2) / 2; zm12 = (z1 + z2) / 2
                _add_tri(edge12, sss and edge13, False, u1, m12, u3 if sss else c, z1, zm12, z3 if sss else zc, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         b1, bm12, b3 if sss else bc, tri_vi, **_jkw)
                _add_tri(edge12, False, sss and edge23, m12, u2, u3 if sss else c, zm12, z2, z3 if sss else zc, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         bm12, b2, b3 if sss else bc, tri_vi, **_jkw)
            elif not sss:
                _add_tri(edge12, False, False, u1, u2, c, z1, z2, zc, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         b1, b2, bc, tri_vi, **_jkw)

            if sub13:
                m13 = (u1 + u3) / 2; zm13 = (z1 + z3) / 2
                _add_tri(sss and edge12, edge13, False, u1, u2 if sss else c, m13, z1, z2 if sss else zc, zm13, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         b1, b2 if sss else bc, bm13, tri_vi, **_jkw)
                _add_tri(False, edge13, sss and edge23, m13, u2 if sss else c, u3, zm13, z2 if sss else zc, z3, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         bm13, b2 if sss else bc, b3, tri_vi, **_jkw)
            elif not sss:
                _add_tri(False, edge13, False, u1, c, u3, z1, zc, z3, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         b1, bc, b3, tri_vi, **_jkw)

            if sub23:
                m23 = (u2 + u3) / 2; zm23 = (z2 + z3) / 2
                _add_tri(False, sss and edge13, edge23, u1 if sss else c, m23, u3, z1 if sss else zc, zm23, z3, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         b1 if sss else bc, bm23, b3, tri_vi, **_jkw)
                _add_tri(sss and edge12, False, edge23, u1 if sss else c, u2, m23, z1 if sss else zc, z2, zm23, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         b1 if sss else bc, b2, bm23, tri_vi, **_jkw)
            elif not sss:
                _add_tri(False, False, edge23, c, u2, u3, zc, z2, z3, depth + 1, tri_opacity, tri_face_idx, tri_cos_theta,
                         bc, b2, b3, tri_vi, **_jkw)

            return

        # Leaf: compute beam amplitudes and accumulate
        ph1 = _phi_gaussian(u1, z1, beam_sigma)
        ph2 = _phi_gaussian(u2, z2, beam_sigma)
        ph3 = _phi_gaussian(u3, z3, beam_sigma)

        # Jones parameters for _add_edge
        _ekw = dict(edge_jones_real=tri_jones_real, edge_jones_imag=tri_jones_imag,
                     edge_tx_s=tri_tx_s, edge_tx_p=tri_tx_p,
                     edge_rx_s=tri_rx_s, edge_rx_p=tri_rx_p)

        # Add boundary edges (with material opacity, face index, barycentrics, and Jones)
        if edge12:
            _add_edge(u2, u1, z2, z1, ph2, ph1, centroid_2d,
                      tri_opacity, tri_face_idx, tri_cos_theta,
                      b2, b1, tri_vi, **_ekw)
        if edge13:
            _add_edge(u1, u3, z1, z3, ph1, ph3, centroid_2d,
                      tri_opacity, tri_face_idx, tri_cos_theta,
                      b1, b3, tri_vi, **_ekw)
        if edge23:
            _add_edge(u3, u2, z3, z2, ph3, ph2, centroid_2d,
                      tri_opacity, tri_face_idx, tri_cos_theta,
                      b3, b2, tri_vi, **_ekw)

        # Accumulate triangle power integrals (opacity-weighted for energy conservation)
        # Jones mode: use jones_power instead of opacity
        # P_t is power (quadratic in φ) → scale by effective_opacity
        # Ψ₀_t and Σ_t are field-level (linear in φ) → scale by √effective_opacity
        eff_opacity = tri_jones_power if (jones_mode and use_material_opacity) else tri_opacity
        Pt_val = _Pt(u1, u2, u3, ph1, ph2, ph3)
        Psi0t_val = _Psi0t(u1, u2, u3, ph1, ph2, ph3)
        Sigmat_val = _Sigmat(u1, u2, u3, ph1, ph2, ph3)
        P_A += eff_opacity * Pt_val
        P_A_geom += tri_opacity * Pt_val  # scalar-opacity version for threshold check
        P_A_bare += Pt_val                # bare geometric power (no material modulation)
        sqrt_op = np.sqrt(eff_opacity)
        Sigmat_0 += sqrt_op * Sigmat_val
        psi_0 += sqrt_op * Psi0t_val
        # Bare accumulators for P_central_bare computation
        Sigmat_0_bare += Sigmat_val
        psi_0_bare += Psi0t_val

    # --- Batch pre-compute material properties for all front-facing triangles ---
    # Replaces per-triangle scalar calls with a single vectorized computation.
    global_faces_all = face_indices[front_indices_local]  # [K] global face indices

    if use_material_opacity:
        fn_all = normals[front_mask].astype(np.float64)  # [K, 3]
        fn_norms = np.linalg.norm(fn_all, axis=1, keepdims=True)
        fn_unit_all = fn_all / np.maximum(fn_norms, 1e-8)
        cos_theta_all = np.abs(np.einsum('j,ij->i', wo_unit, fn_unit_all))  # [K]

        eps_r_vals = tri_eps_real[global_faces_all].astype(np.float64)
        eps_i_vals = tri_eps_imag[global_faces_all].astype(np.float64)

        # Batch Fresnel opacity
        opacity_all = _fresnel_opacity_numpy_batch(eps_r_vals, eps_i_vals, cos_theta_all)

        if jones_mode:
            # Batch Jones reflectance
            _jr_all, _ji_all, _jpow_all, _ts_all, _tp_all, _rs_all, _rp_all = \
                _jones_reflectance_numpy_batch(
                    eps_r_vals, eps_i_vals, cos_theta_all,
                    wo_unit, fn_unit_all, tx_polarization, rx_polarization,
                )

    # Pre-compute barycentric init arrays (shared across all triangles)
    _init_b1 = np.array([0.0, 0.0])
    _init_b2 = np.array([1.0, 0.0])
    _init_b3 = np.array([0.0, 1.0])

    # Process each front-facing triangle using pre-computed batch data
    for k_idx in range(n_front):
        u1 = u1_all[k_idx]
        u2 = u2_all[k_idx]
        u3 = u3_all[k_idx]
        z1 = float(z1_all[k_idx])
        z2 = float(z2_all[k_idx])
        z3 = float(z3_all[k_idx])

        # Boundary edge flags from batch-computed array
        edge12 = bool(edge_is_diffracting[k_idx, 0])
        edge23 = bool(edge_is_diffracting[k_idx, 1])
        edge13 = bool(edge_is_diffracting[k_idx, 2])

        # Look up pre-computed material properties (no per-scalar Fresnel/Jones calls)
        global_face = global_faces_all[k_idx]
        _jones_r, _jones_i, _jones_pow = 1.0, 0.0, 1.0
        _tx_s, _tx_p, _rx_s, _rx_p = 0.0, 0.0, 0.0, 0.0

        if use_material_opacity:
            cos_theta_tri = float(cos_theta_all[k_idx])
            tri_opacity = float(opacity_all[k_idx])
            if jones_mode:
                _jones_r = float(_jr_all[k_idx])
                _jones_i = float(_ji_all[k_idx])
                _jones_pow = float(_jpow_all[k_idx])
                _tx_s = float(_ts_all[k_idx])
                _tx_p = float(_tp_all[k_idx])
                _rx_s = float(_rs_all[k_idx])
                _rx_p = float(_rp_all[k_idx])
        else:
            tri_opacity = 1.0
            cos_theta_tri = 1.0

        # Barycentric init + vertex indices
        if mesh_faces is not None:
            tri_vi = mesh_faces[global_face]
            init_b1, init_b2, init_b3 = _init_b1, _init_b2, _init_b3
        else:
            tri_vi = None
            init_b1 = init_b2 = init_b3 = None

        _add_tri(edge12, edge13, edge23, u1, u2, u3, z1, z2, z3, depth=0,
                 tri_opacity=tri_opacity, tri_face_idx=global_face, tri_cos_theta=cos_theta_tri,
                 b1=init_b1, b2=init_b2, b3=init_b3, tri_vi=tri_vi,
                 tri_jones_real=_jones_r, tri_jones_imag=_jones_i,
                 tri_jones_power=_jones_pow,
                 tri_tx_s=_tx_s, tri_tx_p=_tx_p,
                 tri_rx_s=_rx_s, tri_rx_p=_rx_p)

        if len(edges) >= max_edges:
            break

    # Finalize aperture
    # Use P_A_geom (scalar-opacity weighted) for the significance threshold so
    # Jones-attenuated apertures aren't incorrectly discarded.
    if not edges or P_A_geom < 1e-2:
        aperture.P_A = 0
        return aperture

    # Compute central lobe power [Eq. 35]
    if sum_Phat_j > 0:
        e_avg /= sum_Phat_j
    else:
        e_avg = 0.01  # fallback

    sigma_xi = np.sqrt(3.0) / (k * max(e_avg, 1e-6))

    inv_sigma_xi_sq = 1.0 / (sigma_xi ** 2)

    # Scale covariance: Σ₀ = 6k²/ψ₀ × Σ_t + σ_ξ⁻² I
    if abs(psi_0) > 1e-12:
        Sigmat_0 *= 6.0 * k * k / psi_0
    Sigma0 = Sigmat_0 + np.array([inv_sigma_xi_sq, 0.0, inv_sigma_xi_sq])

    # det(Σ₀) = Σ_xx × Σ_yy - Σ_xy²
    det_Sigma0 = max(0.0, Sigma0[0] * Sigma0[2] - Sigma0[1] ** 2)

    # P̃_central ≈ 2π × (3/(√2 k e_max))² × |ψ(0)|²  [Eq. 35]
    if det_Sigma0 > 0:
        P_central = k * k / (18.0 * np.pi) / np.sqrt(det_Sigma0) * psi_0 ** 2
    else:
        P_central = 0.0

    # Clamped diffracted power [Eq. 36]
    P_A_hat = max(0.0, P_A - P_central)

    # Bare (no material modulation) P_central and P_A_hat for normalization denominator
    if abs(psi_0_bare) > 1e-12:
        Sigmat_0_bare *= 6.0 * k * k / psi_0_bare
    Sigma0_bare = Sigmat_0_bare + np.array([inv_sigma_xi_sq, 0.0, inv_sigma_xi_sq])
    det_Sigma0_bare = max(0.0, Sigma0_bare[0] * Sigma0_bare[2] - Sigma0_bare[1] ** 2)
    if det_Sigma0_bare > 0:
        P_central_bare = k * k / (18.0 * np.pi) / np.sqrt(det_Sigma0_bare) * psi_0_bare ** 2
    else:
        P_central_bare = 0.0
    P_A_hat_bare = max(0.0, P_A_bare - P_central_bare)

    aperture.edges = edges
    aperture.P_A = P_A
    aperture.psi_0 = psi_0
    aperture.P_central = P_central
    aperture.P_A_hat = P_A_hat
    aperture.P_A_hat_bare = P_A_hat_bare
    aperture.P_A_bare = P_A_bare
    aperture.sum_Phat_j = sum_Phat_j
    aperture.Sigma0 = Sigma0.astype(np.float32)

    # Cache edge data as contiguous numpy arrays for fast batch packing
    aperture._cache_edge_arrays()

    return aperture
