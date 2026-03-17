"""
Free-space diffraction BSDF evaluation.

Implements the core mathematical functions from Steinberg et al. 2024:
  - α₁, α₂: auxiliary functions [Eq. 26-27 / S2.20-S2.21]
  - χ: Gaussian central-lobe removal [Eq. 29 / S2.23]
  - Psihat: complex edge-diffracted amplitude [Eq. 23 / S2.19]
  - eval: scalar BSDF value f(ω_o) [Eq. 38]
  - eval_complex: complex diffracted amplitude ψ̃ (for MIMO coherence)
  - FlatEdgeData + batch evaluation: vectorized numpy path for all edges/apertures

Convention:
  - The aperture was constructed with z-axis = wo (toward RX)
  - We evaluate at direction wi (toward TX), projected onto the virtual screen
  - ξ̃ = [tan(θ_x), tan(θ_y)] in the virtual screen frame
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .fsd_aperture import FsdAperture, FsdEdge


# ============================================================================
# Core analytic functions (matching reference code lines 289-292)
# ============================================================================

def _sinc(x: float) -> float:
    """sinc(x) = sin(x)/x with Taylor expansion near 0."""
    ax = abs(x)
    if ax < 1e-8:
        return 1.0
    if ax < 0.019:
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    return np.sin(x) / x


def alpha1(zeta_x: float, zeta_y: float) -> float:
    """
    Auxiliary function α₁(ζ) [Eq. 26 / S2.20].

    α₁(ζ) = ζ_y / (2π ζ² ζ_x) × (cos(ζ_x/2) - sinc(ζ_x/2))

    Represents the "difference mode" diffraction pattern: sensitive to
    amplitude variation across the edge (a ≠ b).
    """
    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y
    if zeta_sq < 1e-20 or abs(zeta_x) < 1e-12:
        return 0.0
    inv_pi = 1.0 / np.pi
    return (
        zeta_y / (zeta_sq) * inv_pi
        * (np.cos(zeta_x / 2.0) - _sinc(zeta_x / 2.0))
        / (2.0 * zeta_x)
    )


def alpha2(zeta_x: float, zeta_y: float) -> float:
    """
    Auxiliary function α₂(ζ) [Eq. 27 / S2.21].

    α₂(ζ) = ζ_y / (2π ζ²) × sinc(ζ_x/2)

    Represents the "sum mode" diffraction pattern: sensitive to the
    constant component of amplitude across the edge.
    """
    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y
    if zeta_sq < 1e-20:
        return 0.0
    inv_pi = 1.0 / np.pi
    return zeta_y / (zeta_sq) * inv_pi * _sinc(zeta_x / 2.0) / 2.0


def chi(r_sq: float) -> float:
    """
    Central-lobe removal function χ [Eq. 29 / S2.23].

    χ(r²) = √(1 - exp(-r² / (2σ_ζ²)))  where σ_ζ = √3

    Returns 0 at the center (r=0), approaches 1 for large r.
    This removes the 0th-order (direct-term) lobe from the diffraction pattern.
    """
    # σ_ζ² = 3, so 2σ_ζ² = 6
    return np.sqrt(max(0.0, 1.0 - np.exp(-0.5 * r_sq / 3.0)))


def Psihat(
    a: complex,
    b: complex,
    e: np.ndarray,      # [2] edge vector on virtual screen
    v: np.ndarray,      # [2] edge midpoint on virtual screen
    k: float,           # wavenumber
    xi: np.ndarray,     # [2] scattering direction on virtual screen
) -> complex:
    """
    Complex edge-diffracted amplitude with central lobe removed [Eq. 23/28, S2.19/S2.26].

    ψ̃_j(ξ̃) = k·e²·χ(|ζ|²)·exp(-ik·v·ξ)·[(a-b)·α₁(ζ) + i·(a+b)/2·α₂(ζ)]

    where ζ = Ξ_j⁻ᵀ ξ̃ (canonical-space coordinates).

    Reference code: free_space_diffraction.cpp lines 293-304.

    Args:
        a, b: Complex beam amplitudes at edge endpoints.
        e: 2D edge vector on virtual screen.
        v: 2D edge midpoint on virtual screen.
        k: Wavenumber 2π/λ.
        xi: 2D scattering direction [tan(θ_x), tan(θ_y)] on virtual screen.

    Returns:
        Complex diffracted amplitude from this edge.
    """
    ee = np.sqrt(e[0] ** 2 + e[1] ** 2)
    if ee < 1e-12:
        return 0.0 + 0.0j

    vxi = float(v[0] * xi[0] + v[1] * xi[1])

    # Perpendicular to edge: m = (e_y, -e_x)  [reference line 296]
    m = np.array([e[1], -e[0]])

    # Transform to canonical space: ζ = k × [dot(e, ξ), dot(m, ξ)]  [reference line 298]
    zeta_x = k * float(e[0] * xi[0] + e[1] * xi[1])
    zeta_y = k * float(m[0] * xi[0] + m[1] * xi[1])

    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y
    chi0 = chi(zeta_sq)

    a1_val = alpha1(zeta_x, zeta_y)
    a2_val = alpha2(zeta_x, zeta_y)

    # Complex amplitude combination [reference lines 300-303]
    c_a1 = (a - b) * a1_val
    c_a2 = (a + b) / 2.0 * a2_val

    # Phase from edge midpoint position [reference line 303]
    phase = np.exp(-1j * k * vxi)

    return k * ee * ee * chi0 * phase * (c_a1 + 1j * c_a2)


def Psihat_vectorized(
    edges: list,         # List[FsdEdge]
    k: float,
    xi: np.ndarray,      # [2] scattering direction
) -> complex:
    """
    Sum Psihat over all boundary edges.

    ψ̃(ξ̃) = Σ_j ψ̃_j(ξ̃)

    This is the total clamped diffracted field amplitude.
    """
    psi = 0.0 + 0.0j
    for edge in edges:
        psi += Psihat(edge.a, edge.b, edge.e, edge.v, k, xi)
    return psi


# ============================================================================
# Vectorized evaluation over arrays of directions (for DrJit integration)
# ============================================================================

def _alpha1_vec(zx: np.ndarray, zy: np.ndarray) -> np.ndarray:
    """Vectorized α₁ over arrays."""
    z_sq = zx * zx + zy * zy
    safe_zx = np.where(np.abs(zx) < 1e-12, 1e-12, zx)
    safe_z_sq = np.where(z_sq < 1e-20, 1e-20, z_sq)
    inv_pi = 1.0 / np.pi
    sinc_half = np.sinc(safe_zx / (2.0 * np.pi))  # np.sinc(x) = sin(πx)/(πx)
    cos_half = np.cos(safe_zx / 2.0)
    result = zy / safe_z_sq * inv_pi * (cos_half - sinc_half) / (2.0 * safe_zx)
    mask = (z_sq < 1e-20) | (np.abs(zx) < 1e-12)
    result[mask] = 0.0
    return result


def _alpha2_vec(zx: np.ndarray, zy: np.ndarray) -> np.ndarray:
    """Vectorized α₂ over arrays."""
    z_sq = zx * zx + zy * zy
    safe_z_sq = np.where(z_sq < 1e-20, 1e-20, z_sq)
    inv_pi = 1.0 / np.pi
    sinc_half = np.sinc(zx / (2.0 * np.pi))
    result = zy / safe_z_sq * inv_pi * sinc_half / 2.0
    result[z_sq < 1e-20] = 0.0
    return result


def _chi_vec(r_sq: np.ndarray) -> np.ndarray:
    """Vectorized χ."""
    return np.sqrt(np.maximum(0.0, 1.0 - np.exp(-0.5 * r_sq / 3.0)))


# ============================================================================
# FsdBSDF class
# ============================================================================

class FsdBSDF:
    """
    Free-space diffraction BSDF.

    NOT a subclass of BSDFBase — this is a supplementary BSDF combined with
    the surface-reflection BSDF via additive energy-conserving combination
    at the integrator level.

    Interface (for single-bounce with pattern sampling):
        eval(xi, aperture) → float           # scalar BSDF value
        eval_complex(xi, aperture) → complex  # complex amplitude for MIMO
        compute_beta(aperture) → float        # energy borrowing fraction
    """

    def __init__(self, k: float, beam_sigma: float, beta_max: float = 0.5):
        """
        Args:
            k: Wavenumber 2π/λ.
            beam_sigma: Gaussian beam standard deviation in meters.
            beta_max: Maximum energy borrowing fraction.
        """
        self.k = k
        self.beam_sigma = beam_sigma
        self.beta_max = beta_max

    def world_to_screen(
        self,
        aperture: FsdAperture,
        direction_world: np.ndarray,  # [3] unit direction in world space
    ) -> np.ndarray:
        """
        Project a world-space direction onto the virtual screen.

        Returns ξ̃ = [tan(θ_x), tan(θ_y)] in screen coordinates.

        The paper uses ξ̃ = [tan θ_x, tan θ_y]^T where θ_x, θ_y are the
        direction sines projected onto tangent and bitangent axes.
        For small angles, tan θ ≈ sin θ. For the BSDF eval, we use
        the exact tangent.

        Convention: the z-component in the screen frame is dot(wo, direction).
        Positive z = toward RX (same side as aperture was constructed).
        """
        d = direction_world.astype(np.float64)
        dx = np.dot(aperture.tangent, d)
        dy = np.dot(aperture.bitangent, d)
        dz = np.dot(aperture.wo_dir, d)

        # Avoid division by zero for directions perpendicular to wo
        if abs(dz) < 1e-8:
            dz = 1e-8

        # ξ̃ = [dx/dz, dy/dz] = [tan θ_x, tan θ_y]
        return np.array([dx / dz, dy / dz])

    def eval(
        self,
        wi_world: np.ndarray,      # [3] direction toward TX (world space)
        aperture: FsdAperture,
    ) -> float:
        """
        Evaluate the fsdBSDF at direction wi (toward TX).

        Returns f(ω_o) = R² / (P̂_Ā · cos θ) × ŵ(ξ̃)  [Eq. 38]

        where ŵ(ξ̃) = |Σ_j ψ̃_j(ξ̃)|² is the clamped diffracted intensity.

        Note: R² cancels with the 1/R² in the Fraunhofer far-field, so
        we effectively compute 1 / (P̂_Ā · cos θ) × |ψ̃|².
        """
        if not aperture.has_diffraction:
            return 0.0

        xi = self.world_to_screen(aperture, wi_world)

        # Sum Psihat over all edges
        psi = Psihat_vectorized(aperture.edges, self.k, xi)

        # BSDF = |ψ̃|² / (cos θ × P̂_Ā)
        cos_theta = 1.0 / np.sqrt(1.0 + xi[0] ** 2 + xi[1] ** 2)
        intensity = abs(psi) ** 2

        return intensity / (cos_theta * aperture.P_A_hat)

    def eval_complex(
        self,
        wi_world: np.ndarray,      # [3] direction toward TX (world space)
        aperture: FsdAperture,
    ) -> complex:
        """
        Evaluate complex diffracted amplitude at direction wi.

        Returns the complex ψ̃(ξ̃) = Σ_j ψ̃_j(ξ̃) normalized by √(P̂_Ā · cos θ),
        so that |result|² gives the BSDF value f(ω_o).

        For MIMO coherence: the phase of this complex value varies correctly
        across TX-RX pairs due to the edge midpoint phase term exp(-ik·v·ξ).
        """
        if not aperture.has_diffraction:
            return 0.0 + 0.0j

        xi = self.world_to_screen(aperture, wi_world)

        psi = Psihat_vectorized(aperture.edges, self.k, xi)

        cos_theta = 1.0 / np.sqrt(1.0 + xi[0] ** 2 + xi[1] ** 2)

        # Normalize so that |result|² = f(ω_o)
        norm = np.sqrt(cos_theta * aperture.P_A_hat)
        if norm < 1e-20:
            return 0.0 + 0.0j

        return psi / norm

    def compute_beta(self, aperture: FsdAperture) -> float:
        """
        Compute the energy borrowing fraction β [Eq. in §3.7 of plan].

        β = P̂_Ā / P_in, clamped to [0, beta_max].

        P_in for a Gaussian beam with σ = beam_sigma is |E|² = 1 (normalized).
        The obstacle power P_Ā is already computed by the aperture.

        The "ratio of diffracted energy" D (Eq. S3.1) is:
           D = P̂_Ā / P_Ā ≤ 1

        And β = D × P_Ā / P_in. For our normalized beam, P_in = 1,
        so β = P̂_Ā.

        In practice, β is small near edges (typically 0.01-0.3) and zero
        away from edges.
        """
        if not aperture.has_diffraction:
            return 0.0

        beta = aperture.P_A_hat
        return min(beta, self.beta_max)

    def eval_at_hits_numpy(
        self,
        wi_world_arr: np.ndarray,    # [N, 3] directions toward TX
        apertures: list,             # List[FsdAperture] length N
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch evaluate fsdBSDF at multiple hit points (numpy, for prototyping).

        Returns:
            f_diff: [N] scalar BSDF values
            psi_real: [N] real part of complex amplitude
            psi_imag: [N] imaginary part of complex amplitude
        """
        N = len(apertures)
        f_diff = np.zeros(N, dtype=np.float64)
        psi_real = np.zeros(N, dtype=np.float64)
        psi_imag = np.zeros(N, dtype=np.float64)

        for i in range(N):
            ap = apertures[i]
            if not ap.has_diffraction:
                continue
            wi = wi_world_arr[i]
            f_diff[i] = self.eval(wi, ap)
            psi = self.eval_complex(wi, ap)
            psi_real[i] = psi.real
            psi_imag[i] = psi.imag

        return f_diff, psi_real, psi_imag


# ============================================================================
# Convenience: single-hit evaluation (for testing / validation)
# ============================================================================

def eval_single_slit(
    slit_width: float,
    wavelength: float,
    beam_sigma_wavelengths: float = 25.0,
    theta_range: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the fsdBSDF for a single slit (two parallel edges).

    Useful for validation against analytic Fraunhofer single-slit pattern.

    Args:
        slit_width: Width of the slit in meters.
        wavelength: Wavelength in meters.
        beam_sigma_wavelengths: Beam sigma in units of wavelength.
        theta_range: Array of scattering angles in radians.
            If None, uses linspace(-pi/4, pi/4, 1000).

    Returns:
        theta: Scattering angles [M].
        intensity: Normalized diffraction intensity [M].
    """
    if theta_range is None:
        theta_range = np.linspace(-np.pi / 4, np.pi / 4, 1000)

    k = 2.0 * np.pi / wavelength
    beam_sigma = beam_sigma_wavelengths * wavelength
    half_w = slit_width / 2.0

    # Two edges: left edge at -half_w, right edge at +half_w
    # Both are vertical (along y-axis), slit is along x-axis
    # Edge 1: left boundary, normal points right (+x)
    #   e = (0, slit_height), v = (-half_w, 0)
    # Edge 2: right boundary, normal points left (-x)
    #   e = (0, -slit_height), v = (+half_w, 0)  (reversed to point normal inward)

    # For a thin slit, the "obstacle" is the screen around the slit.
    # The aperture complement A̅⊥ extends from -inf to -half_w and +half_w to +inf.
    # Boundary edges are at x = ±half_w.

    # Simplified: two vertical boundary edges of height = 2*beam_sigma
    edge_height = 2.0 * beam_sigma

    # Build aperture manually
    aperture = FsdAperture(
        tangent=np.array([1, 0, 0], dtype=np.float32),
        bitangent=np.array([0, 1, 0], dtype=np.float32),
        wo_dir=np.array([0, 0, 1], dtype=np.float32),
        k=k,
        beam_sigma=beam_sigma,
    )

    # Edge 1: left boundary at x = -half_w
    e1 = np.array([0.0, edge_height], dtype=np.float32)
    v1 = np.array([-half_w, 0.0], dtype=np.float32)
    ph_left = np.exp(-0.25 * half_w ** 2 / beam_sigma ** 2) / (np.sqrt(2 * np.pi) * beam_sigma)
    a1 = complex(ph_left)
    b1 = complex(ph_left)

    # Edge 2: right boundary at x = +half_w, edge reversed so normal points left
    e2 = np.array([0.0, -edge_height], dtype=np.float32)
    v2 = np.array([half_w, 0.0], dtype=np.float32)
    ph_right = ph_left  # symmetric
    a2 = complex(ph_right)
    b2 = complex(ph_right)

    from .fsd_aperture import _Pjhat
    P1 = _Pjhat(a1, b1, e1)
    P2 = _Pjhat(a2, b2, e2)

    aperture.edges = [
        FsdEdge(e=e1, v=v1, a=a1, b=b1, Phat=P1, Phat_accum=P1),
        FsdEdge(e=e2, v=v2, a=a2, b=b2, Phat=P2, Phat_accum=P1 + P2),
    ]
    aperture.sum_Phat_j = P1 + P2
    aperture.P_A = 0.1  # approximate
    aperture.P_A_hat = max(0.01, P1 + P2)

    # Evaluate at each angle
    bsdf = FsdBSDF(k=k, beam_sigma=beam_sigma)
    intensity = np.zeros_like(theta_range)

    for i, theta in enumerate(theta_range):
        xi = np.array([np.tan(theta), 0.0])
        psi = Psihat_vectorized(aperture.edges, k, xi)
        intensity[i] = abs(psi) ** 2

    # Normalize to peak
    peak = np.max(intensity)
    if peak > 0:
        intensity /= peak

    return theta_range, intensity


# ============================================================================
# Flat edge data structure for vectorized batch evaluation
# ============================================================================

@dataclass
class FlatEdgeData:
    """All edges from all apertures packed into flat contiguous arrays (CSR format).

    Enables vectorized numpy evaluation of Psihat across ALL edges from ALL
    apertures in a single pass, with np.add.at segment reduction to sum
    per-aperture results.
    """
    # Per-edge arrays (E = total edges across all apertures)
    edge_ex: np.ndarray        # [E] float64 — edge vector x
    edge_ey: np.ndarray        # [E] float64 — edge vector y
    edge_vx: np.ndarray        # [E] float64 — midpoint x
    edge_vy: np.ndarray        # [E] float64 — midpoint y
    edge_a_real: np.ndarray    # [E] float64 — amplitude a, real part
    edge_a_imag: np.ndarray    # [E] float64 — amplitude a, imag part
    edge_b_real: np.ndarray    # [E] float64 — amplitude b, real part
    edge_b_imag: np.ndarray    # [E] float64 — amplitude b, imag part

    # Per-edge material data (for Phase B differentiable re-evaluation)
    edge_opacity: np.ndarray   # [E] float64 — Fresnel opacity from Phase A
    edge_face_idx: np.ndarray  # [E] int32 — global face index of owning triangle
    edge_cos_theta: np.ndarray # [E] float64 — cos(θ_i) at edge's triangle surface

    # Per-edge barycentric + vertex data (for per-vertex material interpolation in Phase B)
    edge_bary_u: np.ndarray    # [E] float64 — barycentric u at edge midpoint (-1 if not set)
    edge_bary_v: np.ndarray    # [E] float64 — barycentric v at edge midpoint (-1 if not set)
    edge_vert_idx_0: np.ndarray  # [E] int32 — vertex index 0 of owning triangle (-1 if not set)
    edge_vert_idx_1: np.ndarray  # [E] int32 — vertex index 1 of owning triangle
    edge_vert_idx_2: np.ndarray  # [E] int32 — vertex index 2 of owning triangle

    # Jones polarization data (for Phase B complex correction)
    edge_jones_real: np.ndarray   # [E] float64 — Re(E_jones) from Phase A
    edge_jones_imag: np.ndarray   # [E] float64 — Im(E_jones) from Phase A
    edge_tx_amp_s: np.ndarray     # [E] float64 — TX s-projection (frozen)
    edge_tx_amp_p: np.ndarray     # [E] float64 — TX p-projection (frozen)
    edge_rx_amp_s: np.ndarray     # [E] float64 — RX s-projection (frozen)
    edge_rx_amp_p: np.ndarray     # [E] float64 — RX p-projection (frozen)

    # Per-edge endpoint barycentrics (for AD vertex position re-projection in E2E)
    # Tracks actual tessellated endpoint positions within the owning triangle.
    edge_ba_u: np.ndarray      # [E] float64 — barycentric u of endpoint A
    edge_ba_v: np.ndarray      # [E] float64 — barycentric v of endpoint A
    edge_bb_u: np.ndarray      # [E] float64 — barycentric u of endpoint B
    edge_bb_v: np.ndarray      # [E] float64 — barycentric v of endpoint B

    # Per-aperture arrays (A = number of apertures with edges)
    ap_edge_start: np.ndarray  # [A] int32 — start index in edge arrays
    ap_edge_count: np.ndarray  # [A] int32 — number of edges
    ap_P_A_hat: np.ndarray     # [A] float64 — clamped diffracted power (opacity-weighted)
    ap_P_A_hat_bare: np.ndarray  # [A] float64 — bare geometric P_A_hat (no material modulation) — for beta
    ap_P_A_bare: np.ndarray      # [A] float64 — bare geometric P_A full (no material modulation) — for normalization
    ap_tangent: np.ndarray     # [A, 3] float32 — screen tangent
    ap_bitangent: np.ndarray   # [A, 3] float32 — screen bitangent
    ap_wo_dir: np.ndarray      # [A, 3] float32 — screen z-axis (wo)
    ap_beta: np.ndarray        # [A] float64 — energy borrowing fraction
    ap_hit_pos: np.ndarray     # [A, 3] float64 — aperture center in world space

    # Mapping: which flat path indices does each aperture correspond to?
    ap_to_path_indices: List[np.ndarray]  # [A] → variable-length arrays of flat indices

    n_edges: int
    n_apertures: int


def pack_apertures(
    apertures: List[Tuple[FsdAperture, List[int], float]],
    k: float,
) -> FlatEdgeData:
    """Pack variable-length apertures into flat contiguous arrays.

    Args:
        apertures: List of (aperture, flat_path_indices, beta) tuples.
        k: Wavenumber (unused here, reserved for future).

    Returns:
        FlatEdgeData with all edges packed in CSR format.
    """
    A = len(apertures)
    if A == 0:
        empty_f64 = np.empty(0, dtype=np.float64)
        empty_i32 = np.empty(0, dtype=np.int32)
        return FlatEdgeData(
            edge_ex=empty_f64, edge_ey=empty_f64, edge_vx=empty_f64, edge_vy=empty_f64,
            edge_a_real=empty_f64, edge_a_imag=empty_f64,
            edge_b_real=empty_f64, edge_b_imag=empty_f64,
            edge_opacity=empty_f64, edge_face_idx=empty_i32, edge_cos_theta=empty_f64,
            edge_bary_u=empty_f64, edge_bary_v=empty_f64,
            edge_vert_idx_0=empty_i32, edge_vert_idx_1=empty_i32, edge_vert_idx_2=empty_i32,
            edge_jones_real=empty_f64, edge_jones_imag=empty_f64,
            edge_tx_amp_s=empty_f64, edge_tx_amp_p=empty_f64,
            edge_rx_amp_s=empty_f64, edge_rx_amp_p=empty_f64,
            edge_ba_u=empty_f64, edge_ba_v=empty_f64,
            edge_bb_u=empty_f64, edge_bb_v=empty_f64,
            ap_edge_start=empty_i32,
            ap_edge_count=empty_i32,
            ap_P_A_hat=empty_f64, ap_P_A_hat_bare=empty_f64, ap_P_A_bare=empty_f64,
            ap_tangent=np.empty((0, 3), dtype=np.float32),
            ap_bitangent=np.empty((0, 3), dtype=np.float32),
            ap_wo_dir=np.empty((0, 3), dtype=np.float32),
            ap_beta=empty_f64,
            ap_hit_pos=np.empty((0, 3), dtype=np.float64),
            ap_to_path_indices=[], n_edges=0, n_apertures=0,
        )

    # Pre-compute counts and CSR offsets
    counts = np.array([len(ap.edges) for ap, _, _ in apertures], dtype=np.int32)
    total_edges = int(np.sum(counts))
    starts = np.empty(A, dtype=np.int32)
    starts[0] = 0
    if A > 1:
        np.cumsum(counts[:-1], out=starts[1:])

    # Per-aperture metadata (pre-allocated arrays, single pass)
    tangents = np.empty((A, 3), dtype=np.float32)
    bitangents = np.empty((A, 3), dtype=np.float32)
    wo_dirs = np.empty((A, 3), dtype=np.float32)
    P_A_hats = np.empty(A, dtype=np.float64)
    P_A_hat_bares = np.empty(A, dtype=np.float64)
    P_A_bares = np.empty(A, dtype=np.float64)
    betas = np.empty(A, dtype=np.float64)
    path_indices_list = []

    for a_idx, (aperture, path_idxs, beta_val) in enumerate(apertures):
        tangents[a_idx] = aperture.tangent
        bitangents[a_idx] = aperture.bitangent
        wo_dirs[a_idx] = aperture.wo_dir
        P_A_hats[a_idx] = aperture.P_A_hat
        P_A_hat_bares[a_idx] = aperture.P_A_hat_bare
        P_A_bares[a_idx] = aperture.P_A_bare
        betas[a_idx] = beta_val
        path_indices_list.append(np.asarray(path_idxs, dtype=np.int32))

    # Extract edge data using cached numpy arrays (O(A) slice copies, not O(E) element copies)
    all_ex = np.empty(total_edges, dtype=np.float64)
    all_ey = np.empty(total_edges, dtype=np.float64)
    all_vx = np.empty(total_edges, dtype=np.float64)
    all_vy = np.empty(total_edges, dtype=np.float64)
    all_ar = np.empty(total_edges, dtype=np.float64)
    all_ai = np.empty(total_edges, dtype=np.float64)
    all_br = np.empty(total_edges, dtype=np.float64)
    all_bi = np.empty(total_edges, dtype=np.float64)
    all_opacity = np.empty(total_edges, dtype=np.float64)
    all_face_idx = np.empty(total_edges, dtype=np.int32)
    all_cos_theta = np.empty(total_edges, dtype=np.float64)
    all_bary_u = np.empty(total_edges, dtype=np.float64)
    all_bary_v = np.empty(total_edges, dtype=np.float64)
    all_vi0 = np.empty(total_edges, dtype=np.int32)
    all_vi1 = np.empty(total_edges, dtype=np.int32)
    all_vi2 = np.empty(total_edges, dtype=np.int32)
    all_jones_real = np.empty(total_edges, dtype=np.float64)
    all_jones_imag = np.empty(total_edges, dtype=np.float64)
    all_tx_s = np.empty(total_edges, dtype=np.float64)
    all_tx_p = np.empty(total_edges, dtype=np.float64)
    all_rx_s = np.empty(total_edges, dtype=np.float64)
    all_rx_p = np.empty(total_edges, dtype=np.float64)

    for a_idx, (aperture, _, _) in enumerate(apertures):
        n = counts[a_idx]
        if n == 0:
            continue
        s = starts[a_idx]
        sl = slice(s, s + n)
        if aperture._cached_e is not None:
            # Fast path: use cached numpy arrays (memcpy-level speed)
            all_ex[sl] = aperture._cached_e[:, 0]
            all_ey[sl] = aperture._cached_e[:, 1]
            all_vx[sl] = aperture._cached_v[:, 0]
            all_vy[sl] = aperture._cached_v[:, 1]
            all_ar[sl] = aperture._cached_a.real
            all_ai[sl] = aperture._cached_a.imag
            all_br[sl] = aperture._cached_b.real
            all_bi[sl] = aperture._cached_b.imag
            if aperture._cached_opacity is not None:
                all_opacity[sl] = aperture._cached_opacity
                all_face_idx[sl] = aperture._cached_face_idx
                all_cos_theta[sl] = aperture._cached_cos_theta
            else:
                all_opacity[sl] = 1.0
                all_face_idx[sl] = -1
                all_cos_theta[sl] = 1.0
            if aperture._cached_bary_u is not None:
                all_bary_u[sl] = aperture._cached_bary_u
                all_bary_v[sl] = aperture._cached_bary_v
                all_vi0[sl] = aperture._cached_vert_idx_0
                all_vi1[sl] = aperture._cached_vert_idx_1
                all_vi2[sl] = aperture._cached_vert_idx_2
            else:
                all_bary_u[sl] = -1.0
                all_bary_v[sl] = -1.0
                all_vi0[sl] = -1
                all_vi1[sl] = -1
                all_vi2[sl] = -1
            if aperture._cached_jones_real is not None:
                all_jones_real[sl] = aperture._cached_jones_real
                all_jones_imag[sl] = aperture._cached_jones_imag
                all_tx_s[sl] = aperture._cached_tx_amp_s
                all_tx_p[sl] = aperture._cached_tx_amp_p
                all_rx_s[sl] = aperture._cached_rx_amp_s
                all_rx_p[sl] = aperture._cached_rx_amp_p
            else:
                all_jones_real[sl] = 1.0
                all_jones_imag[sl] = 0.0
                all_tx_s[sl] = 0.0
                all_tx_p[sl] = 0.0
                all_rx_s[sl] = 0.0
                all_rx_p[sl] = 0.0
        else:
            # Fallback: extract from Python objects
            for j, edge in enumerate(aperture.edges):
                idx = s + j
                all_ex[idx] = edge.e[0]
                all_ey[idx] = edge.e[1]
                all_vx[idx] = edge.v[0]
                all_vy[idx] = edge.v[1]
                all_ar[idx] = edge.a.real
                all_ai[idx] = edge.a.imag
                all_br[idx] = edge.b.real
                all_bi[idx] = edge.b.imag
                all_opacity[idx] = edge.opacity
                all_face_idx[idx] = edge.face_idx
                all_cos_theta[idx] = edge.cos_theta
                all_bary_u[idx] = edge.bary_u
                all_bary_v[idx] = edge.bary_v
                all_vi0[idx] = edge.vert_idx_0
                all_vi1[idx] = edge.vert_idx_1
                all_vi2[idx] = edge.vert_idx_2
                all_jones_real[idx] = edge.jones_real
                all_jones_imag[idx] = edge.jones_imag
                all_tx_s[idx] = edge.tx_amp_s
                all_tx_p[idx] = edge.tx_amp_p
                all_rx_s[idx] = edge.rx_amp_s
                all_rx_p[idx] = edge.rx_amp_p

    # Endpoint barycentrics: serial path doesn't track tessellation barycentrics,
    # so use the midpoint barycentrics as a default (approximate for tessellated edges)
    all_ba_u = np.copy(all_bary_u)
    all_ba_v = np.copy(all_bary_v)
    all_bb_u = np.copy(all_bary_u)
    all_bb_v = np.copy(all_bary_v)

    # Hit positions not available in serial path
    ap_hit_pos = np.zeros((A, 3), dtype=np.float64)

    return FlatEdgeData(
        edge_ex=all_ex, edge_ey=all_ey,
        edge_vx=all_vx, edge_vy=all_vy,
        edge_a_real=all_ar, edge_a_imag=all_ai,
        edge_b_real=all_br, edge_b_imag=all_bi,
        edge_opacity=all_opacity, edge_face_idx=all_face_idx, edge_cos_theta=all_cos_theta,
        edge_bary_u=all_bary_u, edge_bary_v=all_bary_v,
        edge_vert_idx_0=all_vi0, edge_vert_idx_1=all_vi1, edge_vert_idx_2=all_vi2,
        edge_jones_real=all_jones_real, edge_jones_imag=all_jones_imag,
        edge_tx_amp_s=all_tx_s, edge_tx_amp_p=all_tx_p,
        edge_rx_amp_s=all_rx_s, edge_rx_amp_p=all_rx_p,
        edge_ba_u=all_ba_u, edge_ba_v=all_ba_v,
        edge_bb_u=all_bb_u, edge_bb_v=all_bb_v,
        ap_edge_start=starts, ap_edge_count=counts,
        ap_P_A_hat=P_A_hats, ap_P_A_hat_bare=P_A_hat_bares, ap_P_A_bare=P_A_bares,
        ap_tangent=tangents,
        ap_bitangent=bitangents, ap_wo_dir=wo_dirs,
        ap_beta=betas,
        ap_hit_pos=ap_hit_pos,
        ap_to_path_indices=path_indices_list,
        n_edges=total_edges, n_apertures=A,
    )


def eval_psihat_batch_numpy(
    flat: FlatEdgeData,
    xi_per_aperture: np.ndarray,  # [A, 2] scattering direction per aperture
    k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch-evaluate Psihat for ALL edges at ALL scattering directions.

    One scattering direction per aperture; edges within each aperture share
    the same xi. Sums over edges per aperture using np.add.at.

    Args:
        flat: Packed edge data in CSR format.
        xi_per_aperture: [A, 2] screen-space scattering directions.
        k: Wavenumber 2π/λ.

    Returns:
        psi_real: [A] real part of summed Psihat per aperture.
        psi_imag: [A] imaginary part.
    """
    E = flat.n_edges
    A = flat.n_apertures

    if E == 0:
        return np.zeros(A, dtype=np.float64), np.zeros(A, dtype=np.float64)

    # Build edge-to-aperture index: each edge maps to its owning aperture
    edge_to_ap = np.repeat(np.arange(A, dtype=np.int32), flat.ap_edge_count)

    # Expand xi from [A, 2] to [E, 2] via gather
    xi_x = xi_per_aperture[edge_to_ap, 0]  # [E]
    xi_y = xi_per_aperture[edge_to_ap, 1]  # [E]

    # Edge data
    ex, ey = flat.edge_ex, flat.edge_ey
    vx, vy = flat.edge_vx, flat.edge_vy
    ar, ai = flat.edge_a_real, flat.edge_a_imag
    br, bi = flat.edge_b_real, flat.edge_b_imag

    # Edge length |e|
    ee = np.sqrt(ex * ex + ey * ey)  # [E]

    # v · ξ
    vxi = vx * xi_x + vy * xi_y  # [E]

    # Perpendicular to edge: m = (ey, -ex)
    mx, my = ey, -ex

    # Canonical space: ζ = k × [dot(e, ξ), dot(m, ξ)]
    zeta_x = k * (ex * xi_x + ey * xi_y)  # [E]
    zeta_y = k * (mx * xi_x + my * xi_y)  # [E]
    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y  # [E]

    # χ(|ζ|²)
    chi_val = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-0.5 * zeta_sq / 3.0)))

    # α₁, α₂ (vectorized with safe denominators)
    safe_zx = np.where(np.abs(zeta_x) < 1e-12, 1e-12, zeta_x)
    safe_sq = np.where(zeta_sq < 1e-20, 1e-20, zeta_sq)
    half_zx = safe_zx * 0.5
    abs_half = np.abs(half_zx)
    sinc_half = np.where(abs_half < 1e-8, 1.0, np.sin(half_zx) / half_zx)
    cos_half = np.cos(half_zx)
    inv_pi = 1.0 / np.pi

    a1 = zeta_y / safe_sq * inv_pi * (cos_half - sinc_half) / (2.0 * safe_zx)
    a2 = zeta_y / safe_sq * inv_pi * sinc_half / 2.0

    # Zero out degenerate cases
    degen = (zeta_sq < 1e-20) | (np.abs(zeta_x) < 1e-12)
    a1[degen] = 0.0
    a2[zeta_sq < 1e-20] = 0.0

    # Complex amplitude: (a-b)*α₁ + i*(a+b)/2*α₂
    # Split into real/imag pairs
    diff_r = (ar - br) * a1
    diff_i = (ai - bi) * a1

    # i * (a+b)/2 * α₂: i*(sr + i*si) = -si + i*sr
    sum_r_raw = (ar + br) * 0.5 * a2
    sum_i_raw = (ai + bi) * 0.5 * a2
    sum_r = -sum_i_raw   # real part of i*(sum)
    sum_i = sum_r_raw    # imag part of i*(sum)

    c_real = diff_r + sum_r  # [E]
    c_imag = diff_i + sum_i  # [E]

    # Phase: exp(-ik·v·ξ) = cos(-k·vxi) + i·sin(-k·vxi)
    phase_arg = -k * vxi
    phase_r = np.cos(phase_arg)
    phase_i = np.sin(phase_arg)

    # Psihat_j = k * |e|² * χ * exp(-ikv·ξ) * c
    scale = k * ee * ee * chi_val  # [E]
    pc_r = phase_r * c_real - phase_i * c_imag
    pc_i = phase_r * c_imag + phase_i * c_real
    psi_edge_r = scale * pc_r  # [E]
    psi_edge_i = scale * pc_i  # [E]

    # Sum per aperture using np.add.at (segment reduction)
    psi_real = np.zeros(A, dtype=np.float64)
    psi_imag = np.zeros(A, dtype=np.float64)
    np.add.at(psi_real, edge_to_ap, psi_edge_r)
    np.add.at(psi_imag, edge_to_ap, psi_edge_i)

    return psi_real, psi_imag


def eval_batch(
    flat: FlatEdgeData,
    wi_world: np.ndarray,   # [N, 3] TX directions per evaluation
    k: float,
    aperture_indices: np.ndarray = None,  # [N] int — which aperture each eval belongs to
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch-evaluate fsdBSDF for multiple (aperture, TX-direction) pairs at once.

    Each evaluation i uses aperture aperture_indices[i] and direction wi_world[i].
    Multiple evaluations can share the same aperture (different TX for same hit+RX).

    Args:
        flat: Packed edge data in CSR format.
        wi_world: [N, 3] TX directions per evaluation.
        k: Wavenumber 2π/λ.
        aperture_indices: [N] int — which aperture each eval uses.
            If None, assumes N == A and one eval per aperture.

    Returns:
        f_diff: [N] scalar BSDF values.
        psi_real: [N] normalized complex amplitude, real part.
        psi_imag: [N] normalized complex amplitude, imag part.
    """
    A = flat.n_apertures
    N = len(wi_world)

    if A == 0 or N == 0:
        return np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64)

    if aperture_indices is None:
        assert N == A, f"N={N} != A={A}; must provide aperture_indices"
        aperture_indices = np.arange(A, dtype=np.int32)

    # Project wi to screen coordinates per evaluation
    wi64 = wi_world.astype(np.float64)
    tang = flat.ap_tangent[aperture_indices].astype(np.float64)    # [N, 3]
    bitang = flat.ap_bitangent[aperture_indices].astype(np.float64)
    wo = flat.ap_wo_dir[aperture_indices].astype(np.float64)

    dx = np.sum(tang * wi64, axis=1)    # [N]
    dy = np.sum(bitang * wi64, axis=1)
    dz = np.sum(wo * wi64, axis=1)
    dz = np.where(np.abs(dz) < 1e-8, 1e-8, dz)

    xi = np.stack([dx / dz, dy / dz], axis=1)  # [N, 2]

    # Need to evaluate Psihat per (edge, direction) pair.
    # Expand edge data: for each evaluation i, replicate all edges of aperture_indices[i].
    E = flat.n_edges

    # Build edge-to-eval mapping: for each edge in the flat array, find which
    # evaluations use the aperture that owns this edge.
    # Strategy: build per-aperture xi, then use eval_psihat_batch_numpy
    # for each unique aperture. But that's O(A) loops.
    #
    # Better: if each aperture is evaluated at potentially multiple directions,
    # we need to expand. Create expanded edge arrays.

    # Count total expanded edges: sum over evals of edge_count[ap[i]]
    edge_counts_per_eval = flat.ap_edge_count[aperture_indices]  # [N]
    total_exp_edges = int(np.sum(edge_counts_per_eval))

    if total_exp_edges == 0:
        return np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.float64)

    # Vectorized edge expansion using np.repeat + fancy indexing (no Python loop)
    # For each evaluation i, replicate all edges of its aperture
    exp_eval_idx = np.repeat(np.arange(N, dtype=np.int32), edge_counts_per_eval)

    # Compute local edge offset within each evaluation's edge block
    eval_exp_starts = np.empty(N, dtype=np.int64)
    eval_exp_starts[0] = 0
    if N > 1:
        np.cumsum(edge_counts_per_eval[:-1], out=eval_exp_starts[1:])
    local_offset = np.arange(total_exp_edges, dtype=np.int64) - np.repeat(eval_exp_starts, edge_counts_per_eval)

    # Source edge index in flat arrays: ap_edge_start[ap[i]] + local_offset
    ap_starts_per_exp = flat.ap_edge_start[aperture_indices[exp_eval_idx]]
    src_edge_idx = (ap_starts_per_exp + local_offset).astype(np.int64)

    # Gather all edge data at once via fancy indexing
    exp_ex = flat.edge_ex[src_edge_idx]
    exp_ey = flat.edge_ey[src_edge_idx]
    exp_vx = flat.edge_vx[src_edge_idx]
    exp_vy = flat.edge_vy[src_edge_idx]
    exp_ar = flat.edge_a_real[src_edge_idx]
    exp_ai = flat.edge_a_imag[src_edge_idx]
    exp_br = flat.edge_b_real[src_edge_idx]
    exp_bi = flat.edge_b_imag[src_edge_idx]
    exp_xi_x = xi[exp_eval_idx, 0]
    exp_xi_y = xi[exp_eval_idx, 1]

    # Vectorized Psihat over all expanded edges
    ee = np.sqrt(exp_ex * exp_ex + exp_ey * exp_ey)
    vxi = exp_vx * exp_xi_x + exp_vy * exp_xi_y
    mx, my = exp_ey, -exp_ex

    zeta_x = k * (exp_ex * exp_xi_x + exp_ey * exp_xi_y)
    zeta_y = k * (mx * exp_xi_x + my * exp_xi_y)
    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y

    chi_val = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-0.5 * zeta_sq / 3.0)))

    safe_zx = np.where(np.abs(zeta_x) < 1e-12, 1e-12, zeta_x)
    safe_sq = np.where(zeta_sq < 1e-20, 1e-20, zeta_sq)
    half_zx = safe_zx * 0.5
    abs_half = np.abs(half_zx)
    sinc_half = np.where(abs_half < 1e-8, 1.0, np.sin(half_zx) / half_zx)
    cos_half = np.cos(half_zx)
    inv_pi = 1.0 / np.pi

    a1 = zeta_y / safe_sq * inv_pi * (cos_half - sinc_half) / (2.0 * safe_zx)
    a2 = zeta_y / safe_sq * inv_pi * sinc_half / 2.0

    degen = (zeta_sq < 1e-20) | (np.abs(zeta_x) < 1e-12)
    a1[degen] = 0.0
    a2[zeta_sq < 1e-20] = 0.0

    diff_r = (exp_ar - exp_br) * a1
    diff_i = (exp_ai - exp_bi) * a1
    sum_r_raw = (exp_ar + exp_br) * 0.5 * a2
    sum_i_raw = (exp_ai + exp_bi) * 0.5 * a2
    c_real = diff_r + (-sum_i_raw)
    c_imag = diff_i + sum_r_raw

    phase_arg = -k * vxi
    phase_r = np.cos(phase_arg)
    phase_i = np.sin(phase_arg)

    scale = k * ee * ee * chi_val
    pc_r = phase_r * c_real - phase_i * c_imag
    pc_i = phase_r * c_imag + phase_i * c_real
    psi_edge_r = scale * pc_r
    psi_edge_i = scale * pc_i

    # Sum per evaluation using np.add.at
    psi_real_sum = np.zeros(N, dtype=np.float64)
    psi_imag_sum = np.zeros(N, dtype=np.float64)
    np.add.at(psi_real_sum, exp_eval_idx, psi_edge_r)
    np.add.at(psi_imag_sum, exp_eval_idx, psi_edge_i)

    # cos(theta) = 1 / sqrt(1 + xi_x² + xi_y²)
    cos_theta = 1.0 / np.sqrt(1.0 + xi[:, 0] ** 2 + xi[:, 1] ** 2)

    # f_diff = |ψ|² / (cos_theta × P̂_Ā_bare)  — bare geometric normalization
    P_A_hat_bare = flat.ap_P_A_hat_bare[aperture_indices]  # [N]
    denom = np.maximum(cos_theta * P_A_hat_bare, 1e-4)
    intensity = psi_real_sum ** 2 + psi_imag_sum ** 2
    f_diff = intensity / denom

    # Normalized complex amplitude: ψ / √(cos_theta × P̂_Ā)
    norm = np.sqrt(denom)
    psi_norm_real = psi_real_sum / norm
    psi_norm_imag = psi_imag_sum / norm

    return f_diff, psi_norm_real, psi_norm_imag


def _fresnel_opacity_drjit(
    eps_real: 'mi.Float',
    eps_imag: 'mi.Float',
    cos_theta_i: 'mi.Float',
) -> 'mi.Float':
    """
    Differentiable Fresnel power reflectance averaged over polarization.

    opacity = (|R_s|² + |R_p|²) / 2

    Matches _fresnel_opacity_numpy() in fsd_aperture.py but uses DrJit
    for GPU execution and automatic differentiation.

    Args:
        eps_real: Real relative permittivity ε' (mi.Float).
        eps_imag: Imaginary permittivity ε'' (loss) (mi.Float).
        cos_theta_i: Cosine of incidence angle (positive) (mi.Float).

    Returns:
        mi.Float opacity in [0, 1].
    """
    import drjit as dr
    import mitsuba as mi

    # Convert permittivity to complex IOR: ñ = n - jκ
    # |ε| = √(ε'² + ε''²)
    eps_mag = dr.sqrt(eps_real * eps_real + eps_imag * eps_imag)
    # n = √((|ε| + ε') / 2), κ = √((|ε| - ε') / 2)
    n = dr.sqrt((eps_mag + eps_real) * mi.Float(0.5))
    kappa = dr.sqrt(dr.maximum(mi.Float(0.0), (eps_mag - eps_real) * mi.Float(0.5)))

    sin2 = mi.Float(1.0) - cos_theta_i * cos_theta_i

    # ñ² = n² - κ² - 2jnκ
    n2_real = n * n - kappa * kappa
    n2_imag = mi.Float(-2.0) * n * kappa

    # ξ = ñ² - sin²θ
    xi_real = n2_real - sin2
    xi_imag = n2_imag

    # √ξ = a + jb
    xi_mag = dr.sqrt(xi_real * xi_real + xi_imag * xi_imag)
    xi_arg = dr.atan2(xi_imag, xi_real)
    a = dr.sqrt(xi_mag) * dr.cos(xi_arg * mi.Float(0.5))
    b = dr.sqrt(xi_mag) * dr.sin(xi_arg * mi.Float(0.5))

    # R_s = |(cosθ - (a+jb)) / (cosθ + (a+jb))|²
    rs_num = (cos_theta_i - a) * (cos_theta_i - a) + b * b
    rs_den = dr.maximum((cos_theta_i + a) * (cos_theta_i + a) + b * b, mi.Float(1e-10))
    R_s = rs_num / rs_den

    # R_p = |(ñ²cosθ - (a+jb)) / (ñ²cosθ + (a+jb))|²
    n2cos_r = n2_real * cos_theta_i
    n2cos_i = n2_imag * cos_theta_i
    rp_num = (n2cos_r - a) * (n2cos_r - a) + (n2cos_i - b) * (n2cos_i - b)
    rp_den = dr.maximum((n2cos_r + a) * (n2cos_r + a) + (n2cos_i + b) * (n2cos_i + b), mi.Float(1e-10))
    R_p = rp_num / rp_den

    opacity = (R_s + R_p) * mi.Float(0.5)
    return dr.clip(opacity, mi.Float(0.0), mi.Float(1.0))


def _jones_reflectance_drjit(
    eps_real: 'mi.Float',     # per-edge ε'
    eps_imag: 'mi.Float',     # per-edge ε''
    cos_theta_i: 'mi.Float',  # per-edge cos(θ_i)
    tx_amp_s: 'mi.Float',     # per-edge TX s-projection (frozen)
    tx_amp_p: 'mi.Float',     # per-edge TX p-projection (frozen)
    rx_amp_s: 'mi.Float',     # per-edge RX s-projection (frozen)
    rx_amp_p: 'mi.Float',     # per-edge RX p-projection (frozen)
):
    """
    Differentiable Jones reflectance coefficient per edge.

    Same math as _jones_reflectance_numpy() but in DrJit for GPU execution
    and automatic differentiation. Gradients flow through r_s, r_p which
    depend on eps_real, eps_imag. The tx/rx projections are frozen geometry.

    Returns:
        (E_jones_real, E_jones_imag): Complex Jones coefficient (differentiable).
    """
    import drjit as dr
    import mitsuba as mi

    # Convert permittivity to complex IOR: ñ = n - jκ
    eps_mag = dr.sqrt(eps_real * eps_real + eps_imag * eps_imag)
    n = dr.sqrt((eps_mag + eps_real) * mi.Float(0.5))
    kappa = dr.sqrt(dr.maximum(mi.Float(0.0), (eps_mag - eps_real) * mi.Float(0.5)))

    sin2 = mi.Float(1.0) - cos_theta_i * cos_theta_i
    n2_real = n * n - kappa * kappa
    n2_imag = mi.Float(-2.0) * n * kappa

    xi_real = n2_real - sin2
    xi_imag = n2_imag
    xi_mag = dr.sqrt(xi_real * xi_real + xi_imag * xi_imag)
    xi_arg = dr.atan2(xi_imag, xi_real)
    a = dr.sqrt(xi_mag) * dr.cos(xi_arg * mi.Float(0.5))
    b = dr.sqrt(xi_mag) * dr.sin(xi_arg * mi.Float(0.5))

    # r_s = (cos_θ - (a+jb)) / (cos_θ + (a+jb))
    # Complex division: (p_r + j*p_i) / (q_r + j*q_i)
    rs_num_r = cos_theta_i - a
    rs_num_i = -b
    rs_den_r = cos_theta_i + a
    rs_den_i = b
    rs_den_sq = dr.maximum(rs_den_r * rs_den_r + rs_den_i * rs_den_i, mi.Float(1e-10))
    r_s_r = (rs_num_r * rs_den_r + rs_num_i * rs_den_i) / rs_den_sq
    r_s_i = (rs_num_i * rs_den_r - rs_num_r * rs_den_i) / rs_den_sq

    # r_p = (ñ²cos_θ - (a+jb)) / (ñ²cos_θ + (a+jb))
    n2cos_r = n2_real * cos_theta_i
    n2cos_i = n2_imag * cos_theta_i
    rp_num_r = n2cos_r - a
    rp_num_i = n2cos_i - b
    rp_den_r = n2cos_r + a
    rp_den_i = n2cos_i + b
    rp_den_sq = dr.maximum(rp_den_r * rp_den_r + rp_den_i * rp_den_i, mi.Float(1e-10))
    r_p_r = (rp_num_r * rp_den_r + rp_num_i * rp_den_i) / rp_den_sq
    r_p_i = (rp_num_i * rp_den_r - rp_num_r * rp_den_i) / rp_den_sq

    # E_jones = (r_s * tx_s) * rx_s + (r_p * tx_p) * rx_p
    # E_s_out = r_s * tx_s → (r_s_r*tx_s, r_s_i*tx_s)
    # E_p_out = r_p * tx_p → (r_p_r*tx_p, r_p_i*tx_p)
    # E_jones = E_s_out * rx_s + E_p_out * rx_p (complex addition)
    E_r = r_s_r * tx_amp_s * rx_amp_s + r_p_r * tx_amp_p * rx_amp_p
    E_i = r_s_i * tx_amp_s * rx_amp_s + r_p_i * tx_amp_p * rx_amp_p

    return E_r, E_i


def eval_batch_drjit(
    flat: FlatEdgeData,
    wi_world,   # np.ndarray [N, 3] OR tuple (mi.Float, mi.Float, mi.Float) [N]
    k: float,
    aperture_indices: np.ndarray,  # [N] int — which aperture each eval belongs to
    eps_real_all: 'mi.Float' = None,  # [n_faces] grad-enabled ε' for differentiable opacity
    eps_imag_all: 'mi.Float' = None,  # [n_faces] grad-enabled ε'' for differentiable opacity
    eps_real_vertex: 'mi.Float' = None,  # [n_vertices] grad-enabled ε' (per-vertex mode)
    eps_imag_vertex: 'mi.Float' = None,  # [n_vertices] grad-enabled ε'' (per-vertex mode)
    jones_mode: bool = False,  # If True, use Jones complex correction instead of scalar opacity
    # --- AD-attached geometry overrides (E2E pipeline) ---
    cos_theta_live: 'mi.Float' = None,  # [E_total] replaces flat.edge_cos_theta
    live_edge_ex: 'mi.Float' = None,    # [E_total] replaces flat.edge_ex
    live_edge_ey: 'mi.Float' = None,    # [E_total] replaces flat.edge_ey
    live_edge_vx: 'mi.Float' = None,    # [E_total] replaces flat.edge_vx
    live_edge_vy: 'mi.Float' = None,    # [E_total] replaces flat.edge_vy
):
    """
    GPU-accelerated eval_batch using DrJit CUDA kernels.

    Same interface as eval_batch() but runs Psihat math on GPU.
    Returns DrJit mi.Float arrays (stay on GPU, no CPU round-trip).

    When eps_real_all/eps_imag_all are provided, the edge amplitudes are
    re-modulated by differentiable Fresnel opacity, enabling gradients to
    flow through: loss → ADC → diffraction → opacity → ε_real, ε_imag.

    Args:
        flat: Packed edge data in CSR format (on CPU).
        wi_world: [N, 3] TX directions per evaluation. Either:
            - np.ndarray [N, 3]: numpy (detached, backward-compatible)
            - tuple (mi.Float, mi.Float, mi.Float): DrJit (AD-attached for pose gradients)
        k: Wavenumber 2π/λ.
        aperture_indices: [N] int — which aperture each eval uses.
        eps_real_all: [n_faces] DrJit mi.Float — grad-enabled real permittivity.
        eps_imag_all: [n_faces] DrJit mi.Float — grad-enabled imaginary permittivity.
        eps_real_vertex: [n_vertices] DrJit mi.Float — grad-enabled real permittivity (per-vertex).
        eps_imag_vertex: [n_vertices] DrJit mi.Float — grad-enabled imaginary permittivity (per-vertex).
        cos_theta_live: [E_total] mi.Float — AD-attached cos(θ) per edge (normal gradients).
            When provided, replaces frozen flat.edge_cos_theta in material correction.
        live_edge_ex/ey/vx/vy: [E_total] mi.Float — AD-attached screen-space edge geometry
            (vertex position gradients). When provided, replaces frozen flat.edge_ex/ey/vx/vy.

    Returns:
        f_diff: mi.Float [N] scalar BSDF values (on GPU).
        psi_real: mi.Float [N] normalized complex amplitude, real part.
        psi_imag: mi.Float [N] normalized complex amplitude, imag part.
    """
    import drjit as dr
    import mitsuba as mi

    A = flat.n_apertures

    # Handle wi_world as either numpy or DrJit tuple
    _wi_is_drjit = isinstance(wi_world, tuple)
    if _wi_is_drjit:
        N = dr.width(wi_world[0])
    else:
        N = len(wi_world)

    zero_n = dr.zeros(mi.Float, N)
    if A == 0 or N == 0:
        return zero_n, zero_n, zero_n

    # --- Upload edge data to GPU (float32) ---
    _has_live_edges = (live_edge_ex is not None and live_edge_ey is not None
                       and live_edge_vx is not None and live_edge_vy is not None)
    if _has_live_edges:
        g_ex = live_edge_ex   # [E_total] AD-attached mi.Float
        g_ey = live_edge_ey
        g_vx = live_edge_vx
        g_vy = live_edge_vy
    else:
        g_ex = mi.Float(flat.edge_ex.astype(np.float32))
        g_ey = mi.Float(flat.edge_ey.astype(np.float32))
        g_vx = mi.Float(flat.edge_vx.astype(np.float32))
        g_vy = mi.Float(flat.edge_vy.astype(np.float32))
    g_ar = mi.Float(flat.edge_a_real.astype(np.float32))
    g_ai = mi.Float(flat.edge_a_imag.astype(np.float32))
    g_br = mi.Float(flat.edge_b_real.astype(np.float32))
    g_bi = mi.Float(flat.edge_b_imag.astype(np.float32))

    # Per-aperture metadata on GPU
    g_tang_x = mi.Float(flat.ap_tangent[:, 0])
    g_tang_y = mi.Float(flat.ap_tangent[:, 1])
    g_tang_z = mi.Float(flat.ap_tangent[:, 2])
    g_bitang_x = mi.Float(flat.ap_bitangent[:, 0])
    g_bitang_y = mi.Float(flat.ap_bitangent[:, 1])
    g_bitang_z = mi.Float(flat.ap_bitangent[:, 2])
    g_wo_x = mi.Float(flat.ap_wo_dir[:, 0])
    g_wo_y = mi.Float(flat.ap_wo_dir[:, 1])
    g_wo_z = mi.Float(flat.ap_wo_dir[:, 2])
    g_P_A_hat = mi.Float(flat.ap_P_A_hat.astype(np.float32))
    g_P_A_hat_bare = mi.Float(flat.ap_P_A_hat_bare.astype(np.float32))
    g_P_A_bare = mi.Float(flat.ap_P_A_bare.astype(np.float32))

    # --- Compute expansion indices on CPU (integer math) ---
    edge_counts_per_eval = flat.ap_edge_count[aperture_indices]  # [N]
    total_exp_edges = int(np.sum(edge_counts_per_eval))

    if total_exp_edges == 0:
        return zero_n, zero_n, zero_n

    exp_eval_idx_np = np.repeat(np.arange(N, dtype=np.int32), edge_counts_per_eval)
    eval_exp_starts = np.empty(N, dtype=np.int64)
    eval_exp_starts[0] = 0
    if N > 1:
        np.cumsum(edge_counts_per_eval[:-1], out=eval_exp_starts[1:])
    local_offset_np = np.arange(total_exp_edges, dtype=np.int64) - \
        np.repeat(eval_exp_starts, edge_counts_per_eval)
    ap_starts_per_exp_np = flat.ap_edge_start[aperture_indices[exp_eval_idx_np]]
    src_edge_idx_np = (ap_starts_per_exp_np + local_offset_np).astype(np.int32)

    # Upload indices to GPU
    src_edge_idx = mi.UInt32(src_edge_idx_np)
    exp_eval_idx = mi.UInt32(exp_eval_idx_np)
    ap_idx = mi.UInt32(aperture_indices)

    # --- Projection on GPU ---
    if _wi_is_drjit:
        # AD-attached: use DrJit arrays directly (preserves gradient tape)
        wi_x, wi_y, wi_z = wi_world
    else:
        # Detached: create new mi.Float from numpy (backward-compatible)
        wi_x = mi.Float(wi_world[:, 0].astype(np.float32))
        wi_y = mi.Float(wi_world[:, 1].astype(np.float32))
        wi_z = mi.Float(wi_world[:, 2].astype(np.float32))

    tang_x = dr.gather(mi.Float, g_tang_x, ap_idx)
    tang_y = dr.gather(mi.Float, g_tang_y, ap_idx)
    tang_z = dr.gather(mi.Float, g_tang_z, ap_idx)
    bitang_x = dr.gather(mi.Float, g_bitang_x, ap_idx)
    bitang_y = dr.gather(mi.Float, g_bitang_y, ap_idx)
    bitang_z = dr.gather(mi.Float, g_bitang_z, ap_idx)
    wo_x = dr.gather(mi.Float, g_wo_x, ap_idx)
    wo_y = dr.gather(mi.Float, g_wo_y, ap_idx)
    wo_z = dr.gather(mi.Float, g_wo_z, ap_idx)

    dx = tang_x * wi_x + tang_y * wi_y + tang_z * wi_z
    dy = bitang_x * wi_x + bitang_y * wi_y + bitang_z * wi_z
    dz = wo_x * wi_x + wo_y * wi_y + wo_z * wi_z
    dz = dr.select(dr.abs(dz) < 1e-8, mi.Float(1e-8), dz)

    xi_x = dx / dz  # [N]
    xi_y = dy / dz  # [N]

    # --- Gather expanded edge data on GPU ---
    exp_ex = dr.gather(mi.Float, g_ex, src_edge_idx)
    exp_ey = dr.gather(mi.Float, g_ey, src_edge_idx)
    exp_vx = dr.gather(mi.Float, g_vx, src_edge_idx)
    exp_vy = dr.gather(mi.Float, g_vy, src_edge_idx)
    exp_ar = dr.gather(mi.Float, g_ar, src_edge_idx)
    exp_ai = dr.gather(mi.Float, g_ai, src_edge_idx)
    exp_br = dr.gather(mi.Float, g_br, src_edge_idx)
    exp_bi = dr.gather(mi.Float, g_bi, src_edge_idx)
    exp_xi_x = dr.gather(mi.Float, xi_x, exp_eval_idx)
    exp_xi_y = dr.gather(mi.Float, xi_y, exp_eval_idx)

    # --- Differentiable material correction (Phase B) ---
    # When grad-enabled material params are provided, re-compute Fresnel reflectance
    # and apply correction to amplitudes.
    #
    # Jones mode: complex correction E_jones_drjit / E_jones_numpy
    # Scalar mode: real correction √(opacity_drjit) / √(opacity_numpy)
    #
    # Material gathering modes:
    #   1. Per-vertex (eps_real_vertex/eps_imag_vertex): Barycentric interpolation at
    #      the edge midpoint. More accurate — each edge gets material at its actual
    #      position within the triangle, not the centroid average.
    #   2. Per-face (eps_real_all/eps_imag_all): Gather by face index. Fallback when
    #      per-vertex data or barycentrics are unavailable.
    _has_vertex_data = (eps_real_vertex is not None and eps_imag_vertex is not None
                        and flat.edge_bary_u is not None
                        and len(flat.edge_bary_u) > 0
                        and flat.edge_bary_u[0] >= 0)  # -1 = not set
    _has_face_data = (eps_real_all is not None and eps_imag_all is not None
                      and flat.edge_face_idx is not None)

    if _has_vertex_data or _has_face_data:
        # Gather per-edge cos_theta: use AD-attached if provided, else frozen Phase A
        if cos_theta_live is not None:
            exp_cos = dr.gather(mi.Float, cos_theta_live, src_edge_idx)
        else:
            edge_cos_gpu = mi.Float(flat.edge_cos_theta.astype(np.float32))
            exp_cos = dr.gather(mi.Float, edge_cos_gpu, src_edge_idx)

        if _has_vertex_data:
            # --- Per-vertex mode: barycentric interpolation at edge midpoint ---
            g_vi0 = mi.UInt32(flat.edge_vert_idx_0.astype(np.int32))
            g_vi1 = mi.UInt32(flat.edge_vert_idx_1.astype(np.int32))
            g_vi2 = mi.UInt32(flat.edge_vert_idx_2.astype(np.int32))
            g_bu = mi.Float(flat.edge_bary_u.astype(np.float32))
            g_bv = mi.Float(flat.edge_bary_v.astype(np.float32))

            exp_vi0 = dr.gather(mi.UInt32, g_vi0, src_edge_idx)
            exp_vi1 = dr.gather(mi.UInt32, g_vi1, src_edge_idx)
            exp_vi2 = dr.gather(mi.UInt32, g_vi2, src_edge_idx)
            exp_bu = dr.gather(mi.Float, g_bu, src_edge_idx)
            exp_bv = dr.gather(mi.Float, g_bv, src_edge_idx)
            exp_bw = mi.Float(1.0) - exp_bu - exp_bv

            eps_r_v0 = dr.gather(mi.Float, eps_real_vertex, exp_vi0)
            eps_r_v1 = dr.gather(mi.Float, eps_real_vertex, exp_vi1)
            eps_r_v2 = dr.gather(mi.Float, eps_real_vertex, exp_vi2)
            edge_eps_r = exp_bw * eps_r_v0 + exp_bu * eps_r_v1 + exp_bv * eps_r_v2

            eps_i_v0 = dr.gather(mi.Float, eps_imag_vertex, exp_vi0)
            eps_i_v1 = dr.gather(mi.Float, eps_imag_vertex, exp_vi1)
            eps_i_v2 = dr.gather(mi.Float, eps_imag_vertex, exp_vi2)
            edge_eps_i = exp_bw * eps_i_v0 + exp_bu * eps_i_v1 + exp_bv * eps_i_v2
        else:
            # --- Per-face mode: gather by face index ---
            edge_face_gpu = mi.UInt32(flat.edge_face_idx.astype(np.int32))
            exp_face_idx = dr.gather(mi.UInt32, edge_face_gpu, src_edge_idx)
            edge_eps_r = dr.gather(mi.Float, eps_real_all, exp_face_idx)
            edge_eps_i = dr.gather(mi.Float, eps_imag_all, exp_face_idx)

        if jones_mode:
            # --- Jones complex correction ---
            # Upload per-edge frozen TX/RX projections and Phase A Jones values
            g_jones_r = mi.Float(flat.edge_jones_real.astype(np.float32))
            g_jones_i = mi.Float(flat.edge_jones_imag.astype(np.float32))
            g_txs = mi.Float(flat.edge_tx_amp_s.astype(np.float32))
            g_txp = mi.Float(flat.edge_tx_amp_p.astype(np.float32))
            g_rxs = mi.Float(flat.edge_rx_amp_s.astype(np.float32))
            g_rxp = mi.Float(flat.edge_rx_amp_p.astype(np.float32))

            exp_jones_r_np = dr.gather(mi.Float, g_jones_r, src_edge_idx)
            exp_jones_i_np = dr.gather(mi.Float, g_jones_i, src_edge_idx)
            exp_tx_s = dr.gather(mi.Float, g_txs, src_edge_idx)
            exp_tx_p = dr.gather(mi.Float, g_txp, src_edge_idx)
            exp_rx_s = dr.gather(mi.Float, g_rxs, src_edge_idx)
            exp_rx_p = dr.gather(mi.Float, g_rxp, src_edge_idx)

            # Compute differentiable Jones reflectance
            E_dr_r, E_dr_i = _jones_reflectance_drjit(
                edge_eps_r, edge_eps_i, exp_cos,
                exp_tx_s, exp_tx_p, exp_rx_s, exp_rx_p
            )

            # Complex correction: E_jones_drjit / E_jones_numpy
            # (a+jb)/(c+jd) = ((ac+bd) + j(bc-ad)) / (c²+d²)
            denom = dr.maximum(
                exp_jones_r_np * exp_jones_r_np + exp_jones_i_np * exp_jones_i_np,
                mi.Float(1e-20)
            )
            corr_r = (E_dr_r * exp_jones_r_np + E_dr_i * exp_jones_i_np) / denom
            corr_i = (E_dr_i * exp_jones_r_np - E_dr_r * exp_jones_i_np) / denom

            # Apply complex correction to edge amplitudes
            new_ar = exp_ar * corr_r - exp_ai * corr_i
            new_ai = exp_ar * corr_i + exp_ai * corr_r
            new_br = exp_br * corr_r - exp_bi * corr_i
            new_bi = exp_br * corr_i + exp_bi * corr_r
            exp_ar, exp_ai = new_ar, new_ai
            exp_br, exp_bi = new_br, new_bi
        else:
            # --- Scalar opacity correction ---
            opacity_drjit = _fresnel_opacity_drjit(edge_eps_r, edge_eps_i, exp_cos)
            sqrt_opacity_drjit = dr.sqrt(dr.maximum(opacity_drjit, mi.Float(1e-10)))

            edge_opacity_gpu = mi.Float(flat.edge_opacity.astype(np.float32))
            exp_opacity_numpy = dr.gather(mi.Float, edge_opacity_gpu, src_edge_idx)
            sqrt_opacity_numpy = dr.sqrt(dr.maximum(exp_opacity_numpy, mi.Float(1e-10)))

            correction = sqrt_opacity_drjit / sqrt_opacity_numpy

            exp_ar = exp_ar * correction
            exp_ai = exp_ai * correction
            exp_br = exp_br * correction
            exp_bi = exp_bi * correction

    # --- Psihat math on GPU ---
    ee = dr.sqrt(exp_ex * exp_ex + exp_ey * exp_ey)
    vxi = exp_vx * exp_xi_x + exp_vy * exp_xi_y
    mx = exp_ey
    my = -exp_ex

    kf = mi.Float(k)
    zeta_x = kf * (exp_ex * exp_xi_x + exp_ey * exp_xi_y)
    zeta_y = kf * (mx * exp_xi_x + my * exp_xi_y)
    zeta_sq = zeta_x * zeta_x + zeta_y * zeta_y

    chi_val = dr.sqrt(dr.maximum(mi.Float(0.0),
                                 mi.Float(1.0) - dr.exp(mi.Float(-0.5) * zeta_sq / mi.Float(3.0))))

    safe_zx = dr.select(dr.abs(zeta_x) < 1e-12, mi.Float(1e-12), zeta_x)
    safe_sq = dr.select(zeta_sq < 1e-20, mi.Float(1e-20), zeta_sq)
    half_zx = safe_zx * mi.Float(0.5)
    abs_half = dr.abs(half_zx)
    sinc_half = dr.select(abs_half < 1e-8, mi.Float(1.0), dr.sin(half_zx) / half_zx)
    cos_half = dr.cos(half_zx)
    inv_pi = mi.Float(1.0 / np.pi)

    a1 = zeta_y / safe_sq * inv_pi * (cos_half - sinc_half) / (mi.Float(2.0) * safe_zx)
    a2 = zeta_y / safe_sq * inv_pi * sinc_half / mi.Float(2.0)

    degen = (zeta_sq < 1e-20) | (dr.abs(zeta_x) < 1e-12)
    a1 = dr.select(degen, mi.Float(0.0), a1)
    a2 = dr.select(zeta_sq < 1e-20, mi.Float(0.0), a2)

    diff_r = (exp_ar - exp_br) * a1
    diff_i = (exp_ai - exp_bi) * a1
    sum_r_raw = (exp_ar + exp_br) * mi.Float(0.5) * a2
    sum_i_raw = (exp_ai + exp_bi) * mi.Float(0.5) * a2
    c_real = diff_r + (-sum_i_raw)
    c_imag = diff_i + sum_r_raw

    phase_arg = -kf * vxi
    phase_r = dr.cos(phase_arg)
    phase_i = dr.sin(phase_arg)

    scale = kf * ee * ee * chi_val
    pc_r = phase_r * c_real - phase_i * c_imag
    pc_i = phase_r * c_imag + phase_i * c_real
    psi_edge_r = scale * pc_r
    psi_edge_i = scale * pc_i

    # --- Segment reduction on GPU ---
    psi_real_sum = dr.zeros(mi.Float, N)
    psi_imag_sum = dr.zeros(mi.Float, N)
    dr.scatter_reduce(dr.ReduceOp.Add, psi_real_sum, psi_edge_r, exp_eval_idx)
    dr.scatter_reduce(dr.ReduceOp.Add, psi_imag_sum, psi_edge_i, exp_eval_idx)
    dr.eval(psi_real_sum, psi_imag_sum)  # CRITICAL: eval before using scatter-modified vars

    # --- Normalization (use bare geometric P_A_hat — no material modulation) ---
    cos_theta = mi.Float(1.0) / dr.sqrt(mi.Float(1.0) + xi_x * xi_x + xi_y * xi_y)
    P_A_hat_bare = dr.gather(mi.Float, g_P_A_hat_bare, ap_idx)
    denom = dr.maximum(cos_theta * P_A_hat_bare, mi.Float(1e-4))
    intensity = psi_real_sum * psi_real_sum + psi_imag_sum * psi_imag_sum
    f_diff = intensity / denom

    norm_val = dr.sqrt(denom)
    psi_norm_real = psi_real_sum / norm_val
    psi_norm_imag = psi_imag_sum / norm_val

    return f_diff, psi_norm_real, psi_norm_imag
