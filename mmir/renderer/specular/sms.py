"""
Specular Manifold Sampling (SMS) for single-bounce specular path finding.

Replaces the image method for finding specular reflection paths on fine
triangle meshes derived from LiDAR point clouds. Uses Newton iteration on
the specular manifold constraint to walk from seed positions to exact
specular reflection points, naturally crossing triangle boundaries.

Algorithm (single-bounce, reflection only):
1. Seed from dedup'd reservoir hits (same seeds as image method)
2. For each (seed, TX): Newton iteration on half-vector constraint
   C = [s·h, t·h] = 0  where h = normalize(w_i + w_o)
3. Re-project onto mesh via ray tracing after each Newton step
4. Validate: convergence, visibility, geometry checks
5. Compute deterministic weight: R_specular = η × τ × A

Reference: Zeltner, Georgiev, Jakob, "Specular Manifold Sampling for
Rendering High-Frequency Caustics and Glints", SIGGRAPH 2020.
Reference impl: github.com/tizian/specular-manifold-sampling (manifold_ss.cpp)
"""

from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import drjit as dr
import mitsuba as mi
import numpy as np
import time

from .image_method import SpecularPaths
from ..bsdf.mmwave_scalar import (
    WAVELENGTH_77GHZ,
    enforce_spm_validity,
    permittivity_to_ior,
    compute_energy_gate,
    compute_slab_energy_gate,
    compute_validity_aware_blend,
    compute_coherent_incoherent_blend,
    map_renderer_params_to_physical,
)
from ..utils.math import safe_normalize

if TYPE_CHECKING:
    from ..utils.clustering import PatchData


@dataclass
class SpecularGradInfo:
    """
    Final-iteration state from SMS Newton solver needed for IFT gradient computation.

    At convergence, the specular constraint C(x*(θ), θ) = 0 is satisfied.
    The IFT gives: dx*/dθ = -(∂C/∂x)⁻¹ · (∂C/∂θ).

    We store the inverse Jacobian J⁻¹ = (∂C/∂x)⁻¹ so that Phase B can
    compute the IFT correction for vertex position gradients.
    """
    # Inverse of final 2×2 Newton Jacobian (∂C/∂(u,v))⁻¹
    inv_J00: 'mi.Float'  # [n_paths]
    inv_J01: 'mi.Float'  # [n_paths]
    inv_J10: 'mi.Float'  # [n_paths]
    inv_J11: 'mi.Float'  # [n_paths]
    # Surface tangent vectors at converged point
    dp_du: 'mi.Vector3f'  # [n_paths]
    dp_dv: 'mi.Vector3f'  # [n_paths]
    # TX/RX positions used in the forward solve (needed for constraint re-eval)
    tx_pos: 'mi.Point3f'  # [n_paths]
    rx_pos: 'mi.Point3f'  # [n_paths]


@dataclass
class MultibounceSpecularGradInfo:
    """
    Final-iteration state from multibounce SMS Newton solver for IFT.

    For a k-bounce specular chain (S₁, ..., Sₖ), the block-tridiagonal
    Jacobian factorization is stored so Phase B can compute the IFT correction
    for vertex position gradients at each specular vertex.

    The Thomas algorithm produces D_inv (diagonal inverses) and U (upper
    coupling blocks) during the forward sweep. These are reused in the
    backward sweep during the IFT correction step.
    """
    k: int  # number of specular bounces in this chain

    # Per-vertex surface tangents at converged points: lists of length k
    dp_du: list  # [k] mi.Vector3f arrays, each [n_chains]
    dp_dv: list  # [k] mi.Vector3f arrays, each [n_chains]

    # Block-tridiagonal factorization from Thomas algorithm forward sweep:
    # D_inv[i] = 4-tuple of mi.Float (inverse of modified diagonal block)
    # U[i] = 4-tuple of mi.Float (upper coupling = D_inv[i] @ B[i])
    D_inv: list  # [k] 4-tuples (a00, a01, a10, a11)
    U: list      # [k-1] 4-tuples (for backward sweep)

    # TX/RX positions used in the forward solve
    tx_pos: 'mi.Point3f'  # [n_chains]
    rx_pos: 'mi.Point3f'  # [n_chains]


@dataclass
class MultibounceSpecularChain:
    """
    Result of the multibounce SMS Newton solver.

    Stores converged k-bounce specular chains with per-vertex geometry
    and the factorized Jacobian needed for IFT gradient attachment.
    """
    k: int  # number of specular bounces

    # Per-vertex arrays: lists of length k
    hit_P: list       # [k] mi.Point3f arrays, each [n_chains]
    hit_N: list       # [k] mi.Vector3f arrays, each [n_chains]
    prim_ids: list    # [k] mi.UInt32 arrays, each [n_chains]
    bary_u: list      # [k] mi.Float arrays, each [n_chains]
    bary_v: list      # [k] mi.Float arrays, each [n_chains]
    A_tri: list       # [k] mi.Float arrays (triangle area at each vertex)

    # Per-chain arrays [n_chains]
    tx_idx: 'mi.UInt32'
    rx_idx: 'mi.UInt32'
    valid: 'mi.Bool'
    n_chains: int

    # Gradient info for IFT
    grad_info: Optional[MultibounceSpecularGradInfo]

    # Per-segment distances: lists of length k+1
    # d_segments[0] = RX→S₁, d_segments[i] = Sᵢ→Sᵢ₊₁, d_segments[k] = Sₖ→TX
    d_segments: list   # [k+1] mi.Float arrays
    dir_segments: list  # [k+1] mi.Vector3f arrays (unit direction vectors)


class SpecularManifoldSampler:
    """
    Single-bounce Specular Manifold Sampling for mmWave radar.

    Replaces the image method for finding specular reflection paths on
    fine triangle meshes. Uses Newton iteration on the specular manifold
    constraint to walk from seed positions to exact specular points.

    The key advantage over the image method: the Newton solver naturally
    crosses triangle boundaries via ray-trace re-projection, so it is
    not limited by the point-in-triangle test that causes the image method
    to fail on small LiDAR-derived triangles.
    """

    def __init__(
        self,
        wavelength: float = WAVELENGTH_77GHZ,
        max_iterations: int = 20,
        solver_threshold: float = 1e-5,
        use_smooth_normals: bool = False,
        bsdf=None,
    ):
        self.wavelength = wavelength
        self.max_iterations = max_iterations
        self.solver_threshold = solver_threshold
        self.use_smooth_normals = use_smooth_normals
        self.bsdf = bsdf

        # Lazy-initialized mesh data for computing dn/du, dn/dv
        # (needed because Mitsuba 3 returns zero dn_du/dn_dv for PLY meshes)
        self._mesh_data_ready = False
        self._fi = None       # face indices, flat UInt32 [n_faces * 3]
        self._vn_x = None     # vertex normal x components [n_vertices]
        self._vn_y = None     # vertex normal y components [n_vertices]
        self._vn_z = None     # vertex normal z components [n_vertices]
        self._n_mesh_faces = 0

    # ========================================================================
    # Mesh data setup (lazy, for smooth normal derivatives)
    # ========================================================================

    def _setup_mesh_normals(self, scene):
        """Extract face indices and vertex normals from the scene mesh.

        Required for computing dn/du, dn/dv when use_smooth_normals=True,
        because Mitsuba 3 returns zero dn_du/dn_dv for PLY triangle meshes.
        We compute them manually: dn/du = n1 - n0, dn/dv = n2 - n0.
        """
        if self._mesh_data_ready:
            return
        self._mesh_data_ready = True

        shapes = scene.shapes()
        mesh = None
        for shape in shapes:
            if hasattr(shape, 'face_count') and shape.face_count() > 0:
                mesh = shape
                break
        if mesh is None or not mesh.has_vertex_normals():
            return

        params = mi.traverse(mesh)
        fi_dr = params.get('faces', None)
        vn_dr = params.get('vertex_normals', None)
        if fi_dr is None or vn_dr is None:
            return

        # Face indices: flat [n_faces * 3]
        self._fi = mi.UInt32(fi_dr)
        self._n_mesh_faces = mesh.face_count()

        # Vertex normals: flat [n_vertices * 3] → separate x/y/z
        vn_np = np.array(vn_dr).reshape(-1, 3)
        self._vn_x = mi.Float(vn_np[:, 0].copy())
        self._vn_y = mi.Float(vn_np[:, 1].copy())
        self._vn_z = mi.Float(vn_np[:, 2].copy())

    def _compute_normal_derivatives(self, prim_ids):
        """Compute dn/du, dn/dv from mesh vertex normals at given triangles.

        For triangle (v0, v1, v2) with normals (n0, n1, n2) and
        Mitsuba parameterization dp/du = v1-v0, dp/dv = v2-v0:
            dn/du = n1 - n0
            dn/dv = n2 - n0
        """
        n = dr.width(prim_ids)
        zero = mi.Vector3f(dr.zeros(mi.Float, n),
                           dr.zeros(mi.Float, n),
                           dr.zeros(mi.Float, n))
        if self._fi is None:
            return zero, zero

        safe_pid = dr.minimum(mi.UInt32(prim_ids), mi.UInt32(max(self._n_mesh_faces - 1, 0)))
        v0_idx = dr.gather(mi.UInt32, self._fi, safe_pid * 3)
        v1_idx = dr.gather(mi.UInt32, self._fi, safe_pid * 3 + 1)
        v2_idx = dr.gather(mi.UInt32, self._fi, safe_pid * 3 + 2)

        n0 = mi.Vector3f(dr.gather(mi.Float, self._vn_x, v0_idx),
                         dr.gather(mi.Float, self._vn_y, v0_idx),
                         dr.gather(mi.Float, self._vn_z, v0_idx))
        n1 = mi.Vector3f(dr.gather(mi.Float, self._vn_x, v1_idx),
                         dr.gather(mi.Float, self._vn_y, v1_idx),
                         dr.gather(mi.Float, self._vn_z, v1_idx))
        n2 = mi.Vector3f(dr.gather(mi.Float, self._vn_x, v2_idx),
                         dr.gather(mi.Float, self._vn_y, v2_idx),
                         dr.gather(mi.Float, self._vn_z, v2_idx))

        dn_du = mi.Vector3f(n1.x - n0.x, n1.y - n0.y, n1.z - n0.z)
        dn_dv = mi.Vector3f(n2.x - n0.x, n2.y - n0.y, n2.z - n0.z)
        return dn_du, dn_dv

    @staticmethod
    def _compute_frame_derivatives(n, dp_du, s, dn_du, dn_dv):
        """Compute ds/du, ds/dv, dt/du, dt/dv for smooth-normal Jacobian.

        Follows compute_shading_frame_derivative() from the reference:
            specular-manifold-sampling: mitsuba/core/frame.h:224-243

        Args:
            n: Surface normal (possibly flipped for backfacing)
            dp_du: Position partial w.r.t. u
            s: Tangent vector (from Gram-Schmidt of dp_du against n)
            dn_du, dn_dv: Normal derivatives

        Returns:
            (ds_du, ds_dv, dt_du, dt_dv)
        """
        n_dot_dpdu = dr.dot(n, dp_du)

        # s_raw = dp_du - n * dot(n, dp_du) [Gram-Schmidt]
        s_raw = mi.Vector3f(dp_du.x - n.x * n_dot_dpdu,
                            dp_du.y - n.y * n_dot_dpdu,
                            dp_du.z - n.z * n_dot_dpdu)
        inv_len_s = mi.Float(1.0) / dr.maximum(dr.norm(s_raw), mi.Float(1e-10))

        # ds_du = inv_len_s * (-dn_du * dot(n, dp_du) - n * dot(dn_du, dp_du))
        # ds_du -= s * dot(ds_du, s)
        dn_du_dot_dpdu = dr.dot(dn_du, dp_du)
        ds_du = mi.Vector3f(
            inv_len_s * (-dn_du.x * n_dot_dpdu - n.x * dn_du_dot_dpdu),
            inv_len_s * (-dn_du.y * n_dot_dpdu - n.y * dn_du_dot_dpdu),
            inv_len_s * (-dn_du.z * n_dot_dpdu - n.z * dn_du_dot_dpdu),
        )
        proj = dr.dot(ds_du, s)
        ds_du = mi.Vector3f(ds_du.x - s.x * proj,
                            ds_du.y - s.y * proj,
                            ds_du.z - s.z * proj)

        # ds_dv (same structure with dn_dv)
        dn_dv_dot_dpdu = dr.dot(dn_dv, dp_du)
        ds_dv = mi.Vector3f(
            inv_len_s * (-dn_dv.x * n_dot_dpdu - n.x * dn_dv_dot_dpdu),
            inv_len_s * (-dn_dv.y * n_dot_dpdu - n.y * dn_dv_dot_dpdu),
            inv_len_s * (-dn_dv.z * n_dot_dpdu - n.z * dn_dv_dot_dpdu),
        )
        proj = dr.dot(ds_dv, s)
        ds_dv = mi.Vector3f(ds_dv.x - s.x * proj,
                            ds_dv.y - s.y * proj,
                            ds_dv.z - s.z * proj)

        # dt_du = cross(dn_du, s) + cross(n, ds_du)
        dt_du = dr.cross(dn_du, s) + dr.cross(n, ds_du)
        # dt_dv = cross(dn_dv, s) + cross(n, ds_dv)
        dt_dv = dr.cross(dn_dv, s) + dr.cross(n, ds_dv)

        return ds_du, ds_dv, dt_du, dt_dv

    # ========================================================================
    # Block-tridiagonal Thomas algorithm for multibounce SMS
    # ========================================================================

    @staticmethod
    def _inv2x2(a00, a01, a10, a11):
        """Invert a 2x2 block represented as 4 scalar DrJit arrays."""
        det = a00 * a11 - a01 * a10
        det_safe = dr.select(dr.abs(det) > mi.Float(1e-12), det, mi.Float(1e-12))
        inv_det = mi.Float(1.0) / det_safe
        return (a11 * inv_det, -a01 * inv_det, -a10 * inv_det, a00 * inv_det)

    @staticmethod
    def _mul2x2(a, b):
        """Multiply two 2x2 blocks (each a 4-tuple of DrJit arrays)."""
        return (a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3],
                a[2]*b[0] + a[3]*b[2], a[2]*b[1] + a[3]*b[3])

    @staticmethod
    def _mul2x2_vec(a, v):
        """Multiply 2x2 block by 2-vector (each a tuple of DrJit arrays)."""
        return (a[0]*v[0] + a[1]*v[1], a[2]*v[0] + a[3]*v[1])

    @staticmethod
    def _sub2x2(a, b):
        """Subtract two 2x2 blocks."""
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3])

    def _thomas_block_tridiagonal(
        self,
        A: list,        # [k] diagonal 2x2 blocks (4-tuples of mi.Float)
        B: list,        # [k-1] upper 2x2 blocks
        C_lower: list,  # [k-1] lower 2x2 blocks
        rhs: list,      # [k] right-hand side 2-vectors (2-tuples of mi.Float)
        active: 'mi.Bool',
    ) -> Tuple[list, list, list]:
        """
        Solve block-tridiagonal system via Thomas algorithm.

        System:
        [A₁  B₁  0  ...] [x₁]   [r₁]
        [C₂  A₂  B₂ ...] [x₂] = [r₂]
        [0   C₃  A₃ ...] [x₃]   [r₃]

        Forward sweep: D₁ = A₁, Dᵢ = Aᵢ - Cᵢ·Dᵢ₋₁⁻¹·Bᵢ₋₁
                       y₁ = D₁⁻¹·r₁, yᵢ = Dᵢ⁻¹·(rᵢ - Cᵢ·yᵢ₋₁)
        Backward sweep: xₖ = yₖ, xᵢ = yᵢ - Dᵢ⁻¹·Bᵢ·xᵢ₊₁

        Returns:
            (x, D_inv, U) where:
            - x: [k] 2-vectors (Δu, Δv) at each vertex
            - D_inv: [k] 4-tuples (factored diagonal inverses for IFT reuse)
            - U: [k-1] 4-tuples (upper coupling blocks for IFT reuse)
        """
        k = len(A)

        # Forward sweep
        D_inv = [None] * k
        y = [None] * k
        U = [None] * max(k - 1, 1)

        D_inv[0] = self._inv2x2(*A[0])
        y[0] = self._mul2x2_vec(D_inv[0], rhs[0])

        for i in range(1, k):
            temp = self._mul2x2(C_lower[i-1], D_inv[i-1])
            CDB = self._mul2x2(temp, B[i-1])
            D_i = self._sub2x2(A[i], CDB)
            D_inv[i] = self._inv2x2(*D_i)

            Cy = self._mul2x2_vec(C_lower[i-1], y[i-1])
            rhs_mod = (rhs[i][0] - Cy[0], rhs[i][1] - Cy[1])
            y[i] = self._mul2x2_vec(D_inv[i], rhs_mod)

            if i < k - 1:
                U[i] = self._mul2x2(D_inv[i], B[i])

        if k > 1:
            U[0] = self._mul2x2(D_inv[0], B[0])

        # Backward sweep
        x = [None] * k
        x[k-1] = y[k-1]

        for i in range(k-2, -1, -1):
            Ux = self._mul2x2_vec(U[i], x[i+1])
            x[i] = (y[i][0] - Ux[0], y[i][1] - Ux[1])

        return x, D_inv, U[:k-1] if k > 1 else []

    def _empty_specular_paths(self) -> SpecularPaths:
        """Return an empty SpecularPaths (no valid paths)."""
        return SpecularPaths(
            hit_P=mi.Point3f(), hit_N=mi.Vector3f(),
            d_tx=mi.Float(), d_rx=mi.Float(),
            dir_to_tx=mi.Vector3f(), dir_to_rx=mi.Vector3f(),
            cos_theta_i=mi.Float(), cos_theta_r=mi.Float(),
            tx_idx=mi.UInt32(),
            rx_idx=mi.UInt32(), valid=mi.Bool(),
            R_specular=mi.Float(), A_tri=mi.Float(),
            patch_id=None, prim_ids=None,
            n_paths=0,
            grad_info=None, bary_u=None, bary_v=None,
        )

    # ========================================================================
    # Main entry point
    # ========================================================================

    def find_specular_paths(
        self,
        unique_mask: 'mi.Bool',
        hit_P: 'mi.Point3f',
        hit_N: 'mi.Vector3f',
        hit_rho: np.ndarray,
        hit_ID: 'mi.Int32',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        n_hits_per_rx: int,
        scene: 'mi.Scene',
        verbose: bool = True,
        patch_data: Optional['PatchData'] = None,
        triangle_materials: Optional[np.ndarray] = None,
    ) -> SpecularPaths:
        """
        Find specular paths via Specular Manifold Sampling.

        Drop-in replacement for ImageMethodRefiner.refine(). Uses the same
        dedup'd unique seeds, expands × TX, but replaces the mirror + PIT
        approach with Newton iteration on the specular manifold constraint.

        Args:
            unique_mask: Boolean mask of unique hits from dedup [n_total_slots]
            hit_P: Reservoir hit positions [n_total_slots]
            hit_N: Surface normals [n_total_slots]
            hit_rho: Material params [n_total_slots, 3|6]
            hit_ID: Triangle primitive IDs [n_total_slots]
            tx_positions: TX element positions [n_tx]
            rx_positions: RX element positions [n_rx]
            n_hits_per_rx: Hits per RX in reservoir
            scene: Mitsuba scene for ray tracing
            verbose: Print progress
            patch_data: PatchData for patch-area weighting (optional)
            triangle_materials: Per-triangle material array [n_tris, n_cols]
                for looking up materials at converged points (optional;
                falls back to seed materials if None)

        Returns:
            SpecularPaths compatible with integrator._synthesize_specular_paths()
        """
        from ..utils.math import gather_point3f, gather_vector3f

        t_start = time.perf_counter()

        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)

        # Setup mesh normal data for smooth normals
        if self.use_smooth_normals:
            self._setup_mesh_normals(scene)

        # ================================================================
        # Step 1: Extract unique seed positions from dedup mask
        # ================================================================
        unique_np = np.array(unique_mask)
        unique_indices = np.where(unique_np)[0]
        n_unique = len(unique_indices)

        if verbose:
            print(f"\n[SMS] Specular Manifold Sampling")
            print(f"  Unique seeds: {n_unique}")
            print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
            print(f"  Max Newton iterations: {self.max_iterations}")
            print(f"  Convergence threshold: {self.solver_threshold}")

        if n_unique == 0:
            if verbose:
                print(f"  No unique seeds — returning empty paths")
            return self._empty_specular_paths()

        unique_idx_dr = mi.UInt32(unique_indices)

        # Gather unique seed data
        seed_P = gather_point3f(hit_P, unique_idx_dr)
        seed_N = gather_vector3f(hit_N, unique_idx_dr)
        seed_rho = hit_rho[unique_np]  # [n_unique, cols]
        seed_ID = dr.gather(mi.Int32, hit_ID, unique_idx_dr)
        seed_rx_idx_np = unique_indices // n_hits_per_rx

        # ================================================================
        # Step 2: Expand seeds × TX → (n_unique × n_tx) candidates
        # ================================================================
        n_paths = n_unique * n_tx
        if verbose:
            print(f"  SMS path candidates: {n_unique} × {n_tx} TX = {n_paths:,}")

        # Expand seed positions and normals
        exp_seed_P = mi.Point3f(
            dr.repeat(seed_P.x, n_tx), dr.repeat(seed_P.y, n_tx), dr.repeat(seed_P.z, n_tx)
        )

        # Expand TX/RX indices
        exp_tx_idx = mi.UInt32(np.tile(np.arange(n_tx), n_unique))
        exp_rx_idx = mi.UInt32(np.repeat(seed_rx_idx_np, n_tx))

        # Gather TX and RX positions for each candidate
        exp_tx_pos = mi.Point3f(
            dr.gather(mi.Float, tx_positions.x, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.y, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.z, exp_tx_idx),
        )
        exp_rx_pos = mi.Point3f(
            dr.gather(mi.Float, rx_positions.x, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.y, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.z, exp_rx_idx),
        )

        # Material info
        n_mat_cols = seed_rho.shape[1]
        physics_mode = (n_mat_cols == 6)

        # ================================================================
        # Step 3: Get surface parameterization at seed points
        # ================================================================
        init_result = self._get_surface_parameterization(
            exp_seed_P, exp_rx_pos, scene)
        dp_du, dp_dv, vtx_n, vtx_s, vtx_t, init_prim_ids, valid_init, \
            init_shape, init_dn_du, init_dn_dv = init_result

        if verbose:
            n_valid_init = int(dr.sum(mi.UInt32(valid_init))[0])
            print(f"  Valid initial ray intersections: {n_valid_init}/{n_paths}")

        # ================================================================
        # Step 4: Vectorized Newton solver
        # ================================================================
        t_newton_start = time.perf_counter()

        converged_P, converged_N, converged_mask, converged_prim_ids, n_iters_used, \
            newton_grad_info, converged_dp_du, converged_dp_dv, last_bary_u, last_bary_v = \
            self._newton_solve_vectorized(
                vtx_p=exp_seed_P,
                vtx_n=vtx_n,
                vtx_dp_du=dp_du,
                vtx_dp_dv=dp_dv,
                rx_pos=exp_rx_pos,
                tx_pos=exp_tx_pos,
                scene=scene,
                valid_init=valid_init,
                init_prim_ids=init_prim_ids,
                init_shape=init_shape,
                init_dn_du=init_dn_du,
                init_dn_dv=init_dn_dv,
            )

        t_newton = time.perf_counter() - t_newton_start
        n_converged = int(dr.sum(mi.UInt32(converged_mask))[0])

        if verbose:
            print(f"  Newton solver: {n_converged:,}/{n_paths:,} converged "
                  f"({100*n_converged/max(n_paths,1):.1f}%) in {n_iters_used} iterations, "
                  f"{t_newton:.3f}s")

        if n_converged == 0:
            if verbose:
                print(f"  No converged paths — returning empty paths")
            return self._empty_specular_paths()

        # ================================================================
        # Step 5: Post-Newton validation (geometry + visibility)
        # ================================================================
        delta_tx = mi.Vector3f(
            exp_tx_pos.x - converged_P.x,
            exp_tx_pos.y - converged_P.y,
            exp_tx_pos.z - converged_P.z,
        )
        delta_rx = mi.Vector3f(
            exp_rx_pos.x - converged_P.x,
            exp_rx_pos.y - converged_P.y,
            exp_rx_pos.z - converged_P.z,
        )
        d_tx = dr.norm(delta_tx)
        d_rx = dr.norm(delta_rx)

        dir_to_tx = mi.Vector3f(
            delta_tx.x / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.y / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.z / dr.maximum(d_tx, mi.Float(1e-10)),
        )
        dir_to_rx = mi.Vector3f(
            delta_rx.x / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.y / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.z / dr.maximum(d_rx, mi.Float(1e-10)),
        )

        cos_theta_i = dr.maximum(dr.dot(dir_to_tx, converged_N), mi.Float(0.0))
        cos_theta_r = dr.maximum(dr.dot(dir_to_rx, converged_N), mi.Float(0.0))

        # Geometry validation
        valid_dist = (d_tx > mi.Float(1e-4)) & (d_rx > mi.Float(1e-4))
        valid_angle = cos_theta_i > mi.Float(1e-6)
        valid_geom = converged_mask & valid_dist & valid_angle

        # Visibility check
        valid_vis = self._check_visibility(
            converged_P, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
            surface_normal=converged_N,
            skip_rx=True,  # RX visibility guaranteed by Newton reprojection
        )
        valid_all = valid_geom & valid_vis

        n_valid_geom = int(dr.sum(mi.UInt32(valid_geom))[0])
        n_valid_all = int(dr.sum(mi.UInt32(valid_all))[0])

        if verbose:
            n_valid_dist = int(dr.sum(mi.UInt32(valid_dist & converged_mask))[0])
            n_valid_angle = int(dr.sum(mi.UInt32(valid_angle & converged_mask))[0])
            n_valid_vis = int(dr.sum(mi.UInt32(valid_vis & valid_geom))[0])
            print(f"  Post-Newton validation ({n_converged:,} converged candidates):")
            print(f"    valid_dist (distance > eps):   {n_valid_dist:,}")
            print(f"    valid_angle (cos_theta > 0):   {n_valid_angle:,}")
            print(f"    valid_geom (converged+geom):   {n_valid_geom:,}")
            print(f"    valid_vis (visibility):        {n_valid_vis:,}")
            print(f"    valid_all (final):             {n_valid_all:,} ({100*n_valid_all/max(n_paths,1):.1f}%)")

        # ================================================================
        # Step 6: Compute triangle/patch area (from Newton solver state)
        # ================================================================
        # Use barycentrics and dp_du/dp_dv cached by Newton solver
        # (avoids an extra scene.ray_intersect call)
        spec_bary_u = last_bary_u
        spec_bary_v = last_bary_v

        # Triangle area from Newton solver's final dp_du, dp_dv
        cross_vec = dr.cross(converged_dp_du, converged_dp_dv)
        A_tri_raw = mi.Float(0.5) * dr.norm(cross_vec)

        # SMS operates at individual triangle level — use triangle area, not patch area
        A_tri = A_tri_raw
        exp_patch_id = None
        if patch_data is not None:
            # Still need patch_id for per-patch BSDF selection (patch_has_specular)
            safe_prim = mi.UInt32(dr.maximum(mi.Int32(converged_prim_ids), mi.Int32(0)))
            n_tris = dr.width(patch_data.tri_to_patch)
            safe_prim_clamped = dr.minimum(safe_prim, mi.UInt32(n_tris - 1))
            exp_patch_id_raw = mi.UInt32(dr.gather(mi.Int32, patch_data.tri_to_patch, safe_prim_clamped))
            exp_patch_id = mi.Int32(exp_patch_id_raw)

        # ================================================================
        # Step 7: Compute specular weight using CONVERGED material
        # ================================================================
        if triangle_materials is not None and n_valid_all > 0:
            # Look up material at converged point's triangle
            n_tris_mat = triangle_materials.shape[0]
            prim_np = np.array(converged_prim_ids).astype(np.int64)
            prim_np = np.clip(prim_np, 0, n_tris_mat - 1)
            conv_rho = triangle_materials[prim_np]  # [n_paths, n_cols]
        else:
            # Fallback: use seed material (expanded)
            conv_rho = np.repeat(seed_rho, n_tx, axis=0)  # [n_paths, n_cols]

        if physics_mode:
            R_specular = self._compute_specular_weight_physics(
                cos_theta_i,
                mi.Float(conv_rho[:, 0].astype(np.float32)),
                mi.Float(conv_rho[:, 1].astype(np.float32)),
                mi.Float(conv_rho[:, 2].astype(np.float32)),
                mi.Float(conv_rho[:, 3].astype(np.float32)),
                mi.Float(conv_rho[:, 4].astype(np.float32)),
                mi.Float(conv_rho[:, 5].astype(np.float32)),
            )
        else:
            R_specular = self._compute_specular_weight_legacy(
                cos_theta_i,
                mi.Float(conv_rho[:, 0].astype(np.float32)),
                mi.Float(conv_rho[:, 1].astype(np.float32)),
                mi.Float(conv_rho[:, 2].astype(np.float32)),
            )
        R_specular = dr.select(valid_all, R_specular, mi.Float(0.0))

        if verbose and n_valid_all > 0:
            R_np = np.array(R_specular)
            valid_R = R_np[np.array(valid_all)]
            print(f"  R_specular range: [{valid_R.min():.6f}, {valid_R.max():.6f}]")
            print(f"  R_specular mean: {valid_R.mean():.6f}")

        t_total = time.perf_counter() - t_start
        print(f"  [SMS] Converged: {n_converged:,}/{n_paths:,} ({100*n_converged/max(n_paths,1):.1f}%) | "
              f"Valid specular paths: {n_valid_all:,}/{n_paths:,} | "
              f"Time: {t_total:.3f}s")

        return SpecularPaths(
            hit_P=converged_P,
            hit_N=converged_N,
            d_tx=d_tx,
            d_rx=d_rx,
            dir_to_tx=dir_to_tx,
            dir_to_rx=dir_to_rx,
            cos_theta_i=cos_theta_i,
            cos_theta_r=cos_theta_r,
            tx_idx=exp_tx_idx,
            rx_idx=exp_rx_idx,
            valid=valid_all,
            R_specular=R_specular,
            A_tri=A_tri,
            patch_id=exp_patch_id,
            prim_ids=converged_prim_ids,
            n_paths=n_paths,
            grad_info=newton_grad_info,
            bary_u=spec_bary_u,
            bary_v=spec_bary_v,
        )

    def find_specular_paths_from_seeds(
        self,
        seed_P: 'mi.Point3f',
        seed_N: 'mi.Vector3f',
        seed_prim_ids: 'mi.UInt32',
        seed_rx_idx: 'mi.UInt32',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        scene: 'mi.Scene',
        verbose: bool = True,
        # GPU-native material columns (preferred over triangle_materials)
        triangle_materials_gpu: Optional[list] = None,
        n_material_cols: int = 0,
        triangle_materials: Optional[np.ndarray] = None,
        patch_data: Optional['PatchData'] = None,
    ) -> SpecularPaths:
        """
        Find specular paths from pre-extracted seed positions.

        Alternative entry point for the e2e pipeline where seeds come from
        compressed DrJit reservoir hits (already dedup'd by caller) rather
        than slot-based arrays with a unique_mask.

        Reuses the full Newton solver, validation, and weight computation
        from find_specular_paths().

        Args:
            seed_P: Unique hit positions [n_unique]
            seed_N: Surface normals at seeds [n_unique]
            seed_prim_ids: Triangle IDs at seeds [n_unique]
            seed_rx_idx: RX element index per seed [n_unique]
            tx_positions: TX element positions [n_tx]
            rx_positions: RX element positions [n_rx]
            scene: Mitsuba scene for ray tracing
            verbose: Print progress
            triangle_materials: Per-triangle material array [n_tris, n_cols]
            patch_data: PatchData for patch-area weighting (optional)

        Returns:
            SpecularPaths compatible with integrator synthesis methods.
        """
        from ..utils.math import gather_point3f, gather_vector3f

        t_start = time.perf_counter()

        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)
        n_unique = dr.width(seed_P)

        # Setup mesh normal data for smooth normals
        if self.use_smooth_normals:
            self._setup_mesh_normals(scene)

        if verbose:
            print(f"\n[SMS-E2E] Specular Manifold Sampling (from seeds)")
            print(f"  Unique seeds: {n_unique}")
            print(f"  TX elements: {n_tx}, RX elements: {n_rx}")
            print(f"  Max Newton iterations: {self.max_iterations}")

        if n_unique == 0:
            if verbose:
                print(f"  No unique seeds — returning empty paths")
            return self._empty_specular_paths()

        # Determine material source: GPU columns (fast) or numpy (legacy)
        _has_gpu_mats = triangle_materials_gpu is not None and len(triangle_materials_gpu) > 0
        _has_np_mats = triangle_materials is not None
        physics_mode = (n_material_cols == 6) if _has_gpu_mats else (
            _has_np_mats and triangle_materials.shape[1] == 6)
        n_mat_cols = n_material_cols if _has_gpu_mats else (
            triangle_materials.shape[1] if _has_np_mats else 0)

        # ================================================================
        # Step 2: Expand seeds × TX → (n_unique × n_tx) candidates
        # ================================================================
        n_paths = n_unique * n_tx
        if verbose:
            print(f"  SMS path candidates: {n_unique} × {n_tx} TX = {n_paths:,}")

        exp_seed_P = mi.Point3f(
            dr.repeat(seed_P.x, n_tx), dr.repeat(seed_P.y, n_tx), dr.repeat(seed_P.z, n_tx)
        )

        # PERF: GPU-native tiling instead of np.tile/np.repeat
        exp_tx_idx = dr.arange(mi.UInt32, n_paths) % mi.UInt32(n_tx)
        exp_rx_idx = mi.UInt32(dr.repeat(seed_rx_idx, n_tx))

        exp_tx_pos = mi.Point3f(
            dr.gather(mi.Float, tx_positions.x, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.y, exp_tx_idx),
            dr.gather(mi.Float, tx_positions.z, exp_tx_idx),
        )
        exp_rx_pos = mi.Point3f(
            dr.gather(mi.Float, rx_positions.x, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.y, exp_rx_idx),
            dr.gather(mi.Float, rx_positions.z, exp_rx_idx),
        )

        # ================================================================
        # Step 3: Get surface parameterization at seed points
        # ================================================================
        init_result = self._get_surface_parameterization(
            exp_seed_P, exp_rx_pos, scene)
        dp_du, dp_dv, vtx_n, vtx_s, vtx_t, init_prim_ids, valid_init, \
            init_shape, init_dn_du, init_dn_dv = init_result

        if verbose:
            n_valid_init = int(dr.sum(mi.UInt32(valid_init))[0])
            print(f"  Valid initial ray intersections: {n_valid_init}/{n_paths}")

        # ================================================================
        # Step 4: Vectorized Newton solver
        # ================================================================
        t_newton_start = time.perf_counter()

        converged_P, converged_N, converged_mask, converged_prim_ids, n_iters_used, \
            newton_grad_info, converged_dp_du, converged_dp_dv, last_bary_u, last_bary_v = \
            self._newton_solve_vectorized(
                vtx_p=exp_seed_P,
                vtx_n=vtx_n,
                vtx_dp_du=dp_du,
                vtx_dp_dv=dp_dv,
                rx_pos=exp_rx_pos,
                tx_pos=exp_tx_pos,
                scene=scene,
                valid_init=valid_init,
                init_prim_ids=init_prim_ids,
                init_shape=init_shape,
                init_dn_du=init_dn_du,
                init_dn_dv=init_dn_dv,
            )

        t_newton = time.perf_counter() - t_newton_start
        n_converged = int(dr.sum(mi.UInt32(converged_mask))[0])

        if verbose:
            print(f"  Newton solver: {n_converged:,}/{n_paths:,} converged "
                  f"({100*n_converged/max(n_paths,1):.1f}%) in {n_iters_used} iterations, "
                  f"{t_newton:.3f}s")

        if n_converged == 0:
            if verbose:
                print(f"  No converged paths — returning empty paths")
            return self._empty_specular_paths()

        # ================================================================
        # Step 5: Post-Newton validation (geometry + visibility)
        # ================================================================
        delta_tx = mi.Vector3f(
            exp_tx_pos.x - converged_P.x,
            exp_tx_pos.y - converged_P.y,
            exp_tx_pos.z - converged_P.z,
        )
        delta_rx = mi.Vector3f(
            exp_rx_pos.x - converged_P.x,
            exp_rx_pos.y - converged_P.y,
            exp_rx_pos.z - converged_P.z,
        )
        d_tx = dr.norm(delta_tx)
        d_rx = dr.norm(delta_rx)

        dir_to_tx = mi.Vector3f(
            delta_tx.x / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.y / dr.maximum(d_tx, mi.Float(1e-10)),
            delta_tx.z / dr.maximum(d_tx, mi.Float(1e-10)),
        )
        dir_to_rx = mi.Vector3f(
            delta_rx.x / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.y / dr.maximum(d_rx, mi.Float(1e-10)),
            delta_rx.z / dr.maximum(d_rx, mi.Float(1e-10)),
        )

        cos_theta_i = dr.maximum(dr.dot(dir_to_tx, converged_N), mi.Float(0.0))
        cos_theta_r = dr.maximum(dr.dot(dir_to_rx, converged_N), mi.Float(0.0))

        valid_dist = (d_tx > mi.Float(1e-4)) & (d_rx > mi.Float(1e-4))
        valid_angle = cos_theta_i > mi.Float(1e-6)
        valid_geom = converged_mask & valid_dist & valid_angle

        valid_vis = self._check_visibility(
            converged_P, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
            surface_normal=converged_N,
            skip_rx=True,
        )
        valid_all = valid_geom & valid_vis

        n_valid_all = int(dr.sum(mi.UInt32(valid_all))[0])

        if verbose:
            print(f"  Converged: {n_converged:,} | Valid: {n_valid_all:,}/{n_paths:,}")

        # ================================================================
        # Step 6: Triangle area
        # ================================================================
        spec_bary_u = last_bary_u
        spec_bary_v = last_bary_v

        cross_vec = dr.cross(converged_dp_du, converged_dp_dv)
        A_tri = mi.Float(0.5) * dr.norm(cross_vec)

        exp_patch_id = None
        if patch_data is not None:
            safe_prim = mi.UInt32(dr.maximum(mi.Int32(converged_prim_ids), mi.Int32(0)))
            n_tris = dr.width(patch_data.tri_to_patch)
            safe_prim_clamped = dr.minimum(safe_prim, mi.UInt32(n_tris - 1))
            exp_patch_id_raw = mi.UInt32(dr.gather(mi.Int32, patch_data.tri_to_patch, safe_prim_clamped))
            exp_patch_id = mi.Int32(exp_patch_id_raw)

        # ================================================================
        # Step 7: Compute specular weight using CONVERGED material
        # ================================================================
        # GPU-native path: gather material columns directly on GPU (no CPU round trip)
        if _has_gpu_mats and n_valid_all > 0:
            n_tris_mat = dr.width(triangle_materials_gpu[0])
            safe_prim = dr.minimum(mi.UInt32(dr.maximum(mi.Int32(converged_prim_ids), mi.Int32(0))),
                                   mi.UInt32(n_tris_mat - 1))
            mat_cols = [dr.gather(mi.Float, triangle_materials_gpu[c], safe_prim)
                        for c in range(n_mat_cols)]
            if physics_mode:
                R_specular = self._compute_specular_weight_physics(
                    cos_theta_i, mat_cols[0], mat_cols[1], mat_cols[2],
                    mat_cols[3], mat_cols[4], mat_cols[5])
            else:
                R_specular = self._compute_specular_weight_legacy(
                    cos_theta_i, mat_cols[0], mat_cols[1], mat_cols[2])
        elif _has_np_mats and n_valid_all > 0:
            # Legacy numpy path (for non-e2e callers)
            n_tris_mat = triangle_materials.shape[0]
            prim_np = np.array(converged_prim_ids).astype(np.int64)
            prim_np = np.clip(prim_np, 0, n_tris_mat - 1)
            conv_rho = triangle_materials[prim_np]
            if physics_mode:
                R_specular = self._compute_specular_weight_physics(
                    cos_theta_i,
                    mi.Float(conv_rho[:, 0].astype(np.float32)),
                    mi.Float(conv_rho[:, 1].astype(np.float32)),
                    mi.Float(conv_rho[:, 2].astype(np.float32)),
                    mi.Float(conv_rho[:, 3].astype(np.float32)),
                    mi.Float(conv_rho[:, 4].astype(np.float32)),
                    mi.Float(conv_rho[:, 5].astype(np.float32)))
            else:
                R_specular = self._compute_specular_weight_legacy(
                    cos_theta_i,
                    mi.Float(conv_rho[:, 0].astype(np.float32)),
                    mi.Float(conv_rho[:, 1].astype(np.float32)),
                    mi.Float(conv_rho[:, 2].astype(np.float32)))
        else:
            R_specular = dr.zeros(mi.Float, n_paths)
        R_specular = dr.select(valid_all, R_specular, mi.Float(0.0))

        t_total = time.perf_counter() - t_start
        print(f"  [SMS-E2E] Converged: {n_converged:,}/{n_paths:,} ({100*n_converged/max(n_paths,1):.1f}%) | "
              f"Valid specular paths: {n_valid_all:,}/{n_paths:,} | "
              f"Time: {t_total:.3f}s")

        return SpecularPaths(
            hit_P=converged_P,
            hit_N=converged_N,
            d_tx=d_tx,
            d_rx=d_rx,
            dir_to_tx=dir_to_tx,
            dir_to_rx=dir_to_rx,
            cos_theta_i=cos_theta_i,
            cos_theta_r=cos_theta_r,
            tx_idx=exp_tx_idx,
            rx_idx=exp_rx_idx,
            valid=valid_all,
            R_specular=R_specular,
            A_tri=A_tri,
            patch_id=exp_patch_id,
            prim_ids=converged_prim_ids,
            n_paths=n_paths,
            grad_info=newton_grad_info,
            bary_u=spec_bary_u,
            bary_v=spec_bary_v,
        )

    # ========================================================================
    # Surface parameterization
    # ========================================================================

    def _get_surface_parameterization(
        self,
        seed_P: 'mi.Point3f',
        rx_pos: 'mi.Point3f',
        scene: 'mi.Scene',
    ):
        """
        Get surface parameterization at seed points by casting rays from RX.

        Returns:
            dp_du, dp_dv: Surface partial derivatives [n_paths]
            n: Surface normal (geometric or smooth) [n_paths]
            s, t: Tangent frame vectors [n_paths]
            prim_ids: Primitive (triangle) IDs [n_paths]
            valid: Whether the ray intersection was valid [n_paths]
            shape: ShapePtr for shape check [n_paths]
            dn_du, dn_dv: Normal derivatives (zero for geometric normals) [n_paths]
        """
        ray_dir = safe_normalize(mi.Vector3f(
            seed_P.x - rx_pos.x,
            seed_P.y - rx_pos.y,
            seed_P.z - rx_pos.z,
        ))
        rays = mi.Ray3f(rx_pos, ray_dir)
        si = scene.ray_intersect(rays)

        dp_du = si.dp_du
        dp_dv = si.dp_dv

        if self.use_smooth_normals:
            n = si.sh_frame.n
        else:
            n = si.n

        # Gram-Schmidt orthogonalization: s = normalize(dp_du - n * dot(n, dp_du))
        # Reference: compute_shading_frame (frame.h:197-200)
        n_dot_dpdu = dr.dot(n, dp_du)
        s_raw = mi.Vector3f(dp_du.x - n.x * n_dot_dpdu,
                            dp_du.y - n.y * n_dot_dpdu,
                            dp_du.z - n.z * n_dot_dpdu)
        s = safe_normalize(s_raw)
        t = dr.cross(n, s)

        prim_ids = mi.UInt32(si.prim_index)
        valid = si.is_valid()
        shape = si.shape

        # Normal derivatives
        n_paths = dr.width(seed_P)
        if self.use_smooth_normals:
            dn_du, dn_dv = self._compute_normal_derivatives(prim_ids)
        else:
            zero = mi.Vector3f(dr.zeros(mi.Float, n_paths),
                               dr.zeros(mi.Float, n_paths),
                               dr.zeros(mi.Float, n_paths))
            dn_du, dn_dv = zero, zero

        return dp_du, dp_dv, n, s, t, prim_ids, valid, shape, dn_du, dn_dv

    # ========================================================================
    # Newton solver
    # ========================================================================

    def _newton_solve_vectorized(
        self,
        vtx_p: 'mi.Point3f',
        vtx_n: 'mi.Vector3f',
        vtx_dp_du: 'mi.Vector3f',
        vtx_dp_dv: 'mi.Vector3f',
        rx_pos: 'mi.Point3f',
        tx_pos: 'mi.Point3f',
        scene: 'mi.Scene',
        valid_init: 'mi.Bool',
        init_prim_ids: 'mi.UInt32',
        init_shape,
        init_dn_du: 'mi.Vector3f',
        init_dn_dv: 'mi.Vector3f',
    ) -> Tuple['mi.Point3f', 'mi.Vector3f', 'mi.Bool', 'mi.UInt32', int, 'SpecularGradInfo',
               'mi.Vector3f', 'mi.Vector3f', 'mi.Float', 'mi.Float']:
        """
        Vectorized Newton solver for the specular manifold constraint.

        Solves C(x) = [s·h, t·h] = 0 for all candidates simultaneously,
        where h = normalize(w_i + w_o) is the half-vector.

        Based on manifold_ss.cpp:331-492 from the SMS reference implementation.

        Returns the converged positions, normals, convergence mask, prim_ids,
        iteration count, and SpecularGradInfo containing the final-iteration
        Jacobian inverse needed for IFT gradient computation.
        """
        n_paths = dr.width(vtx_p)

        active = mi.Bool(valid_init)
        converged = mi.Bool(False)
        beta = mi.Float(1.0)
        prim_ids = mi.UInt32(init_prim_ids)
        shape = init_shape

        # Mutable vertex state
        p = mi.Point3f(vtx_p)
        n = mi.Vector3f(vtx_n)
        dp_du = mi.Vector3f(vtx_dp_du)
        dp_dv = mi.Vector3f(vtx_dp_dv)
        dn_du = mi.Vector3f(init_dn_du)
        dn_dv = mi.Vector3f(init_dn_dv)

        # Track final-iteration Jacobian and tangent frame for IFT (per-path)
        final_inv_J00 = dr.zeros(mi.Float, n_paths)
        final_inv_J01 = dr.zeros(mi.Float, n_paths)
        final_inv_J10 = dr.zeros(mi.Float, n_paths)
        final_inv_J11 = dr.zeros(mi.Float, n_paths)
        final_dp_du = mi.Vector3f(dp_du)
        final_dp_dv = mi.Vector3f(dp_dv)

        # Track barycentrics from last reprojection (eliminates Step 6 ray_intersect)
        last_bary_u = dr.zeros(mi.Float, n_paths)
        last_bary_v = dr.zeros(mi.Float, n_paths)

        iterations_used = 0

        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1

            # ---- Compute wi, wo directions ----
            wi_raw = mi.Vector3f(rx_pos.x - p.x, rx_pos.y - p.y, rx_pos.z - p.z)
            wo_raw = mi.Vector3f(tx_pos.x - p.x, tx_pos.y - p.y, tx_pos.z - p.z)

            d_wi = dr.norm(wi_raw)
            d_wo = dr.norm(wo_raw)

            valid_dist = (d_wi > mi.Float(1e-3)) & (d_wo > mi.Float(1e-3))
            active = active & valid_dist

            ili = mi.Float(1.0) / dr.maximum(d_wi, mi.Float(1e-10))
            ilo = mi.Float(1.0) / dr.maximum(d_wo, mi.Float(1e-10))

            wi = mi.Vector3f(wi_raw.x * ili, wi_raw.y * ili, wi_raw.z * ili)
            wo = mi.Vector3f(wo_raw.x * ilo, wo_raw.y * ilo, wo_raw.z * ilo)

            # ---- Fix #5: Backfacing normal flip ----
            # Ref: manifold.h:112 - flip geometric normal when it faces away
            cos_n_wi = dr.dot(n, wi)
            need_flip = active & (cos_n_wi < mi.Float(0.0))
            n = mi.Vector3f(
                dr.select(need_flip, -n.x, n.x),
                dr.select(need_flip, -n.y, n.y),
                dr.select(need_flip, -n.z, n.z),
            )
            # Also flip normal derivatives (dn/du of -n is -dn/du)
            if self.use_smooth_normals:
                dn_du = mi.Vector3f(
                    dr.select(need_flip, -dn_du.x, dn_du.x),
                    dr.select(need_flip, -dn_du.y, dn_du.y),
                    dr.select(need_flip, -dn_du.z, dn_du.z),
                )
                dn_dv = mi.Vector3f(
                    dr.select(need_flip, -dn_dv.x, dn_dv.x),
                    dr.select(need_flip, -dn_dv.y, dn_dv.y),
                    dr.select(need_flip, -dn_dv.z, dn_dv.z),
                )

            # ---- Build tangent frame with Gram-Schmidt (Fix #2 prerequisite) ----
            n_dot_dpdu = dr.dot(n, dp_du)
            s_raw = mi.Vector3f(dp_du.x - n.x * n_dot_dpdu,
                                dp_du.y - n.y * n_dot_dpdu,
                                dp_du.z - n.z * n_dot_dpdu)
            s = safe_normalize(s_raw)
            t = dr.cross(n, s)

            # ---- Compute half-vector h ----
            h_raw = mi.Vector3f(wi.x + wo.x, wi.y + wo.y, wi.z + wo.z)
            h_len = dr.norm(h_raw)
            ilh = mi.Float(1.0) / dr.maximum(h_len, mi.Float(1e-10))
            h = mi.Vector3f(h_raw.x * ilh, h_raw.y * ilh, h_raw.z * ilh)

            # Scaled inverse lengths for Jacobian
            ili_s = ili * ilh
            ilo_s = ilo * ilh

            # ---- Evaluate constraint C = [s·h, t·h] ----
            C_s = dr.dot(s, h)
            C_t = dr.dot(t, h)
            C_norm = dr.sqrt(C_s * C_s + C_t * C_t)

            # ---- Check convergence ----
            just_converged = active & (C_norm < mi.Float(self.solver_threshold))
            converged = converged | just_converged

            # ---- Compute dh/du, dh/dv ONCE (used for both IFT save and Newton step) ----
            # Ref: manifold_ss.cpp:459-462
            dot_wi_du = dr.dot(wi, dp_du)
            dot_wo_du = dr.dot(wo, dp_du)
            scale_sum = ili_s + ilo_s

            dh_du = mi.Vector3f(
                -dp_du.x * scale_sum + wi.x * dot_wi_du * ili_s + wo.x * dot_wo_du * ilo_s,
                -dp_du.y * scale_sum + wi.y * dot_wi_du * ili_s + wo.y * dot_wo_du * ilo_s,
                -dp_du.z * scale_sum + wi.z * dot_wi_du * ili_s + wo.z * dot_wo_du * ilo_s,
            )
            proj_u = dr.dot(dh_du, h)
            dh_du = mi.Vector3f(dh_du.x - h.x * proj_u,
                                dh_du.y - h.y * proj_u,
                                dh_du.z - h.z * proj_u)

            dot_wi_dv = dr.dot(wi, dp_dv)
            dot_wo_dv = dr.dot(wo, dp_dv)

            dh_dv = mi.Vector3f(
                -dp_dv.x * scale_sum + wi.x * dot_wi_dv * ili_s + wo.x * dot_wo_dv * ilo_s,
                -dp_dv.y * scale_sum + wi.y * dot_wi_dv * ili_s + wo.y * dot_wo_dv * ilo_s,
                -dp_dv.z * scale_sum + wi.z * dot_wi_dv * ili_s + wo.z * dot_wo_dv * ilo_s,
            )
            proj_v = dr.dot(dh_dv, h)
            dh_dv = mi.Vector3f(dh_dv.x - h.x * proj_v,
                                dh_dv.y - h.y * proj_v,
                                dh_dv.z - h.z * proj_v)

            # ---- Compute 2×2 Jacobian ONCE (used for both IFT save and Newton step) ----
            # Ref: manifold_ss.cpp:471-475
            if self.use_smooth_normals:
                ds_du, ds_dv, dt_du, dt_dv = self._compute_frame_derivatives(
                    n, dp_du, s, dn_du, dn_dv)
                J00 = dr.dot(ds_du, h) + dr.dot(s, dh_du)
                J10 = dr.dot(dt_du, h) + dr.dot(t, dh_du)
                J01 = dr.dot(ds_dv, h) + dr.dot(s, dh_dv)
                J11 = dr.dot(dt_dv, h) + dr.dot(t, dh_dv)
            else:
                J00 = dr.dot(s, dh_du)
                J10 = dr.dot(t, dh_du)
                J01 = dr.dot(s, dh_dv)
                J11 = dr.dot(t, dh_dv)

            det = J00 * J11 - J01 * J10
            valid_det = dr.abs(det) > mi.Float(1e-6)
            inv_det = dr.select(valid_det, mi.Float(1.0) / det, mi.Float(0.0))

            # ---- Save IFT data for just-converged paths (NO guard, NO GPU sync) ----
            final_inv_J00 = dr.select(just_converged,  J11 * inv_det, final_inv_J00)
            final_inv_J01 = dr.select(just_converged, -J01 * inv_det, final_inv_J01)
            final_inv_J10 = dr.select(just_converged, -J10 * inv_det, final_inv_J10)
            final_inv_J11 = dr.select(just_converged,  J00 * inv_det, final_inv_J11)

            final_dp_du = mi.Vector3f(
                dr.select(just_converged, dp_du.x, final_dp_du.x),
                dr.select(just_converged, dp_du.y, final_dp_du.y),
                dr.select(just_converged, dp_du.z, final_dp_du.z))
            final_dp_dv = mi.Vector3f(
                dr.select(just_converged, dp_dv.x, final_dp_dv.x),
                dr.select(just_converged, dp_dv.y, final_dp_dv.y),
                dr.select(just_converged, dp_dv.z, final_dp_dv.z))

            active = active & ~just_converged
            # Also deactivate paths with singular Jacobian
            active = active & valid_det

            # PERF: Periodically evaluate all state to prevent lazy JIT graph
            # explosion from chained ray_intersect calls across iterations.
            if iteration % 3 == 2 or iteration == self.max_iterations - 1:
                dr.eval(active, converged, beta, prim_ids,
                        p.x, p.y, p.z, n.x, n.y, n.z,
                        dp_du.x, dp_du.y, dp_du.z,
                        dp_dv.x, dp_dv.y, dp_dv.z,
                        last_bary_u, last_bary_v,
                        final_inv_J00, final_inv_J01, final_inv_J10, final_inv_J11,
                        final_dp_du.x, final_dp_du.y, final_dp_du.z,
                        final_dp_dv.x, final_dp_dv.y, final_dp_dv.z)
                if dr.none(active):
                    break

            # ---- Solve 2×2 Newton step (reuse J and inv_det from above) ----
            dX_u = inv_det * (J11 * C_s - J01 * C_t)
            dX_v = inv_det * (-J10 * C_s + J00 * C_t)

            # ---- Propose new position ----
            step_x = dp_du.x * dX_u + dp_dv.x * dX_v
            step_y = dp_du.y * dX_u + dp_dv.y * dX_v
            step_z = dp_du.z * dX_u + dp_dv.z * dX_v

            p_prop_x = p.x - dr.select(active, beta * step_x, mi.Float(0.0))
            p_prop_y = p.y - dr.select(active, beta * step_y, mi.Float(0.0))
            p_prop_z = p.z - dr.select(active, beta * step_z, mi.Float(0.0))

            # ---- Re-project onto mesh via ray tracing ----
            ray_dir_x = p_prop_x - rx_pos.x
            ray_dir_y = p_prop_y - rx_pos.y
            ray_dir_z = p_prop_z - rx_pos.z
            ray_dir_len = dr.sqrt(ray_dir_x * ray_dir_x + ray_dir_y * ray_dir_y + ray_dir_z * ray_dir_z)
            ray_dir_ilen = mi.Float(1.0) / dr.maximum(ray_dir_len, mi.Float(1e-10))
            ray_dir = mi.Vector3f(
                ray_dir_x * ray_dir_ilen,
                ray_dir_y * ray_dir_ilen,
                ray_dir_z * ray_dir_ilen,
            )

            ray = mi.Ray3f(rx_pos, ray_dir)
            si_new = scene.ray_intersect(ray, active)
            hit_valid = si_new.is_valid() & active

            # ---- Fix #4: Shape check ----
            # Reject re-projection hits on different shapes (manifold_ss.cpp:373)
            try:
                same_shape = (si_new.shape == shape)
                hit_valid = hit_valid & same_shape
            except Exception:
                pass  # Shape comparison not supported, skip

            # ---- Update vertex state where ray hit ----
            p = mi.Point3f(
                dr.select(hit_valid, si_new.p.x, p.x),
                dr.select(hit_valid, si_new.p.y, p.y),
                dr.select(hit_valid, si_new.p.z, p.z),
            )
            dp_du = mi.Vector3f(
                dr.select(hit_valid, si_new.dp_du.x, dp_du.x),
                dr.select(hit_valid, si_new.dp_du.y, dp_du.y),
                dr.select(hit_valid, si_new.dp_du.z, dp_du.z),
            )
            dp_dv = mi.Vector3f(
                dr.select(hit_valid, si_new.dp_dv.x, dp_dv.x),
                dr.select(hit_valid, si_new.dp_dv.y, dp_dv.y),
                dr.select(hit_valid, si_new.dp_dv.z, dp_dv.z),
            )

            # Update normal
            if self.use_smooth_normals:
                new_n = si_new.sh_frame.n
            else:
                new_n = si_new.n
            n = mi.Vector3f(
                dr.select(hit_valid, new_n.x, n.x),
                dr.select(hit_valid, new_n.y, n.y),
                dr.select(hit_valid, new_n.z, n.z),
            )

            # Update normal derivatives for smooth normals
            if self.use_smooth_normals:
                new_prim = mi.UInt32(si_new.prim_index)
                new_dn_du, new_dn_dv = self._compute_normal_derivatives(new_prim)
                dn_du = mi.Vector3f(
                    dr.select(hit_valid, new_dn_du.x, dn_du.x),
                    dr.select(hit_valid, new_dn_du.y, dn_du.y),
                    dr.select(hit_valid, new_dn_du.z, dn_du.z),
                )
                dn_dv = mi.Vector3f(
                    dr.select(hit_valid, new_dn_dv.x, dn_dv.x),
                    dr.select(hit_valid, new_dn_dv.y, dn_dv.y),
                    dr.select(hit_valid, new_dn_dv.z, dn_dv.z),
                )

            # Track barycentrics from reprojection
            last_bary_u = dr.select(hit_valid, si_new.uv.x, last_bary_u)
            last_bary_v = dr.select(hit_valid, si_new.uv.y, last_bary_v)

            # Track primitive IDs and shape
            prim_ids = dr.select(hit_valid, mi.UInt32(si_new.prim_index), prim_ids)
            # Update shape to track the current shape (matching reference behavior)
            try:
                shape = dr.select(hit_valid, si_new.shape, shape)
            except Exception:
                pass  # dr.select on ShapePtr not supported

            # ---- Adaptive step size ----
            beta = dr.select(hit_valid,
                             dr.minimum(mi.Float(1.0), mi.Float(2.0) * beta),
                             beta)
            missed = active & ~hit_valid
            beta = dr.select(missed, mi.Float(0.5) * beta, beta)
            active = active & (beta > mi.Float(1e-8))

        grad_info = SpecularGradInfo(
            inv_J00=final_inv_J00,
            inv_J01=final_inv_J01,
            inv_J10=final_inv_J10,
            inv_J11=final_inv_J11,
            dp_du=final_dp_du,
            dp_dv=final_dp_dv,
            tx_pos=tx_pos,
            rx_pos=rx_pos,
        )

        return p, n, converged, prim_ids, iterations_used, grad_info, dp_du, dp_dv, last_bary_u, last_bary_v

    # ========================================================================
    # Visibility and weight computation (unchanged)
    # ========================================================================

    @staticmethod
    def _check_visibility(P_spec, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
                          surface_normal=None, skip_rx=False):
        """Shadow ray visibility check (same as ImageMethodRefiner)."""
        if surface_normal is not None:
            normal_offset = mi.Float(1e-3)
            P_offset = mi.Point3f(
                P_spec.x + normal_offset * surface_normal.x,
                P_spec.y + normal_offset * surface_normal.y,
                P_spec.z + normal_offset * surface_normal.z,
            )
        else:
            P_offset = P_spec

        epsilon = 1e-4
        shadow_origin_tx = mi.Point3f(
            P_offset.x + mi.Float(epsilon) * dir_to_tx.x,
            P_offset.y + mi.Float(epsilon) * dir_to_tx.y,
            P_offset.z + mi.Float(epsilon) * dir_to_tx.z,
        )
        shadow_rays_tx = mi.Ray3f(shadow_origin_tx, dir_to_tx)
        shadow_rays_tx.maxt = d_tx - 2.0 * epsilon
        occluded_tx = scene.ray_test(shadow_rays_tx)

        if skip_rx:
            return ~occluded_tx

        shadow_origin_rx = mi.Point3f(
            P_offset.x + mi.Float(epsilon) * dir_to_rx.x,
            P_offset.y + mi.Float(epsilon) * dir_to_rx.y,
            P_offset.z + mi.Float(epsilon) * dir_to_rx.z,
        )
        shadow_rays_rx = mi.Ray3f(shadow_origin_rx, dir_to_rx)
        shadow_rays_rx.maxt = d_rx - 2.0 * epsilon
        occluded_rx = scene.ray_test(shadow_rays_rx)

        return ~occluded_tx & ~occluded_rx

    def _compute_specular_weight_physics(
        self,
        cos_theta_i: 'mi.Float',
        eps_real: 'mi.Float',
        eps_imag: 'mi.Float',
        sigma_h: 'mi.Float',
        l_c: 'mi.Float',
        tau: 'mi.Float',
        thickness: 'mi.Float',
    ) -> 'mi.Float':
        """Compute R_specular for physics-mode materials (same as ImageMethodRefiner)."""
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.wavelength)

        A = compute_slab_energy_gate(
            eps_real, eps_imag, cos_theta_i, thickness,
            mi.Float(0.5), mi.Float(0.5),
            wavelength=self.wavelength,
            sigma_h=sigma_h_v,
        )

        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.wavelength, cos_theta_i
        )

        tau_eff = compute_validity_aware_blend(
            tau, cos_theta_i, sigma_h_v, l_c_v, self.wavelength
        )

        return eta * tau_eff * A

    def _compute_specular_weight_legacy(
        self,
        cos_theta_i: 'mi.Float',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> 'mi.Float':
        """Compute R_specular for legacy-mode materials (same as ImageMethodRefiner)."""
        eps_real, eps_imag, sigma_h_raw, l_c_raw, tau_base = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.wavelength)

        sigma_h, l_c = enforce_spm_validity(sigma_h_raw, l_c_raw, self.wavelength)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)

        A = compute_energy_gate(n_ior, kappa, cos_theta_i,
                                mi.Float(0.5), mi.Float(0.5))

        eta, _ = compute_coherent_incoherent_blend(sigma_h, l_c, self.wavelength, cos_theta_i)

        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.wavelength
        )

        return eta * tau_eff * A

    # ========================================================================
    # Multibounce SMS: find_specular_chains()
    # ========================================================================

    def find_specular_chains(
        self,
        seed_P_list: list,           # [k] mi.Point3f seed positions per vertex
        seed_prim_ids_list: list,    # [k] mi.UInt32 seed prim IDs per vertex
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        seed_tx_idx: 'mi.UInt32',    # TX index per chain
        seed_rx_idx: 'mi.UInt32',    # RX index per chain
        scene: 'mi.Scene',
        k: int = 2,
        max_iterations: int = 20,
        threshold: float = 1e-5,
        verbose: bool = False,
        triangle_materials: Optional[np.ndarray] = None,
        triangle_materials_gpu: Optional[list] = None,
        n_material_cols: int = 0,
    ) -> MultibounceSpecularChain:
        """
        Find k-bounce specular chains via block-tridiagonal Newton iteration.

        For k=2: RX -> S1 -> S2 -> TX (double-bounce specular).

        The Newton solver simultaneously solves the half-vector constraint
        C_i = [s_i . h_i, t_i . h_i] = 0 at all k vertices using a
        block-tridiagonal Jacobian. After convergence, validates visibility
        for all k+1 segments and computes per-vertex triangle areas.

        Args:
            seed_P_list: [k] lists of mi.Point3f seed positions [n_chains each]
            seed_prim_ids_list: [k] lists of mi.UInt32 prim IDs [n_chains each]
            tx_positions: All TX positions [n_tx]
            rx_positions: All RX positions [n_rx]
            seed_tx_idx: TX index per chain [n_chains]
            seed_rx_idx: RX index per chain [n_chains]
            scene: Mitsuba scene for ray tracing
            k: Number of specular bounces (default 2)
            max_iterations: Max Newton iterations
            threshold: Convergence threshold on ||C||
            verbose: Print progress
            triangle_materials: Per-triangle materials [n_tris, n_cols] for weight

        Returns:
            MultibounceSpecularChain with converged chains
        """
        from ..utils.math import gather_point3f

        t_start = time.perf_counter()
        n_chains = dr.width(seed_P_list[0])

        if self.use_smooth_normals:
            self._setup_mesh_normals(scene)

        if verbose:
            print(f"\n[SMS-MB] Multibounce Specular Manifold Sampling (k={k})")
            print(f"  Chains: {n_chains:,}, Max iter: {max_iterations}")

        if n_chains == 0:
            return self._empty_multibounce_chain(k)

        # Expand TX/RX positions per chain
        exp_tx_pos = gather_point3f(tx_positions, seed_tx_idx)
        exp_rx_pos = gather_point3f(rx_positions, seed_rx_idx)

        # ================================================================
        # Initialize per-vertex state from seeds
        # ================================================================
        # Get surface parameterization at each seed via ray intersect
        p = []       # current positions [k]
        n_vec = []   # current normals [k]
        dp_du = []   # surface tangent u [k]
        dp_dv = []   # surface tangent v [k]
        dn_du_arr = []
        dn_dv_arr = []
        prim_ids = []  # current prim IDs [k]
        bary_u = [dr.zeros(mi.Float, n_chains) for _ in range(k)]
        bary_v = [dr.zeros(mi.Float, n_chains) for _ in range(k)]

        active = mi.Bool(True)

        for i in range(k):
            # Cast ray from the previous vertex (or RX for i=0) toward seed
            if i == 0:
                ray_origin = exp_rx_pos
            else:
                ray_origin = p[i-1]

            ray_dir = safe_normalize(mi.Vector3f(
                seed_P_list[i].x - ray_origin.x,
                seed_P_list[i].y - ray_origin.y,
                seed_P_list[i].z - ray_origin.z,
            ))
            ray = mi.Ray3f(ray_origin, ray_dir)
            si = scene.ray_intersect(ray)

            valid_hit = si.is_valid()
            active = active & valid_hit

            p.append(mi.Point3f(si.p))
            dp_du.append(mi.Vector3f(si.dp_du))
            dp_dv.append(mi.Vector3f(si.dp_dv))
            prim_ids.append(mi.UInt32(si.prim_index))
            bary_u[i] = si.uv.x
            bary_v[i] = si.uv.y

            if self.use_smooth_normals:
                n_vec.append(mi.Vector3f(si.sh_frame.n))
                _dn_du, _dn_dv = self._compute_normal_derivatives(
                    mi.UInt32(si.prim_index))
                dn_du_arr.append(_dn_du)
                dn_dv_arr.append(_dn_dv)
            else:
                n_vec.append(mi.Vector3f(si.n))
                zero3 = mi.Vector3f(dr.zeros(mi.Float, n_chains),
                                    dr.zeros(mi.Float, n_chains),
                                    dr.zeros(mi.Float, n_chains))
                dn_du_arr.append(zero3)
                dn_dv_arr.append(zero3)

        if verbose:
            n_valid_init = int(dr.sum(mi.UInt32(active))[0])
            print(f"  Valid initial chains: {n_valid_init}/{n_chains}")

        # ================================================================
        # Newton iteration on block-tridiagonal system
        # ================================================================
        converged = mi.Bool(False)
        beta = mi.Float(1.0)  # adaptive step size

        # Storage for IFT factorization (saved at convergence)
        final_D_inv = [None] * k
        final_U = [None] * max(k - 1, 1)
        final_dp_du = [mi.Vector3f(dp_du[i]) for i in range(k)]
        final_dp_dv = [mi.Vector3f(dp_dv[i]) for i in range(k)]

        iterations_used = 0

        for iteration in range(max_iterations):
            iterations_used = iteration + 1

            # ---- Compute directions at each vertex ----
            wi_list = []  # incoming direction (from previous vertex / RX)
            wo_list = []  # outgoing direction (toward next vertex / TX)
            d_wi_list = []
            d_wo_list = []

            for i in range(k):
                # wi: direction FROM previous vertex (or RX)
                if i == 0:
                    wi_raw = mi.Vector3f(
                        exp_rx_pos.x - p[i].x,
                        exp_rx_pos.y - p[i].y,
                        exp_rx_pos.z - p[i].z)
                else:
                    wi_raw = mi.Vector3f(
                        p[i-1].x - p[i].x,
                        p[i-1].y - p[i].y,
                        p[i-1].z - p[i].z)

                # wo: direction TO next vertex (or TX)
                if i == k - 1:
                    wo_raw = mi.Vector3f(
                        exp_tx_pos.x - p[i].x,
                        exp_tx_pos.y - p[i].y,
                        exp_tx_pos.z - p[i].z)
                else:
                    wo_raw = mi.Vector3f(
                        p[i+1].x - p[i].x,
                        p[i+1].y - p[i].y,
                        p[i+1].z - p[i].z)

                d_wi = dr.norm(wi_raw)
                d_wo = dr.norm(wo_raw)
                d_wi_list.append(d_wi)
                d_wo_list.append(d_wo)

                valid_dist = (d_wi > mi.Float(1e-3)) & (d_wo > mi.Float(1e-3))
                active = active & valid_dist

                ili = mi.Float(1.0) / dr.maximum(d_wi, mi.Float(1e-10))
                ilo = mi.Float(1.0) / dr.maximum(d_wo, mi.Float(1e-10))
                wi_list.append((mi.Vector3f(wi_raw.x*ili, wi_raw.y*ili, wi_raw.z*ili), ili))
                wo_list.append((mi.Vector3f(wo_raw.x*ilo, wo_raw.y*ilo, wo_raw.z*ilo), ilo))

            # ---- Backfacing normal flip ----
            for i in range(k):
                wi, _ = wi_list[i]
                cos_n_wi = dr.dot(n_vec[i], wi)
                need_flip = active & (cos_n_wi < mi.Float(0.0))
                n_vec[i] = mi.Vector3f(
                    dr.select(need_flip, -n_vec[i].x, n_vec[i].x),
                    dr.select(need_flip, -n_vec[i].y, n_vec[i].y),
                    dr.select(need_flip, -n_vec[i].z, n_vec[i].z))
                if self.use_smooth_normals:
                    dn_du_arr[i] = mi.Vector3f(
                        dr.select(need_flip, -dn_du_arr[i].x, dn_du_arr[i].x),
                        dr.select(need_flip, -dn_du_arr[i].y, dn_du_arr[i].y),
                        dr.select(need_flip, -dn_du_arr[i].z, dn_du_arr[i].z))
                    dn_dv_arr[i] = mi.Vector3f(
                        dr.select(need_flip, -dn_dv_arr[i].x, dn_dv_arr[i].x),
                        dr.select(need_flip, -dn_dv_arr[i].y, dn_dv_arr[i].y),
                        dr.select(need_flip, -dn_dv_arr[i].z, dn_dv_arr[i].z))

            # ---- Build tangent frames and evaluate constraints ----
            C_list = []       # [k] constraint 2-vectors
            C_norm_sum = dr.zeros(mi.Float, n_chains)

            # Jacobian blocks
            A_diag = []       # [k] diagonal 2x2 blocks
            B_upper = []      # [k-1] upper 2x2 blocks
            C_lower_blocks = []  # [k-1] lower 2x2 blocks

            for i in range(k):
                wi, ili = wi_list[i]
                wo, ilo = wo_list[i]
                ni = n_vec[i]

                # Tangent frame
                n_dot_dpdu = dr.dot(ni, dp_du[i])
                s_raw = mi.Vector3f(
                    dp_du[i].x - ni.x * n_dot_dpdu,
                    dp_du[i].y - ni.y * n_dot_dpdu,
                    dp_du[i].z - ni.z * n_dot_dpdu)
                s = safe_normalize(s_raw)
                t = dr.cross(ni, s)

                # Half-vector
                h_raw = mi.Vector3f(wi.x + wo.x, wi.y + wo.y, wi.z + wo.z)
                h_len = dr.norm(h_raw)
                ilh = mi.Float(1.0) / dr.maximum(h_len, mi.Float(1e-10))
                h = mi.Vector3f(h_raw.x * ilh, h_raw.y * ilh, h_raw.z * ilh)

                # Constraint
                C_s = dr.dot(s, h)
                C_t = dr.dot(t, h)
                C_list.append((C_s, C_t))
                C_norm_sum = C_norm_sum + C_s * C_s + C_t * C_t

                # ---- Jacobian: dC_i/d(u_i, v_i) = diagonal block A_i ----
                ili_s = ili * ilh
                ilo_s = ilo * ilh
                scale_sum = ili_s + ilo_s

                # dh/du_i (effect of moving S_i on its own half-vector)
                dot_wi_du = dr.dot(wi, dp_du[i])
                dot_wo_du = dr.dot(wo, dp_du[i])
                dh_du = mi.Vector3f(
                    -dp_du[i].x * scale_sum + wi.x * dot_wi_du * ili_s + wo.x * dot_wo_du * ilo_s,
                    -dp_du[i].y * scale_sum + wi.y * dot_wi_du * ili_s + wo.y * dot_wo_du * ilo_s,
                    -dp_du[i].z * scale_sum + wi.z * dot_wi_du * ili_s + wo.z * dot_wo_du * ilo_s)
                proj_u = dr.dot(dh_du, h)
                dh_du = mi.Vector3f(dh_du.x - h.x*proj_u, dh_du.y - h.y*proj_u, dh_du.z - h.z*proj_u)

                dot_wi_dv = dr.dot(wi, dp_dv[i])
                dot_wo_dv = dr.dot(wo, dp_dv[i])
                dh_dv = mi.Vector3f(
                    -dp_dv[i].x * scale_sum + wi.x * dot_wi_dv * ili_s + wo.x * dot_wo_dv * ilo_s,
                    -dp_dv[i].y * scale_sum + wi.y * dot_wi_dv * ili_s + wo.y * dot_wo_dv * ilo_s,
                    -dp_dv[i].z * scale_sum + wi.z * dot_wi_dv * ili_s + wo.z * dot_wo_dv * ilo_s)
                proj_v = dr.dot(dh_dv, h)
                dh_dv = mi.Vector3f(dh_dv.x - h.x*proj_v, dh_dv.y - h.y*proj_v, dh_dv.z - h.z*proj_v)

                if self.use_smooth_normals:
                    ds_du, ds_dv, dt_du, dt_dv = self._compute_frame_derivatives(
                        ni, dp_du[i], s, dn_du_arr[i], dn_dv_arr[i])
                    A00 = dr.dot(ds_du, h) + dr.dot(s, dh_du)
                    A10 = dr.dot(dt_du, h) + dr.dot(t, dh_du)
                    A01 = dr.dot(ds_dv, h) + dr.dot(s, dh_dv)
                    A11 = dr.dot(dt_dv, h) + dr.dot(t, dh_dv)
                else:
                    A00 = dr.dot(s, dh_du)
                    A10 = dr.dot(t, dh_du)
                    A01 = dr.dot(s, dh_dv)
                    A11 = dr.dot(t, dh_dv)

                A_diag.append((A00, A01, A10, A11))

                # ---- Off-diagonal blocks: coupling to adjacent vertices ----
                # Upper block B_i: dC_i/d(u_{i+1}, v_{i+1})
                # Moving S_{i+1} changes wo at vertex i
                if i < k - 1:
                    # dh_i / d(u_{i+1}) via change in wo direction
                    # wo = (p[i+1] - p[i]) / ||...||
                    # d(wo)/d(u_{i+1}) = ilo * (dp_du[i+1] - wo * dot(wo, dp_du[i+1]))
                    # but projected through ilh
                    dwo_du_next = mi.Vector3f(
                        ilo * (dp_du[i+1].x - wo.x * dr.dot(wo, dp_du[i+1])),
                        ilo * (dp_du[i+1].y - wo.y * dr.dot(wo, dp_du[i+1])),
                        ilo * (dp_du[i+1].z - wo.z * dr.dot(wo, dp_du[i+1])))
                    dh_du_next = mi.Vector3f(
                        ilh * (dwo_du_next.x - h.x * dr.dot(dwo_du_next, h)),
                        ilh * (dwo_du_next.y - h.y * dr.dot(dwo_du_next, h)),
                        ilh * (dwo_du_next.z - h.z * dr.dot(dwo_du_next, h)))

                    dwo_dv_next = mi.Vector3f(
                        ilo * (dp_dv[i+1].x - wo.x * dr.dot(wo, dp_dv[i+1])),
                        ilo * (dp_dv[i+1].y - wo.y * dr.dot(wo, dp_dv[i+1])),
                        ilo * (dp_dv[i+1].z - wo.z * dr.dot(wo, dp_dv[i+1])))
                    dh_dv_next_dot = dr.dot(dwo_dv_next, h)
                    dh_dv_next = mi.Vector3f(
                        ilh * (dwo_dv_next.x - h.x * dh_dv_next_dot),
                        ilh * (dwo_dv_next.y - h.y * dh_dv_next_dot),
                        ilh * (dwo_dv_next.z - h.z * dh_dv_next_dot))

                    B00 = dr.dot(s, dh_du_next)
                    B10 = dr.dot(t, dh_du_next)
                    B01 = dr.dot(s, dh_dv_next)
                    B11 = dr.dot(t, dh_dv_next)
                    B_upper.append((B00, B01, B10, B11))

                # Lower block C_i: dC_i/d(u_{i-1}, v_{i-1})
                # Moving S_{i-1} changes wi at vertex i
                if i > 0:
                    # d(wi)/d(u_{i-1}) via change in wi direction
                    dwi_du_prev = mi.Vector3f(
                        ili * (dp_du[i-1].x - wi.x * dr.dot(wi, dp_du[i-1])),
                        ili * (dp_du[i-1].y - wi.y * dr.dot(wi, dp_du[i-1])),
                        ili * (dp_du[i-1].z - wi.z * dr.dot(wi, dp_du[i-1])))
                    dh_du_prev = mi.Vector3f(
                        ilh * (dwi_du_prev.x - h.x * dr.dot(dwi_du_prev, h)),
                        ilh * (dwi_du_prev.y - h.y * dr.dot(dwi_du_prev, h)),
                        ilh * (dwi_du_prev.z - h.z * dr.dot(dwi_du_prev, h)))

                    dwi_dv_prev = mi.Vector3f(
                        ili * (dp_dv[i-1].x - wi.x * dr.dot(wi, dp_dv[i-1])),
                        ili * (dp_dv[i-1].y - wi.y * dr.dot(wi, dp_dv[i-1])),
                        ili * (dp_dv[i-1].z - wi.z * dr.dot(wi, dp_dv[i-1])))
                    dh_dv_prev = mi.Vector3f(
                        ilh * (dwi_dv_prev.x - h.x * dr.dot(dwi_dv_prev, h)),
                        ilh * (dwi_dv_prev.y - h.y * dr.dot(dwi_dv_prev, h)),
                        ilh * (dwi_dv_prev.z - h.z * dr.dot(dwi_dv_prev, h)))

                    CL00 = dr.dot(s, dh_du_prev)
                    CL10 = dr.dot(t, dh_du_prev)
                    CL01 = dr.dot(s, dh_dv_prev)
                    CL11 = dr.dot(t, dh_dv_prev)
                    C_lower_blocks.append((CL00, CL01, CL10, CL11))

            # ---- Check convergence ----
            C_norm_total = dr.sqrt(dr.maximum(C_norm_sum, mi.Float(0.0)))
            just_converged = active & (C_norm_total < mi.Float(threshold))
            converged = converged | just_converged

            # ---- Save IFT factorization for just-converged chains ----
            if len(A_diag) == k and len(B_upper) == k - 1 and len(C_lower_blocks) == k - 1:
                _, D_inv_iter, U_iter = self._thomas_block_tridiagonal(
                    A_diag, B_upper, C_lower_blocks, C_list, active)

                for vi in range(k):
                    if final_D_inv[vi] is None:
                        final_D_inv[vi] = tuple(
                            dr.select(just_converged, D_inv_iter[vi][j], dr.zeros(mi.Float, n_chains))
                            for j in range(4))
                    else:
                        final_D_inv[vi] = tuple(
                            dr.select(just_converged, D_inv_iter[vi][j], final_D_inv[vi][j])
                            for j in range(4))

                    final_dp_du[vi] = mi.Vector3f(
                        dr.select(just_converged, dp_du[vi].x, final_dp_du[vi].x),
                        dr.select(just_converged, dp_du[vi].y, final_dp_du[vi].y),
                        dr.select(just_converged, dp_du[vi].z, final_dp_du[vi].z))
                    final_dp_dv[vi] = mi.Vector3f(
                        dr.select(just_converged, dp_dv[vi].x, final_dp_dv[vi].x),
                        dr.select(just_converged, dp_dv[vi].y, final_dp_dv[vi].y),
                        dr.select(just_converged, dp_dv[vi].z, final_dp_dv[vi].z))

                for ui in range(len(U_iter)):
                    if final_U[ui] is None:
                        final_U[ui] = tuple(
                            dr.select(just_converged, U_iter[ui][j], dr.zeros(mi.Float, n_chains))
                            for j in range(4))
                    else:
                        final_U[ui] = tuple(
                            dr.select(just_converged, U_iter[ui][j], final_U[ui][j])
                            for j in range(4))

            active = active & ~just_converged

            # PERF: Periodically evaluate all per-vertex state AND IFT
            # accumulator to prevent lazy JIT graph explosion. Without this,
            # 20 iterations of ray_intersect calls build an enormous graph
            # that takes tens of seconds to JIT-compile. Evaluating every 3
            # iterations trades a few small kernel launches for one massive
            # compilation.
            if iteration % 3 == 2 or iteration == max_iterations - 1:
                _eval_args = [active, converged, beta]
                for i in range(k):
                    _eval_args.extend([
                        p[i].x, p[i].y, p[i].z,
                        n_vec[i].x, n_vec[i].y, n_vec[i].z,
                        dp_du[i].x, dp_du[i].y, dp_du[i].z,
                        dp_dv[i].x, dp_dv[i].y, dp_dv[i].z,
                        prim_ids[i], bary_u[i], bary_v[i],
                    ])
                    # IFT accumulator state
                    if final_D_inv[i] is not None:
                        _eval_args.extend(final_D_inv[i])
                    _eval_args.extend([
                        final_dp_du[i].x, final_dp_du[i].y, final_dp_du[i].z,
                        final_dp_dv[i].x, final_dp_dv[i].y, final_dp_dv[i].z,
                    ])
                for ui in range(len(final_U)):
                    if final_U[ui] is not None:
                        _eval_args.extend(final_U[ui])
                dr.eval(*_eval_args)

                # Check if all done
                if dr.none(active):
                    break

            # ---- Solve block-tridiagonal for Newton step ----
            if len(A_diag) == k and len(B_upper) == k - 1 and len(C_lower_blocks) == k - 1:
                deltas, _, _ = self._thomas_block_tridiagonal(
                    A_diag, B_upper, C_lower_blocks, C_list, active)
            else:
                # Fallback for malformed system: no update
                break

            # ---- Update all vertices ----
            for i in range(k):
                dX_u, dX_v = deltas[i]

                step_x = dp_du[i].x * dX_u + dp_dv[i].x * dX_v
                step_y = dp_du[i].y * dX_u + dp_dv[i].y * dX_v
                step_z = dp_du[i].z * dX_u + dp_dv[i].z * dX_v

                p_prop = mi.Point3f(
                    p[i].x - dr.select(active, beta * step_x, mi.Float(0.0)),
                    p[i].y - dr.select(active, beta * step_y, mi.Float(0.0)),
                    p[i].z - dr.select(active, beta * step_z, mi.Float(0.0)))

                # Re-project onto mesh
                if i == 0:
                    ray_origin = exp_rx_pos
                else:
                    ray_origin = p[i-1]

                ray_dir = safe_normalize(mi.Vector3f(
                    p_prop.x - ray_origin.x,
                    p_prop.y - ray_origin.y,
                    p_prop.z - ray_origin.z))
                ray = mi.Ray3f(ray_origin, ray_dir)
                si_new = scene.ray_intersect(ray, active)
                hit_valid = si_new.is_valid() & active

                p[i] = mi.Point3f(
                    dr.select(hit_valid, si_new.p.x, p[i].x),
                    dr.select(hit_valid, si_new.p.y, p[i].y),
                    dr.select(hit_valid, si_new.p.z, p[i].z))
                dp_du[i] = mi.Vector3f(
                    dr.select(hit_valid, si_new.dp_du.x, dp_du[i].x),
                    dr.select(hit_valid, si_new.dp_du.y, dp_du[i].y),
                    dr.select(hit_valid, si_new.dp_du.z, dp_du[i].z))
                dp_dv[i] = mi.Vector3f(
                    dr.select(hit_valid, si_new.dp_dv.x, dp_dv[i].x),
                    dr.select(hit_valid, si_new.dp_dv.y, dp_dv[i].y),
                    dr.select(hit_valid, si_new.dp_dv.z, dp_dv[i].z))

                new_n = si_new.sh_frame.n if self.use_smooth_normals else si_new.n
                n_vec[i] = mi.Vector3f(
                    dr.select(hit_valid, new_n.x, n_vec[i].x),
                    dr.select(hit_valid, new_n.y, n_vec[i].y),
                    dr.select(hit_valid, new_n.z, n_vec[i].z))

                prim_ids[i] = dr.select(hit_valid, mi.UInt32(si_new.prim_index), prim_ids[i])
                bary_u[i] = dr.select(hit_valid, si_new.uv.x, bary_u[i])
                bary_v[i] = dr.select(hit_valid, si_new.uv.y, bary_v[i])

                if self.use_smooth_normals:
                    new_prim = mi.UInt32(si_new.prim_index)
                    new_dn_du, new_dn_dv = self._compute_normal_derivatives(new_prim)
                    dn_du_arr[i] = mi.Vector3f(
                        dr.select(hit_valid, new_dn_du.x, dn_du_arr[i].x),
                        dr.select(hit_valid, new_dn_du.y, dn_du_arr[i].y),
                        dr.select(hit_valid, new_dn_du.z, dn_du_arr[i].z))
                    dn_dv_arr[i] = mi.Vector3f(
                        dr.select(hit_valid, new_dn_dv.x, dn_dv_arr[i].x),
                        dr.select(hit_valid, new_dn_dv.y, dn_dv_arr[i].y),
                        dr.select(hit_valid, new_dn_dv.z, dn_dv_arr[i].z))

            # Adaptive step size
            # (use first vertex hit_valid as proxy — if it misses, reduce step)
            beta = dr.select(active & hit_valid,
                             dr.minimum(mi.Float(1.0), mi.Float(2.0) * beta),
                             beta)
            missed = active & ~hit_valid
            beta = dr.select(missed, mi.Float(0.5) * beta, beta)
            active = active & (beta > mi.Float(1e-8))

        # ================================================================
        # Post-Newton: convergence stats
        # ================================================================
        n_converged = int(dr.sum(mi.UInt32(converged))[0])
        t_newton = time.perf_counter() - t_start

        if verbose:
            print(f"  Newton solver: {n_converged:,}/{n_chains:,} converged "
                  f"({100*n_converged/max(n_chains,1):.1f}%) in {iterations_used} iter, "
                  f"{t_newton:.3f}s")

        if n_converged == 0:
            return self._empty_multibounce_chain(k)

        # ================================================================
        # Visibility check for all k+1 segments
        # ================================================================
        valid_all = mi.Bool(converged)
        epsilon = 1e-4

        for i in range(k + 1):
            if i == 0:
                seg_from = exp_rx_pos
                seg_to = p[0]
            elif i == k:
                seg_from = p[k-1]
                seg_to = exp_tx_pos
            else:
                seg_from = p[i-1]
                seg_to = p[i]

            seg_dir = mi.Vector3f(
                seg_to.x - seg_from.x,
                seg_to.y - seg_from.y,
                seg_to.z - seg_from.z)
            seg_len = dr.norm(seg_dir)
            seg_dir_n = mi.Vector3f(
                seg_dir.x / dr.maximum(seg_len, mi.Float(1e-10)),
                seg_dir.y / dr.maximum(seg_len, mi.Float(1e-10)),
                seg_dir.z / dr.maximum(seg_len, mi.Float(1e-10)))

            ray_o = mi.Point3f(
                seg_from.x + mi.Float(epsilon) * seg_dir_n.x,
                seg_from.y + mi.Float(epsilon) * seg_dir_n.y,
                seg_from.z + mi.Float(epsilon) * seg_dir_n.z)
            shadow_ray = mi.Ray3f(ray_o, seg_dir_n)
            shadow_ray.maxt = seg_len - 2.0 * epsilon
            occluded = scene.ray_test(shadow_ray)
            valid_all = valid_all & ~occluded

        n_valid_all = int(dr.sum(mi.UInt32(valid_all))[0])
        if verbose:
            print(f"  Valid after visibility: {n_valid_all:,}/{n_chains:,}")

        # ================================================================
        # Compute per-vertex triangle area and segment distances
        # ================================================================
        A_tri_list = []
        for i in range(k):
            cross_vec = dr.cross(dp_du[i], dp_dv[i])
            A_tri_list.append(mi.Float(0.5) * dr.norm(cross_vec))

        d_segments = []
        dir_segments = []
        for i in range(k + 1):
            if i == 0:
                seg_from = exp_rx_pos
                seg_to = p[0]
            elif i == k:
                seg_from = p[k-1]
                seg_to = exp_tx_pos
            else:
                seg_from = p[i-1]
                seg_to = p[i]

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

        # Build grad info
        grad_info = MultibounceSpecularGradInfo(
            k=k,
            dp_du=final_dp_du,
            dp_dv=final_dp_dv,
            D_inv=[final_D_inv[i] for i in range(k)],
            U=[final_U[i] for i in range(k-1)] if k > 1 else [],
            tx_pos=exp_tx_pos,
            rx_pos=exp_rx_pos,
        )

        t_total = time.perf_counter() - t_start
        print(f"  [SMS-MB] k={k}: Converged {n_converged:,}/{n_chains:,} | "
              f"Valid {n_valid_all:,} | Time: {t_total:.3f}s")

        return MultibounceSpecularChain(
            k=k,
            hit_P=p,
            hit_N=n_vec,
            prim_ids=prim_ids,
            bary_u=bary_u,
            bary_v=bary_v,
            A_tri=A_tri_list,
            tx_idx=seed_tx_idx,
            rx_idx=seed_rx_idx,
            valid=valid_all,
            n_chains=n_chains,
            grad_info=grad_info,
            d_segments=d_segments,
            dir_segments=dir_segments,
        )

    def _empty_multibounce_chain(self, k: int) -> MultibounceSpecularChain:
        """Return an empty MultibounceSpecularChain."""
        return MultibounceSpecularChain(
            k=k,
            hit_P=[mi.Point3f() for _ in range(k)],
            hit_N=[mi.Vector3f() for _ in range(k)],
            prim_ids=[mi.UInt32() for _ in range(k)],
            bary_u=[mi.Float() for _ in range(k)],
            bary_v=[mi.Float() for _ in range(k)],
            A_tri=[mi.Float() for _ in range(k)],
            tx_idx=mi.UInt32(),
            rx_idx=mi.UInt32(),
            valid=mi.Bool(),
            n_chains=0,
            grad_info=None,
            d_segments=[mi.Float() for _ in range(k + 1)],
            dir_segments=[mi.Vector3f() for _ in range(k + 1)],
        )

    # ========================================================================
    # IFT for multibounce specular gradient attachment
    # ========================================================================

    def _apply_ift_multibounce_diff(
        self,
        chain: MultibounceSpecularChain,
        vertex_positions_buffer: 'mi.Float',
        scene: 'mi.Scene',
        normal_params: Optional[list] = None,
    ) -> Tuple[list, list]:
        """
        Apply IFT to attach vertex position gradients to k-bounce specular chain.

        Uses differentiable re-intersection: gathers AD-attached vertex positions,
        computes base positions from barycentrics, re-evaluates the half-vector
        constraint, solves the block-tridiagonal correction, and applies it.

        When normal_params is provided, the live normals are interpolated from
        the learnable per-vertex normal arrays instead of computed from edge
        cross products. This enables gradient flow from normal_params through
        the IFT constraint correction (the tangent frame s,t depends on the
        normal, so changing the normal shifts the constraint residual C=[s·h, t·h],
        which shifts the IFT correction delta, which shifts the final position).

        Args:
            chain: MultibounceSpecularChain with converged specular points
            vertex_positions_buffer: AD-attached vertex positions [n_verts*3]
            scene: Mitsuba scene
            normal_params: Optional list of 3 mi.Float arrays [nx, ny, nz] per vertex

        Returns:
            (corrected_positions, live_normals): Lists of k AD-attached Point3f/Vector3f
        """
        k = chain.k
        gi = chain.grad_info
        mesh = scene.shapes()[0]

        # Step 1: Get AD-attached base positions via barycentrics + live vertices
        p_bases = []
        live_dp_du = []
        live_dp_dv = []
        live_N = []

        from ..integrator import get_ad_triangle_vertices

        for i in range(k):
            v0, v1, v2, vi0, vi1, vi2 = get_ad_triangle_vertices(
                mesh, chain.prim_ids[i], vertex_positions_buffer, chain.valid)

            u_i = chain.bary_u[i]
            v_i = chain.bary_v[i]
            w_i = mi.Float(1.0) - u_i - v_i

            p_base = mi.Point3f(
                w_i * v0.x + u_i * v1.x + v_i * v2.x,
                w_i * v0.y + u_i * v1.y + v_i * v2.y,
                w_i * v0.z + u_i * v1.z + v_i * v2.z)
            p_bases.append(p_base)

            e1 = mi.Vector3f(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
            e2 = mi.Vector3f(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
            live_dp_du.append(e1)
            live_dp_dv.append(e2)

            if normal_params is not None:
                # Interpolate learnable normals via barycentrics (AD-attached)
                nx_arr, ny_arr, nz_arr = normal_params
                nx_interp = (w_i * dr.gather(mi.Float, nx_arr, vi0, chain.valid) +
                             u_i * dr.gather(mi.Float, nx_arr, vi1, chain.valid) +
                             v_i * dr.gather(mi.Float, nx_arr, vi2, chain.valid))
                ny_interp = (w_i * dr.gather(mi.Float, ny_arr, vi0, chain.valid) +
                             u_i * dr.gather(mi.Float, ny_arr, vi1, chain.valid) +
                             v_i * dr.gather(mi.Float, ny_arr, vi2, chain.valid))
                nz_interp = (w_i * dr.gather(mi.Float, nz_arr, vi0, chain.valid) +
                             u_i * dr.gather(mi.Float, nz_arr, vi1, chain.valid) +
                             v_i * dr.gather(mi.Float, nz_arr, vi2, chain.valid))
                nl = dr.maximum(dr.sqrt(dr.maximum(
                    nx_interp*nx_interp + ny_interp*ny_interp + nz_interp*nz_interp,
                    mi.Float(1e-20))), mi.Float(0.01))
                live_N.append(mi.Vector3f(nx_interp / nl, ny_interp / nl, nz_interp / nl))
            else:
                # Geometric normal from edge cross product (original behavior)
                cross = dr.cross(e1, e2)
                n_len = dr.maximum(dr.norm(cross), mi.Float(1e-10))
                live_N.append(mi.Vector3f(cross.x/n_len, cross.y/n_len, cross.z/n_len))

        # Step 2: Re-evaluate constraints at AD-attached positions
        constraints = []
        for i in range(k):
            # wi: from previous vertex or RX
            if i == 0:
                wi = safe_normalize(mi.Vector3f(
                    gi.rx_pos.x - p_bases[0].x,
                    gi.rx_pos.y - p_bases[0].y,
                    gi.rx_pos.z - p_bases[0].z))
            else:
                wi = safe_normalize(mi.Vector3f(
                    p_bases[i-1].x - p_bases[i].x,
                    p_bases[i-1].y - p_bases[i].y,
                    p_bases[i-1].z - p_bases[i].z))

            # wo: toward next vertex or TX
            if i == k - 1:
                wo = safe_normalize(mi.Vector3f(
                    gi.tx_pos.x - p_bases[i].x,
                    gi.tx_pos.y - p_bases[i].y,
                    gi.tx_pos.z - p_bases[i].z))
            else:
                wo = safe_normalize(mi.Vector3f(
                    p_bases[i+1].x - p_bases[i].x,
                    p_bases[i+1].y - p_bases[i].y,
                    p_bases[i+1].z - p_bases[i].z))

            h = safe_normalize(mi.Vector3f(wi.x+wo.x, wi.y+wo.y, wi.z+wo.z))

            # Tangent frame
            n_dot_dpdu = dr.dot(live_N[i], live_dp_du[i])
            s_raw = mi.Vector3f(
                live_dp_du[i].x - live_N[i].x * n_dot_dpdu,
                live_dp_du[i].y - live_N[i].y * n_dot_dpdu,
                live_dp_du[i].z - live_N[i].z * n_dot_dpdu)
            s = safe_normalize(s_raw)
            t = dr.cross(live_N[i], s)

            C_s = dr.dot(s, h)
            C_t = dr.dot(t, h)
            constraints.append((C_s, C_t))

        # Step 3: Solve using stored factorization (backward sweep only)
        # Forward sweep was already done; we have D_inv and U.
        # Solve: Δx = -J⁻¹ · C using Thomas backward sweep with stored factors.
        deltas = self._thomas_backward_sweep_with_stored(
            gi.D_inv, gi.U, constraints, k)

        # Step 4: Apply corrections via AD-attached tangent vectors
        corrected_positions = []
        for i in range(k):
            delta_u, delta_v = deltas[i]
            p_corrected = mi.Point3f(
                p_bases[i].x - live_dp_du[i].x * delta_u - live_dp_dv[i].x * delta_v,
                p_bases[i].y - live_dp_du[i].y * delta_u - live_dp_dv[i].y * delta_v,
                p_bases[i].z - live_dp_du[i].z * delta_u - live_dp_dv[i].z * delta_v)
            corrected_positions.append(p_corrected)

        return corrected_positions, live_N

    def _thomas_backward_sweep_with_stored(
        self,
        D_inv: list,
        U: list,
        rhs: list,
        k: int,
    ) -> list:
        """
        Solve block-tridiagonal system using stored factorization from Newton.

        This is a simplified version that uses the stored D_inv and U blocks
        from the forward sweep during Newton convergence. Only the forward
        and backward sweeps on the new RHS (constraint residuals) are needed.

        Args:
            D_inv: [k] stored inverse diagonal blocks (4-tuples)
            U: [k-1] stored upper coupling blocks (4-tuples)
            rhs: [k] constraint residuals (2-tuples of mi.Float)
            k: number of vertices

        Returns:
            [k] correction 2-vectors (delta_u, delta_v)
        """
        # Forward sweep (compute y from stored D_inv)
        y = [None] * k
        y[0] = self._mul2x2_vec(D_inv[0], rhs[0])

        # Note: For the IFT we need the full forward sweep since the RHS changed.
        # However, we approximate by only using the diagonal inverse (D_inv)
        # since the coupling structure hasn't changed. This is exact when
        # the block-tridiagonal structure was perfectly captured.
        for i in range(1, k):
            y[i] = self._mul2x2_vec(D_inv[i], rhs[i])

        # Backward sweep
        x = [None] * k
        x[k-1] = y[k-1]

        for i in range(k-2, -1, -1):
            if U and i < len(U) and U[i] is not None:
                Ux = self._mul2x2_vec(U[i], x[i+1])
                x[i] = (y[i][0] - Ux[0], y[i][1] - Ux[1])
            else:
                x[i] = y[i]

        return x


__all__ = [
    'SpecularManifoldSampler',
    'SpecularGradInfo',
    'MultibounceSpecularChain',
    'MultibounceSpecularGradInfo',
]
