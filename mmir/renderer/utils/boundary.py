"""
Projective boundary gradient computation for radar rendering.

Adapts Mitsuba 3's projective sampling (Zhang et al. 2023) from optical
to radar. Computes the missing visibility boundary term that the standard
AD graph cannot capture because scene.ray_test() is non-differentiable.

The boundary gradient is an ADDITIVE correction to the interior gradient:
    Total gradient = Interior gradient (from end-to-end AD)
                   + Boundary gradient (from projective sampling)

For each sampled silhouette edge point:
1. Reconstruct position from barycentrics + vertex positions (AD-attached)
2. Compute foreground phasor (surface at boundary -> TX/RX)
3. Compute background phasor (behind boundary -> TX/RX, or void=0)
4. Phasor difference = fg - bg
5. Weight by motion term / pdf / N_samples
6. Scatter-add to ADC arrays

Uses Mitsuba 3's shape-level silhouette APIs directly:
- shape.precompute_silhouette(viewpoint) -> (indices, weights)
- shape.sample_precomputed_silhouette(viewpoint, edge_idx, sample) -> SilhouetteSample3f
"""

from typing import Optional, TYPE_CHECKING, Tuple
import drjit as dr
import mitsuba as mi
import numpy as np

if TYPE_CHECKING:
    from ..config import RenderConfigRef

# Physical constants
C = 299792458.0


class BoundaryGradientComputer:
    """
    Computes boundary gradient contributions from silhouette edges.

    These corrections are scatter-added to the ADC arrays produced by the
    main end-to-end render pass. They account for the visibility
    discontinuity term that the standard AD graph misses.

    Usage:
        bgc = BoundaryGradientComputer(config, integrator)
        bgc.init_silhouettes(scene, tx_centroid)
        # In render loop:
        bgc.compute_boundary_gradients(
            adc_real, adc_imag, scene, tx_positions, rx_positions,
            vertex_offset_params, seed)
    """

    def __init__(self, config: 'RenderConfigRef', integrator=None):
        self.n_boundary_samples = getattr(config, 'n_boundary_samples_primary', 1024)
        self.enabled = getattr(config, 'enable_boundary_gradients', False)
        self.has_silhouettes = False
        self.integrator = integrator

        # Cache radar params from integrator if available
        if integrator is not None:
            self.K = integrator.num_samples
            self.sample_rate = integrator.sample_rate
            self.min_freq = integrator.min_freq
            self.slope = integrator.slope
        else:
            self.K = 256
            self.sample_rate = 8e6
            self.min_freq = 77e9
            self.slope = 29.982e12

    def init_silhouettes(self, scene: 'mi.Scene', tx_centroid: 'mi.ScalarPoint3f'):
        """
        Precompute visible silhouette edges from the TX centroid viewpoint.

        Uses the first mesh shape in the scene. Falls back gracefully if
        the shape has no silhouette edges.
        """
        shapes = scene.shapes()
        if len(shapes) == 0:
            self.has_silhouettes = False
            return

        # Use the first (and typically only) mesh shape
        self.shape = shapes[0]
        self.tx_centroid_scalar = tx_centroid

        try:
            indices, weights = self.shape.precompute_silhouette(tx_centroid)
            n_edges = len(indices)
        except Exception:
            self.has_silhouettes = False
            return

        if n_edges == 0:
            self.has_silhouettes = False
            return

        self.has_silhouettes = True
        self.edge_distribution = mi.DiscreteDistribution(weights)
        self.n_silhouette_edges = n_edges

    def sample_boundary_points(self, n_samples: int, seed: int):
        """
        Sample boundary points on silhouette edges.

        Returns:
            ss: SilhouetteSample3f with .p, .n, .silhouette_d, .prim_index,
                .uv, .pdf, .foreshortening, etc.
            valid: mi.Bool mask
        """
        sampler = mi.load_dict({'type': 'independent'})
        sampler.seed(seed + 9999, n_samples)  # offset seed from main render

        s1 = sampler.next_1d()
        s2 = sampler.next_1d()
        active = dr.full(mi.Bool, True, n_samples)

        # Sample edge from precomputed distribution
        edge_idx, _, _ = self.edge_distribution.sample_reuse_pmf(s1, active)

        # Broadcast viewpoint
        vp = mi.Point3f(
            dr.full(mi.Float, self.tx_centroid_scalar.x, n_samples),
            dr.full(mi.Float, self.tx_centroid_scalar.y, n_samples),
            dr.full(mi.Float, self.tx_centroid_scalar.z, n_samples),
        )

        # Sample point on edge (API: viewpoint, edge_idx, sample_on_edge, active)
        ss = self.shape.sample_precomputed_silhouette(vp, edge_idx, s2, active)
        dr.eval(ss.p, ss.n, ss.silhouette_d, ss.prim_index, ss.uv, ss.pdf)

        valid = active & (ss.pdf > 0)
        return ss, valid

    def _reconstruct_boundary_position(
        self,
        ss: 'mi.SilhouetteSample3f',
        vertex_offset_params: Optional[list],
        valid: 'mi.Bool',
    ) -> 'mi.Point3f':
        """
        Reconstruct boundary point position from barycentrics and vertex
        positions, optionally AD-attached through vertex offsets.

        P = (1-u-v) * V0 + u * V1 + v * V2
        where Vi = Vi_base + vertex_offset[vi]  (AD-attached if provided)
        """
        vbuf = self.shape.vertex_positions_buffer()
        fi = self.shape.face_indices(ss.prim_index)
        v0i, v1i, v2i = fi[0], fi[1], fi[2]

        # Gather base vertex positions
        v0x = dr.gather(mi.Float, vbuf, v0i * 3)
        v0y = dr.gather(mi.Float, vbuf, v0i * 3 + 1)
        v0z = dr.gather(mi.Float, vbuf, v0i * 3 + 2)
        v1x = dr.gather(mi.Float, vbuf, v1i * 3)
        v1y = dr.gather(mi.Float, vbuf, v1i * 3 + 1)
        v1z = dr.gather(mi.Float, vbuf, v1i * 3 + 2)
        v2x = dr.gather(mi.Float, vbuf, v2i * 3)
        v2y = dr.gather(mi.Float, vbuf, v2i * 3 + 1)
        v2z = dr.gather(mi.Float, vbuf, v2i * 3 + 2)

        # Add vertex offsets if provided (AD-attached)
        if vertex_offset_params is not None:
            dx, dy, dz = vertex_offset_params
            v0x = v0x + dr.gather(mi.Float, dx, v0i)
            v0y = v0y + dr.gather(mi.Float, dy, v0i)
            v0z = v0z + dr.gather(mi.Float, dz, v0i)
            v1x = v1x + dr.gather(mi.Float, dx, v1i)
            v1y = v1y + dr.gather(mi.Float, dy, v1i)
            v1z = v1z + dr.gather(mi.Float, dz, v1i)
            v2x = v2x + dr.gather(mi.Float, dx, v2i)
            v2y = v2y + dr.gather(mi.Float, dy, v2i)
            v2z = v2z + dr.gather(mi.Float, dz, v2i)

        # Barycentric interpolation: P = (1-u-v)*V0 + u*V1 + v*V2
        u = ss.uv.x
        v = ss.uv.y
        w = mi.Float(1.0) - u - v

        px = w * v0x + u * v1x + v * v2x
        py = w * v0y + u * v1y + v * v2y
        pz = w * v0z + u * v1z + v * v2z

        return mi.Point3f(px, py, pz)

    def compute_boundary_gradients(
        self,
        adc_real_flat: 'mi.Float',
        adc_imag_flat: 'mi.Float',
        scene: 'mi.Scene',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        vertex_offset_params: Optional[list] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Compute and scatter-add boundary gradient contributions to ADC arrays.

        The boundary term corrects for the visibility discontinuity at
        silhouette edges. For each sampled boundary point:

        1. Reconstruct position from barycentrics (AD-attached via vertex offsets)
        2. Evaluate foreground phasor (surface at boundary)
        3. Evaluate background phasor (behind boundary, or void=0)
        4. Compute motion term: |edge_tangent x view_dir|
        5. Weight: delta_phasor x motion / (pdf x N_samples)
        6. Scatter-add to ADC arrays
        """
        if not self.has_silhouettes:
            return

        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)
        K = self.K
        n_bs = self.n_boundary_samples

        # Step 1: Sample boundary points
        ss, valid = self.sample_boundary_points(n_bs, seed)
        dr.eval(valid)
        n_valid_bs = int(dr.sum(mi.UInt32(valid))[0])
        if n_valid_bs == 0:
            if verbose:
                print("[Boundary] No valid boundary samples")
            return

        if verbose:
            print(f"[Boundary] {n_valid_bs}/{n_bs} valid boundary samples")

        # Step 2: Reconstruct boundary position (AD-attached through vertex offsets)
        hit_P = self._reconstruct_boundary_position(ss, vertex_offset_params, valid)

        # Step 3: Compute motion term
        # motion = |edge_tangent x view_dir| -- the foreshortening of the edge
        # as seen from the viewpoint. This determines how much the visible
        # area changes as the silhouette edge moves.
        edge_tangent = ss.silhouette_d  # unit vector along edge
        view_dir = ss.d  # direction to viewpoint

        cross = dr.cross(edge_tangent, view_dir)
        motion = dr.norm(cross)
        # Clamp to avoid numerical issues
        motion = dr.maximum(motion, mi.Float(1e-8))

        # Step 4: Evaluate foreground phasor at reconstructed boundary position
        fg_phasor_real, fg_phasor_imag = self._eval_phasor_at_point(
            hit_P, ss.n, tx_positions, rx_positions, valid)

        # Step 5: Evaluate background phasor (what's behind the boundary)
        # Shoot ray from viewpoint through boundary point and continue
        vp = mi.Point3f(
            dr.full(mi.Float, self.tx_centroid_scalar.x, n_bs),
            dr.full(mi.Float, self.tx_centroid_scalar.y, n_bs),
            dr.full(mi.Float, self.tx_centroid_scalar.z, n_bs),
        )
        # Use detached position for the background ray (non-diff)
        hit_P_detached = mi.Point3f(dr.detach(hit_P.x), dr.detach(hit_P.y), dr.detach(hit_P.z))
        dir_through = dr.normalize(hit_P_detached - vp)
        ray_through = mi.Ray3f(
            hit_P_detached + mi.Float(1e-4) * dir_through,
            dir_through,
        )
        si_bg = scene.ray_intersect(ray_through, valid)
        bg_valid = valid & si_bg.is_valid()
        dr.eval(bg_valid)

        n_bg = int(dr.sum(mi.UInt32(bg_valid))[0])

        if n_bg > 0:
            bg_phasor_real, bg_phasor_imag = self._eval_phasor_at_point(
                si_bg.p, si_bg.n, tx_positions, rx_positions, bg_valid)
        else:
            n_total_K = n_bs * n_tx * n_rx * K
            bg_phasor_real = dr.zeros(mi.Float, n_total_K)
            bg_phasor_imag = dr.zeros(mi.Float, n_total_K)

        # Zero out background where no hit behind
        bg_active_mask = dr.tile(dr.repeat(bg_valid, n_tx * n_rx), K)
        bg_phasor_real = dr.select(bg_active_mask, bg_phasor_real, mi.Float(0.0))
        bg_phasor_imag = dr.select(bg_active_mask, bg_phasor_imag, mi.Float(0.0))

        # Step 6: Phasor difference x boundary weight
        delta_real = fg_phasor_real - bg_phasor_real
        delta_imag = fg_phasor_imag - bg_phasor_imag

        # Boundary weight = motion / pdf / N_samples
        pdf_safe = dr.maximum(ss.pdf, mi.Float(1e-10))
        boundary_weight = motion / pdf_safe / mi.Float(n_bs)

        # Expand weight for all TX x RX x K
        bw_expanded = dr.tile(dr.repeat(boundary_weight, n_tx * n_rx), K)

        delta_real = delta_real * bw_expanded
        delta_imag = delta_imag * bw_expanded

        # Step 7: Scatter-add to ADC arrays
        n_total_bs = n_bs * n_tx * n_rx
        arange_bs = dr.arange(mi.UInt32, n_total_bs * K)
        path_in_mimo = arange_bs // mi.UInt32(K)
        k_idx = arange_bs % mi.UInt32(K)

        tx_idx = (path_in_mimo // mi.UInt32(n_rx)) % mi.UInt32(n_tx)
        rx_idx = path_in_mimo % mi.UInt32(n_rx)

        flat_adc_idx = tx_idx * mi.UInt32(n_rx * K) + rx_idx * mi.UInt32(K) + k_idx

        # Active mask: only valid boundary points
        active_bs = dr.tile(dr.repeat(valid, n_tx * n_rx), K)

        dr.scatter_add(adc_real_flat, delta_real, flat_adc_idx, active_bs)
        dr.scatter_add(adc_imag_flat, delta_imag, flat_adc_idx, active_bs)

        if verbose:
            print(f"[Boundary] Scattered {n_valid_bs} boundary contributions "
                  f"({n_bg} with background)")

    def _eval_phasor_at_point(
        self,
        hit_P: 'mi.Point3f',
        hit_N: 'mi.Vector3f',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        active: 'mi.Bool',
    ) -> Tuple['mi.Float', 'mi.Float']:
        """
        Evaluate radar phasor contribution at surface points.

        Returns (phasor_real, phasor_imag) arrays of shape
        [n_points x n_tx x n_rx x K].

        Uses simplified weight: 1/(d_tx x d_rx). The boundary correction
        only needs relative foreground/background difference, so exact
        BSDF evaluation is not critical for the first-order correction.
        """
        from .math import gather_point3f

        n_points = dr.width(hit_P)
        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)
        K = self.K

        _n_mimo = n_tx * n_rx
        n_total = n_points * _n_mimo

        # MIMO expansion
        arange_total = dr.arange(mi.UInt32, n_total)
        pt_idx = arange_total // mi.UInt32(_n_mimo)
        tx_idx = (arange_total // mi.UInt32(n_rx)) % mi.UInt32(n_tx)
        rx_idx = arange_total % mi.UInt32(n_rx)

        hit_P_exp = mi.Point3f(
            dr.gather(mi.Float, hit_P.x, pt_idx),
            dr.gather(mi.Float, hit_P.y, pt_idx),
            dr.gather(mi.Float, hit_P.z, pt_idx),
        )
        active_exp = dr.gather(mi.Bool, active, pt_idx)

        tx_pos_exp = gather_point3f(tx_positions, tx_idx)
        rx_pos_exp = gather_point3f(rx_positions, rx_idx)

        # Geometry: distances to TX and RX
        delta_tx = tx_pos_exp - hit_P_exp
        d_tx = dr.norm(delta_tx)
        delta_rx = hit_P_exp - rx_pos_exp
        d_rx = dr.norm(delta_rx)

        # Simplified weight = 1/(d_tx * d_rx)
        d_product = dr.maximum(d_tx * d_rx, mi.Float(1e-8))
        weight = mi.Float(1.0) / d_product
        weight = dr.select(active_exp, weight, mi.Float(0.0))

        # Phase: phi = 2pi * (f0 * tau + S * tau * t_k)
        R_total = d_rx + d_tx
        tau = R_total / mi.Float(C)
        TWO_PI = mi.Float(2.0 * np.pi)
        phi_const = TWO_PI * mi.Float(self.min_freq) * tau
        phi_slope = TWO_PI * mi.Float(self.slope) * tau

        # Expand for K: [n_total x K]
        n_total_K = n_total * K
        path_idx = dr.arange(mi.UInt32, n_total_K) // mi.UInt32(K)
        k_idx = dr.arange(mi.UInt32, n_total_K) % mi.UInt32(K)

        pc = dr.gather(mi.Float, phi_const, path_idx)
        ps = dr.gather(mi.Float, phi_slope, path_idx)
        w = dr.gather(mi.Float, weight, path_idx)
        t_k = mi.Float(k_idx) / mi.Float(self.sample_rate)

        phi = pc + ps * t_k
        phasor_real = w * dr.cos(phi)
        phasor_imag = w * dr.sin(phi)

        return phasor_real, phasor_imag

    def compute_boundary_gradients_multibounce(
        self,
        adc_real_flat: 'mi.Float',
        adc_imag_flat: 'mi.Float',
        scene: 'mi.Scene',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        per_bounce_viewpoints: list,
        vertex_offset_params: Optional[list] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Compute boundary gradient contributions at each bounce level.

        For multibounce rendering, visibility boundary discontinuities arise
        at each bounce:
        - Bounce 1: silhouette edges as seen from RX (primary visibility)
        - Bounce b>1: silhouette edges as seen from bounce b-1's hit centroid
          (inter-bounce visibility)

        At each bounce level, silhouette edges are recomputed from that bounce's
        viewpoint, boundary points are sampled, and phasor difference corrections
        are scatter-added to the ADC arrays.

        Args:
            per_bounce_viewpoints: list of mi.ScalarPoint3f -- the viewpoint
                centroid for each bounce level. per_bounce_viewpoints[0] is
                the RX centroid (bounce 1), per_bounce_viewpoints[1] is the
                centroid of bounce-1 hits (for bounce 2), etc.
        """
        if not self.enabled:
            return

        shapes = scene.shapes()
        if len(shapes) == 0:
            return
        shape = shapes[0]

        n_total_boundary_corrections = 0

        for bounce_idx, viewpoint in enumerate(per_bounce_viewpoints):
            bounce_label = bounce_idx + 1

            # Re-initialize silhouettes from this bounce's viewpoint
            try:
                indices, weights = shape.precompute_silhouette(viewpoint)
                n_edges = len(indices)
            except Exception:
                if verbose:
                    print(f"[Boundary MB] Bounce {bounce_label}: "
                          f"silhouette precomputation failed, skipping")
                continue

            if n_edges == 0:
                if verbose:
                    print(f"[Boundary MB] Bounce {bounce_label}: "
                          f"no silhouette edges from viewpoint, skipping")
                continue

            # Temporarily overwrite instance state for sampling
            self.shape = shape
            self.tx_centroid_scalar = viewpoint
            self.edge_distribution = mi.DiscreteDistribution(weights)
            self.n_silhouette_edges = n_edges
            self.has_silhouettes = True

            # Use a unique seed per bounce to avoid correlated samples
            bounce_seed = seed + bounce_idx * 7777

            # Delegate to the existing single-bounce boundary gradient method
            self.compute_boundary_gradients(
                adc_real_flat=adc_real_flat,
                adc_imag_flat=adc_imag_flat,
                scene=scene,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                vertex_offset_params=vertex_offset_params,
                seed=bounce_seed,
                verbose=False,  # Suppress per-bounce verbosity
            )

            n_total_boundary_corrections += 1

            if verbose:
                print(f"[Boundary MB] Bounce {bounce_label}: "
                      f"computed boundary gradients from viewpoint "
                      f"({viewpoint.x:.2f}, {viewpoint.y:.2f}, {viewpoint.z:.2f}), "
                      f"{n_edges} silhouette edges")

        if verbose:
            print(f"[Boundary MB] Total: {n_total_boundary_corrections} "
                  f"bounce levels with boundary corrections "
                  f"(of {len(per_bounce_viewpoints)} requested)")


__all__ = [
    'BoundaryGradientComputer',
]
