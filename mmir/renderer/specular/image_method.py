"""
Image method refinement for single-bounce specular paths.

Adapted from Sionna RT's image_method.py and path_solver.py.

For each unique triangle discovered via SBR hits, computes the exact
specular reflection point deterministically for every (TX, RX) pair.

Algorithm (single-bounce):
1. Mirror TX across the reflecting plane: TX' = TX - 2 * dot(TX - V, N) * N
2. Find specular point at line-plane intersection: P_spec = TX' + t * dir
3. Validate: point-in-triangle, correct side, visibility
4. Compute deterministic weight: R_specular = η × τ × A

Reference: Sionna RT image_method.py (Phase 2 of their 3-phase pipeline)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import drjit as dr
import mitsuba as mi
import numpy as np

from ..bsdf.mmwave_scalar import (
    WAVELENGTH_77GHZ,
    map_renderer_params_to_physical,
    enforce_spm_validity,
    permittivity_to_ior,
    compute_energy_gate,
    compute_slab_energy_gate,
    compute_validity_aware_blend,
    compute_coherent_incoherent_blend,
)
from ..utils.math import safe_normalize

if TYPE_CHECKING:
    from ..utils.clustering import PatchData


@dataclass
class SpecularPaths:
    """
    Result of image method refinement.

    Contains exact specular reflection points and their weights for
    deterministic specular path contribution.

    All arrays are indexed over valid specular paths (variable length).
    """
    hit_P: 'mi.Point3f'         # Exact specular reflection points
    hit_N: 'mi.Vector3f'        # Surface normals
    d_tx: 'mi.Float'            # Distance TX → P_spec
    d_rx: 'mi.Float'            # Distance P_spec → RX
    dir_to_tx: 'mi.Vector3f'    # Direction P_spec → TX
    dir_to_rx: 'mi.Vector3f'    # Direction P_spec → RX
    cos_theta_i: 'mi.Float'     # cos(angle of incidence) = dot(dir_to_tx, N)
    cos_theta_r: 'mi.Float'     # cos(angle of reflection) = dot(dir_to_rx, N)
    tx_idx: 'mi.UInt32'         # TX index
    rx_idx: 'mi.UInt32'         # RX index
    valid: 'mi.Bool'            # Validity mask
    R_specular: 'mi.Float'      # η × τ × A  (specular reflectance)
    A_tri: 'mi.Float'           # Triangle area
    patch_id: Optional['mi.Int32']  # Patch ID per path (None if not using patches)
    prim_ids: Optional['mi.UInt32']  # Triangle primitive IDs per path (for Phase B material gather)
    n_paths: int                # Total number of potentially valid specular paths
    # IFT gradient support (populated by SMS, not image method)
    grad_info: Optional[object] = None     # SpecularGradInfo from SMS Newton solver
    bary_u: Optional['mi.Float'] = None    # Barycentric u at converged point
    bary_v: Optional['mi.Float'] = None    # Barycentric v at converged point


class ImageMethodRefiner:
    """
    Single-bounce specular path refinement via image method.

    For each unique triangle (identified by deduplication), computes the
    exact specular reflection point for every (TX, RX) pair.

    Args:
        wavelength: Operating wavelength in meters
        use_jones: If True, use Jones Fresnel for specular weight (future)
    """

    def __init__(
        self,
        wavelength: float = WAVELENGTH_77GHZ,
        bsdf=None,
    ):
        self.wavelength = wavelength
        self.bsdf = bsdf

    def _empty_specular_paths(self) -> SpecularPaths:
        """Return an empty SpecularPaths (no valid paths)."""
        return SpecularPaths(
            hit_P=mi.Point3f(), hit_N=mi.Vector3f(),
            d_tx=mi.Float(), d_rx=mi.Float(),
            dir_to_tx=mi.Vector3f(), dir_to_rx=mi.Vector3f(),
            cos_theta_i=mi.Float(), cos_theta_r=mi.Float(), tx_idx=mi.UInt32(),
            rx_idx=mi.UInt32(), valid=mi.Bool(),
            R_specular=mi.Float(), A_tri=mi.Float(),
            patch_id=None, prim_ids=None,
            n_paths=0,
        )

    def refine(
        self,
        unique_mask: 'mi.Bool',
        hit_P: 'mi.Point3f',
        hit_N: 'mi.Vector3f',
        hit_rho: np.ndarray,
        hit_tri_v0: 'mi.Point3f',
        hit_tri_v1: 'mi.Point3f',
        hit_tri_v2: 'mi.Point3f',
        tx_positions: 'mi.Point3f',
        rx_positions: 'mi.Point3f',
        n_hits_per_rx: int,
        scene: 'mi.Scene',
        verbose: bool = True,
        use_patches: bool = False,
        hit_ID: Optional['mi.Int32'] = None,
        patch_data: Optional['PatchData'] = None,
    ) -> SpecularPaths:
        """
        Compute exact specular paths for all unique surfaces x TX x RX.

        When use_patches=True:
          - Unique entries represent patches (clustered coplanar triangles)
          - P_spec computed on patch plane, tested against all triangles in patch
          - Area = total patch area (sum of constituent triangle areas)

        When use_patches=False:
          - Original per-triangle behavior (unchanged)

        Args:
            unique_mask: Boolean mask of unique hits [n_total_slots]
            hit_P: Hit positions [n_total_slots]
            hit_N: Surface normals [n_total_slots]
            hit_rho: Material params [n_total_slots, N] where N=3 (legacy) or N=6 (physics)
            hit_tri_v0/v1/v2: Triangle vertices [n_total_slots]
            tx_positions: TX element positions [n_tx]
            rx_positions: RX element positions [n_rx]
            n_hits_per_rx: Hits per RX in reservoir
            scene: Mitsuba scene for visibility checks
            verbose: Print progress
            use_patches: Use patch-level image method
            hit_ID: Triangle IDs [n_total_slots] (for patch mode)
            patch_data: PatchData from PatchClusterer (for patch mode)

        Returns:
            SpecularPaths with exact specular reflection data
        """
        if use_patches and hit_ID is not None and patch_data is not None:
            return self._refine_patches(
                unique_mask=unique_mask,
                hit_P=hit_P, hit_N=hit_N, hit_rho=hit_rho,
                hit_ID=hit_ID, patch_data=patch_data,
                tx_positions=tx_positions, rx_positions=rx_positions,
                n_hits_per_rx=n_hits_per_rx, scene=scene, verbose=verbose,
            )
        else:
            return self._refine_triangles(
                unique_mask=unique_mask,
                hit_P=hit_P, hit_N=hit_N, hit_rho=hit_rho,
                hit_tri_v0=hit_tri_v0, hit_tri_v1=hit_tri_v1, hit_tri_v2=hit_tri_v2,
                tx_positions=tx_positions, rx_positions=rx_positions,
                n_hits_per_rx=n_hits_per_rx, scene=scene, verbose=verbose,
                hit_ID=hit_ID,
            )

    def _refine_triangles(
        self,
        unique_mask, hit_P, hit_N, hit_rho,
        hit_tri_v0, hit_tri_v1, hit_tri_v2,
        tx_positions, rx_positions, n_hits_per_rx,
        scene, verbose,
        hit_ID=None,
    ) -> SpecularPaths:
        """Original per-triangle image method (unchanged logic)."""
        from ..utils.math import gather_point3f, gather_vector3f

        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)

        unique_np = np.array(unique_mask)
        unique_indices = np.where(unique_np)[0]
        n_unique = len(unique_indices)

        if verbose:
            print(f"\n[ImageMethodRefiner] Refining specular paths (TRIANGLE MODE)")
            print(f"  Unique triangles: {n_unique}")
            print(f"  TX elements: {n_tx}, RX elements: {n_rx}")

        if n_unique == 0:
            return self._empty_specular_paths()

        unique_idx_dr = mi.UInt32(unique_indices)

        # Gather unique hit data
        u_P = gather_point3f(hit_P, unique_idx_dr)
        u_N = gather_vector3f(hit_N, unique_idx_dr)
        u_rho = hit_rho[unique_np]

        # Gather triangle vertices
        u_v0 = gather_point3f(hit_tri_v0, unique_idx_dr)
        u_v1 = gather_point3f(hit_tri_v1, unique_idx_dr)
        u_v2 = gather_point3f(hit_tri_v2, unique_idx_dr)

        # Compute triangle areas
        edge1 = mi.Vector3f(u_v1.x - u_v0.x, u_v1.y - u_v0.y, u_v1.z - u_v0.z)
        edge2 = mi.Vector3f(u_v2.x - u_v0.x, u_v2.y - u_v0.y, u_v2.z - u_v0.z)
        tri_area = mi.Float(0.5) * dr.norm(dr.cross(edge1, edge2))

        u_rx_idx_np = unique_indices // n_hits_per_rx

        n_paths = n_unique * n_tx
        if verbose:
            print(f"  Specular path candidates: {n_unique} x {n_tx} TX = {n_paths:,}")

        # Expand unique hits x TX
        exp_N = mi.Vector3f(dr.repeat(u_N.x, n_tx), dr.repeat(u_N.y, n_tx), dr.repeat(u_N.z, n_tx))
        exp_v0 = mi.Point3f(dr.repeat(u_v0.x, n_tx), dr.repeat(u_v0.y, n_tx), dr.repeat(u_v0.z, n_tx))
        exp_v1 = mi.Point3f(dr.repeat(u_v1.x, n_tx), dr.repeat(u_v1.y, n_tx), dr.repeat(u_v1.z, n_tx))
        exp_v2 = mi.Point3f(dr.repeat(u_v2.x, n_tx), dr.repeat(u_v2.y, n_tx), dr.repeat(u_v2.z, n_tx))
        exp_A_tri = dr.repeat(tri_area, n_tx)
        n_mat_cols = u_rho.shape[1]
        physics_mode = (n_mat_cols == 6)

        if physics_mode:
            exp_eps_real   = mi.Float(np.repeat(u_rho[:, 0], n_tx))
            exp_eps_imag   = mi.Float(np.repeat(u_rho[:, 1], n_tx))
            exp_sigma_h    = mi.Float(np.repeat(u_rho[:, 2], n_tx))
            exp_l_c        = mi.Float(np.repeat(u_rho[:, 3], n_tx))
            exp_tau        = mi.Float(np.repeat(u_rho[:, 4], n_tx))
            exp_thickness  = mi.Float(np.repeat(u_rho[:, 5], n_tx))
        else:
            exp_albedo     = mi.Float(np.repeat(u_rho[:, 0], n_tx))
            exp_roughness  = mi.Float(np.repeat(u_rho[:, 1], n_tx))
            exp_metallic   = mi.Float(np.repeat(u_rho[:, 2], n_tx))
        exp_tx_idx = mi.UInt32(np.tile(np.arange(n_tx), n_unique))
        exp_rx_idx = mi.UInt32(np.repeat(u_rx_idx_np, n_tx))

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

        # Image method: mirror TX across triangle plane (v0 as reference)
        tx_minus_v = mi.Vector3f(exp_tx_pos.x - exp_v0.x, exp_tx_pos.y - exp_v0.y, exp_tx_pos.z - exp_v0.z)
        dot_tn = dr.dot(tx_minus_v, exp_N)
        tx_image = mi.Point3f(
            exp_tx_pos.x - mi.Float(2.0) * dot_tn * exp_N.x,
            exp_tx_pos.y - mi.Float(2.0) * dot_tn * exp_N.y,
            exp_tx_pos.z - mi.Float(2.0) * dot_tn * exp_N.z,
        )
        dir_image_to_rx = safe_normalize(mi.Vector3f(
            exp_rx_pos.x - tx_image.x, exp_rx_pos.y - tx_image.y, exp_rx_pos.z - tx_image.z,
        ))
        v_minus_tximg = mi.Vector3f(exp_v0.x - tx_image.x, exp_v0.y - tx_image.y, exp_v0.z - tx_image.z)
        numer = dr.dot(v_minus_tximg, exp_N)
        denom = dr.dot(dir_image_to_rx, exp_N)
        denom_safe = dr.select(dr.abs(denom) > mi.Float(1e-10), denom, mi.Float(1e-10))
        t_intersect = numer / denom_safe
        P_spec = mi.Point3f(
            tx_image.x + t_intersect * dir_image_to_rx.x,
            tx_image.y + t_intersect * dir_image_to_rx.y,
            tx_image.z + t_intersect * dir_image_to_rx.z,
        )

        # Validation
        valid_t = t_intersect > mi.Float(1e-6)
        valid_pit = self._point_in_triangle(P_spec, exp_v0, exp_v1, exp_v2, exp_N)

        delta_tx = mi.Vector3f(exp_tx_pos.x - P_spec.x, exp_tx_pos.y - P_spec.y, exp_tx_pos.z - P_spec.z)
        delta_rx = mi.Vector3f(exp_rx_pos.x - P_spec.x, exp_rx_pos.y - P_spec.y, exp_rx_pos.z - P_spec.z)
        d_tx = dr.norm(delta_tx)
        d_rx = dr.norm(delta_rx)
        valid_dist = (d_tx > mi.Float(1e-4)) & (d_rx > mi.Float(1e-4))

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
        cos_theta_i = dr.maximum(dr.dot(dir_to_tx, exp_N), mi.Float(0.0))
        cos_theta_r = dr.maximum(dr.dot(dir_to_rx, exp_N), mi.Float(0.0))
        valid_angle = cos_theta_i > mi.Float(1e-6)
        valid_geom = valid_t & valid_pit & valid_dist & valid_angle

        # Visibility (offset along surface normal to avoid self-intersection)
        valid_vis = self._check_visibility(P_spec, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
                                           surface_normal=exp_N)
        valid_all = valid_geom & valid_vis

        # Specular weight
        if physics_mode:
            R_specular = self._compute_specular_weight_physics(
                cos_theta_i, exp_eps_real, exp_eps_imag,
                exp_sigma_h, exp_l_c, exp_tau, exp_thickness
            )
        else:
            R_specular = self._compute_specular_weight(
                cos_theta_i, exp_albedo, exp_roughness, exp_metallic
            )
        R_specular = dr.select(valid_all, R_specular, mi.Float(0.0))

        # Stats with per-check diagnostics
        n_valid_geom = int(dr.sum(mi.UInt32(valid_geom))[0])
        n_valid_vis = int(dr.sum(mi.UInt32(valid_all))[0])
        if verbose:
            # Per-check failure breakdown (Step 4 diagnostic)
            n_valid_t = int(dr.sum(mi.UInt32(valid_t))[0])
            n_valid_pit = int(dr.sum(mi.UInt32(valid_pit))[0])
            n_valid_dist_count = int(dr.sum(mi.UInt32(valid_dist))[0])
            n_valid_angle_count = int(dr.sum(mi.UInt32(valid_angle))[0])
            n_valid_vis_only = int(dr.sum(mi.UInt32(valid_vis))[0])
            print(f"  Specular path validation breakdown ({n_paths:,} candidates):")
            print(f"    valid_t (forward intersect):  {n_valid_t:,} ({100*n_valid_t/max(n_paths,1):.1f}%)")
            print(f"    valid_pit (point-in-tri):     {n_valid_pit:,} ({100*n_valid_pit/max(n_paths,1):.1f}%)")
            print(f"    valid_dist (distance > eps):   {n_valid_dist_count:,} ({100*n_valid_dist_count/max(n_paths,1):.1f}%)")
            print(f"    valid_angle (cos_theta > 0):   {n_valid_angle_count:,} ({100*n_valid_angle_count/max(n_paths,1):.1f}%)")
            print(f"    valid_geom (all geometry):     {n_valid_geom:,} ({100*n_valid_geom/max(n_paths,1):.1f}%)")
            print(f"    valid_vis (visibility):        {n_valid_vis_only:,}")
            print(f"    valid_all (final):             {n_valid_vis:,} ({100*n_valid_vis/max(n_paths,1):.1f}%)")
            if n_valid_vis > 0:
                R_np = np.array(R_specular)
                valid_R = R_np[np.array(valid_all)]
                print(f"  R_specular range: [{valid_R.min():.6f}, {valid_R.max():.6f}]")
                print(f"  R_specular mean: {valid_R.mean():.6f}")

        # Gather primitive IDs for Phase B material re-evaluation
        exp_prim_ids = None
        if hit_ID is not None:
            u_prim_ids = dr.gather(mi.UInt32, mi.UInt32(hit_ID), unique_idx_dr)
            exp_prim_ids = dr.repeat(u_prim_ids, n_tx)

        return SpecularPaths(
            hit_P=P_spec, hit_N=exp_N, d_tx=d_tx, d_rx=d_rx,
            dir_to_tx=dir_to_tx, dir_to_rx=dir_to_rx,
            cos_theta_i=cos_theta_i, cos_theta_r=cos_theta_r,
            tx_idx=exp_tx_idx, rx_idx=exp_rx_idx,
            valid=valid_all, R_specular=R_specular, A_tri=exp_A_tri,
            patch_id=None, prim_ids=exp_prim_ids,
            n_paths=n_paths,
        )

    def _refine_patches(
        self,
        unique_mask, hit_P, hit_N, hit_rho, hit_ID, patch_data,
        tx_positions, rx_positions, n_hits_per_rx, scene, verbose,
    ) -> SpecularPaths:
        """
        Patch-level image method refinement.

        Key differences from _refine_triangles():
        1. Unique entries are patches, not triangles
        2. Specular point computed using patch plane geometry
        3. PIT tests P_spec against ALL triangles in the patch
        4. Area is total patch area (sum of triangle areas)
        """
        n_tx = dr.width(tx_positions)
        n_rx = dr.width(rx_positions)

        unique_np = np.array(unique_mask)
        unique_indices = np.where(unique_np)[0]
        n_unique = len(unique_indices)

        if verbose:
            print(f"\n[ImageMethodRefiner] Refining specular paths (PATCH MODE)")
            print(f"  Unique patches: {n_unique} (out of {patch_data.n_patches} total)")
            print(f"  Max triangles per patch: {patch_data.max_tris_per_patch}")
            print(f"  TX elements: {n_tx}, RX elements: {n_rx}")

        if n_unique == 0:
            return self._empty_specular_paths()

        unique_idx_dr = mi.UInt32(unique_indices)

        # Map hit_ID → patch_id for unique hits
        u_hit_ID = dr.gather(mi.Int32, hit_ID, unique_idx_dr)
        safe_id = mi.UInt32(dr.maximum(u_hit_ID, mi.Int32(0)))
        u_patch_id = mi.UInt32(dr.gather(mi.Int32, patch_data.tri_to_patch, safe_id))

        # Gather patch plane geometry
        u_N = mi.Vector3f(
            dr.gather(mi.Float, patch_data.patch_normal.x, u_patch_id),
            dr.gather(mi.Float, patch_data.patch_normal.y, u_patch_id),
            dr.gather(mi.Float, patch_data.patch_normal.z, u_patch_id),
        )
        u_offset = dr.gather(mi.Float, patch_data.patch_offset, u_patch_id)
        u_area = dr.gather(mi.Float, patch_data.patch_area, u_patch_id)

        # Material from representative SBR hit
        u_rho = hit_rho[unique_np]  # [n_unique, 4]
        u_rx_idx_np = unique_indices // n_hits_per_rx

        # Expand unique patches x TX
        n_paths = n_unique * n_tx
        if verbose:
            print(f"  Specular path candidates: {n_unique} x {n_tx} TX = {n_paths:,}")

        exp_N = mi.Vector3f(dr.repeat(u_N.x, n_tx), dr.repeat(u_N.y, n_tx), dr.repeat(u_N.z, n_tx))
        exp_offset = dr.repeat(u_offset, n_tx)
        exp_area = dr.repeat(u_area, n_tx)
        exp_patch_id = dr.repeat(u_patch_id, n_tx)
        n_mat_cols = u_rho.shape[1]
        physics_mode = (n_mat_cols == 6)

        if physics_mode:
            exp_eps_real   = mi.Float(np.repeat(u_rho[:, 0], n_tx))
            exp_eps_imag   = mi.Float(np.repeat(u_rho[:, 1], n_tx))
            exp_sigma_h    = mi.Float(np.repeat(u_rho[:, 2], n_tx))
            exp_l_c        = mi.Float(np.repeat(u_rho[:, 3], n_tx))
            exp_tau        = mi.Float(np.repeat(u_rho[:, 4], n_tx))
            exp_thickness  = mi.Float(np.repeat(u_rho[:, 5], n_tx))
        else:
            exp_albedo     = mi.Float(np.repeat(u_rho[:, 0], n_tx))
            exp_roughness  = mi.Float(np.repeat(u_rho[:, 1], n_tx))
            exp_metallic   = mi.Float(np.repeat(u_rho[:, 2], n_tx))
        exp_tx_idx = mi.UInt32(np.tile(np.arange(n_tx), n_unique))
        exp_rx_idx = mi.UInt32(np.repeat(u_rx_idx_np, n_tx))

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

        # Image method: mirror TX across patch plane
        # Reconstruct point on plane: V = N * offset (since offset = dot(N, V) and N is unit)
        plane_pt = mi.Point3f(exp_N.x * exp_offset, exp_N.y * exp_offset, exp_N.z * exp_offset)

        tx_minus_v = mi.Vector3f(
            exp_tx_pos.x - plane_pt.x, exp_tx_pos.y - plane_pt.y, exp_tx_pos.z - plane_pt.z,
        )
        dot_tn = dr.dot(tx_minus_v, exp_N)
        tx_image = mi.Point3f(
            exp_tx_pos.x - mi.Float(2.0) * dot_tn * exp_N.x,
            exp_tx_pos.y - mi.Float(2.0) * dot_tn * exp_N.y,
            exp_tx_pos.z - mi.Float(2.0) * dot_tn * exp_N.z,
        )
        dir_image_to_rx = safe_normalize(mi.Vector3f(
            exp_rx_pos.x - tx_image.x, exp_rx_pos.y - tx_image.y, exp_rx_pos.z - tx_image.z,
        ))

        # Line-plane intersection
        v_minus_tximg = mi.Vector3f(
            plane_pt.x - tx_image.x, plane_pt.y - tx_image.y, plane_pt.z - tx_image.z,
        )
        numer = dr.dot(v_minus_tximg, exp_N)
        denom = dr.dot(dir_image_to_rx, exp_N)
        denom_safe = dr.select(dr.abs(denom) > mi.Float(1e-10), denom, mi.Float(1e-10))
        t_intersect = numer / denom_safe
        P_spec = mi.Point3f(
            tx_image.x + t_intersect * dir_image_to_rx.x,
            tx_image.y + t_intersect * dir_image_to_rx.y,
            tx_image.z + t_intersect * dir_image_to_rx.z,
        )

        # Geometry validation (pre-PIT)
        valid_t = t_intersect > mi.Float(1e-6)
        delta_tx = mi.Vector3f(exp_tx_pos.x - P_spec.x, exp_tx_pos.y - P_spec.y, exp_tx_pos.z - P_spec.z)
        delta_rx = mi.Vector3f(exp_rx_pos.x - P_spec.x, exp_rx_pos.y - P_spec.y, exp_rx_pos.z - P_spec.z)
        d_tx = dr.norm(delta_tx)
        d_rx = dr.norm(delta_rx)
        valid_dist = (d_tx > mi.Float(1e-4)) & (d_rx > mi.Float(1e-4))
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
        cos_theta_i = dr.maximum(dr.dot(dir_to_tx, exp_N), mi.Float(0.0))
        cos_theta_r = dr.maximum(dr.dot(dir_to_rx, exp_N), mi.Float(0.0))
        valid_angle = cos_theta_i > mi.Float(1e-6)
        valid_geom_pre_pit = valid_t & valid_dist & valid_angle

        # PATCH-LEVEL Point-in-Triangle test (vectorized over all triangles in patch)
        valid_pit = self._point_in_patch(
            P_spec=P_spec, exp_N=exp_N, exp_patch_id=exp_patch_id,
            patch_data=patch_data, n_paths=n_paths,
            valid_pre_mask=valid_geom_pre_pit,
        )
        valid_geom = valid_geom_pre_pit & valid_pit

        # Visibility (offset along surface normal to avoid self-intersection)
        valid_vis = self._check_visibility(P_spec, dir_to_tx, dir_to_rx, d_tx, d_rx, scene,
                                           surface_normal=exp_N)
        valid_all = valid_geom & valid_vis

        # Specular weight
        if physics_mode:
            R_specular = self._compute_specular_weight_physics(
                cos_theta_i, exp_eps_real, exp_eps_imag,
                exp_sigma_h, exp_l_c, exp_tau, exp_thickness
            )
        else:
            R_specular = self._compute_specular_weight(
                cos_theta_i, exp_albedo, exp_roughness, exp_metallic
            )
        R_specular = dr.select(valid_all, R_specular, mi.Float(0.0))

        # Statistics with per-check diagnostics (Step 4 diagnostic)
        n_valid_pre = int(dr.sum(mi.UInt32(valid_geom_pre_pit))[0])
        n_valid_pit_count = int(dr.sum(mi.UInt32(valid_pit & valid_geom_pre_pit))[0])
        n_valid_geom = int(dr.sum(mi.UInt32(valid_geom))[0])
        n_valid_vis_count = int(dr.sum(mi.UInt32(valid_all))[0])
        if verbose:
            n_valid_t = int(dr.sum(mi.UInt32(valid_t))[0])
            n_valid_dist_count = int(dr.sum(mi.UInt32(valid_dist))[0])
            n_valid_angle_count = int(dr.sum(mi.UInt32(valid_angle))[0])
            n_valid_vis_only = int(dr.sum(mi.UInt32(valid_vis))[0])
            print(f"  Specular path validation breakdown ({n_paths:,} candidates, PATCH mode):")
            print(f"    valid_t (forward intersect):  {n_valid_t:,} ({100*n_valid_t/max(n_paths,1):.1f}%)")
            print(f"    valid_dist (distance > eps):   {n_valid_dist_count:,} ({100*n_valid_dist_count/max(n_paths,1):.1f}%)")
            print(f"    valid_angle (cos_theta > 0):   {n_valid_angle_count:,} ({100*n_valid_angle_count/max(n_paths,1):.1f}%)")
            print(f"    valid_geom_pre_pit:            {n_valid_pre:,} ({100*n_valid_pre/max(n_paths,1):.1f}%)")
            print(f"    valid_pit (point-in-patch):    {n_valid_pit_count:,} ({100*n_valid_pit_count/max(n_paths,1):.1f}%)")
            print(f"    valid_geom (all geometry):     {n_valid_geom:,} ({100*n_valid_geom/max(n_paths,1):.1f}%)")
            print(f"    valid_vis (visibility):        {n_valid_vis_only:,}")
            print(f"    valid_all (final):             {n_valid_vis_count:,} ({100*n_valid_vis_count/max(n_paths,1):.1f}%)")
            if n_valid_vis_count > 0:
                R_np = np.array(R_specular)
                valid_R = R_np[np.array(valid_all)]
                print(f"  R_specular range: [{valid_R.min():.6f}, {valid_R.max():.6f}]")
                print(f"  R_specular mean: {valid_R.mean():.6f}")

        # Expand primitive IDs for Phase B material re-evaluation
        exp_prim_ids = mi.UInt32(dr.repeat(mi.UInt32(dr.maximum(u_hit_ID, mi.Int32(0))), n_tx))

        return SpecularPaths(
            hit_P=P_spec, hit_N=exp_N, d_tx=d_tx, d_rx=d_rx,
            dir_to_tx=dir_to_tx, dir_to_rx=dir_to_rx,
            cos_theta_i=cos_theta_i, cos_theta_r=cos_theta_r,
            tx_idx=exp_tx_idx, rx_idx=exp_rx_idx,
            valid=valid_all, R_specular=R_specular, A_tri=exp_area,
            patch_id=mi.Int32(exp_patch_id), prim_ids=exp_prim_ids,
            n_paths=n_paths,
        )

    def _point_in_patch(
        self,
        P_spec: 'mi.Point3f',
        exp_N: 'mi.Vector3f',
        exp_patch_id: 'mi.UInt32',
        patch_data: 'PatchData',
        n_paths: int,
        valid_pre_mask: 'mi.Bool',
    ) -> 'mi.Bool':
        """
        Vectorized point-in-patch test.

        Tests P_spec against ALL triangles in the corresponding patch.
        Valid if ANY triangle contains P_spec.

        Strategy (pure DrJit, no loops):
        1. Expand [n_paths] -> [n_paths * max_tris]
        2. Gather triangle vertices from padded arrays
        3. Run vectorized PIT on all entries at once
        4. Reduce back: scatter_add per candidate, valid if count > 0
        """
        max_tris = patch_data.max_tris_per_patch
        n_expanded = n_paths * max_tris

        # Expand P_spec: each point repeated max_tris times
        exp_P = mi.Point3f(
            dr.repeat(P_spec.x, max_tris),
            dr.repeat(P_spec.y, max_tris),
            dr.repeat(P_spec.z, max_tris),
        )

        # Expand normal: each normal repeated max_tris times
        exp_pit_N = mi.Vector3f(
            dr.repeat(exp_N.x, max_tris),
            dr.repeat(exp_N.y, max_tris),
            dr.repeat(exp_N.z, max_tris),
        )

        # Compute flat index into padded arrays:
        # flat_idx = patch_id * max_tris + local_tri_idx
        local_tri_idx = dr.tile(dr.arange(mi.UInt32, max_tris), n_paths)
        exp_pid = dr.repeat(exp_patch_id, max_tris)
        flat_idx = exp_pid * mi.UInt32(max_tris) + local_tri_idx

        # Gather triangle vertices from precomputed padded arrays
        exp_v0 = mi.Point3f(
            dr.gather(mi.Float, patch_data.patch_tri_v0.x, flat_idx),
            dr.gather(mi.Float, patch_data.patch_tri_v0.y, flat_idx),
            dr.gather(mi.Float, patch_data.patch_tri_v0.z, flat_idx),
        )
        exp_v1 = mi.Point3f(
            dr.gather(mi.Float, patch_data.patch_tri_v1.x, flat_idx),
            dr.gather(mi.Float, patch_data.patch_tri_v1.y, flat_idx),
            dr.gather(mi.Float, patch_data.patch_tri_v1.z, flat_idx),
        )
        exp_v2 = mi.Point3f(
            dr.gather(mi.Float, patch_data.patch_tri_v2.x, flat_idx),
            dr.gather(mi.Float, patch_data.patch_tri_v2.y, flat_idx),
            dr.gather(mi.Float, patch_data.patch_tri_v2.z, flat_idx),
        )

        # Gather validity mask (False for padding entries)
        tri_valid = dr.gather(mi.Bool, patch_data.patch_tri_valid, flat_idx)

        # Expand pre-PIT validity mask
        exp_valid_pre = dr.repeat(valid_pre_mask, max_tris)

        # Active: only test real triangles for pre-valid candidates
        active = tri_valid & exp_valid_pre

        # Vectorized PIT test on all n_expanded entries
        pit_result = self._point_in_triangle(exp_P, exp_v0, exp_v1, exp_v2, exp_pit_N)
        pit_result = pit_result & active

        # Reduce: scatter_add per candidate, valid if count > 0
        candidate_idx = dr.arange(mi.UInt32, n_expanded) // mi.UInt32(max_tris)
        hit_count = dr.zeros(mi.UInt32, n_paths)
        dr.scatter_add(hit_count, mi.UInt32(pit_result), candidate_idx)
        # CRITICAL: eval before comparison on scatter-modified variable
        dr.eval(hit_count)

        valid_pit = hit_count > mi.UInt32(0)
        return valid_pit

    @staticmethod
    def _check_visibility(P_spec, dir_to_tx, dir_to_rx, d_tx, d_rx, scene, surface_normal=None):
        """Shadow ray visibility check for specular paths.

        The shadow ray origin is offset from P_spec along the surface normal
        (if provided) to avoid self-intersection with the source triangle.
        This offset is for visibility ONLY — the actual P_spec used for
        phase/distance computation is unchanged, preserving MIMO phase coherence.
        """
        # Offset along surface normal to avoid self-intersection
        # Use 1e-3 * normal (larger than the standard 1e-4 directional offset)
        # to clear noisy LiDAR mesh surfaces
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

        shadow_origin_rx = mi.Point3f(
            P_offset.x + mi.Float(epsilon) * dir_to_rx.x,
            P_offset.y + mi.Float(epsilon) * dir_to_rx.y,
            P_offset.z + mi.Float(epsilon) * dir_to_rx.z,
        )
        shadow_rays_rx = mi.Ray3f(shadow_origin_rx, dir_to_rx)
        shadow_rays_rx.maxt = d_rx - 2.0 * epsilon
        occluded_rx = scene.ray_test(shadow_rays_rx)

        return ~occluded_tx & ~occluded_rx

    def _compute_specular_weight(
        self,
        cos_theta_i: 'mi.Float',
        albedo: 'mi.Float',
        roughness: 'mi.Float',
        metallic: 'mi.Float',
    ) -> 'mi.Float':
        """
        Compute deterministic specular path reflectance: R_specular = η × τ × A.

        Following Sionna's eval() (line 596-608): the specular path's field
        coefficient is purely deterministic physics, not MC-sampled.

        Args:
            cos_theta_i: Cosine of incident angle
            albedo: Material albedo
            roughness: Material roughness
            metallic: Material metallic factor

        Returns:
            R_specular: Specular reflectance ∈ [0, 1]
        """
        # Map renderer params to physical params (same path as eval_f)
        eps_real, eps_imag, sigma_h_raw, l_c_raw, tau_base = \
            map_renderer_params_to_physical(albedo, roughness, metallic, self.wavelength)

        sigma_h, l_c = enforce_spm_validity(sigma_h_raw, l_c_raw, self.wavelength)
        n_ior, kappa = permittivity_to_ior(eps_real, eps_imag)

        # Energy gate: A = complex Fresnel reflectance (unpolarized)
        A = compute_energy_gate(n_ior, kappa, cos_theta_i,
                                mi.Float(0.5), mi.Float(0.5))

        # Coherent fraction: η from coherent/incoherent blend
        eta, _ = compute_coherent_incoherent_blend(sigma_h, l_c, self.wavelength, cos_theta_i)

        # KA fraction of coherent: τ
        tau_eff = compute_validity_aware_blend(
            tau_base, cos_theta_i, sigma_h, l_c, self.wavelength
        )

        # R_specular = η × τ × A
        R_specular = eta * tau_eff * A

        return R_specular

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
        """
        Compute specular reflectance for physics-mode materials: R_specular = eta * tau * A.

        Uses compute_slab_energy_gate() (thickness-dependent Fresnel) instead of
        compute_energy_gate() (single-interface), matching the physics-mode BSDF path.

        Args:
            cos_theta_i: Cosine of incident angle
            eps_real: Real relative permittivity
            eps_imag: Imaginary permittivity magnitude
            sigma_h: RMS surface height (m)
            l_c: Correlation length (m)
            tau: KA fraction (specular weight)
            thickness: Slab thickness (m)

        Returns:
            R_specular: Specular reflectance in [0, 1]
        """
        sigma_h_v, l_c_v = enforce_spm_validity(sigma_h, l_c, self.wavelength)

        # Energy gate A via slab Fresnel (matches _eval_f_core physics path)
        A = compute_slab_energy_gate(
            eps_real, eps_imag, cos_theta_i, thickness,
            mi.Float(0.5), mi.Float(0.5),  # unpolarized
            wavelength=self.wavelength,
            sigma_h=sigma_h_v,
        )

        # Coherent fraction eta
        eta, _ = compute_coherent_incoherent_blend(
            sigma_h_v, l_c_v, self.wavelength, cos_theta_i
        )

        # KA fraction tau_eff
        tau_eff = compute_validity_aware_blend(
            tau, cos_theta_i, sigma_h_v, l_c_v, self.wavelength
        )

        R_specular = eta * tau_eff * A
        return R_specular

    @staticmethod
    def _point_in_triangle(
        P: 'mi.Point3f',
        v0: 'mi.Point3f',
        v1: 'mi.Point3f',
        v2: 'mi.Point3f',
        N: 'mi.Vector3f',
    ) -> 'mi.Bool':
        """
        Point-in-triangle test using barycentric coordinates.

        Computes barycentric coords (u, v, w) where P = u*v0 + v*v1 + w*v2.
        P is inside triangle iff u, v, w ∈ [0, 1] and u + v + w ≈ 1.

        Uses the cross-product method for robustness.

        Args:
            P: Test point
            v0, v1, v2: Triangle vertices
            N: Triangle normal (for area computation)

        Returns:
            inside: Boolean mask
        """
        # Edge vectors
        e0 = mi.Vector3f(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z)
        e1 = mi.Vector3f(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z)
        ep = mi.Vector3f(P.x - v0.x, P.y - v0.y, P.z - v0.z)

        # Signed areas via dot with normal
        # Full triangle area (2x): N . (e0 × e1)
        full_area = dr.dot(N, dr.cross(e0, e1))

        # Sub-triangle areas
        # u = N . (e0 × ep) / full_area  (barycentric coord for v2)
        # v = N . (ep × e1) / full_area  (barycentric coord for v1... wait)
        # Actually use standard formulation:
        d00 = dr.dot(e0, e0)
        d01 = dr.dot(e0, e1)
        d11 = dr.dot(e1, e1)
        d20 = dr.dot(ep, e0)
        d21 = dr.dot(ep, e1)

        inv_denom = mi.Float(1.0) / dr.maximum(d00 * d11 - d01 * d01, mi.Float(1e-12))
        v_bary = (d11 * d20 - d01 * d21) * inv_denom
        w_bary = (d00 * d21 - d01 * d20) * inv_denom
        u_bary = mi.Float(1.0) - v_bary - w_bary

        # Tolerance for numerical errors at triangle edges.
        # Relaxed from 1e-4 to 1e-3 for LiDAR meshes with small (1-5cm) triangles
        # where floating-point precision in image point computation matters.
        EPS_BARY = mi.Float(-1e-3)
        ONE_PLUS_EPS = mi.Float(1.0 + 1e-3)

        inside = (u_bary >= EPS_BARY) & (v_bary >= EPS_BARY) & (w_bary >= EPS_BARY) & \
                 (u_bary <= ONE_PLUS_EPS) & (v_bary <= ONE_PLUS_EPS) & (w_bary <= ONE_PLUS_EPS)

        return inside


__all__ = [
    'ImageMethodRefiner',
    'SpecularPaths',
]
