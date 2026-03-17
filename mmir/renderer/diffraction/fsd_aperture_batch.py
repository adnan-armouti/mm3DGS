"""
Batch aperture construction: vectorized NumPy replacement for the serial
construct_aperture loop. Processes all N hits simultaneously with zero
Python for-loops over hits or triangles.

This implements P1 from PLAN_diffraction_performance_optimization.md.
"""

import numpy as np
from typing import Optional, List, Tuple, Union

from .fsd_bsdf import FlatEdgeData
from .fsd_aperture import (
    _fresnel_opacity_numpy_batch,
    _jones_reflectance_numpy_batch,
    _area_circle_tri_batch_numpy,
    _INTEGRAL_1,
    _INTEGRAL_2,
)


def construct_apertures_batch(
    hit_pos_all: np.ndarray,         # [N, 3] hit positions
    wo_all: np.ndarray,              # [N, 3] directions toward RX
    tri_hash,                        # TriangleSpatialHash
    k: float,                        # wavenumber
    beam_sigma: float,               # Gaussian beam sigma
    max_tessellation_depth: int = 5,
    max_edges_per_hit: int = 256,
    fill_min: float = 1e-6,
    fill_max: float = 1.0 - 1e-6,
    tri_eps_real: Optional[np.ndarray] = None,   # [F] float32
    tri_eps_imag: Optional[np.ndarray] = None,   # [F] float32
    tx_polarization: Optional[np.ndarray] = None,  # [3]
    rx_polarization: Optional[np.ndarray] = None,  # [3]
    jones_mode: bool = False,
    beta_max: float = 0.5,
    verbose: bool = False,
    edge_angle_threshold_deg: float = 15.0,  # min dihedral angle (degrees) for diffracting edges
    precomputed_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Tuple[FlatEdgeData, np.ndarray]]:
    """
    Batch-construct apertures for all N hits simultaneously.

    Returns (FlatEdgeData, active_hit_indices) where active_hit_indices is
    an [A] array mapping aperture index → input hit index [0..N-1].
    The ap_to_path_indices field is left empty — the caller must populate it.
    Returns None if no hits produce diffraction.
    """
    N = len(hit_pos_all)
    if N == 0:
        return None

    search_radius = 3.0 * beam_sigma
    circle_area = np.pi * search_radius ** 2
    max_edge_len_sq = beam_sigma ** 2
    use_material_opacity = (tri_eps_real is not None and tri_eps_imag is not None)
    jones_mode = jones_mode and use_material_opacity

    # ================================================================
    # Step 1: Batch frame setup (tangent/bitangent for all N hits)
    # ================================================================
    wo_unit = wo_all.astype(np.float64)
    wo_norms = np.linalg.norm(wo_unit, axis=1, keepdims=True)
    wo_unit = wo_unit / np.maximum(wo_norms, 1e-12)

    up = np.array([0.0, 0.0, 1.0])
    dot_wo_up = wo_unit @ up  # [N]
    tangent = up[None, :] - dot_wo_up[:, None] * wo_unit  # [N, 3]

    # Handle near-parallel case
    parallel_mask = np.abs(dot_wo_up) > 0.999
    if np.any(parallel_mask):
        alt_up = np.array([0.0, 1.0, 0.0])
        dot_alt = wo_unit[parallel_mask] @ alt_up
        tangent[parallel_mask] = alt_up[None, :] - dot_alt[:, None] * wo_unit[parallel_mask]

    t_norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(t_norms, 1e-12)
    bitangent = np.cross(wo_unit, tangent)  # [N, 3]

    # ================================================================
    # Step 2: Batch spatial query — get nearby faces for all hits
    # ================================================================
    if precomputed_pairs is not None:
        # GPU spatial query already done — use provided (hit_idx, face_idx) pairs
        hit_idx, face_idx = precomputed_pairs
        hit_idx = hit_idx.astype(np.int32)
        face_idx = face_idx.astype(np.int32)
        M = len(hit_idx)
        if M == 0:
            return None
    else:
        nearby_lists = tri_hash.query_sphere_batch(
            hit_pos_all.astype(np.float32), search_radius
        )

        # Flatten ragged results into contiguous (hit_idx, face_idx) pairs
        pair_counts = np.array([len(lst) for lst in nearby_lists], dtype=np.int32)
        M = int(np.sum(pair_counts))
        if M == 0:
            return None

        hit_idx = np.repeat(np.arange(N, dtype=np.int32), pair_counts)  # [M]
        face_idx = np.concatenate(nearby_lists) if M > 0 else np.empty(0, dtype=np.int32)  # [M]

    # ================================================================
    # Step 3: Batch back-face culling
    # ================================================================
    face_normals = tri_hash.face_normals  # [F, 3]
    normals_m = face_normals[face_idx].astype(np.float64)  # [M, 3]
    wo_m = wo_unit[hit_idx]  # [M, 3]
    dots = np.einsum('ij,ij->i', wo_m, normals_m)  # [M]
    front_mask = dots > 0

    if not np.any(front_mask):
        return None

    hit_idx_f = hit_idx[front_mask]   # [M_f]
    face_idx_f = face_idx[front_mask]  # [M_f]
    dots_f = dots[front_mask]          # [M_f]
    M_f = len(hit_idx_f)

    # ================================================================
    # Step 4: Batch vertex projection onto per-hit virtual screens
    # ================================================================
    hit_pos_f = hit_pos_all[hit_idx_f].astype(np.float64)  # [M_f, 3]
    v0 = tri_hash.tri_v0[face_idx_f].astype(np.float64)    # [M_f, 3]
    v1 = tri_hash.tri_v1[face_idx_f].astype(np.float64)
    v2 = tri_hash.tri_v2[face_idx_f].astype(np.float64)

    d0 = v0 - hit_pos_f  # [M_f, 3]
    d1 = v1 - hit_pos_f
    d2 = v2 - hit_pos_f

    tang_f = tangent[hit_idx_f]    # [M_f, 3]
    bitang_f = bitangent[hit_idx_f]  # [M_f, 3]
    wo_f = wo_unit[hit_idx_f]       # [M_f, 3]

    # Screen coordinates
    u1x = np.einsum('ij,ij->i', d0, tang_f)    # [M_f]
    u1y = np.einsum('ij,ij->i', d0, bitang_f)
    u2x = np.einsum('ij,ij->i', d1, tang_f)
    u2y = np.einsum('ij,ij->i', d1, bitang_f)
    u3x = np.einsum('ij,ij->i', d2, tang_f)
    u3y = np.einsum('ij,ij->i', d2, bitang_f)

    # Z-depths along -wo direction
    z1 = -np.einsum('ij,ij->i', d0, wo_f)  # [M_f]
    z2 = -np.einsum('ij,ij->i', d1, wo_f)
    z3 = -np.einsum('ij,ij->i', d2, wo_f)

    # ================================================================
    # Step 5: Batch fill fraction (per hit) and early-exit check
    # ================================================================
    # Use shoelace area (fast) for fill fraction in batch mode
    area_2x = np.abs(
        (u2x - u1x) * (u3y - u1y) - (u3x - u1x) * (u2y - u1y)
    )  # [M_f] — twice the triangle area

    fill_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(fill_per_hit, hit_idx_f, area_2x * 0.5)
    fill_per_hit /= circle_area

    skip_fill = (fill_per_hit > fill_max) | (fill_per_hit < fill_min)
    # Mark pairs belonging to skipped hits
    keep_fill = ~skip_fill[hit_idx_f]  # [M_f]

    # Apply fill filter
    if not np.all(keep_fill):
        hit_idx_f = hit_idx_f[keep_fill]
        face_idx_f = face_idx_f[keep_fill]
        dots_f = dots_f[keep_fill]
        u1x = u1x[keep_fill]; u1y = u1y[keep_fill]
        u2x = u2x[keep_fill]; u2y = u2y[keep_fill]
        u3x = u3x[keep_fill]; u3y = u3y[keep_fill]
        z1 = z1[keep_fill]; z2 = z2[keep_fill]; z3 = z3[keep_fill]
        tang_f = tang_f[keep_fill]
        bitang_f = bitang_f[keep_fill]
        wo_f = wo_f[keep_fill]
        hit_pos_f = hit_pos_f[keep_fill]
        M_f = len(hit_idx_f)

    if M_f == 0:
        return None

    # ================================================================
    # Step 6: Batch boundary edge detection with dihedral angle filtering
    # ================================================================
    nn_f = tri_hash.neighbor_normals[face_idx_f].astype(np.float64)  # [M_f, 3, 3]
    wo_dot_nn = np.einsum('ij,ikj->ik', wo_f, nn_f)  # [M_f, 3]
    nn_norms = np.linalg.norm(nn_f, axis=2)  # [M_f, 3]
    is_mesh_boundary = nn_norms < 0.5
    has_neighbor = ~is_mesh_boundary

    # Dihedral angle: cos(angle) = dot(face_normal, neighbor_normal) / |nn|
    fn_f = face_normals[face_idx_f].astype(np.float64)  # [M_f, 3]
    fn_expanded = fn_f[:, np.newaxis, :]  # [M_f, 1, 3]
    cos_dihedral = np.einsum('ikj,ikj->ik', fn_expanded.repeat(3, axis=1), nn_f)  # [M_f, 3]
    cos_dihedral = cos_dihedral / np.maximum(nn_norms, 1e-8)

    # Pre-computed edge suppression flags (Filters E/D/C)
    edge_suppressed_f = tri_hash.edge_is_suppressed[face_idx_f]  # [M_f, 3]

    # An edge is diffracting only if:
    # 1. Has a neighbor AND neighbor is back-facing (wo_dot <= 0)
    # 2. AND the dihedral angle exceeds threshold (not coplanar)
    # 3. AND the edge is not suppressed by mesh-quality filters
    # Boundary edges (no neighbor) are NOT diffracting — on LiDAR meshes,
    # missing neighbors are usually mesh artifacts, not real geometric edges.
    cos_threshold = np.cos(np.radians(edge_angle_threshold_deg))
    is_significant_edge = cos_dihedral < cos_threshold
    edge_is_diffracting = has_neighbor & (wo_dot_nn <= 0) & is_significant_edge & ~edge_suppressed_f  # [M_f, 3]

    # ================================================================
    # Step 7: Batch material pre-computation (Fresnel / Jones)
    # ================================================================
    normals_ff = face_normals[face_idx_f].astype(np.float64)  # [M_f, 3]
    fn_norms = np.linalg.norm(normals_ff, axis=1, keepdims=True)
    fn_unit = normals_ff / np.maximum(fn_norms, 1e-8)
    cos_theta_f = np.abs(np.einsum('ij,ij->i', wo_f, fn_unit))  # [M_f]

    if use_material_opacity:
        eps_r_f = tri_eps_real[face_idx_f].astype(np.float64)  # [M_f]
        eps_i_f = tri_eps_imag[face_idx_f].astype(np.float64)
        opacity_f = _fresnel_opacity_numpy_batch(eps_r_f, eps_i_f, cos_theta_f)  # [M_f]

        if jones_mode:
            jr_f, ji_f, jpow_f, ts_f, tp_f, rs_f, rp_f = \
                _jones_reflectance_numpy_batch(
                    eps_r_f, eps_i_f, cos_theta_f,
                    wo_f[:1][0],  # All hits share same wo (small variation)
                    fn_unit,
                    tx_polarization, rx_polarization,
                )
        else:
            jr_f = np.ones(M_f)
            ji_f = np.zeros(M_f)
            jpow_f = np.ones(M_f)
            ts_f = tp_f = rs_f = rp_f = np.zeros(M_f)
    else:
        opacity_f = np.ones(M_f, dtype=np.float64)
        jr_f = np.ones(M_f)
        ji_f = np.zeros(M_f)
        jpow_f = np.ones(M_f)
        ts_f = tp_f = rs_f = rp_f = np.zeros(M_f)

    # ================================================================
    # Step 8: Iterative tessellation (vectorized longest-edge bisection)
    # ================================================================
    # Reference: free_space_diffraction.cpp addtri()
    # Subdivides triangles whose projected screen-space edges exceed
    # beam_sigma, up to max_tessellation_depth levels. Uses longest-edge
    # bisection (always 2 children) for full vectorizability.
    # When max_tessellation_depth=0, this step is skipped entirely.
    if max_tessellation_depth > 0:
        _orig_M_f = M_f
        # Working arrays for current-level sub-triangles
        cur_u1x = u1x; cur_u1y = u1y
        cur_u2x = u2x; cur_u2y = u2y
        cur_u3x = u3x; cur_u3y = u3y
        cur_z1 = z1; cur_z2 = z2; cur_z3 = z3
        cur_e12 = edge_is_diffracting[:, 0]
        cur_e13 = edge_is_diffracting[:, 1]
        cur_e23 = edge_is_diffracting[:, 2]
        cur_hit = hit_idx_f; cur_face = face_idx_f
        cur_pair = np.arange(M_f, dtype=np.int32)

        # Barycentric coordinates of each sub-triangle vertex within the original triangle.
        # Used to track tessellated edge endpoint positions for AD vertex re-projection.
        # u1=v0 → (0,0), u2=v1 → (1,0), u3=v2 → (0,1)
        cur_bu1 = np.zeros(M_f, dtype=np.float64)   # bary_u of vertex u1
        cur_bv1 = np.zeros(M_f, dtype=np.float64)   # bary_v of vertex u1
        cur_bu2 = np.ones(M_f, dtype=np.float64)     # bary_u of vertex u2
        cur_bv2 = np.zeros(M_f, dtype=np.float64)   # bary_v of vertex u2
        cur_bu3 = np.zeros(M_f, dtype=np.float64)   # bary_u of vertex u3
        cur_bv3 = np.ones(M_f, dtype=np.float64)     # bary_v of vertex u3

        # Leaf sub-triangle collectors (appended at each depth level)
        Lc = {k: [] for k in [
            'u1x','u1y','u2x','u2y','u3x','u3y',
            'z1','z2','z3','e12','e13','e23','hit','face','pair',
            'bu1','bv1','bu2','bv2','bu3','bv3']}

        for _depth in range(max_tessellation_depth):
            T = len(cur_u1x)
            if T == 0:
                break

            # Screen-space edge lengths²
            l12 = (cur_u2x - cur_u1x)**2 + (cur_u2y - cur_u1y)**2
            l13 = (cur_u3x - cur_u1x)**2 + (cur_u3y - cur_u1y)**2
            l23 = (cur_u3x - cur_u2x)**2 + (cur_u3y - cur_u2y)**2

            needs_sub = (l12 > max_edge_len_sq) | (l13 > max_edge_len_sq) | (l23 > max_edge_len_sq)

            # Collect leaves (all edges ≤ beam_sigma)
            leaf_m = ~needs_sub
            if np.any(leaf_m):
                for _k, _a in [('u1x',cur_u1x),('u1y',cur_u1y),
                               ('u2x',cur_u2x),('u2y',cur_u2y),
                               ('u3x',cur_u3x),('u3y',cur_u3y),
                               ('z1',cur_z1),('z2',cur_z2),('z3',cur_z3),
                               ('e12',cur_e12),('e13',cur_e13),('e23',cur_e23),
                               ('hit',cur_hit),('face',cur_face),('pair',cur_pair),
                               ('bu1',cur_bu1),('bv1',cur_bv1),
                               ('bu2',cur_bu2),('bv2',cur_bv2),
                               ('bu3',cur_bu3),('bv3',cur_bv3)]:
                    Lc[_k].append(_a[leaf_m])

            if not np.any(needs_sub):
                cur_u1x = np.empty(0, dtype=np.float64)
                break

            # Extract non-leaves for subdivision
            si = needs_sub
            su1x = cur_u1x[si]; su1y = cur_u1y[si]
            su2x = cur_u2x[si]; su2y = cur_u2y[si]
            su3x = cur_u3x[si]; su3y = cur_u3y[si]
            sz1 = cur_z1[si]; sz2 = cur_z2[si]; sz3 = cur_z3[si]
            se12 = cur_e12[si]; se13 = cur_e13[si]; se23 = cur_e23[si]
            s_hit = cur_hit[si]; s_face = cur_face[si]; s_pair = cur_pair[si]
            sbu1 = cur_bu1[si]; sbv1 = cur_bv1[si]
            sbu2 = cur_bu2[si]; sbv2 = cur_bv2[si]
            sbu3 = cur_bu3[si]; sbv3 = cur_bv3[si]

            # Find longest edge → split axis
            longest = np.argmax(
                np.column_stack([l12[si], l13[si], l23[si]]), axis=1)
            split_12 = longest == 0
            split_13 = longest == 1
            split_23 = longest == 2

            # Midpoint of the longest edge
            mid_x = np.where(split_12, (su1x + su2x) * 0.5,
                     np.where(split_13, (su1x + su3x) * 0.5,
                                        (su2x + su3x) * 0.5))
            mid_y = np.where(split_12, (su1y + su2y) * 0.5,
                     np.where(split_13, (su1y + su3y) * 0.5,
                                        (su2y + su3y) * 0.5))
            mid_z = np.where(split_12, (sz1 + sz2) * 0.5,
                     np.where(split_13, (sz1 + sz3) * 0.5,
                                        (sz2 + sz3) * 0.5))

            # Midpoint barycentrics: average of the two parent vertices
            mid_bu = np.where(split_12, (sbu1 + sbu2) * 0.5,
                      np.where(split_13, (sbu1 + sbu3) * 0.5,
                                         (sbu2 + sbu3) * 0.5))
            mid_bv = np.where(split_12, (sbv1 + sbv2) * 0.5,
                      np.where(split_13, (sbv1 + sbv3) * 0.5,
                                         (sbv2 + sbv3) * 0.5))

            # ----- Sub-triangle A -----
            # edge12 → A=(u1, mid, u3)
            # edge13 → A=(u1, u2, mid)
            # edge23 → A=(u1, mid, u3)
            a_u1x = su1x; a_u1y = su1y; a_z1 = sz1
            a_u2x = np.where(split_13, su2x, mid_x)
            a_u2y = np.where(split_13, su2y, mid_y)
            a_z2  = np.where(split_13, sz2, mid_z)
            a_u3x = np.where(split_13, mid_x, su3x)
            a_u3y = np.where(split_13, mid_y, su3y)
            a_z3  = np.where(split_13, mid_z, sz3)
            # Sub-A edge boundary flags
            a_e12 = se12 & ~split_23
            a_e13 = se13
            a_e23 = se23 & split_23
            # Sub-A vertex barycentrics (same pattern as screen coords)
            a_bu1 = sbu1; a_bv1 = sbv1
            a_bu2 = np.where(split_13, sbu2, mid_bu)
            a_bv2 = np.where(split_13, sbv2, mid_bv)
            a_bu3 = np.where(split_13, mid_bu, sbu3)
            a_bv3 = np.where(split_13, mid_bv, sbv3)

            # ----- Sub-triangle B -----
            # edge12 → B=(mid, u2, u3)
            # edge13 → B=(mid, u2, u3)
            # edge23 → B=(u1, u2, mid)
            b_u1x = np.where(split_23, su1x, mid_x)
            b_u1y = np.where(split_23, su1y, mid_y)
            b_z1  = np.where(split_23, sz1, mid_z)
            b_u2x = su2x; b_u2y = su2y; b_z2 = sz2
            b_u3x = np.where(split_23, mid_x, su3x)
            b_u3y = np.where(split_23, mid_y, su3y)
            b_z3  = np.where(split_23, mid_z, sz3)
            # Sub-B edge boundary flags
            b_e12 = se12 & ~split_13
            b_e13 = se13 & split_13
            b_e23 = se23
            # Sub-B vertex barycentrics
            b_bu1 = np.where(split_23, sbu1, mid_bu)
            b_bv1 = np.where(split_23, sbv1, mid_bv)
            b_bu2 = sbu2; b_bv2 = sbv2
            b_bu3 = np.where(split_23, mid_bu, sbu3)
            b_bv3 = np.where(split_23, mid_bv, sbv3)

            # Concatenate children → next level's working set
            cur_u1x = np.concatenate([a_u1x, b_u1x])
            cur_u1y = np.concatenate([a_u1y, b_u1y])
            cur_u2x = np.concatenate([a_u2x, b_u2x])
            cur_u2y = np.concatenate([a_u2y, b_u2y])
            cur_u3x = np.concatenate([a_u3x, b_u3x])
            cur_u3y = np.concatenate([a_u3y, b_u3y])
            cur_z1 = np.concatenate([a_z1, b_z1])
            cur_z2 = np.concatenate([a_z2, b_z2])
            cur_z3 = np.concatenate([a_z3, b_z3])
            cur_e12 = np.concatenate([a_e12, b_e12])
            cur_e13 = np.concatenate([a_e13, b_e13])
            cur_e23 = np.concatenate([a_e23, b_e23])
            cur_hit = np.concatenate([s_hit, s_hit])
            cur_face = np.concatenate([s_face, s_face])
            cur_pair = np.concatenate([s_pair, s_pair])
            cur_bu1 = np.concatenate([a_bu1, b_bu1])
            cur_bv1 = np.concatenate([a_bv1, b_bv1])
            cur_bu2 = np.concatenate([a_bu2, b_bu2])
            cur_bv2 = np.concatenate([a_bv2, b_bv2])
            cur_bu3 = np.concatenate([a_bu3, b_bu3])
            cur_bv3 = np.concatenate([a_bv3, b_bv3])

        # Remaining triangles at max depth → leaves (regardless of edge length)
        if len(cur_u1x) > 0:
            for _k, _a in [('u1x',cur_u1x),('u1y',cur_u1y),
                           ('u2x',cur_u2x),('u2y',cur_u2y),
                           ('u3x',cur_u3x),('u3y',cur_u3y),
                           ('z1',cur_z1),('z2',cur_z2),('z3',cur_z3),
                           ('e12',cur_e12),('e13',cur_e13),('e23',cur_e23),
                           ('hit',cur_hit),('face',cur_face),('pair',cur_pair),
                           ('bu1',cur_bu1),('bv1',cur_bv1),
                           ('bu2',cur_bu2),('bv2',cur_bv2),
                           ('bu3',cur_bu3),('bv3',cur_bv3)]:
                Lc[_k].append(_a)

        # Concatenate all leaf sub-triangles
        if not Lc['u1x']:
            return None

        leaf_pair_idx = np.concatenate(Lc['pair'])
        L = len(leaf_pair_idx)

        if verbose:
            print(f"  [Tessellation] {_orig_M_f} → {L} leaf sub-tris "
                  f"(max_depth={max_tessellation_depth})")

        # Overwrite working arrays with leaf data
        u1x = np.concatenate(Lc['u1x']); u1y = np.concatenate(Lc['u1y'])
        u2x = np.concatenate(Lc['u2x']); u2y = np.concatenate(Lc['u2y'])
        u3x = np.concatenate(Lc['u3x']); u3y = np.concatenate(Lc['u3y'])
        z1 = np.concatenate(Lc['z1']); z2 = np.concatenate(Lc['z2'])
        z3 = np.concatenate(Lc['z3'])
        edge_is_diffracting = np.stack([
            np.concatenate(Lc['e12']),
            np.concatenate(Lc['e13']),
            np.concatenate(Lc['e23']),
        ], axis=1)
        hit_idx_f = np.concatenate(Lc['hit'])
        face_idx_f = np.concatenate(Lc['face'])
        M_f = L

        # Concatenate vertex barycentrics from leaf sub-triangles
        leaf_bu1 = np.concatenate(Lc['bu1']); leaf_bv1 = np.concatenate(Lc['bv1'])
        leaf_bu2 = np.concatenate(Lc['bu2']); leaf_bv2 = np.concatenate(Lc['bv2'])
        leaf_bu3 = np.concatenate(Lc['bu3']); leaf_bv3 = np.concatenate(Lc['bv3'])

        # Reindex per-pair material data for leaf sub-triangles
        opacity_f = opacity_f[leaf_pair_idx]
        jr_f = jr_f[leaf_pair_idx]; ji_f = ji_f[leaf_pair_idx]
        jpow_f = jpow_f[leaf_pair_idx]
        ts_f = ts_f[leaf_pair_idx]; tp_f = tp_f[leaf_pair_idx]
        rs_f = rs_f[leaf_pair_idx]; rp_f = rp_f[leaf_pair_idx]
        cos_theta_f = cos_theta_f[leaf_pair_idx]

        if M_f == 0:
            return None

    # Default vertex barycentrics when tessellation is skipped
    if max_tessellation_depth <= 0 or 'leaf_bu1' not in dir():
        leaf_bu1 = np.zeros(M_f, dtype=np.float64)   # u1=v0 → (0,0)
        leaf_bv1 = np.zeros(M_f, dtype=np.float64)
        leaf_bu2 = np.ones(M_f, dtype=np.float64)    # u2=v1 → (1,0)
        leaf_bv2 = np.zeros(M_f, dtype=np.float64)
        leaf_bu3 = np.zeros(M_f, dtype=np.float64)   # u3=v2 → (0,1)
        leaf_bv3 = np.ones(M_f, dtype=np.float64)

    # ================================================================
    # Step 9: Batch beam amplitudes
    # ================================================================
    inv_4s2 = -0.25 / (beam_sigma ** 2)
    norm_phi = 1.0 / (np.sqrt(2.0 * np.pi) * beam_sigma)

    r2_1 = u1x**2 + u1y**2 + z1**2  # [M_f]
    r2_2 = u2x**2 + u2y**2 + z2**2
    r2_3 = u3x**2 + u3y**2 + z3**2

    ph1 = np.exp(inv_4s2 * r2_1) * norm_phi  # [M_f]
    ph2 = np.exp(inv_4s2 * r2_2) * norm_phi
    ph3 = np.exp(inv_4s2 * r2_3) * norm_phi

    # ================================================================
    # Step 10: Batch power integrals per leaf sub-triangle
    # ================================================================
    # Shoelace area (2×area)
    area_2x_f = np.abs(
        (u2x - u1x) * (u3y - u1y) - (u3x - u1x) * (u2y - u1y)
    )  # [M_f]

    # _Pt: area * (ph1² + ph2² + ph3² + ph1*ph2 + ph1*ph3 + ph2*ph3) / 12
    Pt_f = area_2x_f * (
        ph1**2 + ph2**2 + ph3**2 + ph1*ph2 + ph1*ph3 + ph2*ph3
    ) / 12.0  # [M_f]

    # _Psi0t: (ph1 + ph2 + ph3) * area / 6
    Psi0t_f = (ph1 + ph2 + ph3) * area_2x_f / 6.0  # [M_f]

    # _Sigmat: weighted second moments (vectorized)
    Sigmat_xx_f, Sigmat_xy_f, Sigmat_yy_f = _batch_Sigmat(
        u1x, u1y, u2x, u2y, u3x, u3y, ph1, ph2, ph3
    )  # each [M_f]

    # Effective opacity
    if jones_mode and use_material_opacity:
        eff_opacity_f = jpow_f  # [M_f]
    else:
        eff_opacity_f = opacity_f  # [M_f]

    sqrt_eff_f = np.sqrt(eff_opacity_f)  # [M_f]

    # ================================================================
    # Step 11: Expand to 3 potential edges per leaf → filter by boundary
    # ================================================================
    # Edge type 0: vertices (1→0), i.e. (v1,v0) with ph2,ph1 — matches construct_aperture edge12
    # Edge type 1: vertices (0→2), i.e. (v0,v2) with ph1,ph3 — matches edge13
    # Edge type 2: vertices (2→1), i.e. (v2,v1) with ph3,ph2 — matches edge23
    # Flatten: 3*M_f potential edges
    edge_valid = edge_is_diffracting.ravel()  # [3*M_f] — edge types 0,1,2 interleaved per pair

    # Parent pair indices for each potential edge
    parent_idx = np.repeat(np.arange(M_f, dtype=np.int32), 3)  # [3*M_f]
    edge_type = np.tile(np.arange(3, dtype=np.int32), M_f)     # [3*M_f]: 0,1,2,0,1,2,...

    # Filter to valid boundary edges
    valid_sel = np.where(edge_valid)[0]  # indices into [3*M_f]
    if len(valid_sel) == 0:
        return None

    E = len(valid_sel)
    par = parent_idx[valid_sel]        # [E] — parent pair index
    etype = edge_type[valid_sel]       # [E] — edge type (0, 1, or 2)

    # ================================================================
    # Edge vertex coordinates based on edge type
    # Edge type 0 (edge12): u_a = u2, u_b = u1, ph_a = ph2, ph_b = ph1, z_a = z2, z_b = z1
    # Edge type 1 (edge13): u_a = u1, u_b = u3, ph_a = ph1, ph_b = ph3, z_a = z1, z_b = z3
    # Edge type 2 (edge23): u_a = u3, u_b = u2, ph_a = ph3, ph_b = ph2, z_a = z3, z_b = z2
    # ================================================================
    # Gather vertex data for each edge (vectorized select by edge type)
    # Build arrays for all 3 edge types then index
    #   type=0: a=(v1=u2), b=(v0=u1)
    #   type=1: a=(v0=u1), b=(v2=u3)
    #   type=2: a=(v2=u3), b=(v1=u2)
    uax = np.where(etype == 0, u2x[par], np.where(etype == 1, u1x[par], u3x[par]))  # [E]
    uay = np.where(etype == 0, u2y[par], np.where(etype == 1, u1y[par], u3y[par]))
    ubx = np.where(etype == 0, u1x[par], np.where(etype == 1, u3x[par], u2x[par]))
    uby = np.where(etype == 0, u1y[par], np.where(etype == 1, u3y[par], u2y[par]))
    za_e = np.where(etype == 0, z2[par], np.where(etype == 1, z1[par], z3[par]))
    zb_e = np.where(etype == 0, z1[par], np.where(etype == 1, z3[par], z2[par]))
    pha_e = np.where(etype == 0, ph2[par], np.where(etype == 1, ph1[par], ph3[par]))
    phb_e = np.where(etype == 0, ph1[par], np.where(etype == 1, ph3[par], ph2[par]))

    # Barycentric coordinates of edge endpoints (tessellation-tracked).
    # These track actual vertex positions through longest-edge bisection.
    #   type=0 (u2→u1): ba at u2's bary, bb at u1's bary
    #   type=1 (u1→u3): ba at u1's bary, bb at u3's bary
    #   type=2 (u3→u2): ba at u3's bary, bb at u2's bary
    ba_u_e = np.where(etype == 0, leaf_bu2[par], np.where(etype == 1, leaf_bu1[par], leaf_bu3[par]))
    ba_v_e = np.where(etype == 0, leaf_bv2[par], np.where(etype == 1, leaf_bv1[par], leaf_bv3[par]))
    bb_u_e = np.where(etype == 0, leaf_bu1[par], np.where(etype == 1, leaf_bu3[par], leaf_bu2[par]))
    bb_v_e = np.where(etype == 0, leaf_bv1[par], np.where(etype == 1, leaf_bv3[par], leaf_bv2[par]))

    # Edge vector and midpoint
    ex_e = ubx - uax  # [E]
    ey_e = uby - uay
    vmx_e = (uax + ubx) * 0.5  # [E]
    vmy_e = (uay + uby) * 0.5

    # Triangle centroid for winding check
    cx_e = (u1x[par] + u2x[par] + u3x[par]) / 3.0  # [E]
    cy_e = (u1y[par] + u2y[par] + u3y[par]) / 3.0

    # ================================================================
    # Winding correction: outward normal m = (ey, -ex) should point
    # away from centroid. If not, flip edge direction and swap endpoints.
    # ================================================================
    mx = ey_e     # [E]
    my = -ex_e
    winding_dot = mx * (vmx_e - cx_e) + my * (vmy_e - cy_e)  # [E]
    flip = winding_dot < 0  # [E] bool — need to flip

    # Flip: negate edge vector, swap a↔b
    ex_e = np.where(flip, -ex_e, ex_e)
    ey_e = np.where(flip, -ey_e, ey_e)
    pha_e_w = np.where(flip, phb_e, pha_e)
    phb_e_w = np.where(flip, pha_e, phb_e)
    za_e_w = np.where(flip, zb_e, za_e)
    zb_e_w = np.where(flip, za_e, zb_e)
    # Swap barycentrics too
    ba_u_e_w = np.where(flip, bb_u_e, ba_u_e)
    ba_v_e_w = np.where(flip, bb_v_e, ba_v_e)
    bb_u_e_w = np.where(flip, ba_u_e, bb_u_e)
    bb_v_e_w = np.where(flip, ba_v_e, bb_v_e)

    # ================================================================
    # Complex amplitudes: ca = phi_a * exp(i*k*z_a) * material_mod
    # ================================================================
    phase_a = np.exp(1j * k * za_e_w)  # [E] complex
    phase_b = np.exp(1j * k * zb_e_w)
    ca = pha_e_w * phase_a  # [E] complex
    cb = phb_e_w * phase_b

    # Material modulation
    edge_opacity_e = opacity_f[par]  # [E]
    if jones_mode and use_material_opacity:
        E_jones = jr_f[par] + 1j * ji_f[par]  # [E] complex
        ca = ca * E_jones
        cb = cb * E_jones
    elif use_material_opacity:
        sqrt_op = np.sqrt(edge_opacity_e)  # [E]
        ca = ca * sqrt_op
        cb = cb * sqrt_op

    # ================================================================
    # Pjhat: edge-diffracted power
    # ================================================================
    e_len_sq = ex_e**2 + ey_e**2  # [E]
    Pjhat = e_len_sq * (
        np.abs(ca - cb)**2 * _INTEGRAL_1 +
        np.abs(ca + cb)**2 / 4.0 * _INTEGRAL_2
    )  # [E]

    # Filter edges with zero power
    power_valid = Pjhat > 0
    valid_edge_mask = power_valid
    if not np.all(valid_edge_mask):
        sel = np.where(valid_edge_mask)[0]
        par = par[sel]; etype = etype[sel]
        ex_e = ex_e[sel]; ey_e = ey_e[sel]
        vmx_e = vmx_e[sel]; vmy_e = vmy_e[sel]
        ca = ca[sel]; cb = cb[sel]
        Pjhat = Pjhat[sel]
        e_len_sq = e_len_sq[sel]
        edge_opacity_e = edge_opacity_e[sel]
        ba_u_e_w = ba_u_e_w[sel]; ba_v_e_w = ba_v_e_w[sel]
        bb_u_e_w = bb_u_e_w[sel]; bb_v_e_w = bb_v_e_w[sel]
        E = len(par)

    if E == 0:
        return None

    # ================================================================
    # Step 12: Per-hit power integral reduction
    # ================================================================
    edge_hit_idx = hit_idx_f[par]  # [E] — which hit each edge belongs to

    # P_A per hit = sum(eff_opacity * Pt) over all triangles of this hit
    PA_contributions = eff_opacity_f * Pt_f  # [M_f]
    P_A_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(P_A_per_hit, hit_idx_f, PA_contributions)

    # P_A_geom per hit = sum(opacity * Pt) — for threshold check
    PA_geom_contributions = opacity_f * Pt_f  # [M_f]
    P_A_geom_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(P_A_geom_per_hit, hit_idx_f, PA_geom_contributions)

    # P_A_bare per hit = sum(Pt) — bare geometric power (no material modulation)
    P_A_bare_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(P_A_bare_per_hit, hit_idx_f, Pt_f)

    # psi_0 per hit = sum(sqrt_eff * Psi0t)
    psi0_contributions = sqrt_eff_f * Psi0t_f  # [M_f]
    psi_0_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(psi_0_per_hit, hit_idx_f, psi0_contributions)

    # psi_0_bare per hit = sum(Psi0t) — bare (no material modulation)
    psi_0_bare_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(psi_0_bare_per_hit, hit_idx_f, Psi0t_f)

    # Sigmat per hit = sum(sqrt_eff * Sigmat_per_tri)
    Sigmat_xx_per_hit = np.zeros(N, dtype=np.float64)
    Sigmat_xy_per_hit = np.zeros(N, dtype=np.float64)
    Sigmat_yy_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(Sigmat_xx_per_hit, hit_idx_f, sqrt_eff_f * Sigmat_xx_f)
    np.add.at(Sigmat_xy_per_hit, hit_idx_f, sqrt_eff_f * Sigmat_xy_f)
    np.add.at(Sigmat_yy_per_hit, hit_idx_f, sqrt_eff_f * Sigmat_yy_f)

    # Sigmat_bare per hit = sum(Sigmat_per_tri) — bare (no material modulation)
    Sigmat_xx_bare_per_hit = np.zeros(N, dtype=np.float64)
    Sigmat_xy_bare_per_hit = np.zeros(N, dtype=np.float64)
    Sigmat_yy_bare_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(Sigmat_xx_bare_per_hit, hit_idx_f, Sigmat_xx_f)
    np.add.at(Sigmat_xy_bare_per_hit, hit_idx_f, Sigmat_xy_f)
    np.add.at(Sigmat_yy_bare_per_hit, hit_idx_f, Sigmat_yy_f)

    # sum_Phat_j per hit
    sum_Phat_per_hit = np.zeros(N, dtype=np.float64)
    np.add.at(sum_Phat_per_hit, edge_hit_idx, Pjhat)

    # e_avg (power-weighted average edge length) per hit
    e_len = np.sqrt(e_len_sq)  # [E]
    e_avg_num = np.zeros(N, dtype=np.float64)
    np.add.at(e_avg_num, edge_hit_idx, Pjhat * e_len)

    # Edge count per hit
    edge_count_per_hit = np.zeros(N, dtype=np.int32)
    np.add.at(edge_count_per_hit, edge_hit_idx, 1)

    # ================================================================
    # Step 13: Per-hit finalization (P_central, P_A_hat)
    # ================================================================
    # Threshold: skip hits with P_A_geom < 1e-2
    has_edges = edge_count_per_hit > 0
    above_threshold = P_A_geom_per_hit >= 1e-2
    active_hit_mask = has_edges & above_threshold  # [N]

    active_hits = np.where(active_hit_mask)[0]  # indices into [0..N-1]
    A = len(active_hits)  # number of active apertures

    if A == 0:
        return None

    # Compute P_central and P_A_hat for active hits
    sum_Pj = sum_Phat_per_hit[active_hits]  # [A]
    e_avg_active = np.where(
        sum_Pj > 0,
        e_avg_num[active_hits] / sum_Pj,
        0.01
    )  # [A]

    sigma_xi = np.sqrt(3.0) / (k * np.maximum(e_avg_active, 1e-6))  # [A]
    inv_sigma_xi_sq = 1.0 / (sigma_xi ** 2)

    psi0_a = psi_0_per_hit[active_hits]  # [A]
    Sxx = Sigmat_xx_per_hit[active_hits]  # [A]
    Sxy = Sigmat_xy_per_hit[active_hits]
    Syy = Sigmat_yy_per_hit[active_hits]

    # Scale covariance: Sigma0 = 6k²/psi0 * Sigmat + sigma_xi^-2 * I
    scale_factor = np.where(np.abs(psi0_a) > 1e-12, 6.0 * k * k / psi0_a, 0.0)  # [A]
    S0_xx = scale_factor * Sxx + inv_sigma_xi_sq  # [A]
    S0_xy = scale_factor * Sxy  # [A]
    S0_yy = scale_factor * Syy + inv_sigma_xi_sq  # [A]

    det_S0 = np.maximum(0.0, S0_xx * S0_yy - S0_xy ** 2)  # [A]

    P_central = np.where(
        det_S0 > 0,
        k * k / (18.0 * np.pi) / np.sqrt(det_S0) * psi0_a ** 2,
        0.0,
    )  # [A]

    P_A_hat = np.maximum(0.0, P_A_per_hit[active_hits] - P_central)  # [A]

    # Bare (no material modulation) P_central and P_A_hat for normalization denominator
    psi0_bare_a = psi_0_bare_per_hit[active_hits]  # [A]
    Sxx_bare = Sigmat_xx_bare_per_hit[active_hits]
    Sxy_bare = Sigmat_xy_bare_per_hit[active_hits]
    Syy_bare = Sigmat_yy_bare_per_hit[active_hits]

    scale_factor_bare = np.where(np.abs(psi0_bare_a) > 1e-12, 6.0 * k * k / psi0_bare_a, 0.0)
    S0_xx_bare = scale_factor_bare * Sxx_bare + inv_sigma_xi_sq
    S0_xy_bare = scale_factor_bare * Sxy_bare
    S0_yy_bare = scale_factor_bare * Syy_bare + inv_sigma_xi_sq
    det_S0_bare = np.maximum(0.0, S0_xx_bare * S0_yy_bare - S0_xy_bare ** 2)
    P_central_bare = np.where(
        det_S0_bare > 0,
        k * k / (18.0 * np.pi) / np.sqrt(det_S0_bare) * psi0_bare_a ** 2,
        0.0,
    )
    P_A_hat_bare = np.maximum(0.0, P_A_bare_per_hit[active_hits] - P_central_bare)  # [A]
    P_A_bare = P_A_bare_per_hit[active_hits]  # [A] — full bare obstacle power (for normalization, matches reference)

    # Re-filter: P_A_hat_bare > 0
    has_diff = P_A_hat_bare > 0
    if not np.any(has_diff):
        return None

    active_hits = active_hits[has_diff]
    P_A_hat = P_A_hat[has_diff]
    P_A_hat_bare = P_A_hat_bare[has_diff]
    P_A_bare = P_A_bare[has_diff]
    A = len(active_hits)

    # Beta (energy borrowing) — uses bare geometric P_A_hat
    beta = np.minimum(P_A_hat_bare, beta_max)  # [A]

    # ================================================================
    # Step 14: Build FlatEdgeData (CSR format)
    # ================================================================
    # Map from original hit index to aperture index (-1 if not active)
    hit_to_ap = np.full(N, -1, dtype=np.int32)
    hit_to_ap[active_hits] = np.arange(A, dtype=np.int32)

    # Map edges to aperture indices; filter out edges for inactive hits
    edge_ap_idx = hit_to_ap[edge_hit_idx]  # [E]
    edge_active = edge_ap_idx >= 0
    if not np.all(edge_active):
        sel = np.where(edge_active)[0]
        par = par[sel]
        ex_e = ex_e[sel]; ey_e = ey_e[sel]
        vmx_e = vmx_e[sel]; vmy_e = vmy_e[sel]
        ca = ca[sel]; cb = cb[sel]
        Pjhat = Pjhat[sel]
        edge_opacity_e = edge_opacity_e[sel]
        ba_u_e_w = ba_u_e_w[sel]; ba_v_e_w = ba_v_e_w[sel]
        bb_u_e_w = bb_u_e_w[sel]; bb_v_e_w = bb_v_e_w[sel]
        edge_ap_idx = edge_ap_idx[sel]
        edge_hit_idx = edge_hit_idx[sel]
        E = len(sel)

    # Sort edges by aperture index for CSR contiguity
    sort_order = np.argsort(edge_ap_idx, kind='stable')
    edge_ap_sorted = edge_ap_idx[sort_order]
    ex_e = ex_e[sort_order]; ey_e = ey_e[sort_order]
    vmx_e = vmx_e[sort_order]; vmy_e = vmy_e[sort_order]
    ca = ca[sort_order]; cb = cb[sort_order]
    Pjhat = Pjhat[sort_order]
    edge_opacity_e = edge_opacity_e[sort_order]
    par_sorted = par[sort_order]
    ba_u_e_w = ba_u_e_w[sort_order]; ba_v_e_w = ba_v_e_w[sort_order]
    bb_u_e_w = bb_u_e_w[sort_order]; bb_v_e_w = bb_v_e_w[sort_order]

    # CSR offsets
    ap_edge_count = np.bincount(edge_ap_sorted, minlength=A).astype(np.int32)
    ap_edge_start = np.zeros(A, dtype=np.int32)
    if A > 1:
        np.cumsum(ap_edge_count[:-1], out=ap_edge_start[1:])

    # Enforce max_edges_per_hit
    if max_edges_per_hit < 256 or np.any(ap_edge_count > max_edges_per_hit):
        # Truncate edges per aperture
        keep_mask = np.ones(E, dtype=bool)
        for a in range(A):
            if ap_edge_count[a] > max_edges_per_hit:
                start = ap_edge_start[a]
                keep_mask[start + max_edges_per_hit:start + ap_edge_count[a]] = False
                ap_edge_count[a] = max_edges_per_hit
        if not np.all(keep_mask):
            sel = np.where(keep_mask)[0]
            ex_e = ex_e[sel]; ey_e = ey_e[sel]
            vmx_e = vmx_e[sel]; vmy_e = vmy_e[sel]
            ca = ca[sel]; cb = cb[sel]
            Pjhat = Pjhat[sel]
            edge_opacity_e = edge_opacity_e[sel]
            par_sorted = par_sorted[sel]
            ba_u_e_w = ba_u_e_w[sel]; ba_v_e_w = ba_v_e_w[sel]
            bb_u_e_w = bb_u_e_w[sel]; bb_v_e_w = bb_v_e_w[sel]
            E = len(sel)
            # Recompute CSR
            ap_edge_start = np.zeros(A, dtype=np.int32)
            if A > 1:
                np.cumsum(ap_edge_count[:-1], out=ap_edge_start[1:])

    # Edge midpoint barycentrics (average of vertex barycentrics)
    bary_u_e = (ba_u_e_w + bb_u_e_w) * 0.5  # [E]
    bary_v_e = (ba_v_e_w + bb_v_e_w) * 0.5

    # Per-edge face/vertex data
    edge_face_idx_e = face_idx_f[par_sorted].astype(np.int32)  # [E]
    cos_theta_e = cos_theta_f[par_sorted]  # [E]

    # Vertex indices
    mesh_faces = tri_hash.faces  # [F, 3]
    edge_vi = mesh_faces[edge_face_idx_e]  # [E, 3]

    # Jones data per edge
    jones_real_e = jr_f[par_sorted]  # [E]
    jones_imag_e = ji_f[par_sorted]
    tx_s_e = ts_f[par_sorted]
    tx_p_e = tp_f[par_sorted]
    rx_s_e = rs_f[par_sorted]
    rx_p_e = rp_f[par_sorted]

    # Per-aperture screen coordinates
    ap_tangent = tangent[active_hits].astype(np.float32)     # [A, 3]
    ap_bitangent = bitangent[active_hits].astype(np.float32)
    ap_wo_dir = wo_unit[active_hits].astype(np.float32)

    if verbose:
        print(f"  [BatchAperture] {A} active apertures, {E} edges "
              f"(from {N} hits, {M_f} front-facing pairs)")

    return (FlatEdgeData(
        edge_ex=ex_e.astype(np.float64),
        edge_ey=ey_e.astype(np.float64),
        edge_vx=vmx_e.astype(np.float64),
        edge_vy=vmy_e.astype(np.float64),
        edge_a_real=ca.real.astype(np.float64),
        edge_a_imag=ca.imag.astype(np.float64),
        edge_b_real=cb.real.astype(np.float64),
        edge_b_imag=cb.imag.astype(np.float64),
        edge_opacity=edge_opacity_e.astype(np.float64),
        edge_face_idx=edge_face_idx_e,
        edge_cos_theta=cos_theta_e.astype(np.float64),
        edge_bary_u=bary_u_e.astype(np.float64),
        edge_bary_v=bary_v_e.astype(np.float64),
        edge_vert_idx_0=edge_vi[:, 0].astype(np.int32),
        edge_vert_idx_1=edge_vi[:, 1].astype(np.int32),
        edge_vert_idx_2=edge_vi[:, 2].astype(np.int32),
        edge_jones_real=jones_real_e.astype(np.float64),
        edge_jones_imag=jones_imag_e.astype(np.float64),
        edge_tx_amp_s=tx_s_e.astype(np.float64),
        edge_tx_amp_p=tx_p_e.astype(np.float64),
        edge_rx_amp_s=rx_s_e.astype(np.float64),
        edge_rx_amp_p=rx_p_e.astype(np.float64),
        edge_ba_u=ba_u_e_w.astype(np.float64),
        edge_ba_v=ba_v_e_w.astype(np.float64),
        edge_bb_u=bb_u_e_w.astype(np.float64),
        edge_bb_v=bb_v_e_w.astype(np.float64),
        ap_edge_start=ap_edge_start,
        ap_edge_count=ap_edge_count,
        ap_P_A_hat=P_A_hat.astype(np.float64),
        ap_P_A_hat_bare=P_A_hat_bare.astype(np.float64),
        ap_P_A_bare=P_A_bare.astype(np.float64),
        ap_tangent=ap_tangent,
        ap_bitangent=ap_bitangent,
        ap_wo_dir=ap_wo_dir,
        ap_beta=beta.astype(np.float64),
        ap_hit_pos=hit_pos_all[active_hits].astype(np.float64),
        ap_to_path_indices=[np.empty(0, dtype=np.int32)] * A,  # Populated by integrator
        n_edges=E,
        n_apertures=A,
    ), active_hits)


def _batch_Sigmat(u1x, u1y, u2x, u2y, u3x, u3y, ph1, ph2, ph3):
    """
    Vectorized Sigmat computation for [M] triangles.
    Returns (Sxx[M], Sxy[M], Syy[M]).
    """
    area = np.abs(
        -u1y * u2x + u1x * u2y + u1y * u3x - u2y * u3x - u1x * u3y + u2x * u3y
    )

    # Sigma_xx
    a = area * (
        (3*ph1 + ph2 + ph3) * u1x**2
        + (ph1 + 3*ph2 + ph3) * u2x**2
        + (ph1 + 2*(ph2 + ph3)) * u2x * u3x
        + (ph1 + ph2 + 3*ph3) * u3x**2
        + u1x * ((2*(ph1 + ph2) + ph3) * u2x + (2*ph1 + ph2 + 2*ph3) * u3x)
    ) / 60.0

    # Sigma_xy
    b = area * (
        u1x * (2*(3*ph1 + ph2 + ph3)*u1y + (2*(ph1+ph2)+ph3)*u2y + (2*ph1+ph2+2*ph3)*u3y)
        + u3x * ((2*ph1+ph2+2*ph3)*u1y + (ph1+2*(ph2+ph3))*u2y + 2*(ph1+ph2+3*ph3)*u3y)
        + u2x * ((2*(ph1+ph2)+ph3)*u1y + 2*(ph1+3*ph2+ph3)*u2y + (ph1+2*(ph2+ph3))*u3y)
    ) / 120.0

    # Sigma_yy
    c = area * (
        (3*ph1 + ph2 + ph3) * u1y**2
        + (ph1 + 3*ph2 + ph3) * u2y**2
        + (ph1 + 2*(ph2 + ph3)) * u2y * u3y
        + (ph1 + ph2 + 3*ph3) * u3y**2
        + u1y * ((2*(ph1 + ph2) + ph3) * u2y + (2*ph1 + ph2 + 2*ph3) * u3y)
    ) / 60.0

    return a, b, c
