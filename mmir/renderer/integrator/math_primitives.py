"""
Differentiable ray-triangle intersection and vertex gathering primitives.

Used for end-to-end AD through geometry in multibounce path tracing.
"""

import drjit as dr
import mitsuba as mi
import numpy as np


def diff_ray_triangle_intersect(ray_o, ray_d, v0, v1, v2):
    """
    Fully differentiable Moller-Trumbore ray-triangle intersection.

    Given AD-attached ray (origin, direction) and AD-attached triangle vertices,
    computes AD-attached intersection parameters (t, u, v) and hit position.

    Gradients flow through:
    - ray_o -> (t, u, v) -> hit_p   (how moving the ray origin changes the hit)
    - ray_d -> (t, u, v) -> hit_p   (how changing direction changes the hit)
    - v0/v1/v2 -> (t, u, v) -> hit_p (how moving vertices changes the hit)

    This is the key enabler for full-chain multibounce gradients:
    at bounce b, ray_o = hit_{b-1} (AD-attached), so
    d(hit_b)/d(hit_{b-1}) is properly captured through (t_b, u_b, v_b).
    """
    e1 = v1 - v0
    e2 = v2 - v0

    pvec = dr.cross(ray_d, e2)
    det = dr.dot(e1, pvec)
    inv_det = mi.Float(1.0) / dr.select(dr.abs(det) > 1e-12, det, mi.Float(1e-12))

    tvec = ray_o - v0
    u = dr.dot(tvec, pvec) * inv_det

    qvec = dr.cross(tvec, e1)
    v = dr.dot(ray_d, qvec) * inv_det

    t = dr.dot(e2, qvec) * inv_det

    # Hit position via parametric form (AD through ray_o, ray_d, t)
    hit_p = mi.Point3f(
        ray_o.x + t * ray_d.x,
        ray_o.y + t * ray_d.y,
        ray_o.z + t * ray_d.z,
    )

    return t, u, v, hit_p


def get_ad_triangle_vertices(mesh, prim_ids, vertex_positions_buffer, active):
    """
    Get AD-attached vertex positions for triangles identified by prim_ids.

    Args:
        mesh: Mitsuba mesh shape
        prim_ids: mi.UInt32 [n] triangle primitive IDs (non-diff, from BVH)
        vertex_positions_buffer: mi.Float [n_verts * 3] AD-attached interleaved positions
        active: mi.Bool [n] mask

    Returns:
        v0, v1, v2: mi.Point3f [n] AD-attached triangle vertex positions
        vi0, vi1, vi2: mi.UInt32 [n] vertex indices
    """
    face_idx = mesh.face_indices(prim_ids, active)
    vi0, vi1, vi2 = face_idx[0], face_idx[1], face_idx[2]

    vp = vertex_positions_buffer
    v0 = mi.Point3f(
        dr.gather(mi.Float, vp, vi0 * 3 + 0, active),
        dr.gather(mi.Float, vp, vi0 * 3 + 1, active),
        dr.gather(mi.Float, vp, vi0 * 3 + 2, active),
    )
    v1 = mi.Point3f(
        dr.gather(mi.Float, vp, vi1 * 3 + 0, active),
        dr.gather(mi.Float, vp, vi1 * 3 + 1, active),
        dr.gather(mi.Float, vp, vi1 * 3 + 2, active),
    )
    v2 = mi.Point3f(
        dr.gather(mi.Float, vp, vi2 * 3 + 0, active),
        dr.gather(mi.Float, vp, vi2 * 3 + 1, active),
        dr.gather(mi.Float, vp, vi2 * 3 + 2, active),
    )
    return v0, v1, v2, vi0, vi1, vi2
