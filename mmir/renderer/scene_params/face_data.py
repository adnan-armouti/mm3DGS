"""
Face data structure for differentiable face normal computation.
"""

from dataclasses import dataclass
import numpy as np
import drjit as dr
import mitsuba as mi
from typing import Optional


@dataclass
class FaceData:
    """Face data structure for differentiable face normal computation."""
    num_faces: int
    face_indices: mi.Vector3u  # [F, 3] Triangle vertex indices


def load_face_data_from_ply(ply_path: str) -> FaceData:
    """Load face data from PLY file."""
    import trimesh

    mesh = trimesh.load(ply_path)
    faces = np.array(mesh.faces, dtype=np.uint32)

    # Convert to DrJit format [F, 3]
    face_data = FaceData(
        num_faces=len(faces),
        face_indices=mi.Vector3u(
            faces[:, 0],  # v0 indices
            faces[:, 1],  # v1 indices
            faces[:, 2]   # v2 indices
        )
    )

    return face_data


def compute_face_normals_differentiable(
    vertices: 'mi.Point3f',  # [V, 3] Vertex positions
    face_data: FaceData       # Face connectivity
) -> 'mi.Vector3f':
    """
    Compute face normals differentiably from vertex positions.

    For each triangle (v0, v1, v2):
        n = normalize((v1 - v0) x (v2 - v0))

    Args:
        vertices: Vertex positions (gradients enabled)
        face_data: Face connectivity information

    Returns:
        Face normals [F, 3] (differentiable w.r.t. vertices)
    """
    # Get vertex indices for each face
    v0_idx = face_data.face_indices.x
    v1_idx = face_data.face_indices.y
    v2_idx = face_data.face_indices.z

    # Gather vertex positions for each triangle
    v0 = mi.Point3f(
        dr.gather(mi.Float, vertices.x, v0_idx),
        dr.gather(mi.Float, vertices.y, v0_idx),
        dr.gather(mi.Float, vertices.z, v0_idx)
    )

    v1 = mi.Point3f(
        dr.gather(mi.Float, vertices.x, v1_idx),
        dr.gather(mi.Float, vertices.y, v1_idx),
        dr.gather(mi.Float, vertices.z, v1_idx)
    )

    v2 = mi.Point3f(
        dr.gather(mi.Float, vertices.x, v2_idx),
        dr.gather(mi.Float, vertices.y, v2_idx),
        dr.gather(mi.Float, vertices.z, v2_idx)
    )

    # Compute edge vectors
    e1 = v1 - v0  # Edge from v0 to v1
    e2 = v2 - v0  # Edge from v0 to v2

    # Compute face normal via cross product
    normal = dr.cross(e1, e2)

    # Normalize (with epsilon for stability)
    normal_length = dr.norm(normal)
    normal_length_safe = dr.maximum(normal_length, 1e-8)
    normal_normalized = normal / normal_length_safe

    return normal_normalized