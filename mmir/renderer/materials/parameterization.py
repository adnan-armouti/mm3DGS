"""
Material parameterization strategies for differentiable rendering.

Provides a strategy pattern for switching between per-triangle (piecewise-constant)
and per-vertex (barycentric-interpolated) material parameterization.

Per-triangle: One material value per face. Gathered via dr.gather(params, prim_id).
Per-vertex:   One material value per vertex. Interpolated at hit points using
              barycentric coordinates: param(P) = w*param[v0] + u*param[v1] + v*param[v2].
              C0-continuous across the mesh, fully differentiable through DrJit AD.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import drjit as dr
import mitsuba as mi

if TYPE_CHECKING:
    from ..integrator import CachedGeometry


class MaterialParameterization(ABC):
    """
    Abstract base class for material parameterization strategies.

    Encapsulates how material parameters are stored (per-triangle vs per-vertex)
    and how they are gathered/interpolated for each hit point.
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameter entries (n_triangles or n_vertices)."""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """'per_triangle' or 'per_vertex'."""
        ...

    @abstractmethod
    def gather(
        self,
        physics_params: List['mi.Float'],
        cached_geom: 'CachedGeometry',
    ) -> List['mi.Float']:
        """
        Gather/interpolate material parameters for each hit.

        Args:
            physics_params: N DrJit Float arrays, each of length self.n_params.
            cached_geom: Cached geometry with hit indices + barycentrics.

        Returns:
            N DrJit Float arrays, each of length cached_geom.n_total (per-hit values).
        """
        ...

    def create_drjit_raw_params(self, raw_params_np: np.ndarray) -> List['mi.Float']:
        """Create grad-enabled DrJit arrays from numpy raw_params."""
        params = []
        for col in range(raw_params_np.shape[1]):
            p = mi.Float(raw_params_np[:, col].astype(np.float32))
            dr.enable_grad(p)
            params.append(p)
        return params

    def create_optimizer_state(self, raw_params_np: np.ndarray) -> dict:
        """Create Adam optimizer state arrays matching raw_params shape."""
        n = self.n_params
        n_cols = raw_params_np.shape[1]
        return {
            'adam_m': [np.zeros(n, dtype=np.float32) for _ in range(n_cols)],
            'adam_v': [np.zeros(n, dtype=np.float32) for _ in range(n_cols)],
        }


class PerTriangleParameterization(MaterialParameterization):
    """Per-triangle (piecewise-constant) material parameterization."""

    def __init__(self, n_triangles: int):
        self._n_triangles = n_triangles

    @property
    def n_params(self) -> int:
        return self._n_triangles

    @property
    def mode(self) -> str:
        return 'per_triangle'

    def gather(self, physics_params, cached_geom):
        active = cached_geom.active
        return [
            dr.gather(mi.Float, physics_params[i], cached_geom.hit_prim_ids, active)
            for i in range(len(physics_params))
        ]


class PerVertexParameterization(MaterialParameterization):
    """
    Per-vertex material parameterization with differentiable barycentric interpolation.

    Material value at hit point P with barycentrics (u, v, w=1-u-v) on triangle
    with vertex indices (i0, i1, i2):

        param(P) = w * param[i0] + u * param[i1] + v * param[i2]

    C0-continuous across the mesh and fully differentiable through DrJit AD.
    Gradients flow back to all 3 vertices of each hit triangle, weighted by
    barycentric coordinates.

    Barycentric convention (Mitsuba 3):
        si.uv.x = u (weight for vertex 1)
        si.uv.y = v (weight for vertex 2)
        w = 1 - u - v (weight for vertex 0)
    """

    def __init__(self, n_vertices: int, faces: Optional[np.ndarray] = None):
        self._n_vertices = n_vertices
        # Mesh faces array (n_triangles, 3) int32 — needed for vertex->triangle
        # conversion in the diffraction pipeline (edges reference face indices).
        self._faces = faces.astype(np.int32) if faces is not None else None

    @property
    def n_params(self) -> int:
        return self._n_vertices

    @property
    def mode(self) -> str:
        return 'per_vertex'

    def gather(self, physics_params, cached_geom):
        active = cached_geom.active
        # Barycentric coordinates
        u = cached_geom.hit_bary_u   # weight for vertex 1
        v = cached_geom.hit_bary_v   # weight for vertex 2
        w = mi.Float(1.0) - u - v    # weight for vertex 0

        # Vertex indices for each hit's triangle
        vi0 = cached_geom.hit_vertex_ids_0  # UInt32 [n_total]
        vi1 = cached_geom.hit_vertex_ids_1  # UInt32 [n_total]
        vi2 = cached_geom.hit_vertex_ids_2  # UInt32 [n_total]

        result = []
        for i in range(len(physics_params)):
            p0 = dr.gather(mi.Float, physics_params[i], vi0, active)
            p1 = dr.gather(mi.Float, physics_params[i], vi1, active)
            p2 = dr.gather(mi.Float, physics_params[i], vi2, active)
            # Differentiable barycentric interpolation
            interp = w * p0 + u * p1 + v * p2
            result.append(interp)
        return result

    def vertex_to_triangle_drjit(
        self,
        vertex_params: List['mi.Float'],
    ) -> List['mi.Float']:
        """
        Convert per-vertex material params to per-triangle by centroid averaging (DrJit).

        Differentiable: gradients flow back to all 3 vertices of each triangle,
        weighted equally (1/3 each).

        Used by the diffraction pipeline in Phase B, where edge_face_idx references
        triangle indices but physics_params are per-vertex.

        Args:
            vertex_params: N DrJit Float arrays, each of length n_vertices.

        Returns:
            N DrJit Float arrays, each of length n_triangles.

        Raises:
            ValueError: If faces array was not provided at construction time.
        """
        if self._faces is None:
            raise ValueError(
                "PerVertexParameterization.vertex_to_triangle_drjit() requires "
                "faces array. Pass faces= to the constructor."
            )
        fi0 = mi.UInt32(self._faces[:, 0])
        fi1 = mi.UInt32(self._faces[:, 1])
        fi2 = mi.UInt32(self._faces[:, 2])
        third = mi.Float(1.0 / 3.0)
        result = []
        for p in vertex_params:
            v0 = dr.gather(mi.Float, p, fi0)
            v1 = dr.gather(mi.Float, p, fi1)
            v2 = dr.gather(mi.Float, p, fi2)
            result.append((v0 + v1 + v2) * third)
        return result


def vertex_to_triangle_materials(vertex_params: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Convert per-vertex materials to per-triangle by centroid averaging.

    Used for the non-differentiable Phase A render path which expects
    triangle_materials (per-face).

    Args:
        vertex_params: (n_vertices, n_cols) material parameters per vertex.
        faces: (n_triangles, 3) triangle vertex indices.

    Returns:
        (n_triangles, n_cols) per-triangle materials (average of 3 vertices).
    """
    v0 = vertex_params[faces[:, 0]]  # (n_triangles, n_cols)
    v1 = vertex_params[faces[:, 1]]
    v2 = vertex_params[faces[:, 2]]
    return (v0 + v1 + v2) / 3.0
