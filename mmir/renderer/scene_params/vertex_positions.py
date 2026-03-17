"""
Per-Vertex Position Parameters with Gradient Support

This module provides infrastructure for making scene vertex positions differentiable.
Uses barycentric interpolation to recompute hit points from learnable vertex positions.

Key concept:
- Mitsuba ray tracer determines WHICH triangle is hit (discrete, non-differentiable)
- We recompute WHERE on the triangle the hit occurred using differentiable vertices
- This creates gradient flow: vertex_positions -> hit_point -> path_length & normals -> loss

Similar to vertex_normals.py and vertex_materials.py in design pattern.
"""

from typing import Optional, TYPE_CHECKING
import numpy as np
import drjit as dr
import mitsuba as mi

if TYPE_CHECKING:
    pass


class VertexPositions:
    """
    Container for learnable per-vertex positions.

    Allows scene vertices to be optimized via gradient descent.
    Gradients flow through barycentric interpolation to hit points.

    Attributes:
        num_vertices: Number of vertices in the mesh
        positions: mi.Point3f array [num_vertices] of (x, y, z) positions
    """

    def __init__(self, num_vertices: int, positions: Optional['mi.Point3f'] = None):
        """
        Initialize vertex positions.

        Args:
            num_vertices: Number of vertices in mesh
            positions: Initial positions [num_vertices]. If None, initializes at origin.
        """
        self.num_vertices = num_vertices

        if positions is None:
            # Default: all vertices at origin
            positions = mi.Point3f(
                dr.zeros(mi.Float, num_vertices),
                dr.zeros(mi.Float, num_vertices),
                dr.zeros(mi.Float, num_vertices)
            )

        # Store positions (should be differentiable if gradients needed)
        self.positions = positions

    @classmethod
    def from_ply(cls, ply_file: str, enable_grad: bool = False) -> 'VertexPositions':
        """
        Load vertex positions from PLY file.

        Args:
            ply_file: Path to PLY mesh file
            enable_grad: If True, enable gradients for positions

        Returns:
            VertexPositions instance with loaded positions

        Example:
            >>> vertex_pos = VertexPositions.from_ply("mesh.ply", enable_grad=True)
            >>> # Use in SBR solver for differentiable geometry
        """
        import trimesh

        print(f"[VertexPositions] Loading from {ply_file}")

        # Load mesh using trimesh
        mesh = trimesh.load(ply_file, force='mesh', process=False)

        # Extract vertices
        # Note: mesh.vertices might be a TrackedArray, so convert to plain numpy
        vertices_np = np.array(mesh.vertices, dtype=np.float32)
        num_vertices = vertices_np.shape[0]

        print(f"  Loaded {num_vertices} vertices")
        print(f"  Bounds: X=[{vertices_np[:, 0].min():.3f}, {vertices_np[:, 0].max():.3f}], "
              f"Y=[{vertices_np[:, 1].min():.3f}, {vertices_np[:, 1].max():.3f}], "
              f"Z=[{vertices_np[:, 2].min():.3f}, {vertices_np[:, 2].max():.3f}]")

        # Convert to Mitsuba Point3f
        # Mitsuba expects shape (3, N) not (N, 3)
        positions = mi.Point3f(
            mi.Float(vertices_np[:, 0]),
            mi.Float(vertices_np[:, 1]),
            mi.Float(vertices_np[:, 2])
        )

        # Enable gradients if requested
        if enable_grad:
            dr.enable_grad(positions.x)
            dr.enable_grad(positions.y)
            dr.enable_grad(positions.z)
            print(f"  Gradients enabled for vertex positions")

        return cls(num_vertices, positions)

    def get_interpolated_position(
        self,
        vertex_indices: 'mi.Vector3u',
        bary_coords: 'mi.Point2f'
    ) -> 'mi.Point3f':
        """
        Interpolate position using barycentric coordinates.

        For a hit on triangle (v0, v1, v2) with barycentric (u, v):
            p_hit = w*p_v0 + u*p_v1 + v*p_v2
        where w = 1 - u - v

        Gradients flow back to vertex positions.

        Args:
            vertex_indices: Triangle vertex indices [3] as (v0_idx, v1_idx, v2_idx)
            bary_coords: Barycentric coordinates [2] as (u, v)

        Returns:
            Interpolated position at hit point
        """
        # Extract vertex indices
        v0_idx = vertex_indices.x
        v1_idx = vertex_indices.y
        v2_idx = vertex_indices.z

        # Extract barycentric coordinates
        u = bary_coords.x
        v = bary_coords.y
        w = 1.0 - u - v

        # Gather vertex positions
        # X coordinates
        px_v0 = dr.gather(mi.Float, self.positions.x, v0_idx)
        px_v1 = dr.gather(mi.Float, self.positions.x, v1_idx)
        px_v2 = dr.gather(mi.Float, self.positions.x, v2_idx)

        # Y coordinates
        py_v0 = dr.gather(mi.Float, self.positions.y, v0_idx)
        py_v1 = dr.gather(mi.Float, self.positions.y, v1_idx)
        py_v2 = dr.gather(mi.Float, self.positions.y, v2_idx)

        # Z coordinates
        pz_v0 = dr.gather(mi.Float, self.positions.z, v0_idx)
        pz_v1 = dr.gather(mi.Float, self.positions.z, v1_idx)
        pz_v2 = dr.gather(mi.Float, self.positions.z, v2_idx)

        # Barycentric interpolation: p = w*p0 + u*p1 + v*p2
        px_interp = w * px_v0 + u * px_v1 + v * px_v2
        py_interp = w * py_v0 + u * py_v1 + v * py_v2
        pz_interp = w * pz_v0 + u * pz_v1 + v * pz_v2

        # Return interpolated position
        position_interp = mi.Point3f(px_interp, py_interp, pz_interp)

        return position_interp

    def get_interpolated_normal(
        self,
        vertex_indices: 'mi.Vector3u',
        bary_coords: 'mi.Point2f'
    ) -> 'mi.Vector3f':
        """
        Compute geometric normal from vertex positions.

        For triangle (v0, v1, v2):
            n = normalize((v1 - v0) x (v2 - v0))

        This provides gradient flow from vertex positions to surface normal.
        Unlike per-vertex normals (which can be independent), this normal
        is geometrically consistent with the vertex positions.

        Args:
            vertex_indices: Triangle vertex indices [3]
            bary_coords: Barycentric coordinates [2] (not used for flat normal)

        Returns:
            Geometric normal (unit vector)
        """
        # Extract vertex indices
        v0_idx = vertex_indices.x
        v1_idx = vertex_indices.y
        v2_idx = vertex_indices.z

        # Gather vertex positions
        v0 = mi.Point3f(
            dr.gather(mi.Float, self.positions.x, v0_idx),
            dr.gather(mi.Float, self.positions.y, v0_idx),
            dr.gather(mi.Float, self.positions.z, v0_idx)
        )
        v1 = mi.Point3f(
            dr.gather(mi.Float, self.positions.x, v1_idx),
            dr.gather(mi.Float, self.positions.y, v1_idx),
            dr.gather(mi.Float, self.positions.z, v1_idx)
        )
        v2 = mi.Point3f(
            dr.gather(mi.Float, self.positions.x, v2_idx),
            dr.gather(mi.Float, self.positions.y, v2_idx),
            dr.gather(mi.Float, self.positions.z, v2_idx)
        )

        # Edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Cross product to get normal
        normal = dr.cross(edge1, edge2)

        # Normalize
        norm = dr.norm(normal)
        norm_safe = dr.maximum(norm, 1e-12)
        normal_unit = normal / norm_safe

        return normal_unit


def get_vertex_positions(si: 'mi.SurfaceInteraction3f',
                        vertex_positions: VertexPositions,
                        scene: 'mi.Scene' = None) -> 'mi.Point3f':
    """
    Get interpolated position for a surface interaction.

    Extracts vertex indices and barycentric coordinates from si,
    then interpolates position from per-vertex parameters.
    For invalid rays, returns the original position.

    Args:
        si: Surface interaction from ray tracing
        vertex_positions: VertexPositions container
        scene: Optional Mitsuba scene (recommended for reliable shape extraction)

    Returns:
        Interpolated position at hit point (or original for invalid rays)
    """
    from .vertex_materials import extract_vertex_indices_from_si

    # Check which rays are valid
    valid = si.is_valid()

    # Extract vertex indices and barycentric coords
    vertex_indices, bary_coords = extract_vertex_indices_from_si(si, scene)

    # Interpolate position
    position = vertex_positions.get_interpolated_position(vertex_indices, bary_coords)

    # For invalid rays, keep the original position
    position = dr.select(valid, position, si.p)

    return position


def get_geometric_normal(si: 'mi.SurfaceInteraction3f',
                        vertex_positions: VertexPositions,
                        scene: 'mi.Scene' = None) -> 'mi.Vector3f':
    """
    Get geometric normal computed from vertex positions.

    This provides gradient flow from vertex positions to surface normal,
    which then affects BRDF evaluation.

    Args:
        si: Surface interaction from ray tracing
        vertex_positions: VertexPositions container
        scene: Optional Mitsuba scene (recommended for reliable shape extraction)

    Returns:
        Geometric normal (unit vector)
    """
    from .vertex_materials import extract_vertex_indices_from_si

    # Extract vertex indices
    vertex_indices, bary_coords = extract_vertex_indices_from_si(si, scene)

    # Compute geometric normal
    normal = vertex_positions.get_interpolated_normal(vertex_indices, bary_coords)

    return normal


def initialize_from_scene(scene: 'mi.Scene') -> VertexPositions:
    """
    Initialize vertex positions from a Mitsuba scene.

    Extracts current vertex positions from scene geometry and
    creates a VertexPositions container with gradient support.

    Args:
        scene: Mitsuba scene

    Returns:
        VertexPositions initialized from scene geometry

    Example:
        >>> scene = mi.load_dict({...})
        >>> vertex_pos = initialize_from_scene(scene)
        >>> dr.enable_grad(vertex_pos.positions)
    """
    # Try to extract vertices from scene shapes
    vertices_list = []

    # Iterate through scene shapes
    for shape in scene.shapes():
        # Try to get vertex data from shape
        try:
            # Method 1: Try traverse
            params = mi.traverse(shape)
            if 'vertex_positions' in params:
                verts = params['vertex_positions']
                vertices_list.append(verts)
            elif 'vertices' in params:
                verts = params['vertices']
                vertices_list.append(verts)
        except:
            pass

    if not vertices_list:
        # Fallback: create dummy vertices
        print("Warning: Could not extract vertices from scene. Creating dummy vertex positions.")
        num_vertices = 3  # Minimum for one triangle
        positions = mi.Point3f(
            dr.zeros(mi.Float, num_vertices),
            dr.zeros(mi.Float, num_vertices),
            dr.ones(mi.Float, num_vertices)  # Z = 1
        )
        return VertexPositions(num_vertices, positions)

    # Concatenate all vertices
    # For now, use first shape only (can be extended to handle multiple shapes)
    verts_flat = vertices_list[0]

    # Convert to Point3f
    # Vertices are usually stored as flat array [x0, y0, z0, x1, y1, z1, ...]
    num_vertices = len(verts_flat) // 3

    x_coords = verts_flat[0::3]
    y_coords = verts_flat[1::3]
    z_coords = verts_flat[2::3]

    positions = mi.Point3f(x_coords, y_coords, z_coords)

    return VertexPositions(num_vertices, positions)


def create_movable_wall_scene(wall_position: 'mi.Float',
                               material: dict = None) -> 'mi.Scene':
    """
    Create a simple test scene with a movable wall.

    The wall is a vertical rectangle that can be moved along the X-axis.
    This is useful for testing vertex position gradients.

    Args:
        wall_position: X-coordinate of wall center (Dr.Jit Float for gradients)
        material: Material dictionary for wall (default: conductor)

    Returns:
        Mitsuba scene with movable wall

    Example:
        >>> wall_x = mi.Float(5.0)
        >>> dr.enable_grad(wall_x)
        >>> scene = create_movable_wall_scene(wall_x)
        >>> # Ray trace, compute loss, backward
        >>> dr.backward(loss)
        >>> print(dr.grad(wall_x))  # Gradient w.r.t. wall position
    """
    # Default material
    if material is None:
        material = {
            'type': 'conductor',
            'material': 'Al'  # Aluminum
        }

    # Evaluate wall position for scene creation
    # (Scene creation is not differentiable, but we'll recompute hit points)
    wall_x_val = float(wall_position) if not isinstance(wall_position, float) else wall_position

    scene_dict = {
        'type': 'scene',
        'wall': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f.translate([wall_x_val, 0.0, 2.5])
                      @ mi.ScalarTransform4f.rotate([0, 1, 0], -90)  # Face -X direction
                      @ mi.ScalarTransform4f.scale([5.0, 5.0, 1.0]),  # 5m x 5m
            'bsdf': material
        }
    }

    return mi.load_dict(scene_dict)
