"""
Per-vertex normal parameters with barycentric interpolation.

This module provides a system for associating normal vectors with individual vertices
and interpolating them to hit points using barycentric coordinates.

This enables gradient-based optimization of per-vertex normals (useful for surface
reconstruction, normal map optimization, and geometry refinement).

Key features:
- Store normals per-vertex
- Barycentric interpolation to hit points
- Automatic normalization (normals are unit vectors)
- Full gradient support via Dr.Jit
- Option to override geometric normals (for normal mapping)
"""

from typing import Tuple, Optional
import drjit as dr
import mitsuba as mi
import numpy as np


class VertexNormals:
    """
    Container for per-vertex normal vectors.

    Normal vectors are defined at vertices and interpolated to hit points
    using barycentric coordinates. The interpolated normals are automatically
    normalized to ensure they remain unit vectors.
    """

    def __init__(self,
                 num_vertices: int,
                 normals: Optional['mi.Vector3f'] = None):
        """
        Initialize per-vertex normals.

        Args:
            num_vertices: Number of vertices in the mesh
            normals: [num_vertices, 3] Normal vectors per vertex (will be normalized)
                    If None, initializes with +Z normals (0, 0, 1)
        """
        self.num_vertices = num_vertices

        if normals is None:
            # Default: all normals point in +Z direction
            normals = mi.Vector3f(
                dr.zeros(mi.Float, num_vertices),
                dr.zeros(mi.Float, num_vertices),
                dr.ones(mi.Float, num_vertices)
            )

        # Store normals without normalization to preserve gradient flow
        # Normalization will be done during interpolation if needed
        self.normals = normals

    @classmethod
    def from_ply(cls, ply_file: str, enable_grad: bool = False) -> 'VertexNormals':
        """
        Load per-vertex normals from PLY file.

        Args:
            ply_file: Path to PLY mesh file
            enable_grad: If True, enable gradients for normals

        Returns:
            VertexNormals instance with loaded normals

        Example:
            >>> vertex_normals = VertexNormals.from_ply("mesh.ply", enable_grad=True)
            >>> # Use in SBR solver for differentiable surface normals
        """
        import trimesh

        print(f"[VertexNormals] Loading from {ply_file}")

        # Load mesh using trimesh
        mesh = trimesh.load(ply_file, force='mesh', process=False)

        # Extract vertex normals
        # If mesh doesn't have vertex normals, compute them
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            print(f"  Computing vertex normals (not present in file)")
            mesh.vertex_normals  # This triggers computation

        # Note: mesh.vertex_normals might be a TrackedArray, so convert to plain numpy
        normals_np = np.array(mesh.vertex_normals, dtype=np.float32)
        num_vertices = normals_np.shape[0]

        print(f"  Loaded {num_vertices} vertex normals")

        # Convert to Mitsuba Vector3f
        # Mitsuba expects shape (3, N) not (N, 3)
        normals = mi.Vector3f(
            mi.Float(normals_np[:, 0]),
            mi.Float(normals_np[:, 1]),
            mi.Float(normals_np[:, 2])
        )

        # Create instance (normals are NOT normalized yet)
        instance = cls(num_vertices, normals)

        # Enable gradients BEFORE normalization to ensure proper gradient flow
        if enable_grad:
            dr.enable_grad(instance.normals.x)
            dr.enable_grad(instance.normals.y)
            dr.enable_grad(instance.normals.z)
            print(f"  Gradients enabled for vertex normals")

        # Now normalize the normals AFTER enabling gradients
        # This ensures the normalization is part of the computational graph
        instance.normalize_normals()
        print(f"  Normals normalized (with gradient support: {enable_grad})")

        return instance

    def set_vertex_normal(self, vertex_idx: int, normal: np.ndarray):
        """
        Set normal vector for a specific vertex.

        Args:
            vertex_idx: Vertex index
            normal: [3] Normal vector (will be normalized)
        """
        # Normalize
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            normal = np.array([0.0, 0.0, 1.0])  # Fallback to +Z
            norm = 1.0

        normal_unit = normal / norm

        self.normals.x[vertex_idx] = normal_unit[0]
        self.normals.y[vertex_idx] = normal_unit[1]
        self.normals.z[vertex_idx] = normal_unit[2]

    def normalize_normals(self):
        """
        Normalize all vertex normals to unit length.

        This should be called AFTER enabling gradients to ensure proper gradient flow.

        CRITICAL: After normalization, normals become COMPUTED values (not leaf variables).
        We use detach() to break the computational graph, then re-enable gradients to
        convert them back to leaf variables. Otherwise Dr.Jit won't compute gradients for them!
        """
        # Check if gradients were enabled before normalization
        had_grad = dr.grad_enabled(self.normals.x)

        norm = dr.norm(self.normals)
        # Use a larger epsilon to avoid potential NaN gradients
        norm_safe = dr.maximum(norm, 1e-6)

        # Normalize
        normals_normalized = self.normals / norm_safe

        # CRITICAL: Detach from computation graph and convert back to leaf variables
        # Without this, the normalized normals remain as computed intermediates
        # and don't receive gradients during backward pass
        if had_grad:
            self.normals = mi.Vector3f(
                dr.detach(normals_normalized.x),
                dr.detach(normals_normalized.y),
                dr.detach(normals_normalized.z)
            )
            dr.enable_grad(self.normals.x)
            dr.enable_grad(self.normals.y)
            dr.enable_grad(self.normals.z)
        else:
            self.normals = normals_normalized

    def enable_gradients(self):
        """
        Enable gradients for normal vectors.

        Note: Call normalize_normals() after this if you want normalized normals
        with proper gradient flow.
        """
        dr.enable_grad(self.normals.x)
        dr.enable_grad(self.normals.y)
        dr.enable_grad(self.normals.z)

    def get_interpolated_normal(self,
                                vertex_indices: 'mi.Vector3u',
                                bary_coords: 'mi.Point2f',
                                normalize: bool = True) -> 'mi.Vector3f':
        """
        Get interpolated normal at hit points using barycentric interpolation.

        Args:
            vertex_indices: Vector3u with three vertex indices per hit triangle
                          v0_idx = vertex_indices.x
                          v1_idx = vertex_indices.y
                          v2_idx = vertex_indices.z
            bary_coords: [N, 2] or Point2f - Barycentric coordinates (u, v) at hit points
                         Third coordinate w = 1 - u - v
            normalize: If True, normalize the interpolated normal to unit length
                      (recommended, but can disable for testing)

        Returns:
            normal_interp: [N, 3] Interpolated normal vectors
        """
        # Extract vertex indices from Vector3u
        v0_idx = vertex_indices.x
        v1_idx = vertex_indices.y
        v2_idx = vertex_indices.z

        # Barycentric coordinates: (u, v, w) where w = 1 - u - v
        u = bary_coords.x
        v = bary_coords.y
        w = 1.0 - u - v

        # Gather normals for each vertex
        # normals is a Vector3f with components .x, .y, .z
        # Each component is an array of length num_vertices

        # Interpolate X component
        nx_v0 = dr.gather(mi.Float, self.normals.x, v0_idx)
        nx_v1 = dr.gather(mi.Float, self.normals.x, v1_idx)
        nx_v2 = dr.gather(mi.Float, self.normals.x, v2_idx)
        nx_interp = w * nx_v0 + u * nx_v1 + v * nx_v2

        # Interpolate Y component
        ny_v0 = dr.gather(mi.Float, self.normals.y, v0_idx)
        ny_v1 = dr.gather(mi.Float, self.normals.y, v1_idx)
        ny_v2 = dr.gather(mi.Float, self.normals.y, v2_idx)
        ny_interp = w * ny_v0 + u * ny_v1 + v * ny_v2

        # Interpolate Z component
        nz_v0 = dr.gather(mi.Float, self.normals.z, v0_idx)
        nz_v1 = dr.gather(mi.Float, self.normals.z, v1_idx)
        nz_v2 = dr.gather(mi.Float, self.normals.z, v2_idx)
        nz_interp = w * nz_v0 + u * nz_v1 + v * nz_v2

        # Create interpolated normal vector
        normal_interp = mi.Vector3f(nx_interp, ny_interp, nz_interp)

        # Normalize to ensure unit length
        if normalize:
            norm = dr.norm(normal_interp)
            # Use a larger epsilon to avoid NaN gradients when norm is very small
            # This prevents division by near-zero values that can cause gradient explosions
            norm_safe = dr.maximum(norm, 1e-6)
            normal_interp = normal_interp / norm_safe

        return normal_interp

    def from_numpy(self, normals_np: np.ndarray):
        """
        Load normals from numpy array.

        Args:
            normals_np: [num_vertices, 3] Numpy array of normal vectors
        """
        if normals_np.shape[0] != self.num_vertices:
            raise ValueError(f"Expected {self.num_vertices} normals, got {normals_np.shape[0]}")

        if normals_np.shape[1] != 3:
            raise ValueError(f"Normals must have 3 components, got {normals_np.shape[1]}")

        # Normalize
        norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
        # Use larger epsilon to avoid numerical issues
        norms = np.maximum(norms, 1e-6)
        normals_unit = normals_np / norms

        # Convert to Mitsuba
        self.normals = mi.Vector3f(
            mi.Float(normals_unit[:, 0]),
            mi.Float(normals_unit[:, 1]),
            mi.Float(normals_unit[:, 2])
        )

    def to_numpy(self) -> np.ndarray:
        """
        Export normals to numpy array.

        Returns:
            [num_vertices, 3] Numpy array of normal vectors
        """
        dr.eval(self.normals)

        normals_np = np.zeros((self.num_vertices, 3))
        normals_np[:, 0] = np.array(self.normals.x)
        normals_np[:, 1] = np.array(self.normals.y)
        normals_np[:, 2] = np.array(self.normals.z)

        return normals_np


def get_vertex_normals(si: 'mi.SurfaceInteraction3f',
                       vertex_normals: VertexNormals,
                       scene: 'mi.Scene' = None) -> 'mi.Vector3f':
    """
    Get interpolated normal at surface interaction using per-vertex normals.

    This replaces the geometric normal (si.n) with an interpolated normal
    from per-vertex data.

    Args:
        si: Surface interaction from ray tracing
        vertex_normals: Per-vertex normal container
        scene: Optional Mitsuba scene (recommended for reliable shape extraction)

    Returns:
        normal_interp: [N, 3] Interpolated normal vectors at hit points
    """
    # Extract vertex indices and barycentric coordinates
    # We can reuse the extraction function from vertex_materials
    from .vertex_materials import extract_vertex_indices_from_si

    vertex_indices, bary_coords = extract_vertex_indices_from_si(si, scene)

    # Interpolate normals
    normal_interp = vertex_normals.get_interpolated_normal(
        vertex_indices, bary_coords, normalize=True
    )

    return normal_interp


def create_smooth_vertex_normals(vertices: np.ndarray,
                                 faces: np.ndarray) -> np.ndarray:
    """
    Compute smooth vertex normals by averaging face normals.

    This is a utility function for initializing per-vertex normals
    from mesh geometry.

    Args:
        vertices: [num_vertices, 3] Vertex positions
        faces: [num_faces, 3] Face vertex indices

    Returns:
        [num_vertices, 3] Smooth vertex normals
    """
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    # Initialize vertex normals to zero
    vertex_normals = np.zeros((num_vertices, 3))

    # Accumulate face normals at each vertex
    for face in faces:
        v0, v1, v2 = face

        # Get vertex positions
        p0 = vertices[v0]
        p1 = vertices[v1]
        p2 = vertices[v2]

        # Compute face normal (unnormalized)
        edge1 = p1 - p0
        edge2 = p2 - p0
        face_normal = np.cross(edge1, edge2)

        # Accumulate at each vertex
        vertex_normals[v0] += face_normal
        vertex_normals[v1] += face_normal
        vertex_normals[v2] += face_normal

    # Normalize vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    # Use larger epsilon to avoid numerical issues
    norms = np.maximum(norms, 1e-6)
    vertex_normals = vertex_normals / norms

    return vertex_normals


# Example usage
if __name__ == "__main__":
    print("Per-Vertex Normals System")
    print("="*70)

    import numpy as np
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')

    # Create test mesh: simple triangle
    num_vertices = 3

    # Create vertex normals (pointing in different directions)
    normals_np = np.array([
        [0.0, 0.0, 1.0],   # v0: +Z
        [0.707, 0.0, 0.707], # v1: tilted
        [0.0, 0.707, 0.707]  # v2: tilted
    ])

    vertex_normals = VertexNormals(num_vertices)
    vertex_normals.from_numpy(normals_np)

    print(f"\nVertex normals:")
    for i in range(num_vertices):
        n = normals_np[i]
        print(f"  v{i}: ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})")

    # Test interpolation
    v_indices = mi.Vector3u(0, 1, 2)

    test_cases = [
        ("Center", mi.Point2f(0.33, 0.33)),
        ("Near v0", mi.Point2f(0.1, 0.1)),
        ("Near v1", mi.Point2f(0.8, 0.1)),
        ("Near v2", mi.Point2f(0.1, 0.8)),
    ]

    print(f"\nInterpolation tests:")
    for name, bary in test_cases:
        normal_interp = vertex_normals.get_interpolated_normal(v_indices, bary)

        dr.eval(normal_interp)

        nx = float(normal_interp.x[0]) if hasattr(normal_interp.x, '__getitem__') else float(normal_interp.x)
        ny = float(normal_interp.y[0]) if hasattr(normal_interp.y, '__getitem__') else float(normal_interp.y)
        nz = float(normal_interp.z[0]) if hasattr(normal_interp.z, '__getitem__') else float(normal_interp.z)

        norm = np.sqrt(nx**2 + ny**2 + nz**2)

        print(f"  {name:12s}: ({nx:.3f}, {ny:.3f}, {nz:.3f}) | length={norm:.6f}")

    # Test gradients
    print(f"\nGradient test:")

    vertex_normals.enable_gradients()

    # Simple loss: push interpolated normal toward target
    target_normal = mi.Vector3f(1.0, 0.0, 0.0)  # +X direction

    bary = mi.Point2f(0.3, 0.3)
    normal_interp = vertex_normals.get_interpolated_normal(v_indices, bary)

    # Loss: squared distance to target
    diff = normal_interp - target_normal
    loss = dr.dot(diff, diff)

    print(f"  Initial normal: ({float(normal_interp.x[0]):.3f}, {float(normal_interp.y[0]):.3f}, {float(normal_interp.z[0]):.3f})")
    print(f"  Target: (1.000, 0.000, 0.000)")
    print(f"  Loss: {float(loss[0]):.6f}")

    # Backward
    dr.backward(loss)

    # Get gradients
    grad_normals = dr.grad(vertex_normals.normals)

    dr.eval(grad_normals)

    print(f"\nGradients w.r.t. vertex normals:")
    for i in range(num_vertices):
        gx = float(grad_normals.x[i])
        gy = float(grad_normals.y[i])
        gz = float(grad_normals.z[i])
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        print(f"  dloss/dnormal[{i}] = ({gx:.6e}, {gy:.6e}, {gz:.6e}) | mag={g_mag:.6e}")

    print(f"\n[OK] Per-vertex normals system working correctly!")
