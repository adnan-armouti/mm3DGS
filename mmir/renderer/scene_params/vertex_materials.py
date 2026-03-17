"""
Per-vertex material parameters with barycentric interpolation.

This module provides a system for associating material properties (conductivity,
permittivity, roughness) with individual vertices and interpolating them to
hit points using barycentric coordinates.

This enables gradient-based optimization of per-vertex material parameters.
"""

from typing import Tuple, Optional
import drjit as dr
import mitsuba as mi
import numpy as np


class VertexMaterialParams:
    """
    Container for per-vertex material parameters.

    Material parameters are defined at vertices and interpolated to hit points
    using barycentric coordinates from ray-triangle intersections.
    """

    def __init__(self,
                 num_vertices: int,
                 alpha: Optional['mi.Float'] = None,
                 eta: Optional['mi.Float'] = None,
                 kappa: Optional['mi.Float'] = None,
                 diffuse_albedo: Optional['mi.Float'] = None):
        """
        Initialize per-vertex material parameters.

        Args:
            num_vertices: Number of vertices in the mesh
            alpha: [num_vertices] GGX roughness per vertex (default: 0.1)
            eta: [num_vertices] Real part of IOR per vertex (default: 12.0 for steel)
            kappa: [num_vertices] Imaginary part of IOR per vertex (default: 85.0)
            diffuse_albedo: [num_vertices] Diffuse reflectance per vertex (default: 0.0)
        """
        self.num_vertices = num_vertices

        # Initialize with defaults if not provided
        if alpha is None:
            alpha = dr.full(mi.Float, 0.1, num_vertices)
        if eta is None:
            eta = dr.full(mi.Float, 12.0, num_vertices)  # Steel-like for mmWave
        if kappa is None:
            kappa = dr.full(mi.Float, 85.0, num_vertices)
        if diffuse_albedo is None:
            diffuse_albedo = dr.full(mi.Float, 0.0, num_vertices)

        self.alpha = alpha
        self.eta = eta
        self.kappa = kappa
        self.diffuse_albedo = diffuse_albedo

    def set_vertex_params(self, vertex_idx: int, alpha: float, eta: float,
                         kappa: float, diffuse_albedo: float):
        """
        Set material parameters for a specific vertex.

        Args:
            vertex_idx: Vertex index
            alpha: Roughness
            eta: Real IOR
            kappa: Imaginary IOR
            diffuse_albedo: Diffuse albedo
        """
        self.alpha[vertex_idx] = alpha
        self.eta[vertex_idx] = eta
        self.kappa[vertex_idx] = kappa
        self.diffuse_albedo[vertex_idx] = diffuse_albedo

    def enable_gradients(self, alpha: bool = True, eta: bool = True,
                        kappa: bool = True, diffuse_albedo: bool = True):
        """
        Enable gradients for material parameters.

        Args:
            alpha: Enable gradients for roughness
            eta: Enable gradients for real IOR
            kappa: Enable gradients for imaginary IOR
            diffuse_albedo: Enable gradients for diffuse albedo
        """
        if alpha:
            dr.enable_grad(self.alpha)
        if eta:
            dr.enable_grad(self.eta)
        if kappa:
            dr.enable_grad(self.kappa)
        if diffuse_albedo:
            dr.enable_grad(self.diffuse_albedo)

    def attach_to_scene(self, scene: 'mi.Scene', mesh_name: str = 'mesh'):
        """
        Attach per-vertex material parameters to Mitsuba scene as mesh attributes.

        This enables Mitsuba's mesh_attribute texture system to drive BSDF parameters
        using per-vertex values with automatic barycentric interpolation.

        Args:
            scene: Mitsuba scene object
            mesh_name: Name of the mesh shape in the scene (default: 'mesh')

        Returns:
            params: Traversable parameters dictionary with references to attributes

        Example:
            >>> materials = VertexMaterialParams.from_random(1000, "metal", enable_grad=True)
            >>> scene = mi.load_dict(scene_dict)
            >>> params = materials.attach_to_scene(scene)
            >>> # Now mesh_attribute textures can reference 'alpha', 'eta', 'kappa', 'diffuse'
        """
        params = mi.traverse(scene)

        # Attach vertex attributes (these become available as textures)
        params[f'{mesh_name}.vertex_attributes.alpha'] = self.alpha
        params[f'{mesh_name}.vertex_attributes.eta'] = self.eta
        params[f'{mesh_name}.vertex_attributes.kappa'] = self.kappa
        params[f'{mesh_name}.vertex_attributes.diffuse'] = self.diffuse_albedo

        # Update scene
        params.update()

        print(f"[VertexMaterialParams] Attached {self.num_vertices} per-vertex parameters to scene")
        print(f"  Mesh attributes: {mesh_name}.vertex_attributes.{{alpha, eta, kappa, diffuse}}")

        return params

    def get_gradients(self):
        """
        Extract gradients after backward pass.

        Returns:
            dict: Gradients for each material parameter (None if gradients not enabled)

        Example:
            >>> # After forward + backward pass
            >>> grads = materials.get_gradients()
            >>> print(f"Alpha gradient norm: {np.linalg.norm(grads['alpha'])}")
        """
        return {
            'alpha': dr.grad(self.alpha) if dr.grad_enabled(self.alpha) else None,
            'eta': dr.grad(self.eta) if dr.grad_enabled(self.eta) else None,
            'kappa': dr.grad(self.kappa) if dr.grad_enabled(self.kappa) else None,
            'diffuse_albedo': dr.grad(self.diffuse_albedo) if dr.grad_enabled(self.diffuse_albedo) else None
        }

    def attach_to_scene_with_blend(self, scene: 'mi.Scene', mesh_name: str = 'mesh', cos_theta_i: float = 0.5):
        """
        Attach per-vertex materials with automatic specular/diffuse blending.

        Computes per-vertex blend weight based on Fresnel reflectance + roughness,
        replacing the exp(-2*alpha) heuristic with physics-based mixing.

        Args:
            scene: Mitsuba scene object
            mesh_name: Name of the mesh shape in the scene
            cos_theta_i: Typical incident angle cosine for Fresnel eval (default: 0.5 = 60deg)

        Returns:
            params: Traversable parameters with spec_weight added

        Example:
            >>> materials = VertexMaterialParams.from_random(1000, "metal", enable_grad=True)
            >>> scene = mi.load_dict(scene_dict_with_blendbsdf)
            >>> params = materials.attach_to_scene_with_blend(scene)
        """
        from .interactions import fresnel_conductor_complex

        # Compute blend weight: spec_weight in [0,1] where 1=fully specular, 0=fully diffuse
        rs_r, rs_i, rp_r, rp_i = fresnel_conductor_complex(
            dr.full(mi.Float, cos_theta_i, self.num_vertices),
            self.eta, self.kappa
        )
        F_avg = 0.5 * (rs_r**2 + rs_i**2 + rp_r**2 + rp_i**2)
        roughness_factor = dr.exp(-2.0 * self.alpha)
        spec_weight = dr.clamp(F_avg * roughness_factor, 0.0, 1.0)

        # Attach to scene
        params = mi.traverse(scene)
        params[f'{mesh_name}.vertex_attributes.alpha'] = self.alpha
        params[f'{mesh_name}.vertex_attributes.eta'] = self.eta
        params[f'{mesh_name}.vertex_attributes.kappa'] = self.kappa
        params[f'{mesh_name}.vertex_attributes.diffuse'] = self.diffuse_albedo
        params[f'{mesh_name}.vertex_attributes.spec_weight'] = spec_weight
        params.update()

        print(f"[VertexMaterialParams] Attached {self.num_vertices} per-vertex parameters with blend weights")
        print(f"  Specular weight range: [{float(dr.min(spec_weight)[0]):.3f}, {float(dr.max(spec_weight)[0]):.3f}]")

        return params

    @classmethod
    def from_random(cls,
                   num_vertices: int,
                   material_type: str = "metal",
                   enable_grad: bool = False,
                   seed: int = 42) -> 'VertexMaterialParams':
        """
        Initialize with random but physically plausible material parameters.

        Args:
            num_vertices: Number of vertices
            material_type: "metal", "dielectric", "random", or "rough_dielectric"
            enable_grad: If True, enable gradients for all parameters
            seed: Random seed for reproducibility

        Returns:
            VertexMaterialParams with random initialization

        Material parameter ranges for 77 GHz mmWave:

        Metal (Conductors):
            - Roughness (alpha): 0.05 - 0.3
            - Real IOR (eta): 10.0 - 50.0
            - Imaginary IOR (kappa): 50.0 - 200.0
            - Diffuse albedo: 0.0 - 0.1
            - Typical: 100% specular interactions

        Dielectric:
            - Roughness (alpha): 0.1 - 0.5
            - Real IOR (eta): 1.5 - 4.0
            - Imaginary IOR (kappa): 0.0 - 1.0
            - Diffuse albedo: 0.2 - 0.8
            - Typical: Mix of specular and diffuse

        Random (Heterogeneous):
            - Roughness (alpha): 0.05 - 0.5 (full range)
            - Real IOR (eta): 1.5 - 50.0 (dielectrics to metals)
            - Imaginary IOR (kappa): 0.0 - 200.0 (non-absorbing to highly conductive)
            - Diffuse albedo: 0.0 - 0.8 (full range)
            - Each vertex gets independent random values (not uniform across mesh)
            - Typical: Mix of specular and diffuse

        Rough Dielectric (High Diffuse):
            - Roughness (alpha): 0.3 - 0.7 (VERY rough)
            - Real IOR (eta): 1.5 - 4.0 (dielectric)
            - Imaginary IOR (kappa): 0.0 - 0.5 (minimal absorption)
            - Diffuse albedo: 0.5 - 0.9 (HIGH diffuse reflectance)
            - Typical: Predominantly diffuse interactions (>>50%)

        Example:
            >>> materials = VertexMaterialParams.from_random(1000, "metal", enable_grad=True)
            >>> materials = VertexMaterialParams.from_random(1000, "random", enable_grad=True)
        """
        print(f"[VertexMaterialParams] Initializing random {material_type} parameters for {num_vertices} vertices")

        # Set random seed
        np.random.seed(seed)

        if material_type == "metal":
            # Metal parameters (77 GHz)
            alpha_np = np.random.uniform(0.05, 0.3, num_vertices).astype(np.float32)
            eta_np = np.random.uniform(10.0, 50.0, num_vertices).astype(np.float32)
            kappa_np = np.random.uniform(50.0, 200.0, num_vertices).astype(np.float32)
            diffuse_np = np.random.uniform(0.0, 0.1, num_vertices).astype(np.float32)

            print(f"  Metal ranges:")
            print(f"    Alpha (roughness): [{alpha_np.min():.3f}, {alpha_np.max():.3f}]")
            print(f"    Eta (real IOR): [{eta_np.min():.1f}, {eta_np.max():.1f}]")
            print(f"    Kappa (conductivity): [{kappa_np.min():.1f}, {kappa_np.max():.1f}]")
            print(f"    Diffuse albedo: [{diffuse_np.min():.3f}, {diffuse_np.max():.3f}]")

        elif material_type == "dielectric":
            # Dielectric parameters (77 GHz)
            alpha_np = np.random.uniform(0.1, 0.5, num_vertices).astype(np.float32)
            eta_np = np.random.uniform(1.5, 4.0, num_vertices).astype(np.float32)
            kappa_np = np.random.uniform(0.0, 1.0, num_vertices).astype(np.float32)
            diffuse_np = np.random.uniform(0.2, 0.8, num_vertices).astype(np.float32)

            print(f"  Dielectric ranges:")
            print(f"    Alpha (roughness): [{alpha_np.min():.3f}, {alpha_np.max():.3f}]")
            print(f"    Eta (real IOR): [{eta_np.min():.2f}, {eta_np.max():.2f}]")
            print(f"    Kappa (extinction): [{kappa_np.min():.3f}, {kappa_np.max():.3f}]")
            print(f"    Diffuse albedo: [{diffuse_np.min():.3f}, {diffuse_np.max():.3f}]")

        elif material_type == "random":
            # Completely random parameters spanning full physical range
            # Each vertex gets independent random values from the union of metal and dielectric ranges
            # This creates a heterogeneous material distribution
            alpha_np = np.random.uniform(0.05, 0.5, num_vertices).astype(np.float32)
            eta_np = np.random.uniform(1.5, 50.0, num_vertices).astype(np.float32)
            kappa_np = np.random.uniform(0.0, 200.0, num_vertices).astype(np.float32)
            diffuse_np = np.random.uniform(0.0, 0.8, num_vertices).astype(np.float32)

            print(f"  Random (heterogeneous) ranges:")
            print(f"    Alpha (roughness): [{alpha_np.min():.3f}, {alpha_np.max():.3f}]")
            print(f"    Eta (real IOR): [{eta_np.min():.2f}, {eta_np.max():.2f}]")
            print(f"    Kappa (conductivity/extinction): [{kappa_np.min():.2f}, {kappa_np.max():.2f}]")
            print(f"    Diffuse albedo: [{diffuse_np.min():.3f}, {diffuse_np.max():.3f}]")
            print(f"  Note: Each vertex has independent random parameters (not uniform across mesh)")

        elif material_type == "rough_dielectric":
            # Rough dielectric with HIGH diffuse reflectance
            # Designed to produce significantly more diffuse interactions than specular
            # Key insight: High roughness + high diffuse albedo + low conductivity = mostly diffuse
            alpha_np = np.random.uniform(0.3, 0.7, num_vertices).astype(np.float32)
            eta_np = np.random.uniform(1.5, 4.0, num_vertices).astype(np.float32)
            kappa_np = np.random.uniform(0.0, 0.5, num_vertices).astype(np.float32)
            diffuse_np = np.random.uniform(0.5, 0.9, num_vertices).astype(np.float32)  # HIGH diffuse albedo

            print(f"  Rough Dielectric (high diffuse) ranges:")
            print(f"    Alpha (roughness): [{alpha_np.min():.3f}, {alpha_np.max():.3f}] (HIGH - more diffuse)")
            print(f"    Eta (real IOR): [{eta_np.min():.2f}, {eta_np.max():.2f}] (dielectric)")
            print(f"    Kappa (extinction): [{kappa_np.min():.3f}, {kappa_np.max():.3f}] (LOW - dielectric)")
            print(f"    Diffuse albedo: [{diffuse_np.min():.3f}, {diffuse_np.max():.3f}] (HIGH - mostly diffuse)")
            print(f"  This material type should produce significantly more diffuse interactions")

        else:
            raise ValueError(f"Unknown material_type: {material_type}. Use 'metal', 'dielectric', 'random', or 'rough_dielectric'")

        # Convert to Dr.Jit
        alpha = mi.Float(alpha_np)
        eta = mi.Float(eta_np)
        kappa = mi.Float(kappa_np)
        diffuse_albedo = mi.Float(diffuse_np)

        # Create instance
        instance = cls(num_vertices, alpha, eta, kappa, diffuse_albedo)

        # Enable gradients if requested
        if enable_grad:
            instance.enable_gradients(alpha=True, eta=True, kappa=True, diffuse_albedo=True)
            print(f"  Gradients enabled for material parameters (alpha, eta, kappa, diffuse_albedo)")

        return instance

    @classmethod
    def from_uniform(cls,
                     num_vertices: int,
                     alpha: float = 0.1,
                     eta: float = 12.0,
                     kappa: float = 85.0,
                     diffuse_albedo: float = 0.0,
                     enable_grad: bool = False) -> 'VertexMaterialParams':
        """
        Initialize with uniform material parameters across all vertices.

        All vertices receive the same material values. This is useful for:
        - Diagnostic testing (e.g., perfect mirror with alpha=0)
        - Homogeneous material scenes
        - Baseline comparisons

        Args:
            num_vertices: Number of vertices
            alpha: GGX roughness (0=mirror, 0.1=typical, 1=rough)
            eta: Real part of complex refractive index
            kappa: Imaginary part (extinction coefficient)
            diffuse_albedo: Diffuse reflectance [0,1]
            enable_grad: If True, enable gradients for optimization

        Returns:
            VertexMaterialParams with uniform values

        Note on alpha=0 (Perfect Mirror):
            When alpha=0, the coherence_gate() in ConductorBSDF returns 0,
            triggering delta mirror reflection (no microfacet sampling).
            This is useful for diagnostic testing of pure specular reflection.

        Example:
            >>> # Perfect mirror for diagnostics
            >>> materials = VertexMaterialParams.from_uniform(1000, alpha=0.0)
            >>> # Typical metal
            >>> materials = VertexMaterialParams.from_uniform(1000, alpha=0.15, eta=28.2, kappa=138.5)
        """
        print(f"[VertexMaterialParams] Initializing uniform parameters for {num_vertices} vertices")
        print(f"  Alpha (roughness): {alpha}")
        print(f"  Eta (real IOR): {eta}")
        print(f"  Kappa (extinction): {kappa}")
        print(f"  Diffuse albedo: {diffuse_albedo}")

        # Create uniform arrays
        alpha_arr = dr.full(mi.Float, alpha, num_vertices)
        eta_arr = dr.full(mi.Float, eta, num_vertices)
        kappa_arr = dr.full(mi.Float, kappa, num_vertices)
        diffuse_arr = dr.full(mi.Float, diffuse_albedo, num_vertices)

        # Create instance
        instance = cls(num_vertices, alpha_arr, eta_arr, kappa_arr, diffuse_arr)

        # Enable gradients if requested
        if enable_grad:
            instance.enable_gradients(alpha=True, eta=True, kappa=True, diffuse_albedo=True)
            print(f"  Gradients enabled for material parameters")

        return instance

    def get_interpolated_params(self,
                                vertex_indices: 'mi.Vector3u',
                                bary_coords: 'mi.Point2f') -> Tuple['mi.Float', 'mi.Float', 'mi.Float', 'mi.Float']:
        """
        Get material parameters at hit points using barycentric interpolation.

        Args:
            vertex_indices: Vector3u with three vertex indices per hit triangle
                          v0_idx = vertex_indices.x
                          v1_idx = vertex_indices.y
                          v2_idx = vertex_indices.z
            bary_coords: [N, 2] or Point2f - Barycentric coordinates (u, v) at hit points
                         Third coordinate w = 1 - u - v

        Returns:
            (alpha, eta, kappa, diffuse_albedo): [N] Interpolated material parameters
        """
        # Extract vertex indices from Vector3u
        # vertex_indices is Vector3u where:
        #   .x = first vertex index
        #   .y = second vertex index
        #   .z = third vertex index
        v0_idx = vertex_indices.x
        v1_idx = vertex_indices.y
        v2_idx = vertex_indices.z

        # Barycentric coordinates: (u, v, w) where w = 1 - u - v
        # Standard convention: vertex 0 has weight w, vertex 1 has weight u, vertex 2 has weight v
        u = bary_coords.x
        v = bary_coords.y
        w = 1.0 - u - v

        # Interpolate alpha (roughness)
        alpha_v0 = dr.gather(mi.Float, self.alpha, v0_idx)
        alpha_v1 = dr.gather(mi.Float, self.alpha, v1_idx)
        alpha_v2 = dr.gather(mi.Float, self.alpha, v2_idx)
        alpha_interp = w * alpha_v0 + u * alpha_v1 + v * alpha_v2

        # Interpolate eta (real IOR)
        eta_v0 = dr.gather(mi.Float, self.eta, v0_idx)
        eta_v1 = dr.gather(mi.Float, self.eta, v1_idx)
        eta_v2 = dr.gather(mi.Float, self.eta, v2_idx)
        eta_interp = w * eta_v0 + u * eta_v1 + v * eta_v2

        # Interpolate kappa (imaginary IOR)
        kappa_v0 = dr.gather(mi.Float, self.kappa, v0_idx)
        kappa_v1 = dr.gather(mi.Float, self.kappa, v1_idx)
        kappa_v2 = dr.gather(mi.Float, self.kappa, v2_idx)
        kappa_interp = w * kappa_v0 + u * kappa_v1 + v * kappa_v2

        # Interpolate diffuse albedo
        albedo_v0 = dr.gather(mi.Float, self.diffuse_albedo, v0_idx)
        albedo_v1 = dr.gather(mi.Float, self.diffuse_albedo, v1_idx)
        albedo_v2 = dr.gather(mi.Float, self.diffuse_albedo, v2_idx)
        albedo_interp = w * albedo_v0 + u * albedo_v1 + v * albedo_v2

        # Clamp to valid ranges
        alpha_interp = dr.clamp(alpha_interp, 0.001, 1.0)  # Roughness: (0, 1]
        eta_interp = dr.maximum(eta_interp, 0.1)  # Real IOR: positive
        kappa_interp = dr.maximum(kappa_interp, 0.0)  # Extinction: non-negative
        albedo_interp = dr.clamp(albedo_interp, 0.0, 1.0)  # Albedo: [0, 1]

        return alpha_interp, eta_interp, kappa_interp, albedo_interp


def extract_vertex_indices_from_si(si: 'mi.SurfaceInteraction3f', scene: 'mi.Scene' = None) -> Tuple['mi.Vector3u', 'mi.Point2f']:
    """
    Extract vertex indices and barycentric coordinates from surface interaction.

    For Mitsuba 3, we extract:
    - Primitive index (triangle/face index)
    - Barycentric coordinates from si.uv
    - Vertex indices by accessing the mesh's face buffer

    Args:
        si: Surface interaction from ray tracing
        scene: Optional Mitsuba scene to extract shape from (recommended)

    Returns:
        (vertex_indices, bary_coords):
            vertex_indices: [N, 3] or Vector3u - Three vertex indices per hit triangle
            bary_coords: [N, 2] or Point2f - Barycentric coordinates (u, v)
                         Third coordinate w = 1 - u - v
    """
    # Get primitive index (triangle/face index)
    if not hasattr(si, 'prim_index'):
        raise ValueError("Surface interaction does not have prim_index")

    prim_idx = si.prim_index

    # Get barycentric coordinates
    # In Mitsuba 3, si.uv contains barycentric coordinates for meshes
    if hasattr(si, 'uv'):
        bary_coords = si.uv  # This is Point2f with (u, v), w = 1 - u - v
    else:
        # Fallback: use center of triangle
        N = dr.width(prim_idx)
        bary_coords = mi.Point2f(
            dr.full(mi.Float, 1.0/3.0, N),
            dr.full(mi.Float, 1.0/3.0, N)
        )

    # Get mesh shape to extract vertex indices
    # If scene is provided, get the shape from the scene (more reliable)
    if scene is not None:
        shape = None
        for s in scene.shapes():
            shape = s
            break  # Use first shape (assumes single mesh scene)
        if shape is None:
            raise ValueError("No shapes found in scene")
    else:
        # Fallback: try to get from si.shape (may be unreliable with ShapePtr)
        if not hasattr(si, 'shape') or si.shape is None:
            raise ValueError("Surface interaction does not have shape and no scene provided")
        shape = si.shape

    # Access the face buffer from the mesh shape
    params = mi.traverse(shape)

    if 'faces' not in params:
        raise ValueError(
            "Mesh shape does not have 'faces' parameter in traversable parameters. "
            "Cannot extract vertex indices. This is required for per-vertex operations."
        )

    # faces is stored as flattened buffer: [f0v0, f0v1, f0v2, f1v0, f1v1, f1v2, ...]
    faces_buffer = params['faces']

    # Gather the 3 vertex indices for each primitive
    # faces_buffer layout: face i has vertices at indices i*3+0, i*3+1, i*3+2
    idx0 = prim_idx * 3 + 0
    idx1 = prim_idx * 3 + 1
    idx2 = prim_idx * 3 + 2

    v0_idx = dr.gather(mi.UInt32, faces_buffer, idx0)
    v1_idx = dr.gather(mi.UInt32, faces_buffer, idx1)
    v2_idx = dr.gather(mi.UInt32, faces_buffer, idx2)

    # Create Vector3u with the three vertex indices
    vertex_indices = mi.Vector3u(v0_idx, v1_idx, v2_idx)

    return vertex_indices, bary_coords


def get_vertex_material_params(si: 'mi.SurfaceInteraction3f',
                                vertex_materials: VertexMaterialParams) -> Tuple['mi.Float', 'mi.Float', 'mi.Float', 'mi.Float']:
    """
    Get material parameters at surface interaction using per-vertex materials.

    Args:
        si: Surface interaction from ray tracing
        vertex_materials: Per-vertex material parameter container

    Returns:
        (alpha, eta, kappa, diffuse_albedo): [N] Material parameters at hit points
    """
    # Extract vertex indices and barycentric coordinates
    vertex_indices, bary_coords = extract_vertex_indices_from_si(si)

    # Interpolate material parameters
    alpha, eta, kappa, diffuse_albedo = vertex_materials.get_interpolated_params(
        vertex_indices, bary_coords
    )

    return alpha, eta, kappa, diffuse_albedo


# ============================================================================
# Simplified API for Testing
# ============================================================================

def create_simple_mesh_with_vertex_materials(vertices: np.ndarray,
                                             faces: np.ndarray,
                                             alpha_per_vertex: Optional[np.ndarray] = None,
                                             eta_per_vertex: Optional[np.ndarray] = None,
                                             kappa_per_vertex: Optional[np.ndarray] = None) -> Tuple['mi.Shape', VertexMaterialParams]:
    """
    Create a simple mesh with per-vertex material parameters for testing.

    Args:
        vertices: [num_vertices, 3] Vertex positions
        faces: [num_faces, 3] Face vertex indices
        alpha_per_vertex: [num_vertices] Roughness per vertex
        eta_per_vertex: [num_vertices] Real IOR per vertex
        kappa_per_vertex: [num_vertices] Imaginary IOR per vertex

    Returns:
        (shape, vertex_materials): Mitsuba shape and vertex material container
    """
    num_vertices = vertices.shape[0]

    # Create Mitsuba mesh
    mesh_dict = {
        'type': 'ply',  # Or 'obj'
        'filename': 'temp_mesh.ply',  # Placeholder
        'bsdf': {
            'type': 'conductor',
            'material': 'Al'  # Default material (overridden by vertex params)
        }
    }

    # Note: This is a simplified version. In practice, you'd use:
    # mi.load_dict() with proper mesh construction

    # Create vertex material parameters
    if alpha_per_vertex is not None:
        alpha = mi.Float(alpha_per_vertex)
    else:
        alpha = None

    if eta_per_vertex is not None:
        eta = mi.Float(eta_per_vertex)
    else:
        eta = None

    if kappa_per_vertex is not None:
        kappa = mi.Float(kappa_per_vertex)
    else:
        kappa = None

    vertex_materials = VertexMaterialParams(
        num_vertices=num_vertices,
        alpha=alpha,
        eta=eta,
        kappa=kappa
    )

    # Return placeholder shape and vertex materials
    # In actual implementation, create proper Mitsuba mesh
    shape = None  # Would be actual mi.Shape

    return shape, vertex_materials
