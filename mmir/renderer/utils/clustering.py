"""
Planar patch clustering for image method acceleration.

Groups coplanar adjacent mesh triangles into patches so the image method
can test specular points against the union of triangles in a patch rather
than against individual tiny triangles.

LiDAR-derived meshes have very small triangles (~1-5 cm edges). The specular
point computed via the law of reflection almost never falls inside a single
tiny triangle (<0.15% acceptance rate). By clustering coplanar triangles into
patches, the acceptance rate jumps to ~30-60%.

Algorithm (runs once per scene load):
1. Compute per-triangle normals, centroids, and areas
2. Build edge-based adjacency graph (shared edge = 2 shared vertices)
3. Flood-fill BFS: merge adjacent triangles if coplanar
   - Normal agreement: dot(n_seed, n_neighbor) > cos(angle_threshold)
   - Plane offset agreement: |offset_diff| < distance_threshold
4. Build padded GPU arrays for vectorized PIT testing at render time
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from collections import defaultdict, deque
import numpy as np
import drjit as dr
import mitsuba as mi


@dataclass
class PatchData:
    """
    Precomputed patch clustering data for GPU-accelerated patch PIT testing.

    Preprocessing (numpy) produces these arrays, which are uploaded to GPU
    as DrJit arrays for render-time use.
    """
    # Patch metadata
    n_patches: int                              # Total number of patches
    max_tris_per_patch: int                     # Actual max after capping (for padding)

    # Triangle-to-patch mapping
    tri_to_patch_np: np.ndarray                 # [n_triangles] int32 -- numpy for CPU
    tri_to_patch: 'mi.Int32'                    # [n_triangles] -- DrJit for GPU gather

    # Per-patch plane geometry [n_patches]
    patch_normal: 'mi.Vector3f'                 # Area-weighted average normal
    patch_offset: 'mi.Float'                    # Plane offset d = dot(N, V_on_plane)
    patch_area: 'mi.Float'                      # Sum of triangle areas in patch

    # Padded triangle vertex arrays [n_patches * max_tris_per_patch] (flat)
    patch_tri_v0: 'mi.Point3f'                  # v0 for each slot
    patch_tri_v1: 'mi.Point3f'                  # v1
    patch_tri_v2: 'mi.Point3f'                  # v2
    patch_tri_valid: 'mi.Bool'                  # True for real entries, False for padding


class PatchClusterer:
    """
    Clusters coplanar mesh triangles into patches for image method acceleration.

    Algorithm:
    1. Compute per-triangle normals, centroids, and areas
    2. Build adjacency graph (edge-based or spatial proximity)
    3. Flood-fill BFS: merge adjacent triangles if coplanar
    4. Produce PatchData with padded vertex arrays for GPU PIT testing

    Spatial adjacency (use_spatial_adjacency=True) connects triangles whose
    vertices are within `spatial_radius` meters, bridging topological gaps
    common in LiDAR-reconstructed meshes (T-junctions, disconnected fans,
    non-manifold edges). This produces much larger patches on original
    undecimated meshes.

    Args:
        angle_threshold_deg: Maximum angle between normals for merging (degrees)
        distance_threshold: Maximum plane offset difference for merging (meters)
        max_tris_per_patch: Cap on triangles per patch (memory guard). Patches
            exceeding this are truncated (nearest-to-centroid triangles kept).
        use_spatial_adjacency: Use spatial proximity instead of edge-only adjacency
        spatial_radius: Vertex proximity radius for spatial adjacency (meters)
        verbose: Print clustering statistics
    """

    def __init__(
        self,
        angle_threshold_deg: float = 3.0,
        distance_threshold: float = 0.02,
        max_tris_per_patch: int = 2000,
        use_spatial_adjacency: bool = False,
        spatial_radius: float = 0.05,
        verbose: bool = True,
    ):
        self.cos_angle_threshold = np.cos(np.radians(angle_threshold_deg))
        self.distance_threshold = distance_threshold
        self.max_tris_per_patch = max_tris_per_patch
        self.use_spatial_adjacency = use_spatial_adjacency
        self.spatial_radius = spatial_radius
        self.verbose = verbose

    def build_patches(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> PatchData:
        """
        Build patch clustering from mesh data.

        Runs once per scene and produces GPU-resident data structures.

        Args:
            vertices: Mesh vertices [n_vertices, 3] float
            faces: Mesh face indices [n_triangles, 3] int

        Returns:
            PatchData with all precomputed arrays
        """
        n_triangles = len(faces)

        if self.verbose:
            print(f"\n[PatchClustering] Building patches from {n_triangles:,} triangles")

        # Step 1: Compute per-triangle properties
        normals, centroids, areas = self._compute_triangle_properties(vertices, faces)

        # Step 2: Build adjacency graph
        if self.use_spatial_adjacency:
            adjacency = self._build_adjacency_spatial(vertices, faces, n_triangles)
        else:
            adjacency = self._build_adjacency(faces, n_triangles)

        if self.verbose:
            n_edges = sum(len(v) for v in adjacency.values()) // 2
            mode_str = f"spatial (r={self.spatial_radius}m)" if self.use_spatial_adjacency else "edge-based"
            print(f"  Adjacency ({mode_str}): {n_edges:,} edges")

        # Step 3: Flood-fill patches
        tri_to_patch = self._flood_fill_patches(normals, centroids, adjacency, n_triangles)
        n_patches = int(tri_to_patch.max()) + 1

        if self.verbose:
            print(f"  Patches: {n_patches:,}")

        # Step 4: Build PatchData with padded GPU arrays
        patch_data = self._build_patch_data(
            vertices, faces, normals, centroids, areas, tri_to_patch, n_patches
        )

        if self.verbose:
            # Print size distribution
            sizes = patch_data.tri_to_patch_np
            patch_sizes = np.bincount(sizes, minlength=n_patches)
            print(f"  Patch size: min={patch_sizes.min()}, median={int(np.median(patch_sizes))}, "
                  f"max={patch_sizes.max()}, mean={patch_sizes.mean():.1f}")
            print(f"  Padded max_tris_per_patch: {patch_data.max_tris_per_patch}")
            print(f"  GPU array size: {n_patches} x {patch_data.max_tris_per_patch} = "
                  f"{n_patches * patch_data.max_tris_per_patch:,} entries")

        return patch_data

    def build_patches_from_scene(self, scene: 'mi.Scene') -> PatchData:
        """
        Build patches from a Mitsuba scene's first mesh shape.

        Extracts vertices and faces via Mitsuba's mesh API, guaranteeing
        index consistency with si.prim_index used during rendering.

        Args:
            scene: Mitsuba scene with at least one mesh shape

        Returns:
            PatchData
        """
        shapes = scene.shapes()
        if len(shapes) == 0:
            raise ValueError("Scene has no shapes")

        mesh = shapes[0]
        n_faces = mesh.face_count()
        n_verts = mesh.vertex_count()

        if self.verbose:
            print(f"[PatchClustering] Extracting mesh: {n_verts:,} vertices, {n_faces:,} faces")

        # Extract face indices as numpy [n_faces, 3]
        all_face_idx = dr.arange(mi.UInt32, n_faces)
        fi = mesh.face_indices(all_face_idx)  # tuple of 3 UInt32 arrays
        faces_np = np.stack([
            np.array(fi[0], dtype=np.int32),
            np.array(fi[1], dtype=np.int32),
            np.array(fi[2], dtype=np.int32),
        ], axis=1)

        # Extract vertex positions as numpy [n_verts, 3]
        all_vert_idx = dr.arange(mi.UInt32, n_verts)
        vp = mesh.vertex_position(all_vert_idx)  # Point3f [n_verts]
        vertices_np = np.stack([
            np.array(vp.x, dtype=np.float32),
            np.array(vp.y, dtype=np.float32),
            np.array(vp.z, dtype=np.float32),
        ], axis=1)

        return self.build_patches(vertices_np, faces_np)

    # ================================================================
    # Internal methods
    # ================================================================

    @staticmethod
    def _compute_triangle_properties(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-triangle normals, centroids, and areas (vectorized numpy).

        Returns:
            normals: [n_triangles, 3] unit normals
            centroids: [n_triangles, 3] centroids
            areas: [n_triangles] areas
        """
        v0 = vertices[faces[:, 0]]  # [n_tri, 3]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms_safe = np.maximum(norms, 1e-12)
        normals = cross / norms_safe
        areas = 0.5 * norms.squeeze(-1)
        centroids = (v0 + v1 + v2) / 3.0

        return normals, centroids, areas

    @staticmethod
    def _build_adjacency(
        faces: np.ndarray,
        n_triangles: int,
    ) -> Dict[int, List[int]]:
        """
        Build edge-based adjacency graph.

        Two triangles are adjacent if they share an edge (2 vertices).
        Uses a dictionary mapping sorted vertex pair -> list of triangle indices.

        Returns:
            adjacency: Dict mapping triangle index -> list of adjacent triangle indices
        """
        edge_to_tris: Dict[tuple, List[int]] = defaultdict(list)

        for tri_idx in range(n_triangles):
            f = faces[tri_idx]
            for i in range(3):
                edge = (min(int(f[i]), int(f[(i + 1) % 3])),
                        max(int(f[i]), int(f[(i + 1) % 3])))
                edge_to_tris[edge].append(tri_idx)

        adjacency: Dict[int, List[int]] = defaultdict(list)
        for edge, tris in edge_to_tris.items():
            if len(tris) == 2:
                adjacency[tris[0]].append(tris[1])
                adjacency[tris[1]].append(tris[0])
            elif len(tris) > 2:
                # Non-manifold edge: connect all pairs
                for i in range(len(tris)):
                    for j in range(i + 1, len(tris)):
                        adjacency[tris[i]].append(tris[j])
                        adjacency[tris[j]].append(tris[i])

        return adjacency

    def _build_adjacency_spatial(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_triangles: int,
    ) -> Dict[int, List[int]]:
        """
        Spatial proximity adjacency: triangles are neighbors if any of their
        vertices are within `spatial_radius` meters of each other.

        This bridges topological gaps in LiDAR-reconstructed meshes
        (T-junctions, disconnected fans, non-manifold edges, duplicate
        vertices at slightly different positions).

        Uses scipy cKDTree for efficient O(n log n) spatial queries.
        """
        from scipy.spatial import cKDTree
        import time

        t0 = time.time()
        radius = self.spatial_radius

        # Build vertex -> triangle mapping
        vert_to_tris: Dict[int, List[int]] = defaultdict(list)
        for tri_idx in range(n_triangles):
            for v in faces[tri_idx]:
                vert_to_tris[int(v)].append(tri_idx)

        # Build KD-tree over unique vertex positions
        # Only include vertices that are actually used by faces
        used_vert_ids = np.unique(faces.ravel())
        used_positions = vertices[used_vert_ids]  # [n_used, 3]

        tree = cKDTree(used_positions)
        pairs = tree.query_pairs(r=radius)  # set of (i, j) index pairs into used_positions

        # Map back to original vertex IDs
        adjacency: Dict[int, List[int]] = defaultdict(list)

        # First add all edge-based adjacency (guaranteed neighbors)
        edge_adj = self._build_adjacency(faces, n_triangles)
        for tri_idx, neighbors in edge_adj.items():
            adjacency[tri_idx] = list(neighbors)

        # Then add spatial adjacency from vertex proximity
        n_spatial_edges = 0
        for i_local, j_local in pairs:
            vi = int(used_vert_ids[i_local])
            vj = int(used_vert_ids[j_local])
            for ti in vert_to_tris[vi]:
                for tj in vert_to_tris[vj]:
                    if ti != tj and tj not in adjacency.get(ti, []):
                        adjacency[ti].append(tj)
                        adjacency[tj].append(ti)
                        n_spatial_edges += 1

        t1 = time.time()
        if self.verbose:
            n_edge_edges = sum(len(v) for v in edge_adj.values()) // 2
            print(f"  Spatial adjacency: {n_edge_edges:,} edge-based + {n_spatial_edges:,} spatial "
                  f"({t1-t0:.1f}s)")

        return adjacency

    def _flood_fill_patches(
        self,
        normals: np.ndarray,
        centroids: np.ndarray,
        adjacency: Dict[int, List[int]],
        n_triangles: int,
    ) -> np.ndarray:
        """
        Flood-fill BFS to assign patch IDs.

        Starting from each unvisited triangle, BFS to neighbors satisfying:
        - dot(n_seed, n_neighbor) > cos_angle_threshold
        - |dot(n_seed, c_neighbor) - dot(n_seed, c_seed)| < distance_threshold

        Tests against the SEED plane (not previous triangle) to prevent drift.

        Returns:
            tri_to_patch: [n_triangles] int32 mapping
        """
        tri_to_patch = np.full(n_triangles, -1, dtype=np.int32)
        patch_id = 0

        for seed in range(n_triangles):
            if tri_to_patch[seed] != -1:
                continue

            # Start new patch with seed's plane
            seed_normal = normals[seed]
            seed_offset = np.dot(seed_normal, centroids[seed])

            queue = deque([seed])
            tri_to_patch[seed] = patch_id

            while queue:
                current = queue.popleft()
                for neighbor in adjacency.get(current, []):
                    if tri_to_patch[neighbor] != -1:
                        continue

                    # Coplanarity test against SEED plane
                    n_dot = np.dot(seed_normal, normals[neighbor])
                    if n_dot < self.cos_angle_threshold:
                        continue

                    neighbor_offset = np.dot(seed_normal, centroids[neighbor])
                    if abs(neighbor_offset - seed_offset) > self.distance_threshold:
                        continue

                    tri_to_patch[neighbor] = patch_id
                    queue.append(neighbor)

            patch_id += 1

        return tri_to_patch

    def _build_patch_data(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray,
        centroids: np.ndarray,
        areas: np.ndarray,
        tri_to_patch: np.ndarray,
        n_patches: int,
    ) -> PatchData:
        """
        Build the PatchData structure with padded GPU arrays.

        For each patch:
        1. Collect all triangle vertex triples
        2. Compute area-weighted average normal
        3. Compute plane offset
        4. Sort by distance to centroid, truncate to max_tris_per_patch cap
        5. Pad to max_tris_per_patch with valid=False
        6. Upload to GPU as DrJit arrays
        """
        cap = self.max_tris_per_patch

        # Group triangles by patch
        patch_tri_lists: List[List[int]] = [[] for _ in range(n_patches)]
        for tri_idx, pid in enumerate(tri_to_patch):
            patch_tri_lists[pid].append(tri_idx)

        # Per-patch metadata
        patch_normals_np = np.zeros((n_patches, 3), dtype=np.float64)
        patch_offsets_np = np.zeros(n_patches, dtype=np.float64)
        patch_areas_np = np.zeros(n_patches, dtype=np.float64)
        n_truncated = 0

        for pid in range(n_patches):
            tri_list = patch_tri_lists[pid]
            if not tri_list:
                continue

            # Area-weighted average normal
            tri_areas = areas[tri_list]
            tri_normals = normals[tri_list]
            total_area = tri_areas.sum()
            patch_areas_np[pid] = total_area

            if total_area > 1e-12:
                weighted_normal = (tri_normals * tri_areas[:, None]).sum(axis=0)
                norm = np.linalg.norm(weighted_normal)
                patch_normals_np[pid] = weighted_normal / max(norm, 1e-12)
            else:
                patch_normals_np[pid] = tri_normals[0]

            # Plane offset: area-weighted average of dot(N, v0)
            pn = patch_normals_np[pid]
            v0_all = vertices[faces[tri_list, 0]]  # [n_tris_in_patch, 3]
            offsets = np.einsum('j,ij->i', pn, v0_all)  # dot(N, v0) per triangle
            if tri_areas.sum() > 0:
                patch_offsets_np[pid] = np.average(offsets, weights=tri_areas)
            else:
                patch_offsets_np[pid] = offsets.mean() if len(offsets) > 0 else 0.0

            # Sort by distance to patch centroid, truncate to cap
            if len(tri_list) > cap:
                if tri_areas.sum() > 0:
                    patch_centroid = np.average(centroids[tri_list], weights=tri_areas, axis=0)
                else:
                    patch_centroid = centroids[tri_list].mean(axis=0)
                tri_centroids = centroids[tri_list]
                dists = np.linalg.norm(tri_centroids - patch_centroid, axis=1)
                sorted_indices = np.argsort(dists)
                patch_tri_lists[pid] = [tri_list[i] for i in sorted_indices[:cap]]
                if self.verbose:
                    print(f"  [PatchClustering] WARNING: Patch {pid} truncated "
                          f"from {len(tri_list)} to {cap} triangles")
                n_truncated += 1

        if self.verbose and n_truncated > 0:
            print(f"  [PatchClustering] {n_truncated} patches truncated to cap={cap}")

        # Determine actual max_tris after capping
        actual_max_tris = max(len(tl) for tl in patch_tri_lists) if patch_tri_lists else 1
        actual_max_tris = max(actual_max_tris, 1)  # At least 1

        # Allocate padded arrays (numpy, then upload to GPU)
        total_entries = n_patches * actual_max_tris

        v0_x = np.zeros(total_entries, dtype=np.float32)
        v0_y = np.zeros(total_entries, dtype=np.float32)
        v0_z = np.zeros(total_entries, dtype=np.float32)
        v1_x = np.zeros(total_entries, dtype=np.float32)
        v1_y = np.zeros(total_entries, dtype=np.float32)
        v1_z = np.zeros(total_entries, dtype=np.float32)
        v2_x = np.zeros(total_entries, dtype=np.float32)
        v2_y = np.zeros(total_entries, dtype=np.float32)
        v2_z = np.zeros(total_entries, dtype=np.float32)
        valid_mask = np.zeros(total_entries, dtype=bool)

        for pid in range(n_patches):
            tri_list = patch_tri_lists[pid]
            base = pid * actual_max_tris
            for local_idx, tri_idx in enumerate(tri_list):
                flat_idx = base + local_idx
                f = faces[tri_idx]
                v0_x[flat_idx] = vertices[f[0], 0]
                v0_y[flat_idx] = vertices[f[0], 1]
                v0_z[flat_idx] = vertices[f[0], 2]
                v1_x[flat_idx] = vertices[f[1], 0]
                v1_y[flat_idx] = vertices[f[1], 1]
                v1_z[flat_idx] = vertices[f[1], 2]
                v2_x[flat_idx] = vertices[f[2], 0]
                v2_y[flat_idx] = vertices[f[2], 1]
                v2_z[flat_idx] = vertices[f[2], 2]
                valid_mask[flat_idx] = True

        # Upload to GPU as DrJit arrays
        return PatchData(
            n_patches=n_patches,
            max_tris_per_patch=actual_max_tris,
            tri_to_patch_np=tri_to_patch,
            tri_to_patch=mi.Int32(tri_to_patch),
            patch_normal=mi.Vector3f(
                mi.Float(patch_normals_np[:, 0].astype(np.float32)),
                mi.Float(patch_normals_np[:, 1].astype(np.float32)),
                mi.Float(patch_normals_np[:, 2].astype(np.float32)),
            ),
            patch_offset=mi.Float(patch_offsets_np.astype(np.float32)),
            patch_area=mi.Float(patch_areas_np.astype(np.float32)),
            patch_tri_v0=mi.Point3f(mi.Float(v0_x), mi.Float(v0_y), mi.Float(v0_z)),
            patch_tri_v1=mi.Point3f(mi.Float(v1_x), mi.Float(v1_y), mi.Float(v1_z)),
            patch_tri_v2=mi.Point3f(mi.Float(v2_x), mi.Float(v2_y), mi.Float(v2_z)),
            patch_tri_valid=mi.Bool(valid_mask),
        )


__all__ = [
    'PatchClusterer',
    'PatchData',
]
