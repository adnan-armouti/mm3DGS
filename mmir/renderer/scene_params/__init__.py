"""Differentiable scene parameters.

Per-vertex learnable representations consumed by the integrator's AD pipeline:
materials (permittivity, conductivity, roughness), surface normals, vertex
positions, and mesh topology (faces, edges for diffraction).

Modules
-------
vertex_materials        Per-vertex material parameters with barycentric interpolation
vertex_positions        Learnable vertex coordinates
vertex_normals          Learnable surface normal maps
face_data               Triangle face index data
edge_data               Edge geometry for diffraction
extract_edges_from_mesh Sharp-edge extraction from PLY meshes
"""
