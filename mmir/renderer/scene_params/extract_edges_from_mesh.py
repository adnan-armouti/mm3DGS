"""
Extract edge data from a mesh file for diffraction.

This module creates edge data structure from a triangle mesh
that can be used for UTD diffraction calculations.
"""

import numpy as np
import drjit as dr
import mitsuba as mi

from typing import Tuple
from mmir.renderer.scene_params.edge_data import EdgeDataDr


def extract_edges_from_ply(ply_path: str, angle_threshold: float = 30.0) -> EdgeDataDr:
    """
    Extract sharp edges from a PLY mesh file.

    Args:
        ply_path: Path to PLY mesh file
        angle_threshold: Minimum dihedral angle in degrees for edge detection

    Returns:
        EdgeDataDr structure with edge information
    """
    import trimesh

    # Load mesh
    mesh = trimesh.load(ply_path)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)

    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12)

    # Build edge to face adjacency
    edge_to_faces = {}

    for face_idx, face in enumerate(faces):
        # Three edges per face
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]

        for edge in edges:
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append(face_idx)

    # Find sharp edges
    edge_list = []
    threshold_rad = np.deg2rad(angle_threshold)

    for edge, adjacent_faces in edge_to_faces.items():
        v0_idx, v1_idx = edge

        if len(adjacent_faces) == 1:
            # Boundary edge - always included
            f0_idx = adjacent_faces[0]
            edge_list.append({
                'v0_idx': v0_idx,
                'v1_idx': v1_idx,
                'f0_idx': f0_idx,
                'f1_idx': -1,  # No second face
                'wedge_angle': np.pi,  # 180 degrees for boundary
                'n0': face_normals[f0_idx],
                'n1': face_normals[f0_idx]  # Duplicate normal for boundary
            })

        elif len(adjacent_faces) == 2:
            # Interior edge - check dihedral angle
            f0_idx, f1_idx = adjacent_faces
            n0 = face_normals[f0_idx]
            n1 = face_normals[f1_idx]

            # Compute dihedral angle
            cos_angle = np.clip(np.dot(n0, n1), -1.0, 1.0)
            dihedral_angle = np.arccos(cos_angle)

            if dihedral_angle > threshold_rad:
                # Sharp edge
                wedge_angle = 2.0 * np.pi - dihedral_angle  # Interior wedge angle
                edge_list.append({
                    'v0_idx': v0_idx,
                    'v1_idx': v1_idx,
                    'f0_idx': f0_idx,
                    'f1_idx': f1_idx,
                    'wedge_angle': wedge_angle,
                    'n0': n0,
                    'n1': n1
                })

    num_edges = len(edge_list)

    if num_edges == 0:
        print(f"  WARNING: No sharp edges found with threshold {angle_threshold}deg")
        # Return empty structure
        return EdgeDataDr(
            num_edges=0,
            edge_points=mi.Point3f(0.0),
            edge_directions=mi.Vector3f(0.0),
            wedge_angles=mi.Float(0.0),
            n0=mi.Vector3f(0.0),
            n1=mi.Vector3f(0.0),
            edge_v_idx=mi.UInt32(0),
            edge_f_idx=mi.Int32(0),
            edge_lengths=mi.Float(0.0),
            total_edge_length=0.0
        )

    print(f"  Found {num_edges} sharp edges (threshold: {angle_threshold}deg)")

    # Convert to arrays
    edge_v0_idx = np.array([e['v0_idx'] for e in edge_list], dtype=np.uint32)
    edge_v1_idx = np.array([e['v1_idx'] for e in edge_list], dtype=np.uint32)
    edge_f0_idx = np.array([e['f0_idx'] for e in edge_list], dtype=np.int32)
    edge_f1_idx = np.array([e['f1_idx'] for e in edge_list], dtype=np.int32)
    wedge_angles_np = np.array([e['wedge_angle'] for e in edge_list], dtype=np.float32)
    n0_np = np.array([e['n0'] for e in edge_list], dtype=np.float32)
    n1_np = np.array([e['n1'] for e in edge_list], dtype=np.float32)

    # Compute edge midpoints and directions
    v0_pos = vertices[edge_v0_idx]
    v1_pos = vertices[edge_v1_idx]
    edge_points_np = 0.5 * (v0_pos + v1_pos)
    edge_vec = v1_pos - v0_pos
    edge_lengths_np = np.linalg.norm(edge_vec, axis=1, keepdims=True).squeeze()  # [N] array
    edge_directions_np = edge_vec / (edge_lengths_np[:, np.newaxis] + 1e-12)

    # Compute total edge length for MIS PDF normalization
    total_edge_length = float(np.sum(edge_lengths_np))

    # Convert to DrJit types
    # Note: Mitsuba expects shape (3, N) not (N, 3) so we need to transpose
    edge_data = EdgeDataDr(
        num_edges=num_edges,
        edge_points=mi.Point3f(edge_points_np.T),  # Transpose from (N, 3) to (3, N)
        edge_directions=mi.Vector3f(edge_directions_np.T),  # Transpose
        wedge_angles=mi.Float(wedge_angles_np),
        n0=mi.Vector3f(n0_np.T),  # Transpose
        n1=mi.Vector3f(n1_np.T),  # Transpose
        # Concatenate vertex indices [v0s, v1s]
        edge_v_idx=mi.UInt32(np.concatenate([edge_v0_idx, edge_v1_idx])),
        # Concatenate face indices [f0s, f1s]
        edge_f_idx=mi.Int32(np.concatenate([edge_f0_idx, edge_f1_idx])),
        # Edge lengths for MIS PDF computation
        edge_lengths=mi.Float(edge_lengths_np),
        total_edge_length=total_edge_length
    )

    # Print statistics
    boundary_edges = np.sum(edge_f1_idx < 0)
    interior_edges = num_edges - boundary_edges
    avg_wedge = np.rad2deg(np.mean(wedge_angles_np))

    print(f"  Edge statistics:")
    print(f"    Interior edges: {interior_edges}")
    print(f"    Boundary edges: {boundary_edges}")
    print(f"    Average wedge angle: {avg_wedge:.1f}deg")

    return edge_data


if __name__ == "__main__":
    # Test edge extraction
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_edges_from_mesh.py <mesh.ply>")
        sys.exit(1)

    mesh_path = sys.argv[1]
    print(f"Extracting edges from {mesh_path}...")
    edge_data = extract_edges_from_ply(mesh_path)

    if edge_data.num_edges > 0:
        print(f"\nSuccessfully extracted {edge_data.num_edges} edges")
    else:
        print("\nNo edges found - try reducing angle threshold")