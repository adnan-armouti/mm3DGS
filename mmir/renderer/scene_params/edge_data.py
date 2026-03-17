"""
Edge data structure for DrJit/Mitsuba diffraction.
"""

from dataclasses import dataclass
import mitsuba as mi


@dataclass
class EdgeDataDr:
    """Edge data structure for DrJit/Mitsuba."""
    num_edges: int
    edge_points: mi.Point3f   # [E, 3] Edge midpoints
    edge_directions: mi.Vector3f  # [E, 3] Edge directions
    wedge_angles: mi.Float  # [E] Wedge angles in radians
    n0: mi.Vector3f  # [E, 3] Normal of face 0
    n1: mi.Vector3f  # [E, 3] Normal of face 1
    edge_v_idx: mi.UInt32  # [2*E] Vertex indices for edges (v0s, then v1s)
    edge_f_idx: mi.Int32  # [2*E] Face indices for edges (f0s, then f1s, -1 for boundary)
    edge_lengths: mi.Float = None  # [E] Length of each edge (for MIS PDF computation)
    total_edge_length: float = 0.0  # Sum of all edge lengths (for MIS PDF computation)
