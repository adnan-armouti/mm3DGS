"""
FNV-1a hashing and specular path deduplication.

Adapted from Sionna RT's path deduplication (sb_candidate_generator.py, lines 484-498).

Purpose: Multiple SBR rays can hit the same triangle. For specular paths, the
image method will produce the same specular reflection point regardless of where
on the triangle the ray hit. Deduplication keeps only the first occurrence per
unique triangle, avoiding redundant specular path contributions.

Algorithm:
1. Quantize hit positions/normals to grid (eps=1e-5)
2. Hash plane (normal + distance) using FNV-1a
3. Use scatter_inc counters with multiple hash functions
4. Keep only first occurrence (counter == 0)
"""

import drjit as dr
import mitsuba as mi
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.clustering import PatchData


# =============================================================================
# FNV-1a Constants (64-bit)
# =============================================================================
FNV_OFFSET_BASIS = np.uint64(14695981039346656037)
FNV_PRIME = np.uint64(1099511628211)


# =============================================================================
# Core Hash Functions
# =============================================================================

def hash_fnv1a_uint32(value: 'mi.UInt32', h: 'mi.UInt64') -> 'mi.UInt64':
    """
    FNV-1a hash: fold a UInt32 value into a running 64-bit hash.

    FNV-1a processes one byte at a time:
        h = (h ^ byte) * prime

    We process all 4 bytes of the UInt32.

    Args:
        value: 32-bit unsigned integer to hash
        h: Running hash state

    Returns:
        Updated hash state
    """
    prime = mi.UInt64(FNV_PRIME)

    # Process 4 bytes (little-endian order)
    for shift in [0, 8, 16, 24]:
        byte_val = mi.UInt64((value >> mi.UInt32(shift)) & mi.UInt32(0xFF))
        h = (h ^ byte_val) * prime

    return h


def quantize_and_hash_float(value: 'mi.Float', eps: float, h: 'mi.UInt64') -> 'mi.UInt64':
    """
    Quantize a float to integer grid and hash it.

    Quantization: int_val = round(value / eps)
    This ensures that values within eps of each other map to the same hash.

    Args:
        value: Float value to quantize and hash
        eps: Quantization step size
        h: Running hash state

    Returns:
        Updated hash state
    """
    # Quantize: round to nearest grid point
    int_val = mi.Int32(dr.round(value / mi.Float(eps)))
    # Reinterpret as unsigned for hashing
    uint_val = dr.reinterpret_array(mi.UInt32, int_val)
    return hash_fnv1a_uint32(uint_val, h)


# =============================================================================
# Plane Hashing
# =============================================================================

class PlaneHasher:
    """
    Hash reflecting planes for specular path deduplication.

    A reflecting plane is defined by its normal N and a point on the plane V.
    We hash the plane equation: N·X = d, where d = N·V.

    To ensure consistent hashing regardless of normal orientation:
    - Orient normals so the largest-magnitude component is positive
    - This handles both sides of double-sided surfaces

    Args:
        eps: Quantization step for float hashing (meters)
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    def hash_planes(
        self,
        hit_N: 'mi.Vector3f',
        hit_P: 'mi.Point3f',
        h_init: 'mi.UInt64' = None,
    ) -> 'mi.UInt64':
        """
        Compute hash for each hit's reflecting plane.

        Args:
            hit_N: Surface normals [N]
            hit_P: Hit positions (any point on the plane) [N]
            h_init: Initial hash state (default: FNV offset basis)

        Returns:
            Hash values [N] - one per hit
        """
        if h_init is None:
            h_init = dr.full(mi.UInt64, int(FNV_OFFSET_BASIS), dr.width(hit_N))

        # Step 1: Orient normals consistently
        # Flip so the component with largest absolute value is positive
        abs_x = dr.abs(hit_N.x)
        abs_y = dr.abs(hit_N.y)
        abs_z = dr.abs(hit_N.z)

        # Find which component has largest magnitude
        xy_max = dr.maximum(abs_x, abs_y)
        max_comp = dr.maximum(xy_max, abs_z)

        # Determine sign of the dominant component
        # If x is dominant: use sign(N.x)
        # If y is dominant: use sign(N.y)
        # If z is dominant: use sign(N.z)
        x_dominant = abs_x >= max_comp - mi.Float(1e-10)
        y_dominant = (~x_dominant) & (abs_y >= max_comp - mi.Float(1e-10))
        # z_dominant is the remaining case

        sign_x = dr.select(hit_N.x >= mi.Float(0), mi.Float(1.0), mi.Float(-1.0))
        sign_y = dr.select(hit_N.y >= mi.Float(0), mi.Float(1.0), mi.Float(-1.0))
        sign_z = dr.select(hit_N.z >= mi.Float(0), mi.Float(1.0), mi.Float(-1.0))

        flip = dr.select(x_dominant, sign_x,
                dr.select(y_dominant, sign_y, sign_z))

        N_oriented = mi.Vector3f(hit_N.x * flip, hit_N.y * flip, hit_N.z * flip)

        # Step 2: Compute plane distance d = N · P
        d = dr.dot(N_oriented, mi.Vector3f(hit_P.x, hit_P.y, hit_P.z))

        # Step 3: Hash normal components and distance
        h = h_init
        h = quantize_and_hash_float(N_oriented.x, self.eps, h)
        h = quantize_and_hash_float(N_oriented.y, self.eps, h)
        h = quantize_and_hash_float(N_oriented.z, self.eps, h)
        h = quantize_and_hash_float(d, self.eps, h)

        return h


# =============================================================================
# Specular Path Deduplication
# =============================================================================

class SpecularDeduplicator:
    """
    Deduplicate specular paths using multi-hash scatter_inc counters.

    Following Sionna RT (sb_candidate_generator.py, lines 484-498):
    - Use multiple independent hash functions (different seeds)
    - For each hash function, scatter_inc into a counter array
    - A path is unique only if ALL hash functions report counter == 0
    - This is a probabilistic data structure (like a counting Bloom filter)

    Args:
        counter_size: Size of each hash counter array (larger = fewer false negatives)
        num_hash_functions: Number of independent hash functions
        eps: Quantization step for plane hashing
    """

    def __init__(
        self,
        counter_size: int = 10000,
        num_hash_functions: int = 2,
        eps: float = 1e-4,
    ):
        self.counter_size = counter_size
        self.num_hash_functions = num_hash_functions
        self.plane_hasher = PlaneHasher(eps=eps)

        # Different seeds for each hash function
        # Using prime multipliers of the offset basis
        self.hash_seeds = [
            np.uint64(14695981039346656037),  # Standard FNV offset basis
            np.uint64((14695981039346656037 * 31 + 7) % (2**64)),  # Different seed
        ][:num_hash_functions]

    def deduplicate(
        self,
        hit_N: 'mi.Vector3f',
        hit_P: 'mi.Point3f',
        valid: 'mi.Bool',
        n_rx: int,
        use_patches: bool = False,
        hit_ID: Optional['mi.Int32'] = None,
        patch_data: Optional['PatchData'] = None,
    ) -> Tuple['mi.Bool', dict]:
        """
        Deduplicate specular paths across all hits.

        Per-RX deduplication: each RX has its own counter space so that
        the same triangle hit from different RX elements is not deduplicated.

        When use_patches=True:
          - Maps hit_ID -> patch_id via gather from patch_data.tri_to_patch
          - Uses patch_id directly as counter index (no hashing, collision-free)
          - Counter space: n_rx * n_patches

        When use_patches=False:
          - Existing plane hash behavior (FNV-1a multi-hash)

        Args:
            hit_N: Surface normals [n_total_slots]
            hit_P: Hit positions [n_total_slots]
            valid: Valid hit mask [n_total_slots]
            n_rx: Number of RX elements
            use_patches: If True, use patch-level deduplication
            hit_ID: Triangle IDs [n_total_slots] (required when use_patches=True)
            patch_data: PatchData from PatchClusterer (required when use_patches=True)

        Returns:
            (unique_mask, stats): Boolean mask of unique hits, and statistics dict
        """
        if use_patches and hit_ID is not None and patch_data is not None:
            return self._deduplicate_patches(hit_ID, valid, n_rx, patch_data)
        else:
            return self._deduplicate_plane_hash(hit_N, hit_P, valid, n_rx)

    def _deduplicate_plane_hash(
        self,
        hit_N: 'mi.Vector3f',
        hit_P: 'mi.Point3f',
        valid: 'mi.Bool',
        n_rx: int,
    ) -> Tuple['mi.Bool', dict]:
        """Existing plane hash deduplication (FNV-1a multi-hash)."""
        n_total = dr.width(hit_N)
        n_hits_per_rx = n_total // n_rx

        # Compute RX index for each hit slot
        slot_indices = dr.arange(mi.UInt32, n_total)
        rx_idx = slot_indices // mi.UInt32(n_hits_per_rx)

        # Start with all valid hits as potentially unique
        unique = mi.Bool(valid)

        # For each hash function, check uniqueness
        for h_idx in range(self.num_hash_functions):
            # Initialize hash with seed
            h_init = dr.full(mi.UInt64, int(self.hash_seeds[h_idx]), n_total)

            # Hash the plane (normal + point)
            plane_hash = self.plane_hasher.hash_planes(hit_N, hit_P, h_init)

            # Compute counter index: (hash % counter_size) + rx_idx * counter_size
            # This gives per-RX counter space
            hash_mod = mi.UInt32(plane_hash % mi.UInt64(self.counter_size))
            counter_idx = rx_idx * mi.UInt32(self.counter_size) + hash_mod

            # Allocate counters (per-RX)
            total_counters = n_rx * self.counter_size
            counters = dr.zeros(mi.UInt32, total_counters)

            # scatter_inc: atomically increment and return previous value
            prev_count = dr.scatter_inc(counters, counter_idx, unique)

            # Only keep if this is the FIRST occurrence (prev_count == 0)
            unique = unique & (prev_count == mi.UInt32(0))

        # Force evaluation of lazy DrJit graph before computing stats
        dr.eval(unique)

        # Compute statistics
        n_valid = int(dr.sum(mi.UInt32(valid))[0])
        n_unique = int(dr.sum(mi.UInt32(unique))[0])

        stats = {
            'n_valid_hits': n_valid,
            'n_unique_triangles': n_unique,
            'n_duplicates_removed': n_valid - n_unique,
            'dedup_ratio': n_unique / max(n_valid, 1),
            'use_patches': False,
        }

        return unique, stats

    def deduplicate_compressed(
        self,
        hit_N: 'mi.Vector3f',
        hit_P: 'mi.Point3f',
        rx_idx: 'mi.UInt32',
        n_rx: int,
        hit_prim_ids: Optional['mi.UInt32'] = None,
        patch_data=None,
    ) -> Tuple['mi.Bool', dict]:
        """
        Deduplication for compressed (valid-only) hit format.

        When patch_data is provided (and hit_prim_ids is available), uses
        patch-level dedup (collision-free, matches sionna reference exactly).
        Otherwise falls back to plane-hash dedup.

        Args:
            hit_N: Surface normals [n_valid]
            hit_P: Hit positions [n_valid]
            rx_idx: Owning RX index per hit [n_valid]
            n_rx: Number of RX elements
            hit_prim_ids: Triangle primitive IDs [n_valid] (for patch dedup)
            patch_data: PatchData from PatchClusterer (for patch dedup)

        Returns:
            (unique_mask, stats): Boolean mask of unique hits, and statistics dict
        """
        if patch_data is not None and hit_prim_ids is not None:
            return self._deduplicate_patches_compressed(
                hit_prim_ids, rx_idx, n_rx, patch_data)
        else:
            return self._deduplicate_plane_hash_compressed(
                hit_N, hit_P, rx_idx, n_rx)

    def _deduplicate_plane_hash_compressed(
        self,
        hit_N: 'mi.Vector3f',
        hit_P: 'mi.Point3f',
        rx_idx: 'mi.UInt32',
        n_rx: int,
    ) -> Tuple['mi.Bool', dict]:
        """Plane-hash dedup for compressed format with explicit rx_idx."""
        n_total = dr.width(hit_N)

        unique = dr.full(mi.Bool, True, n_total)

        for h_idx in range(self.num_hash_functions):
            h_init = dr.full(mi.UInt64, int(self.hash_seeds[h_idx]), n_total)
            plane_hash = self.plane_hasher.hash_planes(hit_N, hit_P, h_init)

            hash_mod = mi.UInt32(plane_hash % mi.UInt64(self.counter_size))
            counter_idx = rx_idx * mi.UInt32(self.counter_size) + hash_mod

            total_counters = n_rx * self.counter_size
            counters = dr.zeros(mi.UInt32, total_counters)

            prev_count = dr.scatter_inc(counters, counter_idx, unique)
            unique = unique & (prev_count == mi.UInt32(0))

        dr.eval(unique)

        n_unique = int(dr.sum(mi.UInt32(unique))[0])

        stats = {
            'n_valid_hits': n_total,
            'n_unique_triangles': n_unique,
            'n_duplicates_removed': n_total - n_unique,
            'dedup_ratio': n_unique / max(n_total, 1),
            'use_patches': False,
        }

        return unique, stats

    def _deduplicate_patches_compressed(
        self,
        hit_prim_ids: 'mi.UInt32',
        rx_idx: 'mi.UInt32',
        n_rx: int,
        patch_data,
    ) -> Tuple['mi.Bool', dict]:
        """
        Patch-level dedup for compressed format with explicit rx_idx.
        Matches sionna reference renderer behavior exactly.
        """
        n_total = dr.width(hit_prim_ids)
        n_patches = patch_data.n_patches

        # Map prim_id -> patch_id
        patch_id = mi.UInt32(dr.gather(mi.Int32, patch_data.tri_to_patch, hit_prim_ids))

        # Counter index: rx_idx * n_patches + patch_id (per-RX, collision-free)
        counter_idx = rx_idx * mi.UInt32(n_patches) + patch_id

        total_counters = n_rx * n_patches
        counters = dr.zeros(mi.UInt32, total_counters)

        unique = dr.full(mi.Bool, True, n_total)
        prev_count = dr.scatter_inc(counters, counter_idx, unique)
        unique = unique & (prev_count == mi.UInt32(0))

        dr.eval(unique)

        n_unique = int(dr.sum(mi.UInt32(unique))[0])

        stats = {
            'n_valid_hits': n_total,
            'n_unique_triangles': n_unique,
            'n_unique_patches': n_unique,
            'n_duplicates_removed': n_total - n_unique,
            'dedup_ratio': n_unique / max(n_total, 1),
            'n_patches_total': n_patches,
            'use_patches': True,
        }

        return unique, stats

    def _deduplicate_patches(
        self,
        hit_ID: 'mi.Int32',
        valid: 'mi.Bool',
        n_rx: int,
        patch_data: 'PatchData',
    ) -> Tuple['mi.Bool', dict]:
        """
        Patch-level deduplication using direct scatter_inc on patch IDs.

        No hashing needed — patch_id is a small integer [0, n_patches).
        Counter space: n_rx * n_patches counters (collision-free).
        """
        n_total = dr.width(hit_ID)
        n_hits_per_rx = n_total // n_rx
        n_patches = patch_data.n_patches

        # Map hit_ID -> patch_id via gather
        # hit_ID can be -1 for invalid hits, clamp to valid range
        safe_id = mi.UInt32(dr.maximum(hit_ID, mi.Int32(0)))
        patch_id = mi.UInt32(dr.gather(mi.Int32, patch_data.tri_to_patch, safe_id))

        # Compute RX index for each slot
        slot_indices = dr.arange(mi.UInt32, n_total)
        rx_idx = slot_indices // mi.UInt32(n_hits_per_rx)

        # Counter index: rx_idx * n_patches + patch_id
        counter_idx = rx_idx * mi.UInt32(n_patches) + patch_id

        # Allocate counters (one per patch per RX)
        total_counters = n_rx * n_patches
        counters = dr.zeros(mi.UInt32, total_counters)

        # scatter_inc with valid mask
        unique = mi.Bool(valid)
        prev_count = dr.scatter_inc(counters, counter_idx, unique)
        unique = unique & (prev_count == mi.UInt32(0))

        # CRITICAL: Force evaluation before dr.sum() on scatter-modified var
        dr.eval(unique)

        # Compute statistics
        n_valid = int(dr.sum(mi.UInt32(valid))[0])
        n_unique = int(dr.sum(mi.UInt32(unique))[0])

        stats = {
            'n_valid_hits': n_valid,
            'n_unique_triangles': n_unique,
            'n_unique_patches': n_unique,
            'n_duplicates_removed': n_valid - n_unique,
            'dedup_ratio': n_unique / max(n_valid, 1),
            'n_patches_total': n_patches,
            'use_patches': True,
        }

        return unique, stats


__all__ = [
    'hash_fnv1a_uint32',
    'quantize_and_hash_float',
    'PlaneHasher',
    'SpecularDeduplicator',
    'FNV_OFFSET_BASIS',
    'FNV_PRIME',
]
