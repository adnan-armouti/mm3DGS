"""World-space voxel grid matched to the cascade radar's RA polar grid.

Used by the v5_v4 voxel-aware init (variant `C1_voxel_v1` and successors).
The voxel grid is the **seed pose's RA grid** projected into world space —
each (range_bin, azimuth_bin) pair defines a wedge in 3D world coordinates
(no elevation discretization; full elevation extent per wedge).

Conventions match `mm25DGS_v5_v4.train_gaussian.build_polar_to_cart_grid` and
`mmir.data.ra_utils.ra_polar_to_cartesian`:

  * azimuth bins n_az = 127 (with num_angle_bins = 128 in the arcsin formula)
  * range bins   n_range = 256
  * range_res    from `mmir.data.io_utils.compute_range_res_from_cfg`
  * azimuth bin center i:   arcsin( (i - 63) * 2/128 )  for i ∈ [0, 127)
  * range  bin center j:    j * range_res                for j ∈ [0, 256)
  * azimuth bin spacing is uniform in sin(theta) (not in theta itself).

In radar-local coords, the boresight is +Y, lateral azimuth is X with the
sign convention `sin(az) = -x/r` (matches the cart-image flip baked into
`build_polar_to_cart_grid`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# Cascade radar polar grid constants (match `build_polar_to_cart_grid`).
N_AZ_BINS = 127
N_RANGE_BINS = 256
N_ANGLE_BINS = N_AZ_BINS + 1   # 128 — the divisor in arcsin formula
ARCSIN_SIN_MAX = (N_ANGLE_BINS // 2 - 1) * (2.0 / N_ANGLE_BINS)  # 126/128
COS_BORE_MIN = float(np.cos(np.arcsin(ARCSIN_SIN_MAX)))           # 0.1761...


@dataclass
class RadarGridSpecs:
    """Parameters defining a radar-pose-local RA voxel grid."""
    rx_center: np.ndarray     # (3,) world coords of array centroid
    boresight: np.ndarray     # (3,) unit vector
    R_world2radar: np.ndarray  # (3, 3) rotates world-frame vectors → radar local frame (boresight → +Y)
    range_res: float
    sin_edges: np.ndarray      # (N_AZ_BINS + 1,) bin edges in sin(theta), uniform
    range_edges: np.ndarray    # (N_RANGE_BINS + 1,) bin edges in metres
    sin_centers: np.ndarray    # (N_AZ_BINS,) bin centers in sin(theta) (= 2*(i-63)/128)
    range_centers: np.ndarray  # (N_RANGE_BINS,) bin centers in metres
    near_field_m: float = 1.5  # minimum useful range
    n_az: int = N_AZ_BINS
    n_range: int = N_RANGE_BINS

    @property
    def max_range(self) -> float:
        return float(self.n_range * self.range_res)

    @property
    def cos_bore_min(self) -> float:
        return float(COS_BORE_MIN)


def _rodrigues_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps src direction → dst direction.

    Uses Rodrigues' formula. Handles the antiparallel edge case.
    Both inputs must be unit-norm.
    """
    src = src / max(np.linalg.norm(src), 1e-12)
    dst = dst / max(np.linalg.norm(dst), 1e-12)
    c = float(np.dot(src, dst))
    if c > 0.9999:
        return np.eye(3, dtype=np.float32)
    if c < -0.9999:
        # Antiparallel — pick any axis perpendicular to src.
        if abs(src[0]) < 0.9:
            axis = np.cross(src, np.array([1.0, 0.0, 0.0])).astype(np.float32)
        else:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0])).astype(np.float32)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        # 180° rotation about axis
        return (np.eye(3, dtype=np.float32) + 2 * (K @ K)).astype(np.float32)
    v = np.cross(src, dst).astype(np.float32)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32) + K + K @ K * (1.0 / (1.0 + c))).astype(np.float32)


def make_grid_specs(rx_center: np.ndarray, boresight: np.ndarray,
                     range_res: float, near_field_m: float = 1.5,
                     ) -> RadarGridSpecs:
    """Construct the voxel-grid spec for a single pose.

    Args:
      rx_center: (3,) world coords of the radar's RX array centroid.
      boresight: (3,) world-direction unit vector for the array boresight.
      range_res: metres per range bin (from `compute_range_res_from_cfg`).
      near_field_m: minimum useful range; points closer are dropped.
    """
    rx_center = np.asarray(rx_center, dtype=np.float32).reshape(3)
    boresight = np.asarray(boresight, dtype=np.float32).reshape(3)
    boresight = boresight / max(np.linalg.norm(boresight), 1e-12)
    R = _rodrigues_align(boresight, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # Azimuth bins linear in sin(theta). 128 edges define 127 bins.
    sin_edges = np.arange(N_AZ_BINS + 1, dtype=np.float32) * (2.0 / N_ANGLE_BINS) \
                  - (N_AZ_BINS / N_ANGLE_BINS)
    # → edges = -127/128, -125/128, …, +127/128  (length 128)
    sin_centers = (sin_edges[:-1] + sin_edges[1:]) * 0.5  # 127 centers

    # Range bins centred at j*range_res; edges at (j-0.5)*range_res.
    range_edges = (np.arange(N_RANGE_BINS + 1, dtype=np.float32) - 0.5) * range_res
    range_centers = np.arange(N_RANGE_BINS, dtype=np.float32) * range_res

    return RadarGridSpecs(
        rx_center=rx_center, boresight=boresight, R_world2radar=R,
        range_res=float(range_res),
        sin_edges=sin_edges, range_edges=range_edges,
        sin_centers=sin_centers, range_centers=range_centers,
        near_field_m=float(near_field_m),
    )


def bin_points(xyz_world: np.ndarray, grid: RadarGridSpecs
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin world-space points into the radar's RA grid.

    Returns:
      r_bin:   (N,) int — range bin index, -1 for out-of-grid.
      az_bin:  (N,) int — azimuth bin index, -1 for out-of-grid.
      valid:   (N,) bool — True iff in [0, n_range) × [0, n_az) AND r >= near_field_m.
      r:       (N,) float — radial distance in metres.
    """
    xyz_world = np.asarray(xyz_world, dtype=np.float32)
    pts_local = xyz_world - grid.rx_center  # (N, 3)
    pts_radar = (grid.R_world2radar @ pts_local.T).T  # (N, 3); boresight = +Y

    r = np.linalg.norm(pts_radar, axis=1).clip(min=1e-9)
    # sign convention: sin(az) = -x/r matches the flip in build_polar_to_cart_grid.
    sin_az = -pts_radar[:, 0] / r

    r_bin = np.searchsorted(grid.range_edges, r, side='right') - 1
    az_bin = np.searchsorted(grid.sin_edges, sin_az, side='right') - 1
    valid = (
        (r >= grid.near_field_m)
        & (r_bin >= 0) & (r_bin < grid.n_range)
        & (az_bin >= 0) & (az_bin < grid.n_az)
    )
    return r_bin.astype(np.int32), az_bin.astype(np.int32), valid, r


def cell_to_world_center(r_bin: int, az_bin: int, grid: RadarGridSpecs) -> np.ndarray:
    """Forward map from (range_bin, az_bin) to a world-space wedge centroid.

    Useful for unit testing the round-trip and for visualisation.
    """
    r = grid.range_centers[r_bin]
    sin_az = grid.sin_centers[az_bin]
    cos_az = float(np.sqrt(max(1.0 - sin_az * sin_az, 0.0)))
    pt_radar = np.array([-sin_az * r, cos_az * r, 0.0], dtype=np.float32)
    R_inv = grid.R_world2radar.T  # rotation back from radar → world
    return (R_inv @ pt_radar) + grid.rx_center


def all_cell_world_centers(grid: RadarGridSpecs) -> np.ndarray:
    """Vectorised version of cell_to_world_center for the full grid.

    Returns: (n_cells, 3) where row r*n_az + a is the world centre of
    cell (range_bin=r, az_bin=a). Cell elevation is fixed at 0 in the
    radar local frame (matches the 2D RA assumption — full elevation
    extent collapsed).
    """
    r_centers = grid.range_centers                              # (n_range,)
    sin_centers = grid.sin_centers                              # (n_az,)
    cos_centers = np.sqrt(np.clip(1.0 - sin_centers ** 2, 0.0, 1.0))
    # Outer product: (n_range, n_az)
    x_radar = -np.outer(r_centers, sin_centers)                 # (n_range, n_az)
    y_radar = np.outer(r_centers, cos_centers)                  # (n_range, n_az)
    z_radar = np.zeros_like(x_radar)
    pts_radar = np.stack([x_radar.ravel(), y_radar.ravel(), z_radar.ravel()], axis=1).astype(np.float32)
    R_inv = grid.R_world2radar.T
    pts_world = (R_inv @ pts_radar.T).T + grid.rx_center
    return pts_world.astype(np.float32)


def per_voxel_ra_signal(grid_seed: RadarGridSpecs,
                         train_pose_dicts,
                         train_ra_mag_list,
                         range_res: float,
                         ) -> np.ndarray:
    """Compute per-voxel measured RA magnitude, averaged across train frames.

    For each cell in `grid_seed`, take its world centre and project into
    every train frame's polar grid. Look up the train RA magnitude at the
    corresponding (az_bin_F, r_bin_F). Return the average across the train
    frames where the projection lands inside the grid.

    Args:
      grid_seed: voxel grid in seed pose's coords.
      train_pose_dicts: list of pose dicts ({rx_positions, tx_boresights}).
      train_ra_mag_list: list of (n_az, n_range) numpy arrays (or torch
        tensors) — measured RA magnitudes per train frame.
      range_res: range resolution (matches the grid).

    Returns:
      signal_avg: (n_cells,) numpy array. signal_avg[r*n_az + a] is the
        mean of train_ra_F[a_F, r_F] across train frames F where the
        seed-pose centre of (r, a) projects into F's grid.
    """
    n_cells = grid_seed.n_az * grid_seed.n_range
    centers = all_cell_world_centers(grid_seed)                # (n_cells, 3)

    signal_sum = np.zeros(n_cells, dtype=np.float64)
    valid_count = np.zeros(n_cells, dtype=np.int64)

    for pose, ra in zip(train_pose_dicts, train_ra_mag_list):
        rx_c = pose['rx_positions'].mean(dim=0).cpu().numpy()
        bs = pose['tx_boresights'].mean(dim=0).cpu().numpy()
        grid_F = make_grid_specs(rx_c, bs, range_res, near_field_m=grid_seed.near_field_m)
        r_bin_F, az_bin_F, valid_F, _ = bin_points(centers, grid_F)
        if hasattr(ra, 'cpu'):
            ra_np = ra.detach().cpu().numpy()
        else:
            ra_np = np.asarray(ra)
        ra_np = ra_np.astype(np.float64)
        # ra shape (n_az, n_range) per project convention.
        valid_idx = np.flatnonzero(valid_F)
        if len(valid_idx) == 0:
            continue
        sig = ra_np[az_bin_F[valid_idx], r_bin_F[valid_idx]]
        signal_sum[valid_idx] += sig
        valid_count[valid_idx] += 1

    signal_avg = np.where(valid_count > 0,
                           signal_sum / np.maximum(valid_count, 1),
                           0.0)
    return signal_avg.astype(np.float32)


def grid_specs_from_rast(rast, range_res: float) -> RadarGridSpecs:
    """Convenience: build grid specs from a Rasterizer instance.

    rast.rx_positions is (n_rx, 3) torch; rast.tx_boresights is (n_tx, 3) torch.
    We collapse to centroid + average boresight (same convention as the
    init pipeline's seed-pose proxies).
    """
    rx_center = rast.rx_positions.mean(dim=0).detach().cpu().numpy()
    boresight = rast.tx_boresights.mean(dim=0).detach().cpu().numpy()
    return make_grid_specs(rx_center, boresight, range_res)


# ---------------------------------------------------------------------------
# Per-voxel budget allocation + within-voxel selection helpers
# (used by the C1_voxel_v1 init variant)
# ---------------------------------------------------------------------------


def proportional_budget(weights: np.ndarray, capacity: np.ndarray, total: int
                         ) -> np.ndarray:
    """Allocate `total` items across cells proportional to `weights`, capped
    by `capacity`. Always returns an integer array summing to exactly `total`
    (provided sum(capacity) >= total and sum(weights) > 0).

    Algorithm: floor(total * w / sum(w)), clip to capacity, then distribute
    the remainder by largest fractional part among cells with headroom.
    """
    weights = np.asarray(weights, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.int64)
    n = len(weights)
    assert capacity.shape == weights.shape

    if int(capacity.sum()) <= total:
        # Capacity fully consumed.
        return capacity.astype(np.int64)

    if weights.sum() <= 0:
        # No signal to bias allocation; uniform across cells with capacity > 0.
        weights = (capacity > 0).astype(np.float64)
        if weights.sum() == 0:
            return np.zeros(n, dtype=np.int64)

    raw = total * weights / weights.sum()
    floor = np.floor(raw).astype(np.int64)
    floor = np.minimum(floor, capacity)
    deficit = total - int(floor.sum())
    if deficit > 0:
        headroom = capacity - floor
        # Sort cells by fractional part (desc); fill until deficit is closed.
        frac = raw - np.floor(raw)
        # Cells with no headroom can't take more.
        score = np.where(headroom > 0, frac, -np.inf)
        # Pick the top-`deficit` cells by score and add 1 each.
        # (Some cells may need >1; iterate the +1 distribution.)
        while deficit > 0:
            order = np.argsort(-score)
            take = min(deficit, int((score > -np.inf).sum()))
            if take == 0:
                break
            chosen = order[:take]
            floor[chosen] += 1
            deficit -= take
            # Update headroom & score for next pass.
            headroom = capacity - floor
            score = np.where(headroom > 0, frac, -np.inf)
    elif deficit < 0:
        # Over-allocated (shouldn't happen with floor + clip); shed.
        excess = -deficit
        order = np.argsort(np.where(floor > 0, raw - floor, np.inf))
        floor[order[:excess]] -= 1
    return floor


def within_voxel_topk(scores: np.ndarray, cell_ids: np.ndarray,
                       budget: np.ndarray) -> np.ndarray:
    """Pick top-`budget[c]` indices in each cell c by `scores`.

    Returns: 1-D index array of selected positions in `scores`/`cell_ids`,
    of length `sum(budget)`.
    """
    n = len(scores)
    # Sort by (cell_id ASC, score DESC).
    order = np.lexsort((-scores, cell_ids))
    sorted_cells = cell_ids[order]
    # Find each cell's start in the sorted array.
    diffs = np.r_[True, sorted_cells[1:] != sorted_cells[:-1]]
    cell_starts = np.flatnonzero(diffs)
    cell_unique = sorted_cells[cell_starts]   # unique cell ids in sorted order
    cell_ends = np.r_[cell_starts[1:], n]     # exclusive end of each cell

    out = []
    # `budget` is indexed by global cell id; map unique cells to their budgets.
    budget_per_cell = budget[cell_unique]
    cell_sizes = cell_ends - cell_starts
    take = np.minimum(budget_per_cell, cell_sizes)
    for s, t in zip(cell_starts, take):
        if t > 0:
            out.append(order[s:s + t])
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


# ---------------------------------------------------------------------------
# Sanity check — round-trip bin ↔ world position is consistent with
# build_polar_to_cart_grid's sin(az) and r conventions.
# ---------------------------------------------------------------------------

def _sanity_check_round_trip():
    rng = np.random.default_rng(0)
    rx = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    bs = rng.normal(size=3).astype(np.float32); bs /= np.linalg.norm(bs)
    grid = make_grid_specs(rx, bs, range_res=0.0593)
    # Pick random valid (r_bin, az_bin) pairs, forward map to world, bin back.
    for _ in range(200):
        rb = int(rng.integers(low=int(1.5 / grid.range_res) + 1, high=grid.n_range - 1))
        ab = int(rng.integers(low=1, high=grid.n_az - 1))
        p = cell_to_world_center(rb, ab, grid)
        rb2, ab2, v, _ = bin_points(p[None], grid)
        assert v[0], 'round-trip lost validity'
        assert int(rb2[0]) == rb, f'r_bin mismatch: {rb} vs {rb2[0]}'
        assert int(ab2[0]) == ab, f'az_bin mismatch: {ab} vs {ab2[0]}'
    print('  voxel_grid sanity check: 200/200 round-trips OK')


if __name__ == '__main__':
    _sanity_check_round_trip()
