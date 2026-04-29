"""Frame-level NVS trainer (v5 CUDA, pass-2 alignment).

Trains on all 16 chirp loops of N training frames (total 16·N RA maps),
holds out one test frame entirely, and evaluates on a single loop of the
test frame (default loop 0 — "first chirp"). The remaining 15 loops of
the test frame are discarded so the NVS test is a single held-out RA.

Pose synthesis:
  For each (frame, loop) pair we reuse the chirp-loop NVS pose
  interpolation (``build_per_loop_poses`` from train_chirp_loop_nvs):
  anchor on ``F-1`` and ``F+1`` aligned configs, interpolate per-loop
  alpha = 0.5 + k·loop_dt/(2·frame_period). Edge train frames missing a
  neighbour fall back to the frame's own aligned pose for all 16 loops.
  The test frame's loop-0 pose is interpolated between its own neighbours
  (test_frame-1 and test_frame+1), so the test pose is NVS-inferred —
  never taken from test_frame's own aligned config.

Usage (example):
    python -m mm25DGS_v5_v4.train_frame_nvs \
        --scene seq_1_frame_438 \
        --test_frame 438 \
        --train_frames 434,435,436,437,439,440,441,442 \
        --iters 500 \
        --use_pass2_alignment
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmir.data.ra_utils import adc_to_ra_complex
from mmir.data.io_utils import compute_range_res_from_cfg

from mm25DGS_v5_v4.rasterizer import Rasterizer, reparameterize_torch  # noqa: F401
from mm25DGS_v5 import cuda as v5cuda
from mm25DGS_v5_v4.train_gaussian import (
    DEVICE,
    init_visible_weighted,
    init_visible_weighted_radar_aware,
    cull_gaussians,
    render_gaussians,
    range_profile_to_ra_mag,
    build_polar_to_cart_grid,
    polar_to_cart_torch,
    cart_corr_torch,
    precompute_gt_loss_norm,
    compute_ra_loss_rp,
    get_lr_scale,
    rms_clip_grad,
    USE_FACTORY_PATTERNS,
)
from mm25DGS_v5_v4.load_pretrained import load_trained_config, load_pattern_data

# Reuse chirp-loop helpers (per-loop pose interpolation + pose applier + GT loader)
from mm25DGS_v5_v4.train_chirp_loop_nvs import (
    build_per_loop_poses, apply_pose, _load_pose_from_config,
)


# ---------------------------------------------------------------------------
# Per-scene data loading (train and test frames)
# ---------------------------------------------------------------------------

def _aligned_config_path(scene, frame, use_pass2, data_root):
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    suffix = '_aligned_pass2' if use_pass2 else '_aligned'
    p = os.path.join(align_dir, f'cascaded_frame_{frame}{suffix}.json')
    return p if os.path.isfile(p) else None


def _pass3_config_path(scene, frame, chirp, data_root):
    """Per-chirp Stage 3 config path (produced by
    ``per_chirp_alignment.run_stage3_for_scene``)."""
    return os.path.join(
        data_root, 'alignment_data', scene, 'cascade', 'per_chirp',
        f'cascaded_frame_{frame}_chirp{chirp:02d}_aligned_pass3.json')


def _build_frame_poses(scene, frame, use_pass2, data_root,
                       loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.1,
                       anchor_source='pass2_lerp',
                       device=DEVICE):
    """Return a list of 16 per-loop pose dicts for ``frame``.

    ``anchor_source``:
      * ``'pass2_lerp'`` (default) — LERP between pass-2 F-1 and F+1
        aligned configs at per-chirp α. This is the historical
        behaviour of the trainer.
      * ``'pass3_per_chirp'`` — load each (frame, chirp)'s own
        Stage 3 refined config directly from
        ``data/alignment_data/<scene>/cascade/per_chirp/
          cascaded_frame_<F>_chirp<CC>_aligned_pass3.json``.
        Falls back to ``pass2_lerp`` for any chirp whose Stage 3 file
        is missing.

    For edge train frames (no F-1 or F+1 under pass2_lerp), falls back
    to ``frame``'s own aligned pose for all 16 loops (sub-mm
    chirp-loop motion is negligible vs the frame-level motion we're
    training on).
    """
    # Pass-3 path: load per-chirp configs directly, one per (F, c).
    if anchor_source == 'pass3_per_chirp':
        poses = []
        fallback_used = False
        for c in range(16):
            cfg_p = _pass3_config_path(scene, frame, c, data_root)
            if not os.path.isfile(cfg_p):
                fallback_used = True
                break
            tx, rx, tb, rb = _load_pose_from_config(cfg_p, device=device)
            poses.append({
                'alpha': 0.5 + c * loop_dt_s / (2 * frame_period_s),
                'tx_positions':  tx.contiguous(),
                'rx_positions':  rx.contiguous(),
                'tx_boresights': tb.contiguous(),
                'rx_boresights': rb.contiguous(),
            })
        if not fallback_used:
            return poses, 'pass3_per_chirp'
        # else fall through to pass2_lerp

    # Pass-2 LERP path (default / fallback).
    cfg_A = _aligned_config_path(scene, frame - 1, use_pass2, data_root)
    cfg_B = _aligned_config_path(scene, frame + 1, use_pass2, data_root)
    if cfg_A is not None and cfg_B is not None:
        poses, _ = build_per_loop_poses(
            cfg_A, cfg_B, n_loops=16,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            device=device)
        return poses, 'interp_neighbours'

    cfg_F = _aligned_config_path(scene, frame, use_pass2, data_root)
    assert cfg_F is not None, (
        f'{scene} f={frame}: no aligned config (neither neighbour nor '
        f'self); cannot build poses')
    tx, rx, tb, rb = _load_pose_from_config(cfg_F, device=device)
    pose = {
        'alpha': 0.5,
        'tx_positions':  tx.contiguous(),
        'rx_positions':  rx.contiguous(),
        'tx_boresights': tb.contiguous(),
        'rx_boresights': rb.contiguous(),
    }
    return [pose] * 16, 'frame_own_pose'


def _build_per_loop_gt(adc_npy_path, loop_idx, loss_type, device):
    """Load a single (frame, loop) GT as a dict with 'gt_loss' +
    'gt_ra_polar' (magnitude). Caller converts polar→cart later once the
    sample_grid is built.
    """
    arr = np.load(adc_npy_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'expected cascaded ADC shape (16,RX,TX,K); got {arr.shape}')
    ri = np.stack([arr[loop_idx].real, arr[loop_idx].imag],
                  axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)  # (TX, RX, K, 2)
    gt_adc_ri = torch.from_numpy(ri).to(device)
    gt_loss = precompute_gt_loss_norm(gt_adc_ri, loss_type=loss_type)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc_ri)
        ra_mag = torch.abs(ra_c).float()
    return {'gt_loss': gt_loss, 'gt_ra_polar': ra_mag}


def build_frame_level_dataset(scene, train_frames, test_frame,
                               held_out_loop, use_pass2,
                               train_loops=None,
                               loss_type='mse_raw',
                               loop_dt_s=7.87e-3 / 16.0,
                               frame_period_s=0.1,
                               data_root='/home/adnan/Desktop/mm3DGS/data',
                               device=DEVICE,
                               anchor_source='pass2_lerp',
                               verbose=True):
    """Return (train_samples, test_sample, diagnostics).

    train_samples is a list of |train_loops|·|train_frames| dicts with keys:
        'frame_idx', 'loop_idx', 'pose', 'gt_loss', 'gt_ra_polar'
    test_sample is a single dict with the same keys (+ nothing under
    'gt_loss'; test uses cart_corr only).

    ``train_loops`` defaults to all 16 chirp loops. Pass e.g. ``[0]`` to
    train only on the first chirp loop of each train frame (the
    "first-chirp" NVS variant).  Note: "single-chip" in this codebase
    refers specifically to the TI IWR1443 3×4 single-chip radar
    hardware, which is a different sensor from the TI MMWCAS 12×16
    cascaded radar this trainer uses; do NOT confuse the two.
    """
    if train_loops is None:
        train_loops = list(range(16))
    train_loops = list(train_loops)

    scene_dir = os.path.join(data_root, scene)
    radar_dir = os.path.join(scene_dir, 'radar')

    train_samples = []
    edges = []  # frames that fell back to frame-own pose
    for f in train_frames:
        poses, mode = _build_frame_poses(
            scene, f, use_pass2=use_pass2, data_root=data_root,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            anchor_source=anchor_source, device=device)
        if mode != 'interp_neighbours':
            edges.append((f, mode))
        adc_npy = os.path.join(radar_dir, f'cascaded_frame_{f}.npy')
        for k in train_loops:
            gt = _build_per_loop_gt(adc_npy, k, loss_type, device)
            train_samples.append({
                'frame_idx': int(f), 'loop_idx': int(k),
                'pose': poses[k],
                **gt,
            })

    # Test: one (frame, loop); pose interpolated from test_frame's neighbours
    # (or loaded directly from per-chirp Stage 3 config under anchor_source='pass3_per_chirp').
    test_poses, test_mode = _build_frame_poses(
        scene, test_frame, use_pass2=use_pass2, data_root=data_root,
        loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
        anchor_source=anchor_source, device=device)
    test_adc = os.path.join(radar_dir, f'cascaded_frame_{test_frame}.npy')
    test_gt = _build_per_loop_gt(test_adc, held_out_loop, loss_type, device)
    test_sample = {
        'frame_idx': int(test_frame), 'loop_idx': int(held_out_loop),
        'pose': test_poses[held_out_loop],
        **test_gt,
    }

    diag = {
        'n_train_samples': len(train_samples),
        'edge_frames': edges,
        'test_pose_mode': test_mode,
    }
    if verbose:
        print(f'  [data] train_samples: {len(train_samples)} '
              f'({len(train_frames)} frames × 16 loops)')
        if edges:
            print(f'  [data] edge frames (fell back to own pose): {edges}')
        print(f'  [data] test (frame={test_frame} loop={held_out_loop})  '
              f'pose_mode={test_mode}')
    return train_samples, test_sample, diag


# ---------------------------------------------------------------------------
# S4 — adaptive density: Fisher-weighted split + prune, fixed N budget.
# ---------------------------------------------------------------------------

def _normal_to_quat_gpu(n):
    """GPU-side version of _normals_to_quaternions (axis–angle from +z→n).

    ``n``: (K, 3) unit normals on device.
    Returns: (K, 4) quaternion [w, x, y, z].
    """
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # axis = cross(+z, n) = (-n_y, n_x, 0); magnitude = sin(θ)
    axis = torch.zeros_like(n)
    axis[:, 0] = -n[:, 1]
    axis[:, 1] = n[:, 0]
    axis_len = axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    axis = axis / axis_len
    dot = n[:, 2]
    # half-angle from cos(θ)
    cos_half = torch.sqrt(((1.0 + dot) / 2.0).clamp(min=0.0))
    sin_half = torch.sqrt(((1.0 - dot) / 2.0).clamp(min=0.0))
    q = torch.zeros((n.shape[0], 4), dtype=n.dtype, device=n.device)
    q[:, 0] = cos_half                       # w
    q[:, 1:] = axis * sin_half.unsqueeze(-1) # x, y, z
    # Near-exact-opposite (dot ≈ −1) → 180° rotation; axis already in xy plane.
    # Above handles it via sin_half → 1, cos_half → 0.
    return torch.nn.functional.normalize(q, dim=-1)


def _pool_knn_assign(parent_pos, pool_xyz, pool_used, radius_m,
                     radius_min_m=0.0, selection='nearest',
                     topk=10, parent_batch=500):
    """For each split parent, pick an unused pool point within a radius
    range, using one of two selection strategies.

    ``selection``:
      * ``'nearest'`` (default): pick the nearest unused pool point within
        ``[0, radius_m)``. Tends to produce near-clone children when the
        pool is dense (sub-cm nearest neighbours often exist).
      * ``'random_annulus'``: pick a UNIFORM-RANDOM unused pool point
        within the annulus ``[radius_min_m, radius_m)``. Trades locality
        for diversity; matches the spatial scale of jitter (2-15 cm) but
        keeps children on real LiDAR surfaces.

    Greedy resolution of parent-parent collisions: parents are processed
    in a random order; each takes a point that is then locked out for
    subsequent parents.

    Returns (assigned: (K,) long, hit: (K,) bool). ``assigned[i] == -1``
    when no eligible neighbour exists (caller falls back to jitter).
    """
    K = parent_pos.shape[0]
    P = pool_xyz.shape[0]
    device = parent_pos.device

    assigned = torch.full((K,), -1, dtype=torch.long, device=device)
    used_this_round = pool_used.clone()
    order = torch.randperm(K, device=device).tolist()

    if selection == 'nearest':
        # Compute top-K nearest pool indices per parent (chunked for memory).
        topk_idx = torch.empty((K, topk), dtype=torch.long, device=device)
        topk_dist = torch.empty((K, topk), dtype=torch.float32, device=device)
        for i in range(0, K, parent_batch):
            j = min(i + parent_batch, K)
            d = torch.cdist(parent_pos[i:j], pool_xyz)           # (b, P)
            d = d.masked_fill(pool_used[None, :], float('inf'))
            v, ix = d.topk(topk, dim=-1, largest=False)
            topk_dist[i:j] = v
            topk_idx[i:j]  = ix
        for p_ord in order:
            for r in range(topk):
                cand = topk_idx[p_ord, r].item()
                if topk_dist[p_ord, r].item() > radius_m:
                    break
                if not used_this_round[cand]:
                    assigned[p_ord] = cand
                    used_this_round[cand] = True
                    break
    elif selection == 'random_annulus':
        # For each parent (in random order), sample uniformly among unused
        # pool points inside [radius_min_m, radius_m). Processed in chunks.
        for i in range(0, K, parent_batch):
            chunk = order[i:i + parent_batch]
            chunk_t = torch.tensor(chunk, device=device, dtype=torch.long)
            d = torch.cdist(parent_pos[chunk_t], pool_xyz)          # (b, P)
            valid = (d >= radius_min_m) & (d < radius_m)
            valid = valid & (~used_this_round[None, :])
            for k, p_ord in enumerate(chunk):
                vk = valid[k]
                if not vk.any():
                    continue
                vi = torch.nonzero(vk, as_tuple=True)[0]
                pick_idx = int(torch.randint(len(vi), (1,), device=device).item())
                pick = int(vi[pick_idx].item())
                assigned[p_ord] = pick
                used_this_round[pick] = True
                # lock out this pool index for subsequent parents in chunk
                valid[:, pick] = False
    else:
        raise ValueError(f'unknown pool selection: {selection}')

    hit = (assigned >= 0)
    return assigned, hit


def _densify_step(model, optimizer, fisher_per_pt, split_n, prune_n,
                  pos_jitter_m, mat_jitter, rot_jitter=0.05,
                  init_raw_ref=None, init_normals_ref=None,
                  child_source='pool_knn',
                  pool_xyz=None, pool_normals=None, slot_to_pool_idx=None,
                  pool_radius_m=0.15, pool_radius_min_m=0.0,
                  pool_selection='nearest', verbose=False):
    """In-place split + prune to reallocate the fixed N-point budget.

    The bottom ``prune_n`` points (lowest Fisher) are *replaced* in-place
    by children of the top ``split_n`` points (highest Fisher).

    ``child_source``:
      * ``'pool_knn'`` (default): draw the child's position + normal from
        the nearest-unused point in the post-visibility LiDAR pool within
        ``pool_radius_m``. Keeps children on real LiDAR surfaces and
        preserves genuine LiDAR normals. Requires ``pool_xyz``,
        ``pool_normals``, ``slot_to_pool_idx`` to be provided.
      * ``'jitter'``: legacy mode — child = parent params + Gaussian noise
        (position σ = ``pos_jitter_m``, rotation σ = ``rot_jitter``).
        Kept for ablation / fallback when the pool is unavailable.

    Child material is always parent + Gaussian jitter (``mat_jitter``);
    Adam can repair this quickly.

    Tensor indices of non-modified points are preserved (prune slots are
    overwritten, not deleted). Adam state at replaced slots is zeroed so
    children start with no momentum. ``init_raw_ref`` / ``init_normals_ref``
    are patched so drift penalties see zero drift at birth.

    Requires ``split_n == prune_n`` and both ≤ 0.45·N.
    """
    assert split_n == prune_n, 'split_n and prune_n must match to preserve N'
    N = model.N
    assert split_n <= 0.45 * N, 'split/prune frac too large; may overlap'

    device = model.positions.device
    _, order = fisher_per_pt.sort(descending=True)
    split_idx = order[:split_n].clone()
    prune_idx = order[-prune_n:].clone()

    overlap = set(split_idx.tolist()) & set(prune_idx.tolist())
    assert not overlap, f'split/prune index overlap: {overlap}'

    n_pool_hits = 0
    n_pool_fallback = 0

    with torch.no_grad():
        # Gather parent params
        pos_parent = model.positions[split_idx].clone()
        rot_parent = model.rotations[split_idx].clone()
        raw_parent = model.raw_materials[split_idx].clone()

        # --- Generate new (child) position + rotation ---
        if child_source == 'pool_knn':
            assert pool_xyz is not None and pool_normals is not None \
                and slot_to_pool_idx is not None, \
                'pool_knn mode requires pool_xyz, pool_normals, slot_to_pool_idx'
            pool_used = torch.zeros(pool_xyz.shape[0], dtype=torch.bool,
                                     device=device)
            pool_used[slot_to_pool_idx] = True

            assigned, hit = _pool_knn_assign(
                pos_parent, pool_xyz, pool_used,
                radius_m=pool_radius_m,
                radius_min_m=pool_radius_min_m,
                selection=pool_selection)

            # pool-sourced children for hits; jitter fallback for misses.
            pos_new = pos_parent.clone()
            rot_new = rot_parent.clone()
            pos_new[hit] = pool_xyz[assigned[hit]]
            rot_new[hit] = _normal_to_quat_gpu(pool_normals[assigned[hit]])

            miss = ~hit
            if miss.any():
                # Fallback: jitter for parents with no nearby unused pool pt.
                pos_new[miss] = pos_parent[miss] + \
                    torch.randn_like(pos_parent[miss]) * pos_jitter_m
                rot_new[miss] = torch.nn.functional.normalize(
                    rot_parent[miss] + torch.randn_like(rot_parent[miss]) * rot_jitter,
                    dim=-1)

            n_pool_hits = int(hit.sum().item())
            n_pool_fallback = int(miss.sum().item())

            # Update slot_to_pool_idx for the replaced (prune) slots
            # Only hits have a real pool idx; misses get -1 (untracked).
            new_pool_mapping = torch.full_like(prune_idx, -1)
            new_pool_mapping[hit] = assigned[hit]
            # Assign back into the slot-level map at prune slots
            slot_to_pool_idx[prune_idx] = new_pool_mapping

        else:  # 'jitter' (legacy)
            pos_new = pos_parent + torch.randn_like(pos_parent) * pos_jitter_m
            rot_new = torch.nn.functional.normalize(
                rot_parent + torch.randn_like(rot_parent) * rot_jitter, dim=-1)

        # Material always: parent + Gaussian noise
        raw_new = raw_parent + torch.randn_like(raw_parent) * mat_jitter

        # In-place overwrite of prune slots with children
        model.positions[prune_idx]     = pos_new
        model.rotations[prune_idx]     = rot_new
        model.raw_materials[prune_idx] = raw_new

        # Adam state surgery: zero momentum + velocity at replaced slots.
        for p in [model.positions, model.rotations, model.raw_materials]:
            st = optimizer.state.get(p)
            if st is None:
                continue
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st and st[k].shape == p.shape:
                    st[k][prune_idx] = 0.0

        # Patch init_* so drift penalties see zero drift at birth
        if init_raw_ref is not None:
            init_raw_ref[prune_idx] = raw_new
        if init_normals_ref is not None:
            q = torch.nn.functional.normalize(rot_new, dim=-1)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            nx = 2 * (x * z + w * y)
            ny = 2 * (y * z - w * x)
            nz = 1 - 2 * (x * x + y * y)
            init_normals_ref[prune_idx] = torch.stack([nx, ny, nz], dim=-1)

    if verbose:
        if child_source == 'pool_knn':
            sel_str = pool_selection
            if pool_selection == 'random_annulus':
                r_str = f'annulus=[{pool_radius_min_m:.2f}, {pool_radius_m:.2f}] m'
            else:
                r_str = f'radius={pool_radius_m:.2f} m'
            print(f'  [S4] densify: split {split_n} top-Fisher → replaced '
                  f'{prune_n} bottom-Fisher  '
                  f'[pool_knn {sel_str}: {n_pool_hits} hits / '
                  f'{n_pool_fallback} jitter-fb @ {r_str}]')
        else:
            print(f'  [S4] densify: split {split_n} top-Fisher → replaced '
                  f'{prune_n} bottom-Fisher  (pos_jitter={pos_jitter_m:.3f} m)')


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_frame_nvs(scene,
                    train_frames,
                    test_frame,
                    held_out_loop=0,
                    train_loops=None,
                    num_iters=500,
                    mat_lr=0.01,
                    rot_lr=5e-3,
                    loss_type='mse_raw',
                    target_n=90000,
                    frame_period_s=0.1,
                    loop_dt_s=7.87e-3 / 16.0,
                    use_pass2_alignment=True,
                    anchor_source='pass2_lerp',
                    data_root='/home/adnan/Desktop/mm3DGS/data',
                    verbose=True,
                    output_dir=None,
                    reg_l2_drift_lambda=0.0,
                    reg_active_top_frac=None,
                    reg_warm_iters=50,
                    reg_fisher_rot_lambda=0.0,
                    reg_fisher_rot_target='init',
                    reg_fisher_rot_ema_alpha=0.95,
                    reg_fisher_rot_threshold_deg=0.0,
                    reg_densify_interval=0,
                    reg_densify_split_frac=0.02,
                    reg_densify_prune_frac=0.02,
                    reg_densify_until=300,
                    reg_densify_pos_jitter_m=0.02,
                    reg_densify_mat_jitter=0.1,
                    reg_densify_child_source='pool_knn',
                    reg_densify_pool_radius_m=0.15,
                    reg_densify_pool_radius_min_m=0.0,
                    reg_densify_pool_selection='nearest',
                    seed_frame=None,
                    init_variant='baseline',  # v5_v4 Phase 2: A1..A5 or 'baseline'
                    densify_signal='fisher',  # v5_v4 Phase 3: 'fisher' (legacy)
                                              # | 'pos_grad_amp' (Phase 3)
                    learn_positions_lr=0.0,   # v5_v2 Phase 1 carryover: > 0 → enable
                                              # learnable positions, AMPLITUDE GRADIENT
                                              # ONLY (phase remains detached). Sub-mm
                                              # bounded; L2 anchor to LiDAR init.
                    learn_positions_l2=1e3,   # L2 anchor coefficient on (pos − init)
                    phase4_d_lambda=0.0,      # v5_v4 Phase 4 (D): > 0 enables per-train-view
                                              # membership. sigmoid(m[:, v]) is opacity per view.
                                              # λ is the L1 sparsity on sigmoid(m).
                    phase4_d_test_knn=2,     # at test, average sigmoid(m[:, v]) over the K
                                              # nearest train views by 6-DoF pose distance.
                    mlp_a_lr=0.0,
                    mlp_a_hidden_dim=64,
                    mlp_a_n_layers=3,
                    mlp_a_max_dpos_m=0.05,
                    mlp_a_max_dalpha=1.0,
                    mlp_a_l2_dpos=100.0,
                    mlp_a_l1_dalpha=0.01,
                    mlp_a_warmup_iters=50,
                    mlp_a_pose_pe_freqs=0):
    assert v5cuda.is_available(), (
        'v5 CUDA extension not built. '
        'cd mm25DGS_v5_v4/cuda && python setup.py build_ext --inplace')

    if train_loops is None:
        train_loops = list(range(16))
    train_loops = list(train_loops)
    test_in_train = int(test_frame) in [int(f) for f in train_frames]

    if verbose:
        print('=' * 72)
        print(f'frame-level NVS  scene={scene}')
        print(f'  train frames ({len(train_frames)}): {train_frames}')
        print(f'  train loops  ({len(train_loops)}): {train_loops}')
        print(f'  test  frame  = {test_frame}  (test loop={held_out_loop})'
              + ('   [test frame ALSO in train_frames → upper-bound run]'
                 if test_in_train else ''))
        print(f'  alignment    = {"pass-2" if use_pass2_alignment else "pass-1"}')
        print(f'  iters        = {num_iters}   loss_type={loss_type}')
        print('=' * 72)

    # ── paths ──
    config = load_trained_config(scene)

    # The seed frame's pose drives FOV culling + RX visibility before FPS
    # subsampling, so it determines *which* 90 000-point subset of pcl.npy
    # this run trains on. Default = `test_frame` so HO and UB variants share
    # an identical position grid (UB's middle-of-train-frames index already
    # lands on test_frame; HO's previously landed on `test_frame+1`,
    # producing two different ~90k subsets — see md/frame_nvs_analysis/
    # findings.md §0). Pass `seed_frame=<other>` to override (e.g. to
    # reproduce the legacy HO grid use `seed_frame=test_frame+1`).
    if seed_frame is None:
        seed_frame = int(test_frame)
    suffix = '_aligned_pass2' if use_pass2_alignment else '_aligned'
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    seed_cfg = os.path.join(align_dir,
                            f'cascaded_frame_{seed_frame}{suffix}.json')
    if verbose:
        print(f'  seed frame   = {seed_frame}  '
              f'(drives FOV+visibility before FPS → position grid)')
    assert os.path.exists(seed_cfg), f'missing: {seed_cfg}'

    # ── Rasterizer + model ──
    rast = Rasterizer(
        config_file=seed_cfg,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE)
    if not USE_FACTORY_PATTERNS:
        rast.inject_trained_params(pattern_data=load_pattern_data(scene))

    # Init model at seed pose — FOV will be recomputed per-sample via
    # cull_gaussians during render_gaussians. We need a Mitsuba scene for
    # the init-time visibility test.
    # If S4 pool-kNN mode is in use, also return the post-visibility
    # LiDAR pool (xyz, normals, fps_sel) so children can be drawn from
    # real on-surface unused points. The pool is the un-resampled /
    # un-FPS-reduced set of LiDAR points that survived FOV + visibility;
    # typically several hundred thousand candidates per scene.
    need_pool = (reg_densify_interval > 0
                  and reg_densify_child_source == 'pool_knn')
    if init_variant == 'baseline':
        if need_pool:
            model, pool_xyz_np, pool_normals_np, fps_sel_np = \
                init_visible_weighted(scene, rast, target_n=target_n,
                                       return_pool=True)
            pool_xyz = torch.from_numpy(pool_xyz_np.astype(np.float32)).to(DEVICE)
            pool_normals = torch.from_numpy(pool_normals_np.astype(np.float32)).to(DEVICE)
            slot_to_pool_idx = torch.from_numpy(fps_sel_np.astype(np.int64)).to(DEVICE)
            if verbose:
                print(f'  [S4] pool_knn enabled: post-visibility pool size = '
                      f'{pool_xyz.shape[0]:,} points  '
                      f'(radius={reg_densify_pool_radius_m:.2f} m)')
        else:
            model = init_visible_weighted(scene, rast, target_n=target_n)
            pool_xyz = pool_normals = slot_to_pool_idx = None
    else:
        # v5_v4 Phase 2 — radar-aware init.
        # Build chirp-0 pose dicts for all train frames (used for the
        # union-amplitude scoring across train poses).
        train_poses_chirp0 = []
        for f in train_frames:
            poses, _ = _build_frame_poses(
                scene, f, use_pass2=use_pass2_alignment, data_root=data_root,
                loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
                anchor_source=anchor_source, device=DEVICE)
            train_poses_chirp0.append(poses[0])  # chirp 0 only
        if verbose:
            print(f'  [v5_v4] init variant: {init_variant}  '
                  f'over {len(train_poses_chirp0)} train poses')
        # B2 / C1 / C1b / C2 need the test pose. Test POSE (geometry only)
        # is permitted under NVS conventions; test SIGNAL is never used for
        # init.
        test_pose_chirp0 = None
        if init_variant in ('B2_strict_and_with_test', 'C1_voxel_v1',
                             'C1b_voxel_capped', 'C2_voxel_v2'):
            test_poses_b2, _ = _build_frame_poses(
                scene, test_frame, use_pass2=use_pass2_alignment,
                data_root=data_root, loop_dt_s=loop_dt_s,
                frame_period_s=frame_period_s,
                anchor_source=anchor_source, device=DEVICE)
            test_pose_chirp0 = test_poses_b2[0]
        # C2 also needs the 8 train RA polar magnitudes (chirp 0).
        train_ra_mag_list = None
        if init_variant == 'C2_voxel_v2':
            radar_dir_c2 = os.path.join(data_root, scene, 'radar')
            train_ra_mag_list = []
            for f in train_frames:
                adc_npy = os.path.join(radar_dir_c2, f'cascaded_frame_{f}.npy')
                gt = _build_per_loop_gt(adc_npy, 0, loss_type, DEVICE)
                train_ra_mag_list.append(gt['gt_ra_polar'])
            if verbose:
                print(f'  [C2] loaded {len(train_ra_mag_list)} train RA polar maps '
                      f'(shape {tuple(train_ra_mag_list[0].shape)})')
        model = init_visible_weighted_radar_aware(
            scene, rast,
            train_poses_chirp0=train_poses_chirp0,
            target_n=target_n,
            variant=init_variant,
            verbose=verbose,
            test_pose_chirp0=test_pose_chirp0,
            train_ra_mag_list=train_ra_mag_list,
        )
        pool_xyz = pool_normals = slot_to_pool_idx = None
    rast.free_mi_scene()
    import gc; gc.collect(); torch.cuda.empty_cache()

    # FOV mask for the seed pose — reused as a conservative initial mask.
    # Per-sample rendering re-masks via cull_gaussians if needed.
    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    # ── polar → cart grid ──
    range_res = compute_range_res_from_cfg(seed_cfg)
    sample_grid = build_polar_to_cart_grid(127, 256, range_res, 400, DEVICE)

    # ── Load train + test data ──
    train_samples, test_sample, diag = build_frame_level_dataset(
        scene=scene, train_frames=train_frames, test_frame=test_frame,
        held_out_loop=held_out_loop, use_pass2=use_pass2_alignment,
        train_loops=train_loops,
        loss_type=loss_type, loop_dt_s=loop_dt_s,
        frame_period_s=frame_period_s, data_root=data_root,
        anchor_source=anchor_source,
        device=DEVICE, verbose=verbose,
    )

    # Finalize GT cart norms for cart_corr evaluation
    # v5_v4 Phase 4 (D) closures — initialise BEFORE _render_and_cart_corr
    # is defined (the closure captures these names; they get filled in
    # below in the optimizer-setup block when phase4_d_lambda > 0).
    phase4_m = None
    train_view_index = {}    # frame_idx → v ∈ [0, V)
    train_view_centers = []  # list of (3,) tensors
    # MLP-A closures — same hoist pattern.
    mlp_a = None
    pose6d_per_sample = {}
    pos_local_seed = None
    R_radar2world_seed_t = None

    def _finalize_cart(sample):
        with torch.no_grad():
            ra_cart = polar_to_cart_torch(sample['gt_ra_polar'], sample_grid)
            mn, mx = ra_cart.min(), ra_cart.max()
            sample['gt_cart_norm'] = (
                (ra_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()
    for s in train_samples:
        _finalize_cart(s)
    _finalize_cart(test_sample)

    # ── Initial (pre-training) test cart_corr ──
    def _knn_membership_for_pose(pose_dict):
        """v5_v4 Phase 4 — KNN-over-train-views deployment.
        For a (test) pose, average sigmoid(m[:, v]) over the K nearest
        train views by 6-DoF (rx-centroid) Euclidean distance.
        Returns a (N,) tensor in [0, 1] or None if Phase 4 inactive.
        """
        if phase4_m is None or len(train_view_centers) == 0:
            return None
        with torch.no_grad():
            test_rxc = pose_dict['rx_positions'].mean(dim=0)
            tv_centers = torch.stack(train_view_centers, dim=0).to(test_rxc.device)
            d2 = ((tv_centers - test_rxc) ** 2).sum(dim=-1)
            K = min(int(phase4_d_test_knn), len(train_view_centers))
            _, knn_idx = torch.topk(-d2, k=K)
            sig_m = torch.sigmoid(phase4_m)        # (N, V)
            return sig_m[:, knn_idx].mean(dim=-1)  # (N,)

    def _render_and_cart_corr(sample):
        apply_pose(rast, sample['pose'])
        with torch.no_grad():
            # Phase 4 — KNN-deploy the per-view membership at test pose.
            v_idx = train_view_index.get(int(sample['frame_idx']))
            if phase4_m is not None and v_idx is not None:
                # train sample: use its own m[:, v]
                areas = vertex_areas * torch.sigmoid(phase4_m[:, v_idx])
            elif phase4_m is not None:
                # test (or held-out) sample: KNN-average
                m_knn = _knn_membership_for_pose(sample['pose'])
                areas = vertex_areas if m_knn is None else (vertex_areas * m_knn)
            else:
                areas = vertex_areas
            # MLP-A — apply pose-conditioned deformation at this sample's
            # pose. Train samples reuse their cached pose_6d; the test
            # sample uses its own. Test pose's geometry is permitted.
            pos_override = None
            if mlp_a is not None:
                if int(sample['frame_idx']) in pose6d_per_sample:
                    pose6d_s = pose6d_per_sample[int(sample['frame_idx'])]
                else:
                    # test sample (or any out-of-cache pose): re-encode now.
                    from .mlp_a import encode_pose_from_dict
                    rx_seed_np = rast.rx_positions.mean(dim=0).detach().cpu().numpy()
                    bs_seed_np = rast.tx_boresights.mean(dim=0).detach().cpu().numpy()
                    bs_seed_np = bs_seed_np / max(np.linalg.norm(bs_seed_np), 1e-12)
                    from .voxel_grid import _rodrigues_align
                    R = _rodrigues_align(bs_seed_np,
                                          np.array([0.0, 1.0, 0.0],
                                                    dtype=np.float32))
                    pose6d_s = encode_pose_from_dict(
                        sample['pose'], rx_seed_np, R)
                dp_local, dalpha = mlp_a(pose6d_s, pos_local_seed)
                dp_world = dp_local @ R_radar2world_seed_t.T
                pos_override = model.positions + dp_world
                areas = areas * torch.clamp(1.0 + dalpha, min=0.0, max=2.0)
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None,
                positions_override=pos_override)
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            cc = cart_corr_torch(ra_cart, sample['gt_cart_norm']).item()
        return float(cc)

    init_test_cc = _render_and_cart_corr(test_sample)
    init_train_ccs = [_render_and_cart_corr(s) for s in train_samples]
    if verbose:
        print(f'\n  [init] test  cc = {init_test_cc:.4f}')
        print(f'  [init] train mean cc = {np.mean(init_train_ccs):.4f} '
              f'(std {np.std(init_train_ccs):.4f}) over {len(init_train_ccs)} samples')

    # ── Optimizer ──
    param_groups = [
        {'params': [model.raw_materials], 'lr': mat_lr, 'name': 'materials'},
        {'params': [model.rotations],     'lr': rot_lr, 'name': 'rotations'},
    ]
    clip_vals = {'materials': 1.0, 'rotations': 0.5}
    # v5_v2 Phase 1 carryover — learnable positions, AMPLITUDE GRADIENT ONLY.
    # detach_phase=True (v5 default) ensures position gradient flows ONLY
    # through alpha_tx, BSDF cos terms, antenna gain — not through
    # phi_carrier or n_peak. NO PHASE GRADIENTS, by construction.
    if learn_positions_lr > 0.0:
        model.positions.requires_grad_(True)
        param_groups.append({'params': [model.positions], 'lr': learn_positions_lr,
                              'name': 'positions'})
        # Tight clip — bounded sub-mm motion. Phase wraps every 0.97 mm at
        # 77 GHz so we want movement << 1 mm per step.
        clip_vals['positions'] = 0.001  # 1 mm/step grad-norm cap
        if verbose:
            print(f'  [v5_v2 P1] learn_positions_lr={learn_positions_lr:g} '
                  f'(amplitude-path only, L2 anchor λ={learn_positions_l2:g})')
    # v5_v4 Phase 4 (D) — populate the closures above when enabled.
    if phase4_d_lambda > 0.0:
        V = len(train_frames)
        for v_idx, f in enumerate(train_frames):
            train_view_index[int(f)] = v_idx
        # Build per-view radar-centroid positions for KNN test deployment.
        # We use the chirp-0 pose's mean rx/tx position.
        for s in train_samples:
            if s['loop_idx'] == 0:
                rxc = s['pose']['rx_positions'].mean(dim=0)  # (3,)
                if int(s['frame_idx']) in train_view_index:
                    train_view_centers.append(rxc.detach().clone())
        phase4_m = torch.nn.Parameter(
            torch.zeros(model.N, V, device=DEVICE))
        param_groups.append({'params': [phase4_m], 'lr': mat_lr, 'name': 'membership'})
        clip_vals['membership'] = 1.0
        if verbose:
            print(f'  [v5_v4 P4] phase4_d_lambda={phase4_d_lambda:g}  '
                  f'V={V} train views; sigmoid(m[:, v]) opacity multiplier '
                  f'(KNN K={phase4_d_test_knn} for test)')
    # MLP-A — pose-conditioned per-Gaussian deformation MLP.
    # mlp_a, pose6d_per_sample, pos_local_seed, R_radar2world_seed_t are
    # already None / empty from the closure-hoist above.
    test_pose6d = None
    if mlp_a_lr > 0.0:
        from .mlp_a import (
            PoseConditionedDeformationMLP, encode_pose_from_dict,
        )
        from .voxel_grid import _rodrigues_align
        # Seed-pose proxies (same as the rasterizer was built from).
        rx_seed = rast.rx_positions.mean(dim=0).detach().cpu().numpy()
        bs_seed = rast.tx_boresights.mean(dim=0).detach().cpu().numpy()
        bs_seed = bs_seed / max(np.linalg.norm(bs_seed), 1e-12)
        R_w2r_seed = _rodrigues_align(bs_seed,
                                       np.array([0.0, 1.0, 0.0],
                                                 dtype=np.float32))
        # Cache (radar→world) = R_w2r.T for converting MLP outputs back.
        R_radar2world_seed_t = torch.from_numpy(
            R_w2r_seed.T.astype(np.float32)).to(model.positions)
        # Gaussian positions in seed-pose radar local frame.
        with torch.no_grad():
            R_t = torch.from_numpy(R_w2r_seed.astype(np.float32)).to(model.positions)
            rx_seed_t = torch.from_numpy(rx_seed.astype(np.float32)).to(model.positions)
            pos_local_seed = (model.positions - rx_seed_t) @ R_t.T  # (N, 3)
        # Build the MLP.
        mlp_a = PoseConditionedDeformationMLP(
            hidden_dim=int(mlp_a_hidden_dim),
            n_layers=int(mlp_a_n_layers),
            max_dpos_m=float(mlp_a_max_dpos_m),
            max_dalpha=float(mlp_a_max_dalpha),
            pose_pe_freqs=int(mlp_a_pose_pe_freqs),
        ).to(DEVICE)
        # Pre-compute pose_6d for every train sample's pose + the test pose.
        # Caches keyed by frame_idx (chirp 0 only, matches train_loops=[0]).
        for s in train_samples:
            if int(s['frame_idx']) not in pose6d_per_sample:
                pose6d_per_sample[int(s['frame_idx'])] = encode_pose_from_dict(
                    s['pose'], rx_seed, R_w2r_seed)
        test_pose6d = encode_pose_from_dict(
            test_sample['pose'], rx_seed, R_w2r_seed)
        # Add MLP params to optimizer.
        param_groups.append({
            'params': list(mlp_a.parameters()),
            'lr': float(mlp_a_lr),
            'name': 'mlp_a',
        })
        clip_vals['mlp_a'] = 0.5
        if verbose:
            print(f'  [MLP-A] enabled  lr={mlp_a_lr:g}  hidden={mlp_a_hidden_dim}'
                  f' × {mlp_a_n_layers}  max_dpos={mlp_a_max_dpos_m:g}m  '
                  f'max_dalpha={mlp_a_max_dalpha:g}  warmup={mlp_a_warmup_iters}')
            print(f'  [MLP-A] pre-computed pose_6d for {len(pose6d_per_sample)} '
                  f'train poses + 1 test pose.')

    base_lrs = {g['name']: g['lr'] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    loss_scale = 1.0 / len(train_samples)

    # ── Regularisation setup ──
    # Snapshot init raw_materials (uniform ITU_CONCRETE across N at init time).
    init_raw = model.raw_materials.detach().clone()
    # Snapshot init surface normals (from the quaternion init computed from
    # pcl.npy LiDAR normals). Needed for S2's rotation-drift penalty.
    init_normals = model.get_normals().detach().clone()  # (N, 3)
    # v5_v2 Phase 1 — snapshot init positions for L2 anchor against drift.
    init_positions = model.positions.detach().clone()  # (N, 3)
    active_pts_mask = None
    # S2: per-point Fisher weights on rotations, snapshot at warmup using
    # Adam's exp_avg_sq. Stays None until reg_warm_iters.
    fisher_rot = None
    # S2 Option 2: EMA of normals during training (rolling target).
    normal_ema = None
    # S4: adaptive density (split/prune) bookkeeping
    n_densify_rounds = 0
    # v5_v4 Phase 3 — position-gradient-magnitude accumulator for densify
    # selection. When densify_signal='pos_grad_amp', positions are kept frozen
    # (NOT in optimizer) but get .requires_grad_(True) so backward() populates
    # .grad through the AMPLITUDE PATH ONLY (phase path is detached as in v5).
    # We accumulate ‖∇p L‖ across iters between densify steps and use it in
    # place of the Fisher proxy.
    pos_grad_accum = None
    # Diagnostics: a never-reset accumulator capturing ‖∇p L‖ across the
    # entire training run, plus per-densify concentration snapshots
    # (top-k cumulative fraction). Saved at end-of-training for offline
    # Fisher-imbalance analysis. Always populated when pos_grad_amp is
    # the signal; cheap (N×4 bytes ≈ 80 KB at N=20k).
    career_pos_grad = None
    grad_concentration_log = []
    if densify_signal == 'pos_grad_amp' and reg_densify_interval > 0:
        model.positions.requires_grad_(True)
        pos_grad_accum = torch.zeros(model.N, device=DEVICE)
        career_pos_grad = torch.zeros(model.N, device=DEVICE)
        if verbose:
            print(f'  [v5_v4 Phase 3] densify_signal=pos_grad_amp; positions '
                  f'requires_grad=True (NOT in optimizer; selection only)')
    reg_enabled = (reg_l2_drift_lambda > 0.0) or (reg_active_top_frac is not None) \
        or (reg_fisher_rot_lambda > 0.0) or (reg_densify_interval > 0)
    reg_rot_thresh_rad = float(reg_fisher_rot_threshold_deg) * np.pi / 180.0
    if verbose and reg_enabled:
        print(f'  [reg] l2_drift_lambda={reg_l2_drift_lambda}  '
              f'active_top_frac={reg_active_top_frac}  '
              f'fisher_rot_lambda={reg_fisher_rot_lambda}  '
              f'warm_iters={reg_warm_iters}')
        if reg_fisher_rot_lambda > 0.0:
            print(f'  [reg] fisher_rot target={reg_fisher_rot_target}  '
                  f'ema_alpha={reg_fisher_rot_ema_alpha:.3f}  '
                  f'threshold_deg={reg_fisher_rot_threshold_deg}')

    # ── Training loop (full-batch: every sample every iter) ──
    best_mean_train_cc = -1.0
    best_iter = 0
    best_state = None
    history = []
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        per_sample_cc = []
        loss_sum = 0.0
        for s in train_samples:
            apply_pose(rast, s['pose'])
            # v5_v4 Phase 4 (D) — per-train-view opacity multiplier.
            # vertex_areas is the FOV mask (N-vector binary). When phase4 is
            # active, multiply by sigmoid(m[:, v]) where v is this train
            # frame's view index. Goes through render → backward to update m.
            if phase4_m is not None:
                v_idx = train_view_index.get(int(s['frame_idx']))
                if v_idx is not None:
                    areas_s = vertex_areas * torch.sigmoid(phase4_m[:, v_idx])
                else:
                    areas_s = vertex_areas
            else:
                areas_s = vertex_areas
            # MLP-A — pose-conditioned per-Gaussian deformation. After warmup,
            # query the MLP with the sample's pose_6d + base position to get
            # (Δp, Δα). Inject Δp via positions_override, modulate Δα onto
            # vertex_areas. AMPLITUDE-PATH ONLY (detach_phase=True is enforced
            # downstream).
            pos_override_s = None
            mlp_dalpha_s = None
            mlp_dp_local_s = None
            if mlp_a is not None and it >= mlp_a_warmup_iters:
                pose6d_s = pose6d_per_sample.get(int(s['frame_idx']))
                if pose6d_s is not None:
                    dp_local, dalpha = mlp_a(pose6d_s, pos_local_seed)
                    mlp_dp_local_s = dp_local
                    mlp_dalpha_s = dalpha
                    # Rotate Δp from seed-radar local frame back to world.
                    dp_world = dp_local @ R_radar2world_seed_t.T
                    pos_override_s = model.positions + dp_world
                    # Multiplicative opacity modulation: clamp(1 + Δα, [0, 2]).
                    areas_s = areas_s * torch.clamp(1.0 + dalpha, min=0.0, max=2.0)
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=areas_s,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None,
                positions_override=pos_override_s)
            loss_k, _ = compute_ra_loss_rp(
                rp_real, rp_imag, s['gt_loss'], loss_type=loss_type)
            # MLP-A regularisation: keep Δp small (L2 anchor), Δα sparse (L1).
            # Folded into the per-sample loss so we backward exactly once.
            if mlp_dp_local_s is not None:
                if mlp_a_l2_dpos > 0.0:
                    loss_k = loss_k + mlp_a_l2_dpos * mlp_dp_local_s.pow(2).mean()
                if mlp_a_l1_dalpha > 0.0:
                    loss_k = loss_k + mlp_a_l1_dalpha * mlp_dalpha_s.abs().mean()
            (loss_k * loss_scale).backward()
            with torch.no_grad():
                ra_polar = range_profile_to_ra_mag(rp_real.detach(), rp_imag.detach())
                ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                per_sample_cc.append(cc)
            loss_sum += loss_k.item() * loss_scale
            del rp_real, rp_imag, ra_polar, ra_cart, loss_k

        # v5_v4 Phase 3 — capture full-batch position-gradient magnitude
        # AFTER all train_samples backward(). positions.grad has been
        # accumulated across all samples (PyTorch default). We only use
        # the magnitude as a densify-selection signal; positions are NOT
        # in the optimizer (they never move).
        if pos_grad_accum is not None and model.positions.grad is not None:
            with torch.no_grad():
                step_grad = model.positions.grad.detach().norm(dim=-1)
                pos_grad_accum += step_grad
                if career_pos_grad is not None:
                    career_pos_grad += step_grad

        # H1 reg: L2 drift from init, mean over all elements (N·6).
        if reg_l2_drift_lambda > 0.0:
            drift = model.raw_materials - init_raw
            reg_loss = reg_l2_drift_lambda * drift.pow(2).mean()
            reg_loss.backward()
            loss_sum += float(reg_loss.item())

        # v5_v2 Phase 1 — L2 anchor on positions when learnable.
        # Phase wraps every 0.97 mm at 77 GHz; we want positions to drift
        # only sub-mm at most. λ ~ 1e3 keeps ‖Δp‖ in O(0.1 mm).
        if learn_positions_lr > 0.0 and learn_positions_l2 > 0.0:
            pos_drift = model.positions - init_positions
            reg_pos = learn_positions_l2 * pos_drift.pow(2).mean()
            reg_pos.backward()
            loss_sum += float(reg_pos.item())

        # v5_v4 Phase 4 (D) — L1 sparsity on sigmoid(m). Encourages each
        # point to be "owned by" few views (rather than 1 in all views).
        if phase4_m is not None and phase4_d_lambda > 0.0:
            sig_m = torch.sigmoid(phase4_m)
            reg_m = phase4_d_lambda * sig_m.mean()
            reg_m.backward()
            loss_sum += float(reg_m.item())

        # S2 reg: Fisher-weighted penalty on rotation drift from a target.
        # Target is either `init_normals` (Option 1) or an EMA of the
        # trajectory (Option 2 — reg_fisher_rot_target='ema'). Penalty is
        # either plain ||n − target||² or a Huber-style thresholded form
        # (Option 3 — reg_fisher_rot_threshold_deg > 0: penalise only
        # drift beyond threshold). Fisher weight F_i = Adam exp_avg_sq on
        # rotations (normalised to max = 1), snapshotted at warmup.
        if reg_fisher_rot_lambda > 0.0 and fisher_rot is not None:
            current_normals = model.get_normals()           # (N, 3) diff'ble
            target = normal_ema if reg_fisher_rot_target == 'ema' \
                                   and normal_ema is not None else init_normals
            if reg_rot_thresh_rad > 0.0:
                # Option 3: angle-thresholded. Penalise excess over θ₀.
                cos = (current_normals * target).sum(dim=-1).clamp(-1 + 1e-7,
                                                                    1 - 1e-7)
                angle = torch.acos(cos.abs())               # [0, π/2]
                excess = (angle - reg_rot_thresh_rad).clamp(min=0.0)
                drift_sq = excess.pow(2)                    # scalar per pt
            else:
                drift_sq = (current_normals - target).pow(2).sum(dim=-1)
            reg_loss_rot = reg_fisher_rot_lambda * (fisher_rot * drift_sq).sum()
            reg_loss_rot.backward()
            loss_sum += float(reg_loss_rot.item())

        # LR schedule (same as chirp-loop trainer)
        lr_scale = get_lr_scale(
            it, total_iters=num_iters,
            warmup_iters=0, warmup_factor=1.0, decay_start=200)
        for g in optimizer.param_groups:
            g['lr'] = base_lrs[g['name']] * lr_scale

        for g in optimizer.param_groups:
            clip = clip_vals.get(g['name'], 1.0)
            for p in g['params']:
                rms_clip_grad(p, clip)

        # H7 reg: snapshot a per-point training-Fisher proxy (Adam's
        # exp_avg_sq summed across columns) at warm-up, then zero gradients
        # on the bottom-(1-frac) points for the rest of training.
        if (reg_active_top_frac is not None
                and reg_active_top_frac > 0.0
                and it == reg_warm_iters):
            v = optimizer.state[model.raw_materials].get('exp_avg_sq')
            assert v is not None, 'Adam state missing at warmup boundary'
            per_pt_v = v.detach().sum(dim=-1)
            keep_n = int(reg_active_top_frac * per_pt_v.shape[0])
            threshold = per_pt_v.sort(descending=True).values[keep_n - 1]
            active_pts_mask = per_pt_v >= threshold
            if verbose:
                print(f'  [reg] iter {it}: activity mask — keeping '
                      f'{int(active_pts_mask.sum())}/{model.N} points '
                      f'(top {reg_active_top_frac*100:.1f}% by Adam v)')

        # S2 reg: snapshot per-point rotation Fisher (Adam exp_avg_sq
        # summed across 4 quaternion components) at warmup, normalise to
        # max=1. Used by the Fisher-weighted rotation drift penalty above.
        if (reg_fisher_rot_lambda > 0.0 and fisher_rot is None
                and it == reg_warm_iters):
            v = optimizer.state[model.rotations].get('exp_avg_sq')
            assert v is not None, 'Adam state missing for rotations at warmup'
            per_pt_v = v.detach().sum(dim=-1)                # (N,)
            denom = per_pt_v.max().clamp(min=1e-30)
            fisher_rot = (per_pt_v / denom).detach()         # (N,) in [0, 1]
            if verbose:
                srt = fisher_rot.sort(descending=True).values
                top_1 = int(0.01 * len(fisher_rot))
                top_5 = int(0.05 * len(fisher_rot))
                print(f'  [reg] iter {it}: rotation Fisher snapshot — '
                      f'top-1% mass={srt[:top_1].sum().item():.3f}  '
                      f'top-5% mass={srt[:top_5].sum().item():.3f}  '
                      f'(of total mass={fisher_rot.sum().item():.3f})')

        if active_pts_mask is not None:
            with torch.no_grad():
                if model.raw_materials.grad is not None:
                    model.raw_materials.grad[~active_pts_mask] = 0.0
                if model.rotations.grad is not None:
                    model.rotations.grad[~active_pts_mask] = 0.0

        optimizer.step()

        # S2 Option 2: update the normal EMA *after* the step. Initialised
        # to the snapshotted post-warmup normals so the target begins
        # where the optimiser settled during data-driven warmup.
        if (reg_fisher_rot_lambda > 0.0
                and reg_fisher_rot_target == 'ema'
                and fisher_rot is not None):
            with torch.no_grad():
                n_now = model.get_normals().detach()
                if normal_ema is None:
                    normal_ema = n_now.clone()
                else:
                    a = float(reg_fisher_rot_ema_alpha)
                    normal_ema = a * normal_ema + (1.0 - a) * n_now

        # S4: adaptive density — split top-Fisher + prune bottom-Fisher.
        # Triggered at iter = reg_warm_iters and every reg_densify_interval
        # iters thereafter, up to reg_densify_until. Fisher signal combines
        # Adam exp_avg_sq on raw_materials and rotations (per-point total).
        if (reg_densify_interval > 0
                and it >= reg_warm_iters
                and it <= reg_densify_until
                and (it - reg_warm_iters) % reg_densify_interval == 0):
            # Build the per-point selection signal `f_comb`. Two sources:
            #   'fisher' (legacy, default) — Adam exp_avg_sq fused across
            #     material + rotation params, max-normalised then summed.
            #   'pos_grad_amp' (v5_v4 Phase 3) — accumulated amplitude-path
            #     position gradient magnitude since the last densify step.
            f_comb = None
            if densify_signal == 'pos_grad_amp':
                if pos_grad_accum is not None and pos_grad_accum.max() > 0:
                    f_comb = pos_grad_accum.clone()  # (N,)
                    # Diagnostic: cumulative top-k fraction of total ‖∇p L‖
                    # in this densify window. Tells us how concentrated the
                    # signal is among a few points (Fisher-imbalance probe).
                    with torch.no_grad():
                        sg, _ = f_comb.sort(descending=True)
                        total = sg.sum().clamp(min=1e-30)
                        cumsum = (sg.cumsum(0) / total)
                        Nm = int(model.N)
                        snap = {'iter': int(it)}
                        parts = []
                        for frac in (0.005, 0.01, 0.05, 0.10, 0.25, 0.50):
                            k = max(1, int(frac * Nm))
                            v = float(cumsum[k - 1].item()) * 100.0
                            snap[f'top_{frac:.4f}'] = v
                            parts.append(f'top-{frac*100:.1f}%={v:.1f}%')
                        grad_concentration_log.append(snap)
                        if verbose:
                            print(f'  [grad-conc iter={it}] N={Nm}  '
                                  + '  '.join(parts))
                    # Reset for next interval.
                    pos_grad_accum.zero_()
            else:
                v_mat = optimizer.state.get(model.raw_materials, {}).get('exp_avg_sq')
                v_rot = optimizer.state.get(model.rotations, {}).get('exp_avg_sq')
                if v_mat is not None and v_rot is not None:
                    # Per-point combined Fisher (normalise each to max=1 before
                    # summing so neither modality dominates).
                    mat_pp = v_mat.detach().sum(dim=-1)
                    rot_pp = v_rot.detach().sum(dim=-1)
                    mat_n  = mat_pp / mat_pp.max().clamp(min=1e-30)
                    rot_n  = rot_pp / rot_pp.max().clamp(min=1e-30)
                    f_comb = mat_n + rot_n                              # (N,)
            if f_comb is not None:
                split_n = int(reg_densify_split_frac * model.N)
                prune_n = int(reg_densify_prune_frac * model.N)
                if split_n > 0 and prune_n > 0 and split_n == prune_n:
                    _densify_step(
                        model, optimizer, f_comb,
                        split_n=split_n, prune_n=prune_n,
                        pos_jitter_m=reg_densify_pos_jitter_m,
                        mat_jitter=reg_densify_mat_jitter,
                        rot_jitter=0.05,
                        init_raw_ref=init_raw,
                        init_normals_ref=init_normals,
                        child_source=reg_densify_child_source,
                        pool_xyz=pool_xyz,
                        pool_normals=pool_normals,
                        slot_to_pool_idx=slot_to_pool_idx,
                        pool_radius_m=reg_densify_pool_radius_m,
                        pool_radius_min_m=reg_densify_pool_radius_min_m,
                        pool_selection=reg_densify_pool_selection,
                        verbose=verbose,
                    )
                    n_densify_rounds += 1
                    # Positions moved — recompute FOV mask at the *first*
                    # train sample's pose (any train pose is fine for the
                    # "is this point in FOV anywhere" estimate). Apply it
                    # as the new conservative mask.
                    apply_pose(rast, train_samples[0]['pose'])
                    new_mask = cull_gaussians(model, rast)
                    with torch.no_grad():
                        active_mask = new_mask
                        vertex_areas = torch.zeros(model.N, device=DEVICE)
                        vertex_areas[active_mask] = 1.0
                    # Reset S2 Fisher snapshot + normal EMA so they
                    # re-establish under the new point set.
                    fisher_rot = None
                    normal_ema = None

        mean_tc = float(np.mean(per_sample_cc))
        history.append({
            'iter': it, 'loss': loss_sum,
            'mean_train_cc': mean_tc,
        })

        if mean_tc > best_mean_train_cc:
            best_mean_train_cc = mean_tc
            best_iter = it
            best_state = {k: v.data.clone() for k, v in model.state_dict().items()}

        if verbose and (it % 10 == 0 or it == num_iters - 1):
            elapsed = time.time() - t0
            print(f'  iter {it:4d}: loss={loss_sum:.5e}  '
                  f'mean_train_cc={mean_tc:.4f} '
                  f'(best={best_mean_train_cc:.4f}@{best_iter})  '
                  f'[{elapsed:.0f}s, n={len(train_samples)}]')

    train_elapsed = time.time() - t0

    # ── Restore best state + final test eval ──
    if best_state is not None:
        with torch.no_grad():
            for k, v in best_state.items():
                model.state_dict()[k].copy_(v)

    final_test_cc = _render_and_cart_corr(test_sample)
    final_train_ccs = [_render_and_cart_corr(s) for s in train_samples]
    final_train_mean = float(np.mean(final_train_ccs))

    if verbose:
        print(f'\n  [final] test  cc = {final_test_cc:.4f} '
              f'(init was {init_test_cc:.4f}, Δ {final_test_cc-init_test_cc:+.4f})')
        print(f'  [final] train mean cc = {final_train_mean:.4f}  '
              f'(best iter {best_iter})')
        print(f'  [final] elapsed {train_elapsed:.0f}s '
              f'({train_elapsed*1000/num_iters:.1f} ms/iter)')

    # ── Save ──
    if output_dir is None:
        tag = (f'train{len(train_frames)}frames_{len(train_loops)}loops'
               f'_test{test_frame}_loop{held_out_loop}')
        if test_in_train:
            tag = f'{tag}_ub'
        if use_pass2_alignment:
            tag = f'{tag}_pass2'
        if anchor_source == 'pass3_per_chirp':
            tag = f'{tag}_pass3anchor'
        if seed_frame != int(test_frame):
            tag = f'{tag}_seed{seed_frame}'
        if int(target_n) != 90000:
            tag = f'{tag}_N{int(target_n)}'
        if reg_l2_drift_lambda > 0.0:
            tag = f'{tag}_regL2{reg_l2_drift_lambda:g}'
        if reg_active_top_frac is not None:
            tag = f'{tag}_actTop{reg_active_top_frac:g}w{reg_warm_iters}'
        if reg_fisher_rot_lambda > 0.0:
            tag = f'{tag}_fshRot{reg_fisher_rot_lambda:g}w{reg_warm_iters}'
            if reg_fisher_rot_target != 'init':
                tag = f'{tag}_{reg_fisher_rot_target}'
            if reg_fisher_rot_threshold_deg > 0.0:
                tag = f'{tag}_thr{reg_fisher_rot_threshold_deg:g}'
        if reg_densify_interval > 0:
            src_tag = 'pk' if reg_densify_child_source == 'pool_knn' else 'jt'
            tag = (f'{tag}_dnsfy{src_tag}{reg_densify_split_frac:g}'
                   f'i{reg_densify_interval}u{reg_densify_until}')
            if reg_densify_child_source == 'pool_knn':
                tag = f'{tag}r{reg_densify_pool_radius_m:g}'
                if reg_densify_pool_selection == 'random_annulus':
                    tag = (f'{tag}ann{reg_densify_pool_radius_min_m:g}')
            else:
                tag = f'{tag}p{reg_densify_pos_jitter_m:g}'
        if init_variant != 'baseline':
            tag = f'{tag}_init{init_variant}'
        if densify_signal != 'fisher' and reg_densify_interval > 0:
            tag = f'{tag}_dsig{densify_signal}'
        if learn_positions_lr > 0.0:
            tag = f'{tag}_lpos{learn_positions_lr:g}L2{learn_positions_l2:g}'
        if phase4_d_lambda > 0.0:
            tag = f'{tag}_p4d{phase4_d_lambda:g}k{phase4_d_test_knn}'
        output_dir = os.path.join(
            PROJECT_ROOT, 'mm25DGS_v5_v4', 'output_frame_nvs', f'{scene}_{tag}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'scene': scene,
        'train_frames': list(map(int, train_frames)),
        'train_loops': list(map(int, train_loops)),
        'test_frame': int(test_frame),
        'held_out_loop': int(held_out_loop),
        'test_in_train': bool(test_in_train),
        'num_iters': int(num_iters),
        'best_iter': int(best_iter),
        'best_mean_train_cc': float(best_mean_train_cc),
        'init_test_cc': float(init_test_cc),
        'final_test_cc': float(final_test_cc),
        'init_train_mean_cc': float(np.mean(init_train_ccs)),
        'final_train_mean_cc': float(final_train_mean),
        'init_train_std_cc': float(np.std(init_train_ccs)),
        'final_train_std_cc': float(np.std(final_train_ccs)),
        'n_train_samples': int(len(train_samples)),
        'edge_frames': diag['edge_frames'],
        'test_pose_mode': diag['test_pose_mode'],
        'alignment_source': 'pass-2' if use_pass2_alignment else 'pass-1',
        'anchor_source':    anchor_source,
        'seed_frame':       int(seed_frame),
        'mat_lr': float(mat_lr), 'rot_lr': float(rot_lr),
        'loss_type': loss_type,
        'target_n': int(target_n),
        'elapsed_s': float(train_elapsed),
        'reg_l2_drift_lambda': float(reg_l2_drift_lambda),
        'reg_active_top_frac': (None if reg_active_top_frac is None
                                 else float(reg_active_top_frac)),
        'reg_warm_iters': int(reg_warm_iters),
        'reg_active_n_kept': (None if active_pts_mask is None
                               else int(active_pts_mask.sum().item())),
        'reg_fisher_rot_lambda': float(reg_fisher_rot_lambda),
        'reg_fisher_rot_snapshotted': bool(fisher_rot is not None),
        'reg_fisher_rot_target': reg_fisher_rot_target,
        'reg_fisher_rot_ema_alpha': float(reg_fisher_rot_ema_alpha),
        'reg_fisher_rot_threshold_deg': float(reg_fisher_rot_threshold_deg),
        'reg_densify_interval': int(reg_densify_interval),
        'reg_densify_split_frac': float(reg_densify_split_frac),
        'reg_densify_prune_frac': float(reg_densify_prune_frac),
        'reg_densify_until': int(reg_densify_until),
        'reg_densify_pos_jitter_m': float(reg_densify_pos_jitter_m),
        'reg_densify_mat_jitter': float(reg_densify_mat_jitter),
        'reg_densify_child_source': reg_densify_child_source,
        'reg_densify_pool_radius_m': float(reg_densify_pool_radius_m),
        'reg_densify_pool_radius_min_m': float(reg_densify_pool_radius_min_m),
        'reg_densify_pool_selection': reg_densify_pool_selection,
        'reg_densify_rounds_completed': int(n_densify_rounds),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(output_dir, 'history.npz'),
             iters=np.array([h['iter'] for h in history]),
             loss=np.array([h['loss'] for h in history]),
             mean_train_cc=np.array([h['mean_train_cc'] for h in history]))
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))

    # Fisher-imbalance diagnostic dump (Phase A of the v5_v4 follow-on
    # plan). Captures the never-reset career ‖∇p L‖ accumulator, the
    # per-densify concentration log, and final + init positions so we
    # can offline-analyse where the live points are in 3D space.
    if career_pos_grad is not None:
        diag_path = os.path.join(output_dir, 'fisher_diagnostic.pt')
        torch.save({
            'career_pos_grad': career_pos_grad.detach().cpu(),
            'last_window_pos_grad': (pos_grad_accum.detach().cpu()
                                      if pos_grad_accum is not None else None),
            'final_positions': model.positions.detach().cpu(),
            'init_positions': init_positions.detach().cpu(),
            'concentration_log': grad_concentration_log,
            'final_test_cc': float(final_test_cc),
            'final_train_mean_cc': float(final_train_mean),
            'N': int(model.N),
        }, diag_path)
        if verbose:
            print(f'  fisher diagnostic saved to: {diag_path}')

    if verbose:
        print(f'\n  results saved to: {output_dir}')
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Frame-level NVS trainer')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--test_frame', type=int, required=True)
    ap.add_argument('--train_frames', required=True,
                    help='comma-separated list, e.g. 434,435,436,437,439,440,441,442')
    ap.add_argument('--held_out_loop', type=int, default=0)
    ap.add_argument('--train_loops', default='0',
                    help='Comma-separated chirp-loop indices to train on; '
                         'default = "0" (first chirp loop only — the v5_v4 '
                         'combo_jitter recipe). Pass "" or omit to fall back '
                         'to None which expands to all 16 loops. NOT to be '
                         'confused with single-chip radar hardware — this is '
                         'still the cascaded radar, just restricted to its '
                         'first chirp loop per frame.')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--mat_lr', type=float, default=0.01)
    ap.add_argument('--rot_lr', type=float, default=5e-3)
    ap.add_argument('--loss_type', default='mse_raw',
                    choices=['mse', 'pearson', 'mse_raw'])
    ap.add_argument('--target_n', type=int, default=20000,
                    help='Target Gaussian count after init. Default 20000 '
                         '(v5_v4 combo_jitter recipe).')
    ap.add_argument('--frame_period_s', type=float, default=0.1)
    ap.add_argument('--loop_dt_s', type=float, default=7.87e-3 / 16.0)
    ap.add_argument('--anchor_source', default='pass2_lerp',
                    choices=['pass2_lerp', 'pass3_per_chirp'],
                    help='Per-chirp pose source. pass2_lerp (default): '
                         'LERP between pass-2 F-1 and F+1 (historical). '
                         'pass3_per_chirp: load each (frame, chirp)\'s Stage 3 '
                         'refined config directly; falls back to pass2_lerp '
                         'if any per-chirp file is missing.')
    ap.add_argument('--no_pass2', action='store_true',
                    help='Use pass-1 _aligned.json instead of pass-2')
    ap.add_argument('--reg_l2_drift_lambda', type=float, default=0.0,
                    help='H1: L2 penalty on raw_materials drift from init '
                         '(mean-over-N units). 0 = disabled.')
    ap.add_argument('--reg_active_top_frac', type=float, default=None,
                    help='H7: keep top-frac points by Adam exp_avg_sq '
                         '(training-set Fisher proxy) trainable after warm-up; '
                         'freeze the rest. None = disabled.')
    ap.add_argument('--reg_warm_iters', type=int, default=100,
                    help='H7 / S2: warmup iters before snapshotting '
                         'Fisher-weighted activity / rotation mask. '
                         'Default 100 (v5_v4 combo_jitter recipe).')
    ap.add_argument('--reg_fisher_rot_lambda', type=float, default=0.0,
                    help='S2: Fisher-weighted L2 on rotation drift from '
                         'init. Penalty: λ · Σ_i F_i · ||n(q_i) − n(target_i)||². '
                         'F_i is the Adam exp_avg_sq on rotations, '
                         'summed across 4 quat components and normalised to '
                         'max=1, snapshotted at iter=reg_warm_iters. '
                         '0 = disabled.')
    ap.add_argument('--reg_fisher_rot_target', default='init',
                    choices=['init', 'ema'],
                    help='S2 target for the rotation drift penalty. '
                         '"init" (default, Option 1): LiDAR normal from '
                         'pcl.npy. "ema" (Option 2): running exponential '
                         'mean of the trajectory (see --reg_fisher_rot_ema_alpha).')
    ap.add_argument('--reg_fisher_rot_ema_alpha', type=float, default=0.95,
                    help='S2 Option 2: EMA decay for the rotation target '
                         '(only used when --reg_fisher_rot_target=ema).')
    ap.add_argument('--reg_fisher_rot_threshold_deg', type=float, default=0.0,
                    help='S2 Option 3: if > 0, penalise only rotation drift '
                         '*beyond* this threshold (in degrees). Soft Huber-'
                         'style: loss ∝ max(0, drift° − θ)². 0 = disabled '
                         '(full L2 penalty). Typical: 30–60°.')
    ap.add_argument('--reg_densify_interval', type=int, default=100,
                    help='S4: densify/prune every N iters after --reg_warm_iters. '
                         'Default 100 (v5_v4 combo_jitter); 0 = disabled.')
    ap.add_argument('--reg_densify_split_frac', type=float, default=0.05,
                    help='S4: fraction of top-signal points to split per round. '
                         'Default 0.05 (v5_v4 combo_jitter).')
    ap.add_argument('--reg_densify_prune_frac', type=float, default=0.05,
                    help='S4: fraction of bottom-signal points to prune per round '
                         '(must equal --reg_densify_split_frac to keep N fixed). '
                         'Default 0.05 (v5_v4 combo_jitter).')
    ap.add_argument('--reg_densify_until', type=int, default=400,
                    help='S4: stop densifying past this iter (lets the final '
                         'iters converge on a stable point set). Default 400 '
                         '(v5_v4 combo_jitter).')
    ap.add_argument('--reg_densify_pos_jitter_m', type=float, default=0.02,
                    help='S4: std (m) of isotropic Gaussian position noise for '
                         'new child points. 0.02 = 2 cm ≈ 5λ at 77 GHz.')
    ap.add_argument('--reg_densify_mat_jitter', type=float, default=0.1,
                    help='S4: std of Gaussian noise added to raw_materials for '
                         'child points (raw units).')
    ap.add_argument('--reg_densify_child_source', default='jitter',
                    choices=['pool_knn', 'jitter'],
                    help='S4 child source. "jitter" (default, v5_v4 combo_'
                         'jitter recipe): child = parent + Gaussian '
                         'position/quaternion noise. Empirically beats '
                         'pool_knn by +0.020 mean test cc. "pool_knn": '
                         'draw child from the nearest-unused point in the '
                         'post-resample LiDAR pool within '
                         '--reg_densify_pool_radius_m of each parent.')
    ap.add_argument('--reg_densify_pool_radius_m', type=float, default=0.15,
                    help='S4 pool_knn mode: max distance (m) from parent to '
                         'a candidate pool point. Parents with no pool point '
                         'inside this radius fall back to jitter.')
    ap.add_argument('--reg_densify_pool_radius_min_m', type=float, default=0.0,
                    help='S4 pool_knn mode (random_annulus only): min distance '
                         '(m) from parent. Ignored under selection=nearest.')
    ap.add_argument('--reg_densify_pool_selection', default='nearest',
                    choices=['nearest', 'random_annulus'],
                    help='S4 pool_knn picking strategy. "nearest": pick '
                         'nearest unused pool point in [0, radius_m) — near-'
                         'clone behaviour when pool is dense. "random_annulus": '
                         'uniform-random pick within [radius_min_m, radius_m) — '
                         'matches jitter-scale diversity while keeping children '
                         'on real LiDAR surfaces.')
    ap.add_argument('--seed_frame', type=int, default=None,
                    help='Frame whose pose seeds the rasterizer (drives FOV + '
                         'RX visibility before FPS, so it determines the 90k-'
                         'point subset of pcl.npy used for training). Default = '
                         '`test_frame` so HO and UB variants share an identical '
                         'position grid (see md/frame_nvs_analysis/findings.md '
                         '§0). Pass an explicit value to reproduce the legacy '
                         'HO grid (`--seed_frame {test_frame+1}`) or to probe '
                         'sensitivity to the seed.')
    ap.add_argument('--init_variant', default='baseline',
                    choices=['baseline', 'A1_no_fps', 'A2_union_cos',
                             'A3_union_amplitude', 'A4_union_amp_lidar',
                             'A5_amp_lidar_fps',
                             'B1_strict_and_train',
                             'B2_strict_and_with_test',
                             'C1_voxel_v1',
                             'C1b_voxel_capped',
                             'C2_voxel_v2'],
                    help='v5_v4 Phase 2 init variant. "baseline" = v5 init '
                         '(seed-pose FOV+RX+cosine resample+FPS). A2-A5 do '
                         'union-amplitude importance sampling across train '
                         'poses (see md/mm25dgs_v5_v4_smart_sampling_'
                         'and_densification.md §2.5). B1/B2 do strict-AND '
                         'visibility (B1: 8 train poses; B2: +test pose) '
                         'then FPS within the surviving set — addresses '
                         'the Fisher-imbalance finding that flatter init '
                         'concentration correlates with better test cc.')
    ap.add_argument('--densify_signal', default='pos_grad_amp',
                    choices=['fisher', 'pos_grad_amp'],
                    help='v5_v4 Phase 3 densify selection signal. '
                         '"pos_grad_amp" (default, v5_v4 combo_jitter '
                         'recipe): accumulated amplitude-path position '
                         'gradient magnitude — positions remain frozen but '
                         'their .grad is captured (NOT in optimizer) and '
                         'used as the densify selection signal. "fisher" '
                         '(legacy v5) uses Adam exp_avg_sq summed across '
                         'material+rotation params (see md/mm25dgs_v5_v4_'
                         'smart_sampling_and_densification.md §4).')
    ap.add_argument('--learn_positions_lr', type=float, default=1e-5,
                    help='v5_v2 Phase 1 carryover. > 0 enables learnable '
                         'positions (added to Adam at this LR). Default '
                         '1e-5 (v5_v4 combo_jitter recipe). Gradient flows '
                         'AMPLITUDE-PATH ONLY (detach_phase=True is enforced '
                         'by the renderer; phi_carrier and n_peak are '
                         'detached). NO PHASE GRADIENTS by construction. '
                         'Set 0.0 to freeze positions.')
    ap.add_argument('--learn_positions_l2', type=float, default=100.0,
                    help='L2 anchor coefficient on (positions − init). '
                         'Default 100 (v5_v4 combo_jitter). Keeps positions '
                         'sub-mm from LiDAR seed.')
    ap.add_argument('--phase4_d_lambda', type=float, default=0.0,
                    help='v5_v4 Phase 4 (D) — per-train-view membership. '
                         '> 0 enables a learnable (N × V) matrix m where '
                         'sigmoid(m[:, v]) is the per-view opacity '
                         'multiplier on each point. λ is the L1 sparsity '
                         'weight on sigmoid(m). At test, KNN-average over '
                         'the nearest train views (--phase4_d_test_knn).')
    ap.add_argument('--phase4_d_test_knn', type=int, default=2,
                    help='K for the KNN-over-train-views test deployment.')
    # MLP-A — pose-conditioned per-Gaussian deformation MLP
    ap.add_argument('--mlp_a_lr', type=float, default=0.0,
                    help='MLP-A learning rate. > 0 enables MLP-A, a '
                         'pose-conditioned per-Gaussian deformation field. '
                         'See md/mm25dgs_v5_v4_mlp_options.md.')
    ap.add_argument('--mlp_a_hidden_dim', type=int, default=64)
    ap.add_argument('--mlp_a_n_layers', type=int, default=3)
    ap.add_argument('--mlp_a_max_dpos_m', type=float, default=0.05,
                    help='Bound on per-Gaussian Δposition magnitude (m).')
    ap.add_argument('--mlp_a_max_dalpha', type=float, default=1.0,
                    help='Bound on per-Gaussian Δopacity magnitude.')
    ap.add_argument('--mlp_a_l2_dpos', type=float, default=100.0,
                    help='L2 anchor weight on Δposition outputs (per-Gaussian).')
    ap.add_argument('--mlp_a_l1_dalpha', type=float, default=0.01,
                    help='L1 sparsity weight on Δopacity outputs.')
    ap.add_argument('--mlp_a_warmup_iters', type=int, default=50,
                    help='Iters to keep MLP-A frozen at zero before training '
                         '(let materials/rotations settle first).')
    ap.add_argument('--mlp_a_pose_pe_freqs', type=int, default=0,
                    help='NeRF-style positional encoding frequencies on '
                         'pose_6d input (0 = no encoding). Helps MLP '
                         'interpolate smoothly across train poses.')
    args = ap.parse_args()

    train_frames = [int(x) for x in args.train_frames.split(',') if x.strip()]
    train_loops = (None if args.train_loops is None else
                   [int(x) for x in args.train_loops.split(',') if x.strip()])
    train_frame_nvs(
        scene=args.scene,
        train_frames=train_frames,
        test_frame=args.test_frame,
        held_out_loop=args.held_out_loop,
        train_loops=train_loops,
        num_iters=args.iters,
        mat_lr=args.mat_lr, rot_lr=args.rot_lr,
        loss_type=args.loss_type, target_n=args.target_n,
        frame_period_s=args.frame_period_s, loop_dt_s=args.loop_dt_s,
        use_pass2_alignment=not args.no_pass2,
        anchor_source=args.anchor_source,
        reg_l2_drift_lambda=args.reg_l2_drift_lambda,
        reg_active_top_frac=args.reg_active_top_frac,
        reg_warm_iters=args.reg_warm_iters,
        reg_fisher_rot_lambda=args.reg_fisher_rot_lambda,
        reg_fisher_rot_target=args.reg_fisher_rot_target,
        reg_fisher_rot_ema_alpha=args.reg_fisher_rot_ema_alpha,
        reg_fisher_rot_threshold_deg=args.reg_fisher_rot_threshold_deg,
        reg_densify_interval=args.reg_densify_interval,
        reg_densify_split_frac=args.reg_densify_split_frac,
        reg_densify_prune_frac=args.reg_densify_prune_frac,
        reg_densify_until=args.reg_densify_until,
        reg_densify_pos_jitter_m=args.reg_densify_pos_jitter_m,
        reg_densify_mat_jitter=args.reg_densify_mat_jitter,
        reg_densify_child_source=args.reg_densify_child_source,
        reg_densify_pool_radius_m=args.reg_densify_pool_radius_m,
        reg_densify_pool_radius_min_m=args.reg_densify_pool_radius_min_m,
        reg_densify_pool_selection=args.reg_densify_pool_selection,
        seed_frame=args.seed_frame,
        init_variant=args.init_variant,
        densify_signal=args.densify_signal,
        learn_positions_lr=args.learn_positions_lr,
        learn_positions_l2=args.learn_positions_l2,
        phase4_d_lambda=args.phase4_d_lambda,
        phase4_d_test_knn=args.phase4_d_test_knn,
        mlp_a_lr=args.mlp_a_lr,
        mlp_a_hidden_dim=args.mlp_a_hidden_dim,
        mlp_a_n_layers=args.mlp_a_n_layers,
        mlp_a_max_dpos_m=args.mlp_a_max_dpos_m,
        mlp_a_max_dalpha=args.mlp_a_max_dalpha,
        mlp_a_l2_dpos=args.mlp_a_l2_dpos,
        mlp_a_l1_dalpha=args.mlp_a_l1_dalpha,
        mlp_a_warmup_iters=args.mlp_a_warmup_iters,
        mlp_a_pose_pe_freqs=args.mlp_a_pose_pe_freqs)
