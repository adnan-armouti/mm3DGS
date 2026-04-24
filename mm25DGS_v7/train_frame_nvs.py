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
    python -m mm25DGS_v5.train_frame_nvs \
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

from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch  # noqa: F401
from mm25DGS_v5 import cuda as v5cuda
from mm25DGS_v5.train_gaussian import (
    DEVICE,
    init_visible_weighted,
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
from mm25DGS_v5.load_pretrained import load_trained_config, load_pattern_data

# Reuse chirp-loop helpers (per-loop pose interpolation + pose applier + GT loader)
from mm25DGS_v5.train_chirp_loop_nvs import (
    build_per_loop_poses, apply_pose, _load_pose_from_config,
)


# ---------------------------------------------------------------------------
# Per-scene data loading (train and test frames)
# ---------------------------------------------------------------------------

def _aligned_config_path(scene, frame, use_pass2, data_root,
                          pass_name: str = None):
    """Resolve the aligned-config path.

    ``pass_name`` (optional) overrides the pass-2 fallback:
      'pass3' → try `_aligned_pass3` first, fall back to pass-2/pass-1.
      'pass2' → try `_aligned_pass2`, fall back to pass-1.
      'pass1' → only `_aligned`.
      None    → use ``use_pass2`` (legacy behaviour).
    """
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    # Build fallback chain (most-refined first)
    if pass_name == 'pass3':
        chain = ['_aligned_pass3', '_aligned_pass2', '_aligned']
    elif pass_name == 'pass2':
        chain = ['_aligned_pass2', '_aligned']
    elif pass_name == 'pass1':
        chain = ['_aligned']
    else:
        chain = ['_aligned_pass2', '_aligned'] if use_pass2 else ['_aligned']
    for suffix in chain:
        p = os.path.join(align_dir, f'cascaded_frame_{frame}{suffix}.json')
        if os.path.isfile(p):
            return p
    return None


def _pass3_config_path(scene, frame, chirp, data_root):
    """Per-chirp Stage 3 config path (produced by
    ``per_chirp_alignment.run_stage3_for_scene``)."""
    return os.path.join(
        data_root, 'alignment_data', scene, 'cascade', 'per_chirp',
        f'cascaded_frame_{frame}_chirp{chirp:02d}_aligned_pass3.json')


def _build_frame_poses(scene, frame, use_pass2, data_root,
                       loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.1,
                       anchor_source='pass2_lerp',
                       device=DEVICE,
                       pass_name: str = None):
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
    cfg_A = _aligned_config_path(scene, frame - 1, use_pass2, data_root,
                                   pass_name=pass_name)
    cfg_B = _aligned_config_path(scene, frame + 1, use_pass2, data_root,
                                   pass_name=pass_name)
    if cfg_A is not None and cfg_B is not None:
        poses, _ = build_per_loop_poses(
            cfg_A, cfg_B, n_loops=16,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            device=device)
        return poses, 'interp_neighbours'

    cfg_F = _aligned_config_path(scene, frame, use_pass2, data_root,
                                   pass_name=pass_name)
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


def _flat_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flattened Pearson correlation between two real tensors.

    Used as a "|RAD| cc" diagnostic next to the chirp-0 |RA| cart_corr
    (plan md/mm25dgs_v7_training_ceiling.md §C2a). Scale/shift-invariant.
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    am = a - a.mean()
    bm = b - b.mean()
    num = (am * bm).sum()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum()).clamp_min(1e-30)
    return float((num / den).item())


def _build_rad_bundle_for_frame(adc_npy_path, poses_16, frame_idx,
                                  v_ego, device, n_dop=None):
    """v7 Doppler per-frame supervision bundle.

    Loads all 16 chirps of ``adc_npy_path``, runs the GT RAD pipeline
    (range FFT + azimuth FFT + Doppler FFT with v5-identical Hann +
    ifftshift + drop-bin-0 + fftshift), and bundles what the training
    loop needs for one frame:

      'frame_idx' : int
      'pose_F'    : chirp-0 pose dict (for apply_pose + render call)
      'v_ego'     : (3,) float tensor — frozen, NOT learnable
      'gt_rad_mag': (D, 127, R) float32 |RAD|
      'gt_rad_max': float, for mse_raw normalisation
    """
    from mm25DGS_v7.data.ra_utils import adc_to_rad_complex, N_DOP_DEFAULT
    if n_dop is None:
        n_dop = N_DOP_DEFAULT

    arr = np.load(adc_npy_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'expected (16, RX, TX, K); got {arr.shape}')
    adc = arr.transpose(0, 2, 1, 3)                                # (CH, TX, RX, ADC)
    ri = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    ri_t = torch.from_numpy(ri).to(device)
    with torch.no_grad():
        rad = adc_to_rad_complex(ri_t, n_dop=n_dop)                # (D, 127, R) cpx
        mag = rad.abs().float()
        mag_max = mag.amax().clamp_min(1e-30).item()
        mag_mean = mag.mean().clamp_min(1e-30).item()
    return {
        'frame_idx': int(frame_idx),
        'pose_F':    poses_16[0],            # chirp-0 anchor pose
        'v_ego':     torch.as_tensor(v_ego, dtype=torch.float32, device=device),
        'gt_rad_mag': mag,
        'gt_rad_max': float(mag_max),
        'gt_rad_mean': float(mag_mean),
    }


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
                               pass_name=None,
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
            anchor_source=anchor_source, device=device,
            pass_name=pass_name)
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
        anchor_source=anchor_source, device=device,
        pass_name=pass_name)
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
                    pass_name=None,        # 'pass3' | 'pass2' | 'pass1' | None
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
                    doppler=False,
                    loss_norm='mean',      # C1 fix (was 'max' pre-2026-04)
                    loss_multitask_lambda=0.0,  # C2b: λ·mse(|RA|_chirp0)
                    use_refined_v_ego=True):    # §4.0: +0.026 RA te / +0.043 RAD te
    assert v5cuda.is_available(), (
        'v5 CUDA extension not built. '
        'cd mm25DGS_v5/cuda && python setup.py build_ext --inplace')

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
    if doppler:
        # v7 Doppler requires all 16 chirps per train frame (GT RAD uses all 16)
        if train_loops != list(range(16)):
            if verbose:
                print('  [v7] forcing --train_loops 0..15 for Doppler mode')
            train_loops = list(range(16))
    train_samples, test_sample, diag = build_frame_level_dataset(
        scene=scene, train_frames=train_frames, test_frame=test_frame,
        held_out_loop=held_out_loop, use_pass2=use_pass2_alignment,
        train_loops=train_loops,
        loss_type=loss_type, loop_dt_s=loop_dt_s,
        frame_period_s=frame_period_s, data_root=data_root,
        anchor_source=anchor_source, pass_name=pass_name,
        device=DEVICE, verbose=verbose,
    )

    # v7 Doppler: build per-frame RAD bundles in addition to per-(frame,
    # loop) train_samples. Each bundle carries GT |RAD| + v_ego.
    train_rad_bundles = []
    test_rad_bundle = None
    if doppler:
        from mm25DGS_v7.preprocessing.v_ego import get_or_compute_v_ego
        radar_dir = os.path.join(data_root, scene, 'radar')
        for f in train_frames:
            poses, _ = _build_frame_poses(
                scene, f, use_pass2=use_pass2_alignment, data_root=data_root,
                loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
                anchor_source=anchor_source, device=DEVICE,
                pass_name=pass_name)
            v = get_or_compute_v_ego(scene, int(f), data_root=data_root,
                                        use_refined=use_refined_v_ego)
            bundle = _build_rad_bundle_for_frame(
                os.path.join(radar_dir, f'cascaded_frame_{f}.npy'),
                poses, frame_idx=f, v_ego=v, device=DEVICE)
            train_rad_bundles.append(bundle)
        # Also build a test RAD bundle so we can log |RAD| cc at the
        # held-out frame each iter (C2a diagnostic).
        test_poses_16, _ = _build_frame_poses(
            scene, test_frame, use_pass2=use_pass2_alignment,
            data_root=data_root,
            loop_dt_s=loop_dt_s, frame_period_s=frame_period_s,
            anchor_source=anchor_source, device=DEVICE,
            pass_name=pass_name)
        test_v_ego = get_or_compute_v_ego(
            scene, int(test_frame), data_root=data_root,
            use_refined=use_refined_v_ego)
        test_rad_bundle = _build_rad_bundle_for_frame(
            os.path.join(radar_dir, f'cascaded_frame_{test_frame}.npy'),
            test_poses_16, frame_idx=test_frame, v_ego=test_v_ego,
            device=DEVICE)
        if verbose:
            d0 = train_rad_bundles[0]
            print(f'  [v7 doppler] {len(train_rad_bundles)} RAD bundles  '
                  f"(|RAD| cube {tuple(d0['gt_rad_mag'].shape)})")
            for b in train_rad_bundles[:3]:
                v = b['v_ego'].cpu().numpy()
                print(f'    F={b["frame_idx"]:<4d}  v_ego=[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}]  '
                      f'|v|={float(np.linalg.norm(v)):.2f} m/s')

    # Finalize GT cart norms for cart_corr evaluation
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
    def _render_and_cart_corr(sample):
        apply_pose(rast, sample['pose'])
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
            cc = cart_corr_torch(ra_cart, sample['gt_cart_norm']).item()
        return float(cc)

    init_test_cc = _render_and_cart_corr(test_sample)
    init_train_ccs = [_render_and_cart_corr(s) for s in train_samples]

    # C2a diagnostic — |RAD| cc at init. Only evaluated when doppler mode
    # is on (we have the bundles). Does a full 16-chirp doppler render
    # per bundle (fused on supported GPUs — O(BSDF)).
    def _render_and_rad_corr(bundle):
        apply_pose(rast, bundle['pose_F'])
        with torch.no_grad():
            from mm25DGS_v7.train_gaussian import render_gaussians_doppler
            from mm25DGS_v7.data.ra_utils import rp_stack_to_rad_complex
            rp_r, rp_i = render_gaussians_doppler(
                model, rast, bundle['v_ego'],
                vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None, n_chirps=16)
            rp_c = torch.complex(rp_r, rp_i)
            rad_pred = rp_stack_to_rad_complex(rp_c).abs()
        return _flat_pearson(rad_pred, bundle['gt_rad_mag'])

    init_test_rad_cc = None
    init_train_rad_cc_mean = None
    if test_rad_bundle is not None:
        init_test_rad_cc = _render_and_rad_corr(test_rad_bundle)
        init_train_rad_ccs = [_render_and_rad_corr(b)
                               for b in train_rad_bundles]
        init_train_rad_cc_mean = float(np.mean(init_train_rad_ccs))

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
    base_lrs = {g['name']: g['lr'] for g in param_groups}
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)

    loss_scale = 1.0 / len(train_samples)

    # ── Regularisation setup ──
    # Snapshot init raw_materials (uniform ITU_CONCRETE across N at init time).
    init_raw = model.raw_materials.detach().clone()
    # Snapshot init surface normals (from the quaternion init computed from
    # pcl.npy LiDAR normals). Needed for S2's rotation-drift penalty.
    init_normals = model.get_normals().detach().clone()  # (N, 3)
    active_pts_mask = None
    # S2: per-point Fisher weights on rotations, snapshot at warmup using
    # Adam's exp_avg_sq. Stays None until reg_warm_iters.
    fisher_rot = None
    # S2 Option 2: EMA of normals during training (rolling target).
    normal_ema = None
    # S4: adaptive density (split/prune) bookkeeping
    n_densify_rounds = 0
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

    # Loss scale for Doppler mode: one gradient step per frame-bundle
    if doppler:
        loss_scale_doppler = 1.0 / max(len(train_rad_bundles), 1)
        from mm25DGS_v7.train_gaussian import render_gaussians_doppler as _rg_doppler
        from mm25DGS_v7.data.ra_utils import rp_stack_to_rad_complex as _rp_to_rad

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        per_sample_cc = []
        per_sample_rad_cc = []   # C2a diagnostic — |RAD| cc per bundle.
        loss_sum = 0.0

        if doppler:
            # --------------------------------------------------------------
            # v7 Doppler training body — 1 render per train frame (all 16
            # chirps in one call), mse_raw on |RAD|. Per-frame cart_corr
            # is still tracked on chirp-0 |RA| for comparability with v5.
            # --------------------------------------------------------------
            for bundle in train_rad_bundles:
                apply_pose(rast, bundle['pose_F'])
                rp_r, rp_i = _rg_doppler(
                    model, rast, bundle['v_ego'],
                    vertex_areas=vertex_areas,
                    active_mask=active_mask, shadow_mask=None,
                    bsdf_mode='full', disabled_components=None,
                    n_chirps=16,
                )                                                  # (16, 12, 16, 256)
                rp_c = torch.complex(rp_r, rp_i)
                rad_pred = _rp_to_rad(rp_c)                        # (D, 127, 256) cpx
                mag_pred = rad_pred.abs()
                gt_mag   = bundle['gt_rad_mag']
                gt_max   = bundle['gt_rad_max']
                gt_scale = (gt_max if loss_norm == 'max'
                             else bundle['gt_rad_mean'])
                loss_k = (mag_pred - gt_mag).pow(2).mean() / (gt_scale ** 2)

                # C2b — multi-task term on chirp-0 |RA| vs the matching
                # train_sample's cached GT |RA| (min-max normalized). Free
                # — re-uses rp_r[0], rp_i[0] from the doppler render stack.
                loss_total = loss_k
                if loss_multitask_lambda > 0.0:
                    # Find the matching (frame, loop=0) train_sample for GT
                    match_s = None
                    for s in train_samples:
                        if (int(s['frame_idx']) == bundle['frame_idx']
                                and int(s['loop_idx']) == 0):
                            match_s = s; break
                    if match_s is not None and 'gt_loss' in match_s:
                        from mm25DGS_v7.train_gaussian import compute_ra_loss_rp
                        loss_chirp0, _ = compute_ra_loss_rp(
                            rp_r[0], rp_i[0], match_s['gt_loss'],
                            loss_type=loss_type)
                        loss_total = loss_k + loss_multitask_lambda * loss_chirp0

                (loss_total * loss_scale_doppler).backward()
                loss_sum += float(loss_total.item()) * loss_scale_doppler
                # Per-frame chirp-0 cc on v5 |RA| — for mean_train_cc history.
                with torch.no_grad():
                    rp0_r = rp_r[0].detach()
                    rp0_i = rp_i[0].detach()
                    ra_polar = range_profile_to_ra_mag(rp0_r, rp0_i)
                    ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                    # Find matching train_sample (frame + loop=0) for gt_cart_norm
                    for s in train_samples:
                        if (int(s['frame_idx']) == bundle['frame_idx']
                                and int(s['loop_idx']) == 0):
                            cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                            per_sample_cc.append(cc)
                            break
                    # C2a — |RAD| cc is free: mag_pred already computed.
                    per_sample_rad_cc.append(
                        _flat_pearson(mag_pred.detach(), gt_mag))
                del rp_r, rp_i, rp_c, rad_pred, mag_pred, loss_k
            # Skip the v5-style per-sample loop below
            _skip_v5_body = True
        else:
            _skip_v5_body = False

        for s in (train_samples if not _skip_v5_body else []):
            apply_pose(rast, s['pose'])
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            loss_k, _ = compute_ra_loss_rp(
                rp_real, rp_imag, s['gt_loss'], loss_type=loss_type)
            (loss_k * loss_scale).backward()
            with torch.no_grad():
                ra_polar = range_profile_to_ra_mag(rp_real.detach(), rp_imag.detach())
                ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                per_sample_cc.append(cc)
            loss_sum += loss_k.item() * loss_scale
            del rp_real, rp_imag, ra_polar, ra_cart, loss_k

        # H1 reg: L2 drift from init, mean over all elements (N·6).
        if reg_l2_drift_lambda > 0.0:
            drift = model.raw_materials - init_raw
            reg_loss = reg_l2_drift_lambda * drift.pow(2).mean()
            reg_loss.backward()
            loss_sum += float(reg_loss.item())

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
        # Per-iter test cc — v5-compatible metric (chirp-0 |RA| cart_corr
        # at the held-out test pose). Matches v5/M1 history format so
        # v5 and v7 runs can be plotted side-by-side.
        test_cc_iter = _render_and_cart_corr(test_sample)
        # C2a — |RAD| metrics alongside |RA|. Test RAD cc is one extra
        # fused doppler render per iter (~50–80 ms); train RAD cc is
        # free (reuses per-bundle mag_pred).
        mean_rad_tc = (float(np.mean(per_sample_rad_cc))
                        if per_sample_rad_cc else float('nan'))
        test_rad_cc_iter = (_render_and_rad_corr(test_rad_bundle)
                             if test_rad_bundle is not None else float('nan'))
        history.append({
            'iter': it, 'loss': loss_sum,
            'mean_train_cc': mean_tc,
            'test_cc': test_cc_iter,
            'mean_train_rad_cc': mean_rad_tc,
            'test_rad_cc': test_rad_cc_iter,
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

    final_test_rad_cc = None
    final_train_rad_cc_mean = None
    if test_rad_bundle is not None:
        final_test_rad_cc = _render_and_rad_corr(test_rad_bundle)
        final_train_rad_ccs = [_render_and_rad_corr(b)
                                for b in train_rad_bundles]
        final_train_rad_cc_mean = float(np.mean(final_train_rad_ccs))

    if verbose:
        print(f'\n  [final] test  cc = {final_test_cc:.4f} '
              f'(init was {init_test_cc:.4f}, Δ {final_test_cc-init_test_cc:+.4f})')
        print(f'  [final] train mean cc = {final_train_mean:.4f}  '
              f'(best iter {best_iter})')
        if final_test_rad_cc is not None:
            print(f'  [final] |RAD| test cc = {final_test_rad_cc:.4f}  '
                  f'train mean = {final_train_rad_cc_mean:.4f}  '
                  f'(init test={init_test_rad_cc:.4f})')
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
        if doppler:
            tag = f'{tag}_v7doppler_norm{loss_norm}'
        if pass_name is not None:
            tag = f'{tag}_{pass_name}'
        if loss_multitask_lambda > 0.0:
            tag = f'{tag}_mt{loss_multitask_lambda:g}'
        if use_refined_v_ego:
            tag = f'{tag}_vegorf'
        output_dir = os.path.join(
            PROJECT_ROOT, 'mm25DGS_v7', 'output_frame_nvs', f'{scene}_{tag}')
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
        # C2a — |RAD| cc (None when --doppler is off).
        'init_test_rad_cc': (None if init_test_rad_cc is None
                              else float(init_test_rad_cc)),
        'final_test_rad_cc': (None if final_test_rad_cc is None
                               else float(final_test_rad_cc)),
        'init_train_rad_cc_mean': (None if init_train_rad_cc_mean is None
                                    else float(init_train_rad_cc_mean)),
        'final_train_rad_cc_mean': (None if final_train_rad_cc_mean is None
                                     else float(final_train_rad_cc_mean)),
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
             mean_train_cc=np.array([h['mean_train_cc'] for h in history]),
             test_cc=np.array([h.get('test_cc', float('nan'))
                                for h in history]),
             mean_train_rad_cc=np.array(
                 [h.get('mean_train_rad_cc', float('nan'))
                  for h in history]),
             test_rad_cc=np.array(
                 [h.get('test_rad_cc', float('nan'))
                  for h in history]))
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))

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
    ap.add_argument('--train_loops', default=None,
                    help='Comma-separated chirp-loop indices to train on; '
                         'default = all 16 loops. Pass "0" for first-chirp-'
                         'only training (NOT to be confused with single-chip '
                         'radar hardware — this is still the cascaded radar, '
                         'just restricted to its first chirp loop per frame).')
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--mat_lr', type=float, default=0.01)
    ap.add_argument('--rot_lr', type=float, default=5e-3)
    ap.add_argument('--loss_type', default='mse_raw',
                    choices=['mse', 'pearson', 'mse_raw'])
    ap.add_argument('--target_n', type=int, default=90000)
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
    ap.add_argument('--reg_warm_iters', type=int, default=50,
                    help='H7 / S2: warmup iters before snapshotting '
                         'Fisher-weighted activity / rotation mask.')
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
    ap.add_argument('--reg_densify_interval', type=int, default=0,
                    help='S4: densify/prune every N iters after --reg_warm_iters. '
                         '0 = disabled. Typical: 50.')
    ap.add_argument('--reg_densify_split_frac', type=float, default=0.02,
                    help='S4: fraction of top-Fisher points to split per round.')
    ap.add_argument('--reg_densify_prune_frac', type=float, default=0.02,
                    help='S4: fraction of bottom-Fisher points to prune per round '
                         '(must equal --reg_densify_split_frac to keep N fixed).')
    ap.add_argument('--reg_densify_until', type=int, default=300,
                    help='S4: stop densifying past this iter (lets the final '
                         'iters converge on a stable point set).')
    ap.add_argument('--reg_densify_pos_jitter_m', type=float, default=0.02,
                    help='S4: std (m) of isotropic Gaussian position noise for '
                         'new child points. 0.02 = 2 cm ≈ 5λ at 77 GHz.')
    ap.add_argument('--reg_densify_mat_jitter', type=float, default=0.1,
                    help='S4: std of Gaussian noise added to raw_materials for '
                         'child points (raw units).')
    ap.add_argument('--reg_densify_child_source', default='pool_knn',
                    choices=['pool_knn', 'jitter'],
                    help='S4 child source. "pool_knn" (default): draw '
                         'child position + normal from the nearest-unused '
                         'point in the post-resample LiDAR pool within '
                         '--reg_densify_pool_radius_m of each parent. '
                         '"jitter" (legacy): child = parent + Gaussian '
                         'position/quaternion noise (may drift off-surface).')
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
    ap.add_argument('--doppler', action='store_true',
                    help='v7 Doppler mode — render all 16 chirps per '
                         'train frame via analytic Doppler phase in the '
                         'scatter kernel, train on v5 mse_raw applied to '
                         'the 3-D |RAD| cube. See '
                         'md/mm25dgs_v7_doppler_plan.md.')
    ap.add_argument('--loss_norm', default='mean', choices=['max', 'mean'],
                    help='Doppler |RAD| loss normalisation (plan '
                         'md/mm25dgs_v7_training_ceiling.md §C1). '
                         'Default "mean" matches v5 mse_raw '
                         '(gt_mag.mean()^2). "max" is the legacy path '
                         '(gt_mag.amax()^2, effectively shrinks the loss '
                         '~2500×). Output dir always tagged with _norm<mode>.')
    ap.add_argument('--loss_multitask_lambda', type=float, default=0.0,
                    help='C2b multi-task: add λ·mse(|RA|_chirp0) to the '
                         '|RAD| training loss. 0 = |RAD| only (default). '
                         '>0 biases optimizer back toward the chirp-0 |RA| '
                         'metric that v5 was specialised for.')
    ap.add_argument('--pass_name', default=None,
                    choices=[None, 'pass3', 'pass2', 'pass1'],
                    help='Which alignment pass to use. None = legacy '
                         '(pass-2 if use_pass2 is on, else pass-1). '
                         '"pass3" prefers `_aligned_pass3.json`, falls '
                         'back to pass-2 then pass-1 if missing.')
    ap.add_argument('--use_refined_v_ego', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='§4.0 deployment: prefer `<scene>/frame_<F>_'
                         'v_ego_refined.npy` from v_ego_refine over the '
                         'seed cache. ON by default (6-scene bench: +0.026 '
                         'on |RA| test, +0.043 on |RAD| test). Silent '
                         'fallback to seed when the refined file is absent. '
                         'Disable with --no-use_refined_v_ego.')
    ap.add_argument('--seed_frame', type=int, default=None,
                    help='Frame whose pose seeds the rasterizer (drives FOV + '
                         'RX visibility before FPS, so it determines the 90k-'
                         'point subset of pcl.npy used for training). Default = '
                         '`test_frame` so HO and UB variants share an identical '
                         'position grid (see md/frame_nvs_analysis/findings.md '
                         '§0). Pass an explicit value to reproduce the legacy '
                         'HO grid (`--seed_frame {test_frame+1}`) or to probe '
                         'sensitivity to the seed.')
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
        doppler=args.doppler,
        loss_norm=args.loss_norm,
        loss_multitask_lambda=args.loss_multitask_lambda,
        pass_name=args.pass_name,
        use_refined_v_ego=args.use_refined_v_ego)
