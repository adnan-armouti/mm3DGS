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


def _build_per_loop_gt(adc_npy_path, loop_idx, loss_type, device,
                       compute_per_virt=False):
    """Load a single (frame, loop) GT as a dict with 'gt_loss' +
    'gt_ra_polar' (magnitude). Caller converts polar→cart later once the
    sample_grid is built.

    ``compute_per_virt=True`` additionally returns ``gt_per_virt``, the
    ``(N_virt=192, N_range=256)`` complex per-virtual range-profile
    tensor in ADC channel order. Required by v6 M1.5+ (Gram loss).
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
    out = {'gt_loss': gt_loss, 'gt_ra_polar': ra_mag}
    if compute_per_virt:
        from mm25DGS_v6.data.ra_utils import adc_to_per_virt_range_profile
        with torch.no_grad():
            out['gt_per_virt'] = adc_to_per_virt_range_profile(gt_adc_ri)
    return out


def _build_m2_frame_bundle(adc_npy_path, poses_16, frame_idx, device,
                             n_dop=None):
    """v6 M2 per-frame supervision bundle.

    Loads all 16 chirps of ``adc_npy_path``, runs
    ``adc_to_rad_complex`` → |RAD| magnitude (real) — the v5-identical
    range + azimuth FFT chain extended with a matching Doppler FFT on
    the chirp axis — and returns a dict consumed by the M2 training
    loop.

    Pipeline (chirp → range → azimuth → Doppler, all v5-identical
    convention + Hann windowing + ifftshift/fft/drop-bin-0/fftshift
    applied across the chirp axis; see mm25DGS_v6/data/ra_utils.py
    and md/doppler_forward_model_plan.md for the full design).

    Returns a dict with:
      'frame_idx'       : int
      'poses_16'        : list of 16 per-chirp poses
      'gt_rad_mag'      : (D, 127, R) float32 |RAD|
      'gt_rad_max'      : float, used for mse_raw normalisation
                           (matches v5's precompute_gt_loss_norm
                           convention of dividing by max before MSE)
    """
    from mm25DGS_v6.data.ra_utils import (
        adc_to_rad_complex, N_DOP_DEFAULT,
    )
    if n_dop is None:
        n_dop = N_DOP_DEFAULT

    arr = np.load(adc_npy_path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'expected (16, RX, TX, K); got {arr.shape}')
    # (CH, TX, RX, ADC, 2) real-imag float32 on device
    adc = arr.transpose(0, 2, 1, 3)                                # (CH, TX, RX, ADC)
    ri = np.stack([adc.real, adc.imag], axis=-1).astype(np.float32)
    ri_t = torch.from_numpy(ri).to(device)
    with torch.no_grad():
        rad = adc_to_rad_complex(ri_t, n_dop=n_dop)                # (D, 127, R) cpx
        mag = rad.abs().float()
        mag_max = mag.amax().clamp_min(1e-30).item()
    return {
        'frame_idx': int(frame_idx),
        'poses_16': list(poses_16),
        'gt_rad_mag': mag,
        'gt_rad_max': float(mag_max),
    }


def build_frame_level_dataset(scene, train_frames, test_frame,
                               held_out_loop, use_pass2,
                               train_loops=None,
                               loss_type='mse_raw',
                               loop_dt_s=7.87e-3 / 16.0,
                               frame_period_s=0.1,
                               data_root='/home/adnan/Desktop/mm3DGS/data',
                               device=DEVICE,
                               anchor_source='pass2_lerp',
                               v6_milestone='M1',
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

    # M1.5+: precompute the per-virtual complex GT for the training
    # samples (M1.5 chirp 0 only; M2+ all 16 chirps). Stored per sample
    # at ~200 KB — negligible. Also built for hybrid_mag variant which
    # needs per-antenna magnitudes on 192 virts.
    need_per_virt = str(v6_milestone) in {'M1_5'}
    # M2: build one per-frame RAD bundle per train frame. Trainer loops
    # over bundles instead of per-chirp samples.
    build_m2_bundles = str(v6_milestone) == 'M2'
    m2_bundles = []

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
            gt = _build_per_loop_gt(adc_npy, k, loss_type, device,
                                     compute_per_virt=need_per_virt)
            train_samples.append({
                'frame_idx': int(f), 'loop_idx': int(k),
                'pose': poses[k],
                **gt,
            })
        if build_m2_bundles:
            m2_bundles.append(_build_m2_frame_bundle(
                adc_npy, poses, frame_idx=f, device=device))

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
        # v6 M1: carry the ADC path on the test sample so the final
        # diagnostic (normalised-gram correlation at test pose) can
        # load the held-out chirp directly without re-walking the data
        # layout. The GT per-virt tensor is derived lazily at
        # diagnostic time to avoid permanently holding 200 KB per
        # test sample.
        'adc_npy': test_adc,
        **test_gt,
    }

    diag = {
        'n_train_samples': len(train_samples),
        'edge_frames': edges,
        'test_pose_mode': test_mode,
        'n_m2_bundles': len(m2_bundles),
    }
    if verbose:
        print(f'  [data] train_samples: {len(train_samples)} '
              f'({len(train_frames)} frames × 16 loops)')
        if edges:
            print(f'  [data] edge frames (fell back to own pose): {edges}')
        if m2_bundles:
            print(f'  [data] M2 per-frame RAD bundles: {len(m2_bundles)}  '
                  f'(each D={m2_bundles[0]["gt_rad_mag"].shape[0]}, '
                  f'A={m2_bundles[0]["gt_rad_mag"].shape[1]}, '
                  f'R={m2_bundles[0]["gt_rad_mag"].shape[2]})')
        print(f'  [data] test (frame={test_frame} loop={held_out_loop})  '
              f'pose_mode={test_mode}')
    return train_samples, test_sample, diag, m2_bundles


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

def _save_ra_pngs_and_cc_history(
    output_dir, train_samples, test_sample, sample_grid, range_res,
    history, best_iter, model, rast, vertex_areas, active_mask, verbose,
):
    """Save per-frame rendered + GT RA Cartesian PNGs (linear + dB) for
    each train/test sample at the **best iteration's** model state, plus
    a cc_history.png plotting train mean cc and test cc over iters.

    Output layout (mirrors /home/adnan/Desktop/mm3DGS/output/RA/
    <scene>/):
      <output_dir>/frame_<F>_{train|test}/
          gt_ra_dB.png, gt_ra_linear.png
          rasterized_ra_dB.png, rasterized_ra_linear.png
          ra_gt_cart.npy, ra_rendered_cart.npy
      <output_dir>/cc_history.png
    """
    from mmir.data.ra_utils import save_ra_cartesian_png
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _render_and_cart(sample):
        apply_pose(rast, sample['pose'])
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
        return ra_cart.cpu().numpy()

    def _gt_cart(sample):
        with torch.no_grad():
            ra_cart = polar_to_cart_torch(sample['gt_ra_polar'], sample_grid)
        return ra_cart.cpu().numpy()

    def _save_frame(sample, tag):
        f = int(sample['frame_idx'])
        frame_dir = os.path.join(output_dir, f'frame_{f:03d}_{tag}')
        os.makedirs(frame_dir, exist_ok=True)
        ra_rend = _render_and_cart(sample)
        ra_gt = _gt_cart(sample)
        np.save(os.path.join(frame_dir, 'ra_rendered_cart.npy'), ra_rend)
        np.save(os.path.join(frame_dir, 'ra_gt_cart.npy'), ra_gt)
        for scale in ('dB', 'linear'):
            save_ra_cartesian_png(
                ra_gt,
                os.path.join(frame_dir, f'gt_ra_{scale}.png'),
                range_res=range_res, scale=scale,
                title=f'GT ({scale}) — frame {f} ({tag})',
            )
            save_ra_cartesian_png(
                ra_rend,
                os.path.join(frame_dir, f'rasterized_ra_{scale}.png'),
                range_res=range_res, scale=scale,
                title=f'Rasterized ({scale}) — frame {f} ({tag}) '
                      f'[best iter {best_iter}]',
            )
        return frame_dir

    for s in train_samples:
        _save_frame(s, 'train')
    _save_frame(test_sample, 'test')

    # cc history plot
    iters_np = np.array([h['iter'] for h in history])
    tcc_np = np.array([h.get('mean_train_cc', float('nan')) for h in history])
    ecc_np = np.array([h.get('test_cc', float('nan')) for h in history])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    ax.plot(iters_np, tcc_np, '-', color='tab:blue', label='train mean cc',
            linewidth=1.3)
    ax.plot(iters_np, ecc_np, '-', color='tab:red', label='test cc',
            linewidth=1.3)
    ax.axvline(best_iter, color='gray', linestyle=':', linewidth=1,
               label=f'best iter ({best_iter})')
    ax.set_xlabel('iteration')
    ax.set_ylabel('cart_corr')
    ax.set_title('CC history')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cc_history.png'), dpi=100,
                bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f'  [ra] saved {len(train_samples) + 1} frame-dirs + '
              f'cc_history.png under {output_dir}')


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
                    v6_milestone='M1',
                    v6_loss_variant='frobenius',
                    lambda_mag=1.0,
                    save_ra_pngs=False):
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
    train_samples, test_sample, diag, m2_bundles = build_frame_level_dataset(
        scene=scene, train_frames=train_frames, test_frame=test_frame,
        held_out_loop=held_out_loop, use_pass2=use_pass2_alignment,
        train_loops=train_loops,
        loss_type=loss_type, loop_dt_s=loop_dt_s,
        frame_period_s=frame_period_s, data_root=data_root,
        anchor_source=anchor_source,
        v6_milestone=v6_milestone,
        device=DEVICE, verbose=verbose,
    )

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
    # v6 M1.5: swap the v5 FFT-RA ``mse_raw`` training loss for the
    # un-normalised Gram Frobenius² loss on the per-virtual complex
    # range profile (chirp 0 only; no Doppler axis at M1.5). This
    # supervises the full rank-1 per-range Gram matrix C = v v^H:
    # diagonal ⇒ per-antenna magnitude, upper-tri ⇒ pairwise relative
    # phase. cart_corr is still reported per-sample for history /
    # best-state selection, but NOT used as the training objective.
    # final_test_cc (v5 FFT-RA path) is still the promotion metric —
    # unchanged from v5.
    use_gram_loss = (str(v6_milestone) == 'M1_5'
                      and str(v6_loss_variant) != 'hybrid_mag')
    use_hybrid_mag = (str(v6_milestone) == 'M1_5'
                      and str(v6_loss_variant) == 'hybrid_mag')
    use_m2 = str(v6_milestone) == 'M2'
    if use_m2:
        from mm25DGS_v6.data.ra_utils import (
            rp_stack_to_rad_complex, N_DOP_DEFAULT,
        )
        assert len(m2_bundles) == len(train_frames), (
            f'expected one M2 bundle per train_frame; got '
            f'{len(m2_bundles)} vs {len(train_frames)}')
        if verbose:
            print(f'  [M2] training loss = v5 mse_raw on |RAD| cube '
                  f'(D={N_DOP_DEFAULT - 1}, Az=127, R=256); rendering '
                  f'16 chirps/frame × {len(m2_bundles)} frames per iter')
        m2_loss_scale = 1.0 / len(m2_bundles)
    gram_loss_fn = None
    gram_loss_kwargs = {}
    if use_hybrid_mag:
        from mm25DGS_v6.data.ra_utils import reorder_rendered_to_adc
        if verbose:
            print(f'  [M1.5] training loss = v5 mse_raw + '
                  f'{lambda_mag:g} · per-antenna magnitude MSE on '
                  f'192 virts')
    if use_gram_loss:
        from mm25DGS_v6.losses import rd_losses as _rdl
        from mm25DGS_v6.data.ra_utils import (
            reorder_rendered_to_adc,
            virt_positions_adc_order,
            baseline_class_map,
        )
        variant = str(v6_loss_variant)
        if variant == 'frobenius':
            gram_loss_fn = _rdl.gram_frobenius_loss
        elif variant == 'coarray':
            _virt_pos = virt_positions_adc_order()
            _pi, _pj, _bi, _nb = baseline_class_map(_virt_pos)
            gram_loss_kwargs = dict(
                pair_i=_pi.to(DEVICE), pair_j=_pj.to(DEVICE),
                baseline_idx=_bi.to(DEVICE), n_baselines=_nb,
            )
            gram_loss_fn = _rdl.coarray_loss
        elif variant == 'smooth_alpha':
            gram_loss_kwargs = dict(smooth_k=11)
            gram_loss_fn = _rdl.smooth_alpha_gram_loss
        elif variant == 'mag_weighted':
            gram_loss_fn = _rdl.mag_weighted_gram_loss
        elif variant == 'baseline_weighted':
            _virt_pos = virt_positions_adc_order().to(DEVICE)
            gram_loss_kwargs = dict(virt_positions=_virt_pos, sigma=30.0)
            gram_loss_fn = _rdl.baseline_weighted_gram_loss
        elif variant == 'inv_variance':
            _virt_pos = virt_positions_adc_order().to(DEVICE)
            gram_loss_kwargs = dict(virt_positions=_virt_pos, L_scale=30.0)
            gram_loss_fn = _rdl.inv_variance_gram_loss
        elif variant == 'diag_offdiag':
            gram_loss_kwargs = dict(lambda_diag=1.0, lambda_off=0.0)
            gram_loss_fn = _rdl.diag_offdiag_gram_loss
        elif variant == 'range_integrated':
            gram_loss_fn = _rdl.range_integrated_gram_loss
        elif variant == 'modulus':
            gram_loss_fn = _rdl.modulus_gram_loss
        elif variant in ('baseline_binned',
                         'baseline_binned_wiener',
                         'baseline_binned_wiener_skipb0'):
            import math
            _virt_pos = virt_positions_adc_order()
            _pi, _pj, _bi, _nb = baseline_class_map(_virt_pos)
            # Multiplicity per bin
            mult = torch.zeros(_nb, dtype=torch.long)
            mult.scatter_add_(0, _bi, torch.ones_like(_bi))
            # b0 bin = baseline (0, 0)
            b0_rows = ((_virt_pos[_pi] == _virt_pos[_pj]).all(dim=-1))
            b0_bin = int(_bi[b0_rows][0].item())
            # Compute per-bin baseline magnitudes (grid units; 1 grid
            # unit = λ/2 for MMWCAS) for the Wiener weighting.
            unique_b = torch.zeros(_nb, 2, dtype=torch.float32)
            # Build unique baseline vectors by iterating over pairs once
            # — simpler than torch.unique on the 2D array.
            baseline_vecs = (_virt_pos[_pj].to(torch.float32)
                             - _virt_pos[_pi].to(torch.float32))
            # For each bin, pick any pair's baseline (they're all equal).
            seen = torch.zeros(_nb, dtype=torch.bool)
            for k in range(_pi.shape[0]):
                b = int(_bi[k].item())
                if not seen[b]:
                    unique_b[b] = baseline_vecs[k]
                    seen[b] = True
            # |Δ| in units of λ/2
            b_norm = unique_b.norm(dim=-1)
            if variant == 'baseline_binned':
                bw = None
            else:
                # Wiener-style baseline weight, σ_θ in radians.
                # At our HO_8 pose error ~5 cm, λ ≈ 4 mm, and typical
                # scatterer distance ~10 m: σ_θ ~ arctan(0.05/10) ~ 5
                # mrad (~0.3°). But the phase-error variance depends on
                # pose error and baseline length together (see
                # baseline_binned_loss_design.md §'Pose noise'). A
                # practical range to test is σ_θ in [0.01, 0.1] rad; we
                # pick 0.03 rad ≈ 1.7° as the default midpoint.
                sigma_theta = 0.03
                lam_grid = 2.0  # baseline is in units of λ/2, so "λ" = 2 grid units
                arg = (2.0 * math.pi * b_norm * sigma_theta / lam_grid)
                bw = 1.0 / (1.0 + arg.pow(2))
                # Boost b=0 weight to keep magnitude term fully weighted
                bw[b0_bin] = 1.0

            gram_loss_kwargs = dict(
                pair_i=_pi.to(DEVICE), pair_j=_pj.to(DEVICE),
                bin_idx=_bi.to(DEVICE),
                multiplicity=mult.to(DEVICE),
                baseline_weights=(None if bw is None else bw.to(DEVICE)),
                skip_b0=(variant == 'baseline_binned_wiener_skipb0'),
                b0_bin=b0_bin,
            )
            gram_loss_fn = _rdl.baseline_binned_loss
        else:
            raise ValueError(f'unknown v6_loss_variant: {variant}')
        if verbose:
            print(f'  [M1.5] training loss = {variant!r} on '
                  f'(N_virt=192, R=256) complex per-virt tensor '
                  f'(chirp 0 only; no Doppler axis)')

    best_mean_train_cc = -1.0
    best_iter = 0
    best_state = None
    history = []
    t0 = time.time()

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        per_sample_cc = []
        loss_sum = 0.0

        if use_m2:
            # v6 M2 — per-frame RAD-cube training.
            # For each train frame: render 16 chirps at 16 LERP-
            # interpolated per-chirp poses, stack → (16, TX, RX, R)
            # complex, run rp_stack_to_rad_complex (v5 azimuth FFT per
            # chirp + slow-time Doppler FFT on chirp axis, both with
            # Hann + ifftshift + drop-bin-0 + fftshift convention) →
            # (D, 127, R) complex → magnitude → mse_raw vs GT |RAD|.
            # Backward per frame to keep peak memory bounded (16-chirp
            # render graph ≈ 1.6 GB per frame, vs 8×16=128 chirps
            # retained simultaneously).
            for bundle in m2_bundles:
                poses_16 = bundle['poses_16']
                rp_list = []
                for k in range(16):
                    apply_pose(rast, poses_16[k])
                    rp_r, rp_i = render_gaussians(
                        model, rast, vertex_areas=vertex_areas,
                        active_mask=active_mask, shadow_mask=None,
                        bsdf_mode='full', disabled_components=None)
                    rp_list.append(torch.complex(rp_r, rp_i))
                rp_stack = torch.stack(rp_list, dim=0)
                # (16, TX, RX, R) complex → (D, 127, R) complex
                rad_pred = rp_stack_to_rad_complex(rp_stack,
                                                    n_dop=N_DOP_DEFAULT)
                mag_pred = rad_pred.abs()
                gt_max  = bundle['gt_rad_max']
                gt_mag  = bundle['gt_rad_mag']
                # mse_raw convention (v5's): normalise both pred and GT
                # by GT's max before MSE. Equivalent to mean((p − g)²) /
                # gt_max². Kept scale-comparable to v5's |RA|-only loss.
                loss_k = (mag_pred - gt_mag).pow(2).mean() / (gt_max ** 2)
                (loss_k * m2_loss_scale).backward()
                loss_sum += float(loss_k.item()) * m2_loss_scale
                # Chirp-0 RA for per-sample cc tracking (matches v5 /
                # mean_train_cc history).
                with torch.no_grad():
                    rp0_r = rp_list[0].real
                    rp0_i = rp_list[0].imag
                    ra_polar = range_profile_to_ra_mag(rp0_r, rp0_i)
                    ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
                    # Find the matching train_sample for this frame's chirp
                    # 0 to get gt_cart_norm (built by _finalize_cart at init).
                    # train_samples is sorted (frame, loop); frame's first
                    # entry corresponds to loop 0.
                    for s in train_samples:
                        if (int(s['frame_idx']) == bundle['frame_idx']
                                and int(s['loop_idx']) == 0):
                            cc = cart_corr_torch(ra_cart, s['gt_cart_norm']).item()
                            per_sample_cc.append(cc)
                            break
                del rp_stack, rad_pred, mag_pred, loss_k, rp_list
                torch.cuda.empty_cache()

            # Fall through to regularisers / optimiser step below.
            # Skip the per-sample loop.
            pass
        for s in (train_samples if not use_m2 else []):
            apply_pose(rast, s['pose'])
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            if use_gram_loss:
                # Renderer emits (n_tx=12, n_rx=16, K=256) in CONFIG TX
                # order. Reorder to ADC channel order, reshape to
                # (192, 256) complex to match GT layout from
                # adc_to_per_virt_range_profile.
                rend_c = torch.complex(rp_real, rp_imag)
                rend_c = reorder_rendered_to_adc(rend_c)
                n_tx_r, n_rx_r, n_range_r = rend_c.shape
                rend_per_virt = rend_c.reshape(n_tx_r * n_rx_r, n_range_r)
                loss_k = gram_loss_fn(
                    rend_per_virt, s['gt_per_virt'], **gram_loss_kwargs)
            elif use_hybrid_mag:
                # L = L_v5 + λ · per-antenna magnitude MSE on 192 virts.
                # Preserves v5's pose-robust FFT-magnitude inductive bias
                # and adds the learnable 192-virt per-antenna magnitude
                # signal on top. Per-antenna magnitude IS learnable at
                # HO_8 (diag_gram_mag rises above v5 in every Gram
                # variant tested); phase is not.
                loss_v5, _ = compute_ra_loss_rp(
                    rp_real, rp_imag, s['gt_loss'], loss_type=loss_type)
                rend_c = torch.complex(rp_real, rp_imag)
                rend_c = reorder_rendered_to_adc(rend_c)
                rend_per_virt = rend_c.reshape(12 * 16, rend_c.shape[-1])
                mag_p = (rend_per_virt.real.pow(2)
                         + rend_per_virt.imag.pow(2)).sqrt()
                gt_pv = s['gt_per_virt']
                mag_g = (gt_pv.real.pow(2) + gt_pv.imag.pow(2)).sqrt()
                mag_denom = mag_g.pow(2).sum().clamp_min(1e-20)
                loss_mag = (mag_p - mag_g).pow(2).sum() / mag_denom
                loss_k = loss_v5 + lambda_mag * loss_mag
            else:
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
        # Per-iter test cc (held-out pose) — needed for the CC history
        # plot and for best-state selection under save_ra_pngs when the
        # user wants test-leading selection. Kept no-grad. Cheap: one
        # extra render per iter.
        test_cc_iter = _render_and_cart_corr(test_sample)
        history.append({
            'iter': it, 'loss': loss_sum,
            'mean_train_cc': mean_tc,
            'test_cc': test_cc_iter,
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

    # ── v6 M1 diagnostic: normalised-gram correlation at test pose ──
    # Two variants computed:
    #   * complex-gram cc: full complex inner product per range bin.
    #     At M1 (v5 loss on FFT-RA magnitude), this is expected to be
    #     NEAR ZERO (~1/N_virt = 1/192 ≈ 0.005) — *not* because phase
    #     is thrown away but because the v5 magnitude-RA loss supervises
    #     only the projection of per-virtual phase onto the azimuth-FFT
    #     magnitude subspace (~32k real DOF of the ~98k real DOF in the
    #     per-virtual complex tensor). The remaining ~66k DOF is the
    #     null space of the magnitude-after-FFT operator — it mixes
    #     physically-relevant pairwise-relative-phase info (e.g. the
    #     106 elevated-TX virtuals dropped by the v5 86-subset FFT)
    #     with measurement noise (hardware phase calibration residuals,
    #     quantisation, thermal). Under v5 training this null-space
    #     component is uniformly unconstrained → gram cc ~ 1/N_virt as
    #     residual phases look random across channels.
    #     M2 replaces the loss with a Gram correlation that also
    #     supervises the null-space component — whether that lifts
    #     final_test_cc depends empirically on the signal/noise ratio
    #     of the extra DOF.
    #   * magnitude-gram cc: same formula on magnitudes only. At M1
    #     this should track final_test_cc — it measures whether the
    #     per-virtual magnitude pattern agrees with GT irrespective
    #     of phase.
    #
    # Formula (per md/gram_vs_fft_derivation.md §10):
    #     g = |v_p^H v_g|² / (‖v_p‖² ‖v_g‖²)   averaged over range bins.
    #
    # NOTE — TX channel reorder: the rasterizer emits per-TX range
    # profiles in CONFIG ORDER (the order of ``tx_array`` in the scene
    # config JSON). The ADC GT is in hardware channel order. See
    # ``mm25DGS_v6/data/ra_utils.CONFIG_TX_TO_ADC_TX_PERM`` for the
    # mapping. We permute the rendered output to ADC order before
    # flattening so both sides are in the same channel layout.
    #
    # This is an ADDITIONAL metric; the v5 FFT-RA final_test_cc path
    # above is bit-identical to v5 and never altered at any v6
    # milestone. Reported in results.json as
    # diag_normalised_gram_cc_test (complex) +
    # diag_normalised_gram_cc_test_mag (magnitude-only).
    diag_gram_cc_test = None
    diag_gram_cc_test_mag = None
    try:
        from mm25DGS_v6.data.ra_utils import (
            adc_to_per_virt_range_profile,
            gram_correlation_mean,
            reorder_rendered_to_adc,
        )
        apply_pose(rast, test_sample['pose'])
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
            rend_c = torch.complex(rp_real, rp_imag)                # (n_tx, n_rx, K) config order
            rend_c = reorder_rendered_to_adc(rend_c)                # ADC order
            n_tx, n_rx, n_range = rend_c.shape
            rend_per_virt = rend_c.reshape(n_tx * n_rx, n_range)    # (192, K) complex
            # GT per-virt at the held-out chirp.
            arr = np.load(test_sample['adc_npy'])
            held_out_k = int(test_sample['loop_idx'])
            ri = np.stack(
                [arr[held_out_k].real, arr[held_out_k].imag], axis=-1
            ).astype(np.float32)
            ri = ri.transpose(1, 0, 2, 3)                          # (TX, RX, K, 2)
            gt_adc = torch.from_numpy(ri).to(DEVICE)
            gt_per_virt = adc_to_per_virt_range_profile(gt_adc)    # (192, K) ADC order
            diag_gram_cc_test = float(
                gram_correlation_mean(rend_per_virt, gt_per_virt).item()
            )
            # Magnitude-only variant (meaningful at M1).
            rend_mag = rend_per_virt.abs().to(torch.complex64)
            gt_mag = gt_per_virt.abs().to(torch.complex64)
            diag_gram_cc_test_mag = float(
                gram_correlation_mean(rend_mag, gt_mag).item()
            )
    except Exception as _e:
        if verbose:
            print(f'  [v6 diag] gram cc computation failed: {_e}')
        diag_gram_cc_test = None
        diag_gram_cc_test_mag = None

    if verbose:
        print(f'\n  [final] test  cc = {final_test_cc:.4f} '
              f'(init was {init_test_cc:.4f}, Δ {final_test_cc-init_test_cc:+.4f})')
        print(f'  [final] train mean cc = {final_train_mean:.4f}  '
              f'(best iter {best_iter})')
        if diag_gram_cc_test is not None:
            if str(v6_milestone) == 'M1_5':
                _tag_c = '[v6 M1.5 diag; this is the quantity the loss is directly driving]'
                _tag_m = '[v6 M1.5 diag; magnitude-only variant for sanity]'
            else:
                _tag_c = ('[v6 M1 diag; ~1/N_virt ≈ 0.005 expected — FFT-null-'
                          'space phase is unsupervised under v5 mag-RA loss]')
                _tag_m = '[v6 M1 diag; should track final_test_cc]'
            print(f'  [final] diag gram cc (test, complex)   = {diag_gram_cc_test:.4f}  '
                  f'{_tag_c}')
            print(f'  [final] diag gram cc (test, magnitude) = {diag_gram_cc_test_mag:.4f}  '
                  f'{_tag_m}')
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
        # v6: tag runs with their milestone label + write under v6/
        # output_frame_nvs/ (v5's dir stays untouched — design-doc invariant).
        tag = f'{tag}_v6{v6_milestone}'
        if str(v6_milestone) == 'M1_5' and str(v6_loss_variant) != 'frobenius':
            tag = f'{tag}_{v6_loss_variant}'
        output_dir = os.path.join(
            PROJECT_ROOT, 'mm25DGS_v6', 'output_frame_nvs', f'{scene}_{tag}')
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
        # v6 M1 diagnostic (always logged when available; never used to
        # promote a variant — final_test_cc above is the bit-identical-
        # to-v5 metric that drives all cross-milestone comparisons).
        'diag_normalised_gram_cc_test': (
            None if diag_gram_cc_test is None else float(diag_gram_cc_test)
        ),
        'diag_normalised_gram_cc_test_mag': (
            None if diag_gram_cc_test_mag is None
            else float(diag_gram_cc_test_mag)
        ),
        'v6_milestone': str(v6_milestone),
        'v6_loss_variant': str(v6_loss_variant),
        'lambda_mag': float(lambda_mag),
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
             mean_train_cc=np.array([h['mean_train_cc'] for h in history]),
             test_cc=np.array([h.get('test_cc', float('nan'))
                                for h in history]))
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, 'best_model.pt'))

    if save_ra_pngs:
        _save_ra_pngs_and_cc_history(
            output_dir=output_dir,
            train_samples=train_samples,
            test_sample=test_sample,
            sample_grid=sample_grid,
            range_res=range_res,
            history=history,
            best_iter=best_iter,
            model=model, rast=rast, vertex_areas=vertex_areas,
            active_mask=active_mask,
            verbose=verbose,
        )

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
    ap.add_argument('--v6_loss_variant', default='frobenius',
                    choices=['frobenius', 'coarray', 'smooth_alpha',
                             'mag_weighted', 'baseline_weighted',
                             'inv_variance', 'diag_offdiag',
                             'range_integrated', 'modulus',
                             'hybrid_mag',
                             'baseline_binned',
                             'baseline_binned_wiener',
                             'baseline_binned_wiener_skipb0'],
                    help='v6 M1.5 loss variant (ignored unless '
                         '--v6_milestone=M1_5). frobenius = baseline Gram '
                         'Frob²; others apply Tier-1/2/5 variants from the '
                         'design doc (coarray, smooth_alpha, mag_weighted, '
                         'baseline_weighted, inv_variance, diag_offdiag, '
                         'range_integrated, modulus).')
    ap.add_argument('--data_root', default='/home/adnan/Desktop/mm3DGS/data',
                    help='Root of preprocessed data (contains seq_* '
                         'dirs + alignment_data). Switch to '
                         '`/home/adnan/Desktop/mm3DGS/data_v2` to use '
                         'the extended-window ported preprocessing '
                         '(see md/extended_window_data_v2_plan.md).')
    ap.add_argument('--lambda_mag', type=float, default=1.0,
                    help='(hybrid_mag variant only) weight of the '
                         'per-antenna magnitude MSE term added to v5 '
                         'mse_raw.')
    ap.add_argument('--save_ra_pngs', action='store_true',
                    help='Save best-iter rendered + GT RA Cartesian '
                         'PNGs (linear + dB) per train/test frame, '
                         'plus cc_history.png.')
    ap.add_argument('--v6_milestone', default='M1',
                    choices=['M1', 'M1_5', 'M2', 'M3', 'M4'],
                    help='v6 milestone label. M1: plumb per-virt tensor + '
                         'log diagnostic gram-cc; training loss stays v5 '
                         'mse_raw (final_test_cc must match v5 within MC '
                         'noise). M1_5: activate the complex Gram loss on '
                         'the (192, 256) per-virt tensor (chirp 0 only, no '
                         'Doppler axis); final_test_cc must not regress v5 '
                         'baseline on any scene.')
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
        v6_loss_variant=args.v6_loss_variant,
        lambda_mag=args.lambda_mag,
        save_ra_pngs=args.save_ra_pngs,
        data_root=args.data_root,
        v6_milestone=args.v6_milestone)
