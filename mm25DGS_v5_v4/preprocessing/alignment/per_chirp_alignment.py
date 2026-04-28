"""Per-chirp two-pass alignment.

Motivation
----------
The existing per-frame alignment pipeline (``cascaded_alignment.run_all`` +
``pass2.run_pass2``) operates on chirp 0 of every cascaded frame — only
chirp 0's RA data is fed to the renderer-2-DOF and LiDAR-4-DOF objectives.
Downstream NVS trainers that need a pose for every chirp currently
interpolate between frame F-1 and F+1's aligned configs, which is a
crude transformation and does not use each chirp's own radar
measurement. This script produces a per-chirp aligned config using the
same two-pass pipeline, saving the result to
``data/alignment_data/<scene>/cascade/per_chirp/cascaded_frame_<F>_chirp<CC>_aligned.json``.

Per-chirp pass 1 (within each frame):
  1. Build one CUDA alignment context per frame (FPS + antenna patterns +
     sample grid + Rasterizer pose buffer). Amortised cost ≈ 1.5 s/frame.
  2. For each chirp k in 0..15:
     a. Update ctx.gt_cart_norm with chirp-k's ADC → RA → cart.
     b. Run renderer 2-DOF grid+refine on the CUDA ctx (≈ 4 s).
     c. Run lidar 4-DOF (cupy) with chirp-k's radar RA (≈ 2 s); re-score
        its pose through the CUDA ctx for a fair cart_corr vs renderer.
     d. Pick the higher-cc winner.

Per-chirp pass 2 (across all 144 chirps of a scene):
  1. Extract the (center_mm, boresight) time-series of all 9×16 per-chirp
     pass-1 poses, ordered by time (frame·T_frame + chirp·T_loop).
  2. LOWESS + Huber fit (same code as the per-frame Stage A).
  3. MAD-based outlier flag (hard: > 3 MAD, soft: 2–3 MAD).
  4. For flagged chirps only: re-run renderer 4-DOF + lidar 4-DOF *with a
     trajectory prior* centred on the smoothed pose (mirrors per-frame
     Stage B, but over the chirp trajectory). Winner picked by adjusted
     cart_corr with the same trajectory-consistency gate.
  5. Non-flagged chirps: use the pass-1 pose. Flagged chirps: use the
     winning pass-2 candidate (or the smoothed prior as fallback).
  6. Overwrite ``per_chirp/cascaded_frame_<F>_chirp<CC>_aligned.json``
     with the final pose. Only the pass-2 output is saved — pass-1 is
     intermediate and never persisted separately.

Usage
-----
    python -m mm25DGS_v5_v4.preprocessing.alignment.per_chirp_alignment \
        --scene seq_1_frame_438 --gpu 0

    # Both target scenes, auto-pick GPU
    python -m mm25DGS_v5_v4.preprocessing.alignment.per_chirp_alignment \
        --scenes seq_1_frame_438,seq_2_frame_105 --gpu 0

Output
------
    data/alignment_data/<scene>/cascade/per_chirp/
        cascaded_frame_<F>_chirp<CC>_aligned.json  (per-chirp pose)
        per_chirp_triage.json                       (Stage A output)
        per_chirp_summary.json                      (Stage B + overall stats)

Timing (1× RTX 4090):
    Per-frame ctx build       ~1.5 s
    Per-chirp pass-1          ~6 s
    Per-frame total           1.5 + 16·6 ≈ 100 s
    Per-scene total           9 · 100 ≈ 15 min (pass 1 only)
    Plus pass 2 (~5 min/scene for Stage A + B)
    Per scene with both passes: ~20 min.
"""

import os
import sys
import json
import glob
import time
import shutil
import argparse
import numpy as np
import torch

import mitsuba as mi
if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm25DGS_v5_v4.preprocessing.alignment.cascaded_renderer_cuda import (
    build_alignment_context, apply_pose_to_ctx, render_and_evaluate_cuda,
    update_gt_for_chirp, destroy_alignment_context,
)
from mm25DGS_v5_v4.preprocessing.alignment.cascaded_renderer import (
    apply_2dof_to_config, apply_4dof_to_config,
    grid_search_2dof, refine_2dof,
    grid_search_4dof, refine_4dof,
    save_aligned_config_from_pose,
)
from mm25DGS_v5_v4.preprocessing.alignment.pass2.trajectory_fit import (
    _lowess_1d, _median_abs_dev,
)


# ---------------------------------------------------------------------------
# Pass-1 per-chirp
# ---------------------------------------------------------------------------

def _pass1_chirp(ctx, base_config, verbose=False):
    """Run pass-1 renderer 2DOF + lidar 4DOF (CUDA-scored) on the current
    ctx.gt_cart_norm. Returns a dict with renderer/lidar scores + the
    winning pose (tx_mm, rx_mm, bore).
    """
    # ── Method 1: renderer 2-DOF ──
    grid_params, _, _ = grid_search_2dof(ctx, base_config, verbose=False)
    refined_params, cc_renderer, _ = refine_2dof(
        ctx, base_config, grid_params, max_evals=30, verbose=False)
    tx_mm_r, rx_mm_r, bs_r = apply_2dof_to_config(
        base_config, refined_params[0], refined_params[1])

    # ── Method 2: lidar 4-DOF ── (lazy import; CUDA/CuPy heavy)
    cc_lidar = None
    tx_mm_l, rx_mm_l, bs_l = None, None, None
    lidar_4dof = None
    try:
        import open3d as o3d
        from .cascaded_lidar_gpu import optimize_alignment_gpu
        from .cascaded_lidar import (
            load_point_cloud, load_radar_config, radar_gt_to_ra_map,
        )
        # We're being called in a per-frame loop, so these loads happen 16×
        # per frame. radar_gt_to_ra_map does np.load → fast (<50 ms); the
        # mesh load is the slower part. We'll cache those in the outer
        # function by passing them in.
    except Exception as e:
        if verbose:
            print(f'    [lidar] skip ({e})')

    return {
        'renderer': {
            'cc':   float(cc_renderer),
            'tx':   tx_mm_r, 'rx': rx_mm_r, 'bore': bs_r,
            'params_2dof': [float(refined_params[0]), float(refined_params[1])],
        },
        # Placeholder for lidar — the outer driver fills these in when it
        # has the frame-level cached lidar pcd + 16 precomputed ra_radar
        # maps.
        'lidar': None,
    }


def _lidar_chirp(ctx, base_config, lidar_pcd, radar_params, ra_radar_chirp,
                 prior_weight=0.0, verbose=False):
    """Run lidar 4-DOF with a per-chirp radar RA map. Returns dict with
    the CUDA-scored cc at the resulting pose (comparable to renderer cc).
    """
    from .cascaded_lidar_gpu import optimize_alignment_gpu

    _, base_origin, base_boresight = None, None, None
    # Derive from base_config (lidar method needs origin + boresight in
    # metres/unit-vec).
    tx = np.array([t['pos_mm'] for t in base_config['tx_array']], dtype=float) / 1000.0
    rx = np.array([r['pos_mm'] for r in base_config['rx_array']], dtype=float) / 1000.0
    base_origin = np.concatenate([tx, rx], axis=0).mean(axis=0)
    base_boresight = np.array(base_config['tx_array'][0]['boresight'], dtype=float)
    base_boresight = base_boresight / max(np.linalg.norm(base_boresight), 1e-9)

    max_range = radar_params['num_adc'] * radar_params['range_resolution']
    res = optimize_alignment_gpu(
        lidar_pcd, ra_radar_chirp, radar_params, base_boresight, base_origin,
        range_search_m=(-max_range * 0.1, max_range * 0.1),
        azimuth_search_deg=(-15.0, 15.0),
        rotation_elev_search_deg=(-10.0, 10.0),
        rotation_azim_search_deg=(-10.0, 10.0),
        metric='correlation', coarse_steps=5,
        near_field_m=1.5, batch_size=100, verbose=False)
    opt = res['optimal_params']
    params_4dof = [
        float(opt['delta_range_m']), float(opt['delta_azimuth_deg']),
        float(opt['rotation_elev_deg']), float(opt['rotation_azim_deg']),
    ]
    tx_mm_l, rx_mm_l, bs_l = apply_4dof_to_config(base_config, *params_4dof)

    # Score lidar-method pose through the same CUDA renderer for a fair
    # comparison with the renderer-method cc.
    apply_pose_to_ctx(ctx, tx_mm_l, rx_mm_l, bs_l)
    cc_lidar = float(render_and_evaluate_cuda(ctx))

    return {
        'cc':   cc_lidar,
        'tx':   tx_mm_l, 'rx': rx_mm_l, 'bore': bs_l,
        'params_4dof': params_4dof,
    }


# ---------------------------------------------------------------------------
# Scene-level driver: pass 1
# ---------------------------------------------------------------------------

def pass1_scene(scene, data_root, out_dir, target_n=30000, verbose=True):
    """Run per-chirp pass 1 for every (frame, chirp) in a scene.

    Returns ``poses`` dict keyed by (frame, chirp) → {tx, rx, bore, source,
    cc_renderer, cc_lidar, winner}. Does NOT persist to disk yet (pass 2
    consumes this dict).
    """
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    frames = []
    for p in sorted(glob.glob(os.path.join(
            align_dir, 'cascaded_frame_*_aligned_pass2.json'))):
        try:
            core = os.path.basename(p).replace(
                'cascaded_frame_', '').replace('_aligned_pass2.json', '')
            frames.append(int(core))
        except ValueError:
            pass
    if not frames:
        raise RuntimeError(f'{scene}: no pass-2 aligned configs found')
    if verbose:
        print(f'[{scene}] pass 1 — {len(frames)} frames × 16 chirps '
              f'= {len(frames) * 16} per-chirp alignments')

    os.makedirs(out_dir, exist_ok=True)
    poses = {}

    # Prepare a lidar pcd (shared across all (frame, chirp) — same scene mesh)
    lidar_pcd = None
    try:
        import open3d as o3d
        from .cascaded_lidar import load_point_cloud
        mesh_path = os.path.join(data_root, scene, 'scene', 'mesh.ply')
        pcl_path = os.path.join(data_root, scene, 'scene', 'pcl.npy')
        lidar_pcd = load_point_cloud(mesh_path)
        if not lidar_pcd.has_points() and os.path.isfile(pcl_path):
            pcl = np.load(pcl_path)
            lidar_pcd = o3d.geometry.PointCloud()
            lidar_pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
            if pcl.shape[1] >= 7:
                inten = pcl[:, 6].astype(np.float32)
                inten = (inten - inten.min()) / max(
                    float(inten.max() - inten.min()), 1e-9)
                lidar_pcd.colors = o3d.utility.Vector3dVector(
                    np.stack([inten, inten, inten], axis=1))
        if verbose:
            n_pts = len(np.asarray(lidar_pcd.points))
            print(f'[{scene}] lidar pcd: {n_pts:,} points (shared across all frames)')
    except Exception as e:
        if verbose:
            print(f'[{scene}] WARN lidar pcd load failed: {e}; lidar method will be skipped')
        lidar_pcd = None

    for frame in frames:
        t0 = time.time()
        base_cfg_path = os.path.join(
            data_root, scene, 'configs', f'cascaded_frame_{frame}.json')
        with open(base_cfg_path) as f:
            base_config = json.load(f)
        adc_npy = os.path.join(
            data_root, scene, 'radar', f'cascaded_frame_{frame}.npy')

        # Build ctx once per frame (chirp 0 initial GT)
        ctx = build_alignment_context(
            config_file=base_cfg_path,
            chirp_idx=0, target_n=target_n, verbose=False)

        # Precompute per-chirp ra_radar maps for the lidar method (if lidar available)
        ra_radar_chirps = [None] * 16
        radar_params = None
        if lidar_pcd is not None:
            from .cascaded_lidar import radar_gt_to_ra_map, load_radar_config
            radar_params, _, _ = load_radar_config(base_cfg_path)
            # We need per-chirp RA. Load ADC once, compute 16 RAs.
            arr = np.load(adc_npy)  # (16, RX, TX, K) complex
            import tempfile
            with tempfile.TemporaryDirectory() as tdir:
                for k in range(16):
                    # radar_gt_to_ra_map internally picks chirp 0, so we
                    # write a per-chirp npy where chirp 0 IS the desired chirp.
                    tmp = os.path.join(tdir, f'chirp_{k}.npy')
                    # Replicate chirp k as all 16 chirps — downstream code
                    # uses [0] which now == arr[k].
                    np.save(tmp, np.broadcast_to(arr[k:k+1], arr.shape).copy())
                    ra_radar_chirps[k] = radar_gt_to_ra_map(tmp)
                # tdir deleted here

        # Per-chirp alignment
        for chirp in range(16):
            # Update GT for this chirp (reuses ctx.sample_grid + model)
            update_gt_for_chirp(ctx, adc_npy, chirp)

            # Method 1: renderer 2-DOF
            m1 = _pass1_chirp(ctx, base_config, verbose=False)['renderer']

            # Method 2: lidar 4-DOF (if available)
            m2 = None
            if lidar_pcd is not None and ra_radar_chirps[chirp] is not None:
                try:
                    m2 = _lidar_chirp(
                        ctx, base_config, lidar_pcd, radar_params,
                        ra_radar_chirps[chirp], verbose=False)
                except Exception as e:
                    if verbose:
                        print(f'    [{scene} f={frame} c={chirp:02d}] lidar failed: {e}')

            # Winner (higher CUDA-scored cc)
            cand = [('renderer_2dof', m1)]
            if m2 is not None:
                cand.append(('lidar_4dof', m2))
            winner_name, winner = max(cand, key=lambda x: x[1]['cc'])

            poses[(frame, chirp)] = {
                'frame':  int(frame), 'chirp': int(chirp),
                'tx_pos_mm':   winner['tx'].tolist(),
                'rx_pos_mm':   winner['rx'].tolist(),
                'boresight':   winner['bore'].tolist(),
                'cc_renderer': float(m1['cc']),
                'cc_lidar':    None if m2 is None else float(m2['cc']),
                'winner':      winner_name,
                'winner_cc':   float(winner['cc']),
                'source':      'pass1',
            }

        destroy_alignment_context(ctx)
        if verbose:
            ccs = [poses[(frame, k)]['winner_cc'] for k in range(16)]
            winners = [poses[(frame, k)]['winner'] for k in range(16)]
            n_rend = sum(1 for w in winners if w == 'renderer_2dof')
            n_lid = sum(1 for w in winners if w == 'lidar_4dof')
            print(f'  f={frame}: mean cc={np.mean(ccs):.4f} '
                  f'std={np.std(ccs):.4f}  '
                  f'winners: renderer_2dof={n_rend} lidar_4dof={n_lid}  '
                  f'[{time.time()-t0:.0f}s]')

    return poses


# ---------------------------------------------------------------------------
# Pass 2 — Stage A: LOWESS smoothing across chirps
# ---------------------------------------------------------------------------

def pass2_smooth_and_flag(poses, frame_period_s=0.1, loop_dt_s=7.87e-3/16.0,
                          lowess_frac=0.1, mad_thresh_hard=3.0,
                          mad_thresh_soft=2.0):
    """Given pass-1 poses, fit a smoothed trajectory over time and flag
    outliers. Returns (smoothed_poses, triage) dicts keyed by (frame, chirp).
    """
    # Build time-ordered sequence
    keys = sorted(poses.keys())  # sorted by (frame, chirp)
    times = np.array([f * frame_period_s + c * loop_dt_s for f, c in keys])

    centers = np.array([[0.0, 0.0, 0.0]] * len(keys), dtype=float)
    bores = np.array([[0.0, 0.0, 0.0]] * len(keys), dtype=float)
    for i, k in enumerate(keys):
        tx = np.array(poses[k]['tx_pos_mm'])
        rx = np.array(poses[k]['rx_pos_mm'])
        centers[i] = np.vstack([tx, rx]).mean(axis=0)
        bores[i] = np.array(poses[k]['boresight'])
        bores[i] = bores[i] / max(np.linalg.norm(bores[i]), 1e-9)

    # Per-component LOWESS
    cs = np.column_stack([
        _lowess_1d(times, centers[:, 0], frac=lowess_frac),
        _lowess_1d(times, centers[:, 1], frac=lowess_frac),
        _lowess_1d(times, centers[:, 2], frac=lowess_frac),
    ])
    bs = np.column_stack([
        _lowess_1d(times, bores[:, 0], frac=lowess_frac),
        _lowess_1d(times, bores[:, 1], frac=lowess_frac),
        _lowess_1d(times, bores[:, 2], frac=lowess_frac),
    ])
    # Renormalise smoothed boresight
    bs = bs / np.maximum(np.linalg.norm(bs, axis=1, keepdims=True), 1e-9)

    # Residuals + MAD
    r_c = centers - cs
    r_b = bores - bs
    sigma = np.array([
        _median_abs_dev(r_c[:, 0]),
        _median_abs_dev(r_c[:, 1]),
        _median_abs_dev(r_c[:, 2]),
        _median_abs_dev(r_b[:, 0]),
        _median_abs_dev(r_b[:, 1]),
        _median_abs_dev(r_b[:, 2]),
    ])
    # Floor sigma so a perfectly smooth axis doesn't flag tiny wiggle
    sigma = np.maximum(sigma, np.array([1.0, 1.0, 1.0, 0.005, 0.005, 0.005]))

    smoothed_poses = {}
    triage = {}
    for i, k in enumerate(keys):
        tx = np.array(poses[k]['tx_pos_mm'])
        rx = np.array(poses[k]['rx_pos_mm'])
        rel_tx = tx - centers[i]
        rel_rx = rx - centers[i]
        new_center = cs[i]
        smoothed_poses[k] = {
            'tx_pos_mm':  (rel_tx + new_center).tolist(),
            'rx_pos_mm':  (rel_rx + new_center).tolist(),
            'boresight':  bs[i].tolist(),
            'center_mm':  new_center.tolist(),
        }
        r_mad = [
            r_c[i, 0] / sigma[0], r_c[i, 1] / sigma[1], r_c[i, 2] / sigma[2],
            r_b[i, 0] / sigma[3], r_b[i, 1] / sigma[4], r_b[i, 2] / sigma[5],
        ]
        max_abs = float(np.max(np.abs(r_mad)))
        status = ('hard' if max_abs > mad_thresh_hard else
                  'soft' if max_abs > mad_thresh_soft else 'ok')
        triage[k] = {
            'residuals_mad': [float(x) for x in r_mad],
            'max_abs_mad':   max_abs,
            'status':        status,
            'sigma':         sigma.tolist(),
        }
    return smoothed_poses, triage


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------

def run_per_chirp_for_scene(scene, data_root='/home/adnan/Desktop/mm3DGS/data',
                             target_n=30000, frame_period_s=0.1,
                             loop_dt_s=7.87e-3/16.0,
                             lowess_frac=0.1,
                             verbose=True):
    """Full per-chirp alignment (pass 1 + pass 2) for one scene."""
    t_total = time.time()

    out_dir = os.path.join(data_root, 'alignment_data', scene,
                           'cascade', 'per_chirp')
    os.makedirs(out_dir, exist_ok=True)

    # Pass 1
    t0 = time.time()
    poses_p1 = pass1_scene(scene, data_root, out_dir,
                           target_n=target_n, verbose=verbose)
    t_p1 = time.time() - t0
    if verbose:
        ccs = [v['winner_cc'] for v in poses_p1.values()]
        print(f'[{scene}] pass 1 done in {t_p1:.0f}s '
              f'({len(poses_p1)} chirps; mean cc={np.mean(ccs):.4f}, '
              f'std={np.std(ccs):.4f})')

    # Pass 2 (Stage A smoothing)
    t0 = time.time()
    smoothed, triage = pass2_smooth_and_flag(
        poses_p1, frame_period_s=frame_period_s, loop_dt_s=loop_dt_s,
        lowess_frac=lowess_frac)
    t_p2 = time.time() - t0

    # Decide final pose for each (frame, chirp):
    # - 'ok'/'soft': smoothed pose (minor correction, keeps trajectory coherent)
    # - 'hard': smoothed pose (conservative fallback). Full Stage B re-alignment
    #           with prior was deemed not worth the extra compute for per-chirp
    #           (sub-mm pose changes); if a subsequent analysis shows hard
    #           chirps are a bottleneck, we can add renderer/lidar-with-prior
    #           re-runs here.
    final = dict(smoothed)

    # Save per-chirp aligned configs (pass-2 output only — overwrites any
    # pass-1 intermediate if it existed). Preserve the original unaligned
    # config's non-pose fields (radar FMCW params) by loading the base cfg.
    n_frames = len(set(k[0] for k in final.keys()))
    for (frame, chirp), pose in final.items():
        base_cfg_path = os.path.join(
            data_root, scene, 'configs', f'cascaded_frame_{frame}.json')
        with open(base_cfg_path) as f:
            base_config = json.load(f)
        out_path = os.path.join(
            out_dir,
            f'cascaded_frame_{frame}_chirp{chirp:02d}_aligned.json')
        save_aligned_config_from_pose(
            base_config,
            np.array(pose['tx_pos_mm']),
            np.array(pose['rx_pos_mm']),
            np.array(pose['boresight']),
            out_path)

    # Save triage + summary
    triage_path = os.path.join(out_dir, 'per_chirp_triage.json')
    with open(triage_path, 'w') as f:
        json.dump({
            str(f'{fr}_{ch}'): tr for (fr, ch), tr in triage.items()
        }, f, indent=2)

    # Counts
    hard = [k for k, t in triage.items() if t['status'] == 'hard']
    soft = [k for k, t in triage.items() if t['status'] == 'soft']
    summary = {
        'scene': scene,
        'n_frames': n_frames,
        'n_chirps_total': len(final),
        'pass1_mean_cc': float(np.mean(
            [v['winner_cc'] for v in poses_p1.values()])),
        'pass1_std_cc': float(np.std(
            [v['winner_cc'] for v in poses_p1.values()])),
        'n_hard': len(hard),
        'n_soft': len(soft),
        'hard_chirps': [f'{f}_{c}' for f, c in hard],
        'soft_chirps': [f'{f}_{c}' for f, c in soft],
        'lowess_frac': lowess_frac,
        'time_pass1_s': float(t_p1),
        'time_pass2_s': float(t_p2),
        'time_total_s': float(time.time() - t_total),
        'pass1_winner_breakdown': {
            'renderer_2dof': sum(
                1 for v in poses_p1.values() if v['winner'] == 'renderer_2dof'),
            'lidar_4dof':    sum(
                1 for v in poses_p1.values() if v['winner'] == 'lidar_4dof'),
        },
    }
    summary_path = os.path.join(out_dir, 'per_chirp_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f'[{scene}] pass 2 done in {t_p2:.0f}s')
        print(f'[{scene}] outliers: hard={len(hard)} soft={len(soft)} '
              f'ok={len(final) - len(hard) - len(soft)}')
        print(f'[{scene}] total elapsed: {time.time() - t_total:.0f}s '
              f'({(time.time() - t_total)/60:.1f} min)')
        print(f'[{scene}] wrote {len(final)} per-chirp configs to {out_dir}')

    return summary


# ===========================================================================
# Stage 3 — LERP/GT/hybrid-anchored per-chirp refinement.
#
# The pass-1 + pass-2 pipeline above (independent per-chirp alignment +
# LOWESS smoothing) is deprecated for this workflow. Stage 3 anchors the
# per-chirp search on a trusted pose (pass-2 per-frame + optional GT
# relative motion) and accepts a refinement only when it beats the anchor
# by a margin that exceeds rendering noise. See
# md/per_chirp_alignment_stage3_plan.md for the full design rationale.
# ===========================================================================

from scipy.spatial.transform import Rotation, Slerp
import copy

# Scene → raw ColoRadar sequence directory. Must match the mapping in
# mmir.preprocessing.alignment.sc_trajectory_transfer.SEQ_MAP. Keep in
# sync if that file changes.
SCENE_TO_RAW_DIR = {
    'seq_0_frame_135': '2_28_2021_outdoors_run0',
    'seq_0_frame_390': '2_28_2021_outdoors_run0',
    'seq_0_frame_451': '2_28_2021_outdoors_run0',
    'seq_1_frame_185': '2_28_2021_outdoors_run1',
    'seq_1_frame_277': '2_28_2021_outdoors_run1',
    'seq_1_frame_438': '2_28_2021_outdoors_run1',
    'seq_2_frame_105': '2_28_2021_outdoors_run2',
    'seq_2_frame_160': '2_28_2021_outdoors_run2',
    'seq_2_frame_300': '2_28_2021_outdoors_run2',
}
COLORADAR_RAW_ROOT = '/home/adnan/Documents/Data/coloRadar/raw/kitti'
CALIB_TRANSFORMS = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'mmir', 'preprocessing', 'calib', 'transforms'))


def _load_T_bc():
    """4x4 base←cascade extrinsic from calib/transforms/base_to_cascade.txt."""
    with open(os.path.join(CALIB_TRANSFORMS, 'base_to_cascade.txt')) as f:
        lines = f.read().strip().split('\n')
    t = np.array([float(x) for x in lines[0].split()], dtype=float)
    q = np.array([float(x) for x in lines[1].split()], dtype=float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3]  = t
    return T


def _load_gt_for_scene(scene):
    """Return (ts, pos, q_xyzw) for the scene's raw ColoRadar sequence.

    Raises FileNotFoundError if the raw dataset isn't available — caller
    should catch this and fall back to ``anchor_source='lerp'``.
    """
    run_dir = SCENE_TO_RAW_DIR.get(scene)
    if run_dir is None:
        raise KeyError(f'unknown scene {scene!r} — extend SCENE_TO_RAW_DIR')
    seq = os.path.join(COLORADAR_RAW_ROOT, run_dir)
    ts_path = os.path.join(seq, 'groundtruth', 'timestamps.txt')
    data_path = os.path.join(seq, 'groundtruth', 'groundtruth_poses.txt')
    if not (os.path.isfile(ts_path) and os.path.isfile(data_path)):
        raise FileNotFoundError(f'GT files missing under {seq}')
    ts = np.loadtxt(ts_path)
    data = np.loadtxt(data_path)
    pos = data[:, :3].astype(float)
    q = data[:, 3:7].astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    return ts, pos, q


def _load_cascade_timestamps_for_scene(scene):
    """Cascade ADC-sample timestamps for the scene's raw ColoRadar sequence."""
    run_dir = SCENE_TO_RAW_DIR.get(scene)
    if run_dir is None:
        raise KeyError(f'unknown scene {scene!r}')
    p = os.path.join(COLORADAR_RAW_ROOT, run_dir,
                     'cascade', 'adc_samples', 'timestamps.txt')
    if not os.path.isfile(p):
        # Fallback to heatmap timestamps if ADC's aren't shipped for this seq
        p = os.path.join(COLORADAR_RAW_ROOT, run_dir,
                         'cascade', 'heatmaps', 'timestamps.txt')
    return np.loadtxt(p)


def _interp_gt_T_wb(t_target, gt_ts, gt_pos, gt_q):
    """Return T_wb (4x4) at t_target via linear pos + slerp quat."""
    t = float(np.clip(t_target, gt_ts[0], gt_ts[-1]))
    i_hi = int(np.clip(np.searchsorted(gt_ts, t, side='right'),
                       1, len(gt_ts) - 1))
    i_lo = i_hi - 1
    a = (t - gt_ts[i_lo]) / (gt_ts[i_hi] - gt_ts[i_lo] + 1e-12)
    pos = (1 - a) * gt_pos[i_lo] + a * gt_pos[i_hi]
    rot = Slerp([0, 1], Rotation.from_quat(gt_q[[i_lo, i_hi]]))(a).as_matrix()
    T = np.eye(4); T[:3, :3] = rot; T[:3, 3] = pos
    return T


def _load_pass2_config(scene, frame, data_root):
    """Load cascaded_frame_<F>_aligned_pass2.json as a dict."""
    p = os.path.join(data_root, 'alignment_data', scene, 'cascade',
                     f'cascaded_frame_{frame}_aligned_pass2.json')
    with open(p) as f:
        return json.load(f)


def _config_to_arrays(cfg):
    """Return per-antenna TX/RX positions (N_tx, 3) + (N_rx, 3) in mm,
    plus boresight unit vector (3,)."""
    tx = np.array([e['pos_mm'] for e in cfg['tx_array']], dtype=float)
    rx = np.array([e['pos_mm'] for e in cfg['rx_array']], dtype=float)
    b = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
    b /= np.linalg.norm(b).clip(min=1e-12)
    return tx, rx, b


def _apply_world_delta_to_config(cfg, delta_pos_mm, delta_R_world):
    """Apply a world-frame rigid-body delta (Δposition in mm, ΔR
    rotation matrix) to every antenna position in the cfg and rotate
    the boresight. Preserves non-pose fields. Returns a fresh dict.
    """
    new_cfg = copy.deepcopy(cfg)
    tx, rx, bore = _config_to_arrays(cfg)

    # Center of antennae so we rotate around the board centre (not origin)
    center = np.vstack([tx, rx]).mean(axis=0)
    tx_rel = tx - center
    rx_rel = rx - center

    # Rotate each antenna's offset from centre, then translate the new
    # centre by delta_pos_mm. (ΔR is a small rotation — the per-antenna
    # rigid offsets inside the board rotate with it.)
    tx_rot = tx_rel @ delta_R_world.T
    rx_rot = rx_rel @ delta_R_world.T
    new_center = center + delta_pos_mm
    tx_new = tx_rot + new_center
    rx_new = rx_rot + new_center
    bore_new = delta_R_world @ bore
    bore_new /= np.linalg.norm(bore_new).clip(min=1e-12)

    for i, t in enumerate(new_cfg['tx_array']):
        t['pos_mm']    = tx_new[i].tolist()
        t['boresight'] = bore_new.tolist()
    for i, r in enumerate(new_cfg['rx_array']):
        r['pos_mm']    = rx_new[i].tolist()
        r['boresight'] = bore_new.tolist()
    return new_cfg


# ---------------------------------------------------------------------------
# Stage 3 anchor constructors
# ---------------------------------------------------------------------------

def _anchor_lerp(scene, frame, chirp, data_root,
                 loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.2):
    """Anchor option A: LERP between pass-2 F-1 and F+1 at α = c / 16.

    Fractional α within the (F-1, F+1) span for chirp c:
        α = 0.5 + c · loop_dt / (2 · T_frame)
    Falls back to pass-2_F directly if either neighbour is missing
    (edge frames).
    """
    cfg_F = _load_pass2_config(scene, frame, data_root)
    try:
        cfg_A = _load_pass2_config(scene, frame - 1, data_root)
        cfg_B = _load_pass2_config(scene, frame + 1, data_root)
    except FileNotFoundError:
        return cfg_F, {'source': 'lerp_fallback_frame_own'}

    tx_A, rx_A, bore_A = _config_to_arrays(cfg_A)
    tx_B, rx_B, bore_B = _config_to_arrays(cfg_B)
    alpha = 0.5 + chirp * loop_dt_s / (2 * frame_period_s)
    tx = (1 - alpha) * tx_A + alpha * tx_B
    rx = (1 - alpha) * rx_A + alpha * rx_B
    bore = (1 - alpha) * bore_A + alpha * bore_B
    bore /= np.linalg.norm(bore).clip(min=1e-12)

    cfg = copy.deepcopy(cfg_F)   # keeps radar params (slope etc.) identical
    for i, t in enumerate(cfg['tx_array']):
        t['pos_mm']    = tx[i].tolist()
        t['boresight'] = bore.tolist()
    for i, r in enumerate(cfg['rx_array']):
        r['pos_mm']    = rx[i].tolist()
        r['boresight'] = bore.tolist()
    return cfg, {'source': 'lerp', 'alpha': alpha}


def _anchor_gt(scene, frame, chirp, data_root,
               loop_dt_s=7.87e-3 / 16.0):
    """Anchor option B: pure GT-derived pose at the chirp's timestamp.

    Discards pass-2's radar-data-driven absolute corrections. Included
    for ablation — not recommended for production (see plan §Option B
    cons — GT has a consistent ~5° boresight offset vs pass-2).
    """
    cfg_F = _load_pass2_config(scene, frame, data_root)
    gt_ts, gt_pos, gt_q = _load_gt_for_scene(scene)
    cas_ts = _load_cascade_timestamps_for_scene(scene)
    T_bc = _load_T_bc()

    t_chirp = cas_ts[frame] + chirp * loop_dt_s
    T_wb = _interp_gt_T_wb(t_chirp, gt_ts, gt_pos, gt_q)
    T_ws = T_wb @ T_bc
    center_mm = T_ws[:3, 3] * 1000.0
    bore = T_ws[:3, 0]
    bore /= np.linalg.norm(bore).clip(min=1e-12)

    # Rotate every per-antenna offset (from board centre) via T_ws' rotation.
    tx, rx, _ = _config_to_arrays(cfg_F)
    board_center = np.vstack([tx, rx]).mean(axis=0)
    tx_rel = tx - board_center
    rx_rel = rx - board_center
    # New world orientation → we need the rotation from cfg_F orientation
    # to T_ws orientation. Reconstruct cfg_F's orientation matrix first
    # (first-column = cfg_F boresight, second = cross(z, bore), third = cross).
    # Simpler: just use T_ws' first 3 columns directly as the new orientation.
    # But that uses per-antenna layout from cfg_F (already computed in some
    # sensor frame). What we actually want is: "take the sensor-frame layout
    # (which is scene-independent) and place it at the new world pose." The
    # sensor-frame layout is encoded in cfg_F's per-antenna pos_mm rotated
    # into T_ws' orientation.
    # Use cfg_F's orientation as the starting point:
    bore_F = np.array(cfg_F['tx_array'][0]['boresight'], dtype=float)
    bore_F /= np.linalg.norm(bore_F).clip(min=1e-12)
    # R_F and R_ws are both "world ← sensor" rotations; the relative
    # rotation we apply to per-antenna rel positions is
    # ΔR = R_ws @ R_F.T. But we only have first columns of R_F (bore_F)
    # and R_ws (bore). Approximate ΔR as a small rotation that aligns
    # bore_F → bore (shortest arc):
    cos_a = np.clip(np.dot(bore_F, bore), -1.0, 1.0)
    if cos_a > 0.9999:
        delta_R = np.eye(3)
    else:
        axis = np.cross(bore_F, bore)
        axis_n = np.linalg.norm(axis)
        if axis_n < 1e-9:
            delta_R = np.eye(3)
        else:
            axis /= axis_n
            angle = float(np.arccos(cos_a))
            delta_R = Rotation.from_rotvec(axis * angle).as_matrix()
    tx_new = tx_rel @ delta_R.T + center_mm
    rx_new = rx_rel @ delta_R.T + center_mm

    cfg = copy.deepcopy(cfg_F)
    for i, t in enumerate(cfg['tx_array']):
        t['pos_mm']    = tx_new[i].tolist()
        t['boresight'] = bore.tolist()
    for i, r in enumerate(cfg['rx_array']):
        r['pos_mm']    = rx_new[i].tolist()
        r['boresight'] = bore.tolist()
    return cfg, {'source': 'gt_absolute', 't_chirp': float(t_chirp)}


def _anchor_hybrid(scene, frame, chirp, data_root,
                   loop_dt_s=7.87e-3 / 16.0):
    """Anchor option C (recommended): pass-2 absolute at frame F + GT
    relative motion from chirp 0 to chirp c applied as a world-frame
    rigid-body delta.
    """
    cfg_F = _load_pass2_config(scene, frame, data_root)
    gt_ts, gt_pos, gt_q = _load_gt_for_scene(scene)
    cas_ts = _load_cascade_timestamps_for_scene(scene)
    T_bc = _load_T_bc()

    t_F     = cas_ts[frame]
    t_chirp = t_F + chirp * loop_dt_s
    T_ws_F  = _interp_gt_T_wb(t_F,     gt_ts, gt_pos, gt_q) @ T_bc
    T_ws_c  = _interp_gt_T_wb(t_chirp, gt_ts, gt_pos, gt_q) @ T_bc

    # Relative motion in world frame (GT-derived, sub-mm stable)
    delta_pos_m      = T_ws_c[:3, 3] - T_ws_F[:3, 3]          # metres
    delta_pos_mm     = delta_pos_m * 1000.0
    delta_R_world    = T_ws_c[:3, :3] @ T_ws_F[:3, :3].T      # small rot

    # Apply to pass-2 per-frame config
    cfg_out = _apply_world_delta_to_config(cfg_F, delta_pos_mm, delta_R_world)
    meta = {
        'source':              'hybrid',
        't_chirp':             float(t_chirp),
        'delta_pos_mm':        delta_pos_mm.tolist(),
        'delta_R_rotvec_deg':  np.degrees(
            Rotation.from_matrix(delta_R_world).as_rotvec()).tolist(),
    }
    return cfg_out, meta


def _compute_anchor(scene, frame, chirp, anchor_source, data_root,
                    loop_dt_s=7.87e-3 / 16.0, frame_period_s=0.2):
    """Dispatch to the requested anchor constructor. Falls back to LERP
    if GT / hybrid are requested but the raw dataset is unavailable.
    """
    try:
        if anchor_source == 'lerp':
            return _anchor_lerp(scene, frame, chirp, data_root,
                                loop_dt_s=loop_dt_s, frame_period_s=frame_period_s)
        elif anchor_source == 'gt':
            return _anchor_gt(scene, frame, chirp, data_root, loop_dt_s=loop_dt_s)
        elif anchor_source == 'hybrid':
            return _anchor_hybrid(scene, frame, chirp, data_root,
                                  loop_dt_s=loop_dt_s)
        else:
            raise ValueError(f'unknown anchor_source: {anchor_source!r}')
    except (FileNotFoundError, KeyError) as e:
        # GT / raw dataset unavailable — fall back silently
        cfg_fb, meta_fb = _anchor_lerp(scene, frame, chirp, data_root,
                                       loop_dt_s=loop_dt_s,
                                       frame_period_s=frame_period_s)
        meta_fb.update({'requested': anchor_source, 'fallback_reason': str(e)})
        meta_fb['source'] = f'{anchor_source}_fallback_to_lerp'
        return cfg_fb, meta_fb


# ---------------------------------------------------------------------------
# Stage 3 inner refinement
# ---------------------------------------------------------------------------

DEFAULT_STAGE3_SEARCH_RADIUS = {
    'range_m':      0.010,   # ±10 mm translation along boresight
    'azim_deg':     0.5,     # ±0.5° azimuth translation
    'elev_rot_deg': 0.5,     # ±0.5° elevation-rotation
    'azim_rot_deg': 0.5,     # ±0.5° azimuth-rotation
}
DEFAULT_STAGE3_GATE_MARGIN_CC = 0.005
DEFAULT_STAGE3_PRIOR_WEIGHT  = 0.05


def stage3_refine_chirp(ctx, anchor_cfg, search_radius=None,
                        prior_weight=DEFAULT_STAGE3_PRIOR_WEIGHT,
                        gate_margin=DEFAULT_STAGE3_GATE_MARGIN_CC,
                        verbose=False):
    """Inner refinement loop for Stage 3. ``ctx.gt_cart_norm`` should
    already be set to chirp k's GT. Returns a dict with:
        winner:       'anchor', 'renderer_4dof', or 'lidar_4dof'
        cc_anchor:    cc at the anchor pose
        cc_renderer:  cc at the best renderer-4DOF refinement
        cc_lidar:     cc at the best lidar-4DOF refinement (or None)
        chosen_cfg:   config dict of the accepted pose
        delta_4dof:   4DOF deltas of the accepted pose (0s if winner='anchor')
    """
    search_radius = search_radius or DEFAULT_STAGE3_SEARCH_RADIUS

    # 1. Score the anchor itself first (the no-op candidate).
    tx_anchor, rx_anchor, bore_anchor = _config_to_arrays(anchor_cfg)
    apply_pose_to_ctx(ctx, tx_anchor, rx_anchor, bore_anchor)
    cc_anchor = render_and_evaluate_cuda(ctx)

    # 2. Renderer-4DOF refinement around the anchor (range, azim, elev-rot, azim-rot)
    rng = search_radius['range_m']
    az  = search_radius['azim_deg']
    e   = search_radius['elev_rot_deg']
    z   = search_radius['azim_rot_deg']
    grid_params_r, _, _ = grid_search_4dof(
        ctx, anchor_cfg,
        range_min=-rng, range_max=rng, range_step=max(rng / 2, 0.002),
        az_min=-az,  az_max=az,  az_step=max(az / 2, 0.1),
        elev_rot_min=-e, elev_rot_max=e, elev_rot_step=max(e / 2, 0.1),
        azim_rot_min=-z, azim_rot_max=z, azim_rot_step=max(z / 2, 0.1),
        prior_weight=prior_weight, verbose=False)
    refined_r, cc_r, _ = refine_4dof(
        ctx, anchor_cfg, grid_params_r, max_evals=30,
        prior_weight=prior_weight, verbose=False)
    tx_r, rx_r, bore_r = apply_4dof_to_config(anchor_cfg, *refined_r)

    # 3. Lidar-4DOF refinement (via the v5 wrapper from per-frame Stage 2 B1-L).
    #    This is optional and gated on cupy/lidar availability. The outer
    #    driver pre-builds a lidar_pcd and per-chirp ra_radar and hands
    #    them to us; if either is None, skip.
    cc_l = None
    refined_l = None
    tx_l = rx_l = bore_l = None
    if getattr(ctx, '_lidar_pcd', None) is not None \
            and getattr(ctx, '_ra_radar_chirp', None) is not None:
        try:
            from .cascaded_lidar_gpu import optimize_alignment_gpu_with_prior

            # Derive base_origin + base_boresight from anchor for the lidar call
            base_origin = np.vstack([
                np.array([e_['pos_mm'] for e_ in anchor_cfg['tx_array']]),
                np.array([e_['pos_mm'] for e_ in anchor_cfg['rx_array']]),
            ]).mean(axis=0) / 1000.0
            base_boresight = np.array(anchor_cfg['tx_array'][0]['boresight'])
            base_boresight /= np.linalg.norm(base_boresight).clip(min=1e-12)

            lc = optimize_alignment_gpu_with_prior(
                ctx._lidar_pcd, ctx._ra_radar_chirp, ctx._radar_params,
                prior_boresight=base_boresight,
                prior_origin=base_origin,
                search_radius={
                    'range_m':      rng,
                    'azim_deg':     az,
                    'elev_rot_deg': e,
                    'azim_rot_deg': z,
                },
                prior_weight=prior_weight,
                metric='correlation', coarse_steps=5,
                near_field_m=1.5, batch_size=100, verbose=False,
            )
            opt = lc['optimal_params']
            refined_l = [opt['delta_range_m'], opt['delta_azimuth_deg'],
                         opt['rotation_elev_deg'], opt['rotation_azim_deg']]
            tx_l, rx_l, bore_l = apply_4dof_to_config(anchor_cfg, *refined_l)
            # Score through CUDA for a fair comparison with the renderer candidate
            apply_pose_to_ctx(ctx, tx_l, rx_l, bore_l)
            cc_l = float(render_and_evaluate_cuda(ctx))
        except Exception as ex:
            if verbose:
                print(f'    [lidar] failed: {ex}')
            cc_l = None

    # 4. Winner selection — anchor wins unless a refinement beats it
    #    by ≥ gate_margin.
    best_cc = cc_anchor
    winner = 'anchor'
    chosen_cfg = anchor_cfg
    chosen_delta = [0.0, 0.0, 0.0, 0.0]

    if cc_r is not None and cc_r > cc_anchor + gate_margin:
        best_cc = cc_r
        winner = 'renderer_4dof'
        chosen_cfg = _apply_world_delta_to_config(anchor_cfg, np.array([0, 0, 0]), np.eye(3))
        # Just write renderer-refined per-antenna positions/bore directly
        for i, e_ in enumerate(chosen_cfg['tx_array']):
            e_['pos_mm']    = tx_r[i].tolist()
            e_['boresight'] = bore_r.tolist()
        for i, e_ in enumerate(chosen_cfg['rx_array']):
            e_['pos_mm']    = rx_r[i].tolist()
            e_['boresight'] = bore_r.tolist()
        chosen_delta = [float(x) for x in refined_r]

    if cc_l is not None and cc_l > best_cc + gate_margin:
        best_cc = cc_l
        winner = 'lidar_4dof'
        chosen_cfg = copy.deepcopy(anchor_cfg)
        for i, e_ in enumerate(chosen_cfg['tx_array']):
            e_['pos_mm']    = tx_l[i].tolist()
            e_['boresight'] = bore_l.tolist()
        for i, e_ in enumerate(chosen_cfg['rx_array']):
            e_['pos_mm']    = rx_l[i].tolist()
            e_['boresight'] = bore_l.tolist()
        chosen_delta = [float(x) for x in refined_l]

    return {
        'winner':      winner,
        'cc_anchor':   float(cc_anchor),
        'cc_renderer': float(cc_r) if cc_r is not None else None,
        'cc_lidar':    float(cc_l) if cc_l is not None else None,
        'cc_winner':   float(best_cc),
        'chosen_cfg':  chosen_cfg,
        'delta_4dof':  chosen_delta,
    }


# ---------------------------------------------------------------------------
# Stage 3 scene driver
# ---------------------------------------------------------------------------

def run_stage3_for_scene(scene,
                         data_root='/home/adnan/Desktop/mm3DGS/data',
                         anchor_source='hybrid',
                         target_n=30000,
                         search_radius=None,
                         prior_weight=DEFAULT_STAGE3_PRIOR_WEIGHT,
                         gate_margin=DEFAULT_STAGE3_GATE_MARGIN_CC,
                         frame_period_s=0.2,
                         loop_dt_s=7.87e-3 / 16.0,
                         frames=None,
                         verbose=True):
    """Stage 3 per-chirp alignment for one scene. Writes:

        data/alignment_data/<scene>/cascade/per_chirp/
            cascaded_frame_<F>_chirp<CC>_aligned_pass3.json
            cascaded_frame_<F>_chirp<CC>_alignment_log_pass3.json
            pass3_summary.json

    Returns the summary dict.
    """
    t_total = time.time()
    out_dir = os.path.join(data_root, 'alignment_data', scene,
                           'cascade', 'per_chirp')
    os.makedirs(out_dir, exist_ok=True)

    # Discover frames with pass-2 alignment present
    align_dir = os.path.join(data_root, 'alignment_data', scene, 'cascade')
    if frames is None:
        discovered = []
        for p in sorted(glob.glob(os.path.join(
                align_dir, 'cascaded_frame_*_aligned_pass2.json'))):
            try:
                core = os.path.basename(p).replace(
                    'cascaded_frame_', '').replace('_aligned_pass2.json', '')
                discovered.append(int(core))
            except ValueError:
                pass
        frames = discovered

    if not frames:
        raise RuntimeError(
            f'{scene}: no pass-2 aligned configs; run Stage 2 first')

    if verbose:
        print(f'[{scene}] Stage 3 — anchor_source={anchor_source}  '
              f'frames={len(frames)}  chirps/frame=16  '
              f'gate_margin={gate_margin}  prior_weight={prior_weight}')

    # Optional lidar pcd for the lidar candidate
    lidar_pcd = None
    try:
        import open3d as o3d
        from .cascaded_lidar import load_point_cloud
        mesh_path = os.path.join(data_root, scene, 'scene', 'mesh.ply')
        lidar_pcd = load_point_cloud(mesh_path)
        pcl_path = os.path.join(data_root, scene, 'scene', 'pcl.npy')
        if not lidar_pcd.has_points() and os.path.isfile(pcl_path):
            pcl = np.load(pcl_path)
            lidar_pcd = o3d.geometry.PointCloud()
            lidar_pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
            if pcl.shape[1] >= 7:
                inten = pcl[:, 6].astype(np.float32)
                inten = (inten - inten.min()) / max(
                    float(inten.max() - inten.min()), 1e-9)
                lidar_pcd.colors = o3d.utility.Vector3dVector(
                    np.stack([inten, inten, inten], axis=1))
        if verbose and lidar_pcd.has_points():
            print(f'[{scene}] lidar pcd: {len(np.asarray(lidar_pcd.points)):,} points')
    except Exception as e:
        if verbose:
            print(f'[{scene}] lidar pcd unavailable: {e}')

    logs = []
    n_anchor = n_rend = n_lid = 0
    for frame in frames:
        t_f = time.time()
        base_cfg_path = os.path.join(
            data_root, scene, 'configs', f'cascaded_frame_{frame}.json')
        if not os.path.isfile(base_cfg_path):
            if verbose:
                print(f'[{scene}] f={frame} SKIP (unaligned base config missing)')
            continue
        adc_npy = os.path.join(
            data_root, scene, 'radar', f'cascaded_frame_{frame}.npy')

        # Build one CUDA alignment ctx per frame (FPS init reused across chirps)
        ctx = build_alignment_context(
            config_file=base_cfg_path,
            chirp_idx=0, target_n=target_n, verbose=False)
        ctx._lidar_pcd = lidar_pcd

        # Precompute per-chirp ra_radar maps for the lidar candidate (if lidar available)
        radar_params = None
        ra_radar_chirps = [None] * 16
        if lidar_pcd is not None and lidar_pcd.has_points():
            try:
                from .cascaded_lidar import radar_gt_to_ra_map, load_radar_config
                radar_params, _, _ = load_radar_config(base_cfg_path)
                ctx._radar_params = radar_params
                arr = np.load(adc_npy)
                import tempfile
                with tempfile.TemporaryDirectory() as tdir:
                    for k in range(16):
                        tmp = os.path.join(tdir, f'chirp_{k}.npy')
                        np.save(tmp, np.broadcast_to(arr[k:k+1], arr.shape).copy())
                        ra_radar_chirps[k] = radar_gt_to_ra_map(tmp)
            except Exception as e:
                if verbose:
                    print(f'[{scene}] f={frame} lidar RA precompute failed: {e}')
                ra_radar_chirps = [None] * 16

        # Per chirp
        for chirp in range(16):
            # Chirp 0 is special: it already equals pass-2_F under the hybrid
            # anchor, and pass-2_F is a data-validated pose. Skip refinement
            # and write pass-2_F directly.
            if chirp == 0 and anchor_source in ('hybrid', 'lerp'):
                anchor_cfg, anchor_meta = _compute_anchor(
                    scene, frame, chirp, anchor_source, data_root,
                    loop_dt_s=loop_dt_s, frame_period_s=frame_period_s)
                update_gt_for_chirp(ctx, adc_npy, chirp)
                tx0, rx0, bore0 = _config_to_arrays(anchor_cfg)
                apply_pose_to_ctx(ctx, tx0, rx0, bore0)
                cc_anchor = render_and_evaluate_cuda(ctx)
                refine_result = {
                    'winner':      'anchor',
                    'cc_anchor':   float(cc_anchor),
                    'cc_renderer': None,
                    'cc_lidar':    None,
                    'cc_winner':   float(cc_anchor),
                    'chosen_cfg':  anchor_cfg,
                    'delta_4dof':  [0.0, 0.0, 0.0, 0.0],
                }
            else:
                # Compute anchor
                anchor_cfg, anchor_meta = _compute_anchor(
                    scene, frame, chirp, anchor_source, data_root,
                    loop_dt_s=loop_dt_s, frame_period_s=frame_period_s)
                # Update ctx GT for this chirp
                update_gt_for_chirp(ctx, adc_npy, chirp)
                ctx._ra_radar_chirp = ra_radar_chirps[chirp]
                # Refine
                refine_result = stage3_refine_chirp(
                    ctx, anchor_cfg,
                    search_radius=search_radius,
                    prior_weight=prior_weight,
                    gate_margin=gate_margin,
                    verbose=False)

            # Count winners
            w = refine_result['winner']
            if   w == 'anchor':        n_anchor += 1
            elif w == 'renderer_4dof': n_rend   += 1
            elif w == 'lidar_4dof':    n_lid    += 1

            # Save chosen config
            out_cfg_path = os.path.join(
                out_dir,
                f'cascaded_frame_{frame}_chirp{chirp:02d}_aligned_pass3.json')
            with open(out_cfg_path, 'w') as f:
                json.dump(refine_result['chosen_cfg'], f, indent=2)

            # Save per-chirp log
            log = {
                'scene':         scene,
                'frame':         int(frame),
                'chirp':         int(chirp),
                'anchor_source': anchor_source,
                'anchor_meta':   anchor_meta if chirp != 0 or anchor_source not in ('hybrid', 'lerp')
                                               else {'source': f'{anchor_source}_chirp0_equals_pass2_F'},
                'cc_anchor':     refine_result['cc_anchor'],
                'cc_renderer':   refine_result['cc_renderer'],
                'cc_lidar':      refine_result['cc_lidar'],
                'cc_winner':     refine_result['cc_winner'],
                'winner':        refine_result['winner'],
                'delta_4dof':    refine_result['delta_4dof'],
                'aligned_pass3_path': out_cfg_path,
            }
            with open(os.path.join(
                    out_dir,
                    f'cascaded_frame_{frame}_chirp{chirp:02d}_alignment_log_pass3.json'),
                    'w') as f:
                json.dump(log, f, indent=2)
            logs.append(log)

        destroy_alignment_context(ctx)
        if verbose:
            anc_ccs = [l['cc_anchor'] for l in logs[-16:]]
            win_ccs = [l['cc_winner'] for l in logs[-16:]]
            improved = sum(1 for l in logs[-16:] if l['winner'] != 'anchor')
            print(f'  f={frame}: anchor cc mean={np.mean(anc_ccs):.4f}  '
                  f'winner cc mean={np.mean(win_ccs):.4f}  '
                  f'refined {improved}/16  '
                  f'[{time.time()-t_f:.0f}s]')

    # Scene summary
    anchor_ccs = [l['cc_anchor'] for l in logs]
    winner_ccs = [l['cc_winner'] for l in logs]
    summary = {
        'scene':              scene,
        'anchor_source':      anchor_source,
        'frames':             sorted(set(l['frame'] for l in logs)),
        'n_chirps_total':     len(logs),
        'mean_anchor_cc':     float(np.mean(anchor_ccs)) if anchor_ccs else None,
        'mean_winner_cc':     float(np.mean(winner_ccs)) if winner_ccs else None,
        'median_cc_gain':     float(np.median([l['cc_winner'] - l['cc_anchor']
                                               for l in logs])) if logs else None,
        'winner_breakdown': {
            'anchor':        n_anchor,
            'renderer_4dof': n_rend,
            'lidar_4dof':    n_lid,
        },
        'search_radius':      search_radius or DEFAULT_STAGE3_SEARCH_RADIUS,
        'gate_margin':        gate_margin,
        'prior_weight':       prior_weight,
        'time_total_s':       float(time.time() - t_total),
    }
    with open(os.path.join(out_dir, 'pass3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f'[{scene}] Stage 3 done in {summary["time_total_s"]:.0f}s '
              f'({summary["time_total_s"]/60:.1f} min)')
        print(f'  winners: anchor={n_anchor}  '
              f'renderer_4dof={n_rend}  lidar_4dof={n_lid}')
        print(f'  cc: anchor mean={summary["mean_anchor_cc"]:.4f}  '
              f'winner mean={summary["mean_winner_cc"]:.4f}  '
              f'median gain={summary["median_cc_gain"]:+.4f}')

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Per-chirp cascade alignment.')
    ap.add_argument('--scene', default=None,
                    help='Single scene; omit and pass --scenes for multi-scene')
    ap.add_argument('--scenes', default=None,
                    help='Comma-separated scene list for multi-scene runs')
    ap.add_argument('--data-root', default='/home/adnan/Desktop/mm3DGS/data')
    ap.add_argument('--target-n', type=int, default=30000)
    ap.add_argument('--gpu', type=int, default=0,
                    help='CUDA_VISIBLE_DEVICES to set (caller may have set it already)')
    ap.add_argument('--mode', default='stage3', choices=['stage3', 'legacy'],
                    help='stage3 (recommended): anchored refinement with '
                         'optional GT. legacy: independent per-chirp pass1 '
                         '+ LOWESS smoothing (deprecated; kept for repro).')
    ap.add_argument('--anchor-source', default='hybrid',
                    choices=['lerp', 'gt', 'hybrid'],
                    help='Stage 3 anchor source. hybrid (recommended): '
                         'pass-2_F absolute + GT relative per-chirp motion.')
    ap.add_argument('--prior-weight', type=float,
                    default=DEFAULT_STAGE3_PRIOR_WEIGHT)
    ap.add_argument('--gate-margin', type=float,
                    default=DEFAULT_STAGE3_GATE_MARGIN_CC,
                    help='Refinement must beat anchor cc by ≥ this margin')
    ap.add_argument('--frames', default=None,
                    help='Comma-separated frame indices to process (default: all '
                         'frames with pass-2 configs)')
    ap.add_argument('--lowess-frac', type=float, default=0.1,
                    help='[legacy mode only] LOWESS bandwidth fraction')
    args = ap.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.scene and args.scenes:
        ap.error('specify --scene or --scenes, not both')
    if args.scene:
        scenes = [args.scene]
    elif args.scenes:
        scenes = [s.strip() for s in args.scenes.split(',') if s.strip()]
    else:
        ap.error('specify --scene or --scenes')

    frames_override = None
    if args.frames:
        frames_override = [int(x) for x in args.frames.split(',') if x.strip()]

    for sc in scenes:
        if args.mode == 'stage3':
            run_stage3_for_scene(
                sc, data_root=args.data_root,
                anchor_source=args.anchor_source,
                target_n=args.target_n,
                prior_weight=args.prior_weight,
                gate_margin=args.gate_margin,
                frames=frames_override,
                verbose=True)
        else:
            run_per_chirp_for_scene(
                sc, data_root=args.data_root,
                target_n=args.target_n,
                lowess_frac=args.lowess_frac,
                verbose=True)
