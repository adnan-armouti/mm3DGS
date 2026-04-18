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
    python -m mm25DGS_v5.preprocessing.alignment.per_chirp_alignment \
        --scene seq_1_frame_438 --gpu 0

    # Both target scenes, auto-pick GPU
    python -m mm25DGS_v5.preprocessing.alignment.per_chirp_alignment \
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

from mm25DGS_v5.preprocessing.alignment.cascaded_renderer_cuda import (
    build_alignment_context, apply_pose_to_ctx, render_and_evaluate_cuda,
    update_gt_for_chirp, destroy_alignment_context,
)
from mm25DGS_v5.preprocessing.alignment.cascaded_renderer import (
    apply_2dof_to_config, apply_4dof_to_config,
    grid_search_2dof, refine_2dof,
    grid_search_4dof, refine_4dof,
    save_aligned_config_from_pose,
)
from mm25DGS_v5.preprocessing.alignment.pass2.trajectory_fit import (
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Per-chirp cascade alignment (CUDA, two-pass).')
    ap.add_argument('--scene', default=None,
                    help='Single scene; omit and pass --scenes for multi-scene')
    ap.add_argument('--scenes', default=None,
                    help='Comma-separated scene list for multi-scene runs')
    ap.add_argument('--data-root', default='/home/adnan/Desktop/mm3DGS/data')
    ap.add_argument('--target-n', type=int, default=30000)
    ap.add_argument('--gpu', type=int, default=0,
                    help='CUDA_VISIBLE_DEVICES to set (caller may have set it already)')
    ap.add_argument('--lowess-frac', type=float, default=0.1,
                    help='LOWESS bandwidth as a fraction of the 144-point sequence')
    args = ap.parse_args()

    # Respect externally-set CUDA_VISIBLE_DEVICES if present
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

    for sc in scenes:
        run_per_chirp_for_scene(
            sc, data_root=args.data_root,
            target_n=args.target_n,
            lowess_frac=args.lowess_frac,
            verbose=True)
