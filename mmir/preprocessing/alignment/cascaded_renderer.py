#!/usr/bin/env python3
"""
Renderer-Based 2DOF Alignment (range + azimuth only, NO boresight rotation).

Key insight: The IMU-derived boresight is already physically correct (~3.4 deg
platform tilt). Only the range offset and azimuth offset need calibration.
Boresight rotations are NOT optimized — they corrupt the antenna pattern.

Strategy per scene:
1. Load unaligned base config (correct IMU boresight)
2. Grid search over delta_range_m x delta_azimuth_deg (zero rotation)
3. Fine-tune with bounded Nelder-Mead on 2DOF
4. Validate with 1500 hits
5. Compare vs existing _aligned_gpu; keep whichever is better
6. Save winning config as _aligned_2dof.json

Output: output/alignment_2dof/
"""

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import drjit as dr
dr.set_flag(dr.JitFlag.VCallOptimize, False)
dr.set_flag(dr.JitFlag.LoopOptimize, False)

import gc
import os
import sys
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize

from mmir.renderer import (
    FMCWRendererRef,
    RenderConfigRef,
)
from mmir.data.ra_utils import adc_to_ra_image, ra_polar_to_cartesian
from mmir.data.io_utils import compute_range_res_from_cfg

# submission/ root (alignment → preprocessing → mmir → submission)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TX_PATTERN = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy")
RX_PATTERN = os.path.join(PROJECT_ROOT, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy")

SCENES = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]


# ============================================================
# Geometry helpers (from renderer_alignment_refinement.py)
# ============================================================

def _norm(v, eps=1e-9):
    return v / (np.linalg.norm(v) + eps)


def _axis_angle_to_matrix(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def apply_2dof_to_config(base_config, delta_range_m, delta_azimuth_deg):
    """Apply 2DOF transform (range + azimuth) with ZERO rotation.

    This preserves the original IMU-derived boresight direction exactly.
    """
    tx_pos = np.array([t["pos_mm"] for t in base_config["tx_array"]], dtype=float) / 1000.0
    rx_pos = np.array([r["pos_mm"] for r in base_config["rx_array"]], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = len(tx_pos)

    board_center = np.mean(all_pos, axis=0)
    boresight = _norm(np.array(base_config["tx_array"][0]["boresight"], dtype=float))

    # No rotation — use original boresight and positions directly
    rel_pos = all_pos - board_center

    # Translation only
    range_offset = boresight * delta_range_m
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)
    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])
    azimuth_offset = x_board * delta_azimuth_m
    translation = range_offset + azimuth_offset

    new_board_center = board_center + translation
    new_all_pos = rel_pos + new_board_center

    new_tx_pos_mm = new_all_pos[:n_tx] * 1000.0
    new_rx_pos_mm = new_all_pos[n_tx:] * 1000.0

    return new_tx_pos_mm, new_rx_pos_mm, boresight


def update_renderer_antennas(renderer, tx_pos_mm, rx_pos_mm, boresight):
    """Update renderer antenna positions and boresights in-place."""
    scene_ctx = renderer.scene_ctx
    n_tx = tx_pos_mm.shape[0]
    n_rx = rx_pos_mm.shape[0]

    tx_pos_m = tx_pos_mm / 1000.0
    rx_pos_m = rx_pos_mm / 1000.0

    scene_ctx.tx_array.positions = mi.Point3f(
        mi.Float(tx_pos_m[:, 0].tolist()),
        mi.Float(tx_pos_m[:, 1].tolist()),
        mi.Float(tx_pos_m[:, 2].tolist()),
    )
    scene_ctx.tx_array.orientations = mi.Vector3f(
        mi.Float([boresight[0]] * n_tx),
        mi.Float([boresight[1]] * n_tx),
        mi.Float([boresight[2]] * n_tx),
    )
    scene_ctx.rx_array.positions = mi.Point3f(
        mi.Float(rx_pos_m[:, 0].tolist()),
        mi.Float(rx_pos_m[:, 1].tolist()),
        mi.Float(rx_pos_m[:, 2].tolist()),
    )
    scene_ctx.rx_array.orientations = mi.Vector3f(
        mi.Float([boresight[0]] * n_rx),
        mi.Float([boresight[1]] * n_rx),
        mi.Float([boresight[2]] * n_rx),
    )


# ============================================================
# RA processing helpers
# ============================================================

def minmax_normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-30:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def adc_to_ra_cart(adc_ri, range_res):
    adc_t = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_t).numpy()
    return ra_polar_to_cartesian(ra_polar, range_res)


def compute_cart_corr(ra1_cart, ra2_cart):
    n1 = minmax_normalize(ra1_cart)
    n2 = minmax_normalize(ra2_cart)
    return float(np.corrcoef(n1.ravel(), n2.ravel())[0, 1])


def load_gt_ra_cart(gt_adc_path, range_res):
    gt_raw = np.load(gt_adc_path)
    if gt_raw.ndim == 4:
        gt_chirp0 = gt_raw[0].transpose(1, 0, 2)
    elif gt_raw.ndim == 3:
        gt_chirp0 = gt_raw
    else:
        raise ValueError(f"Unexpected GT shape: {gt_raw.shape}")
    gt_ri = np.stack([gt_chirp0.real, gt_chirp0.imag], axis=-1).astype(np.float32)
    return adc_to_ra_cart(gt_ri, range_res)


# ============================================================
# Core: render, grid search, refine
# ============================================================

def render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42):
    """Render with current antenna config, return cart_corr."""
    result = renderer.render(seed=seed)
    adc_ri = result.adc_result.get_ri_array()
    ra_rend_cart = adc_to_ra_cart(adc_ri, range_res)
    cc = compute_cart_corr(ra_gt_cart, ra_rend_cart)
    del result, adc_ri
    dr.sync_thread()
    return cc


def grid_search_2dof(renderer, base_config, range_res, ra_gt_cart,
                     range_min=-2.0, range_max=2.0, range_step=0.25,
                     az_min=-10.0, az_max=10.0, az_step=1.0,
                     seed=42):
    """Grid search over range and azimuth with zero rotation."""
    range_grid = np.arange(range_min, range_max + range_step / 2, range_step)
    az_grid = np.arange(az_min, az_max + az_step / 2, az_step)
    total = len(range_grid) * len(az_grid)

    print(f"  Grid search: {len(range_grid)} range x {len(az_grid)} azimuth = {total} evaluations")

    best_cc = -999.0
    best_dr = 0.0
    best_da = 0.0
    all_results = []
    count = 0
    t0 = time.time()

    for dr_m in range_grid:
        for da_deg in az_grid:
            count += 1
            tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, dr_m, da_deg)
            update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
            cc = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=seed)
            all_results.append((float(dr_m), float(da_deg), float(cc)))

            if cc > best_cc:
                best_cc = cc
                best_dr = dr_m
                best_da = da_deg

            if count % 50 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate if rate > 0 else 0
                print(f"    [{count}/{total}] best CC={best_cc:.4f} "
                      f"(range={best_dr:.2f}m, az={best_da:.1f}°) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    print(f"  Grid search complete in {time.time()-t0:.1f}s")
    print(f"  Best: CC={best_cc:.4f} at range={best_dr:.3f}m, azimuth={best_da:.2f}°")

    return [best_dr, best_da], best_cc, all_results


def refine_2dof(renderer, base_config, initial_2dof, range_res, ra_gt_cart,
                max_evals=30, seed=42):
    """Fine-tune 2DOF params with bounded Nelder-Mead."""
    eval_count = [0]
    best_cc = [-999.0]
    best_params = [list(initial_2dof)]

    def objective(x):
        eval_count[0] += 1
        tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, x[0], x[1])
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=seed)
        if cc > best_cc[0]:
            best_cc[0] = cc
            best_params[0] = [float(x[0]), float(x[1])]
        print(f"    refine {eval_count[0]:3d}: CC={cc:.4f}  "
              f"range={x[0]:.3f}m az={x[1]:.2f}°")
        return -cc

    result = minimize(
        objective,
        x0=initial_2dof,
        method="Nelder-Mead",
        options={
            "maxiter": max_evals,
            "maxfev": max_evals + 5,
            "xatol": 0.02,
            "fatol": 0.001,
            "adaptive": True,
        },
    )

    return best_params[0], best_cc[0], eval_count[0]


def save_aligned_config(base_config, delta_range_m, delta_azimuth_deg, output_path):
    """Apply 2DOF params (zero rotation) and save aligned config."""
    tx_mm, rx_mm, boresight = apply_2dof_to_config(
        base_config, delta_range_m, delta_azimuth_deg
    )
    config = json.loads(json.dumps(base_config))
    for i, tx in enumerate(config["tx_array"]):
        tx["pos_mm"] = tx_mm[i].tolist()
        tx["boresight"] = boresight.tolist()
    for i, rx in enumerate(config["rx_array"]):
        rx["pos_mm"] = rx_mm[i].tolist()
        rx["boresight"] = boresight.tolist()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return output_path


# ============================================================
# Evaluate old 4DOF alignment for comparison
# ============================================================

def apply_4dof_to_config(base_config, delta_range_m, delta_azimuth_deg,
                         rotation_elev_deg, rotation_azim_deg):
    """Apply full 4DOF transform (for evaluating old alignments)."""
    tx_pos = np.array([t["pos_mm"] for t in base_config["tx_array"]], dtype=float) / 1000.0
    rx_pos = np.array([r["pos_mm"] for r in base_config["rx_array"]], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    n_tx = len(tx_pos)

    board_center = np.mean(all_pos, axis=0)
    base_boresight = _norm(np.array(base_config["tx_array"][0]["boresight"], dtype=float))

    rot_elev_rad = np.radians(rotation_elev_deg)
    R_elev = _axis_angle_to_matrix(np.array([0.0, 0.0, rot_elev_rad]))
    boresight_after_elev = _norm(R_elev @ base_boresight)

    z_world = np.array([0.0, 0.0, 1.0])
    x_board = _norm(np.cross(boresight_after_elev, z_world))
    if np.linalg.norm(x_board) < 1e-6:
        x_board = np.array([1.0, 0.0, 0.0])

    rot_azim_rad = np.radians(rotation_azim_deg)
    R_azim = _axis_angle_to_matrix(x_board * rot_azim_rad)
    R_combined = R_azim @ R_elev

    new_boresight = _norm(R_combined @ base_boresight)

    rel_pos = all_pos - board_center
    rel_pos_rotated = (R_combined @ rel_pos.T).T

    range_offset = new_boresight * delta_range_m
    reference_range_m = 10.0
    delta_azimuth_m = reference_range_m * np.radians(delta_azimuth_deg)
    x_new = _norm(np.cross(new_boresight, z_world))
    if np.linalg.norm(x_new) < 1e-6:
        x_new = np.array([1.0, 0.0, 0.0])
    azimuth_offset = x_new * delta_azimuth_m
    translation = range_offset + azimuth_offset

    new_board_center = board_center + translation
    new_all_pos = rel_pos_rotated + new_board_center

    return new_all_pos[:n_tx] * 1000.0, new_all_pos[n_tx:] * 1000.0, new_boresight


# ============================================================
# Main
# ============================================================

def main():
    output_dir = os.path.join(PROJECT_ROOT, "output", "alignment_2dof")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    print("=" * 80)
    print("RENDERER-BASED 2DOF ALIGNMENT")
    print("(range + azimuth only, zero boresight rotation)")
    print("=" * 80)

    for scene_name, frame_num in SCENES:
        print(f"\n{'='*70}")
        print(f"Scene: {scene_name}")
        print(f"{'='*70}")

        data_dir = os.path.join(PROJECT_ROOT, "data", scene_name)
        mesh_path = os.path.join(data_dir, "scene", "mesh.ply")
        gt_adc_path = os.path.join(data_dir, "radar", f"cascaded_frame_{frame_num}.npy")
        base_config_path = os.path.join(data_dir, "configs", f"cascaded_frame_{frame_num}.json")

        # Check files exist
        skip = False
        for p, label in [(mesh_path, "mesh"), (gt_adc_path, "GT ADC"), (base_config_path, "base config")]:
            if not os.path.isfile(p):
                print(f"  SKIP: {label} not found: {p}")
                skip = True
        if skip:
            continue

        # Load base config and range resolution
        with open(base_config_path) as f:
            base_config = json.load(f)
        range_res = compute_range_res_from_cfg(base_config_path)

        # Report base boresight (should be ~3.4 deg elevation from IMU)
        bs0 = _norm(np.array(base_config["tx_array"][0]["boresight"], dtype=float))
        bs_elev = np.degrees(np.arcsin(bs0[2]))
        print(f"  Base boresight: [{bs0[0]:.4f}, {bs0[1]:.4f}, {bs0[2]:.4f}]  "
              f"(elevation: {bs_elev:.1f}°)")
        print(f"  Range resolution: {range_res:.4f} m")

        # Load GT RA
        ra_gt_cart = load_gt_ra_cart(gt_adc_path, range_res)
        print(f"  GT RA Cartesian shape: {ra_gt_cart.shape}")

        # ---- Create renderer (800 hits for reliable grid search) ----
        render_config_fast = RenderConfigRef(
            n_hits_per_rx=800,
            use_shared_hits=True,
            bsdf_model="mmwave_jones",
            mmwave_polarization="vertical",
            hemisphere_sampling="cosine",
            double_sided=True,
            enable_image_method=False,
            enable_sms=False,
            use_vertex_normals=False,
            material_columns=6,
        )

        print(f"  Creating renderer (800 hits, mmwave_jones)...")
        t0 = time.time()
        renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_path,
            config_file=base_config_path,
            tx_pattern_file=TX_PATTERN,
            rx_pattern_file=RX_PATTERN,
            material_type="metal",
            render_config=render_config_fast,
            verbose=False,
        )
        print(f"  Renderer created in {time.time()-t0:.1f}s")

        # ---- Evaluate unaligned baseline ----
        print(f"\n  --- Baseline (unaligned) ---")
        tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, 0.0, 0.0)
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc_unaligned = render_and_evaluate(renderer, range_res, ra_gt_cart)
        print(f"  Unaligned CC (800 hits): {cc_unaligned:.4f}")

        # ---- Evaluate old 4DOF alignment for comparison ----
        cc_old_4dof = None
        old_params_4dof = None
        old_align_results_path = os.path.join(
            PROJECT_ROOT, "output", "lidar_radar_alignment_gpu_test",
            scene_name, "alignment_results.json"
        )
        if os.path.isfile(old_align_results_path):
            with open(old_align_results_path) as f:
                old_results = json.load(f)
            op = old_results.get("optimal_params", old_results.get("best_params", {}))
            old_params_4dof = [
                op.get("delta_range_m", 0),
                op.get("delta_azimuth_deg", 0),
                op.get("rotation_elev_deg", 0),
                op.get("rotation_azim_deg", 0),
            ]
            tx_mm, rx_mm, bs = apply_4dof_to_config(base_config, *old_params_4dof)
            update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
            cc_old_4dof = render_and_evaluate(renderer, range_res, ra_gt_cart)
            rot_total = np.sqrt(old_params_4dof[2]**2 + old_params_4dof[3]**2)
            print(f"  Old 4DOF CC (800 hits): {cc_old_4dof:.4f}  "
                  f"(rot={rot_total:.1f}° range={old_params_4dof[0]:.2f}m)")
        else:
            print(f"  Old 4DOF alignment: NOT FOUND")

        # ---- Phase 1: Grid search over 2DOF ----
        print(f"\n  --- Phase 1: 2DOF Grid Search ---")
        grid_params, grid_cc, grid_results = grid_search_2dof(
            renderer, base_config, range_res, ra_gt_cart, seed=42
        )

        # ---- Phase 2: Nelder-Mead refinement on 2DOF ----
        print(f"\n  --- Phase 2: 2DOF Nelder-Mead Refinement ---")
        refined_params, refined_cc, n_refine_evals = refine_2dof(
            renderer, base_config, grid_params, range_res, ra_gt_cart,
            max_evals=30, seed=42
        )
        print(f"  Refined: CC={refined_cc:.4f} at range={refined_params[0]:.3f}m, "
              f"azimuth={refined_params[1]:.2f}° ({n_refine_evals} evals)")

        # ---- Phase 3: Validate with 1500 hits ----
        print(f"\n  --- Phase 3: Validation (1500 hits) ---")
        renderer.config = RenderConfigRef(
            n_hits_per_rx=1500,
            use_shared_hits=True,
            bsdf_model="mmwave_jones",
            mmwave_polarization="vertical",
            hemisphere_sampling="cosine",
            double_sided=True,
            enable_image_method=False,
            enable_sms=False,
            use_vertex_normals=False,
            material_columns=6,
        )
        renderer.sampler.n_hits_per_rx = 1500

        # Validate unaligned at 1500 hits
        tx_mm, rx_mm, bs = apply_2dof_to_config(base_config, 0.0, 0.0)
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc_unaligned_val = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)

        # Validate 2DOF at 1500 hits
        tx_mm, rx_mm, bs = apply_2dof_to_config(
            base_config, refined_params[0], refined_params[1]
        )
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc_2dof_val = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)

        # Validate old 4DOF at 1500 hits
        cc_old_val = None
        if old_params_4dof is not None:
            tx_mm, rx_mm, bs = apply_4dof_to_config(base_config, *old_params_4dof)
            update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
            cc_old_val = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)

        print(f"  Unaligned  (1500 hits): {cc_unaligned_val:.4f}")
        if cc_old_val is not None:
            print(f"  Old 4DOF   (1500 hits): {cc_old_val:.4f}")
        print(f"  New 2DOF   (1500 hits): {cc_2dof_val:.4f}")

        # ---- Phase 4: Pick winner and save config ----
        # Compare 2DOF vs old 4DOF (at 1500 hits)
        use_2dof = True
        winner_label = "2dof"
        winner_cc = cc_2dof_val
        winner_range = refined_params[0]
        winner_az = refined_params[1]

        if cc_old_val is not None and cc_old_val > cc_2dof_val:
            # Old 4DOF was better — save it but still report comparison
            use_2dof = False
            winner_label = "old_4dof"
            winner_cc = cc_old_val
            print(f"\n  ** Old 4DOF wins ({cc_old_val:.4f} > {cc_2dof_val:.4f})")
            print(f"     Saving old 4DOF alignment as _aligned_2dof.json")
        else:
            delta_vs_old = cc_2dof_val - (cc_old_val if cc_old_val is not None else cc_unaligned_val)
            print(f"\n  ** 2DOF wins! ({cc_2dof_val:.4f}, delta={delta_vs_old:+.4f} vs old)")

        # Save config
        out_config_path = os.path.join(
            data_dir, "configs", f"cascaded_frame_{frame_num}_aligned_2dof.json"
        )
        if use_2dof:
            save_aligned_config(
                base_config, refined_params[0], refined_params[1], out_config_path
            )
        else:
            # Save old 4DOF alignment as _aligned_2dof for consistency
            tx_mm, rx_mm, bs = apply_4dof_to_config(base_config, *old_params_4dof)
            config = json.loads(json.dumps(base_config))
            for i, tx in enumerate(config["tx_array"]):
                tx["pos_mm"] = tx_mm[i].tolist()
                tx["boresight"] = bs.tolist()
            for i, rx in enumerate(config["rx_array"]):
                rx["pos_mm"] = rx_mm[i].tolist()
                rx["boresight"] = bs.tolist()
            os.makedirs(os.path.dirname(out_config_path), exist_ok=True)
            with open(out_config_path, "w") as f:
                json.dump(config, f, indent=2)
        print(f"  Saved: {out_config_path}")

        # Save per-scene results
        scene_out_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_out_dir, exist_ok=True)

        scene_results = {
            "base_boresight": bs0.tolist(),
            "base_boresight_elevation_deg": float(bs_elev),
            "cc_unaligned_800hits": float(cc_unaligned),
            "cc_old_4dof_800hits": float(cc_old_4dof) if cc_old_4dof is not None else None,
            "cc_grid_best_800hits": float(grid_cc),
            "cc_refined_800hits": float(refined_cc),
            "cc_unaligned_1500hits": float(cc_unaligned_val),
            "cc_old_4dof_1500hits": float(cc_old_val) if cc_old_val is not None else None,
            "cc_2dof_1500hits": float(cc_2dof_val),
            "winner": winner_label,
            "winner_cc": float(winner_cc),
            "params_2dof": {
                "delta_range_m": float(refined_params[0]),
                "delta_azimuth_deg": float(refined_params[1]),
            },
            "old_params_4dof": {
                "delta_range_m": float(old_params_4dof[0]),
                "delta_azimuth_deg": float(old_params_4dof[1]),
                "rotation_elev_deg": float(old_params_4dof[2]),
                "rotation_azim_deg": float(old_params_4dof[3]),
            } if old_params_4dof is not None else None,
            "n_grid_evals": len(grid_results),
            "n_refine_evals": n_refine_evals,
            "output_config": out_config_path,
        }
        all_results[scene_name] = scene_results

        with open(os.path.join(scene_out_dir, "results.json"), "w") as f:
            json.dump(scene_results, f, indent=2)

        # Save running summary
        with open(os.path.join(output_dir, "comparison_table.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Cleanup
        del renderer
        dr.sync_thread()
        gc.collect()
        if hasattr(dr, "flush_malloc_cache"):
            dr.flush_malloc_cache()

    # ============================================================
    # Print summary table
    # ============================================================
    print(f"\n\n{'='*90}")
    print("SUMMARY: 2DOF Alignment Results (all at 1500 hits)")
    print(f"{'='*90}")
    print(f"{'Scene':<20} {'Unaligned':>10} {'Old 4DOF':>10} {'New 2DOF':>10} "
          f"{'Winner':>10} {'Delta':>8}")
    print("-" * 90)

    for scene_name, _ in SCENES:
        r = all_results.get(scene_name, {})
        un = r.get("cc_unaligned_1500hits")
        old = r.get("cc_old_4dof_1500hits")
        new = r.get("cc_2dof_1500hits")
        winner = r.get("winner", "?")
        wcc = r.get("winner_cc")

        un_s = f"{un:.4f}" if un is not None else "N/A"
        old_s = f"{old:.4f}" if old is not None else "N/A"
        new_s = f"{new:.4f}" if new is not None else "N/A"

        # Delta vs old (or vs unaligned if no old)
        baseline = old if old is not None else (un if un is not None else 0)
        best = wcc if wcc is not None else 0
        delta = best - baseline if baseline else 0
        d_s = f"{delta:+.4f}"

        print(f"{scene_name:<20} {un_s:>10} {old_s:>10} {new_s:>10} "
              f"{winner:>10} {d_s:>8}")

    # Averages
    un_vals = [r.get("cc_unaligned_1500hits", 0) for r in all_results.values()]
    old_vals = [r.get("cc_old_4dof_1500hits", 0) for r in all_results.values() if r.get("cc_old_4dof_1500hits") is not None]
    new_vals = [r.get("cc_2dof_1500hits", 0) for r in all_results.values()]
    win_vals = [r.get("winner_cc", 0) for r in all_results.values()]

    print("-" * 90)
    print(f"{'Mean':<20} {np.mean(un_vals):>10.4f} {np.mean(old_vals):>10.4f} "
          f"{np.mean(new_vals):>10.4f} {'best':>10} {np.mean(win_vals)-np.mean(old_vals):>+8.4f}")

    print(f"{'='*90}")
    print(f"\nResults saved to: {output_dir}/comparison_table.json")
    print(f"Aligned configs: data/<scene>/configs/cascaded_frame_XXX_aligned_2dof.json")


if __name__ == "__main__":
    main()
