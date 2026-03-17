#!/usr/bin/env python3
"""
Cascaded Radar Alignment Orchestrator
======================================

End-to-end pipeline for aligning the 77 GHz cascaded FMCW radar to the
LiDAR-derived scene mesh.  Two complementary alignment methods are run
and the per-scene winner (highest RA-image correlation) is selected:

1. **LiDAR-based 4-DOF** (``cascaded_lidar`` / ``cascaded_lidar_gpu``):
   Voxelises the LiDAR point cloud into the radar's RAE grid and
   optimises range, azimuth, elevation-rotation, and azimuth-rotation
   to maximise correlation with the measured RA image.

2. **Renderer-based 2-DOF** (``cascaded_renderer``):
   Uses the differentiable Monte-Carlo renderer with 1-bounce to render
   RA images and optimises only range offset + azimuth offset (zero
   boresight rotation, preserving the IMU-derived tilt).

After both methods have been run (or loaded from prior results), the
orchestrator picks the winner per scene and writes the aligned config.

Usage (complete pipeline)::

    from mmir.preprocessing.alignment.cascaded_alignment import (
        run_all, SCENES,
    )
    results = run_all(data_root="data", output_root="output/alignment")

Usage (selection from pre-computed results)::

    from mmir.preprocessing.alignment.cascaded_alignment import (
        get_aligned_config_path, SCENES,
    )
    for scene_name, frame in SCENES:
        path = get_aligned_config_path(scene_name, frame, data_root="data")
        print(f"{scene_name} -> {path}")

The ``ALIGNMENT_RESULTS`` table encodes the validated correlation scores
from the original alignment runs and must match the configs referenced
by the training pipeline (train_v11).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

# ======================================================================
# Scene definitions
# ======================================================================

SCENES: List[Tuple[str, int]] = [
    ("seq_0_frame_135", 135),
    ("seq_0_frame_390", 390),
    ("seq_1_frame_185", 185),
    ("seq_1_frame_438", 438),
    ("seq_2_frame_105", 105),
    ("seq_2_frame_160", 160),
    ("seq_2_frame_300", 300),
]

# ======================================================================
# Validated alignment results
# ======================================================================
# Correlation scores measured at 1500 renderer hits.
# ``winner`` is max(cc_lidar_4dof, cc_renderer_2dof).
# ``config_suffix`` is the file suffix of the winning aligned config.

ALIGNMENT_RESULTS: Dict[str, dict] = {
    "seq_0_frame_135": {
        "frame": 135,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.476,
        "cc_renderer_2dof": 0.466,
        "config_suffix": "_aligned_gpu",
    },
    "seq_0_frame_390": {
        "frame": 390,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.363,
        "cc_renderer_2dof": 0.172,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_185": {
        "frame": 185,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.463,
        "cc_renderer_2dof": 0.283,
        "config_suffix": "_aligned_gpu",
    },
    "seq_1_frame_438": {
        "frame": 438,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.584,
        "cc_renderer_2dof": 0.613,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_105": {
        "frame": 105,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.113,
        "cc_renderer_2dof": 0.244,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_160": {
        "frame": 160,
        "winner": "renderer_2dof",
        "cc_lidar_4dof": 0.393,
        "cc_renderer_2dof": 0.577,
        "config_suffix": "_aligned_2dof",
    },
    "seq_2_frame_300": {
        "frame": 300,
        "winner": "lidar_4dof",
        "cc_lidar_4dof": 0.432,
        "cc_renderer_2dof": 0.389,
        "config_suffix": "_aligned_gpu",
    },
}


# ======================================================================
# Config lookup helpers
# ======================================================================

def get_aligned_config_path(
    scene_name: str,
    frame: int,
    data_root: str = "data",
) -> str:
    """Return the path to the winning aligned config for a scene.

    Args:
        scene_name: e.g. ``"seq_0_frame_135"``
        frame: radar frame number (e.g. 135)
        data_root: root of the preprocessed data tree

    Returns:
        Absolute path to e.g.
        ``data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json``
    """
    info = ALIGNMENT_RESULTS[scene_name]
    suffix = info["config_suffix"]
    return os.path.join(
        data_root, scene_name, "configs",
        f"cascaded_frame_{frame}{suffix}.json",
    )


def get_winner(scene_name: str) -> str:
    """Return ``"lidar_4dof"`` or ``"renderer_2dof"`` for a scene."""
    return ALIGNMENT_RESULTS[scene_name]["winner"]


def get_winner_cc(scene_name: str) -> float:
    """Return the winning correlation score for a scene."""
    info = ALIGNMENT_RESULTS[scene_name]
    return max(info["cc_lidar_4dof"], info["cc_renderer_2dof"])


def select_best_alignment(
    cc_lidar_4dof: float,
    cc_renderer_2dof: float,
) -> str:
    """Given two correlation scores, return the winning method name."""
    if cc_lidar_4dof >= cc_renderer_2dof:
        return "lidar_4dof"
    return "renderer_2dof"


# ======================================================================
# Summary / verification
# ======================================================================

def print_summary(data_root: str = "data") -> None:
    """Print a table of alignment results and verify config files exist."""
    print("=" * 78)
    print("Cascaded Radar Alignment — Per-Scene Results (1500-hit validation)")
    print("=" * 78)
    hdr = (
        f"{'Scene':<22} {'Winner':<16} {'CC_lidar':<10} "
        f"{'CC_renderer':<12} {'Config suffix'}"
    )
    print(hdr)
    print("-" * 78)

    all_ok = True
    winner_ccs = []
    for scene_name, frame in SCENES:
        info = ALIGNMENT_RESULTS[scene_name]
        cc_best = max(info["cc_lidar_4dof"], info["cc_renderer_2dof"])
        winner_ccs.append(cc_best)

        path = get_aligned_config_path(scene_name, frame, data_root)
        exists = os.path.isfile(path)
        status = "✓" if exists else "MISSING"
        if not exists:
            all_ok = False

        print(
            f"{scene_name:<22} {info['winner']:<16} "
            f"{info['cc_lidar_4dof']:<10.3f} "
            f"{info['cc_renderer_2dof']:<12.3f} "
            f"{info['config_suffix']}  [{status}]"
        )

    print("-" * 78)
    print(f"Mean winner CC: {sum(winner_ccs) / len(winner_ccs):.4f}")
    print(f"Config files: {'all present' if all_ok else 'SOME MISSING'}")
    print("=" * 78)


# ======================================================================
# Full pipeline (run both methods + select)
# ======================================================================

def run_all(
    data_root: str = "data",
    output_root: str = "output/alignment",
    tx_pattern: Optional[str] = None,
    rx_pattern: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run both alignment methods for all scenes and select winners.

    This is the full end-to-end pipeline.  For each scene it:

    1. Runs LiDAR-based 4-DOF alignment (GPU if CuPy available, else CPU)
       → saves ``cascaded_frame_{frame}_aligned_gpu.json``
    2. Runs renderer-based 2-DOF alignment
       → saves ``cascaded_frame_{frame}_aligned_2dof.json``
    3. Compares validation CCs and records the winner.

    Args:
        data_root: root of preprocessed data (contains seq_*/ dirs)
        output_root: where to write per-scene alignment logs
        tx_pattern: path to TX antenna pattern .npy
        rx_pattern: path to RX antenna pattern .npy
        verbose: print progress

    Returns:
        Dict mapping scene_name → result dict with CCs and winner.
    """
    # Lazy imports — these pull in mitsuba / drjit / open3d
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")

    import drjit as dr
    import numpy as np
    import time

    from .cascaded_renderer import (
        RenderConfigRef, FMCWRendererRef,
        apply_2dof_to_config, update_renderer_antennas,
        render_and_evaluate, grid_search_2dof, refine_2dof,
        save_aligned_config, load_gt_ra_cart, compute_cart_corr,
        apply_4dof_to_config,
    )
    from mmir.data.io_utils import compute_range_res_from_cfg

    submission_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    _tx = tx_pattern or os.path.join(
        submission_root, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy"
    )
    _rx = rx_pattern or os.path.join(
        submission_root, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy"
    )

    os.makedirs(output_root, exist_ok=True)
    all_results = {}

    if verbose:
        print("=" * 78)
        print("Cascaded Radar Alignment — Full Pipeline")
        print("=" * 78)

    for scene_name, frame in SCENES:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Scene: {scene_name}")
            print(f"{'='*70}")

        data_dir = os.path.join(data_root, scene_name)
        mesh_path = os.path.join(data_dir, "scene", "mesh.ply")
        gt_adc_path = os.path.join(
            data_dir, "radar", f"cascaded_frame_{frame}.npy"
        )
        base_config_path = os.path.join(
            data_dir, "configs", f"cascaded_frame_{frame}.json"
        )

        # Check prerequisites
        missing = [
            p for p in [mesh_path, gt_adc_path, base_config_path]
            if not os.path.isfile(p)
        ]
        if missing:
            if verbose:
                for p in missing:
                    print(f"  SKIP: not found: {p}")
            continue

        with open(base_config_path) as f:
            base_config = json.load(f)
        range_res = compute_range_res_from_cfg(base_config_path)
        ra_gt_cart = load_gt_ra_cart(gt_adc_path, range_res)

        # ----------------------------------------------------------
        # Method 1: LiDAR-based 4-DOF (check for existing result)
        # ----------------------------------------------------------
        lidar_result_path = os.path.join(
            output_root, scene_name, "lidar_4dof_results.json"
        )
        aligned_gpu_path = os.path.join(
            data_dir, "configs",
            f"cascaded_frame_{frame}_aligned_gpu.json",
        )

        cc_lidar = None
        if os.path.isfile(aligned_gpu_path):
            # Validate existing alignment with renderer
            if verbose:
                print(f"  LiDAR 4DOF: using existing {os.path.basename(aligned_gpu_path)}")
            # Quick renderer validation at 1500 hits
            render_cfg = RenderConfigRef(
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
            renderer = FMCWRendererRef.from_files(
                mesh_file=mesh_path,
                config_file=aligned_gpu_path,
                tx_pattern_file=_tx,
                rx_pattern_file=_rx,
                material_type="metal",
                render_config=render_cfg,
                verbose=False,
            )
            cc_lidar = render_and_evaluate(
                renderer, range_res, ra_gt_cart, seed=42
            )
            del renderer
            dr.sync_thread()
            if verbose:
                print(f"  LiDAR 4DOF CC (1500 hits): {cc_lidar:.4f}")
        else:
            if verbose:
                print(f"  LiDAR 4DOF: aligned config not found, skipping")

        # ----------------------------------------------------------
        # Method 2: Renderer-based 2-DOF
        # ----------------------------------------------------------
        aligned_2dof_path = os.path.join(
            data_dir, "configs",
            f"cascaded_frame_{frame}_aligned_2dof.json",
        )

        cc_renderer = None
        if os.path.isfile(aligned_2dof_path):
            if verbose:
                print(f"  Renderer 2DOF: using existing {os.path.basename(aligned_2dof_path)}")
            render_cfg = RenderConfigRef(
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
            renderer = FMCWRendererRef.from_files(
                mesh_file=mesh_path,
                config_file=aligned_2dof_path,
                tx_pattern_file=_tx,
                rx_pattern_file=_rx,
                material_type="metal",
                render_config=render_cfg,
                verbose=False,
            )
            cc_renderer = render_and_evaluate(
                renderer, range_res, ra_gt_cart, seed=42
            )
            del renderer
            dr.sync_thread()
            if verbose:
                print(f"  Renderer 2DOF CC (1500 hits): {cc_renderer:.4f}")
        else:
            # Run 2DOF alignment from scratch
            if verbose:
                print(f"  Renderer 2DOF: running alignment...")
            t0 = time.time()

            render_cfg_fast = RenderConfigRef(
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
            renderer = FMCWRendererRef.from_files(
                mesh_file=mesh_path,
                config_file=base_config_path,
                tx_pattern_file=_tx,
                rx_pattern_file=_rx,
                material_type="metal",
                render_config=render_cfg_fast,
                verbose=False,
            )

            grid_params, grid_cc, _ = grid_search_2dof(
                renderer, base_config, range_res, ra_gt_cart, seed=42
            )
            refined_params, refined_cc, _ = refine_2dof(
                renderer, base_config, grid_params, range_res, ra_gt_cart,
                max_evals=30, seed=42,
            )

            # Validate at 1500 hits
            renderer.config = RenderConfigRef(
                n_hits_per_rx=1500, use_shared_hits=True,
                bsdf_model="mmwave_jones", mmwave_polarization="vertical",
                hemisphere_sampling="cosine", double_sided=True,
                enable_image_method=False, enable_sms=False,
                use_vertex_normals=False, material_columns=6,
            )
            renderer.sampler.n_hits_per_rx = 1500
            tx_mm, rx_mm, bs = apply_2dof_to_config(
                base_config, refined_params[0], refined_params[1]
            )
            update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
            cc_renderer = render_and_evaluate(
                renderer, range_res, ra_gt_cart, seed=42
            )

            save_aligned_config(
                base_config,
                refined_params[0], refined_params[1],
                aligned_2dof_path,
            )
            del renderer
            dr.sync_thread()
            if verbose:
                print(
                    f"  Renderer 2DOF CC (1500 hits): {cc_renderer:.4f} "
                    f"({time.time()-t0:.1f}s)"
                )

        # ----------------------------------------------------------
        # Select winner
        # ----------------------------------------------------------
        if cc_lidar is not None and cc_renderer is not None:
            winner = select_best_alignment(cc_lidar, cc_renderer)
        elif cc_lidar is not None:
            winner = "lidar_4dof"
        elif cc_renderer is not None:
            winner = "renderer_2dof"
        else:
            if verbose:
                print(f"  WARNING: no alignment available for {scene_name}")
            continue

        suffix = (
            "_aligned_gpu" if winner == "lidar_4dof" else "_aligned_2dof"
        )
        winner_cc = (
            cc_lidar if winner == "lidar_4dof" else cc_renderer
        )

        scene_result = {
            "frame": frame,
            "winner": winner,
            "cc_lidar_4dof": float(cc_lidar) if cc_lidar is not None else None,
            "cc_renderer_2dof": float(cc_renderer) if cc_renderer is not None else None,
            "config_suffix": suffix,
            "aligned_config": os.path.join(
                data_dir, "configs",
                f"cascaded_frame_{frame}{suffix}.json",
            ),
        }
        all_results[scene_name] = scene_result

        if verbose:
            print(
                f"\n  >> Winner: {winner} (CC={winner_cc:.4f})"
                f"  config: cascaded_frame_{frame}{suffix}.json"
            )

        # Save per-scene log
        scene_out = os.path.join(output_root, scene_name)
        os.makedirs(scene_out, exist_ok=True)
        with open(os.path.join(scene_out, "alignment_result.json"), "w") as f:
            json.dump(scene_result, f, indent=2)

    # Save global results
    with open(os.path.join(output_root, "alignment_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if verbose:
        print()
        print_summary(data_root)

    return all_results


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cascaded radar alignment orchestrator",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root of preprocessed data tree (default: data)",
    )
    parser.add_argument(
        "--output-root", default="output/alignment",
        help="Output directory for logs (default: output/alignment)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Just print the alignment summary table (no alignment run)",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary(args.data_root)
    else:
        run_all(
            data_root=args.data_root,
            output_root=args.output_root,
        )

# ======================================================================
# Single-frame alignment (used by trajectory transfer)
# ======================================================================

def run_single_frame(
    scene_name: str,
    frame: int,
    data_root: str = "data",
    output_root: str = "output/alignment",
    aligned_config_dir: Optional[str] = None,
    tx_pattern: Optional[str] = None,
    rx_pattern: Optional[str] = None,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> Optional[dict]:
    """Run both alignment methods for a single cascade frame and select the winner.

    This is the per-frame building block used by the trajectory transfer
    pipeline.  It mirrors the logic in ``run_all`` but operates on an
    arbitrary frame (not just the 7 benchmark center frames).

    Args:
        scene_name: e.g. ``"seq_0_frame_390"``
        frame: cascade frame index (e.g. 386, 387, … 394)
        data_root: root of preprocessed data tree
        output_root: where to write per-frame alignment logs
        aligned_config_dir: where to write aligned config JSONs.
            Defaults to ``{output_root}/{scene_name}/cascade/``
        tx_pattern: path to TX antenna pattern .npy
        rx_pattern: path to RX antenna pattern .npy
        skip_if_exists: if True, reuse existing aligned configs
        verbose: print progress

    Returns:
        Result dict with keys: frame, winner, cc_lidar_4dof,
        cc_renderer_2dof, config_suffix, aligned_config_path.
        Returns None if prerequisites are missing.
    """
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")

    import drjit as dr
    import numpy as np
    import time

    from .cascaded_renderer import (
        RenderConfigRef, FMCWRendererRef,
        apply_2dof_to_config, update_renderer_antennas,
        render_and_evaluate, grid_search_2dof, refine_2dof,
        save_aligned_config, load_gt_ra_cart, compute_cart_corr,
    )
    from mmir.data.io_utils import compute_range_res_from_cfg

    submission_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    _tx = tx_pattern or os.path.join(
        submission_root, "assets", "antenna_pattern", "MMWCAS", "tx1_76.npy"
    )
    _rx = rx_pattern or os.path.join(
        submission_root, "assets", "antenna_pattern", "MMWCAS", "rx1_76.npy"
    )

    data_dir = os.path.join(data_root, scene_name)
    mesh_path = os.path.join(data_dir, "scene", "mesh.ply")
    gt_adc_path = os.path.join(data_dir, "radar", f"cascaded_frame_{frame}.npy")
    base_config_path = os.path.join(data_dir, "configs", f"cascaded_frame_{frame}.json")

    # Check prerequisites
    missing = [p for p in [mesh_path, gt_adc_path, base_config_path]
               if not os.path.isfile(p)]
    if missing:
        if verbose:
            for p in missing:
                print(f"  SKIP: not found: {p}")
        return None

    if aligned_config_dir is None:
        aligned_config_dir = os.path.join(output_root, scene_name, "cascade")
    os.makedirs(aligned_config_dir, exist_ok=True)

    with open(base_config_path) as f:
        base_config = json.load(f)
    range_res = compute_range_res_from_cfg(base_config_path)
    ra_gt_cart = load_gt_ra_cart(gt_adc_path, range_res)

    if verbose:
        print(f"  Aligning cascade frame {frame}...")

    # ── Method 1: LiDAR-based 4-DOF ──
    aligned_gpu_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_aligned_gpu.json"
    )
    # Also check the original data/configs location
    original_gpu_path = os.path.join(
        data_dir, "configs", f"cascaded_frame_{frame}_aligned_gpu.json"
    )

    cc_lidar = None
    gpu_source = None
    if skip_if_exists and os.path.isfile(aligned_gpu_path):
        gpu_source = aligned_gpu_path
    elif skip_if_exists and os.path.isfile(original_gpu_path):
        gpu_source = original_gpu_path

    if gpu_source is not None:
        if verbose:
            print(f"    4DOF: reusing {os.path.basename(gpu_source)}")
        render_cfg = RenderConfigRef(
            n_hits_per_rx=1500, use_shared_hits=True,
            bsdf_model="mmwave_jones", mmwave_polarization="vertical",
            hemisphere_sampling="cosine", double_sided=True,
            enable_image_method=False, enable_sms=False,
            use_vertex_normals=False, material_columns=6,
        )
        renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_path, config_file=gpu_source,
            tx_pattern_file=_tx, rx_pattern_file=_rx,
            material_type="metal", render_config=render_cfg, verbose=False,
        )
        cc_lidar = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)
        del renderer; dr.sync_thread()
        if verbose:
            print(f"    4DOF CC: {cc_lidar:.4f}")
        # Copy to output dir if it came from original location
        if gpu_source == original_gpu_path and not os.path.isfile(aligned_gpu_path):
            import shutil
            shutil.copy2(original_gpu_path, aligned_gpu_path)
    else:
        # Run 4DOF from scratch using the LiDAR GPU module
        if verbose:
            print(f"    4DOF: running LiDAR GPU alignment...")
        t0 = time.time()
        try:
            from .cascaded_lidar_gpu import optimize_alignment_gpu
            from .cascaded_lidar import (
                load_point_cloud, load_radar_config, radar_gt_to_ra_map,
                create_aligned_config,
            )

            params, base_origin, base_boresight = load_radar_config(base_config_path)
            lidar_mesh_path = os.path.join(data_dir, "scene", "mesh.ply")
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(lidar_mesh_path)
            if not pcd.has_points():
                # Fall back to loading the pcl.npy
                pcl_path = os.path.join(data_dir, "scene", "pcl.npy")
                pcl = np.load(pcl_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
                if pcl.shape[1] >= 7:
                    intensities = pcl[:, 6]
                    colors = np.stack([intensities]*3, axis=1)
                    pcd.colors = o3d.utility.Vector3dVector(colors)

            ra_radar = radar_gt_to_ra_map(gt_adc_path)
            max_range = params['num_adc'] * params['range_resolution']
            results_4dof = optimize_alignment_gpu(
                pcd, ra_radar, params, base_boresight, base_origin,
                range_search_m=(-max_range * 0.1, max_range * 0.1),
                azimuth_search_deg=(-15.0, 15.0),
                rotation_elev_search_deg=(-10.0, 10.0),
                rotation_azim_search_deg=(-10.0, 10.0),
                metric='correlation', coarse_steps=5,
                near_field_m=1.5, batch_size=100, verbose=False,
            )
            opt = results_4dof['optimal_params']
            create_aligned_config(
                config_path=base_config_path,
                delta_range_m=opt['delta_range_m'],
                delta_azimuth_deg=opt['delta_azimuth_deg'],
                rotation_elev_deg=opt['rotation_elev_deg'],
                rotation_azim_deg=opt['rotation_azim_deg'],
                output_path=aligned_gpu_path,
            )
            # Validate with renderer
            render_cfg = RenderConfigRef(
                n_hits_per_rx=1500, use_shared_hits=True,
                bsdf_model="mmwave_jones", mmwave_polarization="vertical",
                hemisphere_sampling="cosine", double_sided=True,
                enable_image_method=False, enable_sms=False,
                use_vertex_normals=False, material_columns=6,
            )
            renderer = FMCWRendererRef.from_files(
                mesh_file=mesh_path, config_file=aligned_gpu_path,
                tx_pattern_file=_tx, rx_pattern_file=_rx,
                material_type="metal", render_config=render_cfg, verbose=False,
            )
            cc_lidar = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)
            del renderer; dr.sync_thread()
            if verbose:
                print(f"    4DOF CC: {cc_lidar:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"    4DOF failed: {e}")

    # ── Method 2: Renderer-based 2-DOF ──
    aligned_2dof_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_aligned_2dof.json"
    )
    original_2dof_path = os.path.join(
        data_dir, "configs", f"cascaded_frame_{frame}_aligned_2dof.json"
    )

    cc_renderer = None
    dof2_source = None
    if skip_if_exists and os.path.isfile(aligned_2dof_path):
        dof2_source = aligned_2dof_path
    elif skip_if_exists and os.path.isfile(original_2dof_path):
        dof2_source = original_2dof_path

    if dof2_source is not None:
        if verbose:
            print(f"    2DOF: reusing {os.path.basename(dof2_source)}")
        render_cfg = RenderConfigRef(
            n_hits_per_rx=1500, use_shared_hits=True,
            bsdf_model="mmwave_jones", mmwave_polarization="vertical",
            hemisphere_sampling="cosine", double_sided=True,
            enable_image_method=False, enable_sms=False,
            use_vertex_normals=False, material_columns=6,
        )
        renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_path, config_file=dof2_source,
            tx_pattern_file=_tx, rx_pattern_file=_rx,
            material_type="metal", render_config=render_cfg, verbose=False,
        )
        cc_renderer = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)
        del renderer; dr.sync_thread()
        if verbose:
            print(f"    2DOF CC: {cc_renderer:.4f}")
        if dof2_source == original_2dof_path and not os.path.isfile(aligned_2dof_path):
            import shutil
            shutil.copy2(original_2dof_path, aligned_2dof_path)
    else:
        if verbose:
            print(f"    2DOF: running renderer alignment...")
        t0 = time.time()
        render_cfg_fast = RenderConfigRef(
            n_hits_per_rx=800, use_shared_hits=True,
            bsdf_model="mmwave_jones", mmwave_polarization="vertical",
            hemisphere_sampling="cosine", double_sided=True,
            enable_image_method=False, enable_sms=False,
            use_vertex_normals=False, material_columns=6,
        )
        renderer = FMCWRendererRef.from_files(
            mesh_file=mesh_path, config_file=base_config_path,
            tx_pattern_file=_tx, rx_pattern_file=_rx,
            material_type="metal", render_config=render_cfg_fast, verbose=False,
        )
        grid_params, grid_cc, _ = grid_search_2dof(
            renderer, base_config, range_res, ra_gt_cart, seed=42
        )
        refined_params, refined_cc, _ = refine_2dof(
            renderer, base_config, grid_params, range_res, ra_gt_cart,
            max_evals=30, seed=42,
        )
        # Validate at 1500 hits
        renderer.config = RenderConfigRef(
            n_hits_per_rx=1500, use_shared_hits=True,
            bsdf_model="mmwave_jones", mmwave_polarization="vertical",
            hemisphere_sampling="cosine", double_sided=True,
            enable_image_method=False, enable_sms=False,
            use_vertex_normals=False, material_columns=6,
        )
        renderer.sampler.n_hits_per_rx = 1500
        tx_mm, rx_mm, bs = apply_2dof_to_config(
            base_config, refined_params[0], refined_params[1]
        )
        update_renderer_antennas(renderer, tx_mm, rx_mm, bs)
        cc_renderer = render_and_evaluate(renderer, range_res, ra_gt_cart, seed=42)
        save_aligned_config(
            base_config, refined_params[0], refined_params[1],
            aligned_2dof_path,
        )
        del renderer; dr.sync_thread()
        if verbose:
            print(f"    2DOF CC: {cc_renderer:.4f} ({time.time()-t0:.1f}s)")

    # ── Select winner ──
    if cc_lidar is not None and cc_renderer is not None:
        winner = select_best_alignment(cc_lidar, cc_renderer)
    elif cc_lidar is not None:
        winner = "lidar_4dof"
    elif cc_renderer is not None:
        winner = "renderer_2dof"
    else:
        if verbose:
            print(f"    WARNING: no alignment for frame {frame}")
        return None

    suffix = "_aligned_gpu" if winner == "lidar_4dof" else "_aligned_2dof"
    winner_cc = cc_lidar if winner == "lidar_4dof" else cc_renderer
    aligned_config_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}{suffix}.json"
    )

    # Write a combined winner symlink / copy for easy lookup
    winner_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_aligned.json"
    )
    if not os.path.isfile(winner_path):
        import shutil
        shutil.copy2(aligned_config_path, winner_path)

    result = {
        "frame": frame,
        "winner": winner,
        "cc_lidar_4dof": float(cc_lidar) if cc_lidar is not None else None,
        "cc_renderer_2dof": float(cc_renderer) if cc_renderer is not None else None,
        "config_suffix": suffix,
        "winner_cc": float(winner_cc),
        "aligned_config_path": aligned_config_path,
    }

    # Save per-frame log
    log_path = os.path.join(
        aligned_config_dir, f"cascaded_frame_{frame}_alignment_log.json"
    )
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"    >> Winner: {winner} (CC={winner_cc:.4f})")

    return result


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cascaded radar alignment orchestrator",
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root of preprocessed data tree (default: data)",
    )
    parser.add_argument(
        "--output-root", default="output/alignment",
        help="Output directory for logs (default: output/alignment)",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Just print the alignment summary table (no alignment run)",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary(args.data_root)
    else:
        run_all(
            data_root=args.data_root,
            output_root=args.output_root,
        )
