"""Evaluation #4: 3D Dense Reconstruction — Spinning Radar.

Extends Evaluation #3 to multiple rendered frames by spinning the radar along
the azimuth direction (like a LiDAR), aggregating scans into a dense 3D
reconstruction.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_evaluator import BaseEvaluator, aggregate_metrics
from .scene_registry import SceneInfo


class ReconstructionEvaluator(BaseEvaluator):
    """Evaluate 3D reconstruction from spinning dense virtual array."""

    def __init__(
        self,
        scene: SceneInfo,
        output_root: str,
        azimuth_range: Tuple[float, float] = (-21.0, 69.0),
        azimuth_step: float = 1.0,
        batch_size: int = 5000,
        near_field_m: float = 1.5,
        radar_percentile: float = 98.0,
        aggregate_percentile: float = 90.0,
        voxel_size_m: float = 0.1,
        verbose: bool = True,
    ):
        super().__init__(scene, output_root, verbose)
        self.azimuth_range = azimuth_range
        self.azimuth_step = azimuth_step
        self.batch_size = batch_size
        self.near_field_m = near_field_m
        self.radar_percentile = radar_percentile
        self.aggregate_percentile = aggregate_percentile
        self.voxel_size_m = voxel_size_m

    @property
    def eval_name(self) -> str:
        return "3d_reconstruction"

    def _get_azimuth_angles(self) -> List[float]:
        """Generate list of azimuth angles for spinning."""
        angles = []
        az = self.azimuth_range[0]
        while az <= self.azimuth_range[1]:
            angles.append(az)
            az += self.azimuth_step
        return angles

    def render(self):
        """Phase 1: Multi-frame spinning render.

        Creates the renderer ONCE then swaps antenna configs for each frame,
        avoiding the ~3s per-frame overhead of mesh/material/pattern loading.
        """
        if not self.scene.dense_config:
            self.log("No dense config available for this scene")
            return

        if not self.scene.our_training_dir:
            self.log("No training output directory available")
            return

        from mmir.evaluation.utils.gen_configs import generate_azimuth_rotated_config
        from .renderer_wrapper import RendererWrapper, RenderConfigRef

        angles = self._get_azimuth_angles()
        frames_dir = os.path.join(self.output_dir, "frames")
        configs_dir = os.path.join(self.output_dir, "frames", "configs")
        os.makedirs(configs_dir, exist_ok=True)

        # Check which frames actually need rendering
        frames_to_render = []
        for frame_idx, angle_deg in enumerate(angles):
            adc_path = os.path.join(frames_dir, f"adc_az_{angle_deg:.1f}.npy")
            if os.path.isfile(adc_path):
                self.log(f"  Frame {frame_idx+1}/{len(angles)} (az={angle_deg:.1f}°): already exists, skipping")
            else:
                frames_to_render.append((frame_idx, angle_deg))

        if not frames_to_render:
            self.log(f"  All {len(angles)} frames already rendered")
            return

        self.log(f"Spinning render: {len(frames_to_render)} frames to render "
                 f"(of {len(angles)} total, {self.azimuth_range[0]}° to {self.azimuth_range[1]}°)")

        # Load base config
        with open(self.scene.dense_config) as f:
            base_config = json.load(f)

        # Dense array config optimized for 100×100 virtual arrays
        render_config = RenderConfigRef(
            n_hits_per_rx=200,
            use_shared_hits=True,
            bsdf_model="mmwave_jones",
            mmwave_polarization="vertical",
            hemisphere_sampling="cosine",
            double_sided=True,
            enable_sms=False,
            enable_image_method=False,
            use_vertex_normals=True,
            material_columns=6,
        )

        # Create renderer ONCE with base config, load materials once
        import time as _time
        t_init = _time.time()
        wrapper = RendererWrapper(
            mesh_file=self.scene.mesh_file,
            config_file=self.scene.dense_config,
            render_config=render_config,
            verbose=False,
        )
        wrapper.load_materials(self.scene.our_training_dir)
        self.log(f"  Renderer initialized in {_time.time() - t_init:.1f}s (one-time)")

        t_render_start = _time.time()
        for render_idx, (frame_idx, angle_deg) in enumerate(frames_to_render):
            adc_path = os.path.join(frames_dir, f"adc_az_{angle_deg:.1f}.npy")

            # Generate rotated antenna config and swap positions (no renderer recreation)
            rotated_config = generate_azimuth_rotated_config(base_config, angle_deg)
            wrapper.update_antenna_config(rotated_config)

            # Save config for reproducibility
            config_path = os.path.join(configs_dir, f"dense_az_{angle_deg:.1f}.json")
            with open(config_path, "w") as f:
                json.dump(rotated_config, f, indent=2)

            t_frame = _time.time()
            adc = wrapper.render_batched(batch_size=100, seed=42 + frame_idx)
            np.save(adc_path, adc)
            elapsed = _time.time() - t_frame

            self.log(f"  Frame {render_idx+1}/{len(frames_to_render)} "
                     f"(az={angle_deg:.1f}°) rendered in {elapsed:.1f}s")

        wrapper.cleanup()
        total = _time.time() - t_render_start
        self.log(f"  All {len(frames_to_render)} frames rendered in {total:.1f}s "
                 f"({total/len(frames_to_render):.1f}s/frame avg)")

    # FFT oversampling factor for azimuth/elevation.  The virtual array is
    # 100×100 elements; the default 128 bins give only 22% zero-padding which
    # produces severe spectral leakage.  192 bins → 48% padding — good balance
    # between sidelobe suppression and CFAR speed (43% fewer voxels than 256).
    _FFT_OVERSAMPLE = 192

    @staticmethod
    def _blackman(n: int, device: torch.device) -> torch.Tensor:
        """Blackman window on *device* (first sidelobe ≈ −58 dB vs Hann's −31.5 dB)."""
        w = torch.blackman_window(n, periodic=False, device=device, dtype=torch.float32)
        return w

    @classmethod
    def _compute_rae_cube_gpu(
        cls, adc_path: str, params: dict, device: torch.device,
    ) -> Tuple[torch.Tensor, dict]:
        """ADC → RAE magnitude cube, kept on GPU for CFAR.

        Uses Blackman window (−58 dB sidelobes) instead of Hann (−31.5 dB)
        and 192-bin FFTs (48% zero-padding) instead of 128 (22%).

        Returns:
            (mag, effective_params) where mag is (Az, El, R) float32 on
            *device* and effective_params has the actual FFT bin counts.
        """
        from mmir.evaluation.utils.consolidate_views import (
            get_hann, to_tensor, txrx_to_vx_chirps_dense_gpu_batched,
        )

        n_az_bins = cls._FFT_OVERSAMPLE
        n_el_bins = cls._FFT_OVERSAMPLE

        adc = np.load(adc_path)
        adc_c = adc[:, :, :, 0] + 1j * adc[:, :, :, 1]
        adc_c = np.expand_dims(adc_c, axis=0)
        adc_c = np.transpose(adc_c, (0, 2, 1, 3))
        adc_t = to_tensor(adc_c, device=device, dtype=torch.complex64)
        vx = txrx_to_vx_chirps_dense_gpu_batched(adc_t, num_ant=int(params['num_ant']))

        # Range FFT — keep Hann here (range sidelobes are less problematic
        # because range resolution is set by bandwidth, not array size)
        h_range = get_hann(int(params['num_adc']), vx.device)
        vx = vx * h_range.view(1, 1, 1, -1)
        vx = torch.fft.fft(vx, n=int(params['num_adc']), dim=-1)
        vol = vx

        # Azimuth FFT (dim 1) — Blackman window + 256-bin FFT
        h_az = cls._blackman(vol.shape[1], vol.device)
        vol = vol * h_az.view(1, -1, 1, 1)
        vol = torch.fft.ifftshift(vol, dim=1)
        vol = torch.fft.fft(vol, n=n_az_bins, dim=1)
        vol = vol[:, 1:, :, :]
        vol = torch.fft.fftshift(vol, dim=1)

        # Elevation FFT (dim 2) — Blackman window + 256-bin FFT
        h_el = cls._blackman(vol.shape[2], vol.device)
        vol = vol * h_el.view(1, 1, -1, 1)
        vol = torch.fft.ifftshift(vol, dim=2)
        vol = torch.fft.fft(vol, n=n_el_bins, dim=2)
        vol = vol[:, :, 1:, :]
        vol = torch.fft.fftshift(vol, dim=2)

        # Return effective params so angle grids use the actual FFT sizes
        eff_params = dict(params)
        eff_params['num_az_bins'] = n_az_bins
        eff_params['num_el_bins'] = n_el_bins

        return torch.abs(vol[0]).to(torch.float32), eff_params  # (Az, El, R)

    def _process_single_view(
        self,
        adc_path: str,
        config_path: str,
        device: torch.device,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process one rendered view: ADC → RAE → CFAR detection → PCA align.

        Uses 3D CA-CFAR instead of simple percentile thresholding to suppress
        sidelobes while preserving weak true detections.

        Returns:
            (pts_world, intensity) as (N,3) and (N,) arrays, or (None, None).
        """
        from mmir.evaluation.utils.single_view_viz import (
            extract_board_frame_from_config,
            load_config_positions_and_boresight,
            load_runtime_config,
        )
        from mmir.evaluation.utils.consolidate_views import (
            simple_3d_cfar_gpu,
        )

        # 1. Load params from this view's config
        params = load_runtime_config(config_path, Path(self.output_dir))

        # 2. ADC → RAE cube (Blackman window + 192-bin FFT, kept on GPU)
        mag_gpu, eff_params = self._compute_rae_cube_gpu(adc_path, params, device)

        # 3. Extract PCA board frame from this view's rotated config
        v2_azimuth, y_range, v1_elevation = extract_board_frame_from_config(config_path)
        R_align = np.column_stack([v2_azimuth, y_range, v1_elevation])

        # 4. Get config center for translation
        _, _, cfg_center, _ = load_config_positions_and_boresight(config_path)

        # 5. Azimuth flip check (same as Eval #3)
        if v2_azimuth[0] >= 0:
            mag_gpu = torch.flip(mag_gpu, dims=[0])

        # 6. CFAR detection — adaptive local threshold suppresses sidelobes
        #    Training/guard windows scaled for 192-bin FFTs (1.5× the 128-bin
        #    base values) so the training region covers the same angular extent.
        cfar_mask = simple_3d_cfar_gpu(
            mag_gpu,
            train=(6, 6, 16),
            guard=(3, 3, 4),
            threshold_factor=4.0,
        )

        # 7. Extract detected points → spherical → Cartesian
        az_idx, el_idx, r_idx = torch.nonzero(cfar_mask, as_tuple=True)
        if az_idx.numel() == 0:
            return None, None

        # Use effective params (256-bin FFTs) for angle grids
        num_az_full = int(eff_params['num_az_bins'])
        num_el_full = int(eff_params['num_el_bins'])
        range_res = float(eff_params['range_resolution'])

        # Angle grids (same convention as radar_points_from_rae)
        t_az = torch.arange(-num_az_full // 2 + 1, num_az_full // 2,
                            device=device, dtype=torch.float32) * (2.0 / num_az_full)
        t_el = torch.arange(-num_el_full // 2 + 1, num_el_full // 2,
                            device=device, dtype=torch.float32) * (2.0 / num_el_full)
        az_angles = torch.arcsin(torch.clamp(t_az, -1.0 + 1e-6, 1.0 - 1e-6))
        el_angles = torch.arcsin(torch.clamp(t_el, -1.0 + 1e-6, 1.0 - 1e-6))

        az_vals = az_angles[az_idx]
        el_vals = el_angles[el_idx]
        r_vals = r_idx.to(torch.float32) * range_res

        # Near-field removal
        keep = r_vals >= self.near_field_m
        if not torch.any(keep):
            return None, None
        az_vals = az_vals[keep]
        el_vals = el_vals[keep]
        r_vals = r_vals[keep]
        intensity = mag_gpu[az_idx[keep], el_idx[keep], r_idx[keep]]

        # Spherical → Cartesian (radar local frame)
        x = r_vals * torch.cos(el_vals) * torch.sin(az_vals)
        y = r_vals * torch.cos(el_vals) * torch.cos(az_vals)
        z = r_vals * torch.sin(el_vals)

        pts_radar = torch.stack([x, y, z], dim=1).cpu().numpy().astype(np.float32)
        intensity_np = intensity.cpu().numpy().astype(np.float32)

        if pts_radar.shape[0] == 0:
            return None, None

        # 8. Rotate to world frame + translate to config center
        pts_world = (pts_radar @ R_align.T).astype(np.float32)
        if cfg_center is not None:
            pts_world = pts_world + cfg_center.reshape(1, 3).astype(np.float32)

        return pts_world, intensity_np

    @staticmethod
    def _apply_global_filtering(
        accumulator,
        device: torch.device,
        count_min: int = 2,
        min_views: int = 6,
        gaussian_sigma: Tuple[float, float, float] = (0.4, 0.8, 1.0),
        min_component_size: int = 40,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply global spatial filtering to accumulated voxels.

        1. Multi-view consistency thresholds (count >= 2, views >= 4)
        2. Gaussian smoothing on counts
        3. Connected-component pruning (min_size=40)

        Azimuth NMS is removed — it was applied along a world-space axis (x)
        which has no physical relationship to radar azimuth after multi-view
        aggregation into (x,y,z) voxels.

        Returns:
            (radar_pts, max_intensity, view_diversity) as (N,3), (N,), (N,)
            arrays, or (None, None, None).
        """
        from mmir.evaluation.utils.consolidate_views import (
            connected_components_3d_gpu,
            gaussian_smooth_gpu,
            to_numpy,
            to_tensor,
        )

        dense_result = accumulator.to_dense()
        if dense_result is None:
            return None, None, None

        counts, max_int, view_div, origin_idx = dense_result

        # Step 1: Multi-view consistency
        keep = (counts >= count_min) & (view_div >= min_views)

        # Step 2: Gaussian smoothing on counts (spatial coherence filter)
        counts_t = to_tensor(counts, device, torch.float32)
        counts_sm = gaussian_smooth_gpu(counts_t, gaussian_sigma)
        counts_sm_np = to_numpy(counts_sm)
        keep &= (counts_sm_np >= float(count_min))

        # Step 3: Connected-component pruning
        keep_t = to_tensor(keep, device, torch.bool)
        keep = to_numpy(connected_components_3d_gpu(keep_t, min_size=min_component_size))

        n_kept = int(np.sum(keep))
        if n_kept == 0:
            return None, None, None

        x_agg, y_agg, z_agg = accumulator.kept_voxel_centers(keep, origin_idx)
        ii, jj, kk = np.nonzero(keep)
        val_agg = max_int[ii, jj, kk]
        vd_agg = view_div[ii, jj, kk].astype(np.float32)
        radar_pts = np.column_stack([x_agg, y_agg, z_agg])

        return radar_pts, val_agg, vd_agg

    def run(self, skip_convergence: bool = False) -> dict:
        """Phase 2: Aggregate per-frame point clouds and evaluate against LiDAR.

        Per-view processing:
          ADC → RAE → CFAR detection → PCA rotation → translation

        Multi-view aggregation uses VoxelAccumulatorGPU + global filtering:
          counts>=2, views>=4, Gaussian smooth, CC pruning (min_size=40)
        """
        metrics = {"scene": self.scene.name}

        frames_dir = os.path.join(self.output_dir, "frames")
        if not os.path.isdir(frames_dir):
            self.log("No frames directory found, triggering render...")
            self.render()
            if not os.path.isdir(frames_dir):
                self.save_metrics(metrics)
                return metrics

        angles = self._get_azimuth_angles()
        configs_dir = os.path.join(frames_dir, "configs")

        # Collect available frame files with their configs
        adc_paths = []
        config_paths = []
        frame_angles = []
        for angle_deg in angles:
            adc_path = os.path.join(frames_dir, f"adc_az_{angle_deg:.1f}.npy")
            cfg_path = os.path.join(configs_dir, f"dense_az_{angle_deg:.1f}.json")
            if os.path.isfile(adc_path) and os.path.isfile(cfg_path):
                adc_paths.append(adc_path)
                config_paths.append(cfg_path)
                frame_angles.append(angle_deg)

        if not adc_paths:
            self.log("No rendered frames found")
            self.save_metrics(metrics)
            return metrics

        self.log(f"Aggregating {len(adc_paths)} frames "
                 f"(CFAR detection, "
                 f"post-threshold={self.radar_percentile}%, "
                 f"voxel={self.voxel_size_m}m)...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from mmir.evaluation.utils.consolidate_views import VoxelAccumulatorGPU

        accumulator = VoxelAccumulatorGPU(self.voxel_size_m, device)

        # Per-view detection cache: save (pts_world, intensity) per view
        # so re-runs with different radar_percentile/min_views skip FFT+CFAR.
        cache_dir = os.path.join(self.output_dir, "frames", "detections")
        os.makedirs(cache_dir, exist_ok=True)

        per_view_data = []  # list of (pts_world, intensity) per view
        n_cached, n_computed = 0, 0
        import time as _time
        t_views_start = _time.time()

        for view_idx, (adc_path, cfg_path, angle_deg) in enumerate(
            zip(adc_paths, config_paths, frame_angles)
        ):
            cache_pts = os.path.join(cache_dir, f"det_az_{angle_deg:.1f}_pts.npy")
            cache_int = os.path.join(cache_dir, f"det_az_{angle_deg:.1f}_int.npy")

            if os.path.isfile(cache_pts):
                # Load cached detections (fast path — uncompressed npy)
                pts_world = np.load(cache_pts)
                intensity = np.load(cache_int)
                if len(pts_world) == 0:
                    pts_world, intensity = None, None
                n_cached += 1
            else:
                # Compute FFT + CFAR (slow path), then cache
                pts_world, intensity = self._process_single_view(adc_path, cfg_path, device)
                if pts_world is not None and len(pts_world) > 0:
                    np.save(cache_pts, pts_world)
                    np.save(cache_int, intensity)
                else:
                    np.save(cache_pts, np.empty((0, 3), dtype=np.float32))
                    np.save(cache_int, np.empty(0, dtype=np.float32))
                n_computed += 1

            if pts_world is not None and len(pts_world) > 0:
                accumulator.add_points_batch(
                    pts_world[:, 0], pts_world[:, 1], pts_world[:, 2],
                    intensity, view_idx,
                )
                per_view_data.append((pts_world, intensity))
            else:
                per_view_data.append((None, None))

            if self.verbose and (view_idx + 1) % 10 == 0:
                self.log(f"  Processed {view_idx + 1}/{len(adc_paths)} views")

        t_views = _time.time() - t_views_start
        self.log(f"  All {len(adc_paths)} views processed in {t_views:.1f}s "
                 f"({n_cached} cached, {n_computed} computed)")

        # Apply global filtering (multi-view consistency + spatial smoothing)
        result = self._apply_global_filtering(accumulator, device)
        if result[0] is None:
            self.log("No points survived global filtering")
            self.save_metrics(metrics)
            return metrics

        radar_pts, val_agg, vd_agg = result
        self.log(f"  {len(radar_pts)} voxel points after global filtering")

        # Post-aggregation threshold using composite score:
        # view_diversity * max_intensity rewards consistent multi-view detections
        # over single bright sidelobe spikes from one view.
        if self.radar_percentile is not None and len(val_agg) > 0:
            composite_score = vd_agg * val_agg
            thr = np.percentile(composite_score, self.radar_percentile)
            keep_int = composite_score >= thr
            n_before = len(radar_pts)
            radar_pts = radar_pts[keep_int]
            val_agg = val_agg[keep_int]
            self.log(f"  {n_before} → {len(radar_pts)} after composite threshold "
                     f"(top {100 - self.radar_percentile:.1f}%, "
                     f"score=view_div*max_int, thr={thr:.2f})")

        # Save aggregated point cloud
        np.save(os.path.join(self.output_dir, "aggregated_pcl.npy"), radar_pts)
        np.save(os.path.join(self.output_dir, "aggregated_intensity.npy"), val_agg)

        # Compare to aggregated LiDAR, clipped to mesh bounding box
        lidar_pts = None
        if self.scene.lidar_pcl and os.path.isfile(self.scene.lidar_pcl):
            lidar_data = np.load(self.scene.lidar_pcl)
            lidar_pts = lidar_data[:, :3] if lidar_data.ndim == 2 and lidar_data.shape[1] >= 3 else lidar_data

            # Clip to mesh bbox — only evaluate within the rendered scene geometry
            if self.scene.mesh_file and os.path.isfile(self.scene.mesh_file):
                n_before = len(lidar_pts)
                lidar_pts = self._clip_to_mesh_bbox(lidar_pts, self.scene.mesh_file)
                self.log(f"  Clipped aggregated LiDAR to mesh bbox: {n_before} → {len(lidar_pts)} points")

            from .utils.metrics import compute_3d_occupancy_metrics
            occ = compute_3d_occupancy_metrics(radar_pts, lidar_pts, distance_threshold=0.5)
            metrics.update(occ)
            prec = occ.get('distance_precision', 0.0)
            rec = occ.get('distance_recall', 0.0)
            acc = occ.get('distance_accuracy', 0.0)
            self.log(f"  Precision={prec:.3f}, Recall={rec:.3f}, Accuracy={acc:.3f}")

        metrics["num_frames"] = len(adc_paths)
        metrics["num_aggregated_points"] = len(radar_pts)

        # --- Per-angle convergence: evaluate at checkpoints ---
        if lidar_pts is not None and not skip_convergence:
            convergence = self._compute_per_angle_convergence(
                per_view_data, lidar_pts, device,
            )
            if convergence:
                metrics["convergence"] = convergence
                np.savez(
                    os.path.join(self.output_dir, "convergence.npz"),
                    n_views=np.array([c["n_views"] for c in convergence]),
                    f1=np.array([c["f1"] for c in convergence]),
                    precision=np.array([c["precision"] for c in convergence]),
                    recall=np.array([c["recall"] for c in convergence]),
                )

        self.save_metrics(metrics)
        return metrics

    def generate_figures(self, metrics: dict) -> List[str]:
        """Generate Open3D 2-set × 9-view PNGs + per-angle convergence PNG.

        Set A: Aggregated dense radar reconstruction + mesh (plasma by intensity)
        Set B: Single-frame LiDAR + mesh (uniform blue)

        No additional viz filtering — per-view percentile + global filtering in
        run() already produces a clean point cloud.
        """
        saved = []

        from .utils.visualization_open3d import render_pointcloud_views, render_lidar_views

        mesh_path = self.scene.mesh_file

        # --- Set A: Aggregated reconstruction + mesh ---
        try:
            agg_path = os.path.join(self.output_dir, "aggregated_pcl.npy")
            if os.path.isfile(agg_path):
                radar_pts = np.load(agg_path)
                intensity = None
                intensity_path = os.path.join(self.output_dir, "aggregated_intensity.npy")
                if os.path.isfile(intensity_path):
                    intensity = np.load(intensity_path)

                self.log(f"  Set A: {len(radar_pts)} points (already filtered by run())")

                set_a_dir = os.path.join(self.output_dir, "dense_radar")
                paths = render_pointcloud_views(
                    radar_pts, intensity, mesh_path, set_a_dir,
                    radar_config_path=self.scene.dense_config,
                    set_name="dense_radar",
                )
                saved.extend(paths)
                self.log(f"  Set A (dense radar): {len(paths)} PNGs → {set_a_dir}")
        except Exception as e:
            self.log(f"  Set A (dense radar) FAILED: {e}")

        try:
            lidar_path = self._get_single_lidar_frame()
            if lidar_path and os.path.isfile(lidar_path):
                lidar_data = np.load(lidar_path)
                lidar_xyz = lidar_data[:, :3] if lidar_data.ndim == 2 and lidar_data.shape[1] >= 3 else lidar_data

                set_b_dir = os.path.join(self.output_dir, "lidar")
                paths = render_lidar_views(
                    lidar_xyz, mesh_path, set_b_dir,
                    set_name="lidar",
                )
                saved.extend(paths)
                self.log(f"  Set B (LiDAR single frame): {len(paths)} PNGs → {set_b_dir}")
            elif self.scene.lidar_pcl and os.path.isfile(self.scene.lidar_pcl):
                lidar_data = np.load(self.scene.lidar_pcl)
                lidar_xyz = lidar_data[:, :3] if lidar_data.ndim == 2 and lidar_data.shape[1] >= 3 else lidar_data
                set_b_dir = os.path.join(self.output_dir, "lidar")
                paths = render_lidar_views(lidar_xyz, mesh_path, set_b_dir, set_name="lidar")
                saved.extend(paths)
                self.log(f"  Set B (LiDAR aggregated fallback): {len(paths)} PNGs → {set_b_dir}")
        except Exception as e:
            self.log(f"  Set B (LiDAR) FAILED: {e}")

        # --- Per-angle convergence plot ---
        try:
            conv_path = os.path.join(self.output_dir, "convergence.npz")
            if os.path.isfile(conv_path):
                from .utils.visualization import save_convergence_figure

                data = np.load(conv_path)
                n_views = data["n_views"].tolist()
                series = {
                    "F1": data["f1"].tolist(),
                    "Precision": data["precision"].tolist(),
                    "Recall": data["recall"].tolist(),
                }
                fig_path = os.path.join(self.output_dir, "convergence.png")
                save_convergence_figure(n_views, series, fig_path, xlabel="Number of views")
                saved.append(fig_path)
                self.log(f"  Saved convergence figure → {fig_path}")
        except Exception as e:
            self.log(f"  Convergence figure FAILED: {e}")

        return saved

    def _get_single_lidar_frame(self, index: int = 4) -> Optional[str]:
        """Get the nth LiDAR frame file (0-indexed) from data/{scene}/lidar/."""
        lidar_dir = os.path.join(self.scene.data_dir, "lidar")
        if not os.path.isdir(lidar_dir):
            return None
        import glob
        frames = sorted(glob.glob(os.path.join(lidar_dir, "lidar_frame_*.npy")))
        if index < len(frames):
            self.log(f"  Using single LiDAR frame: {os.path.basename(frames[index])}")
            return frames[index]
        return None

    def _compute_per_angle_convergence(
        self,
        per_view_data: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        lidar_pts: np.ndarray,
        device: torch.device,
    ) -> List[dict]:
        """Evaluate reconstruction quality at different numbers of accumulated views.

        Uses pre-computed per-view (pts_world, intensity) from run() to avoid
        re-processing ADC files. At each checkpoint, applies the same global
        filtering as the final aggregation.

        Checks at 10%, 25%, 50%, 75%, and 100% of total views.
        """
        from mmir.evaluation.utils.consolidate_views import VoxelAccumulatorGPU
        from .utils.metrics import compute_3d_occupancy_metrics

        n_total = len(per_view_data)
        checkpoints = sorted(set([
            max(1, int(n_total * frac))
            for frac in [0.1, 0.25, 0.5, 0.75, 1.0]
        ]))

        convergence = []
        accumulator = VoxelAccumulatorGPU(self.voxel_size_m, device)

        for view_idx, (pts_world, intensity) in enumerate(per_view_data):
            if pts_world is not None and len(pts_world) > 0:
                accumulator.add_points_batch(
                    pts_world[:, 0], pts_world[:, 1], pts_world[:, 2],
                    intensity, view_idx,
                )

            n_views = view_idx + 1
            if n_views in checkpoints:
                pts, ints, vds = self._apply_global_filtering(accumulator, device)
                # Apply same composite-score threshold as run()
                if pts is not None and len(pts) > 0 and self.radar_percentile is not None:
                    composite_score = vds * ints
                    thr = np.percentile(composite_score, self.radar_percentile)
                    keep_int = composite_score >= thr
                    pts = pts[keep_int]
                    ints = ints[keep_int]
                if pts is not None and len(pts) > 0:
                    occ = compute_3d_occupancy_metrics(pts, lidar_pts, distance_threshold=0.5)
                    p = occ.get("distance_precision", 0.0)
                    r = occ.get("distance_recall", 0.0)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    convergence.append({
                        "n_views": n_views,
                        "f1": f1,
                        "precision": p,
                        "recall": r,
                        "num_points": len(pts),
                    })
                    self.log(f"  Convergence @ {n_views} views: F1={f1:.3f}, "
                             f"P={p:.3f}, R={r:.3f}, pts={len(pts)}")
                else:
                    convergence.append({
                        "n_views": n_views,
                        "f1": 0.0, "precision": 0.0, "recall": 0.0,
                        "num_points": 0,
                    })
                    self.log(f"  Convergence @ {n_views} views: no points after filtering")

        return convergence

def run_3d_reconstruction_all(
    scenes: List[SceneInfo],
    output_root: str,
    render: bool = True,
    batch_size: int = 5000,
    azimuth_range: Tuple[float, float] = (-21.0, 69.0),
    azimuth_step: float = 1.0,
    radar_percentile: float = 98.0,
    aggregate_percentile: float = 90.0,
    skip_figures: bool = False,
    skip_latex: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run 3D reconstruction evaluation for all scenes."""
    per_scene = {}

    for scene in scenes:
        if not scene.dense_config:
            if verbose:
                print(f"  Skipping {scene.name}: no dense config")
            continue

        evaluator = ReconstructionEvaluator(
            scene, output_root,
            azimuth_range=azimuth_range,
            azimuth_step=azimuth_step,
            batch_size=batch_size,
            radar_percentile=radar_percentile,
            aggregate_percentile=aggregate_percentile,
            verbose=verbose,
        )
        if render:
            evaluator.render()
        metrics = evaluator.run(skip_convergence=skip_figures)
        if not skip_figures:
            evaluator.generate_figures(metrics)
        per_scene[scene.name] = metrics

    # Aggregate — keys match compute_distance_based_occupancy_metrics + compute_point_cloud_rmse_and_chamfer
    metric_keys = ["distance_precision", "distance_recall", "distance_accuracy",
                    "pc_rmse", "chamfer_distance", "relative_chamfer_distance",
                    "num_frames", "num_aggregated_points"]
    agg_dir = os.path.join(output_root, "3d_reconstruction")
    os.makedirs(agg_dir, exist_ok=True)

    flat_metrics = {}
    for scene_name, m in per_scene.items():
        flat_metrics[scene_name] = {k: m.get(k) for k in metric_keys}

    agg_keys = [k for k in metric_keys if k not in ("num_frames", "num_aggregated_points")]
    agg = aggregate_metrics(flat_metrics, agg_keys)

    with open(os.path.join(agg_dir, "aggregate_metrics.json"), "w") as f:
        json.dump({"per_scene": flat_metrics, "aggregate": agg}, f, indent=2)

    if not skip_latex:
        from .utils.latex_export import generate_3d_reconstruction_table
        generate_3d_reconstruction_table(
            flat_metrics, agg,
            os.path.join(agg_dir, "table4_reconstruction.tex"),
        )

    if verbose:
        print(f"\n=== 3D Reconstruction Aggregate ===")
        for key, stats in agg.items():
            print(f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    return per_scene
