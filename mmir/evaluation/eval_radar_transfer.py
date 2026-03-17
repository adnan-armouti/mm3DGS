"""Evaluation #2: Radar Transfer (cascaded → single-chip).

After training on cascaded radar (12TX×16RX), re-render with single-chip radar
(3TX×4RX) using frozen trained parameters. Evaluates whether learned materials
generalize to a different sensor.

SC alignment uses trajectory-transfer + direct refinement with trained materials
from ``alignment_data/{scene}/single_chip/single_chip_frame_{N}_refined_trained.json``.
This ensures the alignment pose is optimized for the same materials used during eval.
Range bin cropping uses bins 15..110 (excludes DFT wrap-around near bin 128).
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .base_evaluator import BaseEvaluator, aggregate_metrics
from .scene_registry import SceneInfo

# Single-chip antenna pattern (TI IWR1443) — NOT the cascaded MMWCAS patterns
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SC_ANTENNA_PATTERN = os.path.join(_PROJECT_ROOT, "assets", "antenna_pattern", "IWR1443", "pattern_76.npy")


def _render_benchmark_script(
    mesh_file: str, radar_config_path: str, benchmark_dir: str, output_path: str
) -> str:
    """Generate a Python script string to render benchmark Sionna ADC in a subprocess.

    Sionna requires cuda_ad_mono_polarized variant which conflicts with
    cuda_ad_rgb already loaded in the parent process.
    """
    return f'''
import sys, os, json
import numpy as np

benchmark_dir = {benchmark_dir!r}
mesh_file = {mesh_file!r}
radar_config_path = {radar_config_path!r}
output_path = {output_path!r}

# Load params
with open(os.path.join(benchmark_dir, "best_params.json")) as f:
    best_params = json.load(f)
mat_keys = {{"eta_r", "sigma", "d", "s", "xpd_coefficient"}}
mat_params = {{k: v for k, v in best_params.items() if k in mat_keys}}

with open(os.path.join(benchmark_dir, "benchmark_config.json")) as f:
    cfg = json.load(f)

try:
    from mmir.benchmark_sionna_final.sionna_ad_forward_model import SionnaADForwardModel
except ImportError:
    print("ERROR: mmir.benchmark_sionna_final not available (Sionna RT baseline required)")
    sys.exit(1)

model = SionnaADForwardModel(
    mesh_path=mesh_file,
    radar_config_path=radar_config_path,
    initial_material=mat_params,
)
paths = model.render_ad(
    seed=42,
    max_depth=cfg.get("max_depth", 1),
    samples_per_src=cfg.get("samples_per_src", 1_000_000),
)
adc_real, adc_imag, n_scat = model.synthesize_adc_drjit(paths)
adc_real_np, adc_imag_np = model.extract_adc_values(adc_real, adc_imag)
adc = np.stack([adc_real_np, adc_imag_np], axis=-1)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, adc)
print(f"Benchmark ADC saved: shape={{adc.shape}}, scatterers={{n_scat}}")
'''


class RadarTransferEvaluator(BaseEvaluator):
    """Evaluate radar transfer from cascaded to single-chip."""

    def __init__(
        self,
        scene: SceneInfo,
        output_root: str,
        single_chip_frame: Optional[int] = None,
        n_hits_per_rx: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__(scene, output_root, verbose)
        # Auto-detect closest single-chip frame if not specified
        if single_chip_frame is None:
            single_chip_frame = scene.get_closest_single_chip_frame()
        self.single_chip_frame = single_chip_frame
        self._n_hits_override = n_hits_per_rx

    @property
    def _benchmark_dir(self) -> str:
        """Resolve effective benchmark Sionna training dir.

        Prefers benchmark_training_dir (same source as Eval #1) when it has
        the required files (best_params.json + benchmark_config.json), then
        falls back to benchmark_sionna_dir (dedicated Sionna AD output).
        """
        if self.scene.benchmark_training_dir:
            bp = os.path.join(self.scene.benchmark_training_dir, "best_params.json")
            bc = os.path.join(self.scene.benchmark_training_dir, "benchmark_config.json")
            if os.path.isfile(bp) and os.path.isfile(bc):
                return self.scene.benchmark_training_dir
        if self.scene.benchmark_sionna_dir:
            return self.scene.benchmark_sionna_dir
        return ""

    @property
    def eval_name(self) -> str:
        return "radar_transfer"

    def _find_refined_trained_config(self, sc_frame: int) -> Optional[str]:
        """Look for an existing refined_trained config, trying exact then nearest frame."""
        import glob as _glob
        import re as _re

        sc_dir = os.path.join(
            _PROJECT_ROOT, "data", "alignment_data", self.scene.name, "single_chip"
        )

        # Exact frame match
        exact = os.path.join(sc_dir, f"single_chip_frame_{sc_frame}_refined_trained.json")
        if os.path.isfile(exact):
            return exact

        # Nearest match (frame number may differ by ±2)
        candidates = sorted(_glob.glob(os.path.join(sc_dir, "single_chip_frame_*_refined_trained.json")))
        if candidates:
            def _fnum(p):
                m = _re.search(r"frame_(\d+)_refined_trained", p)
                return int(m.group(1)) if m else 0
            nearest = min(candidates, key=lambda p: abs(_fnum(p) - sc_frame))
            self.single_chip_frame = _fnum(nearest)
            return nearest

        return None

    def _ensure_trajectory_transfer(self) -> bool:
        """Ensure trajectory-transferred SC configs exist (Stage 1).

        Runs cascade alignment + trajectory transfer if not already done.
        Returns True if warm-start configs are available.
        """
        import glob as _glob

        sc_dir = os.path.join(
            _PROJECT_ROOT, "data", "alignment_data", self.scene.name, "single_chip"
        )
        # Check if any _aligned.json exists
        existing = _glob.glob(os.path.join(sc_dir, "single_chip_frame_*_aligned.json"))
        if existing:
            return True

        self.log(f"  Trajectory transfer configs missing — generating...")
        try:
            from mmir.preprocessing.alignment.sc_trajectory_transfer import run_pipeline
            run_pipeline(
                scene_name=self.scene.name,
                seq_idx=self.scene.seq,
                data_root=os.path.join(_PROJECT_ROOT, "data"),
                output_root=os.path.join(_PROJECT_ROOT, "data", "alignment_data"),
                verbose=True,
            )
            # Verify
            existing = _glob.glob(os.path.join(sc_dir, "single_chip_frame_*_aligned.json"))
            return len(existing) > 0
        except Exception as e:
            self.log(f"  Trajectory transfer failed: {e}")
            return False

    def _generate_refined_trained_config(self, sc_frame: int) -> Optional[str]:
        """Generate a refined_trained config by running trained-material alignment.

        Requires trajectory-transferred warm-start config to exist.
        """
        import glob as _glob
        import re as _re
        import gc
        import time
        from scipy.optimize import minimize as _minimize

        sc_dir = os.path.join(
            _PROJECT_ROOT, "data", "alignment_data", self.scene.name, "single_chip"
        )

        # Find warm-start config (trajectory-transferred)
        warm_candidates = sorted(_glob.glob(os.path.join(sc_dir, "single_chip_frame_*_aligned.json")))
        if not warm_candidates:
            self.log(f"  No warm-start configs for trained-material refinement")
            return None

        # Pick nearest to requested sc_frame
        def _fnum(p):
            m = _re.search(r"frame_(\d+)_aligned", p)
            return int(m.group(1)) if m else 0
        warm_config = min(warm_candidates, key=lambda p: abs(_fnum(p) - sc_frame))
        actual_frame = _fnum(warm_config)

        training_dir = self.scene.our_training_dir
        if not training_dir or not os.path.isfile(os.path.join(training_dir, "best_materials.npz")):
            self.log(f"  No training dir for trained-material refinement")
            return None

        mesh = self.scene.mesh_file
        gt_path = self.scene.get_single_chip_gt_adc(actual_frame)
        if gt_path is None:
            gt_path = os.path.join(
                self.scene.data_dir, "radar", f"single_chip_frame_{actual_frame}.npy"
            )
        if not os.path.isfile(gt_path):
            self.log(f"  No GT ADC for frame {actual_frame}")
            return None

        self.log(f"  Running trained-material SC refinement (frame {actual_frame})...")
        self.log(f"    Warm start: {os.path.basename(warm_config)}")

        import drjit as dr
        from mmir.data.ra_utils import adc_to_ra_image_single_chip, ra_polar_to_cartesian
        from mmir.data.io_utils import compute_range_res_from_cfg

        SKIP_NEAR, SKIP_FAR = 15, 110

        with open(warm_config) as f:
            base_config = json.load(f)
        range_res = compute_range_res_from_cfg(warm_config)

        # Load GT
        gt_raw = np.load(gt_path)
        gt_complex = gt_raw[0].transpose(1, 0, 2)
        adc_gt = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
        ra_gt = adc_to_ra_image_single_chip(adc_gt)[:, SKIP_NEAR:SKIP_FAR]
        gt_cart = ra_polar_to_cartesian(ra_gt, range_res)

        def _norm(v, eps=1e-9):
            return v / (np.linalg.norm(v) + eps)

        def _minmax(arr):
            mn, mx = arr.min(), arr.max()
            return np.zeros_like(arr) if mx - mn < 1e-30 else (arr - mn) / (mx - mn)

        def _apply_2dof(cfg, dr_m, da_deg):
            tx = np.array([t["pos_mm"] for t in cfg["tx_array"]], dtype=float) / 1000.0
            rx = np.array([r["pos_mm"] for r in cfg["rx_array"]], dtype=float) / 1000.0
            ap = np.vstack([tx, rx]); nt = len(tx)
            bc = np.mean(ap, axis=0)
            bs = _norm(np.array(cfg["tx_array"][0]["boresight"], dtype=float))
            rel = ap - bc
            z = np.array([0.0, 0.0, 1.0])
            xb = _norm(np.cross(bs, z))
            if np.linalg.norm(xb) < 1e-6: xb = np.array([1.0, 0.0, 0.0])
            tr = bs * dr_m + xb * (10.0 * np.radians(da_deg))
            nap = rel + bc + tr
            return nap[:nt] * 1000.0, nap[nt:] * 1000.0, bs

        def _apply_4dof(cfg, dr_m, da_deg, re_deg, ra_deg):
            tx = np.array([t["pos_mm"] for t in cfg["tx_array"]], dtype=float) / 1000.0
            rx = np.array([r["pos_mm"] for r in cfg["rx_array"]], dtype=float) / 1000.0
            ap = np.vstack([tx, rx]); nt = len(tx)
            bc = np.mean(ap, axis=0)
            bbs = _norm(np.array(cfg["tx_array"][0]["boresight"], dtype=float))
            z = np.array([0.0, 0.0, 1.0])
            def _aa(w):
                th = np.linalg.norm(w)
                if th < 1e-9: return np.eye(3)
                k = w/th; K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
                return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
            Re = _aa(np.array([0,0,np.radians(re_deg)]))
            ba = _norm(Re @ bbs)
            xb = _norm(np.cross(ba, z))
            if np.linalg.norm(xb) < 1e-6: xb = np.array([1.0,0.0,0.0])
            Ra = _aa(xb * np.radians(ra_deg)); R = Ra @ Re
            nb = _norm(R @ bbs)
            rr = (R @ (ap - bc).T).T
            xn = _norm(np.cross(nb, z))
            if np.linalg.norm(xn) < 1e-6: xn = np.array([1.0,0.0,0.0])
            tr = nb * dr_m + xn * (10.0 * np.radians(da_deg))
            nap = rr + bc + tr
            return nap[:nt] * 1000.0, nap[nt:] * 1000.0, nb

        def _update(wrapper, tx_mm, rx_mm, bs):
            import mitsuba as mi
            sc = wrapper.renderer.scene_ctx
            nt, nr = tx_mm.shape[0], rx_mm.shape[0]
            sc.tx_array.positions = mi.Point3f(
                mi.Float((tx_mm[:,0]/1000).tolist()), mi.Float((tx_mm[:,1]/1000).tolist()), mi.Float((tx_mm[:,2]/1000).tolist()))
            sc.tx_array.orientations = mi.Vector3f(
                mi.Float([bs[0]]*nt), mi.Float([bs[1]]*nt), mi.Float([bs[2]]*nt))
            sc.rx_array.positions = mi.Point3f(
                mi.Float((rx_mm[:,0]/1000).tolist()), mi.Float((rx_mm[:,1]/1000).tolist()), mi.Float((rx_mm[:,2]/1000).tolist()))
            sc.rx_array.orientations = mi.Vector3f(
                mi.Float([bs[0]]*nr), mi.Float([bs[1]]*nr), mi.Float([bs[2]]*nr))

        def _eval(wrapper):
            try:
                adc = wrapper.render_forward(seed=42)
                if np.isnan(adc).any(): return -1.0
                ra = adc_to_ra_image_single_chip(adc)[:, SKIP_NEAR:SKIP_FAR]
                rc = ra_polar_to_cartesian(ra, range_res)
                return float(np.corrcoef(_minmax(rc).ravel(), _minmax(gt_cart).ravel())[0, 1])
            except Exception:
                return -1.0

        def _make_wrapper(n_hits=800):
            from .renderer_wrapper import RendererWrapper as _RW
            w = _RW.from_training_dir(
                mesh_file=mesh, config_file=warm_config,
                training_dir=training_dir,
                tx_pattern_file=_SC_ANTENNA_PATTERN, rx_pattern_file=_SC_ANTENNA_PATTERN,
                verbose=False)
            w._render_config.n_hits_per_rx = n_hits
            w.renderer.config = w._render_config
            w.renderer.sampler.n_hits_per_rx = n_hits
            w.load_materials(training_dir)
            w.load_learned_normals(training_dir)
            return w

        t0 = time.time()

        # ── 2DOF grid search ──
        wrapper = _make_wrapper(n_hits=800)
        best2 = [0.0, 0.0]; best2_cc = -999.0
        for dr_m in np.arange(-1.0, 1.25, 0.25):
            for da in np.arange(-5.0, 5.5, 1.0):
                tx, rx, bs = _apply_2dof(base_config, dr_m, da)
                _update(wrapper, tx, rx, bs)
                cc = _eval(wrapper)
                if cc > best2_cc:
                    best2_cc = cc; best2 = [float(dr_m), float(da)]

        # Refine 2DOF
        ref2_cc = [best2_cc]; ref2_p = [list(best2)]
        def _obj2(x):
            tx, rx, bs = _apply_2dof(base_config, x[0], x[1])
            _update(wrapper, tx, rx, bs)
            cc = _eval(wrapper)
            if cc > ref2_cc[0]: ref2_cc[0] = cc; ref2_p[0] = [float(x[0]), float(x[1])]
            return -cc
        _minimize(_obj2, x0=best2, method="Nelder-Mead",
                  options={"maxiter": 30, "maxfev": 35, "xatol": 0.02, "fatol": 0.001})

        # Validate 2DOF
        wrapper._render_config.n_hits_per_rx = 1500
        wrapper.renderer.config = wrapper._render_config
        wrapper.renderer.sampler.n_hits_per_rx = 1500
        tx2, rx2, bs2 = _apply_2dof(base_config, ref2_p[0][0], ref2_p[0][1])
        _update(wrapper, tx2, rx2, bs2)
        cc_2dof = _eval(wrapper)
        self.log(f"    2DOF: CC={cc_2dof:.4f} (r={ref2_p[0][0]:.3f} az={ref2_p[0][1]:.2f})")
        wrapper.cleanup(); del wrapper; dr.sync_thread(); gc.collect()

        # ── 4DOF grid search ──
        wrapper4 = _make_wrapper(n_hits=800)
        best4 = [0.0, 0.0, 0.0, 0.0]; best4_cc = -999.0
        for dr_m in np.arange(-1.0, 1.25, 0.5):
            for da in np.arange(-5.0, 5.5, 2.5):
                for re in np.arange(-5.0, 5.5, 2.5):
                    for ra in np.arange(-5.0, 5.5, 2.5):
                        tx, rx, bs = _apply_4dof(base_config, dr_m, da, re, ra)
                        _update(wrapper4, tx, rx, bs)
                        cc = _eval(wrapper4)
                        if cc > best4_cc:
                            best4_cc = cc; best4 = [float(dr_m), float(da), float(re), float(ra)]

        # Refine 4DOF
        ref4_cc = [best4_cc]; ref4_p = [list(best4)]
        def _obj4(x):
            tx, rx, bs = _apply_4dof(base_config, x[0], x[1], x[2], x[3])
            _update(wrapper4, tx, rx, bs)
            cc = _eval(wrapper4)
            if cc > ref4_cc[0]: ref4_cc[0] = cc; ref4_p[0] = [float(v) for v in x]
            return -cc
        _minimize(_obj4, x0=best4, method="Nelder-Mead",
                  options={"maxiter": 50, "maxfev": 60, "xatol": 0.01, "fatol": 0.001})

        # Validate 4DOF
        wrapper4._render_config.n_hits_per_rx = 1500
        wrapper4.renderer.config = wrapper4._render_config
        wrapper4.renderer.sampler.n_hits_per_rx = 1500
        tx4, rx4, bs4 = _apply_4dof(base_config, *ref4_p[0])
        _update(wrapper4, tx4, rx4, bs4)
        cc_4dof = _eval(wrapper4)
        self.log(f"    4DOF: CC={cc_4dof:.4f}")
        wrapper4.cleanup(); del wrapper4; dr.sync_thread(); gc.collect()

        # ── Select winner ──
        if cc_2dof >= cc_4dof:
            winner_tx, winner_rx, winner_bs, winner_cc = tx2, rx2, bs2, cc_2dof
            winner = "2dof"
        else:
            winner_tx, winner_rx, winner_bs, winner_cc = tx4, rx4, bs4, cc_4dof
            winner = "4dof"

        self.log(f"    Winner: {winner} CC={winner_cc:.4f} ({time.time()-t0:.0f}s)")

        # Save config
        out_path = os.path.join(sc_dir, f"single_chip_frame_{actual_frame}_refined_trained.json")
        config = json.loads(json.dumps(base_config))
        for i, tx in enumerate(config["tx_array"]):
            tx["pos_mm"] = winner_tx[i].tolist()
            tx["boresight"] = winner_bs.tolist()
        for i, rx in enumerate(config["rx_array"]):
            rx["pos_mm"] = winner_rx[i].tolist()
            rx["boresight"] = winner_bs.tolist()
        os.makedirs(sc_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save log
        log = {"scene": self.scene.name, "frame": actual_frame, "winner": winner,
               "cc_2dof": float(cc_2dof), "cc_4dof": float(cc_4dof), "winner_cc": float(winner_cc)}
        with open(os.path.join(sc_dir, f"single_chip_frame_{actual_frame}_trained_alignment_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        self.single_chip_frame = actual_frame
        return out_path

    def _ensure_aligned_sc_config(self) -> Optional[str]:
        """Ensure an aligned single-chip config exists for this scene.

        Cascading generation:
        1. If ``_refined_trained.json`` exists -> use it
        2. Else if trajectory warm-start ``_aligned.json`` exists -> run trained-mat refinement
        3. Else -> run trajectory transfer first, then trained-mat refinement

        Returns:
            Path to the refined SC config, or None on failure.
        """
        if hasattr(self, "_cached_aligned_sc_config"):
            return self._cached_aligned_sc_config

        if self.single_chip_frame is None:
            self._cached_aligned_sc_config = None
            return None

        sc_frame = self.single_chip_frame

        # Step 1: Check for existing refined_trained config
        found = self._find_refined_trained_config(sc_frame)
        if found:
            self.log(f"  Aligned SC config (refined_trained): {found}")
            self._cached_aligned_sc_config = found
            return found

        # Step 2: Ensure trajectory transfer warm-start exists
        if not self._ensure_trajectory_transfer():
            self.log(f"  Cannot generate trajectory transfer for {self.scene.name}")
            self._cached_aligned_sc_config = None
            return None

        # Step 3: Run trained-material refinement
        result = self._generate_refined_trained_config(sc_frame)
        if result and os.path.isfile(result):
            self.log(f"  Generated refined_trained config: {result}")
            self._cached_aligned_sc_config = result
            return result

        self.log(f"  Failed to generate refined_trained config for {self.scene.name}")
        self._cached_aligned_sc_config = None
        return None

    def render(self) -> Optional[np.ndarray]:
        """Phase 1: Forward render with single-chip config + trained materials.

        Also renders with default (untrained) materials for baseline comparison.
        Returns rendered ADC array (n_tx, n_rx, n_adc, 2) or None on failure.
        """
        if self.single_chip_frame is None:
            self.log("No single-chip frame available, skipping render")
            return None

        # Ensure aligned SC config (runs Stage 1 + Stage 2 if needed)
        sc_config = self._ensure_aligned_sc_config()
        if sc_config is None:
            self.log(f"No single-chip config found for frame {self.single_chip_frame}")
            return None

        # Skip main render if already done
        adc_path = os.path.join(self.output_dir, "adc_rendered_single_chip.npy")
        adc = None
        if os.path.isfile(adc_path):
            self.log(f"Rendered ADC already exists, skipping render")
            adc = np.load(adc_path)

        if adc is None and not self.scene.our_training_dir:
            self.log("No training output directory available")
            return None

        if adc is None:
            self.log(f"Rendering single-chip frame {self.single_chip_frame}...")
            self.log(f"  Config: {sc_config}")
            self.log(f"  Mesh: {self.scene.mesh_file}")

            from .renderer_wrapper import RendererWrapper

            # --- Render with trained materials ---
            # Use from_training_dir() to reproduce exact renderer settings
            # (BSDF model, etc.) — matching Eval #1 behavior.
            # Override antenna patterns with single-chip pattern (not cascaded MMWCAS).
            wrapper = RendererWrapper.from_training_dir(
                mesh_file=self.scene.mesh_file,
                config_file=sc_config,
                training_dir=self.scene.our_training_dir,
                tx_pattern_file=_SC_ANTENNA_PATTERN,
                rx_pattern_file=_SC_ANTENNA_PATTERN,
                verbose=self.verbose,
            )
            self.log(f"  SMS={'enabled' if wrapper._render_config.enable_sms else 'disabled'}")
            if self._n_hits_override is not None:
                wrapper._render_config.n_hits_per_rx = self._n_hits_override
                wrapper.renderer.config = wrapper._render_config
                wrapper.renderer.sampler.n_hits_per_rx = self._n_hits_override
                self.log(f"  Overriding n_hits_per_rx → {self._n_hits_override}")
            # Load materials and normals only — NOT learned patterns.
            # Learned patterns are for MMWCAS cascaded array (12TX×16RX) and must
            # not be applied to single-chip (3TX×4RX) which uses its own pattern.
            meta = wrapper.load_materials(self.scene.our_training_dir)
            wrapper.load_learned_normals(self.scene.our_training_dir)
            self.log(f"  Materials+normals loaded (training corr={meta.get('correlation', 'N/A')})")

            adc = wrapper.render_forward(seed=42)
            self.log(f"  Rendered ADC shape: {adc.shape}")

            np.save(adc_path, adc)
            self.log(f"  Saved trained → {adc_path}")
            wrapper.cleanup()

        # --- Render with default (untrained) materials ---
        default_path = os.path.join(self.output_dir, "adc_default_single_chip.npy")
        if not os.path.isfile(default_path) and self.scene.our_training_dir:
            from .renderer_wrapper import RendererWrapper as _RW  # noqa: F811
            self.log("  Rendering default (untrained) baseline...")
            wrapper_default = _RW.from_training_dir(
                mesh_file=self.scene.mesh_file,
                config_file=sc_config,
                training_dir=self.scene.our_training_dir,
                tx_pattern_file=_SC_ANTENNA_PATTERN,
                rx_pattern_file=_SC_ANTENNA_PATTERN,
                verbose=False,
            )
            if self._n_hits_override is not None:
                wrapper_default._render_config.n_hits_per_rx = self._n_hits_override
                wrapper_default.renderer.config = wrapper_default._render_config
                wrapper_default.renderer.sampler.n_hits_per_rx = self._n_hits_override
            # No load_materials() call — uses renderer's default ITU materials
            adc_default = wrapper_default.render_forward(seed=42)
            np.save(default_path, adc_default)
            self.log(f"  Saved default → {default_path}")
            wrapper_default.cleanup()

        # --- Render with benchmark Sionna materials ---
        # NOTE: Sionna requires cuda_ad_mono_polarized variant which conflicts
        # with cuda_ad_rgb already loaded. Must run in a subprocess.
        benchmark_path = os.path.join(self.output_dir, "adc_benchmark_single_chip.npy")
        bm_dir = self._benchmark_dir
        if not os.path.isfile(benchmark_path) and bm_dir:
            best_params_file = os.path.join(bm_dir, "best_params.json")
            if os.path.isfile(best_params_file):
                self.log(f"  Rendering benchmark Sionna baseline (subprocess) from {bm_dir}...")
                import subprocess
                import sys
                script = _render_benchmark_script(
                    mesh_file=self.scene.mesh_file,
                    radar_config_path=sc_config,
                    benchmark_dir=bm_dir,
                    output_path=benchmark_path,
                )
                # Inherit CUDA_VISIBLE_DEVICES from current process
                env = os.environ.copy()
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True, text=True, env=env,
                )
                if result.returncode == 0 and os.path.isfile(benchmark_path):
                    bm_adc = np.load(benchmark_path)
                    self.log(f"  Saved benchmark → {benchmark_path} (shape={bm_adc.shape})")
                else:
                    self.log(f"  Benchmark render failed (exit={result.returncode})")
                    if result.stderr:
                        # Print last few lines of error
                        err_lines = result.stderr.strip().split('\n')
                        for line in err_lines[-5:]:
                            self.log(f"    {line}")

        return adc

    def run(self) -> dict:
        """Phase 2: Evaluate transfer quality.

        Loads pre-rendered single-chip ADC (or renders if not available),
        compares to GT single-chip ADC via RA metrics.
        """
        metrics = {
            "scene": self.scene.name,
            "single_chip_frame": self.single_chip_frame,
        }

        if self.single_chip_frame is None:
            self.log("No single-chip frame available")
            self.save_metrics(metrics)
            return metrics

        # Check if rendered ADC exists
        adc_path = os.path.join(self.output_dir, "adc_rendered_single_chip.npy")
        if not os.path.isfile(adc_path):
            self.log("Rendered ADC not found, triggering render...")
            adc = self.render()
            if adc is None:
                self.save_metrics(metrics)
                return metrics
        else:
            self.log(f"Loading pre-rendered ADC from {adc_path}")

        # Load GT single-chip ADC
        gt_path = self.scene.get_single_chip_gt_adc(self.single_chip_frame)
        if gt_path is None or not os.path.isfile(gt_path):
            self.log(f"No GT single-chip ADC for frame {self.single_chip_frame}")
            self.save_metrics(metrics)
            return metrics

        self.log("Computing transfer metrics...")

        # Beamform ADC → RA image via single-chip virtual array (3TX×4RX → 8-el)
        # This matches the RA images shown in generate_figures().
        from mmir.data.ra_utils import adc_to_ra_image_single_chip, ra_polar_to_cartesian
        from .utils.metrics import compute_ra_metrics, compute_cart_ra_metrics

        adc_rendered = np.load(adc_path)  # (3, 4, 128, 2) float32

        # GT single-chip is (128, 4, 3, 128) complex128 — convert:
        #   arr[0] → (4, 3, 128), transpose → (3, 4, 128), split complex → (3, 4, 128, 2)
        gt_raw = np.load(gt_path)
        gt_complex = gt_raw[0].transpose(1, 0, 2)  # (3, 4, 128) complex
        adc_gt = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)

        # Beamform to RA polar images (63, 128) — Hann-windowed, FFT-shifted
        ra_rendered = adc_to_ra_image_single_chip(adc_rendered)
        ra_gt = adc_to_ra_image_single_chip(adc_gt)

        # Save RA polar arrays for visualization
        np.save(os.path.join(self.output_dir, "ra_rendered_polar.npy"), ra_rendered)
        np.save(os.path.join(self.output_dir, "ra_gt_polar.npy"), ra_gt)

        self.log(f"  RA shapes: rendered={ra_rendered.shape}, GT={ra_gt.shape}")

        # Crop to useful range bins:
        #   Skip bins 0..14: TX-RX coupling leakage (near-field hardware artifact)
        #   Skip bins 110..127: DFT circular wrap-around of near-field energy
        SKIP_NEAR = 15
        SKIP_FAR = 110
        ra_rendered_c = ra_rendered[:, SKIP_NEAR:SKIP_FAR]
        ra_gt_c = ra_gt[:, SKIP_NEAR:SKIP_FAR]

        # Compute metrics on beamformed RA (linear scale, after leakage removal)
        ra_metrics = compute_ra_metrics(ra_rendered_c, ra_gt_c)
        metrics["transfer_corr"] = ra_metrics["correlation"]
        metrics["transfer_psnr"] = ra_metrics["psnr"]
        metrics["transfer_ssim"] = ra_metrics["ssim"]
        metrics["transfer_mse"] = ra_metrics["mse"]

        # Also compute dB-scale correlation (more standard for radar)
        from scipy.stats import pearsonr as _pearsonr
        def _to_db(x, floor=-40.0):
            x_n = x / max(x.max(), 1e-30)
            return np.maximum(20.0 * np.log10(np.maximum(x_n, 1e-10)), floor)

        ra_rend_db = _to_db(ra_rendered_c)
        ra_gt_db = _to_db(ra_gt_c)
        corr_db, _ = _pearsonr(ra_rend_db.flatten(), ra_gt_db.flatten())
        metrics["transfer_corr_dB"] = float(corr_db)

        # Cartesian RA metrics (matching Eval #1 methodology)
        range_res = self._get_range_resolution()
        ra_rend_cart = ra_polar_to_cartesian(ra_rendered_c, range_res)
        ra_gt_cart = ra_polar_to_cartesian(ra_gt_c, range_res)
        cart_metrics = compute_cart_ra_metrics(ra_gt_cart, ra_rend_cart)
        metrics["transfer_cart_corr"] = cart_metrics["cart_corr"]
        metrics["transfer_cart_psnr"] = cart_metrics["psnr"]
        metrics["transfer_cart_ssim"] = cart_metrics["ssim"]
        metrics["transfer_cart_mse"] = cart_metrics["mse"]
        metrics["transfer_cart_rmse"] = cart_metrics["rmse"]

        # ADC-level metrics (matching training script methodology)
        from .utils.adc_metrics import compute_adc_metrics as _compute_adc_metrics

        rendered_complex = adc_rendered[..., 0] + 1j * adc_rendered[..., 1]  # (3, 4, 128)
        adc_metrics = _compute_adc_metrics(rendered_complex, gt_complex)
        metrics["transfer_adc_log_mag_mse"] = adc_metrics["adc_log_mag_mse"]
        metrics["transfer_adc_log_mag_rmse"] = adc_metrics["adc_log_mag_rmse"]
        metrics["transfer_adc_minmax_mag_mse"] = adc_metrics["adc_minmax_mag_mse"]
        metrics["transfer_adc_minmax_mag_rmse"] = adc_metrics["adc_minmax_mag_rmse"]
        metrics["transfer_adc_phase_mse"] = adc_metrics["adc_phase_mse"]
        metrics["transfer_adc_phase_rmse"] = adc_metrics["adc_phase_rmse"]

        # --- Default (untrained) baseline ---
        default_adc_path = os.path.join(self.output_dir, "adc_default_single_chip.npy")
        if os.path.isfile(default_adc_path):
            ra_default = adc_to_ra_image_single_chip(np.load(default_adc_path))
            np.save(os.path.join(self.output_dir, "ra_default_polar.npy"), ra_default)
            ra_default_c = ra_default[:, SKIP_NEAR:SKIP_FAR]
            default_ra_metrics = compute_ra_metrics(ra_default_c, ra_gt_c)
            metrics["default_corr"] = default_ra_metrics["correlation"]
            metrics["default_psnr"] = default_ra_metrics["psnr"]
            metrics["default_ssim"] = default_ra_metrics["ssim"]
            # dB-scale default
            ra_def_db = _to_db(ra_default_c)
            corr_def_db, _ = _pearsonr(ra_def_db.flatten(), ra_gt_db.flatten())
            metrics["default_corr_dB"] = float(corr_def_db)
            # Cartesian RA metrics for default
            ra_def_cart = ra_polar_to_cartesian(ra_default_c, range_res)
            default_cart_metrics = compute_cart_ra_metrics(ra_gt_cart, ra_def_cart)
            metrics["default_cart_corr"] = default_cart_metrics["cart_corr"]
            metrics["default_cart_psnr"] = default_cart_metrics["psnr"]
            metrics["default_cart_ssim"] = default_cart_metrics["ssim"]
            metrics["default_cart_mse"] = default_cart_metrics["mse"]
            metrics["default_cart_rmse"] = default_cart_metrics["rmse"]
            # ADC metrics for default
            default_adc = np.load(default_adc_path)
            default_complex = default_adc[..., 0] + 1j * default_adc[..., 1]
            default_adc_metrics = _compute_adc_metrics(default_complex, gt_complex)
            metrics["default_adc_log_mag_mse"] = default_adc_metrics["adc_log_mag_mse"]
            metrics["default_adc_log_mag_rmse"] = default_adc_metrics["adc_log_mag_rmse"]
            metrics["default_adc_minmax_mag_mse"] = default_adc_metrics["adc_minmax_mag_mse"]
            metrics["default_adc_minmax_mag_rmse"] = default_adc_metrics["adc_minmax_mag_rmse"]
            metrics["default_adc_phase_mse"] = default_adc_metrics["adc_phase_mse"]
            metrics["default_adc_phase_rmse"] = default_adc_metrics["adc_phase_rmse"]

        # --- Benchmark Sionna baseline ---
        benchmark_adc_path = os.path.join(self.output_dir, "adc_benchmark_single_chip.npy")
        if os.path.isfile(benchmark_adc_path):
            benchmark_adc = np.load(benchmark_adc_path)
            ra_benchmark = adc_to_ra_image_single_chip(benchmark_adc)
            np.save(os.path.join(self.output_dir, "ra_benchmark_polar.npy"), ra_benchmark)
            ra_benchmark_c = ra_benchmark[:, SKIP_NEAR:SKIP_FAR]
            bm_ra_metrics = compute_ra_metrics(ra_benchmark_c, ra_gt_c)
            metrics["benchmark_corr"] = bm_ra_metrics["correlation"]
            metrics["benchmark_psnr"] = bm_ra_metrics["psnr"]
            metrics["benchmark_ssim"] = bm_ra_metrics["ssim"]
            # dB-scale benchmark
            ra_bm_db = _to_db(ra_benchmark_c)
            corr_bm_db, _ = _pearsonr(ra_bm_db.flatten(), ra_gt_db.flatten())
            metrics["benchmark_corr_dB"] = float(corr_bm_db)
            # Cartesian RA metrics for benchmark
            ra_bm_cart = ra_polar_to_cartesian(ra_benchmark_c, range_res)
            bm_cart_metrics = compute_cart_ra_metrics(ra_gt_cart, ra_bm_cart)
            metrics["benchmark_cart_corr"] = bm_cart_metrics["cart_corr"]
            metrics["benchmark_cart_psnr"] = bm_cart_metrics["psnr"]
            metrics["benchmark_cart_ssim"] = bm_cart_metrics["ssim"]
            metrics["benchmark_cart_mse"] = bm_cart_metrics["mse"]
            metrics["benchmark_cart_rmse"] = bm_cart_metrics["rmse"]
            # ADC metrics for benchmark
            benchmark_complex = benchmark_adc[..., 0] + 1j * benchmark_adc[..., 1]
            bm_adc_metrics = _compute_adc_metrics(benchmark_complex, gt_complex)
            metrics["benchmark_adc_log_mag_mse"] = bm_adc_metrics["adc_log_mag_mse"]
            metrics["benchmark_adc_log_mag_rmse"] = bm_adc_metrics["adc_log_mag_rmse"]
            metrics["benchmark_adc_minmax_mag_mse"] = bm_adc_metrics["adc_minmax_mag_mse"]
            metrics["benchmark_adc_minmax_mag_rmse"] = bm_adc_metrics["adc_minmax_mag_rmse"]
            metrics["benchmark_adc_phase_mse"] = bm_adc_metrics["adc_phase_mse"]
            metrics["benchmark_adc_phase_rmse"] = bm_adc_metrics["adc_phase_rmse"]

        # Load cascaded training correlation for degradation ratio
        if self.scene.our_training_dir:
            from .material_loader import load_our_training_history
            try:
                history = load_our_training_history(self.scene.our_training_dir)
                cart_corr = history.get("cart_corr", [])
                valid = [c for c in cart_corr if c is not None]
                if valid:
                    metrics["train_corr"] = max(valid)
                    if metrics["train_corr"] > 0:
                        metrics["degradation_pct"] = (
                            1.0 - metrics["transfer_corr"] / metrics["train_corr"]
                        ) * 100.0
            except Exception:
                pass

        self.log(f"  Transfer corr={metrics.get('transfer_corr', 'N/A'):.3f} "
                 f"(dB={metrics.get('transfer_corr_dB', 'N/A'):.3f}), "
                 f"Default corr={metrics.get('default_corr', 'N/A'):.3f} "
                 f"(dB={metrics.get('default_corr_dB', 'N/A'):.3f}), "
                 f"Train corr={metrics.get('train_corr', 'N/A')}")
        self.log(f"  Cartesian RA: transfer cart_corr={metrics.get('transfer_cart_corr', 'N/A'):.3f}, "
                 f"PSNR={metrics.get('transfer_cart_psnr', 'N/A'):.2f}, "
                 f"SSIM={metrics.get('transfer_cart_ssim', 'N/A'):.3f}, "
                 f"MSE={metrics.get('transfer_cart_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_cart_rmse', 'N/A'):.4f}")
        if "default_cart_corr" in metrics:
            self.log(f"  Cartesian RA: default  cart_corr={metrics['default_cart_corr']:.3f}, "
                     f"PSNR={metrics['default_cart_psnr']:.2f}, "
                     f"SSIM={metrics['default_cart_ssim']:.3f}, "
                     f"MSE={metrics['default_cart_mse']:.6f}, "
                     f"RMSE={metrics['default_cart_rmse']:.4f}")
        self.log(f"  ADC (log mag): transfer MSE={metrics.get('transfer_adc_log_mag_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_adc_log_mag_rmse', 'N/A'):.4f}")
        self.log(f"  ADC (minmax): transfer MSE={metrics.get('transfer_adc_minmax_mag_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_adc_minmax_mag_rmse', 'N/A'):.4f}")
        self.log(f"  ADC (phase):  transfer MSE={metrics.get('transfer_adc_phase_mse', 'N/A'):.6f}, "
                 f"RMSE={metrics.get('transfer_adc_phase_rmse', 'N/A'):.4f}")
        if "default_adc_log_mag_mse" in metrics:
            self.log(f"  ADC (log mag): default  MSE={metrics['default_adc_log_mag_mse']:.6f}, "
                     f"RMSE={metrics['default_adc_log_mag_rmse']:.4f}")
            self.log(f"  ADC (minmax): default  MSE={metrics['default_adc_minmax_mag_mse']:.6f}, "
                     f"RMSE={metrics['default_adc_minmax_mag_rmse']:.4f}")
            self.log(f"  ADC (phase):  default  MSE={metrics['default_adc_phase_mse']:.6f}, "
                     f"RMSE={metrics['default_adc_phase_rmse']:.4f}")
        if "benchmark_corr" in metrics:
            self.log(f"  Benchmark corr={metrics['benchmark_corr']:.3f} "
                     f"(dB={metrics['benchmark_corr_dB']:.3f})")
            self.log(f"  Cartesian RA: benchmark cart_corr={metrics['benchmark_cart_corr']:.3f}, "
                     f"PSNR={metrics['benchmark_cart_psnr']:.2f}, "
                     f"SSIM={metrics['benchmark_cart_ssim']:.3f}, "
                     f"MSE={metrics['benchmark_cart_mse']:.6f}, "
                     f"RMSE={metrics['benchmark_cart_rmse']:.4f}")
            self.log(f"  ADC (log mag): benchmark MSE={metrics['benchmark_adc_log_mag_mse']:.6f}, "
                     f"RMSE={metrics['benchmark_adc_log_mag_rmse']:.4f}")
            self.log(f"  ADC (minmax): benchmark MSE={metrics['benchmark_adc_minmax_mag_mse']:.6f}, "
                     f"RMSE={metrics['benchmark_adc_minmax_mag_rmse']:.4f}")
            self.log(f"  ADC (phase):  benchmark MSE={metrics['benchmark_adc_phase_mse']:.6f}, "
                     f"RMSE={metrics['benchmark_adc_phase_rmse']:.4f}")

        self.save_metrics(metrics)
        return metrics

    def generate_figures(self, metrics: dict) -> List[str]:
        """Generate individual RA cartesian PNGs for single-chip transfer.

        Uses single-chip beamforming (3TX×4RX → 8-element virtual array)
        to produce proper RA images for GT, Default, and Trained ADCs.
        """
        saved = []

        adc_rendered_path = os.path.join(self.output_dir, "adc_rendered_single_chip.npy")
        gt_path = self.scene.get_single_chip_gt_adc(self.single_chip_frame) if self.single_chip_frame else None
        adc_default_path = os.path.join(self.output_dir, "adc_default_single_chip.npy")

        if gt_path and os.path.isfile(gt_path):
            try:
                from mmir.data.ra_utils import adc_to_ra_image_single_chip, ra_polar_to_cartesian
                from .utils.visualization import save_ra_cartesian_png

                # Single-chip range resolution (same radar params, different array)
                range_res = self._get_range_resolution()

                # Crop to useful range bins (same as metrics)
                SKIP_VIZ_NEAR = 15
                SKIP_VIZ_FAR = 110

                # GT RA cartesian — raw is (128, 4, 3, 128) complex128
                gt_raw = np.load(gt_path)
                gt_complex = gt_raw[0].transpose(1, 0, 2)  # (3, 4, 128) complex
                gt_adc = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
                ra_gt_polar = adc_to_ra_image_single_chip(gt_adc)[:, SKIP_VIZ_NEAR:SKIP_VIZ_FAR]
                ra_gt_cart = ra_polar_to_cartesian(ra_gt_polar, range_res)
                for scale in ("dB", "linear"):
                    gt_fig = os.path.join(self.output_dir, f"gt_ra_{scale}.png")
                    save_ra_cartesian_png(ra_gt_cart, gt_fig, range_res=range_res,
                                          scale=scale, title=f"GT ({scale})")
                    saved.append(gt_fig)
                    self.log(f"  Saved GT RA {scale} → {gt_fig}")

                # Rendered (trained) RA cartesian
                if os.path.isfile(adc_rendered_path):
                    rend_adc = np.load(adc_rendered_path)
                    ra_rend_polar = adc_to_ra_image_single_chip(rend_adc)[:, SKIP_VIZ_NEAR:SKIP_VIZ_FAR]
                    ra_rend_cart = ra_polar_to_cartesian(ra_rend_polar, range_res)
                    for scale in ("dB", "linear"):
                        rend_fig = os.path.join(self.output_dir, f"rendered_ra_{scale}.png")
                        save_ra_cartesian_png(ra_rend_cart, rend_fig, range_res=range_res,
                                              scale=scale, title=f"Rendered ({scale})")
                        saved.append(rend_fig)
                        self.log(f"  Saved Rendered RA {scale} → {rend_fig}")

                # Default (untrained) RA cartesian
                if os.path.isfile(adc_default_path):
                    def_adc = np.load(adc_default_path)
                    ra_def_polar = adc_to_ra_image_single_chip(def_adc)[:, SKIP_VIZ_NEAR:SKIP_VIZ_FAR]
                    ra_def_cart = ra_polar_to_cartesian(ra_def_polar, range_res)
                    for scale in ("dB", "linear"):
                        def_fig = os.path.join(self.output_dir, f"default_ra_{scale}.png")
                        save_ra_cartesian_png(ra_def_cart, def_fig, range_res=range_res,
                                              scale=scale, title=f"Default ({scale})")
                        saved.append(def_fig)
                        self.log(f"  Saved Default RA {scale} → {def_fig}")

                # Benchmark Sionna RA cartesian
                adc_benchmark_path = os.path.join(self.output_dir, "adc_benchmark_single_chip.npy")
                if os.path.isfile(adc_benchmark_path):
                    bm_adc = np.load(adc_benchmark_path)
                    ra_bm_polar = adc_to_ra_image_single_chip(bm_adc)[:, SKIP_VIZ_NEAR:SKIP_VIZ_FAR]
                    ra_bm_cart = ra_polar_to_cartesian(ra_bm_polar, range_res)
                    for scale in ("dB", "linear"):
                        bm_fig = os.path.join(self.output_dir, f"benchmark_ra_{scale}.png")
                        save_ra_cartesian_png(ra_bm_cart, bm_fig, range_res=range_res,
                                              scale=scale, title=f"Benchmark ({scale})")
                        saved.append(bm_fig)
                        self.log(f"  Saved Benchmark RA {scale} → {bm_fig}")

            except Exception as e:
                self.log(f"  Error generating RA cartesian PNGs: {e}")

        return saved

    def _get_range_resolution(self) -> float:
        """Compute range resolution from single-chip config."""
        sc_config = self._ensure_aligned_sc_config()
        if sc_config is None:
            return 0.0375  # default for 77 GHz, 4 GHz BW
        try:
            from mmir.data.io_utils import compute_range_res_from_cfg
            return compute_range_res_from_cfg(sc_config)
        except Exception:
            return 0.0375


def run_radar_transfer_all(
    scenes: List[SceneInfo],
    output_root: str,
    render: bool = True,
    skip_figures: bool = False,
    skip_latex: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run radar transfer evaluation for all scenes."""
    per_scene = {}

    for scene in scenes:
        evaluator = RadarTransferEvaluator(scene, output_root, verbose=verbose)
        if render:
            evaluator.render()
        metrics = evaluator.run()
        if not skip_figures:
            evaluator.generate_figures(metrics)
        per_scene[scene.name] = metrics

    # Aggregate
    flat_metrics = {}
    for scene_name, m in per_scene.items():
        flat_metrics[scene_name] = {
            "train_corr": m.get("train_corr"),
            "transfer_corr": m.get("transfer_corr"),
            "transfer_psnr": m.get("transfer_psnr"),
            "transfer_ssim": m.get("transfer_ssim"),
            "transfer_cart_corr": m.get("transfer_cart_corr"),
            "transfer_cart_psnr": m.get("transfer_cart_psnr"),
            "transfer_cart_ssim": m.get("transfer_cart_ssim"),
            "transfer_cart_mse": m.get("transfer_cart_mse"),
            "transfer_cart_rmse": m.get("transfer_cart_rmse"),
            "transfer_adc_log_mag_mse": m.get("transfer_adc_log_mag_mse"),
            "transfer_adc_minmax_mag_mse": m.get("transfer_adc_minmax_mag_mse"),
            "transfer_adc_phase_mse": m.get("transfer_adc_phase_mse"),
            "default_corr": m.get("default_corr"),
            "default_psnr": m.get("default_psnr"),
            "default_ssim": m.get("default_ssim"),
            "default_cart_corr": m.get("default_cart_corr"),
            "default_cart_psnr": m.get("default_cart_psnr"),
            "default_cart_ssim": m.get("default_cart_ssim"),
            "default_cart_mse": m.get("default_cart_mse"),
            "default_cart_rmse": m.get("default_cart_rmse"),
            "default_adc_log_mag_mse": m.get("default_adc_log_mag_mse"),
            "default_adc_minmax_mag_mse": m.get("default_adc_minmax_mag_mse"),
            "default_adc_phase_mse": m.get("default_adc_phase_mse"),
            "benchmark_corr": m.get("benchmark_corr"),
            "benchmark_cart_corr": m.get("benchmark_cart_corr"),
            "benchmark_cart_psnr": m.get("benchmark_cart_psnr"),
            "benchmark_cart_ssim": m.get("benchmark_cart_ssim"),
            "benchmark_cart_mse": m.get("benchmark_cart_mse"),
            "benchmark_cart_rmse": m.get("benchmark_cart_rmse"),
            "benchmark_adc_log_mag_mse": m.get("benchmark_adc_log_mag_mse"),
            "benchmark_adc_minmax_mag_mse": m.get("benchmark_adc_minmax_mag_mse"),
            "benchmark_adc_phase_mse": m.get("benchmark_adc_phase_mse"),
            "degradation_pct": m.get("degradation_pct"),
        }

    agg = aggregate_metrics(flat_metrics, [
        "train_corr", "transfer_corr", "transfer_psnr", "transfer_ssim",
        "transfer_cart_corr", "transfer_cart_psnr", "transfer_cart_ssim",
        "transfer_cart_mse", "transfer_cart_rmse",
        "transfer_adc_log_mag_mse", "transfer_adc_minmax_mag_mse",
        "transfer_adc_phase_mse",
        "default_corr", "default_psnr", "default_ssim",
        "default_cart_corr", "default_cart_psnr", "default_cart_ssim",
        "default_cart_mse", "default_cart_rmse",
        "default_adc_log_mag_mse", "default_adc_minmax_mag_mse",
        "default_adc_phase_mse",
        "benchmark_corr", "benchmark_cart_corr", "benchmark_cart_psnr",
        "benchmark_cart_ssim", "benchmark_cart_mse", "benchmark_cart_rmse",
        "benchmark_adc_log_mag_mse", "benchmark_adc_minmax_mag_mse",
        "benchmark_adc_phase_mse",
        "degradation_pct",
    ])

    agg_dir = os.path.join(output_root, "radar_transfer")
    os.makedirs(agg_dir, exist_ok=True)

    import json as _json
    with open(os.path.join(agg_dir, "aggregate_metrics.json"), "w") as f:
        _json.dump({"per_scene": flat_metrics, "aggregate": agg}, f, indent=2)

    if not skip_latex:
        from .utils.latex_export import generate_radar_transfer_table
        generate_radar_transfer_table(
            flat_metrics, agg,
            os.path.join(agg_dir, "table2_radar_transfer.tex"),
        )

    if verbose:
        print(f"\n=== Radar Transfer Aggregate ===")
        for key, stats in agg.items():
            print(f"  {key}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    return per_scene
