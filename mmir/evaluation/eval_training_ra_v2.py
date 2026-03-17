"""Evaluation #1 v2: Training RA — reads metrics from training output when available.

For each scene, produces the same output structure as eval_training_ra.py:
    <scene>/
        gt_ra_dB.png, gt_ra_linear.png          — shared GT figures
        ours/
            rendered_ra_dB.png, rendered_ra_linear.png, error_map.png,
            convergence.png, adc_rendered_cascaded.npy, metrics.json
        benchmark/
            rendered_ra_dB.png, rendered_ra_linear.png, error_map.png,
            convergence.png, ra_rendered_cart.npy, metrics.json

Key difference from v1:
    - Reads best_metrics.json from training dirs when available (no re-rendering needed)
    - Copies ra_rendered_cart.npy and .png files from training dirs
    - Falls back to v1 behavior (re-render + recompute) if best_metrics.json missing

All metrics (Correlation, PSNR, SSIM, MSE, RMSE) are computed on RA
Cartesian images using consistent independent min-max normalization.
"""

import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from .base_evaluator import BaseEvaluator, aggregate_metrics
from .material_loader import (
    load_benchmark_training_history,
    load_our_training_history,
)
from .scene_registry import SceneInfo

from .utils.metrics import compute_cart_ra_metrics as compute_ra_metrics


# ---------------------------------------------------------------------------
# VA Phase Coherence (standalone function)
# ---------------------------------------------------------------------------

def _compute_va_phase_coherence(
    rendered_complex: np.ndarray, gt_complex: np.ndarray, top_percentile: float = 0.9
) -> float:
    """Virtual Array Phase Coherence metric.

    Per-range-bin complex correlation of unit-normalized VA vectors.
    Range [0, 1]: 1 = perfect relative phase match, ~1/N_vx = random.
    """
    NT, NR, K = rendered_complex.shape
    N_vx = NT * NR
    eps = 1e-10

    r = rendered_complex.reshape(N_vx, K)
    g = gt_complex.reshape(N_vx, K)

    gt_energy = np.sum(np.abs(g) ** 2, axis=0)
    thresh = np.percentile(gt_energy, top_percentile * 100)
    valid = gt_energy >= thresh

    if np.sum(valid) < 2:
        return float('nan')

    r_hat = r / (np.abs(r) + eps)
    g_hat = g / (np.abs(g) + eps)

    dot = np.sum(r_hat[:, valid] * np.conj(g_hat[:, valid]), axis=0)
    rho = np.abs(dot) / N_vx

    return float(np.mean(rho))


# ---------------------------------------------------------------------------
# Evaluator v2
# ---------------------------------------------------------------------------

class TrainingRAEvaluator(BaseEvaluator):
    """Evaluate training RA results, preferring pre-computed metrics from training."""

    @property
    def eval_name(self) -> str:
        return "training_ra"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Compute RA Cartesian images and metrics for both methods.

        Prefers reading best_metrics.json from training dirs. Falls back
        to re-rendering (v1 behavior) if not available.

        Returns dict with structure:
            {scene, ours: {cart_corr, mse, rmse, psnr, ssim, ...},
             benchmark: {cart_corr, mse, rmse, psnr, ssim, ...}}
        """
        ours_dir = os.path.join(self.output_dir, "ours")
        bench_dir = os.path.join(self.output_dir, "benchmark")
        os.makedirs(ours_dir, exist_ok=True)
        os.makedirs(bench_dir, exist_ok=True)

        # --- GT RA Cartesian ---
        ra_gt_cart, range_res = self._get_gt_ra_cartesian()
        if ra_gt_cart is None:
            self.log("No GT RA available — skipping")
            return {"scene": self.scene.name, "ours": {}, "benchmark": {}}

        # --- Ours ---
        ours_metrics = self._eval_ours(ra_gt_cart, range_res, ours_dir)

        # --- Benchmark ---
        bench_metrics = self._eval_benchmark(ra_gt_cart, range_res, bench_dir)

        metrics = {
            "scene": self.scene.name,
            "ours": ours_metrics,
            "benchmark": bench_metrics,
        }
        return metrics

    # ------------------------------------------------------------------
    # ADC metrics helper
    # ------------------------------------------------------------------

    _ADC_METRIC_KEYS = (
        "adc_log_mag_mse", "adc_log_mag_rmse",
        "adc_minmax_mag_mse", "adc_minmax_mag_rmse",
        "adc_phase_mse", "adc_phase_rmse",
        "va_phase_coherence",
    )

    def _read_adc_metrics(
        self, training_dir: str, best_metrics: Optional[dict] = None,
        is_benchmark: bool = False,
    ) -> dict:
        """Read ADC metrics from training output.

        Checks best_metrics.json first, then falls back to reading
        training_history.json at the best iteration index.

        Our training history uses 'adc_relphase_*' keys (mapped to 'adc_phase_*').
        Benchmark training history uses 'adc_phase_*' directly.
        """
        adc = {}

        # 1) Try best_metrics.json (benchmark saves ADC metrics there directly)
        if best_metrics:
            if "adc_log_mag_mse" in best_metrics:
                for k in self._ADC_METRIC_KEYS:
                    adc[k] = best_metrics.get(k)
                return adc

        # 2) Fall back to training_history.json at best_iteration
        if not training_dir:
            return adc

        history_path = os.path.join(training_dir, "training_history.json")
        if not os.path.isfile(history_path):
            return adc

        try:
            with open(history_path) as f:
                raw = json.load(f)

            # Determine best iteration index
            if is_benchmark:
                best_iter = raw.get("best_iter")
                h = raw.get("history", {})
            else:
                h = raw
                best_iter = None
                # For ours, find best from best_metrics or max cart_corr
                bm_path = os.path.join(training_dir, "best_metrics.json")
                if os.path.isfile(bm_path):
                    with open(bm_path) as f2:
                        best_iter = json.load(f2).get("best_iteration")
                if best_iter is None:
                    cart_corr = h.get("cart_corr", [])
                    valid = [(i, c) for i, c in enumerate(cart_corr) if c is not None]
                    if valid:
                        best_iter = h.get("iteration", list(range(len(cart_corr))))[
                            max(valid, key=lambda x: x[1])[0]
                        ]

            iterations = h.get("iteration", [])
            if best_iter is None or best_iter not in iterations:
                return adc
            idx = iterations.index(best_iter)

            # Read ADC metrics at best iteration
            # Our training uses 'adc_relphase_*'; benchmark uses 'adc_phase_*'
            key_map = {
                "adc_log_mag_mse": "adc_log_mag_mse",
                "adc_log_mag_rmse": "adc_log_mag_rmse",
                "adc_minmax_mag_mse": "adc_minmax_mag_mse",
                "adc_minmax_mag_rmse": "adc_minmax_mag_rmse",
                "adc_phase_mse": "adc_relphase_mse" if not is_benchmark else "adc_phase_mse",
                "adc_phase_rmse": "adc_relphase_rmse" if not is_benchmark else "adc_phase_rmse",
                "va_phase_coherence": "va_phase_coherence",
            }
            for out_key, hist_key in key_map.items():
                vals = h.get(hist_key, [])
                if idx < len(vals):
                    adc[out_key] = vals[idx]

        except Exception as e:
            self.log(f"  Could not read ADC metrics from history: {e}")

        return adc

    # ------------------------------------------------------------------
    # Ours evaluation
    # ------------------------------------------------------------------

    def _eval_ours(self, ra_gt_cart: np.ndarray, range_res: float, ours_dir: str) -> dict:
        """Evaluate our method: try best_metrics.json first, else fall back to re-render."""
        self.log("Evaluating ours...")
        ours_metrics = {}

        # Try reading pre-computed metrics from training
        if self.scene.our_training_dir:
            best_metrics_path = os.path.join(self.scene.our_training_dir, "best_metrics.json")
            if os.path.isfile(best_metrics_path):
                self.log(f"  Reading pre-computed metrics from {best_metrics_path}")
                with open(best_metrics_path) as f:
                    best = json.load(f)

                ours_metrics = {
                    "cart_corr": best.get("cart_corr"),
                    "psnr": best.get("cart_psnr"),
                    "ssim": best.get("cart_ssim"),
                    "mse": best.get("cart_mse"),
                    "rmse": best.get("cart_rmse"),
                    "best_iteration": best.get("best_iteration"),
                    "polar_corr": best.get("polar_corr"),
                }

                # ADC metrics (from best_metrics.json or training_history.json)
                adc = self._read_adc_metrics(
                    self.scene.our_training_dir, best_metrics=best,
                    is_benchmark=False)
                ours_metrics.update(adc)

                self.log(f"  corr={ours_metrics.get('cart_corr', 'N/A'):.4f} "
                         f"MSE={ours_metrics.get('mse', 'N/A'):.6f} "
                         f"RMSE={ours_metrics.get('rmse', 'N/A'):.4f} "
                         f"PSNR={ours_metrics.get('psnr', 'N/A'):.2f} "
                         f"SSIM={ours_metrics.get('ssim', 'N/A'):.4f}")
                if adc:
                    self.log(f"  ADC: log_mag_mse={adc.get('adc_log_mag_mse', 'N/A'):.4f} "
                             f"minmax_mag_mse={adc.get('adc_minmax_mag_mse', 'N/A'):.6f} "
                             f"phase_mse={adc.get('adc_phase_mse', 'N/A'):.4f} "
                             f"va_phase_coh={adc.get('va_phase_coherence', 'N/A')}")

                # Compute va_phase_coherence if missing (old training runs)
                if ours_metrics.get("va_phase_coherence") is None:
                    va_coh = self._compute_va_phase_coherence_fallback(ours_dir)
                    if va_coh is not None:
                        ours_metrics["va_phase_coherence"] = va_coh
                        self.log(f"  Computed va_phase_coherence={va_coh:.4f} (re-rendered)")

                # Copy ra_rendered_cart.npy from training dir
                src_npy = os.path.join(self.scene.our_training_dir, "ra_rendered_cart.npy")
                if os.path.isfile(src_npy):
                    dst_npy = os.path.join(ours_dir, "ra_rendered_cart.npy")
                    if not os.path.isfile(dst_npy):
                        shutil.copy2(src_npy, dst_npy)
                    self.log(f"  Copied ra_rendered_cart.npy")

                # Copy rendered .png files from training dir
                for png_name in ("rendered_ra_dB.png", "rendered_ra_linear.png"):
                    src_png = os.path.join(self.scene.our_training_dir, png_name)
                    if os.path.isfile(src_png):
                        dst_png = os.path.join(ours_dir, png_name)
                        if not os.path.isfile(dst_png):
                            shutil.copy2(src_png, dst_png)

                # Save per-method metrics
                ours_full = {"scene": self.scene.name, "method": "ours", **ours_metrics}
                self._save_json(os.path.join(ours_dir, "metrics.json"), ours_full)
                return ours_metrics

        # Fallback: re-render like v1
        self.log("  No best_metrics.json found — falling back to re-render (v1)")
        rendered_adc = self._get_rendered_cascaded_adc(ours_dir)
        if rendered_adc is not None:
            from mmir.data.ra_utils import adc_to_ra_image_numpy, ra_polar_to_cartesian

            ra_rend_polar = adc_to_ra_image_numpy(rendered_adc)
            ra_rend_cart = ra_polar_to_cartesian(ra_rend_polar, range_res)

            ours_metrics = compute_ra_metrics(ra_gt_cart, ra_rend_cart)
            self.log(f"  corr={ours_metrics['cart_corr']:.4f} "
                     f"MSE={ours_metrics['mse']:.6f} RMSE={ours_metrics['rmse']:.4f} "
                     f"PSNR={ours_metrics['psnr']:.2f} SSIM={ours_metrics['ssim']:.4f}")

            # Training metadata
            if self.scene.our_training_dir:
                try:
                    history = load_our_training_history(self.scene.our_training_dir)
                    cart_corr_hist = history.get("cart_corr", [])
                    iterations = history.get("iteration", [])
                    polar_corr_hist = history.get("polar_corr", [])
                    if cart_corr_hist:
                        valid = [(i, c) for i, c in enumerate(cart_corr_hist) if c is not None]
                        if valid:
                            best_idx = max(valid, key=lambda x: x[1])[0]
                            ours_metrics["best_iteration"] = iterations[best_idx] if iterations else best_idx
                            if polar_corr_hist and best_idx < len(polar_corr_hist):
                                ours_metrics["polar_corr"] = polar_corr_hist[best_idx]
                except Exception as e:
                    self.log(f"  Could not load training history: {e}")
        else:
            self.log("  No rendered ADC available")

        # Save per-method metrics
        ours_full = {"scene": self.scene.name, "method": "ours", **ours_metrics}
        self._save_json(os.path.join(ours_dir, "metrics.json"), ours_full)
        return ours_metrics

    # ------------------------------------------------------------------
    # Benchmark evaluation
    # ------------------------------------------------------------------

    def _eval_benchmark(self, ra_gt_cart: np.ndarray, range_res: float, bench_dir: str) -> dict:
        """Evaluate benchmark: try best_metrics.json first, else fall back to v1."""
        self.log("Evaluating benchmark...")
        bench_metrics = {}

        # Try reading pre-computed metrics from training
        if self.scene.benchmark_training_dir:
            best_metrics_path = os.path.join(
                self.scene.benchmark_training_dir, "best_metrics.json")
            if os.path.isfile(best_metrics_path):
                self.log(f"  Reading pre-computed metrics from {best_metrics_path}")
                with open(best_metrics_path) as f:
                    best = json.load(f)

                bench_metrics = {
                    "cart_corr": best.get("cart_corr"),
                    "psnr": best.get("cart_psnr"),
                    "ssim": best.get("cart_ssim"),
                    "mse": best.get("cart_mse"),
                    "rmse": best.get("cart_rmse"),
                    "best_iteration": best.get("best_iteration"),
                    "polar_corr": best.get("polar_corr"),
                }

                # ADC metrics (from best_metrics.json or training_history.json)
                adc = self._read_adc_metrics(
                    self.scene.benchmark_training_dir, best_metrics=best,
                    is_benchmark=True)
                bench_metrics.update(adc)

                self.log(f"  corr={bench_metrics.get('cart_corr', 'N/A'):.4f} "
                         f"MSE={bench_metrics.get('mse', 'N/A'):.6f} "
                         f"RMSE={bench_metrics.get('rmse', 'N/A'):.4f} "
                         f"PSNR={bench_metrics.get('psnr', 'N/A'):.2f} "
                         f"SSIM={bench_metrics.get('ssim', 'N/A'):.4f}")
                if adc:
                    self.log(f"  ADC: log_mag_mse={adc.get('adc_log_mag_mse', 'N/A'):.4f} "
                             f"minmax_mag_mse={adc.get('adc_minmax_mag_mse', 'N/A'):.6f} "
                             f"phase_mse={adc.get('adc_phase_mse', 'N/A'):.4f} "
                             f"va_phase_coh={adc.get('va_phase_coherence', 'N/A')}")

                # Copy ra_rendered_cart.npy from training dir
                src_npy = os.path.join(
                    self.scene.benchmark_training_dir, "ra_rendered_cart.npy")
                if os.path.isfile(src_npy):
                    dst_npy = os.path.join(bench_dir, "ra_rendered_cart.npy")
                    if not os.path.isfile(dst_npy):
                        shutil.copy2(src_npy, dst_npy)
                    self.log(f"  Copied ra_rendered_cart.npy")

                # Copy rendered .png files from training dir
                for png_name in ("rendered_ra_dB.png", "rendered_ra_linear.png"):
                    src_png = os.path.join(
                        self.scene.benchmark_training_dir, png_name)
                    if os.path.isfile(src_png):
                        dst_png = os.path.join(bench_dir, png_name)
                        if not os.path.isfile(dst_png):
                            shutil.copy2(src_png, dst_png)

                bench_full = {"scene": self.scene.name, "method": "benchmark", **bench_metrics}
                self._save_json(os.path.join(bench_dir, "metrics.json"), bench_full)
                return bench_metrics

        # Fallback: load pre-computed RA cart and recompute metrics (v1 behavior)
        self.log("  No best_metrics.json found — falling back to v1 recompute")
        ra_bench_cart = self._get_benchmark_ra_cart(bench_dir)
        if ra_bench_cart is not None:
            bench_metrics = compute_ra_metrics(ra_gt_cart, ra_bench_cart)
            self.log(f"  corr={bench_metrics['cart_corr']:.4f} "
                     f"MSE={bench_metrics['mse']:.6f} RMSE={bench_metrics['rmse']:.4f} "
                     f"PSNR={bench_metrics['psnr']:.2f} SSIM={bench_metrics['ssim']:.4f}")

            # Training metadata from history
            if self.scene.benchmark_training_dir:
                try:
                    bench_history = load_benchmark_training_history(
                        self.scene.benchmark_training_dir)
                    bench_metrics["best_iteration"] = bench_history.get("best_train_iter")
                    bench_metrics["init_cart_corr"] = bench_history.get("init_cart_corr")
                    bench_metrics["init_polar_corr"] = bench_history.get("init_polar_corr")
                    bench_metrics["polar_corr"] = bench_history.get("final_polar_corr")
                except Exception as e:
                    self.log(f"  Could not load benchmark history: {e}")
        else:
            self.log("  No benchmark RA cart available")

        bench_full = {"scene": self.scene.name, "method": "benchmark", **bench_metrics}
        self._save_json(os.path.join(bench_dir, "metrics.json"), bench_full)
        return bench_metrics

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def generate_figures(self, metrics: dict) -> List[str]:
        """Generate all figures for both methods.

        GT figures at scene level; per-method figures in ours/ and benchmark/.
        Copies .png from training dir when available, generates otherwise.
        """
        saved = []
        ra_gt_cart, range_res = self._get_gt_ra_cartesian()
        if ra_gt_cart is None:
            return saved

        from .utils.visualization import save_ra_cartesian_png

        # GT figures at scene level — try copying from training dir first
        for scale in ("dB", "linear"):
            path = os.path.join(self.output_dir, f"gt_ra_{scale}.png")
            src_gt = None
            if self.scene.our_training_dir:
                src_gt = os.path.join(self.scene.our_training_dir, f"gt_ra_{scale}.png")
            if src_gt and os.path.isfile(src_gt):
                shutil.copy2(src_gt, path)
            else:
                save_ra_cartesian_png(ra_gt_cart, path, range_res=range_res,
                                      scale=scale, title=f"GT ({scale})")
            saved.append(path)
            self.log(f"Saved {path}")

        # Ours figures
        ours_dir = os.path.join(self.output_dir, "ours")
        ours_figs = self._generate_method_figures(
            ours_dir, ra_gt_cart, range_res,
            metrics.get("ours", {}),
            training_dir=self.scene.our_training_dir,
            adc_key="adc_rendered_cascaded.npy",
            ra_cart_key="ra_rendered_cart.npy",
        )
        saved.extend(ours_figs)

        # Ours convergence
        if self.scene.our_training_dir:
            conv = self._generate_convergence(
                self.scene.our_training_dir, ours_dir, is_benchmark=False)
            if conv:
                saved.append(conv)

        # Benchmark figures
        bench_dir = os.path.join(self.output_dir, "benchmark")
        bench_figs = self._generate_method_figures(
            bench_dir, ra_gt_cart, range_res,
            metrics.get("benchmark", {}),
            training_dir=self.scene.benchmark_training_dir,
            adc_key="ra_rendered_cart.npy",
            ra_cart_key="ra_rendered_cart.npy",
        )
        saved.extend(bench_figs)

        # Benchmark convergence
        if self.scene.benchmark_training_dir:
            conv = self._generate_convergence(
                self.scene.benchmark_training_dir, bench_dir, is_benchmark=True)
            if conv:
                saved.append(conv)

        return saved

    # ------------------------------------------------------------------
    # Data loading helpers (same as v1)
    # ------------------------------------------------------------------

    def _get_gt_ra_cartesian(self) -> tuple:
        """Load GT cascaded ADC → RA Cartesian. Returns (ra_cart, range_res) or (None, None)."""
        # Try loading from training dir first
        if self.scene.our_training_dir:
            gt_npy = os.path.join(self.scene.our_training_dir, "ra_gt_cart.npy")
            if os.path.isfile(gt_npy):
                from mmir.data.io_utils import compute_range_res_from_cfg
                range_res = compute_range_res_from_cfg(self.scene.cascaded_config)
                self.log(f"  Loaded GT RA cart from training dir: {gt_npy}")
                return np.load(gt_npy), range_res

        if not self.scene.gt_cascaded_adc or not os.path.isfile(self.scene.gt_cascaded_adc):
            return None, None

        from mmir.data.ra_utils import adc_to_ra_image_numpy, ra_polar_to_cartesian
        from mmir.data.io_utils import compute_range_res_from_cfg

        range_res = compute_range_res_from_cfg(self.scene.cascaded_config)
        gt_raw = np.load(self.scene.gt_cascaded_adc)
        gt_complex = gt_raw[0].transpose(1, 0, 2)
        gt_adc = np.stack([gt_complex.real, gt_complex.imag], axis=-1).astype(np.float32)
        ra_polar = adc_to_ra_image_numpy(gt_adc)
        ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
        return ra_cart, range_res

    def _get_rendered_cascaded_adc(self, ours_dir: str) -> Optional[np.ndarray]:
        """Get rendered cascaded ADC: check cache in ours_dir, else re-render."""
        if not self.scene.our_training_dir:
            return None

        cached_path = os.path.join(ours_dir, "adc_rendered_cascaded.npy")
        if os.path.isfile(cached_path):
            self.log(f"  Using cached ADC: {cached_path}")
            return np.load(cached_path)

        try:
            from .renderer_wrapper import RendererWrapper

            wrapper = RendererWrapper.from_training_dir(
                mesh_file=self.scene.mesh_file,
                config_file=self.scene.cascaded_config,
                training_dir=self.scene.our_training_dir,
                verbose=False,
            )
            wrapper.load_materials(self.scene.our_training_dir)
            adc = wrapper.render_forward(seed=42)
            np.save(cached_path, adc)
            self.log(f"  Rendered cascaded ADC → {cached_path}")
            wrapper.cleanup()
            return adc
        except Exception as e:
            self.log(f"  Could not render cascaded ADC: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_benchmark_ra_cart(self, bench_dir: str) -> Optional[np.ndarray]:
        """Load benchmark pre-computed RA Cartesian, copy to bench_dir."""
        if not self.scene.benchmark_training_dir:
            return None

        src_path = os.path.join(self.scene.benchmark_training_dir, "ra_rendered_cart.npy")
        if not os.path.isfile(src_path):
            self.log(f"  Benchmark RA cart not found: {src_path}")
            return None

        # Copy to output dir for reproducibility
        dst_path = os.path.join(bench_dir, "ra_rendered_cart.npy")
        if not os.path.isfile(dst_path):
            shutil.copy2(src_path, dst_path)

        return np.load(src_path)

    # ------------------------------------------------------------------
    # Figure helpers
    # ------------------------------------------------------------------

    def _generate_method_figures(
        self,
        method_dir: str,
        ra_gt_cart: np.ndarray,
        range_res: float,
        method_metrics: dict,
        training_dir: Optional[str] = None,
        adc_key: str = "adc_rendered_cascaded.npy",
        ra_cart_key: str = "ra_rendered_cart.npy",
    ) -> List[str]:
        """Generate rendered RA PNGs + error map for one method.

        Tries to copy .png from training_dir first, else regenerates.
        """
        saved = []
        from .utils.visualization import save_ra_cartesian_png, save_ra_error_map

        corr = method_metrics.get("cart_corr")
        corr_str = f", corr={corr:.3f}" if corr is not None else ""

        # Try to copy rendered .png from training dir
        png_copied = False
        if training_dir:
            for scale in ("dB", "linear"):
                src_png = os.path.join(training_dir, f"rendered_ra_{scale}.png")
                dst_png = os.path.join(method_dir, f"rendered_ra_{scale}.png")
                if os.path.isfile(src_png):
                    shutil.copy2(src_png, dst_png)
                    saved.append(dst_png)
                    self.log(f"  Copied {dst_png}")
                    png_copied = True

        # If not copied, generate from .npy
        if not png_copied:
            # Try ra_rendered_cart.npy first, then adc_key
            ra_rend_cart = None
            ra_cart_path = os.path.join(method_dir, ra_cart_key)
            if os.path.isfile(ra_cart_path):
                ra_rend_cart = np.load(ra_cart_path)
            else:
                npy_path = os.path.join(method_dir, adc_key)
                if os.path.isfile(npy_path):
                    if adc_key == "ra_rendered_cart.npy":
                        ra_rend_cart = np.load(npy_path)
                    else:
                        from mmir.data.ra_utils import adc_to_ra_image_numpy, ra_polar_to_cartesian
                        adc = np.load(npy_path)
                        ra_polar = adc_to_ra_image_numpy(adc)
                        ra_rend_cart = ra_polar_to_cartesian(ra_polar, range_res)

            if ra_rend_cart is not None:
                for scale in ("dB", "linear"):
                    path = os.path.join(method_dir, f"rendered_ra_{scale}.png")
                    save_ra_cartesian_png(ra_rend_cart, path, range_res=range_res,
                                          scale=scale, title=f"Rendered ({scale}{corr_str})")
                    saved.append(path)
                    self.log(f"  Saved {path}")

        # Error map — always regenerate from data
        ra_rend_cart = None
        ra_cart_path = os.path.join(method_dir, ra_cart_key)
        if os.path.isfile(ra_cart_path):
            ra_rend_cart = np.load(ra_cart_path)
        elif os.path.isfile(os.path.join(method_dir, adc_key)):
            npy_path = os.path.join(method_dir, adc_key)
            if adc_key == "ra_rendered_cart.npy":
                ra_rend_cart = np.load(npy_path)
            else:
                from mmir.data.ra_utils import adc_to_ra_image_numpy, ra_polar_to_cartesian
                adc = np.load(npy_path)
                ra_polar = adc_to_ra_image_numpy(adc)
                ra_rend_cart = ra_polar_to_cartesian(ra_polar, range_res)

        if ra_rend_cart is not None:
            err_path = os.path.join(method_dir, "error_map.png")
            save_ra_error_map(ra_gt_cart, ra_rend_cart, err_path,
                              range_res=range_res, title="Error (red=too bright)")
            saved.append(err_path)
            self.log(f"  Saved {err_path}")

        return saved

    def _generate_convergence(
        self, training_dir: str, output_dir: str, is_benchmark: bool = False
    ) -> Optional[str]:
        """Generate convergence plot from training history."""
        from .utils.visualization import save_convergence_figure

        history_path = os.path.join(training_dir, "training_history.json")
        if not os.path.isfile(history_path):
            return None

        with open(history_path) as f:
            history = json.load(f)

        if is_benchmark:
            h = history.get("history", {})
            iterations = h.get("iteration", [])
            loss_vals = h.get("loss", [])
            corr_vals = h.get("cart_corr", [])
        else:
            iterations = history.get("iteration", [])
            loss_vals = history.get("total_loss", [])
            corr_vals = history.get("cart_corr", [])

        if not iterations or (not loss_vals and not corr_vals):
            return None

        series = {}
        if loss_vals:
            series["Loss"] = loss_vals
        if corr_vals:
            series["Correlation"] = corr_vals

        conv_path = os.path.join(output_dir, "convergence.png")
        save_convergence_figure(iterations, series, conv_path)
        self.log(f"  Saved convergence → {conv_path}")
        return conv_path

    # ------------------------------------------------------------------
    # VA Phase Coherence fallback (for old training runs without this metric)
    # ------------------------------------------------------------------

    def _compute_va_phase_coherence_fallback(self, method_dir: str) -> Optional[float]:
        """Re-render ADC and compute VA phase coherence when not in training history."""
        if not self.scene.our_training_dir:
            return None

        # Check for cached ADC first
        cached_path = os.path.join(method_dir, "adc_rendered_cascaded.npy")
        rendered_adc = None
        if os.path.isfile(cached_path):
            rendered_adc = np.load(cached_path)
        else:
            # Re-render
            try:
                from .renderer_wrapper import RendererWrapper
                self.log("  Re-rendering ADC for va_phase_coherence...")
                wrapper = RendererWrapper.from_training_dir(
                    mesh_file=self.scene.mesh_file,
                    config_file=self.scene.cascaded_config,
                    training_dir=self.scene.our_training_dir,
                    verbose=False,
                )
                wrapper.load_materials(self.scene.our_training_dir)
                rendered_adc = wrapper.render_forward(seed=42)
                np.save(cached_path, rendered_adc)
                self.log(f"  Rendered ADC → {cached_path}")
                wrapper.cleanup()
            except Exception as e:
                self.log(f"  Could not render ADC for va_phase_coherence: {e}")
                return None

        if rendered_adc is None:
            return None

        # Load GT ADC
        if not self.scene.gt_cascaded_adc or not os.path.isfile(self.scene.gt_cascaded_adc):
            return None

        gt_raw = np.load(self.scene.gt_cascaded_adc)
        gt_complex = gt_raw[0].transpose(1, 0, 2)  # [NT, NR, K]

        # Convert rendered ADC [NT, NR, K, 2] to complex
        rendered_complex = rendered_adc[..., 0] + 1j * rendered_adc[..., 1]

        return _compute_va_phase_coherence(rendered_complex, gt_complex)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _save_json(self, path: str, data: dict):
        """Save dict as JSON with numpy support."""
        from .base_evaluator import _json_default
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        self.log(f"  Saved {path}")

    def generate_latex_row(self, metrics: dict) -> str:
        """Generate a LaTeX table row for this scene."""
        ours = metrics.get("ours", {})
        bench = metrics.get("benchmark", {})
        from .utils.latex_export import bold_best

        ours_corr_s, bench_corr_s = bold_best(
            [ours.get("cart_corr"), bench.get("cart_corr")], ".3f", higher_better=True
        )
        short = self.scene.name.replace("seq_", "S").replace("_frame_", "F")
        return f"  {short} & {ours_corr_s} & {bench_corr_s} \\\\"


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_training_ra_all(
    scenes: List[SceneInfo],
    output_root: str,
    skip_figures: bool = False,
    skip_latex: bool = False,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run Training RA evaluation for all scenes and generate aggregate outputs."""
    per_scene = {}

    for scene in scenes:
        evaluator = TrainingRAEvaluator(scene, output_root, verbose=verbose)
        metrics = evaluator.run()
        if not skip_figures:
            evaluator.generate_figures(metrics)
        per_scene[scene.name] = metrics

    # Build flat metrics for aggregation
    flat_metrics = {}
    for scene_name, m in per_scene.items():
        ours = m.get("ours", {})
        bench = m.get("benchmark", {})
        flat_metrics[scene_name] = {
            "ours_corr": ours.get("cart_corr"),
            "ours_psnr": ours.get("psnr"),
            "ours_ssim": ours.get("ssim"),
            "ours_mse": ours.get("mse"),
            "ours_rmse": ours.get("rmse"),
            "ours_adc_log_mag_mse": ours.get("adc_log_mag_mse"),
            "ours_adc_minmax_mag_mse": ours.get("adc_minmax_mag_mse"),
            "ours_adc_phase_mse": ours.get("adc_phase_mse"),
            "ours_va_phase_coherence": ours.get("va_phase_coherence"),
            "bench_corr": bench.get("cart_corr"),
            "bench_psnr": bench.get("psnr"),
            "bench_ssim": bench.get("ssim"),
            "bench_mse": bench.get("mse"),
            "bench_rmse": bench.get("rmse"),
            "bench_adc_log_mag_mse": bench.get("adc_log_mag_mse"),
            "bench_adc_minmax_mag_mse": bench.get("adc_minmax_mag_mse"),
            "bench_adc_phase_mse": bench.get("adc_phase_mse"),
            "bench_va_phase_coherence": bench.get("va_phase_coherence"),
        }

    agg_keys = [
        "ours_corr", "ours_psnr", "ours_ssim", "ours_mse", "ours_rmse",
        "ours_adc_log_mag_mse", "ours_adc_minmax_mag_mse", "ours_adc_phase_mse",
        "ours_va_phase_coherence",
        "bench_corr", "bench_psnr", "bench_ssim", "bench_mse", "bench_rmse",
        "bench_adc_log_mag_mse", "bench_adc_minmax_mag_mse", "bench_adc_phase_mse",
        "bench_va_phase_coherence",
    ]
    agg = aggregate_metrics(flat_metrics, agg_keys)

    # Save aggregate
    agg_dir = os.path.join(output_root, "training_ra")
    os.makedirs(agg_dir, exist_ok=True)

    with open(os.path.join(agg_dir, "aggregate_metrics.json"), "w") as f:
        json.dump({"per_scene": flat_metrics, "aggregate": agg}, f, indent=2)

    # LaTeX table
    if not skip_latex:
        from .utils.latex_export import generate_training_ra_table_v2
        generate_training_ra_table_v2(
            flat_metrics, agg,
            os.path.join(agg_dir, "table1_training_ra.tex"),
        )

    if verbose:
        print(f"\n=== Training RA Aggregate ===")
        for key, stats in agg.items():
            print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return per_scene
