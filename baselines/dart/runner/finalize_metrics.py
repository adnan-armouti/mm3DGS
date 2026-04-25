"""Phase-B finalization for DART — runs in mmir env.

Reads rendered_ra_polar.npy + train_meta.json, builds the canonical
cascade GT (matches RadarSplat / RadarFields / v6/v7), runs the
mmir polar→cart pipeline + compute_cart_ra_metrics, writes metrics.json
+ the four RA inspection PNGs.

DART now trains and evaluates on cascade data (PLAN Section 4 revision):
  * Rendered: cascade-shaped (Na=8, Nr=256) polar from DART, crop bins
    15..110 → (8, 95), then ra_polar_to_cartesian.
  * GT: cascade ADC → adc_to_ra_complex+abs → (127, 256) sin-space polar
    → ra_polar_to_cartesian on the FULL polar (matches v6/v7 reference).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters as common_adapt  # noqa: E402
from baselines.common import eval as common_eval  # noqa: E402
from baselines.common import nvs_split  # noqa: E402


def _gt_side(scene: str, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """GT for the chosen DART mode.

    Cascade mode: cascade ADC → adc_to_ra_complex+abs → (127, 256) sin-space
        polar → ra_polar_to_cartesian on FULL polar (matches v6/v7 reference,
        identical to RadarSplat/RadarFields finalize).
    SC mode: SC ADC → adc_to_ra_image_single_chip → (63, 128) polar →
        cartesian. Documented as NOT comparable to cascade-trained baselines.
    """
    if mode == "cascaded":
        split = nvs_split.cascaded_split(scene)
        adc = common_adapt.load_cascaded_adc(split["test_file"])
        ra_polar_full = common_adapt.adc_to_polar_ra(adc, sensor="cascaded")
    else:
        split = nvs_split.single_chip_split(scene)
        adc = common_adapt.load_single_chip_adc(split["test_file"])
        ra_polar_full = common_adapt.adc_to_polar_ra(adc, sensor="single_chip")
    ra_polar_cropped = common_adapt.range_crop(ra_polar_full)

    from mmir.data.io_utils import compute_range_res_from_cfg
    from mmir.data.ra_utils import ra_polar_to_cartesian
    range_res = compute_range_res_from_cfg(split["test_config"])
    ra_cart = ra_polar_to_cartesian(ra_polar_full, range_res).astype(np.float32)
    return (ra_polar_full.astype(np.float32),
            ra_polar_cropped.astype(np.float32),
            ra_cart,
            range_res)


def _match_shapes(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    def crop(x):
        dh, dw = (x.shape[0] - H) // 2, (x.shape[1] - W) // 2
        return x[dh:dh + H, dw:dw + W]
    return crop(a), crop(b)


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline-name", default="dart")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.out_dir, "train_meta.json")))
    rendered_polar = np.load(
        os.path.join(args.out_dir, "rendered_ra_polar.npy")
    ).astype(np.float32)
    rendered_polar = np.clip(rendered_polar, 0.0, None)        # non-negative

    mode = meta.get("mode", "cascaded")
    gt_polar_full, gt_polar_cropped, gt_cart, range_res = _gt_side(meta["scene"], mode)

    # Rendered cascade-shaped polar is (Na=8, Nr=256). Crop bins 15..110
    # (drop TX-RX coupling + DFT wrap-around) and run cart conversion.
    rend_polar_cropped = common_adapt.range_crop(rendered_polar)   # (8, 95)

    from mmir.data.ra_utils import ra_polar_to_cartesian
    rend_cart = ra_polar_to_cartesian(rend_polar_cropped, range_res).astype(np.float32)

    gt_cart_m, rend_cart_m = _match_shapes(gt_cart, rend_cart)
    W = min(gt_polar_cropped.shape[1], rend_polar_cropped.shape[1])
    gt_polar_m = gt_polar_cropped[:, :W]
    rend_polar_m = rend_polar_cropped[:, :W]

    extra = {
        "test_frame": int(meta["test_frame"]),
        "train_frames": list(map(int, meta["train_frames"])),
        "wall_time_seconds": float(meta["wall_time_seconds"]),
        "peak_gpu_mem_mib": float(meta.get("peak_gpu_mem_mib", 0)),
        "deviations_from_reference": meta.get("deviations_from_reference", []),
        "upstream_commit": meta.get("upstream_commit", "unknown"),
        "out_dir": meta.get("out_dir", None),
        "epochs": int(meta.get("epochs", 0)),
        "sensor": meta.get("sensor", mode),
        "mode": mode,
    }
    result = common_eval.run_eval(
        args.baseline_name, meta["scene"],
        rendered_ra_cart=rend_cart_m, gt_ra_cart=gt_cart_m,
        rendered_ra_polar_cropped=rend_polar_m,
        gt_ra_polar_cropped=gt_polar_m,
        extra=extra,
    )

    np.save(os.path.join(args.out_dir, "gt_ra_polar_full.npy"), gt_polar_full)
    np.save(os.path.join(args.out_dir, "gt_ra_polar_cropped.npy"), gt_polar_cropped)
    np.save(os.path.join(args.out_dir, "gt_ra_cart.npy"), gt_cart)
    np.save(os.path.join(args.out_dir, "rendered_ra_cart.npy"), rend_cart)
    common_eval.write_metrics_json(os.path.join(args.out_dir, "metrics.json"), result)

    from mmir.evaluation.utils.visualization import save_ra_cartesian_png
    for scale in ("linear", "dB"):
        save_ra_cartesian_png(
            gt_cart_m,
            os.path.join(args.out_dir, f"gt_ra_{scale}.png"),
            range_res=range_res, scale=scale,
            title=f"GT (test frame {meta['test_frame']}, {scale})",
        )
        save_ra_cartesian_png(
            rend_cart_m,
            os.path.join(args.out_dir, f"rasterized_ra_{scale}.png"),
            range_res=range_res, scale=scale,
            title=f"{args.baseline_name} rasterized "
                  f"(test frame {meta['test_frame']}, {scale})",
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
