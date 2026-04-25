"""Train + render DART for one mm3DGS scene.

Builds DART's cfg dict programmatically (mirror of upstream train.py) for
the ``ngpsh`` field (paper default per PLAN Section 10), runs script_train,
then loads the checkpoint and renders the held-out test frame's pose via
``dart.render``. The rendered RDA cube is reduced to RA (sum over Doppler,
matching upstream radar conventions) and dumped to ``rendered_ra_polar.npy``.
``finalize_metrics`` (mmir env) computes the v6/v7-style metrics from there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "upstream"))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_UPSTREAM, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_cfg(scene: str, data_root: str, out_dir: str, epochs: int,
               batch: int, lr: float = 0.01, key: int = 42,
               pval: float = 0.15) -> dict:
    """Mirror of train.py's cfg construction (ngpsh field, Identity adjust)."""
    sensor_path = os.path.join(data_root, scene, "sensor.json")
    train_h5 = os.path.join(data_root, scene, "data.h5")
    with open(sensor_path) as f:
        sensor_cfg = json.load(f)
    sensor_cfg.update(k=128)

    cfg = {
        "sensor": sensor_cfg,
        "shuffle_buffer": 500_000,
        "lr": lr,
        "batch": batch,
        "epochs": epochs,
        "key": key,
        "out": out_dir,
        "loss": {"weight": None, "loss": "l1", "eps": 1e-6, "delta": 1.0},
        "dataset": {
            "pval": pval,
            "iid_val": False,
            "path": train_h5,
            "doppler_decimation": 0,
        },
        "schedules": {
            "alpha_clip": {
                "func": "linear_piecewise",
                "args": {"values": [-1.0, 0.0, 0.05], "steps": [100, 500]},
            }
        },
        # NGPSH (paper default) field
        "field_name": "NGPSH",
        "field": {
            "levels": 12, "exponent": 0.43, "base": 4.0,
            "size": 16, "features": 2, "units": [64, 32],
            "alpha_scale": 0.1, "harmonics": 25,
        },
        # Identity adjustment (no pose refinement; PLAN Section 10).
        "adjustment_name": "Identity",
        "adjustment": {},
    }
    return cfg


def _render_test(out_dir: str, test_meta_path: str) -> None:
    """Load trained DART, build the test RadarPose, render, dump RA polar."""
    import jax
    from jax import numpy as jnp
    from dart import DART
    from dart import types as dart_types

    with open(test_meta_path) as f:
        meta = json.load(f)

    dart = DART.from_config(**json.load(open(os.path.join(out_dir, "metadata.json"))))
    params = dart.load(os.path.join(out_dir, "model"))

    p = meta["test_pose"]
    pose = dart_types.RadarPose(
        v=jnp.array([p["v"]], dtype=jnp.float32),
        s=jnp.array([p["s"]], dtype=jnp.float32),
        p=jnp.array([p["p"]], dtype=jnp.float32),
        q=jnp.array([p["q"]], dtype=jnp.float32),
        x=jnp.array([p["x"]], dtype=jnp.float32),
        A=jnp.array([p["A"]], dtype=jnp.float32),
        i=jnp.array([p["i"]], dtype=jnp.int32),
    )
    rendered = dart.render(params, pose, key=42)             # (1, Nr, Nd, Na)
    rda = np.asarray(rendered[0], dtype=np.float32)          # (Nr, Nd, Na)

    # Reduce Doppler axis: sum |RDA| over Nd → (Nr, Na). Matches upstream
    # radar conventions (incoherent Doppler integration). Then transpose to
    # (Na, Nr) so axis 0 = azimuth, axis 1 = range, matching
    # baselines/common/eval.py expectations.
    ra = rda.sum(axis=1)                                     # (Nr, Na)
    ra_polar = ra.T.astype(np.float32)                       # (Na, Nr)

    np.save(os.path.join(out_dir, "rendered_ra_polar.npy"), ra_polar)
    np.save(os.path.join(out_dir, "rendered_rda.npy"), rda)


def run(scene: str, data_root: str, out_dir: str, epochs: int, batch: int) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    cfg = _build_cfg(scene, data_root, out_dir, epochs=epochs, batch=batch)

    t0 = time.time()
    from dart.script import script_train
    script_train(cfg)
    train_wall = time.time() - t0

    test_meta_path = os.path.join(data_root, scene, "test_meta.json")
    _render_test(out_dir, test_meta_path)

    # Train_meta for the finalizer.
    test_meta = json.load(open(test_meta_path))
    manifest_path = os.path.join(data_root, scene, "adapter_manifest.json")
    deviations = []
    if os.path.isfile(manifest_path):
        try:
            deviations = json.load(open(manifest_path)).get("deviations_from_reference", [])
        except Exception:
            pass
    deviations = list(deviations) + [
        f"Training epochs: {epochs} (PLAN Section 8 default 3); 8 train frames "
        f"× Nd=16 valid columns / batch is small, so we run more epochs to "
        f"give the HashGrid + plenoctree NN time to settle.",
    ]
    meta = {
        "scene": scene,
        "test_frame": int(test_meta["cascade_test_frame"]),
        "train_frames": list(map(int, test_meta["cascade_train_frames"])),
        "wall_time_seconds": float(train_wall),
        "peak_gpu_mem_mib": 0.0,                      # JAX doesn't expose this easily
        "deviations_from_reference": deviations,
        "upstream_commit": "01e7057eba7ffe6e176c18e8f8e63c6576291c4d",
        "out_dir": out_dir,
        "epochs": int(epochs),
        "batch": int(batch),
        "sensor": "cascaded",
        "Nr": int(test_meta["Nr"]),
        "Nd": int(test_meta["Nd"]),
        "Na": int(test_meta["Na"]),
        "range_res": float(test_meta["range_res"]),
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Finalize in mmir env.
    import subprocess
    mmir_py = os.environ.get("MMIR_PYTHON", "/home/adnan/.conda/envs/mmir/bin/python")
    finalize = os.path.join(_HERE, "finalize_metrics.py")
    subprocess.run(
        [mmir_py, finalize, "--out-dir", out_dir, "--baseline-name", "dart"],
        cwd=_REPO,
        check=True,
    )
    with open(os.path.join(out_dir, "metrics.json")) as f:
        return json.load(f)


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--data-root",
        default=os.path.join(_REPO, "baselines/dart/data_dart"),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: baselines/dart/results/<scene>/",
    )
    ap.add_argument("--epochs", type=int, default=300,
                    help="DART epochs (upstream default 3 — but with our 8-frame setup "
                         "that's < 1 batch/epoch); we run more epochs to give the "
                         "HashGrid + plenoctree NN time to settle.")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        _REPO, "baselines/dart/results", args.scene
    )
    result = run(args.scene, args.data_root, out_dir, args.epochs, args.batch)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
