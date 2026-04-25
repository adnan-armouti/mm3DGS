"""Adapter: mm3DGS CASCADE data → DART format (one scene).

REVISED 2026-04-25 per PLAN Section 4 update: DART now uses CASCADE data
to match RadarSplat / RadarFields / v6/v7 evaluation. The original
single-chip path is removed.

Writes under ``baselines/dart/data_dart/<scene>/``:

    sensor.json                    # DART intrinsics for cascade
    data.h5                        # 8 train cascade frames in column format
    data_test.h5                   # 1 test cascade frame in column format
    test_meta.json                 # test pose + frame number for inference
    adapter_manifest.json

Pipeline (per cascade frame):
  1. Load cascade ADC (16, 16, 12, 256) complex (chirps, RX, TX, ADC).
  2. Chirp-by-chirp build 86-element row-0 virtual array via the same
     TX/RX layout used by mmir.data.ra_utils → (16, 86, 256) complex.
  3. Range-FFT along ADC axis (Hann window) → Nr = 256 bins.
  4. Doppler-FFT along chirps axis (Hann window, fftshift) → Nd = 16 bins.
  5. Azimuth-FFT to 8 bins along the 86-element axis (Hann + fftshift) → Na = 8.
  6. Magnitude → (Nr, Nd, Na) = (256, 16, 8) cube.

For each frame's pose:
  - x = sensor_position_world (m), from tx_array[0].pos_mm.
  - A = world_from_sensor 3x3 (FLU), columns = sensor axes in world.
  - v_world = v_ego_refined.npy (m/s, world frame; mm3DGS v_ego cache).
  - DART's ``make_pose`` then transforms v_world → sensor frame and
    computes s, p, q. Per-column ``weight = psi_min/pi/s`` filters
    zero-weight Doppler columns (matches upstream tools/dataset.py:75-83).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.common import adapters as common  # noqa: E402
from baselines.common import nvs_split, scenes  # noqa: E402

# Cascade virtual-array layout (mirrors mmir/data/ra_utils.py:101-104).
_RX_LOCATIONS = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0),
                 (14, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0),
                 (51, 0), (52, 0), (53, 0)]
_TX_LOCATIONS = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6),
                 (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]
_VX_AZIM = 86      # row 0 of virtual array
_VX_ELEV = 0       # we use elevation 0 only

# DART-side dims
NR = 256                   # cascade ADC samples
ND = 16                    # cascade chirps
NA = 8                     # azimuth bins (fixed by `awr1843boost_az8` gain)
AZ_FFT_SIZE = 8

# Doppler unambiguous max (m/s). Our v_ego is 1.3-1.6 m/s; ±5 m/s gives
# ~3x headroom and matches the SC adapter convention. The true cascade
# d_max depends on the chirp PRI which is not in our config files;
# we use a labeled axis here and document as a limitation.
D_MAX_MPS = 5.0


# ---------------------------------------------------------------------------
# Cascade ADC → RDA cube
# ---------------------------------------------------------------------------

def _casc_txrx_to_vx_chirps(adc_complex: np.ndarray) -> np.ndarray:
    """Cascade ADC ``(chirps, RX=16, TX=12, ADC=256)`` complex →
    row-0 virtual array ``(chirps, vx_az=86, ADC)`` complex."""
    n_chirps, n_rx, n_tx, n_adc = adc_complex.shape
    assert (n_rx, n_tx, n_adc) == (16, 12, 256)
    vx = np.zeros((n_chirps, _VX_AZIM, n_adc), dtype=np.complex128)
    filled = np.zeros(_VX_AZIM, dtype=bool)
    for tx_id in range(n_tx):
        tx_x, tx_y = _TX_LOCATIONS[tx_id]
        if tx_y != _VX_ELEV:
            continue
        for rx_id in range(n_rx):
            rx_x, rx_y = _RX_LOCATIONS[rx_id]
            if rx_y != _VX_ELEV:
                continue
            col = rx_x + tx_x
            if col >= _VX_AZIM:
                continue
            if not filled[col]:
                vx[:, col, :] = adc_complex[:, rx_id, tx_id, :]
                filled[col] = True
            else:
                vx[:, col, :] = 0.5 * (vx[:, col, :]
                                       + adc_complex[:, rx_id, tx_id, :])
    return vx


def adc_to_rda_cube(adc_complex: np.ndarray) -> np.ndarray:
    """Cascade ADC ``(16, 16, 12, 256)`` complex → ``(Nr=256, Nd=16, Na=8)``
    magnitude float32."""
    vx = _casc_txrx_to_vx_chirps(adc_complex)            # (16, 86, 256)
    n_chirps, n_az_full, n_adc = vx.shape

    # 1. Range FFT along ADC (last axis), Hann-windowed.
    win_r = np.hanning(n_adc)
    rng = np.fft.fft(vx * win_r[None, None, :], n=n_adc, axis=-1)
    # rng shape: (chirps=16, 86, Nr=256)

    # 2. Doppler FFT along chirps (axis 0), Hann-windowed; centered.
    win_d = np.hanning(n_chirps)
    rng = rng * win_d[:, None, None]
    dop = np.fft.fftshift(np.fft.fft(rng, n=n_chirps, axis=0), axes=0)
    # dop shape: (Nd=16, 86, Nr=256)

    # 3. Azimuth FFT to 8 bins along the 86-element axis, Hann-windowed,
    #    fftshift to center bin 0 around broadside.
    win_a = np.hanning(n_az_full)
    az = np.fft.fftshift(
        np.fft.fft(dop * win_a[None, :, None], n=AZ_FFT_SIZE, axis=1), axes=1
    )
    # az shape: (Nd, Na=8, Nr=256)

    rda = np.transpose(az, (2, 0, 1))                    # (Nr, Nd, Na)
    return np.abs(rda).astype(np.float32)


# ---------------------------------------------------------------------------
# Pose builder + RadarPose for one frame
# ---------------------------------------------------------------------------

def _orthonormal_basis(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror of dart.pose.make_pose internal: build (p, q) orthonormal
    basis with normalized v in the SENSOR frame."""
    p = np.array([1.0, 0.0, 0.0], dtype=np.float64) - v[0] * v
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-9:
        p = np.array([0.0, 1.0, 0.0])
    else:
        p = p / p_norm
    q = np.cross(v, p)
    return p, q


def build_radar_pose(
    pos_world: np.ndarray, A_world_from_sensor: np.ndarray,
    v_ego_world: np.ndarray, frame_idx: int,
) -> Dict[str, np.ndarray]:
    """Compute the RadarPose fields (sensor-frame v, p, q, s, plus x/A)
    in numpy, mirroring DART's ``dart.pose.make_pose``."""
    A = A_world_from_sensor.astype(np.float64)
    A_inv = np.linalg.inv(A)
    v_sensor = A_inv @ v_ego_world.astype(np.float64)
    s = float(np.linalg.norm(v_sensor))
    if s < 1e-9:
        v_n = np.zeros(3)
    else:
        v_n = v_sensor / s
    p, q = _orthonormal_basis(v_n)
    return {
        "v": v_n.astype(np.float32),
        "s": np.float32(s),
        "p": p.astype(np.float32),
        "q": q.astype(np.float32),
        "x": pos_world.astype(np.float32),
        "A": A.astype(np.float32),
        "i": np.int32(frame_idx),
    }


def _get_psi_min(d: float, v_sensor_x: float, s: float) -> float:
    """Match dart.sensor.VirtualRadar.get_psi_min in numpy."""
    if s < 1e-9:
        return 0.0
    dnorm = d / s
    if abs(dnorm) > 1:
        return 0.0
    vx = v_sensor_x
    h = vx * dnorm / max(np.sqrt(1 - vx * vx), 1e-9)
    r = np.sqrt(max(1 - dnorm * dnorm, 0.0))
    if h > r:
        return 0.0
    if h < -r:
        return float(np.pi)
    return float(np.arccos(h / r))


# ---------------------------------------------------------------------------
# Top-level h5 builder
# ---------------------------------------------------------------------------

def _v_ego_for_frame(scene: str, cascade_frame: int) -> np.ndarray:
    cache_dir = os.path.join(scenes.DATA_ROOT, "v_ego_cache", scene)
    refined_p = os.path.join(cache_dir, f"frame_{cascade_frame}_v_ego_refined.npy")
    seed_p = os.path.join(cache_dir, f"frame_{cascade_frame}_v_ego.npy")
    p = refined_p if os.path.isfile(refined_p) else seed_p
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"v_ego cache missing for {scene} cascade frame {cascade_frame}"
        )
    return np.load(p).astype(np.float64)


def _doppler_axis(nd: int = ND, d_max: float = D_MAX_MPS) -> np.ndarray:
    return np.linspace(-d_max, d_max, nd, dtype=np.float32)


def _build_frame_columns(
    rda: np.ndarray, pose_fields: Dict[str, np.ndarray],
    doppler_values: np.ndarray, frame_idx: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Per-doppler-column reshape: (Nr, Nd, Na) → (Nd, Nr, Na) rows."""
    Nr, Nd, Na = rda.shape
    rad_per_col = np.transpose(rda, (1, 0, 2))           # (Nd, Nr, Na)
    cols = {
        "x": np.broadcast_to(pose_fields["x"], (Nd, 3)).copy(),
        "A": np.broadcast_to(pose_fields["A"], (Nd, 3, 3)).copy(),
        "v": np.broadcast_to(pose_fields["v"], (Nd, 3)).copy(),
        "s": np.broadcast_to(np.float32(pose_fields["s"]), (Nd,)).copy(),
        "p": np.broadcast_to(pose_fields["p"], (Nd, 3)).copy(),
        "q": np.broadcast_to(pose_fields["q"], (Nd, 3)).copy(),
        "i": np.broadcast_to(np.int32(pose_fields["i"]), (Nd,)).copy(),
        "doppler": doppler_values.astype(np.float32),
        "doppler_idx": np.arange(Nd, dtype=np.uint16),
        "frame_idx": np.full(Nd, frame_idx, dtype=np.uint16),
    }
    return cols, rad_per_col.astype(np.float16)


def _stack_columns(per_frame: List[Tuple[Dict[str, np.ndarray], np.ndarray]],
                   ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    keys = list(per_frame[0][0].keys())
    out = {k: np.concatenate([f[0][k] for f in per_frame], axis=0) for k in keys}
    rad = np.concatenate([f[1] for f in per_frame], axis=0)
    return out, rad


def _filter_nonzero_weight(
    cols: Dict[str, np.ndarray], rad: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    weights = np.zeros(rad.shape[0], dtype=np.float32)
    for j in range(rad.shape[0]):
        d = float(cols["doppler"][j])
        s = float(cols["s"][j])
        v_sensor_x = float(cols["v"][j, 0])
        psi = _get_psi_min(d, v_sensor_x, s)
        weights[j] = psi / np.pi / max(s, 1e-9)
    keep = weights > 0
    cols2 = {k: v[keep] for k, v in cols.items()}
    cols2["weight"] = weights[keep].astype(np.float32)
    return cols2, rad[keep]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _make_frame_records(
    scene: str, frame_paths: List[str], cascade_frames: List[int],
    doppler_axis: np.ndarray, norm: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[int]]:
    per_frame = []
    used_frames = []
    for cascade_frame, adc_path in zip(cascade_frames, frame_paths):
        adc = common.load_cascaded_adc(adc_path)              # (16, 16, 12, 256)
        rda = adc_to_rda_cube(adc)                            # (Nr, Nd, Na)
        rda = np.maximum(rda, 0.0) / norm

        # Pose for this cascade frame.
        align_dir = scenes.cascade_alignment_dir(scene)
        cfg_path = os.path.join(
            align_dir, f"cascaded_frame_{cascade_frame}_aligned.json"
        )
        cfg = common.load_config(cfg_path)
        T, R, t = common.pose_from_config(cfg)
        v_world = _v_ego_for_frame(scene, cascade_frame)
        pose_fields = build_radar_pose(t, R, v_world, frame_idx=len(used_frames))

        cols, rad = _build_frame_columns(
            rda, pose_fields, doppler_axis, frame_idx=len(used_frames)
        )
        per_frame.append((cols, rad))
        used_frames.append(cascade_frame)
    return (*_stack_columns(per_frame), used_frames)


def build_scene(scene: str, out_root: str) -> dict:
    out_dir = os.path.join(out_root, scene)
    os.makedirs(out_dir, exist_ok=True)

    casc_split = nvs_split.cascaded_split(scene)
    train_frames = casc_split["train_frames"]
    test_frame = casc_split["test_frame"]

    # Range axis from the test frame's config.
    from mmir.data.io_utils import compute_range_res_from_cfg
    range_res = compute_range_res_from_cfg(casc_split["test_config"])
    r_max = (NR - 1) * range_res

    doppler = _doppler_axis(ND, D_MAX_MPS)

    print(f"[{scene}] computing per-scene RDA normalisation...")
    sample_rdas = []
    for p in casc_split["train_files"]:
        adc = common.load_cascaded_adc(p)
        sample_rdas.append(adc_to_rda_cube(adc))
    norm = float(np.percentile(np.concatenate([r.ravel() for r in sample_rdas]), 99.0))
    norm = max(norm, 1.0)

    # 1. data.h5 (8 train frames)
    print(f"[{scene}] building train data.h5 ({len(train_frames)} frames)...")
    train_cols, train_rad, _ = _make_frame_records(
        scene, casc_split["train_files"], train_frames, doppler, norm,
    )
    train_cols, train_rad = _filter_nonzero_weight(train_cols, train_rad)
    print(f"  Valid columns: {train_rad.shape[0]} / {len(train_frames) * ND}")

    with h5py.File(os.path.join(out_dir, "data.h5"), "w") as f:
        for k, v in train_cols.items():
            f.create_dataset(k, data=v)
        f.create_dataset("rad", data=train_rad)

    # 2. data_test.h5
    print(f"[{scene}] building test data_test.h5...")
    test_cols, test_rad, _ = _make_frame_records(
        scene, [casc_split["test_file"]], [test_frame], doppler, norm,
    )
    test_cols, test_rad = _filter_nonzero_weight(test_cols, test_rad)
    print(f"  Valid test columns: {test_rad.shape[0]} / {ND}")

    with h5py.File(os.path.join(out_dir, "data_test.h5"), "w") as f:
        for k, v in test_cols.items():
            f.create_dataset(k, data=v)
        f.create_dataset("rad", data=test_rad)

    # 3. sensor.json
    sensor_cfg = {
        "r": [0.0, float(r_max), int(NR)],
        "d": [-float(D_MAX_MPS), float(D_MAX_MPS), int(ND)],
        "k": 128,
        "gain": "awr1843boost_az8",
    }
    with open(os.path.join(out_dir, "sensor.json"), "w") as f:
        json.dump(sensor_cfg, f, indent=2)

    # 4. test_meta.json
    test_cfg = common.load_config(casc_split["test_config"])
    T, R, t = common.pose_from_config(test_cfg)
    v_world_test = _v_ego_for_frame(scene, test_frame)
    pose_fields_test = build_radar_pose(t, R, v_world_test, frame_idx=0)

    test_meta = {
        "scene": scene,
        "cascade_test_frame": int(test_frame),
        "cascade_train_frames": [int(x) for x in train_frames],
        "test_pose": {
            "x": pose_fields_test["x"].tolist(),
            "A": pose_fields_test["A"].tolist(),
            "v": pose_fields_test["v"].tolist(),
            "s": float(pose_fields_test["s"]),
            "p": pose_fields_test["p"].tolist(),
            "q": pose_fields_test["q"].tolist(),
            "i": int(pose_fields_test["i"]),
        },
        "v_ego_world": v_world_test.tolist(),
        "norm": float(norm),
        "range_res": float(range_res),
        "Nr": int(NR), "Nd": int(ND), "Na": int(NA),
        "d_max_mps": float(D_MAX_MPS),
        "test_file": casc_split["test_file"],
        "test_config": casc_split["test_config"],
    }
    with open(os.path.join(out_dir, "test_meta.json"), "w") as f:
        json.dump(test_meta, f, indent=2)

    manifest = {
        "scene": scene,
        "cascade_train_frames": [int(x) for x in train_frames],
        "cascade_test_frame": int(test_frame),
        "norm": float(norm),
        "range_res": float(range_res),
        "Nr": NR, "Nd": ND, "Na": NA, "d_max_mps": D_MAX_MPS,
        "n_train_columns": int(train_rad.shape[0]),
        "n_test_columns": int(test_rad.shape[0]),
        "deviations_from_reference": [
            "Synthetic 9-frame sequence rather than upstream's 100+ frame collections.",
            "TI MMWCAS cascade (12 TX × 16 RX); azimuth-FFT'd to 8 bins to match "
            "stock 'awr1843boost_az8' gain function (vs natural 127-bin cascade RA).",
            "Doppler axis: 16 chirps → 16 bins (vs upstream Boreas 256 bins). "
            "PLAN Section 9(a) flagged this as gating concern; verified ego speed "
            "1.3-1.6 m/s gives non-zero psi_min weights across the working range.",
            "Doppler axis labeled ±5 m/s; true cascade d_max depends on the "
            "chirp PRI which is not in our config files. Documented limitation.",
            "v_ego from mm3DGS v_ego_refined cache (Stage 0 GT-trajectory "
            "interpolation + Stage 1+2 differentiable refinement).",
            "Per-scene normalisation: 99th percentile of train RDA cubes "
            "(vs upstream norm=1e4).",
            "--pval=0.05 (vs 0 in PLAN); upstream's script_train asserts val is "
            "not None, so a tiny 5% holdout is used.",
            "--adj=Identity (no pose refinement; PLAN Section 10).",
        ],
    }
    with open(os.path.join(out_dir, "adapter_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument(
        "--out-root",
        default=os.path.join(_REPO, "baselines", "dart", "data_dart"),
    )
    args = ap.parse_args()
    m = build_scene(args.scene, args.out_root)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
