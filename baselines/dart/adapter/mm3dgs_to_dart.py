"""Adapter: mm3DGS → DART format (one scene, dual-mode).

REVISED 2026-04-25 (twice). Per PLAN Section 4 (current revision):

  --mode cascaded     (default)  86-elem virtual array → 127 az bins.
                                 ``cascade_az127`` gain function (1-line
                                 patch to upstream antenna.py).
                                 Apples-to-apples vs RadarSplat / RadarFields
                                 / mm3DGS-v6/v7 — uses cascade GT.
                                 Output cube shape: (Nr=256, Nd=16, Na=127).

  --mode single_chip  (alt)      8-elem virtual array → 8 az bins via
                                 stock ``awr1843boost_az8``. NOT comparable
                                 to cascade-trained baselines (different GT).
                                 Output cube shape: (Nr=128, Nd=128, Na=8).

Writes under ``baselines/dart/data_dart/<scene>__<mode>/``:

    sensor.json                    # DART intrinsics for this mode
    data.h5                        # 8 train frames, column format
    data_test.h5                   # 1 test frame, column format
    test_meta.json                 # test pose for inference
    adapter_manifest.json
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
_CASCADE_RX = [(0, 0), (1, 0), (2, 0), (3, 0), (11, 0), (12, 0), (13, 0),
               (14, 0), (46, 0), (47, 0), (48, 0), (49, 0), (50, 0),
               (51, 0), (52, 0), (53, 0)]
_CASCADE_TX = [(0, 0), (4, 0), (8, 0), (9, 1), (10, 4), (11, 6),
               (12, 0), (16, 0), (20, 0), (24, 0), (28, 0), (32, 0)]
_CASCADE_VX = 86
_CASCADE_NA_FFT = 128
_CASCADE_NA_OUT = 127            # drop FFT bin 0 (DC), fftshift

# Single-chip virtual-array layout (mirrors mmir/data/ra_utils.py:223-258).
_SC_TX_LOCS = [(0, 0), (2, 1), (4, 0)]   # TX1, TX2, TX3
_SC_RX_LOCS = [(0, 0), (1, 0), (2, 0), (3, 0)]
_SC_VX_AZ = 8
_SC_NA_OUT = 8

# Common Doppler unambiguous max (m/s); ego speeds are 1.3-1.6 m/s, so
# ±5 m/s gives ~3x headroom. Chirp PRI is not in our cascade configs;
# documented as a label-only assumption.
D_MAX_MPS = 5.0


# ---------------------------------------------------------------------------
# Cascade ADC → RDA cube (Na=127)
# ---------------------------------------------------------------------------

def _casc_txrx_to_vx_chirps(adc_complex: np.ndarray) -> np.ndarray:
    """Cascade ADC ``(chirps, RX=16, TX=12, ADC=256)`` complex →
    row-0 virtual array ``(chirps, vx_az=86, ADC)`` complex."""
    n_chirps, n_rx, n_tx, n_adc = adc_complex.shape
    assert (n_rx, n_tx, n_adc) == (16, 12, 256)
    vx = np.zeros((n_chirps, _CASCADE_VX, n_adc), dtype=np.complex128)
    filled = np.zeros(_CASCADE_VX, dtype=bool)
    for tx_id in range(n_tx):
        tx_x, tx_y = _CASCADE_TX[tx_id]
        if tx_y != 0:
            continue
        for rx_id in range(n_rx):
            rx_x, rx_y = _CASCADE_RX[rx_id]
            if rx_y != 0:
                continue
            col = rx_x + tx_x
            if col >= _CASCADE_VX:
                continue
            if not filled[col]:
                vx[:, col, :] = adc_complex[:, rx_id, tx_id, :]
                filled[col] = True
            else:
                vx[:, col, :] = 0.5 * (vx[:, col, :]
                                       + adc_complex[:, rx_id, tx_id, :])
    return vx


def adc_cascade_to_rda(adc_complex: np.ndarray) -> np.ndarray:
    """Cascade ADC ``(16, 16, 12, 256)`` complex → ``(Nr=256, Nd=16, Na=127)``
    magnitude float32. Pipeline matches mm3DGS-v7's RA convention but adds
    the Doppler axis: range FFT, Doppler FFT (fftshift), azimuth FFT-128
    (fftshift, drop bin 0)."""
    vx = _casc_txrx_to_vx_chirps(adc_complex)            # (16, 86, 256)
    n_chirps, n_az_full, n_adc = vx.shape

    # Range FFT along ADC, Hann-windowed.
    win_r = np.hanning(n_adc)
    rng = np.fft.fft(vx * win_r[None, None, :], n=n_adc, axis=-1)
    # rng: (16, 86, 256)

    # Doppler FFT along chirps (fftshift to center at d=0).
    win_d = np.hanning(n_chirps)
    rng = rng * win_d[:, None, None]
    dop = np.fft.fftshift(np.fft.fft(rng, n=n_chirps, axis=0), axes=0)
    # dop: (Nd=16, 86, 256)

    # Azimuth FFT to 128 bins (matches mm3DGS-v7 convention).
    win_a = np.hanning(n_az_full)
    az = np.fft.fft(dop * win_a[None, :, None], n=_CASCADE_NA_FFT, axis=1)
    # Drop bin 0 (DC) then fftshift to center bin 0 around broadside.
    az = az[:, 1:, :]                                     # (Nd, 127, 256)
    az = np.fft.fftshift(az, axes=1)
    # az: (Nd, Na=127, 256)

    rda = np.transpose(az, (2, 0, 1))                    # (Nr, Nd, Na)
    return np.abs(rda).astype(np.float32)


# ---------------------------------------------------------------------------
# Single-chip ADC → RDA cube (Na=8)
# ---------------------------------------------------------------------------

def _sc_txrx_to_vx_chirps(adc_complex: np.ndarray) -> np.ndarray:
    """SC ADC ``(chirps=128, RX=4, TX=3, ADC=128)`` complex →
    row-0 virtual array ``(chirps, vx_az=8, ADC)`` complex."""
    n_chirps, n_rx, n_tx, n_adc = adc_complex.shape
    assert (n_rx, n_tx, n_adc) == (4, 3, 128)
    vx = np.zeros((n_chirps, 2, _SC_VX_AZ, n_adc), dtype=np.complex128)
    filled = np.zeros((2, _SC_VX_AZ), dtype=bool)
    for tx_id in range(n_tx):
        tx_az, tx_el = _SC_TX_LOCS[tx_id]
        for rx_id in range(n_rx):
            rx_az, rx_el = _SC_RX_LOCS[rx_id]
            col = tx_az + rx_az
            row = tx_el + rx_el
            if col >= _SC_VX_AZ or row >= 2:
                continue
            if not filled[row, col]:
                vx[:, row, col, :] = adc_complex[:, rx_id, tx_id, :]
                filled[row, col] = True
            else:
                vx[:, row, col, :] = 0.5 * (vx[:, row, col, :]
                                            + adc_complex[:, rx_id, tx_id, :])
    return vx[:, 0, :, :]   # (chirps, 8, ADC) — row 0 only


def adc_sc_to_rda(adc_complex: np.ndarray) -> np.ndarray:
    """SC ADC ``(128, 4, 3, 128)`` complex → ``(Nr=128, Nd=128, Na=8)``
    magnitude float32."""
    vx = _sc_txrx_to_vx_chirps(adc_complex)              # (128, 8, 128)
    n_chirps, n_az, n_adc = vx.shape

    win_r = np.hanning(n_adc)
    rng = np.fft.fft(vx * win_r[None, None, :], n=n_adc, axis=-1)

    win_d = np.hanning(n_chirps)
    rng = rng * win_d[:, None, None]
    dop = np.fft.fftshift(np.fft.fft(rng, n=n_chirps, axis=0), axes=0)
    # dop: (Nd=128, 8, Nr=128)

    win_a = np.hanning(n_az)
    az = np.fft.fftshift(
        np.fft.fft(dop * win_a[None, :, None], n=_SC_NA_OUT, axis=1), axes=1
    )
    # az: (Nd, Na=8, Nr)

    rda = np.transpose(az, (2, 0, 1))                    # (Nr, Nd, Na)
    return np.abs(rda).astype(np.float32)


# ---------------------------------------------------------------------------
# Mode-aware dispatch
# ---------------------------------------------------------------------------

def adc_to_rda_cube(adc_complex: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cascaded":
        return adc_cascade_to_rda(adc_complex)
    if mode == "single_chip":
        return adc_sc_to_rda(adc_complex)
    raise ValueError(f"unknown DART mode {mode!r}")


def _dims_for_mode(mode: str) -> Tuple[int, int, int]:
    """(Nr, Nd, Na) for the given DART mode."""
    if mode == "cascaded":
        return 256, 16, _CASCADE_NA_OUT
    if mode == "single_chip":
        return 128, 128, _SC_NA_OUT
    raise ValueError(mode)


def _gain_for_mode(mode: str) -> str:
    return "cascade_az127" if mode == "cascaded" else "awr1843boost_az8"


def _load_adc(scene: str, frame: int, mode: str) -> np.ndarray:
    """Load ADC for a given (scene, mode, on-disk frame index)."""
    if mode == "cascaded":
        path = os.path.join(scenes.scene_dir(scene), "radar",
                            f"cascaded_frame_{frame}.npy")
        return common.load_cascaded_adc(path)
    path = os.path.join(scenes.scene_dir(scene), "radar",
                        f"single_chip_frame_{frame}.npy")
    return common.load_single_chip_adc(path)


def _config_for_frame(scene: str, frame: int, mode: str) -> str:
    if mode == "cascaded":
        return os.path.join(scenes.cascade_alignment_dir(scene),
                            f"cascaded_frame_{frame}_aligned.json")
    return os.path.join(scenes.single_chip_alignment_dir(scene),
                        f"single_chip_frame_{frame}_aligned.json")


def _v_ego_for_cascade_frame(scene: str, cascade_frame: int) -> np.ndarray:
    """Cascade frame's v_ego (refined, world frame, m/s).
    For SC mode we pair SC frames to cascade frames in sorted order and
    use the corresponding cascade v_ego (timestamps offset by ~0.05 s)."""
    cache_dir = os.path.join(scenes.DATA_ROOT, "v_ego_cache", scene)
    refined_p = os.path.join(cache_dir, f"frame_{cascade_frame}_v_ego_refined.npy")
    seed_p = os.path.join(cache_dir, f"frame_{cascade_frame}_v_ego.npy")
    p = refined_p if os.path.isfile(refined_p) else seed_p
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"v_ego cache missing for {scene} cascade frame {cascade_frame}"
        )
    return np.load(p).astype(np.float64)


# ---------------------------------------------------------------------------
# Pose builder + RadarPose for one frame (mirrors dart.pose.make_pose in numpy)
# ---------------------------------------------------------------------------

def _orthonormal_basis(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    A = A_world_from_sensor.astype(np.float64)
    A_inv = np.linalg.inv(A)
    v_sensor = A_inv @ v_ego_world.astype(np.float64)
    s = float(np.linalg.norm(v_sensor))
    v_n = v_sensor / s if s >= 1e-9 else np.zeros(3)
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
# Per-frame column construction
# ---------------------------------------------------------------------------

def _build_frame_columns(
    rda: np.ndarray, pose_fields: Dict[str, np.ndarray],
    doppler_values: np.ndarray, frame_idx: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
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


def _make_frame_records(
    scene: str, frames: List[int], cascade_pair_frames: List[int], mode: str,
    doppler_axis: np.ndarray, norm: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """frames: list of on-disk frame numbers (cascade or SC depending on mode).
    cascade_pair_frames: same length, the cascade-frame number to look up
    v_ego against (for SC mode this is the paired cascade frame; for
    cascade mode it equals frames)."""
    per_frame = []
    Nr, Nd, Na = _dims_for_mode(mode)
    for k, (frame, casc_frame) in enumerate(zip(frames, cascade_pair_frames)):
        adc = _load_adc(scene, frame, mode)
        rda = adc_to_rda_cube(adc, mode)                  # (Nr, Nd, Na)
        rda = np.maximum(rda, 0.0) / norm

        cfg_path = _config_for_frame(scene, frame, mode)
        cfg = common.load_config(cfg_path)
        T, R, t = common.pose_from_config(cfg)
        v_world = _v_ego_for_cascade_frame(scene, casc_frame)
        pose_fields = build_radar_pose(t, R, v_world, frame_idx=k)

        cols, rad = _build_frame_columns(
            rda, pose_fields, doppler_axis, frame_idx=k
        )
        per_frame.append((cols, rad))
    return _stack_columns(per_frame)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_scene(scene: str, mode: str, out_root: str) -> dict:
    Nr, Nd, Na = _dims_for_mode(mode)
    out_dir = os.path.join(out_root, f"{scene}__{mode}")
    os.makedirs(out_dir, exist_ok=True)

    if mode == "cascaded":
        split = nvs_split.cascaded_split(scene)
        train_frames = list(split["train_frames"])
        test_frame = split["test_frame"]
        train_cascade_pair = list(train_frames)
        test_cascade_pair = test_frame
    elif mode == "single_chip":
        sc_split = nvs_split.single_chip_split(scene)
        casc_split = nvs_split.cascaded_split(scene)
        sorted_sc = sorted(sc_split["train_frames"] + [sc_split["test_frame"]])
        sorted_casc = sorted(casc_split["train_frames"] + [casc_split["test_frame"]])
        sc_to_casc = dict(zip(sorted_sc, sorted_casc))
        train_frames = list(sc_split["train_frames"])
        test_frame = sc_split["test_frame"]
        train_cascade_pair = [sc_to_casc[f] for f in train_frames]
        test_cascade_pair = sc_to_casc[test_frame]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Range axis from the test frame's config.
    from mmir.data.io_utils import compute_range_res_from_cfg
    test_cfg_path = _config_for_frame(scene, test_frame, mode)
    range_res = compute_range_res_from_cfg(test_cfg_path)
    r_max = (Nr - 1) * range_res

    doppler = np.linspace(-D_MAX_MPS, D_MAX_MPS, Nd, dtype=np.float32)

    print(f"[{scene}/{mode}] computing per-scene RDA normalisation...")
    sample_rdas = [adc_to_rda_cube(_load_adc(scene, f, mode), mode)
                   for f in train_frames]
    norm = float(np.percentile(np.concatenate(
        [r.ravel() for r in sample_rdas]), 99.0))
    norm = max(norm, 1.0)

    print(f"[{scene}/{mode}] building train data.h5 ({len(train_frames)} frames)...")
    train_cols, train_rad = _make_frame_records(
        scene, train_frames, train_cascade_pair, mode, doppler, norm,
    )
    train_cols, train_rad = _filter_nonzero_weight(train_cols, train_rad)
    print(f"  Valid columns: {train_rad.shape[0]} / {len(train_frames) * Nd}")
    with h5py.File(os.path.join(out_dir, "data.h5"), "w") as f:
        for k, v in train_cols.items():
            f.create_dataset(k, data=v)
        f.create_dataset("rad", data=train_rad)

    print(f"[{scene}/{mode}] building test data_test.h5...")
    test_cols, test_rad = _make_frame_records(
        scene, [test_frame], [test_cascade_pair], mode, doppler, norm,
    )
    test_cols, test_rad = _filter_nonzero_weight(test_cols, test_rad)
    print(f"  Valid test columns: {test_rad.shape[0]} / {Nd}")
    with h5py.File(os.path.join(out_dir, "data_test.h5"), "w") as f:
        for k, v in test_cols.items():
            f.create_dataset(k, data=v)
        f.create_dataset("rad", data=test_rad)

    sensor_cfg = {
        "r": [0.0, float(r_max), int(Nr)],
        "d": [-float(D_MAX_MPS), float(D_MAX_MPS), int(Nd)],
        "k": 128,
        "gain": _gain_for_mode(mode),
    }
    with open(os.path.join(out_dir, "sensor.json"), "w") as f:
        json.dump(sensor_cfg, f, indent=2)

    test_cfg = common.load_config(test_cfg_path)
    T, R, t = common.pose_from_config(test_cfg)
    v_world_test = _v_ego_for_cascade_frame(scene, test_cascade_pair)
    pose_fields_test = build_radar_pose(t, R, v_world_test, frame_idx=0)

    test_meta = {
        "scene": scene,
        "mode": mode,
        "test_frame": int(test_frame),
        "train_frames": [int(x) for x in train_frames],
        "test_cascade_pair": int(test_cascade_pair),
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
        "Nr": int(Nr), "Nd": int(Nd), "Na": int(Na),
        "d_max_mps": float(D_MAX_MPS),
        "test_config": test_cfg_path,
    }
    with open(os.path.join(out_dir, "test_meta.json"), "w") as f:
        json.dump(test_meta, f, indent=2)

    deviations = [
        f"DART mode: {mode}.",
        "Synthetic 9-frame sequence vs upstream's 100+ frame collections.",
    ]
    if mode == "cascaded":
        deviations += [
            "Cascade 86-element row-0 virtual array → azimuth FFT-128 → "
            "drop bin 0 → fftshift → Na=127 (matches mm3DGS-v6/v7 RA).",
            "Custom DART antenna gain `cascade_az127` added via "
            "patches/cascade_az127.patch (mirrors awr1843boost_az8 for an "
            "86-element half-lambda array).",
            "Doppler axis: 16 chirps → 16 bins (vs upstream Boreas 256). "
            "PLAN Section 9(a) flagged this; ego speed 1.3-1.6 m/s gives "
            "non-zero psi_min weights.",
        ]
    else:
        deviations += [
            "TI IWR1443 SC, stock awr1843boost_az8 gain (Na=8).",
            "SC frame paired to cascade frame in sorted order (timestamps "
            "offset by ~0.05 s).",
            "NOT comparable to cascade-trained baselines' GT — kept available "
            "for a 'DART at its design point' reference number only.",
        ]
    deviations += [
        "Doppler axis labeled ±5 m/s; true cascade d_max requires chirp PRI "
        "not in our config files. Documented limitation.",
        "v_ego from mm3DGS v_ego_refined cache.",
        "Per-scene normalisation: 99th percentile of train RDA cubes.",
        "--pval=0.15 (vs 0 in upstream PLAN); ensures non-empty val for "
        "small training sets.",
        "--adj=Identity (no pose refinement; PLAN Section 10).",
    ]

    manifest = {
        "scene": scene,
        "mode": mode,
        "train_frames": [int(x) for x in train_frames],
        "test_frame": int(test_frame),
        "Nr": Nr, "Nd": Nd, "Na": Na, "d_max_mps": D_MAX_MPS,
        "norm": float(norm),
        "range_res": float(range_res),
        "n_train_columns": int(train_rad.shape[0]),
        "n_test_columns": int(test_rad.shape[0]),
        "deviations_from_reference": deviations,
    }
    with open(os.path.join(out_dir, "adapter_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--mode", choices=["cascaded", "single_chip"],
                    default="cascaded")
    ap.add_argument(
        "--out-root",
        default=os.path.join(_REPO, "baselines", "dart", "data_dart"),
    )
    args = ap.parse_args()
    m = build_scene(args.scene, args.mode, args.out_root)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
