#!/usr/bin/env python3
"""
Single-Chip Alignment via Cascade Trajectory Transfer
======================================================

Aligns single-chip radar configs by:
1. Aligning every cascade frame in the scene window (4DOF + 2DOF, pick best)
2. Extracting the fitted cascade trajectory (position + orientation vs time)
3. Interpolating to single-chip timestamps and applying the rigid offset
4. Generating aligned single-chip configs

The key insight is that both radars share the **same rotation quaternion**
in their extrinsics (identical mounting orientation), so the cascade
alignment determines the single-chip orientation exactly — only a
known translation offset needs to be applied.
"""

import json
import os
import sys
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation, Slerp

# ======================================================================
# Constants
# ======================================================================

def _resolve_dataset_dir(dataset_dir: str) -> str:
    """Resolve the ColoRadar dataset directory.

    If not provided, checks the COLORADAR_DATASET_DIR environment variable,
    then falls back to a standard location relative to the project root.
    Raises FileNotFoundError if the directory cannot be found.
    """
    if dataset_dir:
        return dataset_dir
    env = os.environ.get("COLORADAR_DATASET_DIR", "")
    if env and os.path.isdir(env):
        return env
    # Try relative to project root: <project>/data/coloRadar/raw/kitti
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidate = os.path.join(project_root, "data", "coloRadar", "raw", "kitti")
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        "ColoRadar dataset directory not found. Set COLORADAR_DATASET_DIR "
        "environment variable or pass dataset_dir= explicitly."
    )

# Dataset root → sequence directory mapping
SEQ_MAP = {
    0: "2_28_2021_outdoors_run0",
    1: "2_28_2021_outdoors_run1",
    2: "2_28_2021_outdoors_run2",
}

# All 9 scenes
ALL_SCENES = [
    ("seq_0_frame_135", 0, 135),
    ("seq_0_frame_390", 0, 390),
    ("seq_0_frame_451", 0, 451),
    ("seq_1_frame_185", 1, 185),
    ("seq_1_frame_277", 1, 277),
    ("seq_1_frame_438", 1, 438),
    ("seq_2_frame_105", 2, 105),
    ("seq_2_frame_160", 2, 160),
    ("seq_2_frame_300", 2, 300),
]

# Extrinsic calibration: base → sensor transforms
# (from calib/transforms/base_to_*.txt)
CALIB_CASCADE = {
    "translation": [0.03, 0.12, -0.09],
    "rotation_quat": [0.0, 0.0, 0.706825181105, 0.707388269167],  # [qx,qy,qz,qw]
}
CALIB_SINGLE_CHIP = {
    "translation": [-0.145, 0.09, -0.025],
    "rotation_quat": [0.0, 0.0, 0.706825181105, 0.707388269167],
}


def _compute_cascade_to_sc_offset():
    """Compute the pure translation from cascade sensor frame to SC sensor frame.

    Since both sensors share the same rotation quaternion, the transform
    cascade→SC is a pure translation in the cascade local frame.
    """
    R_cas = Rotation.from_quat(CALIB_CASCADE["rotation_quat"]).as_matrix()
    t_cas = np.array(CALIB_CASCADE["translation"])
    T_bs_cas = np.eye(4)
    T_bs_cas[:3, :3] = R_cas
    T_bs_cas[:3, 3] = t_cas

    R_sc = Rotation.from_quat(CALIB_SINGLE_CHIP["rotation_quat"]).as_matrix()
    t_sc = np.array(CALIB_SINGLE_CHIP["translation"])
    T_bs_sc = np.eye(4)
    T_bs_sc[:3, :3] = R_sc
    T_bs_sc[:3, 3] = t_sc

    T_cas_to_sc = np.linalg.inv(T_bs_cas) @ T_bs_sc
    return T_cas_to_sc[:3, 3]  # Pure translation in cascade frame


# Pre-computed offset (cascade local frame → SC local frame)
CASCADE_TO_SC_OFFSET_LOCAL = _compute_cascade_to_sc_offset()
# In base frame: [-0.175, -0.03, 0.065] m


# ======================================================================
# Helpers
# ======================================================================

def _norm(v, eps=1e-9):
    return v / (np.linalg.norm(v) + eps)


def extract_pose_from_config(config_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract board center (m) and boresight from an aligned config JSON."""
    with open(config_path) as f:
        cfg = json.load(f)
    tx_pos = np.array([t["pos_mm"] for t in cfg["tx_array"]], dtype=float) / 1000.0
    rx_pos = np.array([r["pos_mm"] for r in cfg["rx_array"]], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])
    center = np.mean(all_pos, axis=0)
    boresight = _norm(np.array(cfg["tx_array"][0]["boresight"], dtype=float))
    return center, boresight


def load_timestamps(dataset_dir: str, seq_idx: int, sensor: str) -> np.ndarray:
    """Load timestamps for a sensor from the ColoRadar dataset."""
    seq_dir = os.path.join(dataset_dir, SEQ_MAP[seq_idx])
    if sensor == "cascade":
        ts_path = os.path.join(seq_dir, "cascade", "heatmaps", "timestamps.txt")
    elif sensor == "single_chip":
        ts_path = os.path.join(seq_dir, "single_chip", "heatmaps", "timestamps.txt")
    else:
        raise ValueError(f"Unknown sensor: {sensor}")
    with open(ts_path) as f:
        return np.array([float(line.strip()) for line in f], dtype=float)


def discover_cascade_frames(data_root: str, scene_name: str) -> List[int]:
    """Discover all cascade frame indices with ADC data in a scene."""
    radar_dir = os.path.join(data_root, scene_name, "radar")
    pattern = os.path.join(radar_dir, "cascaded_frame_*.npy")
    files = sorted(glob.glob(pattern))
    frames = []
    for f in files:
        basename = os.path.basename(f)
        # cascaded_frame_386.npy → 386
        idx_str = basename.replace("cascaded_frame_", "").replace(".npy", "")
        try:
            frames.append(int(idx_str))
        except ValueError:
            pass
    return sorted(frames)


def discover_sc_frames(data_root: str, scene_name: str) -> List[int]:
    """Discover all single-chip frame indices with ADC data in a scene."""
    radar_dir = os.path.join(data_root, scene_name, "radar")
    pattern = os.path.join(radar_dir, "single_chip_frame_*.npy")
    files = sorted(glob.glob(pattern))
    frames = []
    for f in files:
        basename = os.path.basename(f)
        idx_str = basename.replace("single_chip_frame_", "").replace(".npy", "")
        try:
            frames.append(int(idx_str))
        except ValueError:
            pass
    return sorted(frames)


def interpolate_pose(
    t_target: float,
    timestamps: np.ndarray,
    positions: np.ndarray,
    boresights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate position (linear) and boresight (slerp) at a target timestamp.

    Args:
        t_target: target timestamp
        timestamps: (N,) sorted timestamps
        positions: (N, 3) positions
        boresights: (N, 3) unit boresight vectors

    Returns:
        (position, boresight) at t_target
    """
    if len(timestamps) == 1:
        return positions[0].copy(), boresights[0].copy()

    # Find bracketing index
    idx = np.searchsorted(timestamps, t_target) - 1
    idx = max(0, min(idx, len(timestamps) - 2))

    t0, t1 = timestamps[idx], timestamps[idx + 1]
    if abs(t1 - t0) < 1e-12:
        alpha = 0.0
    else:
        alpha = float(np.clip((t_target - t0) / (t1 - t0), 0.0, 1.0))

    # Linear position interpolation
    pos = (1 - alpha) * positions[idx] + alpha * positions[idx + 1]

    # Boresight interpolation via rotation
    # Build rotations from boresight directions using a consistent "up" reference
    def _boresight_to_rotation(bore):
        y = _norm(bore)
        z_hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y, z_hint)) > 0.99:
            z_hint = np.array([0.0, 1.0, 0.0])
        x = _norm(np.cross(y, z_hint))
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        return Rotation.from_matrix(R)

    r0 = _boresight_to_rotation(boresights[idx])
    r1 = _boresight_to_rotation(boresights[idx + 1])

    slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    R_interp = slerp(alpha).as_matrix()
    bore = _norm(R_interp[:, 1])  # y-axis = boresight

    return pos, bore


def cascade_pose_to_sc_pose(
    cas_center: np.ndarray,
    cas_boresight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transfer a cascade pose to the single-chip sensor.

    Since both sensors share the same rotation quaternion on the rig,
    the SC has the same boresight direction. The position offset is
    the known extrinsic translation, rotated into world frame.
    """
    # Build world-frame rotation of the cascade sensor
    y = _norm(cas_boresight)
    z_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(y, z_hint)) > 0.99:
        z_hint = np.array([0.0, 1.0, 0.0])
    x = _norm(np.cross(y, z_hint))
    z = np.cross(x, y)
    R_world = np.column_stack([x, y, z])

    # The offset in cascade local frame → world frame
    offset_world = R_world @ CASCADE_TO_SC_OFFSET_LOCAL
    sc_center = cas_center + offset_world
    sc_boresight = cas_boresight.copy()  # Same orientation
    return sc_center, sc_boresight


def build_sc_config_from_pose(
    sc_center: np.ndarray,
    sc_boresight: np.ndarray,
    base_sc_config: dict,
) -> dict:
    """Build an aligned single-chip config from a transferred pose.

    Keeps the base SC config's waveform parameters and relative antenna
    layout, but updates TX/RX positions and boresights to match the
    transferred pose.
    """
    # Get the relative antenna layout from the base config
    tx_pos_base = np.array([t["pos_mm"] for t in base_sc_config["tx_array"]], dtype=float) / 1000.0
    rx_pos_base = np.array([r["pos_mm"] for r in base_sc_config["rx_array"]], dtype=float) / 1000.0
    all_base = np.vstack([tx_pos_base, rx_pos_base])
    n_tx = len(tx_pos_base)
    base_center = np.mean(all_base, axis=0)

    # Relative positions in the base frame
    rel_pos = all_base - base_center

    # Compute the rotation between old and new boresight
    old_bore = _norm(np.array(base_sc_config["tx_array"][0]["boresight"], dtype=float))
    new_bore = _norm(sc_boresight)

    # Rotation from old boresight frame to new boresight frame
    def _bore_to_basis(bore):
        y = _norm(bore)
        z_hint = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y, z_hint)) > 0.99:
            z_hint = np.array([0.0, 1.0, 0.0])
        x = _norm(np.cross(y, z_hint))
        z = np.cross(x, y)
        return np.column_stack([x, y, z])

    R_old = _bore_to_basis(old_bore)
    R_new = _bore_to_basis(new_bore)
    R_transform = R_new @ R_old.T

    # Apply rotation to relative positions and translate to new center
    new_all_pos = (R_transform @ rel_pos.T).T + sc_center

    # Build new config (deep copy waveform params, update antennas)
    config = json.loads(json.dumps(base_sc_config))
    bore_list = new_bore.tolist()
    for i, tx in enumerate(config["tx_array"]):
        tx["pos_mm"] = (new_all_pos[i] * 1000.0).tolist()
        tx["boresight"] = bore_list
    for i, rx in enumerate(config["rx_array"]):
        rx["pos_mm"] = (new_all_pos[n_tx + i] * 1000.0).tolist()
        rx["boresight"] = bore_list

    return config


# ======================================================================
# Phase 1: Align all cascade frames
# ======================================================================

def align_all_cascade_frames(
    scene_name: str,
    data_root: str = "data",
    output_root: str = "alignment_data",
    verbose: bool = True,
) -> Dict[int, dict]:
    """Align every cascade frame in the scene window.

    Returns:
        Dict mapping frame_idx → alignment result dict.
    """
    from .cascaded_alignment import run_single_frame

    cascade_frames = discover_cascade_frames(data_root, scene_name)
    if not cascade_frames:
        raise FileNotFoundError(
            f"No cascade ADC frames found in {data_root}/{scene_name}/radar/"
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 1: Aligning {len(cascade_frames)} cascade frames for {scene_name}")
        print(f"  Frames: {cascade_frames}")
        print(f"{'='*70}")

    aligned_config_dir = os.path.join(output_root, scene_name, "cascade")
    results = {}

    for frame in cascade_frames:
        result = run_single_frame(
            scene_name=scene_name,
            frame=frame,
            data_root=data_root,
            output_root=output_root,
            aligned_config_dir=aligned_config_dir,
            skip_if_exists=True,
            verbose=verbose,
        )
        if result is not None:
            results[frame] = result

    if verbose:
        n_aligned = len(results)
        mean_cc = np.mean([r["winner_cc"] for r in results.values()]) if results else 0
        print(f"\n  Phase 1 complete: {n_aligned}/{len(cascade_frames)} frames aligned")
        print(f"  Mean winner CC: {mean_cc:.4f}")

    return results


# ======================================================================
# Phase 2: Extract cascade trajectory
# ======================================================================

def extract_cascade_trajectory(
    scene_name: str,
    seq_idx: int,
    alignment_results: Dict[int, dict],
    output_root: str = "alignment_data",
    dataset_dir: str = "",
    verbose: bool = True,
) -> dict:
    """Extract fitted cascade trajectory from aligned configs.

    Returns:
        Dict with keys:
            frames: list of frame indices
            timestamps: (N,) array
            positions: (N, 3) array of board centers in world frame
            boresights: (N, 3) array of unit boresight vectors
            winner_ccs: (N,) array of alignment correlations
    """
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 2: Extracting cascade trajectory for {scene_name}")
        print(f"{'='*70}")

    cas_timestamps = load_timestamps(dataset_dir, seq_idx, "cascade")
    aligned_dir = os.path.join(output_root, scene_name, "cascade")

    frames = sorted(alignment_results.keys())
    timestamps = []
    positions = []
    boresights = []
    winner_ccs = []

    for frame in frames:
        result = alignment_results[frame]
        # Load the winning aligned config
        config_path = os.path.join(
            aligned_dir, f"cascaded_frame_{frame}_aligned.json"
        )
        if not os.path.isfile(config_path):
            # Try the specific winner suffix
            suffix = result["config_suffix"]
            config_path = os.path.join(
                aligned_dir, f"cascaded_frame_{frame}{suffix}.json"
            )
        if not os.path.isfile(config_path):
            if verbose:
                print(f"  WARNING: aligned config missing for frame {frame}, skipping")
            continue

        center, bore = extract_pose_from_config(config_path)
        timestamps.append(cas_timestamps[frame])
        positions.append(center)
        boresights.append(bore)
        winner_ccs.append(result["winner_cc"])

    trajectory = {
        "frames": frames,
        "timestamps": np.array(timestamps),
        "positions": np.array(positions),
        "boresights": np.array(boresights),
        "winner_ccs": np.array(winner_ccs),
    }

    if verbose:
        print(f"  Trajectory: {len(frames)} frames")
        print(f"  Time span: {timestamps[-1] - timestamps[0]:.3f} s")
        pos = np.array(positions)
        total_dist = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        print(f"  Total distance: {total_dist:.3f} m")
        print(f"  Mean winner CC: {np.mean(winner_ccs):.4f}")

    return trajectory


# ======================================================================
# Phase 3: Transfer to single-chip
# ======================================================================

def transfer_to_single_chip(
    scene_name: str,
    seq_idx: int,
    cascade_trajectory: dict,
    data_root: str = "data",
    output_root: str = "alignment_data",
    dataset_dir: str = "",
    verbose: bool = True,
) -> List[dict]:
    """Generate aligned single-chip configs from the cascade trajectory.

    Returns:
        List of dicts with keys: sc_frame, timestamp, config_path,
        cascade_interpolation_alpha, bracketing_cascade_frames.
    """
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    if verbose:
        print(f"\n{'='*70}")
        print(f"Phase 3: Transferring to single-chip for {scene_name}")
        print(f"{'='*70}")

    sc_frames = discover_sc_frames(data_root, scene_name)
    if not sc_frames:
        raise FileNotFoundError(
            f"No single-chip ADC frames found in {data_root}/{scene_name}/radar/"
        )

    sc_timestamps = load_timestamps(dataset_dir, seq_idx, "single_chip")
    cas_ts = cascade_trajectory["timestamps"]
    cas_pos = cascade_trajectory["positions"]
    cas_bore = cascade_trajectory["boresights"]

    # Load a base SC config for waveform params and antenna layout
    base_sc_config_path = os.path.join(
        data_root, scene_name, "configs",
        f"single_chip_frame_{sc_frames[0]}.json",
    )
    with open(base_sc_config_path) as f:
        base_sc_config = json.load(f)

    sc_out_dir = os.path.join(output_root, scene_name, "single_chip")
    os.makedirs(sc_out_dir, exist_ok=True)

    transfer_results = []
    for sc_frame in sc_frames:
        sc_t = sc_timestamps[sc_frame]

        # Check if SC timestamp falls within cascade trajectory time span
        if sc_t < cas_ts[0] or sc_t > cas_ts[-1]:
            # Extrapolate: use nearest cascade frame
            if sc_t < cas_ts[0]:
                cas_center_interp = cas_pos[0].copy()
                cas_bore_interp = cas_bore[0].copy()
                alpha = 0.0
                bracket = (cascade_trajectory["frames"][0], cascade_trajectory["frames"][0])
            else:
                cas_center_interp = cas_pos[-1].copy()
                cas_bore_interp = cas_bore[-1].copy()
                alpha = 1.0
                bracket = (cascade_trajectory["frames"][-1], cascade_trajectory["frames"][-1])
        else:
            # Interpolate
            cas_center_interp, cas_bore_interp = interpolate_pose(
                sc_t, cas_ts, cas_pos, cas_bore
            )
            idx = max(0, min(np.searchsorted(cas_ts, sc_t) - 1, len(cas_ts) - 2))
            alpha = float((sc_t - cas_ts[idx]) / (cas_ts[idx + 1] - cas_ts[idx]))
            bracket = (cascade_trajectory["frames"][idx], cascade_trajectory["frames"][idx + 1])

        # Apply rigid offset: cascade → single-chip
        sc_center, sc_boresight = cascade_pose_to_sc_pose(
            cas_center_interp, cas_bore_interp
        )

        # Build aligned SC config
        aligned_config = build_sc_config_from_pose(
            sc_center, sc_boresight, base_sc_config
        )

        # Save
        out_path = os.path.join(
            sc_out_dir, f"single_chip_frame_{sc_frame}_aligned.json"
        )
        if not os.path.isfile(out_path):
            with open(out_path, "w") as f:
                json.dump(aligned_config, f, indent=2)

        transfer_results.append({
            "sc_frame": sc_frame,
            "timestamp": float(sc_t),
            "config_path": out_path,
            "interpolation_alpha": float(alpha),
            "bracketing_cascade_frames": list(bracket),
            "sc_center_m": sc_center.tolist(),
            "sc_boresight": sc_boresight.tolist(),
        })

    if verbose:
        print(f"  Generated {len(transfer_results)} aligned SC configs")
        # Compute position deltas from unaligned SC configs
        deltas = []
        for tr in transfer_results:
            orig_path = os.path.join(
                data_root, scene_name, "configs",
                f"single_chip_frame_{tr['sc_frame']}.json",
            )
            if os.path.isfile(orig_path):
                orig_center, _ = extract_pose_from_config(orig_path)
                new_center = np.array(tr["sc_center_m"])
                deltas.append(np.linalg.norm(new_center - orig_center))
        if deltas:
            print(f"  Position shift from unaligned: "
                  f"mean={np.mean(deltas)*1000:.1f}mm, "
                  f"max={np.max(deltas)*1000:.1f}mm")

    return transfer_results


# ======================================================================
# Full pipeline
# ======================================================================

def run_pipeline(
    scene_name: str,
    seq_idx: int,
    data_root: str = "data",
    output_root: str = "alignment_data",
    dataset_dir: str = "",
    verbose: bool = True,
) -> dict:
    """Run the full trajectory transfer pipeline for a single scene.

    Returns:
        Dict with cascade_alignment, trajectory, and sc_transfer results.
    """
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    print(f"\n{'='*78}")
    print(f"SC Trajectory Transfer Pipeline: {scene_name}")
    print(f"{'='*78}")

    # Phase 1: Align all cascade frames
    alignment_results = align_all_cascade_frames(
        scene_name, data_root, output_root, verbose
    )
    if not alignment_results:
        print(f"  ERROR: No cascade frames could be aligned for {scene_name}")
        return {}

    # Phase 2: Extract cascade trajectory
    trajectory = extract_cascade_trajectory(
        scene_name, seq_idx, alignment_results, output_root, dataset_dir, verbose
    )

    # Phase 3: Transfer to single-chip
    sc_results = transfer_to_single_chip(
        scene_name, seq_idx, trajectory, data_root, output_root, dataset_dir, verbose
    )

    # Save summary
    summary = {
        "scene_name": scene_name,
        "seq_idx": seq_idx,
        "n_cascade_frames_aligned": len(alignment_results),
        "n_sc_frames_transferred": len(sc_results),
        "cascade_mean_cc": float(np.mean([r["winner_cc"] for r in alignment_results.values()])),
        "cascade_frames": {
            str(k): {
                "frame": v["frame"],
                "winner": v["winner"],
                "winner_cc": v["winner_cc"],
            }
            for k, v in alignment_results.items()
        },
        "sc_transfers": sc_results,
    }

    summary_path = os.path.join(output_root, scene_name, "transfer_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"\n  Summary saved to: {summary_path}")

    return summary


def run_all_scenes(
    data_root: str = "data",
    output_root: str = "alignment_data",
    dataset_dir: str = "",
    scenes: Optional[List[Tuple[str, int, int]]] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run trajectory transfer for multiple scenes."""
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    if scenes is None:
        scenes = ALL_SCENES

    all_summaries = {}
    for scene_name, seq_idx, center_frame in scenes:
        summary = run_pipeline(
            scene_name, seq_idx, data_root, output_root, dataset_dir, verbose
        )
        all_summaries[scene_name] = summary

    # Save global summary
    global_path = os.path.join(output_root, "all_scenes_summary.json")
    with open(global_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)

    if verbose:
        print(f"\n{'='*78}")
        print("All scenes complete. Summary:")
        print(f"{'='*78}")
        for name, s in all_summaries.items():
            if s:
                print(f"  {name}: {s.get('n_cascade_frames_aligned', 0)} cas aligned, "
                      f"{s.get('n_sc_frames_transferred', 0)} SC transferred, "
                      f"mean CC={s.get('cascade_mean_cc', 0):.4f}")
            else:
                print(f"  {name}: FAILED")
        print(f"\nGlobal summary: {global_path}")

    return all_summaries
