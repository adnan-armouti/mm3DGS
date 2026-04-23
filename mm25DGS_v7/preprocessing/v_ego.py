"""v_ego estimation for v7 Doppler forward model.

Source of truth: raw ColoRadar groundtruth trajectory interpolated at
the cascade-frame timestamps. See md/mm25dgs_v7_doppler_plan.md §5.2.

NOT derived from pass-2 aligned configs — those carry ~10-30 cm of
per-frame pose noise which, when differenced between neighbours
200 ms apart, produces 2-4× inflated velocity estimates. The raw GT
trajectory published by ColoRadar is the clean source.

Cached per-frame in ``data/v_ego_cache/<scene>/frame_<F>_v_ego.npy``.
"""
from __future__ import annotations
import os

import numpy as np


# Raw-dataset path (ColoRadar kitti-format release).
RAW_ROOT = '/home/adnan/Documents/Data/coloRadar/raw/kitti'
# Sequence-idx → raw sequence directory name (matches
# mm25DGS_v5/preprocessing/alignment/sc_trajectory_transfer.py).
SEQ_NAME_FOR_IDX = {
    0: '2_28_2021_outdoors_run0',
    1: '2_28_2021_outdoors_run1',
    2: '2_28_2021_outdoors_run2',
}


def scene_to_seq_idx(scene: str) -> int:
    """seq_0_frame_X -> 0, seq_1_frame_X -> 1, seq_2_frame_X -> 2."""
    assert scene.startswith('seq_'), f'bad scene name: {scene}'
    return int(scene.split('_')[1])


def _load_gt_trajectory(seq_idx: int):
    """Return (gt_pos_xyz: (N, 3), gt_ts: (N,)) from the raw ColoRadar
    groundtruth directory."""
    seq_name = SEQ_NAME_FOR_IDX[seq_idx]
    gt_dir = os.path.join(RAW_ROOT, seq_name, 'groundtruth')
    gt     = np.loadtxt(os.path.join(gt_dir, 'groundtruth_poses.txt'))   # (N, 7): xyz + quat
    gt_ts  = np.loadtxt(os.path.join(gt_dir, 'timestamps.txt'))          # (N,)
    return gt[:, :3].astype(np.float64), gt_ts.astype(np.float64)


def _load_cascade_timestamps(seq_idx: int) -> np.ndarray:
    seq_name = SEQ_NAME_FOR_IDX[seq_idx]
    path = os.path.join(RAW_ROOT, seq_name, 'cascade', 'adc_samples',
                        'timestamps.txt')
    return np.loadtxt(path).astype(np.float64)


def compute_v_ego(scene: str, frame: int) -> np.ndarray:
    """Return v_ego(F) ≈ (GT_pos(t[F+1]) − GT_pos(t[F−1])) / (t[F+1] − t[F−1])
    as a (3,) float32 array in meters/second, world frame.

    Edge-frame fallback: one-sided difference when ``F ± 1`` is out of
    range. For our 500+-frame cascades and HO_8 windows, this
    fallback is never hit in practice.
    """
    seq_idx = scene_to_seq_idx(scene)
    gt_xyz, gt_ts = _load_gt_trajectory(seq_idx)
    cas_ts = _load_cascade_timestamps(seq_idx)

    F_m = max(0, frame - 1)
    F_p = min(len(cas_ts) - 1, frame + 1)
    assert F_p > F_m, (
        f'cannot compute v_ego at frame {frame}: F_m={F_m} F_p={F_p}')

    pos_m = np.array([np.interp(cas_ts[F_m], gt_ts, gt_xyz[:, i])
                       for i in range(3)])
    pos_p = np.array([np.interp(cas_ts[F_p], gt_ts, gt_xyz[:, i])
                       for i in range(3)])
    dt = cas_ts[F_p] - cas_ts[F_m]
    return ((pos_p - pos_m) / dt).astype(np.float32)


def v_ego_cache_path(scene: str, frame: int,
                      data_root: str = '/home/adnan/Desktop/mm3DGS/data') -> str:
    cache_dir = os.path.join(data_root, 'v_ego_cache', scene)
    return os.path.join(cache_dir, f'frame_{frame}_v_ego.npy')


def get_or_compute_v_ego(
    scene: str, frame: int,
    data_root: str = '/home/adnan/Desktop/mm3DGS/data',
    force: bool = False,
) -> np.ndarray:
    """Cache-backed v_ego(F). Computes + caches on first call; loads
    from disk thereafter.
    """
    path = v_ego_cache_path(scene, frame, data_root=data_root)
    if not force and os.path.isfile(path):
        return np.load(path).astype(np.float32)
    v = compute_v_ego(scene, frame)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, v)
    return v


if __name__ == '__main__':
    # Quick CLI smoke: print v_ego for the 6 benchmark scenes + frames.
    SCENES = [
        ('seq_0_frame_135', 135),
        ('seq_1_frame_185', 185),
        ('seq_1_frame_438', 438),
        ('seq_2_frame_105', 105),
        ('seq_2_frame_160', 160),
        ('seq_2_frame_300', 300),
    ]
    import math
    LAMBDA = 3.0e8 / 77e9
    T_c = 7.87e-3 / 16
    v_max = LAMBDA / (4 * T_c)
    print(f'Doppler unambig limit: {v_max:.3f} m/s')
    for scene, F in SCENES:
        v = compute_v_ego(scene, F)
        speed = float(np.linalg.norm(v))
        aliases = speed > v_max
        print(f'{scene:<20s} F={F:<4d}  v_ego=[{v[0]:+6.2f} {v[1]:+6.2f} {v[2]:+6.2f}]'
              f'  |v|={speed:.2f} m/s  {"ALIAS" if aliases else "ok"}')
