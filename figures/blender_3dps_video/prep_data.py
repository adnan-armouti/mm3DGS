#!/usr/bin/env python3
"""Pre-extract 3DPS scene data into Blender-friendly numpy / JSON files.

Blender's bundled Python doesn't ship with PyTorch, so this script runs in
the regular ``mmir`` conda env to dump:

  positions.npy          — (N, 3) float32   point world positions
  normals.npy            — (N, 3) float32   point world normals (from quaternions)
  eps_r.npy              — (N,)   float32   real permittivity ε_r' per point
  antennas.json          — TX/RX positions in metres + scene boresight + per-frame
                           array geometry needed for the video

Run once before launching Blender:
    /home/adnan/.conda/envs/mmir/bin/python figures/blender_3dps_video/prep_data.py
"""

import argparse
import json
import os

import numpy as np
import torch


def _quat_to_normal(quats: np.ndarray) -> np.ndarray:
    q = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n.astype(np.float32)


def _eps_real(raw_mat: np.ndarray) -> np.ndarray:
    return (1.0 + np.log1p(np.exp(np.clip(raw_mat[:, 0], -50, 50)))).astype(np.float32)


def _load_radar_cfg(path: str):
    cfg = json.load(open(path))
    tx = np.asarray([el["pos_mm"] for el in cfg["tx_array"]], dtype=np.float32) / 1000.0
    rx = np.asarray([el["pos_mm"] for el in cfg["rx_array"]], dtype=np.float32) / 1000.0
    centre = np.mean(np.concatenate([tx, rx], axis=0), axis=0)
    bore = np.asarray(cfg["tx_array"][0]["boresight"], dtype=np.float32)
    bore /= np.linalg.norm(bore) + 1e-12
    return {
        "tx": tx.tolist(),
        "rx": rx.tolist(),
        "center": centre.tolist(),
        "boresight": bore.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="seq_1_frame_185")
    ap.add_argument("--test_frame", type=int, default=185)
    ap.add_argument("--repo", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    repo = args.repo
    out_dir = args.out_dir or os.path.join(repo, "output", "blender_video",
                                              "preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    run_dir = os.path.join(
        repo, "mm25DGS_v5_v4", "output_frame_nvs",
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    model_path = os.path.join(run_dir, "best_model.pt")
    cfg_path = os.path.join(repo, "data", "alignment_data", args.scene,
                              "cascade",
                              f"cascaded_frame_{args.test_frame}_aligned_pass2.json")
    if not os.path.exists(cfg_path):
        cfg_path = cfg_path.replace("_pass2.json", ".json")

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    P = state["positions"].numpy().astype(np.float32)
    Q = state["rotations"].numpy().astype(np.float32)
    R = state["raw_materials"].numpy().astype(np.float32)
    N = _quat_to_normal(Q)
    E = _eps_real(R)

    np.save(os.path.join(out_dir, "positions.npy"), P)
    np.save(os.path.join(out_dir, "normals.npy"), N)
    np.save(os.path.join(out_dir, "eps_r.npy"), E)

    antennas = _load_radar_cfg(cfg_path)
    with open(os.path.join(out_dir, "antennas.json"), "w") as f:
        json.dump(antennas, f, indent=2)

    mesh_src = os.path.join(repo, "data", args.scene, "scene", "mesh.ply")
    if os.path.exists(mesh_src):
        # Symlink (or copy) so Blender finds it next to the prep dir.
        link_dst = os.path.join(out_dir, "mesh.ply")
        if os.path.exists(link_dst) or os.path.islink(link_dst):
            os.remove(link_dst)
        os.symlink(mesh_src, link_dst)

    print(f"Wrote prep dir: {out_dir}")
    print(f"  positions.npy : {P.shape}  ({P.dtype})")
    print(f"  normals.npy   : {N.shape}  ({N.dtype})")
    print(f"  eps_r.npy     : {E.shape}  range [{E.min():.2f}, {E.max():.2f}]")
    print(f"  antennas.json : {len(antennas['tx'])} TX, {len(antennas['rx'])} RX")
    print(f"  mesh.ply      : {os.readlink(os.path.join(out_dir, 'mesh.ply'))}")


if __name__ == "__main__":
    main()
