"""Diagnostic: load a trained RadarSplat checkpoint and render BOTH the
held-out test frame (val) AND the first training frame, dumping the polar
RA for each. Then we can compare metrics on train vs test to distinguish
geometry/pipeline bugs from generalization failure on 8 frames.

Usage:
    python -m baselines.radarsplat.runner.diagnose_train_vs_test \
        --scene seq_0_frame_135 \
        --ckpt baselines/radarsplat/upstream_results/.../ckpts/ckpt_1999_rank0.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPSTREAM = os.path.abspath(os.path.join(_HERE, "..", "upstream"))
_UPSTREAM_EX = os.path.join(_UPSTREAM, "examples")
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_UPSTREAM_EX, _UPSTREAM, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import radar.dataset.dataloader as _dl  # noqa: E402

# Same monkey-patch as run_radarsplat_scene.
_orig_ds_init = _dl.WaveSensorDataset.__init__
def _patched_ds_init(self, parser, split="train"):
    _orig_ds_init(self, parser, split=split)
    if split == "train":
        self.indices = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    elif split == "val":
        self.indices = np.array([4])
_dl.WaveSensorDataset.__init__ = _patched_ds_init

from radar_simple_trainer import Config, Runner  # noqa: E402
from gsplat.rendering import (  # noqa: E402
    azimuth_antenna_gain_projection,
    spectral_leakage,
)
from gsplat.strategy import DefaultStrategy  # noqa: E402


@torch.no_grad()
def _forward_one(runner: Runner, data) -> np.ndarray:
    device = runner.device
    radarposes = data["radarpose"].to(device)
    Ks = data["K"].to(device)
    pixels = data["image"].to(device) / 255.0
    height, width = pixels.shape[1:3]
    renders, *_ = runner.rasterize_splats(
        radarposes=radarposes, Ks=Ks, width=width, height=height,
        sh_degree=runner.cfg.sh_degree,
        near_plane=runner.cfg.near_plane, far_plane=runner.cfg.far_plane,
        use_polar=runner.parser.use_polar,
    )
    out_img = renders[0]
    if runner.cfg.spectral_leakage:
        out_img = spectral_leakage(out_img, runner.parser.range_resolution,
                                   sinc_width=runner.cfg.sinc_width)
    out_img = azimuth_antenna_gain_projection(
        out_img,
        new_resolution=runner.parser.azimuth_resolution,
        beamwidth=runner.parser.azimuth_beamwidth,
    )
    out_img = out_img.squeeze()
    H_fov = pixels.shape[1]
    out_img = out_img[:H_fov, :]
    return out_img.detach().cpu().numpy().astype(np.float32)


def main_cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root",
                    default=os.path.join(_REPO, "baselines/radarsplat/data_radarsplat"))
    ap.add_argument("--out-dir",
                    default=os.path.join(_REPO, "baselines/radarsplat/results", "diagnose"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config(strategy=DefaultStrategy(verbose=True))
    cfg.data_dir = args.data_root
    cfg.seq_name = args.scene
    cfg.data_factor = 1
    cfg.result_dir = os.path.join(args.out_dir, "_runner_results")
    cfg.max_steps = 1
    cfg.init_type = "random"
    cfg.init_num_pts = 20000
    cfg.init_scale = 0.5
    cfg.test_every = 999
    cfg.intermediate_azimuth_resolution = 0.1
    cfg.max_range = None
    cfg.multipath_weight = 0.0
    cfg.l1occloss_lambda = 0.0
    cfg.use_wandb = False
    cfg.disable_viewer = True
    cfg.save_fig = False
    cfg.save_steps = []
    cfg.eval_steps = []
    cfg.frame_selection = [0, 9]
    cfg.synced_lidar_map_name = "synced_lidar_map_win5"
    cfg.radar_average_map_name = (
        "radar_average_map_polar/"
        "res:0.0586_dist:50_win_size:5_CR_thres:0.21_smooth:3.0"
    )
    cfg.adjust_steps(1.0)

    import radar_simple_trainer as _rst
    _rst.cfg = cfg

    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    ckpt = torch.load(args.ckpt, map_location=runner.device)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]
    print(f"Loaded checkpoint with {len(runner.splats['means'])} splats")

    train_loader = torch.utils.data.DataLoader(
        runner.trainset, batch_size=1, shuffle=False, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        runner.valset, batch_size=1, shuffle=False, num_workers=0
    )

    # Render first train frame (which is on-disk index 0 → frame 131).
    train_frames = []
    for i, data in enumerate(train_loader):
        rendered = _forward_one(runner, data)
        idx = int(data["image_id"].item())  # local index in trainset
        sorted_disk_idx = int(runner.trainset.indices[idx])  # 0..8
        train_frames.append({
            "trainset_idx": idx,
            "disk_idx": sorted_disk_idx,
            "image_path": runner.parser.image_paths[sorted_disk_idx],
        })
        np.save(os.path.join(args.out_dir, f"rend_train_{sorted_disk_idx}.npy"), rendered)
        if i == 0:
            break  # only the first train frame for now

    # Render val frame (index 4 — frame 135).
    for data in val_loader:
        rendered = _forward_one(runner, data)
        np.save(os.path.join(args.out_dir, "rend_val.npy"), rendered)
        break

    print(json.dumps({
        "ckpt": args.ckpt,
        "scene": args.scene,
        "first_train": train_frames,
        "out_dir": args.out_dir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
