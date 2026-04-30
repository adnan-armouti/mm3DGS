#!/usr/bin/env python3
"""Dump per-frame |RA| panels for the NVS 3x5 grid in v9 teaser.

Outputs (under output/teaser_panels/<scene>/v4/):
  panel_nvs_gt_F<N>.png    — GT |RA| (already exists as _card_F<N>.png; re-uses)
  panel_nvs_ours_F<N>.png  — Ours rendered |RA| for frame N
  panel_nvs_rs_F<N>.png    — RadarSplat rendered |RA| for frame N

For F=185 (held-out test) we use the test-frame npys instead of the train-frame
directory.
"""

import argparse
import os
import shutil
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from figures.prep_teaser_panels_v3 import _save_ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="seq_1_frame_185")
    ap.add_argument("--test_frame", type=int, default=185)
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[181, 184, 185, 186, 189])
    ap.add_argument("--ours_dir",
                    default=os.path.join(_REPO, "mm25DGS_v5_v4/output_frame_nvs"))
    ap.add_argument("--rs_dir",
                    default=os.path.join(_REPO, "baselines/radarsplat/results"))
    ap.add_argument("--panel_root",
                    default=os.path.join(_REPO, "output/teaser_panels"))
    args = ap.parse_args()

    panel_dir = os.path.join(args.panel_root, args.scene, "v4")
    os.makedirs(panel_dir, exist_ok=True)

    run_dir = os.path.join(args.ours_dir,
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    rs_scene_dir = os.path.join(args.rs_dir, args.scene)

    for F in args.frames:
        is_test = (F == args.test_frame)

        # ── GT
        if is_test:
            src = os.path.join(panel_dir, "panel_ra_gt_test.png")
            dst = os.path.join(panel_dir, f"panel_nvs_gt_F{F}.png")
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"GT  F{F}: copied panel_ra_gt_test.png")
        else:
            src = os.path.join(panel_dir, f"_card_F{F}.png")
            dst = os.path.join(panel_dir, f"panel_nvs_gt_F{F}.png")
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"GT  F{F}: copied _card_F{F}.png")

        # ── Ours
        if is_test:
            ra_path = os.path.join(run_dir, "rendered_test_ra_cart.npy")
        else:
            ra_path = os.path.join(run_dir, "train_frames", f"frame_{F}",
                                    "rendered_ra_cart.npy")
        if os.path.exists(ra_path):
            ra = np.load(ra_path)
            _save_ra(ra, os.path.join(panel_dir, f"panel_nvs_ours_F{F}.png"))
            print(f"Ours F{F}: rendered from {ra_path}")
        else:
            print(f"Ours F{F}: MISSING {ra_path}")

        # ── RadarSplat
        if is_test:
            rs_path = os.path.join(rs_scene_dir, "rendered_ra_cart.npy")
        else:
            rs_path = os.path.join(rs_scene_dir, "train_frames",
                                    f"frame_{F}", "rendered_ra_cart.npy")
        if os.path.exists(rs_path):
            ra = np.load(rs_path)
            _save_ra(ra, os.path.join(panel_dir, f"panel_nvs_rs_F{F}.png"))
            print(f"RS   F{F}: rendered from {rs_path}")
        else:
            print(f"RS   F{F}: MISSING {rs_path}")

    print(f"\nDone. NVS grid panels under: {panel_dir}/")


if __name__ == "__main__":
    main()
