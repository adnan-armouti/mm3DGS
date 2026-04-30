#!/usr/bin/env python3
"""Re-render the three top-row primitive panels with colored backgrounds
for v9 teaser variant A (filled tiles, no border).

Outputs:
  panel_mesh_only_redbg.png      — bg = TOP_RED   (#FBE7E6)
  panel_implicit_redbg.png       — bg = TOP_RED   (#FBE7E6)
  panel_3dps_points_greenbg.png  — bg = TOP_GREEN (#E5F4E5)
"""

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Load shared GL libs before importing open3d (matches prep_teaser_panels_v3)
import ctypes
if "mitsuba" not in sys.modules:
    _libstdcxx = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    if os.path.isfile(_libstdcxx):
        try:
            ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
_dri = "/usr/lib/x86_64-linux-gnu/dri"
if "LIBGL_DRIVERS_PATH" not in os.environ and os.path.isdir(_dri):
    os.environ["LIBGL_DRIVERS_PATH"] = _dri

from figures import prep_teaser_panels_v3 as p3


# Match v9 visual tokens
HEX_RED   = "#FBE7E6"
HEX_GREEN = "#E5F4E5"


def _hex_to_rgba(hex_str: str, alpha: float = 1.0):
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, alpha)


def _patch_bg(rgba):
    """Monkey-patch p3._setup_renderer to use a fixed bg color."""
    orig = p3._setup_renderer
    def _wrapped(width, height, bg=None):
        return orig(width, height, bg=rgba)
    p3._setup_renderer = _wrapped
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="seq_1_frame_185")
    ap.add_argument("--test_frame", type=int, default=185)
    ap.add_argument("--data_dir", default=os.path.join(_REPO, "data"))
    ap.add_argument("--alignment_dir",
                    default=os.path.join(_REPO, "data/alignment_data"))
    ap.add_argument("--ours_dir",
                    default=os.path.join(_REPO, "mm25DGS_v5_v4/output_frame_nvs"))
    ap.add_argument("--panel_root",
                    default=os.path.join(_REPO, "output/teaser_panels"))
    args = ap.parse_args()

    run_dir = os.path.join(args.ours_dir,
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    panel_dir = os.path.join(args.panel_root, args.scene, "v4")
    os.makedirs(panel_dir, exist_ok=True)

    mesh_path  = os.path.join(args.data_dir, args.scene, "scene", "mesh.ply")
    model_path = os.path.join(run_dir, "best_model.pt")

    red_rgba   = _hex_to_rgba(HEX_RED, 1.0)
    green_rgba = _hex_to_rgba(HEX_GREEN, 1.0)

    # Mesh + Implicit → red bg
    orig = _patch_bg(red_rgba)
    print("Rendering mesh-only with red bg…")
    p3.render_mesh_only(mesh_path, args.scene, args.test_frame,
                        args.alignment_dir,
                        os.path.join(panel_dir, "panel_mesh_only_redbg.png"))
    print("Rendering implicit with red bg…")
    p3.render_implicit_style(mesh_path, model_path, args.scene, args.test_frame,
                             args.alignment_dir,
                             os.path.join(panel_dir, "panel_implicit_redbg.png"))
    p3._setup_renderer = orig

    # 3DPS points → green bg
    _patch_bg(green_rgba)
    print("Rendering 3DPS points with green bg…")
    p3.render_3dps_points(mesh_path, model_path, args.scene, args.test_frame,
                          args.alignment_dir,
                          os.path.join(panel_dir, "panel_3dps_points_greenbg.png"))
    p3._setup_renderer = orig

    print(f"\nDone. Colored-bg panels under: {panel_dir}/")


if __name__ == "__main__":
    main()
