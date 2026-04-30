#!/usr/bin/env python3
"""V4 panel preparation — adds complex range-profile heatmaps + stacked cards.

Generates panels for one held-out test frame and one representative training
frame so the bottom-row "Rendering" sub-figure can show GT vs 3DPS for both
the ADC, the complex range profile (CRP), and the |RA| — with the CRP shown
explicitly as the renderer's native output.
"""

import argparse
import ctypes
import os
import sys

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm as mpl_cm
from PIL import Image, ImageDraw, ImageFilter

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Re-use the v3 panel renderers (mesh, implicit, points, ADC traces)
from figures.prep_teaser_panels_v3 import (
    render_mesh_only, render_implicit_style, render_3dps_points,
    render_compressed_points,
    _save_ra, _save_adc_traces,
)

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ---------------------------------------------------------------------------
# Complex range profile magnitude heatmap (for "Rendering" panel)
# ---------------------------------------------------------------------------

def save_crp_heatmap(rp_complex: np.ndarray, out_path: str, dpi: int = 300,
                      log_scale: bool = True):
    """Plot |CRP| as a 2D heatmap: pairs (rows) x range bins (columns).

    rp_complex: shape (n_tx, n_rx, K) complex.
    """
    n_tx, n_rx, K = rp_complex.shape
    flat = rp_complex.reshape(n_tx * n_rx, K)
    mag = np.abs(flat).astype(np.float64)
    if log_scale:
        eps = mag.max() * 1e-4 + 1e-12
        img = 20.0 * np.log10(mag + eps)
        lo, hi = np.percentile(img, [2.0, 99.0])
        img = np.clip((img - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    else:
        img = mag / max(mag.max(), 1e-30)

    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=dpi)
    ax.imshow(img, cmap="viridis", aspect="auto", origin="upper",
              vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0,
                facecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stacked-cards composition (for "Compression" panel)
# ---------------------------------------------------------------------------

def make_stacked_cards(image_paths, out_path, card_w=180, card_h=180,
                       dx=10, dy=10):
    """Composite N images as a deck of cards offset along +x,+y.

    Borderless, transparent canvas — the dark RA content of each card creates
    its own visual edge against the row tile background. No frames, no shadows.
    """
    n = len(image_paths)
    canvas_w = card_w + (n - 1) * dx
    canvas_h = card_h + (n - 1) * dy
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Paint back (i=0) to front (i=n-1)
    for i in range(n):
        img = Image.open(image_paths[i]).convert("RGBA").resize(
            (card_w, card_h), Image.LANCZOS)
        x = i * dx
        y = i * dy
        canvas.alpha_composite(img, (x, y))

    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _save_radarsplat_panels(panel_dir, picked_frame, scene, repo_root):
    """Save RadarSplat |RA| renders for the picked training frame and the
    held-out test frame."""
    rs_train_dir = os.path.join(repo_root, "baselines", "radarsplat",
                                 "results", scene, "train_frames",
                                 f"frame_{picked_frame}")
    if os.path.isdir(rs_train_dir):
        ra = np.load(os.path.join(rs_train_dir, "rendered_ra_cart.npy"))
        _save_ra(ra, os.path.join(panel_dir, "panel_ra_radarsplat_train.png"))

    rs_test_dir = os.path.join(repo_root, "baselines", "radarsplat",
                                "results", scene)
    rs_test_path = os.path.join(rs_test_dir, "rendered_ra_cart.npy")
    if os.path.exists(rs_test_path):
        ra = np.load(rs_test_path)
        _save_ra(ra, os.path.join(panel_dir, "panel_ra_radarsplat_test.png"))


def _save_train_frame_panels(run_dir, panel_dir, picked_frame, raw_radar_dir,
                              scene):
    """Generate ADC / CRP / RA panels (GT + Ours) for one training frame."""
    frame_dir = os.path.join(run_dir, "train_frames", f"frame_{picked_frame}")
    if not os.path.isdir(frame_dir):
        sys.exit(f"Train-frame dir not found: {frame_dir}")

    # ── Ours
    rp_ours = np.load(os.path.join(frame_dir, "rendered_rp_complex.npy"))
    adc_ours = np.fft.ifft(rp_ours, axis=-1)
    save_crp_heatmap(rp_ours, os.path.join(panel_dir, "panel_crp_ours.png"))
    _save_adc_traces(adc_ours,
                      os.path.join(panel_dir, "panel_adc_ours_train.png"),
                      color="#0e6b2c")
    ra_ours = np.load(os.path.join(frame_dir, "rendered_ra_cart.npy"))
    _save_ra(ra_ours, os.path.join(panel_dir, "panel_ra_ours_train.png"))

    # ── GT
    raw_path = os.path.join(raw_radar_dir, f"cascaded_frame_{picked_frame}.npy")
    if os.path.exists(raw_path):
        raw_adc = np.load(raw_path)        # (chirps, rx, tx, K) complex
        # Take chirp 0, compute CRP via range FFT
        chirp0 = raw_adc[0]                # (rx, tx, K)
        chirp0_txrx = np.transpose(chirp0, (1, 0, 2))  # (tx, rx, K)
        rp_gt = np.fft.fft(chirp0_txrx, axis=-1).astype(np.complex64)
        save_crp_heatmap(rp_gt, os.path.join(panel_dir, "panel_crp_gt.png"))
        _save_adc_traces(raw_adc,
                          os.path.join(panel_dir, "panel_adc_gt_train.png"),
                          color="#444444")
    else:
        print(f"  WARN: no raw ADC at {raw_path}")

    ra_gt = np.load(os.path.join(frame_dir, "gt_ra_cart.npy"))
    _save_ra(ra_gt, os.path.join(panel_dir, "panel_ra_gt_train.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="seq_1_frame_185")
    p.add_argument("--test_frame", type=int, default=185)
    p.add_argument("--train_frames", type=int, nargs="+",
                   default=[181, 182, 183, 184, 186, 187, 188, 189])
    p.add_argument("--rendering_train_frame", type=int, default=184,
                   help="Which training frame to use for the Rendering panel "
                        "(GT vs 3DPS, ADC/CRP/RA).")
    p.add_argument("--ours_dir",
                   default=os.path.join(_REPO, "mm25DGS_v5_v4/output_frame_nvs"))
    p.add_argument("--data_dir", default=os.path.join(_REPO, "data"))
    p.add_argument("--alignment_dir",
                   default=os.path.join(_REPO, "data/alignment_data"))
    p.add_argument("--output_dir",
                   default=os.path.join(_REPO, "output/teaser_panels"))
    args = p.parse_args()

    run_dir = os.path.join(args.ours_dir,
        f"{args.scene}_train8frames_1loops_test{args.test_frame}_loop0_pass2_N20000")
    panel_dir = os.path.join(args.output_dir, args.scene, "v4")
    os.makedirs(panel_dir, exist_ok=True)
    mesh_path = os.path.join(args.data_dir, args.scene, "scene", "mesh.ply")
    model_path = os.path.join(run_dir, "best_model.pt")
    raw_radar_dir = os.path.join(args.data_dir, args.scene, "radar")

    # Top row: primitive renders
    print("Rendering mesh-only…")
    render_mesh_only(mesh_path, args.scene, args.test_frame, args.alignment_dir,
                      os.path.join(panel_dir, "panel_mesh_only.png"))
    print("Rendering implicit/3DGS-style…")
    render_implicit_style(mesh_path, model_path, args.scene, args.test_frame,
                           args.alignment_dir,
                           os.path.join(panel_dir, "panel_implicit.png"))
    print("Rendering 3DPS oriented points…")
    render_3dps_points(mesh_path, model_path, args.scene, args.test_frame,
                        args.alignment_dir,
                        os.path.join(panel_dir, "panel_3dps_points.png"))

    # Bottom-row "Rendering": GT vs Ours for ADC/CRP/RA on a TRAINING frame
    print(f"Rendering panels for train frame {args.rendering_train_frame}…")
    _save_train_frame_panels(run_dir, panel_dir,
                              args.rendering_train_frame, raw_radar_dir,
                              args.scene)

    # RadarSplat |RA| panels (train frame + held-out test)
    print("Rendering RadarSplat |RA| panels…")
    _save_radarsplat_panels(panel_dir, args.rendering_train_frame,
                             args.scene, _REPO)

    # Bottom-row "Compression": stacked cards + compressed points
    print("Saving stacked-card RA strip + compressed point cloud…")
    train_card_pngs = []
    for F in args.train_frames:
        # Render clean PNG from the .npy (the linear PNG bundles axis labels).
        ra = np.load(os.path.join(run_dir, "train_frames",
                                   f"frame_{F}", "gt_ra_cart.npy"))
        tmp_path = os.path.join(panel_dir, f"_card_F{F}.png")
        _save_ra(ra, tmp_path)
        train_card_pngs.append(tmp_path)
    make_stacked_cards(train_card_pngs,
                        os.path.join(panel_dir, "panel_stacked_cards.png"))
    render_compressed_points(mesh_path, model_path,
                              os.path.join(panel_dir, "panel_compressed_pts.png"))

    # Bottom-row "NVS": held-out test |RA| (Ours and GT)
    print("Saving NVS held-out |RA|…")
    ra_test = np.load(os.path.join(run_dir, "rendered_test_ra_cart.npy"))
    _save_ra(ra_test, os.path.join(panel_dir, "panel_ra_test.png"))
    gt_ra_test = np.load(os.path.join(run_dir, "gt_test_ra_cart.npy"))
    _save_ra(gt_ra_test, os.path.join(panel_dir, "panel_ra_gt_test.png"))

    print(f"\nAll v4 panels under: {panel_dir}/")


if __name__ == "__main__":
    main()
