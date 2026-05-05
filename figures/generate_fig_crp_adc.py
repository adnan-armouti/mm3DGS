"""Qualitative figure for CRP / ADC product-agnostic evaluation.

Reads per-scene phase-corrected CRPs and ADCs saved by
``mmir.evaluation.eval_crp_adc`` and produces:

  fig_crp_adc_heatmaps.pdf   — 6 rows (scenes) × 4 cols
                                 (|CRP|_GT, |CRP|_pred,
                                  |ADC|_GT, |ADC|_pred)
  fig_crp_adc_iq_traces.pdf  — 6 rows × 2 cols (CRP, ADC) showing Re/Im
                                 traces for one representative virtual
                                 antenna per scene, GT vs phase-corrected
                                 prediction

Both are read directly from .npy arrays under
``output/crp_adc_eval/<scene>/{crp,adc}_{pred_corr,gt}.npy``.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER_SCENES = [
    "seq_0_frame_135", "seq_1_frame_185", "seq_1_frame_438",
    "seq_2_frame_105", "seq_2_frame_160", "seq_2_frame_300",
]


def _scene_short(name: str) -> str:
    return name.replace("seq_", "S").replace("_frame_", " F")


def _norm_db(x: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """Magnitude / its own peak in dB, clipped at `floor_db`.

    Matches ``mmir/data/ra_utils.py:save_ra_cartesian_png`` (scale='dB').
    """
    m = np.abs(x).astype(np.float64)
    p = m.max()
    if p < 1e-30:
        return np.zeros_like(m) + floor_db
    db = 20.0 * np.log10(np.clip(m / p, 10 ** (floor_db / 20.0), 1.0))
    return db


def _norm_minmax(x: np.ndarray) -> np.ndarray:
    """|x| min-max normalized to [0, 1].

    Matches the canonical `_minmax` used inside
    ``compute_cartesian_ra_metrics`` and the linear-scale branch of
    `save_ra_cartesian_png`. This is the EXACT same normalization used
    for the metric, so the visual is what's being scored.
    """
    m = np.abs(x).astype(np.float64)
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-30:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def _heatmap(ax, x, title, scale="dB", floor_db=-40.0, cmap="hot",
              xlabel=None, ylabel=None, fontsize=8):
    """Render |x| with the same logic as `save_ra_cartesian_png`.

    `scale='dB'`     : peak-normalize → 20 log10 → clip at floor_db.
    `scale='linear'` : min-max-normalize → [0, 1] (the metric domain).
    """
    if scale == "dB":
        disp = _norm_db(x, floor_db)
        vmin, vmax = floor_db, 0.0
    elif scale == "linear":
        disp = _norm_minmax(x)
        vmin, vmax = 0.0, 1.0
    else:
        raise ValueError(f"unknown scale {scale}")
    im = ax.imshow(disp, aspect="auto", origin="lower", cmap=cmap,
                    vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([]); ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize - 1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize - 1)
    return im


def fig_heatmaps(eval_dir: str, results_json: str, output_pdf: str,
                  scenes=PAPER_SCENES, start_bin: int = 15,
                  scale: str = "dB"):
    """6 rows × 4 cols: |CRP|_GT, |CRP|_pred, |ADC|_GT, |ADC|_pred.

    The CRP panels are CROPPED to range bins [start_bin:] — the same bins
    used in the metric. ADC panels show the full IFFT_range of the masked
    CRP (so the near-field rejection is reflected in the time domain too).

    `scale='dB'` (default): peak-normalize → 20 log10 → clip at -40 dB,
        matches `save_ra_cartesian_png(scale='dB')`.
    `scale='linear'`: min-max → [0, 1], matches the metric's normalization.
    """
    with open(results_json) as f:
        results = json.load(f)
    per = results["per_scene"]

    nrows = len(scenes)
    fig, axes = plt.subplots(nrows, 4, figsize=(7.0, 1.05 * nrows + 0.4),
                              dpi=200)
    if nrows == 1:
        axes = axes[None, :]

    for i, sc in enumerate(scenes):
        crp_gt = np.load(os.path.join(eval_dir, sc, "crp_gt.npy"))
        crp_pr = np.load(os.path.join(eval_dir, sc, "crp_pred_corr.npy"))
        adc_gt = np.load(os.path.join(eval_dir, sc, "adc_gt.npy"))
        adc_pr = np.load(os.path.join(eval_dir, sc, "adc_pred_corr.npy"))

        # Crop CRP panels to the evaluated range bins. ADC stays full-length.
        crp_gt = crp_gt[:, start_bin:]
        crp_pr = crp_pr[:, start_bin:]

        c_metrics = per[sc]["test"]["crp"]
        a_metrics = per[sc]["test"]["adc"]

        ylabel = _scene_short(sc) if i == nrows // 2 else None

        # Use independent dB scaling per column so structure is visible.
        _heatmap(axes[i, 0], crp_gt,
                  f"$|\\mathrm{{CRP}}|$ GT (bins {start_bin}:K)"
                  if i == 0 else "",
                  scale=scale,
                  ylabel=_scene_short(sc), fontsize=7,
                  xlabel=f"range bin (offset by {start_bin})"
                          if i == nrows - 1 else None)
        # CRP pred: image-style annotations (PSNR / SSIM make sense here)
        _heatmap(axes[i, 1], crp_pr,
                  (f"$|\\mathrm{{CRP}}|$ pred  "
                   f"$\\rho$={c_metrics['mag_corr']:.2f}  "
                   f"PSNR={c_metrics['mag_psnr']:.1f}  "
                   f"SSIM={c_metrics['mag_ssim']:.2f} | "
                   f"$|\\rho|^{{\\mathrm{{VR}}}}$={c_metrics['complex_corr_VR']:.2f}")
                  if i == 0 else
                  (f"$\\rho$={c_metrics['mag_corr']:.2f} "
                   f"PSNR={c_metrics['mag_psnr']:.1f} "
                   f"SSIM={c_metrics['mag_ssim']:.2f} | "
                   f"$|\\rho|^{{\\mathrm{{VR}}}}$={c_metrics['complex_corr_VR']:.2f}"),
                  scale=scale, fontsize=6.5,
                  xlabel=f"range bin (offset by {start_bin})"
                          if i == nrows - 1 else None)
        _heatmap(axes[i, 2], adc_gt,
                  f"$|\\mathrm{{ADC}}|$ GT" if i == 0 else "",
                  scale=scale, fontsize=7,
                  xlabel="ADC sample" if i == nrows - 1 else None)
        # ADC pred: time-series annotations (envelope ρ, |ρ|, σ_φ)
        _heatmap(axes[i, 3], adc_pr,
                  (f"$|\\mathrm{{ADC}}|$ pred  "
                   f"env $\\rho$={a_metrics['mag_corr']:.2f} | "
                   f"$|\\rho|^{{\\mathrm{{VR}}}}$={a_metrics['complex_corr_VR']:.2f} "
                   f"$\\sigma_\\phi$={a_metrics['phase_rmse_deg_VR']:.0f}°")
                  if i == 0 else
                  (f"env $\\rho$={a_metrics['mag_corr']:.2f} | "
                   f"$|\\rho|^{{\\mathrm{{VR}}}}$={a_metrics['complex_corr_VR']:.2f} "
                   f"$\\sigma_\\phi$={a_metrics['phase_rmse_deg_VR']:.0f}°"),
                  scale=scale, fontsize=6.5,
                  xlabel="ADC sample" if i == nrows - 1 else None)
        # set y-axis only on leftmost column
        axes[i, 0].set_ylabel(_scene_short(sc), fontsize=7)

    fig.suptitle(
        f"|CRP| (range bins {start_bin}:K, image-style metrics) and |ADC| "
        "(IFFT$_r$ of VR-corrected masked CRP, radar time-series metrics). "
        "GT range FFT uses Hann window (matches trainer). "
        f"Panels normalized to their own "
        f"{'peak (dB, $-40$\\,dB floor)' if scale == 'dB' else 'min-max $[0,1]$'}. "
        "$|\\rho|^{\\mathrm{VR}}$ is the complex correlation after joint per-VA "
        "$\\alpha$ + per-range $\\beta$ LS calibration removal; "
        "$\\sigma_\\phi$ is magnitude-weighted phase RMSE in degrees.",
        fontsize=6.5, y=1.005)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_pdf}")


def _pick_representative_va(crp_gt: np.ndarray, start_bin: int = 0) -> int:
    """Pick the virtual-antenna index with highest peak-range energy in GT.

    Energy is computed only over range bins >= start_bin so the near-field
    coupling at bins 0..14 doesn't dominate the choice.
    """
    energy = np.max(np.abs(crp_gt[:, start_bin:]), axis=1)
    return int(np.argmax(energy))


def fig_iq_traces(eval_dir: str, results_json: str, output_pdf: str,
                   scenes=PAPER_SCENES, start_bin: int = 15):
    """6 rows × 2 cols: CRP and ADC I/Q traces for one VA per scene.

    Each subplot overlays GT real (solid blue), GT imag (solid red),
    pred real (dashed blue), pred imag (dashed red).  Shows that after
    per-range-bin azimuth-DC phase correction the relative phase of the
    rendered samples matches the GT.
    """
    with open(results_json) as f:
        results = json.load(f)
    per = results["per_scene"]

    nrows = len(scenes)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.0, 1.4 * nrows + 0.4),
                              dpi=200, sharex="col")
    if nrows == 1:
        axes = axes[None, :]

    for i, sc in enumerate(scenes):
        crp_gt = np.load(os.path.join(eval_dir, sc, "crp_gt.npy"))
        crp_pr = np.load(os.path.join(eval_dir, sc, "crp_pred_corr.npy"))
        adc_gt = np.load(os.path.join(eval_dir, sc, "adc_gt.npy"))
        adc_pr = np.load(os.path.join(eval_dir, sc, "adc_pred_corr.npy"))

        va = _pick_representative_va(crp_gt, start_bin=start_bin)

        # CRP traces — cropped to the bins actually used in the metric.
        gt = crp_gt[va, start_bin:]
        pr = crp_pr[va, start_bin:]
        # Normalize each pair to GT peak |.| so axes align
        norm = max(np.max(np.abs(gt)), 1e-30)
        ax = axes[i, 0]
        x = np.arange(start_bin, crp_gt.shape[1])
        ax.plot(x, gt.real / norm, color="#1f77b4", lw=0.8, label="GT Re")
        ax.plot(x, gt.imag / norm, color="#d62728", lw=0.8, label="GT Im")
        ax.plot(x, pr.real / norm, color="#1f77b4", lw=0.8,
                 ls="--", alpha=0.85, label="pred Re")
        ax.plot(x, pr.imag / norm, color="#d62728", lw=0.8,
                 ls="--", alpha=0.85, label="pred Im")
        ax.set_xlim(start_bin, crp_gt.shape[1])
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2, lw=0.4)
        c_psnr_VR = per[sc]["test"]["crp"]["complex_psnr_VR"]
        title = (f"{_scene_short(sc)}  CRP  (VA {va},  "
                  f"PSNR$^{{\\mathrm{{VR}}}}_{{\\mathbb{{C}}}}$"
                  f"={c_psnr_VR:.1f}\\,dB)")
        ax.set_title(title, fontsize=7)
        if i == 0:
            ax.legend(fontsize=6, ncol=4, loc="upper right",
                      handlelength=2.5, handletextpad=0.4,
                      columnspacing=1.0)
        if i == nrows - 1:
            ax.set_xlabel("range bin", fontsize=7)
        ax.set_ylabel("amplitude (norm.)", fontsize=7)

        # ADC traces (sample index)
        gt_a = adc_gt[va]
        pr_a = adc_pr[va]
        norm_a = max(np.max(np.abs(gt_a)), 1e-30)
        ax = axes[i, 1]
        x = np.arange(adc_gt.shape[1])
        ax.plot(x, gt_a.real / norm_a, color="#1f77b4", lw=0.8)
        ax.plot(x, gt_a.imag / norm_a, color="#d62728", lw=0.8)
        ax.plot(x, pr_a.real / norm_a, color="#1f77b4", lw=0.8,
                 ls="--", alpha=0.85)
        ax.plot(x, pr_a.imag / norm_a, color="#d62728", lw=0.8,
                 ls="--", alpha=0.85)
        ax.set_xlim(0, adc_gt.shape[1])
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2, lw=0.4)
        a_corr_VR = per[sc]["test"]["adc"]["complex_corr_VR"]
        a_phase_VR = per[sc]["test"]["adc"]["phase_rmse_deg_VR"]
        ax.set_title(
            f"{_scene_short(sc)}  ADC  (VA {va},  "
            f"$|\\rho|^{{\\mathrm{{VR}}}}$={a_corr_VR:.2f},  "
            f"$\\sigma_\\phi$={a_phase_VR:.0f}°)",
            fontsize=7)
        if i == nrows - 1:
            ax.set_xlabel("ADC sample", fontsize=7)

    fig.suptitle(
        "Phase-corrected I/Q traces: GT (solid) vs prediction (dashed) at "
        "the highest-energy virtual antenna per scene.",
        fontsize=8, y=0.997)

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="output/crp_adc_eval")
    ap.add_argument("--results_json",
                     default="output/crp_adc_eval/results.json")
    ap.add_argument("--output_dir",
                     default="output/crp_adc_eval/figures")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # dB heatmaps (peak-normalized; matches save_ra_cartesian_png(scale='dB'))
    fig_heatmaps(args.eval_dir, args.results_json,
                  os.path.join(args.output_dir,
                                "fig_crp_adc_heatmaps_dB.pdf"),
                  scale="dB")
    # Linear heatmaps (min-max [0,1]; the metric domain itself)
    fig_heatmaps(args.eval_dir, args.results_json,
                  os.path.join(args.output_dir,
                                "fig_crp_adc_heatmaps_linear.pdf"),
                  scale="linear")
    fig_iq_traces(args.eval_dir, args.results_json,
                   os.path.join(args.output_dir, "fig_crp_adc_iq_traces.pdf"))


if __name__ == "__main__":
    main()
