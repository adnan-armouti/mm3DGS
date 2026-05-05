"""NeurIPS-style cross-method, cross-product results table for the main paper.

Single combined results table that replaces the per-scene RA tables (now in
the supplement). Headline message: only 3DPS produces correct CRP and ADC;
the three optical-NVS-derived baselines have nothing to report on those
columns and on Complex metrics — signaled by `--`.

Layout (12 columns, single row section per split with 4 method rows each):

  Method | RA cart (Corr PSNR SSIM RMSE)
         | CRP (Corr PSNR SSIM)
         | CRP Complex (|ρ|, σ_φ)
         | ADC (env ρ_|·|)
         | ADC Complex (σ_φ)            -- |ρ| is shown only under CRP since
                                           it's identical between CRP and ADC
                                           by Parseval's theorem (unitary FFT)

Method order: prior work first (RadarSplat, Radar Fields, DART), 3DPS at the
bottom in bold (NeurIPS convention).

Header-text highlighters: yellow on Magnitude / Env., blue on Complex.

Wrapped in \\resizebox{\\textwidth}{!}{...} to guarantee page-width fit.

Reads:
  - mm25DGS_v5_v4/output_frame_nvs/...                    (3DPS RA metrics)
  - baselines/{dart,radarsplat,radarfields}/...           (baseline RA)
  - output/crp_adc_eval/results.json                       (3DPS CRP/ADC)

Writes:
  latex/.../tables/crp_adc_results.tex
"""

import argparse
import json
import math
import os
import sys

# Reuse the existing loaders from generate_tables.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_tables import (
    SCENES, METHODS,
    load_ours_test_metrics, load_ours_train_metrics,
    load_baseline_metrics, load_baseline_train_metrics,
)


METHOD_LABEL = {
    "ours":        "\\textbf{3DPS (Ours)}",
    "radarsplat":  "RadarSplat~\\cite{kung2025radarsplat}",
    "radarfields": "Radar Fields~\\cite{10.1145/3641519.3657510}",
    "dart":        "DART~\\cite{huang2024dart}",
}
# NeurIPS convention: prior work first, ours last (and bolded)
METHOD_ORDER = ("dart", "radarfields", "radarsplat", "ours")


CRP_ADC_RESULTS = "output/crp_adc_eval/results.json"
DEFAULT_OURS_DIR = "mm25DGS_v5_v4/output_frame_nvs"
DEFAULT_BASELINES_DIR = "baselines"
DEFAULT_OUT = (
    "latex/NeurIPS_2026_unpacked/"
    "Physically_Grounded_Novel_View_Synthesis_for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering/"
    "tables/crp_adc_results.tex"
)


# ---------------------------------------------------------------------------
# Aggregation across scenes
# ---------------------------------------------------------------------------

def _scene_mean(values):
    vs = [v for v in values
          if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vs) / len(vs)) if vs else None


def aggregate_ra_test(ours_dir: str, baselines_dir: str) -> dict:
    out = {m: {} for m in METHODS}
    for m in METHODS:
        per_scene = {}
        for sc in SCENES:
            if m == "ours":
                d = load_ours_test_metrics(sc, ours_dir)
            else:
                d = load_baseline_metrics(m, sc, baselines_dir)
            if d is not None:
                per_scene[sc] = d
        for k in ("cart_corr", "cart_psnr", "cart_ssim", "cart_rmse"):
            out[m][k] = _scene_mean([per_scene.get(sc, {}).get(k)
                                       for sc in SCENES])
    return out


def aggregate_ra_train(ours_dir: str, baselines_dir: str) -> dict:
    out = {m: {} for m in METHODS}
    for m in METHODS:
        per_scene = {}
        for sc in SCENES:
            if m == "ours":
                d = load_ours_train_metrics(sc, ours_dir)
            else:
                d = load_baseline_train_metrics(m, sc, baselines_dir)
            if d is None:
                continue
            per_scene[sc] = {k: v["mean"] for k, v in d.items()
                              if isinstance(v, dict)}
        for k in ("cart_corr", "cart_psnr", "cart_ssim", "cart_rmse"):
            out[m][k] = _scene_mean([per_scene.get(sc, {}).get(k)
                                       for sc in SCENES])
    return out


def aggregate_crp_adc(crp_adc_path: str) -> dict:
    """Pull train/test aggregate CRP and ADC for 3DPS only.

    Note on what's identical vs different between CRP and ADC:
      |ρ|     : IDENTICAL (Parseval — unitary FFT preserves the inner product
                 ratio).
      PSNR_C  : DIFFERENT — depends on max(|·|), which IFFT changes per channel.
      σ_φ     : DIFFERENT — per-cell phase differences and |gt|² weights both
                 change after IFFT.
    Hence we report |ρ| once (under CRP) and σ_φ separately for CRP and ADC.
    """
    with open(crp_adc_path) as f:
        r = json.load(f)
    agg = r["aggregate"]
    out = {"train": {}, "test": {}}
    for split in ("train", "test"):
        b = agg[split]
        out[split] = {
            "crp_corr":   b["crp"]["mag_corr"]["mean"],
            "crp_psnr":   b["crp"]["mag_psnr"]["mean"],
            "crp_ssim":   b["crp"]["mag_ssim"]["mean"],
            "crp_rho":    b["crp"]["complex_corr_VR"]["mean"],
            "crp_psnr_c": b["crp"]["complex_psnr_VR"]["mean"],
            "adc_env":    b["adc"]["mag_corr"]["mean"],
            "adc_phase":  b["adc"]["phase_rmse_deg_VR"]["mean"],
        }
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(v, spec):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return spec.format(v)


def _row(method_label: str, ra: dict, crp_adc=None, bold_ours: bool = False) -> str:
    """One method row.

    `crp_adc` is None for baselines (they have nothing to report on
    CRP / ADC / Complex columns — those become `--`).
    """
    cells = [method_label]
    cells.append(fmt(ra.get("cart_corr"),  "{:.3f}"))
    cells.append(fmt(ra.get("cart_psnr"),  "{:.1f}"))
    cells.append(fmt(ra.get("cart_ssim"),  "{:.3f}"))
    cells.append(fmt(ra.get("cart_rmse"),  "{:.4f}"))
    if crp_adc is not None:
        cells.append(fmt(crp_adc["crp_corr"],   "{:.3f}"))
        cells.append(fmt(crp_adc["crp_psnr"],   "{:.1f}"))
        cells.append(fmt(crp_adc["crp_ssim"],   "{:.3f}"))
        cells.append(fmt(crp_adc["crp_rho"],    "{:.3f}"))
        cells.append(fmt(crp_adc["crp_psnr_c"], "{:.1f}"))
        cells.append(fmt(crp_adc["adc_env"],    "{:.3f}"))
        cells.append(fmt(crp_adc["adc_phase"],  "{:.1f}"))
    else:
        cells.extend(["--"] * 7)
    if bold_ours:
        cells = [cells[0]] + [f"\\textbf{{{c}}}" if c != "--" else c
                                for c in cells[1:]]
    return " & ".join(cells) + " \\\\"


def make_table(ra_train, ra_test, crp_adc) -> str:
    L = []
    L.append("% Auto-generated by figures/generate_crp_adc_paper_table.py")
    L.append("\\begin{table*}[t]")
    L.append("\\centering")
    L.append("\\caption{\\textbf{Cross-method, cross-product results on six "
              "ColoRadar scenes (mean across scenes).} \\colorbox{magbg}{Magnitude}: "
              "Pearson Corr / PSNR / SSIM / MSE on min-max-normalized $|\\cdot|$. "
              "\\colorbox{vrbg}{Complex}: $|\\rho|$, complex PSNR (CRP), and "
              "magnitude-weighted phase RMSE $\\sigma_\\phi$ (ADC; degrees; $104^\\circ\\!=$\\,random) "
              "after absolute phase calibration. $|\\rho|$ is reported once "
              "(CRP $\\equiv$ ADC by Parseval). Baselines emit magnitude-only "
              "RA, so CRP/ADC/Complex columns are marked `--'. Per-scene "
              "breakdowns in supplement Tables~\\ref{tab:test_ra}--\\ref{tab:crp_adc_test}.}")
    L.append("\\label{tab:results}")
    L.append("\\setlength{\\tabcolsep}{2.5pt}")
    L.append("\\resizebox{\\textwidth}{!}{")
    # 1 (Method) + 4 (RA) + 3 (CRP mag) + 2 (CRP cmplx) + 1 (ADC env) + 1 (ADC cmplx) = 12 cols
    L.append("\\begin{tabular}{l | cccc | ccc | cc | c | c}")
    L.append("\\toprule")
    # Top header — domain group
    L.append("& \\multicolumn{4}{c|}{\\textbf{RA cart}} "
              "& \\multicolumn{5}{c|}{\\textbf{CRP}} "
              "& \\multicolumn{2}{c}{\\textbf{ADC}} \\\\")
    L.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-10}\\cmidrule(lr){11-12}")
    # Sub-header — metric family with highlighter on label text
    L.append("& \\multicolumn{4}{c|}{\\colorbox{magbg}{\\textbf{Magnitude}}} "
              "& \\multicolumn{3}{c|}{\\colorbox{magbg}{\\textbf{Magnitude}}} "
              "& \\multicolumn{2}{c|}{\\colorbox{vrbg}{\\textbf{Complex}}} "
              "& \\colorbox{magbg}{\\textbf{Env.}} "
              "& \\colorbox{vrbg}{\\textbf{Complex}} \\\\")
    # Metric names
    L.append("Method & "
              "Corr & PSNR & SSIM & RMSE & "
              "Corr & PSNR & SSIM & "
              "$|\\rho|$ & PSNR$_{\\mathbb{C}}$ & "
              "$\\rho_{|\\cdot|}$ & "
              "$\\sigma_\\phi$ \\\\")
    L.append("\\midrule")

    # ── Train Mean ──
    L.append("\\multicolumn{12}{l}{\\textbf{Train Mean}} \\\\")
    for m in METHOD_ORDER:
        crp_adc_block = crp_adc["train"] if m == "ours" else None
        L.append(_row(METHOD_LABEL[m], ra_train[m],
                       crp_adc_block, bold_ours=(m == "ours")))
    L.append("\\midrule")

    # ── Test Mean ──
    L.append("\\multicolumn{12}{l}{\\textbf{Test Mean}} \\\\")
    for m in METHOD_ORDER:
        crp_adc_block = crp_adc["test"] if m == "ours" else None
        L.append(_row(METHOD_LABEL[m], ra_test[m],
                       crp_adc_block, bold_ours=(m == "ours")))

    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("}")
    L.append("\\end{table*}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir",        default=DEFAULT_OURS_DIR)
    ap.add_argument("--baselines_dir",   default=DEFAULT_BASELINES_DIR)
    ap.add_argument("--crp_adc_results", default=CRP_ADC_RESULTS)
    ap.add_argument("--output",          default=DEFAULT_OUT)
    args = ap.parse_args()

    ra_train = aggregate_ra_train(args.ours_dir, args.baselines_dir)
    ra_test = aggregate_ra_test(args.ours_dir, args.baselines_dir)
    crp_adc = aggregate_crp_adc(args.crp_adc_results)

    tex = make_table(ra_train, ra_test, crp_adc)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(tex)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
