"""Generate Supplement Table 8 (`tab:ablations`) for the NeurIPS paper.

Walks the ablation output tree:

    mm25DGS_v5_v4/output_ablations/<tier>/<axis>/<config>/<scene_run_tag>/

For each config: aggregates per-scene `results.json` + `metrics_train.json`
into mean test |RA| Corr/PSNR/SSIM/RMSE, mean train |RA| Corr, mean
wall-clock; pulls CRP / ADC envelope correlation from
`output/crp_adc_eval_ablations/<tier>/<axis>/<config>/results.json`.

Emits the populated LaTeX table to:

    latex/.../tables/ablations.tex

The default 3DPS row (N=20k / 8 views / adaptive on / full LiDAR init /
MIMO factored / L=15 / detach on / lambda_pos=100 / closed-form BSDF) is
read from the canonical run directory ``mm25DGS_v5_v4/output_frame_nvs/``
+ the existing CRP/ADC eval at ``output/crp_adc_eval/results.json``.

Usage::

    python -m figures.generate_ablation_table

Re-uses the loaders in :mod:`figures.generate_tables`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_tables import SCENES  # canonical 6-scene order


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_OURS_DIR = os.path.join(PROJECT_ROOT, "mm25DGS_v5_v4", "output_frame_nvs")
DEFAULT_ABLATIONS_DIR = os.path.join(PROJECT_ROOT, "mm25DGS_v5_v4", "output_ablations")
DEFAULT_CRP_ADC_DEFAULT = os.path.join(PROJECT_ROOT, "output", "crp_adc_eval", "results.json")
DEFAULT_CRP_ADC_ABLATIONS_ROOT = os.path.join(PROJECT_ROOT, "output", "crp_adc_eval_ablations")
DEFAULT_OUTPUT_TEX = os.path.join(
    PROJECT_ROOT, "latex", "NeurIPS_2026_unpacked",
    "Physically_Grounded_Novel_View_Synthesis_for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering",
    "tables", "ablations.tex")


# ---------------------------------------------------------------------------
# Run-tag conventions
# ---------------------------------------------------------------------------
# The trainer auto-builds run dirs as
#   <scene>_train{n}frames_{n_loops}loops_test{F}_loop0_pass2_N{N}[suffixes]
# Ablation runs override --output_dir, so each run lives at
#   output_ablations/<tier>/<axis>/<config>/<scene>_train8frames_1loops_test{F}_loop0_pass2_N{N}/
# We enumerate scene-run-dirs under <config>/ rather than guessing.

def _find_scene_run_dir(config_dir: str, scene: str) -> Optional[str]:
    """Locate the scene's run directory inside an ablation config dir.

    Tries two layouts:
      (a) ``<config>/<scene>``  — when the runner uses the bare scene
          name as the leaf directory.
      (b) ``<config>/<scene>_train*frames_*loops_test*_loop*_pass*_N*``
          — when the trainer's auto-tag is preserved.
    Returns None if no match.
    """
    a = os.path.join(config_dir, scene)
    if os.path.isdir(a) and os.path.isfile(os.path.join(a, "results.json")):
        return a
    pat = os.path.join(config_dir, f"{scene}_*")
    for cand in sorted(glob.glob(pat), key=os.path.getmtime, reverse=True):
        if os.path.isfile(os.path.join(cand, "results.json")):
            return cand
    return None


# ---------------------------------------------------------------------------
# Per-config aggregation
# ---------------------------------------------------------------------------

def _safe_mean(vals):
    vs = [v for v in vals
          if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return (sum(vs) / len(vs)) if vs else None


def aggregate_config(config_dir: str, crp_adc_results_path: Optional[str]) -> dict:
    """Aggregate per-scene results for a single ablation config.

    Reads:
      <config_dir>/<scene_run>/results.json         -> test |RA| metrics + elapsed
      <config_dir>/<scene_run>/metrics_train.json   -> train |RA| Corr mean
      <crp_adc_results_path>                        -> CRP / ADC env corr (test mean)

    Returns
    -------
    dict with keys:
      test_corr, test_psnr, test_ssim, test_rmse  : float | None
      train_corr                                   : float | None
      crp_corr, adc_env                            : float | None
      elapsed_min                                  : float | None
      n_scenes                                     : int
    """
    test_corr_l, test_psnr_l, test_ssim_l, test_rmse_l = [], [], [], []
    train_corr_l, elapsed_l = [], []

    for scene in SCENES:
        run = _find_scene_run_dir(config_dir, scene)
        if run is None:
            continue
        rp = os.path.join(run, "results.json")
        try:
            d = json.load(open(rp))
        except Exception:
            continue
        test_corr_l.append(d.get("final_test_cart_corr"))
        test_psnr_l.append(d.get("final_test_cart_psnr"))
        test_ssim_l.append(d.get("final_test_cart_ssim"))
        test_rmse_l.append(d.get("final_test_cart_rmse"))
        elapsed_l.append(d.get("elapsed_s"))
        tp = os.path.join(run, "metrics_train.json")
        if os.path.isfile(tp):
            try:
                td = json.load(open(tp))
                tc = td.get("ra_corr")
                if isinstance(tc, dict):
                    train_corr_l.append(tc.get("mean"))
            except Exception:
                pass

    crp_corr = adc_env = None
    if crp_adc_results_path is not None and os.path.isfile(crp_adc_results_path):
        try:
            cd = json.load(open(crp_adc_results_path))
            tb = cd.get("aggregate", {}).get("test", {})
            crp_corr = tb.get("crp", {}).get("mag_corr", {}).get("mean")
            adc_env = tb.get("adc", {}).get("mag_corr", {}).get("mean")
        except Exception:
            pass

    elapsed_min = None
    em = _safe_mean(elapsed_l)
    if em is not None:
        elapsed_min = em / 60.0

    return {
        "test_corr": _safe_mean(test_corr_l),
        "test_psnr": _safe_mean(test_psnr_l),
        "test_ssim": _safe_mean(test_ssim_l),
        "test_rmse": _safe_mean(test_rmse_l),
        "train_corr": _safe_mean(train_corr_l),
        "crp_corr": crp_corr,
        "adc_env": adc_env,
        "elapsed_min": elapsed_min,
        "n_scenes": len(test_corr_l),
    }


# ---------------------------------------------------------------------------
# Default 3DPS row (canonical paper-results layout)
# ---------------------------------------------------------------------------

def aggregate_default(ours_dir: str, crp_adc_default_path: str) -> dict:
    """Aggregate the default 3DPS recipe row for the 'reference' line.

    Walks ``mm25DGS_v5_v4/output_frame_nvs/<scene>_..._pass2_N20000`` runs
    using the same loader logic as the ablation tree. Pulls CRP/ADC means
    from the canonical CRP/ADC eval results.
    """
    test_corr_l, test_psnr_l, test_ssim_l, test_rmse_l = [], [], [], []
    train_corr_l, elapsed_l = [], []
    for scene in SCENES:
        # Same patterns as figures/generate_tables.py: prefer the bare
        # ..._pass2_N20000 dir; otherwise the longest-suffix variant.
        runs = sorted(
            glob.glob(os.path.join(ours_dir, f"{scene}_*_pass2_N20000")),
            key=os.path.getmtime, reverse=True)
        runs = [r for r in runs if "_initC" not in r and "_initB" not in r
                 and "_initA" not in r and "_p4d" not in r]
        if not runs:
            continue
        rp = os.path.join(runs[0], "results.json")
        if not os.path.isfile(rp):
            continue
        d = json.load(open(rp))
        test_corr_l.append(d.get("final_test_cart_corr", d.get("final_test_cc")))
        test_psnr_l.append(d.get("final_test_cart_psnr"))
        test_ssim_l.append(d.get("final_test_cart_ssim"))
        test_rmse_l.append(d.get("final_test_cart_rmse"))
        elapsed_l.append(d.get("elapsed_s"))
        tp = os.path.join(runs[0], "metrics_train.json")
        if os.path.isfile(tp):
            td = json.load(open(tp))
            tc = td.get("ra_corr")
            if isinstance(tc, dict):
                train_corr_l.append(tc.get("mean"))

    crp_corr = adc_env = None
    if os.path.isfile(crp_adc_default_path):
        cd = json.load(open(crp_adc_default_path))
        tb = cd.get("aggregate", {}).get("test", {})
        crp_corr = tb.get("crp", {}).get("mag_corr", {}).get("mean")
        adc_env = tb.get("adc", {}).get("mag_corr", {}).get("mean")

    em = _safe_mean(elapsed_l)
    return {
        "test_corr": _safe_mean(test_corr_l),
        "test_psnr": _safe_mean(test_psnr_l),
        "test_ssim": _safe_mean(test_ssim_l),
        "test_rmse": _safe_mean(test_rmse_l),
        "train_corr": _safe_mean(train_corr_l),
        "crp_corr": crp_corr,
        "adc_env": adc_env,
        "elapsed_min": (em / 60.0) if em is not None else None,
        "n_scenes": len(test_corr_l),
    }


# ---------------------------------------------------------------------------
# Row spec — one entry per row of the ablation table (Tier 1 + Tier 2 only;
# Tier 3 deferred). Each tuple is:
#   (display_label, group_id, axis_dir, config_subdir, crp_adc_subdir)
# Where:
#   display_label : LaTeX cell text for the leftmost column
#   group_id      : section header to print before this row (None = skip)
#   axis_dir      : <tier>/<axis_name> subpath inside output_ablations/
#   config_subdir : <config> directory inside axis_dir/
#   crp_adc_subdir: <config> directory inside output/crp_adc_eval_ablations/<tier>/<axis>/
# Defaults bolded (special row); rest of rows are non-bolded.
# ---------------------------------------------------------------------------

ROWS = [
    # Group: defaults (bolded reference row)
    ("\\textbf{3DPS default} ($N{=}20$k, 8 views, adaptive on, full init, "
     "MIMO factored, $L{=}15$, detach on, $\\lambda_{\\mathrm{pos}}{=}100$)",
     "default", None, None, None),

    # Tier 1 axis 1 — point count N
    ("$N{=}\\phantom{0}2{,}000$", "tier1_N", "tier1/point_count_N", "N_2k",  "tier1/point_count_N/N_2k"),
    ("$N{=}\\phantom{0}5{,}000$", "tier1_N", "tier1/point_count_N", "N_5k",  "tier1/point_count_N/N_5k"),
    ("$N{=}10{,}000$",            "tier1_N", "tier1/point_count_N", "N_10k", "tier1/point_count_N/N_10k"),
    ("$N{=}50{,}000$",            "tier1_N", "tier1/point_count_N", "N_50k", "tier1/point_count_N/N_50k"),

    # Tier 1 axis 2 — number of training views
    ("2 train views",  "tier1_views", "tier1/num_train_views", "views_2", "tier1/num_train_views/views_2"),
    ("4 train views",  "tier1_views", "tier1/num_train_views", "views_4", "tier1/num_train_views/views_4"),
    ("6 train views",  "tier1_views", "tier1/num_train_views", "views_6", "tier1/num_train_views/views_6"),

    # Tier 1 axis 3 — adaptive density off
    ("no adaptive density",
     "tier1_density", "tier1/adaptive_density", "off", "tier1/adaptive_density/off"),

    # Tier 1 axis 4 — LiDAR init stages
    ("no azimuth-cone cull",
     "tier1_init", "tier1/lidar_init", "no_cull",          "tier1/lidar_init/no_cull"),
    ("no occlusion ray-cast",
     "tier1_init", "tier1/lidar_init", "no_occlusion",     "tier1/lidar_init/no_occlusion"),
    ("no cosine resample",
     "tier1_init", "tier1/lidar_init", "no_cosine_resample","tier1/lidar_init/no_cosine_resample"),
    ("no FPS (random sub-sample)",
     "tier1_init", "tier1/lidar_init", "no_fps",           "tier1/lidar_init/no_fps"),

    # Tier 1 axis 5 — MIMO factorization off
    ("no MIMO factorization (PyTorch fallback)",
     "tier1_mimo", "tier1/mimo_factorization", "off", "tier1/mimo_factorization/off"),

    # Tier 2 axis 6 — PSF kernel L
    ("PSF $L{=}5$",  "tier2_psf", "tier2/psf_kernel_L", "L_5",  "tier2/psf_kernel_L/L_5"),
    ("PSF $L{=}9$",  "tier2_psf", "tier2/psf_kernel_L", "L_9",  "tier2/psf_kernel_L/L_9"),
    ("PSF $L{=}21$", "tier2_psf", "tier2/psf_kernel_L", "L_21", "tier2/psf_kernel_L/L_21"),
    ("PSF $L{=}25$", "tier2_psf", "tier2/psf_kernel_L", "L_25", "tier2/psf_kernel_L/L_25"),

    # Tier 2 axis 7 — phase detach off
    ("no carrier-phase detach",
     "tier2_detach", "tier2/phase_detach", "off", "tier2/phase_detach/off"),

    # Tier 2 axis 8 — λ_pos
    ("$\\lambda_{\\mathrm{pos}}{=}0$",    "tier2_lpos", "tier2/lambda_pos", "lpos_0",    "tier2/lambda_pos/lpos_0"),
    ("$\\lambda_{\\mathrm{pos}}{=}1$",    "tier2_lpos", "tier2/lambda_pos", "lpos_1",    "tier2/lambda_pos/lpos_1"),
    ("$\\lambda_{\\mathrm{pos}}{=}1000$", "tier2_lpos", "tier2/lambda_pos", "lpos_1000", "tier2/lambda_pos/lpos_1000"),
]


GROUP_RULES = {
    "default":        ("Reference",                    True),
    "tier1_N":        ("Tier 1 — point count $N$",     True),
    "tier1_views":    ("Tier 1 — training views",      True),
    "tier1_density":  ("Tier 1 — adaptive density",    True),
    "tier1_init":     ("Tier 1 — LiDAR init stages",   True),
    "tier1_mimo":     ("Tier 1 — MIMO factorization",  True),
    "tier2_psf":      ("Tier 2 — PSF kernel $L$",      True),
    "tier2_detach":   ("Tier 2 — carrier-phase detach", True),
    "tier2_lpos":     ("Tier 2 — position anchor $\\lambda_{\\mathrm{pos}}$", True),
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(v, spec, dash_for_none=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--" if dash_for_none else ""
    return spec.format(v)


def make_row(label: str, agg: dict, bold: bool = False) -> str:
    cells = [label,
             fmt(agg.get("test_corr"),  "{:.3f}"),
             fmt(agg.get("test_psnr"),  "{:.1f}"),
             fmt(agg.get("test_ssim"),  "{:.3f}"),
             fmt(agg.get("test_rmse"),  "{:.4f}"),
             fmt(agg.get("crp_corr"),   "{:.3f}"),
             fmt(agg.get("adc_env"),    "{:.3f}"),
             fmt(agg.get("train_corr"), "{:.3f}"),
             fmt(agg.get("elapsed_min"),"{:.1f}")]
    if bold:
        cells = [cells[0]] + [f"\\textbf{{{c}}}" if c != "--" else c for c in cells[1:]]
    return " & ".join(cells) + " \\\\"


def build_table(default_agg: dict, ablation_aggs: dict) -> str:
    """Build the populated ablations.tex contents.

    ``ablation_aggs`` keys are (axis_dir, config_subdir) tuples → agg dict.
    """
    L = []
    L.append("% Auto-generated by figures/generate_ablation_table.py")
    L.append("\\begin{table}[h]")
    L.append("\\centering")
    L.append("\\caption{\\textbf{Ablation sweep on the 6-scene benchmark.} "
             "Mean held-out test $|\\mathrm{RA}|$ Corr / PSNR / SSIM / RMSE, "
             "test $|\\mathrm{CRP}|$ Corr, ADC envelope correlation, mean "
             "training $|\\mathrm{RA}|$ Corr, and mean wall-clock per scene "
             "on a single RTX~4090. Default 3DPS configuration in the top "
             "row. Each subsequent row flips a single design choice; all "
             "other knobs match the default. Tier 3 (NeurIPS appendix) "
             "deferred. CRP / ADC columns require \\texttt{rendered\\_test\\_rp"
             "\\_complex.npy} from the renderer; rows where a config has not "
             "yet been run show `--'.}")
    L.append("\\label{tab:ablations}")
    L.append("\\small")
    L.append("\\setlength{\\tabcolsep}{4pt}")
    L.append("\\resizebox{\\linewidth}{!}{")
    L.append("\\begin{tabular}{l|cccc|cc|c|c}")
    L.append("\\toprule")
    L.append("\\textbf{Configuration} & "
             "\\multicolumn{4}{c|}{\\textbf{Test $|\\mathrm{RA}|$}} & "
             "\\multicolumn{2}{c|}{\\textbf{Complex / Env.}} & "
             "\\textbf{Train} & \\textbf{Time} \\\\")
    L.append("\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}")
    L.append("& Corr $\\uparrow$ & PSNR $\\uparrow$ & SSIM $\\uparrow$ & "
             "RMSE $\\downarrow$ & "
             "$|$CRP$|$ Corr $\\uparrow$ & ADC env. $\\uparrow$ & "
             "Corr $\\uparrow$ & (min) \\\\")

    last_group = None
    for (label, group, axis_dir, cfg, _crp_sub) in ROWS:
        if group != last_group:
            header_text, draw_rule = GROUP_RULES.get(group, (group, True))
            if draw_rule:
                L.append("\\midrule")
            L.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{header_text}}}}} \\\\")
            last_group = group

        if group == "default":
            L.append(make_row(label, default_agg, bold=True))
            continue
        agg = ablation_aggs.get((axis_dir, cfg), {})
        L.append(make_row(label, agg))

    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("}")
    L.append("\\end{table}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", default=DEFAULT_OURS_DIR,
                    help="Canonical 3DPS results dir (for the default row).")
    ap.add_argument("--ablations_dir", default=DEFAULT_ABLATIONS_DIR,
                    help="Root of the ablation runs (output_ablations/).")
    ap.add_argument("--crp_adc_default", default=DEFAULT_CRP_ADC_DEFAULT,
                    help="CRP/ADC eval results.json for the default row.")
    ap.add_argument("--crp_adc_ablations_root", default=DEFAULT_CRP_ADC_ABLATIONS_ROOT,
                    help="Root of CRP/ADC eval outputs for ablations.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT_TEX,
                    help="Path to write ablations.tex.")
    args = ap.parse_args()

    print(f"[default] ours_dir = {args.ours_dir}")
    print(f"[default] crp_adc  = {args.crp_adc_default}")
    default_agg = aggregate_default(args.ours_dir, args.crp_adc_default)
    print(f"  default agg: {default_agg}")

    ablation_aggs: dict = {}
    seen_axes = set()
    for (_, _, axis_dir, cfg, crp_sub) in ROWS:
        if axis_dir is None:
            continue
        config_dir = os.path.join(args.ablations_dir, axis_dir, cfg)
        crp_path = os.path.join(args.crp_adc_ablations_root, crp_sub, "results.json")
        agg = aggregate_config(config_dir, crp_path)
        ablation_aggs[(axis_dir, cfg)] = agg
        if axis_dir not in seen_axes:
            print(f"\n[axis] {axis_dir}")
            seen_axes.add(axis_dir)
        print(f"  [{cfg:>20}] n_scenes={agg['n_scenes']:>1} "
              f"test_corr={fmt(agg['test_corr'], '{:.3f}'):>5} "
              f"crp={fmt(agg['crp_corr'], '{:.3f}'):>5} "
              f"adc={fmt(agg['adc_env'], '{:.3f}'):>5} "
              f"time={fmt(agg['elapsed_min'], '{:.1f}'):>5}")

    tex = build_table(default_agg, ablation_aggs)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(tex)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
