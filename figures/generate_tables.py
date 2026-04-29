"""Generate the two LaTeX results tables for the NeurIPS paper.

Mirrors the mmIR ECCV training_ra.tex format: per-scene + mean for the full
metric set (RA Corr, RA PSNR, RA SSIM, RA RMSE) across all four methods
(Ours / RadarSplat / Radar Fields / DART).

Tables produced:
  1. test_ra_results.tex   — held-out test |RA| metrics, all four methods
  2. train_ra_results.tex  — train |RA| metrics, mm3DGS only (baselines do
                              not yet expose per-train-frame metrics)

Reads:
  - mm25DGS_v5_v4/output_frame_nvs/<scene>.../results.json
        (final_test_cart_corr, final_test_cart_mse, _rmse, _psnr, _ssim)
  - mm25DGS_v5_v4/output_frame_nvs/<scene>.../metrics_train.json
        (ra_corr.mean/std, cart_psnr.mean/std, cart_ssim.mean/std,
         cart_mse.mean/std, cart_rmse.mean/std)
  - baselines/{dart,radarsplat,radarfields}/results/<scene>/metrics.json
        (ra_corr, cart_mse, cart_rmse, cart_psnr, cart_ssim)
        For DART, also __cascaded variant.

Usage:
    python -m figures.generate_tables \
        --ours_dir   mm25DGS_v5_v4/output_frame_nvs \
        --baselines_dir baselines \
        --output_dir latex/NeurIPS_2026_unpacked/Physically_Grounded_Novel_View_Synthesis_for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering/tables
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Optional


SCENES = [
    'seq_0_frame_135',
    'seq_1_frame_185',
    'seq_1_frame_438',
    'seq_2_frame_105',
    'seq_2_frame_160',
    'seq_2_frame_300',
]
SCENE_SHORT = {
    'seq_0_frame_135': 'S0\\,F135',
    'seq_1_frame_185': 'S1\\,F185',
    'seq_1_frame_438': 'S1\\,F438',
    'seq_2_frame_105': 'S2\\,F105',
    'seq_2_frame_160': 'S2\\,F160',
    'seq_2_frame_300': 'S2\\,F300',
}

METHODS = ('ours', 'radarsplat', 'radarfields', 'dart')
METHOD_PRETTY = {
    'ours':        '\\textbf{mm3DGS}',
    'radarsplat':  'RadarSplat~\\cite{kung2025radarsplat}',
    'radarfields': 'Radar Fields~\\cite{10.1145/3641519.3657510}',
    'dart':        'DART~\\cite{huang2024dart}',
}

# Metrics to render. higher_is_better controls which method gets bolded
# per scene. Display formatters per metric.
METRICS = [
    ('cart_corr', 'RA Corr',  '$\\uparrow$',  True,  '{:.3f}'),
    ('cart_psnr', 'RA PSNR',  '$\\uparrow$',  True,  '{:.1f}'),
    ('cart_ssim', 'RA SSIM',  '$\\uparrow$',  True,  '{:.3f}'),
    ('cart_rmse', 'RA RMSE',  '$\\downarrow$', False, '{:.4f}'),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def find_ours_results(scene: str, ours_dir: str) -> Optional[str]:
    pat1 = os.path.join(ours_dir, f'{scene}_*_pass2_N20000', 'results.json')
    pat2 = os.path.join(ours_dir,
        f'{scene}_*_pass2_N20000_dnsfyjt0.05i100u400p0.02_dsigpos_grad_amp_lpos1e-05L2100',
        'results.json')
    matches = sorted(glob.glob(pat1) + glob.glob(pat2),
                      key=os.path.getmtime, reverse=True)
    matches = [m for m in matches if '_initC' not in m and '_initB' not in m
                and '_initA' not in m and '_p4d' not in m]
    return matches[0] if matches else None


def load_ours_test_metrics(scene: str, ours_dir: str) -> Optional[dict]:
    """Return the test-frame metric dict (cart_corr, mse, rmse, psnr, ssim).
    Falls back to final_test_cc + None for the other metrics if the run is
    pre-metrics-upgrade."""
    p = find_ours_results(scene, ours_dir)
    if p is None:
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        'cart_corr': d.get('final_test_cart_corr', d.get('final_test_cc')),
        'cart_mse':  d.get('final_test_cart_mse'),
        'cart_rmse': d.get('final_test_cart_rmse'),
        'cart_psnr': d.get('final_test_cart_psnr'),
        'cart_ssim': d.get('final_test_cart_ssim'),
    }


def load_ours_train_metrics(scene: str, ours_dir: str) -> Optional[dict]:
    """Return the per-metric mean+std across train frames for one scene.

    Output dict: {metric_key: {'mean': float, 'std': float}}.
    """
    p = find_ours_results(scene, ours_dir)
    if p is None:
        return None
    train_p = os.path.join(os.path.dirname(p), 'metrics_train.json')
    if not os.path.exists(train_p):
        return None
    with open(train_p) as f:
        d = json.load(f)
    out = {}
    for key in ('ra_corr', 'cart_mse', 'cart_rmse', 'cart_psnr', 'cart_ssim'):
        v = d.get(key)
        if isinstance(v, dict) and v.get('mean') is not None:
            # ours uses ra_corr (== cart_corr) — alias for the table.
            tk = 'cart_corr' if key == 'ra_corr' else key
            out[tk] = {'mean': v['mean'], 'std': v.get('std', 0.0)}
    return out


def load_baseline_metrics(baseline: str, scene: str, baselines_dir: str
                           ) -> Optional[dict]:
    if baseline == 'dart':
        p = os.path.join(baselines_dir, 'dart', 'results',
                          f'{scene}__cascaded', 'metrics.json')
    else:
        p = os.path.join(baselines_dir, baseline, 'results', scene, 'metrics.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return {
        'cart_corr': d.get('ra_corr'),
        'cart_mse':  d.get('cart_mse'),
        'cart_rmse': d.get('cart_rmse'),
        'cart_psnr': d.get('cart_psnr'),
        'cart_ssim': d.get('cart_ssim'),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(v: Optional[float], spec: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '--'
    return spec.format(v)


def fmt_pm(v: Optional[float], s: Optional[float], spec: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '--'
    a = spec.format(v)
    if s is None:
        return a
    return f'{a}$\\pm${spec.format(s)}'


def best_method(per_method: dict[str, Optional[float]],
                higher_is_better: bool) -> Optional[str]:
    cands = [(m, v) for m, v in per_method.items()
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not cands:
        return None
    cands.sort(key=lambda x: -x[1] if higher_is_better else x[1])
    return cands[0][0]


# ---------------------------------------------------------------------------
# Table emitters
# ---------------------------------------------------------------------------

def write_test_table(rows: dict, path: str):
    """Render the held-out test |RA| comparison table.

    Layout:
      |  scene | RA Corr (4 cols) | RA PSNR (4) | RA SSIM (4) | RA RMSE (4) |
    Columns within each metric: mm3DGS / RadarSplat / Radar Fields / DART.
    """
    n_methods = len(METHODS)
    n_metrics = len(METRICS)
    # Column spec
    col_spec = 'l|' + '|'.join('cccc' for _ in METRICS)
    lines = []
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Held-out novel-view test |RA| metrics across six '
                  'ColoRadar scenes. For each scene, mm3DGS trains on the 8 '
                  'cascaded radar frames adjacent to the held-out test frame '
                  '($F\\!\\pm\\!1\\!\\ldots\\!\\pm\\!4$, chirp 0 only) and is '
                  'evaluated on the held-out test frame ($F$, chirp 0). '
                  'Metrics: Pearson correlation (Corr), peak signal-to-noise '
                  'ratio (PSNR, dB), structural similarity (SSIM), and root-'
                  'mean-square error (RMSE) on min-max-normalized 399$\\times$'
                  '399 Cartesian $|\\mathrm{RA}|$ images. Best per-scene '
                  '\\textbf{bold}.}')
    lines.append('\\label{tab:test_ra}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{2.4pt}')
    lines.append('\\resizebox{\\linewidth}{!}{')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    # Header row 1: metric groups
    head1 = ['']
    for _, label, arrow, *_ in METRICS:
        head1.append(f'\\multicolumn{{{n_methods}}}{{c}}{{{label} {arrow}}}')
    lines.append(' & '.join(head1) + ' \\\\')
    # cmidrules
    cm = []
    for i in range(n_metrics):
        a = 2 + i * n_methods
        b = a + n_methods - 1
        cm.append(f'\\cmidrule(lr){{{a}-{b}}}')
    lines.append(''.join(cm))
    # Header row 2: method names per group
    head2 = ['Scene']
    for _ in METRICS:
        for m in METHODS:
            short = {'ours': 'Ours',
                     'radarsplat': 'RSplat',
                     'radarfields': 'RFields',
                     'dart': 'DART'}[m]
            head2.append(short)
    lines.append(' & '.join(head2) + ' \\\\')
    lines.append('\\midrule')
    # Per-scene rows
    for scene in SCENES:
        cells = [SCENE_SHORT[scene]]
        for mkey, _, _, hib, spec in METRICS:
            per_method = {m: rows.get(m, {}).get(scene, {}).get(mkey)
                          for m in METHODS}
            best = best_method(per_method, hib)
            for m in METHODS:
                v = per_method[m]
                s = fmt(v, spec)
                if best == m and s != '--':
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
        lines.append(' & '.join(cells) + ' \\\\')
    # Mean row
    lines.append('\\midrule')
    cells = ['\\textbf{Mean}']
    for mkey, _, _, hib, spec in METRICS:
        # Compute per-method mean across scenes
        means = {}
        for m in METHODS:
            vals = [rows.get(m, {}).get(s, {}).get(mkey) for s in SCENES]
            vals = [v for v in vals
                    if v is not None and not (isinstance(v, float) and math.isnan(v))]
            means[m] = (sum(vals) / len(vals)) if vals else None
        best = best_method(means, hib)
        for m in METHODS:
            s = fmt(means[m], spec)
            if best == m and s != '--':
                s = f'\\textbf{{{s}}}'
            cells.append(s)
    lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('}')
    lines.append('\\end{table*}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_train_table(ours_train: dict, init_test_cc: dict, path: str):
    """Render train-view metric table, mm3DGS only (baselines lack
    per-train-frame metrics in their public release)."""
    metric_keys = ['cart_corr', 'cart_psnr', 'cart_ssim', 'cart_rmse']
    metric_labels = {
        'cart_corr': 'RA Corr $\\uparrow$',
        'cart_psnr': 'RA PSNR $\\uparrow$',
        'cart_ssim': 'RA SSIM $\\uparrow$',
        'cart_rmse': 'RA RMSE $\\downarrow$',
    }
    metric_specs = {
        'cart_corr': '{:.3f}',
        'cart_psnr': '{:.1f}',
        'cart_ssim': '{:.3f}',
        'cart_rmse': '{:.4f}',
    }
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Training-view |RA| metrics for mm3DGS (mean$\\pm$'
                  'std across the 8 train frames per scene; chirp 0 only). '
                  'Per-scene init-test correlation reported alongside to '
                  'quantify the optimisation lift. Baseline methods do not '
                  'expose per-train-frame metrics in their public release; '
                  'all-baseline columns are pending re-runs.}')
    lines.append('\\label{tab:train_ra}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    cs = 'l|c|' + '|'.join('c' for _ in metric_keys)
    lines.append(f'\\begin{{tabular}}{{{cs}}}')
    lines.append('\\toprule')
    head = ['Scene', 'init test CC'] + [metric_labels[k] for k in metric_keys]
    lines.append(' & '.join(head) + ' \\\\')
    lines.append('\\midrule')
    for scene in SCENES:
        rec = ours_train.get(scene, {})
        cells = [SCENE_SHORT[scene],
                 fmt(init_test_cc.get(scene), '{:.3f}')]
        for mk in metric_keys:
            v = rec.get(mk)
            if v is None:
                cells.append('--')
            else:
                cells.append(fmt_pm(v.get('mean'), v.get('std'),
                                      metric_specs[mk]))
        lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    # Mean row
    cells = ['\\textbf{Mean}']
    init_vals = [v for v in init_test_cc.values()
                 if v is not None and not math.isnan(v)]
    cells.append(fmt(sum(init_vals)/len(init_vals) if init_vals else None,
                      '{:.3f}'))
    for mk in metric_keys:
        means = []
        stds = []
        for scene in SCENES:
            rec = ours_train.get(scene, {}).get(mk)
            if rec is not None:
                means.append(rec.get('mean'))
                stds.append(rec.get('std', 0.0))
        m_of_m = sum(means)/len(means) if means else None
        s_of_m = (sum(stds)/len(stds) if stds else None)
        cells.append(fmt_pm(m_of_m, s_of_m, metric_specs[mk]))
    lines.append(' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours_dir', default='mm25DGS_v5_v4/output_frame_nvs')
    ap.add_argument('--baselines_dir', default='baselines')
    ap.add_argument('--output_dir', default=(
        'latex/NeurIPS_2026_unpacked/Physically_Grounded_Novel_View_Synthesis_'
        'for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering/tables'))
    args = ap.parse_args()

    # Test metrics table: rows[method][scene][metric_key] = float
    rows: dict = {m: {} for m in METHODS}
    init_test_cc = {}
    ours_train = {}
    ours_init = {}

    for scene in SCENES:
        # Ours
        ours_t = load_ours_test_metrics(scene, args.ours_dir)
        if ours_t is not None:
            rows['ours'][scene] = ours_t
        ours_tr = load_ours_train_metrics(scene, args.ours_dir)
        if ours_tr is not None:
            ours_train[scene] = ours_tr
        # Init test CC for the per-scene side column.
        p = find_ours_results(scene, args.ours_dir)
        if p is not None:
            with open(p) as f:
                d = json.load(f)
            init_test_cc[scene] = d.get('init_test_cc')

        # Baselines
        for b in ('dart', 'radarsplat', 'radarfields'):
            r = load_baseline_metrics(b, scene, args.baselines_dir)
            if r is not None:
                rows[b][scene] = r

    # Write
    write_test_table(rows,
                     os.path.join(args.output_dir, 'test_ra_results.tex'))
    write_train_table(ours_train, init_test_cc,
                      os.path.join(args.output_dir, 'train_ra_results.tex'))
    print(f'[done] wrote tables under {args.output_dir}/')

    # Summary print
    print('\n=== test |RA| metric means ===')
    for m in METHODS:
        line = f'  {m:>14}: '
        for mkey, label, _, _, spec in METRICS:
            vals = [rows[m].get(s, {}).get(mkey) for s in SCENES]
            vals = [v for v in vals
                    if v is not None and not (isinstance(v, float) and math.isnan(v))]
            mean = sum(vals)/len(vals) if vals else None
            s = fmt(mean, spec)
            line += f'  {label}={s}'
        print(line)


if __name__ == '__main__':
    main()
