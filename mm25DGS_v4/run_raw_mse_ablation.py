"""Follow-up ablation suite under raw MSE loss.

Runs:
  Phase 1.5: 8 BSDF component disable runs (K1-K8)
  Phase 2a (LOO): 6 leave-one-out runs, one per material column
  Phase 2b (TOO): 6 train-only-one runs, one per material column

Total: 20 runs × 7 scenes × 500 iters ≈ 140 min.

All runs use the post-2026-04-13 default config:
  - loss_type='mse_raw'
  - USE_FACTORY_PATTERNS=True
  - LEARN_PATTERNS=False permanent
  - C_radar_gt_match=100

Writes incremental results to md/v4_material_ablation_raw_mse_results.md
and dumps per-run .npz + aggregate.json to
mm25DGS_v4/output/material_ablation_raw_mse/<run_name>/.
"""

import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mm25DGS_v4.train_gaussian import train_gaussians
from mm25DGS_v4.load_pretrained import SCENES
from mm25DGS_v4.material_diagnostics import PARAM_NAMES

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', 'material_ablation_raw_mse')
RESULTS_MD = os.path.join(PROJECT_ROOT, 'md', 'v4_material_ablation_raw_mse_results.md')

BASELINE_CART_CORR = 0.9324  # M1_N1_mse_raw from Phase γ

PHASE15_RUNS = [
    ('K1_no_cbs',       'cbs',       'Disable coherent backscatter sinc factor'),
    ('K2_no_directive', 'directive', 'Disable vMF directive lobe'),
    ('K3_no_broad',     'broad',     'Disable broad/diffuse Lambertian fallback'),
    ('K4_no_spm',       'spm',       'Disable SPM incoherent lobe (KA only)'),
    ('K5_no_ka',        'ka',        'Disable GGX/Cook-Torrance lobe (SPM only)'),
    ('K6_no_blend',     'blend',     'Force eta=1 (drop incoherent term)'),
    ('K7_no_jones',     'jones',     'Replace polarized Jones with scalar |r|² avg'),
    ('K8_no_slab',      'slab',      'Collapse multi-layer slab Fresnel to first-surface'),
]


def run_one(run_name, description, **train_kwargs):
    """Loop train_gaussians over 7 scenes, aggregate, dump npz+json."""
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"RUN: {run_name}")
    print(f"  {description}")
    print(f"  kwargs: {train_kwargs}")
    print(f"{'='*70}")

    per_scene = []
    drifts, fishers, ms_list = [], [], []
    t0 = time.time()
    for scene in SCENES:
        corr, _ = train_gaussians(
            scene, num_iters=500, verbose=False,
            diagnostics_dir=run_dir, run_name=run_name,
            **train_kwargs)
        per_scene.append(corr)
        npz_path = os.path.join(run_dir, f'{run_name}__{scene}.npz')
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            drifts.append(d['drift'])
            fishers.append(d['fisher'])
            ms_list.append(float(d['ms_per_iter']))
        print(f"  {scene}: {corr:.4f}")
    elapsed = time.time() - t0
    mean_corr = float(np.mean(per_scene))
    drift_avg = np.mean(np.stack(drifts), axis=0) if drifts else np.zeros(6)
    fisher_avg = np.mean(np.stack(fishers), axis=0) if fishers else np.zeros(6)

    result = {
        'run_name': run_name,
        'description': description,
        'mean_cart_corr': mean_corr,
        'per_scene_cart_corr': per_scene,
        'delta_vs_baseline': mean_corr - BASELINE_CART_CORR,
        'drift': drift_avg.tolist(),
        'fisher': fisher_avg.tolist(),
        'elapsed_s': elapsed,
    }
    with open(os.path.join(run_dir, 'aggregate.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → mean = {mean_corr:.4f}  (Δ = {result['delta_vs_baseline']:+.4f}), {elapsed:.0f}s")
    return result


def run_phase15():
    print("\n" + "="*70)
    print("PHASE 1.5 (raw MSE) — BSDF COMPONENT ABLATION")
    print("="*70)
    results = {}
    for run_name, comp, desc in PHASE15_RUNS:
        r = run_one(run_name, desc, disabled_components={comp})
        results[run_name] = r
    return results


def run_phase2_loo():
    print("\n" + "="*70)
    print("PHASE 2 (raw MSE) — LEAVE-ONE-OUT")
    print("="*70)
    results = {}
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'LOO_freeze_{name}'
        r = run_one(run_name, f'Freeze {name} (col {k}), train other 5',
                    freeze_mat_cols=[k])
        results[name] = r
    return results


def run_phase2_too():
    print("\n" + "="*70)
    print("PHASE 2 (raw MSE) — TRAIN-ONLY-ONE")
    print("="*70)
    results = {}
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'TOO_only_{name}'
        frozen = [j for j in range(6) if j != k]
        r = run_one(run_name, f'Train only {name} (col {k}), freeze 5 others',
                    freeze_mat_cols=frozen)
        results[name] = r
    return results


def write_results_md(p15_results, loo_results, too_results):
    """Write the complete Phase 1.5 + Phase 2 results back into the .md."""
    with open(RESULTS_MD, 'r') as f:
        text = f.read()

    # Phase 1.5 table
    p15_rows = []
    for run_name, comp, desc in PHASE15_RUNS:
        r = p15_results[run_name]
        delta = r['delta_vs_baseline']
        if delta > -0.005:
            verdict = 'DROP (within noise)'
        elif delta > -0.02:
            verdict = 'borderline'
        else:
            verdict = 'KEEP'
        p15_rows.append(
            f"| {run_name} | {comp} | {r['mean_cart_corr']:.4f} | "
            f"{delta:+.4f} | {verdict} |")
    p15_table = "\n".join(p15_rows)

    # Phase 1.5 summary
    drops = [(c, r['delta_vs_baseline']) for (_, c, _), r in
             zip(PHASE15_RUNS, [p15_results[n] for n, _, _ in PHASE15_RUNS])]
    drop_list = [c for c, d in drops if d > -0.005]
    keep_list = [c for c, d in drops if d < -0.02]
    border_list = [c for c, d in drops if -0.02 <= d <= -0.005]
    p15_summary = (
        f"Under raw MSE + factory patterns + scale fix, the verdicts are: "
        f"DROP={drop_list}, BORDERLINE={border_list}, KEEP={keep_list}. "
        f"Reduced BSDF = full BSDF minus {drop_list}."
    )
    reduced_bsdf = f"Full BSDF minus {drop_list}. Keep: Jones + {keep_list} + {border_list}."

    # LOO / TOO tables
    def fmt_vec(v):
        return "[" + ",".join(f"{x:.2f}" for x in v) + "]"
    def fmt_vec_e(v):
        return "[" + ",".join(f"{x:.1e}" for x in v) + "]"

    loo_rows = []
    for name in PARAM_NAMES:
        r = loo_results[name]
        delta = r['delta_vs_baseline']
        loo_rows.append(
            f"| LOO_freeze_{name} | {name} | {r['mean_cart_corr']:.4f} | "
            f"{delta:+.4f} | {fmt_vec(r['drift'])} | {fmt_vec_e(r['fisher'])} | |")
    loo_table = "\n".join(loo_rows)

    too_rows = []
    for name in PARAM_NAMES:
        r = too_results[name]
        delta = r['delta_vs_baseline']
        too_rows.append(
            f"| TOO_only_{name} | {name} | {r['mean_cart_corr']:.4f} | "
            f"{delta:+.4f} | {fmt_vec(r['drift'])} | {fmt_vec_e(r['fisher'])} | |")
    too_table = "\n".join(too_rows)

    # Verdict matrix
    verdict_rows = []
    for k, name in enumerate(PARAM_NAMES):
        loo_delta = loo_results[name]['delta_vs_baseline']
        too_score = too_results[name]['mean_cart_corr']
        # too_drop = baseline - too_score (how much cart_corr is lost by having only this param)
        loo_big = abs(loo_delta) > 0.01
        # TOO "score" big = TOO is close to baseline => this param alone is sufficient
        too_big = too_score > 0.85
        if loo_big and too_big:
            verdict = 'ESSENTIAL'
        elif loo_big and not too_big:
            verdict = 'CO-FACTOR (needed in combo)'
        elif not loo_big and too_big:
            verdict = 'REDUNDANT'
        else:
            verdict = 'INERT — drop'
        drift_k = loo_results[name]['drift'][k]
        fisher_k = loo_results[name]['fisher'][k]
        verdict_rows.append(
            f"| {name} ({k}) | {loo_delta:+.4f} | {too_score:.4f} | "
            f"{drift_k:.2f} | {fisher_k:.1e} | {verdict} |")
    verdict_matrix = "\n".join(verdict_rows)

    # Slot the filled tables into the .md by replacing the `_pending_` rows
    text = text.replace(
        "| _pending_ | | | | |",
        p15_table,
        1,
    )
    text = text.replace(
        "**Phase 1.5 (raw MSE) summary**: _pending_",
        f"**Phase 1.5 (raw MSE) summary**: {p15_summary}",
    )
    text = text.replace(
        "**Reduced BSDF**: _pending_",
        f"**Reduced BSDF**: {reduced_bsdf}",
    )
    text = text.replace(
        "| _pending_ | | | | | | |",
        loo_table,
        1,
    )
    text = text.replace(
        "| _pending_ | | | | | | |",
        too_table,
        1,
    )
    # Verdict matrix: replace the 6 empty rows
    empty_verdict_block = (
        "| eps_real (0) | | | | | |\n"
        "| eps_imag (1) | | | | | |\n"
        "| sigma_h (2) | | | | | |\n"
        "| l_c (3) | | | | | |\n"
        "| tau_base (4) | | | | | |\n"
        "| thickness (5) | | | | | |"
    )
    text = text.replace(empty_verdict_block, verdict_matrix)

    # Phase 2 summary
    p2_summary_bits = []
    for name in PARAM_NAMES:
        loo_delta = loo_results[name]['delta_vs_baseline']
        too_score = too_results[name]['mean_cart_corr']
        p2_summary_bits.append(f"{name}: LOO Δ={loo_delta:+.4f}, TOO={too_score:.4f}")
    p2_summary = "; ".join(p2_summary_bits)
    text = text.replace(
        "**Phase 2 (raw MSE) summary**: _pending_",
        f"**Phase 2 (raw MSE) summary**: {p2_summary}",
    )

    with open(RESULTS_MD, 'w') as f:
        f.write(text)
    print(f"\nWrote results to {RESULTS_MD}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    t_all = time.time()

    p15 = run_phase15()
    loo = run_phase2_loo()
    too = run_phase2_too()

    print(f"\nTotal compute: {(time.time()-t_all)/60:.1f} min")

    write_results_md(p15, loo, too)


if __name__ == '__main__':
    main()
