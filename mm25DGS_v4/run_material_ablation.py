"""Driver for the v4 material ablation suite.

Runs Phase 1 → Phase 1.5 → Phase 2 in sequence, with decision-rule gating
between phases. Each "run" is `train_gaussians` looped over the 7 benchmark
scenes, aggregated to a mean cart_corr.

Per-run outputs:
  - mm25DGS_v4/output/material_ablation/<run_name>/<scene>.npz   (per scene)
  - md/v4_material_ablation_results.md                           (incrementally appended)

Usage:
  python -m mm25DGS_v4.run_material_ablation --phase 1
  python -m mm25DGS_v4.run_material_ablation --phase 1.5
  python -m mm25DGS_v4.run_material_ablation --phase 2
  python -m mm25DGS_v4.run_material_ablation --all       # 1 → 1.5 → 2 with decision rules
"""

import os
import sys
import time
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mm25DGS_v4.train_gaussian import train_gaussians
from mm25DGS_v4.load_pretrained import SCENES
from mm25DGS_v4.material_diagnostics import PARAM_NAMES

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', 'material_ablation')
RESULTS_MD = os.path.join(PROJECT_ROOT, 'md', 'v4_material_ablation_results.md')
BASELINE_CART_CORR = 0.9351   # Tier A reference, commit 8fe6d6d


# =========================================================================
# Run / aggregate primitives
# =========================================================================

def run_one(run_name, description, num_iters=500, **kwargs):
    """Loop train_gaussians over all 7 scenes for one ablation config.

    Returns dict with mean_cart_corr, per_scene_corr, ms_per_iter (avg),
    drift (avg across scenes), fisher (avg across scenes).
    """
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"RUN: {run_name}")
    print(f"  {description}")
    print(f"  kwargs: {kwargs}")
    print(f"{'='*70}")

    per_scene = []
    drifts = []
    fishers = []
    ms_per_iter_list = []
    t_start = time.time()
    for scene in SCENES:
        corr, _ = train_gaussians(
            scene, num_iters=num_iters, verbose=False,
            diagnostics_dir=run_dir, run_name=run_name,
            **kwargs)
        per_scene.append(corr)
        npz_path = os.path.join(run_dir, f'{run_name}__{scene}.npz')
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            drifts.append(d['drift'])
            fishers.append(d['fisher'])
            ms_per_iter_list.append(float(d['ms_per_iter']))
        print(f"  {scene}: cart_corr={corr:.4f}")
    elapsed = time.time() - t_start
    mean_corr = float(np.mean(per_scene))
    drift_avg = np.mean(np.stack(drifts), axis=0) if drifts else np.zeros(6)
    fisher_avg = np.mean(np.stack(fishers), axis=0) if fishers else np.zeros(6)
    ms_per_iter = float(np.mean(ms_per_iter_list)) if ms_per_iter_list else 0.0

    result = {
        'run_name': run_name,
        'description': description,
        'mean_cart_corr': mean_corr,
        'per_scene_cart_corr': per_scene,
        'delta_vs_baseline': mean_corr - BASELINE_CART_CORR,
        'ms_per_iter': ms_per_iter,
        's_per_run': elapsed,
        'drift': drift_avg.tolist(),
        'fisher': fisher_avg.tolist(),
    }
    # Dump JSON for re-aggregation later
    with open(os.path.join(run_dir, 'aggregate.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  → mean cart_corr = {mean_corr:.4f}  (Δ = {result['delta_vs_baseline']:+.4f}), {elapsed:.0f}s")
    return result


# =========================================================================
# Results .md updater
# =========================================================================

def append_phase1_row(result):
    line = (f"| {result['run_name']} | {result['description']} | "
            f"{result['mean_cart_corr']:.4f} | "
            f"{result['delta_vs_baseline']:+.4f} | "
            f"{result['ms_per_iter']:.1f} | _pending_ |\n")
    _insert_row('## Phase 1 — Baselines', line)


def append_phase15_row(result, disabled, baseline_ms):
    delta_ms = result['ms_per_iter'] - baseline_ms
    line = (f"| {result['run_name']} | {disabled} | "
            f"{result['mean_cart_corr']:.4f} | "
            f"{result['delta_vs_baseline']:+.4f} | "
            f"{result['ms_per_iter']:.1f} | "
            f"{delta_ms:+.1f} | _pending_ |\n")
    _insert_row('## Phase 1.5 — BSDF component ablation', line)


def append_phase2_loo_row(result, frozen_param):
    drift_str = ','.join(f'{x:.2f}' for x in result['drift'])
    fisher_str = ','.join(f'{x:.1e}' for x in result['fisher'])
    line = (f"| {result['run_name']} | {frozen_param} | "
            f"{result['mean_cart_corr']:.4f} | "
            f"{result['delta_vs_baseline']:+.4f} | "
            f"[{drift_str}] | [{fisher_str}] | _pending_ |\n")
    _insert_row('### Leave-one-out (LOO)', line)


def append_phase2_too_row(result, trained_param):
    drift_str = ','.join(f'{x:.2f}' for x in result['drift'])
    fisher_str = ','.join(f'{x:.1e}' for x in result['fisher'])
    line = (f"| {result['run_name']} | {trained_param} | "
            f"{result['mean_cart_corr']:.4f} | "
            f"{result['delta_vs_baseline']:+.4f} | "
            f"[{drift_str}] | [{fisher_str}] | _pending_ |\n")
    _insert_row('### Train-only-one (TOO)', line)


def _insert_row(section_header, line):
    """Insert `line` into the table immediately after `section_header`.

    Replaces the `| _pending_ | ... |` placeholder row on first insert.
    """
    with open(RESULTS_MD, 'r') as f:
        text = f.read()
    if section_header not in text:
        raise ValueError(f"Section {section_header!r} not in {RESULTS_MD}")
    parts = text.split(section_header, 1)
    head = parts[0] + section_header
    tail = parts[1]
    # Find the next `| _pending_` placeholder in tail
    placeholder_marker = '| _pending_ |'
    if placeholder_marker in tail:
        # Replace the placeholder line with our row
        idx_p = tail.find(placeholder_marker)
        line_start = tail.rfind('\n', 0, idx_p) + 1
        line_end = tail.find('\n', idx_p) + 1
        tail = tail[:line_start] + line + tail[line_end:]
    else:
        # Insert just after the table header
        lines = tail.split('\n')
        # Find first line starting with `|---` then insert after it
        for i, ln in enumerate(lines):
            if ln.startswith('|---'):
                lines.insert(i + 1, line.rstrip('\n'))
                break
        tail = '\n'.join(lines)
    with open(RESULTS_MD, 'w') as f:
        f.write(head + tail)


def append_phase_summary(phase_header, summary_text, decision_text):
    """Append a summary block under a phase section, replacing the _pending_ summary."""
    with open(RESULTS_MD, 'r') as f:
        text = f.read()
    marker_summary = f"**Phase {phase_header} summary**: _pending"
    marker_decision = f"**Phase {phase_header} decision**: _pending"
    if marker_summary in text:
        idx = text.find(marker_summary)
        end = text.find('\n', idx) + 1
        text = text[:idx] + f"**Phase {phase_header} summary**: {summary_text}\n" + text[end:]
    if marker_decision in text:
        idx = text.find(marker_decision)
        end = text.find('\n', idx) + 1
        text = text[:idx] + f"**Phase {phase_header} decision**: {decision_text}\n" + text[end:]
    with open(RESULTS_MD, 'w') as f:
        f.write(text)


# =========================================================================
# Phase definitions
# =========================================================================

PHASE1_RUNS = [
    ('B0_fixed_concrete',
     'All 6 params frozen at ITU concrete (no learning)',
     {'mat_mode': 'fixed'}),
    ('B1_scalar_reflectivity',
     'BSDF replaced with sigmoid(rho)*cos_i, rho=raw_materials[:,0]',
     {'mat_mode': 'scalar'}),
    ('B2_global_6param',
     'Per-scene shared (6,) material vector via grad averaging + row broadcast',
     {'mat_mode': 'global'}),
    ('B3_per_point_6param',
     'Full per-point (M, 6) material model with full BSDF (Tier A baseline)',
     {'mat_mode': 'per_point'}),
]

PHASE15_RUNS = [
    ('K1_no_cbs',       'cbs',       'Disable coherent backscatter sinc factor'),
    ('K2_no_directive', 'directive', 'Disable vMF directive lobe'),
    ('K3_no_broad',     'broad',     'Disable broad/diffuse Lambertian fallback'),
    ('K4_no_spm',       'spm',       'Disable SPM incoherent lobe (KA only)'),
    ('K5_no_ka',        'ka',        'Disable GGX/Cook-Torrance lobe (SPM only)'),
    ('K6_no_blend',     'blend',     'Force eta=1 (drop incoherent term)'),
    ('K7_no_jones',     'jones',     'Replace polarized Jones with scalar |r|² average'),
    ('K8_no_slab',      'slab',      'Collapse multi-layer slab Fresnel to first-surface'),
]

# 2^3 LEARN-flag matrix. Tuples are (run_name, M_on, N_on, P_on, description).
# B0_fixed_concrete (M=0,N=1,P=1) and B3_per_point_6param (M=1,N=1,P=1) are
# already done from Phase 1; the driver skips them when their aggregate.json
# already exists.
LEARN_MATRIX_RUNS = [
    ('M0_N0_P0', False, False, False, '(B0_zero) everything frozen at init'),
    ('M1_N0_P0', True,  False, False, 'Materials only — rotations and patterns frozen'),
    ('M0_N1_P0', False, True,  False, 'Normals only — materials and patterns frozen'),
    ('M0_N0_P1', False, False, True,  'Patterns only — materials and normals frozen'),
    ('M1_N1_P0', True,  True,  False, 'Materials + normals, patterns frozen'),
    ('M1_N0_P1', True,  False, True,  'Materials + patterns, normals frozen'),
    ('M0_N1_P1', False, True,  True,  '(B0_fixed_concrete) normals + patterns, materials frozen — already done'),
    ('M1_N1_P1', True,  True,  True,  '(B3_per_point) all three trained — already done'),
]


# =========================================================================
# Phase runners
# =========================================================================

def run_learn_matrix(num_iters=500):
    """Run the 2^3 LEARN-flag ablation matrix.

    Skips runs whose aggregate.json already exists (B0_fixed_concrete = M0_N1_P1,
    B3_per_point = M1_N1_P1 from Phase 1) and aliases them to the matrix names.
    """
    print("\n" + "="*70)
    print("LEARN MATRIX — 2^3 ablation on (materials, normals, patterns)")
    print("="*70)
    results = {}

    # Reuse existing Phase 1 results for the two corners that overlap
    aliases = {
        'M0_N1_P1': 'B0_fixed_concrete',
        'M1_N1_P1': 'B3_per_point_6param',
    }
    for new_name, old_name in aliases.items():
        old_p = os.path.join(OUTPUT_ROOT, old_name, 'aggregate.json')
        if os.path.exists(old_p):
            with open(old_p) as f:
                r = json.load(f)
            r['run_name'] = new_name
            results[new_name] = r
            print(f"  REUSE {new_name} ← {old_name}: mean = {r['mean_cart_corr']:.4f}")
            append_learn_matrix_row(r, new_name)

    for run_name, m_on, n_on, p_on, desc in LEARN_MATRIX_RUNS:
        if run_name in results:
            continue   # already aliased
        r = run_one(run_name, desc, num_iters=num_iters,
                    learn_materials=m_on, learn_normals=n_on, learn_patterns=p_on)
        results[run_name] = r
        append_learn_matrix_row(r, run_name)

    # Print 2x2x2 summary
    print(f"\n{'Run':<12} {'M':>3} {'N':>3} {'P':>3} {'mean':>8} {'Δ vs B3':>10}")
    b3 = results.get('M1_N1_P1', {}).get('mean_cart_corr', BASELINE_CART_CORR)
    for name, m, n, p, _ in LEARN_MATRIX_RUNS:
        r = results[name]
        flags = (int(m), int(n), int(p))
        delta_b3 = r['mean_cart_corr'] - b3
        print(f"  {name:<10} {flags[0]:>3} {flags[1]:>3} {flags[2]:>3} "
              f"{r['mean_cart_corr']:>8.4f} {delta_b3:>+10.4f}")

    summary = (
        f"True floor (everything frozen, M0_N0_P0): {results['M0_N0_P0']['mean_cart_corr']:.4f}. "
        f"Patterns-only: {results['M0_N0_P1']['mean_cart_corr']:.4f}. "
        f"Normals-only: {results['M0_N1_P0']['mean_cart_corr']:.4f}. "
        f"Materials-only: {results['M1_N0_P0']['mean_cart_corr']:.4f}. "
        f"Full (M1_N1_P1): {results['M1_N1_P1']['mean_cart_corr']:.4f}.")
    append_phase_summary('LEARN matrix', summary, "see learn matrix table")
    return results


def append_learn_matrix_row(result, name):
    line = (f"| {name} | {result['description']} | "
            f"{result['mean_cart_corr']:.4f} | "
            f"{result['delta_vs_baseline']:+.4f} | "
            f"{result['ms_per_iter']:.1f} |\n")
    _insert_row('## LEARN matrix', line)


def run_phase1(num_iters=500):
    print("\n" + "="*70)
    print("PHASE 1 — BASELINES")
    print("="*70)
    results = {}
    for run_name, desc, kwargs in PHASE1_RUNS:
        r = run_one(run_name, desc, num_iters=num_iters, **kwargs)
        results[run_name] = r
        append_phase1_row(r)

    # Decision rule
    b0 = results['B0_fixed_concrete']['mean_cart_corr']
    b1 = results['B1_scalar_reflectivity']['mean_cart_corr']
    b2 = results['B2_global_6param']['mean_cart_corr']
    b3 = results['B3_per_point_6param']['mean_cart_corr']
    summary = (f"B0={b0:.4f}, B1={b1:.4f}, B2={b2:.4f}, B3={b3:.4f}. "
               f"Per-point margin over global: {b3-b2:+.4f}. "
               f"Per-point margin over scalar: {b3-b1:+.4f}. "
               f"Per-point margin over fixed: {b3-b0:+.4f}.")
    if b3 <= b2 + 0.005:
        decision = "STOP. Per-point freedom isn't earning its keep — switch to global 6-param."
        proceed = False
    elif b3 <= b1 + 0.005:
        decision = "STOP. BSDF physics isn't earning its keep — switch to scalar reflectivity."
        proceed = False
    elif b3 <= b0 + 0.01:
        decision = "STOP. Nothing being learned — debug first."
        proceed = False
    else:
        decision = "PROCEED to Phase 1.5 — per-point 6-param wins."
        proceed = True
    append_phase_summary('1', summary, decision)
    print(f"\nPhase 1 decision: {decision}\n")
    return results, proceed


def run_phase15(phase1_results, num_iters=500):
    print("\n" + "="*70)
    print("PHASE 1.5 — BSDF COMPONENT ABLATION")
    print("="*70)
    baseline_ms = phase1_results['B3_per_point_6param']['ms_per_iter']
    results = {}
    for run_name, comp, desc in PHASE15_RUNS:
        r = run_one(run_name, desc, num_iters=num_iters,
                    mat_mode='per_point',
                    disabled_components={comp})
        results[run_name] = r
        append_phase15_row(r, comp, baseline_ms)

    b3 = phase1_results['B3_per_point_6param']['mean_cart_corr']
    drops = {comp: b3 - results[name]['mean_cart_corr']
             for name, comp, _ in PHASE15_RUNS}
    keep = [c for c, d in drops.items() if d > 0.02]
    border = [c for c, d in drops.items() if 0.005 <= d <= 0.02]
    drop = [c for c, d in drops.items() if d < 0.005]
    summary = (f"Δ vs B3 per component: " +
               ", ".join(f"{c}={drops[c]:+.4f}" for c, _, _ in
                         [(c, None, None) for c in drops]))
    decision = (f"DROP: {drop}. KEEP: {keep}. BORDERLINE: {border}. "
                f"Reduced model = full BSDF minus {drop}.")
    append_phase_summary('1.5', summary, decision)
    print(f"\nPhase 1.5 decision: {decision}\n")
    return results, drop


def run_phase2(reduced_disabled, num_iters=500):
    print("\n" + "="*70)
    print("PHASE 2 — PARAMETER LOO/TOO")
    print("="*70)
    disabled = set(reduced_disabled) if reduced_disabled else None
    results = {'loo': {}, 'too': {}}

    # LOO: freeze each col individually
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'LOO_freeze_{name}'
        r = run_one(run_name, f'Freeze {name} (col {k})',
                    num_iters=num_iters,
                    mat_mode='per_point',
                    disabled_components=disabled,
                    freeze_mat_cols=[k])
        results['loo'][name] = r
        append_phase2_loo_row(r, name)

    # TOO: train each col individually
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'TOO_only_{name}'
        frozen = [j for j in range(6) if j != k]
        r = run_one(run_name, f'Train only {name} (col {k}), freeze others',
                    num_iters=num_iters,
                    mat_mode='per_point',
                    disabled_components=disabled,
                    freeze_mat_cols=frozen)
        results['too'][name] = r
        append_phase2_too_row(r, name)

    summary = "LOO and TOO sweeps complete; verdict matrix populated below."
    append_phase_summary('2', summary, "See LOO × TOO verdict matrix.")
    print("\nPhase 2 done.\n")
    return results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['1', '1.5', '2', 'learn', 'learn+1.5', 'all'], default='all')
    parser.add_argument('--iters', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if args.phase == 'learn':
        run_learn_matrix(num_iters=args.iters)
        return
    if args.phase == 'learn+1.5':
        run_learn_matrix(num_iters=args.iters)
        # Phase 1.5 needs the B3 ms/iter for delta-ms; load from disk
        phase1_results = {}
        b3_p = os.path.join(OUTPUT_ROOT, 'B3_per_point_6param', 'aggregate.json')
        if os.path.exists(b3_p):
            with open(b3_p) as f:
                phase1_results['B3_per_point_6param'] = json.load(f)
        run_phase15(phase1_results, num_iters=args.iters)
        return

    if args.phase in ('1', 'all'):
        phase1_results, proceed = run_phase1(num_iters=args.iters)
        if not proceed and args.phase == 'all':
            print("Phase 1 decision rule: stopping. Phase 1.5 / 2 not run.")
            return
    if args.phase in ('1.5', 'all'):
        if args.phase == '1.5':
            # Need to load Phase 1 results from disk
            phase1_results = {}
            for run_name, desc, _ in PHASE1_RUNS:
                p = os.path.join(OUTPUT_ROOT, run_name, 'aggregate.json')
                if os.path.exists(p):
                    with open(p) as f:
                        phase1_results[run_name] = json.load(f)
                else:
                    raise RuntimeError(f"Missing Phase 1 result {p}; run --phase 1 first")
        phase15_results, drop = run_phase15(phase1_results, num_iters=args.iters)
    if args.phase in ('2', 'all'):
        if args.phase == '2':
            drop = []
            p = os.path.join(OUTPUT_ROOT, 'phase15_drop.json')
            if os.path.exists(p):
                with open(p) as f:
                    drop = json.load(f)
        run_phase2(drop, num_iters=args.iters)


if __name__ == '__main__':
    main()
