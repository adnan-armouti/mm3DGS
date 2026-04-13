"""Twin of run_raw_mse_ablation.py but with random_init_width=2.0.

Tests whether the "only eps_imag matters" and "slab is essential" findings
hold when material parameters are NOT initialized to ITU concrete but
instead to concrete ± uniform(-2, +2) per-point in raw parameter space.

Total: 1 baseline + 8 Phase 1.5 + 6 LOO + 6 TOO = 21 runs × 7 scenes
× 500 iters ≈ 150 min.

All runs use:
  - loss_type='mse_raw' (default)
  - USE_FACTORY_PATTERNS=True
  - LEARN_PATTERNS=False
  - C_radar_gt_match=100
  - random_init_width=2.0
  - random_init_seed=42

Writes incremental results to
md/v4_material_ablation_raw_mse_random_init_results.md and dumps
per-run .npz + aggregate.json to
mm25DGS_v4/output/material_ablation_raw_mse_random_init/<run_name>/.
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

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', 'material_ablation_raw_mse_random_init')
RESULTS_MD = os.path.join(PROJECT_ROOT, 'md', 'v4_material_ablation_raw_mse_random_init_results.md')

RANDOM_INIT_WIDTH = 2.0
RANDOM_INIT_SEED = 42

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
    train_kwargs.setdefault('random_init_width', RANDOM_INIT_WIDTH)
    train_kwargs.setdefault('random_init_seed', RANDOM_INIT_SEED)
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
        'drift': drift_avg.tolist(),
        'fisher': fisher_avg.tolist(),
        'elapsed_s': elapsed,
    }
    with open(os.path.join(run_dir, 'aggregate.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → mean = {mean_corr:.4f}, {elapsed:.0f}s")
    return result


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    t_all = time.time()

    print("\n" + "="*70)
    print("BASELINE under random init")
    print("="*70)
    baseline = run_one('M1_N1_baseline', 'Full 6-param model under random init')

    print("\n" + "="*70)
    print("PHASE 1.5 (raw MSE, random init) — BSDF COMPONENT ABLATION")
    print("="*70)
    p15 = {}
    for run_name, comp, desc in PHASE15_RUNS:
        r = run_one(run_name, desc, disabled_components={comp})
        p15[run_name] = r

    print("\n" + "="*70)
    print("PHASE 2 (raw MSE, random init) — LEAVE-ONE-OUT")
    print("="*70)
    loo = {}
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'LOO_freeze_{name}'
        r = run_one(run_name, f'Freeze {name} (col {k}), train other 5',
                    freeze_mat_cols=[k])
        loo[name] = r

    print("\n" + "="*70)
    print("PHASE 2 (raw MSE, random init) — TRAIN-ONLY-ONE")
    print("="*70)
    too = {}
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'TOO_only_{name}'
        frozen = [j for j in range(6) if j != k]
        r = run_one(run_name, f'Train only {name} (col {k}), freeze 5 others',
                    freeze_mat_cols=frozen)
        too[name] = r

    print(f"\nTotal compute: {(time.time()-t_all)/60:.1f} min")

    # Print summary tables (md writeup is manual)
    print("\n=== BASELINE ===")
    print(f"M1_N1 (random init): {baseline['mean_cart_corr']:.4f}")

    print("\n=== Phase 1.5 ===")
    print(f"{'run':<20}{'disabled':<12}{'mean':>10}{'Δ vs baseline':>18}")
    for n, c, _ in PHASE15_RUNS:
        r = p15[n]
        d = r['mean_cart_corr'] - baseline['mean_cart_corr']
        print(f"{n:<20}{c:<12}{r['mean_cart_corr']:>10.4f}{d:>+18.4f}")

    print("\n=== Phase 2 LOO ===")
    for name in PARAM_NAMES:
        r = loo[name]
        d = r['mean_cart_corr'] - baseline['mean_cart_corr']
        print(f"  LOO_freeze_{name:<12}  mean={r['mean_cart_corr']:.4f}  Δ={d:+.4f}")

    print("\n=== Phase 2 TOO ===")
    for name in PARAM_NAMES:
        r = too[name]
        d = r['mean_cart_corr'] - baseline['mean_cart_corr']
        print(f"  TOO_only_{name:<12}    mean={r['mean_cart_corr']:.4f}  Δ={d:+.4f}")


if __name__ == '__main__':
    main()
