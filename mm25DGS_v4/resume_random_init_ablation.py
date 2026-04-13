"""Resume the random-init ablation from where it died.

Skips runs whose aggregate.json already exists. Runs the rest with the
same random_init_width=2.0, random_init_seed=42.
"""

import os, sys, json, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mm25DGS_v4.train_gaussian import train_gaussians
from mm25DGS_v4.load_pretrained import SCENES
from mm25DGS_v4.material_diagnostics import PARAM_NAMES

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', 'material_ablation_raw_mse_random_init')

RANDOM_INIT_WIDTH = 2.0
RANDOM_INIT_SEED = 42


def run_one(run_name, description, **train_kwargs):
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    agg_path = os.path.join(run_dir, 'aggregate.json')
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            r = json.load(f)
        print(f"  SKIP {run_name} (already done, mean={r['mean_cart_corr']:.4f})")
        return r

    train_kwargs.setdefault('random_init_width', RANDOM_INIT_WIDTH)
    train_kwargs.setdefault('random_init_seed', RANDOM_INIT_SEED)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n=== RUN: {run_name} ===")
    print(f"  {description}")

    per_scene, drifts, fishers, ms_list = [], [], [], []
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
    with open(agg_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  → mean = {mean_corr:.4f}, {elapsed:.0f}s")
    return result


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    t_all = time.time()

    # LOO
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'LOO_freeze_{name}'
        run_one(run_name, f'Freeze {name} (col {k}), train other 5',
                freeze_mat_cols=[k])

    # TOO
    for k, name in enumerate(PARAM_NAMES):
        run_name = f'TOO_only_{name}'
        frozen = [j for j in range(6) if j != k]
        run_one(run_name, f'Train only {name} (col {k}), freeze 5 others',
                freeze_mat_cols=frozen)

    print(f"\nTotal resume time: {(time.time()-t_all)/60:.1f} min")


if __name__ == '__main__':
    main()
