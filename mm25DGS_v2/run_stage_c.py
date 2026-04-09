"""Run full Stage C evaluation: C2, C3, C4 on all 7 scenes."""

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import json
import torch
from mm25DGS_v2.train_gaussian import train_gaussians, SCENES

MMIR_DIR = '/home/adnan/Desktop/mmIR/output/train_v13'


def get_mmIR_corr(scene):
    m = json.load(open(f'{MMIR_DIR}/{scene}/best_metrics.json'))
    return m['cart_corr']


def run_mode(mode, num_iters, label):
    print(f"\n{'='*60}")
    print(f"Stage {label}")
    print(f"{'='*60}")
    results = {}
    for scene in SCENES:
        torch.cuda.empty_cache()
        corr, it = train_gaussians(scene, mode=mode, num_iters=num_iters, verbose=False)
        mc = get_mmIR_corr(scene)
        results[scene] = (corr, mc)
        print(f"  {scene}: gauss={corr:.4f} mmIR={mc:.4f} gap={corr-mc:+.4f}")

    print(f"\n{'Scene':<25} {'mmIR':>8} {'Gauss':>8} {'Gap':>8}")
    print(f"{'-'*49}")
    for s in SCENES:
        c, m = results[s]
        print(f"{s:<25} {m:>8.4f} {c:>8.4f} {c-m:>+8.4f}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['c2', 'c3', 'c4', 'all'], default='all')
    parser.add_argument('--iters', type=int, default=500)
    args = parser.parse_args()

    if args.mode in ('c2', 'all'):
        run_mode('c2', 0, 'C2: Gaussian init (no training)')
    if args.mode in ('c3', 'all'):
        run_mode('c3', args.iters, 'C3: Train from mmIR init')
    if args.mode in ('c4', 'all'):
        run_mode('c4', args.iters, 'C4: Train from scratch (LiDAR)')
