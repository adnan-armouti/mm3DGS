"""Memory + time sweep over target_n (number of Gaussians per scene).

Finds the maximum number of points we can fit on a single 24 GB 4090
under the current v4 config (all clamp fixes, full microfacet Jones,
raw MSE loss, mat_lr=0.01, rot_lr=5e-3).

Strategy:
  - For each target_n, run ONE full forward+backward iteration on scene 135
  - Record peak CUDA memory and wall-clock time
  - Wrap in try/except torch.cuda.OutOfMemoryError
  - Report a table

We use ONE iteration (not full 500) because:
  - OOM typically happens during backward of iter 0 or 1
  - Peak memory is set by the first iter (autograd graph, buffer
    allocations). Subsequent iters reuse the same memory.
  - One iter × 5 target_n values = ~2 min of compute total
"""

import os
import sys
import time
import gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mm25DGS_v4.train_gaussian import train_gaussians, DEVICE

SWEEP = [50_000, 75_000, 100_000, 150_000, 200_000, 300_000, 500_000, 750_000, 1_000_000]


def run_one(target_n):
    """Run ONE full training iter at the given target_n. Measure peak memory and time."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    try:
        corr, _ = train_gaussians(
            'seq_0_frame_135',
            num_iters=2,           # 2 iters so we can also clock second-iter time
            target_n=target_n,
            verbose=False,
            random_init_width=2.0,
        )
        elapsed = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return dict(
            target_n=target_n,
            status='ok',
            peak_gb=peak_gb,
            elapsed=elapsed,
            cart_corr=corr,
        )
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return dict(
            target_n=target_n,
            status='OOM',
            peak_gb=float('nan'),
            elapsed=time.time() - t0,
            cart_corr=float('nan'),
            error=str(e)[:200],
        )
    except RuntimeError as e:
        msg = str(e)
        if 'out of memory' in msg.lower() or 'CUDA out of memory' in msg:
            torch.cuda.empty_cache()
            return dict(
                target_n=target_n,
                status='OOM',
                peak_gb=float('nan'),
                elapsed=time.time() - t0,
                cart_corr=float('nan'),
                error=msg[:200],
            )
        raise


def main():
    device = torch.cuda.get_device_properties(0)
    total_gb = device.total_memory / (1024 ** 3)
    print(f"GPU: {device.name}, {total_gb:.1f} GB total")

    results = []
    for tn in SWEEP:
        print(f"\n=== target_n = {tn:,} ===", flush=True)
        r = run_one(tn)
        results.append(r)
        if r['status'] == 'ok':
            print(f"  peak={r['peak_gb']:.2f} GB  elapsed={r['elapsed']:.1f}s  "
                  f"cart_corr(2 iters)={r['cart_corr']:.4f}")
        else:
            print(f"  OOM: {r.get('error','')[:120]}")

    print(f"\n{'='*70}")
    print(f"  GPU: {device.name} ({total_gb:.1f} GB)")
    print(f"{'='*70}")
    print(f"{'target_n':>10}  {'status':<5}  {'peak_GB':>9}  {'elapsed_s':>10}")
    for r in results:
        tag = r['status']
        peak = f"{r['peak_gb']:.2f}" if r['status'] == 'ok' else '—'
        el = f"{r['elapsed']:.1f}" if r['status'] == 'ok' else '—'
        print(f"{r['target_n']:>10,}  {tag:<5}  {peak:>9}  {el:>10}")

    # Save raw data
    import json
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'mm25DGS_v4', 'output', 'target_n_sweep.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
