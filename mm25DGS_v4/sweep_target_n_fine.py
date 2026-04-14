"""Finer target_n sweep between 75K and 100K to pinpoint the OOM boundary."""
import os, sys, time, gc, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mm25DGS_v4.train_gaussian import train_gaussians

SWEEP = [75_000, 80_000, 85_000, 90_000, 95_000, 100_000]


def run_one(target_n):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        corr, _ = train_gaussians('seq_0_frame_135', num_iters=2,
                                  target_n=target_n, verbose=False,
                                  random_init_width=2.0)
        return dict(target_n=target_n, status='ok',
                    peak_gb=torch.cuda.max_memory_allocated() / (1024 ** 3),
                    elapsed=time.time() - t0, cart_corr=corr)
    except Exception as e:
        msg = str(e)
        if 'out of memory' in msg.lower():
            torch.cuda.empty_cache()
            return dict(target_n=target_n, status='OOM',
                        peak_gb=float('nan'), elapsed=time.time() - t0,
                        cart_corr=float('nan'), error=msg[:200])
        raise


results = []
for tn in SWEEP:
    print(f"\n=== target_n = {tn:,} ===", flush=True)
    r = run_one(tn)
    results.append(r)
    if r['status'] == 'ok':
        print(f"  peak={r['peak_gb']:.2f} GB  elapsed={r['elapsed']:.1f}s")
    else:
        print(f"  OOM")

print(f"\n{'target_n':>10}  {'status':<5}  {'peak_GB':>9}")
for r in results:
    peak = f"{r['peak_gb']:.2f}" if r['status'] == 'ok' else '—'
    print(f"{r['target_n']:>10,}  {r['status']:<5}  {peak:>9}")

with open('mm25DGS_v4/output/target_n_sweep_fine.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
