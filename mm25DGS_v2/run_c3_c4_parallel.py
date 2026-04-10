"""Run C3 and C4 on all 7 scenes using both GPUs in parallel."""

import subprocess
import sys
import json
import time
import os

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]
MMIR = '/home/adnan/Desktop/mmIR/output/train_v13'
PYTHON = '/home/adnan/.conda/envs/mmir/bin/python'
PROJECT = '/home/adnan/Desktop/mm3DGS'


def run_scene(scene, mode, gpu_id):
    """Run one scene on one GPU. Returns (scene, mode, cart_corr)."""
    cmd = f"""
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import torch; torch.cuda.empty_cache()
from mm25DGS_v2.train_gaussian import train_gaussians
corr, it = train_gaussians('{scene}', mode='{mode}', num_iters=500, verbose=False)
print(f'RESULT {scene} {mode} {{corr:.6f}}', flush=True)
"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    proc = subprocess.Popen(
        [PYTHON, '-c', cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=PROJECT,
    )
    return proc, scene, mode


def parse_result(proc, scene, mode):
    stdout, stderr = proc.communicate()
    for line in stdout.decode().split('\n'):
        if line.startswith(f'RESULT {scene} {mode}'):
            corr = float(line.split()[-1])
            return corr
    print(f"ERROR: no result for {scene} {mode}", file=sys.stderr)
    print(f"STDERR: {stderr.decode()[-500:]}", file=sys.stderr)
    return None


def run_all(mode):
    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} on all 7 scenes (2 GPUs)")
    print(f"{'='*60}")

    results = {}
    t0 = time.time()

    # Process scenes in batches of 4 (2 per GPU)
    for batch_start in range(0, len(SCENES), 4):
        batch = SCENES[batch_start:batch_start + 4]
        procs = []
        for i, scene in enumerate(batch):
            gpu = i % 2
            print(f"  Starting {scene} on GPU {gpu}...")
            p, s, m = run_scene(scene, mode, gpu)
            procs.append((p, s, m))

        for p, s, m in procs:
            corr = parse_result(p, s, m)
            mc = json.load(open(f'{MMIR}/{s}/best_metrics.json'))['cart_corr']
            results[s] = (corr, mc)
            if corr is not None:
                print(f"  {s}: {corr:.4f} (mmIR: {mc:.4f}, gap: {mc-corr:+.4f})")
            else:
                print(f"  {s}: FAILED")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")

    # Print summary
    print(f"\n{'Scene':<25} {'mmIR':>8} {mode.upper():>8} {'Gap':>8}")
    print('-' * 49)
    gaps = []
    for s in SCENES:
        c, m = results[s]
        if c is not None:
            g = m - c
            gaps.append(g)
            print(f"{s:<25} {m:>8.4f} {c:>8.4f} {g:>+8.4f}")
    if gaps:
        print(f"{'Mean gap':<25} {'':>8} {'':>8} {sum(gaps)/len(gaps):>+8.4f}")
    return results


if __name__ == '__main__':
    c3 = run_all('c3')
    c4 = run_all('c4')

    # Final combined table
    print(f"\n{'='*70}")
    print("Combined Results (RX-sphere splatting)")
    print(f"{'='*70}")
    print(f"{'Scene':<25} {'mmIR':>8} {'C3':>8} {'C3 gap':>8} {'C4':>8} {'C4 gap':>8}")
    print('-' * 65)
    for s in SCENES:
        mc = c3[s][1]
        c3v = c3[s][0] or 0
        c4v = c4[s][0] or 0
        print(f"{s:<25} {mc:>8.4f} {c3v:>8.4f} {mc-c3v:>+8.4f} {c4v:>8.4f} {mc-c4v:>+8.4f}")
    print("ALL DONE")
