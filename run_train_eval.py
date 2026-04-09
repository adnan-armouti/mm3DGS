#!/usr/bin/env python3
"""Train mm25DGS on all 9 scenes (parallel across 2 GPUs) and evaluate cart_corr."""

import subprocess
import sys
import os
import re
import json
import time
import numpy as np
import torch

PYTHON = "/home/adnan/.conda/envs/mmir/bin/python"
BASE = "/home/adnan/Desktop/mm3DGS"
OUTPUT_BASE = os.path.join(BASE, "output/mm25dgs_v3")

SCENES = [
    "seq_0_frame_135", "seq_0_frame_390", "seq_0_frame_451",
    "seq_1_frame_185", "seq_1_frame_277", "seq_1_frame_438",
    "seq_2_frame_105", "seq_2_frame_160", "seq_2_frame_300",
]

# mmIR reference cart_corr (7 scenes)
MMIR_REF = {
    "seq_0_frame_135": 0.9208,
    "seq_0_frame_390": 0.9363,
    "seq_1_frame_185": 0.9541,
    "seq_1_frame_438": 0.9422,
    "seq_2_frame_105": 0.8867,
    "seq_2_frame_160": 0.8638,
    "seq_2_frame_300": 0.9301,
}


def get_frame(scene):
    return re.search(r"frame_(\d+)", scene).group(1)


def train_scene(scene, gpu_id):
    """Train one scene on a specific GPU."""
    frame = get_frame(scene)
    output_dir = os.path.join(OUTPUT_BASE, scene)
    os.makedirs(output_dir, exist_ok=True)

    cfg = {
        "scene_dir": f"{BASE}/data/{scene}",
        "config_path": f"{BASE}/data/{scene}/configs/cascaded_frame_{frame}_aligned_gpu.json",
        "pcl_path": f"{BASE}/data/{scene}/scene/pcl.npy",
        "mesh_path": f"{BASE}/data/{scene}/scene/mesh.ply",
        "output_dir": output_dir,
        "tx_pattern_path": f"{BASE}/assets/antenna_pattern/MMWCAS/tx1_76.npy",
        "rx_pattern_path": f"{BASE}/assets/antenna_pattern/MMWCAS/rx1_76.npy",
        "max_iterations": 500,
        "target_n_gaussians": 25000,
        "shading_tier": 2,
        "device": f"cuda:{gpu_id}",
        "ra_use_log": False,
        "ra_mag_weight": 1.0,
        "lr_materials": 0.5,
        "lr_positions": 1.6e-4,
        "lr_rotations": 1e-3,
        "lr_scales": 5e-3,
        "lr_opacities": 5e-2,
        "densify_interval": 100,
        "log_interval": 50,
        "checkpoint_interval": 100,
    }

    cfg_path = os.path.join(output_dir, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # When CUDA_VISIBLE_DEVICES=N, PyTorch sees it as cuda:0
    cfg["device"] = "cuda:0"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    cmd = [PYTHON, "-m", "mm25DGS", "--config", cfg_path]
    log_path = os.path.join(output_dir, "train.log")
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            env=env, cwd=BASE,
        )
    return proc, scene, gpu_id


def evaluate_scene(scene):
    """Evaluate cart_corr for one trained scene."""
    frame = get_frame(scene)
    model_path = os.path.join(OUTPUT_BASE, scene, "best_model.pt")
    config_path = f"{BASE}/data/{scene}/configs/cascaded_frame_{frame}_aligned_gpu.json"
    gt_path = f"{BASE}/data/{scene}/radar/cascaded_frame_{frame}.npy"

    sys.path.insert(0, BASE)
    from mm25DGS.eval_adapter import GaussianRendererWrapper
    from mmir.data.ra_utils import adc_to_ra_image, ra_polar_to_cartesian
    from mmir.data.io_utils import compute_range_res_from_cfg
    from mmir.data.data_utils import load_target

    wrapper = GaussianRendererWrapper(model_path, config_path)
    adc_ri = wrapper.render_forward()

    gt_adc_ri, _ = load_target(gt_path, torch.device("cpu"), normalize=True)
    gt_ri = gt_adc_ri.numpy()

    range_res = compute_range_res_from_cfg(config_path)
    ra_rend = adc_to_ra_image(torch.from_numpy(adc_ri))
    ra_gt = adc_to_ra_image(torch.from_numpy(gt_ri))

    ra_rend_cart = ra_polar_to_cartesian(ra_rend.numpy(), range_res)
    ra_gt_cart = ra_polar_to_cartesian(ra_gt.numpy(), range_res)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-30)

    return float(np.corrcoef(norm(ra_rend_cart).ravel(), norm(ra_gt_cart).ravel())[0, 1])


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Launch training: round-robin across 2 GPUs, max 2 concurrent
    active = []
    queue = list(SCENES)
    gpu_free = [0, 1]

    print(f"Training {len(SCENES)} scenes across 2 GPUs...")
    t0 = time.time()

    while queue or active:
        # Launch new jobs on free GPUs
        while queue and gpu_free:
            scene = queue.pop(0)
            gpu = gpu_free.pop(0)
            proc, s, g = train_scene(scene, gpu)
            active.append((proc, s, g))
            print(f"  Started {scene} on GPU {gpu}")

        # Poll for completion
        still_active = []
        for proc, s, g in active:
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"  Finished {s} [{status}] ({elapsed:.0f}s)")
                gpu_free.append(g)
            else:
                still_active.append((proc, s, g))
        active = still_active

        if active:
            time.sleep(2)

    elapsed_total = time.time() - t0
    print(f"\nAll training complete in {elapsed_total:.0f}s\n")

    # Evaluate all scenes
    print("Evaluating cart_corr...")
    results = {}
    for scene in SCENES:
        try:
            cc = evaluate_scene(scene)
            results[scene] = cc
        except Exception as e:
            print(f"  {scene}: ERROR - {e}")
            results[scene] = None

    # Print results table
    print(f"\n{'Scene':<25} {'mm25DGS':>10} {'mmIR':>10} {'Gap':>10}")
    print("-" * 57)
    vals_ours = []
    vals_ref = []
    for scene in SCENES:
        cc = results.get(scene)
        ref = MMIR_REF.get(scene)
        cc_str = f"{cc:.4f}" if cc is not None else "N/A"
        ref_str = f"{ref:.4f}" if ref is not None else "-"
        gap_str = f"{ref - cc:.4f}" if (cc is not None and ref is not None) else "-"
        print(f"  {scene:<23} {cc_str:>10} {ref_str:>10} {gap_str:>10}")
        if cc is not None:
            vals_ours.append(cc)
        if cc is not None and ref is not None:
            vals_ref.append(ref)

    mean_ours = np.mean(vals_ours) if vals_ours else 0
    mean_ref_7 = np.mean(list(MMIR_REF.values()))
    print("-" * 57)
    print(f"  {'Mean (all 9)':<23} {mean_ours:>10.4f}")
    # Mean over 7 mmIR scenes only
    mmIR_scenes = [s for s in SCENES if s in MMIR_REF]
    vals_7 = [results[s] for s in mmIR_scenes if results.get(s) is not None]
    if vals_7:
        mean_7 = np.mean(vals_7)
        print(f"  {'Mean (7 mmIR scenes)':<23} {mean_7:>10.4f} {mean_ref_7:>10.4f} {mean_ref_7 - mean_7:>10.4f}")

    # Save results
    with open(os.path.join(OUTPUT_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
