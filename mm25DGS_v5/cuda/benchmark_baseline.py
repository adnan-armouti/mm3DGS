"""Phase A baseline instrumentation.

Captures:
  1. Per-stage timings of render_factorized at target_n=90K on scene 135,
     dumped to mm25DGS_v5/output/baseline_stage_times.json.
  2. A snapshot of key intermediate tensors (inputs + outputs of Step 4)
     saved to mm25DGS_v5/output/kernel_validation/reference_forward.npz
     for later numerical validation of the CUDA kernels.

Run with:
    CUDA_VISIBLE_DEVICES=1 /home/adnan/.conda/envs/mmir/bin/python \\
        -m mm25DGS_v5.cuda.benchmark_baseline
"""
import json
import os
import time

import numpy as np
import torch

from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch
from mm25DGS_v5.train_gaussian import (
    init_visible_weighted, render_gaussians, cull_gaussians, PROJECT_ROOT,
    DEVICE,
)
from mm25DGS_v5.load_pretrained import load_trained_config
from mm25DGS_v5 import rasterizer_factorized as rf


# ---------------------------------------------------------------------------
# Stage timing: monkey-patch render_factorized with cuda events wrapped
# around the major blocks.
# ---------------------------------------------------------------------------

STAGE_MARKERS = [
    "step1_2_3",     # material prep + per-TX + per-RX geometry (lines ~60-227)
    "step4_bsdf",    # fused BSDF inner loop (lines ~228-420)
    "step5_splat",   # range-profile splatting (lines ~422-end)
]


def time_render(model, rast, vertex_areas, active_mask, n_warm=3, n_iter=30):
    """Run render_gaussians n_iter times, returning per-stage ms dict.

    Phase A: total render_factorized time only. Stage-level split will be
    added in Phase B once render_factorized is instrumented with cuda
    event checkpoints.
    """
    # Forward-only timing (no autograd graph → stable memory across iters).
    with torch.no_grad():
        for _ in range(n_warm):
            rp_r, rp_i = render_gaussians(model, rast, vertex_areas, active_mask)
        torch.cuda.synchronize()

        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n_iter):
            rp_r, rp_i = render_gaussians(model, rast, vertex_areas, active_mask)
        t1.record()
        torch.cuda.synchronize()
        total_ms = t0.elapsed_time(t1) / n_iter
    return {"render_total_ms": total_ms, "n_iter": n_iter,
            "note": "forward-only (no_grad); full fwd+bwd timing adds ~2x."}


def capture_reference_forward(model, rast, vertex_areas, active_mask, out_path):
    """Run one render_factorized pass and save inputs + outputs + key
    intermediates as a .npz file. Used for numerical validation of the
    CUDA kernels in Phase B/C.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas, active_mask)
    torch.cuda.synchronize()

    def to_np(x):
        return x.detach().cpu().numpy()

    positions = model.positions[active_mask].detach()
    normals = model.get_normals()[active_mask].detach()
    raw_mat = model.raw_materials[active_mask].detach()
    areas = vertex_areas[active_mask].detach()

    np.savez_compressed(
        out_path,
        # Inputs
        positions=to_np(positions),
        normals=to_np(normals),
        areas=to_np(areas),
        raw_materials=to_np(raw_mat),
        tx_positions=to_np(rast.tx_positions),
        rx_positions=to_np(rast.rx_positions),
        tx_boresights=to_np(rast.tx_boresights),
        rx_boresights=to_np(rast.rx_boresights),
        # Geometry constants
        center_freq=float(rast.center_freq),
        slope=float(rast.slope),
        sample_rate=float(rast.sample_rate),
        radar_constant=float(rast.radar_constant),
        rx_dBFS_scale=float(rast.rx_dBFS_scale),
        adc_scale=float(rast.adc_scale),
        n_tx=int(rast.n_tx),
        n_rx=int(rast.n_rx),
        K=int(rast.K),
        # Outputs
        rp_real=to_np(rp_real),
        rp_imag=to_np(rp_imag),
    )
    print(f"  saved reference forward: {out_path}")
    print(f"  rp_real shape: {rp_real.shape}, "
          f"max|rp|: {rp_real.abs().max().item():.3e}")


def main():
    scene = "seq_0_frame_135"
    print(f"[Phase A baseline] scene={scene}, target_n=90000")
    torch.manual_seed(42)
    np.random.seed(42)

    config = load_trained_config(scene)
    rast = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )
    model = init_visible_weighted(scene, rast, target_n=90000)

    # Match training random_init_width=2.0 path: re-init materials to
    # per-point random draws with width 2.0 around concrete default.
    N = model.N
    g = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        noise = torch.randn((N, 6), device="cuda", generator=g) * 2.0
        model.raw_materials.add_(noise)

    active_mask = cull_gaussians(model, rast)
    n_active = int(active_mask.sum().item())
    vertex_areas = torch.zeros(N, device="cuda")
    vertex_areas[active_mask] = 1.0
    print(f"  N={N}, active={n_active}")

    out_dir = os.path.join(PROJECT_ROOT, "mm25DGS_v5", "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Timing
    timings = time_render(model, rast, vertex_areas, active_mask)
    timings["scene"] = scene
    timings["N"] = N
    timings["active"] = n_active
    timings["note"] = (
        "Phase A: total render_factorized time only. "
        "Stage-level split will be added when render_factorized is "
        "instrumented with cuda event checkpoints in Phase B."
    )
    with open(os.path.join(out_dir, "baseline_stage_times.json"), "w") as f:
        json.dump(timings, f, indent=2)
    print(f"  render_total: {timings['render_total_ms']:.1f} ms/iter "
          f"(avg of {timings['n_iter']} iters)")
    print(f"  saved: {out_dir}/baseline_stage_times.json")

    # 2. Reference forward capture
    ref_path = os.path.join(out_dir, "kernel_validation", "reference_forward.npz")
    capture_reference_forward(model, rast, vertex_areas, active_mask, ref_path)


if __name__ == "__main__":
    main()
