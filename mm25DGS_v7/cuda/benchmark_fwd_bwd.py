"""Quick fwd+bwd benchmark for Phase C revisit tuning.

Run with:
    CUDA_VISIBLE_DEVICES=1 /home/adnan/.conda/envs/mmir/bin/python \
        -m mm25DGS_v5.cuda.benchmark_fwd_bwd
"""
import numpy as np
import torch

from mm25DGS_v5.rasterizer import Rasterizer
from mm25DGS_v5.train_gaussian import (
    init_visible_weighted, render_gaussians, cull_gaussians, DEVICE,
)
from mm25DGS_v5.load_pretrained import load_trained_config


def main():
    scene = "seq_0_frame_135"
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
    N = model.N
    g = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        noise = torch.randn((N, 6), device="cuda", generator=g) * 2.0
        model.raw_materials.add_(noise)

    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(N, device="cuda")
    vertex_areas[active_mask] = 1.0

    # Warmup
    for _ in range(3):
        rp_r, rp_i = render_gaussians(model, rast, vertex_areas, active_mask)
        loss = (rp_r * rp_r + rp_i * rp_i).sum()
        loss.backward()
        model.raw_materials.grad = None
        if model.rotations.grad is not None:
            model.rotations.grad = None
    torch.cuda.synchronize()

    n_iter = 20
    # Fwd+bwd
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_iter):
        rp_r, rp_i = render_gaussians(model, rast, vertex_areas, active_mask)
        loss = (rp_r * rp_r + rp_i * rp_i).sum()
        loss.backward()
        model.raw_materials.grad = None
        if model.rotations.grad is not None:
            model.rotations.grad = None
    t1.record()
    torch.cuda.synchronize()
    fwd_bwd_ms = t0.elapsed_time(t1) / n_iter

    # Fwd only
    with torch.no_grad():
        for _ in range(3):
            render_gaussians(model, rast, vertex_areas, active_mask)
        torch.cuda.synchronize()
        t0.record()
        for _ in range(n_iter):
            render_gaussians(model, rast, vertex_areas, active_mask)
        t1.record()
        torch.cuda.synchronize()
        fwd_ms = t0.elapsed_time(t1) / n_iter

    bwd_ms = fwd_bwd_ms - fwd_ms
    print(f"fwd     : {fwd_ms:7.2f} ms/iter")
    print(f"fwd+bwd : {fwd_bwd_ms:7.2f} ms/iter")
    print(f"bwd (diff): {bwd_ms:7.2f} ms/iter")


if __name__ == "__main__":
    main()
