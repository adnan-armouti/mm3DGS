"""
Stage B: Train per-vertex materials on mesh vertices using the PyTorch rasterizer.

Matches mmIR training:
  - RA magnitude MSE with min-max normalization, linear (no log)
  - Adam optimizer: lr_materials=0.5, lr_normals=0.01, lr_patterns=0.05
  - LR warmup: 0.3 -> 1.0 over first 5 iterations
  - Gradient clipping: RMS clip per parameter group
  - 500 iterations
  - Phase detached from autograd (enable_grad_phase=False)
  - Seed rotation interval: 9999 (effectively fixed seed for this training run)

Usage:
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v2.train_mesh --scene seq_0_frame_135
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v2.train_mesh --all
  CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v2.train_mesh --scene seq_0_frame_135 --from-mmIR
"""

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmir.data.ra_utils import adc_to_ra_image, adc_to_ra_complex, ra_polar_to_cartesian, compute_cartesian_ra_metrics
from mmir.data.io_utils import compute_range_res_from_cfg
from mm25DGS_v2.render_mmIR import load_trained_config, load_best_params, TRAIN_OUTPUT_DIR, GOLD_REF_DIR
from mm25DGS_v2.rasterizer_torch import (
    RasterizerTorch, reparameterize_torch, inverse_reparameterize_torch,
    AntennaPatternTorch,
)

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]

DEVICE = "cuda:0"

# ITU concrete defaults (physics space)
ITU_CONCRETE = np.array([5.31, 0.0326, 1e-4, 5e-3, 0.5, 0.15], dtype=np.float32)


# =========================================================================
# Loss function (B1)
# =========================================================================

def compute_ra_loss(adc_real, adc_imag, gt_adc_ri):
    """RA magnitude MSE with min-max normalization (linear, no log).

    Matches mmIR loss_config: ra_mag_use_log=False, ra_mag_loss_type=l2.

    Args:
        adc_real, adc_imag: (N_tx, N_rx, K) rendered ADC
        gt_adc_ri: (N_tx, N_rx, K, 2) GT ADC real/imag

    Returns:
        (loss, loss_dict)
    """
    rendered_ri = torch.stack([adc_real, adc_imag], dim=-1)

    ra_rendered = adc_to_ra_complex(rendered_ri)
    ra_gt = adc_to_ra_complex(gt_adc_ri)

    ra_rend_mag = torch.abs(ra_rendered)
    ra_gt_mag = torch.abs(ra_gt)

    # Min-max normalize (separate)
    ra_rend_norm = _minmax_normalize(ra_rend_mag)
    ra_gt_norm = _minmax_normalize(ra_gt_mag)

    loss = torch.mean((ra_rend_norm - ra_gt_norm) ** 2)

    return loss, {"ra_mse": loss.item()}


def _minmax_normalize(x):
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-30:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


# =========================================================================
# LR schedule (B2)
# =========================================================================

def get_lr_scale(iteration, warmup_iters=5, warmup_factor=0.3):
    """Linear warmup -> constant."""
    if iteration < warmup_iters:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup_iters, 1))
    return 1.0


def rms_clip_grad(param, max_rms):
    """RMS gradient clipping (per parameter)."""
    if param.grad is None:
        return
    g = param.grad.data
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    rms = torch.sqrt(torch.mean(g ** 2))
    if rms > max_rms:
        g.mul_(max_rms / rms)
    param.grad.data = g


# =========================================================================
# Training loop (B3)
# =========================================================================

def train_scene(scene, num_iters=500, from_mmIR=False, verbose=True):
    """Train per-vertex materials on mesh vertices.

    Args:
        scene: Scene name
        num_iters: Number of training iterations
        from_mmIR: If True, initialize from mmIR's trained parameters.
                   If False, initialize from ITU concrete defaults.
    """
    config = load_trained_config(scene)
    raw_params_mmIR, normal_params_mmIR, pattern_data = load_best_params(scene)

    # Create rasterizer
    rast = RasterizerTorch(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )

    # Load GT ADC
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_adc_ri = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32)).to(DEVICE)

    # Initialize parameters
    N = rast.n_vertices
    n_tx, n_rx, K = rast.n_tx, rast.n_rx, rast.K

    if from_mmIR:
        # Start from mmIR trained params (B5: verify no regression)
        raw_materials = torch.from_numpy(raw_params_mmIR.astype(np.float32)).to(DEVICE)
        if normal_params_mmIR is not None:
            normals = torch.from_numpy(normal_params_mmIR.astype(np.float32)).to(DEVICE)
        else:
            normals = torch.from_numpy(rast.mesh_normals_np.copy()).to(DEVICE)
    else:
        # Start from ITU concrete defaults (B4: train from scratch)
        raw_default = inverse_reparameterize_torch(ITU_CONCRETE)
        raw_materials = torch.from_numpy(
            np.tile(raw_default, (N, 1))).to(DEVICE)
        normals = torch.from_numpy(rast.mesh_normals_np.copy()).to(DEVICE)

    # Make learnable
    raw_materials = torch.nn.Parameter(raw_materials)
    normals_param = torch.nn.Parameter(normals)

    # Antenna patterns (learnable)
    # Store E and H patterns for TX and RX as parameters
    if pattern_data is not None:
        # Inject mmIR learned patterns
        rast.inject_trained_params(
            raw_params_mmIR if from_mmIR else np.tile(inverse_reparameterize_torch(ITU_CONCRETE), (N, 1)),
            normal_params_mmIR if from_mmIR else rast.mesh_normals_np.copy(),
            pattern_data)
        tx_E = torch.nn.Parameter(rast.tx_antenna.E.clone())
        tx_H = torch.nn.Parameter(rast.tx_antenna.H.clone())
        rx_E = torch.nn.Parameter(rast.rx_antenna.E.clone())
        rx_H = torch.nn.Parameter(rast.rx_antenna.H.clone())
    else:
        tx_E = torch.nn.Parameter(rast.tx_antenna.E.clone())
        tx_H = torch.nn.Parameter(rast.tx_antenna.H.clone())
        rx_E = torch.nn.Parameter(rast.rx_antenna.E.clone())
        rx_H = torch.nn.Parameter(rast.rx_antenna.H.clone())

    # Optimizer groups matching mmIR
    optimizer = torch.optim.Adam([
        {"params": [raw_materials], "lr": 0.5, "name": "materials"},
        {"params": [normals_param], "lr": 0.01, "name": "normals"},
        {"params": [tx_E, tx_H, rx_E, rx_H], "lr": 0.05, "name": "patterns"},
    ], betas=(0.9, 0.999), eps=1e-8)

    # Clip values
    clip_vals = {"materials": 1.0, "normals": 0.5, "patterns": 1.0}

    # Run reservoir sampler once (fixed seed since seed_rotation=9999)
    hits = rast._run_reservoir_sampler(seed=42)
    verts, hit_normals, areas, hit_raw_params = rast._prepare_hit_data(hits)

    # We need the vertex indices for each hit to map learned params
    bary_u = np.array(hits.hit_bary_u)
    bary_v = np.array(hits.hit_bary_v)
    bary_w = 1.0 - bary_u - bary_v
    if hits.vertex_ids_0 is not None:
        vi0 = np.array(hits.vertex_ids_0)
        vi1 = np.array(hits.vertex_ids_1)
        vi2 = np.array(hits.vertex_ids_2)
    else:
        prim_ids = np.array(hits.hit_prim_ids)
        vi0 = rast.faces_np[prim_ids, 0]
        vi1 = rast.faces_np[prim_ids, 1]
        vi2 = rast.faces_np[prim_ids, 2]

    vi0_t = torch.from_numpy(vi0.astype(np.int64)).to(DEVICE)
    vi1_t = torch.from_numpy(vi1.astype(np.int64)).to(DEVICE)
    vi2_t = torch.from_numpy(vi2.astype(np.int64)).to(DEVICE)
    bary_w_t = torch.from_numpy(bary_w.astype(np.float32)).to(DEVICE)
    bary_u_t = torch.from_numpy(bary_u.astype(np.float32)).to(DEVICE)
    bary_v_t = torch.from_numpy(bary_v.astype(np.float32)).to(DEVICE)

    verts_t = torch.from_numpy(verts.astype(np.float32)).to(DEVICE)
    areas_t = torch.from_numpy(areas.astype(np.float32)).to(DEVICE)

    # Range res for metrics
    range_res = compute_range_res_from_cfg(config.config_file)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {scene} ({'from mmIR' if from_mmIR else 'from scratch'})")
        print(f"  Vertices: {N}, Hits: {len(verts)}, MIMO: {n_tx}x{n_rx}")
        print(f"  Iterations: {num_iters}")
        print(f"{'='*60}")

    best_corr = -1.0
    best_iter = 0
    best_state = None

    t0 = time.time()
    for it in range(num_iters):
        optimizer.zero_grad()

        # Interpolate materials and normals at hit positions (barycentric)
        hit_raw = (bary_w_t[:, None] * raw_materials[vi0_t]
                   + bary_u_t[:, None] * raw_materials[vi1_t]
                   + bary_v_t[:, None] * raw_materials[vi2_t])

        hit_nrm = (bary_w_t[:, None] * normals_param[vi0_t]
                   + bary_u_t[:, None] * normals_param[vi1_t]
                   + bary_v_t[:, None] * normals_param[vi2_t])
        hit_nrm = F.normalize(hit_nrm, dim=-1)

        # Inject current patterns into rasterizer
        rast.tx_antenna.E = tx_E
        rast.tx_antenna.H = tx_H
        rast.rx_antenna.E = rx_E
        rast.rx_antenna.H = rx_H

        # Forward pass (differentiable)
        adc_real, adc_imag = rast.render_differentiable(
            hit_raw, hit_nrm, verts_t, areas_t,
            detach_phase=True, chunk_size=1500)

        # Loss
        loss, loss_dict = compute_ra_loss(adc_real, adc_imag, gt_adc_ri)

        # Backward
        loss.backward()

        # Gradient clipping
        for group in optimizer.param_groups:
            clip = clip_vals.get(group["name"], 1.0)
            for p in group["params"]:
                rms_clip_grad(p, clip)

        # LR warmup
        lr_scale = get_lr_scale(it)
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] / (get_lr_scale(max(it-1, 0)) or 1.0) * lr_scale if it > 0 else group["lr"] * lr_scale / 1.0

        # Actually, just scale the base LR directly:
        base_lrs = [0.5, 0.01, 0.05]
        for gi, group in enumerate(optimizer.param_groups):
            group["lr"] = base_lrs[gi] * lr_scale

        optimizer.step()

        # Normalize normals after step
        with torch.no_grad():
            normals_param.data = F.normalize(normals_param.data, dim=-1)

        # Evaluate periodically
        if it % 50 == 0 or it == num_iters - 1:
            with torch.no_grad():
                adc_ri = torch.stack([adc_real, adc_imag], dim=-1).cpu().numpy()
                ra_polar = adc_to_ra_image(torch.from_numpy(adc_ri).float()).numpy()
                ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
                gt_ri_np = gt_adc_ri.cpu().numpy()
                ra_gt_cart = ra_polar_to_cartesian(
                    adc_to_ra_image(torch.from_numpy(gt_ri_np).float()).numpy(),
                    range_res)
                metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)
                cart_corr = metrics['cart_corr']

                if cart_corr > best_corr:
                    best_corr = cart_corr
                    best_iter = it
                    best_state = {
                        'raw_materials': raw_materials.data.clone(),
                        'normals': normals_param.data.clone(),
                        'tx_E': tx_E.data.clone(),
                        'tx_H': tx_H.data.clone(),
                        'rx_E': rx_E.data.clone(),
                        'rx_H': rx_H.data.clone(),
                    }

                if verbose:
                    elapsed = time.time() - t0
                    print(f"  iter {it:4d}: loss={loss_dict['ra_mse']:.6f}, "
                          f"cart_corr={cart_corr:.4f} "
                          f"(best={best_corr:.4f}@{best_iter}) "
                          f"[{elapsed:.1f}s]")

    if verbose:
        print(f"\n  Best cart_corr: {best_corr:.4f} at iter {best_iter}")

    # Save best
    output_dir = os.path.join(PROJECT_ROOT, 'mm25DGS_v2', 'output',
                              'train_mesh' if not from_mmIR else 'train_mesh_mmIR_init',
                              scene)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(best_state, os.path.join(output_dir, 'best_params.pt'))
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'best_cart_corr': best_corr,
            'best_iter': best_iter,
            'num_iters': num_iters,
            'from_mmIR': from_mmIR,
        }, f, indent=2)

    return best_corr, best_iter


def train_all_scenes(num_iters=500, from_mmIR=False):
    """Train all 7 scenes and print comparison table."""
    results = {}
    for scene in SCENES:
        corr, it = train_scene(scene, num_iters=num_iters, from_mmIR=from_mmIR)
        # Load mmIR reference
        mmIR_metrics = json.load(open(
            os.path.join(TRAIN_OUTPUT_DIR, scene, 'best_metrics.json')))
        mmIR_corr = mmIR_metrics['cart_corr']
        results[scene] = {
            'rast_corr': corr,
            'mmIR_corr': mmIR_corr,
            'gap': abs(corr - mmIR_corr),
        }

    print(f"\n{'='*70}")
    print(f"Stage B{'5' if from_mmIR else '4'} Results: "
          f"{'mmIR init' if from_mmIR else 'Scratch'} Training")
    print(f"{'='*70}")
    print(f"{'Scene':<25} {'mmIR':>8} {'Rast':>8} {'Gap':>6} {'Status':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")

    target = 0.01 if from_mmIR else 0.05
    all_pass = True
    for scene in SCENES:
        r = results[scene]
        status = "PASS" if r['gap'] < target else "FAIL"
        if r['gap'] >= target:
            all_pass = False
        print(f"{scene:<25} {r['mmIR_corr']:>8.4f} {r['rast_corr']:>8.4f} "
              f"{r['gap']:>6.4f} {status:>8}")

    print(f"\nTarget: gap < {target:.2f} per scene")
    print(f"{'OVERALL: PASS' if all_pass else 'OVERALL: FAIL'}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--from-mmIR', action='store_true',
                        help='Initialize from mmIR trained params (B5)')
    parser.add_argument('--iters', type=int, default=500)
    args = parser.parse_args()

    if args.all:
        train_all_scenes(num_iters=args.iters, from_mmIR=args.from_mmIR)
    elif args.scene:
        train_scene(args.scene, num_iters=args.iters, from_mmIR=args.from_mmIR)
    else:
        print("Usage: --scene <name> or --all")
