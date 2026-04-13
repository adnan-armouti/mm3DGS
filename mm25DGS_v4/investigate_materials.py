"""Investigation suite for why the material model under-learns.

Three diagnostics:

  D1. Loss invariance test — add ε to each column of raw_materials on a
      fully-trained model, measure how cart_corr responds. Flat response
      => loss is gauge-invariant to that parameter.

  D2. Per-iter gradient statistics — captured during training, dumped to
      .npz. Plots grad_mean[t, k] (gauge-invariant global direction) and
      grad_std[t, k] (gauge-variant per-point direction) over training.

  D3. Gauge-variant trajectory decomposition — from the existing trajectory
      checkpoints, computes mean_over_points(raw_materials[t, :, k]) and
      std_over_points(raw_materials[t, :, k]) over training.

Outputs: a .npz per run + a set of PNG plots under the run's diagnostics dir.

Usage:
  # Step 1: train a fresh run with grad stats enabled
  python -m mm25DGS_v4.investigate_materials --run scene_135 --iters 500

  # Step 2: analyze an existing run (reads npz + trained model state)
  python -m mm25DGS_v4.investigate_materials --analyze mm25DGS_v4/output/material_investigation/scene_135
"""

import os
import sys
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import mm25DGS_v4.train_gaussian as tg
from mm25DGS_v4.train_gaussian import (
    render_gaussians, polar_to_cart_torch, cart_corr_torch,
    range_profile_to_ra_mag, build_polar_to_cart_grid, cull_gaussians,
    init_visible_weighted,
    DEVICE,
)
from mm25DGS_v4.rasterizer import Rasterizer
from mm25DGS_v4.load_pretrained import load_trained_config, load_pattern_data
from mm25DGS_v4.material_diagnostics import PARAM_NAMES, MaterialDiagnostics
from mmir.data.ra_utils import adc_to_ra_image
from mmir.data.io_utils import compute_range_res_from_cfg


OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', 'material_investigation')


# =========================================================================
# D1 — Loss-invariance test
# =========================================================================

def d1_loss_invariance(scene, epsilons=None, verbose=True):
    """For each material column k, perturb raw_materials[:, k] by ε values
    and measure the resulting cart_corr. Returns a dict with per-column
    cart_corr curves.

    Loads the full trained state (model + antenna patterns) from the
    `best_model.pt` saved by `train_gaussians` in the scene's output dir.
    This ensures the baseline matches the training best — without this,
    only raw_materials would be trained and rotations+patterns would be
    back at init, giving a misleading baseline.

    Flat curve over a wide range of ε ⟹ loss is gauge-invariant to that column.
    """
    if epsilons is None:
        epsilons = np.linspace(-2.0, 2.0, 21, dtype=np.float32)

    # Rebuild the scene exactly as a fresh training run would
    config = load_trained_config(scene)
    pattern_data = load_pattern_data(scene)
    rast = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )
    rast.inject_trained_params(pattern_data=pattern_data)
    model = init_visible_weighted(scene, rast, target_n=50000)
    rast.free_mi_scene()

    # Inject the FULL trained state (model weights + antenna patterns)
    best_state_path = os.path.join(PROJECT_ROOT, 'mm25DGS_v4', 'output', scene, 'best_model.pt')
    if not os.path.exists(best_state_path):
        raise FileNotFoundError(f"best_model.pt missing at {best_state_path} — train first")
    best_state = torch.load(best_state_path, map_location=DEVICE, weights_only=False)
    with torch.no_grad():
        for k, v in best_state['model'].items():
            getattr(model, k).copy_(v)
        if 'tx_E' in best_state:
            rast.tx_antenna.E = best_state['tx_E'].clone()
            rast.tx_antenna.H = best_state['tx_H'].clone()
            rast.rx_antenna.E = best_state['rx_E'].clone()
            rast.rx_antenna.H = best_state['rx_H'].clone()

    base_raw = model.raw_materials.detach().clone()

    # GT cart for the metric — match train_gaussians pipeline exactly
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_adc_ri = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32)).to(DEVICE)
    range_res = compute_range_res_from_cfg(config.config_file)

    _gt_adc_for_eval = torch.from_numpy(gt_adc_ri.cpu().numpy()).float()
    _gt_ra_polar_cpu = adc_to_ra_image(_gt_adc_for_eval).numpy()
    _gt_polar_gpu = torch.from_numpy(_gt_ra_polar_cpu.astype(np.float32)).to(DEVICE)
    n_az, n_range = _gt_polar_gpu.shape
    sample_grid = build_polar_to_cart_grid(n_az, n_range, range_res, grid_res=400, device=DEVICE)
    _gt_cart_gpu = polar_to_cart_torch(_gt_polar_gpu, sample_grid)
    gt_cart_norm = (_gt_cart_gpu - _gt_cart_gpu.min()) / (_gt_cart_gpu.max() - _gt_cart_gpu.min()).clamp(min=1e-30)

    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0

    @torch.no_grad()
    def _eval_cart_corr():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None, bsdf_mode='full')
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
        return cart_corr_torch(ra_cart, gt_cart_norm).item()

    baseline = _eval_cart_corr()
    if verbose:
        print(f"  Baseline cart_corr at trained state: {baseline:.4f}")

    # Per-column sweep: only column k gets offset ε, others stay at base
    results = {}
    for k, name in enumerate(PARAM_NAMES):
        corrs = []
        for eps in epsilons:
            with torch.no_grad():
                model.raw_materials.copy_(base_raw)
                model.raw_materials[:, k] += float(eps)
            c = _eval_cart_corr()
            corrs.append(c)
        results[name] = np.array(corrs, dtype=np.float32)
        if verbose:
            flat = corrs[0] - corrs[-1]
            print(f"  col {k} ({name:10s}): corr range over ε∈[{epsilons[0]:+.1f}, {epsilons[-1]:+.1f}] = "
                  f"[{min(corrs):.4f}, {max(corrs):.4f}]  Δ={max(corrs)-min(corrs):.4f}")

    # Also: uniform-offset test (add ε to ALL columns simultaneously)
    uniform = []
    for eps in epsilons:
        with torch.no_grad():
            model.raw_materials.copy_(base_raw)
            model.raw_materials += float(eps)
        uniform.append(_eval_cart_corr())
    if verbose:
        print(f"  uniform-offset: Δ = {max(uniform) - min(uniform):.4f}")

    with torch.no_grad():
        model.raw_materials.copy_(base_raw)
    return {
        'baseline': baseline,
        'epsilons': epsilons,
        'per_column': results,
        'uniform': np.array(uniform, dtype=np.float32),
    }


# =========================================================================
# D2/D3 — Gradient and trajectory analysis (reads .npz, produces summary)
# =========================================================================

def analyze_run(npz_path, verbose=True):
    """Reads a MaterialDiagnostics .npz dump and summarizes D2 + D3."""
    d = np.load(npz_path, allow_pickle=True)
    init = d['init']              # (M, 6)
    final = d['final']            # (M, 6)
    traj = d['trajectory']        # (T_ckpt, M, 6)
    ckpt_iters = d['checkpoint_iters']  # (T_ckpt,)
    drift = d['drift']            # (6,)
    fisher = d['fisher']          # (6,)

    out = {
        'drift': drift,
        'fisher': fisher,
        'traj_mean': traj.mean(axis=1),  # (T, 6) — gauge-invariant direction
        'traj_std': traj.std(axis=1),    # (T, 6) — gauge-variant direction
        'ckpt_iters': ckpt_iters,
        'final_std': final.std(axis=0),  # (6,) — per-point diversity at end
        'init_std': init.std(axis=0),    # (6,) — per-point diversity at init
    }
    if 'grad_mean' in d.files:
        out['grad_mean'] = d['grad_mean']   # (T_iter, 6)
        out['grad_std'] = d['grad_std']     # (T_iter, 6)
        out['grad_iters'] = d['grad_iters']

    if verbose:
        print("\n--- D3: Gauge-variant trajectory decomposition ---")
        print(f"{'param':<12} {'init_std':>12} {'final_std':>12} {'drift':>12} {'fisher':>12}")
        for k, name in enumerate(PARAM_NAMES):
            print(f"  {name:<10} {init.std(axis=0)[k]:>12.4e} "
                  f"{final.std(axis=0)[k]:>12.4e} {drift[k]:>12.4e} "
                  f"{fisher[k]:>12.4e}")

        if 'grad_mean' in d.files:
            gm = d['grad_mean']
            gs = d['grad_std']
            print("\n--- D2: Per-iter gradient stats (averaged over training) ---")
            print(f"{'param':<12} {'<|grad.mean|>':>16} {'<grad.std>':>16} {'ratio':>10}")
            for k, name in enumerate(PARAM_NAMES):
                mean_of_mean = np.mean(np.abs(gm[:, k]))
                mean_of_std = np.mean(gs[:, k])
                ratio = mean_of_std / max(mean_of_mean, 1e-30)
                print(f"  {name:<10} {mean_of_mean:>16.4e} {mean_of_std:>16.4e} {ratio:>10.2f}")
    return out


# =========================================================================
# Plotting
# =========================================================================

def plot_investigation(analysis, d1_results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: D1 loss invariance
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    eps = d1_results['epsilons']
    baseline = d1_results['baseline']
    for name, corrs in d1_results['per_column'].items():
        ax.plot(eps, corrs, label=name, lw=2)
    ax.plot(eps, d1_results['uniform'], label='uniform (all 6)',
            lw=3, ls='--', color='k')
    ax.axhline(baseline, color='gray', ls=':', label=f'baseline {baseline:.4f}')
    ax.set_xlabel('ε added to raw_materials[:, k]')
    ax.set_ylabel('cart_corr')
    ax.set_title('D1: Loss invariance (flat = gauge-invariant to that column)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'd1_loss_invariance.png'), dpi=120)
    plt.close(fig)

    # Plot 2: D3 trajectory mean vs std
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    traj_mean = analysis['traj_mean']
    traj_std = analysis['traj_std']
    iters = analysis['ckpt_iters']
    for k, name in enumerate(PARAM_NAMES):
        ax = axes.flat[k]
        ax.plot(iters, traj_mean[:, k], label='mean over points', lw=2, color='C0')
        ax.set_ylabel('mean (gauge-invariant)', color='C0')
        ax.tick_params(axis='y', labelcolor='C0')
        ax2 = ax.twinx()
        ax2.plot(iters, traj_std[:, k], label='std over points', lw=2, color='C3')
        ax2.set_ylabel('std (gauge-variant)', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
        ax.set_xlabel('iter')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
    fig.suptitle('D3: Gauge-variant trajectory decomposition (mean = uniform drift, std = per-point diversity)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'd3_trajectory.png'), dpi=120)
    plt.close(fig)

    # Plot 3: D2 grad stats (if available)
    if 'grad_mean' in analysis:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        gm = analysis['grad_mean']
        gs = analysis['grad_std']
        gi = analysis['grad_iters']
        for k, name in enumerate(PARAM_NAMES):
            ax = axes.flat[k]
            ax.semilogy(gi, np.abs(gm[:, k]) + 1e-30, label='|mean| (gauge)', color='C0', lw=1)
            ax.semilogy(gi, gs[:, k] + 1e-30, label='std (per-point)', color='C3', lw=1)
            ax.set_xlabel('iter')
            ax.set_ylabel('gradient magnitude (log)')
            ax.set_title(f'{name}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
        fig.suptitle('D2: Per-iter gradient statistics (mean = null-space, std = useful direction)')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'd2_grad_stats.png'), dpi=120)
        plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='seq_0_frame_135')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--run-name', default='D_scene_135')
    args = parser.parse_args()

    run_dir = os.path.join(OUTPUT_ROOT, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Step 1: Train with grad stats enabled
    print(f"\n=== Step 1: Training {args.scene} for {args.iters} iters with grad stats ===")
    corr, _ = tg.train_gaussians(
        args.scene, num_iters=args.iters, verbose=True,
        diagnostics_dir=run_dir, run_name=args.run_name,
        capture_grad_stats=True)
    print(f"  cart_corr = {corr:.4f}")

    # Step 2: Analyze the dump (D2 + D3)
    npz_path = os.path.join(run_dir, f'{args.run_name}__{args.scene}.npz')
    print(f"\n=== Step 2: Analyzing {npz_path} ===")
    analysis = analyze_run(npz_path)

    # Step 3: D1 loss invariance test
    print(f"\n=== Step 3: D1 loss invariance test ===")
    d1_results = d1_loss_invariance(args.scene)

    # Step 4: Plots
    print(f"\n=== Step 4: Plotting ===")
    plot_investigation(analysis, d1_results, run_dir)
    print(f"  Plots in {run_dir}/")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Trained cart_corr: {corr:.4f}")
    print(f"D1 uniform-offset Δ: {d1_results['uniform'].max() - d1_results['uniform'].min():.4f}")
    print(f"D1 per-column Δ (max over columns):")
    for k, name in enumerate(PARAM_NAMES):
        corrs = d1_results['per_column'][name]
        print(f"  {name:<10}: [{corrs.min():.4f}, {corrs.max():.4f}]  Δ={corrs.max()-corrs.min():.4f}")

    # Save analysis as npz too
    np.savez_compressed(
        os.path.join(run_dir, 'analysis.npz'),
        d1_epsilons=d1_results['epsilons'],
        d1_uniform=d1_results['uniform'],
        **{f'd1_{k}': v for k, v in d1_results['per_column'].items()},
        traj_mean=analysis['traj_mean'],
        traj_std=analysis['traj_std'],
        drift=analysis['drift'],
        fisher=analysis['fisher'],
    )
    print(f"  Analysis saved to {run_dir}/analysis.npz")


if __name__ == '__main__':
    main()
