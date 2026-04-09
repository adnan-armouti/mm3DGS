"""
Step 0: Reproduce mmIR's training-time forward pass.

Loads trained parameters (materials, normals, patterns) and calls mmIR's
render_end_to_end() to reproduce the exact ADC output from training.

This is NOT a new renderer — it IS mmIR, called with saved best params.
"""

import os
import sys
import json
import numpy as np
import torch

# Must set variant BEFORE any mmir imports that use mi types at class level
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train import (
    load_config_from_json,
    create_render_config,
    create_renderer,
    ParameterManagerSionna,
    extract_adc_for_metrics,
)
from mmir.data.ra_utils import adc_to_ra_image, ra_polar_to_cartesian, compute_cartesian_ra_metrics
from mmir.data.io_utils import compute_range_res_from_cfg
from mmir.renderer.bsdf.reparameterization import create_drjit_raw_params, reparameterize_physics_params_drjit

# ============================================================================
# Constants
# ============================================================================

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]

TRAIN_OUTPUT_DIR = '/home/adnan/Desktop/mmIR/output/train_v13'
DATA_DIR = '/home/adnan/Desktop/mm3DGS/data'
ASSETS_DIR = '/home/adnan/Desktop/mm3DGS/assets'
GOLD_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold_references')


def remap_path(path: str) -> str:
    """Remap paths from /home/adnan/Desktop/mmIR/ to /home/adnan/Desktop/mm3DGS/."""
    return path.replace('/home/adnan/Desktop/mmIR/', '/home/adnan/Desktop/mm3DGS/')


def load_trained_config(scene: str) -> 'TrainingConfigSionna':
    """Load and path-remap the training config for a scene."""
    config_path = os.path.join(TRAIN_OUTPUT_DIR, scene, 'config.json')
    config = load_config_from_json(config_path)

    # Remap all file paths to mm3DGS data directory
    config.config_file = remap_path(config.config_file)
    config.scene_file = remap_path(config.scene_file)
    config.gt_adc_file = remap_path(config.gt_adc_file)
    config.tx_pattern_file = remap_path(config.tx_pattern_file)
    config.rx_pattern_file = remap_path(config.rx_pattern_file)

    return config


def load_best_params(scene: str):
    """Load best trained parameters for a scene.

    Returns:
        raw_params: (N_verts, 6) unconstrained material params
        normal_params: (N_verts, 3) vertex normals (or None)
        pattern_data: dict with tx_E_plane, tx_H_plane, rx_E_plane, rx_H_plane (or None)
    """
    train_dir = os.path.join(TRAIN_OUTPUT_DIR, scene)

    # Materials (always present)
    mat = np.load(os.path.join(train_dir, 'best_materials.npz'))
    raw_params = mat['raw_params']

    # Normals (may not exist)
    normal_params = None
    nrm_path = os.path.join(train_dir, 'best_normals.npz')
    if os.path.exists(nrm_path):
        nrm = np.load(nrm_path)
        normal_params = nrm['normal_params']

    # Patterns (may not exist)
    pattern_data = None
    pat_path = os.path.join(train_dir, 'best_patterns.npz')
    if os.path.exists(pat_path):
        pat = np.load(pat_path)
        pattern_data = {k: pat[k] for k in pat.files}

    return raw_params, normal_params, pattern_data


def inject_patterns(renderer, pattern_data):
    """Inject learned antenna patterns into the renderer's pattern loaders."""
    if pattern_data is None:
        return

    tx_loader = renderer.tx_pattern_loader
    rx_loader = renderer.rx_pattern_loader

    if tx_loader is not None and 'tx_E_plane' in pattern_data:
        tx_loader.E_plane_linear = mi.Float(pattern_data['tx_E_plane'].astype(np.float32))
        tx_loader.H_plane_linear = mi.Float(pattern_data['tx_H_plane'].astype(np.float32))

    if rx_loader is not None and 'rx_E_plane' in pattern_data:
        rx_loader.E_plane_linear = mi.Float(pattern_data['rx_E_plane'].astype(np.float32))
        rx_loader.H_plane_linear = mi.Float(pattern_data['rx_H_plane'].astype(np.float32))


def render_scene(scene: str, save_gold=True, verbose=True):
    """Render a single scene using mmIR with trained parameters.

    Reproduces the exact training-time forward pass:
    1. Load config and create renderer (same as training)
    2. Load best materials, normals, patterns
    3. Create ParameterManager, inject trained params
    4. Call render_end_to_end with same seed as best iteration

    Returns:
        dict with keys: cart_corr, ra_cart, adc_ri, ra_polar
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Rendering: {scene}")
        print(f"{'='*60}")

    # 1. Load config
    config = load_trained_config(scene)

    # Force single-bounce: the rasterizer is single-bounce, so the gold
    # reference should be single-bounce too. Multi-bounce MC noise is
    # non-reproducible across fresh renderer instances.
    config.max_bounces = 1

    # 2. Create renderer (same as training)
    renderer = create_renderer(config)

    # 3. Load trained params
    raw_params, normal_params, pattern_data = load_best_params(scene)
    if verbose:
        print(f"  Materials: {raw_params.shape}")
        if normal_params is not None:
            print(f"  Normals: {normal_params.shape}")
        if pattern_data is not None:
            print(f"  Patterns: {list(pattern_data.keys())}")

    # 4. Create parameter manager and inject trained params
    pm = ParameterManagerSionna(renderer, config, config.scene_file)

    # Inject materials
    pm.raw_params = raw_params.copy()

    # Inject normals
    if normal_params is not None and config.LEARN_NRM:
        pm.normal_params_np = normal_params.copy()

    # Inject patterns
    inject_patterns(renderer, pattern_data)

    # 5. Set renderer materials (for SMS solver etc.)
    pm.set_renderer_materials()

    # 6. Create grad-enabled params (same as training forward pass)
    raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr = \
        pm.create_grad_enabled_params()

    # 7. Compute seed for best iteration
    # Training uses: seed = 42 + (iteration // seed_interval) * 7
    best_metrics = json.load(open(
        os.path.join(TRAIN_OUTPUT_DIR, scene, 'best_metrics.json')))
    best_iter = best_metrics.get('best_iteration', 489)
    seed_interval = max(config.e2e_ray_seed_rotation_interval, 1)
    seed = 42 + (best_iter // seed_interval) * 7
    if verbose:
        print(f"  Best iteration: {best_iter}, seed: {seed}")

    # 8. Render (same call as run_end_to_end_forward)
    with dr.suspend_grad():
        adc_real, adc_imag = renderer.render_end_to_end(
            raw_drjit,
            pose_params=pose_dr,
            normal_params=normal_dr,
            vertex_offset_params=vertex_offset_dr,
            pattern_loaders=pattern_loaders_dr,
            seed=seed,
        )

    # 9. Extract ADC numpy
    NT, NR, K = pm.NT, pm.NR, pm.K
    real_np = np.array(adc_real).reshape(NT, NR, K)
    imag_np = np.array(adc_imag).reshape(NT, NR, K)
    adc_ri = np.stack([real_np, imag_np], axis=-1)  # (NT, NR, K, 2)

    # 10. ADC -> RA polar -> RA cartesian
    adc_torch = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_torch).detach().cpu().numpy()

    range_res = compute_range_res_from_cfg(config.config_file)
    ra_cart = ra_polar_to_cartesian(ra_polar, range_res)

    # 11. Load GT and compute metrics
    gt_adc_np = np.load(config.gt_adc_file)
    if gt_adc_np.ndim == 4:
        gt_adc_single = gt_adc_np[0]
    else:
        gt_adc_single = gt_adc_np
    gt_ri = np.stack([gt_adc_single.real, gt_adc_single.imag], axis=-1)
    gt_adc_torch = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32))
    ra_gt_polar = adc_to_ra_image(gt_adc_torch).detach().cpu().numpy()
    ra_gt_cart = ra_polar_to_cartesian(ra_gt_polar, range_res)

    metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)
    cart_corr = metrics['cart_corr']

    # Also compare against saved RA cart from training
    saved_ra_cart_path = os.path.join(TRAIN_OUTPUT_DIR, scene, 'ra_rendered_cart.npy')
    saved_ra_corr = None
    if os.path.exists(saved_ra_cart_path):
        saved_ra_cart = np.load(saved_ra_cart_path)
        # Correlation between our render and the saved training render
        def _minmax(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn) if (mx - mn) > 1e-30 else np.zeros_like(arr)
        saved_ra_corr = float(np.corrcoef(
            _minmax(ra_cart).ravel(), _minmax(saved_ra_cart).ravel())[0, 1])

    if verbose:
        target_corr = best_metrics['cart_corr']
        diff = abs(cart_corr - target_corr)
        # MC noise is +/-0.03 per scene per run (see CLAUDE.md pitfalls)
        status = "PASS" if diff < 0.03 else "FAIL"
        print(f"  cart_corr: {cart_corr:.4f} (target: {target_corr:.4f}, diff: {diff:.4f}) [{status}]")
        if saved_ra_corr is not None:
            print(f"  corr with saved RA: {saved_ra_corr:.4f}")

    # 12. Save gold references
    if save_gold:
        gold_dir = os.path.join(GOLD_REF_DIR, scene)
        os.makedirs(gold_dir, exist_ok=True)
        np.save(os.path.join(gold_dir, 'adc_ri.npy'), adc_ri)
        np.save(os.path.join(gold_dir, 'ra_polar.npy'), ra_polar)
        np.save(os.path.join(gold_dir, 'ra_cart.npy'), ra_cart)
        np.save(os.path.join(gold_dir, 'ra_gt_cart.npy'), ra_gt_cart)
        with open(os.path.join(gold_dir, 'cart_corr.txt'), 'w') as f:
            f.write(f"{cart_corr:.6f}\n")
        if verbose:
            print(f"  Gold references saved to {gold_dir}")

    return {
        'scene': scene,
        'cart_corr': cart_corr,
        'target_corr': best_metrics['cart_corr'],
        'saved_ra_corr': saved_ra_corr,
        'ra_cart': ra_cart,
        'ra_gt_cart': ra_gt_cart,
        'adc_ri': adc_ri,
        'ra_polar': ra_polar,
    }


def render_all_scenes(save_gold=True):
    """Render all 7 scenes and print verification table."""
    results = []
    for scene in SCENES:
        result = render_scene(scene, save_gold=save_gold)
        results.append(result)

    # Print verification table
    print(f"\n{'='*80}")
    print("Step 0 Verification: Reproduce mmIR Forward Pass")
    print(f"{'='*80}")
    print(f"{'Scene':<25} {'mmIR':>8} {'Ours':>8} {'Diff':>8} {'Saved RA corr':>15} {'Status':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*15} {'-'*8}")

    all_pass = True
    for r in results:
        diff = abs(r['cart_corr'] - r['target_corr'])
        status = "PASS" if diff < 0.03 else "FAIL"
        if diff >= 0.03:
            all_pass = False
        saved_str = f"{r['saved_ra_corr']:.4f}" if r['saved_ra_corr'] is not None else "N/A"
        print(f"{r['scene']:<25} {r['target_corr']:>8.4f} {r['cart_corr']:>8.4f} "
              f"{diff:>8.4f} {saved_str:>15} {status:>8}")

    print(f"\n{'OVERALL: PASS' if all_pass else 'OVERALL: FAIL'}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None,
                        help='Single scene to render (default: all)')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving gold references')
    args = parser.parse_args()

    if args.scene:
        render_scene(args.scene, save_gold=not args.no_save)
    else:
        render_all_scenes(save_gold=not args.no_save)
