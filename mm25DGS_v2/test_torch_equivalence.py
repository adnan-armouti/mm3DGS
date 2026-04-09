"""
Stage A verification: Compare PyTorch rasterizer against DrJit rasterizer.

Tests:
  A1: Material reparameterization
  A2: BSDF
  A3: Antenna gain
  A5: Full forward pass (ADC -> RA -> cart_corr)

Run: CUDA_VISIBLE_DEVICES=0 python -m mm25DGS_v2.test_torch_equivalence [--scene X]
"""

import os
import sys
import json
import numpy as np
import torch
import time

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmir.renderer.bsdf.reparameterization import (
    reparameterize_physics_params_drjit, create_drjit_raw_params)
from mmir.sensor.element_patterns import evaluate_combined_gain, AntennaPatternLoader
from mmir.data.ra_utils import (
    adc_to_ra_image, ra_polar_to_cartesian, compute_cartesian_ra_metrics)
from mmir.data.io_utils import compute_range_res_from_cfg

from mm25DGS_v2.render_mmIR import (
    TRAIN_OUTPUT_DIR, GOLD_REF_DIR, load_trained_config, load_best_params)
from mm25DGS_v2.rasterizer_torch import reparameterize_torch, AntennaPatternTorch
from mm25DGS.bsdf_torch import evaluate_bsdf_jones_f_cos


SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]


# =========================================================================
# A1: Material reparameterization
# =========================================================================

def test_reparameterization(scene):
    """Verify PyTorch reparameterization matches DrJit for all vertices."""
    raw_params, _, _ = load_best_params(scene)
    N = raw_params.shape[0]

    # DrJit
    raw_dr = [mi.Float(raw_params[:, c].astype(np.float32)) for c in range(6)]
    physics_dr = reparameterize_physics_params_drjit(raw_dr)
    dr_out = np.column_stack([np.array(physics_dr[i]) for i in range(6)])

    # PyTorch
    raw_t = torch.from_numpy(raw_params.astype(np.float32)).cuda()
    physics_t = reparameterize_torch(raw_t)
    torch_out = physics_t.cpu().numpy()

    # Use relative error (some params like eps_imag can be ~1e7)
    rel_err = np.abs(dr_out - torch_out) / np.maximum(np.abs(dr_out), 1e-10)
    max_rel = np.max(rel_err)
    per_col_rel = [np.max(rel_err[:, c]) for c in range(6)]
    col_names = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thickness']

    status = "PASS" if max_rel < 1e-5 else "FAIL"
    print(f"  A1 Reparam {scene}: max_rel_err={max_rel:.2e} [{status}]")
    for name, err in zip(col_names, per_col_rel):
        flag = " !!!" if err > 1e-5 else ""
        print(f"      {name:12s}: {err:.2e}{flag}")

    return max_rel < 1e-5


# =========================================================================
# A2: BSDF
# =========================================================================

def test_bsdf(scene, n_test=500):
    """Verify PyTorch Jones BSDF matches DrJit for random directions."""
    raw_params, normal_params, _ = load_best_params(scene)
    config = load_trained_config(scene)

    import trimesh
    mesh = trimesh.load(config.scene_file)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    if normal_params is None:
        normals = np.array(mesh.vertex_normals, dtype=np.float32)
    else:
        normals = normal_params.astype(np.float32)

    N = min(n_test, raw_params.shape[0])
    idx = np.random.RandomState(42).choice(raw_params.shape[0], N, replace=False)

    # Get physics params
    raw_sub = raw_params[idx].astype(np.float32)
    nrm_sub = normals[idx]
    pos_sub = vertices[idx]

    # Create directions (TX and RX roughly facing scene)
    tx_pos = np.array([e['pos_mm'] for e in json.load(open(config.config_file))['tx_array']],
                      dtype=np.float32) / 1000.0
    rx_pos = np.array([e['pos_mm'] for e in json.load(open(config.config_file))['rx_array']],
                      dtype=np.float32) / 1000.0
    radar_center = (tx_pos.mean(0) + rx_pos.mean(0)) / 2

    # wi = toward radar, wo = toward radar (monostatic approx)
    wi = radar_center - pos_sub
    wi = wi / np.maximum(np.linalg.norm(wi, axis=1, keepdims=True), 1e-6)
    wo = wi.copy()

    # Normal flip
    cos_out = np.sum(wo * nrm_sub, axis=1)
    flip = cos_out < 0
    nrm_sub[flip] *= -1

    cos_theta_i = np.abs(np.sum(wi * nrm_sub, axis=1)).clip(1e-6)

    # DrJit BSDF
    from mmir.renderer.bsdf.mmwave_jones import BSDFmmWaveJones
    bsdf = BSDFmmWaveJones(
        tx_polarization=mi.Vector3f(0, 0, 1),
        rx_polarization=mi.Vector3f(0, 0, 1),
        enable_incoherent=True,
        use_fresnel_phase=False,
        enable_cbs=True,
    )

    raw_dr = [mi.Float(raw_sub[:, c]) for c in range(6)]
    physics_dr = reparameterize_physics_params_drjit(raw_dr)

    dr_result = bsdf.eval_f_cos_physics(
        wo=mi.Vector3f(mi.Float(wo[:, 0]), mi.Float(wo[:, 1]), mi.Float(wo[:, 2])),
        wi=mi.Vector3f(mi.Float(wi[:, 0]), mi.Float(wi[:, 1]), mi.Float(wi[:, 2])),
        n=mi.Vector3f(mi.Float(nrm_sub[:, 0]), mi.Float(nrm_sub[:, 1]), mi.Float(nrm_sub[:, 2])),
        eps_real=physics_dr[0], eps_imag=physics_dr[1],
        sigma_h=physics_dr[2], l_c=physics_dr[3],
        tau=physics_dr[4], thickness=physics_dr[5],
    )
    dr_val = np.array(dr_result)

    # PyTorch BSDF
    raw_t = torch.from_numpy(raw_sub).cuda()
    physics_t = reparameterize_torch(raw_t)

    torch_result = evaluate_bsdf_jones_f_cos(
        cos_theta_i=torch.from_numpy(cos_theta_i.astype(np.float32)).cuda(),
        wo=torch.from_numpy(wo).cuda(),
        wi=torch.from_numpy(wi).cuda(),
        n=torch.from_numpy(nrm_sub).cuda(),
        eps_real=physics_t[:, 0], eps_imag=physics_t[:, 1],
        sigma_h=physics_t[:, 2], l_c=physics_t[:, 3],
        tau_base=physics_t[:, 4], thickness=physics_t[:, 5],
    )
    torch_val = torch_result.cpu().numpy()

    # Compare
    valid = dr_val > 1e-10
    if valid.sum() == 0:
        print(f"  A2 BSDF {scene}: no valid BSDF values (all zero)")
        return True

    rel_err = np.abs(dr_val[valid] - torch_val[valid]) / np.maximum(np.abs(dr_val[valid]), 1e-10)
    max_rel = rel_err.max()
    mean_rel = rel_err.mean()
    med_rel = np.median(rel_err)

    status = "PASS" if med_rel < 0.01 else "FAIL"
    print(f"  A2 BSDF {scene}: max_rel={max_rel:.4f}, mean_rel={mean_rel:.4f}, "
          f"median_rel={med_rel:.4f} ({valid.sum()}/{N} valid) [{status}]")

    return med_rel < 0.01


# =========================================================================
# A3: Antenna gain
# =========================================================================

def test_antenna(scene, n_test=1000):
    """Verify PyTorch antenna gain matches DrJit."""
    config = load_trained_config(scene)
    _, _, pattern_data = load_best_params(scene)

    # Load DrJit patterns
    tx_loader_dr = AntennaPatternLoader(config.tx_pattern_file)
    rx_loader_dr = AntennaPatternLoader(config.rx_pattern_file)
    if pattern_data is not None:
        tx_loader_dr.E_plane_linear = mi.Float(pattern_data['tx_E_plane'].astype(np.float32))
        tx_loader_dr.H_plane_linear = mi.Float(pattern_data['tx_H_plane'].astype(np.float32))
        rx_loader_dr.E_plane_linear = mi.Float(pattern_data['rx_E_plane'].astype(np.float32))
        rx_loader_dr.H_plane_linear = mi.Float(pattern_data['rx_H_plane'].astype(np.float32))

    # PyTorch patterns
    tx_torch = AntennaPatternTorch(config.tx_pattern_file, device="cuda:0")
    rx_torch = AntennaPatternTorch(config.rx_pattern_file, device="cuda:0")
    if pattern_data is not None:
        tx_torch.inject_patterns(pattern_data['tx_E_plane'], pattern_data['tx_H_plane'])
        rx_torch.inject_patterns(pattern_data['rx_E_plane'], pattern_data['rx_H_plane'])

    # Random directions and boresights
    rng = np.random.RandomState(42)
    dirs = rng.randn(n_test, 3).astype(np.float32)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-6)

    # Boresights from config
    cfg = json.load(open(config.config_file))
    tx_bore = np.array(cfg['tx_array'][0]['boresight'], dtype=np.float32)
    boresights = np.tile(tx_bore, (n_test, 1))

    # DrJit
    dr_dirs = mi.Vector3f(mi.Float(dirs[:, 0]), mi.Float(dirs[:, 1]), mi.Float(dirs[:, 2]))
    dr_bore = mi.Vector3f(mi.Float(boresights[:, 0]), mi.Float(boresights[:, 1]),
                          mi.Float(boresights[:, 2]))
    dr_gain = np.array(evaluate_combined_gain(tx_loader_dr, dr_dirs, dr_bore))

    # PyTorch
    torch_gain = tx_torch.evaluate(
        torch.from_numpy(dirs).cuda(),
        torch.from_numpy(boresights).cuda()
    ).cpu().numpy()

    valid = dr_gain > 1e-10
    if valid.sum() == 0:
        print(f"  A3 Antenna {scene}: no valid gains")
        return True

    rel_err = np.abs(dr_gain[valid] - torch_gain[valid]) / np.maximum(dr_gain[valid], 1e-10)
    max_rel = rel_err.max()
    mean_rel = rel_err.mean()

    status = "PASS" if mean_rel < 0.01 else "FAIL"
    print(f"  A3 Antenna {scene}: max_rel={max_rel:.4f}, mean_rel={mean_rel:.4f} "
          f"({valid.sum()}/{n_test} valid) [{status}]")

    return mean_rel < 0.01


# =========================================================================
# A5: Full forward pass
# =========================================================================

def test_full_forward(scene, verbose=True):
    """Compare PyTorch rasterizer ADC/RA against DrJit rasterizer (direct comparison).

    Both rasterizers use the same reservoir sampler seed, so hits are identical.
    The comparison is between the physics pipelines (BSDF, antenna, phase).
    """
    from mm25DGS_v2.rasterizer_torch import RasterizerTorch
    from mm25DGS_v2.rasterizer import Rasterizer

    config = load_trained_config(scene)
    raw_params, normal_params, pattern_data = load_best_params(scene)

    if verbose:
        print(f"\n  A5 Full forward: {scene}")

    # DrJit rasterizer
    rast_dr = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
    )
    rast_dr.inject_trained_params(raw_params, normal_params, pattern_data)
    adc_dr = rast_dr.render(verbose=False, chunk_size=1500)

    # PyTorch rasterizer
    rast_pt = RasterizerTorch(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
    )
    rast_pt.inject_trained_params(raw_params, normal_params, pattern_data)
    adc_pt = rast_pt.render(verbose=False, chunk_size=1500)

    # ADC -> RA -> Cartesian
    range_res = compute_range_res_from_cfg(config.config_file)
    def _to_ra_cart(adc_ri):
        t = torch.from_numpy(adc_ri).float()
        ra = adc_to_ra_image(t).detach().cpu().numpy()
        return ra_polar_to_cartesian(ra, range_res)

    ra_dr = _to_ra_cart(adc_dr)
    ra_pt = _to_ra_cart(adc_pt)

    # GT for cart_corr
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_torch = torch.from_numpy(gt_ri.transpose(1, 0, 2, 3).astype(np.float32))
    ra_gt_cart = ra_polar_to_cartesian(adc_to_ra_image(gt_torch).numpy(), range_res)

    corr_dr = compute_cartesian_ra_metrics(ra_dr, ra_gt_cart)['cart_corr']
    corr_pt = compute_cartesian_ra_metrics(ra_pt, ra_gt_cart)['cart_corr']

    # Direct torch-vs-drjit RA correlation
    def _mm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(a)

    ra_corr = float(np.corrcoef(_mm(ra_dr).ravel(), _mm(ra_pt).ravel())[0, 1])
    cart_corr_diff = abs(corr_pt - corr_dr)

    status = "PASS" if (ra_corr > 0.99 and cart_corr_diff < 0.01) else "FAIL"

    if verbose:
        print(f"    Torch cart_corr:  {corr_pt:.4f}")
        print(f"    DrJit cart_corr:  {corr_dr:.4f}")
        print(f"    Diff:             {cart_corr_diff:.4f}")
        print(f"    Torch-vs-DrJit:   {ra_corr:.6f}")
        print(f"    [{status}]")

    return {
        'scene': scene,
        'torch_corr': corr_pt,
        'drjit_corr': corr_dr,
        'diff': cart_corr_diff,
        'ra_corr': ra_corr,
        'pass': status == "PASS",
    }


# =========================================================================
# Main
# =========================================================================

def run_all_tests(scenes=None, skip_components=False):
    """Run all Stage A verification tests."""
    if scenes is None:
        scenes = SCENES

    print("=" * 80)
    print("Stage A Verification: PyTorch vs DrJit")
    print("=" * 80)

    if not skip_components:
        # A1: Reparameterization
        print("\n--- A1: Material Reparameterization ---")
        a1_results = {}
        for scene in scenes:
            a1_results[scene] = test_reparameterization(scene)

        # A2: BSDF
        print("\n--- A2: BSDF ---")
        a2_results = {}
        for scene in scenes:
            a2_results[scene] = test_bsdf(scene)

        # A3: Antenna
        print("\n--- A3: Antenna Gain ---")
        a3_results = {}
        for scene in scenes:
            a3_results[scene] = test_antenna(scene)

    # A5: Full forward pass
    print("\n--- A5: Full Forward Pass ---")
    a5_results = []
    for scene in scenes:
        r = test_full_forward(scene)
        a5_results.append(r)

    # Print summary table
    print(f"\n{'='*95}")
    print("Stage A5 Verification Summary")
    print(f"{'='*95}")
    print(f"{'Scene':<25} {'Torch':>8} {'DrJit':>8} {'Diff':>6} {'RA corr':>8} {'Status':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8}")

    all_pass = True
    for r in a5_results:
        status = "PASS" if r['pass'] else "FAIL"
        if not r['pass']:
            all_pass = False
        print(f"{r['scene']:<25} {r['torch_corr']:>8.4f} {r['drjit_corr']:>8.4f} "
              f"{r['diff']:>6.4f} {r['ra_corr']:>8.4f} {status:>8}")

    print(f"\n{'OVERALL: PASS' if all_pass else 'OVERALL: FAIL'}")
    return all_pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--skip-components', action='store_true',
                        help='Skip A1-A3, only run A5')
    parser.add_argument('--a1-only', action='store_true')
    parser.add_argument('--a2-only', action='store_true')
    parser.add_argument('--a3-only', action='store_true')
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else SCENES

    if args.a1_only:
        for s in scenes:
            test_reparameterization(s)
    elif args.a2_only:
        for s in scenes:
            test_bsdf(s)
    elif args.a3_only:
        for s in scenes:
            test_antenna(s)
    else:
        run_all_tests(scenes, skip_components=args.skip_components)
