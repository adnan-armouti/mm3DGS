"""Matched-grid rerun of Analyses 0, A–K for frame-NVS regulariser design.

Key difference from the prior-grid version:
  * All 4 variants now share the same position grid (seed_frame=test_frame
    in train_frame_nvs.py since 2026-04-18).
  * target_n=20000 (down from 90k) to bring the DOF / effective-rank ratio
    down by ~5×.
  * Cross-HO/UB comparisons are now exact per-point (no NN remap).

Outputs: md/frame_nvs_analysis_matched_grid/*.csv/*.png/*.npz
"""
import os, sys, json, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mmir.data.ra_utils import adc_to_ra_complex
from mmir.data.io_utils import compute_range_res_from_cfg

from mm25DGS_v5.rasterizer import (
    Rasterizer, reparameterize_torch, inverse_reparameterize_torch,
)
from mm25DGS_v5.train_gaussian import (
    DEVICE, ITU_CONCRETE, _normals_to_quaternions,
    init_visible_weighted, cull_gaussians, render_gaussians,
    range_profile_to_ra_mag, build_polar_to_cart_grid, polar_to_cart_torch,
    cart_corr_torch, USE_FACTORY_PATTERNS,
)
from mm25DGS_v5.load_pretrained import load_trained_config, load_pattern_data
from mm25DGS_v5.train_chirp_loop_nvs import build_per_loop_poses, apply_pose

# ---------------------------------------------------------------------------
# Config (matched grid = seed=test_frame, target_n=20000)
# ---------------------------------------------------------------------------
SCENES = ['seq_1_frame_438', 'seq_2_frame_105']
SCENE_FRAME = {'seq_1_frame_438': 438, 'seq_2_frame_105': 105}
TARGET_N = 20000
N_SUFFIX = f'_N{TARGET_N}' if TARGET_N != 90000 else ''
VARIANTS = {
    'HO_128': lambda F: f'train8frames_16loops_test{F}_loop0_pass2{N_SUFFIX}',
    'HO_8':   lambda F: f'train8frames_1loops_test{F}_loop0_pass2{N_SUFFIX}',
    'UB_144': lambda F: f'train9frames_16loops_test{F}_loop0_ub_pass2{N_SUFFIX}',
    'UB_9':   lambda F: f'train9frames_1loops_test{F}_loop0_ub_pass2{N_SUFFIX}',
}
VARIANT_ORDER = ['HO_128', 'HO_8', 'UB_144', 'UB_9']
PARAM_NAMES = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thickness']
PARAM_LOG_X = [False, True, True, True, False, True]

RUN_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v5', 'output_frame_nvs')
OUT_DIR  = os.path.join(PROJECT_ROOT, 'md', 'frame_nvs_analysis_matched_grid')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quat_to_normal(q_np):
    q = torch.from_numpy(q_np).float()
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1).numpy()


def load_state(scene, vname):
    F = SCENE_FRAME[scene]
    d = os.path.join(RUN_ROOT, f'{scene}_{VARIANTS[vname](F)}')
    st = torch.load(os.path.join(d, 'best_model.pt'),
                    weights_only=False, map_location='cpu')
    meta = json.load(open(os.path.join(d, 'results.json')))
    history = dict(np.load(os.path.join(d, 'history.npz')))
    return st, meta, history


def load_variant(scene, vname):
    st, meta, history = load_state(scene, vname)
    raw = st['raw_materials'].numpy()
    return {
        'positions': st['positions'].numpy(),
        'rotations': st['rotations'].numpy(),
        'raw':       raw,
        'physics':   reparameterize_torch(torch.from_numpy(raw)).numpy(),
        'normals':   quat_to_normal(st['rotations'].numpy()),
        'meta':      meta,
        'history':   history,
    }


def load_init_state(scene, positions):
    pcl = np.load(os.path.join(PROJECT_ROOT, 'data', scene, 'scene', 'pcl.npy'))
    xyz = pcl[:, :3].astype(np.float32)
    nrm = pcl[:, 3:6].astype(np.float32)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    intensity = pcl[:, 6].astype(np.float32) if pcl.shape[1] >= 7 else None
    tree = cKDTree(xyz)
    dists, idx = tree.query(positions, k=1)
    init_normals = nrm[idx]
    pcl_int = intensity[idx] if intensity is not None else np.zeros(len(positions))
    init_raw_row = inverse_reparameterize_torch(ITU_CONCRETE[None, :])[0]
    N = len(positions)
    init_raw = np.broadcast_to(init_raw_row[None, :], (N, 6)).copy()
    init_rot = _normals_to_quaternions(init_normals)
    return {
        'init_raw': init_raw,
        'init_normals': init_normals,
        'init_rotations': init_rot,
        'pcl_intensity': pcl_int,
        'pcl_nn_dist': dists.astype(np.float32),
    }


def build_rast_and_base_model(scene, seed_frame, target_n):
    config = load_trained_config(scene)
    seed_cfg = os.path.join(PROJECT_ROOT, 'data', 'alignment_data', scene,
                            'cascade', f'cascaded_frame_{seed_frame}_aligned_pass2.json')
    assert os.path.exists(seed_cfg), f'missing: {seed_cfg}'
    rast = Rasterizer(
        config_file=seed_cfg, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file, device=DEVICE)
    if not USE_FACTORY_PATTERNS:
        rast.inject_trained_params(pattern_data=load_pattern_data(scene))
    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    gc.collect(); torch.cuda.empty_cache()
    active_mask = cull_gaussians(model, rast)
    vertex_areas = torch.zeros(model.N, device=DEVICE)
    vertex_areas[active_mask] = 1.0
    range_res = compute_range_res_from_cfg(seed_cfg)
    sample_grid = build_polar_to_cart_grid(127, 256, range_res, 400, DEVICE)
    return rast, model, active_mask, vertex_areas, sample_grid, seed_cfg


def build_test_pose(scene):
    F = SCENE_FRAME[scene]
    align_dir = os.path.join(PROJECT_ROOT, 'data', 'alignment_data', scene, 'cascade')
    cfg_A = os.path.join(align_dir, f'cascaded_frame_{F-1}_aligned_pass2.json')
    cfg_B = os.path.join(align_dir, f'cascaded_frame_{F+1}_aligned_pass2.json')
    poses, _ = build_per_loop_poses(cfg_A, cfg_B, n_loops=16, device=DEVICE)
    return poses[0]


def load_test_gt_cart(scene, sample_grid):
    F = SCENE_FRAME[scene]
    adc_npy = os.path.join(PROJECT_ROOT, 'data', scene, 'radar',
                           f'cascaded_frame_{F}.npy')
    arr = np.load(adc_npy)
    ri = np.stack([arr[0].real, arr[0].imag], axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)
    gt_adc = torch.from_numpy(ri).to(DEVICE)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc)
        ra_mag = torch.abs(ra_c).float()
        gt_cart = polar_to_cart_torch(ra_mag, sample_grid)
        mn, mx = gt_cart.min(), gt_cart.max()
        gt_cart_norm = (gt_cart - mn) / (mx - mn).clamp(min=1e-30)
    return gt_cart_norm.detach()


def render_cart(model, rast, vertex_areas, active_mask, sample_grid,
                grad=False):
    if grad:
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None)
    else:
        with torch.no_grad():
            rp_real, rp_imag = render_gaussians(
                model, rast, vertex_areas=vertex_areas,
                active_mask=active_mask, shadow_mask=None,
                bsdf_mode='full', disabled_components=None)
    if grad:
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
    else:
        with torch.no_grad():
            ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
            ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
    return ra_cart


# ---------------------------------------------------------------------------
# Load 8 variants
# ---------------------------------------------------------------------------
print('=' * 72); print('Matched-grid analyses — loading variants'); print('=' * 72)
DATA = {}
for scene in SCENES:
    DATA[scene] = {}
    for vname in VARIANT_ORDER:
        DATA[scene][vname] = load_variant(scene, vname)
    # Matched grid: HO and UB positions should now be identical.
    assert np.allclose(DATA[scene]['HO_128']['positions'],
                       DATA[scene]['UB_144']['positions']), (
        f'{scene}: HO and UB positions differ — is seed_frame=test_frame active?')
    DATA[scene]['init'] = load_init_state(scene, DATA[scene]['HO_128']['positions'])
    print(f'  {scene}: N={len(DATA[scene]["HO_128"]["positions"])}  '
          f'HO==UB positions ✓  pcl_nn_dist_max={DATA[scene]["init"]["pcl_nn_dist"].max():.4e}')


# ---------------------------------------------------------------------------
# Analysis 0: reproducibility (render → cc vs reported)
# ---------------------------------------------------------------------------
print('\n=== Analysis 0: reproducibility ===')
rows_0 = []
renders_by_sv = {}
gt_cart_by_scene = {}
fisher_by_sv = {}

for scene in SCENES:
    F = SCENE_FRAME[scene]
    test_pose = build_test_pose(scene)

    # Build single rasterizer (both HO and UB are on the same grid now)
    seed = F
    rast, model, _, _, sgrid, _ = build_rast_and_base_model(scene, seed, TARGET_N)
    apply_pose(rast, test_pose)
    amask = cull_gaussians(model, rast)
    varea = torch.zeros(model.N, device=DEVICE); varea[amask] = 1.0
    gt_cart = load_test_gt_cart(scene, sgrid)
    gt_cart_by_scene[scene] = gt_cart.cpu().numpy()

    for vname in VARIANT_ORDER:
        st, meta, _ = load_state(scene, vname)
        with torch.no_grad():
            model.positions.copy_(st['positions'].to(DEVICE))
            model.rotations.copy_(st['rotations'].to(DEVICE))
            model.raw_materials.copy_(st['raw_materials'].to(DEVICE))
        rend = render_cart(model, rast, varea, amask, sgrid)
        cc = cart_corr_torch(rend, gt_cart).item()
        renders_by_sv[(scene, vname)] = rend.cpu().numpy()
        rows_0.append({
            'scene': scene, 'variant': vname,
            'reported': float(meta['final_test_cc']),
            'measured': float(cc),
            'delta':    float(abs(cc - meta['final_test_cc'])),
            'final_train_cc': float(meta['final_train_mean_cc']),
            'seed_frame': int(meta.get('seed_frame', -1)),
        })
        print(f'  {scene:<20} {vname:<8} reported={meta["final_test_cc"]:.4f}  '
              f'measured={cc:.4f}  Δ={abs(cc - meta["final_test_cc"]):.4f}  '
              f'train_cc={meta["final_train_mean_cc"]:.4f}  seed={meta.get("seed_frame","?")}')

        # Analysis G: per-point Fisher
        model.raw_materials.requires_grad_(True)
        model.rotations.requires_grad_(True)
        with torch.no_grad():
            model.raw_materials.copy_(st['raw_materials'].to(DEVICE))
            model.rotations.copy_(st['rotations'].to(DEVICE))
        model.raw_materials.grad = None
        model.rotations.grad = None
        rend = render_cart(model, rast, varea, amask, sgrid, grad=True)
        cc_diff = cart_corr_torch(rend, gt_cart)
        cc_diff.backward()
        fisher_by_sv[(scene, vname)] = {
            'grad_raw': model.raw_materials.grad.detach().cpu().numpy(),
            'grad_rot': model.rotations.grad.detach().cpu().numpy(),
            'cc': float(cc_diff.item()),
        }
        model.raw_materials.requires_grad_(False)
        model.rotations.requires_grad_(False)

    # save renders + gt
    del rast, model; gc.collect(); torch.cuda.empty_cache()

import csv
with open(os.path.join(OUT_DIR, '0_reproducibility.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_0[0].keys()); w.writeheader(); w.writerows(rows_0)
for (scene, vname), arr in renders_by_sv.items():
    np.save(os.path.join(OUT_DIR, f'render_{scene}_{vname}.npy'), arr)
for scene, arr in gt_cart_by_scene.items():
    np.save(os.path.join(OUT_DIR, f'gt_cart_{scene}.npy'), arr)
for (scene, vname), v in fisher_by_sv.items():
    np.savez(os.path.join(OUT_DIR, f'G_fisher_{scene}_{vname}.npz'),
             grad_raw=v['grad_raw'], grad_rot=v['grad_rot'], cc=v['cc'])


# ---------------------------------------------------------------------------
# Analysis A: per-variant parameter distributions
# ---------------------------------------------------------------------------
print('\n=== Analysis A: parameter distributions ===')
rows_A = []
for scene in SCENES:
    for vname in VARIANT_ORDER:
        d = DATA[scene][vname]
        for c, pname in enumerate(PARAM_NAMES):
            arr = d['physics'][:, c]
            rows_A.append({
                'scene': scene, 'variant': vname, 'param': pname,
                'min': float(arr.min()), 'max': float(arr.max()),
                'mean': float(arr.mean()), 'median': float(np.median(arr)),
                'std': float(arr.std()),
                'p25': float(np.percentile(arr, 25)),
                'p75': float(np.percentile(arr, 75)),
                'iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            })
with open(os.path.join(OUT_DIR, 'A_param_distributions.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_A[0].keys()); w.writeheader(); w.writerows(rows_A)

for scene in SCENES:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.flatten()
    for c, (pname, log_x) in enumerate(zip(PARAM_NAMES, PARAM_LOG_X)):
        ax = axes[c]
        for vname in VARIANT_ORDER:
            arr = DATA[scene][vname]['physics'][:, c]
            if log_x:
                arr = np.log10(np.maximum(arr, 1e-12))
                xlabel = f'log10({pname})'
            else:
                xlabel = pname
            ax.hist(arr, bins=80, alpha=0.45, label=vname, density=True)
        ax.set_title(f'{scene}: {pname}'); ax.set_xlabel(xlabel); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f'A_hist_{scene}.png'), dpi=110); plt.close()

print('  Raw-materials std per (scene, variant, param):')
print(f'  {"scene":<20} {"variant":<8} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        raws = [f'{DATA[scene][vname]["raw"][:, c].std():<10.3f}' for c in range(6)]
        print(f'  {scene:<20} {vname:<8} | ' + ' | '.join(raws))


# ---------------------------------------------------------------------------
# Analysis B: per-point drift from init
# ---------------------------------------------------------------------------
print('\n=== Analysis B: drift from init ===')
rows_B = []
for scene in SCENES:
    init = DATA[scene]['init']
    for vname in VARIANT_ORDER:
        d = DATA[scene][vname]
        drift_raw = d['raw'] - init['init_raw']
        drift_L2 = np.linalg.norm(drift_raw, axis=1)
        cos_ang = np.clip((d['normals'] * init['init_normals']).sum(axis=-1), -1, 1)
        drift_deg = np.degrees(np.arccos(cos_ang))
        rows_B.append({
            'scene': scene, 'variant': vname,
            'L2_mean': float(drift_L2.mean()),
            'L2_med': float(np.median(drift_L2)),
            'L2_p95': float(np.percentile(drift_L2, 95)),
            'deg_mean': float(drift_deg.mean()),
            'deg_med': float(np.median(drift_deg)),
            'deg_p95': float(np.percentile(drift_deg, 95)),
            'frac_L2_gt_1':   float((drift_L2 > 1.0).mean()),
            'frac_deg_gt_5':  float((drift_deg > 5.0).mean()),
            'frac_deg_gt_15': float((drift_deg > 15.0).mean()),
        })
        DATA[scene][vname]['_drift_L2'] = drift_L2
        DATA[scene][vname]['_drift_deg'] = drift_deg
with open(os.path.join(OUT_DIR, 'B_drift_summary.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_B[0].keys()); w.writeheader(); w.writerows(rows_B)
for r in rows_B:
    print(f'  {r["scene"]:<20} {r["variant"]:<8} | '
          f'L2 mean={r["L2_mean"]:6.3f} med={r["L2_med"]:6.3f} p95={r["L2_p95"]:6.3f} | '
          f'deg mean={r["deg_mean"]:6.2f} med={r["deg_med"]:6.2f} p95={r["deg_p95"]:6.2f}')
for scene in SCENES:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for vname in VARIANT_ORDER:
        axes[0].hist(DATA[scene][vname]['_drift_L2'], bins=80,
                     alpha=0.45, label=vname, density=True)
        axes[1].hist(DATA[scene][vname]['_drift_deg'], bins=80,
                     alpha=0.45, label=vname, density=True, range=(0, 60))
    axes[0].set_xlabel('||raw_trained - raw_init||_2'); axes[0].legend(fontsize=7)
    axes[0].set_title(f'{scene}: material drift L2')
    axes[1].set_xlabel('normal drift (deg)'); axes[1].legend(fontsize=7)
    axes[1].set_title(f'{scene}: normal drift')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f'B_drift_{scene}.png'), dpi=110); plt.close()


# ---------------------------------------------------------------------------
# Analysis C: UB vs HO per-point divergence (now EXACT — same positions)
# ---------------------------------------------------------------------------
print('\n=== Analysis C: UB vs HO divergence (matched grid, no NN remap) ===')
rows_C = []
for scene in SCENES:
    d_HO128 = DATA[scene]['HO_128']['raw']
    d_HO8   = DATA[scene]['HO_8']['raw']
    d_UB144 = DATA[scene]['UB_144']['raw']
    d_UB9   = DATA[scene]['UB_9']['raw']
    for pair_name, d1, d2 in [
        ('HO_128-HO_8',    d_HO128, d_HO8),
        ('UB_144-UB_9',    d_UB144, d_UB9),
        ('UB_144-HO_128',  d_UB144, d_HO128),   # no NN remap needed
        ('UB_9-HO_8',      d_UB9,   d_HO8),
    ]:
        for c, pname in enumerate(PARAM_NAMES):
            delta = d1[:, c] - d2[:, c]
            rows_C.append({
                'scene': scene, 'pair': pair_name, 'param': pname,
                'mean': float(delta.mean()), 'std': float(delta.std()),
                'abs_mean': float(np.abs(delta).mean()),
                'p95_abs': float(np.percentile(np.abs(delta), 95)),
            })
with open(os.path.join(OUT_DIR, 'C_divergence.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_C[0].keys()); w.writeheader(); w.writerows(rows_C)
print(f'  {"scene":<20} {"pair":<22} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for pair in ['HO_128-HO_8', 'UB_144-UB_9', 'UB_144-HO_128', 'UB_9-HO_8']:
        vals = [r['abs_mean'] for r in rows_C
                if r['scene'] == scene and r['pair'] == pair]
        print(f'  {scene:<20} {pair:<22} | ' + ' | '.join(f'{v:<10.3f}' for v in vals))


# ---------------------------------------------------------------------------
# Analysis D: k-NN material smoothness (single shared kd-tree now)
# ---------------------------------------------------------------------------
print('\n=== Analysis D: spatial smoothness (K=10 NN) ===')
K = 10
rows_D = []
for scene in SCENES:
    pos = DATA[scene]['HO_128']['positions']
    tree = cKDTree(pos)
    _, nn_idx = tree.query(pos, k=K + 1)
    nn_idx = nn_idx[:, 1:]  # drop self
    for vname in VARIANT_ORDER:
        raw = DATA[scene][vname]['raw']
        for c, pname in enumerate(PARAM_NAMES):
            nbr_std = raw[:, c][nn_idx].std(axis=1)
            rows_D.append({
                'scene': scene, 'variant': vname, 'param': pname,
                'nbr_std_mean':   float(nbr_std.mean()),
                'nbr_std_median': float(np.median(nbr_std)),
                'nbr_std_p95':    float(np.percentile(nbr_std, 95)),
            })
with open(os.path.join(OUT_DIR, 'D_spatial_smoothness.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_D[0].keys()); w.writeheader(); w.writerows(rows_D)
print('  mean neighbour-std of raw[c] (smaller ⇒ smoother material field):')
print(f'  {"scene":<20} {"variant":<8} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        vals = [r['nbr_std_mean'] for r in rows_D
                if r['scene'] == scene and r['variant'] == vname]
        print(f'  {scene:<20} {vname:<8} | ' + ' | '.join(f'{v:<10.3f}' for v in vals))


# ---------------------------------------------------------------------------
# Analysis E: normal-field smoothness
# ---------------------------------------------------------------------------
print('\n=== Analysis E: normal-field smoothness ===')
rows_E = []
for scene in SCENES:
    pos = DATA[scene]['HO_128']['positions']
    tree = cKDTree(pos)
    _, nn_idx = tree.query(pos, k=K + 1)
    nn_idx = nn_idx[:, 1:]
    # baseline: init normals
    init_n = DATA[scene]['init']['init_normals']
    nbr = init_n[nn_idx]
    dot = (init_n[:, None, :] * nbr).sum(-1).clip(-1, 1)
    ang = np.degrees(np.arccos(dot)).mean(axis=1)
    rows_E.append({
        'scene': scene, 'variant': 'init',
        'mean': float(ang.mean()), 'median': float(np.median(ang)),
        'p95': float(np.percentile(ang, 95)),
    })
    for vname in VARIANT_ORDER:
        n = DATA[scene][vname]['normals']
        nbr = n[nn_idx]
        dot = (n[:, None, :] * nbr).sum(-1).clip(-1, 1)
        ang = np.degrees(np.arccos(dot)).mean(axis=1)
        rows_E.append({
            'scene': scene, 'variant': vname,
            'mean': float(ang.mean()), 'median': float(np.median(ang)),
            'p95': float(np.percentile(ang, 95)),
        })
with open(os.path.join(OUT_DIR, 'E_normal_smoothness.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_E[0].keys()); w.writeheader(); w.writerows(rows_E)
for r in rows_E:
    print(f'  {r["scene"]:<20} {r["variant"]:<6} mean={r["mean"]:6.2f}° '
          f'med={r["median"]:6.2f}° p95={r["p95"]:6.2f}°')


# ---------------------------------------------------------------------------
# Analysis F: training trajectories
# ---------------------------------------------------------------------------
print('\n=== Analysis F: training trajectories ===')
for scene in SCENES:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for vname in VARIANT_ORDER:
        h = DATA[scene][vname]['history']
        axes[0].plot(h['iters'], h['mean_train_cc'], label=vname)
        axes[1].plot(h['iters'], h['loss'], label=vname)
    axes[0].set_xlabel('iter'); axes[0].set_ylabel('mean train cc')
    axes[0].set_title(f'{scene}: train cc'); axes[0].legend()
    axes[1].set_xlabel('iter'); axes[1].set_ylabel('loss'); axes[1].set_yscale('log')
    axes[1].set_title(f'{scene}: loss'); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f'F_trajectories_{scene}.png'), dpi=110); plt.close()


# ---------------------------------------------------------------------------
# Analysis G summary (uses fisher_by_sv computed above)
# ---------------------------------------------------------------------------
print('\n=== Analysis G: Fisher concentration (matched-grid) ===')
print(f'  {"scene":<20} {"variant":<8} | ' +
      ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        g = fisher_by_sv[(scene, vname)]['grad_raw']
        print(f'  {scene:<20} {vname:<8} | ' +
              ' | '.join(f'{np.abs(g[:, c]).mean():<10.3e}' for c in range(6)))
print()
print('  Per-point Fisher concentration (top-k% of total per-pt Fisher):')
rows_Gconc = []
for scene in SCENES:
    for vname in VARIANT_ORDER:
        g = fisher_by_sv[(scene, vname)]['grad_raw']
        per_pt = (g ** 2).sum(axis=1)
        total = per_pt.sum()
        srt = np.sort(per_pt)[::-1]
        row = {'scene': scene, 'variant': vname, 'N': int(len(per_pt)),
               'cc': float(fisher_by_sv[(scene, vname)]['cc'])}
        for frac in [0.01, 0.05, 0.10, 0.25, 0.50]:
            k = int(frac * len(per_pt))
            row[f'top_{int(frac*100)}pct'] = float(100 * srt[:k].sum() / max(total, 1e-30))
        rows_Gconc.append(row)
        print(f'  {scene:<20} {vname:<8} N={len(per_pt):>6}  '
              f'top1%={row["top_1pct"]:5.1f}  top5%={row["top_5pct"]:5.1f}  '
              f'top10%={row["top_10pct"]:5.1f}  top25%={row["top_25pct"]:5.1f}')
with open(os.path.join(OUT_DIR, 'G_concentration.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_Gconc[0].keys()); w.writeheader(); w.writerows(rows_Gconc)


# ---------------------------------------------------------------------------
# Analysis I: swap ablation (now EXACT — same grid, no NN remap)
# ---------------------------------------------------------------------------
print('\n=== Analysis I: material/normal swap (matched grid) ===')
rows_I = []
for scene in SCENES:
    F = SCENE_FRAME[scene]
    test_pose = build_test_pose(scene)
    rast, model, _, _, sgrid, _ = build_rast_and_base_model(scene, F, TARGET_N)
    apply_pose(rast, test_pose)
    amask = cull_gaussians(model, rast)
    varea = torch.zeros(model.N, device=DEVICE); varea[amask] = 1.0
    gt_cart = load_test_gt_cart(scene, sgrid)

    st_HO128, _, _ = load_state(scene, 'HO_128')
    st_UB144, _, _ = load_state(scene, 'UB_144')
    st_HO8,   _, _ = load_state(scene, 'HO_8')
    st_UB9,   _, _ = load_state(scene, 'UB_9')

    def _render(st_pos, st_rot, st_raw):
        with torch.no_grad():
            model.positions.copy_(st_pos.to(DEVICE))
            model.rotations.copy_(st_rot.to(DEVICE))
            model.raw_materials.copy_(st_raw.to(DEVICE))
        rend = render_cart(model, rast, varea, amask, sgrid)
        return cart_corr_torch(rend, gt_cart).item()

    # Baselines (no swap)
    cc_HO_full = _render(st_HO128['positions'], st_HO128['rotations'], st_HO128['raw_materials'])
    cc_UB_full = _render(st_UB144['positions'], st_UB144['rotations'], st_UB144['raw_materials'])
    # Swap materials and normals (positions identical on matched grid)
    cc_UBmat_HOrot = _render(st_HO128['positions'], st_HO128['rotations'], st_UB144['raw_materials'])
    cc_HOmat_UBrot = _render(st_HO128['positions'], st_UB144['rotations'], st_HO128['raw_materials'])
    rows_I.extend([
        (scene, 'HO_128 native',          cc_HO_full),
        (scene, 'UB_144 native',          cc_UB_full),
        (scene, 'UB_mat + HO_rot',        cc_UBmat_HOrot),
        (scene, 'HO_mat + UB_rot',        cc_HOmat_UBrot),
    ])
    # first-chirp pair too
    cc_HO8 = _render(st_HO8['positions'], st_HO8['rotations'], st_HO8['raw_materials'])
    cc_UB9 = _render(st_UB9['positions'], st_UB9['rotations'], st_UB9['raw_materials'])
    cc_UB9mat_HO8rot = _render(st_HO8['positions'], st_HO8['rotations'], st_UB9['raw_materials'])
    cc_HO8mat_UB9rot = _render(st_HO8['positions'], st_UB9['rotations'], st_HO8['raw_materials'])
    rows_I.extend([
        (scene, 'HO_8 native',            cc_HO8),
        (scene, 'UB_9 native',            cc_UB9),
        (scene, 'UB9_mat + HO8_rot',      cc_UB9mat_HO8rot),
        (scene, 'HO8_mat + UB9_rot',      cc_HO8mat_UB9rot),
    ])
    for row in rows_I[-8:]:
        print(f'  {row[0]:<20} {row[1]:<24} cc = {row[2]:.4f}')
    del rast, model; gc.collect(); torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, 'I_swap_ablation.csv'), 'w') as f:
    w = csv.writer(f); w.writerow(['scene', 'combo', 'cc']); w.writerows(rows_I)


# ---------------------------------------------------------------------------
# Analysis J: intensity prior
# ---------------------------------------------------------------------------
print('\n=== Analysis J: intensity prior (matched grid) ===')
rows_J = []
N_BINS = 10
for scene in SCENES:
    intensity = DATA[scene]['init']['pcl_intensity']
    order = np.argsort(intensity)
    bin_edges = np.linspace(0, len(intensity), N_BINS + 1, dtype=int)
    for vname in VARIANT_ORDER:
        d = DATA[scene][vname]
        for c, pname in enumerate(PARAM_NAMES):
            col = d['raw'][:, c]
            intra = [col[order[bin_edges[b]:bin_edges[b + 1]]].var() for b in range(N_BINS)]
            rows_J.append({
                'scene': scene, 'variant': vname, 'param': pname,
                'mean_intra_var': float(np.mean(intra)),
                'total_var': float(col.var()),
                'ratio': float(np.mean(intra) / max(col.var(), 1e-12)),
            })
with open(os.path.join(OUT_DIR, 'J_intensity_prior.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_J[0].keys()); w.writeheader(); w.writerows(rows_J)
print(f'  {"scene":<20} {"variant":<8} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        vals = [r['ratio'] for r in rows_J
                if r['scene'] == scene and r['variant'] == vname]
        print(f'  {scene:<20} {vname:<8} | ' + ' | '.join(f'{v:<10.3f}' for v in vals))


# ---------------------------------------------------------------------------
# Analysis K: UB_144 vs UB_9 noise-averaging
# ---------------------------------------------------------------------------
print('\n=== Analysis K: UB_144 vs UB_9 ===')
rows_K = []
for scene in SCENES:
    raw_144 = DATA[scene]['UB_144']['raw']
    raw_9   = DATA[scene]['UB_9']['raw']
    for c, pname in enumerate(PARAM_NAMES):
        sh = raw_9[:, c].std() - raw_144[:, c].std()
        iqr9   = np.percentile(raw_9[:, c], 75)   - np.percentile(raw_9[:, c], 25)
        iqr144 = np.percentile(raw_144[:, c], 75) - np.percentile(raw_144[:, c], 25)
        rows_K.append({
            'scene': scene, 'param': pname,
            'std_UB_144': float(raw_144[:, c].std()),
            'std_UB_9':   float(raw_9[:, c].std()),
            'std_delta_9m144': float(sh),
            'iqr_delta_9m144': float(iqr9 - iqr144),
        })
with open(os.path.join(OUT_DIR, 'K_noise_averaging.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_K[0].keys()); w.writeheader(); w.writerows(rows_K)
for r in rows_K:
    print(f'  {r["scene"]:<20} {r["param"]:<12} std(UB_144)={r["std_UB_144"]:.3f}  '
          f'std(UB_9)={r["std_UB_9"]:.3f}  Δstd(9-144)={r["std_delta_9m144"]:+.3f}')


print('\n\nDone. All artefacts under', OUT_DIR)
