"""Analyses 0 (reproducibility), G (Fisher), I (swap ablation).

These need rendering, so we set up Mitsuba ONCE and loop through.
Single process — mi.set_variant called once at import.

Run:
    /home/adnan/.conda/envs/mmir/bin/python md/frame_nvs_analysis/run_analyses_0GI.py
"""
import os, sys, json, gc
import numpy as np
import torch

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from scipy.spatial import cKDTree

from mmir.data.ra_utils import adc_to_ra_complex
from mmir.data.io_utils import compute_range_res_from_cfg

from mm25DGS_v5.rasterizer import Rasterizer
from mm25DGS_v5.train_gaussian import (
    DEVICE, init_visible_weighted, cull_gaussians, render_gaussians,
    range_profile_to_ra_mag, build_polar_to_cart_grid, polar_to_cart_torch,
    cart_corr_torch, USE_FACTORY_PATTERNS,
)
from mm25DGS_v5.load_pretrained import load_trained_config, load_pattern_data
from mm25DGS_v5.train_chirp_loop_nvs import build_per_loop_poses, apply_pose

SCENES = ['seq_1_frame_438', 'seq_2_frame_105']
SCENE_FRAME = {'seq_1_frame_438': 438, 'seq_2_frame_105': 105}
VARIANTS = {
    'HO_128': lambda F: f'train8frames_test{F}_loop0_pass2',
    'HO_8':   lambda F: f'train8frames_1loops_test{F}_loop0_pass2',
    'UB_144': lambda F: f'train9frames_16loops_test{F}_loop0_ub_pass2',
    'UB_9':   lambda F: f'train9frames_1loops_test{F}_loop0_ub_pass2',
}
VARIANT_ORDER = ['HO_128', 'HO_8', 'UB_144', 'UB_9']

RUN_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v5', 'output_frame_nvs')
OUT_DIR  = os.path.join(PROJECT_ROOT, 'md', 'frame_nvs_analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def load_state(scene, vname):
    F = SCENE_FRAME[scene]
    d = os.path.join(RUN_ROOT, f'{scene}_{VARIANTS[vname](F)}')
    st = torch.load(os.path.join(d, 'best_model.pt'),
                    weights_only=False, map_location='cpu')
    meta = json.load(open(os.path.join(d, 'results.json')))
    return st, meta


def build_rast_and_base_model(scene, seed_frame):
    """Build rasterizer + init model at the given seed_frame (matches training)."""
    config = load_trained_config(scene)
    suffix = '_aligned_pass2'
    seed_cfg = os.path.join(PROJECT_ROOT, 'data', 'alignment_data', scene,
                            'cascade', f'cascaded_frame_{seed_frame}{suffix}.json')
    assert os.path.exists(seed_cfg), f'missing: {seed_cfg}'

    rast = Rasterizer(
        config_file=seed_cfg, mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file, device=DEVICE)
    if not USE_FACTORY_PATTERNS:
        rast.inject_trained_params(pattern_data=load_pattern_data(scene))
    model = init_visible_weighted(scene, rast, target_n=90000)
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


def render_cart(model, rast, vertex_areas, active_mask, sample_grid):
    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None,
            bsdf_mode='full', disabled_components=None)
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
    return ra_cart


def render_cart_grad(model, rast, vertex_areas, active_mask, sample_grid):
    rp_real, rp_imag = render_gaussians(
        model, rast, vertex_areas=vertex_areas,
        active_mask=active_mask, shadow_mask=None,
        bsdf_mode='full', disabled_components=None)
    ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
    ra_cart = polar_to_cart_torch(ra_polar, sample_grid)
    return ra_cart


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

results_ALL = {
    'analysis_0': [],     # [(scene, variant, reported, measured, delta)]
    'analysis_I': [],     # [(scene, combo_name, cc)]
}
renders_by_scene_variant = {}  # (scene, variant) -> rend_cart.cpu().numpy()
gt_cart_by_scene = {}          # scene -> gt_cart.cpu().numpy()
fisher_by_scene_variant = {}   # (scene, variant) -> {grad_raw (N,6), grad_rot (N,4)}

for scene in SCENES:
    F = SCENE_FRAME[scene]
    # HO seed = F+1 (train_frames middle = 4th of 8 = 439 for seq_1)
    # UB seed = F   (train_frames middle = 4th of 9 = 438 for seq_1)
    # Reconstruct via the training convention used in train_frame_nvs.
    # Training computes: seed_frame = train_frames[len(train_frames)//2]
    #   HO: train = [F-4..F-1, F+1..F+4] (8 frames) -> middle index 4 -> F+1
    #   UB: train = [F-4..F-1, F, F+1..F+4] (9 frames) -> middle index 4 -> F
    HO_SEED = F + 1
    UB_SEED = F

    print(f'\n\n============== {scene} ==============')

    # Test pose + GT
    test_pose = build_test_pose(scene)

    # ---- HO path: build rast+base model at HO_SEED ----
    print(f'  [HO seed={HO_SEED}] building rasterizer + model...')
    rast_HO, model_HO, amask_HO, varea_HO, sgrid_HO, seed_cfg_HO = \
        build_rast_and_base_model(scene, HO_SEED)
    gt_cart = load_test_gt_cart(scene, sgrid_HO)
    gt_cart_by_scene[scene] = gt_cart.cpu().numpy()
    apply_pose(rast_HO, test_pose)
    amask_HO = cull_gaussians(model_HO, rast_HO)
    varea_HO = torch.zeros(model_HO.N, device=DEVICE)
    varea_HO[amask_HO] = 1.0

    HO_states = {}
    for vname in ['HO_128', 'HO_8']:
        st, meta = load_state(scene, vname)
        HO_states[vname] = {k: v.clone() for k, v in st.items()}
        # Render
        with torch.no_grad():
            model_HO.positions.copy_(st['positions'].to(DEVICE))
            model_HO.rotations.copy_(st['rotations'].to(DEVICE))
            model_HO.raw_materials.copy_(st['raw_materials'].to(DEVICE))
        rend_cart = render_cart(model_HO, rast_HO, varea_HO, amask_HO, sgrid_HO)
        cc = cart_corr_torch(rend_cart, gt_cart).item()
        renders_by_scene_variant[(scene, vname)] = rend_cart.cpu().numpy()
        delta = abs(cc - meta['final_test_cc'])
        results_ALL['analysis_0'].append((scene, vname, float(meta['final_test_cc']),
                                          float(cc), float(delta)))
        print(f'    [A0] {vname}: reported={meta["final_test_cc"]:.4f}  '
              f'measured={cc:.4f}  Δ={delta:.4f}')

        # Analysis G: Fisher (per-point gradient of cc wrt raw_materials + rotations)
        model_HO.raw_materials.requires_grad_(True)
        model_HO.rotations.requires_grad_(True)
        # Re-set from state dict to ensure leaf tensors
        with torch.no_grad():
            model_HO.raw_materials.copy_(st['raw_materials'].to(DEVICE))
            model_HO.rotations.copy_(st['rotations'].to(DEVICE))
        model_HO.raw_materials.grad = None
        model_HO.rotations.grad = None
        rend = render_cart_grad(model_HO, rast_HO, varea_HO, amask_HO, sgrid_HO)
        cc_diff = cart_corr_torch(rend, gt_cart)
        cc_diff.backward()
        grad_raw = model_HO.raw_materials.grad.detach().cpu().numpy()
        grad_rot = model_HO.rotations.grad.detach().cpu().numpy()
        fisher_by_scene_variant[(scene, vname)] = {
            'grad_raw': grad_raw, 'grad_rot': grad_rot, 'cc': float(cc_diff.item()),
        }
        print(f'    [G]  {vname}: |grad_raw| mean/col: '
              + ' '.join(f'{np.abs(grad_raw[:, c]).mean():.3e}' for c in range(6))
              + f'  (cc from autograd={cc_diff.item():.4f})')
        # Reset requires_grad
        model_HO.raw_materials.requires_grad_(False)
        model_HO.rotations.requires_grad_(False)

    # ---- UB path: build rast+base model at UB_SEED ----
    print(f'  [UB seed={UB_SEED}] building rasterizer + model...')
    rast_UB, model_UB, amask_UB, varea_UB, sgrid_UB, seed_cfg_UB = \
        build_rast_and_base_model(scene, UB_SEED)
    apply_pose(rast_UB, test_pose)
    amask_UB = cull_gaussians(model_UB, rast_UB)
    varea_UB = torch.zeros(model_UB.N, device=DEVICE)
    varea_UB[amask_UB] = 1.0
    # GT is identical geometrically but sgrid_UB may slightly differ if range_res
    # depends on scene config — reload with the UB grid
    gt_cart_UB = load_test_gt_cart(scene, sgrid_UB)

    UB_states = {}
    for vname in ['UB_144', 'UB_9']:
        st, meta = load_state(scene, vname)
        UB_states[vname] = {k: v.clone() for k, v in st.items()}
        with torch.no_grad():
            model_UB.positions.copy_(st['positions'].to(DEVICE))
            model_UB.rotations.copy_(st['rotations'].to(DEVICE))
            model_UB.raw_materials.copy_(st['raw_materials'].to(DEVICE))
        rend_cart = render_cart(model_UB, rast_UB, varea_UB, amask_UB, sgrid_UB)
        cc = cart_corr_torch(rend_cart, gt_cart_UB).item()
        renders_by_scene_variant[(scene, vname)] = rend_cart.cpu().numpy()
        delta = abs(cc - meta['final_test_cc'])
        results_ALL['analysis_0'].append((scene, vname, float(meta['final_test_cc']),
                                          float(cc), float(delta)))
        print(f'    [A0] {vname}: reported={meta["final_test_cc"]:.4f}  '
              f'measured={cc:.4f}  Δ={delta:.4f}')

        # Analysis G Fisher for UB variants
        model_UB.raw_materials.requires_grad_(True)
        model_UB.rotations.requires_grad_(True)
        with torch.no_grad():
            model_UB.raw_materials.copy_(st['raw_materials'].to(DEVICE))
            model_UB.rotations.copy_(st['rotations'].to(DEVICE))
        model_UB.raw_materials.grad = None
        model_UB.rotations.grad = None
        rend = render_cart_grad(model_UB, rast_UB, varea_UB, amask_UB, sgrid_UB)
        cc_diff = cart_corr_torch(rend, gt_cart_UB)
        cc_diff.backward()
        grad_raw = model_UB.raw_materials.grad.detach().cpu().numpy()
        grad_rot = model_UB.rotations.grad.detach().cpu().numpy()
        fisher_by_scene_variant[(scene, vname)] = {
            'grad_raw': grad_raw, 'grad_rot': grad_rot, 'cc': float(cc_diff.item()),
        }
        print(f'    [G]  {vname}: |grad_raw| mean/col: '
              + ' '.join(f'{np.abs(grad_raw[:, c]).mean():.3e}' for c in range(6))
              + f'  (cc from autograd={cc_diff.item():.4f})')
        model_UB.raw_materials.requires_grad_(False)
        model_UB.rotations.requires_grad_(False)

    # ---- Analysis I: swap ablation ----
    # Positions differ between HO and UB. We do swaps at HO position set
    # (render via rast_HO) using NN lookup from HO positions -> UB indices.
    print(f'\n  [Analysis I] material/normal swap (rendered at HO seed)')
    pos_HO = HO_states['HO_128']['positions'].numpy()
    pos_UB = UB_states['UB_144']['positions'].numpy()
    tree_UB = cKDTree(pos_UB)
    _, nn_idx = tree_UB.query(pos_HO, k=1)
    nn_idx_t = torch.from_numpy(nn_idx).long().to(DEVICE)

    # Pairs to render at HO grid:
    pairs_full  = ('UB_144', 'HO_128')  # full-chirp
    pairs_first = ('UB_9',   'HO_8')    # first-chirp
    def swap_and_render(mat_src, rot_src, label):
        # Source-grid dispatch: UB variants live on UB position grid; HO on HO.
        def fetch(key, attr):
            st = UB_states[key] if key.startswith('UB') else HO_states[key]
            v = st[attr].to(DEVICE)
            return v, key.startswith('UB')
        mat, mat_from_UB = fetch(mat_src, 'raw_materials')
        rot, rot_from_UB = fetch(rot_src, 'rotations')
        # Always render on the HO grid: remap anything from UB grid via NN.
        if mat_from_UB:
            mat = mat[nn_idx_t]
        if rot_from_UB:
            rot = rot[nn_idx_t]
        pos_on_HO = torch.from_numpy(pos_HO).to(DEVICE)
        with torch.no_grad():
            model_HO.positions.copy_(pos_on_HO)
            model_HO.rotations.copy_(rot)
            model_HO.raw_materials.copy_(mat)
        r = render_cart(model_HO, rast_HO, varea_HO, amask_HO, sgrid_HO)
        cc = cart_corr_torch(r, gt_cart).item()
        print(f'    [I]  {label:<34}  cc={cc:.4f}')
        results_ALL['analysis_I'].append((scene, label, float(cc)))
        return cc

    for mat_v, rot_v in [(pairs_full[0], pairs_full[1]),
                         (pairs_first[0], pairs_first[1])]:
        # baseline pure HO re-render
        swap_and_render(rot_v, rot_v, f'{rot_v} mat + {rot_v} rot (HO baseline)')
        # pure UB re-rendered on HO grid (NN-resampled)
        swap_and_render(mat_v, mat_v, f'{mat_v} mat + {mat_v} rot (UB on HO grid)')
        # UB materials + HO normals
        swap_and_render(mat_v, rot_v, f'{mat_v} mat + {rot_v} rot')
        # HO materials + UB normals
        swap_and_render(rot_v, mat_v, f'{rot_v} mat + {mat_v} rot')

    # cleanup for next scene
    del rast_HO, rast_UB, model_HO, model_UB
    gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
import csv

with open(os.path.join(OUT_DIR, '0_reproducibility.csv'), 'w') as f:
    w = csv.writer(f)
    w.writerow(['scene', 'variant', 'reported_cc', 'measured_cc', 'delta'])
    for row in results_ALL['analysis_0']:
        w.writerow(row)

with open(os.path.join(OUT_DIR, 'I_swap_ablation.csv'), 'w') as f:
    w = csv.writer(f)
    w.writerow(['scene', 'combo', 'cc'])
    for row in results_ALL['analysis_I']:
        w.writerow(row)

# Save Fisher arrays
for (scene, vname), v in fisher_by_scene_variant.items():
    np.savez(os.path.join(OUT_DIR, f'G_fisher_{scene}_{vname}.npz'),
             grad_raw=v['grad_raw'], grad_rot=v['grad_rot'], cc=v['cc'])

# Save renders
for (scene, vname), arr in renders_by_scene_variant.items():
    np.save(os.path.join(OUT_DIR, f'render_{scene}_{vname}.npy'), arr)
for scene, arr in gt_cart_by_scene.items():
    np.save(os.path.join(OUT_DIR, f'gt_cart_{scene}.npy'), arr)

# Print Fisher summary
print('\n\n=== Fisher summary (mean |∂cc/∂raw_c|) ===')
print(f'  {"scene":<20} {"variant":<8} | ' +
      ' | '.join(f'{p:<10}' for p in ['eps_real', 'eps_imag', 'sigma_h',
                                        'l_c', 'tau', 'thick']))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        g = fisher_by_scene_variant[(scene, vname)]['grad_raw']
        print(f'  {scene:<20} {vname:<8} | ' +
              ' | '.join(f'{np.abs(g[:, c]).mean():<10.3e}' for c in range(6)))

print('\nDone. Outputs in', OUT_DIR)
