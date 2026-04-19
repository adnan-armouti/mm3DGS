"""7-scene matched-grid frame-NVS analyses (fast variants only).

Runs Analyses A–K on HO_8 (first-chirp) variants across 7 scenes, using:
  HO_base      : baseline HO_8 (no S4)
  HO_S4_jit    : HO_8 + S4 jitter 0.02/50 (6-scene winner)
  HO_S4_ann    : HO_8 + S4 pool_knn annulus [0.02, 0.10] (7-scene winner)
  UB_9         : first-chirp UB (9 train frames × 1 chirp)

Drops seq_0_frame_390 from the means (known-outlier scene) — reported
separately where relevant.

Outputs: md/frame_nvs_analysis_7scene/
"""
import os, sys, json, gc, csv
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

# Config
SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]
SCENE_FRAME = {s: int(s.split('_')[-1]) for s in SCENES}
OUTLIER_SCENE = 'seq_0_frame_390'
TARGET_N = 20000

# Variant → output-dir pattern
VARIANTS = {
    'HO_base':    lambda s, F: f'{s}_train8frames_1loops_test{F}_loop0_pass2_N{TARGET_N}',
    'HO_S4_jit':  lambda s, F: f'{s}_train8frames_1loops_test{F}_loop0_pass2_N{TARGET_N}_dnsfyjt0.02i50u300p0.02',
    'HO_S4_ann':  lambda s, F: f'{s}_train8frames_1loops_test{F}_loop0_pass2_N{TARGET_N}_dnsfypk0.02i50u300r0.1ann0.02',
    'UB_9':       lambda s, F: f'{s}_train9frames_1loops_test{F}_loop0_ub_pass2_N{TARGET_N}',
}
VARIANT_ORDER = ['HO_base', 'HO_S4_jit', 'HO_S4_ann', 'UB_9']
PARAM_NAMES = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thickness']
PARAM_LOG_X = [False, True, True, True, False, True]

RUN_ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v5', 'output_frame_nvs')
OUT_DIR = os.path.join(PROJECT_ROOT, 'md', 'frame_nvs_analysis_7scene')
os.makedirs(OUT_DIR, exist_ok=True)


def quat_to_normal_np(q_np):
    q = torch.nn.functional.normalize(torch.from_numpy(q_np).float(), dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2 * (x * z + w * y); ny = 2 * (y * z - w * x); nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1).numpy()


def load_state(scene, vname):
    F = SCENE_FRAME[scene]
    d = os.path.join(RUN_ROOT, VARIANTS[vname](scene, F))
    rp = os.path.join(d, 'results.json')
    bp = os.path.join(d, 'best_model.pt')
    if not (os.path.isfile(rp) and os.path.isfile(bp)):
        return None, None, None
    st = torch.load(bp, weights_only=False, map_location='cpu')
    meta = json.load(open(rp))
    history = dict(np.load(os.path.join(d, 'history.npz')))
    return st, meta, history


def load_variant(scene, vname):
    st, meta, history = load_state(scene, vname)
    if st is None:
        return None
    raw = st['raw_materials'].numpy()
    return {
        'positions': st['positions'].numpy(),
        'rotations': st['rotations'].numpy(),
        'raw': raw,
        'physics': reparameterize_torch(torch.from_numpy(raw)).numpy(),
        'normals': quat_to_normal_np(st['rotations'].numpy()),
        'meta': meta, 'history': history,
    }


def load_init_state(scene, positions):
    pcl = np.load(os.path.join(PROJECT_ROOT, 'data', scene, 'scene', 'pcl.npy'))
    xyz = pcl[:, :3].astype(np.float32)
    nrm = pcl[:, 3:6].astype(np.float32)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(xyz)
    dists, idx = tree.query(positions, k=1)
    init_raw_row = inverse_reparameterize_torch(ITU_CONCRETE[None, :])[0]
    return {
        'init_raw': np.broadcast_to(init_raw_row[None, :], (len(positions), 6)).copy(),
        'init_normals': nrm[idx],
        'init_rotations': _normals_to_quaternions(nrm[idx]),
        'pcl_nn_dist': dists.astype(np.float32),
    }


# --- Rendering helpers (for Analyses 0, G, I) ---
def build_rast_and_base_model(scene, seed_frame, target_n):
    config = load_trained_config(scene)
    seed_cfg = os.path.join(PROJECT_ROOT, 'data', 'alignment_data', scene,
                            'cascade', f'cascaded_frame_{seed_frame}_aligned_pass2.json')
    rast = Rasterizer(config_file=seed_cfg, mesh_file=config.scene_file,
                      tx_pattern_file=config.tx_pattern_file,
                      rx_pattern_file=config.rx_pattern_file, device=DEVICE)
    if not USE_FACTORY_PATTERNS:
        rast.inject_trained_params(pattern_data=load_pattern_data(scene))
    model = init_visible_weighted(scene, rast, target_n=target_n)
    rast.free_mi_scene()
    gc.collect(); torch.cuda.empty_cache()
    range_res = compute_range_res_from_cfg(seed_cfg)
    sample_grid = build_polar_to_cart_grid(127, 256, range_res, 400, DEVICE)
    return rast, model, sample_grid, seed_cfg


def build_test_pose(scene):
    F = SCENE_FRAME[scene]
    align_dir = os.path.join(PROJECT_ROOT, 'data', 'alignment_data', scene, 'cascade')
    cfg_A = os.path.join(align_dir, f'cascaded_frame_{F-1}_aligned_pass2.json')
    cfg_B = os.path.join(align_dir, f'cascaded_frame_{F+1}_aligned_pass2.json')
    poses, _ = build_per_loop_poses(cfg_A, cfg_B, n_loops=16, device=DEVICE)
    return poses[0]


def load_test_gt_cart(scene, sample_grid):
    F = SCENE_FRAME[scene]
    adc_npy = os.path.join(PROJECT_ROOT, 'data', scene, 'radar', f'cascaded_frame_{F}.npy')
    arr = np.load(adc_npy)
    ri = np.stack([arr[0].real, arr[0].imag], axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)
    gt_adc = torch.from_numpy(ri).to(DEVICE)
    with torch.no_grad():
        ra_c = adc_to_ra_complex(gt_adc)
        gt_cart = polar_to_cart_torch(torch.abs(ra_c).float(), sample_grid)
        mn, mx = gt_cart.min(), gt_cart.max()
        return ((gt_cart - mn) / (mx - mn).clamp(min=1e-30)).detach()


def render_cart(model, rast, varea, amask, sgrid, grad=False):
    if grad:
        rp_real, rp_imag = render_gaussians(model, rast, vertex_areas=varea,
                                             active_mask=amask, shadow_mask=None,
                                             bsdf_mode='full', disabled_components=None)
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        return polar_to_cart_torch(ra_polar, sgrid)
    with torch.no_grad():
        rp_real, rp_imag = render_gaussians(model, rast, vertex_areas=varea,
                                             active_mask=amask, shadow_mask=None,
                                             bsdf_mode='full', disabled_components=None)
        ra_polar = range_profile_to_ra_mag(rp_real, rp_imag)
        return polar_to_cart_torch(ra_polar, sgrid)


# ===========================================================================
# Main
# ===========================================================================
print('=' * 72); print('7-scene matched-grid analyses'); print('=' * 72)

# Load all variants, filter scenes where any variant is missing.
DATA = {}
for scene in SCENES:
    DATA[scene] = {}
    missing = []
    for v in VARIANT_ORDER:
        d = load_variant(scene, v)
        if d is None:
            missing.append(v)
        DATA[scene][v] = d
    # init state (same for all variants on the matched grid)
    if DATA[scene]['HO_base'] is not None:
        DATA[scene]['init'] = load_init_state(scene, DATA[scene]['HO_base']['positions'])
    if missing:
        print(f'  {scene}: MISSING variants: {missing}')
    else:
        # Sanity: check positions match across HO variants (they should on matched grid)
        pos_ok = np.allclose(DATA[scene]['HO_base']['positions'],
                              DATA[scene]['UB_9']['positions'])
        print(f'  {scene}: loaded all 4 variants. HO==UB positions ✓={pos_ok}')

SCENES_OK = [s for s in SCENES
             if all(DATA[s].get(v) is not None for v in VARIANT_ORDER)]
print(f'\n  Analysable scenes (all 4 variants present): {len(SCENES_OK)}/{len(SCENES)}')
SCENES_OK_NO_OUT = [s for s in SCENES_OK if s != OUTLIER_SCENE]
print(f'  With outlier ({OUTLIER_SCENE}) removed: {len(SCENES_OK_NO_OUT)} scenes')

# -----------------------------------------------------------------------------
# Analysis 0 — reproducibility (re-render + verify cc vs reported)
# -----------------------------------------------------------------------------
print('\n=== Analysis 0: reproducibility ===')
rows_0 = []
renders_by_sv = {}
gt_cart_by_scene = {}
fisher_by_sv = {}

for scene in SCENES_OK:
    F = SCENE_FRAME[scene]
    test_pose = build_test_pose(scene)
    rast, model, sgrid, _ = build_rast_and_base_model(scene, F, TARGET_N)
    apply_pose(rast, test_pose)
    amask = cull_gaussians(model, rast)
    varea = torch.zeros(model.N, device=DEVICE); varea[amask] = 1.0
    gt_cart = load_test_gt_cart(scene, sgrid)
    gt_cart_by_scene[scene] = gt_cart.cpu().numpy()
    for v in VARIANT_ORDER:
        st, meta, _ = load_state(scene, v)
        with torch.no_grad():
            model.positions.copy_(st['positions'].to(DEVICE))
            model.rotations.copy_(st['rotations'].to(DEVICE))
            model.raw_materials.copy_(st['raw_materials'].to(DEVICE))
        rend = render_cart(model, rast, varea, amask, sgrid)
        cc = cart_corr_torch(rend, gt_cart).item()
        renders_by_sv[(scene, v)] = rend.cpu().numpy()
        rows_0.append({'scene': scene, 'variant': v,
                       'reported': meta['final_test_cc'], 'measured': cc,
                       'delta': abs(cc - meta['final_test_cc']),
                       'final_train_cc': meta['final_train_mean_cc']})
        # Fisher snapshot (per-point ∂cc/∂raw + ∂cc/∂rot at converged state)
        model.raw_materials.requires_grad_(True); model.rotations.requires_grad_(True)
        with torch.no_grad():
            model.raw_materials.copy_(st['raw_materials'].to(DEVICE))
            model.rotations.copy_(st['rotations'].to(DEVICE))
        model.raw_materials.grad = None; model.rotations.grad = None
        rend = render_cart(model, rast, varea, amask, sgrid, grad=True)
        cc_diff = cart_corr_torch(rend, gt_cart)
        cc_diff.backward()
        fisher_by_sv[(scene, v)] = {
            'grad_raw': model.raw_materials.grad.detach().cpu().numpy(),
            'grad_rot': model.rotations.grad.detach().cpu().numpy(),
            'cc': float(cc_diff.item()),
        }
        model.raw_materials.requires_grad_(False); model.rotations.requires_grad_(False)
    del rast, model; gc.collect(); torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, '0_reproducibility.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_0[0].keys()); w.writeheader(); w.writerows(rows_0)
for (scene, v), arr in renders_by_sv.items():
    np.save(os.path.join(OUT_DIR, f'render_{scene}_{v}.npy'), arr)
for scene, arr in gt_cart_by_scene.items():
    np.save(os.path.join(OUT_DIR, f'gt_cart_{scene}.npy'), arr)
for (scene, v), d in fisher_by_sv.items():
    np.savez(os.path.join(OUT_DIR, f'G_fisher_{scene}_{v}.npz'),
             grad_raw=d['grad_raw'], grad_rot=d['grad_rot'], cc=d['cc'])

print(f'  Max |reported - measured| across 4 variants × {len(SCENES_OK)} scenes: '
      f'{max(r["delta"] for r in rows_0):.4f}')

# -----------------------------------------------------------------------------
# Analysis A — per-variant parameter distribution summary
# -----------------------------------------------------------------------------
print('\n=== Analysis A: parameter distributions (raw std per col) ===')
print(f'  {"scene":<20} {"variant":<11} | ' + ' | '.join(f'{p:<8}' for p in PARAM_NAMES))
rows_A = []
for scene in SCENES_OK:
    for v in VARIANT_ORDER:
        raw = DATA[scene][v]['raw']
        stds = [float(raw[:, c].std()) for c in range(6)]
        rows_A.append({'scene': scene, 'variant': v,
                        **{f'std_{p}': s for p, s in zip(PARAM_NAMES, stds)}})
        print(f'  {scene:<20} {v:<11} | ' + ' | '.join(f'{s:<8.3f}' for s in stds))
with open(os.path.join(OUT_DIR, 'A_param_std.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_A[0].keys()); w.writeheader(); w.writerows(rows_A)

# -----------------------------------------------------------------------------
# Analysis B — drift from init (L2 material, deg normal)
# -----------------------------------------------------------------------------
print('\n=== Analysis B: drift from init ===')
print(f'  {"scene":<20} {"variant":<11} {"L2 mean":>8} {"deg mean":>9}')
rows_B = []
for scene in SCENES_OK:
    init = DATA[scene]['init']
    for v in VARIANT_ORDER:
        d = DATA[scene][v]
        drift_L2 = np.linalg.norm(d['raw'] - init['init_raw'], axis=1)
        cos_ang = np.clip((d['normals'] * init['init_normals']).sum(-1), -1, 1)
        drift_deg = np.degrees(np.arccos(cos_ang))
        rows_B.append({'scene': scene, 'variant': v,
                       'L2_mean': float(drift_L2.mean()),
                       'L2_p95': float(np.percentile(drift_L2, 95)),
                       'deg_mean': float(drift_deg.mean()),
                       'deg_p95': float(np.percentile(drift_deg, 95))})
        print(f'  {scene:<20} {v:<11} {drift_L2.mean():>8.3f} {drift_deg.mean():>9.2f}')
with open(os.path.join(OUT_DIR, 'B_drift.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_B[0].keys()); w.writeheader(); w.writerows(rows_B)

# -----------------------------------------------------------------------------
# Analysis C — per-point UB vs HO divergence (matched grid = exact subtract)
# -----------------------------------------------------------------------------
print('\n=== Analysis C: per-point UB_9 vs HO_base divergence (|Δraw| mean per col) ===')
print(f'  {"scene":<20} | ' + ' | '.join(f'{p:<8}' for p in PARAM_NAMES))
rows_C = []
for scene in SCENES_OK:
    ub = DATA[scene]['UB_9']['raw']
    ho = DATA[scene]['HO_base']['raw']
    diffs = [float(np.abs(ub[:, c] - ho[:, c]).mean()) for c in range(6)]
    rows_C.append({'scene': scene,
                    **{f'abs_mean_{p}': d for p, d in zip(PARAM_NAMES, diffs)}})
    print(f'  {scene:<20} | ' + ' | '.join(f'{d:<8.3f}' for d in diffs))
with open(os.path.join(OUT_DIR, 'C_divergence.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_C[0].keys()); w.writeheader(); w.writerows(rows_C)

# -----------------------------------------------------------------------------
# Analysis D — k-NN material smoothness
# -----------------------------------------------------------------------------
print('\n=== Analysis D: k-NN material smoothness (mean neighbour-std) ===')
print(f'  {"scene":<20} {"variant":<11} | ' + ' | '.join(f'{p:<8}' for p in PARAM_NAMES))
K = 10
rows_D = []
for scene in SCENES_OK:
    pos = DATA[scene]['HO_base']['positions']
    tree = cKDTree(pos)
    _, nn_idx = tree.query(pos, k=K + 1)
    nn_idx = nn_idx[:, 1:]
    for v in VARIANT_ORDER:
        raw = DATA[scene][v]['raw']
        stds = []
        for c in range(6):
            s_mean = float(raw[:, c][nn_idx].std(axis=1).mean())
            stds.append(s_mean)
        rows_D.append({'scene': scene, 'variant': v,
                        **{f'nbr_std_{p}': s for p, s in zip(PARAM_NAMES, stds)}})
        print(f'  {scene:<20} {v:<11} | ' + ' | '.join(f'{s:<8.3f}' for s in stds))
with open(os.path.join(OUT_DIR, 'D_smoothness.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_D[0].keys()); w.writeheader(); w.writerows(rows_D)

# -----------------------------------------------------------------------------
# Analysis E — normal-field smoothness
# -----------------------------------------------------------------------------
print('\n=== Analysis E: normal-field smoothness (mean neighbour-angle, deg) ===')
rows_E = []
for scene in SCENES_OK:
    pos = DATA[scene]['HO_base']['positions']
    tree = cKDTree(pos)
    _, nn_idx = tree.query(pos, k=K + 1)
    nn_idx = nn_idx[:, 1:]
    for v in VARIANT_ORDER + ['init']:
        if v == 'init':
            n = DATA[scene]['init']['init_normals']
        else:
            n = DATA[scene][v]['normals']
        dot = (n[:, None, :] * n[nn_idx]).sum(-1).clip(-1, 1)
        ang = np.degrees(np.arccos(dot)).mean(axis=1)
        rows_E.append({'scene': scene, 'variant': v,
                       'mean': float(ang.mean()), 'p95': float(np.percentile(ang, 95))})
        print(f'  {scene:<20} {v:<11} mean={ang.mean():6.2f}° p95={np.percentile(ang, 95):6.2f}°')
with open(os.path.join(OUT_DIR, 'E_normal_smoothness.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_E[0].keys()); w.writeheader(); w.writerows(rows_E)

# -----------------------------------------------------------------------------
# Analysis G — Fisher concentration
# -----------------------------------------------------------------------------
print('\n=== Analysis G: Fisher concentration (top-K% of total Fisher) ===')
print(f'  {"scene":<20} {"variant":<11} {"top1%":>6} {"top5%":>6} {"top10%":>6} {"top25%":>6}')
rows_G = []
for scene in SCENES_OK:
    for v in VARIANT_ORDER:
        g = fisher_by_sv[(scene, v)]['grad_raw']
        per_pt = (g ** 2).sum(axis=1)
        srt = np.sort(per_pt)[::-1]
        tot = per_pt.sum()
        row = {'scene': scene, 'variant': v, 'N': int(len(per_pt)),
               'cc': fisher_by_sv[(scene, v)]['cc']}
        for frac in [0.01, 0.05, 0.10, 0.25]:
            k = int(frac * len(per_pt))
            row[f'top_{int(frac*100)}pct'] = float(100 * srt[:k].sum() / max(tot, 1e-30))
        rows_G.append(row)
        print(f'  {scene:<20} {v:<11} '
              f'{row["top_1pct"]:>6.1f} {row["top_5pct"]:>6.1f} '
              f'{row["top_10pct"]:>6.1f} {row["top_25pct"]:>6.1f}')
with open(os.path.join(OUT_DIR, 'G_concentration.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_G[0].keys()); w.writeheader(); w.writerows(rows_G)

# -----------------------------------------------------------------------------
# Analysis I — swap ablation (HO positions+HO rotations vs HO positions+UB rotations etc.)
# All variants share matched positions; swap is exact.
# -----------------------------------------------------------------------------
print('\n=== Analysis I: material/normal swap ablation (exact; matched grid) ===')
rows_I = []
for scene in SCENES_OK:
    F = SCENE_FRAME[scene]
    test_pose = build_test_pose(scene)
    rast, model, sgrid, _ = build_rast_and_base_model(scene, F, TARGET_N)
    apply_pose(rast, test_pose)
    amask = cull_gaussians(model, rast)
    varea = torch.zeros(model.N, device=DEVICE); varea[amask] = 1.0
    gt_cart = load_test_gt_cart(scene, sgrid)

    st_ho = load_state(scene, 'HO_base')[0]
    st_ub = load_state(scene, 'UB_9')[0]

    def _render(st_pos, st_rot, st_raw):
        with torch.no_grad():
            model.positions.copy_(st_pos.to(DEVICE))
            model.rotations.copy_(st_rot.to(DEVICE))
            model.raw_materials.copy_(st_raw.to(DEVICE))
        return cart_corr_torch(render_cart(model, rast, varea, amask, sgrid), gt_cart).item()

    cc_HO      = _render(st_ho['positions'], st_ho['rotations'], st_ho['raw_materials'])
    cc_UB      = _render(st_ub['positions'], st_ub['rotations'], st_ub['raw_materials'])
    cc_UB_mat  = _render(st_ho['positions'], st_ho['rotations'], st_ub['raw_materials'])
    cc_UB_rot  = _render(st_ho['positions'], st_ub['rotations'], st_ho['raw_materials'])
    rows_I.append({'scene': scene, 'cc_HO_native': cc_HO, 'cc_UB_native': cc_UB,
                    'cc_UB_mat_HO_rot': cc_UB_mat, 'cc_HO_mat_UB_rot': cc_UB_rot,
                    'gain_from_UB_mat': cc_UB_mat - cc_HO,
                    'gain_from_UB_rot': cc_UB_rot - cc_HO,
                    'gap_HO_to_UB': cc_UB - cc_HO})
    print(f'  {scene:<20} HO={cc_HO:.4f} UB={cc_UB:.4f}  '
          f'UB_mat+HO_rot={cc_UB_mat:.4f} ({cc_UB_mat-cc_HO:+.3f})  '
          f'HO_mat+UB_rot={cc_UB_rot:.4f} ({cc_UB_rot-cc_HO:+.3f})')
    del rast, model; gc.collect(); torch.cuda.empty_cache()

with open(os.path.join(OUT_DIR, 'I_swap_ablation.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_I[0].keys()); w.writeheader(); w.writerows(rows_I)

# -----------------------------------------------------------------------------
# Summary means (with + without outlier scene)
# -----------------------------------------------------------------------------
print('\n=== SUMMARY: test cc means ===')
import statistics as st_mod
print(f'  {"variant":<11}', end='')
print(f'  mean(7 scenes)     mean(6 scenes, no {OUTLIER_SCENE})')
for v in VARIANT_ORDER:
    vals_all = [r['measured'] for r in rows_0 if r['variant'] == v]
    vals_no_out = [r['measured'] for r in rows_0 if r['variant'] == v and r['scene'] != OUTLIER_SCENE]
    m_all = st_mod.mean(vals_all)
    m_no = st_mod.mean(vals_no_out) if vals_no_out else 0
    print(f'  {v:<11}  {m_all:.4f} ({len(vals_all)})   {m_no:.4f} ({len(vals_no_out)})')

print('\n=== SUMMARY: Fisher-weighted normal drift @ top-1% (diagnostic for S2-rot) ===')
for scene in SCENES_OK:
    init_n = DATA[scene]['init']['init_normals']
    print(f'  {scene}:')
    for v in VARIANT_ORDER:
        n = DATA[scene][v]['normals']
        per_pt_F = (fisher_by_sv[(scene, v)]['grad_raw'] ** 2).sum(-1)
        order = np.argsort(per_pt_F)[::-1]
        top_k = int(0.01 * len(per_pt_F))
        idx_top = order[:top_k]
        cos = np.clip((n[idx_top] * init_n[idx_top]).sum(-1), -1, 1)
        ang = np.degrees(np.arccos(cos)).mean()
        print(f'    {v:<11} top-1% drift from init = {ang:>6.2f}°')

print(f'\nDone. Artefacts under {OUT_DIR}')
