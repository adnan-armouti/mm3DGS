"""Analyses A-F + J + K for frame-NVS regulariser design.

Pure numpy/torch (no rendering). Loads the 8 converged state dicts,
recovers init state by KDTree-matching trained positions to pcl.npy,
produces per-variant + cross-variant parameter analyses, and saves
figures + CSV summaries under md/frame_nvs_analysis/.

Run:
    /home/adnan/.conda/envs/mmir/bin/python md/frame_nvs_analysis/run_analyses_ABCDEFJK.py
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mm25DGS_v5.rasterizer import reparameterize_torch, inverse_reparameterize_torch
from mm25DGS_v5.train_gaussian import ITU_CONCRETE, _normals_to_quaternions

SCENES = ['seq_1_frame_438', 'seq_2_frame_105']
SCENE_FRAME = {'seq_1_frame_438': 438, 'seq_2_frame_105': 105}
VARIANTS = {
    'HO_128': lambda F: f'train8frames_test{F}_loop0_pass2',
    'HO_8':   lambda F: f'train8frames_1loops_test{F}_loop0_pass2',
    'UB_144': lambda F: f'train9frames_16loops_test{F}_loop0_ub_pass2',
    'UB_9':   lambda F: f'train9frames_1loops_test{F}_loop0_ub_pass2',
}
VARIANT_ORDER = ['HO_128', 'HO_8', 'UB_144', 'UB_9']
PARAM_NAMES = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thickness']
PARAM_LOG_X = [False, True, True, True, False, True]

ROOT = os.path.join(PROJECT_ROOT, 'mm25DGS_v5', 'output_frame_nvs')
OUT_DIR = os.path.join(PROJECT_ROOT, 'md', 'frame_nvs_analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def quat_to_normal(q_np):
    q = torch.from_numpy(q_np).float()
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1).numpy()


def load_variant(scene, vname):
    F = SCENE_FRAME[scene]
    d = os.path.join(ROOT, f'{scene}_{VARIANTS[vname](F)}')
    st = torch.load(os.path.join(d, 'best_model.pt'),
                    weights_only=False, map_location='cpu')
    meta = json.load(open(os.path.join(d, 'results.json')))
    history = dict(np.load(os.path.join(d, 'history.npz')))
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
    """Recover init state by KDTree-matching trained positions to pcl.npy.

    Returns dict with init_raw (N,6), init_normals (N,3), init_rotations (N,4),
    pcl_intensity (N,) from pcl column 6, and pcl_nn_dist (N,) sanity stat.
    """
    pcl = np.load(os.path.join(PROJECT_ROOT, 'data', scene, 'scene', 'pcl.npy'))
    xyz = pcl[:, :3].astype(np.float32)
    nrm = pcl[:, 3:6].astype(np.float32)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    intensity = pcl[:, 6].astype(np.float32)

    tree = cKDTree(xyz)
    dists, idx = tree.query(positions, k=1)
    init_normals = nrm[idx]
    pcl_int = intensity[idx]
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


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
print('loading variants + init state ...')
DATA = {}
for scene in SCENES:
    DATA[scene] = {}
    for vname in VARIANT_ORDER:
        DATA[scene][vname] = load_variant(scene, vname)
    # init for HO variants (seed frame = F+1) and UB variants (seed frame = F)
    DATA[scene]['init_HO'] = load_init_state(scene, DATA[scene]['HO_128']['positions'])
    DATA[scene]['init_UB'] = load_init_state(scene, DATA[scene]['UB_144']['positions'])
    # Quick sanity checks
    assert np.allclose(DATA[scene]['HO_128']['positions'],
                       DATA[scene]['HO_8']['positions'])
    assert np.allclose(DATA[scene]['UB_144']['positions'],
                       DATA[scene]['UB_9']['positions'])
    print(f'  {scene}: pcl NN dist max (HO) = '
          f'{DATA[scene]["init_HO"]["pcl_nn_dist"].max():.4e} m, '
          f'(UB) = {DATA[scene]["init_UB"]["pcl_nn_dist"].max():.4e} m')

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

# Save CSV
import csv
with open(os.path.join(OUT_DIR, 'A_param_distributions.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_A[0].keys())
    w.writeheader()
    w.writerows(rows_A)

# Plot histograms: 2 scenes × 6 params, 4 variants overlaid per subplot
for scene in SCENES:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
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
        ax.set_title(f'{scene}: {pname}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('density')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'A_hist_{scene}.png'), dpi=110)
    plt.close()

# Normals (3 components)
for scene in SCENES:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for c in range(3):
        for vname in VARIANT_ORDER:
            arr = DATA[scene][vname]['normals'][:, c]
            axes[c].hist(arr, bins=80, alpha=0.45, label=vname, density=True)
        axes[c].set_title(f'{scene}: normal component {"xyz"[c]}')
        axes[c].set_xlabel(f'n_{"xyz"[c]}')
        axes[c].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'A_normals_{scene}.png'), dpi=110)
    plt.close()

# Summary table (markdown-friendly)
print('\nRaw materials std per (scene, variant, param) — tighter UB supports H1:')
header = f'  {"scene":<20} {"variant":<8} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES)
print(header); print('  ' + '-' * (len(header) - 2))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        raws = [f'{DATA[scene][vname]["raw"][:, c].std():<10.3f}'
                for c in range(6)]
        print(f'  {scene:<20} {vname:<8} | ' + ' | '.join(raws))

# ---------------------------------------------------------------------------
# Analysis B: per-point drift from init
# ---------------------------------------------------------------------------
print('\n=== Analysis B: drift from init ===')
rows_B = []
for scene in SCENES:
    for vname in VARIANT_ORDER:
        d = DATA[scene][vname]
        init = DATA[scene]['init_HO' if vname.startswith('HO') else 'init_UB']
        raw = d['raw']
        init_raw = init['init_raw']
        drift_raw = raw - init_raw           # (N, 6)
        drift_L2 = np.linalg.norm(drift_raw, axis=1)  # (N,)
        normals = d['normals']
        init_n = init['init_normals']
        cos_ang = np.clip((normals * init_n).sum(axis=-1), -1.0, 1.0)
        drift_deg = np.degrees(np.arccos(cos_ang))
        rows_B.append({
            'scene': scene, 'variant': vname,
            'drift_L2_mean': float(drift_L2.mean()),
            'drift_L2_median': float(np.median(drift_L2)),
            'drift_L2_p95': float(np.percentile(drift_L2, 95)),
            'drift_deg_mean': float(drift_deg.mean()),
            'drift_deg_median': float(np.median(drift_deg)),
            'drift_deg_p95': float(np.percentile(drift_deg, 95)),
            'frac_material_drift_gt1': float((drift_L2 > 1.0).mean()),
            'frac_normal_drift_gt1deg': float((drift_deg > 1.0).mean()),
            'frac_normal_drift_gt5deg': float((drift_deg > 5.0).mean()),
            'frac_normal_drift_gt15deg': float((drift_deg > 15.0).mean()),
        })
        # Store arrays for plotting
        DATA[scene][vname]['_drift_L2'] = drift_L2
        DATA[scene][vname]['_drift_deg'] = drift_deg

with open(os.path.join(OUT_DIR, 'B_drift_summary.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_B[0].keys())
    w.writeheader()
    w.writerows(rows_B)

print('  drift_L2 (material raw-space L2 drift) / drift_deg (normal angular drift):')
print(f'  {"scene":<20} {"variant":<8} | L2_mean  L2_med  L2_p95  | deg_mean deg_med deg_p95  | >1u >1deg >5deg >15deg')
for r in rows_B:
    print(f'  {r["scene"]:<20} {r["variant"]:<8} | '
          f'{r["drift_L2_mean"]:7.3f} {r["drift_L2_median"]:7.3f} {r["drift_L2_p95"]:7.3f} | '
          f'{r["drift_deg_mean"]:7.2f} {r["drift_deg_median"]:7.2f} {r["drift_deg_p95"]:7.2f} | '
          f'{r["frac_material_drift_gt1"]:4.2f} {r["frac_normal_drift_gt1deg"]:4.2f} '
          f'{r["frac_normal_drift_gt5deg"]:4.2f} {r["frac_normal_drift_gt15deg"]:4.2f}')

# Histograms
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
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'B_drift_{scene}.png'), dpi=110)
    plt.close()

# Per-column drift (raw units, signed)
for scene in SCENES:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for c, pname in enumerate(PARAM_NAMES):
        ax = axes[c]
        for vname in VARIANT_ORDER:
            init = DATA[scene]['init_HO' if vname.startswith('HO') else 'init_UB']
            sdrift = DATA[scene][vname]['raw'][:, c] - init['init_raw'][:, c]
            ax.hist(sdrift, bins=80, alpha=0.45, label=vname, density=True)
        ax.set_title(f'{scene}: raw[{pname}] drift')
        ax.set_xlabel(f'raw drift {pname}')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'B_raw_col_drift_{scene}.png'), dpi=110)
    plt.close()

# ---------------------------------------------------------------------------
# Analysis C: per-point divergence between matched variants (UB vs HO via NN)
# ---------------------------------------------------------------------------
print('\n=== Analysis C: UB vs HO divergence ===')
# Within-mode (same positions): direct subtract
# Across-mode (different positions): NN match HO -> UB
rows_C = []
for scene in SCENES:
    # HO_128 vs HO_8 — identical positions
    d1 = DATA[scene]['HO_128']['raw']; d2 = DATA[scene]['HO_8']['raw']
    for c, pname in enumerate(PARAM_NAMES):
        delta = d1[:, c] - d2[:, c]
        rows_C.append({'scene': scene, 'pair': 'HO_128-HO_8', 'param': pname,
                       'mean': float(delta.mean()), 'std': float(delta.std()),
                       'abs_mean': float(np.abs(delta).mean()),
                       'p95_abs': float(np.percentile(np.abs(delta), 95))})
    # UB_144 vs UB_9
    d1 = DATA[scene]['UB_144']['raw']; d2 = DATA[scene]['UB_9']['raw']
    for c, pname in enumerate(PARAM_NAMES):
        delta = d1[:, c] - d2[:, c]
        rows_C.append({'scene': scene, 'pair': 'UB_144-UB_9', 'param': pname,
                       'mean': float(delta.mean()), 'std': float(delta.std()),
                       'abs_mean': float(np.abs(delta).mean()),
                       'p95_abs': float(np.percentile(np.abs(delta), 95))})
    # UB_144 vs HO_128 via NN (UB_144 positions -> nearest HO_128 position)
    pos_HO = DATA[scene]['HO_128']['positions']
    pos_UB = DATA[scene]['UB_144']['positions']
    tree = cKDTree(pos_HO)
    nn_d, nn_i = tree.query(pos_UB, k=1)
    print(f'  {scene}: UB->HO NN: median={np.median(nn_d):.3f} m, '
          f'p95={np.percentile(nn_d, 95):.3f} m (larger => poorer match)')
    ub_raw = DATA[scene]['UB_144']['raw']
    ho_raw_on_ub_grid = DATA[scene]['HO_128']['raw'][nn_i]
    for c, pname in enumerate(PARAM_NAMES):
        delta = ub_raw[:, c] - ho_raw_on_ub_grid[:, c]
        rows_C.append({'scene': scene, 'pair': 'UB_144-HO_128_NN', 'param': pname,
                       'mean': float(delta.mean()), 'std': float(delta.std()),
                       'abs_mean': float(np.abs(delta).mean()),
                       'p95_abs': float(np.percentile(np.abs(delta), 95))})
    # UB_9 vs HO_8 via NN
    ub_raw = DATA[scene]['UB_9']['raw']
    ho_raw_on_ub_grid = DATA[scene]['HO_8']['raw'][nn_i]
    for c, pname in enumerate(PARAM_NAMES):
        delta = ub_raw[:, c] - ho_raw_on_ub_grid[:, c]
        rows_C.append({'scene': scene, 'pair': 'UB_9-HO_8_NN', 'param': pname,
                       'mean': float(delta.mean()), 'std': float(delta.std()),
                       'abs_mean': float(np.abs(delta).mean()),
                       'p95_abs': float(np.percentile(np.abs(delta), 95))})

with open(os.path.join(OUT_DIR, 'C_divergence.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_C[0].keys())
    w.writeheader()
    w.writerows(rows_C)

print('\n  pair-wise |Δraw| abs_mean (raw-space, larger ⇒ more divergent):')
print(f'  {"scene":<20} {"pair":<22} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for pair in ['HO_128-HO_8', 'UB_144-UB_9', 'UB_144-HO_128_NN', 'UB_9-HO_8_NN']:
        vals = [r['abs_mean'] for r in rows_C
                if r['scene'] == scene and r['pair'] == pair]
        print(f'  {scene:<20} {pair:<22} | ' +
              ' | '.join(f'{v:<10.3f}' for v in vals))

# ---------------------------------------------------------------------------
# Analysis D: spatial smoothness — k-NN neighbour std
# ---------------------------------------------------------------------------
print('\n=== Analysis D: spatial smoothness (K=10 NN) ===')
K = 10
rows_D = []
for scene in SCENES:
    # Build kd tree once per variant-position-group
    trees = {
        'HO': cKDTree(DATA[scene]['HO_128']['positions']),
        'UB': cKDTree(DATA[scene]['UB_144']['positions']),
    }
    nn_by_group = {}
    for g, tr in trees.items():
        pos = DATA[scene][f'{g}_128' if g == 'HO' else f'{g}_144']['positions']
        dists, idx = tr.query(pos, k=K + 1)  # incl self
        nn_by_group[g] = idx[:, 1:]           # drop self
    for vname in VARIANT_ORDER:
        g = 'HO' if vname.startswith('HO') else 'UB'
        raw = DATA[scene][vname]['raw']
        idx = nn_by_group[g]
        for c, pname in enumerate(PARAM_NAMES):
            col = raw[:, c]                     # (N,)
            nbr_vals = col[idx]                 # (N, K)
            nbr_std = nbr_vals.std(axis=1)       # (N,)
            rows_D.append({
                'scene': scene, 'variant': vname, 'param': pname,
                'nbr_std_mean': float(nbr_std.mean()),
                'nbr_std_median': float(np.median(nbr_std)),
                'nbr_std_p95': float(np.percentile(nbr_std, 95)),
            })

with open(os.path.join(OUT_DIR, 'D_spatial_smoothness.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_D[0].keys())
    w.writeheader()
    w.writerows(rows_D)

print('  mean neighbour-std of raw[c] (smaller ⇒ smoother material field):')
print(f'  {"scene":<20} {"variant":<8} | ' + ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        vals = [r['nbr_std_mean'] for r in rows_D
                if r['scene'] == scene and r['variant'] == vname]
        print(f'  {scene:<20} {vname:<8} | ' +
              ' | '.join(f'{v:<10.3f}' for v in vals))

# ---------------------------------------------------------------------------
# Analysis E: normal-field geometry + neighbour consistency
# ---------------------------------------------------------------------------
print('\n=== Analysis E: normal-field geometry ===')
rows_E = []
for scene in SCENES:
    trees = {
        'HO': cKDTree(DATA[scene]['HO_128']['positions']),
        'UB': cKDTree(DATA[scene]['UB_144']['positions']),
    }
    nn_by_group = {}
    for g, tr in trees.items():
        pos = DATA[scene][f'{g}_128' if g == 'HO' else f'{g}_144']['positions']
        _, idx = tr.query(pos, k=K + 1)
        nn_by_group[g] = idx[:, 1:]
    # Also compute init normal neighbour consistency (baseline)
    for group_key, init_key in [('HO', 'init_HO'), ('UB', 'init_UB')]:
        init_n = DATA[scene][init_key]['init_normals']
        nbr_n = init_n[nn_by_group[group_key]]
        dot = (init_n[:, None, :] * nbr_n).sum(-1).clip(-1, 1)
        ang = np.degrees(np.arccos(dot)).mean(axis=1)
        rows_E.append({
            'scene': scene, 'variant': f'{group_key}_init',
            'nbr_normal_ang_mean': float(ang.mean()),
            'nbr_normal_ang_median': float(np.median(ang)),
            'nbr_normal_ang_p95': float(np.percentile(ang, 95)),
        })
    for vname in VARIANT_ORDER:
        g = 'HO' if vname.startswith('HO') else 'UB'
        n = DATA[scene][vname]['normals']
        nbr_n = n[nn_by_group[g]]
        dot = (n[:, None, :] * nbr_n).sum(-1).clip(-1, 1)
        ang = np.degrees(np.arccos(dot)).mean(axis=1)
        rows_E.append({
            'scene': scene, 'variant': vname,
            'nbr_normal_ang_mean': float(ang.mean()),
            'nbr_normal_ang_median': float(np.median(ang)),
            'nbr_normal_ang_p95': float(np.percentile(ang, 95)),
        })

with open(os.path.join(OUT_DIR, 'E_normal_smoothness.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_E[0].keys())
    w.writeheader()
    w.writerows(rows_E)

print('  mean neighbour-angle of normals (deg, smaller ⇒ smoother normal field):')
for r in rows_E:
    print(f'  {r["scene"]:<20} {r["variant"]:<10} '
          f'mean={r["nbr_normal_ang_mean"]:6.2f}  '
          f'med={r["nbr_normal_ang_median"]:6.2f}  '
          f'p95={r["nbr_normal_ang_p95"]:6.2f}')

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
    axes[0].set_title(f'{scene}: train cc trajectory'); axes[0].legend()
    axes[1].set_xlabel('iter'); axes[1].set_ylabel('loss')
    axes[1].set_title(f'{scene}: loss trajectory'); axes[1].legend()
    axes[1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'F_trajectories_{scene}.png'), dpi=110)
    plt.close()
    # Print plateau metric: difference (final - last-100-mean)
    for vname in VARIANT_ORDER:
        h = DATA[scene][vname]['history']
        last100 = h['mean_train_cc'][-100:]
        print(f'  {scene:<20} {vname:<8} '
              f'final_train={h["mean_train_cc"][-1]:.4f}  '
              f'last100_mean={last100.mean():.4f}  '
              f'last100_std={last100.std():.4f}  '
              f'best_iter={DATA[scene][vname]["meta"]["best_iter"]}')

# ---------------------------------------------------------------------------
# Analysis J: intensity prior validation
# ---------------------------------------------------------------------------
print('\n=== Analysis J: intensity prior validation ===')
rows_J = []
N_BINS = 10
for scene in SCENES:
    for vname in VARIANT_ORDER:
        d = DATA[scene][vname]
        init = DATA[scene]['init_HO' if vname.startswith('HO') else 'init_UB']
        intensity = init['pcl_intensity']
        # per-bin variance of raw_materials
        # Rank-based binning for equal-count bins
        order = np.argsort(intensity)
        bin_edges = np.linspace(0, len(intensity), N_BINS + 1, dtype=int)
        for c, pname in enumerate(PARAM_NAMES):
            col = d['raw'][:, c]
            intra_var = []
            for b in range(N_BINS):
                sel = order[bin_edges[b]:bin_edges[b + 1]]
                intra_var.append(col[sel].var())
            rows_J.append({
                'scene': scene, 'variant': vname, 'param': pname,
                'mean_intra_bin_var': float(np.mean(intra_var)),
                'total_var': float(col.var()),
                'ratio_intra_over_total': float(np.mean(intra_var) / max(col.var(), 1e-12)),
            })

with open(os.path.join(OUT_DIR, 'J_intensity_prior.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_J[0].keys())
    w.writeheader()
    w.writerows(rows_J)

print('  intra-bin-var / total-var per (scene, variant, param)')
print('  Smaller ratio ⇒ intensity bins explain more variance ⇒ intensity prior useful.')
print(f'  {"scene":<20} {"variant":<8} | ' +
      ' | '.join(f'{p:<10}' for p in PARAM_NAMES))
for scene in SCENES:
    for vname in VARIANT_ORDER:
        vals = [r['ratio_intra_over_total'] for r in rows_J
                if r['scene'] == scene and r['variant'] == vname]
        print(f'  {scene:<20} {vname:<8} | ' +
              ' | '.join(f'{v:<10.3f}' for v in vals))

# ---------------------------------------------------------------------------
# Analysis K: UB_144 vs UB_9 — noise-averaging / concentration difference
# ---------------------------------------------------------------------------
print('\n=== Analysis K: UB_144 vs UB_9 (noise-averaging) ===')
rows_K = []
for scene in SCENES:
    raw_144 = DATA[scene]['UB_144']['raw']
    raw_9 = DATA[scene]['UB_9']['raw']
    for c, pname in enumerate(PARAM_NAMES):
        sh = raw_9[:, c].std() - raw_144[:, c].std()
        iqr9 = np.percentile(raw_9[:, c], 75) - np.percentile(raw_9[:, c], 25)
        iqr144 = np.percentile(raw_144[:, c], 75) - np.percentile(raw_144[:, c], 25)
        rows_K.append({
            'scene': scene, 'param': pname,
            'raw_std_UB_144': float(raw_144[:, c].std()),
            'raw_std_UB_9':   float(raw_9[:, c].std()),
            'std_delta_9_minus_144': float(sh),
            'iqr_UB_144': float(iqr144), 'iqr_UB_9': float(iqr9),
            'iqr_delta_9_minus_144': float(iqr9 - iqr144),
        })

with open(os.path.join(OUT_DIR, 'K_noise_averaging.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=rows_K[0].keys())
    w.writeheader()
    w.writerows(rows_K)

print('  raw std UB_144 vs UB_9 (UB_9 - UB_144 > 0 ⇒ UB_9 sharper/less smoothed)')
for r in rows_K:
    print(f'  {r["scene"]:<20} {r["param"]:<10} '
          f'std144={r["raw_std_UB_144"]:.3f}  std9={r["raw_std_UB_9"]:.3f}  '
          f'Δstd={r["std_delta_9_minus_144"]:+.3f}  '
          f'Δiqr={r["iqr_delta_9_minus_144"]:+.3f}')

print('\nAll done. Outputs in', OUT_DIR)
