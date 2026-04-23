"""Generate 101-frame GT RA Cartesian videos (50 before test, test
itself, 50 after) for each selected scene. Used to visually inspect
whether extending the NVS train-frame window is feasible.

Also emits a decorrelation plot per scene: cart_corr(GT[F_test],
GT[F_test + k]) as a function of k ∈ [-50, 50]. This is the
quantitative analog of "how fast does the scene change" and gives a
direct answer to 'can we feasibly extend training to more frames'.

Runs on GPU; sequential per scene but each scene completes in a few
seconds. Matplotlib figure is cached across frames for speed.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = '/home/adnan/Desktop/mm3DGS'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Raw-ADC loading path (ColoRadar_tools) requires this sub-tree on sys.path
_COLORADAR_TOOLS = os.path.join(PROJECT_ROOT, 'mmir', 'preprocessing')
if _COLORADAR_TOOLS not in sys.path:
    sys.path.insert(0, _COLORADAR_TOOLS)

import mitsuba as mi  # noqa: E402
try:
    mi.set_variant('cuda_ad_rgb')
except Exception:
    pass

from mmir.data.ra_utils import adc_to_ra_complex
from mmir.data.io_utils import compute_range_res_from_cfg
from mm25DGS_v5.train_gaussian import (
    build_polar_to_cart_grid, polar_to_cart_torch, cart_corr_torch, DEVICE,
)
from ColoRadar_tools.dataset_loaders import (                       # noqa: E402
    get_adc_frame, get_cascade_params,
)
from ColoRadar_tools.calibration import apply_frequency_calibration  # noqa: E402


SCENES = [
    # (scene_tag, seq_idx, center_frame, raw_seq_subdir_suffix)
    ('seq_0_frame_135', 0, 135),
    ('seq_1_frame_185', 1, 185),
    ('seq_1_frame_277', 1, 277),
    ('seq_1_frame_438', 1, 438),
    ('seq_2_frame_105', 2, 105),
    ('seq_2_frame_160', 2, 160),
    ('seq_2_frame_300', 2, 300),
]
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
RAW_ROOT = '/home/adnan/Documents/Data/coloRadar/raw/kitti'
CALIB_PATH = '/home/adnan/Documents/Data/coloRadar/raw/utils/calib'
# Sequence-idx → raw-sequence-directory mapping (mirrors
# mmir/preprocessing/alignment/sc_trajectory_transfer.py).
SEQ_NAME_FOR_IDX = {
    0: '2_28_2021_outdoors_run0',
    1: '2_28_2021_outdoors_run1',
    2: '2_28_2021_outdoors_run2',
}
OUT_ROOT = os.path.join(PROJECT_ROOT,
                         'mm25DGS_v6/output_frame_nvs/_scene_videos')
N_HALF = 50
FPS = 10

# Cascade params cached per sequence (expensive to parse for every frame).
_CASCADE_PARAM_CACHE: dict = {}


def _get_cascade_params_cached():
    if 'all' not in _CASCADE_PARAM_CACHE:
        _CASCADE_PARAM_CACHE['all'] = get_cascade_params(CALIB_PATH)
    return _CASCADE_PARAM_CACHE['all']


def _load_adc_chirp0_from_npy(scene: str, frame: int):
    path = os.path.join(DATA_ROOT, scene, 'radar',
                         f'cascaded_frame_{frame}.npy')
    if not os.path.isfile(path):
        return None
    arr = np.load(path)
    assert arr.ndim == 4 and arr.shape[0] == 16, (
        f'unexpected ADC shape: {arr.shape}')
    ri = np.stack([arr[0].real, arr[0].imag], axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)                              # (TX, RX, K, 2)
    return torch.from_numpy(ri).to(DEVICE)


def _load_adc_chirp0_from_raw(seq_idx: int, frame: int):
    seq_name = SEQ_NAME_FOR_IDX.get(seq_idx)
    if seq_name is None:
        return None
    seq_path = os.path.join(RAW_ROOT, seq_name)
    all_params = _get_cascade_params_cached()
    wf = all_params['waveform']
    adc = get_adc_frame(frame, seq_path, wf)                   # (TX, RX, CH, ADC) cplx
    if adc is None:
        return None
    adc = apply_frequency_calibration(adc, all_params['frequency'], wf)
    adc = adc.transpose(2, 1, 0, 3)                            # (CH, RX, TX, ADC)
    chirp0 = adc[0]                                            # (RX, TX, ADC) cplx
    ri = np.stack([chirp0.real, chirp0.imag], axis=-1).astype(np.float32)
    ri = ri.transpose(1, 0, 2, 3)                              # (TX, RX, ADC, 2)
    return torch.from_numpy(ri).to(DEVICE)


def _load_adc_chirp0(scene: str, seq_idx: int, frame: int):
    """Try benchmark .npy first; fall back to raw .bin otherwise."""
    t = _load_adc_chirp0_from_npy(scene, frame)
    if t is not None:
        return t
    return _load_adc_chirp0_from_raw(seq_idx, frame)


def _polar_mag(adc_ri: torch.Tensor):
    with torch.no_grad():
        return torch.abs(adc_to_ra_complex(adc_ri)).float()


def _compute_cart_stack(scene: str, seq_idx: int, F_center: int,
                          sample_grid, range_res):
    """Return ``(frames, carts, success_mask)`` for the 2·N_HALF+1 window.
    Missing frames (not on disk) are skipped and noted in success_mask."""
    frames = list(range(F_center - N_HALF, F_center + N_HALF + 1))
    carts = []
    mask = []
    for F in frames:
        adc = _load_adc_chirp0(scene, seq_idx, F)
        if adc is None:
            carts.append(None)
            mask.append(False)
            continue
        pol = _polar_mag(adc)
        with torch.no_grad():
            ct = polar_to_cart_torch(pol, sample_grid).cpu().numpy()
        carts.append(ct)
        mask.append(True)
        del adc, pol
    return frames, carts, mask


def _render_video(scene: str, F_center: int, frames, carts, mask,
                   range_res: float, out_path: str, fps: int = FPS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio

    n_range, n_az = None, None
    for c in carts:
        if c is not None:
            n_range, n_az = c.shape
            break
    range_depth = n_range * range_res
    range_width = range_depth / 2.0
    extent = [-range_width, range_width, 0.0, range_depth]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    # Placeholder initial image
    placeholder = np.zeros((n_range, n_az), dtype=np.float32)
    im = ax.imshow(placeholder, cmap='hot', aspect='auto', origin='lower',
                    vmin=0.0, vmax=1.0, extent=extent)
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('Azimuth (m)')
    ax.set_ylabel('Range (m)')
    title = ax.set_title('')

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264',
                                 quality=8, macro_block_size=1)
    eps = 1e-30
    for F, cart, ok in zip(frames, carts, mask):
        if not ok:
            continue
        mn, mx = float(cart.min()), float(cart.max())
        disp = (cart - mn) / max(mx - mn, eps)
        im.set_data(disp)
        # im vmin/vmax are already [0, 1] for linear scale
        is_test = (F == F_center)
        banner = '*** TEST *** ' if is_test else ''
        title.set_text(f'{banner}{scene}  F={F}  (offset {F - F_center:+d})')
        if is_test:
            title.set_color('red')
        else:
            title.set_color('black')
        fig.canvas.draw()
        # fig.canvas.tostring_rgb() removed in matplotlib 3.10; use
        # buffer_rgba() and strip the alpha channel.
        rgba = np.asarray(fig.canvas.buffer_rgba())                # (H, W, 4)
        frame_rgb = rgba[..., :3].copy()
        writer.append_data(frame_rgb)
    writer.close()
    plt.close(fig)


def _decorrelation_plot(scene: str, F_center: int, frames, carts, mask,
                         out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # cc(GT_test_cart, GT_test+k_cart) for k in [-N_HALF, N_HALF]
    offsets = []
    ccs = []
    # Find test cart
    test_idx = N_HALF
    if not mask[test_idx]:
        return None
    test_cart_np = carts[test_idx]
    mn_t, mx_t = test_cart_np.min(), test_cart_np.max()
    test_cart_norm = (test_cart_np - mn_t) / max(mx_t - mn_t, 1e-30)
    test_t = torch.from_numpy(test_cart_norm).to(DEVICE)

    for k, F, cart, ok in zip(
            range(-N_HALF, N_HALF + 1), frames, carts, mask):
        if not ok:
            continue
        mn, mx = cart.min(), cart.max()
        other_norm = (cart - mn) / max(mx - mn, 1e-30)
        other_t = torch.from_numpy(other_norm).to(DEVICE)
        cc = cart_corr_torch(other_t, test_t).item()
        offsets.append(k)
        ccs.append(cc)
        del other_t

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    ax.plot(offsets, ccs, '-', color='tab:blue', linewidth=1.2)
    ax.axvline(0, color='red', linestyle=':', linewidth=1, label='test frame')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8,
                alpha=0.8, label='cc = 0.5')
    # Current NVS training window = F±4
    ax.axvspan(-4, -1, color='tab:green', alpha=0.15,
                label='current HO_8 train bracket')
    ax.axvspan(1, 4, color='tab:green', alpha=0.15)
    ax.set_xlabel('frame offset k (from test frame F)')
    ax.set_ylabel('cart_corr(GT[F], GT[F+k])')
    ax.set_title(f'Scene-decorrelation vs frame offset — {scene}  (F={F_center})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return list(zip(offsets, ccs))


def _cfg_path(scene, F):
    return os.path.join(DATA_ROOT, 'alignment_data', scene, 'cascade',
                         f'cascaded_frame_{F}_aligned_pass2.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='all',
                     help='Comma-sep subset of scene names, or "all".')
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    if args.scenes != 'all':
        want = set(args.scenes.split(','))
        sel = [(s, seq, F) for (s, seq, F) in SCENES if s in want]
    else:
        sel = SCENES

    summary_rows = []
    for scene, seq_idx, F in sel:
        t0 = time.time()
        cfg = _cfg_path(scene, F)
        if not os.path.isfile(cfg):
            print(f'[skip] {scene}: no pass-2 aligned config at {cfg}')
            continue
        range_res = compute_range_res_from_cfg(cfg)

        # Build polar→cart grid once.
        adc0 = _load_adc_chirp0(scene, seq_idx, F)
        if adc0 is None:
            print(f'[skip] {scene}: no ADC at F={F}')
            continue
        with torch.no_grad():
            pol0 = _polar_mag(adc0)
        n_az_p, n_range_p = pol0.shape
        sample_grid = build_polar_to_cart_grid(
            n_az_p, n_range_p, range_res, 400, DEVICE)
        del adc0, pol0

        frames, carts, mask = _compute_cart_stack(
            scene, seq_idx, F, sample_grid, range_res)
        n_ok = sum(mask)
        print(f'[{scene}] loaded {n_ok}/{len(frames)} frames '
              f'({time.time() - t0:.1f}s)')

        scene_dir = os.path.join(OUT_ROOT, scene)
        os.makedirs(scene_dir, exist_ok=True)

        # Video
        t1 = time.time()
        out_mp4 = os.path.join(scene_dir, f'{scene}_101frames_gt_ra.mp4')
        _render_video(scene, F, frames, carts, mask, range_res, out_mp4)
        print(f'[{scene}] video wrote {out_mp4} ({time.time() - t1:.1f}s)')

        # Decorrelation plot
        t2 = time.time()
        out_plot = os.path.join(scene_dir, f'{scene}_decorrelation.png')
        dc = _decorrelation_plot(scene, F, frames, carts, mask, out_plot)
        print(f'[{scene}] decorr plot {out_plot} ({time.time() - t2:.1f}s)')

        # Gather cc values at offsets -8..+8 for the summary table.
        if dc is not None:
            dc_map = dict(dc)
            row = {'scene': scene, 'F': F, 'n_ok': n_ok}
            for k in (-8, -4, -2, -1, 1, 2, 4, 8, 16, 32):
                row[f'k_{k}'] = dc_map.get(k, float('nan'))
            summary_rows.append(row)

    # Final summary
    print()
    print('=' * 88)
    print('Scene-decorrelation summary: cart_corr(GT[F_test], GT[F_test+k])')
    print('=' * 88)
    ks = (-8, -4, -2, -1, 1, 2, 4, 8, 16, 32)
    hdr = f'{"scene":<22s} F ' + ' '.join(f'k={k:>+3d}' for k in ks)
    print(hdr)
    print('-' * 88)
    for r in summary_rows:
        line = f'{r["scene"]:<22s} {r["F"]:<4d}'
        for k in ks:
            v = r.get(f'k_{k}', float('nan'))
            line += (f'  {v:5.2f}  ' if not np.isnan(v) else '   ???  ')
        print(line)
    print('=' * 88)


if __name__ == '__main__':
    main()
