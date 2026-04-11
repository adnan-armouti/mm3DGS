"""Test different SPREAD values for range-profile splatting accuracy."""
import sys
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import torch, numpy as np
torch.cuda.empty_cache()

from mm25DGS_v2.train_gaussian import render_gaussians_factorized as render_v2
from mm25DGS_v3.train_gaussian import init_from_mesh, _get_visible_vertices
from mm25DGS_v3.psf import hann_psf
from mm25DGS_v2.rasterizer_torch import RasterizerTorch, reparameterize_torch
from mm25DGS_v2.render_mmIR import load_trained_config, load_best_params
from mm25DGS.bsdf_torch import TWO_PI

C_LIGHT = 299792458.0

scene = 'seq_0_frame_135'
model, config, pattern_data, vertex_areas = init_from_mesh(scene)
raw_params_mmIR, _, _ = load_best_params(scene)
rast = RasterizerTorch(config.config_file, config.scene_file,
                       config.tx_pattern_file, config.rx_pattern_file)
rast.inject_trained_params(raw_params_mmIR, None, pattern_data)
hits = rast._run_reservoir_sampler(seed=42)
visible_verts = _get_visible_vertices(hits, rast.faces_np)
visible_mask = torch.zeros(model.N, dtype=torch.bool, device='cuda:0')
visible_mask[torch.from_numpy(visible_verts).long().to('cuda:0')] = True
del hits; rast._mi_scene = None
import gc; gc.collect(); torch.cuda.empty_cache()

active_mask = visible_mask
n = active_mask.sum().item()
if n > 12000:
    ai = active_mask.nonzero(as_tuple=True)[0]
    torch.manual_seed(42)
    p = torch.randperm(n, device='cuda:0')[:12000]
    active_mask = torch.zeros(model.N, dtype=torch.bool, device='cuda:0')
    active_mask[ai[p]] = True

positions = model.positions[active_mask]
normals = model.get_normals()[active_mask]
opacities = model.get_opacities()[active_mask]
raw_materials = model.raw_materials[active_mask]
areas = vertex_areas[active_mask] * opacities
M = positions.shape[0]
K = 256
n_tx, n_rx = rast.n_tx, rast.n_rx
device = 'cuda:0'

with torch.no_grad():
    # Ground truth: v2 ADC -> Hann -> FFT
    adc_r, adc_i = render_v2(model, rast, vertex_areas=vertex_areas, active_mask=active_mask)
    adc_c = torch.complex(adc_r, adc_i)
    W = torch.hann_window(K, device=device)
    rp_gt = torch.fft.fft(adc_c * W[None, None, :], dim=-1)

    # Compute BSDF intermediates from v3 rasterizer (steps 1-4 only)
    # Run once with SPREAD=21 to get the internal state, then re-splat
    from mm25DGS_v3.rasterizer_factorized import render_factorized

    # We need: w_full, n_peak, phi_carrier from the rasterizer internals.
    # Hack: compute them here using the same code as rasterizer_factorized.py steps 1-4
    # But the BSDF is complex. Instead, let's extract by running with SPREAD=K (full)
    # and confirming it matches FFT(ADC), then vary SPREAD.

    # Compute n_peak and phi_carrier
    diff_tx = rast.tx_positions[None,:,:] - positions[:,None,:]
    d_tx = diff_tx.norm(dim=-1).clamp(min=1e-6)
    diff_rx = positions[:,None,:] - rast.rx_positions[None,:,:]
    d_rx = diff_rx.norm(dim=-1).clamp(min=1e-6)
    d_total = d_tx.unsqueeze(-1) + d_rx.unsqueeze(-2)
    range_res = C_LIGHT * rast.sample_rate / (2.0 * rast.slope * K)
    n_peak = d_total / (2.0 * range_res)
    phi_tx_const = TWO_PI * rast.center_freq * d_tx / C_LIGHT
    phi_rx_const = TWO_PI * rast.center_freq * d_rx / C_LIGHT
    phi_carrier = phi_tx_const.unsqueeze(-1) + phi_rx_const.unsqueeze(-2)

    n_floor = n_peak.floor().long()
    n_frac = n_peak - n_floor.float()

    # Get w_full by running the full rasterizer and extracting from result
    # Can't easily extract w_full. Instead, run SPREAD=K as baseline:
    rp_ref_r, rp_ref_i = render_factorized(
        positions, normals, areas, raw_materials, rast, reparameterize_torch)
    rp_ref = torch.complex(rp_ref_r, rp_ref_i)

    # Now reconstruct the carrier*w from the SPREAD=21 result.
    # At peak bin n, PSF(0) contributes most. But overlapping Gaussians prevent clean extraction.
    # BETTER: compute w_full * exp(j*phi_carrier) from the rasterizer internals.

    # Actually, I realize the right approach: just run the splat loop manually
    # with different SPREAD values, using the carrier extracted from the rasterizer.
    # The carrier = w * exp(j*phi) doesn't depend on SPREAD.

    # Extract carrier from rasterizer by adding a return. Instead, recompute:
    # Run the full BSDF computation (steps 1-4) and stop before step 5.
    # This requires importing the guts of render_factorized.

    # SIMPLEST: just measure the SPREAD=21 result (already have it as rp_ref)
    # against rp_gt, then manually do the splat with different SPREAD.

    # The rp_ref at SPREAD=21 already has energy=0.999989 vs rp_gt.
    # For other SPREADs, the relative accuracy vs rp_gt will differ.

    # To test different SPREADs without re-running BSDF:
    # carrier_complex = w * exp(j*phi_carrier) for each (m, t, r)
    # I can reconstruct this by noting that:
    #   rp_ref[n] = sum_m carrier[m,t,r] * PSF_hann(n - n_peak[m,t,r])
    # But I can't invert this for 12K Gaussians.

    # JUST RE-RUN THE FULL RASTERIZER WITH DIFFERENT SPREAD.
    # The BSDF takes 3ms, the splat takes <1ms. Total ~4ms per SPREAD.
    # For 11 SPREAD values: 44ms.

    # But SPREAD is hardcoded... Let me make it a parameter.
    # For this test script, I'll copy-paste the splat loop and parameterize SPREAD.

    # Actually, the render_factorized function in v3 has SPREAD hardcoded at 21.
    # Let me just modify it to accept spread as a parameter, then call it in a loop.

# Modify render_factorized to accept spread parameter
import mm25DGS_v3.rasterizer_factorized as rf_mod
src = open(rf_mod.__file__).read()

# Check current SPREAD value
import re
match = re.search(r'SPREAD = (\d+)', src)
current_spread = int(match.group(1)) if match else 21

# Replace hardcoded SPREAD with a parameter in chunk_size (which is unused)
# Actually, simpler: just edit and reimport for each test.
# Since importlib.reload doesn't work well, use exec().

# Build a function that does only the splat (step 5) given precomputed intermediates.
# First, run the BSDF (steps 1-4) once to get all intermediates.

# I'll modify the rasterizer to optionally return intermediates.
# For this test, let me just extract what I need.

# OK, cleanest approach: modify render_factorized to take spread as kwarg.
# Add `spread=21` to the signature and use it.

new_src = src.replace(
    'def render_factorized(\n    positions,',
    'def render_factorized(\n    positions,'
).replace(
    "chunk_size=2000,    # unused (kept for API compat)\n):",
    "chunk_size=2000,    # unused (kept for API compat)\n    spread=21,\n):"
).replace(
    'SPREAD = 21  # deposit to ±10 bins around peak',
    'SPREAD = spread  # deposit to ±spread//2 bins around peak'
)

# Write and reload
with open(rf_mod.__file__, 'w') as f:
    f.write(new_src)

import importlib
importlib.reload(rf_mod)

mag_gt_np = rp_gt.abs().cpu().numpy()
sig = mag_gt_np > mag_gt_np.max() * 0.01

print(f'{"SPREAD":>6} {"Energy":>10} {"Corr mean":>10} {"RelErr mean":>12} {"PhaseDiff":>10}')
print('-' * 58)

import time
for spread in [3, 5, 7, 9, 11, 15, 21, 31, 51, 101, 256]:
    with torch.no_grad():
        t0 = time.time()
        rp_r_s, rp_i_s = rf_mod.render_factorized(
            positions, normals, areas, raw_materials, rast,
            reparameterize_torch, spread=spread)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        rp_sp = torch.complex(rp_r_s, rp_i_s)

    mag_sp = rp_sp.abs().cpu().numpy()
    energy = np.sum(mag_sp**2) / np.sum(mag_gt_np**2)
    corrs = [float(np.corrcoef(mag_gt_np[t,r], mag_sp[t,r])[0,1])
             for t in range(n_tx) for r in range(n_rx)]
    rel_err = np.abs(mag_gt_np[sig] - mag_sp[sig]) / mag_gt_np[sig]
    phase_diff = np.angle(rp_gt.cpu().numpy()[sig] * np.conj(rp_sp.cpu().numpy()[sig]))

    print(f'{spread:>6} {energy:>10.6f} {np.mean(corrs):>10.6f} '
          f'{rel_err.mean():>12.4e} {np.degrees(np.abs(phase_diff).mean()):>7.2f} deg  '
          f'({elapsed*1000:.0f}ms)')

# Restore original with SPREAD=21 hardcoded
restored = new_src.replace(
    "chunk_size=2000,    # unused (kept for API compat)\n    spread=21,\n):",
    "chunk_size=2000,    # unused (kept for API compat)\n):"
).replace(
    'SPREAD = spread  # deposit to ±spread//2 bins around peak',
    'SPREAD = 21  # deposit to ±10 bins around peak'
)
with open(rf_mod.__file__, 'w') as f:
    f.write(restored)
print('(source restored to SPREAD=21)')
