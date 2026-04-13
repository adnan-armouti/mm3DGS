"""Phase P5: SPM validity clamp audit.

Loads the final raw_materials from the D_P1_simplified_random dump,
runs a forward pass of enforce_spm_validity on the reparameterized
values, and measures:

  1. What fraction of points have sigma_h clamped at end of training?
  2. Which of the three SPM validity constraints (h_max_1, h_max_2,
     h_max_3) is the binding constraint per point?
  3. Same measurement at INIT (to show the clamp already kills most
     gradient on iter 0).
  4. Same measurement on ITU concrete init (as a reference).

Also audits the reparameterize_torch clamps on raw material columns.
"""

import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from mm25DGS_v4.material_diagnostics import PARAM_NAMES
from mm25DGS_v4.rasterizer import reparameterize_torch, inverse_reparameterize_torch
from mm25DGS.bsdf_torch import K_WAVE, WAVELENGTH
from mm25DGS_v4.train_gaussian import ITU_CONCRETE

DEVICE = 'cuda:0'

# Reparam clamp ranges (the ones inside reparameterize_torch)
REPARAM_CLAMPS = {
    'eps_real':  (None, None),         # sigmoid, no hard clamp (but saturates)
    'eps_imag':  (-7.0, 16.0),
    'sigma_h':   (-16.0, -7.0),
    'l_c':       (-7.6, -2.3),
    'tau_base':  (None, None),
    'thickness': (-7.0, -1.2),
}

# SPM validity constraints (from mm25DGS/bsdf_torch.py enforce_spm_validity)
def spm_constraints(l_c_t):
    h_max_1 = torch.full_like(l_c_t, 0.1 / K_WAVE)                   # kh < 0.1
    l_c_min = torch.full_like(l_c_t, WAVELENGTH * 0.5)
    l_c_clamped = torch.maximum(l_c_t, l_c_min)
    h_max_2 = torch.sqrt((torch.tensor(0.1, device=l_c_t.device) / (K_WAVE**3 * l_c_clamped)).clamp(min=1e-20))
    h_max_3 = 0.21 * l_c_clamped
    return h_max_1, h_max_2, h_max_3, l_c_clamped


def audit(raw, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    raw_t = torch.from_numpy(raw.astype(np.float32)).to(DEVICE)
    M = raw_t.shape[0]

    # Reparameterization clamps
    print("\n--- reparameterize_torch clamps ---")
    print(f"{'param':<12}{'lo':>10}{'hi':>10}{'at_lo':>12}{'at_hi':>12}{'either':>12}")
    for k, name in enumerate(PARAM_NAMES):
        lo, hi = REPARAM_CLAMPS[name]
        if lo is None:
            # sigmoid saturates effectively at ±5
            hits_lo = (raw_t[:, k] < -5.0).float().mean().item() * 100
            hits_hi = (raw_t[:, k] > 5.0).float().mean().item() * 100
            lo_s, hi_s = -5.0, 5.0
        else:
            hits_lo = (raw_t[:, k] < lo + 0.01).float().mean().item() * 100
            hits_hi = (raw_t[:, k] > hi - 0.01).float().mean().item() * 100
            lo_s, hi_s = lo, hi
        total = hits_lo + hits_hi
        print(f"  {name:<10}{lo_s:>10.2f}{hi_s:>10.2f}{hits_lo:>11.1f}%{hits_hi:>11.1f}%{total:>11.1f}%")

    # SPM validity
    phys = reparameterize_torch(raw_t)
    sigma_h = phys[:, 2]
    l_c = phys[:, 3]
    h_max_1, h_max_2, h_max_3, l_c_cl = spm_constraints(l_c)
    # Binding constraint: min of the three
    h_max_raw = torch.minimum(torch.minimum(h_max_1, h_max_2), h_max_3)
    sigma_h_clamped = sigma_h > h_max_raw
    pct_clamped = sigma_h_clamped.float().mean().item() * 100
    pct_l_c_clamped = (l_c < WAVELENGTH * 0.5 + 1e-8).float().mean().item() * 100

    # Which constraint bound is the binding one per clamped point?
    if sigma_h_clamped.any():
        bind_1 = (h_max_raw == h_max_1)[sigma_h_clamped].float().mean().item() * 100
        bind_2 = (h_max_raw == h_max_2)[sigma_h_clamped].float().mean().item() * 100
        bind_3 = (h_max_raw == h_max_3)[sigma_h_clamped].float().mean().item() * 100
    else:
        bind_1 = bind_2 = bind_3 = 0.0

    print(f"\n--- enforce_spm_validity ---")
    print(f"  sigma_h clamped (sigma_h > h_max): {pct_clamped:.1f}% of {M} points")
    print(f"  l_c clamped (l_c < λ/2 = {WAVELENGTH*0.5*1e3:.2f} mm): {pct_l_c_clamped:.1f}%")
    if pct_clamped > 0:
        print(f"  Of clamped points, binding constraint is:")
        print(f"    h_max_1 (kh < 0.1, ≈ 62 μm):     {bind_1:.1f}%")
        print(f"    h_max_2 (k³h²l < 0.1):           {bind_2:.1f}%")
        print(f"    h_max_3 (h < 0.21·l_c):          {bind_3:.1f}%")

    # Distributions
    print(f"\n--- physical sigma_h, l_c distributions ---")
    print(f"  sigma_h: {sigma_h.min().item()*1e6:.1f} – {sigma_h.max().item()*1e6:.1f} μm  "
          f"(median {sigma_h.median().item()*1e6:.1f} μm)")
    print(f"  l_c    : {l_c.min().item()*1e3:.2f} – {l_c.max().item()*1e3:.2f} mm      "
          f"(median {l_c.median().item()*1e3:.2f} mm)")
    print(f"  h_max (binding): {h_max_raw.min().item()*1e6:.1f} – {h_max_raw.max().item()*1e6:.1f} μm")


# --- Load the three interesting raw_materials states ---

# 1. Simplified BSDF + random init, final state
d_simp_rand = np.load('/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_P1_simplified_random/D_P1_simplified_random__seq_0_frame_135.npz', allow_pickle=True)
audit(d_simp_rand['init'],  'SIMPLIFIED BSDF + RANDOM INIT @ ITER 0')
audit(d_simp_rand['final'], 'SIMPLIFIED BSDF + RANDOM INIT @ END OF TRAINING')

# 2. Full BSDF + concrete init, final state
d_full_conc = np.load('/home/adnan/Desktop/mm3DGS/mm25DGS_v4/output/material_investigation/D_mse_raw/D_mse_raw__seq_0_frame_135.npz', allow_pickle=True)
audit(d_full_conc['final'], 'FULL BSDF + CONCRETE INIT @ END OF TRAINING  (for reference)')

# 3. Pure ITU concrete init (all points identical)
concrete_raw = inverse_reparameterize_torch(ITU_CONCRETE)
audit(np.tile(concrete_raw, (50000, 1)), 'PURE ITU CONCRETE INIT (all 50K points identical)')
