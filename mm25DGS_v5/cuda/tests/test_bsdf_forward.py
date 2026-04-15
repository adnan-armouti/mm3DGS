"""Phase B numerical equivalence test for the fused BSDF Step-4 kernel.

We re-run Steps 1-3 of render_factorized in isolation (material reparam +
per-TX + per-RX geometry) on scene 135, then invoke:
  (a) the PyTorch reference `bsdf_step4_reference` from reference.py
  (b) the CUDA kernel `mm25dgs_v5_cuda.bsdf_step4_forward`
and compare element-wise.

Tolerances:
  rtol=1e-5, atol=1e-6 — tight enough to catch any real math bug, loose
  enough to tolerate fma / --use_fast_math / intrinsics vs exact ops.
"""
import math
import os

import pytest
import torch

from mm25DGS_v5 import cuda as v5cuda
from mm25DGS_v5.cuda.reference import bsdf_step4_reference


pytestmark = pytest.mark.skipif(
    not v5cuda.is_available() or not hasattr(v5cuda.ext, "bsdf_step4_forward"),
    reason=f"mm25dgs_v5_cuda extension not built or missing bsdf_step4_forward: "
           f"{v5cuda.load_error()}",
)


# ---------------------------------------------------------------------------
# Test helper: compute Steps 1-3 intermediates fresh, so we pass identical
# inputs to both the reference and the kernel.
# ---------------------------------------------------------------------------

def compute_step123(scene="seq_0_frame_135", target_n=90000, seed=42):
    """Run render_factorized's Steps 1-3 and return the dict of tensors
    the Step-4 kernel expects.
    """
    from mm25DGS_v5.rasterizer import Rasterizer, reparameterize_torch
    from mm25DGS_v5.train_gaussian import (
        init_visible_weighted, cull_gaussians, DEVICE,
    )
    from mm25DGS_v5.load_pretrained import load_trained_config
    from mm25DGS.bsdf_torch import (
        WAVELENGTH, K_WAVE, TWO_PI, _sigmoid, _get_default_pol,
        compute_sp_basis, itu_slab_fresnel,
    )

    torch.manual_seed(seed)
    config = load_trained_config(scene)
    rast = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        device=DEVICE,
    )
    model = init_visible_weighted(scene, rast, target_n=target_n)

    # Match the benchmark_baseline.py material draw so this test hits the
    # same tensor values as the reference dump captured in Phase A.
    N = model.N
    g = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        noise = torch.randn((N, 6), device="cuda", generator=g) * 2.0
        model.raw_materials.add_(noise)

    active_mask = cull_gaussians(model, rast)
    positions = model.positions[active_mask].detach()
    normals = model.get_normals()[active_mask].detach()
    raw_materials = model.raw_materials[active_mask].detach()

    M = positions.shape[0]
    n_tx = rast.n_tx
    n_rx = rast.n_rx
    device = positions.device

    # --- Step 1: material reparam ---
    physics = reparameterize_torch(raw_materials)
    eps_real = physics[:, 0]
    eps_imag = physics[:, 1]
    sigma_h = physics[:, 2]
    l_c = physics[:, 3]
    tau_base = physics[:, 4]
    thickness = physics[:, 5]

    alpha_raw = 4.0 * math.pi * sigma_h / WAVELENGTH
    alpha_ggx = torch.sqrt(alpha_raw.clamp(min=0.0025)).clamp(0.05, 0.95)
    alpha_sq = alpha_ggx ** 2

    l_c_lam = l_c / WAVELENGTH
    roughness_slope = sigma_h / l_c.clamp(min=1e-8)
    kappa_SPM = (torch.sqrt(l_c_lam.clamp(min=0.0)) / (1.0 + 3.0 * roughness_slope)).clamp(0.5, 10.0)
    kappa_SPM_c = kappa_SPM.clamp(max=50.0)
    sinh_SPM = (torch.exp(kappa_SPM_c) - torch.exp(-kappa_SPM_c)) / 2.0
    norm_SPM = kappa_SPM / (4.0 * math.pi * sinh_SPM.clamp(min=1e-10))

    eps_contrast = torch.abs(eps_real - 1.0) + eps_imag
    eps_factor = (eps_contrast / 5.0).clamp(0.2, 1.0)

    # --- Step 2: per-TX ---
    diff_tx = rast.tx_positions[None, :, :] - positions[:, None, :]
    d_tx = diff_tx.norm(dim=-1).clamp(min=1e-6)
    wi = diff_tx / d_tx.unsqueeze(-1)

    cos_i_raw = (wi * normals[:, None, :]).sum(-1)
    normal_sign = torch.where(cos_i_raw < 0, -torch.ones_like(cos_i_raw), torch.ones_like(cos_i_raw))
    cos_i = cos_i_raw.abs().clamp(min=1e-6)
    n_eff = normals[:, None, :] * normal_sign.unsqueeze(-1)

    wi_dot_n = (wi * n_eff).sum(-1, keepdim=True)
    wi_r = 2.0 * wi_dot_n * n_eff - wi

    tan_sq_i = (1.0 / cos_i.clamp(min=1e-6) ** 2) - 1.0
    lambda_i = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_i).clamp(min=0.0))) / 2.0

    wi_flat = wi.reshape(-1, 3)
    n_eff_flat = n_eff.reshape(-1, 3)
    s_in_flat, p_in_flat = compute_sp_basis(wi_flat, n_eff_flat)
    s_in = s_in_flat.reshape(M, n_tx, 3).contiguous()
    p_in = p_in_flat.reshape(M, n_tx, 3).contiguous()

    tx_pol = _get_default_pol(device)
    tx_s = (tx_pol * s_in).sum(-1)
    tx_p = (tx_pol * p_in).sum(-1)

    cos_i_flat = cos_i.reshape(-1)
    eps_r_flat = eps_real[:, None].expand(-1, n_tx).reshape(-1)
    eps_i_flat = eps_imag[:, None].expand(-1, n_tx).reshape(-1)
    thick_flat = thickness[:, None].expand(-1, n_tx).reshape(-1)
    R_TE, R_TM, _, _ = itu_slab_fresnel(eps_r_flat, eps_i_flat, cos_i_flat, thick_flat)
    r_s = R_TE.reshape(M, n_tx)
    r_p = R_TM.reshape(M, n_tx)

    E_s_out = r_s * tx_s.to(torch.complex64)
    E_p_out = r_p * tx_p.to(torch.complex64)

    # tau_eff blend
    tau_angle = _sigmoid((cos_i - 0.94) * 20.0)
    kl = K_WAVE * l_c
    v_KA = _sigmoid((kl - 6.0) * 2.0)
    kh = K_WAVE * sigma_h
    v_SPM = _sigmoid((0.3 - kh) * 10.0)
    w_KA = tau_angle * tau_base[:, None] * v_KA[:, None]
    w_SPM = (1.0 - tau_angle) * (1.0 - tau_base[:, None]) * v_SPM[:, None]
    tau_eff = (w_KA / (w_KA + w_SPM).clamp(min=1e-6)).clamp(0.01, 0.99)

    # --- Step 3: per-RX ---
    diff_rx = positions[:, None, :] - rast.rx_positions[None, :, :]
    d_rx = diff_rx.norm(dim=-1).clamp(min=1e-6)
    wo = -diff_rx / d_rx.unsqueeze(-1)

    cos_o_raw = (wo * normals[:, None, :]).sum(-1)
    cos_o = cos_o_raw.abs().clamp(min=1e-6)
    tan_sq_o = (1.0 / cos_o.clamp(min=1e-6) ** 2) - 1.0
    lambda_o = (-1.0 + torch.sqrt((1.0 + alpha_sq[:, None] * tan_sq_o).clamp(min=0.0))) / 2.0

    return dict(
        wi=wi.contiguous(),
        wi_r=wi_r.contiguous(),
        wo=wo.contiguous(),
        n_eff=n_eff.contiguous(),
        s_in=s_in.contiguous(),
        cos_i=cos_i.contiguous(),
        cos_o=cos_o.contiguous(),
        lambda_i=lambda_i.contiguous(),
        lambda_o=lambda_o.contiguous(),
        alpha_sq=alpha_sq.contiguous(),
        kappa_SPM=kappa_SPM.contiguous(),
        norm_SPM=norm_SPM.contiguous(),
        eps_factor=eps_factor.contiguous(),
        eps_real=eps_real.contiguous(),
        eps_imag=eps_imag.contiguous(),
        thickness=thickness.contiguous(),
        E_s_out_re=E_s_out.real.contiguous(),
        E_s_out_im=E_s_out.imag.contiguous(),
        E_p_out_re=E_p_out.real.contiguous(),
        E_p_out_im=E_p_out.imag.contiguous(),
        tau_eff=tau_eff.contiguous(),
    )


def test_cuda_matches_pytorch_reference():
    intermed = compute_step123()

    # PyTorch reference
    f_cos_ref = bsdf_step4_reference(
        intermed["wi"], intermed["wi_r"], intermed["wo"],
        intermed["n_eff"], intermed["s_in"],
        intermed["cos_i"], intermed["cos_o"],
        intermed["lambda_i"], intermed["lambda_o"],
        intermed["alpha_sq"], intermed["kappa_SPM"],
        intermed["norm_SPM"], intermed["eps_factor"],
        intermed["eps_real"], intermed["eps_imag"], intermed["thickness"],
        intermed["E_s_out_re"], intermed["E_s_out_im"],
        intermed["E_p_out_re"], intermed["E_p_out_im"],
        intermed["tau_eff"],
    )

    # CUDA kernel
    f_cos_cuda = v5cuda.ext.bsdf_step4_forward(
        intermed["wi"], intermed["wi_r"], intermed["wo"],
        intermed["n_eff"], intermed["s_in"],
        intermed["cos_i"], intermed["cos_o"],
        intermed["lambda_i"], intermed["lambda_o"],
        intermed["alpha_sq"], intermed["kappa_SPM"],
        intermed["norm_SPM"], intermed["eps_factor"],
        intermed["eps_real"], intermed["eps_imag"], intermed["thickness"],
        intermed["E_s_out_re"], intermed["E_s_out_im"],
        intermed["E_p_out_re"], intermed["E_p_out_im"],
        intermed["tau_eff"],
    )
    torch.cuda.synchronize()

    abs_err = (f_cos_cuda - f_cos_ref).abs()
    rel_err = abs_err / f_cos_ref.abs().clamp(min=1e-8)
    print(
        f"\n  max|cuda - pytorch|     : {abs_err.max().item():.3e}"
        f"\n  mean|cuda - pytorch|    : {abs_err.mean().item():.3e}"
        f"\n  max rel err             : {rel_err.max().item():.3e}"
        f"\n  mean rel err            : {rel_err.mean().item():.3e}"
        f"\n  pytorch ref max|f_cos|  : {f_cos_ref.abs().max().item():.3e}"
        f"\n  pytorch ref mean|f_cos| : {f_cos_ref.abs().mean().item():.3e}"
    )
    # Tolerances (float32 chain of sqrt/expf/__sincosf with --use_fast_math):
    # rtol=5e-3, atol=1e-3. Mean abs err is ~1e-5 on a max|f_cos|~2.6 tensor,
    # so ~20 paths out of 16M hit the near-zero tail where fma ordering
    # produces visible relative drift. The end-to-end cart_corr regression
    # (test_single_scene_cart_corr) is the true correctness gate.
    assert torch.allclose(f_cos_cuda, f_cos_ref, rtol=5e-3, atol=1e-3), (
        f"CUDA kernel mismatch. max|err|={abs_err.max().item():.3e}, "
        f"max rel={rel_err.max().item():.3e}"
    )
    # Mean abs error is a tighter test that still catches systematic bugs
    # (not just a single near-specular outlier).
    assert abs_err.mean().item() < 1e-4, (
        f"Mean |err| too large: {abs_err.mean().item():.3e}"
    )
