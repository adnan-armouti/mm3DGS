"""Per-Gaussian exact-phase ADC synthesis (single bounce).

Pure PyTorch — no DrJit, no Mitsuba.  Full autograd through all parameters
including materials (which the old DrJit wrapper could not provide).

Amplitude convention (matching mmIR synthesis_e2e.py L498-503):
    field_amplitude = sqrt(BSDF_power × antenna_gain × path_loss)

Uses gradient checkpointing on the phasor loop so cos/sin are recomputed
during backward (~5 KB/Gaussian instead of ~800 KB).
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import RadarConfig, C
from .gaussian_model import GaussianModel
from .reparameterization import reparameterize
from .bsdf_torch import evaluate_bsdf_tier1, evaluate_bsdf_jones_f_cos, KAPPA_77GHZ
from .antenna_torch import evaluate_tx_gain, evaluate_rx_gain


def synthesize_adc_single_bounce(
    model: GaussianModel,
    active_mask: Tensor,       # (N,) bool
    radar_cfg: RadarConfig,
    detach_phase: bool = True,
    chunk_size: int = 512,
    enable_gamma: bool = False,
    shading_tier: int = 2,
) -> tuple:
    """Render ADC from active Gaussians (single bounce, exact phase).

    Returns:
        (adc_real, adc_imag) each (N_tx, N_rx, K) float32.
    """
    device = model.device
    K = radar_cfg.num_adc_samples
    N_tx = radar_cfg.n_tx
    N_rx = radar_cfg.n_rx
    f0 = radar_cfg.center_freq
    S = radar_cfg.chirp_slope
    sr = radar_cfg.sample_rate

    # --- Active Gaussian parameters ---
    mu = model.positions[active_mask]                          # (M, 3)
    normals = model.get_normals()[active_mask]                  # (M, 3)
    t1_vecs, t2_vecs, _ = model.get_tangent_frame()
    t1_vecs = t1_vecs[active_mask]                              # (M, 3)
    t2_vecs = t2_vecs[active_mask]                              # (M, 3)
    opacities = model.get_opacities()[active_mask].squeeze(-1)  # (M,)
    raw_mat = model.raw_materials[active_mask]                  # (M, 6)
    physics_mat = reparameterize(raw_mat)                       # (M, 6) — autograd connected!
    M = mu.shape[0]

    if M == 0:
        return (torch.zeros(N_tx, N_rx, K, device=device),
                torch.zeros(N_tx, N_rx, K, device=device))

    # --- Sensor geometry (metres) ---
    tx_pos = torch.from_numpy(radar_cfg.tx_positions_m).float().to(device)   # (N_tx, 3)
    rx_pos = torch.from_numpy(radar_cfg.rx_positions_m).float().to(device)   # (N_rx, 3)
    tx_bore = torch.from_numpy(radar_cfg.tx_boresights).float().to(device)   # (N_tx, 3)
    rx_bore = torch.from_numpy(radar_cfg.rx_boresights).float().to(device)   # (N_rx, 3)

    # Time grid: t[k] = k / sample_rate (mmIR convention)
    t_grid = torch.arange(K, device=device, dtype=torch.float32) / sr

    # --- Distances (M, N_tx) and (M, N_rx) ---
    diff_tx = mu.unsqueeze(1) - tx_pos.unsqueeze(0)             # (M, N_tx, 3)
    diff_rx = mu.unsqueeze(1) - rx_pos.unsqueeze(0)             # (M, N_rx, 3)
    d_tx = diff_tx.norm(dim=-1)                                  # (M, N_tx)
    d_rx = diff_rx.norm(dim=-1)                                  # (M, N_rx)

    # --- Direction unit vectors ---
    dir_tx = diff_tx / d_tx.unsqueeze(-1).clamp(min=1e-6)       # (M, N_tx, 3)  TX→μ direction
    dir_rx = diff_rx / d_rx.unsqueeze(-1).clamp(min=1e-6)       # (M, N_rx, 3)  RX→μ direction

    # --- BSDF evaluation ---
    # Evaluate per-Gaussian using the MEAN TX-RX direction (monostatic approx
    # for the BSDF), then the per-TX-RX variation comes from antenna gains and
    # path loss.  This keeps cost at O(M) for BSDF, not O(M × N_tx × N_rx).
    radar_center = (tx_pos.mean(dim=0) + rx_pos.mean(dim=0)) / 2
    wi_avg = F.normalize(radar_center - mu, dim=-1)              # (M, 3) toward radar
    wo_avg = wi_avg                                               # monostatic approx

    cos_theta_avg = torch.abs((wi_avg * normals).sum(-1)).clamp(min=1e-6)  # (M,)

    if shading_tier >= 2:
        # Full Jones KA+SPM BSDF (uses all 6 ITU params)
        f_cos_power_1d = evaluate_bsdf_jones_f_cos(
            cos_theta_avg,
            wo_avg, wi_avg, normals,
            physics_mat[:, 0], physics_mat[:, 1],  # eps_real, eps_imag
            physics_mat[:, 2], physics_mat[:, 3],  # sigma_h, l_c
            physics_mat[:, 4], physics_mat[:, 5],  # tau, thickness
        )  # (M,) — full autograd to all 6 material params
    else:
        # Tier 1: Fresnel only (4 of 6 params)
        f_cos_power_1d = evaluate_bsdf_tier1(
            cos_theta_avg,
            physics_mat[:, 0], physics_mat[:, 1],
            physics_mat[:, 2], physics_mat[:, 5],
        )  # (M,)

    # Broadcast to (M, N_tx, N_rx) — BSDF is view-independent at this level,
    # per-TX-RX variation comes from antenna gains and path loss below
    f_cos_power = f_cos_power_1d.unsqueeze(-1).unsqueeze(-1)     # (M, 1, 1)

    # --- Antenna gains (native PyTorch) ---
    G_tx = evaluate_tx_gain(
        dir_tx.reshape(-1, 3),
        tx_bore.unsqueeze(0).expand(M, -1, -1).reshape(-1, 3),
    ).reshape(M, N_tx)                                            # (M, N_tx)

    G_rx = evaluate_rx_gain(
        dir_rx.reshape(-1, 3),
        rx_bore.unsqueeze(0).expand(M, -1, -1).reshape(-1, 3),
    ).reshape(M, N_rx)                                            # (M, N_rx)

    # --- Path loss: 1/d_tx² (matching mmIR synthesis_e2e.py L499-500) ---
    # mmIR uses only TX distance in explicit path loss; the RX geometry is
    # implicitly handled by the MC reservoir sampling PDF.  For our direct
    # evaluator, matching mmIR's convention gives the correct range weighting
    # for comparison against measured GT ADC.
    path_loss = 1.0 / (
        d_tx.unsqueeze(-1) ** 2 + 1e-20
    )                                                             # (M, N_tx, 1) broadcasts to (M, N_tx, N_rx)

    # --- Coherence factor γ (optional) ---
    if enable_gamma:
        sigma_em = model.get_sigma_em()[active_mask]              # (M, 1)
        d_perp_sq = _compute_bistatic_tangent_proj(
            normals, t1_vecs, t2_vecs, dir_tx, dir_rx
        )                                                         # (M, N_tx, N_rx)
        gamma = torch.exp(
            -0.5 * KAPPA_77GHZ ** 2
            * sigma_em.unsqueeze(-1) ** 2
            * d_perp_sq
        )                                                         # (M, N_tx, N_rx)
    else:
        gamma = 1.0

    # --- Field amplitude = sqrt(power product) ---
    power_product = (
        f_cos_power * gamma
        * G_tx.unsqueeze(-1) * G_rx.unsqueeze(-2)
        * path_loss
    )                                                             # (M, N_tx, N_rx)

    A = opacities.unsqueeze(-1).unsqueeze(-1) * torch.sqrt(
        power_product.clamp(min=1e-20)
    )                                                             # (M, N_tx, N_rx)

    # --- Total path length ---
    R_tot = d_tx.unsqueeze(-1) + d_rx.unsqueeze(-2)              # (M, N_tx, N_rx)
    tau = R_tot / C                                               # (M, N_tx, N_rx)

    # --- Chunked phasor scatter-add with gradient checkpointing ---
    adc_real = torch.zeros(N_tx, N_rx, K, device=device)
    adc_imag = torch.zeros(N_tx, N_rx, K, device=device)

    two_pi = 2.0 * math.pi

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        cr, ci = checkpoint(
            _phasor_chunk, A[start:end], tau[start:end], t_grid,
            f0, S, two_pi, detach_phase,
            use_reentrant=False,
        )
        adc_real = adc_real + cr
        adc_imag = adc_imag + ci

    return adc_real, adc_imag


# -----------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------

def _phasor_chunk(A_chunk, tau_chunk, t_grid, f0, S, two_pi, detach_phase):
    """One chunk's phasor contribution (wrapped by torch.checkpoint)."""
    phi_const = (two_pi * f0 * tau_chunk).unsqueeze(-1)
    phi_slope = (two_pi * S * tau_chunk).unsqueeze(-1)

    if detach_phase:
        phi_const = phi_const.detach()
        phi_slope = phi_slope.detach()

    phi = phi_const + phi_slope * t_grid
    A_c = A_chunk.unsqueeze(-1)

    return (A_c * torch.cos(phi)).sum(dim=0), (A_c * torch.sin(phi)).sum(dim=0)


def _compute_bistatic_tangent_proj(normals, t1, t2, dir_tx, dir_rx):
    """||d_ij,⊥||² = (t1·d_ij)² + (t2·d_ij)² for coherence factor γ.

    Args:
        normals: (M, 3), t1: (M, 3), t2: (M, 3)
        dir_tx: (M, N_tx, 3), dir_rx: (M, N_rx, 3)

    Returns:
        (M, N_tx, N_rx)
    """
    # d_ij = dir_tx + dir_rx  (bistatic direction sum)
    d_ij = dir_tx.unsqueeze(2) + dir_rx.unsqueeze(1)    # (M, N_tx, N_rx, 3)

    # Project onto tangent plane
    proj_t1 = (t1.unsqueeze(1).unsqueeze(2) * d_ij).sum(-1)  # (M, N_tx, N_rx)
    proj_t2 = (t2.unsqueeze(1).unsqueeze(2) * d_ij).sum(-1)

    return proj_t1 ** 2 + proj_t2 ** 2
