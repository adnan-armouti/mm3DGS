"""Analytical ray-surfel intersection for multi-bounce tracing.

Each surfel is a 2D Gaussian disc defined by (centre, normal, t1, t2, scales).
Ray-disc intersection is:
  1. Intersect ray with the surfel plane.
  2. Project hit point to surfel-local (u, v).
  3. Evaluate 2D Gaussian weight.
  4. Effective opacity = surfel opacity * Gaussian weight.

Brute-force with chunking over surfels to bound memory.
"""

import torch
from torch import Tensor
import math


def intersect_rays_surfels(
    ray_origins: Tensor,       # (N_rays, 3)
    ray_dirs: Tensor,          # (N_rays, 3) unit vectors
    surfel_centers: Tensor,    # (N_s, 3)
    surfel_normals: Tensor,    # (N_s, 3)
    surfel_t1: Tensor,         # (N_s, 3)
    surfel_t2: Tensor,         # (N_s, 3)
    surfel_scales: Tensor,     # (N_s, 2)  positive (already exp'd)
    surfel_opacities: Tensor,  # (N_s,) in [0, 1]
    sigma_cutoff: float = 3.0,
    surfel_chunk: int = 4096,
) -> tuple:
    """Brute-force nearest-hit ray-surfel intersection.

    Returns:
        hit_mask:  (N_rays,) bool
        t_hit:     (N_rays,) distance to hit
        hit_idx:   (N_rays,) surfel index (-1 if miss)
        hit_alpha: (N_rays,) opacity at hit
        hit_uv:    (N_rays, 2) local coordinates
    """
    N_rays = ray_origins.shape[0]
    N_s = surfel_centers.shape[0]
    device = ray_origins.device

    best_t = torch.full((N_rays,), float("inf"), device=device)
    best_idx = torch.full((N_rays,), -1, dtype=torch.long, device=device)
    best_alpha = torch.zeros(N_rays, device=device)
    best_uv = torch.zeros(N_rays, 2, device=device)

    gauss_cutoff = math.exp(-0.5 * sigma_cutoff ** 2)
    batch_arange = torch.arange(N_rays, device=device)

    for s0 in range(0, N_s, surfel_chunk):
        s1 = min(s0 + surfel_chunk, N_s)
        S = s1 - s0

        mu = surfel_centers[s0:s1]       # (S, 3)
        n = surfel_normals[s0:s1]        # (S, 3)
        t1 = surfel_t1[s0:s1]
        t2 = surfel_t2[s0:s1]
        sc = surfel_scales[s0:s1]        # (S, 2)
        op = surfel_opacities[s0:s1]     # (S,)

        # (N_rays, S, 3)
        delta = mu.unsqueeze(0) - ray_origins.unsqueeze(1)
        d_dot_n = (ray_dirs.unsqueeze(1) * n.unsqueeze(0)).sum(-1)          # (N_rays, S)
        delta_dot_n = (delta * n.unsqueeze(0)).sum(-1)

        t = delta_dot_n / (d_dot_n + 1e-12)                                 # (N_rays, S)

        # Intersection point
        p_hit = ray_origins.unsqueeze(1) + t.unsqueeze(-1) * ray_dirs.unsqueeze(1)
        local = p_hit - mu.unsqueeze(0)

        u = (local * t1.unsqueeze(0)).sum(-1)
        v = (local * t2.unsqueeze(0)).sum(-1)

        u_norm = u / (sc[:, 0].unsqueeze(0) + 1e-12)
        v_norm = v / (sc[:, 1].unsqueeze(0) + 1e-12)
        w = torch.exp(-0.5 * (u_norm ** 2 + v_norm ** 2))

        alpha = op.unsqueeze(0) * w

        valid = (t > 1e-4) & (w > gauss_cutoff) & (alpha > 0.01)
        t_masked = torch.where(valid, t, torch.tensor(float("inf"), device=device))

        chunk_best_t, chunk_best_local = t_masked.min(dim=1)
        improved = chunk_best_t < best_t

        chunk_global_idx = chunk_best_local + s0
        c_alpha = alpha[batch_arange, chunk_best_local]
        c_u = u[batch_arange, chunk_best_local]
        c_v = v[batch_arange, chunk_best_local]

        best_t = torch.where(improved, chunk_best_t, best_t)
        best_idx = torch.where(improved, chunk_global_idx, best_idx)
        best_alpha = torch.where(improved, c_alpha, best_alpha)
        best_uv[:, 0] = torch.where(improved, c_u, best_uv[:, 0])
        best_uv[:, 1] = torch.where(improved, c_v, best_uv[:, 1])

    hit_mask = best_idx >= 0
    return hit_mask, best_t, best_idx, best_alpha, best_uv
