"""v6 radar training losses.

All variants operate on complex per-virtual range-profile tensors
``(..., N, R)`` where ``N=192`` virtuals and ``R=256`` range bins.
Each returns a scalar loss; all implement a consistent per-frame
normalisation (divide by a GT-dependent constant, so loss ~ O(1) at
init regardless of scene-dependent power).

Variants:

  * ``gram_frobenius_loss`` — baseline. Full rank-1 Gram-matrix
    Frobenius² distance summed over range bins. Supervises per-antenna
    magnitudes (diagonal) + pairwise relative phases (off-diagonal).

  * ``coarray_loss`` — Tier 1. Aggregate pairwise products by baseline
    vector (co-array), supervise Frob² on the aggregated co-array.
    Pose-invariant by construction on the parallelogram group.

  * ``smooth_alpha_gram_loss`` — Tier 1. Frob² with per-range inner
    product smoothed across the range axis before the
    magnitude-squared term. Penalises non-smooth per-range phase.

  * ``mag_weighted_gram_loss`` — Tier 2. Weighted Gram with pair weights
    ``|v_g[i]|·|v_g[j]|`` (stronger emphasis on high-SNR pairs).

  * ``baseline_weighted_gram_loss`` — Tier 2. Weighted Gram with pair
    weights ``exp(-||pos_i - pos_j||² / σ²)`` (short baselines weighted
    more — they are pose-robust).

  * ``inv_variance_gram_loss`` — Tier 2. Weighted Gram with Cauchy
    kernel ``L² / (L² + ||Δ||²)`` — closer to optimal inverse-variance
    weighting under linear pose-noise model.

  * ``diag_offdiag_gram_loss`` — Tier 5. Gram Frob² with separate
    weights ``λ_d`` (diagonal = per-antenna magnitude) and ``λ_off``
    (off-diagonal = pairwise phase+mag). ``λ_off=0`` ⇒ pure per-antenna
    magnitude matching on all 192 virtuals.

  * ``range_integrated_gram_loss`` — Tier 5. Frob² distance between
    the RANGE-INTEGRATED Grams ``V V^H ∈ ℂ^{N×N}`` (sum over range bins
    before outer product). Invariant under ANY per-range α(r) — fully
    pose-invariant in that dimension. Supervises cross-range
    consistency instead of per-range detail.

  * ``modulus_gram_loss`` — Tier 5. Frob² distance between the
    ELEMENTWISE-MODULUS of the per-range Grams. Drops all pairwise
    phase; supervises only per-antenna magnitudes and pair-magnitude
    products. Strictly pose-robust (no phase at all).

Diagnostic-only utility ``normalised_gram_correlation`` retained from
the earlier M1 work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


__all__ = [
    "gram_frobenius_loss",
    "coarray_loss",
    "smooth_alpha_gram_loss",
    "mag_weighted_gram_loss",
    "baseline_weighted_gram_loss",
    "inv_variance_gram_loss",
    "diag_offdiag_gram_loss",
    "range_integrated_gram_loss",
    "modulus_gram_loss",
    "baseline_binned_loss",
    "normalised_gram_correlation",
]


# ---------------------------------------------------------------------------
# Baseline: full Frobenius² (already validated)
# ---------------------------------------------------------------------------

def gram_frobenius_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Un-normalised Frob² Gram distance summed over range bins.

    Per-bin: ``L_r = ||v_p||^4 + ||v_g||^4 - 2|v_p^H v_g|^2`` via Schur
    identity (O(N·R) per frame, no N² expansion).
    """
    _check(v_pred, v_gt)
    inner = (v_pred.conj() * v_gt).sum(dim=-2)                 # (..., R) cx
    inner_mag_sq = inner.real.pow(2) + inner.imag.pow(2)
    norm_p_sq = (v_pred.real.pow(2) + v_pred.imag.pow(2)).sum(dim=-2)
    norm_g_sq = (v_gt.real.pow(2)   + v_gt.imag.pow(2)  ).sum(dim=-2)
    L = norm_p_sq.pow(2) + norm_g_sq.pow(2) - 2.0 * inner_mag_sq
    L_sum = L.sum()
    if normalize:
        denom = norm_g_sq.pow(2).sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


# ---------------------------------------------------------------------------
# Tier 1: co-array aggregation
# ---------------------------------------------------------------------------

def coarray_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    baseline_idx: torch.Tensor,
    n_baselines: int,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Co-array aggregated Gram loss.

    For each unique baseline vector Δ, aggregate
    ``A[Δ, r] = Σ_{(i,j) : pos_j-pos_i=Δ, i<=j} v[i,r] · v[j,r]*``
    (complex). Loss is ``Σ_{Δ, r} |A_p - A_g|²``, normalised by
    ``Σ_{Δ, r} |A_g|²`` when ``normalize=True``.

    Physical intuition: pairs with the same baseline share the same
    first-order pose-phase response, so coherent summation within each
    baseline class noise-averages pose jitter while preserving the
    scene-signal.
    """
    _check(v_pred, v_gt)
    # Per-pair products: (..., P, R) complex.
    prods_p = v_pred.index_select(-2, pair_i) * v_pred.index_select(-2, pair_j).conj()
    prods_g = v_gt.index_select(-2, pair_i)   * v_gt.index_select(-2, pair_j).conj()

    A_p = _scatter_complex_dim(prods_p, baseline_idx, n_baselines)  # (..., n_bl, R)
    A_g = _scatter_complex_dim(prods_g, baseline_idx, n_baselines)

    diff = A_p - A_g
    L_num = (diff.real.pow(2) + diff.imag.pow(2)).sum()
    if normalize:
        denom = (A_g.real.pow(2) + A_g.imag.pow(2)).sum().clamp_min(eps)
        return L_num / denom
    return L_num


def _scatter_complex_dim(src: torch.Tensor, index: torch.Tensor, n_out: int) -> torch.Tensor:
    """Scatter-add ``src`` (..., P, R) complex along dim=-2 using
    ``index`` (P,) → returns (..., n_out, R) complex.
    """
    leading = src.shape[:-2]
    P = src.shape[-2]
    R = src.shape[-1]
    src_flat = src.reshape(-1, P, R)
    B = src_flat.shape[0]
    out_real = torch.zeros(B, n_out, R, device=src.device, dtype=src.real.dtype)
    out_imag = torch.zeros(B, n_out, R, device=src.device, dtype=src.real.dtype)
    idx = index.unsqueeze(0).unsqueeze(-1).expand(B, P, R)
    out_real.scatter_add_(1, idx, src_flat.real)
    out_imag.scatter_add_(1, idx, src_flat.imag)
    out = torch.complex(out_real, out_imag).reshape(*leading, n_out, R)
    return out


# ---------------------------------------------------------------------------
# Tier 1: smooth-α per-range phase invariance
# ---------------------------------------------------------------------------

def smooth_alpha_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    smooth_k: int = 11,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Frob² Gram loss with the per-range inner product complex-smoothed
    over the range axis before the |·|² step.

    Mechanism: replacing ``|v_p^H v_g|²(r)`` with
    ``|smooth(v_p^H v_g)(r)|²`` penalises configurations where the free
    per-range α(r) would have to vary rapidly to achieve a perfect
    fit. Pose-translation noise induces a SMOOTH α(r) across range
    (scatterer-direction varies smoothly with r), so constraining α
    smooth re-captures training signal that vanilla per-bin Frob²
    leaves invariant.
    """
    _check(v_pred, v_gt)
    inner = (v_pred.conj() * v_gt).sum(dim=-2)                     # (..., R) cx
    inner_re_smooth = _box_smooth_1d(inner.real, smooth_k)
    inner_im_smooth = _box_smooth_1d(inner.imag, smooth_k)
    inner_smooth_mag_sq = inner_re_smooth.pow(2) + inner_im_smooth.pow(2)

    norm_p_sq = (v_pred.real.pow(2) + v_pred.imag.pow(2)).sum(dim=-2)
    norm_g_sq = (v_gt.real.pow(2)   + v_gt.imag.pow(2)  ).sum(dim=-2)
    L = norm_p_sq.pow(2) + norm_g_sq.pow(2) - 2.0 * inner_smooth_mag_sq
    L_sum = L.sum()
    if normalize:
        denom = norm_g_sq.pow(2).sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


def _box_smooth_1d(x: torch.Tensor, k: int) -> torch.Tensor:
    """Moving-average over the last axis with kernel size ``k``, 'same'
    padding (zero-fill at edges, then compensate by actual kernel size)."""
    if k <= 1:
        return x
    orig_shape = x.shape
    x_flat = x.reshape(-1, 1, orig_shape[-1])
    pad = (k - 1) // 2
    # avg_pool1d uses count_include_pad=True by default; switching to
    # False so that boundaries aren't biased toward zero.
    y = F.avg_pool1d(x_flat, kernel_size=k, stride=1, padding=pad,
                      count_include_pad=False)
    return y.reshape(*orig_shape[:-1], y.shape[-1])


# ---------------------------------------------------------------------------
# Tier 2: weighted Gram family (all use the same W @ X primitive)
# ---------------------------------------------------------------------------

def _weighted_gram_via_W(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    W: torch.Tensor,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    """Generic weighted Frob² Gram loss with a real PSD kernel ``W``
    of shape ``(N, N)``. O(N² · R) per frame via three ``W @ X`` matmuls.

    L_r = m_p^T W m_p + m_g^T W m_g - 2·Re(a^H W a)
    where m = |v|² (real) and a = v_p · v_g.conj() (complex).
    With ``W = 11^T`` (all-ones, rank-1) this reduces to the vanilla
    Frob² loss up to the exact Schur identity.
    """
    m_p = v_pred.real.pow(2) + v_pred.imag.pow(2)           # (..., N, R) real
    m_g = v_gt.real.pow(2)   + v_gt.imag.pow(2)
    a   = v_pred * v_gt.conj()                              # (..., N, R) cx

    W_mp = W @ m_p                                          # (..., N, R) real
    W_mg = W @ m_g
    W_a  = W.to(a.dtype) @ a                                # (..., N, R) cx

    mp_W_mp = (m_p * W_mp).sum(dim=-2)                      # (..., R) real
    mg_W_mg = (m_g * W_mg).sum(dim=-2)
    a_W_a   = (a.conj() * W_a).sum(dim=-2).real             # (..., R) real

    L = mp_W_mp + mg_W_mg - 2.0 * a_W_a                     # (..., R) real
    L_sum = L.sum()
    if normalize:
        denom = mg_W_mg.sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


def mag_weighted_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Magnitude-weighted Gram: multiply v_pred / v_gt by per-antenna
    GT magnitude before Frob². Effective pair weight
    ``(|v_g[i]|·|v_g[j]|)²`` — signal-dominant antennas dominate.
    """
    _check(v_pred, v_gt)
    # Per-antenna per-range GT magnitude.
    m_g_sqrt = (v_gt.real.pow(2) + v_gt.imag.pow(2)).sqrt()  # (..., N, R)
    # Time-invariant weight: use max over range as a per-antenna scalar.
    # (Using per-range weight would tangle the Schur identity.)
    w_ant = m_g_sqrt.amax(dim=-1, keepdim=True).detach()     # (..., N, 1)
    w_ant = w_ant / w_ant.amax(dim=-2, keepdim=True).clamp_min(eps)
    v_p_w = v_pred * w_ant
    v_g_w = v_gt * w_ant
    return gram_frobenius_loss(v_p_w, v_g_w, normalize=normalize, eps=eps)


def baseline_weighted_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    virt_positions: torch.Tensor,
    sigma: float = 30.0,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Gaussian-baseline-weighted Gram. Pair weight
    ``W[i,j] = exp(-||pos_i - pos_j||² / σ²)``. PSD. Short baselines
    (pose-robust pairs) are weighted more.
    """
    _check(v_pred, v_gt)
    W = _gaussian_baseline_kernel(virt_positions, sigma).to(
        v_pred.device).to(v_pred.real.dtype)
    return _weighted_gram_via_W(v_pred, v_gt, W, normalize, eps)


def inv_variance_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    virt_positions: torch.Tensor,
    L_scale: float = 30.0,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Cauchy-baseline-weighted Gram (inverse-variance-style). Pair
    weight ``W[i,j] = L² / (L² + ||Δ||²)`` where Δ is the baseline
    vector. Closer to optimal inverse-variance weighting under a
    linear pose-noise model (noise variance ∝ ||Δ||²).
    """
    _check(v_pred, v_gt)
    W = _cauchy_baseline_kernel(virt_positions, L_scale).to(
        v_pred.device).to(v_pred.real.dtype)
    return _weighted_gram_via_W(v_pred, v_gt, W, normalize, eps)


def _gaussian_baseline_kernel(positions: torch.Tensor, sigma: float) -> torch.Tensor:
    """(N, N) real Gaussian kernel of baseline magnitudes."""
    pos = positions.to(torch.float32)
    d = pos.unsqueeze(0) - pos.unsqueeze(1)                    # (N, N, 2)
    d2 = d.pow(2).sum(dim=-1)
    return torch.exp(-d2 / (sigma ** 2))


def _cauchy_baseline_kernel(positions: torch.Tensor, L: float) -> torch.Tensor:
    """(N, N) real Cauchy kernel: L² / (L² + ||Δ||²)."""
    pos = positions.to(torch.float32)
    d = pos.unsqueeze(0) - pos.unsqueeze(1)
    d2 = d.pow(2).sum(dim=-1)
    L2 = float(L) ** 2
    return L2 / (L2 + d2)


# ---------------------------------------------------------------------------
# Tier 5: algebraic / decomposition
# ---------------------------------------------------------------------------

def diag_offdiag_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    lambda_diag: float = 1.0,
    lambda_off: float = 0.0,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Split Gram Frob² into diagonal and off-diagonal components,
    weight independently.

    diag_term(r)    = Σ_i (|v_p[i]|² - |v_g[i]|²)²
    off_diag_term(r) = (Frob² - diag_term) / 2
    L_r = λ_d · diag_term(r) + 2·λ_off · off_diag_term(r)

    (λ_d, λ_off) = (1, 1) recovers vanilla Frob². (λ_d, λ_off) = (1, 0)
    is pure per-antenna magnitude MSE (fully pose-robust).
    """
    _check(v_pred, v_gt)
    m_p = v_pred.real.pow(2) + v_pred.imag.pow(2)                  # (..., N, R)
    m_g = v_gt.real.pow(2)   + v_gt.imag.pow(2)

    diag_term = (m_p - m_g).pow(2).sum(dim=-2)                     # (..., R)

    # Frob² (Schur)
    inner = (v_pred.conj() * v_gt).sum(dim=-2)
    inner_mag_sq = inner.real.pow(2) + inner.imag.pow(2)
    norm_p_sq = m_p.sum(dim=-2)
    norm_g_sq = m_g.sum(dim=-2)
    frob_sq = norm_p_sq.pow(2) + norm_g_sq.pow(2) - 2.0 * inner_mag_sq

    off_twice = (frob_sq - diag_term).clamp_min(0.0)               # = 2·off_tri

    L = lambda_diag * diag_term + lambda_off * off_twice
    L_sum = L.sum()
    if normalize:
        denom = norm_g_sq.pow(2).sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


def range_integrated_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Frob² between range-integrated Grams
    ``G = V V^H ∈ ℂ^{N×N}`` where V=(N, R).

    ``L = ||V_p V_p^H - V_g V_g^H||_F²
        = ||V_p^H V_p||_F² + ||V_g^H V_g||_F² - 2·||V_p^H V_g||_F²``

    Invariant under arbitrary per-range phase α(r) (summing v(r) v(r)^H
    over r cancels r-dependent phases). Fully pose-invariant in the
    per-range-phase sense.
    """
    _check(v_pred, v_gt)
    # V_p^H V_p and V_g^H V_g and V_p^H V_g, all (R, R).
    Vp_H = v_pred.conj().transpose(-1, -2)                         # (..., R, N)
    Vg_H = v_gt.conj().transpose(-1, -2)
    A_pp = Vp_H @ v_pred                                           # (..., R, R)
    A_gg = Vg_H @ v_gt
    A_pg = Vp_H @ v_gt

    def frob_sq(M):
        return (M.real.pow(2) + M.imag.pow(2)).sum(dim=(-1, -2))

    L = frob_sq(A_pp) + frob_sq(A_gg) - 2.0 * frob_sq(A_pg)         # (...,) real
    L_sum = L.sum()
    if normalize:
        denom = frob_sq(A_gg).sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


def baseline_binned_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    bin_idx: torch.Tensor,
    multiplicity: torch.Tensor,
    baseline_weights: torch.Tensor | None = None,
    skip_b0: bool = False,
    b0_bin: int | None = None,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Baseline-binned correlation loss (per md/baseline_binned_loss_design.md).

    For each unique baseline vector Δ = (pos_j - pos_i) in the MMWCAS
    2D sparse co-array, aggregate all pairs with that baseline into a
    **mean** (not sum) complex value per range bin:

        p[Δ, r] = (1/M(Δ)) · Σ_{(i,j) : b_ij = Δ} v[i, r] · conj(v[j, r])

    Loss:

        L = Σ_{Δ, r}  w(Δ) · | p_pred[Δ, r] - p_gt[Δ, r] |²

    The mean-over-pairs (instead of sum) is the key difference vs
    ``coarray_loss`` above: it collapses the ~18k pair contributions
    onto the ~1k physically-distinct baselines without over-enumerating
    the long-baseline (pose-noisy) pairs. Same "sum-then-penalise"
    structure v5's 86-point FFT uses implicitly.

    ``baseline_weights``: optional ``(n_baselines,)`` real weights w(Δ).
    Design-doc Wiener form: ``w(|Δ|) = 1 / (1 + (2π·|Δ|·σ_θ/λ)²)``. If
    None, uniform weights.

    ``skip_b0``: if True, zeroes out the zero-baseline (|Δ|=0) bin from
    the loss so a separate log-magnitude DC term can take over (see
    ``diag_offdiag_gram_loss`` for a simpler version of that split).

    Inputs as with other variants: ``(..., N_virt, R)`` complex. The
    pair and baseline index tensors are precomputed once from the
    virtual-array geometry.
    """
    _check(v_pred, v_gt)

    # Per-pair products, flattened over pair index: (..., P, R) complex
    prods_p = (v_pred.index_select(-2, pair_i)
               * v_pred.index_select(-2, pair_j).conj())
    prods_g = (v_gt.index_select(-2, pair_i)
               * v_gt.index_select(-2, pair_j).conj())

    n_bins = int(multiplicity.shape[0])
    p_pred = _scatter_complex_dim(prods_p, bin_idx, n_bins)        # (..., n_bl, R)
    p_gt   = _scatter_complex_dim(prods_g, bin_idx, n_bins)

    # Divide by multiplicity to get per-baseline MEAN.
    m_vec = multiplicity.clamp_min(1).to(p_pred.real.dtype)        # (n_bl,)
    m_expand = m_vec.view(*([1] * (p_pred.ndim - 2)), -1, 1)
    p_pred = p_pred / m_expand
    p_gt   = p_gt   / m_expand

    diff = p_pred - p_gt
    err = diff.real.pow(2) + diff.imag.pow(2)                      # (..., n_bl, R) real

    if skip_b0:
        if b0_bin is None:
            raise ValueError(
                "baseline_binned_loss: skip_b0=True but b0_bin not given")
        mask = torch.ones(n_bins, device=err.device, dtype=err.dtype)
        mask[b0_bin] = 0.0
        err = err * mask.view(*([1] * (err.ndim - 2)), -1, 1)

    if baseline_weights is not None:
        w = baseline_weights.to(err.dtype).to(err.device)
        err = err * w.view(*([1] * (err.ndim - 2)), -1, 1)

    L_sum = err.sum()

    if normalize:
        gt_energy = (p_gt.real.pow(2) + p_gt.imag.pow(2))          # (..., n_bl, R)
        if skip_b0:
            gt_energy = gt_energy * mask.view(*([1] * (err.ndim - 2)), -1, 1)
        if baseline_weights is not None:
            gt_energy = gt_energy * w.view(*([1] * (err.ndim - 2)), -1, 1)
        denom = gt_energy.sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


def modulus_gram_loss(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Frob² between elementwise-modulus Grams (drops all pairwise
    phase). Per range bin: ``|v[i]·v[j]*| = |v[i]|·|v[j]|``, so

    ``L_r = ||m_p m_p^T - m_g m_g^T||_F²
          = ||m_p||^4 + ||m_g||^4 - 2(m_p^T m_g)²``

    with ``m = |v|`` (real non-negative). Fully pose-robust (no phase).
    """
    _check(v_pred, v_gt)
    m_p = (v_pred.real.pow(2) + v_pred.imag.pow(2)).sqrt()         # (..., N, R)
    m_g = (v_gt.real.pow(2)   + v_gt.imag.pow(2)  ).sqrt()
    inner = (m_p * m_g).sum(dim=-2)                                # (..., R) real
    norm_p_sq = m_p.pow(2).sum(dim=-2)
    norm_g_sq = m_g.pow(2).sum(dim=-2)
    L = norm_p_sq.pow(2) + norm_g_sq.pow(2) - 2.0 * inner.pow(2)
    L_sum = L.sum()
    if normalize:
        denom = norm_g_sq.pow(2).sum().clamp_min(eps)
        return L_sum / denom
    return L_sum


# ---------------------------------------------------------------------------
# Diagnostic only
# ---------------------------------------------------------------------------

def normalised_gram_correlation(
    v_pred: torch.Tensor,
    v_gt: torch.Tensor,
    eps: float = 1e-20,
) -> torch.Tensor:
    """Mean per-range normalised Gram correlation in [0, 1].
    Unsuitable for training (scale-invariant). For logging only.
    """
    _check(v_pred, v_gt)
    inner = (v_pred.conj() * v_gt).sum(dim=-2)
    inner_mag_sq = inner.abs().pow(2)
    norm_p_sq = v_pred.abs().pow(2).sum(dim=-2)
    norm_g_sq = v_gt.abs().pow(2).sum(dim=-2)
    denom = (norm_p_sq * norm_g_sq).clamp_min(eps)
    return (inner_mag_sq / denom).clamp(0.0, 1.0).mean()


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def _check(v_pred: torch.Tensor, v_gt: torch.Tensor) -> None:
    if v_pred.shape != v_gt.shape:
        raise ValueError(
            f"shape mismatch: v_pred {tuple(v_pred.shape)} vs "
            f"v_gt {tuple(v_gt.shape)}"
        )
    if not (v_pred.is_complex() and v_gt.is_complex()):
        raise ValueError(
            f"expected complex tensors; got v_pred={v_pred.dtype}, "
            f"v_gt={v_gt.dtype}"
        )
