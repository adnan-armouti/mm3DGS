# Baseline-binned correlation loss: redesign of `gram_frobenius_loss`

Date: 2026-04-22
Context: replacing v5 MSE-RA loss (86-element ULA) with a loss that uses all
192 virtual antennas on the MMWCAS sparse 2D aperture, while preserving:
1. Per-antenna magnitudes
2. Pairwise relative phases
3. Invariance to absolute/global phase

---

# Diagnosis

Your §4 already tells you the fix. The physics sees `|v̂[k]|² = DFT(p)[k]` where
`p[d] = Σ_m C[m, m-d]` is the **sum along each diagonal** of the Gram matrix,
not its individual entries. For any plane-wave target, the `N²` Gram entries
are not independent — they're forced to be equal up to per-diagonal
accumulation. MSE-RA penalises a DFT of the lag-sum; Gram-Frobenius penalises
each `C[m,n]` individually and so spends gradient budget enforcing coherence
patterns that are *redundant under the forward model but incoherent under pose
noise*. That's the over-enumeration you flagged — the long diagonals (|d|
close to N) collect dozens of pairs, each of which is maximally pose-
sensitive, and your loss is summing their independent squared errors instead
of averaging them first.

The 86-element FFT path quietly does the right thing (sum-then-penalise, via
`p → DFT → |·|`). The 192-element Gram-Frobenius does the wrong one. The
cleanest upgrade is to lift §4's lag-sum identity from 1D ULA to the 2D sparse
aperture.

# The fix: baseline-binned correlation loss

For each pair `(i,j)` define the 2D baseline `b_ij = (x_i - x_j, z_i - z_j)`.
Two pairs with the same `b` carry the same plane-wave phase and their Gram
entries should be averaged, not separately penalised. Replace `p[d]` by

$$
p[b] \;=\; \frac{1}{M(b)} \sum_{(i,j): b_{ij}=b} v[i]\,\overline{v[j]}
$$

where `M(b)` is the multiplicity of baseline `b` in the 2D sparse co-array.
The MMWCAS 192-element aperture has only a few hundred unique 2D baselines —
not 192² = 36864 — so this collapses the loss onto the physically meaningful
degrees of freedom.

The loss is then

$$
L \;=\; \sum_{d,r} \sum_{b} w(b)\,\bigl|\,p_{\text{pred}}[b,d,r] \;-\; p_{\text{gt}}[b,d,r]\,\bigr|^2
$$

Properties:

- **Per-antenna magnitudes**: the `b = 0` bin is `(1/N) Σ_i |v[i]|²`, so it
  supervises total energy per (d,r). You can split out `b = 0` and use a
  separate magnitude-only term if you want the log-domain scaling; see below.
- **Pairwise relative phases**: each `b ≠ 0` bin carries the averaged complex
  phase at that spatial lag — exactly the DBF-relevant statistic.
- **Global-phase invariant**: `p[b]` contains `v[i]·conj(v[j])` so the
  `e^{jα}` on `v` cancels identically, every time, at every `(d, r)`. No
  renormalisation needed.
- **Reduces to MSE-RA**: if you apply a 2D DFT over `b` and take magnitude,
  you recover the 2D range-angle(-elevation) magnitude map, matching your v5
  signal exactly on the azimuth-only subset. MSE-RA is a lossy linear
  projection of this loss — so it is a strict superset of v5's supervisory
  signal.

## Code

Two functions: a one-time setup that computes the pair→bin map, and the
per-step loss.

```python
def precompute_baseline_bins(positions, grid_spacing=0.5):
    """positions: (N, 2) antenna (x, z) in wavelengths.
    Returns (i_idx, j_idx, bin_idx, multiplicity, unique_baselines)."""
    N = positions.shape[0]
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    i_flat, j_flat = ii.flatten(), jj.flatten()
    baselines = positions[i_flat] - positions[j_flat]               # (N*N, 2)
    # Quantise to the natural co-array grid (λ/2 spacing for MMWCAS)
    bq = torch.round(baselines / grid_spacing).long()
    unique_b, bin_idx = torch.unique(bq, dim=0, return_inverse=True)
    n_bins = unique_b.shape[0]
    mult = torch.zeros(n_bins, dtype=torch.long)
    mult.scatter_add_(0, bin_idx, torch.ones_like(bin_idx))
    return i_flat, j_flat, bin_idx, mult, unique_b.float() * grid_spacing


def baseline_binned_loss(
    v_pred, v_gt,                          # (..., N, R) or (..., D, N, R) complex
    i_flat, j_flat, bin_idx, mult,         # from precompute_baseline_bins
    baseline_weights=None,                 # (n_bins,) real, optional
    eps=1e-12,
):
    """Loss = Σ_{d,r,b} w(b) |p_pred[b] - p_gt[b]|²,
    with p[b] = mean over pairs with baseline b of v[i]·conj(v[j])."""
    # Pair-wise outer products, flattened over pair index
    vp_i  = v_pred.index_select(-2, i_flat)
    vp_jc = v_pred.index_select(-2, j_flat).conj()
    vg_i  = v_gt.index_select(-2, i_flat)
    vg_jc = v_gt.index_select(-2, j_flat).conj()
    pair_pred = vp_i * vp_jc                                       # (..., N*N, R)
    pair_gt   = vg_i * vg_jc

    # Scatter-average over pair axis into bins
    shape = list(pair_pred.shape)
    n_bins = mult.shape[0]
    shape[-2] = n_bins
    p_pred = torch.zeros(shape, dtype=pair_pred.dtype, device=pair_pred.device)
    p_gt   = torch.zeros_like(p_pred)
    idx = bin_idx.view(*([1] * (pair_pred.ndim - 2)), -1, 1).expand_as(pair_pred)
    p_pred.scatter_add_(-2, idx, pair_pred)
    p_gt.scatter_add_(-2, idx, pair_gt)
    m = mult.clamp_min(1).to(p_pred.real.dtype).view(*([1] * (p_pred.ndim - 2)), -1, 1)
    p_pred = p_pred / m
    p_gt   = p_gt / m

    diff = p_pred - p_gt                                           # complex
    err  = diff.real.pow(2) + diff.imag.pow(2)                     # (..., n_bins, R)

    if baseline_weights is not None:
        err = err * baseline_weights.view(*([1] * (err.ndim - 2)), -1, 1)

    denom = (p_gt.real.pow(2) + p_gt.imag.pow(2)).sum().clamp_min(eps)
    return err.sum() / denom
```

That's the minimal replacement for `gram_frobenius_loss`. The precompute is
cached once per radar geometry, so the per-step cost is just `N²` complex
multiplies + a scatter — same order as your current implementation, no Gram
materialisation.

# Tuning knobs (address your three hypotheses directly)

**Pose noise → baseline weighting.** Pose error `σ_θ` maps to phase error
`≈ 2π|b|σ_θ/λ` at baseline `b`, so phase SNR falls as `1/|b|`. Use

```python
baseline_weights = 1.0 / (1.0 + (2 * math.pi * baselines_norm * sigma_theta) ** 2)
```

with `baselines_norm = unique_baselines.norm(dim=-1)` and `sigma_theta` set
from your empirical pose-noise floor (try 1°–3° in radians as a starting
sweep). This is a proper Wiener-style weighting: magnitude term (`|b|=0`)
keeps full weight, long baselines are discounted to match their GT fidelity.
You can go further and zero out the top-decile longest baselines entirely —
at 192 elements you still have 100+ short baselines carrying most of the
azimuth resolution.

**Over-enumeration → binning itself.** The mean-over-pairs-per-bin already
handles this. Redundant virtuals (the MMWCAS has ~106 duplicates at elevation
0) now contribute one term, not `106²` terms.

**Doppler-gate the static clutter.** Zero-Doppler bins concentrate ego-motion
residuals — exactly where pose noise contaminates the GT. Mask the loss for
`|d - d_ego| < d_tol`, identical in spirit to your M3 plan, with the bonus
that once the loss is per-`(b, d, r)` the gate is trivially a multiplicative
mask on the `d` axis. Start with a 3-bin gate around `d_ego`.

**Log-magnitude DC term.** Radar returns span 40+ dB. The `b = 0` bin alone —
pure per-(d,r) total power — benefits from a separate log-domain term:

```
L_mag = mean over (d,r) of (log(p_pred[b=0] + eps) - log(p_gt[b=0] + eps))²
L_total = α · L_mag + β · baseline_binned_loss(..., skip_b0=True)
```

This is what MSE-RA implicitly does (its range-azimuth magnitude compresses
dynamic range via the DFT spreading energy across bins). `α=1, β=0.1` is a
reasonable starting ratio given the baseline bins outnumber the magnitude
term.

# On the matrix completion track

The Sun/Petropulu block-Hankel approach is elegant for the *forward* problem
(recover missing virtuals from observed sparse ones under low-rank priors).
For your *inverse-rendering loss* it's the wrong tool:

1. Nuclear-norm backward passes via SVD are slow and numerically soft,
   especially on the 9009×8928 block-Hankel your 2D layout induces.
2. The low-rank prior is that "few targets per range bin" — fragile for
   urban/dense scenes and the source of a hyperparameter (SVT threshold) to
   tune.
3. You already have a non-noisy `v_gt` — you don't need a completion prior,
   you need a consistency metric between `v_pred` and `v_gt`. The
   baseline-binned loss *is* the data-consistency term that would sit inside
   an unrolled-ADMM completion network; skipping the outer optimisation is
   the right call.

Keep matrix completion in the back pocket as a *pretraining* or *data
augmentation* signal (e.g. train the renderer to produce `v_pred` whose
block-Hankel is low-rank as a regulariser), but not as the primary loss.

# Recommended rollout

1. Drop in `baseline_binned_loss` with uniform weights, no Doppler gate.
   Expect it to already match or slightly beat un-normalised Gram-Frobenius.
2. Add Wiener baseline weights with `σ_θ` from your pose-noise audit. This
   is where the pose-robustness hypothesis gets tested.
3. Add the Doppler gate on top of (2).
4. Split out the `b = 0` log-magnitude term. Sweep `α, β`.
5. If still below MSE-RA: the 192-element supervisory signal may genuinely
   be noise-dominated in your setup, and the right answer is **hybrid** —
   use MSE-RA on the 86-ULA (known-good) plus baseline-binned loss
   restricted to the elevation sub-array pairs (the supervision MSE-RA
   can't see). Additive, guaranteed to be no worse than v5 baseline.

One sanity check before you run (1): compute `n_unique_baselines` for your
MMWCAS geometry. If it's in the 300–600 range, you're in good shape. If it's
over 1000, tighten `grid_spacing`; if under 100, loosen it — you want enough
bins to preserve angular resolution but not so many that multiplicity
averaging vanishes.
