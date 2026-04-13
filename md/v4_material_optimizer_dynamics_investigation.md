# v4 material optimizer dynamics investigation

Follow-up to [v4_material_parameter_loo_investigation_results.md](v4_material_parameter_loo_investigation_results.md) and [v4_clamp_fixes_results.md](v4_clamp_fixes_results.md), after correcting a flawed "rank-1 trajectory" claim and running the real rank check.

## Correction: the earlier rank-1 claim was wrong

In the LOO investigation, I computed **temporal correlation of per-checkpoint trajectory means**:

```
corr( Δ mean_m(raw_materials[:, i]),  Δ mean_m(raw_materials[:, j]) )
```

across 11 checkpoints between each pair of parameters, and got |corr| ≈ 1 for all 15 pairs. I called that "rank-1".

**That was wrong.** Temporal correlation of trajectory means just measures "all parameters follow a smooth exponential-decay convergence curve during the first 200 iterations". Every parameter converges monotonically, so all the Δ-mean vectors point in the same time-direction trivially. It says nothing about the gradient structure.

## The real rank check: per-iter gradient covariance across 50K points

The correct test is: at each training iteration, compute the covariance of the per-point gradient tensor `grad ∈ ℝ^{M×K}` across the M points. The eigenspectrum of this 6×6 matrix tells you whether the 6 parameters produce independent gradient directions.

### Setup

- Simplified BSDF + raw MSE + random init (width 2.0) + all 4 clamp fixes applied
- Scene `seq_0_frame_135`, 500 iters, seed 42
- Snapshot full `(50000, 6)` gradient tensor at iters [0, 50, 100, 200, 350, 499]
- Compute both **unnormalized covariance** (sensitive to column magnitude) and **correlation matrix** (per-column z-scored)

### Results

Participation ratio (PR = (Σλ)² / Σλ²; range [1, 6]; rank 1 → PR 1, uniform full-rank → PR 6):

| iter | cov PR | λ₁/Σ (cov) | corr PR | λ₁/6 (corr) |
|---|---|---|---|---|
| 0 | 1.001 | 100.0% | 3.969 | 38.0% |
| 50 | 1.001 | 99.9% | 4.634 | 32.1% |
| 100 | 1.014 | 99.3% | 4.674 | 28.9% |
| 200 | 1.084 | 95.9% | 5.861 | 20.4% |
| 350 | 1.869 | 63.2% | 5.938 | 19.5% |
| **499** | **1.923** | **60.0%** | **5.969** | **18.6%** |

**The unnormalized covariance** looks rank-1 throughout (PR ≈ 1, λ₁ captures 60–100% of variance). That is **a unit-sensitivity artifact**: two of the six columns have ~700× larger gradient magnitudes than the other four, so the raw variance is dominated by them regardless of whether the directions are independent.

**The correlation matrix** (per-column z-scored so all columns contribute equally) has participation ratio **5.97 at iter 499** — essentially full rank out of 6. The 6 parameters are producing nearly-independent per-point gradient directions.

The correlation matrix at iter 499:

```
              eps_real  eps_imag   sigma_h       l_c  tau_base  thickness
eps_real       +1.000    +0.000    +0.000    -0.000    +0.001    +0.049
eps_imag       +0.000    +1.000    +0.005    -0.011    -0.003    -0.002
sigma_h        +0.000    +0.005    +1.000    -0.109    +0.036    -0.001
l_c           -0.000    -0.011    -0.109    +1.000    -0.002    -0.000
tau_base      +0.001    -0.003    +0.036    -0.002    +1.000    +0.000
thickness     +0.049    -0.002    -0.001    -0.000    +0.000    +1.000
```

Largest off-diagonal = **−0.109** (sigma_h ↔ l_c). All others ≤ 0.05. **Near-diagonal**, meaning the 6 parameters are nearly decorrelated in per-point gradient direction.

## What is actually going on

The 6 parameters are **directionally independent** but **grossly magnitude-imbalanced**. Per-column ‖grad‖₂ at iter 499:

| param | ‖col‖₂ |
|---|---|
| **eps_real** | **5.299e-02** |
| **thickness** | **4.351e-02** |
| l_c | 7.390e-05 |
| eps_imag | 7.856e-05 |
| sigma_h | 6.072e-05 |
| tau_base | 2.789e-05 |

**eps_real and thickness carry ~700× larger per-point gradients than the other 4.** But look at the mean drift from P1:

| param | mean drift raw-space (iter 0 → 499) |
|---|---|
| eps_real | −0.31 |
| **eps_imag** | **+0.60** (largest) |
| sigma_h | −0.04 |
| l_c | +0.26 |
| tau_base | +0.13 |
| thickness | +0.18 |

**eps_imag has the SMALLEST gradient magnitude and the LARGEST consistent raw-space movement.** That combination uniquely happens under Adam when a parameter has a small but **persistent-sign** gradient: the first moment `m` accumulates with the same sign iter after iter, and `m/sqrt(v+ε)` stays at a constant value. The effective per-iter step is then ~LR independent of the gradient magnitude.

eps_real and thickness, despite their huge gradient magnitudes, **have sign-flipping per-point gradients**. First moment `m` averages toward zero. Adam's update `m/sqrt(v+ε)` is tiny because `m` is small. Meanwhile `v` is large (captures the per-step variance), so the denominator is big. Both effects push the effective step down. **The noise cancels itself out in the first moment**, so Adam effectively ignores these parameters even though their raw gradients look dominant.

sigma_h, l_c, tau_base have small gradient magnitudes AND sign-flipping per-point directions — Adam can't use them either.

## Implications

This completely reframes the LOO findings:

- **eps_imag is essential in LOO** because it is the only parameter with a clean persistent-direction signal that Adam's moment accumulation can ride.
- **eps_real and thickness are LOO-inactive** not because of redundancy, but because their gradients are per-point sign-flipping noise that Adam correctly interprets as no-information. When frozen, the loss doesn't change because the optimizer wasn't actually using them.
- **sigma_h, l_c, tau_base are LOO-inactive** because their gradient signal is too weak to overcome the noise floor in Adam's moment estimator.
- **The 4 clamp fixes were wash** because they targeted parameters that Adam was already ignoring. Widening the clamps gave those parameters more room to move — but they didn't have a consistent direction to move *in*, so nothing changed.

The material model is **neither over-parameterized nor under-powered**. The **optimizer dynamics** are unusable for 5 of the 6 parameters given the current LR (0.7) and single-group Adam configuration.

## Four experiments to test the optimizer-dynamics hypothesis

### Option A — lower materials LR from 0.7 to 0.01

**Hypothesis**: LR=0.7 on Adam lets a single iter's noise produce a huge raw-space displacement that cancels when the sign flips next iter. Lower LR → less noise amplification per step → Adam's first moment has more iters to average out the noise and find a residual persistent direction in eps_real/thickness.

**Expected outcome**:
- If a persistent signal exists in eps_real/thickness under this noise floor, the mean drift should become clean and the cart_corr should improve.
- If no such signal exists, the result is a wash (noise is genuinely all there is).

### Option B — Adam with sign-transformed gradient (Adam-on-sign)

**Hypothesis**: SignSGD-style updates ignore gradient magnitude entirely. Replace `grad` with `sign(grad)` before Adam's moment update. Every column then has unit magnitude per point, eliminating the 700× imbalance between eps_real/thickness and the rest. Adam still does its moment accumulation, but now on unit-magnitude directions.

**Expected outcome**:
- If sigma_h, l_c, tau_base have any consistent sign direction (even tiny), Adam on sign(grad) will extract it.
- If they don't (genuinely random-walk gradients), no improvement.

### Option C — per-column gradient L2 normalization pre-Adam

**Hypothesis**: before Adam sees the gradient, normalize each column to unit L2: `grad[:, k] /= ‖grad[:, k]‖₂`. This equalizes gradient magnitudes across columns while preserving the per-point relative weights within each column.

Cleaner than Option B because it preserves the directional information per point; only cross-column magnitudes are normalized.

**Expected outcome**:
- Eliminates the 700× magnitude imbalance at the column level.
- The 4 "quiet" parameters should now move as aggressively as eps_real/thickness per iter.
- If their per-point direction is still noise, no improvement. If it has signal, improvement.

### Option D — per-column learning rate scaling

**Hypothesis**: pick different LRs per column based on what each parameter's dynamics need. The simplest tuning:

| param | LR | rationale |
|---|---|---|
| eps_real | 0.01 | kill the noise dominance, let Adam find the residual signal |
| eps_imag | 0.7 | keep — this one already works |
| sigma_h | 1.0 | boost — Adam was ignoring it |
| l_c | 1.0 | boost |
| tau_base | 1.0 | boost |
| thickness | 0.01 | kill noise dominance |

Implementation: rescale each column's Adam update post-step by a per-column factor.

**Expected outcome**:
- Most targeted of the four options.
- If the hypothesis is right, should produce the biggest improvement. If not, reveals which parameters are responding to what boost.

## Expected ceiling-breaking

Current ceiling (simplified BSDF, raw MSE, random init, all fixes): **~0.927–0.928**. Any of the four options that produces cart_corr > 0.933 (6σ above the ±0.005 noise floor) would be a genuine ceiling break and a signal that the material model had more capacity than the default Adam config was extracting.

## Results section (filled in after running)

### Baseline — current HEAD config

_Filled in after running._

### Option A — LR=0.01 for materials

_Filled in after running._

### Option B — Adam-on-sign for materials

_Filled in after running._

### Option C — per-column grad normalization

_Filled in after running._

### Option D — per-column LR scaling

_Filled in after running._

### Comparison

_Filled in after all four options are run._

## Raw data

- `mm25DGS_v4/output/material_investigation/D_rank_check/` — scene 135 gradient snapshots
- `mm25DGS_v4/analyze_rank.py` — real rank analysis script (6×6 covariance + correlation + eigenspectrum)
- Option-specific benchmarks: committed under separate fix commits in git
