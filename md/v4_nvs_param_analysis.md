# NVS parameter analysis — designing a targeted regularizer

## Setup

Compared the trained parameters of two runs on `seq_0_frame_135` at
10K points, pearson loss, 300 iters:

- **A**: all-in, 8 frames {131-139} excluding misaligned 135.
  Train cart_corr = 0.8173. Considered the "good basin".
- **B**: 5/4 split, train {131,133,135,137,139}, test {132,134,136,138}.
  Train cart_corr = 0.8398, **test = 0.5064**. The 5-frame loss has
  a different minimum — overfit per train pose.

FPS is deterministic (seed=42), so positions are identical between
runs and we can compare per-point parameters directly.

## Materials — gauge-ambiguous per-point freedom

### Distributions are statistically identical

Per-column mean and std of `raw_materials` are within 5% between A and B:

| param | A mean | B mean | A std | B std |
|---|---|---|---|---|
| eps_real | 4.214 | 4.223 | 0.295 | 0.276 |
| eps_imag | -4.452 | -4.356 | 0.439 | 0.448 |
| sigma_h | -10.98 | -10.88 | 0.961 | 0.961 |
| l_c | -4.890 | -4.925 | 0.796 | 0.766 |
| tau_base | -0.033 | -0.032 | 0.182 | 0.177 |
| thickness | -1.942 | -1.935 | 0.289 | 0.269 |

Drift from ITU concrete init is also similar (B drifts 6-9% less per
column). Spatial smoothness at k=5/30/100 nearest neighbors: B/A
ratio = 0.93-1.03 — **identical at every scale**.

### But per-point assignments are uncorrelated

| param | corr(A[i], B[i]) | spearman | kendall |
|---|---|---|---|
| eps_real | 0.17 | 0.23 | 0.16 |
| eps_imag | 0.42 | 0.40 | 0.27 |
| sigma_h | 0.54 | 0.54 | 0.39 |
| l_c | 0.58 | 0.57 | 0.43 |
| tau_base | 0.31 | 0.49 | 0.46 |
| thickness | 0.16 | 0.21 | 0.14 |

The per-point material values are essentially a random gauge choice.
A and B produce the same global statistics with completely different
point-to-material assignments.

### Position+normal regression: small shared component

Ridge regression `material[k] ~ poly2(pos, normal)`:

| param | R² on A | R² on B | corr(predA, predB) |
|---|---|---|---|
| eps_real | 0.04 | 0.01 | **0.76** |
| eps_imag | 0.07 | 0.06 | **0.94** |
| sigma_h | 0.19 | 0.13 | **0.97** |
| l_c | 0.22 | 0.13 | **0.98** |
| tau_base | 0.07 | 0.05 | **0.89** |
| thickness | 0.02 | 0.01 | **0.58** |

The R² is low (~0.1-0.2) but the predictions of A and B agree at
~0.6-0.98. This is the smoking gun:

- Both runs find the same small "geometry-determined" material
  component
- The remaining ~80-95% of per-point variance is **independent random
  gauge** in each run

### Implication

The 5-frame loss does not uniquely determine per-point materials.
Many parameter sets yield the same (or even better) train loss than
the all-in solution; the optimizer picks one arbitrarily, and most
of these don't generalize.

**Smoothness regularization will NOT fix this** — both A and B are
already equally smooth at all scales. Anything that constrains the
*statistics* of materials won't help either, because A and B have
matching statistics yet very different test cc.

The fix has to **eliminate the per-point freedom**, not constrain it.

## Normals — structured drift, regularizable

| | A vs init | B vs init | A vs B |
|---|---|---|---|
| Mean angle | 20.1° | 19.9° | **15.5°** |

If A and B were independent random fits, A vs B angle would be ~28°
(√2 × 20°). It's only 15.5°. **Both runs drift in similar directions
from the pcl init — there's a shared "right" rotation that both runs
partly find.**

Per-point normal differences:
- Median: 9°
- P90: 38°
- P99: 78°
- 16% of points have >30° angular difference

So normals have a shared structured drift PLUS a noisy per-point
component. Unlike materials, the structure is recoverable — both
runs converge toward similar corrections.

## Recommended regularizer

Two complementary terms, both targeted at the analysis findings:

### Term 1: Material gauge collapse (CRITICAL)

Parameterize materials as a learnable function of geometry, not as
free per-point floats:

```
material[i] = base + W · features(pos[i], normal[i])
```

where:
- `base` is a learnable (6,) global material vector (like the init)
- `features(.)` is a fixed feature map (polynomial of degree 2 over
  position+normal, normalized to unit scale → ~27 features)
- `W` is a learnable (27, 6) projection

Total learnable: 6 + 27×6 = 168 parameters for materials, vs
10000×6 = 60K with per-point. **350× capacity reduction**, but the
analysis shows only ~10-20% of material variance was geometry-
determined anyway, so we lose almost nothing of "real" signal.

Per-point residuals are killed entirely. The gauge ambiguity is
mathematically removed because there's no per-point degree of freedom.

This is the v4-internal version of "use an MLP for materials" without
introducing a deep network — the polynomial features capture the
geometric dependence the analysis revealed.

### Term 2: Normal anchor regularization (MILD)

Add a soft anchor to the init pcl normals:

```
loss += λ_normal · mean(1 - cos(normal[i], normal_init[i]))²
```

with small `λ_normal` (e.g. 0.001-0.01). The analysis showed both A
and B drift ~20° from init in similar directions → the anchor will
NOT pull against the structured drift, it will just suppress the
per-point random component.

Alternatively (cleaner): anchor in quaternion space:
`λ_normal · ||quat - quat_init||²`.

### What we are NOT doing

- Spatial smoothness reg → ruled out (A and B equally smooth)
- Distance-from-init L2 on materials → ruled out (A and B drift
  similar amounts)
- Cluster regularization → already tested, doesn't help (cluster
  centers themselves are gauge-free)
- Tiny LR → kills training, doesn't fix the underlying issue

## Implementation plan

Modify `train_nvs` to take:
- `mat_feature_basis` flag — when on, replace per-point materials with
  the (base + W·features) parameterization
- `lambda_normal_anchor` — float, the anchor weight

Then run a single A/B test:
- Baseline: current 5/4 split, no reg → expected test cc 0.51
- New: 5/4 split with material gauge collapse + normal anchor → if
  test cc lifts toward 0.7+, the regularizer is working

If lift is 0.6-0.7, push on the parameterization (try MLP instead of
polynomial). If lift is 0.51-0.55, the analysis is wrong and we need
to look again.
