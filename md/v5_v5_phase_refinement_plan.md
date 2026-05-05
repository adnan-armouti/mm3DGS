# `mm25DGS_v5_v5` — Stage-2 phase refinement plan

## Motivation

`v5_v4` trains on `MSE(|RA|_pred, |RA|_gt)` — a **magnitude-only** loss. Per
the recent CRP/ADC eval (`output/crp_adc_eval/`), this gives:

| | train | test |
|---|---|---|
| `|CRP|` Corr | 0.701 | 0.603 |
| `|CRP|` PSNR | 27.93 dB | 24.68 dB |
| Complex `\|ρ|` (VR-corrected) | 0.430 | 0.255 |
| Phase RMSE σ_φ | 65.8° | 82.5° |

The magnitude side is in the same ballpark as the published `|RA|`
(0.812/0.600 train/test Corr). The complex side is bounded above by
`|RA|` correlation — the achievable ceiling experiment showed
`|ρ|^ceiling ≈ |RA|_Corr` (oracle GT-phase substitution gives 0.78
train, 0.58 test). Our rank-1 calibration removal recovers
**~36-44% of the achievable lift**; the rest is per-RA-bin phase
content that the magnitude-only loss leaves unconstrained.

**Goal of v5_v5**: lock in v5_v4's magnitude performance, then refine
each point's position within `±λ/2` (≈ ±1.95 mm at 77 GHz) under a
**complex-valued loss** so the phase converges without disturbing the
already-good magnitude prediction.

## Why ±λ/2 and why positions

Phase encoded in the radar return at point $i$ is
`φ_i = (4π / λ) · path_length_i`. A position perturbation `Δp` along the
radial axis shifts `φ` by `(4π/λ)·Δp`. So:

- Δp = λ/4 (≈ 0.97 mm) gives a full 2π wrap of round-trip phase.
- Δp = λ/2 (≈ 1.95 mm) gives a 4π range — covers ±2π (two full cycles)
  worth of phase shift in either direction. This gives Adam slack to
  "find" the right phase without the constraint becoming the bottleneck
  during gradient descent, while still keeping the position perturbation
  geometrically negligible (sub-millimeter at typical scene scales).

Positions are the **physical** lever for phase: changing a point's
position directly changes path length. No other parameter in the
forward model has this property without affecting magnitude:
- Materials affect Fresnel coefficient phase, but the magnitude effect
  is large.
- Normals affect BSDF angle dependence, again magnitude-coupled.
- Per-point phase offset is non-physical and won't generalize across
  poses (NVS would break).

Position refinement is thus **the** principled phase-only knob.

## Approach 1 (recommended): position refinement, frozen materials/normals

### 1.1 Module layout

```
mm25DGS_v5_v5/
├── train_frame_nvs.py            # extends v5_v4 with --stage stage1|stage2|both
├── train_gaussian.py             # adds Stage-2 loss + bounded position update
├── stage2/                       # NEW
│   ├── __init__.py
│   ├── complex_loss.py           # complex-valued RA loss + |RA| guard
│   ├── bounded_position.py       # tanh-bounded delta within ±λ/2
│   └── stage2_runner.py          # the second-stage optimizer loop
└── ...                           # everything else verbatim from v5_v4
```

### 1.2 Pipeline

1. **Stage 1** (= v5_v4 unchanged): train materials, normals, positions
   for 500 iters with `MSE(|RA|, |RA|_gt)`. Save `stage1_best_model.pt`.

2. **Stage 2** (NEW):
   - Load `stage1_best_model.pt`. Freeze `raw_materials` and `rotations`
     (set `requires_grad=False`).
   - Re-parameterize positions as `pos = pos_init + (λ/2)·tanh(δ)` where
     `δ` is a learnable `(N, 3)` tensor initialized to zero.
   - Optimizer: Adam over `δ` only. LR ≈ 1e-3 (small; phase loss is
     oscillatory).
   - Loss: `L_phase = MSE(Re(RA_pred), Re(RA_gt)) + MSE(Im(RA_pred), Im(RA_gt))`
     plus `λ_mag · MSE(|RA|_pred, |RA|_gt)` as a **magnitude guard**
     (prevent the position drift from breaking magnitudes).
     Start with `λ_mag = 1.0` and ramp down to 0.1 over 100 iters.
   - 100-200 iters total (much fewer than Stage 1 — phase converges
     faster *if* it converges at all).

3. **Output**: `stage2_best_model.pt` + same artifact set as Stage 1.

### 1.3 Forward-model changes

Minimal — `render_gaussians` stays the same. The only delta is that
`model.positions` is now `pos_init + (λ/2)·tanh(δ)` instead of being a
direct `nn.Parameter`. Everything downstream (PSF, splat, FFT) is
identical.

The renderer's PSF code (`mm25DGS_v5_v4/psf.py`) IS already complex-aware
— `HannPSFTable` returns complex values and `splat` integrates complex
phasors. So the gradient path for phase already exists; the loss is just
forced to depend on it.

### 1.4 Risk & mitigation

| Risk | Mitigation |
|---|---|
| Position drift breaks magnitudes | Magnitude guard term `λ_mag · MSE(|RA|, |RA|_gt)` ramped from 1.0→0.1 |
| Phase loss is non-convex (highly oscillatory in position) | Bounded `tanh` delta + small LR (1e-3) + few iters (≤200); Adam exploits local convexity within ±λ/2 |
| Per-point phase fits noise rather than signal | L2 anchor: `λ_anchor · ||δ||²` to prefer minimal motion |
| Generalization to held-out pose breaks | Test pose phase metrics: σ_φ should drop on test too, not just train |
| MC noise in renderer disrupts gradient | Already an issue in Stage 1; same mitigations apply (averaged over multiple chirp loops if needed) |

### 1.5 Hyperparameters (initial)

```python
stage2_iters = 200
stage2_position_lr = 1e-3
stage2_position_l2_anchor = 1e-3
stage2_mag_guard_init = 1.0
stage2_mag_guard_final = 0.1
stage2_mag_guard_warmup_iters = 100
position_bound_lambda_fraction = 0.5    # ±λ/2 → ±1.95 mm at 77 GHz
```

### 1.6 Evaluation criteria for success

A successful Stage 2 must satisfy ALL of:
1. **Magnitude metrics preserved** (within 5% relative): test |CRP| Corr
   stays at 0.60±0.03, |RA| Corr stays at 0.60±0.03.
2. **Complex metrics improve substantially** on test: σ_φ from 83° to
   <70°, |ρ| from 0.26 to >0.40.
3. **Train→test generalization preserved**: complex metric improvements
   transfer to held-out test, not just training views.

If any criterion fails, fall back to v5_v4. If criteria 1+2 pass but 3
fails (overfitting), reduce iterations or increase L2 anchor.

## Approach 2 (alternative): joint complex loss in original training

Add complex term to the existing v5_v4 loss:
```python
loss = (1 − γ) · MSE(|RA|, |RA|_gt) + γ · (MSE(Re(RA), Re(RA_gt)) + MSE(Im(RA), Im(RA_gt)))
```
- γ ramped 0 → 0.3 over training
- Single-stage, no separate run
- **Risk**: phase gradient (rapidly oscillating) may destabilize early
  magnitude convergence. Could lose the headline 0.812 train Corr.
- **Pro**: closer to "principled fix" rather than two-stage hack.
- **Recommendation**: try as an A/B against Approach 1 if Stage 2 lifts
  complex but at a magnitude cost.

## Approach 3 (alternative): per-(TX,RX) learnable phase offset

Replace the offline α[v] LS calibration with a learnable per-virtual-
antenna phase `α_learn[v] ∈ [-π, π]`. Train it as a differentiable
parameter alongside Stage 2 position refinement.

- 86 extra params (negligible vs. 20k×6 material params)
- Captures real RF chain calibration drift in a way that **does**
  generalize across poses (it's a sensor calibration, pose-invariant)
- Could be used as a **standalone Stage 2** (just learn α_learn, no
  position changes) — extremely cheap, low-risk.
- **Recommendation**: include as a supplementary parameter in any
  Stage 2 variant. May lift complex metrics by itself even without
  position refinement.

## Approach 4 (alternative, gold standard): train on raw ADC

`L = MSE(Re(ADC_pred), Re(ADC_gt)) + MSE(Im(ADC_pred), Im(ADC_gt))`,
applied during Stage 1 directly on the 12×16×256 complex tensor.

- **Pro**: maximally honest — fully complex, no magnitude/phase
  decoupling tricks.
- **Con**: ADC has huge dynamic range, low SNR per cell, and the
  loss landscape is highly non-convex in materials and positions
  jointly. Likely to diverge or stagnate.
- **Recommendation**: NOT for v5_v5. Save for a separate paper /
  experiment after we understand the v5_v5 dynamics.

## Approach 5 (additional knob): allow normals (`rotations`) to refine in Stage 2

Normals affect Fresnel reflection phase via the BSDF. Allowing
quaternions to refine in Stage 2 (with the same magnitude guard) could
recover phase content the position refinement misses.

- Risk: normals affect magnitude more strongly than positions; harder
  to guard against magnitude drift.
- **Recommendation**: A/B test as an extension once position-only
  refinement is validated.

## Recommendation

**Start with Approach 1 + Approach 3 combined**:
- Stage 1: v5_v4 unchanged.
- Stage 2: position refinement (`±λ/2` bounded) + per-(TX,RX) learnable
  phase offsets, both optimized under the complex loss with the
  magnitude guard.

This is the highest-leverage / lowest-risk combination:
- Position refinement is the only physically-motivated knob for per-
  point phase.
- Per-(TX,RX) learnable offsets capture real sensor calibration drift,
  generalize across poses, and have only 86 parameters.
- Both share the same Stage 2 optimizer; minimal infrastructure change.

If Stage 2 hits its success criteria on a single canonical scene (say
seq_1_frame_185), expand to the 6-scene benchmark and report A/B against
v5_v4 in the paper as an "optional stage-2 phase refinement" supplement
section.

## Implementation checklist

- [ ] `cp -r mm25DGS_v5_v4 mm25DGS_v5_v5` → bootstrap module.
- [ ] Add `--stage` CLI flag to `train_frame_nvs.py` (`stage1` | `stage2`
      | `both`, default `both`).
- [ ] Implement `stage2/bounded_position.py` (tanh re-parameterization).
- [ ] Implement `stage2/complex_loss.py` (Re/Im MSE + magnitude guard).
- [ ] Implement `stage2/stage2_runner.py` (Adam loop on `δ` + optional
      `α_learn`).
- [ ] Wire Stage 2 to load `stage1_best_model.pt` and freeze materials
      and rotations.
- [ ] Run on seq_1_frame_185 first; verify success criteria; iterate on
      hyperparameters.
- [ ] If single-scene works: run on full 6-scene benchmark; report
      delta vs. v5_v4 in paper supplement.
- [ ] Eval pipeline: rerun `mmir.evaluation.eval_crp_adc` after Stage 2
      to get the new CRP/ADC metrics with same domain conventions.
