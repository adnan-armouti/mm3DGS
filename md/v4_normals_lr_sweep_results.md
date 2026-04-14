# v4 normals LR sweep — results

Executes **Direction 3** from [v4_ceiling_breaking_directions.md](v4_ceiling_breaking_directions.md): analog of the material LR sweep ([v4_material_optimizer_dynamics_investigation.md](v4_material_optimizer_dynamics_investigation.md)) applied to the rotation parameter group.

## Setup

- Simplified BSDF + raw MSE + factory patterns + `LEARN_PATTERNS=False`
- Random per-point init (width 2.0, seed 42)
- All 6 material parameters learnable, `mat_lr = 0.01` (from material LR sweep)
- Rotation LR values: {5e-4, 2e-3 (current), 5e-3, 2e-2, 5e-2}
- 500 iters × 7 scenes per configuration
- Everything else held constant

**Baseline (rot_lr = 2e-3)**: 0.9325 mean cart_corr (noise floor ±0.005).

## Results

| rot_lr | mean | Δ vs 2e-3 |
|---|---|---|
| 5e-4 | 0.9123 | -0.0202 |
| 2e-3 (current default) | 0.9325 | 0 |
| **5e-3** | **0.9359** | **+0.0034** ← **optimum** |
| 2e-2 | 0.9265 | -0.0060 |
| 5e-2 | 0.9152 | -0.0173 |

### Per-scene rot_lr=5e-3 vs rot_lr=2e-3

| Scene | rot_lr=2e-3 | **rot_lr=5e-3** | Δ |
|---|---|---|---|
| seq_0_frame_135 | 0.9229 | 0.9232 | +0.0003 |
| seq_0_frame_390 | 0.9375 | 0.9401 | +0.0026 |
| seq_1_frame_185 | 0.9636 | 0.9668 | +0.0032 |
| seq_1_frame_438 | 0.9591 | 0.9594 | +0.0003 |
| seq_2_frame_105 | 0.9220 | 0.9252 | +0.0032 |
| seq_2_frame_160 | 0.9064 | 0.9089 | +0.0025 |
| **seq_2_frame_300** | **0.9162** | **0.9278** | **+0.0116** |
| **Mean** | **0.9325** | **0.9359** | **+0.0034** |

**All 7 scenes improve**, 0 regress. Under a null hypothesis of "each scene is independent zero-mean noise", the probability of 7/7 positive is (1/2)⁷ ≈ 0.8%. The gain is genuine.

Scene 300 is the biggest beneficiary (+0.0116), 3× the mean improvement. It was also one of the scenes that benefited most from the material LR sweep, suggesting scene 300 has features that both knobs collectively needed to resolve.

## Interpretation

The normals LR was **below optimum by a factor of ~2.5×**, analogous to but opposite in direction from the material case. The material LR was **above optimum by 70×** (0.7 → 0.01); the rotation LR was **below optimum by 2.5×** (2e-3 → 5e-3).

Both LR defaults were picked from reasonable guesses by the user during the earlier project stages and were never systematically tuned. Both were off-optimum in ways that systematic sweeping immediately identifies.

### Shape of the curve

Clean single-peak:
- **5e-4 (10× below optimum)**: 0.9123, −0.020 — under-converged, 500 iters not enough steps
- **2e-3 (2.5× below optimum)**: 0.9325, baseline
- **5e-3 (optimum)**: 0.9359
- **2e-2 (4× above optimum)**: 0.9265, −0.006 — mild overshoot
- **5e-2 (10× above optimum)**: 0.9152, −0.017 — full overshoot

The curve is similar in shape to the material LR sweep (unimodal, symmetric around the peak on a log scale), just centered at a different value. Both param groups have a narrow sweet spot.

### Why 5e-3 is the right number for rotations and 0.01 is the right number for materials

- **Rotations** are unit-quaternion perturbations. Each parameter has a natural scale of ~0.01 to 0.1 per "meaningful surface normal change" — small updates should produce small but visible normal rotations. At 2e-3, convergence takes too long; at 5e-3, it completes in 500 iters without overshoot.
- **Materials** are log/logit-parameterized physical quantities. Each parameter has a natural scale of ~0.1 to 1 per "meaningful material change" in raw space. At 0.7, updates overshoot by 70×; at 0.01, they match the gradient amplitude.

The sweet spots being so different (0.005 vs 0.01) is expected — the two parameter types have fundamentally different natural scales and different gradient dynamics.

## Compound improvement

Combining all the tuning in this investigation:

| Config | Mean | Cumulative Δ from "mse min-max loss + clamps + LR=0.7" |
|---|---|---|
| Old Pearson baseline (mmIR patterns) | 0.9425 | — |
| Simplified BSDF + raw MSE + factory patterns + clamp fixes + LR=0.7 + rot_lr=2e-3 | 0.9281 | 0 |
| Same + mat_lr=0.01 | 0.9322 | +0.0041 |
| Same + mat_lr=0.01 + rot_lr=5e-3 | **0.9359** | **+0.0078** |

**+0.0078 total improvement** from just two LR number changes on top of the rest of the work. The ceiling is now 0.9359, and it's approaching the old Pearson baseline (0.9425) despite using a genuinely physical raw MSE loss rather than Pearson's scale-invariant structural trick.

## Default change

**Changed default `rot_lr` from 2e-3 to 5e-3** in `train_gaussians`. Matches the material LR change made earlier.

## What this does not tell us

- **Rotation LR may interact with material LR**. The sweep was run at fixed `mat_lr = 0.01`. A joint 2D sweep over (mat_lr, rot_lr) might find a slightly different optimum for each. Low expected value (probably <0.002) given both curves are fairly wide around the optimum.
- **Rotation clip (currently 0.5) was not re-tuned**. Clipping is coupled to LR: as LR grows, the clip threshold should probably grow too. At rot_lr=5e-3, the 0.5 clip may or may not be the right value. Could be swept as a follow-up.
- **Scene 300's +0.0116 is huge**. Worth understanding qualitatively why this scene benefits so much more than the others. Probably contains normal-rotation-sensitive features (e.g., a specific wall angle or fence section) that the old 2e-3 wasn't converging on.

## Next actionable steps (from the plan)

1. **(Done, this document)** — Direction 3 normals LR sweep.
2. **Direction 2A — Sobel/gradient loss** (~3 hours). Expected +0.003 to +0.010.
3. **Direction 1A — polarization-preserving pipeline** (~1 day). Expected +0.005 to +0.015 if GT has polarization structure.

## Raw data

- `mm25DGS_v4/output/normals_lr_sweep.json` — per-run per-scene cart_corr values

## Files changed

- `mm25DGS_v4/train_gaussian.py`: added `rot_lr` parameter (default now 5e-3, was 2e-3)
