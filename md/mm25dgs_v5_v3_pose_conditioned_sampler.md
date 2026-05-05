# mm25DGS_v5_v3 — Pose-Conditioned Differentiable Point Sampler + Amplitude-Only Position Gradients

**Date:** 2026-04-23
**Scope:** Architectural extension of `mm25DGS_v5_v2` (see `md/mm25dgs_v5_v2_neural_gaussian_radar_field.md` for the base plan). Adds two complementary capacity-unlock mechanisms.
**Performance goals (carry over from v5_v2):**
- No-holdout multi-frame mean **|RA| train CC > 0.9** (currently 0.82)
- Held-out **|RA| test CC > 0.7** (currently 0.54)

---

## §0 Position summary on the user's two proposals

> "Do you agree with direction #2 (pose-conditioned differentiable sampler)?"

**Yes, with one technical refinement.** A literal "MLP outputs xyz of points" is hard to make differentiable end-to-end *when constrained to be a subset of the LiDAR pcl* — because subset selection is a discrete operation. The clean way to get the user's stated benefit (per-pose unique active-point set, end-to-end backprop, LiDAR-prior inductive bias) is:

- Treat the LiDAR pcl as a fixed **dictionary** of M_total ~ 50K-200K candidate scatterers.
- An MLP outputs a continuous **per-point selection score** σ_p ∈ [0, 1] conditioned on the radar pose.
- σ_p enters the renderer as an opacity multiplier on each point's coherent contribution.
- An L1 sparsity loss on σ pushes most scores toward 0 (the "select few" behaviour).
- At test time, σ_p naturally acts as a soft visibility mask conditioned on pose.

This is a *learnable visibility / attention field* over the LiDAR pcl — fully differentiable, retains the LiDAR distribution as a hard inductive prior, and gradients flow end-to-end as the user intends. There is direct prior art for this pattern (see §6).

> "Do you agree with direction #3 (amplitude-only position gradients)?"

**Yes, this is technically sound and I think the right way to unfreeze positions for a coherent mmWave renderer.** The reasoning is correct:

- At λ = 3.9 mm (77 GHz), 1 mm position error → 1.6 rad phase error → noise dominates any phase-derived position gradient.
- But the **amplitude** depends on position smoothly: 1/d² falloff, cos θ_i terms in BSDF, antenna gain G(direction = position / d). All of these have well-conditioned derivatives w.r.t. position.
- Implementation requires two copies of position-dependent quantities in the renderer: a `positions_phase = positions.detach()` for the phase path and a `positions_amp = positions` (with grad) for the amplitude path. PyTorch autograd does the rest.
- This avoids the phase-wrap pathology while preserving useful position signal.

> "Should we / can we combine #2 and #3?"

**Yes, but they answer different design questions and the cleanest combined architecture stages them.** Direction #2 gives us per-pose unique active sets via score-based attention over the LiDAR pcl. Direction #3 is what we use *if* we additionally want to perturb positions sub-mm away from their LiDAR seeds. Concretely:

- **Phase 7 (Direction #2 alone, stay at LiDAR positions):** simplest, isolates the visibility-attention contribution. Positions remain frozen at LiDAR; only the σ_p scores and per-point material features are learnable. No position-gradient machinery needed.
- **Phase 8 (combine #2 + #3):** add bounded per-point position deltas, learned via the amplitude-only path. Tests whether the LiDAR-position approximation is the binding constraint after Phase 7 lands. If Phase 7 alone gets us to the 0.7 test target, Phase 8 is optional.

This staging lets us measure the contribution of each direction separately — the diagnostic-sprint discipline from `md/diagnostics/RESULTS.md` carries over.

---

## §1 Empirical motivation

From `md/diagnostics/RESULTS.md` and the user's chirp-vs-frame observation:

| training regime | |RA| train | |RA| test |
|---|---:|---:|
| single-frame, all 16 chirps, 1 held-out chirp | ~0.95 train | **0.80–0.85 test** |
| multi-frame, 1 chirp/frame (v5/M1) | 0.83 | 0.54 |
| multi-frame, 16 chirps/frame (v7) | 0.69 | 0.59 |

The user's diagnosis is sharp:
- Within a frame, chirps are 0.5 ms apart — radar moves ~0.6 mm — chirp views are *nearly identical*. The shared representation can fit them all easily.
- Between frames, 0.2 s passes — radar moves ~25 cm — frame views are *highly distinct*. The shared 20K-point representation is overburdened: it must explain **9 different sparse-RA images** with one set of materials and one set of normals.

The held-out-test gap (0.83 train → 0.54 test) is the 6-scene-bench evidence that the shared model is overfitting per-frame quirks while still missing per-frame uniqueness. Both #2 and #3 attack this.

---

## §2 Direction #2 — pose-conditioned differentiable sampler

### §2.1 The problem with literal "MLP outputs xyz"

A naive `MLP(pose) → {xyz_1, ..., xyz_K}` has three issues:

1. **Set-output**: MLPs output vectors, not unordered sets. Need permutation-invariance (Set Transformer / Deep Sets architecture) — extra complexity.
2. **Subset constraint**: "stay near LiDAR distribution" is non-trivial. Snapping to nearest LiDAR point (argmin) is non-differentiable. Soft assignment (Gaussian similarity weights) loses the "subset of LiDAR" property and just becomes deformation.
3. **Variable size**: K is fixed at architecture time, but the actual visible-point set per pose has variable cardinality.

### §2.2 The clean reformulation: per-point pose-conditioned visibility

For each LiDAR point p ∈ ℝ³ in a fixed dictionary of M_total candidates:

```
σ_p(pose) = sigmoid(MLP_θ([f_p, h_pose(pose), p_xyz]))
```

where:
- `f_p ∈ ℝ^D_feat` is a learnable per-point feature (D_feat = 32, similar to v5_v2 plan).
- `h_pose(pose) ∈ ℝ^D_pose` is a small encoding of the 6-DoF radar pose (D_pose = 32). Could be raw 6-vec, or sinusoidal encoding (Mildenhall et al. 2020, NeRF positional encoding [2]).
- The MLP is small (2-3 layers, 64 units) and **shared across all points and poses**.
- Output σ_p ∈ [0, 1] is the per-point visibility/opacity score for this pose.

This MLP is structurally analogous to:
- The deformation MLP in **Deformable 3D Gaussians** (Yang et al. 2024 [6]), which predicts per-Gaussian time-conditioned offsets. We replace "time" with "pose" and "offset" with "selection score".
- The per-point feature decoder in **Point-NeRF** (Xu et al. 2022 [4]), which predicts per-point density/colour from learnable features. We add the pose conditioning.
- The differentiable selection in **SampleNet** (Lang et al. 2020 [12]), which produces a learnable point-cloud subsample via continuous relaxation.

### §2.3 How σ_p enters the renderer

In v5's `render_factorized`, the per-Gaussian "areas" tensor is currently a binary 1.0 / 0.0 active mask. Replace it with the continuous σ_p:

```python
# v5 (current)
areas = vertex_areas  # binary {0, 1}^N from cull_gaussians

# v5_v3 (new)
areas = vertex_areas * sigma_p  # σ_p ∈ [0, 1]^N from MLP
```

Everything downstream — the BSDF amplitude, the coherent path sum, the step5 splat — is unchanged. Gradients flow back into σ_p, then into MLP_θ, then into both `f_p` and the pose-encoding branch.

### §2.4 The sparsity inductive bias

A naive run will land σ_p = 1 for all points (no gradient pressure to be sparse). Add an L1 sparsity loss:

```python
loss_total = mse_raw(|RA|_pred, |RA|_GT) + λ_sparsity · ||σ||_1 / N
```

with `λ_sparsity` tuned so that ~20% of points have σ > 0.5 at convergence (matches the v5 cull_gaussians behaviour where ~20% of LiDAR pcl is FOV-visible per pose).

### §2.5 Why this specifically helps the held-out test gap

At test time, the test pose is *bracketed* by F-1 and F+1 train poses (NVS interpolation, never extrapolation — see [feedback memory: NVS = interpolation only]). The MLP has seen these neighbouring poses during training and learned σ_p(pose) as a smooth function over pose. At a new pose, σ_p interpolates smoothly — it activates the points the model "thinks" should be visible at this pose.

Compare: in v5, the active-point mask is recomputed per-pose by `cull_gaussians` based on FOV geometry alone — no learning, no smoothness. The forward model is constrained to one rigid set of materials/normals that must fit *all* of these geometrically-different active sets. The MLP-conditioned σ relaxes this rigidity: the model can effectively use *different effective subsets* of the pcl per pose, while still sharing materials/normals across the global dictionary.

---

## §3 Direction #3 — amplitude-only position gradients

### §3.1 The phase noise problem

At 77 GHz, λ = 3.896 mm. A position perturbation of 0.5 mm gives a path-length perturbation of ~1 mm (round-trip), which is a phase change of `2π · 1mm / 3.896mm ≈ 1.6 rad`. This is phase noise of ~25% of a full cycle per sub-mm displacement. Gradient descent on coherent-sum loss with respect to position has a wildly oscillating, multi-modal landscape on this scale.

Specifically, the v7 sprint's pose_refine pilot (init-state model, full-coherent-sum objective, ±4 mm bound) walked to the bounds on 7 of 9 frames with apparent CC gain of +0.02 — and when those refined configs were used in training, |RA| test **dropped by 0.05**. The init-state coherent-sum landscape was dominated by phase-noise spurious optima. (See `md/diagnostics/RESULTS.md` §6.)

### §3.2 Why amplitude is well-conditioned

The amplitude factor for path (p → tx → rx) in our v5 model is approximately:

```
A ∝ √(G_tx(d_tx) · G_rx(d_rx)) · √f_BSDF(cos θ_i, cos θ_o, ...) / (d_tx · d_rx)
```

All of these are *smooth* functions of position p:

- `1 / d` falls off as O(1/d) — gradient O(1/d²), no oscillation.
- `cos θ_i = ⟨wi, n⟩` where `wi = (tx - p) / d` — smooth in position.
- Antenna gain `G(direction)` is a smoothly-interpolated lookup table — Lipschitz-bounded.
- BSDF terms (Cook-Torrance, Smith G, Fresnel) are all rational/smooth in cos θ.

The amplitude landscape has **continuous gradients with no 2π wrapping**, which is exactly what Adam needs.

### §3.3 Implementation

In `rasterizer_factorized.py`, modify the "Step 2-3-4" blocks to use a detached copy of positions for phase computations and the original positions for amplitude computations:

```python
def render_factorized(positions, normals, ..., enable_amp_position_grad=False):
    if enable_amp_position_grad and positions.requires_grad:
        positions_phase = positions.detach()
        positions_amp   = positions
    else:
        positions_phase = positions
        positions_amp   = positions

    # Phase path — uses positions_phase ONLY
    diff_tx_phase = rast.tx_positions[None, :] - positions_phase[:, None]
    d_tx_phase    = diff_tx_phase.norm(dim=-1).clamp(min=1e-6)
    tau_tx        = d_tx_phase / C_LIGHT
    phi_tx_const  = TWO_PI * f0 * tau_tx
    n_peak        = (d_tx_phase + d_rx_phase) / (2 * range_res)
    # n_peak and phi_carrier are subsequently .detach()ed for the
    # fused step5 kernel (existing v5 contract).

    # Amplitude path — uses positions_amp (gradient flows)
    diff_tx_amp = rast.tx_positions[None, :] - positions_amp[:, None]
    d_tx_amp    = diff_tx_amp.norm(dim=-1).clamp(min=1e-6)
    wi          = diff_tx_amp / d_tx_amp.unsqueeze(-1)
    G_tx        = rast.tx_antenna.evaluate(wi.reshape(-1, 3), ...).reshape(M, n_tx)
    alpha_tx    = sqrt(G_tx) / d_tx_amp
    # All BSDF terms (cos_i, lambda_i, h_dot_n, R_TE, etc.) use positions_amp via wi, n_eff.
    # f_cos = (BSDF eval, all using positions_amp-derived geometry)
    w_full = C_radar * sqrt(f_cos) * alpha_tx[..., None] * alpha_rx[None, ...]
    # w_full is gradient-aware in positions through the amplitude path.
```

The fused CUDA kernel `step5_fused` operates on `w_full`, `phi_carrier` (detached), `n_peak` (detached), so its existing detached-phase contract is preserved. We only need to change the upstream PyTorch geometry computation.

### §3.4 Bounded per-point delta

Even with smooth amplitude gradients, unbounded position learning is risky (the model can drift far from physically plausible scatterers). Use the v5_v2 plan's bounded-delta parametrisation:

```python
positions_active = lidar_pcl + bound_mm * tanh(pos_delta / bound_mm)
```

with `bound_mm = 3` (well within v5_v2 plan's recommendation, comfortably above the 0.5 mm phase-wrap scale).

`pos_delta` is the learnable parameter, initialized to 0. The L2 regulariser `λ_pos · ||pos_delta||²` with `λ_pos = 0.01` keeps it small unless the data justifies a larger move.

---

## §4 Combined architecture (Phases 7 + 8)

```
LiDAR pcl ── M_total candidate xyz (fixed dictionary)
                    │
                    ├──→ per-point feature f_p ∈ R^32 (learnable)
                    │
              ┌────────────────────────────┐
              │      MLP_θ(σ-net)          │
   pose F ───→│ inputs: [f_p, pose,        │
              │          (xyz)]            │
              │ outputs: σ_p ∈ [0, 1]      │
              └────────────────────────────┘
                    │
              σ_p (per-point pose-conditioned opacity)
                    │
                    ↓
              ┌────────────────────────────┐
              │    Material decoder MLP    │
              │  features f_p → 6 ITU      │
              │  params + 3 normal Δ       │
              └────────────────────────────┘
                    │
              materials, normals (per-point, GLOBAL across poses)
                    │
                    ↓
       positions_active = lidar_pcl + bound · tanh(δ/bound)   [Phase 8 only]
       (Phase 7: positions_active = lidar_pcl, frozen)
                    │
                    ↓
       v5 forward model with:
         - amplitude path: gradient through positions_active via §3
         - phase path:    detached positions_active.detach()
         - opacity per point: σ_p (replaces binary `vertex_areas`)
                    │
                    ↓
       |RA|_pred → mse_raw against GT (single chirp) +
                   λ_sparsity · L1(σ) +
                   λ_pos · L2(δ)        [Phase 8]
                    │
                    ↓ backprop
       gradients flow into:
         - σ-MLP weights θ_σ, pose encoding params
         - per-point features f_p
         - material decoder MLP weights
         - position deltas δ_p (Phase 8 only, AMPLITUDE PATH ONLY)
```

---

## §5 Implementation phases

These are **Phase 7 and Phase 8** of the v5_v2 plan (`md/mm25dgs_v5_v2_neural_gaussian_radar_field.md`). v5_v2 Phases 0–6 must complete first; this plan builds on top.

### Phase 7 — pose-conditioned differentiable sampler (Direction #2 alone)

**Effort:** 2 weeks.

1. Build the LiDAR-point dictionary: load full `pcl.npy` (~50K-200K points, depending on scene), apply scene FOV mask (loose; e.g., union of all train-frame FOVs to keep ~80K candidates).
2. Add per-point features `f_p ∈ ℝ^{M_total × 32}` as a learnable parameter.
3. Add the σ-MLP `θ_σ`: 3-layer MLP, [pose_enc + f_p + pos_xyz] → 1, sigmoid output. Pose encoded as raw 6-vec (translation + Euler) + sinusoidal positional encoding (NeRF-style, 6 frequencies → 36 features).
4. Modify `render_gaussians` to multiply per-point `vertex_areas` by `sigma_p`.
5. Optimizer groups: `f_p_lr = 5e-3`, `mlp_sigma_lr = 1e-3`, weight decay 1e-5 on the MLP.
6. Loss: `mse_raw(|RA|) + λ_sparsity · L1(σ) / N`. Tune `λ_sparsity ∈ {1e-4, 1e-3, 1e-2}` so 10-30% of points are above σ = 0.5 at convergence.
7. **Phase gate:** 6-scene held-out bench. Target: |RA| test mean ≥ 0.65 (vs v5_v2 Phase 4's ~0.60).

### Phase 8 — combined sampler + amplitude-only position deltas (Directions #2 + #3)

**Effort:** 2-3 weeks.

1. Add `pos_delta ∈ ℝ^{M_total × 3}` parameter, initial 0.
2. Compute `positions_active = lidar_pcl + bound_mm · tanh(pos_delta / bound_mm)` with `bound_mm = 3`.
3. Modify `rasterizer_factorized.py` (PyTorch path; the fused kernel doesn't need changes) to split `positions_phase` / `positions_amp` per §3.3. Keep `phi_carrier.detach()` and `n_peak.detach()` for kernel compatibility.
4. Optimizer group: `pos_delta_lr = 5e-5` (10× smaller than v5_v2 plan's recommendation, conservative). `λ_pos = 0.01` L2 reg.
5. Warm-start: freeze `pos_delta` for first 100 iters; unfreeze gradually.
6. **Phase gate:** 6-scene no-holdout + held-out benches. Target:
   - **No-holdout |RA| train mean ≥ 0.90** (P0)
   - **Held-out |RA| test mean ≥ 0.70** (P0)

### Phase 9 — sweep + ablations (1-2 weeks)

| variant | what's tested |
|---|---|
| Phase 7 only (no positions) | sampler contribution |
| Phase 8 (sampler + position deltas) | combined effect |
| Phase 8 minus σ (visibility off, deltas on) | position contribution alone |
| pose encoding: raw vs sinusoidal | input feature design |
| `λ_sparsity` ∈ {1e-4, 1e-3, 1e-2} | sparsity strength |
| `bound_mm` ∈ {1, 3, 10} | position freedom |
| MLP_σ size: {2×64, 3×64, 3×128} | sampler capacity |
| M_total ∈ {30K, 80K, 200K} | dictionary size |

---

## §6 Related work — verified citations

### Core 3D representations (carried from v5_v2 plan)

1. **3D Gaussian Splatting** — B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, ACM Trans. Graph. **42(4)**, SIGGRAPH 2023. arXiv:[2308.04079](https://arxiv.org/abs/2308.04079).

2. **NeRF** — B. Mildenhall, P. Srinivasan, M. Tancik, J. Barron, R. Ramamoorthi, R. Ng, ECCV 2020. We adopt the *positional encoding* (sinusoidal frequency embedding) from §5.1 for our pose-encoding branch.

3. **Instant-NGP** — T. Müller, A. Evans, C. Schied, A. Keller, ACM Trans. Graph. **41(4)**, SIGGRAPH 2022. Optional: replace per-point feature `f_p` with a multi-resolution hash-grid lookup `HashGrid(p_xyz)` for spatial smoothness.

4. **Point-NeRF** — Q. Xu, Z. Xu, J. Philip, S. Bi, Z. Shu, K. Sunkavalli, U. Neumann, CVPR 2022. The closest precedent for *per-point learnable features feeding a shared MLP decoder*. We use the same architectural pattern but extend to pose-conditioned σ_p output.

5. **Mip-NeRF 360** — J. Barron et al., CVPR 2022. Anti-aliasing context.

6. **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction** — Z. Yang, H. Yang, Z. Pan, L. Zhang, CVPR 2024. **Direct architectural precedent for our σ-MLP:** Yang et al.'s deformation MLP takes `(t)` and outputs per-Gaussian δposition, δrotation, δscale offsets at time t; our σ-MLP takes `(pose)` and outputs per-point opacity at pose. Same MLP design pattern.

### Differentiable point-cloud sampling

7. **Categorical Reparameterization with Gumbel-Softmax** — E. Jang, S. Gu, B. Poole, ICLR 2017. arXiv:[1611.01144](https://arxiv.org/abs/1611.01144). The continuous relaxation of categorical sampling that makes "subset selection" differentiable end-to-end. We use the related sigmoid relaxation (each point's σ_p is independently sigmoidal — equivalent to per-point Bernoulli with relaxation), which is the simpler 2-class case of Gumbel-Softmax.

8. **SampleNet: Differentiable Point Cloud Sampling** — I. Lang, A. Manor, S. Avidan, CVPR 2020 (Oral). arXiv:[1912.03663](https://arxiv.org/abs/1912.03663). Code: [itailang/SampleNet](https://github.com/itailang/SampleNet). **Direct precedent.** They present a differentiable relaxation for sampling K points from an input cloud; the sampled points are approximated as a soft mixture of points in the primary input cloud. Our σ_p formulation is functionally equivalent: the soft-mixture weight on each candidate point is differentiable. SampleNet's downstream task is classification/reconstruction; ours is rendering.

9. **Differentiable Patch Selection for Image Recognition** — J.-B. Cordonnier, A. Mahendran, A. Dosovitskiy, D. Weissenborn, J. Uszkoreit, T. Unterthiner, CVPR 2021. arXiv:[2104.03059](https://arxiv.org/abs/2104.03059). Differentiable Top-K via perturbed optimisers. Directly applicable if we want to enforce a hard size budget (top-K active points per pose). We treat this as an optional alternative to the L1-sparsity formulation in Phase 7 step 6.

10. **Categorical Reparameterization with Gumbel-Softmax** — already in [7]; mentioned again here because the *Concrete distribution* formulation by C. Maddison, A. Mnih, Y. Teh, ICLR 2017, is the parallel work usually cited together.

### Radar-specific neural rendering (carried from v5_v2 plan)

11. **DART: Implicit Doppler Tomography for Radar Novel View Synthesis** — T. Huang et al., CVPR 2024. arXiv:[2403.03896](https://arxiv.org/abs/2403.03896). NeRF-style implicit representation. Differentiator: ours preserves explicit Gaussians + coherent path sum.

12. **Radar Fields: Frequency-Space Neural Scene Representations for FMCW Radar** — D. Borts, E. Liang et al., SIGGRAPH 2024. arXiv:[2405.04662](https://arxiv.org/abs/2405.04662). Volumetric implicit field on FFT'd waveforms. Differentiator: ours is explicit.

13. **RadarSplat: Radar Gaussian Splatting** — C.-L. Kung et al., ICCV 2025. arXiv:[2506.01379](https://arxiv.org/abs/2506.01379). Explicit Gaussian splatting for radar. Differentiator: ours adds pose-conditioned sampling + amplitude-only position learning + interpretable ITU materials.

### Phase / coherent-rendering precedent

14. **Coherent integration in mmWave / FMCW MIMO**: Richards, M.A., *Fundamentals of Radar Signal Processing*, McGraw-Hill, 2nd ed. 2014. The standard textbook treatment of why coherent-sum amplitude is well-conditioned in position whereas phase is multi-valued mod 2π.

15. **Texas Instruments**, *"Signal Processing for TDM MIMO FMCW Millimeter-Wave Radar Sensors,"* application note. Contains the exact phase-noise sensitivity analysis at 77 GHz that motivates Direction #3. (User has the PDF in `md/`.)

### Hyper-network / pose-conditioning precedent (background)

16. **HyperNetworks** — D. Ha, A. Dai, Q. V. Le, ICLR 2017. arXiv:[1609.09106](https://arxiv.org/abs/1609.09106). The general framework of using one network's output to parameterise another's behaviour conditioned on input metadata (here: pose). Conceptual ancestor of pose-conditioned generation in NeRF / 3DGS / our σ-MLP.

---

## §7 Validation and risks

### Validation gates

Each phase must clear before advancing:

1. **No regression on v5 / v5_v2 single-frame fit** — single-frame |RA| ≥ 0.95 (rules out a forward-model bug).
2. **No regression on no-holdout train** — never lower than the prior phase's no-holdout |RA| train.
3. **Random-seed variance** — std across 3 seeds on `seq_0_frame_135` < 0.03.

### Risks

| risk | probability | mitigation |
|---|---|---|
| σ-MLP collapses to σ_p = 1 ∀ p | high without sparsity loss | Tune `λ_sparsity` to keep ~20% active at convergence; log σ histogram every 50 iters. |
| σ-MLP overfits to train poses, fails to interpolate to test pose | medium | Pose encoding = sinusoidal (Mildenhall et al. NeRF [2]) for smooth interpolation. Plus: weight decay on MLP. |
| Amplitude-only position grad still too noisy in practice | medium | Bounded-delta parametrisation + small LR + warm-start freeze. Fallback: Phase 7-only (no Phase 8 positions). |
| Two `positions` paths in renderer slow down training | low | Both paths re-use the same upstream tensors; only the BSDF / antenna evaluations are duplicated. ~1.3-1.5× per-iter cost expected. |
| MLP-conditioned σ + densification (v5_v2 Phase 4) interact badly | low | Only MLP σ in Phase 7; densification is a v5_v2-Phase-4 concern that this plan inherits but does not modify. |
| 0.7 test target still unreachable | medium | Document the achieved ceiling + frame paper as bounds-paper (carryover from `md/diagnostics/RESULTS.md` framing). |

### When to abort

If Phase 7 alone gives < +0.03 on |RA| test vs v5_v2 Phase 4, the visibility-attention hypothesis is wrong on this dataset class — re-evaluate before Phase 8. Phase 8 is then unlikely to help either.

If Phase 8 gives < +0.02 on |RA| test vs Phase 7, position deltas are not the binding constraint — write up Phase 7 alone.

---

## §8 Timeline

Cumulative on top of v5_v2 baseline:

| phase | effort | gate |
|---|---|---|
| Phase 7 — sampler MLP | 2 weeks | held-out |RA| test ≥ 0.65 |
| Phase 8 — + amplitude-only positions | 2-3 weeks | **no-holdout ≥ 0.90 AND held-out ≥ 0.70** (P0) |
| Phase 9 — sweep + ablations | 1-2 weeks | A1-A8 ablation matrix |

**Total v5_v3 increment: 5-7 weeks** on top of v5_v2 (which is itself 6-8 weeks).

---

## §9 Paper framing implication

If Phase 7 + Phase 8 hit their gates, the paper's mechanism story becomes:

> "We observe that radar NVS at the **frame level** (frames 0.2 s apart) is fundamentally harder than at the **chirp level** (chirps 0.5 ms apart) because frame-to-frame the radar has moved by ~25 cm — a sparse, viewpoint-distinct measurement — whereas chirp-to-chirp the view is essentially identical. A shared point-cloud + per-point material representation that easily fits all chirps within a single frame becomes overburdened across frames.
>
> We address this with two complementary mechanisms:
>
> (1) A **pose-conditioned differentiable sampler** that learns a per-point pose-dependent visibility/opacity score over a fixed LiDAR-prior dictionary. Each pose effectively renders from a *unique soft-active subset* of the LiDAR pcl, while materials remain shared across the global dictionary. The sampler MLP is end-to-end-differentiable, builds on Point-NeRF [4], Deformable 3D Gaussians [6], and SampleNet [8].
>
> (2) **Amplitude-only position gradients**, exploiting the fact that the coherent forward model is *smooth* in position via the amplitude path (1/d, antenna gain, BSDF cos terms) but *2π-multi-valued* via the phase path. We carefully partition the renderer's position-dependent quantities so position gradients flow only through amplitude — yielding a well-conditioned signal that finds physically-plausible sub-mm position refinements without phase-noise pathology.
>
> Together, on the 6-scene ColoRadar benchmark, these mechanisms lift |RA| held-out test CC from 0.54 (v5 baseline) to 0.7+ (target), exceeding the naive interpolation baseline of 0.65 and approaching the F±1 inter-frame physical coherence ceiling of 0.59 we measured on 5 Hz cascade sampling."

This is a substantively novel contribution distinct from RadarSplat [13] (no pose-conditioned visibility, no coherent forward model with split-amplitude-vs-phase gradient handling) and DART [11] / Radar Fields [12] (volumetric/implicit, no LiDAR-prior dictionary).

---

## §10 Immediate next action

This document is the plan. Upon agreement, kickoff begins with Phase 7 step 1 (LiDAR-dictionary loader), gated against v5_v2 Phase 6 having completed.

If v5_v2 Phases 0-6 are not yet done, this plan is queued behind them.

---

## Appendix — full citation list (BibTeX-ready)

```
[1]  Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.
     "3D Gaussian Splatting for Real-Time Radiance Field Rendering."
     ACM Trans. Graph. 42(4), SIGGRAPH 2023.
     arXiv:2308.04079 — https://arxiv.org/abs/2308.04079

[2]  Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T.,
     Ramamoorthi, R., Ng, R.
     "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."
     ECCV 2020.

[3]  Müller, T., Evans, A., Schied, C., Keller, A.
     "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding."
     ACM Trans. Graph. 41(4), SIGGRAPH 2022.

[4]  Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.
     "Point-NeRF: Point-based Neural Radiance Fields."
     CVPR 2022.

[5]  Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.
     "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields."
     CVPR 2022.

[6]  Yang, Z., Yang, H., Pan, Z., Zhang, L.
     "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction."
     CVPR 2024.

[7]  Jang, E., Gu, S., Poole, B.
     "Categorical Reparameterization with Gumbel-Softmax."
     ICLR 2017.
     arXiv:1611.01144 — https://arxiv.org/abs/1611.01144

[8]  Lang, I., Manor, A., Avidan, S.
     "SampleNet: Differentiable Point Cloud Sampling."
     CVPR 2020 (Oral).
     arXiv:1912.03663 — https://arxiv.org/abs/1912.03663
     Code: https://github.com/itailang/SampleNet

[9]  Cordonnier, J.-B., Mahendran, A., Dosovitskiy, A., Weissenborn, D.,
     Uszkoreit, J., Unterthiner, T.
     "Differentiable Patch Selection for Image Recognition."
     CVPR 2021.
     arXiv:2104.03059 — https://arxiv.org/abs/2104.03059

[10] Maddison, C.J., Mnih, A., Teh, Y.W.
     "The Concrete Distribution: A Continuous Relaxation of Discrete
     Random Variables."  ICLR 2017.

[11] Huang, T., Miller, J., Prabhakara, A., Jin, T., Laroia, T.,
     Kolter, Z., Rowe, A.
     "DART: Implicit Doppler Tomography for Radar Novel View Synthesis."
     CVPR 2024, pp. 24118-24129.
     arXiv:2403.03896 — https://arxiv.org/abs/2403.03896
     Code: https://github.com/WiseLabCMU/dart

[12] Borts, D., Liang, E., Broedermann, T., Ramazzina, A., Walz, S.,
     Palladin, E., Sun, J., Brüggemann, D., Sakaridis, C., Van Gool, L.,
     Bijelic, M., Heide, F.
     "Radar Fields: Frequency-Space Neural Scene Representations for FMCW Radar."
     SIGGRAPH 2024 Conference Papers.
     arXiv:2405.04662 — https://arxiv.org/abs/2405.04662

[13] Kung, C.-L. et al.
     "RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis
     and 3D Reconstruction of Autonomous Driving Scenes."
     ICCV 2025.
     arXiv:2506.01379 — https://arxiv.org/abs/2506.01379

[14] Richards, M.A. "Fundamentals of Radar Signal Processing."
     McGraw-Hill, 2nd ed. 2014.

[15] Texas Instruments. "Signal Processing for TDM MIMO FMCW Millimeter-Wave
     Radar Sensors." App. note. (PDF in repo md/.)

[16] Ha, D., Dai, A., Le, Q.V.
     "HyperNetworks."
     ICLR 2017.
     arXiv:1609.09106 — https://arxiv.org/abs/1609.09106
```
