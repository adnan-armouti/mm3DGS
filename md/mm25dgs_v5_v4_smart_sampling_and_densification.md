# mm25DGS_v5_v4 — Smarter Sampling: Importance-Aware Initialization, Gradient-Driven Densification, and Selective View-Membership

**Date:** 2026-04-23
**Scope:** Capacity unlock through better point-set selection — initialization + train-time densification + per-view membership. **No MLPs**: this plan stays within the user's "no learning beyond materials/normals/positions" principle.
**Status of sister plans:** `md/mm25dgs_v5_v2_neural_gaussian_radar_field.md` and `md/mm25dgs_v5_v3_pose_conditioned_sampler.md` are **shelved**. Only v5_v2 Phase 0 (directory clone + sanity) and v5_v2 Phase 1 (learnable positions, **amplitude-only gradient**) carry over from v5_v2; everything else from those plans is dropped.
**Performance goals:**
- No-holdout multi-frame mean **|RA| train CC > 0.9** (currently 0.82)
- Held-out **|RA| test CC > 0.7** (currently 0.54)

---

## §0 Position summary on the four user proposals

> "[Direction A] Are we sampling correctly at initialization? Is FPS the wrong approach? FOV culling is good, but we take no positive steps after that (cosine hemisphere, per-pose normal alignment, etc.)."

**Strongly agree.** FPS gives spatial uniformity, which is a good prior for cameras (every pixel of every view should be covered). Radar has very different physics: amplitude scales with `cos θ_i / d²`, so the points that matter for fitting a given pose are *cosine-weighted, near-range, on-bore-axis* — and the points that matter for fitting *all* train poses jointly is a non-trivial union. Our current v5 init does:

1. FOV cull at the seed pose (mesh raycast).
2. RX visibility raycast.
3. **Cosine-importance resample (already does this)** — but at the *seed pose only*.
4. FPS to N = 20 K.

Step 3 already takes cosine into account but only for one pose. The natural fix is to compute importance as a **union over all train poses**, weighted by amplitude expected at each pose. Step 4 (FPS) then becomes wasteful — it spreads points uniformly in space when we want them concentrated at high-importance locations.

> "[Direction B] Differentiable sampling / GS-style densification (SampleNet, Mini-Splatting, Taming 3DGS)."

**Strongly agree on the densification side; partial agreement on the differentiable-sampling side.** 3DGS's densify/clone/split/prune machinery (and its budgeted/guided refinements in Mini-Splatting [2] and Taming 3DGS [3]) is explicitly designed for the same problem we have: a fixed-budget primitive set that must explain multiple views, where some primitives are essential and others are wasted capacity. We should adopt this directly. The differentiable-sampling-at-init angle (SampleNet [1]) is interesting as an *alternative initialization scheme*, but we don't strictly need it — a careful non-learned importance sampler probably gets us most of the way at zero learning risk.

> "[Direction C] Enable position gradients to *guide selection* (not movement)."

**Strongly agree.** This is exactly how 3DGS's densification works in the original Kerbl et al. paper [4]: `‖∇_p L‖` is the gradient magnitude on Gaussian *position*, which is used as the **densification signal** — Gaussians with high positional gradient get split, those with low contribution get pruned. The positions can be either moved or kept frozen; the gradient magnitude is a useful selection signal regardless. Switching from our current Fisher-info-from-Adam-`exp_avg_sq` proxy to direct positional gradient magnitude is a clean upgrade. This is fully compatible with our detached-phase / frozen-positions regime — we just compute the gradient through amplitude.

> "[Direction D] Selective per-view membership: middle ground between global and per-view-unique sets."

**Agree as a diagnostic, not as a deployment mechanism.** A direct per-train-view membership table `m_p,v ∈ {0,1}^{N×V}` does not generalize to test poses — the test pose has no training-time membership entry. To deploy at test, the simplest fall-back is KNN-over-train-views (average the K nearest train poses' membership vectors). Per-view membership is most useful as an **analysis tool** to characterise *how view-specific* the optimal point sets actually are — that characterisation is what the v5_v3 MLP plan was trying to address structurally; if D shows mostly-global behaviour, the MLP isn't needed.

The rest of this document lays out a phased plan that combines Directions A + B + C as the **near-term capacity unlock** (no MLP, no learning beyond materials/normals/positions/membership), and treats Direction D as a diagnostic only.

---

## §1 Empirical motivation

Re-stating the gap from `md/diagnostics/RESULTS.md` and the user's chirp-vs-frame observation:

| training regime                       | |RA| train | |RA| test |
|---|---:|---:|
| single-frame fit (1 frame, all 16 chirps) | ~0.95 | 0.80–0.85 (held-out chirp) |
| multi-frame v5 (8 train + 1 test) | 0.83 | 0.54 |
| multi-frame v5 + v_ego refinement (current best) | 0.68–0.78 | 0.59 |

The **shared-representation penalty** — single-frame 0.95 → multi-frame train 0.83 — is what this plan targets. The hypothesis is that we are wasting capacity on points that are not *jointly informative* across the 8 train frames, and we are missing capacity at locations where the union-of-train-poses needs more density. Direction A fixes this at init; Direction B + C fix it during training. Direction D is a discrete instrumentation of how the model wishes to slice up its budget per pose.

---

## §2 Direction A — Initialization beyond FPS

### §2.1 Why FPS is the wrong default for radar NVS

FPS (Eldar et al. 1997 [11]) has one optimization criterion: maximize the minimum pairwise distance. This makes sense for camera views where every pixel of every view should be covered — a *spatial* uniformity prior. But for radar:

1. Per-pose amplitude scales as `√G_tx(direction) · √G_rx(direction) · √f_BSDF · cos θ_i / (d_tx · d_rx)`. A LiDAR point 30 m away contributes ~10× less amplitude than one at 10 m. FPS treats them equally; importance sampling does not.
2. A LiDAR point at glancing incidence (cos θ_i ≈ 0) contributes near-zero coherent amplitude. FPS still keeps it; cosine-importance sampling de-prioritizes it.
3. A point that is FOV-visible from only 1 of 8 train poses is *less useful* than a point visible from all 8. FPS treats them equally; union-importance sampling weights toward jointly-visible points.

We already implement (2) at the seed pose. The fix is to do (1)+(2)+(3) as a union over train poses.

### §2.2 Proposed initialization: amplitude-weighted union-importance sampling

For each candidate LiDAR point p (after raycast FOV + RX-visibility cull at seed pose, retaining ~50K-200K candidates):

```
score(p) = Σ_{F in train_frames} 1[p visible at F] · A(p, F)

  where  A(p, F) = max(0, ⟨n_p, û_p→radar(F)⟩)              # cos θ_i ≥ 0
                · G_tx(û_p→radar(F))                         # antenna gain
                · 1 / d_p→radar(F)²                          # 1/r² falloff
                · I_lidar(p)                                 # LiDAR intensity prior (column 7 of pcl.npy)
```

Then sample `N = target_n` points without replacement, with probability proportional to `score(p)`. This gives:

- Points visible from many train poses get more weight (jointly informative).
- Per-pose visibility weighted by cosine + range + antenna gain (radar-correct importance).
- Surfaces parallel to the line-of-sight from the radar are downweighted (correct: their coherent contribution is small).
- LiDAR intensity adds a per-point physical prior (memory: use it [memory: feedback_lidar_intensity_for_edge_smoothness]).

### §2.3 LiDAR intensity prior

`pcl.npy` has a 7th column (intensity / reflectance score from the LiDAR sensor itself). High intensity = strong scatterer. We should multiply this in directly to `score(p)`. This is a **free** physical prior we are currently not using for sampling (we use it only for edge regularization elsewhere).

### §2.4 Output of this stage

A candidate-point set of size `target_n` that is *jointly-amplitude-aware* across train poses, where every point has nonzero predicted contribution to at least one train pose, and points are concentrated where the union-of-views says signal lives.

This is the new `init_visible_weighted_radar_aware()` in v5_v4 — a drop-in replacement for v5's `init_visible_weighted()`.

### §2.5 Comparison ablation

| method | description |
|---|---|
| v5 baseline (current) | FOV + RX visibility + seed-pose cosine resample + FPS |
| **A1**: FPS removed | replace FPS with random sample (no spatial-uniformity prior at all) |
| **A2**: union-cosine | union over train poses, weight by cos θ_i only |
| **A3**: union-amplitude | union over train poses, weight by full amplitude `cos θ · G · 1/d²` |
| **A4** (proposed default): union-amplitude + LiDAR intensity | A3 plus column-7 reflectance prior |
| A5: A4 followed by light FPS (e.g., FPS to 1.5×N then cosine-amplitude resample to N) | hybrid: union-importance with mild spatial dispersion |

A1 vs baseline isolates the contribution of FPS itself (likely small, possibly negative). A2 vs A3 isolates antenna+range. A3 vs A4 isolates LiDAR intensity. A5 tests whether some spatial dispersion is still useful as a regularizer.

### §2.6 Success criterion for Direction A

- 6-scene no-holdout |RA| train mean ≥ 0.84 (vs v5 baseline 0.82) — a low bar; if A4 doesn't get +0.02, the init choice isn't binding.
- More importantly: visualization of the sampled point cloud overlaid on the LiDAR mesh, with per-pose amplitude-contribution heatmap, to confirm the sampler is doing what we expect.

---

## §3 Direction B — Differentiable sampling + 3DGS-style densification

### §3.1 What translates from 3DGS to radar (and what doesn't)

3DGS's densification recipe from Kerbl et al. [4] §5.2:

| 3DGS step | translates to radar? | how |
|---|---|---|
| ∇_position L magnitude | **yes, directly** | gradient flows through amplitude path even with detached phase (§4) |
| split high-grad Gaussians (clone with smaller scale) | **yes, directly** | scale = isotropic Hann-PSF spread (anisotropic-scale extension is a deferred follow-up) |
| prune low-opacity Gaussians | **yes, with adaptation** | "opacity" = our continuous σ_p (§5) or Adam exp_avg_sq (current proxy) |
| reset opacity periodically | **probably yes** | counter against premature commitment |

What does NOT translate directly: 3DGS's tile-based rasterizer with α-blending and screen-space gradients. Our renderer is coherent-sum splat with FFT — no α-blending on screen, no screen-space gradient. So Mini-Splatting's "blur split" (which detects screen-space blur from too-large Gaussians) needs adaptation: in radar we'd detect *range-bin blur* (a Gaussian whose range-PSF spread is too wide because its 3D scale projected onto the radar-line-of-sight is too large).

### §3.2 Mini-Splatting [2] mechanisms that transfer

Fang & Wang's Mini-Splatting paper introduces three mechanisms to "reorganize the spatial positions of Gaussians":

1. **Blur split:** identify Gaussians whose footprint exceeds a quality threshold and split them. *Radar analog:* identify Gaussians whose range-direction extent exceeds a fraction of the range bin (e.g., σ_r > 0.5 · range_resolution) and clone-split into two narrower Gaussians.
2. **Depth reinitialization:** for under-covered regions of depth space, add new Gaussians. *Radar analog:* for under-covered regions of `(range, azimuth)` space (where GT amplitude is high but pred amplitude is low — visible as `(GT − pred)` residual map), spawn new Gaussians at the corresponding 3D back-projection.
3. **Intersection-preserving simplification:** prune redundant Gaussians while preserving the intersection set with rays. *Radar analog:* prune Gaussians whose contribution to every train-pose's |RA| is bounded above by another Gaussian's (i.e., redundant).

### §3.3 Taming 3DGS [3] mechanisms that transfer

Mallick & Goel et al.'s Taming 3DGS introduces:

1. **Guided, purely constructive densification:** only densify Gaussians that *raise* reconstruction quality, not just those with high gradient. *Radar analog:* only split a Gaussian if a small simulation says the new pair fits the GT amplitude residual better than the original — this is a 2-step lookahead.
2. **Exact primitive budgeting:** hard cap on N_total. *Radar analog:* cap at e.g. 30 K Gaussians; when at budget, every split must be matched by a prune.
3. **Flexible sample guiding:** use rendered-quality residuals to direct where new primitives go (similar to Mini-Splatting's depth reinit). *Radar analog:* the `(range, azimuth)` residual map is the guidance signal.

These are higher-effort improvements than basic 3DGS densification. Recommend implementing basic 3DGS first (§3.4), then layering Mini-Splatting then Taming 3DGS as needed.

### §3.4 Proposed v5_v4 densification schedule (basic)

```
Every iter, accumulate per-point:
    g_p = ‖∇_position L‖₂   (computed via amplitude-only path; see §4)

Every K = 100 iter (between iter 100 and iter 400):
    1. SPLIT top quantile (top 10% by g_p): clone Gaussian into two
       with smaller scale (× 0.6 each), inherit features. Position
       perturb by N(0, σ_split = 0.5 mm).
    2. PRUNE bottom decile (bottom 10% by max σ_p_pose contribution
       summed across train poses): drop entirely.
    3. Maintain budget cap N_max = 30 K (split + prune are matched).

Every K_reset = 200 iter:
    Reset σ_p (or opacity) to a moderate prior, encouraging
    re-discovery of which points matter.
```

Adam state for added/removed Gaussians: when adding, init exp_avg/exp_avg_sq to 0; when removing, drop the corresponding rows. This is the standard 3DGS Adam-state-management pattern.

### §3.5 Differentiable sampling at init (SampleNet [1] adaptation)

SampleNet's recipe: train a small network that maps an input point cloud to a sampled subset, where each sample is a soft mixture of input points; the soft mixture is differentiable. The trained network's output is then used as the down-sampled cloud.

For our case, the "downstream task" is the radar reconstruction loss. SampleNet would let us learn the *initialization* end-to-end: gradients from the training loss flow back into a sampler net which learns *which* LiDAR points to keep.

This is **strictly more powerful but strictly more risk** than the non-learned union-amplitude sampler in §2 — and the user has explicitly opted out of MLP-based learning beyond materials/normals/positions for this cycle. **Skip SampleNet at init.** The non-learned union-amplitude sampler in §2 is the chosen replacement.

---

## §4 Direction C — Position gradients for *selection*, not movement

### §4.1 Why this is different from "learnable positions"

The carryover Phase 1 (§6.4) proposes bounded position deltas via amplitude-only gradient — that's *learnable positions* — actually move the points by sub-mm for better fit.

Direction C proposes something different and *complementary*: positions stay frozen at LiDAR locations, but we still **compute** the position gradient and use the *magnitude* as a selection signal — high-grad points are informative; low-grad points are wasting budget.

This is closer to 3DGS's mechanism than what v5 currently does. v5 uses Adam's `exp_avg_sq` (snapshot of the squared-gradient running average — a Fisher-info proxy from the optimizer state) summed across material params and rotation params. That conflates *material learnability* with *positional importance*. Direct position gradient magnitude is a cleaner signal for "does this point's location matter for the loss?"

### §4.2 Implementation

```python
# In rasterizer_factorized.py — add an instrumentation flag
def render_factorized(positions, ..., capture_position_grad=False):
    if capture_position_grad:
        positions = positions.clone().detach().requires_grad_(True)
        # Use these positions in the AMPLITUDE path only
        # (phase path always uses positions.detach() — no change)
        positions_amp = positions
    ...
    # The model parameters' loss does NOT update positions; this is
    # a hook to compute the gradient for selection only.

# In train_frame_nvs.py main loop:
for it in range(num_iters):
    optimizer.zero_grad(set_to_none=True)
    for sample in train_samples:
        ra_pred = render(model, ..., capture_position_grad=True)
        loss = compute_loss(ra_pred, sample['gt'])
        loss.backward(retain_graph=True)
        # Capture per-point position gradient AFTER backward
        with torch.no_grad():
            grad_pos_norm = positions.grad.norm(dim=-1)  # (N,)
            # Accumulate into a running tally for densification
            running_grad_pos[per-point] += grad_pos_norm

    # Position params remain frozen — we did NOT update positions.
    # We only updated normal/material/feature params.
    optimizer.step()

    if it % 100 == 0 and 100 ≤ it ≤ 400:
        densify_and_prune(model, running_grad_pos)
        running_grad_pos.zero_()
```

The `positions.grad` exists because we evaluated the renderer with `requires_grad_(True)` — but we never update positions (no optimizer group on them). We only use `grad_pos_norm` as a *selection signal*.

### §4.3 Compatibility with detached phase

We compute `positions.grad` through the **amplitude path only**: `phi_carrier` and `n_peak` stay detached (so the existing fused step5 CUDA kernel works unchanged); the position-dependent quantities `d_tx`, `d_rx`, `wi`, `wo`, antenna gain `G(direction)`, and BSDF cos terms are NOT detached, so the gradient flows back to position through these smooth amplitude factors. This sidesteps the phase-noise pathology — we do NOT take the gradient of the loss w.r.t. position via the phase term. So `grad_pos_norm` is a clean, well-conditioned scalar per point, with no 2π wrapping issues.

### §4.4 Comparison vs current `exp_avg_sq` proxy

| signal | current v5 use | proposed v5_v4 use |
|---|---|---|
| Adam `exp_avg_sq` summed over material params | "active points mask" for top-frac% (current S1 reg) | retire — too coarse |
| Adam `exp_avg_sq` summed over rotation params | rotation Fisher for S2 reg | keep — that's a different purpose |
| ‖∇_p L‖ (amplitude path) per point | not used | **densification + prune signal** |
| σ_p (continuous opacity proxy, e.g., from `\|w_full\|` magnitude) | not used | **prune signal** |

The cleaner path is: use `‖∇_p L‖` for *split* and `σ_p` (or contribution magnitude `|w_full|`) for *prune*. They are different signals for different decisions.

---

## §5 Direction D — Selective per-view membership

### §5.1 Architecture

For each (point, view) pair, a learnable scalar `m_p,v ∈ ℝ`:

- View `v` indexes a train pose (one of the 8 train frames in our held-out bench).
- `m_p,v` is unconstrained; we use `sigmoid(m_p,v) ∈ [0, 1]` as the per-view opacity multiplier.
- Memory: `N × V × 4 bytes = 20K × 8 × 4 ≈ 640 KB` — trivial.

### §5.2 Training signal

```
For train sample (frame F, chirp 0):
    v = train_view_index_of(F)
    areas_F = base_areas · sigmoid(m[:, v])
    rendered_F = render(model, areas=areas_F, ...)
    loss_F = mse_raw(rendered_F, GT_F)
loss_total = sum_F loss_F + λ_member · L1(sigmoid(m))   # encourage few-view membership
```

Each view independently learns which points it wants to use. The L1 sparsity term encourages each point to be "owned by" few views — but if the loss signal genuinely needs a point in many views, the sparsity loss is overruled.

### §5.3 What we learn from D

After convergence, we can plot the matrix `sigmoid(m)`:

- Points with σ ≈ 1 in all 8 views: "globally useful" — these are the canonical scatterers.
- Points with σ varying across views: "view-specific" — the LiDAR pcl describes a static scene but radar sees view-specific glints (small geometric variations + speckle).
- Points with σ ≈ 0 in all views: "wasted budget" — should be pruned outright.

The fraction of each category is a diagnostic about how rigid the global-shared-representation assumption is. If most points are ~globally-useful, the global-shared architecture is correct and v5_v4 alone is sufficient. If most points are view-specific, that's a flag that a future deployment-time mechanism (e.g. KNN-over-train-views averaging, or a smooth pose-conditioned predictor) would be valuable — but that decision is *outside* this plan.

### §5.4 Why D doesn't deploy at test

Test pose has no `v` index; there's no `m_p,test`. To deploy at test, the simplest fall-back is **KNN-over-train-views:** for test pose F_test, find the K nearest train poses by 6-DoF distance and average their `m_p,v`. Quick to implement; smooth at the K-boundary.

Recommend: D as **diagnostic only** in this plan. The KNN-deployment is a small follow-up (~3 days) if D analysis shows it's worth doing.

---

## §6 Implementation phases

The full phase list (with cross-plan references) is in §10. This section details the v5_v4 phases. v5_v2 Phase 0 (directory clone) is a prerequisite; v5_v2 Phase 1 (learnable positions, amplitude-gradient-only) is gated on the v5_v4 P0 result and detailed in §6.4.

### Phase 2 — Initialization (Direction A)

**Effort: 1 week.**

1. Implement `init_visible_weighted_radar_aware()` per §2.2.
2. LiDAR intensity prior from `pcl.npy` column 7.
3. Five-variant init ablation (A1-A5 in §2.5). 6-scene no-holdout train CC reported.
4. Pick the best init; replace `init_visible_weighted` calls.

**Phase gate:** no-holdout |RA| train mean ≥ 0.84 (+0.02 over v5 baseline 0.82). If it doesn't, the init isn't the binding constraint at this scale.

### Phase 3 — 3DGS-style densification with position-gradient selection (Directions B + C)

**Effort: 2 weeks.**

1. Add per-iter capture of `‖∇_p L‖` per point. PyTorch autograd, amplitude-only path, **no phase gradient** (positions remain frozen at LiDAR locations; we use the gradient magnitude as a *selection signal* per §4).
2. Implement basic 3DGS densify/prune on the 100-iter cadence per §3.4.
3. Hard budget cap at N_max = 30K (1.5× v5's 20K).
4. Adam state add/remove logic.
5. **Phase gate (P0):** no-holdout |RA| train mean ≥ 0.90.

If 0.90 not reached, layer Mini-Splatting's blur-split + depth-reinit-style residual-guided spawn (§3.2). If still not reached, layer Taming 3DGS's guided-densification 2-step lookahead (§3.3). Each adds ~3-5 days.

### Phase 4 — Selective view membership as diagnostic (Direction D)

**Effort: 3-5 days (instrumentation only).**

1. Add `m ∈ ℝ^{N × V}` parameter. V = 8 train views.
2. Per-view loss with per-view opacity `sigmoid(m[:, v])`.
3. L1 sparsity loss `λ_member · ||sigmoid(m)||_1 / (N × V)`.
4. After training, dump heatmap of `sigmoid(m)` and per-point view-count distribution.

**Phase gate:** none — this is diagnostic, not optimisation. Use the D heatmap to characterise *how view-specific* the optimal point sets are; this characterisation is a future-paper input, not a deployment decision in this plan.

### §6.4 v5_v2 Phase 1 carryover — learnable positions, amplitude-gradient ONLY

**Effort: 2 weeks.** **Only run if Phase 3 misses the P0 train gate (0.90).**

1. Add `pos_delta ∈ ℝ^{N × 3}` parameter, initial 0, bounded by `bound_mm · tanh(pos_delta / bound_mm)` with `bound_mm = 3`.
2. Modify `rasterizer_factorized.py` so position-dependent quantities are split:
   - `positions_phase = positions.detach()` for `n_peak`, `phi_carrier` (phase path stays detached — **NO PHASE GRADIENTS**).
   - `positions_amp = positions` (with grad) for `d_tx`, `d_rx`, `wi`, `wo`, antenna gain `G(direction)`, BSDF cos terms.
3. The fused step5 CUDA kernel needs no changes (it sees only detached `phi_carrier` and `n_peak`).
4. Optimizer group: `pos_delta_lr = 5e-5`, `λ_pos · ||pos_delta||² = 0.01 · ...` regulariser.
5. Warm-start: freeze `pos_delta` for first 100 iters; unfreeze gradually.
6. **Phase gate:** no-holdout |RA| train ≥ 0.92 (over Phase 3's 0.90).

This is morally distinct from Phase 3's use of `‖∇_p L‖`:
- **Phase 3** computes the position gradient and uses its **magnitude** as a *selection signal* — positions stay frozen.
- **Phase 1 (this section)** uses the same amplitude-only gradient to **update positions** — bounded sub-mm refinement.

Both use the amplitude-only-gradient principle from §4; both avoid the 2π phase-wrap pathology.

### §6.5 Held-out test bench

After all enabled phases complete: 6-scene held-out 6-frame-train bench. **P0 gate:** |RA| test mean ≥ 0.70. This is the user's hard target.

---

## §7 Validation gates

Each phase must clear:

1. **No regression on v5 single-frame fit:** any phase that drops single-frame |RA| below 0.95 is rejected (forward model bug).
2. **No regression on no-holdout train CC** between phases.
3. **Random seed std < 0.03** on `seq_0_frame_135` for the post-phase config.
4. **Diagnostic-D coverage gate** (Phase 4 only): the D matrix should show non-trivial per-view variance — if `m_p,v` collapses to constant ∀ v, we have an implementation bug.

---

## §8 Risks

| risk | probability | mitigation |
|---|---|---|
| Union-amplitude init helps less than expected | low | A1-A5 ablation isolates contribution; if even A4 doesn't help, init isn't binding. |
| Densification adds Gaussians where they don't help | medium | Taming 3DGS-style guided densification (§3.3) — only split if 2-step lookahead says quality improves. |
| Position gradient via amplitude path is too noisy | low | Smooth by construction (§4.3); not the same as phase-derived gradient. |
| Memory budget at N_max = 30K hits OOM | low | 30K is 1.5× our current 20K, well within 4090 capacity. |
| D heatmap shows truly per-view scatterers (rare-but-binding) | medium | This is *information*, not a failure: the heatmap motivates a future-cycle KNN-deployment (~3 days) but does not block the v5_v4 P0 gate. |
| Phase 3 densification destabilises the coherent forward model (split positions create phase mismatch) | low | New Gaussians are clones of parents at sub-mm offsets; phase coherence at λ=3.9 mm is preserved within the rigid-body sense. Validation gate 1 catches if it isn't. |

---

## §9 Citations (verified on arXiv / venue sites)

### Differentiable sampling and 3DGS densification (the user's three references)

1. **SampleNet: Differentiable Point Cloud Sampling** — I. Lang, A. Manor, S. Avidan, CVPR 2020 (Oral). arXiv:[1912.03663](https://arxiv.org/abs/1912.03663). Code: [itailang/SampleNet](https://github.com/itailang/SampleNet). Differentiable relaxation: sampled points approximated as soft mixtures of input cloud, enabling end-to-end backprop through subset selection.

2. **Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians** — G. Fang, B. Wang, ECCV 2024. arXiv:[2403.14166](https://arxiv.org/abs/2403.14166). Code: [fatPeter/mini-splatting](https://github.com/fatPeter/mini-splatting). Three mechanisms — *blur split*, *depth reinitialization*, and *intersection-preserving sampling* — to reorganise Gaussian spatial positions for better quality at fixed budget.

3. **Taming 3DGS: High-Quality Radiance Fields with Limited Resources** — S. S. Mallick, R. Goel, B. Kerbl, M. Steinberger, F. V. Carrasco, F. de la Torre, SIGGRAPH Asia 2024 Conference Papers. ACM DOI:[10.1145/3680528.3687694](https://dl.acm.org/doi/10.1145/3680528.3687694). arXiv:[2406.15643](https://arxiv.org/abs/2406.15643). Code: [humansensinglab/taming-3dgs](https://github.com/humansensinglab/taming-3dgs). *Guided, purely constructive densification* under a hard budget cap; 4–5× reduction in model size and training time vs vanilla 3DGS at competitive quality.

### Foundational 3DGS / NeRF

4. **3D Gaussian Splatting for Real-Time Radiance Field Rendering** — B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, ACM Trans. Graph. **42(4)**, SIGGRAPH 2023. arXiv:[2308.04079](https://arxiv.org/abs/2308.04079). The original densify/clone/split/prune recipe.

5. **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** — B. Mildenhall et al., ECCV 2020. The foundational neural-field paper.

6. **Instant Neural Graphics Primitives with a Multiresolution Hash Encoding** — T. Müller et al., ACM Trans. Graph. **41(4)**, SIGGRAPH 2022.

7. **Point-NeRF: Point-based Neural Radiance Fields** — Q. Xu et al., CVPR 2022. Per-point feature + shared MLP.

### Sampling theory and Monte Carlo

8. **Eldar, Y., Lindenbaum, M., Porat, M., Zeevi, Y. Y.** *"The farthest point strategy for progressive image sampling."* IEEE Trans. on Image Processing **6(9)**, 1305-1315, 1997. The original FPS algorithm.

9. **Pharr, M., Jakob, W., Humphreys, G.** *Physically Based Rendering: From Theory to Implementation*, 4th ed., 2023. Canonical reference for Monte Carlo importance sampling, cosine-hemisphere PDFs (§13), and BRDF importance sampling. Available online: [pbr-book.org](https://www.pbr-book.org).

### Concrete / Gumbel for selection (useful for D)

10. **Categorical Reparameterization with Gumbel-Softmax** — E. Jang, S. Gu, B. Poole, ICLR 2017. arXiv:[1611.01144](https://arxiv.org/abs/1611.01144). Continuous relaxation of categorical sampling. Direct relaxation of binary `m_p,v` if we want hard membership.

### Radar-specific neural representation (context)

11. **DART: Implicit Doppler Tomography for Radar Novel View Synthesis** — T. Huang et al., CVPR 2024. arXiv:[2403.03896](https://arxiv.org/abs/2403.03896).

12. **Radar Fields** — D. Borts, E. Liang et al., SIGGRAPH 2024. arXiv:[2405.04662](https://arxiv.org/abs/2405.04662).

13. **RadarSplat** — C.-L. Kung et al., ICCV 2025. arXiv:[2506.01379](https://arxiv.org/abs/2506.01379).

### Physics

14. **Cook, R. L., Torrance, K. E.** *"A Reflectance Model for Computer Graphics."* ACM Trans. Graph. **1(1)**, 1982.

15. **Walter, B., Marschner, S. R., Li, H., Torrance, K. E.** *"Microfacet Models for Refraction through Rough Surfaces."* EGSR 2007.

16. **ITU-R Recommendation P.2040** — *"Effects of building materials and structures on radiowave propagation above about 100 MHz."*

---

## §10 Phase order

```
v5_v2 Phase 0 — directory clone + sanity                       1 day
v5_v2 Phase 1 — learnable positions, AMPLITUDE GRADIENT ONLY   2 weeks
                (NO PHASE GRADIENTS — phi_carrier and n_peak
                 stay detached; positions perturb sub-mm via
                 the well-conditioned amplitude path only.
                 Only run if Phase 2/3 indicate it's needed.)
v5_v4 Phase 2 — radar-aware init (§2)                          1 week
v5_v4 Phase 3 — densification with ‖∇p L‖ selection (§3 + §4)  2 weeks
v5_v4 Phase 4 — D heatmap (diagnostic, §5)                     3–5 days
```

**Total span: ~5-6 weeks** (assuming Phase 1 is run; ~3-4 weeks if Phase 1 is skipped because Phases 2+3 alone hit the P0 gate).

---

## §11 Immediate next action

This document is the plan. **v5_v4 Phase 2 (radar-aware init)** is the lowest-risk, highest-information starting point — it's a non-learning sampling change that costs ~1 week to ablate, and the result *both* (a) tells us if init is binding, and (b) provides a better starting point for everything downstream.

Recommended phase order:

1. **v5_v2 Phase 0** — clone `mm25DGS_v5` → `mm25DGS_v5_v4` (1 day, gate: v5 bench reproduces within ±0.01).
2. **v5_v4 Phase 2** — radar-aware init (1 week, gate: no-holdout |RA| train ≥ 0.84).
3. **v5_v4 Phase 3** — densification with ‖∇p L‖ selection (2 weeks, gate: no-holdout |RA| train ≥ 0.90 P0).
4. **v5_v4 Phase 4** — D heatmap (3–5 days, diagnostic only — produces the per-view-membership matrix; informs whether per-view variance is real).
5. **v5_v2 Phase 1** — learnable positions via amplitude-gradient only (2 weeks, gated on whether Phase 3 hit the P0 train target; only attempt if not).

v5_v2 Phase 1 (learnable positions) is **only run if v5_v4 Phases 2+3 alone fall short of the P0 train target**. If they hit it, Phase 1 becomes optional polish.

---

## Appendix — citations (BibTeX-ready, ordered by §9)

```
[1]  Lang, I., Manor, A., Avidan, S.
     "SampleNet: Differentiable Point Cloud Sampling."
     CVPR 2020 (Oral).
     arXiv:1912.03663 — https://arxiv.org/abs/1912.03663
     Code: https://github.com/itailang/SampleNet

[2]  Fang, G., Wang, B.
     "Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians."
     ECCV 2024.
     arXiv:2403.14166 — https://arxiv.org/abs/2403.14166
     Code: https://github.com/fatPeter/mini-splatting

[3]  Mallick, S.S., Goel, R., Kerbl, B., Steinberger, M.,
     Carrasco, F.V., de la Torre, F.
     "Taming 3DGS: High-Quality Radiance Fields with Limited Resources."
     SIGGRAPH Asia 2024 Conference Papers.
     ACM DOI: https://dl.acm.org/doi/10.1145/3680528.3687694
     arXiv:2406.15643 — https://arxiv.org/abs/2406.15643
     Code: https://github.com/humansensinglab/taming-3dgs

[4]  Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.
     "3D Gaussian Splatting for Real-Time Radiance Field Rendering."
     ACM Trans. Graph. 42(4), SIGGRAPH 2023.
     arXiv:2308.04079 — https://arxiv.org/abs/2308.04079

[5]  Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T.,
     Ramamoorthi, R., Ng, R.
     "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."
     ECCV 2020.

[6]  Müller, T., Evans, A., Schied, C., Keller, A.
     "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding."
     ACM Trans. Graph. 41(4), SIGGRAPH 2022.

[7]  Xu, Q., Xu, Z., Philip, J., Bi, S., Shu, Z., Sunkavalli, K., Neumann, U.
     "Point-NeRF: Point-based Neural Radiance Fields."
     CVPR 2022.

[8]  Eldar, Y., Lindenbaum, M., Porat, M., Zeevi, Y. Y.
     "The farthest point strategy for progressive image sampling."
     IEEE Trans. on Image Processing 6(9), 1305-1315, 1997.

[9]  Pharr, M., Jakob, W., Humphreys, G.
     "Physically Based Rendering: From Theory to Implementation."
     4th edition, 2023.
     https://www.pbr-book.org

[10] Jang, E., Gu, S., Poole, B.
     "Categorical Reparameterization with Gumbel-Softmax."
     ICLR 2017.
     arXiv:1611.01144 — https://arxiv.org/abs/1611.01144

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

[14] Cook, R.L., Torrance, K.E.
     "A Reflectance Model for Computer Graphics."
     ACM Trans. Graph. 1(1), 1982.

[15] Walter, B., Marschner, S.R., Li, H., Torrance, K.E.
     "Microfacet Models for Refraction through Rough Surfaces."
     EGSR 2007.

[16] ITU-R Recommendation P.2040.
     "Effects of building materials and structures on radiowave propagation
     above about 100 MHz."
```
