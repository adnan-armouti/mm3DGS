# v7 Fundamental Ceiling Analysis — First-Principles Investigation Plan

**Date:** 2026-04-23
**Parent plan:** `md/mm25dgs_v7_training_ceiling.md` (the C1/C2/C3 loss-side
triage, now largely complete — see §0 below).
**Target:** train ≥ 0.85 and test ≥ 0.70 on **both** |RA| and |RAD|,
**both** improving (no train/test trade-off).

---

## §0 What we already know — post-C1 empirical result

6-scene bench with the C1 fix (`gt_max²` → `gt_mean²` in doppler loss,
plus the C2a |RAD| cc diagnostic):

| metric | v5/M1 | v7 pre-C1 (max) | v7 post-C1 (mean) | target |
|---|---:|---:|---:|---:|
| \|RA\| train mean  | 0.826 | 0.661 | **0.687** | ≥ 0.85 |
| \|RA\| test  mean  | 0.537 | 0.546 | **0.563** | ≥ 0.70 |
| \|RAD\| train mean | —     | 0.530 | **0.560** | ≥ 0.85 |
| \|RAD\| test  mean | —     | 0.358 | **0.357** | ≥ 0.70 |

C1 delivered **+0.03** on the train side, **≤+0.02** on the test side
— not the **+0.10+** the ceiling plan predicted. **The plan's gradient-
shrinkage reasoning was wrong for Adam**: Adam's update is
`g / √v`, so a constant loss multiplier `k` gives `g·k / √(v·k²) = g / √v`
— *scale-invariant to first order*. The residual C1 gain is from
(a) per-parameter `v` non-uniformity, (b) LR schedule × gradient-clip
interaction, (c) non-stationarity before `v` stabilises. None of those
scale like the plan assumed.

**Consequence:** the ceiling is **structural**, not a loss-scaling issue.
The next cycle of "tweak the loss" will also fail. We need a full audit.

**Additional context:** v5/M1's |RA| train ceiling is **0.826** — also
below the user's 0.85 train target. v5/M1's |RA| test is **0.537** —
far below the 0.70 target. *The ceiling predates v7*; the Doppler work
didn't cause it, it just made it worse on a subset of metrics. Any fix
must apply equally to v5.

---

## §1 First-principles candidates for the true ceiling

These are mutually-non-exclusive sources of the plateau. The
investigation protocol in §2 is designed to *rank* them
by empirical impact, not to commit to any one ahead of data.

### §1.1 Coherent-phase error from frozen positions + detached phases

This is the most likely dominant blocker. It is also the hardest to
talk around — it is *built into* the training contract:

- `LEARN_POSITIONS = False` ([train_gaussian.py:73](../mm25DGS_v7/train_gaussian.py#L73))
  — point positions are taken from the LiDAR pcl and **never optimized**.
- `detach_phase=True` (default in [rasterizer_factorized.py:35](../mm25DGS_v7/rasterizer_factorized.py#L35))
  — the carrier phase `2π·f₀·(τ_tx+τ_rx)` is detached before step 5,
  so `n_peak` and `phi_carrier` carry **no gradient signal**. This is
  *required* for the fused step5 backward (which only emits `grad_w`).
- The only things that get gradients are (a) per-point quaternions
  (→ normals), (b) per-point 6-parameter raw_materials.

**Implication.** Suppose the LiDAR pcl places a scatterer 3 mm off from
its true radar-visible location. At λ = 3.9 mm, the path-length error
is ~6 mm round-trip → phase error ≈ (2π/3.9mm)·6mm ≈ **9.7 rad**, wrapping
several times. The amplitude splat at that (tx, rx) goes into a
coherent sum with ~192 others; random-looking phase offsets across the
192 paths force the coherent sum toward √192 × single-path amplitude
(i.e. **incoherent**). The optimizer can scale the amplitude of each
path via materials/normals — but cannot fix the phase that drives
whether those paths add constructively or destructively. In this
regime, only the **envelope** of |RA| is fittable; the coherent fine
structure is not.

**Why this plausibly caps us at ~0.83 train and ~0.54 test:** v5
reaches exactly those numbers. Coherent structure contributes to
|RA| cart-corr non-uniformly — peaks are roughly right (envelope),
sidelobes are randomized (phase), and CC lands around 0.8 on train
where the envelope has been tuned, and lower on test where even the
envelope isn't exactly right.

**Why this plausibly caps us worse on |RAD|:** |RAD| collapses the
16-chirp coherent structure along the chirp axis too. Any residual
phase noise on top of the Doppler phase further scrambles the Doppler
spectrum. So |RAD| CC is *more* sensitive to position error than
|RA|. Empirically: 0.56 |RAD| train vs 0.83 |RA| train.

### §1.2 Missing non-ego Doppler (physics incompleteness)

Our forward model's ONLY source of Doppler is **ego motion**. Real
scenes contain:

- pedestrians walking (v ≈ 1 m/s → Doppler bin ±4)
- vehicles crossing (v ≈ 15 m/s → aliased across the full spectrum)
- vegetation / wind / micro-motion (v ≈ 0.1 m/s → smeared around zero)
- radar multi-path / clutter oscillations

The GT |RAD| cube *records* all of these. Our prediction puts
**zero** energy at their Doppler bins. If 15–30% of a scene's signal
energy lives in non-ego Doppler, that is an immediate **upper bound
drop of 0.15–0.30** on |RAD| CC — built in, before any optimization.

**Falsifiable test.** Take GT |RAD| cube, zero out all non-zero
Doppler bins, re-normalize; compute CC(GT_original, GT_zeroed). The
CC you get is the maximum |RAD| CC achievable by an ego-motion-only
forward model. If it's already at 0.70–0.80 we can still hit target.
If it's at 0.50, we're fundamentally capped.

### §1.3 Calibration / alignment errors

- **Pose alignment.** Pass-2 alignment is roughly mm-accurate. At
  λ = 3.9 mm, mm-level misalignment is a radian of phase. We treat
  every LiDAR-aligned point as phase-truth and detach — so a 1-mm
  error is a ~1 rad permanent error per point.
- **Radar constant / dBFS scaling.** `rast.radar_constant × rx_dBFS_scale
  × adc_scale` is a scalar C_radar that multiplies **every** path
  amplitude. If it's off by, say, 1.5×, the optimizer tries to
  compensate by pushing materials toward unphysical values.

### §1.4 Absolute-scale loss on sparse cubes

v7's doppler body trains on raw |RAD| magnitudes. v5's `mse_raw`
does the same on |RA|. The `mmir/train.py` reference inverse
renderer (per-frame, *validated* to give good fits) uses
**independent min-max normalisation** on prediction and GT before
the loss ([mmir/losses/loss_utils.py:69](../mmir/losses/loss_utils.py#L69),
[train.py:1709](../train.py#L1709) `use_joint_normalization`).
In that pipeline, the loss sees only the *shape* of the (RA / RAD)
pattern, not its absolute amplitude.

Why this matters: absolute-scale loss couples the optimizer to **two
unknowns** at once — (a) the global `C_radar × material-mean` scale
and (b) the per-cell pattern. Those are hard to disentangle with a
noisy GT. The normalised variant removes (a) from the loss entirely
and forces the optimizer to learn *only* the pattern. This is
exactly the v5 `loss_type='mse'` path that was dropped in favour of
`mse_raw` when the 100× C_radar boost made scales close; but it is
*not* the path that was proven to work per-frame in `mmir/train.py`.

**Worth including in the ablation matrix (§2.5) as a first-class
variant**, not relegated to "legacy". If independent min-max
normalisation + pearson loss gives markedly better per-frame fits in
the MC inverse renderer, there is no reason to assume it will not
also help here.

### §1.5 Loss objective mismatch (C2)

Already discussed in the training-ceiling doc. |RA| metric
specialises to one 2D slice; |RAD| metric reads the full cube.
A multi-task loss (C2b) would re-align objective and metric.
Probably worth a pass but unlikely to be the dominant fix — the
C1 result already suggests loss engineering is not the lever.

---

## §2 Investigation protocol — ordered by information / effort

Each bullet is designed to produce a **number** or a **plot** that
narrows the hypothesis space. Nothing here commits us to a fix yet.

### §2.0 Upper-bound diagnostics *(cheap — do first)*

All of these are single-scene, ~1-minute compute, and directly bound
what we can hope to achieve.

**§2.0.a Intra-frame chirp coherence.** For `seq_0_frame_135`:
- `CC(chirp_0 |RA|, chirp_k |RA|)` for k = 1..15. Mean and std.
- If this is e.g. 0.92, real radar noise floor is ~8% — ceiling on
  train is roughly 0.92. If it's 0.99, noise floor is negligible.

**§2.0.b Inter-frame coherence (NVS ceiling).** For each scene:
- `CC(|RA|_F, |RA|_{F±1})`, `CC(|RA|_F, |RA|_{F±4})`.
- This is the *maximum test CC any NVS method can give you* without
  introducing information beyond temporal interpolation.

**§2.0.c Naive-average baseline (confirm).** Already in summary
scripts as 0.66 mean. Re-check against C1 results to make sure the
6-scene means are self-consistent.

**§2.0.d Per-bin GT variance map.** For `seq_0_frame_135` across 16
chirps: std of |RA| per (az, range) cell / mean. Low-variance cells
are the stable targets — high-variance cells (non-ego Doppler,
clutter) are lost causes. Save as a 127×256 heatmap.

**§2.0.e Ego-motion-only |RAD| ceiling.** For each scene:
1. Compute full GT |RAD|.
2. Compute a "static-scene" reference: render a synthetic cube
   *using the GT |RA| at chirp 0* and applying only the ego-motion
   Doppler phase analytically. This is the best v7 could ever be,
   per-scene, assuming BSDF+positions are perfect.
3. `CC(GT_RAD, static_RAD)`. This is the ceiling on |RAD| CC for an
   ego-Doppler-only model.

**Deliverable:** table of six numbers per scene. If these say we're
capped at, e.g., 0.80 |RA| and 0.60 |RAD|, the user's 0.85/0.70
target is not physically achievable with this model class — the
paper's framing needs adjustment, not the optimizer.

### §2.1 Single-frame fit ceiling *(cheap)*

Train `seq_0_frame_135` on **itself only** (1 frame, 16 chirps, loss_type=mse_raw
on |RAD|), `target_n=90000`, 2000 iters. No NVS, no generalization.
This measures *pure representational capacity* — "how well can the
renderer fit any single frame if given unlimited grace?".

- If final train |RA| CC < 0.90, **the forward model itself is the
  bottleneck** (capacity + geometry). No optimizer trick will rescue
  us; structural change needed (learn positions, add BSDF expressivity,
  multi-bounce).
- If final train |RA| CC ≥ 0.95 on a single frame, the model *can* fit —
  and the plateau on multi-frame training comes from generalization
  fighting with a rigid representation.

### §2.2 Visualization dump *(medium)*

For `seq_0_frame_135` at init and final (v5 baseline and v7 post-C1):

1. **Full |RA| images** — GT vs pred, side-by-side, per scene. Save as
   PNG to `md/diagnostics/<scene>_ra_gt_vs_pred.png`.
2. **Range profile at peak azimuth bin** — 1D curve, GT vs pred,
   log-scale y. Which range bins do we miss?
3. **Doppler spectrum at peak (range, az)** — does our predicted
   spectrum have the *shape* of GT? Is our ego-Doppler peak at the
   right bin? Is GT diffused across many bins (moving scatterers)
   while ours is a clean spike?
4. **(az, range) error heatmap** — `|pred - gt|/max(gt)` log-scaled.
   Cluster of error in specific az/range regions → localized scene
   phenomena we're missing.
5. **Per-point contribution map** — project per-point Σ|w_full(t,r)|
   back onto the scene, save as point cloud colored by log-magnitude.
   Find the *dead points* (contributing nothing) and the *too-hot
   points* (dominating incorrectly).
6. **(Optional) Learned material histograms** — eps_real, eps_imag,
   sigma_h, l_c, thickness, tau_base distributions at init vs final.
   Sanity check: are materials driven to physical ranges, or off into
   pathological attractors?

All deliverables scripted under
`mm25DGS_v7/scripts/diagnostics/` so they're reproducible, not
one-offs.

### §2.3 v_ego sensitivity sweep *(cheap)*

- Render `seq_0_frame_135` |RAD| at `v_ego × {0, 0.5, 0.9, 1.0, 1.1,
  1.5, 2.0}`.
- Compute CC vs GT |RAD| for each.
- Curve width around 1.0× quantifies how much v_ego error we can
  tolerate.
- If CC drops ≥ 0.1 between 0.9× and 1.1×, v_ego accuracy is
  crucial and our pose-derived 1.3 m/s estimate might be meaningfully
  off.

### §2.4 Position-jitter sensitivity *(cheap)*

- Jitter all TX positions by N(0, σ) with σ = 0.1, 1, 10 mm.
- Render |RA|, compute CC vs the un-jittered baseline.
- Curve quantifies alignment-error sensitivity. If σ = 1 mm drops
  CC by 0.1, we know our pass-2 alignment (~mm) is meaningfully
  costing us. If σ = 10 mm changes nothing, alignment is not the
  blocker.

### §2.5 Systematic ablation matrix *(medium — key deliverable)*

Structured, not ad-hoc. One scene (`seq_0_frame_135`), 500 iters,
3 seeds each for MC-noise averaging (±0.03). Baseline is current
v7 post-C1 (mean-norm, full BSDF, doppler on, fused step5_doppler,
detach_phase=True). For every row, flip **one** switch and record
the four-metric vector (|RA| train, |RA| test, |RAD| train,
|RAD| test).

| # | axis | setting | hypothesis being tested |
|---|---|---|---|
| 1 | baseline | — | reference |
| 2 | loss | normalised min-max (pred and GT independently) + mse | §1.4: absolute-scale hurts fit |
| 3 | loss | normalised joint (shared scale) + mse | joint-normalisation comparison to independent |
| 4 | loss | Pearson on flattened \|RAD\| (scale+shift invariant) | purest shape loss |
| 5 | loss | C2b multi-task: λ·mse(\|RA\|_chirp0) + mse(\|RAD\|) at λ∈{0.1, 0.3, 1.0} | objective ≠ metric mitigation |
| 6 | BSDF | disabled\_components=`jones` | Jones-Fresnel net benefit |
| 7 | BSDF | disabled\_components=`ka`  | is SPM-only enough? |
| 8 | BSDF | disabled\_components=`spm` | is KA-only enough? |
| 9 | BSDF | disabled\_components=`slab` | multi-layer coherence benefit |
| 10 | occlusion | shadow_mask=ON (re-enabled) | v5's 2026-04 dropping may have been premature |
| 11 | doppler | T_a = 0 (no TDM phase) | §1.2/§3.2.b — is TDM sign/permutation costing us? |
| 12 | doppler | v_ego = 0 (no Doppler phase) | does Doppler itself help or hurt in 6-scene agg? |
| 13 | doppler | ti\_firing\_index = identity (no TI permutation) | §3.2.b — confirm permutation is correct |
| 14 | learn | LEARN_MATERIALS = False (only normals) | is materials the working lever? |
| 15 | learn | LEARN_NORMALS = False (only materials) | is normals the working lever? |
| 16 | target_n | 45K, 90K (vs baseline 20K) | does more points buy signal? |
| 17 | iters | 2000 (vs baseline 500) | have we converged? |
| 18 | kernel | use_cuda_kernels=False (pytorch path) | CUDA ≡ pytorch sanity (re-confirm) |
| 19 | el axis | include el>0 rows in loss (don't discard §3.2.d) | 25% capacity currently unused |
| 20 | v_ego | per-frame grid-search correction (§4.a) | v_ego noise contribution |

Expected output: `md/diagnostics/ablation_matrix.md` with a full
4-column-per-metric × 20-row table, colour-coded (↑ green, ↓ red,
flat within MC grey). Any green-flagged variant > +0.03 on any
metric is a candidate to promote into the default config.

**Ablation design discipline:** do NOT change two axes at once, do
NOT cherry-pick the best seed, do NOT infer trends from single
scenes when the table is mean±std across 3 seeds. If a row shows
+0.05 on |RA| train but −0.04 on |RA| test, that is information,
not noise — flag it in the report.

### §2.6 Representation-expansion trials *(structural)*

Only attempt after §2.0–§2.5 have narrowed the hypothesis to
representation capacity:

**§2.6.a Learn bounded position deltas.** Add `pos_delta ∈ ℝ^(N,3)`
parameter, bounded by `tanh(x) × 3mm`, penalized by
`λ · ||pos_delta||²`. This re-enables phase gradient indirectly
(positions → τ → phi_carrier) at the cost of making step5 backward
non-detached. Probably requires a new forward path that does NOT
detach `phi_carrier` / `n_peak` (and therefore cannot use the fused
step5 kernel — must use a differentiable-phase backward).

Risk: overfits train frames, hurts test. Mitigation: strong L2
penalty, per-point gradient clip.

**§2.6.b Learn per-frame small v_ego correction.** A scalar 3-vector
`dv_ego` per train frame, bounded to e.g. ±0.2 m/s, regularized to
zero. This *does* apply directly to phi_doppler — but phi_doppler
is `.detach()`ed in render_factorized_doppler. Making it differentiable
would require extending the fused kernel backward to also emit
`grad_dv_ego` (or handling it via an explicit PyTorch path for just
the Doppler phase, which is tiny — M·n_tx floats).

Risk: aliasing with ego-pose calibration error. Not clear these can
be separately identified.

**§2.6.c Volumetric / multi-bounce primitives.** Add a small
population of "corner reflector" primitives (trihedrals) with
learnable position + RCS. Structural change; 2–3 weeks of work.
Don't attempt until §2.0 + §2.1 justify it.

---

## §3 Code correctness audit — what I've inspected

This section records what *has* been checked so far and what still
needs verification. No bug found yet is a silver bullet; this is the
"turning over stones" pass.

### §3.1 Checked — no bug

- `detach_phase=True` in v7 matches v5 — confirmed correct.
  ([rasterizer_factorized.py:35](../mm25DGS_v7/rasterizer_factorized.py#L35))
- Fused `step5_doppler_fused` forward + backward numerical parity
  against 16× `step5_fused` reference: fwd 5e-6 rel, bwd 2e-5 rel.
  ([cuda/tests/test_step5_doppler.py](../mm25DGS_v7/cuda/tests/test_step5_doppler.py))
- Batched `_batch_txrx_to_vx_el0` bit-identical to legacy per-chirp
  `txrx_to_vx_chirps_torch`. (Smoke-tested.)
- Three Doppler validation gates PASS (8.2.1 analytic single-point,
  8.2.2 multi-point vs LERP, 8.2.3 v_ego=0 bit-identity).
  ([validate_doppler_synthesis.py](../mm25DGS_v7/scripts/validate_doppler_synthesis.py))
- C1 loss-norm switch (`max` vs `mean`) with distinct output dirs;
  both variants complete 500-iter runs cleanly.

### §3.2 Plausible concerns — need to verify

**§3.2.a TDM linear-phase approximation.** We model TDM phase
per TX slot as `k(i)·T_a · (4π/λ)·<û, v_ego>`. This assumes (a) the
radar velocity is constant during T_a × 12 = 0.5 ms, (b) the scatterer
is far enough that the path-length change over that burst is
linear in time. Both are strong assumptions. If v_ego has 10%
non-constancy during T_a, the approximation smears 2 rad cumulative
phase by ~0.2 rad — in the noise. Probably fine.

**§3.2.b TI firing-order permutation.** The `TI_FIRING_INDEX_FROM_ADC_CH`
constant was derived from a single TI documentation snippet. If the
mapping from ADC-TX-channel to firing-slot is wrong, TDM phase is
systematically wrong per TX. Test: 8.2.1 single-point with TDM
currently PASSES with phase error 2.2e-3 rad — this suggests our
permutation agrees with whatever is in the analytic reference we
wrote. But both could be wrong by the same sign. A test that *directly*
drops the permutation (use identity instead) and measures |RAD| CC
would disambiguate.

**§3.2.c Doppler FFT sign convention.** `_doppler_fft_on_chirps` uses
`torch.fft.fft` (standard kernel `exp(-j·2π·k·n/N)`) with
`ifftshift → FFT → drop bin 0 → fftshift`. The sign of the Doppler
phase in our forward model must match this kernel. Our forward phase is
`-(4π/λ)·<û,v>·(m·T_c + k·T_a)`. At v>0 (approaching scatterer), phase
**decreases** with chirp index m → FFT energy lands at **positive**
frequency bin k. Sanity-check by rendering a single-scatterer scene
with known approach velocity and verifying the peak is at the expected
bin. This was done in test 8.2.2 (LERP vs analytic) with cc 0.832 — so
likely OK, but an explicit sign-targeted test would be cleaner.

**§3.2.d txrx_to_vx el=0 row picks only 9 of 12 TX.**
Three of the TI TX antennas are at non-zero elevation (tx_loc[1] ∈
{1, 4, 6}). Their (RX, TX) contributions go to vx positions with
vx_y > 0, which we discard (`vx[0, 0, :, :]`). Their forward-model
amplitudes are still *computed* (the step5 kernel writes them), but
they never feed back gradient because the loss never sees them.
Effect: 25% of our TX capacity has no gradient signal. This matches
v5 behaviour (v5's `adc_to_ra_complex` also discards them). Not a
bug per se, but it's pure waste — the renderer is doing 192-path work
then throwing away 48 paths (25%).

Potentially big fix: train on full (7, 86, R) 3D vx cube — preserves
elevation. Doubles-ish the signal. Not done in v5 probably because
we only have azimuth of interest for the final application, but for
*training* the model there's no reason to discard it.

**§3.2.e `rp_stack[:, :, rx_id, tx_id, :]` iteration order.** The
legacy helper iterated `rx outer, tx inner`. Our batched weight matrix
flat index is `rx_id · n_tx + tx_id`. Needs double-check that the
cumulative averaging weights match for the specific collision
patterns. I ran a numerical parity check → PASS. Noted here for
completeness.

**§3.2.f Range FFT normalization.** `torch.fft.fft` is un-normalized
(multiplies by 1, not 1/√N). Our predicted path amplitudes must
match GT's un-normalized convention. Quick-check: print
`GT.mean() / pred.mean()` at init. If it's not O(1), there's a
normalization mismatch hiding under the C_radar global scale.

**§3.2.g Pose time-stamps in build_per_loop_poses.** We LERP 16 chirp
poses from (cfg_{F-1}, cfg_{F+1}) via
`loop_dt_s = 7.87ms / 16 ≈ 491 μs`. But the frame interval is
0.2s at 5 Hz cascade. So the 16 chirps span `491μs × 16 = 7.87ms` —
one burst. The remaining 0.2 s − 7.87 ms ≈ 192 ms is "gap time"
between bursts. If our LERP assumes poses interpolate across 0.2 s
(frame period), the per-chirp v_ego derived from those poses would
be 25× smaller than reality. Already flagged and corrected in the
ceiling plan §5.2 (GT-interpolated v_ego from groundtruth_poses.txt
at cascade timestamps). Verify that the same timing correction is
applied to the rasterizer pose_F used by the renderer — if the
rasterizer uses LERP poses from a wrong time base, predicted path
lengths are systematically off.

**§3.2.h Shadow mask.** v5 dropped the TX shadow mask
([train_gaussian.py:925 comment](../mm25DGS_v7/train_gaussian.py#L925))
based on a 7-scene A/B finding. If dropping shadow mask *helps* it
suggests the shadow computation had a bug (e.g. inverse-
conventions). Worth re-examining under v7 Doppler — a bug in shadow
semantics could explain non-trivial CC loss.

### §3.3 Things *not yet* audited

- Sign convention of `compute_sp_basis` across TX/RX combinations
  at extreme incidence angles.
- Antenna pattern interpolation at boresight singularity.
- `rms_clip_grad` behaviour when gradient magnitudes vary 10×
  across materials (potential per-parameter saturation).
- `init_visible_weighted` determinism — does the FPS subsample
  change across runs? If yes, the run-to-run ±0.03 variation has
  a known source.

---

## §4 v_ego propagation — options without phase gradients

With `detach_phase=True` and `v_ego` treated as an input, we cannot
drive v_ego by gradient descent on |RAD| MSE. But a **dedicated
pre-processing stage** — parallel to the 2-stage pose alignment that
turned the inverse renderer from "really bad" to usable — is the
natural answer here. The pose pipeline went from "trust the raw
sensor config" to "iteratively refine config per-frame before
training." v_ego should go through the same lifecycle.

### §4.0 Dedicated v_ego preprocessing pipeline *(structural — priority)*

Modelled on the 2-stage alignment in
[mmir/preprocessing/alignment/cascaded_alignment.py](../mmir/preprocessing/alignment/cascaded_alignment.py)
pattern. Live as a *new* module
`mm25DGS_v7/preprocessing/v_ego_refine.py`, output cached per-frame
under `data/v_ego_cache/<scene>/frame_<F>_v_ego_refined.npy`. Only
run once per scene; subsequent training loads from cache.

**Pipeline stages:**

1. **Stage 0 — seed (existing).** Current behaviour: GT trajectory
   interpolated at cascade timestamps, as in
   [mm25DGS_v7/preprocessing/v_ego.py](../mm25DGS_v7/preprocessing/v_ego.py).
   This is the *pre-alignment* starting point, directly analogous
   to loading the raw radar config before pose alignment.

2. **Stage 1 — coarse grid refinement per frame.** For each frame:
   render a per-TX-RX range-doppler map (per-antenna RD, not the
   full MIMO |RAD|, so it's cheap — 192 small 2-D FFTs) using the
   Stage-0 v_ego. Compare against the GT per-TX-RX range-doppler
   maps (also cheap; no azimuth FFT). Sweep v_ego ± box in 3-D
   (e.g. {−0.2, −0.1, 0, +0.1, +0.2} m/s on each axis = 125 evals
   per frame). Pick the argmax of per-bundle |RD| cc.
   Wall-clock: ~30 s per scene.

3. **Stage 2 — fine gradient-free refinement.** Seed from Stage 1;
   use Nelder-Mead (scipy.optimize.minimize method='Nelder-Mead')
   or Powell on v_ego's 3 components, objective = −CC(pred_RAD,
   gt_RAD). Terminates at ~10⁻³ m/s precision. Wall-clock: ~60 s
   per frame (because each eval is a full RAD render — still
   acceptable as preprocessing).

4. **Stage 3 — cross-frame consistency check.** Adjacent train
   frames within a scene should have v_ego varying smoothly.
   Refined v_egos that jump by > 0.5 m/s between adjacent frames
   almost certainly indicate spectrum-matching locked onto a
   non-ego scatterer peak. Flag those as "suspect" and either
   fall back to Stage 1 or exclude the frame from the training
   window.

**Why this is worth the structural investment:** the pose-
alignment analogy is almost perfect. In both cases:
- Raw sensor-derived value (IMU/trajectory for v_ego; config-file
  pose for alignment) is the starting point.
- Direct use of the raw value gives "really bad" fits.
- A dedicated per-frame iterative refinement against the actual
  measured signal brings in enough signal to unlock training.
- The refinement is offline (one-shot), gradient-free (avoids phase
  chicken-and-egg), and cache-able.

**Caveat:** Stage-2 spectrum matching *can* lock onto a non-ego
peak (moving car, pedestrian) in scenes with bright dynamic
scatterers. Mitigation: constrain Stage-2 to the ±0.3 m/s box
around Stage-1's Stage-0-seeded initializer. That rules out
macroscopic jumps to moving-scatterer peaks.

**Deliverable:** a preprocessing script that regenerates the
`data/v_ego_cache/` contents end-to-end for all 6 bench scenes,
plus before/after |RAD| CC numbers per frame.

### §4.a Grid-search per-frame v_ego correction *(cheap — §4.0 Stage 1 as a training hook)*

If §4.0 is not yet implemented but we want a quick read on whether
v_ego is a real blocker: during training, for each train frame,
define
`v_ego' = v_ego_GT + (a, b, c)` with `a, b, c ∈ {−0.2, −0.1, 0, +0.1, +0.2} m/s`
(125 combinations). For each, render |RAD|, compute
CC(GT, pred). Pick the argmax. This runs in seconds and is an
upper-bound estimate of how much v_ego error is costing us.

If the best grid point is a 0.1–0.2 m/s correction, v_ego IS a real
source of error and §4.0 is worth the structural investment. If the
argmax is always at `(0, 0, 0)`, v_ego is fine and the ceiling is
elsewhere.

### §4.b Spectrum-matching estimator *(medium — global bias only)*

A cheaper, less-powerful precursor to §4.0: learn a *single* global
v_ego scale-factor across all training frames that maximises
training-set RAD CC. If that scalar deviates significantly from
1.0, the GT-trajectory-to-v_ego pipeline has a systematic bias.
Fixes that bias for every frame simultaneously — but cannot fix
per-frame variation that §4.0 Stage 2 would.

### §4.c Per-bundle Doppler peak alignment *(cheap — sanity check)*

Locate the bin of max |pred|_RAD along the Doppler axis; locate the
bin of max |GT|_RAD. Their offset in Doppler bins, converted to m/s,
*is* the v_ego correction. Per-scene, single number, no training loop.
Useful as a sanity check for §4.0 Stage 1's grid-search result.

### §4.d Full gradient-on-phase path *(structural — last resort)*

Make `phi_carrier` and `phi_doppler` differentiable end-to-end.
Requires:
- Forward/backward CUDA kernel that emits grad_phi as well as grad_w.
- Accept the per-iter compute hit (probably 1.5–2× the fused kernel
  cost).
- Accept the *much larger* risk of the optimizer finding pathological
  phase attractors — random walk on phi with detached amplitude is
  notoriously difficult to train.

Don't attempt §4.d until §4.0 Stage 1+2 have been measured and
still leave a v_ego-attributable gap.

---

## §5 Deliverable and decision gate

**Goal of this investigation phase:** a short (5–8 page) `RESULTS.md`
document by end of week answering, *in order*:

1. **Upper bounds.** What is the measured upper bound on |RA| and
   |RAD| test CC (intra-frame chirp coherence; inter-frame NVS
   ceiling; ego-motion-only static-scene |RAD| ceiling)? — §2.0.
2. **Single-frame fit ceiling.** What train CC do we reach on
   |RA| and |RAD| when training and evaluating on the same frame
   (no NVS)? — §2.1. This has been "good" in prior experience
   per user note; we need an explicit number now for v7 doppler.
3. **Where does GT actually disagree with pred?** — §2.2
   visualisation dump. Specifically: which (az, range, doppler)
   cells do we systematically miss?
4. **v_ego headroom.** How much of the remaining gap is explained
   by v_ego error alone? — §4.a grid search (or §4.0 Stage-1
   preprocessing if we go structural).
5. **Ablation matrix.** Which single-axis change on the matrix
   in §2.5 moves any of the four metrics by > +0.03? — §2.5.
6. **Alignment / geometry headroom.** Does 1-mm pose jitter cost
   ≥ 0.1 CC? If yes, 2-stage pose alignment refinement (analogous
   to the pose pipeline but tighter tolerance) becomes urgent.
7. **Recommended next structural move.** Given the answers to 1–6,
   which single item from §2.6 / §4.0 / better alignment has the
   highest expected benefit per unit of code work?

**Decision gate.** 

- If §2.0 says |RAD| test ceiling is < 0.65, **the target is
  physically impossible** with the current forward-model class.
  The paper's claim must shift from "match GT |RAD|" to a more
  defensible framing (e.g. "match the ego-motion-attributable
  component of |RAD|"), and we stop chasing the impossible.
- If §2.1 says **single-frame fit** for v7 doppler < 0.9, the
  bottleneck is representational, **not** regularisation or loss —
  §2.6 structural work is the only path.
- If §2.5 ablation row 2 (independent min-max normalisation) moves
  metrics by > +0.03 across all four columns, that is a drop-in
  win and the default loss should be changed.
- If §4.a says per-frame v_ego correction buys > +0.03 |RAD| CC,
  §4.0 preprocessing pipeline becomes the next priority.
- If §2.4 says 1-mm TX jitter kills 0.1 CC, re-doing alignment
  becomes higher priority than any of the above.

Only after these questions have numerical answers do we commit to a
direction for v8 / paper cycle.

---

## §6 Work order

No training-side implementation yet. The next PR on `pt` should:

1. Scaffold `mm25DGS_v7/scripts/diagnostics/` with:
   - `upper_bounds.py` (§2.0.a–e)
   - `single_frame_fit.py` (§2.1)
   - `visualize_pred_vs_gt.py` (§2.2.1–6)
   - `sensitivity_sweeps.py` (§2.3, §2.4)
   - `ablation_matrix.py` (§2.5 — driver that runs all 20 rows)
2. Scaffold `mm25DGS_v7/preprocessing/v_ego_refine.py` implementing
   §4.0 Stages 1–3 (offline; cached output).
3. Generate and commit the output PNGs + `md/diagnostics/RESULTS.md`
   answering questions 1–7 in §5.
4. On the strength of those numbers, open a GitHub issue per top-3
   blocker and schedule §2.6 work accordingly.

Nothing in §2.6 (position deltas, phase gradients, new primitives)
should be started until §2.0–§2.5 have run and the numbers are in.

**This is a diagnostic sprint, not a fix sprint.** The user-stated
target (0.85 train, 0.70 test on both |RA| and |RAD|) may or may not
be achievable with the current forward-model class. That is the first
question to answer; fixing anything before answering it risks a
fourth cycle of loss-tweaking that doesn't move the plateau.

---

## §7 Appendix — what this plan does *not* commit us to

Explicitly excluded from this phase (revisit only on the strength
of §5 numbers):

- **BSDF capacity changes** (new lobes, multi-bounce, volumetric
  primitives). Per-frame training with the current BSDF has
  historically given good fits — it is not the weakness. Do not
  spend investigation time here until §2.1 says otherwise.
- **Antenna-pattern replacement** (76 → 77 GHz). Per-frame fits
  work with the current patterns. Do not touch in this cycle.
- **Changing the v5 `mse_raw` → `mse` default globally.** Settle
  that by the ablation-matrix row 2 result only, not by intuition.
- **Learning phase gradients end-to-end (§4.d).** Last-resort option
  only if §4.0 exhausts the non-gradient path and the gap remains.
