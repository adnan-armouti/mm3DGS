# mm25DGS_v6 — design & implementation plan

Date: 2026-04-19 (revised — Option C Gram supervision adopted)
Owner: Adnan Armouti
Status: design → implementation

Goal: cross the **HO_8 test-cc = 0.70 mean across 7 scenes** bar that
frame-NVS has been stuck at (~0.58 after v5 S4 + pool_knn annulus).

The v5 investigation established that parameter-space regularisation
alone cannot close the remaining 0.12 cc gap: UB and HO variants on a
matched 20k-point grid have indistinguishable aggregate parameter
statistics. The fault is in *direction of drift* at the top-1% Fisher
points. v6 fixes this by **giving training a much richer supervision
signal** — moving from "FFT-magnitude on a 86-virtual subset" to
"complex per-(Doppler, virtual, range) tensor with a Gram-matrix
loss that uses every pairwise relative-phase relationship across all
192 virtuals".

| # | Change | Side | Expected 7-scene HO_8 mean-cc lift | Cost |
|---|---|---|---:|---|
| 1 | Plumb the renderer's per-virtual complex range-profile output + GT computation. **Loss unchanged.** | radar/code | 0 (scaffolding) | ~80 LOC |
| 2 | Slow-time Doppler FFT (analytic per-point synthesis) + Gram-matrix loss on the per-(d, virt, r) cube | radar | **+0.06 – 0.15** | ~250 LOC |
| 3 | Doppler-envelope gating from ego-velocity (no per-azimuth mapping needed) | radar | +0.01 – 0.03 | ~30 LOC |
| 4 | SSIM on \|RD-cube\| magnitude per-Doppler-slice | radar | +0.01 – 0.02 | ~20 LOC |

Total expected lift: +0.08 – 0.20 mean cc over v5 baseline (0.524).
Target 0.70 is **plausibly within reach from step 2 alone**.

**Explicitly out of scope** (intentional):
- Azimuth beam-forming on the supervision side. We supervise on the
  per-virtual complex tensor directly via the Gram loss; no DBF, no
  azimuth FFT. (See [`md/gram_vs_fft_derivation.md`](gram_vs_fft_derivation.md)
  for the proof that the Gram loss strictly subsumes FFT-magnitude
  supervision while using all 192 virtuals.)
- Per-primitive spatial footprint / scale parameters (3DGS surfels)
  — would require rasterizer-level integration; deferred to v7.
- HO_128 variants (slow; deprioritised).
- Per-frame appearance embeddings (NeRF-W).
- S5 low-rank material field.

---

## 0. Context carried from v5

### 0.1 What v5 is and is not

v5 is a **point-based splatter** radar inverse renderer. Per-point
state: `(position, rotation-quaternion, 6-param raw_materials)`.
Positions are frozen from LiDAR-FPS; rotations and materials are
learnable.

The v5 forward pass produces a **complex per-virtual range profile
tensor** of shape `(86, 256)` (after MIMO channel combining and
range FFT), then applies the azimuth FFT with averaging of duplicate
virtuals → `(127, 256)` complex RA, then magnitude → 2D RA image,
then polar→cart → cart_corr against FFT-based GT magnitude.

v5 trains with `mse_raw` on RA magnitude. Caps at ~0.58 mean HO_8 cc
on the 7-scene NVS benchmark.

v6 keeps the same point-based representation AND the same rasterizer
primitive (no spatial footprint), and inherits the ColoRadar virtual-
array layout (`TX_LOCATIONS`, `RX_LOCATIONS` from
`mmir/data/ra_utils.py`). What changes is the supervision pipeline:
- ground-truth path skips azimuth processing entirely;
- supervision is on the complex per-(Doppler, virtual, range) tensor;
- loss is the normalised Gram correlation (§2);
- slow-time Doppler is added analytically per-point during render
  (no 16× re-render cost — see §2.2).

### 0.2 What stays the same (strict invariants)

- Point positions remain frozen from LiDAR FPS (v5 behaviour).
- Point primitive is a **coherent-sum delta scatterer** (v5 semantics
  — no per-primitive scale, no footprint).
- Pass-2 per-frame aligned poses (no pass-3 — deprioritised).
- `target_n = 20000` by default (v5 matched-grid benchmark).
- 500-iter training budget per run.
- Seed frame = test frame (v5 matched-grid default).
- **Evaluation `final_test_cc` stays bit-identical to v5's metric.**
  See §0.5. Hard constraint: no v6 change may alter the FFT-RA path
  that produces the benchmark `final_test_cc`.

### 0.3 Benchmark

**Primary**: HO_8 test-cc, 7 scenes: `seq_0_frame_{135,390}`,
`seq_1_frame_{185,438}`, `seq_2_frame_{105,160,300}`. HO_8 means 8
training frames (F-4…F-1, F+1…F+4) × 1 chirp each (chirp 0); test =
held-out frame F chirp 0 (pose interpolated from F-1 and F+1
aligned configs).

- v5 baseline (no S4): **0.5236 mean** (0.5635 excl. seq_0_390)
- v5 best S4 (pool_knn annulus [0.02, 0.10]): **0.5312** (0.5654)
- v5 UB_9 ceiling: 0.8313 (0.8253)
- v5 HO_mat+UB_rot swap ceiling (Analysis I): **0.679** (0.686)
- v6 target: **0.70 mean** across 7 scenes

### 0.4 What the 7-scene data says about what to measure

Cross-scene signals that are STRONG (hold on every scene):
- Rotations dominate materials in the remaining gap (swap ablation,
  +0.156 mean for UB_rot vs +0.056 for UB_mat)
- UB solution basin exists and is parameter-reachable — just not
  from HO's 8-RA loss landscape

Cross-scene signals that are WEAK (don't hold):
- Aggregate drift magnitude (HO ≈ UB at top-1% Fisher on 6/7 scenes)
- Spatial smoothness (HO ≈ UB within 3%)
- LiDAR intensity correlation (ratio 0.99+)

v6 improvements must produce cross-scene gains, not single-scene
wins. Evaluate every change on all 7 scenes.

### 0.5 Two-pipeline architecture (critical)

**Training and evaluation use different pipelines.** The renderer
output is shared up to the per-virtual complex range profiles.
Downstream, training does NOT do azimuth processing; evaluation
follows the v5 FFT-RA path.

```
   ADC (n_chirps=16, n_tx=12, n_rx=16, n_samples=256)
                              │
                              ▼
              range FFT → (16, 12, 16, n_range=256)
                              │
                  MIMO channel combining
                  (TX_LOCATIONS / RX_LOCATIONS layout from
                   mmir/data/ra_utils.py — keep all 192 pairs)
                              │
                              ▼
                  (n_chirps=16, n_virt=192, n_range=256) complex
                              │
              ┌───────────────┴───────────────┐
              │                               │
    (TRAIN)   ▼                     (EVAL, BENCHMARK)
   Slow-time Doppler FFT          Restrict to v5 FFT-subset
   (over n_chirps)                (86 azimuth positions, el=0,
   analytic per-point synthesis    duplicates averaged) → FFT
   on the renderer side (§2.2)     → magnitude → polar→cart
              │                               │
              ▼                               ▼
   `\tilde V \in ℂ^{D=16 ×           |RA_cart| 2D float
                       N=192 × R=256}`               │
              │                               │
   Doppler-envelope gating (M3)               cart_corr against
              │                               v5 FFT-based GT
   Gram-correlation loss (M2):           ↑ THIS IS final_test_cc
   L = Σ_(d,r) [1 − |v_p^H v_g|²       (BIT-IDENTICAL TO v5)
                    /(‖v_p‖² ‖v_g‖²)]
   + SSIM term (M4)
              ↓
          train loss
```

**Hard rule**: v6's `train_frame_nvs.py` MUST continue to call
`mmir/data/ra_utils.adc_to_ra_complex` (the v5 FFT path) for the
quantity reported as `final_test_cc` in `results.json`. v6 only
adds new code paths; nothing in the FFT-RA → cart_corr chain is
modified.

**Diagnostic metrics** (logged but never used to promote a variant):
every v6 run additionally reports
- `diag_normalised_gram_cc_test` — average normalised gram
  correlation over `(d, r)` at the test pose, range [0, 1] (1 =
  perfect match up to per-(d,r) absolute phase).
- `cc_on_complex_RA_test` — complex correlation on the v5 FFT-RA
  path (sanity check that complex agreement tracks magnitude
  agreement).
- `train_loss_history` (already exists in v5 results).

These give us insight into "is the loss actually being driven down"
without affecting the pre-promotion criterion (`final_test_cc`).

---

## 1. M1 — plumb the per-virtual complex tensor + diagnostic logging (scaffolding)

### 1.1 Why

We need to expose the renderer's intermediate **per-virtual complex
range profile** as a first-class output (it currently only flows
into the azimuth FFT). The training pipeline needs to be able to
fetch GT in the same form. M1 wires this up but does NOT change the
loss yet. Pass criterion is: `final_test_cc` matches v5 within MC
noise (±0.003 per scene, 7-scene mean within ±0.002).

### 1.2 What to add (nothing replaced)

#### 1.2.1 GT computation

New file `mm25DGS_v6/data/ra_utils.py`. New function:

```python
def adc_to_per_virt_range_profile(
    adc_ri: torch.Tensor,         # (n_chirps, n_tx=12, n_rx=16, n_samples=256, 2)
    range_window: str = "hann",
) -> torch.Tensor:                # (n_chirps, n_virt=192, n_range=256) complex
    """Range FFT + MIMO flatten only. Keeps all 192 (TX, RX) pairs
    distinct (no averaging). The 192-pair ordering is fixed by the
    same TX_LOCATIONS / RX_LOCATIONS lookup the v5 FFT path uses,
    iterated as `for tx in range(12): for rx in range(16): yield (tx, rx)`.
    """
```

Implementation: range FFT (windowed) → reshape `(n_chirps, n_tx,
n_rx, n_range)` to `(n_chirps, n_virt=192, n_range)` via
`reshape(n_chirps, n_tx*n_rx, n_range)`. **Preserves natural ADC
channel ordering** — no permutation needed at this stage. Channel `v
= tx*16 + rx` corresponds to (TX_LOCATIONS[tx], RX_LOCATIONS[rx]).

Note: v5's `adc_to_ra_complex` remains untouched in `mmir/data/`.
v6's new function lives in v6's own `data/` module.

#### 1.2.2 Renderer per-virtual output

`mm25DGS_v6/rasterizer.py` and `rasterizer_factorized.py` already
compute per-virtual coherent contributions internally; they then
reduce to the v5 86-bin azimuth-virtual layout. We need to expose
the full 192-channel pre-azimuth output.

**Action item before M1 implementation**: read the rasterizer code
to identify the precise tensor that is fed into the azimuth-FFT
step. Add a code path that returns it directly. This is the riskiest
piece of M1 because it depends on the rasterizer's internal
plumbing, which I have not yet inspected. If it turns out to require
deeper kernel changes than expected, M1 may grow from ~80 LOC to
~150 LOC.

Function signature target:
```python
def render_per_virt_complex(
    model, rast, frame_pose, ...
) -> torch.Tensor:               # (n_virt=192, n_range=256) complex
```

#### 1.2.3 Diagnostic metric in trainer

Add to `train_frame_nvs.py`:
- After the v5 FFT-RA `final_test_cc` is computed, ALSO compute:
  - `v_pred = render_per_virt_complex(model, rast, test_pose)`
  - `v_gt = adc_to_per_virt_range_profile(test_adc)[0]`  # chirp 0 only at M1
  - `diag_normalised_gram_cc_test = mean over r of |v_pred[:, r]^H v_gt[:, r]|² / (‖v_pred[:,r]‖² ‖v_gt[:,r]‖²)`
- Persist to `results.json` alongside the existing `final_test_cc`.

CLI flag `--v6_milestone {M1,M2,M3,M4}` (default `M1` for v6 runs).
At M1 only the diagnostic is added; loss code path is untouched.

### 1.3 Validation

`mm25DGS_v6/scripts/validate_per_virt_path.py`:

On `seq_0_frame_135` ADC (chirp 0 only):
1. Compute `v_gt = adc_to_per_virt_range_profile(adc)[0]` →
   `(192, 256)` complex.
2. Apply v5's azimuth FFT path **starting from the 192 channels**
   (not from the v5-86-subset entry point):
   - Pack the 192 channels into the (7, 86) elevation × azimuth
     grid using `txrx_to_vx_chirps_torch`.
   - Apply the existing taper + ifftshift + FFT → `(127, 256)` complex.
   - Magnitude → `(127, 256)` real.
3. Compare to v5's `adc_to_ra_complex` output magnitude. **Must be
   bit-identical** (same operations, same data).

Pass: max abs diff < 1e-5. Anything more means the 192→86 packing
in v6 differs from v5's, which would corrupt the M1 path.

### 1.4 M1 benchmark (7 scenes)

Run HO_8 on 7 scenes with the per-virtual GT plumbed in but the
loss still v5's `mse_raw` on FFT-RA. Confirm `final_test_cc` per
scene matches v5's `output_frame_nvs/.../results.json` value within
MC noise (±0.003). Also report the new
`diag_normalised_gram_cc_test` per scene as a baseline reading.

**M1 is a scaffolding milestone**: passing it means the new GT path
is plumbed correctly without touching the v5 metric. M2 then swaps
the loss.

---

## 2. M2 — slow-time Doppler FFT + Gram-matrix loss (the main lever)

### 2.1 Why this matters

See [`md/gram_vs_fft_derivation.md`](gram_vs_fft_derivation.md) §1–8
for the full derivation. Summary:

- **Information**: Per range bin, the v5 FFT-magnitude carries 127
  real DOF; the per-virtual Gram matrix carries 2N−1 ≈ 383 real DOF
  (rank-1 case) — strictly more. **The "more" is the null-space of
  the (azimuth-FFT → magnitude) operator**: pairwise-relative-phase
  structure that doesn't affect the angular magnitude image.
- **Important correction on what v5 "supervises"**: v5's MSE on
  |RA| is **not** phase-free. The azimuth FFT is a complex→complex
  linear operator that consumes relative phase across virtuals and
  emits an angular spectrum; taking the magnitude afterwards
  preserves angular peak locations and amplitudes. So v5 training
  **does** indirectly constrain the component of per-virtual phase
  that affects |RA| (~ 32k real DOF per frame). What it leaves
  unconstrained is the complement — the ~66k-DOF null space of
  "(FFT → magnitude)". That null space mixes physically-useful info
  (inter-virtual phases for elevated-TX pairs the 86-subset FFT
  drops entirely; per-range residual phase encoding multi-path or
  elevation) with measurement noise (hardware phase calibration
  residuals, quantisation, thermal). The Gram loss supervises this
  complement too — empirically testable at M2.
- **Spatial decomposition**: Gram entry `(i, j)` encodes the
  relative phase between virtual antennas `i` and `j` — i.e. the
  spatial frequency content corresponding to position lag
  `(x_i − x_j)`. So the Gram matrix IS a spatial decomposition of
  azimuth (and elevation, for inter-row pairs); no beam-forming
  required to get angular info.
- **Phase invariance**: `C = v v^H` is invariant under global phase
  rotation `v → e^{jα} v`. Matches the user's stated "we don't care
  about absolute phase" constraint.
- **Doppler is orthogonal extra information**: a 10 cm pose error
  between train and test scrambles ~25 λ of coherent phase, but
  shifts each scatterer's Doppler bin by < 1 bin (because Doppler
  scales with v_ego, not absolute position). RD-cube supervision is
  much more pose-robust than RA at HO_8 pose gaps.
- **Consistency with v5**: the v5 |RA|² is a linear projection of
  the Gram (DFT of diagonal sums; §4 of the derivation). Anything
  v5's loss could see, the Gram loss can see; the converse is false.
- **Revised expected M2 lift** (honest): +0.02 – 0.08 mean cc,
  depending on the signal-to-noise ratio of the extra
  ~66k-DOF-per-frame complement. If it's mostly noise, M2 gain will
  be modest; if elevation / multipath / calibration-residual signal
  dominates, it can go higher. The D-axis (Doppler) is additional
  orthogonal supervision on top of that.

Papers that validate this direction:
- **DART** (Chen et al., ArXiv 2403.03896): Doppler-aided radar
  tomography; uses Doppler to constrain static scene reconstruction.
  Our M3 (gating) adapts their §3.2 directly.
- **Radar Fields** (Bourdon et al., 2024): NeRF-style radar
  reconstruction; argues against pure RA-magnitude losses.
- **Cumming & Wong** (Digital Processing of Synthetic Aperture
  Radar Data, Artech House, 2005, Ch. 6): Doppler beam sharpening as
  the SAR analog of "use motion to make sparse aperture work."

### 2.2 Renderer-side changes (analytic per-point Doppler)

To produce the supervision tensor `\tilde V ∈ ℂ^{D=16, N=192,
R=256}`, we need the per-virtual complex contribution for each of
the 16 chirps. **We do NOT re-render 16 times** — the platform
moves <1 mm in 7.87 ms (<< λ; <0.2 of a range bin) so geometric
quantities (visibility, BSDF, RCS, antenna gain) are constant
across the burst. Only the **per-point range-phase** evolves.

For point `p` with chirp-0 contribution `c_0(p)` to virtual `n` at
range `r`, the chirp-`k` contribution is

$$c_k(p, n, r) = c_0(p, n, r)\cdot e^{j\,2\pi\,f_d(p)\,k\,\Delta t}$$

with

$$f_d(p) = (2/\lambda)\,\langle v_{ego}, \hat r(p)\rangle$$

a per-point scalar that depends on `v_ego` (constant within the
burst, computed from neighbour aligned poses) and `\hat r(p)`
(unit vector from the radar centre to point `p`, also burst-
constant). Both are O(1) per-point evaluations, no re-render
needed.

The renderer's scatter step accumulates per-point contributions
into output bins. The **Option B1** modification: the scatter
step accumulates into 16 output slices (one per chirp) instead of
one, multiplying each point's contribution by `exp(j·φ_k(p))`
before scattering into slice `k`.

Cost increment over v5: only the scatter loop is unrolled 16×.
All expensive work (ray test, BSDF eval, visibility) happens once.
Empirically expected: **1.2–1.5× v5 HO_8 per iter**. NOT 16× (which
is what `train_chirp_loop_nvs.py` costs because it re-renders for
each chirp).

After the renderer produces the (16, 192, 256) coherent cube,
apply slow-time Doppler FFT (windowed) over the chirp dimension:

$$\tilde V[d, n, r] = \sum_{k=0}^{15} c_k[n, r] \cdot w[k] \cdot
                     e^{-j 2\pi k d / D}$$

Output shape: `(D=16, N=192, R=256)` complex.

#### 2.2.1 Validation of analytic synthesis (mandatory before locking it in)

`mm25DGS_v6/scripts/validate_doppler_synthesis.py`:

On seq_0_frame_135 (all 16 chirps):
- **Reference (Option A)**: render at each of 16 chirp-specific poses
  separately → stack into (16, 192, 256). Slow but exact.
- **Synthesis (Option B1)**: render once at chirp-0 pose with the
  analytic Doppler scatter modification → (16, 192, 256).
- Complex normalised correlation between A and B over the full cube.
  Pass: > 0.95.
- Peak Doppler bin per (virtual, range) for the strongest scatterers
  — should agree to ±1 bin between A and B.

If validation fails: fall back to multi-pose Option A → 16× cost.
This would put M2 in HO_128-style territory, which the user
explicitly deprioritised. **In that case M2 is gated on a
per-iter-cost re-discussion before proceeding.**

GT side: `adc_to_per_virt_range_profile` from M1 already produces
`(16, 192, 256)` complex ADC-derived range profiles. Apply the same
slow-time Doppler FFT → `(16, 192, 256)` complex `\tilde V_gt`.
This is the per-frame supervision tensor.

### 2.3 Loss function (normalised Gram)

For each `(d, r)` bin, let `v_p := \tilde V_pred[d, :, r] ∈ ℂ^{192}`
and `v_g := \tilde V_gt[d, :, r] ∈ ℂ^{192}`.

The normalised Gram correlation per bin:

$$g_{d,r}(v_p, v_g) = \frac{|v_p^H v_g|^2}{\|v_p\|^2 \|v_g\|^2}
                    \in [0, 1]$$

The **per-frame loss**:

$$L_{frame} = \sum_{d=0}^{D-1} \sum_{r=0}^{R-1}
              \bigl(1 - g_{d,r}(v_p, v_g)\bigr)$$

Properties (proved in the Gram derivation doc):
- Each term is in [0, 1]. 0 iff `v_p[d, :, r] = e^{jα(d,r)} v_g[d, :, r]`
  for some per-(d,r) absolute-phase scalar α — i.e. perfect agreement
  up to absolute phase, which we don't care about by construction.
- The Gram-equivalent supervision content per frame is
  `D · R · N(N−1)/2 = 16 · 256 · 18 336 ≈ 75M unique pairwise complex
  relationships`, but the **stored autograd tensor is only**
  `(16, 192, 256) complex = 6.3 MB / frame`. The Gram is never
  materialised; the loss factors through scalar inner products
  per (d, r).

**Backward pass**: PyTorch handles complex autograd natively for
the operations involved (conj, sum, complex multiply, abs²). Each
complex element of `\tilde V_pred[d, n, r]` receives gradient
`∂L/∂v_p` of the form

$$\frac{\partial L_{d,r}}{\partial v_p} =
   -2\,\text{Re}\!\left[\frac{v_g^H v_p}{\|v_p\|^2 \|v_g\|^2}
                       \cdot v_g\right]
   +\,2\,\text{Re}\!\left[\frac{|v_p^H v_g|^2}{\|v_p\|^4 \|v_g\|^2}
                       \cdot v_p\right].$$

Implementation file: `mm25DGS_v6/losses/rda_losses.py`.

### 2.4 Ego-velocity estimation

Per training frame F:
$$v_{ego}(F) = \frac{T_{w,b}(F+1).\text{pos} - T_{w,b}(F-1).\text{pos}}{2\,T_{frame}}$$
with `T_frame = 0.1 s` (ColoRadar cascade frame period). Edge frames
of the 9-frame window use one-sided difference. Acceleration treated
as constant over a single 7.87 ms burst (a clean 2nd-order
approximation).

### 2.5 Test-time evaluation (UNCHANGED from v5)

- Render at test pose → get the v5 86-virtual-subset complex output
  (NOT the 192-channel one) → magnitude → polar→cart →
  `cart_corr` against v5 FFT-based GT.
- This is `final_test_cc`. **No v6 milestone touches this.**

In addition, M2 also logs:
- `diag_normalised_gram_cc_test`: mean over `(d, r)` of the
  normalised Gram correlation at the test pose using the same render
  pipeline as M2 training. Indicates how well the loss term itself
  is being optimised.
- `cc_on_complex_RA_test`: complex correlation of the v5-path
  predicted complex RA vs v5-GT complex RA. Should track
  `final_test_cc` closely; deviation indicates phase issues that
  magnitude-only cart_corr can't see.

### 2.6 M2 benchmark

7-scene HO_8 with Gram loss replacing v5 mse_raw. Expected:
`final_test_cc` mean lifts from 0.524 (v5 baseline) to **0.60–0.70**.

Ablation to include: M2 with `D = 1` (no Doppler — supervise
on the (192, 256) complex tensor with Gram loss only). Tests whether
the Doppler dimension is pulling its weight. Expected: D=1 gives
+0.02–0.05; full D=16 gives +0.06–0.15.

---

## 3. M3 — Doppler-envelope gating (DART-inspired)

### 3.1 Why

ColoRadar outdoor runs are 95 %+ static scene. Dynamic scatterers
(pedestrians, cars on adjacent roads) sit at Doppler bins that don't
match the static-scene predicted Doppler `f_d(θ)`. Their RDA
contributions are pure noise to the Gram loss; gating improves
convergence and cross-scene robustness.

### 3.2 Implementation (simplified — no per-azimuth angle needed)

Without azimuth beamforming on the supervision side, we don't have a
per-bin angle to compute per-bin expected Doppler from. Two options:

**(A) Doppler-envelope gating** (preferred, simpler):
- Compute the maximum possible Doppler for any static scatterer in
  the scene from the ego-velocity magnitude:
  `f_d_max = (2/λ) · ‖v_ego‖`.
- Gate ALL `(d, r)` cells whose Doppler frequency `|f_d|` exceeds
  `f_d_max + tolerance` (zero out symmetrically in pred and GT).
- Tolerance: ~1 Doppler bin = ~0.25 m/s; expose as
  `--rda_gate_tolerance_bins`.

This is conservative (keeps everything in the "physically possible"
envelope) but suffices for ColoRadar where v_ego is known well.

**(B) Per-virtual-channel gating using a cheap diagnostic DBF**
(deferred — Option A first):
- Run a coarse DBF (cheap, 16 az bins) just to get expected Doppler
  per-azimuth per-channel.
- Gate per-(d, n, r) cell where the channel's predicted Doppler
  (averaged over its azimuth response) deviates from the cell's
  Doppler by > tolerance.
- More aggressive but requires building the diagnostic DBF.

Start with (A); revisit (B) if M3 disappoints.

### 3.3 Validation

Compare gated `\tilde V_gt` magnitude to ungated. Expected: > 80 %
energy survives (the envelope gate keeps all static-scene returns;
only the small dynamic-contamination outliers are removed).

### 3.4 M3 benchmark

Stack on M2. Expected: +0.01 – 0.03 mean cc.

---

## 4. M4 — SSIM on |RD-cube| (3DGS loss-stack analog)

### 4.1 Why

3DGS uses `L = λ_L1·L1 + λ_SSIM·(1 − SSIM)` as standard. SSIM
captures local structural patterns that magnitude-correlation
misses. For our supervision target (a complex cube), the natural
analog is SSIM applied to the magnitude.

### 4.2 Implementation

In `mm25DGS_v6/losses/rda_losses.py`:

```python
def rd_magnitude_ssim_loss(rd_pred, rd_gt, window_size=7):
    # rd_pred, rd_gt: (D, N, R) complex
    mag_pred = rd_pred.abs()
    mag_gt   = rd_gt.abs()
    # Apply 2D SSIM per-Doppler-slice in (N=192, R=256). Sum.
    # Could also do 3D SSIM over (D, N, R); start with simpler 2D.
    ...
    return 1.0 - ssim_value
```

Combine:
$$L_{total} = L_{Gram} + λ_{SSIM} \cdot L_{SSIM}$$

λ_SSIM = 0.2 to start (3DGS standard); re-tune if needed.

### 4.3 M4 benchmark

Stack on M3. Expected: +0.01 – 0.02 mean cc.

---

## 5. End-to-end target

| Milestone | Changes | Expected `final_test_cc` mean (7 scenes) | Δ vs v5 baseline |
|---|---|---:|---:|
| v5 baseline | — | 0.5236 | — |
| v5 + S4 (current best) | pool_knn annulus [0.02, 0.10] | 0.5312 | +0.008 |
| **v6 M1** | + per-virt path plumbed (loss unchanged; scaffolding) | ≈ 0.524 (matches v5) | ≈ 0 |
| **v6 M2** | + Gram loss on (D, N, R) cube | **0.60 – 0.70** | +0.08 – 0.18 |
| **v6 M3** | + Doppler-envelope gating | 0.61 – 0.72 | +0.09 – 0.20 |
| **v6 M4** | + SSIM loss | 0.62 – 0.74 | +0.10 – 0.22 |

**Step 2 (Gram + Doppler) is the dominant lever.** If M2 lands in
the upper range (≥ 0.68), M3 + M4 push comfortably past 0.70.

---

## 6. Implementation order

Strict dependencies:
- **M1** (per-virt path + diagnostic) — independent. Ship first.
- **M2** (Doppler synthesis + Gram loss) — depends on M1's per-virt
  output infrastructure.
- **M3** (Doppler gating) — depends on M2's RD cube.
- **M4** (SSIM) — depends on M2's RD cube.

Suggested sequencing (single developer):
1. **Pre-M1**: read `mm25DGS_v6/rasterizer_factorized.py` to
   confirm B1 modification feasibility (scatter unrolled across
   chirps) and get a real LOC estimate. ~30 min.
2. **M1**: implement per-virt GT + render path + diagnostic;
   validate against v5 FFT-RA bit-identity; 7-scene benchmark.
   ~half day.
3. **M2**:
   a. Write `validate_doppler_synthesis.py`, run, confirm Option B1
      meets the > 0.95 threshold. ~1–2 hours.
   b. Implement Doppler-rotated scatter in renderer + slow-time
      FFT + Gram loss. ~1 day.
   c. 7-scene benchmark. Ablation: D=1 (no Doppler) vs D=16. ~30
      min wallclock.
4. **M3** layer. ~half day.
5. **M4** layer. ~half day.
6. Final 7-scene sweep + commit. ~half day.

Total: **3–4 developer-days**.

---

## 7. mm25DGS_v6 directory layout (revised)

```
mm25DGS_v6/
├── __init__.py
├── cuda/                     # CUDA extension — unchanged from v5
│   └── ...
├── data/                     # NEW
│   ├── __init__.py
│   └── ra_utils.py           # adc_to_per_virt_range_profile (M1) +
│                             # adc_to_rd_cube (M2)
├── renderer/                 # NEW
│   ├── __init__.py
│   ├── per_virt_render.py    # render_per_virt_complex (M1) +
│                             # render_rd_cube_analytic (M2)
│   └── doppler_synthesis.py  # per-point Doppler scatter helper (M2)
├── losses/                   # NEW
│   ├── __init__.py
│   └── rd_losses.py          # normalised Gram loss + SSIM (M2, M4)
├── rasterizer.py             # may need a per-chirp-output mode (M2)
├── rasterizer_factorized.py  # ditto (assess at start of M1)
├── train_gaussian.py         # unchanged (v5 primitive)
├── train_frame_nvs.py        # extended: M1 diagnostic + M2 loss +
│                             # M3 gating + M4 SSIM via --v6_milestone
├── train_chirp_loop_nvs.py   # unchanged (ported from v5; not used
│                             # by v6 main path)
├── load_pretrained.py
├── scripts/
│   ├── validate_per_virt_path.py
│   └── validate_doppler_synthesis.py
└── preprocessing/            # ported from v5 as-is
    └── alignment/
        └── ...
```

Policy:
- v5 stays untouched. v6 imports nothing from v5 source at runtime.
- Outputs go to `mm25DGS_v6/output_frame_nvs/`.
- v6 run-dir tags include `v6M1` / `v6M2` / etc. for pivot filtering.

---

## 8. Validation & benchmarking

### 8.1 Unit validations (per milestone)

- M1: `validate_per_virt_path.py` — bit-identical to v5 FFT-RA on
  one frame. Pass: max abs diff < 1e-5.
- M2: `validate_doppler_synthesis.py` — analytic synthesis vs
  multi-pose reference, complex correlation > 0.95 on full cube.

### 8.2 Regression benchmark (mandatory per milestone)

After every milestone, run the 7-scene HO_8 benchmark with the new
code at that milestone's config. Compare per-scene + 7-scene-mean
`final_test_cc` (the bit-identical FFT metric from v5) to the
previous milestone. Also report `diag_normalised_gram_cc_test` per
scene.

Scripts:
- `run_v6_M1.sh` — 7 scenes with per-virt diagnostic plumbed
- `run_v6_M2.sh` — adds Gram loss
- `run_v6_M3.sh` — adds Doppler gate
- `run_v6_M4.sh` — adds SSIM

### 8.3 A/B checks

- **M1**: `final_test_cc` must be bit-identical to v5 (within MC
  noise). Drift signals a bug in per-virt plumbing.
- **M2**: ablate D=1 vs D=16 (no Doppler vs full Doppler). The
  D=16 variant must beat D=1 by ≥ +0.03 cc on ≥ 5 of 7 scenes; if
  not, the Doppler analytic synthesis has a bug or the ego-velocity
  estimate is wrong.
- **M3**: ablate `gate_tolerance_bins ∈ {0.5, 1.5, 5.0}`.
  tolerance→∞ must recover M2.
- **M4**: ablate `λ_SSIM ∈ {0.0, 0.1, 0.2, 0.5}`. λ=0 must recover
  M3.

---

## 9. What the user did manually before development

### 9.1 Created the v6 tree (already done)

```bash
cp -r /home/adnan/Desktop/mm3DGS/mm25DGS_v5 /home/adnan/Desktop/mm3DGS/mm25DGS_v6
cd /home/adnan/Desktop/mm3DGS/mm25DGS_v6
rm -rf output_frame_nvs output_chirp_loop __pycache__ 2>/dev/null
```

### 9.2 Smoke-test v6 ≡ v5

Before any v6 changes, run `train_frame_nvs.py` inside v6 on one
scene. `final_test_cc` must match the corresponding v5 run within
MC noise. If not, the copy is dirty.

### 9.3 Rebuild CUDA extension if anything under `cuda/` is touched

```bash
cd /home/adnan/Desktop/mm3DGS/mm25DGS_v6/cuda
python setup.py build_ext --inplace
```

The B1 Doppler-rotated scatter modification (M2 §2.2) MAY require
a CUDA kernel update if the scatter step is in CUDA. That's the
biggest risk in the cost estimate; assess at the start of M2.

---

## 10. References

- **Gram-vs-FFT derivation**: [`md/gram_vs_fft_derivation.md`](gram_vs_fft_derivation.md)
- **3DGS**: Kerbl et al., "3D Gaussian Splatting for Real-Time
  Radiance Field Rendering", SIGGRAPH 2023.
- **DART**: Chen et al., "DART: Implicit Doppler Tomography for
  Radar Novel View Synthesis", ArXiv 2403.03896, 2024.
- **Radar Fields**: Bourdon et al., 2024.
- **Cumming & Wong**: "Digital Processing of Synthetic Aperture
  Radar Data", Artech House, 2005, Ch. 6.
- **mmIR (our prior)**: `/home/adnan/Desktop/mm3DGS/mmir/` —
  validated FFT-based azimuth processing. The Gram loss strictly
  subsumes this (see derivation §5).

---

## 11. Known risks / caveats

- **Rasterizer modification cost** (M1 + M2): the renderer's
  internal "scatter into output bins" step needs to expose the
  per-virtual coherent contributions (M1) and accumulate into 16
  Doppler-rotated output slices (M2 Option B1). If the scatter is
  in a CUDA kernel, this requires a kernel update. Realistically:
  ~150 LOC (Python) or ~250 LOC (Python + CUDA). Assess at start of
  M1.
- **Doppler synthesis approximation** (M2 Option B1): assumes
  linear ego-motion within the 7.87 ms burst. For jerky walking
  this may deviate; fall back to Option A (16× cost) if §2.2.1
  validation fails. Option A puts M2 at HO_128-style cost — gated
  on user re-discussion before proceeding.
- **Dynamic scatterers** (M3): seq_0_frame_390 has been an
  outlier across every prior approach. If M3 gating specifically
  rescues seq_0_390's cc, that's evidence the outlier behaviour is
  dynamic-contamination-driven. Worth reporting either way.
- **Gram loss numerical stability**: the denominator `‖v_p‖² ‖v_g‖²`
  can be very small if either vector has near-zero magnitude (e.g.
  in low-return range bins). Use eps=1e-12 to avoid divide-by-zero.
- **Memory**: stored autograd tensor `\tilde V` is `(16, 192, 256)
  complex = 6.3 MB / frame`, 50 MB / iter at 8 train frames. Fine.
  GT cube is computed once per frame and cached.

---

## 12. What this plan does NOT do (intentionally deferred)

- **Per-primitive spatial footprint / scale** (3DGS surfels) —
  considered in design review and dropped (would require rasterizer
  integration of finite-area surfel response; not a post-hoc kernel
  trick). Net cost ~300+ LOC of rasterizer work, possibly CUDA.
  Deferred to v7 if v6's M1–M4 stack caps below 0.70.
- **DBF (digital beamforming) on the supervision side** —
  considered in design review and dropped. The Gram loss strictly
  subsumes any beam-formed supervision (see derivation §5), uses all
  192 virtuals correctly without elevation assumptions, and skips
  one processing step entirely. DBF may still be useful as a
  diagnostic-only metric (logged, never used to promote).
- HO_128 variant runs (per user deprioritisation).
- Per-frame appearance embeddings (NeRF-W).
- Learned BSDF beyond the 6-param ITU family.
- Joint pose optimisation.
- Multi-scene transfer / meta-learning.
- Low-rank material field (S5 from v5 next-steps).
- Pass-3 per-chirp anchors stacked with v6.
- Per-scene hyperparameter tuning (single config must apply to all
  7 scenes).

---

## 13. Scope-reduction rationale (design review, 2026-04-19)

The v6 plan went through two rounds of structural rethinking:

**Round 1**: The original v6 plan included a 3DGS-inspired
"per-primitive scale" step. That proposal was structurally
incompatible with the current rasterizer, which models each point
as a delta scatterer with no spatial extent — adding a scale
parameter would make it a no-op unless the rasterizer were
simultaneously rewritten to integrate each surfel's projected
footprint over range-azimuth bins. A proper surfel rasterizer is
substantial work (~300+ LOC, potentially CUDA-level) and its
physical justification requires separate validation. Deferred to v7.

**Round 2**: The plan then proposed DBF (digital beamforming) as
the GT processing step on the supervision side. On scrutiny, this
had two problems: (1) azimuth-only DBF on all 192 pairs implicitly
assumes target elevation = 0 and introduces phase error for elevated
scatterers; (2) the Gram-matrix supervision (Option C) strictly
subsumes any beam-formed loss — beam-forming is a lossy projection
of the Gram. The Gram loss uses every pairwise relative-phase
relationship across all 192 virtuals correctly and is invariant to
absolute phase, matching the user's "we don't care about absolute
phase" constraint exactly. DBF is dropped from the supervision side;
v6 supervises directly on the per-virtual complex tensor via the
Gram loss.

What remains is the cleanest possible v6: keep the rasterizer
primitive, expose its per-virtual complex output, supervise on the
full complex per-(Doppler, virt, range) tensor with a Gram-matrix
loss that is mathematically equivalent to "full pairwise relative-
phase agreement" while being computationally O(N·D·R) per frame and
using only ~6 MB of stored autograd memory per frame.
