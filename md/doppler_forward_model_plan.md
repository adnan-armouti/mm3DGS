# Doppler forward-model plan (M2 focused)

Date: 2026-04-22
Owner: Adnan Armouti
Status: plan

## Motivation (why Doppler, now)

The 7-scene HO_8 investigation has produced a specific, empirically-hard result:

- **Zero-model neighbour averaging beats v5/M1 on 5 of 7 scenes** (7-scene mean 0.547 vs v5 0.528). `½·(GT[F-1] + GT[F+1])` in Cartesian RA space gives this "free baseline." The single scene where v5 is clearly stronger is **seq_1_frame_438** (v5 0.567 vs naive-avg 0.328; +0.239) — the scene with the largest frame-to-frame scene change in the benchmark.
- Every non-Doppler supervision variant we tested — 8 Gram-family variants, 1 hybrid `L_v5 + λ · L_per-ant-mag` — lands in a band below v5 (0.36 – 0.49 mean). The best, `smooth_alpha`, closed the gap to −0.036 but did not beat v5.
- The `diag_normalised_gram_cc_test` (complex per-virt Gram at test) is pinned at ~1/N_virt ≈ 0.008 across every variant, including the 500-iter hybrid run. Per-virt complex phase at the test pose is **empirically un-learnable** at HO_8 pose gaps: 10 cm pose error scrambles ~25 λ of coherent phase at 77 GHz, and our positions are frozen at cm-level LiDAR-FPS accuracy.

The common mechanism: the signal v5 actually supervises — `|RA|` magnitude on 86 azimuth virtuals — is **both pose-robust** (magnitude after linear projection kills the pose-scrambled absolute phase) **and near the information ceiling** of what the 192-virt geometry can deliver at HO_8. The 48 "extra" elevated-row measurements add ~no useful angular DOF (elevation aperture too small). The 66k extra Gram DOF beyond `|RA|` live in the complement of `(FFT → magnitude)`, and at HO_8 that complement is net-noise.

What HAS orthogonal, pose-robust, and learnable content that v5 does not supervise: **the slow-time / Doppler axis.** The benchmark ADCs contain 16 chirps per frame. v5 uses only chirp 0. The other 15 chirps carry:

- Radial-velocity information per scatterer (encoded as slow-time phase slope `f_d = (2/λ)·<v_ego, ȓ_s>` per scatterer s, with `v_ego` from IMU or pose-difference).
- Pose-robust: a 10 cm pose error shifts Doppler bins by << 1 (Doppler scales with `v_ego`, not absolute position). Unlike per-virt phase, Doppler phase across chirps is NOT wavelength-scrambled at HO_8.
- 16-fold multiplicative increase in effective supervision DOF that is NOT the FFT-null-space we already showed is noise.

The 3DGS analog: Doppler is a **new conditioning axis** distinct from range and azimuth, not a refinement of them. The renderer needs to become a *range-azimuth-Doppler* renderer, not a *range-azimuth* renderer.

## Scope

This document scopes **only the forward-model side** of Doppler: what the renderer must produce, how the IMU feeds it, what the supervision tensor looks like. The loss side (Gram vs magnitude-RD vs hybrid) is TBD after the forward model lands and we can measure per-axis SNR; the existing §2.1 of `mm25dgs_v6_design.md` contains starting-point options that will be refined based on this doc's outcomes.

## What the forward model must produce

The per-frame supervision target becomes a 3-D complex tensor
$$
\tilde V \;\in\; \mathbb C^{\,D \times N \times R}
$$
with `D = 16` (slow-time Doppler bins), `N = 192` (virtual antennas, ADC channel order), `R = 256` (range bins after range FFT).

GT side (trivial extension of M1): `adc_to_per_virt_range_profile` already returns `(192, 256)` complex for a single chirp. Run it over all 16 chirps → `(16, 192, 256)` complex, apply slow-time Doppler FFT over the chirp axis → `(D=16, 192, 256)` complex. This is the per-frame GT RD-cube `\tilde V_gt`.

Pred side: the renderer currently emits `(n_tx=12, n_rx=16, K=256)` complex per chirp. We need it to emit **the same `(D, N, R)` cube** without 16× re-render cost.

## The key engineering trick: analytic per-point Doppler synthesis

Re-rendering 16 times per training sample is HO_128-class cost and an immediate non-starter. But the platform moves only ~1 mm in a 7.87 ms burst — <0.2 range-bin, << λ in amplitude — so every geometric quantity the renderer computes (ray test, BSDF, visibility, antenna gain) is **constant across the 16 chirps**. Only the per-point carrier phase evolves linearly.

For a scattering point `p`, let `c_0(p, n, r)` be its chirp-0 complex contribution to virtual `n` at range `r`. The chirp-`k` contribution is analytically
$$
c_k(p, n, r) \;=\; c_0(p, n, r) \cdot \exp\!\bigl(j\,2\pi\,f_d(p)\,k\,\Delta t\bigr)
$$
with the per-point Doppler frequency
$$
f_d(p) \;=\; \frac{2}{\lambda} \,\bigl\langle v_{ego},\ \hat r(p)\bigr\rangle
$$
where `v_ego` is the platform ego-velocity at the burst center (scalar 3-vector per frame; see §"IMU / pose-difference" below) and `\hat r(p)` is the unit vector from the radar center to point `p` (per-point O(1) evaluation, burst-constant). No ray-tracing re-run. The only cost above the existing single-chirp renderer is: in the scatter step, loop over `k = 0..15` and scatter each point's contribution into 16 output slices instead of one.

Expected per-iter cost: **1.2 – 1.5× the existing single-chirp cost**, NOT 16×.

## Validation of the analytic synthesis (MANDATORY before locking M2)

Option A (reference, slow): re-render at each of the 16 chirp-specific poses individually → stack into `(16, N, R)`. This is the exact ground-truth prediction we want to approximate.

Option B1 (our proposal above): render once at the chirp-0 pose, apply the per-point analytic Doppler factor during the scatter step.

`mm25DGS_v6/scripts/validate_doppler_synthesis.py` must:
1. On `seq_0_frame_135`, run A and B1 to produce two `(16, 192, 256)` cubes.
2. Complex normalised correlation across the full cube → pass if `> 0.95`.
3. Peak-Doppler-bin agreement (±1 bin) for the top-25 scatterers by magnitude.
4. Peak-range-bin agreement (exact) for the same.

If B1 fails: fall back to A at 16× cost. M2 is then gated on an explicit per-iter-cost re-discussion — HO_128 territory is not automatic.

## Ego-velocity estimation

Two candidate sources; default to the first:

**1. Pose-difference (default).**
For training frame `F`, the aligned poses of the neighbours `F-1` and `F+1` define
$$
v_{ego}(F) \;\approx\; \frac{T_{w,b}(F+1).\text{pos} \;-\; T_{w,b}(F-1).\text{pos}}{2\,T_{\text{frame}}}
$$
with `T_frame = 0.1 s` (ColoRadar cascade). This assumes locally-constant velocity across the 200 ms neighbour span, which is a clean 2nd-order approximation when acceleration is bounded. Edge frames (no F-1 OR no F+1) use the one-sided difference.

**2. IMU (fallback / refinement).**
ColoRadar ships IMU records (linear accel, angular vel) in the per-sequence metadata. Integrating linear accel around the frame timestamp gives an independent ego-velocity estimate. Noisier per-sample but does not depend on pose alignment — useful as a cross-check and for edge frames where pose-difference is one-sided.

Pipeline: a small `mm25DGS_v6/preprocessing/ego_velocity.py` builds a per-frame `v_ego` dict keyed by frame index, cached to disk. Default implementation = pose-difference; `--ego_source imu` flag swaps in IMU-derived values for A/B.

## Rasterizer changes (the only risky surgical bit)

The current `mm25DGS_v6/rasterizer_factorized.py` emits per-chirp `(n_tx, n_rx, K)`. M2 needs it to emit `(D, n_tx, n_rx, K)` where the `D` axis is the slow-time Doppler FFT across chirps. Two-step implementation:

1. **B1 scatter loop unrolled across chirps.** The "accumulate per-point contribution into output bin" step now writes into `D` output slices, each multiplied by `exp(j · φ_k(p))` where `φ_k(p) = 2π · f_d(p) · k · Δt`. Everything else is unchanged.
2. **Slow-time Doppler FFT over the chirp axis.** Standard windowed FFT on `D = 16` — ~10 LOC.

Risk surface:
- If the scatter step is in a CUDA kernel (it is, via `mm25DGS_v5/cuda`), the B1 change requires a kernel update. Assess at the start of implementation; LOC estimate climbs from ~150 (Python) to ~300 (Python + CUDA).
- Float precision: per-point `φ_k(p)` is cheap; no regression risk.
- Gradient flow: `f_d(p)` depends on `\hat r(p)` which depends on point positions — which are FROZEN at LiDAR-FPS. So `f_d` contributes no learnable gradient. Only the per-point `c_0(p)` (driven by materials + rotations) carries gradient through the Doppler cube, as intended.

## Evaluation invariants

- `final_test_cc` stays **bit-identical** to v5 (the 86-subset FFT-RA-magnitude cart_corr). Doppler is ADDED; the promotion metric is not altered.
- New diagnostic: `diag_doppler_peak_match` = fraction of range bins where the Doppler-peak-bin of pred matches GT within ±1 bin. Clean physics check.
- Naive neighbour-averaging baseline must be recomputed per-test-frame as in Q1's sanity check. The correct ranking metric going forward is **`Δ = final_test_cc − naive_avg_cc`** — the ACTUAL rendering contribution. Expected at M2: clear positive Δ on all 7 scenes, not just seq_1_frame_438.

## Sanity checks before M2 sign-off

Before committing engineering time to the rasterizer surgery:

1. **Doppler visibility.** Compute `adc_to_rd_cube` on 1 scene's GT, view `|RD[test_frame]|` for the 16 Doppler bins. Expected: static-scene scatterers concentrate at `d = d_ego`, dynamic scatterers (cars, pedestrians) occupy nearby bins. If everything collapses to a single bin, the Doppler axis carries no discriminating information for our scenes and M2 is premature.
2. **Predicted Doppler spread.** Compute `f_d(p)` for every FPS-selected point in the seed-frame model, histogram. Expected: spread of ~0.5 – 2 Hz (at `v_ego ~ 5 m/s`, λ = 4 mm, this is bins 1..4 of a 16-bin cube over 7.87 ms). If < 1 bin of spread, the axis won't discriminate points — M2 becomes cosmetic.

Both checks are <30-min scripts and should land before we touch the rasterizer.

## Out of scope (intentionally)

- Per-chirp pose interpolation beyond the analytic-synthesis approximation. If B1 fails validation we re-discuss, but the design assumption is linear ego-motion within the 7.87 ms burst — valid to ~1 mm precision at our speeds.
- Dynamic-scene modelling (moving scatterers with their own velocities). Design continues to assume **rigid-scene, ego-motion-only Doppler**. Moving-scatterer extension is a v7+ problem once static-scene M2 works.
- M3 Doppler gating design. Separate doc once M2's RD cube quality is measured.
- Loss-on-RD-cube design. Separate doc; the cleanest starting point is "magnitude of `|RD[d, r]|` per-virt averaged to 86 via v5's packing, per-Doppler-bin mse_raw, summed across D." This preserves v5's pose-robust inductive bias on a NEW axis that IS learnable. Alternatives (per-bin Gram Frobenius², range-integrated over Doppler, etc.) TBD.

## LOC and schedule estimate

- Ego-velocity preprocessing + cache: ~80 LOC, 1–2 hours.
- `adc_to_rd_cube` GT side (range FFT + slow-time FFT): ~60 LOC, 2 hours (reuses M1 code).
- Rasterizer B1 scatter-loop modification: ~100 LOC Python + ~100 LOC CUDA, 1–2 days.
- `validate_doppler_synthesis.py`: ~80 LOC, 2 hours.
- Trainer integration + loss dispatch: ~60 LOC, half day.
- 7-scene bench + sanity checks: half day.

Total: **3 – 4 engineering-days** before we see a result.

## Gating decision

Implement Doppler **only if** the two pre-M2 sanity checks above show that

1. Doppler spread across scatterers is ≥ 2 bins in our benchmark, AND
2. The static-scene Doppler peak is distinguishable from dynamic-scene ones in at least 3 of 7 benchmark scenes.

If either fails: Doppler is structurally insufficient to carry the supervision signal the non-Doppler investigation left on the table, and we pivot to an entirely different axis (e.g. multi-frame temporal consistency, multi-view / multi-sensor transfer).

## References

- `md/mm25dgs_v6_design.md` §2 — original M2 spec (contains the analytic-synthesis argument this doc consolidates).
- `md/gram_vs_fft_derivation.md` — reason v5's `|RA|` is pose-robust and why the Gram complement at HO_8 is un-learnable.
- DART (Chen et al., ArXiv 2403.03896) §3.2 — Doppler-aided radar tomography; blueprint for §3's gating step (out of scope for this doc).
- Cumming & Wong, "Digital Processing of Synthetic Aperture Radar Data", Ch. 6 — Doppler beam sharpening as the SAR analog of "use motion to make sparse aperture work."
