# Why is Fisher ultra-concentrated in Analysis G?

Follow-up investigation for the frame-NVS findings in
[`md/frame_nvs_analysis/findings.md`](frame_nvs_analysis/findings.md) §1-G.
The headline in that document was:

> top 1 % of points (900 / 90 000) carry 93–99 % of test-cc sensitivity on
> every variant.

Two questions were raised:

1. **Why is it this concentrated?** Is it a bug in the pipeline, or
   expected behaviour?
2. **Are we missing a penalty / regularisation term?** What options do we
   have to address it?

This note answers both.

---

## 1. Decomposition: where does the concentration come from?

Using the saved `G_fisher_<scene>_<variant>.npz` arrays and the matching
`best_model.pt` position grids, the per-point Fisher
`F_n ≡ Σ_col (∂ cart_corr / ∂ raw_materials_{n,col})²` was regressed
against three candidate explanations:

### 1.1 Is it FOV culling (active-mask = 0)?

**No.** Re-running the FOV cull at each saved test pose (same logic as
[`cull_gaussians`](../mm25DGS_v5/train_gaussian.py#L479)):

| scene / variant | N | exactly 0 (mask off) | ≥1e-15 | max / median |
|---|---:|---:|---:|---:|
| seq_1_f438 HO_128 | 90 000 | 5 077 (5.6 %) | 84 923 (94.4 %) | **6.4 × 10⁸** |
| seq_1_f438 UB_144 | 90 000 | 5 153 (5.7 %) | 85 041 (94.4 %) | 9.2 × 10⁷ |
| seq_2_f105 HO_128 | 90 000 | 10 164 (11.3 %) | 79 945 (88.8 %) | 3.2 × 10⁹ |
| seq_2_f105 UB_144 | 90 000 | 10 016 (11.1 %) | 80 232 (89.1 %) | 4.5 × 10⁸ |

5–11 % of points are FOV-invisible at the test pose; the remaining
88–94 % **are** active and produce non-zero gradient. The concentration
is within the active set, not between active/inactive.

### 1.2 Is it range / boresight geometry?

**Partially.** Per-range-bin Fisher share on
[`seq_1_frame_438/HO_128`](frame_nvs_analysis/G_fisher_seq_1_frame_438_HO_128.npz)
at the test pose:

| range band | n active | Σ F (% of total) | median F |
|---|---:|---:|---:|
|  1.5–3.0 m |  1 712 |   0.0 % | 8.6e-11 |
|  3.0–5.0 m |  8 182 |   0.0 % | 4.3e-10 |
|  5.0–8.0 m | 34 515 |   7.8 % | 1.4e-08 |
|  8.0–12.0 m | 27 458 |   0.6 % | 7.8e-10 |
| **12.0–18.0 m** | **18 092** | **91.6 %** | **4.1e-08** |

One range band (12–18 m, 20 % of active points) carries **91.6 %** of
the total Fisher. The near-field (< 5 m, 10 k points) carries
essentially **zero** despite its 1/r² amplitude advantage.

The same pattern on [`seq_2_frame_105/HO_128`](frame_nvs_analysis/G_fisher_seq_2_frame_105_HO_128.npz):
40.2 % from 8–12 m and 43.0 % from 12–18 m; and on
[`seq_1_frame_438/UB_144`](frame_nvs_analysis/G_fisher_seq_1_frame_438_UB_144.npz):
86.3 % from 12–18 m.

### 1.3 So where is it coming from?

**The scene's strong scatterers.** At 77 GHz (λ ≈ 3.9 mm) radar returns
are dominated by *coherent* reflections from a small number of
facets where:

1. the surface normal is near-aligned with the
   direct-reflection-to-RX direction (Fresnel / slab reflectivity peaks);
2. the range cell coincides with a concentration of such facets (buildings,
   poles, large smooth surfaces);
3. the antenna pattern × FSPL × BSDF product produces the largest
   amplitude contribution to a single `(range, azimuth)` cart cell.

The top-1% points on seq_1_f438 sit at `dist_median = 13.9 m,
cos_bore_median ≈ 0.99` — they are the building wall that dominates the
RA image in that cart region. The other 99 % of points either:

- fall in range cells where the RA-image energy is small (ground,
  low-return regions → cc gradient tiny);
- fall in range cells where coherent phase cancellation wipes out the
  amplitude even though individual points are active;
- fall on facets whose normals are >45° off the specular direction
  (mmWave BSDF is much "shinier" than optical — Fresnel × Kirchhoff
  roll-off is steep).

The `max / median` ratio of **10⁸–10⁹** reflects 8–9 decades of
amplitude variation across points, which is what you expect when a few
facets produce ~0 dB of coherent gain and the rest are ≤ –80 dB
relative to them.

### 1.4 What does NOT drive the concentration

- **Not** overfitting / optimisation drift — the Fisher is measured
  from `cart_corr` directly against GT at the saved checkpoint. It is a
  property of the **rendering objective at that pose**, not a training
  artefact.
- **Not** a bug in FOV culling — 88–94 % of points pass the mask.
- **Not** a bug in raw-material parameterisation — the same
  concentration appears in all six material channels, and in all 8
  saved variants (HO/UB × 2 scenes × full/first-chirp).
- **Not** an effect of training loss type — the Fisher uses
  `cart_corr` directly (the evaluation metric), not `mse_raw`.

---

## 2. Is the training pipeline missing a regulariser?

**No — but the training loss is missing a **structural** statement
about the low-rank nature of the objective.** What we are seeing is
the signature of a classic ill-posed inverse problem:

- **Parameters**: 90 000 points × 6 material columns = **540 000 DOF**
  (plus 4 rotation columns × 90 000 = 360 000 quaternion DOF).
- **Training signal rank**: `cart_corr` is computed from a
  `127 × 256 = 32 512` cart-image. Of those pixels only O(10³) have
  meaningful signal. The effective rank of the Jacobian
  `∂ cart_corr / ∂ raw_materials` at a single pose is therefore at
  most O(10³).
- Training across 16 × 8 = 128 frames-loops widens the joint rank by
  at most a constant multiplier (the underlying scatterer geometry
  doesn't change much). Empirically G says ~900 points drive the
  test-cc at any given pose, with 10–20 % overlap between poses — so
  the effective joint rank is perhaps a few thousand.
- That leaves **≥ 95 % of the parameter DOF in a gradient-null
  subspace of `cart_corr` at training time**. Any unregularised
  optimiser will wander freely through that null space; the wandering
  **does not hurt train cc** (by definition — the gradient is zero there)
  but it **does hurt test cc** because the null-space at train poses
  only partially overlaps the null-space at a different test pose.

This is not a bug. It is the fundamental reason NVS is hard with
sparse measurements + millions of parameters. What we are missing
is an explicit acknowledgement in the loss / optimiser that the
problem is low-rank.

### 2.1 What have we already tried? (recap from findings.md)

- **H1 — uniform L2 drift from init**: shrinks the null-space
  wandering uniformly. +0.01–0.04 test cc. Works but is blunt: it
  also constrains the ~900 points we *want* to move.
- **H7 — Adam-v activity mask**: freeze the bottom `(1 − frac)` of
  points whose Adam `exp_avg_sq` is low (training-proxy for Fisher).
  Also +0.01 test cc, but train-cc regression 0.02 (borderline). Too
  aggressive at top_frac = 0.01; fine at 0.10.

Both H1 and H7 are **valid but cap out** at the ~0.52 cc ceiling
imposed by the position-grid mismatch (Analysis I).

### 2.2 Options that would address Fisher concentration directly

Ranked by expected impact × implementation cost:

#### (A) Use the test-pose trajectory, not just the test pose, to define the active set

Currently Fisher is measured only at the held-out test pose (which
we don't have access to during training). The training-proxy (Adam
`exp_avg_sq`) is computed at train poses only, so a point that is
high-Fisher at the test pose but low-Fisher at train poses looks
inactive and gets frozen by H7.

**Fix**: during training, periodically compute Fisher at a set of
*candidate test poses* — e.g. pose-interpolated along the
trajectory bracketed by train frames (but held out from gradient
updates). The union of high-Fisher sets across those candidates is a
better activity mask than the train-only proxy.

Effort: ~50 lines. Expected: +0.02–0.04 test cc on top of H1/H7 by
reducing the false-freeze rate.

#### (B) Fisher-weighted per-point learning rate (soft H7)

Instead of a hard mask, scale the per-point LR by
`min(1, F_n / F_ref)` where F_n is the train-time Fisher proxy. Keeps
gradients alive on all points but shrinks updates on low-Fisher
points, reducing null-space wandering without the "cliff" that H7
imposes.

Effort: ~30 lines (Adam `param_groups` with per-point scaling). Expected:
+0.01–0.02 test cc; compatible with H1.

#### (C) Adaptive point density (3D Gaussian Splatting-style)

Periodically **split** high-Fisher points (subdivide into N
sub-points near the same location, inheriting parameters) and **prune**
low-Fisher points. Targets the 540 k-DOF problem directly by
reallocating parameter budget to the ~900 points that matter and
removing budget from the null space.

Effort: ~200 lines (point-set surgery, Adam state surgery to preserve
the split/pruned subsets). Expected: +0.05–0.10 test cc in the
best case if the split points generalise to test poses; risk is that
the split concentrates too hard on *training*-dominant points. Would
benefit from (A) above.

#### (D) Low-rank material field

Replace per-point raw_materials (90 k × 6) with a low-rank
factorisation: e.g. K basis vectors × 6 material columns + per-point
weights of shape (90 k × K). This enforces the low-rank structure
*in the parameterisation* rather than relying on the optimiser to
discover it. Most common in NeRF-type scenes; well-studied.

Effort: ~500 lines (new parameterisation + init + re-projection at
save time). Expected: +0.05 test cc and 5–10× training speedup; risk
is that K is hard to tune and a too-small K truncates the ~900
high-Fisher DOF.

#### (E) Spatial low-rank regulariser on materials & normals

When the per-point budget is large relative to the scene's effective
scatterer count, the *spatial material/normal field* should be
low-rank: nearby points on the same surface should share parameter
values. This is a generic structural prior — it assumes **only**
that materials and normals vary smoothly over physical surfaces, not
that particular points or ranges are more important than others (so
it does not bias the scene representation the way radar-amplitude or
LiDAR-intensity weighting would).

Implementation options:

- **k-NN spatial TV on raw materials** (proper name for H2). Penalise
  `Σ_{(i,j)∈NN} ||raw[i] − raw[j]||²` so neighbouring points
  constrain each other. Already sketched as H2 in
  `md/frame_nvs_analysis/findings.md` but previously under-prioritised
  because Analysis D showed UB–HO smoothness differs by only ≤ 5 % on
  the *old* (mismatched) position grids. With matched grids the
  smoothness signal may become stronger or weaker; rerun Analysis D
  after Stage S1 before committing λ.
- **k-NN TV on trained normals** (H4). Same structure, over
  `(1 − n_i · n_j)`. Low prior unless the post-S1 rerun of Analysis E
  shows a difference.
- **Low-rank factorisation of the spatial material field itself** —
  compute the SVD of `raw_materials (N × 6)` → keep top-K spatial
  modes. Harder to train directly; easier as a *post-hoc projection*
  after convergence (verify it doesn't destroy cc).

Under-determination hint: the user's rationale is that with N large
(90 k → 540 k DOF) and only ~10³ effective rank in `cart_corr`, the
null space is where drift happens. A spatial TV penalty shrinks the
null-space wandering by tying neighbouring points together; it does
not change the geometry of the scatterer set.

Effort: ~60 LOC (kNN index once per run + per-iter loss term).
Expected: +0.02–0.05 test cc if post-S1 Analysis D shows a
UB–HO smoothness gap; +0 otherwise. Stacks cleanly with (A)–(D).

**This option is contingent on post-S1 Analysis D/E results** — it is
the right option *only if* the matched-grid analyses show UB's
material/normal field is spatially smoother than HO's. The prior-grid
analysis could not settle this because HO and UB were training on
different 90 k subsets, making "per-point neighbour-std" a mixed
effect of (position-grid density) + (parameter smoothness).

#### (F) Train on fewer target points (smaller N per run)

If the effective rank is O(10³) and we have 540 k DOF, the problem is
fundamentally under-determined by ~3 orders of magnitude. Dropping
`target_n` from 90 000 to, say, 10 000–20 000 reduces
over-parameterisation by 5–9× and (empirically, in 3DGS) usually
speeds convergence without hurting test cc.

Effort: ~1 line (`--target_n 20000`). Expected: ~0 test cc change
(the low-rank structure doesn't care about N past a few × rank);
3–5× training speedup. Chosen as the default for the S1 rerun so the
8-run matrix completes in ~8 min total.

---

## 3. Direct answer to the user's question

**Are we missing a penalty / regularisation loss term?** No — the
concentration is real physics (mmWave coherent scattering is dominated
by a small number of facets), not a bug. The training pipeline
correctly computes the gradient for every point; it is the *objective*
that is low-rank.

**What we did (2026-04-19):**

1. **Shipped the architectural fix** — `--seed_frame = test_frame`
   is now the default.

2. **Validated on the unified grid** — all 4 variants × 2 scenes
   re-trained at `target_n = 20 000`. Results in
   [`md/frame_nvs_analysis_matched_grid/findings.md`](frame_nvs_analysis_matched_grid/findings.md):

   - HO_128 lift: +0.086 (seq_2), +0.021 (seq_1)
   - HO_8 lift:  +0.052 (seq_2), +0.067 (seq_1)
   - UB cost from 90k→20k: ~−0.09 cc (isolated because UB's grid
     didn't change under the seed fix)
   - Net pure position-grid lift for HO: **+0.11–0.18 cc**

   Analysis A/B/D/E (now clean because no NN-remap noise) show HO
   and UB parameter/normal distributions are **aggregate-identical**.
   Rules out H1, H2, H3 (uniform), H4 definitively.

3. **Matched-grid swap ablation (Analysis I, now exact) revealed
   the rotation-not-material fault**: swapping only HO's normals
   for UB's gains +0.17 cc on seq_1 and +0.05 on seq_2, while the
   material swap gains only +0.06 / +0.01. The aggregate drift
   hides this — Fisher-weighted to the top-1 % points, HO's normal
   drift is **17 ° higher than UB's on seq_2**.

**What we will do next:**

The prioritised option from §2.2 has shifted from (A) on materials
to a Fisher-weighted L2 on **rotations** (S2 in
[`md/frame_nvs_next_steps.md`](frame_nvs_next_steps.md)). This is H3
with Fisher weighting — uniform H3 was ruled out, but the
Fisher-weighted form targets exactly the ~200 high-Fisher points
where HO's normals have over-drifted. Upper bound from swap:
seq_1 0.685 cc (clears 0.70), seq_2 0.639 cc (needs S2 + S3
material-drift stack).

(C) adaptive density and (D) low-rank material field remain in
reserve as multi-week refactors only if S2 + S3 cap below 0.70.

**What we will NOT do** (explicit rejections):

- **Radar-amplitude-weighted FPS init** would bias the scene
  representation toward currently-bright scatterers and degrade
  generalisation to novel viewpoints where different scatterers
  dominate.
- **LiDAR-intensity-weighted FPS init** has a modality/wavelength
  mismatch — LiDAR (~λ ≈ 905 nm) and mmWave radar (~λ ≈ 3.9 mm)
  interact with materials through entirely different physics;
  LiDAR intensity is not a reliable proxy for radar-return amplitude.
- **Changing the training loss from `mse_raw` polar to a cart-corr
  surrogate** — it would make concentration *worse*, not better.

---

## 4. Artefacts

- This analysis: `md/frame_nvs_fisher_concentration_analysis.md`
- Source Fisher arrays: `md/frame_nvs_analysis/G_fisher_*.npz`
- Source checkpoints: `mm25DGS_v5/output_frame_nvs/*/best_model.pt`
- Related: `md/frame_nvs_analysis/findings.md` (Analyses A–K).
