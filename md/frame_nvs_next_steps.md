# Frame-NVS — next steps after Analyses G + I

Date: 2026-04-19 (revised after matched-grid rerun)
Feeds from:
- [`md/frame_nvs_analysis/findings.md`](frame_nvs_analysis/findings.md) (prior grid)
- [`md/frame_nvs_analysis_matched_grid/findings.md`](frame_nvs_analysis_matched_grid/findings.md) (**matched grid — supersedes the prior conclusions**)
- [`md/frame_nvs_fisher_concentration_analysis.md`](frame_nvs_fisher_concentration_analysis.md)

S1 is complete. Matched-grid results: HO_128 → 0.512 (seq_1) / 0.590
(seq_2); HO_8 → 0.567 / 0.563. That is +0.11–0.18 cc pure position-grid
lift net of the target_n cost. **Still below the 0.70 hard target.**

Matched-grid swap ablation (§9 of the matched-grid findings) shows the
remaining gap is **rotation-dominated**: swapping HO's normals for UB's
(keeping HO's materials) gains +0.17 cc on seq_1 and +0.05 on seq_2,
while the reverse swap gains only +0.06 / +0.01. The aggregate drift
and smoothness statistics hide this — but weighted to the top-1 %
Fisher points (~200 of 20 000), HO's normals drift **17 ° more than
UB's on seq_2**. **S2 accordingly pivots from materials to rotations.**

## 0. Summary (updated 2026-04-19 after S2 + S4 experiments)

| Stage | What | Measured result | Status |
|---|---|---|---|
| **S1** | `seed_frame = test_frame` (matched grid), target_n = 20k | seq_1 HO_128 +0.021, HO_8 +0.067 · seq_2 HO_128 +0.086, HO_8 +0.052 | ✅ shipped |
| **S2** | Fisher-weighted rotation drift (init / EMA / thresh-45° targets) | seq_1 HO_8 +0.024–0.030 (best EMA α=0.9 λ=10); seq_2 HO_8 ≤ +0.006; **all three targets hurt HO_128 on both scenes** | ⚠ marginal — sign-correct on HO_8 only |
| **S3** (reserved) | Fisher-weighted material drift | Analysis I bound: +0.01–0.06 | not implemented (S4 dominates) |
| **S4** | Fisher-based adaptive density (split top + prune bottom, budget-preserving) | **seq_1 HO_128 +0.132, seq_1 HO_8 +0.098**; seq_2 HO_8 +0.017 (alt cfg 0.10/100); seq_2 HO_128 pending | ✅ **main result** |
| **S5** (reserved) | Low-rank material field | Multi-week refactor | not needed yet |

**Current HO cc status (S1 + S4 best per scene, matched grid target_n=20k):**

| scene | HO_128 | HO_8 | 0.70 target short-by |
|---|---:|---:|---:|
| seq_1_frame_438 | **0.6440** | **0.6656** | 0.034–0.056 |
| seq_2_frame_105 | **0.6057** | **0.5807** | 0.094–0.119 |

seq_1 closed 93 % of the swap-ablation gap; seq_2 still responds
only modestly. Path to 0.70 is unclear on seq_2 with parameter-space
regularisation alone; likely needs either (a) a different S4 cell
(pos_jitter × split_frac × interval grid), (b) pass-3 per-chirp
anchors stacked with S4, or (c) escalation to S5.

**Rejected hypotheses** (from matched-grid analyses — see
`md/frame_nvs_analysis_matched_grid/findings.md`):

- **H1 (uniform L2 drift on materials)** — A + B show HO/UB drift
  identically in aggregate; nothing to constrain uniformly.
- **H2 (k-NN spatial TV on materials)** — D: UB ≤ 2 % smoother than HO.
  Gap too small to regularise.
- **H3 (uniform L2 on rotation drift)** — B + E: HO/UB normals drift
  identically on average. Uniform penalty would shrink both equally;
  no cc gain. **Superseded by S2 above** (the correct form is
  Fisher-weighted, not uniform).
- **H4 (k-NN spatial TV on normals)** — E: normal-field smoothness
  identical.
- **H8 (LiDAR-intensity prior)** — J: intensity explains ~0 % of
  radar-material variance. Ruled out on physics (905 nm vs 3.9 mm).
- **Radar-aware FPS init** (user rejection, 2026-04-18) — would bias
  the scene representation. Geometric importance weighting
  (cos_bore + visibility + FPS) stays as-is.

---

## Stage 1 — re-run 4 variants on the unified (test-frame) position grid

**Done in code this commit.** `train_frame_nvs.py` now defaults
`seed_frame = test_frame` (previously `train_frames[len(train_frames)//2]`,
which selected `test_frame + 1` for HO). HO and UB variants now see the
same FOV + visibility + FPS → same 20 k point subset of `pcl.npy`
(`target_n` reduced from 90k to 20k per user request — reduces DOF by
4.5× at ~0 cc cost, see `md/frame_nvs_fisher_concentration_analysis.md`
§2.2 (F)).

### What to run

Two scenes × 4 variants (HO_128, HO_8, UB_144, UB_9) = 8 runs; assign
one scene per GPU, 4 runs sequential per GPU.

### Expected results

- If the grid was the sole bottleneck → **HO cc on new grid ≈
  UB-on-HO-grid estimate (0.52 / 0.55) OR even higher** because the new
  HO grid matches UB's geometry exactly (no NN-remap noise). Gap to
  UB-native (0.91 / 0.82) would shrink from 0.42 → 0.30 or less.
- If the grid is only part of the story → HO cc stays at 0.49 / 0.50
  with the new default. In that case S2/S3 are needed.

### Analyses to rerun

Rerun the small subset of `md/frame_nvs_analysis/` that depends on
the HO grid:

- **0** (reproducibility on new checkpoint).
- **G** (per-point Fisher on new HO grid) — the concentration pattern
  may shift if the 90 k subset is now biased toward the test-pose
  scatterer strength.
- **I** (swap ablation). With identical grids this collapses to a
  direct numeric comparison (no NN-remap noise), which is much cleaner.

### Caveat (from findings.md §6)

"Seed at test_frame" changes what "HO" means — the position grid is
now biased toward the test-frame geometry. This is still a valid NVS
evaluation (no test *radar* data leaks in; only the geometry for
sampling), but should be documented as an HO convention change.

---

## Stage 2 (revised) — Fisher-weighted L2 on rotation drift

### Rationale

Matched-grid swap ablation (§9 of
[`md/frame_nvs_analysis_matched_grid/findings.md`](frame_nvs_analysis_matched_grid/findings.md)):

| scene | HO native | `HO_mat + UB_rot` | Δ | `UB_mat + HO_rot` | Δ |
|---|---:|---:|---:|---:|---:|
| seq_1_frame_438 | 0.512 | **0.685** | **+0.174** | 0.573 | +0.061 |
| seq_2_frame_105 | 0.590 | **0.639** | **+0.048** | 0.599 | +0.009 |

Swapping only the rotations (from HO to UB) is the bigger single fix
on both scenes. Aggregate normal drift is identical between HO and UB
(28 ° / 33 ° mean), but Fisher-weighted to the top-1 % points (~200
of 20 000), HO's normals drift **17 ° more than UB's on seq_2**
(69.7 ° vs 52.4 °). The ~200 high-Fisher points freely rotate during
HO training to fit train-set noise in a way UB's full-data
supervision suppresses.

This is **H3 with Fisher weighting** — uniform H3 was ruled out by
aggregate analysis but the Fisher-weighted form targets exactly the
right ~200 points.

### Change

In `train_frame_nvs.py`, add:

```python
--reg_fisher_rot_lambda FLOAT    # λ · Σ_i F_i · ||rot[i] − init_rot[i]||²
--reg_fisher_poses      INT      # # candidate test poses for Fisher (default 8)
--reg_fisher_warmup     INT      # warmup iters before building F_i (default 50)
```

Per-iter loss addition:

```python
# At warmup end, compute F_i = Σ_pose_k (∂cc / ∂rot_i[pose_k])² via backprop
#   where pose_k are K=8 candidate test poses interpolated along train trajectory
# Normalise: F_i ← F_i / F_i.max()   (puts top point at weight 1)
# Per-iter:
reg_rot = lambda_rot * (F_i * (rot − init_rot).pow(2).sum(dim=-1)).sum()
```

### Expected effect

Upper bound = matched-grid swap ablation (from UB-rot substitution):
- seq_1 HO_128: 0.685 cc (hits 0.70 target within noise)
- seq_2 HO_128: 0.639 cc (short of 0.70 — S3 stacks needed)

Practical lift likely to be 50–80 % of the swap bound, since the
regulariser pins toward *init* normals, not UB's (unknown) normals.

### Measurement plan

- λ sweep {1e-3, 1e-2, 1e-1, 1, 10} on seq_1 HO_128 (4 × 40 min =
  ~3 hr). Pick winner.
- Validate winning λ on seq_2 HO_128 without re-tuning.
- If seq_1 hits 0.70 but seq_2 doesn't, stack with S3.
- Report HO cc, train cc (must not regress > 0.05), and
  Fisher-weighted drift-at-top-1% (should drop from 69.7 ° → closer
  to UB's 52.4 ° on seq_2).

---

## Stage 3 (reserved) — Fisher-weighted L2 on material drift

Same structure as S2 but on `raw_materials`. Swap ablation shows a
smaller but still positive material-side contribution (+0.06 / +0.01
over HO native). Ship only if S2 lands seq_1 cleanly but seq_2 still
needs the stack to cross 0.70.

```python
--reg_fisher_mat_lambda FLOAT    # λ · Σ_i F_i · ||raw[i] − init_raw[i]||²
```

Cost: ~30 LOC once the Fisher machinery from S2 is in place (the F_i
buffer is shared).

---

## Stage 4 — adaptive point density (reserved)

**Only if stages 1–3 cap below 0.70.** Periodically split high-Fisher
points (inheriting parameters) and prune low-Fisher points. Mirrors the
3DGS densification pipeline. ~200 LOC including Adam-state surgery to
preserve optimiser momentum across splits/prunes. Not attempted until
we know stages 1–3 are insufficient.

## Stage 5 — low-rank material field (reserved)

**Only if stages 1–4 cap below 0.70.** Replace `raw_materials: (N, 6)`
with `basis: (K, 6) + weights: (N, K)` where K ~ rank(cart_corr)
estimated from Fisher. Multi-week refactor.

---

## Execution order (suggested)

1. ✅ **S1 complete** (4 variants × 2 scenes = 8 runs at target_n=20k).
   Matched-grid analyses done; findings at
   `md/frame_nvs_analysis_matched_grid/findings.md`.
2. **Implement S2 (revised)** — Fisher-weighted rotation drift
   penalty, ~80 LOC in `train_frame_nvs.py`. λ sweep on seq_1
   HO_128 (5 × 40 min = ~3 hr). Validate on seq_2 HO_128.
3. **If seq_2 doesn't cross 0.70 after S2**, stack with S3
   (Fisher-weighted material drift, ~30 LOC). 3×3 λ-grid for
   (λ_rot, λ_mat) on seq_1, validate on seq_2. Compute: +2 hr.
4. **If S2 + S3 still cap below 0.70 on seq_2**, escalate to
   S4 (adaptive density).
5. Once both scenes clear 0.70, run the full 9-scene benchmark with
   the winning (λ_rot, optional λ_mat). Compute: ~4 hr at target_n=20k.

---

## Open questions

- **Is "seed_frame = test_frame" legitimate as the default HO
  convention?** (findings.md §6 flags this as a design-review question.)
  Proposal: yes, because (a) UB already does this implicitly, (b) no
  test *radar* data is used during training — only the test *pose* is
  used to select a geometry subset, (c) the alternative (legacy HO seed
  at `test_frame+1`) is an arbitrary choice that happened to be
  downstream of a different indexing decision. Document this in the
  paper's "implementation details" section.

- **Does the position-grid unification change cross-pose
  generalisation?** S1 biases the 90 k toward the test pose's
  geometry. If we hold out a *different* test pose (e.g. `test_frame +
  10`), the HO grid seeded at the original `test_frame` is no longer
  "aligned" to the new test. This is the usual NVS locality trade-off;
  a `--seed_frame` CLI flag exists to probe it. Proposal: add a seed-
  frame ablation table to the paper supplementary.

- **How does pass-3 interact with stages S1–S3?** Independent: pass-3
  only changes per-chirp poses, not the point grid or Fisher
  concentration. Stack freely.
