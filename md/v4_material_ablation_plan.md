# mm25DGS_v4 material model ablation plan

Goal: answer three questions in priority order.

1. **Do we even need a learnable, per-point, 6-parameter material model?** (Baselines question — must come first.)
2. **Which physical components of the BSDF are doing real work?** (Component ablation question — only if step 1 says we need a learnable model at all. Tells us which code paths to delete.)
3. **Within the surviving model, which of the 6 parameters are essential, which are redundant, which are inert?** (LOO/TOO question — only after step 2 has trimmed dead components.)

This plan deliberately ignores cross-sensor transfer. Decision metric for every run is the **mean cart_corr across the 7-scene benchmark** (already-cached GPU pipeline).

Baseline to beat: commit `8fe6d6d` (Tier A), 6-param per-point, mean cart_corr **0.9351**, ~52 s/scene × 7 scenes ≈ 6 min/run.

## The 6 parameters (3 functional groups)

| # | Param | Group | Role |
|---|---|---|---|
| 0 | `eps_real` | **Fresnel** | Real permittivity → reflection magnitude |
| 1 | `eps_imag` | **Fresnel** | Loss → absorption + SPM amplitude |
| 5 | `thickness` | **Fresnel** | Slab phase → multi-layer interference |
| 2 | `sigma_h` | **Roughness** | RMS height → GGX α, SPM/directive κ, coherence γ |
| 3 | `l_c` | **Roughness** | Correlation length → GGX α, SPM κ, CBS sinc |
| 4 | `tau_base` | **Mixing** | KA-vs-SPM coherent/incoherent blend |

All 6 init to ITU concrete `[5.31, 0.0326, 1e-4, 5e-3, 0.5, 0.15]`. Materials LR 0.7. The 6 enter the BSDF in highly coupled ways (e.g. `sigma_h` and `l_c` both feed GGX α via different paths), so any single ablation lens is insufficient — we need baselines + LOO + TOO + identifiability diagnostics together.

---

## Phase 1 — Baselines (decides whether the model class is worth keeping)

Four runs, each on 7 scenes.

| # | Name | Description | Free DOFs | Why |
|---|---|---|---:|---|
| B0 | **Fixed concrete** | All 6 params frozen at ITU concrete init. No material learning at all. | 0 | What does the geometry + antenna patterns + rotations alone get us? Hard floor. |
| B1 | **Scalar reflectivity** | Replace BSDF with `f_cos = sigmoid(α_m) · cos_i` where `α_m` is one learnable scalar per point. No Fresnel, no GGX, no SPM. | M × 1 | The simplest non-trivial model. If this matches B3, the physics machinery is decorative. |
| B2 | **Global 6-param** | One shared `(6,)` material vector for the whole scene, not `(M, 6)`. Same BSDF, same reparam, same LR. | 6 | Does per-point freedom actually do work, or is the scene effectively monolithic? |
| B3 | **Per-point 6-param** | Current Tier A baseline. | M × 6 | Reference point. Already have: 0.9351. |

**Decision rule for Phase 2:**

- If B3 ≤ B2 + 0.005: per-point freedom isn't earning its keep. **Stop here. Switch to global 6-param** as the production model. Skip Phase 2.
- If B3 ≤ B1 + 0.005: the BSDF physics isn't earning its keep. **Stop here. Switch to scalar reflectivity** as the production model. Skip Phase 2.
- If B3 ≤ B0 + 0.01: nothing is being learned at all. Go debug; don't run Phase 2.
- Otherwise: per-point 6-param wins; proceed to Phase 2 to find which of the 6 are doing the work.

Phase 1 cost: ~24 min (4 runs × 6 min).

---

## Phase 1.5 — BSDF component ablation (which physics terms survive?)

Only runs if Phase 1 confirms B3 ≫ {B0, B1, B2}. Runs **before** Phase 2 because removing dead components shrinks and clarifies the parameter ablation that follows.

### Why this is different from parameter freezing

Parameter freezing tells you "is this knob being used". Component ablation tells you "is this physical mechanism contributing". They overlap but aren't the same:

- `sigma_h` feeds **three** lobes (GGX α, SPM κ, coherence γ). If LOO says `sigma_h` is essential, you don't know *which* of the three lobes earned that verdict.
- `tau_base` only matters if both KA and SPM lobes are nonzero. If KA is doing nothing, `tau_base` is mechanically inert — but parameter ablation can't tell you whether it's `tau_base` or KA that's the dead weight.
- Parameter ablation can never tell you "delete this 200-line code path entirely". Only component ablation can.

The decision this answers: **which physics terms can we delete from the BSDF**, not just "which knobs can we freeze".

### The components in the v4 BSDF

From [rasterizer_factorized.py](../mm25DGS_v4/rasterizer_factorized.py) Step 4:

| # | Component | What it computes | Likely-droppable? |
|---|---|---|---|
| C1 | **Jones / multi-layer slab Fresnel** | Polarized reflection coefficient through dielectric slab with thickness | If radar is unpolarized at the receiver, the Jones machinery may be wasted — scalar `\|r\|²` could be enough. |
| C2 | **GGX / KA coherent lobe** | Cook-Torrance specular | The "standard" specular lobe — probably essential. |
| C3 | **SPM incoherent lobe** | Small Perturbation Method rough-surface scatter | Only differs from KA at large `k·σ`. May be redundant on smooth-ish surfaces. |
| C4 | **Directive (vMF) broadening** | Anisotropic angular broadening of specular | Layered on top of KA. Could be doing nothing if GGX α already covers it. |
| C5 | **Broad / diffuse lobe** | Wide-angle isotropic | The Lambertian fallback. May or may not matter depending on scene. |
| C6 | **CBS (coherent backscatter enhancement)** | Retroreflection peak via `sinc(K·l_c·sin θ)` | Only matters near the backscatter direction. Cheap to test, easy to drop. |
| C7 | **Coherence blend (γ, τ_eff, η)** | Smooth interpolation between coherent and incoherent | If KA and SPM end up dominated by one or the other, the blend math is decoration. |
| C8 | **Multi-layer slab thickness** | Phase delay through `q = (2π/λ)·d·√(η - sin²θ)` | Tests `thickness` *as a mechanism*, not just as a knob. |

### Run battery

Each component gets an `enable_<component>` flag in `render_factorized` that zeros the relevant term before the blend. One toggle per run.

| Run | Disable | Tests |
|---|---|---|
| K1 | C6 (CBS) | Does the retroreflection peak earn its sinc term? |
| K2 | C4 (directive) | Is vMF broadening doing anything that GGX α doesn't? |
| K3 | C5 (broad) | Is the diffuse lobe carrying real signal? |
| K4 | C3 (SPM) — KA only | Can pure Cook-Torrance fit alone? |
| K5 | C2 (KA) — SPM only | Can pure SPM fit alone? |
| K6 | C7 (coherence blend) — fix η=1 | Is smooth blending necessary, or is hard-pick fine? |
| K7 | C1 (Jones) — replace with scalar `\|r\|²` | Does polarization-aware Fresnel beat scalar reflection? |
| K8 | C8 (slab thickness) — single-layer Fresnel, drop param 5 | Does multi-layer interference contribute? |

8 runs × ~6 min = **~48 min**.

### Decision rule

For each Kk run:

| cart_corr drop vs B3 | Verdict |
|---|---|
| < 0.005 | **Drop the component.** Delete the code path. Save the wall-clock time. |
| 0.005–0.02 | Borderline. Look at what it costs in ms/iter; drop if cost-to-gain is bad. |
| > 0.02 | **Keep.** Real signal. |

### Wall-clock dimension

Pair every Kk verdict with **per-component wall-clock cost** measured by the existing per-stage timer in `render_factorized`. A component that contributes +0.005 cart_corr but costs 2 ms/iter (= 1 s over 500 iters per scene) is probably worth dropping; one that contributes +0.005 at 0.1 ms is not. The decision metric is `Δ cart_corr per ms`, not raw `Δ cart_corr`.

### Failure-mode visualization (free)

For each Kk run, dump the rendered RA image on one canonical scene (e.g. seq_0_frame_135) and diff it against GT. A model with cart_corr 0.91 might be missing strong specular peaks (catastrophic) or just blurring fine structure (acceptable). The scalar hides this and it's free to dump.

### Output of Phase 1.5

A **reduced BSDF** with dead components removed. Phase 2 then runs on this reduced model, not the full 6-param model. If e.g. K4 (KA-only) wins and SPM is dropped, then `tau_base` becomes mechanically inert and Phase 2 doesn't need to ablate it.

---

## Phase 2 — Per-parameter ablation of the (reduced) model

Only runs if Phases 1 and 1.5 leave a non-trivial parameter set. Runs against the **reduced BSDF** from Phase 1.5, not the full 6-param model.

### Mechanism

Single CLI flag `--freeze_mat 0,2,5`. Implementation: register a backward hook on `raw_materials` that zeros the listed gradient columns. Adam state for frozen columns stays zero, init values preserved exactly. ~30 lines, no architectural change.

### Run battery

| Group | Runs | What |
|---|---|---|
| **LOO** (leave-one-out) | 6 | Each freezes exactly one of {0,1,2,3,4,5}. Big cart_corr drop ⇒ that param matters at the margin given the others. Tiny drop ⇒ either irrelevant *or* redundant — disambiguated by TOO. |
| **TOO** (train-only-one) | 6 | Each freezes the other five. High score ⇒ this param alone explains most of the gap. Low score ⇒ can't fit alone. |
| **Group freezes** | 2 | Freeze Fresnel triple `{0,1,5}` → can roughness alone fit? Freeze roughness pair `{2,3}` → can Fresnel alone fit? (Mixing = LOO #4.) |
| **Cardinality elbow** | ~4 | Greedy build-up using TOO ranking: best-1 → best-2 → ... → all-6. Plot cart_corr vs DOF count, pick the elbow. |

Phase 2 cost: ~108 min (18 runs × 6 min).

### LOO × TOO verdict table

| LOO drop | TOO score | Verdict |
|---|---|---|
| Large | Large | **Essential** — irreplaceable degree of freedom |
| Large | Small | **Necessary co-factor** — useless alone but needed in combination |
| Small | Large | **Redundant** — others compensate, but can also stand in for them |
| Small | Small | **Inert** — drop it |

---

## Phase 3 — Identifiability diagnostics (every run, automatic)

LOO/TOO can't see joint redundancy: two params that always co-vary look individually important under LOO and individually weak under TOO, even though together they're 1 DOF, not 2. The diagnostics below catch this and cost almost nothing.

Every run in Phases 1 and 2 dumps these to a `.npz` alongside the cart_corr. No extra training runs needed.

### D1. Param drift (free)

After training, compute per column:

```
drift_k = || raw_materials_final[:, k] - raw_materials_init[:, k] ||_2 / sqrt(M)
```

A column with `drift_k ≈ 0` is not being learned at all, regardless of what LOO claims. Free — already have init and final tensors.

### D2. Fisher diagonal (~30 s/scene total)

After training, do one extra forward + backward and record:

```
fisher_k = sum_m ( ∂loss / ∂raw_materials[m, k] )^2
```

This is the diagonal of the Fisher information matrix. Tells you how sensitive the loss is to each parameter at the converged point. Low Fisher ⇒ loss landscape is flat in that direction ⇒ parameter is unidentified by the data. One backward pass per scene, ~30 s for all 7 scenes per run.

### D3. Pairwise trajectory correlation (cheap)

Log `raw_materials.detach().cpu()` every 50 iters during training. After training, for each point compute the Pearson correlation between every pair of param trajectories `(k1, k2)`, then average across the 50K points:

```
corr[k1, k2] = mean_m ( pearson( raw_materials[:, m, k1], raw_materials[:, m, k2] ) )
```

Pairs with `|corr| > 0.9` are degenerate. Logging cost: 11 checkpoints × 50K × 6 floats × 4 B = ~13 MB per scene. Trivial.

### D4. Converged-value histograms (free)

For each of the 6 columns of `raw_materials_final`, plot the distribution across the 50K points. Red flags:

- **Bimodal at sigmoid bounds** → parameter saturated, BSDF clipped, effectively binary not continuous.
- **Single sharp peak at init value** → parameter never moved (consistent with low D1 and low D2).
- **Wide unimodal distribution** → parameter is genuinely free and being used.

Free — derived from the same dump as D1.

---

## How the diagnostics interact with LOO/TOO

The verdict table (Phase 2) is the headline, but a cell can be misread without the diagnostics. Cross-checks:

| Phase 2 says | If diagnostics also show... | Real verdict |
|---|---|---|
| Essential (large LOO, large TOO) | High drift + high Fisher + wide histogram | **True essential** |
| Essential | Low drift + low Fisher | Suspicious — param doesn't move but freezing it tanks the loss? Indicates init value is load-bearing, not the *learning* of it. Try Phase 4. |
| Inert (small LOO, small TOO) | Low drift + low Fisher + peaked-at-init histogram | **True inert** — drop it. |
| Redundant (small LOO, large TOO) | High `corr[k, k']` with another param | Confirms degeneracy with that specific other param. The two are 1 DOF together. |
| Redundant | No high pair correlations | Multiple compensating params; harder to drop a specific one. |

---

## Phase 4 (optional) — Init sensitivity

Only if Phase 3 flags an "essential but doesn't move" param (high LOO drop, low D1+D2). Run B3 with materials init at *glass* `[6.27, 0.062, 1e-4, 5e-3, 0.5, 0.005]` instead of concrete. If cart_corr drops sharply, the init is load-bearing and we should think about init strategy, not just the learning rule.

One run, ~6 min. Do not run by default.

---

## Implementation steps

1. **Commit 1 — Diagnostics scaffolding** (no behavior change):
   - Add per-run `.npz` dump containing `raw_materials_init`, `raw_materials_final`, the iter-50 trajectory checkpoints, and the post-training Fisher diagonal.
   - Add a small `analyze_material_run.py` that takes a `.npz` and prints D1/D2/D3/D4 summary.
   - Verify with a single Tier A baseline run that the dump format works.

2. **Commit 2 — Freeze flag**:
   - `--freeze_mat 0,2,5` parses to a list of column indices.
   - Backward hook on `raw_materials.grad` zeros listed columns.
   - Sanity test: `--freeze_mat 0,1,2,3,4,5` should produce *exactly* the B0 result (all params at init forever).

3. **Commit 3 — Baseline variants**:
   - `--mat_mode global` swaps `(M, 6)` for a shared `(6,)` parameter, broadcast to all points before reparam. ~10 lines.
   - `--mat_mode scalar` swaps the BSDF call for `f_cos = sigmoid(reflectivity) * cos_i`, where `reflectivity` is a learnable `(M,)`. ~50 lines (a separate code path inside `render_factorized`).
   - `--mat_mode fixed` is just `--freeze_mat 0,1,2,3,4,5`, no new code.

4. **Commit 4 — Component disable flags**:
   - Add `--disable_component <name>` flag accepting any subset of `{cbs, directive, broad, spm, ka, blend, jones, slab}`.
   - Each flag gates the corresponding term in `render_factorized` Step 4 with a top-of-function `if` check that zeros the contribution before the blend (or in the `slab` / `jones` cases, replaces the call with a simpler form).
   - Per-stage timer dumps per-component ms/iter so the wall-clock dimension is captured.
   - Save the rendered RA image on seq_0_frame_135 at the end of every run for failure-mode visualization.

5. **Commit 5 — Driver script**:
   - `run_material_ablation.py` runs Phase 1 (4 runs) → decision rule → Phase 1.5 (8 runs) → trim model → Phase 2 (≤18 runs) → optional Phase 4.
   - Dumps everything to `mm25DGS_v4/output/material_ablation/<run_name>/`.
   - Aggregates results into `material_ablation_results.csv` with columns: phase, run_name, cart_corr_mean, per-scene cart_corr, ms/iter, drift, fisher.

6. **Run Phase 1** (~24 min). Read decision rule. Stop if a baseline already wins.

7. **Run Phase 1.5 if warranted** (~48 min). Read drop-component decisions. Produce reduced BSDF.

8. **Run Phase 2 if warranted** (~108 min, possibly less on the reduced model). Read verdict table cross-checked against diagnostics.

9. **Run Phase 4 only if a Phase 2 cell needs it** (~6 min).

Total compute upper bound: ~186 min if every phase runs. Minimum: ~24 min (Phase 1 alone) if a baseline already explains the data.

---

## Results reporting (mandatory)

Every run in every phase must produce a row in a single results file at [md/v4_material_ablation_results.md](v4_material_ablation_results.md). The driver script writes this file incrementally as runs complete, and the file is the single source of truth for the ablation outcomes.

### Required per-run columns

| Column | Source |
|---|---|
| `phase` | one of `1`, `1.5`, `2`, `4` |
| `run_name` | e.g. `B0_fixed_concrete`, `K1_no_cbs`, `LOO_freeze_eps_real`, `TOO_only_sigma_h` |
| `description` | one-line plain-English description of what this run tests |
| `mean_cart_corr` | mean across all 7 scenes |
| `per_scene_cart_corr` | dict-style row of all 7 scene scores |
| `Δ_vs_baseline` | `mean_cart_corr − 0.9351` (the Tier A reference) |
| `ms_per_iter` | mean per-iter wall-clock |
| `s_per_scene` | total wall-clock for the scene |
| `drift_per_param` | D1: per-column L2 drift, only for runs that train materials |
| `fisher_per_param` | D2: per-column Fisher diagonal, only for runs that train materials |
| `verdict` | filled in after the run by the decision rule for that phase |

### Section structure of the results .md

The results file is initialized with one section per phase, each containing an empty results table. As runs finish, rows are appended to the appropriate table. After all runs in a phase complete, a short prose summary block is added below that phase's table noting which decision-rule branch was taken and what Phase comes next (or terminates the investigation).

```
# v4 material ablation results

## Phase 1 — Baselines
| run_name | mean | Δ | ms/iter | verdict |
|---|---|---|---|---|
| B0_fixed_concrete | ... | ... | ... | ... |
...

**Phase 1 summary**: <prose, written after all 4 runs complete>

## Phase 1.5 — Component ablation
...

## Phase 2 — Parameter ablation
...

## Final recommendation
<written at the end>
```

### Update cadence

- **After each individual run**: append the row to the table.
- **After each phase completes**: add the summary prose block and explicitly state which decision branch was taken.
- **After the entire plan completes**: write a `Final recommendation` section at the bottom of the results file naming the recommended material model (component set + parameter set + per-point or global).

This is non-optional. No run is considered complete until its row exists in the results .md.

---

## What this plan deliberately does *not* do

- **Cross-sensor transfer evaluation**. Out of scope per user direction.
- **Multi-seed variance**. Single seed per run. Run-to-run noise on the 7-scene mean is ~±0.005; we apply that as the decision threshold rather than burning 3× compute on error bars.
- **Per-scene drilldowns**. The 7-scene mean is the headline; per-scene values are in the CSV for follow-up if a cell is suspicious.
- **Frozen-at-non-init sweeps** (other than the single Phase 4 glass init). Combinatorial explosion; not worth it before the Phase 1+2 results are in.
- **Spatial smoothness regularization** as an ablation. Different question (is per-point freedom *useful*) than what this plan asks (is per-point freedom *used*). Could be added later if D4 histograms look noisy.
- **Alternative model classes** (neural BSDF, tabulated BRDF lookup). Bigger lifts (~1 day each) and belong in a separate investigation. Worth naming as the destination if Phase 1 + 1.5 both say "current physics model is doing very little":
  - **Neural BSDF**: 6-input MLP `(wi, wo) → reflectance` per point. Maximum flexibility, zero physics.
  - **Tabulated BRDF lookup**: per-point index into a small set of measured/canonical materials. Discrete instead of continuous.
