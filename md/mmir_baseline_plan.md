# mmIR Baseline Comparison Plan

## Goal

Add **mmIR** (the project's existing Monte Carlo radar renderer at `/home/adnan/Desktop/mm3DGS/mmir/`) as a direct comparison baseline on the same 6 ColoRadar scenes used in the main paper. Update Table 2 (main results) and Figure 3 (qualitative comparison) to include mmIR. Convert the current "projected ~160 min per scene" claim into measured numbers.

## Why this matters

The current paper has a known weakness any senior NeurIPS reviewer will flag in rebuttal:

> *"You claim to beat mmIR by 50× per scene, but you never run mmIR end-to-end on your benchmark. The 160-minute number is projected, not measured. Without mmIR's actual quality numbers on these 6 scenes, the speed-vs-quality tradeoff claim is not directly supported by the experiments."*

Closing this gap converts a soft claim into a hard one and removes one of the largest potential rebuttal-phase attacks. The intro and abstract frame the gap as **two prior families** (MC ray tracers + optical-NVS adaptations). Optical-NVS is well-represented (RadarSplat, Radar Fields, DART). MC ray tracers are not — that asymmetry is visible.

---

## Critical context: mmIR is the project's own renderer

The `mmir/` directory at the project root **is the mmIR Monte Carlo ray tracer**. It contains the renderer (`mmir/renderer/`), evaluation harness (`mmir/evaluation/`), data loaders, and BSDF code. 3DPS lives separately under `mm25DGS_v5_v4/`.

This means:
- **No external code to clone**. The runner just needs to wrap the existing `mmir/` package.
- **No port/adaptation work**. mmIR already supports ColoRadar scenes (the 3DPS pipeline shares data loaders with mmIR).
- **Naming convention**: place the runner at `baselines/mmir/runner/run_mmir_scene.py`. The `baselines/mmir/` directory will not collide with the project's `mmir/` package because Python imports them by absolute path (`baselines.mmir.runner...` vs `mmir.renderer...`).

### Decision: don't copy `mmir/` to `baselines/mmir_baseline/`

The existing `mmir/` package is the canonical implementation; the baseline runner should import from it directly to stay in sync. Pinning a specific version is unnecessary for an initial submission. If reproducibility becomes a concern later, document the git commit hash in the baseline README.

---

## Required outputs

### Main paper updates

1. **New row(s) in Table 2** (`crp_adc_results.tex`): mmIR train mean / mmIR test mean across all 12 columns where mmIR can produce output.
   - mmIR produces **complex output**, so it fills RA cart, CRP, and ADC columns (unlike RadarSplat / Radar Fields / DART which only fill RA cart).
   - Verify which complex metrics mmIR can produce on the held-out test view (at minimum: RA Corr/PSNR/SSIM/RMSE, CRP magnitude correlation, ADC envelope correlation; possibly also complex |ρ| and σ_φ if applicable).

2. **Updated Figure 3** (`fig:training_ra` in `figs/training_ra_comparison.pdf`): add an mmIR row.
   - Current rows: GT / Ours / RadarSplat / Radar Fields / DART.
   - New layout: GT / 3DPS (Ours) / mmIR / RadarSplat / Radar Fields / DART.

3. **Update prose claims** throughout the paper:
   - Intro ¶3: replace "mmIR requires ~20 minutes per single-viewpoint fit at 24k surface interactions" with **measured** per-scene cost.
   - Abstract: update "~50× faster than the SOTA MC ray tracer's projected ~160 min per-scene cost" with measured speedup ratio.
   - Methods Sec. forward model: update.
   - Related work Table 1: update mmIR row's `Time` column to measured (currently "160 min" projected).

### Supplement updates

4. **Per-scene mmIR results** in Supplement Tables 3, 4 (per-scene RA breakdowns) and Tables 5, 6 (per-scene CRP and ADC fidelity).
5. **Implementation notes** in Sec. runtime: how mmIR was configured, what 8-viewpoint training meant in practice, any modifications to defaults.

---

## Phase 0 — pre-flight verification (DO NOT SKIP, ~0.5 day)

**Critical: validate the wiring before launching the ~16 GPU-hour sweep.**

### 0.1 — Inventory mmIR's training entry point

1. Read `/home/adnan/Desktop/mm3DGS/mmir/renderer/integrator.py`, `mmir/evaluation/`, and any top-level `train.py` at the project root.
2. Find the existing mmIR training entry point. The CLAUDE.md mentions `python train.py --config output/training/seq_0_frame_135/config.json` — this is likely the mmIR trainer.
3. Document: training script path, expected config format, output directory structure, runtime per scene at default settings.

### 0.2 — Verify mmIR's NVS support

mmIR was designed for **single-viewpoint training-frame fits** (per the paper's framing: "~20 min per single-viewpoint fit at 24k surface interactions"). For an apples-to-apples NVS comparison we need **multi-viewpoint joint optimization**, OR a per-viewpoint approach that we evaluate at the held-out test view.

**Three options, ranked**:

- **Option A (default — most defensible)**: train mmIR independently on each of the 8 training viewpoints, then evaluate the trained per-viewpoint material/geometry params at the held-out test viewpoint. Per-scene compute = 8 × 20 min = 160 min. Aggregate the 8 per-viewpoint outputs into a single "mmIR scene" by averaging materials or picking the best.
- **Option B**: train mmIR jointly across all 8 viewpoints. Requires non-trivial code changes to mmIR's training loop. Higher risk; defer unless Option A shows mmIR can't generalize.
- **Option C**: train mmIR directly on the held-out test viewpoint (single-viewpoint fit). This **isn't NVS** and shouldn't be the headline number, but it's a useful upper bound on mmIR's quality for the same compute. Could appear as a supplementary "mmIR (test fit)" row to show the full Pareto.

**Decide between A and B in Phase 0.2.** Option A is the default; Option B only if mmIR already has joint multi-viewpoint training support (check the code).

### 0.3 — Inventory the existing baseline output convention

Read `/home/adnan/Desktop/mm3DGS/baselines/dart/runner/finalize_metrics.py` and `baselines/radarsplat/runner/finalize_metrics.py`. The convention is:

```
baselines/<method>/results/<scene>/
├── metrics.json                # held-out test: ra_corr, cart_mse, cart_rmse, cart_psnr, cart_ssim
├── rendered_ra_cart.npy        # held-out test cart RA (399x399), used by Figure 3 generator
├── rendered_ra_polar.npy       # cropped polar (8, 95) — optional; used internally by finalize
└── train_frames/
    └── frame_<F>/
        └── metrics.json        # per-train-frame (same fields as test metrics.json)
```

The mmIR baseline output **must** match this convention so `figures/generate_tables.py` and `figures/generate_fig_training_ra.py` can pick it up unchanged.

### 0.4 — Inventory the CRP/ADC eval pipeline

Read whatever script produces `output/crp_adc_eval/results.json` for 3DPS. mmIR will also produce CRP and ADC outputs (unlike the optical-NVS baselines), so we need:

- An mmIR-specific run of the CRP/ADC eval that points at the mmIR baseline output dir.
- The result lands at `output/crp_adc_eval/baselines/mmir/results.json` (or similar).
- `figures/generate_crp_adc_paper_table.py` is updated to read this and add the mmIR row to Table 2.

If the existing CRP/ADC eval pipeline is hard-coded to 3DPS-only, generalize it before launching mmIR runs. **This is the single highest-impact wiring step.**

### 0.5 — Dry-run mmIR on one scene, one viewpoint

1. Pick the fastest scene (e.g., `seq_0_frame_135`) and one training viewpoint.
2. Run mmIR's existing training entry point on this single (scene, viewpoint).
3. Run a stub of `baselines/mmir/runner/finalize_metrics.py` on the output — produces `metrics.json` and `rendered_ra_cart.npy` at `baselines/mmir/results/seq_0_frame_135/`.
4. Run the CRP/ADC eval on the same output → `output/crp_adc_eval/baselines/mmir/results/seq_0_frame_135/results.json`.
5. Run `figures/generate_crp_adc_paper_table.py` to confirm Table 2 picks up the new mmIR row syntactically (numbers can be partial / NaN at this point).

If any step fails or produces output in the wrong format, fix it before kicking off the full 6-scene × 8-viewpoint sweep.

### 0.6 — Time estimate for the full sweep

Time the dry-run training step from 0.5. If a single mmIR training run substantially exceeds 20 min, recompute the budget:

- Best case: 6 × 8 × 20 min = 16 GPU-hours total (~8 hours wall-clock on 2× RTX 4090).
- Realistic case (24k surface interactions per iter, 500 iters): 16-24 GPU-hours.
- Worst case (~30 min/run): up to 24 GPU-hours per parallel run, 12 hours wall-clock.

**Stop here and ask me to confirm before proceeding to Phase 1.**

---

## Phase 1 — Run mmIR on all 6 scenes × 8 viewpoints (~16-24 GPU-hours)

### 1.1 — Create the runner script

`/home/adnan/Desktop/mm3DGS/baselines/mmir/runner/run_mmir_scene.py`:

- Mirror the structure of `baselines/dart/runner/run_dart_scene.py`.
- Iterate over the 8 training viewpoints for a given scene.
- For each viewpoint, invoke the project's `mmir/` training entry point with the appropriate config.
- Save outputs at `baselines/mmir/results/<scene>/per_viewpoint/frame_<F>/`.

### 1.2 — Run the sweep

For each of 6 scenes × 8 viewpoints = 48 mmIR training runs:

- Parallelize across 2 GPUs via `CUDA_VISIBLE_DEVICES`.
- One mmIR run per GPU at a time (DrJit doesn't share GPUs cleanly).
- Total: ~16 GPU-hours sequential = ~8 hours wall-clock on 2× RTX 4090.

### 1.3 — Aggregate per-scene from per-viewpoint outputs

For each scene, combine the 8 per-viewpoint outputs into a single aggregate scene representation. This is the Option-A semantics chosen in Phase 0.2:

- For Option A: pick the best per-viewpoint material/geometry, OR average them, OR concatenate. The choice is method-design dependent — verify with the senior author or follow what mmIR's own paper does.
- Render the aggregated representation at the held-out test viewpoint → `baselines/mmir/results/<scene>/rendered_ra_cart.npy`.
- Save aggregate test metrics → `baselines/mmir/results/<scene>/metrics.json`.

### 1.4 — Per-train-frame metrics

For each scene, evaluate at each of the 8 training viewpoints → `baselines/mmir/results/<scene>/train_frames/frame_<F>/metrics.json`. Match the field schema of the other baselines.

---

## Phase 2 — CRP/ADC evaluation (~1 hour)

For each scene's mmIR output, run the CRP/ADC eval pipeline:

1. Output directory: `output/crp_adc_eval/baselines/mmir/<scene>/results.json`.
2. Aggregate across the 6 scenes → `output/crp_adc_eval/baselines/mmir/results.json` (matches the structure of the existing 3DPS aggregate at `output/crp_adc_eval/results.json`).

---

## Phase 3 — Update paper artifacts (~0.5 day)

### 3.1 — Update table generators

1. **`figures/generate_tables.py`**: register `'mmir'` in the `METHODS` tuple. Add the loader (`load_baseline_metrics(method='mmir', ...)` should work without changes since mmIR follows the standard baseline convention; verify on a single scene).
2. **`figures/generate_crp_adc_paper_table.py`**: add the mmIR row. Since mmIR has CRP and ADC numbers (unlike the optical-NVS baselines), the row is fully populated rather than `--`. Update the `_row()` formatter to handle a mmIR-with-CRP-ADC row.
3. **`figures/generate_crp_adc_table.py`** (supplement Tables 5, 6): add mmIR rows.
4. Regenerate all four tables.

### 3.2 — Update the qualitative figure

`figures/generate_fig_training_ra.py`: add a `mmir` row between `Ours` and `RadarSplat`. Run the figure generator. Save to `figs/training_ra_comparison.pdf`.

### 3.3 — Update prose

Update **measured** mmIR numbers throughout, replacing every "projected ~160 min" / "projected per-scene cost" claim:

- `sec/0_abstract.tex`: speedup claim (last sentence).
- `sec/1_intro.tex`: ¶3 (mmIR cost), ¶5 (results summary).
- `sec/2_relatedworks.tex`: Table 1 Time column for mmIR row.
- `sec/3_methods.tex`: Sec. forward model.
- `sec/4_experiments.tex`: Sec. baselines (add mmIR), Sec. implementation, Sec. results discussion.
- `sec/5_conclusion.tex`: any speedup claim.
- `sec/A_supplement.tex`: Sec. runtime (Table 7), Sec. limitations.

### 3.4 — Rebuild PDF

`pdflatex` × 2 + `bibtex`. Verify clean compile, no undefined references, no LaTeX warnings.

---

## Risk register

- **mmIR achieves better quality than 3DPS** on physics-faithfulness metrics (e.g., higher CRP correlation). This is **fine** for the paper — we claim fast NVS with physics, not absolute SOTA on physics. The trade-off space we describe (fast vs. physics-faithful) becomes more honest with measured numbers. Update the prose accordingly.

- **mmIR achieves much worse quality than expected**. Could mean per-viewpoint training doesn't generalize to held-out poses, or that mmIR needs more iterations. Either is informative. Try Option B (joint multi-viewpoint) if Option A produces near-zero held-out correlation.

- **mmIR per-viewpoint training time exceeds 20 min**. Update measured numbers; the speedup ratio shifts.

- **Out-of-memory with N=24k surface interactions × 8 viewpoints**. Each run is independent, so memory isn't shared — should be fine on 24 GB cards. Stress-test in Phase 0.5.

- **Naming collision** (`mmir/` package vs `baselines/mmir/`). Pure Python imports use absolute paths so there's no actual collision; but humans get confused. Document clearly in `baselines/mmir/README.md`.

---

## Deliverables

1. `baselines/mmir/runner/run_mmir_scene.py` — runner script.
2. `baselines/mmir/runner/finalize_metrics.py` — metrics aggregator (mirrors DART/RadarSplat/RadarFields convention).
3. `baselines/mmir/results/<scene>/{metrics.json, rendered_ra_cart.npy, train_frames/}` for all 6 scenes.
4. `output/crp_adc_eval/baselines/mmir/results.json` — aggregate CRP/ADC across 6 scenes.
5. Updated Table 2 (`crp_adc_results.tex`) with mmIR row.
6. Updated Supplement Tables 3-6 with mmIR per-scene numbers.
7. Updated Figure 3 (`figs/training_ra_comparison.pdf`) with mmIR row.
8. Updated prose in `sec/{0_abstract,1_intro,2_relatedworks,3_methods,4_experiments,5_conclusion,A_supplement}.tex` reflecting **measured** (not projected) mmIR numbers.
9. Clean `main.pdf` build.

---

## Time estimate

- Phase 0 (pre-flight): 0.5 day
- Phase 1 (mmIR training sweep): 0.5-1 day GPU time (overnight on 2× RTX 4090)
- Phase 2 (CRP/ADC eval): 1 hour
- Phase 3 (paper updates): 0.5 day

**Total: ~2-3 working days end-to-end** (much faster than the original 3-4 day estimate, since the codebase is already in place).
