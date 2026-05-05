# mmIR Baseline Comparison — New Chat Session Prompt

Copy-paste the block below into a new Claude Code session at `/home/adnan/Desktop/mm3DGS`.

---

## Prompt

I'm preparing a NeurIPS oral submission for **3DPS: Differentiable 3D Point Splatting for mmWave Radar**. The paper currently compares 3DPS to three optical-NVS baselines (RadarSplat, Radar Fields, DART) but only **projects** mmIR's per-scene training cost (~160 min) without running it end-to-end on our 6-scene ColoRadar benchmark. A senior NeurIPS reviewer will attack this asymmetry in rebuttal phase. I need to add mmIR as a fully measured baseline.

**Critical context: mmIR is the project's own renderer.** The directory `/home/adnan/Desktop/mm3DGS/mmir/` is the existing Monte Carlo radar ray tracer (`mmir/renderer/`, `mmir/evaluation/`, etc.). 3DPS is in `mm25DGS_v5_v4/`. The mmIR baseline runner just imports from the existing `mmir/` package — no clone, no port. Place runner files at `baselines/mmir/runner/run_mmir_scene.py` (this directory does NOT collide with the project's `mmir/` package because Python imports them by absolute path).

Read these in order before doing anything else:

1. `/home/adnan/Desktop/mm3DGS/.claude/CLAUDE.md` — codebase overview, conda env, DrJit pitfalls (never call `mi.set_variant()` more than once per process).
2. `/home/adnan/Desktop/mm3DGS/md/mmir_baseline_plan.md` — detailed plan: 3 phases (pre-flight, run, paper updates), risks, deliverables.
3. `/home/adnan/Desktop/mm3DGS/baselines/dart/runner/run_dart_scene.py` and `baselines/dart/runner/finalize_metrics.py` — the convention every baseline runner follows. Mirror this for mmIR.
4. `/home/adnan/Desktop/mm3DGS/baselines/radarsplat/runner/finalize_metrics.py` — second example of the same convention.
5. `/home/adnan/Desktop/mm3DGS/figures/generate_tables.py` — how baselines are loaded into the per-scene RA tables (`load_baseline_metrics`, `load_baseline_train_metrics`).
6. `/home/adnan/Desktop/mm3DGS/figures/generate_crp_adc_paper_table.py` — Table 2 generator. Currently the optical-NVS baselines have `--` in CRP/ADC columns; mmIR will populate them.
7. `/home/adnan/Desktop/mm3DGS/figures/generate_fig_training_ra.py` — Figure 3 (qualitative comparison) generator. Must accept a new mmIR row.
8. The project's existing mmIR training entry point. CLAUDE.md mentions `python train.py --config output/training/seq_0_frame_135/config.json` — discover the actual entry point in Phase 0.

---

## Compute environment

- 2× NVIDIA RTX 4090 (24 GB each).
- Conda env: `/home/adnan/.conda/envs/mmir/bin/python`.
- One mmIR run per GPU at a time. Parallelize via `CUDA_VISIBLE_DEVICES=0` and `=1`.
- Project root: `/home/adnan/Desktop/mm3DGS/`.

---

## Mandatory Phase 0 — pre-flight verification (DO THIS FIRST)

**Do not launch the multi-scene multi-viewpoint sweep until Phase 0 passes.** Cost of getting the wiring wrong: ~16 GPU-hours of redo time.

### 0.1 — Inventory mmIR's training entry point

- Read `/home/adnan/Desktop/mm3DGS/mmir/renderer/integrator.py` and any top-level `train.py`.
- Find the existing mmIR training entry point. The CLAUDE.md hints at `python train.py --config output/training/seq_0_frame_135/config.json`.
- Document: training script path, expected config schema, output directory layout, runtime per scene at default settings.

### 0.2 — Decide between Option A (per-viewpoint training) vs Option B (joint multi-viewpoint)

The paper describes mmIR as a single-viewpoint trainer (~20 min per fit). For an apples-to-apples NVS comparison we either:

- **Option A** (default): train mmIR independently on each of the 8 training viewpoints, aggregate the 8 outputs into a per-scene representation, evaluate at the held-out test viewpoint. Compute = 8 × 20 min × 6 scenes = **~16 GPU-hours**.
- **Option B**: train mmIR jointly across all 8 viewpoints (if mmIR's training loop already supports multi-pose joint optimization, use it; otherwise non-trivial code changes required).

Read the mmIR codebase. If joint multi-viewpoint training is already supported, prefer Option B. Otherwise default to Option A. Document the choice.

### 0.3 — Inventory the existing baseline output convention

The convention (from DART, RadarSplat, Radar Fields):

```
baselines/<method>/results/<scene>/
├── metrics.json                # held-out test: ra_corr, cart_mse, cart_rmse, cart_psnr, cart_ssim
├── rendered_ra_cart.npy        # 399x399, used by Figure 3 generator
├── rendered_ra_polar.npy       # cropped polar (8, 95) — optional
└── train_frames/
    └── frame_<F>/
        ├── metrics.json
        └── rendered_ra_cart.npy
```

The mmIR baseline runner output **must** match this exactly. Verify by reading `baselines/dart/runner/finalize_metrics.py` and `baselines/common/eval.py`.

### 0.4 — Verify the CRP/ADC eval pipeline can be re-pointed at a baseline

Find the script that produces `output/crp_adc_eval/results.json` for 3DPS. Confirm:

- It can be re-pointed at an arbitrary input directory (i.e., at `baselines/mmir/results/`).
- It writes to a configurable output path (so we can land mmIR results at `output/crp_adc_eval/baselines/mmir/results.json`).

If the existing pipeline is hard-coded to 3DPS-only, generalize it before launching the mmIR sweep. **This is the highest-impact wiring step** — the entire mmIR comparison is pointless if we can't compute its CRP/ADC numbers.

### 0.5 — Dry-run mmIR end-to-end on one (scene, viewpoint)

Pick `seq_0_frame_135` and viewpoint `frame_134` (or any one of the 8 training views).

1. Run mmIR training on this single (scene, viewpoint) at full default settings.
2. Run a stub of `baselines/mmir/runner/finalize_metrics.py` to produce `metrics.json` + `rendered_ra_cart.npy` at `baselines/mmir/results/seq_0_frame_135/`.
3. Run the CRP/ADC eval on the mmIR output → `output/crp_adc_eval/baselines/mmir/seq_0_frame_135/results.json`.
4. Run `figures/generate_crp_adc_paper_table.py` to confirm Table 2 picks up the new mmIR row syntactically (numbers can be partial / NaN at this point — we just want the LaTeX to compile and include the mmIR row).
5. Run `figures/generate_fig_training_ra.py` to confirm the figure generator accepts a new method row.

If any step fails or produces output in the wrong format, **fix it before kicking off the full 48-run sweep**.

### 0.6 — Time estimate

Time the dry-run training step. If a single mmIR run substantially exceeds 20 min, recompute the budget:

- Best case: 6 × 8 × 20 min = ~16 GPU-hours sequential = ~8 hours wall-clock on 2 GPUs.
- Realistic: 16-24 GPU-hours.
- Worst case: 24+ GPU-hours = ~12 hours wall-clock.

**Stop here and ask me to confirm before proceeding to Phase 1.**

---

## Phase 1 — Run mmIR on all 6 scenes × 8 viewpoints (~16-24 GPU-hours)

### 1.1 — Create the runner

`baselines/mmir/runner/run_mmir_scene.py`:

- Mirror the structure of `baselines/dart/runner/run_dart_scene.py`.
- Iterate over the 8 training viewpoints for a given scene.
- Invoke the project's existing mmIR training entry point per-viewpoint.
- Save outputs at `baselines/mmir/results/<scene>/per_viewpoint/frame_<F>/`.

### 1.2 — Sweep

For each of 6 scenes × 8 viewpoints = 48 mmIR training runs:

- Parallelize across the 2 GPUs via `CUDA_VISIBLE_DEVICES`.
- Keep GPU memory monitored — DrJit can OOM in surprising ways.
- Save logs to `baselines/mmir/logs/<scene>/frame_<F>.log`.

### 1.3 — Aggregate per-scene results

`baselines/mmir/runner/finalize_metrics.py`:

- For each scene, combine the 8 per-viewpoint outputs into a single aggregate scene representation. The aggregation rule depends on the Option chosen in Phase 0.2 (e.g., for Option A: pick the run with best train-frame RA Corr, or average material params, or concatenate point clouds — verify what mmIR's own paper does).
- Render the aggregated representation at the held-out test viewpoint → `baselines/mmir/results/<scene>/rendered_ra_cart.npy`.
- Save aggregate test metrics → `baselines/mmir/results/<scene>/metrics.json`.
- Save per-train-frame metrics → `baselines/mmir/results/<scene>/train_frames/frame_<F>/metrics.json`.

### 1.4 — Verify against existing baselines

After Phase 1, run `figures/generate_tables.py` on the new mmIR outputs alongside the existing baselines. Confirm:

- mmIR appears as a new column/row in the per-scene RA tables (supplement Tables 3, 4).
- mmIR's metrics.json has all 5 required fields (`ra_corr`, `cart_mse`, `cart_rmse`, `cart_psnr`, `cart_ssim`).
- `rendered_ra_cart.npy` is 399×399 (matches `baselines/dart/results/<scene>/rendered_ra_cart.npy`).

If anything is off, fix it before Phase 2.

---

## Phase 2 — CRP/ADC evaluation (~1 hour)

For each scene's mmIR output:

1. Run the CRP/ADC eval pipeline (re-pointed per Phase 0.4) → `output/crp_adc_eval/baselines/mmir/<scene>/results.json`.
2. Aggregate across 6 scenes → `output/crp_adc_eval/baselines/mmir/results.json`.

Format must mirror `output/crp_adc_eval/results.json` (the existing 3DPS aggregate) so `generate_crp_adc_paper_table.py` can read it.

---

## Phase 3 — Update paper artifacts

### 3.1 — Table generators

1. **`figures/generate_tables.py`**: register `'mmir'` in `METHODS`. Verify `load_baseline_metrics('mmir', scene, baselines_dir)` works without changes (mmIR follows the standard convention).
2. **`figures/generate_crp_adc_paper_table.py`**:
   - Add `'mmir'` to `METHOD_ORDER`.
   - Update `METHOD_LABEL` with `"mmir": "mmIR~\\cite{mmIR2026}"`.
   - The mmIR row is fully populated (CRP and ADC columns get real numbers, not `--`). Update `_row()` formatter to handle this.
3. **`figures/generate_crp_adc_table.py`** (supplement per-scene): add mmIR.
4. Regenerate all four tables.

### 3.2 — Qualitative figure

`figures/generate_fig_training_ra.py`: add an mmIR row between `Ours` and `RadarSplat`. Regenerate `figs/training_ra_comparison.pdf`.

### 3.3 — Update prose

Replace every "projected" mmIR claim with measured numbers:

- `sec/0_abstract.tex`: speedup claim.
- `sec/1_intro.tex`: ¶3 (mmIR cost), ¶5 (results summary).
- `sec/2_relatedworks.tex`: Table 1 Time column for mmIR row.
- `sec/3_methods.tex`: Sec. forward model.
- `sec/4_experiments.tex`: Sec. baselines (add mmIR description), Sec. implementation, Sec. results discussion.
- `sec/5_conclusion.tex`: any speedup claim.
- `sec/A_supplement.tex`: Sec. runtime (Table 7), Sec. limitations.

### 3.4 — Rebuild PDF

`pdflatex` × 2 + `bibtex`. Verify clean compile, no undefined references, no LaTeX warnings.

---

## Constraints / guardrails

- **Plan first, ask before destructive runs.** Write a top-level execution plan listing all 48 mmIR runs, estimate compute, and confirm with me before kicking off the full sweep.
- **Do not modify** `mm25DGS_v5_v4/output_frame_nvs/` (canonical 3DPS results) or `output/crp_adc_eval/results.json` (canonical 3DPS aggregate). Place mmIR outputs in dedicated directories: `baselines/mmir/results/`, `output/crp_adc_eval/baselines/mmir/`.
- **Do not edit the project's `mmir/` package directly** — the runner imports from it. If you find bugs in `mmir/` that block the runner, surface them and ask before patching.
- **Use the same evaluation harness** as the main paper. Do not introduce new metrics.
- **Match the existing baseline output convention exactly** (DART, RadarSplat, Radar Fields all share it). The figure and table generators rely on it.

---

## Deliverable

A fully populated mmIR row in Table 2 (with CRP and ADC numbers — unique to mmIR among baselines), plus mmIR rows in Supplement Tables 3-6, plus updated Figure 3 (with mmIR row), plus measured-not-projected prose throughout the paper, plus a clean `main.pdf` build.

The headline outcome a senior reviewer wants to see: **mmIR's measured CRP/ADC numbers (likely high, since mmIR is physics-faithful) at ~160 min/scene vs 3DPS's lower-but-substantial CRP/ADC numbers at ~3 min/scene** — making the speed-vs-fidelity tradeoff concrete.
