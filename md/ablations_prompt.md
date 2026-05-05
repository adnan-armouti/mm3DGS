# Ablation Experiments — New Chat Session Prompt

Copy-paste the block below into a new Claude Code session at `/home/adnan/Desktop/mm3DGS`.

---

## Prompt

I'm preparing a NeurIPS oral submission for **3DPS: Differentiable 3D Point Splatting for mmWave Radar**. The paper's supplement currently has a placeholder ablation table (Table 8, `tab:ablations`, in `sec/A_supplement.tex` `\section{Ablations}`). I need to populate it with real experimental results spanning **Tier 1 and Tier 2** of `md/ablations_plan.md`.

Read these in order before doing anything else:

1. `/home/adnan/Desktop/mm3DGS/.claude/CLAUDE.md` — codebase overview, conda env (`/home/adnan/.conda/envs/mmir/bin/python`), DrJit pitfalls (never call `mi.set_variant()` more than once per process — segfault).
2. `/home/adnan/Desktop/mm3DGS/md/ablations_plan.md` — the detailed ablation plan: 9 axes split into Tier 1 (5) and Tier 2 (4), output directory layout, risk register, deliverables.
3. `/home/adnan/Desktop/mm3DGS/latex/NeurIPS_2026_unpacked/Physically_Grounded_Novel_View_Synthesis_for_Millimeter_Wave_Radar_via_Point_Based_Hemisphere_Rendering/sec/A_supplement.tex` — current `\section{Ablations}` with the placeholder Table 8 layout.
4. `/home/adnan/Desktop/mm3DGS/figures/generate_tables.py` — convention for results table generation. Has `SCENES`, `METHODS`, `load_ours_test_metrics`, `load_ours_train_metrics`. The 6 benchmark scenes are the constants `SCENES`. Reuse this loader machinery.
5. `/home/adnan/Desktop/mm3DGS/figures/generate_crp_adc_paper_table.py` — the Table 2 generator that reads `output/crp_adc_eval/results.json` for 3DPS CRP/ADC numbers. The same eval pipeline must be re-pointed at each ablation output for the ablation table.
6. **Existing 3DPS results layout**: peek at `/home/adnan/Desktop/mm3DGS/mm25DGS_v5_v4/output_frame_nvs/seq_0_frame_135_*_pass2_N20000*/`. Note the file inventory (`results.json`, `metrics_train.json`, `rendered_ra_*.npy`).
7. **3DPS training entry point**: discover this in Phase 0. Likely under `mm25DGS_v5_v4/`. Document the flag set on the first invocation.

---

## Compute environment

- **2× NVIDIA RTX 4090 (24 GB each)**.
- Conda env: `/home/adnan/.conda/envs/mmir/bin/python`.
- **One ablation config per GPU at a time** — parallelize via `CUDA_VISIBLE_DEVICES=0` and `=1`. Don't run more than one DrJit process per GPU; the CUDA backend doesn't share well.
- Keep all ablation outputs under `mm25DGS_v5_v4/output_ablations/<tier>/<ablation_name>/<config>/` (do NOT touch `output_frame_nvs/`, which holds the canonical 3DPS results used by the main paper).
- CRP/ADC eval outputs: `output/crp_adc_eval_ablations/<tier>/<ablation_name>/<config>/results.json`.

---

## Mandatory Phase 0 — pre-flight verification (DO THIS FIRST)

**Do not launch any Tier 1 or Tier 2 runs until Phase 0 passes.** A failed Phase 0 means the eventual paper-update step won't work, and the GPU-hours are wasted.

### 0.1 — Inventory the 3DPS pipeline

Find and document:

- The 3DPS training entry point and its full set of CLI/config flags.
- The list of files written into a 3DPS run directory (e.g., `mm25DGS_v5_v4/output_frame_nvs/seq_0_frame_135_*_pass2_N20000*/`).
- The script that produces `output/crp_adc_eval/results.json` for 3DPS, and how to re-point it at a different output directory.

### 0.2 — Verify each ablation axis is configurable

For each of the 9 Tier 1 + Tier 2 axes (see plan §"Required ablations"), find where the design choice lives in the code and confirm the override mechanism:

- Point count `N`
- Number of training views (subset of the 8 bracketing frames)
- Adaptive density control on/off
- LiDAR init stages (cull, occlusion, cosine resample, FPS) — each individually disabled
- Quasi-static MIMO factorization on/off
- PSF kernel size `L`
- Carrier-phase detach for position gradients on/off
- L2 position anchor `λ_pos`
- Closed-form ITU-R P.2040 BSDF vs per-point learned MLP

**If any axis is not currently flag-controlled, add the override before launching Tier 1.** Do not hard-code config edits in 9 separate branches; keep everything in a single training entry point with overrideable flags.

### 0.3 — Dry-run one config end-to-end

Pick the fastest Tier-1 config (e.g., `N=2k`, **single scene `seq_0_frame_135` only**, full 500 iterations). Run the entire pipeline:

1. Train.
2. Compute test |RA| metrics → `results.json`.
3. Compute train |RA| metrics → `metrics_train.json`.
4. Run CRP/ADC eval → `output/crp_adc_eval_ablations/.../results.json`.
5. Run a stub of `figures/generate_ablation_table.py` over the single output to confirm the LaTeX table parses.

If any step fails, fix it before launching the multi-scene runs. **The cost of fixing a wiring bug after running 6 scenes × 9 configs = 54 scenes is hours of redo time.**

### 0.4 — Verify the ablation-table generator works

Write `figures/generate_ablation_table.py` (or extend an existing script). It should:

1. Walk `mm25DGS_v5_v4/output_ablations/<tier>/<ablation>/<config>/` directories.
2. For each config: aggregate per-scene `results.json` → mean test |RA| Corr/PSNR/SSIM/RMSE, plus mean wall-clock and peak GPU memory.
3. For each config: read CRP/ADC results from `output/crp_adc_eval_ablations/<tier>/<ablation>/<config>/results.json` and aggregate.
4. Emit `latex/.../tables/ablations.tex` populating Supplement Table 8.
5. Run on the Phase 0.3 dry-run output to confirm the LaTeX is syntactically valid.

### 0.5 — Confirm GPU-hour estimate

Time the dry-run from 0.3. If a single Tier-1 config × 1 scene takes substantially more than `default_runtime / 6 * scaling_factor`, recompute the total budget and warn me before kicking off.

**Stop here and ask me to confirm before proceeding to Phase 1.**

---

## Phase 1 — Tier 1 runs (~10 GPU-hours)

After Phase 0 passes, kick off Tier 1 in this order, parallelized across the 2 GPUs:

1. **Point count N**: `{2k, 5k, 10k, 50k}` × 6 scenes (skip 20k — already in `output_frame_nvs/`, can symlink).
2. **Number of training views**: `{2, 4, 6}` × 6 scenes (skip 8 — default).
3. **Adaptive density off**: 1 config × 6 scenes.
4. **LiDAR init stages disabled**: 4 configs × 6 scenes.
5. **MIMO factorization off**: 1 config × 6 scenes (slowest — may OOM at N=20k; reduce to N=5k for this config if so, and document).

After each config completes 6 scenes, run the aggregator to update the partial table — that way if anything crashes mid-Tier-1, we don't lose visibility.

---

## Phase 2 — Tier 2 runs (~3 GPU-hours)

1. PSF kernel L: `{5, 9, 21, 25}` × 6 scenes (odd-only, skip 15 — default). Hardcoded `SPREAD=15` at `mm25DGS_v5_v4/rasterizer_factorized.py:495` must be exposed as a flag during Phase 0; the CUDA kernel reads spread at runtime from `psf_real.size(0)`, so wiring the new flag into `HannPSFTable(K, spread=L, ...)` is sufficient. Even L values produce L+1 bins due to `arange(-(L//2), L//2+1)` asymmetry, so stay on odd values.
2. Carrier-phase detach off: 1 config × 6 scenes.
3. λ_pos sweep: `{0, 1, 1000}` × 6 scenes.
4. Learned-BSDF MLP: 1 config × 6 scenes.

---

## Phase 3 — aggregate + write up

1. Run `figures/generate_ablation_table.py` over the full output tree to populate `latex/.../tables/ablations.tex` for Supplement Table 8.
2. Replace placeholder text in `sec/A_supplement.tex` `\section{Ablations}` with the real numbers.
3. Write a 1-paragraph discussion per Tier-1 ablation summarizing what the data validates.
4. Rebuild `main.pdf`. Confirm clean compile, no LaTeX warnings, no undefined references.
5. Note any surprising results in `app:limitations`.

---

## Constraints / guardrails

- **Plan first, ask before destructive runs**: write a top-level execution plan listing all configs you intend to run, estimate total compute, and confirm with me before kicking off Phase 1. Don't kill or overwrite anything in `mm25DGS_v5_v4/output_frame_nvs/` (canonical 3DPS results).
- **Use the same evaluation harness** as the main paper: `mmir/evaluation/cli.py` for RA, the existing CRP/ADC eval pipeline for the complex metrics. Do not introduce new metrics.
- **Keep all ablation output in dedicated directories**: `mm25DGS_v5_v4/output_ablations/` and `output/crp_adc_eval_ablations/`.
- **Defaults bolded in the table**: N=20k / 8 views / adaptive on / full LiDAR init / MIMO factored / **L=15** (hardcoded as `SPREAD=15` in `mm25DGS_v5_v4/rasterizer_factorized.py:495` — must be exposed as a configurable flag in Phase 0) / detach on / λ_pos=100 / closed-form BSDF / phase from path length / 500 Adam iters.

---

## Deliverable

A populated `latex/.../tables/ablations.tex` (Supplement Table 8) covering all Tier 1 and Tier 2 axes, plus discussion paragraphs in `sec/A_supplement.tex`, plus a clean `main.pdf` build. If Tier 3 fits in the remaining time, include those too — but only after Tier 1 + Tier 2 are fully populated.
