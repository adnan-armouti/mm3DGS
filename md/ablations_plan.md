# Ablation Experiments Plan

## Goal

Run a comprehensive ablation suite for the supplement that satisfies NeurIPS senior-reviewer expectations for a method-paper at oral/spotlight quality. Each ablation isolates one design choice on the same 6-scene ColoRadar benchmark used in the main paper, reports impact on test-view |RA| / |CRP| / ADC envelope correlations, and reports wall-clock impact.

The supplement currently has a **placeholder ablation table** (Table 8, `tab:ablations`, in `\section{Ablations}` of `sec/A_supplement.tex`). This plan expands it to a senior-reviewer-quality suite.

**Scope: Tier 1 + Tier 2.** Tier 3 deferred. ~13 GPU-hours total on 2× RTX 4090.

---

## Compute environment

- 2× NVIDIA RTX 4090 (24 GB each).
- Conda env: `/home/adnan/.conda/envs/mmir/bin/python`.
- One ablation config per GPU at a time (parallelize via `CUDA_VISIBLE_DEVICES=0` and `=1`).
- Project root: `/home/adnan/Desktop/mm3DGS/`.
- 3DPS training script: `mm25DGS_v5_v4/<train_entry>` (resolve exact path during Phase 0).
- Existing 3DPS outputs (reference): `mm25DGS_v5_v4/output_frame_nvs/seq_X_frame_Y_*_pass2_N20000*/`.

---

## Required ablations

### Tier 1 — must-have

| # | Ablation | Configurations | Justification |
|---|---|---|---|
| 1 | **Point count `N`** | `{2k, 5k, 10k, 20k (default), 50k}` | Validates the N=20k choice; shows scaling |
| 2 | **Number of training views** | `{2, 4, 6, 8 (default)}` | Validates multi-view-tractability claim |
| 3 | **Adaptive density control** | on (default) / off | Validates the split/prune mechanism |
| 4 | **LiDAR initialization stages** (each individually disabled) | a) no azimuth-cone cull, b) no occlusion ray-casting, c) no cosine-weighted resample, d) no farthest-point sampling | Validates each stage |
| 5 | **Quasi-static MIMO factorization** | on (default) / off (full O(N·N_TX·N_RX) BSDF eval) | Validates the runtime claim |

### Tier 2 — strong-to-have

| # | Ablation | Configurations | Justification |
|---|---|---|---|
| 6 | **PSF kernel size `L`** | `{5, 9, 15 (default), 21, 25}` (odd-only, so the offset grid `arange(-(L//2), L//2+1)` is well-centered at 0) | Validates L=15. **Note**: this is hardcoded as `SPREAD = 15` at `mm25DGS_v5_v4/rasterizer_factorized.py:495`. Phase 0 must convert it to a configurable flag before this ablation can run. Even values for `L` produce `L+1` bins due to integer-division asymmetry in `arange(-(L//2), L//2+1)`, so we stick to odd values. |
| 7 | **Carrier-phase detach for position gradients** | detached (default) / not detached | Validates gradient-stability design |
| 8 | **L2 position anchor `λ_pos`** | `{0, 1, 100 (default), 1000}` | Validates LiDAR anchoring |
| 9 | **Closed-form ITU-R P.2040 BSDF vs learned material features** | closed-form (default) / per-point learned MLP | Validates explicit-physics claim |

---

## Phase 0 — pre-flight verification (DO NOT SKIP, ~0.5 day)

**Critical: do this before launching expensive runs.** Goal is to confirm every artifact the table/figure generators need is actually produced by the ablation runs.

### 0.1 — Inventory the existing 3DPS pipeline

Read and verify:

1. **Training script**: locate the actual entry point. Likely under `mm25DGS_v5_v4/`. Document the command-line args it accepts (config path, override flags, etc.).
2. **Existing 3DPS results layout** at `mm25DGS_v5_v4/output_frame_nvs/seq_0_frame_135_*_pass2_N20000*/`. Inventory all files: `results.json`, `metrics_train.json`, any `rendered_ra_*.npy`, configs, etc.
3. **CRP/ADC eval pipeline**: how is `output/crp_adc_eval/results.json` produced for 3DPS? Find the script and confirm it can be re-pointed at an ablation output directory.

### 0.2 — Check that each Tier 1+2 axis is configurable

For each of the 9 ablation axes, find where in the code it lives and confirm the override mechanism:

| Ablation | Likely location |
|---|---|
| Point count `N` | training config (probably `--N` flag or YAML) |
| Number of training views | data split config; may need to subsample the 8-view list |
| Adaptive density control | optimizer config (likely a boolean or "split iters" list) |
| LiDAR init stages (cull/occlusion/cosine/FPS) | `mmir/preprocessing/...` or 3DPS-specific init module |
| Quasi-static MIMO factorization | renderer config; may be a boolean or tied to BSDF eval mode |
| PSF kernel size `L` | **hardcoded as `SPREAD = 15` at `mm25DGS_v5_v4/rasterizer_factorized.py:495`**; must be exposed as a flag in Phase 0. The CUDA kernel reads spread at runtime from `psf_real.size(0)`, so the flag wires into `HannPSFTable(K, spread, ...)` constructor without further code changes. |
| Carrier-phase detach | optimizer / loss config; likely a flag |
| `λ_pos` | optimizer config |
| Closed-form vs learned BSDF | BSDF module; likely a config-driven choice |

If any axis is not currently configurable via flags, add the override before kicking off Phase 1. **This is the single highest-impact pre-flight step.**

### 0.3 — Dry-run 1 ablation config end-to-end

Pick one fast Tier-1 config (e.g., `N=2k`, single scene, 50 iterations). Run the full training → finalize → metric computation → ablation-table-generator pipeline on it. Verify:

1. Training writes `results.json` (or equivalent) with `final_test_cart_corr` and friends.
2. CRP/ADC eval can be invoked on the new output directory.
3. The output directory naming convention is consistent and parseable by an aggregation script.

If any step fails, fix it before kicking off Phase 1.

### 0.4 — Decide on output directory layout

Recommended:
```
mm25DGS_v5_v4/output_ablations/
├── tier1/
│   ├── point_count_N/
│   │   ├── N_2k/
│   │   │   ├── seq_0_frame_135_*/results.json
│   │   │   └── ...
│   │   ├── N_5k/
│   │   ├── N_10k/
│   │   ├── N_20k_default/   # symlink to existing default 3DPS run, no need to re-run
│   │   └── N_50k/
│   ├── num_train_views/
│   │   ├── views_2/
│   │   ├── views_4/
│   │   ├── views_6/
│   │   └── views_8_default/
│   ├── adaptive_density/
│   │   ├── on_default/
│   │   └── off/
│   ├── lidar_init/
│   │   ├── no_cull/
│   │   ├── no_occlusion/
│   │   ├── no_cosine_resample/
│   │   ├── no_fps/
│   │   └── full_default/
│   └── mimo_factorization/
│       ├── on_default/
│       └── off/
└── tier2/
    ├── psf_kernel_L/
    ├── phase_detach/
    ├── lambda_pos/
    └── bsdf_form/
```

### 0.5 — Write the ablation-table generator (BEFORE running)

`figures/generate_ablation_table.py` should:

1. Walk `mm25DGS_v5_v4/output_ablations/<tier>/<ablation_name>/<config>/` directories.
2. For each config: aggregate per-scene `results.json` into mean test |RA| Corr/PSNR/SSIM/RMSE, plus mean wall-clock and peak GPU memory.
3. For each config: also evaluate CRP and ADC envelope correlation by re-pointing the existing CRP/ADC eval pipeline at the new output dir.
4. Emit `latex/.../tables/ablations.tex` populating Supplement Table 8.

**Run it on the dry-run output from 0.3 to verify it produces a syntactically valid LaTeX table** before launching real runs.

---

## Phase 1 — Tier 1 runs (~10 GPU-hours, parallelized)

Run order (default → ablation, parallelized across 2 GPUs):

1. **Point count N**: `{N=2k, N=5k, N=10k, N=50k}` × 6 scenes — skip `N=20k` (default already exists).
   - Estimated: ~6 GPU-hours total (N=50k may take longer per scene)
2. **Number of training views**: `{2, 4, 6}` × 6 scenes — skip `8` (default).
   - Estimated: ~1 GPU-hour (faster with fewer views)
3. **Adaptive density off**: 1 config × 6 scenes ≈ 0.5 GPU-hour
4. **LiDAR init stages disabled**: 4 configs × 6 scenes ≈ 1 GPU-hour
5. **MIMO factorization off**: 1 config × 6 scenes — slowest of all (~10× default per scene). Estimated: ~3 GPU-hours. May need to reduce `N` for this single config to fit memory; if so, document.

**Risk: MIMO-factorization-off may OOM** at full `O(N·N_TX·N_RX)`. If it OOMs at N=20k, reduce to N=5k for that single config and note in the ablation table.

---

## Phase 2 — Tier 2 runs (~3 GPU-hours)

1. PSF kernel L: `{5, 9, 21, 25}` × 6 scenes ≈ 1.5 GPU-hours (odd-only; skip 15 — default; L=25 is ~1.7× slower than default)
2. Carrier-phase detach off: 1 config × 6 scenes ≈ 0.5 GPU-hour
3. λ_pos sweep: `{0, 1, 1000}` × 6 scenes ≈ 1 GPU-hour
4. Learned-BSDF MLP: 1 config × 6 scenes ≈ 0.5-1 GPU-hour

---

## Phase 3 — aggregate + write up (~1 day)

1. Run `figures/generate_ablation_table.py` to populate Supplement Table 8.
2. Replace the placeholder text in `sec/A_supplement.tex` `\section{Ablations}` with real numbers.
3. Write 1-paragraph discussion per Tier-1 ablation summarizing what the data validates.
4. Note any surprising results in `app:limitations`.
5. Rebuild `main.pdf`.

---

## Metrics per ablation

For every configuration, report on the **6-scene mean held-out test view**:

- `|RA|` Pearson correlation, PSNR, SSIM, RMSE
- `|CRP|` Pearson correlation
- ADC envelope correlation `ρ_|·|`
- Wall-clock per scene (mean across 6 scenes)
- Peak GPU memory (mean across 6 scenes)
- Number of failed / unstable runs (NaNs, divergence)

For the training-view-count ablation, also report **train mean** for each view count.

---

## Default 3DPS configuration (the bolded reference row in Table 8)

- Point count `N = 20{,}000`
- Training views: 8 (all bracketing frames)
- Adaptive density control: on (split top 5%, prune bottom 5% at iters {100, 200, 300, 400})
- LiDAR initialization: full 4-stage pipeline (cull / occlusion / cosine resample / FPS)
- MIMO factorization: quasi-static (factored)
- PSF kernel `L = 15` (hardcoded as `SPREAD = 15` at `rasterizer_factorized.py:495`)
- Carrier-phase detach for position gradients: enabled
- L2 position anchor `λ_pos = 100`
- BSDF: closed-form ITU-R P.2040, 6 parameters per point
- Phase: from geometric path length (not learned)
- 500 Adam iterations, RTX 4090

---

## Risk register

- **MIMO-factorization-off run may OOM at full O(N·N_TX·N_RX).** Mitigation: reduce N to 5k for that single config; note in the table caption.
- **Learned-BSDF MLP may not fit in 500 iterations.** Mitigation: extend training budget for that config and report the elapsed time delta.
- **N=50k may exceed RTX 4090 memory** with default kernel size. Mitigation: reduce L for that config; document.
- **NaN/divergence** on aggressive configs (e.g., adaptive-density-off with large N). Mitigation: track failed runs in a "failed runs" column.
- **Adaptive-density-off may diverge late.** Track and report.
- **Configurability gaps** (Phase 0 may reveal that some ablation axes aren't currently flag-controlled). Mitigation: budget 0.5-1 day in Phase 0 to add the missing overrides.

---

## Deliverables

1. Populated Supplement Table 8 (`latex/.../tables/ablations.tex`) with all Tier 1 + Tier 2 numbers.
2. Per-Tier-1-ablation discussion paragraph in `sec/A_supplement.tex` (replace placeholder text).
3. New supplement subsection `app:ablations:discussion` summarizing what each ablation validates.
4. Optional: 1 figure showing N-vs-test-correlation curve.
5. Clean `main.pdf` build with the populated ablation table.
