# mm3DGS — Claude Code Project Instructions

## Project Overview

mm3DGS is a differentiable mmWave radar renderer for inverse rendering and material reconstruction. It trains per-vertex material parameters (permittivity, roughness, thickness, etc.) by comparing rendered radar Range-Azimuth (RA) images against ground truth measurements from a TI MMWCAS cascaded radar (12TX x 16RX). Generalization is evaluated by transferring learned materials to a different sensor: TI IWR1443 single-chip radar (3TX x 4RX).

## Technical Environment

- **Conda env**: `mmir` — always use `/home/adnan/.conda/envs/mmir/bin/python`
- **Rendering backend**: Mitsuba 3 with DrJit (CUDA)
- **Mitsuba variant**: Must call `mi.set_variant('cuda_ad_rgb')` BEFORE importing any module that uses `mi.Vector3f` etc at class level
- **CRITICAL**: Never call `mi.set_variant()` twice — causes segfault in DrJit CUDA backend
- **GPUs**: 2x NVIDIA RTX 4090 — parallelize across GPUs with `CUDA_VISIBLE_DEVICES`

## Key Architecture

```
train.py                          # Training entry point
mmir/
  renderer/                       # Modular differentiable renderer
    integrator.py                 # Ray tracing + ADC synthesis
    bsdf/                         # mmWave BSDF (ITU materials, Kirchhoff approx)
    sampler.py                    # Monte Carlo ray sampling
  evaluation/                     # Evaluation suite
    eval_training_ra.py           # Eval #1: training RA quality (v2 is current)
    eval_radar_transfer.py        # Eval #2: cascaded->single-chip transfer
    renderer_wrapper.py           # Simplified render API for eval
  preprocessing/alignment/        # Sensor pose alignment
    cascaded_alignment.py         # 2DOF + 4DOF cascade alignment
    sc_trajectory_transfer.py     # Cascade->SC pose transfer
  data/                           # Data loading utilities
  losses/                         # Differentiable loss functions
  sensor/                         # Antenna patterns, ADC accumulation
```

## Common Pitfalls

- **DrJit lazy evaluation**: `dr.eval(var)` MUST be called before `dr.sum()` on variables modified by `dr.scatter_inc()`
- **DrJit AD NaN**: `dr.sqrt(dr.maximum(x, 0.0))` produces NaN backward at x=0. Always use `dr.sqrt(dr.maximum(x, mi.Float(1e-20)))`
- **NumPy uint64 overflow**: Use `np.uint64((val * mult) % (2**64))` not `np.uint64(val * mult)`
- **MC noise**: Renderer is stochastic. Correlations vary ~+/-0.03 per scene per run. Compare means over 7+ scenes.
- **Range bin cropping**: Use bins 15..110 for single-chip metrics. Bins 0-14 are TX-RX coupling; bins 110-127 are DFT wrap-around of near-field energy.

## Data Layout

```
data/
  seq_X_frame_Y/                  # 9 benchmark scenes
    scene/mesh.ply                # LiDAR-derived 3D mesh
    scene/pcl.npy                 # Raw LiDAR point cloud
    radar/cascaded_frame_*.npy    # Cascaded radar ADC (9 frames per scene)
    radar/single_chip_frame_*.npy # Single-chip radar ADC (9 frames per scene)
    configs/*.json                # Sensor configurations (aligned + unaligned)
  alignment_data/                 # Pre-computed SC alignment configs
    seq_X_frame_Y/
      cascade/                    # Per-frame cascade alignment results
      single_chip/                # Trajectory-transferred + refined SC configs
assets/antenna_pattern/
  MMWCAS/tx1_76.npy, rx1_76.npy  # Cascaded radar antenna patterns
  IWR1443/pattern_76.npy         # Single-chip radar antenna pattern
```

## Running

```bash
# Train on one scene
python train.py --config output/training/seq_0_frame_135/config.json

# Evaluate (from mmir/evaluation/)
python -m mmir.evaluation.cli --scenes seq_0_frame_135 --evaluations training transfer
```

## 3DPS canonical results location (post 2026-05-06)

The NeurIPS submission's canonical 3DPS results live at
`mm25DGS_v5_v4/output_ablations/tier1/lidar_init/no_occlusion/<scene>/`
(bare scene-name leaves), NOT at the legacy
`mm25DGS_v5_v4/output_frame_nvs/<scene>_..._pass2_N20000/` location.

The recipe change: an earlier 3DPS revision included a 4th LiDAR-init
stage (Mitsuba-3 ray-cast occlusion test). Tier-1 ablation showed it
net-hurts test |RA| Corr by ~0.013, so we dropped it. Trainer default
is now `enable_occlusion=False`. Pass `--enable_occlusion` to reproduce
the legacy 4-stage init.

Headline 6-scene mean numbers (canonical 3-stage init): test |RA| Corr
**0.600**, train |RA| Corr **0.825**, test |CRP| Corr **0.612**, ADC
envelope **0.587**, ~3.6 min/scene on RTX 4090.

Loaders that auto-resolve to the canonical location:
- `figures/generate_tables.py` `find_ours_results` (bare-name dir,
  legacy glob fallback preserved)
- `figures/generate_crp_adc_paper_table.py` `DEFAULT_OURS_DIR`
- `figures/generate_ablation_table.py` `DEFAULT_OURS_DIR`
- `mmir/evaluation/eval_crp_adc.py` `_DEFAULT_OURS_DIR` +
  `_DEFAULT_RUN_TAG_TEMPLATE`
