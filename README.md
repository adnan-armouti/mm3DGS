# Millimeter-Wave Inverse Rendering

Differentiable FMCW radar renderer with physics-based material optimization
for scene reconstruction from commodity 77 GHz radar.

## Overview

This codebase implements a complete pipeline for millimeter-wave inverse
rendering:

1. **Preprocessing** — Convert raw ColoRadar dataset captures into scene
   meshes, radar ADC data, and configuration files
2. **Radar-LiDAR Alignment** — Cascaded optimization to align radar and
   LiDAR coordinate frames (4-DOF LiDAR-based + 2-DOF renderer-based)
3. **Training** — Differentiable rendering with physics-based BSDF
   optimization (materials, geometry, pose)
4. **Evaluation** — Quantitative metrics (RA correlation, radar transfer,
   3D occupancy, 3D reconstruction) and publication figure generation

---

## Project Structure

```
submission/
├── train.py                          # Main training entry point
├── requirements.txt                  # Pip dependencies
├── environment.yml                   # Conda environment specification
├── configs/scenes/                   # Training scene configs (JSON)
├── assets/antenna_pattern/           # Antenna beam patterns
├── data/                             # Scene data (see Data Preparation below)
│   ├── antenna_patterns/             #   Single-chip antenna pattern
│   └── seq_<N>_frame_<F>/           #   Per-scene data directories
│       ├── configs/                  #     Radar configs (original + aligned)
│       ├── radar/                    #     ADC captures (.npy)
│       ├── lidar/                    #     LiDAR frames (.npy)
│       └── scene/                    #     Mesh + point cloud (generated)
├── third_party/                      # External dependencies
│   ├── PoissonRecon/                 #   Screened Poisson surface reconstruction
│   └── fsdBSDFpaper/                 #   FSD diffraction sampling tables
└── mmir/
    ├── sensor/                       # FMCW radar front-end
    │   ├── config.py                 #   Radar configuration (FMCWConfig)
    │   ├── element_patterns.py       #   Antenna gain patterns
    │   └── adc_accumulator.py        #   Phasor-to-ADC accumulation
    ├── renderer/                     # Differentiable ray-tracing engine
    │   ├── renderer.py               #   Main renderer (FMCWRendererRef)
    │   ├── scene_context.py          #   Scene state and configuration
    │   ├── config.py                 #   Render configuration (RenderConfigRef)
    │   ├── sampler.py                #   Ray sampling strategies
    │   ├── bsdf/                     #   Bidirectional scattering functions
    │   │   ├── mmwave_scalar.py      #     KA+SPM physics BSDF (CSV-BSDF)
    │   │   └── mmwave_jones.py       #     Jones-matrix polarimetric BSDF
    │   ├── integrator/               #   Path integration
    │   │   ├── core.py               #     Bounce loop and NEE
    │   │   ├── synthesis_forward.py  #     Forward (non-diff) ADC synthesis
    │   │   ├── synthesis_differentiable.py  # Cached-geometry diff synthesis
    │   │   └── synthesis_e2e.py      #     End-to-end differentiable synthesis
    │   ├── specular/                 #   Specular path handling
    │   │   ├── sms.py                #     Specular Manifold Sampling
    │   │   ├── image_method.py       #     Image-method specular refinement
    │   │   └── hash_utils.py         #     Path deduplication
    │   ├── diffraction/              #   Edge diffraction (FSD-BSDF)
    │   │   ├── fsd_bsdf.py           #     Fresnel-scattering distribution
    │   │   ├── fsd_aperture.py       #     Aperture construction
    │   │   ├── fsd_aperture_gpu.py   #     GPU-accelerated aperture (Slang)
    │   │   └── fsd_sampling_tables.py #    Precomputed importance sampling
    │   ├── materials/                #   Material parameterization
    │   ├── scene_params/             #   Differentiable scene parameters
    │   └── utils/                    #   Renderer utilities
    ├── data/                         # Data I/O and configuration
    │   ├── ra_utils.py               #   ADC-to-RA conversion
    │   ├── data_utils.py             #   Data loading helpers
    │   ├── io_utils.py               #   File I/O utilities
    │   └── adc_normalization.py      #   ADC signal normalization
    ├── losses/                       # Training loss functions
    │   ├── drjit_ra_loss.py          #   RA-domain correlation loss
    │   ├── loss_utils.py             #   SSIM, normalization, metrics
    │   └── multi_chirp_loss.py       #   Multi-chirp alignment loss
    ├── preprocessing/                # Data preprocessing pipeline
    │   ├── preproc.py                #   Main preprocessing orchestrator
    │   ├── config_utils.py           #   Radar config generation
    │   ├── mesh_utils.py             #   Poisson mesh reconstruction
    │   ├── alignment/                #   Radar-LiDAR alignment
    │   │   ├── cascaded_alignment.py #     Cascaded alignment orchestrator
    │   │   ├── cascaded_lidar.py     #     LiDAR-based 4-DOF alignment
    │   │   ├── cascaded_lidar_gpu.py #     GPU-accelerated alignment
    │   │   └── gpu_utils/            #     GPU voxelization and metrics
    │   └── ColoRadar_tools/          #   ColoRadar dataset utilities (third-party)
    └── evaluation/                   # Post-processing and evaluation
        ├── cli.py                    #   Evaluation CLI orchestrator
        ├── phase1_render.py          #   Forward rendering for evaluation
        ├── phase2_evaluate.py        #   Metrics computation
        ├── fig_common.py             #   Shared figure utilities
        ├── figures/                  #   Publication figure scripts
        ├── supplement_figs/          #   Supplemental figure scripts
        └── utils/                    #   Evaluation helper modules
```

---

## Installation

### 1. Create Conda Environment

```bash
# Option A: From environment.yml (recommended)
conda env create -f environment.yml
conda activate mmir

# Option B: Manual setup
conda create -n mmir python=3.12
conda activate mmir
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA

If not using `environment.yml`, install PyTorch matching your CUDA version:

```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Sionna RT (Optional, for baseline comparison)

```bash
pip install sionna-rt
```

### 4. Initialize Third-Party Dependencies

#### PoissonRecon (required for mesh reconstruction)

```bash
mkdir -p third_party/PoissonRecon
cd third_party/PoissonRecon
git clone https://github.com/mkazhdan/PoissonRecon.git .

# Build
sudo apt-get install -y build-essential libpng-dev libjpeg-dev zlib1g-dev
make poissonrecon surfacetrimmer COMPILER=gcc CFLAGS+=' -DBIG_DATA'
cd ../..
```

#### FSD Diffraction Tables (required for edge diffraction)

The renderer's edge diffraction module uses precomputed inverse CDF tables
from the [fsdBSDFpaper](https://github.com/ssteinberg/fsdBSDFpaper) repository.

```bash
mkdir -p third_party/fsdBSDFpaper
cd third_party/fsdBSDFpaper
git clone https://github.com/ssteinberg/fsdBSDFpaper.git .
cd ../..

# Symlink the precomputed tables into the diffraction module
ln -s "$(pwd)/third_party/fsdBSDFpaper/precompiled_fsd_tables" \
      mmir/renderer/diffraction/fsd_tables
```

### 5. System Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **CUDA**: 11.8+ or 12.x
- **RAM**: 128 GB+ recommended
- **OS**: Ubuntu 20.04+ (Linux required for Mitsuba/DrJit CUDA backend)
- **Python**: 3.10+

---

## Data Preparation

The supplementary material includes lightweight data per scene (configs,
single-chip radar, lidar frames). Several large files must be generated
from the raw ColoRadar dataset before training.

### Required Generated Files

For each scene, the following files are **not** included and must be
generated via preprocessing:

| File | Description | How to generate |
|---|---|---|
| `scene/pcl.npy` | LiDAR scene point cloud | Preprocessing `scene` command |
| `scene/mesh.ply` | Reconstructed triangle mesh | Preprocessing `scene` command (auto) or `mesh_utils` |
| `radar/cascaded_frame_*.npy` | Cascaded radar ADC data | Preprocessing `adc --cascade` command |
| `radar/single_chip_frame_*.npy` | Single-chip radar ADC data | Preprocessing `adc --single-chip` command |
| `lidar/lidar_frame_*.npy` | Synchronized LiDAR frames | Preprocessing `lidar-frames` command |

Only the radar configuration JSON files (in `configs/`) are included in
the supplementary material.

### Generate All Scenes

Run the following commands to generate the missing files for all 9 scenes.
Replace `/path/to/dataset` with the path to the ColoRadar KITTI-format
dataset directory (e.g., `/path/to/coloRadar/raw/kitti/2_28_2021_outdoors_run`).

```bash
DATASET=/path/to/coloRadar/raw/kitti/2_28_2021_outdoors_run
CALIB=mmir/preprocessing/calib

# seq_0_frame_135
python -m mmir.preprocessing all --seq 0 --frame 135 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_0_frame_390
python -m mmir.preprocessing all --seq 0 --frame 390 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_0_frame_451
python -m mmir.preprocessing all --seq 0 --frame 451 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_1_frame_185
python -m mmir.preprocessing all --seq 1 --frame 185 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_1_frame_277
python -m mmir.preprocessing all --seq 1 --frame 277 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_1_frame_438
python -m mmir.preprocessing all --seq 1 --frame 438 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_2_frame_105
python -m mmir.preprocessing all --seq 2 --frame 105 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_2_frame_160
python -m mmir.preprocessing all --seq 2 --frame 160 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose

# seq_2_frame_300
python -m mmir.preprocessing all --seq 2 --frame 300 --num-radar-frames 1 \
    --dataset-dir $DATASET --calib-path $CALIB --out-root data \
    --cascade --single-chip --num-lidar-frames 50 --buffer-distance 1.0 \
    --normals-radius 0.1 --remove-behind-radar --verbose
```

Each command generates `scene/pcl.npy`, `scene/mesh.ply`,
`radar/cascaded_frame_*.npy`, `radar/single_chip_frame_*.npy`, and
`lidar/lidar_frame_*.npy` for the corresponding scene.

---

## Usage

### 1. Preprocessing

Convert raw ColoRadar dataset captures into training-ready format.

#### All-in-One

```bash
python -m mmir.preprocessing all \
    --seq 1 \
    --frame 185 \
    --num-radar-frames 1 \
    --dataset-dir /path/to/coloRadar/raw/kitti/2_28_2021_outdoors_run \
    --calib-path mmir/preprocessing/calib \
    --out-root data \
    --cascade \
    --single-chip \
    --num-lidar-frames 50 \
    --buffer-distance 1.0 \
    --normals-radius 0.1 \
    --remove-behind-radar \
    --verbose
```

This generates:
- `data/seq_1_frame_185/scene/pcl.npy` — LiDAR scene point cloud
- `data/seq_1_frame_185/scene/mesh.ply` — Reconstructed mesh
- `data/seq_1_frame_185/configs/cascaded_frame_185.json` — Radar config
- `data/seq_1_frame_185/radar/cascaded_frame_185.npy` — ADC data
- `data/seq_1_frame_185/lidar/lidar_frame_*.npy` — Synchronized LiDAR

#### Individual Steps

```bash
# Scene point cloud + mesh
python -m mmir.preprocessing scene \
    --seq 1 --frame 185 \
    --dataset-dir /path/to/dataset --calib-path mmir/preprocessing/calib \
    --out-root data --num-lidar-frames 50 --verbose

# Radar configs
python -m mmir.preprocessing configs \
    --seq 1 --frame 185 \
    --dataset-dir /path/to/dataset --calib-path mmir/preprocessing/calib \
    --out-root data --cascade --single-chip --verbose

# ADC data
python -m mmir.preprocessing adc \
    --seq 1 --frame 185 --num-radar-frames 1 \
    --dataset-dir /path/to/dataset --calib-path mmir/preprocessing/calib \
    --out-root data --cascade --verbose

# LiDAR frames
python -m mmir.preprocessing lidar-frames \
    --seq 1 --frame 185 --num-radar-frames 1 \
    --dataset-dir /path/to/dataset --calib-path mmir/preprocessing/calib \
    --out-root data --verbose
```

#### Mesh Reconstruction (standalone)

```bash
python -m mmir.preprocessing.mesh_utils \
    data/seq_1_frame_185/scene/pcl.npy \
    data/seq_1_frame_185/scene/mesh.ply \
    --depth 10
```

---

### 2. Cascaded Radar-LiDAR Alignment

Optimize the rigid alignment between radar and LiDAR coordinate frames.

```bash
# Run full cascaded alignment on all benchmark scenes
python -m mmir.preprocessing.alignment.cascaded_alignment \
    --data-root data \
    --output-root output/alignment

# Print alignment summary only (no optimization)
python -m mmir.preprocessing.alignment.cascaded_alignment \
    --data-root data \
    --summary-only
```

The orchestrator runs two methods per scene:
1. **LiDAR-based 4-DOF** — optimizes range, azimuth, elevation rotation,
   and azimuth rotation offsets using GPU-accelerated voxel correlation
2. **Renderer-based 2-DOF** — optimizes range and azimuth offsets by
   differentiable rendering and RA-image correlation

The best alignment (highest RA correlation) is selected automatically.

---

### 3. Training

Optimize scene parameters (materials, geometry, pose) via differentiable
rendering.

```bash
python train.py --config configs/scenes/my_scene.json
```

#### CLI Options

| Argument | Description |
|---|---|
| `--config` | Path to JSON scene configuration (required) |
| `--output-dir` | Override output directory from config |
| `--max-iterations` | Override number of training iterations |
| `--initial-material` | ITU material name for initialization (e.g. `concrete`) |
| `--error-init` | Run error-map initialization before training |

#### Example

```bash
python train.py \
    --config configs/scenes/seq_1_frame_185.json \
    --max-iterations 50 \
    --initial-material concrete \
    --error-init
```

Training produces:
- `output/<scene>/checkpoints/` — Model checkpoints (`.pt`)
- `output/<scene>/metrics/` — Per-iteration metrics (`.json`)
- `output/<scene>/visualizations/` — RA image comparisons (`.png`)

---

### 4. Evaluation

The evaluation pipeline has two phases: (1) forward rendering of trained
models across multiple views, and (2) quantitative metric computation.

```bash
# Run everything (render + evaluate)
python -m mmir.evaluation all \
    --output-root output/postprocess

# Phase 1 only: render all views
python -m mmir.evaluation render \
    --scenes seq_0_frame_135,seq_1_frame_185 \
    --output-root output/postprocess \
    --batch-size 5000

# Phase 2 only: compute metrics
python -m mmir.evaluation evaluate \
    --output-root output/postprocess
```

#### Evaluation Subcommands

| Command | Description |
|---|---|
| `render` | Phase 1: forward rendering across views |
| `evaluate` | Phase 2: compute metrics and generate tables |
| `all` | Run both phases |
| `training-ra` | Evaluation #1: training RA image metrics |
| `radar-transfer` | Evaluation #2: radar-to-radar transfer |
| `occupancy` | Evaluation #3: 3D occupancy comparison |
| `reconstruction` | Evaluation #4: 3D reconstruction quality |

#### 3D Evaluation (Occupancy / Reconstruction)

These evaluations require the original ColoRadar dataset for LiDAR ground
truth:

```bash
python -m mmir.evaluation occupancy \
    --output-root output/postprocess \
    --coloradar-dataset-dir /path/to/coloRadar/raw/kitti/2_28_2021_outdoors_run \
    --coloradar-calib-dir mmir/preprocessing/calib

python -m mmir.evaluation reconstruction \
    --output-root output/postprocess \
    --az-start -21.0 --az-end 69.0 --az-step 1.0 \
    --radar-percentile 98.0
```

---

### 5. Figure Generation

Generate publication-quality figures from evaluation results.

```bash
# All figures are generated automatically during evaluation:
python -m mmir.evaluation evaluate --output-root output/postprocess

# Or generate individual figures:
python -m mmir.evaluation.figures.generate_fig_training_ra \
    --input_dir output/postprocess/training_ra \
    --output_dir output/postprocess/figures \
    --training_dir output/train \
    --data_dir data

python -m mmir.evaluation.figures.generate_fig_materials \
    --training_dir output/train \
    --output_dir output/postprocess/figures \
    --data_dir data

python -m mmir.evaluation.figures.generate_fig_3d_occupancy \
    --input_dir output/postprocess/occupancy \
    --output_dir output/postprocess/figures

python -m mmir.evaluation.figures.generate_fig_teaser \
    --input_dir output/postprocess \
    --output_dir output/postprocess/figures

python -m mmir.evaluation.figures.generate_fig_pipeline \
    --output_dir output/postprocess/figures
```

Available figure scripts:
- `generate_fig_teaser` — Teaser figure (Figure 1)
- `generate_fig_pipeline` — Pipeline overview (Figure 2)
- `generate_fig_training_ra` — Training RA comparison
- `generate_fig_materials` — Optimized material maps
- `generate_fig_materials_rgb` — Material RGB visualization
- `generate_fig_materials_deviation` — Material deviation analysis
- `generate_fig_normals` — Surface normal maps
- `generate_fig_normals_deviation` — Normal deviation analysis
- `generate_fig_3d_occupancy` — 3D occupancy comparison
- `generate_fig_antenna_layouts` — Antenna array layouts
- `generate_fig_beam_patterns` — Antenna beam patterns

---

## Dataset

This project uses the [ColoRadar](https://arpg.github.io/coloradar/)
dataset, which provides synchronized LiDAR, radar, and IMU measurements
from outdoor driving scenarios. The preprocessing pipeline expects the
KITTI-formatted version of the dataset.

The following files are provided by the ColoRadar dataset authors and are
**not** part of our contribution:
- `mmir/preprocessing/ColoRadar_tools/` — dataset loading utilities,
  adapted from the official
  [ColoRadar devkit](https://github.com/arpg/ColoRadar_tools)
- `mmir/preprocessing/calib/` — sensor calibration files (antenna
  configurations, waveform parameters, inter-sensor transforms) shipped
  with the ColoRadar dataset

---

## Configuration

Training and rendering are configured via JSON files. Key parameters:

```json
{
    "scene_mesh": "data/seq_1_frame_185/scene/mesh.ply",
    "gt_adc_file": "data/seq_1_frame_185/radar/cascaded_frame_185.npy",
    "radar_config": "data/seq_1_frame_185/configs/cascaded_frame_185.json",
    "output_dir": "output/seq_1_frame_185",
    "num_iterations": 50,
    "learning_rate": 0.1,
    "samples_per_tx": 14000,
    "max_depth": 1,
    "optimize_materials": true,
    "optimize_pose": true,
    "optimize_geometry": false,
    "enable_diffraction": true,
    "enable_slab_fresnel": true,
    "enable_image_method": true
}
```

---

## License

[License information will be added upon publication.]
