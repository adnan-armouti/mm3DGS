# mm25DGS_v2: Verified Rasterizer → Trainable Gaussian Rasterizer

**Prerequisite**: The forward-only mesh-vertex rasterizer from `mm25DGS_v2_implementation_plan.md` passes Step 10 verification on all 7 scenes (cart_corr within 0.05 of mmIR).

**Directory**: continues in `/home/adnan/Desktop/mm3DGS/mm25DGS_v2/`

## Goal

Convert the verified mesh-vertex rasterizer into a trainable Gaussian rasterizer that learns per-primitive material parameters from GT radar measurements. The conversion follows the same incremental philosophy: change one thing at a time, verify after each change.

## What mmIR trains (the target to match)

mmIR optimizes these parameters over 500 iterations:
- **Per-vertex materials** (N_verts × 6): eps_real, eps_imag, sigma_h, l_c, tau, thickness
- **Per-vertex normals** (N_verts × 3): normal direction offsets
- **Antenna patterns** (4 × 361): shared TX/RX E-plane and H-plane patterns

mmIR does NOT optimize vertex positions or sensor pose.

Loss: RA magnitude MSE with min-max normalization (linear, no log).

## Conversion stages

The rasterizer-to-trainer conversion has three stages:
1. **DrJit → PyTorch** (make the forward pass differentiable under PyTorch autograd)
2. **Fixed mesh → learnable parameters** (add optimizer, loss, training loop)
3. **Mesh vertices → Gaussians** (replace fixed mesh vertices with learnable Gaussian primitives)

Each stage is verified before proceeding to the next.

---

## Stage A: Port forward pass from DrJit to PyTorch

### Prerequisite state

The verified rasterizer (`rasterizer.py`) calls mmIR's DrJit code for BSDF, antenna, radar equation, and phase scatter. It produces correct ADC but is not differentiable under PyTorch.

### What to do

Port each DrJit component to PyTorch, one at a time, verifying numerical equivalence after each port. The order follows data dependencies (upstream first):

### Step A1: Material reparameterization → PyTorch

**What**: Port `reparameterize_physics_params_drjit()` to PyTorch.

**mmIR code**: `mmir/renderer/bsdf/reparameterization.py` lines 95-115. Sigmoid for bounded params, exp for log-scale params.

**Port**: Already done in `mm25DGS/reparameterization.py`. Verify it matches by running both on the same `raw_params` for all 7 scenes.

**Verification**: `max(|torch_output - drjit_output|) < 1e-6` for all 6 columns, all vertices, all scenes.

### Step A2: BSDF → PyTorch

**What**: Port `mmwave_jones.py:eval_f_cos_physics()` to PyTorch.

**Note**: This was already verified identical in the earlier debugging session (ratio 1.0000 ± 0.0001 across 50 random vertices with same inputs). The existing `mm25DGS/bsdf_torch.py` contains a correct PyTorch Jones BSDF.

**Verification**: Run both (DrJit mmIR and PyTorch) on per-vertex materials for all 7 scenes. For each vertex, the same `(wo, wi, n, materials)` must produce `|f_cos_drjit - f_cos_torch| / |f_cos_drjit| < 1e-4`.

### Step A3: Antenna gain → PyTorch

**What**: Port `evaluate_combined_gain()` to PyTorch.

**mmIR code**: `mmir/sensor/element_patterns.py` lines 318-360 and 613-700. Builds local frame from boresight, projects direction, interpolates E/H plane patterns, multiplies by C_scale.

**Port**: The existing `mm25DGS/antenna_torch.py` has a PyTorch antenna evaluator. Verify it matches mmIR's `evaluate_combined_gain()` for the same direction/boresight inputs. Pay close attention to the local frame construction (Gram-Schmidt from boresight) and the +180° angle offset convention.

**Verification**: For 1000 random (direction, boresight) pairs, `|torch_gain - drjit_gain| / |drjit_gain| < 0.01`.

### Step A4: Phase + phasor scatter → PyTorch

**What**: Replace DrJit `scatter_add` with PyTorch equivalent.

**mmIR code**: `synthesis_e2e.py` lines 521-564. Computes `phi = phi_const + phi_slope × t_k`, then `scatter_add(weight × cos(phi), flat_idx)`.

**Port**: PyTorch chunked phasor accumulation (already implemented in `mm25DGS/adc_synthesis.py:_phasor_chunk()`). The key difference: PyTorch uses a loop over Gaussian chunks with `checkpoint`, while DrJit uses vectorized scatter over all paths × K at once.

**Verification**: Given identical weights, distances, and frequency params, compare ADC outputs. `max(|torch_adc - drjit_adc|) / max(|drjit_adc|) < 1e-5`.

### Step A5: Full PyTorch forward pass verification

**What**: Assemble all ported components into a single PyTorch forward function. Run on all 7 scenes with trained parameters.

**Verification**: For each scene, compare the PyTorch rasterizer ADC against the DrJit rasterizer ADC (from the forward-only plan's Step 10):
- `corr(torch_ra_cart, drjit_ra_cart) > 0.99`
- `|cart_corr_torch - cart_corr_drjit| < 0.01`

---

## Stage B: Add training loop (mesh vertices, fixed topology)

### Prerequisite state

A fully PyTorch forward pass that matches the DrJit rasterizer on all 7 scenes.

### What to do

Add an optimizer and loss function. Train per-vertex materials on the same mesh topology mmIR uses. This is the most direct comparison: same mesh, same parameterization, same loss — only the renderer architecture differs (rasterizer vs ray tracer).

### Step B1: Loss function

**What**: Implement the same loss function mmIR uses.

**mmIR loss**: RA magnitude MSE with min-max normalization, linear (no log). Computed in `train.py` via the same `adc_to_ra_complex` FFT pipeline.

**Implementation**: Already exists in `mm25DGS/losses.py:compute_loss()`. Verify it produces the same loss value as mmIR for the same rendered + GT ADC pair.

**Verification**: Load mmIR's rendered ADC (from gold reference) and GT ADC. Compute loss with both mmIR's code and `compute_loss()`. Must match within 1e-6.

### Step B2: Optimizer setup

**What**: Adam optimizer on per-vertex materials (N_verts × 6), matching mmIR's hyperparameters:
- `lr_materials = 0.5`
- `adam_beta1 = 0.9, adam_beta2 = 0.999, adam_eps = 1e-8`
- LR warmup: 0.3 → 1.0 over first 5 iterations, then constant
- Gradient clipping: `clip_materials = 1.0` (RMS clip per parameter group)

Optionally also optimize normals (`lr_normals = 0.01`) and antenna patterns (`lr_patterns = 0.05`), matching mmIR.

### Step B3: Training loop

**What**: 500-iteration training loop matching mmIR:
1. Forward pass (PyTorch rasterizer)
2. Loss computation
3. Backward pass (autograd)
4. Gradient clipping
5. Adam step
6. Post-step material clamping

**Key detail**: mmIR uses `enable_grad_phase = False` — phase is detached from the AD graph. Only amplitude carries gradients. The rasterizer must do the same.

### Step B4: Train from scratch, compare with mmIR

**What**: For each scene, initialize materials to ITU concrete defaults (same as mmIR), train for 500 iterations, evaluate cart_corr.

**Verification table**:
```
Scene               | mmIR trained | Rasterizer trained | Gap
seq_0_frame_135     | 0.9208       | ???                | < 0.05
...
```

Target: rasterizer-trained cart_corr within 0.05 of mmIR-trained for each scene. If larger gaps exist, diagnose by comparing:
- Loss curves (should have similar shape and final value)
- Per-vertex material distributions (histograms of eps_real, sigma_h, etc.)
- RA difference images

### Step B5: Train from mmIR initialization, verify no regression

**What**: Load mmIR's trained materials as initialization, run 50 more iterations with the rasterizer. Cart_corr should stay the same or improve.

This tests that the rasterizer's gradients are compatible with mmIR's learned parameters — that the optimization landscape is the same.

**Verification**: `cart_corr_after_50_iters >= cart_corr_initial - 0.01` (no regression).

---

## Stage C: Mesh vertices → Gaussian surfels

### Prerequisite state

A trainable PyTorch rasterizer on mesh vertices that matches mmIR's training quality (within 0.05 cart_corr).

### What to do

Replace the fixed mesh vertex representation with learnable Gaussian surfels. This is where the rasterizer becomes a true 3DGS system.

### Step C1: Define Gaussian surfel model

**What**: Each Gaussian stores:
| Parameter | Size | Description | Learnable? |
|-----------|------|-------------|------------|
| μ (position) | 3 | Center position | Yes |
| q (rotation) | 4 | Quaternion → normal direction | Yes |
| s₁, s₂ (scales) | 2 | Lateral extent (log-space) | Yes |
| α (opacity) | 1 | Contribution weight (logit-space) | Yes |
| materials | 6 | ITU params (raw/unconstrained space) | Yes |
| **Total** | **16** | | |

This is the same model as `mm25DGS/gaussian_model.py`.

### Step C2: Initialize Gaussians from mesh vertices

**What**: Place one Gaussian at each mesh vertex, with:
- μ = vertex position
- q = quaternion from vertex normal (same as `initialization.py:_compute_local_frames_and_scales()`)
- s₁, s₂ = from local PCA of k-NN vertices (same as current initialization)
- α = logit(0.5) = 0 (neutral)
- materials = mmIR's trained per-vertex materials (NOT random defaults)

**Why start from mmIR materials**: This verifies the Gaussian representation without conflating representation errors with optimization errors. If Gaussians at mesh vertices with mmIR materials produce the same cart_corr as the mesh-vertex rasterizer, the Gaussian representation itself is correct.

**Verification**: Render the Gaussian scene (no training) and compare cart_corr with the mesh-vertex rasterizer. Should match within 0.02 — the only difference is opacity (which is 0.5 for all Gaussians vs implicit 1.0 for mesh vertices) and scale (which doesn't affect amplitude in the current formulation).

If there's a gap > 0.02, identify the source:
- Opacity effect: try α = logit(1.0) = large positive value
- Normal quality: compare Gaussian normals vs mesh normals
- Scale effect: check if the amplitude formula inadvertently depends on scale

### Step C3: Train Gaussians from mmIR initialization

**What**: Starting from Step C2 initialization (Gaussians at mesh vertices with mmIR materials), train for 500 iterations optimizing materials, positions, rotations, scales, and opacities.

**Loss**: Same RA magnitude MSE (linear, min-max normalized).

**Optimizer**: Per-group Adam matching mmIR's LRs where applicable:
- materials: lr=0.5
- positions: lr=1.6e-4
- rotations: lr=1e-3
- scales: lr=5e-3
- opacities: lr=5e-2

**Verification**: cart_corr after training should be ≥ the initial value (no regression from mmIR's quality). Target: within 0.03 of mmIR per scene.

### Step C4: Train Gaussians from scratch (LiDAR initialization)

**What**: Initialize Gaussians from LiDAR point cloud (FPS downsampling to N_target), with ITU concrete defaults for materials. Train for 500 iterations.

This is the true test: can the Gaussian rasterizer learn the scene from scratch?

**Initialization**:
- Positions: FPS-sampled LiDAR points (same as `initialization.py`)
- Normals: from mesh vertex normals at nearest vertex (or PCA)
- Materials: ITU concrete defaults
- Opacity: logit(0.5)

**N_target choices to test**: Start with N_verts (same count as mesh) to isolate the effect of Gaussian representation. Then reduce to 25K, 50K to test compression.

**Verification table**:
```
Scene               | mmIR  | Gauss (N=mesh) | Gauss (N=50K) | Gauss (N=25K)
seq_0_frame_135     | 0.921 | ???            | ???           | ???
...
```

### Step C5: Add density control (clone/split/prune)

**What**: Adaptive Gaussian count via the standard 3DGS density control:
- **Clone**: Gaussians with large position gradients in under-reconstructed regions
- **Split**: Large Gaussians with large position gradients
- **Prune**: Gaussians with near-zero opacity

This allows starting with fewer Gaussians and growing the representation.

**Implementation**: Already exists in `mm25DGS/density_control.py`. Integrate into the training loop.

**Verification**: Start with 10K Gaussians, train with density control for 1000 iterations. Final Gaussian count and cart_corr should be competitive with the fixed-count runs.

---

## File structure (additions to mm25DGS_v2/)

```
mm25DGS_v2/
    ... (files from forward-only plan) ...
    
    # Stage A: PyTorch port
    rasterizer_torch.py       # A1-A4: PyTorch forward pass
    test_torch_equivalence.py # A5: verify torch matches drjit
    
    # Stage B: Mesh-vertex training
    loss.py                   # B1: loss function (reuse from mm25DGS/losses.py)
    train_mesh.py             # B2-B3: training loop on mesh vertices
    
    # Stage C: Gaussian training
    gaussian_model.py         # C1: Gaussian surfel model (reuse from mm25DGS)
    initialization.py         # C2: init from mesh or LiDAR
    density_control.py        # C5: clone/split/prune
    optimizer.py              # Per-group Adam
    train_gaussian.py         # C3-C4: training loop on Gaussians
    config.py                 # All hyperparameters
    
    eval/
        evaluate_scene.py     # Render + compute cart_corr for one scene
        evaluate_all.py       # Run all 7 scenes, print table
        compare_with_mmIR.py  # Side-by-side comparison with mmIR
```

---

## Verification at each stage boundary

### After Stage A (PyTorch port)
```
For each scene:
    assert |cart_corr_torch - cart_corr_drjit| < 0.01
    assert corr(ra_torch, ra_drjit) > 0.99
```

### After Stage B (mesh-vertex training)
```
For each scene:
    assert |cart_corr_rasterizer_trained - cart_corr_mmIR_trained| < 0.05
```

### After Stage C2 (Gaussians with mmIR materials, no training)
```
For each scene:
    assert |cart_corr_gaussian_init - cart_corr_mesh_rasterizer| < 0.02
```

### After Stage C3 (Gaussians trained from mmIR init)
```
For each scene:
    assert cart_corr_gaussian_trained >= cart_corr_mmIR_trained - 0.03
```

### After Stage C4 (Gaussians trained from scratch)
```
For each scene:
    report cart_corr and gap vs mmIR
    target: mean gap < 0.05
```

---

## What is NOT in this plan

- Custom CUDA kernels (pure PyTorch is sufficient for now)
- Multi-bounce (single-bounce only, matching mmIR's best ablation)
- Doppler / temporal modeling
- Cross-sensor transfer evaluation (train on cascaded, test on single-chip)
- New BSDF models beyond Jones KA+SPM
- Learned antenna patterns (use mmIR's trained patterns as fixed input in Stage C)

These are extensions that build on a verified, working trainable Gaussian rasterizer.

---

## Relationship to the forward-only plan

```
Forward-only plan (mm25DGS_v2_implementation_plan.md)
    Step 0-10: DrJit rasterizer matches mmIR on all 7 scenes
        │
        ▼
This plan (mm25DGS_v2_gaussian_training_plan.md)
    Stage A: Port DrJit → PyTorch (verified equivalent)
    Stage B: Add training on mesh vertices (matches mmIR quality)
    Stage C: Replace mesh with Gaussians (new representation, same quality target)
```

Each arrow represents a verified checkpoint. No stage begins until the previous one passes.
