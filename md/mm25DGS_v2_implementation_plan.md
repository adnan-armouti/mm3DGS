# mm25DGS_v2: Ray Tracer → Rasterizer Conversion Plan

**Directory**: `/home/adnan/Desktop/mm3DGS/mm25DGS_v2/`

## Goal

Build a forward-only rasterizer that produces **numerically identical** output to mmIR's ray tracer when given the same scene, materials, normals, and antenna patterns. Training comes later — this plan is exclusively about the forward model.

## Philosophy

Convert the mmIR ray tracer into a rasterizer **one building block at a time**, verifying numerical equivalence at every step. The rasterizer starts as a literal copy of the ray tracing code. Each step replaces ONE component while keeping everything else identical.

**NO new physics. NO new loss functions. NO new BSDF models.** Only structural changes to eliminate ray tracing while preserving identical numerical output.

## Trained Parameters (Fixed Inputs)

All 7 scenes have trained parameters saved at `/home/adnan/Desktop/mmIR/output/train_v13/`:

| Scene | N_verts | cart_corr | Materials | Normals | Patterns |
|-------|---------|-----------|-----------|---------|----------|
| seq_0_frame_135 | 50,499 | 0.9208 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |
| seq_0_frame_390 | 107,353 | 0.9363 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |
| seq_1_frame_185 | 102,451 | 0.9541 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |
| seq_1_frame_438 | 72,934 | 0.9422 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |
| seq_2_frame_105 | 113,075 | 0.8867 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |
| seq_2_frame_160 | 87,016 | 0.8638 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |
| seq_2_frame_300 | 86,966 | 0.9301 | `best_materials.npz` | `best_normals.npz` | `best_patterns.npz` |

**Per scene, we load:**
- `best_materials.npz` → `raw_params` (N_verts, 6) — per-vertex material parameters in unconstrained optimizer space
- `best_normals.npz` → `normal_params` (N_verts, 3) — learned normal offsets
- `best_patterns.npz` → `tx_E_plane, tx_H_plane, rx_E_plane, rx_H_plane` — learned antenna pattern planes
- `config.json` → training config (includes `scene_file`, `config_file` paths, all hyperparameters)

**Per scene, we also have gold reference outputs:**
- `ra_rendered_cart.npy` — (399, 399) mmIR rendered Cartesian RA image
- `ra_gt_cart.npy` — (399, 399) GT Cartesian RA image
- `best_metrics.json` → `cart_corr` value (the target to match)

**Data paths** (meshes and radar configs):
- Mesh: `/home/adnan/Desktop/mm3DGS/data/{scene}/scene/mesh.ply`
- Radar config: `/home/adnan/Desktop/mm3DGS/data/{scene}/configs/cascaded_frame_{N}_aligned_gpu.json`
- GT ADC: `/home/adnan/Desktop/mm3DGS/data/{scene}/radar/cascaded_frame_{N}.npy`
- Antenna patterns (base): `/home/adnan/Desktop/mmIR/assets/antenna_pattern/MMWCAS/tx1_76.npy`, `rx1_76.npy`

---

## mmIR Pipeline (Reference)

```
Reservoir Sampling → MIMO Expansion → Normal Flip → Shadow Rays → Material Gather
→ BSDF Eval → Antenna Gain → MC Correction → Radar Equation → Phase → Scatter-Add → ADC
```

**Key files** (all under `mmir/renderer/`):
| Stage | File | Function | Lines |
|-------|------|----------|-------|
| Ray cast | `sampler.py` | `sample_reservoir_drjit()` | 163-428 |
| MIMO expand | `integrator/synthesis_e2e.py` | inline | 200-248 |
| Normal flip | `integrator/synthesis_e2e.py` | inline | 278-289 |
| Shadow rays | `integrator/synthesis_e2e.py` | `scene.ray_test()` | 296-321 |
| Material gather | `materials/parameterization.py` | `gather()` | 81-126 |
| BSDF | `bsdf/mmwave_jones.py` | `eval_f_cos_physics()` | 940-946 |
| Antenna | `sensor/element_patterns.py` | `evaluate_combined_gain()` | 318-360 |
| MC correction | `integrator/synthesis_e2e.py` | inline | 468-476 |
| Radar equation | `integrator/synthesis_e2e.py` | inline | 495-519 |
| Phase+scatter | `integrator/synthesis_e2e.py` | inline | 521-564 |

---

## Conversion Steps

### Step 0: Reproduce mmIR forward pass and establish gold reference

**What**: Create `mm25DGS_v2/` with a script that calls mmIR's forward pass directly, loading the trained parameters for each scene. This is NOT a new renderer — it IS mmIR. The purpose is to:
1. Confirm we can load all trained parameters correctly
2. Reproduce the published cart_corr values exactly
3. Save intermediate quantities (per-vertex weights, BSDF values, etc.) as gold references for later steps

**Files to create**:
- `mm25DGS_v2/__init__.py`
- `mm25DGS_v2/render_mmIR.py` — Loads trained params (materials, normals, patterns), calls mmIR `synthesize_end_to_end()`, returns ADC
- `mm25DGS_v2/test_harness.py` — For each scene: loads GT, renders via `render_mmIR.py`, computes cart_corr, compares against saved `best_metrics.json`

**Verification**: For each of the 7 scenes:
```
scene               | mmIR cart_corr | Our reproduction | Match?
seq_0_frame_135     | 0.9208         | ???              | must be within 0.01
seq_0_frame_390     | 0.9363         | ???              | must be within 0.01
...
```

Also cross-check against saved `ra_rendered_cart.npy`: `corr(our_ra_cart, saved_ra_cart) > 0.99`.

**Critical**: mmIR's training-time forward pass uses a SPECIFIC code path with `pattern_loaders` (differentiable antenna) and a specific random seed for reservoir sampling. We must reproduce that exact path. The `RendererWrapper.render_forward()` uses a DIFFERENT code path (non-diff antenna, different seed) and gives different results. Study the training code in `train.py` lines 1544-1581 (`run_end_to_end_forward`) to replicate the exact call sequence.

**What to save** (per scene, in `mm25DGS_v2/gold_references/{scene}/`):
- `adc_ri.npy` — (N_tx, N_rx, K, 2) rendered ADC
- `ra_polar.npy` — polar RA image
- `ra_cart.npy` — Cartesian RA image  
- `cart_corr.txt` — scalar correlation value
- `per_path_weights.npy` — (n_total,) weight per path (for later comparison)
- `per_path_bsdf.npy` — (n_total,) BSDF output per path
- `per_path_antenna.npy` — (n_total,) antenna gain per path
- `active_mask.npy` — (n_total,) bool visibility mask after shadow rays

---

### Step 1: Replace reservoir sampling with deterministic vertex enumeration

**What**: Instead of casting random rays and collecting hit points, directly use ALL mesh vertices as scatterers. Everything downstream (BSDF, antenna, phase) stays as mmIR code.

**mmIR original**: `sampler.py:sample_reservoir_drjit()` → casts random rays, collects ~1500 hits per RX. Output: `ReservoirHitsDrJit` with `hit_P, hit_N, hit_prim_ids, hit_bary_u/v, hit_pdf, n_attempted`.

**Rasterizer replacement**: Build a `ReservoirHitsDrJit`-compatible object from mesh vertices:
- `hit_P` = vertex positions (loaded from mesh, with learned normal offsets applied)
- `hit_N` = vertex normals (from `best_normals.npz`)
- `hit_pdf` = 1.0 for all (uniform enumeration)
- `n_attempted` = N_vertices

**Critical structural change**: mmIR associates each hit with ONE RX element. The MIMO expansion then replicates for all TX. In the rasterizer, each vertex must be associated with ALL RX elements. The MIMO expansion creates `N_verts × N_tx × N_rx` paths (vs mmIR's `n_hits × N_tx`).

**Implementation**:
```python
# mm25DGS_v2/vertex_sampler.py
def enumerate_vertices(mesh_path, training_dir):
    """Replace reservoir sampling with all-vertex enumeration.
    
    Loads mesh vertices and applies learned normals from training_dir.
    Returns data in the same format as ReservoirHitsDrJit.
    """
```

**Verification**: This step will produce DIFFERENT ADC from mmIR (different sampling). Compare:
1. Per-vertex BSDF values (using gold reference from Step 0 at matching vertices) — should match for vertices that appear in both
2. RA image structure — visually similar pattern, correlation with GT > 0.7

---

### Step 2: Replace MIMO expansion with explicit (vertex, TX, RX) geometry

**What**: Instead of mmIR's index-based MIMO replication, directly compute distances and directions for all (vertex, TX, RX) combinations.

**mmIR original**: `synthesis_e2e.py:200-248` — Expands hit arrays by N_tx factor using `dr.gather` with index arithmetic.

**Rasterizer replacement**:
```python
# d_tx[v, t] = ||vertex_v - tx_t||         (N_verts, N_tx)
# d_rx[v, r] = ||vertex_v - rx_r||         (N_verts, N_rx)
# dir_hit_to_tx[v, t, :] = (tx_t - vertex_v) / d_tx[v,t]    (N_verts, N_tx, 3)
# dir_hit_to_rx[v, r, :] = (rx_r - vertex_v) / d_rx[v,r]    (N_verts, N_rx, 3)
```

**Verification**: Pick 100 random vertices. For each, compute distances to all TX/RX. Compare with mmIR's expanded distances for the same vertex — must match to float precision.

---

### Step 3: Replace shadow rays with mesh-based visibility

**What**: mmIR uses Mitsuba's `scene.ray_test()` for per-path occlusion. The rasterizer pre-computes visibility per vertex using trimesh.

**mmIR original**: `synthesis_e2e.py:296-321` — For each expanded path, casts shadow ray from hit toward TX. Also filters grazing angles (`cos_theta_in > 1e-6` and `cos_theta_out > 1e-6`).

**Rasterizer replacement**:
```python
# mm25DGS_v2/visibility.py
def compute_visibility(mesh, vertices, tx_positions, epsilon=1e-4):
    """For each (vertex, TX) pair, test line-of-sight.
    
    Since TX elements are within ~10cm, we can test against radar
    center as proxy (one ray per vertex instead of N_tx).
    Also applies grazing-angle filter matching mmIR's cos_theta > 1e-6.
    
    Returns: bool array (N_verts,)
    """
```

**Verification**:
1. Extract mmIR's active mask from Step 0 gold reference
2. Map mmIR hits to nearest vertices
3. Compare visibility: agreement rate should be > 90%
4. Diff the RA images with and without visibility — visibility should improve correlation

---

### Step 4: Per-vertex material loading (direct, no interpolation)

**What**: Load `best_materials.npz` per-vertex parameters directly. No barycentric interpolation needed since vertices ARE the scatterers.

**mmIR original**: `materials/parameterization.py:gather()` — Per-vertex gather using vertex IDs from triangle barycentric coords: `w*p[v0] + u*p[v1] + v*p[v2]`.

**Rasterizer**: Direct indexing — `raw_params[vertex_idx]`. Then reparameterize to physics space using the SAME `reparameterize_physics_params_drjit()` from mmIR.

**Verification**: For vertices that mmIR samples (from Step 0), the material values must be identical (same raw_params, same reparameterization). Check for 100 random vertices.

---

### Step 5: BSDF evaluation — IDENTICAL mmIR code

**What**: Call `mmwave_jones.py:eval_f_cos_physics()` with the SAME DrJit function, SAME direction conventions, SAME material inputs.

No changes. Import directly from mmIR.

**Verification**: Compare BSDF output for 1000 vertices against Step 0 `per_path_bsdf.npy`. Must be bit-for-bit identical for matching (vertex, TX, RX) tuples.

---

### Step 6: Antenna gain — IDENTICAL mmIR code

**What**: Call `evaluate_combined_gain()` with the SAME DrJit function, loading the SAME learned patterns from `best_patterns.npz`.

Direction convention (matching mmIR training path, Path 1):
- TX: `evaluate_combined_gain(tx_loader, dir_hit_to_tx, tx_bore)` — direction FROM vertex TOWARD TX
- RX: `evaluate_combined_gain(rx_loader, dir_rx_to_hit, rx_bore)` — direction FROM RX TOWARD vertex

**Verification**: Compare against Step 0 `per_path_antenna.npy`. Bit-for-bit identical.

---

### Step 7: MC correction → vertex area weighting

**What**: Replace mmIR's `1/(pdf × n_attempted)` with vertex-area-based normalization.

**mmIR original**: `mc_correction = 1 / (pdf × n_attempted)` where `pdf ∝ cos(θ)/π`.

**Rasterizer**: Each vertex represents a surface patch. Weight by Voronoi area:
```python
vertex_areas = trimesh_mesh.area_faces  # → compute per-vertex area
mc_correction[v] = vertex_areas[v]  # area of surface patch this vertex represents
```

Under min-max normalization, the absolute scale cancels — only RELATIVE area weighting matters. Larger surface patches get proportionally more weight.

**Verification**: This is the step most likely to cause quantitative divergence from mmIR. Compare:
1. RA image with uniform MC (all vertices equal) vs area-weighted
2. Area-weighted should be closer to mmIR's RA image
3. Target: cart_corr with GT within 0.05 of mmIR for each scene

---

### Step 8: Radar equation — IDENTICAL mmIR code

**What**: Same `radar_constant`, `path_loss = 1/d_tx²`, `weight = radar_scale × sqrt(brdf_weight × path_loss)`.

Import constants and formula directly from mmIR's `synthesis_e2e.py` and `core.py`.

**Verification**: Same inputs → same outputs.

---

### Step 9: Phase + phasor scatter — IDENTICAL mmIR code

**What**: Same phase formula, same DrJit scatter-add.

```
R_total = d_tx + d_rx
tau = R_total / c
phi = 2π × f0 × tau + 2π × S × tau × t_k
adc[tx, rx, k] += weight × cos(phi), weight × sin(phi)
```

**Verification**: Given identical weights and distances, ADC output is bit-for-bit identical.

---

### Step 10: End-to-end verification on ALL 7 scenes

**What**: Run the complete rasterizer pipeline on all 7 scenes, loading trained parameters from `train_v13/`. Compare against both:
1. mmIR gold reference (Step 0) — measures rasterizer-vs-raytracer fidelity
2. GT — measures absolute reconstruction quality

**Verification table** (THE deliverable of this plan):

```
Scene               | mmIR   | Rasterizer | Rast vs mmIR corr | Target
seq_0_frame_135     | 0.9208 | ???        | ???               | > 0.90
seq_0_frame_390     | 0.9363 | ???        | ???               | > 0.90
seq_1_frame_185     | 0.9541 | ???        | ???               | > 0.90
seq_1_frame_438     | 0.9422 | ???        | ???               | > 0.90
seq_2_frame_105     | 0.8867 | ???        | ???               | > 0.85
seq_2_frame_160     | 0.8638 | ???        | ???               | > 0.85
seq_2_frame_300     | 0.9301 | ???        | ???               | > 0.90
```

"Rast vs mmIR corr" = correlation between the rasterizer's RA image and mmIR's RA image (NOT GT). This isolates how faithful the rasterizer is to the ray tracer, independent of GT quality.

**Debugging protocol**: If any scene fails the target:
1. Compare per-vertex weight histograms (rasterizer vs mmIR)
2. Visualize RA image difference map
3. Check which stage introduces the divergence by comparing intermediate outputs
4. Likely culprits: MC correction (Step 7) or visibility (Step 3)

---

### Step 11: Port DrJit → PyTorch (future, not in this plan's scope)

Once Step 10 passes for all 7 scenes, the rasterizer is verified correct. The DrJit→PyTorch port follows the same one-at-a-time methodology, comparing against the verified DrJit rasterizer at each sub-step.

---

## File Structure

```
mm25DGS_v2/
    __init__.py
    render_mmIR.py           # Step 0: call mmIR forward pass, save gold references
    test_harness.py          # Verification: load gold ref, render, compare
    vertex_sampler.py        # Step 1: vertex enumeration
    mimo_geometry.py         # Step 2: (vertex, TX, RX) distance/direction computation
    visibility.py            # Step 3: mesh-based shadow testing
    mc_correction.py         # Step 7: vertex-area weighting
    rasterizer.py            # Steps 1-9 assembled: full rasterizer using mmIR internals
    run_all_scenes.py        # Step 10: run all 7 scenes, print verification table
    
    gold_references/         # Saved by Step 0, read by all later steps
        seq_0_frame_135/
            adc_ri.npy
            ra_cart.npy
            cart_corr.txt
            per_path_weights.npy
            per_path_bsdf.npy
            per_path_antenna.npy
            active_mask.npy
        seq_0_frame_390/
            ...
        (7 scenes total)
    
    tests/
        test_step00_reproduce_mmIR.py
        test_step01_vertices.py
        test_step02_mimo.py
        test_step03_visibility.py
        test_step04_materials.py
        test_step05_bsdf.py
        test_step06_antenna.py
        test_step07_mc.py
        test_step08_radar_eq.py
        test_step09_phase.py
        test_step10_e2e.py
```

---

## Verification Protocol

Every step follows this protocol:

1. **Isolate**: Change ONLY the component under test. Everything else is literal mmIR code.
2. **Same inputs**: Load the EXACT trained parameters from `train_v13/{scene}/`.
3. **Compare outputs**: Against Step 0 gold reference for that scene.
4. **Tolerance**:
   - Bit-for-bit for unchanged components (Steps 5, 6, 8, 9)
   - `< 1e-4` relative error for structural changes with same underlying math (Steps 2, 4)
   - cart_corr with GT within 0.05 of mmIR for pipeline-level changes (Steps 1, 3, 7)
5. **All 7 scenes**: Every step must pass on ALL 7 scenes, not just one. Scene-specific failures indicate geometry-dependent bugs.
6. **Save artifacts**: Each test saves results to `mm25DGS_v2/tests/results/step_XX/`

---

## Dependencies

The rasterizer uses mmIR code directly for:
- `mitsuba` + `drjit` — array types, scene loading
- `mmir.renderer.bsdf.mmwave_jones` — BSDF evaluation (Step 5)
- `mmir.sensor.element_patterns` — antenna gain (Step 6)
- `mmir.renderer.bsdf.reparameterization` — material reparameterization (Step 4)
- `mmir.renderer.integrator.core` — radar constants (Step 8)
- `mmir.data.ra_utils` — ADC→RA conversion
- `mmir.data.data_utils` — GT loading
- `trimesh` — mesh loading, vertex areas, ray casting (Step 3)

---

## What is NOT in this plan

- Training / optimization (no loss function, no gradients, no optimizer)
- PyTorch port (comes after verification)
- New BSDF models
- Custom CUDA kernels
- Multi-bounce
- Doppler
- Density control
- Gaussians (this plan operates on mesh vertices, not Gaussians)

The path from verified rasterizer → trainable Gaussian rasterizer is a separate plan that builds on this one.
