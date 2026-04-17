# mm25DGS_v4 — Novel View Synthesis (NVS) Plan

This document captures the design rationale for the NVS variant of the v4
trainer. The implementation lives in `mm25DGS_v4/train_gaussian_nvs.py`,
which is a **copy** of `train_gaussian.py` so the single-frame pipeline is
not at risk while we iterate.

## 1. Problem formulation — is this *truly* NVS?

Each scene has 9 sequential cascaded radar frames captured roughly 100 ms
apart, with the cascaded radar (12 TX × 16 RX) mounted on a moving stage
that translates and slightly rotates between frames. The frames are pose-
aligned in `data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned.json`.

We split the 9 frames into:

- **train**: positions {0, 2, 4, 6, 8}  (interleaved every other)
- **test**:  positions {1, 3, 5, 7}

The split is interpolation-style (test poses sit *between* train poses).
This is novel-view synthesis only in the strict-radar sense:

- The scene geometry is **identical** across frames (rigid scene, moving
  sensor) — there is no view-conditioned material, just a learned
  radiometric model that should generalize to unseen poses.
- The radar's **wavelength** and **bandwidth** do not change between
  frames; only the sensor pose changes. The learnable parameters
  (per-point materials + per-point quaternion → normal + pose-independent
  globals) are all view-independent quantities, so train→test should
  generalize naturally if the model is well-posed.
- The frames are **temporally adjacent** so the test poses lie *inside*
  the convex hull of the train poses — this is interpolation NVS, not
  extrapolation NVS. It is the easier of the two, and it is the right
  starting point: if interpolation does not transfer, extrapolation
  certainly won't.

Caveats that make this less NVS-pure than an optical NeRF would be:

1. **Antenna pattern is shared**. The factory MMWCAS pattern is reused
   for every frame; in a vision NeRF you would also re-use the camera
   intrinsics, so this is fine.
2. **Per-point normals are learnable** (`LEARN_NORMALS=True`). Normals
   are intrinsic, not view-conditioned, so this is OK; but if the
   training frames bias the learned normals toward only the surfaces
   they see well, test-frame quality will degrade. The active matrix
   below addresses this directly.
3. **Material LRs and rotation LRs are tuned for single-frame** (mat_lr=
   0.01, rot_lr=5e-3). Multi-frame gradient accumulation increases the
   *effective* LR per parameter by ~5×. We start with the same LRs and
   note this as a known knob to revisit.

## 2. Multi-frame point selection strategy

The single-frame trainer (`init_visible_weighted` in train_gaussian.py)
builds the active point set by:

  pcl → FOV cull (single rast) → ray-test (single RX center)
       → cosine importance resample → FPS to 90K

For NVS the same set has to serve *all 9 frames*. Two failure modes drove
the design:

- **Too restrictive** (intersect FOV/visibility across frames): we'd lose
  most of the scene to per-frame edge effects, leaving a tiny shared core
  that is unrepresentative.
- **Too permissive** (full pcl every frame): blows the 90K GPU budget and
  spends compute on points outside any frame's FOV.

Chosen approach: **union FOV + union visibility, then a per-frame active
matrix to gate the renderer**.

  pcl → per-frame FOV mask + cos_bore (9 frames)
       → union FOV  (kept iff in FOV of ≥1 frame)
       → per-frame ray-test against each RX center
       → union visibility (kept iff visible from ≥1 frame)
       → cosine resample weighted by max_f cos_bore_f
       → FPS to target_n=90K (same single-GPU budget)
       → recompute per-frame (FOV ∧ visibility) on the *final* points,
         producing an active_matrix of shape (90K, 9) bool

Why this works:

1. **Coverage**: the union ensures every frame's renderable surfaces are
   represented. A point that only frame 7 sees still survives.
2. **Budget**: total point count stays at 90K — same memory ceiling as
   the single-frame trainer (verified to fit in 19.5 GB on a 4090).
3. **Per-frame gating**: at render time, frame f only computes BSDF on
   `active_matrix[:, f]`, so per-frame compute is bounded by single-frame
   compute. The (90K, 9) bool matrix is ~810 KB, trivial.
4. **Importance sampling stays correct**: max_f cos_bore_f is the right
   weight because a point should be sampled densely if *any* frame
   prefers it. Min would penalize points seen by only one frame.

Per-frame active counts will vary — frames at the ends of the trajectory
may activate ~50-70 % of points, central frames more — but this is
expected and correct.

The **test frames {1,3,5,7}** are *not* used during init. Their active
masks are computed on-the-fly at evaluation time using the same FOV +
ray-test logic. This keeps train/test fully separated.

## 3. Script setup

Layout:

- `train_gaussian.py` — UNTOUCHED single-frame trainer. Continues to be
  the baseline for `output/<scene>/`.
- `train_gaussian_nvs.py` — copy with NVS additions:
  - `load_nvs_scene(scene)` — reads 9 aligned configs + ADC files
  - `init_visible_weighted_nvs(scene, rasts, target_n)` — multi-frame
    init returning `(model, active_matrix)`
  - `train_nvs(scene, train_indices, test_indices, num_iters, ...)` —
    multi-frame training loop with per-iter loss accumulated across all
    training frames, one optimizer step per iter
  - `evaluate_all_frames(...)` — renders every frame at the best model
    state, saves per-frame PNGs and metrics
  - CLI: `--scene`, `--all`, `--iters`, `--target_n`, `--train_indices`,
    `--test_indices`
- Output directory: `mm25DGS_v4/output_nvs/<scene>/` so single-frame
  results in `mm25DGS_v4/output/<scene>/` are not clobbered.

Training loop semantics:

```
for iter in range(num_iters):
    optimizer.zero_grad()
    total_loss = 0
    for f in train_indices:
        rp_real, rp_imag = render(model, rasts[f], active_matrix[:, f])
        loss_f = compute_ra_loss_rp(rp_real, rp_imag, gt_cached[f])
        total_loss = total_loss + loss_f
    total_loss.backward()
    optimizer.step()
```

Best-state tracking is by **mean cart_corr on the training frames** (we
should not peek at the test frames during training). At the end we render
*all 9* frames once at the best state and report train-mean and
test-mean separately.
