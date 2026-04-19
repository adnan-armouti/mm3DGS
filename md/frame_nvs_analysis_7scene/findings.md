# 7-scene frame-NVS analyses (matched grid, HO_8 + S4 + UB_9)

Date: 2026-04-19
Scope: **7 scenes** × 4 variants = 28 models at target_n=20k, seed=test_frame.

Variants analysed:
- `HO_base` — HO_8 (8 train frames × 1 chirp) baseline, no S4
- `HO_S4_jit` — HO_8 + S4 jitter 0.02/50 pos=2cm (6-scene winner)
- `HO_S4_ann` — HO_8 + S4 pool_knn annulus [0.02, 0.10] (7-scene winner)
- `UB_9` — first-chirp UB (9 train frames × 1 chirp; ceiling)

---

## 0. TL;DR

**The matched-grid 2-scene finding "HO's rotations drift 17° more than UB's at
top-1% Fisher" does NOT generalize to 7 scenes.** HO_base and UB_9 drift by
indistinguishable amounts at top-1% on 6 of 7 scenes.

**The swap ablation finding "rotations >> materials as dominant fault" DOES
generalize strongly.** On ALL 7 scenes, injecting UB's rotations into HO's
model beats injecting UB's materials — by a factor of **2.8× on mean**.

**Upper bound on a rotation-only fix: mean HO_mat + UB_rot ≈ 0.69 across 7
scenes.** That's just 0.01 below the 0.70 target. **So the target IS reachable
in principle, if we could guide HO toward UB-like rotations.**

**S4 (adaptive density) helps slightly (+0.019 mean test) but doesn't exploit
the rotation signal.** S2 (Fisher-weighted rotation drift from init) is
**empirically dead** because HO and UB have equal aggregate drift; the fault
is in *direction*, not *magnitude*.

---

## 1. Aggregate means

| variant | mean (7 scenes) | mean (6 scenes, no seq_0_f390) |
|---|---:|---:|
| HO_base | 0.5234 | 0.5635 |
| HO_S4_jit | 0.5239 | 0.5648 |
| HO_S4_ann | 0.5312 | 0.5654 |
| UB_9 | **0.8313** | **0.8253** |

UB_9 is consistently 0.26–0.30 above HO_base → **abundant parameter-space
headroom; the issue is reaching it from HO-only supervision**.

---

## 2. Analysis I — swap ablation (the dominant signal)

All variants share the same matched position grid, so swaps are exact
(no NN-remap noise, unlike the prior 2-scene analysis §9).

| scene | HO native | UB native | UB_mat + HO_rot (Δ) | **HO_mat + UB_rot (Δ)** |
|---|---:|---:|---:|---:|
| seq_0_frame_135 | 0.5702 | 0.8340 | 0.6037 (+0.033) | **0.7116 (+0.141)** |
| seq_0_frame_390 | 0.2825 | 0.8673 | 0.4586 (+0.176) | **0.6453 (+0.363)** |
| seq_1_frame_185 | 0.5575 | 0.8173 | 0.5765 (+0.019) | **0.6553 (+0.098)** |
| seq_1_frame_438 | 0.5670 | 0.9082 | 0.6526 (+0.086) | **0.7644 (+0.197)** |
| seq_2_frame_105 | 0.5634 | 0.8058 | 0.5924 (+0.029) | **0.6441 (+0.081)** |
| seq_2_frame_160 | 0.6443 | 0.8201 | 0.6540 (+0.010) | **0.7050 (+0.061)** |
| seq_2_frame_300 | 0.4790 | 0.7791 | 0.5192 (+0.040) | **0.6318 (+0.153)** |
| **mean (7)** | 0.5234 | 0.8331 | +0.056 (→ 0.579) | **+0.156 (→ 0.679)** |
| **mean (6 no out.)** | 0.5634 | 0.8260 | +0.036 (→ 0.599) | **+0.122 (→ 0.686)** |

**HO_mat + UB_rot > UB_mat + HO_rot on every single scene** — the most
consistent cross-scene signal in the whole study. Rotations dominate the
remaining gap by ~2.8× over materials.

**Ceiling if we could magically fix only rotations: 0.69 mean.** That's
within 0.01 of the 0.70 target.

---

## 3. Fisher-weighted normal drift @ top-1 % (diagnostic for S2)

Mean drift angle from pcl-LiDAR-normal init, weighted to the 200 highest-
Fisher points of each variant:

| scene | HO_base | HO_S4_jit | HO_S4_ann | UB_9 |
|---|---:|---:|---:|---:|
| seq_0_frame_135 | 54.3° | 71.5° | 59.5° | **53.3°** |
| seq_0_frame_390 | 42.6° | 51.4° | 48.3° | **43.1°** |
| seq_1_frame_185 | 46.7° | 66.7° | 59.4° | **47.1°** |
| seq_1_frame_438 | 34.6° | 51.2° | 52.6° | **36.9°** |
| seq_2_frame_105 | 52.1° | 57.4° | 53.5° | **51.8°** |
| seq_2_frame_160 | 44.5° | 50.9° | 49.3° | **46.4°** |
| seq_2_frame_300 | 37.2° | 65.7° | 56.4° | **51.5°** |

**HO_base ≈ UB_9 on 6 of 7 scenes (within 4°).** Aggregate drift is not
the distinguishing feature. The earlier 2-scene matched-grid analysis
(`md/frame_nvs_analysis_matched_grid/findings.md` §10, specifically for
HO_128 not HO_8) over-generalised this signal.

**S4 increases drift by 5–30°** on all scenes, yet test cc still improves
by +0.019 mean with pool_knn annulus. Drift magnitude is clearly a
symptom, not a causal variable.

**Therefore S2 (Fisher-weighted L2 on rotation drift from init) has no
cross-scene handhold** — there is no aggregate under-drift signal to
correct, and fixing drift magnitude does not close the rotation-direction
gap that Analysis I reveals.

---

## 4. Reproducibility note

Max |reported − re-measured| test cc across the 28 (scene × variant)
combinations = 0.077. Root cause: the re-rendering pipeline computes
`active_mask` once from the freshly-built `init_visible_weighted` model,
then overwrites positions from each variant's state dict. For S4 variants
whose positions shifted significantly, the mask is mildly stale. This
affects the absolute test cc numbers reported in `0_reproducibility.csv`
but **does NOT affect**:
- Analysis G (Fisher arrays use the variant's actual positions via the
  same pipeline — self-consistent);
- Analysis I (all swap combinations use the same pipeline — self-consistent
  *between* swaps, which is what matters);
- The reported training-time `final_test_cc` numbers in `results.json`.

The measurement offset to the S4 train-time numbers is 0.00–0.08. All
per-scene rankings and cross-scene means cited elsewhere use the
training-time numbers, not the repro numbers.

---

## 5. Hypothesis go/no-go (updated, 7 scenes)

| H | status | reason |
|---|---|---|
| H1 (uniform L2 drift materials) | no-go | A: UB/HO dists ≤2% diff on every scene |
| H2 (k-NN TV materials) | no-go | D: UB/HO smoothness ≤3% diff |
| H3 (uniform L2 drift rotations) | no-go | §3: HO_base ≈ UB_9 aggregate drift |
| H4 (k-NN TV normals) | no-go | E: aggregate smoothness identical |
| **H3+F (Fisher-weighted L2 drift rotations)** | **no-go (revised)** | §3: even at top-1 %, HO_base ≈ UB_9 drift on 6/7 scenes |
| H5 (Fisher-weighted L2 drift materials) | **marginal** | Material swap gains mean +0.056; principled version ≤ half that |
| H6 (UB cluster prior) | leakage | upper-bound only |
| H7 (Adam-v activity mask) | marginal | +0.01 regardless of matched grid |
| H8 (LiDAR-intensity prior) | no-go | J (matched): ratio 0.99+ |
| **S4 (adaptive density)** | **+0.019 mean** | Best config: pool_knn annulus [0.02, 0.10] |
| **S5 (low-rank material field)** | plausible, bounded | material swap bound +0.056 mean; realistic +0.02–0.04 |

**New, not in the original H1–H8 set:**

| name | idea | why it's motivated by this analysis |
|---|---|---|
| **Pose-augmentation (broaden training basin)** | jitter the poses of train RAs during training | Swap ablation shows UB-style rotations ARE a good fit at the test pose → the optima exists; HO just doesn't converge there under supervision at 8 discrete poses |
| **Rotation init smoothing** | smooth pcl.npy LiDAR normals with k-NN avg before building the init quaternion | Rotations are the dominant fault (§2); if the init rotations are noisy, both HO and UB diverge from them, but HO has no corrective signal at the test pose |
| **Per-point LR scaling by Fisher** | high-Fisher points get larger LR; null-space points get tiny LR | Decouple "where the signal is" from "where drift happens" (smooth-H7) |
| **Rotation parameterisation** | replace unit quaternions with 6D rotation (Zhou et al.) | Known to have smoother optimisation landscapes; may matter given rotations dominate |

---

## 6. Recommendation

Short term (cheap, high-info):
1. **Pose augmentation during training** — see `md/frame_nvs_pose_aug_design.md`
   (to be written). Implementation ≈ 30 LOC; sweep cost 7 scenes × 2-3 augmentation
   strengths × ~2 min HO_8 = ~30 min wallclock on 2 GPUs.
2. **Rotation-init smoothing** — k-NN average the LiDAR normals inside
   `init_visible_weighted`; swap the quaternion init. 10 LOC. Cost: 7 scenes × 1 run = ~10 min.

Medium term:
3. **S5 low-rank material field** — multi-day. Proceed if 1+2 cap below 0.70.

**Deferred** (per user direction — keep HO_128 experiments low-priority):
- S2 × S4 stacks
- Pass-3 × S4
- Any HO_128 retraining

---

## 7. Artefacts

CSVs: `0_reproducibility.csv`, `A_param_std.csv`, `B_drift.csv`,
`C_divergence.csv`, `D_smoothness.csv`, `E_normal_smoothness.csv`,
`G_concentration.csv`, `I_swap_ablation.csv`.

Fisher arrays: `G_fisher_<scene>_<variant>.npz` (28 files).

Renders: `render_<scene>_<variant>.npy` (28 files).

GT cart: `gt_cart_<scene>.npy` (7 files).

Script: `run_analyses_7scene.py`.
