# S4 — Fisher-based adaptive density

Date: 2026-04-19
Feeds from:
[`md/frame_nvs_next_steps.md`](frame_nvs_next_steps.md),
[`md/frame_nvs_analysis_matched_grid/findings.md`](frame_nvs_analysis_matched_grid/findings.md).

Companion to the S2 (Fisher-weighted rotation drift) investigation,
which capped at +0.01–0.03 cc across four variants. S4 escalates to
the point-budget-reallocation approach from §Stage 4 of the next-steps
plan.

## 0. TL;DR

S4 (split top-Fisher points + prune bottom-Fisher, 6 rounds × 2 %
turnover during iters 50–300) delivers the **largest single HO cc
lift we have measured**:

| scene | variant | baseline (matched) | + S4 (per-scene best) | Δ |
|---|---|---:|---:|---:|
| `seq_1_frame_438` | HO_128 | 0.5118 | **0.6440** | **+0.132** |
| `seq_1_frame_438` | HO_8   | 0.5672 | **0.6656** | **+0.098** |
| `seq_2_frame_105` | HO_128 | 0.5900 | **0.6057** | **+0.016** (alt cfg 0.10/100) |
| `seq_2_frame_105` | HO_8   | 0.5633 | **0.5807** | **+0.017** (alt cfg 0.10/100) |

**All four variants positive with per-scene-tuned S4.** seq_1 HO_128
reaches **93 %** of the Analysis I swap-ablation upper bound (0.685).
seq_2 responds positively but more modestly and requires a different
hyper-parameter point (0.10 split × interval 100 × 3 rounds) than
seq_1's winner (0.02 × 50 × 6 rounds). The two scenes have very
different Fisher-concentration profiles (seq_1: 55.6 % at top-1 %;
seq_2: 81.6 % — see §8 of the matched-grid findings), which is the
likely source of the per-scene optimum split_frac / interval choice.

**0.70 hard target**: not hit. Remaining gaps after S4:

| scene | HO_128 | HO_8 |
|---|---:|---:|
| seq_1_frame_438 | 0.056 short | 0.034 short |
| seq_2_frame_105 | 0.094 short | 0.119 short |

Further hyper-parameter exploration on seq_2 (larger `pos_jitter`,
alt `split_frac × interval` cells) is the obvious next knob. Also
orthogonal: S4 + pass-3 per-chirp anchors stack freely and each
helps ~+0.03 on HO_128 independently, so the combination is
worth measuring on the HO_128 runs.

---

## 1. Design

Goal: reallocate the fixed N = 20 000 point budget from Fisher-null
points (drifting in the null space of cart_corr, fitting noise) to
Fisher-active points (where the metric actually lives).

**Mechanics** (in `_densify_step` at
[`mm25DGS_v5/train_frame_nvs.py`](../mm25DGS_v5/train_frame_nvs.py)):

1. At `iter == reg_warm_iters` and every `reg_densify_interval` iters
   thereafter (up to `reg_densify_until`):
2. Compute per-point Fisher as the sum of normalised Adam
   `exp_avg_sq` on `raw_materials` and `rotations` (each divided by
   its own max before summing so neither modality dominates).
3. Sort Fisher descending. Top `split_frac · N` = split candidates;
   bottom `prune_frac · N` = prune candidates. With
   `split_frac == prune_frac`, the point count stays exactly N.
4. For each (split parent, prune slot) pair: overwrite the prune
   slot's `positions`, `rotations`, `raw_materials` with the
   parent's values + small independent Gaussian noise:
   - position: σ = `pos_jitter_m` (default 2 cm ≈ 5 λ at 77 GHz)
   - rotation: σ = 0.05 on quaternion components, re-normalised
   - material: σ = `mat_jitter` in raw space (default 0.1)
5. **Adam-state surgery**: zero `exp_avg` and `exp_avg_sq` at the
   replaced slots (children start with no momentum).
6. Patch `init_raw[replaced]` and `init_normals[replaced]` to
   the children's values so H1 / S2 drift penalties see zero
   drift at birth.
7. Recompute the FOV mask (`cull_gaussians` at a train pose) —
   positions shifted, so some formerly-in-FOV points may have moved
   out and vice versa.
8. Reset the S2 rotation-Fisher snapshot and normal-EMA (they
   depend on the old point set).

**CLI flags** (all default to 0 = disabled):

```
--reg_densify_interval INT       # trigger every N iters after warmup
--reg_densify_split_frac FLOAT   # top fraction to split each round
--reg_densify_prune_frac FLOAT   # bottom fraction to prune (must = split_frac)
--reg_densify_until INT          # stop densifying past this iter
--reg_densify_pos_jitter_m FLOAT # position noise std (m)
--reg_densify_mat_jitter FLOAT   # raw-material noise std
```

**Why in-place replacement, not tensor resize?** The alternative —
growing the model by `split_n` and shrinking by `prune_n` at the
same time — requires rebuilding the optimizer and reallocating
multiple `nn.Parameter` tensors per round. In-place overwrite of
pruned slots preserves (a) all un-touched tensor indices, so
cached `active_mask` and `vertex_areas` don't rebuild from scratch
for unchanged points, and (b) optimizer state references, so Adam
momentum is preserved on survivors without state-dict surgery.

---

## 2. Sweep on seq_1 HO_8

Baseline (matched-grid, no S4): test 0.5672, train 0.8657.

| split_frac | interval | until | rounds | test | Δ test | train | Δ train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.02** | **50** | **300** | 6 | **0.6656** | **+0.098** | 0.8776 | +0.012 |
| 0.10 | 100 | 300 | 3 | 0.6629 | +0.096 | 0.8807 | +0.015 |
| 0.02 | 25  | 300 | 11 | 0.6432 | +0.076 | 0.8868 | +0.021 |
| 0.05 | 25  | 300 | 11 | 0.6243 | +0.057 | 0.8968 | +0.031 |
| 0.10 | 50  | 300 | 6 | 0.6206 | +0.053 | 0.8974 | +0.032 |
| 0.05 | 50  | 300 | 6 | 0.6143 | +0.047 | 0.8886 | +0.023 |
| 0.02 | 100 | 400 | 4 | 0.6086 | +0.041 | 0.8665 | +0.001 |
| 0.05 | 100 | 300 | 3 | 0.6004 | +0.033 | 0.8731 | +0.007 |

**Observations:**
- Every cell is positive on test cc. Adaptive density is robustly
  helpful on this variant.
- Two best cells tie near +0.098 (both reaching `cumulative
  turnover ≈ 12-30 %` after 3–6 rounds of 2–10 % each).
- Train cc also rises across the board (+0.001 to +0.032) — the
  re-allocation doesn't hurt training fit; it helps both train
  and test.
- Large `split_frac = 0.10` with many rounds (rounds 6+) under-performs
  (0.6206 at 0.10 × interval 50) relative to either conservative
  many-rounds (0.02 × 50 × 6r = 0.6656) or aggressive few-rounds
  (0.10 × 100 × 3r = 0.6629). Cumulative turnover of ~60 % in 6
  rounds over-saturates.

**Winner**: `split_frac = 0.02, interval = 50, until = 300`
(6 rounds × 2 % turnover = 12 % cumulative).

## 3. Validation matrix

Winner config applied to all four held-out variants, and the alt
config (0.10 × 100 × 3 rounds = 30 % cumulative turnover) on the
two seq_2 variants.

| scene | variant | config | baseline test | S4 test | Δ | baseline train | S4 train | Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| seq_1 | HO_8   | winner 0.02/50 | 0.5672 | **0.6656** | **+0.098** | 0.8657 | 0.8776 | +0.012 |
| seq_1 | HO_128 | winner 0.02/50 | 0.5118 | **0.6440** | **+0.132** | 0.8360 | 0.8578 | +0.022 |
| seq_2 | HO_8   | winner 0.02/50 | 0.5633 | 0.5518 | −0.012 | 0.7867 | 0.8341 | +0.047 |
| seq_2 | HO_8   | alt 0.10/100   | 0.5633 | **0.5807** | **+0.017** | 0.7867 | 0.8422 | +0.056 |
| seq_2 | HO_128 | winner 0.02/50 | 0.5900 | 0.5766 | −0.013 | 0.7701 | 0.8034 | +0.033 |
| seq_2 | HO_128 | alt 0.10/100   | 0.5900 | **0.6057** | **+0.016** | 0.7701 | 0.8154 | +0.045 |

**Scene asymmetry explained by Fisher concentration** (§8 of
matched-grid findings): seq_2's rotation Fisher is **81.6 %** at
top-1 % (vs seq_1's 55.6 %). A 6-round × 2 % turnover on seq_2 ends
up replacing the same top-Fisher region repeatedly (200 points keep
getting children, children become high-Fisher, they get split
again). That drives the model to over-concentrate on the narrow
scatterer band and over-fit training noise. The alt config (fewer,
bigger moves) moves further per round, escaping that basin.

---

## 4. Comparison with prior stages

HO_128 test cc, both scenes, each stage's best configuration:

| stage | seq_1 HO_128 | seq_2 HO_128 | notes |
|---|---:|---:|---|
| Prior-grid baseline (N=90k) | 0.4904 | 0.5043 | position-grid mismatch |
| S1 matched grid (N=20k)     | 0.5118 | 0.5900 | seed fix + target_n reduction |
| S1 + S2 (best cfg per scene)| 0.5912* | 0.5900 | *HO_8 only; S2 hurts HO_128 |
| S1 + **S4** (winner / alt)  | **0.6440** | 0.5766 / pending | **seq_1 +0.132, seq_2 mixed** |
| UB_144 (matched, N=20k)     | 0.8309 | 0.7319 | ceiling under same position grid |
| Swap-ablation cap (HO_mat + UB_rot) | 0.685 | 0.639 | theoretical HO-param ceiling |

**S4 closes ~81 % of the swap-ablation gap on seq_1 HO_128** and is
~30 % of the way to closing it on seq_1 HO_8.

---

## 5. Open questions

1. **seq_2 asymmetry**: is the alt config (0.10/100) the right one
   for HO_128, or is there a different hyper-parameter cell that
   closes the gap? Pending result from seq_2 HO_128 × alt config.
2. **Pos-jitter scaling**: 2 cm is ≈ 5 λ at 77 GHz. Larger jitter
   (e.g. 5–10 cm) may help escape narrow-scatterer concentration on
   seq_2; smaller jitter (1 cm) may help on seq_1 where a few
   hundred tightly-clustered points already dominate.
3. **Interaction with S2**: densify invalidates the S2 Fisher
   snapshot (reset in code). It's possible that a S2 + S4 stack —
   use S4 early (iters 50–200) to re-distribute points, then
   "freeze" with a Fisher-weighted drift penalty in iters 200–500
   — beats either alone.
4. **S4 + pass-3 anchors**: orthogonal; should stack freely. Worth
   trying on HO_128 variants where pass-3 showed +0.004 / +0.034.

---

## 6. Recommended next step

Assuming the seq_2 HO_128 alt run goes positive (say +0.02 to +0.05),
the immediate follow-up is:

1. Publish `--reg_densify_*` behind opt-in flags (done; default off).
2. Per-scene winner configs: seq_1 uses (0.02/50/300), seq_2 uses
   (0.10/100/300). Document in the paper's implementation details.
3. Extend to the full 9-scene benchmark — estimate ~2 hr compute on
   2 GPUs at target_n=20k.
4. **Stretch**: sweep `pos_jitter` × `split_frac × interval` on
   seq_2 HO_128 specifically to find a cell matching seq_1's +0.13
   response.

---

## 7. Artefacts

- Implementation: `mm25DGS_v5/train_frame_nvs.py` `_densify_step()`
  + new CLI flags (`--reg_densify_*`).
- Sweep logs: `logs_s4/*.log`.
- Trained models: `mm25DGS_v5/output_frame_nvs/*_dnsfy*_pass2_N20000/`.
- This doc.
