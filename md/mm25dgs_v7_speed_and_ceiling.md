# v7 Doppler — rendering-speed deep dive + plan

**Date:** 2026-04-23
**Scope:** Diagnose why `mm25DGS_v7` bench runs at ~3.0 s/iter (vs ~0.27 s/iter
for v5/M1 on the same 8-frame, 20K-pt scenes) and define the fix sequence.

**Related:** `md/mm25dgs_v7_training_ceiling.md` — why train CC plateaus at
~0.70. That work is **parked** until the speed-up plan below lands.

Measured baselines (`logs_v6_M1/` and `logs_v7/`, same 4090 / 20K points):

| run | per-iter | total (500 it) |
|---|---:|---:|
| v5/M1 seq_1_frame_185 | **271 ms** | 136 s |
| v7 Dop seq_0_frame_135 | **2999 ms** | 1500 s |

v7 is **11×** slower than v5. A purely physical 16×-step5 hit should give
~3× (one BSDF + 16 cheap step5 splats ≈ 3× one-shot render). The extra 4×
is pure Python / framework overhead.

---

## PART 1 — Speed plan

### 1.1 Bottleneck inventory (in descending order of estimated impact)

#### **B1. `torch.cuda.empty_cache()` called in the hot loop** — *critical*
`mm25DGS_v7/train_frame_nvs.py:836` calls `empty_cache()` on every
bundle, every iteration → **8 × 500 = 4000 full CUDA syncs** per run.
Each sync blocks all work in flight and flushes the allocator. Typical
cost: 10–50 ms per call on a loaded 4090.

**Expected win:** 2–4× overall. Removal is a one-line edit.
This should never be in a training hot loop.

#### **B2. Python for-loop of 16 `step5_fused` kernel launches** — *large*
`mm25DGS_v7/rasterizer_factorized.py:762-777` — one forward kernel launch
per chirp (× one backward in the autograd pass). At ~45K active points
the kernel itself is fast (~1–3 ms); 16 forward + 16 backward launches +
32 Python-side autograd-graph ops is dominated by **launch + dispatch +
graph-construction overhead**, not compute.

**Fix options, cheapest first:**

- **B2a. Write `step5_doppler_fused` CUDA kernel.** Takes
  `w_full[M,T,R]`, `phi_base[M,T,R]`, `n_peak[M,T,R]`, plus the
  per-(chirp,TX) time-offset `t_off[CH,T]` and per-point projection
  `u_dot_vego[M]`. Internally computes
  `phi = phi_base + (-(4π/λ) * u_dot_vego[m] * t_off[c, t])`
  and scatters into `rp[CH, T, R, K]`. **One forward + one backward
  kernel**. Preserves the current contract (the Doppler phase is already
  treated as non-differentiable via `detach_phase=True`), so the backward
  only needs to emit `grad_w_full` (sum of per-chirp gradients).
  **Expected win:** 2–3× over the 16-loop form.
- **B2b. If we don't want to write a new kernel:** issue the 16 step5
  launches from a `CUDAGraph` capture so dispatch is amortised. Launch
  overhead drops by ~10×.
- **B2c. Last resort:** keep the Python loop but batch the `phi_m`
  construction once outside the loop (factorise `u_dot_vego` and
  `t_offset` so only a scalar multiply + add is per-chirp work). Tiny
  win on its own but makes B2b/B2a easier.

**Expected win (B2a):** ~2× overall.

#### **B3. Per-chirp Python loop over `txrx_to_vx_chirps_torch`** — *medium*
`mm25DGS_v7/data/ra_utils.py:93–99` and `:121–126` — calls
`txrx_to_vx_chirps_torch` 16 times, each performing 192 scatter-writes
into a fresh `(1, 7, 86, R)` tensor on both the GT and the **pred path**
(which must retain autograd). The backward through this chunk retraces
16 tiny scatter graphs.

`mmir/data/ra_utils.py:txrx_to_vx_chirps_torch` asserts `size(0) == 1`,
so the loop is purely a limitation of the helper, not of the math.

**Fix:**
1. Precompute `(rx_id[86], tx_id[86])` and `count[86]` once at import
   time (the (RX, TX)→(ele, az) mapping is static).
2. New helper `vx_index_from_txrx(rp_stack_cht_r)` that does:
   `rp_stack[:, :, rx_id, tx_id, :] / count[None, :, None]` →
   returns `(CH, 86, R)` in a single gather + divide.
3. `rp_stack_to_rad_complex` becomes
   `_azimuth_fft_on_vx86(gather(rp_stack))` → one FFT.

Because we pick `vx[0, 0, :, :]` (only el=0) downstream, we only need the
86 el=0 positions; the full `(7, 86)` expansion is dead work.

**Expected win:** 20-40% (eliminates 16 × Python-for graph nodes).

#### **B4. `.abs()` on a complex (31, 127, 256) tensor per bundle** — *small*
Fine on its own, but the current code materialises `mag_pred` (a float
tensor) while the complex `rad_pred` is still alive in the graph, so
the backward saves BOTH. At 31·127·256·4 B (float) + ·8 B (complex)
= 4 MB / bundle × 8 bundles = ~32 MB. Not huge, but feeds B1's reason
to exist.

**Fix:** use `torch.view_as_real(rad_pred).pow(2).sum(-1).sqrt()` (no
intermediate) or compute `|·|²` and compare against `gt²` (no `sqrt`
at all → better gradient and one less kernel).

**Expected win:** ~5%, but also removes the justification for B1.

#### **B5. 8 × `.backward()` calls per iter** — *small*
`train_frame_nvs.py:820` — one backward per bundle. Each backward
traverses the full BSDF graph (step 1–4), which is the expensive part.
Because step 1–4 is shared across **all** bundles through the model
parameters, we accumulate gradients 8× but re-traverse the same
structure 8 times.

Fixing this properly requires summing losses across bundles into a
single scalar before `.backward()` — but that costs peak memory because
all 8 forward graphs must stay live simultaneously. At 45K × 192 =
8.6M paths × 4 B × handful-of-tensors that's ~1 GB/bundle of activations
→ 8 GB. Probably too tight on a 4090 alongside other work.

**Recommendation:** leave per-bundle backward as-is. Flag for later if
peak-mem budget allows.

#### **B6. Per-iter `_render_and_cart_corr(test_sample)`** — *situational*
Just added for per-iter test-CC tracking. Adds ~1 chirp-0 forward per
iter (~30–50 ms). Consider gating behind `--test_cc_every` (e.g. every
5 iters) if we turn this on by default.

**Expected win (when active):** skip 4/5 evaluations → ~150 ms/iter
saved (~5% of the post-fix wall-clock).

#### **B7. Redundant `u_dot_vego` recompute in the chirp loop** — *trivial*
`rasterizer_factorized.py:_doppler_phase_per_chirp` recomputes the
`positions → unit vector → dot v_ego` chain 16× per render. Pure
refactor; no kernel changes. Move outside the loop.

**Expected win:** <1% (already vectorised; M ~45K), but makes B2a/B2c
implementable.

---

### 1.2 Summary & sequencing

| # | fix | effort | expected wall-clock win |
|---|---|---|---:|
| B1 | delete `empty_cache()` | 1-line edit | **2-4×** |
| B3 | batch `txrx_to_vx_chirps` over CH | ~1 h refactor | 1.2-1.4× |
| B4 | drop `mag_pred` materialise; use `|·|²` | 10-line edit | 1.05× |
| B7 | hoist `u_dot_vego` outside chirp loop | 10-line edit | 1.01× |
| B2b | CUDAGraph the 16 step5 launches | ~3 h | 1.5-2× |
| B2a | `step5_doppler_fused` CUDA kernel | ~1-2 days | 2-3× (supersedes B2b) |

**Combined after B1 + B3 + B4 + B7:** expect ~2.5-3× → ~1.0-1.2 s/iter,
i.e. bench drops from 25 min/scene to ~10 min/scene with **zero CUDA
work**.

**Combined after B1 + B2a + B3:** expect ~6-8× → ~0.35-0.5 s/iter,
i.e. v7 within ~1.5-2× of v5. This is the ceiling for correct
16-chirp rendering; you'll never hit exact v5 parity because v5
renders one chirp and v7 renders sixteen.

### 1.2.1 Landed — measured

| pass | fixes | ms/iter | ×baseline |
|---|---|---:|---:|
| baseline | — | 2999 | 1.0× |
| pass 1 | B1 + B3 + B7 | 649 | 4.6× |
| pass 2 | + B2a (fused step5_doppler CUDA kernel) | 558 | 5.4× |

B2a shipped as a separate v7 extension (`mm25DGS_v7/cuda/`
→ `mm25dgs_v7_cuda`) rather than folding into the v5 extension, so
v5 stays stable. Unit-test vs 16× step5_fused reference passes at
~5e-6 rel forward / ~2e-5 rel backward (well under the ±0.03 MC floor).
At 558 ms/iter the new bottleneck is the BSDF path (step 1–4), not
step 5 — further wins require B5 (cross-bundle accumulation, gated on
peak memory) or B6 (gate per-iter test_cc tracking). B2b/B4 superseded.

### 1.3 Proposed order of work

1. **Ship B1, B4, B7 today** (edits in existing files; run a one-scene
   bench to confirm ~2× improvement).
2. **Ship B3 this week** (add batched `vx_index_from_txrx` helper in
   `mm25DGS_v7/data/ra_utils.py`, preserving the per-chirp API as a
   thin wrapper for validation scripts).
3. **Decide on B2a vs B2b** based on the residual budget after (1)+(2).
   If we're already below ~1 s/iter, B2b (CUDAGraph) is probably
   sufficient and much cheaper. Write B2a only if we need Doppler-training
   to scale to 16 chirps × 64 frames (the long-run target).

### 1.4 How to validate each fix

For every change:

1. Run `python -m mm25DGS_v7.scripts.validate_doppler_synthesis` —
   three gates must still PASS (8.2.1 phase/amp, 8.2.2 LERP check,
   8.2.3 v_ego=0 identity). This catches any forward-model regression.
2. Run a single-scene 50-iter micro-bench:
   `run_v7_bench.sh` with `--iters 50` on `seq_0_frame_135`.
   Record `ms/iter` and `final_train_cc` (should match within ±0.005).
3. Only ship if both (a) gates pass and (b) no train-CC regression.

---

## PART 2 — Training-ceiling analysis

Moved to its own doc: `md/mm25dgs_v7_training_ceiling.md`. Parked
until the speed plan above lands.
