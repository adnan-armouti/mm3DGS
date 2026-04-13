# v4 ceiling-breaking directions

Three concrete future investigations that could push the current v4 ceiling of **0.9322** mean cart_corr beyond where clamp fixes + LR tuning have taken us. Written after the optimizer-dynamics investigation ([v4_material_optimizer_dynamics_investigation.md](v4_material_optimizer_dynamics_investigation.md)) established that 0.932 is the depth of the converged loss minimum under the current BSDF + raw-MSE + factory-patterns configuration, and that no further material-parameter or LR tuning will improve it.

Each direction targets a different limiting factor:

1. **Richer BSDF output** attacks the "scalar f_cos per path" bottleneck that causes every material parameter to map through a single real number
2. **Structural losses on |RA|** attacks the "per-bin MSE treats all errors equally" bottleneck
3. **Normals LR tuning** attacks the possibility that the rotation optimizer may also be off-optimum in the same way the material optimizer was, which would cost performance on the normal side of the decomposition

The three directions are independent and can be pursued in any order. Direction 3 is by far the cheapest and least risky; it should probably be run first regardless of whether we pursue the other two.

---

## Direction 1 — Richer BSDF output

### Current state

The simplified BSDF produces a single real scalar `f_cos(m, t, r)` per path. Every material parameter feeds through:

```
f_cos = R_jones(eps_real, eps_imag, thickness, θ) ×
        [τ_eff(σ_h, l_c, τ_base) · f_KA(σ_h, l_c) +
         (1 − τ_eff) · f_SPM(σ_h, l_c, eps_contrast)] ×
        cos_i
```

After scatter+FFT, the loss is MSE on `|RA|`. So each parameter's gradient routes through one scalar per path → 192-bin range profile → 127×256 RA magnitude → one loss number. This collapses the parameter space: a change to `sigma_h` and a change to `eps_imag` that produce the same Δ|f_cos| at the same path are indistinguishable to the loss.

The rank check showed the 6 per-point gradients are nearly full-rank (correlation PR ≈ 5.97/6), so they do point in different directions — but those directions all map to the same scalar output via `f_cos`. The information bottleneck is the scalar output, not the input parameters.

### Motivation

If we could preserve more per-path feature dimensions through the forward pipeline, different material parameters would contribute through different output channels and the loss would be able to distinguish them even after aggregation. Concretely, there are two readily-available extra dimensions:

- **Polarization**: the Jones computation already produces `E_rx` as a complex 2D vector (E_s, E_p components). Currently we collapse `|E_rx|²` into a single scalar `R_jones`. If we preserved the s/p components as two separate output channels, eps_real and eps_imag would contribute differently to each (TE vs TM Fresnel coefficients have different dependencies), and the loss could fit them independently.
- **Frequency band**: the chirp has bandwidth 2–4 GHz around 77 GHz. Currently the BSDF evaluates at the center frequency only. Evaluating at (say) 5 frequency bins across the bandwidth would expose different `kσ_h` and `k·l_c` regimes, which are qualitatively different scattering regimes.

### Approach A — polarization-preserving pipeline

The `R_jones` computation already has all the information. Instead of reducing to a scalar `|E_rx|²` and routing through `f_lobe × cos_i`, carry the complex `E_rx ∈ ℂ²` forward through scatter+FFT separately for the s- and p-components. The range profile becomes `(rp_s, rp_p)` of shape `(n_tx, n_rx, K, 2)`. The GT likewise has polarization info in the raw I/Q (the radar transmits in one polarization; the 16 RX elements receive in specific polarizations). Compare per-polarization.

**Concrete changes**:
- `render_factorized`: instead of computing `R_jones = |E_rx|²` at line 337, keep `E_s`, `E_p` as complex tensors `(M, n_tx, n_rx)`. Replace scalar `f_cos = R_jones × f_lobe × cos_i` with polarized `f_cos_s = E_s × sqrt(f_lobe) × sqrt(cos_i)` and same for p. Both become complex.
- Scatter+FFT: the existing splat path splats a single complex amplitude `(rp_real, rp_imag)`. Change it to splat two complex amplitudes in parallel — double the memory, same compute pattern.
- Loss: compute MSE on `|rp_s|, |rp_p|` separately and sum. Or weight them differently based on the data.

**Effort**: ~1 day of code. The trickiest part is knowing whether the MMWCAS GT actually contains polarization-separated data (the 16 RX elements are configured with specific polarizations per the TI spec sheet). Need to audit `mmir/data/ra_utils.py` and the TI config files to confirm.

**Risk**: medium. If the GT is effectively single-polarization, splitting into s/p will just duplicate the same signal into two channels and give no lift. If the GT has real polarization structure that's currently being collapsed, the lift could be substantial.

**Expected outcome**: +0.005 to +0.015 cart_corr improvement if polarization structure is present in GT. Zero if not.

### Approach B — per-frequency-band BSDF evaluation

Currently the BSDF uses a single `λ = c / f_center ≈ 3.9 mm` at 77 GHz. The chirp spans 2–4 GHz though, so the effective wavelength sweeps `3.75 mm` to `4.05 mm` across the chirp. Different `λ` → different `kσ_h`, different `k·l_c`, different slab Fresnel phase. These variations matter because at the chirp extremes, scattering regimes can qualitatively shift (e.g., SPM applicable at f_high but not f_low if sigma_h is near the boundary).

Evaluate the BSDF at 3–5 frequency samples across the chirp bandwidth and produce separate output channels. The range compression FFT already resolves frequency-dependent effects, but only within the K=256 range bins. Making the BSDF itself frequency-dependent would add information the current pipeline doesn't use.

**Concrete changes**:
- `rasterizer_factorized.py` Step 1: introduce `lambdas = [lambda_low, ..., lambda_high]` instead of single `WAVELENGTH`. Compute `alpha_ggx`, `kappa_SPM`, `R_jones` at each lambda.
- Produce `f_cos_freq` of shape `(M, n_tx, n_rx, n_freq)`.
- At the scatter step, the range bin `n_peak` already depends on the chirp slope × delay. Per-frequency evaluations should be splatted at slightly different bin positions (or use a linear combination that approximates the true frequency-dependent response).
- Loss: MSE on the per-freq-channel output.

**Effort**: ~2 days. The tricky part is correctly aligning the per-frequency BSDF evaluations with the FFT range bins. This is non-trivial and probably needs a second-opinion on the radar signal processing side.

**Risk**: higher than Approach A. Small numerical errors in frequency alignment can produce artifacts that swamp any benefit. The ADC data is complex I/Q in the time domain, and the RA magnitude is computed via FFT — introducing per-freq BSDF evaluations requires a non-trivial rethinking of what "range bin" means in the forward model.

**Expected outcome**: +0.005 to +0.020 if done correctly. Potential large regression if done incorrectly (signal processing alignment errors).

### Recommended first try

**Approach A (polarization)** before **Approach B (frequency)**. Polarization is already half-computed in the existing Jones code — we just need to stop collapsing the complex E vector at the end. Frequency is a bigger architectural change and has more failure modes.

### What would NOT help

- Adding more input parameters (e.g., complex dielectric as 2 params × 2 frequencies = 4 per point). We already showed the 6 params have enough degrees of freedom; the bottleneck is the output, not the input.
- Higher-order BSDFs (multiple scatter, volumetric). Same problem: if they collapse to scalar f_cos, the loss can't distinguish the new parameters from the existing ones.

---

## Direction 2 — Structural losses on |RA|

### Current state

The loss is raw MSE on `|RA|` magnitude, computed per-bin with uniform weight:

```python
loss = ((ra_rend_mag - gt_mag) ** 2).mean() / gt_mean_squared
```

Every range-bin × azimuth cell contributes with equal weight, scaled by the GT mean squared for unit normalization. This is the **simplest possible** image-space loss and makes no assumptions about scene structure.

### Motivation

The cart_corr metric (our evaluation target) is a structural correlation — it cares about relative patterns, not absolute pixel values. The MSE loss cares about absolute values. These are aligned but not identical. If we could make the training loss more aligned with the eval metric in its treatment of structural information, we'd push cart_corr higher without changing anything else.

Three concrete structural losses, all of which preserve the "no absolute phase, no absolute per-point position" invariants the user specified:

### Approach A — Sobel / gradient loss

Compute the image gradient (Sobel or simple finite difference) of both |RA_rend| and |RA_gt|, MSE the gradients:

```python
def gradient_loss(rend_mag, gt_mag):
    gx_r = rend_mag[:, 1:] - rend_mag[:, :-1]
    gy_r = rend_mag[1:, :] - rend_mag[:-1, :]
    gx_g = gt_mag[:, 1:] - gt_mag[:, :-1]
    gy_g = gt_mag[1:, :] - gt_mag[:-1, :]
    return ((gx_r - gx_g) ** 2).mean() + ((gy_r - gy_g) ** 2).mean()

loss = alpha * mse_loss + (1 - alpha) * gradient_loss
```

**Motivation**: gradient loss rewards matching edges and transitions, which is exactly what cart_corr rewards (structural correlation). Radar RA images are dominated by sharp specular peaks against a low background; a gradient loss emphasizes those peaks.

**Effort**: ~2 hours (including LR-balance tuning).

**Expected outcome**: +0.003 to +0.010.

### Approach B — Per-channel (TX/RX) weighted loss

Currently the 192 (TX, RX) channel pairs are averaged. Each channel has a different angle geometry, different effective antenna gain, and different material sensitivity (per-channel Fresnel coefficients). Uniform averaging may be wasting signal from channels where materials matter more.

Compute MSE per (tx, rx) channel independently and apply a weighting. Two options:

- **Equal-variance weighting**: normalize each channel by its own variance before MSE. Prevents high-amplitude channels from dominating.
- **Per-channel RA correlation**: weight by the per-channel cart_corr of the GT — channels with higher SNR (stronger structure) get more loss weight.
- **Learned weighting**: let the optimizer decide — one weight per channel, subject to a constraint that weights sum to 1. Bad idea (introduces a new over-parameterization), skip.

**Motivation**: the 192 channels are not equally informative. The existing uniform average is the simplest choice but not necessarily the best one.

**Effort**: ~4 hours.

**Expected outcome**: +0.002 to +0.008. Smaller than Approach A because it's a weighting change, not a new information channel.

### Approach C — Frequency-domain loss

FFT the RA image (which is already in the polar-azimuth frequency domain — well, almost; it's RA, which is range-and-angle) and compare the magnitude spectra.

**Motivation**: the spatial-frequency content of an RA image encodes structure at different scales. Large-scale features (walls) have low spatial frequencies; small-scale features (vehicles, foliage) have high spatial frequencies. Uniform per-pixel MSE doesn't distinguish these; a frequency-weighted loss can.

**Effort**: ~3 hours.

**Expected outcome**: +0.002 to +0.008. Similar in magnitude to B.

### Recommended first try

**Approach A (gradient loss)** first. It's the smallest change, most closely aligned with the cart_corr metric, and has the cleanest interpretation. If it gives a meaningful lift, escalate to combining it with B and C.

### What would NOT help

- Per-point L2 regularization on materials. We tried symmetry-breaking (T1) and per-cluster (T3) earlier — all washes.
- Any loss that penalizes absolute phase or per-point path length. Those violate the deliberate design decisions to keep positions and radar positions frozen.
- Replacing `|RA|` with `log|RA|`. It's a natural dB-space loss but in practice it mostly reweights the noise floor upward, making the optimizer chase background fluctuations.

---

## Direction 3 — Normals LR tuning

### Current state

In `train_gaussians`, the normal rotations have `lr=2e-3`. This has never been tuned systematically. Normals provide ~0.54 of the 0.56 cart_corr gap from pure-init to trained (from the earlier LEARN matrix) — they are the dominant learnable knob. If the LR is off-optimum in the same way material LR was, the current ceiling is partly the result of normal-side overshoot, and tuning it should push further.

### Motivation

The material LR investigation found LR=0.7 was ~70× too large, and lowering to 0.01 improved mean cart_corr by +0.0041. The normal LR=2e-3 was picked without systematic sweeping. It may be:

- **Correct** — 2e-3 is already the optimum for rotation parameters
- **Too large** — overshooting the normal-side minimum, analogous to the material case. Lowering would give a similar mechanism-driven improvement.
- **Too small** — rotations never fully converge in 500 iters, leaving per-scene ceilings unrealized.

### Approach

Run an LR sweep exactly analogous to the material sweep:

1. Set LRs = {5e-4, 2e-3 (current), 5e-3, 2e-2, 5e-2}
2. Benchmark each on 7 scenes × 500 iters
3. Pick the optimum
4. Optionally: run the same gradient diagnostic (mean drift, sign consistency, column norms) at the current vs optimum LR to verify the mechanism is analogous

**Effort**: ~30 min of code, ~30 min of compute (5 configs × 7 scenes × 500 iters ≈ 35 min).

**Expected outcome**: three possible scenarios:

- **2e-3 is optimal** — zero improvement, but at least we know it. This is the null result.
- **Lower LR helps** — a +0.002 to +0.005 mean cart_corr improvement analogous to the material case. Per-scene consistency check will confirm it's real.
- **Higher LR helps** — less likely but possible. Would indicate normals were under-converged.

### Why this is the highest-priority direction right now

- Cheapest: ~1 hour of work total
- Lowest-risk: it's a hyperparameter tweak on an existing well-understood code path
- Highest expected value per unit effort: if normals are off-optimum the way materials were, we get ~+0.004 for almost no work
- **No hypotheses to test, no architectural changes, just a sweep**. The simplest possible follow-up to the material LR sweep.

### What this direction is NOT

- Not a change to the rotation parameterization. Quaternions stay as is.
- Not a change to the gradient clipping (currently `clip=0.5` for rotations). Clipping is coupled to LR, but the sweep will reveal whether clip needs adjustment.
- Not a change to the cosine decay schedule (`warmup_iters=0, decay_start=200`). The current schedule is applied uniformly to all groups, and if LR scales change, the schedule rescales too.

---

## Recommended execution order

1. **Direction 3 (normals LR sweep) first** — ~1 hour total. Null-result upside. Small-positive downside.
2. **Direction 2A (Sobel/gradient loss) second** — ~3 hours. Structural alignment with the eval metric.
3. **Direction 1A (polarization-preserving pipeline) third** — ~1 day. Bigger architectural change but high potential upside.
4. **Direction 1B (frequency-domain BSDF)** — only if 1A works and the team wants to push further. Risky and expensive.
5. **Direction 2B/2C (per-channel, frequency)** — can be combined with 2A if it works.

If the goal is to ship at 0.932 and stop optimizing, skip all three and accept the current ceiling. The reviewer defense in that case is:

> We validated that no individual material parameter fix (clamp widening, SPM validity removal, sigmoid → softplus), no alternative optimizer configuration (sign-grad, per-column normalization, per-column LR scaling), and no single-hyperparameter tuning (LR sweep) improves the current ceiling beyond 0.932 ± 0.005. The ceiling is structural to the loss landscape of the current BSDF + loss + data configuration, not a symptom of over-parameterization, under-parameterization, or off-optimum tuning.

## Out of scope for this plan

- **Cross-sensor transfer** (cascaded → single-chip IWR1443). Explicitly out of scope per user direction.
- **Longer training** (2000+ iters). Excluded per user direction.
- **Complex-valued losses** (penalizing phase). Rejected because of user's clarification that phase is deliberately thrown away to prevent per-point position gradients from dominating.
- **Alternative physics** (neural BSDF, tabulated BRDF). Too big a change for this plan; would be a separate investigation.
