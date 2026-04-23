# mm25DGS_v7 — Doppler-aware forward model

Date: 2026-04-23
Owner: Adnan Armouti
Status: plan (awaiting approval)

## 0. Context and Scope

### 0.1 Why v7 instead of extending v6

The mm25DGS_v6 investigation (now committed on `pt`) iterated on non-Doppler losses (Gram Frobenius², baseline-binned, hybrid, etc.), extended-training-window preprocessing under `data_v2/`, and a first Doppler prototype ("M2") that rendered 16 physical chirps at LERP-interpolated poses. Every v6 milestone regressed the v5/M1 baseline:

- M1 (per-virt plumbing) — neutral.
- M1.5 variants (9 losses × 7 scenes) — all below v5/M1.
- data_v2 extended-window v5/M1 — −0.052 mean cc vs v5/M1 on data/.
- M2 Option-A (16 physical renders, v5 mse_raw on |RAD|) — **−0.070 mean cc vs v5/M1**.

Most damning: the naive `½·(GT[F-1] + GT[F+1])` predictor beats v5/M1 on 5/7 scenes. v6's Doppler prototype (M2) beat naive on only 1 of 6 scenes — the dynamic-scene seq_1_frame_438 where v5 also wins.

**v7 starts from the known-good v5 baseline (`data/` tree, v5 rasterizer, v5 mse_raw on |RA|) and adds the one piece of physics v5 is missing: ego-motion-induced Doppler phase in the forward model.** No v6 code is imported unless explicitly required by the Doppler change.

### 0.2 What v7 explicitly is NOT

- **Not a copy of v6.** v6 is treated as a failed branch. v7 has no dependency on `mm25DGS_v6/`.
- **Not a rewrite of the rasterizer primitive.** v7 keeps v5's point-scatterer + KA+SPM + Mitsuba+DrJit CUDA renderer. We add a per-chirp analytic phase factor in the scatter step only.
- **Not a new alignment stack.** v5's pass-2 trajectory-refined alignment (already producing `data/alignment_data/<scene>/cascade/cascaded_frame_<F>_aligned_pass2.json`) is the canonical pose source.
- **Not the DART architecture.** DART renders volumetric Range-Doppler maps directly; v7 keeps v5's per-virtual complex range-profile primitive and adds Doppler as an orthogonal axis. Differentiation-from-DART in §7.
- **Not v_ego-learnable.** Justification in §5. v_ego is computed once per frame from pose-difference and frozen.

## 1. Forward-model equations (consistent with Li et al., IEEE Access 2021)

Reference: [md/Signal_Processing_for_TDM_MIMO_FMCW_Millimeter-Wave_Radar_Sensors-2.pdf](Signal_Processing_for_TDM_MIMO_FMCW_Millimeter-Wave_Radar_Sensors-2.pdf).

### 1.1 Single-scatterer IF signal (PDF §II, eqs. 8-10)

For a static scatterer at range `r` from a stationary radar, the IF signal is
$$
x_\mathrm{IF}(t) \;=\; A\,e^{j(2\pi f_\mathrm{IF} t + \phi_\mathrm{IF})}
\;,\qquad
f_\mathrm{IF} = \tfrac{2Sr}{c},\;\;\phi_\mathrm{IF} = \tfrac{4\pi r}{\lambda}.
$$
v5 renders this post-range-DFT: a complex amplitude at range bin `k ≈ r/r_res` with phase `φ_IF(p) = 4π r(p) / λ`, per (TX, RX) virtual. **Everything below is a multiplicative phase-modulation on top of this v5 primitive.**

### 1.2 Doppler — inter-chirp slow-time (PDF §IV.A, eq. 20)

For a moving scatterer at radial velocity `v`, the phase shift between consecutive chirps (separated by `T_c` in time) is
$$
\Delta\phi_\mathrm{chirp} \;=\; \tfrac{4\pi v T_c}{\lambda}\,.
$$

**Sign convention (per user directive, 2026-04-23)**:
- Radar boresight `ê_bore` defines the positive direction in the radar's local frame.
- An object moving AWAY from the radar has POSITIVE radial velocity `v_p > 0` ⇒ Doppler frequency `f_d > 0`.
- An object moving TOWARDS the radar has NEGATIVE radial velocity `v_p < 0` ⇒ `f_d < 0`.

Our scene is static, but the radar moves with `v_ego`. For a static scatterer at `r_s(p)` and radar-center position `r_\mathrm{radar}(t) = r_\mathrm{radar}(0) + v_\mathrm{ego}\,t`, the range is `r_p(t) = \|r_s(p) - r_\mathrm{radar}(t)\|` and its time derivative is
$$
\tfrac{dr_p}{dt} \;=\; -\langle\hat u(p),\,v_\mathrm{ego}\rangle
\qquad\text{with}\qquad
\hat u(p) \;=\; \bigl(r_s(p) - r_\mathrm{radar}(0)\bigr) \,/\, \|\cdot\|.
$$
`dr/dt < 0` means range is DECREASING (scatterer appears to approach) ⇒ `v_p = dr_p/dt < 0` (negative radial velocity, as per the sign convention above). Consistency check: when `v_\mathrm{ego}` is aligned with boresight AND the scatterer is ahead of the radar (`\hat u \approx \hat e_\mathrm{bore}`), `⟨\hat u, v_\mathrm{ego}⟩ > 0` ⇒ `v_p < 0` — matching PDF convention and the user's stated sign rule.

The chirp-axis phase accumulation at chirp index `m ∈ [0, N_c)` becomes
$$
\phi^{(m)}_\mathrm{chirp}(p) \;=\; \tfrac{4\pi\,v_p \,m T_c}{\lambda}
\;=\; -\tfrac{4\pi\,\langle\hat u(p), v_\mathrm{ego}\rangle\,m T_c}{\lambda}\,.
$$

### 1.3 TDM intra-burst — per-TX slow-time offset (PDF §VI.B, eqs. 52-55)

The MMWCAS cascade operates in TDM: within each chirp slot of duration `T_c`, the 12 TX fire sequentially with per-TX duration `T_a = T_c / N_T`. Per TI documentation for AWR2243BOOST / MMWCAS, TX fire in order `TI-TX1, TI-TX2, …, TI-TX12`.

**Critical: TI's TX labels differ from our config labels.** Cross-referenced against the board-PCA visualisation ([output/cascade_board_pca_seq_0_frame_135_F135.png](../output/cascade_board_pca_seq_0_frame_135_F135.png)) and TI's TIDUEN5A user guide:

| TI label | Our config label | Our config index | ADC channel index |
|---|---|---:|---:|
| TI-TX1 (fires first) | TX12 | 11 | 5 |
| TI-TX2 | TX11 | 10 | 4 |
| TI-TX3 | TX10 | 9 | 3 |
| TI-TX4 | TX9 | 8 | 11 |
| TI-TX5 | TX8 | 7 | 10 |
| TI-TX6 | TX7 | 6 | 9 |
| TI-TX7 | TX6 | 5 | 8 |
| TI-TX8 | TX5 | 4 | 7 |
| TI-TX9 | TX4 | 3 | 6 |
| TI-TX10 | TX3 | 2 | 2 |
| TI-TX11 | TX2 | 1 | 1 |
| TI-TX12 (fires last) | TX1 | 0 | 0 |

From this we derive the **per-ADC-channel firing index** `k(i) ∈ [0, 11]` (how many `T_a` intervals after the burst-start the TX at ADC channel `i` fires):

```python
# ADC-channel index i = 0..11 → TI firing index 0..11
TI_FIRING_INDEX_FROM_ADC_CH = (11, 10, 9, 2, 1, 0, 8, 7, 6, 5, 4, 3)
# Inverse: ADC channel fired k-th is ADC_CH_FIRED_KTH[k]
ADC_CH_FIRED_KTH = (5, 4, 3, 11, 10, 9, 8, 7, 6, 2, 1, 0)
```

The TDM phase term at chirp index `m` and ADC channel `i` for scatterer `p` is then
$$
\boxed{\;
\phi^{(m,i)}(p) \;=\; -\tfrac{4\pi}{\lambda}\,\langle\hat u(p), v_\mathrm{ego}\rangle\,\bigl(m T_c + k(i)\,T_a\bigr)
\;}
$$
where `k(i) = TI_FIRING_INDEX_FROM_ADC_CH[i]` maps ADC channel to firing-order index. The PDF (eq. 55) calls the TDM compensation `exp(-j·2πl/(N_c·N_T))` — that is the PDF's post-DFT compensation form. Our forward model applies the physical phase shift directly, so the PDF's compensation is automatically baked in.

### 1.4 Per-point scatter output

The v5 output per-point per-virtual is
$$
c^{(0,0)}_{i,j}(k;\,p) \;=\; G_\mathrm{Tx}(\hat u, i)\cdot G_\mathrm{Rx}(\hat u, j)\cdot \mathrm{BSDF}(p)\cdot \tfrac{1}{r^2}\cdot e^{j\phi_\mathrm{IF}(p)}
$$
(indexed by TX ADC channel `i`, RX ADC channel `j`, range bin `k`). v7's output per-point per-(chirp, tx, rx, range_bin) is
$$
\boxed{\;
c^{(m,i)}_{j}(k;\,p) \;=\; c^{(0,0)}_{i,j}(k;\,p) \,\cdot\, \exp\bigl(j\,\phi^{(m,i)}(p)\bigr)
\;}
$$
with `φ^(m,i)(p)` from eq. (§1.3). Note: the phase factor depends on `(m, i, p)` only — not on `j` (RX, simultaneous within each TX firing) nor on `k` (range bin, all of the scatterer's range-bin PSF gets the same phase). This means the v5 kernel's amplitude calculation is reused bit-identically; only the scatter index and an extra `exp(j·φ)` factor change.

### 1.5 Rendered tensor shape

```
v5 rasterizer output:   (n_tx=12,    n_rx=16,    n_range=256)          complex
v7 rasterizer output:   (n_chirps=16, n_tx=12,   n_rx=16,    n_range=256) complex
```
Matches the layout of the cascaded GT ADC `cascaded_frame_<F>.npy` after range-FFT. Direct apples-to-apples with GT.

## 2. Directory and file plan

### 2.1 Create `mm25DGS_v7` as a bit-identical copy of `mm25DGS_v5`

```
cp -r /home/adnan/Desktop/mm3DGS/mm25DGS_v5 /home/adnan/Desktop/mm3DGS/mm25DGS_v7
cd /home/adnan/Desktop/mm3DGS/mm25DGS_v7
rm -rf output_frame_nvs output_chirp_loop __pycache__ cuda/build */__pycache__
```
Post-copy the tree contains:
```
mm25DGS_v7/
├── cuda/                    # CUDA extension sources (will be modified — §3.1)
├── preprocessing/alignment/ # pass-1 + pass-2 (kept as-is)
├── rasterizer.py            # diff-renderer Python side (modified — §3.2)
├── rasterizer_factorized.py # KA+SPM factorised primitive (modified — §3.2)
├── train_gaussian.py        # rendering + loss + eval (modified — §4)
├── train_frame_nvs.py       # frame-NVS entry (modified — §4)
├── train_chirp_loop_nvs.py  # chirp-loop entry (kept as-is — v7 Doppler is
│                              native in train_frame_nvs; train_chirp_loop_nvs
│                              stays usable as a 16-chirp physical re-render
│                              reference for validate_doppler_synthesis)
├── load_pretrained.py
└── chirp_loop_report.py, chirp_loop_sweep.py, frame_nvs_report.py,
    material_diagnostics.py, psf.py   # kept as-is
```

### 2.2 What (if anything) we import from v6

**Nothing by default.** The user directive is explicit: v6 is a failed branch and v7 must not depend on it.

However, two v6 artefacts are worth reviewing BEFORE committing to re-implementation:

1. `mm25DGS_v6/data/ra_utils.py::adc_to_rad_complex` — GT-side ADC → |RAD| pipeline (range-FFT + txrx_to_vx + el=0 row + azimuth FFT per chirp + Doppler FFT on chirp axis, all v5-identical FFT chain). The logic is correct and closely matches this plan's convention. **Decision**: port to `mm25DGS_v7/data/ra_utils.py` (pure function, no v6 dependency). This is a file-content re-use, not a structural import.
2. `mm25DGS_v6/scripts/doppler_gate1_rd_cube_viz.py` and `doppler_gate2_spread_histogram.py` — already-passing gating sanity checks. **Decision**: port to `mm25DGS_v7/scripts/` with paths rewritten.

Outside of these two pure-function ports, **v7 imports nothing from v6**.

### 2.3 Where v7 output lands

- Training outputs: `mm25DGS_v7/output_frame_nvs/<scene>_<tag>_v7/`.
- Logs: `logs_v7_<bench>/`.
- Results directory in `data/` and `data/alignment_data/` is consumed, not modified. No new `data_v2/` required.

## 3. Rasterizer changes

### 3.1 CUDA extension (`mm25DGS_v7/cuda/`)

The v5 CUDA extension is the scatter-loop accelerator. v7 modifies it to emit a chirp axis.

**Interface changes:**
- Kernel signature gains two inputs: `v_ego: (3,) float32` and `scalar T_c, T_a`.
- Kernel signature gains one output-shape parameter: `n_chirps`.
- Output tensor shape changes from `(n_tx, n_rx, n_range)` to `(n_chirps, n_tx, n_rx, n_range)`.

**Scatter-step body change (per point `p`, after the existing amplitude compute):**
```
direction_component = dot(u_hat_p, v_ego);                // scalar
phase_per_chirp     = -4π / λ * direction_component * T_c;
phase_per_tdm       = -4π / λ * direction_component * T_a;
for (int m = 0; m < n_chirps; ++m) {
    for (int i = 0; i < n_tx; ++i) {
        float phi_total = m * phase_per_chirp + i * phase_per_tdm;
        complex scale   = {cos(phi_total), sin(phi_total)};
        for (int j = 0; j < n_rx; ++j) {
            // c_0_i_j(k; p) is the v5 per-point-per-virtual amplitude, unchanged
            complex v5_amp = v5_amplitude(p, i, j, k);
            complex v7_amp = complex_mul(v5_amp, scale);
            atomicAdd_complex(out[m, i, j, k], v7_amp);
        }
    }
}
```
Cost:
- Ray test + BSDF evaluation (dominant): **unchanged (1×)**.
- Scatter writes: **16× (n_chirps = 16)**.
- Net per-iter slowdown: the plan's §2.2 estimate of 1.2–1.5× vs v5, empirically verified before scope expansion.

**Autograd**: `phi_total` depends on `u_hat_p` (from FROZEN positions) and `v_ego` (FROZEN per-frame, §5). `scale` has no learnable gradient. Material and rotation gradients flow through `v5_amp` as in v5; Doppler adds NO new learnable parameter.

**LOC budget**: ~120 LOC CUDA + ~30 LOC Python wrapper.

### 3.2 Python rasterizer (`rasterizer.py`, `rasterizer_factorized.py`)

- New `apply_pose_and_doppler(rast, pose, v_ego, T_c, T_a)` wrapper (replaces/supplements v5's `apply_pose`).
- Existing `render_gaussians(...)` returns `(rp_real, rp_imag)` of shape `(16, 12, 16, 256)` instead of `(12, 16, 256)`. Minor Python edits; most of the change is in the CUDA kernel invocation.

**LOC budget**: ~60 LOC across `rasterizer.py` / `rasterizer_factorized.py`.

## 4. Trainer changes

### 4.1 `train_frame_nvs.py`

New per-frame supervision object (one per train frame):
```python
bundle = {
    'frame_idx': int,
    'pose_F':    pose dict,              # chirp-0 / TX-0 pose, from pass-2 config
    'v_ego':     (3,) float tensor,      # precomputed per-frame, §5
    'gt_rad':    (D, 127, R) real,       # GT |RAD| magnitude (computed once per frame)
    'gt_rad_max': scalar,                # for mse_raw normalisation
}
```

Per-iter training loop (pseudocode):
```python
for bundle in train_bundles:
    apply_pose_and_doppler(rast, bundle['pose_F'],
                             v_ego=bundle['v_ego'],
                             T_c=T_C_SECONDS, T_a=T_A_SECONDS)
    rp_r, rp_i = render_gaussians(model, rast, ...)
    # rp_r, rp_i shape: (16, 12, 16, 256)
    rp_c = torch.complex(rp_r, rp_i)
    rad_pred = rp_stack_to_rad_complex(rp_c, n_dop=N_DOP_DEFAULT)  # (D, 127, R) cpx
    mag_pred = rad_pred.abs()
    gt_mag   = bundle['gt_rad']
    gt_max   = bundle['gt_rad_max']
    # v5 mse_raw convention extended to Doppler axis:
    loss = (mag_pred - gt_mag).pow(2).mean() / (gt_max ** 2)
    (loss * loss_scale).backward()
```

**Critical invariants preserved from v5**:
- Test-time `final_test_cc` = v5 cart_corr on chirp-0 |RA|. Unchanged.
- Scene point cloud and visibility mask unchanged.
- target_n = 20000 default.
- 500 iter training budget.
- pass-2 alignment configs for pose source.

**LOC budget**: ~80 LOC in `train_frame_nvs.py` (new bundle builder + new training loop branch).

### 4.2 `train_gaussian.py` (loss + eval utilities)

- Import `rp_stack_to_rad_complex` and `adc_to_rad_complex` from `mm25DGS_v7/data/ra_utils.py`.
- Update `range_profile_to_ra_*` helpers to handle `(16, 12, 16, 256)` input by slicing chirp 0 for backward-compat reports.

**LOC budget**: ~20 LOC.

## 5. Should v_ego be learnable?

**Recommendation: NO. `v_ego` is FROZEN per-frame, computed once from pose-difference.**

### 5.1 Justification

**(a) Gradient scale / instability.** The Doppler phase term is
$$
\phi^{(m,i)}(p) \;=\; -\tfrac{4\pi}{\lambda}\,\langle\hat u(p), v_\mathrm{ego}\rangle\,(m T_c + i T_a).
$$
The partial w.r.t. v_ego component `v_ego[α]` is
$$
\tfrac{\partial\phi}{\partial v_\mathrm{ego}[\alpha]}
\;=\; -\tfrac{4\pi}{\lambda}\,\hat u_\alpha(p)\,(mT_c + iT_a).
$$
At `m = 15, i = 11`, `T_c ≈ 468 μs`, `T_a ≈ 39 μs`, `λ ≈ 3.9 mm`, `|û| ≤ 1`:
$$
\left|\tfrac{\partial\phi}{\partial v_\mathrm{ego}}\right|_\max
\;=\; \tfrac{4\pi}{3.9\times10^{-3}}\,\cdot\,(15\cdot 4.68\times10^{-4} + 11\cdot 3.9\times10^{-5})
\;\approx\; 23\,\mathrm{rad}/(\mathrm{m/s}).
$$
A 0.3 m/s perturbation in v_ego (typical uncertainty at centimeter-level LiDAR-FPS pose accuracy over a 0.2 s pose-difference interval) flips the phase by ≈ 6.9 rad ≈ 1.1 full cycles. **Phase is brittle AND wraps at 2π, so learning v_ego via SGD on a mse_raw-style loss would be catastrophic** for the same reason M1.5 learning pairwise phase was: the optimizer cannot disambiguate wrap-around from signal.

**(b) Amplitude-dominated learning**. v5 trains materials and rotations primarily through amplitude gradients from the mse_raw loss on |RA|. Phase contributes a subtle amplitude-coupled gradient (we validated this in the M1.5 analysis: Fresnel complex coefficient has both magnitude and phase). Adding a large-scale learnable v_ego would inject a DOMINANT phase gradient that competes with and overwhelms the amplitude gradients we rely on.

**(c) v_ego is well-determined from pose-difference**. The aligned pass-2 configs give radar center positions with cm-level accuracy. Pose-difference `v_ego(F) = (T(F+1).pos − T(F−1).pos)/(2·T_frame)` inherits that accuracy → `|δv_ego| ≈ 0.01 m / 0.2 s = 0.05 m/s` — the ≈ 1 rad / (m/s) sensitivity to v_ego translates to ≈ 0.05 rad phase uncertainty at max (m,i). Well within Hann-window leakage; not a limiting factor.

**(d) Learnable v_ego would reduce to pose-refinement**. If we did make it learnable, SGD would mostly be estimating whatever is needed to minimize phase mismatch — i.e. reconstructing the same v_ego we can compute directly in ~5 LOC from pass-2 configs. No added information, only new failure modes.

### 5.2 What we do instead — GT-interpolated v_ego

**IMPORTANT (corrected 2026-04-23 after cross-check).** An earlier draft used pass-2 aligned configs for v_ego estimation; empirical audit showed pass-2 inter-frame displacements are ~2–4× noisier than ColoRadar's published ground-truth trajectory (pass-2 was optimised for per-frame radar-to-scene alignment, not velocity estimation). Worse, the draft used `T_frame = 0.1 s` — the ColoRadar cascade actually runs at **5 Hz (`T_frame = 0.2 s`)**, so pass-2-derived v_ego was additionally 2× inflated.

v7 uses the raw ColoRadar groundtruth trajectory, interpolated at the exact cascade-frame timestamps.

**Pipeline**:
```python
import numpy as np, os

def v_ego_for_frame(scene: str, frame: int,
                      raw_seq_root: str) -> np.ndarray:
    """Estimate v_ego(F) = (GT_pos(t[F+1]) - GT_pos(t[F-1])) / (t[F+1] - t[F-1])
    using raw ColoRadar groundtruth trajectory + cascade timestamps.

    raw_seq_root points at e.g. /home/adnan/Documents/Data/coloRadar/raw/kitti/
                                 2_28_2021_outdoors_run<seq>/
    """
    cas_ts = np.loadtxt(os.path.join(
        raw_seq_root, 'cascade', 'adc_samples', 'timestamps.txt'))
    gt     = np.loadtxt(os.path.join(
        raw_seq_root, 'groundtruth', 'groundtruth_poses.txt'))        # (N, 7): xyz + quat
    gt_ts  = np.loadtxt(os.path.join(
        raw_seq_root, 'groundtruth', 'timestamps.txt'))
    pm = np.array([np.interp(cas_ts[frame - 1], gt_ts, gt[:, i])
                    for i in range(3)])
    pp = np.array([np.interp(cas_ts[frame + 1], gt_ts, gt[:, i])
                    for i in range(3)])
    Δt = cas_ts[frame + 1] - cas_ts[frame - 1]                       # ≈ 0.4 s
    return (pp - pm) / Δt                                             # (3,) m/s
```

Cached per-frame v_ego lives at `data/v_ego/<scene>/frame_<F>_v_ego.npy` (3,) float32 — computed once at preprocessing-time and loaded by the trainer.

**Edge frames** (F ± 1 outside `cas_ts` bounds): one-sided differenceF. For our standard HO_8 F±4 training window, all 8 train frames have both `F ± 1` neighbours within the full 500+-frame cascade trajectory, so one-sided fallback is never hit in practice.

**Empirical check** (executed during plan drafting): GT-interpolated |v_ego| across the 6 benchmark scenes:

| Scene | GT \|v_ego\| | \|f_d\|_max (unaliased) | Aliases? |
|---|---:|---:|---:|
| seq_0_frame_135 | 1.22 m/s (4.4 km/h) | 626 Hz | no |
| seq_1_frame_185 | 1.40 m/s (5.0 km/h) | 716 Hz | no |
| seq_1_frame_438 | 1.19 m/s (4.3 km/h) | 611 Hz | no |
| seq_2_frame_105 | 1.34 m/s (4.8 km/h) | 686 Hz | no |
| seq_2_frame_160 | 1.35 m/s (4.9 km/h) | 692 Hz | no |
| seq_2_frame_300 | 1.44 m/s (5.2 km/h) | 741 Hz | no |

**All 6 scenes sit well below the unambiguous velocity limit of 1.98 m/s.** The Gate-1 aliasing signature reported earlier was a consequence of the buggy v_ego, not physical aliasing. **Risk R4 (aliasing-reduced effective DOF) is substantially mitigated for v7.**

**Future extension (not in v7 first pass)**: an OPTIONAL `--allow_v_ego_jitter` flag can add small isotropic Gaussian noise to `v_ego` during training to regularise against over-fitting. Not implemented initially.

## 6. GT `|RAD|` pipeline

Ported from `mm25DGS_v6/data/ra_utils.py::adc_to_rad_complex` to
`mm25DGS_v7/data/ra_utils.py`. The chain, per-chirp:

```
ADC  →  Hann(range-axis) × range-FFT  →  txrx_to_vx_chirps  →  keep el=0 row
     →  Hann(az-axis) × ifftshift(az) × FFT(az, n=128) × drop bin[0] × fftshift(az)
     →  (127, 256) complex per chirp
```
Then Doppler across the 16-chirp axis:
```
Hann(chirp) × ifftshift(chirp) × FFT(chirp, n=N_DOP=32) × drop bin[0] × fftshift(chirp)
     →  (D = N_DOP - 1 = 31, 127, 256) complex.
```
Per user directive on 2026-04-23: N_DOP = 32, matching v5 azimuth zero-pad ratio of 128/86 ≈ 1.5×. Drop-bin-0 is the SAME v5 convention, applied on the chirp axis.

Magnitude → `(31, 127, 256)` real. GT loss normaliser = `gt_mag.amax()` (v5 mse_raw convention).

## 7. Differentiation from DART

| Aspect | DART (Chen et al. 2024) | v7 Doppler |
|---|---|---|
| Forward primitive | Implicit voxel-indexed occupancy + signed-distance fields | Explicit point-scatterers with KA+SPM mmWave-Jones BSDF, frozen LiDAR-FPS positions |
| Scene rep | Continuous implicit neural field | Discrete 20k-point scatterer set with per-point (mat, rot) |
| Rendering output | Range-Doppler heatmap directly | Per-virtual complex range profile × chirp axis (pre-azimuth-FFT) |
| Azimuth structure | Baked into RD heatmap projection | Preserved as v5's azimuth-FFT on 86-virt subset |
| Ego-velocity modelling | Implicit in trajectory supervision | Explicit analytic phase term in forward model (this plan) |
| Loss | Range-Doppler heatmap MSE | v5 mse_raw extended to 3D |RAD| magnitude |
| Alignment | Joint optimisation with poses | Two-stage external alignment (pass-1 + pass-2) + frozen rendering |

Differentiators for publication: (i) per-virtual complex forward model (not RD-heatmap), preserving the angular degrees of freedom for future extensions; (ii) KA+SPM physics-based BSDF with per-point ITU material parameters, enabling material transfer; (iii) pose-aligned from pass-2 cascaded-alignment pipeline; (iv) no learnable v_ego — pose-difference is exact; (v) explicit TDM phase correction.

## 8. Validation

### 8.1 Pre-implementation gating (already passing)

Ports of `mm25DGS_v6/scripts/doppler_gate{1,2}_*.py`. Pre-confirmed on seq_0_frame_135, seq_1_frame_438, seq_2_frame_160:
- Gate 1: RD-cube energy spreads 5–7 Doppler bins (not collapsed).
- Gate 2: 16/16 wrapped Doppler bins occupied with top-bin share 11–14%.

### 8.2 `validate_doppler_synthesis.py`

Three tests, in order of diagnostic specificity. **All three must pass** before launching any training run.

#### 8.2.1 Single-point-scatterer analytic test (addresses risk R2)

The sharpest TDM-order / sign-convention test. Construct a synthetic single-scatterer scene at a known 3D position `r_s`, fire v7's renderer at chirp-0 pose with a known `v_ego`, and verify each of the `16 × 12 = 192` per-(m, i) complex outputs matches the ANALYTIC formula bit-closely:
$$
c^{(m,i)}_{j}(k;\,p) \;=\; c^{(0,0)}_{i,j}(k;\,p)\,\exp\!\bigl(j\,\phi^{(m,i)}(p)\bigr)
$$
with
$$
\phi^{(m,i)}(p) \;=\; -\tfrac{4\pi}{\lambda}\,\langle\hat u(p), v_\mathrm{ego}\rangle\,(m T_c + i T_a).
$$

Pass criteria (per (m, i) cell):
- Phase error `|arg(v7[m,i]) − arg(v5_ref) − φ^(m,i)| < 1e-3 rad`.
- Amplitude error `||v7[m,i]| − |v5_ref|| / |v5_ref| < 1e-3`.

**This test catches** TX-firing-order bugs (CONFIG_TX_TO_ADC_TX_PERM mismatches), direction-sign errors, T_c-vs-T_a confusion, and any single-point arithmetic bug in the CUDA scatter loop.

#### 8.2.2 Multi-point cross-check vs 16-physical-render reference (regression test)

**Reference (exact-physics)**: 16 calls to v5's `render_gaussians` at 16 LERP-interpolated per-chirp poses (the v6 M2 approach). Output: `(16, 12, 16, 256)` complex.

**Test (v7 analytic)**: 1 call to v7's `render_gaussians_doppler` at chirp-0 pose with v_ego computed from pass-2 poses. Output: `(16, 12, 16, 256)` complex.

**Pass criterion**:
- Complex normalised correlation over the full cube ≥ 0.90.
- Top-50-peak-bin agreement within ±1 bin on both range and Doppler axes.

If 0.90 ≤ cc < 0.95, suspected root causes (in order of investigation): (a) TDM-phase sign or ordering (rerun test 8.2.1), (b) `v_ego` coordinate-frame mismatch between poses and scatterer positions, (c) amplitude-variation approximation (B1 ignores amplitude changes across the burst, which for < 4 cm motion should be < 10⁻³ fractional). If cc < 0.90, do not launch training; debug in this order.

#### 8.2.3 `v_ego = 0` bit-identity test (addresses risk R7)

When `v_ego = 0`, the Doppler phase `φ^(m,i)(p) = 0 ∀ (m, i, p)`, so v7's output must reduce to 16 identical copies of v5's chirp-0 render. Pass criterion:
- For any `(m, i)` slice: `v7[m, i, :, :]` bit-identical to `v5_ref[i, :, :]` (max abs diff < 1e-6).
- `final_test_cc` computed on `v7[0, 0, :, :].abs()` matches `final_test_cc` from a v5 run by < 0.01 per scene.

This test is a belt-and-braces check that the Doppler-augmented code path doesn't corrupt the non-Doppler fallback.

### 8.3 Training bench

After validation:
- **Run v7 on 6 scenes**: seq_0_frame_135, seq_1_frame_185, seq_1_frame_438, seq_2_frame_105, seq_2_frame_160, seq_2_frame_300 (the 6 with comparable v5/M1 data-tree baselines). F±4 HO_8 window.
- **Report three metrics per scene, not one**:
  - `final_test_cc` = v5-identity metric: cart_corr on `|RA|(chirp=0)` (unchanged from v5).
  - `final_test_cc_RAD` = new: cart_corr on the summed-over-Doppler `|RA|` reconstruction (i.e. `Σ_d |RAD(d, :, :)|` → cart → cart_corr vs GT-derived equivalent).
  - `train_mean_cc` = v5-style training cart_corr (chirp-0 RA).

  **Rationale (R1 mitigation)**: v5's training objective and eval metric coincide on `|RA|`. v7's training objective is `|RAD|`, and the eval metric `final_test_cc` is only a PROJECTION of that. The two can diverge. By reporting both `final_test_cc` and `final_test_cc_RAD` we learn whether:
  - v7 is improving on its own training objective (`_cc_RAD` rising) but missing v5's evaluation axis (`final_test_cc` flat or dropping).
  - v7 is improving on BOTH (real win).
  - v7 is regressing on both (method/physics broken).

- **Pass criterion (R4 mitigation — revised from "≥ v5 + 0.02 mean" to a tiered ladder)**:

  | Tier | Criterion | Conclusion |
  |---|---|---|
  | **GOLD** | `final_test_cc` mean ≥ v5/M1 + 0.02, all scenes ≥ v5/M1 | v7 wins; the plan was right. Publishable win. |
  | **SILVER** | `final_test_cc` mean ≥ v5/M1 + 0.00 on ≥ 4/6 scenes AND seq_1_frame_438 strictly improves (≥ +0.03) | Doppler adds useful signal on scenes where naive-avg fails. Conservative but real win. |
  | **BRONZE** | `final_test_cc` flat or −0.02, BUT `final_test_cc_RAD` mean ≥ v5/M1 `_cc_RAD` + 0.02 | v7 learns the Doppler axis correctly but the evaluation metric doesn't reward it. Consider the hybrid-loss fallback (§8.4). |
  | **NULL** | Both metrics regress by > 0.02 mean | Diagnose before iterating. Likely causes: validation was incomplete (rerun 8.2.1-8.2.3), or the Doppler-aliasing-reduced-DOF concern (R4) is dominant. If diagnosis points to aliasing, §10 rollback triggers. |

  **SILVER or GOLD** are realistic expectations now that R4 (Doppler aliasing) is substantially mitigated — GT-interpolated v_ego shows all 6 scenes run at walking-pace below the 1.98 m/s unambiguous limit. Aliasing is no longer the expected DOF-limiter; if bench regresses it's likely forward-model bugs (§8.2 catches these) or benchmark saturation (R8). BRONZE remains the hybrid-fallback path; NULL remains the reformulation-trigger.

### 8.4 Hybrid-loss fallback (contingency for BRONZE outcome)

If 8.3 lands in BRONZE (v7 training objective improves but eval metric does not), the natural next step is a hybrid loss that anchors the v5 objective while still training on the Doppler axis:
$$
L_\mathrm{hybrid} \;=\; \alpha\,L_\mathrm{v5}(|\mathrm{RA}|(\mathrm{chirp}=0)) \,+\, \beta\,L_\mathrm{v7}(|\mathrm{RAD}|)
$$
with `α, β ∈ [0, 1]` and `α + β = 1` (or unconstrained with a log-search). Grid: `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`. `α = 1` recovers v5; `α = 0` recovers pure v7; intermediate values trade off train-eval alignment vs Doppler signal.

Implemented only if 8.3 lands BRONZE — ~20 LOC in `train_frame_nvs.py`, 1-hour sweep per scene.

### 8.5 Unit correctness checks

1. `v_ego` magnitude per scene: print to confirm 2–5 m/s range and stability across frames.
2. Per-point `|f_d(p)|` histogram per scene: confirm ≥ 4 wrapped bins occupied.
3. `final_test_cc` bit-identity with v5 when `v_ego = 0` (Doppler phase trivially zero) — **already covered by 8.2.3**.

## 9. LOC + wallclock budget

| Phase | LOC | Wallclock |
|---|---:|---|
| 2.1 Copy v5 → v7 | 0 (shell) | 5 min |
| 3.1 CUDA kernel mod | 120 | ½ day |
| 3.2 Python rasterizer mod | 60 | 2 hours |
| 4.1 Trainer integration | 80 | 3 hours |
| 4.2 Helpers + `ra_utils.py` port | 80 | 2 hours |
| 5 `v_ego` preprocessing | 60 | 1 hour |
| 8.2 `validate_doppler_synthesis.py` | 100 | 3 hours |
| Gate-check ports | 60 | 1 hour |
| First smoke run (1 scene, 100 iter) | 0 | ½ hour |
| 6-scene bench (500 iter × 6) | 0 | 1–2 hours on 2 GPUs |
| **Total** | **~560 LOC** | **~3 eng-days** |

## 10. Gating decision + rollback

### 10.1 Pre-training gate

Proceed with v7 training **only if** all three validation tests in §8.2 pass:
- 8.2.1 single-point analytic test (per-cell phase / amplitude within tolerance).
- 8.2.2 multi-point reference cross-check (complex cc ≥ 0.90 vs 16-physical-render reference).
- 8.2.3 `v_ego = 0` bit-identity (v7 reduces to v5 when Doppler is turned off).

If any fails:
- Rerun 8.2.1 with targeted diagnostic prints (per-cell phase expectation vs observed) to localise sign/order bug.
- Option fallback: if only 8.2.2 fails (but 8.2.1 and 8.2.3 pass), the residual gap is probably the B1 amplitude-constant approximation; quantify magnitude of the discrepancy. If it's a uniform, direction-insensitive noise floor < 5%, proceed; otherwise dig deeper.
- Last-resort fallback: revert to Option #1 (16 physical renders). v7 code can still do this by calling its renderer 16 times with `v_ego = 0` and manually interpolated poses — effectively reverting to the v6 M2 behaviour. This is a regression but a known-working one.

### 10.2 Post-training decision ladder (§8.3 tiers)

| Outcome | Action |
|---|---|
| **GOLD** | Publish. |
| **SILVER** | Log as partial win. Consider extending v7 with curriculum / TDM tuning for a stronger result. |
| **BRONZE** | Run the hybrid-loss sweep (§8.4). If hybrid lands SILVER or better, publish hybrid. |
| **NULL** | **Stop iterating in v7.** Diagnose via §8.5 + `diag_*` prints per scene. Likely roots: Doppler aliasing (R4) — inherent to our sampling, cannot be fixed by method changes — or benchmark saturation (R8) — requires benchmark reformulation, not a v7 tweak. |

The NULL outcome is the plan's honest worst-case: even a physics-correct Doppler forward model may not improve `final_test_cc` on a benchmark dominated by scene-consistency. In that case the next step is NOT further v7 surgery but **reformulation of the eval metric** (e.g., use `final_test_cc − naive_avg_cc` as the headline number; evaluate on harder F-2 / F+2 test-frame positions per Q2 findings; weight scenes by how much they actually need a renderer).

Note: R4 (aliasing) is no longer a primary concern given GT-interpolated v_ego lands all 6 scenes ~1.2–1.5 m/s (walking) — well below the 1.98 m/s unambiguous limit. If NULL triggers, aliasing is not the expected root cause; focus diagnostics on R1 (loss-metric mismatch) and R8 (benchmark saturation).

## 11. Out-of-scope (intentional)

- Doppler gating ("DART-inspired M3" in the v6 plan). Not attempted in v7.
- SSIM loss (v6 M4). Not attempted.
- Learnable v_ego (§5).
- Per-chirp pose refinement (pass-3). Not used.
- Per-primitive scale / 3DGS surfels. Not touched.
- New BSDF or material model. KA+SPM ITU-concrete-initialised, same as v5.
- Per-frame appearance embeddings.
- Multi-frame consistency loss.
- Any data_v2 / extended-training-window preprocessing.

## 12. Summary of decisions (TL;DR)

1. **Create mm25DGS_v7 as a clean copy of v5**. No v6 dependency except two pure-function ports (RAD-cube utility + gate scripts).
2. **Render output shape**: `(n_chirps=16, n_tx=12, n_rx=16, n_range=256)` complex per call, matching GT ADC layout.
3. **Doppler is analytic**, as a per-point per-chirp per-ADC-TX phase factor in the CUDA scatter kernel. No 16× rendering cost.
4. **TDM intra-burst phase included from day 1** using the correct TI firing order `k(i) = TI_FIRING_INDEX_FROM_ADC_CH[i]` (§1.3). Accounts for TI's TX-labelling convention which is *reversed* relative to our config labels.
5. **`v_ego` is FROZEN per-frame**, computed from the raw ColoRadar groundtruth trajectory interpolated at cascade timestamps (NOT from pass-2 aligned configs — those are ~2–4× noisier for velocity). Empirical GT |v_ego| is 1.19–1.44 m/s across the 6 benchmark scenes, all below the 1.98 m/s unambiguous limit → no aliasing (R4 mitigated).
6. **Loss**: v5 `mse_raw` extended to 3D `|RAD|` cube. No Gram, no hybrid — v6's failed loss space is explicitly not revisited.
7. **Baseline reference**: v5/M1 on `data/` tree. data_v2 not used.
8. **Gating check**: three-tier `validate_doppler_synthesis.py` (single-point analytic, multi-point reference, v_ego=0 identity) must ALL pass before launching the training bench.
9. **T_frame = 0.2 s** (cascade runs at 5 Hz, corrected from an earlier 0.1 s assumption).

Awaiting approval to proceed with implementation.
