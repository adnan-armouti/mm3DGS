# Per-range-bin Gram matrix supervision vs azimuth-FFT magnitude — derivation

Date: 2026-04-19
Audience: anyone validating whether Option C (Gram-matrix supervision)
is consistent with the v5 azimuth-FFT magnitude approach, and by how
much more information it carries.

---

## 1. Setup and notation

Consider a single frame, a single range bin `r`, and a single chirp (no
Doppler for the moment — we add that in §9). After range FFT and MIMO
channel combining, the per-virtual-antenna complex response at range
bin `r` is a complex vector

$$
v \in \mathbb{C}^{N}, \qquad v_n = v[n], \quad n = 0, \ldots, N-1
$$

where `n` indexes virtual antennas (pairs of TX × RX combinations) and
`N` is either

- `N = 86` for the FFT path (unique azimuth positions at elevation = 0,
  with duplicates *averaged* before the FFT), or
- `N = 192` for any "use all virtuals" approach (every TX × RX pair
  kept as a separate channel).

When we need to emphasise range-bin dependence we write `v_r[n]` so
`v_r \in \mathbb{C}^N` is the per-virtual complex response at range bin
`r`. The full frame is the stack `V \in \mathbb{C}^{N \times R}` with
`V[n, r] = v_r[n]` and `R = 256` range bins.

A single DFT matrix `F \in \mathbb{C}^{K \times N}` is defined via

$$
F_{kn} = e^{-j\frac{2\pi}{K} n k}, \qquad k = 0, \ldots, K-1.
$$

For our setup with `K = 128` (azimuth bins, pre-drop-first-bin), this
is the operator the current code applies after an `ifftshift` + taper;
the taper and shift are a real-diagonal and a permutation and do not
change the information content of the argument. We absorb them into
`F` without loss of generality.

---

## 2. Current approach (v5 azimuth-FFT, magnitude, MSE)

The v5 GT pipeline:

1. Pack 192 (TX, RX) pairs into the elevation-0 row, averaging the
   ~106 duplicate azimuth positions → `v \in \mathbb{C}^{86}` per range.
2. Apply taper + ifftshift + DFT → `\hat v = F v \in \mathbb{C}^{K=128}`.
3. Drop bin 0, fftshift → `\hat v' \in \mathbb{C}^{K'=127}`.
4. Take magnitude → `|\hat v'| \in \mathbb{R}^{127}`.
5. Repeat for every range bin.
6. Loss: element-wise MSE between predicted and GT magnitudes
   (optionally normalised).

The supervisory **information** at one range bin is the 127-element
real vector `|\hat v'|`. Equivalently, the loss gradient can only
correct features of the signal that change `|\hat v'|`. It is **blind**
to any component of `v` that lies in the kernel of the "DFT, then
magnitude" operator.

---

## 3. Option C — per-range-bin Gram matrix supervision

Define

$$
C_r \;=\; v_r v_r^H \;\in\; \mathbb{C}^{N \times N}.
$$

`C_r` is **Hermitian positive semi-definite of rank 1** with entries

$$
C_r[i, j] = v_r[i] \, \overline{v_r[j]}.
$$

Each off-diagonal entry is the **complex cross-correlation** of
virtual antenna `i` with virtual antenna `j` at range `r` — i.e. it
captures the relative magnitude and relative phase between the two
virtuals. The diagonal entries are the per-virtual squared
magnitudes `|v_r[n]|^2`.

Supervision quantity: `\{C_r\}_{r=0}^{R-1}` — a tensor of shape
`(R, N, N)` Hermitian.

**Dimension counts** (real DOF per range bin):
- `|\hat v'|`: `K' = 127` real numbers.
- Full `C_r` Hermitian rank-1: for a general Hermitian matrix, `N²`
  real DOF; constrained to rank-1, it is `2N-1` complex DOF =
  `2(2N-1)` real DOF... but as a rank-1 outer product
  `v v^H` it is **specified by `v` up to a global phase** — so the
  intrinsic DOF is `2N - 1` real. For `N = 192`, that is 383 real
  DOF vs 127 for the FFT magnitude.

So **Option C carries ≈ 3× more real information per range bin** than
the v5 FFT-magnitude for the full-192 case, and strictly more than
`K' = 127` for any N > 64.

This counting is a conservative lower bound because it treats `C_r`
as rank-1. In practice, the network's predicted `v_{pred,r}` has
noise; training stability typically benefits from the Gram matrix
being implicitly treated as Hermitian PSD rather than forced rank-1,
which raises the effective DOF.

---

## 4. The key identity: FFT-magnitude² = DFT of Gram diagonal sums

This is the Wiener-Khinchin relation written out at signal level.

Start from the definition of `\hat v[k]`:

$$
\hat v[k] \;=\; \sum_{n=0}^{N-1} v[n] \, e^{-j\frac{2\pi}{K} n k}.
$$

Compute the power spectral density:

$$
\begin{aligned}
|\hat v[k]|^2
&= \hat v[k] \cdot \overline{\hat v[k]} \\
&= \Bigg(\sum_m v[m] e^{-j\frac{2\pi}{K} m k}\Bigg)
   \Bigg(\sum_n \overline{v[n]} e^{+j\frac{2\pi}{K} n k}\Bigg) \\
&= \sum_m \sum_n v[m]\overline{v[n]}
           e^{-j\frac{2\pi}{K} (m - n) k} \\
&= \sum_m \sum_n C[m, n]
           e^{-j\frac{2\pi}{K} (m - n) k}.
\end{aligned}
$$

Group by the lag `d := m - n`:

$$
|\hat v[k]|^2
\;=\; \sum_{d = -(N-1)}^{N-1}
      \Bigg[\sum_{m: m, m-d \in [0, N)} C[m, m-d]\Bigg]
      \cdot e^{-j\frac{2\pi}{K} d k}.
$$

Define the **lag-sum** (aka "diagonal sum" of the Gram matrix):

$$
p[d] \;:=\; \sum_{m} C[m, m - d].
$$

Then

$$
\boxed{\;|\hat v[k]|^2 \;=\; \sum_d p[d] \, e^{-j\frac{2\pi}{K} d k}
   \;=\; (\text{DFT of } p)[k].\;}
$$

**Interpretation.** `|\hat v[k]|^2` is the DFT of the diagonal-summed
Gram matrix. Equivalently, the azimuth power spectrum is the Fourier
transform of the autocorrelation sequence `p[d]` — this is the
Wiener-Khinchin theorem specialised to finite arrays.

A picture:

```
Gram matrix C (N × N)           its diagonals        |\hat v|²  (K)
┌─ ─ ─ ─ ─ ─ ─ ─┐             d = N-1: C[N-1, 0]      DFT
│ ·  d=+1 d=+2  │                   ...               ───→
│d=-1 ·  d=+1  ·│             d = 0:   diag(C)
│d=-2 d=-1 ·   ·│             d = -1:  C[0, 1], ...
│ ·  ·   ·   · │                   ...
└─ ─ ─ ─ ─ ─ ─ ─┘             d = -(N-1): C[0, N-1]
```

The Gram matrix has `N²` entries (`≈ 2N² - N` real DOF after
Hermiticity); the lag-sum sequence `p[d]` has `2N - 1` complex entries
(Hermitian so ≈ `N` real DOF with symmetry). The DFT of `p` has `K`
complex entries, real-valued because `p` is Hermitian in d.

**Information-loss ratio**: the map `C → |\hat v|²` loses

$$
\frac{N^2 - N}{N^2 - N}  \;-\;
\frac{\text{entries of } p}{\text{entries of } C}
\approx 1 - \frac{2N-1}{N^2} \longrightarrow 1 \text{ as } N \to \infty.
$$

For `N = 192`: roughly `1 - 383/36864 ≈ 99%` of the information in `C`
is **thrown away** by going from `C` to `|\hat v|²`. Only the shift-
invariant statistic (diagonal sums) survives.

---

## 5. Consistency: the FFT magnitude is a linear projection of the Gram matrix

From §4:

$$
|\hat v[k]|^2
\;=\; \sum_{m,n} C[m,n] \cdot e^{-j\frac{2\pi}{K}(m-n)k}
\;=\; \langle C, \, D_k \rangle_{\text{Frobenius}},
$$

where `D_k \in \mathbb{C}^{N \times N}` is the rank-1 matrix

$$
D_k[m, n] \;=\; e^{-j\frac{2\pi}{K}(m-n)k}.
$$

So `|\hat v[k]|^2` is a **linear functional** (Frobenius inner
product) of the Gram matrix `C`. The whole azimuth magnitude spectrum
`\{|\hat v[k]|^2\}_{k}` is a **set of `K` linear projections** of `C`.

**Therefore** any function of the FFT-magnitude-squared spectrum
(including the current v5 MSE loss on `|\hat v|`, since `|\hat v|²` is
monotone in `|\hat v|`) is a function of `C` composed with linear
projection — i.e. of the form `L(PC)` for some linear operator `P`.

The Gram-matrix loss `\|C_{pred} - C_{gt}\|_F` is `L_0(C)` where `L_0`
is the full Frobenius distance — it is a **strict generalisation** of
anything built from `|\hat v|`.

**Consistency property**: if two rank-1 Hermitian matrices `C_1` and
`C_2` satisfy `\|C_1 - C_2\|_F = 0`, then necessarily `PC_1 = PC_2` for
any linear operator `P`, so `|\hat v_1|² = |\hat v_2|²`. The reverse is
false: `|\hat v_1|² = |\hat v_2|²` does NOT imply `C_1 = C_2`. There
exist many `v`'s with the same FFT-magnitude spectrum but different
pairwise phase patterns — those are exactly the things current training
is blind to.

---

## 6. Absolute-phase invariance of both approaches

Define the global phase rotation `v' := e^{j\alpha} v`. Then:

**FFT magnitude**:

$$
|\hat v'[k]| = |F v'|[k] = |e^{j\alpha} F v|[k] = |\hat v[k]|.
$$
Invariant. ✓

**Gram matrix**:

$$
C' = v'(v')^H = e^{j\alpha} v (e^{j\alpha} v)^H
   = e^{j\alpha} v v^H e^{-j\alpha} = v v^H = C.
$$
Invariant. ✓

**Both are equally immune to global absolute phase.** The user's
constraint "we do not care about absolute phase" is fully satisfied by
Option C, no extra trick needed.

---

## 7. Per-range-bin phase — kept or cancelled?

A separate question from "absolute phase" is "per-range-bin phase":
should the supervision enforce a consistent absolute phase at each
range bin?

- **In the current FFT approach**: per-range-bin phase IS kept
  (the magnitude at range `r` is the magnitude of the azimuth
  spectrum at that specific range; phase within `\hat v_r` is absolute
  and enters the magnitude indirectly via the Fourier relation).
  Actually this is wrong: magnitude discards phase entirely. The
  per-range-bin phase is **thrown away** by |·|.

- **In Option C with a full `||C_pred - C_gt||_F` loss**:
  per-range-bin phase at the virtual-array level is thrown away
  (Gram cancels it globally). **But** inter-range-bin phase relationships
  are preserved only if we also consider cross-range entries — which
  we don't, in the per-range-bin Gram formulation.

If you want inter-range phase correlations too, enlarge the Gram to
the full `(N·R, N·R)` matrix — that is Option 4 in the previous
message. Usually this is overkill; the per-range `C_r` is sufficient
and is what we'll use.

---

## 8. Spatial interpretation — relative phase IS azimuth information

Here is the most important point for motivation. For virtuals on a
uniform linear array at positions `x_n = n \Delta` (with `\Delta =
\lambda/2`) and a plane wave from angle `\theta`:

$$
v_r[n] = s_r \cdot e^{+j\frac{2\pi}{\lambda} n \Delta \sin\theta},
$$

where `s_r` absorbs range-dependent and material-dependent amplitude.
Then

$$
C_r[m, n] = s_r \overline{s_r}
            \cdot e^{+j\pi (m-n) \sin\theta},
$$

i.e. the Gram entry encodes exactly the angular phase difference at
spatial lag `(m-n)`. For a non-uniform array (our actual layout),
replace `(m - n)\Delta` by `x_m - x_n`; the principle is the same.

**So the off-diagonal entries of the Gram matrix ARE the spatially
decomposed azimuth information** (plus elevation for pairs that
straddle elevation rows). There is no need to beam-form first. In
fact, beam-forming then taking the magnitude is a lossy summary of
the Gram matrix; the user's intuition "I want spatial decomposition"
is already satisfied by C's `(i, j)` structure.

For our MMWCAS layout, virtuals have positions in both azimuth
(x-axis) and elevation (z-axis). The Gram matrix naturally handles
this: entries `C[m, n]` where `m, n` straddle the elevation-0 row
encode azimuth phase differences; entries straddling elevation rows
encode elevation phase differences. **No a-priori choice of azimuth
vs elevation binning is needed.**

---

## 9. Adding Doppler (slow-time FFT)

Chirps across the 16-chirp frame can be coherently integrated into a
Doppler axis. Stacking per-chirp complex vectors into
`V_{chirp}[c, n, r]` and applying a slow-time FFT over chirps yields

$$
\tilde V[d, n, r] \;=\; \sum_c V_{chirp}[c, n, r] \, e^{-j\frac{2\pi}{D} c d}.
$$

Now the supervision tensor becomes
`\tilde V \in \mathbb{C}^{D \times N \times R}` with D=16.

The Gram matrix can now be computed per (Doppler, range) bin:

$$
C_{d, r} \;=\; \tilde V[d, :, r] \cdot \tilde V[d, :, r]^H \;\in\; \mathbb{C}^{N \times N}.
$$

Option C + Doppler = `(D, R, N, N)` Hermitian tensor. With
`D = 16, R = 256, N = 192`, that is nominally `16 \cdot 256 \cdot
192^2 \cdot 8` bytes ≈ 18 GB — way too big to store.

**BUT** — see §10 — we never materialise the Gram matrices. The loss
has a closed form in `V` and `\tilde V` directly.

---

## 10. Practical loss formulation (no Gram materialisation)

The "Gram-matrix Frobenius² loss" at one (d, r) bin is:

$$
\|C_{d,r}^{pred} - C_{d,r}^{gt}\|_F^2
\;=\; \|v_p v_p^H - v_g v_g^H\|_F^2,
$$

with `v_p := \tilde V_{pred}[d, :, r]` and likewise `v_g`. Expand:

$$
\begin{aligned}
\|v_p v_p^H - v_g v_g^H\|_F^2
&= \mathrm{tr}\bigl[(v_p v_p^H - v_g v_g^H)
                    (v_p v_p^H - v_g v_g^H)\bigr] \\
&= \mathrm{tr}(v_p v_p^H v_p v_p^H)
 - \mathrm{tr}(v_p v_p^H v_g v_g^H)
 - \mathrm{tr}(v_g v_g^H v_p v_p^H)
 + \mathrm{tr}(v_g v_g^H v_g v_g^H) \\
&= \|v_p\|^4
 - 2\, |v_p^H v_g|^2
 + \|v_g\|^4.
\end{aligned}
$$

(Used `\mathrm{tr}(a b^H a b^H) = |a^H b|^2` and
`\mathrm{tr}(a a^H a a^H) = \|a\|^4`.)

So the per-(d, r) loss is

$$
\boxed{\;L_{d,r}
  \;=\; \|v_p\|^4 \;-\; 2\,|v_p^H v_g|^2 \;+\; \|v_g\|^4.\;}
$$

For a whole frame, sum over `(d, r)`:

$$
L_{\text{frame}}
  \;=\; \sum_{d,r} \bigl[\|v_p\|^4 - 2|v_p^H v_g|^2 + \|v_g\|^4\bigr].
$$

**This is O(N·D·R) work, fully vectorisable.** Never builds the
`N × N` Gram matrix explicitly.

### Normalised variants

For loss-landscape stability, normalise by the magnitudes:

$$
L^{\text{norm}}_{d,r}
\;=\; 1 \;-\; \frac{|v_p^H v_g|^2}{\|v_p\|^2 \|v_g\|^2}.
$$

Each term is in `[0, 1]`: 0 iff `v_p = e^{j\alpha_{d,r}} v_g` (the
two virtual-array vectors agree up to a per-(d,r) phase). The per-
(d,r) phase freedom is exactly the "absolute phase" we explicitly
do not care about. **This is the cleanest form of Option C as a
training loss.**

Sum over (d, r) for the frame loss:

$$
L^{\text{norm}}_{\text{frame}}
\;=\; \sum_{d,r} \Bigg[1 - \frac{|v_p[d, :, r]^H v_g[d, :, r]|^2}
                             {\|v_p[d,:,r]\|^2 \|v_g[d,:,r]\|^2}\Bigg].
$$

### Gradient structure

`|v_p^H v_g|^2 = (v_p^H v_g)(v_g^H v_p)` is quadratic in `v_p`, so
its gradient in `v_p` is straightforward:

$$
\nabla_{v_p} |v_p^H v_g|^2 \;=\; 2 (v_g^H v_p) v_g.
$$

PyTorch handles the complex autograd via the
`abs`-on-complex convention (via
`torch.view_as_real` or the native complex autograd path); both are
tested. Gradients flow into rotations + raw_materials through
the renderer exactly as in v5.

---

## 11. Consistency summary (one table)

| Property | v5 FFT magnitude (current) | Option C (Gram) |
|---|---|---|
| Uses virtuals | 86 (elevation-0 row only; duplicates averaged) | all 192 (or all N) |
| Phase used during training | Yes — the FFT consumes relative phases across virtuals to produce angular spectra; taking magnitude preserves angular peak locations ⇒ phase IS supervised (projected onto mag-subspace) | Yes — full per-(i, j) pairwise relative phase |
| Real DOF per range bin | `K' = 127` (mag-subspace projection) | `2N - 1 ≈ 383` (rank-1 Gram) |
| Absolute phase | Invariant ✓ | Invariant ✓ |
| Per-range-bin phase | Thrown away by |·| | Thrown away under normalised form; kept under un-normalised form |
| Spatial decomposition | Yes (DFT basis) | Yes (per-(i, j) basis; finer) |
| Is v5 FFT a projection of this? | — | **Yes** (§5, eq. for linear functional) |
| Includes elevated pairs? | No (v5 drops them) | Yes, naturally |
| Requires a beam-form grid? | Yes (K azimuth bins) | No |

**Strict consistency**: everything v5's FFT-magnitude can supervise
can be derived from the Gram. Option C is a superset supervision
signal — it cannot be *less* informative than the v5 signal. But
the "more" it supervises is the complement of the
magnitude-after-FFT operator's range. **That complement mixes
physically useful info** (e.g. elevated-TX virtuals the 86-subset
FFT drops, per-range-bin inter-virtual phase encoding multi-path
or elevation) **with measurement noise** (hardware calibration
residuals, thermal, quantisation). Whether adding the Gram
supervision helps generalisation is empirically testable — it
depends on the SNR of that complement.

**Not correct to claim**: "v5 throws away phase information."
v5 supervises the phase that affects |RA|. What it doesn't
supervise is the null-space of the (FFT → magnitude) map.

---

## 12. What this means for v6

The plan revisions that fall out of preferring Option C:

1. **No DBF on the supervision side.** We skip azimuth beam-forming
   entirely; the complex per-virtual range profiles the renderer
   already produces are our `v` vectors. M1 "DBF plumbing" becomes
   instead "per-virtual complex range profile extraction on the
   renderer output + parse radar config to know which ADC channel
   is which virtual position (only needed for the diagnostic
   DBF-RA metric, not for the loss)."

2. **M2 loss is the normalised Gram loss** in eq. of §10 (the
   `1 - |v_p^H v_g|² / (||v_p||² ||v_g||²)` form), summed over
   `(d, r)` — doppler × range.

3. **Evaluation metric unchanged**: render → v5 FFT azimuth path →
   polar→cart → cart_corr. Remains bit-identical to v5 because the
   training loss change doesn't alter the eval path.

4. **M3 (Doppler gate)** generalises: gate zero out (d, r) cells
   whose Doppler index is far from the ego-motion-predicted value
   for any reasonable scene angle. The `angle → Doppler` mapping
   still uses the azimuth-axis projection of ego-velocity, but we
   need an *angle* for each virtual channel for the per-channel
   gate. Options:
   a. Gate the whole virtual channel at once using the scene's
      expected Doppler range `[−2v_ego_max/λ, +2v_ego_max/λ]`.
   b. Do a cheap DBF just for the gate (not for the loss), to
      establish per-virtual azimuthal sensitivity.
   Simpler is (a); start there.

5. **M4 (SSIM)** applies to the magnitude of the cube
   `|\tilde V|` before Gram — i.e., SSIM on `(D, N, R)` magnitude.
   Some tweaks needed: SSIM assumes spatial structure in 2D; for
   the virtual dimension this is less natural. Alternative: apply
   SSIM after an on-the-fly DBF diagnostic pass on magnitude
   (for the SSIM term only; loss on Gram stays on complex `v`).
   Revisit at M4.

---

## 13. Bottom line

**Yes — Option C is fully consistent with the v5 azimuth-FFT
approach, in the strong sense that:**

- **v5's loss quantities are a linear projection of Option C's
  Gram matrix** (§5).
- **Both approaches are invariant to global absolute phase** (§6).
- **Option C carries strictly more information per range bin**
  (§4, §11), and that "more" is exactly the pairwise relative
  phase information that v5 throws away.
- **The spatial / azimuth decomposition the user asked for is
  already present in the Gram via virtual-pair indexing (§8)** —
  no beam-forming required.

The practical loss (§10) has no Gram matrix storage cost — the
Frobenius² reduces to scalar inner products per `(d, r)`. Fully
compatible with the renderer's native output (complex range profiles
per virtual pair) and with the Doppler FFT step.

Cost estimate vs the earlier plan: **roughly the same code volume**
(~200 LOC total for M2, mostly in the loss module), simpler
conceptually, and strictly more information per sample.
