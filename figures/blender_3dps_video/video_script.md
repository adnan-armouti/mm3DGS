# 3DPS Pipeline Video — Storyboard / Script

A narrative shot-by-shot script for a ~60 s educational video that explains
how 3D Point Splatting (3DPS) renders complex range profiles for
mmWave radar, suitable for paper supplement material and oral
presentation. Stills lifted from this video populate the four-panel
pipeline figure of the NeurIPS paper.

**Target spec.** 1920 × 1080, 30 fps, 1800 frames, ≈ 60 s. Carrier
77 GHz so λ ≈ 3.89 mm, λ/2 ≈ 1.95 mm — references inserted on-screen
where relevant.

**Visual language.**

* `mesh_grey` `#9E9EB0` — LiDAR scene mesh, matte diffuse.
* `pink_3dps` `#D73C8C` — 3DPS point primitives.
* `normal_dark` `#1A1A1F` — surface-normal indicator lines.
* `viridis(ε_r')` — material-coloured points after S3 transition.
* `tx_blue` `#3373D9`, `rx_red` `#D94C40` — antenna markers, with soft emission.
* `wave_amber` `#FFD933` — bright wavefront tracer.
* `peak_pink` `#D73C8C` — highlighted bin in range-profile inset.
* Background `#F0F1F4` (very light grey, no pure white).

Narration is suggested but not required; on-screen text overlays are the
primary information channel.

---

## S0 — Mesh fade-in (1.0 s, frames 1–30)

**Action.** A camera-static establishing shot. The LiDAR-reconstructed
scene mesh (an indoor stairwell from S1-F185) fades from 0% to 100%
opacity. No other geometry is in frame.

**On-screen text.** Top-left, fades in alongside mesh:
```
Scene mesh (LiDAR-derived)
```

**Camera.** Fixed: positioned ~6 m back, ~3 m up, looking down at the
scene centre at a 25° pitch. 35 mm lens. Stays still through the end of
S3.

**Narration (suggested).** *"Our scene representation begins from a
LiDAR-derived 3D mesh of the room."*

---

## S1 — Points appear (4.0 s, frames 31–150)

**Action.** ~1500 small pink spheres pop into existence at the optimised
3DPS point positions, scattered over the mesh surface. Pop-in is
staggered linearly across the 4-second window so the cloud "fills out"
front-to-back.

**On-screen text.**
```
Optimised 3DPS points
N = 1,500 oriented primitives
```

**Camera.** Static.

**Narration.** *"Optimisation produces an explicit set of point
primitives — 1500 here — that will replace the mesh as our renderable
representation."*

---

## S2 — Normals visualise (3.0 s, frames 151–240)

**Action.** A short dark cylinder (length ~6 cm in scene units) extrudes
from each point along the point's optimised surface normal. They grow
from compressed (0) to full length over the window with a slight
per-point stagger.

**On-screen text.**
```
Each point carries a normal direction n_i
```

**Camera.** Static.

**Narration.** *"Each point also carries an oriented surface normal —
the direction along which it scatters incoming radar energy."*

---

## S3 — Materials (3.0 s, frames 241–330)

**Action.** Cross-fade each point's base colour from `pink_3dps` to its
viridis-encoded ε_r' value. Order is again front-to-back.

**On-screen text.**
```
Material: real permittivity ε_r' ∈ [3.4, 8.0]
```

A small horizontal viridis colorbar appears at the bottom-right with end
labels `ε_r'=3.4` and `ε_r'=8.0`.

**Camera.** Static.

**Narration.** *"Optimisation also recovers the point's material — its
real permittivity ε_r' — colour-coded here."*

---

## S4 — Camera zoom + single TX/RX appear (3.0 s, frames 331–420)

**Action.** Camera dollies in toward one chosen *focal point* near the
visual centre (selected by score: roughly central, normal pointing up).
All other points fade out over the same window. Two new spheres appear:

* TX antenna marker at the top-left, blue, soft emission.
* RX antenna marker at the top-right, red, soft emission.

The two antennas are placed on a **virtual array board** ~0.6 m above
the scene midpoint. Their layout is a 100× scaled-up version of the real
mm-scale spacing so it's visible on-screen.

**On-screen text.**
```
Focusing on one point + one TX + one RX
```

**Camera.** Smooth dolly: from establishing shot to a 3-shot of (focal
point, TX, RX) framed roughly equally.

**Narration.** *"To explain how a point produces a measurement, let's
reduce to one point, one transmitter, and one receiver."*

---

## S5 — Wavefront TX → point (4.0 s, frames 421–540)

**Action.** A bright amber ball (radius ~3 cm) emerges from TX and
travels along the straight line TX → focal point. It scales from 0 → 1
in the first ~0.2 s, holds at 1, and shrinks to 0 in the last ~0.2 s as
it "absorbs" into the point.

A faint, wider cone-of-light effect can be added behind the ball to
suggest it's a moving wavefront.

**On-screen text.**
```
Outgoing wavefront from TX:
distance R_i^TX
```

A live distance readout under the text (updating each frame):
`R_i^TX = X.XX m`.

**Camera.** Holds the 3-shot, slight orbital drift (10° azimuth) for
parallax cues.

**Narration.** *"The transmitter emits a wave. The portion of that
wavefront that hits our point of interest travels a distance R_i^TX in
free space."*

---

## S6 — Wavefront point → RX (4.0 s, frames 541–660)

**Action.** Another amber ball appears at the focal point and travels
along the straight line focal point → RX. Same fade-in / fade-out as
S5.

**On-screen text.**
```
Reflected wavefront to RX:
distance R_i^RX
```

A live distance readout under the text, plus a running total displayed
slightly below:
`R_i = R_i^TX + R_i^RX = Y.YY m  (round-trip)`.

**Camera.** Continues subtle orbital drift.

**Narration.** *"After interacting with the point, energy scatters back
to the receiver, travelling another distance R_i^RX."*

---

## S7 — Phase visualisation: phasor wraps with R (6.0 s, frames 661–840)

**Action.** Two simultaneous threads:

1. The focal point oscillates **radially** (toward/away from the
   midpoint of TX-RX) following a damped sinusoid:
   `d(t) = 0.25 · exp(-2t) · sin(12πt)` metres. So early in the
   sequence motion is fast and large; by the end it has nearly stopped.

2. A **phasor inset** in the bottom-left of the frame appears: a
   reference circle ring and a single phasor arrow that points from the
   origin toward the unit circle, rotated by
   `φ_i(t) = -2π · 2 R_i(t) / λ`. The factor 2 accounts for the
   round-trip path. At λ ≈ 3.89 mm, every 1.95 mm of point displacement
   produces a full 2π wrap.

Visual outcome: while the point is moving fast, the phasor blurs around
the ring (many wraps per frame). As motion damps out, the phasor
settles, revealing how *small* a motion still produces a measurable
phase change — the 77 GHz sensitivity story.

**On-screen text.**
```
Phase φ_i = -2π R_i / λ
λ at 77 GHz ≈ 3.89 mm
2π wrap per λ/2 ≈ 1.95 mm of point motion
```

**Camera.** Hold the 3-shot. The phasor inset is parented to the camera
so it's stable in screen space.

**Narration.** *"The round-trip distance defines the wave's phase. At
77 GHz, half a wavelength is under 2 mm, so even small point motions
wrap the phasor through full rotations."*

---

## S8 — Magnitude + range profile manifestation (6.0 s, frames 841–1020)

**Action.** Two simultaneous threads, taking over from S7:

1. The focal point translates radially (smoothly, half-cosine path) over
   ±0.6 m around its starting R. This sweeps the phasor's contribution
   across a band of range bins.

2. A **range-profile inset** at the bottom-centre of the frame appears:
   a row of 32 small grey bars representing range bins. At each frame,
   the bar at index `k_i = round(R_i / Δr)` is highlighted in
   `peak_pink` and gets the tallest height. Adjacent bins receive a
   Gaussian PSF spread (σ ≈ 1.4 bins).

Optionally, a small annotation reads:
```
|z_i| = G_TX · f_BSDF(θ_i, θ_o, n_i, ε_r', σ, t) · G_RX / R_i^2
```

with the BSDF term highlighted in cyan to underline that magnitude
depends jointly on the surface normal *and* the material.

**On-screen text.**
```
Magnitude |z_i| ∝ BSDF(n_i, material) / R_i²
Range profile bin k_i = round(R_i / Δr)    (Δr ≈ 4.3 cm)
```

**Camera.** Static 3-shot.

**Narration.** *"The phasor's magnitude depends on the BSDF — a function
of the point's normal and material — and falls off as one over R²."*
*"Each phasor lives in exactly one range bin, set by the round-trip
distance."*

---

## S9 — Multiple RX reveal (6.0 s, frames 1021–1200)

**Action.** RX[1], RX[2], RX[3] pop into existence sequentially (one
every ~50 frames), forming a small horizontal RX array on the radar
board. Light a thin grey horizontal line beneath them to denote the RX
array axis.

For each new RX, a faint amber tracer sketches the ray
`focal point → RX[m]`, then dims. (Reuse the wavefront helper from S6
but at lower brightness.)

A second, smaller phasor inset appears for each new RX, stacked
vertically on the right side of the frame. Each phasor has a slightly
different phase encoding the azimuth-dependent path-length difference.

**On-screen text.**
```
4-RX receive aperture
phase difference ≈ -2π d sin(θ) / λ
```

**Camera.** Slight pull-back to fit the RX array board comfortably.

**Narration.** *"Multiple receivers listening to the same transmit pulse
each see a slightly different phase — a signature of the angle of
arrival."*

---

## S10 — TDM: multiple TX fire sequentially (6.0 s, frames 1201–1380)

**Action.** TX[1] through TX[11] pop into existence in TDM order, one
every ~16 frames. As each TX appears it briefly pulses bright (emission
spike for ~6 frames) and a faint outgoing tracer is drawn from it to
the focal point. Once all 12 TXs are visible, a pale grid joining TXs
and RXs shows the **virtual array** has been formed.

**On-screen text.**
```
TDM transmit: 12 TX × 16 RX = 192 virtual elements
```

**Camera.** Hold the wide pull-back from S9.

**Narration.** *"Time-division-multiplexed transmits build up a
192-element virtual array in a single frame."*

---

## S11 — Azimuth FFT → RA map (6.0 s, frames 1381–1560)

**Action.** Highlight all phasors at the peak range bin (12 × 16 = 192
of them). Animate a stylised **azimuth FFT**: the phasors rotate into a
column of complex samples, then transform into an angular spectrum.

A new **RA-map inset** appears in the bottom-right: a small plane,
initially dark with a single bright pixel at the (range, azimuth) of
the focal point. The single bright pixel is centered visually so a
viewer can map "focal point → that pixel".

**On-screen text.**
```
Azimuth FFT across virtual array
→ Range-Azimuth (RA) map: one peak (one reflector)
```

**Camera.** Static.

**Narration.** *"An FFT across the spatial virtual-array dimension
recovers azimuth. With one reflector, we get one bright peak in the
range-azimuth map."*

---

## S12 — Fill the scene → fill the RA map (8.0 s, frames 1561–1800)

**Action.** Re-spawn the rest of the 1500 points (skipping the focal
one), staggered front-to-back. As each new point pops in, briefly draw
a faint single tracer from TX[0] to that point and from that point to
RX[0] (very faint, ~10% opacity, 10 frames each). Each new point adds
its own peak in the RA-map inset, which gradually saturates and starts
to look like the scene's RA map.

By frame 1800, the full 3DPS scene is back, the RA map shows an
identifiable angular signature of the scene, and the camera does a
final 0.5-second push-in for emphasis.

**On-screen text.**
```
N points → N phasors → coherent RA-map
3DPS = explicit, compressible, complex-valued representation
```

**Camera.** Slight final push-in (~0.5 m) starting at frame 1740.

**Narration.** *"Repeating this sum over all points produces the full
range-azimuth map. The scene representation that started this video is
recoverable through 192 phasors per range bin — explicit, compressible,
and with phase preserved."*

---

## End frame text (frames 1740–1800, fades over last 0.5 s)

```
3DPS — Physically Grounded Novel View Synthesis for mmWave Radar
NeurIPS 2026 (under review)
```

Faintly recede the scene to ~50% opacity behind this text.

---

# Stills extraction guide

Use these representative frames in the four-panel pipeline figure:

| Pipeline panel | Source frame | Reason |
|---|---|---|
| (a) Scene rep. | 330 (end of S3) | Full point cloud with material colours visible. |
| (b) Per-point ras. | 600 (mid S6) | Both wavefront paths visible — TX→p→RX clearly drawn. |
| (c) Range profile | 1010 (late S8) | One peak crystallised, clean PSF on bars. |
| (d) RA map | 1560 (end S11) | First single-reflector RA map appears. |

# Implementation checklist

The Python implementation lives at
`figures/blender_3dps_video/blender_video.py`. It already contains
keyframed stubs for every sequence; refining the visual quality is the
ongoing task. In particular:

- [ ] Replace the phasor inset's basic 3D arrow with a Compositor 2D
  overlay (or a custom shader on a parented plane) that reads cleaner
  at small inset size.
- [ ] Bake an Image Sequence for the RA-map inset so its content
  reflects an actual computed RA map per frame, not a colour proxy.
- [ ] Add a second amber tracer for the *return* path that runs
  in parallel with multiple-RX reveal in S9 (currently single-tracer).
- [ ] Tune sun angle so shadows in S0–S3 fall *toward* the camera, not
  away (currently away).
- [ ] Add live distance / phase / bin-index numerical readouts as
  Blender Text objects in screen space (TODO in `blender_video.py`).
