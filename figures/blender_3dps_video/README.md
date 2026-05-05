# 3DPS Pipeline Animation — Blender Workflow

## Files

```
figures/blender_3dps_video/
├── prep_data.py     # extracts torch model → numpy/JSON (run in mmir conda env)
├── blender_video.py # main Blender script (run in Blender 4.x)
└── README.md        # this file
```

## One-time setup

### 1. Pre-extract data (once per scene)

Blender's bundled Python doesn't have PyTorch, so we dump the optimised 3DPS
state to numpy/JSON first:

```bash
/home/adnan/.conda/envs/mmir/bin/python figures/blender_3dps_video/prep_data.py \
    --scene seq_1_frame_185 --test_frame 185
```

This populates `output/blender_video/preprocessed/` with `positions.npy`,
`normals.npy`, `eps_r.npy`, `antennas.json`, and a symlink `mesh.ply`.

### 2. Install Blender (if not present)

```bash
sudo apt-get install -y blender             # gets a recent stable build
# OR download Blender 4.2 LTS from https://www.blender.org/download/lts/
```

Verify:

```bash
blender --version
```

The script targets Blender **4.0+** (uses the post-3.0 PLY import op
`bpy.ops.wm.ply_import`).

## Iterative workflow

Open Blender, switch to the **Scripting** workspace, open
`figures/blender_3dps_video/blender_video.py`, and click **Run Script**.
The scene gets built and the timeline is populated with all keyframes
(no rendering yet).

You can then:

* Scrub the timeline to preview animations at any frame.
* Tweak constants at the top of the script (frame ranges, point counts,
  colours) and re-run.
* Export individual frames as PNG via `F12`.
* Export the full clip via `Render → Render Animation`.

Headless render (no UI):

```bash
blender -b output/blender_video/blender_video.blend -a
```

## Sequence breakdown

| Frames     | Section                        | What happens                                      |
|------------|--------------------------------|---------------------------------------------------|
| 1–30       | S0 — Mesh fade-in              | LiDAR mesh appears                                |
| 31–150     | S1 — Points appear             | ~1500 oriented points scale-in (staggered)        |
| 151–240    | S2 — Normals visualise         | Thin cylinders extrude along each point's normal  |
| 241–330    | S3 — Materials (colours)       | Points fade pink → viridis(ε_r')                  |
| 331–420    | S4 — Camera zoom               | Camera dollies to one focal point; TX[0]+RX[0] in |
| 421–540    | S5 — Wavefront TX → point      | Bright tracer travels along the incoming ray      |
| 541–660    | S6 — Wavefront point → RX      | Tracer reflects off point and travels to RX       |
| 661–840    | S7 — Phasor wrap               | Focal point oscillates radially; phasor inset spins |
| 841–1020   | S8 — Range profile             | 1D bar chart fills in; peak bin shifts as R moves |
| 1021–1200  | S9 — Multiple RX               | RX[1..3] appear, illustrating receive aperture    |
| 1201–1380  | S10 — TDM                      | TX[1..11] flash sequentially; virtual array forms |
| 1381–1560  | S11 — Azimuth FFT → RA map     | RA-map plane fades in; single-reflector peak      |
| 1561–1800  | S12 — Fill scene → fill RA map | All points return; RA-map saturates               |

Total: **1800 frames @ 30 fps = 60 s**.

## Stills extraction (for the paper figure)

Once a render pass completes, every frame is at
`output/blender_video/frames/frame_####.png`.  Pick representative frames
(suggested: end of each S?) and crop into the 4-tile pipeline figure
constructed by `figures/generate_fig_pipeline_3dps.py`.

Suggested still selections for the four pipeline panels:

* Panel **(a)** — frame 330 (end of S3, materials visible).
* Panel **(b)** — frame 600 (mid S6, wave returning to RX, geometry clear).
* Panel **(c)** — frame 1020 (end of S8, range profile fully formed).
* Panel **(d)** — frame 1560 (end of S11, RA map populated by single point).

## Known limitations / things you'll likely want to refine

* The **phasor inset** is a single 3D arrow on a torus ring.  For a tighter
  visual, swap in a 2D image-sequence overlay or use Blender's compositing
  step to add an annotation pass.
* The **RA map** plane is currently driven by a uniform emission colour —
  swap it for an Image Texture loading a precomputed RA-map PNG sequence
  (one frame per point activation) for true scene-fill realism.
* Camera **easing** uses Bezier interpolation everywhere; you'll likely want
  tighter ease-in/-out around S4 and S7 cuts.
* The sun/fill light intensities target an indoor stairwell scene; bump them
  up if your scene is darker.

## Quick test render (preview-quality)

To validate the timeline without burning hours on Cycles:

```bash
# Edit blender_video.py → set ENGINE = "BLENDER_EEVEE_NEXT" and SAMPLES = 16
# Then headless:
blender -b output/blender_video/blender_video.blend -a
```

Eevee Next at 16 samples renders ~10× faster than Cycles at 64 spp.
