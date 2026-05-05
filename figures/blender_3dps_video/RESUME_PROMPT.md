# Resume prompt — 3DPS pipeline video implementation

Copy the block below into a fresh Claude Code session at the project root
(`/home/adnan/Desktop/mm3DGS`). Self-contained: tells the assistant what
the project is, where to read the spec, what files already exist, and
what the iteration loop should look like.

---

## Prompt to paste

```
Project: 3DPS — Physically Grounded Novel View Synthesis for mmWave Radar
(NeurIPS 2026 paper). I'm continuing work on an educational video that
explains the 3DPS forward model. The narrative spec lives in:

    /home/adnan/Desktop/mm3DGS/figures/blender_3dps_video/video_script.md

Read that .md first — it's the storyboard, with frame ranges, on-screen
text, camera moves, and visual tokens for every sequence (S0 through S12,
60 s @ 30 fps). Stills from the rendered video populate the four-panel
pipeline figure of the paper.

A first-draft Blender 4.x implementation is at:

    figures/blender_3dps_video/blender_video.py     (~1100 lines, all
                                                     sequences keyframed)
    figures/blender_3dps_video/prep_data.py         (extracts torch model
                                                     into .npy/.json that
                                                     Blender can load)
    figures/blender_3dps_video/README.md            (how to install Blender,
                                                     run prep, and launch
                                                     the script)

Pre-extracted scene data already lives in:

    /home/adnan/Desktop/mm3DGS/output/blender_video/preprocessed/
        positions.npy   (20000, 3) float32
        normals.npy     (20000, 3) float32
        eps_r.npy       (20000,)   float32
        antennas.json   (12 TX, 16 RX positions in metres)
        mesh.ply        (symlink to the LiDAR scene)

Constraints:

* Match the paper's visual language. Outer figure tile bg #f5f5f5, inner
  tile bg #ebebeb (matches figures/generate_fig_pipeline_3dps.py and the
  teaser figure).
* Cycles renderer, GPU if available, 64 spp for final, 16 spp Eevee for
  preview iterations.
* Carrier 77 GHz so λ ≈ 3.89 mm. Phase wraps every 1.95 mm of point motion
  must read clearly in S7's phasor inset.

Workflow I want you to follow:

1. Verify Blender is installed (`which blender`). If not, install it
   (`sudo apt-get install -y blender`) and confirm version 4.0+. Do not
   silently continue if Blender is missing.
2. Re-run the prep script if its outputs are missing or older than
   best_model.pt:
       /home/adnan/.conda/envs/mmir/bin/python \
           figures/blender_3dps_video/prep_data.py
3. Build the .blend by running the script headless:
       blender --background --python figures/blender_3dps_video/blender_video.py
   This writes output/blender_video/blender_video.blend.
4. Render a sub-range first (e.g. frames 661–840 = sequence S7) at low
   spp / Eevee for fast preview, save to PNG sequence under
   output/blender_video/frames_preview/, and report back what you see.
   Use a short headless command like:
       blender -b output/blender_video/blender_video.blend \
               -s 661 -e 840 -o //frames_preview/preview_ -a
5. Sample a few frames (e.g. frames 30, 240, 540, 840, 1200, 1560, 1800)
   and inspect them. Critique the result against the script with
   specific frame-by-frame notes (lighting, framing, readability of
   inset elements, on-screen text legibility).
6. Apply targeted edits to blender_video.py to fix the highest-impact
   issues. Re-run step 4. Iterate.

Do not try to render all 1800 frames at full quality during iteration —
that's a many-hour job. Reserve full Cycles renders for once each
sequence reads cleanly in preview.

Open issues to expect (already documented at the bottom of
video_script.md):

* Phasor inset is a single 3D arrow on a torus ring → may need to be
  replaced with a Compositor 2D overlay or a custom shader plane for
  clean readability at inset size.
* RA-map inset uses a colour-proxy emission node → swap in an Image
  Sequence baked from an actual per-frame RA map for true scene-fill
  realism.
* Sun direction in S0–S3 currently casts shadows away from camera; flip
  to pull shadows toward viewer for depth cues.
* No live numerical readouts yet (R_TX, R_RX, R, k_i) — add Blender Text
  objects parented to the camera, with their content driven by the same
  expressions used to drive the focal-point keyframes.

When you're ready, run the workflow and report back step 5's findings.
Be specific — frame numbers, what's wrong, and which function in
blender_video.py to edit.
```

---

## Notes for the human pasting this

* If Blender 4.x is not installed and you don't want to install via apt
  (e.g. on a headless machine or you prefer Blender 4.2 LTS specifically),
  the assistant should pause and ask you for the local path to Blender
  before proceeding.

* If the assistant tries to render at full Cycles quality for the
  full 1800 frames in one shot, push back — a single full-quality render
  is many hours of GPU time. Always preview first.

* When iterating, ask the assistant to crop to a specific panel region
  of a still and embed the cropped image inline so you can both see the
  same view. (Earlier iteration loops on the teaser figure used this
  pattern with `PIL.Image.crop` calls.)

* Final stills for the paper figure should be lifted from
  `output/blender_video/frames/` after a full-quality Cycles render.
  Cropping suggestions are at the bottom of `video_script.md`.
