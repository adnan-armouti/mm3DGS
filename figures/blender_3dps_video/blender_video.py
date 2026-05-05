"""3DPS pipeline animation — Blender Python script (Blender 4.x).

Run from inside Blender:

    1. (one-off, in conda env) python figures/blender_3dps_video/prep_data.py
    2. Open Blender 4.x → Scripting workspace → load this file → Run Script
       OR headless:
       blender --background --python figures/blender_3dps_video/blender_video.py

The script builds the entire scene programmatically (no .blend file) and
animates it via keyframes.  Total length: 60 s @ 30 fps = 1800 frames.

Sequences (frame ranges are [start, end]):

    S0  001-030  : LiDAR mesh fades in.
    S1  031-150  : ~1500 oriented points appear (scale 0→1, staggered).
    S2  151-240  : Normal indicators (thin cylinders) fade in on each point.
    S3  241-330  : Points recolour from "ours-pink" → viridis(ε_r').
    S4  331-420  : Camera dollies in to one focal point; spawn 1 TX + 1 RX.
    S5  421-540  : Bright wavefront "tracer" travels TX → focal point.
    S6  541-660  : Bright wavefront "tracer" travels focal point → RX.
    S7  661-840  : Focal point oscillates radially; phasor inset spins,
                    showing 2π wrap density at 77 GHz (λ/2 = 1.95 mm).
    S8  841-1020 : Range-profile inset bars; peak bin shifts as point moves.
    S9  1021-1200: 4 RX appear; each draws its own phasor (azimuth phase).
    S10 1201-1380: TDM — 12 TX fire sequentially → virtual array forms.
    S11 1381-1560: Azimuth FFT animates → RA-map inset shows one bright peak.
    S12 1561-1800: Population: re-add many points; RA-map fills in to scene.

Conventions
-----------
* All world-space coordinates are in metres (matches the LiDAR mesh + radar
  config).  Antenna positions and the mesh are loaded *as-is*.
* Insets (phasor, range profile, RA map) are 3D objects parented to the
  camera so they sit in screen space.  No texture image sequences needed.
* The phasor's spin is *driven by the point's distance to the radar centre*
  (so that a moving point literally produces 2π·R/λ rotations, true to the
  physics).
* Color tokens match the teaser/pipeline figures so video stills can be
  pasted straight into the paper.
"""

import bpy
import bmesh
import json
import math
import os
import sys
from mathutils import Vector, Quaternion, Matrix, Euler

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Adjust these to your local layout.
PROJ_ROOT  = "/home/adnan/Desktop/mm3DGS"
PREP_DIR   = f"{PROJ_ROOT}/output/blender_video/preprocessed"
OUT_DIR    = f"{PROJ_ROOT}/output/blender_video/frames"
RENDER_PNG = True        # save PNGs per frame
RENDER_VIDEO = True      # also encode to MP4 at the end

# ── Render settings ─────────────────────────────────────────────────────────
FPS            = 30
RES_X, RES_Y   = 1920, 1080
SAMPLES        = 64                  # Cycles samples; lower for preview
ENGINE         = "CYCLES"            # "CYCLES" or "BLENDER_EEVEE_NEXT"
USE_GPU        = True

# ── Frame allocation (matches the docstring) ────────────────────────────────
FRAMES = {
    "S0":  (1,    30),
    "S1":  (31,   150),
    "S2":  (151,  240),
    "S3":  (241,  330),
    "S4":  (331,  420),
    "S5":  (421,  540),
    "S6":  (541,  660),
    "S7":  (661,  840),
    "S8":  (841,  1020),
    "S9":  (1021, 1200),
    "S10": (1201, 1380),
    "S11": (1381, 1560),
    "S12": (1561, 1800),
}
FRAME_END = 1800

# ── Scene constants (radar physics) ─────────────────────────────────────────
C_LIGHT  = 299_792_458.0   # m/s
F_C      = 77.0e9          # Hz
LAMBDA_R = C_LIGHT / F_C   # ≈ 3.89 mm
DELTA_R  = 0.0432          # m, range-bin width (cascaded chirp params)

# ── Scene-level visual tokens (match teaser figure) ─────────────────────────
MESH_COLOR        = (0.62, 0.62, 0.68, 1.0)
PINK              = (0.840, 0.235, 0.549, 1.0)
NORMAL_COLOR      = (0.10, 0.10, 0.12, 1.0)
TX_COLOR          = (0.20, 0.45, 0.85, 1.0)
RX_COLOR          = (0.85, 0.30, 0.25, 1.0)
WAVE_COLOR        = (1.0, 0.85, 0.20, 1.0)   # bright amber tracer
BG_COLOR          = (0.94, 0.95, 0.97, 1.0)

# ── Geometry sizes (in scene-metres) ────────────────────────────────────────
SCENE_POINT_RADIUS   = 0.020
SCENE_POINT_LINE_LEN = 0.100
SCENE_POINT_LINE_RAD = 0.0055
ANTENNA_RADIUS       = 0.025
WAVEFRONT_RADIUS     = 0.030       # bright marker that travels along TX→p / p→RX

# Pedagogical close-up positions for TX[0] / RX[0] during S4–S8.  Offsets are
# relative to the focal point (world frame).  Chosen so a 35 mm lens at the
# S4-end camera location frames a clean 3-shot of focal + TX + RX.
TX0_CLOSE_OFFSET     = ( -0.55, -0.40, 0.70)
RX0_CLOSE_OFFSET     = (  0.55, -0.40, 0.70)

N_SCENE_POINTS_FULL  = 1500        # total at S12
N_SCENE_POINTS_S1    = 1500        # appearing in S1
FOCAL_POINT_INDEX    = None        # picked deterministically in S4 from prep data


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _frames(key):
    """Return inclusive (start, end) frame indices for a sequence key."""
    return FRAMES[key]


def clear_scene():
    """Remove every object/material/mesh from the default startup scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.curves, bpy.data.images,
                 bpy.data.collections):
        for item in list(col):
            if not item.users:
                col.remove(item)


def make_principled_material(name, base_color, metallic=0.0, roughness=0.55,
                              emission_color=None, emission_strength=0.0,
                              alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission_strength > 0:
        ec = emission_color or base_color
        bsdf.inputs["Emission Color"].default_value = ec
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.blend_method = "BLEND"
    return mat


def keyframe_value(obj_or_data, data_path, value, frame, index=-1,
                    interp="BEZIER"):
    """Set a value at frame and insert a keyframe."""
    if hasattr(obj_or_data, data_path):
        setattr(obj_or_data, data_path, value)
    else:
        # nested attribute, e.g. "rotation_euler"; path may be like a chain
        parts = data_path.split(".")
        target = obj_or_data
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], value)
    obj_or_data.keyframe_insert(data_path=data_path, frame=frame, index=index)
    if interp != "BEZIER":
        # Find the F-curve for this property and adjust interpolation.
        anim = obj_or_data.animation_data
        if anim and anim.action:
            for fc in anim.action.fcurves:
                if fc.data_path == data_path and (index < 0 or fc.array_index == index):
                    for kp in fc.keyframe_points:
                        if int(kp.co.x) == frame:
                            kp.interpolation = interp


def _viridis(t):
    """Approximate viridis colormap without matplotlib (Blender's bundled
    Python doesn't have it).  ``t`` in [0,1]."""
    # 8 anchor points sampled from matplotlib's viridis.
    anchors = np.array([
        [0.267, 0.005, 0.329],
        [0.282, 0.140, 0.457],
        [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553],
        [0.164, 0.471, 0.558],
        [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518],
        [0.267, 0.749, 0.441],
        [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150],
        [0.993, 0.906, 0.144],
    ])
    t = float(np.clip(t, 0.0, 1.0))
    n = len(anchors) - 1
    i = int(t * n)
    f = t * n - i
    c0 = anchors[i]
    c1 = anchors[min(i + 1, n)]
    c = (1 - f) * c0 + f * c1
    return (float(c[0]), float(c[1]), float(c[2]), 1.0)


def _make_collection(name, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def _link(obj, collection):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection.objects.link(obj)


# ════════════════════════════════════════════════════════════════════════════
# RENDER + WORLD SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_render():
    sc = bpy.context.scene
    sc.render.engine = ENGINE
    sc.render.resolution_x = RES_X
    sc.render.resolution_y = RES_Y
    sc.render.fps = FPS
    sc.frame_start = 1
    sc.frame_end = FRAME_END
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 30
    if ENGINE == "CYCLES":
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = True
        if USE_GPU:
            sc.cycles.device = "GPU"
            prefs = bpy.context.preferences.addons.get("cycles")
            if prefs is not None:
                cprefs = prefs.preferences
                cprefs.compute_device_type = "CUDA"  # or "OPTIX" on RTX
                cprefs.refresh_devices()
                for dev in cprefs.devices:
                    dev.use = dev.type != "CPU"
    sc.view_settings.view_transform = "Filmic"
    sc.view_settings.look = "Medium Contrast"
    os.makedirs(OUT_DIR, exist_ok=True)
    sc.render.filepath = os.path.join(OUT_DIR, "frame_")


def setup_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = BG_COLOR
    bg.inputs["Strength"].default_value = 0.50


def setup_lighting():
    # Key sun.  Camera looks roughly along +Y at the scene; placing the sun at
    # (-3, -6, 8) and tilting it to point toward +Y/+X makes shadows fall
    # *toward* the camera, giving depth cues to the mesh and points.
    bpy.ops.object.light_add(type="SUN", location=(-3, -6, 8))
    sun = bpy.context.object
    sun.name = "KeySun"
    sun.data.energy = 1.2
    sun.data.angle = math.radians(6.0)
    sun.rotation_euler = Euler((math.radians(-55), math.radians(-12),
                                  math.radians(20)))

    # Soft fill from camera-left
    bpy.ops.object.light_add(type="AREA", location=(-3, -2, 4))
    fill = bpy.context.object
    fill.name = "Fill"
    fill.data.energy = 20.0
    fill.data.size = 4.0


def setup_camera(target_pos):
    bpy.ops.object.camera_add(location=Vector(target_pos) + Vector((-4.0, -6.0, 3.0)))
    cam = bpy.context.object
    cam.name = "MainCam"
    bpy.context.scene.camera = cam
    cam.data.lens = 35
    cam.data.sensor_width = 36
    cam.data.clip_end = 200.0
    # Track the target (use TRACK_TO with an empty)
    bpy.ops.object.empty_add(location=target_pos)
    target = bpy.context.object
    target.name = "CamTarget"
    cnst = cam.constraints.new("TRACK_TO")
    cnst.target = target
    cnst.track_axis = "TRACK_NEGATIVE_Z"
    cnst.up_axis = "UP_Y"
    return cam, target


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_prep_data():
    P = np.load(os.path.join(PREP_DIR, "positions.npy"))
    N = np.load(os.path.join(PREP_DIR, "normals.npy"))
    E = np.load(os.path.join(PREP_DIR, "eps_r.npy"))
    with open(os.path.join(PREP_DIR, "antennas.json")) as f:
        ant = json.load(f)
    mesh_path = os.path.join(PREP_DIR, "mesh.ply")
    return P, N, E, ant, mesh_path


def load_mesh(mesh_path, name="LiDARMesh"):
    """Import the LiDAR PLY and apply a uniform diffuse material."""
    if not os.path.exists(mesh_path):
        print(f"[load_mesh] missing {mesh_path}")
        return None
    bpy.ops.wm.ply_import(filepath=mesh_path)
    obj = bpy.context.selected_objects[-1]
    obj.name = name
    mat = make_principled_material("MeshMat", MESH_COLOR, roughness=0.85)
    obj.data.materials.append(mat)
    return obj


# ════════════════════════════════════════════════════════════════════════════
# GEOMETRY BUILDERS
# ════════════════════════════════════════════════════════════════════════════

def _new_uv_sphere(radius, segments=10, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius,
                                           segments=segments, ring_count=rings)
    obj = bpy.context.object
    bpy.ops.object.shade_smooth()
    return obj


def _new_cylinder(radius, depth, segments=12):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                          vertices=segments)
    obj = bpy.context.object
    bpy.ops.object.shade_smooth()
    return obj


def build_points(positions, normals, eps_r, n_points, collection):
    """Create N point spheres + N normal-line cylinders, parented to one
    empty per point so we can keyframe scale/visibility together.

    Returns
    -------
    points : list[Object]   per-point empty (anchor) — animate scale on this
    normals_objs : list[Object]  the normal-line child cylinder
    """
    rng = np.random.default_rng(0)
    n_total = len(positions)
    n = min(n_points, n_total)
    idx = rng.choice(n_total, n, replace=False)
    P = positions[idx]
    N = normals[idx]
    E = eps_r[idx]

    eps_norm = np.clip(
        (E - np.percentile(E, 2)) /
        (np.percentile(E, 98) - np.percentile(E, 2) + 1e-6), 0.0, 1.0)

    pink_mat = make_principled_material("PointPink", PINK, roughness=0.45)
    normal_mat = make_principled_material("NormalLine", NORMAL_COLOR,
                                            roughness=0.6)
    # Per-eps_r material cache (16 buckets)
    n_buckets = 16
    eps_mats = [
        make_principled_material(f"EpsBucket_{i:02d}",
                                   _viridis(i / max(n_buckets - 1, 1)),
                                   roughness=0.45)
        for i in range(n_buckets)
    ]

    # Single sphere mesh shared across all instances
    sph_template = _new_uv_sphere(SCENE_POINT_RADIUS, segments=10, rings=8)
    sph_template.name = "SphereTemplate"
    sph_template.hide_viewport = True
    sph_template.hide_render = True
    _link(sph_template, collection)

    cyl_template = _new_cylinder(SCENE_POINT_LINE_RAD, SCENE_POINT_LINE_LEN,
                                   segments=8)
    cyl_template.name = "NormalTemplate"
    cyl_template.hide_viewport = True
    cyl_template.hide_render = True
    _link(cyl_template, collection)

    points = []
    normals_objs = []
    for i in range(n):
        pos = Vector(P[i].tolist())
        nrm = Vector(N[i].tolist())

        # Anchor empty
        bpy.ops.object.empty_add(location=pos)
        anchor = bpy.context.object
        anchor.name = f"PointAnchor_{i:04d}"
        _link(anchor, collection)
        anchor.empty_display_size = 0.0001  # invisible

        # Sphere child — per-instance mesh copy so per-point materials work
        # (sharing sph_template.data caused S3 to recolour every point with
        # the last-iteration material, producing uniform teal at S12).
        sph = sph_template.copy()
        sph.data = sph_template.data.copy()
        sph.location = (0, 0, 0)
        sph.parent = anchor
        sph.name = f"Point_{i:04d}"
        sph.hide_viewport = False
        sph.hide_render = False
        sph.data.materials.clear()
        # Use eps bucket material directly (we'll switch materials in S3
        # by swapping; for now use pink in S1 via a user-placed override)
        bucket = int(round(eps_norm[i] * (n_buckets - 1)))
        sph.data.materials.append(pink_mat)        # default colour (S1/S2)
        sph["eps_bucket"] = bucket                  # Custom prop for S3 swap
        # Initially scale 0 — will animate in S1
        sph.scale = (0, 0, 0)
        _link(sph, collection)

        # Normal-line child cylinder (its +Z axis aligned with normal)
        cyl = cyl_template.copy()
        cyl.data = cyl_template.data
        cyl.parent = anchor
        cyl.name = f"Normal_{i:04d}"
        cyl.hide_viewport = False
        cyl.hide_render = False
        cyl.data.materials.clear()
        cyl.data.materials.append(normal_mat)
        # Place cylinder so its base sits on the point and tip on +n.
        # (Our cylinder template has its centre at origin along +Z.)
        cyl.location = (nrm.x * SCENE_POINT_LINE_LEN / 2,
                         nrm.y * SCENE_POINT_LINE_LEN / 2,
                         nrm.z * SCENE_POINT_LINE_LEN / 2)
        # Rotate +Z to nrm
        z = Vector((0, 0, 1))
        nv = nrm.normalized()
        if nv.dot(z) < 1.0 - 1e-6:
            axis = z.cross(nv)
            angle = z.angle(nv)
            cyl.rotation_mode = "AXIS_ANGLE"
            cyl.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        cyl.scale = (1, 1, 0.0001)        # Initially compressed (will grow in S2)
        _link(cyl, collection)

        points.append(sph)
        normals_objs.append(cyl)

    return points, normals_objs, idx, eps_mats


def build_antennas(antennas, scene_centre, collection):
    """Create 12 TX dots and 16 RX dots from the radar config.  The TX/RX
    *array geometry* is tiny (sub-millimetre spacing for 77 GHz cascaded),
    so we offset them onto a visible array board behind the scene centre.

    Returns dicts keyed by index so animation code can keyframe them.
    """
    tx_mat = make_principled_material("TXmat", TX_COLOR, roughness=0.35,
                                        emission_color=TX_COLOR,
                                        emission_strength=2.0)
    rx_mat = make_principled_material("RXmat", RX_COLOR, roughness=0.35,
                                        emission_color=RX_COLOR,
                                        emission_strength=2.0)

    # Build a "radar board": flat plane that visualises the array.  Place it
    # ~3 m above the scene centre, facing the scene.
    centre = Vector(antennas["center"])
    bore = Vector(antennas["boresight"])
    board_pos = centre - 0.5 * bore + Vector((0, 0, 0.6))

    tx_objs, rx_objs = [], []
    tx_arr = np.asarray(antennas["tx"])
    rx_arr = np.asarray(antennas["rx"])

    # Visualise the array geometry by 100× scale (mm → 10 cm spacing) so it's
    # legible.  We retain the relative layout — TXs in a row, RXs in another.
    SCALE = 100.0

    for i, pos in enumerate(tx_arr):
        delta = (Vector(pos.tolist()) - centre) * SCALE
        sph = _new_uv_sphere(ANTENNA_RADIUS, 12, 8)
        sph.location = board_pos + delta + Vector((0, 0, 0.10))
        sph.name = f"TX_{i:02d}"
        sph.data.materials.clear(); sph.data.materials.append(tx_mat)
        sph.scale = (0, 0, 0)        # hidden until S4/S10
        _link(sph, collection)
        tx_objs.append(sph)

    for i, pos in enumerate(rx_arr):
        delta = (Vector(pos.tolist()) - centre) * SCALE
        sph = _new_uv_sphere(ANTENNA_RADIUS, 12, 8)
        sph.location = board_pos + delta + Vector((0, 0, -0.10))
        sph.name = f"RX_{i:02d}"
        sph.data.materials.clear(); sph.data.materials.append(rx_mat)
        sph.scale = (0, 0, 0)        # hidden until S4/S9
        _link(sph, collection)
        rx_objs.append(sph)

    return tx_objs, rx_objs, board_pos


def build_wavefront_marker(name, color, collection, emission_strength=8.0):
    sph = _new_uv_sphere(WAVEFRONT_RADIUS, 14, 10)
    mat = make_principled_material(f"{name}Mat", color, roughness=0.2,
                                     emission_color=color,
                                     emission_strength=emission_strength)
    sph.data.materials.clear()
    sph.data.materials.append(mat)
    sph.name = name
    sph.scale = (0, 0, 0)
    _link(sph, collection)
    return sph


def build_phasor_inset(camera, collection):
    """A small 2D-styled phasor in screen space, parented to the camera.

    Layout (all parented to PhasorAnchor at camera-local (-0.30, -0.18, -1.0)):
      * PhasorRing — torus, default XY orientation → face-on to camera-Z.
      * PhasorPivot — empty at the ring centre; rotation_euler.z is driven
        by phase, so its children spin in the inset plane.
      * PhasorShaft — cylinder laid along anchor-local +X (length = ring radius),
        translated so its *base* is at the pivot and its tip touches the ring.

    Returns (anchor, ring, pivot) — the animation drives `pivot.rotation_euler.z`.
    """
    R_RING = 0.060

    bpy.ops.object.empty_add(location=(-0.32, -0.18, -1.0))
    anchor = bpy.context.object
    anchor.name = "PhasorAnchor"
    anchor.parent = camera
    _link(anchor, collection)

    # Reference circle — torus default lies in XY, axis along Z.  Since the
    # anchor's local -Z points away from the camera, the ring is face-on to
    # the camera with rotation_euler = (0, 0, 0).
    bpy.ops.mesh.primitive_torus_add(
        major_radius=R_RING, minor_radius=0.0035,
        major_segments=96, minor_segments=8)
    ring = bpy.context.object
    ring.name = "PhasorRing"
    ring.parent = anchor
    ring.rotation_euler = (0, 0, 0)
    ring.location = (0, 0, 0)
    ring.scale = (0, 0, 0)
    ring_mat = make_principled_material("PhasorRing", (0.30, 0.30, 0.32, 1.0),
                                          emission_color=(0.55, 0.55, 0.60, 1.0),
                                          emission_strength=1.5)
    ring.data.materials.clear(); ring.data.materials.append(ring_mat)
    _link(ring, collection)

    # Pivot empty at the ring centre.  Rotating its Z drives the shaft in the
    # inset plane (since its children inherit the rotation).
    bpy.ops.object.empty_add(location=(0, 0, 0))
    pivot = bpy.context.object
    pivot.name = "PhasorPivot"
    pivot.parent = anchor
    pivot.empty_display_size = 0.0001
    _link(pivot, collection)

    # Shaft — default cylinder is along Z (length 0.060).  We want it laid
    # along +X with its base at the pivot.  Steps:
    #   1. translate the *mesh* by +Z so the base is at local origin
    #   2. rotate -90° around Y so the long axis points along +X
    bpy.ops.mesh.primitive_cylinder_add(radius=0.0035, depth=R_RING,
                                          vertices=10, location=(0, 0, 0))
    shaft = bpy.context.object
    shaft.name = "PhasorShaft"
    # Push the mesh up by half its length so the object's origin is at the
    # cylinder's base (one of its caps).
    from mathutils import Matrix as _M
    shaft.data.transform(_M.Translation((0, 0, R_RING / 2)))
    shaft.parent = pivot
    shaft.rotation_euler = (0, math.radians(-90), 0)  # base at pivot, tip at +X·R_RING
    shaft.location = (0, 0, 0)
    shaft_mat = make_principled_material("PhasorArrow", PINK,
                                           emission_color=PINK,
                                           emission_strength=5.0)
    shaft.data.materials.clear(); shaft.data.materials.append(shaft_mat)
    shaft.scale = (0, 0, 0)
    _link(shaft, collection)

    # Arrow head — cone, base-attached to shaft tip
    bpy.ops.mesh.primitive_cone_add(radius1=0.008, radius2=0.0,
                                      depth=0.014, vertices=12,
                                      location=(0, 0, 0))
    head = bpy.context.object
    head.name = "PhasorHead"
    head.data.transform(_M.Translation((0, 0, 0.014 / 2)))
    head.parent = pivot
    head.rotation_euler = (0, math.radians(-90), 0)
    head.location = (R_RING, 0, 0)        # at shaft tip
    head.data.materials.clear(); head.data.materials.append(shaft_mat)
    head.scale = (0, 0, 0)
    _link(head, collection)

    # Return pivot in place of shaft so animation can drive a single rotation.
    # Stash references to the visible children so animate_S7 can scale them in.
    pivot["_visible_children"] = [shaft.name, head.name]
    return anchor, ring, pivot


def build_range_profile_inset(camera, collection, n_bins=32):
    """A row of small 3D bars in screen space, animated by setting bar
    Z-scale per frame to encode |z| at each bin.

    Returns list of bar objects (one per bin).
    """
    bpy.ops.object.empty_add(location=(0.00, -0.30, -1.0))
    anchor = bpy.context.object
    anchor.name = "RangeAnchor"
    anchor.parent = camera
    _link(anchor, collection)

    bars = []
    bar_mat_idle = make_principled_material("RangeIdle", (0.55, 0.55, 0.55, 1.0))
    bar_mat_peak = make_principled_material("RangePeak", PINK,
                                              emission_color=PINK,
                                              emission_strength=3.0)

    bin_w = 0.013
    bin_gap = 0.004
    bar_depth = 0.010
    total_w = n_bins * bin_w + (n_bins - 1) * bin_gap
    x0 = -total_w / 2 + bin_w / 2

    from mathutils import Matrix as _M
    for i in range(n_bins):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        bar = bpy.context.object
        bar.name = f"RangeBar_{i:02d}"
        bar.parent = anchor
        # Anchor-local axes (parented to camera): +X right, +Y up, +Z toward
        # camera-back.  Translate the cube mesh so its bottom face sits at
        # local Y=0, then Y-scale grows the bar *up* on screen.
        bar.data.transform(_M.Translation((0, 0.5, 0)))
        bar.location = (x0 + i * (bin_w + bin_gap), 0, 0)
        bar.scale = (bin_w, 0.001, bar_depth)     # invisible until S8 reveal
        bar.data.materials.clear(); bar.data.materials.append(bar_mat_idle)
        _link(bar, collection)
        bars.append(bar)

    return anchor, bars, bar_mat_peak


def build_ra_map_inset(camera, collection):
    """Plane with a procedural emission shader that we update by changing the
    base colour over frames (proxy for an actual RA texture).

    The proper approach uses an Image Sequence — left as a TODO so the user
    can swap in a precomputed RA image stack.
    """
    bpy.ops.mesh.primitive_plane_add(size=0.18,
                                       location=(0.30, -0.20, -1.0))
    plane = bpy.context.object
    plane.name = "RAMap"
    plane.parent = camera
    plane.rotation_euler = (0, 0, 0)
    plane.scale = (0, 0, 0)
    mat = bpy.data.materials.new("RAMapMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    em.inputs["Color"].default_value = (0.05, 0.0, 0.0, 1.0)
    mat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    plane.data.materials.append(mat)
    _link(plane, collection)
    return plane, em


# ════════════════════════════════════════════════════════════════════════════
# ANIMATION SEQUENCES
# ════════════════════════════════════════════════════════════════════════════

def animate_S0_mesh(mesh_obj):
    """Mesh fades in by ramping its single-material alpha 0→1."""
    s, e = _frames("S0")
    if mesh_obj is None:
        return
    mat = mesh_obj.data.materials[0]
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    mat.blend_method = "BLEND"
    bsdf.inputs["Alpha"].default_value = 0.0
    bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=s)
    bsdf.inputs["Alpha"].default_value = 1.0
    bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=e)


def animate_S1_points(points):
    """Points scale 0→1 with a small per-point stagger."""
    s, e = _frames("S1")
    n = len(points)
    span = max(e - s - 6, 1)
    for i, p in enumerate(points):
        t0 = s + int((i / max(n - 1, 1)) * span * 0.35)
        t1 = t0 + 6
        for axis in range(3):
            p.scale[axis] = 0.0
            p.keyframe_insert("scale", frame=t0, index=axis)
            p.scale[axis] = 1.0
            p.keyframe_insert("scale", frame=t1, index=axis)


def animate_S2_normals(normals_objs):
    """Normal-line cylinders grow from compressed (Z=0) to full length."""
    s, e = _frames("S2")
    n = len(normals_objs)
    span = max(e - s - 6, 1)
    for i, c in enumerate(normals_objs):
        t0 = s + int((i / max(n - 1, 1)) * span * 0.30)
        t1 = t0 + 6
        c.scale[2] = 0.0001
        c.keyframe_insert("scale", frame=t0, index=2)
        c.scale[2] = 1.0
        c.keyframe_insert("scale", frame=t1, index=2)


def animate_S3_materials(points, eps_mats):
    """Swap each point's material from pink → its eps-bucket viridis colour
    over the sequence.  We crossfade using a Mix shader.

    Simpler approach used here: keyframe a *node socket* that mixes between
    two BSDFs.  We rebuild the material to support this just-in-time so we
    don't disturb the S1/S2 setup.
    """
    s, e = _frames("S3")
    n = len(points)
    span = max(e - s - 8, 1)
    for i, p in enumerate(points):
        bucket = p.get("eps_bucket", 0)
        target_mat = eps_mats[bucket]

        # Clone the pink material into a Mix BSDF that fades to target.
        if "PointMix_%04d" % i in bpy.data.materials:
            mat = bpy.data.materials["PointMix_%04d" % i]
        else:
            mat = bpy.data.materials.new(f"PointMix_{i:04d}")
            mat.use_nodes = True
            mat.node_tree.nodes.clear()
            tree = mat.node_tree
            out = tree.nodes.new("ShaderNodeOutputMaterial")
            mix = tree.nodes.new("ShaderNodeMixShader")
            pink_b = tree.nodes.new("ShaderNodeBsdfPrincipled")
            tgt_b = tree.nodes.new("ShaderNodeBsdfPrincipled")
            pink_b.inputs["Base Color"].default_value = PINK
            pink_b.inputs["Roughness"].default_value = 0.45
            tgt_color = target_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value
            tgt_b.inputs["Base Color"].default_value = tgt_color
            tgt_b.inputs["Roughness"].default_value = 0.45
            tree.links.new(pink_b.outputs[0], mix.inputs[1])
            tree.links.new(tgt_b.outputs[0], mix.inputs[2])
            tree.links.new(mix.outputs[0], out.inputs["Surface"])
            mat["mix_node"] = mix.name
        mix = mat.node_tree.nodes[mat["mix_node"]]

        # Replace the point's material slot and animate the mix factor.
        p.data.materials.clear()
        p.data.materials.append(mat)

        t0 = s + int((i / max(n - 1, 1)) * span * 0.25)
        t1 = t0 + 8
        mix.inputs["Fac"].default_value = 0.0
        mix.inputs["Fac"].keyframe_insert("default_value", frame=t0)
        mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Fac"].keyframe_insert("default_value", frame=t1)


def _hide_show(obj, show, frame):
    """Keyframe both viewport and render visibility."""
    obj.hide_viewport = not show
    obj.hide_render = not show
    obj.keyframe_insert("hide_viewport", frame=frame)
    obj.keyframe_insert("hide_render", frame=frame)


def animate_S4_zoom(camera, target, focal_pos, points, normals_objs,
                     tx_objs, rx_objs):
    """Camera dollies toward the focal point; non-focal points fade out;
    one TX and one RX become visible (TX 0, RX 0).

    Also moves TX[0] and RX[0] from their physical-array world positions to
    pedagogical close-up positions (focal_pos + TX0_CLOSE_OFFSET / RX0_CLOSE_OFFSET).
    Otherwise, with SCALE=100 on cm-scale cascade-radar centroid offsets, the
    antennas land 5+ m off-axis and fall outside the 35 mm-lens FoV.
    """
    s, e = _frames("S4")

    # Animate camera + target to converge near focal point
    target.keyframe_insert("location", frame=s)
    camera.keyframe_insert("location", frame=s)

    target.location = Vector(focal_pos)
    camera.location = Vector(focal_pos) + Vector((-1.5, -2.5, 0.8))
    target.keyframe_insert("location", frame=e)
    camera.keyframe_insert("location", frame=e)

    # Fade out non-focal points (and their normals) by scaling to 0
    fade_start = s + 10
    fade_end = e - 5
    for i, p in enumerate(points):
        if i == 0:                    # keep one focal point — index 0 (anchor pick)
            continue
        for axis in range(3):
            p.scale[axis] = p.scale[axis]
            p.keyframe_insert("scale", frame=fade_start, index=axis)
            p.scale[axis] = 0.0
            p.keyframe_insert("scale", frame=fade_end, index=axis)
        # Normal cylinder
        if i < len(normals_objs):
            n = normals_objs[i]
            n.scale[2] = n.scale[2]
            n.keyframe_insert("scale", frame=fade_start, index=2)
            n.scale[2] = 0.0
            n.keyframe_insert("scale", frame=fade_end, index=2)

    # Pop in TX[0] and RX[0] at their array positions, then translate them
    # into the close-up pedagogical positions over the dolly window.
    tx0, rx0 = tx_objs[0], rx_objs[0]
    tx0_close = Vector(focal_pos) + Vector(TX0_CLOSE_OFFSET)
    rx0_close = Vector(focal_pos) + Vector(RX0_CLOSE_OFFSET)

    for obj, target_loc in ((tx0, tx0_close), (rx0, rx0_close)):
        for axis in range(3):
            obj.scale[axis] = 0.0
            obj.keyframe_insert("scale", frame=s + 5, index=axis)
            obj.scale[axis] = 1.0
            obj.keyframe_insert("scale", frame=e, index=axis)
        # Hold start position for first 5 frames, then slide to close-up.
        obj.keyframe_insert("location", frame=s + 5)
        obj.location = target_loc
        obj.keyframe_insert("location", frame=e)


def animate_S5_wavefront_out(wave_marker, focal_pos, tx0_pos):
    """Tracer travels TX→focal point."""
    s, e = _frames("S5")
    # Visible at start, hidden at end (gets absorbed at the point)
    for axis in range(3):
        wave_marker.scale[axis] = 0.0
        wave_marker.keyframe_insert("scale", frame=s - 2, index=axis)
        wave_marker.scale[axis] = 1.0
        wave_marker.keyframe_insert("scale", frame=s + 4, index=axis)
        wave_marker.scale[axis] = 1.0
        wave_marker.keyframe_insert("scale", frame=e - 4, index=axis)
        wave_marker.scale[axis] = 0.0
        wave_marker.keyframe_insert("scale", frame=e, index=axis)

    # Linear path TX → focal
    wave_marker.location = Vector(tx0_pos)
    wave_marker.keyframe_insert("location", frame=s)
    wave_marker.location = Vector(focal_pos)
    wave_marker.keyframe_insert("location", frame=e)


def animate_S6_wavefront_back(wave_marker, focal_pos, rx0_pos):
    """Tracer travels focal point → RX (return path)."""
    s, e = _frames("S6")
    wave_marker.location = Vector(focal_pos)
    wave_marker.keyframe_insert("location", frame=s)
    wave_marker.location = Vector(rx0_pos)
    wave_marker.keyframe_insert("location", frame=e)
    for axis in range(3):
        wave_marker.scale[axis] = 0.0
        wave_marker.keyframe_insert("scale", frame=s - 2, index=axis)
        wave_marker.scale[axis] = 1.0
        wave_marker.keyframe_insert("scale", frame=s + 4, index=axis)
        wave_marker.scale[axis] = 1.0
        wave_marker.keyframe_insert("scale", frame=e - 4, index=axis)
        wave_marker.scale[axis] = 0.0
        wave_marker.keyframe_insert("scale", frame=e, index=axis)


def animate_S7_phasor(focal_point_obj, focal_pos_orig, tx0_pos, rx0_pos,
                       phasor_anchor, ring, pivot):
    """The focal point oscillates radially (toward/away from the radar
    midpoint) while the phasor inset rotates by Δϕ = -2π·R/λ.  As the point
    decelerates the phasor visibly stops spinning.

    Implementation: drive the focal point's location with a damped sinusoid;
    drive the phasor pivot's rotation_euler.z (camera-axis) with the
    resulting round-trip phase.
    """
    s, e = _frames("S7")
    midpoint = (Vector(tx0_pos) + Vector(rx0_pos)) * 0.5
    radial = (Vector(focal_pos_orig) - midpoint)
    r0 = radial.length
    radial_unit = radial.normalized()

    # Show ring + shaft + head (ring scales in, then shaft/head scale in via
    # the pivot's stashed child names).
    for axis in range(3):
        ring.scale[axis] = 0.0
        ring.keyframe_insert("scale", frame=s, index=axis)
        ring.scale[axis] = 1.0
        ring.keyframe_insert("scale", frame=s + 12, index=axis)
    child_names = pivot.get("_visible_children", [])
    for cn in child_names:
        c = bpy.data.objects.get(cn)
        if c is None:
            continue
        for axis in range(3):
            c.scale[axis] = 0.0
            c.keyframe_insert("scale", frame=s + 6, index=axis)
            c.scale[axis] = 1.0
            c.keyframe_insert("scale", frame=s + 18, index=axis)

    # Per-frame keyframes: oscillate the point and update phasor rotation.
    # Drive the pivot's local Z (camera-forward axis), so children rotate in
    # the inset's screen plane.
    n_frames = e - s
    for k in range(n_frames + 1):
        f = s + k
        t = k / max(n_frames, 1)
        env = math.exp(-2.0 * t)             # decays so motion slows
        d = 0.25 * env * math.sin(t * 12.0)  # ±25 cm radial sweep, damped
        new_R = r0 + d
        new_pos = midpoint + radial_unit * new_R

        focal_point_obj.location = new_pos
        focal_point_obj.keyframe_insert("location", frame=f)

        # Round-trip phase: -2π·(R_TX + R_RX)/λ.  We use 2·R as a stand-in
        # (TX and RX co-located near the midpoint for this visualisation).
        phase = -2 * math.pi * (2.0 * new_R) / LAMBDA_R
        pivot.rotation_euler[2] = phase
        pivot.keyframe_insert("rotation_euler", frame=f, index=2)


def animate_S8_range_profile(focal_point_obj, focal_pos_orig, tx0_pos, rx0_pos,
                                bars, bar_mat_peak):
    """Range-profile bars activate; the *peak* bar walks across as the focal
    point moves in range."""
    s, e = _frames("S8")
    n_bins = len(bars)
    midpoint = (Vector(tx0_pos) + Vector(rx0_pos)) * 0.5
    radial_unit = (Vector(focal_pos_orig) - midpoint).normalized()
    r0 = (Vector(focal_pos_orig) - midpoint).length

    # Reveal: grow each bar's height (Y axis = screen-up) from 0 to an idle
    # baseline so the full row reads as a histogram before any peak appears.
    # Pre-S8 keyframe at frame 1 is required because F-curves use *constant*
    # pre-extrapolation by default — without it, the first per-frame keyframe
    # at frame s would back-fill the entire pre-S8 range with a Gaussian
    # peak, leaking the bars onto S0–S7.
    H_BASELINE = 0.014
    H_PEAK     = 0.080
    for b in bars:
        b.scale[1] = 0.001
        b.keyframe_insert("scale", frame=1, index=1)
        b.keyframe_insert("scale", frame=s, index=1)
        b.scale[1] = H_BASELINE
        b.keyframe_insert("scale", frame=s + 10, index=1)

    n_frames = e - s
    for k in range(n_frames + 1):
        f = s + k
        t = k / max(n_frames, 1)
        # Smoothstep traversal from r0 - 0.6 to r0 + 0.6 m
        d = (math.cos(t * math.pi) * -0.6)
        new_R = r0 + d
        new_pos = midpoint + radial_unit * new_R
        focal_point_obj.location = new_pos
        focal_point_obj.keyframe_insert("location", frame=f)

        # Determine which bin currently holds the peak
        peak_bin = int(round((new_R - r0) / DELTA_R + n_bins / 2))
        peak_bin = max(0, min(n_bins - 1, peak_bin))

        # Each bar's height: Gaussian PSF around peak_bin, plus a baseline so
        # non-peak bars stay visible as the surrounding histogram.
        for bi, bar in enumerate(bars):
            sigma = 1.4
            h = math.exp(-0.5 * ((bi - peak_bin) / sigma) ** 2)
            bar.scale[1] = H_BASELINE + h * (H_PEAK - H_BASELINE)
            bar.keyframe_insert("scale", frame=f, index=1)

            # Peak material assignment is one-shot — only on bar nearest peak.
            if bi == peak_bin and bar_mat_peak.name not in [m.name for m in bar.data.materials]:
                bar.data.materials.clear()
                bar.data.materials.append(bar_mat_peak)


def animate_S9_multi_rx(rx_objs):
    """Reveal RX[1..3] (already RX[0]).  Their phasors would be staggered in
    azimuth — left as a stub: the user can extend by adding 4 phasor copies
    keyframed with phase offsets ϕ_m = -2π d sin(θ) m / λ."""
    s, e = _frames("S9")
    for i in range(1, min(4, len(rx_objs))):
        rx = rx_objs[i]
        for axis in range(3):
            rx.scale[axis] = 0.0
            rx.keyframe_insert("scale", frame=s, index=axis)
            rx.scale[axis] = 1.0
            rx.keyframe_insert("scale", frame=s + 30 + i * 5, index=axis)


def animate_S10_tdm(tx_objs):
    """TDM: TX[1..11] flash on sequentially.  Keep all visible after their
    onset to mark the virtual array filling out."""
    s, e = _frames("S10")
    n_tx = len(tx_objs)
    span = e - s
    step = span // max(n_tx - 1, 1)
    for i in range(1, n_tx):
        t = s + i * step
        for axis in range(3):
            tx_objs[i].scale[axis] = 0.0
            tx_objs[i].keyframe_insert("scale", frame=t, index=axis)
            tx_objs[i].scale[axis] = 1.0
            tx_objs[i].keyframe_insert("scale", frame=t + 6, index=axis)


def animate_S11_az_fft(ra_plane, ra_emission_node):
    """Bring up the RA-map plane — single-reflector content shown as a
    one-pixel-bright proxy via the emission color (blue→red sweep)."""
    s, e = _frames("S11")
    for axis in range(3):
        ra_plane.scale[axis] = 0.0
        ra_plane.keyframe_insert("scale", frame=1, index=axis)
        ra_plane.keyframe_insert("scale", frame=s, index=axis)
        ra_plane.scale[axis] = 1.0
        ra_plane.keyframe_insert("scale", frame=s + 24, index=axis)

    # Animate brightness rising
    ra_emission_node.inputs["Color"].default_value = (0.05, 0.0, 0.0, 1.0)
    ra_emission_node.inputs["Color"].keyframe_insert("default_value",
                                                       frame=s + 24)
    ra_emission_node.inputs["Color"].default_value = (0.85, 0.18, 0.18, 1.0)
    ra_emission_node.inputs["Color"].keyframe_insert("default_value",
                                                       frame=e)


def animate_S12_fill_ra(points, normals_objs, ra_emission_node):
    """Re-add many points and gradually saturate the RA-map plane to a
    near-white emission, suggesting the scene "filling in"."""
    s, e = _frames("S12")
    # Re-spawn points in waves
    n = len(points)
    span = e - s - 30
    for i, p in enumerate(points[1:], start=1):           # skip focal
        t0 = s + int((i / max(n - 1, 1)) * span * 0.95)
        t1 = t0 + 6
        for axis in range(3):
            p.scale[axis] = 0.0
            p.keyframe_insert("scale", frame=t0, index=axis)
            p.scale[axis] = 1.0
            p.keyframe_insert("scale", frame=t1, index=axis)
        if i < len(normals_objs):
            nrm = normals_objs[i]
            nrm.scale[2] = 0.0
            nrm.keyframe_insert("scale", frame=t0, index=2)
            nrm.scale[2] = 1.0
            nrm.keyframe_insert("scale", frame=t1, index=2)

    # RA map saturates
    ra_emission_node.inputs["Color"].default_value = (0.85, 0.18, 0.18, 1.0)
    ra_emission_node.inputs["Color"].keyframe_insert("default_value",
                                                       frame=s)
    ra_emission_node.inputs["Color"].default_value = (1.0, 0.95, 0.85, 1.0)
    ra_emission_node.inputs["Color"].keyframe_insert("default_value",
                                                       frame=e)


# ════════════════════════════════════════════════════════════════════════════
# DRIVER
# ════════════════════════════════════════════════════════════════════════════

def main():
    clear_scene()
    setup_render()
    setup_world()
    setup_lighting()

    P, N, E, antennas, mesh_path = load_prep_data()

    # Pick a focal point — a point near the centre of the scene with a
    # roughly upward-facing normal (for visual clarity).
    centre = np.asarray(antennas["center"])
    dist = np.linalg.norm(P - centre, axis=1)
    upness = N[:, 2]
    score = -dist - 2.0 * np.maximum(upness - 0.4, 0)
    # Pick the highest-scoring point that's between 1.5 and 4 m from centre
    mask = (dist > 1.5) & (dist < 4.0) & (upness > 0.0)
    if mask.sum() == 0:
        mask = np.ones_like(dist, dtype=bool)
    score[~mask] = np.inf
    focal_idx = int(np.argmin(score))
    focal_pos = P[focal_idx]

    # Make sure that point is index 0 in our subsampled set, so it's the
    # "kept" focal during S4.
    rng = np.random.default_rng(0)
    idx_pool = list(rng.choice(len(P), N_SCENE_POINTS_FULL, replace=False))
    if focal_idx in idx_pool:
        idx_pool.remove(focal_idx)
    idx_pool = [focal_idx] + idx_pool[:N_SCENE_POINTS_FULL - 1]
    P_sub = P[idx_pool]
    N_sub = N[idx_pool]
    E_sub = E[idx_pool]

    # ── Collections ─────────────────────────────────────────────────────────
    col_scene  = _make_collection("Scene")
    col_points = _make_collection("Points")
    col_array  = _make_collection("RadarArray")
    col_inset  = _make_collection("Insets")

    # ── Geometry ────────────────────────────────────────────────────────────
    mesh_obj = load_mesh(mesh_path)
    if mesh_obj is not None:
        _link(mesh_obj, col_scene)

    points, normals_objs, _, eps_mats = build_points(
        P_sub, N_sub, E_sub, n_points=N_SCENE_POINTS_FULL,
        collection=col_points)

    tx_objs, rx_objs, board_pos = build_antennas(antennas, centre, col_array)

    cam, cam_target = setup_camera(centre.tolist())
    _link(cam, col_inset)
    _link(cam_target, col_inset)

    wave_marker = build_wavefront_marker("WaveTracer", WAVE_COLOR, col_inset,
                                            emission_strength=12.0)

    phasor_anchor, phasor_ring, phasor_pivot = build_phasor_inset(cam, col_inset)
    range_anchor, range_bars, range_peak_mat = build_range_profile_inset(
        cam, col_inset, n_bins=32)
    ra_plane, ra_emission = build_ra_map_inset(cam, col_inset)

    # ── Animations ──────────────────────────────────────────────────────────
    animate_S0_mesh(mesh_obj)
    animate_S1_points(points)
    animate_S2_normals(normals_objs)
    animate_S3_materials(points, eps_mats)
    animate_S4_zoom(cam, cam_target, focal_pos, points, normals_objs,
                     tx_objs, rx_objs)
    animate_S5_wavefront_out(wave_marker, focal_pos.tolist(),
                                 tx_objs[0].location)
    animate_S6_wavefront_back(wave_marker, focal_pos.tolist(),
                                  rx_objs[0].location)
    # S7 needs the focal point object — points[0] is the focal sphere
    focal_obj = points[0]
    animate_S7_phasor(focal_obj, focal_pos.tolist(),
                       tx_objs[0].location, rx_objs[0].location,
                       phasor_anchor, phasor_ring, phasor_pivot)
    animate_S8_range_profile(focal_obj, focal_pos.tolist(),
                                tx_objs[0].location, rx_objs[0].location,
                                range_bars, range_peak_mat)
    animate_S9_multi_rx(rx_objs)
    animate_S10_tdm(tx_objs)
    animate_S11_az_fft(ra_plane, ra_emission)
    animate_S12_fill_ra(points, normals_objs, ra_emission)

    # ── Save .blend for iteration ───────────────────────────────────────────
    blend_path = os.path.join(OUT_DIR, "..", "blender_video.blend")
    os.makedirs(os.path.dirname(blend_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved .blend → {blend_path}")
    print(f"To render headless: blender -b {blend_path} -a")


if __name__ == "__main__":
    main()
