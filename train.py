#!/usr/bin/env python3
"""
Single-frame FMCW inverse renderer training script (Final Renderer).

Combines the physics-based renderer_final (end-to-end multibounce) with the
training framework from train_simple_single_frame_sionna.py.

Key changes from the Sionna variant:
- Uses mmir.renderer instead of renderer_ref_ti_claude_vectorized_sionna
- End-to-end differentiable rendering (single AD graph, no two-phase cache)
- Multibounce support via max_bounces config (NEE at every bounce)
- Seed rotation for MC variance reduction across iterations

Features:
- Physics-based 6-parameter material model (eps', eps'', sigma_h, l_c, tau, thickness)
- Reparameterized physics params (softplus/sigmoid for bounded optimization)
- End-to-end differentiable multibounce rendering
- ITU material initialization
- Error-map initialization
- 7-component BSDF decomposition visualization
- SMS for specular paths (within same AD graph)
- Multi-domain loss (ADC + RA + Phase + SSIM + LPIPS)
- Multi-chirp loss with per-chirp complex scalar alignment
- 30+ metrics with publication target evaluation
- JSON config system for reproducibility
- tqdm progress bar
- Unit phasor phase metrics
- Staged training (pose first, then materials)
- Comprehensive checkpointing and visualization

Usage:
    python train_simple_single_frame_final.py --config training_config_final.json
"""

# ============================================================================
# Section 1: Imports & Environment Setup
# ============================================================================

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import json
import time
import math
import copy
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

import torch

# DrJit memory optimizations (BEFORE mitsuba import)
import drjit as dr
dr.set_flag(dr.JitFlag.VCallOptimize, False)
dr.set_flag(dr.JitFlag.LoopOptimize, False)

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Renderer imports (renderer_final: end-to-end multibounce)
from mmir.renderer import (
    FMCWRendererRef, RenderConfigRef,
    reparameterize_physics_params, inverse_reparameterize, LR_SCALES,
    create_drjit_raw_params,
    CachedGeometry,
    get_material_properties,
    vertex_to_triangle_materials,
    PerVertexParameterization, PerTriangleParameterization,
    DiffractionConfig,
    create_pose_params,
)

# RA & loss utilities (shared with TSSF)
from mmir.data.ra_utils import adc_to_ra_image, ra_polar_to_cartesian
from mmir.losses.drjit_ra_loss import compute_ra_loss_with_gradients
from mmir.data.io_utils import compute_range_res_from_cfg
from mmir.losses.loss_utils import (
    compute_psnr, compute_ssim,
    minmax_normalize_numpy, minmax_normalize_torch,
)
from mmir.losses.multi_chirp_loss import compute_multi_chirp_loss

try:
    import trimesh
except ImportError:
    trimesh = None

# Optional LPIPS import
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    lpips = None


# ============================================================================
# Section 2: Dataclasses
# ============================================================================

def load_all_chirps(path: str, device: str = 'cpu') -> torch.Tensor:
    """Load all chirps from GT ADC file for multi-chirp training.

    Args:
        path: Path to .npy file containing GT ADC data
        device: Device to load tensor to

    Returns:
        torch.Tensor: Shape [n_chirp, n_rx, n_tx, n_adc, 2] (real/imag)
    """
    arr = np.load(path)
    if arr.ndim == 4:
        pass  # (n_chirp, n_rx, n_tx, n_adc) complex
    elif arr.ndim == 3:
        arr = arr[np.newaxis, ...]  # single chirp -> add dimension
    else:
        raise ValueError(f"Unexpected GT shape: {arr.shape}")

    arr_torch = torch.from_numpy(arr.astype(np.complex64))
    arr_real_imag = torch.view_as_real(arr_torch).float()
    return arr_real_imag.to(device)


@dataclass
class LossConfigSionna:
    """Unified configuration for all loss functions (from TSSF)."""
    # === Loss weights (0 = disabled) ===
    adc_mag_weight: float = 0.0              # ADC magnitude loss (disabled by default for Sionna)
    adc_phase_weight: float = 0.0            # ADC phase loss
    ra_mag_weight: float = 1.0               # RA magnitude loss (primary for Sionna)
    ra_mag_ssim_weight: float = 0.0          # SSIM on RA
    ra_mag_perceptual_weight: float = 0.0    # LPIPS on RA
    mat_regularization_weight: float = 0.0   # Material regularization (0: physics params use reparameterization)

    # === ADC magnitude loss settings ===
    adc_mag_use_log: bool = False
    adc_mag_loss_type: str = "l2"

    # === ADC phase loss settings ===
    adc_phase_loss_type: str = "none"

    # === RA magnitude loss settings ===
    ra_mag_use_log: bool = True              # Log-compression (recommended for radar)
    ra_mag_loss_type: str = "l2"

    # === Log-compression parameters ===
    log_epsilon: float = 1e-6
    log_scale: float = 1.0

    # === Phase loss parameters ===
    phase_magnitude_weighting: bool = True
    phase_magnitude_power: float = 0.5
    phase_reference_channel: int = 0
    phase_use_gt_magnitude: bool = True
    phase_magnitude_clip_max: float = 0.0

    # === Other parameters ===
    huber_delta: float = 1.0

    # === Normalization mode ===
    use_joint_normalization: bool = False

    # === Parameter anchoring (Option B2) ===
    use_parameter_anchoring: bool = False
    parameter_anchor_checkpoint: str = ""
    parameter_anchor_weight: float = 0.01

    # === Multi-chirp loss ===
    use_multi_chirp_loss: bool = False
    multi_chirp_phase_weight: float = 0.0
    multi_chirp_top_percentile: float = 0.9

    # === Phase metrics computation ===
    phase_metrics_type: str = "unit_phasor"


@dataclass
class TrainingConfigSionna:
    """Training configuration for Sionna renderer optimization."""

    # ---- Files & Paths ----
    config_file: str = ""
    scene_file: str = ""
    gt_adc_file: str = ""
    output_dir: str = "output/train_sionna"
    tx_pattern_file: str = ""
    rx_pattern_file: str = ""

    # ---- Sionna Renderer Config ----
    n_hits_per_rx: int = 500
    n_rays_per_res: int = 16
    n_avg_samples: int = 1
    bsdf_model: str = "mmwave_scalar"
    mmwave_polarization: str = "vertical"
    hemisphere_sampling: str = "cosine"
    double_sided: bool = True
    use_pattern_importance_sampling: bool = False

    # ---- Specular Path Finding ----
    enable_image_method: bool = False
    enable_sms: bool = True
    sms_max_iterations: int = 20
    sms_solver_threshold: float = 1e-5
    sms_use_smooth_normals: bool = False
    sms_cache_interval: int = 0  # 0=recompute every iter, N=cache for N iters
    enable_diffraction: bool = False
    use_patch_clustering: bool = True
    patch_angle_threshold_deg: float = 10.0
    patch_distance_threshold: float = 0.05
    patch_max_tris: int = 2000
    use_spatial_adjacency: bool = False
    spatial_radius: float = 0.05

    # ---- End-to-End Differentiable & Multibounce ----
    use_end_to_end_ad: bool = True
    max_bounces: int = 3
    nee_every_bounce: bool = True
    rr_start_bounce: int = 3
    rr_prob: float = 0.5
    e2e_ray_seed_rotation_interval: int = 10
    e2e_phase_chunk_size: int = 64
    enable_grad_phase: bool = False
    enable_boundary_gradients: bool = True

    # ---- Physics Material Model ----
    physics_mode: bool = True
    param_mode: str = "per_vertex"
    initial_material: str = ""
    enable_error_map_init: bool = False
    material_columns: int = 6

    # ---- Material Bounds (raw-space clamps for physical plausibility) ----
    # Values are raw-space (log or logit) bounds applied after each Adam step.
    # See reparameterization.py for the mapping: physics = exp(clamp(raw, lo, hi))
    #
    # eps_imag upper bound:
    #   original 16.0 → exp(16)≈8.9e6 (allows metals, but acts as amplitude knob)
    #   4.6 → exp(4.6)≈100   (covers all dielectrics with 10x margin)
    #   2.3 → exp(2.3)≈10    (tight: max ITU dielectric at 77 GHz is ~10)
    eps_imag_raw_upper: float = 4.6
    #
    # sigma_h (RMS surface roughness) upper bound:
    #   original -7.0 → exp(-7)≈0.9mm  (too low: ITU asphalt=5mm, ground=10mm, vegetation=50mm)
    #   -2.3 → exp(-2.3)≈100mm         (covers all ITU materials incl. vegetation)
    #   -3.0 → exp(-3)≈50mm            (matches ITU vegetation, the roughest material)
    sigma_h_raw_upper: float = -2.3
    #
    # thickness upper bound:
    #   original -1.2 → exp(-1.2)≈0.3m  (slightly low: ITU ground=0.5m)
    #   -0.7 → exp(-0.7)≈0.5m           (matches ITU ground thickness)
    thickness_raw_upper: float = -0.7

    # ---- Optimization ----
    num_iterations: int = 200
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # ---- Per-Group Learning Rates ----
    lr_materials: float = 0.5
    lr_pose_rotation: float = 5e-3
    lr_pose_translation: float = 5e-4
    lr_normals: float = 0.01
    lr_vertex_positions: float = 5e-4
    lr_patterns: float = 0.05

    # ---- Gradient Clipping (T2) ----
    clip_materials: float = 1.0
    clip_pose_rotation: float = 0.5
    clip_pose_translation: float = 0.5
    clip_normals: float = 0.5
    clip_vertex_positions: float = 0.1
    clip_patterns: float = 1.0

    # ---- Parameter Learning Flags ----
    LEARN_MAT: bool = True
    LEARN_POSE: bool = False
    LEARN_NRM: bool = False
    LEARN_VTX: bool = False
    LEARN_PAT: bool = False

    # ---- Staged Training ----
    enable_staged_training: bool = False
    stage1_iterations: int = 50
    material_warmup_iters: int = 5

    # ---- LR Scheduling (T4) ----
    enable_lr_schedule: bool = True
    lr_warmup_iters: int = 5
    lr_warmup_factor: float = 0.3

    # ---- Post-Step Constraints (T5) ----
    pose_rotation_max_rad: float = 0.0349
    pose_translation_max_m: float = 1.95e-3
    geom_laplacian_weight: float = 0.3

    # ---- Pose Initialization ----
    init_pitch_deg: float = 0.0
    init_roll_deg: float = 0.0
    init_yaw_deg: float = 0.0
    init_translation_x: float = 0.0
    init_translation_y: float = 0.0
    init_translation_z: float = 0.0

    # ---- Loss Configuration ----
    loss_config: LossConfigSionna = field(default_factory=LossConfigSionna)

    # ---- Visualization ----
    visualization_interval: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 1
    viz_scale: str = "dB"
    viz_db_floor: float = -40.0

    # ---- Convergence ----
    target_correlation: float = 0.90
    target_mse: float = 1e-4
    enable_early_stopping: bool = False

    # ---- Antenna Patterns ----
    pattern_mode: str = "shared"

    # ---- Checkpoint Loading ----
    init_checkpoint: str = ""

    # ---- Computed ----
    frozen_params: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Resolve antenna pattern paths
        if not self.tx_pattern_file and self.pattern_mode == "shared":
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.tx_pattern_file = os.path.join(base, "assets/antenna_pattern/MMWCAS/tx1_76.npy")
            self.rx_pattern_file = os.path.join(base, "assets/antenna_pattern/MMWCAS/rx1_76.npy")
        elif not self.tx_pattern_file and self.pattern_mode == "mmwcas":
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.tx_pattern_file = os.path.join(base, "assets/antenna_pattern/MMWCAS/tx1_76.npy")
            self.rx_pattern_file = os.path.join(base, "assets/antenna_pattern/MMWCAS/rx1_76.npy")

        # Auto-set image method: only if SMS disabled
        if self.enable_sms:
            self.enable_image_method = False

        # Physics mode always uses 6 columns
        if self.physics_mode:
            self.material_columns = 6

        # Compute frozen_params from LEARN_* flags
        if not self.frozen_params:
            self.frozen_params = []
            if not self.LEARN_MAT:
                self.frozen_params.append('materials')
            if not self.LEARN_POSE:
                self.frozen_params.extend(['pose_rotation', 'pose_translation'])
            if not self.LEARN_NRM:
                self.frozen_params.append('normals')
            if not self.LEARN_VTX:
                self.frozen_params.append('vertex_positions')
            if not self.LEARN_PAT:
                self.frozen_params.append('patterns')


def load_config_from_json(json_path: str) -> TrainingConfigSionna:
    """Load config from JSON file."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    config = TrainingConfigSionna()

    for key, value in config_dict.items():
        if hasattr(config, key):
            if key == 'loss_config' and isinstance(value, dict):
                loss_cfg = LossConfigSionna()
                for lk, lv in value.items():
                    if hasattr(loss_cfg, lk):
                        setattr(loss_cfg, lk, lv)
                config.loss_config = loss_cfg
            else:
                setattr(config, key, value)

    # Reset frozen_params so __post_init__ recomputes from LEARN_* flags
    config.frozen_params = []
    config.__post_init__()
    return config


# ============================================================================
# Section 3: ParameterManagerSionna
# ============================================================================

class ParameterManagerSionna:
    """Manages all differentiable parameters for Sionna renderer training.

    Materials live in NumPy space (6-param physics with reparameterization).
    Pose/normals/geometry also in NumPy. Fresh DrJit copies created each iteration.
    """

    def __init__(self, renderer: FMCWRendererRef, config: TrainingConfigSionna,
                 mesh_path: str):
        self.renderer = renderer
        self.config = config
        self.mesh_path = mesh_path

        # Load mesh geometry via trimesh (HFMO pattern)
        if trimesh is None:
            raise ImportError("trimesh is required for ParameterManagerSionna")
        mesh = trimesh.load(mesh_path)
        self.mesh_vertices = np.array(mesh.vertices, dtype=np.float32)
        self.mesh_faces = np.array(mesh.faces, dtype=np.int32)
        self.n_vertices = len(self.mesh_vertices)
        self.n_triangles = len(self.mesh_faces)

        # Determine optimization size and set material parameterization
        if config.param_mode == 'per_vertex':
            self.n_opt_params = self.n_vertices
            self.material_param = PerVertexParameterization(
                self.n_vertices, faces=self.mesh_faces)
        else:
            self.n_opt_params = self.n_triangles
            self.material_param = PerTriangleParameterization(self.n_triangles)
        # Propagate to renderer (sets on both renderer and integrator)
        renderer.material_param = self.material_param

        # ---- Material State (NumPy, 6-param physics) ----
        self.raw_params = np.zeros((self.n_opt_params, 6), dtype=np.float32)
        self._init_default_physics_params()

        # ---- Pose State (NumPy, 6 scalars) ----
        self.pose_params_np = np.zeros(6, dtype=np.float32)
        if config.LEARN_POSE:
            self.pose_params_np[:3] = np.radians([
                config.init_pitch_deg, config.init_roll_deg, config.init_yaw_deg
            ])
            self.pose_params_np[3:] = [
                config.init_translation_x, config.init_translation_y, config.init_translation_z
            ]

        # ---- Normal State (NumPy) ----
        self.normal_params_np = None
        if config.LEARN_NRM:
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                self.normal_params_np = np.array(mesh.vertex_normals, dtype=np.float32)
            else:
                # Default: +Y up normals
                self.normal_params_np = np.zeros((self.n_vertices, 3), dtype=np.float32)
                self.normal_params_np[:, 1] = 1.0

        # ---- Geometry State (NumPy offsets) ----
        self.vertex_offset_np = None
        self.laplacian_neighbors = None
        if config.LEARN_VTX:
            self.vertex_offset_np = np.zeros((self.n_vertices, 3), dtype=np.float32)
            self._build_laplacian_adjacency()

        # ---- Antenna Pattern Loaders ----
        self.tx_pattern_loader = getattr(renderer, 'tx_pattern_loader', None)
        self.rx_pattern_loader = getattr(renderer, 'rx_pattern_loader', None)

        # ---- Wavelength ----
        self.wavelength = 3e8 / 77e9  # ~3.9mm

        # ---- Radar params ----
        self.NT = renderer.scene_ctx.tx_array.num_elements
        self.NR = renderer.scene_ctx.rx_array.num_elements
        self.K = renderer.integrator.num_samples

    def _init_default_physics_params(self):
        """Initialize with neutral physics defaults."""
        defaults = np.array([4.0, 0.1, 3e-5, 0.01, 0.5, 0.1], dtype=np.float32)
        physics_params = np.tile(defaults, (self.n_opt_params, 1))
        self.raw_params = inverse_reparameterize(physics_params)

    def initialize_from_itu(self, material_name: str):
        """Initialize physics params from ITU material database."""
        props = get_material_properties(material_name, 77e9)
        physics_params = reparameterize_physics_params(self.raw_params)

        physics_params[:, 0] = props['eps_real']
        physics_params[:, 1] = props['eps_imag']
        if 'sigma_h' in props:
            physics_params[:, 2] = props['sigma_h']
        if 'l_c' in props:
            physics_params[:, 3] = props['l_c']
        if 'thickness' in props:
            physics_params[:, 5] = props['thickness']

        self.raw_params = inverse_reparameterize(physics_params)

        phys = reparameterize_physics_params(self.raw_params)
        print(f"ITU init '{material_name}':")
        print(f"  eps_real={phys[:, 0].mean():.3f}, eps_imag={phys[:, 1].mean():.4f}")
        print(f"  sigma_h={phys[:, 2].mean():.2e}, l_c={phys[:, 3].mean():.4f}")
        print(f"  tau={phys[:, 4].mean():.3f}, thickness={phys[:, 5].mean():.4f}")

    def set_renderer_materials(self):
        """Push raw_params -> physics -> renderer triangle materials."""
        physics_params = reparameterize_physics_params(self.raw_params)
        if self.config.param_mode == 'per_vertex':
            tri_materials = vertex_to_triangle_materials(physics_params, self.mesh_faces)
        else:
            tri_materials = physics_params
        self.renderer.triangle_materials = tri_materials

    def create_grad_enabled_params(self):
        """Create fresh DrJit params with gradients enabled for AD."""
        raw_drjit = create_drjit_raw_params(self.raw_params)

        pose_dr = None
        if self.config.LEARN_POSE and 'pose_rotation' not in self.config.frozen_params:
            pose_dr = create_pose_params(
                pitch=float(self.pose_params_np[0]),
                roll=float(self.pose_params_np[1]),
                yaw=float(self.pose_params_np[2]),
                tx=float(self.pose_params_np[3]),
                ty=float(self.pose_params_np[4]),
                tz=float(self.pose_params_np[5]),
                enable_grad=True,
            )

        normal_dr = None
        if self.config.LEARN_NRM and self.normal_params_np is not None \
                and 'normals' not in self.config.frozen_params:
            normal_dr = [mi.Float(self.normal_params_np[:, i].astype(np.float32))
                         for i in range(3)]
            for n in normal_dr:
                dr.enable_grad(n)

        vertex_offset_dr = None
        if self.config.LEARN_VTX and self.vertex_offset_np is not None \
                and 'vertex_positions' not in self.config.frozen_params:
            vertex_offset_dr = [mi.Float(self.vertex_offset_np[:, i].astype(np.float32))
                                for i in range(3)]
            for v in vertex_offset_dr:
                dr.enable_grad(v)

        pattern_loaders_dr = None
        if self.config.LEARN_PAT and 'patterns' not in self.config.frozen_params:
            pattern_loaders_dr = {}
            for key, loader in [('tx', self.tx_pattern_loader), ('rx', self.rx_pattern_loader)]:
                if loader is not None:
                    loader.enable_gradients()
                    pattern_loaders_dr[key] = loader

        return raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr

    def post_step_constraints(self):
        """Apply post-step constraints (T5)."""
        cfg = self.config

        # ---- Material raw_params: enforce physical bounds ----
        # Each column is clamped to a physically plausible range in raw (log/logit)
        # space. This prevents the optimizer from exploiting unconstrained params
        # as per-vertex amplitude knobs that don't generalize across viewing angles.
        #
        # Col 0 (eps_real):   sigmoid → [1.5, 10.0]  — already bounded, no extra clamp
        # Col 1 (eps_imag):   exp → [1e-3, exp(upper)] — tighten upper from 8.9e6 to ~100
        # Col 2 (sigma_h):   exp → [1e-7, exp(upper)] — widen upper from 0.9mm to ~100mm
        # Col 3 (l_c):       exp → [5e-4, 0.1]        — already physical, no extra clamp
        # Col 4 (tau):        sigmoid → [0.05, 0.95]   — already bounded, no extra clamp
        # Col 5 (thickness): exp → [1e-3, exp(upper)] — widen upper from 0.3m to ~0.5m
        self.raw_params[:, 1] = np.clip(
            self.raw_params[:, 1], -7.0, cfg.eps_imag_raw_upper
        )
        self.raw_params[:, 2] = np.clip(
            self.raw_params[:, 2], -16.0, cfg.sigma_h_raw_upper
        )
        self.raw_params[:, 5] = np.clip(
            self.raw_params[:, 5], -7.0, cfg.thickness_raw_upper
        )

        # Pose rotation: clamp to ±max_rad
        self.pose_params_np[:3] = np.clip(
            self.pose_params_np[:3],
            -cfg.pose_rotation_max_rad, cfg.pose_rotation_max_rad
        )
        # Pose translation: clamp norm to max_m
        t_norm = np.linalg.norm(self.pose_params_np[3:])
        if t_norm > cfg.pose_translation_max_m:
            self.pose_params_np[3:] *= cfg.pose_translation_max_m / t_norm

        # Normal re-normalization
        if self.normal_params_np is not None:
            norms = np.linalg.norm(self.normal_params_np, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.normal_params_np /= norms

        # Geometry Laplacian smoothing
        if self.vertex_offset_np is not None and cfg.geom_laplacian_weight > 0:
            self._apply_laplacian_reg()

    def _build_laplacian_adjacency(self):
        """Build per-vertex neighbor lists from mesh faces."""
        adj = defaultdict(set)
        for f in self.mesh_faces:
            v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
            adj[v0].update([v1, v2])
            adj[v1].update([v0, v2])
            adj[v2].update([v0, v1])
        self.laplacian_neighbors = [np.array(sorted(adj[i]), dtype=np.int32)
                                    for i in range(self.n_vertices)]

    def _apply_laplacian_reg(self):
        """Pull each vertex offset toward neighbor average."""
        if self.laplacian_neighbors is None or self.vertex_offset_np is None:
            return
        w = self.config.geom_laplacian_weight
        if w <= 0:
            return
        offsets = self.vertex_offset_np
        smoothed = np.zeros_like(offsets)
        for i, nbrs in enumerate(self.laplacian_neighbors):
            if len(nbrs) > 0:
                smoothed[i] = offsets[nbrs].mean(axis=0)
            else:
                smoothed[i] = offsets[i]
        self.vertex_offset_np = (1.0 - w) * offsets + w * smoothed

    def get_physics_param_means(self) -> Dict[str, float]:
        """Get mean values of all 6 physics params (for history tracking)."""
        phys = reparameterize_physics_params(self.raw_params)
        names = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau', 'thickness']
        return {f'{n}_mean': float(phys[:, i].mean()) for i, n in enumerate(names)}


# ============================================================================
# Section 4: SimpleAdamSionna Optimizer
# ============================================================================

class SimpleAdamSionna:
    """Adam optimizer for physics-parameterized training.

    Parameters live in NumPy space. Gradients extracted from DrJit after
    backward pass, sanitized with nan_to_num, then used for NumPy Adam updates.
    """

    def __init__(self, param_manager: ParameterManagerSionna, config: TrainingConfigSionna):
        self.pm = param_manager
        self.config = config
        self.beta1 = config.adam_beta1
        self.beta2 = config.adam_beta2
        self.eps = config.adam_eps
        self.t = 0

        # Per-group Adam states (m, v as NumPy arrays)
        self.states = {}
        self._init_states()

    def _init_states(self):
        """Initialize Adam momentum buffers for all parameter groups."""
        n = self.pm.n_opt_params
        # Materials: 6 separate columns
        self.states['materials'] = {
            'm': [np.zeros(n, dtype=np.float32) for _ in range(6)],
            'v': [np.zeros(n, dtype=np.float32) for _ in range(6)],
        }
        # Pose: 6 scalars
        self.states['pose_rotation'] = {
            'm': np.zeros(3, dtype=np.float32),
            'v': np.zeros(3, dtype=np.float32),
        }
        self.states['pose_translation'] = {
            'm': np.zeros(3, dtype=np.float32),
            'v': np.zeros(3, dtype=np.float32),
        }
        # Normals: n_vertices x 3
        nv = self.pm.n_vertices
        self.states['normals'] = {
            'm': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
            'v': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
        }
        # Vertex positions: n_vertices x 3
        self.states['vertex_positions'] = {
            'm': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
            'v': [np.zeros(nv, dtype=np.float32) for _ in range(3)],
        }
        # Patterns: placeholder (sized when first gradient arrives)
        self.states['patterns'] = {}

    def step(self, gradients: Dict[str, Any], lr_scale: float = 1.0):
        """Perform one Adam step with extracted gradients."""
        self.t += 1
        cfg = self.config

        for group_name, grad in gradients.items():
            if group_name in cfg.frozen_params:
                continue

            # Get learning rate
            lr_map = {
                'materials': cfg.lr_materials,
                'pose_rotation': cfg.lr_pose_rotation,
                'pose_translation': cfg.lr_pose_translation,
                'normals': cfg.lr_normals,
                'vertex_positions': cfg.lr_vertex_positions,
                'patterns': cfg.lr_patterns,
            }
            base_lr = lr_map.get(group_name, 0.01)
            lr = base_lr * lr_scale

            state = self.states.get(group_name)
            if state is None:
                continue

            if group_name == 'materials':
                # grad shape: (n_opt, 6) - update per column with LR_SCALES
                for i in range(6):
                    g = grad[:, i] * LR_SCALES[i]
                    state['m'][i] = self.beta1 * state['m'][i] + (1 - self.beta1) * g
                    state['v'][i] = self.beta2 * state['v'][i] + (1 - self.beta2) * g ** 2
                    m_hat = state['m'][i] / (1 - self.beta1 ** self.t)
                    v_hat = state['v'][i] / (1 - self.beta2 ** self.t)
                    self.pm.raw_params[:, i] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

            elif group_name in ('pose_rotation', 'pose_translation'):
                idx_start = 0 if group_name == 'pose_rotation' else 3
                g = grad
                state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * g
                state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * g ** 2
                m_hat = state['m'] / (1 - self.beta1 ** self.t)
                v_hat = state['v'] / (1 - self.beta2 ** self.t)
                self.pm.pose_params_np[idx_start:idx_start + 3] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

            elif group_name == 'normals' and self.pm.normal_params_np is not None:
                # Exponential-map update on unit sphere
                for i in range(3):
                    g = grad[:, i] if grad.ndim == 2 else grad[i]
                    state['m'][i] = self.beta1 * state['m'][i] + (1 - self.beta1) * g
                    state['v'][i] = self.beta2 * state['v'][i] + (1 - self.beta2) * g ** 2
                    m_hat = state['m'][i] / (1 - self.beta1 ** self.t)
                    v_hat = state['v'][i] / (1 - self.beta2 ** self.t)
                    step_i = lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    self.pm.normal_params_np[:, i] -= step_i

            elif group_name == 'vertex_positions' and self.pm.vertex_offset_np is not None:
                for i in range(3):
                    g = grad[:, i] if grad.ndim == 2 else grad[i]
                    state['m'][i] = self.beta1 * state['m'][i] + (1 - self.beta1) * g
                    state['v'][i] = self.beta2 * state['v'][i] + (1 - self.beta2) * g ** 2
                    m_hat = state['m'][i] / (1 - self.beta1 ** self.t)
                    v_hat = state['v'][i] / (1 - self.beta2 ** self.t)
                    step_i = lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    self.pm.vertex_offset_np[:, i] -= step_i

            elif group_name == 'patterns' and isinstance(grad, dict):
                # Pattern gradients: dict of {key: np.ndarray}
                for pat_key, pat_grad in grad.items():
                    if pat_key not in state:
                        state[pat_key] = {
                            'm': np.zeros_like(pat_grad),
                            'v': np.zeros_like(pat_grad),
                        }
                    ps = state[pat_key]
                    ps['m'] = self.beta1 * ps['m'] + (1 - self.beta1) * pat_grad
                    ps['v'] = self.beta2 * ps['v'] + (1 - self.beta2) * pat_grad ** 2
                    m_hat = ps['m'] / (1 - self.beta1 ** self.t)
                    v_hat = ps['v'] / (1 - self.beta2 ** self.t)
                    step_p = lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    loader = self.pm.tx_pattern_loader if pat_key == 'tx' else self.pm.rx_pattern_loader
                    if loader is not None:
                        loader.apply_update(-step_p)


# ============================================================================
# Section 5: Metric Functions
# ============================================================================

def normalize_for_lpips(ra_map: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize RA magnitude map to [0, 1] and add channel dims for LPIPS."""
    if ra_map.dim() == 4:
        ra_map = ra_map.squeeze(0).squeeze(0)
    elif ra_map.dim() == 3:
        ra_map = ra_map.squeeze(0)
    ra_min = ra_map.min()
    ra_max = ra_map.max()
    ra_norm = (ra_map - ra_min) / (ra_max - ra_min + eps)
    ra_norm = ra_norm.unsqueeze(0).unsqueeze(0)
    ra_norm = ra_norm.repeat(1, 3, 1, 1)
    return ra_norm


def compute_all_metrics(rendered_ra: torch.Tensor, gt_ra: torch.Tensor,
                        lpips_model=None, range_res: float = None,
                        ra_gt_cart: np.ndarray = None) -> Dict[str, float]:
    """Compute all RA metrics with SEPARATE min-max normalization (from TSSF).

    If range_res is provided (or ra_gt_cart is pre-computed), also computes
    Cartesian RA metrics: cart_corr, cart_psnr, cart_ssim, cart_mse, cart_rmse.
    """
    metrics = {}
    rendered_ra = rendered_ra.detach().float()
    gt_ra = gt_ra.detach().float()

    if rendered_ra.dim() > 2:
        rendered_ra = rendered_ra.squeeze()
    if gt_ra.dim() > 2:
        gt_ra = gt_ra.squeeze()

    rendered_ra_norm = minmax_normalize_torch(rendered_ra)
    gt_ra_norm = minmax_normalize_torch(gt_ra)

    metrics['ra_mse'] = torch.mean((rendered_ra_norm - gt_ra_norm)**2).item()
    metrics['ra_rmse'] = math.sqrt(metrics['ra_mse'])
    metrics['ra_ssim'] = compute_ssim(rendered_ra_norm, gt_ra_norm)
    metrics['ra_psnr'] = compute_psnr(rendered_ra_norm, gt_ra_norm)

    if lpips_model is not None and LPIPS_AVAILABLE:
        try:
            ra_norm_pred = normalize_for_lpips(rendered_ra_norm)
            ra_norm_gt = normalize_for_lpips(gt_ra_norm)
            device = next(lpips_model.parameters()).device
            ra_norm_pred = ra_norm_pred.to(device)
            ra_norm_gt = ra_norm_gt.to(device)
            with torch.no_grad():
                lpips_val = lpips_model(ra_norm_pred, ra_norm_gt).mean().item()
            metrics['ra_lpips'] = lpips_val
        except Exception:
            metrics['ra_lpips'] = float('nan')
    else:
        metrics['ra_lpips'] = float('nan')

    # Cartesian RA metrics
    if range_res is not None or ra_gt_cart is not None:
        from mmir.data.ra_utils import ra_polar_to_cartesian, compute_cartesian_ra_metrics
        rendered_ra_np = rendered_ra.detach().cpu().numpy()
        ra_rend_cart = ra_polar_to_cartesian(rendered_ra_np, range_res)
        if ra_gt_cart is None:
            gt_ra_np = gt_ra.detach().cpu().numpy()
            ra_gt_cart = ra_polar_to_cartesian(gt_ra_np, range_res)
        cart_metrics = compute_cartesian_ra_metrics(ra_rend_cart, ra_gt_cart)
        metrics.update(cart_metrics)
        metrics['_ra_rend_cart'] = ra_rend_cart  # cached for reuse

    return metrics


def evaluate_against_targets(metrics: Dict[str, float]) -> Dict:
    """Check if metrics meet publication targets (from TSSF)."""
    targets = {
        'ra_mse': {'threshold': 1e-4, 'op': 'less', 'target_str': '< 1e-4'},
        'ra_rmse': {'threshold': 0.2, 'op': 'less', 'target_str': '< 0.2'},
        'ra_psnr': {'threshold': 25.0, 'op': 'greater', 'target_str': '> 25 dB'},
        'ra_ssim': {'threshold': 0.4, 'op': 'greater', 'target_str': '> 0.4'},
        'ra_lpips': {'threshold': 0.4, 'op': 'less', 'target_str': '< 0.4'},
    }

    evaluation = {}
    passing_count = 0

    for metric_name, target_info in targets.items():
        value = metrics.get(metric_name, float('nan'))
        threshold = target_info['threshold']

        if not math.isfinite(value):
            evaluation[metric_name] = {
                'value': value, 'target': target_info['target_str'],
                'threshold': threshold, 'passes': False, 'status': '?'
            }
            continue

        passes = (value < threshold) if target_info['op'] == 'less' else (value > threshold)
        evaluation[metric_name] = {
            'value': value, 'target': target_info['target_str'],
            'threshold': threshold, 'passes': passes,
            'status': '[OK]' if passes else '[WARNING]'
        }
        if passes:
            passing_count += 1

    evaluation['passing_count'] = passing_count
    evaluation['all_pass'] = (passing_count == 5)
    return evaluation


def compute_correlation_metrics(ra_rendered: np.ndarray, ra_gt: np.ndarray,
                                range_res: float = 0.0) -> Tuple[float, float, float]:
    """Compute polar correlation, MSE, and Cartesian correlation."""
    ra_r_norm = minmax_normalize_numpy(ra_rendered)
    ra_g_norm = minmax_normalize_numpy(ra_gt)

    polar_corr = np.corrcoef(ra_r_norm.flatten(), ra_g_norm.flatten())[0, 1]
    if np.isnan(polar_corr):
        polar_corr = 0.0

    mse = np.mean((ra_r_norm - ra_g_norm) ** 2)

    cart_corr = 0.0
    if range_res > 0:
        ra_r_cart = ra_polar_to_cartesian(ra_rendered, range_res)
        ra_g_cart = ra_polar_to_cartesian(ra_gt, range_res)
        ra_r_cart_norm = minmax_normalize_numpy(ra_r_cart)
        ra_g_cart_norm = minmax_normalize_numpy(ra_g_cart)
        cart_corr = np.corrcoef(ra_r_cart_norm.flatten(), ra_g_cart_norm.flatten())[0, 1]
        if np.isnan(cart_corr):
            cart_corr = 0.0

    return polar_corr, mse, cart_corr


def compute_adc_metrics(
    rendered_adc_complex: np.ndarray,
    gt_adc_complex: np.ndarray,
    adc_use_log: bool = False,
    log_epsilon: float = 1e-6,
    log_scale: float = 1.0,
    top_percentile: float = 0.9,
    phase_metrics_type: str = "unit_phasor"
) -> Dict[str, float]:
    """Compute reconstruction metrics on complex-valued ADC data (from TSSF).

    Computes metrics with BOTH log and min-max normalization:
    1. Log-normalized ADC magnitude: L1, L2, MSE, RMSE, MAE, Pearson
    2. Min-max normalized ADC magnitude: L1, L2, MSE, RMSE, MAE, Pearson
    3. ADC Phase: unit phasor 2(1-cos(Δφ)) or relative phase metrics
    """
    metrics = {}
    eps = 1e-10

    rendered_mag = np.abs(rendered_adc_complex)
    gt_mag = np.abs(gt_adc_complex)

    # Log normalization
    rendered_mag_log = np.log(log_epsilon + log_scale * rendered_mag)
    gt_mag_log = np.log(log_epsilon + log_scale * gt_mag)
    # Min-max normalization
    rendered_mag_minmax = minmax_normalize_numpy(rendered_mag)
    gt_mag_minmax = minmax_normalize_numpy(gt_mag)

    # === Log-normalized ADC magnitude metrics ===
    mag_diff_log = rendered_mag_log - gt_mag_log
    metrics['adc_log_mag_l1'] = float(np.mean(np.abs(mag_diff_log)))
    metrics['adc_log_mag_l2'] = float(np.sum(mag_diff_log ** 2))
    metrics['adc_log_mag_mse'] = float(np.mean(mag_diff_log ** 2))
    metrics['adc_log_mag_rmse'] = float(np.sqrt(metrics['adc_log_mag_mse']))
    metrics['adc_log_mag_mae'] = metrics['adc_log_mag_l1']
    r_log_flat = rendered_mag_log.flatten()
    g_log_flat = gt_mag_log.flatten()
    if np.std(r_log_flat) < eps or np.std(g_log_flat) < eps:
        metrics['adc_log_mag_pearson'] = 0.0
    else:
        metrics['adc_log_mag_pearson'] = float(np.corrcoef(r_log_flat, g_log_flat)[0, 1])

    # === Min-max normalized ADC magnitude metrics ===
    mag_diff_mm = rendered_mag_minmax - gt_mag_minmax
    metrics['adc_minmax_mag_l1'] = float(np.mean(np.abs(mag_diff_mm)))
    metrics['adc_minmax_mag_l2'] = float(np.sum(mag_diff_mm ** 2))
    metrics['adc_minmax_mag_mse'] = float(np.mean(mag_diff_mm ** 2))
    metrics['adc_minmax_mag_rmse'] = float(np.sqrt(metrics['adc_minmax_mag_mse']))
    metrics['adc_minmax_mag_mae'] = metrics['adc_minmax_mag_l1']
    r_mm_flat = rendered_mag_minmax.flatten()
    g_mm_flat = gt_mag_minmax.flatten()
    if np.std(r_mm_flat) < eps or np.std(g_mm_flat) < eps:
        metrics['adc_minmax_mag_pearson'] = 0.0
    else:
        metrics['adc_minmax_mag_pearson'] = float(np.corrcoef(r_mm_flat, g_mm_flat)[0, 1])

    # === Legacy metrics (based on adc_use_log flag) ===
    mag_diff = mag_diff_log if adc_use_log else mag_diff_mm
    r_norm_flat = r_log_flat if adc_use_log else r_mm_flat
    g_norm_flat = g_log_flat if adc_use_log else g_mm_flat

    metrics['adc_l1'] = float(np.mean(np.abs(mag_diff)))
    metrics['adc_l2'] = float(np.sum(mag_diff ** 2))
    metrics['adc_mse'] = float(np.mean(mag_diff ** 2))
    metrics['adc_rmse'] = float(np.sqrt(metrics['adc_mse']))
    metrics['adc_mae'] = metrics['adc_l1']
    if np.std(r_norm_flat) < eps or np.std(g_norm_flat) < eps:
        metrics['adc_pearson'] = 0.0
    else:
        metrics['adc_pearson'] = float(np.corrcoef(r_norm_flat, g_norm_flat)[0, 1])
    for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
        metrics[f'adc_mag_{s}'] = metrics[f'adc_{s}']

    # === ADC Phase Metrics ===
    NT, NR, K = rendered_adc_complex.shape
    mag_threshold = np.percentile(gt_mag.flatten(), top_percentile * 100)
    valid_mask = gt_mag >= mag_threshold

    if phase_metrics_type == "unit_phasor":
        if np.sum(valid_mask) > 10:
            rendered_phase = np.angle(rendered_adc_complex)
            gt_phase = np.angle(gt_adc_complex)
            phase_diff = rendered_phase - gt_phase
            unit_phasor_loss = 2.0 * (1.0 - np.cos(phase_diff))
            metrics['adc_phase_l1'] = float(np.mean(np.abs(unit_phasor_loss[valid_mask])))
            metrics['adc_phase_mse'] = float(np.mean(unit_phasor_loss[valid_mask]))
            metrics['adc_phase_rmse'] = float(np.sqrt(metrics['adc_phase_mse']))
            metrics['adc_phase_mae'] = float(np.mean(unit_phasor_loss[valid_mask]))
            metrics['adc_phase_l2'] = float(np.sum(unit_phasor_loss[valid_mask]))
            cos_r = np.cos(rendered_phase[valid_mask]).flatten()
            cos_g = np.cos(gt_phase[valid_mask]).flatten()
            if np.std(cos_r) < eps or np.std(cos_g) < eps:
                metrics['adc_phase_pearson'] = 0.0
            else:
                metrics['adc_phase_pearson'] = float(np.corrcoef(cos_r, cos_g)[0, 1])
        else:
            for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
                metrics[f'adc_phase_{s}'] = float('nan')
    else:  # relative_phase
        if NR > 1 and np.sum(valid_mask) > 10:
            r_ref = rendered_adc_complex[:, 0:1, :]
            g_ref = gt_adc_complex[:, 0:1, :]
            r_rel = np.angle(rendered_adc_complex * np.conj(r_ref))
            g_rel = np.angle(gt_adc_complex * np.conj(g_ref))
            pd = r_rel - g_rel
            pd = np.arctan2(np.sin(pd), np.cos(pd))
            pd_valid = pd[valid_mask] / np.pi
            metrics['adc_phase_l1'] = float(np.mean(np.abs(pd_valid)))
            metrics['adc_phase_l2'] = float(np.sum(pd_valid ** 2))
            metrics['adc_phase_mse'] = float(np.mean(pd_valid ** 2))
            metrics['adc_phase_rmse'] = float(np.sqrt(metrics['adc_phase_mse']))
            metrics['adc_phase_mae'] = metrics['adc_phase_l1']
            r_rel_v = r_rel[valid_mask].flatten()
            g_rel_v = g_rel[valid_mask].flatten()
            if np.std(r_rel_v) < eps or np.std(g_rel_v) < eps:
                metrics['adc_phase_pearson'] = 0.0
            else:
                metrics['adc_phase_pearson'] = float(np.corrcoef(r_rel_v, g_rel_v)[0, 1])
        else:
            for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
                metrics[f'adc_phase_{s}'] = float('nan') if NR <= 1 else 0.0

    # Phase aliases for dashboard compatibility
    for pfx in ['adc_relphase', 'adc_log_phase', 'adc_minmax_phase']:
        for s in ['l1', 'l2', 'mse', 'rmse', 'mae', 'pearson']:
            metrics[f'{pfx}_{s}'] = metrics[f'adc_phase_{s}']

    # Virtual array phase coherence
    metrics['va_phase_coherence'] = _compute_va_phase_coherence(
        rendered_adc_complex, gt_adc_complex, top_percentile)

    return metrics


def _compute_va_phase_coherence(rendered_complex, gt_complex, top_percentile=0.9):
    """Virtual Array Phase Coherence metric.

    For each range bin, measures how well the relative phase pattern across
    the virtual array (TX x RX) matches between rendered and GT ADC.

    Complex correlation: rho[k] = |v_hat_r . conj(v_hat_g)| / N_vx
    where v_hat = v / |v| (unit-magnitude normalization per element).
    Global phase offsets cancel via the magnitude of the dot product.

    Averaged over top-percentile range bins by GT energy.

    Returns value in [0, 1] where 1 = perfect relative phase match,
    ~1/N_vx = random phase.
    """
    NT, NR, K = rendered_complex.shape
    N_vx = NT * NR
    eps = 1e-10

    r = rendered_complex.reshape(N_vx, K)
    g = gt_complex.reshape(N_vx, K)

    gt_energy = np.sum(np.abs(g) ** 2, axis=0)
    thresh = np.percentile(gt_energy, top_percentile * 100)
    valid = gt_energy >= thresh

    if np.sum(valid) < 2:
        return float('nan')

    r_hat = r / (np.abs(r) + eps)
    g_hat = g / (np.abs(g) + eps)

    dot = np.sum(r_hat[:, valid] * np.conj(g_hat[:, valid]), axis=0)
    rho = np.abs(dot) / N_vx

    return float(np.mean(rho))


def _compute_single_chirp_metrics(
    rendered_adc_complex: np.ndarray, gt_adc_complex: np.ndarray,
    top_percentile: float = 0.9, adc_use_log: bool = False,
    ra_use_log: bool = False, log_epsilon: float = 1e-6,
    log_scale: float = 1.0, phase_metrics_type: str = "unit_phasor",
    eps: float = 1e-10
) -> Dict[str, float]:
    """Compute metrics for a single chirp (helper for multi-chirp averaging)."""
    metrics = compute_adc_metrics(
        rendered_adc_complex, gt_adc_complex,
        adc_use_log=adc_use_log, log_epsilon=log_epsilon, log_scale=log_scale,
        top_percentile=top_percentile, phase_metrics_type=phase_metrics_type)

    # RA metrics via FFT
    r_ra = np.fft.fft(rendered_adc_complex, axis=-1)
    g_ra = np.fft.fft(gt_adc_complex, axis=-1)
    if ra_use_log:
        r_ra_n = np.log(log_epsilon + log_scale * np.abs(r_ra))
        g_ra_n = np.log(log_epsilon + log_scale * np.abs(g_ra))
    else:
        r_ra_n = minmax_normalize_numpy(np.abs(r_ra))
        g_ra_n = minmax_normalize_numpy(np.abs(g_ra))
    ra_diff = r_ra_n - g_ra_n
    metrics['ra_mse'] = float(np.mean(ra_diff ** 2))
    metrics['ra_rmse'] = float(np.sqrt(metrics['ra_mse']))
    metrics['ra_psnr'] = float(10 * np.log10(1.0 / metrics['ra_mse'])) if metrics['ra_mse'] > 0 else float('inf')
    rf, gf = r_ra_n.flatten(), g_ra_n.flatten()
    if np.std(rf) < eps or np.std(gf) < eps:
        metrics['ra_ssim'] = 0.0
    else:
        corr = np.corrcoef(rf, gf)[0, 1]
        mx, my = np.mean(rf), np.mean(gf)
        vx, vy = np.var(rf), np.var(gf)
        c1, c2 = 0.01**2, 0.03**2
        metrics['ra_ssim'] = float(
            (2*mx*my+c1)/(mx**2+my**2+c1) * (2*np.sqrt(vx*vy)+c2)/(vx+vy+c2) * (corr+1)/2)
    metrics['ra_lpips'] = float('nan')
    return metrics


def compute_adc_metrics_multichirp(
    rendered_adc_complex: np.ndarray, gt_all_chirps: np.ndarray,
    top_percentile: float = 0.9, adc_use_log: bool = False,
    ra_use_log: bool = False, log_epsilon: float = 1e-6,
    log_scale: float = 1.0, phase_metrics_type: str = "unit_phasor",
    eps: float = 1e-10
) -> Dict[str, float]:
    """Compute metrics averaged over all chirps with per-chirp alignment (from TSSF)."""
    if gt_all_chirps.ndim == 5:
        gt_cpx = gt_all_chirps[..., 0] + 1j * gt_all_chirps[..., 1]
    else:
        gt_cpx = gt_all_chirps
    n_chirp = gt_cpx.shape[0]
    gt_cpx = gt_cpx.transpose(0, 2, 1, 3)  # (chirp, rx, tx, adc) -> (chirp, tx, rx, adc)

    all_chirp_metrics = []
    for c in range(n_chirp):
        gt_c = gt_cpx[c]
        gt_mag = np.abs(gt_c)
        w = (gt_mag / np.maximum(gt_mag.max(), eps)).clip(0, 1)
        alpha = np.sum(w * gt_c * np.conj(rendered_adc_complex)) / \
                np.maximum(np.sum(w * np.abs(rendered_adc_complex)**2), eps)
        chirp_m = _compute_single_chirp_metrics(
            alpha * rendered_adc_complex, gt_c, top_percentile,
            adc_use_log=adc_use_log, ra_use_log=ra_use_log,
            log_epsilon=log_epsilon, log_scale=log_scale,
            phase_metrics_type=phase_metrics_type, eps=eps)
        all_chirp_metrics.append(chirp_m)

    metrics = {}
    for key in all_chirp_metrics[0].keys():
        vals = [m[key] for m in all_chirp_metrics if not np.isnan(m[key])]
        metrics[key] = float(np.mean(vals)) if vals else float('nan')

    # Unaligned RA metrics
    for c in range(n_chirp):
        gt_c = gt_cpx[c]
        r_ra_n = minmax_normalize_numpy(np.abs(np.fft.fft(rendered_adc_complex, axis=-1)))
        g_ra_n = minmax_normalize_numpy(np.abs(np.fft.fft(gt_c, axis=-1)))
        ra_diff = r_ra_n - g_ra_n
        mse = float(np.mean(ra_diff ** 2))
        rf, gf = r_ra_n.flatten(), g_ra_n.flatten()
        if np.std(rf) < eps or np.std(gf) < eps:
            ssim = 0.0
        else:
            corr = np.corrcoef(rf, gf)[0, 1]
            mx, my = np.mean(rf), np.mean(gf)
            vx, vy = np.var(rf), np.var(gf)
            c1, c2 = 0.01**2, 0.03**2
            ssim = float((2*mx*my+c1)/(mx**2+my**2+c1) * (2*np.sqrt(vx*vy)+c2)/(vx+vy+c2) * (corr+1)/2)
        if c == 0:
            ua = {'mse': [mse], 'rmse': [np.sqrt(mse)],
                  'psnr': [10*np.log10(1/mse) if mse > 0 else float('inf')], 'ssim': [ssim]}
        else:
            ua['mse'].append(mse); ua['rmse'].append(np.sqrt(mse))
            ua['psnr'].append(10*np.log10(1/mse) if mse > 0 else float('inf')); ua['ssim'].append(ssim)

    for k in ['mse', 'rmse', 'psnr', 'ssim']:
        vals = [v for v in ua[k] if not np.isnan(v) and not np.isinf(v)]
        metrics[f'ra_unaligned_{k}'] = float(np.mean(vals)) if vals else float('nan')
    metrics['ra_unaligned_lpips'] = float('nan')
    return metrics


def compute_ra_metrics(
    rendered_adc_complex: np.ndarray, gt_adc_complex: np.ndarray,
    ra_use_log: bool = False, log_epsilon: float = 1e-6,
    log_scale: float = 1.0, eps: float = 1e-10
) -> Dict[str, float]:
    """Compute RA-domain metrics for single chirp (from TSSF)."""
    r_ra = np.fft.fft(rendered_adc_complex, axis=-1)
    g_ra = np.fft.fft(gt_adc_complex, axis=-1)
    if ra_use_log:
        r_n = np.log(log_epsilon + log_scale * np.abs(r_ra))
        g_n = np.log(log_epsilon + log_scale * np.abs(g_ra))
    else:
        r_n = minmax_normalize_numpy(np.abs(r_ra))
        g_n = minmax_normalize_numpy(np.abs(g_ra))
    d = r_n - g_n
    metrics = {}
    metrics['ra_mse'] = float(np.mean(d ** 2))
    metrics['ra_rmse'] = float(np.sqrt(metrics['ra_mse']))
    metrics['ra_psnr'] = float(10 * np.log10(1.0 / metrics['ra_mse'])) if metrics['ra_mse'] > 0 else float('inf')
    rf, gf = r_n.flatten(), g_n.flatten()
    if np.std(rf) < eps or np.std(gf) < eps:
        metrics['ra_ssim'] = 0.0
    else:
        corr = np.corrcoef(rf, gf)[0, 1]
        mx, my = np.mean(rf), np.mean(gf)
        vx, vy = np.var(rf), np.var(gf)
        c1, c2 = 0.01**2, 0.03**2
        metrics['ra_ssim'] = float(
            (2*mx*my+c1)/(mx**2+my**2+c1) * (2*np.sqrt(vx*vy)+c2)/(vx+vy+c2) * (corr+1)/2)
    metrics['ra_lpips'] = float('nan')
    return metrics


# ============================================================================
# Section 6: Initialization Functions
# ============================================================================

def create_render_config(config: TrainingConfigSionna) -> RenderConfigRef:
    """Create renderer_final configuration from training config."""
    return RenderConfigRef(
        n_hits_per_rx=config.n_hits_per_rx,
        n_rays_per_res=config.n_rays_per_res,
        max_distance=1e10,
        seed=42,
        verbose=False,
        use_radar_equation=True,
        use_pattern_importance_sampling=config.use_pattern_importance_sampling,
        bsdf_model=config.bsdf_model,
        mmwave_polarization=config.mmwave_polarization,
        hemisphere_sampling=config.hemisphere_sampling,
        double_sided=config.double_sided,
        enable_image_method=config.enable_image_method,
        enable_sms=config.enable_sms,
        sms_max_iterations=config.sms_max_iterations,
        sms_solver_threshold=config.sms_solver_threshold,
        sms_use_smooth_normals=config.sms_use_smooth_normals,
        use_patch_clustering=config.use_patch_clustering,
        patch_angle_threshold_deg=config.patch_angle_threshold_deg,
        patch_distance_threshold=config.patch_distance_threshold,
        patch_max_tris=config.patch_max_tris,
        use_spatial_adjacency=config.use_spatial_adjacency,
        spatial_radius=config.spatial_radius,
        material_columns=config.material_columns,
        use_vertex_normals=(config.param_mode == 'per_vertex'),
        diffraction_config=DiffractionConfig(use_gpu_apertures=True) if config.enable_diffraction else None,
        # End-to-end differentiable & multibounce
        use_end_to_end_ad=config.use_end_to_end_ad,
        max_bounces=config.max_bounces,
        nee_every_bounce=config.nee_every_bounce,
        rr_start_bounce=config.rr_start_bounce,
        rr_prob=config.rr_prob,
        e2e_ray_seed_rotation_interval=config.e2e_ray_seed_rotation_interval,
        e2e_phase_chunk_size=config.e2e_phase_chunk_size,
        enable_grad_phase=config.enable_grad_phase,
        enable_boundary_gradients=config.enable_boundary_gradients,
        sms_cache_interval=config.sms_cache_interval,
    )


def create_renderer(config: TrainingConfigSionna) -> FMCWRendererRef:
    """Initialize the Sionna renderer from config."""
    render_config = create_render_config(config)
    renderer = FMCWRendererRef.from_files(
        mesh_file=config.scene_file,
        config_file=config.config_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
        material_type="metal",
        render_config=render_config,
        verbose=False,
    )
    return renderer


class BinTriangleMapper:
    """Maps RA bins to contributing mesh triangles for per-triangle error-map initialization.

    Uses coordinate transform (radar frame) + searchsorted binning to map
    each triangle centroid to an (az, range) bin in the RA map.
    """

    def __init__(self, mesh_vertices: np.ndarray, mesh_faces: np.ndarray,
                 radar_origin: np.ndarray, radar_boresight: np.ndarray,
                 num_az: int, num_r: int, range_res: float,
                 near_field_m: float = 1.5):
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces
        self.radar_origin = radar_origin.astype(np.float32)
        self.radar_boresight = radar_boresight.astype(np.float32)
        self.radar_boresight /= np.linalg.norm(self.radar_boresight)
        self.near_field_m = near_field_m
        self.num_az = num_az
        self.num_r = num_r
        self.range_res = range_res

        self._create_bin_edges()
        self.bin_to_triangles: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}

    def _create_bin_edges(self):
        """Create bin edges matching the radar RA grid."""
        try:
            from mmir.evaluation.utils.single_view_proc import make_angle_grids_np, _centers_to_edges
            fft_size_az = self.num_az + 1
            az_centers, _ = make_angle_grids_np(fft_size_az, 2)
            self.az_edges = _centers_to_edges(
                az_centers, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
        except ImportError:
            # Fallback: uniform azimuth bins
            self.az_edges = np.linspace(-np.pi/2, np.pi/2, self.num_az + 1).astype(np.float32)
        self.r_edges = np.arange(self.num_r + 1, dtype=np.float32) * self.range_res

    def _compute_rotation_matrix(self) -> np.ndarray:
        """Rotation matrix to align boresight with +Y."""
        boresight = self.radar_boresight
        target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dot = np.dot(boresight, target)
        if dot > 0.9999:
            return np.eye(3, dtype=np.float32)
        if dot < -0.9999:
            perp = np.array([1, 0, 0], dtype=np.float32) if abs(boresight[0]) < 0.9 \
                else np.array([0, 1, 0], dtype=np.float32)
            axis = np.cross(boresight, perp)
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]], dtype=np.float32)
            return np.eye(3, dtype=np.float32) + 2 * K @ K
        axis = np.cross(boresight, target)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def compute_mapping(self):
        """Compute triangle-to-RA-bin mapping using radar-frame coordinates."""
        print("[BinTriangleMapper] Computing triangle-to-RA-bin mapping...")
        t_start = time.time()

        # Triangle centroids and areas
        v0 = self.mesh_vertices[self.mesh_faces[:, 0]]
        v1 = self.mesh_vertices[self.mesh_faces[:, 1]]
        v2 = self.mesh_vertices[self.mesh_faces[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        # Transform to radar frame
        pts_local = centroids - self.radar_origin
        R = self._compute_rotation_matrix()
        pts_radar = (R @ pts_local.T).T

        # Spherical coordinates
        x, y, z = pts_radar[:, 0], pts_radar[:, 1], pts_radar[:, 2]
        r = np.linalg.norm(pts_radar, axis=1)
        valid_mask = r >= self.near_field_m
        az = np.arctan2(-x, y)

        # Bin indices
        az_idx = np.searchsorted(self.az_edges, az, side='right') - 1
        r_idx = np.searchsorted(self.r_edges, r, side='right') - 1
        valid_bins = valid_mask & (az_idx >= 0) & (az_idx < self.num_az) & \
                     (r_idx >= 0) & (r_idx < self.num_r)

        # Build mapping
        self.bin_to_triangles = {}
        valid_indices = np.where(valid_bins)[0]
        for idx in valid_indices:
            key = (int(az_idx[idx]), int(r_idx[idx]))
            if key not in self.bin_to_triangles:
                self.bin_to_triangles[key] = []
            self.bin_to_triangles[key].append((int(idx), float(areas[idx])))

        # Normalize weights
        for key in self.bin_to_triangles:
            entries = self.bin_to_triangles[key]
            total = sum(w for _, w in entries)
            if total > 0:
                self.bin_to_triangles[key] = [(t, w / total) for t, w in entries]

        print(f"  Mapped {len(valid_indices)}/{len(self.mesh_faces)} triangles "
              f"to {len(self.bin_to_triangles)} RA bins ({time.time()-t_start:.2f}s)")
        return self.bin_to_triangles


def run_error_map_initialization(pm: ParameterManagerSionna,
                                  ra_gt: np.ndarray, range_res: float):
    """Per-triangle error-map initialization using BinTriangleMapper.

    Maps RA bins to mesh triangles, then adjusts per-triangle raw_params
    based on per-triangle error sign and magnitude:
    - Over-reflecting (error > 0.05): increase sigma_h, decrease eps_real
    - Under-reflecting (error < -0.05): boost eps_real, decrease sigma_h, increase tau
    """
    print("\n--- Error-map initialization ---")
    pm.set_renderer_materials()
    result = pm.renderer.render(seed=42)
    adc_np = pm.renderer.get_adc_numpy()

    # Convert to RA
    adc_torch = torch.from_numpy(adc_np).float()
    ra_rendered = adc_to_ra_image(adc_torch).detach().cpu().numpy()

    # Compute normalized error map (rendered - GT)
    error_map = minmax_normalize_numpy(ra_rendered) - minmax_normalize_numpy(ra_gt)

    # Try per-triangle mapping via BinTriangleMapper
    try:
        radar_origin = np.array(pm.renderer.scene_ctx.tx_array.positions[0], dtype=np.float32)
        radar_boresight = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        num_az, num_r = ra_gt.shape

        mapper = BinTriangleMapper(
            mesh_vertices=pm.mesh_vertices,
            mesh_faces=pm.mesh_faces,
            radar_origin=radar_origin,
            radar_boresight=radar_boresight,
            num_az=num_az, num_r=num_r,
            range_res=range_res,
        )
        mapper.compute_mapping()

        # Accumulate per-triangle error
        n_tri = len(pm.mesh_faces)
        tri_error = np.zeros(n_tri, dtype=np.float32)
        tri_count = np.zeros(n_tri, dtype=np.float32)

        for (az_bin, r_bin), tri_weights in mapper.bin_to_triangles.items():
            if az_bin >= num_az or r_bin >= num_r:
                continue
            err = error_map[az_bin, r_bin]
            for tri_id, weight in tri_weights:
                tri_error[tri_id] += err * weight
                tri_count[tri_id] += weight

        mask = tri_count > 0
        tri_error[mask] /= tri_count[mask]

        # Map to per-optimization-unit (per_vertex or per_triangle)
        if pm.config.param_mode == 'per_vertex':
            # Average triangle errors to vertices
            opt_error = np.zeros(pm.n_opt_params, dtype=np.float32)
            opt_count = np.zeros(pm.n_opt_params, dtype=np.float32)
            for ti in range(n_tri):
                for vi in pm.mesh_faces[ti]:
                    opt_error[vi] += tri_error[ti]
                    opt_count[vi] += 1
            msk = opt_count > 0
            opt_error[msk] /= opt_count[msk]
        else:
            opt_error = tri_error[:pm.n_opt_params]

        # Per-triangle adjustments
        over_mask = opt_error > 0.05
        under_mask = opt_error < -0.05

        n_over = int(np.sum(over_mask))
        n_under = int(np.sum(under_mask))
        n_ok = pm.n_opt_params - n_over - n_under

        if np.any(over_mask):
            pm.raw_params[over_mask, 2] += 0.5    # sigma_h: rougher
            pm.raw_params[over_mask, 0] -= 0.5    # eps_real: lower

        if np.any(under_mask):
            pm.raw_params[under_mask, 0] += 1.0   # eps_real: higher
            pm.raw_params[under_mask, 2] -= 1.0   # sigma_h: smoother
            pm.raw_params[under_mask, 4] += 0.5   # tau: more KA

        print(f"  Per-triangle error-init: {n_over} over, {n_under} under, {n_ok} OK")

    except Exception as e:
        print(f"  Warning: BinTriangleMapper failed ({e}), falling back to uniform adjustment")
        # Fallback: uniform adjustment
        over_frac = (error_map > 0.05).mean()
        under_frac = (error_map < -0.05).mean()
        if over_frac > under_frac:
            pm.raw_params[:, 2] += 0.3
            pm.raw_params[:, 0] -= 0.3
        elif under_frac > over_frac:
            pm.raw_params[:, 0] += 0.5
            pm.raw_params[:, 2] -= 0.5
            pm.raw_params[:, 4] += 0.3

    phys = reparameterize_physics_params(pm.raw_params)
    print(f"  Post-adjustment: eps'={phys[:, 0].mean():.2f}, tau={phys[:, 4].mean():.3f}")
    print("--- Error-map initialization done ---\n")


# ============================================================================
# Section 7: Forward Pass & Backward Gradient Flow
# ============================================================================

def ensure_cached_geometry(pm: 'ParameterManagerSionna', iteration: int,
                           base_seed: int) -> 'Optional[CachedGeometry]':
    """Build or reuse persistent geometry cache (Phase A) for two-phase rendering."""
    cfg = pm.config
    cache = getattr(pm, '_persistent_cache', None)
    cache_iter = getattr(pm, '_cache_iteration', -1)
    cache_refresh = getattr(cfg, 'cache_refresh_interval', 0)

    need_refresh = (
        cache is None
        or (cache_refresh > 0 and (iteration - cache_iter) >= cache_refresh)
        or (cfg.LEARN_POSE and 'pose_rotation' not in cfg.frozen_params)
        or (cfg.LEARN_VTX and 'vertex_positions' not in cfg.frozen_params)
    )

    if need_refresh:
        pm.set_renderer_materials()
        _, cached_geom = pm.renderer.render_with_cached_geometry(seed=base_seed)
        pm._persistent_cache = cached_geom
        pm._cache_iteration = iteration
    return pm._persistent_cache


def run_two_phase_forward(pm: 'ParameterManagerSionna',
                          cached_geom: 'CachedGeometry'):
    """Phase B: Differentiable re-synthesis with grad-enabled params.

    Uses cached geometry from Phase A. Only BSDF weights carry material gradients.
    This is the same approach used by the Sionna reference renderer.

    Returns (adc_real, adc_imag, raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr).
    """
    raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr = \
        pm.create_grad_enabled_params()

    adc_real, adc_imag = pm.renderer.render_differentiable(
        cached_geom, raw_drjit,
        pose_params=pose_dr,
        normal_params=normal_dr,
        pattern_loaders=pattern_loaders_dr,
        vertex_offset_params=vertex_offset_dr,
    )

    return adc_real, adc_imag, raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr


def run_end_to_end_forward(pm: 'ParameterManagerSionna', iteration: int):
    """End-to-end differentiable forward pass (single AD graph).

    Replaces the two-phase (cached geometry + differentiable re-synthesis) approach.
    Everything — ray tracing, BSDF evaluation, phase computation, ADC accumulation —
    is in one DrJit AD graph with gradient flow through materials, pose, normals, etc.

    Seed is rotated every e2e_ray_seed_rotation_interval iterations for MC variance reduction.

    Returns (adc_real, adc_imag, raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr).
    """
    cfg = pm.config

    # Create fresh grad-enabled parameters
    raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr = \
        pm.create_grad_enabled_params()

    # Set renderer materials for non-diff auxiliary operations (SMS solver setup, etc.)
    pm.set_renderer_materials()

    # Rotate seed for MC variance reduction
    seed_interval = max(cfg.e2e_ray_seed_rotation_interval, 1)
    seed = 42 + (iteration // seed_interval) * 7

    # Enable forward timing debug for first few iterations
    pm.renderer._debug_forward_timing = True

    # End-to-end render (single-bounce or multibounce depending on config.max_bounces)
    adc_real, adc_imag = pm.renderer.render_end_to_end(
        raw_drjit,
        pose_params=pose_dr,
        normal_params=normal_dr,
        vertex_offset_params=vertex_offset_dr,
        pattern_loaders=pattern_loaders_dr,
        seed=seed,
    )

    return adc_real, adc_imag, raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr


def extract_adc_for_metrics(adc_real: mi.Float, adc_imag: mi.Float,
                             NT: int, NR: int, K: int) -> np.ndarray:
    """Convert flat DrJit ADC to complex numpy (NR, NT, K) for metrics."""
    with dr.suspend_grad():
        real_np = np.array(adc_real).reshape(NT, NR, K)
        imag_np = np.array(adc_imag).reshape(NT, NR, K)
    return (real_np + 1j * imag_np).transpose(1, 0, 2)  # (NR, NT, K)


def extract_ra_from_adc(adc_real: mi.Float, adc_imag: mi.Float,
                         NT: int, NR: int, K: int) -> np.ndarray:
    """Extract RA magnitude from DrJit ADC (suspend grad)."""
    with dr.suspend_grad():
        real_np = np.array(adc_real).reshape(NT, NR, K)
        imag_np = np.array(adc_imag).reshape(NT, NR, K)
    adc_ri = np.stack([real_np, imag_np], axis=-1)
    adc_torch = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_torch).detach().cpu().numpy()
    return ra_polar


def clip_gradients(gradients: Dict[str, np.ndarray],
                   config: TrainingConfigSionna) -> Dict[str, np.ndarray]:
    """Per-group RMS gradient clipping (T2)."""
    clip_map = {
        'materials': config.clip_materials,
        'pose_rotation': config.clip_pose_rotation,
        'pose_translation': config.clip_pose_translation,
        'normals': config.clip_normals,
        'vertex_positions': config.clip_vertex_positions,
        'patterns': config.clip_patterns,
    }
    for name, grad in gradients.items():
        max_norm = clip_map.get(name, 1.0)
        if max_norm <= 0:
            continue
        if isinstance(grad, np.ndarray):
            rms = np.sqrt(np.mean(grad ** 2))
            if rms > max_norm:
                gradients[name] = grad * (max_norm / rms)
    return gradients


def run_backward_and_extract_gradients(
    adc_real: mi.Float,
    adc_imag: mi.Float,
    gt_adc_torch: torch.Tensor,
    loss_config: LossConfigSionna,
    raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr,
    config: TrainingConfigSionna,
    NT: int, NR: int, K: int,
    gt_all_chirps: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run backward pass and extract sanitized gradients.

    Returns (gradients_dict, loss_dict).
    """
    loss_dict = {}

    # ---- Compute RA loss via PyTorch bridge ----
    ra_grad_real = None
    ra_grad_imag = None
    multi_chirp_grad_real = None
    multi_chirp_grad_imag = None

    if loss_config.use_multi_chirp_loss and gt_all_chirps is not None:
        # Multi-chirp loss path
        with dr.suspend_grad():
            real_np = np.array(adc_real).reshape(NT, NR, K)
            imag_np = np.array(adc_imag).reshape(NT, NR, K)

        # Create PyTorch tensors with requires_grad
        rendered_real_torch = torch.tensor(real_np, dtype=torch.float32, requires_grad=True)
        rendered_imag_torch = torch.tensor(imag_np, dtype=torch.float32, requires_grad=True)

        # Stack and transpose to match GT format: [RX, TX, ADC, 2]
        pred_stacked = torch.stack([rendered_real_torch, rendered_imag_torch], dim=-1)  # [NT, NR, K, 2]
        pred_transposed = pred_stacked.permute(1, 0, 2, 3)  # [NR, NT, K, 2]
        pred_mc = pred_transposed.unsqueeze(0)  # [1, NR, NT, K, 2]

        mc_loss, mc_loss_dict = compute_multi_chirp_loss(
            pred_mc, gt_all_chirps,
            w_adc_mag=loss_config.adc_mag_weight,
            w_ra_mag=loss_config.ra_mag_weight,
            w_phase=loss_config.multi_chirp_phase_weight,
            adc_use_log=loss_config.adc_mag_use_log,
            ra_use_log=loss_config.ra_mag_use_log,
            adc_loss_type=loss_config.adc_mag_loss_type,
            ra_loss_type=loss_config.ra_mag_loss_type,
            log_epsilon=loss_config.log_epsilon,
            log_scale=loss_config.log_scale,
            top_magnitude_percentile=loss_config.multi_chirp_top_percentile,
        )

        mc_loss.backward()

        # Extract gradients
        grad_real_np = rendered_real_torch.grad.numpy().flatten()
        grad_imag_np = rendered_imag_torch.grad.numpy().flatten()
        multi_chirp_grad_real = mi.Float(grad_real_np)
        multi_chirp_grad_imag = mi.Float(grad_imag_np)

        loss_dict['total'] = mc_loss.item()
        for k, v in mc_loss_dict.items():
            loss_dict[f'mc_{k}'] = v

    elif loss_config.ra_mag_weight > 0:
        # Single-chirp RA loss via PyTorch bridge
        _, ra_grad_real_dr, ra_grad_imag_dr, ra_loss_dict = compute_ra_loss_with_gradients(
            adc_real, adc_imag, gt_adc_torch, NT, NR, K,
            normalize=True,
            weight=loss_config.ra_mag_weight,
            ssim_weight=loss_config.ra_mag_ssim_weight,
            lpips_weight=loss_config.ra_mag_perceptual_weight,
            ra_use_log=loss_config.ra_mag_use_log,
            ra_loss_type=loss_config.ra_mag_loss_type,
            log_epsilon=loss_config.log_epsilon,
            log_scale=loss_config.log_scale,
            adc_phase_weight=loss_config.adc_phase_weight,
            adc_phase_loss_type=loss_config.adc_phase_loss_type,
            phase_magnitude_weighting=loss_config.phase_magnitude_weighting,
            phase_magnitude_power=loss_config.phase_magnitude_power,
            phase_reference_channel=loss_config.phase_reference_channel,
            phase_use_gt_magnitude=loss_config.phase_use_gt_magnitude,
            phase_magnitude_clip_max=loss_config.phase_magnitude_clip_max,
            use_joint_normalization=loss_config.use_joint_normalization,
        )
        ra_grad_real = ra_grad_real_dr
        ra_grad_imag = ra_grad_imag_dr
        if ra_loss_dict:
            loss_dict.update(ra_loss_dict)
            loss_dict['total'] = ra_loss_dict.get('ra_l2', 0.0)

    # ---- Compute ADC magnitude loss (DrJit, single-chirp only) ----
    adc_loss = None
    if loss_config.adc_mag_weight > 0 and not (loss_config.use_multi_chirp_loss and gt_all_chirps is not None):
        gt_real_flat = mi.Float(gt_adc_torch[..., 0].numpy().flatten().astype(np.float32))
        gt_imag_flat = mi.Float(gt_adc_torch[..., 1].numpy().flatten().astype(np.float32))

        # Log-magnitude MSE loss (differentiable)
        log_eps = mi.Float(loss_config.log_epsilon)
        log_scale = mi.Float(loss_config.log_scale)
        adc_mag_r = dr.sqrt(adc_real * adc_real + adc_imag * adc_imag + mi.Float(1e-20))
        adc_mag_gt = dr.sqrt(gt_real_flat * gt_real_flat + gt_imag_flat * gt_imag_flat + mi.Float(1e-20))

        adc_mag_r_log = dr.log(adc_mag_r * log_scale + log_eps)
        adc_mag_gt_log = dr.log(adc_mag_gt * log_scale + log_eps)
        diff_adc = adc_mag_r_log - adc_mag_gt_log
        adc_loss = dr.mean(diff_adc * diff_adc) * loss_config.adc_mag_weight

        with dr.suspend_grad():
            adc_loss_val = float(adc_loss[0])
        loss_dict['adc_mag_loss'] = adc_loss_val
        if 'total' not in loss_dict:
            loss_dict['total'] = adc_loss_val
        else:
            loss_dict['total'] += adc_loss_val

    # ---- Determine backward mode ----
    use_adc = adc_loss is not None
    effective_ra_grad_real = multi_chirp_grad_real if multi_chirp_grad_real is not None else ra_grad_real
    effective_ra_grad_imag = multi_chirp_grad_imag if multi_chirp_grad_imag is not None else ra_grad_imag
    use_ra = effective_ra_grad_real is not None

    # ---- Backward pass ----
    if use_ra and not use_adc:
        # RA-ONLY MODE: Inject gradients and traverse
        dr.set_grad(adc_real, effective_ra_grad_real)
        dr.set_grad(adc_imag, effective_ra_grad_imag)
        dr.enqueue(dr.ADMode.Backward, adc_real)
        dr.enqueue(dr.ADMode.Backward, adc_imag)
        dr.traverse(dr.ADMode.Backward)

    elif use_ra and use_adc:
        # COMBINED ADC+RA MODE: ADC backward first, then accumulate RA grads
        dr.backward(adc_loss, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

        # Accumulate RA gradients with ADC gradients on ADC outputs
        existing_grad_real = dr.grad(adc_real)
        existing_grad_imag = dr.grad(adc_imag)
        accumulated_real = (existing_grad_real + effective_ra_grad_real) if existing_grad_real is not None else effective_ra_grad_real
        accumulated_imag = (existing_grad_imag + effective_ra_grad_imag) if existing_grad_imag is not None else effective_ra_grad_imag

        dr.set_grad(adc_real, accumulated_real)
        dr.set_grad(adc_imag, accumulated_imag)
        dr.enqueue(dr.ADMode.Backward, adc_real)
        dr.enqueue(dr.ADMode.Backward, adc_imag)
        dr.traverse(dr.ADMode.Backward)

    elif use_adc:
        # ADC-ONLY MODE: Standard backward
        dr.backward(adc_loss)

    # ---- Extract & sanitize gradients ----
    gradients = {}

    # Materials (6 physics params)
    if 'materials' not in config.frozen_params:
        mat_grads = np.zeros((config.n_opt_params_hint, 6), dtype=np.float32) \
            if hasattr(config, 'n_opt_params_hint') else None
        mat_grads_list = []
        for i in range(6):
            g = np.array(dr.grad(raw_drjit[i]))
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            mat_grads_list.append(g)
        gradients['materials'] = np.stack(mat_grads_list, axis=-1)

    # Pose (6 scalars)
    if pose_dr is not None and 'pose_rotation' not in config.frozen_params:
        pose_grads = np.zeros(6, dtype=np.float32)
        for idx, key in enumerate(['pitch', 'roll', 'yaw', 'tx', 'ty', 'tz']):
            g_val = float(np.array(dr.grad(pose_dr[key]))[0])
            if np.isnan(g_val) or np.isinf(g_val):
                g_val = 0.0
            pose_grads[idx] = g_val
        gradients['pose_rotation'] = pose_grads[:3]
        gradients['pose_translation'] = pose_grads[3:]

    # Normals (n_vertices x 3)
    if normal_dr is not None and 'normals' not in config.frozen_params:
        normal_grads_list = []
        for i in range(3):
            g = np.array(dr.grad(normal_dr[i]))
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            normal_grads_list.append(g)
        gradients['normals'] = np.stack(normal_grads_list, axis=-1)

    # Geometry (n_vertices x 3)
    if vertex_offset_dr is not None and 'vertex_positions' not in config.frozen_params:
        geom_grads_list = []
        for i in range(3):
            g = np.array(dr.grad(vertex_offset_dr[i]))
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            geom_grads_list.append(g)
        gradients['vertex_positions'] = np.stack(geom_grads_list, axis=-1)

    # Patterns
    if pattern_loaders_dr is not None and 'patterns' not in config.frozen_params:
        pat_grads = {}
        for key, loader in pattern_loaders_dr.items():
            g = loader.get_gradients()
            if g is not None:
                g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                pat_grads[key] = g
        if pat_grads:
            gradients['patterns'] = pat_grads

    return gradients, loss_dict


# ============================================================================
# Section 8: Visualization Functions
# ============================================================================

def apply_viz_scale(data: np.ndarray, data_max: float,
                    scale: str = "dB", db_floor: float = -40.0):
    """Apply visualization scale to RA magnitude data."""
    if data_max <= 0:
        data_max = 1.0

    if scale == 'dB':
        data_norm = data / data_max
        data_norm = np.maximum(data_norm, 1e-30)
        data_dB = 20.0 * np.log10(data_norm)
        data_scaled = np.clip(data_dB, db_floor, 0.0)
        return data_scaled, db_floor, 0.0

    elif scale == 'log10':
        data_norm = data / data_max
        data_scaled = np.log10(1.0 + 999.0 * data_norm) / 3.0
        return data_scaled, 0.0, 1.0

    else:  # linear
        data_scaled = data / data_max
        return data_scaled, 0.0, 1.0


def visualize_ra_comparison(ra_rendered: np.ndarray, ra_gt: np.ndarray,
                             iteration: int, output_dir: str, range_res: float,
                             viz_scale: str = "dB", viz_db_floor: float = -40.0):
    """Save GT vs Rendered vs Error RA comparison image."""
    ra_r_cart = ra_polar_to_cartesian(ra_rendered, range_res)
    ra_g_cart = ra_polar_to_cartesian(ra_gt, range_res)

    r_max = np.max(ra_r_cart) if np.max(ra_r_cart) > 0 else 1.0
    g_max = np.max(ra_g_cart) if np.max(ra_g_cart) > 0 else 1.0

    r_scaled, vmin_r, vmax_r = apply_viz_scale(ra_r_cart, r_max, viz_scale, viz_db_floor)
    g_scaled, vmin_g, vmax_g = apply_viz_scale(ra_g_cart, g_max, viz_scale, viz_db_floor)

    # Error on min-max normalized
    r_norm = minmax_normalize_numpy(ra_r_cart)
    g_norm = minmax_normalize_numpy(ra_g_cart)
    error = r_norm - g_norm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(g_scaled.T, origin='lower', aspect='auto', cmap='viridis',
                          vmin=vmin_g, vmax=vmax_g)
    axes[0].set_title(f'Ground Truth ({viz_scale})')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(r_scaled.T, origin='lower', aspect='auto', cmap='viridis',
                          vmin=vmin_r, vmax=vmax_r)
    axes[1].set_title(f'Rendered ({viz_scale})')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error.T, origin='lower', aspect='auto', cmap='RdBu_r',
                          vmin=-0.3, vmax=0.3)
    axes[2].set_title('Error (rendered - GT)')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f'Iteration {iteration + 1}', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'visualizations', f'ra_comparison_{iteration + 1:04d}.png'),
                dpi=150)
    plt.close(fig)


def visualize_bsdf_components(renderer: FMCWRendererRef, ra_gt: np.ndarray,
                               iteration: int, output_dir: str, range_res: float,
                               viz_scale: str = "dB", viz_db_floor: float = -40.0):
    """7-component BSDF decomposition RA visualization (from HFMO)."""
    if not hasattr(renderer.integrator, 'bsdf') or \
       not hasattr(renderer.integrator.bsdf, 'eval_f_cos_components'):
        return

    try:
        component_result = renderer.render_with_components(seed=42 + iteration * 100)
    except Exception as e:
        tqdm.write(f"  Component rendering failed: {e}")
        return

    if component_result is None:
        return

    component_names = ['coherent', 'incoherent', 'ka', 'spm', 'directive', 'broad', 'diffraction']
    component_labels = [
        'Coherent (KA+SPM)', 'Incoherent (Dir+Broad)', 'KA Specular',
        'SPM Diffuse', 'Directive Scatter', 'Broad Diffuse', 'Diffraction (fsdBSDF)',
    ]

    # Convert all components to Cartesian RA
    component_ra = {}
    for name in component_names:
        adc_complex = component_result.get_component(name)
        if adc_complex is None:
            component_ra[name] = None
            continue
        adc_ri = np.stack([adc_complex.real, adc_complex.imag], axis=-1)
        adc_torch = torch.from_numpy(adc_ri).float()
        ra_polar = adc_to_ra_image(adc_torch).detach().cpu().numpy()
        ra_cart = ra_polar_to_cartesian(ra_polar, range_res)
        component_ra[name] = ra_cart

    # Total rendered
    total_adc = component_result.total if hasattr(component_result, 'total') else None
    if total_adc is not None:
        adc_ri = np.stack([total_adc.real, total_adc.imag], axis=-1)
        ra_total = ra_polar_to_cartesian(
            adc_to_ra_image(torch.from_numpy(adc_ri).float()).numpy(), range_res)
    else:
        ra_total = None

    # Global normalization
    all_vals = []
    for ra in component_ra.values():
        if ra is not None:
            all_vals.append(ra.flatten())
    if ra_total is not None:
        all_vals.append(ra_total.flatten())
    if not all_vals:
        return
    global_max = np.max(np.concatenate(all_vals)) if all_vals else 1.0
    global_max = max(global_max, 1e-30)

    ra_gt_cart = ra_polar_to_cartesian(ra_gt, range_res)
    gt_max = max(np.max(ra_gt_cart), 1e-30)

    # 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    layout = [
        (0, component_names[0], component_labels[0]),  # Coherent
        (1, component_names[1], component_labels[1]),  # Incoherent
        (2, component_names[2], component_labels[2]),  # KA
        (3, component_names[3], component_labels[3]),  # SPM
        (4, component_names[4], component_labels[4]),  # Directive
        (5, component_names[5], component_labels[5]),  # Broad
        (6, component_names[6], component_labels[6]),  # Diffraction
    ]

    for idx, name, label in layout:
        row, col = idx // 3, idx % 3
        ra = component_ra.get(name)
        if ra is not None:
            scaled, vmin, vmax = apply_viz_scale(ra, global_max, viz_scale, viz_db_floor)
            axes[row, col].imshow(scaled.T, origin='lower', aspect='auto',
                                   cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, col].set_title(label, fontsize=10)

    # Total rendered
    if ra_total is not None:
        scaled, vmin, vmax = apply_viz_scale(ra_total, global_max, viz_scale, viz_db_floor)
        axes[2, 1].imshow(scaled.T, origin='lower', aspect='auto',
                           cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, 1].set_title('Total Rendered', fontsize=10)

    # Ground Truth
    gt_scaled, vmin_g, vmax_g = apply_viz_scale(ra_gt_cart, gt_max, viz_scale, viz_db_floor)
    axes[2, 2].imshow(gt_scaled.T, origin='lower', aspect='auto',
                       cmap='viridis', vmin=vmin_g, vmax=vmax_g)
    axes[2, 2].set_title('Ground Truth', fontsize=10)

    plt.suptitle(f'BSDF Component Decomposition - Iteration {iteration + 1} ({viz_scale})',
                 fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'components'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'components', f'components_{iteration + 1:04d}.png'),
                dpi=150)
    plt.close(fig)


def plot_training_dashboard(history: Dict, output_dir: str):
    """Plot comprehensive training dashboard (8 rows x 5 columns).

    Layout:
    - Row 0: Total Loss (span 2), Multi-Chirp Components (span 3)
    - Row 1: Regularization Loss (span 2), LR Schedule (span 3)
    - Row 2: Log ADC Mag (L1, MSE, RMSE, MAE, Pearson)
    - Row 3: ADC Phase (L1, MSE, RMSE, MAE, Pearson)
    - Row 4: MinMax ADC Mag (L1, MSE, RMSE, MAE, Pearson)
    - Row 5: RA metrics with alignment (MSE, RMSE, PSNR, SSIM, LPIPS)
    - Row 6: RA metrics without alignment (MSE, RMSE, PSNR, SSIM, LPIPS)
    - Row 7: Cartesian RA metrics (MSE, RMSE, PSNR, SSIM, Corr)
    """
    if not history.get('iteration'):
        return

    iters = history['iteration']
    fig, axes = plt.subplots(8, 5, figsize=(25, 32))
    fig.suptitle('Training Metrics: Loss Components and Reconstruction Quality',
                 fontsize=16, fontweight='bold')

    def safe_plot(ax, key, title, ylabel, color='b', log_scale=False):
        """Helper to safely plot a metric if available."""
        if key in history and len(history[key]) > 0:
            vals = history[key]
            valid_pairs = [(it, v) for it, v in zip(iters, vals)
                          if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if len(valid_pairs) > 0:
                x_vals, valid_vals = zip(*valid_pairs)
                marker_size = 2 if len(valid_vals) > 100 else 3
                ax.plot(x_vals, valid_vals, color=color, linewidth=1.5,
                        marker='o', markersize=marker_size)
                ax.set_xlabel('Iteration', fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(title, fontsize=10, fontweight='bold')
                if log_scale:
                    ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                return True
        ax.set_title(f'{title} (no data)', fontsize=10, fontweight='bold', color='gray')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        return False

    # ---- Row 0: Total Loss (span 2) + Multi-Chirp Components (span 3) ----
    ax_total = plt.subplot2grid((7, 5), (0, 0), colspan=2, fig=fig)
    if 'total_loss' in history and history['total_loss']:
        ax_total.plot(iters, history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    ax_total.set_xlabel('Iteration', fontsize=10)
    ax_total.set_ylabel('Loss', fontsize=10)
    ax_total.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax_total.grid(True, alpha=0.3)
    ax_total.legend()

    ax_mc = plt.subplot2grid((7, 5), (0, 2), colspan=3, fig=fig)
    has_mc = False
    for key, color, label in [('mc_adc_mag', 'tab:blue', 'ADC Mag'),
                               ('mc_ra_mag', 'tab:green', 'RA Mag'),
                               ('mc_phase', 'tab:red', 'Phase')]:
        if key in history and history[key]:
            pairs = [(it, v) for it, v in zip(iters, history[key]) if v is not None]
            if pairs:
                x, y = zip(*pairs)
                ax_mc.plot(x, y, color=color, linewidth=2, label=f'{label} Loss',
                          marker='o', markersize=2)
                has_mc = True
    if has_mc:
        ax_mc.set_title('Multi-Chirp Loss Components', fontsize=12, fontweight='bold')
        ax_mc.legend(loc='upper right')
    else:
        ax_mc.set_title('Multi-Chirp (not enabled)', fontsize=12, fontweight='bold', color='gray')
    ax_mc.set_xlabel('Iteration', fontsize=10)
    ax_mc.grid(True, alpha=0.3)

    for col in range(5):
        axes[0, col].axis('off')

    # ---- Row 1: Regularization (span 2) + LR Schedule (span 3) ----
    ax_reg = plt.subplot2grid((7, 5), (1, 0), colspan=2, fig=fig)
    if 'loss_reg' in history and history['loss_reg']:
        ax_reg.plot(iters, history['loss_reg'], 'r-', linewidth=2, label='Reg Loss')
    if 'adc_mag_loss' in history and history['adc_mag_loss']:
        pairs = [(it, v) for it, v in zip(iters, history['adc_mag_loss']) if v is not None and v > 0]
        if pairs:
            x, y = zip(*pairs)
            ax_reg.plot(x, y, 'm-', linewidth=2, label='ADC Mag Loss')
    ax_reg.set_xlabel('Iteration', fontsize=10)
    ax_reg.set_ylabel('Loss', fontsize=10)
    ax_reg.set_title('Loss Components', fontsize=12, fontweight='bold')
    ax_reg.grid(True, alpha=0.3)
    ax_reg.legend()

    ax_lr = plt.subplot2grid((7, 5), (1, 2), colspan=3, fig=fig)
    if 'lr_multiplier' in history and history['lr_multiplier']:
        ax_lr.plot(iters, history['lr_multiplier'], 'purple', linewidth=2)
        ax_lr.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    else:
        ax_lr.set_title('LR Schedule (no data)', fontsize=12, fontweight='bold', color='gray')
    ax_lr.set_xlabel('Iteration', fontsize=10)
    ax_lr.grid(True, alpha=0.3)

    for col in range(5):
        axes[1, col].axis('off')

    # ---- Row 2: Log ADC Magnitude metrics ----
    safe_plot(axes[2, 0], 'adc_log_mag_l1', 'Log ADC Mag L1', 'L1', color='tab:blue')
    safe_plot(axes[2, 1], 'adc_log_mag_mse', 'Log ADC Mag MSE', 'MSE', color='tab:green', log_scale=True)
    safe_plot(axes[2, 2], 'adc_log_mag_rmse', 'Log ADC Mag RMSE', 'RMSE', color='tab:orange')
    safe_plot(axes[2, 3], 'adc_log_mag_mae', 'Log ADC Mag MAE', 'MAE', color='tab:red')
    safe_plot(axes[2, 4], 'adc_log_mag_pearson', 'Log ADC Mag Pearson', 'Pearson r', color='tab:brown')

    # ---- Row 3: ADC Phase metrics ----
    safe_plot(axes[3, 0], 'adc_relphase_l1', 'ADC Phase L1', 'L1', color='tab:purple')
    safe_plot(axes[3, 1], 'adc_relphase_mse', 'ADC Phase MSE', 'MSE', color='tab:cyan', log_scale=True)
    safe_plot(axes[3, 2], 'adc_relphase_rmse', 'ADC Phase RMSE', 'RMSE', color='tab:pink')
    safe_plot(axes[3, 3], 'adc_relphase_mae', 'ADC Phase MAE', 'MAE', color='tab:olive')
    safe_plot(axes[3, 4], 'adc_relphase_pearson', 'ADC Phase Pearson', 'Pearson r', color='tab:brown')

    # ---- Row 4: MinMax ADC Magnitude metrics ----
    safe_plot(axes[4, 0], 'adc_minmax_mag_l1', 'MinMax ADC L1', 'L1', color='tab:blue')
    safe_plot(axes[4, 1], 'adc_minmax_mag_mse', 'MinMax ADC MSE', 'MSE', color='tab:green', log_scale=True)
    safe_plot(axes[4, 2], 'adc_minmax_mag_rmse', 'MinMax ADC RMSE', 'RMSE', color='tab:orange')
    safe_plot(axes[4, 3], 'adc_minmax_mag_mae', 'MinMax ADC MAE', 'MAE', color='tab:red')
    safe_plot(axes[4, 4], 'adc_minmax_mag_pearson', 'MinMax ADC Pearson', 'Pearson r', color='tab:brown')

    # ---- Row 5: RA metrics with alignment ----
    safe_plot(axes[5, 0], 'ra_mse', 'RA MSE (aligned)', 'MSE', color='tab:pink', log_scale=True)
    safe_plot(axes[5, 1], 'ra_rmse', 'RA RMSE (aligned)', 'RMSE', color='tab:olive')
    safe_plot(axes[5, 2], 'ra_psnr', 'RA PSNR (aligned)', 'PSNR (dB)', color='tab:cyan')
    safe_plot(axes[5, 3], 'ra_ssim', 'RA SSIM (aligned)', 'SSIM', color='orange')
    safe_plot(axes[5, 4], 'ra_lpips', 'RA LPIPS (aligned)', 'LPIPS', color='tab:gray')

    # ---- Row 6: RA metrics without alignment ----
    safe_plot(axes[6, 0], 'ra_unaligned_mse', 'RA MSE (unaligned)', 'MSE', color='tab:pink', log_scale=True)
    safe_plot(axes[6, 1], 'ra_unaligned_rmse', 'RA RMSE (unaligned)', 'RMSE', color='tab:olive')
    safe_plot(axes[6, 2], 'ra_unaligned_psnr', 'RA PSNR (unaligned)', 'PSNR (dB)', color='tab:cyan')
    safe_plot(axes[6, 3], 'ra_unaligned_ssim', 'RA SSIM (unaligned)', 'SSIM', color='orange')
    safe_plot(axes[6, 4], 'ra_unaligned_lpips', 'RA LPIPS (unaligned)', 'LPIPS', color='tab:gray')

    # ---- Row 7: Cartesian RA metrics ----
    safe_plot(axes[7, 0], 'cart_mse', 'Cart RA MSE', 'MSE', color='tab:pink', log_scale=True)
    safe_plot(axes[7, 1], 'cart_rmse', 'Cart RA RMSE', 'RMSE', color='tab:olive')
    safe_plot(axes[7, 2], 'cart_psnr', 'Cart RA PSNR', 'PSNR (dB)', color='tab:cyan')
    safe_plot(axes[7, 3], 'cart_ssim', 'Cart RA SSIM', 'SSIM', color='orange')
    safe_plot(axes[7, 4], 'cart_corr', 'Cart RA Corr', 'Correlation', color='tab:blue')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_physics_materials(raw_params: np.ndarray, iteration: int, output_dir: str):
    """Visualize 6 physics parameters as histograms."""
    phys = reparameterize_physics_params(raw_params)
    names = ["ε' (eps_real)", "ε'' (eps_imag)", "σ_h (m)", "l_c (m)", "τ", "thickness (m)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (ax, name) in enumerate(zip(axes.flatten(), names)):
        vals = phys[:, i]
        ax.hist(vals, bins=50, alpha=0.7)
        ax.set_title(f'{name}\nmean={vals.mean():.4f}', fontsize=9)
        ax.set_xlabel('Value')

    plt.suptitle(f'Physics Parameters - Iteration {iteration + 1}', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'materials'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'materials', f'physics_params_{iteration + 1:04d}.png'),
                dpi=150)
    plt.close(fig)


# ============================================================================
# Section 9: Checkpointing Functions
# ============================================================================

def _extract_pattern_data(pm) -> Optional[Dict]:
    """Extract current pattern loader state as numpy arrays for saving."""
    data = {}
    for key, loader in [('tx', getattr(pm, 'tx_pattern_loader', None)),
                         ('rx', getattr(pm, 'rx_pattern_loader', None))]:
        if loader is not None and hasattr(loader, 'E_plane_linear'):
            data[f'{key}_E_plane'] = np.array(loader.E_plane_linear, dtype=np.float32)
            data[f'{key}_H_plane'] = np.array(loader.H_plane_linear, dtype=np.float32)
    return data if data else None


def save_checkpoint(pm: ParameterManagerSionna, optimizer: SimpleAdamSionna,
                    iteration: int, output_dir: str, metrics: Dict, history: Dict):
    """Save full training state including Adam momentum buffers."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save parameters
    save_dict = {
        'iteration': iteration,
        'raw_params': pm.raw_params,
        'pose_params': pm.pose_params_np,
        'optimizer_t': optimizer.t,
    }
    if pm.normal_params_np is not None:
        save_dict['normal_params'] = pm.normal_params_np
    if pm.vertex_offset_np is not None:
        save_dict['vertex_offset'] = pm.vertex_offset_np

    # Save Adam momentum buffers (m and v for each group)
    for group_name, state in optimizer.states.items():
        if group_name == 'patterns':
            # Pattern states are dict-of-dicts, skip for now
            continue
        if isinstance(state.get('m'), list):
            for i, arr in enumerate(state['m']):
                save_dict[f'adam_m_{group_name}_{i}'] = arr
            for i, arr in enumerate(state['v']):
                save_dict[f'adam_v_{group_name}_{i}'] = arr
        elif isinstance(state.get('m'), np.ndarray):
            save_dict[f'adam_m_{group_name}'] = state['m']
            save_dict[f'adam_v_{group_name}'] = state['v']

    np.savez(os.path.join(ckpt_dir, f'checkpoint_{iteration + 1:06d}.npz'), **save_dict)

    # Save metrics
    with open(os.path.join(ckpt_dir, f'metrics_{iteration + 1:06d}.json'), 'w') as f:
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
                  f, indent=2, default=str)

    # Save training history
    serializable_history = {}
    for k, v in history.items():
        if isinstance(v, list):
            serializable_history[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            serializable_history[k] = v
    with open(os.path.join(ckpt_dir, f'history_{iteration + 1:06d}.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2, default=str)


def save_final_results(pm: ParameterManagerSionna, best_raw_params: np.ndarray,
                        history: Dict, ra_gt: np.ndarray, output_dir: str,
                        best_normal_params=None, best_pose_params=None,
                        best_vertex_offset=None, best_pattern_data=None):
    """Save best materials and all learned parameters as separate files."""
    # Best materials (always saved)
    phys = reparameterize_physics_params(best_raw_params)
    np.savez(os.path.join(output_dir, 'best_materials.npz'),
             raw_params=best_raw_params,
             eps_real=phys[:, 0], eps_imag=phys[:, 1],
             sigma_h=phys[:, 2], l_c=phys[:, 3],
             tau=phys[:, 4], thickness=phys[:, 5])

    # Best normals (if normals were learned)
    if best_normal_params is not None:
        np.savez(os.path.join(output_dir, 'best_normals.npz'),
                 normal_params=best_normal_params)
        print(f"  Saved best_normals.npz ({best_normal_params.shape})")

    # Best patterns (if patterns were learned)
    if best_pattern_data is not None:
        np.savez(os.path.join(output_dir, 'best_patterns.npz'), **best_pattern_data)
        print(f"  Saved best_patterns.npz ({list(best_pattern_data.keys())})")

    # Best pose (if pose was learned)
    if best_pose_params is not None:
        np.savez(os.path.join(output_dir, 'best_pose.npz'),
                 pose_params=best_pose_params)
        print(f"  Saved best_pose.npz ({best_pose_params.shape})")

    # Best vertex offset (if vertex positions were learned)
    if best_vertex_offset is not None:
        np.savez(os.path.join(output_dir, 'best_vertex_offset.npz'),
                 vertex_offset=best_vertex_offset)
        print(f"  Saved best_vertex_offset.npz ({best_vertex_offset.shape})")

    # History
    serializable_history = {}
    for k, v in history.items():
        if isinstance(v, list):
            serializable_history[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        else:
            serializable_history[k] = v
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(serializable_history, f, indent=2, default=str)


def load_checkpoint(pm: ParameterManagerSionna, checkpoint_path: str,
                    optimizer: Optional[SimpleAdamSionna] = None) -> int:
    """Load checkpoint including parameters and optionally Adam state.

    Returns iteration number.
    """
    data = np.load(checkpoint_path, allow_pickle=True)
    pm.raw_params = data['raw_params']
    pm.pose_params_np = data['pose_params']
    if 'normal_params' in data and pm.normal_params_np is not None:
        pm.normal_params_np = data['normal_params']
    if 'vertex_offset' in data and pm.vertex_offset_np is not None:
        pm.vertex_offset_np = data['vertex_offset']

    # Restore Adam state if optimizer provided
    if optimizer is not None:
        if 'optimizer_t' in data:
            optimizer.t = int(data['optimizer_t'])
        for group_name, state in optimizer.states.items():
            if group_name == 'patterns':
                continue
            if isinstance(state.get('m'), list):
                for i in range(len(state['m'])):
                    m_key = f'adam_m_{group_name}_{i}'
                    v_key = f'adam_v_{group_name}_{i}'
                    if m_key in data:
                        state['m'][i] = data[m_key]
                        state['v'][i] = data[v_key]
            elif isinstance(state.get('m'), np.ndarray):
                m_key = f'adam_m_{group_name}'
                v_key = f'adam_v_{group_name}'
                if m_key in data:
                    state['m'] = data[m_key]
                    state['v'] = data[v_key]

    return int(data['iteration'])


# ============================================================================
# Section 10: LR Scheduling & Staged Training
# ============================================================================

def get_lr_scale(iteration: int, config: TrainingConfigSionna) -> float:
    """Compute LR multiplier: linear warmup -> constant."""
    if not config.enable_lr_schedule:
        return 1.0

    warmup = min(config.lr_warmup_iters, max(3, config.num_iterations // 10))
    warmup_factor = config.lr_warmup_factor

    if iteration < warmup:
        return warmup_factor + (1.0 - warmup_factor) * (iteration / max(warmup, 1))
    else:
        return 1.0


def update_frozen_params_for_stage(config: TrainingConfigSionna, iteration: int):
    """Update frozen_params based on training stage."""
    if not config.enable_staged_training:
        return

    if iteration < config.stage1_iterations:
        # Stage 1: pose + patterns only
        config.frozen_params = ['materials', 'normals', 'vertex_positions']
        if iteration == 0:
            tqdm.write(f"  [Stage 1] Training pose+patterns for {config.stage1_iterations} iterations")
    else:
        # Stage 2: materials + normals + geometry
        config.frozen_params = ['pose_rotation', 'pose_translation']
        if iteration == config.stage1_iterations:
            tqdm.write(f"  [Stage 2] Training materials+normals+geometry")


# ============================================================================
# Section 11: Main Training Loop
# ============================================================================

def train(config: TrainingConfigSionna):
    """Main training loop."""

    # ---- Setup ----
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)

    print("=" * 80)
    print("Sionna Renderer Training")
    print("=" * 80)

    # Create renderer
    print("Creating renderer...")
    renderer = create_renderer(config)

    # Load GT ADC
    print(f"Loading GT ADC from {config.gt_adc_file}")
    gt_adc_np = np.load(config.gt_adc_file)
    if gt_adc_np.ndim == 4:
        # Multi-chirp: use first chirp for single-chirp training
        gt_adc_single = gt_adc_np[0]
    elif gt_adc_np.ndim == 3:
        gt_adc_single = gt_adc_np
    else:
        raise ValueError(f"Unexpected GT shape: {gt_adc_np.shape}")

    # Convert to torch (NR, NT, K) complex -> (NT, NR, K, 2) real/imag
    gt_complex = gt_adc_single  # (NR, NT, K) complex
    gt_real_imag = np.stack([gt_complex.real, gt_complex.imag], axis=-1)  # (NR, NT, K, 2)
    # Transpose to (NT, NR, K, 2) for compute_ra_loss_with_gradients
    gt_adc_torch = torch.from_numpy(gt_real_imag.transpose(1, 0, 2, 3).astype(np.float32))

    # Compute GT RA
    ra_gt = adc_to_ra_image(gt_adc_torch).detach().cpu().numpy()

    # Compute range resolution
    range_res = compute_range_res_from_cfg(config.config_file)

    # Load multi-chirp GT if needed
    gt_all_chirps = None
    if config.loss_config.use_multi_chirp_loss:
        print("Loading multi-chirp GT ADC...")
        gt_all_chirps = load_all_chirps(config.gt_adc_file)

    # Create parameter manager
    print("Creating parameter manager...")
    pm = ParameterManagerSionna(renderer, config, config.scene_file)

    NT, NR, K = pm.NT, pm.NR, pm.K
    print(f"  NT={NT}, NR={NR}, K={K}, n_opt_params={pm.n_opt_params}")

    # ITU initialization
    if config.initial_material:
        pm.initialize_from_itu(config.initial_material)

    # Error-map initialization
    if config.enable_error_map_init:
        run_error_map_initialization(pm, ra_gt, range_res)

    # Create optimizer (before checkpoint so we can restore Adam state)
    optimizer = SimpleAdamSionna(pm, config)

    # Load checkpoint
    start_iter = 0
    if config.init_checkpoint:
        print(f"Loading checkpoint: {config.init_checkpoint}")
        start_iter = load_checkpoint(pm, config.init_checkpoint, optimizer=optimizer)
        print(f"  Resuming from iteration {start_iter} (Adam t={optimizer.t})")

    # LPIPS model
    lpips_model = None
    if LPIPS_AVAILABLE:
        try:
            lpips_model = lpips.LPIPS(net='alex').cuda()
        except Exception:
            lpips_model = None

    # History tracking - comprehensive metric suite
    history = {
        'iteration': [], 'total_loss': [], 'loss_reg': [], 'adc_mag_loss': [],
        'polar_corr': [], 'cart_corr': [], 'mse': [], 'lr_multiplier': [],
        # Multi-chirp loss components
        'mc_adc_mag': [], 'mc_ra_mag': [], 'mc_phase': [],
        # Physics params
        'eps_real_mean': [], 'eps_imag_mean': [], 'sigma_h_mean': [],
        'l_c_mean': [], 'tau_mean': [], 'thickness_mean': [],
        # ADC metrics (log magnitude)
        'adc_log_mag_l1': [], 'adc_log_mag_mse': [], 'adc_log_mag_rmse': [],
        'adc_log_mag_mae': [], 'adc_log_mag_pearson': [],
        # ADC metrics (relative phase)
        'adc_relphase_l1': [], 'adc_relphase_mse': [], 'adc_relphase_rmse': [],
        'adc_relphase_mae': [], 'adc_relphase_pearson': [],
        # ADC metrics (minmax magnitude)
        'adc_minmax_mag_l1': [], 'adc_minmax_mag_mse': [], 'adc_minmax_mag_rmse': [],
        'adc_minmax_mag_mae': [], 'adc_minmax_mag_pearson': [],
        # RA metrics (with alignment / from loss)
        'ra_mse': [], 'ra_rmse': [], 'ra_psnr': [], 'ra_ssim': [], 'ra_lpips': [],
        # RA metrics (without alignment / unaligned)
        'ra_unaligned_mse': [], 'ra_unaligned_rmse': [], 'ra_unaligned_psnr': [],
        'ra_unaligned_ssim': [], 'ra_unaligned_lpips': [],
        # Cartesian RA metrics (computed every iteration)
        'cart_psnr': [], 'cart_ssim': [], 'cart_mse': [], 'cart_rmse': [],
    }

    best_loss = float('inf')
    best_raw_params = pm.raw_params.copy()
    best_normal_params = pm.normal_params_np.copy() if pm.normal_params_np is not None else None
    best_pose_params = pm.pose_params_np.copy() if pm.pose_params_np is not None else None
    best_vertex_offset = pm.vertex_offset_np.copy() if pm.vertex_offset_np is not None else None
    best_pattern_data = _extract_pattern_data(pm)
    best_metrics_snapshot = {}
    best_ra_rendered = None
    best_iteration = 0

    print(f"\nStarting training: {config.num_iterations} iterations")
    print(f"  LEARN: MAT={config.LEARN_MAT} POSE={config.LEARN_POSE} "
          f"NRM={config.LEARN_NRM} VTX={config.LEARN_VTX} PAT={config.LEARN_PAT}")
    print(f"  Loss: RA_weight={config.loss_config.ra_mag_weight}, "
          f"ADC_weight={config.loss_config.adc_mag_weight}")
    print(f"  Multi-chirp: {config.loss_config.use_multi_chirp_loss}")
    print()

    # ---- Pre-training: save GT figures and precompute GT Cartesian RA ----
    from mmir.data.ra_utils import ra_polar_to_cartesian, compute_cartesian_ra_metrics, save_ra_cartesian_png
    ra_gt_cart = ra_polar_to_cartesian(ra_gt, range_res)
    np.save(os.path.join(config.output_dir, 'ra_gt_cart.npy'), ra_gt_cart)
    for scale in ('dB', 'linear'):
        save_ra_cartesian_png(
            ra_gt_cart,
            os.path.join(config.output_dir, f'gt_ra_{scale}.png'),
            range_res=range_res, scale=scale, title=f'GT ({scale})')
    print(f"  Saved GT figures + ra_gt_cart.npy to {config.output_dir}")

    # ---- Training Loop ----
    pbar = tqdm(range(start_iter, config.num_iterations), desc="Training", ncols=120)
    for iteration in pbar:
        t_start = time.time()

        # [A] Staged Training Update
        update_frozen_params_for_stage(config, iteration)

        # [B] LR Scheduling
        lr_scale = get_lr_scale(iteration, config)

        # [C] DrJit Cache Management
        # NOTE: flush_kernel_cache() removed — it forces JIT recompilation every
        # iteration, which is extremely expensive when vertex positions carry AD
        # (150K params → large compute graph). The graph structure is identical
        # across iterations, so cached kernels are always valid. This alone saves
        # ~50-70s/iter for VTX+POSE training.
        # dr.flush_kernel_cache()
        # dr.flush_malloc_cache()

        # [D+E] Forward pass (E2E or two-phase depending on config)
        t_forward = time.time()
        try:
            if config.use_end_to_end_ad:
                (adc_real, adc_imag, raw_drjit, pose_dr, normal_dr,
                 vertex_offset_dr, pattern_loaders_dr) = run_end_to_end_forward(pm, iteration)
            else:
                cached_geom = ensure_cached_geometry(pm, iteration, base_seed=42)
                if cached_geom is None:
                    tqdm.write(f"  Warning: No cached geometry at iteration {iteration}, skipping")
                    continue
                (adc_real, adc_imag, raw_drjit, pose_dr, normal_dr,
                 vertex_offset_dr, pattern_loaders_dr) = run_two_phase_forward(pm, cached_geom)
        except Exception as e:
            tqdm.write(f"  Warning: Forward pass failed at iteration {iteration}: {e}")
            continue
        t_forward = time.time() - t_forward

        # [F] Extract RA for metrics (suspend grad)
        ra_rendered = extract_ra_from_adc(adc_real, adc_imag, NT, NR, K)

        # [G] Backward pass & gradient extraction
        t_backward = time.time()
        gradients, loss_dict = run_backward_and_extract_gradients(
            adc_real, adc_imag, gt_adc_torch, config.loss_config,
            raw_drjit, pose_dr, normal_dr, vertex_offset_dr, pattern_loaders_dr,
            config, NT, NR, K,
            gt_all_chirps=gt_all_chirps,
        )
        t_backward = time.time() - t_backward

        # [H] Gradient clipping (T2)
        gradients = clip_gradients(gradients, config)

        # [H2] Track per-group gradient RMS (before optimizer step)
        grad_rms = {}
        for gname, gval in gradients.items():
            if isinstance(gval, np.ndarray):
                grad_rms[gname] = float(np.sqrt(np.mean(gval ** 2)))
            elif isinstance(gval, dict):
                # Pattern gradients: dict of arrays
                rms_vals = [float(np.sqrt(np.mean(v ** 2))) for v in gval.values() if isinstance(v, np.ndarray)]
                grad_rms[gname] = float(np.mean(rms_vals)) if rms_vals else 0.0

        # [I] Optimizer step
        optimizer.step(gradients, lr_scale=lr_scale)

        # [J] Post-step constraints (T5)
        pm.post_step_constraints()

        # [K] Compute metrics
        polar_corr, mse, cart_corr = compute_correlation_metrics(ra_rendered, ra_gt, range_res)
        total_loss = loss_dict.get('total', 0.0)

        # [L] Track history
        history['iteration'].append(iteration)
        history['total_loss'].append(total_loss)
        history['polar_corr'].append(polar_corr)
        history['cart_corr'].append(cart_corr)
        history['mse'].append(mse)
        history['lr_multiplier'].append(lr_scale)
        history['loss_reg'].append(loss_dict.get('loss_reg', 0.0))
        history['adc_mag_loss'].append(loss_dict.get('adc_mag_loss', None))
        history['mc_adc_mag'].append(loss_dict.get('mc_adc_mag', None))
        history['mc_ra_mag'].append(loss_dict.get('mc_ra_mag', None))
        history['mc_phase'].append(loss_dict.get('mc_phase', None))
        # Per-group gradient RMS
        for gname in ['materials', 'pose_rotation', 'pose_translation',
                       'normals', 'vertex_positions', 'patterns']:
            key = f'grad_rms_{gname}'
            if key not in history:
                history[key] = []
            history[key].append(grad_rms.get(gname, None))
        # Parameter state tracking
        if 'pose_translation_norm' not in history:
            history['pose_translation_norm'] = []
            history['pose_rotation_norm'] = []
            history['vertex_offset_rms'] = []
        history['pose_translation_norm'].append(float(np.linalg.norm(pm.pose_params_np[3:])))
        history['pose_rotation_norm'].append(float(np.linalg.norm(pm.pose_params_np[:3])))
        history['vertex_offset_rms'].append(
            float(np.sqrt(np.mean(pm.vertex_offset_np ** 2))) if pm.vertex_offset_np is not None else 0.0)
        phys_means = pm.get_physics_param_means()
        for k, v in phys_means.items():
            if k not in history:
                history[k] = []
            history[k].append(v)

        # [M] Track best (by lowest total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            best_raw_params = pm.raw_params.copy()
            best_ra_rendered = ra_rendered.copy()
            best_iteration = iteration
            if pm.normal_params_np is not None:
                best_normal_params = pm.normal_params_np.copy()
            if pm.pose_params_np is not None:
                best_pose_params = pm.pose_params_np.copy()
            if pm.vertex_offset_np is not None:
                best_vertex_offset = pm.vertex_offset_np.copy()
            best_pattern_data = _extract_pattern_data(pm)

        # [N] tqdm progress bar
        t_total = time.time() - t_start
        pbar.set_postfix({
            'Loss': f'{total_loss:.4e}',
            'Corr': f'{cart_corr:.4f}',
            'Fwd': f'{t_forward:.1f}s',
            'Bwd': f'{t_backward:.1f}s',
            'Tot': f'{t_total:.1f}s',
        })

        # [O] Detailed logging with full metrics suite
        if (iteration + 1) % config.log_interval == 0:
            # RA metrics (unaligned - separate min-max normalization) + Cartesian RA metrics
            ra_rendered_torch = torch.from_numpy(ra_rendered).float()
            ra_gt_torch = torch.from_numpy(ra_gt).float()
            ra_metrics = compute_all_metrics(ra_rendered_torch, ra_gt_torch, lpips_model,
                                             range_res=range_res, ra_gt_cart=ra_gt_cart)
            targets = evaluate_against_targets(ra_metrics)

            # Track RA metrics (unaligned)
            history['ra_unaligned_mse'].append(ra_metrics.get('ra_mse', None))
            history['ra_unaligned_rmse'].append(ra_metrics.get('ra_rmse', None))
            history['ra_unaligned_psnr'].append(ra_metrics.get('ra_psnr', None))
            history['ra_unaligned_ssim'].append(ra_metrics.get('ra_ssim', None))
            history['ra_unaligned_lpips'].append(ra_metrics.get('ra_lpips', None))

            # Track Cartesian RA metrics
            history['cart_psnr'].append(ra_metrics.get('cart_psnr', None))
            history['cart_ssim'].append(ra_metrics.get('cart_ssim', None))
            history['cart_mse'].append(ra_metrics.get('cart_mse', None))
            history['cart_rmse'].append(ra_metrics.get('cart_rmse', None))

            # Per-epoch rendered .png saves (linear + dB)
            ra_rend_cart = ra_metrics.get('_ra_rend_cart')
            if ra_rend_cart is not None:
                viz_dir = os.path.join(config.output_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                for scale in ('dB', 'linear'):
                    save_ra_cartesian_png(
                        ra_rend_cart,
                        os.path.join(viz_dir, f'rendered_ra_{scale}_iter_{iteration+1:04d}.png'),
                        range_res=range_res, scale=scale,
                        title=f'Rendered ({scale}, iter={iteration+1}, corr={cart_corr:.4f})')

            # Update best_metrics_snapshot if this was the best iteration
            if iteration == best_iteration:
                best_metrics_snapshot = {
                    'cart_corr': cart_corr,
                    'cart_psnr': ra_metrics.get('cart_psnr'),
                    'cart_ssim': ra_metrics.get('cart_ssim'),
                    'cart_mse': ra_metrics.get('cart_mse'),
                    'cart_rmse': ra_metrics.get('cart_rmse'),
                    'polar_corr': polar_corr,
                    'total_loss': total_loss,
                    'best_iteration': iteration,
                }

            # ADC metrics (30+ metrics with log/minmax normalization + phase)
            adc_rendered = extract_adc_for_metrics(adc_real, adc_imag, NT, NR, K)
            adc_metrics = compute_adc_metrics(adc_rendered, gt_adc_single)

            # Track ADC log magnitude metrics
            history['adc_log_mag_l1'].append(adc_metrics.get('adc_log_mag_l1', None))
            history['adc_log_mag_mse'].append(adc_metrics.get('adc_log_mag_mse', None))
            history['adc_log_mag_rmse'].append(adc_metrics.get('adc_log_mag_rmse', None))
            history['adc_log_mag_mae'].append(adc_metrics.get('adc_log_mag_mae', None))
            history['adc_log_mag_pearson'].append(adc_metrics.get('adc_log_mag_pearson', None))

            # Track ADC phase metrics
            history['adc_relphase_l1'].append(adc_metrics.get('adc_relphase_l1', None))
            history['adc_relphase_mse'].append(adc_metrics.get('adc_relphase_mse', None))
            history['adc_relphase_rmse'].append(adc_metrics.get('adc_relphase_rmse', None))
            history['adc_relphase_mae'].append(adc_metrics.get('adc_relphase_mae', None))
            history['adc_relphase_pearson'].append(adc_metrics.get('adc_relphase_pearson', None))

            # Track ADC minmax magnitude metrics
            history['adc_minmax_mag_l1'].append(adc_metrics.get('adc_minmax_mag_l1', None))
            history['adc_minmax_mag_mse'].append(adc_metrics.get('adc_minmax_mag_mse', None))
            history['adc_minmax_mag_rmse'].append(adc_metrics.get('adc_minmax_mag_rmse', None))
            history['adc_minmax_mag_mae'].append(adc_metrics.get('adc_minmax_mag_mae', None))
            history['adc_minmax_mag_pearson'].append(adc_metrics.get('adc_minmax_mag_pearson', None))

            # Track RA metrics from loss (with alignment)
            history['ra_mse'].append(loss_dict.get('ra_l2', ra_metrics.get('ra_mse', None)))
            history['ra_rmse'].append(math.sqrt(loss_dict['ra_l2']) if 'ra_l2' in loss_dict else ra_metrics.get('ra_rmse', None))
            history['ra_psnr'].append(ra_metrics.get('ra_psnr', None))
            history['ra_ssim'].append(ra_metrics.get('ra_ssim', None))
            history['ra_lpips'].append(ra_metrics.get('ra_lpips', None))

            tqdm.write(
                f"  [{iteration+1}/{config.num_iterations}] "
                f"corr={cart_corr:.4f} mse={mse:.2e} "
                f"psnr={ra_metrics.get('ra_psnr', 0):.1f}dB "
                f"ssim={ra_metrics.get('ra_ssim', 0):.3f} "
                f"adc_log_l1={adc_metrics.get('adc_log_mag_l1', 0):.4f} "
                f"phase_l1={adc_metrics.get('adc_relphase_l1', 0):.4f} "
                f"({t_forward:.1f}s fwd, {t_backward:.1f}s bwd, {t_total:.1f}s total)"
            )

            # Write live metrics
            live_path = os.path.join(config.output_dir, 'normalized_metrics_live.txt')
            with open(live_path, 'a') as f:
                f.write(f"iter={iteration+1} corr={cart_corr:.4f} mse={mse:.2e} "
                        f"psnr={ra_metrics.get('ra_psnr', 0):.2f} "
                        f"ssim={ra_metrics.get('ra_ssim', 0):.4f} "
                        f"lpips={ra_metrics.get('ra_lpips', float('nan')):.4f} "
                        f"adc_log_l1={adc_metrics.get('adc_log_mag_l1', float('nan')):.4f} "
                        f"adc_phase_l1={adc_metrics.get('adc_relphase_l1', float('nan')):.4f}\n")
        else:
            # Pad metric histories with None on non-log iterations
            for key in ['ra_unaligned_mse', 'ra_unaligned_rmse', 'ra_unaligned_psnr',
                        'ra_unaligned_ssim', 'ra_unaligned_lpips',
                        'adc_log_mag_l1', 'adc_log_mag_mse', 'adc_log_mag_rmse',
                        'adc_log_mag_mae', 'adc_log_mag_pearson',
                        'adc_relphase_l1', 'adc_relphase_mse', 'adc_relphase_rmse',
                        'adc_relphase_mae', 'adc_relphase_pearson',
                        'adc_minmax_mag_l1', 'adc_minmax_mag_mse', 'adc_minmax_mag_rmse',
                        'adc_minmax_mag_mae', 'adc_minmax_mag_pearson',
                        'ra_mse', 'ra_rmse', 'ra_psnr', 'ra_ssim', 'ra_lpips',
                        'cart_psnr', 'cart_ssim', 'cart_mse', 'cart_rmse']:
                history[key].append(None)

        # [P] Visualization
        if (iteration + 1) % config.visualization_interval == 0:
            visualize_ra_comparison(ra_rendered, ra_gt, iteration, config.output_dir,
                                    range_res, config.viz_scale, config.viz_db_floor)
            visualize_bsdf_components(renderer, ra_gt, iteration, config.output_dir,
                                       range_res, config.viz_scale, config.viz_db_floor)
            visualize_physics_materials(pm.raw_params, iteration, config.output_dir)
            plot_training_dashboard(history, config.output_dir)

        # [Q] Checkpointing
        if (iteration + 1) % config.checkpoint_interval == 0:
            save_checkpoint(pm, optimizer, iteration, config.output_dir,
                            {'cart_corr': cart_corr, 'mse': mse, 'loss': total_loss},
                            history)

        # [R] Early stopping
        if config.enable_early_stopping:
            if cart_corr >= config.target_correlation and mse <= config.target_mse:
                tqdm.write(f"  Converged at iteration {iteration+1}! "
                           f"corr={cart_corr:.4f} >= {config.target_correlation}, "
                           f"mse={mse:.2e} <= {config.target_mse}")
                break

    # ---- Final ----
    save_final_results(pm, best_raw_params, history, ra_gt, config.output_dir,
                       best_normal_params=best_normal_params,
                       best_pose_params=best_pose_params,
                       best_vertex_offset=best_vertex_offset,
                       best_pattern_data=best_pattern_data)
    plot_training_dashboard(history, config.output_dir)

    # Save best-iteration artifacts
    if best_ra_rendered is not None:
        ra_best_cart = ra_polar_to_cartesian(best_ra_rendered, range_res)
        np.save(os.path.join(config.output_dir, 'ra_rendered_cart.npy'), ra_best_cart)
        for scale in ('dB', 'linear'):
            save_ra_cartesian_png(
                ra_best_cart,
                os.path.join(config.output_dir, f'rendered_ra_{scale}.png'),
                range_res=range_res, scale=scale,
                title=f'Best Rendered ({scale}, iter={best_iteration+1})')

    # Save best_metrics.json
    best_metrics_out = dict(best_metrics_snapshot)
    best_metrics_out.setdefault('best_iteration', best_iteration)
    best_metrics_out.setdefault('best_loss', best_loss)
    with open(os.path.join(config.output_dir, 'best_metrics.json'), 'w') as f:
        json.dump(best_metrics_out, f, indent=2)

    print(f"\nTraining complete!")
    print(f"  Best loss: {best_loss:.4e} at iteration {best_iteration+1}")
    if best_metrics_snapshot:
        print(f"  Best cart_corr: {best_metrics_snapshot.get('cart_corr', 'N/A')}")
    print(f"  Results saved to: {config.output_dir}")

    return history


# ============================================================================
# Section 12: Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sionna Renderer Training")
    parser.add_argument('--config', type=str, required=True,
                        help="JSON config file")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Override output directory")
    parser.add_argument('--max-iterations', type=int, default=None,
                        help="Override max iterations")
    parser.add_argument('--initial-material', type=str, default=None,
                        help="ITU material name for initialization")
    parser.add_argument('--error-init', action='store_true',
                        help="Run error-map initialization")
    args = parser.parse_args()

    # Load config from JSON
    config = load_config_from_json(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_iterations:
        config.num_iterations = args.max_iterations
    if args.initial_material:
        config.initial_material = args.initial_material
    if args.error_init:
        config.enable_error_map_init = True

    # Train
    train(config)


if __name__ == '__main__':
    main()
