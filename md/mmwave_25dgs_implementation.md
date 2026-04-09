# mm25DGS: Detailed Implementation Plan

This document specifies every file, class, function, and integration point needed to implement the 2.5D Gaussian Splatting mmWave renderer at `/home/adnan/Desktop/mm3DGS/mm25DGS/`.

---

## Directory Layout

```
mm25DGS/
  __init__.py
  config.py                  # Config dataclass + radar param extraction
  gaussian_model.py          # 2.5D surfel storage + derived quantities
  initialization.py          # LiDAR point cloud -> Gaussians
  rae_grid.py                # RAE grid construction + coordinate transforms
  splatting.py               # Project Gaussians to RAE, rasterize
  culling.py                 # Contribution-based active-set selection
  bsdf_torch.py              # PyTorch wrapper around DrJit BSDF
  antenna_torch.py           # PyTorch wrapper around DrJit antenna patterns
  adc_synthesis.py           # Per-Gaussian exact-phase ADC synthesis
  ray_surfel.py              # Analytical ray-surfel intersection
  multibounce.py             # Multi-bounce via ray-surfel tracing
  losses.py                  # Loss functions (wraps existing FFT + loss)
  reparameterization.py      # Material raw <-> physics transforms
  optimizer.py               # Per-group Adam with gradient clipping
  density_control.py         # Densification, pruning, opacity reset
  training.py                # Training loop orchestrator
  eval_adapter.py            # Adapter for mmir/evaluation/ pipeline
  train_cli.py               # CLI entry point: python -m mm25DGS.train_cli
```

---

## File-by-File Specification

---

### 1. `__init__.py`

```python
"""mm25DGS: 2.5D Gaussian Splatting for mmWave Radar."""
```

No imports. Just a package marker.

---

### 2. `config.py`

Defines all hyperparameters and extracts radar hardware parameters from the JSON config files used by mmIR.

```python
from dataclasses import dataclass, field
from typing import Optional, List
import json
import numpy as np

C = 299_792_458.0  # speed of light (m/s)

@dataclass
class RadarConfig:
    """Hardware parameters extracted from a single JSON config file."""
    center_freq: float          # Hz, ~77e9
    chirp_slope: float          # Hz/s, ~79e12
    sample_rate: float          # Hz, ~8e6
    num_adc_samples: int        # 256
    chirp_duration: float       # seconds, computed from rampEndTime - adcStartTime
    adc_start_time: float       # seconds

    tx_positions_mm: np.ndarray # (N_tx, 3)
    tx_boresights: np.ndarray   # (N_tx, 3) unit vectors
    rx_positions_mm: np.ndarray # (N_rx, 3)
    rx_boresights: np.ndarray   # (N_rx, 3) unit vectors

    n_tx: int
    n_rx: int

    @property
    def wavelength(self) -> float:
        return C / self.center_freq

    @property
    def bandwidth(self) -> float:
        return self.chirp_slope * self.chirp_duration

    @property
    def range_resolution(self) -> float:
        return C / (2.0 * self.bandwidth)

    @property
    def tx_positions_m(self) -> np.ndarray:
        return self.tx_positions_mm * 1e-3

    @property
    def rx_positions_m(self) -> np.ndarray:
        return self.rx_positions_mm * 1e-3

    @classmethod
    def from_json(cls, path: str) -> 'RadarConfig':
        """Parse the aligned config JSON (same format as mmIR).

        JSON schema (from mmir/sensor/config.py:FMCWConfig.from_json):
          carrierFrequency: float (Hz)
          freqSlope: float (Hz/s)
          sampleRate: float (Hz)
          numAdcSamples: int
          adcStartTime: float (s)
          rampEndTime: float (s)
          tx_array: list of {pos_mm: [x,y,z], boresight: [bx,by,bz], ...}
          rx_array: list of {pos_mm: [x,y,z], boresight: [bx,by,bz], ...}
        """
        with open(path) as f:
            cfg = json.load(f)

        tx_arr = cfg['tx_array']
        rx_arr = cfg['rx_array']

        tx_pos = np.array([t['pos_mm'] for t in tx_arr], dtype=np.float64)
        tx_bore = np.array([t['boresight'] for t in tx_arr], dtype=np.float64)
        rx_pos = np.array([r['pos_mm'] for r in rx_arr], dtype=np.float64)
        rx_bore = np.array([r['boresight'] for r in rx_arr], dtype=np.float64)

        adc_start = cfg.get('adcStartTime', cfg.get('adc_start_time', 2e-6))
        ramp_end = cfg.get('rampEndTime', cfg.get('ramp_end_time', 3.4e-5))

        return cls(
            center_freq=cfg['carrierFrequency'],
            chirp_slope=cfg['freqSlope'],
            sample_rate=cfg['sampleRate'],
            num_adc_samples=cfg['numAdcSamples'],
            chirp_duration=ramp_end - adc_start,
            adc_start_time=adc_start,
            tx_positions_mm=tx_pos,
            tx_boresights=tx_bore,
            rx_positions_mm=rx_pos,
            rx_boresights=rx_bore,
            n_tx=len(tx_arr),
            n_rx=len(rx_arr),
        )


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop."""

    # --- Data paths ---
    scene_dir: str = ""            # e.g. data/seq_0_frame_135
    config_path: str = ""          # aligned JSON config
    output_dir: str = ""           # where to save checkpoints

    # --- Antenna patterns ---
    tx_pattern_path: str = "assets/antenna_pattern/MMWCAS/tx1_76.npy"
    rx_pattern_path: str = "assets/antenna_pattern/MMWCAS/rx1_76.npy"

    # --- Gaussian init ---
    pcl_path: str = ""             # LiDAR point cloud .npy
    pca_k_neighbors: int = 20
    initial_scale_clamp_min: float = 0.01   # metres
    initial_scale_clamp_max: float = 0.50
    initial_material: str = "concrete"      # ITU material name for init

    # --- Culling ---
    culling_threshold: float = 0.97         # cumulative amplitude fraction
    culling_full_inclusion_interval: int = 10

    # --- Training ---
    max_iterations: int = 500
    seed: int = 42

    # Per-group learning rates
    lr_positions: float = 1.6e-4
    lr_positions_final: float = 1.6e-6
    lr_rotations: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_materials: float = 0.5

    # Per-group gradient clipping (RMS)
    clip_positions: float = 1.0
    clip_rotations: float = 0.5
    clip_scales: float = 1.0
    clip_opacities: float = 1.0
    clip_materials: float = 1.0

    # Material sub-LR scales (per column of the 6-param vector)
    material_lr_scales: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.3, 0.3, 0.5, 0.1]
    )

    # --- Density control ---
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.01
    opacity_reset_interval: int = 500
    prune_range_min: float = 1.5        # metres
    prune_range_max: float = 30.0

    # --- Loss ---
    ra_mag_weight: float = 1.0
    adc_mag_weight: float = 0.0
    phase_weight: float = 0.0
    ra_use_log: bool = True
    log_epsilon: float = 1e-6

    # --- Multi-bounce ---
    enable_multibounce: bool = False
    multibounce_warmup: int = 100
    n_first_bounce_samples: int = 3000
    n_secondary_rays: int = 8
    max_bounces: int = 3

    # --- Misc ---
    log_interval: int = 10
    checkpoint_interval: int = 50
    device: str = "cuda:0"
```

**Integration**: `RadarConfig.from_json` parses the same JSON files as `mmir/sensor/config.py:FMCWConfig.from_json`. The key names match (`carrierFrequency`, `freqSlope`, `sampleRate`, `numAdcSamples`, `adcStartTime`, `rampEndTime`, `tx_array`, `rx_array`).

---

### 3. `gaussian_model.py`

Central data structure holding all per-Gaussian parameters as contiguous PyTorch tensors.

```python
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class GaussianModel(nn.Module):
    """
    2.5D Gaussian Surfel representation.

    Each Gaussian is a flat disc with:
      - Centre position mu (3,)
      - Rotation quaternion q (4,) -> tangent frame [t1, t2, n]
      - Two lateral log-scales s (2,)
      - Logit-opacity alpha (1,)
      - Raw material parameters (6,) using mmIR reparameterisation

    All tensors have leading dimension N (number of Gaussians).
    """

    def __init__(self, N: int, device: str = "cuda:0"):
        super().__init__()
        self.device = device
        # Learnable parameters (registered as nn.Parameters for autograd)
        self.positions    = nn.Parameter(torch.zeros(N, 3, device=device))          # mu
        self.rotations    = nn.Parameter(torch.zeros(N, 4, device=device))          # q (wxyz)
        self.log_scales   = nn.Parameter(torch.zeros(N, 2, device=device))          # s
        self.logit_opacities = nn.Parameter(torch.zeros(N, 1, device=device))       # alpha
        self.raw_materials = nn.Parameter(torch.zeros(N, 6, device=device))         # mat

        # Initialise quaternions to identity [1, 0, 0, 0]
        with torch.no_grad():
            self.rotations[:, 0] = 1.0

    @property
    def N(self) -> int:
        return self.positions.shape[0]

    # ------------------------------------------------------------------ #
    #  Derived geometric quantities                                       #
    # ------------------------------------------------------------------ #

    def get_rotation_matrices(self) -> Tensor:
        """Quaternion (N,4) -> rotation matrix (N,3,3).

        Convention: q = [w, x, y, z].
        Columns of R are [t1, t2, normal].
        """
        q = torch.nn.functional.normalize(self.rotations, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros(self.N, 3, 3, device=self.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        return R

    def get_normals(self) -> Tensor:
        """(N, 3) surfel normals = third column of rotation matrix."""
        R = self.get_rotation_matrices()
        return R[:, :, 2]   # (N, 3)

    def get_tangent_frame(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns (t1, t2, n) each (N, 3)."""
        R = self.get_rotation_matrices()
        return R[:, :, 0], R[:, :, 1], R[:, :, 2]

    def get_scales(self) -> Tensor:
        """(N, 2) positive lateral scales."""
        return torch.exp(self.log_scales)

    def get_opacities(self) -> Tensor:
        """(N, 1) opacities in [0, 1]."""
        return torch.sigmoid(self.logit_opacities)

    def get_covariance_3d(self) -> Tensor:
        """(N, 3, 3) rank-2 world-space covariance.

        Sigma = s1^2 * t1 @ t1^T  +  s2^2 * t2 @ t2^T
        """
        t1, t2, _ = self.get_tangent_frame()       # (N, 3) each
        s = self.get_scales()                       # (N, 2)
        s1_sq = (s[:, 0] ** 2).unsqueeze(-1).unsqueeze(-1)  # (N,1,1)
        s2_sq = (s[:, 1] ** 2).unsqueeze(-1).unsqueeze(-1)

        t1_outer = t1.unsqueeze(-1) @ t1.unsqueeze(-2)  # (N,3,3)
        t2_outer = t2.unsqueeze(-1) @ t2.unsqueeze(-2)
        return s1_sq * t1_outer + s2_sq * t2_outer       # (N,3,3)

    # ------------------------------------------------------------------ #
    #  Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Save all parameters to a .pt file."""
        torch.save({
            'positions': self.positions.data,
            'rotations': self.rotations.data,
            'log_scales': self.log_scales.data,
            'logit_opacities': self.logit_opacities.data,
            'raw_materials': self.raw_materials.data,
        }, path)

    def load(self, path: str):
        """Load parameters from a .pt file."""
        ckpt = torch.load(path, map_location=self.device)
        with torch.no_grad():
            self.positions.copy_(ckpt['positions'])
            self.rotations.copy_(ckpt['rotations'])
            self.log_scales.copy_(ckpt['log_scales'])
            self.logit_opacities.copy_(ckpt['logit_opacities'])
            self.raw_materials.copy_(ckpt['raw_materials'])
```

---

### 4. `initialization.py`

```python
import numpy as np
import torch
from typing import Optional
from .gaussian_model import GaussianModel
from .reparameterization import inverse_reparameterize, ITU_DEFAULTS

def initialize_from_lidar(
    pcl_path: str,
    device: str = "cuda:0",
    k_neighbors: int = 20,
    scale_clamp_min: float = 0.01,
    scale_clamp_max: float = 0.50,
    initial_material: str = "concrete",
) -> GaussianModel:
    """
    Create a GaussianModel from an aggregated LiDAR point cloud.

    Args:
        pcl_path: Path to .npy file with shape (M, 7):
                  columns [x, y, z, nx, ny, nz, intensity].
                  This is the format used in data/seq_X_frame_Y/scene/pcl.npy.
        device: PyTorch device.
        k_neighbors: Number of nearest neighbours for local PCA.
        scale_clamp_min/max: Bounds on initial lateral scale (metres).
        initial_material: ITU material name for default material params.

    Returns:
        GaussianModel with N = M Gaussians (no downsampling).
    """
    pcl = np.load(pcl_path)                     # (M, 7)
    xyz = pcl[:, :3].astype(np.float32)         # (M, 3)
    normals = pcl[:, 3:6].astype(np.float32)    # (M, 3)
    N = xyz.shape[0]

    model = GaussianModel(N, device=device)

    # --- positions ---
    with torch.no_grad():
        model.positions.copy_(torch.from_numpy(xyz).to(device))

    # --- rotations from normals via local PCA ---
    #   For each point, build a local frame [t1, t2, n] where n = normal.
    #   Use PCA of the k-NN neighbourhood projected onto the tangent plane
    #   to determine t1 and t2 orientations, and the eigenvalues to
    #   initialise the two lateral scales.
    rotations_np, scales_np = _compute_local_frames_and_scales(
        xyz, normals, k_neighbors, scale_clamp_min, scale_clamp_max
    )

    with torch.no_grad():
        model.rotations.copy_(torch.from_numpy(rotations_np).to(device))
        model.log_scales.copy_(torch.from_numpy(np.log(scales_np)).to(device))

    # --- opacities: logit(0.5) = 0 ---
    # Already zero-initialised in GaussianModel.__init__

    # --- materials: ITU default in raw space ---
    physics_default = ITU_DEFAULTS[initial_material]   # (6,) float32
    raw_default = inverse_reparameterize(physics_default)
    with torch.no_grad():
        model.raw_materials.copy_(
            torch.from_numpy(raw_default).unsqueeze(0).expand(N, -1).to(device)
        )

    return model


def _compute_local_frames_and_scales(
    xyz: np.ndarray,         # (N, 3)
    normals: np.ndarray,     # (N, 3)
    k: int,
    s_min: float,
    s_max: float,
) -> tuple:  # (rotations (N,4), scales (N,2))
    """
    For each point:
      1. Find k nearest neighbours.
      2. Centre and project onto tangent plane (remove normal component).
      3. PCA on projected 2D cloud -> two eigenvalues = scale^2.
      4. Build rotation quaternion from [t1, t2, n].

    Uses scipy.spatial.KDTree for the neighbour search (runs on CPU,
    called once at init).
    """
    from scipy.spatial import KDTree

    tree = KDTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)   # (N, k+1), first col is self
    idx = idx[:, 1:]                     # drop self

    N = xyz.shape[0]
    quats = np.zeros((N, 4), dtype=np.float32)
    scales = np.zeros((N, 2), dtype=np.float32)

    for i in range(N):
        n_i = normals[i]
        n_i = n_i / (np.linalg.norm(n_i) + 1e-12)

        neighbors = xyz[idx[i]] - xyz[i]                 # (k, 3)
        proj = neighbors - np.outer(neighbors @ n_i, n_i)  # project to tangent plane

        # PCA in 3D on projected points
        cov = (proj.T @ proj) / k
        eigvals, eigvecs = np.linalg.eigh(cov)             # ascending order

        # Two largest eigenvalues correspond to tangent directions
        t1 = eigvecs[:, 2]  # largest
        t2 = eigvecs[:, 1]  # second largest

        # Ensure right-handed: t1 x t2 should align with n_i
        cross = np.cross(t1, t2)
        if np.dot(cross, n_i) < 0:
            t2 = -t2

        # Build rotation matrix [t1, t2, n] and convert to quaternion
        R = np.column_stack([t1, t2, n_i])
        q = _rotation_matrix_to_quaternion(R)
        quats[i] = q

        # Scales from sqrt of eigenvalues, clamped
        s1 = np.clip(np.sqrt(max(eigvals[2], 1e-12)), s_min, s_max)
        s2 = np.clip(np.sqrt(max(eigvals[1], 1e-12)), s_min, s_max)
        scales[i] = [s1, s2]

    return quats, scales


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to [w, x, y, z] quaternion.

    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)
```

---

### 5. `reparameterization.py`

Replicates the exact raw-space <-> physics-space transforms from `train.py:ParameterManagerSionna`.

```python
import numpy as np
import torch
from torch import Tensor

# ------------------------------------------------------------------
# ITU default material parameters (physics space)
# Format: [eps_real, eps_imag, sigma_h, l_c, tau, thickness]
# ------------------------------------------------------------------
ITU_DEFAULTS = {
    "concrete": np.array([5.31, 0.0326, 1e-4, 5e-3, 0.5, 0.15], dtype=np.float32),
    "brick":    np.array([3.75, 0.038,  3e-4, 8e-3, 0.4, 0.10], dtype=np.float32),
    "glass":    np.array([6.27, 0.0043, 1e-5, 1e-3, 0.8, 0.006], dtype=np.float32),
    "metal":    np.array([1.0,  1e6,    1e-5, 1e-3, 0.9, 0.01],  dtype=np.float32),
    "wood":     np.array([1.99, 0.012,  5e-4, 1e-2, 0.3, 0.05],  dtype=np.float32),
}

# ------------------------------------------------------------------
# Raw-space bounds (same as TrainingConfigSionna defaults)
# ------------------------------------------------------------------
EPS_IMAG_RAW_UPPER = 4.6     # exp(4.6) ~ 100
SIGMA_H_RAW_UPPER  = -2.3    # exp(-2.3) ~ 0.1 m
THICKNESS_RAW_UPPER = -0.7   # exp(-0.7) ~ 0.5 m


def reparameterize(raw: Tensor) -> Tensor:
    """Map unconstrained raw parameters -> bounded physics parameters.

    Args:
        raw: (*, 6) tensor in raw (optimiser) space.
    Returns:
        (*, 6) tensor in physics space:
          col 0: eps_real   in [1.5, 10.0]   via sigmoid * 8.5 + 1.5
          col 1: eps_imag   in [~1e-3, ~100]  via exp(clamp)
          col 2: sigma_h    in [~1e-7, ~0.1]  via exp(clamp)
          col 3: l_c        in [~5e-4, 0.1]   via exp(clamp)
          col 4: tau        in [0.05, 0.95]   via sigmoid * 0.9 + 0.05
          col 5: thickness  in [~1e-3, ~0.5]  via exp(clamp)
    """
    out = torch.empty_like(raw)
    out[..., 0] = 1.5 + 8.5 * torch.sigmoid(raw[..., 0])
    out[..., 1] = torch.exp(torch.clamp(raw[..., 1], -7.0, EPS_IMAG_RAW_UPPER))
    out[..., 2] = torch.exp(torch.clamp(raw[..., 2], -16.0, SIGMA_H_RAW_UPPER))
    out[..., 3] = torch.exp(torch.clamp(raw[..., 3], -7.6, -2.3))
    out[..., 4] = 0.05 + 0.9 * torch.sigmoid(raw[..., 4])
    out[..., 5] = torch.exp(torch.clamp(raw[..., 5], -7.0, THICKNESS_RAW_UPPER))
    return out


def inverse_reparameterize(physics: np.ndarray) -> np.ndarray:
    """Map physics parameters -> raw (optimiser) space. NumPy, used at init.

    Args:
        physics: (6,) or (N, 6) array in physics space.
    Returns:
        Same shape in raw space.
    """
    raw = np.empty_like(physics)

    def logit(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))

    raw[..., 0] = logit((physics[..., 0] - 1.5) / 8.5)
    raw[..., 1] = np.log(np.clip(physics[..., 1], 1e-3, np.exp(EPS_IMAG_RAW_UPPER)))
    raw[..., 2] = np.log(np.clip(physics[..., 2], 1e-7, np.exp(SIGMA_H_RAW_UPPER)))
    raw[..., 3] = np.log(np.clip(physics[..., 3], 5e-4, 0.1))
    raw[..., 4] = logit((physics[..., 4] - 0.05) / 0.9)
    raw[..., 5] = np.log(np.clip(physics[..., 5], 1e-3, np.exp(THICKNESS_RAW_UPPER)))
    return raw.astype(np.float32)
```

---

### 6. `rae_grid.py`

```python
import numpy as np
import torch
from torch import Tensor
from .config import RadarConfig, C


class RAEGrid:
    """
    Precomputed Range-Azimuth-Elevation grid.

    Bin centres and edges are computed exactly as in
    mmir/evaluation/utils/single_view_proc.py:make_angle_grids_np()
    and mmir/evaluation/utils/single_view_viz.py:build_fov_wireframe().
    """

    def __init__(self, radar_cfg: RadarConfig, az_fft_size: int = 128,
                 el_fft_size: int = 128, device: str = "cuda:0"):
        self.device = device

        # --- Range axis ---
        self.N_r = radar_cfg.num_adc_samples
        self.range_res = radar_cfg.range_resolution
        self.range_axis = torch.arange(self.N_r, device=device) * self.range_res  # (N_r,)

        # --- Azimuth axis (arcsin-spaced, DC removed) ---
        self.N_az_full = az_fft_size
        self.N_az = az_fft_size - 1          # 127 after DC removal
        t_az = (torch.arange(-az_fft_size // 2 + 1, az_fft_size // 2,
                             device=device).float() * (2.0 / az_fft_size))
        t_az = torch.clamp(t_az, -1.0 + 1e-6, 1.0 - 1e-6)
        self.az_angles = torch.arcsin(t_az)  # (N_az,) radians

        # --- Elevation axis ---
        self.N_el_full = el_fft_size
        self.N_el = el_fft_size - 1
        t_el = (torch.arange(-el_fft_size // 2 + 1, el_fft_size // 2,
                             device=device).float() * (2.0 / el_fft_size))
        t_el = torch.clamp(t_el, -1.0 + 1e-6, 1.0 - 1e-6)
        self.el_angles = torch.arcsin(t_el)  # (N_el,)

    def cartesian_to_rae(self, points: Tensor) -> Tensor:
        """Convert (*, 3) Cartesian -> (*, 3) [range, azimuth, elevation].

        Coordinate convention (from mmir eval):
          x -> azimuth  (sin(az))
          y -> range     (forward / boresight)
          z -> elevation (sin(el))
        """
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-12)
        az = torch.atan2(x, y)
        el = torch.asin(torch.clamp(z / r, -1 + 1e-6, 1 - 1e-6))
        return torch.stack([r, az, el], dim=-1)

    def compute_jacobian(self, points: Tensor) -> Tensor:
        """Jacobian d(r, az, el)/d(x, y, z) at each point.

        Args:
            points: (N, 3) Cartesian positions.
        Returns:
            (N, 3, 3) Jacobian matrices.
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-12)
        rho = torch.sqrt(x**2 + y**2 + 1e-12)

        J = torch.zeros(points.shape[0], 3, 3, device=points.device)
        # d(r)/d(x,y,z)
        J[:, 0, 0] = x / r
        J[:, 0, 1] = y / r
        J[:, 0, 2] = z / r
        # d(az)/d(x,y,z)
        J[:, 1, 0] = y / (rho**2)
        J[:, 1, 1] = -x / (rho**2)
        # d(el)/d(x,y,z)
        J[:, 2, 0] = -x * z / (r**2 * rho)
        J[:, 2, 1] = -y * z / (r**2 * rho)
        J[:, 2, 2] = rho / (r**2)
        return J
```

---

### 7. `bsdf_torch.py`

PyTorch-compatible wrapper around mmIR's DrJit BSDF. Transfers tensors between frameworks at the boundary.

```python
"""
PyTorch-compatible BSDF evaluation via mmIR's DrJit BSDF.

Strategy: Convert PyTorch tensors -> DrJit arrays, call eval_f_cos_physics(),
convert result back to PyTorch. Gradients flow via a custom autograd Function
that caches the forward computation and uses finite differences (or DrJit AD)
for the backward pass.

NOTE: For the initial implementation, the BSDF is evaluated in the FORWARD
pass only (no gradient through BSDF internals). Gradients reach material
parameters through the amplitude term, which is sufficient since mmIR also
detaches phase and relies on amplitude-only gradients.
"""
import torch
import numpy as np
from torch import Tensor

# Lazy imports to avoid triggering mi.set_variant at import time
_bsdf_instance = None

def _get_bsdf():
    global _bsdf_instance
    if _bsdf_instance is None:
        import mitsuba as mi
        mi.set_variant('cuda_ad_rgb')
        from mmir.renderer.bsdf.mmwave_scalar import BSDFmmWaveScalar
        _bsdf_instance = BSDFmmWaveScalar(
            polarization='vertical',
            enable_incoherent=True,
            enable_slab_fresnel=True,
        )
    return _bsdf_instance


def eval_bsdf_physics_batch(
    wo: Tensor,           # (N, 3) outgoing direction (toward RX)
    wi: Tensor,           # (N, 3) incoming direction (from TX)
    normals: Tensor,      # (N, 3) surface normals
    eps_real: Tensor,     # (N,)
    eps_imag: Tensor,     # (N,)
    sigma_h: Tensor,      # (N,)
    l_c: Tensor,          # (N,)
    tau: Tensor,          # (N,)
    thickness: Tensor,    # (N,)
) -> Tensor:
    """
    Evaluate f_cos = BSDF(wo, wi, n, material) * |cos(theta_i)|.

    All inputs are PyTorch tensors on CUDA. Returns (N,) tensor.
    Runs forward-only through DrJit (no AD attachment).
    """
    import drjit as dr
    import mitsuba as mi

    bsdf = _get_bsdf()

    # PyTorch -> NumPy -> DrJit
    def to_drjit_vec(t):
        a = t.detach().cpu().numpy().astype(np.float32)
        return mi.Vector3f(a[:, 0], a[:, 1], a[:, 2])

    def to_drjit_float(t):
        return mi.Float(t.detach().cpu().numpy().astype(np.float32))

    wo_dr = to_drjit_vec(wo)
    wi_dr = to_drjit_vec(wi)
    n_dr  = to_drjit_vec(normals)

    result_dr = bsdf.eval_f_cos_physics(
        wo_dr, wi_dr, n_dr,
        to_drjit_float(eps_real),
        to_drjit_float(eps_imag),
        to_drjit_float(sigma_h),
        to_drjit_float(l_c),
        to_drjit_float(tau),
        to_drjit_float(thickness),
    )

    # DrJit -> NumPy -> PyTorch
    dr.eval(result_dr)
    result_np = np.array(result_dr, dtype=np.float32)
    return torch.from_numpy(result_np).to(wo.device)
```

**Note on gradient flow**: In the initial implementation, BSDF evaluation is forward-only. Gradients reach material parameters through the *amplitude* term `A_g = alpha * bsdf_val * antenna * path_loss`, where `bsdf_val` is treated as a detached constant w.r.t. materials. The optimizer still updates materials because the *loss landscape* changes with materials (different materials produce different amplitudes in the next forward pass). This matches mmIR's approach where amplitude-only gradients (phase detached) are sufficient.

For a tighter gradient path, a `torch.autograd.Function` wrapping DrJit AD can be added later. The function would call `dr.enable_grad()` on the material DrJit arrays, run `dr.backward()` after the forward, and inject the DrJit-computed gradients back into the PyTorch graph.

---

### 8. `antenna_torch.py`

```python
"""
PyTorch wrapper for mmIR antenna pattern evaluation.

Similar strategy to bsdf_torch.py: PyTorch -> DrJit -> evaluate -> PyTorch.
Pattern data is loaded once and cached.
"""
import torch
import numpy as np
from torch import Tensor
from typing import Optional

_tx_loader = None
_rx_loader = None


def load_patterns(tx_pattern_path: str, rx_pattern_path: str):
    """Load antenna patterns from .npy files (called once at init)."""
    global _tx_loader, _rx_loader
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    from mmir.sensor.element_patterns import AntennaPatternLoader
    _tx_loader = AntennaPatternLoader(tx_pattern_path)
    _rx_loader = AntennaPatternLoader(rx_pattern_path)


def evaluate_tx_gain(
    directions: Tensor,             # (N, 3) unit vectors TX->Gaussian
    orientations: Tensor,           # (N, 3) TX boresight directions
) -> Tensor:
    """Returns (N,) linear-scale TX antenna gain."""
    return _evaluate_gain(_tx_loader, directions, orientations)


def evaluate_rx_gain(
    directions: Tensor,             # (N, 3) unit vectors Gaussian->RX (NEGATED internally)
    orientations: Tensor,           # (N, 3) RX boresight directions
) -> Tensor:
    """Returns (N,) linear-scale RX antenna gain.

    Note: mmIR convention is to negate the direction for RX patterns
    (incoming wave direction). This function handles the negation.
    """
    return _evaluate_gain(_rx_loader, -directions, orientations)


def _evaluate_gain(loader, directions: Tensor, orientations: Tensor) -> Tensor:
    """Evaluate antenna pattern gain (forward-only, no DrJit AD)."""
    import drjit as dr
    import mitsuba as mi
    from mmir.sensor.element_patterns import evaluate_combined_gain

    d_np = directions.detach().cpu().numpy().astype(np.float32)
    o_np = orientations.detach().cpu().numpy().astype(np.float32)

    d_dr = mi.Vector3f(d_np[:, 0], d_np[:, 1], d_np[:, 2])
    o_dr = mi.Vector3f(o_np[:, 0], o_np[:, 1], o_np[:, 2])

    gain_dr = evaluate_combined_gain(loader, d_dr, o_dr)
    dr.eval(gain_dr)
    gain_np = np.array(gain_dr, dtype=np.float32)
    return torch.from_numpy(gain_np).to(directions.device)
```

---

### 9. `culling.py`

```python
import torch
from torch import Tensor
from .gaussian_model import GaussianModel


def compute_contribution_estimates(
    model: GaussianModel,
    radar_center: Tensor,      # (3,) radar position in world frame
    radar_boresight: Tensor,   # (3,) average boresight direction
) -> Tensor:
    """
    Fast per-Gaussian amplitude estimate for contribution culling.

    Returns (N,) tensor of estimated contribution magnitudes.
    Cheap: only uses opacity, normal alignment, and range.
    """
    mu = model.positions                          # (N, 3)
    normals = model.get_normals()                 # (N, 3)
    opacities = model.get_opacities().squeeze(-1) # (N,)

    # Direction from radar to each Gaussian
    to_g = mu - radar_center.unsqueeze(0)         # (N, 3)
    dist = torch.norm(to_g, dim=-1).clamp(min=0.1)  # (N,)
    to_g_norm = to_g / dist.unsqueeze(-1)

    # Normal alignment: |cos(angle between normal and radar direction)|
    cos_n = torch.abs((normals * to_g_norm).sum(dim=-1))  # (N,)

    # Rough antenna gain: cosine of angle from boresight
    cos_bore = (to_g_norm * radar_boresight.unsqueeze(0)).sum(dim=-1).clamp(min=0)

    # Amplitude estimate ~ opacity * cos_normal * cos_boresight / range^2
    A_est = opacities * cos_n * cos_bore / (dist ** 2 + 1e-8)
    return A_est


def select_active_set(
    contributions: Tensor,     # (N,)
    threshold: float = 0.97,
) -> Tensor:
    """
    Select the top-contributing Gaussians covering `threshold` fraction
    of total estimated amplitude.

    Returns boolean mask (N,) with True for active Gaussians.
    """
    sorted_vals, sorted_idx = torch.sort(contributions, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=0)
    total = cumsum[-1].clamp(min=1e-12)
    cutoff_idx = torch.searchsorted(cumsum, total * threshold)
    cutoff_idx = min(cutoff_idx.item() + 1, contributions.shape[0])

    mask = torch.zeros_like(contributions, dtype=torch.bool)
    mask[sorted_idx[:cutoff_idx]] = True
    return mask
```

---

### 10. `adc_synthesis.py`

The core rendering function. Vectorised per-Gaussian ADC contribution with exact phase.

```python
import torch
import math
from torch import Tensor
from .config import RadarConfig, C
from .gaussian_model import GaussianModel
from .reparameterization import reparameterize
from .bsdf_torch import eval_bsdf_physics_batch
from .antenna_torch import evaluate_tx_gain, evaluate_rx_gain


def synthesize_adc_single_bounce(
    model: GaussianModel,
    active_mask: Tensor,           # (N,) bool
    radar_cfg: RadarConfig,
    detach_phase: bool = True,
) -> tuple:  # (adc_real, adc_imag) each (N_tx, N_rx, K)
    """
    Render ADC from all active Gaussians (single bounce, exact phase).

    Phase computation:
      phi(t_k) = 2*pi * (f0 * tau + S * tau * t_k)
      where tau = (d_tx + d_rx) / c

    Amplitude:
      A = opacity * f_cos_bsdf * G_tx * G_rx / (d_tx * d_rx) * radar_const

    All loops are eliminated via broadcasting:
      Gaussians: (K_active, 1, 1, 1)
      TX:        (1, N_tx, 1, 1)
      RX:        (1, 1, N_rx, 1)
      ADC:       (1, 1, 1, K_adc)
    """
    device = model.device
    K = radar_cfg.num_adc_samples
    N_tx = radar_cfg.n_tx
    N_rx = radar_cfg.n_rx
    f0 = radar_cfg.center_freq
    S = radar_cfg.chirp_slope
    sample_rate = radar_cfg.sample_rate

    # --- Active Gaussian parameters ---
    mu = model.positions[active_mask]                      # (M, 3)
    normals = model.get_normals()[active_mask]              # (M, 3)
    opacities = model.get_opacities()[active_mask].squeeze(-1)  # (M,)
    raw_mat = model.raw_materials[active_mask]              # (M, 6)
    physics_mat = reparameterize(raw_mat)                   # (M, 6)
    M = mu.shape[0]

    # --- Sensor geometry (in metres) ---
    tx_pos = torch.from_numpy(radar_cfg.tx_positions_m).float().to(device)   # (N_tx, 3)
    rx_pos = torch.from_numpy(radar_cfg.rx_positions_m).float().to(device)   # (N_rx, 3)
    tx_bore = torch.from_numpy(radar_cfg.tx_boresights).float().to(device)   # (N_tx, 3)
    rx_bore = torch.from_numpy(radar_cfg.rx_boresights).float().to(device)   # (N_rx, 3)

    # --- Time grid ---
    # mmIR convention: t[k] = k / sample_rate (NOT linspace)
    t_grid = torch.arange(K, device=device, dtype=torch.float32) / sample_rate  # (K,)

    # --- Distances: (M, N_tx) and (M, N_rx) ---
    # mu: (M, 1, 3) - tx_pos: (1, N_tx, 3) -> diff: (M, N_tx, 3)
    d_tx = torch.norm(mu.unsqueeze(1) - tx_pos.unsqueeze(0), dim=-1)  # (M, N_tx)
    d_rx = torch.norm(mu.unsqueeze(1) - rx_pos.unsqueeze(0), dim=-1)  # (M, N_rx)

    # --- BSDF evaluation ---
    # For each (Gaussian, TX, RX) triple, evaluate BSDF.
    # This is the expensive part: M * N_tx * N_rx evaluations.
    # Optimisation: Factor into TX-dependent and RX-dependent parts.
    #
    # Simplification for initial implementation:
    # Evaluate BSDF once per Gaussian using the mean TX and mean RX direction.
    # This is an approximation that avoids M * N_tx * N_rx BSDF calls.
    # The per-TX-RX variation is captured by antenna gains and path loss.

    radar_center = (tx_pos.mean(dim=0) + rx_pos.mean(dim=0)) / 2  # (3,)
    wi_avg = torch.nn.functional.normalize(radar_center - mu, dim=-1)  # (M, 3) toward radar
    wo_avg = wi_avg  # monostatic approximation for BSDF (bistatic correction is small)

    bsdf_val = eval_bsdf_physics_batch(
        wo_avg, wi_avg, normals,
        physics_mat[:, 0], physics_mat[:, 1], physics_mat[:, 2],
        physics_mat[:, 3], physics_mat[:, 4], physics_mat[:, 5],
    )  # (M,)

    # --- Antenna gains ---
    # TX gain: direction from TX to Gaussian, per TX element
    # Approximate: evaluate at average TX position, then scale per element
    # Full per-TX evaluation:
    dir_tx = torch.nn.functional.normalize(
        mu.unsqueeze(1) - tx_pos.unsqueeze(0), dim=-1
    )  # (M, N_tx, 3)
    dir_rx = torch.nn.functional.normalize(
        rx_pos.unsqueeze(0) - mu.unsqueeze(1), dim=-1
    )  # (M, N_rx, 3)

    # Flatten for batch evaluation, then reshape
    G_tx_flat = evaluate_tx_gain(
        dir_tx.reshape(-1, 3),
        tx_bore.unsqueeze(0).expand(M, -1, -1).reshape(-1, 3),
    )  # (M * N_tx,)
    G_tx = G_tx_flat.reshape(M, N_tx)  # (M, N_tx)

    G_rx_flat = evaluate_rx_gain(
        dir_rx.reshape(-1, 3),
        rx_bore.unsqueeze(0).expand(M, -1, -1).reshape(-1, 3),
    )  # (M * N_rx,)
    G_rx = G_rx_flat.reshape(M, N_rx)  # (M, N_rx)

    # --- Amplitude ---
    # A = opacity * bsdf * G_tx * G_rx / (d_tx * d_rx)
    # Shape: (M, N_tx, N_rx)
    A = (opacities.unsqueeze(-1).unsqueeze(-1)                # (M, 1, 1)
         * bsdf_val.unsqueeze(-1).unsqueeze(-1)               # (M, 1, 1)
         * G_tx.unsqueeze(-1)                                 # (M, N_tx, 1)
         * G_rx.unsqueeze(-2)                                 # (M, 1, N_rx)
         / (d_tx.unsqueeze(-1) * d_rx.unsqueeze(-2) + 1e-10)) # (M, N_tx, N_rx)

    # --- Phase ---
    # R_tot = d_tx + d_rx  for each (Gaussian, TX, RX)
    # Shape: (M, N_tx, N_rx)
    R_tot = d_tx.unsqueeze(-1) + d_rx.unsqueeze(-2)  # (M, N_tx, N_rx)
    tau = R_tot / C                                    # (M, N_tx, N_rx)

    # phi(t_k) = 2*pi * (f0 * tau + S * tau * t_k)
    # = 2*pi * tau * (f0 + S * t_k)
    phi_const = 2 * math.pi * f0 * tau                # (M, N_tx, N_rx)
    phi_slope = 2 * math.pi * S * tau                  # (M, N_tx, N_rx)

    if detach_phase:
        phi_const = phi_const.detach()
        phi_slope = phi_slope.detach()

    # Expand for ADC samples: (M, N_tx, N_rx, K)
    phi = phi_const.unsqueeze(-1) + phi_slope.unsqueeze(-1) * t_grid  # (M, N_tx, N_rx, K)

    # --- Phasor scatter-add ---
    # cos/sin with amplitude weighting, sum over Gaussians
    A_expanded = A.unsqueeze(-1)  # (M, N_tx, N_rx, 1)

    adc_real = (A_expanded * torch.cos(phi)).sum(dim=0)  # (N_tx, N_rx, K)
    adc_imag = (A_expanded * torch.sin(phi)).sum(dim=0)  # (N_tx, N_rx, K)

    return adc_real, adc_imag
```

**Memory note**: The intermediate tensor `phi` has shape (M, N_tx, N_rx, K). For M=100K, N_tx=12, N_rx=16, K=256: 100K x 12 x 16 x 256 x 4 bytes = **188 GB**. This will NOT fit in memory.

**Solution: chunked accumulation.** Process Gaussians in chunks of size C (e.g. C=512):

```python
    # --- Chunked phasor accumulation ---
    adc_real = torch.zeros(N_tx, N_rx, K, device=device)
    adc_imag = torch.zeros(N_tx, N_rx, K, device=device)

    CHUNK = 512   # tune based on GPU memory
    for start in range(0, M, CHUNK):
        end = min(start + CHUNK, M)
        # Slice all per-Gaussian tensors to [start:end]
        A_chunk = A[start:end]                        # (C, N_tx, N_rx)
        tau_chunk = tau[start:end]                     # (C, N_tx, N_rx)

        phi_const_c = (2 * math.pi * f0 * tau_chunk).unsqueeze(-1)  # (C, N_tx, N_rx, 1)
        phi_slope_c = (2 * math.pi * S * tau_chunk).unsqueeze(-1)    # (C, N_tx, N_rx, 1)
        if detach_phase:
            phi_const_c = phi_const_c.detach()
            phi_slope_c = phi_slope_c.detach()

        phi_c = phi_const_c + phi_slope_c * t_grid     # (C, N_tx, N_rx, K)
        A_c = A_chunk.unsqueeze(-1)                     # (C, N_tx, N_rx, 1)

        adc_real += (A_c * torch.cos(phi_c)).sum(dim=0)
        adc_imag += (A_c * torch.sin(phi_c)).sum(dim=0)

    return adc_real, adc_imag
```

With C=512: 512 x 12 x 16 x 256 x 4 bytes = **965 MB**. Fits on a 24 GB GPU with room to spare.

---

### 11. `ray_surfel.py`

```python
import torch
from torch import Tensor


def intersect_rays_surfels(
    ray_origins: Tensor,       # (N_rays, 3)
    ray_dirs: Tensor,          # (N_rays, 3) unit vectors
    surfel_centers: Tensor,    # (N_surfels, 3)
    surfel_normals: Tensor,    # (N_surfels, 3)
    surfel_t1: Tensor,         # (N_surfels, 3) first tangent
    surfel_t2: Tensor,         # (N_surfels, 3) second tangent
    surfel_scales: Tensor,     # (N_surfels, 2) exp(log_scales)
    surfel_opacities: Tensor,  # (N_surfels,) in [0, 1]
    sigma_cutoff: float = 3.0,
) -> tuple:
    """
    Batch ray-surfel intersection (brute force).

    For each ray, finds the nearest surfel hit (smallest positive t_hit
    with alpha_hit > epsilon).

    Returns:
        hit_mask:  (N_rays,) bool - True if ray hit a surfel
        t_hit:     (N_rays,) float - distance along ray to hit
        hit_idx:   (N_rays,) long  - index of hit surfel
        hit_alpha: (N_rays,) float - opacity at hit point
        hit_uv:    (N_rays, 2) float - local coordinates on surfel

    Implementation processes in chunks over surfels to limit memory.
    """
    N_rays = ray_origins.shape[0]
    N_surfels = surfel_centers.shape[0]
    device = ray_origins.device

    best_t = torch.full((N_rays,), float('inf'), device=device)
    best_idx = torch.full((N_rays,), -1, dtype=torch.long, device=device)
    best_alpha = torch.zeros(N_rays, device=device)
    best_uv = torch.zeros(N_rays, 2, device=device)

    CHUNK = 4096
    for s_start in range(0, N_surfels, CHUNK):
        s_end = min(s_start + CHUNK, N_surfels)
        S = s_end - s_start

        mu = surfel_centers[s_start:s_end]       # (S, 3)
        n = surfel_normals[s_start:s_end]        # (S, 3)
        t1 = surfel_t1[s_start:s_end]            # (S, 3)
        t2 = surfel_t2[s_start:s_end]            # (S, 3)
        sc = surfel_scales[s_start:s_end]        # (S, 2)
        op = surfel_opacities[s_start:s_end]     # (S,)

        # Ray-plane intersection: t = dot(mu - o, n) / dot(d, n)
        # (N_rays, 1, 3) - (1, S, 3) -> (N_rays, S, 3)
        delta = mu.unsqueeze(0) - ray_origins.unsqueeze(1)  # (N_rays, S, 3)
        d_dot_n = (ray_dirs.unsqueeze(1) * n.unsqueeze(0)).sum(-1)  # (N_rays, S)
        delta_dot_n = (delta * n.unsqueeze(0)).sum(-1)              # (N_rays, S)

        t = delta_dot_n / (d_dot_n + 1e-12)  # (N_rays, S)

        # Intersection point in world space
        p_hit = ray_origins.unsqueeze(1) + t.unsqueeze(-1) * ray_dirs.unsqueeze(1)  # (N_rays, S, 3)

        # Project to surfel-local coords
        local = p_hit - mu.unsqueeze(0)                      # (N_rays, S, 3)
        u = (local * t1.unsqueeze(0)).sum(-1)                # (N_rays, S)
        v = (local * t2.unsqueeze(0)).sum(-1)                # (N_rays, S)

        # Gaussian weight
        u_norm = u / (sc[:, 0].unsqueeze(0) + 1e-12)
        v_norm = v / (sc[:, 1].unsqueeze(0) + 1e-12)
        w = torch.exp(-0.5 * (u_norm**2 + v_norm**2))       # (N_rays, S)

        alpha = op.unsqueeze(0) * w                           # (N_rays, S)

        # Valid hits: t > 0, within sigma cutoff, non-negligible opacity
        valid = (t > 1e-4) & (w > torch.exp(torch.tensor(-0.5 * sigma_cutoff**2))) & (alpha > 0.01)

        # Set invalid to inf so they don't win the argmin
        t_masked = torch.where(valid, t, torch.tensor(float('inf'), device=device))

        # Update best hits
        closer = t_masked < best_t.unsqueeze(1)               # (N_rays, S)
        # For each ray, find the closest valid surfel in this chunk
        chunk_best_t, chunk_best_local_idx = t_masked.min(dim=1)  # (N_rays,)
        improved = chunk_best_t < best_t

        chunk_best_global_idx = chunk_best_local_idx + s_start

        # Gather alpha and uv for the best hit in this chunk
        batch_idx = torch.arange(N_rays, device=device)
        chunk_alpha = alpha[batch_idx, chunk_best_local_idx]
        chunk_u = u[batch_idx, chunk_best_local_idx]
        chunk_v = v[batch_idx, chunk_best_local_idx]

        best_t = torch.where(improved, chunk_best_t, best_t)
        best_idx = torch.where(improved, chunk_best_global_idx, best_idx)
        best_alpha = torch.where(improved, chunk_alpha, best_alpha)
        best_uv[:, 0] = torch.where(improved, chunk_u, best_uv[:, 0])
        best_uv[:, 1] = torch.where(improved, chunk_v, best_uv[:, 1])

    hit_mask = best_idx >= 0
    return hit_mask, best_t, best_idx, best_alpha, best_uv
```

---

### 12. `multibounce.py`

```python
import torch
import math
from torch import Tensor
from .config import RadarConfig, C
from .gaussian_model import GaussianModel
from .ray_surfel import intersect_rays_surfels
from .reparameterization import reparameterize
from .bsdf_torch import eval_bsdf_physics_batch
from .antenna_torch import evaluate_tx_gain, evaluate_rx_gain


def synthesize_multibounce(
    model: GaussianModel,
    first_bounce_amplitudes: Tensor,  # (N,) from single-bounce
    radar_cfg: RadarConfig,
    n_first_bounce: int = 3000,
    n_secondary_rays: int = 8,
    detach_phase: bool = True,
) -> tuple:  # (adc_real, adc_imag) each (N_tx, N_rx, K)
    """
    Two-bounce ADC contributions via ray-surfel tracing.

    1. Importance-sample first-bounce Gaussians by amplitude.
    2. Sample outgoing directions from cosine hemisphere at each.
    3. Ray-surfel intersect to find second-bounce Gaussians.
    4. Compute two-bounce ADC contribution.
    """
    device = model.device
    N = model.N
    K = radar_cfg.num_adc_samples
    N_tx = radar_cfg.n_tx
    N_rx = radar_cfg.n_rx
    f0 = radar_cfg.center_freq
    S = radar_cfg.chirp_slope
    sample_rate = radar_cfg.sample_rate

    tx_pos = torch.from_numpy(radar_cfg.tx_positions_m).float().to(device)
    rx_pos = torch.from_numpy(radar_cfg.rx_positions_m).float().to(device)

    t_grid = torch.arange(K, device=device, dtype=torch.float32) / sample_rate

    # --- 1. Importance-sample first-bounce Gaussians ---
    probs = first_bounce_amplitudes.clamp(min=0)
    probs = probs / (probs.sum() + 1e-12)
    g1_indices = torch.multinomial(probs, n_first_bounce, replacement=True)  # (n_first_bounce,)
    g1_pdf = probs[g1_indices]  # (n_first_bounce,)

    mu_g1 = model.positions[g1_indices]         # (B, 3) where B = n_first_bounce
    n_g1 = model.get_normals()[g1_indices]       # (B, 3)

    # --- 2. Sample outgoing directions (cosine hemisphere around normal) ---
    B = n_first_bounce
    R_total = B * n_secondary_rays  # total rays

    # Repeat each g1 for n_secondary_rays
    mu_g1_rep = mu_g1.repeat_interleave(n_secondary_rays, dim=0)  # (R_total, 3)
    n_g1_rep = n_g1.repeat_interleave(n_secondary_rays, dim=0)    # (R_total, 3)
    g1_idx_rep = g1_indices.repeat_interleave(n_secondary_rays)    # (R_total,)
    g1_pdf_rep = g1_pdf.repeat_interleave(n_secondary_rays)        # (R_total,)

    ray_dirs = _sample_cosine_hemisphere(n_g1_rep)                 # (R_total, 3)

    # --- 3. Ray-surfel intersection ---
    t1_all, t2_all, n_all = model.get_tangent_frame()
    hit_mask, t_hit, hit_idx, hit_alpha, _ = intersect_rays_surfels(
        mu_g1_rep, ray_dirs,
        model.positions.detach(),
        n_all.detach(),
        t1_all.detach(),
        t2_all.detach(),
        model.get_scales().detach(),
        model.get_opacities().squeeze(-1).detach(),
    )

    if hit_mask.sum() == 0:
        return (torch.zeros(N_tx, N_rx, K, device=device),
                torch.zeros(N_tx, N_rx, K, device=device))

    # --- 4. Compute two-bounce ADC ---
    # Filter to valid hits
    valid = hit_mask.nonzero(as_tuple=True)[0]
    mu_g1_v = mu_g1_rep[valid]               # (V, 3)
    mu_g2_v = model.positions[hit_idx[valid]]  # (V, 3)
    n_g2_v = model.get_normals()[hit_idx[valid]]
    alpha_g2_v = hit_alpha[valid]

    # BSDF at second bounce (simplified: use average radar direction)
    radar_center = (tx_pos.mean(0) + rx_pos.mean(0)) / 2
    wi_g2 = torch.nn.functional.normalize(mu_g1_v - mu_g2_v, dim=-1)
    wo_g2 = torch.nn.functional.normalize(radar_center - mu_g2_v, dim=-1)

    mat_g2 = reparameterize(model.raw_materials[hit_idx[valid]])
    bsdf_g2 = eval_bsdf_physics_batch(
        wo_g2, wi_g2, n_g2_v,
        mat_g2[:, 0], mat_g2[:, 1], mat_g2[:, 2],
        mat_g2[:, 3], mat_g2[:, 4], mat_g2[:, 5],
    )

    # MC weight: 1 / (pdf_g1 * n_secondary_rays * pdf_cosine)
    # pdf_cosine = cos(theta) / pi, but we absorb it into the BSDF (eval_f_cos)
    mc_weight = 1.0 / (g1_pdf_rep[valid] * n_secondary_rays + 1e-12)

    # Two-bounce path distances and ADC contribution
    # Chunked to manage memory (similar to single-bounce)
    V = valid.shape[0]
    adc_real = torch.zeros(N_tx, N_rx, K, device=device)
    adc_imag = torch.zeros(N_tx, N_rx, K, device=device)

    CHUNK = 256
    for start in range(0, V, CHUNK):
        end = min(start + CHUNK, V)
        c = end - start

        # Distances: TX -> g1 -> g2 -> RX
        d_tx_g1 = torch.norm(
            mu_g1_v[start:end].unsqueeze(1) - tx_pos.unsqueeze(0), dim=-1
        )  # (c, N_tx)
        d_g1_g2 = torch.norm(
            mu_g1_v[start:end] - mu_g2_v[start:end], dim=-1
        )  # (c,)
        d_g2_rx = torch.norm(
            mu_g2_v[start:end].unsqueeze(1) - rx_pos.unsqueeze(0), dim=-1
        )  # (c, N_rx)

        R_tot = (d_tx_g1.unsqueeze(-1)                  # (c, N_tx, 1)
                 + d_g1_g2.unsqueeze(-1).unsqueeze(-1)  # (c, 1, 1)
                 + d_g2_rx.unsqueeze(-2))                # (c, 1, N_rx)
        # R_tot shape: (c, N_tx, N_rx)

        tau = R_tot / C
        phi_const = (2 * math.pi * f0 * tau).unsqueeze(-1)
        phi_slope = (2 * math.pi * S * tau).unsqueeze(-1)
        if detach_phase:
            phi_const = phi_const.detach()
            phi_slope = phi_slope.detach()

        phi = phi_const + phi_slope * t_grid  # (c, N_tx, N_rx, K)

        # Amplitude (simplified: omit per-TX-RX antenna gain for speed)
        A = (alpha_g2_v[start:end] * bsdf_g2[start:end]
             * mc_weight[start:end]
             / (d_g1_g2[start:end]**2 + 1e-10))  # (c,)
        A = A.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (c, 1, 1, 1)

        adc_real += (A * torch.cos(phi)).sum(dim=0)
        adc_imag += (A * torch.sin(phi)).sum(dim=0)

    return adc_real, adc_imag


def _sample_cosine_hemisphere(normals: Tensor) -> Tensor:
    """Sample directions from cosine-weighted hemisphere around each normal.

    Args:
        normals: (N, 3) unit normals.
    Returns:
        (N, 3) unit direction vectors.
    """
    N = normals.shape[0]
    device = normals.device

    # Sample on unit disk
    u1 = torch.rand(N, device=device)
    u2 = torch.rand(N, device=device)
    r = torch.sqrt(u1)
    phi = 2 * math.pi * u2
    x_local = r * torch.cos(phi)
    y_local = r * torch.sin(phi)
    z_local = torch.sqrt((1 - u1).clamp(min=0))

    # Build local frame from normal
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, -1)
    # Handle degenerate case where normal ~ up
    alt = torch.tensor([1.0, 0.0, 0.0], device=device).expand(N, -1)
    parallel = (torch.abs((normals * up).sum(-1)) > 0.999)
    ref = torch.where(parallel.unsqueeze(-1), alt, up)

    t1 = torch.nn.functional.normalize(torch.cross(normals, ref, dim=-1), dim=-1)
    t2 = torch.cross(normals, t1, dim=-1)

    # Transform to world
    dirs = x_local.unsqueeze(-1) * t1 + y_local.unsqueeze(-1) * t2 + z_local.unsqueeze(-1) * normals
    return torch.nn.functional.normalize(dirs, dim=-1)
```

---

### 13. `losses.py`

```python
import torch
from torch import Tensor
from typing import Dict, Tuple

# Import the FFT pipeline from mmIR
from mmir.data.ra_utils import adc_to_ra_complex


def compute_loss(
    adc_real: Tensor,          # (N_tx, N_rx, K) rendered
    adc_imag: Tensor,          # (N_tx, N_rx, K) rendered
    gt_adc_ri: Tensor,         # (N_tx, N_rx, K, 2) ground truth real/imag
    w_ra_mag: float = 1.0,
    w_adc_mag: float = 0.0,
    w_phase: float = 0.0,
    ra_use_log: bool = True,
    log_epsilon: float = 1e-6,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Compute loss between rendered and ground truth ADC.

    Pipeline:
      1. Stack rendered into (N_tx, N_rx, K, 2) format.
      2. Run identical FFT pipeline (adc_to_ra_complex) on both.
      3. Compute RA magnitude loss (primary).
      4. Optionally: ADC magnitude loss, phase loss.

    Returns (total_loss, loss_dict).
    """
    device = adc_real.device

    # --- Assemble rendered ADC in mmIR format ---
    rendered_ri = torch.stack([adc_real, adc_imag], dim=-1)  # (N_tx, N_rx, K, 2)

    # --- FFT to RA domain (differentiable) ---
    ra_rendered = adc_to_ra_complex(rendered_ri)   # (127, 256) complex
    ra_gt = adc_to_ra_complex(gt_adc_ri)           # (127, 256) complex

    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)

    # --- RA magnitude loss ---
    if w_ra_mag > 0:
        ra_rend_mag = torch.abs(ra_rendered)
        ra_gt_mag = torch.abs(ra_gt)

        if ra_use_log:
            ra_rend_mag = torch.log(log_epsilon + ra_rend_mag)
            ra_gt_mag = torch.log(log_epsilon + ra_gt_mag)

        # Independent min-max normalization
        ra_rend_norm = _minmax_normalize(ra_rend_mag)
        ra_gt_norm = _minmax_normalize(ra_gt_mag)

        ra_loss = torch.mean((ra_rend_norm - ra_gt_norm) ** 2)
        total_loss = total_loss + w_ra_mag * ra_loss
        loss_dict['ra_mag'] = ra_loss.item()

    # --- ADC magnitude loss ---
    if w_adc_mag > 0:
        adc_rend_mag = torch.sqrt(adc_real**2 + adc_imag**2)
        adc_gt_mag = torch.sqrt(gt_adc_ri[..., 0]**2 + gt_adc_ri[..., 1]**2)
        adc_rend_norm = _minmax_normalize(adc_rend_mag)
        adc_gt_norm = _minmax_normalize(adc_gt_mag)
        adc_loss = torch.mean((adc_rend_norm - adc_gt_norm) ** 2)
        total_loss = total_loss + w_adc_mag * adc_loss
        loss_dict['adc_mag'] = adc_loss.item()

    # --- Phase loss (unit phasor, weighted by GT magnitude) ---
    if w_phase > 0:
        ra_rend_unit = ra_rendered / (torch.abs(ra_rendered) + 1e-10)
        ra_gt_unit = ra_gt / (torch.abs(ra_gt) + 1e-10)
        # Weight by GT magnitude (only penalise phase at high-SNR bins)
        weights = torch.abs(ra_gt)
        weights = weights / (weights.max() + 1e-10)
        phase_err = torch.abs(ra_rend_unit - ra_gt_unit) ** 2
        phase_loss = (weights * phase_err).mean()
        total_loss = total_loss + w_phase * phase_loss
        loss_dict['phase'] = phase_loss.item()

    loss_dict['total'] = total_loss.item()
    return total_loss, loss_dict


def _minmax_normalize(x: Tensor) -> Tensor:
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-30:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)
```

---

### 14. `optimizer.py`

```python
import torch
from torch import Tensor
from typing import Dict, List
from .gaussian_model import GaussianModel
from .config import TrainingConfig
import numpy as np


class PerGroupAdam:
    """
    Adam optimizer with per-parameter-group learning rates and RMS gradient clipping.

    Mirrors mmIR's SimpleAdamSionna but operates on PyTorch parameters.
    """

    def __init__(self, model: GaussianModel, cfg: TrainingConfig):
        self.cfg = cfg
        self.groups = self._build_groups(model, cfg)
        self.step_count = 0

    def _build_groups(self, model, cfg):
        """Create parameter groups with per-group LR and clip."""
        mat_lr_scales = torch.tensor(cfg.material_lr_scales, device=model.device)

        return [
            {'name': 'positions',  'params': [model.positions],
             'lr': cfg.lr_positions, 'clip': cfg.clip_positions},
            {'name': 'rotations',  'params': [model.rotations],
             'lr': cfg.lr_rotations, 'clip': cfg.clip_rotations},
            {'name': 'scales',     'params': [model.log_scales],
             'lr': cfg.lr_scales, 'clip': cfg.clip_scales},
            {'name': 'opacities',  'params': [model.logit_opacities],
             'lr': cfg.lr_opacities, 'clip': cfg.clip_opacities},
            {'name': 'materials',  'params': [model.raw_materials],
             'lr': cfg.lr_materials, 'clip': cfg.clip_materials,
             'lr_scales': mat_lr_scales},
        ]

    def setup(self):
        """Initialize Adam state (m, v) for all parameters."""
        for group in self.groups:
            group['m'] = [torch.zeros_like(p) for p in group['params']]
            group['v'] = [torch.zeros_like(p) for p in group['params']]

    def step(self, lr_scale: float = 1.0, beta1: float = 0.9,
             beta2: float = 0.999, eps: float = 1e-8):
        """One Adam step with per-group clipping."""
        self.step_count += 1
        t = self.step_count

        for group in self.groups:
            lr = group['lr'] * lr_scale
            clip = group['clip']
            lr_scales = group.get('lr_scales', None)

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                g = p.grad.data

                # NaN safety
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

                # RMS gradient clipping
                rms = torch.sqrt(torch.mean(g ** 2))
                if rms > clip:
                    g = g * (clip / rms)

                # Per-column LR scaling (for materials)
                if lr_scales is not None and g.dim() == 2:
                    g = g * lr_scales.unsqueeze(0)

                # Adam update
                m = group['m'][i]
                v = group['v'][i]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                p.data.add_(m_hat / (torch.sqrt(v_hat) + eps), alpha=-lr)

    def zero_grad(self):
        for group in self.groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
```

---

### 15. `density_control.py`

```python
import torch
from torch import Tensor
from .gaussian_model import GaussianModel
from .config import TrainingConfig


def densify_and_prune(
    model: GaussianModel,
    grad_accum: Tensor,       # (N,) accumulated position gradient magnitudes
    grad_count: Tensor,       # (N,) number of gradient accumulations
    cfg: TrainingConfig,
    radar_center: Tensor,     # (3,)
) -> GaussianModel:
    """
    Adaptive density control: clone small high-grad Gaussians, split large
    high-grad ones, prune low-opacity and out-of-range ones.

    Returns a new GaussianModel with updated count.
    """
    N = model.N
    device = model.device

    avg_grad = grad_accum / (grad_count.clamp(min=1))
    scales = model.get_scales()         # (N, 2)
    max_scale = scales.max(dim=-1).values   # (N,)
    opacities = model.get_opacities().squeeze(-1)  # (N,)

    # --- Ranges for pruning ---
    dist = torch.norm(model.positions - radar_center, dim=-1)

    # --- Clone candidates: high gradient, small scale ---
    clone_mask = (avg_grad > cfg.densify_grad_threshold) & (max_scale < 0.02)

    # --- Split candidates: high gradient, large scale ---
    split_mask = (avg_grad > cfg.densify_grad_threshold) & (max_scale >= 0.02)

    # --- Prune candidates ---
    prune_mask = ((opacities < cfg.prune_opacity_threshold)
                  | (dist < cfg.prune_range_min)
                  | (dist > cfg.prune_range_max))

    # Build index lists
    clone_idx = clone_mask.nonzero(as_tuple=True)[0]
    split_idx = split_mask.nonzero(as_tuple=True)[0]
    keep_mask = ~prune_mask & ~split_mask  # keep originals minus splits and prunes

    # --- Clone: duplicate with small position offset ---
    N_clone = clone_idx.shape[0]
    clone_pos = model.positions[clone_idx].detach().clone()
    clone_pos += torch.randn_like(clone_pos) * 0.001  # 1mm offset

    # --- Split: replace with two half-scale Gaussians ---
    N_split = split_idx.shape[0]
    split_pos_1 = model.positions[split_idx].detach().clone()
    split_pos_2 = model.positions[split_idx].detach().clone()
    offset = model.get_scales()[split_idx].max(dim=-1).values.unsqueeze(-1)
    normals = model.get_normals()[split_idx]
    t1, _, _ = model.get_tangent_frame()
    t1_split = t1[split_idx]
    split_pos_1 += t1_split * offset * 0.5
    split_pos_2 -= t1_split * offset * 0.5

    # --- Assemble new model ---
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    N_new = keep_idx.shape[0] + N_clone + 2 * N_split

    new_model = GaussianModel(N_new, device=device)
    with torch.no_grad():
        # Copy kept Gaussians
        offset_ptr = 0
        n_keep = keep_idx.shape[0]
        new_model.positions.data[:n_keep] = model.positions[keep_idx]
        new_model.rotations.data[:n_keep] = model.rotations[keep_idx]
        new_model.log_scales.data[:n_keep] = model.log_scales[keep_idx]
        new_model.logit_opacities.data[:n_keep] = model.logit_opacities[keep_idx]
        new_model.raw_materials.data[:n_keep] = model.raw_materials[keep_idx]
        offset_ptr = n_keep

        # Add clones
        if N_clone > 0:
            new_model.positions.data[offset_ptr:offset_ptr + N_clone] = clone_pos
            new_model.rotations.data[offset_ptr:offset_ptr + N_clone] = model.rotations[clone_idx]
            new_model.log_scales.data[offset_ptr:offset_ptr + N_clone] = model.log_scales[clone_idx]
            new_model.logit_opacities.data[offset_ptr:offset_ptr + N_clone] = model.logit_opacities[clone_idx]
            new_model.raw_materials.data[offset_ptr:offset_ptr + N_clone] = model.raw_materials[clone_idx]
            offset_ptr += N_clone

        # Add splits (two per original, half scale)
        if N_split > 0:
            log_scale_half = model.log_scales[split_idx] - 0.693  # log(0.5)
            for j, pos in enumerate([split_pos_1, split_pos_2]):
                new_model.positions.data[offset_ptr:offset_ptr + N_split] = pos
                new_model.rotations.data[offset_ptr:offset_ptr + N_split] = model.rotations[split_idx]
                new_model.log_scales.data[offset_ptr:offset_ptr + N_split] = log_scale_half
                new_model.logit_opacities.data[offset_ptr:offset_ptr + N_split] = model.logit_opacities[split_idx]
                new_model.raw_materials.data[offset_ptr:offset_ptr + N_split] = model.raw_materials[split_idx]
                offset_ptr += N_split

    return new_model


def reset_opacities(model: GaussianModel):
    """Reset all opacities to sigmoid^{-1}(0.5) = 0."""
    with torch.no_grad():
        model.logit_opacities.fill_(0.0)
```

---

### 16. `training.py`

```python
import torch
import numpy as np
import os
import json
import time
from typing import Optional

from .config import RadarConfig, TrainingConfig, C
from .gaussian_model import GaussianModel
from .initialization import initialize_from_lidar
from .reparameterization import reparameterize
from .rae_grid import RAEGrid
from .culling import compute_contribution_estimates, select_active_set
from .adc_synthesis import synthesize_adc_single_bounce
from .multibounce import synthesize_multibounce
from .losses import compute_loss
from .optimizer import PerGroupAdam
from .density_control import densify_and_prune, reset_opacities
from .antenna_torch import load_patterns

# Data loading from mmIR
from mmir.data.data_utils import load_target


def train(cfg: TrainingConfig):
    """Main training entry point."""
    device = cfg.device
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load radar config ---
    radar_cfg = RadarConfig.from_json(cfg.config_path)

    # --- Load antenna patterns ---
    load_patterns(cfg.tx_pattern_path, cfg.rx_pattern_path)

    # --- Load ground truth ADC ---
    gt_adc_path = _find_gt_adc(cfg.scene_dir, "cascaded")
    gt_adc_ri, norm_stats = load_target(gt_adc_path, device, normalize=True)
    # gt_adc_ri shape: (N_tx, N_rx, K, 2)

    # --- Initialise Gaussians ---
    model = initialize_from_lidar(
        cfg.pcl_path, device,
        k_neighbors=cfg.pca_k_neighbors,
        scale_clamp_min=cfg.initial_scale_clamp_min,
        scale_clamp_max=cfg.initial_scale_clamp_max,
        initial_material=cfg.initial_material,
    )
    print(f"Initialised {model.N} Gaussians from LiDAR")

    # --- RAE grid (for diagnostics / future splatting) ---
    rae_grid = RAEGrid(radar_cfg, device=device)

    # --- Optimizer ---
    optimizer = PerGroupAdam(model, cfg)
    optimizer.setup()

    # --- Radar geometry ---
    radar_center = torch.from_numpy(
        (radar_cfg.tx_positions_m.mean(0) + radar_cfg.rx_positions_m.mean(0)) / 2
    ).float().to(device)
    radar_boresight = torch.from_numpy(
        radar_cfg.tx_boresights.mean(0)
    ).float().to(device)
    radar_boresight = torch.nn.functional.normalize(radar_boresight, dim=0)

    # --- Training history ---
    history = {'iteration': [], 'total_loss': [], 'ra_mag_loss': [],
               'n_active': [], 'n_total': []}
    best_loss = float('inf')
    best_state = None

    # --- Gradient accumulation for density control ---
    grad_accum = torch.zeros(model.N, device=device)
    grad_count = torch.zeros(model.N, device=device)

    # --- Training loop ---
    t_start = time.time()
    for it in range(cfg.max_iterations):
        optimizer.zero_grad()

        # LR warmup (0.3x -> 1x over first 5 iterations) + exponential decay
        if it < 5:
            lr_scale = 0.3 + 0.7 * (it / 5)
        else:
            # Exponential decay for position LR
            decay = (cfg.lr_positions_final / cfg.lr_positions) ** (
                it / cfg.max_iterations)
            lr_scale = decay

        # --- Contribution culling ---
        with torch.no_grad():
            contributions = compute_contribution_estimates(
                model, radar_center, radar_boresight)
            if it % cfg.culling_full_inclusion_interval == 0:
                active_mask = torch.ones(model.N, dtype=torch.bool, device=device)
            else:
                active_mask = select_active_set(
                    contributions, cfg.culling_threshold)

        n_active = active_mask.sum().item()

        # --- Forward: single-bounce ADC ---
        adc_real, adc_imag = synthesize_adc_single_bounce(
            model, active_mask, radar_cfg, detach_phase=True)

        # --- Forward: multi-bounce (optional) ---
        if cfg.enable_multibounce and it >= cfg.multibounce_warmup:
            with torch.no_grad():
                fb_amplitudes = contributions.clone()
            adc_mb_real, adc_mb_imag = synthesize_multibounce(
                model, fb_amplitudes, radar_cfg,
                n_first_bounce=cfg.n_first_bounce_samples,
                n_secondary_rays=cfg.n_secondary_rays,
                detach_phase=True,
            )
            adc_real = adc_real + adc_mb_real
            adc_imag = adc_imag + adc_mb_imag

        # --- Loss ---
        loss, loss_dict = compute_loss(
            adc_real, adc_imag, gt_adc_ri,
            w_ra_mag=cfg.ra_mag_weight,
            w_adc_mag=cfg.adc_mag_weight,
            w_phase=cfg.phase_weight,
            ra_use_log=cfg.ra_use_log,
            log_epsilon=cfg.log_epsilon,
        )

        # --- Backward ---
        loss.backward()

        # --- Accumulate position gradients for density control ---
        if model.positions.grad is not None:
            pos_grad_mag = model.positions.grad.norm(dim=-1)  # (N,)
            grad_accum[:model.N] += pos_grad_mag
            grad_count[:model.N] += 1

        # --- Optimizer step ---
        optimizer.step(lr_scale=lr_scale)

        # --- Post-step: clamp material raw params ---
        with torch.no_grad():
            model.raw_materials[:, 1].clamp_(-7.0, 4.6)
            model.raw_materials[:, 2].clamp_(-16.0, -2.3)
            model.raw_materials[:, 5].clamp_(-7.0, -0.7)

        # --- Density control ---
        if (it + 1) % cfg.densify_interval == 0 and it < cfg.max_iterations - 100:
            model = densify_and_prune(
                model, grad_accum[:model.N], grad_count[:model.N],
                cfg, radar_center)
            # Rebuild optimizer for new model
            optimizer = PerGroupAdam(model, cfg)
            optimizer.setup()
            grad_accum = torch.zeros(model.N, device=device)
            grad_count = torch.zeros(model.N, device=device)

        if (it + 1) % cfg.opacity_reset_interval == 0:
            reset_opacities(model)

        # --- Logging ---
        history['iteration'].append(it)
        history['total_loss'].append(loss_dict['total'])
        history['ra_mag_loss'].append(loss_dict.get('ra_mag', 0))
        history['n_active'].append(n_active)
        history['n_total'].append(model.N)

        if loss_dict['total'] < best_loss:
            best_loss = loss_dict['total']
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (it + 1) % cfg.log_interval == 0:
            elapsed = time.time() - t_start
            print(f"[{it+1}/{cfg.max_iterations}] loss={loss_dict['total']:.6f} "
                  f"ra={loss_dict.get('ra_mag', 0):.6f} "
                  f"active={n_active}/{model.N} "
                  f"time={elapsed:.1f}s")

        # --- Checkpoint ---
        if (it + 1) % cfg.checkpoint_interval == 0:
            model.save(os.path.join(cfg.output_dir, f"checkpoint_{it+1}.pt"))

    # --- Save best model ---
    if best_state is not None:
        best_path = os.path.join(cfg.output_dir, "best_model.pt")
        torch.save(best_state, best_path)
        print(f"Best model saved to {best_path} (loss={best_loss:.6f})")

    # Save history
    with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Save best materials in mmIR-compatible format (for eval adapter)
    if best_state is not None:
        model.load_state_dict(best_state)
        physics_mat = reparameterize(model.raw_materials).detach().cpu().numpy()
        np.savez(os.path.join(cfg.output_dir, "best_materials.npz"),
                 materials=physics_mat)

    return model, history


def _find_gt_adc(scene_dir: str, sensor: str) -> str:
    """Find the ground truth ADC file in the scene directory."""
    import glob
    pattern = os.path.join(scene_dir, "radar", f"{sensor}_frame_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No GT ADC files matching {pattern}")
    return files[0]
```

---

### 17. `eval_adapter.py`

```python
"""
Adapter for mmir/evaluation/ pipeline.

Provides a renderer-like interface that the existing evaluation scripts
can call without modification.
"""
import numpy as np
import torch
from typing import Optional

from .config import RadarConfig, TrainingConfig
from .gaussian_model import GaussianModel
from .adc_synthesis import synthesize_adc_single_bounce
from .culling import compute_contribution_estimates, select_active_set
from .antenna_torch import load_patterns


class GaussianRendererWrapper:
    """
    Drop-in replacement for mmir.evaluation.renderer_wrapper.RendererWrapper.

    Provides render_forward() that returns ADC in the same format:
      (N_tx, N_rx, K, 2) float32 [real, imag].
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        tx_pattern_path: str = "assets/antenna_pattern/MMWCAS/tx1_76.npy",
        rx_pattern_path: str = "assets/antenna_pattern/MMWCAS/rx1_76.npy",
        device: str = "cuda:0",
    ):
        self.device = device
        self.radar_cfg = RadarConfig.from_json(config_path)
        load_patterns(tx_pattern_path, rx_pattern_path)

        # Load model
        self.model = GaussianModel(1, device=device)  # placeholder
        ckpt = torch.load(model_path, map_location=device)
        N = ckpt['positions'].shape[0]
        self.model = GaussianModel(N, device=device)
        self.model.load(model_path)
        self.model.eval()

        # Precompute radar geometry
        self.radar_center = torch.from_numpy(
            (self.radar_cfg.tx_positions_m.mean(0)
             + self.radar_cfg.rx_positions_m.mean(0)) / 2
        ).float().to(device)
        self.radar_boresight = torch.from_numpy(
            self.radar_cfg.tx_boresights.mean(0)
        ).float().to(device)
        self.radar_boresight = torch.nn.functional.normalize(
            self.radar_boresight, dim=0)

    def render_forward(self, seed: int = 42) -> np.ndarray:
        """Render ADC. Returns (N_tx, N_rx, K, 2) float32."""
        torch.manual_seed(seed)
        with torch.no_grad():
            contributions = compute_contribution_estimates(
                self.model, self.radar_center, self.radar_boresight)
            active_mask = select_active_set(contributions, threshold=0.99)

            adc_real, adc_imag = synthesize_adc_single_bounce(
                self.model, active_mask, self.radar_cfg, detach_phase=True)

        adc_ri = torch.stack([adc_real, adc_imag], dim=-1)
        return adc_ri.cpu().numpy().astype(np.float32)

    def update_antenna_config(self, config_path: str):
        """Swap radar config (for cross-sensor transfer evaluation)."""
        self.radar_cfg = RadarConfig.from_json(config_path)
        self.radar_center = torch.from_numpy(
            (self.radar_cfg.tx_positions_m.mean(0)
             + self.radar_cfg.rx_positions_m.mean(0)) / 2
        ).float().to(self.device)
        self.radar_boresight = torch.from_numpy(
            self.radar_cfg.tx_boresights.mean(0)
        ).float().to(self.device)
        self.radar_boresight = torch.nn.functional.normalize(
            self.radar_boresight, dim=0)
```

---

### 18. `train_cli.py`

```python
"""
CLI entry point: python -m mm25DGS.train_cli --config path/to/config.json

Or with explicit arguments:
  python -m mm25DGS.train_cli \
    --scene_dir data/seq_0_frame_135 \
    --config_path data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json \
    --pcl_path data/seq_0_frame_135/scene/pcl.npy \
    --output_dir output/mm25dgs/seq_0_frame_135
"""
import argparse
import json
from .config import TrainingConfig
from .training import train


def main():
    parser = argparse.ArgumentParser(description="mm25DGS Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON training config")
    parser.add_argument("--scene_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--pcl_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Build config
    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = TrainingConfig(**cfg_dict)
    else:
        cfg = TrainingConfig()

    # Override from CLI
    for key in ['scene_dir', 'config_path', 'pcl_path',
                'output_dir', 'max_iterations', 'device']:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)

    # Train
    model, history = train(cfg)
    print("Training complete.")


if __name__ == "__main__":
    main()
```

---

## Integration Test Plan

### Test 1: Initialisation smoke test
```bash
python -c "
from mm25DGS.initialization import initialize_from_lidar
model = initialize_from_lidar('data/seq_0_frame_135/scene/pcl.npy')
print(f'N={model.N}, positions={model.positions.shape}, normals={model.get_normals().shape}')
print(f'Scales range: {model.get_scales().min():.4f} - {model.get_scales().max():.4f}')
"
```

### Test 2: Forward pass (untrained)
```bash
python -c "
from mm25DGS.config import RadarConfig
from mm25DGS.initialization import initialize_from_lidar
from mm25DGS.culling import compute_contribution_estimates, select_active_set
from mm25DGS.adc_synthesis import synthesize_adc_single_bounce
from mm25DGS.antenna_torch import load_patterns
import torch

load_patterns('assets/antenna_pattern/MMWCAS/tx1_76.npy',
              'assets/antenna_pattern/MMWCAS/rx1_76.npy')

radar_cfg = RadarConfig.from_json(
    'data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json')
model = initialize_from_lidar('data/seq_0_frame_135/scene/pcl.npy')

rc = torch.from_numpy((radar_cfg.tx_positions_m.mean(0)+radar_cfg.rx_positions_m.mean(0))/2).float().cuda()
rb = torch.from_numpy(radar_cfg.tx_boresights.mean(0)).float().cuda()
rb = torch.nn.functional.normalize(rb, dim=0)

contribs = compute_contribution_estimates(model, rc, rb)
mask = select_active_set(contribs, 0.97)
print(f'Active: {mask.sum()}/{model.N}')

adc_r, adc_i = synthesize_adc_single_bounce(model, mask, radar_cfg)
print(f'ADC shape: {adc_r.shape}')  # expect (12, 16, 256)
print(f'ADC magnitude range: {torch.sqrt(adc_r**2+adc_i**2).max():.6f}')
"
```

### Test 3: Full training (1 scene, 50 iterations)
```bash
python -m mm25DGS.train_cli \
  --scene_dir data/seq_0_frame_135 \
  --config_path data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json \
  --pcl_path data/seq_0_frame_135/scene/pcl.npy \
  --output_dir output/mm25dgs_test/seq_0_frame_135 \
  --max_iterations 50
```

### Test 4: Evaluation
```bash
python -c "
from mm25DGS.eval_adapter import GaussianRendererWrapper
wrapper = GaussianRendererWrapper(
    'output/mm25dgs_test/seq_0_frame_135/best_model.pt',
    'data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json')
adc = wrapper.render_forward()
print(f'ADC output shape: {adc.shape}')  # (12, 16, 256, 2)
"
```

---

## Implementation Order

| Step | File(s) | Depends on | Verification |
|------|---------|------------|-------------|
| 1 | `config.py` | None | Parse a JSON config, print derived params |
| 2 | `reparameterization.py` | None | Round-trip: inverse(forward(x)) == x |
| 3 | `gaussian_model.py` | None | Create model, verify shapes, save/load |
| 4 | `initialization.py` | 2, 3 | Test 1: init from LiDAR, check shapes |
| 5 | `rae_grid.py` | 1 | Convert known (x,y,z) -> (r,az,el), verify |
| 6 | `bsdf_torch.py` | None (mmIR) | Eval BSDF at known geometry, compare to mmIR |
| 7 | `antenna_torch.py` | None (mmIR) | Eval gain at boresight, verify ~peak value |
| 8 | `culling.py` | 3 | Verify top-97% selects reasonable count |
| 9 | `adc_synthesis.py` | 1-8 | Test 2: forward pass, non-zero ADC output |
| 10 | `losses.py` | 9, mmIR | Compute loss between rendered and GT ADC |
| 11 | `optimizer.py` | 3 | One step, verify params changed |
| 12 | `density_control.py` | 3 | Clone/split/prune, verify N changes |
| 13 | `training.py` | 1-12 | Test 3: 50 iterations, loss decreases |
| 14 | `ray_surfel.py` | 3 | Trace ray at known surfel, verify hit |
| 15 | `multibounce.py` | 14, 6, 7 | Add to training, verify extra ADC signal |
| 16 | `eval_adapter.py` | 9, mmIR eval | Test 4: render + metrics match format |
| 17 | `train_cli.py` | 13 | Full CLI invocation |
