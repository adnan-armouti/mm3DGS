"""Slimmed Rasterizer for mm25DGS_v4 (c6 only).

Vendored from mm25DGS_v2/rasterizer_torch.py with everything removed except
what c6 actually needs:
  - Material reparameterization (raw -> physics)
  - AntennaPatternTorch (TX/RX gain evaluation + injectable trained patterns)
  - Rasterizer holding FMCW/array config + Mitsuba scene loader for ray tracing

Removed (lived in v2 only):
  - _run_reservoir_sampler / _prepare_hit_data / _shadow_test_torch
  - render / render_differentiable / _render_chunk_torch / _render_chunk_return
  - All v2-internal vertex BSDF rendering paths
  - Mesh trimesh load + face_areas / vertex_areas / faces_np / mesh_normals_np

The mesh is loaded as a Mitsuba scene only — that's the sole remaining mesh
dependency, and it's used solely for ray-traced visibility/occlusion testing.
"""

import json
import numpy as np
import torch
from torch import Tensor

C_LIGHT = 299_792_458.0


# ---------------------------------------------------------------------------
# Material reparameterization (matches DrJit bounds exactly)
# ---------------------------------------------------------------------------

def reparameterize_torch(raw: Tensor) -> Tensor:
    """Map raw params -> physics params.

    Original DrJit bounds (mmir/renderer/bsdf/reparameterization.py L109-114):
      eps_real:  1.5 + 8.5 * sigmoid(x)        -> [1.5, 10]
      eps_imag:  exp(clamp(x, -7.0, 16.0))     -> [~1e-3, ~9e6]
      sigma_h:   exp(clamp(x, -16.0, -7.0))    -> [~1e-7, ~1e-3]
      l_c:       exp(clamp(x, -7.6, -2.3))     -> [~5e-4, 0.1]
      tau:       0.05 + 0.9 * sigmoid(x)       -> [0.05, 0.95]
      thickness: exp(clamp(x, -7.0, -1.2))     -> [~1e-3, 0.3]

    Post-2026-04-13 Fix 1: thickness upper bound widened from -1.2 to 2.0
    (exp(-1.2)=0.3 m → exp(2.0)=7.39 m). Phase P5 audit showed 78% of
    points under random init were saturating at the 300 mm upper bound,
    so the optimizer was pushing thickness past the clamp. The new bound
    gives effectively unbounded upward movement (7 m covers "infinite-
    thickness" interferometric averaging without losing physics sense).

    Post-2026-04-13 Fix 3: l_c reparam range widened from [-7.6, -2.3]
    (0.5 mm – 100 mm) to [-10, 2] (45 μm – 7.39 m). P5 showed 64% of
    points hit the old clamp under random init. Physical correlation
    lengths at mmWave-scale roughness span anywhere from tens of μm
    (fine textures) to meters (large-scale structures), well beyond the
    original bound.
    """
    out = torch.empty_like(raw)
    out[..., 0] = 1.5 + 8.5 * torch.sigmoid(raw[..., 0])
    out[..., 1] = torch.exp(torch.clamp(raw[..., 1], -7.0, 16.0))
    out[..., 2] = torch.exp(torch.clamp(raw[..., 2], -16.0, -7.0))
    out[..., 3] = torch.exp(torch.clamp(raw[..., 3], -10.0, 2.0))
    out[..., 4] = 0.05 + 0.9 * torch.sigmoid(raw[..., 4])
    out[..., 5] = torch.exp(torch.clamp(raw[..., 5], -7.0, 2.0))
    return out


def inverse_reparameterize_torch(physics: np.ndarray) -> np.ndarray:
    """Physics params -> raw params (numpy, used at init)."""
    raw = np.empty_like(physics)

    def _logit(x):
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(x / (1.0 - x))

    raw[..., 0] = _logit((physics[..., 0] - 1.5) / 8.5)
    raw[..., 1] = np.log(np.clip(physics[..., 1], np.exp(-7.0), np.exp(16.0)))
    raw[..., 2] = np.log(np.clip(physics[..., 2], np.exp(-16.0), np.exp(-7.0)))
    raw[..., 3] = np.log(np.clip(physics[..., 3], np.exp(-10.0), np.exp(2.0)))
    raw[..., 4] = _logit((physics[..., 4] - 0.05) / 0.9)
    raw[..., 5] = np.log(np.clip(physics[..., 5], np.exp(-7.0), np.exp(2.0)))
    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Antenna pattern (PyTorch evaluation, matches mmIR element_patterns.py)
# ---------------------------------------------------------------------------

class AntennaPatternTorch:
    """Antenna pattern for PyTorch evaluation, with injectable learned patterns."""

    def __init__(self, pattern_path: str, device: str = "cuda:0"):
        data = np.load(pattern_path)  # (361, 2) [E_dB, H_dB]
        E_lin = np.power(10.0, data[:, 0] / 10.0)
        H_lin = np.power(10.0, data[:, 1] / 10.0)

        G_max_dB = max(data[:, 0].max(), data[:, 1].max())
        G_max_lin = 10.0 ** (G_max_dB / 10.0)
        P_max = (E_lin * H_lin).max()
        self.C_scale = float(G_max_lin / P_max) if P_max > 1e-12 else 1.0

        self.E = torch.from_numpy(E_lin.astype(np.float32)).to(device)
        self.H = torch.from_numpy(H_lin.astype(np.float32)).to(device)
        self.device = device

    def evaluate(self, directions: Tensor, boresights: Tensor) -> Tensor:
        """Evaluate antenna gain. directions, boresights: (N, 3)."""
        y_local = boresights

        # Gram-Schmidt local frame (matches mmIR element_patterns.py L638-651)
        aux = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(y_local)
        parallel = torch.abs((y_local * aux).sum(-1)) > 0.99
        if parallel.any():
            alt = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(y_local)
            aux = torch.where(parallel.unsqueeze(-1), alt, aux)

        z_local = torch.cross(y_local, aux, dim=-1)
        z_local = z_local / z_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_local = torch.cross(z_local, y_local, dim=-1)

        d_x = (directions * x_local).sum(-1)
        d_y = (directions * y_local).sum(-1)
        d_z = (directions * z_local).sum(-1)

        angle_E_deg = torch.rad2deg(torch.atan2(d_z, d_y)) + 180.0
        angle_H_deg = torch.rad2deg(torch.atan2(d_x, d_y)) + 180.0

        gain_E = self._interp(angle_E_deg, self.E)
        gain_H = self._interp(angle_H_deg, self.H)
        return (self.C_scale * gain_E * gain_H).clamp(min=0.0)

    def _interp(self, angle_deg: Tensor, pattern: Tensor) -> Tensor:
        """Interpolate 361-sample pattern matching mmIR's indexing exactly."""
        idx_f = angle_deg.clamp(0.0, 360.0)
        idx_lo = idx_f.floor().long().clamp(0, 360)
        idx_hi = (idx_lo + 1)
        idx_hi = torch.where(idx_hi >= 361, torch.zeros_like(idx_hi), idx_hi)
        frac = idx_f - idx_f.floor()
        return pattern[idx_lo] * (1.0 - frac) + pattern[idx_hi] * frac

    def inject_patterns(self, E_plane: np.ndarray, H_plane: np.ndarray):
        """Replace pattern data with learned patterns from mmIR training.

        C_scale is NOT recalculated — it was computed from the original .npy
        file patterns at init, and mmIR keeps that same C_scale when learned
        patterns are injected.
        """
        self.E = torch.from_numpy(E_plane.astype(np.float32)).to(self.device)
        self.H = torch.from_numpy(H_plane.astype(np.float32)).to(self.device)


# ---------------------------------------------------------------------------
# Rasterizer: FMCW config + array geometry + antenna patterns + Mitsuba scene
# ---------------------------------------------------------------------------

class Rasterizer:
    """Holds the static radar configuration that the renderer reads at every
    iteration: FMCW parameters, TX/RX positions and boresights, antenna
    patterns, and the Mitsuba scene used for one-shot init-time ray tracing.
    """

    def __init__(self, config_file, mesh_file, tx_pattern_file, rx_pattern_file,
                 device="cuda:0"):
        with open(config_file) as f:
            radar_cfg = json.load(f)

        # FMCW parameters
        self.center_freq = float(radar_cfg['carrierFrequency'])
        self.slope = float(radar_cfg['freqSlope'])
        self.sample_rate = float(radar_cfg['sampleRate'])
        self.num_samples = int(radar_cfg['numAdcSamples'])
        self.min_freq = self.center_freq
        self.wavelength = C_LIGHT / self.center_freq
        self.lambda_squared = self.wavelength ** 2
        self.K = self.num_samples

        # TX/RX positions (mm -> m) and boresights
        tx_elements = radar_cfg['tx_array']
        rx_elements = radar_cfg['rx_array']
        self.n_tx = len(tx_elements)
        self.n_rx = len(rx_elements)

        self.tx_positions = torch.tensor(
            [e['pos_mm'] for e in tx_elements], dtype=torch.float32,
            device=device) / 1000.0
        self.rx_positions = torch.tensor(
            [e['pos_mm'] for e in rx_elements], dtype=torch.float32,
            device=device) / 1000.0
        self.tx_boresights = torch.tensor(
            [e['boresight'] for e in tx_elements], dtype=torch.float32,
            device=device)
        self.rx_boresights = torch.tensor(
            [e['boresight'] for e in rx_elements], dtype=torch.float32,
            device=device)

        # Radar constants (must match mmIR)
        Pt_watts = 10 ** (13.0 / 10) / 1000.0
        antenna_loss_linear = 10 ** (-4.0 / 10)
        four_pi_squared = (4 * np.pi) ** 2
        self.rx_dBFS_scale = 10 ** ((30.0 - 13.0) / 20)
        self.adc_scale = np.sqrt(50.0) * 32768.0
        # Empirical GT-match factor. The physical constants above yield
        # C_radar ≈ 45.33, but at factory antenna patterns + ITU concrete init,
        # rendered |RA| is ~100× dimmer than GT across all 7 scenes (0.22 decades
        # residual spread). Multiplying radar_constant by 100 brings rendered
        # within factor ~1.5 of GT at init, which is the regime where raw MSE
        # loss can work. See md/v4_unnormalized_mse_loss_plan.md, Phase α.
        self.C_radar_gt_match = 100.0
        self.radar_constant = float(np.sqrt(
            Pt_watts * antenna_loss_linear * self.lambda_squared
            / four_pi_squared)) * self.C_radar_gt_match

        # Antenna patterns (PyTorch)
        self.tx_antenna = AntennaPatternTorch(tx_pattern_file, device)
        self.rx_antenna = AntennaPatternTorch(rx_pattern_file, device)

        # Store paths so the Mitsuba scene can be lazy-loaded for ray tracing
        self._config_file = config_file
        self._mesh_file = mesh_file
        self._tx_pattern_file = tx_pattern_file
        self._rx_pattern_file = rx_pattern_file
        self._mi_scene = None

        self.device = device

    def load_mi_scene(self):
        """Lazy-load the Mitsuba scene for ray tracing.

        Called once at init by the visibility ray-test code. The mesh is the
        ONLY remaining mesh dependency in v4 — used solely for occlusion tests.
        """
        if self._mi_scene is not None:
            return self._mi_scene
        from mmir.renderer.scene_context import SceneContext
        scene_ctx = SceneContext.from_files(
            config_file=self._config_file, scene_file=self._mesh_file,
            pattern_file=None,
            tx_pattern_file=self._tx_pattern_file,
            rx_pattern_file=self._rx_pattern_file,
            material_type="metal", enable_gradients=False, verbose=False,
        )
        self._mi_scene = scene_ctx.scene
        return self._mi_scene

    def free_mi_scene(self):
        """Drop the Mitsuba scene reference once init-time ray tracing is done."""
        self._mi_scene = None

    def inject_trained_params(self, pattern_data=None):
        """Inject mmIR-trained antenna patterns (E/H planes) if available."""
        if pattern_data is not None:
            self.tx_antenna.inject_patterns(
                pattern_data['tx_E_plane'], pattern_data['tx_H_plane'])
            self.rx_antenna.inject_patterns(
                pattern_data['rx_E_plane'], pattern_data['rx_H_plane'])
