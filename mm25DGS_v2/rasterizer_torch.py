"""
mm25DGS_v2 PyTorch Rasterizer: Differentiable vertex-based radar renderer.

Pure PyTorch replacement for rasterizer.py (which uses DrJit for physics).
Uses mmIR's reservoir sampler for visibility, then runs BSDF/antenna/radar-eq/phase
entirely in PyTorch with full autograd support.

Pipeline (same as rasterizer.py but in PyTorch):
  1. Run mmIR reservoir sampler to identify hit positions (DrJit, non-differentiable)
  2. All physics in PyTorch: reparameterization, BSDF, antenna, radar eq, phase scatter
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import trimesh
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

C = 299_792_458.0


# ---------------------------------------------------------------------------
#  Material reparameterization (matching DrJit bounds exactly)
# ---------------------------------------------------------------------------

def reparameterize_torch(raw: Tensor) -> Tensor:
    """Map raw params -> physics params, matching DrJit bounds exactly.

    DrJit bounds (from mmir/renderer/bsdf/reparameterization.py L109-114):
      eps_real:  1.5 + 8.5 * sigmoid(x)        -> [1.5, 10]
      eps_imag:  exp(clamp(x, -7.0, 16.0))     -> [~1e-3, ~9e6]
      sigma_h:   exp(clamp(x, -16.0, -7.0))    -> [~1e-7, ~1e-3]
      l_c:       exp(clamp(x, -7.6, -2.3))     -> [~5e-4, 0.1]
      tau:       0.05 + 0.9 * sigmoid(x)       -> [0.05, 0.95]
      thickness: exp(clamp(x, -7.0, -1.2))     -> [~1e-3, 0.3]
    """
    out = torch.empty_like(raw)
    out[..., 0] = 1.5 + 8.5 * torch.sigmoid(raw[..., 0])
    out[..., 1] = torch.exp(torch.clamp(raw[..., 1], -7.0, 16.0))
    out[..., 2] = torch.exp(torch.clamp(raw[..., 2], -16.0, -7.0))
    out[..., 3] = torch.exp(torch.clamp(raw[..., 3], -7.6, -2.3))
    out[..., 4] = 0.05 + 0.9 * torch.sigmoid(raw[..., 4])
    out[..., 5] = torch.exp(torch.clamp(raw[..., 5], -7.0, -1.2))
    return out


def inverse_reparameterize_torch(physics: np.ndarray) -> np.ndarray:
    """Physics params -> raw params (numpy, for init)."""
    raw = np.empty_like(physics)

    def _logit(x):
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        return np.log(x / (1.0 - x))

    raw[..., 0] = _logit((physics[..., 0] - 1.5) / 8.5)
    raw[..., 1] = np.log(np.clip(physics[..., 1], np.exp(-7.0), np.exp(16.0)))
    raw[..., 2] = np.log(np.clip(physics[..., 2], np.exp(-16.0), np.exp(-7.0)))
    raw[..., 3] = np.log(np.clip(physics[..., 3], np.exp(-7.6), np.exp(-2.3)))
    raw[..., 4] = _logit((physics[..., 4] - 0.05) / 0.9)
    raw[..., 5] = np.log(np.clip(physics[..., 5], np.exp(-7.0), np.exp(-1.2)))
    return raw.astype(np.float32)


# ---------------------------------------------------------------------------
#  Antenna gain (PyTorch) — reuses mm25DGS/antenna_torch.py logic
# ---------------------------------------------------------------------------

class AntennaPatternTorch:
    """Antenna pattern for PyTorch evaluation."""

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

        # Gram-Schmidt local frame (matching mmIR element_patterns.py L638-651)
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
        """Interpolate 361-sample pattern matching mmIR's indexing exactly.

        mmIR uses: idx = angle * (num_angles-1) / 360 = angle (for 361 samples).
        Pattern has 361 elements at indices 0..360 (0deg..360deg inclusive).
        Learned patterns may NOT be periodic (pattern[0] != pattern[360]).
        """
        # Map angle to index in [0, 360] (matching mmIR: angle * 360/360)
        idx_f = angle_deg.clamp(0.0, 360.0)
        idx_lo = idx_f.floor().long().clamp(0, 360)
        idx_hi = (idx_lo + 1)
        # Wrap only 361 -> 0 (matching mmIR)
        idx_hi = torch.where(idx_hi >= 361, torch.zeros_like(idx_hi), idx_hi)
        frac = idx_f - idx_f.floor()
        return pattern[idx_lo] * (1.0 - frac) + pattern[idx_hi] * frac

    def inject_patterns(self, E_plane: np.ndarray, H_plane: np.ndarray):
        """Replace pattern data with learned patterns.

        C_scale is NOT recalculated — it was computed from the original .npy
        file patterns at init, and mmIR keeps that same C_scale when learned
        patterns are injected.
        """
        self.E = torch.from_numpy(E_plane.astype(np.float32)).to(self.device)
        self.H = torch.from_numpy(H_plane.astype(np.float32)).to(self.device)


# ---------------------------------------------------------------------------
#  BSDF: import from mm25DGS (already verified)
# ---------------------------------------------------------------------------

from mm25DGS.bsdf_torch import evaluate_bsdf_jones_f_cos


# ---------------------------------------------------------------------------
#  PyTorch Rasterizer
# ---------------------------------------------------------------------------

class RasterizerTorch:
    """Differentiable vertex-based radar renderer using PyTorch physics."""

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
        self.wavelength = C / self.center_freq
        self.lambda_squared = self.wavelength ** 2
        self.K = self.num_samples

        # TX/RX positions
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

        # Radar constants (same as mmIR)
        Pt_watts = 10 ** (13.0 / 10) / 1000.0
        antenna_loss_linear = 10 ** (-4.0 / 10)
        four_pi_squared = (4 * np.pi) ** 2
        self.rx_dBFS_scale = 10 ** ((30.0 - 13.0) / 20)
        self.adc_scale = np.sqrt(50.0) * 32768.0
        self.radar_constant = float(np.sqrt(
            Pt_watts * antenna_loss_linear * self.lambda_squared
            / four_pi_squared))

        # Mesh
        mesh = trimesh.load(mesh_file)
        self.vertices_np = np.array(mesh.vertices, dtype=np.float32)
        self.faces_np = np.array(mesh.faces, dtype=np.int32)
        self.n_vertices = len(self.vertices_np)
        self.mesh_normals_np = np.array(mesh.vertex_normals, dtype=np.float32)

        # Per-vertex areas
        face_areas = mesh.area_faces
        vertex_areas = np.zeros(self.n_vertices, dtype=np.float32)
        for i in range(3):
            np.add.at(vertex_areas, self.faces_np[:, i], face_areas / 3.0)
        self.vertex_areas_np = vertex_areas

        # Antenna patterns (PyTorch)
        self.tx_antenna = AntennaPatternTorch(tx_pattern_file, device)
        self.rx_antenna = AntennaPatternTorch(rx_pattern_file, device)

        # Store paths for reservoir sampler
        self._config_file = config_file
        self._mesh_file = mesh_file
        self._tx_pattern_file = tx_pattern_file
        self._rx_pattern_file = rx_pattern_file

        self.device = device

    def inject_trained_params(self, raw_params, normal_params=None,
                              pattern_data=None):
        """Load trained parameters (numpy)."""
        self.raw_params_np = raw_params.astype(np.float32)

        if normal_params is not None:
            self.normals_np = normal_params.astype(np.float32)
        else:
            self.normals_np = self.mesh_normals_np.copy()

        if pattern_data is not None:
            self.tx_antenna.inject_patterns(
                pattern_data['tx_E_plane'], pattern_data['tx_H_plane'])
            self.rx_antenna.inject_patterns(
                pattern_data['rx_E_plane'], pattern_data['rx_H_plane'])

    def _run_reservoir_sampler(self, seed=42):
        """Run mmIR reservoir sampler (requires DrJit/Mitsuba)."""
        import mitsuba as mi
        import drjit as dr
        from mmir.renderer.renderer import FMCWRendererRef
        from mmir.renderer.scene_context import SceneContext
        from mmir.renderer.config import RenderConfigRef

        render_config = RenderConfigRef(
            n_hits_per_rx=1500, n_rays_per_res=16, max_distance=1e10,
            seed=seed, verbose=False, use_radar_equation=True,
            bsdf_model='mmwave_jones', mmwave_polarization='vertical',
            hemisphere_sampling='cosine', double_sided=True,
            use_vertex_normals=True,
        )

        scene_ctx = SceneContext.from_files(
            config_file=self._config_file, scene_file=self._mesh_file,
            pattern_file=None,
            tx_pattern_file=self._tx_pattern_file,
            rx_pattern_file=self._rx_pattern_file,
            material_type="metal", enable_gradients=False, verbose=False,
        )

        renderer = FMCWRendererRef(
            scene_ctx, render_config, verbose=False,
            tx_pattern_file=self._tx_pattern_file,
            rx_pattern_file=self._rx_pattern_file,
        )

        tx_pos, rx_pos, tx_bore, rx_bore = renderer._get_antenna_arrays()

        hits = renderer.sampler.sample_reservoir_drjit(
            scene=scene_ctx.scene, rx_positions=rx_pos,
            rx_boresights=rx_bore, seed=seed, verbose=False,
            use_vertex_normals=True,
        )

        self._mi_scene = scene_ctx.scene
        return hits

    def _prepare_hit_data(self, hits):
        """Extract hit data from reservoir sampler output."""
        import drjit as dr
        import mitsuba as mi

        positions = np.column_stack([
            np.array(hits.hit_P.x),
            np.array(hits.hit_P.y),
            np.array(hits.hit_P.z),
        ])

        # Barycentric interpolation of materials and normals
        bary_u = np.array(hits.hit_bary_u)
        bary_v = np.array(hits.hit_bary_v)
        bary_w = 1.0 - bary_u - bary_v

        if hits.vertex_ids_0 is not None:
            vi0 = np.array(hits.vertex_ids_0)
            vi1 = np.array(hits.vertex_ids_1)
            vi2 = np.array(hits.vertex_ids_2)
        else:
            prim_ids = np.array(hits.hit_prim_ids)
            vi0 = self.faces_np[prim_ids, 0]
            vi1 = self.faces_np[prim_ids, 1]
            vi2 = self.faces_np[prim_ids, 2]

        raw_params = (bary_w[:, None] * self.raw_params_np[vi0]
                      + bary_u[:, None] * self.raw_params_np[vi1]
                      + bary_v[:, None] * self.raw_params_np[vi2])

        normals = (bary_w[:, None] * self.normals_np[vi0]
                   + bary_u[:, None] * self.normals_np[vi1]
                   + bary_v[:, None] * self.normals_np[vi2])
        nrm_len = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 0.01)
        normals = normals / nrm_len

        pdfs = np.array(hits.hit_pdf)
        rx_idx = np.array(hits.rx_element_idx)
        n_attempted_per_rx = np.array(hits.n_attempted_per_rx)
        n_attempted = n_attempted_per_rx[rx_idx].astype(np.float32)

        areas = 1.0 / (np.maximum(pdfs, 1e-8) * np.maximum(n_attempted, 1.0))

        return positions, normals, areas, raw_params

    def _shadow_test_torch(self, hit_P, dir_to_tx, d_to_tx):
        """Shadow ray test using Mitsuba scene (non-differentiable)."""
        import mitsuba as mi
        import drjit as dr

        if not hasattr(self, '_mi_scene') or self._mi_scene is None:
            return torch.ones(hit_P.shape[0], dtype=torch.bool, device=self.device)

        N = hit_P.shape[0]
        eps = 1e-4

        # Convert to DrJit for shadow test
        origins = mi.Point3f(
            mi.Float(hit_P[:, 0].cpu().numpy() + eps * dir_to_tx[:, 0].cpu().numpy()),
            mi.Float(hit_P[:, 1].cpu().numpy() + eps * dir_to_tx[:, 1].cpu().numpy()),
            mi.Float(hit_P[:, 2].cpu().numpy() + eps * dir_to_tx[:, 2].cpu().numpy()),
        )
        dirs = mi.Vector3f(
            mi.Float(dir_to_tx[:, 0].cpu().numpy()),
            mi.Float(dir_to_tx[:, 1].cpu().numpy()),
            mi.Float(dir_to_tx[:, 2].cpu().numpy()),
        )
        rays = mi.Ray3f(origins, dirs)
        rays.maxt = mi.Float(d_to_tx.cpu().numpy() - 2e-4)

        occluded = self._mi_scene.ray_test(rays)
        occluded_np = np.array(occluded)

        return torch.from_numpy(~occluded_np).to(self.device)

    def _render_chunk_torch(self, verts_t, normals_t, areas_t, raw_params_t,
                            adc_real, adc_imag, detach_phase=True):
        """Render a chunk of scatterers in PyTorch and accumulate into ADC.

        Matches the DrJit rasterizer exactly: flat per-path evaluation with
        per-path normal flip, per-path BSDF, per-path antenna gain.

        Args:
            verts_t: (n_vis, 3) positions
            normals_t: (n_vis, 3) normals
            areas_t: (n_vis,) area weights
            raw_params_t: (n_vis, 6) raw material params
            adc_real: (n_tx, n_rx, K) accumulator
            adc_imag: (n_tx, n_rx, K) accumulator
            detach_phase: if True, phase is not differentiable (matching mmIR)
        """
        n_tx = self.n_tx
        n_rx = self.n_rx
        K = self.K
        n_vis = verts_t.shape[0]
        n_mimo = n_tx * n_rx
        n_total = n_vis * n_mimo

        # Flat indices: v varies slowest, then tx, then rx (matching DrJit)
        v_idx = torch.arange(n_vis, device=self.device).repeat_interleave(n_mimo)
        t_idx = torch.arange(n_tx, device=self.device).repeat_interleave(n_rx).repeat(n_vis)
        r_idx = torch.arange(n_rx, device=self.device).repeat(n_vis * n_tx)

        # Flat geometry
        hit_P = verts_t[v_idx]                                # (n_total, 3)
        hit_N = normals_t[v_idx]                              # (n_total, 3)

        delta_tx = self.tx_positions[t_idx] - hit_P
        d_tx = delta_tx.norm(dim=-1).clamp(min=1e-10)
        dir_to_tx = delta_tx / d_tx.unsqueeze(-1)

        delta_rx = hit_P - self.rx_positions[r_idx]
        d_rx = delta_rx.norm(dim=-1).clamp(min=1e-10)
        dir_to_rx = -delta_rx / d_rx.unsqueeze(-1)

        # Per-path double-sided normal flip (matching DrJit exactly)
        cos_out = (dir_to_rx * hit_N).sum(-1)
        flip = cos_out < 0
        hit_N = torch.where(flip.unsqueeze(-1), -hit_N, hit_N)

        # Cosine filtering
        cos_theta_out = (dir_to_rx * hit_N).sum(-1)
        cos_theta_in = (dir_to_tx * hit_N).sum(-1)
        active = (cos_theta_out > 1e-6) & (cos_theta_in > 1e-6)

        # Shadow test
        if hasattr(self, '_mi_scene') and self._mi_scene is not None:
            visible = self._shadow_test_torch(hit_P, dir_to_tx, d_tx)
            active = active & visible

        # Materials: reparameterize per-vertex, then index per-path
        physics = reparameterize_torch(raw_params_t)          # (n_vis, 6)
        eps_real = physics[v_idx, 0]
        eps_imag = physics[v_idx, 1]
        sigma_h = physics[v_idx, 2]
        l_c = physics[v_idx, 3]
        tau_mat = physics[v_idx, 4]
        thickness = physics[v_idx, 5]

        # Per-path BSDF
        brdf_weight = evaluate_bsdf_jones_f_cos(
            cos_theta_in.clamp(min=1e-6),
            wo=dir_to_rx, wi=dir_to_tx, n=hit_N,
            eps_real=eps_real, eps_imag=eps_imag,
            sigma_h=sigma_h, l_c=l_c,
            tau_base=tau_mat, thickness=thickness,
        )  # (n_total,)

        # Per-path antenna gains
        gain_tx = self.tx_antenna.evaluate(dir_to_tx, self.tx_boresights[t_idx])
        gain_rx = self.rx_antenna.evaluate(-dir_to_rx, self.rx_boresights[r_idx])
        brdf_weight = brdf_weight * gain_tx * gain_rx

        # Area weighting
        brdf_weight = brdf_weight * areas_t[v_idx]

        # Radar equation
        d_safe = d_tx.clamp(min=1e-4)
        path_loss = 1.0 / (d_safe * d_safe)
        radar_scale = self.radar_constant * self.rx_dBFS_scale * self.adc_scale
        weight = radar_scale * torch.sqrt(
            (brdf_weight * path_loss).clamp(min=1e-20))

        # Zero out inactive paths
        weight = weight * active.float()

        # Phase + phasor scatter
        R_total = d_rx + d_tx
        tau_delay = R_total / C
        TWO_PI = 2.0 * math.pi
        phi_const = TWO_PI * self.min_freq * tau_delay
        phi_slope = TWO_PI * self.slope * tau_delay

        if detach_phase:
            phi_const = phi_const.detach()
            phi_slope = phi_slope.detach()

        # Scatter into ADC bins in K sub-chunks for memory
        K_CHUNK = 64
        t_grid = torch.arange(K, device=self.device, dtype=torch.float32) / self.sample_rate
        adc_real_flat = adc_real.reshape(-1)
        adc_imag_flat = adc_imag.reshape(-1)
        base_idx = t_idx * (n_rx * K) + r_idx * K  # (n_total,)

        for k_start in range(0, K, K_CHUNK):
            k_end = min(k_start + K_CHUNK, K)
            t_k = t_grid[k_start:k_end]  # (k_size,)

            phi = phi_const.unsqueeze(-1) + phi_slope.unsqueeze(-1) * t_k
            w = weight.unsqueeze(-1)

            real_contrib = (w * torch.cos(phi)).reshape(-1)
            imag_contrib = (w * torch.sin(phi)).reshape(-1)

            k_idx = torch.arange(k_start, k_end, device=self.device)
            flat_idx = (base_idx.unsqueeze(-1) + k_idx).reshape(-1)

            adc_real_flat.scatter_add_(0, flat_idx, real_contrib)
            adc_imag_flat.scatter_add_(0, flat_idx, imag_contrib)

    def render(self, verbose=False, chunk_size=2000, detach_phase=True):
        """Run the full rasterizer pipeline.

        Returns:
            adc_ri: (n_tx, n_rx, K, 2) real/imag numpy array
        """
        t0 = time.time()

        # Use reservoir sampler for hit positions
        if verbose:
            print("  [TorchRast] Running reservoir sampler...")
        hits = self._run_reservoir_sampler()
        verts, normals, areas, raw_params = self._prepare_hit_data(hits)

        n_vis = len(verts)
        if verbose:
            print(f"  [TorchRast] {n_vis} scatterers, "
                  f"{n_vis * self.n_tx * self.n_rx:,} paths")

        # Move to device
        verts_t = torch.from_numpy(verts.astype(np.float32)).to(self.device)
        normals_t = torch.from_numpy(normals.astype(np.float32)).to(self.device)
        areas_t = torch.from_numpy(areas.astype(np.float32)).to(self.device)
        raw_params_t = torch.from_numpy(raw_params.astype(np.float32)).to(self.device)

        # Allocate ADC
        adc_real = torch.zeros(self.n_tx, self.n_rx, self.K,
                               device=self.device, dtype=torch.float32)
        adc_imag = torch.zeros(self.n_tx, self.n_rx, self.K,
                               device=self.device, dtype=torch.float32)

        # Process in chunks
        n_chunks = (n_vis + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            i0 = ci * chunk_size
            i1 = min(i0 + chunk_size, n_vis)
            self._render_chunk_torch(
                verts_t[i0:i1], normals_t[i0:i1], areas_t[i0:i1],
                raw_params_t[i0:i1], adc_real, adc_imag,
                detach_phase=detach_phase)
            if verbose and (ci + 1) % max(1, n_chunks // 5) == 0:
                print(f"    chunk {ci+1}/{n_chunks} ({time.time()-t0:.1f}s)")

        if verbose:
            print(f"  [TorchRast] Done in {time.time()-t0:.1f}s")

        # To numpy
        adc_ri = torch.stack([adc_real, adc_imag], dim=-1).cpu().detach().numpy()
        return adc_ri

    def render_differentiable(self, raw_params_t, normals_t, verts_t, areas_t,
                              detach_phase=True, chunk_size=2000,
                              skip_shadow=False, use_checkpoint=False):
        """Differentiable forward pass — for training.

        All inputs are PyTorch tensors on device. Returns (adc_real, adc_imag)
        as differentiable tensors on device.

        Args:
            skip_shadow: If True, skip the DrJit shadow ray test.
            use_checkpoint: If True, use gradient checkpointing per chunk
                to reduce peak memory (recomputes forward during backward).
        """
        n_vis = verts_t.shape[0]
        n_tx, n_rx, K = self.n_tx, self.n_rx, self.K

        saved_scene = getattr(self, '_mi_scene', None)
        if skip_shadow:
            self._mi_scene = None

        adc_real = torch.zeros(n_tx, n_rx, K, device=self.device)
        adc_imag = torch.zeros(n_tx, n_rx, K, device=self.device)

        n_chunks = (n_vis + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            i0 = ci * chunk_size
            i1 = min(i0 + chunk_size, n_vis)

            if use_checkpoint:
                cr, ci_adc = torch.utils.checkpoint.checkpoint(
                    self._render_chunk_return,
                    verts_t[i0:i1], normals_t[i0:i1], areas_t[i0:i1],
                    raw_params_t[i0:i1], detach_phase,
                    use_reentrant=False,
                )
            else:
                cr = torch.zeros(n_tx, n_rx, K, device=self.device)
                ci_adc = torch.zeros(n_tx, n_rx, K, device=self.device)
                self._render_chunk_torch(
                    verts_t[i0:i1], normals_t[i0:i1], areas_t[i0:i1],
                    raw_params_t[i0:i1], cr, ci_adc,
                    detach_phase=detach_phase)

            adc_real = adc_real + cr
            adc_imag = adc_imag + ci_adc

        if skip_shadow:
            self._mi_scene = saved_scene

        return adc_real, adc_imag

    def _render_chunk_return(self, verts_t, normals_t, areas_t,
                             raw_params_t, detach_phase):
        """Wrapper for _render_chunk_torch that returns (real, imag) tensors.

        Used by torch.utils.checkpoint which requires a function that returns
        tensors (not in-place scatter_add).
        """
        n_tx, n_rx, K = self.n_tx, self.n_rx, self.K
        cr = torch.zeros(n_tx, n_rx, K, device=self.device)
        ci = torch.zeros(n_tx, n_rx, K, device=self.device)
        self._render_chunk_torch(
            verts_t, normals_t, areas_t, raw_params_t,
            cr, ci, detach_phase=detach_phase)
        return cr, ci
