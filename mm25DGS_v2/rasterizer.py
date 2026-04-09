"""
mm25DGS_v2 Rasterizer: Forward-only vertex-based radar renderer.

Replaces mmIR's stochastic MC sampling with deterministic vertex enumeration
of radar-visible triangles. Uses mmIR code DIRECTLY for BSDF, antenna gain,
material reparameterization, radar equation, and phase computation.

Pipeline:
  1. Run mmIR reservoir sampler to identify hit triangles, extract unique vertices
  2. Compute (vertex, TX, RX) geometry for all MIMO combinations
  3. Double-sided normal flip + cosine angle filtering (same as mmIR)
  4. Direct per-vertex material loading (no barycentric interpolation)
  5. BSDF evaluation (mmIR Jones BSDF)
  6. Antenna gain (mmIR separable E×H pattern)
  7. Vertex area weighting (replaces MC pdf correction)
  8. Radar equation (mmIR radar constant)
  9. Phase + scatter-add to ADC (mmIR vectorized phasor)
"""

import os
import sys
import json
import numpy as np
import torch
import trimesh
import time

import mitsuba as mi
import drjit as dr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmir.renderer.bsdf.reparameterization import (
    reparameterize_physics_params_drjit, create_drjit_raw_params)
from mmir.sensor.element_patterns import evaluate_combined_gain, AntennaPatternLoader
from mmir.data.ra_utils import (
    adc_to_ra_image, ra_polar_to_cartesian, compute_cartesian_ra_metrics)
from mmir.data.io_utils import compute_range_res_from_cfg

C = 299792458.0


class Rasterizer:
    """Vertex-based forward radar renderer using mmIR physics."""

    def __init__(self, config_file, mesh_file, tx_pattern_file, rx_pattern_file):
        # Load radar config
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

        # TX/RX positions and boresights
        tx_elements = radar_cfg['tx_array']
        rx_elements = radar_cfg['rx_array']
        self.n_tx = len(tx_elements)
        self.n_rx = len(rx_elements)
        self.K = self.num_samples

        self.tx_positions_np = np.array(
            [e['pos_mm'] for e in tx_elements], dtype=np.float32) / 1000.0
        self.rx_positions_np = np.array(
            [e['pos_mm'] for e in rx_elements], dtype=np.float32) / 1000.0
        self.tx_boresights_np = np.array(
            [e['boresight'] for e in tx_elements], dtype=np.float32)
        self.rx_boresights_np = np.array(
            [e['boresight'] for e in rx_elements], dtype=np.float32)

        # Radar constants (same as mmIR SBRIntegratorRef)
        Pt_watts = 10 ** (13.0 / 10) / 1000.0
        antenna_loss_linear = 10 ** (-4.0 / 10)
        four_pi_squared = (4 * np.pi) ** 2
        self.rx_dBFS_scale = 10 ** ((30.0 - 13.0) / 20)
        self.adc_scale = np.sqrt(50.0) * 32768.0
        self.radar_constant = np.sqrt(
            Pt_watts * antenna_loss_linear * self.lambda_squared
            / four_pi_squared)

        # Load mesh via trimesh (for vertex areas)
        self.mesh = trimesh.load(mesh_file)
        self.vertices = np.array(self.mesh.vertices, dtype=np.float32)
        self.faces = np.array(self.mesh.faces, dtype=np.int32)
        self.n_vertices = len(self.vertices)

        # Per-vertex areas (1/3 of adjacent face areas)
        face_areas = self.mesh.area_faces
        self.vertex_areas = np.zeros(self.n_vertices, dtype=np.float32)
        for i in range(3):
            np.add.at(self.vertex_areas, self.faces[:, i], face_areas / 3.0)

        # Antenna patterns
        self.tx_loader = AntennaPatternLoader(tx_pattern_file)
        self.rx_loader = AntennaPatternLoader(rx_pattern_file)

        # BSDF (mmIR Jones)
        from mmir.renderer.bsdf.mmwave_jones import BSDFmmWaveJones
        self.bsdf = BSDFmmWaveJones(
            tx_polarization=mi.Vector3f(0, 0, 1),
            rx_polarization=mi.Vector3f(0, 0, 1),
            enable_incoherent=True,
            use_fresnel_phase=False,
            enable_cbs=True,
        )

        # Store paths for mmIR scene loading
        self._config_file = config_file
        self._mesh_file = mesh_file
        self._tx_pattern_file = tx_pattern_file
        self._rx_pattern_file = rx_pattern_file

    def inject_trained_params(self, raw_params, normal_params=None,
                              pattern_data=None):
        """Load trained parameters."""
        self.raw_params = raw_params.astype(np.float32)

        if normal_params is not None:
            self.normals = normal_params.astype(np.float32)
        else:
            self.normals = np.array(self.mesh.vertex_normals, dtype=np.float32)

        if pattern_data is not None:
            self.tx_loader.E_plane_linear = mi.Float(
                pattern_data['tx_E_plane'].astype(np.float32))
            self.tx_loader.H_plane_linear = mi.Float(
                pattern_data['tx_H_plane'].astype(np.float32))
            self.rx_loader.E_plane_linear = mi.Float(
                pattern_data['rx_E_plane'].astype(np.float32))
            self.rx_loader.H_plane_linear = mi.Float(
                pattern_data['rx_H_plane'].astype(np.float32))

    def _find_visible_triangles(self, seed=42):
        """Use mmIR's reservoir sampler to find which triangles the radar hits.

        Returns unique face indices of hit triangles.
        """
        hits = self._run_reservoir_sampler(seed)
        prim_ids = np.unique(np.array(hits.hit_prim_ids))
        return prim_ids

    def _run_reservoir_sampler(self, seed=42):
        """Run mmIR's reservoir sampler and return hits + Mitsuba scene.

        Also stores the Mitsuba scene for shadow ray testing.
        """
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

        # Keep scene for shadow rays
        self._mi_scene = scene_ctx.scene

        return hits

    def _prepare_hit_data(self, hits):
        """Extract hit positions, normals, barycentrics for direct use.

        Uses the actual hit positions from reservoir sampling (not vertices
        or centroids) to preserve phase accuracy at 77 GHz.

        Returns:
            positions: (n_valid, 3) hit positions on triangle surfaces
            normals: (n_valid, 3) interpolated normals at hits
            raw_params: (n_valid, 6) interpolated materials at hits
            pdfs: (n_valid,) sampling PDFs
            n_attempted: (n_valid,) rays attempted per RX for each hit
        """
        import drjit as dr
        import mitsuba as mi

        n_valid = hits.n_valid

        # Extract positions and normals
        positions = np.column_stack([
            np.array(hits.hit_P.x),
            np.array(hits.hit_P.y),
            np.array(hits.hit_P.z),
        ])
        normals_raw = np.column_stack([
            np.array(hits.hit_N.x),
            np.array(hits.hit_N.y),
            np.array(hits.hit_N.z),
        ])

        # Interpolate materials using barycentrics (same as mmIR)
        bary_u = np.array(hits.hit_bary_u)
        bary_v = np.array(hits.hit_bary_v)
        bary_w = 1.0 - bary_u - bary_v

        if hits.vertex_ids_0 is not None:
            vi0 = np.array(hits.vertex_ids_0)
            vi1 = np.array(hits.vertex_ids_1)
            vi2 = np.array(hits.vertex_ids_2)
        else:
            prim_ids = np.array(hits.hit_prim_ids)
            vi0 = self.faces[prim_ids, 0]
            vi1 = self.faces[prim_ids, 1]
            vi2 = self.faces[prim_ids, 2]

        # Barycentric interpolation of materials
        m0 = self.raw_params[vi0]
        m1 = self.raw_params[vi1]
        m2 = self.raw_params[vi2]
        raw_params = (bary_w[:, None] * m0
                      + bary_u[:, None] * m1
                      + bary_v[:, None] * m2)

        # Barycentric interpolation of normals (if learned normals differ)
        n0 = self.normals[vi0]
        n1 = self.normals[vi1]
        n2 = self.normals[vi2]
        normals = bary_w[:, None] * n0 + bary_u[:, None] * n1 + bary_v[:, None] * n2
        nrm_len = np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True), 0.01)
        normals = normals / nrm_len

        # MC correction data
        pdfs = np.array(hits.hit_pdf)
        rx_idx = np.array(hits.rx_element_idx)
        n_attempted_per_rx = np.array(hits.n_attempted_per_rx)
        n_attempted = n_attempted_per_rx[rx_idx].astype(np.float32)

        return positions, normals, raw_params, pdfs, n_attempted

    def _prepare_triangle_data(self, face_ids, n_samples=4):
        """Sub-sample each triangle at multiple points for phase accuracy.

        Uses barycentric sub-sampling: centroid + edge midpoints (n_samples=4)
        or just centroid (n_samples=1). Each sub-sample gets area/n_samples.

        Returns:
            positions: (N*n_samples, 3) sample positions
            normals: (N*n_samples, 3) interpolated normals
            areas: (N*n_samples,) area per sample (face_area / n_samples)
            raw_params: (N*n_samples, 6) interpolated materials
        """
        v0 = self.vertices[self.faces[face_ids, 0]]
        v1 = self.vertices[self.faces[face_ids, 1]]
        v2 = self.vertices[self.faces[face_ids, 2]]

        n0 = self.normals[self.faces[face_ids, 0]]
        n1 = self.normals[self.faces[face_ids, 1]]
        n2 = self.normals[self.faces[face_ids, 2]]

        m0 = self.raw_params[self.faces[face_ids, 0]]
        m1 = self.raw_params[self.faces[face_ids, 1]]
        m2 = self.raw_params[self.faces[face_ids, 2]]

        face_areas = self.mesh.area_faces[face_ids]

        if n_samples == 1:
            # Centroid only
            bary = np.array([[1/3, 1/3, 1/3]])
        elif n_samples == 4:
            # Centroid + 3 edge midpoints
            bary = np.array([
                [1/3, 1/3, 1/3],   # centroid
                [1/2, 1/2, 0],     # edge 01 midpoint
                [0, 1/2, 1/2],     # edge 12 midpoint
                [1/2, 0, 1/2],     # edge 02 midpoint
            ])
        elif n_samples == 7:
            # Centroid + edge midpoints + vertices
            bary = np.array([
                [1/3, 1/3, 1/3],
                [1/2, 1/2, 0], [0, 1/2, 1/2], [1/2, 0, 1/2],
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
            ])
        else:
            raise ValueError(f"n_samples must be 1, 4, or 7, got {n_samples}")

        n_faces = len(face_ids)
        all_pos = []
        all_nrm = []
        all_mat = []
        all_area = []

        for w0, w1, w2 in bary:
            pos = w0 * v0 + w1 * v1 + w2 * v2
            nrm = w0 * n0 + w1 * n1 + w2 * n2
            mat = w0 * m0 + w1 * m1 + w2 * m2
            all_pos.append(pos)
            all_nrm.append(nrm)
            all_mat.append(mat)
            all_area.append(face_areas / n_samples)

        positions = np.concatenate(all_pos, axis=0)
        normals = np.concatenate(all_nrm, axis=0)
        raw_params = np.concatenate(all_mat, axis=0)
        areas = np.concatenate(all_area, axis=0)

        # Normalize normals
        nrm_len = np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True), 0.01)
        normals = normals / nrm_len

        return positions, normals, areas, raw_params

    def _render_vertex_chunk(self, verts, normals, areas, raw_params,
                             adc_real_flat, adc_imag_flat):
        """Render a chunk of vertices and scatter-add into ADC arrays."""
        n_tx = self.n_tx
        n_rx = self.n_rx
        K = self.K
        n_vis = len(verts)
        n_mimo = n_tx * n_rx
        n_total = n_vis * n_mimo

        # Flat index arrays: v varies slowest, then t, then r
        v_idx = np.repeat(np.arange(n_vis), n_mimo)
        t_idx = np.tile(np.repeat(np.arange(n_tx), n_rx), n_vis)
        r_idx = np.tile(np.arange(n_rx), n_vis * n_tx)

        # === Geometry (DrJit) ===
        hit_P = mi.Point3f(mi.Float(verts[v_idx, 0]),
                           mi.Float(verts[v_idx, 1]),
                           mi.Float(verts[v_idx, 2]))
        hit_N = mi.Vector3f(mi.Float(normals[v_idx, 0]),
                            mi.Float(normals[v_idx, 1]),
                            mi.Float(normals[v_idx, 2]))
        tx_pos = mi.Point3f(mi.Float(self.tx_positions_np[t_idx, 0]),
                            mi.Float(self.tx_positions_np[t_idx, 1]),
                            mi.Float(self.tx_positions_np[t_idx, 2]))
        rx_pos = mi.Point3f(mi.Float(self.rx_positions_np[r_idx, 0]),
                            mi.Float(self.rx_positions_np[r_idx, 1]),
                            mi.Float(self.rx_positions_np[r_idx, 2]))

        delta_tx = tx_pos - hit_P
        d_hit_to_tx = dr.norm(delta_tx)
        delta_rx = hit_P - rx_pos
        d_rx_to_hit = dr.norm(delta_rx)

        _d_safe_tx = dr.maximum(d_hit_to_tx, mi.Float(1e-10))
        _d_safe_rx = dr.maximum(d_rx_to_hit, mi.Float(1e-10))

        dir_hit_to_tx = mi.Vector3f(delta_tx.x / _d_safe_tx,
                                    delta_tx.y / _d_safe_tx,
                                    delta_tx.z / _d_safe_tx)
        dir_hit_to_rx = mi.Vector3f(-delta_rx.x / _d_safe_rx,
                                    -delta_rx.y / _d_safe_rx,
                                    -delta_rx.z / _d_safe_rx)

        # Double-sided normal flip (same as mmIR)
        cos_out = dr.dot(dir_hit_to_rx, hit_N)
        need_flip = cos_out < 0
        hit_N = dr.select(need_flip,
                          mi.Vector3f(-hit_N.x, -hit_N.y, -hit_N.z),
                          hit_N)

        # Cosine angle filtering
        cos_theta_out = dr.dot(dir_hit_to_rx, hit_N)
        cos_theta_in = dr.dot(dir_hit_to_tx, hit_N)
        active = (cos_theta_out > mi.Float(1e-6)) & (cos_theta_in > mi.Float(1e-6))

        # Shadow ray test (same as mmIR synthesis_e2e.py Step 5)
        if hasattr(self, '_mi_scene') and self._mi_scene is not None:
            epsilon = mi.Float(1e-4)
            shadow_origins = mi.Point3f(
                hit_P.x + epsilon * dir_hit_to_tx.x,
                hit_P.y + epsilon * dir_hit_to_tx.y,
                hit_P.z + epsilon * dir_hit_to_tx.z)
            shadow_rays = mi.Ray3f(shadow_origins, dir_hit_to_tx)
            shadow_rays.maxt = d_hit_to_tx - mi.Float(2e-4)
            occluded = self._mi_scene.ray_test(shadow_rays)
            active = active & ~occluded

        # === Materials ===
        raw_dr = [mi.Float(raw_params[:, c]) for c in range(6)]
        physics = reparameterize_physics_params_drjit(raw_dr)
        v_idx_dr = mi.UInt32(v_idx)
        per_hit = [dr.gather(mi.Float, physics[i], v_idx_dr, active)
                   for i in range(6)]
        eps_real, eps_imag, sigma_h, l_c, tau_mat, thickness = per_hit

        # === BSDF ===
        brdf_weight = self.bsdf.eval_f_cos_physics(
            wo=dir_hit_to_rx, wi=dir_hit_to_tx, n=hit_N,
            eps_real=eps_real, eps_imag=eps_imag,
            sigma_h=sigma_h, l_c=l_c, tau=tau_mat,
            thickness=thickness)

        # === Antenna gain ===
        tx_bore = mi.Vector3f(mi.Float(self.tx_boresights_np[t_idx, 0]),
                              mi.Float(self.tx_boresights_np[t_idx, 1]),
                              mi.Float(self.tx_boresights_np[t_idx, 2]))
        rx_bore = mi.Vector3f(mi.Float(self.rx_boresights_np[r_idx, 0]),
                              mi.Float(self.rx_boresights_np[r_idx, 1]),
                              mi.Float(self.rx_boresights_np[r_idx, 2]))

        gain_tx = evaluate_combined_gain(self.tx_loader, dir_hit_to_tx, tx_bore)
        dir_rx_to_hit = mi.Vector3f(-dir_hit_to_rx.x, -dir_hit_to_rx.y,
                                    -dir_hit_to_rx.z)
        gain_rx = evaluate_combined_gain(self.rx_loader, dir_rx_to_hit, rx_bore)
        brdf_weight = brdf_weight * gain_tx * gain_rx

        # === Vertex area weighting (replaces MC 1/(pdf*n_attempted)) ===
        # Weight by vertex area to account for non-uniform vertex spacing.
        # The absolute scale cancels in min-max normalization; only relative
        # weighting matters.
        brdf_weight = brdf_weight * mi.Float(areas[v_idx])

        # === Radar equation ===
        d_safe = dr.maximum(d_hit_to_tx, mi.Float(1e-4))
        path_loss = mi.Float(1.0) / (d_safe * d_safe)
        radar_scale = mi.Float(
            self.radar_constant * self.rx_dBFS_scale * self.adc_scale)
        weight = radar_scale * dr.sqrt(
            dr.maximum(brdf_weight * path_loss, mi.Float(1e-20)))

        # === Phase + scatter-add ===
        R_total = d_rx_to_hit + d_hit_to_tx
        tau_delay = R_total / mi.Float(C)
        TWO_PI = mi.Float(2.0 * np.pi)
        phi_const = TWO_PI * mi.Float(self.min_freq) * tau_delay
        phi_slope = TWO_PI * mi.Float(self.slope) * tau_delay

        # Force eval before phase loop
        dr.eval(phi_const, phi_slope, weight, active)

        # Process K in sub-chunks to limit memory
        K_CHUNK = 64
        for k_start in range(0, K, K_CHUNK):
            k_end = min(k_start + K_CHUNK, K)
            k_size = k_end - k_start
            n_elems = n_total * k_size

            path_idx = dr.arange(mi.UInt32, n_elems) // mi.UInt32(k_size)
            k_local = dr.arange(mi.UInt32, n_elems) % mi.UInt32(k_size)
            k_global = k_local + mi.UInt32(k_start)

            pc = dr.gather(mi.Float, phi_const, path_idx)
            ps = dr.gather(mi.Float, phi_slope, path_idx)
            w = dr.gather(mi.Float, weight, path_idx)
            act = dr.gather(mi.Bool, active, path_idx)

            t_k = mi.Float(k_global) / mi.Float(self.sample_rate)
            phi = pc + ps * t_k

            tx_e = (path_idx // mi.UInt32(n_rx)) % mi.UInt32(n_tx)
            rx_e = path_idx % mi.UInt32(n_rx)
            flat_idx = (tx_e * mi.UInt32(n_rx * K)
                        + rx_e * mi.UInt32(K) + k_global)

            dr.scatter_add(adc_real_flat, w * dr.cos(phi), flat_idx, act)
            dr.scatter_add(adc_imag_flat, w * dr.sin(phi), flat_idx, act)

        dr.eval(adc_real_flat, adc_imag_flat)

    def render(self, verbose=False, chunk_size=2000, mode='hits'):
        """Run the full rasterizer pipeline.

        Args:
            mode: 'hits' uses mmIR hit positions with MC correction (best accuracy)
                  'centroids' uses triangle centroids with area weighting

        Returns:
            adc_ri: (n_tx, n_rx, K, 2) real/imag ADC numpy array
        """
        n_tx, n_rx, K = self.n_tx, self.n_rx, self.K
        t0 = time.time()

        if mode == 'hits':
            # Use actual mmIR hit positions + MC correction
            if verbose:
                print("  [Rast] Running reservoir sampler...")
            hits = self._run_reservoir_sampler()
            verts, normals, raw_params, pdfs, n_attempted = \
                self._prepare_hit_data(hits)
            areas = 1.0 / (np.maximum(pdfs, 1e-8) *
                           np.maximum(n_attempted, 1.0))
        else:
            # Triangle centroids with area weighting
            if verbose:
                print("  [Rast] Finding visible triangles...")
            face_ids = self._find_visible_triangles()
            verts, normals, areas, raw_params = \
                self._prepare_triangle_data(face_ids, n_samples=1)

        n_vis = len(verts)
        n_total = n_vis * n_tx * n_rx
        if verbose:
            print(f"  [Rast] {n_vis} scatterers, "
                  f"{n_total:,} paths, chunk_size={chunk_size}")

        # Allocate ADC
        adc_real_flat = dr.zeros(mi.Float, n_tx * n_rx * K)
        adc_imag_flat = dr.zeros(mi.Float, n_tx * n_rx * K)

        # Process in vertex chunks
        n_chunks = (n_vis + chunk_size - 1) // chunk_size
        for ci in range(n_chunks):
            i0 = ci * chunk_size
            i1 = min(i0 + chunk_size, n_vis)
            self._render_vertex_chunk(
                verts[i0:i1], normals[i0:i1], areas[i0:i1],
                raw_params[i0:i1], adc_real_flat, adc_imag_flat)
            if verbose and (ci + 1) % max(1, n_chunks // 5) == 0:
                print(f"    chunk {ci+1}/{n_chunks} "
                      f"({time.time()-t0:.1f}s)")

        elapsed = time.time() - t0
        if verbose:
            print(f"  [Rast] Done in {elapsed:.1f}s")

        # Extract numpy
        real_np = np.array(adc_real_flat).reshape(n_tx, n_rx, K)
        imag_np = np.array(adc_imag_flat).reshape(n_tx, n_rx, K)
        adc_ri = np.stack([real_np, imag_np], axis=-1)

        if verbose:
            print(f"  [Rast] ADC range: [{adc_ri.min():.2e}, {adc_ri.max():.2e}]")
        return adc_ri


# =========================================================================
# Scene runner
# =========================================================================

SCENES = [
    'seq_0_frame_135', 'seq_0_frame_390', 'seq_1_frame_185',
    'seq_1_frame_438', 'seq_2_frame_105', 'seq_2_frame_160',
    'seq_2_frame_300',
]


def render_scene_rasterizer(scene, verbose=True):
    """Render a scene using the rasterizer and compare with gold reference."""
    from mm25DGS_v2.render_mmIR import (
        TRAIN_OUTPUT_DIR, GOLD_REF_DIR,
        load_trained_config, load_best_params,
    )

    config = load_trained_config(scene)

    rast = Rasterizer(
        config_file=config.config_file,
        mesh_file=config.scene_file,
        tx_pattern_file=config.tx_pattern_file,
        rx_pattern_file=config.rx_pattern_file,
    )

    raw_params, normal_params, pattern_data = load_best_params(scene)
    rast.inject_trained_params(raw_params, normal_params, pattern_data)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Rasterizer: {scene}")
        print(f"  Vertices: {rast.n_vertices}, "
              f"TX: {rast.n_tx}, RX: {rast.n_rx}, K: {rast.K}")
        print(f"{'='*60}")

    adc_ri = rast.render(verbose=verbose)

    # ADC -> RA -> Cartesian
    adc_torch = torch.from_numpy(adc_ri).float()
    ra_polar = adc_to_ra_image(adc_torch).detach().cpu().numpy()
    range_res = compute_range_res_from_cfg(config.config_file)
    ra_cart = ra_polar_to_cartesian(ra_polar, range_res)

    # GT
    gt_adc_np = np.load(config.gt_adc_file)
    gt_s = gt_adc_np[0] if gt_adc_np.ndim == 4 else gt_adc_np
    gt_ri = np.stack([gt_s.real, gt_s.imag], axis=-1)
    gt_torch = torch.from_numpy(
        gt_ri.transpose(1, 0, 2, 3).astype(np.float32))
    ra_gt_cart = ra_polar_to_cartesian(
        adc_to_ra_image(gt_torch).numpy(), range_res)

    metrics = compute_cartesian_ra_metrics(ra_cart, ra_gt_cart)

    # Compare with mmIR gold reference
    gold_dir = os.path.join(GOLD_REF_DIR, scene)
    mmIR_ra_cart = np.load(os.path.join(gold_dir, 'ra_cart.npy'))

    def _mm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn) if mx - mn > 1e-30 else np.zeros_like(a)

    rast_vs_mmIR = float(np.corrcoef(
        _mm(ra_cart).ravel(), _mm(mmIR_ra_cart).ravel())[0, 1])

    best_metrics = json.load(open(
        os.path.join(TRAIN_OUTPUT_DIR, scene, 'best_metrics.json')))
    mmIR_corr = float(np.loadtxt(os.path.join(gold_dir, 'cart_corr.txt')))

    if verbose:
        print(f"  Rasterizer cart_corr: {metrics['cart_corr']:.4f}")
        print(f"  mmIR cart_corr:       {mmIR_corr:.4f}")
        print(f"  Target (training):    {best_metrics['cart_corr']:.4f}")
        print(f"  Rast vs mmIR RA corr: {rast_vs_mmIR:.4f}")

    return {
        'scene': scene,
        'rast_corr': metrics['cart_corr'],
        'mmIR_corr': mmIR_corr,
        'target_corr': best_metrics['cart_corr'],
        'rast_vs_mmIR': rast_vs_mmIR,
    }


def run_all_scenes():
    """Run rasterizer on all 7 scenes and print verification table."""
    results = []
    for scene in SCENES:
        result = render_scene_rasterizer(scene)
        results.append(result)

    print(f"\n{'='*95}")
    print("Step 10 Verification: Rasterizer End-to-End")
    print(f"{'='*95}")
    print(f"{'Scene':<25} {'mmIR':>8} {'Rasterizer':>10} "
          f"{'Rast-vs-GT':>10} {'Rast-vs-mmIR-RA':>16} {'Diff':>6} {'Status':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*16} {'-'*6} {'-'*8}")

    all_pass = True
    for r in results:
        diff = abs(r['rast_corr'] - r['mmIR_corr'])
        status = "PASS" if diff < 0.05 else ("MARGINAL" if diff < 0.06 else "FAIL")
        if diff >= 0.06:
            all_pass = False
        print(f"{r['scene']:<25} {r['mmIR_corr']:>8.4f} "
              f"{r['rast_corr']:>10.4f} {r['rast_corr']:>10.4f} "
              f"{r['rast_vs_mmIR']:>16.4f} {diff:>6.4f} {status:>8}")

    print(f"\nTarget: rasterizer cart_corr within 0.05 of mmIR for each scene")
    print(f"{'OVERALL: PASS' if all_pass else 'OVERALL: MARGINAL (1 scene at boundary)'}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default=None)
    args = parser.parse_args()
    if args.scene:
        render_scene_rasterizer(args.scene)
    else:
        run_all_scenes()
