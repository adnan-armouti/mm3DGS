import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

try:
    import open3d as o3d
except Exception:
    o3d = None


def extract_board_frame_from_config(config_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract orthonormal board frame [azimuth, range, elevation] from dense antenna config.

    Uses PCA on antenna positions projected to the board plane to find the principal
    directions (horizontal and vertical axes). This recovers the exact frame used when
    the dense array was generated.

    The dense array structure:
    - TX: Two vertical columns (50 per column for 100 TX)
    - RX: Two horizontal rows (50 per row for 100 RX)
    - Antennas lie on a plane perpendicular to boresight

    Args:
        config_path: Path to config JSON file

    Returns:
        v2_azimuth: Horizontal axis (azimuth/right direction) [3]
        y_range: Boresight axis (range/forward direction) [3]
        v1_elevation: Vertical axis (elevation/up direction) [3]

    Coordinate mapping:
        Default radar frame -> Board frame
        X (azimuth)        -> v2 (horizontal on board)
        Y (range)          -> boresight (normal to board)
        Z (elevation)      -> v1 (vertical on board)
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # Extract antenna positions (in meters)
    tx_pos = np.array([t['pos_mm'] for t in cfg['tx_array']], dtype=float) / 1000.0
    rx_pos = np.array([r['pos_mm'] for r in cfg['rx_array']], dtype=float) / 1000.0
    all_pos = np.vstack([tx_pos, rx_pos])

    # Compute board center (geometric center of all antennas)
    center = np.mean(all_pos, axis=0)

    # Extract boresight from config (normal to board plane)
    boresight = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
    boresight /= np.linalg.norm(boresight)

    # Project all antenna positions onto board plane
    # Remove component along boresight to get in-plane vectors
    rel_pos = all_pos - center  # Relative to center
    plane_proj = rel_pos - np.outer(np.dot(rel_pos, boresight), boresight)

    # PCA on projected positions to find principal directions
    # This gives us the axes of maximum variance (horizontal and vertical on board)
    cov = plane_proj.T @ plane_proj
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest variance first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Principal directions on board plane
    v_principal1 = eigenvectors[:, 0]  # Largest variance
    v_principal2 = eigenvectors[:, 1]  # Second largest variance

    # Normalize
    v_principal1 /= np.linalg.norm(v_principal1)
    v_principal2 /= np.linalg.norm(v_principal2)

    # Determine which principal component corresponds to horizontal vs vertical
    # TX antennas form vertical columns, RX form horizontal rows
    # Check which direction has more TX variation
    tx_rel = tx_pos - center
    tx_proj = tx_rel - np.outer(np.dot(tx_rel, boresight), boresight)

    # Variance of TX along each principal direction
    tx_var1 = np.var(np.dot(tx_proj, v_principal1))
    tx_var2 = np.var(np.dot(tx_proj, v_principal2))

    # TX columns are vertical, so larger TX variance -> vertical axis
    if tx_var1 > tx_var2:
        v1_elevation = v_principal1  # Vertical (elevation)
        v2_azimuth = v_principal2    # Horizontal (azimuth)
    else:
        v1_elevation = v_principal2  # Vertical (elevation)
        v2_azimuth = v_principal1    # Horizontal (azimuth)

    # Ensure right-handed coordinate system: v2 x boresight = v1
    # If not, flip v2
    cross_check = np.cross(v2_azimuth, boresight)
    if np.dot(cross_check, v1_elevation) < 0:
        v2_azimuth = -v2_azimuth

    return v2_azimuth, boresight, v1_elevation

def _try_read_ascii_ply_intensity(ply_path: str) -> Optional[np.ndarray]:
    """Best-effort parser for ASCII PLY with a float 'intensity' vertex property.
    Returns a 1D float array if successful, else None.
    """
    try:
        with open(ply_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_lines: List[str] = []
            line = f.readline()
            if not line or not line.strip().lower().startswith('ply'):
                return None
            header_lines.append(line)
            vertex_count = None
            props: List[str] = []
            while True:
                line = f.readline()
                if not line:
                    return None
                header_lines.append(line)
                s = line.strip()
                if s.lower().startswith('format') and 'ascii' not in s.lower():
                    # Only handle ASCII here
                    return None
                if s.lower().startswith('element vertex'):
                    try:
                        vertex_count = int(s.split()[-1])
                    except Exception:
                        vertex_count = None
                if s.lower().startswith('property'):
                    toks = s.split()
                    if len(toks) >= 3:
                        # property <type> <name>
                        props.append(toks[-1])
                if s.lower() == 'end_header':
                    break
            if vertex_count is None or vertex_count <= 0 or not props:
                return None
            # Find intensity column (case-insensitive)
            col_idx = None
            for i, p in enumerate(props):
                if p.lower() == 'intensity':
                    col_idx = i
                    break
            if col_idx is None:
                return None
            # Read remainder quickly with numpy
            import numpy as _np
            data = _np.loadtxt(f, dtype=_np.float64, max_rows=vertex_count)
            if data.ndim == 1:
                # Single column edge case
                if col_idx != 0:
                    return None
                vals = data
            else:
                if col_idx < 0 or col_idx >= data.shape[1]:
                    return None
                vals = data[:, col_idx]
            if vals.shape[0] != vertex_count:
                return None
            return vals.astype(_np.float64)
    except Exception:
        return None

from .single_view_proc import (
    # primitives
    to_tensor,
    get_hann,
    txrx_to_vx_chirps_dense_gpu_batched,
    process_adc_to_ra_map_enhanced,
    plot_range_azimuth_heatmap_img,
    make_angle_grids_np,
    _centers_to_edges,
    build_point_cloud,
    load_point_cloud,
    get_pcl_intensities,
    create_colored_sphere,
    create_axis_arrow,
    _compute_range_resolution_from_config,
)
from .eval_single_view import evaluate_radar_vs_lidar_metrics

# Import RA utils for cartesian conversion and image saving
from mmir.data.ra_utils import ra_polar_to_cartesian, save_ra_image


def load_config_positions_and_boresight(config_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if config_path is None:
        return None, None, None, None
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        tx_m = np.array([t['pos_mm'] for t in cfg['tx_array']], dtype=float) / 1000.0
        rx_m = np.array([r['pos_mm'] for r in cfg['rx_array']], dtype=float) / 1000.0
        center = np.mean(np.vstack([tx_m, rx_m]), axis=0)
        bore = None
        if 'boresight' in cfg['tx_array'][0]:
            bore = np.array(cfg['tx_array'][0]['boresight'], dtype=float)
            n = float(np.linalg.norm(bore))
            if n > 1e-9:
                bore = bore / n
            else:
                bore = None
        return tx_m, rx_m, center, bore
    except Exception:
        return None, None, None, None


def load_runtime_config(explicit_config_path: Optional[str], output_directory: Path) -> Dict[str, Any]:
    params = {
        'range_resolution': 0.117,
        'num_adc': 256,
        'num_ant': 100,
        'num_az_bins': 128,
        'num_el_bins': 128,
    }
    cfg0 = None
    if explicit_config_path is not None:
        p = Path(explicit_config_path)
        if p.exists():
            cfg0 = p
    if cfg0 is None:
        cfg_dir = output_directory / '01_configs'
        files = sorted(cfg_dir.glob('config_angle_*.json'))
        if files:
            cfg0 = files[0]
    if cfg0 is None:
        return params
    try:
        dr = _compute_range_resolution_from_config(cfg0)
        if dr > 0:
            params['range_resolution'] = float(dr)
        with open(cfg0, 'r') as f:
            cfg = json.load(f)
        if 'numAdcSamples' in cfg:
            params['num_adc'] = int(cfg['numAdcSamples'])
        n_tx = len(cfg.get('tx_array', []))
        n_rx = len(cfg.get('rx_array', []))
        if n_tx > 0 and n_tx == n_rx:
            params['num_ant'] = int(n_tx)
        return params
    except Exception:
        return params


def compute_rae_cube(adc_file: Path, params: Dict[str, Any], device: torch.device) -> np.ndarray:
    adc = np.load(str(adc_file))
    adc_c = adc[:, :, :, 0] + 1j * adc[:, :, :, 1]
    adc_c = np.expand_dims(adc_c, axis=0)
    adc_c = np.transpose(adc_c, (0, 2, 1, 3))
    adc_t = to_tensor(adc_c, device=device, dtype=torch.complex64)
    vx = txrx_to_vx_chirps_dense_gpu_batched(adc_t, num_ant=int(params['num_ant']))
    h_range = get_hann(int(params['num_adc']), vx.device)
    vx = vx * h_range.view(1, 1, 1, -1)
    vx = torch.fft.fft(vx, n=int(params['num_adc']), dim=-1)
    vol = vx
    # Swap FFT dimensions: dim 1 for azimuth, dim 2 for elevation
    h_az = get_hann(vol.shape[1], vx.device)  # Changed from shape[2] to shape[1]
    vol = vol * h_az.view(1, -1, 1, 1)         # Changed view pattern
    vol = torch.fft.ifftshift(vol, dim=1)      # Changed from dim=2 to dim=1
    vol = torch.fft.fft(vol, n=int(params['num_az_bins']), dim=1)  # Changed from dim=2 to dim=1
    vol = vol[:, 1:, :, :]                     # Changed from [:, :, 1:, :] to [:, 1:, :, :]
    vol = torch.fft.fftshift(vol, dim=1)       # Changed from dim=2 to dim=1
    h_el = get_hann(vol.shape[2], vx.device)  # Changed from shape[1] to shape[2]
    vol = vol * h_el.view(1, 1, -1, 1)         # Changed view pattern
    vol = torch.fft.ifftshift(vol, dim=2)      # Changed from dim=1 to dim=2
    vol = torch.fft.fft(vol, n=int(params['num_el_bins']), dim=2)  # Changed from dim=1 to dim=2
    vol = vol[:, :, 1:, :]                     # Changed from [:, 1:, :, :] to [:, :, 1:, :]
    vol = torch.fft.fftshift(vol, dim=2)       # Changed from dim=1 to dim=2
    mag = torch.abs(vol[0]).to(torch.float32).cpu().numpy()  # (Az, El, R) - swapped due to FFT dimension swap
    rae = mag  # Already in (Az, El, R) format, no transpose needed
    return rae


def collapse_to_ra(rae: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra_map = np.sum(rae, axis=1)
    num_az = int(params['num_az_bins'])
    t = np.arange(-num_az // 2 + 1, num_az // 2) * (2.0 / num_az)
    az_deg = np.degrees(np.arcsin(t))[::-1]
    r_m = np.arange(int(params['num_adc'])) * float(params['range_resolution'])
    return ra_map, az_deg, r_m


def radar_points_from_rae(rae: np.ndarray, params: Dict[str, Any], near_field_m: float, percentile: Optional[float]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    R = int(params['num_adc'])
    num_az_full = int(params['num_az_bins'])
    num_el_full = int(params['num_el_bins'])
    t_az = np.arange(-num_az_full//2 + 1, num_az_full//2) * (2.0 / num_az_full)
    t_el = np.arange(-num_el_full//2 + 1, num_el_full//2) * (2.0 / num_el_full)
    az_angles = np.arcsin(np.clip(t_az, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    el_angles = np.arcsin(np.clip(t_el, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    mag = rae  # (Az, El, R)
    Rg, Ag, Eg = np.meshgrid(np.arange(mag.shape[2]), np.arange(mag.shape[0]), np.arange(mag.shape[1]), indexing='ij')
    r_vals = (Rg.ravel().astype(np.float32)) * float(params['range_resolution'])
    az_vals = az_angles[Ag.ravel()]
    el_vals = el_angles[Eg.ravel()]
    x =  r_vals * np.cos(el_vals) * np.sin(az_vals)  # Fixed: removed negative sign
    y =  r_vals * np.cos(el_vals) * np.cos(az_vals)
    z =  r_vals * np.sin(el_vals)                     # Fixed: removed negative sign
    intensity = mag.transpose(2, 0, 1).ravel().astype(np.float32)
    rng = np.sqrt(x*x + y*y + z*z)
    keep = rng >= float(near_field_m)
    x = x[keep]; y = y[keep]; z = z[keep]; intensity = intensity[keep]
    keys = None
    if percentile is not None:
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = float(np.percentile(intensity, p))
        keep2 = intensity >= thr if np.any(intensity >= thr) else intensity >= intensity.max() if intensity.size else np.array([], dtype=bool)
        x = x[keep2]; y = y[keep2]; z = z[keep2]; intensity = intensity[keep2]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, intensity, keys


def load_and_align_lidar(lidar_pcl_path: Optional[str], cfg_center: Optional[np.ndarray]) -> Optional[o3d.geometry.PointCloud]:
    if o3d is None or lidar_pcl_path is None:
        return None
    pcd = load_point_cloud(lidar_pcl_path)
    if pcd is None or (not pcd.has_points()):
        return None
    # LiDAR is already in world coordinates (matching mesh frame), so do NOT translate
    # The voxelization code will handle coordinate transforms as needed
    return pcd


def voxelize_lidar_to_radar_grid(
    pcd: o3d.geometry.PointCloud,
    params: Dict[str, Any],
    near_field_m: float,
    percentile: float,
    cfg_bore: Optional[np.ndarray] = None,
    agg_method: str = 'median',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_cent, el_cent = make_angle_grids_np(int(params['num_az_bins']), int(params['num_el_bins']))
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))
    pts_world = np.asarray(pcd.points, dtype=np.float32)
    # Apply inverse boresight to bring LiDAR into radar default (+Y) frame before binning
    if cfg_bore is not None:
        try:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
            b = cfg_bore.astype(float)
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            if abs(dot - 1.0) < 1e-8:
                R_inv = np.eye(3)
            elif abs(dot + 1.0) < 1e-8:
                axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = axis - np.dot(axis, a) * a
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                ux, uy, uz = axis
                c = -1.0; s = 0.0
                R_forward = np.array([
                    [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                ], dtype=float)
                R_inv = R_forward.T
            else:
                axis = np.cross(a, b)
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                ang = np.arccos(dot)
                ux, uy, uz = axis
                c = np.cos(ang); s = np.sin(ang)
                R_forward = np.array([
                    [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                ], dtype=float)
                R_inv = R_forward.T
            pts_world = (pts_world @ R_inv.T).astype(np.float32)
        except Exception:
            pass
    intens = get_pcl_intensities(pcd).astype(np.float32)
    if intens.shape[0] != pts_world.shape[0]:
        intens = np.ones((pts_world.shape[0],), dtype=np.float32)
    # Compute spherical in sensor default coordinates (assuming already centered)
    x = pts_world[:, 0]; y = pts_world[:, 1]; z = pts_world[:, 2]
    r = np.linalg.norm(pts_world, axis=1)
    keep = (r >= float(near_field_m)) & np.isfinite(r)
    x = x[keep]; y = y[keep]; z = z[keep]; r = r[keep]; intens = intens[keep]
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    az = np.arctan2(x, y)  # Fixed: removed negative sign to match spherical-to-Cartesian conversion
    el = np.arcsin(np.clip(z / np.clip(r, 1e-12, None), -1.0, 1.0))  # Fixed: removed negative sign
    az_idx = np.searchsorted(az_edges, az, side='right') - 1
    el_idx = np.searchsorted(el_edges, el, side='right') - 1
    r_idx  = np.searchsorted(r_edges,  r,  side='right') - 1
    A = az_cent.size; E = el_cent.size; Rb = int(params['num_adc'])
    # Strict validity mask (match 09_v2): discard out-of-range indices
    valid = (az_idx >= 0) & (az_idx < A) & (el_idx >= 0) & (el_idx < E) & (r_idx >= 0) & (r_idx < Rb)
    az_idx = az_idx[valid]; el_idx = el_idx[valid]; r_idx = r_idx[valid]; intens = intens[valid]
    if intens.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    keys = (az_idx.astype(np.int64) * E + el_idx.astype(np.int64)) * Rb + r_idx.astype(np.int64)
    order = np.argsort(keys)
    k_sorted = keys[order]
    v_sorted = intens[order]
    boundaries = np.where(np.diff(k_sorted) != 0)[0] + 1
    groups = np.split(v_sorted, boundaries)
    # Aggregate each voxel's intensities per requested method (default: median)
    if str(agg_method).lower() == 'mean':
        agg_vals = np.array([float(np.mean(g)) for g in groups], dtype=np.float32)
    else:
        agg_vals = np.array([float(np.median(g)) for g in groups], dtype=np.float32)
    unique_keys = k_sorted[np.r_[0, boundaries]]
    az_u = (unique_keys // (E * Rb)).astype(np.int32)
    rem = unique_keys % (E * Rb)
    el_u = (rem // Rb).astype(np.int32)
    r_u = (rem % Rb).astype(np.int32)
    az_c = az_cent[az_u].astype(np.float32)
    el_c = el_cent[el_u].astype(np.float32)
    r_c = (r_u.astype(np.float32) * float(params['range_resolution'])).astype(np.float32)
    x_s =  r_c * np.cos(el_c) * np.sin(az_c)  # Fixed: removed negative sign
    y_s =  r_c * np.cos(el_c) * np.cos(az_c)
    z_s =  r_c * np.sin(el_c)                  # Fixed: removed negative sign
    centers = np.stack([x_s, y_s, z_s], axis=1).astype(np.float32)
    return centers, agg_vals, unique_keys.astype(np.int64)


def load_mesh(path: Optional[str]) -> Optional['o3d.geometry.TriangleMesh']:
    if o3d is None or path is None:
        return None
    try:
        mesh_o3d = o3d.io.read_triangle_mesh(path)
        if mesh_o3d is not None and mesh_o3d.has_triangles():
            mesh_o3d.compute_vertex_normals()
            return mesh_o3d
    except Exception:
        return None
    return None


def build_fov_wireframe(params: Dict[str, Any], range_step: int, grid_step_az: int, grid_step_el: int,
                        cfg_center: Optional[np.ndarray], cfg_bore: Optional[np.ndarray], orient_to_config: bool,
                        config_path: Optional[str] = None) -> Optional['o3d.geometry.LineSet']:
    if o3d is None:
        return None
    try:
        def _make_angles(az_bins_full: int, el_bins_full: int):
            eps = 1e-6
            t_az = np.arange(-az_bins_full // 2 + 1, az_bins_full // 2, dtype=np.float64) * (2.0 / float(az_bins_full))
            t_el = np.arange(-el_bins_full // 2 + 1, el_bins_full // 2, dtype=np.float64) * (2.0 / float(el_bins_full))
            t_az = np.clip(t_az, -1.0 + eps, 1.0 - eps)
            t_el = np.clip(t_el, -1.0 + eps, 1.0 - eps)
            return np.arcsin(t_az), np.arcsin(t_el)

        def _sph_to_xyz(r_vals, az_vec, el_vec):
            x =  r_vals * np.cos(el_vec) * np.sin(az_vec)  # Fixed: removed negative sign
            y =  r_vals * np.cos(el_vec) * np.cos(az_vec)
            z =  r_vals * np.sin(el_vec)                    # Fixed: removed negative sign
            return x, y, z

        az_ang, el_ang = _make_angles(int(params['num_az_bins']), int(params['num_el_bins']))
        r_axis = np.arange(int(params['num_adc']), dtype=np.float64) * float(params['range_resolution'])
        num_az = az_ang.shape[0]
        num_el = el_ang.shape[0]
        az_idx = np.arange(0, num_az, max(1, int(grid_step_az)))
        el_idx = np.arange(0, num_el, max(1, int(grid_step_el)))
        if num_az > 0 and (az_idx.size == 0 or az_idx[-1] != (num_az - 1)):
            az_idx = np.unique(np.append(az_idx, num_az - 1))
        if num_el > 0 and (el_idx.size == 0 or el_idx[-1] != (num_el - 1)):
            el_idx = np.unique(np.append(el_idx, num_el - 1))
        r_idx = np.arange(0, r_axis.shape[0], max(1, int(range_step)))
        pts = []
        segs = []
        for k in r_idx:
            r = r_axis[k]
            if r <= 0:
                continue
            # rings at constant elevation
            for j in el_idx:
                az_vec = az_ang
                el_vec = np.full_like(az_vec, el_ang[j])
                r_vec = np.full_like(az_vec, r)
                x_, y_, z_ = _sph_to_xyz(r_vec, az_vec, el_vec)
                start = len(pts)
                for ii in range(az_vec.shape[0]):
                    pts.append([x_[ii], y_[ii], z_[ii]])
                for ii in range(az_vec.shape[0] - 1):
                    segs.append([start + ii, start + ii + 1])
                if az_vec.shape[0] > 1:
                    segs.append([start + az_vec.shape[0] - 1, start + 0])
            # columns at constant azimuth
            for i_a in az_idx:
                el_vec = el_ang
                az_vec = np.full_like(el_vec, az_ang[i_a])
                r_vec = np.full_like(el_vec, r)
                x_, y_, z_ = _sph_to_xyz(r_vec, az_vec, el_vec)
                start = len(pts)
                for jj in range(el_vec.shape[0]):
                    pts.append([x_[jj], y_[jj], z_[jj]])
                for jj in range(el_vec.shape[0] - 1):
                    segs.append([start + jj, start + jj + 1])
                if el_vec.shape[0] > 1:
                    segs.append([start + el_vec.shape[0] - 1, start + 0])
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            return None
        # Align to board frame (PCA-based) then translate, if requested
        if orient_to_config and cfg_bore is not None and config_path is not None:
            try:
                # Extract full board frame from config (same as radar points)
                v2_azimuth, y_range, v1_elevation = extract_board_frame_from_config(config_path)

                # Build rotation matrix
                R_align = np.column_stack([v2_azimuth, y_range, v1_elevation])

                # Apply rotation
                pts = (R_align @ pts.T).T
            except Exception as e:
                print(f"[WIREFRAME] Failed to apply board frame rotation: {e}")
                # Fall back to identity (no rotation)
                pass
        if cfg_center is not None:
            pts = pts + cfg_center.reshape(1, 3)
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts.astype(np.float64)),
            lines=o3d.utility.Vector2iVector(np.asarray(segs, dtype=np.int32)),
        )
        col = np.array([0.9, 0.9, 0.9], dtype=np.float64)
        ls.colors = o3d.utility.Vector3dVector(np.repeat(col[None, :], len(segs), axis=0))
        return ls
    except Exception:
        return None


def optional_mesh_proximity_filter(centers_world: np.ndarray, vals: np.ndarray, keys: np.ndarray,
                                   mesh: Optional['o3d.geometry.TriangleMesh'], eps_m: float,
                                   samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mesh is None or centers_world.shape[0] == 0 or float(eps_m) <= 0.0:
        return centers_world, vals, keys
    try:
        samples = max(1000, int(samples))
    except Exception:
        samples = 200000
    try:
        mesh_samples = mesh.sample_points_uniformly(number_of_points=samples)
        mesh_pts = np.asarray(mesh_samples.points, dtype=np.float32)
    except Exception:
        return centers_world, vals, keys
    if mesh_pts.shape[0] == 0:
        return centers_world, vals, keys
    try:
        import faiss
        index = faiss.IndexFlatL2(3)
        index.add(mesh_pts.astype(np.float32))
        D, _ = index.search(centers_world.astype(np.float32), 1)
        dist = np.sqrt(D[:, 0])
    except Exception:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(mesh_pts.astype(np.float32))
            dist, _ = tree.query(centers_world.astype(np.float32), k=1)
        except Exception:
            return centers_world, vals, keys
    keep = dist <= float(eps_m)
    if not np.any(keep):
        return centers_world, vals, keys
    return centers_world[keep], vals[keep], keys[keep]


def build_scene(geoms: List[Any], add_axes_at: Optional[np.ndarray]) -> None:
    if o3d is None:
        return

    # Add coordinate frame if requested
    if add_axes_at is not None:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    # Use remote visualization (WebRTC for SSH tunnel support)
    # Import the remote_viz utility (optional — only for interactive debugging)
    try:
        from mmir.remote_viz.viz_utils import visualize_geometries
        visualize_geometries(geoms, remote_viz=True, title='Single View - Component Visualization')
    except ImportError:
        print("  [WARN] mmir.remote_viz not available, skipping visualization")


def prepare_eval_artifacts(
    adc_file: str,
    config_path: Optional[str],
    lidar_pcl_path: Optional[str],
    mesh_path: Optional[str],
    output_directory: Path,
    device: torch.device,
    near_field_m: float,
    viz_percentile: float,
) -> Dict[str, Any]:
    params = load_runtime_config(config_path, output_directory)
    rae = compute_rae_cube(Path(adc_file), params, device)
    # Build radar points and keys for evaluation
    num_az_full = int(params['num_az_bins'])
    num_el_full = int(params['num_el_bins'])
    t_az = np.arange(-num_az_full // 2 + 1, num_az_full // 2) * (2.0 / num_az_full)
    t_el = np.arange(-num_el_full // 2 + 1, num_el_full // 2) * (2.0 / num_el_full)
    az_angles = np.arcsin(np.clip(t_az, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    el_angles = np.arcsin(np.clip(t_el, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    mag = rae  # (Az, El, R)
    Rg, Ag, Eg = np.meshgrid(np.arange(mag.shape[2]), np.arange(mag.shape[0]), np.arange(mag.shape[1]), indexing='ij')
    r_vals = (Rg.ravel().astype(np.float32)) * float(params['range_resolution'])
    az_vals = az_angles[Ag.ravel()]
    el_vals = el_angles[Eg.ravel()]
    x =  r_vals * np.cos(el_vals) * np.sin(az_vals)  # Fixed: removed negative sign
    y =  r_vals * np.cos(el_vals) * np.cos(az_vals)
    z =  r_vals * np.sin(el_vals)                     # Fixed: removed negative sign
    intensity = mag.transpose(2, 0, 1).ravel().astype(np.float32)
    rng = np.sqrt(x*x + y*y + z*z)
    keep_nf = rng >= float(near_field_m)
    x = x[keep_nf]; y = y[keep_nf]; z = z[keep_nf]; intensity = intensity[keep_nf]
    ridx_flat = Rg.ravel().astype(np.int32)[keep_nf]
    azidx_flat = Ag.ravel().astype(np.int32)[keep_nf]
    elidx_flat = Eg.ravel().astype(np.int32)[keep_nf]
    if intensity.size > 0:
        thr = float(np.percentile(intensity, float(viz_percentile)))
        keep_thr = intensity >= thr if np.any(intensity >= thr) else intensity >= intensity.max()
    else:
        keep_thr = np.zeros((0,), dtype=bool)
    x = x[keep_thr]; y = y[keep_thr]; z = z[keep_thr]; intensity = intensity[keep_thr]
    azidx_flat = azidx_flat[keep_thr]; elidx_flat = elidx_flat[keep_thr]; ridx_flat = ridx_flat[keep_thr]
    radar_keys = ((azidx_flat.astype(np.int64) * num_el_full).astype(np.int64) * int(params['num_adc']) \
                  + elidx_flat.astype(np.int64) * int(params['num_adc']) \
                  + ridx_flat.astype(np.int64))
    order_r = np.argsort(radar_keys)
    k_sorted_r = radar_keys[order_r]
    v_sorted_r = intensity[order_r]
    boundaries_r = np.where(np.diff(k_sorted_r) != 0)[0] + 1
    groups_r = np.split(v_sorted_r, boundaries_r)
    radar_vals_agg = np.array([float(np.percentile(g, 95.0)) for g in groups_r], dtype=np.float32)
    radar_keys_uniq = k_sorted_r[np.r_[0, boundaries_r]].astype(np.int64)
    radar_pcd = None
    if o3d is not None and x.size > 0:
        radar_pcd = build_point_cloud(x, y, z, intensity, percentile=0.0, color_by_intensity=True)
    # Config transforms
    tx_m, rx_m, cfg_center, cfg_bore = load_config_positions_and_boresight(config_path)
    if radar_pcd is not None and cfg_bore is not None:
        try:
            a = np.array([0.0, 1.0, 0.0], dtype=float)
            b = cfg_bore.astype(float)
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            if abs(dot - 1.0) < 1e-8:
                R_align = np.eye(3)
            elif abs(dot + 1.0) < 1e-8:
                axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = axis - np.dot(axis, a) * a
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                ux, uy, uz = axis
                c = -1.0; s = 0.0
                R_align = np.array([
                    [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                ], dtype=float)
            else:
                axis = np.cross(a, b)
                axis = axis / (np.linalg.norm(axis) + 1e-12)
                ang = np.arccos(dot)
                ux, uy, uz = axis
                c = np.cos(ang); s = np.sin(ang)
                R_align = np.array([
                    [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                ], dtype=float)
            pts = np.asarray(radar_pcd.points)
            pts = (pts @ R_align.T)
            if cfg_center is not None:
                pts = pts + cfg_center.reshape(1, 3)
            radar_pcd.points = o3d.utility.Vector3dVector(pts)
        except Exception:
            pass
    # LiDAR voxelization artifacts
    mesh_o3d = load_mesh(mesh_path)
    lidar_pcd_voxel = None
    lidar_keys = np.zeros((0,), dtype=np.int64)
    agg_vals = np.zeros((0,), dtype=np.float32)
    lidar_pcd_raw = None
    if lidar_pcl_path is not None and o3d is not None:
        raw = load_point_cloud(lidar_pcl_path)
        if raw is not None and raw.has_points():
            lidar_pcd_raw = o3d.geometry.PointCloud(raw)
            if cfg_center is not None:
                try:
                    lidar_pcd_raw.translate(cfg_center.tolist())
                except Exception:
                    pass
            pcd_cent = o3d.geometry.PointCloud(raw)
            if cfg_center is not None:
                try:
                    pcd_cent.translate((-cfg_center).tolist())
                except Exception:
                    pass
            centers, agg_vals, lidar_keys = voxelize_lidar_to_radar_grid(
                pcd_cent, params, near_field_m=near_field_m, percentile=float(viz_percentile), cfg_bore=cfg_bore
            )
            # forward rotate + translate
            if centers.shape[0] > 0 and cfg_bore is not None:
                try:
                    a = np.array([0.0, 1.0, 0.0], dtype=float)
                    b = cfg_bore.astype(float)
                    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
                    if abs(dot - 1.0) < 1e-8:
                        R_align2 = np.eye(3)
                    elif abs(dot + 1.0) < 1e-8:
                        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                        axis = axis - np.dot(axis, a) * a
                        axis = axis / (np.linalg.norm(axis) + 1e-12)
                        ux, uy, uz = axis
                        c = -1.0; s = 0.0
                        R_align2 = np.array([
                            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                        ], dtype=float)
                    else:
                        axis = np.cross(a, b)
                        axis = axis / (np.linalg.norm(axis) + 1e-12)
                        ang = np.arccos(dot)
                        ux, uy, uz = axis
                        c = np.cos(ang); s = np.sin(ang)
                        R_align2 = np.array([
                            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                        ], dtype=float)
                    centers = (centers @ R_align2.T).astype(np.float32)
                except Exception:
                    pass
            if cfg_center is not None and centers.shape[0] > 0:
                centers = centers + cfg_center.reshape(1, 3).astype(np.float32)
            if mesh_o3d is not None and centers.shape[0] > 0:
                centers, agg_vals, lidar_keys = optional_mesh_proximity_filter(centers, agg_vals, lidar_keys, mesh_o3d, eps_m=0.0, samples=200000)
            if centers.shape[0] > 0:
                lidar_pcd_voxel = build_point_cloud(centers[:,0], centers[:,1], centers[:,2], agg_vals, percentile=float(viz_percentile), color_by_intensity=True)
    return {
        'radar_pcd': radar_pcd if (o3d is None or radar_pcd is not None) else o3d.geometry.PointCloud(),
        'lidar_pcd_raw': lidar_pcd_raw if (o3d is None or lidar_pcd_raw is not None) else o3d.geometry.PointCloud(),
        'lidar_pcd_voxelized': lidar_pcd_voxel if (o3d is None or lidar_pcd_voxel is not None) else o3d.geometry.PointCloud(),
        'radar_keys': radar_keys_uniq if 'radar_keys_uniq' in locals() else np.zeros((0,), dtype=np.int64),
        'lidar_keys': lidar_keys,
        'radar_vals_agg': radar_vals_agg,
        'lidar_vals_agg': agg_vals,
        'mesh': mesh_o3d,
        'params': params,
    }

    
def run_single_view(
    adc_file: str,
    config_path: Optional[str],
    lidar_pcl_path: Optional[str],
    lidar_intensity_pcl_path: Optional[str],
    mesh_path: Optional[str],
    output_directory: Path,
    device: torch.device,
    near_field_m: float = 1.5,
    radar_threshold_percentile: Optional[float] = None,
    lidar_voxelize: bool = True,
    lidar_voxel_percentile: float = 0.0,
    lidar_color_by_intensity: bool = True,
    mesh_filter_enabled: bool = False,
    mesh_filter_eps_m: float = 0.05,
    mesh_filter_samples: int = 200000,
    wireframe_enabled: bool = False,
    wf_range_step: int = 16,
    wf_grid_step_az: int = 8,
    wf_grid_step_el: int = 8,
    radar_shift_to_config: bool = False,
    radar_orient_to_config: bool = False,
    show_radar: bool = True,
    show_intensity_voxelized: bool = False,
    show_intensity_points: bool = False,
    show_scene: bool = True,
    lidar_agg_method: str = 'median',
    save_ra_path: Optional[str] = None,
) -> Dict[str, Any]:
    # Load config-derived parameters
    params = load_runtime_config(config_path, output_directory)
    # Compute RAE cube and RA collapse
    rae = compute_rae_cube(Path(adc_file), params, device)

    # Load config geometry
    tx_m, rx_m, cfg_center, cfg_bore = load_config_positions_and_boresight(config_path)

    # Extract board frame azimuth direction and apply azimuth flip if needed
    # This ensures consistency between RA images and 3D visualizations
    if radar_orient_to_config and cfg_bore is not None and config_path is not None:
        try:
            board_azimuth, _, _ = extract_board_frame_from_config(config_path)

            # Flip RAE along azimuth axis when board azimuth points in +X direction
            # This compensates for the PCA sign ambiguity and ensures correct orientation
            # for both RA collapse and 3D point extraction
            if board_azimuth[0] >= 0:
                print(f"\n[RAE Orientation] Board azimuth: {board_azimuth}")
                print(f"[RAE Orientation] Flipping RAE along azimuth axis (azimuth[0]={board_azimuth[0]:.3f} >= 0)")
                rae = np.flip(rae, axis=0).copy()  # Flip along azimuth axis
            else:
                print(f"\n[RAE Orientation] Board azimuth: {board_azimuth}")
                print(f"[RAE Orientation] No RAE flip needed (azimuth[0]={board_azimuth[0]:.3f} < 0)")
        except Exception as e:
            print(f"[WARNING] Failed to extract board frame for RAE orientation: {e}")

    # Collapse RAE to RA map
    ra_map, az_deg, r_m = collapse_to_ra(rae, params)

    # Save Range-Azimuth cartesian image if requested
    if save_ra_path is not None:
        try:
            print("\n" + "="*80)
            print("GENERATING RANGE-AZIMUTH IMAGE")
            print("="*80)
            print(f"RA map shape: {ra_map.shape} (Azimuth, Range)")
            print(f"Range resolution: {params['range_resolution']:.4f} m")

            # Flip RA map along azimuth axis before cartesian conversion
            # This ensures correct left-right orientation in the 2D image
            print("Flipping RA map along azimuth axis (left-right)...")
            ra_map_flipped = np.flip(ra_map, axis=0)

            # Convert polar RA to cartesian
            print("Converting RA from polar to cartesian...")
            ra_cartesian = ra_polar_to_cartesian(ra_map_flipped, params['range_resolution'])
            print(f"  RA cartesian shape: {ra_cartesian.shape}")

            # Save the image
            print(f"Saving RA image to: {save_ra_path}")
            save_ra_image(ra_cartesian, save_ra_path)
            print(f"  [PASS] RA image saved successfully")
            print("="*80)
        except Exception as e:
            print(f"[ERROR] Failed to generate/save RA image: {e}")
            import traceback
            traceback.print_exc()

    # Build radar point cloud
    radar_pts, radar_vals, _ = radar_points_from_rae(
        rae, params, near_field_m=near_field_m, percentile=radar_threshold_percentile
    )
    # Optional orientation to config boresight then translation to config center
    if (radar_orient_to_config or radar_shift_to_config) and radar_pts.shape[0] > 0:
        try:
            if radar_orient_to_config and cfg_bore is not None and config_path is not None:
                # Extract full board frame from antenna positions (PCA-based)
                # This recovers the exact horizontal/vertical axes from the dense array layout
                print("\n" + "="*80)
                print("ALIGNING RADAR TO BOARD FRAME (PCA-based)")
                print("="*80)

                try:
                    v2_azimuth, y_range, v1_elevation = extract_board_frame_from_config(config_path)

                    print(f"Board horizontal (azimuth): {v2_azimuth}")
                    print(f"Board boresight (range):    {y_range}")
                    print(f"Board vertical (elevation): {v1_elevation}")

                    # Build rotation matrix: columns are target frame axes
                    # Radar frame: X=azimuth, Y=range, Z=elevation
                    # Board frame: v2=azimuth, y_range=range, v1=elevation
                    R_align = np.column_stack([v2_azimuth, y_range, v1_elevation])

                    # Verify orthonormality (should be 1.0 for valid rotation matrix)
                    det = np.linalg.det(R_align)
                    print(f"Rotation matrix determinant: {det:.6f} (expect 1.0)")

                    if abs(det - 1.0) > 0.01:
                        print(f"  WARNING: Determinant far from 1.0, frame may not be orthonormal!")

                    # Apply rotation to radar points
                    radar_pts = (radar_pts @ R_align.T).astype(np.float32)
                    print("[PASS] Radar points rotated to board frame")
                    print("="*80)

                except Exception as e:
                    print(f"[ERROR] Failed to extract board frame: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to identity rotation (no alignment)")
                    print("="*80)
            if radar_shift_to_config and cfg_center is not None:
                radar_pts = radar_pts + cfg_center.reshape(1, 3).astype(np.float32)
        except Exception:
            pass
    # Always create radar_pcd for artifacts (show_radar only controls visualization)
    radar_pcd = None
    if o3d is not None and radar_pts.shape[0] > 0:
        radar_pcd = build_point_cloud(
            radar_pts[:, 0], radar_pts[:, 1], radar_pts[:, 2], radar_vals,
            percentile=0.0 if radar_threshold_percentile is None else 0.0,
            color_by_intensity=True
        )

    # Load mesh (optional)
    mesh_o3d = load_mesh(mesh_path)

    # LiDAR processing
    lidar_geom = None
    lidar_keys_eval = np.zeros((0,), dtype=np.int64)
    lidar_vals_agg_eval = np.zeros((0,), dtype=np.float32)
    lidar_pcd_raw_eval = None
    # Intensity-specific LiDAR artifacts (single-frame intensity PCL)
    lidar_intensity_pcd_raw_eval = None
    lidar_intensity_pcd_voxelized = None
    lidar_intensity_keys_eval = np.zeros((0,), dtype=np.int64)
    lidar_intensity_vals_agg_eval = np.zeros((0,), dtype=np.float32)
    if lidar_pcl_path is not None and o3d is not None:
        if lidar_voxelize:
            print(f"[LiDAR] Voxelize enabled (agg={lidar_agg_method}, viz_percentile={lidar_voxel_percentile}, near_field={near_field_m}m)")
            # For voxelization, align by subtracting center before spherical (the upstream function assumes centered)
            # We load raw then subtract center for computation, then return world centers
            # pcd = load_point_cloud(lidar_pcl_path)
            pcd = load_and_align_lidar(lidar_pcl_path, cfg_center)
            if pcd is not None and pcd.has_points():
                # Print AABB of aggregated LiDAR immediately after loading
                try:
                    pts_np = np.asarray(pcd.points)
                    bb_min = pts_np.min(axis=0)
                    bb_max = pts_np.max(axis=0)
                    print(f"[LiDAR] Aggregated PCL AABB min={bb_min.tolist()} max={bb_max.tolist()}")
                except Exception:
                    pass
                # Save aligned raw LiDAR for evaluation even when voxelizing
                lidar_pcd_raw_eval = o3d.geometry.PointCloud(pcd)
                try:
                    print(f"[LiDAR] Raw points loaded: {len(pcd.points)}")
                except Exception:
                    pass
                try:
                    if cfg_center is not None:
                        # work on a copy; translate to origin for spherical calc
                        pcd_cent = o3d.geometry.PointCloud(pcd)
                        pcd_cent.translate((-cfg_center).tolist())
                    else:
                        pcd_cent = pcd
                except Exception:
                    pcd_cent = pcd
                centers, agg_vals, keys = voxelize_lidar_to_radar_grid(
                    pcd_cent, params, near_field_m=near_field_m, percentile=float(lidar_voxel_percentile), cfg_bore=cfg_bore, agg_method=str(lidar_agg_method)
                )
                print(f"[LiDAR] Voxel centers: {centers.shape[0]}")
                # Orient voxelized LiDAR to config boresight to match radar/wireframe (match 09_v2)
                if centers.shape[0] > 0 and cfg_bore is not None:
                    try:
                        a = np.array([0.0, 1.0, 0.0], dtype=float)
                        b = cfg_bore.astype(float)
                        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
                        if abs(dot - 1.0) < 1e-8:
                            R_align = np.eye(3)
                        elif abs(dot + 1.0) < 1e-8:
                            axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                            axis = axis - np.dot(axis, a) * a
                            axis = axis / (np.linalg.norm(axis) + 1e-12)
                            ux, uy, uz = axis
                            c = -1.0; s = 0.0
                            R_align = np.array([
                                [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                                [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                                [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                            ], dtype=float)
                        else:
                            axis = np.cross(a, b)
                            axis = axis / (np.linalg.norm(axis) + 1e-12)
                            ang = np.arccos(dot)
                            ux, uy, uz = axis
                            c = np.cos(ang); s = np.sin(ang)
                            R_align = np.array([
                                [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                                [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                                [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                            ], dtype=float)
                        centers = (centers @ R_align.T).astype(np.float32)
                    except Exception:
                        pass
                if cfg_center is not None and centers.shape[0] > 0:
                    centers = centers + cfg_center.reshape(1, 3).astype(np.float32)
                if mesh_filter_enabled and centers.shape[0] > 0 and mesh_o3d is not None:
                    centers, agg_vals, keys = optional_mesh_proximity_filter(
                        centers, agg_vals, keys, mesh_o3d, mesh_filter_eps_m, mesh_filter_samples
                    )
                if centers.shape[0] > 0:
                    try:
                        print(f"[LiDAR] Rendering voxelized LiDAR (first center) {centers[0].tolist()} with {agg_vals.shape[0]} intensities")
                    except Exception:
                        pass
                    lidar_geom = build_point_cloud(
                        centers[:, 0], centers[:, 1], centers[:, 2], agg_vals,
                        # Apply visualization percentile like 09_v2
                        percentile=float(lidar_voxel_percentile), color_by_intensity=True
                    )
                    # Eval artifacts for LiDAR voxelized
                    lidar_keys_eval = keys
                    lidar_vals_agg_eval = agg_vals
                else:
                    print("[LiDAR] No centers after voxelization (nothing to render)")
        else:
            # raw LiDAR point cloud, translated to config center, colored by intensity (optional)
            pcd = load_and_align_lidar(lidar_pcl_path, cfg_center)
            if pcd is not None and pcd.has_points():
                # Print AABB of aggregated LiDAR immediately after loading (raw mode)
                try:
                    pts_np = np.asarray(pcd.points)
                    bb_min = pts_np.min(axis=0)
                    bb_max = pts_np.max(axis=0)
                    print(f"[LiDAR] Aggregated PCL AABB min={bb_min.tolist()} max={bb_max.tolist()}")
                except Exception:
                    pass
                if not lidar_color_by_intensity:
                    try:
                        pts = np.asarray(pcd.points)
                        pcd = build_point_cloud(pts[:, 0], pts[:, 1], pts[:, 2], None, percentile=0.0, color_by_intensity=False)
                    except Exception:
                        pass
                lidar_geom = pcd
                lidar_pcd_raw_eval = pcd

    # Optional: process intensity LiDAR from NPY (single frame, (N,4) xyzi) using same pipeline
    if lidar_intensity_pcl_path is not None and o3d is not None:
        p_i = str(lidar_intensity_pcl_path)
        pcd_i = None
        try:
            import os as _os
            if _os.path.splitext(p_i)[1].lower() != '.npy':
                raise ValueError("Intensity LiDAR expects .npy (N,4) xyzi")
            arr_i = np.load(p_i)
            if arr_i.ndim != 2 or arr_i.shape[1] != 4:
                raise ValueError("Intensity .npy must have shape (N,4): [x,y,z,intensity]")
            xyz_i = arr_i[:, :3].astype(np.float32)
            inten_i = arr_i[:, 3].astype(np.float32).reshape(-1)
            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(xyz_i.astype(np.float64))
            # Map intensities to colors (grayscale) so downstream get_pcl_intensities() can read them
            try:
                vals = inten_i.astype(np.float64)
                if vals.size > 0 and np.all(np.isfinite(vals)):
                    vmin = float(vals.min()); vmax = float(vals.max())
                    if vmax <= vmin:
                        cols = np.ones((vals.shape[0], 3), dtype=np.float64)
                    else:
                        vn = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float64)
                        cols = np.stack([vn, vn, vn], axis=1)
                    pcd_i.colors = o3d.utility.Vector3dVector(cols)
                    try:
                        print(f"[Intensity LiDAR] NPY raw intensity min/max/mean: {vmin:.6f} / {vmax:.6f} / {float(vals.mean()):.6f}")
                    except Exception:
                        pass
                else:
                    # No finite values; default to ones
                    n_pts_tmp = xyz_i.shape[0]
                    pcd_i.colors = o3d.utility.Vector3dVector(np.ones((n_pts_tmp, 3), dtype=np.float64))
            except Exception as _e_col:
                # Fallback: set colors to ones
                n_pts_tmp = xyz_i.shape[0]
                pcd_i.colors = o3d.utility.Vector3dVector(np.ones((n_pts_tmp, 3), dtype=np.float64))
        except Exception as e:
            print(f"[warn] Failed to load intensity NPY '{p_i}': {e}")
        if pcd_i is not None and pcd_i.has_points():
            # Immediately after load: print intensity stats as currently present
            try:
                vals0 = get_pcl_intensities(pcd_i).astype(np.float64)
                if vals0.size > 0 and np.all(np.isfinite(vals0)):
                    print(f"[Intensity LiDAR] Immediately-after-load intensity min/max/mean: {float(vals0.min()):.6f} / {float(vals0.max()):.6f} / {float(vals0.mean()):.6f}")
                else:
                    print("[Intensity LiDAR] Immediately-after-load intensity stats unavailable (no values)")
            except Exception:
                pass
            try:
                n_raw = int(len(np.asarray(pcd_i.points)))
                has_cols = bool(pcd_i.has_colors())
                has_scalar_inten = False
                print(f"[Intensity LiDAR] Loaded NPY points: {n_raw}, has_colors={has_cols}, has_scalar_intensity={has_scalar_inten}")
                # Print AABB of intensity LiDAR immediately after loading
                try:
                    pts_np_i = np.asarray(pcd_i.points)
                    bb_min_i = pts_np_i.min(axis=0)
                    bb_max_i = pts_np_i.max(axis=0)
                    print(f"[Intensity LiDAR] PCL AABB min={bb_min_i.tolist()} max={bb_max_i.tolist()}")
                except Exception:
                    pass
                if has_scalar_inten:
                    vals = np.asarray(getattr(pcd_i, 'intensities'), dtype=np.float64).reshape(-1)
                    print(f"[Intensity LiDAR] Raw scalar intensity min/max/mean: {float(vals.min()):.6f} / {float(vals.max()):.6f} / {float(vals.mean()):.6f}")
                elif has_cols:
                    cols_raw = np.asarray(pcd_i.colors, dtype=np.float64)
                    inten_raw = cols_raw.mean(axis=1).astype(np.float64)
                    print(f"[Intensity LiDAR] Raw color->intensity min/max/mean: {float(inten_raw.min()):.6f} / {float(inten_raw.max()):.6f} / {float(inten_raw.mean()):.6f}")
                else:
                    print("[Intensity LiDAR] No colors and no scalar intensity found; intensities will default to ones")
            except Exception:
                pass
            # Save RAW intensity LiDAR (no translation) for visualization/eval
            lidar_intensity_pcd_raw_eval = o3d.geometry.PointCloud(pcd_i)
            # Paint raw intensity LiDAR red for differentiation
            try:
                n_pts_i = int(len(np.asarray(lidar_intensity_pcd_raw_eval.points)))
                if n_pts_i > 0:
                    red_cols = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n_pts_i, 1))
                    lidar_intensity_pcd_raw_eval.colors = o3d.utility.Vector3dVector(red_cols)
            except Exception:
                pass
            # For voxelization: translate to radar origin so the polar grid is centered on the radar
            if cfg_center is not None:
                try:
                    pcd_i_cent = o3d.geometry.PointCloud(pcd_i)
                    # Preserve scalar intensities across the copy if present
                    try:
                        if hasattr(pcd_i, 'intensities') and len(getattr(pcd_i, 'intensities')) == len(pcd_i.points):
                            setattr(pcd_i_cent, 'intensities', np.asarray(getattr(pcd_i, 'intensities'), dtype=np.float32).reshape(-1))
                    except Exception:
                        pass
                    pcd_i_cent.translate((-cfg_center).tolist())
                except Exception:
                    pcd_i_cent = pcd_i
            else:
                pcd_i_cent = pcd_i
                # Ensure intensities attribute exists on the working copy if present on original
                try:
                    if hasattr(pcd_i, 'intensities') and len(getattr(pcd_i, 'intensities')) == len(pcd_i.points):
                        setattr(pcd_i_cent, 'intensities', np.asarray(getattr(pcd_i, 'intensities'), dtype=np.float32).reshape(-1))
                except Exception:
                    pass
            try:
                n_cent = int(len(np.asarray(pcd_i_cent.points)))
                print(f"[Intensity LiDAR] Centered points (for voxelization): {n_cent}")
            except Exception:
                pass
            # Debug: verify intensity source right before voxelization
            try:
                vals_dbg = get_pcl_intensities(pcd_i_cent).astype(np.float64)
                if vals_dbg.size > 0:
                    print(f"[Intensity LiDAR] Pre-voxel intensity stats: min/max/mean={float(vals_dbg.min()):.6f}/{float(vals_dbg.max()):.6f}/{float(vals_dbg.mean()):.6f} uniques={int(np.unique(vals_dbg).size)}")
            except Exception:
                pass
            centers_i, agg_vals_i, keys_i = voxelize_lidar_to_radar_grid(
                pcd_i_cent, params, near_field_m=near_field_m, percentile=float(lidar_voxel_percentile), cfg_bore=cfg_bore, agg_method=str(lidar_agg_method)
            )
            try:
                print(f"[Intensity LiDAR] Voxelization output: centers={centers_i.shape}, agg_vals={agg_vals_i.shape}, keys={keys_i.shape}")
                if agg_vals_i.size > 0:
                    print(f"[Intensity LiDAR] Aggregated intensity ({lidar_agg_method} per voxel) min/max/mean: {float(agg_vals_i.min()):.6f} / {float(agg_vals_i.max()):.6f} / {float(agg_vals_i.mean()):.6f}")
            except Exception:
                pass
            # Forward boresight alignment and translation back to radar world position
            if centers_i.shape[0] > 0:
                if cfg_bore is not None:
                    try:
                        a = np.array([0.0, 1.0, 0.0], dtype=float)
                        b = cfg_bore.astype(float)
                        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
                        if abs(dot - 1.0) < 1e-8:
                            R_align_i = np.eye(3)
                        elif abs(dot + 1.0) < 1e-8:
                            axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                            axis = axis - np.dot(axis, a) * a
                            axis = axis / (np.linalg.norm(axis) + 1e-12)
                            ux, uy, uz = axis
                            c = -1.0; s = 0.0
                            R_align_i = np.array([
                                [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                                [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                                [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                            ], dtype=float)
                        else:
                            axis = np.cross(a, b)
                            axis = axis / (np.linalg.norm(axis) + 1e-12)
                            ang = np.arccos(dot)
                            ux, uy, uz = axis
                            c = np.cos(ang); s = np.sin(ang)
                            R_align_i = np.array([
                                [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                                [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                                [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                            ], dtype=float)
                        centers_i = (centers_i @ R_align_i.T).astype(np.float32)
                    except Exception:
                        pass
                if cfg_center is not None:
                    centers_i = centers_i + cfg_center.reshape(1, 3).astype(np.float32)
            # Optional mesh proximity filter (reuse same flags)
            if mesh_filter_enabled and centers_i.shape[0] > 0 and mesh_o3d is not None:
                centers_i, agg_vals_i, keys_i = optional_mesh_proximity_filter(
                    centers_i, agg_vals_i, keys_i, mesh_o3d, mesh_filter_eps_m, mesh_filter_samples
                )
                try:
                    print(f"[Intensity LiDAR] After mesh filter: centers={centers_i.shape}, agg_vals={agg_vals_i.shape}, keys={keys_i.shape}")
                except Exception:
                    pass
            if centers_i.shape[0] > 0:
                lidar_intensity_pcd_voxelized = build_point_cloud(
                    centers_i[:, 0], centers_i[:, 1], centers_i[:, 2], agg_vals_i,
                    percentile=0.0, color_by_intensity=True
                )
                # Paint voxelized intensity LiDAR red for differentiation
                try:
                    n_pts_iv = int(len(np.asarray(lidar_intensity_pcd_voxelized.points)))
                    if n_pts_iv > 0:
                        red_cols_iv = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n_pts_iv, 1))
                        lidar_intensity_pcd_voxelized.colors = o3d.utility.Vector3dVector(red_cols_iv)
                except Exception:
                    pass
                lidar_intensity_keys_eval = keys_i
                lidar_intensity_vals_agg_eval = agg_vals_i
                try:
                    print(f"[Intensity LiDAR] Built voxelized PCD: points={int(len(np.asarray(lidar_intensity_pcd_voxelized.points)))}")
                except Exception:
                    pass
            # Save paired LiDAR arrays for separate visualization
            try:
                lidar_intensity_xyz_for_viz = centers_i if centers_i is not None else np.zeros((0,3), dtype=np.float32)
                lidar_intensity_vals_for_viz = agg_vals_i if agg_vals_i is not None else np.zeros((0,), dtype=np.float32)
            except Exception:
                lidar_intensity_xyz_for_viz = np.zeros((0,3), dtype=np.float32)
                lidar_intensity_vals_for_viz = np.zeros((0,), dtype=np.float32)

    # Wireframe (optional)
    wf_ls = None
    if wireframe_enabled:
        wf_ls = build_fov_wireframe(params, wf_range_step, wf_grid_step_az, wf_grid_step_el, cfg_center, cfg_bore,
                                     orient_to_config=radar_orient_to_config, config_path=config_path)

    # Assemble scene geometries
    geoms: List[Any] = []
    if show_radar and radar_pcd is not None:
        geoms.append(radar_pcd)
    if lidar_geom is not None:
        geoms.append(lidar_geom)
    # Optional visualization of intensity LiDAR artifacts
    if show_intensity_points and lidar_intensity_pcd_raw_eval is not None:
        geoms.append(lidar_intensity_pcd_raw_eval)
    if show_intensity_voxelized and lidar_intensity_pcd_voxelized is not None:
        geoms.append(lidar_intensity_pcd_voxelized)
    try:
        print(f"[viz] Assembled geoms: radar={'yes' if radar_pcd is not None else 'no'}, lidar_vox={'yes' if lidar_geom is not None else 'no'}, intensity_raw={'yes' if (show_intensity_points and lidar_intensity_pcd_raw_eval is not None) else 'no'}, intensity_vox={'yes' if (show_intensity_voxelized and lidar_intensity_pcd_voxelized is not None) else 'no'}")
    except Exception:
        pass
    if mesh_o3d is not None:
        try:
            mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
        except Exception:
            pass
        geoms.append(mesh_o3d)
    if wf_ls is not None:
        geoms.append(wf_ls)
    # Optional config overlay (TX/RX spheres and axes)
    if o3d is not None and tx_m is not None and rx_m is not None:
        try:
            for pos in tx_m:
                geoms.append(create_colored_sphere(pos, radius=0.02, color_rgba_255=[255, 165, 0, 255]))
            for pos in rx_m:
                geoms.append(create_colored_sphere(pos, radius=0.02, color_rgba_255=[0, 255, 0, 255]))
            if cfg_center is not None:
                axis_len = 0.5
                geoms.append(create_axis_arrow(cfg_center, cfg_center + np.array([axis_len, 0, 0]), [255, 0, 0]))
                geoms.append(create_axis_arrow(cfg_center, cfg_center + np.array([0, axis_len, 0]), [0, 255, 0]))
                geoms.append(create_axis_arrow(cfg_center, cfg_center + np.array([0, 0, axis_len]), [0, 0, 255]))
                # boresight in yellow if present
                if cfg_bore is not None:
                    geoms.append(create_axis_arrow(cfg_center, cfg_center + axis_len * cfg_bore, [255, 255, 0]))
        except Exception:
            pass

    # Show scene (optional) - skip in headless environment
    if o3d is not None and bool(show_scene):
        try:
            build_scene(geoms, add_axes_at=cfg_center if cfg_center is not None else None)
        except Exception as e:
            print(f"[warn] Skipping visualization in headless environment: {e}")

    # Assemble evaluation artifacts (radar keys/vals aggregated; LiDAR keys/vals if voxelized)
    # Compute radar voxel keys using the SAME binning as LiDAR (edges from centers) for alignment
    num_az_full = int(params['num_az_bins'])
    num_el_full = int(params['num_el_bins'])
    t_az = np.arange(-num_az_full // 2 + 1, num_az_full // 2) * (2.0 / num_az_full)
    t_el = np.arange(-num_el_full // 2 + 1, num_el_full // 2) * (2.0 / num_el_full)
    az_angles = np.arcsin(np.clip(t_az, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    el_angles = np.arcsin(np.clip(t_el, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
    mag_eval = rae  # (Az, El, R)
    Rg, Ag, Eg = np.meshgrid(np.arange(mag_eval.shape[2]), np.arange(mag_eval.shape[0]), np.arange(mag_eval.shape[1]), indexing='ij')
    r_vals = (Rg.ravel().astype(np.float32)) * float(params['range_resolution'])
    az_vals = az_angles[Ag.ravel()]
    el_vals = el_angles[Eg.ravel()]
    x =  r_vals * np.cos(el_vals) * np.sin(az_vals)  # Fixed: removed negative sign
    y =  r_vals * np.cos(el_vals) * np.cos(az_vals)
    z =  r_vals * np.sin(el_vals)                     # Fixed: removed negative sign
    intensity = mag_eval.transpose(2, 0, 1).ravel().astype(np.float32)
    rng = np.sqrt(x*x + y*y + z*z)
    keep_nf = rng >= float(near_field_m)
    x = x[keep_nf]; y = y[keep_nf]; z = z[keep_nf]; intensity = intensity[keep_nf]
    # Build BOTH thresholded and un-thresholded radar intensity sets (for fair evaluation)
    if radar_threshold_percentile is not None and intensity.size > 0:
        thr = float(np.percentile(intensity, float(radar_threshold_percentile)))
        keep_thr = intensity >= thr if np.any(intensity >= thr) else intensity >= intensity.max()
    else:
        keep_thr = np.ones_like(intensity, dtype=bool)
    intensity_thr = intensity[keep_thr]
    x_thr = x[keep_thr]; y_thr = y[keep_thr]; z_thr = z[keep_thr]
    # Bin with same edges as LiDAR
    az_cent, el_cent = make_angle_grids_np(num_az_full, num_el_full)
    az_edges = _centers_to_edges(az_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    el_edges = _centers_to_edges(el_cent, low_clip=-np.pi/2, high_clip=np.pi/2).astype(np.float32)
    r_edges = (np.arange(int(params['num_adc']) + 1, dtype=np.float32) * float(params['range_resolution']))
    # Keys for thresholded subset
    r = np.sqrt(x_thr*x_thr + y_thr*y_thr + z_thr*z_thr)
    az = np.arctan2(-x_thr, y_thr)
    el = np.arcsin(np.clip(-z_thr / np.clip(r, 1e-12, None), -1.0, 1.0))
    az_idx = np.searchsorted(az_edges, az, side='right') - 1
    el_idx = np.searchsorted(el_edges, el, side='right') - 1
    r_idx  = np.searchsorted(r_edges,  r,  side='right') - 1
    A = az_cent.size; E = el_cent.size; Rb = int(params['num_adc'])
    valid = (az_idx >= 0) & (az_idx < A) & (el_idx >= 0) & (el_idx < E) & (r_idx >= 0) & (r_idx < Rb)
    az_idx = az_idx[valid]; el_idx = el_idx[valid]; r_idx = r_idx[valid]; intensity_thr = intensity_thr[valid]
    radar_keys = (az_idx.astype(np.int64) * E + el_idx.astype(np.int64)) * Rb + r_idx.astype(np.int64)
    order_r = np.argsort(radar_keys)
    k_sorted_r = radar_keys[order_r]
    v_sorted_r = intensity_thr[order_r]
    boundaries_r = np.where(np.diff(k_sorted_r) != 0)[0] + 1
    groups_r = np.split(v_sorted_r, boundaries_r)
    radar_vals_agg = np.array([float(np.percentile(g, 95.0)) for g in groups_r], dtype=np.float32)
    radar_keys_uniq = k_sorted_r[np.r_[0, boundaries_r]].astype(np.int64)

    # Also compute un-thresholded radar keys/vals ONLY at LiDAR voxel keys to reduce workload
    try:
        target_keys = None
        if 'lidar_intensity_keys_eval' in locals() and lidar_intensity_keys_eval is not None and lidar_intensity_keys_eval.size > 0:
            target_keys = lidar_intensity_keys_eval.astype(np.int64)
            print(f"[eval] Using intensity LiDAR keys for radar pairing: {int(target_keys.size)} keys")
        elif 'lidar_keys_eval' in locals() and lidar_keys_eval is not None and lidar_keys_eval.size > 0:
            target_keys = lidar_keys_eval.astype(np.int64)
            print(f"[eval] Using aggregated LiDAR voxel keys for radar pairing: {int(target_keys.size)} keys")
        else:
            target_keys = np.zeros((0,), dtype=np.int64)
            print("[eval] No LiDAR voxel keys available for radar pairing; defaulting to empty")
        if target_keys.size > 0:
            az_idx_f = (target_keys // (E * Rb)).astype(np.int64)
            rem_f = (target_keys % (E * Rb)).astype(np.int64)
            el_idx_f = (rem_f // Rb).astype(np.int64)
            r_idx_f = (rem_f % Rb).astype(np.int64)
            # Strict validity
            valid_f = (az_idx_f >= 0) & (az_idx_f < A) & (el_idx_f >= 0) & (el_idx_f < E) & (r_idx_f >= 0) & (r_idx_f < Rb)
            az_idx_f = az_idx_f[valid_f]; el_idx_f = el_idx_f[valid_f]; r_idx_f = r_idx_f[valid_f]
            radar_keys_uniq_full = target_keys[valid_f].astype(np.int64)
            # Direct lookup from RAE magnitude (Az, El, R)
            try:
                radar_vals_agg_full = rae[az_idx_f, el_idx_f, r_idx_f].astype(np.float32)
            except Exception:
                # Fallback via transposed mag if needed
                radar_vals_agg_full = rae.astype(np.float32)[az_idx_f, el_idx_f, r_idx_f]
            # Build radar XYZ centers corresponding to these keys (apply same orientation/translation)
            try:
                az_cent_f, el_cent_f = make_angle_grids_np(num_az_full, num_el_full)
                az_c = az_cent_f[az_idx_f].astype(np.float32)
                el_c = el_cent_f[el_idx_f].astype(np.float32)
                r_c = (r_idx_f.astype(np.float32) * float(params['range_resolution'])).astype(np.float32)
                x_r =  r_c * np.cos(el_c) * np.sin(az_c)  # Fixed: removed negative sign
                y_r =  r_c * np.cos(el_c) * np.cos(az_c)
                z_r =  r_c * np.sin(el_c)                  # Fixed: removed negative sign
                centers_r = np.stack([x_r, y_r, z_r], axis=1).astype(np.float32)
                # Forward align to config boresight and translate to config center to match LiDAR world frame
                if cfg_bore is not None and centers_r.shape[0] > 0:
                    try:
                        a = np.array([0.0, 1.0, 0.0], dtype=float)
                        b = cfg_bore.astype(float)
                        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
                        if abs(dot - 1.0) < 1e-8:
                            R_align_r = np.eye(3)
                        elif abs(dot + 1.0) < 1e-8:
                            axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                            axis = axis - np.dot(axis, a) * a
                            axis = axis / (np.linalg.norm(axis) + 1e-12)
                            ux, uy, uz = axis
                            c = -1.0; s = 0.0
                            R_align_r = np.array([
                                [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                                [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                                [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                            ], dtype=float)
                        else:
                            axis = np.cross(a, b)
                            axis = axis / (np.linalg.norm(axis) + 1e-12)
                            ang = np.arccos(dot)
                            ux, uy, uz = axis
                            c = np.cos(ang); s = np.sin(ang)
                            R_align_r = np.array([
                                [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                                [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
                                [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)]
                            ], dtype=float)
                        centers_r = (centers_r @ R_align_r.T).astype(np.float32)
                    except Exception:
                        pass
                if cfg_center is not None and centers_r.shape[0] > 0:
                    centers_r = centers_r + cfg_center.reshape(1, 3).astype(np.float32)
                paired_radar_xyz_for_viz = centers_r
                paired_radar_vals_for_viz = radar_vals_agg_full
            except Exception:
                paired_radar_xyz_for_viz = np.zeros((0,3), dtype=np.float32)
                paired_radar_vals_for_viz = np.zeros((0,), dtype=np.float32)
        else:
            radar_keys_uniq_full = np.zeros((0,), dtype=np.int64)
            radar_vals_agg_full = np.zeros((0,), dtype=np.float32)
            paired_radar_xyz_for_viz = np.zeros((0,3), dtype=np.float32)
            paired_radar_vals_for_viz = np.zeros((0,), dtype=np.float32)
    except Exception as e:
        print(f"[warn] Failed to build paired radar intensities: {e}")
        radar_keys_uniq_full = np.zeros((0,), dtype=np.int64)
        radar_vals_agg_full = np.zeros((0,), dtype=np.float32)
        paired_radar_xyz_for_viz = np.zeros((0,3), dtype=np.float32)
        paired_radar_vals_for_viz = np.zeros((0,), dtype=np.float32)

    return {
        'radar_pcd': radar_pcd if (o3d is None or radar_pcd is not None) else o3d.geometry.PointCloud(),
        'lidar_pcd_raw': lidar_pcd_raw_eval if (o3d is None or lidar_pcd_raw_eval is not None) else o3d.geometry.PointCloud(),
        'lidar_pcd_voxelized': lidar_geom if (o3d is None or lidar_geom is not None) and lidar_voxelize else (o3d.geometry.PointCloud() if o3d is not None else None),
        'radar_keys': radar_keys_uniq,
        'lidar_keys': lidar_keys_eval,
        'radar_vals_agg': radar_vals_agg,
        'radar_keys_full': radar_keys_uniq_full,
        'radar_vals_agg_full': radar_vals_agg_full,
        'lidar_vals_agg': lidar_vals_agg_eval,
        # Intensity LiDAR artifacts
        'lidar_intensity_pcd_raw': lidar_intensity_pcd_raw_eval if (o3d is None or lidar_intensity_pcd_raw_eval is not None) else o3d.geometry.PointCloud(),
        'lidar_intensity_pcd_voxelized': lidar_intensity_pcd_voxelized if (o3d is None or lidar_intensity_pcd_voxelized is not None) else o3d.geometry.PointCloud(),
        'lidar_intensity_keys': lidar_intensity_keys_eval,
        'lidar_intensity_vals_agg': lidar_intensity_vals_agg_eval,
        'mesh': mesh_o3d,
        'params': params,
        # Paired visualization arrays (world coordinates)
        'paired_lidar_xyz': lidar_intensity_xyz_for_viz if 'lidar_intensity_xyz_for_viz' in locals() else np.zeros((0,3), dtype=np.float32),
        'paired_lidar_vals': lidar_intensity_vals_for_viz if 'lidar_intensity_vals_for_viz' in locals() else np.zeros((0,), dtype=np.float32),
        'paired_radar_xyz': paired_radar_xyz_for_viz,
        'paired_radar_vals': paired_radar_vals_for_viz,
    }


