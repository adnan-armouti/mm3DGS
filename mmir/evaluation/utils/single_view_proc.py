import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better figure management
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import time
from scipy import interpolate
import open3d as o3d
# Cached Hann windows by (length, device)
_HANN_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}

def get_hann(n: int, device: torch.device) -> torch.Tensor:
    key = (int(n), str(device))
    t = _HANN_CACHE.get(key)
    if t is None:
        t = torch.tensor(np.hanning(n), device=device, dtype=torch.float32)
        _HANN_CACHE[key] = t
    return t

def to_tensor(x, device: torch.device, dtype=torch.float32, ):
    """Convert numpy array to torch tensor on the correct device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def plot_range_azimuth_heatmap_img(
        data,
        range_res,
        range_bias=0,
        heat_choice=0,
        COLORMAP: str = 'plasma',
        save_img=False,
        overlay_spot=None
    ):
    """Render a matplotlib figure showing a range-azimuth heat-map.
    
    This is the EXACT function from plot_boresight_video_ref.py
    
    Parameters
    ----------
    data : np.ndarray
        Expected shape -> (azimuth_angles, range)
    range_res : float
        Range resolution in metres.
    range_bias : float, optional
        Vertical shift applied to the range axis of the plot (m).
    heat_choice : int, optional
        0 -> relative normalisation; 1 -> absolute (0..(256^2)/2) colour-scale.
    save_img : bool, optional
        If *True*, writes *range_azimuth_heatmap.png* to disk.
    overlay_spot : tuple(int,int) | None
        Optional (range_idx, az_idx) tuple to highlight with a red circle.
    """
    num_angle_bins = data.shape[0] + 1
    num_adc        = data.shape[1]
    data = data.T

    t = np.arange(-num_angle_bins//2 + 1, num_angle_bins//2) * (2 / num_angle_bins)
    t = np.arcsin(t)
    r = np.arange(num_adc) * range_res

    range_depth = num_adc * range_res
    range_width, grid_res = range_depth / 2, 400
    xi = np.linspace(-range_width,  range_width, grid_res)
    yi = np.linspace(0,            range_depth, grid_res)
    xi, yi = np.meshgrid(xi, yi)

    x = r[:, None] * np.sin(t)
    y = r[:, None] * np.cos(t) - range_bias

    zi = interpolate.griddata((x.ravel(), y.ravel()),
                              data.ravel(),
                              (xi, yi),
                              method='linear')
    zi = zi[:-1, :-1]

    fig = plt.figure(figsize=(6, 6))
    ax  = plt.subplot(1, 1, 1)
    cm  = ax.imshow(((0,) * grid_res,) * grid_res,
                    cmap=COLORMAP,  # Use our plasma colormap
                    extent=[-range_width, +range_width, 0, range_depth],
                    alpha=0.95)

    ax.set_title(f'Azimuth-Range FFT Heatmap [{num_angle_bins};{num_adc}]', fontsize=10)
    ax.set_xlabel('Lateral distance [m]')
    ax.set_ylabel('Longitudinal distance [m]')
    ax.plot([0, 0],              [0, range_depth],  color='white', lw=0.5, ls=':', zorder=1)
    ax.plot([0, -range_width],   [0, range_width],  color='white', lw=0.5, ls=':', zorder=1)
    ax.plot([0, +range_width],   [0, range_width],  color='white', lw=0.5, ls=':', zorder=1)

    if overlay_spot is not None:
        r_idx, a_idx = overlay_spot
        a_spot, r_spot = t[a_idx], r[r_idx]
        x_spot = r_spot * np.sin(a_spot)
        y_spot = r_spot * np.cos(a_spot)
        ax.add_patch(plt.Circle((x_spot, y_spot), 0.1, color='red', fill=False))

    ax.set_ylim([0, range_depth])
    ax.set_xlim([-range_width, range_width])

    cm.set_array(zi[::-1, ::-1])  # rotate 180deg for consistent orientation
    if ('rel', 'abs')[heat_choice] == 'rel':
        cm.autoscale()
    else:
        cm.set_clim(0, 256**2 // 2)

    if save_img:
        fig.savefig('./range_azimuth_heatmap.png', dpi=150)

    return fig, ax, cm


def generate_uniform_virtual_antennas(num_antennas, spacing=1.0):
    """
    Generate uniform virtual antenna positions based on a single antenna count.
    
    Args:
        num_antennas: Total number of transmitters and receivers (must be even)
                     num_tx = num_rx = num_antennas
        spacing: Spacing between virtual antenna positions (default: 1.0)
        
    Returns:
        tuple: (virtual_antennas, tx_locations, rx_locations)
            - virtual_antennas: set of (x, y) tuples
            - tx_locations: list of (x, y) tuples for transmitters
            - rx_locations: list of (x, y) tuples for receivers
            
    Raises:
        AssertionError: If num_antennas is not even
    """
    
    # Step 1: Assert that input is an even number
    assert num_antennas % 2 == 0, f"Number of antennas ({num_antennas}) must be an even number"
    
    # Set num_tx = num_rx = num_antennas
    total_tx = num_antennas
    total_rx = num_antennas
    
    # Step 2: Divide by 2 (receivers form two rows and transmitters form two columns)
    tx_per_column = total_tx // 2
    rx_per_row = total_rx // 2
    
    # Step 3: Establish transmitter and receiver locations
    tx_locations = []
    rx_locations = []
    
    # Transmitter positions (two columns: left and right)
    # Left column: x=0.5, y from 1 to tx_per_column
    # Right column: x=tx_per_column+0.5, y from 1 to tx_per_column
    for y in (np.arange(-(tx_per_column-1)/2, (tx_per_column)/2, 1)):
        # Left column
        tx_locations.append((-(tx_per_column/2), y))
        # Right column (positioned to create uniform spacing)
        tx_locations.append(((tx_per_column/2), y))
    
    # Receiver positions (two rows: bottom and top)
    # Bottom row: y=0, x from 0.5 to rx_per_row-0.5
    # Top row: y=rx_per_row, x from 0.5 to rx_per_row-0.5
    for x in (np.arange(-(rx_per_row-1)/2, (rx_per_row)/2, 1)):
        # Bottom row
        rx_locations.append((x, -(tx_per_column/2)))
        # Top row (positioned to create uniform spacing)
        rx_locations.append((x, (tx_per_column/2)))
    
    # Step 4: Map to uniformly spaced virtual antenna positions
    virtual_antennas = []
    
    for rx_loc in rx_locations:
        for tx_loc in tx_locations:
            vx_x = rx_loc[0] + tx_loc[0]
            vx_y = rx_loc[1] + tx_loc[1]
            virtual_antennas.append((vx_x, vx_y))
    
    return np.array(virtual_antennas), np.array(tx_locations), np.array(rx_locations)


def txrx_to_vx_chirps_dense_gpu_batched(adc_batch_t: torch.Tensor, num_ant: int = 50) -> torch.Tensor:
    """
    Vectorized TX-RX -> virtual array mapping for a batch on GPU.
    adc_batch_t: (B, N_Rx, N_Tx, N_ADC) complex64
    returns:     (B, Vy, Vx, N_ADC)    complex64
    """
    virtual_antennas, tx_locs, rx_locs = generate_uniform_virtual_antennas(num_ant)
    vx_x = virtual_antennas[:, 0]
    vx_y = virtual_antennas[:, 1]
    min_x, max_x = vx_x.min(), vx_x.max()
    min_y, max_y = vx_y.min(), vx_y.max()
    x_off = int(-min_x)
    y_off = int(-min_y)
    Vx = int(max_x - min_x + 1)
    Vy = int(max_y - min_y + 1)

    # Build pair indices
    pair_rx = []
    pair_tx = []
    pair_lin = []
    for rx_id, (rx_x, rx_y) in enumerate(rx_locs):
        for tx_id, (tx_x, tx_y) in enumerate(tx_locs):
            vx_xi = int(rx_x + tx_x + x_off)
            vx_yi = int(rx_y + tx_y + y_off)
            lin = vx_yi * Vx + vx_xi
            pair_rx.append(rx_id)
            pair_tx.append(tx_id)
            pair_lin.append(lin)

    dev = adc_batch_t.device
    pair_rx = torch.tensor(pair_rx, device=dev, dtype=torch.long)
    pair_tx = torch.tensor(pair_tx, device=dev, dtype=torch.long)
    pair_lin = torch.tensor(pair_lin, device=dev, dtype=torch.long)
    counts = torch.bincount(pair_lin, minlength=Vx * Vy).to(device=dev)
    counts = counts.clamp_min_(1)

    B, NR, NT, NADC = adc_batch_t.shape
    adc_pairs = adc_batch_t[:, pair_rx, pair_tx, :]  # (B, P, NADC)
    out_real = torch.zeros((B, Vx * Vy, NADC), device=dev, dtype=torch.float32)
    out_imag = torch.zeros_like(out_real)
    lin_idx = pair_lin.view(1, -1, 1).expand(B, -1, NADC)
    out_real = out_real.scatter_add(1, lin_idx, adc_pairs.real)
    out_imag = out_imag.scatter_add(1, lin_idx, adc_pairs.imag)
    cnts = counts.view(1, -1, 1)
    out_real = out_real / cnts
    out_imag = out_imag / cnts
    out = torch.complex(out_real, out_imag).view(B, Vy, Vx, NADC)
    return out


def _axis_angle_to_matrix(w):
    """
    Rodrigues' formula - converts a 3-vector axis-angle `w` (rad) to a 3x3
    rotation matrix.  Works with batched (...,3) inputs.
    """
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    
    k = w / theta
    kx, ky, kz = k[0], k[1], k[2]
    
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])
    
    eye = np.eye(3)
    return eye + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _norm(v, eps=1e-9):
    """L2-normalise `v` along its last dimension."""
    norm_val = np.linalg.norm(v)
    if norm_val < eps:
        return v
    return v / norm_val


def _calibrated_geometry_corrected(tx_pos, rx_pos, rx_bor, translation, rot_vec):
    """
    CORRECTED version of calibrated geometry that prevents drift.
    Based on the reference renderer_core.py implementation.
    
    This function:
    1. Keeps the board center fixed in world coordinates
    2. Applies rotations around the board center (not around world origin)
    3. Maintains relative antenna positions during rotation
    4. Applies translations in the rotated board frame
    
    Args:
        tx_pos: (N_tx, 3) transmitter positions
        rx_pos: (N_rx, 3) receiver positions  
        rx_bor: (N_rx, 3) receiver boresights
        translation: (3,) translation in board frame
        rot_vec: (3,) axis-angle rotation in global frame (rad)
    
    Returns:
        tx_new, rx_new, rx_bor_new: transformed positions and boresights
    """
    # 1) canonical board basis from the shared boresight
    bs0 = rx_bor[0]
    y0 = _norm(bs0)
    z_try = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(y0, z_try)) > 0.99:  # near collinear
        z_try = np.array([0.0, 1.0, 0.0])
    x0 = _norm(np.cross(y0, z_try))
    z0 = _norm(np.cross(x0, y0))

    # 2) rotate the entire board in global coords
    R = _axis_angle_to_matrix(rot_vec)
    y_hat = _norm(R @ y0)

    # 3) new orthonormal board frame after rotation
    z_try2 = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(y_hat, z_try2)) > 0.99:
        z_try2 = np.array([0.0, 1.0, 0.0])
    x_hat = _norm(np.cross(y_hat, z_try2))
    z_hat = _norm(np.cross(x_hat, y_hat))

    # 4) re-express every antenna in the rotated frame
    all_p = np.vstack([rx_pos, tx_pos])
    board_c = np.mean(all_p, axis=0, keepdims=True)  # board center
    
    # Calculate relative positions from board center
    delta = all_p - board_c
    
    # Project onto original board basis
    rel_x = np.dot(delta, x0) - np.mean(np.dot(delta, x0))
    rel_z = np.dot(delta, z0) - np.mean(np.dot(delta, z0))
    
    # Reconstruct in rotated frame
    recon = board_c + rel_x[:, np.newaxis] * x_hat + rel_z[:, np.newaxis] * z_hat

    # 5) apply translation in rotated frame
    recon = recon + translation[0] * x_hat + translation[1] * y_hat + translation[2] * z_hat

    # 6) split back into RX/TX
    n_rx = rx_pos.shape[0]
    rx_new = recon[:n_rx]
    tx_new = recon[n_rx:]
    rx_bor_new = np.tile(y_hat, (n_rx, 1))  # all receivers get same rotated boresight
    
    return tx_new, rx_new, rx_bor_new


def process_adc_to_ra_map_enhanced(
        adc_data_path: str,
        angle_deg: float, 
        elev_angle_deg: float = 0.0, 
        elev_pos_m: float = 0.0, 
        depth_pos_m: float = 0.0,
        DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        NUM_ANT: int = 100,
        NUM_ADC: int = 256,
        NUM_AZIMUTH_BINS: int = 128,
        NUM_ELEVATION_BINS: int = 128,
        RANGE_RESOLUTION: float = 0.117
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ENHANCED version that processes ADC data with additional controls.
    
    Args:
        adc_data_path: Path to ADC data file
        angle_deg: Azimuth angle for this view
        elev_angle_deg: Elevation angle (board tilt about X-axis) in degrees
        elev_pos_m: Vertical position offset in meters
        depth_pos_m: Forward/backward position offset in meters
    
    Returns:
        Tuple of (ra_map, range_axis, azimuth_axis)
            - ra_map: 2D array of shape (azimuth_bins, range_bins)
            - range_axis: Range values in meters
            - azimuth_axis: Azimuth values in degrees
    """
    print(f"Processing ADC data for {angle_deg:+.1f}deg azimuth, "
          f"elevation {elev_angle_deg:+.1f}deg, "
          f"elev_pos {elev_pos_m:+.3f}m, depth_pos {depth_pos_m:+.3f}m...")
    
    num_range_bins = NUM_ADC
    num_azimuth_bins = NUM_AZIMUTH_BINS
    num_elevation_bins = NUM_ELEVATION_BINS

    # Load ADC data
    adc_data = np.load(adc_data_path)  # shape (N_Tx, N_Rx, N_ADC, 2 for real and imag)
    print(f"ADC data shape: {adc_data.shape}")
    
    # Check if ADC data has sufficient content
    if adc_data.size == 0:
        raise ValueError(f"ADC data is empty for angle {angle_deg}deg")
    
    # Check if all values are zero (no radar returns)
    if np.all(adc_data == 0):
        print(f"Warning: All ADC values are zero for angle {angle_deg}deg - no radar returns detected")
        # Still process but expect minimal results
    
    # Make complex valued and move to GPU
    adc_data = adc_data[:,:,:,0] + 1j * adc_data[:,:,:,1]
    adc_data = np.expand_dims(adc_data, axis=0)            # (1, N_Tx, N_Rx, N_ADC)
    adc_data = adc_data.transpose(0, 2, 1, 3)              # (1, N_Rx, N_Tx, N_ADC)
    adc_t = to_tensor(adc_data, device=DEVICE, dtype=torch.complex64)     # GPU

    # TXxRX -> virtual array (GPU, batched function with B=1)
    vx = txrx_to_vx_chirps_dense_gpu_batched(adc_t, num_ant=NUM_ANT)  # (1, Vy, Vx, N_ADC)

    # Range window + FFT
    h_range = get_hann(num_range_bins, vx.device)
    vx = vx * h_range.view(1, 1, 1, -1)
    vx = torch.fft.fft(vx, n=num_range_bins, dim=-1)        # (1, Vy, Vx, R)

    # Bring to (B, El, Az, R); in our mapping Vy=elev, Vx=azimuth
    vol = vx  # (1, El, Az, R)

    # Azimuth FFT along Az dimension
    h_az = get_hann(vol.shape[2], vol.device)
    vol = vol * h_az.view(1, 1, -1, 1)
    vol = torch.fft.ifftshift(vol, dim=2)
    vol = torch.fft.fft(vol, n=num_azimuth_bins, dim=2)
    # Drop center azimuth (DC) bin before fftshift
    vol = vol[:, :, 1:, :]
    vol = torch.fft.fftshift(vol, dim=2)

    # Elevation FFT along El dimension
    h_el = get_hann(vol.shape[1], vol.device)
    vol = vol * h_el.view(1, -1, 1, 1)
    vol = torch.fft.ifftshift(vol, dim=1)
    vol = torch.fft.fft(vol, n=num_elevation_bins, dim=1)
    # Drop center elevation (DC) bin before fftshift
    vol = vol[:, 1:, :, :]
    vol = torch.fft.fftshift(vol, dim=1)

    adc_3d_rae_t = torch.abs(vol).to(torch.float32)  # (1, El, Az, R)
    adc_3d_rae = adc_3d_rae_t[0].permute(1, 0, 2).contiguous().cpu().numpy()  # (Az, El, R)
    print(f"3D RAE shape: {adc_3d_rae.shape}")

    # KEY STEP: Collapse the elevation dimension to create 2D RA map
    # Use sum to preserve intensity information across elevation bins
    ra_map = np.sum(adc_3d_rae, axis=1)  # Shape: (azimuth_bins, range_bins)

    # Create coordinate axes    
    # Azimuth axis: arcsin mapping for proper coordinate system
    # SWAP LEFT AND RIGHT LIMITS: Use reverse order to swap limits
    t = np.arange(-num_azimuth_bins//2 + 1, num_azimuth_bins//2) * (2 / num_azimuth_bins)
    azimuth_axis = np.degrees(np.arcsin(t))  # Convert to degrees
    azimuth_axis = azimuth_axis[::-1]  # SWAP: Reverse to get correct left/right limits
    
    # Range axis: linear spacing including DC at 0 m
    range_axis = np.arange(num_range_bins) * RANGE_RESOLUTION  # meters
    
    print(f"RA map shape: {ra_map.shape}")
    print(f"Range axis: {range_axis[0]:.3f} to {range_axis[-1]:.3f} meters")
    print(f"Azimuth axis: {azimuth_axis[0]:.1f} to {azimuth_axis[-1]:.1f} degrees (SWAPPED)")
    
    return ra_map, range_axis, azimuth_axis


def process_adc_to_ra_map(
        adc_data_path: str,
        angle_deg: float,
        DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        NUM_ANT: int = 100,
        NUM_ADC: int = 256,
        NUM_AZIMUTH_BINS: int = 128,
        NUM_ELEVATION_BINS: int = 128,
        RANGE_RESOLUTION: float = 0.117
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy function for backward compatibility.
    Calls the enhanced version with default parameters.
    """
    return process_adc_to_ra_map_enhanced(adc_data_path, angle_deg, 0.0, 0.0, 0.0, DEVICE, NUM_ANT, NUM_ADC, NUM_AZIMUTH_BINS, NUM_ELEVATION_BINS, RANGE_RESOLUTION)


def visualize_ra_map(
        ra_map: np.ndarray,
        range_axis: np.ndarray,
        azimuth_axis: np.ndarray, 
        angle_deg: float, output_folder: str, save_plot: bool = True,
        RA_MAPS_SUBDIR: str = 'ra_maps',
        DPI: int = 100,
        COLORMAP: str = 'plasma',
        DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        NUM_ANT: int = 100,
        NUM_ADC: int = 256,
        NUM_AZIMUTH_BINS: int = 128,
        NUM_ELEVATION_BINS: int = 128,
        RANGE_RESOLUTION: float = 0.117
    ) -> None:
    """
    Visualize a single Range-Azimuth map using the reference plotting function
    from plot_boresight_video_ref.py for polar-to-cartesian conversion.
    
    Args:
        ra_map: 2D array of shape (azimuth_bins, range_bins) - from reference function
        range_axis: Range values in meters
        azimuth_axis: Azimuth values in degrees
        angle_deg: Azimuth angle for this view
        output_folder: Folder to save plots
        save_plot: Whether to save the plot to file
    """
    print(f"Visualizing RA map for {angle_deg:+.1f}deg azimuth using reference plotting...")
    
    # Clear any existing figures to prevent conflicts
    plt.close('all')
    
    # Use the reference plotting function for polar-to-cartesian conversion
    # Note: ra_map should be (azimuth_bins, range_bins) for the reference function
    fig, ax, cm = plot_range_azimuth_heatmap_img(
        data=ra_map,  # Shape: (azimuth_bins, range_bins)
        range_res=RANGE_RESOLUTION,
        range_bias=0,
        heat_choice=0,  # Relative normalization
        save_img=False,
        overlay_spot=None,
        COLORMAP=COLORMAP
    )
    
    # Update title to show this is from our virtual spinning radar
    ax.set_title(f'Virtual Spinning Radar - {angle_deg:+.1f}deg Azimuth View\n'
                f'Range-Azimuth Map (Polar-to-Cartesian)', fontsize=12, fontweight='bold')
    
    if save_plot:
        # Create output directory if it doesn't exist
        output_dir = Path(output_folder) / RA_MAPS_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        plot_filename = f"ra_map_{angle_deg:+.1f}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved plot: {plot_path}")
    
    # Show plot (blocking)
    plt.show(block=True)
    
    # Close figure to free memory
    plt.close(fig)
    time.sleep(0.1)  # Small delay to ensure figure is closed


def save_ra_map_data(
        ra_map: np.ndarray,
        range_axis: np.ndarray,
        azimuth_axis: np.ndarray,
        angle_deg: float,
        output_folder: str,
        NUM_ANT: int = 100,
        NUM_ADC: int = 256,
        NUM_AZIMUTH_BINS: int = 128,
        NUM_ELEVATION_BINS: int = 128,
        RANGE_RESOLUTION: float = 0.117) -> str:
    """
    Save RA map data as numpy arrays for later analysis.
    
    Args:
        ra_map: 2D array of shape (range_bins, azimuth_bins)
        range_axis: Range values in meters
        azimuth_axis: Azimuth values in degrees
        angle_deg: Azimuth angle for this view
        output_folder: Folder to save data
    
    Returns:
        Path to saved data file
    """
    # Create filename
    data_filename = f"ra_map_data_angle_{angle_deg:+.1f}.npz"
    data_path = os.path.join(output_folder, data_filename)
    
    # Save as compressed numpy array
    np.savez_compressed(data_path,
                        ra_map=ra_map,
                        range_axis=range_axis,
                        azimuth_axis=azimuth_axis,
                        angle_deg=angle_deg,
                        metadata={
                            'range_resolution': RANGE_RESOLUTION,
                            'num_ant': NUM_ANT,
                            'num_adc': NUM_ADC,
                            'num_azimuth_bins': NUM_AZIMUTH_BINS,
                            'num_elevation_bins': NUM_ELEVATION_BINS
                        })
    
    print(f"Saved RA map data: {data_path}")
    return data_path


# Verbatim helper from 05_visualize_voxels.py
def build_point_cloud(x: np.ndarray,
                      y: np.ndarray,
                      z: np.ndarray,
                      values: np.ndarray | None,
                      percentile: float,
                      color_by_intensity: bool) -> o3d.geometry.PointCloud:
    """Create an Open3D PointCloud from arrays, applying percentile filtering and coloring."""
    # Filter by percentile
    if values is not None and values.size == x.size and values.size > 0:
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = float(np.percentile(values, p))
        keep = values >= thr
        if not np.any(keep):
            keep = values >= values.max()
        x, y, z = x[keep], y[keep], z[keep]
        values = values[keep]

    pts = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color_by_intensity and values is not None and values.size == x.size:
        from matplotlib import cm
        vmin = float(values.min()) if values.size > 0 else 0.0
        vmax = float(values.max()) if values.size > 0 else 1.0
        if vmax <= vmin:
            vals_n = np.ones_like(values, dtype=np.float32)
        else:
            vals_n = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
        cols = cm.get_cmap('plasma')(vals_n)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(cols)

    return pcd


# LiDAR point cloud helper (from 09_viz_lidar_mesh_pcl.py)
def load_point_cloud(pcl_path: str) -> o3d.geometry.PointCloud | None:
    if o3d is None:
        print("Error: open3d is required to visualize geometries. pip install open3d")
        return None
    try:
        import os as _os
        ext = _os.path.splitext(pcl_path)[1].lower()

        # Handle .npy files separately
        if ext == '.npy':
            data = np.load(pcl_path)
            if data.shape[1] < 3:
                print(f"[warn] .npy file has insufficient columns: {data.shape}")
                return None
            # Extract XYZ (first 3 columns)
            points = data[:, :3].astype(np.float64)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            # If there's a 4th column, treat it as intensity
            if data.shape[1] >= 4:
                intensity = data[:, 3].astype(np.float32)
                try:
                    setattr(pcd, 'intensities', intensity)
                    print(f"[loader] Loaded .npy with intensity: n={len(points)}, intensity range=[{float(intensity.min()):.6f}, {float(intensity.max()):.6f}]")
                except Exception:
                    print(f"[loader] Loaded .npy: n={len(points)} (intensity cannot be attached as attribute, skipping)")
            else:
                print(f"[loader] Loaded .npy: n={len(points)} (no intensity)")
            # Assign default color if no colors
            if not pcd.has_colors():
                pcd.paint_uniform_color([0.1, 0.8, 1.0])
            return pcd

        # For other formats, use Open3D's loader
        pcd = o3d.io.read_point_cloud(pcl_path)
        if pcd is None or (not pcd.has_points()):
            print(f"[warn] Point cloud is empty: {pcl_path}")
            return None
        # Attempt to read scalar intensity from Open3D Tensor API and attach
        try:
            import os as _os
            ext = _os.path.splitext(pcl_path)[1].lower()
            if hasattr(o3d, 't') and ext in ('.ply', '.pcd'):
                try:
                    tpcd = o3d.t.io.read_point_cloud(pcl_path)
                    # Common field name is 'intensity'; allow case-insensitive
                    inten = None
                    if 'intensity' in tpcd.point:
                        inten = tpcd.point['intensity'].numpy().reshape(-1)
                        try:
                            if inten.size > 0:
                                print(f"[loader] Tensor 'intensity' detected: min/max/mean={float(np.min(inten)):.6f}/{float(np.max(inten)):.6f}/{float(np.mean(inten)):.6f}")
                        except Exception:
                            pass
                    elif 'Intensity' in tpcd.point:
                        inten = tpcd.point['Intensity'].numpy().reshape(-1)
                        try:
                            if inten.size > 0:
                                print(f"[loader] Tensor 'Intensity' detected: min/max/mean={float(np.min(inten)):.6f}/{float(np.max(inten)):.6f}/{float(np.mean(inten)):.6f}")
                        except Exception:
                            pass
                    if inten is not None:
                        # Prefer converting the tensor PCD to legacy to guarantee alignment of attributes
                        try:
                            pcd_legacy = tpcd.to_legacy()
                            n_t = int(len(pcd_legacy.points))
                            if inten.shape[0] == n_t:
                                pcd = pcd_legacy
                                setattr(pcd, 'intensities', inten.astype(np.float32))
                                print(f"[loader] Attached scalar intensity from tensor API (n={n_t})")
                            else:
                                # Fallback: attach only if it coincidentally matches the initially loaded legacy count
                                if inten.shape[0] == len(pcd.points):
                                    setattr(pcd, 'intensities', inten.astype(np.float32))
                                    print(f"[loader] Attached scalar intensity matching legacy count (n={len(pcd.points)})")
                                else:
                                    print(f"[loader] WARNING: intensity length {inten.shape[0]} mismatches legacy({len(pcd.points)}) and tensor({n_t}); skipping attach")
                        except Exception as _e_to_legacy:
                            # As a last resort, try to attach to the already-loaded legacy if counts match
                            if inten.shape[0] == len(pcd.points):
                                setattr(pcd, 'intensities', inten.astype(np.float32))
                                print(f"[loader] Attached scalar intensity to legacy (to_legacy failed): n={len(pcd.points)}")
                except Exception:
                    pass
        except Exception:
            pass
        # If no colors, assign a pleasant cyan-like color
        if not pcd.has_colors():
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                pcd.paint_uniform_color([0.1, 0.8, 1.0])
        # Debug summary about loaded PCD
        try:
            n_pts = int(len(pcd.points))
            has_cols = bool(pcd.has_colors())
            has_scalar = bool(hasattr(pcd, 'intensities') and len(pcd.intensities) == n_pts)
            inten_stats = None
            if has_scalar:
                vals = np.asarray(pcd.intensities, dtype=np.float64).reshape(-1)
                if vals.size > 0 and np.all(np.isfinite(vals)):
                    inten_stats = (float(vals.min()), float(vals.max()), float(vals.mean()))
            gray_stats = None
            if has_cols:
                cols = np.asarray(pcd.colors, dtype=np.float64)
                if cols.ndim == 2 and cols.shape[0] == n_pts and cols.shape[1] >= 3:
                    gray = (0.2126 * cols[:, 0] + 0.7152 * cols[:, 1] + 0.0722 * cols[:, 2]).astype(np.float64)
                    if gray.size > 0 and np.all(np.isfinite(gray)):
                        gray_stats = (float(gray.min()), float(gray.max()), float(gray.mean()))
            print(f"[loader] Loaded PCD: n={n_pts}, has_colors={has_cols}, has_scalar_intensity={has_scalar}")
            if inten_stats is not None:
                print(f"[loader] Scalar intensity stats: min/max/mean={inten_stats[0]:.6f}/{inten_stats[1]:.6f}/{inten_stats[2]:.6f}")
            if gray_stats is not None:
                print(f"[loader] Color->gray stats: min/max/mean={gray_stats[0]:.6f}/{gray_stats[1]:.6f}/{gray_stats[2]:.6f}")
        except Exception:
            pass
        return pcd
    except Exception as e:
        print(f"[warn] Failed to load point cloud '{pcl_path}': {e}")
        return None


# Extract per-point intensity from an Open3D point cloud
def get_pcl_intensities(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    n = len(pcd.points)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    # If Open3D exposes intensities (not always present), use them
    if hasattr(pcd, 'intensities') and len(pcd.intensities) == n:
        try:
            vals = np.asarray(pcd.intensities, dtype=np.float32)
            vals = vals.reshape(-1)
            try:
                if vals.size > 0:
                    print(f"[loader] get_pcl_intensities: using scalar intensities; min/max/mean={float(vals.min()):.6f}/{float(vals.max()):.6f}/{float(vals.mean()):.6f}")
            except Exception:
                pass
            return vals
        except Exception:
            pass
    # Fallback: derive grayscale intensity from colors if present
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
        if cols.ndim == 2 and cols.shape[0] == n and cols.shape[1] >= 3:
            r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
            gray = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
            try:
                if gray.size > 0:
                    print(f"[loader] get_pcl_intensities: using color->gray; min/max/mean={float(gray.min()):.6f}/{float(gray.max()):.6f}/{float(gray.mean()):.6f}")
            except Exception:
                pass
            return gray
    # Default: ones
    try:
        print("[loader] get_pcl_intensities: no intensity or colors; returning ones")
    except Exception:
        pass
    return np.ones((n,), dtype=np.float32)


# Helpers analogous to 11_lidar_visualization_new.py, adapted for Open3D
def load_antenna_config(config_path: str):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    tx_positions = np.array([tx['pos_mm'] for tx in cfg['tx_array']], dtype=float) / 1000.0
    rx_positions = np.array([rx['pos_mm'] for rx in cfg['rx_array']], dtype=float) / 1000.0
    return tx_positions, rx_positions


def create_colored_sphere(center: np.ndarray, radius: float, color_rgba_255: list):
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    sph.compute_vertex_normals()
    # Normalize 0-255 to 0-1 if needed
    if len(color_rgba_255) >= 3 and max(color_rgba_255[:3]) > 1.0:
        col = [c / 255.0 for c in color_rgba_255[:3]]
    else:
        col = color_rgba_255[:3]
    sph.paint_uniform_color(col)
    sph.translate(np.asarray(center, dtype=float).tolist())
    return sph


def create_axis_arrow(start: np.ndarray, end: np.ndarray, color_rgba_255: list):
    pts = o3d.utility.Vector3dVector(np.vstack([np.asarray(start, dtype=float), np.asarray(end, dtype=float)]))
    lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls = o3d.geometry.LineSet(points=pts, lines=lines)
    if len(color_rgba_255) >= 3 and max(color_rgba_255[:3]) > 1.0:
        col = [color_rgba_255[0] / 255.0, color_rgba_255[1] / 255.0, color_rgba_255[2] / 255.0]
    else:
        col = color_rgba_255[:3]
    ls.colors = o3d.utility.Vector3dVector(np.array([col], dtype=float))
    return ls


def _compute_range_resolution_from_config(cfg_path: Path) -> float:
    """Compute range resolution DeltaR = c / (2 * S * (rampEndTime - adcStartTime))."""
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    c = 299_792_458.0
    slope_hz_per_s = float(cfg["freqSlope"])  # Hz/s
    if slope_hz_per_s <= 0:
        raise ValueError(f"Invalid freqSlope in config: {slope_hz_per_s}")
    if "rampEndTime" not in cfg or "adcStartTime" not in cfg:
        raise KeyError("Config must contain 'rampEndTime' and 'adcStartTime' to compute DeltaR = c/(2*B)")
    T_chirp = float(cfg["rampEndTime"]) - float(cfg["adcStartTime"])  # s
    if T_chirp <= 0:
        raise ValueError(f"Invalid chirp timing: rampEndTime - adcStartTime = {T_chirp}")
    bandwidth = slope_hz_per_s * T_chirp      # Hz
    dr = c / (2.0 * bandwidth)
    return float(dr)

# Consistent board rotation from boresight (matches other viewers)
def board_rotation_from_boresight(boresight: np.ndarray) -> np.ndarray:
    y_hat = _norm(boresight.astype(np.float64))
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(y_hat, up))) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_hat = _norm(np.cross(y_hat, up))
    z_hat = _norm(np.cross(x_hat, y_hat))
    if float(np.dot(z_hat, np.array([0.0, 0.0, 1.0], dtype=np.float64))) < 0.0:
        x_hat = -x_hat; z_hat = -z_hat
    if float(np.dot(y_hat, np.array([0.0, 1.0, 0.0], dtype=np.float64))) < 0.0:
        y_hat = -y_hat; x_hat = -x_hat
    return np.stack([x_hat, y_hat, z_hat], axis=1)

# RA-style angle grids (DC-removed centers) as used for wireframes/radar
def make_angle_grids_np(az_bins_full: int, el_bins_full: int):
    eps = 1e-6
    t_az = np.arange(-az_bins_full // 2 + 1, az_bins_full // 2, dtype=np.float64) * (2.0 / float(az_bins_full))
    t_el = np.arange(-el_bins_full // 2 + 1, el_bins_full // 2, dtype=np.float64) * (2.0 / float(el_bins_full))
    t_az = np.clip(t_az, -1.0 + eps, 1.0 - eps)
    t_el = np.clip(t_el, -1.0 + eps, 1.0 - eps)
    return np.arcsin(t_az), np.arcsin(t_el)

def _centers_to_edges(centers: np.ndarray, low_clip: float, high_clip: float) -> np.ndarray:
    centers = centers.astype(np.float64)
    if centers.size == 1:
        width = 1e-3
        return np.array([centers[0] - width, centers[0] + width], dtype=np.float64)
    diffs = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + 0.5 * diffs
    edges[0] = centers[0] - 0.5 * diffs[0]
    edges[-1] = centers[-1] + 0.5 * diffs[-1]
    edges[0] = max(edges[0], low_clip)
    edges[-1] = min(edges[-1], high_clip)
    return edges


