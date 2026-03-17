import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import open3d as o3d
from matplotlib import cm

# ----------------------------- GPU Utility Functions -----------------------------
def to_tensor(x, device: torch.device, dtype=torch.float32, ):
    """Convert numpy array to torch tensor on the correct device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def to_numpy(x):
    """Convert torch tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x

# Cached Hann windows by (length, device)
_HANN_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}

def get_hann(n: int, device: torch.device) -> torch.Tensor:
    key = (int(n), str(device))
    t = _HANN_CACHE.get(key)
    if t is None:
        t = torch.tensor(np.hanning(n), device=device, dtype=torch.float32)
        _HANN_CACHE[key] = t
    return t


def _compute_range_resolution_from_config(cfg_path: Path) -> float:
    """Compute range resolution DeltaR from a single config JSON.

    Preferred: DeltaR = c / (2 * B) with B = S * (rampEndTime - adcStartTime),
    where S=freqSlope [Hz/s]. Falls back to ADC window if needed.
    """
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

def generate_uniform_virtual_antennas(num_antennas: int, spacing: float = 1.0):
    """
    Generate uniform virtual antenna positions for a square dense array.
    Returns (virtual_antennas, tx_locations, rx_locations).
    This is a copy from the existing pipeline for 100x100 dense mapping support.
    """
    assert num_antennas % 2 == 0, "Number of antennas must be even"
    total_tx = num_antennas
    total_rx = num_antennas

    tx_per_column = total_tx // 2
    rx_per_row = total_rx // 2

    tx_locations = []
    rx_locations = []

    for y in (np.arange(-(tx_per_column - 1) / 2, (tx_per_column) / 2, 1)):
        tx_locations.append((-(tx_per_column / 2), y))
        tx_locations.append(((tx_per_column / 2), y))

    for x in (np.arange(-(rx_per_row - 1) / 2, (rx_per_row) / 2, 1)):
        rx_locations.append((x, -(tx_per_column / 2)))
        rx_locations.append((x, (tx_per_column / 2)))

    virtual_antennas = []
    for rx_loc in rx_locations:
        for tx_loc in tx_locations:
            vx_x = rx_loc[0] + tx_loc[0]
            vx_y = rx_loc[1] + tx_loc[1]
            virtual_antennas.append((vx_x, vx_y))

    return np.array(virtual_antennas), np.array(tx_locations), np.array(rx_locations)


# ---- VX mapper cache (global) -----------------------------------------------
_MAPPER_CACHE = {}

class VxMapper:
    """
    Precomputes index mapping from (rx, tx) pairs to (vy, vx) bins for a uniform dense array.
    Allows vectorized scatter-add/mean for a batch of ADC tensors.
    """
    def __init__(self, num_ant: int, device: torch.device):
        self.num_ant = int(num_ant)
        self.device = device

        virtual_antennas, tx_locs, rx_locs = generate_uniform_virtual_antennas(self.num_ant)

        vx_x = virtual_antennas[:, 0]
        vx_y = virtual_antennas[:, 1]
        min_x, max_x = vx_x.min(), vx_x.max()
        min_y, max_y = vx_y.min(), vx_y.max()
        self.x_off = int(-min_x)
        self.y_off = int(-min_y)
        self.Vx = int(max_x - min_x + 1)
        self.Vy = int(max_y - min_y + 1)

        rx_locations = rx_locs
        tx_locations = tx_locs

        pair_rx = []
        pair_tx = []
        pair_lin = []

        for rx_id, (rx_x, rx_y) in enumerate(rx_locations):
            for tx_id, (tx_x, tx_y) in enumerate(tx_locations):
                vx_xi = int(rx_x + tx_x + self.x_off)
                vx_yi = int(rx_y + tx_y + self.y_off)
                lin = vx_yi * self.Vx + vx_xi
                pair_rx.append(rx_id)
                pair_tx.append(tx_id)
                pair_lin.append(lin)

        device_str = device if isinstance(device, str) else str(device)
        dev = torch.device(device_str)
        pair_rx = torch.tensor(pair_rx, device=dev, dtype=torch.long)
        pair_tx = torch.tensor(pair_tx, device=dev, dtype=torch.long)
        pair_lin = torch.tensor(pair_lin, device=dev, dtype=torch.long)

        counts = torch.bincount(pair_lin, minlength=self.Vx * self.Vy).to(device=dev)
        counts = counts.clamp_min_(1)

        self.rx_idx = pair_rx
        self.tx_idx = pair_tx
        self.lin_idx = pair_lin
        self.counts_lin = counts
        self.VyVx = self.Vy * self.Vx

    @staticmethod
    def get(num_ant: int, device: torch.device):
        key = (int(num_ant), str(device))
        mapper = _MAPPER_CACHE.get(key)
        if mapper is None:
            mapper = VxMapper(num_ant, device)
            _MAPPER_CACHE[key] = mapper
        return mapper


def txrx_to_vx_chirps_dense_gpu_batched(adc_batch: torch.Tensor, num_ant: int = 50) -> torch.Tensor:
    """
    Vectorized TX-RX -> virtual array mapping for a batch.
    adc_batch: (B, N_Rx, N_Tx, N_ADC) complex64
    returns:   (B, Vy, Vx, N_ADC)    complex64
    """
    assert adc_batch.ndim == 4 and torch.is_complex(adc_batch)
    B, NR, NT, NADC = adc_batch.shape
    mapper = VxMapper.get(num_ant, adc_batch.device)

    adc_pairs = adc_batch[:, mapper.rx_idx, mapper.tx_idx, :]   # (B, P, NADC)

    out_real = torch.zeros((B, mapper.VyVx, NADC), device=adc_batch.device, dtype=torch.float32)
    out_imag = torch.zeros_like(out_real)

    lin_idx = mapper.lin_idx.view(1, -1, 1).expand(B, -1, NADC)

    out_real = out_real.scatter_add(1, lin_idx, adc_pairs.real)
    out_imag = out_imag.scatter_add(1, lin_idx, adc_pairs.imag)

    counts = mapper.counts_lin.view(1, -1, 1)
    out_real = out_real / counts
    out_imag = out_imag / counts

    out = torch.complex(out_real, out_imag)
    out = out.view(B, mapper.Vy, mapper.Vx, NADC)
    return out

def txrx_to_vx_chirps_dense_gpu(adc_data: torch.Tensor, num_ant: int = 50) -> torch.Tensor:
    """
    GPU-accelerated version of TX-RX to virtual antenna conversion.
    adc_data: (N_Chirp, N_Rx, N_Tx, N_ADC) complex
    Returns:  (N_Chirp, vy, vx, N_ADC) complex
    """
    virtual_antennas, tx_locations, rx_locations = generate_uniform_virtual_antennas(num_ant)

    vx_x_coords = [vx[0] for vx in virtual_antennas]
    vx_y_coords = [vx[1] for vx in virtual_antennas]
    min_x, max_x = min(vx_x_coords), max(vx_x_coords)
    min_y, max_y = min(vx_y_coords), max(vx_y_coords)
    x_offset = int(-min_x)
    y_offset = int(-min_y)

    max_x_idx = int(max_x - min_x + 1)
    max_y_idx = int(max_y - min_y + 1)

    out = torch.zeros((adc_data.shape[0], max_y_idx, max_x_idx, adc_data.shape[-1]), 
                      device=adc_data.device, dtype=adc_data.dtype)

    # Vectorized mapping using GPU tensors
    for chirp_id in range(adc_data.shape[0]):
        for rx_id in range(num_ant):
            for tx_id in range(num_ant):
                rx_loc = rx_locations[rx_id]
                tx_loc = tx_locations[tx_id]
                vx_x = rx_loc[0] + tx_loc[0]
                vx_y = rx_loc[1] + tx_loc[1]
                ix = int(vx_x + x_offset)
                iy = int(vx_y + y_offset)
                if 0 <= ix < max_x_idx and 0 <= iy < max_y_idx:
                    if torch.all(out[chirp_id, iy, ix] == 0):
                        out[chirp_id, iy, ix, :] = adc_data[chirp_id, rx_id, tx_id, :]
                    else:
                        out[chirp_id, iy, ix, :] = 0.5 * (
                            out[chirp_id, iy, ix, :] + adc_data[chirp_id, rx_id, tx_id, :]
                        )
    return out


def simple_3d_cfar_gpu(volume: torch.Tensor,
                       train: Tuple[int, int, int],
                       guard: Tuple[int, int, int],
                       threshold_factor: float) -> torch.Tensor:
    """
    GPU-accelerated 3D CA-CFAR using PyTorch convolution.
    volume: torch.Tensor [az, el, range] on GPU
    """
    tz, ty, tx = train
    gz, gy, gx = guard

    # Build kernel for training cells: ones, with central guard+center zeroed out
    size_a = 2 * tz + 2 * gz + 1
    size_e = 2 * ty + 2 * gy + 1
    size_r = 2 * tx + 2 * gx + 1
    
    # Ensure kernel size doesn't exceed volume dimensions
    if size_a > volume.shape[0] or size_e > volume.shape[1] or size_r > volume.shape[2]:
        print(f"  [CFAR] Kernel too large for volume {volume.shape}, using global threshold")
        q = torch.quantile(volume, 0.99)
        return volume >= q
    
    kernel = torch.ones((size_a, size_e, size_r), device=volume.device, dtype=volume.dtype)

    # Zero guard + center region
    kernel[tz:tz + 2 * gz + 1, ty:ty + 2 * gy + 1, tx:tx + 2 * gx + 1] = 0.0
    num_training = torch.sum(kernel)
    if num_training <= 1:
        print(f"  [CFAR] No training cells available, using global threshold")
        q = torch.quantile(volume, 0.99)
        return volume >= q

    print(f"  [CFAR] Using {num_training.item():.0f} training cells, kernel size {kernel.shape}")
    
    # GPU convolution using PyTorch
    # Reshape for 3D convolution: (1, 1, az, el, range)
    volume_5d = volume.unsqueeze(0).unsqueeze(0)
    kernel_5d = kernel.unsqueeze(0).unsqueeze(0)
    
    # Use F.conv3d for efficient GPU convolution
    local_sum = F.conv3d(volume_5d, kernel_5d, padding='same').squeeze(0).squeeze(0)
    
    local_mean = local_sum / float(num_training)
    detections = volume >= (threshold_factor * (local_mean + 1e-9))
    print(f"  [CFAR] Detected {torch.sum(detections).item():,} points out of {volume.numel():,}")
    return detections


@torch.no_grad()
def simple_3d_cfar_gpu_batched_per_sample(
        volume_batch: torch.Tensor,
        train: Tuple[int, int, int],
        guard: Tuple[int, int, int],
        threshold_factor: float,
    ) -> torch.Tensor:
    """
    Batched CFAR with per-sample grouped conv. Mirrors per-view semantics.
    volume_batch: (B, Az, El, R) float32
    Returns (B, Az, El, R) bool
    """
    assert volume_batch.ndim == 4, "Expected (B, Az, El, R)"
    vol = volume_batch.to(torch.float32)
    device = vol.device
    B, Az, El, R = vol.shape
    tz, ty, tx = train
    gz, gy, gx = guard

    kA = 2 * tz + 2 * gz + 1
    kE = 2 * ty + 2 * gy + 1
    kR = 2 * tx + 2 * gx + 1

    if kA > Az or kE > El or kR > R:
        flat = vol.view(B, -1)
        q = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        thr = q.view(B, 1, 1, 1)
        return vol >= thr

    base_kernel = torch.ones((kA, kE, kR), device=device, dtype=torch.float32)
    base_kernel[tz:tz + 2 * gz + 1, ty:ty + 2 * gy + 1, tx:tx + 2 * gx + 1] = 0.0
    num_training = float(base_kernel.sum().item())
    if num_training <= 1:
        flat = vol.view(B, -1)
        q = torch.quantile(flat, 0.99, dim=1, keepdim=True)
        thr = q.view(B, 1, 1, 1)
        return vol >= thr

    x = vol.unsqueeze(0)  # (1,B,Az,El,R)
    w = base_kernel.expand(B, 1, kA, kE, kR).contiguous()  # (B,1,kA,kE,kR)
    padA, padE, padR = (kA // 2), (kE // 2), (kR // 2)
    local_sum = F.conv3d(x, w, stride=1, padding=(padA, padE, padR), groups=B)
    local_sum = local_sum.squeeze(0)  # (B,Az,El,R)

    local_mean = local_sum / num_training
    detections = vol >= (threshold_factor * (local_mean + 1e-9))
    return detections


def fft_to_rae_magnitude_gpu(
        adc_data_path: str,
        NUM_ANT: int,
        NUM_ADC: int,
        NUM_AZIMUTH_BINS: int,
        NUM_ELEVATION_BINS: int,
        DEVICE: torch.device
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    GPU-accelerated FFT processing for a single view.
    Returns (mag_rae, shape_info) on GPU.
    """
    adc = np.load(adc_data_path)  # (N_Tx, N_Rx, N_ADC, 2)
    adc = adc[:, :, :, 0] + 1j * adc[:, :, :, 1]
    adc = np.expand_dims(adc, axis=0)              # (1, N_Tx, N_Rx, N_ADC)
    adc = adc.transpose(0, 2, 1, 3)                # (1, N_Rx, N_Tx, N_ADC)
    
    # Convert to GPU tensor
    adc_tensor = to_tensor(adc, device=DEVICE, dtype=torch.complex64)
    vx = txrx_to_vx_chirps_dense_gpu(adc_tensor, num_ant=NUM_ANT)

    adc_3d = vx[0, :, :, :]                        # (elev, az, range) in our mapping

    # Range FFT (GPU accelerated)
    hanning_range = to_tensor(np.hanning(NUM_ADC), device=DEVICE, dtype=torch.float32)
    adc_3d *= hanning_range[None, None, :]
    adc_3d = torch.fft.fft(adc_3d, n=NUM_ADC, dim=-1)

    # Azimuth FFT (GPU accelerated)
    hanning_az = to_tensor(np.hanning(adc_3d.shape[1]), device=DEVICE, dtype=torch.float32)
    adc_3d *= hanning_az[None, :, None]
    adc_3d = torch.fft.ifftshift(adc_3d, dim=1)
    adc_3d = torch.fft.fft(adc_3d, n=NUM_AZIMUTH_BINS, dim=1)
    adc_3d = adc_3d[:, 1:, :]  # drop DC az bin
    adc_3d = torch.fft.fftshift(adc_3d, dim=1)

    # Elevation FFT (GPU accelerated)
    hanning_el = to_tensor(np.hanning(adc_3d.shape[0]), device=DEVICE, dtype=torch.float32)
    adc_3d *= hanning_el[:, None, None]
    adc_3d = torch.fft.ifftshift(adc_3d, dim=0)
    adc_3d = torch.fft.fft(adc_3d, n=NUM_ELEVATION_BINS, dim=0)
    adc_3d = adc_3d[1:, :, :]  # drop DC el bin
    adc_3d = torch.fft.fftshift(adc_3d, dim=0)

    mag_rae = torch.abs(adc_3d)                     # (elev, az, range)
    # PyTorch transpose only takes 2 arguments, so we need to do it in steps
    mag_rae = mag_rae.permute(1, 0, 2)              # (az, el, range)
    return mag_rae, mag_rae.shape


def fft_to_rae_magnitude_gpu_batch(
        adc_data_paths: List[str],
        NUM_ANT: int,
        NUM_ADC: int,
        NUM_AZIMUTH_BINS: int,
        NUM_ELEVATION_BINS: int,
        DEVICE: torch.device
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Load a batch of views, map to virtual array, and compute |FFT| in (az, el, range) for all.
    Returns (mag_rae_batch, (Az, El, R)) where mag_rae_batch is (B, Az, El, R) float32 on GPU.
    """
    data_np = []
    for p in adc_data_paths:
        d = np.load(p)  # (N_Tx, N_Rx, N_ADC, 2)
        data_np.append(d)
    arr = np.stack(data_np, axis=0)                   # (B, N_Tx, N_Rx, N_ADC, 2)
    adc = arr[..., 0] + 1j * arr[..., 1]             # (B, N_Tx, N_Rx, N_ADC)
    adc = np.transpose(adc, (0, 2, 1, 3))            # (B, N_Rx, N_Tx, N_ADC)

    adc_t = to_tensor(adc, device=DEVICE, dtype=torch.complex64)

    # Map TXxRX -> Virtual array (vectorized, batched)
    vx = txrx_to_vx_chirps_dense_gpu_batched(adc_t, num_ant=NUM_ANT)   # (B, Vy, Vx, N_ADC)

    # Range window + FFT (cache-friendly)
    h_range = get_hann(NUM_ADC, vx.device)
    vx = vx * h_range.view(1, 1, 1, -1)
    vx = torch.fft.fft(vx, n=NUM_ADC, dim=-1)        # (B, Vy, Vx, R)

    # Bring to (B, El, Az, R); in our mapping Vy=elev, Vx=azimuth
    vol = vx  # (B, El, Az, R)

    # Azimuth FFT along Az dimension
    h_az = get_hann(vol.shape[2], vol.device)
    vol = vol * h_az.view(1, 1, -1, 1)
    vol = torch.fft.ifftshift(vol, dim=2)
    vol = torch.fft.fft(vol, n=NUM_AZIMUTH_BINS, dim=2)
    vol = vol[:, :, 1:, :]
    vol = torch.fft.fftshift(vol, dim=2)

    # Elevation FFT along El dimension
    h_el = get_hann(vol.shape[1], vol.device)
    vol = vol * h_el.view(1, -1, 1, 1)
    vol = torch.fft.ifftshift(vol, dim=1)
    vol = torch.fft.fft(vol, n=NUM_ELEVATION_BINS, dim=1)
    vol = vol[:, 1:, :, :]
    vol = torch.fft.fftshift(vol, dim=1)

    mag = torch.abs(vol).to(torch.float32)           # (B, El, Az, R)
    mag = mag.permute(0, 2, 1, 3).contiguous()       # (B, Az, El, R)
    shape = (mag.shape[1], mag.shape[2], mag.shape[3])
    return mag, shape


def rae_indices_to_world_points_gpu(
        mask: torch.Tensor,
        mag: torch.Tensor,
        sensor_yaw_deg: float,
        az_grid: torch.Tensor,
        el_grid: torch.Tensor,
        RANGE_RESOLUTION: float,
        MIN_RANGE_M: float,
        YAW_SIGN: float,
        FLIP_AZ: bool,
        return_torch: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated conversion of detected R-A-E indices to world XYZ and intensities.
    Returns numpy arrays for compatibility with the voxel accumulator.
    """
    az_idx, el_idx, r_idx = torch.nonzero(mask, as_tuple=True)
    if az_idx.numel() == 0:
        return (np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    # Map indices to physical angles and ranges using provided FFT-aware grids (GPU)
    az = az_grid[az_idx]
    el = el_grid[el_idx]
    r = (r_idx.to(torch.float32)) * float(RANGE_RESOLUTION)

    # Near-field removal: configurable minimum range (meters)
    nf_keep = r >= float(MIN_RANGE_M)
    if nf_keep.numel() == 0 or not torch.any(nf_keep):
        return (np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    # Filter indices by near-field mask
    az_idx = az_idx[nf_keep]
    el_idx = el_idx[nf_keep]
    r_idx = r_idx[nf_keep]
    az = az[nf_keep]
    el = el[nf_keep]
    r = r[nf_keep]
    intensity = mag[az_idx, el_idx, r_idx].to(torch.float32)

    # Map directly to world coordinates by folding sensor yaw into azimuth.
    if FLIP_AZ:
        az = -az
    yaw = torch.deg2rad(torch.tensor(YAW_SIGN * sensor_yaw_deg, device=mask.device, dtype=torch.float32))
    az_eff = az + yaw
    x_world = r * torch.cos(el) * torch.sin(az_eff)
    y_world = r * torch.cos(el) * torch.cos(az_eff)
    z_world = r * torch.sin(el)

    # Apply flips to match single-view visualization:
    # - Flip azimuth: left/right swap by negating X
    # - Flip elevation: up/down swap by negating Z
    x_world = -x_world
    z_world = -z_world

    if return_torch:
        return x_world, y_world, z_world, intensity
    # Convert back to numpy for compatibility
    return (to_numpy(x_world), to_numpy(y_world), to_numpy(z_world), to_numpy(intensity))


class VoxelAccumulatorGPU:
    """
    GPU-accelerated voxel accumulator using PyTorch tensors.
    """
    def __init__(self, voxel_size_m: float, device: torch.device):
        self.voxel_size = float(voxel_size_m)
        self.device = device
        self.data: Dict[Tuple[int, int, int], Dict[str, object]] = {}

    def _key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        return (
            int(math.floor(x / self.voxel_size)),
            int(math.floor(y / self.voxel_size)),
            int(math.floor(z / self.voxel_size)),
        )

    def add_points(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   intensity: np.ndarray, view_id: int):
        for xi, yi, zi, vi in zip(x, y, z, intensity):
            k = self._key(float(xi), float(yi), float(zi))
            slot = self.data.get(k)
            if slot is None:
                self.data[k] = {
                    "count": 1,
                    "sum_intensity": float(vi),
                    "max_intensity": float(vi),
                    "views": {view_id},
                }
            else:
                slot["count"] += 1
                slot["sum_intensity"] += float(vi)
                if float(vi) > slot["max_intensity"]:
                    slot["max_intensity"] = float(vi)
                slot["views"].add(view_id)

    def add_points_batch(self,
                         x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         intensity: np.ndarray, view_id: int):
        if x.size == 0:
            return
        vs = self.voxel_size
        i = np.floor(x / vs).astype(np.int32)
        j = np.floor(y / vs).astype(np.int32)
        k = np.floor(z / vs).astype(np.int32)

        vox = np.stack([i, j, k], axis=1)
        uniq, inv = np.unique(vox, axis=0, return_inverse=True)
        M = uniq.shape[0]

        counts = np.bincount(inv, minlength=M).astype(np.int32)
        sum_int = np.bincount(inv, weights=intensity, minlength=M).astype(np.float64)
        max_int = np.full(M, -np.inf, dtype=np.float32)
        np.maximum.at(max_int, inv, intensity.astype(np.float32))

        for idx in range(M):
            ii, jj, kk = map(int, uniq[idx])
            key = (ii, jj, kk)
            slot = self.data.get(key)
            if slot is None:
                self.data[key] = {
                    "count": int(counts[idx]),
                    "sum_intensity": float(sum_int[idx]),
                    "max_intensity": float(max_int[idx]),
                    "views": {view_id},
                }
            else:
                slot["count"] += int(counts[idx])
                slot["sum_intensity"] += float(sum_int[idx])
                if float(max_int[idx]) > slot["max_intensity"]:
                    slot["max_intensity"] = float(max_int[idx])
                slot["views"].add(view_id)

    def to_dense(self):
        if not self.data:
            return None
        keys = np.array(list(self.data.keys()), dtype=np.int32)
        i_min = keys.min(axis=0)
        i_max = keys.max(axis=0)
        shape = (i_max - i_min + 1)

        counts = np.zeros(shape, dtype=np.int32)
        max_int = np.zeros(shape, dtype=np.float32)
        view_div = np.zeros(shape, dtype=np.int32)

        for (i, j, k), slot in self.data.items():
            ii, jj, kk = i - i_min[0], j - i_min[1], k - i_min[2]
            counts[ii, jj, kk] = slot["count"]
            max_int[ii, jj, kk] = slot["max_intensity"]
            if "view_count" in slot:
                view_div[ii, jj, kk] = int(slot["view_count"])
            elif "view_mask" in slot:
                view_div[ii, jj, kk] = int(slot["view_mask"]).bit_count()
            else:
                view_div[ii, jj, kk] = len(slot.get("views", ()))

        # Ensure origin_idx is a numpy array for consistent behavior
        origin_idx = np.array(i_min, dtype=np.int32)
        return counts, max_int, view_div, origin_idx

    def kept_voxel_centers(self, keep_mask: np.ndarray, origin_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ii, jj, kk = np.nonzero(keep_mask)
        if ii.size == 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))
        centers_x = (ii + origin_idx[0] + 0.5) * self.voxel_size
        centers_y = (jj + origin_idx[1] + 0.5) * self.voxel_size
        centers_z = (kk + origin_idx[2] + 0.5) * self.voxel_size
        return centers_x, centers_y, centers_z

    @torch.no_grad()
    def add_points_batch_gpu(
        self,
        x_t: torch.Tensor, y_t: torch.Tensor, z_t: torch.Tensor,
        intensity_t: torch.Tensor, view_id: int
    ):
        if x_t.numel() == 0:
            return
        # Compute voxel indices on device
        vs = float(self.voxel_size)
        i = torch.floor(x_t / vs).to(torch.int32)
        j = torch.floor(y_t / vs).to(torch.int32)
        k = torch.floor(z_t / vs).to(torch.int32)

        ii_u, jj_u, kk_u, cnt_u, sum_u, max_u, viewbm_u = reduce_points_to_voxels_gpu_fast(
            i, j, k, intensity_t.to(torch.float32), None
        )
        ii_u = ii_u.cpu().numpy().astype(np.int32, copy=False)
        jj_u = jj_u.cpu().numpy().astype(np.int32, copy=False)
        kk_u = kk_u.cpu().numpy().astype(np.int32, copy=False)
        counts = cnt_u.cpu().numpy().astype(np.int32, copy=False)
        sum_int = sum_u.cpu().numpy().astype(np.float64, copy=False)
        max_int = max_u.cpu().numpy().astype(np.float32, copy=False)

        for ii, jj, kk, c, s, m in zip(ii_u, jj_u, kk_u, counts, sum_int, max_int):
            key = (int(ii), int(jj), int(kk))
            slot = self.data.get(key)
            if slot is None:
                self.data[key] = {
                    "count": int(c),
                    "sum_intensity": float(s),
                    "max_intensity": float(m),
                    "views": {view_id},
                }
            else:
                slot["count"] += int(c)
                slot["sum_intensity"] += float(s)
                if float(m) > slot["max_intensity"]:
                    slot["max_intensity"] = float(m)
                slot["views"].add(view_id)


def gaussian_smooth_gpu(volume: torch.Tensor, sigma) -> torch.Tensor:
    """
    GPU-accelerated 3D Gaussian smoothing using PyTorch.
    """
    # Support scalar or per-axis sigmas (az, el, range)
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            return volume
        sigma_az = sigma_el = sigma_r = float(sigma)
    else:
        sigma_az, sigma_el, sigma_r = [float(s) for s in sigma]
        if sigma_az <= 0 and sigma_el <= 0 and sigma_r <= 0:
            return volume
    
    # Convert to float32 for better GPU performance
    volume = volume.to(torch.float32)
    
    # Build 1D kernels per axis
    def make_kernel(sig):
        if sig <= 0:
            return None
        ks = int(2 * round(3 * sig) + 1)
        if ks % 2 == 0:
            ks += 1
        x = torch.arange(ks, device=volume.device, dtype=torch.float32) - ks // 2
        g = torch.exp(-(x ** 2) / (2 * sig ** 2))
        g = g / (g.sum() + 1e-12)
        return g

    k_az = make_kernel(sigma_az)
    k_el = make_kernel(sigma_el)
    k_r  = make_kernel(sigma_r)

    volume_5d = volume.unsqueeze(0).unsqueeze(0)
    volume_smooth = volume_5d
    # Apply along azimuth (D axis)
    if k_az is not None:
        kernel_az = k_az.view(1, 1, -1, 1, 1)
        volume_smooth = F.conv3d(volume_smooth, kernel_az, padding=(k_az.numel() // 2, 0, 0))
    # Apply along elevation (H axis)
    if k_el is not None:
        kernel_el = k_el.view(1, 1, 1, -1, 1)
        volume_smooth = F.conv3d(volume_smooth, kernel_el, padding=(0, k_el.numel() // 2, 0))
    # Apply along range (W axis)
    if k_r is not None:
        kernel_r = k_r.view(1, 1, 1, 1, -1)
        volume_smooth = F.conv3d(volume_smooth, kernel_r, padding=(0, 0, k_r.numel() // 2))
    
    return volume_smooth.squeeze(0).squeeze(0)


def azimuth_nms_gpu(
        volume_np: np.ndarray,
        window: int = 3,
        DEVICE: torch.device = torch.device('cuda')
    ) -> np.ndarray:
    """
    Simple azimuth-wise non-maximum suppression on a 3D volume (Az, El, R).
    Keeps voxels that are local maxima along azimuth within the given window.
    Returns a boolean mask with the same shape.
    """
    assert window % 2 == 1, "window must be odd"
    if volume_np.size == 0:
        return np.zeros_like(volume_np, dtype=bool)
    vol = to_tensor(volume_np, device=DEVICE, dtype=torch.float32)
    # reshape to (B, C, L) where B = El*R, L = Az
    vol_perm = vol.permute(1, 2, 0)  # (El, R, Az)
    B, Rdim, Az = vol_perm.shape[0], vol_perm.shape[1], vol_perm.shape[2]
    seq = vol_perm.reshape(B * Rdim, 1, Az)
    pad = window // 2
    pooled = F.max_pool1d(seq, kernel_size=window, stride=1, padding=pad)
    is_max = (seq >= pooled - 1e-12)
    is_max = is_max.reshape(B, Rdim, Az).permute(2, 0, 1)  # back to (Az, El, R)
    return to_numpy(is_max)


@torch.no_grad()
def reduce_points_to_voxels_gpu_fast(
    i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
    intensity: torch.Tensor,
    view_bits: torch.Tensor = None,
):
    """
    Packed-key sort + segment reduce on CUDA for voxel stats.
    Returns (ii_u, jj_u, kk_u, count, sum_i, max_i, viewbm) where viewbm may be None.
    """
    BIAS = 1 << 20
    ii = (i.to(torch.int64) + BIAS) & ((1 << 21) - 1)
    jj = (j.to(torch.int64) + BIAS) & ((1 << 21) - 1)
    kk = (k.to(torch.int64) + BIAS) & ((1 << 21) - 1)
    keys = (ii << 42) | (jj << 21) | kk

    order = torch.argsort(keys)
    keys_s = keys[order]
    inten = intensity[order]

    new_seg = torch.ones_like(keys_s, dtype=torch.bool)
    new_seg[1:] = keys_s[1:] != keys_s[:-1]
    seg_ids = torch.cumsum(new_seg.to(torch.int64), 0) - 1
    M = int(seg_ids[-1].item()) + 1 if seg_ids.numel() > 0 else 0

    if M == 0:
        empty_i = torch.empty((0,), device=keys.device, dtype=torch.int32)
        empty_f = torch.empty((0,), device=keys.device, dtype=torch.float32)
        empty_c = torch.empty((0,), device=keys.device, dtype=torch.int32)
        empty_u = torch.empty((0,), device=keys.device, dtype=torch.uint64)
        return empty_i, empty_i, empty_i, empty_c, empty_f, empty_f, (empty_u if view_bits is not None else None)

    one = torch.ones_like(inten, dtype=torch.int32)
    count = torch.zeros(M, device=inten.device, dtype=torch.int32).scatter_add_(0, seg_ids, one)
    sum_i = torch.zeros(M, device=inten.device, dtype=torch.float32).scatter_add_(0, seg_ids, inten)

    if hasattr(torch, "scatter_reduce"):
        max_i = torch.full((M,), -float("inf"), device=inten.device)
        max_i = torch.scatter_reduce(max_i, 0, seg_ids, inten, reduce="amax", include_self=True)
    else:
        max_i = torch.zeros(M, device=inten.device).scatter_reduce_(0, seg_ids, inten, reduce="amax", include_self=False)

    viewbm = None
    if view_bits is not None:
        # use signed 64-bit to ensure bitwise shifts are supported on CUDA
        vb = view_bits[order].to(torch.int64)
        viewbm = torch.zeros(M, device=vb.device, dtype=torch.int64).scatter_add_(0, seg_ids, vb)

    kk_u = (keys_s[new_seg] & ((1 << 21) - 1)).to(torch.int32) - BIAS
    jj_u = ((keys_s[new_seg] >> 21) & ((1 << 21) - 1)).to(torch.int32) - BIAS
    ii_u = ((keys_s[new_seg] >> 42) & ((1 << 21) - 1)).to(torch.int32) - BIAS
    return ii_u, jj_u, kk_u, count, sum_i, max_i.to(torch.float32), viewbm


@torch.no_grad()
def reduce_points_to_voxels_gpu_uniqueviews(
    i: torch.Tensor, j: torch.Tensor, k: torch.Tensor,
    intensity: torch.Tensor, view_idx: torch.Tensor,
):
    """
    Segment-reduce to voxels and also count unique view_idx per voxel.
    Returns: (ii_u, jj_u, kk_u, count, sum_i, max_i, view_count)
    All tensors are on the same device as inputs.
    """
    BIAS = 1 << 20
    ii = (i.to(torch.int64) + BIAS) & ((1 << 21) - 1)
    jj = (j.to(torch.int64) + BIAS) & ((1 << 21) - 1)
    kk = (k.to(torch.int64) + BIAS) & ((1 << 21) - 1)
    keys = (ii << 42) | (jj << 21) | kk

    v = view_idx.to(torch.int64)
    combined = keys * 1024 + v
    order = torch.argsort(combined)

    keys_s = keys[order]
    inten = intensity[order]
    v_s = v[order]

    new_voxel = torch.ones_like(keys_s, dtype=torch.bool)
    new_voxel[1:] = keys_s[1:] != keys_s[:-1]
    seg_ids = torch.cumsum(new_voxel.to(torch.int64), 0) - 1
    if seg_ids.numel() == 0:
        empty_i = torch.empty((0,), device=keys.device, dtype=torch.int32)
        empty_f = torch.empty((0,), device=keys.device, dtype=torch.float32)
        empty_c = torch.empty((0,), device=keys.device, dtype=torch.int32)
        return empty_i, empty_i, empty_i, empty_c, empty_f, empty_f, empty_c

    M = int(seg_ids[-1].item()) + 1
    one = torch.ones_like(inten, dtype=torch.int32)
    count = torch.zeros(M, device=inten.device, dtype=torch.int32).scatter_add_(0, seg_ids, one)
    sum_i = torch.zeros(M, device=inten.device, dtype=torch.float32).scatter_add_(0, seg_ids, inten)
    if hasattr(torch, "scatter_reduce"):
        max_i = torch.full((M,), -float("inf"), device=inten.device)
        max_i = torch.scatter_reduce(max_i, 0, seg_ids, inten, reduce="amax", include_self=True)
    else:
        max_i = torch.zeros(M, device=inten.device).scatter_reduce_(0, seg_ids, inten, reduce="amax", include_self=False)

    # unique view count inside each voxel segment
    new_view = new_voxel.clone()
    same_voxel = ~new_voxel[1:]
    new_view[1:] = new_view[1:] | ((v_s[1:] != v_s[:-1]) & same_voxel)
    view_count = torch.zeros(M, device=v.device, dtype=torch.int32).scatter_add_(0, seg_ids, new_view.to(torch.int32))

    kk_u = (keys_s[new_voxel] & ((1 << 21) - 1)).to(torch.int32) - BIAS
    jj_u = ((keys_s[new_voxel] >> 21) & ((1 << 21) - 1)).to(torch.int32) - BIAS
    ii_u = ((keys_s[new_voxel] >> 42) & ((1 << 21) - 1)).to(torch.int32) - BIAS
    return ii_u, jj_u, kk_u, count, sum_i, max_i.to(torch.float32), view_count

def make_angle_grids(az_bins_full: int, el_bins_full: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RA-style angle grids (radians) using full bin counts then DC-removed index range [-N/2+1, ..., N/2-1].
    Returns tensors of shape (az_bins_full-1,), (el_bins_full-1,).
    """
    eps = 1e-6
    t_az = torch.arange(-az_bins_full // 2 + 1, az_bins_full // 2, dtype=torch.float32)
    t_el = torch.arange(-el_bins_full // 2 + 1, el_bins_full // 2, dtype=torch.float32)
    t_az = (2.0 / float(az_bins_full)) * t_az
    t_el = (2.0 / float(el_bins_full)) * t_el
    az_angles = torch.asin(torch.clamp(t_az, -1.0 + eps, 1.0 - eps)).to(device=device, dtype=torch.float32)
    el_angles = torch.asin(torch.clamp(t_el, -1.0 + eps, 1.0 - eps)).to(device=device, dtype=torch.float32)
    return az_angles, el_angles
def _chunk_list(items: List, chunk_size: int) -> List[List]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _merge_accumulator_data_dicts(dicts: List[Dict[Tuple[int, int, int], Dict[str, object]]]) -> Dict[Tuple[int, int, int], Dict[str, object]]:
    merged: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for d in dicts:
        for k, slot in d.items():
            vc = (
                int(slot.get("view_count", 0))
                if "view_count" in slot
                else (
                    int(slot.get("view_mask", 0)).bit_count()
                    if "view_mask" in slot
                    else len(slot.get("views", ()))
                )
            )
            if k not in merged:
                merged[k] = {
                    "count": int(slot["count"]),
                    "sum_intensity": float(slot["sum_intensity"]),
                    "max_intensity": float(slot["max_intensity"]),
                    "view_count": vc,
                }
            else:
                m = merged[k]
                m["count"] += int(slot["count"])
                m["sum_intensity"] += float(slot["sum_intensity"])
                if float(slot["max_intensity"]) > m["max_intensity"]:
                    m["max_intensity"] = float(slot["max_intensity"])
                m["view_count"] += vc
    return merged


def _process_views_subset(
        device_index: int,
        view_pairs: List[Tuple[float, Path]],
        start_view_id: int,
        voxel_size_m: float,
        cfar_train: Tuple[int, int, int],
        cfar_guard: Tuple[int, int, int],
        cfar_thresh: float,
        batch_size: int,
        min_range_m: float,
        NUM_ANT: int,
        NUM_ADC: int,
        NUM_AZIMUTH_BINS: int,
        NUM_ELEVATION_BINS: int,
        RANGE_RESOLUTION: float,
        MIN_RANGE_M: float,
        YAW_SIGN: float,
        FLIP_AZ: bool,
        OUTPUT_FOLDER: str,
        DEVICE: torch.device
    ) -> Dict[Tuple[int, int, int], Dict[str, object]]:
    # global DEVICE
    if torch.cuda.is_available() and device_index is not None and device_index >= 0:
        torch.cuda.set_device(device_index)
        DEVICE = torch.device(f'cuda:{device_index}')
    else:
        DEVICE = torch.device('cpu')

    print(f"[worker:{device_index}] Using device: {DEVICE}")
    # Set per-worker minimum range for near-field filtering
    print(f"[worker:{device_index}] Using min-range: {MIN_RANGE_M:.3f} m")
    try:
        # global MIN_RANGE_M
        MIN_RANGE_M = float(min_range_m)
        print(f"[worker:{device_index}] Using min-range: {MIN_RANGE_M:.3f} m")
    except Exception as e:
        print(f"[worker:{device_index}] warn: could not set MIN_RANGE_M: {e}")
    # Derive per-worker RANGE_RESOLUTION/NUM_ADC/NUM_ANT from configs to avoid default fallbacks
    try:
        configs_path = Path(OUTPUT_FOLDER) / "01_configs"
        cfg_file: Path | None = None
        # Prefer +0.0 view if present
        cfg0 = configs_path / "config_angle_+0.0.json"
        if cfg0.exists():
            cfg_file = cfg0
        elif len(view_pairs) > 0:
            ang0 = float(view_pairs[0][0])
            sign = "+" if ang0 >= 0 else ""
            cand = configs_path / (f"config_angle_{sign}{ang0:.1f}.json")
            if cand.exists():
                cfg_file = cand
        if cfg_file is None:
            files = sorted(configs_path.glob("config_angle_*.json"))
            if files:
                cfg_file = files[0]
        if cfg_file is not None:
            dr = _compute_range_resolution_from_config(cfg_file)
            if dr > 0:
                # global RANGE_RESOLUTION
                RANGE_RESOLUTION = float(dr)
                print(f"[worker:{device_index}] Derived DeltaR={RANGE_RESOLUTION:.6f} m from {cfg_file.name}")
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)
            if "numAdcSamples" in cfg:
                # global NUM_ADC
                NUM_ADC = int(cfg["numAdcSamples"])
                print(f"[worker:{device_index}] Set NUM_ADC={NUM_ADC}")
            if "tx_array" in cfg and "rx_array" in cfg:
                n_tx = len(cfg["tx_array"])
                n_rx = len(cfg["rx_array"])
                if n_tx == n_rx and n_tx > 0:
                    # global NUM_ANT
                    NUM_ANT = int(n_tx)
                    print(f"[worker:{device_index}] Set NUM_ANT={NUM_ANT}")
        else:
            print(f"[worker:{device_index}] No config files found; using defaults DeltaR={RANGE_RESOLUTION:.6f}, NUM_ADC={NUM_ADC}, NUM_ANT={NUM_ANT}")
    except Exception as e:
        print(f"[worker:{device_index}] warn: could not derive DeltaR/NUM_* from configs: {e}")
    accumulator = VoxelAccumulatorGPU(voxel_size_m=voxel_size_m, device=DEVICE)

    # Precompute angle grids once
    az_grid, el_grid = make_angle_grids(NUM_AZIMUTH_BINS, NUM_ELEVATION_BINS, device=DEVICE)

    view_id = start_view_id
    total_batches = (len(view_pairs) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc=f"[worker:{device_index}] batches", position=(device_index if isinstance(device_index, int) and device_index >= 0 else 0), dynamic_ncols=True)
    for chunk in _chunk_list(view_pairs, batch_size):
        batch_angles = [angle for angle, _ in chunk]
        batch_paths = [str(p) for _, p in chunk]
        if not batch_paths:
            continue

        # Heavy lifting in batch: TX/RX->VX and FFTs
        mag_batch, _ = fft_to_rae_magnitude_gpu_batch(batch_paths, NUM_ANT, NUM_ADC, NUM_AZIMUTH_BINS, NUM_ELEVATION_BINS, DEVICE)  # (B, Az, El, R)

        # Batched CFAR per-sample
        det_batch = simple_3d_cfar_gpu_batched_per_sample(
            mag_batch, train=cfar_train, guard=cfar_guard, threshold_factor=cfar_thresh
        )  # (B, Az, El, R)

        # Collect all points for the batch and reduce once using packed keys
        pts_i = []
        pts_j = []
        pts_k = []
        pts_int = []
        pts_vidx = []
        for b_idx, angle_deg in enumerate(batch_angles):
            mag_rae = mag_batch[b_idx]
            det_mask = det_batch[b_idx]
            x_t, y_t, z_t, intens_t = rae_indices_to_world_points_gpu(det_mask, mag_rae, angle_deg, az_grid, el_grid, return_torch=True, RANGE_RESOLUTION=RANGE_RESOLUTION, MIN_RANGE_M=MIN_RANGE_M, YAW_SIGN=YAW_SIGN, FLIP_AZ=FLIP_AZ)
            if x_t.numel() == 0:
                continue
            vs = voxel_size_m
            i = torch.floor(x_t / vs).to(torch.int32)
            j = torch.floor(y_t / vs).to(torch.int32)
            k = torch.floor(z_t / vs).to(torch.int32)
            pts_i.append(i)
            pts_j.append(j)
            pts_k.append(k)
            pts_int.append(intens_t.to(torch.float32))
            # Store per-point view index for per-voxel unique-view reduction
            pts_vidx.append(torch.full_like(i, b_idx, dtype=torch.int64))

        if len(pts_i) > 0:
            i_cat = torch.cat(pts_i)
            j_cat = torch.cat(pts_j)
            k_cat = torch.cat(pts_k)
            inten_cat = torch.cat(pts_int)
            vidx_cat = torch.cat(pts_vidx)

            ii_u, jj_u, kk_u, cnt_u, sum_u, max_u, vcnt_u = reduce_points_to_voxels_gpu_uniqueviews(
                i_cat, j_cat, k_cat, inten_cat, vidx_cat
            )

            ii_u = ii_u.cpu().numpy(); jj_u = jj_u.cpu().numpy(); kk_u = kk_u.cpu().numpy()
            cnt_u = cnt_u.cpu().numpy(); sum_u = sum_u.cpu().numpy(); max_u = max_u.cpu().numpy()
            vcnt_u = vcnt_u.cpu().numpy()

            for t in range(ii_u.size):
                key = (int(ii_u[t]), int(jj_u[t]), int(kk_u[t]))
                slot = accumulator.data.get(key)
                if slot is None:
                    accumulator.data[key] = {
                        "count": int(cnt_u[t]),
                        "sum_intensity": float(sum_u[t]),
                        "max_intensity": float(max_u[t]),
                        "view_count": int(vcnt_u[t]),
                    }
                else:
                    slot["count"] += int(cnt_u[t])
                    slot["sum_intensity"] += float(sum_u[t])
                    if float(max_u[t]) > slot["max_intensity"]:
                        slot["max_intensity"] = float(max_u[t])
                    slot["view_count"] = int(slot.get("view_count", 0)) + int(vcnt_u[t])

        view_id += len(batch_angles)

        # Update per batch
        pbar.update(1)

        # Avoid empty_cache() inside loop; it can slow things down due to syncs/heap churn.

    pbar.close()

    # Return raw data dict for merging in parent
    return accumulator.data



@torch.no_grad()
def connected_components_3d_gpu(binary_mask: torch.Tensor,
                                min_size: int = 10,
                                max_iters: int = 128,
                                connectivity: int = 26,
                                verbose: bool = True) -> torch.Tensor:
    """
    Fully vectorized GPU connected components for 3D boolean masks.
    - Uses iterative min-label propagation with 26-connectivity (3x3x3 neighborhood).
    - Returns a boolean mask keeping only components with size >= min_size.

    Args:
        binary_mask: 3D bool tensor [A, E, R] on CUDA (or CPU; will move to CUDA if available).
        min_size: minimum component size to keep (in voxels)
        max_iters: safety cap on iterations (usually converges well before this)
        connectivity: 26 only (recommended). 6-connectivity can be added if needed.
        verbose: print progress

    Returns:
        keep_mask (bool, same shape)
    """
    assert binary_mask.ndim == 3, "binary_mask must be 3D [A,E,R]"
    device = binary_mask.device
    if device.type != 'cuda' and torch.cuda.is_available():
        binary_mask = binary_mask.to('cuda')
        device = binary_mask.device

    fg = binary_mask
    if not torch.any(fg):
        return torch.zeros_like(fg, dtype=torch.bool)

    # --- initialize per-voxel unique labels for foreground
    # Use float64 to preserve exact integers > 2^24 during pooling steps.
    numel = fg.numel()
    # 1..N per-voxel ids, 0 for background
    init_ids = torch.arange(1, numel + 1, device=device, dtype=torch.float64).reshape(fg.shape)
    labels = torch.where(fg, init_ids, torch.zeros(1, device=device, dtype=torch.float64))

    # Big value for background so it doesn't affect neighbor-min
    BIG = float(numel + 1)

    def min_pool3d_26(x_float64: torch.Tensor) -> torch.Tensor:
        """Min over 3x3x3 neighborhood via negative max_pool3d. Keeps shape."""
        # Replace background (0) with BIG so it doesn't win the min
        x = torch.where(x_float64 > 0, x_float64, torch.full((), BIG, device=device, dtype=torch.float64))
        x = x.unsqueeze(0).unsqueeze(0)  # (N=1,C=1,D,H,W)
        # min = -max(-x)
        m = -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
        return m.squeeze(0).squeeze(0)

    if connectivity != 26:
        if verbose:
            print("[GPU-CC] Only connectivity=26 is implemented efficiently here; using 26.")
        connectivity = 26

    # --- iterate until labels stop decreasing (fixed point)
    last_changed = None
    for it in range(max_iters):
        nbr_min = min_pool3d_26(labels)
        new_labels = torch.where(fg, torch.minimum(labels, nbr_min), labels)
        changed = torch.count_nonzero(new_labels != labels)

        labels = new_labels
        if verbose and (it % 5 == 0 or changed == 0):
            print(f"  [GPU-CC] iter {it:3d} | changed voxels = {int(changed):,}")

        if changed == 0:
            break
        last_changed = int(changed)

    if verbose and last_changed is not None and last_changed > 0:
        print("  [GPU-CC] converged.")

    # --- count component sizes
    labels_int = labels.long()
    fg_idx = labels_int > 0
    if not torch.any(fg_idx):
        return torch.zeros_like(fg, dtype=torch.bool)

    unique_labels, inv = torch.unique(labels_int[fg_idx], sorted=False, return_inverse=True)
    counts = torch.bincount(inv, minlength=unique_labels.numel())

    # Map counts back to full grid with a LUT
    lut_size = int(labels_int.max().item()) + 1
    lut = torch.zeros(lut_size, device=device, dtype=torch.int64)
    lut[unique_labels] = counts

    keep = torch.zeros_like(fg, dtype=torch.bool)
    keep_vals = lut[labels_int[fg_idx]] >= int(min_size)
    keep[fg_idx] = keep_vals

    return keep


def visualize_points(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        values: np.ndarray,
        title: str = "Global Aggregated Voxel Cloud (GPU)",
        percentile: float = 95.0,
        COLOR_BY_INTENSITY: bool = True
    ):
    # Bright-only visualization: keep top 5% intensities and map with vmin from full distribution (p95)
    vmin_full = None
    vmax_full = None
    if values is not None and values.size == x.size and values.size > 0:
        # Use original global min/max for color scaling
        vmin_full = float(values.min())
        vmax_full = float(values.max())
        # Keep only top (100 - percentile)% brightest points
        p = float(np.clip(percentile, 0.0, 100.0))
        thr = float(np.percentile(values, p))
        keep_mask = values >= thr
        if not np.any(keep_mask):
            keep_mask = values >= values.max()  # ensure at least the brightest point remains
        x, y, z, values = x[keep_mask], y[keep_mask], z[keep_mask], values[keep_mask]

    pts = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if COLOR_BY_INTENSITY and values is not None and values.size == x.size:
        # Use vmin/vmax from pre-threshold distribution to preserve original colors
        vmin = vmin_full if vmin_full is not None else (np.percentile(values, 95.0) if values.size > 0 else 0.0)
        vmax = vmax_full if vmax_full is not None else (values.max() if values.size > 0 else 1.0)
        if vmax <= vmin:
            vals_n = np.ones_like(values, dtype=np.float32)
        else:
            vals_n = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
        cols = cm.get_cmap("plasma")(vals_n)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(cols)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.06, 0.06, 0.06])
    opt.point_size = 4.0
    print("[viz] Q to close")
    vis.run()
    vis.destroy_window()