"""
Configuration classes for FMCW ADC accumulation.
"""

import json
from dataclasses import dataclass
from typing import Optional, Callable, TYPE_CHECKING
import numpy as np
import drjit as dr
import mitsuba as mi

if TYPE_CHECKING:
    # Type hints only, not evaluated at runtime
    pass


@dataclass
class FMCWConfig:
    """FMCW radar and SBR simulation configuration"""

    # FMCW parameters
    center_freq: float          # f0: Center frequency (Hz)
    bandwidth: float            # Chirp bandwidth (Hz)
    chirp_slope: float          # S: Chirp slope (Hz/s)
    chirp_duration: float       # Chirp duration (seconds)

    # ADC parameters
    num_adc_samples: int        # K: Number of ADC samples per chirp
    adc_sample_rate: float      # Samples per second

    # RX array
    num_rx: int                 # Number of RX elements

    # Optional parameters with defaults
    adc_start_time: float = 0.0 # ADC start time (seconds) - defines minimum range

    # SBR parameters
    max_depth: int = 10         # Maximum bounce depth
    rr_depth: int = 3           # Russian roulette starts at this depth
    rr_threshold: float = 0.1   # Threshold for RR probability computation
    samples_per_tx: int = 1000000  # Rays launched per TX
    min_gain: float = 1e-10     # Minimum E-field magnitude to continue

    # Sampling allocation (for mixed cosine + edge sampling)
    edge_allocation_fraction: float = 0.2   # Fraction of samples for edge diffraction (0.1-0.4)
    min_edge_samples: int = 50              # Minimum edge samples regardless of allocation
    min_cosine_samples: int = 50            # Minimum surface samples regardless of allocation

    # Russian Roulette parameters
    rr_max_prob: float = 0.95   # Maximum RR survival probability (caps throughput)

    # Numerical stability
    shadow_ray_epsilon: float = 1e-6  # Absolute offset (meters) to prevent self-intersection
                                       # At 77 GHz (lambda~3.9mm), 1e-6m = 1mum ~ 0.03% wavelength
                                       # CRITICAL: Must be << wavelength for phase accuracy!

    # Material interaction
    reset_tube_on_diffuse: bool = True  # Reset tube weight for diffuse (Sionna-RT style)

    # Physical constants
    c: float = 299792458.0      # Speed of light (m/s)

    # TX Phase Coherence (toggleable)
    # Controls whether TX elements trace the same paths or different paths:
    # - True (default): All TXs trace SAME paths (shared ray directions) -> full phase coherence
    # - False: Each TX traces DIFFERENT paths (independent rays) -> RX coherence only
    # Note: RX phase coherence is ALWAYS enabled (all paths evaluated by all RXs)
    use_coherent_tx: bool = True

    # Path loss
    enable_path_loss: bool = True  # Apply free-space path loss (1/R decay) to E-field for physically-based rendering
    path_loss_exponent: float = 2.0  # Exponent for path loss: 1=E-field decay (1/d), 2=power decay (1/d^2)

    @property
    def wavelength(self) -> float:
        """Compute wavelength from center frequency: lambda = c / f0"""
        return self.c / self.center_freq

    @classmethod
    def from_json(cls, config_path: str) -> 'FMCWConfig':
        """Load configuration from JSON file (cascaded_frame_*.json style)"""
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Handle different JSON field names
        # carrier frequency
        center_freq = (data.get('carrierFrequency') or
                      data.get('center_freq') or
                      data.get('start_freq'))
        if center_freq is None:
            raise ValueError("Config must specify 'carrierFrequency', 'center_freq', or 'start_freq'")

        # chirp slope
        chirp_slope = (data.get('freqSlope') or
                      data.get('chirp_slope') or
                      data.get('freq_slope'))
        if chirp_slope is None:
            raise ValueError("Config must specify 'freqSlope' or 'chirp_slope'")

        # ADC samples
        num_adc_samples = (data.get('numAdcSamples') or
                          data.get('num_adc_samples'))
        if num_adc_samples is None:
            raise ValueError("Config must specify 'numAdcSamples' or 'num_adc_samples'")

        # Sample rate
        sample_rate = (data.get('sampleRate') or
                      data.get('adc_sample_rate') or
                      data.get('sample_rate'))

        # Compute derived parameters
        # Chirp duration from rampEndTime or compute from other params
        if 'rampEndTime' in data:
            chirp_duration = data['rampEndTime']
        elif 'chirp_duration' in data:
            chirp_duration = data['chirp_duration']
        else:
            # Compute from sample rate and num samples
            if sample_rate is None:
                raise ValueError("Cannot determine chirp duration")
            chirp_duration = num_adc_samples / sample_rate

        # Bandwidth from slope and duration
        bandwidth = chirp_slope * chirp_duration

        # Final sample rate
        if sample_rate is None:
            sample_rate = num_adc_samples / chirp_duration

        # Number of RX elements
        if 'rx_array' in data:
            num_rx = len(data['rx_array'])
        elif 'num_rx' in data:
            num_rx = data['num_rx']
        else:
            raise ValueError("Config must specify 'rx_array' or 'num_rx'")

        # ADC start time (defines minimum range)
        adc_start_time = (data.get('adcStartTime') or
                          data.get('adc_start_time') or
                          0.0)

        return cls(
            center_freq=center_freq,
            bandwidth=bandwidth,
            chirp_slope=chirp_slope,
            chirp_duration=chirp_duration,
            num_adc_samples=num_adc_samples,
            adc_sample_rate=sample_rate,
            adc_start_time=adc_start_time,
            num_rx=num_rx,
            max_depth=data.get('max_depth', 10),
            rr_depth=data.get('rr_depth', 3),
            samples_per_tx=data.get('samples_per_tx', 1000000),
            min_gain=data.get('min_gain', 1e-10),
        )


@dataclass
class RxArray:
    """Receiver array configuration and geometry"""

    positions: 'mi.Point3f'       # [NR, 3] RX element positions
    orientations: 'mi.Vector3f'   # [NR, 3] RX element boresight direction (3D unit vector)
    polarization: Optional['mi.UInt32'] = None  # [NR] Polarization: 0=H (horizontal), 1=V (vertical)
    pattern: Optional[Callable] = None  # Element pattern function (legacy)
    pattern_data: Optional[np.ndarray] = None  # Raw antenna pattern data if loaded from file
    pattern_loader: Optional = None  # AntennaPatternLoader for polarization-aware evaluation

    @property
    def num_elements(self) -> int:
        """Number of RX elements"""
        return dr.width(self.positions.x)

    @classmethod
    def from_config(cls, config_data: dict) -> 'RxArray':
        """Load RX array from config dictionary"""
        # Check if config has rx_array with individual elements
        if 'rx_array' in config_data:
            rx_elements = config_data['rx_array']
            num_rx = len(rx_elements)

            # Extract positions (convert from mm to meters if needed)
            positions_list = []
            orientations_list = []
            polarization_list = []

            for elem in rx_elements:
                # Position
                if 'pos_mm' in elem:
                    # Convert mm to meters
                    pos = np.array(elem['pos_mm'], dtype=np.float32) / 1000.0
                elif 'pos' in elem:
                    pos = np.array(elem['pos'], dtype=np.float32)
                else:
                    raise ValueError("RX element must have 'pos_mm' or 'pos'")

                positions_list.append(pos)

                # Orientation (boresight)
                if 'boresight' in elem:
                    ori = np.array(elem['boresight'], dtype=np.float32)
                elif 'orientation' in elem:
                    ori = np.array(elem['orientation'], dtype=np.float32)
                else:
                    # Default to +y (forward/boresight direction)
                    ori = np.array([0, 1, 0], dtype=np.float32)

                orientations_list.append(ori)

                # Polarization: 'V' -> 1, 'H' -> 0, default to V if not specified
                if 'polarization' in elem:
                    pol = 1 if elem['polarization'].upper() == 'V' else 0
                else:
                    pol = 1  # Default to vertical polarization

                polarization_list.append(pol)

            pos_array = np.array(positions_list, dtype=np.float32)
            ori_array = np.array(orientations_list, dtype=np.float32)
            pol_array = np.array(polarization_list, dtype=np.uint32)

        # Legacy format with flat arrays
        elif 'rx_positions' in config_data or 'rx_pos' in config_data:
            if 'rx_positions' in config_data:
                pos_array = np.array(config_data['rx_positions'], dtype=np.float32)
            else:
                pos_array = np.array(config_data['rx_pos'], dtype=np.float32)

            # Ensure shape is [NR, 3]
            if pos_array.ndim == 1:
                pos_array = pos_array.reshape(1, 3)

            # Extract orientations (default to +y forward if not specified)
            if 'rx_orientations' in config_data:
                ori_array = np.array(config_data['rx_orientations'], dtype=np.float32)
            else:
                # Default to +y direction (forward/boresight)
                ori_array = np.tile([0, 1, 0], (pos_array.shape[0], 1)).astype(np.float32)

            # Extract polarization (default to V if not specified)
            if 'rx_polarization' in config_data:
                pol_array = np.array([
                    1 if p.upper() == 'V' else 0
                    for p in config_data['rx_polarization']
                ], dtype=np.uint32)
            else:
                # Default to vertical polarization
                pol_array = np.ones(pos_array.shape[0], dtype=np.uint32)

        else:
            raise ValueError("Config must specify 'rx_array', 'rx_positions', or 'rx_pos'")

        # Mitsuba expects shape (3, N) not (N, 3), so transpose
        positions = mi.Point3f(pos_array.T)
        orientations = mi.Vector3f(ori_array.T)
        polarization = mi.UInt32(pol_array)

        return cls(
            positions=positions,
            orientations=orientations,
            polarization=polarization,
            pattern=None,  # Will be set later
            pattern_data=None
        )

    @classmethod
    def from_json(cls, config_path: str) -> 'RxArray':
        """Load RX array from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls.from_config(data)


@dataclass
class TxArray:
    """Transmitter array configuration and geometry"""

    positions: 'mi.Point3f'       # [NT, 3] TX element positions
    orientations: 'mi.Vector3f'   # [NT, 3] TX element boresight direction (3D unit vector)
    power: 'mi.Float'             # [NT] TX power per element (Watts)
    polarization: Optional['mi.UInt32'] = None  # [NT] Polarization: 0=H (horizontal), 1=V (vertical)
    pattern: Optional[Callable] = None  # Element pattern function (legacy)
    pattern_data: Optional[np.ndarray] = None  # Raw antenna pattern data if loaded from file
    pattern_loader: Optional = None  # AntennaPatternLoader for polarization-aware evaluation

    @property
    def num_elements(self) -> int:
        """Number of TX elements"""
        return dr.width(self.positions.x)

    @classmethod
    def from_config(cls, config_data: dict, default_power: float = 1.0) -> 'TxArray':
        """Load TX array from config dictionary"""
        # Check if config has tx_array with individual elements
        if 'tx_array' in config_data:
            tx_elements = config_data['tx_array']
            num_tx = len(tx_elements)

            # Extract positions (convert from mm to meters if needed)
            positions_list = []
            orientations_list = []
            power_list = []
            polarization_list = []

            for elem in tx_elements:
                # Position
                if 'pos_mm' in elem:
                    # Convert mm to meters
                    pos = np.array(elem['pos_mm'], dtype=np.float32) / 1000.0
                elif 'pos' in elem:
                    pos = np.array(elem['pos'], dtype=np.float32)
                else:
                    raise ValueError("TX element must have 'pos_mm' or 'pos'")

                positions_list.append(pos)

                # Orientation (boresight)
                if 'boresight' in elem:
                    ori = np.array(elem['boresight'], dtype=np.float32)
                elif 'orientation' in elem:
                    ori = np.array(elem['orientation'], dtype=np.float32)
                else:
                    # Default to +y (forward/boresight direction)
                    ori = np.array([0, 1, 0], dtype=np.float32)

                orientations_list.append(ori)

                # Power
                if 'power' in elem:
                    pwr = elem['power']
                elif 'power_dBm' in elem:
                    # Convert dBm to watts: P(W) = 10^((P_dBm - 30)/10)
                    pwr = 10**((elem['power_dBm'] - 30) / 10)
                else:
                    pwr = default_power

                power_list.append(pwr)

                # Polarization: 'V' -> 1, 'H' -> 0, default to V if not specified
                if 'polarization' in elem:
                    pol = 1 if elem['polarization'].upper() == 'V' else 0
                else:
                    pol = 1  # Default to vertical polarization

                polarization_list.append(pol)

            pos_array = np.array(positions_list, dtype=np.float32)
            ori_array = np.array(orientations_list, dtype=np.float32)
            power_array = np.array(power_list, dtype=np.float32)
            pol_array = np.array(polarization_list, dtype=np.uint32)

        # Legacy format with flat arrays
        elif 'tx_positions' in config_data or 'tx_pos' in config_data:
            if 'tx_positions' in config_data:
                pos_array = np.array(config_data['tx_positions'], dtype=np.float32)
            else:
                pos_array = np.array(config_data['tx_pos'], dtype=np.float32)

            # Ensure shape is [NT, 3]
            if pos_array.ndim == 1:
                pos_array = pos_array.reshape(1, 3)

            # Extract orientations (default to +y forward if not specified)
            if 'tx_orientations' in config_data:
                ori_array = np.array(config_data['tx_orientations'], dtype=np.float32)
            else:
                # Default to +y direction (forward/boresight)
                ori_array = np.tile([0, 1, 0], (pos_array.shape[0], 1)).astype(np.float32)

            # Extract power
            if 'tx_power' in config_data:
                power_array = np.array(config_data['tx_power'], dtype=np.float32)
                if power_array.size == 1:
                    power_array = np.full(pos_array.shape[0], power_array.item(), dtype=np.float32)
            else:
                power_array = np.full(pos_array.shape[0], default_power, dtype=np.float32)

            # Extract polarization (default to V if not specified)
            if 'tx_polarization' in config_data:
                pol_array = np.array([
                    1 if p.upper() == 'V' else 0
                    for p in config_data['tx_polarization']
                ], dtype=np.uint32)
            else:
                # Default to vertical polarization
                pol_array = np.ones(pos_array.shape[0], dtype=np.uint32)

        else:
            raise ValueError("Config must specify 'tx_array', 'tx_positions', or 'tx_pos'")

        # Mitsuba expects shape (3, N) not (N, 3), so transpose
        positions = mi.Point3f(pos_array.T)
        orientations = mi.Vector3f(ori_array.T)
        power = mi.Float(power_array)
        polarization = mi.UInt32(pol_array)

        return cls(
            positions=positions,
            orientations=orientations,
            power=power,
            polarization=polarization,
            pattern=None,  # Will be set later
            pattern_data=None
        )

    @classmethod
    def from_json(cls, config_path: str, default_power: float = 1.0) -> 'TxArray':
        """Load TX array from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        return cls.from_config(data, default_power)


# ==============================================================================
# Quaternion Utilities
# ==============================================================================

def boresight_to_quaternion(boresight: np.ndarray) -> np.ndarray:
    """
    Convert boresight direction vector to quaternion (w, x, y, z).
    Aligns local +Y axis to boresight direction with zero roll.

    Coordinate Convention:
        World and Local Frames use +Y as forward/boresight direction.
        This ensures consistency with rigid body transformations.

    Args:
        boresight: [3] Direction vector (will be normalized)

    Returns:
        [4] Quaternion as (w, x, y, z) in numpy array

    Example:
        >>> boresight = np.array([1.0, 0.0, 0.0])  # Point in +X direction
        >>> quat = boresight_to_quaternion(boresight)
        >>> print(quat)  # Quaternion that rotates +Y to +X
    """
    # Normalize boresight
    by = boresight / (np.linalg.norm(boresight) + 1e-20)
    y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # World +Y (forward)

    # Compute dot product (cosine of angle)
    c = float(np.clip(np.dot(y_ref, by), -1.0, 1.0))

    # Special case: boresight is already +Y
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Special case: boresight is -Y (180 degree flip)
    if c < -1.0 + 1e-12:
        # Rotate 180deg around Z-axis (preserves up direction)
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # General case: compute rotation axis and angle
    axis = np.cross(y_ref, by)
    axis /= (np.linalg.norm(axis) + 1e-20)
    angle = np.arccos(c)

    # Convert to quaternion: q = [cos(theta/2), sin(theta/2) * axis]
    s = np.sin(angle / 2.0)
    w = np.cos(angle / 2.0)
    x, y, z = axis * s

    return np.array([w, x, y, z], dtype=np.float32)


def quat_to_drjit(quat_np: np.ndarray) -> 'mi.Vector4f':
    """
    Convert numpy quaternion to Dr.Jit Vector4f.

    Args:
        quat_np: [4] or [N, 4] Quaternion(s) as numpy array (w, x, y, z)

    Returns:
        mi.Vector4f quaternion(s)
    """
    if quat_np.ndim == 1:
        # Single quaternion
        # FIXED: Vector4f expects (x, y, z, w) order, not (w, x, y, z)!
        # Convert numpy scalars to Python floats for mi.Vector4f constructor
        return mi.Vector4f(float(quat_np[1]), float(quat_np[2]), float(quat_np[3]), float(quat_np[0]))
    else:
        # Multiple quaternions [N, 4] -> need to transpose to [4, N] for Mitsuba
        # FIXED: Reorder components from (w,x,y,z) to (x,y,z,w)
        return mi.Vector4f(
            mi.Float(quat_np[:, 1]),  # x component
            mi.Float(quat_np[:, 2]),  # y component
            mi.Float(quat_np[:, 3]),  # z component
            mi.Float(quat_np[:, 0])   # w component
        )


def quat_rotate_vector(quat: 'mi.Vector4f', vec: 'mi.Vector3f') -> 'mi.Vector3f':
    """
    Rotate vector by quaternion using Dr.Jit operations.

    q * v * q^(-1) in quaternion multiplication
    Equivalent to: v' = v + 2*w*(u x v) + 2*(u x (u x v))
    where q = [w, u] and u = [x, y, z]

    Args:
        quat: [N, 4] Quaternion as (w, x, y, z)
        vec: [N, 3] Vector to rotate

    Returns:
        [N, 3] Rotated vector
    """
    # Normalize quaternion
    quat_len_sq = (quat.w * quat.w + quat.x * quat.x +
                   quat.y * quat.y + quat.z * quat.z)
    quat_len = dr.sqrt(quat_len_sq)
    quat_len_safe = dr.maximum(quat_len, 1e-12)

    w = quat.w / quat_len_safe
    x = quat.x / quat_len_safe
    y = quat.y / quat_len_safe
    z = quat.z / quat_len_safe

    # Extract vector components
    vx, vy, vz = vec.x, vec.y, vec.z

    # u = [x, y, z] (imaginary part of quaternion)
    # Compute u x v
    ux_v_x = y * vz - z * vy
    ux_v_y = z * vx - x * vz
    ux_v_z = x * vy - y * vx

    # Compute u x (u x v)
    ux_ux_v_x = y * ux_v_z - z * ux_v_y
    ux_ux_v_y = z * ux_v_x - x * ux_v_z
    ux_ux_v_z = x * ux_v_y - y * ux_v_x

    # v' = v + 2*w*(u x v) + 2*(u x (u x v))
    v_rot_x = vx + 2.0 * w * ux_v_x + 2.0 * ux_ux_v_x
    v_rot_y = vy + 2.0 * w * ux_v_y + 2.0 * ux_ux_v_y
    v_rot_z = vz + 2.0 * w * ux_v_z + 2.0 * ux_ux_v_z

    return mi.Vector3f(v_rot_x, v_rot_y, v_rot_z)


def quat_inverse(quat: 'mi.Vector4f') -> 'mi.Vector4f':
    """
    Compute inverse (conjugate for unit quaternions) of quaternion.

    For unit quaternions: q^(-1) = q* = [w, -x, -y, -z]

    Args:
        quat: [N, 4] Quaternion (stored as quat.x, quat.y, quat.z, quat.w)

    Returns:
        [N, 4] Inverse quaternion

    Note:
        mi.Vector4f constructor uses (x, y, z, w) order, NOT (w, x, y, z)!
    """
    # FIXED: Vector4f constructor expects (x, y, z, w) order
    return mi.Vector4f(-quat.x, -quat.y, -quat.z, quat.w)


def quat_multiply(q1: 'mi.Vector4f', q2: 'mi.Vector4f') -> 'mi.Vector4f':
    """
    Multiply two quaternions using Hamilton product: q1 * q2

    The result rotates by q2 THEN q1 (right-to-left application).
    This is useful for composing rotations in rigid body transformations.

    Args:
        q1: Left quaternion (x, y, z, w)
        q2: Right quaternion (x, y, z, w)

    Returns:
        Product quaternion (x, y, z, w)

    Note:
        mi.Vector4f stores as (x, y, z, w), NOT (w, x, y, z)!

    Example:
        >>> # Rotate 45deg around Z, then 30deg around X
        >>> qz = axis_angle_to_quaternion([0,0,1], np.pi/4)
        >>> qx = axis_angle_to_quaternion([1,0,0], np.pi/6)
        >>> q_combined = quat_multiply(quat_to_drjit(qx), quat_to_drjit(qz))
    """
    # Extract components (stored as x,y,z,w)
    w1, x1, y1, z1 = q1.w, q1.x, q1.y, q1.z
    w2, x2, y2, z2 = q2.w, q2.x, q2.y, q2.z

    # Hamilton product
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return mi.Vector4f(x, y, z, w)


def quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (numpy version): q1 * q2

    Args:
        q1: [4] Quaternion (w, x, y, z) in numpy
        q2: [4] Quaternion (w, x, y, z) in numpy

    Returns:
        [4] Product quaternion (w, x, y, z) in numpy

    Note:
        Numpy quaternions stored as (w, x, y, z), different from mi.Vector4f!
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z], dtype=np.float32)


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to quaternion.

    Args:
        axis: [3] Rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        [4] Quaternion (w, x, y, z) in numpy

    Example:
        >>> # 45deg rotation around Z-axis
        >>> quat = axis_angle_to_quaternion(np.array([0, 0, 1]), np.pi/4)
        >>> print(quat)  # [0.9239, 0, 0, 0.3827]
    """
    # Normalize axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        # Zero rotation
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axis_normalized = axis / axis_norm

    # Convert to quaternion: q = [cos(theta/2), sin(theta/2) * axis]
    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    w = c
    x = axis_normalized[0] * s
    y = axis_normalized[1] * s
    z = axis_normalized[2] * s

    return np.array([w, x, y, z], dtype=np.float32)


def euler_to_quaternion(
    pitch_deg: float,
    roll_deg: float,
    yaw_deg: float,
    order: str = 'ZYX'
) -> np.ndarray:
    """
    Convert Euler angles (degrees) to quaternion.

    Args:
        pitch_deg: Rotation around X-axis (degrees) - nose up/down
        roll_deg: Rotation around Y-axis (degrees) - wing tilt
        yaw_deg: Rotation around Z-axis (degrees) - heading
        order: Rotation order (default 'ZYX' = yaw-pitch-roll)

    Returns:
        [4] Quaternion (w, x, y, z) in numpy

    Note:
        This is a convenience wrapper. For precise control,
        use axis_angle_to_quaternion() and quat_multiply_np()

    Example:
        >>> # 10deg pitch up, 5deg yaw left
        >>> quat = euler_to_quaternion(pitch_deg=10, roll_deg=0, yaw_deg=5)
    """
    # Convert to radians
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    yaw = np.radians(yaw_deg)

    # Create individual quaternions
    qx = axis_angle_to_quaternion(np.array([1.0, 0.0, 0.0]), pitch)
    qy = axis_angle_to_quaternion(np.array([0.0, 1.0, 0.0]), roll)
    qz = axis_angle_to_quaternion(np.array([0.0, 0.0, 1.0]), yaw)

    # Compose based on order (intrinsic rotations)
    if order == 'ZYX':
        # Yaw (Z) first, then pitch (Y), then roll (X)
        q = quat_multiply_np(quat_multiply_np(qz, qy), qx)
    elif order == 'XYZ':
        # Roll (X) first, then pitch (Y), then yaw (Z)
        q = quat_multiply_np(quat_multiply_np(qx, qy), qz)
    elif order == 'YXZ':
        # Pitch (Y) first, then roll (X), then yaw (Z)
        q = quat_multiply_np(quat_multiply_np(qy, qx), qz)
    else:
        raise ValueError(f"Unsupported rotation order: {order}. Use 'ZYX', 'XYZ', or 'YXZ'")

    return q
