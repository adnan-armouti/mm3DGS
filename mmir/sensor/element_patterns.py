"""
Antenna element pattern functions for TX and RX arrays.

These functions compute the complex gain pattern of antenna elements
as a function of incident/outgoing direction.
"""

import drjit as dr
import mitsuba as mi
from typing import Tuple


def isotropic_pattern(direction: mi.Vector3f,
                     orientation: mi.Vector3f) -> mi.Float:
    """
    Isotropic antenna pattern (constant gain in all directions).

    Args:
        direction: [N,3] Incident/outgoing direction (unit vector)
        orientation: [N,3] Element orientation (not used for isotropic)

    Returns:
        mi.Float: [N] Real gain (complex phase = 0)
    """
    # Return unit gain
    return dr.full(mi.Float, 1.0, dr.width(direction.x))


def cosine_pattern(direction: mi.Vector3f,
                  orientation: mi.Vector3f,
                  power: float = 1.0) -> mi.Float:
    """
    Cosine pattern (Lambertian-style antenna).
    Gain ~ cos(theta) where theta is angle from element normal.

    Args:
        direction: [N,3] Incident/outgoing direction (unit vector)
        orientation: [N,3] Element orientation (normal direction)
        power: Exponent for cosine (1.0 = Lambertian, higher = narrower)

    Returns:
        mi.Float: [N] Real gain
    """
    # Normalize directions
    direction = dr.normalize(direction)
    orientation = dr.normalize(orientation)

    # Compute cos(theta)
    cos_theta = dr.dot(direction, orientation)

    # Clamp to [0, 1] (only forward hemisphere)
    cos_theta = dr.maximum(cos_theta, 0.0)

    # Apply power law
    gain = dr.power(cos_theta, power)

    return gain


def patch_antenna_pattern(direction: mi.Vector3f,
                         orientation: mi.Vector3f,
                         half_power_beamwidth: float = 65.0) -> mi.Float:
    """
    Approximate patch antenna pattern with Gaussian-like beamwidth.

    Args:
        direction: [N,3] Incident/outgoing direction (unit vector)
        orientation: [N,3] Element orientation (boresight direction)
        half_power_beamwidth: HPBW in degrees

    Returns:
        mi.Float: [N] Real gain
    """
    # Normalize directions
    direction = dr.normalize(direction)
    orientation = dr.normalize(orientation)

    # Compute angle from boresight
    cos_theta = dr.dot(direction, orientation)
    theta = dr.acos(dr.clamp(cos_theta, -1.0, 1.0))  # radians

    # Convert HPBW to standard deviation
    hpbw_rad = half_power_beamwidth * dr.pi / 180.0
    sigma = hpbw_rad / (2.0 * dr.sqrt(2.0 * dr.log(2.0)))

    # Gaussian envelope
    gain = dr.exp(-0.5 * (theta / sigma)**2)

    # Ensure zero gain in back hemisphere
    gain = dr.select(cos_theta > 0, gain, 0.0)

    return gain


def horn_antenna_pattern(direction: mi.Vector3f,
                        orientation: mi.Vector3f,
                        E_plane_beamwidth: float = 30.0,
                        H_plane_beamwidth: float = 45.0) -> Tuple[mi.Float, mi.Float]:
    """
    Realistic pyramidal horn antenna pattern based on aperture field theory.

    Implements the standard sinc function pattern for rectangular aperture,
    with different beamwidths in E and H planes as typical for horn antennas
    at mmWave frequencies.

    Args:
        direction: [N,3] Incident/outgoing direction (unit vector)
        orientation: [N,3] Element orientation (boresight direction)
        E_plane_beamwidth: Beamwidth in E-plane (degrees)
        H_plane_beamwidth: Beamwidth in H-plane (degrees)

    Returns:
        Tuple[mi.Float, mi.Float]: [N] Real gains for E and H polarizations
    """
    # Normalize directions
    direction = dr.normalize(direction)
    orientation = dr.normalize(orientation)

    # Build local coordinate system for antenna
    # Coordinate Convention: Y-axis is boresight (forward), matching world frame
    # This ensures consistency with evaluate_linear() and rigid body transformations
    y_axis = orientation

    # Choose arbitrary perpendicular vectors for X and Z
    # X-axis is H-plane (azimuth), Z-axis is E-plane (elevation)
    # Find a vector not parallel to Y
    ref_vec = dr.select(
        dr.abs(y_axis.y) < 0.9,
        mi.Vector3f(0, 1, 0),  # Use world Y if antenna not pointing forward/back
        mi.Vector3f(1, 0, 0)   # Otherwise use world X
    )

    # Gram-Schmidt orthogonalization
    x_axis = dr.normalize(ref_vec - dr.dot(ref_vec, y_axis) * y_axis)
    z_axis = dr.cross(x_axis, y_axis)

    # Project direction onto local coordinate system
    # This gives us the direction cosines in antenna frame
    u = dr.dot(direction, x_axis)  # H-plane component (azimuth)
    v = dr.dot(direction, z_axis)  # E-plane component (elevation)
    w = dr.dot(direction, y_axis)  # Boresight component (forward)

    # Convert beamwidths to aperture dimensions (in wavelength units)
    # Using standard relationship: HPBW ~ 51lambda/L for uniform aperture
    # where L is aperture dimension in wavelengths
    # For horn: HPBW ~ 58lambda/L is more typical
    E_plane_bw_rad = E_plane_beamwidth * dr.pi / 180.0
    H_plane_bw_rad = H_plane_beamwidth * dr.pi / 180.0

    # Effective aperture dimensions (normalized)
    # These determine the sinc pattern width
    a_eff = 58.0 * dr.pi / 180.0 / H_plane_bw_rad  # H-plane aperture
    b_eff = 58.0 * dr.pi / 180.0 / E_plane_bw_rad  # E-plane aperture

    # Compute pattern using sinc functions
    # For pyramidal horn: E(theta,phi) = E0 * sinc(ka*sin(theta)*cos(phi)/2) * sinc(kb*sin(theta)*sin(phi)/2)
    # where k = 2pi/lambda (normalized to 1 here)

    # Angular deviation from boresight
    sin_theta = dr.sqrt(u*u + v*v)  # sin(theta) in spherical coords

    # Avoid division by zero at boresight
    eps = 1e-10

    # H-plane pattern factor (along X-axis)
    arg_h = dr.pi * a_eff * u
    sinc_h = dr.select(
        dr.abs(arg_h) > eps,
        dr.sin(arg_h) / arg_h,
        mi.Float(1.0)
    )

    # E-plane pattern factor (along Y-axis)
    arg_e = dr.pi * b_eff * v
    sinc_e = dr.select(
        dr.abs(arg_e) > eps,
        dr.sin(arg_e) / arg_e,
        mi.Float(1.0)
    )

    # Combined pattern (product of sinc functions)
    # Note: Real horn patterns have slight amplitude taper, not perfectly uniform
    # We'll use a cosine taper factor for more realistic pattern
    taper = 0.8 + 0.2 * w  # Slight cosine taper (0.8 to 1.0)

    # Main lobe pattern
    pattern = dr.abs(sinc_h * sinc_e) * taper

    # Ensure zero gain in back hemisphere
    pattern = dr.select(w > 0, pattern, mi.Float(0.0))

    # Apply realistic gain factor
    # Typical horn antenna gain: G ~ 10-25 dBi at mmWave
    # Convert to directivity (pattern peak = 1.0 normalized)
    # Then apply efficiency factor (~0.7-0.9 for horn)
    efficiency = 0.85  # Typical aperture efficiency for horn

    # For dual-pol horn, both polarizations have similar patterns
    # but may have slight differences due to feed structure
    gain_e_pol = pattern * efficiency
    gain_h_pol = pattern * efficiency * 0.95  # Slightly lower for H-pol

    return gain_e_pol, gain_h_pol


def dual_pol_element_pattern(direction: mi.Vector3f,
                            orientation: mi.Vector3f,
                            pattern_type: str = "isotropic") -> Tuple[mi.Float, mi.Float]:
    """
    Dual-polarization element pattern.

    Args:
        direction: [N,3] Incident/outgoing direction
        orientation: [N,3] Element orientation
        pattern_type: "isotropic", "cosine", "patch", or "horn"

    Returns:
        Tuple[mi.Float, mi.Float]: (gain_H, gain_V) real gains for H and V pols
    """
    if pattern_type == "isotropic":
        gain = isotropic_pattern(direction, orientation)
        return gain, gain

    elif pattern_type == "cosine":
        gain = cosine_pattern(direction, orientation)
        return gain, gain

    elif pattern_type == "patch":
        gain = patch_antenna_pattern(direction, orientation)
        return gain, gain

    elif pattern_type == "horn":
        return horn_antenna_pattern(direction, orientation)

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def apply_element_pattern(E_real: mi.Vector2f,
                         E_imag: mi.Vector2f,
                         direction: mi.Vector3f,
                         orientation: mi.Vector3f,
                         pattern_func: callable) -> Tuple[mi.Vector2f, mi.Vector2f]:
    """
    Apply element pattern to E-field.

    For now assumes real scalar gain. Can be extended to complex 2x2 matrix.

    Args:
        E_real: [N,2] E-field real part
        E_imag: [N,2] E-field imaginary part
        direction: [N,3] Direction
        orientation: [N,3] Element orientation
        pattern_func: Pattern function to apply

    Returns:
        Tuple: (E_real_out, E_imag_out) with pattern applied
    """
    # Get gain (real scalar for now)
    gain = pattern_func(direction, orientation)

    # Apply to both polarizations
    E_real_out = mi.Vector2f(
        E_real[0] * gain,
        E_real[1] * gain
    )
    E_imag_out = mi.Vector2f(
        E_imag[0] * gain,
        E_imag[1] * gain
    )

    return E_real_out, E_imag_out


# ==============================================================================
# Antenna Pattern Loading from File
# ==============================================================================

import numpy as np
from typing import Optional


def evaluate_polarized_gain(pattern_loader: 'AntennaPatternLoader',
                           direction: mi.Vector3f,
                           orientation: Optional['mi.Vector3f'],
                           polarization: mi.UInt32) -> mi.Float:
    """
    DEPRECATED: This function incorrectly couples antenna pattern to polarization.
    Use evaluate_combined_gain() instead.

    Evaluate antenna gain for specific polarization (convenience function).

    This function automatically selects the correct pattern plane (E or H)
    based on the antenna's polarization configuration.

    WARNING: This implementation is INCORRECT. Antenna beam pattern and
    polarization are separate concepts. Both E and H plane patterns should
    be applied regardless of polarization state.

    Args:
        pattern_loader: Loaded antenna pattern
        direction: [N, 3] Direction vector(s)
        orientation: [N, 3] Antenna boresight orientation (unit vector)
        polarization: [N] Polarization code (0=H, 1=V)

    Returns:
        [N] Gain value(s) for specified polarization in linear scale

    Example:
        >>> pattern_loader = AntennaPatternLoader("pattern_76.npy")
        >>> # For V-polarized antenna (code=1)
        >>> gain = evaluate_polarized_gain(pattern_loader, direction, orientation, 1)
        >>> # This will return gain from E-plane pattern
    """
    return pattern_loader.evaluate_polarized(direction, orientation, polarization)


def evaluate_combined_gain(pattern_loader: 'AntennaPatternLoader',
                           direction: mi.Vector3f,
                           orientation: Optional['mi.Vector3f'] = None) -> mi.Float:
    """
    Evaluate antenna gain by combining E and H plane patterns (CORRECT approach).

    Antenna beam pattern is INDEPENDENT of polarization state. The E and H plane
    patterns describe the antenna's directional response in elevation and azimuth,
    not the polarization of the radiated field.

    For separable 3D radiation patterns, the total gain is the product of the
    E-plane and H-plane gains at the appropriate angles, scaled by the factor C:
        gain_3D = C x gain_E(phi_E) x gain_H(phi_H)

    where C = G_max / P_max anchors the separable product to the true peak gain.

    Args:
        pattern_loader: Loaded antenna pattern
        direction: [N, 3] Direction vector(s) in world frame
        orientation: [N, 3] Antenna boresight orientation (unit vector)
                    If None, assumes pattern aligned with +Y axis

    Returns:
        [N] Combined gain value(s) in linear scale

    Example:
        >>> pattern_loader = AntennaPatternLoader("pattern_76.npy")
        >>> gain = evaluate_combined_gain(pattern_loader, direction, orientation)
        >>> # Returns C x gain_E x gain_H (scaled 3D pattern)
    """
    # Get both plane gains
    gain_E, gain_H = pattern_loader.evaluate_linear(direction, orientation)

    # Combine using product (standard for separable patterns)
    # This represents the full 3D radiation pattern
    gain_unscaled = gain_E * gain_H

    # Apply scaling factor C to anchor to absolute peak gain
    # C = G_max / P_max, computed at pattern load time
    # Use Python float directly to avoid blocking gradients
    gain_combined = pattern_loader.C_scale * gain_unscaled

    return gain_combined


class AntennaPatternLoader:
    """
    Load and evaluate antenna radiation patterns from file.

    Supports loading patterns from .npy files with E/H plane cuts.
    Pattern data is interpolated for arbitrary angles and can be
    rotated using quaternion orientations.
    """

    def __init__(self, pattern_file: str):
        """
        Load antenna pattern from .npy file.

        Expected format: [N_angles, 2] array where:
            - [:, 0]: Gain in E-plane (dB)
            - [:, 1]: Gain in H-plane (dB)

        The angles are assumed to span 0-360 degrees uniformly.

        Args:
            pattern_file: Path to .npy file containing pattern data

        Example:
            >>> loader = AntennaPatternLoader("pattern_76.npy")
            >>> gain_E, gain_H = loader.evaluate_db(theta, phi)
        """
        # Load pattern data as numpy (for backup and info)
        self.pattern_data = np.load(pattern_file)

        if self.pattern_data.ndim != 2 or self.pattern_data.shape[1] != 2:
            raise ValueError(f"Expected pattern shape [N, 2], got {self.pattern_data.shape}")

        self.num_angles = self.pattern_data.shape[0]

        # Extract E and H plane patterns (in dB) - numpy arrays for info only
        E_plane_db_np = self.pattern_data[:, 0]
        H_plane_db_np = self.pattern_data[:, 1]

        # Convert to linear scale - numpy arrays for info only
        E_plane_linear_np = 10.0 ** (E_plane_db_np / 10.0)
        H_plane_linear_np = 10.0 ** (H_plane_db_np / 10.0)

        # CRITICAL: Store patterns as DrJit arrays for differentiability
        # These are the arrays that will be used in forward/backward pass
        self.E_plane_linear = mi.Float(E_plane_linear_np)
        self.H_plane_linear = mi.Float(H_plane_linear_np)

        # Angle grid (assume uniform spacing 0 to 360 degrees)
        self.angles_deg = np.linspace(0, 360, self.num_angles, dtype=np.float32)
        self.angles_rad = self.angles_deg * np.pi / 180.0

        # ====== COMPUTE SCALING FACTOR C ======
        # Following the algorithm: C = G_max / P_max
        # where G_max is estimated from the peak of the digitized cuts

        # Step 1: Estimate G_max from the maximum dB value in either cut
        # This assumes the true 2D peak occurs near a principal plane peak
        G_max_dB = max(E_plane_db_np.max(), H_plane_db_np.max())
        G_max_lin = 10.0 ** (G_max_dB / 10.0)

        # Step 2: Compute P_max = max(G_E x G_H) over all angles
        # This is the peak of the separable product
        P_max = (E_plane_linear_np * H_plane_linear_np).max()

        # Step 3: Compute scaling factor C (as Python float - it's a constant)
        self.C_scale = float(G_max_lin / P_max)

        print(f"[AntennaPatternLoader] Loaded pattern from {pattern_file}")
        print(f"  Angles: {self.num_angles} samples from 0 to 360 degrees")
        print(f"  E-plane range: {E_plane_db_np.min():.2f} to {E_plane_db_np.max():.2f} dBi")
        print(f"  H-plane range: {H_plane_db_np.min():.2f} to {H_plane_db_np.max():.2f} dBi")
        print(f"  Estimated G_max: {G_max_dB:.2f} dBi ({G_max_lin:.4f} linear)")
        print(f"  Separable P_max: {10*np.log10(P_max):.2f} dBi ({P_max:.4f} linear)")
        print(f"  Scaling factor C: {self.C_scale:.6f} ({10*np.log10(self.C_scale):.2f} dB)")
        print(f"  Pattern data stored as DrJit arrays (differentiable)")

    def enable_gradients(self):
        """
        Enable gradient tracking for antenna pattern parameters.

        Call this method when LEARN_PAT=True to allow optimization of
        antenna beam patterns during training.
        """
        dr.enable_grad(self.E_plane_linear)
        dr.enable_grad(self.H_plane_linear)
        print(f"  Gradients enabled for antenna pattern data ({self.num_angles} samples per plane)")

    def get_gradients(self):
        """Extract gradients for E-plane and H-plane pattern arrays.

        Returns numpy array of shape (2, num_angles) with [E_plane_grads, H_plane_grads],
        or None if no gradients are available.
        """
        e_grad = dr.grad(self.E_plane_linear)
        h_grad = dr.grad(self.H_plane_linear)
        if dr.width(e_grad) == 0 and dr.width(h_grad) == 0:
            return None
        e_np = np.array(e_grad, dtype=np.float32)
        h_np = np.array(h_grad, dtype=np.float32)
        return np.stack([e_np, h_np], axis=0)

    def apply_update(self, delta: np.ndarray):
        """Apply an additive update to the pattern parameters.

        Args:
            delta: numpy array of shape (2, num_angles).
                   delta[0] updates E-plane, delta[1] updates H-plane.
                   Values are in linear scale.
        """
        e_np = np.array(self.E_plane_linear, dtype=np.float32) + delta[0]
        h_np = np.array(self.H_plane_linear, dtype=np.float32) + delta[1]
        # Clamp to positive (pattern gains must be > 0)
        e_np = np.maximum(e_np, 1e-6)
        h_np = np.maximum(h_np, 1e-6)
        self.E_plane_linear = mi.Float(e_np)
        self.H_plane_linear = mi.Float(h_np)

    def compute_0db_limits(self) -> tuple:
        """
        Compute azimuth and elevation half-widths at 0dB cutoff.

        This method finds the angular extent where gain drops to 0dB (1.0 linear).
        Uses differentiable operations so limits adjust as pattern parameters change.

        Algorithm:
        1. Use soft threshold (tanh) to create smooth indicator: 0->1 transition at 0dB
        2. Find angles where indicator ~ 0.5 (within [0.2, 0.8]) - these are the boundary regions
        3. Compute angular distance of each boundary point from boresight (180deg)
        4. Average all boundary distances to get half-width (handles asymmetric beams)
        5. This gives mean of |crossing_1| + |crossing_2| / 2

        The pattern file has boresight at 180deg (pi radians), NOT at 0deg!
        Main lobe region is [90deg, 270deg] = [pi/2, 3pi/2].

        Returns:
            (azimuth_half_width_rad, elevation_half_width_rad): Tuple of mi.Float
                Half-widths in radians at 0dB cutoff, symmetric around boresight

        Note: Returns differentiable DrJit types (mi.Float) not Python floats.

        Example:
            For pattern_76.npy (IWR1443):
            - H-plane crossings: 112.87deg and 247.79deg -> Half-width: 67.5deg ~ 67.46deg
            - E-plane crossings: 147deg and 199.41deg -> Half-width: 24.0deg ~ 26.20deg
        """
        # Threshold in linear scale: 0 dB = 1.0 linear
        threshold_linear = mi.Float(1.0)

        # Create angle array in DrJit (for differentiability)
        angles_rad = mi.Float(self.angles_rad)

        # IMPORTANT: Pattern boresight (peak gain) is at 180deg (pi radians), NOT at 0deg!
        # Main lobe region is [90deg, 270deg] = [pi/2, 3pi/2] centered around 180deg
        # This excludes back lobes which could bias the result
        main_lobe_mask = (angles_rad >= (dr.pi / 2.0)) & (angles_rad <= (3.0 * dr.pi / 2.0))

        # ========== H-PLANE (AZIMUTH) LIMIT ==========
        # Use moderately sharp tanh to approximate hard threshold
        # Lower steepness = wider transition region to capture asymmetric beams
        tanh_steepness = 5.0  # Balance between sharpness and capturing both crossings

        above_threshold_H = 0.5 * (1.0 + dr.tanh(tanh_steepness * (self.H_plane_linear - threshold_linear)))

        # Find where indicator is near 0.5 (gain near 0dB)
        # Wide window to capture both crossings in asymmetric beams
        is_at_boundary_H = (above_threshold_H >= 0.2) & (above_threshold_H <= 0.8)

        # Apply main lobe mask
        valid_mask_H = main_lobe_mask & is_at_boundary_H

        # Compute angular distance from boresight (180deg = pi radians)
        # Boresight is where pattern peaks
        boresight_angle = dr.pi  # 180 degrees in radians
        angle_distance_H = dr.abs(angles_rad - boresight_angle)

        # Set valid boundary distances, invalid to 0 (won't contribute to average)
        boundary_distances_H = dr.select(valid_mask_H, angle_distance_H, mi.Float(0.0))

        # Average of boundary distances (handles asymmetric beams)
        # This gives (|crossing_1| + |crossing_2|) / 2
        sum_distances_H = dr.sum(boundary_distances_H)[0]
        count_boundary_H = dr.sum(dr.select(valid_mask_H, mi.Float(1.0), mi.Float(0.0)))[0]
        azimuth_half_width = sum_distances_H / (count_boundary_H + 1e-8)

        # If no valid angles found, max will be -1000, clamp will fix it to minimum
        # Clamp to reasonable range: [10deg, 85deg]
        azimuth_half_width = dr.clamp(
            azimuth_half_width,
            mi.Float(np.deg2rad(10.0)),
            mi.Float(np.deg2rad(85.0))
        )

        # ========== E-PLANE (ELEVATION) LIMIT ==========
        # Same process for E-plane
        above_threshold_E = 0.5 * (1.0 + dr.tanh(tanh_steepness * (self.E_plane_linear - threshold_linear)))

        # Find where indicator is at the boundary (same wide window)
        is_at_boundary_E = (above_threshold_E >= 0.2) & (above_threshold_E <= 0.8)

        # Apply main lobe mask
        valid_mask_E = main_lobe_mask & is_at_boundary_E

        # Compute angular distance from boresight (same as H-plane)
        angle_distance_E = dr.abs(angles_rad - boresight_angle)

        # Set valid boundary distances, invalid to 0 (won't contribute to average)
        boundary_distances_E = dr.select(valid_mask_E, angle_distance_E, mi.Float(0.0))

        # Average of boundary distances (handles asymmetric beams)
        sum_distances_E = dr.sum(boundary_distances_E)[0]
        count_boundary_E = dr.sum(dr.select(valid_mask_E, mi.Float(1.0), mi.Float(0.0)))[0]
        elevation_half_width = sum_distances_E / (count_boundary_E + 1e-8)

        # Clamp to reasonable range: [10deg, 85deg]
        elevation_half_width = dr.clamp(
            elevation_half_width,
            mi.Float(np.deg2rad(10.0)),
            mi.Float(np.deg2rad(85.0))
        )

        return (azimuth_half_width, elevation_half_width)

    def interpolate_pattern(self, angles_rad: np.ndarray,
                          plane: str = 'E') -> np.ndarray:
        """
        Interpolate pattern at given angles using linear interpolation.

        Args:
            angles_rad: [N] Angles in radians (0 to 2pi)
            plane: 'E' or 'H' plane

        Returns:
            [N] Interpolated gain in linear scale
        """
        # Select plane
        if plane == 'E':
            pattern = self.E_plane_linear
        elif plane == 'H':
            pattern = self.H_plane_linear
        else:
            raise ValueError(f"Unknown plane: {plane}. Use 'E' or 'H'")

        # Wrap angles to [0, 2pi]
        angles_wrapped = np.mod(angles_rad, 2 * np.pi)

        # Interpolate (linear)
        gain_interp = np.interp(angles_wrapped, self.angles_rad, pattern, period=2*np.pi)

        return gain_interp

    def evaluate_linear(self,
                       direction: mi.Vector3f,
                       orientation: Optional['mi.Vector3f'] = None) -> Tuple[mi.Float, mi.Float]:
        """
        Evaluate antenna pattern at given direction(s) in linear scale.

        Uses differentiable interpolation of the loaded pattern data.

        Coordinate Convention:
            Local antenna frame: +Y is boresight (forward), +X is right, +Z is up
            This matches the world frame convention for consistency.

        Args:
            direction: [N, 3] Direction vectors (world frame)
            orientation: [N, 3] or [3] Antenna boresight orientation (unit vector)
                        Represents +Y axis in antenna local frame
                        If None, assumes pattern aligned with +Y axis (forward)

        Returns:
            (gain_E, gain_H): [N] Gains in E and H planes (linear scale)
        """
        # If orientation provided, transform direction to antenna local frame
        if orientation is not None:
            # Build orthonormal frame with orientation as Y-axis (boresight)
            # This replaces the quaternion rotation approach
            y_local = dr.normalize(orientation)

            # Construct perpendicular X and Z axes
            # Use (1, 0, 0) as auxiliary vector, unless parallel to Y
            aux = mi.Vector3f(1.0, 0.0, 0.0)
            # If Y is nearly parallel to X axis, use Z axis instead
            parallel_threshold = 0.99
            is_parallel = dr.abs(dr.dot(y_local, aux)) > parallel_threshold
            aux = dr.select(is_parallel, mi.Vector3f(0.0, 0.0, 1.0), aux)

            # Z = normalize(Y x aux)  [up direction]
            z_local = dr.normalize(dr.cross(y_local, aux))
            # X = Z x Y  [right direction, completes right-handed frame]
            x_local = dr.cross(z_local, y_local)

            # Transform direction to local frame: dot with each basis vector
            direction_local = mi.Vector3f(
                dr.dot(direction, x_local),  # X component
                dr.dot(direction, y_local),  # Y component (along boresight)
                dr.dot(direction, z_local)   # Z component
            )
        else:
            direction_local = direction

        # Normalize direction
        direction_local = dr.normalize(direction_local)

        # Convert to spherical coordinates with +Y as boresight
        x = direction_local.x
        y = direction_local.y
        z = direction_local.z

        # Theta: polar angle from +Y axis (0 at boresight, pi at back)
        # NOT used directly - we use phi_E and phi_H instead

        # Phi_E: elevation angle in YZ plane (for E-plane)
        # Positive = upward (+Z), Negative = downward (-Z)
        # atan2(z, y) gives angle from +Y in YZ plane
        phi_E = dr.atan2(z, y)

        # Phi_H: azimuth angle in XY plane (for H-plane)
        # Positive = rightward (+X), Negative = leftward (-X)
        # atan2(x, y) gives angle from +Y in XY plane
        phi_H = dr.atan2(x, y)

        # Map angles to pattern index [0, 360) degrees
        # Pattern convention: [0=-180deg (back), 180=0deg (boresight), 360=+180deg (back)]

        # E-plane: use elevation angle phi_E
        # phi_E in [-pi, pi] represents [-180deg, +180deg] in elevation
        # Add 180deg to map to [0deg, 360deg] for array indexing
        angle_E_deg = phi_E * 180.0 / dr.pi + 180.0

        # H-plane: use azimuth angle phi_H
        # phi_H in [-pi, pi] represents [-180deg, +180deg] in azimuth
        # Add 180deg to map to [0deg, 360deg] for array indexing
        angle_H_deg = phi_H * 180.0 / dr.pi + 180.0

        # DEBUG: Print first sample to verify angle mapping
        if False:  # Set to True to enable debug
            dr.eval(phi_E, phi_H, angle_E_deg, angle_H_deg)
            phi_E_0 = float(phi_E[0]) if hasattr(phi_E, '__getitem__') else float(phi_E)
            phi_H_0 = float(phi_H[0]) if hasattr(phi_H, '__getitem__') else float(phi_H)
            angle_E_0 = float(angle_E_deg[0]) if hasattr(angle_E_deg, '__getitem__') else float(angle_E_deg)
            angle_H_0 = float(angle_H_deg[0]) if hasattr(angle_H_deg, '__getitem__') else float(angle_H_deg)
            print(f"[DEBUG Pattern] dir_local=[{float(x):.3f}, {float(y):.3f}, {float(z):.3f}]")
            print(f"[DEBUG Pattern] phi_E={phi_E_0:.3f} rad ({phi_E_0*180/3.14159:.1f}deg) -> angle_E={angle_E_0:.1f}deg")
            print(f"[DEBUG Pattern] phi_H={phi_H_0:.3f} rad ({phi_H_0*180/3.14159:.1f}deg) -> angle_H={angle_H_0:.1f}deg")

        # Use differentiable interpolation
        gain_E = self._differentiable_interp(angle_E_deg, self.E_plane_linear)
        gain_H = self._differentiable_interp(angle_H_deg, self.H_plane_linear)

        return gain_E, gain_H

    def _differentiable_interp(self, angle_deg: mi.Float, pattern_data: mi.Float) -> mi.Float:
        """
        Differentiable interpolation of pattern data.

        Uses linear interpolation with Dr.Jit gather operations.

        Args:
            angle_deg: [N] Angles in degrees [0, 360)
            pattern_data: [361] Pattern values (linear scale) - DrJit array

        Returns:
            [N] Interpolated gain values
        """
        # pattern_data is already a DrJit array, use directly
        pattern_dr = pattern_data

        # Map angle to continuous index in [0, num_angles)
        # Pattern has 361 samples for 0-360 degrees (inclusive endpoints)
        idx_float = angle_deg * (self.num_angles - 1) / 360.0

        # Clamp to valid range [0, num_angles-1]
        # Pattern is periodic, so we can safely clamp
        max_idx = float(self.num_angles - 1)
        idx_float = dr.clamp(idx_float, 0.0, max_idx)

        # Get integer indices for linear interpolation
        idx_low = dr.floor(idx_float)
        idx_high = idx_low + 1.0

        # Compute interpolation weight
        weight = idx_float - idx_low

        # Wrap high index (circular boundary)
        idx_high = dr.select(idx_high >= float(self.num_angles),
                            idx_high - float(self.num_angles),
                            idx_high)

        # Gather values using integer indices
        idx_low_int = mi.UInt32(idx_low)
        idx_high_int = mi.UInt32(idx_high)

        val_low = dr.gather(mi.Float, pattern_dr, idx_low_int)
        val_high = dr.gather(mi.Float, pattern_dr, idx_high_int)

        # Linear interpolation
        gain = val_low * (1.0 - weight) + val_high * weight

        return gain

    def evaluate_db(self,
                   direction: mi.Vector3f,
                   orientation: Optional['mi.Vector3f'] = None) -> Tuple[mi.Float, mi.Float]:
        """
        Evaluate antenna pattern in dB scale.

        Args:
            direction: [N, 3] Direction vectors
            orientation: [N, 3] Antenna boresight orientation (unit vector)

        Returns:
            (gain_E_db, gain_H_db): [N] Gains in dB
        """
        gain_E_linear, gain_H_linear = self.evaluate_linear(direction, orientation)

        # Convert to dB
        gain_E_db = 10.0 * dr.log(gain_E_linear) / dr.log(10.0)
        gain_H_db = 10.0 * dr.log(gain_H_linear) / dr.log(10.0)

        return gain_E_db, gain_H_db

    def evaluate_polarized(self,
                          direction: mi.Vector3f,
                          orientation: Optional['mi.Vector3f'],
                          polarization: mi.UInt32) -> mi.Float:
        """
        Evaluate antenna pattern for specific polarization.

        Selects the correct pattern plane (E or H) based on polarization config.

        Args:
            direction: [N, 3] Direction vectors (world frame)
            orientation: [N, 3] Antenna boresight orientation (unit vector)
            polarization: [N] Polarization for each antenna (0=H, 1=V)

        Returns:
            [N] Gain in linear scale for the specified polarization
        """
        # Get both plane gains
        gain_E, gain_H = self.evaluate_linear(direction, orientation)

        # Select based on polarization
        # V-polarization (1) uses E-plane pattern
        # H-polarization (0) uses H-plane pattern
        is_v_pol = (polarization == 1)
        gain = dr.select(is_v_pol, gain_E, gain_H)

        return gain


class DifferentiableAntennaPattern:
    """
    Wrapper for antenna pattern with learnable parameters.

    Allows for fine-tuning of antenna pattern via gradient descent.
    """

    def __init__(self,
                 base_pattern: AntennaPatternLoader,
                 enable_scale_learning: bool = False,
                 enable_offset_learning: bool = False):
        """
        Create differentiable antenna pattern wrapper.

        Args:
            base_pattern: Base antenna pattern loader
            enable_scale_learning: If True, add learnable scale parameter
            enable_offset_learning: If True, add learnable offset parameter
        """
        self.base_pattern = base_pattern

        # Learnable parameters (initialized to identity)
        self.scale = mi.Float(1.0) if enable_scale_learning else 1.0
        self.offset_db = mi.Float(0.0) if enable_offset_learning else 0.0

        if enable_scale_learning:
            dr.enable_grad(self.scale)
        if enable_offset_learning:
            dr.enable_grad(self.offset_db)

    def evaluate(self,
                direction: mi.Vector3f,
                quaternion: Optional['mi.Vector4f'] = None) -> Tuple[mi.Float, mi.Float]:
        """
        Evaluate pattern with learnable parameters applied.

        Args:
            direction: [N, 3] Direction vectors
            quaternion: [N, 4] Antenna orientation quaternions

        Returns:
            (gain_E, gain_H): [N] Gains in linear scale with adjustments
        """
        # Get base pattern
        gain_E, gain_H = self.base_pattern.evaluate_linear(direction, quaternion)

        # Apply learnable scale and offset
        # offset is in dB, so convert: linear_adjusted = linear * 10^(offset_db/10)
        if isinstance(self.offset_db, mi.Float):
            offset_linear = dr.power(10.0, self.offset_db / 10.0)
            gain_E = gain_E * offset_linear
            gain_H = gain_H * offset_linear

        if isinstance(self.scale, mi.Float):
            gain_E = gain_E * self.scale
            gain_H = gain_H * self.scale

        return gain_E, gain_H


# ==============================================================================
# RRTS-Compatible Antenna Pattern Functions
# ==============================================================================

def load_pattern_rrts_format(npy_path: str) -> np.ndarray:
    """
    Load antenna pattern from NPY file in RRTS format.

    IMPORTANT: RRTS uses dB values DIRECTLY as multipliers without converting
    to linear scale. This is mathematically unusual but matches RRTS behavior.

    RRTS expects pattern tensor [nPhi, nTheta, 5] where:
    - nPhi = number of elevation samples (361)
    - nTheta = 2 (E-plane and H-plane patterns)
    - Channel 0 = gain in dB (NOT linear!)

    The NPY file has shape [361, 2] with [E_dB, H_dB] per row.

    Args:
        npy_path: Path to .npy file containing [361, 2] pattern data

    Returns:
        pattern: [361, 2, 5] array matching RRTS format (dB values)
    """
    data = np.load(npy_path)
    if data.shape != (361, 2):
        raise ValueError(f"Expected pattern shape [361, 2], got {data.shape}")

    # RRTS uses dB values directly - DO NOT convert to linear!
    E_dB = data[:, 0]  # [361]
    H_dB = data[:, 1]  # [361]

    # Build RRTS format: [361, 2, 5]
    # Only channel 0 (abs gain) is used by bilinear_gain_single
    pattern = np.zeros((361, 2, 5), dtype=np.float32)
    pattern[:, 0, 0] = E_dB  # E-plane gain in dB (RRTS uses raw dB values!)
    pattern[:, 1, 0] = H_dB  # H-plane gain in dB

    print(f"[load_pattern_rrts_format] Loaded {npy_path}")
    print(f"  Shape: {pattern.shape}")
    print(f"  E-plane dB range: [{E_dB.min():.2f}, {E_dB.max():.2f}] dB")
    print(f"  H-plane dB range: [{H_dB.min():.2f}, {H_dB.max():.2f}] dB")
    print(f"  NOTE: RRTS uses dB values directly as multipliers (not linear)")

    return pattern


def evaluate_gain_rrts_style_numpy(
    direction: np.ndarray,
    pattern: np.ndarray
) -> np.ndarray:
    """
    Evaluate antenna pattern using RRTS-style bilinear interpolation (NumPy version).

    Matches RRTS antenna_lookup.slang:bilinear_gain_single() exactly.

    Args:
        direction: [N, 3] Direction vectors (from antenna to target)
        pattern: [nPhi, nTheta, 5] RRTS-format pattern (dB values)

    Returns:
        [N] Gain values (dB used directly as multipliers)
    """
    if direction.ndim == 1:
        direction = direction.reshape(1, 3)

    n = len(direction)
    gains = np.zeros(n, dtype=np.float64)

    nPhi = pattern.shape[0]    # 361
    nTheta = pattern.shape[1]  # 2

    eps = 1e-6

    for i in range(n):
        w = direction[i]
        r = np.linalg.norm(w) + eps

        # RRTS: to_spherical(w)
        theta = np.arctan2(w[1], w[0])  # azimuth in XY plane
        if theta < 0:
            theta += 2 * np.pi
        phi = np.arccos(np.clip(w[2] / r, -1.0, 1.0))  # elevation from Z axis

        # RRTS: pattern indices
        ip_x = float(nPhi) * phi / np.pi           # elevation index
        ip_y = float(nTheta) * theta / (2 * np.pi)  # azimuth index (0-2)

        # Fractional part for interpolation (RRTS: frac(ip - 0.5))
        f_x = (ip_x - 0.5) - np.floor(ip_x - 0.5)
        f_y = (ip_y - 0.5) - np.floor(ip_y - 0.5)

        # Integer indices (clamped)
        idx_i = max(0, min(nPhi - 2, int(ip_x - 0.5)))
        idx_j = max(0, min(nTheta - 2, int(ip_y - 0.5)))

        # Bilinear interpolation (channel 0 = gain in dB)
        p00 = pattern[idx_i, idx_j, 0]
        p10 = pattern[idx_i + 1, idx_j, 0]
        p01 = pattern[idx_i, idx_j + 1, 0]
        p11 = pattern[idx_i + 1, idx_j + 1, 0]

        # RRTS uses lerp: lerp(a, b, t) = a*(1-t) + b*t
        y0 = p00 * (1 - f_x) + p10 * f_x
        y1 = p01 * (1 - f_x) + p11 * f_x
        gain = y0 * (1 - f_y) + y1 * f_y

        gains[i] = gain

    return gains


def evaluate_gain_rrts_style_vectorized(
    direction: np.ndarray,
    pattern: np.ndarray,
    orientation: np.ndarray = None,
    combine_mode: str = 'product'
) -> np.ndarray:
    """
    Vectorized antenna pattern evaluation with orientation-aware coordinate system.

    This function evaluates the antenna pattern for given directions relative to
    the antenna boresight (orientation). It correctly computes elevation and azimuth
    angles from the boresight direction, not from the Z-axis.

    Pattern format: [361, 2, 5] where:
    - dim=0 (361 samples): angles from -180° to +180° centered on boresight (0° = boresight)
    - dim=1 (2 planes): E-plane (elevation) and H-plane (azimuth)
    - dim=2 (5 channels): channel 0 = gain in dB

    Args:
        direction: [N, 3] Direction vectors (from antenna to target)
        pattern: [nPhi, nTheta, 5] RRTS-format pattern (dB values)
        orientation: [3] or [N, 3] Antenna boresight direction (unit vector).
                    If None, defaults to +Y = [0, 1, 0].
        combine_mode: How to combine E and H plane gains:
                     'product' (default): C * gain_E * gain_H (separable pattern)
                     'e_only': E-plane gain only
                     'h_only': H-plane gain only
                     'legacy': Old behavior (bilinear interpolation, no orientation)

    Returns:
        [N] Gain values in dB
    """
    if direction.ndim == 1:
        direction = direction.reshape(1, 3)

    n = len(direction)
    nPhi = pattern.shape[0]    # 361

    # ========== LEGACY MODE (OLD BEHAVIOR) ==========
    if combine_mode == 'legacy' or orientation is None:
        # Original RRTS-style bilinear interpolation (kept for backward compatibility)
        # WARNING: This does NOT handle orientation correctly!
        nTheta = pattern.shape[1]  # 2
        eps = 1e-6

        # Vectorized spherical coordinates (from Z-axis, NOT from boresight)
        r = np.linalg.norm(direction, axis=1) + eps

        # theta = azimuth in XY plane
        theta = np.arctan2(direction[:, 1], direction[:, 0])
        theta = np.where(theta < 0, theta + 2.0 * np.pi, theta)

        # phi = elevation from Z axis
        phi = np.arccos(np.clip(direction[:, 2] / r, -1.0, 1.0))

        # Pattern indices
        ip_x = float(nPhi) * phi / np.pi
        ip_y = float(nTheta) * theta / (2.0 * np.pi)

        # Fractional part for interpolation
        f_x = (ip_x - 0.5) - np.floor(ip_x - 0.5)
        f_y = (ip_y - 0.5) - np.floor(ip_y - 0.5)

        # Integer indices (clamped)
        idx_i = np.clip((ip_x - 0.5).astype(np.int32), 0, nPhi - 2)
        idx_j = np.clip((ip_y - 0.5).astype(np.int32), 0, nTheta - 2)

        # Bilinear interpolation
        p00 = pattern[idx_i, idx_j, 0]
        p10 = pattern[idx_i + 1, idx_j, 0]
        p01 = pattern[idx_i, idx_j + 1, 0]
        p11 = pattern[idx_i + 1, idx_j + 1, 0]

        y0 = p00 * (1.0 - f_x) + p10 * f_x
        y1 = p01 * (1.0 - f_x) + p11 * f_x
        gains = y0 * (1.0 - f_y) + y1 * f_y

        return gains

    # ========== ORIENTATION-AWARE MODE (CORRECT) ==========
    # Handle orientation shape
    if orientation.ndim == 1:
        orientation = np.tile(orientation.reshape(1, 3), (n, 1))
    elif len(orientation) == 1 and n > 1:
        orientation = np.tile(orientation, (n, 1))

    # Normalize directions and orientations
    direction = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-10)
    orientation = orientation / (np.linalg.norm(orientation, axis=1, keepdims=True) + 1e-10)

    # Build local coordinate system with orientation as Y-axis (boresight)
    y_local = orientation  # [N, 3]

    # Choose auxiliary vector for Gram-Schmidt
    # Use (1, 0, 0) unless nearly parallel to Y
    parallel_mask = np.abs(y_local[:, 0]) > 0.9
    aux = np.where(parallel_mask[:, np.newaxis],
                   np.tile([0.0, 0.0, 1.0], (n, 1)),
                   np.tile([1.0, 0.0, 0.0], (n, 1)))

    # Z = normalize(Y × aux) [up direction]
    z_local = np.cross(y_local, aux)
    z_local = z_local / (np.linalg.norm(z_local, axis=1, keepdims=True) + 1e-10)

    # X = Z × Y [right direction]
    x_local = np.cross(z_local, y_local)

    # Transform direction to local frame
    dir_x = np.sum(direction * x_local, axis=1)  # Component along X (right)
    dir_y = np.sum(direction * y_local, axis=1)  # Component along Y (boresight)
    dir_z = np.sum(direction * z_local, axis=1)  # Component along Z (up)

    # Compute angles relative to boresight (+Y in local frame)
    # phi_E: elevation angle in YZ plane (for E-plane)
    # atan2(z, y) gives angle from +Y: 0 = boresight, ±π = back
    phi_E = np.arctan2(dir_z, dir_y)  # [-π, π]

    # phi_H: azimuth angle in XY plane (for H-plane)
    # atan2(x, y) gives angle from +Y: 0 = boresight, ±π = back
    phi_H = np.arctan2(dir_x, dir_y)  # [-π, π]

    # Map angles to pattern index [0, 361)
    # Pattern convention: index 0 = -180°, index 180 = 0° (boresight), index 360 = +180°
    # phi in [-π, π] maps to [0, 360] by adding 180°

    angle_E_deg = phi_E * 180.0 / np.pi + 180.0  # [0, 360]
    angle_H_deg = phi_H * 180.0 / np.pi + 180.0  # [0, 360]

    # Interpolate E-plane pattern (pattern[:, 0, 0])
    idx_E_float = angle_E_deg * (nPhi - 1) / 360.0
    idx_E_float = np.clip(idx_E_float, 0.0, nPhi - 1)
    idx_E_low = np.floor(idx_E_float).astype(np.int32)
    idx_E_high = np.minimum(idx_E_low + 1, nPhi - 1)
    weight_E = idx_E_float - idx_E_low
    gain_E_dB = pattern[idx_E_low, 0, 0] * (1 - weight_E) + pattern[idx_E_high, 0, 0] * weight_E

    # Interpolate H-plane pattern (pattern[:, 1, 0])
    idx_H_float = angle_H_deg * (nPhi - 1) / 360.0
    idx_H_float = np.clip(idx_H_float, 0.0, nPhi - 1)
    idx_H_low = np.floor(idx_H_float).astype(np.int32)
    idx_H_high = np.minimum(idx_H_low + 1, nPhi - 1)
    weight_H = idx_H_float - idx_H_low
    gain_H_dB = pattern[idx_H_low, 1, 0] * (1 - weight_H) + pattern[idx_H_high, 1, 0] * weight_H

    # Combine based on mode
    if combine_mode == 'e_only':
        return gain_E_dB
    elif combine_mode == 'h_only':
        return gain_H_dB
    elif combine_mode == 'product':
        # Convert to linear, multiply, convert back to dB
        # For separable pattern: gain_total = C * gain_E * gain_H
        # In dB: gain_total_dB = C_dB + gain_E_dB + gain_H_dB
        # But we need to compute C properly based on the pattern

        # Estimate C: C = G_max / P_max where P_max = max(E * H)
        E_linear = 10.0 ** (pattern[:, 0, 0] / 10.0)
        H_linear = 10.0 ** (pattern[:, 1, 0] / 10.0)
        G_max_dB = max(pattern[:, 0, 0].max(), pattern[:, 1, 0].max())
        G_max_linear = 10.0 ** (G_max_dB / 10.0)
        P_max = (E_linear * H_linear).max()
        C = G_max_linear / P_max
        C_dB = 10.0 * np.log10(C + 1e-10)

        # Total gain in dB
        gain_total_dB = C_dB + gain_E_dB + gain_H_dB
        return gain_total_dB
    else:
        raise ValueError(f"Unknown combine_mode: {combine_mode}")


def evaluate_gain_rrts_style_drjit(
    direction: 'mi.Vector3f',
    pattern: np.ndarray
) -> 'mi.Float':
    """
    Evaluate antenna pattern using RRTS-style bilinear interpolation (DrJit version).

    Matches RRTS antenna_lookup.slang:bilinear_gain_single() exactly.
    Uses DrJit operations for differentiability.

    Args:
        direction: [N, 3] Direction vectors (from antenna to target) as DrJit Vector3f
        pattern: [nPhi, nTheta, 5] RRTS-format pattern (dB values) as numpy array

    Returns:
        mi.Float: [N] Gain values (dB used directly as multipliers)
    """
    nPhi = pattern.shape[0]    # 361
    nTheta = pattern.shape[1]  # 2

    eps = 1e-6

    # Extract components
    x = direction.x
    y = direction.y
    z = direction.z

    # Compute r = ||w|| + eps
    r = dr.sqrt(x*x + y*y + z*z) + eps

    # RRTS: to_spherical(w)
    # theta = azimuth in XY plane
    theta = dr.atan2(y, x)
    # Wrap negative theta to [0, 2pi]
    theta = dr.select(theta < 0, theta + 2.0 * dr.pi, theta)

    # phi = elevation from Z axis
    phi = dr.acos(dr.clamp(z / r, -1.0, 1.0))

    # Pattern indices
    ip_x = float(nPhi) * phi / dr.pi           # elevation index
    ip_y = float(nTheta) * theta / (2.0 * dr.pi)  # azimuth index

    # Fractional part for interpolation
    f_x = (ip_x - 0.5) - dr.floor(ip_x - 0.5)
    f_y = (ip_y - 0.5) - dr.floor(ip_y - 0.5)

    # Integer indices (clamped)
    idx_i = dr.clamp(mi.UInt32(dr.floor(ip_x - 0.5)), 0, nPhi - 2)
    idx_j = dr.clamp(mi.UInt32(dr.floor(ip_y - 0.5)), 0, nTheta - 2)

    # Flatten pattern for gather (channel 0 only)
    # pattern shape: [nPhi, nTheta, 5], we need [:, :, 0]
    pattern_flat = mi.Float(pattern[:, :, 0].flatten())  # [nPhi * nTheta]

    # Compute linear indices for 2D array access
    # Index = i * nTheta + j
    idx_00 = idx_i * nTheta + idx_j
    idx_10 = (idx_i + 1) * nTheta + idx_j
    idx_01 = idx_i * nTheta + (idx_j + 1)
    idx_11 = (idx_i + 1) * nTheta + (idx_j + 1)

    # Clamp to valid range
    max_idx = nPhi * nTheta - 1
    idx_00 = dr.clamp(idx_00, mi.UInt32(0), mi.UInt32(max_idx))
    idx_10 = dr.clamp(idx_10, mi.UInt32(0), mi.UInt32(max_idx))
    idx_01 = dr.clamp(idx_01, mi.UInt32(0), mi.UInt32(max_idx))
    idx_11 = dr.clamp(idx_11, mi.UInt32(0), mi.UInt32(max_idx))

    # Gather pattern values
    p00 = dr.gather(mi.Float, pattern_flat, idx_00)
    p10 = dr.gather(mi.Float, pattern_flat, idx_10)
    p01 = dr.gather(mi.Float, pattern_flat, idx_01)
    p11 = dr.gather(mi.Float, pattern_flat, idx_11)

    # Bilinear interpolation: lerp(a, b, t) = a*(1-t) + b*t
    y0 = p00 * (1.0 - f_x) + p10 * f_x
    y1 = p01 * (1.0 - f_x) + p11 * f_x
    gain = y0 * (1.0 - f_y) + y1 * f_y

    return gain


def _pattern_scaling_constant_dB(pattern: np.ndarray) -> float:
    """Precompute the separable pattern scaling constant C (in dB).

    For a separable pattern G(θ,φ) = C · G_E(θ) · G_H(φ), the scaling
    constant C = G_max / max(G_E · G_H) ensures the peak of the combined
    pattern matches the measured G_max.

    Args:
        pattern: [nPhi, 2, 5] RRTS-format pattern (dB values).

    Returns:
        C_dB: Scaling constant in dB.
    """
    E_linear = np.power(10.0, pattern[:, 0, 0] / 10.0)
    H_linear = np.power(10.0, pattern[:, 1, 0] / 10.0)
    G_max_dB = max(pattern[:, 0, 0].max(), pattern[:, 1, 0].max())
    G_max_linear = np.power(10.0, G_max_dB / 10.0)
    P_max = (E_linear * H_linear).max()
    C = G_max_linear / P_max
    C_dB = 10.0 * np.log10(C + 1e-10)
    return float(C_dB)


def evaluate_gain_rrts_style_drjit_product(
    direction: 'mi.Vector3f',
    pattern: np.ndarray,
    orientation: 'mi.Vector3f',
) -> 'mi.Float':
    """
    Orientation-aware separable E×H antenna pattern evaluation (DrJit GPU version).

    Matches the 'product' mode of evaluate_gain_rrts_style_vectorized() exactly,
    but runs entirely on GPU with no CPU↔GPU transfers.

    The antenna pattern is evaluated in a local coordinate frame where the
    boresight (orientation) is the Y-axis.  E-plane gain is looked up from
    the elevation angle (YZ plane) and H-plane gain from the azimuth angle
    (XY plane).  The combined gain is C_dB + gain_E_dB + gain_H_dB.

    Args:
        direction: [N] Direction vectors (from antenna to target) as DrJit Vector3f.
        pattern:   [nPhi, 2, 5] RRTS-format pattern (dB, numpy — small, uploaded once).
        orientation: [N] Boresight direction per element as DrJit Vector3f.

    Returns:
        mi.Float: [N] Total gain in dB.
    """
    import math
    nPhi = pattern.shape[0]  # 361

    # Upload E-plane and H-plane pattern data to GPU (361 floats each — tiny)
    pattern_E = mi.Float(pattern[:, 0, 0].astype(np.float32))  # [nPhi]
    pattern_H = mi.Float(pattern[:, 1, 0].astype(np.float32))  # [nPhi]

    # Precompute scaling constant (once, on CPU — 361 elements)
    C_dB = _pattern_scaling_constant_dB(pattern)

    # ------------------------------------------------------------------
    # Build local coordinate frame:  Y = boresight, Z = up, X = right
    # ------------------------------------------------------------------
    eps_f = mi.Float(1e-10)

    # Normalize inputs
    dir_norm = dr.normalize(direction)
    y_local = dr.normalize(orientation)

    # Gram-Schmidt: choose auxiliary vector not parallel to Y
    parallel = dr.abs(y_local.x) > mi.Float(0.9)
    aux = mi.Vector3f(
        dr.select(parallel, mi.Float(0.0), mi.Float(1.0)),
        mi.Float(0.0),
        dr.select(parallel, mi.Float(1.0), mi.Float(0.0)),
    )
    z_local = dr.normalize(dr.cross(y_local, aux))
    x_local = dr.cross(z_local, y_local)

    # Project direction into local frame
    dir_x = dr.dot(dir_norm, x_local)
    dir_y = dr.dot(dir_norm, y_local)
    dir_z = dr.dot(dir_norm, z_local)

    # ------------------------------------------------------------------
    # Compute E-plane and H-plane angles relative to boresight
    # ------------------------------------------------------------------
    phi_E = dr.atan2(dir_z, dir_y)  # elevation in YZ plane  [-π, π]
    phi_H = dr.atan2(dir_x, dir_y)  # azimuth  in XY plane  [-π, π]

    # Map to degrees [0, 360]:  pattern index 0 = −180°, 180 = 0° (boresight)
    RAD2DEG = mi.Float(180.0 / math.pi)
    angle_E_deg = phi_E * RAD2DEG + mi.Float(180.0)
    angle_H_deg = phi_H * RAD2DEG + mi.Float(180.0)

    # ------------------------------------------------------------------
    # E-plane interpolation
    # ------------------------------------------------------------------
    scale = mi.Float((nPhi - 1) / 360.0)
    idx_E_f = dr.clamp(angle_E_deg * scale, mi.Float(0.0), mi.Float(nPhi - 1))
    idx_E_lo = mi.UInt32(dr.floor(idx_E_f))
    idx_E_hi = dr.minimum(idx_E_lo + mi.UInt32(1), mi.UInt32(nPhi - 1))
    w_E = idx_E_f - mi.Float(idx_E_lo)
    gain_E = dr.gather(mi.Float, pattern_E, idx_E_lo) * (mi.Float(1.0) - w_E) \
           + dr.gather(mi.Float, pattern_E, idx_E_hi) * w_E

    # ------------------------------------------------------------------
    # H-plane interpolation
    # ------------------------------------------------------------------
    idx_H_f = dr.clamp(angle_H_deg * scale, mi.Float(0.0), mi.Float(nPhi - 1))
    idx_H_lo = mi.UInt32(dr.floor(idx_H_f))
    idx_H_hi = dr.minimum(idx_H_lo + mi.UInt32(1), mi.UInt32(nPhi - 1))
    w_H = idx_H_f - mi.Float(idx_H_lo)
    gain_H = dr.gather(mi.Float, pattern_H, idx_H_lo) * (mi.Float(1.0) - w_H) \
           + dr.gather(mi.Float, pattern_H, idx_H_hi) * w_H

    # Combined gain: C_dB + gain_E_dB + gain_H_dB
    return mi.Float(C_dB) + gain_E + gain_H


class RRTSPatternLoader:
    """
    RRTS-compatible antenna pattern loader.

    Loads patterns in RRTS format and provides evaluation matching
    RRTS antenna_lookup.slang exactly.
    """

    def __init__(self, pattern_file: str):
        """
        Load antenna pattern from .npy file in RRTS format.

        Args:
            pattern_file: Path to .npy file containing [361, 2] pattern data
        """
        self.pattern = load_pattern_rrts_format(pattern_file)
        self.pattern_file = pattern_file

    def evaluate(self, direction: 'mi.Vector3f') -> 'mi.Float':
        """
        Evaluate antenna pattern at given direction(s).

        Args:
            direction: [N, 3] Direction vectors (from antenna to target)

        Returns:
            [N] Gain values (dB used directly as multipliers)
        """
        return evaluate_gain_rrts_style_drjit(direction, self.pattern)

    def evaluate_numpy(self, direction: np.ndarray) -> np.ndarray:
        """
        Evaluate antenna pattern at given direction(s) using NumPy.

        Args:
            direction: [N, 3] Direction vectors (from antenna to target)

        Returns:
            [N] Gain values (dB used directly as multipliers)
        """
        return evaluate_gain_rrts_style_numpy(direction, self.pattern)
