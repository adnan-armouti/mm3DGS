"""
Antenna pattern importance sampling for RX-centric Monte Carlo.

Provides sampling weighted by cos(theta) x G_r(omega) for variance reduction.
This reduces sampling variance by concentrating samples where the antenna
has high gain, rather than uniform cosine-weighted hemisphere sampling.

The sampler uses the SAME gain evaluation function as the integrator
to ensure consistency (see PLAN_improve_ra_correlation.md Phase 1).
"""

import numpy as np
from typing import Tuple, Optional


class PatternImportanceSampler:
    """
    Importance sampler for RX directions weighted by antenna pattern.

    Builds 2D CDF over (theta, phi) with weights:
        w(theta, phi) = cos(theta) x G_r(theta, phi) x sin(theta) x d_theta x d_phi
                        |             |                 |
                        cosine        pattern           solid-angle measure

    The sampling distribution is proportional to:
        rho_rx(omega) ~ cos(theta) x G_r(omega)

    This matches the RX-centric MC estimator requirements while reducing
    variance by concentrating samples in high-gain regions.
    """

    def __init__(
        self,
        pattern_data: np.ndarray,  # [361, 2] E/H plane in dB OR [361, 2, 5] RRTS format
        n_theta: int = 90,         # bins from 0 to pi/2
        n_phi: int = 360,          # bins from 0 to 2pi
        combine_mode: str = 'product',  # 'product', 'e_only', 'h_only'
    ):
        """
        Build CDF from antenna pattern.

        CRITICAL: Pattern must be converted from dB to LINEAR before building CDF:
            G_linear = 10^(G_dB / 10)

        Args:
            pattern_data: Antenna pattern in RRTS format [361, 2, 5] or [361, 2]
                         Values are in dB (will be converted to linear)
            n_theta: Number of theta bins (elevation) from 0 to pi/2
            n_phi: Number of phi bins (azimuth) from 0 to 2pi
            combine_mode: How to combine E/H planes:
                         'product': C x G_E x G_H (separable 3D pattern)
                         'e_only': E-plane only
                         'h_only': H-plane only
        """
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.combine_mode = combine_mode

        # Extract pattern data (handle both [361, 2] and [361, 2, 5] formats)
        if pattern_data.ndim == 3:
            # RRTS format [361, 2, 5] - channel 0 is gain in dB
            self.E_plane_dB = pattern_data[:, 0, 0]
            self.H_plane_dB = pattern_data[:, 1, 0]
        else:
            # Simple format [361, 2]
            self.E_plane_dB = pattern_data[:, 0]
            self.H_plane_dB = pattern_data[:, 1]

        self.nPhi_pattern = len(self.E_plane_dB)  # 361

        # Bin edges
        self.theta_edges = np.linspace(0, np.pi / 2, n_theta + 1)
        self.phi_edges = np.linspace(0, 2 * np.pi, n_phi + 1)
        self.theta_centers = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
        self.phi_centers = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])

        # Bin sizes
        self.d_theta = self.theta_edges[1] - self.theta_edges[0]
        self.d_phi = self.phi_edges[1] - self.phi_edges[0]

        # Compute bin solid angles: d_omega_ij ~ sin(theta_i) x d_theta x d_phi
        # Shape: [n_theta, n_phi] - same for all phi at given theta
        self.solid_angles = np.outer(np.sin(self.theta_centers), np.ones(n_phi)) * self.d_theta * self.d_phi

        # Precompute scaling factor C for separable pattern
        self._compute_C_scale()

        # Build weight grid and CDFs
        self._build_cdf()

    def _compute_C_scale(self):
        """Compute scaling factor C = G_max / P_max for separable pattern."""
        E_linear = 10.0 ** (self.E_plane_dB / 10.0)
        H_linear = 10.0 ** (self.H_plane_dB / 10.0)

        G_max_dB = max(self.E_plane_dB.max(), self.H_plane_dB.max())
        G_max_linear = 10.0 ** (G_max_dB / 10.0)
        P_max = (E_linear * H_linear).max()

        self.C_scale = G_max_linear / (P_max + 1e-10)

    def _evaluate_pattern_linear(self, theta: float, phi: float) -> float:
        """
        Evaluate pattern gain in LINEAR scale at (theta, phi) in local frame.

        Uses the SAME interpolation as evaluate_gain_rrts_style_vectorized()
        in the integrator for consistency.

        Args:
            theta: Elevation angle from boresight (0 = boresight, pi/2 = sideways)
            phi: Azimuth angle around boresight

        Returns:
            Linear gain value
        """
        # In local frame: boresight = +Y, so we need to map theta,phi to pattern indices
        # theta is angle from +Y axis (boresight)
        # phi is rotation around +Y axis

        # For E-plane (elevation): angle in YZ plane
        # phi_E = atan2(z, y) where z = sin(theta)*sin(phi), y = cos(theta)
        # Simplify: when phi=0, direction is in XY plane -> phi_E ~ theta
        # When phi=pi/2, direction is in YZ plane -> phi_E ~ theta

        # For simplicity and consistency, we use the same approach as
        # evaluate_gain_rrts_style_vectorized() but in scalar form:

        # Direction in local frame (boresight = +Y)
        # x = sin(theta)*cos(phi), y = cos(theta), z = sin(theta)*sin(phi)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)

        dir_x = sin_t * cos_p
        dir_y = cos_t
        dir_z = sin_t * sin_p

        # Elevation angle in YZ plane (E-plane)
        phi_E = np.arctan2(dir_z, dir_y)  # [-pi, pi]

        # Azimuth angle in XY plane (H-plane)
        phi_H = np.arctan2(dir_x, dir_y)  # [-pi, pi]

        # Map to pattern index [0, 360]
        # Pattern: index 0 = -180deg, index 180 = 0deg (boresight), index 360 = +180deg
        angle_E_deg = phi_E * 180.0 / np.pi + 180.0
        angle_H_deg = phi_H * 180.0 / np.pi + 180.0

        # Interpolate E-plane pattern
        idx_E_float = angle_E_deg * (self.nPhi_pattern - 1) / 360.0
        idx_E_float = np.clip(idx_E_float, 0.0, self.nPhi_pattern - 1)
        idx_E_low = int(np.floor(idx_E_float))
        idx_E_high = min(idx_E_low + 1, self.nPhi_pattern - 1)
        weight_E = idx_E_float - idx_E_low
        gain_E_dB = self.E_plane_dB[idx_E_low] * (1 - weight_E) + self.E_plane_dB[idx_E_high] * weight_E

        # Interpolate H-plane pattern
        idx_H_float = angle_H_deg * (self.nPhi_pattern - 1) / 360.0
        idx_H_float = np.clip(idx_H_float, 0.0, self.nPhi_pattern - 1)
        idx_H_low = int(np.floor(idx_H_float))
        idx_H_high = min(idx_H_low + 1, self.nPhi_pattern - 1)
        weight_H = idx_H_float - idx_H_low
        gain_H_dB = self.H_plane_dB[idx_H_low] * (1 - weight_H) + self.H_plane_dB[idx_H_high] * weight_H

        # Convert to linear
        gain_E_linear = 10.0 ** (gain_E_dB / 10.0)
        gain_H_linear = 10.0 ** (gain_H_dB / 10.0)

        # Combine based on mode
        if self.combine_mode == 'e_only':
            return gain_E_linear
        elif self.combine_mode == 'h_only':
            return gain_H_linear
        else:  # 'product'
            return self.C_scale * gain_E_linear * gain_H_linear

    def _build_cdf(self):
        """Build marginal and conditional CDFs for 2D sampling."""
        weights = np.zeros((self.n_theta, self.n_phi), dtype=np.float64)

        for i, theta in enumerate(self.theta_centers):
            for j, phi in enumerate(self.phi_centers):
                # Evaluate pattern gain in linear scale
                G_linear = self._evaluate_pattern_linear(theta, phi)

                # Weight = cos(theta) x G_r x solid_angle_measure
                # cos(theta) is the cosine weighting from RX-centric sampling
                # G_r is the antenna gain
                # solid_angle includes sin(theta) x d_theta x d_phi
                weights[i, j] = np.cos(theta) * G_linear * self.solid_angles[i, j]

        # Ensure all weights are non-negative
        weights = np.maximum(weights, 0.0)

        # Normalize to get probability mass
        self.total_weight = weights.sum()
        if self.total_weight > 1e-10:
            self.prob_mass = weights / self.total_weight
        else:
            # Fallback to uniform if pattern is degenerate
            self.prob_mass = np.ones_like(weights) / weights.size

        # Marginal CDF over theta (rows)
        self.marginal_theta = self.prob_mass.sum(axis=1)  # [n_theta]
        self.cdf_theta = np.cumsum(self.marginal_theta)
        if self.cdf_theta[-1] > 1e-10:
            self.cdf_theta /= self.cdf_theta[-1]  # Normalize to [0, 1]

        # Conditional CDF over phi given theta
        self.cdf_phi_given_theta = np.zeros((self.n_theta, self.n_phi))
        for i in range(self.n_theta):
            if self.marginal_theta[i] > 1e-10:
                conditional = self.prob_mass[i, :] / self.marginal_theta[i]
                self.cdf_phi_given_theta[i, :] = np.cumsum(conditional)
                if self.cdf_phi_given_theta[i, -1] > 1e-10:
                    self.cdf_phi_given_theta[i, :] /= self.cdf_phi_given_theta[i, -1]
            else:
                # Uniform distribution if marginal is zero
                self.cdf_phi_given_theta[i, :] = np.linspace(0, 1, self.n_phi)

    def sample(self, u: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        """
        Sample direction and return (omega, pdf).

        Uses inverse CDF sampling:
        1. Sample theta from marginal CDF using u1
        2. Sample phi from conditional CDF given theta using u2

        Args:
            u: 2D uniform random sample in [0,1]^2

        Returns:
            (direction, pdf) where:
            - direction is unit vector in local frame (boresight = +Y)
            - pdf is per-steradian probability density
        """
        u1, u2 = u

        # Sample theta bin from marginal CDF
        theta_idx = np.searchsorted(self.cdf_theta, u1, side='right')
        theta_idx = np.clip(theta_idx, 0, self.n_theta - 1)

        # Sample phi bin from conditional CDF
        phi_idx = np.searchsorted(self.cdf_phi_given_theta[theta_idx], u2, side='right')
        phi_idx = np.clip(phi_idx, 0, self.n_phi - 1)

        # Get bin center angles (could add uniform jitter within bin for smoothness)
        theta = self.theta_centers[theta_idx]
        phi = self.phi_centers[phi_idx]

        # Convert to Cartesian direction in local frame (boresight = +Y)
        # x = sin(theta)cos(phi), y = cos(theta), z = sin(theta)sin(phi)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)

        direction = np.array([sin_t * cos_p, cos_t, sin_t * sin_p])

        # Compute per-steradian PDF: p_ij / d_omega_ij
        pdf = self.prob_mass[theta_idx, phi_idx] / (self.solid_angles[theta_idx, phi_idx] + 1e-10)

        return direction, pdf

    def sample_batch(self, u1: np.ndarray, u2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized batch sampling.

        Args:
            u1: [N] Uniform random samples for theta
            u2: [N] Uniform random samples for phi

        Returns:
            (directions, pdfs) where:
            - directions: [N, 3] unit vectors in local frame
            - pdfs: [N] per-steradian probability densities
        """
        n = len(u1)
        directions = np.zeros((n, 3), dtype=np.float64)
        pdfs = np.zeros(n, dtype=np.float64)

        for i in range(n):
            directions[i], pdfs[i] = self.sample((u1[i], u2[i]))

        return directions, pdfs

    def pdf(self, omega: np.ndarray) -> float:
        """
        Evaluate per-steradian PDF for direction omega (local frame).

        Args:
            omega: Unit direction vector in local frame (boresight = +Y)

        Returns:
            Per-steradian probability density
        """
        # Convert direction to (theta, phi) in local frame
        # y = cos(theta), so theta = arccos(y)
        # x = sin(theta)cos(phi), z = sin(theta)sin(phi), so phi = atan2(z, x)
        theta = np.arccos(np.clip(omega[1], -1, 1))  # y component = cos(theta)
        phi = np.arctan2(omega[2], omega[0])  # atan2(z, x)
        if phi < 0:
            phi += 2 * np.pi

        # Clamp theta to valid range [0, pi/2]
        if theta > np.pi / 2:
            return 0.0  # Back hemisphere has zero probability

        # Find bin
        theta_idx = np.searchsorted(self.theta_edges, theta) - 1
        phi_idx = np.searchsorted(self.phi_edges, phi) - 1
        theta_idx = np.clip(theta_idx, 0, self.n_theta - 1)
        phi_idx = np.clip(phi_idx, 0, self.n_phi - 1)

        return self.prob_mass[theta_idx, phi_idx] / (self.solid_angles[theta_idx, phi_idx] + 1e-10)

    def pdf_batch(self, omega: np.ndarray) -> np.ndarray:
        """
        Vectorized PDF evaluation.

        Args:
            omega: [N, 3] Unit direction vectors in local frame

        Returns:
            [N] Per-steradian probability densities
        """
        n = len(omega)
        pdfs = np.zeros(n, dtype=np.float64)

        for i in range(n):
            pdfs[i] = self.pdf(omega[i])

        return pdfs


def create_pattern_importance_sampler(
    pattern_path: str,
    n_theta: int = 45,
    n_phi: int = 180,
    combine_mode: str = 'product'
) -> PatternImportanceSampler:
    """
    Convenience function to create PatternImportanceSampler from file.

    Args:
        pattern_path: Path to antenna pattern .npy file
        n_theta: Number of theta bins (default 45 for ~2deg resolution)
        n_phi: Number of phi bins (default 180 for ~2deg resolution)
        combine_mode: 'product', 'e_only', or 'h_only'

    Returns:
        PatternImportanceSampler instance
    """
    # Load pattern
    pattern_data = np.load(pattern_path)

    # Convert to RRTS format if needed
    if pattern_data.ndim == 2 and pattern_data.shape == (361, 2):
        # Simple [361, 2] format - convert to [361, 2, 5]
        pattern_rrts = np.zeros((361, 2, 5), dtype=np.float32)
        pattern_rrts[:, 0, 0] = pattern_data[:, 0]  # E-plane dB
        pattern_rrts[:, 1, 0] = pattern_data[:, 1]  # H-plane dB
        pattern_data = pattern_rrts

    sampler = PatternImportanceSampler(
        pattern_data=pattern_data,
        n_theta=n_theta,
        n_phi=n_phi,
        combine_mode=combine_mode
    )

    print(f"[PatternImportanceSampler] Created from {pattern_path}")
    print(f"  Grid: {n_theta} theta x {n_phi} phi bins")
    print(f"  Combine mode: {combine_mode}")
    print(f"  Total weight: {sampler.total_weight:.4f}")

    return sampler


__all__ = [
    'PatternImportanceSampler',
    'create_pattern_importance_sampler',
]
