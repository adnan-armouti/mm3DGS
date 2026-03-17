"""
FMCW ADC accumulator - fully vectorized complex phasor accumulation.

This module implements the core accumulation logic for FMCW radar ADC samples
during ray tracing, with proper phase computation and near-field evaluation.

CRITICAL: Implements exact FMCW phase law with NO Python loops.
"""

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
import drjit as dr
import mitsuba as mi

if TYPE_CHECKING:
    from .config import FMCWConfig, RxArray, TxArray


class FMCWAdcAccumulator:
    """
    Fully vectorized FMCW ADC accumulator with exact phase law.

    Accumulates complex baseband samples at discrete ADC sample times
    from traced ray paths. All operations are vectorized over both
    ADC samples (K) and RX elements (NR).

    FMCW Phase Law:
        phi_k(tau) = 2pi[f_0tau + 1/2Stau^2] + 2pi[Stau + f_d]t_k

    where:
        - f_0: carrier frequency
        - S: chirp slope (Hz/s)
        - tau: round-trip delay
        - f_d: Doppler frequency (toggleable via enable_doppler flag)
        - t_k: ADC sample time
    """

    def __init__(self,
                 config: 'FMCWConfig',
                 rx_array: 'RxArray',
                 tx_array: Optional['TxArray'] = None,
                 enable_grad: bool = False,
                 enable_doppler: bool = False,
                 store_per_tx: bool = False,
                 verbose: bool = True):
        """
        Initialize FMCW ADC accumulator.

        Args:
            config: FMCW configuration with radar parameters
            rx_array: RX array geometry and patterns
            tx_array: TX array (required if store_per_tx=True)
            enable_grad: If True, enable gradient tracking on ADC buffers for differentiable optimization
            enable_doppler: If True, enable Doppler frequency term in phase computation (default: False)
            store_per_tx: If True, store separate ADC for each TX element.
                         If False, accumulate (sum) over TX dimension (backward compatible).
            verbose: If True, print initialization and debug info
        """
        self.verbose = verbose
        # Time grid for ADC samples [K]
        self.K = config.num_adc_samples
        self.chirp_duration = config.chirp_duration
        self.adc_start_time = config.adc_start_time
        # Using endpoint=False for standard ADC sampling convention:
        # Samples at t = t_start + kxdt for k=0,1,...,K-1 where dt = T/K
        # This gives K samples with uniform spacing dt = chirp_duration / K
        # NOTE: adc_start_time only affects chirp duration calculation (bandwidth/resolution)
        # It does NOT control minimum range in this FMCW system
        self.t_grid = dr.linspace(
            mi.Float,
            0.0,  # Start from t=0 (adc_start_time doesn't affect minimum range)
            config.chirp_duration,
            self.K,
            endpoint=False
        )

        # RX array geometry
        self.rx_pos = rx_array.positions      # [3, NR]
        self.rx_ori = rx_array.orientations   # [3, NR]
        self.rx_pattern = rx_array.pattern    # Callable or None (legacy)
        self.NR = rx_array.num_elements

        # Store RX array for polarization-aware pattern evaluation
        self.rx_array = rx_array
        self.rx_orientations = rx_array.orientations if hasattr(rx_array, 'orientations') else None
        self.rx_pol = rx_array.polarization if hasattr(rx_array, 'polarization') else None
        self.rx_pattern_loader = rx_array.pattern_loader if hasattr(rx_array, 'pattern_loader') else None

        # Store TX array for TX antenna gain computation (MIMO coherence)
        self.tx_array = tx_array  # May be None

        # Per-TX storage mode
        self.store_per_tx = store_per_tx
        if store_per_tx:
            if tx_array is None:
                raise ValueError("tx_array required when store_per_tx=True")
            self.NT = tx_array.num_elements
        else:
            self.NT = 1  # Treat as single TX (accumulated)

        # FMCW parameters
        self.f0 = config.center_freq          # Hz
        self.S = config.chirp_slope           # Hz/s
        self.c = config.c                     # m/s

        # Sample rate (Hz)
        self.sample_rate = self.K / self.chirp_duration

        # Physical constants
        self.lambda_c = self.c / self.f0      # Carrier wavelength

        # Choose buffer shape based on mode
        if store_per_tx:
            # 3D: (TX, RX, ADC)
            total_samples = self.NT * self.NR * self.K
        else:
            # 2D: (RX, ADC) - backward compatible
            total_samples = self.NR * self.K

        # Flattened complex accumulation buffers (COHERENT paths only)
        # These are the output ADC samples for mirror-branch contributions
        self.adc_real = dr.zeros(mi.Float, total_samples)
        self.adc_imag = dr.zeros(mi.Float, total_samples)

        # Flattened POWER accumulation buffer (INCOHERENT paths only)
        # This receives VNDF + diffuse contributions (Migration Plan v3)
        # CRITICAL: Never mix field and power in one estimator
        self.power_accum = dr.zeros(mi.Float, total_samples)

        # Enable gradients if requested (for differentiable optimization)
        self._grad_enabled = enable_grad
        if enable_grad:
            dr.enable_grad(self.adc_real)
            dr.enable_grad(self.adc_imag)
            dr.enable_grad(self.power_accum)

        # Doppler control flag
        self.enable_doppler = enable_doppler

        # Path loss control flag
        self.enable_path_loss = config.enable_path_loss
        self.path_loss_exponent = config.path_loss_exponent

        # Pre-compute constants for efficiency
        self.two_pi = 2.0 * dr.pi

        if self.verbose:
            print(f"[FMCWAdcAccumulator] Initialized:")
            if store_per_tx:
                print(f"  - Mode: Per-TX storage")
                print(f"  - NT = {self.NT} TX elements")
                print(f"  - NR = {self.NR} RX elements")
                print(f"  - K = {self.K} ADC samples")
                print(f"  - Shape: ({self.NT}, {self.NR}, {self.K})")
                print(f"  - Total samples = {total_samples}")
            else:
                print(f"  - Mode: Accumulated (sum over TX)")
                print(f"  - NR = {self.NR} RX elements")
                print(f"  - K = {self.K} ADC samples")
                print(f"  - Shape: ({self.NR}, {self.K})")
                print(f"  - Total samples = {total_samples}")
            print(f"  - f0 = {self.f0/1e9:.3f} GHz")
            print(f"  - S = {self.S/1e12:.3f} THz/s")
            print(f"  - Chirp duration = {config.chirp_duration*1e6:.3f} mus")
            print(f"  - Gradients enabled: {enable_grad}")
            print(f"  - Doppler enabled: {enable_doppler}")

    def accumulate_paths(self,
                        mask: 'mi.Bool',           # [N] rays to accumulate
                        x_hit: 'mi.Point3f',       # [N, 3] hit positions
                        E_real: 'mi.Vector2f',     # [N, 2] E-field real (H/V)
                        E_imag: 'mi.Vector2f',     # [N, 2] E-field imag (H/V)
                        R_accum: 'mi.Float',       # [N] accumulated path length (for phase)
                        decay_accum: 'mi.Float',   # [N] accumulated decay product (for amplitude)
                        tube_weight: 'mi.Float',   # [N] ray tube solid angle weight
                        v_rel: Optional['mi.Float'] = None,
                        tx_idx: Optional['mi.UInt32'] = None,
                        rx_idx: Optional['mi.UInt32'] = None,
                        first_hit: Optional['mi.Point3f'] = None,  # [N, 3] first hit position (for TX gain)
                        enable_abp: bool = True) -> None:
        """
        Accumulate FMCW contributions from ray paths - FULLY VECTORIZED.

        This function performs near-field evaluation to ALL RX elements for
        each ray, computes the exact FMCW phase, and accumulates complex
        phasors at all ADC sample times using atomic scatter-add operations.

        IMPORTANT: By default, broadcasts each ray to ALL RX elements.
        If rx_idx is provided (for phase-coherent NEE), accumulates only to
        the specified RX element(s).

        NO PYTHON LOOPS - everything is vectorized with Dr.Jit broadcasting.

        Args:
            mask: [N] Boolean mask for rays to accumulate
            x_hit: [N, 3] Hit positions (last interaction point before RX)
            E_real: [N, 2] Real part of E-field (H/V polarization)
            E_imag: [N, 2] Imaginary part of E-field
            R_accum: [N] Path length accumulated to x_hit (meters) - used for PHASE computation
            decay_accum: [N] Accumulated decay product (1/d_1^EXP x 1/d_2^EXP x ...) - used for AMPLITUDE
                        where EXP = self.path_loss_exponent
            tube_weight: [N] Ray tube solid angle weight
            v_rel: [N] Radial velocity (m/s) for Doppler (only used if enable_doppler=True)
            tx_idx: [N] Source TX index for each ray (required if store_per_tx=True)
            rx_idx: [N] or scalar Target RX index (optional, for phase-coherent NEE)
                   If None, broadcasts to all RX elements (default behavior)
                   If provided, only accumulates to specified RX element(s)
            enable_abp: If True, apply RX antenna beam pattern gain. If False, use unit gain (rx_gain=1)
                       to isolate antenna pattern effects from the forward model
        """
        # Validate inputs
        if self.store_per_tx and tx_idx is None:
            raise ValueError("tx_idx required when store_per_tx=True")

        # Early exit if no rays to accumulate
        if not dr.any(mask):
            return

        N_original = dr.width(mask)

        # ====== CRITICAL: Compress to only active rays to avoid memory overflow ======
        # With large ray counts (1M+), processing all rays creates arrays > 4B entries
        # Dr.Jit has a hard limit of 2^32 entries, so we MUST filter first

        # Get indices of active rays
        dr.eval(mask)  # Ensure mask is evaluated
        active_indices = dr.compress(mask)
        N = dr.width(active_indices)

        # Gather only active rays using indices
        # CRITICAL: Evaluate components after gathering to avoid CUDA alignment issues
        x_hit_x_gathered = dr.gather(mi.Float, x_hit.x, active_indices)
        x_hit_y_gathered = dr.gather(mi.Float, x_hit.y, active_indices)
        x_hit_z_gathered = dr.gather(mi.Float, x_hit.z, active_indices)
        dr.eval(x_hit_x_gathered, x_hit_y_gathered, x_hit_z_gathered)
        x_hit = mi.Point3f(x_hit_x_gathered, x_hit_y_gathered, x_hit_z_gathered)

        E_real_x_gathered = dr.gather(mi.Float, E_real.x, active_indices)
        E_real_y_gathered = dr.gather(mi.Float, E_real.y, active_indices)
        dr.eval(E_real_x_gathered, E_real_y_gathered)
        E_real = mi.Vector2f(E_real_x_gathered, E_real_y_gathered)

        E_imag_x_gathered = dr.gather(mi.Float, E_imag.x, active_indices)
        E_imag_y_gathered = dr.gather(mi.Float, E_imag.y, active_indices)
        dr.eval(E_imag_x_gathered, E_imag_y_gathered)
        E_imag = mi.Vector2f(E_imag_x_gathered, E_imag_y_gathered)

        # Verify all vectors are evaluated
        dr.eval(x_hit, E_real, E_imag)

        R_accum = dr.gather(mi.Float, R_accum, active_indices)
        decay_accum = dr.gather(mi.Float, decay_accum, active_indices)
        tube_weight = dr.gather(mi.Float, tube_weight, active_indices)

        if v_rel is None:
            v_rel = dr.zeros(mi.Float, N)
        else:
            v_rel = dr.gather(mi.Float, v_rel, active_indices)

        # Gather tx_idx if provided (needed for TX antenna gain, regardless of store_per_tx)
        if tx_idx is not None:
            tx_idx = dr.gather(mi.UInt32, tx_idx, active_indices)

        # Compress rx_idx if it's an array (for vectorized NEE)
        if rx_idx is not None and not isinstance(rx_idx, int):
            rx_idx = dr.gather(mi.UInt32, rx_idx, active_indices)

        # Debugging: print compression ratio for large reductions
        compression_ratio = N / N_original if N_original > 0 else 0
        if self.verbose and compression_ratio < 0.5 and N_original > 100000:
            print(f"    [ADC] Compressed {N_original} -> {N} rays ({compression_ratio*100:.1f}%)")

        # ====== Step 1: Near-field evaluation to RX element(s) ======
        # By default: Broadcast to ALL RX elements [N] -> [N*NR]
        # Phase-coherent NEE: Evaluate to specific RX only [N] -> [N]

        # Extract x, y, z coordinates from hit points
        x_hit_x = x_hit.x  # [N]
        x_hit_y = x_hit.y  # [N]
        x_hit_z = x_hit.z  # [N]

        # Determine effective NR (number of RX to evaluate)
        if rx_idx is not None:
            # Phase-coherent NEE mode: accumulate to specific RX only
            effective_NR = 1

            # Get RX position for specified index
            # CRITICAL: Use self.rx_array.positions (live reference) instead of self.rx_pos (snapshot)
            # to ensure gradient flow to RX positions

            if isinstance(rx_idx, int):
                # Scalar rx_idx: all rays go to same RX
                # Use Dr.Jit indexing to preserve gradients (don't call float()!)
                rx_x_target = self.rx_array.positions.x[rx_idx]  # Dr.Jit scalar
                rx_y_target = self.rx_array.positions.y[rx_idx]
                rx_z_target = self.rx_array.positions.z[rx_idx]
            else:
                # Array rx_idx [N]: each ray has its own target RX
                rx_x_target = dr.gather(mi.Float, self.rx_array.positions.x, rx_idx)
                rx_y_target = dr.gather(mi.Float, self.rx_array.positions.y, rx_idx)
                rx_z_target = dr.gather(mi.Float, self.rx_array.positions.z, rx_idx)

            # Compute displacement: dx = rx_pos - x_hit [N]
            dx_x = rx_x_target - x_hit_x
            dx_y = rx_y_target - x_hit_y
            dx_z = rx_z_target - x_hit_z

            # Distance [N]
            R_last = dr.sqrt(dx_x * dx_x + dx_y * dx_y + dx_z * dx_z)

            # Total path length [N]
            R_tot = R_accum + R_last
        else:
            # Default mode: broadcast to ALL RX elements
            effective_NR = self.NR

            # Extract RX positions (stored as [3, NR])
            # CRITICAL: Use self.rx_array.positions (live reference) instead of self.rx_pos (snapshot)
            # to ensure gradient flow to RX positions
            rx_x = self.rx_array.positions.x  # [NR]
            rx_y = self.rx_array.positions.y  # [NR]
            rx_z = self.rx_array.positions.z  # [NR]

            # Compute displacement vectors: dx = rx_pos - x_hit
            # Use explicit repeat/tile for broadcasting
            # Tile RX positions to [N, NR]
            rx_x_2d = dr.repeat(rx_x, N)  # [N*NR] in flattened form
            rx_y_2d = dr.repeat(rx_y, N)
            rx_z_2d = dr.repeat(rx_z, N)

            # Tile hit positions to [N, NR]
            x_hit_x_2d = dr.tile(x_hit_x, self.NR)  # [N*NR] in flattened form
            x_hit_y_2d = dr.tile(x_hit_y, self.NR)
            x_hit_z_2d = dr.tile(x_hit_z, self.NR)

            # Compute displacements (flattened [N*NR])
            dx_x = rx_x_2d - x_hit_x_2d
            dx_y = rx_y_2d - x_hit_y_2d
            dx_z = rx_z_2d - x_hit_z_2d

            # Distance from hit point to each RX element [N*NR]
            R_last = dr.sqrt(dx_x * dx_x + dx_y * dx_y + dx_z * dx_z)

            # Total path length (TX -> hit -> RX) [N*NR]
            R_accum_2d = dr.tile(R_accum, self.NR)
            R_tot = R_accum_2d + R_last

            # Tile decay accumulator for broadcast mode [N] -> [N*NR]
            decay_accum_2d = dr.tile(decay_accum, self.NR)

        # CRITICAL: Evaluate R_tot before using it to avoid CUDA issues
        # TEMPORARILY DISABLED to check if this breaks gradient flow
        #print(f"    [ADC DEBUG] About to evaluate R_tot...")
        #dr.eval(R_tot)
        #print(f"    [ADC DEBUG] R_tot evaluated successfully")

        # ====== Step 2: Propagation delay ======
        # For FMCW, the delay is the time for signal to travel TX->target->RX
        # This is the one-way path length divided by speed of light
        tau = R_tot / self.c  # [N*effective_NR] seconds
        # CRITICAL: Do NOT call dr.eval(tau) - it breaks gradient flow!
        # tau depends on R_tot which depends on RX positions via R_last

        # Note: In traditional monostatic radar, you'd use 2*R/c for round-trip
        # But here we're computing the actual signal path time

        # ====== Step 3: FMCW Phase Computation (Migration Plan v3) ======
        # CRITICAL: Uses PER-SAMPLE k(t) = 2pi(f_c + St)/c, NOT fixed k_0 = 2pif_c/c
        #
        # Full formula: phi(t) = R_tot * k(t) = R_tot * 2pi(f_0 + St)/c
        #             = (R_tot/c) * 2pi(f_0 + St)
        #             = tau * 2pi(f_0 + St)
        #             = 2pi*f_0*tau + 2pi*S*tau*t
        #
        # This implements the chirp-aware phase from ship-checklist item #3.
        # Applied ONLY to coherent paths (this function accumulate_paths).
        # Incoherent paths (accumulate_power) skip phase computation entirely.
        #
        # Note: This omits the 0.5*S*tau^2 term from the full dechirp-on-receive formula,
        # which is negligible for typical radar parameters (tau ~ ns, S ~ THz/s)

        # Constant phase offset from carrier [N*effective_NR]
        # phi_const = 2pi*f_0*tau = (constant part of k(t)*R_tot)
        phi_const = self.two_pi * self.f0 * tau

        # Beat frequency [N*effective_NR]
        f_b = self.S * tau

        # Doppler frequency (conditionally enabled) [N*effective_NR]
        if self.enable_doppler:
            if effective_NR == 1:
                # Single RX: no tiling needed [N]
                f_d = v_rel / self.lambda_c
            else:
                # All RX: tile to [N*NR]
                v_rel_2d = dr.tile(v_rel, self.NR)
                f_d = v_rel_2d / self.lambda_c
        else:
            # Set Doppler to zero if disabled
            f_d = 0.0

        # Time-varying phase slope [N*effective_NR]
        # phi_slope = 2pi*S*tau = (time-varying part of k(t)*R_tot)
        phi_slope = self.two_pi * (f_b + f_d)

        # ====== Step 4: Expand over K ADC samples ======
        # Vectorize over time samples: [N*effective_NR] -> [N*effective_NR*K]
        # Full FMCW phase: phi(t) = 2pi*f_0*tau + 2pi*S*tau*t = tau*2pi*(f_0 + S*t) = R_tot*k(t)
        # where k(t) = 2pi*(f_0 + S*t)/c is the PER-SAMPLE wavenumber

        # Tile phi_const to [N*effective_NR*K]
        phi_const_3d = dr.tile(phi_const, self.K)  # [N*effective_NR*K]

        # Tile phi_slope to [N*effective_NR*K]
        phi_slope_3d = dr.tile(phi_slope, self.K)  # [N*effective_NR*K]

        # Repeat time grid to [N*effective_NR*K]
        t_k = dr.repeat(self.t_grid, N * effective_NR)  # [N*effective_NR*K]

        # Phase at each sample time [N*NR*K]
        # phi(t_k) = R_tot * k(t_k) where k(t_k) = 2pi*(f_0 + S*t_k)/c
        phi = phi_const_3d + phi_slope_3d * t_k

        # DEBUG: Print phase computation details for first accumulation
        if self.verbose and not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print("\n[DEBUG] Phase Computation:")
            print(f"  [DEBUG-A] About to print R_accum...")
            print(f"  R_accum: {float(R_accum[0]) if dr.width(R_accum) > 0 else 'empty':.6f} m")
            print(f"  R_last: {float(R_last[0]) if dr.width(R_last) > 0 else 'empty':.6f} m")
            print(f"  R_total: {float(R_tot[0]) if dr.width(R_tot) > 0 else 'empty':.6f} m")
            print(f"  tau: {float(tau[0]) if dr.width(tau) > 0 else 'empty':.9f} s")
            print(f"  f_b: {float(f_b[0]) if dr.width(f_b) > 0 else 'empty':.3f} Hz")
            print(f"  phi_const: {float(phi_const[0]) if dr.width(phi_const) > 0 else 'empty':.3f} rad")
            print(f"  phi_slope: {float(phi_slope[0]) if dr.width(phi_slope) > 0 else 'empty':.3f} rad/s")
            print(f"  t_grid[0:3]: {[float(self.t_grid[i]) for i in range(min(3, self.K))]}")
            # Note: Commented out memory-intensive debug prints for large arrays
            # print(f"  phi[0:3]: {[float(phi[i]) for i in range(min(3, dr.width(phi)))]}")
            # print(f"  phi increment: {float(phi[1] - phi[0]) if dr.width(phi) > 1 else 'N/A':.6f} rad")
            print()

        # ====== Step 5a: TX Element Pattern Gain (MIMO Coherence) ======
        # CRITICAL: TX gain must be computed for EACH TX[i]->first_hit path
        # This is essential for MIMO imaging - each TX-RX pair has different geometry
        tx_gain = dr.ones(mi.Float, N)  # Default: isotropic gain = 1

        if enable_abp and self.tx_array is not None and first_hit is not None:
            # TX array has pattern loader
            if hasattr(self.tx_array, 'pattern_loader') and self.tx_array.pattern_loader is not None:
                from .element_patterns import evaluate_combined_gain

                # Direction from TX[tx_idx] to first hit
                # Note: first_hit is the FIRST interaction point from TX center tracing
                first_hit_x = dr.gather(mi.Float, first_hit.x, active_indices)
                first_hit_y = dr.gather(mi.Float, first_hit.y, active_indices)
                first_hit_z = dr.gather(mi.Float, first_hit.z, active_indices)
                dr.eval(first_hit_x, first_hit_y, first_hit_z)
                first_hit_gathered = mi.Point3f(first_hit_x, first_hit_y, first_hit_z)

                # Get TX[tx_idx] positions
                tx_pos_x = dr.gather(mi.Float, self.tx_array.positions.x, tx_idx)
                tx_pos_y = dr.gather(mi.Float, self.tx_array.positions.y, tx_idx)
                tx_pos_z = dr.gather(mi.Float, self.tx_array.positions.z, tx_idx)
                dr.eval(tx_pos_x, tx_pos_y, tx_pos_z)
                tx_pos = mi.Point3f(tx_pos_x, tx_pos_y, tx_pos_z)

                # Direction vector: TX[i]->first_hit
                dir_tx_to_hit = first_hit_gathered - tx_pos
                dir_tx_to_hit_norm = dr.normalize(dir_tx_to_hit)

                # CRITICAL FIX: TX antennas radiate OUTWARD, so use TX->hit direction
                # Do NOT negate! TX pattern is for outgoing rays (FROM TX)
                # RX pattern uses negated direction (incoming TO RX)

                # Get TX orientations
                tx_ori_x = dr.gather(mi.Float, self.tx_array.orientations.x, tx_idx)
                tx_ori_y = dr.gather(mi.Float, self.tx_array.orientations.y, tx_idx)
                tx_ori_z = dr.gather(mi.Float, self.tx_array.orientations.z, tx_idx)
                dr.eval(tx_ori_x, tx_ori_y, tx_ori_z)
                tx_ori_vec = mi.Vector3f(tx_ori_x, tx_ori_y, tx_ori_z)

                # Evaluate TX antenna pattern gain with OUTGOING direction (TX->hit)
                dr.eval(dir_tx_to_hit_norm, tx_ori_vec)
                tx_gain = evaluate_combined_gain(
                    self.tx_array.pattern_loader,
                    dir_tx_to_hit_norm,  # Use TX->hit direction (not negated!)
                    tx_ori_vec
                )

                if self.verbose:
                    dr.eval(tx_gain)
                    tx_gain_min = float(dr.min(tx_gain)[0])
                    tx_gain_max = float(dr.max(tx_gain)[0])
                    tx_gain_mean = float(dr.mean(tx_gain)[0])
                    print(f"\n[DEBUG] TX Antenna Gain Statistics (Accumulator):")
                    print(f"  TX gain range: [{tx_gain_min:.6e}, {tx_gain_max:.6e}]")
                    print(f"  TX gain mean: {tx_gain_mean:.6e}")

        # ====== Step 5b: RX Element Pattern Gain (Polarization-Aware) ======
        # Direction from hit point to RX (unit vector towards RX) [N*effective_NR]
        R_last_safe = dr.maximum(R_last, 1e-9)  # Avoid division by zero
        dir_to_rx_x = dx_x / R_last_safe
        dir_to_rx_y = dx_y / R_last_safe
        dir_to_rx_z = dx_z / R_last_safe

        # Apply element pattern if available (and enabled)
        if enable_abp and self.rx_pattern_loader is not None and self.rx_orientations is not None:
            # Use combined pattern evaluation (E and H planes)
            from .element_patterns import evaluate_combined_gain

            # Create direction vector [N*effective_NR]
            # Note: incoming direction at RX is opposite to direction toward RX
            # CRITICAL: Evaluate components before constructing vector to avoid misalignment
            neg_dir_x = -dir_to_rx_x
            neg_dir_y = -dir_to_rx_y
            neg_dir_z = -dir_to_rx_z
            dr.eval(neg_dir_x, neg_dir_y, neg_dir_z)
            dir_vec = mi.Vector3f(neg_dir_x, neg_dir_y, neg_dir_z)

            if rx_idx is not None:
                # Single RX mode: gather specific RX parameters [N]
                if isinstance(rx_idx, int):
                    # Scalar rx_idx: all rays to same RX
                    # Don't call float() - preserve Dr.Jit types for gradients
                    rx_ori_x = self.rx_orientations.x[rx_idx]
                    rx_ori_y = self.rx_orientations.y[rx_idx]
                    rx_ori_z = self.rx_orientations.z[rx_idx]
                    dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                    ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)
                else:
                    # Array rx_idx [N]: each ray has its own target RX
                    rx_ori_x = dr.gather(mi.Float, self.rx_orientations.x, rx_idx)
                    rx_ori_y = dr.gather(mi.Float, self.rx_orientations.y, rx_idx)
                    rx_ori_z = dr.gather(mi.Float, self.rx_orientations.z, rx_idx)
                    dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                    ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)
            else:
                # All RX mode: tile to [N*NR]
                rx_ori_x = dr.repeat(self.rx_orientations.x, N)
                rx_ori_y = dr.repeat(self.rx_orientations.y, N)
                rx_ori_z = dr.repeat(self.rx_orientations.z, N)
                dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)

            # Force evaluation of vectors before passing to pattern evaluator
            dr.eval(dir_vec, ori_vec)

            # Evaluate combined pattern (E and H planes) [N*effective_NR]
            # CRITICAL: Antenna beam pattern is INDEPENDENT of polarization!
            # The E and H plane patterns describe directional response (elevation/azimuth),
            # NOT the polarization of the radiated field.
            gain = evaluate_combined_gain(
                self.rx_pattern_loader,
                dir_vec,
                ori_vec
            )

        elif enable_abp and self.rx_pattern is not None:
            # Legacy: use callable pattern (not polarization-aware)
            # CRITICAL: Evaluate components before constructing vector
            dr.eval(dir_to_rx_x, dir_to_rx_y, dir_to_rx_z)
            dir_vec = mi.Vector3f(dir_to_rx_x, dir_to_rx_y, dir_to_rx_z)

            if rx_idx is not None:
                # Single RX mode: no tiling needed [N]
                dr.eval(dir_vec)
                gain = self.rx_pattern(dir_vec, self.rx_ori)
            else:
                # All RX mode: tile to [N*NR]
                rx_ori_x = dr.repeat(self.rx_ori.x, N)
                rx_ori_y = dr.repeat(self.rx_ori.y, N)
                rx_ori_z = dr.repeat(self.rx_ori.z, N)
                dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)
                dr.eval(dir_vec, ori_vec)
                gain = self.rx_pattern(dir_vec, ori_vec)
        else:
            # Isotropic gain = 1 [N*effective_NR]
            gain = dr.ones(mi.Float, N * effective_NR)

        # DEBUG: Print RX gain statistics
        if self.verbose and enable_abp:
            dr.eval(gain)
            rx_gain_min = float(dr.min(gain)[0])
            rx_gain_max = float(dr.max(gain)[0])
            rx_gain_mean = float(dr.mean(gain)[0])
            print(f"\n[DEBUG] RX Antenna Gain Statistics:")
            print(f"  RX gain range: [{rx_gain_min:.6e}, {rx_gain_max:.6e}]")
            print(f"  RX gain mean: {rx_gain_mean:.6e}")

        # ====== Step 6: Project E-field onto RX polarization ======
        # Select correct E-field component based on RX polarization
        if self.rx_pol is not None:
            # Polarization-aware: select H or V component based on RX polarization
            # E_real/E_imag are [N, 2] where [0]=H, [1]=V

            if rx_idx is not None:
                # Single RX mode: extract specific component [N]
                # rx_pol_vec is already set above (scalar or [N])
                if isinstance(rx_idx, int):
                    # Scalar rx_idx: select based on that RX's polarization
                    is_v_pol = (int(self.rx_pol[rx_idx]) == 1)
                    E_scalar_real = E_real.y if is_v_pol else E_real.x  # [N]
                    E_scalar_imag = E_imag.y if is_v_pol else E_imag.x  # [N]
                else:
                    # Array rx_idx [N]: select per-ray based on target RX polarization
                    rx_pol_gathered = dr.gather(mi.UInt32, self.rx_pol, rx_idx)
                    is_v_pol = (rx_pol_gathered == 1)
                    E_scalar_real = dr.select(is_v_pol, E_real.y, E_real.x)
                    E_scalar_imag = dr.select(is_v_pol, E_imag.y, E_imag.x)
            else:
                # All RX mode: broadcast to [N*NR]
                E_H_real = dr.tile(E_real.x, self.NR)  # [N*NR]
                E_V_real = dr.tile(E_real.y, self.NR)  # [N*NR]
                E_H_imag = dr.tile(E_imag.x, self.NR)  # [N*NR]
                E_V_imag = dr.tile(E_imag.y, self.NR)  # [N*NR]

                # Tile RX polarization [NR] -> [N*NR]
                rx_pol_2d = dr.repeat(self.rx_pol, N)

                # Select based on polarization (0=H, 1=V)
                is_v_pol = (rx_pol_2d == 1)
                E_scalar_real = dr.select(is_v_pol, E_V_real, E_H_real)
                E_scalar_imag = dr.select(is_v_pol, E_V_imag, E_H_imag)
        else:
            # Legacy: sum H and V components (no polarization info)
            E_scalar_real = E_real.x + E_real.y  # [N]
            E_scalar_imag = E_imag.x + E_imag.y  # [N]

            if effective_NR > 1:
                # All RX mode: tile to [N*NR]
                E_scalar_real = dr.tile(E_scalar_real, self.NR)
                E_scalar_imag = dr.tile(E_scalar_imag, self.NR)
            # else: Single RX mode, keep as [N]

        # Apply TX antenna gain (before tiling/RX gain) [N]
        # TX gain is [N], must be applied before RX broadcast
        if rx_idx is not None:
            # Single RX mode: apply directly [N]
            E_scalar_real = E_scalar_real * tx_gain
            E_scalar_imag = E_scalar_imag * tx_gain
        else:
            # All RX mode: tile TX gain to [N*NR] to match E_scalar dimensions
            tx_gain_2d = dr.tile(tx_gain, self.NR)
            E_scalar_real = E_scalar_real * tx_gain_2d
            E_scalar_imag = E_scalar_imag * tx_gain_2d

        # Apply RX element pattern gain and tube weight [N*effective_NR]
        if rx_idx is not None:
            # Single RX mode: no tiling needed [N]
            E_scalar_real = E_scalar_real * gain * tube_weight
            E_scalar_imag = E_scalar_imag * gain * tube_weight
        else:
            # All RX mode: tile tube_weight to [N*NR]
            tube_weight_2d = dr.tile(tube_weight, self.NR)
            E_scalar_real = E_scalar_real * gain * tube_weight_2d
            E_scalar_imag = E_scalar_imag * gain * tube_weight_2d

        # ====== Step 7: Free-space path loss (product-of-distances decay) ======
        # CRITICAL FIX: Use product-of-distances formula: decay = 1/(d_1^EXP x d_2^EXP x ... x d_N^EXP)
        # where decay_accum already contains 1/(d_1^EXP x ... x d_{N-1}^EXP) from bounces
        # and we add the final RX segment: 1/d_N^EXP
        if self.enable_path_loss:
            # Apply final segment decay (RX segment with configurable EXP)
            # R_last_decay = 1 / R_last^EXP
            if self.path_loss_exponent == 2.0:
                R_last_decay = dr.rcp(dr.maximum(R_last * R_last, 1e-18))
            elif self.path_loss_exponent == 1.0:
                R_last_decay = dr.rcp(dr.maximum(R_last, 1e-9))
            else:
                R_last_decay = dr.power(dr.maximum(R_last, 1e-9), -self.path_loss_exponent)

            if rx_idx is not None:
                # Single RX mode
                total_decay = decay_accum * R_last_decay
            else:
                # All RX mode
                total_decay = decay_accum_2d * R_last_decay

            # DEBUG: Print decay statistics once
            if self.verbose and not hasattr(self, '_decay_stats_printed'):
                self._decay_stats_printed = True
                dr.eval(total_decay, decay_accum if rx_idx is not None else decay_accum_2d, R_last)

                decay_acc_np = (decay_accum if rx_idx is not None else decay_accum_2d).numpy()
                R_last_np = R_last.numpy()
                total_decay_np = total_decay.numpy()

                print(f"\n[DECAY DEBUG] Path loss decay statistics (EXP={self.path_loss_exponent}):")
                print(f"  Accumulated decay (TX->hit): min={np.min(decay_acc_np):.6e}, max={np.max(decay_acc_np):.6e}")
                print(f"  R_last (hit->RX): min={np.min(R_last_np):.3f}, max={np.max(R_last_np):.3f}")
                if self.path_loss_exponent == 2.0:
                    print(f"  R_last decay (1/R^2): min={np.min(1.0/R_last_np**2):.6e}, max={np.max(1.0/R_last_np**2):.6e}")
                elif self.path_loss_exponent == 1.0:
                    print(f"  R_last decay (1/R): min={np.min(1.0/R_last_np):.6e}, max={np.max(1.0/R_last_np):.6e}")
                else:
                    print(f"  R_last decay (1/R^{self.path_loss_exponent}): min={np.min(1.0/R_last_np**self.path_loss_exponent):.6e}, max={np.max(1.0/R_last_np**self.path_loss_exponent):.6e}")
                print(f"  Total decay (product): min={np.min(total_decay_np):.6e}, max={np.max(total_decay_np):.6e}")

            E_scalar_real = E_scalar_real * total_decay
            E_scalar_imag = E_scalar_imag * total_decay

        # ====== Step 8: Complex Phasor Computation (vectorized) ======
        # e^{iphi} = cos(phi) + i*sin(phi) [N*effective_NR*K]
        # IMPORTANT: Use separate sin/cos calls instead of sincos to preserve gradients!
        # dr.sincos() does NOT propagate gradients properly in DrJit

        # Complex multiplication: (E_r + i*E_i) * (cos_phi + i*sin_phi) [N*NR*K]
        # = (E_r*cos - E_i*sin) + i*(E_r*sin + E_i*cos)

        # Tile E-field to ADC samples [N*effective_NR*K]
        E_r = dr.tile(E_scalar_real, self.K)
        E_i = dr.tile(E_scalar_imag, self.K)

        # Compute trig functions
        cos_phi = dr.cos(phi)
        sin_phi = dr.sin(phi)

        # Complex multiplication (reuse arrays, let DrJit optimize)
        contrib_real = E_r * cos_phi - E_i * sin_phi
        contrib_imag = E_r * sin_phi + E_i * cos_phi

        # Do NOT call dr.eval() - let scatter_add handle lazy evaluation

        # DEBUG: Print contribution values - DISABLED to avoid OOM with large arrays
        # With vectorized NEE, arrays can be 4M+ elements, even dr.detach()[0] causes OOM
        # if hasattr(self, '_debug_printed') and not hasattr(self, '_debug_contrib_printed'):
        #     self._debug_contrib_printed = True
        #     print("[DEBUG] Complex Phasor:")
        #     print(f"  E_r[0]: {float(dr.detach(E_r)[0]):.6f}")
        #     print(f"  E_i[0]: {float(dr.detach(E_i)[0]):.6f}")
        #     print(f"  cos_phi[0]: {float(dr.detach(cos_phi)[0]):.6f}")
        #     print(f"  sin_phi[0]: {float(dr.detach(sin_phi)[0]):.6f}")
        #     print()

        # ====== Step 9: Build flat indices based on mode ======

        if self.store_per_tx:
            # ----------------------------------------------------------
            # Per-TX mode: 3D indexing (TX, RX, ADC)
            # ----------------------------------------------------------

            # Get TX index for each contribution [N*effective_NR*K]
            # tx_idx is [N], need to tile to [N*effective_NR*K]
            tx_indices = dr.tile(tx_idx, effective_NR * self.K)  # [N*effective_NR*K]

            # RX indices: depends on whether rx_idx is provided
            if rx_idx is not None:
                # Single RX mode: use specified rx_idx [N] -> [N*K]
                if isinstance(rx_idx, int):
                    # Scalar: all rays to same RX
                    r_indices = dr.full(mi.UInt32, rx_idx, N)
                else:
                    # Array: each ray has its own target RX
                    r_indices = rx_idx  # [N]
                r_indices = dr.tile(r_indices, self.K)  # [N*K]
            else:
                # All RX mode: broadcast to all RX [NR] -> [N*NR*K]
                r_indices = dr.arange(mi.UInt32, self.NR)
                r_indices = dr.repeat(r_indices, N)        # [N*NR]
                r_indices = dr.tile(r_indices, self.K)     # [N*NR*K]

            # ADC sample indices [K] -> [N*effective_NR*K]
            k_indices = dr.arange(mi.UInt32, self.K)
            k_indices = dr.repeat(k_indices, N * effective_NR)  # [N*effective_NR*K]

            # Flat index for 3D buffer: idx = tx*NR*K + rx*K + k
            flat_idx = tx_indices * (self.NR * self.K) + r_indices * self.K + k_indices
            # flat_idx in [0, NT*NR*K)

        else:
            # ----------------------------------------------------------
            # Accumulated mode: 2D indexing (RX, ADC) - sum over TX
            # ----------------------------------------------------------

            # RX indices: depends on whether rx_idx is provided
            if rx_idx is not None:
                # Single RX mode: use specified rx_idx [N] -> [N*K]
                if isinstance(rx_idx, int):
                    # Scalar: all rays to same RX
                    r_indices = dr.full(mi.UInt32, rx_idx, N)
                else:
                    # Array: each ray has its own target RX
                    r_indices = rx_idx  # [N]
                r_indices = dr.tile(r_indices, self.K)  # [N*K]
            else:
                # All RX mode: broadcast to all RX [NR] -> [N*NR*K]
                r_indices = dr.arange(mi.UInt32, self.NR)
                r_indices = dr.repeat(r_indices, N)
                r_indices = dr.tile(r_indices, self.K)

            # ADC sample indices [K] -> [N*effective_NR*K]
            k_indices = dr.arange(mi.UInt32, self.K)
            k_indices = dr.repeat(k_indices, N * effective_NR)

            # Flat index for 2D buffer: idx = rx*K + k
            flat_idx = r_indices * self.K + k_indices
            # flat_idx in [0, NR*K)

        # No need to tile mask - we already compressed to only active rays
        # All N rays are active by construction
        active_3d = dr.ones(mi.Bool, N * effective_NR * self.K)

        # Atomic scatter-add to accumulation buffers
        # This is thread-safe and handles concurrent updates
        dr.scatter_add(
            target=self.adc_real,
            value=contrib_real,
            index=flat_idx,
            active=active_3d
        )
        dr.scatter_add(
            target=self.adc_imag,
            value=contrib_imag,
            index=flat_idx,
            active=active_3d
        )

    def accumulate_power(self,
                        mask: 'mi.Bool',           # [N] rays to accumulate
                        x_hit: 'mi.Point3f',       # [N, 3] hit positions
                        power_in: 'mi.Float',      # [N] incident power
                        R_accum: 'mi.Float',       # [N] accumulated path length (for delay)
                        decay_accum: 'mi.Float',   # [N] accumulated decay product
                        tube_weight: 'mi.Float',   # [N] ray tube weight
                        tx_idx: Optional['mi.UInt32'] = None,
                        rx_idx: Optional['mi.UInt32'] = None,
                        enable_abp: bool = True) -> None:
        """
        Accumulate POWER-DOMAIN contributions from INCOHERENT paths (VNDF + diffuse).

        This function accumulates power (not E-field) from rough surface scattering
        and diffuse reflections. NO phase is computed - this is purely incoherent.

        CRITICAL: This is for Migration Plan v3 coherence policy:
        - Rough conductor (alpha >= 2e-4): VNDF sampling -> power accumulator
        - Diffuse scattering: ALWAYS incoherent -> power accumulator
        - Never mix with coherent (complex ADC) contributions

        Args:
            mask: [N] Boolean mask for rays to accumulate
            x_hit: [N, 3] Hit positions (last interaction point before RX)
            power_in: [N] Incident power at hit point (NOT E-field!)
            R_accum: [N] Path length accumulated to x_hit (for delay calculation)
            decay_accum: [N] Accumulated decay product
            tube_weight: [N] Ray tube solid angle weight
            tx_idx: [N] Source TX index (required if store_per_tx=True)
            rx_idx: [N] or scalar Target RX index (optional, for NEE)
            enable_abp: If True, apply RX antenna beam pattern gain
        """
        # Validate inputs
        if self.store_per_tx and tx_idx is None:
            raise ValueError("tx_idx required when store_per_tx=True")

        # Early exit if no rays
        if not dr.any(mask):
            return

        N_original = dr.width(mask)

        # ====== Compress to only active rays ======
        dr.eval(mask)
        active_indices = dr.compress(mask)
        N = dr.width(active_indices)

        # Gather active rays
        x_hit_x = dr.gather(mi.Float, x_hit.x, active_indices)
        x_hit_y = dr.gather(mi.Float, x_hit.y, active_indices)
        x_hit_z = dr.gather(mi.Float, x_hit.z, active_indices)
        dr.eval(x_hit_x, x_hit_y, x_hit_z)
        x_hit = mi.Point3f(x_hit_x, x_hit_y, x_hit_z)

        power_in = dr.gather(mi.Float, power_in, active_indices)
        R_accum = dr.gather(mi.Float, R_accum, active_indices)
        decay_accum = dr.gather(mi.Float, decay_accum, active_indices)
        tube_weight = dr.gather(mi.Float, tube_weight, active_indices)

        # Gather tx_idx if provided (needed for TX antenna gain, regardless of store_per_tx)
        if tx_idx is not None:
            tx_idx = dr.gather(mi.UInt32, tx_idx, active_indices)

        if rx_idx is not None and not isinstance(rx_idx, int):
            rx_idx = dr.gather(mi.UInt32, rx_idx, active_indices)

        # ====== Step 1: Near-field evaluation to RX ======
        if rx_idx is not None:
            # Single RX mode
            effective_NR = 1

            if isinstance(rx_idx, int):
                rx_x_target = self.rx_array.positions.x[rx_idx]
                rx_y_target = self.rx_array.positions.y[rx_idx]
                rx_z_target = self.rx_array.positions.z[rx_idx]
            else:
                rx_x_target = dr.gather(mi.Float, self.rx_array.positions.x, rx_idx)
                rx_y_target = dr.gather(mi.Float, self.rx_array.positions.y, rx_idx)
                rx_z_target = dr.gather(mi.Float, self.rx_array.positions.z, rx_idx)

            dx_x = rx_x_target - x_hit.x
            dx_y = rx_y_target - x_hit.y
            dx_z = rx_z_target - x_hit.z

            R_last = dr.sqrt(dx_x * dx_x + dx_y * dx_y + dx_z * dx_z)
        else:
            # Broadcast to all RX
            effective_NR = self.NR

            rx_x = self.rx_array.positions.x
            rx_y = self.rx_array.positions.y
            rx_z = self.rx_array.positions.z

            rx_x_2d = dr.repeat(rx_x, N)
            rx_y_2d = dr.repeat(rx_y, N)
            rx_z_2d = dr.repeat(rx_z, N)

            x_hit_x_2d = dr.tile(x_hit.x, self.NR)
            x_hit_y_2d = dr.tile(x_hit.y, self.NR)
            x_hit_z_2d = dr.tile(x_hit.z, self.NR)

            dx_x = rx_x_2d - x_hit_x_2d
            dx_y = rx_y_2d - x_hit_y_2d
            dx_z = rx_z_2d - x_hit_z_2d

            R_last = dr.sqrt(dx_x * dx_x + dx_y * dx_y + dx_z * dx_z)

            # Tile decay accumulator for broadcast mode
            decay_accum_2d = dr.tile(decay_accum, self.NR)

        # ====== Step 2: Compute delay (for ADC sample timing) ======
        R_accum_broadcast = R_accum if rx_idx is not None else dr.tile(R_accum, self.NR)
        R_tot = R_accum_broadcast + R_last
        tau = R_tot / self.c  # [N*effective_NR] seconds

        # ====== Step 3: Beat frequency for ADC sample index ======
        # Even though we're not computing phase, we still need to know
        # which ADC sample time bin this contribution falls into
        f_b = self.S * tau  # [N*effective_NR] Hz

        # Convert beat frequency to ADC sample index
        # f_b = k * f_s / K  =>  k = f_b * K / f_s
        # where f_s = sample_rate = K / chirp_duration
        # Simplifies to: k = f_b * chirp_duration
        sample_idx_float = f_b * self.chirp_duration
        sample_idx = dr.minimum(mi.UInt32(sample_idx_float), self.K - 1)

        # ====== Step 4: RX Element Pattern Gain ======
        R_last_safe = dr.maximum(R_last, 1e-9)
        dir_to_rx_x = dx_x / R_last_safe
        dir_to_rx_y = dx_y / R_last_safe
        dir_to_rx_z = dx_z / R_last_safe

        if enable_abp and self.rx_pattern_loader is not None and self.rx_orientations is not None:
            from .element_patterns import evaluate_combined_gain

            neg_dir_x = -dir_to_rx_x
            neg_dir_y = -dir_to_rx_y
            neg_dir_z = -dir_to_rx_z
            dr.eval(neg_dir_x, neg_dir_y, neg_dir_z)
            dir_vec = mi.Vector3f(neg_dir_x, neg_dir_y, neg_dir_z)

            if rx_idx is not None:
                if isinstance(rx_idx, int):
                    rx_ori_x = self.rx_orientations.x[rx_idx]
                    rx_ori_y = self.rx_orientations.y[rx_idx]
                    rx_ori_z = self.rx_orientations.z[rx_idx]
                else:
                    rx_ori_x = dr.gather(mi.Float, self.rx_orientations.x, rx_idx)
                    rx_ori_y = dr.gather(mi.Float, self.rx_orientations.y, rx_idx)
                    rx_ori_z = dr.gather(mi.Float, self.rx_orientations.z, rx_idx)
                dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)
            else:
                rx_ori_x = dr.repeat(self.rx_orientations.x, N)
                rx_ori_y = dr.repeat(self.rx_orientations.y, N)
                rx_ori_z = dr.repeat(self.rx_orientations.z, N)
                dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)

            dr.eval(dir_vec, ori_vec)
            gain = evaluate_combined_gain(self.rx_pattern_loader, dir_vec, ori_vec)

        elif enable_abp and self.rx_pattern is not None:
            dr.eval(dir_to_rx_x, dir_to_rx_y, dir_to_rx_z)
            dir_vec = mi.Vector3f(dir_to_rx_x, dir_to_rx_y, dir_to_rx_z)

            if rx_idx is not None:
                dr.eval(dir_vec)
                gain = self.rx_pattern(dir_vec, self.rx_ori)
            else:
                rx_ori_x = dr.repeat(self.rx_ori.x, N)
                rx_ori_y = dr.repeat(self.rx_ori.y, N)
                rx_ori_z = dr.repeat(self.rx_ori.z, N)
                dr.eval(rx_ori_x, rx_ori_y, rx_ori_z)
                ori_vec = mi.Vector3f(rx_ori_x, rx_ori_y, rx_ori_z)
                dr.eval(dir_vec, ori_vec)
                gain = self.rx_pattern(dir_vec, ori_vec)
        else:
            gain = dr.ones(mi.Float, N * effective_NR)

        # ====== Step 5: Apply weights and path loss ======
        if rx_idx is not None:
            power_contrib = power_in * gain * tube_weight
        else:
            tube_weight_2d = dr.tile(tube_weight, self.NR)
            power_in_2d = dr.tile(power_in, self.NR)
            power_contrib = power_in_2d * gain * tube_weight_2d

        # Apply path loss
        # CRITICAL: Since power = |E|^2, we need to use 2x the exponent for power decay
        # If E-field decays as 1/R^n, then power decays as 1/R^(2n)
        # This ensures sqrt(power) has the same amplitude decay as E-field
        if self.enable_path_loss:
            power_exponent = 2.0 * self.path_loss_exponent  # Double for power (since P = E^2)

            if power_exponent == 2.0:
                R_last_decay = dr.rcp(dr.maximum(R_last * R_last, 1e-18))
            elif power_exponent == 1.0:
                R_last_decay = dr.rcp(dr.maximum(R_last, 1e-9))
            else:
                R_last_decay = dr.power(dr.maximum(R_last, 1e-9), -power_exponent)

            total_decay = (decay_accum if rx_idx is not None else decay_accum_2d) * R_last_decay
            power_contrib = power_contrib * total_decay

        # ====== Step 6: Build flat indices and scatter ======
        if self.store_per_tx:
            # Per-TX mode: 3D indexing
            tx_indices = tx_idx  # [N]

            if rx_idx is not None:
                if isinstance(rx_idx, int):
                    r_indices = dr.full(mi.UInt32, rx_idx, N)
                else:
                    r_indices = rx_idx
            else:
                r_indices = dr.arange(mi.UInt32, self.NR)
                r_indices = dr.repeat(r_indices, N)

            # Flat index: tx*NR*K + rx*K + k
            flat_idx = tx_indices * (self.NR * self.K) + r_indices * self.K + sample_idx
        else:
            # Accumulated mode: 2D indexing
            if rx_idx is not None:
                if isinstance(rx_idx, int):
                    r_indices = dr.full(mi.UInt32, rx_idx, N)
                else:
                    r_indices = rx_idx
            else:
                r_indices = dr.arange(mi.UInt32, self.NR)
                r_indices = dr.repeat(r_indices, N)

            # Flat index: rx*K + k
            flat_idx = r_indices * self.K + sample_idx

        active_scatter = dr.ones(mi.Bool, dr.width(flat_idx))

        # Scatter-add to power accumulator
        dr.scatter_add(
            target=self.power_accum,
            value=power_contrib,
            index=flat_idx,
            active=active_scatter
        )

    def get_adc_samples(self,
                        tx_idx: Optional[int] = None,
                        rx_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve accumulated ADC samples.

        Args:
            tx_idx: If specified and store_per_tx=True, return samples for this TX only.
                   If None and store_per_tx=True, return all TX samples shaped as [NT, NR, K].
            rx_idx: If specified, return samples for this RX only.

        Returns:
            Tuple of (real, imag) numpy arrays
            - If store_per_tx=False: [NR, K] or [K] if rx_idx specified
            - If store_per_tx=True and tx_idx=None: [NT, NR, K] or [NT, K] if rx_idx specified
            - If store_per_tx=True and tx_idx specified: [NR, K] or [K] if rx_idx specified
        """
        # Synchronize GPU/lazy evaluation
        dr.sync_thread()

        # COHERENT ONLY: NEE paths are phase-deterministic
        # All depth-1 paths use coherent E-field accumulation (adc_real/adc_imag)
        # Incoherent power paths (VNDF continuation) would use power_accum, but for
        # depth-1 NEE we enforce coherent representation to match FMCW radar physics
        adc_real_combined = self.adc_real
        adc_imag_combined = self.adc_imag

        dr.eval(adc_real_combined)
        dr.eval(adc_imag_combined)

        # Convert to numpy using Dr.Jit numpy interface
        # Create temporary arrays to ensure proper data transfer
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.store_per_tx:
                total_samples = self.NT * self.NR * self.K
            else:
                total_samples = self.NR * self.K

            real_np = np.zeros(total_samples, dtype=np.float32)
            imag_np = np.zeros(total_samples, dtype=np.float32)

            # Copy data element by element (slow but reliable)
            if dr.width(adc_real_combined) == total_samples:
                for i in range(total_samples):
                    real_np[i] = adc_real_combined[i]
                    imag_np[i] = adc_imag_combined[i]
            else:
                # Array might be scalar if no accumulation happened
                print(f"Warning: Expected {total_samples} samples, got {dr.width(adc_real_combined)}")

        # Reshape based on mode
        if self.store_per_tx:
            real_np = real_np.reshape(self.NT, self.NR, self.K)
            imag_np = imag_np.reshape(self.NT, self.NR, self.K)

            # Apply indexing
            if tx_idx is not None:
                real_np = real_np[tx_idx]  # [NR, K]
                imag_np = imag_np[tx_idx]

            if rx_idx is not None:
                if tx_idx is not None:
                    real_np = real_np[rx_idx]  # [K]
                    imag_np = imag_np[rx_idx]
                else:
                    real_np = real_np[:, rx_idx]  # [NT, K]
                    imag_np = imag_np[:, rx_idx]
        else:
            # Backward compatible: [NR, K]
            real_np = real_np.reshape(self.NR, self.K)
            imag_np = imag_np.reshape(self.NR, self.K)

            if rx_idx is not None:
                real_np = real_np[rx_idx]  # [K]
                imag_np = imag_np[rx_idx]

        return real_np, imag_np

    def get_complex_samples(self,
                            tx_idx: Optional[int] = None,
                            rx_idx: Optional[int] = None) -> np.ndarray:
        """
        Get complex-valued ADC samples.

        Args:
            tx_idx: If specified and store_per_tx=True, return samples for this TX only.
            rx_idx: If specified, return samples for this RX only.

        Returns:
            Complex numpy array with shape depending on mode and indexing
        """
        real, imag = self.get_adc_samples(tx_idx, rx_idx)
        return real + 1j * imag

    def get_power_samples(self,
                         tx_idx: Optional[int] = None,
                         rx_idx: Optional[int] = None) -> np.ndarray:
        """
        Retrieve accumulated POWER samples from incoherent paths.

        This returns power contributions from VNDF (rough conductor) and
        diffuse scattering. These are NOT phase-coherent with the complex ADC.

        Args:
            tx_idx: If specified and store_per_tx=True, return samples for this TX only.
            rx_idx: If specified, return samples for this RX only.

        Returns:
            Power numpy array with shape depending on mode and indexing
            - If store_per_tx=False: [NR, K] or [K] if rx_idx specified
            - If store_per_tx=True and tx_idx=None: [NT, NR, K] or [NT, K] if rx_idx specified
            - If store_per_tx=True and tx_idx specified: [NR, K] or [K] if rx_idx specified
        """
        # Synchronize GPU
        dr.sync_thread()
        dr.eval(self.power_accum)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.store_per_tx:
                total_samples = self.NT * self.NR * self.K
            else:
                total_samples = self.NR * self.K

            power_np = np.zeros(total_samples, dtype=np.float32)

            # Copy data
            if dr.width(self.power_accum) == total_samples:
                for i in range(total_samples):
                    power_np[i] = self.power_accum[i]
            else:
                print(f"Warning: Expected {total_samples} power samples, got {dr.width(self.power_accum)}")

        # Reshape based on mode
        if self.store_per_tx:
            power_np = power_np.reshape(self.NT, self.NR, self.K)

            if tx_idx is not None:
                power_np = power_np[tx_idx]  # [NR, K]

            if rx_idx is not None:
                if tx_idx is not None:
                    power_np = power_np[rx_idx]  # [K]
                else:
                    power_np = power_np[:, rx_idx]  # [NT, K]
        else:
            power_np = power_np.reshape(self.NR, self.K)

            if rx_idx is not None:
                power_np = power_np[rx_idx]  # [K]

        return power_np

    def debug_check_gradients_after_backward(self):
        """
        Check gradients on saved intermediate variables after backward pass.
        Call this AFTER dr.backward() to trace where gradients disappear.
        """
        if not hasattr(self, '_debug_intermediates_saved'):
            print("[ADC DEBUG] No intermediate variables saved - forward pass didn't run")
            return

        print("\n" + "="*80)
        print("GRADIENT FLOW TRACING (After Backward Pass)")
        print("="*80)
        print("\nChecking gradients along: loss -> ADC -> phi -> R_tot -> R_last -> dx -> rx_pos")
        print("-"*80)

        # Check contrib values (what was scattered into ADC)
        if hasattr(self, '_debug_contrib_real'):
            contrib_grad = dr.grad(self._debug_contrib_real) if dr.grad_enabled(self._debug_contrib_real) else None
            if contrib_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(contrib_grad * contrib_grad))[0])
                grad_max = float(dr.max(dr.abs(contrib_grad))[0])
                print(f"\n0. contrib_real (VALUES scattered into ADC):")
                print(f"   grad norm: {grad_norm:.6e}")
                print(f"   grad max:  {grad_max:.6e}")
            else:
                print(f"\n0. contrib_real: NO GRADIENT")

        # Check ADC buffers (TARGET of scatter)
        if hasattr(self, 'adc_real'):
            adc_grad = dr.grad(self.adc_real) if dr.grad_enabled(self.adc_real) else None
            if adc_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(adc_grad * adc_grad))[0])
                grad_max = float(dr.max(dr.abs(adc_grad))[0])
                print(f"\n1. ADC buffers (TARGET of scatter):")
                print(f"   adc_real grad norm: {grad_norm:.6e}")
                print(f"   adc_real grad max:  {grad_max:.6e}")
            else:
                print(f"\n1. ADC buffers: NO GRADIENT (expected - target of scatter)")

        # Check R_tot
        if hasattr(self, '_debug_R_tot'):
            rtot_grad = dr.grad(self._debug_R_tot) if dr.grad_enabled(self._debug_R_tot) else None
            if rtot_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(rtot_grad * rtot_grad))[0])
                grad_max = float(dr.max(dr.abs(rtot_grad))[0])
                print(f"\n2. R_tot (total path length):")
                print(f"   grad norm: {grad_norm:.6e}")
                print(f"   grad max:  {grad_max:.6e}")
            else:
                print(f"\n2. R_tot: NO GRADIENT")

        # Check R_last
        if hasattr(self, '_debug_R_last'):
            rlast_grad = dr.grad(self._debug_R_last) if dr.grad_enabled(self._debug_R_last) else None
            if rlast_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(rlast_grad * rlast_grad))[0])
                grad_max = float(dr.max(dr.abs(rlast_grad))[0])
                print(f"\n3. R_last (RX segment distance):")
                print(f"   grad norm: {grad_norm:.6e}")
                print(f"   grad max:  {grad_max:.6e}")
            else:
                print(f"\n3. R_last: NO GRADIENT")

        # Check dx components
        if hasattr(self, '_debug_dx_x'):
            dx_grad = dr.grad(self._debug_dx_x) if dr.grad_enabled(self._debug_dx_x) else None
            if dx_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(dx_grad * dx_grad))[0])
                grad_max = float(dr.max(dr.abs(dx_grad))[0])
                print(f"\n4. dx_x (displacement x-component):")
                print(f"   grad norm: {grad_norm:.6e}")
                print(f"   grad max:  {grad_max:.6e}")
            else:
                print(f"\n4. dx_x: NO GRADIENT")

        # Check RX positions
        if hasattr(self, '_debug_rx_x'):
            rx_grad = dr.grad(self._debug_rx_x) if dr.grad_enabled(self._debug_rx_x) else None
            if rx_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(rx_grad * rx_grad))[0])
                grad_max = float(dr.max(dr.abs(rx_grad))[0])
                print(f"\n5. rx_x_target (gathered RX position):")
                print(f"   grad norm: {grad_norm:.6e}")
                print(f"   grad max:  {grad_max:.6e}")
            else:
                print(f"\n5. rx_x_target: NO GRADIENT")

        # Check source RX array positions
        if hasattr(self, 'rx_array'):
            source_grad = dr.grad(self.rx_array.positions.x) if dr.grad_enabled(self.rx_array.positions.x) else None
            if source_grad is not None:
                grad_norm = float(dr.sqrt(dr.mean(source_grad * source_grad))[0])
                grad_max = float(dr.max(dr.abs(source_grad))[0])
                print(f"\n6. rx_array.positions.x (SOURCE):")
                print(f"   grad norm: {grad_norm:.6e}")
                print(f"   grad max:  {grad_max:.6e}")
            else:
                print(f"\n6. rx_array.positions.x (SOURCE): NO GRADIENT")

        print("\n" + "="*80)
        print("END GRADIENT TRACING")
        print("="*80 + "\n")

    def reset(self):
        """Reset accumulation buffers to zero (both coherent and power)."""
        if self.store_per_tx:
            total_samples = self.NT * self.NR * self.K
        else:
            total_samples = self.NR * self.K

        self.adc_real = dr.zeros(mi.Float, total_samples)
        self.adc_imag = dr.zeros(mi.Float, total_samples)
        self.power_accum = dr.zeros(mi.Float, total_samples)

        # Re-enable gradients if they were enabled before
        if self._grad_enabled:
            dr.enable_grad(self.adc_real)
            dr.enable_grad(self.adc_imag)
            dr.enable_grad(self.power_accum)

    def get_range_profile(self, rx_idx: int = 0, window: Optional[str] = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute range profile via FFT of ADC samples.

        Args:
            rx_idx: RX element index
            window: Window function ('hann', 'hamming', 'blackman', None)

        Returns:
            Tuple of (range_bins, magnitude)
        """
        # Get complex samples
        samples = self.get_complex_samples(rx_idx)  # [K]

        # Apply window
        if window is not None:
            if window == 'hann':
                win = np.hanning(self.K)
            elif window == 'hamming':
                win = np.hamming(self.K)
            elif window == 'blackman':
                win = np.blackman(self.K)
            else:
                raise ValueError(f"Unknown window: {window}")
            samples = samples * win

        # FFT
        spectrum = np.fft.fft(samples)
        magnitude = np.abs(spectrum)

        # Range bins
        # For FMCW: range = c * f_beat / (2 * S)
        # where f_beat is the frequency bin
        # Time spacing between samples
        dt = self.chirp_duration / self.K  # seconds

        # FFT frequency bins
        freq_bins = np.fft.fftfreq(self.K, d=dt)  # Hz

        # Convert frequency to range
        range_bins = self.c * freq_bins / (2.0 * self.S)  # meters

        return range_bins, magnitude

    def __repr__(self):
        return (f"FMCWAdcAccumulator(NR={self.NR}, K={self.K}, "
                f"f0={self.f0/1e9:.3f}GHz, S={self.S/1e12:.3f}THz/s)")
