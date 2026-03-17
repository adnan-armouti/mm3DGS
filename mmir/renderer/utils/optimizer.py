"""
DrJit-native Adam optimizer.

Keeps all parameters and optimizer state (moment estimates) as DrJit arrays
on GPU. No NumPy round-trips that would break the AD graph or cause
unnecessary GPU<->CPU data transfers.

Usage:
    optimizer = DrJitAdam(
        param_groups={
            'materials': raw_params_drjit,     # list of 6 mi.Float arrays
            'pose_rotation': [pitch, roll, yaw],
            'pose_translation': [tx, ty, tz],
            'normals': [nx, ny, nz],
            'vertex_positions': [dx, dy, dz],
        },
        lr_dict={
            'materials': 0.5,
            'pose_rotation': 5e-3,
            'pose_translation': 5e-4,
            'normals': 0.01,
            'vertex_positions': 5e-4,
        },
    )

    # In training loop:
    optimizer.zero_grad()
    # ... forward, loss, backward ...
    optimizer.step()
"""

import drjit as dr
import mitsuba as mi


class DrJitAdam:
    """
    Adam optimizer that keeps all state in DrJit arrays on GPU.

    Key properties:
    - Parameters remain as mi.Float throughout (no numpy conversion)
    - Moment estimates (m, v) stored as DrJit arrays on GPU
    - Gradient clipping and NaN sanitization done in DrJit
    - Single dr.eval() at end of step() to materialize all updates
    """

    def __init__(
        self,
        param_groups: dict,
        lr_dict: dict,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        clip_dict: dict = None,
    ):
        """
        Args:
            param_groups: {name: [mi.Float, ...]} -- each group is a list of
                grad-enabled mi.Float parameter arrays.
            lr_dict: {name: float} -- learning rate per group.
            beta1: First moment decay rate.
            beta2: Second moment decay rate.
            eps: Numerical epsilon for denominator.
            clip_dict: Optional {name: float} -- per-group gradient RMS clip threshold.
        """
        self.param_groups = param_groups
        self.lr_dict = lr_dict
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_dict = clip_dict or {}
        self.t = 0

        # Initialize first and second moment estimates as zero DrJit arrays
        self.m = {}
        self.v = {}
        for name, params in param_groups.items():
            self.m[name] = [dr.zeros(mi.Float, dr.width(p)) for p in params]
            self.v[name] = [dr.zeros(mi.Float, dr.width(p)) for p in params]

    def step(self, frozen_params: set = None):
        """
        Update all non-frozen parameters using Adam.

        Args:
            frozen_params: Set of group names to skip this step.
        """
        self.t += 1
        frozen = frozen_params or set()

        # Bias correction factors (Python scalars -- no GPU ops)
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for name, params in self.param_groups.items():
            if name in frozen:
                continue

            lr = self.lr_dict.get(name, 1e-4)
            clip_thresh = self.clip_dict.get(name, 0.0)

            for i, p in enumerate(params):
                if not dr.grad_enabled(p):
                    continue

                g = dr.grad(p)

                # NaN/Inf sanitization (stays on GPU)
                g = dr.select(dr.isnan(g) | dr.isinf(g), mi.Float(0.0), g)

                # Per-group RMS gradient clipping (stays on GPU)
                if clip_thresh > 0:
                    # RMS norm: sqrt(mean(g^2))
                    n_elem = dr.width(g)
                    if n_elem > 0:
                        g_rms = dr.sqrt(dr.mean(g * g))
                        # Scale down if exceeds threshold
                        scale = dr.minimum(
                            mi.Float(clip_thresh) / dr.maximum(g_rms, mi.Float(1e-10)),
                            mi.Float(1.0),
                        )
                        g = g * scale

                # Adam moment updates (all DrJit operations, stays on GPU)
                self.m[name][i] = mi.Float(self.beta1) * self.m[name][i] + mi.Float(1.0 - self.beta1) * g
                self.v[name][i] = mi.Float(self.beta2) * self.v[name][i] + mi.Float(1.0 - self.beta2) * g * g

                # Bias-corrected estimates
                m_hat = self.m[name][i] / mi.Float(bc1)
                v_hat = self.v[name][i] / mi.Float(bc2)

                # Compute update
                update = mi.Float(lr) * m_hat / (dr.sqrt(v_hat) + mi.Float(self.eps))

                # Apply update -- detach and re-enable grad for next iteration
                new_p = mi.Float(p - update)
                dr.disable_grad(new_p)
                dr.enable_grad(new_p)

                # Replace parameter in the group
                self.param_groups[name][i] = new_p

        # Single eval to materialize all updated parameters
        all_params = [p for params in self.param_groups.values() for p in params]
        if all_params:
            dr.eval(*all_params)

    def zero_grad(self):
        """Zero all gradients on all parameters."""
        for params in self.param_groups.values():
            for p in params:
                if dr.grad_enabled(p):
                    dr.set_grad(p, mi.Float(0.0))

    def get_grad_norms(self) -> dict:
        """
        Get gradient L2 norms for each parameter group (for logging).

        Returns dict {name: float} with the L2 norm of the gradient for
        each group. Only reads small scalars from GPU.
        """
        norms = {}
        for name, params in self.param_groups.items():
            total_sq = 0.0
            for p in params:
                if dr.grad_enabled(p):
                    g = dr.grad(p)
                    # Sum of squares -- reduce to scalar on GPU, then read
                    total_sq += float(dr.sum(g * g)[0])
            norms[name] = total_sq ** 0.5
        return norms

    def state_dict(self) -> dict:
        """Export optimizer state for checkpointing (moves to CPU/numpy)."""
        import numpy as np
        state = {'t': self.t, 'lr_dict': self.lr_dict, 'm': {}, 'v': {}}
        for name in self.param_groups:
            state['m'][name] = [np.array(m) for m in self.m[name]]
            state['v'][name] = [np.array(v) for v in self.v[name]]
        return state

    def load_state_dict(self, state: dict):
        """Load optimizer state from checkpoint."""
        self.t = state['t']
        for name in self.param_groups:
            if name in state['m']:
                for i, m_np in enumerate(state['m'][name]):
                    self.m[name][i] = mi.Float(m_np)
                for i, v_np in enumerate(state['v'][name]):
                    self.v[name][i] = mi.Float(v_np)
