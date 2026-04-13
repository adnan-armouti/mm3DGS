"""Material parameter diagnostics for the v4 ablation suite.

Captures, per training run:
  - raw_materials at init           (M, 6)
  - raw_materials checkpoints       every `checkpoint_every` iters → (T, M, 6)
  - raw_materials at end             (M, 6)
  - per-column drift                 (6,)        D1
  - per-column Fisher diagonal      (6,)        D2 — extra forward+backward at end

Outputs a single .npz per run for offline D1/D2/D3/D4 analysis.

The Fisher diagonal is computed only for runs where raw_materials is in the
optimizer (i.e. LEARN_MATERIALS=True and not all 6 columns frozen). For runs
that train no material columns at all (e.g. fixed-concrete baseline), the
drift / Fisher arrays are reported as zeros.
"""

import os
import numpy as np
import torch


class MaterialDiagnostics:
    def __init__(self, raw_materials, checkpoint_every=50, enabled=True,
                 capture_grad_stats=False):
        """Capture init state. Call once before the training loop.

        If `capture_grad_stats=True`, also records per-iter gradient
        mean and std over points (D2 — gauge-variant decomposition).
        """
        self.enabled = enabled
        self.checkpoint_every = checkpoint_every
        self.capture_grad_stats = capture_grad_stats
        if not enabled:
            return
        # (M, 6) snapshot — copy to CPU to avoid holding extra GPU memory
        self.init = raw_materials.detach().cpu().clone()
        self.checkpoints = [self.init]   # T grows over training
        self.checkpoint_iters = [0]
        self.final = None
        # D2: per-iter gradient statistics
        # grad_mean[t, k] = mean over M of raw_materials.grad[:, k] at iter t
        # grad_std[t, k]  = std  over M of raw_materials.grad[:, k] at iter t
        self.grad_mean = []
        self.grad_std = []
        self.grad_iters = []

    def maybe_checkpoint(self, raw_materials, it):
        if not self.enabled:
            return
        if (it + 1) % self.checkpoint_every == 0:
            self.checkpoints.append(raw_materials.detach().cpu().clone())
            self.checkpoint_iters.append(it + 1)

    def record_grad(self, raw_materials, it):
        """D2: record per-column grad mean and std across points. Call
        AFTER loss.backward() and BEFORE gradient clipping / zeroing.
        """
        if not self.enabled or not self.capture_grad_stats:
            return
        if raw_materials.grad is None:
            return
        g = raw_materials.grad.detach()
        self.grad_mean.append(g.mean(dim=0).cpu().numpy())
        self.grad_std.append(g.std(dim=0).cpu().numpy())
        self.grad_iters.append(it)

    def finalize(self, raw_materials):
        if not self.enabled:
            return
        self.final = raw_materials.detach().cpu().clone()
        # Make sure the last state is in the checkpoint list
        if self.checkpoint_iters[-1] != self.final.shape[0] and \
           not torch.equal(self.checkpoints[-1], self.final):
            self.checkpoints.append(self.final)

    def compute_drift(self):
        """D1: per-column L2 drift, normalized by sqrt(M)."""
        if not self.enabled or self.final is None:
            return np.zeros(6, dtype=np.float32)
        delta = (self.final - self.init).numpy()
        M = delta.shape[0]
        return np.sqrt((delta ** 2).sum(axis=0) / M).astype(np.float32)

    def compute_fisher(self, model, rast, vertex_areas, active_mask,
                        gt_loss_norm_cached, render_gaussians_fn,
                        compute_ra_loss_rp_fn):
        """D2: per-column Fisher diagonal, summed over points.

        One extra forward + backward at the trained state. Returns shape (6,).
        """
        if not self.enabled:
            return np.zeros(6, dtype=np.float32)
        if not model.raw_materials.requires_grad:
            return np.zeros(6, dtype=np.float32)
        # Clear any leftover grad
        if model.raw_materials.grad is not None:
            model.raw_materials.grad = None
        rp_real, rp_imag = render_gaussians_fn(
            model, rast, vertex_areas=vertex_areas,
            active_mask=active_mask, shadow_mask=None)
        loss, _ = compute_ra_loss_rp_fn(rp_real, rp_imag, gt_loss_norm_cached)
        loss.backward()
        g = model.raw_materials.grad.detach().cpu().numpy()
        fisher = (g ** 2).sum(axis=0).astype(np.float32)
        # Clear so the next iter (if any) starts clean
        model.raw_materials.grad = None
        return fisher

    def save(self, output_dir, run_name, mean_cart_corr, per_scene_corr,
             ms_per_iter, drift, fisher, extra=None):
        """Save the .npz dump for this run."""
        if not self.enabled:
            return
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{run_name}.npz')
        # Stack checkpoints into (T, M, 6)
        if len(self.checkpoints) > 1:
            traj = torch.stack(self.checkpoints, dim=0).numpy()
        else:
            traj = self.init.unsqueeze(0).numpy()
        payload = {
            'run_name': run_name,
            'init': self.init.numpy(),
            'final': self.final.numpy() if self.final is not None else self.init.numpy(),
            'trajectory': traj,
            'checkpoint_iters': np.array(self.checkpoint_iters, dtype=np.int32),
            'mean_cart_corr': float(mean_cart_corr),
            'per_scene_cart_corr': np.array(per_scene_corr, dtype=np.float32),
            'ms_per_iter': float(ms_per_iter),
            'drift': drift.astype(np.float32),
            'fisher': fisher.astype(np.float32),
        }
        if extra:
            payload.update(extra)
        if self.capture_grad_stats and len(self.grad_mean) > 0:
            payload['grad_mean'] = np.stack(self.grad_mean, axis=0).astype(np.float32)  # (T, 6)
            payload['grad_std'] = np.stack(self.grad_std, axis=0).astype(np.float32)   # (T, 6)
            payload['grad_iters'] = np.array(self.grad_iters, dtype=np.int32)
        np.savez_compressed(path, **payload)
        return path


PARAM_NAMES = ['eps_real', 'eps_imag', 'sigma_h', 'l_c', 'tau_base', 'thickness']
