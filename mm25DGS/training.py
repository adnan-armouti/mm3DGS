"""Main training loop for mm25DGS."""

import os
import json
import time
import glob

import numpy as np
import torch

from .config import RadarConfig, TrainingConfig
from .gaussian_model import GaussianModel
from .initialization import initialize_from_lidar
from .reparameterization import reparameterize
from .rae_grid import RAEGrid
from .culling import compute_contribution_estimates, select_active_set
from .adc_synthesis import synthesize_adc_single_bounce
from .multibounce import synthesize_multibounce
from .losses import compute_loss
from .optimizer import PerGroupAdam
from .density_control import densify_and_prune, reset_opacities
from .antenna_torch import load_patterns

# Data loading from mmIR
from mmir.data.data_utils import load_target


def train(cfg: TrainingConfig):
    """Main training entry point.

    Returns:
        (model, history) — trained GaussianModel and dict of per-iteration metrics.
    """
    device = cfg.device
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load radar config ---
    radar_cfg = RadarConfig.from_json(cfg.config_path)

    # --- Load antenna patterns (once) ---
    load_patterns(cfg.tx_pattern_path, cfg.rx_pattern_path)

    # --- Load ground truth ADC ---
    gt_adc_path = _find_gt_adc(cfg.scene_dir, "cascaded")
    gt_adc_ri, _ = load_target(gt_adc_path, device, normalize=True)
    # gt_adc_ri: (N_tx, N_rx, K, 2)

    # --- Initialise Gaussians ---
    if cfg.init_checkpoint:
        # Resume from checkpoint
        import torch as _torch
        ckpt = _torch.load(cfg.init_checkpoint, map_location=device, weights_only=True)
        N = ckpt["positions"].shape[0]
        model = GaussianModel(N, device=device)
        model.load(cfg.init_checkpoint)
        print(f"Resumed {model.N} Gaussians from {cfg.init_checkpoint}")
    else:
        model = initialize_from_lidar(
            cfg.pcl_path,
            radar_cfg,
            device,
            target_n_gaussians=cfg.target_n_gaussians,
            k_neighbors=cfg.pca_k_neighbors,
            scale_clamp_min=cfg.initial_scale_clamp_min,
            scale_clamp_max=cfg.initial_scale_clamp_max,
            initial_material=cfg.initial_material,
            mesh_path=cfg.mesh_path if cfg.mesh_path else None,
        )
        print(f"Initialised {model.N} Gaussians from LiDAR")

    # --- RAE grid (for diagnostics) ---
    rae_grid = RAEGrid(radar_cfg, device=device)

    # --- Optimizer ---
    optimizer = PerGroupAdam(model, cfg)
    optimizer.setup()

    # --- Radar geometry ---
    radar_center = torch.from_numpy(
        (radar_cfg.tx_positions_m.mean(0) + radar_cfg.rx_positions_m.mean(0)) / 2
    ).float().to(device)
    radar_boresight = torch.nn.functional.normalize(
        torch.from_numpy(radar_cfg.tx_boresights.mean(0)).float().to(device),
        dim=0,
    )

    # --- History ---
    history = {
        "iteration": [], "total_loss": [], "ra_mag_loss": [],
        "n_active": [], "n_total": [],
    }
    best_loss = float("inf")
    best_state = None

    # --- Gradient accumulation for density control ---
    grad_accum = torch.zeros(model.N, device=device)
    grad_count = torch.zeros(model.N, device=device)

    # --- Training loop ---
    t_start = time.time()

    for it in range(cfg.max_iterations):
        optimizer.zero_grad()

        # LR schedule: warmup (0.3->1 over 5 iters) then CONSTANT
        # Matches mmIR's schedule (no decay). The original exponential decay
        # killed material optimization by reducing LR to 1% by iteration 500.
        if it < 5:
            lr_scale = 0.3 + 0.7 * (it / 5)
        else:
            lr_scale = 1.0

        # --- Contribution culling ---
        with torch.no_grad():
            contributions = compute_contribution_estimates(
                model, radar_center, radar_boresight
            )
            if it % cfg.culling_full_inclusion_interval == 0:
                active_mask = torch.ones(model.N, dtype=torch.bool, device=device)
            else:
                active_mask = select_active_set(contributions, cfg.culling_threshold)

        n_active = active_mask.sum().item()

        # --- Forward: single-bounce ADC ---
        adc_real, adc_imag = synthesize_adc_single_bounce(
            model, active_mask, radar_cfg,
            detach_phase=True,
            enable_gamma=cfg.enable_coherence_gamma,
            shading_tier=cfg.shading_tier,
        )

        # --- Forward: multi-bounce (optional) ---
        if cfg.enable_multibounce and it >= cfg.multibounce_warmup:
            adc_mb_r, adc_mb_i = synthesize_multibounce(
                model, radar_cfg,
                r_max=cfg.multibounce_interaction_radius,
                detach_phase=True,
            )
            adc_real = adc_real + adc_mb_r
            adc_imag = adc_imag + adc_mb_i

        # --- Loss ---
        loss, loss_dict = compute_loss(
            adc_real, adc_imag, gt_adc_ri,
            w_ra_mag=cfg.ra_mag_weight,
            w_adc_mag=cfg.adc_mag_weight,
            w_phase=cfg.phase_weight,
            ra_use_log=cfg.ra_use_log,
            log_epsilon=cfg.log_epsilon,
        )

        # --- Backward ---
        loss.backward()

        # --- Accumulate position gradients for density control ---
        if model.positions.grad is not None:
            pos_grad_mag = model.positions.grad.detach().norm(dim=-1)
            n_cur = min(pos_grad_mag.shape[0], grad_accum.shape[0])
            grad_accum[:n_cur] += pos_grad_mag[:n_cur]
            grad_count[:n_cur] += 1

        # --- Optimizer step ---
        optimizer.step(lr_scale=lr_scale)

        # --- Post-step: clamp material raw params ---
        with torch.no_grad():
            model.raw_materials[:, 1].clamp_(-7.0, 4.6)
            model.raw_materials[:, 2].clamp_(-16.0, -2.3)
            model.raw_materials[:, 5].clamp_(-7.0, -0.7)

        # --- Density control ---
        if (
            (it + 1) % cfg.densify_interval == 0
            and it < cfg.max_iterations - 100
        ):
            model = densify_and_prune(
                model,
                grad_accum[: model.N],
                grad_count[: model.N],
                cfg,
                radar_center,
            )
            optimizer = PerGroupAdam(model, cfg)
            optimizer.setup()
            grad_accum = torch.zeros(model.N, device=device)
            grad_count = torch.zeros(model.N, device=device)

        if (it + 1) % cfg.opacity_reset_interval == 0:
            reset_opacities(model)

        # --- Logging ---
        history["iteration"].append(it)
        history["total_loss"].append(loss_dict["total"])
        history["ra_mag_loss"].append(loss_dict.get("ra_mag", 0))
        history["n_active"].append(n_active)
        history["n_total"].append(model.N)

        if loss_dict["total"] < best_loss:
            best_loss = loss_dict["total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (it + 1) % cfg.log_interval == 0:
            elapsed = time.time() - t_start
            print(
                f"[{it + 1}/{cfg.max_iterations}] "
                f"loss={loss_dict['total']:.6f} "
                f"ra={loss_dict.get('ra_mag', 0):.6f} "
                f"active={n_active}/{model.N} "
                f"time={elapsed:.1f}s"
            )

        # --- Checkpoint ---
        if (it + 1) % cfg.checkpoint_interval == 0:
            model.save(os.path.join(cfg.output_dir, f"checkpoint_{it + 1}.pt"))

    # --- Save best model ---
    if best_state is not None:
        best_path = os.path.join(cfg.output_dir, "best_model.pt")
        torch.save(best_state, best_path)
        print(f"Best model saved to {best_path} (loss={best_loss:.6f})")

    with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # Save materials in mmIR-compatible format for eval
    if best_state is not None:
        model.load_state_dict(best_state)
        physics_mat = reparameterize(model.raw_materials).detach().cpu().numpy()
        np.savez(os.path.join(cfg.output_dir, "best_materials.npz"),
                 materials=physics_mat)

    return model, history


def _find_gt_adc(scene_dir: str, sensor: str) -> str:
    """Find the GT ADC .npy matching the scene's frame number.

    Scene directories are named 'seq_X_frame_Y', so the correct file
    is '{sensor}_frame_Y.npy'.  Falls back to first file if no match.
    """
    import re
    scene_name = os.path.basename(scene_dir)
    m = re.search(r"frame_(\d+)", scene_name)
    if m:
        frame_num = m.group(1)
        exact = os.path.join(scene_dir, "radar", f"{sensor}_frame_{frame_num}.npy")
        if os.path.exists(exact):
            return exact

    # Fallback
    pattern = os.path.join(scene_dir, "radar", f"{sensor}_frame_*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No GT ADC files matching {pattern}")
    return files[0]
