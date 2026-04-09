"""Evaluation adapter — drop-in for mmir.evaluation.renderer_wrapper.RendererWrapper.

Provides render_forward() returning ADC in mmIR format:
  (N_tx, N_rx, K, 2) float32 [real, imag].
"""

import numpy as np
import torch

from .config import RadarConfig
from .gaussian_model import GaussianModel
from .adc_synthesis import synthesize_adc_single_bounce
from .culling import compute_contribution_estimates, select_active_set
from .antenna_torch import load_patterns


class GaussianRendererWrapper:
    """Renders ADC from a trained mm25DGS model.

    Interface matches RendererWrapper so existing evaluation scripts
    can be used without modification.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        tx_pattern_path: str = "assets/antenna_pattern/MMWCAS/tx1_76.npy",
        rx_pattern_path: str = "assets/antenna_pattern/MMWCAS/rx1_76.npy",
        device: str = "cuda:0",
        shading_tier: int = 1,
    ):
        self.device = device
        self.shading_tier = shading_tier
        self.radar_cfg = RadarConfig.from_json(config_path)
        load_patterns(tx_pattern_path, rx_pattern_path)

        # Load trained model
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        N = ckpt["positions"].shape[0]
        self.model = GaussianModel(N, device=device)
        self.model.load(model_path)
        self.model.eval()

        self._update_radar_geometry()

    def _update_radar_geometry(self):
        self.radar_center = torch.from_numpy(
            (self.radar_cfg.tx_positions_m.mean(0)
             + self.radar_cfg.rx_positions_m.mean(0)) / 2
        ).float().to(self.device)
        self.radar_boresight = torch.nn.functional.normalize(
            torch.from_numpy(self.radar_cfg.tx_boresights.mean(0)).float().to(self.device),
            dim=0,
        )

    def render_forward(self, seed: int = 42) -> np.ndarray:
        """Render ADC.  Returns (N_tx, N_rx, K, 2) float32."""
        torch.manual_seed(seed)
        with torch.no_grad():
            contributions = compute_contribution_estimates(
                self.model, self.radar_center, self.radar_boresight
            )
            active_mask = select_active_set(contributions, threshold=0.99)
            adc_real, adc_imag = synthesize_adc_single_bounce(
                self.model, active_mask, self.radar_cfg, detach_phase=True,
                shading_tier=self.shading_tier,
            )
        adc_ri = torch.stack([adc_real, adc_imag], dim=-1)
        return adc_ri.cpu().numpy().astype(np.float32)

    def update_antenna_config(self, config_path: str):
        """Swap radar config (for cross-sensor transfer evaluation)."""
        self.radar_cfg = RadarConfig.from_json(config_path)
        self._update_radar_geometry()
