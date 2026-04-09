"""Configuration dataclasses and radar parameter extraction."""

from dataclasses import dataclass, field
from typing import List
import json
import numpy as np

C = 299_792_458.0  # speed of light (m/s)


@dataclass
class RadarConfig:
    """Hardware parameters extracted from a single JSON config file.

    Parses the same JSON format as mmir/sensor/config.py:FMCWConfig.from_json.
    """

    center_freq: float          # Hz, ~77e9
    chirp_slope: float          # Hz/s, ~79e12
    sample_rate: float          # Hz, ~8e6
    num_adc_samples: int        # 256
    chirp_duration: float       # seconds
    adc_start_time: float       # seconds

    tx_positions_mm: np.ndarray  # (N_tx, 3)
    tx_boresights: np.ndarray    # (N_tx, 3) unit vectors
    rx_positions_mm: np.ndarray  # (N_rx, 3)
    rx_boresights: np.ndarray    # (N_rx, 3) unit vectors

    n_tx: int
    n_rx: int

    @property
    def wavelength(self) -> float:
        return C / self.center_freq

    @property
    def bandwidth(self) -> float:
        return self.chirp_slope * self.chirp_duration

    @property
    def range_resolution(self) -> float:
        return C / (2.0 * self.bandwidth)

    @property
    def tx_positions_m(self) -> np.ndarray:
        return self.tx_positions_mm * 1e-3

    @property
    def rx_positions_m(self) -> np.ndarray:
        return self.rx_positions_mm * 1e-3

    @classmethod
    def from_json(cls, path: str) -> "RadarConfig":
        """Parse an aligned config JSON.

        Expected keys (matching mmIR convention):
          carrierFrequency, freqSlope, sampleRate, numAdcSamples,
          adcStartTime, rampEndTime, tx_array[], rx_array[].
        """
        with open(path) as f:
            cfg = json.load(f)

        tx_arr = cfg["tx_array"]
        rx_arr = cfg["rx_array"]

        tx_pos = np.array([t["pos_mm"] for t in tx_arr], dtype=np.float64)
        tx_bore = np.array([t["boresight"] for t in tx_arr], dtype=np.float64)
        rx_pos = np.array([r["pos_mm"] for r in rx_arr], dtype=np.float64)
        rx_bore = np.array([r["boresight"] for r in rx_arr], dtype=np.float64)

        adc_start = cfg.get("adcStartTime", cfg.get("adc_start_time", 2e-6))
        ramp_end = cfg.get("rampEndTime", cfg.get("ramp_end_time", 3.4e-5))

        return cls(
            center_freq=cfg["carrierFrequency"],
            chirp_slope=cfg["freqSlope"],
            sample_rate=cfg["sampleRate"],
            num_adc_samples=cfg["numAdcSamples"],
            chirp_duration=ramp_end - adc_start,
            adc_start_time=adc_start,
            tx_positions_mm=tx_pos,
            tx_boresights=tx_bore,
            rx_positions_mm=rx_pos,
            rx_boresights=rx_bore,
            n_tx=len(tx_arr),
            n_rx=len(rx_arr),
        )


@dataclass
class TrainingConfig:
    """Hyperparameters for the mm25DGS training loop."""

    # --- Data paths ---
    scene_dir: str = ""
    config_path: str = ""
    output_dir: str = ""

    # --- Antenna patterns ---
    tx_pattern_path: str = "assets/antenna_pattern/MMWCAS/tx1_76.npy"
    rx_pattern_path: str = "assets/antenna_pattern/MMWCAS/rx1_76.npy"

    # --- Gaussian init ---
    pcl_path: str = ""
    mesh_path: str = ""  # Optional: .ply mesh for accurate normal init
    init_checkpoint: str = ""  # Optional: .pt checkpoint to resume from
    target_n_gaussians: int = 100_000  # ~5 KB/Gaussian with gradient checkpointing
    pca_k_neighbors: int = 20
    initial_scale_clamp_min: float = 0.01  # metres
    initial_scale_clamp_max: float = 0.50
    initial_material: str = "concrete"

    # --- Culling ---
    culling_threshold: float = 0.97
    culling_full_inclusion_interval: int = 10

    # --- Training ---
    max_iterations: int = 500
    seed: int = 42

    # Per-group learning rates
    lr_positions: float = 1.6e-4
    lr_positions_final: float = 1.6e-6
    lr_rotations: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_materials: float = 0.5

    # Per-group gradient clipping (RMS)
    clip_positions: float = 1.0
    clip_rotations: float = 0.5
    clip_scales: float = 1.0
    clip_opacities: float = 1.0
    clip_materials: float = 1.0

    # Material sub-LR scales (per column of the 6-param vector)
    # mmIR does NOT use per-column scaling — all columns get same LR.
    material_lr_scales: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )

    # --- Density control ---
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.01
    opacity_reset_interval: int = 500
    prune_range_min: float = 1.5  # metres
    prune_range_max: float = 30.0

    # --- Shading ---
    shading_tier: int = 1  # 1=Fresnel, 2=Jones KA+SPM (set to 2 for full model)
    enable_coherence_gamma: bool = False  # enable γ coherence factor

    # --- Coherence (σ_em) ---
    lr_sigma_em: float = 1e-3
    clip_sigma_em: float = 1.0

    # --- Loss ---
    ra_mag_weight: float = 1.0
    adc_mag_weight: float = 0.0
    phase_weight: float = 0.0
    ra_use_log: bool = False  # paper uses linear magnitude (no log)
    log_epsilon: float = 1e-6

    # --- Multi-bounce ---
    enable_multibounce: bool = False
    multibounce_warmup: int = 100
    multibounce_interaction_radius: float = 2.0

    # --- Misc ---
    log_interval: int = 10
    checkpoint_interval: int = 50
    device: str = "cuda:0"
