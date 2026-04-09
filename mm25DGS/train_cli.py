"""CLI entry point for mm25DGS training.

Usage:
    python -m mm25DGS.train_cli \\
        --scene_dir data/seq_0_frame_135 \\
        --config_path data/seq_0_frame_135/configs/cascaded_frame_135_aligned_gpu.json \\
        --pcl_path data/seq_0_frame_135/scene/pcl.npy \\
        --output_dir output/mm25dgs/seq_0_frame_135

    # Or with a JSON config file:
    python -m mm25DGS.train_cli --config path/to/training_config.json
"""

import argparse
import json

from .config import TrainingConfig
from .training import train


def main():
    parser = argparse.ArgumentParser(description="mm25DGS Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON training config")
    parser.add_argument("--scene_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--pcl_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = json.load(f)
        cfg = TrainingConfig(**cfg_dict)
    else:
        cfg = TrainingConfig()

    # Override from CLI args
    for key in ["scene_dir", "config_path", "pcl_path",
                "output_dir", "max_iterations", "device"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)

    model, history = train(cfg)
    print("Training complete.")


if __name__ == "__main__":
    main()
