#!/usr/bin/env python3
"""Create training configuration files for different experiments."""

import argparse
import json
import os
from mmir.data.config_loader import create_default_config


def main():
    parser = argparse.ArgumentParser(description="Create training configuration files")
    parser.add_argument("--output", required=True, help="Output config file path")
    parser.add_argument("--scene_root", help="Scene root directory")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--adc_loss_weight", type=float, default=1.0, help="ADC loss weight")
    parser.add_argument("--ra_loss_weight", type=float, default=1.0, help="RA loss weight")
    parser.add_argument("--ra_loss_rep", type=str, default="log", choices=["log", "linear"], help="RA loss representation")
    parser.add_argument("--ra_clip_percentile", type=float, default=99.0, help="Percentile for RA clipping in linear mode (0 disables)")
    parser.add_argument("--ra_loss_kind", type=str, default="charbonnier", choices=["charbonnier", "huber", "l1"], help="Robust loss for RA")
    parser.add_argument("--loss_balancing", type=str, default="ema", choices=["fixed", "ema", "uncertainty"], help="Multi-loss balancing strategy")
    parser.add_argument("--ema_momentum", type=float, default=0.99, help="EMA momentum for loss normalization")
    parser.add_argument("--uncertainty_init_logvar", type=float, default=-2.0, help="Initial log-variance for uncertainty weighting")
    parser.add_argument("--hits_per_rx", type=int, default=500, help="Hits per RX")
    parser.add_argument("--ra_save_frequency", type=int, default=10, help="RA map save frequency (epochs)")
    # Checkpointing / resume
    parser.add_argument("--run_dir", type=str, default="", help="Run directory to write logs/checkpoints (empty -> auto)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint in run_dir")
    parser.add_argument("--resume_which", type=str, default="latest", choices=["latest", "best"], help="Which checkpoint to resume")
    parser.add_argument("--save_best_metric", type=str, default="psnr", choices=["psnr", "ssim", "loss"], help="Metric used to select best checkpoint")
    args = parser.parse_args()
    
    # Create default config
    create_default_config(args.output, args.scene_root)
    
    # Load and modify with command-line args
    with open(args.output, "r") as f:
        config = json.load(f)
    
    if args.epochs != 500:
        config["epochs"] = args.epochs
    if args.lr != 0.1:
        config["lr"] = args.lr
    # Loss weights and stabilization
    if args.adc_loss_weight != 1.0:
        config["adc_loss_weight"] = args.adc_loss_weight
    if args.ra_loss_weight != 1.0:
        config["ra_loss_weight"] = args.ra_loss_weight
    if args.ra_loss_rep != "log":
        config["ra_loss_rep"] = args.ra_loss_rep
    if args.ra_clip_percentile != 99.0:
        config["ra_clip_percentile"] = args.ra_clip_percentile
    if args.ra_loss_kind != "charbonnier":
        config["ra_loss_kind"] = args.ra_loss_kind
    if args.loss_balancing != "ema":
        config["loss_balancing"] = args.loss_balancing
    if args.ema_momentum != 0.99:
        config["ema_momentum"] = args.ema_momentum
    if args.uncertainty_init_logvar != -2.0:
        config["uncertainty_init_logvar"] = args.uncertainty_init_logvar
    if args.hits_per_rx != 500:
        config["hits_per_rx"] = args.hits_per_rx
    if args.ra_save_frequency != 10:
        config["ra_save_frequency"] = args.ra_save_frequency
    # Checkpointing / resume values (write even if default provided by user)
    if args.run_dir:
        config["run_dir"] = args.run_dir
    if args.resume:
        config["resume"] = True
    if args.resume_which != "latest":
        config["resume_which"] = args.resume_which
    if args.save_best_metric != "psnr":
        config["save_best_metric"] = args.save_best_metric
    
    # Save modified config
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created training config: {args.output}")
    print(f"Scene root: {config['scene_root']}")
    print(
        f"Epochs: {config['epochs']}, LR: {config['lr']}, hits_per_rx: {config['hits_per_rx']}\n"
        f"ADC_w: {config.get('adc_loss_weight',1.0)}, RA_w: {config.get('ra_loss_weight',1.0)}, rep: {config.get('ra_loss_rep','log')}, clip_p: {config.get('ra_clip_percentile',0.0)}, kind: {config.get('ra_loss_kind','charbonnier')}\n"
        f"balancing: {config.get('loss_balancing','fixed')}, ema_m: {config.get('ema_momentum',0.99)}, ra_save_freq: {config['ra_save_frequency']}\n"
        f"run_dir: {config.get('run_dir','(auto)')}, resume: {config.get('resume', False)}, resume_which: {config.get('resume_which','latest')}, best_metric: {config.get('save_best_metric','psnr')}"
    )


if __name__ == "__main__":
    main()
