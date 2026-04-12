#!/usr/bin/env python3
"""Main entry point for running VAE-based oversampling pipeline."""

import argparse
from pathlib import Path

from src.pipeline.run_vae_pipeline import run_vae_pipeline
from src.utils.config import PipelineConfig, load_config


def main():
    """Run VAE-based oversampling pipeline."""
    parser = argparse.ArgumentParser(
        description="Run VAE-based oversampling on CIFAR-10-LT",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vae.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    if args.config.exists():
        config = load_config(args.config)
    else:
        config = PipelineConfig()

    # Override output dir if provided
    if args.output_dir:
        config.output_dir = args.output_dir

    # Run pipeline
    results = run_vae_pipeline(config)

    print("\n" + "=" * 50)
    print("VAE-BASED OVERSAMPLING RESULTS")
    print("=" * 50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1 Score: {results['macro_f1']:.4f}")
    print(f"Synthetic Samples Generated: {results['num_synthetic_samples']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
