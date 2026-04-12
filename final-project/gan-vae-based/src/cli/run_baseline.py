#!/usr/bin/env python3
"""Main entry point for running baseline pipeline."""

import argparse
from pathlib import Path

from src.pipeline.run_baseline import run_baseline_pipeline
from src.utils.config import PipelineConfig, load_config


def main():
    """Run baseline pipeline."""
    parser = argparse.ArgumentParser(
        description="Run baseline CIFAR-10-LT classifier (no oversampling)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
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
    results = run_baseline_pipeline(config)

    print("\n" + "=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1 Score: {results['macro_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
