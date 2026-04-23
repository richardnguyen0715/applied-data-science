"""Main entry point for contrastive learning pipeline."""

import argparse
from pathlib import Path

import torch

from src.pipeline.run_contrastive import run_contrastive_pipeline
from src.utils.config import (
    load_config_from_yaml,
    Config,
)
from src.utils.seed import set_seed


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Contrastive Learning for Imbalanced Data"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10-lt", "credit-card-fraud"],
        default="cifar10-lt",
        help="Dataset to use.",
    )

    # General arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use.",
    )

    args = parser.parse_args()

    # Load config from YAML if provided, otherwise use defaults
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = Config()
    
    # Override with command-line arguments if provided
    if args.dataset != "cifar10-lt":
        config.data.dataset_name = args.dataset
    if args.seed != 42:
        config.seed = args.seed
    if args.device != "cuda":
        config.device = args.device

    # Set random seed
    set_seed(config.seed)

    # Run pipeline
    metrics = run_contrastive_pipeline(config=config)

    print("\n" + "="*50)
    print("Final Results:")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
