#!/usr/bin/env python
"""Run imbalance handling comparison pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.cifar import create_cifar10_dataloaders
from src.utils.comparison import ImbalanceComparisonAnalyzer, create_comparison_plots
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main() -> None:
    """Main entry point for comparison pipeline."""
    parser = argparse.ArgumentParser(description="Compare imbalance handling results.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config.yaml",
        help="Path to training config.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/cifar10",
        help="Directory containing training results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results.",
    )

    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    logger = setup_logger(
        name="comparison_pipeline",
        log_dir=str(output_dir / "logs"),
        log_file="comparison.log",
    )
    logger.info("Starting imbalance handling comparison pipeline")
    logger.info(f"Device: {device}")

    # Initialize analyzer
    analyzer = ImbalanceComparisonAnalyzer(config, device, results_dir)

    # Load data
    logger.info("Loading test data...")
    _, test_loader, train_dist = create_cifar10_dataloaders(config)

    # Load training history
    logger.info("Loading training history...")
    history_df, history_loaded = analyzer.load_training_history()
    if history_loaded:
        logger.info(f"✓ Loaded {len(history_df)} epochs of training history")
    else:
        logger.warning("⚠ Training history not found")

    # Load checkpoint and evaluate
    logger.info("Loading best checkpoint...")
    model, checkpoint, checkpoint_loaded = analyzer.load_checkpoint("best.pt")
    if not checkpoint_loaded:
        logger.error("✗ Could not load checkpoint")
        return

    logger.info(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown') + 1}")

    # Evaluate and compare
    logger.info("Evaluating model and generating comparison metrics...")
    comparison_results = analyzer.evaluate_and_compare(model, test_loader, train_dist)

    # Generate report
    report = analyzer.generate_summary_report(history_df, comparison_results, train_dist)
    logger.info(report)

    # Save results
    logger.info(f"Saving analysis results to {output_dir}...")
    analyzer.save_analysis_results(comparison_results, output_dir)

    # Generate plots
    logger.info("Generating comparison plots...")
    create_comparison_plots(
        history_df,
        comparison_results["per_class_accuracy"],
        comparison_results["class_names"],
        comparison_results["class_counts"],
        comparison_results["test_metrics"],
        output_dir,
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("Files generated:")
    logger.info("  - per_class_performance.csv")
    logger.info("  - overall_metrics.csv")
    logger.info("  - training_curves.png")
    logger.info("  - per_class_accuracy.png")
    logger.info("  - imbalance_vs_accuracy.png")


if __name__ == "__main__":
    main()
