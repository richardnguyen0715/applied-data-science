"""Main entry point for the project."""

import argparse
from pathlib import Path

import torch

from src.pipeline.run_baseline import run_baseline_pipeline
from src.pipeline.run_diffusion_pipeline import run_diffusion_pipeline
from src.utils.config import DiffusionConfig, ClassifierConfig, DataConfig, TrainingConfig
from src.utils.seed import set_seed


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diffusion-based oversampling for imbalanced CIFAR-10-LT"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["baseline", "diffusion", "both"],
        default="both",
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--diff-epochs",
        type=int,
        default=100,
        help="Number of diffusion training epochs.",
    )
    parser.add_argument(
        "--clf-epochs",
        type=int,
        default=200,
        help="Number of classifier training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=500,
        help="Number of synthetic samples to generate per class.",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create configurations
    training_config = TrainingConfig(
        seed=args.seed,
        device=args.device,
    )

    data_config = DataConfig(
        dataset_name="tomas-gajarsky/cifar10-lt",
        dataset_config="r-20",
        num_classes=10,
        image_size=32,
        num_workers=4,
        pin_memory=True,
    )

    diffusion_config = DiffusionConfig(
        num_epochs=args.diff_epochs,
        batch_size=args.batch_size,
        num_samples_per_class=args.num_samples_per_class,
    )

    classifier_config = ClassifierConfig(
        num_epochs=args.clf_epochs,
        batch_size=args.batch_size,
        model_name="resnet18",
    )

    # Run pipelines
    if args.pipeline in ["baseline", "both"]:
        print("\n" + "=" * 60)
        print("Running baseline pipeline (no balancing)...")
        print("=" * 60)
        baseline_metrics = run_baseline_pipeline(
            output_dir=args.output_dir,
            data_config=data_config,
            classifier_config=classifier_config,
            training_config=training_config,
            device=torch.device(args.device),
        )
        print("\nBaseline Results:")
        for metric, value in baseline_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if args.pipeline in ["diffusion", "both"]:
        print("\n" + "=" * 60)
        print("Running diffusion-based pipeline...")
        print("=" * 60)
        diffusion_metrics = run_diffusion_pipeline(
            output_dir=args.output_dir,
            diffusion_config=diffusion_config,
            classifier_config=classifier_config,
            data_config=data_config,
            training_config=training_config,
            device=torch.device(args.device),
        )
        print("\nDiffusion Results:")
        for metric, value in diffusion_metrics.items():
            print(f"  {metric}: {value:.4f}")

    if args.pipeline == "both":
        print("\n" + "=" * 60)
        print("Comparison: Diffusion vs. Baseline")
        print("=" * 60)
        print(f"Accuracy improvement: {(diffusion_metrics['accuracy'] - baseline_metrics['accuracy']) * 100:.2f}%")
        print(f"Macro F1 improvement: {(diffusion_metrics['macro_f1'] - baseline_metrics['macro_f1']) * 100:.2f}%")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {args.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
