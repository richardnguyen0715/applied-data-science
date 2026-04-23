"""
Example script showing how to use the contrastive learning pipeline programmatically.

This script demonstrates:
1. How to configure the pipeline
2. How to run training with custom parameters
3. How to access results
"""

from pathlib import Path

import torch

from src.pipeline.run_contrastive import run_contrastive_pipeline
from src.utils.config import (
    AugmentationConfig,
    ClassifierConfig,
    ClassifierTrainingConfig,
    Config,
    ContrastiveLossConfig,
    ContrastiveTrainingConfig,
    DataConfig,
    EncoderConfig,
    EvaluationConfig,
)
from src.utils.seed import set_seed


def example_cifar10lt_basic():
    """Example: Basic CIFAR-10-LT training."""
    print("\n" + "="*50)
    print("Example 1: Basic CIFAR-10-LT Training")
    print("="*50)

    # Create configuration
    config = Config(
        data=DataConfig(
            dataset_name="cifar10-lt",
            cifar10_config="r-100",
            num_workers=4,
        ),
        encoder=EncoderConfig(
            architecture="resnet18",
            projection_dim=128,
        ),
        contrastive_training=ContrastiveTrainingConfig(
            num_epochs=200,
            batch_size=512,
            learning_rate=0.5,
        ),
        classifier_training=ClassifierTrainingConfig(
            num_epochs=100,
            learning_rate=0.001,
        ),
        seed=42,
        output_dir=Path("outputs/example_basic"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run pipeline
    metrics = run_contrastive_pipeline(config=config)

    print("\nFinal Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def example_cifar10lt_large():
    """Example: Large model training for better performance."""
    print("\n" + "="*50)
    print("Example 2: Large Model Training (ResNet50)")
    print("="*50)

    config = Config(
        data=DataConfig(
            dataset_name="cifar10-lt",
            cifar10_config="r-100",
        ),
        encoder=EncoderConfig(
            architecture="resnet50",  # Larger encoder
            projection_dim=256,  # Larger projection
            hidden_dim=4096,
        ),
        contrastive_training=ContrastiveTrainingConfig(
            num_epochs=300,  # More epochs
            batch_size=1024,  # Larger batch
            learning_rate=0.5,
        ),
        classifier_training=ClassifierTrainingConfig(
            num_epochs=150,
            learning_rate=0.001,
        ),
        seed=42,
        output_dir=Path("outputs/example_large"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = run_contrastive_pipeline(config=config)

    print("\nFinal Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def example_creditcard():
    """Example: Credit Card Fraud Detection."""
    print("\n" + "="*50)
    print("Example 3: Credit Card Fraud Detection")
    print("="*50)

    config = Config(
        data=DataConfig(
            dataset_name="credit-card-fraud",
            num_classes=2,
            num_workers=4,
        ),
        encoder=EncoderConfig(
            architecture="resnet18",
            projection_dim=128,
        ),
        contrastive_training=ContrastiveTrainingConfig(
            num_epochs=100,
            batch_size=256,  # Smaller batch for fewer samples
            learning_rate=0.5,
        ),
        classifier_training=ClassifierTrainingConfig(
            num_epochs=50,
            learning_rate=0.001,
            patience=15,
        ),
        seed=42,
        output_dir=Path("outputs/example_creditcard"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = run_contrastive_pipeline(config=config)

    print("\nFinal Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def example_custom_augmentation():
    """Example: Custom augmentation parameters."""
    print("\n" + "="*50)
    print("Example 4: Custom Augmentation")
    print("="*50)

    config = Config(
        data=DataConfig(
            dataset_name="cifar10-lt",
            cifar10_config="r-50",
        ),
        augmentation=AugmentationConfig(
            crop_scale=(0.2, 1.0),  # Different crop range
            color_jitter_strength=0.8,  # Stronger jitter
            blur_probability=0.8,  # More blur
            blur_kernel_size=31,
        ),
        encoder=EncoderConfig(
            architecture="resnet18",
            projection_dim=256,
        ),
        contrastive_training=ContrastiveTrainingConfig(
            num_epochs=200,
            batch_size=512,
        ),
        seed=42,
        output_dir=Path("outputs/example_augmentation"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = run_contrastive_pipeline(config=config)

    print("\nFinal Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def example_different_temperature():
    """Example: Different temperature for contrastive loss."""
    print("\n" + "="*50)
    print("Example 5: Different Temperature Values")
    print("="*50)

    for temperature in [0.01, 0.07, 0.2]:
        print(f"\nTraining with temperature={temperature}...")

        config = Config(
            data=DataConfig(dataset_name="cifar10-lt"),
            contrastive_loss=ContrastiveLossConfig(temperature=temperature),
            contrastive_training=ContrastiveTrainingConfig(
                num_epochs=200,
                batch_size=512,
            ),
            seed=42,
            output_dir=Path(f"outputs/example_temp_{temperature}"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        metrics = run_contrastive_pipeline(config=config)

        print(f"Accuracy with temperature={temperature}: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    import sys

    print("Contrastive Learning Examples")
    print("="*50)
    print("Select an example to run:")
    print("1. Basic CIFAR-10-LT Training")
    print("2. Large Model Training")
    print("3. Credit Card Fraud Detection")
    print("4. Custom Augmentation")
    print("5. Different Temperature Values")
    print("0. Exit")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (0-5): ")

    if choice == "1":
        example_cifar10lt_basic()
    elif choice == "2":
        example_cifar10lt_large()
    elif choice == "3":
        example_creditcard()
    elif choice == "4":
        example_custom_augmentation()
    elif choice == "5":
        example_different_temperature()
    else:
        print("Exiting...")
        sys.exit(0)
