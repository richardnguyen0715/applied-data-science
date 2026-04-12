"""Baseline pipeline - train classifier on original imbalanced data."""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CIFAR10LTDataset, create_data_loaders
from src.data.imbalance import analyze_class_distribution, print_class_distribution
from src.data.transforms import get_train_transforms, get_test_transforms
from src.evaluation.evaluator import evaluate_classifier
from src.evaluation.visualize import plot_training_curves
from src.models.classifier import create_classifier
from src.training.train_classifier import train_classifier
from src.utils.config import ClassifierConfig, DataConfig, TrainingConfig
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def run_baseline_pipeline(
    output_dir: Path = Path("outputs"),
    data_config: Optional[DataConfig] = None,
    classifier_config: Optional[ClassifierConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run baseline pipeline - train classifier on original imbalanced data.

    Args:
        output_dir: Output directory.
        data_config: Data configuration.
        classifier_config: Classifier configuration.
        training_config: Training configuration.
        device: Device to use.

    Returns:
        Dictionary of evaluation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_config is None:
        data_config = DataConfig()
    if classifier_config is None:
        classifier_config = ClassifierConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # Setup directories
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs" / "baseline"
    checkpoint_dir = output_dir / "checkpoints" / "baseline"
    figure_dir = output_dir / "figures" / "baseline"

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("baseline", log_dir)

    # Set seed
    set_seed(training_config.seed)

    logger.info("Starting baseline evaluation...")
    logger.info(f"Using device: {device}")

    # Load training data
    logger.info("Loading training data...")
    train_dataset = CIFAR10LTDataset(
        split="train",
        transform=get_train_transforms(data_config.image_size),
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
    )
    train_loader = create_data_loaders(
        train_dataset,
        batch_size=classifier_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        shuffle=True,
    )

    # Analyze class distribution
    class_dist = analyze_class_distribution(
        [train_dataset.labels[i] for i in range(len(train_dataset))]
    )
    logger.info("Training data class distribution:")
    print_class_distribution(class_dist)

    # Load test data
    logger.info("Loading test data...")
    test_dataset = CIFAR10LTDataset(
        split="test",
        transform=get_test_transforms(data_config.image_size),
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
    )
    test_loader = create_data_loaders(
        test_dataset,
        batch_size=classifier_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        shuffle=False,
    )

    # Create classifier
    logger.info("Creating classifier...")
    classifier = create_classifier(
        model_name=classifier_config.model_name,
        num_classes=data_config.num_classes,
        pretrained=False,
    )
    logger.info(f"Model parameters: {classifier.count_parameters():,}")

    # Train classifier
    logger.info("Training classifier on original imbalanced data...")
    history = train_classifier(
        model=classifier,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=classifier_config.num_epochs,
        learning_rate=classifier_config.learning_rate,
        weight_decay=classifier_config.weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        log_every_n_steps=training_config.log_every_n_steps,
    )

    # Evaluate
    logger.info("Evaluating classifier...")
    overall_metrics, per_class_metrics = evaluate_classifier(
        classifier,
        test_loader,
        num_classes=data_config.num_classes,
        device=device,
    )

    # Plot training curves
    logger.info("Plotting training curves...")
    plot_training_curves(
        history,
        save_path=figure_dir / "training_curves.png",
    )

    logger.info("Baseline pipeline completed!")

    return overall_metrics


if __name__ == "__main__":
    run_baseline_pipeline()
