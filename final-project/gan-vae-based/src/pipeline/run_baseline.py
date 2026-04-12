"""Baseline pipeline (no oversampling)."""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data.dataset import get_train_val_datasets
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualize import (
    plot_class_distribution,
    plot_training_curves,
)
from src.models.classifier import create_classifier
from src.training.train_classifier import ClassifierTrainer
from src.utils.config import PipelineConfig, save_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_baseline_pipeline(config: PipelineConfig) -> dict:
    """
    Run baseline pipeline without oversampling.

    Args:
        config: Pipeline configuration.

    Returns:
        Dictionary of results.
    """
    # Setup
    set_seed(config.random_seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    output_dir = config.output_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    logger_baseline = get_logger(__name__, log_dir=log_dir)
    logger_baseline.info("Starting baseline pipeline")

    # Save config
    save_config(config, output_dir / "config.yaml")

    # Load data
    logger_baseline.info("Loading CIFAR-10-LT dataset")
    train_dataset, val_dataset, test_dataset = get_train_val_datasets(
        dataset_name=config.data.dataset_name,
        config_name=config.data.config_name,
        val_split=config.data.val_split,
        image_size=config.data.image_size,
    )

    # Extract targets for distribution analysis
    train_targets = train_dataset.get_targets()
    logger_baseline.info(f"Train distribution: {sorted(set(train_targets))}")

    # Plot class distribution
    import numpy as np
    fig_dir = output_dir / "figures"
    plot_class_distribution(
        np.array(train_targets),
        title="CIFAR-10-LT Training Distribution (Baseline)",
        output_path=fig_dir / "class_distribution_baseline.png",
    )

    # Create dataloaders
    # Note: num_workers=0 required because HuggingFace datasets don't serialize
    # correctly across multiprocessing boundaries
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    logger_baseline.info("Creating ResNet18 classifier")
    model = create_classifier(
        architecture=config.classifier.architecture,
        num_classes=config.data.config_name.split('-')[0].count('r') or 10,
        pretrained=False,
    )

    # Train classifier
    checkpoint_dir = output_dir / "checkpoints"
    trainer = ClassifierTrainer(
        model=model,
        config=config.classifier.to_dict() if hasattr(config.classifier, 'to_dict') else vars(config.classifier),
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    logger_baseline.info("Training classifier...")
    trainer.fit(train_loader, val_loader)

    # Plot training curves
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        title="Baseline Classifier Training",
        output_path=fig_dir / "training_curves_baseline.png",
    )

    # Evaluate on test set
    logger_baseline.info("Evaluating on test set")
    evaluator = Evaluator(model, device, num_classes=10)
    test_metrics = evaluator.evaluate(test_loader)

    evaluator.log_metrics(test_metrics)
    evaluator.save_metrics(test_metrics, output_dir / "test_metrics.txt")

    results = {
        'method': 'baseline',
        'accuracy': test_metrics['accuracy'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'macro_f1': test_metrics['macro_f1'],
        'metrics': test_metrics,
        'model_path': checkpoint_dir / "classifier_best.pt",
    }

    logger_baseline.info("Baseline pipeline completed")
    return results
