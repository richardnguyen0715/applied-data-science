"""Model evaluation utilities."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import ClassificationMetrics
from src.models.base import BaseModel
from src.utils.logger import get_logger


def evaluate_classifier(
    model: BaseModel,
    data_loader: DataLoader,
    num_classes: int = 10,
    device: torch.device = None,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Evaluate classifier on a dataset.

    Args:
        model: Classifier model.
        data_loader: DataLoader for evaluation.
        num_classes: Number of classes.
        device: Device to evaluate on.

    Returns:
        Tuple of (overall_metrics, per_class_metrics).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger("evaluator")

    model = model.to(device)
    model.eval()

    metrics_calculator = ClassificationMetrics(num_classes)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            metrics_calculator.update(logits, labels)

    overall_metrics = metrics_calculator.compute()
    per_class_metrics = metrics_calculator.compute_per_class()

    logger.info("Overall Metrics:")
    for metric_name, metric_value in overall_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    logger.info("\nPer-Class Metrics:")
    for class_idx, class_metrics in per_class_metrics.items():
        logger.info(f"  Class {class_idx}: {class_metrics}")

    return overall_metrics, per_class_metrics


def get_confusion_matrix(
    model: BaseModel,
    data_loader: DataLoader,
    num_classes: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Get confusion matrix.

    Args:
        model: Classifier model.
        data_loader: DataLoader for evaluation.
        num_classes: Number of classes.
        device: Device to use.

    Returns:
        Confusion matrix.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    metrics_calculator = ClassificationMetrics(num_classes)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            metrics_calculator.update(logits, labels)

    return metrics_calculator.get_confusion_matrix()
