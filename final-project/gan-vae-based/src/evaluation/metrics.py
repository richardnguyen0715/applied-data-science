"""Evaluation metrics."""

from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute accuracy.

    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.

    Returns:
        Accuracy value.
    """
    return float((predictions == targets).mean())


def compute_balanced_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> float:
    """
    Compute balanced accuracy (average of per-class recall).

    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        num_classes: Number of classes.

    Returns:
        Balanced accuracy value.
    """
    recalls = []
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_recall = (predictions[mask] == c).mean()
            recalls.append(class_recall)

    return float(np.mean(recalls))


def compute_f1_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = 'macro',
) -> float:
    """
    Compute F1 score.

    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        average: Averaging method ('macro', 'micro', 'weighted').

    Returns:
        F1 score.
    """
    return float(f1_score(targets, predictions, average=average, zero_division=0))


def compute_precision_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = 'macro',
) -> Tuple[float, float]:
    """
    Compute precision and recall.

    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        average: Averaging method.

    Returns:
        Tuple of (precision, recall).
    """
    precision = float(precision_score(targets, predictions, average=average, zero_division=0))
    recall = float(recall_score(targets, predictions, average=average, zero_division=0))
    return precision, recall


def compute_per_class_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict:
    """
    Compute per-class metrics.

    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        num_classes: Number of classes.

    Returns:
        Dictionary of per-class metrics.
    """
    metrics = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            recall = (predictions[mask] == c).mean()
            metrics[f'class_{c}_recall'] = float(recall)
        else:
            metrics[f'class_{c}_recall'] = 0.0

    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted labels.
        targets: Ground truth labels.
        num_classes: Number of classes.

    Returns:
        Confusion matrix.
    """
    return confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Args:
        model: Model to evaluate.
        data_loader: Data loader.
        device: Device to evaluate on.

    Returns:
        Tuple of (predictions, targets).
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(targets)
