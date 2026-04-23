"""Metrics calculation utilities."""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class ClassificationMetrics:
    """Calculate classification metrics."""

    def __init__(self, num_classes: int = 10) -> None:
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset metrics."""
        self.predictions = []
        self.ground_truth = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update metrics with new predictions.

        Args:
            logits: Predicted logits (B, num_classes).
            labels: Ground truth labels (B,).
        """
        _, predicted = torch.max(logits, 1)
        self.predictions.extend(predicted.cpu().numpy())
        self.ground_truth.extend(labels.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute overall metrics.

        Returns:
            Dictionary of metrics.
        """
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)

        accuracy = accuracy_score(ground_truth, predictions)
        balanced_accuracy = balanced_accuracy_score(ground_truth, predictions)
        macro_f1 = f1_score(ground_truth, predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(ground_truth, predictions, average="weighted", zero_division=0)
        macro_precision = precision_score(ground_truth, predictions, average="macro", zero_division=0)
        weighted_precision = precision_score(ground_truth, predictions, average="weighted", zero_division=0)
        macro_recall = recall_score(ground_truth, predictions, average="macro", zero_division=0)
        weighted_recall = recall_score(ground_truth, predictions, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_precision,
            "weighted_precision": weighted_precision,
            "macro_recall": macro_recall,
            "weighted_recall": weighted_recall,
        }

    def compute_per_class(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-class metrics.

        Returns:
            Dictionary of per-class metrics.
        """
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)

        per_class_metrics = {}

        for class_id in range(self.num_classes):
            class_mask = ground_truth == class_id
            if class_mask.sum() == 0:
                continue

            class_predictions = predictions[class_mask]
            class_ground_truth = ground_truth[class_mask]

            accuracy = accuracy_score(class_ground_truth, class_predictions)
            precision = precision_score(
                class_ground_truth,
                class_predictions,
                average="binary" if self.num_classes == 2 else "weighted",
                zero_division=0,
            )
            recall = recall_score(
                class_ground_truth,
                class_predictions,
                average="binary" if self.num_classes == 2 else "weighted",
                zero_division=0,
            )
            f1 = f1_score(
                class_ground_truth,
                class_predictions,
                average="binary" if self.num_classes == 2 else "weighted",
                zero_division=0,
            )

            per_class_metrics[class_id] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return per_class_metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix (num_classes, num_classes).
        """
        from sklearn.metrics import confusion_matrix

        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)

        return confusion_matrix(ground_truth, predictions, labels=range(self.num_classes))
