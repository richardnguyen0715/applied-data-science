"""Evaluation metrics for classification."""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class ClassificationMetrics:
    """Compute classification metrics."""

    def __init__(self, num_classes: int) -> None:
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset accumulated predictions."""
        self.all_preds = []
        self.all_labels = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Update with batch predictions.

        Args:
            logits: Model logits.
            labels: Ground truth labels.
        """
        _, preds = logits.max(1)
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metrics.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        }

        return metrics

    def compute_per_class(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-class metrics.

        Returns:
            Dictionary of per-class metrics.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        per_class_metrics = {}

        for class_idx in range(self.num_classes):
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue

            class_preds = preds[class_mask]
            class_labels = labels[class_mask]

            per_class_metrics[class_idx] = {
                "accuracy": accuracy_score(class_labels, class_preds),
                "f1": f1_score(
                    class_labels,
                    class_preds,
                    average="weighted",
                    zero_division=0,
                ),
                "precision": precision_score(
                    class_labels,
                    class_preds,
                    average="weighted",
                    zero_division=0,
                ),
                "recall": recall_score(
                    class_labels,
                    class_preds,
                    average="weighted",
                    zero_division=0,
                ),
                "count": class_mask.sum(),
            }

        return per_class_metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix.
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        return confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
