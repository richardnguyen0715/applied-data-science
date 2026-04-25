"""Metrics for imbalanced multi-class classification."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


ArrayLikeInt = Iterable[int] | np.ndarray


def compute_balanced_accuracy(y_true: ArrayLikeInt, y_pred: ArrayLikeInt) -> float:
    """Compute balanced accuracy over all classes.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Balanced accuracy score in [0, 1].
    """
    y_true_np = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true)
    y_pred_np = np.asarray(list(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred)
    return float(balanced_accuracy_score(y_true_np, y_pred_np))


def compute_confusion_matrix(
    y_true: ArrayLikeInt,
    y_pred: ArrayLikeInt,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix with fixed class index range.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        num_classes: Number of classes.

    Returns:
        Confusion matrix with shape (num_classes, num_classes).
    """
    y_true_np = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true)
    y_pred_np = np.asarray(list(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred)
    labels = np.arange(num_classes)
    return confusion_matrix(y_true_np, y_pred_np, labels=labels)


def compute_per_class_accuracy(conf_matrix: np.ndarray) -> Dict[int, float]:
    """Compute accuracy for each class from a confusion matrix.

    Args:
        conf_matrix: Confusion matrix with rows as true classes.

    Returns:
        Dictionary mapping class index to per-class accuracy.
    """
    per_class_acc: Dict[int, float] = {}
    for class_idx in range(conf_matrix.shape[0]):
        row_sum = conf_matrix[class_idx].sum()
        if row_sum == 0:
            per_class_acc[class_idx] = 0.0
        else:
            per_class_acc[class_idx] = float(conf_matrix[class_idx, class_idx] / row_sum)
    return per_class_acc
