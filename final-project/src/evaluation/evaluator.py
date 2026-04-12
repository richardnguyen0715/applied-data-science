"""Evaluator class for comprehensive model evaluation."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.evaluation.metrics import (
    compute_balanced_accuracy,
    compute_confusion_matrix,
    compute_f1_scores,
    compute_per_class_metrics,
    compute_precision_recall,
    evaluate_model,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive model evaluator."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes: int,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate.
            device: Device for evaluation.
            num_classes: Number of classes.
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float | np.ndarray]:
        """
        Evaluate model comprehensively.

        Args:
            data_loader: Data loader.

        Returns:
            Dictionary of metrics.
        """
        # Get predictions and targets
        predictions, targets = evaluate_model(self.model, data_loader, self.device)

        # Compute metrics
        metrics = {}

        # Overall metrics
        accuracy = float((predictions == targets).mean())
        metrics['accuracy'] = accuracy

        balanced_acc = compute_balanced_accuracy(predictions, targets, self.num_classes)
        metrics['balanced_accuracy'] = balanced_acc

        macro_f1 = compute_f1_scores(predictions, targets, average='macro')
        metrics['macro_f1'] = macro_f1

        micro_f1 = compute_f1_scores(predictions, targets, average='micro')
        metrics['micro_f1'] = micro_f1

        weighted_f1 = compute_f1_scores(predictions, targets, average='weighted')
        metrics['weighted_f1'] = weighted_f1

        precision, recall = compute_precision_recall(predictions, targets, average='macro')
        metrics['macro_precision'] = precision
        metrics['macro_recall'] = recall

        # Per-class metrics
        per_class = compute_per_class_metrics(predictions, targets, self.num_classes)
        metrics.update(per_class)

        # Confusion matrix
        cm = compute_confusion_matrix(predictions, targets, self.num_classes)
        metrics['confusion_matrix'] = cm

        return metrics

    def log_metrics(self, metrics: Dict[str, float | np.ndarray]) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics.
        """
        logger.info("=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)

        # Overall metrics
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Micro F1: {metrics['micro_f1']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
        logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")

        # Per-class metrics
        logger.info("\nPer-class Recall:")
        for c in range(self.num_classes):
            recall = metrics.get(f'class_{c}_recall', 0.0)
            logger.info(f"  Class {c}: {recall:.4f}")

        logger.info("=" * 50)

    def save_metrics(
        self,
        metrics: Dict[str, float | np.ndarray],
        output_path: Path,
    ) -> None:
        """
        Save metrics to file.

        Args:
            metrics: Dictionary of metrics.
            output_path: Path to save metrics.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("Evaluation Metrics\n")
            f.write("=" * 50 + "\n\n")

            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"  Micro F1: {metrics['micro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {metrics['weighted_f1']:.4f}\n")
            f.write(f"  Macro Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall: {metrics['macro_recall']:.4f}\n\n")

            # Per-class metrics
            f.write("Per-class Recall:\n")
            for c in range(self.num_classes):
                recall = metrics.get(f'class_{c}_recall', 0.0)
                f.write(f"  Class {c}: {recall:.4f}\n")

            # Confusion matrix
            f.write("\nConfusion Matrix:\n")
            cm = metrics['confusion_matrix']
            for row in cm:
                f.write("  " + " ".join(f"{x:5d}" for x in row) + "\n")

        logger.info(f"Metrics saved to {output_path}")
