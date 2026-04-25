"""Evaluation utilities for CIFAR-10 long-tail experiments."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.metrics import (
    compute_balanced_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
)


class Evaluator:
    """Evaluate classification model on standard and imbalance-aware metrics."""

    def __init__(self, num_classes: int, device: torch.device, criterion: nn.Module | None = None) -> None:
        """Initialize evaluator.

        Args:
            num_classes: Number of output classes.
            device: Device used for forward passes.
            criterion: Optional criterion for reporting evaluation loss.
        """
        self.num_classes = num_classes
        self.device = device
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def evaluate(self, model: nn.Module, dataloader: DataLoader[Any]) -> Dict[str, Any]:
        """Run a full evaluation pass.

        Args:
            model: Model to evaluate.
            dataloader: Evaluation dataloader.

        Returns:
            Dictionary with loss, accuracy, balanced accuracy, confusion matrix,
            and per-class accuracy.
        """
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_examples = 0
        total_correct = 0
        all_targets: List[int] = []
        all_predictions: List[int] = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits: Tensor = model(images)
                loss = self.criterion(logits, targets)

                predictions = torch.argmax(logits, dim=1)

                batch_size = targets.size(0)
                total_examples += batch_size
                total_loss += float(loss.item() * batch_size)
                total_correct += int((predictions == targets).sum().item())

                all_targets.extend(targets.detach().cpu().tolist())
                all_predictions.extend(predictions.detach().cpu().tolist())

        if was_training:
            model.train()

        avg_loss = total_loss / max(total_examples, 1)
        accuracy = total_correct / max(total_examples, 1)
        balanced_acc = compute_balanced_accuracy(all_targets, all_predictions)
        conf_matrix = compute_confusion_matrix(all_targets, all_predictions, self.num_classes)
        per_class_acc = compute_per_class_accuracy(conf_matrix)

        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "confusion_matrix": conf_matrix,
            "per_class_accuracy": per_class_acc,
        }
