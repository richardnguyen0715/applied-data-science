"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Loss for diffusion model training."""

    def __init__(self, reduction: str = "mean") -> None:
        """
        Initialize diffusion loss.

        Args:
            reduction: 'mean' or 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target noise.

        Args:
            predicted_noise: Predicted noise from model.
            target_noise: Ground truth noise.

        Returns:
            Loss value.
        """
        loss = F.mse_loss(predicted_noise, target_noise, reduction=self.reduction)
        return loss


class ClassifierLoss(nn.Module):
    """Cross-entropy loss for classifier."""

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        """
        Initialize classifier loss.

        Args:
            reduction: 'mean' or 'sum'.
            label_smoothing: Label smoothing parameter.
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model logits.
            labels: Ground truth labels.

        Returns:
            Loss value.
        """
        return self.loss(logits, labels)


class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced data."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor.
            gamma: Focusing parameter.
            reduction: 'mean' or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model logits.
            labels: Ground truth labels.

        Returns:
            Loss value.
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
