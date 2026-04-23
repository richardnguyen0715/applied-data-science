"""Downstream classifier models."""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import BaseModel


class LinearClassifier(BaseModel):
    """Linear classifier for downstream task."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
    ) -> None:
        """
        Initialize linear classifier.

        Args:
            input_dim: Input dimension.
            num_classes: Number of classes.
            hidden_dim: Hidden dimension (if None, use linear layer only).
            num_layers: Number of layers.
        """
        super().__init__()

        if hidden_dim is None or num_layers == 1:
            # Simple linear layer
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            # Multi-layer classifier
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dim, num_classes))

            self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, input_dim).

        Returns:
            Logits (B, num_classes).
        """
        return self.classifier(x)
