"""Base model and utility classes."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self) -> None:
        """Initialize base model."""
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_device(self) -> torch.device:
        """
        Get the device of the model.

        Returns:
            Device (cpu or cuda).
        """
        return next(self.parameters()).device

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class ExponentialMovingAverage:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """
        Initialize EMA.

        Args:
            model: Model to apply EMA to.
            decay: Decay rate for EMA.
        """
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data
                    + (1 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data

    def restore(self) -> None:
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
