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
