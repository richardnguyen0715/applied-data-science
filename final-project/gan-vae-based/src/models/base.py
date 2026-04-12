"""Base model classes."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGenerator(nn.Module, ABC):
    """Base class for generator models."""

    def __init__(self, latent_dim: int, num_classes: int, output_channels: int = 3):
        """
        Initialize generator.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
            output_channels: Number of output channels (default: 3 for RGB).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_channels = output_channels

    @abstractmethod
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate samples.

        Args:
            z: Latent vectors (B, latent_dim).
            labels: Class labels (B,).

        Returns:
            Generated images.
        """
        pass


class BaseDiscriminator(nn.Module, ABC):
    """Base class for discriminator models."""

    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        Initialize discriminator.

        Args:
            num_classes: Number of classes.
            input_channels: Number of input channels (default: 3 for RGB).
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Discriminate samples.

        Args:
            x: Input images.
            labels: Class labels (B,).

        Returns:
            Discrimination scores.
        """
        pass


class BaseEncoder(nn.Module, ABC):
    """Base class for encoder models."""

    def __init__(self, latent_dim: int, num_classes: int):
        """
        Initialize encoder.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode samples.

        Args:
            x: Input images.
            labels: Class labels (B,).

        Returns:
            Tuple of (mean, log_var) for latent distribution.
        """
        pass


class BaseDecoder(nn.Module, ABC):
    """Base class for decoder models."""

    def __init__(self, latent_dim: int, num_classes: int, output_channels: int = 3):
        """
        Initialize decoder.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
            output_channels: Number of output channels (default: 3 for RGB).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.output_channels = output_channels

    @abstractmethod
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Decode samples.

        Args:
            z: Latent vectors (B, latent_dim).
            labels: Class labels (B,).

        Returns:
            Reconstructed images.
        """
        pass
