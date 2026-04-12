"""Conditional GAN (cGAN) for oversampling."""

import torch
import torch.nn as nn

from src.models.base import BaseDiscriminator, BaseGenerator


class ConditionalGenerator(BaseGenerator):
    """Conditional Generator for CIFAR-10 images."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        hidden_dim: int = 128,
    ):
        """
        Initialize conditional generator.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__(latent_dim, num_classes, output_channels=3)
        self.hidden_dim = hidden_dim

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Main generator network
        self.model = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim * 8),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 8, hidden_dim * 4 * 4 * 4),
            nn.BatchNorm1d(hidden_dim * 4 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Reshape to (hidden_dim * 4, 4, 4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate images.

        Args:
            z: Latent vectors (B, latent_dim).
            labels: Class labels (B,).

        Returns:
            Generated images (B, 3, 32, 32).
        """
        # Embed labels
        label_embedding = self.label_embedding(labels)

        # Concatenate noise and label embedding
        x = torch.cat([z, label_embedding], dim=1)

        # Linear layers
        x = self.model(x)

        # Reshape for convolution
        x = x.view(-1, self.hidden_dim * 4, 4, 4)

        # Convolutional layers
        x = self.conv(x)

        return x


class ConditionalDiscriminator(BaseDiscriminator):
    """Conditional Discriminator for CIFAR-10 images."""

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 128,
    ):
        """
        Initialize conditional discriminator.

        Args:
            num_classes: Number of classes.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__(num_classes, input_channels=3)
        self.hidden_dim = hidden_dim

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 32 * 32)

        # Main discriminator network
        self.conv = nn.Sequential(
            nn.Conv2d(4, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 4 * 4, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Discriminate images.

        Args:
            x: Input images (B, 3, 32, 32).
            labels: Class labels (B,).

        Returns:
            Discrimination scores (B, 1).
        """
        # Embed and reshape labels to match image spatial dimensions
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(-1, 1, 32, 32)

        # Concatenate image and label embedding
        x = torch.cat([x, label_embedding], dim=1)

        # Convolutional layers
        x = self.conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x
