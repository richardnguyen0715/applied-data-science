"""Conditional VAE (CVAE) for oversampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseDecoder, BaseEncoder


class ConditionalEncoder(BaseEncoder):
    """Conditional Encoder for CVAE."""

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ):
        """
        Initialize conditional encoder.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__(latent_dim, num_classes)
        self.hidden_dim = hidden_dim

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 32 * 32)

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output layer for mean and log_var
        self.fc_mean = nn.Linear(hidden_dim * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4 * 4, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space.

        Args:
            x: Input images (B, 3, 32, 32).
            labels: Class labels (B,).

        Returns:
            Tuple of (mean, log_var).
        """
        # Embed and reshape labels
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(-1, 1, 32, 32)

        # Concatenate image and label
        x = torch.cat([x, label_embedding], dim=1)

        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        # Output mean and log_var
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class ConditionalDecoder(BaseDecoder):
    """Conditional Decoder for CVAE."""

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ):
        """
        Initialize conditional decoder.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__(latent_dim, num_classes, output_channels=3)
        self.hidden_dim = hidden_dim

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)

        # Linear layer to reshape
        self.fc = nn.Linear(2 * latent_dim, hidden_dim * 4 * 4)

        # Convolutional decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Latent vectors (B, latent_dim).
            labels: Class labels (B,).

        Returns:
            Reconstructed images (B, 3, 32, 32).
        """
        # Embed labels
        label_embedding = self.label_embedding(labels)

        # Concatenate latent and label embedding
        x = torch.cat([z, label_embedding], dim=1)

        # Linear layer
        x = self.fc(x)
        x = x.view(-1, self.hidden_dim, 4, 4)

        # Decode
        x = self.decoder(x)

        return x


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder."""

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 10,
        hidden_dim: int = 256,
    ):
        """
        Initialize CVAE.

        Args:
            latent_dim: Dimension of latent space.
            num_classes: Number of classes.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = ConditionalEncoder(latent_dim, num_classes, hidden_dim)
        self.decoder = ConditionalDecoder(latent_dim, num_classes, hidden_dim)

    def encode(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space.

        Args:
            x: Input images (B, 3, 32, 32).
            labels: Class labels (B,).

        Returns:
            Tuple of (mean, log_var).
        """
        return self.encoder(x, labels)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick.

        Args:
            mean: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            Sampled latent vectors.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to images.

        Args:
            z: Latent vectors (B, latent_dim).
            labels: Class labels (B,).

        Returns:
            Reconstructed images (B, 3, 32, 32).
        """
        return self.decoder(z, labels)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input images (B, 3, 32, 32).
            labels: Class labels (B,).

        Returns:
            Tuple of (reconstructed, mean, log_var).
        """
        mean, logvar = self.encode(x, labels)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z, labels)
        return reconstruction, mean, logvar

    def sample(
        self,
        num_samples: int,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample images from latent space.

        Args:
            num_samples: Number of samples to generate.
            labels: Class labels (num_samples,).
            device: Device to generate on.

        Returns:
            Generated images (num_samples, 3, 32, 32).
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decode(z, labels)
        return samples
