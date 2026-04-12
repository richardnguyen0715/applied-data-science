"""Training loss functions."""

import torch
import torch.nn as nn


class VAELoss(nn.Module):
    """VAE Loss = Reconstruction + KL Divergence."""

    def __init__(self, kld_weight: float = 0.00025):
        """
        Initialize VAE loss.

        Args:
            kld_weight: Weight for KL divergence term.
        """
        super().__init__()
        self.kld_weight = kld_weight
        self.reconstruction_loss = nn.BCELoss(reduction='mean')

    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.

        Args:
            reconstruction: Reconstructed images.
            original: Original images.
            mean: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            Tuple of (total_loss, recon_loss, kld_loss).
        """
        # Reconstruction loss (BCE)
        recon_loss = self.reconstruction_loss(reconstruction, original)

        # KL divergence
        kld_loss = torch.mean(
            -0.5 * torch.sum(
                1 + logvar - mean.pow(2) - logvar.exp(),
                dim=1,
            ),
            dim=0,
        )

        # Total loss
        total_loss = recon_loss + self.kld_weight * kld_loss

        return total_loss, recon_loss, kld_loss


class GANLoss(nn.Module):
    """Binary cross-entropy loss for GAN."""

    def __init__(self):
        """Initialize GAN loss."""
        super().__init__()
        self.loss = nn.BCELoss()

    def __call__(
        self,
        predictions: torch.Tensor,
        is_real: bool,
    ) -> torch.Tensor:
        """
        Compute GAN loss.

        Args:
            predictions: Discriminator predictions.
            is_real: Whether samples are real (True) or fake (False).

        Returns:
            Loss value.
        """
        if is_real:
            labels = torch.ones_like(predictions)
        else:
            labels = torch.zeros_like(predictions)

        return self.loss(predictions, labels)


def gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.

    Args:
        discriminator: Discriminator model.
        real_data: Real data samples.
        fake_data: Generated data samples.
        labels: Class labels.
        device: Device to compute on.
        lambda_gp: Gradient penalty weight.

    Returns:
        Gradient penalty loss.
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    # Interpolate
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # Discriminator output
    d_interpolates = discriminator(interpolates, labels)

    # Gradient computation
    fake = torch.ones(batch_size, 1).to(device).requires_grad_(False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

    return gradient_penalty_loss
