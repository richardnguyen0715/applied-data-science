"""Diffusion model implementation."""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base import BaseModel, ExponentialMovingAverage
from src.models.scheduler import NoiseScheduler
from src.models.unet import UNet


class DiffusionModel(BaseModel):
    """Conditional Denoising Diffusion Probabilistic Model (DDPM)."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        model_channels: int = 64,
        num_residual_blocks: int = 2,
        attention_resolutions: tuple = (8, 16),
        num_classes: int = 10,
        class_embed_dim: int = 128,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ) -> None:
        """
        Initialize diffusion model.

        Args:
            num_timesteps: Number of diffusion timesteps.
            beta_start: Starting variance.
            beta_end: Ending variance.
            beta_schedule: Schedule type ('linear' or 'cosine').
            model_channels: Base number of channels for U-Net.
            num_residual_blocks: Number of residual blocks.
            attention_resolutions: Resolutions to apply attention.
            num_classes: Number of classes.
            class_embed_dim: Class embedding dimension.
            use_ema: Whether to use exponential moving average.
            ema_decay: EMA decay rate.
        """
        super().__init__()

        self.num_timesteps = num_timesteps
        self.num_classes = num_classes
        self.use_ema = use_ema

        # Noise scheduler
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=beta_schedule,
        )

        # U-Net backbone
        self.model = UNet(
            in_channels=3,
            out_channels=3,
            model_channels=model_channels,
            num_residual_blocks=num_residual_blocks,
            attention_resolutions=attention_resolutions,
            num_classes=num_classes,
            class_embed_dim=class_embed_dim,
        )

        # EMA
        if use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
        else:
            self.ema = None

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (noisy images).
            t: Timestep indices.
            y: Class labels.

        Returns:
            Predicted noise.
        """
        return self.model(x, t, y)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to clean sample (forward diffusion).

        Args:
            x_0: Clean sample.
            t: Timestep indices.
            noise: Gaussian noise.

        Returns:
            Noisy sample x_t.
        """
        return self.scheduler.add_noise(x_0, t, noise)

    @torch.no_grad()
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single denoising step.

        Args:
            x_t: Noisy sample at timestep t.
            t: Timestep indices.
            y: Class labels.

        Returns:
            Less noisy sample.
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Predict noise
        predicted_noise = self.model(x_t, t, y)

        # Schedule parameters
        alpha = self.scheduler.alphas[t]
        alpha_cumprod = self.scheduler.alphas_cumprod[t]
        alpha_cumprod_prev = self.scheduler.alphas_cumprod_prev[t]

        # Posterior mean
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod).unsqueeze(-1).unsqueeze(-1) * predicted_noise) / torch.sqrt(alpha_cumprod).unsqueeze(-1).unsqueeze(-1)
        
        mean = (
            torch.sqrt(alpha_cumprod_prev).unsqueeze(-1).unsqueeze(-1) * self.scheduler.betas[t].unsqueeze(-1).unsqueeze(-1) * x_0_pred
            + torch.sqrt(alpha).unsqueeze(-1).unsqueeze(-1) * (1 - alpha_cumprod_prev).unsqueeze(-1).unsqueeze(-1) * x_t
        ) / (1 - alpha_cumprod).unsqueeze(-1).unsqueeze(-1)

        # Add noise if not at last step
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.scheduler.posterior_variance[t].unsqueeze(-1).unsqueeze(-1)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_classes: int,
        image_size: int = 32,
        device: torch.device = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor:
        """
        Sample images from the diffusion model.

        Args:
            num_samples: Number of samples to generate per class.
            num_classes: Number of classes.
            image_size: Size of generated images.
            device: Device to generate on.
            return_all_steps: Whether to return all diffusion steps.

        Returns:
            Generated images.
        """
        if device is None:
            device = self.get_device()

        all_samples = []

        for class_idx in range(num_classes):
            # Start from Gaussian noise
            x_t = torch.randn(
                num_samples, 3, image_size, image_size, device=device
            )
            y = torch.full((num_samples,), class_idx, dtype=torch.long, device=device)

            # Reverse diffusion
            for t in range(self.num_timesteps - 1, -1, -1):
                t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
                x_t = self.denoise_step(x_t, t_tensor, y)

            all_samples.append(x_t)

        samples = torch.cat(all_samples, dim=0)
        return samples

    def to(self, device: torch.device) -> "DiffusionModel":
        """
        Move model and scheduler to device.

        Args:
            device: Target device.

        Returns:
            Self (for chaining).
        """
        super().to(device)
        self.scheduler.to(device)
        return self

    def get_ema_model(self) -> nn.Module:
        """
        Get the EMA model.

        Returns:
            EMA model or regular model if EMA not used.
        """
        if self.ema is None:
            return self.model

        self.ema.apply_shadow()
        ema_model = self.model
        self.ema.restore()
        return ema_model
