"""Noise scheduler for diffusion models."""

from typing import Optional

import numpy as np
import torch


class NoiseScheduler:
    """Scheduler for noise schedule in diffusion models."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ) -> None:
        """
        Initialize noise scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps.
            beta_start: Starting variance.
            beta_end: Ending variance.
            schedule: Schedule type ('linear' or 'cosine').
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = torch.arange(num_timesteps + 1)
            alphas_cumprod = torch.cos(
                ((steps / num_timesteps) + s) / (1 + s) * np.pi * 0.5
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1), self.alphas_cumprod[:-1]]
        )

        # Precompute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1.0
        )

        # Posterior variance
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device) -> "NoiseScheduler":
        """
        Move all scheduler tensors to device.

        Args:
            device: Target device.

        Returns:
            Self (for chaining).
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def add_noise(
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
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_0
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def predict_x_0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        Args:
            x_t: Noisy sample at timestep t.
            t: Timestep indices.
            noise_pred: Predicted noise.

        Returns:
            Predicted x_0.
        """
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return (
            sqrt_recip_alphas_cumprod_t * x_t
            - sqrt_recipm1_alphas_cumprod_t * noise_pred
        )

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """
        Compute posterior mean and variance for sampling.

        Args:
            x_0: Clean sample.
            x_t: Noisy sample.
            t: Timestep indices.

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped).
        """
        posterior_mean_coef1_t = self._extract(
            self.posterior_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2_t = self._extract(
            self.posterior_mean_coef2, t, x_t.shape
        )
        posterior_log_variance_clipped_t = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        posterior_mean = (
            posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        )

        return posterior_mean, posterior_log_variance_clipped_t

    @staticmethod
    def _extract(
        arr: torch.Tensor,
        timesteps: torch.Tensor,
        x_shape: tuple,
    ) -> torch.Tensor:
        """
        Extract values from a 1-D tensor and reshape for broadcasting.

        Args:
            arr: 1-D tensor to extract from.
            timesteps: Indices to extract.
            x_shape: Shape to reshape to.

        Returns:
            Extracted and reshaped tensor.
        """
        batch_size = timesteps.shape[0]
        out = arr.gather(-1, timesteps)
        return out.reshape(batch_size, *([1] * (len(x_shape) - 1)))
