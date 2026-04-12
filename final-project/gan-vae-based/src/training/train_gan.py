"""Training logic for GAN."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import GANLoss

logger = logging.getLogger(__name__)


class GANTrainer:
    """Trainer for Conditional GAN."""

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        config: dict,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize GAN trainer.

        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            config: Configuration dictionary.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.lr = config.get('learning_rate', 0.0002)
        self.beta1 = config.get('beta1', 0.5)
        self.beta2 = config.get('beta2', 0.999)
        self.epochs = config.get('epochs', 50)
        self.critic_iterations = config.get('critic_iterations', 5)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        self.d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )

        # Loss
        self.loss = GANLoss()

        # Metrics
        self.d_losses = []
        self.g_losses = []

    def train(self, train_loader: DataLoader) -> None:
        """
        Train GAN for one epoch.

        Args:
            train_loader: Training dataloader.
        """
        self.generator.train()
        self.discriminator.train()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for batch_idx, (real_images, labels) in progress_bar:
            real_images = real_images.to(self.device)
            labels = labels.to(self.device)
            batch_size = real_images.size(0)

            # Train discriminator
            for _ in range(self.critic_iterations):
                self.d_optimizer.zero_grad()

                # Real data loss
                real_output = self.discriminator(real_images, labels)
                d_loss_real = self.loss(real_output, is_real=True)

                # Fake data loss
                z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
                fake_images = self.generator(z, labels)
                fake_output = self.discriminator(fake_images.detach(), labels)
                d_loss_fake = self.loss(fake_output, is_real=False)

                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

            # Train generator
            self.g_optimizer.zero_grad()

            z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
            fake_images = self.generator(z, labels)
            fake_output = self.discriminator(fake_images, labels)
            g_loss = self.loss(fake_output, is_real=True)

            g_loss.backward()
            self.g_optimizer.step()

            # Update progress bar with metrics
            self.d_losses.append(d_loss.item())
            self.g_losses.append(g_loss.item())
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })

    def fit(self, train_loader: DataLoader) -> None:
        """
        Train GAN for multiple epochs.

        Args:
            train_loader: Training dataloader.
        """
        logger.info(f"Starting GAN training for {self.epochs} epochs")

        for epoch in tqdm(range(self.epochs), desc="GAN Training", unit="epoch"):
            self.train(train_loader)

            # Log epoch metrics
            avg_d_loss = sum(self.d_losses[-len(train_loader):]) / len(train_loader)
            avg_g_loss = sum(self.g_losses[-len(train_loader):]) / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

            # Save checkpoint
            if self.checkpoint_dir is not None:
                self.save_checkpoint(epoch)

        logger.info("GAN training completed")

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch.
        """
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
        }

        checkpoint_path = self.checkpoint_dir / f"gan_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
