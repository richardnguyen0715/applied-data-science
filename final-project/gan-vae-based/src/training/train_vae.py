"""Training logic for VAE."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import VAELoss

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer for Conditional VAE."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize VAE trainer.

        Args:
            model: VAE model.
            config: Configuration dictionary.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.lr = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 50)
        kld_weight = config.get('kld_weight', 0.00025)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
        )

        # Loss
        self.loss_fn = VAELoss(kld_weight=kld_weight)

        # Metrics
        self.train_losses = []
        self.recon_losses = []
        self.kld_losses = []

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float, float]:
        """
        Train VAE for one epoch.

        Args:
            train_loader: Training dataloader.

        Returns:
            Tuple of (total_loss, recon_loss, kld_loss).
        """
        self.model.train()
        total_loss = 0.0
        recon_loss = 0.0
        kld_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstruction, mean, logvar = self.model(images, labels)

            # Loss
            loss, recon, kld = self.loss_fn(reconstruction, images, mean, logvar)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            recon_loss += recon.item()
            kld_loss += kld.item()

            if batch_idx % 50 == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Recon: {recon.item():.4f} | KLD: {kld.item():.4f}"
                )

        # Average metrics
        num_batches = len(train_loader)
        return total_loss / num_batches, recon_loss / num_batches, kld_loss / num_batches

    def fit(self, train_loader: DataLoader) -> None:
        """
        Train VAE for multiple epochs.

        Args:
            train_loader: Training dataloader.
        """
        logger.info(f"Starting VAE training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            total_loss, recon_loss, kld_loss = self.train_epoch(train_loader)

            logger.info(
                f"Epoch {epoch + 1} - "
                f"Total Loss: {total_loss:.4f}, "
                f"Recon Loss: {recon_loss:.4f}, "
                f"KLD Loss: {kld_loss:.4f}"
            )

            self.train_losses.append(total_loss)
            self.recon_losses.append(recon_loss)
            self.kld_losses.append(kld_loss)

            # Save checkpoint
            if self.checkpoint_dir is not None:
                self.save_checkpoint(epoch)

        logger.info("VAE training completed")

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
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }

        checkpoint_path = self.checkpoint_dir / f"vae_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
