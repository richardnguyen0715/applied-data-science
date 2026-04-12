"""Training logic for VAE."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        self.patience = config.get('patience', 15)  # Early stopping patience
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
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate VAE.

        Args:
            val_loader: Validation dataloader.

        Returns:
            Tuple of (total_loss, recon_loss, kld_loss).
        """
        self.model.eval()
        total_loss = 0.0
        recon_loss = 0.0
        kld_loss = 0.0

        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                reconstruction, mean, logvar = self.model(images, labels)
                loss, recon, kld = self.loss_fn(reconstruction, images, mean, logvar)

                total_loss += loss.item()
                recon_loss += recon.item()
                kld_loss += kld.item()

        num_batches = len(val_loader)
        return total_loss / num_batches, recon_loss / num_batches, kld_loss / num_batches

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
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

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
        for batch_idx, (images, labels) in progress_bar:
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

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon.item():.4f}',
                'kld': f'{kld.item():.4f}'
            })

        # Average metrics
        num_batches = len(train_loader)
        return total_loss / num_batches, recon_loss / num_batches, kld_loss / num_batches

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Train VAE for multiple epochs.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader (optional).
        """
        logger.info(f"Starting VAE training for {self.epochs} epochs")
        logger.info(f"Early stopping patience: {self.patience} epochs")

        for epoch in tqdm(range(self.epochs), desc="VAE Training", unit="epoch"):
            total_loss, recon_loss, kld_loss = self.train_epoch(train_loader)

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Total Loss: {total_loss:.4f}, "
                f"Recon Loss: {recon_loss:.4f}, "
                f"KLD Loss: {kld_loss:.4f}"
            )

            self.train_losses.append(total_loss)
            self.recon_losses.append(recon_loss)
            self.kld_losses.append(kld_loss)

            # Validate if validation loader provided
            if val_loader is not None:
                val_total_loss, val_recon_loss, val_kld_loss = self.validate(val_loader)
                self.val_losses.append(val_total_loss)
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Val Total Loss: {val_total_loss:.4f}, "
                    f"Val Recon Loss: {val_recon_loss:.4f}, "
                    f"Val KLD Loss: {val_kld_loss:.4f}"
                )

                # Save best model and check early stopping
                if val_total_loss < self.best_val_loss:
                    self.best_val_loss = val_total_loss
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"Best model updated with val loss: {val_total_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping triggered after {self.patience} epochs without improvement. "
                            f"Best model at epoch {self.best_epoch} with val loss: {self.best_val_loss:.4f}"
                        )
                        break
            else:
                # Save checkpoint based on training loss if no validation loader
                if total_loss < self.best_val_loss:
                    self.best_val_loss = total_loss
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    if self.checkpoint_dir is not None:
                        self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"Best model updated with train loss: {total_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(
                            f"Early stopping triggered after {self.patience} epochs without improvement. "
                            f"Best model at epoch {self.best_epoch} with train loss: {self.best_val_loss:.4f}"
                        )
                        break

            # Save checkpoint
            if self.checkpoint_dir is not None:
                self.save_checkpoint(epoch, is_best=False)

        logger.info("VAE training completed")

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model.
        """
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }

        filename = "vae_best.pt" if is_best else f"vae_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / filename
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
