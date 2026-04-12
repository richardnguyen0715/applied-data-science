"""Training logic for classifier."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """Trainer for image classifier."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize classifier trainer.

        Args:
            model: Classifier model.
            config: Configuration dictionary.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.lr = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.momentum = config.get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.patience = config.get('patience', 20)  # Early stopping patience

        # Optimizer
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
        )

        # Metrics
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train classifier for one epoch.

        Args:
            train_loader: Training dataloader.

        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=False)
        for batch_idx, (images, labels) in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate classifier.

        Args:
            val_loader: Validation dataloader.

        Returns:
            Tuple of (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Train classifier for multiple epochs.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        """
        logger.info(f"Starting classifier training for {self.epochs} epochs")
        logger.info(f"Early stopping patience: {self.patience} epochs")
        best_val_acc = 0.0

        for epoch in tqdm(range(self.epochs), desc="Classifier Training", unit="epoch"):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save best model and check early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                if self.checkpoint_dir is not None:
                    self.save_checkpoint(epoch, is_best=True)
                logger.info(f"Best model updated with val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping triggered after {self.patience} epochs without improvement. "
                        f"Best model at epoch {self.best_epoch} with val loss: {self.best_val_loss:.4f}"
                    )
                    break

            # Update learning rate
            self.scheduler.step()

        logger.info("Classifier training completed")

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

        filename = "classifier_best.pt" if is_best else f"classifier_epoch_{epoch}.pt"
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
