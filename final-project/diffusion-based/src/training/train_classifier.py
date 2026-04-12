"""Classifier training."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseModel
from src.training.losses import ClassifierLoss
from src.utils.logger import get_logger, setup_logger


def train_classifier(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    checkpoint_dir: Path = None,
    log_dir: Path = None,
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 10,
    gradient_clip_val: float = 0.0,
    patience: int = 30,
) -> Dict[str, list]:
    """
    Train classifier model.

    Args:
        model: Classifier model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (optional).
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.
        device: Device to train on.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        log_every_n_steps: Log every n steps.
        save_every_n_epochs: Save checkpoint every n epochs.
        gradient_clip_val: Gradient clipping value.
        patience: Early stopping patience (number of epochs without improvement before stopping).

    Returns:
        Dictionary with training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    if log_dir is not None:
        logger = setup_logger("classifier", log_dir)
    else:
        logger = get_logger("classifier")

    # Setup checkpoint directory
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Move model to device
    model = model.to(device)

    # Optimizer with learning rate scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss function
    criterion = ClassifierLoss()

    # Training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch": [],
    }
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()

            # Accumulate statistics
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            num_batches += 1

            # Log
            if (batch_idx + 1) % log_every_n_steps == 0:
                avg_loss = train_loss / num_batches
                avg_acc = 100.0 * train_correct / train_total
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Step {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {avg_acc:.2f}%"
                )

        # Epoch statistics
        train_loss /= num_batches
        train_acc = 100.0 * train_correct / train_total

        # Validation phase
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # Save best model
            if checkpoint_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                checkpoint_path = checkpoint_dir / "classifier_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"Best model saved to {checkpoint_path} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                )
            else:
                if val_loader is not None:
                    patience_counter += 1

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            # Check early stopping
            if val_loader is not None and patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs without improvement. "
                    f"Best model at epoch {best_epoch} with val loss: {best_val_loss:.4f}"
                )
                break

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
            )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["epoch"].append(epoch + 1)

        # Learning rate scheduling
        scheduler.step()

        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"classifier_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    logger.info(f"Training completed! Early stopping: {patience_counter >= patience if val_loader else 'N/A'}")
    return history
