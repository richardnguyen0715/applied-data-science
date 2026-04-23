"""Downstream classifier training."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseModel
from src.training.losses import ClassificationLoss
from src.utils.logger import get_logger, setup_logger


def train_classifier(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    device: torch.device = None,
    checkpoint_dir: Path = None,
    log_dir: Path = None,
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 10,
    gradient_clip_val: float = 0.0,
    patience: int = 20,
    class_weights: Optional[torch.Tensor] = None,
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
        momentum: Momentum for SGD optimizer.
        device: Device to train on.
        checkpoint_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        log_every_n_steps: Log every n steps.
        save_every_n_epochs: Save checkpoint every n epochs.
        gradient_clip_val: Gradient clipping value.
        patience: Early stopping patience.
        class_weights: Class weights for handling imbalance.

    Returns:
        Dictionary with training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log_dir is not None:
        logger = setup_logger("classifier_training", log_dir)
    else:
        logger = get_logger("classifier_training")

    model = model.to(device)

    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Setup loss
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = ClassificationLoss(weight=class_weights)

    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }

    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(x)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            num_batches += 1

            if (batch_idx + 1) % log_every_n_steps == 0:
                avg_loss = train_loss / num_batches
                avg_acc = 100 * train_correct / train_total
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

        avg_train_loss = train_loss / num_batches
        avg_train_acc = 100 * train_correct / train_total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Train Acc: {avg_train_acc:.2f}% - "
            f"LR: {current_lr:.6f}"
        )

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    x, labels = batch
                    x = x.to(device)
                    labels = labels.to(device)

                    logits = model(x)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            avg_val_acc = 100 * val_correct / val_total
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)

            logger.info(
                f"Val Loss: {avg_val_loss:.4f} - "
                f"Val Acc: {avg_val_acc:.2f}%"
            )

            # Save checkpoint if val acc improved
            if checkpoint_dir is not None:
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    no_improve_epochs = 0

                    checkpoint_path = checkpoint_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": avg_val_acc,
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
                else:
                    no_improve_epochs += 1

                # Early stopping
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Save checkpoint periodically
        if checkpoint_dir is not None and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)

    return history
