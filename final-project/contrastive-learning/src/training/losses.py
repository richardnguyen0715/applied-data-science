"""Contrastive loss functions."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-Scaled Cross Entropy) loss."""

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter for scaling.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of samples.

        Args:
            z_i: First view embeddings (B, dim).
            z_j: Second view embeddings (B, dim).

        Returns:
            Loss value (scalar).
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2B, dim)

        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Create labels: (0, 1), (1, 0), (2, 3), (3, 2), ...
        labels = torch.arange(batch_size, dtype=torch.long, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Create mask: True for positive pairs, False for negative pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        # Set diagonal to False (remove self-pairs)
        mask.fill_diagonal_(False)

        # Get positive pairs
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            pos_mask[i, batch_size + i] = True
            pos_mask[batch_size + i, i] = True

        # Remove self-pairs from similarity
        similarity_without_diag = similarity.clone()
        similarity_without_diag.fill_diagonal_(-float("inf"))

        # Compute loss
        pos_sim = similarity[pos_mask].view(2 * batch_size, 1)  # (2B, 1)
        neg_sim = similarity_without_diag  # (2B, 2B)

        # NT-Xent loss
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (2B, 2B)
        labels_nt = torch.zeros(2 * batch_size, dtype=torch.long, device=device)  # All zeros (pos is first)

        loss = F.cross_entropy(logits, labels_nt)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature: float = 0.07) -> None:
        """
        Initialize Supervised Contrastive loss.

        Args:
            temperature: Temperature parameter for scaling.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            z_i: First view embeddings (B, dim).
            z_j: Second view embeddings (B, dim).
            labels: Class labels (B,).

        Returns:
            Loss value (scalar).
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # (2B, dim)

        # Concatenate labels
        if labels is not None:
            labels = torch.cat([labels, labels], dim=0)  # (2B,)

        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Remove self-pairs
        similarity_without_diag = similarity.clone()
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_without_diag.masked_fill_(mask, -float("inf"))

        # Compute loss per sample
        loss = 0.0
        for i in range(2 * batch_size):
            # Positive pairs (same label, different views)
            if labels is not None:
                pos_mask = (labels == labels[i]) & (~mask[i])
                neg_mask = (labels != labels[i]) & (~mask[i])
            else:
                # If no labels, use original NT-Xent logic
                if i < batch_size:
                    pos_mask = torch.zeros(2 * batch_size, dtype=torch.bool, device=device)
                    pos_mask[batch_size + i] = True
                    neg_mask = ~pos_mask & (~mask[i])
                else:
                    pos_mask = torch.zeros(2 * batch_size, dtype=torch.bool, device=device)
                    pos_mask[i - batch_size] = True
                    neg_mask = ~pos_mask & (~mask[i])

            pos_sim = similarity[i, pos_mask].sum()
            neg_sim = torch.exp(similarity[i, neg_mask]).sum()

            if pos_mask.sum() > 0:
                loss -= torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sim))

        loss = loss / (2 * batch_size)

        return loss


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for classification."""

    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        """
        Initialize classification loss.

        Args:
            weight: Class weights for handling imbalance.
        """
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            logits: Predicted logits (B, num_classes).
            labels: Ground truth labels (B,).

        Returns:
            Loss value (scalar).
        """
        return self.loss(logits, labels)
