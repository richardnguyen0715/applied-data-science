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
        else:
            raise ValueError("SupConLoss requires labels")

        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Numerical stability: subtract max
        similarity_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - similarity_max.detach()

        # Create mask for self-pairs
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)

        # Create positive pairs mask: same label, not self
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (2B, 2B)
        pos_mask = labels_eq & (~self_mask)

        # Compute exp(logits) with self-pairs masked out
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (~self_mask).float()

        # Compute log(sum(exp(all))) for denominator
        log_sum_exp_all = torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Compute log probability for positive pairs
        log_prob = logits - log_sum_exp_all

        # Number of positive samples for each anchor i
        num_pos = pos_mask.sum(dim=1)  # Shape: (N,)

        # Sum of log-probabilities over positive pairs for each anchor
        # Broadcasting: mask keeps only positive entries, others become 0
        sum_log_prob_pos = (log_prob * pos_mask).sum(dim=1)  # Shape: (N,)

        # Compute mean log-probability over positives for each anchor
        # clamp(min=1) avoids division by zero (for anchors with no positives)
        mean_log_prob_pos = sum_log_prob_pos / num_pos.clamp(min=1)

        # Identify anchors that actually have at least one positive
        valid_mask = num_pos > 0

        # Final loss
        loss = -mean_log_prob_pos[valid_mask].mean()

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
