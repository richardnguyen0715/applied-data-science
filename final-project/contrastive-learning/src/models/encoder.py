"""Contrastive encoder models."""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base import BaseModel


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_bn: bool = True,
    ) -> None:
        """
        Initialize projection head.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension (projection dimension).
            num_layers: Number of layers.
            use_bn: Whether to use batch normalization.
        """
        super().__init__()

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class ContrastiveEncoder(BaseModel):
    """Encoder for contrastive learning on image data."""

    def __init__(
        self,
        architecture: str = "resnet18",
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        use_bn: bool = True,
    ) -> None:
        """
        Initialize contrastive encoder.

        Args:
            architecture: Backbone architecture (resnet18, resnet50).
            projection_dim: Projection dimension.
            hidden_dim: Hidden dimension of projection head.
            num_layers: Number of projection head layers.
            use_bn: Whether to use batch normalization.
        """
        super().__init__()

        # Create backbone
        if architecture == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone_dim = 512
        elif architecture == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Remove classification layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Create projection head
        self.projection_head = ProjectionHead(
            input_dim=backbone_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            num_layers=num_layers,
            use_bn=use_bn,
        )

        self.backbone_dim = backbone_dim
        self.projection_dim = projection_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Tuple of (representation, projection).
                - representation: (B, backbone_dim) - features from backbone
                - projection: (B, projection_dim) - projected features
        """
        # Backbone features
        representation = self.backbone(x)  # (B, backbone_dim, 1, 1)
        representation = representation.view(representation.size(0), -1)  # (B, backbone_dim)

        # Projection
        projection = self.projection_head(representation)  # (B, projection_dim)

        return representation, projection

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get representation without projection.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            Representation tensor (B, backbone_dim).
        """
        representation, _ = self.forward(x)
        return representation


class MLPEncoder(BaseModel):
    """MLP Encoder for contrastive learning on tabular data."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        num_layers: int = 3,
        use_bn: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize MLP encoder for tabular data.

        Args:
            input_dim: Input dimension (number of features).
            hidden_dim: Hidden dimension.
            projection_dim: Projection dimension.
            num_layers: Number of hidden layers in backbone.
            use_bn: Whether to use batch normalization.
            dropout: Dropout rate.
        """
        super().__init__()

        # Create backbone
        backbone_layers = []
        backbone_layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            backbone_layers.append(nn.BatchNorm1d(hidden_dim))
        backbone_layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            backbone_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            backbone_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                backbone_layers.append(nn.BatchNorm1d(hidden_dim))
            backbone_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                backbone_layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*backbone_layers)

        # Create projection head
        self.projection_head = ProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            num_layers=2,
            use_bn=use_bn,
        )

        self.backbone_dim = hidden_dim
        self.projection_dim = projection_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, input_dim).

        Returns:
            Tuple of (representation, projection).
                - representation: (B, backbone_dim) - features from backbone
                - projection: (B, projection_dim) - projected features
        """
        # Backbone features
        representation = self.backbone(x)  # (B, hidden_dim)

        # Projection
        projection = self.projection_head(representation)  # (B, projection_dim)

        return representation, projection

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get representation without projection.

        Args:
            x: Input tensor (B, input_dim).

        Returns:
            Representation tensor (B, backbone_dim).
        """
        representation, _ = self.forward(x)
        return representation
