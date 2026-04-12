"""Classifier for evaluating oversampling methods."""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleConvNet(nn.Module):
    """Simple convolutional network for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        """
        Initialize simple CNN.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, 32, 32).

        Returns:
            Class predictions (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_classifier(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """
    Create classifier model.

    Args:
        architecture: Model architecture ('resnet18', 'simple_cnn').
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights (for ResNet).

    Returns:
        Classifier model.

    Raises:
        ValueError: If architecture is unknown.
    """
    if architecture == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Modify first conv layer for 32x32 images (no stride in conv1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Replace final classification layer
        model.fc = nn.Linear(512, num_classes)
        return model
    elif architecture == "simple_cnn":
        return SimpleConvNet(num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
