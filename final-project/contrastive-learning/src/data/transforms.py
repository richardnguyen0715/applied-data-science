"""Data transformation and augmentation for contrastive learning."""

import torchvision.transforms as transforms


def get_cifar10_transform(
    train: bool = True,
    image_size: int = 32,
    horizontal_flip: bool = True,
    crop_padding: int = 4,
) -> transforms.Compose:
    """
    Get CIFAR-10 transformation pipeline for train/test.

    Args:
        train: Whether to use training augmentation.
        image_size: Size of the image.
        horizontal_flip: Whether to apply horizontal flip (train only).
        crop_padding: Padding for random crop (train only).

    Returns:
        Transformation pipeline.
    """
    if train:
        transform_list = [
            transforms.RandomCrop(image_size, padding=crop_padding),
            transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0.0),
        ]
    else:
        transform_list = []

    transform_list.extend([
        transforms.ToTensor(),
    ])

    return transforms.Compose(transform_list)


def get_creditcard_transform() -> transforms.Compose:
    """
    Get transformation for credit card fraud dataset (tabular data).
    
    For tabular data, no transformation needed as data is already normalized tensors.
    
    Returns:
        Identity transformation (returns input unchanged).
    """
    return transforms.Compose([])

