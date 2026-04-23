"""Data transformation and augmentation for contrastive learning."""

import torchvision.transforms as transforms


def get_cifar10_augmentation(
    image_size: int = 32,
    horizontal_flip: bool = True,
    crop_padding: int = 4,
) -> transforms.Compose:
    """
    Get minimal contrastive learning augmentation pipeline for CIFAR-10.
    
    Only applies:
    - RandomCrop with padding
    - RandomHorizontalFlip
    
    Args:
        image_size: Size of the image.
        horizontal_flip: Whether to apply horizontal flip.
        crop_padding: Padding for random crop.
        
    Returns:
        Augmentation pipeline.
    """
    augmentation = [
        transforms.RandomCrop(image_size, padding=crop_padding),
        transforms.RandomHorizontalFlip(p=0.5 if horizontal_flip else 0.0),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ]
    
    return transforms.Compose(augmentation)


def get_creditcard_transform():
    """
    Get transformation for credit card fraud dataset (tabular data).
    
    For tabular data, no transformation needed as data is already normalized tensors.
    
    Returns:
        Identity transformation (returns input unchanged).
    """
    return lambda x: x


def get_base_transform(dataset_name: str = "cifar10-lt") -> transforms.Compose:
    """
    Get base transformation (no augmentation) for evaluation.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Base transformation pipeline.
    """
    if dataset_name == "cifar10-lt":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])



