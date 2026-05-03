"""Main contrastive learning pipeline."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    CIFAR10LTContrastiveDataset,
    CreditCardFraudDataset,
)
from src.data.imbalance import analyze_class_distribution, get_class_weights, print_class_distribution
from src.data.transforms import get_cifar10_transform, get_creditcard_transform
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.models.classifier import LinearClassifier
from src.models.encoder import ContrastiveEncoder, MLPEncoder
from src.training.train_classifier import train_classifier
from src.training.train_contrastive import train_contrastive_encoder
from src.utils.config import Config
from src.utils.logger import get_logger, setup_logger
from src.utils.seed import set_seed


def run_contrastive_pipeline(
    config: Config = None,
) -> Dict[str, float]:
    """
    Run complete contrastive learning pipeline.

    Args:
        config: Configuration object.

    Returns:
        Dictionary of evaluation metrics.
    """
    if config is None:
        config = Config()

    # Setup
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    # Setup directories
    output_dir = config.output_dir
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("pipeline", log_dir)
    logger.info(f"Starting contrastive learning pipeline on device: {device}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Output directory: {output_dir}")

    # ============ Load Data ============
    logger.info("Loading datasets...")

    if config.data.dataset_name == "cifar10-lt":
        # Load CIFAR10-LT
        train_transform = get_cifar10_transform(
            train=True,
            image_size=config.data.image_size,
            horizontal_flip=config.augmentation.horizontal_flip,
            crop_padding=config.augmentation.crop_padding,
        )
        test_transform = get_cifar10_transform(train=False)

        # Create augmented datasets for contrastive learning
        train_dataset_contrastive = CIFAR10LTContrastiveDataset(
            split="train",
            transform=train_transform,
            contrastive=True,
            dataset_config=config.data.cifar10_config,
        )

        val_dataset_contrastive = CIFAR10LTContrastiveDataset(
            split="val",
            transform=train_transform,
            contrastive=True,
            dataset_config=config.data.cifar10_config,
        )

        test_dataset = CIFAR10LTContrastiveDataset(
            split="test",
            transform=test_transform,
            dataset_config=config.data.cifar10_config,
        )

    elif config.data.dataset_name == "credit-card-fraud":
        # Load Credit Card Fraud Detection (no augmentation for tabular data)
        train_transform = get_creditcard_transform()
        test_transform = get_creditcard_transform()

        # Create augmented datasets for contrastive learning
        train_dataset_contrastive = CreditCardFraudDataset(
            split="train",
            transform=train_transform,
            contrastive=True,
            normalize=True,
        )

        val_dataset_contrastive = CreditCardFraudDataset(
            split="val",
            transform=train_transform,
            contrastive=True,
            normalize=True,
        )

        test_dataset = CreditCardFraudDataset(
            split="test",
            transform=test_transform,
            normalize=True,
        )

        # Update num_classes
        config.data.num_classes = 2

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")

    # Analyze class distribution
    class_counts = analyze_class_distribution(train_dataset_contrastive)
    print_class_distribution(class_counts)

    # Plot class distribution
    plot_class_distribution(
        class_counts,
        save_path=figure_dir / "class_distribution.png",
    )

    # ---- Class weights ----
    class_weights = get_class_weights(class_counts, config.data.num_classes)

    # Stratified split
    from torch.utils.data import WeightedRandomSampler

    # ---- Sampler ONLY for train_subset ----
    train_labels = train_dataset_contrastive.labels

    sample_weights = torch.tensor(
        [class_weights[label] for label in train_labels],
        dtype=torch.float
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset_contrastive,
        batch_size=config.contrastive_training.batch_size,
        sampler=sampler, 
        num_workers=config.data.num_workers,
    )

    val_loader = DataLoader(
        val_dataset_contrastive,
        batch_size=config.contrastive_training.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.eval_batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
    )

    # ============ Train Contrastive Encoder ============
    logger.info("\n" + "="*50)
    logger.info("Stage 1: Training Contrastive Encoder")
    logger.info("="*50)

    # Create encoder based on dataset type
    if config.data.dataset_name == "cifar10-lt":
        encoder = ContrastiveEncoder(
            architecture=config.encoder.architecture,
            projection_dim=config.encoder.projection_dim,
            hidden_dim=config.encoder.hidden_dim,
            num_layers=config.encoder.num_layers,
            use_bn=config.encoder.use_bn,
        )
        logger.info("Using ContrastiveEncoder (CNN) for image data")

    elif config.data.dataset_name == "credit-card-fraud":
        encoder = MLPEncoder(
            input_dim=30,
            hidden_dim=config.encoder.hidden_dim,
            projection_dim=config.encoder.projection_dim,
            num_layers=config.encoder.num_layers,
            use_bn=config.encoder.use_bn,
            dropout=0.05,
        )
        logger.info("Using MLPEncoder for tabular data")

    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")

    logger.info(f"Encoder parameters: {encoder.count_parameters():,}")

    contrastive_checkpoint_dir = checkpoint_dir / "contrastive"
    contrastive_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_contrastive = train_contrastive_encoder(
        model=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.contrastive_training.num_epochs,
        learning_rate=config.contrastive_training.learning_rate,
        weight_decay=config.contrastive_training.weight_decay,
        momentum=config.contrastive_training.momentum,
        temperature=config.contrastive_loss.temperature,
        warmup_epochs=config.contrastive_training.warmup_epochs,
        cosine_annealing=config.contrastive_training.cosine_annealing,
        device=device,
        checkpoint_dir=contrastive_checkpoint_dir,
        log_dir=log_dir,
    )

    # Plot contrastive training curves
    plot_training_curves(
        history_contrastive,
        save_path=figure_dir / "contrastive_training.png",
    )

    # ============ Train Downstream Classifier ============
    logger.info("\n" + "="*50)
    logger.info("Stage 2: Training Downstream Classifier")
    logger.info("="*50)

    # Freeze encoder and create classifier
    encoder.freeze()

    # Get representation dimension
    representation_dim = encoder.backbone_dim

    # Create classifier
    classifier = LinearClassifier(
        input_dim=representation_dim,
        num_classes=config.data.num_classes,
        hidden_dim=config.classifier.hidden_dim,
        num_layers=config.classifier.num_layers,
    )

    logger.info(f"Classifier parameters: {classifier.count_parameters():,}")

    # Create feature extractor (encoder without projection head)
    class FeatureExtractor(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x):
            representation, _ = self.encoder(x)
            return representation

    feature_extractor = FeatureExtractor(encoder).to(device)

    # Create datasets for downstream task
    logger.info("Creating downstream task datasets...")

    # Get features from contrastive encoder
    train_feature_list = []
    train_label_list = []

    val_feature_list = []
    val_label_list = []

    # Because the contrastive dataset returns ((x_i, x_j), label), 
    # we can either use x_i or x_j for the classifier training
    feature_extractor.eval()
    with torch.no_grad():
        # Extract features for train sets  
        for (x_i, _), labels in train_loader:
            images = x_i.to(device)
            labels = labels.to(device)

            features = feature_extractor(images)

            train_feature_list.append(features.cpu())
            train_label_list.append(labels.cpu())

        train_features = torch.cat(train_feature_list, dim=0)
        train_labels = torch.cat(train_label_list, dim=0)

        # Extract features for val sets
        for (x_i, _), labels in val_loader:
            images = x_i.to(device)
            labels = labels.to(device)

            features = feature_extractor(images)

            val_feature_list.append(features.cpu())
            val_label_list.append(labels.cpu())

        val_features = torch.cat(val_feature_list, dim=0)
        val_labels = torch.cat(val_label_list, dim=0)

    # Create simple dataset from features
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]


    feature_train_dataset = FeatureDataset(train_features, train_labels)
    feature_val_dataset = FeatureDataset(val_features, val_labels)

    feature_train_loader = DataLoader(
        feature_train_dataset,
        batch_size=config.classifier_training.batch_size,
        num_workers=0,
        shuffle=True,
    )

    feature_val_loader = DataLoader(
        feature_val_dataset,
        batch_size=config.classifier_training.batch_size,
        num_workers=0,
        shuffle=False,
    )

    classifier_checkpoint_dir = checkpoint_dir / "classifier"
    classifier_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_classifier = train_classifier(
        model=classifier,
        train_loader=feature_train_loader,
        val_loader=feature_val_loader,
        num_epochs=config.classifier_training.num_epochs,
        learning_rate=config.classifier_training.learning_rate,
        weight_decay=config.classifier_training.weight_decay,
        momentum=config.classifier_training.momentum,
        device=device,
        checkpoint_dir=classifier_checkpoint_dir,
        log_dir=log_dir,
        patience=config.classifier_training.patience,
        class_weights=class_weights,
    )

    # Plot classifier training curves
    plot_training_curves(
        history_classifier,
        save_path=figure_dir / "classifier_training.png",
    )

    # ============ Evaluation ============
    logger.info("\n" + "="*50)
    logger.info("Stage 3: Evaluation")
    logger.info("="*50)

    # Load best classifier checkpoint
    best_classifier_checkpoint = classifier_checkpoint_dir / "best_model.pt"
    if best_classifier_checkpoint.exists():
        checkpoint = torch.load(best_classifier_checkpoint, map_location=device)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best classifier from {best_classifier_checkpoint}")

    # Extract test features
    test_features = []
    test_labels = []

    feature_extractor.eval()
    with torch.no_grad():
         for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            features = feature_extractor(images)    

            test_features.append(features.cpu())
            test_labels.append(labels.cpu())

    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Evaluate classifier on test features
    classifier.to(device)
    classifier.eval()
    test_logits = classifier(test_features.to(device))

    # Calculate metrics
    metrics_calculator = ClassificationMetrics(config.data.num_classes)
    metrics_calculator.update(test_logits, test_labels.to(device))

    overall_metrics = metrics_calculator.compute()
    per_class_metrics = metrics_calculator.compute_per_class()

    logger.info("\nTest Results:")
    for metric_name, metric_value in overall_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")

    logger.info("\nPer-class metrics:")
    for class_id, metrics in per_class_metrics.items():
        logger.info(
            f"\nClass {class_id}: "
            f"- Precision = {metrics['precision']:.4f}, "
            f"- Recall = {metrics['recall']:.4f}, "
            f"- F1 = {metrics['f1']:.4f}"
        )

    # Get confusion matrix
    confusion_mat = metrics_calculator.get_confusion_matrix()

    # Plot confusion matrix
    plot_confusion_matrix(
        confusion_mat,
        save_path=figure_dir / "confusion_matrix.png",
    )

    logger.info("\n" + "="*50)
    logger.info("Pipeline completed!")
    logger.info("="*50)

    return overall_metrics
