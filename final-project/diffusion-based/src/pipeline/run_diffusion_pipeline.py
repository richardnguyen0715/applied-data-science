"""Diffusion-based pipeline for handling imbalanced data."""

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CIFAR10LTDataset, create_data_loaders
from src.data.imbalance import analyze_class_distribution, plot_class_distribution, print_class_distribution
from src.data.transforms import get_train_transforms, get_test_transforms, get_no_normalizing_transforms
from src.evaluation.evaluator import evaluate_classifier
from src.evaluation.visualize import plot_training_curves, plot_generated_samples
from src.models.classifier import create_classifier
from src.models.diffusion import DiffusionModel
from src.pipeline.sampling import sample_from_diffusion, create_balanced_dataset
from src.training.train_classifier import train_classifier
from src.training.train_diffusion import train_diffusion
from src.utils.config import (
    DiffusionConfig,
    ClassifierConfig,
    DataConfig,
    TrainingConfig,
)
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def run_diffusion_pipeline(
    output_dir: Path = Path("outputs"),
    diffusion_config: Optional[DiffusionConfig] = None,
    classifier_config: Optional[ClassifierConfig] = None,
    data_config: Optional[DataConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run full diffusion-based pipeline.

    Args:
        output_dir: Output directory.
        diffusion_config: Diffusion model configuration.
        classifier_config: Classifier configuration.
        data_config: Data configuration.
        training_config: Training configuration.
        device: Device to use.

    Returns:
        Dictionary of evaluation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if diffusion_config is None:
        diffusion_config = DiffusionConfig()
    if classifier_config is None:
        classifier_config = ClassifierConfig()
    if data_config is None:
        data_config = DataConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # Setup directories
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs" / "diffusion"
    diff_checkpoint_dir = output_dir / "checkpoints" / "diffusion"
    clf_checkpoint_dir = output_dir / "checkpoints" / "classifier"
    figure_dir = output_dir / "figures" / "diffusion"

    log_dir.mkdir(parents=True, exist_ok=True)
    diff_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    clf_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("diffusion_pipeline", log_dir)

    # Set seed
    set_seed(training_config.seed)

    logger.info("Starting diffusion-based pipeline...")
    logger.info(f"Using device: {device}")

    # ==================== LOAD DATA ====================
    logger.info("Loading training data...")
    train_dataset = CIFAR10LTDataset(
        split="train",
        transform=get_no_normalizing_transforms(data_config.image_size),
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
    )
    train_loader = create_data_loaders(
        train_dataset,
        batch_size=diffusion_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        shuffle=True,
    )

    # Analyze class distribution
    class_dist = analyze_class_distribution(
        [train_dataset.labels[i] for i in range(len(train_dataset))]
    )
    logger.info("Original training data class distribution:")
    print_class_distribution(class_dist)

    # Plot original distribution
    plot_class_distribution(
        class_dist,
        title="Original CIFAR-10-LT Class Distribution",
        save_path=str(figure_dir / "original_distribution.png"),
    )

    # Load test data
    logger.info("Loading test data...")
    test_dataset = CIFAR10LTDataset(
        split="test",
        transform=get_test_transforms(data_config.image_size),
        dataset_name=data_config.dataset_name,
        dataset_config=data_config.dataset_config,
    )
    test_loader = create_data_loaders(
        test_dataset,
        batch_size=classifier_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        shuffle=False,
    )

    # ==================== TRAIN DIFFUSION MODEL ====================
    logger.info("Creating diffusion model...")
    diffusion_model = DiffusionModel(
        num_timesteps=diffusion_config.num_timesteps,
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end,
        beta_schedule=diffusion_config.beta_schedule,
        model_channels=diffusion_config.model_channels,
        num_residual_blocks=diffusion_config.num_residual_blocks,
        attention_resolutions=diffusion_config.attention_resolutions,
        num_classes=data_config.num_classes,
        class_embed_dim=diffusion_config.class_embed_dim,
        use_ema=diffusion_config.use_ema,
        ema_decay=diffusion_config.ema_decay,
    )
    logger.info(f"Diffusion model parameters: {diffusion_model.count_parameters():,}")

    logger.info("Training diffusion model...")
    diff_history = train_diffusion(
        model=diffusion_model,
        train_loader=train_loader,
        num_epochs=diffusion_config.num_epochs,
        learning_rate=diffusion_config.learning_rate,
        device=device,
        checkpoint_dir=diff_checkpoint_dir,
        log_dir=log_dir,
        log_every_n_steps=training_config.log_every_n_steps,
        save_every_n_epochs=training_config.save_every_n_epochs,
        gradient_clip_val=training_config.gradient_clip_val,
    )

    # Plot diffusion training curves
    plot_training_curves(
        diff_history,
        save_path=figure_dir / "diffusion_training.png",
    )

    # ==================== SAMPLE SYNTHETIC DATA ====================
    logger.info("Sampling synthetic data from diffusion model...")
    synthetic_images, synthetic_labels = sample_from_diffusion(
        diffusion_model,
        num_samples_per_class=diffusion_config.num_samples_per_class,
        image_size=data_config.image_size,
        device=device,
    )
    logger.info(f"Generated {len(synthetic_images)} synthetic samples")

    # Plot generated samples
    synthetic_tensor = torch.stack(synthetic_images)
    synthetic_tensor_labels = torch.tensor(synthetic_labels)
    plot_generated_samples(
        synthetic_tensor,
        synthetic_tensor_labels,
        num_samples_per_class=5,
        save_path=figure_dir / "generated_samples.png",
    )

    # ==================== CREATE BALANCED DATASET ====================
    logger.info("Creating balanced dataset...")
    balanced_dataset = create_balanced_dataset(
        train_dataset,
        synthetic_images,
        synthetic_labels,
    )
    balanced_loader = create_data_loaders(
        balanced_dataset,
        batch_size=classifier_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        shuffle=True,
    )

    # Analyze balanced distribution
    balanced_dist = analyze_class_distribution(balanced_dataset.labels)
    logger.info("Balanced dataset class distribution:")
    print_class_distribution(balanced_dist)

    # Plot balanced distribution
    plot_class_distribution(
        balanced_dist,
        title="Balanced Dataset Class Distribution",
        save_path=str(figure_dir / "balanced_distribution.png"),
    )

    # ==================== TRAIN CLASSIFIER ====================
    logger.info("Creating classifier...")
    classifier = create_classifier(
        model_name=classifier_config.model_name,
        num_classes=data_config.num_classes,
        pretrained=False,
    )
    logger.info(f"Classifier parameters: {classifier.count_parameters():,}")

    logger.info("Training classifier on balanced dataset...")
    clf_history = train_classifier(
        model=classifier,
        train_loader=balanced_loader,
        val_loader=test_loader,
        num_epochs=classifier_config.num_epochs,
        learning_rate=classifier_config.learning_rate,
        weight_decay=classifier_config.weight_decay,
        device=device,
        checkpoint_dir=clf_checkpoint_dir,
        log_dir=log_dir,
        log_every_n_steps=training_config.log_every_n_steps,
        save_every_n_epochs=training_config.save_every_n_epochs,
    )

    # Plot classifier training curves
    plot_training_curves(
        clf_history,
        save_path=figure_dir / "classifier_training.png",
    )

    # ==================== EVALUATION ====================
    logger.info("Evaluating classifier on test set...")
    overall_metrics, per_class_metrics = evaluate_classifier(
        classifier,
        test_loader,
        num_classes=data_config.num_classes,
        device=device,
    )

    logger.info("Diffusion pipeline completed!")

    return overall_metrics


if __name__ == "__main__":
    run_diffusion_pipeline()
