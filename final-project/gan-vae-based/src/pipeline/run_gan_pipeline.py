"""GAN-based oversampling pipeline."""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import get_train_val_datasets
from src.data.imbalance import get_samples_needed, identify_minority_classes
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualize import (
    plot_class_distribution,
    plot_comparison_distributions,
    plot_generated_images,
    plot_training_curves,
)
from src.models.classifier import create_classifier
from src.models.gan import ConditionalDiscriminator, ConditionalGenerator
from src.training.train_classifier import ClassifierTrainer
from src.training.train_gan import GANTrainer
from src.utils.config import PipelineConfig, save_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_gan_pipeline(config: PipelineConfig) -> dict:
    """
    Run GAN-based oversampling pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Dictionary of results.
    """
    # Setup
    set_seed(config.random_seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    output_dir = config.output_dir / "gan"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    logger_gan = get_logger(__name__, log_dir=log_dir)
    logger_gan.info("Starting GAN pipeline")

    # Save config
    save_config(config, output_dir / "config.yaml")

    # Load data
    logger_gan.info("Loading CIFAR-10-LT dataset")
    train_dataset, val_dataset, test_dataset = get_train_val_datasets(
        dataset_name=config.data.dataset_name,
        config_name=config.data.config_name,
        val_split=config.data.val_split,
        image_size=config.data.image_size,
    )

    # Extract targets from raw dataset without transforms
    train_targets = np.array(train_dataset.get_targets())
    fig_dir = output_dir / "figures"

    # Plot original distribution
    plot_class_distribution(
        train_targets,
        title="CIFAR-10-LT Original Distribution",
        output_path=fig_dir / "class_distribution_original.png",
    )

    # Create dataloaders for GAN training
    # Note: num_workers=0 required because HuggingFace datasets don't serialize
    # correctly across multiprocessing boundaries
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create GAN models
    logger_gan.info("Creating Conditional GAN")
    generator = ConditionalGenerator(
        latent_dim=config.gan.latent_dim,
        num_classes=10,
        hidden_dim=config.gan.generator_hidden_dim,
    )
    discriminator = ConditionalDiscriminator(
        num_classes=10,
        hidden_dim=config.gan.discriminator_hidden_dim,
    )

    # Train GAN
    checkpoint_dir = output_dir / "checkpoints"
    trainer_gan = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        config=vars(config.gan),
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    logger_gan.info("Training GAN...")
    trainer_gan.fit(train_loader)

    # Generate synthetic samples for minority classes
    logger_gan.info("Generating synthetic samples")
    minority_classes = identify_minority_classes(train_targets, threshold=0.5)
    samples_needed = get_samples_needed(train_targets, target_ratio=1.0)

    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for class_idx in minority_classes:
            if class_idx in samples_needed:
                num_samples = samples_needed[class_idx]
                z = torch.randn(num_samples, config.gan.latent_dim).to(device)
                labels = torch.full((num_samples,), class_idx, dtype=torch.long).to(device)

                generated = generator(z, labels)
                synthetic_images.extend(generated.cpu().numpy())
                synthetic_labels.extend([class_idx] * num_samples)

    synthetic_images = np.array(synthetic_images)
    synthetic_labels = np.array(synthetic_labels)

    logger_gan.info(f"Generated {len(synthetic_images)} synthetic samples")

    # Plot generated images
    plot_generated_images(
        synthetic_images[:16],
        synthetic_labels[:16],
        title="Generated Samples (GAN)",
        output_path=fig_dir / "generated_images_gan.png",
    )

    # Combine original and synthetic data
    train_images, _ = train_dataset.get_images_and_targets()
    combined_images = np.vstack([train_images, synthetic_images])
    combined_labels = np.hstack([train_targets, synthetic_labels])

    # Plot comparison
    plot_comparison_distributions(
        train_targets,
        combined_labels,
        output_path=fig_dir / "distribution_comparison_gan.png",
    )

    # Create combined dataset
    combined_images_torch = torch.from_numpy(combined_images).float()
    combined_labels_torch = torch.from_numpy(combined_labels).long()
    combined_dataset = TensorDataset(combined_images_torch, combined_labels_torch)

    # Create dataloaders
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Train classifier on combined data
    logger_gan.info("Training classifier on balanced data")
    classifier = create_classifier(
        architecture=config.classifier.architecture,
        num_classes=10,
        pretrained=False,
    )

    trainer_classifier = ClassifierTrainer(
        model=classifier,
        config=vars(config.classifier),
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    trainer_classifier.fit(combined_loader, val_loader)

    # Plot training curves
    plot_training_curves(
        trainer_classifier.train_losses,
        trainer_classifier.val_losses,
        title="Classifier Training (GAN-based)",
        output_path=fig_dir / "training_curves_classifier_gan.png",
    )

    # Evaluate
    logger_gan.info("Evaluating classifier")
    evaluator = Evaluator(classifier, device, num_classes=10)
    test_metrics = evaluator.evaluate(test_loader)

    evaluator.log_metrics(test_metrics)
    evaluator.save_metrics(test_metrics, output_dir / "test_metrics.txt")

    results = {
        'method': 'gan',
        'accuracy': test_metrics['accuracy'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'macro_f1': test_metrics['macro_f1'],
        'metrics': test_metrics,
        'model_path': checkpoint_dir / "classifier_best.pt",
        'num_synthetic_samples': len(synthetic_images),
    }

    logger_gan.info("GAN pipeline completed")
    return results
