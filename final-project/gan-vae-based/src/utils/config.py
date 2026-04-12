"""Configuration loading and management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""

    dataset_name: str = "tomas-gajarsky/cifar10-lt"
    config_name: str = "r-20"
    batch_size: int = 32
    val_split: float = 0.1
    num_workers: int = 4
    image_size: int = 32


@dataclass
class GANConfig:
    """Configuration for GAN training."""

    latent_dim: int = 100
    num_classes: int = 10
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 50
    critic_iterations: int = 5
    generator_hidden_dim: int = 128
    discriminator_hidden_dim: int = 128


@dataclass
class VAEConfig:
    """Configuration for VAE training."""

    latent_dim: int = 64
    num_classes: int = 10
    learning_rate: float = 0.001
    epochs: int = 50
    kld_weight: float = 0.00025
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256


@dataclass
class ClassifierConfig:
    """Configuration for classifier training."""

    learning_rate: float = 0.001
    epochs: int = 100
    momentum: float = 0.9
    weight_decay: float = 5e-4
    architecture: str = "resnet18"


@dataclass
class PipelineConfig:
    """Configuration for end-to-end pipelines."""

    random_seed: int = 42
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    oversampling_ratio: float = 1.0
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "random_seed": self.random_seed,
            "device": self.device,
            "data": self.data.__dict__,
            "gan": self.gan.__dict__,
            "vae": self.vae.__dict__,
            "classifier": self.classifier.__dict__,
            "oversampling_ratio": self.oversampling_ratio,
            "output_dir": str(self.output_dir),
        }


def load_config(config_path: Path) -> PipelineConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        PipelineConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return _dict_to_config(config_dict)


def _dict_to_config(config_dict: Dict[str, Any]) -> PipelineConfig:
    """Convert dictionary to PipelineConfig."""
    data_dict = config_dict.get("data", {})
    gan_dict = config_dict.get("gan", {})
    vae_dict = config_dict.get("vae", {})
    classifier_dict = config_dict.get("classifier", {})

    return PipelineConfig(
        random_seed=config_dict.get("random_seed", 42),
        device=config_dict.get("device", "cuda"),
        data=DataConfig(**data_dict),
        gan=GANConfig(**gan_dict),
        vae=VAEConfig(**vae_dict),
        classifier=ClassifierConfig(**classifier_dict),
        oversampling_ratio=config_dict.get("oversampling_ratio", 1.0),
        output_dir=Path(config_dict.get("output_dir", "outputs")),
    )


def save_config(config: PipelineConfig, output_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object.
        output_path: Path to save YAML file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
