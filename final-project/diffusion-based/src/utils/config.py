"""Configuration management using YAML files."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DiffusionConfig:
    """Configuration for diffusion model training and sampling."""

    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # linear or cosine
    model_channels: int = 64
    num_residual_blocks: int = 2
    attention_resolutions: tuple = (8, 16)
    num_classes: int = 10
    class_embed_dim: int = 128
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    num_samples_per_class: int = 500
    sample_steps: int = 1000
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class ClassifierConfig:
    """Configuration for classifier model training."""

    num_classes: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 200
    weight_decay: float = 1e-4
    model_name: str = "resnet18"
    num_workers: int = 4


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_name: str = "tomas-gajarsky/cifar10-lt"
    dataset_config: str = "r-20"
    num_classes: int = 10
    image_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    seed: int = 42
    device: str = "cuda"
    num_gpus: int = 1
    mixed_precision: bool = False
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 10
    checkpoint_dir: Path = Path("outputs/checkpoints")
    log_dir: Path = Path("outputs/logs")
    figure_dir: Path = Path("outputs/figures")


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict if config_dict is not None else {}


def save_config_to_yaml(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save YAML configuration file.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(
    diffusion_cfg: Optional[DiffusionConfig] = None,
    classifier_cfg: Optional[ClassifierConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    training_cfg: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Merge multiple configuration dataclasses into a single dictionary.

    Args:
        diffusion_cfg: Diffusion model configuration.
        classifier_cfg: Classifier model configuration.
        data_cfg: Data loading configuration.
        training_cfg: Overall training configuration.

    Returns:
        Merged configuration dictionary.
    """
    config = {}
    if diffusion_cfg is not None:
        config["diffusion"] = asdict(diffusion_cfg)
    if classifier_cfg is not None:
        config["classifier"] = asdict(classifier_cfg)
    if data_cfg is not None:
        config["data"] = asdict(data_cfg)
    if training_cfg is not None:
        config["training"] = asdict(training_cfg)
    return config
