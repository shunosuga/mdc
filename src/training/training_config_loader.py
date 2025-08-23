#!/usr/bin/env python3
"""
Training configuration loader for ModernBERT training
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration"""

    model_name: str = "answerdotai/ModernBERT-base"
    num_labels: int = 2
    dropout: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1


@dataclass
class DataConfig:
    """Data configuration"""

    train_data_path: str
    validation_split: float = 0.2
    max_length: int = 128
    batch_size: int = 16
    shuffle_train: bool = True
    shuffle_seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration"""

    output_dir: str = "models/modernbert_data_identifier"
    num_epochs: int = 3
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01


@dataclass
class HardwareConfig:
    """Hardware configuration"""

    device: str = "auto"
    mixed_precision: bool = True
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""

    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_steps: int = 500
    save_strategy: str = "steps"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""

    logging_dir: str = "logs/modernbert_training"
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=list)
    log_level: str = "info"
    verbose: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""

    enabled: bool = False
    dropout_prob: float = 0.1


@dataclass
class AdvancedConfig:
    """Advanced training options"""

    gradient_checkpointing: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    class_weights: Optional[List[float]] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class ModernBERTTrainingConfig:
    """Complete ModernBERT training configuration"""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    def validate(self) -> None:
        """Validate configuration values"""
        # Validate data paths
        if not Path(self.data.train_data_path).exists():
            raise ValueError(
                f"Training data file not found: {self.data.train_data_path}"
            )

        # Validate data configuration
        if not 0.0 <= self.data.validation_split <= 0.5:
            raise ValueError("validation_split must be between 0.0 and 0.5")

        if self.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.data.max_length <= 0:
            raise ValueError("max_length must be positive")

        # Validate training configuration
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")

        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0.0 <= self.training.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0.0 and 1.0")

        if self.training.lr_scheduler_type not in ["linear", "cosine", "polynomial"]:
            raise ValueError(
                "lr_scheduler_type must be 'linear', 'cosine', or 'polynomial'"
            )

        # Validate hardware configuration
        if self.hardware.device not in ["auto", "cuda", "cpu"]:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")

        # Validate evaluation configuration
        if self.evaluation.eval_strategy not in ["steps", "epoch"]:
            raise ValueError("eval_strategy must be 'steps' or 'epoch'")

        if self.evaluation.save_strategy not in ["steps", "epoch"]:
            raise ValueError("save_strategy must be 'steps' or 'epoch'")

        if self.evaluation.metric_for_best_model not in [
            "eval_loss",
            "eval_f1",
            "eval_precision",
            "eval_recall",
            "eval_accuracy",
        ]:
            raise ValueError("metric_for_best_model must be a valid evaluation metric")

        # Validate logging configuration
        valid_log_levels = ["debug", "info", "warning", "error", "critical"]
        if self.logging.log_level.lower() not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {valid_log_levels}")

        # Validate advanced configuration
        if (
            self.advanced.class_weights
            and len(self.advanced.class_weights) != self.model.num_labels
        ):
            raise ValueError(
                f"class_weights must have {self.model.num_labels} elements"
            )


def load_training_config_from_yaml(config_path: str | Path) -> ModernBERTTrainingConfig:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ModernBERTTrainingConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

    try:
        # Parse nested configuration
        model_config = ModelConfig(**yaml_data["model"])
        data_config = DataConfig(**yaml_data["data"])
        training_config = TrainingConfig(**yaml_data.get("training", {}))
        hardware_config = HardwareConfig(**yaml_data.get("hardware", {}))
        evaluation_config = EvaluationConfig(**yaml_data.get("evaluation", {}))
        logging_config = LoggingConfig(**yaml_data.get("logging", {}))

        # Parse advanced config with nested augmentation
        advanced_data = yaml_data.get("advanced", {})
        augmentation_config = AugmentationConfig(
            **advanced_data.get("augmentation", {})
        )
        advanced_config = AdvancedConfig(
            **{k: v for k, v in advanced_data.items() if k != "augmentation"}
        )
        advanced_config.augmentation = augmentation_config

        config = ModernBERTTrainingConfig(
            model=model_config,
            data=data_config,
            training=training_config,
            hardware=hardware_config,
            evaluation=evaluation_config,
            logging=logging_config,
            seed=yaml_data.get("seed", 42),
            advanced=advanced_config,
        )

        # Validate configuration
        config.validate()

        return config

    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid configuration structure: {e}")


def save_training_config_to_yaml(
    config: ModernBERTTrainingConfig, output_path: str | Path
) -> None:
    """
    Save training configuration to YAML file.

    Args:
        config: ModernBERTTrainingConfig instance
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    config_dict = {
        "model": {
            "model_name": config.model.model_name,
            "num_labels": config.model.num_labels,
            "dropout": config.model.dropout,
            "hidden_dropout_prob": config.model.hidden_dropout_prob,
            "attention_probs_dropout_prob": config.model.attention_probs_dropout_prob,
        },
        "data": {
            "train_data_path": config.data.train_data_path,
            "validation_split": config.data.validation_split,
            "max_length": config.data.max_length,
            "batch_size": config.data.batch_size,
            "shuffle_train": config.data.shuffle_train,
            "shuffle_seed": config.data.shuffle_seed,
        },
        "training": {
            "output_dir": config.training.output_dir,
            "num_epochs": config.training.num_epochs,
            "learning_rate": config.training.learning_rate,
            "warmup_ratio": config.training.warmup_ratio,
            "weight_decay": config.training.weight_decay,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "max_grad_norm": config.training.max_grad_norm,
            "lr_scheduler_type": config.training.lr_scheduler_type,
            "early_stopping": config.training.early_stopping,
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_threshold": config.training.early_stopping_threshold,
        },
        "hardware": {
            "device": config.hardware.device,
            "mixed_precision": config.hardware.mixed_precision,
            "dataloader_num_workers": config.hardware.dataloader_num_workers,
            "dataloader_pin_memory": config.hardware.dataloader_pin_memory,
        },
        "evaluation": {
            "eval_steps": config.evaluation.eval_steps,
            "eval_strategy": config.evaluation.eval_strategy,
            "save_steps": config.evaluation.save_steps,
            "save_strategy": config.evaluation.save_strategy,
            "save_total_limit": config.evaluation.save_total_limit,
            "load_best_model_at_end": config.evaluation.load_best_model_at_end,
            "metric_for_best_model": config.evaluation.metric_for_best_model,
            "greater_is_better": config.evaluation.greater_is_better,
        },
        "logging": {
            "logging_dir": config.logging.logging_dir,
            "logging_steps": config.logging.logging_steps,
            "report_to": config.logging.report_to,
            "log_level": config.logging.log_level,
            "verbose": config.logging.verbose,
        },
        "seed": config.seed,
        "advanced": {
            "gradient_checkpointing": config.advanced.gradient_checkpointing,
            "push_to_hub": config.advanced.push_to_hub,
            "hub_model_id": config.advanced.hub_model_id,
            "class_weights": config.advanced.class_weights,
            "augmentation": {
                "enabled": config.advanced.augmentation.enabled,
                "dropout_prob": config.advanced.augmentation.dropout_prob,
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
