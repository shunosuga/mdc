#!/usr/bin/env python3
"""
Configuration loader for BERT Training Data Generator
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Data paths configuration"""

    pmc_dir: str
    corpus_file: str
    pmc_ids_file: str


@dataclass
class ModelConfig:
    """Model configuration"""

    tokenizer_name: str = "answerdotai/ModernBERT-base"


@dataclass
class GenerationConfig:
    """Generation settings configuration"""

    num_samples: int = 1000
    positive_ratio: float = 0.7
    min_token_length: int = 10
    max_token_length: int = 128
    text_unit: str = "sentence"
    random_seed: int = 42


@dataclass
class OutputConfig:
    """Output settings configuration"""

    output_path: str = "training_data/bert_samples.jsonl"
    output_format: str = "jsonl"


@dataclass
class ProcessingConfig:
    """Processing settings configuration"""

    include_restoration_test: bool = True
    sampling_percentage: float | None = None


@dataclass
class CropperConfig:
    """Text cropper configuration"""

    strategy: str = "window"
    window_size: int = 100
    sentence_context: int = 1
    paragraph_context: int = 0


@dataclass
class NegativeSamplerConfig:
    """Negative sampler configuration"""

    strategy: str = "balanced"
    sample_length: int = 100
    min_gap: int = 50


@dataclass
class TextProcessingConfig:
    """Text processing strategies configuration"""

    cropper: CropperConfig = field(default_factory=CropperConfig)
    negative_sampler: NegativeSamplerConfig = field(
        default_factory=NegativeSamplerConfig
    )


@dataclass
class LoggingConfig:
    """Logging configuration"""

    verbose: bool = True
    progress_bar: bool = True


@dataclass
class BERTTrainingConfig:
    """Complete BERT training configuration"""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """Validate configuration values"""
        # Validate paths
        if not Path(self.data.pmc_dir).exists():
            raise ValueError(f"PMC directory not found: {self.data.pmc_dir}")
        if not Path(self.data.corpus_file).exists():
            raise ValueError(f"Corpus file not found: {self.data.corpus_file}")
        if not Path(self.data.pmc_ids_file).exists():
            raise ValueError(f"PMC IDs file not found: {self.data.pmc_ids_file}")

        # Validate generation settings
        if not 0.0 <= self.generation.positive_ratio <= 1.0:
            raise ValueError("positive_ratio must be between 0.0 and 1.0")

        if self.generation.min_token_length >= self.generation.max_token_length:
            raise ValueError("min_token_length must be less than max_token_length")

        if self.generation.text_unit not in ["sentence", "paragraph"]:
            raise ValueError("text_unit must be 'sentence' or 'paragraph'")

        # Validate output format
        if self.output.output_format not in ["jsonl", "json"]:
            raise ValueError("output_format must be 'jsonl' or 'json'")

        # Validate text processing strategies
        valid_cropper_strategies = ["window", "sentence", "paragraph"]
        if self.text_processing.cropper.strategy not in valid_cropper_strategies:
            raise ValueError(
                f"cropper strategy must be one of: {valid_cropper_strategies}"
            )

        valid_sampler_strategies = ["random", "contextual", "balanced"]
        if (
            self.text_processing.negative_sampler.strategy
            not in valid_sampler_strategies
        ):
            raise ValueError(
                f"negative_sampler strategy must be one of: {valid_sampler_strategies}"
            )

        # Validate sampling percentage
        if self.processing.sampling_percentage is not None:
            if not 0.0 < self.processing.sampling_percentage <= 1.0:
                raise ValueError("sampling_percentage must be between 0.0 and 1.0")


def load_config_from_yaml(config_path: str | Path) -> BERTTrainingConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        BERTTrainingConfig instance

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
        data_config = DataConfig(**yaml_data["data"])
        model_config = ModelConfig(**yaml_data.get("model", {}))
        generation_config = GenerationConfig(**yaml_data.get("generation", {}))
        output_config = OutputConfig(**yaml_data.get("output", {}))
        processing_config = ProcessingConfig(**yaml_data.get("processing", {}))

        # Parse text processing config
        text_proc_data = yaml_data.get("text_processing", {})
        cropper_config = CropperConfig(**text_proc_data.get("cropper", {}))
        sampler_config = NegativeSamplerConfig(
            **text_proc_data.get("negative_sampler", {})
        )
        text_processing_config = TextProcessingConfig(
            cropper=cropper_config, negative_sampler=sampler_config
        )

        logging_config = LoggingConfig(**yaml_data.get("logging", {}))

        config = BERTTrainingConfig(
            data=data_config,
            model=model_config,
            generation=generation_config,
            output=output_config,
            processing=processing_config,
            text_processing=text_processing_config,
            logging=logging_config,
        )

        # Validate configuration
        config.validate()

        return config

    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid configuration structure: {e}")


def save_config_to_yaml(config: BERTTrainingConfig, output_path: str | Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: BERTTrainingConfig instance
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    config_dict = {
        "data": {
            "pmc_dir": config.data.pmc_dir,
            "corpus_file": config.data.corpus_file,
            "pmc_ids_file": config.data.pmc_ids_file,
        },
        "model": {
            "tokenizer_name": config.model.tokenizer_name,
        },
        "generation": {
            "num_samples": config.generation.num_samples,
            "positive_ratio": config.generation.positive_ratio,
            "min_token_length": config.generation.min_token_length,
            "max_token_length": config.generation.max_token_length,
            "text_unit": config.generation.text_unit,
            "random_seed": config.generation.random_seed,
        },
        "output": {
            "output_path": config.output.output_path,
            "output_format": config.output.output_format,
        },
        "processing": {
            "include_restoration_test": config.processing.include_restoration_test,
            "sampling_percentage": config.processing.sampling_percentage,
        },
        "text_processing": {
            "cropper": {
                "strategy": config.text_processing.cropper.strategy,
                "window_size": config.text_processing.cropper.window_size,
                "sentence_context": config.text_processing.cropper.sentence_context,
                "paragraph_context": config.text_processing.cropper.paragraph_context,
            },
            "negative_sampler": {
                "strategy": config.text_processing.negative_sampler.strategy,
                "sample_length": config.text_processing.negative_sampler.sample_length,
                "min_gap": config.text_processing.negative_sampler.min_gap,
            },
        },
        "logging": {
            "verbose": config.logging.verbose,
            "progress_bar": config.logging.progress_bar,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
