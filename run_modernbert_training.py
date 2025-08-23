#!/usr/bin/env python3
"""
ModernBERT Training Runner

Usage:
    python run_modernbert_training.py --config config/training_config.yaml
    python run_modernbert_training.py --config config/training_config.yaml --epochs 5 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

from src.training.modern_bert_trainer import ModernBERTTrainer

# Add src to path
# sys.path.append(str(Path(__file__).parent / "src"))
from src.training.training_config_loader import (
    ModernBERTTrainingConfig,
    load_training_config_from_yaml,
)


def setup_device_info():
    """Display device information"""
    try:
        import torch

        print("=== Device Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("Using CPU for training")
        print()
    except ImportError:
        print("PyTorch not available - cannot show device info")
        print()


def run_training(config: ModernBERTTrainingConfig) -> dict:
    """
    Run ModernBERT training with given configuration.

    Args:
        config: ModernBERTTrainingConfig instance

    Returns:
        Training results dictionary
    """
    if config.logging.verbose:
        print("=== ModernBERT Training ===")
        print(f"Configuration loaded successfully")
        print(f"Model: {config.model.model_name}")
        print(f"Training data: {config.data.train_data_path}")
        print(f"Output directory: {config.training.output_dir}")
        print(f"Epochs: {config.training.num_epochs}")
        print(f"Batch size: {config.data.batch_size}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Max length: {config.data.max_length}")
        print()

    # Setup device information
    setup_device_info()

    # Create output directories
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = Path(config.logging.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    if config.logging.verbose:
        print("Initializing trainer...")
        print()

    trainer = ModernBERTTrainer(config)

    # Run training
    if config.logging.verbose:
        print("Starting training pipeline...")
        print()

    result = trainer.run_full_training()

    if config.logging.verbose:
        print()
        print("=== Training Results ===")

        train_metrics = result["train_result"].metrics
        print(f"Final training loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Training runtime: {train_metrics.get('train_runtime', 0):.1f}s")
        print(
            f"Training samples per second: {train_metrics.get('train_samples_per_second', 0):.2f}"
        )

        if result["eval_result"]:
            eval_metrics = result["eval_result"]
            print(f"Final validation loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
            print(f"Final validation F1: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
            print(
                f"Final validation accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}"
            )

        print()
        print(f"Model saved to: {config.training.output_dir}")
        print(f"Logs saved to: {config.logging.logging_dir}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ModernBERT Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_modernbert_training.py --config config/training_config.yaml
  python run_modernbert_training.py --config config/training_config.yaml --epochs 5
  python run_modernbert_training.py --config config/training_config.yaml --batch-size 32 --lr 1e-5
        """,
    )

    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML training configuration file"
    )

    parser.add_argument("--output-dir", help="Override output directory for model")

    parser.add_argument("--epochs", type=int, help="Override number of training epochs")

    parser.add_argument("--batch-size", type=int, help="Override batch size")

    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        dest="learning_rate",
        help="Override learning rate",
    )

    parser.add_argument(
        "--max-length", type=int, help="Override maximum sequence length"
    )

    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], help="Override device selection"
    )

    parser.add_argument(
        "--mixed-precision", action="store_true", help="Enable mixed precision training"
    )

    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load configuration and show settings without training",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_training_config_from_yaml(args.config)

        # Apply command-line overrides
        if args.output_dir:
            config.training.output_dir = args.output_dir

        if args.epochs:
            config.training.num_epochs = args.epochs

        if args.batch_size:
            config.data.batch_size = args.batch_size

        if args.learning_rate:
            config.training.learning_rate = args.learning_rate

        if args.max_length:
            config.data.max_length = args.max_length

        if args.device:
            config.hardware.device = args.device

        if args.mixed_precision:
            config.hardware.mixed_precision = True
        elif args.no_mixed_precision:
            config.hardware.mixed_precision = False

        if args.gradient_checkpointing:
            config.advanced.gradient_checkpointing = True

        if args.quiet:
            config.logging.verbose = False

        # Re-validate configuration after overrides
        config.validate()

        if args.dry_run:
            print("=== Dry Run - Configuration Check ===")
            print(f"Model: {config.model.model_name}")
            print(f"Training data: {config.data.train_data_path}")
            print(f"Output directory: {config.training.output_dir}")
            print(f"Epochs: {config.training.num_epochs}")
            print(f"Batch size: {config.data.batch_size}")
            print(f"Learning rate: {config.training.learning_rate}")
            print(f"Max length: {config.data.max_length}")
            print(f"Device: {config.hardware.device}")
            print(f"Mixed precision: {config.hardware.mixed_precision}")
            print(f"Validation split: {config.data.validation_split}")
            print("\nConfiguration is valid. Ready for training.")
            return 0

        # Run training
        result = run_training(config)

        print("Training completed successfully!")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
