#!/usr/bin/env python3
"""
BERT Training Data Generator - Configuration-based runner

Usage:
    python run_bert_training_generator.py --config config/bert_training_config.yaml
    python run_bert_training_generator.py --config config/my_config.yaml --output-dir results/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer

from bert_training.bert_training_data_generator import BERTTrainingDataGenerator
from bert_training.config_loader import BERTTrainingConfig, load_config_from_yaml
from bert_training.strategies.text_processing_strategies import (
    create_negative_sampler,
    create_text_cropper,
)


def create_output_directory(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def run_generation(config: BERTTrainingConfig) -> dict:
    """
    Run BERT training data generation with given configuration.

    Args:
        config: BERTTrainingConfig instance

    Returns:
        Generation results dictionary
    """
    if config.logging.verbose:
        print("=== BERT Training Data Generator ===")
        print(f"Configuration loaded successfully")
        print(f"PMC directory: {config.data.pmc_dir}")
        print(f"Corpus file: {config.data.corpus_file}")
        print(f"Output path: {config.output.output_path}")
        print(f"Target samples: {config.generation.num_samples}")
        print()

    # Create output directory
    create_output_directory(config.output.output_path)

    # Initialize tokenizer
    if config.logging.verbose:
        print(f"Loading tokenizer: {config.model.tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)

    # Initialize generator
    generator = BERTTrainingDataGenerator(
        pmc_dir=config.data.pmc_dir,
        corpus_file=config.data.corpus_file,
        pmc_ids_file=config.data.pmc_ids_file,
        tokenizer=tokenizer,
    )

    # Create text processing strategies
    text_cropper = None
    if config.text_processing.cropper.strategy:
        # Select appropriate kwargs based on strategy
        if config.text_processing.cropper.strategy == "window":
            cropper_kwargs = {"window_size": config.text_processing.cropper.window_size}
        elif config.text_processing.cropper.strategy == "sentence":
            cropper_kwargs = {
                "sentence_context": config.text_processing.cropper.sentence_context
            }
        elif config.text_processing.cropper.strategy == "paragraph":
            cropper_kwargs = {
                "paragraph_context": config.text_processing.cropper.paragraph_context
            }
        else:
            cropper_kwargs = {}

        text_cropper = create_text_cropper(
            config.text_processing.cropper.strategy, **cropper_kwargs
        )

    negative_sampler = None
    if config.text_processing.negative_sampler.strategy:
        # Select appropriate kwargs based on strategy
        if config.text_processing.negative_sampler.strategy == "random":
            sampler_kwargs = {
                "sample_length": config.text_processing.negative_sampler.sample_length,
                "min_gap": config.text_processing.negative_sampler.min_gap,
            }
        elif config.text_processing.negative_sampler.strategy == "contextual":
            sampler_kwargs = {}  # ContextualNegativeSampler doesn't take these args
        elif config.text_processing.negative_sampler.strategy == "balanced":
            sampler_kwargs = {}  # BalancedNegativeSampler doesn't take these args
        else:
            sampler_kwargs = {}

        negative_sampler = create_negative_sampler(
            config.text_processing.negative_sampler.strategy, **sampler_kwargs
        )

    # Generate training data
    if config.logging.verbose:
        print("Starting data generation...")
        print()

    result = generator.generate(
        num_samples=config.generation.num_samples,
        positive_ratio=config.generation.positive_ratio,
        tokenizer_name=config.model.tokenizer_name,
        text_unit=config.generation.text_unit,
        min_token_length=config.generation.min_token_length,
        max_token_length=config.generation.max_token_length,
        output_format=config.output.output_format,
        output_path=config.output.output_path,
        random_seed=config.generation.random_seed,
        include_restoration_test=config.processing.include_restoration_test,
        text_cropper=text_cropper,
        negative_sampler=negative_sampler,
    )

    if config.logging.verbose:
        print()
        print("=== Generation Results ===")
        gen_summary = result.get("generation_summary", {})
        print(f"Total samples generated: {gen_summary.get('total_samples', 0)}")
        print(f"Positive samples: {gen_summary.get('positive_samples', 0)}")
        print(f"Negative samples: {gen_summary.get('negative_samples', 0)}")
        print(
            f"Average tokens per sample: {gen_summary.get('average_tokens_per_sample', 0):.1f}"
        )

        if config.processing.include_restoration_test:
            restoration_results = result.get("restoration_test_results", {})
            print(f"Restoration accuracy: {restoration_results.get('f1_score', 0):.3f}")

        print()
        output_files = result.get("output_files", [])
        print("Output files:")
        for file_path in output_files:
            print(f"  - {file_path}")

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BERT Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bert_training_generator.py --config config/bert_training_config.yaml
  python run_bert_training_generator.py --config my_config.yaml --output-dir results/
  python run_bert_training_generator.py --config config.yaml --samples 500 --positive-ratio 0.8
        """,
    )

    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML configuration file"
    )

    parser.add_argument("--output-dir", help="Override output directory")

    parser.add_argument(
        "--samples", type=int, help="Override number of samples to generate"
    )

    parser.add_argument(
        "--positive-ratio", type=float, help="Override positive sample ratio (0.0-1.0)"
    )

    parser.add_argument("--random-seed", type=int, help="Override random seed")

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config_from_yaml(args.config)

        # Apply command-line overrides
        if args.output_dir:
            # Update output path to use new directory
            filename = Path(config.output.output_path).name
            config.output.output_path = str(Path(args.output_dir) / filename)

        if args.samples:
            config.generation.num_samples = args.samples

        if args.positive_ratio is not None:
            if not 0.0 <= args.positive_ratio <= 1.0:
                print("Error: positive-ratio must be between 0.0 and 1.0")
                return 1
            config.generation.positive_ratio = args.positive_ratio

        if args.random_seed is not None:
            config.generation.random_seed = args.random_seed

        if args.quiet:
            config.logging.verbose = False

        # Re-validate configuration after overrides
        config.validate()

        # Run generation
        result = run_generation(config)

        print("Generation completed successfully!")
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
