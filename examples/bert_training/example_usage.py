#!/usr/bin/env python3
"""
Example Usage of BERT Training Data Generator

Demonstrates how to use the BERT training data generator with different
text cropping and negative sampling strategies.
"""

from transformers import AutoTokenizer

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.bert_training import BERTTrainingDataGenerator, RestorationTester
from src.bert_training.restoration_tester import test_restoration_algorithm
from src.bert_training.strategies import (
    BalancedNegativeSampler,
    ContextualNegativeSampler,
    ParagraphCropper,
    RandomNegativeSampler,
    SentenceCropper,
    WindowCropper,
    create_negative_sampler,
    create_text_cropper,
)


def example_basic_usage():
    """Basic usage example with default settings."""
    print("=== Basic Usage Example ===")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Initialize generator
    generator = BERTTrainingDataGenerator(
        pmc_dir="data/pmc/txt",
        corpus_file="data/corpus/corpus_consolidated.json",
        pmc_ids_file="data/pmc/PMC-ids.csv",
        tokenizer=tokenizer,
    )

    # Generate training data with default settings
    result = generator.generate(
        num_samples=100,
        positive_ratio=0.7,
        output_path="training_data/basic_samples.jsonl",
        random_seed=42,
    )

    print("Generation completed!")
    print(f"Total samples: {result['generation_summary']['total_samples']}")
    print(f"Positive samples: {result['generation_summary']['positive_samples']}")
    print(f"Negative samples: {result['generation_summary']['negative_samples']}")

    if result["restoration_test_results"]:
        print(
            f"Restoration accuracy: {result['restoration_test_results']['exact_match_rate']:.3f}"
        )
        print(f"F1 Score: {result['restoration_test_results']['f1_score']:.3f}")


def example_custom_strategies():
    """Example with custom text cropping and negative sampling strategies."""
    print("\n=== Custom Strategies Example ===")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Initialize generator
    generator = BERTTrainingDataGenerator(
        pmc_dir="data/pmc/txt",
        corpus_file="data/corpus/corpus_consolidated.json",
        pmc_ids_file="data/pmc/PMC-ids.csv",
        tokenizer=tokenizer,
    )

    # Define custom strategies
    text_cropper = SentenceCropper(sentence_context=1)  # Include 1 sentence context
    negative_sampler = ContextualNegativeSampler()  # Use contextual sampling

    # Generate training data with custom strategies
    result = generator.generate(
        num_samples=500,
        positive_ratio=0.6,
        max_token_length=256,
        output_path="training_data/custom_samples.jsonl",
        text_cropper=text_cropper,
        negative_sampler=negative_sampler,
        random_seed=123,
    )

    print("Custom generation completed!")
    print(
        f"Average tokens per sample: {result['generation_summary']['average_tokens_per_sample']:.1f}"
    )
    print(
        f"Unique identifiers found: {result['generation_summary']['unique_identifiers_found']}"
    )


def example_factory_functions():
    """Example using factory functions to create strategies."""
    print("\n=== Factory Functions Example ===")

    # Create strategies using factory functions
    window_cropper = create_text_cropper("window", window_size=150)
    balanced_sampler = create_negative_sampler("balanced")

    # Initialize generator
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    generator = BERTTrainingDataGenerator(
        pmc_dir="data/pmc/txt",
        corpus_file="data/corpus/corpus_consolidated.json",
        pmc_ids_file="data/pmc/PMC-ids.csv",
        tokenizer=tokenizer,
    )

    # Generate with factory-created strategies
    result = generator.generate(
        num_samples=200,
        positive_ratio=0.8,
        text_cropper=window_cropper,
        negative_sampler=balanced_sampler,
        output_path="training_data/factory_samples.jsonl",
    )

    print("Factory-based generation completed!")
    print(f"Total samples generated: {result['generation_summary']['total_samples']}")


def example_different_croppers():
    """Example comparing different text cropping strategies."""
    print("\n=== Text Cropper Comparison ===")

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Different cropping strategies
    croppers = {
        "window_small": WindowCropper(window_size=80),
        "window_large": WindowCropper(window_size=200),
        "sentence": SentenceCropper(sentence_context=0),
        "sentence_context": SentenceCropper(sentence_context=2),
        "paragraph": ParagraphCropper(),
    }

    for name, cropper in croppers.items():
        print(f"\nTesting {name} cropper...")

        generator = BERTTrainingDataGenerator(
            pmc_dir="data/pmc/txt",
            corpus_file="data/corpus/corpus_consolidated.json",
            pmc_ids_file="data/pmc/PMC-ids.csv",
            tokenizer=tokenizer,
        )

        result = generator.generate(
            num_samples=50,
            positive_ratio=1.0,  # Only positive samples to test cropping
            text_cropper=cropper,
            output_path=f"training_data/{name}_samples.jsonl",
            random_seed=42,
        )

        avg_tokens = result["generation_summary"]["average_tokens_per_sample"]
        print(f"  Average tokens: {avg_tokens:.1f}")
        print(f"  Samples generated: {result['generation_summary']['total_samples']}")


def example_negative_samplers():
    """Example comparing different negative sampling strategies."""
    print("\n=== Negative Sampler Comparison ===")

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Different negative sampling strategies
    samplers = {
        "random": RandomNegativeSampler(sample_length=100),
        "random_long": RandomNegativeSampler(sample_length=200),
        "contextual": ContextualNegativeSampler(),
        "balanced": BalancedNegativeSampler(),
    }

    for name, sampler in samplers.items():
        print(f"\nTesting {name} negative sampler...")

        generator = BERTTrainingDataGenerator(
            pmc_dir="data/pmc/txt",
            corpus_file="data/corpus/corpus_consolidated.json",
            pmc_ids_file="data/pmc/PMC-ids.csv",
            tokenizer=tokenizer,
        )

        result = generator.generate(
            num_samples=50,
            positive_ratio=0.0,  # Only negative samples to test sampling
            negative_sampler=sampler,
            output_path=f"training_data/{name}_negative.jsonl",
            random_seed=42,
        )

        avg_tokens = result["generation_summary"]["average_tokens_per_sample"]
        print(f"  Average tokens: {avg_tokens:.1f}")
        print(f"  Samples generated: {result['generation_summary']['total_samples']}")


def example_restoration_testing():
    """Example of testing restoration accuracy."""
    print("\n=== Restoration Testing Example ===")

    # Run built-in restoration algorithm tests
    test_restoration_algorithm()

    # Test with custom examples
    tester = RestorationTester()

    custom_samples = [
        {
            "tokens": [
                "The",
                "data",
                "from",
                "GS",
                "##E",
                "##54321",
                "shows",
                "significance",
                ".",
            ],
            "labels": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "expected_identifiers": ["GSE54321"],
        },
        {
            "tokens": [
                "Results",
                "available",
                "in",
                "PMC",
                "##123",
                "and",
                "PMC",
                "##456",
                ".",
            ],
            "labels": [0, 0, 0, 1, 1, 0, 1, 1, 0],
            "expected_identifiers": ["PMC123", "PMC456"],
        },
    ]

    print("\nTesting custom samples:")
    results = tester.test_with_examples(custom_samples, num_examples=2)

    print(f"Overall accuracy: {results['exact_match_rate']:.3f}")
    print(f"Average precision: {results['average_precision']:.3f}")
    print(f"Average recall: {results['average_recall']:.3f}")
    print(f"F1 score: {results['f1_score']:.3f}")

    if results["failed_examples"]:
        print("\nFailed examples:")
        for i, example in enumerate(results["failed_examples"]):
            print(f"  {i + 1}. Expected: {example['expected']}")
            print(f"     Restored: {example['restored']}")

    if results["success_examples"]:
        print("\nSuccess examples:")
        for i, example in enumerate(results["success_examples"]):
            print(f"  {i + 1}. Tokens: {' '.join(example['tokens'][:10])}...")
            print(f"     Identifiers: {example['expected']}")


def example_complete_workflow():
    """Complete workflow example with all features."""
    print("\n=== Complete Workflow Example ===")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # Create custom strategies
    text_cropper = SentenceCropper(sentence_context=1)
    negative_sampler = BalancedNegativeSampler()

    # Initialize generator
    generator = BERTTrainingDataGenerator(
        pmc_dir="data/pmc/txt",
        corpus_file="data/corpus/corpus_consolidated.json",
        pmc_ids_file="data/pmc/PMC-ids.csv",
        tokenizer=tokenizer,
    )

    # Generate comprehensive training data
    result = generator.generate(
        num_samples=1000,
        positive_ratio=0.7,
        tokenizer_name="allenai/scibert_scivocab_uncased",
        text_unit="sentence",
        min_token_length=10,
        max_token_length=128,
        output_format="jsonl",
        output_path="training_data/complete_workflow.jsonl",
        include_restoration_test=True,
        text_cropper=text_cropper,
        negative_sampler=negative_sampler,
        random_seed=42,
    )

    # Display comprehensive results
    summary = result["generation_summary"]
    restoration = result["restoration_test_results"]

    print("=== Generation Summary ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Positive samples: {summary['positive_samples']}")
    print(f"Negative samples: {summary['negative_samples']}")
    print(f"Average tokens per sample: {summary['average_tokens_per_sample']:.1f}")
    print(f"Unique identifiers found: {summary['unique_identifiers_found']}")

    print("\n=== Restoration Test Results ===")
    print(f"Exact match rate: {restoration['exact_match_rate']:.3f}")
    print(f"Average precision: {restoration['average_precision']:.3f}")
    print(f"Average recall: {restoration['average_recall']:.3f}")
    print(f"F1 score: {restoration['f1_score']:.3f}")
    print(f"Failed samples: {restoration['failed_samples_count']}")

    print(f"\nOutput files: {result['output_files']}")

    return result


if __name__ == "__main__":
    # Run all examples
    try:
        example_basic_usage()
        example_custom_strategies()
        example_factory_functions()
        example_different_croppers()
        example_negative_samplers()
        example_restoration_testing()
        example_complete_workflow()

        print("\n🎉 All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("Make sure data files exist and paths are correct.")
