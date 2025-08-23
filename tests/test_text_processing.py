#!/usr/bin/env python3
"""
Test module for text processing strategies

Tests the text cropping and negative sampling strategies used in
BERT training data generation.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.bert_training.strategies.text_processing_strategies import (
    WindowCropper,
    SentenceCropper,
    ParagraphCropper,
    RandomNegativeSampler,
    ContextualNegativeSampler,
    BalancedNegativeSampler
)


def test_text_processing_strategies():
    """Test text processing strategies with example data."""
    print("=== Testing Text Processing Strategies ===")

    # Example text with identifiers
    sample_text = """
    In this study, we analyzed gene expression data from GSE12345. 
    The protein sequences were retrieved from PDB1ABC and compared with SRR123456.
    Results showed significant correlation with previous findings.
    Methods involved standard protocols for data processing.
    """

    # Example identifier positions
    identifier_positions = [
        {"identifier": "GSE12345", "char_start": 56, "char_end": 64},
        {"identifier": "PDB1ABC", "char_start": 115, "char_end": 122},
        {"identifier": "SRR123456", "char_start": 140, "char_end": 149},
    ]

    print(f"Original text length: {len(sample_text)}")
    print(f"Identifier positions: {len(identifier_positions)}")

    # Test text croppers
    print("\n--- Text Cropping Strategies ---")

    croppers = {
        "Window (50 chars)": WindowCropper(window_size=50),
        "Window (100 chars)": WindowCropper(window_size=100),
        "Sentence": SentenceCropper(sentence_context=0),
        "Sentence + context": SentenceCropper(sentence_context=1),
    }

    for name, cropper in croppers.items():
        cropped = cropper.crop(sample_text, identifier_positions)
        print(f"\n{name}:")
        print(f"  Length: {len(cropped)}")
        print(f"  Text: {cropped.strip()[:100]}...")

    # Test negative samplers
    print("\n--- Negative Sampling Strategies ---")

    # Create excluded regions around identifiers
    excluded_regions = [
        {"char_start": pos["char_start"] - 10, "char_end": pos["char_end"] + 10}
        for pos in identifier_positions
    ]

    samplers = {
        "Random": RandomNegativeSampler(sample_length=80),
        "Contextual": ContextualNegativeSampler(),
        "Balanced": BalancedNegativeSampler(),
    }

    for name, sampler in samplers.items():
        negative_sample = sampler.sample(sample_text, excluded_regions)
        print(f"\n{name}:")
        print(f"  Length: {len(negative_sample)}")
        print(f"  Text: {negative_sample.strip()[:100]}...")

    print("\n=== Test Completed ===")


if __name__ == "__main__":
    test_text_processing_strategies()