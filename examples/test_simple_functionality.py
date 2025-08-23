#!/usr/bin/env python3
"""
Simple functionality test for BERT training components
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bert_training.restoration_tester import restore_identifiers, reconstruct_from_subwords
from src.bert_training.strategies.text_processing_strategies import (
    WindowCropper, SentenceCropper, RandomNegativeSampler
)

def test_restoration():
    """Test the restoration functionality with simple examples"""
    print("=== Testing Restoration Functionality ===")
    
    # Test case 1: Simple identifier restoration
    tokens = ["Data", "under", "GS", "##E", "##12345", "and", "SR", "##R123456"]
    labels = [0, 0, 1, 1, 1, 0, 1, 1]
    
    identifiers = restore_identifiers(tokens, labels)
    print(f"Tokens: {tokens}")
    print(f"Labels: {labels}")
    print(f"Restored identifiers: {identifiers}")
    print(f"Expected: ['GSE12345', 'SRR123456']")
    print(f"Match: {identifiers == ['GSE12345', 'SRR123456']}")
    
    # Test case 2: Subword reconstruction
    subword_tokens = ["GS", "##E", "##12345"]
    reconstructed = reconstruct_from_subwords(subword_tokens)
    print(f"\nSubword tokens: {subword_tokens}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Expected: 'GSE12345'")
    print(f"Match: {reconstructed == 'GSE12345'}")

def test_text_processing():
    """Test text processing strategies"""
    print("\n=== Testing Text Processing Strategies ===")
    
    # Test text
    text = "This study analyzed data from GSE12345 repository. The dataset contains gene expression data."
    identifier_positions = [{"char_start": 32, "char_end": 40, "text": "GSE12345"}]
    
    # Test window cropper
    cropper = WindowCropper(window_size=50)
    cropped_text = cropper.crop(text, identifier_positions)
    print(f"Original text: {text}")
    print(f"Window cropped: {cropped_text}")
    
    # Test sentence cropper
    sentence_cropper = SentenceCropper(sentence_context=0)
    sentence_cropped = sentence_cropper.crop(text, identifier_positions)
    print(f"Sentence cropped: {sentence_cropped}")
    
    # Test negative sampler
    excluded_regions = identifier_positions
    sampler = RandomNegativeSampler(sample_length=30)
    negative_sample = sampler.sample(text, excluded_regions)
    print(f"Negative sample: {negative_sample}")

if __name__ == "__main__":
    try:
        test_restoration()
        test_text_processing()
        print("\n[SUCCESS] All simple tests passed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()