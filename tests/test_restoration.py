#!/usr/bin/env python3
"""
Test module for restoration algorithm

Tests the accuracy of converting token-level predictions back to original
data identifiers for validation and quality assurance.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_generation.restoration_tester import RestorationTester


def test_restoration_algorithm():
    """Test the restoration algorithm with example cases."""
    print("=== Testing Restoration Algorithm ===")

    test_cases = [
        {
            "name": "Simple identifier",
            "tokens": ["Data", "under", "GSE12345", "."],
            "labels": [0, 0, 1, 0],
            "expected": ["GSE12345"],
        },
        {
            "name": "Subword identifier",
            "tokens": ["Data", "under", "GS", "##E", "##12345", "."],
            "labels": [0, 0, 1, 1, 1, 0],
            "expected": ["GSE12345"],
        },
        {
            "name": "Multiple identifiers",
            "tokens": ["GS", "##E", "##123", "and", "SR", "##R", "##456", "."],
            "labels": [1, 1, 1, 0, 1, 1, 1, 0],
            "expected": ["GSE123", "SRR456"],
        },
        {
            "name": "No identifiers",
            "tokens": ["This", "is", "text", "."],
            "labels": [0, 0, 0, 0],
            "expected": [],
        },
        {
            "name": "End with identifier",
            "tokens": ["Available", "at", "PMC", "##123456"],
            "labels": [0, 0, 1, 1],
            "expected": ["PMC123456"],
        },
    ]

    tester = RestorationTester()

    for i, test_case in enumerate(test_cases):
        sample = {
            "tokens": test_case["tokens"],
            "labels": test_case["labels"],
            "expected_identifiers": test_case["expected"],
        }

        result = tester.test_sample(sample)

        print(f"\nTest {i + 1}: {test_case['name']}")
        print(f"  Tokens: {test_case['tokens']}")
        print(f"  Labels: {test_case['labels']}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Restored: {result['restored']}")
        print(f"  Exact match: {result['exact_match']}")
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall: {result['recall']:.3f}")
        print(f"  F1 Score: {result['f1_score']:.3f}")

        if not result["exact_match"]:
            print("  ❌ FAILED")
        else:
            print("  ✅ PASSED")


if __name__ == "__main__":
    # Run tests
    test_restoration_algorithm()
