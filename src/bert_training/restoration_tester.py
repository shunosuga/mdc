#!/usr/bin/env python3
"""
Restoration Tester for BERT Data Identifier Detection

Tests the accuracy of converting token-level predictions back to original
data identifiers for validation and quality assurance.
"""

from typing import Any


def restore_identifiers(tokens: list[str], labels: list[int]) -> list[str]:
    """
    Restore data identifiers from tokenized predictions.

    Args:
        tokens: BERT tokenizer output tokens
        labels: Binary labels (0=normal, 1=data_identifier)

    Returns:
        List of restored data identifiers

    Example:
        Input:
            tokens = ["Data", "under", "GS", "##E", "##12345", "and", "SR", "##R123456"]
            labels = [0, 0, 1, 1, 1, 0, 1, 1]

        Output:
            ["GSE12345", "SRR123456"]
    """
    identifiers = []
    current_tokens = []

    for token, label in zip(tokens, labels):
        if label == 1:  # Binary label: 1 indicates data identifier
            current_tokens.append(token)
        else:
            if current_tokens:
                # Reconstruct identifier from subword tokens
                identifier = reconstruct_from_subwords(current_tokens)
                if identifier:  # Only add non-empty identifiers
                    identifiers.append(identifier)
                current_tokens = []

    # Handle case where sequence ends with label 1
    if current_tokens:
        identifier = reconstruct_from_subwords(current_tokens)
        if identifier:
            identifiers.append(identifier)

    return identifiers


def reconstruct_from_subwords(tokens: list[str]) -> str:
    """
    Reconstruct original text from BERT subword tokens.

    Args:
        tokens: List of BERT tokens (may include ## prefixes)

    Returns:
        Reconstructed text string

    Example:
        Input: ["GS", "##E", "##12345"]
        Output: "GSE12345"
    """
    if not tokens:
        return ""

    text = ""
    for token in tokens:
        if token.startswith("##"):
            text += token[2:]  # Remove ## prefix
        else:
            text += token

    return text.strip()


def extend_to_word_boundary(
    tokens: list[str], labels: list[int], original_text: str, token_spans: list[dict]
) -> list[int]:
    """
    Extend label=1 predictions to complete word boundaries.

    This implements the requirement that if any token is predicted as data identifier,
    we extend to the next word boundary.

    Args:
        tokens: BERT tokens
        labels: Current binary labels
        original_text: Original text before tokenization
        token_spans: Character-level spans for each token

    Returns:
        Extended binary labels
    """
    extended_labels = labels.copy()

    if not token_spans or len(token_spans) != len(tokens):
        return extended_labels

    i = 0
    while i < len(labels):
        if labels[i] == 1:
            # Found start of identifier, extend to word boundary
            start_idx = i

            # Find end of current identifier sequence
            while i < len(labels) and labels[i] == 1:
                i += 1
            end_idx = i

            # Extend to word boundary if needed
            if end_idx < len(tokens):
                # Check if we're at a word boundary
                current_char_end = token_spans[end_idx - 1]["end"]

                # Extend until we hit a word boundary (whitespace or punctuation)
                while (
                    end_idx < len(tokens)
                    and current_char_end < len(original_text)
                    and not original_text[current_char_end].isspace()
                    and original_text[current_char_end].isalnum()
                ):
                    extended_labels[end_idx] = 1
                    end_idx += 1
                    if end_idx - 1 < len(token_spans):
                        current_char_end = token_spans[end_idx - 1]["end"]
                    else:
                        break

            i = end_idx
        else:
            i += 1

    return extended_labels


class RestorationTester:
    """Test identifier restoration accuracy."""

    def __init__(self):
        """Initialize restoration tester."""
        pass

    def test_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Test a single sample's restoration accuracy.

        Args:
            sample: Training sample with tokens, labels, and expected_identifiers

        Returns:
            Dictionary with restoration test results
        """
        tokens = sample.get("tokens", [])
        labels = sample.get("labels", [])
        expected = sample.get("expected_identifiers", [])

        # Restore identifiers from tokens and labels
        restored = restore_identifiers(tokens, labels)

        # Calculate metrics
        restored_set = set(restored)
        expected_set = set(expected)

        intersection = restored_set & expected_set

        precision = len(intersection) / len(restored_set) if restored_set else 0.0
        recall = (
            len(intersection) / len(expected_set)
            if expected_set
            else 1.0
            if not restored_set
            else 0.0
        )
        exact_match = restored_set == expected_set

        return {
            "restored": restored,
            "expected": expected,
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0,
        }

    def test_dataset(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Test entire dataset restoration accuracy.

        Args:
            samples: List of training samples

        Returns:
            Dictionary with overall test statistics
        """
        if not samples:
            return {
                "exact_match_rate": 0.0,
                "average_precision": 0.0,
                "average_recall": 0.0,
                "f1_score": 0.0,
                "total_samples": 0,
                "failed_samples_count": 0,
            }

        results = [self.test_sample(sample) for sample in samples]

        # Calculate overall statistics
        total_samples = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        total_precision = sum(r["precision"] for r in results)
        total_recall = sum(r["recall"] for r in results)
        failed_samples = sum(1 for r in results if not r["exact_match"])

        avg_precision = total_precision / total_samples if total_samples > 0 else 0.0
        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        f1_score = (
            2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )

        return {
            "exact_match_rate": exact_matches / total_samples,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "f1_score": f1_score,
            "total_samples": total_samples,
            "failed_samples_count": failed_samples,
        }

    def test_with_examples(
        self, samples: list[dict[str, Any]], num_examples: int = 5
    ) -> dict[str, Any]:
        """
        Test dataset with detailed examples of failures.

        Args:
            samples: List of training samples
            num_examples: Number of failure examples to include

        Returns:
            Test results with failure examples
        """
        basic_results = self.test_dataset(samples)

        # Find examples of failed restorations
        failed_examples = []
        for sample in samples[
            : num_examples * 10
        ]:  # Look at more samples to find failures
            result = self.test_sample(sample)
            if not result["exact_match"] and len(failed_examples) < num_examples:
                failed_examples.append(
                    {
                        "tokens": sample.get("tokens", []),
                        "labels": sample.get("labels", []),
                        "restored": result["restored"],
                        "expected": result["expected"],
                        "original_text": sample.get("original_text", ""),
                        "precision": result["precision"],
                        "recall": result["recall"],
                    }
                )

        # Find examples of successful restorations
        success_examples = []
        for sample in samples[: num_examples * 10]:
            result = self.test_sample(sample)
            if (
                result["exact_match"]
                and result["expected"]
                and len(success_examples) < num_examples
            ):
                success_examples.append(
                    {
                        "tokens": sample.get("tokens", []),
                        "labels": sample.get("labels", []),
                        "restored": result["restored"],
                        "expected": result["expected"],
                        "original_text": sample.get("original_text", ""),
                    }
                )

        basic_results["failed_examples"] = failed_examples
        basic_results["success_examples"] = success_examples

        return basic_results


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
