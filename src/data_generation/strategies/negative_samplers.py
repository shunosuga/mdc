#!/usr/bin/env python3
"""
Text Processing Strategies for BERT Training Data Generation

Implements dependency injection interfaces and example implementations for
text cropping and negative sampling strategies.
"""

import random
import re
from typing import Protocol


class NegativeSampler(Protocol):
    """Interface for negative sample generation strategies."""

    def sample(
        self,
        text: str,
        excluded_regions: list[dict],
        tokenizer,
        min_token_length: int,
        max_token_length: int,
    ) -> str:
        """
        Generate negative samples from text.

        Args:
            text: Source text content
            excluded_regions: Regions to avoid (containing identifiers)
            tokenizer: HuggingFace tokenizer for length validation
            min_token_length: Minimum token length
            max_token_length: Maximum token length

        Returns:
            Negative sample text segment
        """
        ...


# =============================================================================
# Negative Sampling Implementations
# =============================================================================


class RandomNegativeSampler:
    """Generate random negative samples avoiding identifier regions."""

    def __init__(self, sample_length: int = 100, min_gap: int = 50):
        """
        Initialize random negative sampler.

        Args:
            sample_length: Target length for negative samples
            min_gap: Minimum distance from identifier regions
        """
        self.sample_length = sample_length
        self.min_gap = min_gap

    def sample(
        self,
        text: str,
        excluded_regions: list[dict],
        tokenizer,
        min_token_length: int,
        max_token_length: int,
    ) -> str:
        """
        Generate random negative sample avoiding excluded regions.

        Args:
            text: Source text
            excluded_regions: Regions containing identifiers to avoid
            tokenizer: HuggingFace tokenizer for length validation
            min_token_length: Minimum token length
            max_token_length: Maximum token length

        Returns:
            Random text sample without identifiers
        """
        # Use tokenizer to measure actual token lengths
        max_attempts = 100

        for _ in range(max_attempts):
            # Estimate character length from token length
            estimated_chars = max_token_length * 4

            if len(text) < estimated_chars:
                candidate_text = text
            else:
                # Create list of valid start positions
                valid_positions = []

                for start_pos in range(len(text) - estimated_chars + 1):
                    end_pos = start_pos + estimated_chars

                    # Check if this sample would overlap with any excluded region
                    valid = True
                    for region in excluded_regions:
                        region_start = max(0, region["char_start"] - self.min_gap)
                        region_end = region["char_end"] + self.min_gap

                        # Check for overlap
                        if not (end_pos <= region_start or start_pos >= region_end):
                            valid = False
                            break

                    if valid:
                        valid_positions.append(start_pos)

                if not valid_positions:
                    # If no valid positions, try smaller sample
                    estimated_chars = min_token_length * 4
                    if len(text) < estimated_chars:
                        candidate_text = text
                    else:
                        start_pos = 0
                        candidate_text = text[start_pos : start_pos + estimated_chars]
                else:
                    # Choose random valid position
                    start_pos = random.choice(valid_positions)
                    candidate_text = text[start_pos : start_pos + estimated_chars]

            # Align to word boundaries
            candidate_text = self._align_to_words(
                text, start_pos if "start_pos" in locals() else 0, candidate_text
            )

            if not candidate_text.strip():
                continue

            # Check actual token length
            tokens = tokenizer.tokenize(candidate_text)
            token_count = len(tokens)

            if min_token_length <= token_count <= max_token_length:
                return candidate_text.strip()
            elif token_count > max_token_length:
                # Truncate to max_token_length
                truncated_tokens = tokens[:max_token_length]
                candidate_text = tokenizer.convert_tokens_to_string(truncated_tokens)
                if tokenizer.tokenize(candidate_text):  # Ensure it's valid
                    return candidate_text.strip()

        # Fallback: return empty string if no valid sample found
        return ""

    def _align_to_words(self, original_text: str, start_pos: int, sample: str) -> str:
        """Align sample to word boundaries."""
        # Find start of first complete word
        sample_start = 0
        while sample_start < len(sample) and not sample[sample_start].isspace():
            if (
                start_pos + sample_start == 0
                or original_text[start_pos + sample_start - 1].isspace()
            ):
                break
            sample_start += 1

        # Find end of last complete word
        sample_end = len(sample)
        while sample_end > sample_start and not sample[sample_end - 1].isspace():
            abs_pos = start_pos + sample_end - 1
            if (
                abs_pos + 1 >= len(original_text)
                or original_text[abs_pos + 1].isspace()
            ):
                break
            sample_end -= 1

        return sample[sample_start:sample_end] if sample_end > sample_start else sample


class ContextualNegativeSampler:
    """Generate negative samples from similar scientific contexts."""

    def __init__(self, context_keywords: list[str] | None = None):
        """
        Initialize contextual negative sampler.

        Args:
            context_keywords: Keywords indicating scientific context
        """
        self.context_keywords = context_keywords or [
            "analysis",
            "data",
            "study",
            "research",
            "method",
            "result",
            "conclusion",
            "experiment",
            "sample",
            "protein",
            "gene",
            "cell",
            "tissue",
            "patient",
            "treatment",
            "control",
            "significant",
            "correlation",
            "expression",
        ]

    def sample(
        self,
        text: str,
        excluded_regions: list[dict],
        tokenizer,
        min_token_length: int,
        max_token_length: int,
    ) -> str:
        """
        Generate negative sample from similar scientific context.

        Args:
            text: Source text
            excluded_regions: Regions to avoid
            tokenizer: HuggingFace tokenizer for length validation
            min_token_length: Minimum token length
            max_token_length: Maximum token length

        Returns:
            Contextual negative sample
        """
        # Split text into sentences
        sentences = re.split(r"[.!?]+\s+", text)

        # Score sentences based on scientific context
        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue

            score = self._calculate_context_score(sentence)
            sentence_scores.append((sentence, score))

        # Sort by score
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Find sentences that don't contain identifiers and meet token constraints
        for sentence, score in sentence_scores:
            # Check if sentence overlaps with excluded regions
            sentence_start = text.find(sentence)
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)

                valid = True
                for region in excluded_regions:
                    if not (
                        sentence_end <= region["char_start"]
                        or sentence_start >= region["char_end"]
                    ):
                        valid = False
                        break

                if valid:
                    # Check token length constraints
                    tokens = tokenizer.tokenize(sentence)
                    token_count = len(tokens)

                    if min_token_length <= token_count <= max_token_length:
                        return sentence.strip()
                    elif token_count > max_token_length:
                        # Truncate to max_token_length
                        truncated_tokens = tokens[:max_token_length]
                        truncated_text = tokenizer.convert_tokens_to_string(
                            truncated_tokens
                        )
                        if tokenizer.tokenize(truncated_text):  # Ensure it's valid
                            return truncated_text.strip()

        # Fallback to random sampling
        random_sampler = RandomNegativeSampler()
        return random_sampler.sample(
            text, excluded_regions, tokenizer, min_token_length, max_token_length
        )

    def _calculate_context_score(self, sentence: str) -> float:
        """Calculate how similar a sentence is to scientific context."""
        sentence_lower = sentence.lower()

        # Count context keywords
        keyword_count = sum(
            1 for keyword in self.context_keywords if keyword in sentence_lower
        )

        # Normalize by sentence length (in words)
        word_count = len(sentence_lower.split())

        return keyword_count / max(word_count, 1)


class BalancedNegativeSampler:
    """Generate balanced negative samples using multiple strategies."""

    def __init__(self, strategies: list[NegativeSampler] | None = None):
        """
        Initialize balanced negative sampler.

        Args:
            strategies: List of sampling strategies to use
        """
        self.strategies = strategies or [
            RandomNegativeSampler(),
            ContextualNegativeSampler(),
        ]

    def sample(
        self,
        text: str,
        excluded_regions: list[dict],
        tokenizer,
        min_token_length: int,
        max_token_length: int,
    ) -> str:
        """
        Generate negative sample using random strategy selection.

        Args:
            text: Source text
            excluded_regions: Regions to avoid
            tokenizer: HuggingFace tokenizer for length validation
            min_token_length: Minimum token length
            max_token_length: Maximum token length

        Returns:
            Negative sample from randomly selected strategy
        """
        strategy = random.choice(self.strategies)
        return strategy.sample(
            text, excluded_regions, tokenizer, min_token_length, max_token_length
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_negative_sampler(strategy: str = "random", **kwargs) -> NegativeSampler:
    """
    Create negative sampler instance.

    Args:
        strategy: Sampling strategy ("random", "contextual", "balanced")
        **kwargs: Strategy-specific parameters

    Returns:
        Negative sampler instance
    """
    if strategy == "random":
        return RandomNegativeSampler(**kwargs)
    elif strategy == "contextual":
        return ContextualNegativeSampler(**kwargs)
    elif strategy == "balanced":
        return BalancedNegativeSampler(**kwargs)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


# =============================================================================
# Testing and Examples moved to tests/test_text_processing.py
# =============================================================================
