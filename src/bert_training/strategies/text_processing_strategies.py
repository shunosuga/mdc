#!/usr/bin/env python3
"""
Text Processing Strategies for BERT Training Data Generation

Implements dependency injection interfaces and example implementations for
text cropping and negative sampling strategies.
"""

import random
import re
from typing import Protocol


class TextCropper(Protocol):
    """Interface for text cropping strategies."""

    def crop(self, text: str, identifier_positions: list[dict], **kwargs) -> str:
        """
        Crop text around data identifiers.

        Args:
            text: Original text content
            identifier_positions: List of identifier position dicts
            **kwargs: Additional parameters (window_size, etc.)

        Returns:
            Cropped text segment
        """
        ...


class NegativeSampler(Protocol):
    """Interface for negative sample generation strategies."""

    def sample(self, text: str, excluded_regions: list[dict], **kwargs) -> str:
        """
        Generate negative samples from text.

        Args:
            text: Source text content
            excluded_regions: Regions to avoid (containing identifiers)
            **kwargs: Additional parameters

        Returns:
            Negative sample text segment
        """
        ...


# =============================================================================
# Text Cropping Implementations
# =============================================================================


class WindowCropper:
    """Crop text with fixed window around identifiers."""

    def __init__(self, window_size: int = 100):
        """
        Initialize window cropper.

        Args:
            window_size: Number of characters to include around identifiers
        """
        self.window_size = window_size

    def crop(self, text: str, identifier_positions: list[dict], **kwargs) -> str:
        """
        Crop text using fixed window around identifiers.

        Args:
            text: Original text
            identifier_positions: List of identifier positions
            **kwargs: Additional parameters (window_size override)

        Returns:
            Cropped text with window around identifiers
        """
        window_size = kwargs.get("window_size", self.window_size)

        if not identifier_positions:
            return text

        # Find the span that covers all identifiers
        min_start = min(pos["char_start"] for pos in identifier_positions)
        max_end = max(pos["char_end"] for pos in identifier_positions)

        # Expand window around the identifier span
        crop_start = max(0, min_start - window_size // 2)
        crop_end = min(len(text), max_end + window_size // 2)

        # Try to align to word boundaries
        crop_start = self._find_word_boundary(text, crop_start, direction="backward")
        crop_end = self._find_word_boundary(text, crop_end, direction="forward")

        return text[crop_start:crop_end].strip()

    def _find_word_boundary(
        self, text: str, position: int, direction: str = "forward"
    ) -> int:
        """Find nearest word boundary from position."""
        if direction == "forward":
            while position < len(text) and not text[position].isspace():
                position += 1
        else:  # backward
            while position > 0 and not text[position - 1].isspace():
                position -= 1

        return position


class SentenceCropper:
    """Crop text to complete sentences containing identifiers."""

    def __init__(self, sentence_context: int = 1):
        """
        Initialize sentence cropper.

        Args:
            sentence_context: Number of additional sentences to include on each side
        """
        self.sentence_context = sentence_context

    def crop(self, text: str, identifier_positions: list[dict], **kwargs) -> str:
        """
        Crop text to sentences containing identifiers plus context.

        Args:
            text: Original text
            identifier_positions: List of identifier positions
            **kwargs: Additional parameters (sentence_context override)

        Returns:
            Cropped text with complete sentences
        """
        sentence_context = kwargs.get("sentence_context", self.sentence_context)

        if not identifier_positions:
            return text

        # Split text into sentences
        sentences = self._split_sentences(text)

        # Find sentences containing identifiers
        identifier_sentences = set()

        for pos in identifier_positions:
            char_start = pos["char_start"]
            char_end = pos["char_end"]

            current_pos = 0
            for i, sentence in enumerate(sentences):
                sentence_start = current_pos
                sentence_end = current_pos + len(sentence)

                # Check if identifier overlaps with this sentence
                if not (char_end <= sentence_start or char_start >= sentence_end):
                    identifier_sentences.add(i)

                current_pos = sentence_end

        if not identifier_sentences:
            return text

        # Expand to include context sentences
        min_sentence = max(0, min(identifier_sentences) - sentence_context)
        max_sentence = min(
            len(sentences), max(identifier_sentences) + sentence_context + 1
        )

        return "".join(sentences[min_sentence:max_sentence]).strip()

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences preserving whitespace."""
        # Use regex to split on sentence endings while preserving the delimiter
        sentences = re.split(r"([.!?]+\s*)", text)

        # Rejoin sentence content with its delimiter
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence_content = sentences[i]
            delimiter = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence_content.strip():  # Only include non-empty sentences
                result.append(sentence_content + delimiter)

        return result


class ParagraphCropper:
    """Crop text to complete paragraphs containing identifiers."""

    def __init__(self, paragraph_context: int = 0):
        """
        Initialize paragraph cropper.

        Args:
            paragraph_context: Number of additional paragraphs to include
        """
        self.paragraph_context = paragraph_context

    def crop(self, text: str, identifier_positions: list[dict], **kwargs) -> str:
        """
        Crop text to paragraphs containing identifiers.

        Args:
            text: Original text
            identifier_positions: List of identifier positions
            **kwargs: Additional parameters

        Returns:
            Cropped text with complete paragraphs
        """
        paragraph_context = kwargs.get("paragraph_context", self.paragraph_context)

        if not identifier_positions:
            return text

        # Split text into paragraphs
        paragraphs = text.split("\n\n")

        # Find paragraphs containing identifiers
        identifier_paragraphs = set()

        current_pos = 0
        for i, paragraph in enumerate(paragraphs):
            paragraph_start = current_pos
            paragraph_end = current_pos + len(paragraph)

            for pos in identifier_positions:
                char_start = pos["char_start"]
                char_end = pos["char_end"]

                # Check if identifier is in this paragraph
                if not (char_end <= paragraph_start or char_start >= paragraph_end):
                    identifier_paragraphs.add(i)

            current_pos = paragraph_end + 2  # +2 for \n\n

        if not identifier_paragraphs:
            return text

        # Expand to include context paragraphs
        min_paragraph = max(0, min(identifier_paragraphs) - paragraph_context)
        max_paragraph = min(
            len(paragraphs), max(identifier_paragraphs) + paragraph_context + 1
        )

        return "\n\n".join(paragraphs[min_paragraph:max_paragraph]).strip()


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

    def sample(self, text: str, excluded_regions: list[dict], **kwargs) -> str:
        """
        Generate random negative sample avoiding excluded regions.

        Args:
            text: Source text
            excluded_regions: Regions containing identifiers to avoid
            **kwargs: Additional parameters

        Returns:
            Random text sample without identifiers
        """
        sample_length = kwargs.get("sample_length", self.sample_length)
        min_gap = kwargs.get("min_gap", self.min_gap)

        if len(text) < sample_length:
            return text

        # Create list of valid start positions
        valid_positions = []

        for start_pos in range(len(text) - sample_length + 1):
            end_pos = start_pos + sample_length

            # Check if this sample would overlap with any excluded region
            valid = True
            for region in excluded_regions:
                region_start = max(0, region["char_start"] - min_gap)
                region_end = region["char_end"] + min_gap

                # Check for overlap
                if not (end_pos <= region_start or start_pos >= region_end):
                    valid = False
                    break

            if valid:
                valid_positions.append(start_pos)

        if not valid_positions:
            # If no valid positions, return text as-is (fallback)
            return text

        # Choose random valid position
        start_pos = random.choice(valid_positions)
        sample_text = text[start_pos : start_pos + sample_length]

        # Try to align to word boundaries
        sample_text = self._align_to_words(text, start_pos, sample_text)

        return sample_text.strip()

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

    def sample(self, text: str, excluded_regions: list[dict], **kwargs) -> str:
        """
        Generate negative sample from similar scientific context.

        Args:
            text: Source text
            excluded_regions: Regions to avoid
            **kwargs: Additional parameters

        Returns:
            Contextual negative sample
        """
        context_similarity = kwargs.get("context_similarity", 0.8)
        sample_length = kwargs.get("sample_length", 100)

        # Split text into sentences
        sentences = re.split(r"[.!?]+\s+", text)

        # Score sentences based on scientific context
        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue

            score = self._calculate_context_score(sentence)
            sentence_scores.append((sentence, score))

        # Sort by score and filter by similarity threshold
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Find sentences that don't contain identifiers
        valid_sentences = []
        for sentence, score in sentence_scores:
            if score >= context_similarity:
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
                        valid_sentences.append(sentence)

        if valid_sentences:
            return random.choice(valid_sentences).strip()
        else:
            # Fallback to random sampling
            random_sampler = RandomNegativeSampler(sample_length)
            return random_sampler.sample(text, excluded_regions, **kwargs)

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

    def sample(self, text: str, excluded_regions: list[dict], **kwargs) -> str:
        """
        Generate negative sample using random strategy selection.

        Args:
            text: Source text
            excluded_regions: Regions to avoid
            **kwargs: Additional parameters

        Returns:
            Negative sample from randomly selected strategy
        """
        strategy = random.choice(self.strategies)
        return strategy.sample(text, excluded_regions, **kwargs)


# =============================================================================
# Factory Functions
# =============================================================================


def create_text_cropper(strategy: str = "window", **kwargs) -> TextCropper:
    """
    Create text cropper instance.

    Args:
        strategy: Cropping strategy ("window", "sentence", "paragraph")
        **kwargs: Strategy-specific parameters

    Returns:
        Text cropper instance
    """
    if strategy == "window":
        return WindowCropper(**kwargs)
    elif strategy == "sentence":
        return SentenceCropper(**kwargs)
    elif strategy == "paragraph":
        return ParagraphCropper(**kwargs)
    else:
        raise ValueError(f"Unknown cropping strategy: {strategy}")


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
# Testing and Examples
# =============================================================================


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
