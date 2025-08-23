#!/usr/bin/env python3
"""
BERT Training Data Generator for Data Identifier Detection

Generates token-level training data from PMC full-text papers for BERT-based
data identifier detection using AutoModelForTokenClassification.
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Protocol

import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer


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


class BERTTrainingDataGenerator:
    """Generate BERT training data for data identifier detection."""

    def __init__(
        self,
        pmc_dir: str,
        corpus_file: str,
        pmc_ids_file: str,
        tokenizer: AutoTokenizer,
    ):
        """
        Initialize the training data generator.

        Args:
            pmc_dir: Path to PMC text files directory
            corpus_file: Path to corpus_consolidated.json
            pmc_ids_file: Path to PMC-ids.csv
            tokenizer: HuggingFace tokenizer instance
        """
        self.pmc_dir = Path(pmc_dir)
        self.corpus_file = corpus_file
        self.pmc_ids_file = pmc_ids_file
        self.tokenizer = tokenizer

        # Load data mappings
        self.doi_to_datasets = {}
        self.pmc_to_doi = {}
        self._load_data_mappings()

    def _load_data_mappings(self):
        """Load corpus data and PMC-DOI mappings."""
        print("=== Loading data mappings ===")

        # Load corpus data
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        print(f"Loaded {len(corpus_data)} records from corpus file")

        for record in tqdm(corpus_data, desc="Processing corpus records"):
            publication = record.get("publication", "").strip()
            datasets = record.get("datasets", [])

            if publication and datasets:
                # DOI normalization - extract 10.xxxx prefix only
                doi = publication.lower()

                # Remove protocol prefixes
                doi = doi.replace("https://doi.org/", "").replace(
                    "http://dx.doi.org/", ""
                )
                doi = doi.replace("https://", "").replace("http://", "")

                # Extract DOI pattern starting with 10.
                if "10." in doi:
                    doi_start = doi.find("10.")
                    if doi_start != -1:
                        doi = doi[doi_start:]
                        doi = doi.split()[0]  # Get first word only

                for dataset_id in datasets:
                    if dataset_id and isinstance(dataset_id, str):
                        dataset_id = dataset_id.strip()
                        if dataset_id and dataset_id != "nan":
                            if doi not in self.doi_to_datasets:
                                self.doi_to_datasets[doi] = []
                            self.doi_to_datasets[doi].append(dataset_id)

        # Load PMC-DOI mapping with error handling for mixed types
        try:
            # Use schema_overrides to handle mixed column types
            df = pl.read_csv(
                self.pmc_ids_file,
                ignore_errors=True,
                infer_schema_length=10000,
                schema_overrides={
                    "Page": pl.Utf8,  # Treat Page column as string to avoid parsing errors
                    "Issue": pl.Utf8,  # Treat Issue column as string
                    "Volume": pl.Utf8,  # Treat Volume column as string
                },
            )

            for row in df.iter_rows(named=True):
                if row.get("DOI") and row.get("PMCID"):
                    pmcid = str(row["PMCID"]).replace("PMC", "")
                    doi = str(row["DOI"]).strip().lower()
                    self.pmc_to_doi[pmcid] = doi

            print(f"Loaded {len(self.pmc_to_doi)} PMC-DOI mappings")
        except Exception as e:
            print(f"Error loading PMC-DOI mapping: {e}")

    def _find_pmc_text_files(self) -> list[Path]:
        """Find PMC text files."""
        print("=== Finding PMC text files ===")

        text_files = []

        # Try different possible directory structures
        possible_paths = [
            self.pmc_dir / "txt",  # Standard structure: pmc_dir/txt/PMC*/PMC*.txt
            self.pmc_dir,  # Direct structure: pmc_dir/PMC*/PMC*.txt
        ]

        base_path = None
        for path in possible_paths:
            if path.exists():
                base_path = path
                break

        if base_path is None:
            print(f"Warning: No valid PMC directory found in {self.pmc_dir}")
            return text_files

        for subdir in base_path.iterdir():
            if subdir.is_dir() and "PMC" in subdir.name:
                for txt_file in subdir.glob("PMC*.txt"):
                    text_files.append(txt_file)

        print(f"Found {len(text_files)} PMC text files")
        return text_files

    def _create_search_patterns(self, dataset_id: str) -> list[str]:
        """Create search patterns for data identifiers."""
        patterns = []

        # DOI/URL special handling
        if dataset_id.startswith(("http://", "https://")) or "10." in dataset_id:
            clean_id = dataset_id
            if clean_id.startswith(("http://", "https://")):
                clean_id = clean_id.replace("https://", "").replace("http://", "")

            if "10." in clean_id:
                doi_start = clean_id.find("10.")
                if doi_start != -1:
                    clean_id = clean_id[doi_start:]
                    clean_id = clean_id.split()[0]

            patterns.extend(
                [
                    re.escape(clean_id.lower()),
                    re.escape(clean_id.upper()),
                    re.escape(clean_id),
                ]
            )
        else:
            # Regular identifier handling
            patterns.extend(
                [
                    re.escape(dataset_id),
                    re.escape(dataset_id.upper()),
                    re.escape(dataset_id.lower()),
                ]
            )

        # Add spaced variations
        spaced_id = re.sub(r"([A-Za-z])(\d)", r"\1 \2", dataset_id)
        if spaced_id != dataset_id:
            patterns.append(re.escape(spaced_id))

        # Add dot/hyphen variations
        if "." in dataset_id or "-" in dataset_id:
            alt_id = dataset_id.replace(".", " ").replace("-", " ")
            patterns.append(re.escape(alt_id))

        return patterns

    def _find_identifiers_in_text(
        self, text: str, dataset_ids: list[str]
    ) -> list[dict]:
        """Find data identifiers in text with position information."""
        found_identifiers = []

        for dataset_id in dataset_ids:
            patterns = self._create_search_patterns(dataset_id)

            for pattern in patterns:
                regex_pattern = r"\b" + pattern + r"\b"

                try:
                    matches = list(re.finditer(regex_pattern, text, re.IGNORECASE))
                    for match in matches:
                        found_identifiers.append(
                            {
                                "identifier": dataset_id,
                                "pattern": pattern,
                                "char_start": match.start(),
                                "char_end": match.end(),
                                "matched_text": match.group(),
                            }
                        )
                        break  # One match per identifier is enough
                except re.error:
                    # Fall back to simple string search
                    if dataset_id.lower() in text.lower():
                        start_idx = text.lower().find(dataset_id.lower())
                        if start_idx != -1:
                            found_identifiers.append(
                                {
                                    "identifier": dataset_id,
                                    "pattern": "simple_match",
                                    "char_start": start_idx,
                                    "char_end": start_idx + len(dataset_id),
                                    "matched_text": text[
                                        start_idx : start_idx + len(dataset_id)
                                    ],
                                }
                            )
                        break

        return found_identifiers

    def _segment_text(self, text: str, text_unit: str = "sentence") -> list[str]:
        """Segment text into sentences or paragraphs."""
        if text_unit == "sentence":
            # Simple sentence segmentation
            sentences = re.split(r"[.!?]+\s+", text)
            return [s.strip() for s in sentences if s.strip()]
        elif text_unit == "paragraph":
            paragraphs = text.split("\n\n")
            return [p.strip() for p in paragraphs if p.strip()]
        else:
            return [text]  # Return whole text

    def _tokenize_with_spans(self, text: str) -> dict:
        """Tokenize text and track character spans."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        # Get tokens and their spans
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        token_spans = []
        for i, (start, end) in enumerate(encoding["offset_mapping"]):
            token_spans.append({"start": start, "end": end})

        return {
            "tokens": tokens,
            "token_spans": token_spans,
            "input_ids": encoding["input_ids"],
        }

    def _assign_token_labels(
        self,
        tokens: list[str],
        token_spans: list[dict],
        identifier_positions: list[dict],
    ) -> list[int]:
        """Assign binary labels to tokens based on identifier positions."""
        labels = [0] * len(tokens)

        for identifier_pos in identifier_positions:
            char_start = identifier_pos["char_start"]
            char_end = identifier_pos["char_end"]

            # Find tokens that overlap with identifier
            for i, token_span in enumerate(token_spans):
                token_start = token_span["start"]
                token_end = token_span["end"]

                # Check if token overlaps with identifier
                if token_start < char_end and token_end > char_start:
                    labels[i] = 1

        return labels

    def _create_training_sample(
        self,
        text: str,
        identifier_positions: list[dict],
        source_pmcid: str,
        source_doi: str,
        sample_type: str = "positive",
    ) -> dict | None:
        """Create a single training sample."""
        try:
            tokenization = self._tokenize_with_spans(text)
            tokens = tokenization["tokens"]  # Token strings for debugging/readability
            input_ids = tokenization["input_ids"]  # Token IDs for training
            token_spans = tokenization["token_spans"]

            if sample_type == "positive":
                labels = self._assign_token_labels(
                    tokens, token_spans, identifier_positions
                )
                expected_identifiers = [
                    pos["identifier"] for pos in identifier_positions
                ]
            else:
                labels = [0] * len(tokens)
                expected_identifiers = []

            # Create identifier spans for positive samples
            identifier_spans = []
            if sample_type == "positive":
                for pos in identifier_positions:
                    # Find token indices that correspond to this identifier
                    token_start = None
                    token_end = None

                    for i, span in enumerate(token_spans):
                        if span["start"] <= pos["char_start"] < span["end"]:
                            token_start = i
                        if span["start"] < pos["char_end"] <= span["end"]:
                            token_end = i + 1
                            break

                    if token_start is not None and token_end is not None:
                        identifier_spans.append(
                            {
                                "identifier": pos["identifier"],
                                "char_start": pos["char_start"],
                                "char_end": pos["char_end"],
                                "token_start": token_start,
                                "token_end": token_end,
                            }
                        )

            return {
                "input_ids": input_ids,  # Token IDs for training (list[int])
                "tokens": tokens,  # Token strings for debugging (list[str])
                "labels": labels,
                "original_text": text,
                "expected_identifiers": expected_identifiers,
                "token_spans": token_spans,
                "identifier_spans": identifier_spans,
                "source_pmcid": source_pmcid,
                "source_doi": source_doi,
                "metadata": {
                    "token_count": len(tokens),
                    "identifier_count": len(expected_identifiers),
                    "sample_type": sample_type,
                    "text_type": "sentence",  # or paragraph
                },
            }

        except Exception as e:
            print(f"Error creating sample: {e}")
            return None

    def generate(
        self,
        num_samples: int = 10000,
        positive_ratio: float = 0.7,
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        text_unit: str = "sentence",
        min_token_length: int = 10,
        max_token_length: int = 128,
        output_format: str = "jsonl",
        output_path: str = "training_data.jsonl",
        random_seed: int = 42,
        include_restoration_test: bool = True,
        text_cropper: TextCropper | None = None,
        negative_sampler: NegativeSampler | None = None,
    ) -> dict:
        """
        Generate BERT training data efficiently.

        Args:
            num_samples: Total number of samples to generate across all files
            positive_ratio: Ratio of positive samples (0.0-1.0)
            tokenizer_name: HuggingFace tokenizer name
            text_unit: Text segmentation unit ("sentence" | "paragraph")
            min_token_length: Minimum token length
            max_token_length: Maximum token length
            output_format: Output format ("jsonl" | "json")
            output_path: Output file path
            random_seed: Random seed for reproducibility
            include_restoration_test: Whether to test restoration accuracy
            text_cropper: Text cropping strategy
            negative_sampler: Negative sampling strategy

        Returns:
            Generation results with statistics and test results
        """
        print("=== BERT Training Data Generation (Efficient) ===")
        print(f"Target samples: {num_samples} (positive ratio: {positive_ratio})")

        # Set random seed
        random.seed(random_seed)

        # Initialize tokenizer
        if self.tokenizer is None:
            print(f"Loading tokenizer: {tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Calculate targets
        target_positive = int(num_samples * positive_ratio)
        target_negative = num_samples - target_positive

        print(f"Target: {target_positive} positive, {target_negative} negative samples")

        # Find and shuffle all text files
        print("=== Finding and shuffling files ===")
        text_files = self._find_pmc_text_files()
        random.shuffle(text_files)
        print(f"Available files: {len(text_files)}")

        # Phase 1: Generate positive samples
        print(f"\n=== Phase 1: Generating {target_positive} positive samples ===")
        positive_samples = self._generate_positive_samples_efficient(
            text_files,
            target_positive,
            min_token_length,
            max_token_length,
            text_cropper,
        )

        # Phase 2: Generate negative samples
        print(f"\n=== Phase 2: Generating {target_negative} negative samples ===")
        negative_samples = self._generate_negative_samples_efficient(
            text_files,
            target_negative,
            min_token_length,
            max_token_length,
            negative_sampler,
        )

        # Combine and shuffle all samples
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)

        print("\n=== Generation Complete ===")
        print(f"Generated {len(all_samples)} samples total")
        print(f"Positive: {len(positive_samples)}, Negative: {len(negative_samples)}")

        # Test restoration if requested
        restoration_results = {}
        if include_restoration_test:
            print("\nTesting restoration accuracy...")
            restoration_results = self._test_restoration_accuracy(all_samples)

        # Save results
        print(f"\nSaving to {output_path}...")
        self._save_samples(all_samples, output_path, output_format)

        # Create summary
        generation_summary = {
            "total_samples": len(all_samples),
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples),
            "available_files": len(text_files),
            "average_tokens_per_sample": sum(len(s["input_ids"]) for s in all_samples)
            / len(all_samples)
            if all_samples
            else 0,
            "unique_identifiers_found": len(
                set(
                    identifier
                    for sample in all_samples
                    for identifier in sample["expected_identifiers"]
                )
            ),
            "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parameters": {
                "tokenizer_name": tokenizer_name,
                "num_samples": num_samples,
                "positive_ratio": positive_ratio,
                "max_token_length": max_token_length,
                "random_seed": random_seed,
            },
        }

        # Save summary and test results
        summary_file = output_path.replace(".jsonl", "_summary.json").replace(
            ".json", "_summary.json"
        )
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generation_summary": generation_summary,
                    "restoration_test_results": restoration_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "generation_summary": generation_summary,
            "restoration_test_results": restoration_results,
            "output_files": [output_path, summary_file],
        }

    def _generate_positive_samples_efficient(
        self,
        text_files: list[Path],
        target_count: int,
        min_token_length: int,
        max_token_length: int,
        text_cropper: TextCropper | None,
    ) -> list[dict]:
        """
        Efficiently generate positive samples by processing whole files at once.

        Args:
            text_files: List of PMC text files
            target_count: Number of positive samples to generate
            min_token_length: Minimum token length
            max_token_length: Maximum token length
            text_cropper: Text cropping strategy

        Returns:
            List of positive samples
        """
        positive_samples = []
        processed_files = 0

        for text_file in tqdm(text_files, desc="Generating positive samples"):
            if len(positive_samples) >= target_count:
                break

            try:
                # Get PMC ID and DOI
                pmcid = text_file.stem.replace("PMC", "")
                doi = self.pmc_to_doi.get(pmcid)
                if not doi:
                    continue

                # Get expected datasets for this DOI
                expected_datasets = self.doi_to_datasets.get(doi, [])
                if not expected_datasets:
                    continue

                # Read entire file
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

                # Find ALL identifiers in the entire text at once
                all_identifiers = self._find_identifiers_in_text(
                    full_text, expected_datasets
                )

                if not all_identifiers:
                    continue

                # Randomly select one identifier as the focus
                focus_identifier = random.choice(all_identifiers)

                # Extract text around the focus identifier
                sample_text = self._extract_text_around_identifier(
                    full_text, focus_identifier, max_token_length, text_cropper
                )

                if not sample_text:
                    continue

                # Find all identifiers in the extracted sample text
                sample_identifiers = self._find_identifiers_in_text(
                    sample_text, expected_datasets
                )

                if not sample_identifiers:
                    continue

                # Create training sample
                sample = self._create_training_sample(
                    sample_text,
                    sample_identifiers,
                    f"PMC{pmcid}",
                    doi,
                    "positive",
                )

                # Validate sample length
                if (
                    sample
                    and min_token_length <= len(sample["input_ids"]) <= max_token_length
                ):
                    positive_samples.append(sample)

                processed_files += 1

                # Progress update
                if processed_files % 100 == 0:
                    print(
                        f"Progress: {len(positive_samples)}/{target_count} positive samples from {processed_files} files"
                    )

            except Exception as e:
                print(f"Error processing {text_file}: {e}")
                continue

        print(
            f"Generated {len(positive_samples)} positive samples from {processed_files} files"
        )
        return positive_samples

    def _generate_negative_samples_efficient(
        self,
        text_files: list[Path],
        target_count: int,
        min_token_length: int,
        max_token_length: int,
        negative_sampler: NegativeSampler | None,
    ) -> list[dict]:
        """
        Efficiently generate negative samples by avoiding identifier regions.

        Args:
            text_files: List of PMC text files
            target_count: Number of negative samples to generate
            min_token_length: Minimum token length
            max_token_length: Maximum token length
            negative_sampler: Negative sampling strategy

        Returns:
            List of negative samples
        """
        negative_samples = []
        processed_files = 0

        for text_file in tqdm(text_files, desc="Generating negative samples"):
            if len(negative_samples) >= target_count:
                break

            try:
                # Get PMC ID and DOI
                pmcid = text_file.stem.replace("PMC", "")
                doi = self.pmc_to_doi.get(pmcid)
                if not doi:
                    continue

                # Get expected datasets for this DOI
                expected_datasets = self.doi_to_datasets.get(doi, [])

                # Read entire file
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()

                # Find ALL identifiers in the entire text to avoid them
                all_identifiers = self._find_identifiers_in_text(
                    full_text, expected_datasets
                )

                # Extract text that doesn't contain identifiers
                sample_text = self._extract_text_without_identifiers(
                    full_text, all_identifiers, max_token_length, negative_sampler
                )

                if not sample_text:
                    continue

                # Double-check: ensure no identifiers in sample
                sample_identifiers = self._find_identifiers_in_text(
                    sample_text, expected_datasets
                )
                if sample_identifiers:
                    continue  # Skip if identifiers found

                # Create training sample
                sample = self._create_training_sample(
                    sample_text,
                    [],  # No identifiers for negative sample
                    f"PMC{pmcid}",
                    doi,
                    "negative",
                )

                # Validate sample length
                if (
                    sample
                    and min_token_length <= len(sample["input_ids"]) <= max_token_length
                ):
                    negative_samples.append(sample)

                processed_files += 1

                # Progress update
                if processed_files % 100 == 0:
                    print(
                        f"Progress: {len(negative_samples)}/{target_count} negative samples from {processed_files} files"
                    )

            except Exception as e:
                print(f"Error processing {text_file}: {e}")
                continue

        print(
            f"Generated {len(negative_samples)} negative samples from {processed_files} files"
        )
        return negative_samples

    # TODO: 修正
    def _extract_text_without_identifiers(
        self, text: str, identifiers: list[dict], max_length: int = 128
    ) -> tuple[str, list[int]]:
        """
        Extract text that doesn't contain any identifiers.

        Args:
            text: Full text content
            identifiers: List of identifier dicts with 'start', 'end' keys
            max_length: Maximum token length

        Returns:
            Tuple of (extracted_text, token_labels)
        """
        if not identifiers:
            # No identifiers, can extract from anywhere
            start_pos = random.randint(0, max(0, len(text) - max_length * 4))
            end_pos = min(len(text), start_pos + max_length * 4)
            extracted_text = text[start_pos:end_pos]

            # Tokenize and truncate
            tokens = self.tokenizer.tokenize(extracted_text)
            if len(tokens) > max_length - 2:
                tokens = tokens[: max_length - 2]
                extracted_text = self.tokenizer.convert_tokens_to_string(tokens)

            labels = [0] * len(tokens)
            return extracted_text, labels

        # Sort identifiers by position
        sorted_identifiers = sorted(identifiers, key=lambda x: x["start"])

        # Find safe regions between identifiers
        safe_regions = []

        # Region before first identifier
        if sorted_identifiers[0]["start"] > max_length * 4:
            safe_regions.append((0, sorted_identifiers[0]["start"]))

        # Regions between identifiers
        for i in range(len(sorted_identifiers) - 1):
            current_end = sorted_identifiers[i]["end"]
            next_start = sorted_identifiers[i + 1]["start"]

            if next_start - current_end > max_length * 4:
                safe_regions.append((current_end, next_start))

        # Region after last identifier
        last_end = sorted_identifiers[-1]["end"]
        if len(text) - last_end > max_length * 4:
            safe_regions.append((last_end, len(text)))

        if not safe_regions:
            # No safe regions found, return empty
            return "", []

        # Randomly select a safe region
        region_start, region_end = random.choice(safe_regions)

        # Extract text from the safe region
        window_size = max_length * 4
        if region_end - region_start > window_size:
            # Region is large enough, extract random portion
            extract_start = random.randint(region_start, region_end - window_size)
            extract_end = extract_start + window_size
        else:
            # Use entire region
            extract_start = region_start
            extract_end = region_end

        extracted_text = text[extract_start:extract_end]

        # Tokenize and truncate
        tokens = self.tokenizer.tokenize(extracted_text)
        if len(tokens) > max_length - 2:
            tokens = tokens[: max_length - 2]
            extracted_text = self.tokenizer.convert_tokens_to_string(tokens)

        labels = [0] * len(tokens)
        return extracted_text, labels

    # TODO: TextCropperを使うようにする
    def _extract_text_around_identifier(
        self, text: str, identifier: dict, max_length: int = 128
    ) -> tuple[str, list[int]]:
        """
        Extract text around a specific identifier with proper tokenization.

        Args:
            text: Full text content
            identifier: Dict with 'start', 'end', 'text', 'type' keys
            max_length: Maximum token length

        Returns:
            Tuple of (extracted_text, token_labels)
        """

        # Calculate window around identifier
        identifier_start = identifier["start"]
        identifier_end = identifier["end"]
        identifier_text = identifier["text"]

        # Estimate token positions (rough approximation)
        window_chars = max_length * 4  # Rough chars per token estimate
        # TODO: これはさすがにダメすぎ。 tokenizerでトークン化してからやるべき。
        # この方法だと、常にtextの中心にidentifierが来ることになる。
        text_start = max(0, identifier_start - window_chars // 2)
        text_end = min(len(text), identifier_end + window_chars // 2)

        # Extract surrounding text
        extracted_text = text[text_start:text_end]

        # Tokenize the extracted text
        tokens = self.tokenizer.tokenize(extracted_text)

        # Truncate to max_length if necessary
        if len(tokens) > max_length - 2:  # Account for [CLS] and [SEP]
            tokens = tokens[: max_length - 2]
            extracted_text = self.tokenizer.convert_tokens_to_string(tokens)

        # Create labels - find identifier position in tokenized text
        labels = [0] * len(tokens)

        # Find where the identifier appears in the tokenized text
        identifier_relative_start = identifier_start - text_start
        identifier_relative_end = identifier_end - text_start

        if identifier_relative_start >= 0 and identifier_relative_start < len(
            extracted_text
        ):
            # Tokenize up to identifier start to find token position
            prefix_tokens = self.tokenizer.tokenize(
                extracted_text[:identifier_relative_start]
            )
            identifier_tokens = self.tokenizer.tokenize(identifier_text)

            start_token_idx = len(prefix_tokens)
            end_token_idx = min(start_token_idx + len(identifier_tokens), len(tokens))

            # Mark identifier tokens as 1
            for i in range(start_token_idx, end_token_idx):
                labels[i] = 1

        return extracted_text, labels

    def _extract_text_without_identifiers(
        self,
        full_text: str,
        all_identifiers: list[dict],
        max_token_length: int,
        negative_sampler: NegativeSampler | None,
    ) -> str:
        """
        Extract text that doesn't contain any identifiers.

        Args:
            full_text: Full text content
            all_identifiers: List of all identifiers to avoid
            max_token_length: Maximum token length
            negative_sampler: Negative sampling strategy

        Returns:
            Extracted text without identifiers
        """
        # Use negative sampler if provided
        # TODO: ほぼこれを使うようにする
        if negative_sampler:
            return negative_sampler.sample(full_text, all_identifiers)

        # Default: find text regions without identifiers
        if not all_identifiers:
            # If no identifiers, extract random segment
            return self._extract_random_text_segment(full_text, max_token_length)

        # Find safe regions between identifiers
        safe_regions = self._find_safe_text_regions(full_text, all_identifiers)

        if not safe_regions:
            return ""

        # Select random safe region
        region = random.choice(safe_regions)
        region_text = full_text[region["start"] : region["end"]]

        # Truncate if too long
        chars_per_token = 4
        max_chars = max_token_length * chars_per_token

        if len(region_text) > max_chars:
            # Extract random portion
            start_pos = random.randint(0, max(0, len(region_text) - max_chars))
            region_text = region_text[start_pos : start_pos + max_chars]

            # Align to word boundaries
            region_text = self._align_to_word_boundaries(region_text)

        return region_text.strip()

    def _extract_random_text_segment(
        self, full_text: str, max_token_length: int
    ) -> str:
        """Extract a random segment of text."""
        chars_per_token = 4
        max_chars = max_token_length * chars_per_token

        if len(full_text) <= max_chars:
            return full_text

        start_pos = random.randint(0, len(full_text) - max_chars)
        segment = full_text[start_pos : start_pos + max_chars]

        return self._align_to_word_boundaries(segment)

    def _find_safe_text_regions(
        self, full_text: str, identifiers: list[dict]
    ) -> list[dict]:
        """Find text regions that don't contain identifiers."""
        if not identifiers:
            return [{"start": 0, "end": len(full_text)}]

        # Sort identifiers by position
        sorted_identifiers = sorted(identifiers, key=lambda x: x["char_start"])

        safe_regions = []
        min_region_size = 100  # Minimum characters for a safe region

        # Region before first identifier
        if sorted_identifiers[0]["char_start"] > min_region_size:
            safe_regions.append(
                {
                    "start": 0,
                    "end": sorted_identifiers[0]["char_start"] - 10,  # Small buffer
                }
            )

        # Regions between identifiers
        for i in range(len(sorted_identifiers) - 1):
            current_end = sorted_identifiers[i]["char_end"]
            next_start = sorted_identifiers[i + 1]["char_start"]

            if next_start - current_end > min_region_size:
                safe_regions.append(
                    {
                        "start": current_end + 10,  # Small buffer
                        "end": next_start - 10,
                    }
                )

        # Region after last identifier
        last_end = sorted_identifiers[-1]["char_end"]
        if len(full_text) - last_end > min_region_size:
            safe_regions.append({"start": last_end + 10, "end": len(full_text)})

        return safe_regions

    def _align_to_word_boundaries(self, text: str) -> str:
        """Align text to word boundaries."""
        # Find start of first complete word
        start = 0
        while start < len(text) and not text[start].isspace() and start > 0:
            start += 1

        # Find end of last complete word
        end = len(text)
        while end > start and not text[end - 1].isspace() and end < len(text):
            end -= 1

        return text[start:end].strip()

    def _save_samples(self, samples: list[dict], output_path: str, output_format: str):
        """Save generated samples to file."""
        if output_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        elif output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"samples": samples}, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _test_restoration_accuracy(self, samples: list[dict]) -> dict:
        """Test restoration accuracy on generated samples."""
        try:
            from .restoration_tester import RestorationTester
        except ImportError:
            # Handle direct script execution
            import sys
            from pathlib import Path

            sys.path.append(str(Path(__file__).parent))
            from restoration_tester import RestorationTester

        tester = RestorationTester()
        return tester.test_dataset(samples)


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    generator = BERTTrainingDataGenerator(
        pmc_dir="data/pmc",  # Updated to correct base path
        corpus_file="data/corpus/corpus_consolidated.json",
        pmc_ids_file="data/pmc/PMC-ids.csv",
        tokenizer=tokenizer,
    )

    result = generator.generate(
        num_samples=1000,
        positive_ratio=0.7,
        output_path="training_data/bert_samples.jsonl",
        random_seed=42,
    )

    print("Generation completed!")
    print(
        f"Restoration accuracy: {result['restoration_test_results'].get('exact_match_rate', 0):.3f}"
    )
