# BERT Training Data Generation Plan

## Overview

This document outlines the implementation plan for generating training data for BERT-based data identifier detection. The goal is to create balanced positive and negative samples from PMC full-text papers for training a token classification model.

## Architecture

### Data Flow
```
PMC Text Files → Text Processing → Identifier Detection → Sample Generation → Training Dataset
     ↓              ↓                    ↓                  ↓                ↓
corpus_consolidated.json → DOI Mapping → Pattern Matching → Balancing → JSON/CSV Output
```

## Data Structure

### Token-Level Classification Design

Since we're using `AutoModelForTokenClassification` for binary classification of each token, the data structure is designed around tokenized text with corresponding labels.

### Output Data Items (Generated Samples)
Each training sample will contain:

- **tokens** (`list[str]`): BERT tokenizer output tokens
  ```python
  ["Data", "available", "under", "GS", "##E", "##12345", "in", "the", "repository"]
  ```
- **labels** (`list[int]`): Token-level binary labels (0 | 1), where 1 indicates data identifier
  ```python
  [0, 0, 0, 1, 1, 1, 0, 0, 0]
  ```
- **original_text** (`str`): Original text before tokenization
- **expected_identifiers** (`list[str]`): Ground truth data identifiers
  ```python
  ["GSE12345"]
  ```
- **token_spans** (`list[dict]`): Character-level spans for each token
  ```python
  [{"start": 0, "end": 4}, {"start": 5, "end": 14}, {"start": 15, "end": 20}, ...]
  ```
- **identifier_spans** (`list[dict]`): Character spans of original identifiers
  ```python
  [{"identifier": "GSE12345", "char_start": 21, "char_end": 29, "token_start": 3, "token_end": 6}]
  ```
- **source_pmcid** (`str`): Original PMC ID (e.g., "PMC1234567")
- **source_doi** (`str`): Corresponding DOI (e.g., "10.1234/example")
- **metadata** (`dict`): Additional information
  ```python
  {
    "token_count": 9, 
    "identifier_count": 1,
    "sample_type": "positive",
    "text_type": "sentence"
  }
  ```

### User Configurable Parameters
Users can specify:

- **num_samples** (`int`): Total number of samples to generate (default: 10000)
- **positive_ratio** (`float`): Ratio of positive samples (0.0-1.0, default: 0.7)
- **tokenizer_name** (`str`): HuggingFace tokenizer name (default: "allenai/scibert_scivocab_uncased")
- **text_unit** (`str`): Text segmentation unit ("sentence" | "paragraph", default: "sentence")
- **min_token_length** (`int`): Minimum text length in tokens (default: 10)
- **max_token_length** (`int`): Maximum text length in tokens (default: 128)
- **output_format** (`str`): Output format ("json" | "jsonl" | "parquet", default: "jsonl")
- **output_path** (`str`): Output file path
- **random_seed** (`int`): Random seed for reproducibility (default: 42)
- **include_restoration_test** (`bool`): Whether to include restoration test data (default: True)
- **text_cropper** (`callable`): Function for cropping text around identifiers (dependency injection)
- **negative_sampler** (`callable`): Function for generating negative samples (dependency injection)

## Implementation Strategy

### Phase 1: Text Processing & Tokenization
1. **PMC Text Reader**: Load and parse PMC text files
2. **Text Segmentation**: Split text into sentences or paragraphs
3. **DOI Mapping**: Use corpus_consolidated.json to map PMC IDs to expected identifiers
4. **Tokenization**: Apply BERT tokenizer to text segments
5. **Token Span Mapping**: Track character-level spans for each token

### Phase 2: Identifier Detection & Token Labeling
1. **Pattern Matching**: Reuse logic from `verify_data_identifiers.py`
2. **Token-Level Labeling**: Map character-level identifier positions to token labels
3. **Label Validation**: Ensure consistent labeling across subword tokens
4. **Multi-Token Identifier Handling**: Handle identifiers split across multiple tokens

### Phase 3: Sample Generation & Quality Control
1. **Positive Sample Creation**: Generate samples with label=1 tokens
2. **Negative Sample Selection**: Create samples with only label=0 tokens
3. **Text Cropping**: Apply configurable text cropper around identifiers
4. **Negative Sampling**: Apply configurable negative sample generation strategy
5. **Token Length Filtering**: Filter by token count (min/max limits)
6. **Balancing**: Maintain specified positive/negative ratio

### Phase 4: Restoration Testing & Export
1. **Restoration Algorithm**: Implement token-to-identifier reconstruction
2. **Accuracy Testing**: Verify restoration accuracy against ground truth
3. **Data Export**: Export to specified format (JSONL recommended for streaming)
4. **Validation Reports**: Generate quality and accuracy statistics

## Identifier Restoration Algorithm

### Core Algorithm
The restoration process converts token predictions back to original identifiers:

```python
def restore_identifiers(tokens, labels):
    """
    Restore data identifiers from tokenized predictions.
    
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
                identifiers.append(identifier)
                current_tokens = []
    
    # Handle case where sequence ends with label 1
    if current_tokens:
        identifier = reconstruct_from_subwords(current_tokens)
        identifiers.append(identifier)
    
    return identifiers

def reconstruct_from_subwords(tokens):
    """Reconstruct original text from BERT subword tokens."""
    text = ""
    for token in tokens:
        if token.startswith("##"):
            text += token[2:]  # Remove ## prefix
        else:
            text += token
    return text
```

### Word Boundary Consideration
As specified, if any token is predicted as "DATA_ID", we extend to the next word boundary:

```python
def extend_to_word_boundary(tokens, labels, original_text, token_spans):
    """Extend label=1 predictions to complete word boundaries."""
    extended_labels = labels.copy()
    
    for i, label in enumerate(labels):
        if label == 1:  # Binary label: 1 indicates data identifier
            # Find word boundaries and extend prediction
            # Implementation details for boundary detection...
            pass
    
    return extended_labels
```

### Restoration Testing Framework

```python
class RestorationTester:
    """Test identifier restoration accuracy."""
    
    def test_sample(self, sample):
        """Test a single sample's restoration accuracy."""
        restored = restore_identifiers(sample["tokens"], sample["labels"])
        expected = sample["expected_identifiers"]
        
        return {
            "restored": restored,
            "expected": expected,
            "exact_match": set(restored) == set(expected),
            "precision": len(set(restored) & set(expected)) / len(restored) if restored else 0,
            "recall": len(set(restored) & set(expected)) / len(expected) if expected else 0
        }
    
    def test_dataset(self, samples):
        """Test entire dataset restoration accuracy."""
        results = [self.test_sample(sample) for sample in samples]
        
        total_exact = sum(r["exact_match"] for r in results)
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_recall = sum(r["recall"] for r in results) / len(results)
        
        return {
            "exact_match_rate": total_exact / len(results),
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "f1_score": 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        }
```

## Technical Considerations

### Tokenization Alignment
- Ensure character-to-token span mapping is accurate
- Handle edge cases with special characters and punctuation
- Validate token reconstruction against original text

### Multi-Token Identifier Handling
- Consistent labeling for subword tokens of same identifier
- Proper reconstruction of identifiers split across multiple tokens
- Word boundary detection for extending predictions

### Memory Management
- Process files in batches to handle large datasets
- Use generators for memory-efficient iteration
- Stream processing for large token sequences

### Data Quality & Validation
- Restoration accuracy testing on all generated samples
- Identifier format validation against known patterns
- Token-text alignment verification

### Dependency Injection Architecture
- **Text Cropping Strategy**: Pluggable functions for cropping text around identifiers
- **Negative Sampling Strategy**: Configurable methods for generating negative samples
- **Extensible Design**: Easy to add new cropping and sampling strategies

## Expected Output Format

### JSONL Format Example (Recommended)
Each line contains one training sample:

```jsonl
{"tokens": ["Data", "were", "deposited", "under", "GS", "##E", "##12345", "."], "labels": [0, 0, 0, 0, 1, 1, 1, 0], "original_text": "Data were deposited under GSE12345.", "expected_identifiers": ["GSE12345"], "token_spans": [{"start": 0, "end": 4}, {"start": 5, "end": 9}, {"start": 10, "end": 19}, {"start": 20, "end": 25}, {"start": 26, "end": 28}, {"start": 28, "end": 29}, {"start": 29, "end": 34}, {"start": 34, "end": 35}], "identifier_spans": [{"identifier": "GSE12345", "char_start": 26, "char_end": 34, "token_start": 4, "token_end": 7}], "source_pmcid": "PMC1234567", "source_doi": "10.1234/example.2023.001", "metadata": {"token_count": 8, "identifier_count": 1, "sample_type": "positive", "text_type": "sentence"}}
{"tokens": ["This", "study", "analyzed", "protein", "interactions", "."], "labels": [0, 0, 0, 0, 0, 0], "original_text": "This study analyzed protein interactions.", "expected_identifiers": [], "token_spans": [{"start": 0, "end": 4}, {"start": 5, "end": 10}, {"start": 11, "end": 19}, {"start": 20, "end": 27}, {"start": 28, "end": 40}, {"start": 40, "end": 41}], "identifier_spans": [], "source_pmcid": "PMC1234568", "source_doi": "10.1234/example.2023.002", "metadata": {"token_count": 6, "identifier_count": 0, "sample_type": "negative", "text_type": "sentence"}}
```

### Restoration Test Results Format
```json
{
  "restoration_test_results": {
    "exact_match_rate": 0.94,
    "average_precision": 0.96,
    "average_recall": 0.95,
    "f1_score": 0.955,
    "failed_samples_count": 15,
    "test_timestamp": "2025-08-23T11:00:00Z"
  },
  "generation_summary": {
    "total_samples": 10000,
    "positive_samples": 7000,
    "negative_samples": 3000,
    "average_tokens_per_sample": 45.2,
    "unique_identifiers_found": 2341,
    "source_files_processed": 892,
    "generation_timestamp": "2025-08-23T10:30:00Z",
    "parameters": {
      "tokenizer_name": "allenai/scibert_scivocab_uncased",
      "num_samples": 10000,
      "positive_ratio": 0.7,
      "max_token_length": 128,
      "random_seed": 42
    }
  }
}
```

## Usage Example

```python
from bert_training_data_generator import BERTTrainingDataGenerator
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Initialize generator
generator = BERTTrainingDataGenerator(
    pmc_dir="data/pmc/txt",
    corpus_file="data/corpus/corpus_consolidated.json",
    pmc_ids_file="data/pmc/PMC-ids.csv",
    tokenizer=tokenizer
)

# Define custom text cropping and negative sampling strategies
def window_cropper(text, identifier_positions, window_size=50):
    """Crop text around identifiers with specified window size."""
    # Implementation for cropping text around identifiers
    pass

def random_negative_sampler(text, excluded_regions):
    """Generate random negative samples avoiding identifier regions."""
    # Implementation for generating negative samples
    pass

# Generate training data
result = generator.generate(
    num_samples=10000,
    positive_ratio=0.7,
    tokenizer_name="allenai/scibert_scivocab_uncased",
    text_unit="sentence",
    max_token_length=128,
    output_format="jsonl",
    output_path="training_data/bert_training_samples.jsonl",
    include_restoration_test=True,
    text_cropper=window_cropper,
    negative_sampler=random_negative_sampler,
    random_seed=42
)

# View restoration test results
print(f"Restoration accuracy: {result['restoration_test_results']['exact_match_rate']:.3f}")
print(f"F1 Score: {result['restoration_test_results']['f1_score']:.3f}")

# Test restoration on a sample
from restoration_tester import RestorationTester
tester = RestorationTester()

# Load a sample
sample = {
    "tokens": ["Data", "under", "GS", "##E", "##12345", "."],
    "labels": [0, 0, 1, 1, 1, 0],
    "expected_identifiers": ["GSE12345"]
}

# Test restoration
test_result = tester.test_sample(sample)
print(f"Restored: {test_result['restored']}")
print(f"Expected: {test_result['expected']}")
print(f"Exact match: {test_result['exact_match']}")
```

## Next Steps

1. **Implement core `BERTTrainingDataGenerator` class**
   - Token-level processing pipeline
   - Character-to-token span mapping
   - Integration with HuggingFace tokenizers

2. **Create token-level labeling utilities**
   - Character-level identifier detection
   - Token-level label assignment
   - Multi-token identifier handling

3. **Implement restoration algorithm**
   - Subword token reconstruction
   - Word boundary detection
   - Validation against ground truth

4. **Add restoration testing framework**
   - Sample-level accuracy testing
   - Dataset-level statistics
   - Quality assurance metrics

5. **Integrate with existing verification logic**
   - Reuse patterns from `verify_data_identifiers.py`
   - Adapt for token-level processing
   - Maintain identifier detection accuracy

6. **Implement data export and streaming**
   - JSONL format for efficient processing
   - Memory-efficient batch processing
   - Validation and quality reports

7. **Add comprehensive testing and validation**
   - Unit tests for restoration algorithm
   - Integration tests with real PMC data
   - Performance benchmarks and optimization

## Dependency Injection Design Patterns

### Text Cropping Strategy Interface
```python
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

# Example implementations
class WindowCropper:
    """Crop text with fixed window around identifiers."""
    
    def crop(self, text: str, identifier_positions: list[dict], window_size: int = 100) -> str:
        # Implementation for window-based cropping
        pass

class SentenceCropper:
    """Crop text to complete sentences containing identifiers."""
    
    def crop(self, text: str, identifier_positions: list[dict], sentence_context: int = 1) -> str:
        # Implementation for sentence-based cropping
        pass
```

### Negative Sampling Strategy Interface
```python
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

# Example implementations
class RandomNegativeSampler:
    """Generate random negative samples avoiding identifier regions."""
    
    def sample(self, text: str, excluded_regions: list[dict], sample_length: int = 100) -> str:
        # Implementation for random negative sampling
        pass

class ContextualNegativeSampler:
    """Generate negative samples from similar scientific contexts."""
    
    def sample(self, text: str, excluded_regions: list[dict], context_similarity: float = 0.8) -> str:
        # Implementation for contextual negative sampling
        pass
```