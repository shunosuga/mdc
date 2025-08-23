# Project Overview

## Purpose
This is a Kaggle competition solution for "Make Data Count - Finding Data References". The goal is to develop a system that automatically detects and classifies data citations in scientific papers.

## Main Tasks
1. **PDF Parsing**: Extract text from scientific paper PDFs
2. **Data Citation Mining**: Detect references to research datasets using regex patterns and NLP
3. **Classification**: Categorize citations as Primary (original research data) or Secondary (reused/derived data)

## Tech Stack
- **Python**: 3.13+ (specified in pyproject.toml)
- **Key Dependencies**:
  - `torch` >= 2.8.0 (PyTorch for deep learning)
  - `transformers` >= 4.55.2 (Hugging Face transformers)
  - `scikit-learn` >= 1.7.1 (Machine learning)
  - `pandas` >= 2.3.1 (Data manipulation)
  - `polars` >= 1.32.3 (High-performance data frames)
  - `paperscraper` >= 0.3.2 (PDF/paper processing)
  - `jupyter` >= 1.1.1 (Notebook development)

## Current Architecture
The project uses a BERT-based approach implemented in `bert_data_identifier_trainer.py`:

### Core Components
1. **PMCTextReader**: Efficiently reads full-text PMC papers and maps DOI to content
2. **DataIdentifierPattern**: Validates and classifies data identifiers using regex patterns
3. **TrainingDataGenerator**: Creates positive/negative examples from real paper-dataset pairs
4. **BERTDataIdentifierTrainer**: Fine-tunes ModernBERT for token-level data identifier detection
5. **PaperChunker**: Chunks papers into manageable sections for BERT processing

### Training Strategy
- **Model**: ModernBERT-base (answerdotai/ModernBERT-base) for token classification
- **Training Data**: Dynamic generation from MDC corpus using PMC full-text papers
- **Token Classification**: BIO-style tagging where data identifiers get label 1, others get 0
- **Current Performance**: F1 Score of 0.591 (notebooks/score-0.591.ipynb)

## Constraints
- **Kaggle Submission**: Must be notebook format with 9-hour execution limit
- **Evaluation**: F1 score metric balancing precision and recall
- **Output Format**: CSV with columns: row_id, article_id, dataset_id, type
- **Critical**: Only include papers with detected citations (no false positives for papers without citations)