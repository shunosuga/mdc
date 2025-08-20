# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition solution for "Make Data Count - Finding Data References". The goal is to develop a system that automatically detects and classifies data citations in scientific papers. The task involves:

1. **PDF Parsing**: Extract text from scientific paper PDFs
2. **Data Citation Mining**: Detect references to research datasets using regex patterns and NLP
3. **Classification**: Categorize citations as Primary (original research data) or Secondary (reused/derived data)

## Development Environment

This project uses Python 3.12+ with minimal dependencies defined in `pyproject.toml`. The current setup is bare-bones and will need package additions as development progresses.

## Project Architecture Strategy

Based on the README analysis, the solution follows a three-phase approach:

### Phase 1: Basic Regex Engine
- Implement DOI pattern detection (`10.xxxx/yyyy` format)
- Database name recognition (CHEMBL, PDB, GenBank, etc.)
- URL pattern matching for data repositories
- Target F1 score: 0.4-0.5

### Phase 2: Advanced Pattern Matching
- Multi-pattern integration with confidence scoring
- False positive reduction techniques
- Enhanced recall through pattern expansion

### Phase 3: LLM Integration (if needed)
- Few-shot learning for edge cases
- Context-aware classification
- Ensemble methods combining regex + LLM approaches

## Key Constraints

- **Kaggle Submission**: Must be notebook format with 9-hour execution limit
- **Evaluation**: F1 score metric balancing precision and recall
- **Output Format**: CSV with columns: row_id, article_id, dataset_id, type
- **Critical**: Only include papers with detected citations (no false positives for papers without citations)

## Expected File Structure

As development progresses, expect:
- `src/`: Python modules for PDF parsing, citation detection, and classification
- `notebooks/`: Jupyter notebooks for experimentation and final Kaggle submission
- Additional dependencies in `pyproject.toml` for PDF processing (PyMuPDF, marker) and ML libraries

## Performance Targets

- **Immediate**: Achieve 0.4-0.5 F1 with regex-only approach
- **Short-term**: 0.6-0.7 F1 with advanced techniques
- **Competition Goal**: Top-tier performance for prize money ($40K-$10K range)