# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition solution for "Make Data Count - Finding Data References". The goal is to develop a system that automatically detects and classifies data citations in scientific papers. The task involves:

1. **PDF Parsing**: Extract text from scientific paper PDFs
2. **Data Citation Mining**: Detect references to research datasets using regex patterns and NLP
3. **Classification**: Categorize citations as Primary (original research data) or Secondary (reused/derived data)

## Development Environment

This project uses Python 3.12+ with minimal dependencies defined in `pyproject.toml`. The current setup is bare-bones and will need package additions as development progresses.

### Virtual Environment Setup
- **Package Manager**: Uses `uv` to create and manage the virtual environment in `.venv`
- **Jupyter**: Installed in the `.venv` virtual environment
- **Notebooks**: Use the `.venv` kernel for all notebook execution

## Current Architecture Strategy

The project has evolved to use a sophisticated BERT-based approach implemented in `bert_data_identifier_trainer.py`:

### BERT Data Identifier Training System
- **Model**: SciBERT (allenai/scibert_scivocab_uncased) for token classification
- **Training Data**: Dynamic generation from MDC corpus using PMC full-text papers
- **Pattern Recognition**: Built-in validation for common data identifiers (GEO, SRA, PDB, GenBank, DOI, etc.)
- **Memory Efficiency**: Sequential learning without storing large datasets on disk

### Key Components
1. **PMCTextReader**: Efficiently reads full-text PMC papers and maps DOI to content
2. **DataIdentifierPattern**: Validates and classifies data identifiers using regex patterns
3. **TrainingDataGenerator**: Creates positive/negative examples from real paper-dataset pairs
4. **BERTDataIdentifierTrainer**: Fine-tunes SciBERT for token-level data identifier detection

### Training Strategy
- **Positive Examples**: Sentences containing known data identifiers from MDC corpus
- **Negative Examples**: Regular text without data identifiers (30% ratio)
- **Token Classification**: BIO-style tagging where data identifiers get label 1, others get 0
- **Dynamic Generation**: Training examples generated on-the-fly from corpus data

### Current Performance
- **Achieved F1 Score**: 0.591 (notebooks/score-0.591.ipynb)
- **Target**: Approaching 0.6-0.7 range for competitive performance

## Key Constraints

- **Kaggle Submission**: Must be notebook format with 9-hour execution limit
- **Evaluation**: F1 score metric balancing precision and recall
- **Output Format**: CSV with columns: row_id, article_id, dataset_id, type
- **Critical**: Only include papers with detected citations (no false positives for papers without citations)

## Tool Usage Requirements

### CRITICAL: Use Serena MCP for All Code Analysis Tasks
**ALWAYS use Serena MCP tools for the following tasks - NEVER use alternative tools:**

- **String/Pattern Searching**: Use `mcp__serena__search_for_pattern` instead of Grep, Bash grep, or other search tools
- **File Finding**: Use `mcp__serena__find_file` instead of Glob or Bash find commands
- **Code Symbol Analysis**: Use `mcp__serena__find_symbol`, `mcp__serena__get_symbols_overview`
- **Code Reference Finding**: Use `mcp__serena__find_referencing_symbols`
- **Directory Listing**: Use `mcp__serena__list_dir` instead of LS or Bash ls
- **Code Reading**: Use symbolic tools first, then Read only when necessary
- **Code Editing**: Use `mcp__serena__replace_symbol_body`, `mcp__serena__insert_after_symbol`, etc.

**Why This Matters:**
- Serena provides semantic understanding of code structure
- More efficient token usage through targeted reads
- Better context awareness for code relationships
- Prevents reading entire files unnecessarily
- Essential for maintaining code quality and understanding

**Exception:** Only use non-Serena tools when:
- Working with non-code files (images, PDFs, data files)
- Performing system operations (git, package management)
- Serena tools explicitly cannot handle the task

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

      
      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
