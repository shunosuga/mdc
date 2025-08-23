# Codebase Structure

## Directory Structure
```
mdc/
├── .serena/                    # Serena MCP configuration
├── .vscode/                    # VS Code settings
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks
│   ├── paperscraper.ipynb     # PDF downloading and processing
│   ├── score-0.563.ipynb      # Earlier model version (F1: 0.563)
│   └── score-0.591.ipynb      # Current best model (F1: 0.591)
├── src/                        # Source code modules
│   └── paper_chunker.py       # BERT-based paper text chunking
├── tmp_scripts/                # Temporary utility scripts
├── bert_data_identifier_trainer.py  # Main BERT training system
├── main.py                     # Entry point
├── pyproject.toml              # Project configuration
└── README.md                   # Project documentation (Japanese)
```

## Key Files

### Main Implementation Files
- **`bert_data_identifier_trainer.py`**: Core BERT-based training system with classes:
  - `PMCTextReader`: Full-text PMC paper processing
  - `DataIdentifierPattern`: Regex-based data identifier validation
  - `TrainingDataGenerator`: Dynamic training data creation
  - `BERTDataIdentifierTrainer`: SciBERT fine-tuning for token classification

- **`src/paper_chunker.py`**: BERT-compatible text chunking
  - `PaperChunker`: Chunks papers into token-limited sections for BERT processing

- **`main.py`**: Project entry point

### Analysis Files
- **`analyze_pmc_corpus.py`**: PMC corpus analysis utility
- **`verify_data_identifiers.py`**: Data identifier verification script
- **`pmc_corpus_matched.txt`** / **`pmc_corpus_unmatched.txt`**: Corpus matching results

### Configuration
- **`pyproject.toml`**: Python project configuration with dependencies and Ruff linting setup
- **`uv.lock`**: UV package manager lock file
- **`.python-version`**: Python version specification
- **`.gitignore`**: Git ignore rules

## Module Dependencies
The main modules are interdependent:
- `bert_data_identifier_trainer.py` uses components from the entire system
- `paper_chunker.py` is used by the trainer for text preprocessing
- Notebooks import and use the main training classes for experimentation