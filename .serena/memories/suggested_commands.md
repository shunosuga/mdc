# Suggested Commands

## Development Environment Setup
```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install/update dependencies
uv sync
```

## Code Quality Commands

### Linting and Formatting
```bash
# Run Ruff linting (automatically fixes fixable issues)
ruff check .

# Run Ruff formatting
ruff format .

# Check specific file
ruff check bert_data_identifier_trainer.py
```

## Running the Application

### Main Entry Points
```bash
# Run main application
python main.py

# Run BERT data identifier trainer
python bert_data_identifier_trainer.py

# Run paper chunker
python src/paper_chunker.py

# Run analysis scripts
python analyze_pmc_corpus.py
python verify_data_identifiers.py
```

### Jupyter Notebooks
```bash
# Start Jupyter server (ensure .venv kernel is selected)
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

## Testing
**Note**: No formal test suite is currently implemented. Testing is done through:
- Manual execution of main scripts
- Jupyter notebook validation
- F1 score evaluation in notebooks

## Utility Commands (Windows)

### File Operations
```cmd
# List directory contents
dir
dir /s  # recursive

# Find files
where /r . *.py
dir /s /b *.py

# Search in files (using findstr - Windows equivalent of grep)
findstr /s /i "pattern" *.py
```

### Git Operations
```bash
git status
git add .
git commit -m "message"
git push origin pmc-biorxiv  # current branch
```

## Project-Specific Workflows

### Model Training
```bash
# Train new BERT model
python bert_data_identifier_trainer.py

# Evaluate model performance
# Use notebooks/score-0.591.ipynb for current evaluation
```

### Data Processing
```bash
# Process PMC corpus
python analyze_pmc_corpus.py

# Verify data identifiers
python verify_data_identifiers.py
```

## Performance Monitoring
```bash
# Monitor GPU usage (if available)
nvidia-smi

# Monitor system resources
# Use Task Manager on Windows or Resource Monitor
```