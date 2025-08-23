# Code Style and Conventions

## Language and Version
- **Python**: 3.13+ (specified in pyproject.toml requires-python = ">=3.13")
- **Package Manager**: UV (uv.lock present)
- **Virtual Environment**: `.venv` directory

## Code Style Configuration

### Ruff Configuration (pyproject.toml)
The project uses Ruff for linting and formatting with the following rules:

```toml
[tool.ruff]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes (includes F401 for unused imports)
    "I",    # isort (import sorting)
]

# Fix unused imports automatically
fix = true
show-fixes = true

# Automatically remove unused imports
fixable = [
    "F401",  # unused-import
    "I001",  # unsorted-imports
    "I002",  # missing-required-import
]
```

### Import Sorting
- Uses isort integration through Ruff
- No force-single-line imports (allows grouped imports)
- Known first-party: "your_project_name" (needs to be updated to "mdc")

## Coding Patterns Observed

### Class Structure
Classes follow a consistent pattern:
- Clear docstrings for classes and methods
- Type hints where applicable
- Device management for PyTorch (CPU/GPU detection)
- Logging integration using Python's logging module

### Method Naming
- Snake case for methods and variables
- Clear, descriptive names (e.g., `count_tokens`, `generate_training_data`)
- Private methods prefixed with underscore where appropriate

### Error Handling
- Robust error handling in file operations
- Graceful fallbacks (e.g., CPU fallback when GPU unavailable)
- Logging of errors and warnings

### Data Processing
- Generator patterns for memory efficiency
- Batch processing for large datasets
- Progress bars using tqdm for long operations

## File Organization
- Single responsibility principle: each file has a clear purpose
- Main classes in dedicated modules
- Utility functions grouped logically
- Configuration separated from implementation

## Documentation Style
- Japanese documentation in README.md (competition-focused)
- English code comments and docstrings
- Comprehensive documentation of model architecture and training strategy