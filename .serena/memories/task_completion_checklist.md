# Task Completion Checklist

## When a Development Task is Completed

### 1. Code Quality Checks
**Always run these commands before considering a task complete:**

```bash
# Run Ruff linting and auto-fix issues
ruff check .

# Run Ruff formatting
ruff format .
```

**Requirements:**
- All Ruff checks must pass without errors
- Code must be properly formatted
- No unused imports (automatically removed by Ruff)

### 2. Functionality Testing
**Since there are no formal unit tests, verify functionality by:**

```bash
# Test main functionality
python main.py

# Test specific modules
python bert_data_identifier_trainer.py
python src/paper_chunker.py

# For BERT training changes, run training validation
python -c "from bert_data_identifier_trainer import BERTDataIdentifierTrainer; trainer = BERTDataIdentifierTrainer(); print('Trainer initialized successfully')"
```

### 3. Model Performance (if applicable)
**For changes affecting model training or prediction:**
- Run relevant Jupyter notebook (notebooks/score-0.591.ipynb)
- Verify F1 score maintains or improves current performance (0.591)
- Check memory usage and training time within Kaggle constraints

### 4. Documentation Updates
**Update documentation if needed:**
- Update CLAUDE.md if architecture changes
- Update memory files if major changes to codebase structure
- Add comments for complex logic

### 5. Version Control
**Before pushing changes:**
```bash
git status
git add .
git commit -m "descriptive commit message"
# Only push when explicitly requested by user
```

## Quality Standards

### Code Requirements
- Type hints where applicable
- Proper error handling with logging
- Memory-efficient processing for large datasets
- GPU/CPU compatibility
- Progress bars for long-running operations

### Performance Requirements
- Training must complete within reasonable time (< 1 hour locally)
- Memory usage must be manageable for Kaggle environment
- No memory leaks in long-running processes

### Kaggle Constraints
- Code must be notebook-compatible
- Total execution time must be under 9 hours
- No internet access during execution
- All dependencies must be available in Kaggle environment

## Red Flags to Avoid
- ❌ Breaking changes without testing notebooks
- ❌ Introducing new dependencies without updating pyproject.toml
- ❌ Memory-intensive operations without cleanup
- ❌ Hardcoded paths that won't work in Kaggle environment
- ❌ Code that only works on specific hardware (GPU-only)
- ❌ Ignoring Ruff linting errors