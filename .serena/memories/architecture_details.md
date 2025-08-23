# Architecture Details

## BERT-based Data Identifier System

### Model Architecture
- **Base Model**: ModernBERT-base (answerdotai/ModernBERT-base)
- **Task**: Token classification for data identifier detection
- **Labels**: Binary classification (0 = non-identifier, 1 = data identifier)
- **Approach**: BIO-style tagging adapted for token-level detection

### Core Components

#### 1. PMCTextReader
- **Purpose**: Reads and processes PMC full-text papers
- **Input**: DOI or paper ID
- **Output**: Structured text content
- **Key Features**:
  - Maps DOI to paper content
  - Handles various paper formats
  - Memory-efficient streaming

#### 2. DataIdentifierPattern
- **Purpose**: Validates and classifies data identifiers using regex
- **Supported Patterns**:
  - DOI: `10.xxxx/yyyy` format
  - Database IDs: GEO, SRA, PDB, GenBank, CHEMBL
  - URLs to data repositories
  - Dataset names and accession numbers
- **Validation**: Built-in pattern matching for common identifiers

#### 3. TrainingDataGenerator
- **Purpose**: Creates training examples from MDC corpus
- **Strategy**:
  - Positive examples: Sentences with known data identifiers
  - Negative examples: Regular text without identifiers (30% ratio)
  - Dynamic generation to avoid large disk storage
- **Features**:
  - Real paper-dataset pairs from corpus
  - Contextual sentence extraction
  - Balanced positive/negative sampling

#### 4. BERTDataIdentifierTrainer
- **Training Process**:
  - Fine-tunes ModernBERT for token classification
  - Sequential learning without storing datasets
  - GPU acceleration with CPU fallback
  - Progress tracking with tqdm
- **Key Methods**:
  - `train()`: Main training loop
  - `predict()`: Inference on new text
  - `save_model()`: Model serialization

#### 5. PaperChunker
- **Purpose**: Splits papers into BERT-compatible chunks
- **Strategy**:
  - Respects BERT token limits (default: 512 tokens)
  - Preserves sentence boundaries
  - Maintains section context
- **Chunking Modes**:
  - Sentence-based: Splits on sentence boundaries
  - Token-based: Splits on token count
  - Section-aware: Preserves document structure

### Training Data Pipeline

```
PMC Corpus → PMCTextReader → DataIdentifierPattern → TrainingDataGenerator → BERT Training
```

1. **Corpus Processing**: PMC papers loaded and processed
2. **Pattern Matching**: Known data identifiers extracted using regex
3. **Example Generation**: Positive/negative examples created dynamically
4. **Token Classification**: ModernBERT trained to identify data identifiers at token level

### Performance Metrics
- **Current F1 Score**: 0.591 (notebooks/score-0.591.ipynb)
- **Target Range**: 0.6-0.7 for competitive performance
- **Evaluation**: Standard F1 score balancing precision and recall

### Memory and Performance Optimization
- **Sequential Learning**: Avoids loading entire corpus into memory
- **Batch Processing**: Efficient GPU utilization
- **Generator Patterns**: Memory-efficient data streaming
- **Device Management**: Automatic GPU/CPU selection

### Integration Points
- **Notebook Compatibility**: Designed for Kaggle notebook execution
- **Time Constraints**: Optimized for 9-hour execution limit
- **External Data**: Compatible with pre-trained models and public datasets
- **Output Format**: Generates competition-required CSV format