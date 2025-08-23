#!/usr/bin/env python3
"""
Minimal test for BERT Training Data Generator using mock data
"""

import sys
import os
import tempfile
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bert_training.bert_training_data_generator import BERTTrainingDataGenerator

def create_mock_data():
    """Create minimal mock data for testing"""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create mock corpus file with just a few records
    corpus_data = [
        {
            "publication": "10.1234/example1",
            "repository": "NCBI GEO",
            "datasets": ["GSE12345", "GSE67890"]
        },
        {
            "publication": "10.1234/example2", 
            "repository": "SRA",
            "datasets": ["SRR123456"]
        }
    ]
    
    corpus_file = temp_dir / "corpus.json"
    with open(corpus_file, 'w') as f:
        json.dump(corpus_data, f)
    
    # Create mock PMC text files
    pmc_dir = temp_dir / "pmc" / "txt"
    pmc_dir.mkdir(parents=True)
    
    # Create subdirectories with text files
    pmc1_dir = pmc_dir / "PMC12345"
    pmc1_dir.mkdir()
    with open(pmc1_dir / "PMC12345.txt", 'w') as f:
        f.write("This study presents gene expression data deposited as GSE12345 in the GEO database. "
                "Additional data includes GSE67890 for comparison analysis. "
                "The methodology follows standard protocols for RNA sequencing.")
    
    pmc2_dir = pmc_dir / "PMC67890"
    pmc2_dir.mkdir()
    with open(pmc2_dir / "PMC67890.txt", 'w') as f:
        f.write("Sequence reads were deposited in SRR123456 repository. "
                "The raw data processing pipeline was implemented using standard tools. "
                "Quality control metrics show high confidence in the results.")
    
    # Create mock DOI-PMC mapping (minimal CSV)
    doi_mapping_file = temp_dir / "doi_mapping.csv"
    with open(doi_mapping_file, 'w') as f:
        f.write("DOI,PMCID\n")
        f.write("10.1234/example1,PMC12345\n")
        f.write("10.1234/example2,PMC67890\n")
    
    return temp_dir, corpus_file, pmc_dir.parent, doi_mapping_file

def test_minimal_generation():
    """Test generator with minimal mock data"""
    print("=== Testing BERT Training Data Generator ===")
    
    # Create mock data
    temp_dir, corpus_file, pmc_base, doi_mapping = create_mock_data()
    
    try:
        print(f"Using temporary directory: {temp_dir}")
        print(f"Corpus file: {corpus_file}")
        print(f"PMC base directory: {pmc_base}")
        print(f"DOI mapping: {doi_mapping}")
        
        # Initialize generator
        generator = BERTTrainingDataGenerator(
            pmc_dir=str(pmc_base),  # The generator will add /txt internally
            corpus_file=str(corpus_file),
            pmc_ids_file=str(doi_mapping)
        )
        
        print("\n=== Generator initialized successfully ===")
        
        # Generate small number of samples
        output_file = temp_dir / "test_samples.jsonl"
        
        print("\n=== Starting sample generation ===")
        result = generator.generate(
            num_samples=5,  # Very small number for testing
            positive_ratio=0.8,
            output_path=str(output_file),
            random_seed=42
        )
        
        print(f"\n=== Generation completed ===")
        gen_summary = result.get('generation_summary', {})
        print(f"Generated samples: {gen_summary.get('total_samples', 0)}")
        print(f"Positive samples: {gen_summary.get('positive_samples', 0)}")
        print(f"Negative samples: {gen_summary.get('negative_samples', 0)}")
        print(f"Average tokens per sample: {gen_summary.get('average_tokens_per_sample', 0):.1f}")
        
        # Check if output file was created and read a sample
        if output_file.exists():
            with open(output_file, 'r') as f:
                first_line = f.readline()
                if first_line.strip():
                    sample = json.loads(first_line)
                    print(f"\n=== Sample structure ===")
                    print(f"Sample keys: {list(sample.keys())}")
                    print(f"Number of tokens: {len(sample.get('tokens', []))}")
                    print(f"Number of labels: {len(sample.get('labels', []))}")
                    print(f"Original text preview: {sample.get('original_text', '')[:100]}...")
        
        print("\n[SUCCESS] Minimal generator test completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Generator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except:
            pass

if __name__ == "__main__":
    test_minimal_generation()