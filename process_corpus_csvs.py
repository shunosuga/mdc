#!/usr/bin/env python3

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def process_corpus_csvs():
    """Process all CSV files in data/corpus and create consolidated JSON."""
    
    # Find all CSV files
    corpus_dir = Path("data/corpus")
    csv_files = list(corpus_dir.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Dictionary to group datasets by publication
    publications_data = defaultdict(lambda: {
        "publication": "",
        "repository": "",
        "datasets": []
    })
    
    total_rows = 0
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            total_rows += len(df)
            
            # Extract required columns
            for _, row in df.iterrows():
                publication = row.get('publication', '')
                repository = row.get('repository', '')
                dataset = row.get('dataset', '')
                
                # Skip rows with missing required data
                if not publication or pd.isna(publication):
                    continue
                    
                # Group by publication
                if publication not in publications_data:
                    publications_data[publication] = {
                        "publication": publication,
                        "repository": repository if not pd.isna(repository) else "",
                        "datasets": []
                    }
                
                # Add dataset if it exists and is not already in the list
                if dataset and not pd.isna(dataset) and dataset not in publications_data[publication]["datasets"]:
                    publications_data[publication]["datasets"].append(dataset)
                    
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    # Convert to list format
    result = list(publications_data.values())
    
    # Sort by publication for consistent output
    result.sort(key=lambda x: x["publication"])
    
    print(f"Processed {total_rows} total rows")
    print(f"Found {len(result)} unique publications")
    
    # Save to JSON
    output_file = Path("data/corpus_consolidated.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Saved consolidated data to {output_file}")
    print(f"Sample entry: {result[0] if result else 'No data'}")

if __name__ == "__main__":
    process_corpus_csvs()