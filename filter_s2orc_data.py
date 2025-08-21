#!/usr/bin/env python3
"""
Filter S2ORC JSONL files to keep only papers that have DOIs present in corpus_consolidated.json
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse


def load_corpus_dois(corpus_file):
    """Load DOIs from corpus_consolidated.json"""
    print(f"Loading DOIs from {corpus_file}...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    # Extract DOIs from the publication field
    dois = set()
    for record in corpus_data:
        publication = record.get('publication', '')
        if publication:
            # Clean up the publication/DOI - handle various formats
            doi = publication.strip()
            # Remove common prefixes if present
            if doi.startswith('10.'):
                dois.add(doi)
            elif '10.' in doi:
                # Extract DOI part if it's embedded in text
                doi_start = doi.find('10.')
                extracted_doi = doi[doi_start:]
                # Take until first space or end
                extracted_doi = extracted_doi.split()[0]
                dois.add(extracted_doi)
    
    print(f"Loaded {len(dois)} unique DOIs from corpus")
    return dois


def normalize_doi(doi_str):
    """Normalize DOI string for comparison"""
    if not doi_str:
        return None
    
    doi = doi_str.strip()
    # Remove URL prefixes
    if doi.startswith('https://doi.org/'):
        doi = doi[15:]
    elif doi.startswith('http://dx.doi.org/'):
        doi = doi[18:]
    elif doi.startswith('doi:'):
        doi = doi[4:]
    
    # Ensure it starts with 10.
    if doi.startswith('10.'):
        return doi
    
    return None


def filter_jsonl_file(input_file, output_file, valid_dois):
    """Filter a single JSONL file based on valid DOIs"""
    print(f"Processing {input_file}...")
    
    matched_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing records"):
            try:
                record = json.loads(line.strip())
                total_count += 1
                
                # Extract DOI from the record
                doi = None
                if 'openaccessinfo' in record and record['openaccessinfo']:
                    if 'externalids' in record['openaccessinfo']:
                        doi = record['openaccessinfo']['externalids'].get('DOI')
                
                # Normalize and check if DOI is in our valid set
                normalized_doi = normalize_doi(doi)
                if normalized_doi and normalized_doi in valid_dois:
                    outfile.write(line)
                    matched_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
            except Exception as e:
                print(f"Error processing record: {e}")
                continue
    
    print(f"Filtered {input_file}: {matched_count}/{total_count} records matched")
    return matched_count, total_count


def main():
    parser = argparse.ArgumentParser(description='Filter S2ORC data based on corpus_consolidated.json DOIs')
    parser.add_argument('--corpus', default='data/corpus/corpus_consolidated.json',
                       help='Path to corpus_consolidated.json file')
    parser.add_argument('--input-dir', default='data/s2orc',
                       help='Directory containing S2ORC JSONL files')
    parser.add_argument('--output-dir', default='data/s2orc_filtered',
                       help='Output directory for filtered files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load valid DOIs from corpus
    valid_dois = load_corpus_dois(args.corpus)
    
    # Find all JSONL files in input directory
    input_dir = Path(args.input_dir)
    jsonl_files = list(input_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    total_matched = 0
    total_processed = 0
    
    # Process each JSONL file
    for jsonl_file in jsonl_files:
        print(total_matched, total_processed)
        output_file = output_dir / f"filtered_{jsonl_file.name}"
        matched, processed = filter_jsonl_file(jsonl_file, output_file, valid_dois)
        total_matched += matched
        total_processed += processed
    
    print(f"\n=== Summary ===")
    print(f"Total records processed: {total_processed}")
    print(f"Total records matched: {total_matched}")
    print(f"Match rate: {total_matched/total_processed*100:.2f}%" if total_processed > 0 else "0%")
    print(f"Filtered files saved to: {output_dir}")


if __name__ == '__main__':
    main()