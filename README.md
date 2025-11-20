# TRACER BM25 Scripts

This directory contains scripts for building CTI datasets and running a BM25-based baseline model for malware / actor attribution.

## Python files

- `bm25_model.py`  
  Core BM25 model implementation and helper functions (e.g., fitting on text fields, scoring candidate groups, returning ranked results).

- `bm25_cli.py`  
  Command-line interface for running the BM25 model on a dataset or a single query report (e.g., load model, score, and print top-k groups).

- `build_alias_map.py`  
  Builds a mapping from group aliases to canonical group IDs (e.g., using MITRE ATT&CK or custom alias JSON) so that different names resolve to the same actor.

- `build_cti_dataset.py`  
  Builds the main CTI dataset from raw CTI report directories (e.g., PDFs/CSVs) into a cleaned CSV used for training and evaluation.

- `dataset_utils.py`  
  Shared utilities for loading, splitting, and preprocessing datasets (e.g., train/val/test splits, text cleaning, column checks).

- `sanitize_csv_aliases.py`  
  Cleans and normalizes alias fields in CSV files (e.g., trimming whitespace, deduplicating aliases, enforcing consistent formats).


## BM25 Training Pipeline
![BM25 Training Pipeline](images/bm25pipeline.png)