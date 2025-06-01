# src/data_loader.py
import os
import json
import gzip
import logging
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

def load_nq_data(file_path, max_examples=None):
    """Loads NQ data from a .jsonl.gz file."""
    data = []
    if not os.path.exists(file_path):
        logger.error(f"Error: Data file not found at {file_path}")
        return data
    
    logger.info(f"Attempting to load NQ data from: {file_path}")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading NQ data", unit=" lines")):
                if max_examples and i >= max_examples:
                    logger.info(f"Reached max_examples limit of {max_examples}. Loaded {len(data)} examples.")
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {i+1} due to JSON decode error: {e}")
                    continue
        logger.info(f"Successfully loaded {len(data)} examples from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        return [] # Return empty list on failure
    return data

if __name__ == '__main__':
    # Example usage (for testing the module directly)
    # This requires config.py to be in the same directory or Python path
    # and the data file to be correctly pathed.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy config for testing if run directly
    class DummyConfig:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        DATA_DIR = os.path.join(PROJECT_ROOT, "data") # Adjust if your structure is different
        NQ_TRAIN_FILE_NAME = "v1.0-simplified-nq-train.jsonl.gz" # Ensure this file exists for testing
        NQ_TRAIN_PATH = os.path.join(DATA_DIR, NQ_TRAIN_FILE_NAME)

    cfg = DummyConfig()
    
    # Create dummy data directory and file for testing
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    dummy_data_content = [
        {"example_id": 1, "question_text": "Q1", "document_text": "Doc1 text", "long_answer_candidates": [], "annotations": []},
        {"example_id": 2, "question_text": "Q2", "document_text": "Doc2 text", "long_answer_candidates": [], "annotations": []}
    ]
    if not os.path.exists(cfg.NQ_TRAIN_PATH):
        with gzip.open(cfg.NQ_TRAIN_PATH, 'wt', encoding='utf-8') as f_gz:
            for item in dummy_data_content:
                f_gz.write(json.dumps(item) + '\n')
        logger.info(f"Created dummy data file at {cfg.NQ_TRAIN_PATH} for testing.")

    loaded_data = load_nq_data(cfg.NQ_TRAIN_PATH, max_examples=5)
    if loaded_data:
        logger.info(f"Test load successful. First example: {loaded_data[0]}")
    else:
        logger.error("Test load failed.")
