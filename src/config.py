# src/config.py
import os

# --- Dataset and Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NQ_TRAIN_FILE_NAME = "v1.0-simplified-nq-train.jsonl.gz"
NQ_TRAIN_PATH = os.path.join(DATA_DIR, NQ_TRAIN_FILE_NAME)

# --- Data Processing Parameters ---
NUM_TEST_EXAMPLES = 300
KNOWLEDGE_BASE_SIZE = 20000 # Max NQ examples to load for KB and test set pool

# --- Qdrant Settings ---
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "nq_rag_lab_collection"

# --- Embedding Model ---
EMBED_MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIM = 384  # Dimension for e5-small-v2
EMBED_BATCH_SIZE = 512 # Tune based on VRAM

# --- LLM API Endpoint ---
LLM_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://192.168.0.114:1234/v1")
LLM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "not-needed-for-local-lm-studio")

# --- Model Context Windows ---
# Max context window for the generator LLM, matching LM Studio setting
MODEL_CTX_WINDOW = 8192
# Max context window for the judge LLM, can be smaller if judge prompts are shorter
JUDGE_MODEL_CTX_WINDOW = 4096 
# Safety margin for prompt construction (e.g., leave 300-500 tokens for the answer and system messages)
PROMPT_SAFETY_MARGIN = 500

# --- Generator LLM ---
GENERATOR_LLM_MODEL_NAME = "gemma-3-4b-it" # Ensure this matches the LM Studio model

# --- Judge LLM ---
JUDGE_LLM_MODEL_NAME = "gemma-3-27b-it" # Ensure this matches the LM Studio model

# --- Retrieval Settings ---
TOP_K_RETRIEVAL_FOR_GENERATION = 10
TOP_K_FOR_RECALL_EVALUATION = [1, 3, 5, 10, 20, 30]

# --- Node Parser Settings (for LlamaIndex) ---
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

# --- Indexing Settings ---
RECREATE_QDRANT_COLLECTION = False  # Set to True to force delete and recreate on each run
# QDRANT_INSERT_BATCH_SIZE is defined in rag_pipeline_logic.py (e.g., 8192)

# --- EDA Settings ---
EDA_SAMPLE_SIZE = 1000

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Random Seed ---
RANDOM_SEED = 42
