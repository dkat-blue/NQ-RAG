# src/llm_setup.py
import logging
import qdrant_client
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
import torch # Import torch to check for CUDA availability

logger = logging.getLogger(__name__)

def setup_qdrant_client(qdrant_url):
    """Initializes and returns a Qdrant client."""
    try:
        client = qdrant_client.QdrantClient(url=qdrant_url)
        client.get_collections()
        logger.info(f"Successfully connected to Qdrant at {qdrant_url} and listed collections.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant at {qdrant_url}: {e}")
        logger.error("Please ensure Qdrant server is running. E.g., Docker: docker run -p 6333:6333 qdrant/qdrant")
        return None

def setup_embedding_model(model_name, device_preference="cuda", batch_size=256):
    """Initializes and returns a HuggingFace embedding model, attempting to use CUDA if available."""
    try:
        actual_device = "cpu" # Default to CPU
        if device_preference == "cuda" and torch.cuda.is_available():
            actual_device = "cuda"

        logger.info(f"Attempting to set up embedding model '{model_name}' on device: '{actual_device}' with embed_batch_size: {batch_size}")

        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=actual_device,
            embed_batch_size=batch_size
        )
        # Test with a dummy sentence to confirm it loads on the device
        _ = embed_model.get_text_embedding("Test embedding")
        logger.info(f"Successfully loaded embedding model: {model_name} on device '{actual_device}'")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name} (tried device: {actual_device}): {e}")
        return None

def setup_llm(api_base, api_key, model_name, llm_type="Generator", temperature=0.0): # Added temperature parameter, default 0.0
    """Initializes and returns an LLM (OpenAI-compatible)."""
    if not api_base:
        logger.error(f"API base URL for {llm_type} LLM is not set. Cannot initialize.")
        return None
    try:
        # Ensure model_name from config is used, not a hardcoded one.
        llm = LlamaOpenAI(
            base_url=api_base,
            api_key=api_key,
            model=model_name, # Uses the model_name passed from config
            temperature=temperature, # Use the temperature parameter
            # max_tokens can be set if needed, but often managed by prompt length
        )
        # Attempt a simple call to check connectivity (e.g. get_model_info, or a very short complete).
        logger.info(f"Successfully initialized {llm_type} LLM: {model_name} via {api_base} with temperature: {temperature}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize {llm_type} LLM {model_name} using API base {api_base}: {e}")
        logger.error("Ensure your LLM server (e.g., LM Studio) is running and the API base URL/key are correct, and model is loaded.")
        return None


def configure_llama_index_settings(llm_instance, embed_model_instance, chunk_size, chunk_overlap):
    """Configures global LlamaIndex Settings."""
    if llm_instance:
        Settings.llm = llm_instance
        # Set the tokenizer for LlamaIndex based on the primary LLM
        # OpenAI class in LlamaIndex has a tokenizer attribute
        if hasattr(llm_instance, 'tokenizer') and llm_instance.tokenizer:
            Settings.tokenizer = llm_instance.tokenizer
        else:
            # Fallback or explicit tiktoken setup if needed, though LlamaOpenAI should provide it.
            # from llama_index.core.utils import get_tokenizer
            # Settings.tokenizer = get_tokenizer() # This is a function that returns the default tiktoken tokenizer
            logger.warning("LLM instance does not have a 'tokenizer' attribute. Global Settings.tokenizer might not be optimally set. Using LlamaIndex default.")
            # LlamaIndex defaults Settings.tokenizer to tiktoken.get_encoding("cl100k_base")
            # which is generally fine for OpenAI compatible models.
    else:
        logger.warning("LLM instance not provided for LlamaIndex Settings. LLM and its tokenizer not set globally.")

    if embed_model_instance:
        Settings.embed_model = embed_model_instance
    else:
        logger.warning("Embedding model instance not provided for LlamaIndex Settings.")

    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # These are also set for SentenceSplitter, but setting them globally can be good for consistency if used elsewhere.
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    logger.info("LlamaIndex Settings configured (or attempted).")

if __name__ == '__main__':
    import os
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Assuming config.py is in the same directory or PYTHONPATH
    from config import MODEL_CTX_WINDOW, GENERATOR_LLM_MODEL_NAME, JUDGE_LLM_MODEL_NAME, EMBED_MODEL_NAME, EMBED_BATCH_SIZE, LLM_API_BASE, LLM_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, QDRANT_URL

    q_client = setup_qdrant_client(QDRANT_URL)

    emb_model = setup_embedding_model(EMBED_MODEL_NAME, device_preference="cuda", batch_size=EMBED_BATCH_SIZE)
    if not emb_model:
        emb_model = setup_embedding_model(EMBED_MODEL_NAME, device_preference="cpu", batch_size=EMBED_BATCH_SIZE)

    # Setup generator with temperature 0.0
    gen_llm = setup_llm(LLM_API_BASE, LLM_API_KEY, GENERATOR_LLM_MODEL_NAME, "Generator", temperature=0.0)
    # Setup judge with temperature 0.0
    judge_llm = setup_llm(LLM_API_BASE, LLM_API_KEY, JUDGE_LLM_MODEL_NAME, "Judge", temperature=0.0)


    if gen_llm and emb_model:
        # Pass the generator LLM instance for global settings
        configure_llama_index_settings(gen_llm, emb_model, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info(f"Global tokenizer set to: {Settings.tokenizer}")
    else:
        logger.error("Could not configure LlamaIndex settings due to missing LLM or embed model.")

    if gen_llm:
        try:
            logger.info(f"Testing {GENERATOR_LLM_MODEL_NAME} with a simple prompt...")
            # response = gen_llm.complete("Hello!")
            # logger.info(f"Test response from {GENERATOR_LLM_MODEL_NAME}: {response.text}")
            # LiteLLM / OpenAI API might not have a simple ping, actual completion is the test.
        except Exception as e:
            logger.error(f"Error during test completion with {GENERATOR_LLM_MODEL_NAME}: {e}")
