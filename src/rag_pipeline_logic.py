# src/rag_pipeline_logic.py
import logging
import time

from llama_index.core import VectorStoreIndex, StorageContext, Settings, QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.prompts import PromptTemplate

from tqdm.notebook import tqdm

from src.config import MODEL_CTX_WINDOW, PROMPT_SAFETY_MARGIN

logger = logging.getLogger(__name__)

QDRANT_INSERT_BATCH_SIZE = 8192

def get_token_count(text: str) -> int:
    """Counts tokens using the globally set LlamaIndex tokenizer."""
    tokenizer = Settings.tokenizer
    if not tokenizer:
        logger.warning("Settings.tokenizer is None. Falling back to len(text.split()).")
        return len(text.split())
    try:
        if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
            # It's an object with an .encode() method (e.g., tiktoken.Encoding instance)
            return len(tokenizer.encode(text))
        elif callable(tokenizer):
            # It's a callable function (e.g., the default functools.partial from LlamaIndex's get_tokenizer)
            return len(tokenizer(text))
        else:
            logger.warning(f"Settings.tokenizer (type: {type(tokenizer)}) is not a recognized tokenizer. Falling back to len(text.split()).")
            return len(text.split())
    except Exception as e:
        logger.error(f"Error using Settings.tokenizer (type: {type(tokenizer)}): {e}. Falling back to len(text.split()).", exc_info=True)
        return len(text.split())


def index_knowledge_base(qdrant_client_instance, collection_name, embedding_dim,
                         kb_docs_llama,
                         recreate_collection=False):
    vector_store = None
    vector_index = None

    if not qdrant_client_instance:
        logger.error("Qdrant client instance is not available. Cannot proceed.")
        return None, None
    if not Settings.embed_model:
        logger.error("Global LlamaIndex Settings.embed_model is not configured. Cannot proceed.")
        return None, None
    if not Settings.node_parser:
        logger.error("Global LlamaIndex Settings.node_parser is not configured. Cannot proceed with new indexing if needed.")

    collection_exists = False
    try:
        qdrant_client_instance.get_collection(collection_name=collection_name)
        collection_exists = True
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
    except Exception as e:
        if "not found" in str(e).lower() or \
           (hasattr(e, 'status_code') and e.status_code == 404) or \
           (hasattr(e, 'code') and hasattr(e.code(), 'value') and e.code().value[0] == 5): # type: ignore
            logger.info(f"Qdrant collection '{collection_name}' does not exist.")
            collection_exists = False
        else:
            logger.error(f"Error checking for Qdrant collection '{collection_name}': {e}", exc_info=True)
            return None, None

    if recreate_collection and collection_exists:
        logger.info(f"recreate_collection is True. Deleting existing Qdrant collection '{collection_name}'.")
        try:
            qdrant_client_instance.delete_collection(collection_name=collection_name)
            time.sleep(1)
            collection_exists = False
        except Exception as e:
            logger.error(f"Failed to delete Qdrant collection '{collection_name}': {e}", exc_info=True)
            return None, None

    if not collection_exists:
        logger.info(f"Creating new Qdrant collection '{collection_name}' and indexing documents.")
        if not kb_docs_llama:
            logger.error("kb_docs_llama is empty or None. Cannot create and index a new collection without documents.")
            return None, None
        if not Settings.node_parser: # Strict check before new indexing
            logger.error("Global LlamaIndex Settings.node_parser is not configured. Cannot proceed with new indexing.")
            return None, None

        try:
            qdrant_client_instance.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE) # type: ignore
            )
            logger.info(f"Successfully created Qdrant collection: {collection_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                 logger.warning(f"Collection '{collection_name}' reported as already existing during creation attempt. Proceeding by loading.")
                 collection_exists = True
            else:
                logger.error(f"Failed to create Qdrant collection '{collection_name}': {e}", exc_info=True)
                return None, None

        if not collection_exists:
            vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            logger.info(f"Indexing {len(kb_docs_llama)} documents into Qdrant collection '{collection_name}' using insert_batch_size={QDRANT_INSERT_BATCH_SIZE}.")
            vector_index = VectorStoreIndex.from_documents(
                kb_docs_llama,
                storage_context=storage_context,
                show_progress=True,
                insert_batch_size=QDRANT_INSERT_BATCH_SIZE
            )
            logger.info(f"New Knowledge Base indexing complete for collection '{collection_name}'.")

    if collection_exists:
        logger.info(f"Attempting to load existing VectorStoreIndex from Qdrant collection '{collection_name}'.")
        try:
            vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=collection_name,
            )
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store
            )
            if vector_index and vector_store.client:
                 collection_info = vector_store.client.get_collection(collection_name=collection_name)
                 num_vectors = collection_info.vectors_count if collection_info and hasattr(collection_info, 'vectors_count') else 0
                 if num_vectors is not None and num_vectors > 0:
                    logger.info(f"Successfully loaded existing index with approximately {num_vectors} vectors from collection '{collection_name}'.")
                 else:
                    logger.warning(f"Loaded index from collection '{collection_name}', but it appears to be empty (vectors_count: {num_vectors}). You might need to recreate it by setting RECREATE_QDRANT_COLLECTION = True in config.")
            else:
                 logger.warning(f"Loaded index from collection '{collection_name}', but it appears to be empty or invalid. You might need to recreate it.")
        except Exception as e:
            logger.error(f"Failed to load existing index from Qdrant collection '{collection_name}': {e}", exc_info=True)
            logger.error("Consider setting RECREATE_QDRANT_COLLECTION to True in config.py if the collection is corrupted or incompatible.")
            return None, None

    if not vector_index:
        logger.error(f"Failed to obtain a valid vector_index for collection '{collection_name}'.")
        return None, None

    return vector_store, vector_index


def generate_no_rag_answers(generator_llm, test_set, progress_desc="No-RAG Generation"):
    no_rag_answers = []
    if not generator_llm:
        logger.warning("Generator LLM not initialized. Skipping no-RAG generation.")
        return ["ERROR_LLM_NOT_INITIALIZED"] * len(test_set) if test_set else []
    if not test_set:
        logger.info("Test set is empty. No no-RAG answers to generate.")
        return []

    no_rag_prompt_tmpl_str = "Question: {query_str}\nAnswer:"
    no_rag_prompt_tmpl = PromptTemplate(no_rag_prompt_tmpl_str)

    logger.info("Generating answers without RAG...")
    for item in tqdm(test_set, desc=progress_desc, unit=" questions"):
        query_str = item['question_text']
        fmt_prompt = no_rag_prompt_tmpl.format(query_str=query_str)
        try:
            response = generator_llm.complete(fmt_prompt)
            no_rag_answers.append(str(response).strip())
        except Exception as e:
            logger.error(f"Error during no-RAG generation for query '{query_str}': {e}")
            no_rag_answers.append("ERROR_GENERATING_ANSWER")
    return no_rag_answers

def generate_rag_answers(vector_index, generator_llm, test_set, top_k_retrieval, progress_desc="RAG Generation"):
    rag_answers = []
    retrieved_contexts_for_rag = []

    if not vector_index or not generator_llm:
        logger.warning("Vector index or Generator LLM not initialized. Skipping RAG generation.")
        error_msg = "ERROR_PIPELINE_NOT_INITIALIZED"
        if test_set:
            return ["ERROR_GENERATING_ANSWER"] * len(test_set), [error_msg] * len(test_set)
        return [],[]
    if not test_set:
        logger.info("Test set is empty. No RAG answers to generate.")
        return [], []

    logger.info(f"Setting up RAG retriever with top_k_retrieval = {top_k_retrieval}")
    retriever_for_generation = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=top_k_retrieval,
    )

    base_query_wrapper_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query concisely. "
        "If the context does not provide an answer, state that the information is not available in the provided context.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    query_wrapper_prompt_tmpl = PromptTemplate(base_query_wrapper_prompt_str)

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=query_wrapper_prompt_tmpl,
        llm=generator_llm
    )

    logger.info("Generating answers with RAG...")
    for item in tqdm(test_set, desc=progress_desc, unit=" questions"):
        query_str = item['question_text']
        try:
            retrieved_nodes = retriever_for_generation.retrieve(query_str)
            
            prompt_fixed_parts_str = base_query_wrapper_prompt_str.format(context_str="", query_str=query_str)
            fixed_tokens = get_token_count(prompt_fixed_parts_str)
            context_budget_tokens = MODEL_CTX_WINDOW - fixed_tokens - PROMPT_SAFETY_MARGIN
            
            budgeted_nodes = []
            current_context_tokens_for_nodes = 0
            final_context_str_for_eval = ""

            for node in retrieved_nodes:
                node_text = node.get_content()
                node_tokens = get_token_count(node_text) + get_token_count("\n\n") 
                
                if current_context_tokens_for_nodes + node_tokens <= context_budget_tokens:
                    budgeted_nodes.append(node)
                    current_context_tokens_for_nodes += node_tokens
                else:
                    break
            
            if not budgeted_nodes and retrieved_nodes:
                 logger.warning(f"No nodes fit RAG context budget for query '{query_str[:50]}...'. First node: {get_token_count(retrieved_nodes[0].get_content())} tokens. Context budget: {context_budget_tokens}")
                 final_context_str_for_eval = "CONTEXT_EMPTY_DUE_TO_BUDGET"
            elif not retrieved_nodes:
                 final_context_str_for_eval = "NO_CONTEXT_RETRIEVED"
            else:
                 final_context_str_for_eval = "\\n\\n".join([n.get_content() for n in budgeted_nodes])

            retrieved_contexts_for_rag.append(final_context_str_for_eval)

            response = response_synthesizer.synthesize(query=QueryBundle(query_str), nodes=budgeted_nodes)
            rag_answers.append(str(response).strip())

        except Exception as e:
            logger.error(f"Error during RAG generation for query '{query_str}': {e}", exc_info=True)
            rag_answers.append("ERROR_GENERATING_ANSWER")
            retrieved_contexts_for_rag.append("ERROR_RETRIEVING_CONTEXT")

    return rag_answers, retrieved_contexts_for_rag


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("--- Starting Test for RAG Pipeline Logic (Token Counting) ---")

    try:
        from src.llm_setup import setup_llm, configure_llama_index_settings, setup_embedding_model
        from src.config import (
            LLM_API_BASE, LLM_API_KEY, GENERATOR_LLM_MODEL_NAME,
            EMBED_MODEL_NAME, EMBED_BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP
        )
    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        logger.info("Skipping __main__ test block.")
        exit()

    test_llm = setup_llm(LLM_API_BASE, LLM_API_KEY, GENERATOR_LLM_MODEL_NAME, "TokenizerSetupLLM", temperature=0.0)
    if not test_llm:
        logger.error("Failed to initialize test LLM for tokenizer setup. Aborting test.")
        exit()

    dummy_embed_model = setup_embedding_model(EMBED_MODEL_NAME, batch_size=EMBED_BATCH_SIZE, device_preference="cpu")
    if not dummy_embed_model:
        logger.warning("Could not initialize dummy embedding model for LlamaIndex settings.")
        if hasattr(test_llm, 'tokenizer') and test_llm.tokenizer: # type: ignore
             Settings.tokenizer = test_llm.tokenizer # type: ignore
             logger.info(f"Manually set Settings.tokenizer from test LLM: {Settings.tokenizer}")
        else:
             logger.error("Cannot set Settings.tokenizer. Token counting will be inaccurate.")
             exit()
    else:
        configure_llama_index_settings(test_llm, dummy_embed_model, CHUNK_SIZE, CHUNK_OVERLAP) # type: ignore
        logger.info(f"LlamaIndex settings configured. Settings.tokenizer: {Settings.tokenizer}")

    if Settings.tokenizer:
        logger.info("--- Testing get_token_count ---")
        test_string_1 = "Hello world, this is a test."
        count_1 = get_token_count(test_string_1)
        logger.info(f"Token count for '{test_string_1}': {count_1}")
        test_string_2 = "LlamaIndex"
        count_2 = get_token_count(test_string_2)
        logger.info(f"Token count for '{test_string_2}': {count_2}")
    else:
        logger.warning("Settings.tokenizer was not set. Skipping get_token_count tests.")
    logger.info("--- RAG Pipeline Logic Module Test Run Finished ---")
