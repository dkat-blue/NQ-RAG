# src/evaluation_utils.py
import logging
import numpy as np
from tqdm.notebook import tqdm # or tqdm.auto
import re
import os

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings, PromptTemplate

from src.config import JUDGE_MODEL_CTX_WINDOW, PROMPT_SAFETY_MARGIN

logger = logging.getLogger(__name__)

def get_token_count_eval(text: str) -> int:
    """Counts tokens using the globally set LlamaIndex tokenizer for evaluation purposes."""
    tokenizer = Settings.tokenizer
    if not tokenizer:
        logger.warning("Settings.tokenizer is None for eval. Falling back to len(text.split()).")
        return len(text.split())
    try:
        if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
            # It's an object with an .encode() method (e.g., tiktoken.Encoding instance)
            return len(tokenizer.encode(text))
        elif callable(tokenizer):
            # It's a callable function (e.g., the default functools.partial from LlamaIndex's get_tokenizer)
            return len(tokenizer(text))
        else:
            logger.warning(f"Settings.tokenizer (type: {type(tokenizer)}) is not a recognized tokenizer for eval. Falling back to len(text.split()).")
            return len(text.split())
    except Exception as e:
        logger.error(f"Error using Settings.tokenizer (type: {type(tokenizer)}) for token count in eval: {e}. Falling back to len(text.split()).", exc_info=True)
        return len(text.split())

def evaluate_recall_at_k(test_set_items, vector_idx, gold_mapping, k_values):
    """Evaluates Recall@K for the retriever. Uses node.ref_doc_id for matching."""
    recall_scores = {k: [] for k in k_values}
    max_k_for_eval = 0
    if k_values:
        max_k_for_eval = max(k_values)
    else:
        logger.warning("k_values for recall evaluation is empty. Skipping recall.")
        return {k: 0.0 for k in k_values} # type: ignore

    if not vector_idx:
        logger.warning("Vector index not initialized. Skipping recall evaluation.")
        return {k: 0.0 for k in k_values} # type: ignore
    if not test_set_items or not gold_mapping:
        logger.warning("Test set or gold mapping is empty. Skipping recall evaluation.")
        return {k: 0.0 for k in k_values} # type: ignore

    eval_retriever = VectorIndexRetriever(index=vector_idx, similarity_top_k=max_k_for_eval)
    logger.info(f"Evaluating recall with retriever top_k={max_k_for_eval}")

    for item in tqdm(test_set_items, desc="Evaluating Recall@K", unit=" questions"):
        question = item['question_text']
        gold_ref_doc_id = gold_mapping.get(question)

        if not gold_ref_doc_id:
            for k_val in k_values: recall_scores[k_val].append(0)
            continue

        try:
            retrieved_nodes_with_scores = eval_retriever.retrieve(question)
            retrieved_doc_ids = [node_with_score.node.ref_doc_id for node_with_score in retrieved_nodes_with_scores if node_with_score.node]
            for k in k_values:
                if gold_ref_doc_id in retrieved_doc_ids[:k]:
                    recall_scores[k].append(1)
                else:
                    recall_scores[k].append(0)
        except Exception as e:
            logger.error(f"Error during recall evaluation for query '{question}': {e}", exc_info=True)
            for k_val in k_values: recall_scores[k_val].append(0)

    avg_recall_scores = {k: np.mean(scores) if scores else 0.0 for k, scores in recall_scores.items()}
    return avg_recall_scores


def calculate_rouge_l_and_bleu(predictions, references):
    """Calculates average ROUGE-L (F-measure) and BLEU-4 scores."""
    rouge_l_f_scores = []
    bleu_scores = []

    if not predictions or not references or len(predictions) != len(references):
        logger.warning("Predictions or references are empty or mismatched. Returning zero ROUGE/BLEU.")
        return {'rougeL_fmeasure': 0.0, 'bleu_4': 0.0}

    rouge_l_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction().method1

    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Calculating ROUGE & BLEU", unit=" pairs"):
        if not (isinstance(pred, str) and pred and isinstance(ref, str) and ref):
            rouge_l_f_scores.append(0.0)
            bleu_scores.append(0.0)
            continue
        try:
            rouge_l_score = rouge_l_scorer_instance.score(ref, pred)['rougeL'].fmeasure
            rouge_l_f_scores.append(rouge_l_score)
        except Exception as e:
            logger.error(f"Error calculating ROUGE for pred='{pred}', ref='{ref}': {e}")
            rouge_l_f_scores.append(0.0)
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        try:
            bleu_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry, auto_reweigh=True)
        except ZeroDivisionError:
            bleu_score = 0.0
        except Exception as e:
            logger.error(f"Error calculating BLEU for pred='{pred}', ref='{ref}': {e}")
            bleu_score = 0.0
        bleu_scores.append(bleu_score)
    return {
        'rougeL_fmeasure': np.mean(rouge_l_f_scores) if rouge_l_f_scores else 0.0,
        'bleu_4': np.mean(bleu_scores) if bleu_scores else 0.0
    }

def parse_judge_response(response_text):
    """Parses YES/NO response from the judge LLM."""
    response_text_upper = str(response_text).strip().upper()
    if re.search(r'\bYES\b', response_text_upper) or response_text_upper.startswith("YES"):
        return 1
    if re.search(r'\bNO\b', response_text_upper) or response_text_upper.startswith("NO"):
        return 0
    return 0

def evaluate_with_llm_judge(judge_llm_instance, prompt_template, items,
                            generated_answers, references=None, contexts=None,
                            progress_desc="LLM Judge Evaluation"):
    scores = []
    if not judge_llm_instance:
        logger.warning("Judge LLM not initialized. Skipping LLM-based evaluation.")
        return [0.0] * len(items) if items else []
    if not items or not generated_answers or len(items) != len(generated_answers):
        logger.warning("Items or generated_answers are empty or mismatched for LLM Judge. Skipping.")
        return []
    if references and len(items) != len(references): # type: ignore
        logger.warning("Items and references mismatched for LLM Judge. Skipping.")
        return [0.0] * len(items) if items else []
    if contexts and len(items) != len(contexts): # type: ignore
        logger.warning("Items and contexts mismatched for LLM Judge. Skipping.")
        return [0.0] * len(items) if items else []
    
    judge_prompt_budget = JUDGE_MODEL_CTX_WINDOW - PROMPT_SAFETY_MARGIN

    for i, item in tqdm(enumerate(items), total=len(items), desc=progress_desc, unit=" items"):
        query = item['question_text']
        generated_answer = generated_answers[i]
        reference_answer = references[i] if references and i < len(references) else "" # type: ignore
        context_str = contexts[i] if contexts and i < len(contexts) else "" # type: ignore

        if generated_answer == "ERROR_GENERATING_ANSWER" or \
           (contexts and context_str == "ERROR_RETRIEVING_CONTEXT" and "context_str" in prompt_template.template_vars) or \
           (contexts and context_str == "CONTEXT_EMPTY_OR_TRUNCATED" and "context_str" in prompt_template.template_vars) or \
           (contexts and context_str == "CONTEXT_EMPTY_DUE_TO_BUDGET" and "context_str" in prompt_template.template_vars) :
            scores.append(0.0)
            continue

        prompt_args = {
            'query_str': query,
            'generated_answer_str': generated_answer
        }
        current_template_vars = set(prompt_template.template_vars)
        if 'reference_answer_str' in current_template_vars:
            prompt_args['reference_answer_str'] = reference_answer
        if 'context_str' in current_template_vars:
            prompt_args['context_str'] = context_str

        try:
            temp_args_no_context = prompt_args.copy()
            if 'context_str' in temp_args_no_context: temp_args_no_context['context_str'] = ""
            
            min_args_for_base_format = {k: temp_args_no_context.get(k, f"{{{k}}}") for k in current_template_vars if k != 'context_str'}
            if 'context_str' in current_template_vars :
                 min_args_for_base_format['context_str'] = ""
            temp_fmt_prompt_no_context = prompt_template.format(**min_args_for_base_format)
        except KeyError as e:
            logger.error(f"KeyError formatting judge prompt (no context) for query '{query}': {e}. Skipping. Args: {min_args_for_base_format}, Template Vars: {current_template_vars}")
            scores.append(0.0)
            continue
        
        base_tokens = get_token_count_eval(temp_fmt_prompt_no_context)
        available_for_context = judge_prompt_budget - base_tokens
        
        if 'context_str' in prompt_args and prompt_args['context_str']:
            original_context_tokens = get_token_count_eval(prompt_args['context_str'])
            if original_context_tokens > available_for_context:
                if available_for_context <= 0:
                    logger.warning(f"No budget for context in judge prompt for query '{query[:50]}...'. Context will be empty. Base tokens: {base_tokens}")
                    prompt_args['context_str'] = ""
                else:
                    # Simple char-based trim (rough)
                    estimated_chars_per_token = len(prompt_args['context_str']) / original_context_tokens if original_context_tokens > 0 else 4
                    max_chars = int(available_for_context * estimated_chars_per_token)
                    prompt_args['context_str'] = prompt_args['context_str'][:max_chars]
        try:
            missing_vars = current_template_vars - set(prompt_args.keys())
            for var in missing_vars: prompt_args[var] = ""
            fmt_prompt = prompt_template.format(**prompt_args)
            final_prompt_tokens = get_token_count_eval(fmt_prompt)
            if final_prompt_tokens > JUDGE_MODEL_CTX_WINDOW :
                logger.warning(f"Judge prompt for query '{query[:50]}...' is too long: {final_prompt_tokens} tokens (JUDGE_MODEL_CTX_WINDOW {JUDGE_MODEL_CTX_WINDOW}). Skipping.")
                scores.append(0.0)
                continue
        except KeyError as e:
            logger.error(f"KeyError formatting judge prompt for query '{query}': {e}. Available args: {prompt_args.keys()}. Skipping.")
            scores.append(0.0)
            continue
        except Exception as e:
            logger.error(f"General error formatting judge prompt for query '{query}': {e}. Skipping.")
            scores.append(0.0)
            continue
        try:
            judge_response_obj = judge_llm_instance.complete(fmt_prompt)
            scores.append(parse_judge_response(str(judge_response_obj)))
        except Exception as e:
            logger.error(f"Error during LLM judge API call for query '{query}': {e}. Assigning 0.", exc_info=True)
            scores.append(0.0)
    return scores

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        from src.llm_setup import setup_llm, configure_llama_index_settings, setup_embedding_model
        from src.config import (
            LLM_API_BASE, LLM_API_KEY, JUDGE_LLM_MODEL_NAME,
            EMBED_MODEL_NAME, EMBED_BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP,
            JUDGE_MODEL_CTX_WINDOW as cfg_judge_window,
            PROMPT_SAFETY_MARGIN as cfg_safety_margin
        )
    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        logger.info("Skipping __main__ test block for evaluate_with_llm_judge.")
        exit()
    logger.info("--- Starting Test for evaluate_with_llm_judge ---")
    actual_judge_llm = setup_llm(LLM_API_BASE, LLM_API_KEY, JUDGE_LLM_MODEL_NAME, "JudgeTest", temperature=0.0)
    if not actual_judge_llm:
        logger.error("Failed to initialize the judge LLM. Aborting test.")
        exit()
    dummy_embed_model = setup_embedding_model(EMBED_MODEL_NAME, batch_size=EMBED_BATCH_SIZE, device_preference="cpu") # type: ignore
    if not dummy_embed_model:
        logger.warning("Could not initialize a dummy embedding model.")
        if hasattr(actual_judge_llm, 'tokenizer') and actual_judge_llm.tokenizer: # type: ignore
             Settings.tokenizer = actual_judge_llm.tokenizer # type: ignore
             logger.info(f"Manually set Settings.tokenizer from judge LLM: {Settings.tokenizer}")
        else:
             logger.error("Cannot set Settings.tokenizer. Token counting will be inaccurate.")
             exit()
    else:
        configure_llama_index_settings(actual_judge_llm, dummy_embed_model, CHUNK_SIZE, CHUNK_OVERLAP) # type: ignore
        logger.info(f"LlamaIndex settings configured. Settings.tokenizer: {Settings.tokenizer}")

    test_items_mock = [{'question_text': 'What is the color of the sky?'},{'question_text': 'Is Python a programming language?'}]
    gen_answers_mock = ['The sky is typically blue during the day.','Yes, Python is a widely used high-level programming language.']
    refs_mock = ['Blue.','Yes.']
    long_context_text = "This is a very long string designed to test the context trimming. " * (cfg_judge_window // 5)
    contexts_mock = ['The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere.',long_context_text]
    judge_prompt_str_mock = ("You are an impartial AI assistant evaluating the correctness and relevance of a Generated Answer to a Question, "
                             "using a Reference Answer as a guide and considering the Provided Context if available. "
                             "Consider if the Generated Answer accurately addresses the Question and aligns "
                             "with the information expected, as exemplified by the Reference Answer and supported by Context.\n\n"
                             "Provided Context: {context_str}\n\nQuestion: {query_str}\nReference Answer: {reference_answer_str}\n"
                             "Generated Answer: {generated_answer_str}\n\n"
                             "Is the Generated Answer correct, relevant, and faithful to the context (if provided)? Respond with only YES or NO.")
    judge_prompt_tmpl_mock = PromptTemplate(judge_prompt_str_mock)
    logger.info(f"Testing with JUDGE_MODEL_CTX_WINDOW={cfg_judge_window}, PROMPT_SAFETY_MARGIN={cfg_safety_margin}")
    logger.info("Calling evaluate_with_llm_judge with mock data and actual LLM...")
    llm_judge_scores = evaluate_with_llm_judge(actual_judge_llm, judge_prompt_tmpl_mock, test_items_mock, gen_answers_mock,
                                               references=refs_mock, contexts=contexts_mock, progress_desc="Testing LLM Judge")
    logger.info(f"LLM Judge Scores: {llm_judge_scores}")
    for i, item in enumerate(test_items_mock):
        logger.info(f"--- Test Item {i+1} ---")
        logger.info(f"  Question: {item['question_text']}")
        logger.info(f"  Generated: {gen_answers_mock[i]}")
        logger.info(f"  Reference: {refs_mock[i]}")
        logger.info(f"  Context Provided: {'Yes' if contexts_mock[i] else 'No'}")
        logger.info(f"  Judge Score: {llm_judge_scores[i] if i < len(llm_judge_scores) else 'N/A'}")
    logger.info("--- Test Run Finished ---")

