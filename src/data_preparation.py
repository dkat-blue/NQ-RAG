# src/data_preparation.py
import logging
import random
from tqdm.notebook import tqdm 
from llama_index.core import Document 

from src.text_processing import extract_and_filter_text_from_simplified_doc, is_html_like

logger = logging.getLogger(__name__)

def get_generation_target(example, document_tokens_list):
    """Extracts the generation target (reference answer) using pre-split document_tokens_list."""
    if 'annotations' not in example or not example['annotations']:
        logger.debug(f"Example {example.get('example_id')} missing annotations. Cannot get target.")
        return None
        
    annotation = example['annotations'][0]

    if annotation.get('short_answers'):
        texts = []
        for sa in annotation['short_answers']:
            start_tok, end_tok = sa.get('start_token', -1), sa.get('end_token', -1)
            if start_tok != -1 and end_tok != -1 and start_tok < end_tok:
                span_tokens_sa = document_tokens_list[start_tok:end_tok]
                if not span_tokens_sa: continue
                extracted_text = extract_and_filter_text_from_simplified_doc(
                    span_tokens_sa, 0, len(span_tokens_sa)
                )
                if extracted_text:
                    texts.append(extracted_text)
        if texts:
             return " ".join(texts)

    if annotation.get('yes_no_answer') and annotation['yes_no_answer'] in ['YES', 'NO']:
        return annotation['yes_no_answer']

    if annotation.get('long_answer'):
        la_ann = annotation['long_answer']
        cand_idx = la_ann.get('candidate_index', -1)
        if 0 <= cand_idx < len(example.get('long_answer_candidates', [])):
            cand = example['long_answer_candidates'][cand_idx]
            start_tok, end_tok = cand.get('start_token', -1), cand.get('end_token', -1)
            if start_tok != -1 and end_tok != -1 and start_tok < end_tok:
                span_tokens_la = document_tokens_list[start_tok:end_tok]
                if not span_tokens_la: return None
                long_answer_text = extract_and_filter_text_from_simplified_doc(
                    span_tokens_la, 0, len(span_tokens_la)
                )
                if long_answer_text: 
                    return long_answer_text
    return None

def prepare_kb_and_test_set(all_data, num_test_examples, random_seed=42):
    """Prepares the knowledge base and test set with optimized candidate selection."""
    random.seed(random_seed)
    knowledge_base_docs_llama = []
    
    logger.info("Building Knowledge Base with optimized candidate selection...")
    skipped_candidates_count = 0
    processed_candidates_count = 0

    for example in tqdm(all_data, desc="Building KB", unit=" examples"):
        doc_text_str = example.get('document_text')
        if not doc_text_str or not isinstance(doc_text_str, str):
            continue
        
        document_tokens_list = doc_text_str.split(' ')
        
        # Determine the gold candidate index for this example
        gold_candidate_idx = -1
        if example.get('annotations') and example['annotations'][0].get('long_answer'):
            gold_candidate_idx = example['annotations'][0]['long_answer'].get('candidate_index', -1)

        for cand_idx, cand in enumerate(example.get('long_answer_candidates', [])):
            processed_candidates_count +=1
            start_token = cand.get('start_token', -1)
            end_token = cand.get('end_token', -1)
            
            if start_token == -1 or end_token == -1 or start_token >= end_token:
                skipped_candidates_count += 1
                continue

            is_gold_candidate = (cand_idx == gold_candidate_idx)
            is_top_level_candidate = cand.get("top_level", False)

            if not (is_gold_candidate or is_top_level_candidate):
                skipped_candidates_count += 1
                continue # Skip if not gold AND not top-level

            actual_start = max(0, start_token)
            actual_end = min(len(document_tokens_list), end_token)
            if actual_start >= actual_end:
                skipped_candidates_count += 1
                continue

            span_tokens = document_tokens_list[actual_start:actual_end]
            if not span_tokens:
                skipped_candidates_count += 1
                continue

            if not any(t and not is_html_like(t) for t in span_tokens):
                skipped_candidates_count += 1
                continue
            
            text_content = extract_and_filter_text_from_simplified_doc(
                span_tokens, 0, len(span_tokens) 
            )
            
            if not text_content:
                skipped_candidates_count += 1
                continue
            
            doc_id = f"{example['example_id']}_{cand_idx}"
            llama_doc = Document(
                text=text_content,
                doc_id=doc_id,
                metadata={
                    'example_id': str(example.get('example_id', 'N/A')),
                    'candidate_index': cand_idx,
                    'is_gold': is_gold_candidate,
                    'is_top_level': is_top_level_candidate,
                    'original_question': example.get('question_text', ''),
                    'document_url': example.get('document_url', ''),
                }
            )
            knowledge_base_docs_llama.append(llama_doc)
            
    logger.info(f"Total candidates processed: {processed_candidates_count}, Skipped: {skipped_candidates_count}")
    logger.info(f"Built KB with {len(knowledge_base_docs_llama)} LlamaIndex documents (after optimized selection).")

    # Test set preparation
    test_set = []
    gold_map_for_recall = {}
    potential_test_examples = []
    logger.info("Screening examples for test set...")

    for example in tqdm(all_data, desc="Screening for test set", unit=" examples"):
        doc_text_str_for_test = example.get('document_text')
        if not all(k in example for k in ['annotations', 'long_answer_candidates', 'question_text', 'example_id']) or \
           not doc_text_str_for_test or not isinstance(doc_text_str_for_test, str):
            continue
        
        document_tokens_list_for_test = doc_text_str_for_test.split(' ')
        target_answer = get_generation_target(example, document_tokens_list_for_test)
        
        if target_answer:
            first_annotation = example['annotations'][0]
            if first_annotation.get('long_answer'):
                la_ann = first_annotation['long_answer']
                # Ensure the gold candidate for the test example is valid and exists
                gold_cand_idx_for_test = la_ann.get('candidate_index', -1)
                if 0 <= gold_cand_idx_for_test < len(example.get('long_answer_candidates', [])):
                    # Check if this gold candidate would have been included in the KB
                    gold_cand_for_test = example['long_answer_candidates'][gold_cand_idx_for_test]
                    is_gold = True # by definition for the annotation
                    is_top_level = gold_cand_for_test.get("top_level", False)
                    
                    if is_gold or is_top_level: # Ensure the gold answer for test set is from a candidate we'd keep for KB
                        potential_test_examples.append(example)
    
    random.shuffle(potential_test_examples)
    selected_test_examples_full = potential_test_examples[:min(len(potential_test_examples), num_test_examples)]

    logger.info(f"Selected {len(selected_test_examples_full)} examples for the test set.")
    for example in tqdm(selected_test_examples_full, desc="Processing test set", unit=" examples"):
        document_tokens_list_for_final_test = example['document_text'].split(' ')
        target_answer = get_generation_target(example, document_tokens_list_for_final_test)
        
        test_item = {
            'example_id': str(example['example_id']),
            'question_text': example['question_text'],
            'target_answer': target_answer,
        }
        test_set.append(test_item)
        
        la_ann = example['annotations'][0]['long_answer']
        gold_candidate_idx = la_ann['candidate_index']
        gold_doc_id_for_recall = f"{example['example_id']}_{gold_candidate_idx}"
        gold_map_for_recall[example['question_text']] = gold_doc_id_for_recall
        
    logger.info(f"Prepared {len(test_set)} test examples.")
    if gold_map_for_recall:
         logger.info(f"Prepared gold map for recall for {len(gold_map_for_recall)} questions.")
    return test_set, knowledge_base_docs_llama, gold_map_for_recall

if __name__ == '__main__':
    import json 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    dummy_example_1 = {
        "example_id": "ex1", "question_text": "Q1", 
        "document_text": "<P> Gold answer . </P> <P> Top level one . </P> <P> Not top level , not gold . </P>",
        "long_answer_candidates": [
            {"start_token": 0, "end_token": 5, "top_level": True, "candidate_index_in_source": 0}, # Gold, Top-level
            {"start_token": 6, "end_token": 11, "top_level": True, "candidate_index_in_source": 1}, # Not Gold, Top-level
            {"start_token": 12, "end_token": 18, "top_level": False, "candidate_index_in_source": 2} # Not Gold, Not Top-level
        ],
        "annotations": [{"long_answer": {"candidate_index": 0}}] # First candidate is gold
    }
    dummy_all_data = [dummy_example_1]
    test_set, kb_docs, recall_map = prepare_kb_and_test_set(dummy_all_data, num_test_examples=1)

    logger.info(f"\n--- KB Docs ({len(kb_docs)} found) ---") # Expect 2 docs
    for i, doc in enumerate(kb_docs):
        logger.info(f"KB Doc [{i}]: '{doc.text}', Metadata: {doc.metadata}")
