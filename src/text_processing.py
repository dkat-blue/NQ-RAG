# src/text_processing.py
import logging

logger = logging.getLogger(__name__)

SUPPRESS_EMPTY_SPAN_WARNINGS = True # Set to True to suppress warnings about empty spans after filtering

def is_html_like(token_str):
    """
    A basic heuristic to identify HTML tags.
    """
    is_html = isinstance(token_str, str) and \
              token_str.startswith('<') and \
              token_str.endswith('>') and \
              ' ' not in token_str.strip('<>')
    return is_html

def extract_and_filter_text_from_simplified_doc(document_text_str_or_list, start_token_idx, end_token_idx):
    """
    Extracts a span of tokens from the simplified document_text (or a pre-split list of tokens),
    filters out HTML-like tokens, and joins the remaining tokens.
    The input `document_text_str_or_list` is expected to be the *specific span of tokens*
    if this function is called after the pre-filter in data_preparation.py.
    In that case, start_token_idx should be 0 and end_token_idx should be len(document_text_str_or_list).
    """
    if not isinstance(document_text_str_or_list, (str, list)):
        logger.error("Input to extract_and_filter_text_from_simplified_doc must be a string or a list of token strings.")
        return ""

    all_token_strings = []
    if isinstance(document_text_str_or_list, str):
        # This path might be less used if pre-filtering passes a list
        all_token_strings = document_text_str_or_list.split(' ')
    else: # Is a list
        all_token_strings = document_text_str_or_list
    
    if not all_token_strings:
        return ""

    # If all_token_strings is already the specific span, start_token_idx is 0 and end_token_idx is len(all_token_strings)
    # If it's the full document's tokens, these indices are relative to that full list.
    # The logic here assumes the indices are relative to the passed `all_token_strings`.
    start = max(0, start_token_idx)
    end = min(len(all_token_strings), end_token_idx) 
    
    if start >= end:
        return ""
            
    candidate_token_strings_slice = all_token_strings[start:end] # This is the actual slice to work on
    
    if not candidate_token_strings_slice: 
        return ""

    kept_tokens = []
    for token in candidate_token_strings_slice:
        if not is_html_like(token):
            kept_tokens.append(token)
    
    result_text = " ".join(kept_tokens).strip()
    
    # Check if the original slice had non-empty content before filtering
    # The `any(t and t.strip() for t in candidate_token_strings_slice)` checks if there was any non-whitespace content initially.
    if not result_text and candidate_token_strings_slice and any(t and t.strip() for t in candidate_token_strings_slice):
        if not SUPPRESS_EMPTY_SPAN_WARNINGS:
            logger.warning(
                f"TEXT_PROCESSING_DEBUG: All non-empty tokens in span "
                f"(relative indices {start_token_idx}-{end_token_idx} of input list/slice) were filtered. "
                f"Input slice sample (first 20 tokens): {candidate_token_strings_slice[:20]}. "
                # f"Kept tokens sample (first 20): {kept_tokens[:20]}" # Kept tokens will be empty here
            )
        return "" # Return empty string as per the patch logic (silently drop)

    return result_text

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    SUPPRESS_EMPTY_SPAN_WARNINGS = False # For testing, show warnings

    sample_doc_text = "<P> This is a <B> test </B> document . </P> <TABLE> <TR> <TD> Cell </TD> </TR> </TABLE> "
    logger.info(f"Original test string: '{sample_doc_text}'")
    
    tokens = sample_doc_text.split(' ')
    logger.info(f"Split tokens: {tokens}")

    # Test case 1: Mixed content - pass the relevant slice
    # Span is "<P> This is a <B> test </B> document ." -> tokens[0:10]
    slice1 = tokens[0:10]
    extracted1 = extract_and_filter_text_from_simplified_doc(slice1, 0, len(slice1))
    logger.info(f"Test 1 (slice 0-10): Extracted='{extracted1}'. Expected around: 'This is a test document .'")

    # Test case 2: All content - pass all tokens
    extracted_all = extract_and_filter_text_from_simplified_doc(tokens, 0, len(tokens))
    logger.info(f"Test 2 (all tokens): Extracted='{extracted_all}'. Expected around: 'This is a test document . Cell comment'")
    
    # Test case 3: Candidate that is all HTML - pass the relevant slice
    # Slice: tokens[10:12] is "<TABLE> <TR>"
    slice3 = tokens[10:12]
    extracted_html_only_strict = extract_and_filter_text_from_simplified_doc(slice3, 0, len(slice3))
    logger.info(f"Test 3 (HTML only slice, 10-12): Extracted='{extracted_html_only_strict}'. Expected: ''")

    # Test case 4: Candidate that is just "<P> </P>"
    empty_candidate_text = "<P> </P>"
    empty_candidate_tokens = empty_candidate_text.split(' ')
    extracted_empty_candidate = extract_and_filter_text_from_simplified_doc(empty_candidate_tokens, 0, len(empty_candidate_tokens))
    logger.info(f"Test 4 (Empty P tags): Extracted='{extracted_empty_candidate}'. Expected: ''")
    
    # Test case from logs:
    log_sample_tokens = ['<Tr>', '<Td>', '<Ul>', '<Li>', '</Li>', '<Li>', '</Li>', '<Li>', '</Li>', '</Ul>', '</Td>', '</Tr>']
    extracted_log_sample = extract_and_filter_text_from_simplified_doc(log_sample_tokens, 0, len(log_sample_tokens))
    logger.info(f"Test Log Sample: Extracted='{extracted_log_sample}'. Expected: '' (and should trigger warning if SUPPRESS_EMPTY_SPAN_WARNINGS is False)")

