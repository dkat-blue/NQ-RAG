# NLP Lab: Retrieval-Augmented Generation (RAG) Pipeline

This repository contains the code and notebooks for an NLP lab, focusing on building and evaluating a Retrieval-Augmented Generation (RAG) pipeline. The system uses a vector database (Qdrant), an embedding model (e.g., E5-small-v2), and a generator LLM (e.g., Gemma-family) to answer questions based on a knowledge base derived from the Natural Questions (NQ) dataset.

## Project Structure

```
.
├── data/                       # (Ignored by .gitignore) Placeholder for datasets like NQ
├── notebooks/                  # Jupyter notebooks for main pipeline execution
├── src/                        # Python source code for modules
│   ├── config.py               # Configuration parameters
│   ├── data_loader.py          # Scripts for loading data
│   ├── data_preparation.py     # Scripts for preparing data for indexing
│   ├── evaluation_utils.py     # Utilities for evaluating retrieval and generation
│   ├── llm_setup.py            # LLM and LlamaIndex setup utilities
│   ├── rag_pipeline_logic.py   # Core RAG pipeline logic (indexing, retrieval, generation)
│   └── text_processing.py      # Text processing utilities
├── .gitignore                  # Specifies intentionally untracked files that Git should ignore
└── requirements.txt            # Python package dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Qdrant:** Ensure a Qdrant instance is running and accessible (e.g., via Docker). Configure the URL in `src/config.py` if necessary.
5.  **Set up Local LLM Server:** Ensure your local LLM inference server (e.g., LM Studio, Ollama with OpenAI-compatible API) is running and the API endpoint is correctly configured in `src/config.py` (`LLM_API_BASE`).
6.  **Download Data:** Place the Natural Questions dataset (e.g., `v1.0-simplified-nq-train.jsonl.gz`) into a `data/` directory in the project root (this directory will be created if it doesn't exist, but is ignored by git).

## Running the Pipeline

The main RAG pipeline, including data loading, indexing, retrieval, generation, and evaluation, can typically be run via the Jupyter notebook in the `notebooks/` directory (e.g., `nq_rag.ipynb`). Follow the steps outlined in the notebook.

Key steps usually involve:
1.  Loading and preparing the NQ dataset.
2.  Building or loading the knowledge base in Qdrant.
3.  Running retrieval evaluations.
4.  Generating answers with and without RAG.
5.  Evaluating the generated answers using metrics like ROUGE, BLEU, and LLM-as-a-Judge.

## Evaluation Results Summary

The following results were obtained from a sample run:

---

### Retrieval Performance

| Metric             | Value  |
| ------------------ | ------ |
| Recall@1           | 0.2900 |
| Recall@3           | 0.5567 |
| Recall@5           | 0.6867 |
| Recall@10          | 0.8067 |

---

### Generation Performance

#### Without RAG (Baseline)

| Metric                               | Value  |
| ------------------------------------ | ------ |
| ROUGE-L (F-measure)                  | 0.0703 |
| BLEU-4                               | 0.0089 |
| LLM Judge Avg Correctness/Relevance  | 0.4633 |

#### With RAG

| Metric                               | Value  |
| ------------------------------------ | ------ |
| ROUGE-L (F-measure)                  | 0.3318 |
| BLEU-4                               | 0.1420 |
| LLM Judge Avg Correctness/Relevance  | 0.7033 |
| LLM Judge Avg Faithfulness to Context| 0.9233 |

---

These results demonstrate a significant improvement in answer quality and faithfulness when using the RAG pipeline compared to the baseline LLM without retrieval augmentation.

## Configuration

Key parameters for the pipeline, such as model names, Qdrant settings, `TOP_K` values, and file paths, can be adjusted in `src/config.py`.

