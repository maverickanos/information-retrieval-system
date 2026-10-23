
# Hybrid Information Retrieval

A document search system that combines lexical matching, statistical term similarity, and semantic retrieval to produce a unified ranking of relevant documents. The system indexes a local JSON corpus, retrieves results using BM25, TF-IDF, and Sentence Transformers, and combines the rankings through Reciprocal Rank Fusion (RRF). Search and evaluation are exposed through a FastAPI backend and a browser-based interface.

## Overview

Traditional keyword-based search can miss relevant documents when they use different wording from a query, while semantic retrieval can overlook exact terminology. This project combines complementary retrieval strategies to improve document discovery by considering both term-level and semantic relevance.

The system provides document indexing, hybrid retrieval, ranking fusion, evaluation metrics, an HTTP API, and a web interface for interactive searches.

## Features

- Text preprocessing with NLTK, including tokenization, lowercasing, punctuation removal, stop-word filtering, and lemmatization.
- BM25Okapi for lexical retrieval.
- TF-IDF vectorization and cosine similarity for statistical term matching.
- Sentence Transformer embeddings using `all-MiniLM-L6-v2`.
- FAISS inner-product search over normalized 384-dimensional embeddings.
- Reciprocal Rank Fusion (RRF) to combine rankings from multiple retrieval methods.
- Search results containing document details, snippets, retrieval scores, individual ranks, and final ranking.
- FastAPI endpoints for indexing, searching, health checks, and evaluation.
- Browser interface built with HTML, CSS, and JavaScript.
- Retrieval evaluation using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- Automated tests using pytest.
- Docker-based application execution.

## System Architecture

```text
                         Search Query
                              |
                              v
                       Text Preprocessing
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           BM25            TF-IDF       Sentence Transformer
              |               |               |
              |               |               v
              |               |              FAISS
              +---------------+---------------+
                              |
                              v
                   Reciprocal Rank Fusion
                              |
                              v
                    Unified Ranked Results
                              |
                              v
                       FastAPI / Web UI
```

The indexing pipeline preprocesses the document collection and builds the retrieval indexes. For each search request, the system independently ranks documents using BM25, TF-IDF, and semantic similarity. Reciprocal Rank Fusion then combines the rankings into a single ordered result list.

## Retrieval Pipeline

### 1. BM25 Retrieval

BM25Okapi ranks documents using term frequency, inverse document frequency, and document-length normalization. It is particularly useful when query terms appear explicitly in relevant documents.

### 2. TF-IDF Retrieval

The document collection is represented using TF-IDF vectors. Each processed query is compared with the document matrix using cosine similarity, generating a ranking based on statistical term similarity.

### 3. Semantic Search with Sentence Transformers and FAISS

The `all-MiniLM-L6-v2` model encodes documents and queries into 384-dimensional embeddings. The vectors are L2-normalized and stored in a FAISS inner-product index.

For normalized vectors, inner-product search provides a measure of semantic similarity between a query and indexed documents.

### 4. Reciprocal Rank Fusion

Each retrieval method generates its own ranked list. Reciprocal Rank Fusion combines the document positions across these lists using:

```text
RRF(document) = sum(1 / (k + rank))
```

The implementation uses `k = 60`. The RRF score is a ranking score rather than a percentage. Documents are sorted by their fused scores to generate the final ranking.

This approach combines lexical matching, statistical term similarity, and semantic retrieval without requiring the individual retrieval scores to share the same scale.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10+ |
| Text Processing | NLTK |
| Lexical Retrieval | rank-bm25 / BM25Okapi |
| Statistical Retrieval | scikit-learn TF-IDF and cosine similarity |
| Semantic Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Backend API | FastAPI and Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Evaluation and Testing | Python, pytest |
| Containerization | Docker |
| Data Storage | JSON |

## Dataset and Evaluation

The system loads documents from `data/documents.json`. Each document contains an identifier, title, and content. Evaluation queries and their relevant document identifiers are stored in `data/ground_truth.json`.

Retrieval quality is evaluated using the following metrics:

- **Precision@K:** Measures the proportion of the top K retrieved documents that are relevant.
- **Recall@K:** Measures the proportion of all known relevant documents retrieved within the top K results.
- **Mean Reciprocal Rank (MRR):** Measures the average reciprocal rank of the first relevant result across evaluation queries.

Evaluation results depend on the document corpus and the quality of the relevance labels.

## Project Structure

```text
.
├── api.py
├── config.py
├── Dockerfile
├── evaluator.py
├── indexer.py
├── main.py
├── preprocess.py
├── README.md
├── requirements.txt
├── search_engine.py
├── artifacts/
│   ├── bm25.pkl
│   ├── faiss.index
│   ├── metadata.json
│   └── tfidf.pkl
├── data/
│   ├── documents.json
│   └── ground_truth.json
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
└── tests/
    ├── test_api.py
    ├── test_evaluator.py
    ├── test_indexer.py
    ├── test_preprocess.py
    └── test_search_engine.py
```

The files in `artifacts/` are generated during indexing and can be rebuilt when the document corpus changes.

## Installation

Python 3.11 is recommended.

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first indexing operation may download the Sentence Transformer model and required NLTK resources. Network access is required unless these resources are already available locally.

## Running the Application

Start the FastAPI development server:

```bash
python -m uvicorn api:app --reload
```

Open the application in your browser:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

Before the first search, or after modifying the corpus, select **Rebuild index** in the web interface. This preprocesses the documents and creates the BM25, TF-IDF, and FAISS artifacts.

### Performing a Search

1. Start the application and open the browser interface.
2. Rebuild the index if it does not exist or the corpus has changed.
3. Enter a query, such as `machine learning algorithms`.
4. Select the number of results to retrieve.
5. Review the returned documents, snippets, individual retrieval scores, method ranks, RRF scores, and final ranking.
6. Select **Evaluate** to measure retrieval performance against the labeled queries in `data/ground_truth.json`.

The alternative entry point `python main.py` also starts Uvicorn using the host and port configured in `config.py`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Reports service health and whether the search engine is loaded. |
| `POST` | `/index` | Builds the BM25, TF-IDF, and FAISS artifacts from the JSON corpus. |
| `POST` | `/search` | Executes the retrieval methods and returns the fused ranking. |
| `POST` | `/evaluate` | Evaluates retrieval against the ground-truth file. |

### Search Request

```json
{
  "query": "machine learning",
  "top_k": 5
}
```

The query must be non-empty, and `top_k` must be greater than zero.

### Evaluation Request

```json
{
  "k": 10
}
```

The value of `k` must be greater than zero.

## Configuration and Maintenance

- Add or edit documents in `data/documents.json`, then rebuild the index.
- Update relevance judgments in `data/ground_truth.json` before evaluating changes.
- Adjust the model, port, RRF, and default-result settings in `config.py`.
- Modify text normalization in `preprocess.py`.
- Update index construction in `indexer.py`.
- Change retrieval or ranking-fusion logic in `search_engine.py`.
- Modify API routes or request validation in `api.py`.
- Update the browser interface in `frontend/index.html`, `frontend/script.js`, and `frontend/style.css`.
- Run `pytest -q` after code changes.

Generated files in `artifacts/` should be rebuilt when stale rather than edited manually.

## Testing

Run the automated test suite:

```bash
pytest -q
```

The documented test result is:

```text
12 passed, 2 warnings
```

The warnings were previously identified as dependency deprecation warnings rather than test failures. Test results may differ across dependency versions or code changes.

## Docker

Build the Docker image:

```bash
docker build -t hybrid-ir .
```

Run the container:

```bash
docker run --rm -p 8000:8000 hybrid-ir
```

The Docker image installs Python dependencies and downloads NLTK resources during the build. The Sentence Transformer model is downloaded during the first indexing request, so the container requires network access unless the model cache is supplied separately.

## Limitations

- Uses a local JSON corpus rather than a production-scale document store.
- Retrieval quality depends on the corpus and the completeness of relevance labels.
- The embedding model and FAISS index require local storage and computational resources.
- Index artifacts must be rebuilt after corpus changes.
- Initial setup or indexing may require network access to retrieve model and language resources.
- Authentication, user management, and production deployment hardening are outside the current implementation.

## Future Improvements

Potential extensions include:

- Incremental indexing and document updates.
- Experiments with additional embedding models and retrieval configurations.
- Larger document collections and improved relevance judgments.
- Configurable retrieval weights and alternative ranking-fusion strategies.
- Additional result filtering, pagination, and search analytics.
- Improved API authentication, observability, and production deployment.

## Project Status

The system supports local document indexing, hybrid retrieval using BM25, TF-IDF, and semantic search, RRF-based ranking, retrieval evaluation, a FastAPI backend, a browser-based interface, automated tests, and Docker-based execution.
