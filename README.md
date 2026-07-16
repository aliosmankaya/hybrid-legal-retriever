# Hybrid Legal Retriever

A production-ready **hybrid RAG (Retrieval-Augmented Generation) pipeline for Turkish legal documents**, combining sparse (BM25) and dense (Qdrant vector) retrieval fused with Reciprocal Rank Fusion (RRF), an optional cross-encoder reranker, and LLM-powered answer generation via OpenRouter.

The reference corpus is the **Karayolları Trafik Kanunu** (Turkish Highway Traffic Law), but the pipeline is law-agnostic — any legal PDF can be chunked, indexed, and queried.

---

## Features

- **Hybrid retrieval** — combines lexical BM25 search with semantic dense search for high recall and precision on legal text.
- **Reciprocal Rank Fusion** — rank-based fusion (`k=60`) that blends sparse and dense result lists without needing score normalization.
- **Optional cross-encoder reranker** — `BAAI/bge-reranker-base` re-ranks the fused candidates for improved top-k relevance.
- **Turkish-aware tokenization** — custom `tokenize_tr` normalizes Turkish characters (İ→i, I→ı) for BM25.
- **FastAPI service** — upload PDFs, chunk by legal article (`Madde`), build indexes, and chat with the law.
- **LLM answer generation** — grounded responses using `nvidia/nemotron-3-ultra-550b-a55b:free` via LangChain-OpenRouter, with strict "answer only from context" instructions and a legal disclaimer.
- **Evaluation toolkit** — `Hit@k` and `MRR` metrics plus a labeled Q&A dataset (`evaluation/questions.json`) and experiment notebooks.
- **Docker orchestration** — one-command deployment with Qdrant as a sidecar service.

---

## Architecture

```
                ┌─────────────┐
  PDF (upload)  │  src/api    │   FastAPI
       │        │  /file      │
       └──► upload to data/<law>/upload/
                └─────────────┘
                       │
        POST /index/chunking   →  split PDF by "Madde" → chunks.jsonl
        POST /index/indexing   →  dense (Qdrant) + sparse (BM25) indexes
                       │
        POST /chat/chat        →  query
                       │
                ┌──────▼───────────────────────────────┐
                │            Hybrid Retriever            │
                │                                        │
                │   BM25 (rank_bm25)        Dense (Qdrant│
                │        │                      │        │
                │        └──────► RRF ──────────┘        │
                │                  │                     │
                │           (optional Reranker)          │
                └──────────────────┬─────────────────────┘
                                   │ top-k chunks (context)
                            ┌──────▼──────┐
                            │ LLM (chat)  │  OpenRouter + LangChain
                            └──────┬──────┘
                                grounded answer
```

**Retrieval strategy**

| Stage | Component | Detail |
|-------|-----------|--------|
| Sparse | `BM25Okapi` | Turkish-normalized tokens, persisted as `bm25_index.pkl` |
| Dense | `Qdrant` + `intfloat/multilingual-e5-small` | 384-dim cosine vectors, one collection per law |
| Fusion | Reciprocal Rank Fusion | `score += 1 / (k + rank + 1)`, `k=60` |
| Rerank | `CrossEncoder` (`BAAI/bge-reranker-base`) | re-scores `(query, chunk)` pairs, returns top-k |

---

## Project Structure

```
hybrid-legal-retriever/
├── main.py                  # uvicorn entrypoint (host 127.0.0.1:8000)
├── Dockerfile               # python:3.12-slim + Poetry install
├── docker-compose.yml       # app + qdrant services
├── pyproject.toml           # Poetry dependencies
├── src/
│   ├── api/                 # FastAPI routers
│   │   ├── index.py         # /index/chunking, /index/indexing
│   │   ├── chat.py          # /chat/chat
│   │   ├── file.py          # /file/upload, /list, /update, /delete
│   │   └── __init__.py      # app = FastAPI(), router registration
│   ├── core/                # business logic
│   │   ├── index.py         # PDF chunking + Qdrant/BM25 index build
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py  # RRF fusion
│   │   ├── chat.py          # LLM invoke (LangChain-OpenRouter)
│   │   └── helper.py        # tokenize_tr, load_chunks
│   ├── retrieval/           # standalone/experimental retrieval utils
│   │   ├── hybrid_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── reranker.py      # CrossEncoder reranker
│   ├── evaluation/          # metrics + synthetic question generation
│   │   ├── metrics.py       # Hit@k, MRR, evaluate_pipeline
│   │   └── generate_questions.py
│   └── parser/              # Pydantic request models
├── notebooks/               # 0–8 experiments (extraction → evaluation)
├── evaluation/
│   └── questions.json       # 54 labeled Q&A pairs for eval
└── data/
    └── trafik_kanunu/       # chunks.jsonl, bm25_index.pkl, upload/*.pdf
```

---

## Tech Stack

| Concern | Technology |
|---------|-----------|
| Language | Python 3.12 |
| Package manager | Poetry |
| Web framework | FastAPI + Uvicorn |
| Vector store | Qdrant (`qdrant-client`) |
| Embeddings | Sentence-Transformers `intfloat/multilingual-e5-small` |
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Reranker | Sentence-Transformers `BAAI/bge-reranker-base` (CrossEncoder) |
| LLM | LangChain-OpenRouter (`nvidia/nemotron-3-ultra-550b-a55b:free`) |
| PDF parsing | `pdfplumber` |
| Orchestration | Docker, docker-compose |

---

## Prerequisites

- Python 3.12+ (for local runs) **or** Docker + Docker Compose
- A running **Qdrant** instance (provided automatically via docker-compose, or run locally on `localhost:6333`)
- Hugging Face token (`HF_TOKEN`) for model downloads
- OpenRouter API key (`OPENROUTER_API_KEY`) for answer generation

---

## Installation

### Option A — Docker (recommended)

```bash
docker compose up --build
```

This starts:
- `app` on `http://localhost:8000` (with `--reload`)
- `qdrant` on `http://localhost:6333`

### Option B — Local (Poetry)

```bash
# 1. Install dependencies
poetry install

# 2. Start Qdrant locally (e.g. via Docker)
docker run -p 6333:6333 qdrant/qdrant:latest

# 3. Configure environment (see below) and run
poetry run python main.py
```

---

## Environment Variables

Create a `.env` file in the project root:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | yes | — | Hugging Face token for downloading models |
| `OPENROUTER_API_KEY` | yes | — | OpenRouter key for LLM answer generation |
| `QDRANT_HOST` | (Docker) | `localhost` | Qdrant host |
| `QDRANT_PORT` | (Docker) | `6333` | Qdrant port |

> Note: `src/core/dense_retriever.py` connects to `localhost:6333` directly; the docker-compose `app` service uses `QDRANT_HOST=qdrant` / `QDRANT_PORT=6333` env overrides via QdrantClient when configured.

---

## Usage

### 1. Upload a legal PDF

```bash
curl -X POST "http://localhost:8000/file/upload?law_name=trafik_kanunu" \
  -F "file=@data/trafik_kanunu/upload/1.5.2918.pdf"
```

### 2. Chunk the PDF (split by `Madde`)

```bash
curl -X POST "http://localhost:8000/index/chunking" \
  -H "Content-Type: application/json" \
  -d '{"law_name": "trafik_kanunu"}'
```

### 3. Build the indexes (Qdrant dense + BM25 sparse)

```bash
curl -X POST "http://localhost:8000/index/indexing" \
  -H "Content-Type: application/json" \
  -d '{"law_name": "trafik_kanunu"}'
```

### 4. Chat with the law

```bash
curl -X POST "http://localhost:8000/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Trafik zabıtasının görev ve yetkileri nelerdir?", "law_name": "trafik_kanunu", "limit": 20}'
```

### Other file endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/file/list` | List indexed law directories |
| `PUT`  | `/file/update` | Rename a law directory (`current_name`, `new_name`) |
| `DELETE` | `/file/delete` | Remove a law directory (`current_name`) |

Interactive API docs are available at `http://localhost:8000/docs`.

---

## Evaluation

The evaluation module measures retrieval quality with:

- **`Hit@k`** — was the relevant chunk in the top-k results?
- **`MRR`** — Mean Reciprocal Rank of the first relevant chunk.

`evaluation/questions.json` contains 54 labeled question → `relevant_chunk_id` pairs over the Traffic Law. Synthetic question generation (`src/evaluation/generate_questions.py`) uses a local Ollama model (`llama3.2:3b`) to expand the dataset.

Run the experiment notebooks in order to reproduce the analysis:

| Notebook | Topic |
|----------|-------|
| `0_data_extraction` | PDF text extraction |
| `1_indexing` | Chunking & indexing |
| `2_embedding_comparision` | Embedding model comparison |
| `3_naive_rag` | Baseline naive RAG |
| `4_bm25_test` | BM25 retrieval test |
| `5_hybrid_test` | Hybrid (BM25 + dense) test |
| `6_reranker` | Cross-encoder reranking |
| `7_generate_eval_dataset` | Synthetic Q&A generation |
| `8_evaluation` | Final Hit@k / MRR evaluation |

---

## License

Licensed under the [Apache License 2.0](LICENSE).