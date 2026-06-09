# RepoMind — Codebase Question Answering with Retrieval-Augmented Generation

A production-grade RAG system that enables natural language querying of any GitHub repository.
RepoMind ingests source code, builds a hybrid semantic and keyword index, and streams grounded,
citation-backed answers using a multi-stage retrieval and generation pipeline.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Evaluation](#evaluation)

---

## Overview

RepoMind addresses the challenge of navigating and understanding large, unfamiliar codebases.
Traditional code search tools rely on exact keyword matching and require the user to already know
what they are looking for. RepoMind instead allows developers to ask questions in plain language —
"Where is authentication handled?" or "How does the retry logic work?" — and receive precise,
contextually grounded answers with direct source references.

The system operates in two phases. During **indexing**, a repository is cloned, its source files
are parsed into semantic units (functions, classes, methods) using language-aware AST analysis,
and each unit is embedded and stored in a hybrid vector and keyword index. During **querying**, a
multi-stage retrieval pipeline fetches the most relevant code chunks, an LLM evaluates their
quality and retries if needed, and a streaming answer with citations is returned to the user.

---

## System Architecture

```
User Query
    |
    v
 LangGraph State Machine
    |
    +---> Retrieve Node
    |         |
    |         +---> Multi-Query Fusion (diverse query reformulations via Groq)
    |         |         |
    |         |         v
    |         +---> Dense Search (Cohere embed-english-v3.0 + Qdrant)  [per query]
    |         +---> Sparse Search (BM25 over Redis-cached index)       [per query]
    |         +---> RRF Fusion (Reciprocal Rank Fusion over all result sets)
    |         +---> Cohere Reranking (rerank-english-v3.0)
    |
    +---> Reflect Node (LLM relevance check, up to 2 retries)
    |
    v
 Answer Streaming (Groq llama-3.3-70b-versatile via SSE)
    |
    v
 Citations + Source References returned to frontend
```

**Indexing Pipeline (Celery Background Worker)**

```
GitHub URL --> Git Clone (shallow) --> File Walk + Language Detection
    --> AST-based Semantic Chunking (tree-sitter, 13 languages)
    --> Batch Embedding (Cohere embed-english-v3.0, batch_size=96)
    --> Qdrant Upsert (dense vectors, 1024-dim, cosine distance)
    --> BM25 Index Build + Redis Cache
    --> Status: ready
```

---

## Key Features

### Hybrid Retrieval with RRF Fusion
Combines dense semantic search (Cohere embeddings in Qdrant) with sparse keyword search
(BM25 over a Redis-cached token corpus). Every query reformulation produces its own dense and
sparse result lists, all of which are fused using Reciprocal Rank Fusion (k=60, Cormack et al.
2009), capturing results that neither approach — nor any single phrasing — would surface alone.

### Multi-Query Fusion — Diverse Query Expansion
Rather than searching with the raw user query alone, RepoMind first prompts Groq — in a single
LLM call — to generate several semantically diverse reformulations of the question. Each
reformulation (plus the original) is retrieved independently and the result sets are combined via
RRF before reranking. Expanding the question into multiple angles substantially improves recall
for technical questions without multiplying LLM usage. (This replaces the earlier HyDE approach,
which embedded a single hypothetical code snippet; see `app/query/fusion.py`.)

### Language-Aware Semantic Chunking
Source files are parsed using tree-sitter grammars for 13 programming languages. Chunks are
extracted at the function, method, and class level based on AST node types — not arbitrary
fixed-size windows — preserving semantic boundaries and reducing context fragmentation.

### Cohere Reranking
After RRF fusion, the top 20 candidates are passed to Cohere's cross-encoder reranking model
(`rerank-english-v3.0`), which scores each chunk against the original query independently. The
top 5 reranked results are passed to the generation step.

### LangGraph Retrieval-Reflection Loop
Retrieval is orchestrated as a LangGraph state machine. A reflection node uses Groq to evaluate
whether the retrieved chunks are genuinely relevant to the query. If not, the graph retries
retrieval with an adjusted strategy, up to a maximum of two retry attempts.

### Streaming Answer Generation
Answers are streamed token-by-token to the frontend via Server-Sent Events (SSE) using
incremental JSON parsing. Users receive live output without waiting for the full response to
complete, maintaining responsiveness regardless of answer length.

### Structured Citations
The LLM is prompted to return a structured JSON payload containing both the answer text and a
citations array. Each citation references a specific file path and line range. The frontend
renders these as interactive cards and includes a collapsible sources panel.

### Multi-Turn Conversation Memory
Conversation history is maintained per session in Redis (last 6 messages, 1-hour TTL). Each
query carries prior context, enabling follow-up questions and iterative exploration without
repeating context.

### Observability with Langfuse
Every query pipeline execution is traced as a Langfuse span, capturing inputs, outputs, and
timing across retrieval and generation steps for debugging and performance analysis.

### RAGAS Evaluation Dashboard
An integrated evaluation dashboard runs RAGAS metrics (Faithfulness, Answer Relevancy, Context
Recall) against a curated test set using Groq as the LLM judge. Results are visualised in an
interactive HTML dashboard with per-question breakdowns and aggregate scores.

---

## Technologies Used

### Backend

| Component | Technology |
|---|---|
| Web Framework | FastAPI 0.111+ |
| ASGI Server | Uvicorn 0.30+ |
| Task Queue | Celery 5.4+ |
| ORM | SQLAlchemy 2.0+ (async) |
| Database Driver | asyncpg 0.29+ |
| Schema Migrations | Alembic 1.13+ |
| RAG Orchestration | LangGraph 1.2+ |
| Observability | Langfuse 4.6.1+ |

### AI / ML Services

| Component | Technology |
|---|---|
| Embeddings | Cohere embed-english-v3.0 (1024-dim) |
| Reranking | Cohere rerank-english-v3.0 |
| LLM (generation, query fusion, reflection) | Groq llama-3.3-70b-versatile |
| BM25 Sparse Retrieval | rank-bm25 0.2.2 |

### Code Parsing

| Language | Parser |
|---|---|
| Python, JavaScript, TypeScript, Java | tree-sitter with official grammars |
| Go, Rust, C++, C, Ruby | tree-sitter with official grammars |
| C#, PHP, Scala | tree-sitter with official grammars |

### Infrastructure

| Component | Technology |
|---|---|
| Vector Database | Qdrant 1.18+ |
| Relational Database | PostgreSQL 16 |
| Cache / Message Broker | Redis 7 |
| Repository Access | GitPython 3.1+, PyGithub 2.3+ |

### Frontend

| Component | Technology |
|---|---|
| UI | Vanilla HTML / CSS / JavaScript (no framework) |
| Syntax Highlighting | highlight.js 11.9 |
| Streaming | Server-Sent Events (SSE) |
| Fonts | Inter, JetBrains Mono (Google Fonts) |

### Evaluation

| Component | Technology |
|---|---|
| RAG Evaluation | RAGAS 0.4.3 |
| Testing | pytest 8.0+, pytest-asyncio 0.23+ |

---

## Project Structure

```
CodeBase-Q-A-with-RAG/
|
+-- app/
|   +-- main.py                  # FastAPI application, lifespan, API endpoints
|   +-- ingestion/
|   |   +-- cloner.py            # GitHub repository cloning and file walking
|   |   +-- chunker.py           # AST-based semantic chunking (tree-sitter)
|   |   +-- embedder.py          # Batch embedding via Cohere API
|   |   +-- indexer.py           # Qdrant upsert and BM25 index construction
|   +-- query/
|   |   +-- graph.py             # LangGraph retrieval-reflection state machine
|   |   +-- retriever.py         # Hybrid search (multi-query + dense + BM25 + RRF + rerank)
|   |   +-- fusion.py            # Multi-Query Fusion query expansion via Groq
|   |   +-- hyde.py              # (retired) Hypothetical Document Expansion — kept for reference
|   |   +-- answerer.py          # Streaming answer generation with citations
|   +-- workers/
|   |   +-- celery_app.py        # Celery application configuration
|   |   +-- tasks.py             # index_repo_task background job
|   +-- api/
|   |   +-- eval.py              # Evaluation API routes
|   +-- utils/
|   |   +-- tracing.py           # Langfuse singleton
|   |   +-- memory.py            # Redis-backed conversation history
|   +-- db/
|       +-- models.py            # SQLAlchemy ORM models (Repo, Chunk)
|       +-- database.py          # Async engine and session factory
|
+-- migrations/                  # Alembic migration scripts
+-- eval/
|   +-- run_ragas.py             # RAGAS evaluation runner
|   +-- results.json             # Latest evaluation output
+-- index.html                   # Single-page frontend application
+-- eval.html                    # Evaluation dashboard
+-- docker-compose.yml           # PostgreSQL, Redis, Qdrant service definitions
+-- pyproject.toml               # Project metadata and dependencies
+-- .env                         # Environment variables (not committed)
```

---

### API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/repos` | Submit a GitHub URL for indexing |
| GET | `/repos/{id}/status` | Poll indexing status |
| POST | `/repos/{id}/reindex` | Reset and re-trigger indexing |
| POST | `/query` | Stream an answer (SSE) for a query against an indexed repo |
| GET | `/eval` | Serve the evaluation dashboard |

---

## Evaluation

RepoMind includes an integrated RAGAS evaluation pipeline. To run it:

1. Navigate to `http://127.0.0.1:8000/eval`.
2. Click "Run Evaluation" to execute the test suite against the live pipeline.
3. Results are displayed as gauge metrics for Faithfulness, Answer Relevancy, and Context Recall,
   with a per-question breakdown table.

Alternatively, run the evaluation script directly:

```bash
python eval/run_ragas.py
```

Results are saved to `eval/results.json`.

---

