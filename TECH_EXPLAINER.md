# Technical Explainer: RAG Engine

This document details the internal architecture, mathematical mechanisms, data structures, and engineering decisions of the local retrieval-augmented generation engine.

---

## 1. System Architecture & Layer Hierarchy

The engine follows a strict four-layer architecture with unidirectional downward dependencies:

```
[ L4: Interface Layer ]
   ├── ui/ (React 19 + Vite SPA, Tailwind CSS)
   └── CLI Entrypoints (python -m src.rag.graph, python -m src.ingestion.pipeline)
           │
           ▼
[ L3: HTTP & Execution Layer ]
   ├── src/api/ (FastAPI routers, SSE endpoints, Pydantic schemas)
   └── src/jobs/ (Thread-based JobRunner, cooperative cancellation, progress hooks)
           │
           ▼
[ L2: Domain Layer ]
   ├── src/rag/ (LangGraph workflow, nodes, prompts, citations, history, model catalog)
   ├── src/ingestion/ (Units parser, multi-format loaders, splitters, pipeline, file provenance)
   └── src/benchmark/ (Benchmark suite file inspection, custom test uploads)
           │
           ▼
[ L1: Infrastructure Layer ]
   ├── src/vectorstore/ (PGVector store, tsvector lexical search, RRF hybrid fusion, cross-encoder rerank)
   ├── src/observability/ (Span tracing, structured JSON logging, NVML GPU/host metrics)
   ├── src/db/ (SQLAlchemy pooled engine cache)
   └── src/config.py (Environment variable parsing)
```

**Backing Storage & Compute Services:**
- **Postgres 16 + pgvector:** Vector embeddings, full-text tsvector search index (GIN), query history (JSONB), and file provenance metadata.
- **Ollama:** Local LLM inference server providing chat models (`ChatOllama`) and embedding models (`OllamaEmbeddings`).
- **HuggingFace Hub / Local PyTorch:** Cross-encoder reranker models running on CPU/CUDA via `sentence_transformers`.

---

## 2. Hardware Budgeting & Constraint Engineering

The system is designed and calibrated for resource-constrained environments (baseline: **4GB VRAM GPU**, e.g., NVIDIA GTX 1050):

1. **VRAM Partitioning:**
   - **Generation LLM:** Allocates ~2.0–2.6 GB VRAM for small parameter models (`llama3.2:3b`, `qwen2.5:3b`, `gemma2:2b`).
   - **Context Window Limits:** Context is configured via `RAG_NUM_CTX=8192` to prevent KV-cache spilling into system RAM / CPU execution (which causes 400s+ inference stalls).
   - **Ollama Slot Limits:** `OLLAMA_NUM_PARALLEL=1` prevents multi-slot VRAM pre-allocation crashes.
2. **CPU Offloading for Cross-Encoder Reranking:**
   - `cross-encoder/ms-marco-MiniLM-L-6-v2` runs on **CPU** by default (`RAG_RERANK_DEVICE=cpu`).
   - Because generation occupies GPU VRAM while leaving the CPU idle, CPU execution saves GPU VRAM entirely for generation.
   - At `FETCH_K=20` and `MAX_LENGTH=512`, CPU inference takes ~2.4s across 8 CPU threads (`torch.set_num_threads`).
3. **Query-Path Vectorstore Embedding CPU Offload:**
   - `load_vectorstore()` configures `OllamaEmbeddings(num_gpu=0)` during query time, ensuring query vectorization does not compete with generation for GPU memory.
4. **Ingest-Path Vectorstore Embedding GPU Acceleration:**
   - `build_vectorstore()` uses `OllamaEmbeddings(num_gpu=999)` to accelerate offline batch document embedding.

---

## 3. The Query & Retrieval Pipeline

Query execution is orchestrated by `src/rag/graph.py` via an explicit LangGraph state machine:

```
User Query + Model
       │
       ▼
[ history.start() ] ── (Records pending status in DB)
       │
       ▼
[ retrieve_node ]
   ├── Hybrid Retrieval (Dense k-NN + Lexical tsvector)
   ├── Reciprocal Rank Fusion (RRF k=60)
   └── Cross-Encoder Reranking (Squashed Logits -> [0, 1])
       │
       ▼
[ grade_node ]
   ├── Drop empty/whitespace chunks
   ├── Filter: Dense score >= max(RELEVANCE_FLOOR, max(dense_scores) * RELEVANCE_RATIO)
   └── Lexical hits preserved
       │
   ┌───┴────────────────────────┐
[Pass]                        [Fail]
   │                            │
   ▼                            ▼
[ generate_node ]         [ _refusal() ]
   ├── Prompt formatting        └── Return REFUSAL_ANSWER
   ├── ChatOllama invocation        (No hallucination/sources)
   ├── Think block stripping
   ├── Deterministic citations
   └── Confidence calculation
       │
       ▼
[ history.complete() ] ── (Stores answer, citations, confidence, latency)
```

### 3.1. Retrieval Subsystem (`src/vectorstore/`)

1. **Dense k-NN Retrieval (`store.py`):**
   - Query text is vectorized via `nomic-embed-text` (768 dimensions).
   - Distance metric: Cosine distance over `langchain_pg_embedding`.
   - Returns top candidates with cosine similarity scores.
2. **Lexical Full-Text Retrieval (`lexical.py`):**
   - PostgreSQL generated stored column `doc_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', document)) STORED`.
   - Indexed via `GIN (doc_tsv)`.
   - Query parsing uses `websearch_to_tsquery(:config, :query)`, ranked using `ts_rank_cd(e.doc_tsv, q)`.
3. **Reciprocal Rank Fusion (`hybrid.py`):**
   - Rank-based score fusion across dense and lexical lists:
     $$RRF(d) = \sum_{m \in M} \frac{1}{k_{RRF} + \text{rank}_m(d)}$$
     where $k_{RRF} = 60$.
   - Fused candidates break ties on lexical score first (exact match preference), then dense relevance score.
4. **Cross-Encoder Reranking (`rerank.py`):**
   - Evaluates $(query, passage)$ pairs simultaneously in a full transformer forward pass.
   - Raw logits are monotonically transformed onto $[0, 1]$ via standard logistic sigmoid:
     $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   - Candidate list is sorted by rerank score and truncated to `RETRIEVE_K=5`.

### 3.2. Grading and Filtering (`src/rag/nodes.py`)

Retrieved chunks undergo dual cutoff filtering:
- **Absolute Floor:** `RELEVANCE_FLOOR = 0.56` (drops irrelevant background noise).
- **Relative Ratio:** `RELEVANCE_RATIO = 0.9` (drops chunks weaker than 90% of the top chunk's score).
- **Cutoff Formula:**
  $$\text{Cutoff} = \max(\text{RELEVANCE\_FLOOR}, \max(\text{dense\_scores}) \times \text{RELEVANCE\_RATIO})$$
- **Lexical Hits Invariant:** Lexical exact-match hits bypass the dense score cutoff.
- **Empty Survival:** If no chunks survive grading, execution immediately routes to `_refusal()`, returning `REFUSAL_ANSWER` (*"I don't have enough context to answer that question."*) without LLM invocation.

### 3.3. Grounded Generation (`src/rag/prompts.py`, `src/rag/nodes.py`)

- **Prompt Construction:** Graded chunk texts are concatenated separated by double newlines (`format_context`). Headers/file paths are omitted from prompt text to prevent prompt pollution and token waste.
- **Prompt Instruction:** Explicitly instructs plain prose without hallucinated inline source tags.
- **Confidence Metric:**
  - `confidence_of(docs)` takes top-1 squashed cross-encoder score (or top-1 dense cosine similarity if reranking is disabled).

### 3.4. Deterministic Citation Attribution (`src/rag/citations.py`)

Rather than relying on LLM self-reporting, citations are computed deterministically in code from surviving graded documents:
- Each citation structure contains: `file_id`, `filename`, `unit_kind` (row/line/page), `unit_index`, `label`, and optional source `url`.
- Deduplicated by identity tuple `(file_id, unit_index)`.

### 3.5. Real-Time Streaming (`ask_stream` & `POST /api/query/stream`)

1. `ask_stream()` yields generator events:
   - `{"type": "stage", "stage": "retrieve"}`
   - `{"type": "stage", "stage": "grade", "detail": {"retrieved": N}}`
   - `{"type": "stage", "stage": "generate", "detail": {"retrieved": N, "kept": M}}`
   - `{"type": "token", "text": "..."}`
   - `{"type": "done", "answer": "...", "citations": [...], "confidence": 0.85, ...}`
2. **Async Bridge (`src/api/routes/query.py`):**
   - Synchronous graph operations run on a background worker thread.
   - Events are passed via `asyncio.Queue` to FastAPI's `EventSourceResponse`.
   - Client disconnection triggers cooperative thread cancellation and closes the Ollama stream generator immediately.

---

## 4. Ingestion Engine & Addressable Units

### 4.1. Addressable Units (`src/ingestion/units.py`)

Documents are broken down into stable, addressable units that match the user's mental model:
| File Extension | Unit Kind | Indexing Method | Text Extraction |
|---|---|---|---|
| `.csv` | `row` | 1-based (header excluded) | Key-value pairs per row (excluding ID/URL) |
| `.json` | `row` | 1-based array item | Flattened JSON key-value string |
| `.jsonl` | `line` | 1-based physical line | Parsed JSON record |
| `.txt`, `.md` | `line` | 1-based line of paragraph start | Paragraph blocks (`\n\n` separator) |
| `.pdf` | `page` | 1-based page number | `pypdf` page text extraction |

- **Key/URL Exclusion:** Identifier columns (`passage_id`, `document_index`, `id`) and URL columns (`source_url`, `url`, `link`) are separated from embedded text and stored purely in metadata, preserving chunk token budget.

### 4.2. Chunking & Splitting (`src/ingestion/splitter.py`)

- **Recursive Splitter (`recursive`, default):** Uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with `chunk_size=1000` chars, `chunk_overlap=150` chars.
- **Token Splitter (`token`):** Uses `TokenTextSplitter` for fixed-token chunks.

### 4.3. Batched Vectorstore Building & Deduplication (`src/vectorstore/store.py`)

1. **Text Deduplication:** Identical passage texts across dataset rows are mapped via `positions: dict[str, list[int]]`. Distinct texts are sent to Ollama once, and resulting vectors are mapped back to all duplicate rows.
2. **Batch Embedding:** Embeds in batches (`RAG_EMBED_BATCH=512`/`1024`) with `ThreadPoolExecutor`.
3. **Fault Tolerance:** `_embed_batch()` employs exponential backoff retry. If failures persist, it halves the batch recursively (divide-and-conquer) to isolate corrupted or oversized inputs.
4. **Incremental Storage:** Each batch is committed to PGVector immediately.
5. **GIN Index Backfill:** `ensure_index()` executes `ALTER TABLE langchain_pg_embedding ADD COLUMN doc_tsv ...` and creates the GIN index if missing.

---

## 5. Storage Schemas & Database Engine Pool

SQLAlchemy connection pooling is centralized in `src/db/engine.py` using `get_engine(url, init)` to eliminate duplicate engine creation and pool leaks.

### 5.1. Ingested Files Table (`ingested_files`)
```sql
CREATE TABLE ingested_files (
    id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    sha256 VARCHAR(64) NOT NULL UNIQUE,
    size_bytes INTEGER NOT NULL,
    documents INTEGER,
    chunk_ids JSONB
);
CREATE INDEX ix_ingested_files_sha256 ON ingested_files (sha256);
```

### 5.2. Query History Table (`query_history`)
```sql
CREATE TABLE query_history (
    id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    query TEXT NOT NULL,
    answer TEXT,
    sources JSONB,
    citations JSONB,
    refused BOOLEAN,
    confidence DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION,
    rerank_ms DOUBLE PRECISION,
    generate_ms DOUBLE PRECISION,
    model VARCHAR(128),
    status VARCHAR(16) NOT NULL DEFAULT 'done' -- 'pending', 'done', 'error', 'cancelled'
);
CREATE INDEX ix_query_history_created_at ON query_history (created_at);
```

---

## 6. Background Jobs & Concurrency Model

Long-running tasks (file ingestion, benchmark suites, model downloads) are managed by `src/jobs/runner.py`:
- **Thread-per-job Execution:** Each job runs on a dedicated daemon thread.
- **Exclusive Job Mutex (`submit_exclusive`):** Enforces single active job for mutual exclusion domains (e.g. `ingest`).
- **Cooperative Cancellation:** Tasks poll `reporter.cancelled` / `cancel_event.is_set()` to cleanly interrupt execution and record partial results.
- **Thread Safety:** State updates and dictionary snapshots are protected by re-entrant thread locks.

---

## 7. Observability & System Telemetry

1. **Span Tracing (`src/observability/tracing.py`):**
   - Uses `contextvars` to track span hierarchy and timing across synchronous and asynchronous invocations.
   - Emits structured JSON events (`span`, `duration_ms`, `detail`).
2. **In-Process Log Ring Buffer (`src/observability/logging.py`):**
   - Thread-safe deque ring buffer (`RING_BUFFER_SIZE=1000`) for in-memory log tailing.
3. **Host & GPU Telemetry (`src/observability/sysmetrics.py`):**
   - Collects CPU per-core utilization, RAM/Swap metrics (`psutil`), disk usage, and disk I/O.
   - Direct NVML bindings via `pynvml` for NVIDIA GPU utilization, VRAM usage/capacity, temperature, and power consumption.
   - Handles NVML errors gracefully with a 30-second initialization cooldown.
   - Streamed to the UI via Server-Sent Events (`GET /api/metrics/stream`) at 1Hz.

---

## 8. Summary of Core Technical Decisions

| Decision | Rationale | Tradeoff |
|---|---|---|
| **Local-Only Inference (Ollama)** | Zero data egress, no per-token billing, complete data privacy. | Lower speed and reasoning ceiling compared to 70B+ cloud LLMs. |
| **Postgres + pgvector** | Unifies vector embeddings, relational metadata, and full-text search in one ACID engine. | Slightly higher memory footprint than specialized in-memory vector indexes (e.g., Faiss). |
| **Reciprocal Rank Fusion (RRF)** | Merges dense semantics with exact-keyword lexical recall without score calibration dependencies. | Requires dual retrieval queries per request. |
| **CPU Cross-Encoder Reranker** | Preserves all 4GB GPU VRAM for the generation LLM. | Adds ~1.0–2.4s CPU latency per query. |
| **Deterministic Code Citations** | Eliminates LLM hallucination of citations; guarantees cited units actually existed in context. | Attributed at document/unit level rather than per-sentence. |
| **Linear Graph (No Retry Loop)** | Deterministic retrieval cannot surface better candidates on identical retry. | Requires future query rewriting module before retry loops make sense. |
| **Batch Ingest Deduplication** | Reduces Ollama embedding work significantly on datasets with repeated passages. | Requires upfront passage hashing and position mapping. |
