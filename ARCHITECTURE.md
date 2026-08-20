# Architecture

A local-only retrieval-augmented generation system. Nothing leaves the machine: Postgres (pgvector) stores the chunks and their embeddings, Ollama serves both the embedding model and the chat model, and a FastAPI service wires them together behind a React dashboard.

## 1. Why it runs entirely locally

Every component — the LLM, the embedding model, the vector database — runs in a container on this machine. No document text and no query ever leaves the host, there is no per-token cost, and the whole stack comes up with one `docker compose up`.

The tradeoff is quality and speed: `llama3.2:3b` is a small model chosen to fit inside the 4GB of VRAM on a GTX 1050. Swapping to a hosted API later means changing one `ChatOllama(...)` construction.

## 2. The Pieces

- **Ollama:** Serves two models over HTTP: `llama3.2:3b` generates answers, and `nomic-embed-text` turns text into vectors. Uses GPU passthrough.
- **Postgres + pgvector:** Stores document chunks alongside vectors and answers nearest-neighbour queries. Also holds a `tsvector` column + GIN index for lexical search.
- **LangChain & LangGraph:** LangChain supplies the adapters (OllamaEmbeddings, PGVector, loaders). LangGraph orchestrates control flow as an explicit state graph (retrieve → grade → generate).

## 3. The Four Layers

```mermaid
flowchart TB
    subgraph L4["Interface"]
        UI["ui/ — React dashboard<br/>(Ask, Ingest, Benchmark, System, Settings)"]
        CLI["CLI entrypoints<br/>src.rag.graph · src.ingestion.pipeline"]
    end

    subgraph L3["HTTP"]
        API["src/api — routers, Pydantic schemas<br/>HTTP mapping only"]
        JOBS["src/jobs — background job runner<br/>thread-per-job, cooperative cancel"]
    end

    subgraph L2["Domain"]
        RAG["src/rag — graph, nodes, prompts,<br/>citations, history, model catalog"]
        ING["src/ingestion — loaders, splitter,<br/>pipeline, file provenance"]
    end

    subgraph L1["Infrastructure"]
        VS["src/vectorstore — pgvector, lexical,<br/>hybrid fusion, reranker"]
        OBS["src/observability — tracing,<br/>JSON logging, system metrics"]
        CFG["src/config.py — env access"]
    end

    PG[("Postgres + pgvector")]
    OLL[["Ollama<br/>embed + chat, GPU"]]

    UI --> API
    CLI --> RAG
    CLI --> ING
    API --> JOBS
    API --> RAG
    API --> ING
    JOBS --> ING
    RAG --> VS
    ING --> VS
    VS --> PG
    RAG --> OLL
    VS --> OLL
    RAG --> OBS
    API --> OBS

    classDef store fill:#1f6feb22,stroke:#1f6feb
    class PG,OLL store
```

**The rule that keeps this honest:** arrows only point downward.

## 4. Answering a Query

`ask()` and `ask_stream()` in `src/rag/graph.py` walk three nodes.

```mermaid
flowchart TD
    Q["query + model"] --> RESOLVE{"model in<br/>AVAILABLE_MODELS?"}
    RESOLVE -->|no| ERR["ValueError — no default model"]
    RESOLVE -->|yes| HIST[("history.start()<br/>write pending row")]

    HIST --> RETRIEVE["<b>retrieve_node</b>"]

    subgraph R["retrieve — src/rag/nodes.py"]
        direction TB
        HYB{"hybrid<br/>enabled?"}
        DENSE["dense k-NN<br/>pgvector, FETCH_K=20"]
        LEX["lexical search<br/>tsvector full-text"]
        RRF["RRF fusion<br/>src/vectorstore/hybrid.py"]
        RER{"rerank<br/>enabled?"}
        CE["cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2, CPU"]
        TOPK["top RETRIEVE_K = 5"]
        HYB -->|yes| DENSE --> RRF
        HYB -->|yes| LEX --> RRF
        HYB -->|no| DENSE
        RRF --> RER
        DENSE --> RER
        RER -->|yes| CE --> TOPK
        RER -->|no| TOPK
    end

    RETRIEVE --> HYB
    TOPK --> GRADE["<b>grade_node</b><br/>drop chunks below RELEVANCE_FLOOR 0.56<br/>and below RELEVANCE_RATIO 0.9 of the best"]

    GRADE --> KEPT{"any chunks<br/>survive?"}
    KEPT -->|no| REFUSE["REFUSAL_ANSWER<br/>no sources — a refusal, not a guess"]
    KEPT -->|yes| GEN["<b>generate_node</b><br/>prompt = template + context chunks<br/>ChatOllama, num_ctx per model"]

    GEN --> THINK{"thinking model?<br/>(qwen3*)"}
    THINK -->|yes| STRIP["strip &lt;think&gt; block<br/>append /no_think"]
    THINK -->|no| CITE
    STRIP --> CITE["citations — computed in code,<br/>not asked of the LLM"]

    REFUSE --> DONE
    CITE --> DONE[("history.complete()<br/>answer, sources, latency breakdown")]

    classDef bad fill:#f8514922,stroke:#f85149
    class ERR,REFUSE bad
```

- **There is no retry edge:** Retrieval is deterministic; re-running the same query returns the same results.
- **Citations are computed, not generated:** Small models don't reliably follow inline-citation rules, so `src/rag/citations.py` deterministically attributes sources from graded chunks.

## 5. Ingesting Documents

```mermaid
flowchart LR
    UP["POST /api/ingest<br/>multipart upload"] --> HASH["stream to disk,<br/>hash while writing"]
    HASH --> DEDUP{"hash already<br/>ingested?"}
    DEDUP -->|yes| SKIP["409 — already ingested"]
    DEDUP -->|no| PARSE{"CSV parses<br/>to usable rows?"}
    PARSE -->|no| REJECT["422 — UnusableCSV"]
    PARSE -->|yes| REC[("files.record()<br/>provenance row")]
    REC --> SUBMIT["runner.submit()<br/>background thread"]

    SUBMIT --> P1["load_documents<br/>src/ingestion/loaders.py"]
    P1 --> P2["split_documents<br/>1000 chars, 150 overlap"]
    P2 --> P3["build_vectorstore<br/>embed via Ollama, write to pgvector"]
    P3 --> P4["ensure_index<br/>tsvector column + GIN index"]
    P4 --> OK["job complete<br/>documents, chunks, chunk_ids"]

    classDef bad fill:#f8514922,stroke:#f85149
    class SKIP,REJECT bad
```

**Invariant:** The embedding model used at ingest must match the one used at query time. A mismatch silently returns garbage neighbours.

## 6. UI Architecture and Metrics

The UI is built with React/Vite and talks to the FastAPI backend. It provides views for Ask, Ingest, Benchmark, System, etc.

Since browsers don't expose host system metrics (CPU load, VRAM, GPU temps), these are collected server-side (`src/observability/sysmetrics.py`) using `psutil`, `pynvml`, and Docker APIs, and streamed via Server-Sent Events (SSE).

### Configuration Lifetimes

| Lifetime | Examples | Changing it means |
|---|---|---|
| **Baked into the index** | chunk size, overlap, embedding model | Re-ingesting the entire corpus |
| **Live per query** | `k`, relevance cutoffs, chat model | Takes effect on the next query |
| **Server-owned** | `DATABASE_URL`, `OLLAMA_BASE_URL` | Restart; never exposed to browser |

## 7. Where New Code Goes

| You are adding… | It belongs in |
|---|---|
| A new document format | `src/ingestion/loaders.py` |
| A new chunking strategy | `src/ingestion/splitter.py` (register it in `SPLITTERS`) |
| A new retrieval signal | `src/vectorstore/`, fused in `hybrid.py` |
| A change to how answers are produced | `src/rag/nodes.py` + `src/rag/prompts.py` |
| A new endpoint | A router in `src/api/routes/`, schemas in `src/api/schemas.py` |
| Anything long-running | A job kind on `src/jobs/runner.py`, never inline in a route |
| A new dashboard screen | `ui/src/views/`, wired into `ui/src/App.jsx` |
