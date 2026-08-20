# Architecture

A local-only retrieval-augmented generation system. Nothing leaves the machine: Postgres
(pgvector) stores the chunks and their embeddings, Ollama serves both the embedding model and
the chat model, and a FastAPI service wires them together behind a React dashboard.

This document is the map. It explains what each layer owns, how a query and an ingest actually
flow through the code, and where to put new code so it lands in the right place.

---

## 1. The four layers

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

**The rule that keeps this honest: arrows only point downward.** `src/vectorstore` must never
import `src/rag`; `src/rag` must never import `src/api`. If you need something from a layer
above, the dependency is inverted — pass it in as an argument.

---

## 2. Answering a query

`ask()` and `ask_stream()` in `src/rag/graph.py` are the two entrypoints. Both walk the same
three nodes; the streaming one steps them by hand so it can emit stage boundaries and tokens.

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

Two design decisions worth knowing before you change anything here:

- **There is no retry edge.** Retrieval is deterministic and returns results sorted by
  descending similarity, so re-running the same query can never surface a chunk that clears
  the grader when the top results did not. A retry loop only becomes useful alongside query
  rewriting.
- **Citations are computed, not generated.** Small local models do not reliably follow
  inline-citation instructions, so `src/rag/citations.py` derives sources deterministically
  from the chunks that survived grading.

---

## 3. Ingesting documents

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

**The invariant that matters most: the embedding model used at ingest must match the one used
at query time.** A mismatch does not raise — it silently returns garbage neighbours. Treat the
embedding model as baked into the index, not as a runtime setting.

---

## 4. Configuration lifetimes

Not all configuration is the same kind of thing, and conflating the three is the most common
source of confusion in this codebase.

| Lifetime | Examples | Changing it means |
|---|---|---|
| **Baked into the index** | chunk size, chunk overlap, embedding model | Re-ingesting the entire corpus |
| **Live per query** | `k`, relevance floor/ratio, chat model, citations on/off | Takes effect on the next query |
| **Server-owned** | `DATABASE_URL`, `OLLAMA_BASE_URL` | Restart; never exposed to the browser |

---

## 5. Where new code goes

| You are adding… | It belongs in |
|---|---|
| A new document format | `src/ingestion/loaders.py` |
| A new chunking strategy | `src/ingestion/splitter.py` (register it in `SPLITTERS`) |
| A new retrieval signal | `src/vectorstore/`, fused in `hybrid.py` |
| A change to how answers are produced | `src/rag/nodes.py` + `src/rag/prompts.py` |
| A new endpoint | A router in `src/api/routes/`, schemas in `src/api/schemas.py` |
| Anything long-running | A job kind on `src/jobs/runner.py`, never inline in a route |
| A new dashboard screen | `ui/src/views/`, wired into `ui/src/App.jsx` |

A route function should read as HTTP mapping and nothing else: parse, call one domain function,
translate errors to status codes. When a route body starts orchestrating storage, the logic has
escaped its layer.
