# Roadmap

This document outlines completed milestones, the refactor backlog, future UI/architecture roadmap items, and historical benchmark baselines.

---

## 1. Completed Milestones

The following capabilities have been fully implemented and verified in the codebase:

- **Integrated Document Viewer & Unit Inspector:** Complete inline modal (`DocumentModal.jsx`) and dedicated route (`/documents/:fileId`) enabling users to browse ingested files and jump directly to cited rows, lines, or pages (`src/api/routes/documents.py`).
- **Multi-Format Ingestion & Addressable Units:** Native parsing and 1-based indexing for `.csv`, `.json`, `.jsonl`, `.txt`, `.md`, and `.pdf` files (`src/ingestion/units.py`), with dataset key and source URL separation.
- **Batched Ingestion with Text Deduplication:** `build_vectorstore` embeds duplicate text once, maps vectors back to duplicate rows, uses divide-and-conquer fault tolerance, and writes incrementally to pgvector (`src/vectorstore/store.py`).
- **Retrieval Impact Check (A/B Grounding Comparison):** `POST /api/benchmark/compare` runs grounded vs direct LLM queries side-by-side without polluting query history (`src/api/routes/benchmark.py`, `src/rag/graph.py`).
- **Custom Benchmark Test File Management:** Complete API and UI workflows to upload, validate, inspect, list, and delete custom evaluation test CSVs (`src/benchmark/files.py`).
- **Model Deletion & Registry Management:** Delete downloaded Ollama chat models and Hugging Face cross-encoder rerankers from disk via API and Settings UI (`src/api/routes/models.py`, `src/rag/model_catalog.py`).
- **Shared Database Connection Pooling & Health Pre-Ping:** Centralized pooled SQLAlchemy engines via `get_engine(url, init)` with `pool_pre_ping=True` and connection recycling (`src/db/engine.py`).
- **Route-Based Frontend Code Splitting:** Code-split React router with `React.lazy` and `Suspense`, dropping initial bundle transfer size by ~70% (`ui/src/App.jsx`).
- **Benchmark Runner Production Relocation:** Moved `runner.py` and `cache.py` to `src/benchmark/` to eliminate production API dependencies on the test tree (`src/benchmark/runner.py`, `src/api/routes/benchmark.py`).
- **Centralized Settings Dataclass:** Unified environment configuration into a frozen `Settings` dataclass via `get_settings()` in `src/config.py`.
- **Decoupled Model Registry & Modular Graph Nodes:** Decomposed monolithic nodes into `src/rag/models.py`, `src/rag/thinking.py`, `src/rag/retrieve.py`, `src/rag/grade.py`, `src/rag/generate.py`, and `src/rag/model_policy.py`.
- **Bounded Background Worker Pool:** Protected system resources by executing background tasks through a managed `ThreadPoolExecutor` in `src/jobs/runner.py`.
- **Buffered Log Inspection Endpoint:** Added `GET /api/metrics/logs` exposing structured ring buffer logs (`src/api/routes/metrics.py`).
- **Multi-Step Deletion Boundary:** Wrapped file, vector chunk, and DB record removals in safe error boundaries (`src/api/routes/ingest.py`).

---

## 2. Refactor Backlog

- **Ingest Upload Orchestration:** Move file upload domain logic (streaming hash, deduplication check, parsing validation, provenance recording) out of `src/api/routes/ingest.py` and into `src/ingestion/uploads.py`.
- **Split `src/rag/history.py`:** Separate table definition, migrations, DAO, and row serialization.
- **Eliminate Import-Time Side Effects:** Build the LangGraph lazily, move `load_dotenv()` to entrypoints, scope `csv.field_size_limit` inside loaders, and ensure `create_app()` acts as an ASGI factory.
- **Backend Guard on Embed Model Deletion:** `DELETE /api/models/{model}` currently blocks deleting the active embed model only in the UI (`ModelRow`'s `deletable` flag) -- add the same check server-side in `model_catalog.delete_model` / the route so a direct API call can't corrupt the vectorstore's embedding dimensions.

---

## 3. UI / UX Backlog

- **Retrieval Inspector:** Add a dedicated visual panel illustrating the retrieval candidate funnel (dense + lexical hits, RRF rank fusion, cross-encoder rerank ordering, and grading cutoffs).
- **Closed-Loop Evaluation:** Add thumbs up/down feedback on answers in the Ask view, persisting ratings to `query_history` and optionally appending difficult questions to benchmark test suites.
- **Prompt Budget Warning:** Dynamically calculate prompt token consumption against model context limits (e.g. `QWEN3_NUM_CTX=3072`) and display visual warnings in the UI before generation starts.
- **Multi-Turn Conversations:** Implement a `rewrite` graph node to condense conversational chat history into standalone retrieval queries, persisting thread session IDs.
- **UI Polish:** Drag-and-drop dropzone improvements for all supported file types, pre-ingest CSV dry run preview, and persistent metric ring buffers for charts.

---

## 4. Future Ideas & Epics

- **Hardware Auto-Tuning:** Probe host specifications (VRAM, system RAM, CPU cores) to automatically recommend the optimal local model and batching parameters.
- **Multi-Tenant / Enterprise Packaging:** Extend the local single-user architecture to support authentication, per-tenant vectorstore isolation, and multi-user access control.
- **Opt-In Hosted Model Providers:** Add optional adapters for Anthropic, OpenAI, or Gemini via secure local environment API keys, preserving zero-data-egress as the default while allowing cloud LLM generation when explicitly enabled.

---

## 5. Benchmark Baseline & Measurements

The following baseline was measured on 120 labeled questions with `llama3.2:3b` and `nomic-embed-text` at `k=5`, comparing dense-only vs hybrid retrieval:

| Question set | Metric | Dense (0.6 / 0.8) | Dense (0.48 / 0.9) | **Hybrid (0.56 / 0.9)** |
|---|---|---|---|---|
| single-passage (n=40) | recall@5 | 0.80 | 0.80 | **0.88** |
| | mean overlap | 0.16 | 0.39 | **0.43** |
| | pass rate | 0.20 | 0.47 | **0.57** |
| multi-passage (n=40) | recall@5 | 0.80 | 0.80 | **0.82** |
| | mean overlap | 0.20 | 0.31 | **0.30** |
| | pass rate | 0.25 | 0.42 | **0.35** |
| no-answer (n=40) | correct refusal | 0.62 | 0.20 | **0.35** |
| **All 120 Questions** | **Combined Right** | **0.357** | **0.363** | **0.423** |

*Analysis:* Hybrid retrieval combined with cross-encoder reranking and calibrated cutoffs raised overall combined accuracy to **42.3%**.
