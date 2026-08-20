# Roadmap

This document outlines the remaining work, refactor plans, future ideas, and baseline metrics for the RAG system.

## 1. Refactor Backlog

- **Settings Object:** Introduce a frozen `Settings` dataclass built once via `get_settings()` to replace scattered `os.environ` reads (which currently happen at import time).
- **Model Registry:** Move `AVAILABLE_MODELS`, `_num_ctx_for`, and the `ChatOllama` client cache into a new `rag/models.py`. Ensure vectorstore cache and LLM cache have separate locks.
- **Model Resolution Policy:** Consolidate `graph._resolve_model`, `query._validated_model`, and `benchmark.start_benchmark` checks into a single `resolve_model(name)` in `rag/model_policy.py`.
- **Ingest Orchestration:** Move the domain logic for ingest (streaming hash, deduplication, CSV pre-parse, provenance recording) out of `ingest.upload_and_ingest` and into `ingestion/uploads.py`.
- **Split `rag/nodes.py`:** Break into `rag/retrieve.py`, `rag/grade.py`, `rag/generate.py`, and `rag/thinking.py`.
- **Split `rag/history.py`:** Separate the table definition, migrations, DAO, and row mapping.
- **Benchmark Runner:** Move the benchmark runner out of `tests/` and into `src/benchmark/` to prevent production API dependency on test packages.
- **Remove Import-Time Side Effects:** Build the graph lazily, move `load_dotenv()` to the entrypoint, and keep `app = create_app()` behind an ASGI factory.
- **Other minor issues:** Pin `requirements.txt`, distinguish 4xx/5xx in `run_query`, add error handling to `history` and `files` paths, bound the daemon threads in `runner.submit`.

## 2. UI / UX Backlog

- **Rich Sources:** Expand `SourceList.jsx` to show chunk texts, highlighting, and a "dropped by grader" section to explain refusals. Needs a `Source` schema change.
- **Retrieval Inspector:** Add a panel showing the full retrieval funnel (all candidates, rerank ordering, pass/fail grading) using tracing data.
- **A/B Playground:** Two-column Ask variant to diff queries side-by-side. Requires threading a config object through `RAGState`.
- **Closed Eval Loop:** Thumbs up/down on answers. Write ratings to `query_history` and append expected answers to the benchmark datasets automatically.
- **Prompt Budget Warning:** Estimate tokens dynamically (e.g., QWEN3_NUM_CTX spilling to CPU) and show a warning in the UI if retrieved chunks exceed the context window.
- **Corpus Browser:** View chunks inside an ingested file via paginated lists, full-text search, and chunk-boundary visualizations.
- **Multi-Turn Conversations:** Add a `rewrite` node to condense follow-up queries using the chat history, and persist thread IDs.
- **Other UI Polish:** Drag-and-drop ingestion, pre-ingest CSV dry run, persistent ring buffer for system charts, reference lines on latency charts, and error boundaries.

## 3. Future Ideas & Epics

- **Auto-Suggest Models:** Probe the local machine specs (VRAM, RAM) and document characteristics to automatically recommend an optimal local model and an alternative hosted LLM.
- **SaaS / On-Premise Multi-User Packaging:** Turn the single-user local tool into a multi-tenant application (e.g., with Okta/SAML, per-tenant vectorstore isolation, and billing metrics).
- **Hosted Model Providers (Opt-in):** Support Anthropic, OpenAI, or Gemini via API keys for stronger answer generation (improving inline citations). Requires careful credential handling (keys never hit browser, redacted logs, env-only storage) and cost control mechanisms (spend caps, pre-flight estimates).

## 4. Benchmark Baseline & Measurements

The following baseline was measured on 120 labeled questions with `llama3.2:3b` and `nomic-embed-text` at `k=5`, comparing dense-only vs hybrid retrieval.

| Question set | metric | dense, 0.6/0.8 | dense, 0.48/0.9 | **hybrid, 0.56/0.9** |
|---|---|---|---|---|
| single-passage (n=40) | recall@5 | 0.80 | 0.80 | **0.88** |
| | mean overlap | 0.16 | 0.39 | **0.43** |
| | pass rate | 0.20 | 0.47 | **0.57** |
| multi-passage (n=40) | recall@5 | 0.80 | 0.80 | **0.82** |
| | mean overlap | 0.20 | 0.31 | **0.30** |
| | pass rate | 0.25 | 0.42 | **0.35** |
| no-answer (n=40) | correct refusal | 0.62 | 0.20 | **0.35** |
| **all 120** | **questions right** | **0.357** | **0.363** | **0.423** |

*Note:* Hybrid retrieval raised the ceiling on recall@5 and overall combined score. The correct-refusal rate dropped, which highlights that the LLM generation step is the weakest link (hallucination on marginal context) once the retrieval cutoffs are relaxed.
