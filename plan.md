# RAG System Implementation Plan (LangChain + LangGraph)

## Overview

Build retrieval-augmented generation pipeline. User query retrieves relevant document chunks from vector store, chunks + query feed LLM, LLM generates grounded answer. LangGraph orchestrates control flow (retrieve → grade → generate → maybe retry), LangChain handles loaders/splitters/embeddings/vector store integrations.

**Target architecture:**

```
Query → [Retrieve Node] → [Grade/Filter Node] → [Generate Node] → Answer
              ↑                    │
              └──── (retry if docs weak) ────┘
```

---

## Phase 1: Environment & Dependencies

- [ ] Create virtual environment (`python -m venv .venv`)
- [ ] Install core packages:
  ```
  langchain
  langgraph
  langchain-community
  langchain-openai       # or langchain-anthropic
  langchain-postgres      # pgvector integration
  psycopg[binary]
  pgvector
  tiktoken                # token counting
  python-dotenv
  pypdf
  ```
- [ ] Set env vars in `.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DATABASE_URL` (Postgres connection string)
- [ ] Project structure:
  ```
  rag/
  ├── src/
  │   ├── loaders.py
  │   ├── splitter.py
  │   ├── vectorstore.py
  │   ├── graph.py
  │   ├── nodes.py
  │   └── prompts.py
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── tests/
  ├── .env
  ├── .env.example
  ├── Dockerfile
  ├── docker-compose.yml
  ├── .dockerignore
  ├── requirements.txt
  ├── plan.md
  └── explainer.md
  ```
- [ ] Verify installs: `python -c "import langgraph, langchain; print('ok')"`

**Done marker:** environment activates, imports succeed, API key loads from `.env`.

### Docker (local run)

- [ ] `Dockerfile` — Python 3.11-slim app image, installs `requirements.txt`, runs `src/graph.py`
- [ ] `docker-compose.yml` — two services:
  - `db` — `pgvector/pgvector:pg16` image, exposes 5432, persists via named volume, healthcheck via `pg_isready`
  - `app` — builds from `Dockerfile`, waits on `db` healthy, reads `.env`, `DATABASE_URL` points at `db` service (not localhost)
- [ ] `.dockerignore` — excludes `.venv`, `.git`, `.env`, `__pycache__`
- [ ] Run: `docker compose up --build`
- [ ] Verify: `docker compose exec db psql -U rag -d rag_db -c "CREATE EXTENSION IF NOT EXISTS vector;"` (image ships extension, just needs enabling per DB)
- [ ] Confirm app container connects to db container and ingest/query works end-to-end

**Done marker:** `docker compose up` brings up Postgres+pgvector and app with zero manual local Postgres install.

---

## Phase 2: Document Loading

- [ ] Pick loader(s) matching source data type:
  - `PyPDFLoader` — PDFs
  - `WebBaseLoader` — web pages
  - `TextLoader` — plain text
  - `DirectoryLoader` — batch load folder
- [ ] Implement `load_documents(path) -> list[Document]` in `src/loaders.py`
- [ ] Handle load failures gracefully (skip corrupt files, log which failed)
- [ ] Sanity check: print doc count, sample first doc's `page_content` and `metadata`

**Gotcha:** preserve `metadata` (source filename, page number) — needed later for citations.

---

## Phase 3: Text Splitting / Chunking

- [ ] Use `RecursiveCharacterTextSplitter` (default, splits on paragraph/sentence boundaries)
- [ ] Config:
  - `chunk_size=1000` (tokens or chars — decide, tiktoken-based counter recommended)
  - `chunk_overlap=100–200` (10–20% of chunk size)
- [ ] Implement `split_documents(docs) -> list[Document]` in `src/splitter.py`
- [ ] Validate: no chunk exceeds embedding model's max input tokens
- [ ] Spot-check 5–10 chunks manually — do they read as coherent units, or cut mid-sentence?

**Gotcha:** too small → loses context across chunk boundary. Too large → dilutes relevance signal, wastes LLM context window. Iterate empirically against real queries.

---

## Phase 4: Embeddings & Vector Store

- [ ] Choose embedding model:
  - `text-embedding-3-small` (OpenAI, cheap, 1536 dim) — default
  - `text-embedding-3-large` — higher quality, costlier
  - HuggingFace local model — no API cost, needs GPU/CPU tradeoff
- [ ] Vector store: **PostgreSQL + pgvector** (self-hosted or managed e.g. Supabase/RDS)
- [ ] Ensure Postgres has `pgvector` extension available (`CREATE EXTENSION IF NOT EXISTS vector;`)
- [ ] Implement `src/vectorstore.py`:
  ```python
  import os
  from langchain_openai import OpenAIEmbeddings
  from langchain_postgres import PGVector

  COLLECTION_NAME = "rag_chunks"

  def build_vectorstore(chunks, connection=None):
      connection = connection or os.environ["DATABASE_URL"]
      embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
      return PGVector.from_documents(
          documents=chunks,
          embedding=embeddings,
          collection_name=COLLECTION_NAME,
          connection=connection,
      )

  def load_vectorstore(connection=None):
      connection = connection or os.environ["DATABASE_URL"]
      embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
      return PGVector(
          embeddings=embeddings,
          collection_name=COLLECTION_NAME,
          connection=connection,
      )
  ```
- [ ] `DATABASE_URL` in `.env`: `postgresql+psycopg://user:password@localhost:5432/rag_db`
- [ ] Ingest all chunks (writes rows to Postgres, index built via pgvector)
- [ ] Test retrieval manually: `vectorstore.similarity_search("test query", k=5)` — inspect results for relevance
- [ ] (production) tune pgvector index — `ivfflat` or `hnsw` — once row count grows past a few thousand

**Gotcha:** embedding model used at ingest time MUST match model used at query time. Mismatched dims/model = broken retrieval, often silent (no error, just bad results).

---

## Phase 5: LangGraph Workflow

- [ ] Define state schema (`src/graph.py`):
  ```python
  from typing_extensions import TypedDict
  from langchain_core.documents import Document

  class RAGState(TypedDict):
      query: str
      retrieved_docs: list[Document]
      graded_docs: list[Document]
      answer: str
      retry_count: int
  ```
- [ ] Implement nodes (`src/nodes.py`):
  - `retrieve_node(state)` — calls retriever, populates `retrieved_docs`
  - `grade_node(state)` — (optional) LLM or heuristic filters irrelevant docs
  - `generate_node(state)` — builds prompt from graded docs, calls LLM, returns `answer`
- [ ] Wire graph:
  ```python
  from langgraph.graph import StateGraph, END

  workflow = StateGraph(RAGState)
  workflow.add_node("retrieve", retrieve_node)
  workflow.add_node("grade", grade_node)
  workflow.add_node("generate", generate_node)

  workflow.set_entry_point("retrieve")
  workflow.add_edge("retrieve", "grade")

  def should_retry(state):
      if not state["graded_docs"] and state["retry_count"] < 2:
          return "retrieve"
      return "generate"

  workflow.add_conditional_edges("grade", should_retry, {
      "retrieve": "retrieve",
      "generate": "generate",
  })
  workflow.add_edge("generate", END)

  graph = workflow.compile()
  ```
- [ ] Test invocation: `graph.invoke({"query": "...", "retry_count": 0})`
- [ ] Add tracing/logging (LangSmith optional, or manual print at each node)

**Design note:** conditional retry loop is optional complexity — skip if simple retrieve→generate suffices. Add only if grading reveals frequent empty/weak retrievals.

---

## Phase 6: Prompt & Generation

- [ ] Design prompt template (`src/prompts.py`):
  ```python
  RAG_PROMPT = """Answer the question using ONLY the context below.
  If the context doesn't contain the answer, say so — don't guess.

  Context:
  {context}

  Question: {question}

  Answer:"""
  ```
- [ ] Format retrieved docs into context string, include source metadata for citation
- [ ] Choose LLM: `gpt-4o` (quality) vs `gpt-4o-mini` (cost) vs Claude equivalent
- [ ] Test generation on 10+ representative queries, check for:
  - Hallucination (answer not grounded in context)
  - Missed context (answer ignores relevant retrieved doc)
  - Citation accuracy

**Done marker:** answers are grounded, cite sources, refuse gracefully when context insufficient.

---

## Phase 7: Optional Enhancements (post-MVP)

- [ ] **Re-ranking** — Cohere rerank or cross-encoder model on top-k results before generation (improves precision when k is large)
- [ ] **Hybrid search** — combine BM25 keyword search + semantic search (catches exact-match terms embeddings miss)
- [ ] **Multi-turn memory** — add conversation history to state, rewrite follow-up queries with context
- [ ] **Query rewriting/expansion** — LLM rephrases vague queries before retrieval
- [ ] **Streaming responses** — stream `generate_node` output token-by-token
- [ ] **Web UI** — Streamlit (fast prototype) or FastAPI + frontend (production)

---

## Phase 8: Testing & Evaluation

- [ ] Unit tests: loader, splitter, retriever (mock vector store)
- [ ] Integration test: full graph invoke on known query → assert answer contains expected fact
- [ ] Retrieval eval: build small labeled set (query → expected doc IDs), measure recall@k
- [ ] Answer quality eval: manual review or LLM-as-judge scoring on relevance/faithfulness
- [ ] Load/latency test: measure end-to-end response time under expected query volume

---

## Key Decisions to Lock Early

| Decision | Default choice | Revisit if... |
|---|---|---|
| Vector store | PostgreSQL + pgvector | need managed/serverless scale → Pinecone |
| Chunk size | 1000 tokens, 100 overlap | retrieval missing context → increase; noisy results → decrease |
| Embedding model | text-embedding-3-small | quality insufficient → text-embedding-3-large |
| LLM | gpt-4o-mini | quality insufficient → gpt-4o |
| Reranking | off | top-k results noisy/irrelevant → add Cohere rerank |

## Overall Done Markers

- [ ] Pipeline runs end-to-end from raw doc to answer
- [ ] Retrieval returns relevant chunks for test query set
- [ ] Generated answers grounded in retrieved context, cite sources
- [ ] Tests pass, eval metrics meet threshold
- [ ] (if applicable) UI functional for manual querying
