# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rag-trainer` — local-only retrieval-augmented generation system. Postgres/pgvector stores
chunks + embeddings, Ollama serves both the embedding and chat models, FastAPI wires it together
behind a React dashboard. No data leaves the host. Tuned for a 4GB GTX 1050 — defaults assume
real VRAM pressure, not abundance.

Read `ARCHITECTURE.md` before changing code — it has the layer diagram, query-flow and
ingest-flow mermaid diagrams, and a "where new code goes" table. Read `explainer.md` for RAG
concepts, `plan.md` for locked decisions and benchmark baselines.

## Commands

```bash
# Run stack
cp .env.example .env
docker compose up -d                                          # postgres+pgvector, ollama, api, ui
docker compose exec app python -m src.ingestion.pipeline data/uploads/your.csv

# Without Docker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --reload        # API on :8000
cd ui && npm install && npm run dev     # dashboard on :5173

# Tests
pytest                                       # unit tests only (RAG_INTEGRATION unset skips integration marker)
RAG_INTEGRATION=1 pytest -m integration      # also hits live Postgres + Ollama
pytest tests/rag/test_graph.py               # single file
pytest tests/rag/test_graph.py::test_name    # single test

# UI
cd ui && npm run lint       # oxlint
cd ui && npm run build      # vite build
```

A chat model must be pulled before the first query — there is deliberately no default model
(`ollama pull llama3.2:3b`, or use the Settings view).

## Architecture — the four layers

Arrows only point downward: `src/vectorstore` never imports `src/rag`; `src/rag` never imports
`src/api`. Need something from a layer above? Pass it in as an argument instead of importing up.

- **Interface** — `ui/` (React dashboard: Ask, Ingest, Benchmark, System, Settings), CLI
  entrypoints (`src.rag.graph`, `src.ingestion.pipeline`)
- **HTTP** — `src/api` (routers + Pydantic schemas, HTTP mapping only — no orchestration logic),
  `src/jobs` (thread-per-job background runner with cooperative cancel)
- **Domain** — `src/rag` (graph, nodes, prompts, citations, history, model catalog),
  `src/ingestion` (loaders, splitter, pipeline, file provenance)
- **Infrastructure** — `src/vectorstore` (pgvector, lexical search, hybrid RRF fusion,
  cross-encoder reranker), `src/observability` (tracing spans, JSON logging, host/GPU metrics),
  `src/config.py` (env access)

### Query path

`ask()` / `ask_stream()` in `src/rag/graph.py` walk three nodes: retrieve → grade → generate.

- Retrieve (`src/rag/nodes.py`): dense k-NN (pgvector, `FETCH_K=20`) optionally fused via RRF
  (`src/vectorstore/hybrid.py`) with lexical tsvector search, optionally reranked by a
  cross-encoder (`ms-marco-MiniLM-L-6-v2`, CPU), cut to `RETRIEVE_K=5`.
- Grade: drops chunks below `RELEVANCE_FLOOR=0.56` or below `RELEVANCE_RATIO=0.9` of the best
  score. If nothing survives, the answer is a hardcoded refusal — never a guess.
- Generate: prompt template + surviving chunks → ChatOllama. `qwen3*` models get their `<think>`
  block stripped and `/no_think` appended. **Citations are computed deterministically in
  `src/rag/citations.py` from the surviving chunks — never asked of the LLM**, since small local
  models don't reliably follow inline-citation instructions.
- No retry edge: retrieval is deterministic, so re-running the same query can't surface a chunk
  that clears the grader when the top results didn't.

### Ingest path

`POST /api/ingest` → hash while streaming to disk → dedup by hash (409 if already ingested) →
CSV parse (422 `UnusableCSV` if unusable) → provenance row via `files.record()` →
`runner.submit()` background job → `load_documents` (`src/ingestion/loaders.py`) →
`split_documents` (1000 chars, 150 overlap) → `build_vectorstore` (embed via Ollama, write to
pgvector) → `ensure_index` (tsvector column + GIN index).

**Invariant: the embedding model used at ingest must match the one used at query time.** A
mismatch does not raise — it silently returns garbage neighbours. Treat the embedding model as
baked into the index, not a runtime setting.

### Configuration has three different lifetimes — don't conflate them

| Lifetime | Examples | Changing it means |
|---|---|---|
| Baked into the index | chunk size, chunk overlap, embedding model | Re-ingesting the entire corpus |
| Live per query | `k`, relevance floor/ratio, chat model, citations on/off | Takes effect next query |
| Server-owned | `DATABASE_URL`, `OLLAMA_BASE_URL` | Restart; never exposed to the browser |

### Where new code goes

| Adding… | Goes in… |
|---|---|
| A new document format | `src/ingestion/loaders.py` |
| A new chunking strategy | `src/ingestion/splitter.py` (register in `SPLITTERS`) |
| A new retrieval signal | `src/vectorstore/`, fused in `hybrid.py` |
| A change to how answers are produced | `src/rag/nodes.py` + `src/rag/prompts.py` |
| A new endpoint | Router in `src/api/routes/`, schema in `src/api/schemas.py` |
| Anything long-running | A job kind on `src/jobs/runner.py`, never inline in a route |
| A new dashboard screen | `ui/src/views/`, wired into `ui/src/App.jsx` |

A route function should read as HTTP mapping only: parse, call one domain function, translate
errors to status codes. If a route body starts orchestrating storage, the logic escaped its
layer.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (60-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk go test             # Go test failures only (90%)
rtk jest                # Jest failures only (99.5%)
rtk vitest              # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk pytest              # Python test failures only (90%)
rtk rake test           # Ruby test failures only (90%)
rtk rspec               # RSpec test failures only (60%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
rtk uv run <cmd>        # Compact uv project command output
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%). Format flags (-c, -l, -L, -o, -Z) run raw.
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->
