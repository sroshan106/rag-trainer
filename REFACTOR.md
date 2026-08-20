# Refactor backlog

Findings from an architecture review of `src/` and `ui/src/`, ordered so that each item makes
the next one easier. Everything above the line is already done; everything below it is not.

The codebase is in good shape overall — no file exceeds 400 LOC, there are no bare `except:`
clauses, and there is no dead code. What follows is about layering and lifetimes, not rot.

---

## Done

| # | Change | Files |
|---|---|---|
| 1 | `stream_query` no longer blocks the event loop on a synchronous `httpx.get` to Ollama — model validation runs via `asyncio.to_thread` | `api/routes/query.py` |
| 2 | Ingest "one at a time" is now atomic: `JobRunner.submit_exclusive` checks and inserts under one lock, raising `JobAlreadyRunning`. `Job.to_dict()` and `cancel()` take the lock, so snapshots are consistent | `jobs/runner.py`, `api/routes/ingest.py` |
| 3 | Silent exception swallows narrowed and logged — a real NVML fault no longer looks identical to "this machine has no GPU", and a failed reranker-cache probe no longer looks like "not installed" | `observability/sysmetrics.py`, `rag/model_catalog.py` |
| 4 | Model pulls are genuinely cancellable: a read timeout on the stream plus a cancellation check between lines. Previously `cancel` returned success while the pull ran forever | `api/routes/models.py`, `rag/model_catalog.py` |
| 5 | One shared engine cache in `src/db/engine.py`, replacing three near-identical copies. `count_chunks` no longer builds and disposes an engine on every Ask page load | new `db/engine.py`, `rag/history.py`, `ingestion/files.py`, `vectorstore/lexical.py`, `vectorstore/store.py` |

---

## Remaining

### 6. A settings object instead of scattered `os.environ` reads — *medium risk*

`os.environ` is read at import time in `nodes.py` (5 vars), `store.py`, `rerank.py` (4),
`hybrid.py`, `lexical.py`, `logging.py`, `model_catalog.py`, and `app.py`. `config.py` exposes
only `env_flag` — there is no settings type and nothing takes configuration as a parameter.

Because the reads happen at import time, changing an environment variable in a test requires
reloading the module. Introduce a frozen `Settings` dataclass built once via `get_settings()`,
and convert the import-time reads to lookups. Keep `env_flag`'s semantics as-is.

### 7. Move the model registry out of the graph nodes — *medium risk*

`AVAILABLE_MODELS` is defined in `rag/nodes.py:24` and re-exported as `model_catalog.CATALOG`,
which `/models/pull` then uses as its download allow-list. A download endpoint's allow-list
should not be transitively defined by generation-node internals.

Move `AVAILABLE_MODELS`, `_num_ctx_for`, and the `ChatOllama` client cache into a new
`rag/models.py`; have `model_catalog` import the registry rather than the node module. While
there, give the vectorstore cache and the LLM cache separate locks — today one lock covers
both, so building the vectorstore blocks LLM client construction.

### 8. One authoritative model-resolution policy — *medium risk*

There are currently two, and neither wins: `graph._resolve_model` checks catalog membership and
raises `ValueError`; `query._validated_model` and `benchmark.start_benchmark` each
independently check *installed* membership and raise `HTTPException` — the same block, written
twice. Replace with a single `resolve_model(name)` in `rag/model_policy.py` that checks both and
raises a domain error the routes translate to 422.

### 9. Move ingest orchestration out of the route — *medium risk*

`ingest.upload_and_ingest` is ~70 lines of domain logic (streaming hash, size quota,
dedup-by-hash, CSV pre-parse, provenance record, job submit); `delete_ingested_file` orchestrates
vector-store, filesystem, and database deletion from a route body with no transaction, so a
failure between steps orphans chunks or records. Move both into `ingestion/uploads.py` and leave
the route as ~25 lines of HTTP mapping.

### 10. Split `rag/nodes.py` — *medium risk*

382 lines covering six concerns: the model registry, the lazy singleton caches, retrieval,
grading policy, generation, and qwen3 think-tag handling. Split into `rag/retrieve.py`,
`rag/grade.py`, `rag/generate.py`, and `rag/thinking.py`. Do this after #6 and #7, once the
constants and caches have already moved out.

`rag/history.py` (251) has the same shape — table definition, hand-rolled `ALTER TABLE`
migrations, DAO, and row mapping in one file — and `ingestion/files.py` duplicates its row-mapping
pattern.

### 11. Move the benchmark runner out of `tests/` — *medium risk*

`api/routes/benchmark.py:29` imports `tests.benchmark.run_benchmark`, so the production API
depends on the test package and the route degrades to 503 if `tests/` is not shipped. The runner
is application code in the wrong tree: move it to `src/benchmark/` and delete the
`try/except ImportError` branch.

### 12. Remove import-time side effects — *high risk, do last*

`src/__init__.py` runs `load_dotenv()` for anything that imports the package; `loaders.py:8`
mutates the interpreter-global `csv.field_size_limit`; `graph.py:64` compiles the LangGraph at
import; `app.py:54` builds the app *and* configures logging at import. Build the graph lazily,
move `load_dotenv()` to the entrypoint, move the csv limit inside `load_documents`, and keep
`app = create_app()` behind an ASGI factory. This touches every import path.

---

## Smaller items

- `docker-compose.yml` bind-mounts `./src:/app/src` over the built image, so the running code is
  not the built code. It also reserves a GPU for the `app` container, which installs the CPU
  torch wheel and never uses one. (The bind-mount is also why `src/__pycache__` ends up owned by
  root.)
- `observability/logging.tail()` has no route — the Logs feature is wired up to nothing.
- `requirements.txt` is entirely unpinned, so two installs a week apart are two different
  systems. Pin, or move to a lockfile.
- `run_query` maps every exception to 502, collapsing the 4xx/5xx distinction — a bad model name
  and a dead database look the same to the client.
- `history.recent/get/delete_all/delete` and `files.recent/get/find_by_hash` have no error
  handling, so a database outage returns a 500 with a SQLAlchemy traceback shape, while the
  write paths on the same tables degrade quietly. Pick one.
- `runner.submit` spawns an unbounded daemon thread per job, and `_prune()` can delete a job a
  client is still polling, turning the poll into a 404 with no terminal state ever observed.
- `ask_stream`'s `GeneratorExit` handler performs a blocking database write during generator
  teardown.
