# Future Ideas

Loose backlog of directions worth exploring later. Not committed, not scoped — just captured so they don't get lost.

## 1. Auto-suggest model pair based on uploaded document + local hardware

When a user uploads a document (or corpus) to ingest, auto-recommend:

- **One local model** — chosen based on document characteristics (size, language, domain) *and* the current machine's actual capability (VRAM, RAM, CPU core count, whether a CUDA/ROCm GPU is present at all). The suggestion should degrade gracefully: e.g. suggest a 3B model on an 8GB-VRAM laptop, a 7-8B model on a 24GB GPU, and fall back to CPU-only quantized options if no GPU detected.
- **One remote/hosted LLM** — chosen based on document complexity (e.g. long-context needs, multi-hop reasoning implied by the questions) rather than hardware, since it isn't hardware-bound.

Needs:
- A hardware probe step (detect GPU vendor/VRAM, RAM, CPU) — could reuse ideas from the earlier GPU-monitoring discussion (`nvidia-smi`, `nvitop`, or a vendor-neutral probe).
- A lightweight document profiler (token count, vocabulary/domain heuristics, language detection) to drive the "why this model" reasoning.
- A recommendation table mapping (doc profile × hardware tier) → suggested local model, kept as a simple config/lookup rather than another ML model.

## 2. Auto-benchmark by uploading a test case file

Let a user upload their own question/answer CSV (same shape as `tests/benchmark/data/*.csv`) through a UI, and have the system:

1. Run it through the current live pipeline (retrieve → grade → generate).
2. Compute the same metrics `run_benchmark.py` already produces (recall@k, answer overlap, refusal rate).
3. Store the result (see idea 3) so it can be compared across runs/models over time.

Could reuse `tests/benchmark/run_benchmark.py` almost as-is as the core eval engine — just needs an entry point that takes an uploaded file path instead of the fixed `data/` directory, and returns structured results instead of printing.

Worth revisiting the eval metric itself too — the keyword-overlap `pass_rate` metric was seen to be harsh on correct-but-differently-worded answers (see benchmark diagnostic session where a graded answer got 0.67 overlap despite being essentially correct). A semantic-similarity-based scorer (embedding cosine sim between expected and actual answer) might be a truer signal than raw keyword overlap.

## 3. Persist benchmark history: hashes + model + results

Track benchmark runs over time so scores are comparable:

- Hash the uploaded test file (content hash) so re-runs of the same test set are recognized as the same "suite."
- Hash or fingerprint the corpus/vectorstore state (e.g. hash of ingested document set) so results are attributable to a specific data snapshot.
- Store per-run: test-set hash, corpus hash, model used (local + remote, if both were tried), metrics (recall@k, overlap, pass_rate, refusal_rate), timestamp.
- This becomes the audit trail for "did changing embedding model / grading threshold / prompt actually help."

Simplest storage: a small relational table (SQLite is already in the stack per `plan.md`/`ui_plan.md` references) — no need for anything fancier until scale demands it.

## 4. SaaS packaging: multi-user, login, hosted benchmarking

Turn the above into a hosted product:

- **Auth**: user accounts, login (email/password or OAuth). Each user's uploads, corpora, and benchmark history scoped to their account.
- **Per-user isolation**: separate vectorstore namespace/collection per user (or per project within a user), separate benchmark history table rows keyed by user_id.
- **Compute model**: since the "local model" recommendation is inherently tied to *someone's* hardware, in a SaaS context this either means:
  - running the "local" model on shared backend infra sized for the median user (loses the hardware-personalization angle), or
  - keeping true local-model support only for a self-hosted/desktop tier, and using only remote/hosted LLMs in the SaaS tier.
  - Worth deciding explicitly which tier does what before building further — the hardware-aware recommendation (idea 1) only makes sense if the app truly has access to the end machine's specs, which is true for a local/desktop app but not for a multi-tenant server.
- **Billing hook**: benchmark runs and remote-LLM calls are the natural metering unit (cost-bearing), while local-model runs on the user's own machine are free to the operator — could shape a pricing tier around that split.
- **Storage**: multi-tenant Postgres likely replaces SQLite at this point (matches "Postgres for both text collection and vector DB" option already noted in `plan.md`/document data).

## 5. On-premise hosting for multi-user orgs (internal or external tenants)

A third deployment shape, distinct from idea 4's public SaaS: the app runs inside an organization's own infrastructure (their datacenter, VPC, or a dedicated instance we manage on their behalf), but still serves multiple users — either purely internal staff, or a mix of internal + external users the org grants access to (e.g. an org's customers, contractors, partners).

This sits between "single local desktop app" and "public multi-tenant SaaS":

- **Auth stays multi-user, but scoped to the org**: SSO/SAML/OIDC against the org's own identity provider (Okta, Azure AD, Google Workspace) is more likely required than plain email/password, since it's an org deployment. Role distinction matters — internal admins vs internal users vs external guest users likely need different permission levels (e.g. who can upload new corpora/ingest documents vs who can only query).
- **Hardware-aware model suggestion (idea 1) becomes viable again here** — unlike public SaaS, the org's own server hardware is known and fixed, so the "local model" recommendation logic can target that one machine/cluster's actual specs rather than a hypothetical median user.
- **Data stays on the org's infrastructure** — this is likely the entire pitch versus public SaaS: no document or query ever leaves the org's network boundary. Matters for orgs with compliance constraints (legal, healthcare, finance) that currently rule out public SaaS entirely.
- **External users add a wrinkle**: if the org grants access to outside users (e.g. their own customers), need per-tenant data isolation *within* the on-prem deployment too — not just user-level scoping, but making sure org A's guest users can't see org B's guest users' data if the deployment ever serves more than one org's guest population. Simplest: one deployment = one org, full stop, and any external users are still just "users of that org's instance," fully isolated from any other org's separate deployment.
- **Deployment mechanics**: this is the natural fit for the existing `Dockerfile`/`docker-compose.yml` in the repo — package as a self-contained stack (app + vectorstore + optional local LLM runtime) the org's IT team runs on their own box. Licensing/updates become the operational question (how do orgs get patches — pull an image, or an update mechanism?).
- **Pricing shape**: likely a license/seat model per org rather than the metered per-call billing that made sense for idea 4's public SaaS, since local-model compute cost is now the org's own hardware, not ours.

This is arguably the deployment shape most consistent with the project's current "local-only, privacy-first" architecture decision (see `rag_architecture_decisions` memory) — it keeps that property while still adding multi-user support, which idea 4 (public SaaS) trades away.

## 6. Hybrid retrieval: BM25 + dense with Reciprocal Rank Fusion

Dense retrieval alone is weak on literal-phrase lookups, and the grading thresholds that compensate for it are fragile. Measured on the current 223-chunk corpus with `nomic-embed-text`:

| Query | Top-5 relevance scores |
| --- | --- |
| who said I love you and ran off | 0.5685 … 0.5218 |
| Ruby phoned Carla and said I love you | 0.6691 … 0.5455 |
| who is the Doctor | 0.6126 … 0.5963 |
| capital of Mongolia (off-topic) | 0.4454 … 0.4215 |
| quantum chromodynamics (off-topic) | 0.4531 … 0.4432 |

On-topic and off-topic scores are separated by roughly 0.07, and for the "I love you / ran off" query the correct chunk beats the best irrelevant chunk by only **+0.0156** cosine. `RELEVANCE_FLOOR` currently sits at 0.48 — about 0.027 above the observed noise ceiling. That works for this corpus, but any new document set shifts the distribution and the floor has to be re-tuned by hand.

BM25 over the same chunks, same query, ranks the correct chunk first with a wide margin (16.86 vs 13.01 for the runner-up) — precisely because the query is a literal-phrase lookup (`"I love you"`, `"ran off"`), which is dense retrieval's weak spot and lexical search's strength.

### Proposed shape

Postgres already hosts the vectors via `PGVector`, so the lexical half needs no new infrastructure:

1. Add a `tsvector` column over chunk text plus a GIN index on the existing collection table.
2. Run `ts_rank_cd` full-text search alongside the vector search, retrieving top-k from each independently.
3. Fuse the two ranked lists with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), rather than trying to normalize a cosine score against a BM25 score — the scales aren't comparable, ranks are.

The payoff is that `grade_node` stops depending on absolute cosine thresholds. Grading becomes rank-based (keep the top-n fused results, refuse when neither retriever produced a confident hit), which removes the per-corpus threshold tuning that `RELEVANCE_FLOOR`/`RELEVANCE_RATIO` currently require.

### Tested and rejected

- **nomic task prefixes** (`search_query:` / `search_document:`). Measured on this corpus, the gold-vs-best-noise gap *shrank* from 0.0156 to 0.0013. Not worth a re-ingest here, despite being the documented usage for the model.
- **Retrieval retry loop.** The comment in `src/rag/graph.py` is correct: re-running a deterministic, similarity-sorted retrieval can't surface a chunk the first pass missed. Only worth revisiting alongside query rewriting.

## Open questions to resolve before committing to any of the above

- Is the product positioned as self-hosted (privacy-first, current direction per architecture decisions) or hosted SaaS (idea 4)? These pull in different directions — worth an explicit decision before idea 4 gets built, since it changes the hardware-detection story (idea 1) fundamentally.
- What's the actual unit of "test case file" — same CSV schema as internal benchmark data, or does it need a friendlier upload format for external users?
- Does the semantic-similarity scorer (idea 2) replace or supplement the keyword-overlap metric — need both for backward comparability with existing benchmark data?
