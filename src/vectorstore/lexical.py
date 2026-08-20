"""Full-text (lexical) retrieval over the same rows pgvector already stores.

Dense retrieval is weak on literal-phrase lookups -- measured on this corpus,
a query quoting an exact line beat the best irrelevant chunk by 0.0156 cosine,
while BM25-style ranking separated them by a wide margin. This module adds the
lexical half so the two can be fused (see ``hybrid.py``).

No re-ingest is needed: the chunk text is already in
``langchain_pg_embedding.document``, so the index is a generated column
backfilled by Postgres itself.
"""

import os

import sqlalchemy as sa
from langchain_core.documents import Document

from src.db.engine import get_engine
from src.vectorstore.store import COLLECTION_NAME

TEXT_CONFIG = os.environ.get("RAG_TSVECTOR_CONFIG", "english")
TSV_COLUMN = "doc_tsv"

# ``websearch_to_tsquery`` is the only tsquery parser that accepts free-form
# user input without raising on stray punctuation -- plainto_ tokenises quotes
# away, and to_tsquery rejects them outright.
_SEARCH_SQL = f"""
SELECT e.id,
       e.document,
       e.cmetadata,
       ts_rank_cd(e.{TSV_COLUMN}, q) AS rank
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id,
     websearch_to_tsquery(:config, :query) AS q
WHERE c.name = :collection
  AND e.{TSV_COLUMN} @@ q
ORDER BY rank DESC
LIMIT :k
"""


def _engine(connection: str | None = None) -> sa.Engine:
    """One pooled engine per connection string.

    Retrieval runs once per query and the benchmark runs it from a thread pool,
    so building an engine per call would open (and discard) a connection pool
    each time.
    """
    return get_engine(connection)


def ensure_index(connection: str | None = None) -> bool:
    """Create the tsvector column and its GIN index if absent.

    Idempotent, and safe to call on every startup. Returns True when it had to
    build something, so callers can log the one-off backfill.
    """
    with _engine(connection).begin() as conn:
        exists = conn.execute(
            sa.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'langchain_pg_embedding' AND column_name = :col"
            ),
            {"col": TSV_COLUMN},
        ).scalar()
        if exists:
            return False

        # A STORED generated column backfills every existing row on creation
        # and stays correct for future inserts without touching the ingest path.
        conn.execute(
            sa.text(
                f"ALTER TABLE langchain_pg_embedding "
                f"ADD COLUMN {TSV_COLUMN} tsvector GENERATED ALWAYS AS "
                f"(to_tsvector('{TEXT_CONFIG}', document)) STORED"
            )
        )
        conn.execute(
            sa.text(
                f"CREATE INDEX IF NOT EXISTS langchain_pg_embedding_{TSV_COLUMN}_idx "
                f"ON langchain_pg_embedding USING GIN ({TSV_COLUMN})"
            )
        )
    return True


def search(
    query: str,
    k: int,
    collection: str = COLLECTION_NAME,
    connection: str | None = None,
) -> list[tuple[Document, float]]:
    """Return the top-k lexical matches with their ts_rank_cd scores."""
    with _engine(connection).connect() as conn:
        rows = conn.execute(
            sa.text(_SEARCH_SQL),
            {"config": TEXT_CONFIG, "query": query, "collection": collection, "k": k},
        ).fetchall()

    return [
        (Document(page_content=row.document, metadata=dict(row.cmetadata or {})), row.rank)
        for row in rows
    ]
