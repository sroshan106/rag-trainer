"""Full-text (lexical) retrieval over pgvector's stored rows, fused with dense search in hybrid.py."""

import os

import sqlalchemy as sa
from langchain_core.documents import Document

from src.db.engine import get_engine
from src.vectorstore.store import COLLECTION_NAME

TEXT_CONFIG = os.environ.get("RAG_TSVECTOR_CONFIG", "english")
TSV_COLUMN = "doc_tsv"

# websearch_to_tsquery accepts free-form user input safely.
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
    """Pooled engine, reused across queries."""
    return get_engine(connection)


def ensure_index(connection: str | None = None) -> bool:
    """Create tsvector column and GIN index if absent. Idempotent."""
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

        # STORED generated column backfills and stays correct for future inserts.
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
