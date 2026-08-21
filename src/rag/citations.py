"""Deterministic document-level source citations.

A citation names the file a chunk came from and where in that file it sits, so
the viewer can open the document at exactly that unit. Both halves come from
metadata stamped at ingest -- see src/ingestion/units.py for what a unit is and
why its index is not the same thing as the dataset's own record key.
"""

from src.config import env_flag

DEFAULT_ENABLED = True


def citations_enabled() -> bool:
    return env_flag("RAG_CITATIONS", default=DEFAULT_ENABLED)


def _citation(doc) -> dict:
    metadata = doc.metadata
    kind = metadata.get("unit_kind")
    index = metadata.get("unit_index")
    return {
        "file_id": metadata.get("file_id"),
        "filename": metadata.get("filename"),
        "unit_kind": kind,
        "unit_index": index,
        "label": f"{kind} {index}" if kind and index is not None else None,
        # Only set when the source data carried a link of its own; the UI
        # prefers it over the stored copy when it is there.
        "url": metadata.get("url"),
        # Values of the columns picked (or detected) as this row's citation
        # source, e.g. {"id": "31776899", "source_url": "https://..."}.
        "fields": metadata.get("citation_fields"),
    }


def collect_citations(docs: list) -> list[dict]:
    """One citation per retrieved chunk, in rank order, deduplicated.

    Two chunks of the same unit cite it once. Chunks ingested before citations
    carried provenance have no file_id and no locator; they are dropped rather
    than rendered as an unclickable blank.
    """
    seen = set()
    citations = []
    for doc in docs:
        citation = _citation(doc)
        if citation["file_id"] is None or citation["unit_index"] is None:
            continue
        identity = (citation["file_id"], citation["unit_index"])
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(citation)
    return citations


def format_citations(citations: list[dict]) -> str:
    """Render citations as a numbered block. Empty string when there are none."""
    if not citations:
        return ""
    def _suffix(c: dict) -> str:
        parts = [f"{k}: {v}" for k, v in (c.get("fields") or {}).items()]
        if c["url"] and c["url"] not in (c.get("fields") or {}).values():
            parts.append(c["url"])
        return f" ({', '.join(parts)})" if parts else ""

    lines = [
        f"  [{i}] {c['filename']} -- {c['label']}{_suffix(c)}"
        for i, c in enumerate(citations, start=1)
    ]
    return "sources:\n" + "\n".join(lines)
