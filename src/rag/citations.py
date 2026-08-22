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
        "url": metadata.get("url"),
        "fields": metadata.get("citation_fields"),
    }


def collect_citations(docs: list) -> list[dict]:
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
