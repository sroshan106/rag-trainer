"""Deterministic document-level source citations."""

from src.config import env_flag

DEFAULT_ENABLED = True
UNKNOWN_SOURCE = "unknown"


def citations_enabled() -> bool:
    return env_flag("RAG_CITATIONS", default=DEFAULT_ENABLED)


def collect_sources(docs: list) -> list[str]:
    """Deduplicate document sources, preserving retrieval rank order."""
    return list(
        dict.fromkeys(doc.metadata.get("source") or UNKNOWN_SOURCE for doc in docs)
    )


def format_sources(sources: list[str]) -> str:
    """Render sources as a numbered block. Empty string when there are none."""
    if not sources:
        return ""
    lines = [f"  [{i}] {source}" for i, source in enumerate(sources, start=1)]
    return "sources:\n" + "\n".join(lines)
