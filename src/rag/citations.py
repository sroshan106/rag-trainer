"""Optional extension: deterministic source citations.

The generation model is not reliable at following the inline-citation
instruction in RAG_PROMPT, so sources are collected in code from the
documents that actually reached the prompt. This is document-level
attribution, not sentence-level.

Disable with RAG_CITATIONS=false to fall back to answer-only output.
"""

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
