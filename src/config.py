"""Shared environment-variable parsing.

Keeps the truthiness convention identical across RAG_CITATIONS, RAG_TRACE, and
any flag added later — one definition of what counts as "off".
"""

import os

FALSY = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var. Unset returns ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in FALSY
