"""One pooled SQLAlchemy engine per connection string, shared process-wide.

History, file provenance, lexical search and the chunk count all live in the
same database, and each used to keep its own engine cache -- four pools to one
Postgres, and a count query that built and threw away a pool on every Ask page
load. They share this one instead.

``init`` exists because some callers need a one-off schema step (``create_all``,
a migration) the first time a given URL is used. Sharing the engine must not
mean sharing that step: it is tracked per (url, init), so whichever module
happens to open the engine first doesn't rob the others of their setup.
"""

import os
import threading
from typing import Callable

import sqlalchemy as sa

_engines: dict[str, sa.Engine] = {}
_initialized: set[tuple[str, Callable[[sa.Engine], None]]] = set()
_lock = threading.Lock()


def get_engine(
    url: str | None = None,
    init: Callable[[sa.Engine], None] | None = None,
) -> sa.Engine:
    """The cached engine for ``url``, defaulting to ``DATABASE_URL``.

    ``init`` is run once per URL, under the lock, before the engine is handed
    out. It must be a stable module-level function, not a fresh closure, or it
    will re-run on every call.
    """
    url = url or os.environ["DATABASE_URL"]

    engine = _engines.get(url)
    if engine is not None and (init is None or (url, init) in _initialized):
        return engine

    with _lock:
        engine = _engines.get(url)
        if engine is None:
            engine = sa.create_engine(url)
            _engines[url] = engine
        if init is not None and (url, init) not in _initialized:
            # Left unmarked if it raises, so the next caller retries it.
            init(engine)
            _initialized.add((url, init))
    return engine
