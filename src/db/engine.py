import threading
from typing import Callable

import sqlalchemy as sa

from src.config import get_settings

_engines: dict[str, sa.Engine] = {}
_initialized: set[tuple[str, Callable[[sa.Engine], None]]] = set()
_lock = threading.Lock()


def get_engine(
    url: str | None = None,
    init: Callable[[sa.Engine], None] | None = None,
) -> sa.Engine:
    url = url or get_settings().database_url

    engine = _engines.get(url)
    if engine is not None and (init is None or (url, init) in _initialized):
        return engine

    with _lock:
        engine = _engines.get(url)
        if engine is None:
            kwargs = {"pool_pre_ping": True}
            if not url.startswith("sqlite"):
                kwargs.update({"pool_recycle": 3600, "pool_size": 10, "max_overflow": 20})
            engine = sa.create_engine(url, **kwargs)
            _engines[url] = engine
        if init is not None and (url, init) not in _initialized:
            init(engine)
            _initialized.add((url, init))
    return engine
