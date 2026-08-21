"""FastAPI instance for the RAG dashboard.

Binds to 127.0.0.1 by default (ui_plan.md open decision 1: no auth,
single user) -- see ``run`` below, which is what ``uvicorn src.api.app:app``
or ``python -m src.api.app`` both end up using.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import (
    benchmark,
    documents,
    history,
    ingest,
    jobs,
    metrics,
    models,
    query,
)
from src.observability.logging import configure_logging

HOST = os.environ.get("RAG_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAG_API_PORT", "8000"))

# The Vite dev server runs on a different origin during development; the
# production build is served same-origin (see ui/README) and needs no CORS
# entry at all. Only localhost origins are allowed -- this is a local tool.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="RAG Dashboard API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(history.router)
    app.include_router(query.router)
    app.include_router(ingest.router)
    app.include_router(documents.router)
    app.include_router(benchmark.router)
    app.include_router(metrics.router)
    app.include_router(jobs.router)
    app.include_router(models.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
