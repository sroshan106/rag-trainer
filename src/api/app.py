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
from src.config import get_settings
from src.observability.logging import configure_logging

_settings = get_settings()

HOST = _settings.api_host
PORT = _settings.api_port

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
