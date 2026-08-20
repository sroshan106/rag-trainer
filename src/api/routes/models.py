"""Settings view's backing route: what models exist, and downloading more.

Pulls run as a background job through the same runner ingest/benchmark use --
a model download is minutes over a slow connection, too long to hold a
request open, and the job's ``result`` doubles as the last-seen progress
fraction so a browser refresh mid-pull still shows something.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import JobResponse, PullModelRequest
from src.jobs.runner import ProgressReporter, runner
from src.rag import model_catalog
from src.vectorstore.rerank import rerank_enabled

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models() -> dict:
    return {
        "catalog": list(model_catalog.CATALOG),
        "installed": model_catalog.list_installed(),
        "embed_models": list(model_catalog.EMBED_MODELS),
        "embed_installed": model_catalog.embed_models_installed(),
        "rerank_model": model_catalog.RERANK_MODEL,
        "rerank_installed": model_catalog.reranker_installed(),
        # Whether reranking is even switched on (RAG_RERANK) -- shown
        # alongside the download status since a downloaded-but-disabled
        # reranker is a different state than not-downloaded.
        "rerank_enabled": rerank_enabled(),
    }


@router.post("/pull", response_model=JobResponse, status_code=202)
def start_pull(body: PullModelRequest) -> dict:
    ollama_pullable = (*model_catalog.CATALOG, *model_catalog.EMBED_MODELS)
    is_reranker = body.model == model_catalog.RERANK_MODEL
    if body.model not in ollama_pullable and not is_reranker:
        raise HTTPException(
            status_code=422,
            detail=f"unknown model {body.model!r} -- choose from "
            f"{[*ollama_pullable, model_catalog.RERANK_MODEL]}",
        )

    def _run(reporter: ProgressReporter) -> dict | None:
        reporter.update(progress=0.0, message="starting download")

        def on_progress(fraction: float | None, status: str) -> None:
            # ``update`` leaves progress untouched when passed None, so a
            # pre-manifest status line (no byte counts yet) only updates the
            # message, not the bar.
            reporter.update(progress=fraction, message=status)

        if is_reranker:
            # The HF download is one opaque blocking call with no hook to poll
            # from, so cancellation can only be honoured at its edges.
            reporter.raise_if_cancelled()
            model_catalog.pull_reranker(on_progress)
        else:
            # Cooperative stop, same as the benchmark job: polled between the
            # pull's progress lines, so a cancel lands within one line instead
            # of never (the download previously ran to completion regardless).
            model_catalog.pull_ollama_model(
                body.model, on_progress, should_stop=lambda: reporter.cancelled
            )
        if reporter.cancelled:
            reporter.update(message="download cancelled")
            return None
        return {"model": body.model}

    job = runner.submit("pull", _run)
    return job.to_dict()


@router.get("/pull/history", response_model=list[JobResponse])
def pull_history() -> list[dict]:
    return [j.to_dict() for j in runner.list() if j.kind == "pull"]
