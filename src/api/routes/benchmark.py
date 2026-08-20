"""The Benchmark view's backing route.

Wraps ``tests.benchmark.run_benchmark.run_all`` as a background job -- a full
run is on the order of minutes (roughly 100 questions, several LLM calls
each), too long to hold a request open. ``run_all`` interleaves its suites in
chunks and reports after each one, so the job publishes a real percentage and
running per-suite metrics as it goes: the view can show numbers for a partial
run instead of nothing until the end.

Stopping is cooperative -- ``should_stop`` is polled between chunks, so a
cancel lands within one chunk rather than instantly. Killing threads mid-LLM
call would be instant but would throw away answers that are seconds from
completing (and from being cached).

Imported behind a try/except per the integration contract: ``run_all`` may
not exist yet in a partially-synced checkout, and the rest of the API must
keep working if so.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import BenchmarkRequest, JobResponse
from src.jobs.runner import ProgressReporter, runner
from src.rag.model_catalog import list_installed

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

try:
    from tests.benchmark.run_benchmark import run_all
except ImportError:  # pragma: no cover - exercised only if run_all is missing
    run_all = None


def _run_benchmark(body: BenchmarkRequest):
    def _run(reporter: ProgressReporter) -> list[dict]:
        reporter.update(progress=0.0, message="running benchmark suites")

        def on_progress(metrics: list[dict], done: int, total: int) -> None:
            reporter.update(
                progress=done / total if total else 1.0,
                message=f"{done}/{total} questions answered",
                result=metrics,
            )

        results = run_all(
            body.workers,
            body.sample,
            use_cache=body.use_cache,
            model=body.model,
            chunk_size=body.chunk_size,
            on_progress=on_progress,
            should_stop=lambda: reporter.cancelled,
        )
        message = "benchmark stopped early" if reporter.cancelled else "benchmark complete"
        reporter.update(message=message)
        return results

    return _run


@router.get("/models")
def list_models() -> dict:
    # Only models actually pulled -- see src.rag.model_catalog. No "default":
    # a run must name a model, there is no fallback.
    return {"models": list_installed()}


@router.post("", response_model=JobResponse, status_code=202)
def start_benchmark(body: BenchmarkRequest) -> dict:
    if run_all is None:
        raise HTTPException(
            status_code=503,
            detail="tests.benchmark.run_benchmark.run_all is not available yet",
        )
    installed = list_installed()
    if body.model not in installed:
        if not installed:
            raise HTTPException(
                status_code=422,
                detail="no chat model downloaded -- download one in Settings first",
            )
        raise HTTPException(
            status_code=422,
            detail=f"unknown model {body.model!r} -- choose from {installed}",
        )
    job = runner.submit("benchmark", _run_benchmark(body))
    return job.to_dict()


@router.get("/history", response_model=list[JobResponse])
def benchmark_history() -> list[dict]:
    return [j.to_dict() for j in runner.list() if j.kind == "benchmark"]
