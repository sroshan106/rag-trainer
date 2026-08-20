"""The Benchmark view's backing routes.

Wraps ``tests.benchmark.run_benchmark.run_all`` as a background job -- a full
run is on the order of minutes (roughly 100 questions, several LLM calls
each), too long to hold a request open. ``run_all`` interleaves its suites in
chunks and reports after each one, so the job publishes a real percentage and
running per-suite metrics as it goes: the view can show numbers for a partial
run instead of nothing until the end.

Supports modular custom test suites and datasets uploaded by users.

Stopping is cooperative -- ``should_stop`` is polled between chunks, so a
cancel lands within one chunk rather than instantly. Killing threads mid-LLM
call would be instant but would throw away answers that are seconds from
completing (and from being cached).

Imported behind a try/except per the integration contract: ``run_all`` may
not exist yet in a partially-synced checkout, and the rest of the API must
keep working if so.
"""

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import BenchmarkRequest, BenchmarkTestFileEntry, JobResponse
from src.benchmark import files as benchmark_files
from src.jobs.runner import ProgressReporter, runner
from src.rag.model_catalog import list_installed

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

try:
    from tests.benchmark.run_benchmark import run_all
except ImportError:  # pragma: no cover - exercised only if run_all is missing
    run_all = None


def _run_benchmark(body: BenchmarkRequest, test_paths: list[str] | None = None):
    def _run(reporter: ProgressReporter) -> list[dict]:
        reporter.update(progress=0.0, message="running benchmark suites")

        def on_progress(metrics: list[dict], done: int, total: int) -> None:
            reporter.update(
                progress=done / total if total else 1.0,
                message=f"{done}/{total} questions answered",
                result=metrics,
            )

        kwargs = {
            "workers": body.workers,
            "sample": body.sample,
            "use_cache": body.use_cache,
            "model": body.model,
            "chunk_size": body.chunk_size,
            "on_progress": on_progress,
            "should_stop": lambda: reporter.cancelled,
        }
        if test_paths is not None:
            kwargs["test_files"] = test_paths

        results = run_all(**kwargs)
        message = "benchmark stopped early" if reporter.cancelled else "benchmark complete"
        reporter.update(message=message)
        return results

    return _run


@router.get("/models")
def list_models() -> dict:
    # Only models actually pulled -- see src.rag.model_catalog. No "default":
    # a run must name a model, there is no fallback.
    return {"models": list_installed()}


@router.get("/test-files", response_model=list[BenchmarkTestFileEntry])
def list_test_files() -> list[dict]:
    """List all available benchmark test suites, both built-in and user-uploaded."""
    return benchmark_files.list_test_files()


@router.post("/test-files/upload", response_model=BenchmarkTestFileEntry, status_code=201)
def upload_test_file(file: UploadFile = File(...)) -> dict:
    """Upload a custom benchmark test CSV containing questions and optional answers/document indices."""
    original = Path(file.filename or "test_questions.csv").name
    if not original.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="only .csv files are supported")

    content = file.file.read()
    try:
        entry = benchmark_files.save_uploaded_test_file(original, content)
    except benchmark_files.UnusableTestFile as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not process test CSV: {exc}") from exc

    return entry


@router.delete("/test-files/{file_id}")
def delete_test_file(file_id: str) -> dict:
    """Delete an uploaded custom benchmark test suite."""
    try:
        deleted = benchmark_files.delete_uploaded_test_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such uploaded test file")
    return {"id": deleted["id"], "filename": deleted["filename"]}


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

    resolved_paths: list[str] | None = None
    if body.test_files:
        resolved_paths = []
        for tf in body.test_files:
            try:
                path = benchmark_files.resolve_test_file_path(tf)
                resolved_paths.append(str(path))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    job_params = {
        "model": body.model,
        "workers": body.workers,
        "sample": body.sample,
        "chunk_size": body.chunk_size,
        "use_cache": body.use_cache,
        "test_files": body.test_files,
    }
    job = runner.submit("benchmark", _run_benchmark(body, resolved_paths), params=job_params)
    return job.to_dict()


@router.get("/history", response_model=list[JobResponse])
def benchmark_history() -> list[dict]:
    return [j.to_dict() for j in runner.list() if j.kind == "benchmark"]


@router.get("/active", response_model=JobResponse | None)
def active_benchmark() -> dict | None:
    """Lets the view re-attach to a benchmark run in progress after a page reload."""
    job = runner.active("benchmark")
    return job.to_dict() if job else None
