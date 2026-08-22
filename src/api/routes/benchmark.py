from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.deps import validated_model
from src.api.schemas import (
    BenchmarkRequest,
    BenchmarkTestFileEntry,
    CompareRequest,
    CompareResponse,
    JobResponse,
)
from src.benchmark import files as benchmark_files
from src.benchmark.runner import run_all
from src.jobs.runner import ProgressReporter, runner
from src.rag.graph import ask_compare, ask_direct
from src.rag.model_catalog import list_installed

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


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
    return {"models": list_installed()}


@router.post("/compare", response_model=CompareResponse)
def compare(body: CompareRequest) -> dict:
    model = validated_model(body.model)
    try:
        grounded = ask_compare(body.query, model=model)
        direct = ask_direct(body.query, model=model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"comparison failed: {exc}") from exc
    return {"model": model, "grounded": grounded, "direct": direct}


@router.get("/test-files", response_model=list[BenchmarkTestFileEntry])
def list_test_files() -> list[dict]:
    return benchmark_files.get_uploaded_test_files()


@router.post("/test-files/upload", response_model=BenchmarkTestFileEntry, status_code=201)
def upload_test_file(
    file: UploadFile = File(...),
    question_col: str | None = Form(None),
    answer_col: str | None = Form(None),
    doc_index_col: str | None = Form(None),
) -> dict:
    original = Path(file.filename or "test_questions.csv").name
    if not original.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="only .csv files are supported")

    content = file.file.read()
    try:
        entry = benchmark_files.save_uploaded_test_file(
            original,
            content,
            question_col=question_col,
            answer_col=answer_col,
            doc_index_col=doc_index_col,
        )
    except benchmark_files.UnusableTestFile as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not process test CSV: {exc}") from exc

    return entry


@router.delete("/test-files/{file_id}")
def delete_test_file(file_id: str) -> dict:
    try:
        deleted = benchmark_files.delete_test_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such test file")
    return {"id": deleted["id"], "filename": deleted["filename"]}


@router.post("", response_model=JobResponse, status_code=202)
def start_benchmark(body: BenchmarkRequest) -> dict:
    if run_all is None:
        raise HTTPException(
            status_code=503,
            detail="benchmark runner is not available",
        )
    validated_model(body.model)

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
    job = runner.active("benchmark")
    return job.to_dict() if job else None
