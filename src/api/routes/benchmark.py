"""The Benchmark view's backing route.

Wraps ``tests.benchmark.run_benchmark.run_all`` as a background job -- a full
run is on the order of minutes (roughly 100 questions, several LLM calls
each), too long to hold a request open. Progress is coarse (three suites run
inside ``run_all`` with no per-suite hook exposed), so this reports
pending/running/done rather than a live percentage; the job result itself
carries the full per-suite metrics once it finishes.

Imported behind a try/except per the integration contract: ``run_all`` may
not exist yet in a partially-synced checkout, and the rest of the API must
keep working if so.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import BenchmarkRequest, JobResponse
from src.jobs.runner import ProgressReporter, runner

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

try:
    from tests.benchmark.run_benchmark import run_all
except ImportError:  # pragma: no cover - exercised only if run_all is missing
    run_all = None


def _run_benchmark(body: BenchmarkRequest):
    def _run(reporter: ProgressReporter) -> list[dict]:
        reporter.update(progress=0.05, message="running benchmark suites")
        results = run_all(body.workers, body.sample, use_cache=body.use_cache)
        reporter.update(progress=1.0, message="benchmark complete")
        return results

    return _run


@router.post("", response_model=JobResponse, status_code=202)
def start_benchmark(body: BenchmarkRequest) -> dict:
    if run_all is None:
        raise HTTPException(
            status_code=503,
            detail="tests.benchmark.run_benchmark.run_all is not available yet",
        )
    job = runner.submit("benchmark", _run_benchmark(body))
    return job.to_dict()


@router.get("/history", response_model=list[JobResponse])
def benchmark_history() -> list[dict]:
    return [j.to_dict() for j in runner.list() if j.kind == "benchmark"]
