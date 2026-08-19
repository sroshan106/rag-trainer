"""Generic job status endpoints, shared by ingest and benchmark polling.

One registry (`src.jobs.runner.runner`), one status shape -- the Ingest and
Benchmark views both poll `GET /api/jobs/{id}` rather than each growing their
own status route.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import JobResponse
from src.jobs.runner import runner

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=list[JobResponse])
def list_jobs(kind: str | None = None) -> list[dict]:
    jobs = runner.list()
    if kind:
        jobs = [j for j in jobs if j.kind == kind]
    return [j.to_dict() for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> dict:
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()
