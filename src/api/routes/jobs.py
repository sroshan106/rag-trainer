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


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str) -> dict:
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not runner.cancel(job_id):
        raise HTTPException(status_code=409, detail=f"job already {job.status}")
    return job.to_dict()
