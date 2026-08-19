"""In-process background job registry for long-running requests.

Ingestion and benchmark runs take seconds to minutes -- too long to hold a
request open. A thread (not asyncio) runs the job body because the work it
wraps (``ingestion.pipeline.main``, ``run_benchmark.run_all``) is synchronous,
blocking I/O against Postgres and Ollama.

No task queue, no persistence table: this is a single-user local tool and the
one requirement -- a status endpoint that survives a browser refresh -- only
needs the job to outlive the *request*, not the *process*. If that changes,
this is the module to grow a `jobs` table, not the routes.
"""

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

JobStatus = Literal["pending", "running", "done", "failed"]

# Bound how many finished jobs stay addressable, so a long-lived dashboard
# session can't grow this dict without limit.
MAX_FINISHED_JOBS = 200


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = "pending"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProgressReporter:
    """Passed into a job body so it can report progress without knowing about Job."""

    def __init__(self, job: "Job", lock: threading.Lock):
        self._job = job
        self._lock = lock

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        with self._lock:
            if progress is not None:
                self._job.progress = max(0.0, min(1.0, progress))
            if message is not None:
                self._job.message = message
            self._job.updated_at = time.time()


class JobRunner:
    """Registry of jobs keyed by id, each executed on its own thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[ProgressReporter], Any]) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            with self._lock:
                job.status = "running"
                job.updated_at = time.time()
            reporter = ProgressReporter(job, self._lock)
            try:
                result = fn(reporter)
                with self._lock:
                    job.status = "done"
                    job.progress = 1.0
                    job.result = result
                    job.updated_at = time.time()
            except Exception as exc:  # noqa: BLE001 - surfaced to the client, not swallowed
                with self._lock:
                    job.status = "failed"
                    job.error = f"{exc}\n{traceback.format_exc()}"
                    job.updated_at = time.time()
            finally:
                self._prune()

        threading.Thread(target=run, name=f"job-{kind}-{job.id[:8]}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self, kind: str) -> Job | None:
        """The pending-or-running job of this kind, if one exists.

        Lets a route refuse to start a second ingest. A disabled button cannot
        carry that guarantee -- a second browser tab, or a reloaded page, has
        no idea the first one is mid-run.
        """
        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.status in ("pending", "running"):
                    return job
        return None

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _prune(self) -> None:
        # Cheap bound on memory: drop the oldest finished jobs once past the
        # cap. Running/pending jobs are never pruned.
        with self._lock:
            finished = [j for j in self._jobs.values() if j.status in ("done", "failed")]
            if len(finished) <= MAX_FINISHED_JOBS:
                return
            finished.sort(key=lambda j: j.updated_at)
            for j in finished[: len(finished) - MAX_FINISHED_JOBS]:
                del self._jobs[j.id]


# Process-wide singleton -- routes import this rather than constructing their own.
runner = JobRunner()
