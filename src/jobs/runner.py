"""In-process background job registry for long-running requests."""

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

JobStatus = Literal["pending", "running", "done", "failed", "cancelled"]

TERMINAL_STATUSES = ("done", "failed", "cancelled")
MAX_FINISHED_JOBS = 200
MAX_CONCURRENT_JOBS = 8


class JobAlreadyRunning(Exception):
    """Raised by JobRunner.submit_exclusive when the kind is already busy."""

    def __init__(self, job: "Job"):
        self.job = job
        super().__init__(f"a {job.kind} job is already running (job {job.id})")


class JobCancelled(Exception):
    """Raised by a job body that noticed its cancel event and stopped early."""


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = "pending"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    params: dict | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Set by ``JobRunner.cancel``. Cooperative: nothing interrupts the thread,
    # the body is expected to poll ``ProgressReporter.cancelled`` at whatever
    # granularity it can stop cleanly at.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # The runner hands every job *its* lock, so a snapshot below is taken under
    # the same lock the worker threads mutate these fields with. Defaults to a
    # private one so a stand-alone Job is still safe to read.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "params": self.params,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProgressReporter:
    """Passed into a job body so it can report progress without knowing about Job."""

    def __init__(self, job: "Job", lock: threading.Lock):
        self._job = job
        self._lock = lock

    def update(
        self,
        progress: float | None = None,
        message: str | None = None,
        result: Any = None,
    ) -> None:
        """Publish progress and optional partial result."""
        with self._lock:
            if progress is not None:
                self._job.progress = max(0.0, min(1.0, progress))
            if message is not None:
                self._job.message = message
            if result is not None:
                self._job.result = result
            self._job.updated_at = time.time()

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled()


class JobRunner:
    """Registry of jobs keyed by id, executed on a managed worker pool."""

    def __init__(self, max_workers: int = MAX_CONCURRENT_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job-worker")

    def submit(
        self,
        kind: str,
        fn: Callable[[ProgressReporter], Any],
        params: dict | None = None,
    ) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind, params=params, lock=self._lock)
        with self._lock:
            self._jobs[job.id] = job
        return self._start(job, fn)

    def submit_exclusive(
        self,
        kind: str,
        fn: Callable[[ProgressReporter], Any],
        params: dict | None = None,
    ) -> Job:
        """Submit, but only if no job of this kind is active."""
        job = Job(id=str(uuid.uuid4()), kind=kind, params=params, lock=self._lock)
        with self._lock:
            running = self._active_locked(kind)
            if running is not None:
                raise JobAlreadyRunning(running)
            self._jobs[job.id] = job
        return self._start(job, fn)

    def _start(self, job: Job, fn: Callable[[ProgressReporter], Any]) -> Job:
        kind = job.kind

        def run() -> None:
            with self._lock:
                job.status = "running"
                job.updated_at = time.time()
            reporter = ProgressReporter(job, self._lock)
            try:
                result = fn(reporter)
                with self._lock:
                    job.status = "cancelled" if job.cancel_event.is_set() else "done"
                    if job.status == "done":
                        job.progress = 1.0
                    if result is not None:
                        job.result = result
                    job.updated_at = time.time()
            except JobCancelled:
                with self._lock:
                    job.status = "cancelled"
                    job.updated_at = time.time()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    job.status = "failed"
                    job.error = f"{exc}\n{traceback.format_exc()}"
                    job.updated_at = time.time()
            finally:
                self._prune()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. False if it has already finished."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
        job.cancel_event.set()
        return True

    def active(self, kind: str) -> Job | None:
        """The pending-or-running job of this kind, if one exists."""
        with self._lock:
            return self._active_locked(kind)

    def _active_locked(self, kind: str) -> Job | None:
        for job in self._jobs.values():
            if job.kind == kind and job.status in ("pending", "running"):
                return job
        return None

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def _prune(self) -> None:
        with self._lock:
            finished = [j for j in self._jobs.values() if j.status in TERMINAL_STATUSES]
            if len(finished) <= MAX_FINISHED_JOBS:
                return
            finished.sort(key=lambda j: j.updated_at)
            for j in finished[: len(finished) - MAX_FINISHED_JOBS]:
                del self._jobs[j.id]


# Process-wide singleton -- routes import this rather than constructing their own.
runner = JobRunner()
