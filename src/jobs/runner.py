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

JobStatus = Literal["pending", "running", "done", "failed", "cancelled"]

TERMINAL_STATUSES = ("done", "failed", "cancelled")

# Bound how many finished jobs stay addressable, so a long-lived dashboard
# session can't grow this dict without limit.
MAX_FINISHED_JOBS = 200


class JobAlreadyRunning(Exception):
    """Raised by ``JobRunner.submit_exclusive`` when the kind is already busy.

    A domain exception rather than an HTTP one so the runner stays free of
    FastAPI; the route decides what status code that maps to.
    """

    def __init__(self, job: "Job"):
        self.job = job
        super().__init__(f"a {job.kind} job is already running (job {job.id})")


class JobCancelled(Exception):
    """Raised by a job body that noticed its cancel event and stopped early.

    Distinct from a plain exception so the runner can mark the job cancelled
    rather than failed, and -- crucially -- keep whatever partial result the
    body already published instead of discarding it as a failure.
    """


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
    # Set by ``JobRunner.cancel``. Cooperative: nothing interrupts the thread,
    # the body is expected to poll ``ProgressReporter.cancelled`` at whatever
    # granularity it can stop cleanly at.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # The runner hands every job *its* lock, so a snapshot below is taken under
    # the same lock the worker threads mutate these fields with. Defaults to a
    # private one so a stand-alone Job is still safe to read.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict:
        # Under the lock so a poller can't catch a half-updated job -- e.g. a
        # "running" status next to the progress of the step that just finished.
        # Never call this while already holding the lock: it is not reentrant.
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
        """Publish progress, and optionally a partial result.

        ``result`` is what lets a long job be useful before it finishes: the
        body writes its running totals into ``job.result``, so a poller sees
        real numbers mid-run instead of an empty box until the very end. It is
        overwritten wholesale each call, not merged.
        """
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
    """Registry of jobs keyed by id, each executed on its own thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[ProgressReporter], Any]) -> Job:
        job = Job(id=str(uuid.uuid4()), kind=kind, lock=self._lock)
        with self._lock:
            self._jobs[job.id] = job
        return self._start(job, fn)

    def submit_exclusive(self, kind: str, fn: Callable[[ProgressReporter], Any]) -> Job:
        """Submit, but only if no job of this kind is active.

        The check and the insert happen under one lock acquisition: doing them
        as two separate calls leaves a window where two concurrent uploads both
        see "nothing running" and both start embedding.
        """
        job = Job(id=str(uuid.uuid4()), kind=kind, lock=self._lock)
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
                    # A body can stop early and still return what it has; only
                    # a None return means "keep what I already published".
                    job.status = "cancelled" if job.cancel_event.is_set() else "done"
                    if job.status == "done":
                        job.progress = 1.0
                    if result is not None:
                        job.result = result
                    job.updated_at = time.time()
            except JobCancelled:
                # Partial result already on the job stays there -- that is the
                # whole reason cancellation is not just an exception.
                with self._lock:
                    job.status = "cancelled"
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

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. False if it has already finished.

        Only sets the flag -- the status does not flip to "cancelled" until
        the body actually notices and unwinds, so a caller polling the job
        keeps seeing "running" until the stop really took effect.
        """
        # Lookup and status read share one acquisition -- reading the status
        # outside it can see a job that finished between the two.
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
        job.cancel_event.set()
        return True

    def active(self, kind: str) -> Job | None:
        """The pending-or-running job of this kind, if one exists.

        Lets a route refuse to start a second ingest. A disabled button cannot
        carry that guarantee -- a second browser tab, or a reloaded page, has
        no idea the first one is mid-run.
        """
        with self._lock:
            return self._active_locked(kind)

    def _active_locked(self, kind: str) -> Job | None:
        # Caller must already hold ``self._lock`` (it is not reentrant).
        for job in self._jobs.values():
            # A cancelled-but-still-unwinding job still counts as active:
            # its threads are alive and the GPU is still busy.
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
            finished = [j for j in self._jobs.values() if j.status in TERMINAL_STATUSES]
            if len(finished) <= MAX_FINISHED_JOBS:
                return
            finished.sort(key=lambda j: j.updated_at)
            for j in finished[: len(finished) - MAX_FINISHED_JOBS]:
                del self._jobs[j.id]


# Process-wide singleton -- routes import this rather than constructing their own.
runner = JobRunner()
