"""The Ingest view's backing route: run the existing pipeline as a background job.

``src.ingestion.pipeline.main`` prints and takes no arguments -- it is not
edited here (out of scope for this change) so progress is coarse: pending ->
running -> done, with the printed lines captured and parsed back into a
document/chunk count rather than lost. A real progress bar would need
``main`` to accept a progress callback, which is a pipeline.py change, not
an API one.
"""

import contextlib
import io
import re

from fastapi import APIRouter

from src.api.schemas import JobResponse
from src.ingestion.pipeline import main as run_pipeline
from src.jobs.runner import ProgressReporter, runner

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

_DOC_RE = re.compile(r"loaded (\d+) documents")
_CHUNK_RE = re.compile(r"split into (\d+) chunks")


def _run_ingest(reporter: ProgressReporter) -> dict:
    reporter.update(progress=0.05, message="starting ingest")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_pipeline()
    stdout = buffer.getvalue()

    doc_match = _DOC_RE.search(stdout)
    chunk_match = _CHUNK_RE.search(stdout)
    reporter.update(progress=1.0, message="ingest complete")
    return {
        "documents": int(doc_match.group(1)) if doc_match else None,
        "chunks": int(chunk_match.group(1)) if chunk_match else None,
        "stdout": stdout,
    }


@router.post("", response_model=JobResponse, status_code=202)
def start_ingest() -> dict:
    job = runner.submit("ingest", _run_ingest)
    return job.to_dict()
