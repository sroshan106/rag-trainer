"""Read access to the stored question/answer history.

Writes happen in ``src.rag.graph.ask`` -- every query records itself, whether
it came from this API or the CLI, so there is no "save" endpoint here.
"""

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import HistoryEntry
from src.rag import history

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=list[HistoryEntry])
def list_history(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return history.recent(limit)


@router.get("/{entry_id}", response_model=HistoryEntry)
def get_entry(entry_id: str) -> dict:
    entry = history.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="no such history entry")
    return entry


@router.delete("", status_code=200)
def clear_history() -> dict:
    return {"deleted": history.delete_all()}
