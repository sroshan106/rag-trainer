from fastapi import HTTPException

from src.rag.model_policy import resolve_model


def validated_model(model: str | None) -> str:
    try:
        return resolve_model(model, check_installed=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
