"""Unified model resolution and validation policy."""

from src.rag.model_catalog import CATALOG, list_installed


class ModelNotInstalledError(ValueError):
    """Raised when the requested model is not downloaded in the local Ollama instance."""


class UnknownModelError(ValueError):
    """Raised when the requested model is not in the supported catalog."""


def resolve_model(model: str | None, check_installed: bool = False) -> str:
    """Validate and resolve model name against catalog and optionally installed list."""
    if not model:
        raise ValueError(f"model is required -- choose one of {list(CATALOG)}")

    if model not in CATALOG:
        raise UnknownModelError(f"unknown model {model!r} -- choose from {list(CATALOG)}")

    if check_installed:
        installed = list_installed()
        if model not in installed:
            if not installed:
                raise ModelNotInstalledError("no chat model downloaded -- download one in Settings first")
            raise ModelNotInstalledError(f"model {model!r} is not installed -- installed models: {installed}")

    return model
