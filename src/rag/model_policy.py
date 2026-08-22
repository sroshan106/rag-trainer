"""Single point of truth for "is this model allowed, and is it usable?"

Two questions, deliberately separate. ``CATALOG`` membership is a static fact
about what this app supports; being installed is a live fact about the local
Ollama instance. The graph only needs the first (it is a caller error to name a
model this app never supports), while the HTTP routes need both -- a catalog
model that was never pulled cannot answer a query.
"""

from src.rag.model_catalog import CATALOG, list_installed


class ModelNotInstalledError(ValueError):
    """In the catalog, but not pulled into the local Ollama instance."""


def resolve_model(model: str | None, check_installed: bool = False) -> str:
    """Return ``model`` if it is usable, else raise ``ValueError``.

    Absent and unrecognised collapse to the same error on purpose: from the
    caller's side both mean "name one of these", and there is no default to
    fall back to.
    """
    if not model or model not in CATALOG:
        raise ValueError(f"model is required -- choose one of {list(CATALOG)}")

    if check_installed:
        installed = list_installed()
        if model not in installed:
            if not installed:
                raise ModelNotInstalledError(
                    "no chat model downloaded -- download one in Settings first"
                )
            raise ModelNotInstalledError(
                f"unknown model {model!r} -- choose from {installed}"
            )

    return model
