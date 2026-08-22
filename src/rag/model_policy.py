from src.rag.model_catalog import CATALOG, list_installed


class ModelNotInstalledError(ValueError):
    pass


def resolve_model(model: str | None, check_installed: bool = False) -> str:
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
