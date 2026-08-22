from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.units import (
    SUPPORTED_EXTENSIONS,
    UnreadableFile,
    UnsupportedFileType,
    is_supported,
    iter_units,
)


def load_documents(
    path: str | Path,
    file_id: str | None = None,
    filename: str | None = None,
    index_columns: list[str] | None = None,
    citation_columns: list[str] | None = None,
) -> list[Document]:
    path = Path(path)
    display_name = filename or path.name

    return [
        Document(
            page_content=unit.text,
            metadata={
                "file_id": file_id,
                "filename": display_name,
                "unit_kind": unit.kind,
                "unit_index": unit.index,
                "row_index": unit.key,
                "url": unit.url,
                "index_columns": index_columns,
                "citation_fields": unit.fields,
            },
        )
        for unit in iter_units(
            path, index_columns=index_columns, citation_columns=citation_columns
        )
    ]
