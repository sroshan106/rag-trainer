"""Turn a file's units into LangChain Documents.

The unit boundaries and their numbering come from ``src.ingestion.units`` --
this module only decides what metadata rides along with each one so a retrieved
chunk can be traced back to the exact row, line, or page it came from.
"""

from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.units import (  # noqa: F401 - re-exported for callers
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
) -> list[Document]:
    """Load ``path`` into Documents carrying their own provenance.

    ``file_id`` and ``filename`` identify the upload this came from. Both are
    optional: ``files.record`` returns None when the provenance write fails,
    and the CLI ingests a path that was never uploaded at all. A missing id
    costs the citation its link, not the ingest.
    """
    path = Path(path)
    display_name = filename or path.name

    return [
        Document(
            page_content=unit.text,
            metadata={
                "file_id": file_id,
                "filename": display_name,
                "unit_kind": unit.kind,
                # Where the record sits in the file -- what a citation seeks to.
                "unit_index": unit.index,
                # What the dataset calls the record, when it says. This is what
                # an evaluation answer sheet refers to, and it is usually NOT
                # the position: bioasq passage 31776899 is row 37946. Null for
                # files that declare no identifier of their own.
                "row_index": unit.key,
                # Only set when the row carried a link of its own, in which
                # case a citation can point at the original document instead of
                # our stored copy.
                "url": unit.url,
            },
        )
        for unit in iter_units(path)
    ]
