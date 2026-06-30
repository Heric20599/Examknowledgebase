"""Chunk models for PDF ingest.

Matches ``POST /books/upload`` (``app/routers/ingest.py``): multipart fields ``class``,
``subject``, ``chapter``, ``publication`` (integers) plus derived ``book_id`` from
``scoped_book_id``. Indexed metadata uses string ids for ``class``, ``subject``, and
``publication``; ``chapter`` is an integer per chunk.
"""

from pydantic import BaseModel, ConfigDict, Field


class ChunkDocument(BaseModel):
    """One vector chunk after PDF split; metadata is written to Pinecone via ``to_metadata``."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    text: str
    class_str: str = Field(
        description="Digit string for multipart field `class` (same int as upload); Pinecone metadata key `class`.",
    )
    subject: str = Field(
        description="Digit string for multipart `subject` (same int as upload).",
    )
    book_id: str = Field(
        description="Stable id from ``scoped_book_id(publication, class, subject, chapter)`` for this upload scope.",
    )
    publication: str = Field(
        description="Digit string for multipart `publication` (same int as upload); Pinecone key `publication`.",
    )
    chapter: int = Field(
        description="Chapter index in chunk metadata; default comes from multipart `chapter`, or detected from PDF headings.",
    )
    chapter_name: str
    page: int
    chunk_index: int

    def to_metadata(self) -> dict:
        return {
            "class": self.class_str,
            "subject": self.subject,
            "book_id": self.book_id,
            "publication": self.publication,
            "chapter": self.chapter,
            "chapter_name": self.chapter_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


class BatchChapterResult(BaseModel):
    """Per-chapter outcome from ``POST /books/upload-zip``."""

    file: str
    chapter: int
    status: str
    book_id: str | None = None
    chunks_upserted: int | None = None
    error: str | None = None


class BatchUploadResponse(BaseModel):
    """Response for ``POST /books/upload-zip``."""

    model_config = ConfigDict(populate_by_name=True)

    status: str
    message: str
    publication: int
    class_id: int = Field(alias="class")
    subject: int
    total_chapters: int
    successful: int
    failed: int
    results: list[BatchChapterResult]
