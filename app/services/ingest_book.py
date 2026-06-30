from __future__ import annotations

from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone

from app.config import Settings
from app.schemas.ingest import ChunkDocument
from app.services.book_scope import scoped_book_id
from app.services.embeddings import embed_texts
from app.services.pdf_loader import pdf_to_chunk_documents
from app.services.pinecone_store import upsert_chunks


def ingest_pdf_from_path(
    pdf_path: Path,
    *,
    publication: int,
    class_id: int,
    subject: int,
    chapter: int,
    settings: Settings,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
) -> dict:
    """Chunk, embed, and upsert one PDF. Returns the same shape as POST /books/upload."""
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.max_pdf_mb:
        raise ValueError(f"PDF too large ({size_mb:.1f} MB). Max allowed is {settings.max_pdf_mb} MB")

    book_id = scoped_book_id(
        publication=publication,
        class_id=class_id,
        subject=subject,
        chapter=chapter,
    )
    docs: list[ChunkDocument] = pdf_to_chunk_documents(
        pdf_path,
        book_id=book_id,
        class_str=str(class_id),
        subject=str(subject),
        publication=str(publication),
        default_chapter=chapter,
        default_chapter_name=f"Chapter {chapter}",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vectors = embed_texts(openai_client, settings.openai_embed_model, [d.text for d in docs])
    chunks_upserted = upsert_chunks(pinecone_client, settings.pinecone_index, docs, vectors)
    return {
        "status": "completed",
        "message": "Book uploaded and indexed successfully.",
        "book_id": book_id,
        "publication": publication,
        "class": class_id,
        "subject": subject,
        "chapter": chapter,
        "chunks_upserted": chunks_upserted,
    }
