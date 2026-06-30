from __future__ import annotations

import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from app.schemas.ingest import ChunkDocument

CHAPTER_PATTERN = re.compile(r"^\s*chapter\s+(\d+)\s*[:\-\s]*(.*)$", re.IGNORECASE)


@lru_cache(maxsize=1)
def _tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    enc = _tokenizer()
    tokens = enc.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + chunk_size]
        if not piece:
            continue
        decoded = enc.decode(piece).strip()
        if decoded:
            chunks.append(decoded)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def iter_pdf_chunk_documents(
    pdf_path: str | Path,
    *,
    book_id: str,
    class_str: str,
    subject: str,
    publication: str,
    default_chapter: int | None,
    default_chapter_name: str | None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> Iterator[ChunkDocument]:
    """Yield chunk documents page-by-page to keep peak memory low."""
    reader = PdfReader(str(pdf_path))

    chapter_num = default_chapter or 1
    chapter_name = _normalize_text(default_chapter_name or "General")
    lock_chapter = default_chapter is not None

    for page_i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        if not lock_chapter:
            first_line = page_text.splitlines()[0] if page_text.splitlines() else ""
            m = CHAPTER_PATTERN.match(first_line)
            if m:
                chapter_num = int(m.group(1))
                chapter_name = _normalize_text(m.group(2) or f"Chapter {chapter_num}")

        chunks = _chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{book_id}::ch{chapter_num}::p{page_i}::c{chunk_idx}"
            yield ChunkDocument(
                id=doc_id,
                text=chunk,
                class_str=class_str,
                subject=subject,
                book_id=book_id,
                publication=publication,
                chapter=chapter_num,
                chapter_name=chapter_name,
                page=page_i,
                chunk_index=chunk_idx,
            )


def pdf_to_chunk_documents(
    pdf_path: str | Path,
    *,
    book_id: str,
    class_str: str,
    subject: str,
    publication: str,
    default_chapter: int | None,
    default_chapter_name: str | None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[ChunkDocument]:
    return list(
        iter_pdf_chunk_documents(
            pdf_path,
            book_id=book_id,
            class_str=class_str,
            subject=subject,
            publication=publication,
            default_chapter=default_chapter,
            default_chapter_name=default_chapter_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
