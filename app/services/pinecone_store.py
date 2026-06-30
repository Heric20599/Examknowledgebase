from __future__ import annotations

from collections import defaultdict

from pinecone import Pinecone

from app.schemas.ingest import ChunkDocument


def metadata_publication_string(md: dict) -> str:
    """Publication id from metadata; supports legacy book_type / author keys."""
    v = md.get("publication")
    if v is None or str(v).strip() == "":
        v = md.get("book_type")
    if v is None or str(v).strip() == "":
        v = md.get("author")
    return str(v).strip() if v is not None else ""


def metadata_class_string(md: dict) -> str:
    """Class id from metadata; supports legacy Pinecone key `grade`."""
    v = md.get("class")
    if v is None or str(v).strip() == "":
        v = md.get("grade")
    return str(v).strip() if v is not None else ""


def pinecone_publication_or_legacy_filter(value: str) -> dict:
    """Match vectors indexed with publication, or older book_type / author fields."""
    return {
        "$or": [
            {"publication": {"$eq": value}},
            {"book_type": {"$eq": value}},
            {"author": {"$eq": value}},
        ]
    }


def pinecone_class_or_legacy_filter(value: str) -> dict:
    """Match vectors indexed with `class`, or older `grade` metadata."""
    return {"$or": [{"class": {"$eq": value}}, {"grade": {"$eq": value}}]}


def _index(pc: Pinecone, index_name: str):
    return pc.Index(index_name)


def _drop_chunk_text(md: dict) -> dict:
    """Return metadata without the large `text` field when callers do not need it."""
    if not md or "text" not in md:
        return md
    return {k: v for k, v in md.items() if k != "text"}


def upsert_chunks(pc: Pinecone, index_name: str, chunks: list[ChunkDocument], vectors: list[list[float]]) -> int:
    idx = _index(pc, index_name)
    payload = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        payload.append({"id": chunk.id, "values": vector, "metadata": chunk.to_metadata()})
    if payload:
        idx.upsert(vectors=payload)
    return len(payload)


def query_chunks(
    pc: Pinecone,
    index_name: str,
    vector: list[float],
    top_k: int,
    metadata_filter: dict,
) -> list[dict]:
    idx = _index(pc, index_name)
    res = idx.query(vector=vector, top_k=top_k, include_metadata=True, filter=metadata_filter)
    return res.get("matches", [])


def list_books(pc: Pinecone, index_name: str, probe_vector: list[float], top_k: int = 500) -> list[dict]:
    idx = _index(pc, index_name)
    res = idx.query(vector=probe_vector, top_k=top_k, include_metadata=True)
    grouped: dict[str, dict] = {}
    for m in res.get("matches", []):
        md = _drop_chunk_text(m.get("metadata") or {})
        book_id = md.get("book_id")
        if not book_id:
            continue
        current = grouped.get(book_id)
        if current is None:
            grouped[book_id] = {
                "book_id": book_id,
                "class": metadata_class_string(md) or None,
                "subject": md.get("subject"),
                "publication": metadata_publication_string(md) or None,
                "chapter_count": 0,
                "_chapters": set(),
            }
        chapter = md.get("chapter")
        if chapter is not None:
            grouped[book_id]["_chapters"].add(chapter)
    for v in grouped.values():
        v["chapter_count"] = len(v["_chapters"])
        del v["_chapters"]
    return list(grouped.values())


def get_book_overview(pc: Pinecone, index_name: str, probe_vector: list[float], book_id: str, top_k: int = 1000) -> dict | None:
    idx = _index(pc, index_name)
    res = idx.query(
        vector=probe_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"book_id": {"$eq": book_id}},
    )
    matches = res.get("matches", [])
    if not matches:
        return None

    chapters: dict[int, set[str]] = defaultdict(set)
    pages = set()
    sample = None
    for m in matches:
        md = _drop_chunk_text(m.get("metadata") or {})
        sample = sample or md
        chapter = md.get("chapter")
        chapter_name = md.get("chapter_name")
        if chapter is not None:
            chapter_int = int(chapter)
            if chapter_name:
                chapters[chapter_int].add(str(chapter_name))
            else:
                chapters[chapter_int].add(f"Chapter {chapter_int}")
        page = md.get("page")
        if page is not None:
            pages.add(int(page))

    return {
        "book_id": book_id,
        "class": metadata_class_string(sample) if sample else None,
        "subject": sample.get("subject") if sample else None,
        "publication": metadata_publication_string(sample) if sample else None,
        "chunks_found": len(matches),
        "page_count": len(pages),
        "chapter_count": len(chapters),
        "chapters": [
            {"chapter": ch, "chapter_names": sorted(names)}
            for ch, names in sorted(chapters.items())
        ],
    }


def list_chapters_for_book(pc: Pinecone, index_name: str, probe_vector: list[float], book_id: str, top_k: int = 700) -> list[dict]:
    idx = _index(pc, index_name)
    res = idx.query(
        vector=probe_vector,
        top_k=top_k,
        include_metadata=True,
        filter={"book_id": {"$eq": book_id}},
    )
    data = defaultdict(set)
    for m in res.get("matches", []):
        md = _drop_chunk_text(m.get("metadata") or {})
        if md.get("chapter") is not None:
            data[int(md["chapter"])].add(md.get("chapter_name", f"Chapter {md['chapter']}"))
    return [{"chapter": ch, "chapter_names": sorted(names)} for ch, names in sorted(data.items())]


def delete_book(pc: Pinecone, index_name: str, book_id: str) -> None:
    idx = _index(pc, index_name)
    idx.delete(filter={"book_id": {"$eq": book_id}})


def delete_chapter(pc: Pinecone, index_name: str, book_id: str, chapter: int) -> None:
    idx = _index(pc, index_name)
    idx.delete(filter={"book_id": {"$eq": book_id}, "chapter": {"$eq": chapter}})


def update_book_metadata(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    book_id: str,
    metadata_updates: dict,
    top_k: int = 1000,
) -> int:
    idx = _index(pc, index_name)
    res = idx.query(
        vector=probe_vector,
        top_k=top_k,
        include_metadata=False,
        filter={"book_id": {"$eq": book_id}},
    )
    matches = res.get("matches", [])
    for m in matches:
        vector_id = m.get("id")
        if vector_id:
            idx.update(id=vector_id, set_metadata=metadata_updates)
    return len(matches)
