from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile  # Form used by /upload

from app.schemas.ingest import BatchChapterResult, BatchUploadResponse
from app.services.batch_upload import prepare_batch_upload
from app.services.ingest_book import ingest_pdf_from_path

router = APIRouter(prefix="/books", tags=["ingest"])

_STREAM_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def _temp_dir() -> Path:
    temp_dir = Path("tmp_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


async def _stream_upload_to_path(upload: UploadFile, dest: Path, *, max_bytes: int) -> int:
    """Write upload to disk in chunks; raise HTTPException 413 if over max_bytes."""
    written = 0
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload too large. Max allowed is {max_bytes // (1024 * 1024)} MB",
                )
            fh.write(chunk)
    return written


@router.post("/upload")
async def upload_book(
    request: Request,
    file: UploadFile = File(...),
    subject: int = Form(...),
    class_id: int = Form(..., alias="class"),
    chapter: int = Form(...),
    publication: int = Form(...),
):
    settings = request.app.state.settings
    max_bytes = settings.max_pdf_mb * 1024 * 1024
    temp_path = _temp_dir() / f"{uuid.uuid4()}-{file.filename}"

    try:
        await _stream_upload_to_path(file, temp_path, max_bytes=max_bytes)
        return ingest_pdf_from_path(
            temp_path,
            publication=publication,
            class_id=class_id,
            subject=subject,
            chapter=chapter,
            settings=settings,
            openai_client=request.app.state.openai,
            pinecone_client=request.app.state.pinecone,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(str(exc).strip() or "Upload processing failed due to an internal error."),
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/upload-zip", response_model=BatchUploadResponse)
async def upload_books_zip(
    request: Request,
    file: UploadFile = File(
        ...,
        description=(
            "ZIP named classid_publicationid_subjectid.zip (e.g. 1_1_1.zip) "
            "containing chapter PDFs named 1.pdf, 2.pdf, etc."
        ),
    ),
):
    """Upload many chapter PDFs from one ZIP. IDs come from the ZIP filename; chapters from PDF names."""
    settings = request.app.state.settings
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file.")

    original_filename = Path(file.filename).name
    zip_path = _temp_dir() / f"{uuid.uuid4()}-{original_filename}"
    extract_dir: Path | None = None
    max_bytes = settings.max_zip_mb * 1024 * 1024

    try:
        await _stream_upload_to_path(file, zip_path, max_bytes=max_bytes)
    except HTTPException:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        raise

    try:
        scope, chapter_items, extract_dir = prepare_batch_upload(
            zip_path,
            original_filename=original_filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted ZIP file.") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(str(exc).strip() or "Could not read ZIP contents."),
        ) from exc
    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)

    results: list[BatchChapterResult] = []
    successful = 0
    failed = 0

    try:
        for item in chapter_items:
            try:
                outcome = ingest_pdf_from_path(
                    item.pdf_path,
                    publication=scope.publication,
                    class_id=scope.class_id,
                    subject=scope.subject,
                    chapter=item.chapter,
                    settings=settings,
                    openai_client=request.app.state.openai,
                    pinecone_client=request.app.state.pinecone,
                )
                successful += 1
                results.append(
                    BatchChapterResult(
                        file=item.file,
                        chapter=item.chapter,
                        status="completed",
                        book_id=outcome["book_id"],
                        chunks_upserted=outcome["chunks_upserted"],
                    )
                )
            except ValueError as exc:
                failed += 1
                results.append(
                    BatchChapterResult(
                        file=item.file,
                        chapter=item.chapter,
                        status="failed",
                        error=str(exc),
                    )
                )
            except Exception as exc:
                failed += 1
                results.append(
                    BatchChapterResult(
                        file=item.file,
                        chapter=item.chapter,
                        status="failed",
                        error=(str(exc).strip() or "Upload processing failed."),
                    )
                )
    finally:
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)

    total = len(chapter_items)
    if failed == 0:
        status = "completed"
        message = f"Batch upload completed. {successful} chapter(s) indexed."
    elif successful == 0:
        status = "failed"
        message = f"Batch upload failed. No chapters were indexed ({failed} error(s))."
    else:
        status = "partial"
        message = f"Batch upload partially completed. {successful} succeeded, {failed} failed."

    return BatchUploadResponse(
        status=status,
        message=message,
        publication=scope.publication,
        class_id=scope.class_id,
        subject=scope.subject,
        total_chapters=total,
        successful=successful,
        failed=failed,
        results=results,
    )
