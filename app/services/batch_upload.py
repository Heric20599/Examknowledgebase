from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CHAPTER_FILENAME_PATTERNS = (
    re.compile(r"chapter[_\-\s]?(\d+)", re.IGNORECASE),
    re.compile(r"ch[_\-\s]?(\d+)", re.IGNORECASE),
    re.compile(r"^(\d+)$", re.IGNORECASE),
)
ZIP_SCOPE_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)$")


@dataclass(frozen=True)
class ChapterUploadItem:
    file: str
    chapter: int
    pdf_path: Path


@dataclass(frozen=True)
class BatchScope:
    class_id: int
    subject: int
    publication: int


def _safe_member_path(name: str, dest_dir: Path) -> Path | None:
    """Resolve a zip member to dest_dir; reject path traversal."""
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    target = (dest_dir / Path(*relative.parts)).resolve()
    if not str(target).startswith(str(dest_dir.resolve())):
        return None
    return target


def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract zip members into dest_dir. Returns paths of extracted PDF files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = _safe_member_path(info.filename, dest_dir)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if target.suffix.lower() == ".pdf":
                pdf_paths.append(target)
    return pdf_paths


def chapter_from_filename(filename: str) -> int | None:
    stem = Path(filename).stem
    for pattern in CHAPTER_FILENAME_PATTERNS:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))
    return None


def _load_manifest_json(extract_dir: Path) -> dict | None:
    for name in ("manifest.json", "Manifest.json"):
        path = extract_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    for path in extract_dir.rglob("manifest.json"):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def scope_from_zip_filename(filename: str) -> BatchScope | None:
    """Parse classid_publicationid_subjectid from the uploaded ZIP name."""
    stem = Path(filename).stem
    match = ZIP_SCOPE_PATTERN.fullmatch(stem)
    if not match:
        return None
    class_id, publication, subject = (int(value) for value in match.groups())
    return BatchScope(class_id=class_id, subject=subject, publication=publication)


def _resolve_scope(
    manifest: dict | None,
    *,
    original_filename: str,
) -> BatchScope:
    zip_scope = scope_from_zip_filename(original_filename)
    if zip_scope is not None:
        return zip_scope

    if manifest is not None:
        resolved_class = manifest.get("class")
        resolved_subject = manifest.get("subject")
        resolved_publication = manifest.get("publication")
        if resolved_class is not None and resolved_subject is not None and resolved_publication is not None:
            return BatchScope(
                class_id=int(resolved_class),
                subject=int(resolved_subject),
                publication=int(resolved_publication),
            )

    raise ValueError(
        "ZIP must be named classid_publicationid_subjectid.zip (e.g. 1_1_1.zip). "
        "Inside, use chapter PDFs named 1.pdf, 2.pdf, 3.pdf, etc."
    )


def _chapter_items_from_manifest(manifest: dict, extract_dir: Path) -> list[ChapterUploadItem]:
    raw_chapters = manifest.get("chapters")
    if not isinstance(raw_chapters, list) or not raw_chapters:
        raise ValueError("manifest.json must include a non-empty chapters array.")

    items: list[ChapterUploadItem] = []
    seen_chapters: set[int] = set()
    for entry in raw_chapters:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest chapters entry must be an object with file and chapter.")
        file_name = entry.get("file")
        chapter = entry.get("chapter")
        if not file_name or chapter is None:
            raise ValueError("Each manifest chapters entry needs file (string) and chapter (integer).")
        chapter_int = int(chapter)
        if chapter_int in seen_chapters:
            raise ValueError(f"Duplicate chapter number in manifest: {chapter_int}")
        seen_chapters.add(chapter_int)

        pdf_path = _find_pdf(extract_dir, str(file_name))
        if pdf_path is None:
            raise ValueError(f"PDF not found in ZIP: {file_name}")
        items.append(ChapterUploadItem(file=str(file_name), chapter=chapter_int, pdf_path=pdf_path))
    return sorted(items, key=lambda item: item.chapter)


def _chapter_items_from_filenames(pdf_paths: list[Path]) -> list[ChapterUploadItem]:
    items: list[ChapterUploadItem] = []
    seen_chapters: set[int] = set()
    for pdf_path in sorted(pdf_paths, key=lambda p: p.name.lower()):
        chapter = chapter_from_filename(pdf_path.name)
        if chapter is None:
            raise ValueError(
                f"Cannot detect chapter from filename: {pdf_path.name}. "
                "Name PDFs as 1.pdf, 2.pdf, 3.pdf, etc."
            )
        if chapter in seen_chapters:
            raise ValueError(f"Duplicate chapter number from filenames: {chapter}")
        seen_chapters.add(chapter)
        items.append(ChapterUploadItem(file=pdf_path.name, chapter=chapter, pdf_path=pdf_path))
    if not items:
        raise ValueError("ZIP contains no PDF files.")
    return items


def _find_pdf(extract_dir: Path, file_name: str) -> Path | None:
    direct = extract_dir / file_name
    if direct.is_file():
        return direct
    normalized = file_name.replace("\\", "/").lstrip("/")
    candidate = extract_dir / normalized
    if candidate.is_file():
        return candidate
    matches = list(extract_dir.rglob(Path(normalized).name))
    if len(matches) == 1:
        return matches[0]
    return None


def prepare_batch_upload(
    zip_path: Path,
    *,
    original_filename: str,
) -> tuple[BatchScope, list[ChapterUploadItem], Path]:
    """Extract ZIP and return scope + chapter list. Caller must remove extract_dir when done."""
    extract_dir = zip_path.parent / f"extract-{uuid.uuid4().hex}"
    try:
        pdf_paths = extract_zip(zip_path, extract_dir)
        manifest = _load_manifest_json(extract_dir)
        scope = _resolve_scope(manifest, original_filename=original_filename)
        if manifest is not None and manifest.get("chapters"):
            items = _chapter_items_from_manifest(manifest, extract_dir)
        else:
            items = _chapter_items_from_filenames(pdf_paths)
        return scope, items, extract_dir
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise
