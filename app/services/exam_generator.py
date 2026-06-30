from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from pinecone import Pinecone
from pydantic import ValidationError

from app.errors import ConflictError, UpstreamError
from app.prompts.exam_prompts import build_exam_prompt
from app.schemas.exam import ExamPayload, ExamResponse
from app.services.embeddings import embed_texts
from app.services.pinecone_store import (
    metadata_class_string,
    metadata_publication_string,
    pinecone_class_or_legacy_filter,
    pinecone_publication_or_legacy_filter,
    query_chunks,
)

logger = logging.getLogger(__name__)

# Cap broad Pinecone fallbacks during chapter resolution (metadata includes full chunk text).
_CHAPTER_RESOLVE_FALLBACK_TOP_K = 100


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.strip().lower())
    return " ".join(cleaned.split())


def _normalize_class_id(value: str) -> str:
    compact = "".join(ch for ch in value if ch.isdigit())
    return compact or _normalize_text(value)


def _metadata_text(md: dict, key: str) -> str:
    return str(md.get(key) or "").strip()


def _collect_unique_sources(matches: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], set[int]] = {}
    for m in matches:
        md = m.get("metadata") or {}
        book_id = md.get("book_id")
        chapter = md.get("chapter")
        page = md.get("page")
        if book_id is None or chapter is None or page is None:
            continue
        key = (str(book_id), int(chapter))
        grouped.setdefault(key, set()).add(int(page))
    return [
        {"book_id": book_id, "chapter": chapter, "pages": sorted(pages)}
        for (book_id, chapter), pages in grouped.items()
    ]


def _make_schema_strict(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("type", "object")
            node["additionalProperties"] = False
            props = node.get("properties")
            if isinstance(props, dict):
                # OpenAI strict json_schema requires required to include every property key.
                node["required"] = list(props.keys())
        for value in node.values():
            _make_schema_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_schema_strict(item)


def _count_non_strict_objects(node: Any) -> int:
    count = 0
    if isinstance(node, dict):
        is_object_like = node.get("type") == "object" or "properties" in node
        if is_object_like and node.get("additionalProperties") is not False:
            count += 1
        for value in node.values():
            count += _count_non_strict_objects(value)
    elif isinstance(node, list):
        for item in node:
            count += _count_non_strict_objects(item)
    return count


_GROUNDING_STOPWORDS = frozenset(
    {
        "about", "after", "also", "been", "being", "both", "each", "from", "have",
        "into", "more", "most", "only", "other", "same", "some", "such", "than",
        "that", "their", "them", "then", "there", "these", "they", "this", "those",
        "very", "what", "when", "where", "which", "while", "with", "would", "your",
    }
)


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in _normalize_text(text).split()
        if len(w) >= 4 and w not in _GROUNDING_STOPWORDS
    }


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", text))


def _build_context_corpus(context_matches: list[dict]) -> tuple[set[str], set[str]]:
    """Word and number sets from Pinecone chunk text used for grounding checks."""
    corpus_parts: list[str] = []
    for match in context_matches:
        md = match.get("metadata") or {}
        chunk_text = str(md.get("text") or "").strip()
        if chunk_text:
            corpus_parts.append(chunk_text)
    corpus = "\n".join(corpus_parts)
    return _content_words(corpus), _extract_numbers(corpus)


_FALLBACK_SNIPPET = "Refer to the chapter material covered in class."


def _extract_context_texts(context_matches: list[dict]) -> list[str]:
    texts: list[str] = []
    for match in context_matches:
        md = match.get("metadata") or {}
        chunk_text = str(md.get("text") or "").strip()
        if chunk_text:
            texts.append(chunk_text)
    return texts


def _context_snippets(context_matches: list[dict], max_len: int = 280) -> list[str]:
    """Question-sized lines derived from RAG chunks (used when the model leaves fields blank)."""
    snippets: list[str] = []
    for text in _extract_context_texts(context_matches):
        for part in re.split(r"(?<=[.!?])\s+", text):
            part = part.strip()
            if len(part) >= 24:
                snippets.append(part[:max_len])
        if not snippets or len(text) >= 24:
            compact = " ".join(text.split())
            if len(compact) >= 24:
                snippets.append(compact[:max_len])
    if not snippets:
        return [_FALLBACK_SNIPPET]
    return snippets


def _snippet_at(snippets: list[str], index: int) -> str:
    return snippets[index % len(snippets)]


def _is_missing_text(value: Any) -> bool:
    s = str(value or "").strip()
    return not s or s == "(missing)"


def _mcq_options_empty(question: dict) -> bool:
    opts = question.get("options")
    if not isinstance(opts, list) or not opts:
        return True
    return all(_is_missing_text(o.get("text") if isinstance(o, dict) else o) for o in opts)


def _question_has_empty_content(question: dict) -> bool:
    qtype = str(question.get("type") or "")
    if qtype == "MTF":
        if _is_missing_text(question.get("instruction")):
            return True
        for section in question.get("sections") or []:
            if not isinstance(section, dict):
                continue
            for pair in section.get("matchPairs") or []:
                if not isinstance(pair, dict):
                    continue
                if _is_missing_text(pair.get("leftText")) and _is_missing_text(pair.get("rightText")):
                    return True
        return False
    if qtype == "MCQ":
        return _is_missing_text(question.get("text")) or _mcq_options_empty(question)
    if qtype in {"TOF", "FIB", "DES"}:
        return _is_missing_text(question.get("text"))
    return _is_missing_text(question.get("text"))


def _find_empty_question_issues(questions: list[dict]) -> list[str]:
    issues: list[str] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        if _question_has_empty_content(question):
            issues.append(f"{_question_label(question)}: empty question content")
    return issues


def _fill_mcq_from_snippet(question: dict, snippet: str, opt_b: str) -> None:
    question["text"] = f"Which statement is supported by the lesson material?\n{snippet}"
    question["options"] = [
        {"optionLabel": "A", "text": snippet[:240], "displayOrder": 1},
        {"optionLabel": "B", "text": opt_b[:240], "displayOrder": 2},
    ]
    question["correctOption"] = "A"


def _fill_fib_from_snippet(question: dict, snippet: str) -> None:
    words = snippet.split()
    if len(words) >= 5:
        answer = words[len(words) // 2]
        blanked = words[: len(words) // 2] + ["_____"] + words[len(words) // 2 + 1 :]
        question["text"] = " ".join(blanked)
        question["answers"] = [answer]
    else:
        question["text"] = f"Complete: {snippet} _____"
        question["answers"] = ["answer"]


def _fill_single_question_from_context(question: dict, snippets: list[str], index: int) -> int:
    qtype = str(question.get("type") or "")
    if not _question_has_empty_content(question):
        return index
    primary = _snippet_at(snippets, index)
    secondary = _snippet_at(snippets, index + 1)
    index += 1
    if qtype == "MCQ":
        if _is_missing_text(question.get("text")):
            _fill_mcq_from_snippet(question, primary, secondary)
        elif _mcq_options_empty(question):
            _fill_mcq_from_snippet(question, primary, secondary)
        else:
            for i, opt in enumerate(question.get("options") or []):
                if isinstance(opt, dict) and _is_missing_text(opt.get("text")):
                    opt["text"] = _snippet_at(snippets, index + i)[:240]
            _normalize_mcq_correct_option(question)
    elif qtype == "TOF":
        question["text"] = primary
        question.setdefault("answer", True)
    elif qtype == "FIB":
        _fill_fib_from_snippet(question, primary)
        _normalize_fib_answers(question)
    elif qtype == "DES":
        question["text"] = f"Explain the following based on the chapter:\n{primary[:220]}"
        if _is_missing_text(question.get("modelAnswer")):
            question["modelAnswer"] = primary
    elif qtype == "MTF":
        if _is_missing_text(question.get("instruction")):
            question["instruction"] = "Match the following items from the lesson."
        for section in question.get("sections") or []:
            if not isinstance(section, dict):
                continue
            for pair in section.get("matchPairs") or []:
                if not isinstance(pair, dict):
                    continue
                if _is_missing_text(pair.get("leftText")) and _is_missing_text(pair.get("rightText")):
                    pair["leftText"] = primary[:200]
                    pair["rightText"] = secondary[:200]
                    index += 1
                    primary = _snippet_at(snippets, index)
                    secondary = _snippet_at(snippets, index + 1)
                elif _is_missing_text(pair.get("leftText")):
                    pair["leftText"] = primary[:200]
                elif _is_missing_text(pair.get("rightText")):
                    pair["rightText"] = secondary[:200]
    else:
        question["text"] = primary
    return index


def _fill_empty_questions_from_context(
    questions: list[dict], context_matches: list[dict]
) -> int:
    snippets = _context_snippets(context_matches)
    cursor = 0
    filled = 0
    for question in questions:
        if not isinstance(question, dict):
            continue
        before = _question_has_empty_content(question)
        cursor = _fill_single_question_from_context(question, snippets, cursor)
        if before and not _question_has_empty_content(question):
            filled += 1
        elif before:
            # Last resort so Pydantic/clients never see blank stems.
            fallback = _snippet_at(snippets, cursor)
            cursor += 1
            if str(question.get("type") or "") == "MCQ":
                _fill_mcq_from_snippet(question, fallback, _snippet_at(snippets, cursor))
                cursor += 1
            else:
                question["text"] = fallback
            filled += 1
    if filled:
        logger.info("Filled %d question(s) with context snippets (empty LLM fields)", filled)
    return filled


def _is_grounded_field(text: str, corpus_words: set[str], corpus_numbers: set[str]) -> bool:
    """Return True when generated text plausibly comes from the RAG context."""
    value = text.strip()
    if not value or value == "(missing)":
        return True

    for number in _extract_numbers(value):
        # Ignore single-digit ordinals/labels; flag invented multi-digit facts.
        if len(number) >= 2 and number not in corpus_numbers:
            return False

    words = _content_words(value)
    if len(words) <= 2:
        return True

    overlap = len(words & corpus_words)
    return overlap >= max(2, int(len(words) * 0.25))


def _iter_question_text_fields(question: dict):
    qtype = str(question.get("type") or "")
    if qtype == "MTF":
        yield str(question.get("instruction") or "")
        for section in question.get("sections") or []:
            if not isinstance(section, dict):
                continue
            for pair in section.get("matchPairs") or []:
                if isinstance(pair, dict):
                    yield str(pair.get("leftText") or "")
                    yield str(pair.get("rightText") or "")
        return
    if qtype == "MCQ":
        yield str(question.get("text") or "")
        for option in question.get("options") or []:
            if isinstance(option, dict):
                yield str(option.get("text") or "")
        return
    yield str(question.get("text") or "")
    if qtype == "FIB":
        for answer in question.get("answers") or []:
            yield str(answer)
    elif qtype == "DES":
        yield str(question.get("modelAnswer") or "")


def _question_label(question: dict) -> str:
    code = str(question.get("questionCode") or "").strip()
    if code:
        return code
    return str(question.get("type") or "question")


def _find_ungrounded_questions(questions: list[dict], context_matches: list[dict]) -> list[str]:
    if not context_matches:
        return []
    corpus_words, corpus_numbers = _build_context_corpus(context_matches)
    if not corpus_words and not corpus_numbers:
        return []

    issues: list[str] = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        label = _question_label(question)
        for field_text in _iter_question_text_fields(question):
            if not _is_grounded_field(field_text, corpus_words, corpus_numbers):
                preview = field_text.strip().replace("\n", " ")[:100]
                issues.append(f"{label}: not grounded in context — {preview!r}")
                break
    return issues


def _infer_question_type(question: dict) -> str | None:
    if question.get("type"):
        return str(question["type"])
    if isinstance(question.get("options"), list):
        return "MCQ"
    if "statement" in question and isinstance(question.get("answer"), bool):
        return "TOF"
    if isinstance(question.get("blanks"), list):
        return "FIB"
    if isinstance(question.get("sections"), list):
        return "MTF"
    if isinstance(question.get("matchPairs"), list):
        return "MTF"
    if "instruction" in question and isinstance(question.get("questions"), list):
        return "MTF"
    if isinstance(question.get("leftColumn"), list) and isinstance(question.get("rightColumn"), list):
        return "MTF"
    if "modelAnswer" in question or "keyPoints" in question:
        return "DES"
    return None


def _normalize_source_list(value: Any) -> list[dict]:
    raw_items: list[dict] = []
    if isinstance(value, list):
        raw_items = [v for v in value if isinstance(v, dict)]
    elif isinstance(value, dict):
        raw_items = [value]
    grouped: dict[tuple[str, int], set[int]] = {}
    for item in raw_items:
        book_id = item.get("book_id")
        chapter = item.get("chapter")
        if book_id is None or chapter is None:
            continue
        key = (str(book_id), int(chapter))
        pages = item.get("pages")
        if isinstance(pages, list):
            for p in pages:
                if isinstance(p, int):
                    grouped.setdefault(key, set()).add(p)
                elif isinstance(p, str) and p.isdigit():
                    grouped.setdefault(key, set()).add(int(p))
        elif isinstance(item.get("page"), int):
            grouped.setdefault(key, set()).add(int(item["page"]))
    return [
        {"book_id": book_id, "chapter": chapter, "pages": sorted(pages)}
        for (book_id, chapter), pages in grouped.items()
        if pages
    ]


def _normalize_mcq_options(question: dict) -> None:
    raw_options = question.get("options")
    normalized_options: list[dict] = []
    if isinstance(raw_options, list):
        for idx, opt in enumerate(raw_options, start=1):
            label = chr(64 + idx) if idx <= 26 else f"O{idx}"
            if isinstance(opt, dict):
                normalized_options.append(
                    {
                        "optionLabel": str(opt.get("optionLabel") or label),
                        "text": str(opt.get("text") or ""),
                        "displayOrder": int(opt.get("displayOrder") or idx),
                    }
                )
            else:
                normalized_options.append(
                    {
                        "optionLabel": label,
                        "text": str(opt),
                        "displayOrder": idx,
                    }
                )
    if not normalized_options:
        normalized_options = [
            {"optionLabel": "A", "text": "", "displayOrder": 1},
            {"optionLabel": "B", "text": "", "displayOrder": 2},
        ]
    question["options"] = normalized_options


def _normalize_mcq_correct_option(question: dict) -> None:
    co = question.get("correctOption")
    if isinstance(co, str) and co.strip():
        question["correctOption"] = co.strip().upper()[:8]
        return
    for key in ("correctOptionLabel", "correctAnswer", "correct"):
        v = question.get(key)
        if isinstance(v, str) and v.strip():
            question["correctOption"] = v.strip().upper()[:8]
            return
    ca = question.get("correctAnswer")
    if isinstance(ca, int) and isinstance(question.get("options"), list):
        opts = question["options"]
        if 1 <= ca <= len(opts):
            question["correctOption"] = str(opts[ca - 1].get("optionLabel") or "").strip().upper()[:8]
            return
    opts = question.get("options")
    if isinstance(opts, list):
        for o in opts:
            if not isinstance(o, dict):
                continue
            if o.get("isCorrect") is True or o.get("correct") is True:
                question["correctOption"] = str(o.get("optionLabel") or "").strip().upper()[:8]
                return
    question.setdefault("correctOption", "")


def _normalize_tof_answer(question: dict) -> None:
    a = question.get("answer")
    if isinstance(a, bool):
        return
    if isinstance(a, str):
        s = a.strip().lower()
        if s in ("true", "t", "yes", "1"):
            question["answer"] = True
        elif s in ("false", "f", "no", "0"):
            question["answer"] = False
        return
    ca = question.get("correctAnswer")
    if isinstance(ca, bool):
        question["answer"] = ca
    elif isinstance(ca, str):
        s = ca.strip().lower()
        if s in ("true", "t", "yes", "1"):
            question["answer"] = True
        elif s in ("false", "f", "no", "0"):
            question["answer"] = False


_MTF_PAIR_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _mtf_pair_label(idx_one_based: int) -> str:
    if idx_one_based <= len(_MTF_PAIR_LABELS):
        return _MTF_PAIR_LABELS[idx_one_based - 1]
    return f"O{idx_one_based}"


def _looks_like_pair_label(value: str) -> bool:
    s = (value or "").strip()
    if not s:
        return False
    if len(s) == 1 and s.isalpha():
        return True
    if s.startswith("O") and s[1:].isdigit():
        return True
    return False


def _normalize_mtf_pair_dict(raw: dict, idx_one_based: int) -> dict:
    left_text = str(raw.get("leftText") or raw.get("left") or "").strip()
    right_text = str(raw.get("rightText") or raw.get("right") or "").strip()
    if not right_text:
        legacy_pk = str(raw.get("pairKey") or "").strip()
        if legacy_pk and not _looks_like_pair_label(legacy_pk):
            right_text = legacy_pk
    return {
        "pairKey": _mtf_pair_label(idx_one_based),
        "leftText": left_text,
        "rightText": right_text,
        "displayOrder": idx_one_based,
    }


def _dedupe_mtf_pair_pool(pool: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for p in pool:
        lt = str(p.get("leftText") or p.get("left") or "").strip()
        rt = str(p.get("rightText") or p.get("right") or "").strip()
        if not lt and not rt:
            continue
        key = (lt.lower(), rt.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"leftText": lt, "rightText": rt})
    return out


def _pool_mtf_pairs_from_emitted(mtf_blocks: list[dict]) -> tuple[dict[str, list[dict]], str]:
    """Collect pairs grouped by difficulty from any MTF shape the LLM may emit."""
    pool_by_diff: dict[str, list[dict]] = {}
    instruction = ""

    for block in mtf_blocks:
        inst = str(block.get("instruction") or block.get("heading") or block.get("title") or "").strip()
        if inst and not instruction:
            instruction = inst

        block_diff = _normalize_difficulty(
            block.get("difficulty") or block.get("difficultyLevel"), "EASY"
        )

        sections = block.get("sections")
        if isinstance(sections, list) and sections:
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                diff = _normalize_difficulty(sec.get("difficulty") or block_diff, block_diff)
                pool_by_diff.setdefault(diff, [])
                for p in sec.get("matchPairs") or []:
                    if isinstance(p, dict):
                        pool_by_diff[diff].append(p)
            continue

        inner = block.get("questions")
        if isinstance(inner, list):
            for inner_q in inner:
                if isinstance(inner_q, dict) and isinstance(inner_q.get("matchPairs"), list):
                    diff = _normalize_difficulty(
                        inner_q.get("difficulty") or block_diff, block_diff
                    )
                    pool_by_diff.setdefault(diff, [])
                    for p in inner_q["matchPairs"]:
                        if isinstance(p, dict):
                            pool_by_diff[diff].append(p)
            if pool_by_diff:
                continue

        if isinstance(block.get("matchPairs"), list):
            pool_by_diff.setdefault(block_diff, [])
            for p in block["matchPairs"]:
                if isinstance(p, dict):
                    pool_by_diff[block_diff].append(p)
            continue

        left = block.get("leftColumn")
        right = block.get("rightColumn")
        if isinstance(left, list) and isinstance(right, list):
            pool_by_diff.setdefault(block_diff, [])
            size = min(len(left), len(right))
            for i in range(size):
                pool_by_diff[block_diff].append(
                    {"leftText": str(left[i]), "rightText": str(right[i])}
                )

    if not instruction:
        instruction = "Match the following"
    return pool_by_diff, instruction


def _merge_mtf_into_single_block(
    normalized_questions: list[dict],
    payload: ExamPayload,
    context_matches: list[dict] | None = None,
) -> list[dict]:
    """Merge all MTF payload rows into one top-level MTF block with `sections[]`.

    Each MTF row in the payload becomes one section (difficulty + matchPairs).
    `instruction` is set once on the outer block; sections carry pairs only.
    """
    mtf_rows: list[tuple[str, int]] = [
        (spec.difficultyLevel.value, spec.numberOfQuestions)
        for spec in payload.questionTypes
        if spec.type.value == "MTF"
    ]
    if not mtf_rows:
        kept = [q for q in normalized_questions if q.get("type") != "MTF"]
        if len(kept) != len(normalized_questions):
            logger.info(
                "MTF merge: dropped %d unexpected MTF blocks (payload has no MTF rows)",
                len(normalized_questions) - len(kept),
            )
        return kept

    mtf_emitted = [q for q in normalized_questions if q.get("type") == "MTF"]
    pool_by_diff, instruction = _pool_mtf_pairs_from_emitted(mtf_emitted)

    deduped_pools: dict[str, list[dict]] = {
        diff: _dedupe_mtf_pair_pool(pool) for diff, pool in pool_by_diff.items()
    }

    sections: list[dict] = []
    consumed_by_diff: dict[str, int] = {}
    for diff, n in mtf_rows:
        cursor = consumed_by_diff.get(diff, 0)
        pool = deduped_pools.get(diff, [])
        slice_pairs = list(pool[cursor : cursor + n])
        snippets = _context_snippets(context_matches) if context_matches else [_FALLBACK_SNIPPET]
        pad_idx = cursor + len(slice_pairs)
        while len(slice_pairs) < n:
            left = _snippet_at(snippets, pad_idx)
            right = _snippet_at(snippets, pad_idx + 1)
            slice_pairs.append({"leftText": left, "rightText": right})
            pad_idx += 2
        consumed_by_diff[diff] = cursor + len(slice_pairs)
        normalized_pairs = [
            _normalize_mtf_pair_dict(p, i + 1) for i, p in enumerate(slice_pairs)
        ]
        sections.append({"difficulty": diff, "matchPairs": normalized_pairs})

    single_block: dict = {
        "type": "MTF",
        "questionCode": "Q?",
        "displayOrder": 0,
        "instruction": instruction,
        "sections": sections,
    }

    rebuilt: list[dict] = []
    mtf_inserted = False
    for q in normalized_questions:
        if q.get("type") == "MTF":
            if not mtf_inserted:
                rebuilt.append(single_block)
                mtf_inserted = True
            continue
        rebuilt.append(q)
    if not mtf_inserted:
        rebuilt.append(single_block)

    if len(mtf_emitted) != 1:
        logger.info(
            "MTF merge: collapsed %d LLM MTF block(s) into 1 (sections=%d)",
            len(mtf_emitted),
            len(sections),
        )
    return rebuilt


def _normalize_fib_answers(question: dict) -> None:
    br = question.get("blanks")
    if not str(question.get("text") or "").strip() and br:
        if isinstance(br, list) and br and isinstance(br[0], dict):
            question["text"] = " | ".join(
                str(b.get("prompt") or b.get("label") or b.get("text") or "") for b in br
            )
        elif isinstance(br, list):
            question["text"] = " | ".join(str(b) for b in br)
    br = question.get("blanks")
    if isinstance(br, list) and br and isinstance(br[0], dict):
        stems: list[str] = []
        answers: list[str] = []
        for b in br:
            if not isinstance(b, dict):
                continue
            stems.append(str(b.get("prompt") or b.get("clue") or b.get("label") or b.get("text") or "").strip())
            answers.append(str(b.get("answer") or b.get("solution") or b.get("correct") or "").strip())
        if stems:
            question["answers"] = answers
    raw_ans = question.get("answers")
    if isinstance(raw_ans, list) and all(not isinstance(x, dict) for x in raw_ans):
        question["answers"] = [str(x) for x in raw_ans]
    elif not isinstance(raw_ans, list):
        for key in ("correctAnswers", "correct_answers", "fibAnswers", "solutions"):
            v = question.get(key)
            if isinstance(v, list) and v:
                question["answers"] = [str(x) for x in v]
                break
        else:
            sol = question.get("solution")
            if isinstance(sol, str) and sol.strip():
                question["answers"] = [sol.strip()]
            else:
                a = question.get("answer")
                if isinstance(a, list):
                    question["answers"] = [str(x) for x in a]
                elif isinstance(a, str) and a.strip():
                    question["answers"] = [a.strip()]
                else:
                    question.setdefault("answers", [])
    question.setdefault("answers", [])
    question.pop("blanks", None)


def _normalize_des_for_response(question: dict) -> None:
    ma = question.get("modelAnswer")
    if ma is None or not str(ma).strip():
        for key in ("answer", "suggestedAnswer", "model_answer", "exemplarAnswer", "markingNotes"):
            v = question.get(key)
            if isinstance(v, str) and v.strip():
                question["modelAnswer"] = v.strip()
                break
            if isinstance(v, list) and v:
                question["modelAnswer"] = "\n".join(str(x).strip() for x in v if str(x).strip()).strip()
                break
    if question.get("modelAnswer") is None:
        question["modelAnswer"] = ""
    extras: list[str] = []
    kp = question.get("keyPoints")
    if isinstance(kp, list):
        extras.extend(str(x).strip() for x in kp if isinstance(x, str) and x.strip())
    rub = question.get("rubric")
    if isinstance(rub, list):
        extras.extend(str(x).strip() for x in rub if str(x).strip())
    if extras:
        bullets = "\n".join(f"• {x}" for x in extras)
        ma = str(question.get("modelAnswer") or "").strip()
        question["modelAnswer"] = f"{ma}\n\n{bullets}".strip() if ma else bullets
    for k in ("keyPoints", "rubric"):
        question.pop(k, None)


def _normalize_difficulty(value: Any, fallback: str) -> str:
    txt = str(value or fallback).strip().upper()
    return txt if txt in {"VERY_EASY", "EASY", "MEDIUM", "HARD", "VERY_HARD"} else fallback


def _placeholder_non_mtf_question(
    qtype: str, difficulty: str, snippet: str, opt_b: str
) -> dict:
    """Pad a short bucket with context-backed content (never blank stems)."""
    base: dict = {
        "type": qtype,
        "questionCode": "Q?",
        "displayOrder": 0,
        "difficulty": difficulty,
        "text": snippet,
    }
    if qtype == "MCQ":
        _fill_mcq_from_snippet(base, snippet, opt_b)
    elif qtype == "TOF":
        base["answer"] = True
    elif qtype == "FIB":
        _fill_fib_from_snippet(base, snippet)
    elif qtype == "DES":
        base["text"] = f"Explain the following based on the chapter:\n{snippet[:220]}"
        base["modelAnswer"] = snippet
    return base


def _order_questions_by_payload(
    normalized_questions: list[dict],
    payload: ExamPayload,
    context_matches: list[dict] | None = None,
) -> list[dict]:
    """Reorder top-level `questions[]` to match `questionTypes[]` payload order.

    Walk payload rows sequentially. For each non-MTF row, emit exactly
    `numberOfQuestions` consecutive blocks with matching `(type, difficulty)`.
    Insert the single merged MTF block at the position of the first MTF row.
    Drop unrequested types, trim overflow buckets, and pad short buckets.
    """
    default_difficulty = payload.questionTypes[0].difficultyLevel.value if payload.questionTypes else "EASY"
    expected_total: dict[tuple[str, str], int] = {}
    for spec in payload.questionTypes:
        if spec.type.value == "MTF":
            continue
        key = (spec.type.value, spec.difficultyLevel.value)
        expected_total[key] = expected_total.get(key, 0) + spec.numberOfQuestions

    mtf_blocks = [q for q in normalized_questions if q.get("type") == "MTF"]
    mtf_block = mtf_blocks[0] if mtf_blocks else None
    if len(mtf_blocks) > 1:
        logger.info(
            "Payload ordering: using first of %d MTF blocks (expected 1 after merge)",
            len(mtf_blocks),
        )

    pools: dict[tuple[str, str], list[dict]] = {}
    dropped_unrequested = 0
    dropped_overflow = 0
    for q in normalized_questions:
        if q.get("type") == "MTF":
            continue
        t = str(q.get("type") or "")
        diff = _normalize_difficulty(q.get("difficulty") or q.get("difficultyLevel"), default_difficulty)
        key = (t, diff)
        cap = expected_total.get(key, 0)
        if cap <= 0:
            dropped_unrequested += 1
            continue
        bucket = pools.setdefault(key, [])
        if len(bucket) >= cap:
            dropped_overflow += 1
            continue
        bucket.append(q)

    snippets = _context_snippets(context_matches or [])
    pad_cursor = sum(len(v) for v in pools.values())
    padded = 0
    rebuilt: list[dict] = []
    mtf_inserted = False

    for spec in payload.questionTypes:
        if spec.type.value == "MTF":
            if not mtf_inserted:
                if mtf_block is not None:
                    rebuilt.append(mtf_block)
                mtf_inserted = True
            continue

        key = (spec.type.value, spec.difficultyLevel.value)
        bucket = pools.get(key, [])
        for _ in range(spec.numberOfQuestions):
            if bucket:
                rebuilt.append(bucket.pop(0))
                continue
            primary = _snippet_at(snippets, pad_cursor)
            secondary = _snippet_at(snippets, pad_cursor + 1)
            pad_cursor += 2
            rebuilt.append(
                _placeholder_non_mtf_question(spec.type.value, spec.difficultyLevel.value, primary, secondary)
            )
            padded += 1

    if not expected_total and not mtf_inserted and mtf_block is not None:
        rebuilt.append(mtf_block)

    if dropped_unrequested or dropped_overflow or padded:
        logger.info(
            "Payload ordering: dropped_unrequested=%d dropped_overflow=%d padded=%d",
            dropped_unrequested,
            dropped_overflow,
            padded,
        )
    return rebuilt


def _repair_generated_exam_data(data: dict, payload: ExamPayload, context_matches: list[dict]) -> dict:
    repaired = dict(data)
    raw_questions = repaired.get("questions")
    if not isinstance(raw_questions, list):
        raw_questions = []
    default_difficulty = payload.questionTypes[0].difficultyLevel.value if payload.questionTypes else "EASY"
    normalized_questions: list[dict] = []
    for idx, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            continue
        q = dict(item)
        q_type = _infer_question_type(q)
        if q_type:
            q["type"] = q_type
        q.pop("sources", None)
        if q.get("type") == "MTF":
            normalized_questions.append(q)
            continue
        q["questionCode"] = str(q.get("questionCode") or f"Q{idx}")
        q["text"] = str(q.get("text") or q.get("question") or q.get("statement") or q.get("prompt") or "")
        q["displayOrder"] = int(q.get("displayOrder") or idx)
        q["difficulty"] = _normalize_difficulty(q.get("difficulty") or q.get("difficultyLevel"), default_difficulty)
        if q.get("type") == "MCQ":
            _normalize_mcq_options(q)
            _normalize_mcq_correct_option(q)
        if q.get("type") == "TOF":
            _normalize_tof_answer(q)
        if q.get("type") == "FIB":
            _normalize_fib_answers(q)
        if q.get("type") == "DES":
            _normalize_des_for_response(q)
        normalized_questions.append(q)

    normalized_questions = _merge_mtf_into_single_block(
        normalized_questions, payload, context_matches
    )

    # Reorder to payload row sequence; drop unrequested buckets, trim overflow,
    # pad short buckets. MTF block is inserted at the first MTF row position.
    normalized_questions = _order_questions_by_payload(
        normalized_questions, payload, context_matches
    )

    # Single global Q1..Qn sequence over top-level blocks (MTF counts as one block).
    seq = 1
    for q in normalized_questions:
        q["questionCode"] = f"Q{seq}"
        q["displayOrder"] = seq
        seq += 1
    _fill_empty_questions_from_context(normalized_questions, context_matches)
    repaired["questions"] = normalized_questions
    repaired.pop("sources", None)
    base = str(payload.description or "").strip()
    summ = str(repaired.get("summary") or repaired.get("examSummary") or "").strip()
    ana = str(repaired.get("analysis") or repaired.get("paperAnalysis") or "").strip()
    for k in ("summary", "analysis", "examSummary", "paperAnalysis"):
        repaired.pop(k, None)
    parts: list[str] = []
    if summ:
        parts.append(f"Exam summary\n{summ}")
    if ana:
        parts.append(f"Exam analytics\n{ana}")
    if base:
        parts.append(f"Teacher instructions\n{base}")
    repaired["description"] = "\n\n".join(parts).strip()
    return repaired


def _chapter_exists(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    class_str: str,
    subject: str,
    chapter_name: str,
    publication: str | None,
) -> bool:
    parts: list[dict] = [
        pinecone_class_or_legacy_filter(class_str),
        {"subject": {"$eq": subject}},
        {"chapter_name": {"$eq": chapter_name}},
    ]
    if publication:
        parts.append(pinecone_publication_or_legacy_filter(publication))
    metadata_filter: dict = {"$and": parts} if len(parts) > 1 else parts[0]
    matches = query_chunks(
        pc=pc,
        index_name=index_name,
        vector=probe_vector,
        top_k=1,
        metadata_filter=metadata_filter,
    )
    return len(matches) > 0


def _resolve_chapter_by_number(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    class_str: str,
    subject: str,
    publication: str | None,
    chapter_number: int,
) -> dict | None:
    parts: list[dict] = [
        pinecone_class_or_legacy_filter(class_str),
        {"subject": {"$eq": subject}},
        {"chapter": {"$eq": chapter_number}},
    ]
    if publication:
        parts.append(pinecone_publication_or_legacy_filter(publication))
    metadata_filter: dict = {"$and": parts}
    matches = query_chunks(
        pc=pc,
        index_name=index_name,
        vector=probe_vector,
        top_k=1,
        metadata_filter=metadata_filter,
    )
    if not matches:
        # Fallback for metadata drift in class/subject/publication formatting.
        broad_matches = query_chunks(
            pc=pc,
            index_name=index_name,
            vector=probe_vector,
            top_k=_CHAPTER_RESOLVE_FALLBACK_TOP_K,
            metadata_filter={"chapter": {"$eq": chapter_number}},
        )
        requested_class = _normalize_class_id(class_str)
        requested_subject = _normalize_text(subject)
        requested_publication = _normalize_text(publication or "")
        for m in broad_matches:
            md = m.get("metadata") or {}
            candidate_class = _normalize_class_id(metadata_class_string(md))
            candidate_subject = _normalize_text(_metadata_text(md, "subject"))
            candidate_publication = _normalize_text(metadata_publication_string(md))
            if candidate_class != requested_class:
                continue
            if candidate_subject != requested_subject:
                continue
            if requested_publication and candidate_publication != requested_publication:
                continue
            matches = [m]
            break
        if not matches:
            return None
    md = matches[0].get("metadata") or {}
    return {
        "class": metadata_class_string(md) or class_str,
        "subject": _metadata_text(md, "subject") or subject,
        "publication": metadata_publication_string(md) or (publication or ""),
        "chapter_name": _metadata_text(md, "chapter_name") or f"Chapter {chapter_number}",
    }


def _resolve_chapter_match(
    pc: Pinecone,
    index_name: str,
    probe_vector: list[float],
    class_str: str,
    subject: str,
    publication: str | None,
    requested_chapter_name: str | int,
) -> dict | None:
    if isinstance(requested_chapter_name, int):
        return _resolve_chapter_by_number(
            pc=pc,
            index_name=index_name,
            probe_vector=probe_vector,
            class_str=class_str,
            subject=subject,
            publication=publication,
            chapter_number=requested_chapter_name,
        )
    if _chapter_exists(
        pc=pc,
        index_name=index_name,
        probe_vector=probe_vector,
        class_str=class_str,
        subject=subject,
        chapter_name=requested_chapter_name,
        publication=publication,
    ):
        return {
            "class": class_str,
            "subject": subject,
            "publication": publication or "",
            "chapter_name": requested_chapter_name,
        }

    # Fallback for metadata drift (case/spacing/punctuation or "Class 9" vs "9", etc.).
    nearby_matches = query_chunks(
        pc=pc,
        index_name=index_name,
        vector=probe_vector,
        top_k=_CHAPTER_RESOLVE_FALLBACK_TOP_K,
        metadata_filter={},
    )

    requested_normalized = _normalize_text(requested_chapter_name)
    requested_subject = _normalize_text(subject)
    requested_class = _normalize_class_id(class_str)
    requested_publication = _normalize_text(publication or "")

    strict_candidates: list[dict] = []
    loose_candidates: list[dict] = []
    for match in nearby_matches:
        md = match.get("metadata") or {}
        chapter_name = _metadata_text(md, "chapter_name")
        if not chapter_name or _normalize_text(chapter_name) != requested_normalized:
            continue

        candidate = {
            "class": metadata_class_string(md),
            "subject": _metadata_text(md, "subject"),
            "publication": metadata_publication_string(md),
            "chapter_name": chapter_name,
        }
        subject_ok = _normalize_text(candidate["subject"]) == requested_subject
        class_ok = _normalize_class_id(candidate["class"]) == requested_class
        publication_ok = requested_publication == "" or _normalize_text(candidate["publication"]) == requested_publication
        if subject_ok and class_ok and publication_ok:
            strict_candidates.append(candidate)
        else:
            loose_candidates.append(candidate)

    if not strict_candidates and not loose_candidates:
        # Try a metadata-scoped fallback to avoid semantic miss in broad nearest-neighbor retrieval.
        scoped_parts: list[dict] = [
            pinecone_class_or_legacy_filter(class_str),
            {"subject": {"$eq": subject}},
        ]
        if publication:
            scoped_parts.append(pinecone_publication_or_legacy_filter(publication))
        scoped_filter: dict = {"$and": scoped_parts}
        scoped_matches = query_chunks(
            pc=pc,
            index_name=index_name,
            vector=probe_vector,
            top_k=_CHAPTER_RESOLVE_FALLBACK_TOP_K,
            metadata_filter=scoped_filter,
        )
        requested_tokens = set(requested_normalized.split())
        for match in scoped_matches:
            md = match.get("metadata") or {}
            chapter_name = _metadata_text(md, "chapter_name")
            normalized_chapter = _normalize_text(chapter_name)
            if not chapter_name or not normalized_chapter:
                continue
            chapter_tokens = set(normalized_chapter.split())
            overlap = len(requested_tokens & chapter_tokens)
            if normalized_chapter == requested_normalized or overlap >= max(2, len(requested_tokens) - 1):
                strict_candidates.append(
                    {
                        "class": metadata_class_string(md),
                        "subject": _metadata_text(md, "subject"),
                        "publication": metadata_publication_string(md),
                        "chapter_name": chapter_name,
                    }
                )

    if strict_candidates:
        return strict_candidates[0]
    if loose_candidates:
        return loose_candidates[0]
    return None


def generate_exam(payload: ExamPayload, openai_client: OpenAI, pinecone_client: Pinecone, index_name: str, embed_model: str, chat_model: str) -> ExamResponse:
    class_str = str(payload.class_id)
    subject_str = str(payload.subject)
    publication_str = str(payload.publication)
    chapter_numbers = list(payload.chapters)

    logger.info(
        "Exam request received: class=%s subject=%s publication=%s chapters=%s question_types=%d",
        payload.class_id,
        payload.subject,
        payload.publication,
        chapter_numbers,
        len(payload.questionTypes),
    )
    ch_label = " ".join(str(c) for c in chapter_numbers)
    probe = embed_texts(
        openai_client,
        embed_model,
        [f"{subject_str} {publication_str} chapters {ch_label}"],
    )[0]

    resolved_chapters: list[dict] = []
    missing: list[int] = []
    for ch_num in chapter_numbers:
        resolved = _resolve_chapter_by_number(
            pc=pinecone_client,
            index_name=index_name,
            probe_vector=probe,
            class_str=class_str,
            subject=subject_str,
            publication=publication_str,
            chapter_number=ch_num,
        )
        if resolved is None:
            missing.append(ch_num)
        else:
            resolved_chapters.append(resolved)
    if missing:
        logger.warning(
            "Exam request missing chapters after resolution: class=%s subject=%s publication=%s missing=%s",
            payload.class_id,
            payload.subject,
            payload.publication,
            missing,
        )
        raise ConflictError(
            "Some chapters are not uploaded yet. Please upload first.",
            details={
                "missing_chapters": sorted(set(missing)),
                "hint": "POST /books/upload with the same class, subject, publication, and each chapter you reference in `chapters` (or legacy `chapter`).",
            },
        )

    context_matches: list[dict] = []
    seen_ids: set[str] = set()
    for chapter_match in resolved_chapters:
        chapter_query_vec = embed_texts(
            openai_client,
            embed_model,
            [f"{chapter_match['subject'] or subject_str} {chapter_match['chapter_name']} exam questions"],
        )[0]
        parts_pf: list[dict] = [
            pinecone_class_or_legacy_filter(chapter_match["class"]),
            {"subject": {"$eq": chapter_match["subject"]}},
            {"chapter_name": {"$eq": chapter_match["chapter_name"]}},
        ]
        pub = (chapter_match.get("publication") or "").strip()
        if pub:
            parts_pf.append(pinecone_publication_or_legacy_filter(pub))
        primary_filter = {"$and": parts_pf}
        chapter_matches = query_chunks(
            pc=pinecone_client,
            index_name=index_name,
            vector=chapter_query_vec,
            top_k=8,
            metadata_filter=primary_filter,
        )
        for m in chapter_matches:
            mid = m.get("id")
            key = str(mid) if mid is not None else None
            if key and key in seen_ids:
                continue
            if key:
                seen_ids.add(key)
            context_matches.append(m)
    logger.info("Resolved chapters=%d context_matches=%d", len(resolved_chapters), len(context_matches))

    prompt = build_exam_prompt(payload.model_dump(by_alias=True), context_matches)

    schema = ExamResponse.model_json_schema()
    _make_schema_strict(schema)
    non_strict_after_patch = _count_non_strict_objects(schema)
    logger.info("Schema strictness check: non_strict_objects=%d", non_strict_after_patch)
    for attempt in range(2):
        try:
            logger.info("Calling OpenAI for exam generation attempt=%d", attempt + 1)
            completion = openai_client.chat.completions.create(
                model=chat_model,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
                # OpenAI response_format json_schema currently rejects oneOf in nested fields.
                # We request JSON object output and enforce the full schema via Pydantic below.
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            data = json.loads(content)
            data = _repair_generated_exam_data(data, payload, context_matches)
            empty_issues = _find_empty_question_issues(data.get("questions") or [])
            if empty_issues:
                logger.warning(
                    "Empty question check failed attempt=%d issues=%d sample=%s",
                    attempt + 1,
                    len(empty_issues),
                    empty_issues[0] if empty_issues else "",
                )
                if attempt == 0:
                    prompt += (
                        "\n\nPrevious output had empty question text, options, or match pairs:\n"
                        + "\n".join(f"- {issue}" for issue in empty_issues[:12])
                        + "\n\nRegenerate with non-empty `text`, MCQ `options[].text`, "
                        "MTF `leftText`/`rightText`, and DES stems — all drawn from Context chunks."
                    )
                    continue
                _fill_empty_questions_from_context(
                    data.get("questions") or [], context_matches
                )
            grounding_issues = _find_ungrounded_questions(data.get("questions") or [], context_matches)
            if grounding_issues:
                logger.warning(
                    "Grounding check failed attempt=%d issues=%d sample=%s",
                    attempt + 1,
                    len(grounding_issues),
                    grounding_issues[0] if grounding_issues else "",
                )
                if attempt == 0:
                    prompt += (
                        "\n\nPrevious output contained content not grounded in the Context chunks:\n"
                        + "\n".join(f"- {issue}" for issue in grounding_issues[:12])
                        + "\n\nRegenerate ALL questions using ONLY facts, terms, names, and numbers "
                        "that appear in the Context chunks above. Do not invent or assume content."
                    )
                    continue
            data["generated_at"] = data.get("generated_at") or datetime.now(timezone.utc).isoformat()
            data["class"] = class_str
            data["subject"] = subject_str
            # Count top-level blocks: each MTF block is 1 regardless of pair count;
            # each non-MTF question is 1. Matches the shape returned to clients.
            data["totalQuestions"] = len(data.get("questions") or [])
            data["publication"] = str(payload.publication)
            data["chapters"] = list(payload.chapters)
            logger.info("Exam generation success attempt=%d questions=%s", attempt + 1, len(data.get("questions", [])))
            return ExamResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Exam JSON/validation error attempt=%d reason=%s", attempt + 1, str(exc))
            if attempt == 1:
                raise UpstreamError("LLM returned invalid exam format", {"reason": str(exc)}) from exc
            prompt += "\n\nPrevious output failed schema validation. Return strictly valid JSON."
        except Exception as exc:  # pragma: no cover - network path
            logger.exception("Exam generation upstream error attempt=%d", attempt + 1)
            raise UpstreamError("Exam generation failed", {"reason": str(exc)}) from exc

    raise UpstreamError("Exam generation failed unexpectedly")
