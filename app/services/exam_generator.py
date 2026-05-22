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


def _infer_question_type(question: dict) -> str | None:
    if question.get("type"):
        return str(question["type"])
    if isinstance(question.get("options"), list):
        return "MCQ"
    if "statement" in question and isinstance(question.get("answer"), bool):
        return "TOF"
    if isinstance(question.get("blanks"), list):
        return "FIB"
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


def _collect_mtf_raw_pairs(question: dict) -> list[dict]:
    """Pull raw pair-shaped dicts out of any MTF representation the LLM might emit."""
    raw: list[dict] = []
    inner = question.get("questions")
    if isinstance(inner, list):
        for inner_q in inner:
            if isinstance(inner_q, dict) and isinstance(inner_q.get("matchPairs"), list):
                for p in inner_q["matchPairs"]:
                    if isinstance(p, dict):
                        raw.append(p)
    if not raw and isinstance(question.get("matchPairs"), list):
        for p in question["matchPairs"]:
            if isinstance(p, dict):
                raw.append(p)
    if not raw:
        left = question.get("leftColumn")
        right = question.get("rightColumn")
        if isinstance(left, list) and isinstance(right, list):
            size = min(len(left), len(right))
            for i in range(size):
                raw.append({"leftText": str(left[i]), "rightText": str(right[i])})
    return raw


def _normalize_mtf_block(question: dict, default_difficulty: str) -> None:
    """Restructure an MTF entry to the schema: {type, difficulty, instruction, questions[]}.

    Server-side responsibilities:
      - Derive a shared `instruction` heading (fall back to `text`/`heading`/`title`).
      - Build one inner question whose `matchPairs` holds all collected pairs.
      - Overwrite `pairKey` with server-assigned A, B, C, ... by displayOrder.
      - Migrate legacy pairs where `pairKey` was the right-side answer to `rightText`.
    """
    instruction_raw = (
        question.get("instruction")
        or question.get("text")
        or question.get("heading")
        or question.get("title")
        or ""
    )
    instruction = str(instruction_raw).strip() or "Match the items in Column A with Column B."

    raw_pairs = _collect_mtf_raw_pairs(question)

    normalized_pairs: list[dict] = []
    for idx, p in enumerate(raw_pairs, start=1):
        left_text = str(p.get("leftText") or p.get("left") or "").strip()
        right_text = str(p.get("rightText") or p.get("right") or "").strip()
        if not right_text:
            legacy_pk = str(p.get("pairKey") or "").strip()
            # Legacy shape used `pairKey` as the actual right-side answer text.
            if legacy_pk and not _looks_like_pair_label(legacy_pk):
                right_text = legacy_pk
        normalized_pairs.append(
            {
                "pairKey": _mtf_pair_label(idx),
                "leftText": left_text,
                "rightText": right_text,
                "displayOrder": idx,
            }
        )

    if not normalized_pairs:
        normalized_pairs = [
            {"pairKey": "A", "leftText": "", "rightText": "", "displayOrder": 1}
        ]

    inner_q_code = ""
    inner_display_order = 0
    inner_list = question.get("questions")
    if isinstance(inner_list, list) and inner_list and isinstance(inner_list[0], dict):
        inner_q_code = str(inner_list[0].get("questionCode") or "").strip()
        inner_display_order = int(inner_list[0].get("displayOrder") or 0)
    inner_q_code = inner_q_code or str(question.get("questionCode") or "").strip()
    inner_display_order = inner_display_order or int(question.get("displayOrder") or 1)

    difficulty = _normalize_difficulty(
        question.get("difficulty") or question.get("difficultyLevel"), default_difficulty
    )

    question.clear()
    question["type"] = "MTF"
    question["difficulty"] = difficulty
    question["instruction"] = instruction
    question["questions"] = [
        {
            "questionCode": inner_q_code or "Q?",
            "displayOrder": inner_display_order,
            "matchPairs": normalized_pairs,
        }
    ]


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


def _placeholder_non_mtf_question(qtype: str, difficulty: str) -> dict:
    """Build a minimal valid placeholder for a non-MTF type. Used to pad short
    buckets when the LLM produced fewer questions than the payload requested,
    so the response still satisfies the per-row count contract.
    """
    base: dict = {
        "type": qtype,
        "questionCode": "Q?",
        "displayOrder": 0,
        "difficulty": difficulty,
        "text": "",
    }
    if qtype == "MCQ":
        base["options"] = [
            {"optionLabel": "A", "text": "", "displayOrder": 1},
            {"optionLabel": "B", "text": "", "displayOrder": 2},
        ]
        base["correctOption"] = ""
    elif qtype == "TOF":
        base["answer"] = False
    elif qtype == "FIB":
        base["answers"] = []
    elif qtype == "DES":
        base["text"] = "(missing)"
        base["modelAnswer"] = ""
    return base


def _enforce_non_mtf_counts(
    normalized_questions: list[dict], payload: ExamPayload
) -> list[dict]:
    """Enforce per-row counts for non-MTF question types.

    For each non-MTF row in the payload, the response must contain exactly
    `numberOfQuestions` top-level questions whose `type` AND `difficulty` match.
    LLMs frequently invent question types/difficulties not requested (e.g. user
    asked MTF-only and the LLM also returned a FIB and a TOF). This pass:

      - Drops any non-MTF question whose `(type, difficulty)` is not in the payload.
      - Trims any bucket that has more entries than the payload requested.
      - Pads any short bucket with a placeholder of the correct type/difficulty.

    MTF blocks pass through untouched here — they are governed by
    `_enforce_mtf_row_to_block` upstream.
    """
    expected: dict[tuple[str, str], int] = {}
    for spec in payload.questionTypes:
        if spec.type.value == "MTF":
            continue
        key = (spec.type.value, spec.difficultyLevel.value)
        expected[key] = expected.get(key, 0) + spec.numberOfQuestions

    if not expected:
        kept = [q for q in normalized_questions if q.get("type") == "MTF"]
        dropped = len(normalized_questions) - len(kept)
        if dropped:
            logger.info(
                "Non-MTF enforcement: dropped %d non-MTF blocks (payload has no non-MTF rows)",
                dropped,
            )
        return kept

    used: dict[tuple[str, str], int] = {}
    rebuilt: list[dict] = []
    dropped_unrequested = 0
    dropped_overflow = 0
    for q in normalized_questions:
        t = str(q.get("type") or "")
        if t == "MTF":
            rebuilt.append(q)
            continue
        diff = str(q.get("difficulty") or "").strip().upper()
        key = (t, diff)
        cap = expected.get(key, 0)
        if cap <= 0:
            dropped_unrequested += 1
            continue
        seen = used.get(key, 0)
        if seen >= cap:
            dropped_overflow += 1
            continue
        used[key] = seen + 1
        rebuilt.append(q)

    padded = 0
    for key, cap in expected.items():
        qtype, diff = key
        seen = used.get(key, 0)
        for _ in range(cap - seen):
            rebuilt.append(_placeholder_non_mtf_question(qtype, diff))
            padded += 1

    if dropped_unrequested or dropped_overflow or padded:
        logger.info(
            "Non-MTF enforcement: dropped_unrequested=%d dropped_overflow=%d padded=%d",
            dropped_unrequested,
            dropped_overflow,
            padded,
        )
    return rebuilt


def _enforce_mtf_row_to_block(
    normalized_questions: list[dict], payload: ExamPayload
) -> list[dict]:
    """Enforce: 1 MTF block per MTF row in payload, each block holds exactly N pairs.

    LLMs often emit one MTF block per requested "question" (N blocks of N pairs each)
    even though the contract is "1 block whose `matchPairs` has length N". This pass
    fixes that drift server-side:

      - Pool all pair-shaped dicts from MTF blocks the LLM emitted, grouped by
        difficulty, deduped by (leftText, rightText).
      - For each MTF row in payload (in payload order), draw exactly that row's
        `numberOfQuestions` pairs from the matching difficulty's pool. Multiple
        MTF rows of the same difficulty consume the pool sequentially.
      - Re-stamp pairKey labels (A, B, C, ...) and displayOrder.
      - Excess MTF blocks the LLM produced are dropped. Missing pairs are padded
        with empty placeholders so the schema still validates.

    Non-MTF questions are passed through untouched. Relative position of MTF vs
    non-MTF blocks is preserved as best as possible (first MTF-position per
    difficulty receives the rebuilt block).
    """
    mtf_rows: list[tuple[str, int]] = [
        (spec.difficultyLevel.value, spec.numberOfQuestions)
        for spec in payload.questionTypes
        if spec.type.value == "MTF"
    ]
    if not mtf_rows:
        # No MTF expected; drop any MTF blocks the LLM emitted.
        kept = [q for q in normalized_questions if q.get("type") != "MTF"]
        if len(kept) != len(normalized_questions):
            logger.info(
                "MTF enforcement: dropped %d unexpected MTF blocks (payload has no MTF rows)",
                len(normalized_questions) - len(kept),
            )
        return kept

    pool_by_diff: dict[str, list[dict]] = {}
    instruction_by_diff: dict[str, str] = {}
    mtf_block_count = 0
    for q in normalized_questions:
        if q.get("type") != "MTF":
            continue
        mtf_block_count += 1
        diff = str(q.get("difficulty") or "")
        pool_by_diff.setdefault(diff, [])
        inst = str(q.get("instruction") or "").strip()
        if inst and diff not in instruction_by_diff:
            instruction_by_diff[diff] = inst
        for inner_q in q.get("questions") or []:
            if isinstance(inner_q, dict):
                for p in inner_q.get("matchPairs") or []:
                    if isinstance(p, dict):
                        pool_by_diff[diff].append(p)

    # Dedupe each pool by (leftText, rightText), preserve order.
    deduped_pools: dict[str, list[dict]] = {}
    for diff, pool in pool_by_diff.items():
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for p in pool:
            lt = str(p.get("leftText") or "").strip()
            rt = str(p.get("rightText") or "").strip()
            if not lt and not rt:
                continue
            key = (lt.lower(), rt.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"leftText": lt, "rightText": rt})
        deduped_pools[diff] = out

    # Build one expected block per MTF row in payload order.
    expected_blocks: list[dict] = []
    consumed_by_diff: dict[str, int] = {}
    for diff, n in mtf_rows:
        cursor = consumed_by_diff.get(diff, 0)
        pool = deduped_pools.get(diff, [])
        slice_pairs = list(pool[cursor : cursor + n])
        while len(slice_pairs) < n:
            slice_pairs.append({"leftText": "", "rightText": ""})
        consumed_by_diff[diff] = cursor + len(slice_pairs)
        normalized_pairs = [
            {
                "pairKey": _mtf_pair_label(i + 1),
                "leftText": str(p.get("leftText") or "").strip(),
                "rightText": str(p.get("rightText") or "").strip(),
                "displayOrder": i + 1,
            }
            for i, p in enumerate(slice_pairs)
        ]
        expected_blocks.append(
            {
                "type": "MTF",
                "difficulty": diff,
                "instruction": instruction_by_diff.get(
                    diff, "Match the items in Column A with Column B."
                ),
                "questions": [
                    {
                        # Global Q-code/displayOrder are stamped by the caller after this pass.
                        "questionCode": "Q?",
                        "displayOrder": 0,
                        "matchPairs": normalized_pairs,
                    }
                ],
            }
        )

    # Index expected blocks by difficulty for in-order placement.
    expected_by_diff: dict[str, list[dict]] = {}
    for blk in expected_blocks:
        expected_by_diff.setdefault(str(blk["difficulty"]), []).append(blk)

    rebuilt: list[dict] = []
    placed_by_diff: dict[str, int] = {}
    for q in normalized_questions:
        if q.get("type") != "MTF":
            rebuilt.append(q)
            continue
        diff = str(q.get("difficulty") or "")
        bucket = expected_by_diff.get(diff, [])
        used = placed_by_diff.get(diff, 0)
        if used < len(bucket):
            rebuilt.append(bucket[used])
            placed_by_diff[diff] = used + 1
        # Else: excess MTF block of this difficulty -> drop.

    # Any expected blocks not yet placed (LLM produced fewer MTF blocks of that
    # difficulty than payload required) get appended at the end.
    for diff, bucket in expected_by_diff.items():
        used = placed_by_diff.get(diff, 0)
        for blk in bucket[used:]:
            rebuilt.append(blk)

    if mtf_block_count != len(mtf_rows):
        logger.info(
            "MTF enforcement: collapsed %d LLM MTF blocks into %d (one per payload MTF row)",
            mtf_block_count,
            len(mtf_rows),
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
            # MTF lives in its own nested shape: {type, difficulty, instruction, questions[]}.
            # Top-level `questionCode` / `text` / `displayOrder` do not exist on MTF blocks;
            # they live on the inner question and are stamped in the global Q-code pass below.
            _normalize_mtf_block(q, default_difficulty)
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

    # Enforce "1 MTF block per MTF row in payload, each block has exactly N pairs".
    # This is the safety net for when the LLM emits N separate MTF blocks despite
    # the prompt telling it to emit one block with N pairs.
    normalized_questions = _enforce_mtf_row_to_block(normalized_questions, payload)

    # Enforce non-MTF per-row counts: drop unrequested (type, difficulty) buckets,
    # trim over-count buckets, pad short buckets. Mirrors the MTF enforcement so
    # the response shape always matches the payload contract exactly.
    normalized_questions = _enforce_non_mtf_counts(normalized_questions, payload)

    # Single global Q1..Qn sequence over top-level blocks. For MTF the code lives
    # on the inner question; for everything else it lives at the top.
    seq = 1
    for q in normalized_questions:
        if q.get("type") == "MTF":
            inner_list = q.get("questions") or []
            for inner_q in inner_list:
                if isinstance(inner_q, dict):
                    inner_q["questionCode"] = f"Q{seq}"
                    inner_q["displayOrder"] = seq
                    seq += 1
        else:
            q["questionCode"] = f"Q{seq}"
            q["displayOrder"] = seq
            seq += 1
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
            top_k=900,
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
        top_k=600,
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
            top_k=900,
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
