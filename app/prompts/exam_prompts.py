import json

from app.schemas.exam import ExamResponse
from app.services.pinecone_store import metadata_publication_string


def build_exam_prompt(payload: dict, context_chunks: list[dict]) -> str:
    teacher_notes = (payload.get("description") or "").strip()
    teacher_notes_section = (
        f"Teacher instructions:\n{teacher_notes}\n\nUse these instructions to enhance the exam while staying faithful to the provided context."
        if teacher_notes
        else "Teacher instructions:\nNone provided."
    )

    # Count top-level question objects in the response.
    #   - non-MTF rows  : `numberOfQuestions` separate question objects.
    #   - all MTF rows  : ONE MTF block total; each row becomes one `sections[]` entry.
    # Total top-level blocks = sum of (n if non-MTF else 0) + (1 if any MTF row else 0).
    qts = payload.get("questionTypes") or []
    row_lines: list[str] = []
    total_blocks = 0
    has_mtf = False
    mtf_section_lines: list[str] = []
    for i, row in enumerate(qts, start=1):
        n = int(row.get("numberOfQuestions") or 0)
        t = str(row.get("type") or "").upper()
        diff = row.get("difficultyLevel")
        if t == "MTF":
            has_mtf = True
            mtf_section_lines.append(
                f"    - section difficulty={diff} with EXACTLY {n} matchPairs"
            )
            row_lines.append(
                f"  Row {i}: type=MTF difficulty={diff} numberOfQuestions={n}"
                f"  ->  one section inside the single MTF block (EXACTLY {n} pairs)"
            )
        else:
            total_blocks += n
            row_lines.append(
                f"  Row {i}: type={t} difficulty={diff} numberOfQuestions={n}"
                f"  ->  emit EXACTLY {n} separate top-level {t} questions"
            )
    if has_mtf:
        total_blocks += 1
    breakdown = "\n".join(row_lines) if row_lines else "  (none)"
    mtf_sections_breakdown = "\n".join(mtf_section_lines) if mtf_section_lines else "    (none)"
    # Kept for prompt text references that read more naturally as "questions" than "blocks".
    total_required = total_blocks

    context_lines = []
    for i, c in enumerate(context_chunks, start=1):
        md = c.get("metadata", {})
        context_lines.append(
            f"[{i}] book={md.get('book_id')} publication={metadata_publication_string(md)} "
            f"chapter={md.get('chapter')} page={md.get('page')} "
            f"text={md.get('text', '')[:1200]}"
        )
    context = "\n".join(context_lines)

    # Pin the response shape to the live Pydantic schema so it cannot drift
    # across calls. Whatever ExamResponse defines today is what the LLM must
    # emit today — automatically kept in sync with app/schemas/exam.py.
    response_schema_json = json.dumps(
        ExamResponse.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )

    return f"""
You are an expert school exam setter.
Generate questions strictly from the context.
Do not invent facts outside context.

Payload:
{payload}

{teacher_notes_section}

Context chunks:
{context}

Guardrails (DO NOT HALLUCINATE — ZERO TOLERANCE):
- Do NOT invent any fact, number, date, name, definition, formula, or quotation
  that is not present in the Context chunks above.
- Do NOT invent or rename JSON fields. Emit ONLY the fields defined by the
  Pydantic schema at the bottom of this prompt.
- Do NOT add fields the schema does not declare (no `sources`, no `explanation`,
  no `tags`, no extras). Unknown fields will be rejected.
- Do NOT change the response shape across calls. The same Pydantic schema must
  hold for every response — same field names, same nesting, same types, same
  discriminator (`type`) values.
- Do NOT fabricate `book_id`, `chapter`, or `page` references — use only values
  that appear in the Context chunks.
- Do NOT generate questions for chapters/pages not in the Context chunks.
- Do NOT output commentary, prose, code fences, markdown, or trailing text
  around the JSON. Return a single JSON object only.
- If the context is too thin for a requested question, rephrase the closest
  on-topic content from the context — never invent new content to fill a slot.

Exact question count (HARD CONSTRAINT — NO HALLUCINATION):
- The top-level `questions[]` array length MUST equal {total_blocks}.
- Per-row contract (fill each bucket EXACTLY):
{breakdown}
- For non-MTF rows (MCQ / TOF / FIB / DES): the number of top-level question
  objects whose `type` matches AND whose `difficulty` matches that row's
  `difficultyLevel` MUST equal that row's `numberOfQuestions`.
- For MTF rows: emit EXACTLY ONE top-level MTF block for ALL MTF rows combined.
  Each MTF payload row becomes one entry in `sections[]` (difficulty + matchPairs).
  Put `instruction` ONCE on the outer MTF block only — sections do NOT repeat it.
- Do NOT emit ANY question whose `(type, difficulty)` is not one of the rows
  above. If the payload has only MTF rows, emit ONLY one MTF block — do NOT add
  MCQ, TOF, FIB, or DES "to balance out" the exam. The server drops any such
  unrequested questions.
- Do NOT generate extra questions, even if the context supports more.
- Do NOT skip questions, even if the context seems thin — rephrase from context.
- Do NOT merge two requested non-MTF questions into one. Do NOT split one into two.
- `questionCode` is a single GLOBAL sequence across the whole exam: Q1, Q2, ...,
  Q{total_blocks}. No duplicates, no gaps. The MTF block gets one questionCode
  like any other top-level question.

Rules:
- Respect exact numberOfQuestions per type and difficulty (each questionTypes row: 1–50 questions).
- Ensure final JSON matches schema exactly.
- Add per-question sources with book_id, chapter, page.
- Treat `description` as teacher guidance (focus/topics/style/constraints), not as chapter content.
- If the payload lists multiple `chapters`, draw questions fairly across all of them using the provided context chunks.
- For each non-MTF question include at the top level: `questionCode` (from the global Q1..Q{total_blocks} sequence), `type`, `difficulty`, `text`, `displayOrder`.
- MCQ options must be objects with: optionLabel, text, displayOrder. You MUST also set `correctOption` to the optionLabel of the one correct choice (e.g. `"B"`).
- TOF: set boolean `answer` (true if the statement is correct, false if incorrect). Put the statement text in `text` (or legacy `statement` which is copied to `text`).
- FIB: include `text` (stem with _____ or clear blanks) and `answers` only; do not include a `blanks` field in output.
- MTF: ALL MTF PAYLOAD ROWS → EXACTLY ONE TOP-LEVEL MTF BLOCK.
  `numberOfQuestions` on each MTF row is the pair count for that row's section,
  NOT the number of top-level blocks.

  Required shape of the single MTF block (top-level keys exactly):
    type:         "MTF"
    questionCode: global Q-sequence value
    displayOrder: global sequence value
    instruction:  shared heading ONCE (e.g. "Match the following")
    sections:     one entry per MTF payload row, in payload order:
                    difficulty:  that row's `difficultyLevel`
                    matchPairs:  list of length EXACTLY that row's `numberOfQuestions`
                                 each pair = {{leftText, rightText, displayOrder, pairKey}}
                                 pairKey is "A","B","C",... restarting per section

  Sections carry ONLY `difficulty` and `matchPairs` — no instruction, no questionCode.
  Do NOT repeat `instruction` on sections or pairs.
  Do NOT emit multiple top-level MTF blocks when the payload has multiple MTF rows.

  MTF sections required for this payload:
{mtf_sections_breakdown}

  WORKED EXAMPLE — payload rows MTF EASY×2 and MTF HARD×3:
    CORRECT (1 top-level MTF block, 2 sections):
      {{
        "type": "MTF",
        "questionCode": "Q1",
        "displayOrder": 1,
        "instruction": "Match the following",
        "sections": [
          {{
            "difficulty": "EASY",
            "matchPairs": [
              {{"pairKey":"A","leftText":"...","rightText":"...","displayOrder":1}},
              {{"pairKey":"B","leftText":"...","rightText":"...","displayOrder":2}}
            ]
          }},
          {{
            "difficulty": "HARD",
            "matchPairs": [
              {{"pairKey":"A","leftText":"...","rightText":"...","displayOrder":1}},
              {{"pairKey":"B","leftText":"...","rightText":"...","displayOrder":2}},
              {{"pairKey":"C","leftText":"...","rightText":"...","displayOrder":3}}
            ]
          }}
        ]
      }}

    WRONG — multiple top-level MTF blocks (DO NOT EMIT):
      [{{ "type":"MTF", "instruction":"Match the following", ... }},
       {{ "type":"MTF", "instruction":"Match the following", ... }}]

    WRONG — instruction repeated on each section (DO NOT EMIT).
- DES: put ONLY the learner-facing prompt in `text`. Put marking content in `modelAnswer` only (if the model emits rubric bullets in `keyPoints` or `rubric`, the server merges them into `modelAnswer`; clients do not see `keyPoints`).
- At the root, include strings `summary` and `analysis` for the model only; the API merges them into `description` in this order: (1) exam summary, (2) exam analytics, (3) the request `description` / teacher instructions last. Clients only see the single `description` field.

Required Pydantic response shape (ExamResponse):
Your output MUST be a single JSON object that validates against this exact
JSON Schema. Field names, types, and nesting are fixed and identical on every
call. Do not add, rename, or omit fields.
```json
{response_schema_json}
```

Final self-check before emitting JSON:
- Output is a single JSON object — no prose, no markdown fences.
- Output validates against the ExamResponse JSON Schema above (no extra keys).
- len(top-level questions[]) == {total_blocks}.
- For each non-MTF row, count of top-level questions with matching (type, difficulty)
  equals that row's `numberOfQuestions`.
- For each MTF payload row, the matching `sections[]` entry has that row's
  `difficultyLevel` and `matchPairs` length equals that row's `numberOfQuestions`.
- There is EXACTLY ONE top-level MTF block when the payload has any MTF rows.
- Every MTF block has exactly: type, questionCode, displayOrder, instruction, sections.
  Every section has exactly: difficulty, matchPairs.
  Every matchPair has exactly: pairKey, leftText, rightText, displayOrder.
- questionCode values across the whole response are exactly Q1..Q{total_blocks}
  with no duplicates and no gaps.
- Every cited `book_id`, `chapter`, `page` appears in the Context chunks.
If any check fails, fix the output before returning. Return strictly valid JSON only.
"""
