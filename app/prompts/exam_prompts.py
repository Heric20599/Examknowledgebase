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
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return f"""
You are an expert school exam setter known for writing outstanding, publication-quality papers.
Every question must be clear, fair, engaging, and worth asking — not generic filler.
Generate questions strictly from the context. Do not invent facts outside context.

Payload:
{payload}

{teacher_notes_section}

Context chunks:
{context}

Question quality (HIGH STANDARD — EVERY QUESTION MUST EARN ITS PLACE):
- Write questions a skilled teacher would proudly put on a real exam paper.
- Each question must test meaningful understanding of the chapter material — concepts,
  relationships, applications, definitions, processes, or cause-and-effect — not trivial
  copy-paste of a single sentence from context.
- Use clear, age-appropriate language. Avoid vague stems like "Which is correct?" or
  "What is mentioned?" — make the learner know exactly what is being asked.
- Calibrate difficulty honestly to each row's `difficultyLevel`:
    VERY_EASY / EASY  — direct recall, simple identification, one-step reasoning.
    MEDIUM            — connect two ideas, compare, classify, or apply a concept.
    HARD / VERY_HARD  — multi-step reasoning, inference, or synthesis from context.
- MCQ: write one unambiguous correct answer. Distractors must be plausible, distinct,
  and grounded in context — never silly, identical, or obviously wrong.
- TOF: statements must be precise and test a real fact or misconception from context.
- FIB: blanks must target key terms or values that matter; avoid arbitrary word removal.
- MTF: pair related but distinct items; left and right columns should require real matching
  skill, not trivial one-to-one labels.
- DES: prompts should invite structured explanation or analysis, not one-word answers.
- Prefer varied cognitive angles across the paper (define, explain, apply, compare, identify,
  reason) while staying within the requested types and difficulty rows.

Uniqueness (NO REPETITION — EACH QUESTION MUST BE DISTINCT):
- Every question in the paper must cover a DIFFERENT concept, fact, example, or angle
  from the context. No two questions may feel like the same question rephrased.
- Do NOT repeat the same stem, keyword, definition, example, or page focus across questions.
- Do NOT ask about the same named entity, formula, date, or process twice in different words.
- When multiple questions share the same `(type, difficulty)` row, spread them across
  different context chunks, subtopics, and pages — maximize coverage of the chapter material.
- MCQ options across the whole exam must not reuse the same distractor text in multiple questions.
- MTF pairs must not duplicate left/right text used elsewhere in the paper.
- Before finalizing, mentally scan all {total_required} top-level question blocks: if any two
  overlap in focus or wording, rewrite the weaker one to target fresh content from context.

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

Question order (HARD CONSTRAINT — MUST MATCH PAYLOAD):
- Emit top-level `questions[]` in EXACT `questionTypes[]` payload order.
- Process each payload row sequentially from Row 1 to Row N.
- For each non-MTF row: emit EXACTLY that row's `numberOfQuestions` consecutive
  top-level questions with matching `type` and `difficultyLevel` BEFORE moving
  to the next row.
- For MTF rows: emit ONE top-level MTF block at the position of the FIRST MTF
  row in the payload. All MTF rows become `sections[]` inside that single block,
  in payload order.
- `displayOrder` and `questionCode` must follow this final sequence:
  Q1..Q{total_blocks} left-to-right in `questions[]`.
- Do NOT group all MCQs together unless the payload lists MCQ rows first.
- Do NOT sort by type or difficulty unless the payload is already in that order.

WORKED EXAMPLE — payload:
  Row 1: MCQ EASY ×2
  Row 2: TOF MEDIUM ×1
  Row 3: FIB EASY ×1
CORRECT top-level order:
  [ MCQ(EASY), MCQ(EASY), TOF(MEDIUM), FIB(EASY) ]
WRONG:
  [ TOF(MEDIUM), MCQ(EASY), FIB(EASY), MCQ(EASY) ]

Rules:
- Respect exact numberOfQuestions per type and difficulty (each questionTypes row: 1–50 questions).
- Ensure final JSON matches schema exactly.
- Add per-question sources with book_id, chapter, page.
- Treat `description` as teacher guidance (focus/topics/style/constraints), not as chapter content.
- If the payload lists multiple `chapters`, draw questions fairly across all of them using the provided context chunks — and ensure questions from different chapters are genuinely distinct in topic.
- In `summary`, briefly note how the paper balances quality, difficulty, and topic coverage.
- In `analysis`, briefly note uniqueness choices (e.g. which subtopics were covered and how repetition was avoided).
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
- Every question is high quality: clear stem, fair difficulty, meaningful learning target.
- No two questions duplicate the same concept, fact, or phrasing — each is unique.
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
- Top-level `questions[]` order matches `questionTypes[]` payload row order
  (each non-MTF row's blocks are consecutive; one MTF block at the first MTF row).
- Every cited `book_id`, `chapter`, `page` appears in the Context chunks.
If any check fails, fix the output before returning. Return strictly valid JSON only.
"""
