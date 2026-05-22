# School Knowledge Base RAG (FastAPI + OpenAI + Pinecone)

API to:
- Upload textbook PDFs and chunk/index into Pinecone.
- Generate exams from your payload schema using RAG.
- Delete indexed data by book or chapter.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your `OPENAI_API_KEY` and `PINECONE_API_KEY`.

## 2) Run

```bash
uvicorn app.main:app --reload
```

## 3) Upload a PDF

Multipart fields (only these, plus the file):

- `file` — PDF
- `class` — class id (integer)
- `subject` — subject id (integer)
- `chapter` — chapter number (integer; used as default chapter and in the derived `book_id`)
- `publication` — publication id (integer; stored in Pinecone metadata as string field **`publication`**)

The server derives **`book_id`** as `p{publication}-c{class}-s{subject}-h{chapter}`. Use that value as query `book_id` on **`GET /books/one`** and on **`DELETE /books`** / **`PATCH /books`**, or pass the same four integers as query params on those routes.

```bash
curl -X POST "http://127.0.0.1:8000/books/upload" ^
  -F "file=@maths10.pdf" ^
  -F "class=10" ^
  -F "subject=2" ^
  -F "chapter=1" ^
  -F "publication=1"
```

Indexed metadata uses **string** values: **`class`** (class id), **`subject`**, **`publication`**, **`book_id`**. Older vectors may still use legacy keys **`grade`**, **`book_type`**, or **`author`**; reads and RAG filters accept both. `/exam/generate` uses the same four ids as JSON integers (plus `questionTypes`).

Upload API is synchronous and returns completion details directly (no `job_id` polling).

## 4) Generate exam

JSON body (ids must match what you indexed; strings like `"1"` are accepted and coerced to integers):

- **`class`**, **`subject`** — integer or digit string.
- **`publication`** — integer or digit string (same as upload `publication`). Legacy key **`book_type`** is accepted as an alias.
- **`chapters`** — non-empty array of chapter numbers (metadata `chapter`) to include in RAG context, e.g. `[1, 2]`. For a single chapter you can still send legacy **`chapter`**: `1` (same as `"chapters": [1]`).
- **`questionTypes`**, **`description`** — unchanged.

```bash
curl -X POST "http://127.0.0.1:8000/exam/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"class\":\"1\",\"subject\":\"1\",\"book_type\":\"1\",\"chapters\":[1,2],\"questionTypes\":[{\"numberOfQuestions\":1,\"type\":\"DES\",\"difficultyLevel\":\"VERY_EASY\"},{\"numberOfQuestions\":1,\"type\":\"DES\",\"difficultyLevel\":\"EASY\"},{\"numberOfQuestions\":1,\"type\":\"DES\",\"difficultyLevel\":\"MEDIUM\"},{\"numberOfQuestions\":1,\"type\":\"DES\",\"difficultyLevel\":\"HARD\"},{\"numberOfQuestions\":1,\"type\":\"DES\",\"difficultyLevel\":\"VERY_HARD\"}],\"description\":\"Generate at least one question for every question type and every difficulty level.\"}"
```

Equivalent with integer ids and **`publication`** + legacy single **`chapter`**:

```bash
curl -X POST "http://127.0.0.1:8000/exam/generate" ^
  -H "Content-Type: application/json" ^
  -d "{\"class\":10,\"subject\":2,\"publication\":1,\"chapter\":1,\"questionTypes\":[{\"numberOfQuestions\":5,\"type\":\"MCQ\",\"difficultyLevel\":\"EASY\"}],\"description\":\"\"}"
```

If any requested chapter cannot be resolved in the index, the API returns `409` with **`missing_chapters`** and an upload hint.

Notes:
- Use the same `class`, `subject`, `publication` (or `book_type`) as upload; each entry in **`chapters`** must exist as chunk metadata for that scope.
- Response `class` / `subject` are digit strings for metadata consistency.
- **`questionTypes[].numberOfQuestions`**: each row allows **1–50** questions for that type+difficulty combination (total across rows is not capped beyond what the model returns and `totalQuestions` reflects).
- Response **`description`**: built as **exam summary** (from the model), then **exam analytics**, then your request **`description`** / teacher line **last** (single string, no separate `summary` / `analysis` keys).
- `description` is for teacher instructions (e.g., "focus numericals", "avoid tricky language", "add real-life examples").

## 5) Book API (minimal surface)

Only **two read** routes (plus upload) and **two write** routes target books. Use the same integer ids as upload everywhere.

**Derived `book_id`:** `p{publication}-c{class}-s{subject}-h{chapter}` (`app/services/book_scope.py`).

### Read

| Method | Path | Purpose |
|--------|------|--------|
| `GET` | `/books` | List books. Optional filters: `class`, `subject`, **`publication`**. Response: **`count`**, **`books`** (each: `book_id`, **`class`**, `subject`, `publication`, `chapter_count`), **`classes`**, **`subjects`**, **`publications`**. |
| `GET` | `/books/one` | One book: **`book_id`**, **`class`**, **`subject`**, **`publication`**, **`stats`**, **`chapters`**. Query **`book_id`** or **`class`**, **`subject`**, **`publication`**, **`chapter`**. |

Example `GET /books/one?...`:

```json
{
  "book_id": "p1-c6-s2-h6",
  "class": "6",
  "subject": "2",
  "publication": "1",
  "stats": { "chunks": 29, "pages": 29, "chapter_count": 1 },
  "chapters": [6]
}
```

```bash
curl "http://127.0.0.1:8000/books"
curl "http://127.0.0.1:8000/books?class=10&subject=2&publication=1"
curl "http://127.0.0.1:8000/books/one?book_id=p1-c10-s2-h1"
curl "http://127.0.0.1:8000/books/one?class=10&subject=2&publication=1&chapter=1"
```

### Write (same targeting rule as `GET /books/one`)

| Method | Path | Purpose |
|--------|------|--------|
| `PATCH` | `/books` | Metadata update. Query: `book_id` **or** `class`+`subject`+`publication`+`chapter`. JSON body: optional ints `class`, `subject`, `publication` (at least one required). |
| `DELETE` | `/books` | Delete whole book, **or** one slice of vectors if **`indexed_chapter`** is set (metadata `chapter` inside that book). Target with **`book_id`** **or** **`class`+`subject`+`publication`+`chapter`** (that `chapter` is the upload scope in `book_id`, not the delete slice). |

```bash
curl -X PATCH "http://127.0.0.1:8000/books?book_id=p1-c10-s2-h1" ^
  -H "Content-Type: application/json" ^
  -d "{\"subject\":3}"

curl -X PATCH "http://127.0.0.1:8000/books?class=10&subject=2&publication=1&chapter=1" ^
  -H "Content-Type: application/json" ^
  -d "{\"subject\":3}"

curl -X DELETE "http://127.0.0.1:8000/books?book_id=p1-c10-s2-h1"
curl -X DELETE "http://127.0.0.1:8000/books?class=10&subject=2&publication=1&chapter=1"
curl -X DELETE "http://127.0.0.1:8000/books?book_id=p1-c10-s2-h1&indexed_chapter=3"
```

## 6) Metadata format reference

Use this metadata shape while indexing chunks:

```json
{
  "book_id": "p1-c10-s2-h1",
  "publication": "1",
  "chapter": 4,
  "chapter_name": "travel and adventure",
  "chunk_index": 0,
  "class": "10",
  "page": 36,
  "subject": "2",
  "text": "…"
}
```

Important:

- New uploads store **`publication`** and **`class`** only (no `book_type`, `author`, `grade`, or `source_id`). Queries still match legacy indexes via `$or` on the new and old field names where applicable.

## 7) In-process keep-alive (built-in)

The app runs an internal `asyncio` loop (started from FastAPI's `lifespan` in `app/main.py`) that issues a `GET` against a configured URL on a fixed interval. There is **no external scheduler, no `/internal/cron` route, and no shared secret** anymore.

Configure with two env vars:

```
KEEPALIVE_URL=https://your-app.onrender.com/health
KEEPALIVE_INTERVAL_SECONDS=600
```

- Leave `KEEPALIVE_URL` blank to disable the loop entirely (no background task is spawned).
- `KEEPALIVE_INTERVAL_SECONDS` minimum is **5s** (lower values are clamped). Defaults to `600` (10 min).
- Point `KEEPALIVE_URL` at the **public** URL (not `127.0.0.1`) — only public traffic goes through Render's load balancer.
- The loop logs every ping under the `keepalive` logger; check `logs/app.log` to verify.

Caveats:

- If you run `uvicorn --workers N`, **each worker starts its own loop** (N× pings). Use a single worker or gate the loop yourself if that matters.
- Free-tier sleep on Render is not officially documented as "any inbound request resets it". A self-ping reduces cold starts in practice but is not a hard guarantee — an external pinger (cron-job.org, UptimeRobot) is still more reliable if uptime is critical.
- Public `GET /health` stays unauthenticated for uptime checks.

## 8) Notes

- **`GET /books`** returns both the **`books`** list and **`classes` / `subjects` / `publications`** for the same filtered view. Use **`count`** as the number of books (no separate `options` flag).

