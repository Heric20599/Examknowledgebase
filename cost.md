# AI Cost Sheet — School Knowledge Base RAG

Estimated OpenAI and Pinecone costs for **uploading one PDF chapter** and **generating one exam paper** with this app.

**Spreadsheet:** import [`cost-sheet.csv`](cost-sheet.csv) into Excel or Google Sheets for sortable cost rows.

> **Note:** The app does not return cost in API responses today. Use this sheet for planning, or check [OpenAI Usage](https://platform.openai.com/usage) and [Pinecone Billing](https://app.pinecone.io) for actual spend.

**Pricing reference:** June 2026 standard API rates (not Batch API).

---

## Default models (`.env`)

| Setting | Default value | Used for |
|---------|---------------|----------|
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | PDF upload, RAG search |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Exam generation |

---

## How the app spends tokens

### PDF upload (`POST /books/upload`)

1. PDF is split into chunks (`CHUNK_SIZE=800` tokens, `CHUNK_OVERLAP=120`).
2. Every chunk is sent to OpenAI embeddings.
3. Vectors are stored in Pinecone.

**Typical chunk count:** about **1 chunk per PDF page** (varies by text density).

### Exam generation (`POST /exam/generate`)

1. Small embedding calls (probe + one per chapter) — negligible cost.
2. Pinecone query: up to **8 chunks per chapter** (`top_k=8`).
3. One large chat completion (may **retry once** if validation fails).

**Prompt size drivers:** JSON schema, guardrails, up to 8 RAG chunks × 1200 chars each.

---

## Pricing tables

### Embeddings

| Model | Price per 1M tokens |
|-------|---------------------|
| `text-embedding-3-small` *(default)* | **$0.02** |
| `text-embedding-3-large` | $0.13 |

### Chat models (5-model comparison)

| # | Model | Input / 1M | Output / 1M | Best for |
|---|--------|------------|-------------|----------|
| 1 | `gpt-4o-mini` *(current default)* | $0.15 | $0.60 | Lowest-cost production exams |
| 2 | `gpt-4.1-nano` | $0.10 | $0.40 | Cheapest option |
| 3 | `gpt-5-mini` | $0.25 | $2.00 | Better quality, moderate cost |
| 4 | `gpt-4.1-mini` | $0.40 | $1.60 | Mid-tier quality |
| 5 | `gpt-4o` | $2.50 | $10.00 | Legacy; avoid for cost-sensitive use |

---

## Upload cost sheet (embedding only)

Formula:

```
Upload cost (USD) = (chunks × 800) / 1_000_000 × 0.02
```

| PDF size | Approx. chunks | Embed tokens | Cost (`text-embedding-3-small`) |
|----------|----------------|--------------|----------------------------------|
| Small — 30 pages | ~30 | ~24,000 | **$0.0005** |
| Medium — 125 pages | ~125 | ~100,000 | **$0.002** |
| Large — ~625 pages | ~625 | ~500,000 | **~$0.01** |

**Pinecone:** storage and upsert/query costs depend on your plan; usually small at textbook scale.

---

## Exam cost sheet — sample payloads

### Sample A — one question per type (5 questions)

Scope: `class=1`, `subject=1`, `publication=1`, `chapters=[1]`

```json
{
  "class": 1,
  "subject": 1,
  "publication": 1,
  "chapters": [1],
  "questionTypes": [
    { "numberOfQuestions": 1, "type": "MCQ", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "TOF", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "FIB", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "MTF", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "DES", "difficultyLevel": "EASY" }
  ],
  "description": "Generate exactly one question for every question type."
}
```

**Token assumptions:** ~11,000 input + ~3,500 output (1 attempt)

| Model | Input | Output | **Exam cost** |
|--------|-------|--------|---------------|
| gpt-4o-mini | $0.0017 | $0.0021 | **$0.0038** |
| gpt-4.1-nano | $0.0011 | $0.0014 | **$0.0025** |
| gpt-5-mini | $0.0028 | $0.0070 | **$0.0098** |
| gpt-4.1-mini | $0.0044 | $0.0056 | **$0.0100** |
| gpt-4o | $0.0275 | $0.0350 | **$0.0625** |

---

### Sample B — full paper (5 types × 5 difficulties = 25 questions)

```json
{
  "class": 1,
  "subject": 1,
  "publication": 1,
  "chapters": [1],
  "questionTypes": [
    { "numberOfQuestions": 1, "type": "MCQ", "difficultyLevel": "VERY_EASY" },
    { "numberOfQuestions": 1, "type": "MCQ", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "MCQ", "difficultyLevel": "MEDIUM" },
    { "numberOfQuestions": 1, "type": "MCQ", "difficultyLevel": "HARD" },
    { "numberOfQuestions": 1, "type": "MCQ", "difficultyLevel": "VERY_HARD" },
    { "numberOfQuestions": 1, "type": "TOF", "difficultyLevel": "VERY_EASY" },
    { "numberOfQuestions": 1, "type": "TOF", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "TOF", "difficultyLevel": "MEDIUM" },
    { "numberOfQuestions": 1, "type": "TOF", "difficultyLevel": "HARD" },
    { "numberOfQuestions": 1, "type": "TOF", "difficultyLevel": "VERY_HARD" },
    { "numberOfQuestions": 1, "type": "FIB", "difficultyLevel": "VERY_EASY" },
    { "numberOfQuestions": 1, "type": "FIB", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "FIB", "difficultyLevel": "MEDIUM" },
    { "numberOfQuestions": 1, "type": "FIB", "difficultyLevel": "HARD" },
    { "numberOfQuestions": 1, "type": "FIB", "difficultyLevel": "VERY_HARD" },
    { "numberOfQuestions": 1, "type": "MTF", "difficultyLevel": "VERY_EASY" },
    { "numberOfQuestions": 1, "type": "MTF", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "MTF", "difficultyLevel": "MEDIUM" },
    { "numberOfQuestions": 1, "type": "MTF", "difficultyLevel": "HARD" },
    { "numberOfQuestions": 1, "type": "MTF", "difficultyLevel": "VERY_HARD" },
    { "numberOfQuestions": 1, "type": "DES", "difficultyLevel": "VERY_EASY" },
    { "numberOfQuestions": 1, "type": "DES", "difficultyLevel": "EASY" },
    { "numberOfQuestions": 1, "type": "DES", "difficultyLevel": "MEDIUM" },
    { "numberOfQuestions": 1, "type": "DES", "difficultyLevel": "HARD" },
    { "numberOfQuestions": 1, "type": "DES", "difficultyLevel": "VERY_HARD" }
  ],
  "description": "Generate one question for every type and every difficulty level."
}
```

**Token assumptions:** ~12,000 input + ~15,000 output (1 attempt)

| Model | Input | Output | **Exam cost** |
|--------|-------|--------|---------------|
| gpt-4o-mini | $0.0018 | $0.0090 | **$0.011** |
| gpt-4.1-nano | $0.0012 | $0.0060 | **$0.007** |
| gpt-5-mini | $0.0030 | $0.0300 | **$0.033** |
| gpt-4.1-mini | $0.0048 | $0.0240 | **$0.029** |
| gpt-4o | $0.0300 | $0.1500 | **$0.180** |

> **Retry note:** The app may call the chat model twice. If validation fails on the first attempt, add roughly **+80–100%** to exam cost.

---

## Total cost — upload + one exam

### Scenario 1: Small PDF (30 pages) + Sample A (5 questions)

| Model | Upload | Exam | **Total** |
|--------|--------|------|-----------|
| gpt-4o-mini | $0.0005 | $0.0038 | **~$0.004** |
| gpt-4.1-nano | $0.0005 | $0.0025 | **~$0.003** |
| gpt-5-mini | $0.0005 | $0.0098 | **~$0.010** |
| gpt-4.1-mini | $0.0005 | $0.0100 | **~$0.011** |
| gpt-4o | $0.0005 | $0.0625 | **~$0.063** |

### Scenario 2: Large PDF (~$0.01 embed) + Sample B (25 questions)

| Model | Upload | Exam | **Total** |
|--------|--------|------|-----------|
| gpt-4o-mini | $0.010 | $0.011 | **~$0.021** |
| gpt-4.1-nano | $0.010 | $0.007 | **~$0.017** |
| gpt-5-mini | $0.010 | $0.033 | **~$0.043** |
| gpt-4.1-mini | $0.010 | $0.029 | **~$0.039** |
| gpt-4o | $0.010 | $0.180 | **~$0.190** |

This matches a common dashboard pattern: **~$0.01 upload + ~$0.01 exam ≈ $0.02** with `gpt-4o-mini` or `gpt-4.1-nano` on a medium/large PDF and a full sheet.

---

## Monthly estimates

| Usage per month | gpt-4o-mini | gpt-4.1-nano | gpt-5-mini | gpt-4.1-mini | gpt-4o |
|-----------------|-------------|--------------|------------|--------------|--------|
| 100 exams (no re-upload) | ~$1.10 | ~$0.70 | ~$3.30 | ~$2.90 | ~$18 |
| 100 PDF uploads + 100 exams | ~$3–12* | ~$2–11* | ~$7–14* | ~$6–14* | ~$28–38* |

\*Upload range depends on PDF page count per chapter.

---

## Cost formulas (copy-paste)

```
chunks          = PDF_pages × 1   (approximate)
upload_usd      = (chunks × 800 / 1_000_000) × embed_price_per_1M

exam_usd        = (input_tokens  / 1_000_000) × model_input_price
                + (output_tokens / 1_000_000) × model_output_price

total_usd       = upload_usd + exam_usd
```

**Rough token guides for exams:**

| Exam size | Input tokens | Output tokens |
|-----------|--------------|---------------|
| 5 questions (1 per type) | ~11,000 | ~3,500 |
| 25 questions (full sheet) | ~12,000 | ~15,000 |
| 50 questions (max per row) | ~13,000 | ~30,000+ |

---

## Recommendations

| Goal | Choice |
|------|--------|
| Lowest cost | `gpt-4.1-nano` + `text-embedding-3-small` |
| Best default (current) | `gpt-4o-mini` — already very cheap |
| Better quality, moderate cost | `gpt-5-mini` or `gpt-4.1-mini` (~$0.01/exam) |
| Avoid for exams | `gpt-4o` — 10–50× more expensive |
| Avoid for upload | `text-embedding-3-large` — 6.5× embedding cost vs small |

---

## Verify actual spend

1. **OpenAI:** [platform.openai.com/usage](https://platform.openai.com/usage) — filter by `text-embedding-3-small` and your chat model.
2. **Pinecone:** [app.pinecone.io](https://app.pinecone.io) — index metrics and billing.
3. **Test method:** note usage before/after one upload and one `POST /exam/generate`.

---

## Supported question types and difficulties

**Types:** `MCQ`, `TOF`, `FIB`, `MTF`, `DES`

**Difficulties:** `VERY_EASY`, `EASY`, `MEDIUM`, `HARD`, `VERY_HARD`

**MTF note:** `numberOfQuestions` on an MTF row is the **match pair count** for that section, not a separate top-level block. All MTF rows merge into **one** MTF block in the response.

---

## Final block — worst-case costing

Single reference for **maximum plausible spend per upload** and **per exam** given this app’s hard limits.

### Worst-case assumptions

| Limit | Source | Worst-case value used |
|-------|--------|------------------------|
| Max PDF size | `MAX_PDF_MB=50` | **50 MB** file |
| Chunks | 800 tokens/chunk, ~2 chunks/page on dense text | **~1,600 chunks** (~800 pages) |
| Embed tokens | 1,600 × 800 | **~1,280,000 tokens** |
| Max chapters per exam | `ExamPayload.chapters` | **40 chapters** |
| Max question rows | `questionTypes` max 32 rows | **32 rows** |
| Max questions per row | `numberOfQuestions` max 50 | **50 per row** |
| RAG context | 8 chunks/chapter × 1,200 chars | **320 chunks in prompt** |
| Chat retries | `exam_generator.py` | **2 attempts** (validation retry) |
| Exam input tokens | 40 ch × 8 chunks + schema + payload | **~110,000 tokens/attempt** |
| Exam output tokens | Model output cap (large JSON exam) | **~16,000 tokens/attempt** |
| Embedding model | Default | `text-embedding-3-small` |

> **Note:** A payload with 32×50 questions may fail or truncate in practice (output token limits). The numbers below are the **billing ceiling** if the API accepts the request and retries once.

---

### Per upload — worst case (one PDF)

Embedding cost only (same for all chat models):

| Item | Tokens | Cost |
|------|--------|------|
| **Worst-case PDF upload** | ~1,280,000 | **$0.026** |
| Pinecone upsert (plan-dependent) | 1,600 vectors | usually negligible on starter tiers |

```
Upload worst case = (1,600 chunks × 800 / 1,000,000) × $0.02 = $0.0256 ≈ $0.026
```

| If you switch embed model | Worst upload cost |
|---------------------------|-------------------|
| `text-embedding-3-small` *(default)* | **$0.026** |
| `text-embedding-3-large` | **~$0.17** |

---

### Per exam — worst case (one generate call)

Assumes: **40 chapters**, **32 question rows × 50 questions**, **2 chat attempts**, default embed model.

| Model | Embed (41 calls) | Chat input (2×110k) | Chat output (2×16k) | **Total per exam** |
|--------|------------------|---------------------|---------------------|---------------------|
| **gpt-4.1-nano** | ~$0.00001 | $0.022 | $0.013 | **~$0.035** |
| **gpt-4o-mini** *(default)* | ~$0.00001 | $0.033 | $0.019 | **~$0.052** |
| **gpt-5-mini** | ~$0.00001 | $0.055 | $0.064 | **~$0.12** |
| **gpt-4.1-mini** | ~$0.00001 | $0.088 | $0.051 | **~$0.14** |
| **gpt-4o** | ~$0.00001 | $0.55 | $0.32 | **~$0.87** |

```
Exam worst case (any model):
  input_cost  = (110,000 × 2 / 1_000_000) × input_price_per_1M
  output_cost = (16,000 × 2 / 1_000_000) × output_price_per_1M
  exam_total  = input_cost + output_cost + ~$0.00001 embed
```

---

### Combined worst case — one upload + one exam

| Model | Upload | Exam | **Grand total** |
|--------|--------|------|-----------------|
| gpt-4.1-nano | $0.026 | $0.035 | **~$0.06** |
| **gpt-4o-mini** *(default)* | $0.026 | $0.052 | **~$0.08** |
| gpt-5-mini | $0.026 | $0.12 | **~$0.15** |
| gpt-4.1-mini | $0.026 | $0.14 | **~$0.17** |
| gpt-4o | $0.026 | $0.87 | **~$0.90** |

---

### Realistic worst case (what you are more likely to see)

Typical “heavy but normal” use: **125-page PDF**, **5 chapters**, **25-question full sheet**, **1 attempt** (no retry).

| Action | Cost (`gpt-4o-mini`) |
|--------|----------------------|
| **Per upload** (125 pages, ~100k embed tokens) | **~$0.002** |
| **Per exam** (25 questions, 1 attempt) | **~$0.011** |
| **Upload + exam combined** | **~$0.013** |

If dashboard shows **~$0.01 upload + ~$0.01 exam**, you are in this realistic range — not the absolute API-max worst case.

---

### Quick reference card

| | Best case | Realistic | Worst case |
|--|-----------|-----------|------------|
| **Per upload** | $0.0005 (30 pg) | $0.002–0.01 | **$0.026** (50 MB) |
| **Per exam** (`gpt-4o-mini`) | $0.004 (5 Q) | $0.011 (25 Q) | **$0.052** (max payload + retry) |
| **Upload + exam** (`gpt-4o-mini`) | $0.004 | $0.013–0.021 | **$0.08** |

**Budget rule of thumb (gpt-4o-mini):** plan **$0.02 per exam** and **$0.01 per upload** — covers almost all real usage with headroom. Absolute ceiling per workflow is **~$0.08** (upload + exam together).

