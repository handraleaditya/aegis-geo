# AEGIS — Answer Engine & Generative Intelligence Suite

AI-powered content scoring and query fan-out engine for AEO/GEO optimization.

## Quick Start

### Prerequisites
- Python 3.11+
- A Gemini API key (free at [Google AI Studio](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone and enter the project
git clone <repo-url> && cd aegis-assignment

# Create a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm

# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Run the server
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run Tests

```bash
pytest tests/ -v
```

62 tests across 4 test files. No API key is needed — LLM calls are mocked.

---

## Endpoints

### `POST /api/aeo/analyze`

Accepts a URL or raw HTML/text and returns an AEO Readiness Score (0-100) with per-check diagnostics.

```bash
curl -X POST http://localhost:8000/api/aeo/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_type": "text", "input_value": "<h1>Title</h1><p>Python is a programming language used for web development.</p><h2>Features</h2><p>It is simple and readable. Many developers use it every day. The syntax is clean and easy to learn.</p>"}'
```

### `POST /api/fanout/generate`

Generates sub-queries for a target query and optionally analyzes content coverage gaps.

```bash
curl -X POST http://localhost:8000/api/fanout/generate \
  -H "Content-Type: application/json" \
  -d '{"target_query": "best AI writing tool for SEO", "existing_content": "Our AI writing tool uses NLP to optimize content for search engines..."}'
```

---

## What I Built

### Feature 1 — AEO Content Scorer (Complete)

Three modular, independently testable NLP checks:

- **Check A — Direct Answer Detection:** Extracts the first `<p>` tag (or first paragraph from plain text), counts words, verifies declarative sentence structure using spaCy's dependency parser (checks for `nsubj` + `ROOT` verb), and detects hedge phrases via case-insensitive substring matching.

- **Check B — H-tag Hierarchy:** Parses all heading tags in DOM order with BeautifulSoup. Validates three rules: exactly one H1, no headings before the H1, no skipped levels (e.g. H1→H3 without H2).

- **Check C — Snippet Readability:** Strips boilerplate tags (nav, footer, header, aside, script, style) before computing Flesch-Kincaid grade level via `textstat`. Identifies the 3 most complex sentences by syllable-to-word ratio.

All checks extend a `BaseCheck` abstract class with a standard `.run()` interface, making it straightforward to add new checks.

### Feature 2 — Query Fan-Out Engine (Complete)

- **LLM Integration:** Gemini 2.5 Flash via direct REST API (httpx async client).
- **Prompt Design:** Structured system prompt with explicit JSON schema, the 6 required sub-query types, a concrete example, and defensive instructions (no markdown, no extra fields).
- **Retry Logic:** Exponential backoff (1s, 2s, 4s) across 3 attempts. Handles both JSON parse errors and API failures.
- **Gap Analysis:** Sentence-level chunking + `all-MiniLM-L6-v2` embeddings with cosine similarity.

---

## Key Engineering Decisions

### 1. LLM JSON Reliability

The prompt explicitly instructs the model to return only valid JSON with no markdown or commentary. As a defense layer, `_extract_json()` strips markdown code fences if present. After parsing, `_validate_sub_queries()` enforces the schema: it checks for the `sub_queries` key, filters out entries with invalid types or extra fields, and raises if fewer than 6 valid queries remain. If parsing/validation fails, the engine retries up to 3 times with exponential backoff before returning a 503.

For production, I would add Pydantic model validation on the parsed JSON and consider using Gemini's structured output mode (JSON mode) to further reduce parse failures.

### 2. Embedding Model Choice

I chose **`all-MiniLM-L6-v2`** over `all-mpnet-base-v2`. Reasoning:

- **5x faster inference** — matters for a synchronous API endpoint where a user is waiting.
- **80MB vs 420MB model size** — faster cold start, less memory.
- **Accuracy is sufficient** for this use case. We're comparing short search queries against content chunks — the semantic gap between "covered" and "not covered" is large enough that MiniLM's slightly lower accuracy doesn't meaningfully affect classification at the 0.72 threshold.

For production, I'd use `all-mpnet-base-v2` behind an async embedding service with caching, where latency is amortized.

### 3. Similarity Threshold (0.72)

I kept the 0.72 threshold as the default. My reasoning:

- For normalized sentence-transformer embeddings, 0.72 cosine similarity represents a moderately strong semantic match — the query and content are clearly about the same topic.
- Below ~0.65 you get too many false positives (topically adjacent but not actually covering the query). Above ~0.80 you'd miss valid paraphrases.
- **How I'd tune it in production:** Build a labeled dataset of (query, content_chunk, is_covered) pairs, sweep thresholds from 0.60-0.85, and optimize for F1 score. Different content domains may need different thresholds.

The threshold is configurable via the `SIMILARITY_THRESHOLD` constant in `gap_analyzer.py`.

### 4. Content Parsing Robustness

- **No first paragraph found:** Falls back from `<p>` tag extraction to splitting on `\n\n` for plain text, then to the full text.
- **JS-rendered pages:** Returns a 422 with a descriptive error. The assignment doesn't require headless browser rendering, and adding Playwright/Selenium would be overengineering for the scope.
- **Login walls / empty HTML:** Detected via empty body text check after parsing, returns 422.
- **Boilerplate stripping:** Removes `<nav>`, `<footer>`, `<header>`, `<aside>`, `<script>`, `<style>`, `<noscript>` before readability analysis.

### 5. Failure Modes

- **LLM timeout/failure:** Returns 503 with `llm_unavailable` error after 3 retries. The error includes the last failure reason.
- **URL fetch failure:** Returns 422 with `url_fetch_failed` and the specific HTTP error or timeout message.
- **Empty content:** Returns 422 with `empty_content` error.
- **No existing content for gap analysis:** Gap fields are simply omitted from the response — sub-queries are still returned.

### 6. Async vs Sync

Endpoints are `async` because:
- URL fetching and LLM calls are I/O-bound — async lets us not block the event loop.
- The Gemini integration uses httpx `AsyncClient` for native async HTTP calls, avoiding the event loop issues that arise from wrapping synchronous SDK clients with `asyncio.to_thread()`.
- spaCy and sentence-transformers are CPU-bound, but they're fast enough per-request that they don't meaningfully block. For high throughput, I'd move embedding computation to a background worker.

---

## What I'd Improve With More Time

1. **Caching:** Cache LLM responses and embeddings for repeated queries. Redis or in-memory LRU.
2. **Async embedding:** Move sentence-transformer inference to a thread pool to avoid blocking.
3. **Integration tests:** Test the full endpoint flow with httpx + TestClient, including URL fetching with mocked responses.
4. **Configurable thresholds:** Expose similarity threshold and FK grade targets as API parameters.
5. **Structured output:** Use Gemini's JSON mode for more reliable structured responses.
6. **Better sentence splitting:** Use spaCy's sentence boundary detection instead of regex for the readability check.
7. **Rate limiting:** Add per-client rate limits on the fan-out endpoint to control LLM costs.

---

## Project Structure

```
├── ASSIGNMENT_README.md       # Original assignment brief
├── README.md                  # This file
├── PROMPT_LOG.md              # Prompt iteration log
├── requirements.txt
├── app/
│   ├── main.py                # FastAPI app
│   ├── api/
│   │   ├── aeo.py             # AEO router + endpoint
│   │   └── fanout.py          # Fan-out router + endpoint
│   ├── services/
│   │   ├── content_parser.py  # HTML fetch + parse + boilerplate strip
│   │   ├── aeo_checks/
│   │   │   ├── base.py        # BaseCheck abstract class
│   │   │   ├── direct_answer.py
│   │   │   ├── htag_hierarchy.py
│   │   │   └── readability.py
│   │   ├── fanout_engine.py   # LLM call, prompt, response parsing
│   │   └── gap_analyzer.py    # Embedding + cosine similarity logic
│   └── models/
│       └── schemas.py         # All Pydantic request/response models
└── tests/
    ├── test_direct_answer.py
    ├── test_htag_hierarchy.py
    ├── test_readability.py
    └── test_fanout_parsing.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for fan-out endpoint) | Google AI Studio API key for Gemini 2.5 Flash |
