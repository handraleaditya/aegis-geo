"""Fan-out Engine — LLM prompt design, Gemini call, JSON parsing with retries.

Calls the Gemini API to decompose a user query into 10-15 sub-queries
across 6 search intent categories. Includes defensive JSON parsing,
schema validation, and exponential backoff retry logic.
"""

from __future__ import annotations

import json
import os
import re
import logging
import asyncio

import httpx

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
LLM_TIMEOUT = 30.0  # seconds per API call

VALID_TYPES = {
    "comparative",
    "feature_specific",
    "use_case",
    "trust_signals",
    "how_to",
    "definitional",
}

# ── Prompt ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a search-query decomposition engine. Your job is to simulate how an AI \
search engine (like ChatGPT Search, Perplexity, or Google AI Mode) would break \
a user's query into sub-queries to build a comprehensive answer.

You will be given a TARGET QUERY. Generate 10-15 sub-queries across exactly \
these 6 types, with AT LEAST 2 sub-queries per type:

1. "comparative" — The target vs. alternatives or competitors.
2. "feature_specific" — Focused on a specific capability or feature.
3. "use_case" — A real-world application or scenario.
4. "trust_signals" — Reviews, case studies, credibility, proof points.
5. "how_to" — Procedural, instructional, step-by-step.
6. "definitional" — Conceptual, "what is", explainer-style.

RULES:
- Return ONLY a valid JSON object. No markdown, no commentary, no extra text.
- The JSON must have a single key "sub_queries" containing an array of objects.
- Each object must have exactly two keys: "type" (one of the 6 types above) and "query" (a string).
- Do NOT add any extra fields beyond "type" and "query".
- Generate between 10 and 15 sub-queries total, with at least 2 per type.
- Make sub-queries specific and realistic — they should sound like real searches.

EXAMPLE OUTPUT for the target query "best project management software":
{
  "sub_queries": [
    {"type": "comparative", "query": "Asana vs Monday.com vs ClickUp comparison 2025"},
    {"type": "comparative", "query": "best project management software for small teams vs enterprise"},
    {"type": "feature_specific", "query": "project management software with built-in time tracking"},
    {"type": "feature_specific", "query": "project management tool with Gantt charts and dependencies"},
    {"type": "use_case", "query": "project management software for remote marketing teams"},
    {"type": "use_case", "query": "best project management tool for software development sprints"},
    {"type": "trust_signals", "query": "project management software G2 reviews 2025"},
    {"type": "trust_signals", "query": "enterprise companies using Monday.com case studies"},
    {"type": "how_to", "query": "how to set up a project management workflow for a new team"},
    {"type": "how_to", "query": "how to migrate from spreadsheets to project management software"},
    {"type": "definitional", "query": "what is project management software and why use it"},
    {"type": "definitional", "query": "project management methodology types agile waterfall kanban"}
  ]
}

Now generate sub-queries for the following target query.\
"""


def _build_prompt(target_query: str) -> str:
    return f'{SYSTEM_PROMPT}\n\nTARGET QUERY: "{target_query}"'


# ── JSON parsing & validation ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON from the model's response, handling markdown fences.

    LLMs sometimes wrap JSON in ```json ... ``` code fences despite being
    told not to. This function strips fences before parsing.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _validate_sub_queries(data: dict) -> list[dict]:
    """Validate and clean the parsed sub-queries, returning only valid ones.

    Filters out entries with unrecognized types or empty queries,
    and strips any extra fields the LLM may have hallucinated.
    """
    if "sub_queries" not in data:
        raise ValueError("Response missing 'sub_queries' key")

    raw = data["sub_queries"]
    if not isinstance(raw, list):
        raise ValueError("'sub_queries' is not a list")

    validated = []
    for item in raw:
        q_type = item.get("type", "")
        query = item.get("query", "")
        if q_type in VALID_TYPES and isinstance(query, str) and query.strip():
            # Only keep the two expected fields — strip any extras
            validated.append({"type": q_type, "query": query.strip()})

    if len(validated) < 6:
        raise ValueError(
            f"Only {len(validated)} valid sub-queries returned; need at least 6 "
            f"(one per type minimum)."
        )

    return validated


# ── Gemini API call with retries ──────────────────────────────────────────

def _get_api_key() -> str:
    """Retrieve the Gemini API key from the environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return api_key


async def _call_gemini(prompt: str, api_key: str) -> str:
    """Make an async HTTP call to the Gemini REST API and return the text response.

    Uses httpx for native async support, avoiding event loop issues
    that can arise from wrapping synchronous SDK clients.
    """
    url = GEMINI_API_URL.format(model=MODEL_NAME)
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        resp = await client.post(url, params={"key": api_key}, json=body)
        resp.raise_for_status()
        data = resp.json()

    # Extract text from Gemini response structure
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def generate_sub_queries(target_query: str) -> tuple[list[dict], str]:
    """Call Gemini to generate sub-queries. Returns (sub_queries, model_name).

    Retries up to MAX_RETRIES times with exponential backoff on parse/validation
    failures or API errors. Raises RuntimeError if all retries fail.
    """
    api_key = _get_api_key()
    prompt = _build_prompt(target_query)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = await _call_gemini(prompt, api_key)
            data = _extract_json(text)
            sub_queries = _validate_sub_queries(data)
            return sub_queries, MODEL_NAME

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d: parse/validation error — %s", attempt, MAX_RETRIES, exc
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Attempt %d/%d: LLM call error — %s", attempt, MAX_RETRIES, exc
            )

        if attempt < MAX_RETRIES:
            await asyncio.sleep(INITIAL_BACKOFF * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"Fan-out generation failed after {MAX_RETRIES} retries. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )
