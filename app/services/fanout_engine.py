"""Fan-out Engine — LLM prompt design, Gemini call, JSON parsing with retries."""

from __future__ import annotations

import json
import os
import re
import logging
import asyncio

from google import genai

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-1.5-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds

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
    """Extract JSON from the model's response, handling markdown fences."""
    # Strip markdown code fences if present
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _validate_sub_queries(data: dict) -> list[dict]:
    """Validate and clean the parsed sub-queries, returning only valid ones."""
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
            # Only keep the two expected fields
            validated.append({"type": q_type, "query": query.strip()})

    if len(validated) < 6:
        raise ValueError(
            f"Only {len(validated)} valid sub-queries returned; need at least 6 "
            f"(one per type minimum)."
        )

    return validated


# ── Gemini call with retries ──────────────────────────────────────────────

def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


async def generate_sub_queries(target_query: str) -> tuple[list[dict], str]:
    """Call Gemini to generate sub-queries. Returns (sub_queries, model_name).

    Retries up to MAX_RETRIES times with exponential backoff on failure.
    Raises RuntimeError if all retries fail.
    """
    client = _get_client()
    prompt = _build_prompt(target_query)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=MODEL_NAME,
                contents=prompt,
            )
            text = response.text
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
