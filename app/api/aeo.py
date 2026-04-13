"""AEO Content Scorer endpoint.

Accepts a URL or raw HTML/text content, runs three NLP-based checks,
and returns an AEO Readiness Score (0-100) with per-check diagnostics.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
import httpx

from app.models.schemas import AEORequest, AEOResponse, CheckResult, InputType
from app.services.content_parser import (
    fetch_url,
    parse_html,
    get_body_text,
    get_first_paragraph,
    get_heading_tags,
)
from app.services.aeo_checks import DirectAnswerCheck, HtagHierarchyCheck, ReadabilityCheck

logger = logging.getLogger(__name__)

router = APIRouter()

# Score bands ordered by threshold (highest first) for lookup
_BANDS = [
    (85, "AEO Optimized"),
    (65, "Needs Improvement"),
    (40, "Significant Gaps"),
    (0, "Not AEO Ready"),
]


def _get_band(score: int) -> str:
    """Map a 0-100 AEO score to its classification band."""
    for threshold, label in _BANDS:
        if score >= threshold:
            return label
    return "Not AEO Ready"


@router.post("/analyze", response_model=AEOResponse)
async def analyze(req: AEORequest):
    """Analyze content for AEO readiness across three NLP checks.

    Accepts either a URL (fetched server-side) or raw HTML/text content.
    Returns a normalized 0-100 score with per-check breakdowns.
    """
    # Step 1: Retrieve raw content
    try:
        if req.input_type == InputType.url:
            raw_html = await fetch_url(req.input_value)
        else:
            raw_html = req.input_value
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "url_fetch_failed",
                "message": "Could not retrieve content from the provided URL.",
                "detail": str(exc),
            },
        )
    except (httpx.ConnectTimeout, httpx.ReadTimeout):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "url_fetch_failed",
                "message": "Could not retrieve content from the provided URL.",
                "detail": "Connection timeout after 10s",
            },
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "url_fetch_failed",
                "message": "Could not retrieve content from the provided URL.",
                "detail": str(exc),
            },
        )

    # Step 2: Parse content into structured components
    soup = parse_html(raw_html)
    first_para = get_first_paragraph(raw_html)
    headings = get_heading_tags(soup)
    body_text = get_body_text(soup)

    if not body_text.strip():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "empty_content",
                "message": "No readable text content found.",
                "detail": "The page may be JavaScript-rendered or behind a login wall.",
            },
        )

    # Step 3: Run the three independent AEO checks
    checks: list[CheckResult] = [
        DirectAnswerCheck().run(first_paragraph=first_para),
        HtagHierarchyCheck().run(headings=headings),
        ReadabilityCheck().run(body_text=body_text),
    ]

    # Step 4: Aggregate raw scores and normalize to 0-100
    raw_score = sum(c.score for c in checks)
    max_possible = sum(c.max_score for c in checks)  # 60
    aeo_score = round((raw_score / max_possible) * 100)

    return AEOResponse(
        aeo_score=aeo_score,
        band=_get_band(aeo_score),
        checks=checks,
    )
