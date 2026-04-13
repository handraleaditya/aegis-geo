"""Query Fan-Out Generation endpoint.

Decomposes a target query into 10-15 sub-queries across 6 categories
using an LLM, and optionally performs semantic gap analysis against
provided content using sentence-transformer embeddings.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    FanoutRequest,
    FanoutResponse,
    GapSummary,
    SubQuery,
)
from app.services.fanout_engine import generate_sub_queries, VALID_TYPES

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=FanoutResponse)
async def generate(req: FanoutRequest):
    """Generate sub-queries for a target query with optional gap analysis.

    If existing_content is provided, each sub-query is checked against
    the content using cosine similarity to identify coverage gaps.
    """
    # Step 1: Generate sub-queries via LLM (with retries)
    try:
        raw_sub_queries, model_used = await generate_sub_queries(req.target_query)
    except RuntimeError as exc:
        logger.error("Fan-out generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_unavailable",
                "message": (
                    "Fan-out generation failed. The LLM returned an invalid "
                    "response after 3 retries."
                ),
                "detail": str(exc),
            },
        )

    # Step 2: Run gap analysis if content is provided
    gap_summary = None
    if req.existing_content and req.existing_content.strip():
        # Lazy import to avoid loading heavy sentence-transformers at app startup
        from app.services.gap_analyzer import analyze_gaps
        enriched = analyze_gaps(raw_sub_queries, req.existing_content)

        covered_count = sum(1 for sq in enriched if sq["covered"])
        total = len(enriched)

        # Determine which query types have at least one covered sub-query
        covered_types_set = {sq["type"] for sq in enriched if sq["covered"]}
        missing_types_set = VALID_TYPES - covered_types_set

        gap_summary = GapSummary(
            covered=covered_count,
            total=total,
            coverage_percent=round((covered_count / total) * 100) if total else 0,
            covered_types=sorted(covered_types_set),
            missing_types=sorted(missing_types_set),
        )

        sub_queries = [
            SubQuery(
                type=sq["type"],
                query=sq["query"],
                covered=sq["covered"],
                similarity_score=sq["similarity_score"],
            )
            for sq in enriched
        ]
    else:
        # No content provided — return sub-queries without gap fields
        sub_queries = [
            SubQuery(type=sq["type"], query=sq["query"])
            for sq in raw_sub_queries
        ]

    return FanoutResponse(
        target_query=req.target_query,
        model_used=model_used,
        total_sub_queries=len(sub_queries),
        sub_queries=sub_queries,
        gap_summary=gap_summary,
    )
