"""Pydantic request/response models for the AEGIS API.

Defines typed request and response schemas for both the AEO Content Scorer
and Query Fan-Out Engine endpoints, ensuring strict validation at the API boundary.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Shared enums ──────────────────────────────────────────────────────────


class InputType(str, Enum):
    """Supported input formats for the AEO analyzer."""
    url = "url"
    text = "text"


class ScoreBand(str, Enum):
    """AEO readiness score classification bands."""
    optimized = "AEO Optimized"
    needs_improvement = "Needs Improvement"
    significant_gaps = "Significant Gaps"
    not_ready = "Not AEO Ready"


# ── AEO request / response ───────────────────────────────────────────────


class AEORequest(BaseModel):
    """Request body for POST /api/aeo/analyze."""
    input_type: InputType
    input_value: str = Field(
        ..., min_length=1, description="URL or raw HTML/text content to analyze"
    )


class CheckResult(BaseModel):
    """Result of a single AEO check with score, details, and recommendation."""
    check_id: str
    name: str
    passed: bool
    score: int = Field(..., ge=0, le=20)
    max_score: int = 20
    details: dict
    recommendation: Optional[str] = None


class AEOResponse(BaseModel):
    """Response body for POST /api/aeo/analyze."""
    aeo_score: int = Field(..., ge=0, le=100, description="Normalized AEO readiness score")
    band: str = Field(..., description="Score classification band")
    checks: list[CheckResult]


class AEOError(BaseModel):
    """Error response for AEO analysis failures."""
    error: str
    message: str
    detail: Optional[str] = None


# ── Fan-out request / response ────────────────────────────────────────────


class SubQueryType(str, Enum):
    """The 6 sub-query categories used by AI search engines to decompose queries."""
    comparative = "comparative"
    feature_specific = "feature_specific"
    use_case = "use_case"
    trust_signals = "trust_signals"
    how_to = "how_to"
    definitional = "definitional"


class SubQuery(BaseModel):
    """A single generated sub-query with optional gap analysis fields."""
    type: SubQueryType
    query: str
    covered: Optional[bool] = Field(
        None, description="Whether existing content covers this sub-query (only present with gap analysis)"
    )
    similarity_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Max cosine similarity against content chunks"
    )


class GapSummary(BaseModel):
    """Aggregate gap analysis summary across all sub-queries."""
    covered: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    coverage_percent: int = Field(..., ge=0, le=100)
    covered_types: list[str]
    missing_types: list[str]


class FanoutRequest(BaseModel):
    """Request body for POST /api/fanout/generate."""
    target_query: str = Field(
        ..., min_length=1, description="The search query to decompose into sub-queries"
    )
    existing_content: Optional[str] = Field(
        None, description="Optional content to analyze for coverage gaps"
    )


class FanoutResponse(BaseModel):
    """Response body for POST /api/fanout/generate."""
    target_query: str
    model_used: str
    total_sub_queries: int = Field(..., ge=0)
    sub_queries: list[SubQuery]
    gap_summary: Optional[GapSummary] = None


class FanoutError(BaseModel):
    """Error response for fan-out generation failures."""
    error: str
    message: str
    detail: Optional[str] = None
