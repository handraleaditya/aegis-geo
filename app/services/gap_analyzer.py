"""Semantic Gap Analyzer — sentence-transformer embeddings + cosine similarity.

Compares LLM-generated sub-queries against user-provided content to identify
which topics are covered and which represent content gaps. Uses normalized
embeddings so cosine similarity reduces to a simple dot product.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Using MiniLM for speed over mpnet's accuracy — see README for justification.
# Lazy-loaded so the 80MB model is only downloaded/loaded on first use.
_model: SentenceTransformer | None = None
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default similarity threshold — sub-queries with max similarity >= this value
# are considered "covered" by the content. See README for tuning discussion.
SIMILARITY_THRESHOLD = 0.72


def _get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model.

    Import is deferred to avoid loading the heavy transformers stack
    at application startup — the model is only needed when gap analysis
    is actually requested.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _chunk_text(text: str) -> list[str]:
    """Split content into sentence-level chunks for embedding.

    Uses regex splitting on sentence-ending punctuation (.!?).
    Filters out very short chunks (< 5 words) that are unlikely to
    carry enough semantic meaning for meaningful similarity comparison.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) >= 5]


def analyze_gaps(
    sub_queries: list[dict],
    content: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Compute semantic coverage of sub-queries against content.

    For each sub-query, computes the maximum cosine similarity against all
    content sentence chunks. If max similarity >= threshold, the sub-query
    is marked as covered.

    Args:
        sub_queries: List of dicts with 'type' and 'query' keys.
        content: Raw text content to check coverage against.
        threshold: Cosine similarity threshold for "covered" classification.

    Returns:
        Enriched sub-query dicts with added 'covered' and 'similarity_score' fields.
    """
    model = _get_model()
    chunks = _chunk_text(content)

    if not chunks:
        # No meaningful content to compare against — everything is a gap
        return [
            {**sq, "covered": False, "similarity_score": 0.0}
            for sq in sub_queries
        ]

    # Batch encode for efficiency — normalize so dot product = cosine similarity
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    query_texts = [sq["query"] for sq in sub_queries]
    query_embeddings = model.encode(query_texts, normalize_embeddings=True)

    results = []
    for i, sq in enumerate(sub_queries):
        # Cosine similarity via dot product on L2-normalized vectors
        similarities = query_embeddings[i] @ chunk_embeddings.T
        max_sim = float(np.max(similarities))
        results.append({
            **sq,
            "covered": max_sim >= threshold,
            "similarity_score": round(max_sim, 2),
        })

    return results
