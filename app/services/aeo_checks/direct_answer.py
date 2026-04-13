"""Check A — Direct Answer Detection.

Tests whether the first paragraph answers the likely primary query in <=60 words
with a clear declarative statement and no hedge phrases.

Scoring (max 20 pts):
    <= 60 words + declarative + no hedge  → 20
    <= 60 words but hedging or incomplete → 12
    61-90 words                           →  8
    > 90 words                            →  0
"""

from __future__ import annotations

import spacy

from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck

# Lazy-loaded spaCy model — loaded once on first use, then cached.
_nlp: spacy.Language | None = None

# Hedge phrases that weaken direct answers and reduce AI extraction confidence.
HEDGE_PHRASES = [
    "it depends",
    "may vary",
    "in some cases",
    "this varies",
    "generally speaking",
]


def _get_nlp() -> spacy.Language:
    """Load and cache the spaCy English model.

    Falls back to downloading en_core_web_sm if not already installed.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _has_hedge(text: str) -> bool:
    """Check if text contains any hedge phrases (case-insensitive)."""
    lower = text.lower()
    return any(phrase in lower for phrase in HEDGE_PHRASES)


def _is_declarative(text: str) -> bool:
    """Verify that the paragraph contains at least one declarative sentence.

    Uses spaCy's dependency parser to check for the presence of:
    - A subject (nsubj or nsubjpass dependency)
    - A root verb (VERB or AUX with ROOT dependency)

    This catches both active voice ("Python is a language") and passive
    voice ("The data was processed"), while filtering out questions and fragments.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    for sent in doc.sents:
        has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
        has_root_verb = any(
            tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX")
            for tok in sent
        )
        if has_subject and has_root_verb:
            return True

    return False


class DirectAnswerCheck(BaseCheck):
    """Checks if the first paragraph provides a concise, direct answer."""

    check_id = "direct_answer"
    name = "Direct Answer Detection"
    max_score = 20

    def run(self, first_paragraph: str, **kwargs) -> CheckResult:
        word_count = len(first_paragraph.split())
        has_hedge = _has_hedge(first_paragraph)
        is_decl = _is_declarative(first_paragraph)

        # Score based on word count, declarative structure, and hedge presence
        if word_count <= 60 and is_decl and not has_hedge:
            score = 20
        elif word_count <= 60:
            score = 12
        elif word_count <= 90:
            score = 8
        else:
            score = 0

        # Build actionable recommendation for imperfect scores
        rec = None
        if score < self.max_score:
            parts: list[str] = []
            if word_count > 60:
                parts.append(
                    f"Your opening paragraph is {word_count} words. "
                    f"Trim it to under 60 words with a direct, declarative answer."
                )
            if has_hedge:
                parts.append("Remove hedge phrases like 'it depends' or 'may vary'.")
            if not is_decl:
                parts.append(
                    "Rewrite the opening as a clear declarative statement with a subject and verb."
                )
            rec = " ".join(parts)

        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details={
                "word_count": word_count,
                "threshold": 60,
                "is_declarative": is_decl,
                "has_hedge_phrase": has_hedge,
            },
            recommendation=rec,
        )
