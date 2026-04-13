"""Check C — Snippet Readability Scorer.

Targets Flesch-Kincaid Grade Level 7-9 for optimal AI answer extraction.
This range is complex enough to be credible but simple enough for AI systems
to cleanly extract and cite passages.

Scoring (max 20 pts):
    FK Grade 7-9  → 20
    FK Grade 6/10 → 14
    FK Grade 5/11 →  8
    FK Grade <=4 or >=12 → 0
"""

from __future__ import annotations

import re

import textstat

from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    Filters out very short fragments (< 5 words) that are unlikely
    to be meaningful complete sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 5]


def _syllable_ratio(sentence: str) -> float:
    """Compute average syllables per word for a sentence.

    Higher ratios indicate more complex vocabulary. Used to rank sentences
    by complexity for the "most complex sentences" output.
    """
    words = sentence.split()
    if not words:
        return 0.0
    return textstat.syllable_count(sentence) / len(words)


def _find_complex_sentences(text: str, top_n: int = 3) -> list[str]:
    """Return the top_n most complex sentences ranked by syllable/word ratio."""
    sentences = _split_sentences(text)
    ranked = sorted(sentences, key=_syllable_ratio, reverse=True)
    return ranked[:top_n]


class ReadabilityCheck(BaseCheck):
    """Scores content readability against the FK Grade 7-9 target range."""

    check_id = "readability"
    name = "Snippet Readability"
    max_score = 20

    def run(self, body_text: str, **kwargs) -> CheckResult:
        """Compute Flesch-Kincaid grade level and identify complex sentences.

        Args:
            body_text: Cleaned text content (boilerplate already stripped).
        """
        fk_grade = round(textstat.flesch_kincaid_grade(body_text), 1)
        complex_sents = _find_complex_sentences(body_text)

        # Score based on how close the FK grade is to the 7-9 target range
        grade_rounded = round(fk_grade)
        if 7 <= grade_rounded <= 9:
            score = 20
        elif grade_rounded in (6, 10):
            score = 14
        elif grade_rounded in (5, 11):
            score = 8
        else:
            score = 0

        # Tailor recommendation based on whether content is too complex or too simple
        rec = None
        if score < self.max_score:
            if fk_grade > 9:
                rec = (
                    f"Content reads at Grade {fk_grade}. Shorten sentences and replace "
                    f"technical jargon with plain language to reach Grade 7-9."
                )
            else:
                rec = (
                    f"Content reads at Grade {fk_grade}. Add slightly more detail and "
                    f"specificity to reach Grade 7-9."
                )

        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details={
                "fk_grade_level": fk_grade,
                "target_range": "7-9",
                "complex_sentences": complex_sents,
            },
            recommendation=rec,
        )
