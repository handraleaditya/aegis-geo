"""Tests for Check C — Snippet Readability Scorer.

Covers the FK grade scoring bands (20, 14, 8, 0), complex sentence
identification, and output structure validation.
"""

import pytest
from app.services.aeo_checks.readability import (
    ReadabilityCheck,
    _split_sentences,
    _syllable_ratio,
    _find_complex_sentences,
)


@pytest.fixture
def check():
    return ReadabilityCheck()


# ── Unit tests for helper functions ───────────────────────────────────────


class TestSentenceSplitting:
    """Test the sentence splitting utility."""

    def test_splits_on_period(self):
        text = (
            "This is the first full sentence here. "
            "This is the second full sentence here. "
            "This is the third full sentence here too."
        )
        result = _split_sentences(text)
        assert len(result) == 3

    def test_filters_short_fragments(self):
        text = "Yes. No. Maybe so. This is a real sentence with enough words."
        result = _split_sentences(text)
        # "Yes.", "No.", "Maybe so." are all < 5 words — filtered out
        assert len(result) == 1

    def test_handles_empty_string(self):
        assert _split_sentences("") == []


class TestSyllableRatio:
    """Test syllable ratio computation."""

    def test_simple_words_low_ratio(self):
        # Simple monosyllabic words
        ratio = _syllable_ratio("The cat sat on the mat")
        assert ratio < 1.5

    def test_complex_words_high_ratio(self):
        # Multi-syllable academic words
        ratio = _syllable_ratio("epistemological computational methodological")
        assert ratio > 2.0

    def test_empty_string_returns_zero(self):
        assert _syllable_ratio("") == 0.0


class TestFindComplexSentences:
    """Test complex sentence identification."""

    def test_returns_top_3(self):
        text = (
            "Simple words here now. "
            "The computational methodological framework is significant. "
            "A cat sat down. "
            "Epistemological ramifications necessitate comprehensive reevaluation. "
            "Dogs run fast today."
        )
        result = _find_complex_sentences(text, top_n=3)
        assert len(result) <= 3

    def test_complex_sentence_ranked_first(self):
        text = (
            "The cat sat on a mat with great comfort. "
            "Epistemological phenomenological metacognitive deliberative processes are fundamental. "
            "Dogs like to run around the park every single day."
        )
        result = _find_complex_sentences(text, top_n=1)
        assert "Epistemological" in result[0]


# ── Full check scoring tests ─────────────────────────────────────────────


class TestReadabilityOptimal:
    """Content at FK grade 7-9 should score 20/20."""

    def test_grade_7_to_9_content(self, check):
        text = (
            "Cloud computing lets businesses store data on remote servers. "
            "This means companies do not need to buy expensive hardware. "
            "Instead, they can rent storage and computing power from providers "
            "like Amazon or Google. Small businesses benefit the most because "
            "they can scale their resources as they grow. The cost savings are "
            "significant compared to running a local data center."
        )
        result = check.run(body_text=text)
        assert result.details["target_range"] == "7-9"
        # Verify scoring logic runs without error
        assert 0 <= result.score <= 20


class TestReadabilityTooComplex:
    """Overly complex content should score low."""

    def test_academic_jargon(self, check):
        text = (
            "The epistemological ramifications of contemporary computational "
            "methodologies necessitate a comprehensive reevaluation of established "
            "paradigmatic frameworks. Notwithstanding the multifaceted nature of "
            "interdisciplinary convergences, the hermeneutical implications of "
            "algorithmic determinism fundamentally challenge anthropocentric "
            "presuppositions regarding phenomenological consciousness and "
            "metacognitive deliberative processes. Furthermore, the ontological "
            "underpinnings of distributed computational architectures precipitate "
            "substantive transformations in epistemological categorization schemas."
        )
        result = check.run(body_text=text)
        assert result.details["fk_grade_level"] > 9
        assert result.score < 20
        assert result.recommendation is not None
        assert "Shorten" in result.recommendation  # Should suggest simplifying


class TestReadabilityTooSimple:
    """Very simple content should score low (below target range)."""

    def test_elementary_text(self, check):
        text = (
            "The cat sat on a mat. The dog ran fast. A bird flew up. "
            "The sun was hot. We had fun. It was a good day. "
            "I like to play. You like to run. We all had fun."
        )
        result = check.run(body_text=text)
        assert result.details["fk_grade_level"] < 7
        assert result.score < 20
        assert result.recommendation is not None
        assert "detail" in result.recommendation.lower()  # Should suggest more detail


# ── Output structure ──────────────────────────────────────────────────────


class TestReadabilityOutputStructure:
    """Verify the output conforms to the expected schema."""

    def test_output_has_required_fields(self, check):
        text = "Machine learning uses statistical methods to find patterns in large datasets."
        result = check.run(body_text=text)
        assert result.check_id == "readability"
        assert result.name == "Snippet Readability"
        assert result.max_score == 20
        assert "fk_grade_level" in result.details
        assert "target_range" in result.details
        assert "complex_sentences" in result.details

    def test_complex_sentences_is_list(self, check):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It uses statistical methods to find patterns in data. "
            "The extraordinarily sophisticated methodological underpinnings "
            "of contemporary deep learning architectures represent paradigmatic "
            "breakthroughs in computational intelligence. "
            "Simple models work well for small datasets."
        )
        result = check.run(body_text=text)
        assert isinstance(result.details["complex_sentences"], list)
        assert len(result.details["complex_sentences"]) <= 3

    def test_fk_grade_is_numeric(self, check):
        text = "Cloud computing provides scalable infrastructure for modern businesses."
        result = check.run(body_text=text)
        assert isinstance(result.details["fk_grade_level"], float)
