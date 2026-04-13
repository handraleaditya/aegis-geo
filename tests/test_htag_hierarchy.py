"""Tests for Check B — H-tag Hierarchy Checker.

Covers all three scoring tiers (20, 12, 0) with edge cases for
missing H1, multiple H1s, skipped levels, and headings before H1.
"""

import pytest
from app.services.aeo_checks.htag_hierarchy import HtagHierarchyCheck


@pytest.fixture
def check():
    return HtagHierarchyCheck()


# ── Perfect hierarchy (score 20) ──────────────────────────────────────────


class TestHtagPass:
    """Valid heading structures — should score 20/20."""

    def test_perfect_hierarchy(self, check):
        headings = [
            {"tag": "h1", "text": "Main Title"},
            {"tag": "h2", "text": "Section A"},
            {"tag": "h3", "text": "Subsection A.1"},
            {"tag": "h2", "text": "Section B"},
        ]
        result = check.run(headings=headings)
        assert result.score == 20
        assert result.passed is True
        assert result.details["violations"] == []
        assert result.recommendation is None

    def test_single_h1_only(self, check):
        headings = [{"tag": "h1", "text": "Title"}]
        result = check.run(headings=headings)
        assert result.score == 20

    def test_deep_hierarchy_no_skips(self, check):
        """H1 -> H2 -> H3 -> H4 with no skips is valid."""
        headings = [
            {"tag": "h1", "text": "Title"},
            {"tag": "h2", "text": "Section"},
            {"tag": "h3", "text": "Subsection"},
            {"tag": "h4", "text": "Detail"},
            {"tag": "h2", "text": "Another Section"},  # going back up is OK
        ]
        result = check.run(headings=headings)
        assert result.score == 20

    def test_going_back_up_is_not_a_violation(self, check):
        """H3 -> H2 (going up) should not be flagged as a skip."""
        headings = [
            {"tag": "h1", "text": "Title"},
            {"tag": "h2", "text": "Section"},
            {"tag": "h3", "text": "Sub"},
            {"tag": "h2", "text": "Next Section"},  # up, not a skip
        ]
        result = check.run(headings=headings)
        assert result.score == 20


# ── Partial violations (score 12) ────────────────────────────────────────


class TestHtagPartial:
    """1-2 violations — should score 12."""

    def test_skipped_level(self, check):
        """H1 -> H3 without H2 is a single violation."""
        headings = [
            {"tag": "h1", "text": "Title"},
            {"tag": "h3", "text": "Jumped to H3"},
        ]
        result = check.run(headings=headings)
        assert result.score == 12
        assert len(result.details["violations"]) == 1
        assert "Skipped" in result.details["violations"][0]

    def test_multiple_h1(self, check):
        headings = [
            {"tag": "h1", "text": "Title 1"},
            {"tag": "h2", "text": "Section"},
            {"tag": "h1", "text": "Title 2"},
        ]
        result = check.run(headings=headings)
        assert result.score == 12
        assert any("Multiple" in v for v in result.details["violations"])

    def test_single_heading_before_h1(self, check):
        headings = [
            {"tag": "h2", "text": "Before"},
            {"tag": "h1", "text": "Title"},
            {"tag": "h2", "text": "Section"},
        ]
        result = check.run(headings=headings)
        assert result.score == 12


# ── Critical failures (score 0) ──────────────────────────────────────────


class TestHtagFail:
    """Missing H1 or 3+ violations — should score 0."""

    def test_missing_h1(self, check):
        """No H1 at all is a critical failure regardless of other violations."""
        headings = [
            {"tag": "h2", "text": "No H1 here"},
            {"tag": "h3", "text": "Sub"},
        ]
        result = check.run(headings=headings)
        assert result.score == 0
        assert result.passed is False
        assert result.recommendation is not None

    def test_three_plus_violations(self, check):
        """Multiple H1s + heading before H1 + skipped level = 3+ violations."""
        headings = [
            {"tag": "h3", "text": "Before H1"},
            {"tag": "h2", "text": "Also before"},
            {"tag": "h1", "text": "Late H1"},
            {"tag": "h1", "text": "Duplicate H1"},
            {"tag": "h4", "text": "Skipped"},
        ]
        result = check.run(headings=headings)
        assert result.score == 0
        assert len(result.details["violations"]) >= 3

    def test_empty_headings(self, check):
        """No headings at all = missing H1 = score 0."""
        result = check.run(headings=[])
        assert result.score == 0
        assert "Missing H1" in result.details["violations"][0]


# ── Output structure ──────────────────────────────────────────────────────


class TestHtagOutputStructure:
    """Verify the output conforms to the expected schema."""

    def test_output_has_required_fields(self, check):
        headings = [{"tag": "h1", "text": "Title"}]
        result = check.run(headings=headings)
        assert result.check_id == "htag_hierarchy"
        assert result.name == "H-tag Hierarchy"
        assert result.max_score == 20
        assert "violations" in result.details
        assert "h_tags_found" in result.details
        assert isinstance(result.details["violations"], list)
        assert isinstance(result.details["h_tags_found"], list)

    def test_h_tags_found_preserves_order(self, check):
        headings = [
            {"tag": "h1", "text": "A"},
            {"tag": "h2", "text": "B"},
            {"tag": "h3", "text": "C"},
        ]
        result = check.run(headings=headings)
        assert result.details["h_tags_found"] == ["h1", "h2", "h3"]
