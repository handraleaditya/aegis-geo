"""Tests for fan-out engine JSON parsing and validation.

Tests the three layers of defense against LLM output issues:
  1. _extract_json: handles markdown fences and raw JSON
  2. _validate_sub_queries: enforces schema, filters invalid types, strips extra fields
  3. generate_sub_queries: full flow with mocked Gemini, including retry logic
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.services.fanout_engine import (
    _extract_json,
    _validate_sub_queries,
    generate_sub_queries,
    VALID_TYPES,
)


# ── Helper ────────────────────────────────────────────────────────────────

ALL_TYPES = ["comparative", "feature_specific", "use_case",
             "trust_signals", "how_to", "definitional"]


def _make_valid_response(per_type: int = 2) -> str:
    """Generate a valid JSON response string with the given count per type."""
    return json.dumps({
        "sub_queries": [
            {"type": t, "query": f"test query for {t} #{i}"}
            for t in ALL_TYPES
            for i in range(per_type)
        ]
    })


# ── JSON extraction tests ────────────────────────────────────────────────


class TestExtractJson:
    """Test JSON extraction from various LLM response formats."""

    def test_plain_json(self):
        text = '{"sub_queries": [{"type": "comparative", "query": "test"}]}'
        result = _extract_json(text)
        assert "sub_queries" in result

    def test_json_in_markdown_fence(self):
        text = '```json\n{"sub_queries": [{"type": "comparative", "query": "test"}]}\n```'
        result = _extract_json(text)
        assert "sub_queries" in result

    def test_json_in_plain_fence(self):
        text = '```\n{"sub_queries": [{"type": "comparative", "query": "test"}]}\n```'
        result = _extract_json(text)
        assert "sub_queries" in result

    def test_json_with_surrounding_whitespace(self):
        text = '  \n  {"sub_queries": []}  \n  '
        result = _extract_json(text)
        assert "sub_queries" in result

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("this is not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")

    def test_prose_with_json_in_fence(self):
        """Model sometimes adds explanation before/after the JSON fence."""
        text = 'Here are the sub-queries:\n```json\n{"sub_queries": [{"type": "comparative", "query": "test"}]}\n```\nHope this helps!'
        result = _extract_json(text)
        assert "sub_queries" in result


# ── Validation tests ─────────────────────────────────────────────────────


class TestValidateSubQueries:
    """Test schema validation and cleaning of parsed sub-queries."""

    def test_valid_data_passes(self):
        data = json.loads(_make_valid_response(2))
        result = _validate_sub_queries(data)
        assert len(result) == 12
        assert all("type" in sq and "query" in sq for sq in result)

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing 'sub_queries' key"):
            _validate_sub_queries({"wrong_key": []})

    def test_non_list_value_raises(self):
        with pytest.raises(ValueError, match="not a list"):
            _validate_sub_queries({"sub_queries": "not a list"})

    def test_filters_invalid_types(self):
        data = {
            "sub_queries": [
                {"type": t, "query": f"valid {t}"} for t in ALL_TYPES
            ] + [
                {"type": "INVALID_TYPE", "query": "should be filtered"},
                {"type": "random", "query": "also filtered"},
            ]
        }
        result = _validate_sub_queries(data)
        assert len(result) == 6
        result_types = {sq["type"] for sq in result}
        assert result_types == set(ALL_TYPES)

    def test_strips_extra_fields(self):
        """LLM sometimes adds fields like 'rationale' — these should be removed."""
        data = {
            "sub_queries": [
                {"type": t, "query": f"q {t}", "rationale": "because", "score": 0.9}
                for t in ALL_TYPES
            ]
        }
        result = _validate_sub_queries(data)
        for sq in result:
            assert set(sq.keys()) == {"type", "query"}

    def test_filters_empty_queries(self):
        """Sub-queries with blank query strings should be filtered out."""
        data = {
            "sub_queries": [
                {"type": t, "query": f"valid {t}"} for t in ALL_TYPES
            ] + [
                {"type": "comparative", "query": ""},
                {"type": "comparative", "query": "   "},
            ]
        }
        result = _validate_sub_queries(data)
        assert len(result) == 6  # The 2 blank ones are filtered

    def test_too_few_valid_queries_raises(self):
        data = {"sub_queries": [{"type": "comparative", "query": "only one"}]}
        with pytest.raises(ValueError, match="valid sub-queries"):
            _validate_sub_queries(data)

    def test_all_valid_types_accepted(self):
        """Verify every expected type is recognized."""
        data = {
            "sub_queries": [{"type": t, "query": f"q for {t}"} for t in VALID_TYPES]
        }
        result = _validate_sub_queries(data)
        assert len(result) == 6


# ── Mocked LLM integration tests ─────────────────────────────────────────


class TestGenerateSubQueriesMocked:
    """Test the full generate flow with a mocked Gemini response."""

    async def test_successful_generation(self):
        mock_response = MagicMock()
        mock_response.text = _make_valid_response(2)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("app.services.fanout_engine._get_client", return_value=mock_client):
            result, model_name = await generate_sub_queries("best CRM software")

            assert len(result) == 12
            assert model_name == "gemini-1.5-flash"
            # Verify all 6 types are represented
            result_types = {sq["type"] for sq in result}
            assert result_types == set(ALL_TYPES)

    async def test_retries_on_invalid_json(self):
        """First attempt returns garbage; second attempt succeeds."""
        bad_response = MagicMock()
        bad_response.text = "not valid json at all"

        good_response = MagicMock()
        good_response.text = _make_valid_response(2)

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [bad_response, good_response]

        with patch("app.services.fanout_engine._get_client", return_value=mock_client), \
             patch("app.services.fanout_engine.asyncio.sleep", return_value=None):

            result, _ = await generate_sub_queries("test query")
            assert len(result) == 12
            # Verify it retried
            assert mock_client.models.generate_content.call_count == 2

    async def test_all_retries_fail_raises_runtime_error(self):
        """If all 3 attempts fail, a RuntimeError should be raised."""
        bad_response = MagicMock()
        bad_response.text = "still not json"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = bad_response

        with patch("app.services.fanout_engine._get_client", return_value=mock_client), \
             patch("app.services.fanout_engine.asyncio.sleep", return_value=None):

            with pytest.raises(RuntimeError, match="failed after 3 retries"):
                await generate_sub_queries("test query")

            # Verify all 3 attempts were made
            assert mock_client.models.generate_content.call_count == 3

    async def test_missing_api_key_raises(self):
        """Should raise RuntimeError when GEMINI_API_KEY is not set."""
        with patch("app.services.fanout_engine.os.environ.get", return_value=None):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                await generate_sub_queries("test query")
