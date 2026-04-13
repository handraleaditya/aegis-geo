"""Check B — H-tag Hierarchy Checker.

Validates that the heading structure follows a logical H1 -> H2 -> H3 order.
Checks three rules:
  1. Exactly one <h1> is present
  2. No heading level is skipped (e.g. H1 -> H3 without H2)
  3. No H-tag appears before the H1

Scoring (max 20 pts):
    0 violations   → 20
    1-2 violations → 12
    3+ violations OR missing H1 → 0
"""

from __future__ import annotations

from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck


def _heading_level(tag_name: str) -> int:
    """Extract numeric heading level from tag name (e.g. 'h2' -> 2)."""
    return int(tag_name[1])


class HtagHierarchyCheck(BaseCheck):
    """Validates heading tag hierarchy for proper document structure."""

    check_id = "htag_hierarchy"
    name = "H-tag Hierarchy"
    max_score = 20

    def run(self, headings: list[dict], **kwargs) -> CheckResult:
        """Analyze heading structure for hierarchy violations.

        Args:
            headings: List of dicts with 'tag' and 'text' keys, in DOM order.
                      e.g. [{"tag": "h1", "text": "Title"}, {"tag": "h2", "text": "Section"}]
        """
        violations: list[str] = []
        h_tags_found = [h["tag"] for h in headings]

        # Rule 1: Exactly one H1 must be present
        h1_count = sum(1 for t in h_tags_found if t == "h1")
        if h1_count == 0:
            violations.append("Missing H1 tag.")
        elif h1_count > 1:
            violations.append(
                f"Multiple H1 tags found ({h1_count}). There should be exactly one."
            )

        # Rule 2: No heading should appear before the H1
        if h1_count >= 1:
            first_h1_idx = h_tags_found.index("h1")
            if first_h1_idx > 0:
                before = h_tags_found[:first_h1_idx]
                violations.append(
                    f"Heading(s) {before} appear before the H1."
                )

        # Rule 3: No heading level should be skipped
        for i in range(1, len(h_tags_found)):
            prev_level = _heading_level(h_tags_found[i - 1])
            curr_level = _heading_level(h_tags_found[i])
            # Only flag when going deeper (H2->H4), not when going back up (H3->H2)
            if curr_level > prev_level + 1:
                skipped = ", ".join(
                    f"H{level}" for level in range(prev_level + 1, curr_level)
                )
                violations.append(
                    f"Skipped heading level: {h_tags_found[i-1].upper()} -> "
                    f"{h_tags_found[i].upper()} (missing {skipped})."
                )

        # Scoring: missing H1 is always a critical failure
        missing_h1 = h1_count == 0
        n_violations = len(violations)

        if n_violations >= 3 or missing_h1:
            score = 0
        elif n_violations >= 1:
            score = 12
        else:
            score = 20

        rec = None
        if score < self.max_score:
            rec = "Fix heading hierarchy: " + " ".join(violations)

        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details={
                "violations": violations,
                "h_tags_found": h_tags_found,
            },
            recommendation=rec,
        )
