"""Abstract base class for all AEO checks.

Each check in the AEO scoring pipeline implements this interface,
ensuring consistent structure and making it easy to add new checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import CheckResult


class BaseCheck(ABC):
    """Base interface for AEO content checks.

    Subclasses must define check_id, name, and implement run().
    The max_score defaults to 20 (matching the assignment spec).
    """

    check_id: str
    name: str
    max_score: int = 20

    @abstractmethod
    def run(self, **kwargs) -> CheckResult:
        """Execute the check and return a structured CheckResult."""
        ...
