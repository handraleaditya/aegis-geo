"""AEO content checks package.

Each check implements BaseCheck and can be run independently or as part
of the full AEO scoring pipeline.
"""

from app.services.aeo_checks.direct_answer import DirectAnswerCheck
from app.services.aeo_checks.htag_hierarchy import HtagHierarchyCheck
from app.services.aeo_checks.readability import ReadabilityCheck

__all__ = ["DirectAnswerCheck", "HtagHierarchyCheck", "ReadabilityCheck"]
