import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")
import pytest
from investigation_team.agents.evidence_analyst import normalize_assessment_status


@pytest.mark.parametrize("s", ["supported", "Weakened", " rejected ", "proposed"])
def test_valid_pass_through(s):
    assert normalize_assessment_status(s) == s.strip().lower()


@pytest.mark.parametrize("s", ["unresolved", "Inconclusive", "insufficient_evidence", "unknown"])
def test_insufficient_synonyms_map_to_proposed(s):
    assert normalize_assessment_status(s) == "proposed"


@pytest.mark.parametrize("s", ["confirmed", "banana", ""])
def test_other_values_still_raise(s):
    with pytest.raises(ValueError):
        normalize_assessment_status(s)
