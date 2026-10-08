"""Offline wiring test: stage names and call signatures after the split."""
import os
import re

for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

import json
import pytest
from investigation_team import orchestrator as orch
from investigation_team.agents.evidence_planner import EvidenceGap, EvidenceGapSelection


def _config():
    nb = json.load(open("notebooks/Scenario 1 - Simple BGP Link Down.ipynb"))
    ns = {}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if src.startswith("import pprint") or src.startswith("nika_simple_bgp_config ="):
                src = src.replace("check_connection()", "")
                exec(src, ns)
    return ns["nika_simple_bgp_config"]


class F:
    investigation_question = "q"
    incident_id = "inc"
    def validate(self): pass
    def to_dict(self): return {}


class HS:
    incident_id = "inc"
    hypotheses = []
    def validate(self): pass


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(orch, "incident_framing_agent", lambda **kw: F())
    monkeypatch.setattr(orch, "hypothesis_generator_agent", lambda **kw: HS())


def test_gap_selection_failure_stage(patched, monkeypatch):
    def boom(**kw):
        assert "round_number" not in kw
        raise RuntimeError("x")
    monkeypatch.setattr(orch, "evidence_planning_agent", boom)
    r = orch.run_investigation(_config())
    assert r.failed_stage == "evidence_gap_selection_round_1"


def test_test_creation_failure_stage(patched, monkeypatch):
    sel = EvidenceGapSelection("inc", [EvidenceGap("h1", "d", "topology")])
    monkeypatch.setattr(orch, "evidence_planning_agent", lambda **kw: sel)
    seen = {}
    def boom(**kw):
        seen.update(kw)
        raise RuntimeError("y")
    monkeypatch.setattr(orch, "evidence_test_creation_agent", boom)
    r = orch.run_investigation(_config())
    assert r.failed_stage == "evidence_test_creation_round_1"
    assert seen["round_number"] == 1 and seen["evidence_gap_selection"] is sel
