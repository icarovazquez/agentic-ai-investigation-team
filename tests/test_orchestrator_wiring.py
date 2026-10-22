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


def test_loop_stops_on_convergence_and_records_residual_concerns(monkeypatch):
    """Challenger disputes every round; the same supported leader holds -> stop after round 2."""
    from types import SimpleNamespace as NS
    from investigation_team.agents.evidence_analyst import EvidenceAnalysis, HypothesisAssessment
    from investigation_team.agents.hypothesis_challenger import ChallengeReport, HypothesisChallenge
    from investigation_team.domain import EvidencePlan, EvidenceTest, HypothesisStatus as S

    monkeypatch.setattr(orch, "incident_framing_agent", lambda **kw: F())
    monkeypatch.setattr(orch, "hypothesis_generator_agent", lambda **kw: HS())
    monkeypatch.setattr(orch, "evidence_planning_agent",
                        lambda **kw: EvidenceGapSelection("inc", [EvidenceGap("h1", "d", "topology")]))
    rounds = {"n": 0}
    def create(**kw):
        rounds["n"] += 1
        return EvidencePlan(incident_id="inc", tests=[EvidenceTest(
            test_id=f"inc-r{kw['round_number']}-test-1", hypothesis_id="h1", objective="o", capability="topology")])
    monkeypatch.setattr(orch, "evidence_test_creation_agent", create)
    from investigation_team.domain import EvidenceTestResult
    monkeypatch.setattr(orch, "execute_evidence_plan", lambda **kw: [EvidenceTestResult(
        test_id=kw["evidence_plan"].tests[0].test_id, hypothesis_id="h1", capability="topology", status="completed",
        evidence={"topology_evidence": {"observations": [{"observation_id": kw["evidence_plan"].tests[0].test_id}]}})])
    analysis = EvidenceAnalysis(incident_id="inc", assessments=[
        HypothesisAssessment(hypothesis_id="h1", status=S.SUPPORTED, confidence=0.92, rationale="r"),
        HypothesisAssessment(hypothesis_id="h2", status=S.REJECTED, confidence=0.05, rationale="r")])
    monkeypatch.setattr(orch, "evidence_analyst_agent", lambda **kw: analysis)
    report = ChallengeReport(incident_id="inc", challenges=[HypothesisChallenge(
        hypothesis_id="h1", original_status="supported", original_confidence=0.92, challenge_outcome="disputed",
        rebuttal="no optical readings", alternative_explanation="x", additional_evidence_needed=["optical power"])],
        systemic_concerns=["s"])
    monkeypatch.setattr(orch, "hypothesis_challenger_agent", lambda **kw: report)
    rec = NS(status="root_cause_identified", hypothesis_id="h1", confidence=0.92, remediation_steps=[], validate=lambda: None,
             to_dict=lambda: {})
    monkeypatch.setattr(orch, "root_cause_remediation_agent", lambda **kw: rec)

    r = orch.run_investigation(_config())
    assert rounds["n"] == 2, "should stop after 2 stable rounds, not run all 5"
    assert "Converged in round 2" in r.convergence_note
    assert any("no optical readings" in c for c in r.residual_concerns)
    assert r.status.value == "remediation_recommended"


def test_load_failure_reports_the_underlying_cause(monkeypatch):
    def boom(**kw):
        try:
            raise KeyError("No adapter registered as 'NIKAQoSContentionAdapter'")
        except KeyError as inner:
            raise RuntimeError("Required evidence source failed: topo") from inner
    monkeypatch.setattr(orch, "load_investigation_evidence_executor", boom)
    r = orch.run_investigation(_config())
    assert r.failed_stage == "load_evidence"
    assert "Required evidence source failed: topo" in r.error
    assert "No adapter registered as 'NIKAQoSContentionAdapter'" in r.error
