import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

from types import SimpleNamespace as NS

import pytest

import investigation_team.agents.causal_resolution as cr
import investigation_team.agents.hypothesis_challenger as ch
import investigation_team.agents.root_cause_remediation as rcr
import investigation_team.orchestrator as orch
from investigation_team import capabilities  # noqa: F401
from investigation_team.agents.evidence_analyst import EvidenceAnalysis, HypothesisAssessment
from investigation_team.agents.hypothesis_challenger import (
    ChallengeReport, HypothesisChallenge, enforce_dispute_actionability)
from investigation_team.config import InvestigationPolicy
from investigation_team.domain import EvidenceTestResult, HypothesisStatus as S
from investigation_team.evidence_progress import evidence_ids


def A(hid, status, conf):
    return HypothesisAssessment(hypothesis_id=hid, status=status, confidence=conf, rationale="r")


def HSET():
    return NS(incident_id="inc", hypotheses=[
        NS(hypothesis_id="inc-h1", title="Congestion", to_dict=lambda: {"id": "inc-h1"}),
        NS(hypothesis_id="inc-h2", title="Policy missing", to_dict=lambda: {"id": "inc-h2"}),
    ])


# Scenario 2 outcome shape: trigger 0.82, underlying cause 0.78, both supported.
ANALYSIS = EvidenceAnalysis("inc", [A("inc-h1", S.SUPPORTED, 0.82), A("inc-h2", S.SUPPORTED, 0.78),
                                    A("inc-h3", S.REJECTED, 0.1)])


def test_policy_field_defaults_and_validates():
    assert InvestigationPolicy().contributing_cause_minimum_confidence == 0.60
    with pytest.raises(ValueError):
        InvestigationPolicy(contributing_cause_minimum_confidence=1.5).validate()


def test_chain_candidates_are_supported_and_strong_enough():
    ids = [a.hypothesis_id for a in cr.chain_candidates(ANALYSIS, 0.6)]
    assert ids == ["inc-h1", "inc-h2"]
    assert [a.hypothesis_id for a in cr.chain_candidates(ANALYSIS, 0.8)] == ["inc-h1"]


def test_valid_chain_is_accepted_and_overrides_the_confidence_leader():
    parsed = {"relationship": "causal_chain", "root_cause_hypothesis_id": "h2",
              "contributing_hypothesis_ids": ["h1"], "rationale": "policy missing exposed congestion",
              "deviation_from_intended_state": "policy detached"}
    res = cr.validate_causal_resolution(parsed, ["inc-h1", "inc-h2"], "inc-h1", HSET())
    assert res.relationship == "causal_chain"
    assert res.root_cause_hypothesis_id == "inc-h2"
    assert res.contributing_hypothesis_ids == ["inc-h1"]


@pytest.mark.parametrize("parsed", [
    {"relationship": "maybe"},
    {"relationship": "causal_chain", "root_cause_hypothesis_id": "inc-h9",
     "contributing_hypothesis_ids": ["inc-h1"], "rationale": "x"},
    {"relationship": "causal_chain", "root_cause_hypothesis_id": "inc-h2",
     "contributing_hypothesis_ids": [], "rationale": "x"},
    {"relationship": "causal_chain", "root_cause_hypothesis_id": "inc-h2",
     "contributing_hypothesis_ids": ["inc-h2"], "rationale": "x"},
    {"relationship": "causal_chain", "root_cause_hypothesis_id": "inc-h2",
     "contributing_hypothesis_ids": ["inc-h1"], "rationale": " "},
    {"relationship": "causal_chain", "root_cause_hypothesis_id": "inc-h2",
     "contributing_hypothesis_ids": ["inc-h3"], "rationale": "x"},  # h3 not a candidate
])
def test_bad_chains_fall_back_to_rivals_with_the_deterministic_leader(parsed):
    res = cr.validate_causal_resolution(parsed, ["inc-h1", "inc-h2"], "inc-h1", HSET())
    assert res.relationship == "rivals" and res.root_cause_hypothesis_id == "inc-h1"
    assert res.contributing_hypothesis_ids == []


def test_rivals_keep_the_leader():
    res = cr.validate_causal_resolution({"relationship": "rivals", "rationale": "r"},
                                        ["inc-h1", "inc-h2"], "inc-h1", HSET())
    assert res.relationship == "rivals" and res.root_cause_hypothesis_id == "inc-h1" and not res.notes


def test_agent_end_to_end(monkeypatch):
    monkeypatch.setattr(cr, "llm_call", lambda **kw: {"content": "x"})
    parsed = {"relationship": "causal_chain", "root_cause_hypothesis_id": "inc-h2",
              "contributing_hypothesis_ids": ["inc-h1"], "rationale": "r"}
    monkeypatch.setattr(cr, "parse_or_repair_agent_response", lambda **kw: parsed)
    frame = NS(to_dict=lambda: {})
    for a in ANALYSIS.assessments:
        a.to_dict = a.to_dict
    res = cr.causal_resolution_agent(frame, HSET(), ANALYSIS,
                                     cr.chain_candidates(ANALYSIS, 0.6), "inc-h1")
    assert res.root_cause_hypothesis_id == "inc-h2"


# ---- challenger rival routing

def C(rival, outcome="disputed"):
    return HypothesisChallenge(
        hypothesis_id="inc-h1", original_status="supported", original_confidence=0.82,
        challenge_outcome=outcome, rebuttal="r", alternative_explanation="alt", rival_hypothesis_id=rival,
        distinguishing_capability="deep_diagnostics", distinguishing_observable="interface_error_count",
        distinguishing_observation="obs")


def test_dispute_whose_rival_is_supported_is_routed_to_causal_resolution():
    c = C("inc-h2")
    notes = enforce_dispute_actionability([c], supported_ids=["inc-h1", "inc-h2"])
    assert c.challenge_outcome == "upheld" and "causal resolution" in c.downgrade_note and notes


def test_dispute_whose_rival_is_not_supported_stays_disputed():
    c = C("inc-h3")
    assert enforce_dispute_actionability([c], supported_ids=["inc-h1", "inc-h2"]) == []
    assert c.challenge_outcome == "disputed"
    c = C(None)
    assert enforce_dispute_actionability([c], supported_ids=["inc-h1", "inc-h2"]) == []


def test_challenger_agent_drops_made_up_rival_ids(monkeypatch):
    parsed = {"challenges": [{"hypothesis_id": "inc-h1", "challenge_outcome": "disputed", "rebuttal": "r",
                              "alternative_explanation": "a", "rival_hypothesis_id": "inc-h42",
                              "distinguishing_capability": "deep_diagnostics",
                              "distinguishing_observable": "interface_error_count",
                              "distinguishing_observation": "o", "additional_evidence_needed": []}],
              "systemic_concerns": []}
    monkeypatch.setattr(ch, "llm_call", lambda **kw: {"content": "x"})
    monkeypatch.setattr(ch, "parse_or_repair_agent_response", lambda **kw: parsed)
    frame = NS(incident_id="inc", to_dict=lambda: {})
    ev = NS(assessments=ANALYSIS.assessments, to_dict=lambda: {})
    try:
        report = ch.hypothesis_challenger_agent(frame, HSET(), ANALYSIS)
    except Exception as exc:  # context building may need richer fakes; surface it
        raise AssertionError(exc)
    assert report.challenges[0].rival_hypothesis_id is None


# ---- remediation context and selection

def test_remediation_context_includes_contributing_factors():
    ctx = rcr.build_root_cause_context(NS(to_dict=lambda: {}), HSET(), ANALYSIS.assessments[1], None,
                                       contributing_assessments=[ANALYSIS.assessments[0]],
                                       causal_rationale="chain")
    assert ctx["contributing_factors"][0]["assessment"]["hypothesis_id"] == "inc-h1"
    assert ctx["causal_rationale"] == "chain"
    plain = rcr.build_root_cause_context(NS(to_dict=lambda: {}), HSET(), ANALYSIS.assessments[0], None)
    assert "contributing_factors" not in plain


def test_remediation_agent_uses_chain_root_cause(monkeypatch):
    seen = {}
    def fake_llm(**kw):
        seen["prompt"] = kw["messages"][1]["content"]
        return {"content": "x"}
    monkeypatch.setattr(rcr, "llm_call", fake_llm)
    monkeypatch.setattr(rcr, "parse_or_repair_agent_response", lambda **kw: {
        "root_cause_explanation": "e", "remediation_steps": [{"action": "a", "rationale": "r", "risk_level": "low"}],
        "residual_risk": "x", "challenge_acknowledged": "n"})
    rec = rcr.root_cause_remediation_agent(
        NS(incident_id="inc", to_dict=lambda: {}), HSET(), ANALYSIS, None, InvestigationPolicy(),
        root_cause_hypothesis_id="inc-h2", contributing_hypothesis_ids=["inc-h1"], causal_rationale="chain")
    assert rec.hypothesis_id == "inc-h2" and rec.confidence == 0.78
    assert rec.causal_relationship == "causal_chain" and rec.contributing_hypothesis_ids == ["inc-h1"]
    assert "contributing_factors" in seen["prompt"]


def test_remediation_without_chain_keeps_the_leader(monkeypatch):
    monkeypatch.setattr(rcr, "llm_call", lambda **kw: {"content": "x"})
    monkeypatch.setattr(rcr, "parse_or_repair_agent_response", lambda **kw: {
        "root_cause_explanation": "e", "remediation_steps": [{"action": "a", "rationale": "r", "risk_level": "low"}]})
    rec = rcr.root_cause_remediation_agent(
        NS(incident_id="inc", to_dict=lambda: {}), HSET(), ANALYSIS, None, InvestigationPolicy())
    assert rec.hypothesis_id == "inc-h1" and rec.causal_relationship == "single_cause"


# ---- evidence progress and orchestrator

def R(test_id, obs_ids):
    return EvidenceTestResult(test_id=test_id, hypothesis_id="h1", capability="traffic", status="completed",
                              evidence={"traffic_evidence": {"observations": [{"observation_id": o} for o in obs_ids]}})


def test_evidence_ids_collects_by_kind_and_ignores_other_keys():
    r = EvidenceTestResult("t", "h", "c", "completed", evidence={
        "x_evidence": {"observations": [{"observation_id": "o1"}], "events": [{"event_id": "e1"}],
                       "changes": [{"change_id": "c1"}]}, "note": {"observations": [{"observation_id": "zz"}]}})
    assert evidence_ids([r]) == {"obs:o1", "evt:e1", "chg:c1"}


def _wire(monkeypatch, round_results, analyses, challenge):
    from investigation_team.agents.evidence_planner import EvidenceGap, EvidenceGapSelection
    from investigation_team.domain import EvidencePlan, EvidenceTest
    from tests.test_orchestrator_wiring import F, HS, _config
    monkeypatch.setattr(orch, "incident_framing_agent", lambda **kw: F())
    monkeypatch.setattr(orch, "hypothesis_generator_agent", lambda **kw: HS())
    monkeypatch.setattr(orch, "evidence_planning_agent",
                        lambda **kw: EvidenceGapSelection("inc", [EvidenceGap("h1", "d", "topology")]))
    monkeypatch.setattr(orch, "evidence_test_creation_agent", lambda **kw: EvidencePlan(
        incident_id="inc", tests=[EvidenceTest(test_id=f"inc-r{kw['round_number']}-test-1", hypothesis_id="h1",
                                               objective="o", capability="topology")]))
    calls = {"exec": 0, "analyst": 0}
    def execute(**kw):
        calls["exec"] += 1
        return round_results[min(calls["exec"], len(round_results)) - 1]
    def analyst(**kw):
        calls["analyst"] += 1
        return analyses[min(calls["analyst"], len(analyses)) - 1]
    monkeypatch.setattr(orch, "execute_evidence_plan", execute)
    monkeypatch.setattr(orch, "evidence_analyst_agent", analyst)
    monkeypatch.setattr(orch, "hypothesis_challenger_agent",
                        lambda **kw: ChallengeReport(incident_id="inc", challenges=challenge(), systemic_concerns=[]))
    return calls, _config()


def test_loop_stops_when_a_round_returns_nothing_new(monkeypatch):
    weak = EvidenceAnalysis("inc", [A("h1", S.SUPPORTED, 0.6)])  # below the bar: loop wants another round
    calls, cfg = _wire(monkeypatch, [[R("a", ["o1"])], [R("b", ["o1"])]], [weak],
                       lambda: [HypothesisChallenge("h1", "supported", 0.6, "unresolved", "r")])
    rec = NS(status="insufficient_evidence", hypothesis_id=None, confidence=None, remediation_steps=[],
             validate=lambda: None, to_dict=lambda: {})
    monkeypatch.setattr(orch, "root_cause_remediation_agent", lambda **kw: rec)
    r = orch.run_investigation(cfg)
    assert calls["exec"] == 2 and calls["analyst"] == 1
    assert r.exhaustion_note and "round 2" in r.exhaustion_note


def test_new_evidence_keeps_the_loop_going(monkeypatch):
    weak = EvidenceAnalysis("inc", [A("h1", S.SUPPORTED, 0.6)])
    calls, cfg = _wire(monkeypatch, [[R("a", ["o1"])], [R("b", ["o2"])], [R("c", ["o3"])],
                                     [R("d", ["o4"])], [R("e", ["o5"])]], [weak],
                       lambda: [HypothesisChallenge("h1", "supported", 0.6, "unresolved", "r")])
    rec = NS(status="insufficient_evidence", hypothesis_id=None, confidence=None, remediation_steps=[],
             validate=lambda: None, to_dict=lambda: {})
    monkeypatch.setattr(orch, "root_cause_remediation_agent", lambda **kw: rec)
    r = orch.run_investigation(cfg)
    assert calls["exec"] == 5 and r.exhaustion_note is None


def test_two_supported_hypotheses_end_in_round_one_and_resolve_as_a_chain(monkeypatch):
    analysis = EvidenceAnalysis("inc", [A("inc-h1", S.SUPPORTED, 0.82), A("inc-h2", S.SUPPORTED, 0.78)])
    # The Challenger disputed h1 naming h2 as the rival; the routing downgrades it.
    def challenges():
        c = HypothesisChallenge("inc-h1", "supported", 0.82, "disputed", "r", alternative_explanation="a",
                                rival_hypothesis_id="inc-h2", distinguishing_capability="traffic",
                                distinguishing_observable="queue_drops", distinguishing_observation="o")
        enforce_dispute_actionability([c], supported_ids=["inc-h1", "inc-h2"])
        return [c]
    calls, cfg = _wire(monkeypatch, [[R("a", ["o1"])]], [analysis], challenges)
    resolution = cr.CausalResolution("causal_chain", "inc-h2", ["inc-h1"], rationale="chain")
    monkeypatch.setattr(orch, "causal_resolution_agent", lambda **kw: resolution)
    got = {}
    def remediation(**kw):
        got.update(kw)
        return NS(status="root_cause_identified", hypothesis_id=kw["root_cause_hypothesis_id"], confidence=0.78,
                  remediation_steps=[], validate=lambda: None, to_dict=lambda: {})
    monkeypatch.setattr(orch, "root_cause_remediation_agent", remediation)
    r = orch.run_investigation(cfg)
    assert calls["exec"] == 1  # no wasted rounds
    assert r.causal_resolution is resolution
    assert got["root_cause_hypothesis_id"] == "inc-h2" and got["contributing_hypothesis_ids"] == ["inc-h1"]
    assert r.causal_resolution.to_dict()["relationship"] == "causal_chain"


def test_causal_resolution_failure_falls_back_to_the_leader(monkeypatch):
    analysis = EvidenceAnalysis("inc", [A("inc-h1", S.SUPPORTED, 0.82), A("inc-h2", S.SUPPORTED, 0.78)])
    calls, cfg = _wire(monkeypatch, [[R("a", ["o1"])]], [analysis],
                       lambda: [HypothesisChallenge("inc-h1", "supported", 0.82, "upheld", "r")])
    def boom(**kw):
        raise RuntimeError("llm down")
    monkeypatch.setattr(orch, "causal_resolution_agent", boom)
    got = {}
    def remediation(**kw):
        got.update(kw)
        return NS(status="root_cause_identified", hypothesis_id="inc-h1", confidence=0.82,
                  remediation_steps=[], validate=lambda: None, to_dict=lambda: {})
    monkeypatch.setattr(orch, "root_cause_remediation_agent", remediation)
    r = orch.run_investigation(cfg)
    assert r.causal_resolution is None and got["root_cause_hypothesis_id"] is None


# ---- concerns cleanup after a causal chain

from investigation_team.convergence import residual_concerns, resolved_by_causal_resolution


def _report():
    routed = HypothesisChallenge("inc-h1", "supported", 0.92, "disputed", "h1 might just be coincidence",
                                 rival_hypothesis_id="inc-h2", distinguishing_capability="traffic",
                                 distinguishing_observable="queue_drops", distinguishing_observation="o",
                                 additional_evidence_needed=["ws3 backup start time"])
    enforce_dispute_actionability([routed], supported_ids=["inc-h1", "inc-h2"])
    upheld = HypothesisChallenge("inc-h2", "supported", 0.88, "upheld", "fine")
    return ChallengeReport("inc", [routed, upheld], systemic_concerns=["H1 and H2 not distinguished"])


def test_routed_rebuttal_is_not_an_open_concern_after_a_chain_but_the_evidence_gap_is():
    res = cr.CausalResolution("causal_chain", "inc-h2", ["inc-h1"], rationale="policy missing exposed load")
    open_ = residual_concerns(_report(), "inc-h2", extra_ids=["inc-h1"], after_causal_chain=True)
    assert not any(c.startswith("Challenger") for c in open_)
    assert "Evidence that would resolve it: ws3 backup start time" in open_
    assert any(c.startswith("Systemic (raised before causal resolution)") for c in open_)
    settled = resolved_by_causal_resolution(_report(), res)
    assert len(settled) == 1 and "policy missing exposed load" in settled[0]


def test_without_a_chain_concerns_are_unchanged():
    open_ = residual_concerns(_report(), "inc-h1")
    assert any(c.startswith("Challenger (downgraded dispute)") for c in open_)
    assert resolved_by_causal_resolution(_report(), cr.CausalResolution("rivals", "inc-h1")) == []
