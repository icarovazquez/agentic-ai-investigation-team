import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

from types import SimpleNamespace as NS

import investigation_team.agents.hypothesis_challenger as ch
import investigation_team.orchestrator as orch
from investigation_team import capabilities  # noqa: F401  (registers capabilities)
from investigation_team.agents.hypothesis_challenger import (
    ChallengeReport, HypothesisChallenge, describe_challenge_outcome, enforce_dispute_actionability)
from investigation_team.convergence import residual_concerns
from investigation_team.evaluation import hypothesis_text_matches_fault as matches

FT, RC = "link_failure", "link_down"

# The four hypotheses from a real Scenario 1 run (title + cause + mechanism, abridged).
H1 = ("Physical Layer Link Failure on Core Interconnect. The core link has experienced a physical layer failure "
      "(cable disconnection, transceiver failure). A physical layer fault prevents transmission of any frames; "
      "the interfaces transition to down.")
H2 = ("BGP Session Failure Due to Configuration or Protocol Issue. The BGP session has failed due to "
      "misconfiguration, even though the physical link may be operational.")
H3 = ("Interface Administratively Shut Down on Core Link. Interfaces were administratively disabled. This blocks "
      "all traffic regardless of physical layer status, and prevents BGP adjacency.")
H4 = ("Packet Loss Due to Congestion or QoS Policy on Core Link. The physical link and BGP session remain "
      "operational, but packets are discarded before transmission.")


def test_only_the_link_failure_hypothesis_matches_real_run_text():
    assert matches(H1, FT, RC)
    for text in (H2, H3, H4):
        assert not matches(text, FT, RC)


def test_different_wordings_of_the_same_fault_match():
    assert matches("Physical Layer Failure on Core Interconnect", FT, RC)
    assert matches("The interface down condition on router1", FT, RC)
    assert matches("Link-Down event between the routers", FT, RC)   # hyphen/underscore normalization
    assert matches("A transceiver failure on the core port", FT, RC)


def test_root_cause_identifier_is_normalized_not_matched_literally():
    # previously the raw "link_down" was searched for, which prose never contains
    assert matches("the link down condition", None, "link_down")


def test_unknown_fault_types_still_match_on_their_own_label():
    assert matches("A bgp misconfiguration on router2", "bgp_misconfiguration", None)
    assert not matches("A link failure", "bgp_misconfiguration", None)


def C(outcome="disputed", cap=None, obs=None, status="supported", observable=None):
    return HypothesisChallenge(hypothesis_id="h1", original_status=status, original_confidence=0.9,
                               challenge_outcome=outcome, rebuttal="r", alternative_explanation="alt",
                               additional_evidence_needed=["history"],
                               distinguishing_capability=cap, distinguishing_observable=observable,
                               distinguishing_observation=obs)


def test_dispute_without_a_collectable_observation_is_downgraded():
    c = C(cap=None, obs="CRC history", observable="interface_error_count")
    notes = enforce_dispute_actionability([c])
    assert c.challenge_outcome == "upheld" and c.downgrade_note and notes
    c2 = C(cap="historical_telemetry", obs="CRC history", observable="x")     # not a registered capability
    enforce_dispute_actionability([c2])
    assert c2.challenge_outcome == "upheld"
    c3 = C(cap="deep_diagnostics", obs=" ", observable="interface_error_count")
    enforce_dispute_actionability([c3])
    assert c3.challenge_outcome == "upheld"


def test_dispute_naming_a_real_capability_and_observation_stays_disputed():
    c = C(cap="deep_diagnostics", obs="CRC counts on router2", observable="interface_error_count")
    assert enforce_dispute_actionability([c]) == []
    assert c.challenge_outcome == "disputed" and c.distinguishing_capability == "deep_diagnostics"
    assert c.downgrade_note is None


def test_dispute_asking_for_something_no_capability_returns_is_downgraded():
    # the real run: optical power requested "via deep_diagnostics"
    c = C(cap="deep_diagnostics", obs="optical power levels", observable="optical_power")
    notes = enforce_dispute_actionability([c])
    assert c.challenge_outcome == "upheld" and "optical_power" in c.downgrade_note and notes
    # a real capability with an observable that belongs to a DIFFERENT capability
    c2 = C(cap="reachability", obs="CRC counts", observable="interface_error_count")
    enforce_dispute_actionability([c2])
    assert c2.challenge_outcome == "upheld"


def test_other_outcomes_are_left_alone():
    c = C(outcome="unresolved")
    enforce_dispute_actionability([c])
    assert c.challenge_outcome == "unresolved"


def test_labels_say_what_upheld_means():
    assert describe_challenge_outcome(C("upheld", status="rejected")) == "upheld (rejection stands)"
    assert describe_challenge_outcome(C("upheld", status="supported")) == "upheld (support stands)"
    assert describe_challenge_outcome(C("disputed")) == "disputed"
    c = C(); enforce_dispute_actionability([c])
    assert "dispute downgraded" in describe_challenge_outcome(c)


def test_downgraded_dispute_is_kept_as_a_residual_concern():
    c = C(); enforce_dispute_actionability([c])
    report = ChallengeReport(incident_id="i", challenges=[c], systemic_concerns=[])
    out = residual_concerns(report, "h1")
    assert any("downgraded dispute" in x for x in out)
    assert any("history" in x for x in out)


def test_agent_downgrades_unactionable_dispute_end_to_end(monkeypatch):
    from investigation_team.agents.evidence_analyst import EvidenceAnalysis, HypothesisAssessment
    from investigation_team.domain import HypothesisStatus as S
    parsed = {"challenges": [{
        "hypothesis_id": "h1", "challenge_outcome": "disputed", "rebuttal": "transient vs permanent",
        "alternative_explanation": "transient fault", "distinguishing_capability": None, "distinguishing_observable": None,
        "distinguishing_observation": "CRC history", "additional_evidence_needed": ["CRC history"]}],
        "systemic_concerns": []}
    captured = {}
    def fake_llm(**kw):
        captured["system"] = kw["messages"][0]["content"]
        return {"content": "x"}
    monkeypatch.setattr(ch, "llm_call", fake_llm)
    monkeypatch.setattr(ch, "parse_or_repair_agent_response", lambda **kw: parsed)
    frame = NS(incident_id="inc", to_dict=lambda: {})
    hs = NS(hypotheses=[NS(hypothesis_id="h1", to_dict=lambda: {})])
    ea = EvidenceAnalysis(incident_id="inc", assessments=[
        HypothesisAssessment(hypothesis_id="h1", status=S.SUPPORTED, confidence=0.92, rationale="r")])
    report = ch.hypothesis_challenger_agent.__wrapped__(frame, hs, ea) if hasattr(
        ch.hypothesis_challenger_agent, "__wrapped__") else ch.hypothesis_challenger_agent(frame, hs, ea)
    assert report.challenges[0].challenge_outcome == "upheld"
    assert report.challenges[0].downgrade_note
    for name in ("network_state", "deep_diagnostics", "reachability", "vendor_alert", "topology"):
        assert name in captured["system"], "capability catalog must be in the Challenger prompt"
    assert "interface_error_count" in captured["system"] and "ping_success" in captured["system"]


def test_downgraded_dispute_lets_loop_finish_in_round_one(monkeypatch):
    from investigation_team.agents.evidence_analyst import EvidenceAnalysis, HypothesisAssessment
    from investigation_team.agents.evidence_planner import EvidenceGap, EvidenceGapSelection
    from investigation_team.domain import EvidencePlan, EvidenceTest, HypothesisStatus as S
    from tests.test_orchestrator_wiring import F, HS, _config
    monkeypatch.setattr(orch, "incident_framing_agent", lambda **kw: F())
    monkeypatch.setattr(orch, "hypothesis_generator_agent", lambda **kw: HS())
    monkeypatch.setattr(orch, "evidence_planning_agent",
                        lambda **kw: EvidenceGapSelection("inc", [EvidenceGap("h1", "d", "topology")]))
    n = {"rounds": 0}
    def create(**kw):
        n["rounds"] += 1
        return EvidencePlan(incident_id="inc", tests=[EvidenceTest(
            test_id=f"inc-r{kw['round_number']}-test-1", hypothesis_id="h1", objective="o", capability="topology")])
    monkeypatch.setattr(orch, "evidence_test_creation_agent", create)
    monkeypatch.setattr(orch, "execute_evidence_plan", lambda **kw: [])
    analysis = EvidenceAnalysis(incident_id="inc", assessments=[
        HypothesisAssessment(hypothesis_id="h1", status=S.SUPPORTED, confidence=0.92, rationale="r")])
    monkeypatch.setattr(orch, "evidence_analyst_agent", lambda **kw: analysis)
    c = C(); enforce_dispute_actionability([c])
    monkeypatch.setattr(orch, "hypothesis_challenger_agent",
                        lambda **kw: ChallengeReport(incident_id="inc", challenges=[c], systemic_concerns=[]))
    rec = NS(status="root_cause_identified", hypothesis_id="h1", confidence=0.92, remediation_steps=[],
             validate=lambda: None, to_dict=lambda: {})
    monkeypatch.setattr(orch, "root_cause_remediation_agent", lambda **kw: rec)
    r = orch.run_investigation(_config())
    assert n["rounds"] == 1
    assert any("downgraded dispute" in x for x in r.residual_concerns)


def test_qos_policy_misconfiguration_wording_matches_the_qos_ground_truth():
    text = ("QoS Policy Misconfiguration on WAN Edge. The QoS policy was misconfigured after the router upgrade, "
            "so voice is not prioritised.")
    assert matches(text, "qos_misconfiguration", "qos_policy_detached")
    assert not matches("WAN uplink congestion from a backup transfer", "qos_misconfiguration", "qos_policy_detached")


import pytest as _pytest


@_pytest.mark.parametrize("text", [
    "QoS policy regression on router1 WAN interface after the firmware upgrade",
    "Missing QoS policy on the WAN uplink",
    "The WAN-EDGE QoS policy was not reapplied after the reload",
    "QoS configuration absent from the egress interface",
    "Teams voice lost its QoS protection when the policy was removed",
])
def test_qos_wording_variants_match_structurally(text):
    assert matches(text, "qos_misconfiguration", "qos_policy_detached")


@_pytest.mark.parametrize("text", [
    "WAN uplink bandwidth contention from a backup transfer saturating the 100 Mbps link",
    "Switch port contention on switch1 causing queueing",
    "QoS marking on Teams traffic is correct and the policy is attached",
    "Packet loss on the internet path between router1 and the Teams relay",
])
def test_unrelated_hypotheses_do_not_match_the_qos_fault(text):
    assert not matches(text, "qos_misconfiguration", "qos_policy_detached")
