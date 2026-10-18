import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import investigation_team  # noqa
from test_analyst_context import _graph
from investigation_team.capabilities.registry import get_capability

PC1, PC2 = "nika:simple_bgp:pc1", "nika:simple_bgp:pc2"
R1, R2 = "nika:simple_bgp:router1", "nika:simple_bgp:router2"


def test_end_to_end_question_now_surfaces_the_failing_segment():
    g = _graph()[1]
    ev = get_capability("reachability").collect_evidence(evidence_graph=g, source_entity_id=PC1, target_entity_id=PC2)
    assert ev["evidence_status"] == "available"          # was "insufficient_evidence"
    obs = ev["reachability_evidence"]["observations"]
    pairs = {(o["entity_id"], o["dimensions"]["target_entity_id"]): o["metric_value"] for o in obs}
    assert pairs[(R1, R2)] is False                      # the decisive measurement
    assert pairs[(PC1, R1)] is True and pairs[(R2, PC2)] is True
    matches = ev["reachability_evidence"]["match_by_observation_id"]
    assert set(matches.values()) == {"path_segment"}     # none is an exact pc1->pc2 test


def test_exact_match_is_labelled_exact_and_unrelated_pairs_excluded():
    from investigation_team.executors import reachability_evidence_executor
    g = _graph()[1]
    out = reachability_evidence_executor(g, R1, R2, path_entity_ids=[R1, R2])
    assert list(out["match_by_observation_id"].values()) == ["exact"]
    assert out["observation_count"] == 1
    out = reachability_evidence_executor(g, PC1, R1, path_entity_ids=[PC1, R1])  # only pc1->router1
    assert out["observation_count"] == 1


def test_analyst_context_carries_match_labels():
    from investigation_team.agents.evidence_analyst import build_evidence_analysis_context
    from investigation_team.agents.hypothesis_generator import HypothesisRecord, HypothesisSet
    from investigation_team.domain import EvidencePlan, EvidenceTestResult
    cfg, g = _graph()
    inc = cfg.investigation_id
    h = HypothesisRecord(hypothesis_id=f"{inc}-h1", incident_id=inc, title="t", proposed_cause="c", causal_mechanism="m")
    hs = HypothesisSet.__new__(HypothesisSet); hs.incident_id = inc; hs.hypotheses = [h]
    ev = get_capability("reachability").collect_evidence(evidence_graph=g, source_entity_id=PC1, target_entity_id=PC2)
    ctx = build_evidence_analysis_context(hs, EvidencePlan(incident_id=inc, tests=[]),
        [EvidenceTestResult(test_id="t", hypothesis_id=h.hypothesis_id, capability="reachability", status="completed", evidence=ev)])
    t = ctx["hypotheses"][h.hypothesis_id]["tests"][0]
    assert set(t["observation_match"].values()) == {"path_segment"} and len(t["observation_ids"]) == 3
