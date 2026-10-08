"""Offline tests for the planner/test-creator split. No LLM calls."""
import os

for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

import pytest

from investigation_team.agents import evidence_planner as ep
from investigation_team.agents import evidence_test_creator as etc
from investigation_team.agents.evidence_planner import (
    EvidenceGap, EvidenceGapSelection, select_capped_gaps,
    MAX_GAPS_PER_HYPOTHESIS, MAX_TOTAL_GAPS,
)
from investigation_team.agents.hypothesis_generator import HypothesisRecord, HypothesisSet
from investigation_team.agents.incident_framing import IncidentFrame

INC = "inc-1"


def make_hset(n=5):
    hs = [
        HypothesisRecord(hypothesis_id=f"{INC}-h{i}", incident_id=INC, title=f"Title {i}",
                         proposed_cause="c", causal_mechanism="m")
        for i in range(1, n + 1)
    ]
    hset = HypothesisSet.__new__(HypothesisSet)
    hset.incident_id = INC
    hset.hypotheses = hs
    return hset


def gap(h, prio=1, cap="topology"):
    return {"hypothesis_id": h, "gap_description": "d", "suggested_capability": cap, "priority": prio}


def test_per_hypothesis_cap():
    out = select_capped_gaps([gap("H1", p) for p in range(1, 6)], make_hset())
    assert len(out) == MAX_GAPS_PER_HYPOTHESIS
    assert [g["priority"] for g in out] == [1, 2]
    assert all(g["hypothesis_id"] == f"{INC}-h1" for g in out)


def test_total_cap():
    raw = [gap(f"H{h}", p) for h in range(1, 6) for p in (1, 2)]  # 10 gaps
    out = select_capped_gaps(raw, make_hset())
    assert len(out) == MAX_TOTAL_GAPS


def test_composite_and_unresolvable_and_missing_skipped():
    raw = [gap("H1_H2_H4"), gap("nonsense"), {"hypothesis_id": "H1"}, gap("H3")]
    out = select_capped_gaps(raw, make_hset())
    assert [g["hypothesis_id"] for g in out] == [f"{INC}-h3"]


def test_title_and_canonical_ids_resolve():
    out = select_capped_gaps([gap("Title 2"), gap(f"{INC}-h4")], make_hset())
    assert {g["hypothesis_id"] for g in out} == {f"{INC}-h2", f"{INC}-h4"}


def test_bad_priority_defaults():
    out = select_capped_gaps([gap("H1", "high")], make_hset())
    assert len(out) == 1


class Frame:
    incident_id = INC
    def to_dict(self): return {}


def _selection():
    return EvidenceGapSelection(INC, [
        EvidenceGap(f"{INC}-h2", "gap A", "topology", 1),
        EvidenceGap(f"{INC}-h1", "gap B", "reachability", 2),
        EvidenceGap(f"{INC}-h3", "gap C", "network_state", 3),
    ])


def _run_creator(monkeypatch, tests):
    monkeypatch.setattr(etc, "llm_call", lambda **kw: {"content": "x"})
    monkeypatch.setattr(etc, "parse_or_repair_agent_response", lambda **kw: {"tests": tests})
    return etc.evidence_test_creation_agent.__wrapped__(Frame(), make_hset(), _selection(), round_number=2) \
        if hasattr(etc.evidence_test_creation_agent, "__wrapped__") \
        else etc.evidence_test_creation_agent(Frame(), make_hset(), _selection(), round_number=2)


def test_creator_uses_gap_hypothesis_id_and_position(monkeypatch):
    tests = [
        {"hypothesis_id": "H1_H2_H4", "objective": "o1", "capability": "topology"},
        {"hypothesis_id": f"{INC}-h1", "objective": "o2", "capability": "reachability"},
        {"objective": "o3", "capability": "network_state"},
    ]
    plan = _run_creator(monkeypatch, tests)
    assert [t.hypothesis_id for t in plan.tests] == [f"{INC}-h2", f"{INC}-h1", f"{INC}-h3"]
    assert [t.priority for t in plan.tests] == [1, 2, 3]
    assert plan.tests[0].test_id == f"{INC}-r2-test-1"


def test_creator_count_mismatch_truncates_and_defaults(monkeypatch):
    plan = _run_creator(monkeypatch, [{"capability": "topology"}] * 5)
    assert len(plan.tests) == 3
    plan = _run_creator(monkeypatch, [{}])
    assert len(plan.tests) == 1
    assert plan.tests[0].objective == "gap A" and plan.tests[0].capability == "topology"


def test_planner_end_to_end_stubbed(monkeypatch):
    monkeypatch.setattr(ep, "llm_call", lambda **kw: {"content": "x"})
    monkeypatch.setattr(ep, "parse_or_repair_agent_response",
                        lambda **kw: {"gaps": [gap(f"H{h}", p) for h in range(1, 6) for p in (1, 2, 3)]})
    sel = ep.evidence_planning_agent(Frame(), make_hset())
    assert len(sel.gaps) == MAX_TOTAL_GAPS


def test_token_budgets_and_models():
    from investigation_team.llm import AGENT_MAX_TOKENS, AGENT_MODEL_MAP
    assert AGENT_MAX_TOKENS["evidence_planning_agent"] == 2500
    assert AGENT_MAX_TOKENS["evidence_test_creation_agent"] == 5000
    assert "evidence_test_creation_agent" in AGENT_MODEL_MAP


def test_orchestrator_imports_and_wiring():
    import inspect
    from investigation_team import orchestrator
    src = inspect.getsource(orchestrator)
    assert "evidence_gap_selection_round_" in src
    assert "evidence_test_creation_round_" in src
    assert "round_number=round_number,\n                prior_evidence_analysis" not in src
