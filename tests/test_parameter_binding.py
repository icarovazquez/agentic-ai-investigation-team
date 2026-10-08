import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")
import json
import investigation_team  # noqa: registers capabilities
from investigation_team.capabilities.registry import bind_parameters, capability_parameter_specs

ENT = ["nika:simple_bgp:router1", "nika:simple_bgp:router2"]


def test_specs_come_from_signatures():
    specs = capability_parameter_specs()
    assert specs["deep_diagnostics"]["required"] == ["entity_ids"]
    assert specs["reachability"]["required"] == ["source_entity_id", "target_entity_id"]
    assert specs["topology"]["optional"] == ["target_entity_id"]


def test_invented_parameters_dropped_and_required_filled():
    params, notes = bind_parameters("deep_diagnostics", {"interfaces": ["eth0"], "log_time_window_start": "x"}, ENT)
    assert params == {"entity_ids": ENT}
    assert any("dropped" in n for n in notes)
    params, _ = bind_parameters("reachability", {"ping_source": "pc1"}, ENT)
    assert params == {"source_entity_id": ENT[0], "target_entity_id": ENT[1]}


def test_valid_parameters_untouched_and_str_coerced():
    p, n = bind_parameters("network_state", {"entity_ids": ENT[0]}, ENT)
    assert p == {"entity_ids": [ENT[0]]}
    p, n = bind_parameters("reachability", {"source_entity_id": "a", "target_entity_id": "b"}, ENT)
    assert p == {"source_entity_id": "a", "target_entity_id": "b"} and n == []


def test_garbage_model_output_now_executes_on_real_graph(monkeypatch):
    """The round-1 failure from the Colab run: every test failed on invented params."""
    from investigation_team.agents import evidence_test_creator as etc
    from investigation_team.agents.evidence_planner import EvidenceGap, EvidenceGapSelection
    from investigation_team.executors import execute_evidence_plan
    from investigation_team.adapters.base import load_investigation_evidence_executor
    from investigation_team.agents.hypothesis_generator import HypothesisRecord, HypothesisSet

    nb = json.load(open("notebooks/Scenario 1 - Simple BGP Link Down.ipynb"))
    ns = {}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if src.startswith("import pprint") or src.startswith("nika_simple_bgp_config ="):
                exec(src.replace("check_connection()", ""), ns)
    cfg = ns["nika_simple_bgp_config"]
    from investigation_team.evidence_graph import EvidenceGraph
    graph = load_investigation_evidence_executor(
        investigation_config=cfg, evidence_graph=EvidenceGraph()
    )["evidence_graph"]

    inc = cfg.investigation_id
    hyp = HypothesisRecord(hypothesis_id=f"{inc}-h1", incident_id=inc, title="t", proposed_cause="c",
                           causal_mechanism="m", suspected_entity_ids=ENT)
    hset = HypothesisSet.__new__(HypothesisSet); hset.incident_id = inc; hset.hypotheses = [hyp]

    class Frame:
        incident_id = inc
        def to_dict(self): return {}

    sel = EvidenceGapSelection(inc, [
        EvidenceGap(hyp.hypothesis_id, "a", "deep_diagnostics", 1),
        EvidenceGap(hyp.hypothesis_id, "b", "network_state", 2),
        EvidenceGap(hyp.hypothesis_id, "c", "reachability", 3),
    ])
    garbage = [
        {"capability": "deep_diagnostics", "parameters": {"interfaces": ["x"], "log_time_window_start": "t"}},
        {"capability": "network_state", "parameters": {"bgp_session": "y"}},
        {"capability": "reachability", "parameters": {"ping_source": "pc1"}},
    ]
    monkeypatch.setattr(etc, "llm_call", lambda **kw: {"content": "x"})
    monkeypatch.setattr(etc, "parse_or_repair_agent_response", lambda **kw: {"tests": garbage})
    plan = etc.evidence_test_creation_agent(Frame(), hset, sel)
    results = execute_evidence_plan(plan, graph)
    assert [r.status for r in results] == ["completed"] * 3, [(r.capability, r.error) for r in results]
