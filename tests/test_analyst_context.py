import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")
import json
import investigation_team  # noqa
from investigation_team.agents.evidence_analyst import build_evidence_analysis_context
from investigation_team.agents.hypothesis_generator import HypothesisRecord, HypothesisSet
from investigation_team.capabilities.registry import get_capability
from investigation_team.domain import EvidencePlan, EvidenceTestResult
from investigation_team.evidence_graph import EvidenceGraph
from investigation_team.adapters.base import load_investigation_evidence_executor

R1, R2 = "nika:simple_bgp:router1", "nika:simple_bgp:router2"


def _graph():
    nb = json.load(open("notebooks/Scenario 1 - Simple BGP Link Down.ipynb")); ns = {}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if s.startswith("import pprint") or s.startswith("nika_simple_bgp_config ="):
                exec(s.replace("check_connection()", ""), ns)
    cfg = ns["nika_simple_bgp_config"]
    return cfg, load_investigation_evidence_executor(investigation_config=cfg, evidence_graph=EvidenceGraph())["evidence_graph"]


def test_analyst_sees_deep_diagnostics_and_vendor_alert_evidence():
    cfg, g = _graph()
    inc = cfg.investigation_id
    hyp = HypothesisRecord(hypothesis_id=f"{inc}-h1", incident_id=inc, title="t", proposed_cause="c", causal_mechanism="m")
    hset = HypothesisSet.__new__(HypothesisSet); hset.incident_id = inc; hset.hypotheses = [hyp]
    results = []
    for i, name in enumerate(["deep_diagnostics", "network_state", "vendor_alert"], start=1):
        ev = get_capability(name).collect_evidence(evidence_graph=g, entity_ids=[R1, R2])
        results.append(EvidenceTestResult(test_id=f"t{i}", hypothesis_id=hyp.hypothesis_id, capability=name, status="completed", evidence=ev))
    ctx = build_evidence_analysis_context(hset, EvidencePlan(incident_id=inc, tests=[]), results)
    tests = {t["capability"]: t for t in ctx["hypotheses"][hyp.hypothesis_id]["tests"]}
    metrics = lambda t: {o["metric_name"] for o in t["observations"]}
    assert {"bgp_session_state", "interface_error_count", "interface_admin_status"} <= metrics(tests["deep_diagnostics"])
    assert "interface_oper_status" in metrics(tests["network_state"])
    assert tests["vendor_alert"]["events"], "vendor alert events must reach the analyst"
