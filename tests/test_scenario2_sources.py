import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

import pytest

import investigation_team  # noqa: F401
from investigation_team.adapters.base import ingest_adapter_result_executor
from investigation_team.adapters.connectors import servicenow as sn
from investigation_team.adapters.connectors.servicenow import ServiceNowTicketAdapter
from investigation_team.agents.evidence_analyst import build_evidence_analysis_context
from investigation_team.agents.hypothesis_generator import HypothesisRecord, HypothesisSet
from investigation_team.capabilities.registry import capability_observables, get_capability
from investigation_team.config import EvidenceSourceConfig
from investigation_team.domain import AccessMode, EvidencePlan, EvidenceTestResult, EvidenceType, SourceFormat
from investigation_team.entity_resolution import EntityResolver
from investigation_team.evidence_graph import EvidenceGraph
from tests.test_scenario2_foundation import P, src


def stub_extractor(raw):
    if raw["kind"] == "incident":
        return {"reporter": "Maria Lopez",
                "affected_endpoints": ["WS-0142", "Ravi's desk"],       # second one is NOT in the text
                "symptoms": ["Teams calls breaking up", "video freezes"],
                "reporter_theories": ["headset is dying", "laptop dock", "VPN client", "webcam"],
                "onset": "about 9 this morning", "change_type": None, "actor": None,
                "change_summary": None, "change_details": []}
    return {"reporter": "J. Alvarez", "affected_endpoints": ["RTR-EDGE-01"], "symptoms": [],
            "reporter_theories": [], "onset": None, "change_type": "software_upgrade",
            "actor": "J. Alvarez", "change_summary": "Upgraded RTR-EDGE-01 from 17.6.4 to 17.9.3.",
            "change_details": ["Interface configurations were reapplied from the standard branch template"]}


def make_adapter(graph, extractor=stub_extractor):
    a = ServiceNowTicketAdapter(src("tickets", "ServiceNowTicketAdapter", EvidenceType.EVENT))
    a.entity_resolver = EntityResolver.from_graph(graph)
    a.extractor = extractor
    return a


@pytest.fixture
def graph():
    from investigation_team.adapters.base import load_evidence_source_executor
    g = EvidenceGraph()
    load_evidence_source_executor(src("topo", "NIKAQoSContentionAdapter", EvidenceType.TOPOLOGY), g)
    return g


@pytest.fixture
def loaded(graph):
    result = make_adapter(graph).load()
    ingest_adapter_result_executor(evidence_graph=graph, adapter_result=result)
    return graph, result


# ---- ServiceNow adapter ----

def test_incident_becomes_an_event_on_the_right_workstation_and_theories_stay_claims(loaded):
    graph, result = loaded
    [event] = result.events
    assert event.entity_ids == [f"{P}:ws1"]
    assert event.event_type == "service_ticket"
    assert "headset" in " ".join(event.attributes["unverified_reporter_theories"])
    # a reporter's guess must not leak into the message or the symptoms
    assert "headset" not in event.message.lower()
    assert not any("headset" in s.lower() for s in event.attributes["symptoms"])


def test_change_ticket_becomes_a_change_record_on_the_router(loaded):
    graph, result = loaded
    [change] = result.changes
    assert change.entity_ids == [f"{P}:router1"] and change.change_type == "software_upgrade"
    assert change.actor == "J. Alvarez"
    assert any("template" in d for d in change.attributes["details_as_recorded"])
    assert graph.get_changes(entity_ids=[f"{P}:router1"])


def test_endpoint_the_model_invented_is_discarded_with_a_warning(loaded):
    _, result = loaded
    assert any("Ravi's desk" in w and "does not appear in the ticket text" in w for w in result.warnings)


def test_endpoint_in_text_but_unknown_to_the_resolver_is_reported(graph, monkeypatch):
    tickets = [dict(sn.RAW_TICKETS[0], description=sn.RAW_TICKETS[0]["description"] + " Also seen on WS-9999.")]
    monkeypatch.setattr(sn, "RAW_TICKETS", tickets)
    ext = lambda raw: dict(stub_extractor(raw), affected_endpoints=["WS-0142", "WS-9999"])
    result = make_adapter(graph, ext).load()
    assert [e.entity_ids for e in result.events] == [[f"{P}:ws1"]]
    assert any("WS-9999" in w and "does not match any known entity" in w for w in result.warnings)


def test_ticket_with_no_resolvable_endpoint_is_skipped_and_reported(graph):
    ext = lambda raw: dict(stub_extractor(raw), affected_endpoints=[])
    result = make_adapter(graph, ext).load()
    assert not result.events and not result.changes
    assert sum("ticket skipped" in w for w in result.warnings) == 2


def test_malformed_extraction_fails_loudly(graph):
    bad = lambda raw: dict(stub_extractor(raw), affected_endpoints="WS-0142")
    with pytest.raises(ValueError, match="affected_endpoints"):
        make_adapter(graph, bad).load()
    no_type = lambda raw: dict(stub_extractor(raw), change_type=None)
    with pytest.raises(ValueError, match="change_type"):
        make_adapter(graph, no_type).load()


# ---- capabilities ----

def test_service_desk_and_change_history_and_vendor_alert_are_partitioned(loaded):
    graph, _ = loaded
    desk = get_capability("service_desk").collect_evidence(graph, [f"{P}:ws1"])
    assert desk["service_desk_evidence"]["event_count"] == 1
    chg = get_capability("change_history").collect_evidence(graph, [f"{P}:router1"])
    assert chg["change_history_evidence"]["change_count"] == 1
    # tickets are not vendor alerts
    va = get_capability("vendor_alert").collect_evidence(graph, [f"{P}:ws1"])
    assert va["vendor_alert_evidence"]["event_count"] == 0
    assert va["evidence_status"] == "insufficient_evidence"


def test_traffic_capability_fixture_tells_the_detached_policy_story(graph):
    cap = get_capability("traffic")
    out = cap.collect_evidence(graph, [f"{P}:router1", f"{P}:ws1", f"{P}:ws2", f"{P}:ws3"])
    obs = {(o["entity_id"].split(":")[-1], o["metric_name"]): o for o in out["traffic_evidence"]["observations"]}
    assert obs[("router1", "link_utilization_pct")]["metric_value"] == 97
    assert obs[("router1", "service_policy_attached")]["metric_value"] is False
    assert obs[("ws3", "top_talker_bps")]["metric_value"] > 50_000_000
    assert obs[("ws1", "dscp_marking")]["metric_value"] == "EF"        # clients mark voice correctly
    # idempotent across rounds: second call returns the same evidence, adds nothing new
    again = cap.collect_evidence(graph, [f"{P}:router1"])
    assert again["traffic_evidence"]["observation_count"] == 3
    assert graph.summary()["observations"] == len(out["traffic_evidence"]["observations"])


def test_new_capabilities_declare_observables():
    o = capability_observables()
    assert o["traffic"] and "service_policy_attached" in o["traffic"]
    assert o["service_desk"] == ("service_ticket",) and o["change_history"] == ("change_record",)


def test_analyst_context_carries_changes_and_tickets(loaded):
    graph, _ = loaded
    inc = "i"
    hyp = HypothesisRecord(hypothesis_id="h1", incident_id=inc, title="t", proposed_cause="c", causal_mechanism="m")
    hset = HypothesisSet.__new__(HypothesisSet); hset.incident_id = inc; hset.hypotheses = [hyp]
    results = [
        EvidenceTestResult(test_id="a", hypothesis_id="h1", capability="change_history", status="completed",
                           evidence=get_capability("change_history").collect_evidence(graph, [f"{P}:router1"])),
        EvidenceTestResult(test_id="b", hypothesis_id="h1", capability="service_desk", status="completed",
                           evidence=get_capability("service_desk").collect_evidence(graph, [f"{P}:ws1"])),
    ]
    ctx = build_evidence_analysis_context(hset, EvidencePlan(incident_id=inc, tests=[]), results)
    assert [c["change_id"] for c in ctx["shared_changes"]] == ["servicenow:CHG0031207"]
    assert [e["event_id"] for e in ctx["shared_events"]] == ["servicenow:INC0045821"]
    tests = {t["capability"]: t for t in ctx["hypotheses"]["h1"]["tests"]}
    assert tests["change_history"]["change_ids"] == ["servicenow:CHG0031207"]


# ---- Scenario 2 notebook config, end to end through the loader (no LLM) ----

def _scenario2_config():
    import json
    nb = json.load(open("notebooks/Scenario 2 - QoS Contention.ipynb")); ns = {}
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if s.startswith("import pprint") or s.startswith("nika_qos_contention_config ="):
                exec(s.replace("check_connection()", ""), ns)
    return ns["nika_qos_contention_config"]


def test_scenario2_config_loads_all_four_sources_in_order(monkeypatch, capsys):
    from investigation_team.adapters.base import load_investigation_evidence_executor
    monkeypatch.setattr(sn, "llm_extract_ticket", stub_extractor)
    cfg = _scenario2_config()
    out = load_investigation_evidence_executor(investigation_config=cfg, evidence_graph=EvidenceGraph())
    assert out["source_errors"] == {}
    assert out["graph_summary"] == {"entities": 6, "relationships": 5, "observations": 19, "events": 1, "changes": 1}
    printed = capsys.readouterr().out
    assert "UNKNOWN-DEVICE-77" in printed            # Teams record that could not be attached
    assert "Ravi's desk" in printed                  # endpoint the model invented in the ticket


def test_optional_source_failure_says_why(monkeypatch, capsys):
    from investigation_team.adapters.base import load_investigation_evidence_executor
    def boom(raw):
        raise RuntimeError("model returned nonsense")
    monkeypatch.setattr(sn, "llm_extract_ticket", boom)
    out = load_investigation_evidence_executor(investigation_config=_scenario2_config(), evidence_graph=EvidenceGraph())
    assert "servicenow_tickets" in out["source_errors"]
    assert "model returned nonsense" in capsys.readouterr().out


def test_matcher_recognises_the_detached_policy_in_varied_wording():
    from investigation_team.evaluation import hypothesis_text_matches_fault as m
    ft, rc = "qos_misconfiguration", "qos_policy_detached"
    assert m("QoS service-policy not attached to WAN interface after the router upgrade", ft, rc)
    assert m("The WAN QoS policy was detached during the change", ft, rc)
    assert m("Missing QoS on the uplink leaves voice unprotected", ft, rc)
    assert not m("Bulk transfer from ws3 saturates the WAN uplink", ft, rc)
    assert not m("Teams DSCP marking is not applied by clients", ft, rc)
    assert not m("Maria's headset hardware is failing", ft, rc)
