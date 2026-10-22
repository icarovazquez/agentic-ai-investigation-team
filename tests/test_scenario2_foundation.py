import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

import pytest

from investigation_team import adapters, capabilities  # noqa: F401  (register everything)
from investigation_team.adapters.base import load_evidence_source_executor
from investigation_team.capabilities.registry import capability_observables, get_capability
from investigation_team.config import EvidenceSourceConfig
from investigation_team.domain import AccessMode, EvidenceType, SourceFormat
from investigation_team.entity_resolution import EntityResolver
from investigation_team.evidence_graph import EvidenceGraph

P = "nika:qos_contention"


def src(source_id, adapter, etype=EvidenceType.TELEMETRY):
    return EvidenceSourceConfig(source_id=source_id, evidence_type=etype, source_format=SourceFormat.API,
                                access_mode=AccessMode.API, adapter_name=adapter,
                                connection_parameters={"provider": "fixture"},
                                metadata={"scenario_id": "qos_contention"})


@pytest.fixture
def graph():
    g = EvidenceGraph()
    load_evidence_source_executor(src("topo", "NIKAQoSContentionAdapter", EvidenceType.TOPOLOGY), g)
    return g


# ---- resolver ----

def test_resolver_handles_every_identifier_form_for_one_entity(graph):
    r = EntityResolver.from_graph(graph)
    for raw in ["ws1", "WS-0142", "ws-0142.corp.example.com", "MARIA-LAPTOP.corp.example.com",
                "Maria.Lopez@corp.example.com", "AA-BB-CC-00-01-42", "aabb.cc00.0142", "10.1.1.11",
                "Maria’s desk", f"{P}:ws1"]:
        res = r.resolve(raw)
        assert res.resolved and res.entity_id == f"{P}:ws1", raw


def test_unknown_identifier_is_unresolved_not_guessed(graph):
    res = EntityResolver.from_graph(graph).resolve("UNKNOWN-DEVICE-77.corp.example.com")
    assert res.status == "unresolved" and res.entity_id is None


def test_an_alias_claimed_by_two_entities_is_ambiguous():
    r = EntityResolver()
    r.register("a", ["shared-name"])
    r.register("b", ["shared-name"])
    res = r.resolve("shared-name")
    assert res.status == "ambiguous" and set(res.candidates) == {"a", "b"}


# ---- topology + operational ----

def test_topology_has_three_workstations_and_the_wan_uplink(graph):
    names = {e.name for e in graph.list_entities()}
    assert {"ws1", "ws2", "ws3", "switch1", "router1", "teams-relay"} == names
    assert graph.shortest_path(f"{P}:ws1", f"{P}:teams-relay")[-2:] == [f"{P}:router1", f"{P}:teams-relay"]


def test_operational_state_looks_healthy(graph):
    load_evidence_source_executor(src("ops", "NIKAQoSContentionOperationalAdapter"), graph)
    obs = graph.get_observations(entity_ids=[f"{P}:router1"])
    assert obs and all(o.metric_value in ("up", True) for o in obs)


# ---- teams ----

def test_teams_records_attach_to_the_right_endpoints_and_unknown_one_is_reported(graph):
    out = load_evidence_source_executor(src("teams", "TeamsCallQualityAdapter"), graph)
    warnings = out["ingestion_result"]["warnings"]
    assert len(warnings) == 1 and "UNKNOWN-DEVICE-77" in warnings[0] and "c-0905" in warnings[0]

    # 4 resolvable calls x 3 metrics; the unresolved call produced none
    assert out["adapter_summary"]["observations"] == 12

    def calls(ws):
        return {o.dimensions["call_id"] for o in graph.get_observations(entity_ids=[f"{P}:{ws}"])
                if o.metric_name == "call_jitter_ms"}
    assert calls("ws1") == {"c-0801", "c-0902", "c-0904"}   # by FQDN, by UPN, by asset tag
    assert calls("ws2") == {"c-0903"}                        # by MAC
    assert calls("ws3") == set()                             # ws3 makes no calls


def test_adapter_without_a_resolver_fails_loudly():
    from investigation_team.adapters.connectors.teams import TeamsCallQualityAdapter
    with pytest.raises(RuntimeError, match="needs an entity resolver"):
        TeamsCallQualityAdapter(src("teams", "TeamsCallQualityAdapter")).load()


# ---- capability ----

def test_call_quality_capability_returns_calls_and_declares_observables(graph):
    load_evidence_source_executor(src("teams", "TeamsCallQualityAdapter"), graph)
    out = get_capability("call_quality").collect_evidence(graph, [f"{P}:ws1", f"{P}:ws3"])
    payload = out["call_quality_evidence"]
    assert out["evidence_status"] == "available" and payload["observation_count"] == 9
    assert {o["entity_id"] for o in payload["observations"]} == {f"{P}:ws1"}
    assert "call_packet_loss_pct" in capability_observables()["call_quality"]
    empty = get_capability("call_quality").collect_evidence(graph, [f"{P}:ws3"])
    assert empty["evidence_status"] == "insufficient_evidence"
