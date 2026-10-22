import os
for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.setdefault(k, "dummy")

import investigation_team  # noqa: F401
from investigation_team.adapters.base import load_evidence_source_executor
from investigation_team.agents.hypothesis_generator import build_hypothesis_generation_context
from investigation_team.agents.incident_framing import IncidentFrame
from investigation_team.capabilities.topology import TopologyCapability
from investigation_team.domain import EvidenceType
from investigation_team.evidence_graph import EvidenceGraph
from investigation_team.topology_context import network_scope
from tests.test_scenario2_foundation import P, src


def qos_graph():
    g = EvidenceGraph()
    load_evidence_source_executor(src("topo", "NIKAQoSContentionAdapter", EvidenceType.TOPOLOGY), g)
    return g


def test_single_reported_endpoint_still_exposes_shared_infrastructure():
    scope = network_scope(qos_graph(), [f"{P}:ws1"])
    ids = {e["entity_id"] for e in scope["entities"]}
    # before this existed the context contained ws1 and nothing else
    assert {f"{P}:{n}" for n in ("ws1", "ws2", "ws3", "switch1", "router1", "teams-relay")} <= ids
    [path] = scope["paths_to_services"]
    assert path["path"] == [f"{P}:{n}" for n in ("ws1", "switch1", "router1", "teams-relay")]
    uplink = [r for r in scope["relationships"] if r.get("link_name") == "wan-uplink"]
    assert uplink and uplink[0]["capacity_mbps"] == 100


def test_scope_is_bounded_and_nearest_first():
    scope = network_scope(qos_graph(), [f"{P}:ws1"], hops=1)
    assert {e["name"] for e in scope["entities"]} == {"ws1", "switch1"}
    capped = network_scope(qos_graph(), [f"{P}:ws1"], max_entities=3)
    assert len(capped["entities"]) == 3 and capped["entities"][0]["entity_id"] == f"{P}:ws1"


def test_unknown_seed_gives_an_empty_scope_not_an_error():
    assert network_scope(qos_graph(), ["nope"])["entities"] == []


def test_generator_context_carries_the_scope():
    frame = IncidentFrame.__new__(IncidentFrame)
    frame.incident_id = "i"
    frame.affected_entity_ids = [f"{P}:ws1"]
    frame.potentially_relevant_entity_ids = []
    frame.to_dict = lambda: {}
    ctx = build_hypothesis_generation_context(frame, qos_graph(), TopologyCapability())
    assert f"{P}:router1" in {e["entity_id"] for e in ctx["network_scope"]["entities"]}
