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



def test_recent_changes_are_scoped_sorted_and_labelled_as_leads():
    from investigation_team.domain import ChangeRecord
    from investigation_team.topology_context import recent_changes
    g = qos_graph()
    for cid, when, ents in (("c-old", "2026-09-01T10:00:00Z", [f"{P}:switch1"]),
                            ("c-new", "2026-09-13T22:00:00Z", [f"{P}:switch1", f"{P}:teams-relay"]),
                            ("c-away", "2026-09-13T23:00:00Z", [f"{P}:teams-relay"])):
        g.add_change(ChangeRecord(change_id=cid, source_id="s", changed_at=when, change_type="upgrade",
                                  entity_ids=ents, actor="a", attributes={"summary": cid + " summary"}))
    out = recent_changes(g, network_scope(g, [f"{P}:ws1"], hops=1))
    assert [c["change_id"] for c in out] == ["c-new", "c-old"]   # in scope only, newest first
    assert out[0]["entity_ids"] == [f"{P}:switch1"] and out[0]["summary"] == "c-new summary"
    assert recent_changes(g, network_scope(g, ["unknown"])) == []


def test_generation_context_carries_recent_changes():
    from investigation_team.domain import ChangeRecord
    g = qos_graph()
    g.add_change(ChangeRecord(change_id="c1", source_id="s", changed_at="2026-09-13T22:00:00Z",
                              change_type="upgrade", entity_ids=[f"{P}:router1"]))
    frame = IncidentFrame(incident_id="inc", investigation_question="q", affected_entity_ids=[f"{P}:ws1"],
                          potentially_relevant_entity_ids=[f"{P}:router1"], scope_summary="s")
    ctx = build_hypothesis_generation_context(frame, g, TopologyCapability())
    assert [c["change_id"] for c in ctx["recent_changes"]] == ["c1"]
