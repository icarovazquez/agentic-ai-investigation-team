"""
NIKA qos_contention connectors (Scenario 2): topology and first-response
operational state.

Topology:

    ws1 --\
    ws2 ---+-- switch1 -- router1 == wan-uplink (100 Mbps) == teams-relay
    ws3 --/

The topology source is the system of record for names: each entity
declares the identifiers other systems use for it (asset tags, UPNs,
hostnames, MACs, IPs, what people call the desk) in attributes["aliases"].
Every other adapter resolves through those (see entity_resolution.py), so
no adapter keeps a private name table.

Version 1 is a deterministic fixture, like the Scenario 1 connectors.
"""

from __future__ import annotations

from ...domain import EntityType, NetworkEntity, NetworkRelationship, ObservationRecord, RelationshipType
from ..base import EvidenceAdapter, EvidenceAdapterResult, register_adapter

P = "nika:qos_contention"


def _check_scenario(adapter: EvidenceAdapter, name: str) -> str:
    scenario_id = adapter.source_config.metadata.get("scenario_id")
    if scenario_id != "qos_contention":
        raise ValueError(f"{name} only supports scenario_id='qos_contention'.")
    return scenario_id


class NIKAQoSContentionAdapter(EvidenceAdapter):
    """Topology + identifier aliases for the qos_contention scenario."""

    def load(self) -> EvidenceAdapterResult:
        _check_scenario(self, "NIKAQoSContentionAdapter")

        def host(key, name, aliases, owner):
            return NetworkEntity(
                entity_id=f"{P}:{key}",
                entity_type=EntityType.HOST,
                name=name,
                attributes={"provider": "nika", "scenario": "qos_contention", "role": "workstation",
                            "owner": owner, "aliases": aliases},
                source_ids=[self.source_id],
            )

        def device(key, name, role, aliases):
            return NetworkEntity(
                entity_id=f"{P}:{key}",
                entity_type=EntityType.DEVICE,
                name=name,
                attributes={"provider": "nika", "scenario": "qos_contention", "role": role, "aliases": aliases},
                source_ids=[self.source_id],
            )

        entities = [
            host("ws1", "ws1", ["WS-0142", "maria.lopez@corp.example.com", "MARIA-LAPTOP",
                                "aa:bb:cc:00:01:42", "10.1.1.11", "Maria's desk"], "Maria Lopez"),
            host("ws2", "ws2", ["WS-0143", "ravi.patel@corp.example.com", "RAVI-DESKTOP",
                                "aa:bb:cc:00:01:43", "10.1.1.12", "Ravi's desk"], "Ravi Patel"),
            host("ws3", "ws3", ["WS-0187", "sam.okafor@corp.example.com", "SAM-DESKTOP",
                                "aa:bb:cc:00:01:87", "10.1.1.13", "Sam's desk"], "Sam Okafor"),
            device("switch1", "switch1", "access_switch", ["SW-ACCESS-01", "10.1.1.2"]),
            device("router1", "router1", "edge_router", ["RTR-EDGE-01", "10.1.1.1"]),
            NetworkEntity(
                entity_id=f"{P}:teams-relay",
                entity_type=EntityType.SERVICE,
                name="teams-relay",
                attributes={"provider": "nika", "scenario": "qos_contention", "role": "media_relay",
                            "aliases": ["teams-media-relay", "Microsoft Teams media relay"]},
                source_ids=[self.source_id],
            ),
        ]

        def link(a, b, **attrs):
            return NetworkRelationship(
                relationship_id=f"{P}:{a}-{b}",
                source_entity_id=f"{P}:{a}",
                target_entity_id=f"{P}:{b}",
                relationship_type=RelationshipType.CONNECTED_TO,
                attributes={"bidirectional": True, **attrs},
                source_ids=[self.source_id],
            )

        relationships = [
            link("ws1", "switch1"),
            link("ws2", "switch1"),
            link("ws3", "switch1"),
            link("switch1", "router1"),
            link("router1", "teams-relay", link_name="wan-uplink", capacity_mbps=100),
        ]

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            entities=entities,
            relationships=relationships,
            metadata={"provider": "nika", "scenario_id": "qos_contention",
                      "adapter": self.__class__.__name__, "integration_stage": "static_topology"},
        )
        self.validate_result(result)
        return result


class NIKAQoSContentionOperationalAdapter(EvidenceAdapter):
    """
    First-response operational state: everything is up and reachable.
    That is the point of the scenario. There is no hard failure to find,
    only degraded quality, so the first look at the network says
    "healthy".
    """

    def load(self) -> EvidenceAdapterResult:
        _check_scenario(self, "NIKAQoSContentionOperationalAdapter")
        observed_at = self.source_config.metadata.get("observed_at", "2026-09-14T09:50:00Z")

        def oper(entity, peer):
            return ObservationRecord(
                observation_id=f"{P}:obs:{entity}-{peer}-interface-state",
                source_id=self.source_id, entity_id=f"{P}:{entity}",
                metric_name="interface_oper_status", metric_value="up", observed_at=observed_at,
                dimensions={"peer_entity_id": f"{P}:{peer}"})

        def ping(entity, target):
            return ObservationRecord(
                observation_id=f"{P}:obs:{entity}-{target}-ping",
                source_id=self.source_id, entity_id=f"{P}:{entity}",
                metric_name="ping_success", metric_value=True, observed_at=observed_at,
                dimensions={"target_entity_id": f"{P}:{target}"})

        observations = [
            oper("switch1", "router1"), oper("router1", "switch1"), oper("router1", "teams-relay"),
            ping("ws1", "router1"), ping("ws2", "router1"), ping("ws3", "router1"),
            ping("router1", "teams-relay"),
        ]

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            observations=observations,
            metadata={"provider": "nika", "scenario_id": "qos_contention",
                      "adapter": self.__class__.__name__,
                      "integration_stage": "operational_evidence_fixture", "evidence_tier": "first_response"},
        )
        self.validate_result(result)
        return result


register_adapter("NIKAQoSContentionAdapter", NIKAQoSContentionAdapter)
register_adapter("NIKAQoSContentionOperationalAdapter", NIKAQoSContentionOperationalAdapter)
