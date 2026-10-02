"""
NIKA network telemetry connector: topology + first-response
operational state. Deep diagnostics (BGP session state, error
counters, admin status) live in capabilities/deep_diagnostics.py
instead, since that data isn't loaded upfront — it's fetched on
explicit request, mirroring a real diagnostic escalation.
"""

from __future__ import annotations

from ...domain import EntityType, NetworkEntity, NetworkRelationship, ObservationRecord, RelationshipType
from ..base import EvidenceAdapter, EvidenceAdapterResult, register_adapter


class NIKASimpleBGPAdapter(EvidenceAdapter):
    """
    Normalize NIKA's simple_bgp scenario into Evidence Graph
    domain records.

    Initial topology:

        pc1 -- router1 -- router2 -- pc2

    This adapter is intentionally narrow. It validates the NIKA
    integration path before we generalize into a reusable NIKAAdapter.
    """

    def load(self) -> EvidenceAdapterResult:
        scenario_id = self.source_config.metadata.get("scenario_id")

        if scenario_id != "simple_bgp":
            raise ValueError(
                "NIKASimpleBGPAdapter only supports "
                "scenario_id='simple_bgp'."
            )

        entities = [
            NetworkEntity(
                entity_id="nika:simple_bgp:pc1",
                entity_type=EntityType.HOST,
                name="pc1",
                attributes={
                    "provider": "nika",
                    "scenario": "simple_bgp",
                    "role": "host",
                },
                source_ids=[self.source_id],
            ),

            NetworkEntity(
                entity_id="nika:simple_bgp:router1",
                entity_type=EntityType.DEVICE,
                name="router1",
                attributes={
                    "provider": "nika",
                    "scenario": "simple_bgp",
                    "role": "bgp_router",
                },
                source_ids=[self.source_id],
            ),

            NetworkEntity(
                entity_id="nika:simple_bgp:router2",
                entity_type=EntityType.DEVICE,
                name="router2",
                attributes={
                    "provider": "nika",
                    "scenario": "simple_bgp",
                    "role": "bgp_router",
                },
                source_ids=[self.source_id],
            ),

            NetworkEntity(
                entity_id="nika:simple_bgp:pc2",
                entity_type=EntityType.HOST,
                name="pc2",
                attributes={
                    "provider": "nika",
                    "scenario": "simple_bgp",
                    "role": "host",
                },
                source_ids=[self.source_id],
            ),
        ]

        relationships = [
            NetworkRelationship(
                relationship_id="nika:simple_bgp:pc1-router1",
                source_entity_id="nika:simple_bgp:pc1",
                target_entity_id="nika:simple_bgp:router1",
                relationship_type=RelationshipType.CONNECTED_TO,
                attributes={
                    "bidirectional": True,
                },
                source_ids=[self.source_id],
            ),

            NetworkRelationship(
                relationship_id="nika:simple_bgp:router1-router2",
                source_entity_id="nika:simple_bgp:router1",
                target_entity_id="nika:simple_bgp:router2",
                relationship_type=RelationshipType.CONNECTED_TO,
                attributes={
                    "bidirectional": True,
                    "routing_protocol": "bgp",
                },
                source_ids=[self.source_id],
            ),

            NetworkRelationship(
                relationship_id="nika:simple_bgp:router2-pc2",
                source_entity_id="nika:simple_bgp:router2",
                target_entity_id="nika:simple_bgp:pc2",
                relationship_type=RelationshipType.CONNECTED_TO,
                attributes={
                    "bidirectional": True,
                },
                source_ids=[self.source_id],
            ),
        ]

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            entities=entities,
            relationships=relationships,
            metadata={
                "provider": "nika",
                "scenario_id": "simple_bgp",
                "adapter": self.__class__.__name__,
                "integration_stage": "static_topology",
            },
        )

        self.validate_result(result)

        return result


register_adapter(
    "NIKASimpleBGPAdapter",
    NIKASimpleBGPAdapter,
)


class NIKASimpleBGPOperationalAdapter(EvidenceAdapter):
    """
    Normalize FIRST-RESPONSE operational evidence for the NIKA
    simple_bgp link_down scenario into generic ObservationRecord
    objects.

    This represents what would realistically be available in an
    initial telemetry snapshot: interface operational status and
    basic reachability. It deliberately does NOT include BGP
    session state or deep hardware diagnostics — those require a
    separate, explicit diagnostic query (see
    DeepDiagnosticsCapability), mirroring how a real first alert
    carries limited signal compared to what's available on request.

    Version 1 is a deterministic fixture representing the known
    injected incident. A later version will obtain the same evidence
    from NIKA's live MCP/telemetry interfaces.
    """

    def load(self) -> EvidenceAdapterResult:
        scenario_id = self.source_config.metadata.get("scenario_id")
        problem_id = self.source_config.metadata.get("problem_id")

        if scenario_id != "simple_bgp":
            raise ValueError(
                "NIKASimpleBGPOperationalAdapter only supports "
                "scenario_id='simple_bgp'."
            )

        if problem_id != "link_down":
            raise ValueError(
                "This initial operational adapter only supports "
                "problem_id='link_down'."
            )

        observed_at = self.source_config.metadata.get(
            "observed_at",
            "2026-08-07T00:00:01Z",
        )

        observations = [
            # ------------------------------------------------
            # Inter-router interface state (first response only —
            # BGP session state is a tier-2 deep diagnostic)
            # ------------------------------------------------

            ObservationRecord(
                observation_id=(
                    "nika:simple_bgp:obs:"
                    "router1-router2-interface-state"
                ),
                source_id=self.source_id,
                entity_id="nika:simple_bgp:router1",
                metric_name="interface_oper_status",
                metric_value="down",
                observed_at=observed_at,
                dimensions={
                    "peer_entity_id":
                        "nika:simple_bgp:router2",
                },
            ),

            ObservationRecord(
                observation_id=(
                    "nika:simple_bgp:obs:"
                    "router2-router1-interface-state"
                ),
                source_id=self.source_id,
                entity_id="nika:simple_bgp:router2",
                metric_name="interface_oper_status",
                metric_value="down",
                observed_at=observed_at,
                dimensions={
                    "peer_entity_id":
                        "nika:simple_bgp:router1",
                },
            ),

            # ------------------------------------------------
            # Inter-router reachability
            # ------------------------------------------------

            ObservationRecord(
                observation_id=(
                    "nika:simple_bgp:obs:"
                    "router1-router2-ping"
                ),
                source_id=self.source_id,
                entity_id="nika:simple_bgp:router1",
                metric_name="ping_success",
                metric_value=False,
                observed_at=observed_at,
                dimensions={
                    "target_entity_id":
                        "nika:simple_bgp:router2",
                },
            ),

            # ------------------------------------------------
            # Healthy edge reachability
            # ------------------------------------------------

            ObservationRecord(
                observation_id=(
                    "nika:simple_bgp:obs:"
                    "pc1-router1-ping"
                ),
                source_id=self.source_id,
                entity_id="nika:simple_bgp:pc1",
                metric_name="ping_success",
                metric_value=True,
                observed_at=observed_at,
                dimensions={
                    "target_entity_id":
                        "nika:simple_bgp:router1",
                },
            ),

            ObservationRecord(
                observation_id=(
                    "nika:simple_bgp:obs:"
                    "router2-pc2-ping"
                ),
                source_id=self.source_id,
                entity_id="nika:simple_bgp:router2",
                metric_name="ping_success",
                metric_value=True,
                observed_at=observed_at,
                dimensions={
                    "target_entity_id":
                        "nika:simple_bgp:pc2",
                },
            ),
        ]

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            observations=observations,
            metadata={
                "provider": "nika",
                "scenario_id": scenario_id,
                "problem_id": problem_id,
                "adapter": self.__class__.__name__,
                "integration_stage": "operational_evidence_fixture",
                "evidence_tier": "first_response",
            },
        )

        self.validate_result(result)

        return result


register_adapter(
    "NIKASimpleBGPOperationalAdapter",
    NIKASimpleBGPOperationalAdapter,
)
