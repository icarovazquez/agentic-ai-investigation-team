from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import entity_lookup_executor, network_state_evidence_executor
from .registry import register_capability


@register_capability(
    name="network_state",
    description=(
        "Provides operational state such as interface status, "
        "device availability, routing adjacency state, and "
        "route presence when available."
    ),
)
class NetworkStateCapability:
    """
    Answer the broad investigative question:

        What operational network state is known for these entities?

    This may eventually cover interfaces, routing adjacencies,
    route presence, and device availability.
    """

    def collect_evidence(
        self,
        evidence_graph: EvidenceGraph,
        entity_ids: List[str],
    ) -> Dict[str, Any]:

        entity_details = []

        for entity_id in entity_ids:
            entity_details.append(
                entity_lookup_executor(
                    evidence_graph=evidence_graph,
                    entity_id=entity_id,
                )
            )

        network_state = (
            network_state_evidence_executor(
                evidence_graph=evidence_graph,
                entity_ids=entity_ids,
            )
        )

        if network_state["observation_count"] == 0:
            evidence_status = "insufficient_evidence"
        else:
            evidence_status = "available"

        return {
            "capability": "network_state",
            "evidence_status": evidence_status,
            "entities": entity_details,
            "network_state_evidence": network_state,
        }
