from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import entity_lookup_executor, service_desk_evidence_executor
from .registry import register_capability


@register_capability(
    name="service_desk",
    description=(
        "Provides service-desk tickets that name these entities: what "
        "users reported, when it started, and what they believe the "
        "cause is. Reporter theories are unverified claims, never "
        "evidence of cause."
    ),
    observables=("service_ticket",),
)
class ServiceDeskCapability:
    """
    Answer the investigative question:

        What have people reported about these entities?

    Tickets are written by people describing symptoms. Treat the
    symptoms and timing as evidence; treat stated causes as hypotheses
    to test.
    """

    def collect_evidence(
        self,
        evidence_graph: EvidenceGraph,
        entity_ids: List[str],
    ) -> Dict[str, Any]:

        entity_details = [
            entity_lookup_executor(evidence_graph=evidence_graph, entity_id=entity_id)
            for entity_id in entity_ids
        ]

        tickets = service_desk_evidence_executor(
            evidence_graph=evidence_graph,
            entity_ids=entity_ids,
        )

        return {
            "capability": "service_desk",
            "evidence_status": "available" if tickets["event_count"] else "insufficient_evidence",
            "entities": entity_details,
            "service_desk_evidence": tickets,
        }
