from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import entity_lookup_executor, traffic_evidence_executor
from .registry import register_capability


@register_capability(
    name="traffic",
    description=(
        "Provides link and flow state NOT in the first-response "
        "snapshot: link utilization, top talkers, how traffic is "
        "marked (DSCP), per-class queue drops, and whether a QoS "
        "service-policy is attached to an interface. Use it when "
        "endpoints degrade while links and reachability look healthy."
    ),
    observables=(
        "link_utilization_pct",
        "top_talker_bps",
        "dscp_marking",
        "queue_drops",
        "service_policy_attached",
    ),
)
class TrafficCapability:
    """
    Answer the investigative question:

        How loaded is the path, who is using it, and is traffic
        protected by the intended QoS policy?

    Explicit escalation, like deep diagnostics: this data is fetched on
    request rather than preloaded.
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

        traffic = traffic_evidence_executor(
            evidence_graph=evidence_graph,
            entity_ids=entity_ids,
        )

        return {
            "capability": "traffic",
            "evidence_status": "available" if traffic["observation_count"] else "insufficient_evidence",
            "entities": entity_details,
            "traffic_evidence": traffic,
        }
