from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import call_quality_evidence_executor, entity_lookup_executor
from .registry import register_capability


@register_capability(
    name="call_quality",
    description=(
        "Provides measured voice/video call quality for endpoints: "
        "jitter, packet loss and round-trip time per call, with call "
        "times. Shows WHEN and HOW BADLY calls degraded and for WHICH "
        "endpoints; it does not say why."
    ),
    observables=("call_jitter_ms", "call_packet_loss_pct", "call_rtt_ms"),
)
class CallQualityCapability:
    """
    Answer the investigative question:

        How did call quality look for these endpoints, and when?

    Measurements are symptoms at the endpoint. Whether they come from
    the network, the endpoint itself or the service is for the Evidence
    Analyst to weigh against other evidence.
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

        call_quality = call_quality_evidence_executor(
            evidence_graph=evidence_graph,
            entity_ids=entity_ids,
        )

        evidence_status = (
            "available" if call_quality["observation_count"] > 0
            else "insufficient_evidence"
        )

        return {
            "capability": "call_quality",
            "evidence_status": evidence_status,
            "entities": entity_details,
            "call_quality_evidence": call_quality,
        }
