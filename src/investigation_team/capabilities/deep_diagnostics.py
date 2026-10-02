from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import deep_diagnostics_evidence_executor, entity_lookup_executor
from .registry import register_capability


@register_capability(
    name="deep_diagnostics",
    description=(
        "Provides device-level diagnostics NOT included in "
        "routine first-response telemetry: BGP session "
        "state, interface error counters, and administrative "
        "(configured) interface status. Use this capability "
        "specifically when network_state evidence alone "
        "cannot distinguish between competing causes (e.g. "
        "physical failure vs. protocol-driven state changes)."
    ),
)
class DeepDiagnosticsCapability:
    """
    Answer the investigative question:

        What do device-level diagnostics (BGP session state,
        interface error counters, administrative status) show for
        these entities, beyond routine first-response telemetry?

    Unlike NetworkStateCapability, this data is not present in the
    graph until this capability is invoked — it represents an
    explicit escalation to deeper diagnostics, not passive
    telemetry that was always available.
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

        diagnostics = deep_diagnostics_evidence_executor(
            evidence_graph=evidence_graph,
            entity_ids=entity_ids,
        )

        evidence_status = (
            "available" if diagnostics["observation_count"] > 0
            else "insufficient_evidence"
        )

        return {
            "capability": "deep_diagnostics",
            "evidence_status": evidence_status,
            "entities": entity_details,
            "deep_diagnostics_evidence": diagnostics,
        }
