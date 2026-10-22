from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import change_history_evidence_executor, entity_lookup_executor
from .registry import register_capability


@register_capability(
    name="change_history",
    description=(
        "Provides recorded changes to these entities (software "
        "upgrades, configuration pushes, hardware swaps): when, by "
        "whom, and what the change record says was done. A change "
        "record shows what someone intended and reported, not "
        "necessarily the resulting device state."
    ),
    observables=("change_record",),
)
class ChangeHistoryCapability:
    """
    Answer the investigative question:

        What changed on these entities, and when?
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

        changes = change_history_evidence_executor(
            evidence_graph=evidence_graph,
            entity_ids=entity_ids,
        )

        return {
            "capability": "change_history",
            "evidence_status": "available" if changes["change_count"] else "insufficient_evidence",
            "entities": entity_details,
            "change_history_evidence": changes,
        }
