from __future__ import annotations

from typing import Any, Dict, Optional

from ..evidence_graph import EvidenceGraph
from ..executors import blast_radius_executor, entity_lookup_executor, path_analysis_executor
from .registry import register_capability


@register_capability(
    name="topology",
    description=(
        "Provides entity relationships, neighbors, paths, "
        "and structural blast-radius evidence."
    ),
)
class TopologyCapability:
    """
    Deterministic topology investigation capability.

    Provides a simplified interface over topology-related executors.
    """

    def collect_evidence(
        self,
        evidence_graph: EvidenceGraph,
        source_entity_id: str,
        target_entity_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        result = {
            "capability": "topology",
        }

        # --------------------------------------------------
        # Source entity
        # --------------------------------------------------

        result["source_entity"] = entity_lookup_executor(
            evidence_graph=evidence_graph,
            entity_id=source_entity_id,
        )

        # --------------------------------------------------
        # Blast radius
        # --------------------------------------------------

        result["blast_radius"] = blast_radius_executor(
            evidence_graph=evidence_graph,
            failed_entity_id=source_entity_id,
        )

        # --------------------------------------------------
        # Optional path analysis
        # --------------------------------------------------

        if target_entity_id is not None:

            result["path_analysis"] = (
                path_analysis_executor(
                    evidence_graph=evidence_graph,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                )
            )

        return result
