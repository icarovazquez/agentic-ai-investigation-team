from __future__ import annotations

from typing import Any, Dict

from ..evidence_graph import EvidenceGraph
from ..executors import entity_lookup_executor, path_analysis_executor, reachability_evidence_executor
from .registry import register_capability


@register_capability(
    name="reachability",
    description=(
        "Provides end-to-end or hop-level reachability "
        "evidence such as ping, loss, latency, or path tests."
    ),
)
class ReachabilityCapability:
    """
    Answer the broad investigative question:

        Can entity A reach entity B, and what reachability
        evidence do we currently possess?

    The capability orchestrates deterministic executors only.
    """

    def collect_evidence(
        self,
        evidence_graph: EvidenceGraph,
        source_entity_id: str,
        target_entity_id: str,
    ) -> Dict[str, Any]:

        # Verify that both entities exist and capture context.
        source_entity = entity_lookup_executor(
            evidence_graph=evidence_graph,
            entity_id=source_entity_id,
        )

        target_entity = entity_lookup_executor(
            evidence_graph=evidence_graph,
            entity_id=target_entity_id,
        )

        # Structural path evidence.
        try:
            path_evidence = path_analysis_executor(
                evidence_graph=evidence_graph,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                include_all_paths=True,
            )

        except ValueError:
            path_evidence = {
                "path_exists": False,
                "shortest_path": None,
            }

        # Operational reachability evidence.
        reachability_evidence = (
            reachability_evidence_executor(
                evidence_graph=evidence_graph,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
            )
        )

        if (
            reachability_evidence["observation_count"]
            == 0
        ):
            evidence_status = "insufficient_evidence"
        else:
            evidence_status = "available"

        return {
            "capability": "reachability",
            "evidence_status": evidence_status,
            "source_entity": source_entity,
            "target_entity": target_entity,
            "path_evidence": path_evidence,
            "reachability_evidence": (
                reachability_evidence
            ),
        }
