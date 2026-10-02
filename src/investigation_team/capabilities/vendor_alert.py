from __future__ import annotations

from typing import Any, Dict, List

from ..evidence_graph import EvidenceGraph
from ..executors import entity_lookup_executor, vendor_alert_evidence_executor
from .registry import register_capability


@register_capability(
    name="vendor_alert",
    description=(
        "Provides externally reported incidents/alerts from "
        "monitoring or alerting vendors (e.g. PagerDuty). "
        "Describes what a vendor detected and reported, not "
        "the underlying root cause — treat as symptom-level "
        "evidence, same as any other observation."
    ),
)
class VendorAlertCapability:
    """
    Answer the broad investigative question:

        What have external monitoring/alerting vendors already
        reported about these entities?

    This surfaces vendor-sourced incidents (PagerDuty, and later
    Datadog or others) as evidence, but deliberately does not
    interpret or diagnose them — vendor alerts describe symptoms,
    not root cause. Interpretation is the Evidence Analyst's job,
    same as every other capability.
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

        vendor_alerts = (
            vendor_alert_evidence_executor(
                evidence_graph=evidence_graph,
                entity_ids=entity_ids,
            )
        )

        if vendor_alerts["event_count"] == 0:
            evidence_status = "insufficient_evidence"
        else:
            evidence_status = "available"

        return {
            "capability": "vendor_alert",
            "evidence_status": evidence_status,
            "entities": entity_details,
            "vendor_alert_evidence": vendor_alerts,
        }
