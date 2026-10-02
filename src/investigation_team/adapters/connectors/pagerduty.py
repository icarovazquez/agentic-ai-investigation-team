"""
PagerDuty connector: the first vendor-alert adapter built for this
project, and the template every subsequent vendor connector follows.
"""

from __future__ import annotations

from ...domain import EventRecord
from ..base import EvidenceAdapter, EvidenceAdapterResult, register_adapter


class PagerDutySimpleBGPIncidentAdapter(EvidenceAdapter):
    """
    Normalize a PagerDuty incident (shaped like a real PagerDuty
    incident.trigger webhook payload) for the NIKA simple_bgp
    link_down scenario into a generic EventRecord.

    Version 1 is a deterministic fixture representing what a
    monitoring vendor would have actually sent: a symptom-level
    alert with no root cause, referencing a service name rather
    than an internal entity ID. Entity resolution from the vendor's
    service name to internal entity IDs happens here, at the
    adapter boundary — reasoning agents never see the raw
    PagerDuty payload or vendor-specific naming.

    A later version will consume this from PagerDuty's real
    Events API / webhook delivery.
    """

    SERVICE_NAME_TO_ENTITY_IDS = {
        "core-network-router1-router2-link": [
            "nika:simple_bgp:router1",
            "nika:simple_bgp:router2",
        ],
    }

    def load(self) -> EvidenceAdapterResult:
        scenario_id = self.source_config.metadata.get("scenario_id")
        problem_id = self.source_config.metadata.get("problem_id")

        if scenario_id != "simple_bgp":
            raise ValueError(
                "PagerDutySimpleBGPIncidentAdapter only supports "
                "scenario_id='simple_bgp'."
            )

        if problem_id != "link_down":
            raise ValueError(
                "This initial PagerDuty adapter only supports "
                "problem_id='link_down'."
            )

        raw_incident = {
            "id": "PD1234567",
            "type": "incident",
            "status": "triggered",
            "title": (
                "High packet loss / connection timeout on "
                "core-network-router1-router2-link"
            ),
            "urgency": "high",
            "service": {
                "id": "PSVC001",
                "summary": "core-network-router1-router2-link",
            },
            "created_at": "2026-08-07T00:00:05Z",
            "html_url": "https://nika.pagerduty.com/incidents/PD1234567",
            "assignments": [
                {"assignee": {"summary": "NetOps On-Call"}}
            ],
            "custom_details": {
                "monitor": "Synthetic BGP Peer Reachability Check",
                "alert_source": "nika-synthetic-monitoring",
                "description": (
                    "Synthetic monitor detected 100% packet loss "
                    "between BGP peers on the core interconnect. "
                    "No further diagnosis available — see network "
                    "team for root cause."
                ),
            },
        }

        service_name = raw_incident["service"]["summary"]
        entity_ids = self.SERVICE_NAME_TO_ENTITY_IDS.get(service_name, [])

        if not entity_ids:
            raise ValueError(
                "PagerDutySimpleBGPIncidentAdapter could not resolve "
                f"service '{service_name}' to any known entity IDs."
            )

        event = EventRecord(
            event_id=f"pagerduty:{raw_incident['id']}",
            source_id=self.source_id,
            event_type="vendor_incident",
            occurred_at=raw_incident["created_at"],
            entity_ids=entity_ids,
            severity=raw_incident["urgency"],
            message=raw_incident["title"],
            attributes={
                "vendor": "pagerduty",
                "vendor_incident_id": raw_incident["id"],
                "status": raw_incident["status"],
                "url": raw_incident["html_url"],
                "assigned_to": raw_incident["assignments"][0]["assignee"]["summary"],
                "monitor": raw_incident["custom_details"]["monitor"],
                "vendor_description": raw_incident["custom_details"]["description"],
            },
        )

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            events=[event],
            metadata={
                "provider": "pagerduty",
                "scenario_id": scenario_id,
                "problem_id": problem_id,
                "adapter": self.__class__.__name__,
                "integration_stage": "vendor_alert_fixture",
            },
        )

        self.validate_result(result)

        return result


register_adapter("PagerDutySimpleBGPIncidentAdapter", PagerDutySimpleBGPIncidentAdapter)
