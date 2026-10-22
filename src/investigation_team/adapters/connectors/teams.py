"""
Microsoft Teams call-quality connector.

Call-quality platforms identify the endpoint in whatever form they have:
a UPN on one record, a fully qualified device name on another, a MAC on
a third. This adapter normalizes the measurements into generic
observations on the right entity by resolving through the shared
EntityResolver. A record whose endpoint cannot be resolved is skipped
and reported as a warning, never attached to a guess.

Version 1 is a deterministic fixture shaped like call-quality records.
A later version will read them from the vendor's reporting API.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ...domain import ObservationRecord
from ..base import EvidenceAdapter, EvidenceAdapterResult, register_adapter

P = "nika:qos_contention"

# The measurements this adapter emits, with units.
CALL_METRICS = {
    "avgJitterMs": ("call_jitter_ms", "ms"),
    "avgPacketLossPct": ("call_packet_loss_pct", "percent"),
    "avgRoundTripMs": ("call_rtt_ms", "ms"),
}

RAW_CALL_RECORDS: List[Dict[str, Any]] = [
    {"callId": "c-0801", "startTime": "2026-09-14T08:10:00Z", "endTime": "2026-09-14T08:25:00Z",
     "deviceName": "MARIA-LAPTOP.corp.example.com", "upn": "maria.lopez@corp.example.com",
     "networkType": "wired", "avgJitterMs": 5, "avgPacketLossPct": 0.1, "avgRoundTripMs": 38},
    {"callId": "c-0902", "startTime": "2026-09-14T09:05:00Z", "endTime": "2026-09-14T09:30:00Z",
     "deviceName": None, "upn": "Maria.Lopez@corp.example.com",
     "networkType": "wired", "avgJitterMs": 64, "avgPacketLossPct": 8.6, "avgRoundTripMs": 212},
    {"callId": "c-0903", "startTime": "2026-09-14T09:12:00Z", "endTime": "2026-09-14T09:40:00Z",
     "deviceName": "AA-BB-CC-00-01-43", "upn": None,
     "networkType": "wired", "avgJitterMs": 71, "avgPacketLossPct": 11.2, "avgRoundTripMs": 240},
    {"callId": "c-0904", "startTime": "2026-09-14T09:35:00Z", "endTime": "2026-09-14T09:50:00Z",
     "deviceName": "WS-0142", "upn": None,
     "networkType": "wired", "avgJitterMs": 58, "avgPacketLossPct": 7.9, "avgRoundTripMs": 198},
    {"callId": "c-0905", "startTime": "2026-09-14T09:20:00Z", "endTime": "2026-09-14T09:33:00Z",
     "deviceName": "UNKNOWN-DEVICE-77.corp.example.com", "upn": None,
     "networkType": "wired", "avgJitterMs": 66, "avgPacketLossPct": 9.0, "avgRoundTripMs": 220},
]


class TeamsCallQualityAdapter(EvidenceAdapter):
    def load(self) -> EvidenceAdapterResult:
        scenario_id = self.source_config.metadata.get("scenario_id")
        if scenario_id != "qos_contention":
            raise ValueError("TeamsCallQualityAdapter only supports scenario_id='qos_contention'.")

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            metadata={"provider": "teams", "scenario_id": scenario_id,
                      "adapter": self.__class__.__name__,
                      "integration_stage": "call_quality_fixture",
                      "records_read": len(RAW_CALL_RECORDS)},
        )

        for record in RAW_CALL_RECORDS:
            entity_id = None
            for field_name in ("deviceName", "upn"):
                raw = record.get(field_name)
                if not raw:
                    continue
                # Try quietly first; only warn if every identifier fails.
                resolution = self.entity_resolver.resolve(raw) if self.entity_resolver else None
                if resolution is not None and resolution.resolved:
                    entity_id = resolution.entity_id
                    break

            if entity_id is None:
                tried = [record.get("deviceName"), record.get("upn")]
                self.resolve_entity(next((t for t in tried if t), record["callId"]), result,
                                    context=f"call {record['callId']}")
                continue

            for raw_field, (metric, unit) in CALL_METRICS.items():
                result.observations.append(ObservationRecord(
                    observation_id=f"teams:call:{record['callId']}:{metric}",
                    source_id=self.source_id,
                    entity_id=entity_id,
                    metric_name=metric,
                    metric_value=record[raw_field],
                    observed_at=record["endTime"],
                    unit=unit,
                    dimensions={
                        "call_id": record["callId"],
                        "started_at": record["startTime"],
                        "network_type": record["networkType"],
                        "peer_entity_id": f"{P}:teams-relay",
                    },
                ))

        self.validate_result(result)
        return result


register_adapter("TeamsCallQualityAdapter", TeamsCallQualityAdapter)
