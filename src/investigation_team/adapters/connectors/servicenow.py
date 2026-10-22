"""
ServiceNow connector: incident and change tickets.

Tickets are the least structured source in the investigation: free text
written by a person. Getting structure out of them is a job for a model,
and this adapter is where that happens, at the boundary. Reasoning agents
downstream never see raw ticket text.

The model is not trusted. Everything it returns is checked in code:

- an endpoint it extracts must literally appear in the ticket text
  (otherwise it was invented) and must resolve through the shared
  EntityResolver (otherwise it is reported and skipped);
- what the reporter *believes* caused the problem is stored as
  unverified_reporter_theories and never used to build the event
  message or attach evidence;
- malformed output fails loudly instead of producing a half-filled record.

The extraction call is injectable (adapter.extractor) so tests run with a
stub and no API calls.

Version 1 reads fixture records shaped like ServiceNow Table API rows.
A later version will read them from the live Table API.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from langfuse import observe

from ...domain import ChangeRecord, EventRecord
from ...llm import llm_call, parse_or_repair_agent_response
from ..base import EvidenceAdapter, EvidenceAdapterResult, register_adapter

P = "nika:qos_contention"

TICKET_EXTRACTION_SCHEMA = """
{
    "reporter": str | None,
    "affected_endpoints": list[str],
    "symptoms": list[str],
    "reporter_theories": list[str],
    "onset": str | None,
    "change_type": str | None,
    "actor": str | None,
    "change_summary": str | None,
    "change_details": list[str]
}
"""

RAW_TICKETS: List[Dict[str, Any]] = [
    {
        "number": "INC0045821",
        "kind": "incident",
        "opened_at": "2026-09-14T09:42:00Z",
        "opened_by": "Maria Lopez",
        "short_description": "Teams calls sounding robotic and dropping",
        "description": (
            "Since about 9 this morning my Teams calls keep breaking up. People say I sound "
            "like a robot and the video freezes. I'm pretty sure my headset is dying (it's the "
            "one from 2021), though IT swapped my laptop dock last month so maybe that. The VPN "
            "team says it's probably the VPN client. I'm at my desk, WS-0142, wired. Ravi at the "
            "next desk says it's happening to him too but he's blaming his webcam. Please fix "
            "before my 11am customer call."
        ),
    },
    {
        "number": "CHG0031207",
        "kind": "change",
        "opened_at": "2026-09-13T22:00:00Z",
        "opened_by": "J. Alvarez",
        "short_description": "Edge router RTR-EDGE-01 software upgrade",
        "description": (
            "Planned maintenance window 20:00-22:00. Upgraded RTR-EDGE-01 from 17.6.4 to 17.9.3 "
            "per vendor security advisory. Interface configurations were reapplied from the "
            "standard branch template after reload. Post-change checks: all interfaces up, "
            "neighbors established, ping to WAN gateway OK. No issues observed. Backout plan: "
            "reload previous image."
        ),
    },
]

_EXTRACTION_PROMPT = """
You extract structured facts from one IT service-desk ticket.

Rules:
- Use only what the ticket text says. Do not infer, and do not diagnose.
- affected_endpoints: device or desk identifiers exactly as written in the
  ticket (hostnames, asset tags, IPs, "<name>'s desk"). Copy the text
  verbatim; do not normalize it.
- symptoms: what is observed to be going wrong, as stated.
- reporter_theories: what the author BELIEVES or was told the cause is.
  These are unverified opinions; list them separately from symptoms.
- onset: when it started, as written (null if not stated).
- For change tickets: change_type (e.g. "software_upgrade"), actor,
  change_summary (one sentence), change_details (specific actions the
  text says were performed). For incident tickets set these to null/[].
- Use null / [] for anything not present. Never invent values.

Return ONLY a JSON object with exactly this shape:
""" + TICKET_EXTRACTION_SCHEMA


@observe(name="ticket_extraction_agent")
def llm_extract_ticket(raw_ticket: Dict[str, Any]) -> Dict[str, Any]:
    agent_name = "ticket_extraction_agent"
    text = f"Ticket {raw_ticket['number']} ({raw_ticket['kind']})\n" \
           f"Opened by: {raw_ticket['opened_by']}\n" \
           f"Short description: {raw_ticket['short_description']}\n\n{raw_ticket['description']}"

    response = llm_call(
        agent_name=agent_name,
        messages=[{"role": "system", "content": _EXTRACTION_PROMPT},
                  {"role": "user", "content": text}],
        temperature=0.0,
    )
    if not response["content"]:
        raise ValueError("Ticket extraction returned empty LLM content.")

    return parse_or_repair_agent_response(
        raw_output=response["content"],
        expected_schema=TICKET_EXTRACTION_SCHEMA,
        agent_name=agent_name,
    )


def _as_str_list(value: Any, field_name: str, ticket: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"Ticket {ticket}: extraction field '{field_name}' must be a list of strings.")
    return [v.strip() for v in value if v.strip()]


def _appears_in(text: str, snippet: str) -> bool:
    norm = lambda t: re.sub(r"\s+", " ", t.replace("’", "'")).strip().lower()
    return norm(snippet) in norm(text)


class ServiceNowTicketAdapter(EvidenceAdapter):
    def __init__(self, source_config) -> None:
        super().__init__(source_config)
        self.extractor: Callable[[Dict[str, Any]], Dict[str, Any]] = llm_extract_ticket

    def load(self) -> EvidenceAdapterResult:
        scenario_id = self.source_config.metadata.get("scenario_id")
        if scenario_id != "qos_contention":
            raise ValueError("ServiceNowTicketAdapter only supports scenario_id='qos_contention'.")

        result = EvidenceAdapterResult(
            source_id=self.source_id,
            metadata={"provider": "servicenow", "scenario_id": scenario_id,
                      "adapter": self.__class__.__name__, "integration_stage": "ticket_fixture",
                      "tickets_read": len(RAW_TICKETS), "extraction": "llm"},
        )

        for raw in RAW_TICKETS:
            number = raw["number"]
            full_text = f"{raw['short_description']}\n{raw['description']}"
            extracted = self.extractor(raw)

            if not isinstance(extracted, dict):
                raise ValueError(f"Ticket {number}: extraction did not return an object.")

            claimed = _as_str_list(extracted.get("affected_endpoints"), "affected_endpoints", number)
            symptoms = _as_str_list(extracted.get("symptoms"), "symptoms", number)
            theories = _as_str_list(extracted.get("reporter_theories"), "reporter_theories", number)
            details = _as_str_list(extracted.get("change_details"), "change_details", number)

            entity_ids: List[str] = []
            for endpoint in claimed:
                if not _appears_in(full_text, endpoint):
                    result.warnings.append(
                        f"{self.source_id} (ticket {number}): extracted endpoint '{endpoint}' does not "
                        "appear in the ticket text; discarded."
                    )
                    continue
                entity_id = self.resolve_entity(endpoint, result, context=f"ticket {number}")
                if entity_id and entity_id not in entity_ids:
                    entity_ids.append(entity_id)

            if not entity_ids:
                result.warnings.append(
                    f"{self.source_id} (ticket {number}): no endpoint resolved to a known entity; "
                    "ticket skipped."
                )
                continue

            if raw["kind"] == "incident":
                result.events.append(EventRecord(
                    event_id=f"servicenow:{number}",
                    source_id=self.source_id,
                    event_type="service_ticket",
                    occurred_at=raw["opened_at"],
                    entity_ids=entity_ids,
                    severity=None,
                    message=raw["short_description"],
                    attributes={
                        "ticket_id": number,
                        "ticket_kind": "incident",
                        "reporter": extracted.get("reporter") or raw["opened_by"],
                        "symptoms": symptoms,
                        "onset_as_written": extracted.get("onset"),
                        "unverified_reporter_theories": theories,
                    },
                ))
            else:
                change_type = (extracted.get("change_type") or "").strip()
                if not change_type:
                    raise ValueError(f"Ticket {number}: change ticket extraction has no change_type.")
                result.changes.append(ChangeRecord(
                    change_id=f"servicenow:{number}",
                    source_id=self.source_id,
                    changed_at=raw["opened_at"],
                    change_type=change_type,
                    entity_ids=entity_ids,
                    actor=extracted.get("actor") or raw["opened_by"],
                    attributes={
                        "ticket_id": number,
                        "summary": extracted.get("change_summary") or raw["short_description"],
                        "details_as_recorded": details,
                    },
                ))

        self.validate_result(result)
        return result


register_adapter("ServiceNowTicketAdapter", ServiceNowTicketAdapter)
