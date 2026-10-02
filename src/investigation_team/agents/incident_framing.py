"""
Incident Framing Agent: turns a vague symptom into a scoped
investigation. First agent in the chain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langfuse import observe

from ..capabilities.registry import available_capability_names, capability_descriptions
from ..capabilities.topology import TopologyCapability
from ..config import NetworkInvestigationConfig
from ..evidence_graph import EvidenceGraph
from ..llm import llm_call, parse_or_repair_agent_response

INCIDENT_FRAME_SCHEMA = """
{
    "investigation_question": str,
    "scope_summary": str,
    "known_facts": list[str],
    "assumptions": list[str],
    "unknowns": list[str],
    "affected_entity_ids": list[str],
    "potentially_relevant_entity_ids": list[str],
    "required_capabilities": list[str],
    "severity": str | None
}
"""


@dataclass
class IncidentFrame:
    """
    Structured output produced by the Incident Framing Agent.

    It converts the raw incident seed into a clear investigation
    problem without attempting to diagnose root cause.
    """

    incident_id: str

    investigation_question: str
    scope_summary: str

    known_facts: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)

    affected_entity_ids: List[str] = field(default_factory=list)
    potentially_relevant_entity_ids: List[str] = field(default_factory=list)

    required_capabilities: List[str] = field(default_factory=list)

    severity: Optional[str] = None

    def validate(self) -> None:
        if not self.incident_id.strip():
            raise ValueError("incident_id cannot be empty.")

        if not self.investigation_question.strip():
            raise ValueError(
                "investigation_question cannot be empty."
            )

        if not self.scope_summary.strip():
            raise ValueError(
                "scope_summary cannot be empty."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_incident_framing_context(
    investigation_config: NetworkInvestigationConfig,
    evidence_graph: EvidenceGraph,
    topology_capability: Optional[TopologyCapability] = None,
) -> Dict[str, Any]:
    """
    Build the bounded context supplied to the Incident Framing Agent.

    The agent receives normalized evidence and capability output,
    never raw source data.
    """

    seed = investigation_config.incident_seed

    reported_entities = []

    for entity_id in seed.reported_entities:
        try:
            reported_entities.append(
                evidence_graph.get_entity(entity_id).to_dict()
            )

        except KeyError:
            reported_entities.append(
                {
                    "entity_id": entity_id,
                    "status": "not_found_in_evidence_graph",
                }
            )

    topology_evidence = None

    if (
        topology_capability is not None
        and len(seed.reported_entities) >= 2
    ):
        topology_evidence = topology_capability.collect_evidence(
            evidence_graph=evidence_graph,
            source_entity_id=seed.reported_entities[0],
            target_entity_id=seed.reported_entities[1],
        )

    # Vendor-reported incidents already known about this investigation.
    # Surfaced explicitly so the framing agent treats "what has a
    # monitoring vendor already told us" as a known fact from the
    # start, rather than something only reachable if a later
    # hypothesis happens to request vendor_alert evidence.
    vendor_alerts = [
        event.to_dict()
        for event in evidence_graph.get_events(
            event_types=["vendor_incident"]
        )
    ]

    return {
        "investigation_id": (
            investigation_config.investigation_id
        ),
        "investigation_name": (
            investigation_config.investigation_name
        ),
        "investigation_type": (
            investigation_config.investigation_type
        ),

        "reported_symptom": seed.reported_symptom,
        "incident_start_time": seed.incident_start_time,
        "affected_service": seed.affected_service,
        "reported_locations": list(
            seed.reported_locations
        ),
        "reported_entities": reported_entities,
        "initial_severity": seed.initial_severity,

        "available_evidence_types": (
            investigation_config.available_evidence_types
        ),

        "evidence_graph_summary": (
            evidence_graph.summary()
        ),

        "topology_evidence": topology_evidence,

        "vendor_alerts": vendor_alerts,

        # Fix vs. the original notebook: this list was hardcoded and
        # had drifted out of sync with the actual set of registered
        # capabilities (missing vendor_alert and deep_diagnostics).
        # Now derived from the same registry every other context
        # builder and the dispatch layer read from, so it can never
        # drift again.
        "available_capabilities": available_capability_names(),
        "capability_descriptions": capability_descriptions(),
    }


@observe(name="incident_framing_agent")
def incident_framing_agent(
    investigation_config: NetworkInvestigationConfig,
    evidence_graph: EvidenceGraph,
    topology_capability: TopologyCapability,
) -> IncidentFrame:
    """
    Frame the incident into a structured investigation problem.

    This agent must not diagnose root cause.
    """

    agent_name = "incident_framing_agent"

    context = build_incident_framing_context(
        investigation_config=investigation_config,
        evidence_graph=evidence_graph,
        topology_capability=topology_capability,
    )

    system_prompt = """
You are the Incident Framing Agent for an autonomous network
investigation team.

Your job is to transform the reported network symptom into a precise,
bounded investigation problem.

You must distinguish:

- reported facts;
- assumptions;
- unknowns;
- affected entities;
- potentially relevant entities; and
- investigative capabilities that may be needed.

Do NOT diagnose root cause.
Do NOT recommend remediation.
Do NOT generate detailed hypotheses.

Return only a Python dictionary with exactly these keys:

{
    "investigation_question": str,
    "scope_summary": str,
    "known_facts": list[str],
    "assumptions": list[str],
    "unknowns": list[str],
    "affected_entity_ids": list[str],
    "potentially_relevant_entity_ids": list[str],
    "required_capabilities": list[str],
    "severity": str | None
}

Use only entity IDs that appear in the supplied context.
Only request capabilities listed under available_capabilities.
"""

    user_prompt = f"""
Investigation context:

{context}

Frame this network incident.
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = response["content"]

    if not content:
        raise ValueError(
            "Incident Framing Agent returned empty LLM content."
        )

    # Fix vs. the original notebook: this agent used to call
    # ast.literal_eval directly, bypassing the repair mechanism every
    # other agent uses. It was the one agent with zero resilience to
    # malformed structured output.
    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=INCIDENT_FRAME_SCHEMA,
        agent_name=agent_name,
    )

    incident_frame = IncidentFrame(
        incident_id=(f"{investigation_config.investigation_id}-incident-frame"),
        investigation_question=parsed["investigation_question"],
        scope_summary=parsed["scope_summary"],
        known_facts=parsed.get("known_facts", []),
        assumptions=parsed.get("assumptions", []),
        unknowns=parsed.get("unknowns", []),
        affected_entity_ids=parsed.get("affected_entity_ids", []),
        potentially_relevant_entity_ids=parsed.get(
            "potentially_relevant_entity_ids", []
        ),
        required_capabilities=parsed.get("required_capabilities", []),
        severity=parsed.get("severity"),
    )

    incident_frame.validate()

    return incident_frame
