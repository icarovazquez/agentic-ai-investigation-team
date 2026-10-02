"""
Hypothesis Generator Agent: proposes falsifiable explanations for
the framed incident. Second agent in the chain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from langfuse import observe

from ..capabilities.registry import available_capability_names
from ..capabilities.topology import TopologyCapability
from ..domain import HypothesisStatus
from ..evidence_graph import EvidenceGraph
from ..llm import llm_call, parse_or_repair_agent_response
from .incident_framing import IncidentFrame

HYPOTHESIS_SET_SCHEMA = """
{
    "hypotheses": [
        {
            "title": str,
            "proposed_cause": str,
            "causal_mechanism": str,
            "suspected_entity_ids": list[str],
            "expected_observations": list[str],
            "falsifying_observations": list[str],
            "required_capabilities": list[str],
            "prior_confidence": float
        }
    ]
}
"""


@dataclass
class HypothesisRecord:
    """
    One falsifiable explanation for the reported incident.
    """

    hypothesis_id: str
    incident_id: str

    title: str
    proposed_cause: str
    causal_mechanism: str

    suspected_entity_ids: List[str] = field(default_factory=list)
    expected_observations: List[str] = field(default_factory=list)
    falsifying_observations: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)

    prior_confidence: float = 0.0

    status: HypothesisStatus = HypothesisStatus.PROPOSED

    def validate(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty.")

        if not self.title.strip():
            raise ValueError("Hypothesis title cannot be empty.")

        if not self.proposed_cause.strip():
            raise ValueError("proposed_cause cannot be empty.")

        if not self.causal_mechanism.strip():
            raise ValueError("causal_mechanism cannot be empty.")

        if not self.falsifying_observations:
            raise ValueError(
                "Every hypothesis must define at least "
                "one falsifying observation."
            )

        if not 0.0 <= self.prior_confidence <= 1.0:
            raise ValueError(
                "prior_confidence must be between 0 and 1."
            )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class HypothesisSet:
    """
    Competing hypotheses generated for one incident.
    """

    incident_id: str
    hypotheses: List[HypothesisRecord]

    def validate(self) -> None:
        if len(self.hypotheses) < 2:
            raise ValueError(
                "At least two competing hypotheses are required."
            )

        hypothesis_ids = [h.hypothesis_id for h in self.hypotheses]

        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("Hypothesis IDs must be unique.")

        for hypothesis in self.hypotheses:
            hypothesis.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
        }


def build_hypothesis_generation_context(
    incident_frame: IncidentFrame,
    evidence_graph: EvidenceGraph,
    topology_capability: TopologyCapability,
) -> Dict[str, Any]:
    """
    Build bounded evidence context for hypothesis generation.
    """

    topology_evidence = None

    if len(incident_frame.affected_entity_ids) >= 2:
        topology_evidence = topology_capability.collect_evidence(
            evidence_graph=evidence_graph,
            source_entity_id=incident_frame.affected_entity_ids[0],
            target_entity_id=incident_frame.affected_entity_ids[1],
        )

    return {
        "incident_frame": incident_frame.to_dict(),
        "topology_evidence": topology_evidence,
        # Fix vs. the original notebook: this list was hardcoded and
        # had drifted out of sync (missing vendor_alert and
        # deep_diagnostics). Now derived from the capability
        # registry, same as every other context builder.
        "available_capabilities": available_capability_names(),
    }


@observe(name="hypothesis_generator_agent")
def hypothesis_generator_agent(
    incident_frame: IncidentFrame,
    evidence_graph: EvidenceGraph,
    topology_capability: TopologyCapability,
) -> HypothesisSet:
    """
    Generate competing, falsifiable explanations for the incident.

    This agent proposes hypotheses only.
    It does not decide which hypothesis is correct.
    """

    agent_name = "hypothesis_generator_agent"

    context = build_hypothesis_generation_context(
        incident_frame=incident_frame,
        evidence_graph=evidence_graph,
        topology_capability=topology_capability,
    )

    system_prompt = """
You are the Hypothesis Generator Agent for an autonomous
network investigation team.

Your responsibility is to generate multiple competing,
falsifiable explanations for a framed network incident.

Important rules:

1. Generate between 3 and 5 competing hypotheses.
2. Do NOT select a root cause.
3. Do NOT recommend remediation.
4. Each hypothesis must explain a plausible causal mechanism.
5. Each hypothesis must specify observable evidence that would
   support it.
6. Each hypothesis must specify at least one observation that
   would falsify it.
7. Use only entity IDs present in the supplied context.
8. Request only capabilities listed under available_capabilities.
9. Do not treat benchmark ground truth as investigation evidence.
10. Avoid duplicate hypotheses that describe the same failure
    using different wording.

Return ONLY a Python dictionary with exactly this shape:

{
    "hypotheses": [
        {
            "title": str,
            "proposed_cause": str,
            "causal_mechanism": str,
            "suspected_entity_ids": list[str],
            "expected_observations": list[str],
            "falsifying_observations": list[str],
            "required_capabilities": list[str],
            "prior_confidence": float
        }
    ]
}

prior_confidence must be between 0.0 and 1.0.

The confidence values represent initial plausibility only.
They do not need to sum to 1.0.
"""

    user_prompt = f"""
Investigation context:

{context}

Generate competing hypotheses for this incident.
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    print("Hypothesis generation LLM call completed")

    content = response["content"]

    if not content:
        raise ValueError(
            "Hypothesis Generator Agent returned empty LLM content."
        )

    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=HYPOTHESIS_SET_SCHEMA,
        agent_name=agent_name,
    )

    raw_hypotheses = parsed.get("hypotheses", [])

    hypotheses = []

    for index, item in enumerate(raw_hypotheses, start=1):
        hypothesis = HypothesisRecord(
            hypothesis_id=f"{incident_frame.incident_id}-h{index}",
            incident_id=incident_frame.incident_id,
            title=item["title"],
            proposed_cause=item["proposed_cause"],
            causal_mechanism=item["causal_mechanism"],
            suspected_entity_ids=item.get("suspected_entity_ids", []),
            expected_observations=item.get("expected_observations", []),
            falsifying_observations=item.get("falsifying_observations", []),
            required_capabilities=item.get("required_capabilities", []),
            prior_confidence=float(item.get("prior_confidence", 0.0)),
        )

        hypothesis.validate()
        hypotheses.append(hypothesis)

    hypothesis_set = HypothesisSet(
        incident_id=incident_frame.incident_id,
        hypotheses=hypotheses,
    )

    hypothesis_set.validate()

    return hypothesis_set
