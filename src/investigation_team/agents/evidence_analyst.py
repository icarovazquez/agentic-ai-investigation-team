"""
Evidence Analyst Agent: scores each hypothesis against collected
evidence, honestly. Fourth agent in the chain.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from langfuse import observe

from ..domain import EvidencePlan, EvidenceTestResult, HypothesisStatus
from ..llm import llm_call, parse_or_repair_agent_response
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame

EVIDENCE_ANALYSIS_SCHEMA = """
{
    "assessments": [
        {
            "hypothesis_id": str,
            "status": str,
            "confidence": float,
            "supporting_evidence": list[str],
            "contradicting_evidence": list[str],
            "missing_evidence": list[str],
            "rationale": str
        }
    ]
}
"""

EVIDENCE_ANALYST_BANNED_PHRASES = [
    "root cause",
    "confirmed",
    "definitively",
]


@dataclass
class HypothesisAssessment:
    hypothesis_id: str

    status: HypothesisStatus
    confidence: float

    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)

    rationale: str = ""

    def validate(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty.")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")

        if not self.rationale.strip():
            raise ValueError("rationale cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class EvidenceAnalysis:
    incident_id: str
    assessments: List[HypothesisAssessment]

    def validate(self) -> None:
        if not self.assessments:
            raise ValueError(
                "EvidenceAnalysis must contain assessments."
            )

        for assessment in self.assessments:
            assessment.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "assessments": [a.to_dict() for a in self.assessments],
        }


def normalize_hypothesis_id(
    returned_id: str,
    hypothesis_set: HypothesisSet,
) -> str:
    """
    Normalize shorthand hypothesis IDs such as "H1", "h1", "1",
    "H1: BGP Configuration Error", or a bare hypothesis title to
    the canonical hypothesis_id used by the investigation.
    """

    canonical_ids = {
        hypothesis.hypothesis_id
        for hypothesis in hypothesis_set.hypotheses
    }

    if returned_id in canonical_ids:
        return returned_id

    normalized = returned_id.strip().lower()

    # Try a digit-based match first (h1, H1: title, bare "1", etc.)
    match = re.search(r"\d+", normalized)

    if match:
        index = int(match.group(0))
        candidate = f"{hypothesis_set.incident_id}-h{index}"

        if candidate in canonical_ids:
            return candidate

    # Fall back to matching against the hypothesis's own title —
    # covers the model returning a bare title with no numeric
    # reference at all.
    for hypothesis in hypothesis_set.hypotheses:
        if hypothesis.title.strip().lower() == normalized:
            return hypothesis.hypothesis_id

    raise ValueError(
        "Evidence Analyst returned unresolvable hypothesis_id: "
        f"{returned_id}"
    )


def check_evidence_analysis_leakage(
    evidence_analysis: EvidenceAnalysis,
) -> List[str]:
    """
    Deterministic guardrail: flag any assessment text that leaks
    root-cause language the Evidence Analyst is contractually
    forbidden from using.

    Returns a list of violation descriptions. An empty list means
    no violations were found.
    """

    violations = []

    text_fields = (
        "rationale",
        "supporting_evidence",
        "contradicting_evidence",
        "missing_evidence",
    )

    for assessment in evidence_analysis.assessments:
        for field_name in text_fields:
            value = getattr(assessment, field_name)

            if isinstance(value, list):
                haystack = " ".join(value).lower()
            else:
                haystack = str(value).lower()

            for phrase in EVIDENCE_ANALYST_BANNED_PHRASES:
                if phrase in haystack:
                    violations.append(
                        f"{assessment.hypothesis_id}.{field_name} "
                        f"contains banned phrase '{phrase}'"
                    )

    return violations


def build_evidence_analysis_context(
    hypothesis_set: HypothesisSet,
    evidence_plan: EvidencePlan,
    evidence_results: List[EvidenceTestResult],
) -> Dict[str, Any]:
    """
    Build a compact hypothesis-specific evidence package.

    The Evidence Analyst receives only evidence relevant to each
    hypothesis rather than complete executor/capability payloads.
    """

    hypotheses_by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in hypothesis_set.hypotheses
    }

    tests_by_id = {
        test.test_id: test
        for test in evidence_plan.tests
    }

    evidence_by_hypothesis = {}

    for hypothesis_id, hypothesis in hypotheses_by_id.items():
        evidence_by_hypothesis[hypothesis_id] = {
            "hypothesis": hypothesis.to_dict(),
            "tests": [],
        }

    for result in evidence_results:

        if result.hypothesis_id not in evidence_by_hypothesis:
            continue

        test = tests_by_id.get(result.test_id)

        compact_result = {
            "test_id": result.test_id,
            "objective": (test.objective if test is not None else None),
            "capability": result.capability,
            "execution_status": result.status,
            "evidence_status": result.evidence.get("evidence_status"),
            "observations": [],
        }

        # --------------------------------------------
        # Reachability evidence
        # --------------------------------------------

        if result.capability == "reachability":
            reachability = result.evidence.get("reachability_evidence", {})
            compact_result["observations"] = reachability.get("observations", [])

            path = result.evidence.get("path_evidence", {})
            compact_result["structural_path"] = path.get("shortest_path")

        # --------------------------------------------
        # Network-state evidence
        # --------------------------------------------

        elif result.capability == "network_state":
            network_state = result.evidence.get("network_state_evidence", {})
            compact_result["observations"] = network_state.get("observations", [])

        # --------------------------------------------
        # Topology evidence
        # --------------------------------------------

        elif result.capability == "topology":
            compact_result["topology_evidence"] = result.evidence

        evidence_by_hypothesis[result.hypothesis_id]["tests"].append(
            compact_result
        )

    return {"hypotheses": evidence_by_hypothesis}


@observe(name="evidence_analyst_agent")
def evidence_analyst_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_plan: EvidencePlan,
    evidence_results: List[EvidenceTestResult],
) -> EvidenceAnalysis:
    """
    Evaluate each hypothesis against its collected evidence.

    The Evidence Analyst:
      - identifies supporting evidence;
      - identifies contradicting evidence;
      - identifies missing evidence; and
      - assesses hypothesis status.

    It does NOT determine root cause or recommend remediation.
    """

    agent_name = "evidence_analyst_agent"

    context = build_evidence_analysis_context(
        hypothesis_set=hypothesis_set,
        evidence_plan=evidence_plan,
        evidence_results=evidence_results,
    )

    system_prompt = """
You are the Evidence Analyst Agent for an autonomous network
investigation team.

You evaluate existing hypotheses against collected evidence.

You are NOT the Root Cause Agent.

For each hypothesis independently:

1. Review ONLY the tests listed under that hypothesis.
2. Identify observations that support the hypothesis.
3. Identify observations that contradict the hypothesis.
4. Identify important evidence that remains unavailable.
5. Assign exactly one status:
   - supported
   - weakened
   - rejected
   - proposed

Use "proposed" when the evidence is insufficient to evaluate the
hypothesis materially.

Critical rules:

- Never use evidence belonging to another hypothesis.
- Never invent observations.
- A successful test execution is not evidence by itself.
- "insufficient_evidence" means missing evidence.
- A structural topology path does not prove operational reachability.
- Successful reachability may falsify a hypothesis asserting that
  same segment is unreachable.
- confidence represents how likely this hypothesis is to be the
  TRUE ROOT CAUSE of the incident, not how confident you are in
  your verdict. A "rejected" hypothesis must have confidence close
  to 0.0 (the evidence shows it did NOT cause the incident). A
  "supported" hypothesis must have confidence close to 1.0 (the
  evidence shows it likely DID cause the incident). Do NOT invert
  this — confidence is never "how sure am I that this status label
  is correct."
- Do NOT determine root cause.
- Do NOT recommend remediation.
- Do NOT use words such as "confirmed" or "root cause".
- Do NOT generate new hypotheses.
- Do NOT reference other hypotheses by ID (e.g. "H2") or compare
  one hypothesis's plausibility against another's — assess each
  hypothesis strictly against its own evidence in isolation.
- Use only the supplied evidence.
- Confidence must be between 0.0 and 1.0.

Return ONLY a Python dictionary.
No Markdown.
No headings.
No prose before or after the dictionary.
No code fences.

The dictionary must have exactly this shape:

{
    "assessments": [
        {
            "hypothesis_id": str,
            "status": str,
            "confidence": float,
            "supporting_evidence": list[str],
            "contradicting_evidence": list[str],
            "missing_evidence": list[str],
            "rationale": str
        }
    ]
}
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
                Incident frame:

                {incident_frame.to_dict()}

                Evidence analysis context:

                {context}

                Evaluate every hypothesis.
                """,
            },
        ],
        temperature=0.1,
    )

    print("Evidence analysis LLM call completed")

    parsed = parse_or_repair_agent_response(
        raw_output=response["content"],
        expected_schema=EVIDENCE_ANALYSIS_SCHEMA,
        agent_name=agent_name,
    )

    valid_statuses = {"supported", "weakened", "rejected", "proposed"}

    for item in parsed.get("assessments", []):
        item["hypothesis_id"] = normalize_hypothesis_id(
            returned_id=item["hypothesis_id"],
            hypothesis_set=hypothesis_set,
        )

        if item["status"] not in valid_statuses:
            raise ValueError(
                "Evidence Analyst returned invalid status: "
                f"{item['status']}"
            )

    assessments = []

    for item in parsed.get("assessments", []):
        assessment = HypothesisAssessment(
            hypothesis_id=item["hypothesis_id"],
            status=HypothesisStatus(item["status"]),
            confidence=float(item["confidence"]),
            supporting_evidence=item.get("supporting_evidence", []),
            contradicting_evidence=item.get("contradicting_evidence", []),
            missing_evidence=item.get("missing_evidence", []),
            rationale=item.get("rationale", ""),
        )

        assessment.validate()
        assessments.append(assessment)

    analysis = EvidenceAnalysis(
        incident_id=incident_frame.incident_id,
        assessments=assessments,
    )

    analysis.validate()

    return analysis
