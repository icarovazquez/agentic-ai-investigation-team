"""
Causal Resolution: when more than one hypothesis is supported, are they
rivals or a causal chain?

The rest of the pipeline treats hypotheses as competing explanations of a
single cause. Real incidents are often a trigger plus an underlying
condition: a bulk transfer saturates a link (trigger) and voice suffers
because the QoS policy that should protect it is missing (underlying
cause). Both hypotheses can be supported, and ranking them by a confidence
number picks a "root cause" on noise.

This step asks the one question that separates them, and the answer is
checked in code:

    The underlying cause is the deviation from the intended state of the
    system. The trigger is normal load or behaviour the system was meant
    to tolerate.

If the hypotheses are rivals, the deterministic leader stands and nothing
changes. If they form a chain, the model names which hypothesis is the
underlying cause and which are contributing; every id must be one of the
supported candidates, and a chain with no contributing member, no
rationale, or an unknown id falls back to "rivals".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langfuse import observe

from ..domain import HypothesisStatus
from ..llm import llm_call, parse_or_repair_agent_response
from .evidence_analyst import EvidenceAnalysis, HypothesisAssessment, normalize_hypothesis_id
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame
from .root_cause_remediation import directional_confidence

CAUSAL_RESOLUTION_SCHEMA = """
{
    "relationship": str,
    "root_cause_hypothesis_id": str | None,
    "contributing_hypothesis_ids": list[str],
    "deviation_from_intended_state": str | None,
    "rationale": str
}
"""


@dataclass
class CausalResolution:
    relationship: str  # "rivals" | "causal_chain"
    root_cause_hypothesis_id: str
    contributing_hypothesis_ids: List[str] = field(default_factory=list)
    deviation_from_intended_state: Optional[str] = None
    rationale: str = ""
    notes: List[str] = field(default_factory=list)  # why a fallback was used, etc.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chain_candidates(
    evidence_analysis: EvidenceAnalysis,
    minimum_confidence: float,
) -> List[HypothesisAssessment]:
    """Supported hypotheses strong enough to be part of the explanation."""
    return sorted(
        (
            a for a in evidence_analysis.assessments
            if a.status == HypothesisStatus.SUPPORTED
            and directional_confidence(a) >= minimum_confidence
        ),
        key=directional_confidence,
        reverse=True,
    )


def validate_causal_resolution(
    parsed: Dict[str, Any],
    candidate_ids: List[str],
    default_root_id: str,
    hypothesis_set: HypothesisSet,
) -> CausalResolution:
    """
    Turn the model's answer into a resolution that is safe to act on.
    Anything that does not check out falls back to "rivals", where the
    deterministic leader (default_root_id) stands.
    """

    def fallback(reason: str) -> CausalResolution:
        return CausalResolution(
            relationship="rivals",
            root_cause_hypothesis_id=default_root_id,
            rationale=str(parsed.get("rationale") or "").strip(),
            notes=[reason],
        )

    if not isinstance(parsed, dict):
        return fallback("Causal resolution returned no usable answer; treated as rivals.")

    relationship = str(parsed.get("relationship") or "").strip().lower()
    if relationship == "rivals":
        return CausalResolution(
            relationship="rivals",
            root_cause_hypothesis_id=default_root_id,
            rationale=str(parsed.get("rationale") or "").strip(),
        )
    if relationship != "causal_chain":
        return fallback(f"Unknown relationship '{relationship}'; treated as rivals.")

    def canonical(raw: Any) -> Optional[str]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            found = normalize_hypothesis_id(raw, hypothesis_set)
        except Exception:
            return None
        return found if found in candidate_ids else None

    root = canonical(parsed.get("root_cause_hypothesis_id"))
    if root is None:
        return fallback("Chain named a root cause that is not a supported candidate; treated as rivals.")

    raw_contributing = parsed.get("contributing_hypothesis_ids") or []
    contributing = []
    for raw in raw_contributing if isinstance(raw_contributing, list) else []:
        found = canonical(raw)
        if found and found != root and found not in contributing:
            contributing.append(found)
    if not contributing:
        return fallback("Chain named no contributing hypothesis among the candidates; treated as rivals.")

    rationale = str(parsed.get("rationale") or "").strip()
    if not rationale:
        return fallback("Chain gave no rationale; treated as rivals.")

    return CausalResolution(
        relationship="causal_chain",
        root_cause_hypothesis_id=root,
        contributing_hypothesis_ids=contributing,
        deviation_from_intended_state=(parsed.get("deviation_from_intended_state") or None),
        rationale=rationale,
    )


@observe(name="causal_resolution_agent")
def causal_resolution_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_analysis: EvidenceAnalysis,
    candidates: List[HypothesisAssessment],
    default_root_id: str,
) -> CausalResolution:
    agent_name = "causal_resolution_agent"
    hypotheses_by_id = {h.hypothesis_id: h for h in hypothesis_set.hypotheses}

    context = {
        "incident_frame": incident_frame.to_dict(),
        "supported_hypotheses": [
            {
                "hypothesis": hypotheses_by_id[a.hypothesis_id].to_dict()
                if a.hypothesis_id in hypotheses_by_id else None,
                "assessment": a.to_dict(),
            }
            for a in candidates
        ],
    }

    system_prompt = """
You are the Causal Resolution Agent for an autonomous network
investigation team.

Several hypotheses are supported by the evidence. Decide how they
relate. You do NOT collect evidence and you do NOT recommend
remediation.

relationship must be exactly one of:

- "rivals": they are alternative explanations of the same single cause
  and the evidence cannot yet tell them apart, or only one can be true.
- "causal_chain": they are all true together, as a trigger plus an
  underlying condition or contributing factors (for example: heavy
  traffic saturated a link, and voice suffered because the protection
  that should have prevented it was missing).

If "causal_chain", identify the UNDERLYING CAUSE with this test:

  The underlying cause is the deviation from the intended state of the
  system: something that was supposed to be different (a policy that
  should be attached, a setting that should be set, a component that
  should be healthy). A trigger is normal load or behaviour the system
  was designed to tolerate. Ask: if this deviation had not existed,
  would the same trigger still have caused the incident? If not, the
  deviation is the underlying cause.

Rules:

- Use only the supplied hypotheses and assessments. Do not invent facts.
- root_cause_hypothesis_id and every contributing_hypothesis_ids entry
  must be hypothesis_id values from supported_hypotheses.
- For "rivals", set root_cause_hypothesis_id to null.
- deviation_from_intended_state: one sentence naming the deviation
  (null for "rivals").
- rationale: two or three sentences grounded in the cited evidence.

Return ONLY a Python dictionary with exactly this shape:

{
    "relationship": str,
    "root_cause_hypothesis_id": str | None,
    "contributing_hypothesis_ids": list[str],
    "deviation_from_intended_state": str | None,
    "rationale": str
}
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Investigation context:\n\n{context}\n\nDecide how these hypotheses relate."},
        ],
        temperature=0.0,
    )

    print("Causal resolution LLM call completed")

    if not response["content"]:
        raise ValueError("Causal Resolution Agent returned empty LLM content.")

    parsed = parse_or_repair_agent_response(
        raw_output=response["content"],
        expected_schema=CAUSAL_RESOLUTION_SCHEMA,
        agent_name=agent_name,
    )

    return validate_causal_resolution(
        parsed=parsed,
        candidate_ids=[a.hypothesis_id for a in candidates],
        default_root_id=default_root_id,
        hypothesis_set=hypothesis_set,
    )
