"""
Root Cause & Remediation Agent: commits to a root cause (or reports
insufficient evidence) and recommends remediation. Final agent in
the chain — only runs once the Challenger has had its say.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langfuse import observe

from ..config import InvestigationPolicy
from ..domain import HypothesisStatus
from ..llm import llm_call, parse_or_repair_agent_response
from .evidence_analyst import EvidenceAnalysis, HypothesisAssessment
from .hypothesis_challenger import ChallengeReport
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame

ROOT_CAUSE_RECOMMENDATION_SCHEMA = """
{
    "root_cause_explanation": str,
    "remediation_steps": [
        {
            "action": str,
            "rationale": str,
            "risk_level": str
        }
    ],
    "residual_risk": str,
    "challenge_acknowledged": str
}
"""


@dataclass
class RemediationStep:
    """One concrete, ordered remediation action."""

    step_number: int
    action: str
    rationale: str
    risk_level: str  # "low" | "medium" | "high"

    def validate(self) -> None:
        if self.step_number < 1:
            raise ValueError("step_number must be >= 1.")
        if not self.action.strip():
            raise ValueError("action cannot be empty.")
        if self.risk_level not in {"low", "medium", "high"}:
            raise ValueError(
                f"risk_level must be low/medium/high, got '{self.risk_level}'."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RootCauseRecommendation:
    """
    Final output of an investigation: either a committed root cause
    with remediation, or an explicit statement that evidence was
    insufficient to commit to one.
    """

    incident_id: str
    status: str  # "root_cause_identified" | "insufficient_evidence"

    hypothesis_id: Optional[str] = None
    root_cause_explanation: Optional[str] = None
    confidence: Optional[float] = None

    remediation_steps: List[RemediationStep] = field(default_factory=list)
    residual_risk: Optional[str] = None

    # how the challenger's dispute (if any) was addressed
    challenge_acknowledged: Optional[str] = None

    def validate(self) -> None:
        valid_statuses = {"root_cause_identified", "insufficient_evidence"}

        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {sorted(valid_statuses)}, "
                f"got '{self.status}'."
            )

        if self.status == "root_cause_identified":
            if not self.hypothesis_id:
                raise ValueError(
                    "hypothesis_id is required when status is "
                    "'root_cause_identified'."
                )
            if not self.root_cause_explanation:
                raise ValueError(
                    "root_cause_explanation is required when status "
                    "is 'root_cause_identified'."
                )
            if not self.remediation_steps:
                raise ValueError(
                    "At least one remediation step is required when "
                    "status is 'root_cause_identified'."
                )
            for step in self.remediation_steps:
                step.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "status": self.status,
            "hypothesis_id": self.hypothesis_id,
            "root_cause_explanation": self.root_cause_explanation,
            "confidence": self.confidence,
            "remediation_steps": [s.to_dict() for s in self.remediation_steps],
            "residual_risk": self.residual_risk,
            "challenge_acknowledged": self.challenge_acknowledged,
        }


def directional_confidence(assessment: HypothesisAssessment) -> float:
    """
    Return confidence as "likelihood this hypothesis is the true
    cause", normalized regardless of how the Evidence Analyst
    happened to interpret confidence direction this run.

    supported / weakened -> confidence already points toward
        "likely the cause", used as-is.
    rejected -> confidence is expected to point toward "likely NOT
        the cause" per the prompt, but the Evidence Analyst has
        been observed inverting this. Since a rejected hypothesis
        should never be treated as a plausible cause regardless of
        its reported number, this is deterministically floored to
        0.0 rather than trusted.
    proposed -> insufficient evidence to say either way; treated as
        unknown, not zero.
    """

    if assessment.status == HypothesisStatus.REJECTED:
        return 0.0

    if assessment.status == HypothesisStatus.PROPOSED:
        return assessment.confidence  # informational only, not a claim of likelihood

    return assessment.confidence  # supported / weakened


def select_leading_hypothesis(
    evidence_analysis: EvidenceAnalysis,
    minimum_confidence: float,
) -> Optional[HypothesisAssessment]:
    """
    Deterministically select the strongest candidate root cause.

    Only "supported" hypotheses are eligible, and only if they clear
    the configured minimum_diagnosis_confidence threshold. This is
    pre-filtering done in code, not by the LLM, so the LLM's job
    narrows to explaining and recommending — not ranking.
    """

    supported = [
        a for a in evidence_analysis.assessments
        if a.status == HypothesisStatus.SUPPORTED
        and directional_confidence(a) >= minimum_confidence
    ]

    if not supported:
        return None

    return max(supported, key=directional_confidence)


def build_root_cause_context(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    leading_assessment: HypothesisAssessment,
    challenge_report: Optional[ChallengeReport],
) -> Dict[str, Any]:
    """
    Build bounded context for Root Cause & Remediation.

    Only the leading (already-selected) hypothesis and its matching
    Challenger verdict are included — this agent explains and
    recommends, it does not re-select among hypotheses.
    """

    hypotheses_by_id = {h.hypothesis_id: h for h in hypothesis_set.hypotheses}
    leading_hypothesis = hypotheses_by_id.get(leading_assessment.hypothesis_id)

    matching_challenge = None
    if challenge_report is not None:
        for c in challenge_report.challenges:
            if c.hypothesis_id == leading_assessment.hypothesis_id:
                matching_challenge = c.to_dict()
                break

    return {
        "incident_frame": incident_frame.to_dict(),
        "leading_hypothesis": (
            leading_hypothesis.to_dict() if leading_hypothesis else None
        ),
        "leading_assessment": leading_assessment.to_dict(),
        "challenger_verdict": matching_challenge,
    }


@observe(name="root_cause_remediation_agent")
def root_cause_remediation_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_analysis: EvidenceAnalysis,
    challenge_report: Optional[ChallengeReport],
    investigation_policy: InvestigationPolicy,
) -> RootCauseRecommendation:
    """
    Commit to a root cause and recommend remediation, or explicitly
    report insufficient evidence.

    Hypothesis selection is deterministic (select_leading_hypothesis)
    and happens in code before this agent is called. The LLM's job
    is narrower: explain the chosen cause in plain language, address
    the Challenger's dispute if one exists, and propose concrete
    remediation steps.
    """

    agent_name = "root_cause_remediation_agent"

    leading_assessment = select_leading_hypothesis(
        evidence_analysis=evidence_analysis,
        minimum_confidence=investigation_policy.minimum_diagnosis_confidence,
    )

    if leading_assessment is None:
        recommendation = RootCauseRecommendation(
            incident_id=incident_frame.incident_id,
            status="insufficient_evidence",
        )
        recommendation.validate()
        return recommendation

    context = build_root_cause_context(
        incident_frame=incident_frame,
        hypothesis_set=hypothesis_set,
        leading_assessment=leading_assessment,
        challenge_report=challenge_report,
    )

    system_prompt = """
You are the Root Cause & Remediation Agent for an autonomous network
investigation team.

A leading hypothesis has already been selected deterministically —
you do NOT choose among hypotheses. Your job is:

1. Explain, in plain language a network engineer would write in an
   incident report, why the leading hypothesis is the root cause,
   grounded only in the supplied evidence.
2. If a challenger_verdict is present and its challenge_outcome is
   "disputed", you must explicitly address the dispute in your
   explanation — either explain why the original evidence still
   holds despite the dispute, or note it as an open risk in
   residual_risk. Do NOT ignore a disputed challenge.
3. Propose concrete, ordered remediation steps. Each step needs a
   risk_level ("low", "medium", "high") reflecting the risk of
   performing that action, not the risk of the incident itself.
4. State any residual_risk: what could still be wrong even if this
   remediation is applied, given any gaps the evidence has.

Critical rules:

- Do NOT propose remediation steps that are not grounded in the
  evidence (e.g. do not suggest replacing hardware if the evidence
  only shows a logical state, not a confirmed physical fault).
- Do NOT invent additional hypotheses or entities.
- Prefer the least invasive remediation that addresses the
  evidenced cause; do not escalate to drastic steps without
  justification.
- If challenger_verdict shows challenge_outcome "disputed", you
  MUST populate challenge_acknowledged with how you addressed it.
  If there is no challenger_verdict or the outcome is "upheld",
  set challenge_acknowledged to a brief note saying no unresolved
  dispute exists.

Return ONLY a Python dictionary with exactly this shape:

{
    "root_cause_explanation": str,
    "remediation_steps": [
        {
            "action": str,
            "rationale": str,
            "risk_level": str
        }
    ],
    "residual_risk": str,
    "challenge_acknowledged": str
}
"""

    user_prompt = f"""
Investigation context:

{context}

Explain the root cause and recommend remediation.
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    print("Root cause remediation LLM call completed")

    content = response["content"]

    if not content:
        raise ValueError(
            "Root Cause & Remediation Agent returned empty LLM content."
        )

    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=ROOT_CAUSE_RECOMMENDATION_SCHEMA,
        agent_name=agent_name,
    )

    remediation_steps = []

    for index, item in enumerate(parsed.get("remediation_steps", []), start=1):
        step = RemediationStep(
            step_number=index,
            action=item["action"],
            rationale=item.get("rationale", ""),
            risk_level=item.get("risk_level", "medium"),
        )
        step.validate()
        remediation_steps.append(step)

    recommendation = RootCauseRecommendation(
        incident_id=incident_frame.incident_id,
        status="root_cause_identified",
        hypothesis_id=leading_assessment.hypothesis_id,
        root_cause_explanation=parsed.get("root_cause_explanation", ""),
        confidence=directional_confidence(leading_assessment),
        remediation_steps=remediation_steps,
        residual_risk=parsed.get("residual_risk"),
        challenge_acknowledged=parsed.get("challenge_acknowledged"),
    )

    recommendation.validate()

    return recommendation
