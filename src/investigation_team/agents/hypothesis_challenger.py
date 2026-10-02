"""
Hypothesis Challenger Agent: argues against the leading hypothesis
using only the evidence already on the table. Fifth agent in the
chain, and the loop's exit gate alongside confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langfuse import observe

from ..llm import llm_call, parse_or_repair_agent_response
from .evidence_analyst import EvidenceAnalysis, HypothesisAssessment
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame

CHALLENGE_REPORT_SCHEMA = """
{
    "challenges": [
        {
            "hypothesis_id": str,
            "challenge_outcome": str,
            "rebuttal": str,
            "alternative_explanation": str | None,
            "additional_evidence_needed": list[str]
        }
    ],
    "systemic_concerns": list[str]
}
"""


@dataclass
class HypothesisChallenge:
    """
    Adversarial review of one hypothesis assessment.

    The Challenger does not re-score confidence independently — it
    either upholds the Evidence Analyst's assessment or disputes it
    with a specific rebuttal grounded in the same evidence.
    """

    hypothesis_id: str
    original_status: str
    original_confidence: float

    challenge_outcome: str  # "upheld" | "disputed" | "unresolved"
    rebuttal: str

    alternative_explanation: Optional[str] = None
    additional_evidence_needed: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty.")

        valid_outcomes = {"upheld", "disputed", "unresolved"}

        if self.challenge_outcome not in valid_outcomes:
            raise ValueError(
                "challenge_outcome must be one of "
                f"{sorted(valid_outcomes)}, got "
                f"'{self.challenge_outcome}'."
            )

        if not self.rebuttal.strip():
            raise ValueError("rebuttal cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChallengeReport:
    """
    Full adversarial review of an investigation's evidence analysis.
    """

    incident_id: str
    challenges: List[HypothesisChallenge]

    systemic_concerns: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.incident_id.strip():
            raise ValueError("incident_id cannot be empty.")

        if not self.challenges:
            raise ValueError(
                "ChallengeReport must contain at least one challenge."
            )

        for challenge in self.challenges:
            challenge.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "challenges": [c.to_dict() for c in self.challenges],
            "systemic_concerns": list(self.systemic_concerns),
        }


def rank_hypothesis_assessments(
    evidence_analysis: EvidenceAnalysis,
) -> List[HypothesisAssessment]:
    """
    Deterministically rank assessments — this is code, not an LLM
    call, so ranking is reproducible and the LLM never has to
    invent an ordering from scratch.
    """

    status_priority = {
        "supported": 0,
        "proposed": 1,
        "weakened": 2,
        "rejected": 3,
    }

    return sorted(
        evidence_analysis.assessments,
        key=lambda a: (
            status_priority.get(a.status.value, 99),
            -a.confidence,
        ),
    )


def build_challenger_context(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_analysis: EvidenceAnalysis,
) -> Dict[str, Any]:
    """
    Build bounded context for the Hypothesis Challenger.

    Only the top-ranked and runner-up assessments are passed in
    full detail — the Challenger's job is to stress-test the
    leading explanation(s), not re-review everything from scratch.
    """

    ranked = rank_hypothesis_assessments(evidence_analysis)

    hypotheses_by_id = {
        h.hypothesis_id: h
        for h in hypothesis_set.hypotheses
    }

    def assessment_package(assessment: HypothesisAssessment) -> Dict[str, Any]:
        hypothesis = hypotheses_by_id.get(assessment.hypothesis_id)

        return {
            "assessment": assessment.to_dict(),
            "hypothesis": (
                hypothesis.to_dict() if hypothesis else None
            ),
        }

    return {
        "incident_frame": incident_frame.to_dict(),

        "leading_assessment": (
            assessment_package(ranked[0]) if ranked else None
        ),

        "runner_up_assessment": (
            assessment_package(ranked[1]) if len(ranked) > 1 else None
        ),

        "all_assessment_statuses": [
            {
                "hypothesis_id": a.hypothesis_id,
                "status": a.status.value,
                "confidence": a.confidence,
            }
            for a in ranked
        ],
    }


@observe(name="hypothesis_challenger_agent")
def hypothesis_challenger_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_analysis: EvidenceAnalysis,
) -> ChallengeReport:
    """
    Adversarially review the leading hypothesis assessment(s).

    This agent argues against the current leading explanation using
    only the supplied evidence. It does not select a root cause or
    recommend remediation.
    """

    agent_name = "hypothesis_challenger_agent"

    context = build_challenger_context(
        incident_frame=incident_frame,
        hypothesis_set=hypothesis_set,
        evidence_analysis=evidence_analysis,
    )

    system_prompt = """
You are the Hypothesis Challenger Agent for an autonomous network
investigation team.

Your job is to argue AGAINST the leading hypothesis assessment,
using only the evidence already supplied. You are a devil's
advocate, not a second Evidence Analyst.

For the leading_assessment and runner_up_assessment supplied:

1. Identify the strongest counter-argument the evidence allows,
   even if you believe the original assessment is likely correct.
2. State whether the original assessment is "upheld" (you found no
   genuine counter-argument), "disputed" (you found a real
   weakness), or "unresolved" (evidence is too thin to argue either
   way).
3. If you dispute an assessment, name a specific alternative
   explanation the evidence would also be consistent with.
4. List any additional evidence that would resolve the dispute.

Also assess systemic_concerns across the whole investigation, such
as: was only one hypothesis meaningfully tested; did evidence
planning favor the leading hypothesis; are there hypotheses with
"proposed" status that were never actually challenged.

Critical rules:

- Do NOT determine root cause.
- Do NOT recommend remediation.
- Do NOT invent evidence not present in the supplied context.
- Every hypothesis passed to you (leading and runner-up, if
  present) must receive exactly one challenge entry.
- A rebuttal must reference specific evidence, not general doubt.

Return ONLY a Python dictionary with exactly this shape:

{
    "challenges": [
        {
            "hypothesis_id": str,
            "challenge_outcome": str,
            "rebuttal": str,
            "alternative_explanation": str | None,
            "additional_evidence_needed": list[str]
        }
    ],
    "systemic_concerns": list[str]
}

challenge_outcome must be exactly one of: "upheld", "disputed",
"unresolved".
"""

    user_prompt = f"""
Investigation context:

{context}

Challenge the leading hypothesis assessment(s).
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    print("Hypothesis challenge LLM call completed")

    content = response["content"]

    if not content:
        raise ValueError(
            "Hypothesis Challenger Agent returned empty LLM content."
        )

    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=CHALLENGE_REPORT_SCHEMA,
        agent_name=agent_name,
    )

    challenges = []

    for item in parsed.get("challenges", []):
        challenge = HypothesisChallenge(
            hypothesis_id=item["hypothesis_id"],
            original_status="",  # filled in below
            original_confidence=0.0,
            challenge_outcome=item["challenge_outcome"],
            rebuttal=item["rebuttal"],
            alternative_explanation=item.get("alternative_explanation"),
            additional_evidence_needed=item.get(
                "additional_evidence_needed", []
            ),
        )
        challenges.append(challenge)

    # Backfill original status/confidence from the source assessments
    # rather than trusting the LLM to echo them accurately.
    assessments_by_id = {
        a.hypothesis_id: a for a in evidence_analysis.assessments
    }

    for challenge in challenges:
        source = assessments_by_id.get(challenge.hypothesis_id)

        if source is not None:
            challenge.original_status = source.status.value
            challenge.original_confidence = source.confidence

        challenge.validate()

    report = ChallengeReport(
        incident_id=incident_frame.incident_id,
        challenges=challenges,
        systemic_concerns=parsed.get("systemic_concerns", []),
    )

    report.validate()

    return report
