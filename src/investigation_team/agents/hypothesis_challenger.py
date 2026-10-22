"""
Hypothesis Challenger Agent: argues against the leading hypothesis
using only the evidence already on the table. Fifth agent in the
chain, and the loop's exit gate alongside confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from langfuse import observe

from ..capabilities.registry import capability_descriptions, capability_observables, resolve_capability_name
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
            "rival_hypothesis_id": str | None,
            "distinguishing_capability": str | None,
            "distinguishing_observable": str | None,
            "distinguishing_observation": str | None,
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

    # The hypothesis the Challenger believes is the competing
    # explanation, when it is one of the hypotheses already assessed.
    # If that rival is itself supported, the question is how the two
    # relate (rivals or a causal chain), which more evidence cannot
    # settle: it goes to causal resolution.
    rival_hypothesis_id: Optional[str] = None

    # A dispute must say which registered capability could produce the
    # observation that separates the two explanations. Code, not the
    # prompt, enforces this (see enforce_dispute_actionability).
    distinguishing_capability: Optional[str] = None
    distinguishing_observable: Optional[str] = None
    distinguishing_observation: Optional[str] = None

    # Set when a dispute was downgraded to "upheld" because nothing the
    # investigation can collect would resolve it.
    downgrade_note: Optional[str] = None

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


def enforce_dispute_actionability(
    challenges: List[HypothesisChallenge],
    supported_ids: Optional[List[str]] = None,
) -> List[str]:
    """
    A dispute only keeps the loop running if another round could
    resolve it. That requires the Challenger to name (a) a registered
    capability that can measure the distinguishing observation and
    (b) the observation itself. Otherwise another round cannot help:
    the dispute is downgraded to "upheld" and kept as a recorded
    concern (downgrade_note) instead of a blocking objection.

    Mutates the challenges in place and returns human-readable notes
    for the run log. This is code on purpose: asking the Challenger
    to police its own requests is the same self-policing that failed
    elsewhere in the pipeline.
    """

    notes: List[str] = []

    for challenge in challenges:
        if challenge.challenge_outcome != "disputed":
            continue

        rival = (challenge.rival_hypothesis_id or "").strip()
        if (
            rival
            and rival != challenge.hypothesis_id
            and rival in (supported_ids or [])
        ):
            challenge.challenge_outcome = "upheld"
            challenge.downgrade_note = (
                f"Dispute routed to causal resolution: the rival "
                f"({rival}) is itself supported, so the open question is "
                "how the two relate (rivals or a causal chain), which "
                "another evidence round cannot settle."
            )
            notes.append(f"{challenge.hypothesis_id}: {challenge.downgrade_note}")
            continue

        capability = resolve_capability_name(
            challenge.distinguishing_capability or ""
        )
        observable = (challenge.distinguishing_observable or "").strip()
        observation = (challenge.distinguishing_observation or "").strip()
        declared = capability_observables().get(capability or "", ())

        if capability is not None and observable in declared and observation:
            challenge.distinguishing_capability = capability
            continue

        if capability is None:
            reason = (
                "no available capability was named that could produce "
                "the distinguishing observation"
            )
        elif observable not in declared:
            reason = (
                f"'{observable or 'nothing'}' is not something {capability} "
                f"can return (it returns: {', '.join(declared) or 'nothing'})"
            )
        else:
            reason = "no distinguishing observation was specified"

        challenge.challenge_outcome = "upheld"
        challenge.downgrade_note = (
            f"Dispute downgraded to upheld: {reason}, so another "
            "evidence round could not resolve it."
        )
        notes.append(f"{challenge.hypothesis_id}: {challenge.downgrade_note}")

    return notes


def describe_challenge_outcome(challenge: HypothesisChallenge) -> str:
    """
    Reader-facing label. "upheld" means the Challenger accepted the
    Analyst's assessment, which for a rejected hypothesis means the
    rejection stands, so say that instead of a bare "upheld".
    """

    if challenge.challenge_outcome != "upheld":
        return challenge.challenge_outcome

    stands = {
        "supported": "support stands",
        "rejected": "rejection stands",
        "weakened": "weakening stands",
    }.get(challenge.original_status)

    label = "upheld"
    if stands:
        label += f" ({stands})"
    if challenge.downgrade_note:
        label += " [dispute downgraded]"
    return label


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
2. Then judge whether the assessment SURVIVES that counter-argument:
   - "upheld": the strongest counter-argument does not change the
     conclusion. The evidence already in hand is more consistent
     with the assessment than with the counter-argument. Missing
     evidence alone is NOT a reason to dispute: every investigation
     has missing evidence. Use "upheld" when you would still act on
     the assessment after considering your own objection.
   - "disputed": the counter-argument names a specific alternative
     explanation that the evidence in hand is EQUALLY consistent
     with, so the conclusion is not yet safe. You must be able to
     name the observation that would tell the two apart AND the
     available capability (from the list below) that could collect
     it. If no available capability could produce that observation,
     another evidence round cannot help: use "upheld" and put the
     observation in additional_evidence_needed instead.
     A more specific or less specific form of the SAME cause (for
     example transient versus permanent forms of the same fault)
     refines the hypothesis; it is not a rival explanation.
   - "unresolved": the evidence is too thin to argue either way.
3. If you dispute an assessment, name that specific alternative
   explanation, rival_hypothesis_id (the hypothesis_id from
   all_assessment_statuses that states that alternative, or null if
   it is none of them), plus distinguishing_capability (exactly one name from
   the available capabilities), distinguishing_observable (exactly one
   of the observables listed for that capability) and
   distinguishing_observation (what you expect it to show under each
   explanation). For upheld or unresolved challenges set all three to
   null. Anything not in an observables list (optical power, signal
   quality, historical trends, timestamps, logs) cannot be collected.
4. List any additional evidence that would resolve the dispute. If
   you upheld the assessment, list only evidence that would further
   strengthen it (it need not be collected).

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
            "rival_hypothesis_id": str | None,
            "distinguishing_capability": str | None,
            "distinguishing_observable": str | None,
            "distinguishing_observation": str | None,
            "additional_evidence_needed": list[str]
        }
    ],
    "systemic_concerns": list[str]
}

challenge_outcome must be exactly one of: "upheld", "disputed",
"unresolved".
"""

    observables = capability_observables()
    capability_catalog = "\n".join(
        f"- {name}: {description} [observables: {', '.join(observables.get(name, ())) or 'none'}]"
        for name, description in capability_descriptions().items()
    )
    system_prompt += (
        "\nAvailable capabilities (the ONLY evidence this investigation "
        "can collect):\n" + capability_catalog + "\n"
    )

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
            rival_hypothesis_id=item.get("rival_hypothesis_id"),
            distinguishing_capability=item.get("distinguishing_capability"),
            distinguishing_observable=item.get("distinguishing_observable"),
            distinguishing_observation=item.get("distinguishing_observation"),
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

    # A rival id the model made up is dropped, not trusted.
    for challenge in challenges:
        if challenge.rival_hypothesis_id not in assessments_by_id:
            challenge.rival_hypothesis_id = None

    supported_ids = [
        a.hypothesis_id for a in evidence_analysis.assessments
        if a.status.value == "supported"
    ]
    for note in enforce_dispute_actionability(challenges, supported_ids):
        print(f"    ℹ {note}")

    report = ChallengeReport(
        incident_id=incident_frame.incident_id,
        challenges=challenges,
        systemic_concerns=parsed.get("systemic_concerns", []),
    )

    report.validate()

    return report
