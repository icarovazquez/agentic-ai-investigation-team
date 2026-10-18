"""
Deterministic convergence rule for the reasoning loop.

The loop used to exit only when the Challenger returned "upheld" for
the leading hypothesis. A devil's advocate is built to object, so that
gate was effectively unreachable and the loop ran to its round limit
even when the same hypothesis was supported above the confidence bar
every single round.

This rule is code, not a prompt: it looks only at the analysis history.
Open objections are never discarded -- the caller records them as
residual concerns on the result.
"""

from __future__ import annotations

from typing import List, Optional

from .agents.evidence_analyst import EvidenceAnalysis
from .agents.hypothesis_challenger import ChallengeReport
from .agents.root_cause_remediation import directional_confidence, select_leading_hypothesis
from .domain import HypothesisStatus


def check_convergence(
    analysis_history: List[EvidenceAnalysis],
    minimum_confidence: float,
    stable_rounds: int = 2,
    rival_confidence_ceiling: float = 0.5,
) -> Optional[str]:
    """
    Return the converged leader's hypothesis_id, or None.

    Converged means that for each of the last `stable_rounds` rounds
    (and at least that many have run):
      - the SAME hypothesis is the leader (supported, at or above
        `minimum_confidence`), and
      - no other hypothesis is credible (directional confidence above
        `rival_confidence_ceiling`).

    stable_rounds < 1 disables the rule.
    """
    if stable_rounds < 1 or len(analysis_history) < stable_rounds:
        return None

    leader_ids = set()
    for analysis in analysis_history[-stable_rounds:]:
        leader = select_leading_hypothesis(analysis, minimum_confidence)
        if leader is None:
            return None
        for other in analysis.assessments:
            if (
                other.hypothesis_id != leader.hypothesis_id
                and other.status != HypothesisStatus.REJECTED
                and directional_confidence(other) > rival_confidence_ceiling
            ):
                return None
        leader_ids.add(leader.hypothesis_id)

    return leader_ids.pop() if len(leader_ids) == 1 else None


def residual_concerns(
    challenge_report: Optional[ChallengeReport],
    leader_id: str,
) -> List[str]:
    """Objections still open when the loop stops, kept for the final report."""
    if challenge_report is None:
        return []
    concerns: List[str] = []
    for c in challenge_report.challenges:
        if c.hypothesis_id == leader_id and c.challenge_outcome != "upheld":
            concerns.append(f"Challenger ({c.challenge_outcome}): {c.rebuttal}")
            concerns.extend(f"Evidence that would resolve it: {e}" for e in c.additional_evidence_needed)
    concerns.extend(f"Systemic: {s}" for s in challenge_report.systemic_concerns)
    return concerns
