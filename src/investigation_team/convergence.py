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
from .agents.hypothesis_challenger import ROUTED_NOTE_PREFIX
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
    extra_ids: Optional[List[str]] = None,
    after_causal_chain: bool = False,
) -> List[str]:
    """
    Objections still open when the loop stops, kept for the final report.

    After a causal chain was resolved (after_causal_chain=True) the
    Challenger's disputes that were routed to causal resolution are
    settled by it, so their rebuttal is not an open concern (see
    resolved_by_causal_resolution); the evidence they asked for still is.
    Systemic concerns are free text written before the resolution, so they
    are kept but labelled as such instead of being filtered by guesswork.
    """
    if challenge_report is None:
        return []
    ids = {leader_id, *(extra_ids or [])}
    concerns: List[str] = []
    for c in challenge_report.challenges:
        if c.hypothesis_id not in ids:
            continue
        if c.challenge_outcome != "upheld" or c.downgrade_note:
            routed = after_causal_chain and is_routed_to_causal_resolution(c)
            if not routed:
                label = "downgraded dispute" if c.downgrade_note else c.challenge_outcome
                concerns.append(f"Challenger ({label}): {c.rebuttal}")
            concerns.extend(f"Evidence that would resolve it: {e}" for e in c.additional_evidence_needed)
    prefix = "Systemic (raised before causal resolution)" if after_causal_chain else "Systemic"
    concerns.extend(f"{prefix}: {s}" for s in challenge_report.systemic_concerns)
    return concerns


def is_routed_to_causal_resolution(challenge) -> bool:
    return bool(challenge.downgrade_note) and challenge.downgrade_note.startswith(ROUTED_NOTE_PREFIX)


def resolved_by_causal_resolution(
    challenge_report: Optional[ChallengeReport],
    resolution,
) -> List[str]:
    """Disputes that causal resolution settled, with how it settled them."""
    if challenge_report is None or resolution is None or resolution.relationship != "causal_chain":
        return []
    return [
        f"{c.hypothesis_id} was disputed on the grounds that {c.rebuttal[:300]} "
        f"Settled by causal resolution: {resolution.rationale[:300]}"
        for c in challenge_report.challenges
        if is_routed_to_causal_resolution(c)
    ]
