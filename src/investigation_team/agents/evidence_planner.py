"""
Evidence Planner Agent: decides WHICH gaps matter and HOW MANY are
worth investigating, deterministically capped in code. Produces an
EvidenceGapSelection, not full test objects -- writing fully-formed,
correctly-sized test objects is evidence_test_creation_agent's job
(agents/evidence_test_creator.py). Splitting these was necessary
because one agent doing both kept failing in four different ways:
invalid syntax, abandoned schema, excessive volume, and composite
hypothesis_ids -- and an explicit "max 8 tests total" rule in the
prompt was ignored twice in a row even after the per-hypothesis cap
was already enforced in code.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from langfuse import observe

from ..capabilities.registry import (
    available_capability_names,
    capability_descriptions,
    resolve_capability_name,
)
from ..llm import llm_call, parse_or_repair_agent_response
from .evidence_analyst import EvidenceAnalysis, normalize_hypothesis_id
from .hypothesis_challenger import ChallengeReport
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame

EVIDENCE_GAP_SCHEMA = """
{
    "gaps": [
        {
            "hypothesis_id": str,
            "gap_description": str,
            "suggested_capability": str,
            "priority": int
        }
    ]
}
"""

MAX_GAPS_PER_HYPOTHESIS = 2
MAX_TOTAL_GAPS = 8


@dataclass
class EvidenceGap:
    """
    One identified gap in what's known about a hypothesis -- not yet
    a test. suggested_capability is a strong default that
    evidence_test_creation_agent may override with justification,
    not a hard constraint.
    """

    hypothesis_id: str
    gap_description: str
    suggested_capability: str
    priority: int = 1

    def validate(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty.")
        if not self.gap_description.strip():
            raise ValueError("gap_description cannot be empty.")
        if not self.suggested_capability.strip():
            raise ValueError("suggested_capability cannot be empty.")
        if self.priority < 1:
            raise ValueError("priority must be at least 1.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceGapSelection:
    """
    A bounded, already-capped list of gaps worth investigating this
    round. By the time this exists, MAX_GAPS_PER_HYPOTHESIS and
    MAX_TOTAL_GAPS have already been enforced -- nothing downstream
    needs to re-check volume.
    """

    incident_id: str
    gaps: List[EvidenceGap]

    def validate(self) -> None:
        if not self.incident_id.strip():
            raise ValueError("incident_id cannot be empty.")
        if not self.gaps:
            raise ValueError("EvidenceGapSelection must contain at least one gap.")
        for gap in self.gaps:
            gap.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {"incident_id": self.incident_id, "gaps": [g.to_dict() for g in self.gaps]}


def build_evidence_planning_context(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    prior_evidence_analysis: Optional[EvidenceAnalysis] = None,
    prior_challenge_report: Optional[ChallengeReport] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "incident_frame": incident_frame.to_dict(),
        "hypotheses": [h.to_dict() for h in hypothesis_set.hypotheses],
        "available_capabilities": available_capability_names(),
        "capability_descriptions": capability_descriptions(),
    }

    if prior_evidence_analysis is not None:
        context["prior_round_missing_evidence"] = [
            {
                "hypothesis_id": a.hypothesis_id,
                "status": a.status.value,
                "confidence": a.confidence,
                "missing_evidence": a.missing_evidence,
            }
            for a in prior_evidence_analysis.assessments
            if a.missing_evidence
        ]

    if prior_challenge_report is not None:
        context["prior_round_challenger_gaps"] = [
            {
                "hypothesis_id": c.hypothesis_id,
                "challenge_outcome": c.challenge_outcome,
                "additional_evidence_needed": c.additional_evidence_needed,
            }
            for c in prior_challenge_report.challenges
            if c.additional_evidence_needed
        ]

    return context


def _priority(item: Dict[str, Any]) -> int:
    try:
        return int(item.get("priority", 1))
    except (TypeError, ValueError):
        return 1


def select_capped_gaps(
    raw_gaps: List[Dict[str, Any]],
    hypothesis_set: HypothesisSet,
) -> List[Dict[str, Any]]:
    """
    Deterministic post-processing of the model's gap list: drop
    malformed items, normalize hypothesis ids, then cap per
    hypothesis and in total. Pure function, no LLM involved.
    """
    required = ("hypothesis_id", "gap_description", "suggested_capability")

    # Step 1: drop items missing required fields
    valid: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_gaps):
        if not isinstance(item, dict):
            print(f"⚠ evidence_planning_agent gap #{i + 1} is not a dict -- skipping it")
            continue
        missing = [f for f in required if f not in item or not item[f]]
        if missing:
            print(
                f"⚠ evidence_planning_agent gap #{i + 1} is missing "
                f"required field(s) {missing} -- skipping it: {item}"
            )
            continue
        valid.append(dict(item))

    # Step 2: normalize hypothesis_id and group by hypothesis
    by_hypothesis: Dict[str, List[Dict[str, Any]]] = {}
    canonical_ids = {h.hypothesis_id for h in hypothesis_set.hypotheses}
    for item in valid:
        raw_id = str(item["hypothesis_id"])
        # A composite id like "H1_H2_H4" would silently resolve to H1
        # via the digit fallback in normalize_hypothesis_id. A gap must
        # address exactly one hypothesis, so reject it instead.
        if raw_id not in canonical_ids and len(set(re.findall(r"\d+", raw_id))) > 1:
            print(
                f"⚠ evidence_planning_agent gap has composite "
                f"hypothesis_id '{raw_id}' -- skipping it"
            )
            continue
        try:
            normalized_id = normalize_hypothesis_id(
                returned_id=item["hypothesis_id"],
                hypothesis_set=hypothesis_set,
            )
        except ValueError:
            print(
                f"⚠ evidence_planning_agent gap has unresolvable "
                f"hypothesis_id '{item['hypothesis_id']}' -- skipping it"
            )
            continue
        capability = resolve_capability_name(item["suggested_capability"])
        if capability is None:
            print(
                f"⚠ evidence_planning_agent gap has unknown capability "
                f"'{item['suggested_capability']}' -- skipping it"
            )
            continue
        item["suggested_capability"] = capability
        item["hypothesis_id"] = normalized_id
        by_hypothesis.setdefault(normalized_id, []).append(item)

    # Step 3: deterministic caps -- per hypothesis, then total
    capped: List[Dict[str, Any]] = []
    for hyp_id, items in by_hypothesis.items():
        ordered = sorted(items, key=_priority)
        capped.extend(ordered[:MAX_GAPS_PER_HYPOTHESIS])
        if len(ordered) > MAX_GAPS_PER_HYPOTHESIS:
            print(
                f"⚠ evidence_planning_agent proposed {len(ordered)} gaps "
                f"for hypothesis '{hyp_id}' -- capping to "
                f"{MAX_GAPS_PER_HYPOTHESIS} by priority"
            )

    if len(capped) > MAX_TOTAL_GAPS:
        print(
            f"⚠ evidence_planning_agent proposed {len(capped)} gaps "
            f"total -- capping to {MAX_TOTAL_GAPS} by priority"
        )
        capped = sorted(capped, key=_priority)[:MAX_TOTAL_GAPS]

    return capped


@observe(name="evidence_planning_agent")
def evidence_planning_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    prior_evidence_analysis: Optional[EvidenceAnalysis] = None,
    prior_challenge_report: Optional[ChallengeReport] = None,
) -> EvidenceGapSelection:
    agent_name = "evidence_planning_agent"

    context = build_evidence_planning_context(
        incident_frame=incident_frame,
        hypothesis_set=hypothesis_set,
        prior_evidence_analysis=prior_evidence_analysis,
        prior_challenge_report=prior_challenge_report,
    )

    is_subsequent_round = (
        prior_evidence_analysis is not None or prior_challenge_report is not None
    )

    system_prompt = """
You are the Evidence Planning Agent for an autonomous network
investigation team.

Your ONLY job is to identify which gaps in the evidence are worth
investigating this round. You do NOT write detailed test
objectives, parameters, or expected observations -- that is a
separate agent's job. You only identify WHAT is missing and WHICH
capability would likely address it.

Important rules:

1. Do NOT decide whether a hypothesis is correct.
2. Do NOT collect evidence yourself.
3. Do NOT invoke executors directly.
4. Do NOT write test objectives, parameters, or expected
   observations -- only a short gap_description.
5. Every gap must identify exactly ONE hypothesis, using its exact
   hypothesis_id from the context. Never combine ids.
6. suggested_capability must be one of available_capabilities.
7. Maximum 2 gaps per hypothesis.
8. Maximum 8 gaps total, across all hypotheses combined.
9. Use only entity IDs / hypotheses present in the supplied context.
10. Do not invent capability names.
11. Do not use benchmark ground truth.

If prior_round_missing_evidence or prior_round_challenger_gaps are
present in the context, this is NOT the first planning round for
this investigation. In this case:

12. Prioritize gaps from prior_round_challenger_gaps first -- these
    come from a dedicated adversarial review of the leading
    hypothesis, and are the highest-value gaps to close.
13. Then address gaps in prior_round_missing_evidence not already
    covered.
14. Do NOT repeat a gap that would collect the same evidence a
    prior round already attempted and found insufficient -- describe
    a DIFFERENT angle that could close it another way, or omit it.

Return ONLY a Python dictionary with exactly this shape:

{
    "gaps": [
        {
            "hypothesis_id": str,
            "gap_description": str,
            "suggested_capability": str,
            "priority": int
        }
    ]
}

Priority 1 means highest priority.
"""

    subsequent_note = (
        "This is a subsequent reasoning round. Prior evidence was insufficient "
        "to reach a confident, unchallenged conclusion. Focus on the specific "
        "gaps identified above."
        if is_subsequent_round
        else ""
    )

    user_prompt = f"""
Investigation context:

{context}

{subsequent_note}

Identify the evidence gaps worth investigating this round.
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    print("Evidence planning LLM call completed")

    content = response["content"]
    if not content:
        raise ValueError("Evidence Planning Agent returned empty LLM content.")

    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=EVIDENCE_GAP_SCHEMA,
        agent_name=agent_name,
    )

    capped = select_capped_gaps(parsed.get("gaps", []), hypothesis_set)

    if not capped:
        raise ValueError(
            "Evidence Planning Agent: no valid gaps remained after "
            "filtering malformed items."
        )

    gaps = []
    for item in capped:
        gap = EvidenceGap(
            hypothesis_id=item["hypothesis_id"],
            gap_description=item["gap_description"],
            suggested_capability=item["suggested_capability"],
            priority=_priority(item),
        )
        gap.validate()
        gaps.append(gap)

    gap_selection = EvidenceGapSelection(incident_id=incident_frame.incident_id, gaps=gaps)
    gap_selection.validate()
    return gap_selection
