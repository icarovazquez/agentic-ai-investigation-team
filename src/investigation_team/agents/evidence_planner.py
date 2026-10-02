"""
Evidence Planner Agent: decides what to check and how, deterministically
in code wherever possible. Third agent in the chain, and the one that
re-runs each round of the reasoning loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langfuse import observe

from ..capabilities.registry import available_capability_names, capability_descriptions
from ..domain import EvidencePlan, EvidenceTest
from ..llm import llm_call, parse_or_repair_agent_response
from .evidence_analyst import EvidenceAnalysis, normalize_hypothesis_id
from .hypothesis_challenger import ChallengeReport
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame

EVIDENCE_PLAN_SCHEMA = """
{
    "tests": [
        {
            "hypothesis_id": str,
            "objective": str,
            "capability": str,
            "parameters": dict,
            "expected_supporting_observations": list[str],
            "expected_falsifying_observations": list[str],
            "priority": int
        }
    ]
}
"""


def build_evidence_planning_context(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    prior_evidence_analysis: Optional[EvidenceAnalysis] = None,
    prior_challenge_report: Optional[ChallengeReport] = None,
) -> Dict[str, Any]:
    """
    Build the bounded context supplied to the Evidence Planning Agent.

    If prior_evidence_analysis / prior_challenge_report are supplied,
    this is a subsequent reasoning-loop round: the context includes
    what's still missing so far, so planning targets those gaps
    specifically rather than re-deriving the same tests as round 1.
    """

    context: Dict[str, Any] = {
        "incident_frame": incident_frame.to_dict(),

        "hypotheses": [
            hypothesis.to_dict()
            for hypothesis in hypothesis_set.hypotheses
        ],

        "available_capabilities": available_capability_names(),
        "capability_descriptions": capability_descriptions(),
    }

    if prior_evidence_analysis is not None:
        context["prior_round_missing_evidence"] = [
            {
                "hypothesis_id": assessment.hypothesis_id,
                "status": assessment.status.value,
                "confidence": assessment.confidence,
                "missing_evidence": assessment.missing_evidence,
            }
            for assessment in prior_evidence_analysis.assessments
            if assessment.missing_evidence
        ]

    if prior_challenge_report is not None:
        context["prior_round_challenger_gaps"] = [
            {
                "hypothesis_id": challenge.hypothesis_id,
                "challenge_outcome": challenge.challenge_outcome,
                "additional_evidence_needed": challenge.additional_evidence_needed,
            }
            for challenge in prior_challenge_report.challenges
            if challenge.additional_evidence_needed
        ]

    return context


@observe(name="evidence_planning_agent")
def evidence_planning_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    round_number: int = 1,
    prior_evidence_analysis: Optional[EvidenceAnalysis] = None,
    prior_challenge_report: Optional[ChallengeReport] = None,
) -> EvidencePlan:
    """
    Convert competing hypotheses into explicit capability requests.

    This agent plans evidence collection only.
    It does not execute tests or judge hypotheses.

    round_number identifies which reasoning-loop pass this is
    (1 for the first round). It is used only to keep test_id unique
    across rounds within one investigation — evidence_results
    accumulates across rounds, so a round-2 test_id colliding with
    a round-1 test_id would make two distinct tests indistinguishable
    in the combined results list.

    When prior_evidence_analysis / prior_challenge_report are
    supplied, this is a subsequent reasoning-loop round: planning
    should prioritize the gaps those identified over re-deriving
    tests from the hypotheses alone.
    """

    agent_name = "evidence_planning_agent"

    context = build_evidence_planning_context(
        incident_frame=incident_frame,
        hypothesis_set=hypothesis_set,
        prior_evidence_analysis=prior_evidence_analysis,
        prior_challenge_report=prior_challenge_report,
    )

    is_subsequent_round = (
        prior_evidence_analysis is not None
        or prior_challenge_report is not None
    )

    system_prompt = """
You are the Evidence Planning Agent for an autonomous network
investigation team.

Your responsibility is to convert each hypothesis into a minimal set
of deterministic evidence tests.

Important rules:

1. Do NOT decide whether a hypothesis is correct.
2. Do NOT collect evidence yourself.
3. Do NOT invoke executors directly.
4. Request only capabilities listed under available_capabilities.
5. Every test must have a clear objective.
6. Every test must identify which hypothesis it evaluates.
7. Reuse a test when one evidence request can evaluate multiple ideas,
   but each returned test must reference one hypothesis_id.
8. Prefer the smallest set of high-information tests. Maximum 2
   tests per hypothesis per round — if you find yourself writing a
   3rd test for the same hypothesis, combine it into one of the
   first 2 by broadening that test's objective instead.
9. Use only entity IDs present in the supplied context.
10. Parameters must contain only values needed by the capability.
11. Do not invent capability names.
12. Do not use benchmark ground truth.
13. A single test's objective may ask a capability to report on
    MULTIPLE related diagnostic questions at once (e.g. "check
    error counters, administrative status, and signal quality on
    this interface" is ONE test, not three) — capabilities return
    whatever evidence they have for the requested entities, so
    splitting related questions about the same entity into separate
    tests wastes budget without adding information.

If prior_round_missing_evidence or prior_round_challenger_gaps are
present in the context, this is NOT the first planning round for
this investigation. Evidence has already been collected and
analyzed, and it was not sufficient. In this case:

14. Prioritize tests that would resolve the specific gaps listed in
    prior_round_challenger_gaps first — these come from a
    dedicated adversarial review of the leading hypothesis, and
    are the highest-value gaps to close.
15. Then address gaps in prior_round_missing_evidence not already
    covered by a test targeting the challenger gaps.
16. Do NOT repeat a test that would collect the same evidence as a
    prior round already attempted and found insufficient — the gap
    exists because that evidence is genuinely unavailable or was
    already checked, not because no one asked. Plan a DIFFERENT
    test/capability/parameters that could close the gap another way,
    or state within the objective why no further test can close it.
17. It is acceptable for this round's plan to contain fewer tests
    than the hypothesis count, if only a few real gaps remain.
18. The max-2-tests-per-hypothesis rule (rule 8) still applies here.
    If more than 2 gaps exist for one hypothesis, prioritize the
    single highest-value gap and combine the rest into that same
    test's objective — do NOT write one test per listed gap.

Return ONLY a Python dictionary with exactly this shape:

{
    "tests": [
        {
            "hypothesis_id": str,
            "objective": str,
            "capability": str,
            "parameters": dict,
            "expected_supporting_observations": list[str],
            "expected_falsifying_observations": list[str],
            "priority": int
        }
    ]
}

For topology and reachability tests, parameters may include:
{
    "source_entity_id": str,
    "target_entity_id": str
}

For network_state tests, parameters may include:
{
    "entity_ids": list[str]
}

Priority 1 means highest priority.
"""

    user_prompt = f"""
Investigation context:

{context}

{"This is a subsequent reasoning round. Prior evidence was insufficient to reach a confident, unchallenged conclusion. Focus this round's plan on closing the specific gaps identified above." if is_subsequent_round else ""}

Create the evidence collection plan.
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
        raise ValueError(
            "Evidence Planning Agent returned empty LLM content."
        )

    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=EVIDENCE_PLAN_SCHEMA,
        agent_name=agent_name,
    )

    raw_tests = parsed.get("tests", [])

    # ------------------------------------------------------
    # Step 1: filter out items missing required fields
    # ------------------------------------------------------

    REQUIRED_TEST_FIELDS = ("hypothesis_id", "objective", "capability")

    valid_raw_tests = []

    for i, item in enumerate(raw_tests):
        missing_fields = [
            field for field in REQUIRED_TEST_FIELDS
            if field not in item or not item[field]
        ]

        if missing_fields:
            print(
                f"⚠ evidence_planning_agent test #{i + 1} is missing "
                f"required field(s) {missing_fields} — skipping it: "
                f"{item}"
            )
            continue

        valid_raw_tests.append(item)

    raw_tests = valid_raw_tests

    # ------------------------------------------------------
    # Step 2: normalize hypothesis_id and group by hypothesis
    # ------------------------------------------------------

    raw_tests_by_hypothesis: Dict[str, List[Dict[str, Any]]] = {}

    for item in raw_tests:
        try:
            normalized_id = normalize_hypothesis_id(
                returned_id=item["hypothesis_id"],
                hypothesis_set=hypothesis_set,
            )
        except ValueError:
            print(
                f"⚠ evidence_planning_agent test has unresolvable "
                f"hypothesis_id '{item['hypothesis_id']}' — skipping it"
            )
            continue

        item["hypothesis_id"] = normalized_id
        raw_tests_by_hypothesis.setdefault(normalized_id, []).append(item)

    # ------------------------------------------------------
    # Step 3: deterministic cap — max N tests per hypothesis,
    # kept by priority. Backstop against the model ignoring
    # rule 8/18, which it has done multiple times.
    # ------------------------------------------------------

    MAX_TESTS_PER_HYPOTHESIS = 2

    def _priority(item: Dict[str, Any]) -> int:
        try:
            return int(item.get("priority", 1))
        except (TypeError, ValueError):
            return 1

    capped_raw_tests = []

    for hyp_id, items in raw_tests_by_hypothesis.items():
        sorted_items = sorted(items, key=_priority)
        capped_raw_tests.extend(sorted_items[:MAX_TESTS_PER_HYPOTHESIS])

        if len(sorted_items) > MAX_TESTS_PER_HYPOTHESIS:
            print(
                f"⚠ evidence_planning_agent requested "
                f"{len(sorted_items)} tests for hypothesis "
                f"'{hyp_id}' — capping to "
                f"{MAX_TESTS_PER_HYPOTHESIS} by priority"
            )

    raw_tests = capped_raw_tests

    if not raw_tests:
        raise ValueError(
            "Evidence Planning Agent: no valid tests remained after "
            "filtering malformed items."
        )

    # ------------------------------------------------------
    # Step 4: build EvidenceTest objects
    # ------------------------------------------------------

    tests = []

    for index, item in enumerate(raw_tests, start=1):
        test = EvidenceTest(
            test_id=f"{incident_frame.incident_id}-r{round_number}-test-{index}",
            hypothesis_id=item["hypothesis_id"],
            objective=item["objective"],
            capability=item["capability"],
            parameters=item.get("parameters", {}),
            expected_supporting_observations=item.get(
                "expected_supporting_observations", []
            ),
            expected_falsifying_observations=item.get(
                "expected_falsifying_observations", []
            ),
            priority=int(item.get("priority", 1)),
        )

        test.validate()
        tests.append(test)

    evidence_plan = EvidencePlan(
        incident_id=incident_frame.incident_id,
        tests=tests,
    )

    evidence_plan.validate()

    return evidence_plan
