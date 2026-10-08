"""
Evidence Test Creation Agent: turns an already-bounded
EvidenceGapSelection (at most 8 gaps, capped by
evidence_planning_agent) into fully-formed EvidenceTest objects.

Deliberately narrow: it never decides WHAT to test or HOW MANY tests
to write -- both are already fixed by the time this runs. The
hypothesis_id on each resulting test comes from the gap, not from
whatever the model echoes back.
"""

from __future__ import annotations

from typing import Any, Dict

from langfuse import observe

from ..capabilities.registry import available_capability_names, capability_descriptions
from ..domain import EvidencePlan, EvidenceTest
from ..llm import llm_call, parse_or_repair_agent_response
from .evidence_planner import EvidenceGapSelection
from .hypothesis_generator import HypothesisSet
from .incident_framing import IncidentFrame

EVIDENCE_TEST_CREATION_SCHEMA = """
{
    "tests": [
        {
            "hypothesis_id": str,
            "objective": str,
            "capability": str,
            "parameters": dict,
            "expected_supporting_observations": list[str],
            "expected_falsifying_observations": list[str]
        }
    ]
}
"""


def build_evidence_test_creation_context(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_gap_selection: EvidenceGapSelection,
) -> Dict[str, Any]:
    hypotheses_by_id = {h.hypothesis_id: h for h in hypothesis_set.hypotheses}
    return {
        "incident_frame": incident_frame.to_dict(),
        "gaps": [
            {
                **gap.to_dict(),
                "hypothesis": (
                    hypotheses_by_id[gap.hypothesis_id].to_dict()
                    if gap.hypothesis_id in hypotheses_by_id else None
                ),
            }
            for gap in evidence_gap_selection.gaps
        ],
        "available_capabilities": available_capability_names(),
        "capability_descriptions": capability_descriptions(),
    }


@observe(name="evidence_test_creation_agent")
def evidence_test_creation_agent(
    incident_frame: IncidentFrame,
    hypothesis_set: HypothesisSet,
    evidence_gap_selection: EvidenceGapSelection,
    round_number: int = 1,
) -> EvidencePlan:
    agent_name = "evidence_test_creation_agent"

    context = build_evidence_test_creation_context(
        incident_frame=incident_frame,
        hypothesis_set=hypothesis_set,
        evidence_gap_selection=evidence_gap_selection,
    )
    expected_count = len(evidence_gap_selection.gaps)

    system_prompt = f"""
You are the Evidence Test Creation Agent for an autonomous network
investigation team.

You are given a FIXED, already-prioritized list of {expected_count}
evidence gaps. Your ONLY job is to write exactly ONE evidence test
per gap, in the SAME ORDER the gaps are given.

Important rules:

1. Produce EXACTLY {expected_count} tests -- one per gap, no more,
   no fewer. Do NOT add tests for gaps not listed. Do NOT split one
   gap into multiple tests.
2. Preserve gap order: the Nth test you return must correspond to
   the Nth gap in the supplied list.
3. Prefer each gap's suggested_capability. Only use a different
   capability (still from available_capabilities) if the suggested
   one is clearly wrong for that gap.
4. Do NOT decide whether a hypothesis is correct.
5. Parameters must contain only values the chosen capability needs.
6. Use only entity IDs present in the supplied context.
7. Do not invent capability names.

For topology and reachability tests, parameters may include:
{{
    "source_entity_id": str,
    "target_entity_id": str
}}

For network_state tests, parameters may include:
{{
    "entity_ids": list[str]
}}

Return ONLY a Python dictionary with exactly this shape:

{{
    "tests": [
        {{
            "hypothesis_id": str,
            "objective": str,
            "capability": str,
            "parameters": dict,
            "expected_supporting_observations": list[str],
            "expected_falsifying_observations": list[str]
        }}
    ]
}}
"""

    user_prompt = f"""
Investigation context:

{context}

Write exactly {expected_count} evidence tests, one per gap, in order.
"""

    response = llm_call(
        agent_name=agent_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    print("Evidence test creation LLM call completed")

    content = response["content"]
    if not content:
        raise ValueError("Evidence Test Creation Agent returned empty LLM content.")

    parsed = parse_or_repair_agent_response(
        raw_output=content,
        expected_schema=EVIDENCE_TEST_CREATION_SCHEMA,
        agent_name=agent_name,
    )

    raw_tests = parsed.get("tests", [])

    if len(raw_tests) != expected_count:
        print(
            f"⚠ evidence_test_creation_agent returned {len(raw_tests)} "
            f"tests for {expected_count} gaps -- matching by position "
            "up to the shorter length."
        )

    tests = []
    for index, (gap, item) in enumerate(zip(evidence_gap_selection.gaps, raw_tests), start=1):
        if item.get("hypothesis_id") and item["hypothesis_id"] != gap.hypothesis_id:
            print(
                f"⚠ evidence_test_creation_agent test #{index}'s "
                f"hypothesis_id ('{item['hypothesis_id']}') doesn't "
                f"match its gap's ('{gap.hypothesis_id}') -- using the "
                "gap's, since position, not the model's echo, is the "
                "authoritative link between a gap and its test."
            )

        test = EvidenceTest(
            test_id=f"{incident_frame.incident_id}-r{round_number}-test-{index}",
            hypothesis_id=gap.hypothesis_id,
            objective=item.get("objective", gap.gap_description),
            capability=item.get("capability", gap.suggested_capability),
            parameters=item.get("parameters", {}),
            expected_supporting_observations=item.get("expected_supporting_observations", []),
            expected_falsifying_observations=item.get("expected_falsifying_observations", []),
            priority=gap.priority,
        )
        test.validate()
        tests.append(test)

    if not tests:
        raise ValueError(
            "Evidence Test Creation Agent: no valid tests could be "
            "constructed from the supplied gaps."
        )

    evidence_plan = EvidencePlan(incident_id=incident_frame.incident_id, tests=tests)
    evidence_plan.validate()
    return evidence_plan
