"""
Persisting investigation results and summarizing LLM usage/cost.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import NetworkInvestigationConfig
from .orchestrator import InvestigationRunResult


def save_investigation_result(
    run_result: InvestigationRunResult,
    investigation_config: NetworkInvestigationConfig,
) -> Path:
    """
    Persist a completed investigation run to
    investigation_config.report_path as JSON.

    Creates the investigation's output directory if it doesn't
    exist yet. Overwrites any existing report for this
    investigation_id — callers who want history across multiple
    runs of the same investigation should use history_path instead
    of relying on this file alone (not yet implemented).
    """

    output_dir = investigation_config.investigation_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = investigation_config.report_path

    with open(report_path, "w") as f:
        json.dump(run_result.to_dict(), f, indent=2, default=str)

    print(f"✓ Investigation result saved to {report_path}")

    return report_path


def print_run_summary(run_result: InvestigationRunResult) -> None:
    print(f"Status: {run_result.status.value}")
    print(f"Failed stage: {run_result.failed_stage}")
    print(f"Error: {run_result.error}")
    print()

    if run_result.evidence_analysis:
        print("Final evidence_analysis:")
        for a in run_result.evidence_analysis.assessments:
            print(f"    {a.hypothesis_id}: {a.status.value} (confidence={a.confidence})")
        print()

    if run_result.challenge_report:
        print("Final challenge_report:")
        for c in run_result.challenge_report.challenges:
            print(f"    {c.hypothesis_id}: {c.challenge_outcome}")
        print()

    print("Rounds seen in evidence_results (by test_id prefix):")
    rounds_seen = sorted({
        r.test_id.split("-test-")[0].split("-r")[-1]
        for r in run_result.evidence_results
    })
    print(f"    {rounds_seen}")
    print()

    capabilities_used = sorted({r.capability for r in run_result.evidence_results})
    print(f"Capabilities used across all rounds: {capabilities_used}")
    print()

    if run_result.root_cause_recommendation:
        rec = run_result.root_cause_recommendation
        print(f"Root cause recommendation: {rec.hypothesis_id} (confidence={rec.confidence})")
        print(f"Status: {rec.status}")


# ============================================================
# EFFICIENCY METRICS
# ============================================================


@dataclass
class EfficiencyMetrics:
    """
    Summary of LLM usage for one investigation run.

    Computed from a slice of the global LLM_USAGE list (see
    investigation_team.llm.LLM_USAGE) — the caller is responsible
    for capturing the right slice, since LLM_USAGE has no
    investigation_id field to scope by itself:

        from investigation_team.llm import LLM_USAGE
        from investigation_team.orchestrator import run_investigation
        from investigation_team.persistence import compute_efficiency_metrics

        usage_start = len(LLM_USAGE)
        run_result = run_investigation(investigation_config=my_config)
        usage_end = len(LLM_USAGE)

        efficiency = compute_efficiency_metrics(LLM_USAGE[usage_start:usage_end])
    """

    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    calls_by_agent: Dict[str, int] = field(default_factory=dict)
    tokens_by_agent: Dict[str, int] = field(default_factory=dict)

    repair_call_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_efficiency_metrics(
    llm_usage_slice: List[Dict[str, Any]],
) -> EfficiencyMetrics:
    """
    Summarize a slice of LLM_USAGE into per-investigation metrics.

    repair_call_count counts entries whose agent name ends in
    "_format_repair" — repair_agent_dict_response tags its own
    llm_call with that suffix rather than reusing the parent
    agent's name, so repair calls are directly identifiable rather
    than inferred from call-count anomalies.
    """

    total_calls = len(llm_usage_slice)
    total_input_tokens = sum(e["input_tokens"] for e in llm_usage_slice)
    total_output_tokens = sum(e["output_tokens"] for e in llm_usage_slice)
    total_tokens = sum(e["total_tokens"] for e in llm_usage_slice)

    calls_by_agent: Dict[str, int] = {}
    tokens_by_agent: Dict[str, int] = {}

    for entry in llm_usage_slice:
        agent = entry["agent"]
        calls_by_agent[agent] = calls_by_agent.get(agent, 0) + 1
        tokens_by_agent[agent] = (
            tokens_by_agent.get(agent, 0) + entry["total_tokens"]
        )

    repair_call_count = sum(
        1 for entry in llm_usage_slice
        if entry["agent"].endswith("_format_repair")
    )

    return EfficiencyMetrics(
        total_calls=total_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        calls_by_agent=calls_by_agent,
        tokens_by_agent=tokens_by_agent,
        repair_call_count=repair_call_count,
    )
