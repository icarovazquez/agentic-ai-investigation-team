"""
The investigation orchestrator: wires every stage together into one
runnable pipeline, including the reasoning loop (plan -> execute ->
analyze -> challenge, repeating until a hypothesis both clears the
confidence bar and survives the Challenger, or the round limit is
hit) and the final Root Cause & Remediation stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langfuse import observe

from . import capabilities  # noqa: F401 -- import triggers capability registration
from .adapters.base import load_investigation_evidence_executor
from .agents.evidence_analyst import check_evidence_analysis_leakage, evidence_analyst_agent, EvidenceAnalysis
from .agents.evidence_planner import evidence_planning_agent
from .agents.hypothesis_challenger import ChallengeReport, hypothesis_challenger_agent
from .agents.hypothesis_generator import HypothesisSet, hypothesis_generator_agent
from .agents.incident_framing import IncidentFrame, incident_framing_agent
from .agents.root_cause_remediation import RootCauseRecommendation, root_cause_remediation_agent, select_leading_hypothesis
from .capabilities.deep_diagnostics import DeepDiagnosticsCapability
from .capabilities.network_state import NetworkStateCapability
from .capabilities.reachability import ReachabilityCapability
from .capabilities.topology import TopologyCapability
from .capabilities.vendor_alert import VendorAlertCapability
from .config import NetworkInvestigationConfig
from .domain import EvidencePlan, EvidenceTestResult, InvestigationStatus
from .evidence_graph import EvidenceGraph
from .executors import execute_evidence_plan


@dataclass
class InvestigationRunResult:
    """
    Complete output of one end-to-end investigation run.

    This bundles every intermediate artifact so failures can be
    traced back to the stage that produced them, rather than only
    seeing the final assessments.
    """

    investigation_id: str
    status: InvestigationStatus

    evidence_graph_summary: Dict[str, int]
    incident_frame: Optional[IncidentFrame] = None
    hypothesis_set: Optional[HypothesisSet] = None
    evidence_plan: Optional[EvidencePlan] = None
    evidence_results: List[EvidenceTestResult] = field(default_factory=list)
    evidence_analysis: Optional[EvidenceAnalysis] = None
    challenge_report: Optional[ChallengeReport] = None
    root_cause_recommendation: Optional[RootCauseRecommendation] = None

    failed_stage: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "status": self.status.value,
            "evidence_graph_summary": self.evidence_graph_summary,
            "incident_frame": (
                self.incident_frame.to_dict()
                if self.incident_frame else None
            ),
            "hypothesis_set": (
                self.hypothesis_set.to_dict()
                if self.hypothesis_set else None
            ),
            "evidence_plan": (
                self.evidence_plan.to_dict()
                if self.evidence_plan else None
            ),
            "evidence_results": [
                r.to_dict() for r in self.evidence_results
            ],
            "evidence_analysis": (
                self.evidence_analysis.to_dict()
                if self.evidence_analysis else None
            ),
            "challenge_report": (
                self.challenge_report.to_dict()
                if self.challenge_report else None
            ),
            "root_cause_recommendation": (
                self.root_cause_recommendation.to_dict()
                if self.root_cause_recommendation else None
            ),
            "failed_stage": self.failed_stage,
            "error": self.error,
        }


@observe(name="run_investigation")
def run_investigation(
    investigation_config: NetworkInvestigationConfig,
) -> InvestigationRunResult:
    """
    Run one full investigation:

      load evidence
        -> frame incident
        -> generate hypotheses
        -> [plan -> execute -> analyze -> challenge]  (looped)
        -> determine root cause and recommend remediation

    Any stage failure is caught and reported with the stage name,
    rather than letting one exception obscure where the pipeline
    broke.
    """

    investigation_id = investigation_config.investigation_id
    evidence_graph = EvidenceGraph()

    # ------------------------------------------------------
    # Stage 0: Load evidence
    # ------------------------------------------------------

    try:
        load_result = load_investigation_evidence_executor(
            investigation_config=investigation_config,
            evidence_graph=evidence_graph,
        )
        evidence_graph = load_result["evidence_graph"]

    except Exception as exc:
        return InvestigationRunResult(
            investigation_id=investigation_id,
            status=InvestigationStatus.CREATED,
            evidence_graph_summary=evidence_graph.summary(),
            failed_stage="load_evidence",
            error=str(exc),
        )

    print(f"✓ Evidence loaded: {evidence_graph.summary()}")

    topology_capability = TopologyCapability()
    reachability_capability = ReachabilityCapability()
    network_state_capability = NetworkStateCapability()
    vendor_alert_capability = VendorAlertCapability()
    deep_diagnostics_capability = DeepDiagnosticsCapability()

    # ------------------------------------------------------
    # Stage 1: Frame the incident
    # ------------------------------------------------------

    try:
        incident_frame = incident_framing_agent(
            investigation_config=investigation_config,
            evidence_graph=evidence_graph,
            topology_capability=topology_capability,
        )
        incident_frame.validate()

    except Exception as exc:
        return InvestigationRunResult(
            investigation_id=investigation_id,
            status=InvestigationStatus.IN_PROGRESS,
            evidence_graph_summary=evidence_graph.summary(),
            failed_stage="incident_framing",
            error=str(exc),
        )

    print(f"✓ Incident framed: {incident_frame.investigation_question}")

    # ------------------------------------------------------
    # Stage 2: Generate hypotheses
    # ------------------------------------------------------

    try:
        hypothesis_set = hypothesis_generator_agent(
            incident_frame=incident_frame,
            evidence_graph=evidence_graph,
            topology_capability=topology_capability,
        )
        hypothesis_set.validate()

    except Exception as exc:
        return InvestigationRunResult(
            investigation_id=investigation_id,
            status=InvestigationStatus.IN_PROGRESS,
            evidence_graph_summary=evidence_graph.summary(),
            incident_frame=incident_frame,
            failed_stage="hypothesis_generation",
            error=str(exc),
        )

    print(f"✓ {len(hypothesis_set.hypotheses)} hypotheses generated")

    # ------------------------------------------------------
    # Stages 3-6: Plan, execute, analyze, and challenge evidence,
    # looping until a hypothesis both clears the confidence bar
    # AND survives the Challenger, or until max_iterations runs out.
    # ------------------------------------------------------

    policy = investigation_config.investigation_policy
    max_iterations = policy.maximum_reasoning_iterations

    evidence_plan = None
    evidence_results: List[EvidenceTestResult] = []
    evidence_analysis = None
    challenge_report = None
    round_number = 0

    for round_number in range(1, max_iterations + 1):

        # ------------------------------------------------------
        # Stage 3: Plan evidence tests
        # ------------------------------------------------------

        try:
            evidence_plan = evidence_planning_agent(
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                round_number=round_number,
                prior_evidence_analysis=evidence_analysis,  # None on round 1
                prior_challenge_report=challenge_report,    # None on round 1
            )
            evidence_plan.validate()

        except Exception as exc:
            return InvestigationRunResult(
                investigation_id=investigation_id,
                status=InvestigationStatus.IN_PROGRESS,
                evidence_graph_summary=evidence_graph.summary(),
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_results=evidence_results,
                failed_stage=f"evidence_planning_round_{round_number}",
                error=str(exc),
            )

        print(f"✓ Round {round_number}: {len(evidence_plan.tests)} evidence tests planned")

        # ------------------------------------------------------
        # Stage 4: Execute evidence tests (deterministic)
        # ------------------------------------------------------

        try:
            round_results = execute_evidence_plan(
                evidence_plan=evidence_plan,
                evidence_graph=evidence_graph,
            )

        except Exception as exc:
            return InvestigationRunResult(
                investigation_id=investigation_id,
                status=InvestigationStatus.WAITING_FOR_EVIDENCE,
                evidence_graph_summary=evidence_graph.summary(),
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_plan=evidence_plan,
                evidence_results=evidence_results,
                failed_stage=f"evidence_execution_round_{round_number}",
                error=str(exc),
            )

        evidence_results = evidence_results + round_results

        print(f"✓ Round {round_number}: {len(round_results)} evidence tests executed")

        # ------------------------------------------------------
        # Stage 5: Analyze evidence against hypotheses
        # ------------------------------------------------------

        try:
            evidence_analysis = evidence_analyst_agent(
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_plan=evidence_plan,
                evidence_results=evidence_results,  # cumulative across rounds
            )
            evidence_analysis.validate()

        except Exception as exc:
            return InvestigationRunResult(
                investigation_id=investigation_id,
                status=InvestigationStatus.WAITING_FOR_EVIDENCE,
                evidence_graph_summary=evidence_graph.summary(),
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_plan=evidence_plan,
                evidence_results=evidence_results,
                failed_stage=f"evidence_analysis_round_{round_number}",
                error=str(exc),
            )

        print(f"✓ Round {round_number}: evidence analysis complete")
        for a in evidence_analysis.assessments:
            print(f"    {a.hypothesis_id}: {a.status.value} (confidence={a.confidence})")

        leakage_violations = check_evidence_analysis_leakage(evidence_analysis)
        if leakage_violations:
            print(f"⚠ Round {round_number}: Evidence Analyst contract violations detected:")
            for v in leakage_violations:
                print(f"    {v}")

        leading_assessment = select_leading_hypothesis(
            evidence_analysis=evidence_analysis,
            minimum_confidence=policy.minimum_diagnosis_confidence,
        )

        # If review is disabled, use the old, simpler exit condition and
        # skip the Challenger entirely.
        if not policy.require_challenger_review:
            if leading_assessment is not None:
                break

            if round_number < max_iterations:
                print(f"⚠ Round {round_number}: no hypothesis cleared the confidence bar — planning another round")

            continue

        # ------------------------------------------------------
        # Stage 6: Challenge the leading hypothesis
        # ------------------------------------------------------

        try:
            challenge_report = hypothesis_challenger_agent(
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_analysis=evidence_analysis,
            )
            challenge_report.validate()

        except Exception as exc:
            return InvestigationRunResult(
                investigation_id=investigation_id,
                status=InvestigationStatus.IN_PROGRESS,
                evidence_graph_summary=evidence_graph.summary(),
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_plan=evidence_plan,
                evidence_results=evidence_results,
                evidence_analysis=evidence_analysis,
                failed_stage=f"hypothesis_challenge_round_{round_number}",
                error=str(exc),
            )

        print(f"✓ Round {round_number}: hypothesis challenge complete")
        for c in challenge_report.challenges:
            print(f"    {c.hypothesis_id}: {c.challenge_outcome}")
        if challenge_report.systemic_concerns:
            print(f"    ⚠ {len(challenge_report.systemic_concerns)} systemic concern(s) raised")

        # Exit only if something cleared the confidence bar AND the
        # Challenger upheld it. Disputed or unresolved keeps looping,
        # using this round's challenger gaps to steer round N+1's plan.
        if leading_assessment is not None:
            leading_challenge = next(
                (
                    c for c in challenge_report.challenges
                    if c.hypothesis_id == leading_assessment.hypothesis_id
                ),
                None,
            )

            if leading_challenge is not None and leading_challenge.challenge_outcome == "upheld":
                break

        if round_number < max_iterations:
            print(f"⚠ Round {round_number}: leading hypothesis not confirmed — planning another round")

    # ------------------------------------------------------
    # Stage 7: Determine root cause and recommend remediation
    # ------------------------------------------------------

    if (
        investigation_config.investigation_policy.require_challenger_review
        and challenge_report is None
    ):
        return InvestigationRunResult(
            investigation_id=investigation_id,
            status=InvestigationStatus.WAITING_FOR_EVIDENCE,
            evidence_graph_summary=evidence_graph.summary(),
            incident_frame=incident_frame,
            hypothesis_set=hypothesis_set,
            evidence_plan=evidence_plan,
            evidence_results=evidence_results,
            evidence_analysis=evidence_analysis,
            challenge_report=challenge_report,
            failed_stage="root_cause_remediation",
            error=(
                "InvestigationPolicy.require_challenger_review is True "
                "but no challenge_report is available."
            ),
        )

    try:
        root_cause_recommendation = root_cause_remediation_agent(
            incident_frame=incident_frame,
            hypothesis_set=hypothesis_set,
            evidence_analysis=evidence_analysis,
            challenge_report=challenge_report,
            investigation_policy=investigation_config.investigation_policy,
        )

    except Exception as exc:
        return InvestigationRunResult(
            investigation_id=investigation_id,
            status=InvestigationStatus.IN_PROGRESS,
            evidence_graph_summary=evidence_graph.summary(),
            incident_frame=incident_frame,
            hypothesis_set=hypothesis_set,
            evidence_plan=evidence_plan,
            evidence_results=evidence_results,
            evidence_analysis=evidence_analysis,
            challenge_report=challenge_report,
            failed_stage="root_cause_remediation",
            error=str(exc),
        )

    if root_cause_recommendation.status == "insufficient_evidence":
        print("⚠ No hypothesis met minimum_diagnosis_confidence — investigation inconclusive")
        final_status = InvestigationStatus.INCONCLUSIVE
    else:
        print(
            f"✓ Root cause identified: {root_cause_recommendation.hypothesis_id} "
            f"(confidence={root_cause_recommendation.confidence})"
        )
        print(
            f"    {len(root_cause_recommendation.remediation_steps)} "
            "remediation step(s) recommended"
        )
        final_status = InvestigationStatus.REMEDIATION_RECOMMENDED

    return InvestigationRunResult(
        investigation_id=investigation_id,
        status=final_status,
        evidence_graph_summary=evidence_graph.summary(),
        incident_frame=incident_frame,
        hypothesis_set=hypothesis_set,
        evidence_plan=evidence_plan,
        evidence_results=evidence_results,
        evidence_analysis=evidence_analysis,
        challenge_report=challenge_report,
        root_cause_recommendation=root_cause_recommendation,
    )
