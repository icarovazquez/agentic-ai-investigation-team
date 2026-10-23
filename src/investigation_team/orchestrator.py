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
from .agents.causal_resolution import CausalResolution, causal_resolution_agent, chain_candidates
from .agents.evidence_planner import evidence_planning_agent
from .agents.evidence_test_creator import evidence_test_creation_agent
from .agents.hypothesis_challenger import ChallengeReport, describe_challenge_outcome, hypothesis_challenger_agent
from .agents.hypothesis_generator import HypothesisSet, hypothesis_generator_agent
from .agents.incident_framing import IncidentFrame, incident_framing_agent
from .agents.root_cause_remediation import RootCauseRecommendation, root_cause_remediation_agent, select_leading_hypothesis
from .capabilities.deep_diagnostics import DeepDiagnosticsCapability
from .capabilities.network_state import NetworkStateCapability
from .capabilities.reachability import ReachabilityCapability
from .capabilities.topology import TopologyCapability
from .capabilities.vendor_alert import VendorAlertCapability
from .config import NetworkInvestigationConfig
from .convergence import check_convergence, residual_concerns, resolved_by_causal_resolution
from .domain import EvidencePlan, EvidenceTestResult, InvestigationStatus
from .evidence_graph import EvidenceGraph
from .evidence_progress import evidence_ids
from .executors import execute_evidence_plan


def describe_exception(exc: BaseException) -> str:
    """
    The exception and its causes, outermost first. Loaders wrap failures
    ("Required evidence source failed: X") and the wrapped message is the
    one that says what went wrong, so a bare str(exc) hides the answer.
    """

    parts = []
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current) or type(current).__name__
        parts.append(f"{type(current).__name__}: {text}" if parts else text)
        current = current.__cause__ or current.__context__
    return " <- caused by ".join(parts)


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

    # Set when the loop stopped because the same leader held for several
    # rounds, not because the Challenger upheld it. The objections that
    # were still open at that point are kept here, not discarded.
    convergence_note: Optional[str] = None
    residual_concerns: List[str] = field(default_factory=list)

    # Set when the loop stopped because a round returned no evidence the
    # investigation did not already have.
    exhaustion_note: Optional[str] = None

    # Challenger disputes that causal resolution settled, with its rationale.
    resolved_concerns: List[str] = field(default_factory=list)

    # How the supported hypotheses relate (rivals or causal chain).
    causal_resolution: Optional[CausalResolution] = None

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
            "convergence_note": self.convergence_note,
            "residual_concerns": self.residual_concerns,
            "exhaustion_note": self.exhaustion_note,
            "resolved_concerns": self.resolved_concerns,
            "causal_resolution": (
                self.causal_resolution.to_dict()
                if self.causal_resolution else None
            ),
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
            error=describe_exception(exc),
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
            error=describe_exception(exc),
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
            error=describe_exception(exc),
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
    all_tests = []  # every test from every round, for the analyst
    analysis_history = []
    convergence_note = None
    exhaustion_note = None
    seen_evidence_ids = set()
    open_concerns = []
    evidence_results: List[EvidenceTestResult] = []
    evidence_analysis = None
    challenge_report = None
    round_number = 0
    leading_assessment = None

    for round_number in range(1, max_iterations + 1):

        # ------------------------------------------------------
        # Stage 3a: Select evidence gaps
        # ------------------------------------------------------

        try:
            evidence_gap_selection = evidence_planning_agent(
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                prior_evidence_analysis=evidence_analysis,  # None on round 1
                prior_challenge_report=challenge_report,    # None on round 1
            )
            evidence_gap_selection.validate()

        except Exception as exc:
            return InvestigationRunResult(
                investigation_id=investigation_id,
                status=InvestigationStatus.IN_PROGRESS,
                evidence_graph_summary=evidence_graph.summary(),
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_results=evidence_results,
                failed_stage=f"evidence_gap_selection_round_{round_number}",
                error=describe_exception(exc),
            )

        print(f"✓ Round {round_number}: {len(evidence_gap_selection.gaps)} evidence gaps selected")

        # ------------------------------------------------------
        # Stage 3b: Construct evidence tests from the selected gaps
        # ------------------------------------------------------

        try:
            evidence_plan = evidence_test_creation_agent(
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_gap_selection=evidence_gap_selection,
                round_number=round_number,
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
                failed_stage=f"evidence_test_creation_round_{round_number}",
                error=describe_exception(exc),
            )

        all_tests.extend(evidence_plan.tests)
        print(f"✓ Round {round_number}: {len(evidence_plan.tests)} evidence tests created")

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
                error=describe_exception(exc),
            )

        evidence_results = evidence_results + round_results

        failed_results = [r for r in round_results if r.status != "completed"]
        print(f"✓ Round {round_number}: {len(round_results)} evidence tests executed")
        if failed_results:
            print(f"⚠ Round {round_number}: {len(failed_results)} test(s) FAILED to execute:")
            for r in failed_results:
                print(f"    {r.test_id} [{r.capability}]: {r.error}")

        # Evidence exhaustion: if this round surfaced nothing the
        # investigation did not already hold, re-analysing it repeats the
        # last answer and another round would plan the same tests again.
        # The previous round's analysis and challenge stand.
        round_ids = evidence_ids(round_results)
        new_ids = round_ids - seen_evidence_ids
        seen_evidence_ids |= round_ids
        if round_number > 1 and not new_ids and evidence_analysis is not None:
            exhaustion_note = (
                f"Stopped in round {round_number}: the tests returned no evidence "
                "beyond what earlier rounds already collected. The remaining "
                "objections need data none of the available sources provide."
            )
            print(f"✓ Round {round_number}: {exhaustion_note}")
            if challenge_report is not None and leading_assessment is not None:
                open_concerns = residual_concerns(
                    challenge_report, leading_assessment.hypothesis_id
                )
            break

        # ------------------------------------------------------
        # Stage 5: Analyze evidence against hypotheses
        # ------------------------------------------------------

        try:
            evidence_analysis = evidence_analyst_agent(
                incident_frame=incident_frame,
                hypothesis_set=hypothesis_set,
                evidence_plan=EvidencePlan(incident_id=incident_frame.incident_id, tests=list(all_tests)),
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
                error=describe_exception(exc),
            )

        analysis_history.append(evidence_analysis)
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
                error=describe_exception(exc),
            )

        print(f"✓ Round {round_number}: hypothesis challenge complete")
        for c in challenge_report.challenges:
            print(f"    {c.hypothesis_id}: {describe_challenge_outcome(c)}")
            if c.challenge_outcome == "disputed":
                print(f"        rebuttal   : {c.rebuttal[:400]}")
                if c.alternative_explanation:
                    print(f"        alternative: {c.alternative_explanation[:300]}")
                print(
                    f"        to resolve : {c.distinguishing_observation} "
                    f"(via {c.distinguishing_capability}: {c.distinguishing_observable})"
                )
            elif c.downgrade_note:
                print(f"        {c.downgrade_note}")
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
                if leading_challenge.downgrade_note:
                    open_concerns = residual_concerns(
                        challenge_report, leading_assessment.hypothesis_id
                    )
                break

        converged_id = check_convergence(
            analysis_history,
            minimum_confidence=policy.minimum_diagnosis_confidence,
            stable_rounds=policy.convergence_stable_rounds,
        )
        if converged_id is not None:
            open_concerns = residual_concerns(challenge_report, converged_id)
            convergence_note = (
                f"Converged in round {round_number}: {converged_id} was the supported "
                f"leader for {policy.convergence_stable_rounds} consecutive rounds with no "
                f"credible rival. {len(open_concerns)} objection(s) remained open "
                "and are listed in residual_concerns."
            )
            print(f"✓ Round {round_number}: {convergence_note}")
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

    # ------------------------------------------------------
    # Stage 6b: Causal resolution. When more than one hypothesis is
    # supported, decide whether they are rivals or a causal chain, so the
    # root cause is not picked by a confidence gap of a few hundredths.
    # ------------------------------------------------------

    resolution = None
    final_leader = select_leading_hypothesis(
        evidence_analysis=evidence_analysis,
        minimum_confidence=policy.minimum_diagnosis_confidence,
    )
    if final_leader is not None:
        candidates = chain_candidates(
            evidence_analysis, policy.contributing_cause_minimum_confidence
        )
        if len(candidates) >= 2:
            try:
                resolution = causal_resolution_agent(
                    incident_frame=incident_frame,
                    hypothesis_set=hypothesis_set,
                    evidence_analysis=evidence_analysis,
                    candidates=candidates,
                    default_root_id=final_leader.hypothesis_id,
                )
            except Exception as exc:
                print(
                    "⚠ Causal resolution failed; using the highest-confidence "
                    f"hypothesis as the root cause: {describe_exception(exc)}"
                )
            if resolution is not None:
                print(
                    f"✓ Causal resolution: {resolution.relationship}; root cause "
                    f"{resolution.root_cause_hypothesis_id}"
                    + (
                        f", contributing {', '.join(resolution.contributing_hypothesis_ids)}"
                        if resolution.contributing_hypothesis_ids else ""
                    )
                )
                for note in resolution.notes:
                    print(f"    ℹ {note}")

    resolved_concerns: List[str] = []
    if (
        resolution is not None
        and resolution.relationship == "causal_chain"
        and challenge_report is not None
    ):
        resolved_concerns = resolved_by_causal_resolution(challenge_report, resolution)
        open_concerns = residual_concerns(
            challenge_report,
            resolution.root_cause_hypothesis_id,
            extra_ids=resolution.contributing_hypothesis_ids,
            after_causal_chain=True,
        )

    try:
        root_cause_recommendation = root_cause_remediation_agent(
            incident_frame=incident_frame,
            hypothesis_set=hypothesis_set,
            evidence_analysis=evidence_analysis,
            challenge_report=challenge_report,
            investigation_policy=investigation_config.investigation_policy,
            root_cause_hypothesis_id=(
                resolution.root_cause_hypothesis_id
                if resolution and resolution.relationship == "causal_chain" else None
            ),
            contributing_hypothesis_ids=(
                resolution.contributing_hypothesis_ids if resolution else None
            ),
            causal_rationale=(resolution.rationale if resolution else None),
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
            error=describe_exception(exc),
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
        convergence_note=convergence_note,
        residual_concerns=open_concerns,
        exhaustion_note=exhaustion_note,
        resolved_concerns=resolved_concerns,
        causal_resolution=resolution,
    )
