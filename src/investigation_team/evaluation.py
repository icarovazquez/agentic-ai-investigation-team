"""
Ground-truth evaluation.

This module sits ABOVE config.py and orchestrator.py/agents.* in the
dependency graph deliberately. In the original notebook,
EvaluationResult/compute_evaluation lived in the same cell block as
NetworkInvestigationConfig but depended on InvestigationRunResult and
rank_hypothesis_assessments — both defined much later in the
notebook. That only worked because notebook cells don't enforce
import order at definition time, only at call time. A real package
can't do that: config.py now stays a pure leaf, and this evaluation
logic lives here instead, free to depend on everything below it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .agents.hypothesis_challenger import rank_hypothesis_assessments
from .config import EvaluationConfig
from .orchestrator import InvestigationRunResult


# ------------------------------------------------------------
# Fault-type matching
#
# Ground truth is an identifier ("link_failure", "link_down"); the
# Hypothesis Generator words the same fault differently from run to
# run ("Physical Layer Link Failure", "interface down", ...). Plain
# substring matching on the identifier therefore scored identical
# diagnoses differently between runs. Phrases are normalized
# (case, underscores, hyphens) and a ground-truth label that belongs
# to a known fault group matches any phrase in that group.
#
# Phrases are deliberately specific to the fault ("physical layer
# failure", not "physical layer"), because hypotheses often mention
# the same words while ruling the fault out ("regardless of physical
# layer status", "the physical link remains operational").
# ------------------------------------------------------------

FAULT_SYNONYM_GROUPS: List[Dict[str, List[str]]] = [
    {
        "labels": [
            "link failure",
            "link down",
            "link outage",
            "link fault",
            "physical link failure",
        ],
        "phrases": [
            "link failure",
            "link down",
            "link outage",
            "link fault",
            "physical link failure",
            "physical layer failure",
            "physical layer fault",
            "physical layer link",
            "physical layer outage",
            "interface down",
            "interfaces down",
            "cable disconnect",
            "cable failure",
            "cable fault",
            "transceiver failure",
            "port failure",
            "hardware port",
            "signal loss",
            "fiber cut",
        ],
    },
    {
        "labels": [
            "qos policy detached",
            "qos policy missing",
            "qos policy not applied",
            "qos misconfiguration",
        ],
        "phrases": [
            "qos policy detached",
            "qos policy missing",
            "qos policy not applied",
            "qos policy not attached",
            "qos policy was removed",
            "qos misconfiguration",
            "policy detached",
            "policy was detached",
            "policy not attached",
            "policy is not attached",
            "policy removed",
            "policy was removed",
            "policy missing",
            "policy is missing",
            "missing qos",
            "no qos policy",
            "without qos",
            "lack of qos",
            "absence of qos",
            "service policy",
        ],
    },
]


def normalize_label(text: str) -> str:
    """Lowercase; underscores and hyphens become spaces; collapse spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", (text or "").lower())).strip()


def fault_match_phrases(*ground_truth_labels: str) -> List[str]:
    """Phrases that count as describing the ground-truth fault."""
    phrases: List[str] = []
    for raw in ground_truth_labels:
        label = normalize_label(raw)
        if not label:
            continue
        phrases.append(label)
        for group in FAULT_SYNONYM_GROUPS:
            if label in group["labels"]:
                phrases.extend(group["phrases"])
    return list(dict.fromkeys(phrases))


def hypothesis_text_matches_fault(
    hypothesis_text: str,
    ground_truth_fault_type: Optional[str],
    ground_truth_root_cause: Optional[str],
) -> bool:
    haystack = normalize_label(hypothesis_text)
    phrases = fault_match_phrases(
        ground_truth_fault_type or "", ground_truth_root_cause or ""
    )
    return any(phrase in haystack for phrase in phrases)


@dataclass
class EvaluationResult:
    """
    Comparison of an investigation's outcome against benchmark
    ground truth, where available.

    Not all metrics in EvaluationConfig.metrics are computed here.
    localization_accuracy is a deterministic set comparison over
    entity IDs. top_1/top_3_root_cause_accuracy are keyword-match
    heuristics against free-text ground_truth_root_cause / fault
    type, since that field has no structured form to compare
    against exactly — treat these two as approximate signals, not
    ground truth verification. evidence_grounding_score and
    investigation_efficiency are not computed by this function (see
    note in compute_evaluation docstring).
    """

    investigation_id: str
    has_ground_truth: bool

    top_1_root_cause_accuracy: Optional[bool] = None
    top_3_root_cause_accuracy: Optional[bool] = None
    localization_accuracy: Optional[float] = None

    matched_entity_ids: List[str] = field(default_factory=list)
    missed_entity_ids: List[str] = field(default_factory=list)
    unexpected_entity_ids: List[str] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_evaluation(
    run_result: InvestigationRunResult,
    evaluation_config: Optional[EvaluationConfig],
) -> EvaluationResult:
    """
    Score a completed investigation run against benchmark ground
    truth, if present.

    Only computes localization_accuracy (deterministic) and
    top_1/top_3_root_cause_accuracy (keyword-match heuristic against
    ground_truth_fault_type — approximate, since ground_truth_root_cause
    has no structured form). evidence_grounding_score and
    investigation_efficiency from EvaluationConfig.metrics are not
    computed here: grounding would require validating every citation
    in supporting_evidence against evidence_results, and efficiency
    needs LLM_USAGE token/call accounting — both worth building
    separately once there's a reason to trust the numbers, rather
    than stubbing something that looks precise but isn't.
    """

    investigation_id = run_result.investigation_id

    if evaluation_config is None or not evaluation_config.has_ground_truth:
        return EvaluationResult(
            investigation_id=investigation_id,
            has_ground_truth=False,
            notes=["No ground truth configured for this investigation."],
        )

    recommendation = run_result.root_cause_recommendation

    if recommendation is None or recommendation.status != "root_cause_identified":
        return EvaluationResult(
            investigation_id=investigation_id,
            has_ground_truth=True,
            top_1_root_cause_accuracy=False,
            top_3_root_cause_accuracy=False,
            localization_accuracy=0.0,
            notes=[
                "No root cause was identified by the investigation "
                "(insufficient_evidence or earlier stage failure) — "
                "cannot match against ground truth."
            ],
        )

    hypotheses_by_id = {
        h.hypothesis_id: h
        for h in (run_result.hypothesis_set.hypotheses if run_result.hypothesis_set else [])
    }

    winning_hypothesis = hypotheses_by_id.get(recommendation.hypothesis_id)

    # --- localization_accuracy: deterministic entity-ID overlap ---

    ground_truth_ids = set(evaluation_config.ground_truth_entity_ids)
    predicted_ids = set(
        winning_hypothesis.suspected_entity_ids if winning_hypothesis else []
    )

    matched = sorted(ground_truth_ids & predicted_ids)
    missed = sorted(ground_truth_ids - predicted_ids)
    unexpected = sorted(predicted_ids - ground_truth_ids)

    if ground_truth_ids:
        localization_accuracy = len(matched) / len(ground_truth_ids)
    else:
        localization_accuracy = None

    # --- top_1 / top_3 root_cause_accuracy: keyword-match heuristic ---

    def hypothesis_matches_ground_truth(assessment_hypothesis_id: str) -> bool:
        hypothesis = hypotheses_by_id.get(assessment_hypothesis_id)
        if hypothesis is None:
            return False

        return hypothesis_text_matches_fault(
            " ".join([
                hypothesis.title,
                hypothesis.proposed_cause,
                hypothesis.causal_mechanism,
            ]),
            evaluation_config.ground_truth_fault_type,
            evaluation_config.ground_truth_root_cause,
        )

    top_1_match = hypothesis_matches_ground_truth(recommendation.hypothesis_id)

    ranked = (
        rank_hypothesis_assessments(run_result.evidence_analysis)
        if run_result.evidence_analysis else []
    )
    top_3_ids = [a.hypothesis_id for a in ranked[:3]]
    top_3_match = any(hypothesis_matches_ground_truth(hid) for hid in top_3_ids)

    notes = [
        "top_1/top_3_root_cause_accuracy are keyword/synonym-match "
        "heuristics against the ground-truth fault label, not exact "
        "verification.",
    ]

    if not ground_truth_ids:
        notes.append(
            "ground_truth_entity_ids is empty — localization_accuracy "
            "not computed."
        )

    return EvaluationResult(
        investigation_id=investigation_id,
        has_ground_truth=True,
        top_1_root_cause_accuracy=top_1_match,
        top_3_root_cause_accuracy=top_3_match,
        localization_accuracy=localization_accuracy,
        matched_entity_ids=matched,
        missed_entity_ids=missed,
        unexpected_entity_ids=unexpected,
        notes=notes,
    )
