"""
Configuration layer: describes one reproducible investigation.

This module is a leaf — it depends only on domain.py, never on
agents, the orchestrator, or evaluation. In the original notebook,
EvaluationResult/compute_evaluation lived in this same cell block but
actually depended on InvestigationRunResult (orchestrator.py) and
rank_hypothesis_assessments (agents/hypothesis_challenger.py) — a
real circular dependency that only worked because notebook cells
don't enforce import order. That logic now lives in its own
evaluation.py, which sits above orchestrator.py and agents.*, so this
module stays a true leaf.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .domain import AccessMode, EvidenceType, RemediationMode, SourceFormat


@dataclass
class IncidentSeed:
    """
    Initial incident information supplied to the investigation.

    This is not the evolving IncidentRecord. It is the reproducible
    starting point used by the Incident Framing Agent.
    """

    reported_symptom: str
    incident_start_time: str

    affected_service: Optional[str] = None
    reported_locations: List[str] = field(default_factory=list)
    reported_entities: List[str] = field(default_factory=list)

    source: str = "benchmark"
    initial_severity: Optional[str] = None
    initial_context: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.reported_symptom.strip():
            raise ValueError("reported_symptom cannot be empty.")

        if not self.incident_start_time.strip():
            raise ValueError("incident_start_time cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSourceConfig:
    """
    Configuration for one raw evidence source.

    The adapter_name identifies the translator that converts the raw
    source into vendor-neutral Evidence Graph domain records.
    """

    source_id: str
    evidence_type: EvidenceType
    source_format: SourceFormat
    access_mode: AccessMode
    adapter_name: str

    path: Optional[str] = None
    url: Optional[str] = None

    connection_parameters: Dict[str, Any] = field(
        default_factory=dict
    )
    query_parameters: Dict[str, Any] = field(
        default_factory=dict
    )

    timestamp_column: Optional[str] = None
    entity_id_columns: List[str] = field(
        default_factory=list
    )
    schema_mapping: Dict[str, str] = field(
        default_factory=dict
    )

    required: bool = True
    enabled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty.")

        if not self.adapter_name.strip():
            raise ValueError(
                f"Source '{self.source_id}' requires adapter_name."
            )

        if (
            self.access_mode == AccessMode.FILE
            and not self.path
        ):
            raise ValueError(
                f"File source '{self.source_id}' requires a path."
            )

        if (
            self.access_mode == AccessMode.API
            and not self.url
            and not self.connection_parameters
        ):
            raise ValueError(
                f"API source '{self.source_id}' requires a URL "
                "or connection parameters."
            )

        if self.access_mode in {
            AccessMode.DATABASE,
            AccessMode.COMMAND,
            AccessMode.LIVE_LAB,
        } and not self.connection_parameters:
            raise ValueError(
                f"Source '{self.source_id}' requires "
                "connection_parameters."
            )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)

        result["evidence_type"] = self.evidence_type.value
        result["source_format"] = self.source_format.value
        result["access_mode"] = self.access_mode.value

        return result


@dataclass
class InvestigationPolicy:
    """
    Guardrails and stopping conditions for the investigation workflow.
    """

    remediation_mode: RemediationMode = (
        RemediationMode.RECOMMEND_ONLY
    )

    require_human_approval: bool = True
    minimum_diagnosis_confidence: float = 0.80

    maximum_reasoning_iterations: int = 5
    maximum_hypotheses: int = 5
    maximum_tests_per_hypothesis: int = 5

    require_falsifiable_hypotheses: bool = True
    require_challenger_review: bool = True
    require_evidence_references: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.minimum_diagnosis_confidence <= 1.0:
            raise ValueError(
                "minimum_diagnosis_confidence must be "
                "between 0 and 1."
            )

        if self.maximum_reasoning_iterations < 1:
            raise ValueError(
                "maximum_reasoning_iterations must be at least 1."
            )

        if self.maximum_hypotheses < 2:
            raise ValueError(
                "maximum_hypotheses must be at least 2."
            )

        if self.maximum_tests_per_hypothesis < 1:
            raise ValueError(
                "maximum_tests_per_hypothesis must be at least 1."
            )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["remediation_mode"] = self.remediation_mode.value
        return result


@dataclass
class EvaluationConfig:
    """
    Optional benchmark ground truth and evaluation metrics.

    Production incidents may not have ground truth, so this object
    is optional.
    """

    ground_truth_root_cause: Optional[str] = None
    ground_truth_entity_ids: List[str] = field(
        default_factory=list
    )
    ground_truth_fault_type: Optional[str] = None

    metrics: List[str] = field(
        default_factory=lambda: [
            "top_1_root_cause_accuracy",
            "top_3_root_cause_accuracy",
            "localization_accuracy",
            "evidence_grounding_score",
            "investigation_efficiency",
        ]
    )

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth_root_cause is not None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkInvestigationConfig:
    """
    Top-level configuration for one reproducible network
    investigation.
    """

    investigation_id: str
    investigation_name: str
    environment_name: str
    investigation_type: str

    scenario_provider: str
    scenario_id: str

    incident_seed: IncidentSeed
    evidence_sources: List[EvidenceSourceConfig]

    twin_source_id: Optional[str] = None

    investigation_policy: InvestigationPolicy = field(
        default_factory=InvestigationPolicy
    )

    evaluation_config: Optional[EvaluationConfig] = None

    output_dir: str = "/content/network_investigations"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def enabled_sources(self) -> List[EvidenceSourceConfig]:
        return [
            source
            for source in self.evidence_sources
            if source.enabled
        ]

    @property
    def required_sources(self) -> List[EvidenceSourceConfig]:
        return [
            source
            for source in self.enabled_sources
            if source.required
        ]

    @property
    def available_evidence_types(self) -> List[str]:
        return sorted(
            {
                source.evidence_type.value
                for source in self.enabled_sources
            }
        )

    @property
    def investigation_output_dir(self) -> Path:
        return Path(self.output_dir) / self.investigation_id

    @property
    def report_path(self) -> Path:
        return (
            self.investigation_output_dir
            / "investigation_report.json"
        )

    @property
    def history_path(self) -> Path:
        return (
            self.investigation_output_dir
            / "investigation_history.json"
        )

    def get_source(
        self,
        source_id: str,
    ) -> EvidenceSourceConfig:
        for source in self.enabled_sources:
            if source.source_id == source_id:
                return source

        raise KeyError(
            f"Evidence source '{source_id}' does not exist "
            "or is disabled."
        )

    def get_sources_by_type(
        self,
        evidence_type: EvidenceType,
    ) -> List[EvidenceSourceConfig]:
        return [
            source
            for source in self.enabled_sources
            if source.evidence_type == evidence_type
        ]

    def validate(self) -> None:
        if not self.investigation_id.strip():
            raise ValueError(
                "investigation_id cannot be empty."
            )

        if not self.investigation_name.strip():
            raise ValueError(
                "investigation_name cannot be empty."
            )

        if not self.scenario_provider.strip():
            raise ValueError(
                "scenario_provider cannot be empty."
            )

        if not self.scenario_id.strip():
            raise ValueError(
                "scenario_id cannot be empty."
            )

        self.incident_seed.validate()
        self.investigation_policy.validate()

        source_ids = [
            source.source_id
            for source in self.enabled_sources
        ]

        duplicate_source_ids = sorted(
            {
                source_id
                for source_id in source_ids
                if source_ids.count(source_id) > 1
            }
        )

        if duplicate_source_ids:
            raise ValueError(
                "Duplicate evidence source IDs: "
                f"{duplicate_source_ids}"
            )

        for source in self.enabled_sources:
            source.validate()

        if self.twin_source_id is not None:
            twin_source = self.get_source(
                self.twin_source_id
            )

            if twin_source.evidence_type not in {
                EvidenceType.TOPOLOGY,
            }:
                raise ValueError(
                    f"twin_source_id '{self.twin_source_id}' "
                    "must reference a topology source."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "investigation_name": self.investigation_name,
            "environment_name": self.environment_name,
            "investigation_type": self.investigation_type,
            "scenario_provider": self.scenario_provider,
            "scenario_id": self.scenario_id,
            "incident_seed": self.incident_seed.to_dict(),
            "evidence_sources": [
                source.to_dict()
                for source in self.evidence_sources
            ],
            "twin_source_id": self.twin_source_id,
            "available_evidence_types": (
                self.available_evidence_types
            ),
            "investigation_policy": (
                self.investigation_policy.to_dict()
            ),
            "evaluation_config": (
                self.evaluation_config.to_dict()
                if self.evaluation_config
                else None
            ),
            "output_dir": self.output_dir,
            "report_path": str(self.report_path),
            "history_path": str(self.history_path),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            "NetworkInvestigationConfig("
            f"id='{self.investigation_id}', "
            f"provider='{self.scenario_provider}', "
            f"scenario='{self.scenario_id}', "
            f"sources={len(self.enabled_sources)})"
        )
