"""
Shared domain model: enums and cross-cutting dataclasses used by more
than one layer (evidence graph, adapters, executors, and more than one
agent). Anything owned by exactly one agent lives in that agent's own
module instead — see agents/*.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# ENUMS
# ============================================================


class EntityType(str, Enum):
    """Types of entities that can appear in the Evidence Graph."""

    SITE = "site"
    DEVICE = "device"
    INTERFACE = "interface"
    LINK = "link"
    HOST = "host"
    SERVICE = "service"
    APPLICATION = "application"
    NETWORK = "network"
    VRF = "vrf"
    VLAN = "vlan"
    ROUTE = "route"


class RelationshipType(str, Enum):
    """Relationships between entities."""

    CONTAINS = "contains"
    CONNECTED_TO = "connected_to"
    DEPENDS_ON = "depends_on"
    ROUTES_TO = "routes_to"
    HOSTS = "hosts"
    MEMBER_OF = "member_of"
    PEERS_WITH = "peers_with"
    BACKS_UP = "backs_up"


class EvidenceType(str, Enum):
    """Evidence categories."""

    TOPOLOGY = "topology"
    TELEMETRY = "telemetry"
    CONFIGURATION = "configuration"
    EVENT = "event"
    CHANGE = "change"
    FLOW = "flow"
    SYNTHETIC_TEST = "synthetic_test"
    ROUTING = "routing"


class InvestigationStatus(str, Enum):
    """Overall investigation lifecycle."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_EVIDENCE = "waiting_for_evidence"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    REMEDIATION_RECOMMENDED = "remediation_recommended"
    CLOSED = "closed"
    INCONCLUSIVE = "inconclusive"


class HypothesisStatus(str, Enum):
    """Status of an investigation hypothesis."""

    PROPOSED = "proposed"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"


class EvidenceDirection(str, Enum):
    """How evidence affects a hypothesis."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class RemediationMode(str, Enum):
    """How remediation is handled."""

    RECOMMEND_ONLY = "recommend_only"
    HUMAN_APPROVAL = "human_approval"
    AUTOMATIC = "automatic"


class SourceFormat(str, Enum):
    """Physical or logical format exposed by an evidence source."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    GRAPHML = "graphml"
    YAML = "yaml"
    API = "api"
    DATABASE = "database"
    COMMAND = "command"
    LIVE_LAB = "live_lab"


class AccessMode(str, Enum):
    """How an adapter accesses an evidence source."""

    FILE = "file"
    API = "api"
    DATABASE = "database"
    COMMAND = "command"
    LIVE_LAB = "live_lab"


# ============================================================
# EVIDENCE GRAPH RECORDS
# ============================================================


@dataclass
class NetworkEntity:
    entity_id: str
    entity_type: EntityType
    name: str

    attributes: Dict[str, Any] = field(default_factory=dict)
    source_ids: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty.")

        if not self.name.strip():
            raise ValueError("name cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["entity_type"] = self.entity_type.value
        return result


@dataclass
class NetworkRelationship:
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType

    attributes: Dict[str, Any] = field(default_factory=dict)

    valid_from: Optional[str] = None
    valid_to: Optional[str] = None

    source_ids: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.relationship_id.strip():
            raise ValueError("relationship_id cannot be empty.")

        if not self.source_entity_id.strip():
            raise ValueError("source_entity_id cannot be empty.")

        if not self.target_entity_id.strip():
            raise ValueError("target_entity_id cannot be empty.")

        if self.source_entity_id == self.target_entity_id:
            raise ValueError(
                "A relationship cannot connect an entity to itself."
            )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["relationship_type"] = self.relationship_type.value
        return result


@dataclass
class ObservationRecord:
    observation_id: str
    source_id: str
    entity_id: str

    metric_name: str
    metric_value: Any
    observed_at: str

    unit: Optional[str] = None
    dimensions: Dict[str, Any] = field(default_factory=dict)

    quality_score: float = 1.0

    def validate(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id cannot be empty.")

        if not self.entity_id.strip():
            raise ValueError("entity_id cannot be empty.")

        if not self.metric_name.strip():
            raise ValueError("metric_name cannot be empty.")

        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                "quality_score must be between 0 and 1."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventRecord:
    event_id: str
    source_id: str
    event_type: str
    occurred_at: str

    entity_ids: List[str] = field(default_factory=list)

    severity: Optional[str] = None
    message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id cannot be empty.")

        if not self.event_type.strip():
            raise ValueError("event_type cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChangeRecord:
    change_id: str
    source_id: str
    changed_at: str
    change_type: str

    entity_ids: List[str] = field(default_factory=list)

    previous_state: Any = None
    new_state: Any = None

    actor: Optional[str] = None
    approved: Optional[bool] = None
    rollback_reference: Optional[str] = None

    attributes: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.change_id.strip():
            raise ValueError("change_id cannot be empty.")

        if not self.change_type.strip():
            raise ValueError("change_type cannot be empty.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# EVIDENCE TEST CONTRACTS
#
# Shared between the Evidence Planner agent (produces these) and
# the executors/orchestrator (consume these) — kept here rather
# than in agents/evidence_planner.py so executors.py never has to
# import upward from the agents package.
# ============================================================


@dataclass
class EvidenceTest:
    """
    One planned evidence request for evaluating a hypothesis.
    """

    test_id: str
    hypothesis_id: str

    objective: str
    capability: str

    parameters: Dict[str, Any] = field(default_factory=dict)

    expected_supporting_observations: List[str] = field(
        default_factory=list
    )

    expected_falsifying_observations: List[str] = field(
        default_factory=list
    )

    priority: int = 1

    def validate(self) -> None:
        if not self.test_id.strip():
            raise ValueError("test_id cannot be empty.")

        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id cannot be empty.")

        if not self.objective.strip():
            raise ValueError("objective cannot be empty.")

        if not self.capability.strip():
            raise ValueError("capability cannot be empty.")

        if self.priority < 1:
            raise ValueError("priority must be at least 1.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePlan:
    """
    Planned deterministic evidence tests for one incident.
    """

    incident_id: str
    tests: List[EvidenceTest]

    def validate(self) -> None:
        if not self.incident_id.strip():
            raise ValueError("incident_id cannot be empty.")

        if not self.tests:
            raise ValueError(
                "EvidencePlan must contain at least one test."
            )

        test_ids = [test.test_id for test in self.tests]

        if len(test_ids) != len(set(test_ids)):
            raise ValueError("Evidence test IDs must be unique.")

        for test in self.tests:
            test.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "tests": [
                test.to_dict()
                for test in self.tests
            ],
        }


@dataclass
class EvidenceTestResult:
    test_id: str
    hypothesis_id: str
    capability: str

    status: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
