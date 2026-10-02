"""
Evidence adapter contract and registry.

This is CORE — stable infrastructure every adapter depends on, and
rarely changes itself. Concrete adapters (one per vendor/source) live
under adapters/connectors/ and import from here; this module never
imports from connectors/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from ..config import EvidenceSourceConfig, NetworkInvestigationConfig
from ..domain import ChangeRecord, EventRecord, NetworkEntity, NetworkRelationship, ObservationRecord
from ..evidence_graph import EvidenceGraph


@dataclass
class EvidenceAdapterResult:
    """
    Standardized output returned by every evidence adapter.

    An adapter may populate some or all record collections depending
    on the source. For example, Topology Zoo may return entities and
    relationships, while Prometheus may primarily return observations.
    """

    source_id: str

    entities: List[NetworkEntity] = field(default_factory=list)
    relationships: List[NetworkRelationship] = field(default_factory=list)
    observations: List[ObservationRecord] = field(default_factory=list)
    events: List[EventRecord] = field(default_factory=list)
    changes: List[ChangeRecord] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty.")

        for entity in self.entities:
            entity.validate()

        for relationship in self.relationships:
            relationship.validate()

        for observation in self.observations:
            observation.validate()

        for event in self.events:
            event.validate()

        for change in self.changes:
            change.validate()

    def summary(self) -> Dict[str, int]:
        return {
            "entities": len(self.entities),
            "relationships": len(self.relationships),
            "observations": len(self.observations),
            "events": len(self.events),
            "changes": len(self.changes),
            "warnings": len(self.warnings),
        }


class EvidenceAdapter(ABC):
    """
    Abstract translator between a raw evidence source and the
    vendor-neutral Evidence Graph domain model.

    Concrete adapters must:
      1. read a source;
      2. normalize source-specific schemas;
      3. create typed domain records; and
      4. return an EvidenceAdapterResult.

    Adapters do not diagnose incidents or evaluate hypotheses.
    """

    def __init__(
        self,
        source_config: EvidenceSourceConfig,
    ) -> None:
        self.source_config = source_config
        self.source_config.validate()

    @property
    def source_id(self) -> str:
        return self.source_config.source_id

    @abstractmethod
    def load(self) -> EvidenceAdapterResult:
        """
        Load and normalize the configured evidence source.

        Returns
        -------
        EvidenceAdapterResult
            Vendor-neutral entities, relationships, observations,
            events, and changes.
        """
        raise NotImplementedError

    def validate_result(
        self,
        result: EvidenceAdapterResult,
    ) -> None:
        """
        Validate the adapter output and ensure source attribution
        matches the adapter configuration.
        """

        result.validate()

        if result.source_id != self.source_id:
            raise ValueError(
                "Adapter result source_id does not match the "
                f"configured source_id: expected '{self.source_id}', "
                f"received '{result.source_id}'."
            )


# ============================================================
# ADAPTER REGISTRY
# ============================================================

ADAPTER_REGISTRY: Dict[str, type] = {}


def register_adapter(
    adapter_name: str,
    adapter_class: type,
) -> None:
    """
    Register an EvidenceAdapter implementation by name.
    """

    if not adapter_name.strip():
        raise ValueError("adapter_name cannot be empty.")

    if not issubclass(adapter_class, EvidenceAdapter):
        raise TypeError(
            f"{adapter_class.__name__} must inherit from EvidenceAdapter."
        )

    if adapter_name in ADAPTER_REGISTRY:
        raise ValueError(
            f"Adapter '{adapter_name}' is already registered."
        )

    ADAPTER_REGISTRY[adapter_name] = adapter_class


def get_adapter_class(
    adapter_name: str,
) -> type:
    """
    Resolve the adapter class referenced by EvidenceSourceConfig.
    """

    try:
        return ADAPTER_REGISTRY[adapter_name]

    except KeyError as exc:
        raise KeyError(
            f"No adapter registered as '{adapter_name}'. "
            f"Available adapters: {sorted(ADAPTER_REGISTRY)}"
        ) from exc


# ============================================================
# INGESTION / LOADING EXECUTORS
#
# Deterministic — no LLM involved. Belong here rather than in the
# top-level executors.py because they're specifically about moving
# data from an adapter into the graph, the adapter layer's own job.
# ============================================================


def ingest_adapter_result_executor(
    evidence_graph: EvidenceGraph,
    adapter_result: EvidenceAdapterResult,
    *,
    upsert_entities: bool = True,
) -> Dict[str, Any]:
    """
    Insert normalized adapter records into an EvidenceGraph.

    This is an executor because ingestion is deterministic. It does
    not interpret evidence or make root-cause conclusions.
    """

    adapter_result.validate()

    ingested_counts = {
        "entities": 0,
        "relationships": 0,
        "observations": 0,
        "events": 0,
        "changes": 0,
    }

    # Entities must be inserted first because every other record may
    # reference them.
    for entity in adapter_result.entities:
        if upsert_entities:
            evidence_graph.upsert_entity(entity)
        else:
            evidence_graph.add_entity(entity)

        ingested_counts["entities"] += 1

    # Relationships require both endpoint entities to already exist.
    for relationship in adapter_result.relationships:
        evidence_graph.add_relationship(relationship)
        ingested_counts["relationships"] += 1

    for observation in adapter_result.observations:
        evidence_graph.add_observation(observation)
        ingested_counts["observations"] += 1

    for event in adapter_result.events:
        evidence_graph.add_event(event)
        ingested_counts["events"] += 1

    for change in adapter_result.changes:
        evidence_graph.add_change(change)
        ingested_counts["changes"] += 1

    return {
        "executor": "ingest_adapter_result_executor",
        "source_id": adapter_result.source_id,
        "ingested_counts": ingested_counts,
        "warnings": list(adapter_result.warnings),
        "adapter_metadata": dict(adapter_result.metadata),
        "graph_summary": evidence_graph.summary(),
    }


def load_evidence_source_executor(
    source_config: EvidenceSourceConfig,
    evidence_graph: EvidenceGraph,
) -> Dict[str, Any]:
    """
    Resolve the configured adapter, normalize one source,
    and ingest the result into the Evidence Graph.
    """

    source_config.validate()

    adapter_class = get_adapter_class(
        source_config.adapter_name
    )

    adapter = adapter_class(
        source_config=source_config
    )

    adapter_result = adapter.load()

    adapter.validate_result(
        adapter_result
    )

    ingestion_result = ingest_adapter_result_executor(
        evidence_graph=evidence_graph,
        adapter_result=adapter_result,
    )

    return {
        "executor": "load_evidence_source_executor",
        "source_id": source_config.source_id,
        "adapter_name": source_config.adapter_name,
        "adapter_summary": adapter_result.summary(),
        "ingestion_result": ingestion_result,
    }


def load_investigation_evidence_executor(
    investigation_config: NetworkInvestigationConfig,
    evidence_graph: EvidenceGraph = None,
) -> Dict[str, Any]:
    """
    Load every enabled evidence source configured for an investigation.

    Required-source failures stop execution.
    Optional-source failures are recorded and skipped.
    """

    investigation_config.validate()

    if evidence_graph is None:
        evidence_graph = EvidenceGraph()

    source_results: Dict[str, Any] = {}
    source_errors: Dict[str, str] = {}

    for source_config in investigation_config.enabled_sources:

        try:
            result = load_evidence_source_executor(
                source_config=source_config,
                evidence_graph=evidence_graph,
            )

            source_results[
                source_config.source_id
            ] = result

            print(
                f"✓ Loaded {source_config.source_id} "
                f"using {source_config.adapter_name}"
            )

        except Exception as exc:
            source_errors[
                source_config.source_id
            ] = str(exc)

            if source_config.required:
                raise RuntimeError(
                    "Required evidence source failed: "
                    f"{source_config.source_id}"
                ) from exc

            print(
                f"⚠ Optional evidence source unavailable: "
                f"{source_config.source_id}"
            )

    return {
        "executor": "load_investigation_evidence_executor",
        "investigation_id": (
            investigation_config.investigation_id
        ),
        "source_results": source_results,
        "source_errors": source_errors,
        "graph_summary": evidence_graph.summary(),
        "evidence_graph": evidence_graph,
    }
