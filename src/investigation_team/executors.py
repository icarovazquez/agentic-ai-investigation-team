"""
Deterministic executors: plain code, no LLM involved, that answer
specific questions against the Evidence Graph. Capabilities wrap one
or more of these into a single normalized result for agents to use.

execute_evidence_test / execute_evidence_plan dispatch by capability
name through the capability registry (capabilities/registry.py)
rather than a hardcoded if/elif chain and a growing list of
`*_capability` parameters — adding a new capability (like the
TrafficCapability planned for the multi-source scenario) requires no
change here at all, just the new capability module being imported
somewhere before an investigation runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from .capabilities.registry import get_capability
from .domain import EvidenceTest, EvidenceTestResult, ObservationRecord
from .evidence_graph import EvidenceGraph


# ============================================================
# ENTITY / TOPOLOGY EXECUTORS
# ============================================================


def entity_lookup_executor(
    evidence_graph: EvidenceGraph,
    entity_id: str,
) -> Dict[str, Any]:
    entity = evidence_graph.get_entity(entity_id)

    return {
        "executor": "entity_lookup_executor",
        "entity": entity.to_dict(),
        "neighbors": [
            neighbor.to_dict()
            for neighbor in evidence_graph.get_neighbors(
                entity_id
            )
        ],
    }


def path_analysis_executor(
    evidence_graph: EvidenceGraph,
    source_entity_id: str,
    target_entity_id: str,
    *,
    include_all_paths: bool = False,
    cutoff: Optional[int] = None,
) -> Dict[str, Any]:
    shortest_path = evidence_graph.shortest_path(
        source_entity_id,
        target_entity_id,
    )

    result = {
        "executor": "path_analysis_executor",
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "shortest_path": shortest_path,
        "hop_count": len(shortest_path) - 1,
    }

    if include_all_paths:
        all_paths = evidence_graph.find_paths(
            source_entity_id,
            target_entity_id,
            cutoff=cutoff,
        )

        result["all_paths"] = all_paths
        result["path_count"] = len(all_paths)

    return result


def blast_radius_executor(
    evidence_graph: EvidenceGraph,
    failed_entity_id: str,
) -> Dict[str, Any]:
    failed_entity = evidence_graph.get_entity(
        failed_entity_id
    )

    simple_graph = nx.DiGraph(evidence_graph._graph)

    affected_entity_ids = sorted(
        nx.descendants(
            simple_graph,
            failed_entity_id,
        )
    )

    return {
        "executor": "blast_radius_executor",
        "failed_entity": failed_entity.to_dict(),
        "affected_entity_ids": affected_entity_ids,
        "affected_count": len(affected_entity_ids),
    }


# ============================================================
# REACHABILITY / NETWORK STATE / VENDOR ALERT EXECUTORS
# ============================================================


def reachability_evidence_executor(
    evidence_graph: EvidenceGraph,
    source_entity_id: str,
    target_entity_id: str,
) -> Dict[str, Any]:
    """
    Retrieve normalized reachability observations between two entities.

    Expected future metrics may include:
      - ping_success
      - packet_loss
      - latency_ms
      - traceroute_hop_count

    The executor does not perform live network commands. It queries
    evidence already normalized into the Evidence Graph.
    """

    source_observations = evidence_graph.get_observations(
        entity_ids=[source_entity_id]
    )

    reachability_metrics = {
        "ping_success",
        "packet_loss",
        "latency_ms",
        "traceroute_hop_count",
    }

    matching_observations = []

    for observation in source_observations:

        if observation.metric_name not in reachability_metrics:
            continue

        observed_target = observation.dimensions.get(
            "target_entity_id"
        )

        if observed_target == target_entity_id:
            matching_observations.append(
                observation.to_dict()
            )

    return {
        "executor": "reachability_evidence_executor",
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "observations": matching_observations,
        "observation_count": len(
            matching_observations
        ),
    }


def network_state_evidence_executor(
    evidence_graph: EvidenceGraph,
    entity_ids: List[str],
) -> Dict[str, Any]:
    """
    Retrieve normalized operational-state observations for entities.

    Expected future metrics may include:
      - interface_oper_status
      - interface_admin_status
      - bgp_session_state
      - route_present
      - device_reachable
    """

    network_state_metrics = {
        "interface_oper_status",
        "interface_admin_status",
        "bgp_session_state",
        "route_present",
        "device_reachable",
    }

    observations = evidence_graph.get_observations(
        entity_ids=entity_ids
    )

    matching_observations = [
        observation.to_dict()
        for observation in observations
        if observation.metric_name
        in network_state_metrics
    ]

    return {
        "executor": "network_state_evidence_executor",
        "entity_ids": list(entity_ids),
        "observations": matching_observations,
        "observation_count": len(
            matching_observations
        ),
    }


def vendor_alert_evidence_executor(
    evidence_graph: EvidenceGraph,
    entity_ids: List[str],
) -> Dict[str, Any]:
    """
    Retrieve vendor-sourced alert/incident events for entities.

    Unlike network_state_evidence_executor, this does not filter by
    a fixed metric-name allowlist — vendor event_type values vary
    by source (PagerDuty's "vendor_incident", a future Datadog
    adapter's "monitor_alert", etc.), so all events touching the
    given entities are returned. Callers that care about a specific
    vendor or event type can filter the result further.
    """

    events = evidence_graph.get_events(
        entity_ids=entity_ids
    )

    return {
        "executor": "vendor_alert_evidence_executor",
        "entity_ids": list(entity_ids),
        "events": [event.to_dict() for event in events],
        "event_count": len(events),
    }


def deep_diagnostics_evidence_executor(
    evidence_graph: EvidenceGraph,
    entity_ids: List[str],
) -> Dict[str, Any]:
    """
    Fetch deep diagnostic telemetry (BGP session state, interface
    error counters, administrative status) for the given entities
    and add it to the evidence graph as it's retrieved.

    This is a deliberately separate, explicit query from
    network_state_evidence_executor — it represents evidence that
    is NOT part of routine first-response telemetry and must be
    specifically requested, mirroring a real investigator escalating
    from a dashboard check to pulling device-level diagnostics.

    Deterministic fixture, scoped to the NIKA simple_bgp scenario's
    injected router1<->router2 link failure. A later version would
    query NIKA's live diagnostic interfaces.
    """

    SOURCE_ID = "nika_simple_bgp_deep_diagnostics"
    OBSERVED_AT = "2026-08-07T00:00:03Z"

    fixture_observations_by_entity = {
        "nika:simple_bgp:router1": [
            ObservationRecord(
                observation_id="nika:simple_bgp:obs:router1-router2-bgp-state",
                source_id=SOURCE_ID,
                entity_id="nika:simple_bgp:router1",
                metric_name="bgp_session_state",
                metric_value="down",
                observed_at=OBSERVED_AT,
                dimensions={"peer_entity_id": "nika:simple_bgp:router2"},
            ),
            ObservationRecord(
                observation_id="nika:simple_bgp:obs:router1-router2-error-count",
                source_id=SOURCE_ID,
                entity_id="nika:simple_bgp:router1",
                metric_name="interface_error_count",
                metric_value=48213,
                observed_at=OBSERVED_AT,
                dimensions={"peer_entity_id": "nika:simple_bgp:router2"},
                unit="crc_errors",
            ),
            ObservationRecord(
                observation_id="nika:simple_bgp:obs:router1-router2-admin-status",
                source_id=SOURCE_ID,
                entity_id="nika:simple_bgp:router1",
                metric_name="interface_admin_status",
                metric_value="up",
                observed_at=OBSERVED_AT,
                dimensions={"peer_entity_id": "nika:simple_bgp:router2"},
            ),
        ],
        "nika:simple_bgp:router2": [
            ObservationRecord(
                observation_id="nika:simple_bgp:obs:router2-router1-bgp-state",
                source_id=SOURCE_ID,
                entity_id="nika:simple_bgp:router2",
                metric_name="bgp_session_state",
                metric_value="down",
                observed_at=OBSERVED_AT,
                dimensions={"peer_entity_id": "nika:simple_bgp:router1"},
            ),
            ObservationRecord(
                observation_id="nika:simple_bgp:obs:router2-router1-error-count",
                source_id=SOURCE_ID,
                entity_id="nika:simple_bgp:router2",
                metric_name="interface_error_count",
                metric_value=51002,
                observed_at=OBSERVED_AT,
                dimensions={"peer_entity_id": "nika:simple_bgp:router1"},
                unit="crc_errors",
            ),
            ObservationRecord(
                observation_id="nika:simple_bgp:obs:router2-router1-admin-status",
                source_id=SOURCE_ID,
                entity_id="nika:simple_bgp:router2",
                metric_name="interface_admin_status",
                metric_value="up",
                observed_at=OBSERVED_AT,
                dimensions={"peer_entity_id": "nika:simple_bgp:router1"},
            ),
        ],
    }

    new_observations = []

    for entity_id in entity_ids:
        for observation in fixture_observations_by_entity.get(entity_id, []):
            # Avoid duplicate inserts if this capability is called
            # more than once for the same entity across rounds.
            try:
                evidence_graph.add_observation(observation)
                new_observations.append(observation)
            except ValueError:
                # Already present from a prior round's call — that's fine,
                # still include it in this round's returned evidence.
                existing = evidence_graph.get_observations(
                    entity_ids=[entity_id],
                    metric_names=[observation.metric_name],
                )
                new_observations.extend(existing)

    return {
        "executor": "deep_diagnostics_evidence_executor",
        "entity_ids": list(entity_ids),
        "observations": [o.to_dict() for o in new_observations],
        "observation_count": len(new_observations),
    }


# ============================================================
# EVIDENCE TEST EXECUTION — dispatches via the capability registry
# ============================================================


def execute_evidence_test(
    test: EvidenceTest,
    evidence_graph: EvidenceGraph,
) -> EvidenceTestResult:

    try:
        capability = get_capability(test.capability)

        evidence = capability.collect_evidence(
            evidence_graph=evidence_graph,
            **test.parameters,
        )

        return EvidenceTestResult(
            test_id=test.test_id,
            hypothesis_id=test.hypothesis_id,
            capability=test.capability,
            status="completed",
            evidence=evidence,
        )

    except Exception as exc:
        return EvidenceTestResult(
            test_id=test.test_id,
            hypothesis_id=test.hypothesis_id,
            capability=test.capability,
            status="failed",
            error=str(exc),
        )


def execute_evidence_plan(
    evidence_plan,
    evidence_graph: EvidenceGraph,
) -> List[EvidenceTestResult]:

    ordered_tests = sorted(
        evidence_plan.tests,
        key=lambda test: test.priority,
    )

    results = []

    for test in ordered_tests:
        result = execute_evidence_test(
            test=test,
            evidence_graph=evidence_graph,
        )

        results.append(result)

    return results
