"""
Network scope: the part of the topology around the affected entities,
handed to the agents that decide what to suspect.

Why this exists. The Hypothesis Generator may only suspect entity ids
that appear in its context. In the first scenario the incident named two
endpoints, so a path between them put every device on it in the context.
In the second, one user reported the problem, there was no pair of
endpoints to trace a path between, and the only entity in the whole
context was the reporter's own workstation. Every hypothesis then
suspected that workstation, the evidence tests only ever asked about it,
and the shared infrastructure where the fault actually was never entered
the investigation. Nothing downstream could fix that: the scope was
decided before any evidence was looked at.

So scope is built from the topology, deterministically: everything within
a few hops of the affected entities, plus the path from each of them to
every service in the graph.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import networkx as nx

from .domain import EntityType
from .evidence_graph import EvidenceGraph

_RELATIONSHIP_ATTRIBUTES = ("link_name", "capacity_mbps", "routing_protocol")


def network_scope(
    evidence_graph: EvidenceGraph,
    seed_entity_ids: Iterable[str],
    hops: int = 3,
    max_entities: int = 40,
) -> Dict[str, Any]:
    seeds = [e for e in dict.fromkeys(seed_entity_ids) if _known(evidence_graph, e)]
    if not seeds:
        return {"seed_entity_ids": [], "entities": [], "relationships": [], "paths_to_services": []}

    undirected = evidence_graph._connectivity_graph().to_undirected()

    # Nearest first, so a large graph is cut off at the far edge, not the middle.
    distance: Dict[str, int] = {}
    for seed in seeds:
        for node, d in nx.single_source_shortest_path_length(undirected, seed, cutoff=hops).items():
            distance[node] = min(d, distance.get(node, d))
    included = sorted(distance, key=lambda n: (distance[n], n))[:max_entities]
    included_set = set(included)

    entities: List[Dict[str, Any]] = []
    for entity_id in included:
        entity = evidence_graph.get_entity(entity_id)
        entities.append({
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "role": entity.attributes.get("role"),
            "hops_from_affected": distance[entity_id],
        })

    relationships: List[Dict[str, Any]] = []
    for relationship in evidence_graph.list_relationships():
        if relationship.source_entity_id in included_set and relationship.target_entity_id in included_set:
            item = {
                "source": relationship.source_entity_id,
                "target": relationship.target_entity_id,
                "type": relationship.relationship_type.value,
            }
            item.update({k: relationship.attributes[k] for k in _RELATIONSHIP_ATTRIBUTES
                         if k in relationship.attributes})
            relationships.append(item)

    paths: List[Dict[str, Any]] = []
    services = [e.entity_id for e in evidence_graph.list_entities(EntityType.SERVICE)]
    for seed in seeds:
        for service in services:
            if seed == service:
                continue
            try:
                path = evidence_graph.shortest_path(seed, service)
            except (ValueError, KeyError):
                continue
            paths.append({"from": seed, "to": service, "path": path})

    return {
        "seed_entity_ids": seeds,
        "entities": entities,
        "relationships": relationships,
        "paths_to_services": paths,
    }


def _known(evidence_graph: EvidenceGraph, entity_id: str) -> bool:
    try:
        evidence_graph.get_entity(entity_id)
        return True
    except KeyError:
        return False

