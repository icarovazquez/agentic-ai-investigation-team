"""
The Evidence Graph: single source of truth that agents reason over
and executors populate/query. Agents never touch this directly —
only through capabilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from .domain import (
    ChangeRecord,
    EntityType,
    EventRecord,
    NetworkEntity,
    NetworkRelationship,
    ObservationRecord,
    RelationshipType,
)


class EvidenceGraph:
    """
    Domain wrapper around a NetworkX MultiDiGraph.

    Agents and executors interact with this domain class rather than
    accessing NetworkX directly.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

        self._entities: Dict[str, NetworkEntity] = {}
        self._relationships: Dict[str, NetworkRelationship] = {}
        self._observations: Dict[str, ObservationRecord] = {}
        self._events: Dict[str, EventRecord] = {}
        self._changes: Dict[str, ChangeRecord] = {}

    # --------------------------------------------------------
    # Counts and summaries
    # --------------------------------------------------------

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def change_count(self) -> int:
        return len(self._changes)

    def summary(self) -> Dict[str, int]:
        """Return basic Evidence Graph record counts."""

        return {
            "entities": self.entity_count,
            "relationships": self.relationship_count,
            "observations": self.observation_count,
            "events": self.event_count,
            "changes": self.change_count,
        }

    # --------------------------------------------------------
    # Entity and relationship mutation
    # --------------------------------------------------------

    def add_entity(self, entity: NetworkEntity) -> None:
        entity.validate()

        if entity.entity_id in self._entities:
            raise ValueError(
                f"Entity '{entity.entity_id}' already exists."
            )

        self._entities[entity.entity_id] = entity

        self._graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            attributes=dict(entity.attributes),
            source_ids=list(entity.source_ids),
        )

    def upsert_entity(self, entity: NetworkEntity) -> None:
        entity.validate()

        self._entities[entity.entity_id] = entity

        self._graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            attributes=dict(entity.attributes),
            source_ids=list(entity.source_ids),
        )

    def add_relationship(
        self,
        relationship: NetworkRelationship,
    ) -> None:
        relationship.validate()

        if relationship.relationship_id in self._relationships:
            raise ValueError(
                f"Relationship "
                f"'{relationship.relationship_id}' already exists."
            )

        if relationship.source_entity_id not in self._entities:
            raise KeyError(
                f"Unknown source entity: "
                f"{relationship.source_entity_id}"
            )

        if relationship.target_entity_id not in self._entities:
            raise KeyError(
                f"Unknown target entity: "
                f"{relationship.target_entity_id}"
            )

        self._relationships[
            relationship.relationship_id
        ] = relationship

        self._graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            key=relationship.relationship_id,
            relationship_id=relationship.relationship_id,
            relationship_type=relationship.relationship_type.value,
            attributes=dict(relationship.attributes),
            valid_from=relationship.valid_from,
            valid_to=relationship.valid_to,
            source_ids=list(relationship.source_ids),
        )

    # --------------------------------------------------------
    # Evidence mutation
    # --------------------------------------------------------

    def add_observation(
        self,
        observation: ObservationRecord,
    ) -> None:
        observation.validate()

        if observation.observation_id in self._observations:
            raise ValueError(
                f"Observation "
                f"'{observation.observation_id}' already exists."
            )

        if observation.entity_id not in self._entities:
            raise KeyError(
                f"Unknown observation entity: "
                f"{observation.entity_id}"
            )

        self._observations[
            observation.observation_id
        ] = observation

    def add_event(self, event: EventRecord) -> None:
        event.validate()

        if event.event_id in self._events:
            raise ValueError(
                f"Event '{event.event_id}' already exists."
            )

        missing_entities = [
            entity_id
            for entity_id in event.entity_ids
            if entity_id not in self._entities
        ]

        if missing_entities:
            raise KeyError(
                f"Unknown event entities: {missing_entities}"
            )

        self._events[event.event_id] = event

    def add_change(self, change: ChangeRecord) -> None:
        change.validate()

        if change.change_id in self._changes:
            raise ValueError(
                f"Change '{change.change_id}' already exists."
            )

        missing_entities = [
            entity_id
            for entity_id in change.entity_ids
            if entity_id not in self._entities
        ]

        if missing_entities:
            raise KeyError(
                f"Unknown change entities: {missing_entities}"
            )

        self._changes[change.change_id] = change

    # --------------------------------------------------------
    # Entity and relationship queries
    # --------------------------------------------------------

    def get_entity(
        self,
        entity_id: str,
    ) -> NetworkEntity:
        try:
            return self._entities[entity_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown entity: {entity_id}"
            ) from exc

    def get_relationship(
        self,
        relationship_id: str,
    ) -> NetworkRelationship:
        try:
            return self._relationships[relationship_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown relationship: {relationship_id}"
            ) from exc

    def list_entities(
        self,
        entity_type: Optional[EntityType] = None,
    ) -> List[NetworkEntity]:
        entities = list(self._entities.values())

        if entity_type is None:
            return entities

        return [
            entity
            for entity in entities
            if entity.entity_type == entity_type
        ]

    def get_neighbors(
        self,
        entity_id: str,
        relationship_type: Optional[
            RelationshipType
        ] = None,
    ) -> List[NetworkEntity]:
        if entity_id not in self._entities:
            raise KeyError(f"Unknown entity: {entity_id}")

        neighbor_ids = set()

        for _, target_id, _, edge_data in self._graph.out_edges(
            entity_id,
            keys=True,
            data=True,
        ):
            if (
                relationship_type is None
                or edge_data.get("relationship_type")
                == relationship_type.value
            ):
                neighbor_ids.add(target_id)

        for source_id, _, _, edge_data in self._graph.in_edges(
            entity_id,
            keys=True,
            data=True,
        ):
            if (
                relationship_type is None
                or edge_data.get("relationship_type")
                == relationship_type.value
            ):
                neighbor_ids.add(source_id)

        return [
            self._entities[neighbor_id]
            for neighbor_id in sorted(neighbor_ids)
        ]

    # --------------------------------------------------------
    # Topology queries
    # --------------------------------------------------------

    def _connectivity_graph(self) -> nx.DiGraph:
        """
        Return a directed connectivity projection of the Evidence Graph.

        Relationships marked as bidirectional are represented in both
        directions for path analysis.
        """

        graph = nx.DiGraph()

        graph.add_nodes_from(self._graph.nodes(data=True))

        for (
            source_id,
            target_id,
            _,
            edge_data,
        ) in self._graph.edges(
            keys=True,
            data=True,
        ):

            graph.add_edge(
                source_id,
                target_id,
                **edge_data,
            )

            relationship_attributes = edge_data.get(
                "attributes",
                {},
            )

            if relationship_attributes.get(
                "bidirectional",
                False,
            ):
                graph.add_edge(
                    target_id,
                    source_id,
                    **edge_data,
                )

        return graph

    def find_paths(
        self,
        source_entity_id: str,
        target_entity_id: str,
        cutoff: Optional[int] = None,
    ) -> List[List[str]]:
        if source_entity_id not in self._entities:
            raise KeyError(
                f"Unknown source entity: {source_entity_id}"
            )

        if target_entity_id not in self._entities:
            raise KeyError(
                f"Unknown target entity: {target_entity_id}"
            )

        return list(
            nx.all_simple_paths(
                self._connectivity_graph(),
                source=source_entity_id,
                target=target_entity_id,
                cutoff=cutoff,
            )
        )

    def shortest_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
    ) -> List[str]:
        if source_entity_id not in self._entities:
            raise KeyError(
                f"Unknown source entity: {source_entity_id}"
            )

        if target_entity_id not in self._entities:
            raise KeyError(
                f"Unknown target entity: {target_entity_id}"
            )

        try:
            return nx.shortest_path(
                self._connectivity_graph(),
                source=source_entity_id,
                target=target_entity_id,
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                f"No directed path exists between "
                f"'{source_entity_id}' and "
                f"'{target_entity_id}'."
            ) from exc

    def descendants(
        self,
        entity_id: str,
    ) -> List[str]:
        """
        Return entities downstream according to explicitly directed
        Evidence Graph relationships.

        Bidirectional connectivity semantics are intentionally not
        applied here.
        """

        if entity_id not in self._entities:
            raise KeyError(f"Unknown entity: {entity_id}")

        simple_graph = nx.DiGraph(self._graph)

        return sorted(
            nx.descendants(
                simple_graph,
                entity_id,
            )
        )

    # --------------------------------------------------------
    # Evidence queries
    # --------------------------------------------------------

    def get_observations(
        self,
        entity_ids: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[ObservationRecord]:
        results = list(self._observations.values())

        if entity_ids is not None:
            entity_id_set = set(entity_ids)
            results = [
                record
                for record in results
                if record.entity_id in entity_id_set
            ]

        if metric_names is not None:
            metric_name_set = set(metric_names)
            results = [
                record
                for record in results
                if record.metric_name in metric_name_set
            ]

        if start_time is not None:
            results = [
                record
                for record in results
                if record.observed_at >= start_time
            ]

        if end_time is not None:
            results = [
                record
                for record in results
                if record.observed_at <= end_time
            ]

        return sorted(
            results,
            key=lambda record: record.observed_at,
        )

    def get_events(
        self,
        entity_ids: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[EventRecord]:
        results = list(self._events.values())

        if entity_ids is not None:
            entity_id_set = set(entity_ids)
            results = [
                record
                for record in results
                if entity_id_set & set(record.entity_ids)
            ]

        if event_types is not None:
            event_type_set = set(event_types)
            results = [
                record
                for record in results
                if record.event_type in event_type_set
            ]

        if start_time is not None:
            results = [
                record
                for record in results
                if record.occurred_at >= start_time
            ]

        if end_time is not None:
            results = [
                record
                for record in results
                if record.occurred_at <= end_time
            ]

        return sorted(
            results,
            key=lambda record: record.occurred_at,
        )
