"""
Entity resolution: one place that maps the identifiers each source
uses (hostnames, asset tags, UPNs, MAC and IP addresses, what a person
types into a ticket) onto the canonical entity ids in the Evidence
Graph.

Every source names things differently. A ticket says "WS-0142", the
call-quality platform says "MARIA-LAPTOP.corp.example.com" and the
topology says "nika:qos_contention:ws1". Without a shared resolver each
adapter grows its own private name table, the tables drift apart, and a
mismatch shows up as evidence that silently attaches to nothing. So:

- the topology source (the system of record) declares aliases on its
  entities (attributes["aliases"]);
- every other adapter resolves through one EntityResolver built from the
  graph;
- an identifier that cannot be resolved, or that matches more than one
  entity, is returned as such. Adapters turn those into warnings.
  Nothing is guessed, and nothing is dropped without being reported.

Deterministic on purpose. No fuzzy matching and no LLM: a wrong match
here would attach evidence to the wrong device with full confidence.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

_MAC_RE = re.compile(r"^(?:[0-9a-f]{2}[:\-.]?){5}[0-9a-f]{2}$")
_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def normalize_identifier(raw: str) -> str:
    """Lowercase, normalize apostrophes and whitespace. Nothing cleverer."""
    text = (raw or "").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip().strip(".,;:").lower()


def _mac_key(key: str) -> Optional[str]:
    if _MAC_RE.match(key):
        return re.sub(r"[:\-.]", "", key)
    return None


def _hostname_key(key: str) -> Optional[str]:
    """'ws-0142.corp.example.com' -> 'ws-0142'. Not for IPs or emails."""
    if "." not in key or "@" in key or " " in key or _IP_RE.match(key):
        return None
    return key.split(".", 1)[0]


@dataclass
class Resolution:
    raw: str
    status: str  # "resolved" | "unresolved" | "ambiguous"
    entity_id: Optional[str] = None
    method: Optional[str] = None  # "exact" | "mac" | "hostname"
    candidates: List[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def describe(self) -> str:
        if self.status == "ambiguous":
            return f"'{self.raw}' is ambiguous: could be {', '.join(sorted(self.candidates))}"
        if self.status == "unresolved":
            return f"'{self.raw}' does not match any known entity"
        return f"'{self.raw}' -> {self.entity_id} ({self.method})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EntityResolver:
    def __init__(self) -> None:
        self._exact: Dict[str, Set[str]] = {}
        self._mac: Dict[str, Set[str]] = {}
        self._host: Dict[str, Set[str]] = {}
        self._entity_ids: Set[str] = set()

    @property
    def known_entity_ids(self) -> Set[str]:
        return set(self._entity_ids)

    def register(self, entity_id: str, aliases: Iterable[str]) -> None:
        if not entity_id.strip():
            raise ValueError("entity_id cannot be empty.")
        self._entity_ids.add(entity_id)
        for alias in [entity_id, *aliases]:
            key = normalize_identifier(str(alias))
            if not key:
                continue
            self._exact.setdefault(key, set()).add(entity_id)
            mac = _mac_key(key)
            if mac:
                self._mac.setdefault(mac, set()).add(entity_id)
            host = _hostname_key(key)
            if host:
                self._host.setdefault(host, set()).add(entity_id)
            # A bare short name ("ws-0142") must also be reachable from
            # the FQDN form, so index the short form of every alias.
            if "." not in key and "@" not in key and not _IP_RE.match(key):
                self._host.setdefault(key, set()).add(entity_id)

    @classmethod
    def from_graph(cls, evidence_graph: Any) -> "EntityResolver":
        resolver = cls()
        for entity in evidence_graph.list_entities():
            aliases = [entity.name, *entity.attributes.get("aliases", [])]
            resolver.register(entity.entity_id, aliases)
        return resolver

    def resolve(self, raw: str) -> Resolution:
        key = normalize_identifier(raw)
        if not key:
            return Resolution(raw=raw, status="unresolved")

        lookups = [("exact", self._exact, key)]
        mac = _mac_key(key)
        if mac:
            lookups.append(("mac", self._mac, mac))
        host = _hostname_key(key)
        if host:
            lookups.append(("hostname", self._host, host))

        for method, index, lookup_key in lookups:
            matches = index.get(lookup_key)
            if not matches:
                continue
            if len(matches) == 1:
                return Resolution(raw=raw, status="resolved",
                                  entity_id=next(iter(matches)), method=method)
            return Resolution(raw=raw, status="ambiguous", candidates=sorted(matches))

        return Resolution(raw=raw, status="unresolved")

    def resolve_many(self, raws: Iterable[str]) -> List[Resolution]:
        return [self.resolve(r) for r in raws]
