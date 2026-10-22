"""
Has the investigation learned anything new?

Another round only helps if it can return evidence the investigation does
not already have. When a round's tests return only observations, events and
changes seen before, re-analysing them yields the same answer and another
round yields the same tests again, so the loop should stop and say so.
Deterministic: it compares evidence ids.
"""

from __future__ import annotations

from typing import Iterable, Set

from .domain import EvidenceTestResult

_ID_FIELDS = (
    ("observations", "observation_id", "obs"),
    ("events", "event_id", "evt"),
    ("changes", "change_id", "chg"),
)


def evidence_ids(results: Iterable[EvidenceTestResult]) -> Set[str]:
    ids: Set[str] = set()
    for result in results:
        for key, payload in (result.evidence or {}).items():
            if not key.endswith("_evidence") or not isinstance(payload, dict):
                continue
            for collection, id_field, prefix in _ID_FIELDS:
                for record in payload.get(collection, []) or []:
                    if isinstance(record, dict) and record.get(id_field):
                        ids.add(f"{prefix}:{record[id_field]}")
    return ids
