"""
Importing this package pulls in the adapter base layer (registry,
EvidenceAdapter ABC) and every connector, so all adapters are
registered as soon as investigation_team itself is imported.
"""

from . import base  # noqa: F401
from . import connectors  # noqa: F401
from .base import (
    ADAPTER_REGISTRY,
    EvidenceAdapter,
    EvidenceAdapterResult,
    get_adapter_class,
    ingest_adapter_result_executor,
    load_evidence_source_executor,
    load_investigation_evidence_executor,
    register_adapter,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "EvidenceAdapter",
    "EvidenceAdapterResult",
    "get_adapter_class",
    "ingest_adapter_result_executor",
    "load_evidence_source_executor",
    "load_investigation_evidence_executor",
    "register_adapter",
]
