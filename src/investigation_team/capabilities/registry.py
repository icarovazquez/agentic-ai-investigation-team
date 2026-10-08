"""
Capability registry.

This is the single source of truth for "what capabilities exist."
In the original notebook, three separate places each hardcoded their
own copy of the capability list (two context builders and the
execute_evidence_test dispatch), and they drifted out of sync —
build_incident_framing_context and build_hypothesis_generation_context
never got vendor_alert/deep_diagnostics added when those capabilities
were built. This registry fixes that at the root: every capability
self-registers via the @register_capability decorator, and anything
needing "what capabilities are available" reads from here instead of
keeping its own list.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CapabilityRegistration:
    name: str
    description: str
    instance: Any  # the capability object, exposing collect_evidence(...)


CAPABILITY_REGISTRY: Dict[str, CapabilityRegistration] = {}


def register_capability(name: str, description: str):
    """
    Class decorator: registers a capability under `name`, with the
    description agents use to know it exists and when to reach for it.

    Capabilities are treated as stateless singletons — instantiated
    once, here, at import time. This is safe because every capability
    built so far is a thin, stateless wrapper around whatever
    evidence_graph is passed into collect_evidence() per call. If a
    future capability ever needs genuine per-investigation state,
    this assumption would need revisiting.
    """

    def decorator(cls):
        if name in CAPABILITY_REGISTRY:
            raise ValueError(
                f"Capability '{name}' is already registered as "
                f"{CAPABILITY_REGISTRY[name].instance.__class__.__name__} — "
                f"check for a duplicate registration."
            )

        CAPABILITY_REGISTRY[name] = CapabilityRegistration(
            name=name,
            description=description,
            instance=cls(),
        )
        return cls

    return decorator


def available_capability_names() -> List[str]:
    return list(CAPABILITY_REGISTRY.keys())


def capability_descriptions() -> Dict[str, str]:
    return {name: reg.description for name, reg in CAPABILITY_REGISTRY.items()}


def resolve_capability_name(raw: str) -> Optional[str]:
    """
    Map a model-supplied capability string to a registered capability
    name, or None. Models sometimes return a list-as-string such as
    "network_state, deep_diagnostics"; the first registered name in it
    wins. Enforced in code because the prompt already says "one of
    available_capabilities" and was ignored.
    """
    if not isinstance(raw, str):
        return None
    tokens = [t for t in re.split(r"[^A-Za-z0-9_]+", raw) if t]
    for token in tokens:
        if token in CAPABILITY_REGISTRY:
            return token
    return None


def get_capability(name: str):
    if name not in CAPABILITY_REGISTRY:
        raise ValueError(
            f"No capability registered as '{name}'. "
            f"Available: {available_capability_names()}"
        )
    return CAPABILITY_REGISTRY[name].instance
