"""
Importing this package triggers every capability module's
@register_capability decorator, populating CAPABILITY_REGISTRY.

Anything that needs the registry populated (the orchestrator, the
agent context builders) should import from investigation_team.capabilities
(or investigation_team itself, which imports this) rather than
importing an individual capability module directly -- otherwise a
capability added later but never explicitly imported would silently
never register.
"""

from . import deep_diagnostics, network_state, reachability, topology, vendor_alert
from .registry import (
    CAPABILITY_REGISTRY,
    available_capability_names,
    capability_descriptions,
    get_capability,
    register_capability,
)

__all__ = [
    "CAPABILITY_REGISTRY",
    "available_capability_names",
    "capability_descriptions",
    "get_capability",
    "register_capability",
]
