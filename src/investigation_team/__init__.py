"""
Agentic AI Investigation Team.

A vendor-agnostic multi-agent framework for autonomous enterprise
incident investigation and root cause analysis: evidence from
whatever sources you have (network telemetry, vendor alerts,
tickets, application telemetry) gets normalized into one evidence
graph through small per-source adapters, and a chain of six agents
reasons over that graph through deterministic capabilities, looping
until a hypothesis both clears a confidence threshold and survives
adversarial challenge.

Quick start (see the notebooks/ directory for full examples):

    import os
    os.environ["ANTHROPIC_API_KEY"] = "..."
    os.environ["LANGFUSE_PUBLIC_KEY"] = "..."
    os.environ["LANGFUSE_SECRET_KEY"] = "..."

    from investigation_team.orchestrator import run_investigation
    from investigation_team.config import NetworkInvestigationConfig

    run_result = run_investigation(investigation_config=my_config)
"""

from . import adapters  # noqa: F401 -- base adapter registry
from . import capabilities  # noqa: F401 -- triggers capability self-registration
from .orchestrator import InvestigationRunResult, run_investigation

__all__ = [
    "InvestigationRunResult",
    "run_investigation",
]
