# Agentic AI Investigation Team

An Agentic AI system that investigates enterprise incidents — not just a
network, but evidence pulled from the SaaS tools most enterprises already
use (PagerDuty, ServiceNow, Microsoft Teams, and more as it grows).

This is a fork and generalization of an earlier
[network-specific investigation team](https://github.com/icarovazquez/agentic-ai-network-investigation-team),
rebuilt to treat *any* enterprise data source — network telemetry, vendor
alerts, tickets, application telemetry — as just another input into one
evidence graph.

## Architecture

- **Evidence Graph** — a normalized model of entities and what's been
  observed about them, regardless of which source the evidence came from.
- **Adapters** — one small piece of code per source, translating that
  source's native shape into the graph's normalized shape.
- **Entity Resolver** — shared infrastructure that maps identities and
  locations (a user, a floor) to the network entities they actually
  correspond to.
- **Deterministic Executors & Capabilities** — plain code that answers
  specific questions (reachability, blast radius, traffic state) without
  any LLM involved.
- **Agents** — Incident Framing, Hypothesis Generator, Evidence Planner,
  Evidence Analyst, Hypothesis Challenger, and Root Cause & Remediation —
  reasoning in a loop that continues until a hypothesis both clears a
  confidence threshold and survives adversarial challenge, or the
  investigation runs out of rounds.

## Status

This is an active, evolving notebook, not a finished product. Read the
build log in [this Medium series](https://medium.com/@icaro_vazquez) —
start with
["Agentic AI for the Enterprise"](https://medium.com/@icaro_vazquez/agentic-ai-for-the-enterprise-building-an-investigation-team-that-argues-with-itself-7f68fb06d4fc)
for the architecture, the failure modes, and what's still unfinished.

## Current sources

- NIKA network telemetry (topology + operational state)
- PagerDuty-shaped vendor incidents
- (in progress) ServiceNow tickets, Microsoft Teams call-quality telemetry

## Known open items

See the "What's Still Unfinished" section of the article above — in short:
the reasoning loop hasn't been stress-tested on genuinely harder,
multi-cause scenarios yet, the evaluation set is one scenario with one
checked outcome, and there's an open question about whether one agent
(Evidence Planner) is being asked to do too much in a single call.
